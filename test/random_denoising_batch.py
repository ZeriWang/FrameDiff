#!/usr/bin/env python3
"""随机抽样PDB并执行直接去噪、距离分析与TM-score关联分析的工具。

步骤概览:
1. 在指定目录递归检索PDB文件并随机抽样。
2. 复用 direct_denoising_predictor 的去噪逻辑，只保存最后一步旋转/平移score。
3. 并行计算所有样本对之间的 TM-score、score 欧氏距离与余弦距离。
4. 输出 Markdown 报告，汇总相关性分析结果。
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import re
import subprocess
import threading
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import importlib.util
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from Bio import PDB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import se3_diffuser
from data import utils as du
from model import score_network
from openfold.utils import rigid_utils as ru

try:  # Optional SciPy support for correlations
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover - fallback path
    pearsonr = spearmanr = None

DEFAULT_SOURCE_DIR = Path("/home/wangzeli/1ake_B")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "random_batch_output"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best_weights.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "base.yaml"
DEFAULT_TMALIGN = PROJECT_ROOT / "test" / "TMalign" / "TM-align"


def _load_helper_module(name: str, relative_path: Path):
    """Load a helper module (direct scripts) without adding to sys.path."""
    spec = importlib.util.spec_from_file_location(name, relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {name} 自 {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


direct_module = _load_helper_module(
    "direct_denoising_predictor_module",
    PROJECT_ROOT / "test" / "direct_denoising_predictor.py",
)


@dataclass
class SampleResult:
    """Container for a single denoised sample."""
    name: str
    pdb_path: Path
    chain_id: str
    num_res: int
    rot_score: np.ndarray
    trans_score: np.ndarray
    rot_score_path: Path
    trans_score_path: Path


@dataclass
class PairMetrics:
    """Pairwise metrics between two samples."""
    pair: Tuple[str, str]
    tm_score: float
    rot_euclidean: float
    rot_cosine_distance: float
    trans_euclidean: float
    trans_cosine_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机PDB直接去噪 + TM-score/score相关性分析",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="包含原始PDB的目录 (递归扫描)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录 (score与报告)",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="ScoreNetwork 权重路径",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help="配置文件 (diffuser + model)",
    )
    parser.add_argument(
        "--tm-align-bin",
        type=Path,
        default=DEFAULT_TMALIGN,
        help="TM-align 可执行文件路径",
    )
    parser.add_argument(
        "--chain-id",
        type=str,
        default=None,
        help="指定需要处理的链ID；缺省时对整个结构进行处理",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="随机抽样的PDB数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="随机种子",
    )
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=5,
        help="直接去噪步数",
    )
    parser.add_argument(
        "--min-t",
        type=float,
        default=0.01,
        help="最小时间步",
    )
    parser.add_argument(
        "--max-t",
        type=float,
        default=0.05,
        help="最大时间步",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.0,
        help="噪声缩放因子",
    )
    parser.add_argument(
        "--disable-self-conditioning",
        action="store_true",
        help="关闭自条件",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(8, (os.cpu_count() or 4)),
        help="并行计算pair metrics的最大线程数",
    )
    parser.add_argument(
        "--denoise-workers",
        type=int,
        default=0,
        help="同时在GPU上运行的去噪任务数 (0 表示自动与可用GPU数量一致)",
    )
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="逗号分隔的GPU编号列表 (默认使用所有可用GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="强制指定计算设备 (默认自动选择)",
    )
    return parser.parse_args()


def resolve_chain_id(pdb_path: Path, preferred_chain: Optional[str]) -> Optional[str]:
    """Return chosen chain or None to indicate whole-structure processing."""
    if preferred_chain is None:
        return None
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("chain_scan", str(pdb_path))
    available = sorted({chain.id for chain in structure.get_chains()})
    if not available:
        raise ValueError(f"{pdb_path} 不包含任何链")
    if preferred_chain not in available:
        raise ValueError(
            f"{pdb_path} 中不存在链 {preferred_chain}；可选链包括: {', '.join(available)}"
        )
    return preferred_chain


def merge_chain_features(chain_feat_map: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate per-chain features along residue dimension."""
    merged: Dict[str, List[np.ndarray]] = {}
    for chain_id in sorted(chain_feat_map.keys()):
        feats = chain_feat_map[chain_id]
        for key, value in feats.items():
            arr = np.asarray(value)
            merged.setdefault(key, []).append(arr)
    return {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}


def sample_pdb_files(source_dir: Path, sample_size: int, seed: int) -> List[Path]:
    all_pdbs = sorted(source_dir.rglob("*.pdb"))
    if not all_pdbs:
        raise FileNotFoundError(f"在 {source_dir} 未找到任何 .pdb 文件")
    actual_size = min(sample_size, len(all_pdbs))
    rng = random.Random(seed)
    return rng.sample(all_pdbs, actual_size)


def load_model_blueprint(config_path: Path, weights_path: Path):
    conf = OmegaConf.load(config_path)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    conf_container = OmegaConf.to_container(conf, resolve=True)
    return conf_container, state_dict


def build_model_from_blueprint(conf_container, state_dict, device: torch.device):
    conf = OmegaConf.create(copy.deepcopy(conf_container))
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    model = score_network.ScoreNetwork(conf.model, diffuser)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, diffuser


def prepare_inputs(pdb_path: Path, chain_id: Optional[str]):
    pdb_name = pdb_path.stem
    effective_chain = resolve_chain_id(pdb_path, chain_id)
    pdb_feats_raw = du.parse_pdb_feats(pdb_name, str(pdb_path), chain_id=effective_chain or None)
    if isinstance(pdb_feats_raw, dict):
        pdb_feats = merge_chain_features(pdb_feats_raw)
        chain_label = "ALL"
    else:
        pdb_feats = pdb_feats_raw
        chain_label = effective_chain or "ALL"
    chain_feats = direct_module.process_chain_feats(pdb_feats)
    bb_mask = np.asarray(pdb_feats['bb_mask']).astype(bool)
    num_res = int(bb_mask.sum())
    if num_res == 0:
        raise ValueError(f"{pdb_path} 链 {chain_id} 不包含有效主链残基")

    mask_tensor = torch.from_numpy(bb_mask).to(torch.bool)
    rigid_frames = chain_feats['rigidgroups_gt_frames'][mask_tensor, 0].detach().cpu().float()
    rigids_0 = ru.Rigid.from_tensor_4x4(rigid_frames)
    sc_ca_init = rigids_0.get_trans().detach().cpu().numpy().astype(np.float32)

    torsion_angles = chain_feats['torsion_angles_sin_cos'].detach().cpu().numpy()[bb_mask]
    if torsion_angles.dtype == np.object_:
        torsion_angles = np.stack([x.astype(np.float32) for x in torsion_angles], axis=0)
    else:
        torsion_angles = torsion_angles.astype(np.float32)

    res_mask_tensor = torch.ones(num_res, dtype=torch.float32)
    seq_idx_tensor = torch.arange(1, num_res + 1, dtype=torch.float32)
    fixed_mask_tensor = torch.zeros(num_res, dtype=torch.float32)
    torsion_tensor = torch.tensor(torsion_angles, dtype=torch.float32)
    sc_ca_tensor = torch.tensor(sc_ca_init, dtype=torch.float32)

    return {
        'rigids_0': rigids_0,
        'res_mask': res_mask_tensor,
        'seq_idx': seq_idx_tensor,
        'fixed_mask': fixed_mask_tensor,
        'torsion_angles': torsion_tensor,
        'sc_ca': sc_ca_tensor,
        'num_res': num_res,
        'chain_id': chain_label,
    }


def run_denoising_for_sample(
    model,
    diffuser,
    device: torch.device,
    pdb_path: Path,
    chain_id: Optional[str],
    output_dir: Path,
    num_steps: int,
    min_t: float,
    max_t: float,
    noise_scale: float,
    enable_self_conditioning: bool,
) -> SampleResult:
    prep = prepare_inputs(pdb_path, chain_id)
    effective_chain = prep['chain_id']

    denoising_result = direct_module.direct_denoising(
        model=model,
        diffuser=diffuser,
        original_rigids=prep['rigids_0'],
        res_mask=prep['res_mask'],
        seq_idx=prep['seq_idx'],
        fixed_mask=prep['fixed_mask'],
        torsion_angles=prep['torsion_angles'],
        sc_ca=prep['sc_ca'],
        num_steps=num_steps,
        min_t=min_t,
        max_t=max_t,
        device=str(device),
        noise_scale=noise_scale,
        enable_self_conditioning=enable_self_conditioning,
    )

    rot_score = denoising_result['final_rot_score']
    trans_score = denoising_result['final_trans_score']
    if rot_score is None or trans_score is None:
        raise RuntimeError("去噪未返回最终score，无法继续")

    prefix = f"{pdb_path.stem}_{effective_chain}"
    rot_path = output_dir / f"{prefix}_rot_score.npy"
    trans_path = output_dir / f"{prefix}_trans_score.npy"
    np.save(rot_path, rot_score)
    np.save(trans_path, trans_score)

    return SampleResult(
        name=prefix,
        pdb_path=pdb_path,
        chain_id=effective_chain,
        num_res=prep['num_res'],
        rot_score=rot_score,
        trans_score=trans_score,
        rot_score_path=rot_path,
        trans_score_path=trans_path,
    )


def compute_tm_score(tm_align_bin: Path, pdb_a: Path, pdb_b: Path) -> float:
    if not tm_align_bin.exists():
        raise FileNotFoundError(f"TM-align未找到: {tm_align_bin}")
    cmd = [str(tm_align_bin), str(pdb_a), str(pdb_b)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    match = re.search(r"TM-score\s*=\s*([0-9.]+)", result.stdout)
    if not match:
        raise RuntimeError(f"TM-align 输出无法解析:\n{result.stdout}")
    return float(match.group(1))


def align_scores(score_a: np.ndarray, score_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure score arrays share the same shape by trimming to the shortest length."""
    if score_a.ndim == 3:
        score_a = score_a.squeeze(0)
    if score_b.ndim == 3:
        score_b = score_b.squeeze(0)
    if score_a.shape[0] != score_b.shape[0]:
        min_len = min(score_a.shape[0], score_b.shape[0])
        score_a = score_a[:min_len]
        score_b = score_b[:min_len]
    return score_a, score_b


def align_and_tensorize(score_a: np.ndarray, score_b: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    a_np, b_np = align_scores(score_a, score_b)
    a_tensor = torch.as_tensor(a_np, dtype=torch.float32, device=device)
    b_tensor = torch.as_tensor(b_np, dtype=torch.float32, device=device)
    return a_tensor, b_tensor


def compute_pair_metrics(
    tm_align_bin: Path,
    sample_a: SampleResult,
    sample_b: SampleResult,
    device: torch.device,
) -> PairMetrics:
    tm_score = compute_tm_score(tm_align_bin, sample_a.pdb_path, sample_b.pdb_path)

    rot_a, rot_b = align_and_tensorize(sample_a.rot_score, sample_b.rot_score, device)
    trans_a, trans_b = align_and_tensorize(sample_a.trans_score, sample_b.trans_score, device)

    rot_diff = rot_a - rot_b
    trans_diff = trans_a - trans_b

    rot_euc = torch.linalg.vector_norm(rot_diff).item()
    trans_euc = torch.linalg.vector_norm(trans_diff).item()

    def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        a_flat = a.reshape(1, -1)
        b_flat = b.reshape(1, -1)
        a_norm = torch.linalg.vector_norm(a_flat)
        b_norm = torch.linalg.vector_norm(b_flat)
        if a_norm.item() == 0.0 or b_norm.item() == 0.0:
            return float('nan')
        cos_sim = F.cosine_similarity(a_flat, b_flat).item()
        return 1.0 - cos_sim

    rot_cos = cosine_distance(rot_a, rot_b)
    trans_cos = cosine_distance(trans_a, trans_b)

    return PairMetrics(
        pair=(sample_a.name, sample_b.name),
        tm_score=tm_score,
        rot_euclidean=rot_euc,
        rot_cosine_distance=rot_cos,
        trans_euclidean=trans_euc,
        trans_cosine_distance=trans_cos,
    )


def analyze_correlations(pairs: Sequence[PairMetrics]) -> Dict[str, Dict[str, float]]:
    metrics = {
        'rot_euclidean': [p.rot_euclidean for p in pairs],
        'rot_cosine_distance': [p.rot_cosine_distance for p in pairs],
        'trans_euclidean': [p.trans_euclidean for p in pairs],
        'trans_cosine_distance': [p.trans_cosine_distance for p in pairs],
    }
    tm_scores = np.asarray([p.tm_score for p in pairs], dtype=float)
    results: Dict[str, Dict[str, float]] = {}
    for name, values in metrics.items():
        values_np = np.asarray(values, dtype=float)
        mask = np.isfinite(values_np) & np.isfinite(tm_scores)
        if mask.sum() < 2:
            continue
        tm_masked = tm_scores[mask]
        val_masked = values_np[mask]
        if pearsonr is not None:
            pear = float(pearsonr(tm_masked, val_masked)[0])
        else:  # 简易Pearson
            pear = float(np.corrcoef(tm_masked, val_masked)[0, 1])
        if spearmanr is not None:
            spear = float(spearmanr(tm_masked, val_masked)[0])
        else:
            spear = float('nan')
        results[name] = {
            'pearson': pear,
            'spearman': spear,
            'count': int(mask.sum()),
        }
    return results


def plot_tm_vs_distances(pair_metrics: Sequence[PairMetrics], plot_dir: Path) -> List[Path]:
    artifacts: List[Path] = []
    if not pair_metrics:
        return artifacts
    metric_map = {
        'rot_euclidean': 'Rotational Euclidean Distance',
        'rot_cosine_distance': 'Rotational Cosine Distance',
        'trans_euclidean': 'Translational Euclidean Distance',
        'trans_cosine_distance': 'Translational Cosine Distance',
    }
    tm_scores = np.array([pm.tm_score for pm in pair_metrics], dtype=float)
    for attr, label in metric_map.items():
        values = np.array([getattr(pm, attr) for pm in pair_metrics], dtype=float)
        mask = np.isfinite(tm_scores) & np.isfinite(values)
        if mask.sum() < 2:
            continue
        x = tm_scores[mask]
        y = values[mask]
        plt.figure(figsize=(6, 4))
        if sns is not None:
            sns.scatterplot(x=x, y=y)
            sns.regplot(x=x, y=y, scatter=False, color='orange', lowess=True)
        else:
            plt.scatter(x, y, alpha=0.7)
        plt.title(f'TM-score vs {label}')
        plt.xlabel('TM-score')
        plt.ylabel(label)
        plt.grid(True, linestyle='--', alpha=0.3)
        out_path = plot_dir / f'tm_vs_{attr}.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        artifacts.append(out_path)
    return artifacts


def build_metric_matrix(
    samples: Sequence[SampleResult],
    pair_metrics: Sequence[PairMetrics],
    attr: str,
) -> Tuple[List[str], np.ndarray]:
    names = [s.name for s in samples]
    n = len(names)
    matrix = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for pm in pair_metrics:
        if pm.pair[0] not in name_to_idx or pm.pair[1] not in name_to_idx:
            continue
        i = name_to_idx[pm.pair[0]]
        j = name_to_idx[pm.pair[1]]
        value = getattr(pm, attr, np.nan)
        matrix[i, j] = value
        matrix[j, i] = value
    return names, matrix


def plot_metric_heatmap(names: List[str], matrix: np.ndarray, title: str, out_path: Path, cmap: str = 'viridis'):
    plt.figure(figsize=(max(6, len(names) * 0.4), max(5, len(names) * 0.4)))
    mask = np.isnan(matrix)
    display_matrix = np.copy(matrix)
    display_matrix[mask] = 0.0
    if sns is not None:
        sns.heatmap(
            display_matrix,
            mask=mask,
            xticklabels=names,
            yticklabels=names,
            cmap=cmap,
            square=True,
            cbar_kws={'shrink': 0.8},
        )
    else:
        plt.imshow(display_matrix, cmap=cmap)
        plt.colorbar(shrink=0.8)
        plt.xticks(range(len(names)), names, rotation=90)
        plt.yticks(range(len(names)), names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_visualizations(
    samples: Sequence[SampleResult],
    pair_metrics: Sequence[PairMetrics],
    output_dir: Path,
) -> List[Path]:
    if not pair_metrics:
        return []
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    artifacts: List[Path] = []
    artifacts.extend(plot_tm_vs_distances(pair_metrics, plot_dir))

    heatmap_specs = [
        ('tm_score', 'TM-score Matrix', 'Blues'),
        ('rot_euclidean', 'Rotational Euclidean Distance Matrix', 'magma'),
        ('trans_euclidean', 'Translational Euclidean Distance Matrix', 'magma'),
    ]
    for attr, title, cmap in heatmap_specs:
        names, matrix = build_metric_matrix(samples, pair_metrics, attr)
        if np.all(np.isnan(matrix)):
            continue
        if attr == 'tm_score':
            np.fill_diagonal(matrix, 1.0)
        out_path = plot_dir / f'heatmap_{attr}.png'
        plot_metric_heatmap(names, matrix, title, out_path, cmap)
        artifacts.append(out_path)
    return artifacts


def render_report(
    output_path: Path,
    args: argparse.Namespace,
    samples: Sequence[SampleResult],
    pair_metrics: Sequence[PairMetrics],
    correlations: Dict[str, Dict[str, float]],
):
    lines: List[str] = []
    lines.append("# 随机直接去噪与相关性分析报告")
    lines.append("")
    lines.append("## 运行配置")
    lines.append(f"- 时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 源目录: {args.source_dir}")
    lines.append(f"- 请求样本数: {args.sample_size}")
    lines.append(f"- 实际样本数: {len(samples)}")
    lines.append(f"- 链 ID: {args.chain_id}")
    lines.append(f"- 去噪步数: {args.num_denoising_steps}")
    lines.append(f"- 时间范围: {args.max_t} -> {args.min_t}")
    lines.append(f"- 噪声缩放: {args.noise_scale}")
    lines.append(f"- 自条件: {not args.disable_self_conditioning}")
    lines.append("")

    lines.append("## 样本列表")
    lines.append("| # | 样本 | 残基数 | rot_score | trans_score |")
    lines.append("| --- | --- | --- | --- | --- |")
    for idx, sample in enumerate(samples, start=1):
        lines.append(
            f"| {idx} | {sample.name} | {sample.num_res} | "
            f"{sample.rot_score_path.name} | {sample.trans_score_path.name} |"
        )
    lines.append("")

    lines.append("## Pairwise TM-score 与 Score 距离")
    lines.append("| Pair | TM-score | Rot Euc | Rot CosDist | Trans Euc | Trans CosDist |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for pm in pair_metrics:
        lines.append(
            f"| {pm.pair[0]} vs {pm.pair[1]} | {pm.tm_score:.4f} | "
            f"{pm.rot_euclidean:.4f} | {pm.rot_cosine_distance:.4f} | "
            f"{pm.trans_euclidean:.4f} | {pm.trans_cosine_distance:.4f} |"
        )
    lines.append("")

    lines.append("## 相关性分析")
    if not correlations:
        lines.append("- 数据不足，无法计算相关性。")
    else:
        lines.append("| 指标 | Pearson | Spearman | 样本对数量 |")
        lines.append("| --- | --- | --- | --- |")
        for metric, stats in correlations.items():
            pear = stats['pearson']
            spear = stats['spearman']
            count = stats['count']
            lines.append(
                f"| {metric} | {pear:.4f} | {spear if math.isnan(spear) else f'{spear:.4f}'} | {count} |"
            )
        if spearmanr is None:
            lines.append("")
            lines.append("> ⚠️ 未安装 SciPy，Spearman 相关系数不可用 (显示 NaN)。")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = args.output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    if args.device and args.device not in {"cpu", "cuda"}:
        raise ValueError("device 只能是 'cpu' 或 'cuda'")
    if args.device == "cpu":
        raise RuntimeError("当前版本仅支持 CUDA 运行，请在GPU环境下执行或省略 --device")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("当前版本要求 CUDA 设备，但未检测到可用GPU")

    if args.device_ids:
        requested_ids = [int(x.strip()) for x in args.device_ids.split(',') if x.strip()]
    else:
        requested_ids = list(range(gpu_count))

    invalid_ids = [gid for gid in requested_ids if gid < 0 or gid >= gpu_count]
    if invalid_ids:
        raise ValueError(f"无效的GPU编号: {invalid_ids}，可用范围为 0~{gpu_count-1}")

    denoise_devices = [torch.device(f'cuda:{gid}') for gid in requested_ids]
    if not denoise_devices:
        raise RuntimeError("未选择任何GPU用于去噪任务")

    requested_workers = args.denoise_workers if args.denoise_workers > 0 else len(denoise_devices)
    worker_count = min(requested_workers, len(denoise_devices))
    worker_devices = denoise_devices[:worker_count]
    primary_device = worker_devices[0]

    print(f"使用GPU: {[str(dev) for dev in worker_devices]}")

    enable_self_conditioning = not args.disable_self_conditioning

    print("加载模型配置与权重...")
    conf_container, state_dict = load_model_blueprint(args.config_path, args.weights_path)
    model_cache: Dict[str, Tuple[score_network.ScoreNetwork, se3_diffuser.SE3Diffuser]] = {}
    cache_lock = threading.Lock()

    def get_or_create_model(device_obj: torch.device):
        key = str(device_obj)
        with cache_lock:
            if key not in model_cache:
                model_obj, diffuser_obj = build_model_from_blueprint(conf_container, state_dict, device_obj)
                model_cache[key] = (model_obj, diffuser_obj)
        return model_cache[key]

    pdb_paths = sample_pdb_files(args.source_dir, args.sample_size, args.seed)
    print(f"抽样到 {len(pdb_paths)} 个PDB文件")

    samples: List[SampleResult] = []

    device_task_map: Dict[torch.device, List[Tuple[int, Path]]] = {dev: [] for dev in worker_devices}
    device_cycle = itertools.cycle(worker_devices)
    for idx, path in enumerate(pdb_paths, start=1):
        assigned_device = next(device_cycle)
        device_task_map[assigned_device].append((idx, path))

    def process_device_queue(device_obj: torch.device, assignments: List[Tuple[int, Path]]):
        results: List[SampleResult] = []
        if not assignments:
            return results
        model_obj, diffuser_obj = get_or_create_model(device_obj)
        device_label = device_obj.index if device_obj.index is not None else 0
        total = len(pdb_paths)
        for idx, path in assignments:
            print(f"[GPU {device_label}] [{idx}/{total}] 处理 {path}")
            result = run_denoising_for_sample(
                model=model_obj,
                diffuser=diffuser_obj,
                device=device_obj,
                pdb_path=path,
                chain_id=args.chain_id,
                output_dir=scores_dir,
                num_steps=args.num_denoising_steps,
                min_t=args.min_t,
                max_t=args.max_t,
                noise_scale=args.noise_scale,
                enable_self_conditioning=enable_self_conditioning,
            )
            results.append(result)
        return results

    active_devices = [dev for dev, tasks in device_task_map.items() if tasks]
    if not active_devices:
        raise RuntimeError("没有分配任何去噪任务，请检查 sample-size 设置")
    if len(active_devices) == 1:
        device_obj = active_devices[0]
        samples.extend(process_device_queue(device_obj, device_task_map[device_obj]))
    else:
        with ThreadPoolExecutor(max_workers=len(active_devices)) as executor:
            futures = {
                executor.submit(process_device_queue, dev, device_task_map[dev]): dev
                for dev in active_devices
            }
            for future in as_completed(futures):
                samples.extend(future.result())

    samples.sort(key=lambda s: s.name)

    if len(samples) < 2:
        raise RuntimeError("样本数不足，无法执行pair分析")

    print("并行计算pair metrics...")
    pairs = list(itertools.combinations(samples, 2))
    pair_metrics: List[PairMetrics] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(
                compute_pair_metrics,
                args.tm_align_bin,
                a,
                b,
                primary_device,
            ): (a.name, b.name)
            for (a, b) in pairs
        }
        for future in as_completed(future_map):
            pair_name = future_map[future]
            try:
                pair_metrics.append(future.result())
            except Exception as exc:  # pragma: no cover - reporting path
                raise RuntimeError(f"计算 pair {pair_name} 失败: {exc}") from exc

    correlations = analyze_correlations(pair_metrics)

    report_path = args.output_dir / "random_denoising_report.md"
    render_report(report_path, args, samples, pair_metrics, correlations)
    generated_plots = generate_visualizations(samples, pair_metrics, args.output_dir)
    print(f"报告已生成: {report_path}")
    if generated_plots:
        print("生成的可视化图像:")
        for path in generated_plots:
            print(f"  - {path}")


if __name__ == "__main__":

    main()
