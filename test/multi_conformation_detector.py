#!/usr/bin/env python3
"""
多构象蛋白质检测工具

功能:
1. 批量加载PDB文件并进行去噪，提取旋转分数和平移分数
2. 计算两两之间旋转分数和平移分数加权组合的余弦距离
3. 如果余弦距离小于阈值，使用PyTorch计算TM-score
4. 筛选出梯度相似但结构不同的多构象蛋白质对

"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from omegaconf import OmegaConf
from Bio import PDB
from tqdm import tqdm

matplotlib.use('Agg')

try:
    from sklearn.metrics.pairwise import cosine_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: sklearn未安装，请运行 pip install scikit-learn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import se3_diffuser
from data import utils as du
from model import score_network
from openfold.utils import rigid_utils as ru
from openfold.data import data_transforms

DEFAULT_SOURCE_DIR = Path("/home/wangzeli/1ake_B")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "multi_conformation_output"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best_weights.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "base.yaml"


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class SampleResult:
    """单个样本的去噪结果"""
    name: str
    pdb_path: Path
    chain_id: str
    seq_len: int
    rot_score: np.ndarray  # Shape: (num_steps, L, 3, 3)
    trans_score: np.ndarray  # Shape: (num_steps, L, 3)
    final_coords: np.ndarray  # Shape: (L, 3)
    residue_mask: np.ndarray  # Shape: (L,)


@dataclass
class PreparedInput:
    """预处理后的输入数据"""
    name: str
    pdb_path: Path
    chain_id: str
    seq_len: int
    aatype: np.ndarray
    atom_positions: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    b_factors: np.ndarray


@dataclass
class MultiConformationPair:
    """多构象蛋白质对"""
    idx1: int
    idx2: int
    name1: str
    name2: str
    cosine_distance: float
    tm_score: float
    rot_distance: float
    trans_distance: float


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多构象蛋白质检测工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 输入输出
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="包含PDB文件的目录 (递归扫描)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS,
                        help="ScoreNetwork权重路径")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG,
                        help="配置文件路径")
    
    # 模型参数
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批处理大小")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="去噪步数")
    parser.add_argument("--min-t", type=float, default=0.01,
                        help="最小时间步")
    parser.add_argument("--max-t", type=float, default=0.05,
                        help="最大时间步")
    parser.add_argument("--noise-scale", type=float, default=0,
                        help="噪声尺度")
    parser.add_argument("--self-condition", action="store_true",
                        help="启用自条件")
    parser.add_argument("--use-fp16", action="store_true",
                        help="使用FP16混合精度")
    
    # 距离和阈值
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="旋转分数权重 (平移权重为1-alpha)")
    parser.add_argument("--cosine-threshold", type=float, default=0.5,
                        help="余弦距离阈值 (小于此值认为梯度相似)")
    parser.add_argument("--tm-threshold", type=float, default=0.7,
                        help="TM-score阈值 (小于此值认为结构不同)")
    
    # 处理参数
    parser.add_argument("--chain-id", type=str, default=None,
                        help="指定链ID (None表示自动选择)")
    parser.add_argument("--num-workers", type=int, default=cpu_count(),
                        help="并行工作进程数")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数 (用于调试)")
    
    # 特征提取
    parser.add_argument("--pooling", type=str, default="flatten",
                        choices=["flatten", "mean", "max"],
                        help="特征池化方式")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")
    
    return parser.parse_args()


# ============================================================================
# PDB预处理函数 (从score_clustering_analyzer.py复用)
# ============================================================================

def resolve_chain_id(pdb_path: Path, preferred_chain: Optional[str]) -> Optional[str]:
    """自动解析链ID"""
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", str(pdb_path))
        chains = list(structure[0].get_chains())
        if not chains:
            return None
        if preferred_chain:
            chain_ids = [c.id for c in chains]
            if preferred_chain in chain_ids:
                return preferred_chain
        return chains[0].id
    except Exception as e:
        logging.warning(f"解析链ID失败 {pdb_path}: {e}")
        return None


def process_chain_feats(pdb_feats):
    """处理链特征"""
    ca_idx = du.residue_constants.atom_order["CA"]
    chain_feats = {
        "aatype": torch.tensor(pdb_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(pdb_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(pdb_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    
    return {
        "aatype": chain_feats["aatype"].numpy(),
        "atom_positions": chain_feats["all_atom_positions"].numpy(),
        "atom_mask": chain_feats["all_atom_mask"].numpy(),
        "residue_index": pdb_feats["residue_index"],
        "b_factors": pdb_feats["b_factors"][:, ca_idx],
    }


def merge_chain_features(chain_feat_map: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """合并多链特征"""
    merged = {k: np.concatenate([v[k] for v in chain_feat_map.values()], axis=0)
              for k in chain_feat_map[list(chain_feat_map.keys())[0]].keys()}
    return merged


def prepare_single_input(pdb_path: Path, chain_id: Optional[str]) -> PreparedInput:
    """预处理单个PDB文件"""
    try:
        actual_chain = resolve_chain_id(pdb_path, chain_id)
        if actual_chain is None:
            raise ValueError(f"无法解析链ID: {pdb_path}")
        
        # 使用parse_pdb_feats函数直接解析PDB文件
        pdb_feats_raw = du.parse_pdb_feats(
            pdb_path.stem,
            str(pdb_path),
            chain_id=actual_chain
        )
        
        # 检查是否为多链字典结构
        if isinstance(pdb_feats_raw, dict) and all(isinstance(v, dict) for v in pdb_feats_raw.values()):
            # 多链结构，需要合并
            chain_feat_map = {}
            for cid, chain_pdb in pdb_feats_raw.items():
                chain_feats = process_chain_feats(chain_pdb)
                chain_feat_map[str(cid)] = chain_feats
            merged_feats = merge_chain_features(chain_feat_map)
        else:
            # 单链结构
            merged_feats = process_chain_feats(pdb_feats_raw)
        
        seq_len = len(merged_feats["aatype"])
        
        return PreparedInput(
            name=pdb_path.stem,
            pdb_path=pdb_path,
            chain_id=actual_chain,
            seq_len=seq_len,
            **merged_feats
        )
    except Exception as e:
        logging.error(f"预处理失败 {pdb_path}: {e}")
        raise


def prepare_inputs_parallel(
    pdb_paths: List[Path],
    chain_id: Optional[str],
    num_workers: int,
) -> List[PreparedInput]:
    """并行预处理"""
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(prepare_single_input, p, chain_id): p for p in pdb_paths}
        
        # 使用进度条显示预处理进度
        with tqdm(total=len(pdb_paths), desc="预处理PDB文件", unit="文件") as pbar:
            for future in as_completed(futures):
                pdb_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    pbar.set_postfix({"当前": pdb_path.name[:30], "长度": result.seq_len})
                except Exception as e:
                    logging.warning(f"✗ 预处理失败: {pdb_path.name} - {e}")
                pbar.update(1)
    
    return results


# ============================================================================
# 批量去噪推理 (从score_clustering_analyzer.py复用)
# ============================================================================

def pad_to_max_length(tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """填充张量到相同长度"""
    max_len = max(t.shape[0] for t in tensors)
    masks = []
    padded = []
    
    for t in tensors:
        L = t.shape[0]
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[L:] = False
        masks.append(mask)
        
        pad_shape = list(t.shape)
        pad_shape[0] = max_len
        padded_t = torch.full(pad_shape, pad_value, dtype=t.dtype)
        padded_t[:L] = t
        padded.append(padded_t)
    
    return torch.stack(padded), torch.stack(masks)


def create_batched_input(
    inputs: List[PreparedInput],
    device: torch.device,
    use_fp16: bool = False
) -> Dict[str, torch.Tensor]:
    """创建批量输入"""
    dtype = torch.float16 if use_fp16 else torch.float32
    
    aatype_list = [torch.tensor(inp.aatype, dtype=torch.long) for inp in inputs]
    pos_list = [torch.tensor(inp.atom_positions, dtype=dtype) for inp in inputs]
    mask_list = [torch.tensor(inp.atom_mask, dtype=dtype) for inp in inputs]
    residue_index_list = [torch.tensor(inp.residue_index, dtype=torch.long) for inp in inputs]
    
    aatype_batch, _ = pad_to_max_length(aatype_list, pad_value=0)
    pos_batch, pos_mask = pad_to_max_length(pos_list, pad_value=0.0)
    mask_batch, _ = pad_to_max_length(mask_list, pad_value=0.0)
    residue_index_batch, _ = pad_to_max_length(residue_index_list, pad_value=0)
    
    return {
        "aatype": aatype_batch.to(device),
        "atom_positions": pos_batch.to(device),
        "atom_mask": mask_batch.to(device),
        "residue_index": residue_index_batch.to(device),
        "seq_mask": pos_mask.to(device),
    }


def batched_direct_denoising(
    model: torch.nn.Module,
    diffuser,
    batched_input: Dict[str, torch.Tensor],
    num_steps: int,
    min_t: float,
    max_t: float,
    noise_scale: float,
    enable_self_conditioning: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """批量去噪"""
    B, L = batched_input["seq_mask"].shape
    
    # 从atom37坐标创建初始rigid frames
    gt_pos = batched_input["atom_positions"]
    ca_idx = 1  # CA在atom37中的索引
    ca_pos = gt_pos[:, :, ca_idx, :]  # (B, L, 3)
    
    # 创建初始rigid frames
    identity_rots = torch.eye(3, device=device)[None, None, :, :].repeat(B, L, 1, 1)
    
    if noise_scale > 0:
        noisy_rots = identity_rots + torch.randn_like(identity_rots) * noise_scale
        noisy_trans = ca_pos + torch.randn_like(ca_pos) * noise_scale
    else:
        noisy_rots = identity_rots
        noisy_trans = ca_pos
    
    # 创建4x4变换矩阵并转为Rigid对象
    rigid_4x4 = torch.eye(4, device=device)[None, None, :, :].repeat(B, L, 1, 1)
    rigid_4x4[:, :, :3, :3] = noisy_rots
    rigid_4x4[:, :, :3, 3] = noisy_trans
    rigids_t = ru.Rigid.from_tensor_4x4(rigid_4x4)
    
    t_schedule = torch.linspace(min_t, max_t, num_steps, device=device)

    rot_score_history = []
    trans_score_history = []
    
    # 时间步占位符
    t_placeholder = torch.ones(B, device=device)
    
    # 准备sample_feats字典
    sample_feats = {
        "res_mask": batched_input["seq_mask"],
        "seq_idx": batched_input["residue_index"],
        "fixed_mask": torch.zeros_like(batched_input["seq_mask"]),
        "torsion_angles_sin_cos": torch.zeros(B, L, 7, 2, device=device),
        "sc_ca_t": torch.zeros(B, L, 3, device=device),
        "rigids_t": rigids_t.to_tensor_7(),
    }
    
    # 设置时间相关特征的辅助函数
    def set_t_feats(feats, t_value):
        feats["t"] = t_placeholder * float(t_value)
        rot_scale, trans_scale = diffuser.score_scaling(float(t_value))
        feats["rot_score_scaling"] = torch.full((B,), float(rot_scale), device=device)
        feats["trans_score_scaling"] = torch.full((B,), float(trans_scale), device=device)
    
    for t_value in t_schedule:
        set_t_feats(sample_feats, t_value.item())
        
        # 模型推理
        with torch.no_grad():
            model_out = model(sample_feats)
        
        # 从模型输出获取score
        rot_score = model_out["rot_score"]  # (B, L, 3, 3)
        trans_score = model_out["trans_score"]  # (B, L, 3)
        
        rot_score_history.append(rot_score.detach().cpu().numpy())
        trans_score_history.append(trans_score.detach().cpu().numpy())
        
        # 更新rigids（如果需要可以使用模型输出更新）
        # 这里暂时保持原rigids不变
    
    # 提取最终坐标（CA位置）
    final_coords = sample_feats["rigids_t"][..., 4:].detach().cpu().numpy()  # (B, L, 3)
    
    return {
        "rot_score_history": np.stack(rot_score_history, axis=1),
        "trans_score_history": np.stack(trans_score_history, axis=1),
        "final_coords": final_coords,
    }


def process_all_samples(
    model: torch.nn.Module,
    diffuser,
    prepared_inputs: List[PreparedInput],
    device: torch.device,
    batch_size: int,
    num_steps: int,
    min_t: float,
    max_t: float,
    noise_scale: float,
    enable_self_conditioning: bool,
    use_fp16: bool,
    output_dir: Path,
) -> List[SampleResult]:
    """处理所有样本"""
    results = []
    
    for i in range(0, len(prepared_inputs), batch_size):
        batch_inputs = prepared_inputs[i:i+batch_size]
        logging.info(f"处理批次 {i//batch_size + 1}/{(len(prepared_inputs)-1)//batch_size + 1}")
        
        batched_input = create_batched_input(batch_inputs, device, use_fp16)
        
        batch_results = batched_direct_denoising(
            model=model,
            diffuser=diffuser,
            batched_input=batched_input,
            num_steps=num_steps,
            min_t=min_t,
            max_t=max_t,
            noise_scale=noise_scale,
            enable_self_conditioning=enable_self_conditioning,
            device=device,
        )
        
        for j, inp in enumerate(batch_inputs):
            L = inp.seq_len
            result = SampleResult(
                name=inp.name,
                pdb_path=inp.pdb_path,
                chain_id=inp.chain_id,
                seq_len=L,
                rot_score=batch_results["rot_score_history"][j, :, :L],
                trans_score=batch_results["trans_score_history"][j, :, :L],
                final_coords=batch_results["final_coords"][j, :L],
                residue_mask=batched_input["seq_mask"][j, :L].cpu().numpy(),
            )
            results.append(result)
            
            # 保存分数
            scores_dir = output_dir / "scores"
            scores_dir.mkdir(parents=True, exist_ok=True)
            np.save(scores_dir / f"{result.name}_rot_score.npy", result.rot_score)
            np.save(scores_dir / f"{result.name}_trans_score.npy", result.trans_score)
    
    return results


# ============================================================================
# 特征提取与距离计算
# ============================================================================

def extract_features(
    samples: List[SampleResult],
    pooling: str = "flatten"
) -> Tuple[np.ndarray, np.ndarray]:
    """提取旋转和平移分数特征"""
    rot_features = []
    trans_features = []
    
    for sample in samples:
        rot_score = sample.rot_score
        trans_score = sample.trans_score
        
        if pooling == "flatten":
            rot_feat = rot_score.reshape(rot_score.shape[0], -1)
            trans_feat = trans_score.reshape(trans_score.shape[0], -1)
        elif pooling == "mean":
            rot_feat = rot_score.mean(axis=1)
            trans_feat = trans_score.mean(axis=1)
        elif pooling == "max":
            rot_feat = rot_score.max(axis=1)
            trans_feat = trans_score.max(axis=1)
        else:
            raise ValueError(f"未知的池化方式: {pooling}")
        
        rot_feat = rot_feat.reshape(-1)
        trans_feat = trans_feat.reshape(-1)
        
        rot_features.append(rot_feat)
        trans_features.append(trans_feat)
    
    return np.array(rot_features), np.array(trans_features)


def compute_pairwise_cosine_distances(
    rot_features: np.ndarray,
    trans_features: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """计算成对余弦距离"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("需要sklearn来计算余弦距离")
    
    rot_dist = cosine_distances(rot_features)
    trans_dist = cosine_distances(trans_features)
    
    combined_dist = alpha * rot_dist + (1 - alpha) * trans_dist
    
    return combined_dist, rot_dist, trans_dist


# ============================================================================
# PyTorch实现的TM-score计算
# ============================================================================

def compute_tm_score_pytorch(
    coords1: np.ndarray,
    coords2: np.ndarray,
    device: torch.device,
) -> float:
    """
    使用PyTorch实现TM-score计算
    
    TM-score公式: TM = (1/L) * sum_i [1 / (1 + (d_i/d0)^2)]
    其中 d0 = 1.24 * (L-15)^(1/3) - 1.8 (对于L > 15)
    d_i 是第i个残基在最佳叠合后的距离
    
    Args:
        coords1: 第一个结构的CA坐标 (L1, 3)
        coords2: 第二个结构的CA坐标 (L2, 3)
        device: 计算设备
    
    Returns:
        TM-score值 (0-1之间)
    """
    # 确保长度相同
    L = min(len(coords1), len(coords2))
    coords1 = coords1[:L]
    coords2 = coords2[:L]
    
    # 转换为PyTorch张量
    c1 = torch.tensor(coords1, dtype=torch.float32, device=device)
    c2 = torch.tensor(coords2, dtype=torch.float32, device=device)
    
    # 计算d0
    if L > 15:
        d0 = 1.24 * ((L - 15) ** (1.0/3.0)) - 1.8
    else:
        d0 = 0.5
    d0 = torch.tensor(d0, dtype=torch.float32, device=device)
    
    # 中心化
    c1_centered = c1 - c1.mean(dim=0, keepdim=True)
    c2_centered = c2 - c2.mean(dim=0, keepdim=True)
    
    # 使用Kabsch算法进行最佳叠合
    # 计算协方差矩阵
    H = c1_centered.T @ c2_centered
    
    # SVD分解
    U, S, Vt = torch.linalg.svd(H)
    
    # 计算旋转矩阵
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], device=device)) @ U.T
    
    # 应用旋转
    c1_rotated = c1_centered @ R.T
    
    # 计算距离
    distances = torch.sqrt(((c1_rotated - c2_centered) ** 2).sum(dim=1))
    
    # 计算TM-score
    tm_score = (1.0 / (1.0 + (distances / d0) ** 2)).mean()
    
    return tm_score.item()


def compute_tm_scores_for_pairs(
    samples: List[SampleResult],
    pairs: List[Tuple[int, int]],
    device: torch.device,
) -> Dict[Tuple[int, int], float]:
    """批量计算TM-score"""
    tm_scores = {}
    
    for idx1, idx2 in pairs:
        coords1 = samples[idx1].final_coords
        coords2 = samples[idx2].final_coords
        
        tm_score = compute_tm_score_pytorch(coords1, coords2, device)
        tm_scores[(idx1, idx2)] = tm_score
    
    return tm_scores


# ============================================================================
# 多构象检测
# ============================================================================

def detect_multi_conformations(
    samples: List[SampleResult],
    rot_features: np.ndarray,
    trans_features: np.ndarray,
    alpha: float,
    cosine_threshold: float,
    tm_threshold: float,
    device: torch.device,
) -> List[MultiConformationPair]:
    """检测多构象蛋白质对"""
    
    logging.info("计算余弦距离...")
    combined_dist, rot_dist, trans_dist = compute_pairwise_cosine_distances(
        rot_features, trans_features, alpha
    )
    
    # 找到余弦距离小于阈值的配对
    N = len(samples)
    candidate_pairs = []
    
    for i in range(N):
        for j in range(i+1, N):
            if combined_dist[i, j] < cosine_threshold:
                candidate_pairs.append((i, j))
    
    logging.info(f"发现 {len(candidate_pairs)} 对候选多构象对 (余弦距离 < {cosine_threshold})")
    
    if not candidate_pairs:
        logging.info("没有发现符合余弦距离阈值的配对")
        return []
    
    # 计算TM-score
    logging.info("计算TM-score...")
    tm_scores = compute_tm_scores_for_pairs(samples, candidate_pairs, device)
    
    # 筛选符合条件的多构象对
    multi_conf_pairs = []
    
    for (idx1, idx2), tm_score in tm_scores.items():
        if tm_score < tm_threshold:
            pair = MultiConformationPair(
                idx1=idx1,
                idx2=idx2,
                name1=samples[idx1].name,
                name2=samples[idx2].name,
                cosine_distance=combined_dist[idx1, idx2],
                tm_score=tm_score,
                rot_distance=rot_dist[idx1, idx2],
                trans_distance=trans_dist[idx1, idx2],
            )
            multi_conf_pairs.append(pair)
            logging.info(f"✓ 发现多构象对: {pair.name1} <-> {pair.name2} "
                        f"(余弦距离={pair.cosine_distance:.4f}, TM-score={pair.tm_score:.4f})")
    
    return multi_conf_pairs


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(
    output_path: Path,
    args: argparse.Namespace,
    samples: List[SampleResult],
    multi_conf_pairs: List[MultiConformationPair],
    timing_info: Dict[str, float],
):
    """生成分析报告"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 多构象蛋白质检测报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 参数设置\n\n")
        f.write(f"- 源目录: `{args.source_dir}`\n")
        f.write(f"- 输出目录: `{args.output_dir}`\n")
        f.write(f"- 样本数量: {len(samples)}\n")
        f.write(f"- 去噪步数: {args.num_steps}\n")
        f.write(f"- 批处理大小: {args.batch_size}\n")
        f.write(f"- 旋转权重α: {args.alpha}\n")
        f.write(f"- 余弦距离阈值: {args.cosine_threshold}\n")
        f.write(f"- TM-score阈值: {args.tm_threshold}\n\n")
        
        f.write("## 检测结果\n\n")
        f.write(f"发现 **{len(multi_conf_pairs)}** 对多构象蛋白质\n\n")
        
        if multi_conf_pairs:
            f.write("### 多构象配对详情\n\n")
            f.write("| 序号 | 蛋白质1 | 蛋白质2 | 余弦距离 | TM-score | 旋转距离 | 平移距离 |\n")
            f.write("|------|---------|---------|----------|----------|----------|----------|\n")
            
            for i, pair in enumerate(multi_conf_pairs, 1):
                f.write(f"| {i} | {pair.name1} | {pair.name2} | "
                       f"{pair.cosine_distance:.4f} | {pair.tm_score:.4f} | "
                       f"{pair.rot_distance:.4f} | {pair.trans_distance:.4f} |\n")
            
            f.write("\n### 统计信息\n\n")
            cosine_dists = [p.cosine_distance for p in multi_conf_pairs]
            tm_scores = [p.tm_score for p in multi_conf_pairs]
            
            f.write(f"- 平均余弦距离: {np.mean(cosine_dists):.4f} ± {np.std(cosine_dists):.4f}\n")
            f.write(f"- 平均TM-score: {np.mean(tm_scores):.4f} ± {np.std(tm_scores):.4f}\n")
            f.write(f"- 余弦距离范围: [{np.min(cosine_dists):.4f}, {np.max(cosine_dists):.4f}]\n")
            f.write(f"- TM-score范围: [{np.min(tm_scores):.4f}, {np.max(tm_scores):.4f}]\n")
        
        f.write("\n## 性能统计\n\n")
        total_time = sum(timing_info.values())
        f.write(f"- 总耗时: {total_time:.2f}秒\n")
        for key, value in timing_info.items():
            f.write(f"- {key}: {value:.2f}秒 ({value/total_time*100:.1f}%)\n")
    
    logging.info(f"报告已保存到: {output_path}")


def save_results(
    output_dir: Path,
    samples: List[SampleResult],
    multi_conf_pairs: List[MultiConformationPair],
):
    """保存结果到文件"""
    # 保存多构象配对
    pairs_file = output_dir / "multi_conformation_pairs.json"
    pairs_data = [
        {
            "idx1": p.idx1,
            "idx2": p.idx2,
            "name1": p.name1,
            "name2": p.name2,
            "cosine_distance": float(p.cosine_distance),
            "tm_score": float(p.tm_score),
            "rot_distance": float(p.rot_distance),
            "trans_distance": float(p.trans_distance),
        }
        for p in multi_conf_pairs
    ]
    
    with open(pairs_file, "w", encoding="utf-8") as f:
        json.dump(pairs_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"多构象配对已保存到: {pairs_file}")


def plot_results(
    output_dir: Path,
    multi_conf_pairs: List[MultiConformationPair],
):
    """生成可视化图表"""
    if not multi_conf_pairs:
        logging.info("没有多构象配对，跳过可视化")
        return
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 散点图: 余弦距离 vs TM-score
    plt.figure(figsize=(10, 6))
    cosine_dists = [p.cosine_distance for p in multi_conf_pairs]
    tm_scores = [p.tm_score for p in multi_conf_pairs]
    
    plt.scatter(cosine_dists, tm_scores, alpha=0.6, s=100)
    plt.xlabel("Cosine Distance", fontsize=12)
    plt.ylabel("TM-score", fontsize=12)
    plt.title("Multi-conformation: Gradient Similarity vs Structural Similarity", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加阈值线
    plt.axhline(y=0.7, color='r', linestyle='--', label='TM-score threshold')
    plt.axvline(x=0.5, color='b', linestyle='--', label='Cosine distance threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_vs_tmscore.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 直方图: 余弦距离分布
    plt.figure(figsize=(10, 6))
    plt.hist(cosine_dists, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Cosine Distance", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Cosine Distance Distribution", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_distance_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 直方图: TM-score分布
    plt.figure(figsize=(10, 6))
    plt.hist(tm_scores, bins=20, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel("TM-score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("TM-score Distribution", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "tmscore_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"可视化图表已保存到: {plots_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录开始时间
    timing_info = {}
    
    # 1. 扫描PDB文件
    logging.info("=" * 80)
    logging.info("步骤1: 扫描PDB文件")
    logging.info("=" * 80)
    
    t0 = time.time()
    pdb_files = sorted(args.source_dir.rglob("*.pdb"))
    
    if args.max_samples:
        pdb_files = pdb_files[:args.max_samples]
    
    logging.info(f"发现 {len(pdb_files)} 个PDB文件")
    timing_info["扫描文件"] = time.time() - t0
    
    if not pdb_files:
        logging.error("未找到PDB文件!")
        return
    
    # 2. 预处理
    logging.info("=" * 80)
    logging.info("步骤2: 预处理PDB文件")
    logging.info("=" * 80)
    
    t0 = time.time()
    prepared_inputs = prepare_inputs_parallel(
        pdb_files,
        args.chain_id,
        args.num_workers,
    )
    timing_info["预处理"] = time.time() - t0
    
    if not prepared_inputs:
        logging.error("预处理失败，没有有效样本!")
        return
    
    logging.info(f"成功预处理 {len(prepared_inputs)}/{len(pdb_files)} 个样本")
    
    # 3. 加载模型
    logging.info("=" * 80)
    logging.info("步骤3: 加载模型")
    logging.info("=" * 80)
    
    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 加载配置
    config = OmegaConf.load(args.config_path)
    
    # 创建扩散器
    diffuser = se3_diffuser.SE3Diffuser(config.diffuser)
    
    # 创建模型
    model = score_network.ScoreNetwork(config.model, diffuser).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.weights_path, map_location=device)
    
    # 处理DataParallel保存的权重（移除"module."前缀）
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith('module.'):
        # 移除"module."前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # 移除"module."前缀（7个字符）
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    timing_info["加载模型"] = time.time() - t0
    logging.info("模型加载完成")
    
    # 4. 批量去噪
    logging.info("=" * 80)
    logging.info("步骤4: 批量去噪推理")
    logging.info("=" * 80)
    
    t0 = time.time()
    with torch.no_grad():
        samples = process_all_samples(
            model=model,
            diffuser=diffuser,
            prepared_inputs=prepared_inputs,
            device=device,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            min_t=args.min_t,
            max_t=args.max_t,
            noise_scale=args.noise_scale,
            enable_self_conditioning=args.self_condition,
            use_fp16=args.use_fp16,
            output_dir=args.output_dir,
        )
    timing_info["去噪推理"] = time.time() - t0
    
    logging.info(f"完成 {len(samples)} 个样本的去噪")
    
    # 5. 提取特征
    logging.info("=" * 80)
    logging.info("步骤5: 提取特征")
    logging.info("=" * 80)
    
    t0 = time.time()
    rot_features, trans_features = extract_features(samples, args.pooling)
    timing_info["特征提取"] = time.time() - t0
    
    logging.info(f"旋转特征形状: {rot_features.shape}")
    logging.info(f"平移特征形状: {trans_features.shape}")
    
    # 6. 检测多构象
    logging.info("=" * 80)
    logging.info("步骤6: 检测多构象蛋白质")
    logging.info("=" * 80)
    
    t0 = time.time()
    multi_conf_pairs = detect_multi_conformations(
        samples=samples,
        rot_features=rot_features,
        trans_features=trans_features,
        alpha=args.alpha,
        cosine_threshold=args.cosine_threshold,
        tm_threshold=args.tm_threshold,
        device=device,
    )
    timing_info["多构象检测"] = time.time() - t0
    
    # 7. 保存结果
    logging.info("=" * 80)
    logging.info("步骤7: 保存结果")
    logging.info("=" * 80)
    
    t0 = time.time()
    
    # 生成报告
    report_path = args.output_dir / "detection_report.md"
    generate_report(report_path, args, samples, multi_conf_pairs, timing_info)
    
    # 保存配对
    save_results(args.output_dir, samples, multi_conf_pairs)
    
    # 生成可视化
    plot_results(args.output_dir, multi_conf_pairs)
    
    timing_info["保存结果"] = time.time() - t0
    
    # 总结
    logging.info("=" * 80)
    logging.info("分析完成!")
    logging.info("=" * 80)
    logging.info(f"总样本数: {len(samples)}")
    logging.info(f"多构象配对数: {len(multi_conf_pairs)}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"总耗时: {sum(timing_info.values()):.2f}秒")


if __name__ == "__main__":
    main()
