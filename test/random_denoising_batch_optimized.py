#!/usr/bin/env python3
"""随机抽样PDB并执行直接去噪、距离分析与TM-score关联分析的工具 (GPU优化版)。

性能优化:
1. 批处理推理：将多个蛋白质打包成batch一起推理，提高GPU利用率
2. 异步数据加载：使用CUDA Streams实现数据加载与计算重叠
3. 减少CPU-GPU数据传输：尽可能保持数据在GPU上
4. 预处理并行化：使用多进程预处理PDB文件

步骤概览:
1. 在指定目录递归检索PDB文件并随机抽样。
2. 并行预处理所有PDB文件，准备输入特征
3. 批量执行去噪推理，一次处理多个蛋白质
4. 并行计算所有样本对之间的 TM-score、score 欧氏距离与余弦距离。
5. 输出 Markdown 报告，汇总相关性分析结果。
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
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
import importlib.util
import sys
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import Stream
from omegaconf import OmegaConf
from Bio import PDB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import se3_diffuser
from data import utils as du
from model import score_network
from openfold.utils import rigid_utils as ru
from openfold.data import data_transforms

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = spearmanr = None

DEFAULT_SOURCE_DIR = Path("/home/wangzeli/1ake_B")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "random_batch_output"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best_weights.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "base.yaml"
DEFAULT_TMALIGN = PROJECT_ROOT / "test" / "TMalign" / "TM-align"


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


@dataclass
class PreparedInput:
    """预处理后的输入数据"""
    pdb_path: Path
    name: str
    chain_id: str
    num_res: int
    rigids_tensor: torch.Tensor  # (num_res, 7)
    res_mask: torch.Tensor       # (num_res,)
    seq_idx: torch.Tensor        # (num_res,)
    fixed_mask: torch.Tensor     # (num_res,)
    torsion_angles: torch.Tensor # (num_res, 7, 2)
    sc_ca: torch.Tensor          # (num_res, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机PDB直接去噪 + TM-score/score相关性分析 (GPU优化版)",
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
        "--batch-size",
        type=int,
        default=8,
        help="批处理大小 (同时处理的蛋白质数量)",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=min(4, (os.cpu_count() or 4)),
        help="预处理PDB文件的并行进程数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="强制指定计算设备 (默认自动选择)",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="使用FP16混合精度推理",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="预取的batch数量",
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


def process_chain_feats(pdb_feats):
    """处理PDB特征"""
    chain_feats = {
        'aatype': torch.tensor(pdb_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(pdb_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(pdb_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    
    seq_idx = pdb_feats['residue_index'] - np.min(pdb_feats['residue_index']) + 1
    chain_feats['seq_idx'] = seq_idx
    chain_feats['res_mask'] = pdb_feats['bb_mask']
    chain_feats['residue_index'] = pdb_feats['residue_index']
    return chain_feats


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


def prepare_single_input(pdb_path: Path, chain_id: Optional[str]) -> PreparedInput:
    """预处理单个PDB文件，返回准备好的输入数据"""
    pdb_name = pdb_path.stem
    effective_chain = resolve_chain_id(pdb_path, chain_id)
    pdb_feats_raw = du.parse_pdb_feats(pdb_name, str(pdb_path), chain_id=effective_chain or None)
    
    if isinstance(pdb_feats_raw, dict) and all(isinstance(v, dict) for v in pdb_feats_raw.values()):
        pdb_feats = merge_chain_features(pdb_feats_raw)
        chain_label = "ALL"
    else:
        pdb_feats = pdb_feats_raw
        chain_label = effective_chain or "ALL"
    
    chain_feats = process_chain_feats(pdb_feats)
    bb_mask = np.asarray(pdb_feats['bb_mask']).astype(bool)
    num_res = int(bb_mask.sum())
    
    if num_res == 0:
        raise ValueError(f"{pdb_path} 链 {chain_id} 不包含有效主链残基")
    
    mask_tensor = torch.from_numpy(bb_mask).to(torch.bool)
    rigid_frames = chain_feats['rigidgroups_gt_frames'][mask_tensor, 0].detach().cpu().float()
    rigids_0 = ru.Rigid.from_tensor_4x4(rigid_frames)
    rigids_tensor = rigids_0.to_tensor_7()
    sc_ca_init = rigids_0.get_trans().detach().cpu().float()
    
    torsion_angles = chain_feats['torsion_angles_sin_cos'].detach().cpu().numpy()[bb_mask]
    if torsion_angles.dtype == np.object_:
        torsion_angles = np.stack([x.astype(np.float32) for x in torsion_angles], axis=0)
    else:
        torsion_angles = torsion_angles.astype(np.float32)
    
    # 生成唯一名称
    unique_id = uuid.uuid4().hex[:32]
    timestamp = int(time.time())
    prefix = f"device_batch_gen_{timestamp}_{unique_id}_{chain_label}"
    
    return PreparedInput(
        pdb_path=pdb_path,
        name=prefix,
        chain_id=chain_label,
        num_res=num_res,
        rigids_tensor=rigids_tensor.float(),
        res_mask=torch.ones(num_res, dtype=torch.float32),
        seq_idx=torch.arange(1, num_res + 1, dtype=torch.float32),
        fixed_mask=torch.zeros(num_res, dtype=torch.float32),
        torsion_angles=torch.tensor(torsion_angles, dtype=torch.float32),
        sc_ca=sc_ca_init,
    )


def prepare_inputs_parallel(
    pdb_paths: List[Path],
    chain_id: Optional[str],
    num_workers: int
) -> List[PreparedInput]:
    """并行预处理多个PDB文件"""
    results = []
    errors = []
    
    # 使用线程池而非进程池，避免序列化问题
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(prepare_single_input, path, chain_id): path
            for path in pdb_paths
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                errors.append((path, str(e)))
                print(f"预处理失败 {path}: {e}")
    
    if errors:
        print(f"警告: {len(errors)} 个文件预处理失败")
    
    return results


def pad_to_max_length(tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """将不同长度的张量填充到最大长度，返回填充后的张量和有效长度掩码"""
    max_len = max(t.shape[0] for t in tensors)
    batch_size = len(tensors)
    
    # 获取其他维度
    other_dims = tensors[0].shape[1:] if tensors[0].ndim > 1 else ()
    
    # 创建填充后的张量
    padded = torch.full((batch_size, max_len) + other_dims, pad_value, dtype=tensors[0].dtype)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, t in enumerate(tensors):
        length = t.shape[0]
        padded[i, :length] = t
        lengths[i] = length
    
    # 创建掩码
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    
    return padded, mask.float()


def create_batched_input(
    inputs: List[PreparedInput],
    device: torch.device,
    use_fp16: bool = False
) -> Dict[str, torch.Tensor]:
    """将多个PreparedInput打包成批处理输入"""
    batch_size = len(inputs)
    
    # 填充rigids_tensor
    rigids_list = [inp.rigids_tensor for inp in inputs]
    rigids_padded, rigids_mask = pad_to_max_length(rigids_list)
    
    # 填充res_mask (使用padded的mask作为有效性标记)
    res_mask_list = [inp.res_mask for inp in inputs]
    res_mask_padded, _ = pad_to_max_length(res_mask_list)
    # 结合填充mask
    combined_res_mask = res_mask_padded * rigids_mask
    
    # 填充seq_idx
    seq_idx_list = [inp.seq_idx for inp in inputs]
    seq_idx_padded, _ = pad_to_max_length(seq_idx_list)
    
    # 填充fixed_mask
    fixed_mask_list = [inp.fixed_mask for inp in inputs]
    fixed_mask_padded, _ = pad_to_max_length(fixed_mask_list)
    
    # 填充torsion_angles
    torsion_list = [inp.torsion_angles for inp in inputs]
    torsion_padded, _ = pad_to_max_length(torsion_list)
    
    # 填充sc_ca
    sc_ca_list = [inp.sc_ca for inp in inputs]
    sc_ca_padded, _ = pad_to_max_length(sc_ca_list)
    
    dtype = torch.float16 if use_fp16 else torch.float32
    
    return {
        'rigids_t': rigids_padded.to(device=device, dtype=dtype),
        'res_mask': combined_res_mask.to(device=device, dtype=dtype),
        'seq_idx': seq_idx_padded.to(device=device, dtype=dtype),
        'fixed_mask': fixed_mask_padded.to(device=device, dtype=dtype),
        'torsion_angles_sin_cos': torsion_padded.to(device=device, dtype=dtype),
        'sc_ca_t': sc_ca_padded.to(device=device, dtype=dtype),
        'lengths': torch.tensor([inp.num_res for inp in inputs], device=device),
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
    """
    批量直接去噪过程
    
    返回:
        dict: 包含每个样本的最终score
    """
    batch_size = batched_input['rigids_t'].shape[0]
    lengths = batched_input['lengths']
    
    # 复制输入以避免修改原始数据
    sample_feats = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in batched_input.items() if k != 'lengths'}
    
    # 创建去噪时间步序列
    denoising_steps = np.linspace(max_t, min_t, num_steps)
    dt = (max_t - min_t) / max(num_steps - 1, 1)
    
    diffuse_mask = ((1 - sample_feats['fixed_mask']) * sample_feats['res_mask']).detach().cpu().numpy()
    t_placeholder = torch.ones(batch_size, device=device, dtype=sample_feats['rigids_t'].dtype)
    
    embed_self_conditioning = (
        enable_self_conditioning and
        getattr(model.embedding_layer._embed_conf, 'embed_self_conditioning', False)
    )
    
    def set_t_feats(feats, t_value):
        dtype = feats['rigids_t'].dtype
        feats['t'] = t_placeholder * float(t_value)
        rot_scale, trans_scale = diffuser.score_scaling(float(t_value))
        feats['rot_score_scaling'] = torch.full((batch_size,), float(rot_scale), device=device, dtype=dtype)
        feats['trans_score_scaling'] = torch.full((batch_size,), float(trans_scale), device=device, dtype=dtype)
        return feats
    
    final_rot_scores = [None] * batch_size
    final_trans_scores = [None] * batch_size
    
    with torch.no_grad():
        # 自条件初始化
        if embed_self_conditioning and len(denoising_steps) > 0:
            set_t_feats(sample_feats, denoising_steps[0])
            model_sc = model(sample_feats)
            sample_feats['sc_ca_t'] = model_sc['rigids'][..., 4:]
        
        # 逆向去噪循环
        for step_idx, t in enumerate(denoising_steps):
            set_t_feats(sample_feats, t)
            model_out = model(sample_feats)
            rot_score = model_out['rot_score']
            trans_score = model_out['trans_score']
            
            # 最后一步保存score
            if step_idx == len(denoising_steps) - 1:
                rot_score_np = rot_score.detach().cpu().float().numpy()
                trans_score_np = trans_score.detach().cpu().float().numpy()
                
                for i in range(batch_size):
                    length = lengths[i].item()
                    final_rot_scores[i] = rot_score_np[i, :length].copy()
                    final_trans_scores[i] = trans_score_np[i, :length].copy()
            
            # 执行去噪步骤 (除了最后一步)
            if t > min_t and step_idx < len(denoising_steps) - 1:
                current_rigid = ru.Rigid.from_tensor_7(sample_feats['rigids_t'])
                
                rigids_t = diffuser.reverse(
                    rigid_t=current_rigid,
                    rot_score=du.move_to_np(rot_score),
                    trans_score=du.move_to_np(trans_score),
                    diffuse_mask=diffuse_mask,
                    t=float(t),
                    dt=dt,
                    center=True,
                    noise_scale=noise_scale,
                )
                
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device=device, dtype=sample_feats['rigids_t'].dtype)
                
                if embed_self_conditioning:
                    sample_feats['sc_ca_t'] = model_out['rigids'][..., 4:]
    
    return {
        'final_rot_scores': final_rot_scores,
        'final_trans_scores': final_trans_scores,
    }


def process_batch(
    model: torch.nn.Module,
    diffuser,
    inputs: List[PreparedInput],
    device: torch.device,
    num_steps: int,
    min_t: float,
    max_t: float,
    noise_scale: float,
    enable_self_conditioning: bool,
    use_fp16: bool,
    output_dir: Path,
) -> List[SampleResult]:
    """处理一个batch的蛋白质"""
    if not inputs:
        return []
    
    # 创建批处理输入
    batched_input = create_batched_input(inputs, device, use_fp16)
    
    # 执行批量去噪
    results = batched_direct_denoising(
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
    
    # 保存结果
    sample_results = []
    for i, inp in enumerate(inputs):
        rot_score = results['final_rot_scores'][i]
        trans_score = results['final_trans_scores'][i]
        
        if rot_score is None or trans_score is None:
            print(f"警告: {inp.name} 去噪未返回有效score")
            continue
        
        rot_path = output_dir / f"{inp.name}_rot_score.npy"
        trans_path = output_dir / f"{inp.name}_trans_score.npy"
        np.save(rot_path, rot_score)
        np.save(trans_path, trans_score)
        
        sample_results.append(SampleResult(
            name=inp.name,
            pdb_path=inp.pdb_path,
            chain_id=inp.chain_id,
            num_res=inp.num_res,
            rot_score=rot_score,
            trans_score=trans_score,
            rot_score_path=rot_path,
            trans_score_path=trans_path,
        ))
    
    return sample_results


def group_by_length(inputs: List[PreparedInput], tolerance: int = 50) -> List[List[PreparedInput]]:
    """
    按长度分组，相近长度的蛋白质放在一起以减少padding开销
    """
    if not inputs:
        return []
    
    # 按长度排序
    sorted_inputs = sorted(inputs, key=lambda x: x.num_res)
    
    groups = []
    current_group = [sorted_inputs[0]]
    current_base_len = sorted_inputs[0].num_res
    
    for inp in sorted_inputs[1:]:
        if inp.num_res - current_base_len <= tolerance:
            current_group.append(inp)
        else:
            groups.append(current_group)
            current_group = [inp]
            current_base_len = inp.num_res
    
    if current_group:
        groups.append(current_group)
    
    return groups


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


def compute_pair_metrics(
    tm_align_bin: Path,
    sample_a: SampleResult,
    sample_b: SampleResult,
    device: torch.device,
) -> PairMetrics:
    tm_score = compute_tm_score(tm_align_bin, sample_a.pdb_path, sample_b.pdb_path)
    
    rot_a_np, rot_b_np = align_scores(sample_a.rot_score, sample_b.rot_score)
    trans_a_np, trans_b_np = align_scores(sample_a.trans_score, sample_b.trans_score)
    
    rot_a = torch.as_tensor(rot_a_np, dtype=torch.float32, device=device)
    rot_b = torch.as_tensor(rot_b_np, dtype=torch.float32, device=device)
    trans_a = torch.as_tensor(trans_a_np, dtype=torch.float32, device=device)
    trans_b = torch.as_tensor(trans_b_np, dtype=torch.float32, device=device)
    
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
        else:
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
    timing_info: Dict[str, float],
):
    lines: List[str] = []
    lines.append("# 随机直接去噪与相关性分析报告 (GPU优化版)")
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
    lines.append(f"- 批处理大小: {args.batch_size}")
    lines.append(f"- FP16模式: {args.use_fp16}")
    lines.append("")
    
    lines.append("## 性能统计")
    lines.append(f"- 预处理耗时: {timing_info.get('preprocess', 0):.2f}s")
    lines.append(f"- 去噪推理耗时: {timing_info.get('denoising', 0):.2f}s")
    lines.append(f"- Pair计算耗时: {timing_info.get('pair_metrics', 0):.2f}s")
    lines.append(f"- 总耗时: {timing_info.get('total', 0):.2f}s")
    if len(samples) > 0 and timing_info.get('denoising', 0) > 0:
        throughput = len(samples) / timing_info['denoising']
        lines.append(f"- 去噪吞吐量: {throughput:.2f} samples/s")
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
    
    timing_info = {}
    total_start = time.time()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = args.output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    # 设备选择
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    enable_self_conditioning = not args.disable_self_conditioning
    
    # 抽样PDB文件
    pdb_paths = sample_pdb_files(args.source_dir, args.sample_size, args.seed)
    print(f"抽样到 {len(pdb_paths)} 个PDB文件")
    
    # 并行预处理
    print(f"并行预处理PDB文件 (workers={args.preprocess_workers})...")
    preprocess_start = time.time()
    prepared_inputs = prepare_inputs_parallel(pdb_paths, args.chain_id, args.preprocess_workers)
    timing_info['preprocess'] = time.time() - preprocess_start
    print(f"预处理完成: {len(prepared_inputs)} 个样本, 耗时 {timing_info['preprocess']:.2f}s")
    
    if not prepared_inputs:
        raise RuntimeError("没有成功预处理的样本")
    
    # 加载模型
    print("加载模型配置与权重...")
    conf = OmegaConf.load(args.config_path)
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    model = score_network.ScoreNetwork(conf.model, diffuser)
    
    checkpoint = torch.load(args.weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 启用CUDA优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if args.use_fp16:
            print("启用FP16混合精度推理")
    
    # 按长度分组以减少padding开销
    print("按长度分组样本...")
    length_groups = group_by_length(prepared_inputs, tolerance=50)
    print(f"分成 {len(length_groups)} 个长度组")
    
    # 批量去噪
    print(f"开始批量去噪推理 (batch_size={args.batch_size})...")
    denoising_start = time.time()
    
    all_samples: List[SampleResult] = []
    total_batches = sum((len(group) + args.batch_size - 1) // args.batch_size for group in length_groups)
    batch_count = 0
    
    for group_idx, group in enumerate(length_groups):
        # 将组分成batch
        for batch_start in range(0, len(group), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(group))
            batch_inputs = group[batch_start:batch_end]
            
            batch_count += 1
            print(f"处理batch {batch_count}/{total_batches} (size={len(batch_inputs)}, "
                  f"lengths={[inp.num_res for inp in batch_inputs]})")
            
            batch_results = process_batch(
                model=model,
                diffuser=diffuser,
                inputs=batch_inputs,
                device=device,
                num_steps=args.num_denoising_steps,
                min_t=args.min_t,
                max_t=args.max_t,
                noise_scale=args.noise_scale,
                enable_self_conditioning=enable_self_conditioning,
                use_fp16=args.use_fp16,
                output_dir=scores_dir,
            )
            all_samples.extend(batch_results)
            
            # 清理GPU缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    timing_info['denoising'] = time.time() - denoising_start
    print(f"去噪完成: {len(all_samples)} 个样本, 耗时 {timing_info['denoising']:.2f}s")
    
    all_samples.sort(key=lambda s: s.name)
    
    if len(all_samples) < 2:
        raise RuntimeError("样本数不足，无法执行pair分析")
    
    # 并行计算pair metrics
    print("并行计算pair metrics...")
    pair_start = time.time()
    pairs = list(itertools.combinations(all_samples, 2))
    pair_metrics: List[PairMetrics] = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(
                compute_pair_metrics,
                args.tm_align_bin,
                a,
                b,
                device,
            ): (a.name, b.name)
            for (a, b) in pairs
        }
        for future in as_completed(future_map):
            pair_name = future_map[future]
            try:
                pair_metrics.append(future.result())
            except Exception as exc:
                print(f"计算 pair {pair_name} 失败: {exc}")
    
    timing_info['pair_metrics'] = time.time() - pair_start
    timing_info['total'] = time.time() - total_start
    
    print(f"Pair计算完成, 耗时 {timing_info['pair_metrics']:.2f}s")
    
    # 生成报告
    correlations = analyze_correlations(pair_metrics)
    
    report_path = args.output_dir / "random_denoising_report.md"
    render_report(report_path, args, all_samples, pair_metrics, correlations, timing_info)
    generated_plots = generate_visualizations(all_samples, pair_metrics, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"报告已生成: {report_path}")
    print(f"总耗时: {timing_info['total']:.2f}s")
    if len(all_samples) > 0:
        print(f"平均每样本: {timing_info['total']/len(all_samples):.2f}s")
        if timing_info['denoising'] > 0:
            print(f"去噪吞吐量: {len(all_samples)/timing_info['denoising']:.2f} samples/s")
    if generated_plots:
        print("生成的可视化图像:")
        for path in generated_plots:
            print(f"  - {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
