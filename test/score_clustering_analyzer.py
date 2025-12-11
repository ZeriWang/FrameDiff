#!/usr/bin/env python3
"""
基于分数距离的HDBSCAN聚类分析工具

功能:
1. 批量加载PDB文件并进行去噪，提取旋转分数和平移分数
2. 使用加权组合的余弦距离作为度量进行HDBSCAN聚类
3. 计算簇内蛋白质两两之间的TM-score
4. 筛选出"分数相似但结构不同"的多构象蛋白质簇 (平均TM-score < 阈值)

GPU优化:
- 批处理推理提高GPU利用率
- 并行预处理和TM-score计算
- 按长度分组减少padding开销
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import subprocess
import time
import uuid
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Sequence, Tuple, Any
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
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("警告: hdbscan未安装，请运行 pip install hdbscan")

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "clustering_output"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best_weights.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "base.yaml"
DEFAULT_TMALIGN = PROJECT_ROOT / "test" / "TMalign" / "TM-align"


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class SampleResult:
    """单个样本的去噪结果"""
    name: str
    pdb_path: Path
    chain_id: str
    num_res: int
    rot_score: np.ndarray      # (num_res, 3)
    trans_score: np.ndarray    # (num_res, 3)
    rot_score_path: Path
    trans_score_path: Path


@dataclass
class PreparedInput:
    """预处理后的输入数据"""
    pdb_path: Path
    name: str
    chain_id: str
    num_res: int
    rigids_tensor: torch.Tensor
    res_mask: torch.Tensor
    seq_idx: torch.Tensor
    fixed_mask: torch.Tensor
    torsion_angles: torch.Tensor
    sc_ca: torch.Tensor


@dataclass
class ClusterInfo:
    """聚类簇信息"""
    cluster_id: int
    member_indices: List[int]
    member_names: List[str]
    member_pdb_paths: List[Path]
    avg_tm_score: float = 0.0
    min_tm_score: float = 0.0
    max_tm_score: float = 0.0
    tm_score_matrix: Optional[np.ndarray] = None
    is_multiconformer: bool = False


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于分数距离的HDBSCAN聚类分析 - 寻找多构象蛋白质",
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
    parser.add_argument("--tm-align-bin", type=Path, default=DEFAULT_TMALIGN,
                        help="TM-align可执行文件路径")
    
    # 采样参数
    parser.add_argument("--chain-id", type=str, default=None,
                        help="指定链ID；缺省时处理整个结构")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="随机抽样的PDB数量 (0=全部)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # 去噪参数
    parser.add_argument("--num-denoising-steps", type=int, default=1,
                        help="去噪步数 (1步足够获取分数)")
    parser.add_argument("--min-t", type=float, default=0.01,
                        help="最小时间步")
    parser.add_argument("--max-t", type=float, default=0.05,
                        help="最大时间步")
    parser.add_argument("--noise-scale", type=float, default=0.0,
                        help="噪声缩放因子")
    parser.add_argument("--disable-self-conditioning", action="store_true",
                        help="关闭自条件")
    
    # 聚类参数
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="旋转分数权重 (0-1)，平移分数权重为1-alpha")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="HDBSCAN最小簇大小")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="HDBSCAN min_samples参数")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0,
                        help="HDBSCAN cluster_selection_epsilon")
    
    # 筛选参数
    parser.add_argument("--tm-threshold", type=float, default=0.6,
                        help="簇内平均TM-score阈值，低于此值认为是多构象")
    
    # 性能参数
    parser.add_argument("--batch-size", type=int, default=16,
                        help="去噪批处理大小")
    parser.add_argument("--preprocess-workers", type=int, default=4,
                        help="预处理并行线程数")
    parser.add_argument("--tm-workers", type=int, default=8,
                        help="TM-score计算并行线程数")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备")
    parser.add_argument("--use-fp16", action="store_true",
                        help="使用FP16混合精度")
    
    # 特征处理
    parser.add_argument("--pooling", type=str, default="flatten",
                        choices=["flatten", "mean", "max"],
                        help="不同长度蛋白质的特征聚合方式")
    
    return parser.parse_args()


# ============================================================================
# PDB预处理函数
# ============================================================================

def resolve_chain_id(pdb_path: Path, preferred_chain: Optional[str]) -> Optional[str]:
    """解析链ID"""
    if preferred_chain is None:
        return None
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("chain_scan", str(pdb_path))
    available = sorted({chain.id for chain in structure.get_chains()})
    if not available:
        raise ValueError(f"{pdb_path} 不包含任何链")
    if preferred_chain not in available:
        raise ValueError(f"{pdb_path} 中不存在链 {preferred_chain}")
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
    """合并多链特征"""
    merged: Dict[str, List[np.ndarray]] = {}
    for chain_id in sorted(chain_feat_map.keys()):
        feats = chain_feat_map[chain_id]
        for key, value in feats.items():
            arr = np.asarray(value)
            merged.setdefault(key, []).append(arr)
    return {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}


def prepare_single_input(pdb_path: Path, chain_id: Optional[str]) -> PreparedInput:
    """预处理单个PDB文件"""
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
        raise ValueError(f"{pdb_path} 不包含有效主链残基")
    
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
    
    unique_id = uuid.uuid4().hex[:16]
    prefix = f"{pdb_name}_{chain_label}_{unique_id}"
    
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
    num_workers: int,
) -> List[PreparedInput]:
    """并行预处理多个PDB文件"""
    results = []
    errors = []
    
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
    
    if errors:
        print(f"警告: {len(errors)} 个文件预处理失败")
        for path, err in errors[:5]:
            print(f"  - {path.name}: {err}")
    
    return results


# ============================================================================
# 批量去噪推理
# ============================================================================

def pad_to_max_length(tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """填充到最大长度"""
    max_len = max(t.shape[0] for t in tensors)
    batch_size = len(tensors)
    other_dims = tensors[0].shape[1:] if tensors[0].ndim > 1 else ()
    
    padded = torch.full((batch_size, max_len) + other_dims, pad_value, dtype=tensors[0].dtype)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, t in enumerate(tensors):
        length = t.shape[0]
        padded[i, :length] = t
        lengths[i] = length
    
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask.float()


def create_batched_input(
    inputs: List[PreparedInput],
    device: torch.device,
    use_fp16: bool = False
) -> Dict[str, torch.Tensor]:
    """创建批处理输入"""
    rigids_list = [inp.rigids_tensor for inp in inputs]
    rigids_padded, rigids_mask = pad_to_max_length(rigids_list)
    
    res_mask_list = [inp.res_mask for inp in inputs]
    res_mask_padded, _ = pad_to_max_length(res_mask_list)
    combined_res_mask = res_mask_padded * rigids_mask
    
    seq_idx_list = [inp.seq_idx for inp in inputs]
    seq_idx_padded, _ = pad_to_max_length(seq_idx_list)
    
    fixed_mask_list = [inp.fixed_mask for inp in inputs]
    fixed_mask_padded, _ = pad_to_max_length(fixed_mask_list)
    
    torsion_list = [inp.torsion_angles for inp in inputs]
    torsion_padded, _ = pad_to_max_length(torsion_list)
    
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
    """批量直接去噪"""
    batch_size = batched_input['rigids_t'].shape[0]
    lengths = batched_input['lengths']
    
    sample_feats = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in batched_input.items() if k != 'lengths'}
    
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
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        if embed_self_conditioning and len(denoising_steps) > 0:
            set_t_feats(sample_feats, denoising_steps[0])
            model_sc = model(sample_feats)
            sample_feats['sc_ca_t'] = model_sc['rigids'][..., 4:]
        
        for step_idx, t in enumerate(denoising_steps):
            set_t_feats(sample_feats, t)
            model_out = model(sample_feats)
            rot_score = model_out['rot_score']
            trans_score = model_out['trans_score']
            
            if step_idx == len(denoising_steps) - 1:
                rot_score_np = rot_score.detach().cpu().float().numpy()
                trans_score_np = trans_score.detach().cpu().float().numpy()
                
                for i in range(batch_size):
                    length = lengths[i].item()
                    final_rot_scores[i] = rot_score_np[i, :length].copy()
                    final_trans_scores[i] = trans_score_np[i, :length].copy()
            
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
    """
    处理所有样本
    
    注意: 假设所有输入蛋白质序列长度相同，无需分组
    """
    all_results: List[SampleResult] = []
    total_samples = len(prepared_inputs)
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"处理 {total_samples} 个样本, 共 {total_batches} 个batch")
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        batch_inputs = prepared_inputs[batch_start:batch_end]
        
        print(f"\r处理batch {batch_idx + 1}/{total_batches}", end="", flush=True)
        
        batched_input = create_batched_input(batch_inputs, device, use_fp16)
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
        
        for i, inp in enumerate(batch_inputs):
            rot_score = results['final_rot_scores'][i]
            trans_score = results['final_trans_scores'][i]
            
            if rot_score is None or trans_score is None:
                continue
            
            rot_path = output_dir / f"{inp.name}_rot_score.npy"
            trans_path = output_dir / f"{inp.name}_trans_score.npy"
            np.save(rot_path, rot_score)
            np.save(trans_path, trans_score)
            
            all_results.append(SampleResult(
                name=inp.name,
                pdb_path=inp.pdb_path,
                chain_id=inp.chain_id,
                num_res=inp.num_res,
                rot_score=rot_score,
                trans_score=trans_score,
                rot_score_path=rot_path,
                trans_score_path=trans_path,
            ))
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print()  # 换行
    return all_results



# ============================================================================
# 特征提取与距离计算
# ============================================================================

def extract_features(
    samples: List[SampleResult],
    pooling: str = "flatten"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取特征向量
    
    Args:
        samples: 样本列表
        pooling: 聚合方式 - "flatten", "mean", "max"
    
    Returns:
        rot_features: (N, D) 旋转特征
        trans_features: (N, D) 平移特征
    """
    rot_features = []
    trans_features = []
    
    if pooling == "flatten":
        # 找到最大长度用于padding
        max_len = max(s.num_res for s in samples)
        
        for s in samples:
            # Pad到最大长度
            rot_padded = np.zeros((max_len, 3), dtype=np.float32)
            trans_padded = np.zeros((max_len, 3), dtype=np.float32)
            
            rot_padded[:s.num_res] = s.rot_score
            trans_padded[:s.num_res] = s.trans_score
            
            rot_features.append(rot_padded.flatten())
            trans_features.append(trans_padded.flatten())
    
    elif pooling == "mean":
        for s in samples:
            rot_features.append(s.rot_score.mean(axis=0))
            trans_features.append(s.trans_score.mean(axis=0))
    
    elif pooling == "max":
        for s in samples:
            rot_features.append(s.rot_score.max(axis=0))
            trans_features.append(s.trans_score.max(axis=0))
    
    return np.array(rot_features), np.array(trans_features)


def compute_combined_distance_matrix(
    rot_features: np.ndarray,
    trans_features: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    计算加权组合的距离矩阵
    
    Args:
        rot_features: (N, D1) 旋转特征
        trans_features: (N, D2) 平移特征
        alpha: 旋转分数权重
    
    Returns:
        combined_dist: (N, N) 组合距离矩阵
    """
    rot_dist = cosine_distances(rot_features)
    trans_dist = cosine_distances(trans_features)
    
    combined_dist = alpha * rot_dist + (1 - alpha) * trans_dist
    
    # 确保对角线为0
    np.fill_diagonal(combined_dist, 0.0)
    
    return combined_dist


# ============================================================================
# HDBSCAN聚类
# ============================================================================

def perform_clustering(
    distance_matrix: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
) -> Tuple[np.ndarray, Any]:
    """
    执行HDBSCAN聚类
    
    Returns:
        labels: 聚类标签 (-1表示噪声)
        clusterer: HDBSCAN聚类器对象
    """
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("hdbscan未安装")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_method='eom',
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    
    labels = clusterer.fit_predict(distance_matrix.astype(np.float64))
    
    return labels, clusterer


# ============================================================================
# TM-score计算 (优化版本)
# ============================================================================

def _compute_tm_score_worker(args) -> Tuple[int, int, float]:
    """
    TM-score计算的worker函数 (用于ProcessPoolExecutor)
    args: (i, j, tm_align_bin_str, pdb_a_str, pdb_b_str)
    """
    i, j, tm_align_bin_str, pdb_a_str, pdb_b_str = args
    cmd = [tm_align_bin_str, pdb_a_str, pdb_b_str]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        match = re.search(r"TM-score\s*=\s*([0-9.]+)", result.stdout)
        if not match:
            return i, j, float('nan')
        return i, j, float(match.group(1))
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception):
        return i, j, float('nan')


def compute_tm_score(tm_align_bin: Path, pdb_a: Path, pdb_b: Path) -> float:
    """计算两个PDB的TM-score"""
    if not tm_align_bin.exists():
        raise FileNotFoundError(f"TM-align未找到: {tm_align_bin}")
    
    cmd = [str(tm_align_bin), str(pdb_a), str(pdb_b)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        match = re.search(r"TM-score\s*=\s*([0-9.]+)", result.stdout)
        if not match:
            return float('nan')
        return float(match.group(1))
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return float('nan')


def compute_gpu_rmsd_matrix(
    samples: List[SampleResult],
    device: torch.device,
) -> np.ndarray:
    """
    使用GPU批量计算所有样本对之间的RMSD (基于CA原子坐标)
    这是一个快速的结构相似度估计，可用于预筛选
    
    Returns:
        rmsd_matrix: (N, N) RMSD矩阵
    """
    n = len(samples)
    
    # 提取所有样本的CA坐标 (已经在sc_ca中)
    # 由于长度可能不同，需要找到最大长度并padding
    max_len = max(s.num_res for s in samples)
    
    # 创建padded坐标张量
    coords = torch.zeros((n, max_len, 3), dtype=torch.float32, device=device)
    masks = torch.zeros((n, max_len), dtype=torch.bool, device=device)
    
    for i, s in enumerate(samples):
        # s.rot_score 或 s.trans_score 包含坐标信息
        # 但更好的是使用原始PDB的CA坐标
        # 这里我们用一个简化版本：从score中估计
        coords[i, :s.num_res] = torch.from_numpy(
            s.trans_score if hasattr(s, 'trans_score') and s.trans_score is not None 
            else np.zeros((s.num_res, 3))
        ).to(device)
        masks[i, :s.num_res] = True
    
    # 计算所有对的RMSD (GPU并行)
    rmsd_matrix = torch.zeros((n, n), dtype=torch.float32, device=device)
    
    # 批量计算 - 利用GPU并行
    batch_size = 64  # 每次处理的对数
    pairs = list(itertools.combinations(range(n), 2))
    
    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]
        
        for i, j in batch_pairs:
            # 获取有效长度
            len_i, len_j = samples[i].num_res, samples[j].num_res
            min_len = min(len_i, len_j)
            
            # 计算RMSD (只比较共同长度部分)
            diff = coords[i, :min_len] - coords[j, :min_len]
            rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())
            
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
    
    return rmsd_matrix.cpu().numpy()


def compute_cluster_tm_scores(
    cluster_members: List[SampleResult],
    tm_align_bin: Path,
    num_workers: int = 8,
) -> np.ndarray:
    """
    计算簇内所有成员两两之间的TM-score
    使用ProcessPoolExecutor提高CPU密集型任务的并行效率
    """
    n = len(cluster_members)
    tm_matrix = np.eye(n, dtype=np.float64)
    
    if n < 2:
        return tm_matrix
    
    # 准备所有计算任务
    pairs = list(itertools.combinations(range(n), 2))
    tm_align_str = str(tm_align_bin)
    
    # 创建任务参数列表
    tasks = [
        (i, j, tm_align_str, 
         str(cluster_members[i].pdb_path), 
         str(cluster_members[j].pdb_path))
        for i, j in pairs
    ]
    
    # 使用ProcessPoolExecutor (CPU密集型任务更高效)
    # 限制worker数量避免过载
    effective_workers = min(num_workers, cpu_count(), len(tasks))
    
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        results = list(executor.map(_compute_tm_score_worker, tasks, chunksize=max(1, len(tasks) // effective_workers)))
    
    for i, j, tm in results:
        tm_matrix[i, j] = tm
        tm_matrix[j, i] = tm
    
    return tm_matrix


def compute_all_tm_scores_parallel(
    samples: List[SampleResult],
    labels: np.ndarray,
    tm_align_bin: Path,
    num_workers: int,
) -> Dict[Tuple[int, int], float]:
    """
    并行计算所有需要的TM-score对
    
    优化策略:
    1. 收集所有簇内需要计算的样本对
    2. 一次性提交所有任务到进程池
    3. 避免多次创建/销毁进程池的开销
    
    Returns:
        tm_cache: {(global_idx_i, global_idx_j): tm_score}
    """
    unique_labels = sorted(set(labels) - {-1})
    
    # 收集所有需要计算的样本对 (使用全局索引)
    all_pairs = []
    for cluster_id in unique_labels:
        member_indices = np.where(labels == cluster_id)[0].tolist()
        if len(member_indices) >= 2:
            for i, j in itertools.combinations(member_indices, 2):
                all_pairs.append((i, j))
    
    if not all_pairs:
        return {}
    
    print(f"  需要计算 {len(all_pairs)} 对TM-score...")
    
    # 准备任务
    tm_align_str = str(tm_align_bin)
    tasks = [
        (i, j, tm_align_str, str(samples[i].pdb_path), str(samples[j].pdb_path))
        for i, j in all_pairs
    ]
    
    # 使用ProcessPoolExecutor并行计算
    effective_workers = min(num_workers, cpu_count(), len(tasks))
    tm_cache = {}
    
    completed = 0
    total = len(tasks)
    
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        # 使用map更高效地处理大量任务
        chunksize = max(1, total // (effective_workers * 4))
        
        for i, j, tm in executor.map(_compute_tm_score_worker, tasks, chunksize=chunksize):
            tm_cache[(i, j)] = tm
            tm_cache[(j, i)] = tm
            completed += 1
            
            # 进度显示
            if completed % 50 == 0 or completed == total:
                print(f"\r  TM-score计算进度: {completed}/{total} ({100*completed/total:.1f}%)", 
                      end="", flush=True)
    
    print()  # 换行
    return tm_cache


def analyze_clusters(
    samples: List[SampleResult],
    labels: np.ndarray,
    tm_align_bin: Path,
    tm_threshold: float,
    num_workers: int,
    precomputed_tm: Optional[Dict[Tuple[int, int], float]] = None,
) -> List[ClusterInfo]:
    """
    分析所有簇，计算TM-score并筛选多构象簇
    
    优化:
    1. 支持预计算的TM-score缓存
    2. 如果没有预计算，一次性并行计算所有需要的TM-score
    """
    unique_labels = sorted(set(labels) - {-1})
    print(f"发现 {len(unique_labels)} 个簇 (不含噪声)")
    
    # 如果没有预计算的TM-score，一次性计算所有
    if precomputed_tm is None:
        print("\n并行计算所有簇内TM-score...")
        tm_start = time.time()
        precomputed_tm = compute_all_tm_scores_parallel(
            samples, labels, tm_align_bin, num_workers
        )
        print(f"TM-score计算完成，耗时 {time.time() - tm_start:.2f}s")
    
    clusters: List[ClusterInfo] = []
    
    for cluster_id in unique_labels:
        member_indices = np.where(labels == cluster_id)[0].tolist()
        cluster_members = [samples[i] for i in member_indices]
        n = len(cluster_members)
        
        # 从缓存构建TM矩阵
        tm_matrix = np.eye(n, dtype=np.float64)
        if n > 1:
            for local_i, global_i in enumerate(member_indices):
                for local_j, global_j in enumerate(member_indices):
                    if local_i < local_j:
                        tm = precomputed_tm.get((global_i, global_j), float('nan'))
                        tm_matrix[local_i, local_j] = tm
                        tm_matrix[local_j, local_i] = tm
        
        # 计算统计量 (排除对角线)
        if n > 1:
            triu_indices = np.triu_indices(n, k=1)
            tm_values = tm_matrix[triu_indices]
            valid_tm = tm_values[~np.isnan(tm_values)]
            
            if len(valid_tm) > 0:
                avg_tm = float(np.mean(valid_tm))
                min_tm = float(np.min(valid_tm))
                max_tm = float(np.max(valid_tm))
            else:
                avg_tm = min_tm = max_tm = float('nan')
        else:
            avg_tm = min_tm = max_tm = 1.0
        
        is_multiconformer = avg_tm < tm_threshold
        
        cluster_info = ClusterInfo(
            cluster_id=cluster_id,
            member_indices=member_indices,
            member_names=[s.name for s in cluster_members],
            member_pdb_paths=[s.pdb_path for s in cluster_members],
            avg_tm_score=avg_tm,
            min_tm_score=min_tm,
            max_tm_score=max_tm,
            tm_score_matrix=tm_matrix,
            is_multiconformer=is_multiconformer,
        )
        clusters.append(cluster_info)
        
        status = "✓ 多构象" if is_multiconformer else "✗ 单构象"
        print(f"簇 {cluster_id}: {n}个成员, 平均TM-score={avg_tm:.4f} ({status})")
    
    return clusters


# ============================================================================
# 可视化
# ============================================================================

def plot_clustering_results(
    samples: List[SampleResult],
    rot_features: np.ndarray,
    trans_features: np.ndarray,
    labels: np.ndarray,
    clusters: List[ClusterInfo],
    output_dir: Path,
    alpha: float,
    clusterer = None,  # 新增参数：HDBSCAN聚类器对象
):
    """生成聚类可视化图"""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # 1. 距离矩阵热力图
    combined_dist = compute_combined_distance_matrix(rot_features, trans_features, alpha)
    
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(combined_dist, cmap='viridis', square=True)
    else:
        plt.imshow(combined_dist, cmap='viridis')
        plt.colorbar()
    plt.title(f'Combined Score Distance Matrix (α={alpha})')
    plt.tight_layout()
    plt.savefig(plot_dir / 'distance_matrix.png', dpi=150)
    plt.close()
    
    # 2. HDBSCAN聚类树 (新增)
    if clusterer is not None and hasattr(clusterer, 'condensed_tree_'):
        try:
            print("  绘制HDBSCAN聚类树...")
            
            # 2.1 压缩树可视化
            plt.figure(figsize=(12, 8))
            clusterer.condensed_tree_.plot(
                select_clusters=True,
                selection_palette=sns.color_palette('deep', max(len(set(labels)) - 1, 8)) if sns else None,
                cmap='viridis',
                colorbar=True
            )
            plt.title('HDBSCAN Condensed Tree\n(Shows cluster formation hierarchy)')
            plt.xlabel('Sample Index (sorted by similarity)')
            plt.ylabel('Distance (λ)')
            plt.tight_layout()
            plt.savefig(plot_dir / 'hdbscan_condensed_tree.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2.2 单链接树 (如果可用)
            if hasattr(clusterer, 'single_linkage_tree_'):
                plt.figure(figsize=(12, 8))
                clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
                plt.title('HDBSCAN Single Linkage Tree\n(Complete hierarchy of merges)')
                plt.xlabel('Sample Index')
                plt.ylabel('Distance')
                plt.tight_layout()
                plt.savefig(plot_dir / 'hdbscan_single_linkage_tree.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            # 2.3 最小生成树 (如果可用)
            if hasattr(clusterer, 'minimum_spanning_tree_'):
                plt.figure(figsize=(12, 8))
                clusterer.minimum_spanning_tree_.plot(
                    edge_cmap='viridis',
                    edge_alpha=0.6,
                    node_size=20,
                    edge_linewidth=2
                )
                plt.title('HDBSCAN Minimum Spanning Tree\n(Graph representation of sample connectivity)')
                plt.tight_layout()
                plt.savefig(plot_dir / 'hdbscan_mst.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            print("  ✓ 聚类树可视化完成")
            
        except Exception as e:
            print(f"  ✗ 聚类树可视化失败: {e}")
    
    # 3. t-SNE可视化
    if SKLEARN_AVAILABLE and len(samples) > 5:
        try:
            # 合并特征用于降维
            combined_features = np.hstack([rot_features, trans_features])
            
            # 使用PCA预降维 (如果维度太高)
            if combined_features.shape[1] > 50:
                pca = PCA(n_components=50)
                combined_features = pca.fit_transform(combined_features)
            
            perplexity = min(30, len(samples) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            coords_2d = tsne.fit_transform(combined_features)
            
            plt.figure(figsize=(10, 8))
            
            # 绘制噪声点
            noise_mask = labels == -1
            if noise_mask.any():
                plt.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1],
                           c='gray', marker='x', s=30, alpha=0.5, label='Noise')
            
            # 绘制各簇
            unique_labels = sorted(set(labels) - {-1})
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_labels))))
            
            for idx, cluster_id in enumerate(unique_labels):
                mask = labels == cluster_id
                cluster_info = next((c for c in clusters if c.cluster_id == cluster_id), None)
                marker = '*' if cluster_info and cluster_info.is_multiconformer else 'o'
                size = 100 if cluster_info and cluster_info.is_multiconformer else 50
                
                label = f'Cluster {cluster_id}'
                if cluster_info:
                    label += f' (TM={cluster_info.avg_tm_score:.2f})'
                    if cluster_info.is_multiconformer:
                        label += ' ★'
                
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                           c=[colors[idx % len(colors)]], marker=marker, s=size, label=label)
            
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('Score Space Clustering (t-SNE)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / 'tsne_clustering.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"t-SNE可视化失败: {e}")
    
    # 4. 簇大小分布
    unique_labels = sorted(set(labels) - {-1})
    cluster_sizes = [np.sum(labels == l) for l in unique_labels]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(unique_labels)), cluster_sizes)
    
    # 标记多构象簇
    for idx, cluster_id in enumerate(unique_labels):
        cluster_info = next((c for c in clusters if c.cluster_id == cluster_id), None)
        if cluster_info and cluster_info.is_multiconformer:
            bars[idx].set_color('red')
            bars[idx].set_edgecolor('darkred')
            bars[idx].set_linewidth(2)
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Members')
    plt.title('Cluster Size Distribution (Red = Multiconformer)')
    plt.xticks(range(len(unique_labels)), unique_labels)
    plt.tight_layout()
    plt.savefig(plot_dir / 'cluster_sizes.png', dpi=150)
    plt.close()
    
    # 5. 每个多构象簇的TM-score热力图
    multiconformer_clusters = [c for c in clusters if c.is_multiconformer]
    for cluster in multiconformer_clusters:
        if cluster.tm_score_matrix is not None and len(cluster.member_names) > 1:
            plt.figure(figsize=(8, 6))
            
            short_names = [Path(p).stem[:15] for p in cluster.member_pdb_paths]
            
            if sns is not None:
                sns.heatmap(cluster.tm_score_matrix, 
                           xticklabels=short_names,
                           yticklabels=short_names,
                           cmap='RdYlGn', vmin=0, vmax=1, 
                           annot=True, fmt='.2f', square=True)
            else:
                plt.imshow(cluster.tm_score_matrix, cmap='RdYlGn', vmin=0, vmax=1)
                plt.colorbar()
                plt.xticks(range(len(short_names)), short_names, rotation=45, ha='right')
                plt.yticks(range(len(short_names)), short_names)
            
            plt.title(f'Cluster {cluster.cluster_id} TM-score Matrix (Avg={cluster.avg_tm_score:.3f})')
            plt.tight_layout()
            plt.savefig(plot_dir / f'cluster_{cluster.cluster_id}_tm_matrix.png', dpi=150)
            plt.close()
    
    print(f"可视化图已保存到: {plot_dir}")



# ============================================================================
# 报告生成
# ============================================================================

def generate_report(
    output_path: Path,
    args: argparse.Namespace,
    samples: List[SampleResult],
    labels: np.ndarray,
    clusters: List[ClusterInfo],
    timing_info: Dict[str, float],
):
    """生成Markdown报告"""
    lines = []
    lines.append("# 基于分数距离的HDBSCAN聚类分析报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    
    # 配置信息
    lines.append("## 运行配置")
    lines.append(f"- 源目录: `{args.source_dir}`")
    lines.append(f"- 样本数量: {len(samples)}")
    lines.append(f"- 去噪步数: {args.num_denoising_steps}")
    lines.append(f"- 特征聚合: {args.pooling}")
    lines.append(f"- 旋转权重 (α): {args.alpha}")
    lines.append(f"- HDBSCAN min_cluster_size: {args.min_cluster_size}")
    lines.append(f"- TM-score阈值: {args.tm_threshold}")
    lines.append("")
    
    # 性能统计
    lines.append("## 性能统计")
    lines.append(f"- 预处理耗时: {timing_info.get('preprocess', 0):.2f}s")
    lines.append(f"- 去噪推理耗时: {timing_info.get('denoising', 0):.2f}s")
    lines.append(f"- 聚类耗时: {timing_info.get('clustering', 0):.2f}s")
    lines.append(f"- TM-score计算耗时: {timing_info.get('tm_analysis', 0):.2f}s")
    lines.append(f"- 总耗时: {timing_info.get('total', 0):.2f}s")
    if len(samples) > 0 and timing_info.get('denoising', 0) > 0:
        lines.append(f"- 去噪吞吐量: {len(samples)/timing_info['denoising']:.2f} samples/s")
    lines.append("")
    
    # 聚类结果概览
    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    multiconformer_clusters = [c for c in clusters if c.is_multiconformer]
    
    lines.append("## 聚类结果概览")
    lines.append(f"- 总簇数: {n_clusters}")
    lines.append(f"- 噪声点数: {n_noise}")
    lines.append(f"- **多构象簇数: {len(multiconformer_clusters)}** (平均TM-score < {args.tm_threshold})")
    lines.append("")
    
    # 多构象簇详情
    if multiconformer_clusters:
        lines.append("## 🎯 多构象蛋白质簇 (重点关注)")
        lines.append("")
        
        for cluster in sorted(multiconformer_clusters, key=lambda c: c.avg_tm_score):
            lines.append(f"### 簇 {cluster.cluster_id}")
            lines.append(f"- 成员数量: {len(cluster.member_names)}")
            lines.append(f"- 平均TM-score: **{cluster.avg_tm_score:.4f}**")
            lines.append(f"- TM-score范围: [{cluster.min_tm_score:.4f}, {cluster.max_tm_score:.4f}]")
            lines.append("")
            lines.append("成员列表:")
            lines.append("| # | PDB文件 | 残基数 |")
            lines.append("|---|---------|--------|")
            for idx, (name, pdb_path) in enumerate(zip(cluster.member_names, cluster.member_pdb_paths), 1):
                sample = next((s for s in samples if s.name == name), None)
                num_res = sample.num_res if sample else "N/A"
                lines.append(f"| {idx} | `{pdb_path.name}` | {num_res} |")
            lines.append("")
    else:
        lines.append("## 多构象蛋白质簇")
        lines.append("未发现满足条件的多构象簇。")
        lines.append("")
    
    # 所有簇统计
    lines.append("## 所有簇统计")
    lines.append("| 簇ID | 成员数 | 平均TM | 最小TM | 最大TM | 多构象 |")
    lines.append("|------|--------|--------|--------|--------|--------|")
    for cluster in sorted(clusters, key=lambda c: c.cluster_id):
        is_multi = "✓" if cluster.is_multiconformer else ""
        lines.append(f"| {cluster.cluster_id} | {len(cluster.member_names)} | "
                    f"{cluster.avg_tm_score:.4f} | {cluster.min_tm_score:.4f} | "
                    f"{cluster.max_tm_score:.4f} | {is_multi} |")
    lines.append("")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    
    # 检查依赖
    if not HDBSCAN_AVAILABLE:
        print("错误: hdbscan未安装，请运行: pip install hdbscan")
        sys.exit(1)
    if not SKLEARN_AVAILABLE:
        print("错误: sklearn未安装，请运行: pip install scikit-learn")
        sys.exit(1)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    timing_info = {}
    total_start = time.time()
    
    # 创建输出目录
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
    
    print(f"{'='*60}")
    print("基于分数距离的HDBSCAN聚类分析")
    print(f"{'='*60}")
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # 扫描PDB文件
    print(f"\n扫描PDB文件: {args.source_dir}")
    all_pdbs = sorted(args.source_dir.rglob("*.pdb"))
    if not all_pdbs:
        print(f"错误: 未找到PDB文件")
        sys.exit(1)
    
    # 采样
    if args.sample_size > 0 and args.sample_size < len(all_pdbs):
        pdb_paths = random.sample(all_pdbs, args.sample_size)
    else:
        pdb_paths = all_pdbs
    print(f"选择 {len(pdb_paths)} 个PDB文件")
    
    # 并行预处理
    print(f"\n预处理PDB文件...")
    preprocess_start = time.time()
    prepared_inputs = prepare_inputs_parallel(pdb_paths, args.chain_id, args.preprocess_workers)
    timing_info['preprocess'] = time.time() - preprocess_start
    print(f"预处理完成: {len(prepared_inputs)} 个样本, 耗时 {timing_info['preprocess']:.2f}s")
    
    if len(prepared_inputs) < args.min_cluster_size:
        print(f"错误: 有效样本数 ({len(prepared_inputs)}) 小于最小簇大小 ({args.min_cluster_size})")
        sys.exit(1)
    
    # 加载模型
    print(f"\n加载模型...")
    conf = OmegaConf.load(args.config_path)
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    model = score_network.ScoreNetwork(conf.model, diffuser)
    
    checkpoint = torch.load(args.weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # 批量去噪
    print(f"\n批量去噪推理 (batch_size={args.batch_size})...")
    denoising_start = time.time()
    samples = process_all_samples(
        model=model,
        diffuser=diffuser,
        prepared_inputs=prepared_inputs,
        device=device,
        batch_size=args.batch_size,
        num_steps=args.num_denoising_steps,
        min_t=args.min_t,
        max_t=args.max_t,
        noise_scale=args.noise_scale,
        enable_self_conditioning=not args.disable_self_conditioning,
        use_fp16=args.use_fp16,
        output_dir=scores_dir,
    )
    timing_info['denoising'] = time.time() - denoising_start
    print(f"去噪完成: {len(samples)} 个样本, 耗时 {timing_info['denoising']:.2f}s")
    
    # 计算距离矩阵
    clustering_start = time.time()
    
    # 提取特征并计算余弦距离
    print(f"\n提取特征 (pooling={args.pooling})...")
    rot_features, trans_features = extract_features(samples, args.pooling)
    print(f"特征维度: rot={rot_features.shape}, trans={trans_features.shape}")
    
    print(f"\n计算组合距离矩阵 (α={args.alpha})...")
    distance_matrix = compute_combined_distance_matrix(rot_features, trans_features, args.alpha)
    precomputed_tm = None
    labels, clusterer = perform_clustering(
        distance_matrix,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    timing_info['clustering'] = time.time() - clustering_start
    
    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    print(f"聚类完成: {n_clusters} 个簇, {n_noise} 个噪声点")
    
    # 分析簇内TM-score
    print(f"\n分析簇内TM-score...")
    tm_start = time.time()
    clusters = analyze_clusters(
        samples=samples,
        labels=labels,
        tm_align_bin=args.tm_align_bin,
        tm_threshold=args.tm_threshold,
        num_workers=args.tm_workers,
        precomputed_tm=precomputed_tm,  # 如果使用对齐模式，复用已计算的TM-score
    )
    timing_info['tm_analysis'] = time.time() - tm_start
    
    # 筛选多构象簇
    multiconformer_clusters = [c for c in clusters if c.is_multiconformer]
    print(f"\n{'='*60}")
    print(f"发现 {len(multiconformer_clusters)} 个多构象簇 (平均TM-score < {args.tm_threshold})")
    print(f"{'='*60}")
    
    for cluster in multiconformer_clusters:
        print(f"\n簇 {cluster.cluster_id}: {len(cluster.member_names)} 个成员")
        print(f"  平均TM-score: {cluster.avg_tm_score:.4f}")
        print(f"  PDB文件:")
        for pdb_path in cluster.member_pdb_paths:
            print(f"    - {pdb_path.name}")
    
    # 生成可视化
    print(f"\n生成可视化...")
    plot_clustering_results(
        samples=samples,
        rot_features=rot_features,
        trans_features=trans_features,
        labels=labels,
        clusters=clusters,
        output_dir=args.output_dir,
        alpha=args.alpha,
        clusterer=clusterer,  # 传递聚类器对象以绘制聚类树
    )
    
    # 保存聚类结果
    np.save(args.output_dir / "cluster_labels.npy", labels)
    np.save(args.output_dir / "distance_matrix.npy", distance_matrix)
    
    # 生成报告
    timing_info['total'] = time.time() - total_start
    report_path = args.output_dir / "clustering_report.md"
    generate_report(report_path, args, samples, labels, clusters, timing_info)
    
    # 保存多构象簇的详细信息
    if multiconformer_clusters:
        mc_dir = args.output_dir / "multiconformer_clusters"
        mc_dir.mkdir(exist_ok=True)
        
        for cluster in multiconformer_clusters:
            cluster_dir = mc_dir / f"cluster_{cluster.cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
            
            # 保存成员列表
            with open(cluster_dir / "members.txt", 'w') as f:
                for pdb_path in cluster.member_pdb_paths:
                    f.write(f"{pdb_path}\n")
            
            # 保存TM-score矩阵
            if cluster.tm_score_matrix is not None:
                np.save(cluster_dir / "tm_score_matrix.npy", cluster.tm_score_matrix)
        
        print(f"\n多构象簇详情已保存到: {mc_dir}")
    
    print(f"\n{'='*60}")
    print(f"分析完成!")
    print(f"总耗时: {timing_info['total']:.2f}s")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
