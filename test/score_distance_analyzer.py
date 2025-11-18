#!/usr/bin/env python3
"""
Score距离分析器

功能:
1. 计算rot_score之间的欧氏距离和余弦距离
2. 计算trans_score之间的欧氏距离和余弦距离
3. 进行数据分析

"""

import os
import re
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use English-friendly font

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 输入参数
INPUT_DIR = str(PROJECT_ROOT / 'test' / 'output_dir_direct_denoising')
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'score_analysis_output')
PDB_DIR = str(PROJECT_ROOT / 'test' / 'pdb_dir')
TMALIGN_BIN = os.environ.get('TMALIGN_BIN', 'TM-align')
REFERENCE_PREFIX = os.environ.get('REFERENCE_PREFIX')


def extract_structure_prefix(filename):
    """Infer structure prefix from score filename."""
    base = Path(filename).name
    if base.endswith('.npy'):
        base = base[:-4]
    for suffix in ('_rot_score', '_trans_score'):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    raise ValueError(f"无法从文件名 {filename} 中解析前缀")


def candidate_structure_basenames(prefix):
    """Generate ordered candidate basenames for locating the corresponding PDB."""
    candidates = [prefix]
    if '_step' in prefix:
        candidates.append(prefix.split('_step')[0])
    if '_denoised' in prefix:
        candidates.append(prefix.split('_denoised')[0])
    if '_' in prefix:
        candidates.append(prefix.rsplit('_', 1)[0])
    candidates.append(prefix.split('_')[0])
    seen = set()
    ordered = []
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            ordered.append(cand)
    return ordered


def resolve_structure_path(prefix, search_dirs):
    """Locate a PDB file that matches the provided prefix."""
    candidate_bases = candidate_structure_basenames(prefix)
    for directory in search_dirs:
        if not directory.exists():
            continue
        for candidate in candidate_bases:
            exact = directory / f"{candidate}.pdb"
            if exact.exists():
                return str(exact)
        for candidate in candidate_bases:
            matches = sorted(directory.glob(f"{candidate}*.pdb"))
            if matches:
                return str(matches[0])
    raise FileNotFoundError(f"在 {', '.join(str(d) for d in search_dirs)} 中找不到 {prefix} 对应的PDB文件")


def locate_tmalign_binary(binary_hint):
    """Resolve TM-align executable path."""
    candidates = [binary_hint, TMALIGN_BIN, 'TM-align']
    for cand in candidates:
        if not cand:
            continue
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
        resolved = shutil.which(cand)
        if resolved:
            return resolved
    raise FileNotFoundError("未找到 TM-align 可执行文件，请设置 TMALIGN_BIN 环境变量或将其加入PATH")


def parse_tmalign_transform(stdout):
    """Parse rotation matrix and translation vector from TM-align output."""
    rotation = np.zeros((3, 3), dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    rot_entries = {}
    trans_entries = {}
    rot_pattern = re.compile(r"m\((\d),(\d)\)=\s*([-+Ee0-9\.]+)")
    trans_pattern = re.compile(r"t\((\d)\)=\s*([-+Ee0-9\.]+)")
    for line in stdout.splitlines():
        for i_str, j_str, value in rot_pattern.findall(line):
            rot_entries[(int(i_str) - 1, int(j_str) - 1)] = float(value)
        for idx_str, value in trans_pattern.findall(line):
            trans_entries[int(idx_str) - 1] = float(value)
    if len(rot_entries) != 9 or len(trans_entries) != 3:
        # 尝试解析新版TM-align在 -m 输出中的矩阵表格格式
        table_pattern = re.compile(r"^(\d+)\s+([-+Ee0-9\.]+)\s+([-+Ee0-9\.]+)\s+([-+Ee0-9\.]+)\s+([-+Ee0-9\.]+)")
        table_rotation = np.zeros((3, 3), dtype=np.float64)
        table_translation = np.zeros(3, dtype=np.float64)
        table_counts = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line or line.startswith('-') or line.startswith('m '):
                continue
            match = table_pattern.match(line)
            if not match:
                continue
            idx = int(match.group(1))
            if idx < 0 or idx > 2:
                continue
            table_translation[idx] = float(match.group(2))
            table_rotation[idx, 0] = float(match.group(3))
            table_rotation[idx, 1] = float(match.group(4))
            table_rotation[idx, 2] = float(match.group(5))
            table_counts += 1
        if table_counts == 3:
            rotation = table_rotation
            translation = table_translation
            rot_entries = {(i, j): rotation[i, j] for i in range(3) for j in range(3)}
            trans_entries = {i: translation[i] for i in range(3)}
        else:
            snippet = "\n".join(stdout.splitlines()[:40])
            raise ValueError("无法从 TM-align 输出中解析旋转矩阵\n" + snippet)

    if len(trans_entries) != 3:
        snippet = "\n".join(stdout.splitlines()[:40])
        raise ValueError("无法从 TM-align 输出中解析平移向量\n" + snippet)
    for (i, j), value in rot_entries.items():
        rotation[i, j] = value
    for idx, value in trans_entries.items():
        translation[idx] = value
    return rotation, translation


def run_tmalign_alignment(tmalign_bin, reference_pdb, target_pdb):
    """Execute TM-align and return rigid transform aligning target to reference."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='_tmalign.txt') as tmp_file:
        tmp_path = tmp_file.name
    try:
        try:
            result = subprocess.run(
                [tmalign_bin, reference_pdb, target_pdb, '-m', tmp_path],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"TM-align 执行失败: {exc.stderr or exc.stdout}") from exc

        if os.path.exists(tmp_path):
            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as fh:
                transform_text = fh.read()
        else:
            transform_text = ''

        if not transform_text.strip():
            transform_text = result.stdout

        return parse_tmalign_transform(transform_text)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def align_structures_for_scores(scores, search_dirs, tmalign_bin=None, reference_prefix=None):
    """Align structures for all prefixes and return rigid transforms."""
    prefixes = sorted({entry['prefix'] for entries in scores.values() for entry in entries})
    if not prefixes:
        return {}
    ref_prefix = reference_prefix or REFERENCE_PREFIX or prefixes[0]
    if ref_prefix not in prefixes:
        raise ValueError(f"参考前缀 {ref_prefix} 不在可用前缀列表 {prefixes} 中")
    directories = [Path(d) for d in search_dirs if d and Path(d).exists()]
    if not directories:
        raise FileNotFoundError("未找到任何可用的结构目录用于对齐")
    structure_paths = {prefix: resolve_structure_path(prefix, directories) for prefix in prefixes}
    transforms = {}
    transforms[ref_prefix] = {
        'rotation': np.eye(3, dtype=np.float64),
        'translation': np.zeros(3, dtype=np.float64),
        'pdb_path': structure_paths[ref_prefix],
    }
    if len(prefixes) == 1:
        print(f"仅检测到一个前缀 {ref_prefix}，使用其自身坐标系")
        return transforms
    tmalign_exec = locate_tmalign_binary(tmalign_bin or TMALIGN_BIN)
    print(f"参考结构: {ref_prefix} -> {structure_paths[ref_prefix]}")
    for prefix in prefixes:
        if prefix == ref_prefix:
            continue
        print(f"使用 TM-align 将 {prefix} 对齐至 {ref_prefix}...")
        rotation, translation = run_tmalign_alignment(tmalign_exec, structure_paths[ref_prefix], structure_paths[prefix])
        transforms[prefix] = {
            'rotation': rotation,
            'translation': translation,
            'pdb_path': structure_paths[prefix],
        }
    return transforms


def apply_rigid_transform(score_array, rotation, translation):
    """Apply rigid transform (rotation + translation) to score array."""
    data = np.asarray(score_array, dtype=np.float64)
    if data.ndim < 2 or data.shape[-1] != 3:
        raise ValueError("score 数组最后一个维度必须为3以应用刚体变换")
    original_shape = data.shape
    flattened = data.reshape(-1, 3)
    transformed = flattened @ rotation.T + translation.reshape(1, 3)
    transformed = transformed.reshape(original_shape)
    return transformed.astype(score_array.dtype)


def transform_scores_with_alignment(scores, transforms):
    """Apply precomputed rigid transforms to every score entry."""
    if not transforms:
        return
    for score_type, entries in scores.items():
        for entry in entries:
            prefix = entry['prefix']
            if prefix not in transforms:
                raise KeyError(f"未找到 {prefix} 的刚体变换")
            transform = transforms[prefix]
            entry['data'] = apply_rigid_transform(entry['data'], transform['rotation'], transform['translation'])
    print("所有score已根据对齐结果完成刚体变换")


def load_score_pairs(input_dir, prefix=None):
    """
    加载成对的score文件
    
    Args:
        input_dir: 包含.npy文件的目录
        prefix: 文件名前缀（如果为None，则加载所有文件）
    
    Returns:
        dict: 包含rot和trans score对的字典
    """
    scores = {}
    
    # 查找所有匹配的文件（不限制前缀）
    if prefix is None:
        rot_files = sorted([f for f in os.listdir(input_dir) if 'rot_score' in f and f.endswith('.npy')])
        trans_files = sorted([f for f in os.listdir(input_dir) if 'trans_score' in f and f.endswith('.npy')])
    else:
        rot_files = sorted([f for f in os.listdir(input_dir) if f.startswith(prefix) and 'rot_score' in f and f.endswith('.npy')])
        trans_files = sorted([f for f in os.listdir(input_dir) if f.startswith(prefix) and 'trans_score' in f and f.endswith('.npy')])
    
    print(f"找到 {len(rot_files)} 个rot_score文件")
    print(f"找到 {len(trans_files)} 个trans_score文件")
    
    # 加载rot_score
    rot_scores = []
    for f in rot_files:
        path = os.path.join(input_dir, f)
        score = np.load(path)
        prefix = extract_structure_prefix(f)
        rot_scores.append({'filename': f, 'data': score, 'prefix': prefix})
        print(f"  加载: {f}, shape: {score.shape}")
    
    # 加载trans_score
    trans_scores = []
    for f in trans_files:
        path = os.path.join(input_dir, f)
        score = np.load(path)
        prefix = extract_structure_prefix(f)
        trans_scores.append({'filename': f, 'data': score, 'prefix': prefix})
        print(f"  加载: {f}, shape: {score.shape}")
    
    scores['rot'] = rot_scores
    scores['trans'] = trans_scores
    
    return scores


def compute_euclidean_distance(score1, score2):
    """
    计算两个score之间的欧氏距离
    
    Args:
        score1, score2: numpy数组，形状为 (1, num_res, 3)
    
    Returns:
        float: 总体欧氏距离
        np.array: 每个残基的欧氏距离
    """
    # 去掉batch维度
    if score1.ndim == 3:
        score1 = score1.squeeze(0)
    if score2.ndim == 3:
        score2 = score2.squeeze(0)
    
    # 计算每个残基的欧氏距离
    per_residue_distances = np.linalg.norm(score1 - score2, axis=1)
    
    # 计算总体距离（展平后）
    overall_distance = np.linalg.norm(score1.flatten() - score2.flatten())
    
    return overall_distance, per_residue_distances


def compute_cosine_similarity(score1, score2):
    """
    计算两个score之间的余弦相似度和余弦距离
    
    Args:
        score1, score2: numpy数组，形状为 (1, num_res, 3)
    
    Returns:
        float: 总体余弦距离
        np.array: 每个残基的余弦距离
        float: 总体余弦相似度
    """
    # 去掉batch维度
    if score1.ndim == 3:
        score1 = score1.squeeze(0)
    if score2.ndim == 3:
        score2 = score2.squeeze(0)
    
    # 计算每个残基的余弦距离
    per_residue_distances = []
    for i in range(score1.shape[0]):
        vec1 = score1[i]
        vec2 = score2[i]
        # 处理零向量
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            per_residue_distances.append(1.0)  # 最大距离
        else:
            cos_dist = cosine(vec1, vec2)
            per_residue_distances.append(cos_dist)
    
    per_residue_distances = np.array(per_residue_distances)
    
    # 计算总体余弦距离（展平后）
    vec1_flat = score1.flatten()
    vec2_flat = score2.flatten()
    
    if np.linalg.norm(vec1_flat) == 0 or np.linalg.norm(vec2_flat) == 0:
        overall_cosine_distance = 1.0
        overall_cosine_similarity = 0.0
    else:
        overall_cosine_distance = cosine(vec1_flat, vec2_flat)
        overall_cosine_similarity = 1 - overall_cosine_distance
    
    return overall_cosine_distance, per_residue_distances, overall_cosine_similarity


def analyze_score_pairs(scores, score_type='rot'):
    """
    分析score对之间的距离
    
    Args:
        scores: score数据列表
        score_type: 'rot' 或 'trans'
    
    Returns:
        dict: 分析结果
    """
    results = {
        'pairs': [],
        'euclidean_distances': [],
        'cosine_distances': [],
        'cosine_similarities': [],
        'per_residue_euclidean': [],
        'per_residue_cosine': [],
    }
    
    # 计算所有成对距离
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            score1 = scores[i]['data']
            score2 = scores[j]['data']
            name1 = scores[i]['filename']
            name2 = scores[j]['filename']
            
            pair_name = f"{name1} vs {name2}"
            
            # 计算欧氏距离
            euc_dist, per_res_euc = compute_euclidean_distance(score1, score2)
            
            # 计算余弦距离和相似度
            cos_dist, per_res_cos, cos_sim = compute_cosine_similarity(score1, score2)
            
            results['pairs'].append(pair_name)
            results['euclidean_distances'].append(euc_dist)
            results['cosine_distances'].append(cos_dist)
            results['cosine_similarities'].append(cos_sim)
            results['per_residue_euclidean'].append(per_res_euc)
            results['per_residue_cosine'].append(per_res_cos)
            
            print(f"\n{score_type.upper()} Score: {pair_name}")
            print(f"  欧氏距离: {euc_dist:.6f}")
            print(f"  余弦距离: {cos_dist:.6f}")
            print(f"  余弦相似度: {cos_sim:.6f}")
            print(f"  每残基欧氏距离 - 均值: {np.mean(per_res_euc):.6f}, 标准差: {np.std(per_res_euc):.6f}")
            print(f"  每残基余弦距离 - 均值: {np.mean(per_res_cos):.6f}, 标准差: {np.std(per_res_cos):.6f}")
    
    return results


def compute_correlation_statistics(score1, score2):
    """
    计算两个score之间的相关统计量
    
    Args:
        score1, score2: numpy数组
    
    Returns:
        dict: 包含各种统计量的字典
    """
    # 去掉batch维度并展平
    if score1.ndim == 3:
        score1 = score1.squeeze(0)
    if score2.ndim == 3:
        score2 = score2.squeeze(0)
    
    # 按维度计算相关性
    stats = {}
    for dim in range(3):
        vec1 = score1[:, dim]
        vec2 = score2[:, dim]
        
        pearson_r, pearson_p = pearsonr(vec1, vec2)
        spearman_r, spearman_p = spearmanr(vec1, vec2)
        
        stats[f'dim_{dim}'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mean_diff': np.mean(vec1 - vec2),
            'std_diff': np.std(vec1 - vec2),
            'max_diff': np.max(np.abs(vec1 - vec2)),
        }
    
    return stats


def save_detailed_report(rot_results, trans_results, scores, output_dir):
    """
    保存详细分析报告
    """
    report_path = os.path.join(output_dir, 'distance_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Score距离分析详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        # ROT Score分析
        f.write("【旋转分数 (Rotation Score) 分析】\n")
        f.write("-" * 80 + "\n")
        for i, pair in enumerate(rot_results['pairs']):
            f.write(f"\n配对 {i+1}: {pair}\n")
            f.write(f"  欧氏距离: {rot_results['euclidean_distances'][i]:.6f}\n")
            f.write(f"  余弦距离: {rot_results['cosine_distances'][i]:.6f}\n")
            f.write(f"  余弦相似度: {rot_results['cosine_similarities'][i]:.6f}\n")
            
            per_euc = rot_results['per_residue_euclidean'][i]
            per_cos = rot_results['per_residue_cosine'][i]
            
            f.write(f"  每残基欧氏距离统计:\n")
            f.write(f"    均值: {np.mean(per_euc):.6f}\n")
            f.write(f"    中位数: {np.median(per_euc):.6f}\n")
            f.write(f"    标准差: {np.std(per_euc):.6f}\n")
            f.write(f"    最小值: {np.min(per_euc):.6f}\n")
            f.write(f"    最大值: {np.max(per_euc):.6f}\n")
            
            f.write(f"  每残基余弦距离统计:\n")
            f.write(f"    均值: {np.mean(per_cos):.6f}\n")
            f.write(f"    中位数: {np.median(per_cos):.6f}\n")
            f.write(f"    标准差: {np.std(per_cos):.6f}\n")
            f.write(f"    最小值: {np.min(per_cos):.6f}\n")
            f.write(f"    最大值: {np.max(per_cos):.6f}\n")
        
        # TRANS Score分析
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("【平移分数 (Translation Score) 分析】\n")
        f.write("-" * 80 + "\n")
        for i, pair in enumerate(trans_results['pairs']):
            f.write(f"\n配对 {i+1}: {pair}\n")
            f.write(f"  欧氏距离: {trans_results['euclidean_distances'][i]:.6f}\n")
            f.write(f"  余弦距离: {trans_results['cosine_distances'][i]:.6f}\n")
            f.write(f"  余弦相似度: {trans_results['cosine_similarities'][i]:.6f}\n")
            
            per_euc = trans_results['per_residue_euclidean'][i]
            per_cos = trans_results['per_residue_cosine'][i]
            
            f.write(f"  每残基欧氏距离统计:\n")
            f.write(f"    均值: {np.mean(per_euc):.6f}\n")
            f.write(f"    中位数: {np.median(per_euc):.6f}\n")
            f.write(f"    标准差: {np.std(per_euc):.6f}\n")
            f.write(f"    最小值: {np.min(per_euc):.6f}\n")
            f.write(f"    最大值: {np.max(per_euc):.6f}\n")
            
            f.write(f"  每残基余弦距离统计:\n")
            f.write(f"    均值: {np.mean(per_cos):.6f}\n")
            f.write(f"    中位数: {np.median(per_cos):.6f}\n")
            f.write(f"    标准差: {np.std(per_cos):.6f}\n")
            f.write(f"    最小值: {np.min(per_cos):.6f}\n")
            f.write(f"    最大值: {np.max(per_cos):.6f}\n")
        
        # 相关性分析
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("【相关性分析】\n")
        f.write("-" * 80 + "\n")
        
        if len(scores['rot']) >= 2:
            f.write("\nROT Score 相关性:\n")
            for i in range(len(scores['rot'])):
                for j in range(i + 1, len(scores['rot'])):
                    stats = compute_correlation_statistics(
                        scores['rot'][i]['data'],
                        scores['rot'][j]['data']
                    )
                    f.write(f"\n  {scores['rot'][i]['filename']} vs {scores['rot'][j]['filename']}:\n")
                    for dim in range(3):
                        f.write(f"    维度 {dim}:\n")
                        f.write(f"      Pearson相关系数: {stats[f'dim_{dim}']['pearson_r']:.6f} (p={stats[f'dim_{dim}']['pearson_p']:.6e})\n")
                        f.write(f"      Spearman相关系数: {stats[f'dim_{dim}']['spearman_r']:.6f} (p={stats[f'dim_{dim}']['spearman_p']:.6e})\n")
                        f.write(f"      平均差异: {stats[f'dim_{dim}']['mean_diff']:.6f}\n")
                        f.write(f"      标准差: {stats[f'dim_{dim}']['std_diff']:.6f}\n")
                        f.write(f"      最大差异: {stats[f'dim_{dim}']['max_diff']:.6f}\n")
        
        if len(scores['trans']) >= 2:
            f.write("\nTRANS Score 相关性:\n")
            for i in range(len(scores['trans'])):
                for j in range(i + 1, len(scores['trans'])):
                    stats = compute_correlation_statistics(
                        scores['trans'][i]['data'],
                        scores['trans'][j]['data']
                    )
                    f.write(f"\n  {scores['trans'][i]['filename']} vs {scores['trans'][j]['filename']}:\n")
                    for dim in range(3):
                        f.write(f"    维度 {dim}:\n")
                        f.write(f"      Pearson相关系数: {stats[f'dim_{dim}']['pearson_r']:.6f} (p={stats[f'dim_{dim}']['pearson_p']:.6e})\n")
                        f.write(f"      Spearman相关系数: {stats[f'dim_{dim}']['spearman_r']:.6f} (p={stats[f'dim_{dim}']['spearman_p']:.6e})\n")
                        f.write(f"      平均差异: {stats[f'dim_{dim}']['mean_diff']:.6f}\n")
                        f.write(f"      标准差: {stats[f'dim_{dim}']['std_diff']:.6f}\n")
                        f.write(f"      最大差异: {stats[f'dim_{dim}']['max_diff']:.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("分析完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n详细报告已保存: {report_path}")


def plot_distance_matrix(results, score_type, output_dir):
    """
    Plot pairwise distance matrix heatmap
    """
    n_pairs = len(results['pairs'])
    
    if n_pairs == 0:
        print("Not enough data to plot distance matrix")
        return
    
    # Create distance matrices (symmetric)
    n_files = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2) + 1
    
    # For simplicity, create a bar plot showing all pairwise distances
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{score_type.upper()} Score Distance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Euclidean distances bar plot
    ax = axes[0, 0]
    x = np.arange(len(results['pairs']))
    ax.bar(x, results['euclidean_distances'], alpha=0.8, color='steelblue')
    ax.set_xlabel('Score Pairs', fontsize=12)
    ax.set_ylabel('Euclidean Distance', fontsize=12)
    ax.set_title('Pairwise Euclidean Distances', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Pair {i+1}' for i in range(len(results['pairs']))], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(results['euclidean_distances']):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Cosine distances bar plot
    ax = axes[0, 1]
    ax.bar(x, results['cosine_distances'], alpha=0.8, color='coral')
    ax.set_xlabel('Score Pairs', fontsize=12)
    ax.set_ylabel('Cosine Distance', fontsize=12)
    ax.set_title('Pairwise Cosine Distances', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Pair {i+1}' for i in range(len(results['pairs']))], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(results['cosine_distances']):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Cosine similarities bar plot
    ax = axes[1, 0]
    ax.bar(x, results['cosine_similarities'], alpha=0.8, color='mediumseagreen')
    ax.set_xlabel('Score Pairs', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Pairwise Cosine Similarities', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Pair {i+1}' for i in range(len(results['pairs']))], rotation=45, ha='right')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(results['cosine_similarities']):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Statistics comparison
    ax = axes[1, 1]
    stats_data = {
        'Mean Euc': [np.mean(per_res) for per_res in results['per_residue_euclidean']],
        'Std Euc': [np.std(per_res) for per_res in results['per_residue_euclidean']],
        'Mean Cos': [np.mean(per_res) for per_res in results['per_residue_cosine']],
        'Std Cos': [np.std(per_res) for per_res in results['per_residue_cosine']],
    }
    
    x_stat = np.arange(len(results['pairs']))
    width = 0.2
    
    ax.bar(x_stat - 1.5*width, stats_data['Mean Euc'], width, label='Mean Euclidean', alpha=0.8)
    ax.bar(x_stat - 0.5*width, stats_data['Std Euc'], width, label='Std Euclidean', alpha=0.8)
    ax.bar(x_stat + 0.5*width, stats_data['Mean Cos'], width, label='Mean Cosine', alpha=0.8)
    ax.bar(x_stat + 1.5*width, stats_data['Std Cos'], width, label='Std Cosine', alpha=0.8)
    
    ax.set_xlabel('Score Pairs', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Per-Residue Statistics Comparison', fontsize=13)
    ax.set_xticks(x_stat)
    ax.set_xticklabels([f'Pair {i+1}' for i in range(len(results['pairs']))], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{score_type}_distance_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distance analysis plot saved: {output_path}")
    plt.close()


def plot_per_residue_distances(results, score_type, output_dir):
    """
    Plot per-residue distance distributions
    """
    n_pairs = len(results['pairs'])
    
    if n_pairs == 0:
        print("Not enough data to plot per-residue distances")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{score_type.upper()} Score Per-Residue Distance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Euclidean distance curves
    ax = axes[0, 0]
    for i, per_res_dist in enumerate(results['per_residue_euclidean']):
        ax.plot(per_res_dist, alpha=0.7, linewidth=2, label=f'Pair {i+1}')
    ax.set_xlabel('Residue Index', fontsize=12)
    ax.set_ylabel('Euclidean Distance', fontsize=12)
    ax.set_title('Per-Residue Euclidean Distances', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Cosine distance curves
    ax = axes[0, 1]
    for i, per_res_dist in enumerate(results['per_residue_cosine']):
        ax.plot(per_res_dist, alpha=0.7, linewidth=2, label=f'Pair {i+1}')
    ax.set_xlabel('Residue Index', fontsize=12)
    ax.set_ylabel('Cosine Distance', fontsize=12)
    ax.set_title('Per-Residue Cosine Distances', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Euclidean distance box plots
    ax = axes[1, 0]
    data_euc = [per_res for per_res in results['per_residue_euclidean']]
    if data_euc:
        bp = ax.boxplot(data_euc, labels=[f'Pair {i+1}' for i in range(len(data_euc))],
                        patch_artist=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Score Pairs', fontsize=12)
        ax.set_ylabel('Euclidean Distance', fontsize=12)
        ax.set_title('Euclidean Distance Distribution (Box Plot)', fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Cosine distance box plots
    ax = axes[1, 1]
    data_cos = [per_res for per_res in results['per_residue_cosine']]
    if data_cos:
        bp = ax.boxplot(data_cos, labels=[f'Pair {i+1}' for i in range(len(data_cos))],
                        patch_artist=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax.set_xlabel('Score Pairs', fontsize=12)
        ax.set_ylabel('Cosine Distance', fontsize=12)
        ax.set_title('Cosine Distance Distribution (Box Plot)', fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{score_type}_per_residue_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-residue analysis plot saved: {output_path}")
    plt.close()


def plot_distance_heatmap(results, score_type, output_dir):
    """
    Plot distance heatmap for better visualization of all pairwise comparisons
    """
    n_pairs = len(results['pairs'])
    
    if n_pairs == 0:
        print("Not enough data to plot heatmap")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{score_type.upper()} Score Distance Heatmap', fontsize=16, fontweight='bold')
    
    # Prepare data for heatmap
    euc_data = np.array(results['euclidean_distances']).reshape(1, -1)
    cos_data = np.array(results['cosine_distances']).reshape(1, -1)
    sim_data = np.array(results['cosine_similarities']).reshape(1, -1)
    
    # 1. Euclidean distance heatmap
    ax = axes[0]
    sns.heatmap(euc_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, 
                xticklabels=[f'Pair {i+1}' for i in range(n_pairs)],
                yticklabels=['Euclidean'], cbar_kws={'label': 'Distance'})
    ax.set_title('Euclidean Distances', fontsize=13)
    
    # 2. Cosine distance heatmap
    ax = axes[1]
    sns.heatmap(cos_data, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax,
                xticklabels=[f'Pair {i+1}' for i in range(n_pairs)],
                yticklabels=['Cosine'], cbar_kws={'label': 'Distance'})
    ax.set_title('Cosine Distances', fontsize=13)
    
    # 3. Cosine similarity heatmap
    ax = axes[2]
    sns.heatmap(sim_data, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
                xticklabels=[f'Pair {i+1}' for i in range(n_pairs)],
                yticklabels=['Similarity'], cbar_kws={'label': 'Similarity'})
    ax.set_title('Cosine Similarities', fontsize=13)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{score_type}_distance_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distance heatmap saved: {output_path}")
    plt.close()


def plot_correlation_analysis(results, score_type, output_dir):
    """
    Plot correlation analysis between different pairs
    """
    n_pairs = len(results['pairs'])
    
    if n_pairs == 0:
        print("Not enough data for correlation analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{score_type.upper()} Score Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Euclidean vs Cosine distance scatter
    ax = axes[0, 0]
    ax.scatter(results['euclidean_distances'], results['cosine_distances'], 
               s=100, alpha=0.6, c=range(n_pairs), cmap='viridis')
    ax.set_xlabel('Euclidean Distance', fontsize=12)
    ax.set_ylabel('Cosine Distance', fontsize=12)
    ax.set_title('Euclidean vs Cosine Distance', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if n_pairs > 1:
        corr, _ = pearsonr(results['euclidean_distances'], results['cosine_distances'])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Cosine distance vs similarity
    ax = axes[0, 1]
    ax.scatter(results['cosine_distances'], results['cosine_similarities'], 
               s=100, alpha=0.6, c=range(n_pairs), cmap='viridis')
    ax.set_xlabel('Cosine Distance', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Cosine Distance vs Similarity', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # 3. Mean per-residue distances comparison
    ax = axes[1, 0]
    mean_euc = [np.mean(per_res) for per_res in results['per_residue_euclidean']]
    mean_cos = [np.mean(per_res) for per_res in results['per_residue_cosine']]
    ax.scatter(mean_euc, mean_cos, s=100, alpha=0.6, c=range(n_pairs), cmap='viridis')
    ax.set_xlabel('Mean Per-Residue Euclidean Distance', fontsize=12)
    ax.set_ylabel('Mean Per-Residue Cosine Distance', fontsize=12)
    ax.set_title('Mean Per-Residue Distance Comparison', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    if n_pairs > 1:
        corr, _ = pearsonr(mean_euc, mean_cos)
        ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Histogram of all distances
    ax = axes[1, 1]
    all_euc = np.concatenate(results['per_residue_euclidean'])
    all_cos = np.concatenate(results['per_residue_cosine'])
    
    ax.hist(all_euc, bins=50, alpha=0.5, label='Euclidean', density=True, color='steelblue')
    ax.hist(all_cos, bins=50, alpha=0.5, label='Cosine', density=True, color='coral')
    ax.set_xlabel('Distance Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Overall Distance Distribution', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{score_type}_correlation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation analysis plot saved: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Score距离分析器")
    print("=" * 80)
    print("功能:")
    print("  1. 对目录下所有rot_score文件进行两两距离计算")
    print("  2. 对目录下所有trans_score文件进行两两距离计算")
    print("  3. 计算欧氏距离和余弦距离/相似度")
    print("  4. 生成详细的文本报告和可视化图表")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # 加载score文件（不限制前缀，加载所有文件）
    print(f"\n正在加载score文件...")
    print(f"输入目录: {INPUT_DIR}")
    scores = load_score_pairs(INPUT_DIR, prefix=None)

    # 进行结构对齐并对score应用相同的刚体变换
    structure_dirs = []
    if PDB_DIR:
        structure_dirs.append(Path(PDB_DIR))
    structure_dirs.append(Path(INPUT_DIR))
    print("\n准备对齐对应的蛋白质结构，并同步变换score...")
    transforms = align_structures_for_scores(scores, structure_dirs)
    transform_scores_with_alignment(scores, transforms)
    
    if len(scores['rot']) < 2 or len(scores['trans']) < 2:
        print("\n警告: 需要至少2个rot_score文件和2个trans_score文件进行分析")
        return
    
    # 分析ROT Score
    print(f"\n{'='*80}")
    print("分析 ROT Score...")
    print(f"{'='*80}")
    rot_results = analyze_score_pairs(scores['rot'], 'rot')
    
    # 分析TRANS Score
    print(f"\n{'='*80}")
    print("分析 TRANS Score...")
    print(f"{'='*80}")
    trans_results = analyze_score_pairs(scores['trans'], 'trans')
    
    # 生成可视化图表
    print(f"\n{'='*80}")
    print("生成可视化图表...")
    print(f"{'='*80}")
    
    print("\n生成 ROT Score 可视化...")
    plot_distance_matrix(rot_results, 'rot', OUTPUT_DIR)
    plot_per_residue_distances(rot_results, 'rot', OUTPUT_DIR)
    plot_distance_heatmap(rot_results, 'rot', OUTPUT_DIR)
    plot_correlation_analysis(rot_results, 'rot', OUTPUT_DIR)
    
    print("\n生成 TRANS Score 可视化...")
    plot_distance_matrix(trans_results, 'trans', OUTPUT_DIR)
    plot_per_residue_distances(trans_results, 'trans', OUTPUT_DIR)
    plot_distance_heatmap(trans_results, 'trans', OUTPUT_DIR)
    plot_correlation_analysis(trans_results, 'trans', OUTPUT_DIR)
    
    # 保存报告
    print(f"\n{'='*80}")
    print("保存分析报告...")
    print(f"{'='*80}")
    
    save_detailed_report(rot_results, trans_results, scores, OUTPUT_DIR)
    
    # 统计生成的文件
    png_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    txt_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')]
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📋 文本报告: {len(txt_files)} 个")
    print(f"📊 可视化图表: {len(png_files)} 个")
    print(f"\n生成的图表:")
    for score_type in ['rot', 'trans']:
        print(f"  - {score_type}_distance_analysis.png")
        print(f"  - {score_type}_per_residue_analysis.png")
        print(f"  - {score_type}_distance_heatmap.png")
        print(f"  - {score_type}_correlation_analysis.png")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
