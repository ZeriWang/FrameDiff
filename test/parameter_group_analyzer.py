#!/usr/bin/env python3
"""
参数组分析器

功能:
1. 对同一NUM_STEPS下不同MAX_T的数据进行分析
2. 对同一MAX_T下不同NUM_STEPS的数据进行分析
3. 计算PDB文件之间的RMSD
4. 分析旋转分数(rot_score)之间的欧氏距离和余弦相似度
5. 分析平移分数(trans_score)之间的欧氏距离和余弦相似度
6. 生成详细的分析报告和可视化图表

基于: parameter_sweep_analyzer.py 和 score_distance_analyzer.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
import json
from collections import defaultdict

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from openfold.np import protein

# ==================== 配置参数 ====================
INPUT_DIR = str(PROJECT_ROOT / 'test' / 'parameter_sweep_results' / 'run_20251112_173439')
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'parameter_group_analysis_results')
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'analysis_{TIMESTAMP}')

# ==================== 辅助函数 ====================

def load_pdb(pdb_path):
    """加载PDB文件"""
    try:
        with open(pdb_path, 'r') as f:
            pdb_string = f.read()
        return protein.from_pdb_string(pdb_string)
    except Exception as e:
        print(f"警告: 无法加载PDB文件 {pdb_path}: {e}")
        return None


def compute_rmsd(prot1, prot2):
    """
    计算两个蛋白质结构之间的RMSD
    
    Args:
        prot1, prot2: protein.Protein对象
    
    Returns:
        float: RMSD值 (Angstroms)
    """
    if prot1 is None or prot2 is None:
        return None
    
    # 获取CA原子位置 (索引1对应CA)
    pos1 = prot1.atom_positions[:, 1, :]  # [N, 3]
    pos2 = prot2.atom_positions[:, 1, :]  # [N, 3]
    
    # 获取mask
    mask1 = prot1.atom_mask[:, 1]  # [N]
    mask2 = prot2.atom_mask[:, 1]  # [N]
    
    # 确保两个结构的残基数相同
    if pos1.shape[0] != pos2.shape[0]:
        print(f"警告: 残基数不匹配 {pos1.shape[0]} vs {pos2.shape[0]}")
        return None
    
    # 组合mask
    common_mask = mask1 * mask2
    
    if np.sum(common_mask) == 0:
        print("警告: 没有共同的有效CA原子")
        return None
    
    # 计算RMSD
    pos1_valid = pos1[common_mask > 0]
    pos2_valid = pos2[common_mask > 0]
    
    diff = pos1_valid - pos2_valid
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def compute_euclidean_distance(score1, score2):
    """
    计算两个score之间的欧氏距离
    
    Args:
        score1, score2: numpy数组，形状为 (1, num_res, 3) 或 (num_res, 3)
    
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
        score1, score2: numpy数组，形状为 (1, num_res, 3) 或 (num_res, 3)
    
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
            per_residue_distances.append(1.0)
        else:
            per_residue_distances.append(cosine(vec1, vec2))
    
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


def load_experiment_data(input_dir):
    """
    加载所有实验数据
    
    Returns:
        dict: 按照 (num_steps, max_t) 为键的字典
    """
    data = {}
    
    # 遍历所有子目录
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
        
        # 解析目录名，例如 'steps10_maxT0.05'
        if not subdir.startswith('steps'):
            continue
        
        try:
            parts = subdir.split('_')
            num_steps = int(parts[0].replace('steps', ''))
            max_t = float(parts[1].replace('maxT', ''))
        except:
            print(f"警告: 无法解析目录名 {subdir}")
            continue
        
        # 查找文件
        rot_score_file = None
        trans_score_file = None
        pdb_file = None
        
        for filename in os.listdir(subdir_path):
            if 'rot_score' in filename and filename.endswith('.npy'):
                rot_score_file = os.path.join(subdir_path, filename)
            elif 'trans_score' in filename and filename.endswith('.npy'):
                trans_score_file = os.path.join(subdir_path, filename)
            elif filename.endswith('.pdb'):
                pdb_file = os.path.join(subdir_path, filename)
        
        # 加载数据
        if rot_score_file and trans_score_file:
            try:
                rot_score = np.load(rot_score_file)
                trans_score = np.load(trans_score_file)
                pdb_prot = load_pdb(pdb_file) if pdb_file else None
                
                data[(num_steps, max_t)] = {
                    'num_steps': num_steps,
                    'max_t': max_t,
                    'rot_score': rot_score,
                    'trans_score': trans_score,
                    'pdb_prot': pdb_prot,
                    'pdb_path': pdb_file,
                    'rot_score_path': rot_score_file,
                    'trans_score_path': trans_score_file,
                }
                print(f"✓ 加载: steps={num_steps}, maxT={max_t}")
            except Exception as e:
                print(f"警告: 加载数据失败 {subdir}: {e}")
    
    return data


def group_by_steps(data):
    """按照num_steps分组"""
    groups = defaultdict(list)
    for key, value in data.items():
        num_steps, max_t = key
        groups[num_steps].append((max_t, value))
    
    # 对每组按max_t排序
    for num_steps in groups:
        groups[num_steps].sort(key=lambda x: x[0])
    
    return dict(groups)


def group_by_max_t(data):
    """按照max_t分组"""
    groups = defaultdict(list)
    for key, value in data.items():
        num_steps, max_t = key
        groups[max_t].append((num_steps, value))
    
    # 对每组按num_steps排序
    for max_t in groups:
        groups[max_t].sort(key=lambda x: x[0])
    
    return dict(groups)


def analyze_group(group_data, group_name, group_type, output_subdir):
    """
    分析一个组的数据
    
    Args:
        group_data: [(param_value, data_dict), ...]
        group_name: 组名 (例如 'steps10' 或 'maxT0.05')
        group_type: 'steps' 或 'max_t'
        output_subdir: 输出子目录
    """
    os.makedirs(output_subdir, exist_ok=True)
    
    results = []
    
    # 计算所有成对比较
    n = len(group_data)
    for i in range(n):
        for j in range(i + 1, n):
            param1, data1 = group_data[i]
            param2, data2 = group_data[j]
            
            result = {
                'param1': param1,
                'param2': param2,
            }
            
            # 计算RMSD
            if data1['pdb_prot'] and data2['pdb_prot']:
                rmsd = compute_rmsd(data1['pdb_prot'], data2['pdb_prot'])
                result['rmsd'] = rmsd
            else:
                result['rmsd'] = None
            
            # 计算旋转分数距离
            rot1 = data1['rot_score']
            rot2 = data2['rot_score']
            rot_euc, rot_per_euc = compute_euclidean_distance(rot1, rot2)
            rot_cos, rot_per_cos, rot_sim = compute_cosine_similarity(rot1, rot2)
            
            result['rot_euclidean'] = rot_euc
            result['rot_cosine_dist'] = rot_cos
            result['rot_cosine_sim'] = rot_sim
            result['rot_per_euclidean_mean'] = np.mean(rot_per_euc)
            result['rot_per_euclidean_std'] = np.std(rot_per_euc)
            result['rot_per_cosine_mean'] = np.mean(rot_per_cos)
            result['rot_per_cosine_std'] = np.std(rot_per_cos)
            
            # 计算平移分数距离
            trans1 = data1['trans_score']
            trans2 = data2['trans_score']
            trans_euc, trans_per_euc = compute_euclidean_distance(trans1, trans2)
            trans_cos, trans_per_cos, trans_sim = compute_cosine_similarity(trans1, trans2)
            
            result['trans_euclidean'] = trans_euc
            result['trans_cosine_dist'] = trans_cos
            result['trans_cosine_sim'] = trans_sim
            result['trans_per_euclidean_mean'] = np.mean(trans_per_euc)
            result['trans_per_euclidean_std'] = np.std(trans_per_euc)
            result['trans_per_cosine_mean'] = np.mean(trans_per_cos)
            result['trans_per_cosine_std'] = np.std(trans_per_cos)
            
            results.append(result)
    
    # 保存结果CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_subdir, f'{group_name}_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"  ✓ 保存CSV: {csv_path}")
    
    # 生成可视化
    if len(results) > 0:
        plot_group_analysis(df, group_data, group_name, group_type, output_subdir)
    
    return df


def plot_group_analysis(df, group_data, group_name, group_type, output_subdir):
    """生成组分析可视化"""
    
    # 准备标签
    if group_type == 'steps':
        x_label = 'max_t'
        params = [param for param, _ in group_data]
        pair_labels = [f'{df.iloc[i]["param1"]:.2f} vs {df.iloc[i]["param2"]:.2f}' 
                      for i in range(len(df))]
    else:  # max_t
        x_label = 'num_steps'
        params = [param for param, _ in group_data]
        pair_labels = [f'{int(df.iloc[i]["param1"])} vs {int(df.iloc[i]["param2"])}' 
                      for i in range(len(df))]
    
    # 创建大图表
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(f'{group_name} - Pairwise Analysis', fontsize=16, fontweight='bold')
    
    # 1. RMSD
    ax = axes[0, 0]
    if 'rmsd' in df.columns and df['rmsd'].notna().any():
        ax.bar(range(len(df)), df['rmsd'], alpha=0.8, color='steelblue')
        ax.set_ylabel('RMSD (Å)', fontsize=11)
        ax.set_title('PDB RMSD Between Pairs', fontsize=12)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(df['rmsd']):
            if not np.isnan(v):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No PDB Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('PDB RMSD (No Data)', fontsize=12)
    
    # 2. 旋转分数 - 欧氏距离
    ax = axes[0, 1]
    ax.bar(range(len(df)), df['rot_euclidean'], alpha=0.8, color='coral')
    ax.set_ylabel('Euclidean Distance', fontsize=11)
    ax.set_title('Rotation Score - Euclidean Distance', fontsize=12)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(df['rot_euclidean']):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    
    # 3. 旋转分数 - 余弦相似度
    ax = axes[0, 2]
    ax.bar(range(len(df)), df['rot_cosine_sim'], alpha=0.8, color='mediumseagreen')
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title('Rotation Score - Cosine Similarity', fontsize=12)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(df['rot_cosine_sim']):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    
    # 4. 平移分数 - 欧氏距离
    ax = axes[1, 0]
    ax.bar(range(len(df)), df['trans_euclidean'], alpha=0.8, color='purple')
    ax.set_ylabel('Euclidean Distance', fontsize=11)
    ax.set_title('Translation Score - Euclidean Distance', fontsize=12)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(df['trans_euclidean']):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    
    # 5. 平移分数 - 余弦相似度
    ax = axes[1, 1]
    ax.bar(range(len(df)), df['trans_cosine_sim'], alpha=0.8, color='orange')
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title('Translation Score - Cosine Similarity', fontsize=12)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(df['trans_cosine_sim']):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    
    # 6. 旋转vs平移 欧氏距离相关性
    ax = axes[1, 2]
    ax.scatter(df['rot_euclidean'], df['trans_euclidean'], s=100, alpha=0.6, c=range(len(df)), cmap='viridis')
    ax.set_xlabel('Rotation Euclidean', fontsize=11)
    ax.set_ylabel('Translation Euclidean', fontsize=11)
    ax.set_title('Rotation vs Translation (Euclidean)', fontsize=12)
    ax.grid(True, alpha=0.3)
    if len(df) > 1:
        corr, _ = pearsonr(df['rot_euclidean'], df['trans_euclidean'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. 旋转分数 - 每残基统计
    ax = axes[2, 0]
    x_pos = np.arange(len(df))
    width = 0.35
    ax.bar(x_pos - width/2, df['rot_per_euclidean_mean'], width, 
           label='Mean Euclidean', alpha=0.8, color='skyblue')
    ax.bar(x_pos + width/2, df['rot_per_euclidean_std'], width, 
           label='Std Euclidean', alpha=0.8, color='lightcoral')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Rotation Score - Per-Residue Statistics', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 8. 平移分数 - 每残基统计
    ax = axes[2, 1]
    ax.bar(x_pos - width/2, df['trans_per_euclidean_mean'], width, 
           label='Mean Euclidean', alpha=0.8, color='lightgreen')
    ax.bar(x_pos + width/2, df['trans_per_euclidean_std'], width, 
           label='Std Euclidean', alpha=0.8, color='plum')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Translation Score - Per-Residue Statistics', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 9. 综合热图
    ax = axes[2, 2]
    metrics = ['RMSD', 'Rot Euc', 'Rot Sim', 'Trans Euc', 'Trans Sim']
    heatmap_data = []
    for i in range(len(df)):
        row = [
            df.iloc[i]['rmsd'] if not pd.isna(df.iloc[i]['rmsd']) else 0,
            df.iloc[i]['rot_euclidean'],
            df.iloc[i]['rot_cosine_sim'],
            df.iloc[i]['trans_euclidean'],
            df.iloc[i]['trans_cosine_sim'],
        ]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data).T
    # 归一化每行
    for k in range(len(metrics)):
        row_max = np.max(heatmap_data[k])
        row_min = np.min(heatmap_data[k])
        if row_max > row_min:
            heatmap_data[k] = (heatmap_data[k] - row_min) / (row_max - row_min)
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_title('Normalized Metrics Heatmap', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    output_path = os.path.join(output_subdir, f'{group_name}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存图表: {output_path}")
    plt.close()


def generate_summary_report(all_results, output_dir):
    """生成总结报告"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("参数组分析总结报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 按步数分组的结果
        f.write("=" * 80 + "\n")
        f.write("【按照NUM_STEPS分组的分析】\n")
        f.write("=" * 80 + "\n\n")
        
        for group_name, df in all_results['by_steps'].items():
            f.write(f"\n{'─' * 80}\n")
            f.write(f"组: {group_name}\n")
            f.write(f"{'─' * 80}\n")
            f.write(f"比较数量: {len(df)}\n\n")
            
            if 'rmsd' in df.columns and df['rmsd'].notna().any():
                f.write(f"RMSD统计:\n")
                f.write(f"  平均值: {df['rmsd'].mean():.4f} Å\n")
                f.write(f"  标准差: {df['rmsd'].std():.4f} Å\n")
                f.write(f"  最小值: {df['rmsd'].min():.4f} Å\n")
                f.write(f"  最大值: {df['rmsd'].max():.4f} Å\n\n")
            
            f.write(f"旋转分数统计:\n")
            f.write(f"  欧氏距离均值: {df['rot_euclidean'].mean():.4f}\n")
            f.write(f"  余弦相似度均值: {df['rot_cosine_sim'].mean():.6f}\n\n")
            
            f.write(f"平移分数统计:\n")
            f.write(f"  欧氏距离均值: {df['trans_euclidean'].mean():.4f}\n")
            f.write(f"  余弦相似度均值: {df['trans_cosine_sim'].mean():.6f}\n\n")
        
        # 按max_t分组的结果
        f.write("\n" + "=" * 80 + "\n")
        f.write("【按照MAX_T分组的分析】\n")
        f.write("=" * 80 + "\n\n")
        
        for group_name, df in all_results['by_max_t'].items():
            f.write(f"\n{'─' * 80}\n")
            f.write(f"组: {group_name}\n")
            f.write(f"{'─' * 80}\n")
            f.write(f"比较数量: {len(df)}\n\n")
            
            if 'rmsd' in df.columns and df['rmsd'].notna().any():
                f.write(f"RMSD统计:\n")
                f.write(f"  平均值: {df['rmsd'].mean():.4f} Å\n")
                f.write(f"  标准差: {df['rmsd'].std():.4f} Å\n")
                f.write(f"  最小值: {df['rmsd'].min():.4f} Å\n")
                f.write(f"  最大值: {df['rmsd'].max():.4f} Å\n\n")
            
            f.write(f"旋转分数统计:\n")
            f.write(f"  欧氏距离均值: {df['rot_euclidean'].mean():.4f}\n")
            f.write(f"  余弦相似度均值: {df['rot_cosine_sim'].mean():.6f}\n\n")
            
            f.write(f"平移分数统计:\n")
            f.write(f"  欧氏距离均值: {df['trans_euclidean'].mean():.4f}\n")
            f.write(f"  余弦相似度均值: {df['trans_cosine_sim'].mean():.6f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("分析完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✅ 总结报告已保存: {report_path}")


def main():
    print("=" * 80)
    print("参数组分析器")
    print("=" * 80)
    print("功能:")
    print("  1. 按照NUM_STEPS分组，比较不同MAX_T的结果")
    print("  2. 按照MAX_T分组，比较不同NUM_STEPS的结果")
    print("  3. 计算PDB文件的RMSD")
    print("  4. 分析旋转和平移分数的欧氏距离和余弦相似度")
    print("  5. 生成详细的分析报告和可视化图表")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {RUN_OUTPUT_DIR}")
    
    # 加载数据
    print(f"\n{'='*80}")
    print("加载实验数据...")
    print(f"{'='*80}")
    print(f"输入目录: {INPUT_DIR}")
    
    data = load_experiment_data(INPUT_DIR)
    print(f"\n✓ 总共加载了 {len(data)} 个实验结果")
    
    # 按步数分组
    print(f"\n{'='*80}")
    print("按照NUM_STEPS分组...")
    print(f"{'='*80}")
    groups_by_steps = group_by_steps(data)
    print(f"✓ 共 {len(groups_by_steps)} 个步数组")
    
    # 按max_t分组
    print(f"\n{'='*80}")
    print("按照MAX_T分组...")
    print(f"{'='*80}")
    groups_by_max_t = group_by_max_t(data)
    print(f"✓ 共 {len(groups_by_max_t)} 个max_t组")
    
    # 分析各组
    all_results = {
        'by_steps': {},
        'by_max_t': {},
    }
    
    # 分析按步数分组的数据
    print(f"\n{'='*80}")
    print("分析按NUM_STEPS分组的数据...")
    print(f"{'='*80}")
    
    for num_steps, group_data in groups_by_steps.items():
        group_name = f'steps{num_steps}'
        print(f"\n分析组: {group_name} (共{len(group_data)}个实验)")
        
        output_subdir = os.path.join(RUN_OUTPUT_DIR, 'by_steps', group_name)
        df = analyze_group(group_data, group_name, 'steps', output_subdir)
        all_results['by_steps'][group_name] = df
    
    # 分析按max_t分组的数据
    print(f"\n{'='*80}")
    print("分析按MAX_T分组的数据...")
    print(f"{'='*80}")
    
    for max_t, group_data in groups_by_max_t.items():
        group_name = f'maxT{max_t:.2f}'
        print(f"\n分析组: {group_name} (共{len(group_data)}个实验)")
        
        output_subdir = os.path.join(RUN_OUTPUT_DIR, 'by_max_t', group_name)
        df = analyze_group(group_data, group_name, 'max_t', output_subdir)
        all_results['by_max_t'][group_name] = df
    
    # 生成总结报告
    print(f"\n{'='*80}")
    print("生成总结报告...")
    print(f"{'='*80}")
    generate_summary_report(all_results, RUN_OUTPUT_DIR)
    
    # 统计结果
    total_csv = len([f for f in Path(RUN_OUTPUT_DIR).rglob('*.csv')])
    total_png = len([f for f in Path(RUN_OUTPUT_DIR).rglob('*.png')])
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}")
    print(f"📁 输出目录: {RUN_OUTPUT_DIR}")
    print(f"📊 生成CSV: {total_csv} 个")
    print(f"📈 生成图表: {total_png} 个")
    print(f"📋 总结报告: summary_report.txt")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
