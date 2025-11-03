#!/usr/bin/env python3
"""
Score距离分析器

功能:
1. 计算rot_score之间的欧氏距离和余弦距离
2. 计算trans_score之间的欧氏距离和余弦距离
3. 进行数据分析

"""

import os
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


def load_score_pairs(input_dir, prefix='1AKE_B'):
    """
    加载成对的score文件
    
    Args:
        input_dir: 包含.npy文件的目录
        prefix: 文件名前缀
    
    Returns:
        dict: 包含rot和trans score对的字典
    """
    scores = {}
    
    # 查找所有匹配的文件
    rot_files = sorted([f for f in os.listdir(input_dir) if f.startswith(prefix) and 'rot_score' in f and f.endswith('.npy')])
    trans_files = sorted([f for f in os.listdir(input_dir) if f.startswith(prefix) and 'trans_score' in f and f.endswith('.npy')])
    
    print(f"找到 {len(rot_files)} 个rot_score文件")
    print(f"找到 {len(trans_files)} 个trans_score文件")
    
    # 加载rot_score
    rot_scores = []
    for f in rot_files:
        path = os.path.join(input_dir, f)
        score = np.load(path)
        rot_scores.append({'filename': f, 'data': score})
        print(f"  加载: {f}, shape: {score.shape}")
    
    # 加载trans_score
    trans_scores = []
    for f in trans_files:
        path = os.path.join(input_dir, f)
        score = np.load(path)
        trans_scores.append({'filename': f, 'data': score})
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
    Plot pairwise distance matrix visualizations
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



def main():
    print("=" * 80)
    print("Score距离分析器")
    print("=" * 80)
    print("功能:")
    print("  1. 对所有rot_score文件进行两两距离计算")
    print("  2. 对所有trans_score文件进行两两距离计算")
    print("  3. 计算欧氏距离和余弦距离/相似度")
    print("  4. 生成详细的文本报告和可视化图表")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # 加载score文件
    print(f"\n正在加载score文件...")
    scores = load_score_pairs(INPUT_DIR)
    
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
    
    print("\n生成 TRANS Score 可视化...")
    plot_distance_matrix(trans_results, 'trans', OUTPUT_DIR)
    
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
        print(f"  - {score_type}_correlation_analysis.png")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
