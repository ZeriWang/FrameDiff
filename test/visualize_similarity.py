#!/usr/bin/env python3
"""
蛋白质相似度可视化分析程序

读取相似度计算结果并生成可视化图表和深度分析报告

功能特性:
1. 时间演化曲线图
2. 距离分布直方图
3. 相关性分析
4. 相似度热图
5. 综合分析报告
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_similarity_data(data_file):
    """加载相似度数据"""
    data = np.load(data_file)
    return {
        'time_steps': data['time_steps'],
        'euclidean_rotation': data['euclidean_rotation_means'],
        'euclidean_translation': data['euclidean_translation_means'],
        'euclidean_total': data['euclidean_total_means'],
        'cosine_rotation': data['cosine_rotation_means'],
        'cosine_translation': data['cosine_translation_means'],
        'cosine_total': data['cosine_total_means']
    }

def create_time_evolution_plot(data, output_dir):
    """创建时间演化图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('蛋白质结构相似度时间演化分析', fontsize=16, fontweight='bold')
    
    time_steps = data['time_steps']
    
    # 欧氏距离图
    axes[0, 0].plot(time_steps, data['euclidean_rotation'], 'o-', linewidth=2, markersize=4, label='旋转距离')
    axes[0, 0].set_title('欧氏距离 - 旋转')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('欧氏距离')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_steps, data['euclidean_translation'], 'o-', linewidth=2, markersize=4, color='orange', label='平移距离')
    axes[0, 1].set_title('欧氏距离 - 平移')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('欧氏距离')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(time_steps, data['euclidean_total'], 'o-', linewidth=2, markersize=4, color='red', label='加权总距离')
    axes[0, 2].set_title('欧氏距离 - 加权总距离')
    axes[0, 2].set_xlabel('时间步')
    axes[0, 2].set_ylabel('欧氏距离')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 余弦距离图
    axes[1, 0].plot(time_steps, data['cosine_rotation'], 'o-', linewidth=2, markersize=4, color='green', label='旋转距离')
    axes[1, 0].set_title('余弦距离 - 旋转')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('余弦距离')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_steps, data['cosine_translation'], 'o-', linewidth=2, markersize=4, color='purple', label='平移距离')
    axes[1, 1].set_title('余弦距离 - 平移')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('余弦距离')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(time_steps, data['cosine_total'], 'o-', linewidth=2, markersize=4, color='brown', label='加权总距离')
    axes[1, 2].set_title('余弦距离 - 加权总距离')
    axes[1, 2].set_xlabel('时间步')
    axes[1, 2].set_ylabel('余弦距离')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'time_evolution_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 时间演化图保存到: {output_file}")

def create_comparison_plot(data, output_dir):
    """创建对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('欧氏距离 vs 余弦距离对比', fontsize=14, fontweight='bold')
    
    time_steps = data['time_steps']
    
    # 旋转距离对比
    axes[0].plot(time_steps, data['euclidean_rotation'], 'o-', linewidth=2, markersize=4, label='欧氏距离', alpha=0.8)
    axes[0].plot(time_steps, data['cosine_rotation'], 'o-', linewidth=2, markersize=4, label='余弦距离', alpha=0.8)
    axes[0].set_title('旋转距离对比')
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('距离值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 平移距离对比
    axes[1].plot(time_steps, data['euclidean_translation'], 'o-', linewidth=2, markersize=4, label='欧氏距离', alpha=0.8)
    axes[1].plot(time_steps, data['cosine_translation'], 'o-', linewidth=2, markersize=4, label='余弦距离', alpha=0.8)
    axes[1].set_title('平移距离对比')
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('距离值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'distance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 对比图保存到: {output_file}")

def create_heatmap(data, output_dir):
    """创建相似度热图"""
    # 构建矩阵数据
    matrix_data = np.array([
        data['euclidean_rotation'],
        data['euclidean_translation'], 
        data['euclidean_total'],
        data['cosine_rotation'],
        data['cosine_translation'],
        data['cosine_total']
    ])
    
    labels = ['欧氏-旋转', '欧氏-平移', '欧氏-总计', '余弦-旋转', '余弦-平移', '余弦-总计']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 创建热图
    im = ax.imshow(matrix_data, cmap='YlOrRd', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(0, len(data['time_steps']), 5))
    ax.set_xticklabels([f"{data['time_steps'][i]:.3f}" for i in range(0, len(data['time_steps']), 5)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('距离值', rotation=270, labelpad=20)
    
    # 设置标题和标签
    ax.set_title('蛋白质相似度距离热图', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间步')
    ax.set_ylabel('距离类型')
    
    plt.tight_layout()
    output_file = output_dir / 'similarity_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 热图保存到: {output_file}")

def analyze_correlations(data):
    """分析相关性"""
    print("\n" + "="*60)
    print("         相关性分析")
    print("="*60)
    
    # 计算各种距离之间的相关性
    correlations = {}
    distance_types = ['euclidean_rotation', 'euclidean_translation', 'euclidean_total',
                     'cosine_rotation', 'cosine_translation', 'cosine_total']
    
    for i, type1 in enumerate(distance_types):
        for j, type2 in enumerate(distance_types):
            if i < j:  # 避免重复计算
                corr = np.corrcoef(data[type1], data[type2])[0, 1]
                correlations[f"{type1}_vs_{type2}"] = corr
    
    print("📊 距离类型间相关性:")
    for pair, corr in correlations.items():
        type1, type2 = pair.split('_vs_')
        print(f"   • {type1} vs {type2}: {corr:.4f}")
    
    # 与时间的相关性
    print(f"\n⏱️ 与时间步的相关性:")
    for distance_type in distance_types:
        time_corr = np.corrcoef(data['time_steps'], data[distance_type])[0, 1]
        print(f"   • {distance_type}: {time_corr:.4f}")
    
    return correlations

def generate_insights(data):
    """生成洞察分析"""
    print("\n" + "="*60)
    print("         深度洞察分析")
    print("="*60)
    
    # 计算变化趋势
    euclidean_rot_change = data['euclidean_rotation'][-1] - data['euclidean_rotation'][0]
    euclidean_trans_change = data['euclidean_translation'][-1] - data['euclidean_translation'][0]
    cosine_rot_change = data['cosine_rotation'][-1] - data['cosine_rotation'][0]
    cosine_trans_change = data['cosine_translation'][-1] - data['cosine_translation'][0]
    
    print(f"📈 距离变化趋势:")
    print(f"   • 欧氏旋转距离变化: {euclidean_rot_change:+.4f}")
    print(f"   • 欧氏平移距离变化: {euclidean_trans_change:+.4f}")
    print(f"   • 余弦旋转距离变化: {cosine_rot_change:+.4f}")
    print(f"   • 余弦平移距离变化: {cosine_trans_change:+.4f}")
    
    # 找出最相似和最不相似的时间点
    euclidean_total = data['euclidean_total']
    cosine_total = data['cosine_total']
    
    euc_min_idx = np.argmin(euclidean_total)
    euc_max_idx = np.argmax(euclidean_total)
    cos_min_idx = np.argmin(cosine_total)
    cos_max_idx = np.argmax(cosine_total)
    
    print(f"\n🎯 关键时间点:")
    print(f"   • 欧氏距离最相似时间点: t={data['time_steps'][euc_min_idx]:.4f} (距离={euclidean_total[euc_min_idx]:.4f})")
    print(f"   • 欧氏距离最不相似时间点: t={data['time_steps'][euc_max_idx]:.4f} (距离={euclidean_total[euc_max_idx]:.4f})")
    print(f"   • 余弦距离最相似时间点: t={data['time_steps'][cos_min_idx]:.4f} (距离={cosine_total[cos_min_idx]:.4f})")
    print(f"   • 余弦距离最不相似时间点: t={data['time_steps'][cos_max_idx]:.4f} (距离={cosine_total[cos_max_idx]:.4f})")
    
    # 计算收敛性
    final_portion = len(data['time_steps']) // 4  # 最后25%的时间步
    euclidean_stability = np.std(euclidean_total[-final_portion:])
    cosine_stability = np.std(cosine_total[-final_portion:])
    
    print(f"\n📐 后期稳定性 (最后25%时间步的标准差):")
    print(f"   • 欧氏距离稳定性: {euclidean_stability:.6f}")
    print(f"   • 余弦距离稳定性: {cosine_stability:.6f}")

def create_comprehensive_report(data, correlations, output_dir, protein1, protein2):
    """创建综合分析报告"""
    report_file = output_dir / f"{protein1}_vs_{protein2}_comprehensive_analysis.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 蛋白质结构相似度综合分析报告\n\n")
        f.write(f"**分析对象**: {protein1} vs {protein2}\n")
        f.write(f"**分析时间**: 2025-09-22\n")
        f.write(f"**时间步数量**: {len(data['time_steps'])}\n")
        f.write(f"**时间步范围**: {data['time_steps'].min():.4f} - {data['time_steps'].max():.4f}\n\n")
        
        f.write("## 📊 主要发现\n\n")
        
        # 距离变化
        euclidean_rot_change = data['euclidean_rotation'][-1] - data['euclidean_rotation'][0]
        euclidean_trans_change = data['euclidean_translation'][-1] - data['euclidean_translation'][0]
        cosine_rot_change = data['cosine_rotation'][-1] - data['cosine_rotation'][0]
        cosine_trans_change = data['cosine_translation'][-1] - data['cosine_translation'][0]
        
        f.write(f"### 距离变化趋势\n")
        f.write(f"- **欧氏旋转距离**: {data['euclidean_rotation'][0]:.4f} → {data['euclidean_rotation'][-1]:.4f} (变化: {euclidean_rot_change:+.4f})\n")
        f.write(f"- **欧氏平移距离**: {data['euclidean_translation'][0]:.4f} → {data['euclidean_translation'][-1]:.4f} (变化: {euclidean_trans_change:+.4f})\n")
        f.write(f"- **余弦旋转距离**: {data['cosine_rotation'][0]:.4f} → {data['cosine_rotation'][-1]:.4f} (变化: {cosine_rot_change:+.4f})\n")
        f.write(f"- **余弦平移距离**: {data['cosine_translation'][0]:.4f} → {data['cosine_translation'][-1]:.4f} (变化: {cosine_trans_change:+.4f})\n\n")
        
        # 关键时间点
        euclidean_total = data['euclidean_total']
        cosine_total = data['cosine_total']
        euc_min_idx = np.argmin(euclidean_total)
        euc_max_idx = np.argmax(euclidean_total)
        cos_min_idx = np.argmin(cosine_total)
        cos_max_idx = np.argmax(cosine_total)
        
        f.write(f"### 关键时间点\n")
        f.write(f"- **欧氏距离最相似**: t={data['time_steps'][euc_min_idx]:.4f} (距离={euclidean_total[euc_min_idx]:.4f})\n")
        f.write(f"- **欧氏距离最不相似**: t={data['time_steps'][euc_max_idx]:.4f} (距离={euclidean_total[euc_max_idx]:.4f})\n")
        f.write(f"- **余弦距离最相似**: t={data['time_steps'][cos_min_idx]:.4f} (距离={cosine_total[cos_min_idx]:.4f})\n")
        f.write(f"- **余弦距离最不相似**: t={data['time_steps'][cos_max_idx]:.4f} (距离={cosine_total[cos_max_idx]:.4f})\n\n")
        
        # 统计信息
        f.write(f"### 统计摘要\n")
        for distance_type in ['euclidean_rotation', 'euclidean_translation', 'euclidean_total',
                             'cosine_rotation', 'cosine_translation', 'cosine_total']:
            values = data[distance_type]
            f.write(f"- **{distance_type}**: 均值={np.mean(values):.6f}, 标准差={np.std(values):.6f}, 范围=[{np.min(values):.6f}, {np.max(values):.6f}]\n")
        
        f.write(f"\n### 相关性分析\n")
        for pair, corr in correlations.items():
            f.write(f"- **{pair.replace('_vs_', ' vs ')}**: {corr:.4f}\n")
        
        f.write(f"\n## 📈 生成图表\n")
        f.write(f"- `time_evolution_plots.png`: 时间演化分析图\n")
        f.write(f"- `distance_comparison.png`: 距离类型对比图\n")
        f.write(f"- `similarity_heatmap.png`: 相似度热图\n")
        
        f.write(f"\n---\n*报告由蛋白质相似度可视化分析程序自动生成*\n")
    
    print(f"✓ 综合分析报告保存到: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='蛋白质相似度可视化分析程序')
    parser.add_argument('data_file', help='相似度数据文件路径 (.npz)')
    parser.add_argument('--output-dir', help='输出目录路径')
    parser.add_argument('--protein1', default='Protein1', help='第一个蛋白质名称')
    parser.add_argument('--protein2', default='Protein2', help='第二个蛋白质名称')
    
    args = parser.parse_args()
    
    try:
        # 设置输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.data_file).parent / "visualization"
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"蛋白质相似度可视化分析")
        print(f"数据文件: {args.data_file}")
        print(f"输出目录: {output_dir}")
        print("="*60)
        
        # 加载数据
        data = load_similarity_data(args.data_file)
        
        # 创建可视化
        print("📊 生成可视化图表...")
        create_time_evolution_plot(data, output_dir)
        create_comparison_plot(data, output_dir)
        create_heatmap(data, output_dir)
        
        # 分析相关性
        correlations = analyze_correlations(data)
        
        # 生成洞察
        generate_insights(data)
        
        # 创建综合报告
        create_comprehensive_report(data, correlations, output_dir, args.protein1, args.protein2)
        
        print(f"\n✅ 可视化分析完成！")
        print(f"所有文件保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()