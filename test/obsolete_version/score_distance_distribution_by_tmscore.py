#!/usr/bin/env python3
"""
按TM-score分组绘制分数距离分布的工具。

该脚本从random_denoising_report.md中解析Pairwise数据，
将样本对按TM-score分为两组 (TM-score <= 0.4 和 TM-score > 0.6)，
并绘制各种分数距离指标在这两组中的分布直方图。

用途：探索"分数相似但结构不相似"的多构象蛋白质现象。
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = PROJECT_ROOT / "test" / "random_batch_output" / "1lah_E_1000" / "random_denoising_report.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "random_batch_output" / "1lah_E_1000" / "tmscore_stratified_plots"


@dataclass
class PairMetric:
    """存储单个样本对的度量数据"""
    pair_name: str
    tm_score: float
    rot_euclidean: float
    rot_cosine_dist: float
    trans_euclidean: float
    trans_cosine_dist: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按TM-score分组绘制分数距离分布",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT,
        help="random_denoising_report.md 的路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出图像的目录",
    )
    parser.add_argument(
        "--low-tm-threshold",
        type=float,
        default=0.4,
        help="低TM-score阈值 (<=此值的样本对)",
    )
    parser.add_argument(
        "--high-tm-threshold",
        type=float,
        default=0.6,
        help="高TM-score阈值 (>此值的样本对)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="直方图的bin数量",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="输出图像的DPI",
    )
    return parser.parse_args()


def parse_report(report_path: Path) -> List[PairMetric]:
    """
    解析random_denoising_report.md中的Pairwise数据。
    
    数据格式示例:
    | Pair | TM-score | Rot Euc | Rot CosDist | Trans Euc | Trans CosDist |
    | sample_a vs sample_b | 0.2824 | 110.8394 | 0.8822 | 243.5724 | 0.8607 |
    """
    print(f"正在解析报告: {report_path}")
    
    metrics: List[PairMetric] = []
    in_pairwise_section = False
    header_seen = False
    
    # 正则表达式匹配数据行
    # 格式: | pair_name | tm_score | rot_euc | rot_cos | trans_euc | trans_cos |
    data_pattern = re.compile(
        r'\|\s*(.+?)\s+vs\s+(.+?)\s*\|\s*'  # Pair names
        r'([0-9.]+)\s*\|\s*'                 # TM-score
        r'([0-9.]+)\s*\|\s*'                 # Rot Euc
        r'([0-9.]+)\s*\|\s*'                 # Rot CosDist
        r'([0-9.]+)\s*\|\s*'                 # Trans Euc
        r'([0-9.]+)\s*\|'                    # Trans CosDist
    )
    
    with open(report_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 检测进入Pairwise部分
            if "## Pairwise TM-score" in line:
                in_pairwise_section = True
                continue
            
            # 检测离开Pairwise部分 (遇到下一个section)
            if in_pairwise_section and line.startswith("## ") and "Pairwise" not in line:
                break
            
            if not in_pairwise_section:
                continue
            
            # 跳过表头行
            if "| Pair |" in line or "| --- |" in line:
                header_seen = True
                continue
            
            if not header_seen:
                continue
            
            # 尝试匹配数据行
            match = data_pattern.search(line)
            if match:
                try:
                    pair_name = f"{match.group(1).strip()} vs {match.group(2).strip()}"
                    tm_score = float(match.group(3))
                    rot_euc = float(match.group(4))
                    rot_cos = float(match.group(5))
                    trans_euc = float(match.group(6))
                    trans_cos = float(match.group(7))
                    
                    metrics.append(PairMetric(
                        pair_name=pair_name,
                        tm_score=tm_score,
                        rot_euclidean=rot_euc,
                        rot_cosine_dist=rot_cos,
                        trans_euclidean=trans_euc,
                        trans_cosine_dist=trans_cos,
                    ))
                except ValueError as e:
                    print(f"警告: 第 {line_num} 行解析失败: {e}")
                    continue
            
            # 每10万条打印进度
            if len(metrics) > 0 and len(metrics) % 100000 == 0:
                print(f"  已解析 {len(metrics)} 条数据...")
    
    print(f"共解析到 {len(metrics)} 条样本对数据")
    return metrics


def split_by_tmscore(
    metrics: List[PairMetric],
    low_threshold: float,
    high_threshold: float,
) -> Tuple[List[PairMetric], List[PairMetric]]:
    """
    按TM-score阈值分割数据。
    
    Returns:
        (low_tm_group, high_tm_group)
        - low_tm_group: TM-score <= low_threshold
        - high_tm_group: TM-score > high_threshold
    """
    low_group = [m for m in metrics if m.tm_score <= low_threshold]
    high_group = [m for m in metrics if m.tm_score > high_threshold]
    
    print(f"TM-score <= {low_threshold}: {len(low_group)} 对")
    print(f"TM-score > {high_threshold}: {len(high_group)} 对")
    
    return low_group, high_group


def extract_metric_values(
    metrics: List[PairMetric],
    metric_name: str,
) -> np.ndarray:
    """提取指定度量的值数组"""
    attr_map = {
        'rot_euclidean': 'rot_euclidean',
        'rot_cosine_dist': 'rot_cosine_dist',
        'trans_euclidean': 'trans_euclidean',
        'trans_cosine_dist': 'trans_cosine_dist',
    }
    attr = attr_map.get(metric_name)
    if attr is None:
        raise ValueError(f"未知度量: {metric_name}")
    
    values = np.array([getattr(m, attr) for m in metrics], dtype=float)
    return values


def plot_distribution_comparison(
    low_group: List[PairMetric],
    high_group: List[PairMetric],
    metric_name: str,
    metric_label: str,
    low_threshold: float,
    high_threshold: float,
    output_path: Path,
    bins: int = 50,
    dpi: int = 200,
):
    """
    绘制两组数据在指定度量上的分布对比图。
    """
    low_values = extract_metric_values(low_group, metric_name)
    high_values = extract_metric_values(high_group, metric_name)
    
    # 过滤无效值
    low_values = low_values[np.isfinite(low_values)]
    high_values = high_values[np.isfinite(high_values)]
    
    if len(low_values) == 0 and len(high_values) == 0:
        print(f"警告: {metric_name} 没有有效数据，跳过绘图")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 确定共享的x轴范围
    all_values = np.concatenate([low_values, high_values]) if len(low_values) > 0 and len(high_values) > 0 else (
        low_values if len(low_values) > 0 else high_values
    )
    x_min, x_max = np.percentile(all_values, [1, 99])
    x_range = (x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    
    # 左图: TM-score <= low_threshold
    ax1 = axes[0]
    if len(low_values) > 0:
        if HAS_SEABORN:
            sns.histplot(low_values, bins=bins, kde=True, ax=ax1, color='steelblue', alpha=0.7)
        else:
            ax1.hist(low_values, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axvline(np.mean(low_values), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(low_values):.2f}')
        ax1.axvline(np.median(low_values), color='orange', linestyle='--', linewidth=1.5, label=f'Median: {np.median(low_values):.2f}')
    ax1.set_xlim(x_range)
    ax1.set_xlabel(metric_label, fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'TM-score ≤ {low_threshold}\n(n={len(low_values):,})', fontsize=13)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 右图: TM-score > high_threshold
    ax2 = axes[1]
    if len(high_values) > 0:
        if HAS_SEABORN:
            sns.histplot(high_values, bins=bins, kde=True, ax=ax2, color='coral', alpha=0.7)
        else:
            ax2.hist(high_values, bins=bins, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axvline(np.mean(high_values), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(high_values):.2f}')
        ax2.axvline(np.median(high_values), color='orange', linestyle='--', linewidth=1.5, label=f'Median: {np.median(high_values):.2f}')
    ax2.set_xlim(x_range)
    ax2.set_xlabel(metric_label, fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'TM-score > {high_threshold}\n(n={len(high_values):,})', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.suptitle(f'{metric_label} Distribution by TM-score Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_single_distribution(
    data_group: List[PairMetric],
    metric_name: str,
    metric_label: str,
    tm_condition: str,
    tm_threshold: float,
    output_path: Path,
    bins: int = 50,
    dpi: int = 200,
    color: str = 'steelblue',
):
    """
    为单个TM-score分组绘制指定度量的分布图。
    
    Args:
        data_group: 该TM-score分组的数据
        metric_name: 度量名称（用于提取数据）
        metric_label: 度量标签（用于显示）
        tm_condition: TM-score条件描述（如 '≤ 0.4' 或 '> 0.6'）
        tm_threshold: TM-score阈值
        output_path: 输出路径
        bins: 直方图bin数量
        dpi: 输出DPI
        color: 直方图颜色
    """
    values = extract_metric_values(data_group, metric_name)
    values = values[np.isfinite(values)]
    
    if len(values) == 0:
        print(f"警告: {metric_name} (TM-score {tm_condition}) 没有有效数据，跳过绘图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    if HAS_SEABORN:
        sns.histplot(values, bins=bins, kde=True, ax=ax, color=color, alpha=0.7)
    else:
        ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加统计线
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.2f}')
    ax.axvline(mean_val - std_val, color='green', linestyle=':', linewidth=1.5, 
               label=f'Mean ± Std: [{mean_val-std_val:.2f}, {mean_val+std_val:.2f}]')
    ax.axvline(mean_val + std_val, color='green', linestyle=':', linewidth=1.5)
    
    # 添加Q25和Q75
    q25, q75 = np.percentile(values, [25, 75])
    ax.axvline(q25, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7,
               label=f'Q25: {q25:.2f}')
    ax.axvline(q75, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7,
               label=f'Q75: {q75:.2f}')
    
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{metric_label}\nTM-score {tm_condition} (n={len(values):,})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加统计信息文本框
    stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {np.min(values):.4f}\nMax: {np.max(values):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_overlay_distribution(
    low_group: List[PairMetric],
    high_group: List[PairMetric],
    metric_name: str,
    metric_label: str,
    low_threshold: float,
    high_threshold: float,
    output_path: Path,
    bins: int = 50,
    dpi: int = 200,
):
    """
    绘制两组数据在指定度量上的叠加分布图。
    """
    low_values = extract_metric_values(low_group, metric_name)
    high_values = extract_metric_values(high_group, metric_name)
    
    # 过滤无效值
    low_values = low_values[np.isfinite(low_values)]
    high_values = high_values[np.isfinite(high_values)]
    
    if len(low_values) == 0 and len(high_values) == 0:
        print(f"警告: {metric_name} 没有有效数据，跳过绘图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算共同的bin边界
    all_values = np.concatenate([low_values, high_values]) if len(low_values) > 0 and len(high_values) > 0 else (
        low_values if len(low_values) > 0 else high_values
    )
    bin_edges = np.histogram_bin_edges(all_values, bins=bins)
    
    # 绘制两组的直方图 (归一化为密度以便比较)
    if len(low_values) > 0:
        ax.hist(low_values, bins=bin_edges, density=True, alpha=0.5, 
                color='steelblue', label=f'TM-score ≤ {low_threshold} (n={len(low_values):,})', edgecolor='darkblue', linewidth=0.5)
    if len(high_values) > 0:
        ax.hist(high_values, bins=bin_edges, density=True, alpha=0.5,
                color='coral', label=f'TM-score > {high_threshold} (n={len(high_values):,})', edgecolor='darkred', linewidth=0.5)
    
    # 添加KDE曲线
    if HAS_SEABORN:
        if len(low_values) > 0:
            sns.kdeplot(low_values, ax=ax, color='darkblue', linewidth=2)
        if len(high_values) > 0:
            sns.kdeplot(high_values, ax=ax, color='darkred', linewidth=2)
    
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{metric_label} Distribution: Low vs High TM-score', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def generate_summary_stats(
    low_group: List[PairMetric],
    high_group: List[PairMetric],
    low_threshold: float,
    high_threshold: float,
    output_path: Path,
):
    """生成统计摘要报告"""
    metrics_info = [
        ('rot_euclidean', 'Rotational Euclidean Distance'),
        ('rot_cosine_dist', 'Rotational Cosine Distance'),
        ('trans_euclidean', 'Translational Euclidean Distance'),
        ('trans_cosine_dist', 'Translational Cosine Distance'),
    ]
    
    lines = []
    lines.append("# 分数距离分布统计报告 (按TM-score分组)")
    lines.append("")
    lines.append("## 分组信息")
    lines.append(f"- 低TM-score组: TM-score ≤ {low_threshold}, 共 {len(low_group):,} 对")
    lines.append(f"- 高TM-score组: TM-score > {high_threshold}, 共 {len(high_group):,} 对")
    lines.append("")
    
    for metric_name, metric_label in metrics_info:
        low_values = extract_metric_values(low_group, metric_name)
        high_values = extract_metric_values(high_group, metric_name)
        
        low_values = low_values[np.isfinite(low_values)]
        high_values = high_values[np.isfinite(high_values)]
        
        lines.append(f"## {metric_label}")
        lines.append("")
        lines.append("| 统计量 | TM-score ≤ {:.1f} | TM-score > {:.1f} |".format(low_threshold, high_threshold))
        lines.append("| --- | --- | --- |")
        
        if len(low_values) > 0:
            low_mean = np.mean(low_values)
            low_std = np.std(low_values)
            low_median = np.median(low_values)
            low_min = np.min(low_values)
            low_max = np.max(low_values)
            low_q25 = np.percentile(low_values, 25)
            low_q75 = np.percentile(low_values, 75)
        else:
            low_mean = low_std = low_median = low_min = low_max = low_q25 = low_q75 = float('nan')
        
        if len(high_values) > 0:
            high_mean = np.mean(high_values)
            high_std = np.std(high_values)
            high_median = np.median(high_values)
            high_min = np.min(high_values)
            high_max = np.max(high_values)
            high_q25 = np.percentile(high_values, 25)
            high_q75 = np.percentile(high_values, 75)
        else:
            high_mean = high_std = high_median = high_min = high_max = high_q25 = high_q75 = float('nan')
        
        lines.append(f"| Mean | {low_mean:.4f} | {high_mean:.4f} |")
        lines.append(f"| Std | {low_std:.4f} | {high_std:.4f} |")
        lines.append(f"| Median | {low_median:.4f} | {high_median:.4f} |")
        lines.append(f"| Min | {low_min:.4f} | {high_min:.4f} |")
        lines.append(f"| Max | {low_max:.4f} | {high_max:.4f} |")
        lines.append(f"| Q25 | {low_q25:.4f} | {high_q25:.4f} |")
        lines.append(f"| Q75 | {low_q75:.4f} | {high_q75:.4f} |")
        lines.append("")
    
    output_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"已保存统计报告: {output_path}")


def main():
    args = parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析报告数据
    metrics = parse_report(args.report_path)
    if not metrics:
        print("错误: 未能解析到任何数据")
        return
    
    # 按TM-score分组
    low_group, high_group = split_by_tmscore(
        metrics, args.low_tm_threshold, args.high_tm_threshold
    )
    
    # 定义要绘制的度量
    metrics_to_plot = [
        ('rot_euclidean', 'Rotational Euclidean Distance'),
        ('rot_cosine_dist', 'Rotational Cosine Distance'),
        ('trans_euclidean', 'Translational Euclidean Distance'),
        ('trans_cosine_dist', 'Translational Cosine Distance'),
    ]
    
    print("\n正在绘制分布图...")
    
    # 为每种度量创建子目录
    for metric_name, metric_label in metrics_to_plot:
        metric_dir = args.output_dir / metric_name
        metric_dir.mkdir(exist_ok=True)
        
        # ========== 独立分组图 (每种度量分别绘制低/高TM-score组) ==========
        # 低TM-score组的独立图
        low_tm_path = metric_dir / f"{metric_name}_low_tmscore_le_{args.low_tm_threshold}.png"
        plot_single_distribution(
            low_group,
            metric_name, metric_label,
            tm_condition=f"≤ {args.low_tm_threshold}",
            tm_threshold=args.low_tm_threshold,
            output_path=low_tm_path,
            bins=args.bins,
            dpi=args.dpi,
            color='steelblue',
        )
        
        # 高TM-score组的独立图
        high_tm_path = metric_dir / f"{metric_name}_high_tmscore_gt_{args.high_tm_threshold}.png"
        plot_single_distribution(
            high_group,
            metric_name, metric_label,
            tm_condition=f"> {args.high_tm_threshold}",
            tm_threshold=args.high_tm_threshold,
            output_path=high_tm_path,
            bins=args.bins,
            dpi=args.dpi,
            color='coral',
        )
        
        # ========== 对比图 (保留原有功能) ==========
        # 并排对比图
        comparison_path = metric_dir / f"{metric_name}_comparison.png"
        plot_distribution_comparison(
            low_group, high_group,
            metric_name, metric_label,
            args.low_tm_threshold, args.high_tm_threshold,
            comparison_path,
            bins=args.bins,
            dpi=args.dpi,
        )
        
        # 叠加分布图
        overlay_path = metric_dir / f"{metric_name}_overlay.png"
        plot_overlay_distribution(
            low_group, high_group,
            metric_name, metric_label,
            args.low_tm_threshold, args.high_tm_threshold,
            overlay_path,
            bins=args.bins,
            dpi=args.dpi,
        )
    
    # 生成统计摘要
    summary_path = args.output_dir / "statistics_summary.md"
    generate_summary_stats(
        low_group, high_group,
        args.low_tm_threshold, args.high_tm_threshold,
        summary_path,
    )
    
    print(f"\n所有图表已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
