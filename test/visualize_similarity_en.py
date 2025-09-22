#!/usr/bin/env python3
"""
Protein Similarity Visualization Analysis Program

Reads similarity calculation results and generates visualization charts and deep analysis reports

Features:
1. Time evolution curve plots
2. Distance distribution histograms
3. Correlation analysis
4. Similarity heatmaps
5. Comprehensive analysis reports

Author: AI Assistant
Date: 2025-09-22
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
sns.set_palette("husl")

def load_similarity_data(data_file):
    """Load similarity data"""
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
    """Create time evolution plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Protein Structure Similarity Time Evolution Analysis', fontsize=16, fontweight='bold')
    
    time_steps = data['time_steps']
    
    # Euclidean distance plots
    axes[0, 0].plot(time_steps, data['euclidean_rotation'], 'o-', linewidth=2, markersize=4, label='Rotation Distance')
    axes[0, 0].set_title('Euclidean Distance - Rotation')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Euclidean Distance')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_steps, data['euclidean_translation'], 'o-', linewidth=2, markersize=4, color='orange', label='Translation Distance')
    axes[0, 1].set_title('Euclidean Distance - Translation')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Euclidean Distance')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(time_steps, data['euclidean_total'], 'o-', linewidth=2, markersize=4, color='red', label='Weighted Total Distance')
    axes[0, 2].set_title('Euclidean Distance - Weighted Total')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Euclidean Distance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Cosine distance plots
    axes[1, 0].plot(time_steps, data['cosine_rotation'], 'o-', linewidth=2, markersize=4, color='green', label='Rotation Distance')
    axes[1, 0].set_title('Cosine Distance - Rotation')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Cosine Distance')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_steps, data['cosine_translation'], 'o-', linewidth=2, markersize=4, color='purple', label='Translation Distance')
    axes[1, 1].set_title('Cosine Distance - Translation')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Cosine Distance')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(time_steps, data['cosine_total'], 'o-', linewidth=2, markersize=4, color='brown', label='Weighted Total Distance')
    axes[1, 2].set_title('Cosine Distance - Weighted Total')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Cosine Distance')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'time_evolution_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Time evolution plots saved to: {output_file}")

def create_comparison_plot(data, output_dir):
    """Create comparison plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Euclidean vs Cosine Distance Comparison', fontsize=14, fontweight='bold')
    
    time_steps = data['time_steps']
    
    # Rotation distance comparison
    axes[0].plot(time_steps, data['euclidean_rotation'], 'o-', linewidth=2, markersize=4, label='Euclidean Distance', alpha=0.8)
    axes[0].plot(time_steps, data['cosine_rotation'], 'o-', linewidth=2, markersize=4, label='Cosine Distance', alpha=0.8)
    axes[0].set_title('Rotation Distance Comparison')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Distance Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Translation distance comparison
    axes[1].plot(time_steps, data['euclidean_translation'], 'o-', linewidth=2, markersize=4, label='Euclidean Distance', alpha=0.8)
    axes[1].plot(time_steps, data['cosine_translation'], 'o-', linewidth=2, markersize=4, label='Cosine Distance', alpha=0.8)
    axes[1].set_title('Translation Distance Comparison')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Distance Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'distance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plots saved to: {output_file}")

def create_heatmap(data, output_dir):
    """Create similarity heatmap"""
    # Build matrix data
    matrix_data = np.array([
        data['euclidean_rotation'],
        data['euclidean_translation'], 
        data['euclidean_total'],
        data['cosine_rotation'],
        data['cosine_translation'],
        data['cosine_total']
    ])
    
    labels = ['Euclidean-Rotation', 'Euclidean-Translation', 'Euclidean-Total', 
              'Cosine-Rotation', 'Cosine-Translation', 'Cosine-Total']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(matrix_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(0, len(data['time_steps']), 5))
    ax.set_xticklabels([f"{data['time_steps'][i]:.3f}" for i in range(0, len(data['time_steps']), 5)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Distance Value', rotation=270, labelpad=20)
    
    # Set title and labels
    ax.set_title('Protein Similarity Distance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Distance Type')
    
    plt.tight_layout()
    output_file = output_dir / 'similarity_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved to: {output_file}")

def analyze_correlations(data):
    """Analyze correlations"""
    print("\n" + "="*60)
    print("         CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlations between different distance types
    correlations = {}
    distance_types = ['euclidean_rotation', 'euclidean_translation', 'euclidean_total',
                     'cosine_rotation', 'cosine_translation', 'cosine_total']
    
    for i, type1 in enumerate(distance_types):
        for j, type2 in enumerate(distance_types):
            if i < j:  # Avoid duplicate calculations
                corr = np.corrcoef(data[type1], data[type2])[0, 1]
                correlations[f"{type1}_vs_{type2}"] = corr
    
    print("📊 Inter-distance correlations:")
    for pair, corr in correlations.items():
        type1, type2 = pair.split('_vs_')
        print(f"   • {type1} vs {type2}: {corr:.4f}")
    
    # Correlation with time
    print(f"\n⏱️ Time step correlations:")
    for distance_type in distance_types:
        time_corr = np.corrcoef(data['time_steps'], data[distance_type])[0, 1]
        print(f"   • {distance_type}: {time_corr:.4f}")
    
    return correlations

def generate_insights(data):
    """Generate insights analysis"""
    print("\n" + "="*60)
    print("         DEEP INSIGHTS ANALYSIS")
    print("="*60)
    
    # Calculate change trends
    euclidean_rot_change = data['euclidean_rotation'][-1] - data['euclidean_rotation'][0]
    euclidean_trans_change = data['euclidean_translation'][-1] - data['euclidean_translation'][0]
    cosine_rot_change = data['cosine_rotation'][-1] - data['cosine_rotation'][0]
    cosine_trans_change = data['cosine_translation'][-1] - data['cosine_translation'][0]
    
    print(f"📈 Distance change trends:")
    print(f"   • Euclidean rotation distance change: {euclidean_rot_change:+.4f}")
    print(f"   • Euclidean translation distance change: {euclidean_trans_change:+.4f}")
    print(f"   • Cosine rotation distance change: {cosine_rot_change:+.4f}")
    print(f"   • Cosine translation distance change: {cosine_trans_change:+.4f}")
    
    # Find most and least similar time points
    euclidean_total = data['euclidean_total']
    cosine_total = data['cosine_total']
    
    euc_min_idx = np.argmin(euclidean_total)
    euc_max_idx = np.argmax(euclidean_total)
    cos_min_idx = np.argmin(cosine_total)
    cos_max_idx = np.argmax(cosine_total)
    
    print(f"\n🎯 Key time points:")
    print(f"   • Most similar (Euclidean): t={data['time_steps'][euc_min_idx]:.4f} (distance={euclidean_total[euc_min_idx]:.4f})")
    print(f"   • Least similar (Euclidean): t={data['time_steps'][euc_max_idx]:.4f} (distance={euclidean_total[euc_max_idx]:.4f})")
    print(f"   • Most similar (Cosine): t={data['time_steps'][cos_min_idx]:.4f} (distance={cosine_total[cos_min_idx]:.4f})")
    print(f"   • Least similar (Cosine): t={data['time_steps'][cos_max_idx]:.4f} (distance={cosine_total[cos_max_idx]:.4f})")
    
    # Calculate convergence/stability
    final_portion = len(data['time_steps']) // 4  # Last 25% of time steps
    euclidean_stability = np.std(euclidean_total[-final_portion:])
    cosine_stability = np.std(cosine_total[-final_portion:])
    
    print(f"\n📐 Late-stage stability (std dev of last 25% time steps):")
    print(f"   • Euclidean distance stability: {euclidean_stability:.6f}")
    print(f"   • Cosine distance stability: {cosine_stability:.6f}")

def create_additional_plots(data, output_dir):
    """Create additional analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Additional Analysis Plots', fontsize=16, fontweight='bold')
    
    time_steps = data['time_steps']
    
    # Plot 1: Distance ratios
    euclidean_ratio = data['euclidean_rotation'] / data['euclidean_translation']
    cosine_ratio = data['cosine_rotation'] / data['cosine_translation']
    
    axes[0, 0].plot(time_steps, euclidean_ratio, 'o-', linewidth=2, markersize=4, label='Euclidean Rot/Trans Ratio')
    axes[0, 0].plot(time_steps, cosine_ratio, 'o-', linewidth=2, markersize=4, label='Cosine Rot/Trans Ratio')
    axes[0, 0].set_title('Rotation to Translation Distance Ratios')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Normalized distances (z-scores)
    from scipy.stats import zscore
    euc_rot_norm = zscore(data['euclidean_rotation'])
    euc_trans_norm = zscore(data['euclidean_translation'])
    
    axes[0, 1].plot(time_steps, euc_rot_norm, 'o-', linewidth=2, markersize=4, label='Rotation (normalized)')
    axes[0, 1].plot(time_steps, euc_trans_norm, 'o-', linewidth=2, markersize=4, label='Translation (normalized)')
    axes[0, 1].set_title('Normalized Euclidean Distances')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Z-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distance derivatives (rate of change)
    euc_total_diff = np.diff(data['euclidean_total'])
    cos_total_diff = np.diff(data['cosine_total'])
    
    axes[1, 0].plot(time_steps[1:], euc_total_diff, 'o-', linewidth=2, markersize=4, label='Euclidean Rate of Change')
    axes[1, 0].plot(time_steps[1:], cos_total_diff, 'o-', linewidth=2, markersize=4, label='Cosine Rate of Change')
    axes[1, 0].set_title('Distance Rate of Change')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Rate of Change')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Moving averages
    window = 5
    if len(time_steps) >= window:
        euc_ma = np.convolve(data['euclidean_total'], np.ones(window)/window, mode='valid')
        cos_ma = np.convolve(data['cosine_total'], np.ones(window)/window, mode='valid')
        ma_time = time_steps[window-1:]
        
        axes[1, 1].plot(time_steps, data['euclidean_total'], 'o-', alpha=0.5, linewidth=1, markersize=2, label='Euclidean Raw')
        axes[1, 1].plot(ma_time, euc_ma, '-', linewidth=3, label=f'Euclidean MA({window})')
        axes[1, 1].plot(time_steps, data['cosine_total'], 'o-', alpha=0.5, linewidth=1, markersize=2, label='Cosine Raw')
        axes[1, 1].plot(ma_time, cos_ma, '-', linewidth=3, label=f'Cosine MA({window})')
        axes[1, 1].set_title(f'Moving Averages (window={window})')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor moving average', 
                       transform=axes[1, 1].transAxes, ha='center', va='center')
        axes[1, 1].set_title('Moving Averages (Insufficient Data)')
    
    plt.tight_layout()
    output_file = output_dir / 'additional_analysis_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Additional analysis plots saved to: {output_file}")

def create_comprehensive_report(data, correlations, output_dir, protein1, protein2):
    """Create comprehensive analysis report"""
    report_file = output_dir / f"{protein1}_vs_{protein2}_comprehensive_analysis.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Protein Structure Similarity Comprehensive Analysis Report\n\n")
        f.write(f"**Analysis Subject**: {protein1} vs {protein2}\n")
        f.write(f"**Analysis Date**: 2025-09-22\n")
        f.write(f"**Number of Time Steps**: {len(data['time_steps'])}\n")
        f.write(f"**Time Step Range**: {data['time_steps'].min():.4f} - {data['time_steps'].max():.4f}\n\n")
        
        f.write("## 📊 Key Findings\n\n")
        
        # Distance changes
        euclidean_rot_change = data['euclidean_rotation'][-1] - data['euclidean_rotation'][0]
        euclidean_trans_change = data['euclidean_translation'][-1] - data['euclidean_translation'][0]
        cosine_rot_change = data['cosine_rotation'][-1] - data['cosine_rotation'][0]
        cosine_trans_change = data['cosine_translation'][-1] - data['cosine_translation'][0]
        
        f.write(f"### Distance Change Trends\n")
        f.write(f"- **Euclidean Rotation Distance**: {data['euclidean_rotation'][0]:.4f} → {data['euclidean_rotation'][-1]:.4f} (change: {euclidean_rot_change:+.4f})\n")
        f.write(f"- **Euclidean Translation Distance**: {data['euclidean_translation'][0]:.4f} → {data['euclidean_translation'][-1]:.4f} (change: {euclidean_trans_change:+.4f})\n")
        f.write(f"- **Cosine Rotation Distance**: {data['cosine_rotation'][0]:.4f} → {data['cosine_rotation'][-1]:.4f} (change: {cosine_rot_change:+.4f})\n")
        f.write(f"- **Cosine Translation Distance**: {data['cosine_translation'][0]:.4f} → {data['cosine_translation'][-1]:.4f} (change: {cosine_trans_change:+.4f})\n\n")
        
        # Key time points
        euclidean_total = data['euclidean_total']
        cosine_total = data['cosine_total']
        euc_min_idx = np.argmin(euclidean_total)
        euc_max_idx = np.argmax(euclidean_total)
        cos_min_idx = np.argmin(cosine_total)
        cos_max_idx = np.argmax(cosine_total)
        
        f.write(f"### Key Time Points\n")
        f.write(f"- **Most similar (Euclidean)**: t={data['time_steps'][euc_min_idx]:.4f} (distance={euclidean_total[euc_min_idx]:.4f})\n")
        f.write(f"- **Least similar (Euclidean)**: t={data['time_steps'][euc_max_idx]:.4f} (distance={euclidean_total[euc_max_idx]:.4f})\n")
        f.write(f"- **Most similar (Cosine)**: t={data['time_steps'][cos_min_idx]:.4f} (distance={cosine_total[cos_min_idx]:.4f})\n")
        f.write(f"- **Least similar (Cosine)**: t={data['time_steps'][cos_max_idx]:.4f} (distance={cosine_total[cos_max_idx]:.4f})\n\n")
        
        # Statistical summary
        f.write(f"### Statistical Summary\n")
        for distance_type in ['euclidean_rotation', 'euclidean_translation', 'euclidean_total',
                             'cosine_rotation', 'cosine_translation', 'cosine_total']:
            values = data[distance_type]
            f.write(f"- **{distance_type}**: mean={np.mean(values):.6f}, std={np.std(values):.6f}, range=[{np.min(values):.6f}, {np.max(values):.6f}]\n")
        
        f.write(f"\n### Correlation Analysis\n")
        for pair, corr in correlations.items():
            f.write(f"- **{pair.replace('_vs_', ' vs ')}**: {corr:.4f}\n")
        
        f.write(f"\n## 📈 Generated Charts\n")
        f.write(f"- `time_evolution_plots.png`: Time evolution analysis plots\n")
        f.write(f"- `distance_comparison.png`: Distance type comparison plots\n")
        f.write(f"- `similarity_heatmap.png`: Similarity heatmap\n")
        f.write(f"- `additional_analysis_plots.png`: Additional analysis plots\n")
        
        f.write(f"\n---\n*Report automatically generated by Protein Similarity Visualization Analysis Program*\n")
    
    print(f"✓ Comprehensive analysis report saved to: {report_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Protein Similarity Visualization Analysis Program')
    parser.add_argument('data_file', help='Similarity data file path (.npz)')
    parser.add_argument('--output-dir', help='Output directory path')
    parser.add_argument('--protein1', default='Protein1', help='First protein name')
    parser.add_argument('--protein2', default='Protein2', help='Second protein name')
    
    args = parser.parse_args()
    
    try:
        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.data_file).parent / "visualization"
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"Protein Similarity Visualization Analysis")
        print(f"Data file: {args.data_file}")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        # Load data
        data = load_similarity_data(args.data_file)
        
        # Create visualizations
        print("📊 Generating visualization charts...")
        create_time_evolution_plot(data, output_dir)
        create_comparison_plot(data, output_dir)
        create_heatmap(data, output_dir)
        create_additional_plots(data, output_dir)
        
        # Analyze correlations
        correlations = analyze_correlations(data)
        
        # Generate insights
        generate_insights(data)
        
        # Create comprehensive report
        create_comprehensive_report(data, correlations, output_dir, args.protein1, args.protein2)
        
        print(f"\n✅ Visualization analysis complete!")
        print(f"All files saved in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()