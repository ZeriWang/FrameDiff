#!/usr/bin/env python3
"""
蛋白质结构相似度计算程序

计算同一时间步下两个蛋白质结构的相似度，支持欧氏距离和余弦距离两种度量方法。
分别计算旋转距离、平移距离，并进行加权求和得到总距离。

功能特性:
1. 支持欧氏距离和余弦距离度量
2. 分别计算旋转和平移的相似度
3. 可配置的权重组合
4. 详细的统计分析和报告
"""

import os
import numpy as np
from pathlib import Path
import argparse
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ProteinSimilarityCalculator:
    """蛋白质结构相似度计算器"""
    
    def __init__(self, data_dir, protein1_prefix, protein2_prefix):
        """
        初始化相似度计算器
        
        Args:
            data_dir: 数据目录路径
            protein1_prefix: 第一个蛋白质的文件前缀 (如 "1AKE_A")
            protein2_prefix: 第二个蛋白质的文件前缀 (如 "4AKE_A")
        """
        self.data_dir = Path(data_dir)
        self.protein1_prefix = protein1_prefix
        self.protein2_prefix = protein2_prefix
        
        # 数据容器
        self.protein1_data = {}
        self.protein2_data = {}
        self.time_steps = None
        
        print(f"初始化蛋白质相似度计算器")
        print(f"数据目录: {self.data_dir}")
        print(f"蛋白质1: {protein1_prefix}")
        print(f"蛋白质2: {protein2_prefix}")
        
    def load_protein_data(self):
        """加载两个蛋白质的数据"""
        print("\n" + "="*60)
        print("         加载蛋白质数据")
        print("="*60)
        
        # 定义需要加载的文件
        file_patterns = {
            'all_rot_scores': '{}_all_rot_scores.npy',
            'all_trans_scores': '{}_all_trans_scores.npy',
            'time_steps': '{}_time_steps.npy'
        }
        
        # 加载蛋白质1数据
        print(f"加载 {self.protein1_prefix} 数据:")
        for key, pattern in file_patterns.items():
            filename = pattern.format(self.protein1_prefix)
            filepath = self.data_dir / filename
            
            if filepath.exists():
                data = np.load(filepath)
                self.protein1_data[key] = data
                print(f"  ✓ {filename}: {data.shape}")
            else:
                print(f"  ❌ 未找到文件: {filename}")
                return False
        
        # 加载蛋白质2数据
        print(f"\n加载 {self.protein2_prefix} 数据:")
        for key, pattern in file_patterns.items():
            filename = pattern.format(self.protein2_prefix)
            filepath = self.data_dir / filename
            
            if filepath.exists():
                data = np.load(filepath)
                self.protein2_data[key] = data
                print(f"  ✓ {filename}: {data.shape}")
            else:
                print(f"  ❌ 未找到文件: {filename}")
                return False
        
        # 验证数据一致性
        return self._validate_data_consistency()
    
    def _validate_data_consistency(self):
        """验证两个蛋白质数据的一致性"""
        print(f"\n🔍 验证数据一致性:")
        
        # 检查形状一致性
        rot1_shape = self.protein1_data['all_rot_scores'].shape
        rot2_shape = self.protein2_data['all_rot_scores'].shape
        trans1_shape = self.protein1_data['all_trans_scores'].shape
        trans2_shape = self.protein2_data['all_trans_scores'].shape
        
        if rot1_shape != rot2_shape:
            print(f"  ❌ 旋转scores形状不一致: {rot1_shape} vs {rot2_shape}")
            return False
        
        if trans1_shape != trans2_shape:
            print(f"  ❌ 平移scores形状不一致: {trans1_shape} vs {trans2_shape}")
            return False
        
        # 检查时间步一致性
        time1 = self.protein1_data['time_steps']
        time2 = self.protein2_data['time_steps']
        
        if not np.allclose(time1, time2):
            print(f"  ❌ 时间步不一致")
            return False
        
        self.time_steps = time1
        
        print(f"  ✅ 数据形状一致: {rot1_shape}")
        print(f"  ✅ 时间步一致: {len(self.time_steps)} 个时间步")
        print(f"  ✅ 时间步范围: {self.time_steps.min():.4f} - {self.time_steps.max():.4f}")
        
        return True
    
    def calculate_euclidean_distance(self, vec1, vec2):
        """
        计算欧氏距离
        
        Args:
            vec1, vec2: 形状为 (..., feature_dim) 的数组
            
        Returns:
            距离数组
        """
        return np.linalg.norm(vec1 - vec2, axis=-1)
    
    def calculate_cosine_distance(self, vec1, vec2):
        """
        计算余弦距离
        
        Args:
            vec1, vec2: 形状为 (..., feature_dim) 的数组
            
        Returns:
            距离数组 (1 - cosine_similarity)
        """
        # 将数据重塑为2D以便计算余弦相似度
        original_shape = vec1.shape[:-1]  # 除了最后一维的所有维度
        feature_dim = vec1.shape[-1]
        
        vec1_flat = vec1.reshape(-1, feature_dim)
        vec2_flat = vec2.reshape(-1, feature_dim)
        
        # 计算余弦距离
        cosine_distances = []
        for i in range(len(vec1_flat)):
            v1 = vec1_flat[i]
            v2 = vec2_flat[i]
            
            # 处理零向量情况
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                # 如果任一向量接近零向量，使用欧氏距离归一化
                if norm1 < 1e-8 and norm2 < 1e-8:
                    cosine_dist = 0.0  # 两个零向量相似度为1，距离为0
                else:
                    cosine_dist = 1.0  # 零向量与非零向量距离为1
            else:
                # 计算余弦相似度
                cos_sim = np.dot(v1, v2) / (norm1 * norm2)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 防止数值误差
                cosine_dist = 1.0 - cos_sim  # 转换为距离
            
            cosine_distances.append(cosine_dist)
        
        # 重塑回原始形状
        cosine_distances = np.array(cosine_distances).reshape(original_shape)
        return cosine_distances
    
    def calculate_timestep_similarity(self, time_index, distance_type='euclidean'):
        """
        计算指定时间步的相似度
        
        Args:
            time_index: 时间步索引
            distance_type: 距离类型 ('euclidean' 或 'cosine')
            
        Returns:
            相似度结果字典
        """
        # 获取指定时间步的数据
        rot1 = self.protein1_data['all_rot_scores'][time_index]  # (num_samples, num_residues, 3)
        rot2 = self.protein2_data['all_rot_scores'][time_index]
        trans1 = self.protein1_data['all_trans_scores'][time_index]  # (num_samples, num_residues, 3)
        trans2 = self.protein2_data['all_trans_scores'][time_index]
        
        # 计算距离
        if distance_type == 'euclidean':
            rot_distances = self.calculate_euclidean_distance(rot1, rot2)
            trans_distances = self.calculate_euclidean_distance(trans1, trans2)
        elif distance_type == 'cosine':
            rot_distances = self.calculate_cosine_distance(rot1, rot2)
            trans_distances = self.calculate_cosine_distance(trans1, trans2)
        else:
            raise ValueError(f"不支持的距离类型: {distance_type}")
        
        # 计算统计信息
        results = {
            'time_step': self.time_steps[time_index],
            'time_index': time_index,
            'distance_type': distance_type,
            'rotation_distances': rot_distances,  # (num_samples, num_residues)
            'translation_distances': trans_distances,  # (num_samples, num_residues)
            'rotation_stats': {
                'mean': np.mean(rot_distances),
                'std': np.std(rot_distances),
                'min': np.min(rot_distances),
                'max': np.max(rot_distances),
                'median': np.median(rot_distances)
            },
            'translation_stats': {
                'mean': np.mean(trans_distances),
                'std': np.std(trans_distances),
                'min': np.min(trans_distances),
                'max': np.max(trans_distances),
                'median': np.median(trans_distances)
            }
        }
        
        return results
    
    def calculate_weighted_total_distance(self, rot_distances, trans_distances, 
                                        rot_weight=0.5, trans_weight=0.5):
        """
        计算加权总距离
        
        Args:
            rot_distances: 旋转距离数组
            trans_distances: 平移距离数组
            rot_weight: 旋转权重
            trans_weight: 平移权重
            
        Returns:
            加权总距离数组
        """
        # 归一化权重
        total_weight = rot_weight + trans_weight
        rot_weight_norm = rot_weight / total_weight
        trans_weight_norm = trans_weight / total_weight
        
        # 计算加权总距离
        total_distances = (rot_weight_norm * rot_distances + 
                          trans_weight_norm * trans_distances)
        
        return total_distances
    
    def analyze_all_timesteps(self, rot_weight=0.5, trans_weight=0.5):
        """
        分析所有时间步的相似度
        
        Args:
            rot_weight: 旋转权重
            trans_weight: 平移权重
            
        Returns:
            完整分析结果
        """
        print("\n" + "="*60)
        print("         分析所有时间步相似度")
        print("="*60)
        
        print(f"权重设置: 旋转={rot_weight}, 平移={trans_weight}")
        
        # 存储所有结果
        all_results = {
            'euclidean': [],
            'cosine': [],
            'weighted_euclidean': [],
            'weighted_cosine': [],
            'time_steps': self.time_steps.copy()
        }
        
        num_timesteps = len(self.time_steps)
        
        for time_idx in range(num_timesteps):
            time_step = self.time_steps[time_idx]
            
            # 计算欧氏距离
            euclidean_results = self.calculate_timestep_similarity(time_idx, 'euclidean')
            all_results['euclidean'].append(euclidean_results)
            
            # 计算余弦距离
            cosine_results = self.calculate_timestep_similarity(time_idx, 'cosine')
            all_results['cosine'].append(cosine_results)
            
            # 计算加权总距离
            euclidean_total = self.calculate_weighted_total_distance(
                euclidean_results['rotation_distances'],
                euclidean_results['translation_distances'],
                rot_weight, trans_weight
            )
            
            cosine_total = self.calculate_weighted_total_distance(
                cosine_results['rotation_distances'],
                cosine_results['translation_distances'],
                rot_weight, trans_weight
            )
            
            # 存储加权结果
            euclidean_weighted_stats = {
                'time_step': time_step,
                'time_index': time_idx,
                'total_distances': euclidean_total,
                'stats': {
                    'mean': np.mean(euclidean_total),
                    'std': np.std(euclidean_total),
                    'min': np.min(euclidean_total),
                    'max': np.max(euclidean_total),
                    'median': np.median(euclidean_total)
                }
            }
            
            cosine_weighted_stats = {
                'time_step': time_step,
                'time_index': time_idx,
                'total_distances': cosine_total,
                'stats': {
                    'mean': np.mean(cosine_total),
                    'std': np.std(cosine_total),
                    'min': np.min(cosine_total),
                    'max': np.max(cosine_total),
                    'median': np.median(cosine_total)
                }
            }
            
            all_results['weighted_euclidean'].append(euclidean_weighted_stats)
            all_results['weighted_cosine'].append(cosine_weighted_stats)
            
            # 显示进度
            if (time_idx + 1) % 5 == 0 or time_idx == num_timesteps - 1:
                print(f"已处理 {time_idx + 1}/{num_timesteps} 个时间步")
        
        return all_results
    
    def print_summary_statistics(self, results):
        """打印汇总统计信息"""
        print("\n" + "="*60)
        print("         相似度分析汇总统计")
        print("="*60)
        
        # 提取时间演化数据
        euclidean_rot_means = [r['rotation_stats']['mean'] for r in results['euclidean']]
        euclidean_trans_means = [r['translation_stats']['mean'] for r in results['euclidean']]
        euclidean_total_means = [r['stats']['mean'] for r in results['weighted_euclidean']]
        
        cosine_rot_means = [r['rotation_stats']['mean'] for r in results['cosine']]
        cosine_trans_means = [r['translation_stats']['mean'] for r in results['cosine']]
        cosine_total_means = [r['stats']['mean'] for r in results['weighted_cosine']]
        
        print(f"📊 欧氏距离统计:")
        print(f"   • 旋转距离均值: {np.mean(euclidean_rot_means):.6f} ± {np.std(euclidean_rot_means):.6f}")
        print(f"   • 平移距离均值: {np.mean(euclidean_trans_means):.6f} ± {np.std(euclidean_trans_means):.6f}")
        print(f"   • 加权总距离均值: {np.mean(euclidean_total_means):.6f} ± {np.std(euclidean_total_means):.6f}")
        
        print(f"\n📊 余弦距离统计:")
        print(f"   • 旋转距离均值: {np.mean(cosine_rot_means):.6f} ± {np.std(cosine_rot_means):.6f}")
        print(f"   • 平移距离均值: {np.mean(cosine_trans_means):.6f} ± {np.std(cosine_trans_means):.6f}")
        print(f"   • 加权总距离均值: {np.mean(cosine_total_means):.6f} ± {np.std(cosine_total_means):.6f}")
        
        # 时间演化趋势
        print(f"\n⏱️ 时间演化趋势:")
        time_steps = results['time_steps']
        
        print(f"   欧氏距离变化:")
        print(f"     - 旋转: {euclidean_rot_means[0]:.6f} → {euclidean_rot_means[-1]:.6f}")
        print(f"     - 平移: {euclidean_trans_means[0]:.6f} → {euclidean_trans_means[-1]:.6f}")
        print(f"     - 总距离: {euclidean_total_means[0]:.6f} → {euclidean_total_means[-1]:.6f}")
        
        print(f"   余弦距离变化:")
        print(f"     - 旋转: {cosine_rot_means[0]:.6f} → {cosine_rot_means[-1]:.6f}")
        print(f"     - 平移: {cosine_trans_means[0]:.6f} → {cosine_trans_means[-1]:.6f}")
        print(f"     - 总距离: {cosine_total_means[0]:.6f} → {cosine_total_means[-1]:.6f}")
    
    def save_results(self, results, output_dir=None, rot_weight=0.5, trans_weight=0.5):
        """保存计算结果"""
        if output_dir is None:
            output_dir = self.data_dir / "similarity_analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 保存结果到: {output_dir}")
        
        # 生成文件名前缀
        prefix = f"{self.protein1_prefix}_vs_{self.protein2_prefix}_w{rot_weight}-{trans_weight}"
        
        # 保存时间演化数据
        time_evolution_data = {
            'time_steps': results['time_steps'],
            'euclidean_rotation_means': [r['rotation_stats']['mean'] for r in results['euclidean']],
            'euclidean_translation_means': [r['translation_stats']['mean'] for r in results['euclidean']],
            'euclidean_total_means': [r['stats']['mean'] for r in results['weighted_euclidean']],
            'cosine_rotation_means': [r['rotation_stats']['mean'] for r in results['cosine']],
            'cosine_translation_means': [r['translation_stats']['mean'] for r in results['cosine']],
            'cosine_total_means': [r['stats']['mean'] for r in results['weighted_cosine']]
        }
        
        # 保存为numpy文件
        np.savez(
            output_dir / f"{prefix}_time_evolution.npz",
            **time_evolution_data
        )
        
        # 保存详细统计报告
        report_file = output_dir / f"{prefix}_similarity_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"蛋白质结构相似度分析报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"蛋白质1: {self.protein1_prefix}\n")
            f.write(f"蛋白质2: {self.protein2_prefix}\n")
            f.write(f"权重设置: 旋转={rot_weight}, 平移={trans_weight}\n")
            f.write(f"时间步数量: {len(results['time_steps'])}\n")
            f.write(f"时间步范围: {results['time_steps'].min():.4f} - {results['time_steps'].max():.4f}\n\n")
            
            # 写入详细统计
            euclidean_rot_means = time_evolution_data['euclidean_rotation_means']
            euclidean_trans_means = time_evolution_data['euclidean_translation_means']
            euclidean_total_means = time_evolution_data['euclidean_total_means']
            cosine_rot_means = time_evolution_data['cosine_rotation_means']
            cosine_trans_means = time_evolution_data['cosine_translation_means']
            cosine_total_means = time_evolution_data['cosine_total_means']
            
            f.write("欧氏距离统计:\n")
            f.write(f"  旋转距离均值: {np.mean(euclidean_rot_means):.6f} ± {np.std(euclidean_rot_means):.6f}\n")
            f.write(f"  平移距离均值: {np.mean(euclidean_trans_means):.6f} ± {np.std(euclidean_trans_means):.6f}\n")
            f.write(f"  加权总距离均值: {np.mean(euclidean_total_means):.6f} ± {np.std(euclidean_total_means):.6f}\n\n")
            
            f.write("余弦距离统计:\n")
            f.write(f"  旋转距离均值: {np.mean(cosine_rot_means):.6f} ± {np.std(cosine_rot_means):.6f}\n")
            f.write(f"  平移距离均值: {np.mean(cosine_trans_means):.6f} ± {np.std(cosine_trans_means):.6f}\n")
            f.write(f"  加权总距离均值: {np.mean(cosine_total_means):.6f} ± {np.std(cosine_total_means):.6f}\n\n")
            
            # 写入逐时间步详细数据
            f.write("逐时间步详细数据:\n")
            f.write("-" * 50 + "\n")
            f.write("时间步\t欧氏旋转\t欧氏平移\t欧氏总计\t余弦旋转\t余弦平移\t余弦总计\n")
            
            for i, t in enumerate(results['time_steps']):
                f.write(f"{t:.4f}\t{euclidean_rot_means[i]:.6f}\t{euclidean_trans_means[i]:.6f}\t{euclidean_total_means[i]:.6f}\t")
                f.write(f"{cosine_rot_means[i]:.6f}\t{cosine_trans_means[i]:.6f}\t{cosine_total_means[i]:.6f}\n")
        
        print(f"  ✓ 时间演化数据: {prefix}_time_evolution.npz")
        print(f"  ✓ 详细报告: {prefix}_similarity_report.txt")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='蛋白质结构相似度计算程序')
    parser.add_argument('data_dir', help='数据目录路径')
    parser.add_argument('protein1', help='第一个蛋白质前缀 (如 1AKE_A)')
    parser.add_argument('protein2', help='第二个蛋白质前缀 (如 4AKE_A)')
    parser.add_argument('--rot-weight', type=float, default=0.5, help='旋转权重 (默认: 0.5)')
    parser.add_argument('--trans-weight', type=float, default=0.5, help='平移权重 (默认: 0.5)')
    parser.add_argument('--output-dir', help='输出目录路径')
    
    args = parser.parse_args()
    
    try:
        # 创建相似度计算器
        calculator = ProteinSimilarityCalculator(
            args.data_dir, args.protein1, args.protein2
        )
        
        # 加载数据
        if not calculator.load_protein_data():
            print("❌ 数据加载失败")
            return
        
        # 执行分析
        results = calculator.analyze_all_timesteps(
            args.rot_weight, args.trans_weight
        )
        
        # 打印汇总统计
        calculator.print_summary_statistics(results)
        
        # 保存结果
        calculator.save_results(
            results, args.output_dir, args.rot_weight, args.trans_weight
        )
        
        print(f"\n✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()