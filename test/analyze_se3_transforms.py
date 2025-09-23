#!/usr/bin/env python3
"""
SE(3)变换矩阵分析程序

分析从SE(3)扩散模型批处理结果生成的4x4齐次变换矩阵，
提供旋转、平移、时间演化等多维度的分析。

功能:
1. 验证变换矩阵的数学性质
2. 分析旋转和平移的统计特性
3. 研究时间演化模式
4. 生成可视化和报告
"""

import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import argparse

class SE3TransformAnalyzer:
    """SE(3)变换矩阵分析器"""
    
    def __init__(self, transform_dir):
        """
        初始化分析器
        
        Args:
            transform_dir: 变换矩阵文件目录
        """
        self.transform_dir = Path(transform_dir)
        self.transform_files = []
        self.time_steps = []
        self.loaded_data = {}
        
        print(f"初始化SE(3)变换矩阵分析器")
        print(f"数据目录: {self.transform_dir}")
        
    def load_transform_files(self):
        """加载所有变换矩阵文件"""
        # 找到所有变换矩阵文件
        pattern = "*_transforms.npy"
        self.transform_files = sorted(self.transform_dir.glob(pattern))
        
        if not self.transform_files:
            print("❌ 未找到变换矩阵文件")
            return False
        
        print(f"找到 {len(self.transform_files)} 个变换矩阵文件")
        
        # 提取时间步信息
        for file_path in self.transform_files:
            # 从文件名提取时间步: 1AKE_A_time_step_0.0100_transforms.npy
            filename = file_path.name
            if 'time_step_' in filename:
                time_str = filename.split('time_step_')[1].split('_transforms.npy')[0]
                try:
                    time_step = float(time_str)
                    self.time_steps.append(time_step)
                except ValueError:
                    print(f"警告: 无法从 {filename} 解析时间步")
        
        self.time_steps = np.array(self.time_steps)
        print(f"时间步范围: {self.time_steps.min():.4f} - {self.time_steps.max():.4f}")
        
        return True
    
    def load_data_sample(self, max_files=None):
        """加载数据样本用于分析"""
        files_to_load = self.transform_files
        if max_files:
            files_to_load = files_to_load[:max_files]
        
        print(f"加载 {len(files_to_load)} 个文件的数据...")
        
        for i, file_path in enumerate(files_to_load):
            try:
                data = np.load(file_path)
                self.loaded_data[self.time_steps[i]] = data
                print(f"  ✓ {file_path.name}: {data.shape}")
            except Exception as e:
                print(f"  ❌ {file_path.name}: {e}")
        
        return len(self.loaded_data) > 0
    
    def validate_transforms(self, tolerance=1e-6):
        """验证变换矩阵的数学性质"""
        print("\n" + "="*60)
        print("         变换矩阵数学性质验证")
        print("="*60)
        
        all_results = {
            'total_matrices': 0,
            'valid_rotations': 0,
            'invalid_rotations': 0,
            'determinant_errors': [],
            'orthogonality_errors': [],
            'bottom_row_errors': 0
        }
        
        for time_step, transforms in self.loaded_data.items():
            num_samples, num_residues, _, _ = transforms.shape
            
            for sample in range(num_samples):
                for residue in range(num_residues):
                    matrix = transforms[sample, residue]
                    all_results['total_matrices'] += 1
                    
                    # 检查旋转部分
                    rot_part = matrix[:3, :3]
                    
                    # 行列式检查
                    det = np.linalg.det(rot_part)
                    det_error = abs(det - 1.0)
                    all_results['determinant_errors'].append(det_error)
                    
                    # 正交性检查
                    ortho_matrix = rot_part @ rot_part.T
                    ortho_error = np.max(np.abs(ortho_matrix - np.eye(3)))
                    all_results['orthogonality_errors'].append(ortho_error)
                    
                    # 底部行检查
                    bottom_row = matrix[3, :]
                    expected_bottom = np.array([0, 0, 0, 1])
                    if not np.allclose(bottom_row, expected_bottom, atol=tolerance):
                        all_results['bottom_row_errors'] += 1
                    
                    # 整体有效性
                    if det_error < tolerance and ortho_error < tolerance:
                        all_results['valid_rotations'] += 1
                    else:
                        all_results['invalid_rotations'] += 1
        
        # 转换为numpy数组便于统计
        all_results['determinant_errors'] = np.array(all_results['determinant_errors'])
        all_results['orthogonality_errors'] = np.array(all_results['orthogonality_errors'])
        
        # 打印验证结果
        print(f"📊 验证统计:")
        print(f"   • 总变换矩阵数: {all_results['total_matrices']}")
        print(f"   • 有效旋转矩阵: {all_results['valid_rotations']}")
        print(f"   • 无效旋转矩阵: {all_results['invalid_rotations']}")
        print(f"   • 底部行错误: {all_results['bottom_row_errors']}")
        
        print(f"\n🎯 行列式误差分析:")
        det_errors = all_results['determinant_errors']
        print(f"   • 均值: {np.mean(det_errors):.2e}")
        print(f"   • 最大值: {np.max(det_errors):.2e}")
        print(f"   • 标准差: {np.std(det_errors):.2e}")
        print(f"   • >1e-6的数量: {np.sum(det_errors > 1e-6)}")
        
        print(f"\n⚖️ 正交性误差分析:")
        ortho_errors = all_results['orthogonality_errors']
        print(f"   • 均值: {np.mean(ortho_errors):.2e}")
        print(f"   • 最大值: {np.max(ortho_errors):.2e}")
        print(f"   • 标准差: {np.std(ortho_errors):.2e}")
        print(f"   • >1e-6的数量: {np.sum(ortho_errors > 1e-6)}")
        
        return all_results
    
    def analyze_rotations(self):
        """分析旋转特性"""
        print("\n" + "="*60)
        print("             旋转特性分析")
        print("="*60)
        
        rotation_stats = {
            'angles': [],
            'axes': [],
            'time_steps': []
        }
        
        for time_step, transforms in self.loaded_data.items():
            num_samples, num_residues, _, _ = transforms.shape
            
            for sample in range(num_samples):
                for residue in range(num_residues):
                    rot_matrix = transforms[sample, residue, :3, :3]
                    
                    # 转换为旋转对象以获取角度和轴
                    try:
                        rotation = Rotation.from_matrix(rot_matrix)
                        rotvec = rotation.as_rotvec()
                        
                        angle = np.linalg.norm(rotvec)
                        if angle > 1e-8:
                            axis = rotvec / angle
                        else:
                            axis = np.array([0, 0, 1])  # 默认轴
                        
                        rotation_stats['angles'].append(angle)
                        rotation_stats['axes'].append(axis)
                        rotation_stats['time_steps'].append(time_step)
                        
                    except Exception as e:
                        print(f"旋转分析失败: {e}")
                        continue
        
        # 转换为numpy数组
        angles = np.array(rotation_stats['angles'])
        axes = np.array(rotation_stats['axes'])
        times = np.array(rotation_stats['time_steps'])
        
        print(f"🔄 旋转角度统计:")
        print(f"   • 总旋转数: {len(angles)}")
        print(f"   • 平均角度: {np.mean(angles):.4f} 弧度 ({np.degrees(np.mean(angles)):.2f}°)")
        print(f"   • 角度标准差: {np.std(angles):.4f} 弧度")
        print(f"   • 最大角度: {np.max(angles):.4f} 弧度 ({np.degrees(np.max(angles)):.2f}°)")
        print(f"   • 最小角度: {np.min(angles):.4f} 弧度 ({np.degrees(np.min(angles)):.2f}°)")
        
        # 角度分布分析
        small_rotations = np.sum(angles < 0.1)  # <5.7度
        medium_rotations = np.sum((0.1 <= angles) & (angles < 0.5))  # 5.7-28.6度
        large_rotations = np.sum(angles >= 0.5)  # >28.6度
        
        print(f"\n📊 旋转幅度分布:")
        print(f"   • 小旋转 (<5.7°): {small_rotations} ({100*small_rotations/len(angles):.1f}%)")
        print(f"   • 中等旋转 (5.7°-28.6°): {medium_rotations} ({100*medium_rotations/len(angles):.1f}%)")
        print(f"   • 大旋转 (>28.6°): {large_rotations} ({100*large_rotations/len(angles):.1f}%)")
        
        # 旋转轴分布（计算主方向）
        print(f"\n🧭 旋转轴分布:")
        avg_axis = np.mean(axes, axis=0)
        avg_axis_norm = avg_axis / np.linalg.norm(avg_axis)
        print(f"   • 平均旋转轴方向: [{avg_axis_norm[0]:.3f}, {avg_axis_norm[1]:.3f}, {avg_axis_norm[2]:.3f}]")
        
        return {
            'angles': angles,
            'axes': axes,
            'time_steps': times,
            'statistics': {
                'mean_angle': np.mean(angles),
                'std_angle': np.std(angles),
                'max_angle': np.max(angles),
                'min_angle': np.min(angles)
            }
        }
    
    def analyze_translations(self):
        """分析平移特性"""
        print("\n" + "="*60)
        print("             平移特性分析")
        print("="*60)
        
        translation_stats = {
            'vectors': [],
            'magnitudes': [],
            'time_steps': []
        }
        
        for time_step, transforms in self.loaded_data.items():
            num_samples, num_residues, _, _ = transforms.shape
            
            for sample in range(num_samples):
                for residue in range(num_residues):
                    trans_vector = transforms[sample, residue, :3, 3]
                    magnitude = np.linalg.norm(trans_vector)
                    
                    translation_stats['vectors'].append(trans_vector)
                    translation_stats['magnitudes'].append(magnitude)
                    translation_stats['time_steps'].append(time_step)
        
        vectors = np.array(translation_stats['vectors'])
        magnitudes = np.array(translation_stats['magnitudes'])
        times = np.array(translation_stats['time_steps'])
        
        print(f"📍 平移向量统计:")
        print(f"   • 总平移数: {len(magnitudes)}")
        print(f"   • 平均幅度: {np.mean(magnitudes):.4f}")
        print(f"   • 幅度标准差: {np.std(magnitudes):.4f}")
        print(f"   • 最大幅度: {np.max(magnitudes):.4f}")
        print(f"   • 最小幅度: {np.min(magnitudes):.4f}")
        
        # 方向分析
        print(f"\n🧭 平移方向统计:")
        mean_vector = np.mean(vectors, axis=0)
        print(f"   • 平均平移向量: [{mean_vector[0]:.4f}, {mean_vector[1]:.4f}, {mean_vector[2]:.4f}]")
        print(f"   • 平均幅度: {np.linalg.norm(mean_vector):.4f}")
        
        # 各轴分量分析
        print(f"\n📐 各轴分量统计:")
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            axis_values = vectors[:, i]
            print(f"   • {axis_name}轴: 均值={np.mean(axis_values):.4f}, 标准差={np.std(axis_values):.4f}")
        
        return {
            'vectors': vectors,
            'magnitudes': magnitudes, 
            'time_steps': times,
            'statistics': {
                'mean_magnitude': np.mean(magnitudes),
                'std_magnitude': np.std(magnitudes),
                'max_magnitude': np.max(magnitudes),
                'min_magnitude': np.min(magnitudes)
            }
        }
    
    def analyze_time_evolution(self):
        """分析时间演化模式"""
        print("\n" + "="*60)
        print("            时间演化模式分析")
        print("="*60)
        
        if len(self.loaded_data) < 2:
            print("数据不足，无法进行时间演化分析")
            return None
        
        time_evolution = {
            'time_steps': [],
            'mean_rotation_angles': [],
            'mean_translation_magnitudes': [],
            'rotation_angle_stds': [],
            'translation_magnitude_stds': []
        }
        
        for time_step in sorted(self.loaded_data.keys()):
            transforms = self.loaded_data[time_step]
            num_samples, num_residues, _, _ = transforms.shape
            
            # 收集当前时间步的所有旋转角度和平移幅度
            rotation_angles = []
            translation_magnitudes = []
            
            for sample in range(num_samples):
                for residue in range(num_residues):
                    # 旋转角度
                    rot_matrix = transforms[sample, residue, :3, :3]
                    try:
                        rotation = Rotation.from_matrix(rot_matrix)
                        angle = np.linalg.norm(rotation.as_rotvec())
                        rotation_angles.append(angle)
                    except:
                        continue
                    
                    # 平移幅度
                    trans_vector = transforms[sample, residue, :3, 3]
                    magnitude = np.linalg.norm(trans_vector)
                    translation_magnitudes.append(magnitude)
            
            # 计算统计量
            time_evolution['time_steps'].append(time_step)
            time_evolution['mean_rotation_angles'].append(np.mean(rotation_angles))
            time_evolution['mean_translation_magnitudes'].append(np.mean(translation_magnitudes))
            time_evolution['rotation_angle_stds'].append(np.std(rotation_angles))
            time_evolution['translation_magnitude_stds'].append(np.std(translation_magnitudes))
        
        # 转换为numpy数组
        for key in time_evolution:
            time_evolution[key] = np.array(time_evolution[key])
        
        print(f"⏱️ 时间演化趋势:")
        print(f"   • 分析时间步数: {len(time_evolution['time_steps'])}")
        
        rot_angles = time_evolution['mean_rotation_angles']
        trans_mags = time_evolution['mean_translation_magnitudes']
        
        print(f"   • 旋转角度变化: {rot_angles[0]:.4f} → {rot_angles[-1]:.4f} 弧度")
        print(f"   • 平移幅度变化: {trans_mags[0]:.4f} → {trans_mags[-1]:.4f}")
        
        # 简单趋势分析
        if len(time_evolution['time_steps']) > 2:
            times = time_evolution['time_steps']
            rot_trend = (rot_angles[-1] - rot_angles[0]) / (times[-1] - times[0])
            trans_trend = (trans_mags[-1] - trans_mags[0]) / (times[-1] - times[0])
            
            print(f"   • 旋转角度趋势斜率: {rot_trend:.4f} 弧度/时间单位")
            print(f"   • 平移幅度趋势斜率: {trans_trend:.4f} /时间单位")
        
        return time_evolution
    
    def generate_report(self, output_file=None):
        """生成综合分析报告"""
        if output_file is None:
            output_file = self.transform_dir / "se3_transform_analysis_report.md"
        
        print(f"\n📋 生成分析报告: {output_file}")
        
        # 执行所有分析
        validation_results = self.validate_transforms()
        rotation_analysis = self.analyze_rotations()
        translation_analysis = self.analyze_translations()
        time_evolution = self.analyze_time_evolution()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# SE(3)变换矩阵分析报告\n\n")
            f.write(f"**生成时间**: {Path(__file__).stat().st_mtime}\n")
            f.write(f"**数据目录**: {self.transform_dir}\n")
            f.write(f"**分析文件数**: {len(self.loaded_data)}\n\n")
            
            # 数据概览
            f.write("## 📊 数据概览\n\n")
            total_matrices = validation_results['total_matrices']
            f.write(f"- **总变换矩阵数**: {total_matrices:,}\n")
            f.write(f"- **时间步数量**: {len(self.time_steps)}\n")
            f.write(f"- **时间步范围**: {self.time_steps.min():.4f} - {self.time_steps.max():.4f}\n")
            
            if self.loaded_data:
                sample_shape = list(self.loaded_data.values())[0].shape
                f.write(f"- **每时间步形状**: {sample_shape}\n")
            
            # 验证结果
            f.write("\n## ✅ 数学性质验证\n\n")
            f.write(f"- **有效旋转矩阵**: {validation_results['valid_rotations']:,} ({100*validation_results['valid_rotations']/total_matrices:.2f}%)\n")
            f.write(f"- **无效旋转矩阵**: {validation_results['invalid_rotations']:,}\n")
            f.write(f"- **底部行错误**: {validation_results['bottom_row_errors']}\n\n")
            
            det_errors = validation_results['determinant_errors']
            ortho_errors = validation_results['orthogonality_errors']
            f.write("### 行列式误差统计\n")
            f.write(f"- 均值: {np.mean(det_errors):.2e}\n")
            f.write(f"- 最大值: {np.max(det_errors):.2e}\n")
            f.write(f"- 超过1e-6的数量: {np.sum(det_errors > 1e-6)}\n\n")
            
            f.write("### 正交性误差统计\n")
            f.write(f"- 均值: {np.mean(ortho_errors):.2e}\n")
            f.write(f"- 最大值: {np.max(ortho_errors):.2e}\n")
            f.write(f"- 超过1e-6的数量: {np.sum(ortho_errors > 1e-6)}\n\n")
            
            # 旋转分析
            f.write("## 🔄 旋转特性分析\n\n")
            rot_stats = rotation_analysis['statistics']
            f.write(f"- **平均旋转角度**: {rot_stats['mean_angle']:.4f} 弧度 ({np.degrees(rot_stats['mean_angle']):.2f}°)\n")
            f.write(f"- **角度标准差**: {rot_stats['std_angle']:.4f} 弧度\n")
            f.write(f"- **最大角度**: {rot_stats['max_angle']:.4f} 弧度 ({np.degrees(rot_stats['max_angle']):.2f}°)\n")
            f.write(f"- **最小角度**: {rot_stats['min_angle']:.4f} 弧度 ({np.degrees(rot_stats['min_angle']):.2f}°)\n\n")
            
            # 平移分析
            f.write("## 📍 平移特性分析\n\n")
            trans_stats = translation_analysis['statistics']
            f.write(f"- **平均平移幅度**: {trans_stats['mean_magnitude']:.4f}\n")
            f.write(f"- **幅度标准差**: {trans_stats['std_magnitude']:.4f}\n")
            f.write(f"- **最大幅度**: {trans_stats['max_magnitude']:.4f}\n")
            f.write(f"- **最小幅度**: {trans_stats['min_magnitude']:.4f}\n\n")
            
            # 时间演化
            if time_evolution:
                f.write("## ⏱️ 时间演化分析\n\n")
                rot_angles = time_evolution['mean_rotation_angles']
                trans_mags = time_evolution['mean_translation_magnitudes']
                f.write(f"- **旋转角度变化**: {rot_angles[0]:.4f} → {rot_angles[-1]:.4f} 弧度\n")
                f.write(f"- **平移幅度变化**: {trans_mags[0]:.4f} → {trans_mags[-1]:.4f}\n")
                
                if len(time_evolution['time_steps']) > 2:
                    times = time_evolution['time_steps']
                    rot_trend = (rot_angles[-1] - rot_angles[0]) / (times[-1] - times[0])
                    trans_trend = (trans_mags[-1] - trans_mags[0]) / (times[-1] - times[0])
                    f.write(f"- **旋转趋势斜率**: {rot_trend:.4f} 弧度/时间单位\n")
                    f.write(f"- **平移趋势斜率**: {trans_trend:.4f} /时间单位\n")
            
            f.write("\n---\n")
            f.write("*报告由 SE3TransformAnalyzer 自动生成*\n")
        
        print(f"✅ 分析报告已保存到: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SE(3)变换矩阵分析程序')
    parser.add_argument('transform_dir', help='变换矩阵目录路径')
    parser.add_argument('--max-files', type=int, help='最大加载文件数量（用于大数据集）')
    parser.add_argument('--output-report', help='输出报告文件路径')
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = SE3TransformAnalyzer(args.transform_dir)
        
        # 加载数据
        if not analyzer.load_transform_files():
            return
        
        if not analyzer.load_data_sample(args.max_files):
            print("❌ 无法加载数据")
            return
        
        # 生成综合报告
        analyzer.generate_report(args.output_report)
        
        print(f"\n✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()