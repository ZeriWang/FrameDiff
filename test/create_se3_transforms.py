#!/usr/bin/env python3
"""
SE(3)变换矩阵构建程序

将批处理结果中的旋转scores和平移scores重组为4x4齐次变换矩阵。

功能:
1. 从旋转scores(3维旋转向量)转换为旋转矩阵(3x3)
2. 与平移scores(3维向量)组合成4x4齐次变换矩阵
3. 为每个时间步和样本生成变换矩阵

作者: AI Assistant
日期: 2025-09-22
"""

import os
import numpy as np
import torch
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation

def rodrigues_rotation_matrix(rot_vec):
    """
    使用Rodrigues公式将旋转向量转换为旋转矩阵
    
    Args:
        rot_vec: (N, 3) 旋转向量数组
    
    Returns:
        rotation_matrices: (N, 3, 3) 旋转矩阵数组
    """
    # 计算旋转角度
    angles = np.linalg.norm(rot_vec, axis=-1, keepdims=True)
    
    # 处理零旋转情况
    small_angle_mask = (angles < 1e-8).squeeze()
    
    # 单位化旋转轴
    safe_angles = np.where(angles < 1e-8, 1.0, angles)
    axes = rot_vec / safe_angles
    
    # 对于较小角度，使用单位矩阵
    rotation_matrices = np.eye(3)[None, :, :].repeat(rot_vec.shape[0], axis=0)
    
    # 对于非零旋转，使用Rodrigues公式
    if not np.all(small_angle_mask):
        large_angle_indices = ~small_angle_mask
        if np.any(large_angle_indices):
            angles_large = angles[large_angle_indices].squeeze()
            axes_large = axes[large_angle_indices]
            
            cos_angles = np.cos(angles_large)
            sin_angles = np.sin(angles_large)
            
            # 构建反对称矩阵 [v]×
            K = np.zeros((len(angles_large), 3, 3))
            K[:, 0, 1] = -axes_large[:, 2]
            K[:, 0, 2] = axes_large[:, 1]
            K[:, 1, 0] = axes_large[:, 2]
            K[:, 1, 2] = -axes_large[:, 0]
            K[:, 2, 0] = -axes_large[:, 1]
            K[:, 2, 1] = axes_large[:, 0]
            
            # Rodrigues公式: R = I + sin(θ)[v]× + (1-cos(θ))[v]×²
            I = np.eye(3)[None, :, :].repeat(len(angles_large), axis=0)
            K_squared = np.matmul(K, K)
            
            rotation_matrices[large_angle_indices] = (
                I + 
                sin_angles[:, None, None] * K + 
                (1 - cos_angles)[:, None, None] * K_squared
            )
    
    return rotation_matrices

def scipy_rotation_matrix(rot_vec):
    """
    使用scipy的Rotation类将旋转向量转换为旋转矩阵
    
    Args:
        rot_vec: (N, 3) 旋转向量数组
    
    Returns:
        rotation_matrices: (N, 3, 3) 旋转矩阵数组
    """
    # 处理零向量情况
    norms = np.linalg.norm(rot_vec, axis=-1)
    zero_mask = norms < 1e-8
    
    if np.all(zero_mask):
        # 所有向量都是零向量，返回单位矩阵
        return np.tile(np.eye(3), (rot_vec.shape[0], 1, 1))
    
    # 对于非零向量使用scipy
    rotation_matrices = np.zeros((rot_vec.shape[0], 3, 3))
    
    if np.any(~zero_mask):
        non_zero_vecs = rot_vec[~zero_mask]
        if len(non_zero_vecs) > 0:
            rotation_matrices[~zero_mask] = Rotation.from_rotvec(non_zero_vecs).as_matrix()
    
    # 零向量对应单位矩阵
    rotation_matrices[zero_mask] = np.eye(3)
    
    return rotation_matrices

def create_homogeneous_transform_matrix(rot_matrix, trans_vec):
    """
    创建4x4齐次变换矩阵
    
    Args:
        rot_matrix: (N, 3, 3) 旋转矩阵
        trans_vec: (N, 3) 平移向量
    
    Returns:
        transform_matrix: (N, 4, 4) 齐次变换矩阵
    """
    N = rot_matrix.shape[0]
    transform_matrix = np.zeros((N, 4, 4))
    
    # 填入旋转矩阵 (左上角 3x3)
    transform_matrix[:, :3, :3] = rot_matrix
    
    # 填入平移向量 (右上角 3x1)
    transform_matrix[:, :3, 3] = trans_vec
    
    # 填入底部行 [0, 0, 0, 1]
    transform_matrix[:, 3, 3] = 1.0
    
    return transform_matrix

def process_single_timestep(rot_scores, trans_scores, output_dir, timestep_name, method='scipy'):
    """
    处理单个时间步的数据
    
    Args:
        rot_scores: (num_samples, num_residues, 3) 旋转scores
        trans_scores: (num_samples, num_residues, 3) 平移scores
        output_dir: 输出目录
        timestep_name: 时间步名称
        method: 旋转矩阵转换方法 ('scipy' 或 'rodrigues')
    """
    num_samples, num_residues, _ = rot_scores.shape
    
    print(f"处理时间步 {timestep_name}: {num_samples} 个样本, {num_residues} 个残基")
    
    # 为每个样本处理
    all_transforms = []
    
    for sample_idx in range(num_samples):
        sample_rot_scores = rot_scores[sample_idx]  # (num_residues, 3)
        sample_trans_scores = trans_scores[sample_idx]  # (num_residues, 3)
        
        # 转换旋转向量为旋转矩阵
        if method == 'scipy':
            rotation_matrices = scipy_rotation_matrix(sample_rot_scores)
        else:
            rotation_matrices = rodrigues_rotation_matrix(sample_rot_scores)
        
        # 创建齐次变换矩阵
        transform_matrices = create_homogeneous_transform_matrix(
            rotation_matrices, sample_trans_scores
        )
        
        all_transforms.append(transform_matrices)
    
    # 堆叠所有样本 (num_samples, num_residues, 4, 4)
    all_transforms = np.stack(all_transforms, axis=0)
    
    # 保存结果
    output_file = output_dir / f"{timestep_name}_transform_matrices.npy"
    np.save(output_file, all_transforms)
    
    print(f"  ✓ 保存变换矩阵到: {output_file}")
    print(f"  形状: {all_transforms.shape}")
    
    # 保存统计信息
    stats_file = output_dir / f"{timestep_name}_transform_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"时间步 {timestep_name} 变换矩阵统计信息\n")
        f.write("="*50 + "\n")
        f.write(f"数据形状: {all_transforms.shape}\n")
        f.write(f"样本数: {num_samples}\n")
        f.write(f"残基数: {num_residues}\n")
        f.write(f"转换方法: {method}\n\n")
        
        # 计算旋转矩阵的行列式统计
        det_values = np.linalg.det(all_transforms[:, :, :3, :3])
        f.write(f"旋转矩阵行列式统计:\n")
        f.write(f"  均值: {np.mean(det_values):.6f}\n")
        f.write(f"  标准差: {np.std(det_values):.6f}\n")
        f.write(f"  最小值: {np.min(det_values):.6f}\n")
        f.write(f"  最大值: {np.max(det_values):.6f}\n")
        f.write(f"  接近1的数量: {np.sum(np.abs(det_values - 1.0) < 0.01)}\n\n")
        
        # 计算平移向量统计
        trans_norms = np.linalg.norm(all_transforms[:, :, :3, 3], axis=-1)
        f.write(f"平移向量范数统计:\n")
        f.write(f"  均值: {np.mean(trans_norms):.6f}\n")
        f.write(f"  标准差: {np.std(trans_norms):.6f}\n")
        f.write(f"  最小值: {np.min(trans_norms):.6f}\n")
        f.write(f"  最大值: {np.max(trans_norms):.6f}\n")
    
    return all_transforms

def validate_transform_matrix(transform_matrix, tolerance=1e-4):
    """
    验证变换矩阵的有效性
    
    Args:
        transform_matrix: (4, 4) 变换矩阵
        tolerance: 数值容差
    
    Returns:
        validation_results: 验证结果字典
    """
    results = {}
    
    # 检查旋转部分 (3x3)
    rot_part = transform_matrix[:3, :3]
    
    # 检查是否正交
    should_be_identity = rot_part @ rot_part.T
    is_orthogonal = np.allclose(should_be_identity, np.eye(3), atol=tolerance)
    results['is_orthogonal'] = is_orthogonal
    
    # 检查行列式是否为1
    det = np.linalg.det(rot_part)
    is_proper_rotation = np.abs(det - 1.0) < tolerance
    results['is_proper_rotation'] = is_proper_rotation
    results['determinant'] = det
    
    # 检查底部行
    bottom_row = transform_matrix[3, :]
    expected_bottom = np.array([0, 0, 0, 1])
    bottom_correct = np.allclose(bottom_row, expected_bottom, atol=tolerance)
    results['bottom_row_correct'] = bottom_correct
    
    # 获取平移部分
    trans_part = transform_matrix[:3, 3]
    results['translation_norm'] = np.linalg.norm(trans_part)
    
    results['is_valid'] = is_orthogonal and is_proper_rotation and bottom_correct
    
    return results

def process_batch_results(results_dir, output_dir=None, method='scipy', max_timesteps=None):
    """
    处理整个批处理结果目录
    
    Args:
        results_dir: 批处理结果目录
        output_dir: 输出目录 (默认为 results_dir/transform_matrices)
        method: 旋转矩阵转换方法
        max_timesteps: 最大处理时间步数量 (用于测试)
    """
    results_path = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_path / "transform_matrices"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"处理批处理结果目录: {results_path}")
    print(f"输出目录: {output_dir}")
    print(f"旋转矩阵转换方法: {method}")
    print("="*60)
    
    # 找到所有时间步目录
    timestep_dirs = sorted([d for d in results_path.iterdir() 
                           if d.is_dir() and d.name.startswith('1AKE_A_time_step_')])
    
    if max_timesteps:
        timestep_dirs = timestep_dirs[:max_timesteps]
    
    print(f"找到 {len(timestep_dirs)} 个时间步目录")
    
    processed_count = 0
    all_transform_files = []
    
    for timestep_dir in timestep_dirs:
        timestep_name = timestep_dir.name
        
        # 查找旋转和平移scores文件
        rot_file = timestep_dir / "1AKE_A_rot_scores.npy"
        trans_file = timestep_dir / "1AKE_A_trans_scores.npy"
        
        if not (rot_file.exists() and trans_file.exists()):
            print(f"⚠ 跳过 {timestep_name}: 缺少score文件")
            continue
        
        try:
            # 加载数据
            rot_scores = np.load(rot_file)
            trans_scores = np.load(trans_file)
            
            # 检查数据形状
            if rot_scores.shape != trans_scores.shape:
                print(f"⚠ 跳过 {timestep_name}: 数据形状不匹配")
                continue
            
            if len(rot_scores.shape) != 3 or rot_scores.shape[-1] != 3:
                print(f"⚠ 跳过 {timestep_name}: 数据形状不正确 {rot_scores.shape}")
                continue
            
            # 处理当前时间步
            transform_matrices = process_single_timestep(
                rot_scores, trans_scores, output_dir, timestep_name, method
            )
            
            all_transform_files.append(output_dir / f"{timestep_name}_transform_matrices.npy")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理 {timestep_name} 时出错: {e}")
            continue
    
    print(f"\n✅ 处理完成!")
    print(f"成功处理 {processed_count} 个时间步")
    
    # 生成总体汇总报告
    summary_file = output_dir / "processing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("SE(3)变换矩阵构建处理汇总\n")
        f.write("="*50 + "\n")
        f.write(f"输入目录: {results_path}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"处理方法: {method}\n")
        f.write(f"总时间步数: {len(timestep_dirs)}\n")
        f.write(f"成功处理: {processed_count}\n")
        f.write(f"失败数量: {len(timestep_dirs) - processed_count}\n\n")
        
        f.write("生成的文件:\n")
        for i, file_path in enumerate(all_transform_files, 1):
            f.write(f"{i:2d}. {file_path.name}\n")
    
    print(f"汇总报告保存到: {summary_file}")
    
    # 验证几个样本矩阵
    if processed_count > 0:
        print(f"\n🔍 验证样本变换矩阵...")
        sample_file = all_transform_files[0]
        sample_data = np.load(sample_file)
        
        # 验证第一个样本的第一个残基
        sample_matrix = sample_data[0, 0]  # (4, 4)
        validation = validate_transform_matrix(sample_matrix)
        
        print(f"样本验证结果:")
        print(f"  • 旋转矩阵正交性: {'✓' if validation['is_orthogonal'] else '✗'}")
        print(f"  • 行列式为1: {'✓' if validation['is_proper_rotation'] else '✗'} (值: {validation['determinant']:.6f})")
        print(f"  • 底部行正确: {'✓' if validation['bottom_row_correct'] else '✗'}")
        print(f"  • 平移向量范数: {validation['translation_norm']:.6f}")
        print(f"  • 整体有效性: {'✓' if validation['is_valid'] else '✗'}")

def create_analysis_script(output_dir):
    """创建分析脚本用于进一步分析变换矩阵"""
    output_path = Path(output_dir)
    script_file = output_path / "analyze_transforms.py"
    
    script_content = '''#!/usr/bin/env python3
"""
SE(3)变换矩阵分析脚本
用于分析生成的变换矩阵的性质和统计信息
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_transform_matrices(transform_dir):
    """分析变换矩阵目录中的所有文件"""
    transform_path = Path(transform_dir)
    
    # 找到所有变换矩阵文件
    matrix_files = sorted(transform_path.glob("*_transform_matrices.npy"))
    
    print(f"找到 {len(matrix_files)} 个变换矩阵文件")
    
    all_determinants = []
    all_translation_norms = []
    
    for file_path in matrix_files:
        matrices = np.load(file_path)  # (num_samples, num_residues, 4, 4)
        
        # 提取旋转部分并计算行列式
        rot_parts = matrices[:, :, :3, :3]
        dets = np.linalg.det(rot_parts)
        all_determinants.extend(dets.flatten())
        
        # 提取平移部分并计算范数
        trans_parts = matrices[:, :, :3, 3]
        trans_norms = np.linalg.norm(trans_parts, axis=-1)
        all_translation_norms.extend(trans_norms.flatten())
    
    all_determinants = np.array(all_determinants)
    all_translation_norms = np.array(all_translation_norms)
    
    # 生成分析报告
    print(f"\\n旋转矩阵行列式分析:")
    print(f"  均值: {np.mean(all_determinants):.6f}")
    print(f"  标准差: {np.std(all_determinants):.6f}")
    print(f"  接近1的比例: {np.mean(np.abs(all_determinants - 1.0) < 0.01):.3f}")
    
    print(f"\\n平移向量范数分析:")
    print(f"  均值: {np.mean(all_translation_norms):.6f}")
    print(f"  标准差: {np.std(all_translation_norms):.6f}")
    print(f"  最大值: {np.max(all_translation_norms):.6f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python analyze_transforms.py <transform_matrices_directory>")
        sys.exit(1)
    
    analyze_transform_matrices(sys.argv[1])
'''
    
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"分析脚本已创建: {script_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SE(3)变换矩阵构建程序')
    parser.add_argument('results_dir', help='批处理结果目录路径')
    parser.add_argument('--output-dir', help='输出目录 (默认为 results_dir/transform_matrices)')
    parser.add_argument('--method', choices=['scipy', 'rodrigues'], default='scipy',
                       help='旋转矩阵转换方法 (默认: scipy)')
    parser.add_argument('--max-timesteps', type=int, help='最大处理时间步数量 (用于测试)')
    parser.add_argument('--create-analysis-script', action='store_true',
                       help='创建分析脚本')
    
    args = parser.parse_args()
    
    try:
        # 处理批处理结果
        process_batch_results(
            args.results_dir, 
            args.output_dir, 
            args.method, 
            args.max_timesteps
        )
        
        # 创建分析脚本
        if args.create_analysis_script:
            output_dir = args.output_dir or (Path(args.results_dir) / "transform_matrices")
            create_analysis_script(output_dir)
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()