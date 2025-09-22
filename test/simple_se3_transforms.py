#!/usr/bin/env python3
"""
简化的SE(3)变换矩阵构建程序
"""

import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

def create_transform_matrices(results_dir):
    """处理批处理结果，生成4x4变换矩阵"""
    results_path = Path(results_dir)
    output_dir = results_path / "transform_matrices"
    output_dir.mkdir(exist_ok=True)
    
    print(f"处理目录: {results_path}")
    print(f"输出目录: {output_dir}")
    print("="*50)
    
    # 找到所有时间步目录
    timestep_dirs = sorted([d for d in results_path.iterdir() 
                           if d.is_dir() and d.name.startswith('1AKE_A_time_step_')])
    
    print(f"找到 {len(timestep_dirs)} 个时间步目录")
    
    processed_count = 0
    
    for timestep_dir in timestep_dirs:
        timestep_name = timestep_dir.name
        
        # 检查文件
        rot_file = timestep_dir / "1AKE_A_rot_scores.npy"
        trans_file = timestep_dir / "1AKE_A_trans_scores.npy"
        
        if not (rot_file.exists() and trans_file.exists()):
            print(f"⚠ 跳过 {timestep_name}: 缺少文件")
            continue
        
        try:
            # 加载数据
            rot_scores = np.load(rot_file)  # (num_samples, num_residues, 3)
            trans_scores = np.load(trans_file)  # (num_samples, num_residues, 3)
            
            num_samples, num_residues, _ = rot_scores.shape
            print(f"处理 {timestep_name}: {num_samples} 样本, {num_residues} 残基")
            
            # 为每个样本处理
            all_transforms = []
            
            for sample_idx in range(num_samples):
                sample_rot_scores = rot_scores[sample_idx]  # (num_residues, 3)
                sample_trans_scores = trans_scores[sample_idx]  # (num_residues, 3)
                
                # 转换每个残基的旋转向量为旋转矩阵
                rotation_matrices = []
                for residue_idx in range(num_residues):
                    rot_vec = sample_rot_scores[residue_idx]
                    
                    # 检查是否是零向量
                    if np.linalg.norm(rot_vec) < 1e-8:
                        rot_matrix = np.eye(3)
                    else:
                        rot_matrix = Rotation.from_rotvec(rot_vec).as_matrix()
                    
                    rotation_matrices.append(rot_matrix)
                
                rotation_matrices = np.array(rotation_matrices)  # (num_residues, 3, 3)
                
                # 创建4x4变换矩阵
                transform_matrices = np.zeros((num_residues, 4, 4))
                transform_matrices[:, :3, :3] = rotation_matrices
                transform_matrices[:, :3, 3] = sample_trans_scores
                transform_matrices[:, 3, 3] = 1.0
                
                all_transforms.append(transform_matrices)
            
            # 堆叠所有样本 (num_samples, num_residues, 4, 4)
            all_transforms = np.array(all_transforms)
            
            # 保存结果
            output_file = output_dir / f"{timestep_name}_transforms.npy"
            np.save(output_file, all_transforms)
            
            print(f"  ✓ 保存到: {output_file.name} | 形状: {all_transforms.shape}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理 {timestep_name} 出错: {e}")
            continue
    
    print(f"\\n✅ 完成! 成功处理 {processed_count}/{len(timestep_dirs)} 个时间步")
    
    # 验证第一个文件
    if processed_count > 0:
        first_file = output_dir / f"{timestep_dirs[0].name}_transforms.npy"
        if first_file.exists():
            test_data = np.load(first_file)
            sample_matrix = test_data[0, 0]  # 第一个样本第一个残基
            
            rot_part = sample_matrix[:3, :3]
            det = np.linalg.det(rot_part)
            is_orthogonal = np.allclose(rot_part @ rot_part.T, np.eye(3), atol=1e-4)
            
            print(f"\\n🔍 验证样本变换矩阵:")
            print(f"  • 旋转部分行列式: {det:.6f}")
            print(f"  • 正交性检查: {'✓' if is_orthogonal else '✗'}")
            print(f"  • 底部行: {sample_matrix[3, :]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python simple_se3_transforms.py <results_directory>")
        sys.exit(1)
    
    create_transform_matrices(sys.argv[1])