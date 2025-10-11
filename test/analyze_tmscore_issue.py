#!/usr/bin/env python3
"""
分析TM-score低的原因
"""

import numpy as np
from Bio.PDB import PDBParser
import os

# 参考PDB文件
reference_pdb = 'pdb_dir/1AKE.pdb'

# 生成的结构目录
generated_dir = 'output_dir_batch/1AKE_structures'

def analyze_structure(pdb_path):
    """分析PDB结构的基本信息"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    
    coords = []
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
                    residues.append(residue.get_resname())
    
    coords = np.array(coords)
    
    # 计算统计信息
    center = np.mean(coords, axis=0)
    distances_from_center = np.linalg.norm(coords - center, axis=1)  # 修复: axis=1
    max_distance = np.max(distances_from_center)
    
    # 计算相邻CA原子距离
    ca_distances = []
    for i in range(len(coords) - 1):
        dist = np.linalg.norm(coords[i+1] - coords[i])
        ca_distances.append(dist)
    
    return {
        'num_residues': len(coords),
        'center': center,
        'max_distance_from_center': max_distance,
        'ca_distances_mean': np.mean(ca_distances) if ca_distances else 0,
        'ca_distances_std': np.std(ca_distances) if ca_distances else 0,
        'ca_distances_min': np.min(ca_distances) if ca_distances else 0,
        'ca_distances_max': np.max(ca_distances) if ca_distances else 0,
        'coords_range': np.ptp(coords, axis=0)  # range in each dimension
    }

print("=" * 80)
print("分析TM-score低的原因")
print("=" * 80)

# 分析参考结构
print("\n1. 参考结构分析 (1AKE.pdb):")
print("-" * 80)
ref_info = analyze_structure(reference_pdb)
for key, value in ref_info.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# 分析几个生成的结构
print("\n2. 生成结构分析 (前5个):")
print("-" * 80)

generated_files = sorted([f for f in os.listdir(generated_dir) if f.endswith('.pdb')])[:5]

for i, pdb_file in enumerate(generated_files):
    pdb_path = os.path.join(generated_dir, pdb_file)
    print(f"\n  文件 {i+1}: {pdb_file}")
    gen_info = analyze_structure(pdb_path)
    for key, value in gen_info.items():
        if isinstance(value, np.ndarray):
            print(f"    {key}: {value}")
        else:
            print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")

# 关键发现
print("\n" + "=" * 80)
print("3. 关键发现:")
print("=" * 80)

# CA-CA 距离理论值应该是约3.8 Å
ideal_ca_distance = 3.8
print(f"\n  理想CA-CA距离: {ideal_ca_distance:.2f} Å")
print(f"  参考结构CA-CA距离: {ref_info['ca_distances_mean']:.2f} ± {ref_info['ca_distances_std']:.2f} Å")

if generated_files:
    gen_info = analyze_structure(os.path.join(generated_dir, generated_files[0]))
    print(f"  生成结构CA-CA距离: {gen_info['ca_distances_mean']:.2f} ± {gen_info['ca_distances_std']:.2f} Å")
    
    # 检查是否结构合理
    if gen_info['ca_distances_mean'] < 2.0 or gen_info['ca_distances_mean'] > 5.0:
        print("\n  ⚠️  警告: 生成结构的CA-CA距离异常!")
        print("     这可能导致TM-score计算错误。")
    
    if gen_info['max_distance_from_center'] > 1000:
        print("\n  ⚠️  警告: 生成结构展开过大!")
        print("     结构可能未正确折叠。")

print("\n" + "=" * 80)
print("4. 代码问题诊断:")
print("=" * 80)

print("""
可能的问题:
1. **使用了噪声结构而非去噪结构**
   - 当前代码保存的是 rigids_t (噪声状态的结构)
   - 应该保存去噪后的结构 rigids_0
   
2. **Score预测器不是去噪器**
   - score_network 输出的是 score (梯度)，不是去噪后的结构
   - 需要使用逆向扩散过程从 t=1 逐步去噪到 t=0
   
3. **rigids_t来自sample_ref()** 
   - sample_ref() 采样的是随机参考分布
   - 这些是噪声结构，与原始结构无关
   
解决方案:
- 应该实现完整的逆向扩散采样过程
- 或者使用已有的inference脚本来生成结构
""")

print("\n" + "=" * 80)
