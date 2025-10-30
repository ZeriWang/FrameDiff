#!/usr/bin/env python3
"""
前向加噪预测器

功能:
1. 读取原始PDB文件
2. 应用前向扩散加噪
3. 保存加噪后的PDB结构
4. 计算原结构与加噪结构的TM-score
5. 保存ground truth旋转Score和平移Score为.npy格式

基于: score_predictor_TMscore_optimized.py
参考: train_se3_diffusion.py中的loss_fn函数
"""

import os
import sys
import copy
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from data import utils as du
from data import se3_diffuser
from data import all_atom
from openfold.data import data_transforms
from openfold.utils import rigid_utils as ru
from openfold.np import protein
from openfold.np import residue_constants

try:
    import tmtools
    from Bio.PDB import PDBParser
    TMTOOLS_AVAILABLE = True
except ImportError:
    TMTOOLS_AVAILABLE = False
    print("Warning: tmtools not available, TM-score calculation will be skipped")

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 输入参数
PDB_PATH = str(PROJECT_ROOT / 'test' / 'pdb_dir' / '1AKE.pdb')
CHAIN_ID = 'B'
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'test_forward_noise')
CONF_PATH = str(PROJECT_ROOT / 'config' / 'base.yaml')

# 前向加噪参数
NOISE_TIME_T = 0.00001       # 加噪的时间步 t ∈ [0, 1]，越大噪声越强
NUM_NOISE_SAMPLES = 3     # 生成的加噪样本数量（可以生成多个）
SAVE_SCORES = True        # 保存ground truth scores为npy格式


def process_chain_feats(pdb_feats):
    """处理PDB特征"""
    chain_feats = {
        'aatype': torch.tensor(pdb_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(pdb_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(pdb_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
    
    seq_idx = pdb_feats['residue_index'] - np.min(pdb_feats['residue_index']) + 1
    chain_feats['seq_idx'] = seq_idx
    chain_feats['res_mask'] = pdb_feats['bb_mask']
    chain_feats['residue_index'] = pdb_feats['residue_index']
    return chain_feats


def rigids_to_protein(rigids_t, aatype, residue_index):
    """将SE(3)样本转换为Protein对象，复用官方骨架重建逻辑。"""

    if isinstance(rigids_t, torch.Tensor):
        rigid_tensor = rigids_t
    elif isinstance(rigids_t, ru.Rigid):
        rigid_tensor = rigids_t.to_tensor_7()
    else:
        raise ValueError(f"Unsupported rigids_t type: {type(rigids_t)}")

    if rigid_tensor.ndim == 2:
        rigid_tensor = rigid_tensor.unsqueeze(0)
    elif rigid_tensor.ndim != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {rigid_tensor.ndim}D")

    rigids_batch = ru.Rigid.from_tensor_7(rigid_tensor)
    batch_size, num_res = rigids_batch.get_rots().get_rot_mats().shape[:2]

    psi_torsions = rigid_tensor.new_zeros((batch_size, num_res, 2))
    psi_torsions[..., 0] = 1.0

    atom37_pos, atom37_mask, _, _ = all_atom.compute_backbone(
        rigids_batch, psi_torsions
    )

    atom37_pos = atom37_pos[0]
    atom37_mask = atom37_mask[0]

    if isinstance(atom37_pos, torch.Tensor):
        atom37_pos = atom37_pos.cpu().numpy()
    if isinstance(atom37_mask, torch.Tensor):
        atom37_mask = atom37_mask.cpu().numpy()

    if isinstance(aatype, torch.Tensor):
        aatype = aatype.cpu().numpy()
    if isinstance(residue_index, torch.Tensor):
        residue_index = residue_index.cpu().numpy()

    if atom37_pos.shape[0] != len(aatype):
        raise ValueError(f"Shape mismatch: {atom37_pos.shape[0]} != {len(aatype)}")

    b_factors = np.zeros_like(atom37_mask, dtype=np.float32)

    return protein.Protein(
        atom_positions=atom37_pos,
        aatype=aatype,
        atom_mask=atom37_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=np.zeros(len(aatype), dtype=np.int32),
    )


def save_protein_to_pdb(prot, output_path):
    """保存PDB文件"""
    pdb_string = protein.to_pdb(prot)
    with open(output_path, 'w') as f:
        f.write(pdb_string)
    print(f"已保存PDB文件: {output_path}")


def calculate_tm_score(pdb_path1, pdb_path2):
    """计算TM-score，参考score_predictor_TMscore_optimized.py的实现"""
    if not TMTOOLS_AVAILABLE:
        return None
    
    try:
        # 使用Bio.PDB解析PDB文件并提取CA坐标和序列
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure('target', pdb_path1)
        structure2 = parser.get_structure('pred', pdb_path2)
        
        # 提取第一个结构的CA坐标和序列
        coords1 = []
        seq1 = []
        for model in structure1:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords1.append(residue['CA'].get_coord())
                        seq1.append(residue.get_resname())
        
        # 提取第二个结构的CA坐标和序列
        coords2 = []
        seq2 = []
        for model in structure2:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords2.append(residue['CA'].get_coord())
                        seq2.append(residue.get_resname())
        
        # 转换为numpy数组
        coords1 = np.array(coords1, dtype=np.float64)
        coords2 = np.array(coords2, dtype=np.float64)
        
        # 将三字母氨基酸代码转换为单字母代码
        from openfold.np.residue_constants import restype_3to1
        seq1_str = ''.join([restype_3to1.get(res, 'X') for res in seq1])
        seq2_str = ''.join([restype_3to1.get(res, 'X') for res in seq2])
        
        # 使用tmtools计算TM-score，参考score_predictor_TMscore_optimized.py
        result = tmtools.tm_align(coords1, coords2, seq1_str, seq2_str)
        tm_score = result.tm_norm_chain1
        
        print(f"\nTM-score计算结果: {tm_score:.4f}")
        
        # 返回TM-score值，与score_predictor_TMscore_optimized.py保持一致
        return tm_score
        
    except Exception as e:
        print(f"TM-score计算失败: {e}")
        return None


def forward_noise_sampling(
        diffuser,
        original_feats,
        t,
        num_samples=1,
        device='cpu'
    ):
    """
    前向加噪采样过程
    
    参数:
        diffuser: SE3扩散器
        original_feats: 原始特征字典
        t: 时间步 [0, 1]
        num_samples: 生成样本数量
        device: 计算设备
    
    返回:
        list of dict: 每个样本包含加噪后的结构和ground truth scores
    """
    results = []
    
    # 提取原始rigids
    rigids_0 = ru.Rigid.from_tensor_7(original_feats['rigids_0'])
    res_mask = original_feats['res_mask']
    
    # 创建diffuse_mask（这里假设所有残基都参与扩散）
    diffuse_mask = np.ones_like(res_mask)
    
    print(f"\n执行前向加噪:")
    print(f"  时间步 t = {t}")
    print(f"  残基数量 = {rigids_0.shape[0]}")
    print(f"  生成样本数 = {num_samples}")
    
    for sample_idx in range(num_samples):
        print(f"\n生成样本 {sample_idx + 1}/{num_samples}...")
        
        # 应用前向扩散
        # forward_marginal返回: rigids_t, trans_score, rot_score, trans_score_scaling, rot_score_scaling
        diff_feats = diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=t,
            diffuse_mask=diffuse_mask,
            as_tensor_7=True
        )
        
        # 提取结果
        rigids_t = diff_feats['rigids_t']  # 加噪后的结构
        rot_score = diff_feats['rot_score']  # ground truth旋转score
        trans_score = diff_feats['trans_score']  # ground truth平移score
        rot_score_scaling = diff_feats['rot_score_scaling']
        trans_score_scaling = diff_feats['trans_score_scaling']
        
        print(f"  rot_score shape: {rot_score.shape}")
        print(f"  trans_score shape: {trans_score.shape}")
        print(f"  rot_score_scaling: {rot_score_scaling}")
        print(f"  trans_score_scaling: {trans_score_scaling}")
        
        results.append({
            'rigids_t': rigids_t,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'rot_score_scaling': rot_score_scaling,
            'trans_score_scaling': trans_score_scaling,
            't': t,
            'sample_idx': sample_idx
        })
    
    return results


def save_scores_to_npy(rot_score, trans_score, output_dir, pdb_name, sample_idx, t):
    """保存ground truth scores为.npy格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    rot_score_path = os.path.join(
        output_dir, 
        f'{pdb_name}_sample_{sample_idx:03d}_t_{t:.3f}_rot_score.npy'
    )
    trans_score_path = os.path.join(
        output_dir, 
        f'{pdb_name}_sample_{sample_idx:03d}_t_{t:.3f}_trans_score.npy'
    )
    
    np.save(rot_score_path, rot_score)
    np.save(trans_score_path, trans_score)
    
    print(f"\n已保存ground truth scores:")
    print(f"  旋转score: {rot_score_path}")
    print(f"  平移score: {trans_score_path}")
    
    return rot_score_path, trans_score_path


def save_summary(results, output_dir, pdb_name):
    """保存汇总结果"""
    summary_path = os.path.join(output_dir, f'{pdb_name}_forward_noise_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("前向加噪结果汇总\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"PDB文件: {pdb_name}\n")
        f.write(f"加噪时间步 t: {results[0]['t']}\n")
        f.write(f"生成样本数: {len(results)}\n\n")
        
        for i, result in enumerate(results):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"样本 {i + 1}\n")
            f.write(f"{'=' * 80}\n")
            
            # tm_score现在是float而不是dict
            if 'tm_score' in result and result['tm_score'] is not None:
                f.write(f"  TM-score: {result['tm_score']:.4f}\n")
            
            f.write(f"  加噪后PDB: {result['noised_pdb']}\n")
            
            if 'rot_score_path' in result:
                f.write(f"  旋转score文件: {result['rot_score_path']}\n")
                f.write(f"  平移score文件: {result['trans_score_path']}\n")
            
            rot_score = result['rot_score']
            trans_score = result['trans_score']
            f.write(f"\n  Ground Truth Scores统计:\n")
            f.write(f"    旋转score范围: [{np.min(rot_score):.4f}, {np.max(rot_score):.4f}]\n")
            f.write(f"    旋转score均值: {np.mean(rot_score):.4f}\n")
            f.write(f"    旋转score标准差: {np.std(rot_score):.4f}\n")
            f.write(f"    平移score范围: [{np.min(trans_score):.4f}, {np.max(trans_score):.4f}]\n")
            f.write(f"    平移score均值: {np.mean(trans_score):.4f}\n")
            f.write(f"    平移score标准差: {np.std(trans_score):.4f}\n")
    
    print(f"\n已保存汇总文件: {summary_path}")


def main():
    print("=" * 80)
    print("前向加噪预测器")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载配置
    print(f"\n加载配置文件: {CONF_PATH}")
    conf = OmegaConf.load(CONF_PATH)
    
    # 初始化diffuser
    print("初始化SE3 Diffuser...")
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    
    # 读取PDB文件
    print(f"\n读取PDB文件: {PDB_PATH}")
    pdb_name = os.path.splitext(os.path.basename(PDB_PATH))[0]
    
    # 使用正确的参数顺序和名称，参考score_predictor_TMscore_optimized.py
    pdb_feats = du.parse_pdb_feats('query', PDB_PATH, chain_id=CHAIN_ID)
    
    chain_feats = process_chain_feats(pdb_feats)
    
    # 使用bb_mask来获取正确的残基数量，参考score_predictor_TMscore_optimized.py
    bb_mask = np.array(pdb_feats['bb_mask']).astype(bool)
    num_res = int(np.sum(bb_mask))
    print(f"成功读取PDB，有效残基数量: {num_res}")
    print(f"总原子数量: {len(chain_feats['aatype'])}")
    
    # 准备原始特征
    # 从rigidgroups_gt_frames提取backbone frames (索引0表示backbone)
    # 使用bb_mask而不是res_mask，并正确转换为torch tensor
    mask_tensor = torch.from_numpy(bb_mask).to(torch.bool)
    rigid_frames = chain_feats['rigidgroups_gt_frames'][mask_tensor, 0].detach().cpu().float()
    rigids_0 = ru.Rigid.from_tensor_4x4(rigid_frames)
    
    original_feats = {
        'rigids_0': rigids_0.to_tensor_7(),
        'res_mask': np.ones(num_res, dtype=np.float32),  # 只保留有效残基的mask，shape: (214,)
        'aatype': chain_feats['aatype'][mask_tensor],  # 只取有效残基的aatype
        'residue_index': chain_feats['residue_index'][mask_tensor]  # 只取有效残基的residue_index
    }
    
    # 执行前向加噪
    noised_samples = forward_noise_sampling(
        diffuser=diffuser,
        original_feats=original_feats,
        t=NOISE_TIME_T,
        num_samples=NUM_NOISE_SAMPLES,
        device='cpu'
    )
    
    # 处理每个样本
    all_results = []
    
    for sample_result in noised_samples:
        sample_idx = sample_result['sample_idx']
        rigids_t = sample_result['rigids_t']
        rot_score = sample_result['rot_score']
        trans_score = sample_result['trans_score']
        
        print(f"\n处理样本 {sample_idx + 1}/{NUM_NOISE_SAMPLES}...")
        
        # 转换为Protein对象并保存
        noised_protein = rigids_to_protein(
            rigids_t,
            original_feats['aatype'],
            original_feats['residue_index']
        )
        
        noised_pdb_path = os.path.join(
            OUTPUT_DIR,
            f'{pdb_name}_sample_{sample_idx:03d}_t_{NOISE_TIME_T:.3f}_noised.pdb'
        )
        save_protein_to_pdb(noised_protein, noised_pdb_path)
        
        # 计算TM-score
        tm_result = None
        if TMTOOLS_AVAILABLE:
            tm_result = calculate_tm_score(PDB_PATH, noised_pdb_path)
        
        # 保存ground truth scores
        rot_score_path = None
        trans_score_path = None
        if SAVE_SCORES:
            rot_score_path, trans_score_path = save_scores_to_npy(
                rot_score,
                trans_score,
                OUTPUT_DIR,
                pdb_name,
                sample_idx,
                NOISE_TIME_T
            )
        
        all_results.append({
            'sample_idx': sample_idx,
            't': NOISE_TIME_T,
            'noised_pdb': noised_pdb_path,
            'rot_score': rot_score,
            'trans_score': trans_score,
            'rot_score_path': rot_score_path,
            'trans_score_path': trans_score_path,
            'tm_score': tm_result
        })
    
    # 保存汇总结果
    save_summary(all_results, OUTPUT_DIR, pdb_name)
    
    print("\n" + "=" * 80)
    print("前向加噪完成！")
    print("=" * 80)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"生成样本数: {NUM_NOISE_SAMPLES}")
    if all_results and all_results[0]['tm_score'] is not None:
        # tm_score现在是float而不是dict
        avg_tm = np.mean([r['tm_score'] for r in all_results if r['tm_score'] is not None])
        print(f"平均TM-score: {avg_tm:.4f}")


if __name__ == '__main__':
    main()
