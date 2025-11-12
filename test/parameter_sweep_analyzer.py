#!/usr/bin/env python3
"""
参数扫描分析器

功能:
1. 系统测试不同NUM_DENOISING_STEPS和MAX_T参数组合
2. 保存每种组合的PDB文件、旋转Score和平移Score
3. 对结果进行统计分析和可视化
4. 生成详细的分析报告

参数组合:
- NUM_DENOISING_STEPS: [10, 100, 500, 1000]
- MAX_T: [0.05, 0.1, 0.3, 0.5, 0.9]
- 总共 4 × 5 = 20 种组合

基于: direct_denoising_predictor.py
"""
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
import json

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from data import utils as du
from data import se3_diffuser
from data import all_atom
from model import score_network
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

# ==================== 配置参数 ====================
# 输入参数
PDB_PATH = str(PROJECT_ROOT / 'test' / 'pdb_dir' / '4AKE.pdb')
CHAIN_ID = 'B'
WEIGHTS_PATH = str(PROJECT_ROOT / 'weights' / 'best_weights.pth')
CONF_PATH = str(PROJECT_ROOT / 'config' / 'base.yaml')

# 输出目录
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'parameter_sweep_results')
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'run_{TIMESTAMP}')

# 参数扫描范围
NUM_STEPS_LIST = [10, 100, 500, 1000]
MAX_T_LIST = [0.05, 0.1, 0.3, 0.5, 0.9]

# 固定参数
MIN_T = 0.01
NOISE_SCALE = 0.1
ENABLE_SELF_CONDITIONING = True

# ==================== 辅助函数 ====================

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
    """将SE(3)样本转换为Protein对象"""
    if isinstance(rigids_t, torch.Tensor):
        rigid_tensor = rigids_t
    elif isinstance(rigids_t, ru.Rigid):
        rigid_tensor = rigids_t.to_tensor_7()
    else:
        rigid_tensor = torch.tensor(rigids_t)

    if rigid_tensor.ndim == 2:
        rigid_tensor = rigid_tensor.unsqueeze(0)
    elif rigid_tensor.ndim != 3:
        raise ValueError(f"Expected tensor of shape [B, N, 7] or [N, 7], got {rigid_tensor.shape}")

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
        raise ValueError(f"Mismatch: atom37_pos has {atom37_pos.shape[0]} residues but aatype has {len(aatype)}")

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


def calculate_tm_score(pdb_path1, pdb_path2, chain_id1=None, chain_id2=None):
    """计算TM-score"""
    if not TMTOOLS_AVAILABLE:
        return None
    
    try:
        parser = PDBParser(QUIET=True)
        
        struct1 = parser.get_structure('ref', pdb_path1)
        if chain_id1:
            chain1 = struct1[0][chain_id1]
        else:
            chain1 = list(struct1.get_chains())[0]
        
        coords1 = []
        seq1 = []
        for res in chain1.get_residues():
            if 'CA' in res:
                coords1.append(res['CA'].coord)
                seq1.append(res.resname)
        
        struct2 = parser.get_structure('query', pdb_path2)
        if chain_id2:
            chain2 = struct2[0][chain_id2]
        else:
            chain2 = list(struct2.get_chains())[0]
        
        coords2 = []
        seq2 = []
        for res in chain2.get_residues():
            if 'CA' in res:
                coords2.append(res['CA'].coord)
                seq2.append(res.resname)
        
        coords1 = np.array(coords1)
        coords2 = np.array(coords2)
        
        if coords1.shape[0] != coords2.shape[0]:
            min_len = min(coords1.shape[0], coords2.shape[0])
            coords1 = coords1[:min_len]
            coords2 = coords2[:min_len]
            seq1 = seq1[:min_len]
            seq2 = seq2[:min_len]
        
        tm_result = tmtools.tm_score(coords1, coords2, seq1, seq2)
        return tm_result.tm_norm_chain1
        
    except Exception as e:
        print(f"计算TM-score时出错: {e}")
        return None


def calculate_rmsd(coords1, coords2):
    """计算RMSD"""
    if coords1.shape != coords2.shape:
        min_len = min(coords1.shape[0], coords2.shape[0])
        coords1 = coords1[:min_len]
        coords2 = coords2[:min_len]
    
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def direct_denoising(
        model,
        diffuser,
        original_rigids,
        res_mask,
        seq_idx,
        fixed_mask,
        torsion_angles,
        sc_ca,
        num_steps=100,
        min_t=0.001,
        max_t=0.05,
        device='cuda',
        noise_scale=0.1,
        enable_self_conditioning=True,
    ):
    """
    直接去噪过程：从原始结构开始，进行轻微的去噪优化
    
    返回:
        dict: 包含最终结构、score历史、以及最后一步的score
    """
    print(f"开始直接去噪过程...")
    print(f"  去噪步数: {num_steps}")
    print(f"  时间范围: {min_t} -> {max_t}")
    print(f"  噪声缩放: {noise_scale}")
    
    # 准备输入特征
    sample_feats = {
        'rigids_t': original_rigids.to_tensor_7().unsqueeze(0).to(device),
        'res_mask': res_mask.unsqueeze(0).to(device),
        'seq_idx': seq_idx.unsqueeze(0).to(device),
        'fixed_mask': fixed_mask.unsqueeze(0).to(device),
        'torsion_angles_sin_cos': torsion_angles.unsqueeze(0).to(device),
        'sc_ca_t': sc_ca.unsqueeze(0).to(device),
    }
    
    batch_size = sample_feats['rigids_t'].shape[0]
    
    # 创建去噪时间步序列（从max_t到min_t）
    denoising_steps = np.linspace(max_t, min_t, num_steps)
    dt = (max_t - min_t) / max(num_steps - 1, 1)
    
    all_rot_scores = []
    all_trans_scores = []
    
    diffuse_mask = ((1 - sample_feats['fixed_mask']) * sample_feats['res_mask']).detach().cpu().numpy()
    fixed_mask_np = (sample_feats['fixed_mask'] * sample_feats['res_mask']).detach().cpu().numpy()
    t_placeholder = torch.ones(batch_size, device=device)

    embed_self_conditioning = (
        enable_self_conditioning and
        getattr(model.embedding_layer._embed_conf, 'embed_self_conditioning', False)
    )

    def set_t_feats(feats, t_value):
        feats['t'] = t_placeholder * float(t_value)
        rot_scale, trans_scale = diffuser.score_scaling(float(t_value))
        feats['rot_score_scaling'] = torch.full((batch_size,), float(rot_scale), device=device)
        feats['trans_score_scaling'] = torch.full((batch_size,), float(trans_scale), device=device)
        return feats

    # 保存最后一步的score
    final_rot_score = None
    final_trans_score = None

    with torch.no_grad():
        # 自条件初始化
        if embed_self_conditioning and len(denoising_steps) > 0:
            set_t_feats(sample_feats, denoising_steps[0])
            model_sc = model(sample_feats)
            sample_feats['sc_ca_t'] = model_sc['rigids'][..., 4:]

        # 逆向去噪循环
        for step_idx, t in enumerate(tqdm(denoising_steps, desc="去噪进行中")):
            set_t_feats(sample_feats, t)
            model_out = model(sample_feats)
            rot_score = model_out['rot_score']
            trans_score = model_out['trans_score']

            # 保存所有时间步的score
            all_rot_scores.append({'t': float(t), 'score': du.move_to_np(rot_score)})
            all_trans_scores.append({'t': float(t), 'score': du.move_to_np(trans_score)})

            # 如果是最后一步，保存score
            if step_idx == len(denoising_steps) - 1:
                final_rot_score = du.move_to_np(rot_score).copy()
                final_trans_score = du.move_to_np(trans_score).copy()

            # 执行去噪步骤
            if t > min_t:
                current_rigid = ru.Rigid.from_tensor_7(sample_feats['rigids_t'])
                
                rigids_t = diffuser.reverse(
                    rigid_t=current_rigid,
                    rot_score=du.move_to_np(rot_score),
                    trans_score=du.move_to_np(trans_score),
                    diffuse_mask=diffuse_mask,
                    t=float(t),
                    dt=dt,
                    center=True,
                    noise_scale=noise_scale,
                )
                
                # 确保结果移回GPU
                rigids_t_tensor = rigids_t.to_tensor_7().to(device)
                sample_feats['rigids_t'] = rigids_t_tensor
                
                if embed_self_conditioning:
                    sample_feats['sc_ca_t'] = model_out['rigids'][..., 4:]
            else:
                rigids_t = ru.Rigid.from_tensor_7(sample_feats['rigids_t'])

    print(f"去噪完成！")
    
    return {
        'final_rigids': rigids_t,
        'all_rot_scores': all_rot_scores,
        'all_trans_scores': all_trans_scores,
        'final_rot_score': final_rot_score,
        'final_trans_score': final_trans_score,
        'fixed_mask': fixed_mask_np,
        'diffuse_mask': diffuse_mask,
    }




def run_single_experiment(model, diffuser, chain_feats, pdb_feats, bb_mask, 
                         num_steps, max_t, device, output_subdir):
    """运行单个参数组合的实验"""
    
    num_res = int(np.sum(bb_mask))
    
    # 准备特征
    mask_tensor = torch.from_numpy(bb_mask).to(torch.bool)
    torsion_angles = chain_feats['torsion_angles_sin_cos'].detach().cpu().numpy()[bb_mask]
    
    if torsion_angles.dtype == np.object_:
        torsion_angles_list = []
        for angle in torsion_angles:
            if isinstance(angle, np.ndarray):
                torsion_angles_list.append(angle)
            else:
                torsion_angles_list.append(np.zeros((7, 2), dtype=np.float32))
        torsion_angles = np.stack(torsion_angles_list, axis=0).astype(np.float32)
    else:
        torsion_angles = torsion_angles.astype(np.float32)

    # 获取原始刚体变换
    rigid_frames = chain_feats['rigidgroups_gt_frames'][mask_tensor, 0].detach().cpu().float()
    rigids_0 = ru.Rigid.from_tensor_4x4(rigid_frames)
    sc_ca_init = rigids_0.get_trans().detach().cpu().numpy().astype(np.float32)

    # 准备张量
    res_mask_tensor = torch.ones(num_res, dtype=torch.float32)
    seq_idx_tensor = torch.arange(1, num_res + 1, dtype=torch.float32)
    fixed_mask_tensor = torch.zeros(num_res, dtype=torch.float32)
    torsion_tensor = torch.tensor(torsion_angles, dtype=torch.float32)
    sc_ca_tensor = torch.tensor(sc_ca_init, dtype=torch.float32)
    
    # 提取aatype和residue_index
    aatype = pdb_feats['aatype'][bb_mask]
    residue_index = pdb_feats['residue_index'][bb_mask]
    
    # 执行去噪
    denoising_result = direct_denoising(
        model=model,
        diffuser=diffuser,
        original_rigids=rigids_0,
        res_mask=res_mask_tensor,
        seq_idx=seq_idx_tensor,
        fixed_mask=fixed_mask_tensor,
        torsion_angles=torsion_tensor,
        sc_ca=sc_ca_tensor,
        num_steps=num_steps,
        min_t=MIN_T,
        max_t=max_t,
        device=device,
        noise_scale=NOISE_SCALE,
        enable_self_conditioning=ENABLE_SELF_CONDITIONING,
    )
    
    # 保存结果
    os.makedirs(output_subdir, exist_ok=True)
    
    # 保存PDB
    final_rigids = denoising_result['final_rigids']
    final_prot = rigids_to_protein(final_rigids, aatype, residue_index)
    pdb_filename = f'denoised_steps{num_steps}_maxT{max_t:.2f}.pdb'
    pdb_path = os.path.join(output_subdir, pdb_filename)
    save_protein_to_pdb(final_prot, pdb_path)
    
    # 保存Score
    rot_score_filename = f'rot_score_steps{num_steps}_maxT{max_t:.2f}.npy'
    trans_score_filename = f'trans_score_steps{num_steps}_maxT{max_t:.2f}.npy'
    np.save(os.path.join(output_subdir, rot_score_filename), 
            denoising_result['final_rot_score'])
    np.save(os.path.join(output_subdir, trans_score_filename), 
            denoising_result['final_trans_score'])
    
    # 计算TM-score
    tm_score = calculate_tm_score(PDB_PATH, pdb_path, CHAIN_ID, None)
    
    # 计算RMSD
    original_ca = sc_ca_init
    final_ca = final_rigids.get_trans().detach().cpu().numpy()
    if final_ca.ndim == 3:
        final_ca = final_ca[0]
    rmsd = calculate_rmsd(original_ca, final_ca)
    
    # 计算Score统计
    rot_score_norm = np.linalg.norm(denoising_result['final_rot_score'])
    trans_score_norm = np.linalg.norm(denoising_result['final_trans_score'])
    
    result = {
        'num_steps': num_steps,
        'max_t': max_t,
        'tm_score': tm_score,
        'rmsd': rmsd,
        'rot_score_norm': rot_score_norm,
        'trans_score_norm': trans_score_norm,
        'pdb_path': pdb_path,
        'rot_score_path': os.path.join(output_subdir, rot_score_filename),
        'trans_score_path': os.path.join(output_subdir, trans_score_filename),
    }
    
    return result


def create_visualizations(df, output_dir):
    """创建可视化图表"""
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. TM-score vs 参数热图
    if 'tm_score' in df.columns and df['tm_score'].notna().any():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # TM-score热图
        pivot_tm = df.pivot(index='num_steps', columns='max_t', values='tm_score')
        sns.heatmap(pivot_tm, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('TM-Score Heatmap')
        axes[0, 0].set_xlabel('MAX_T')
        axes[0, 0].set_ylabel('NUM_STEPS')
        
        # RMSD热图
        pivot_rmsd = df.pivot(index='num_steps', columns='max_t', values='rmsd')
        sns.heatmap(pivot_rmsd, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[0, 1])
        axes[0, 1].set_title('RMSD Heatmap')
        axes[0, 1].set_xlabel('MAX_T')
        axes[0, 1].set_ylabel('NUM_STEPS')
        
        # Rotation Score Norm热图
        pivot_rot = df.pivot(index='num_steps', columns='max_t', values='rot_score_norm')
        sns.heatmap(pivot_rot, annot=True, fmt='.2f', cmap='RdPu', ax=axes[1, 0])
        axes[1, 0].set_title('Rotation Score Norm Heatmap')
        axes[1, 0].set_xlabel('MAX_T')
        axes[1, 0].set_ylabel('NUM_STEPS')
        
        # Translation Score Norm热图
        pivot_trans = df.pivot(index='num_steps', columns='max_t', values='trans_score_norm')
        sns.heatmap(pivot_trans, annot=True, fmt='.2f', cmap='Greens', ax=axes[1, 1])
        axes[1, 1].set_title('Translation Score Norm Heatmap')
        axes[1, 1].set_xlabel('MAX_T')
        axes[1, 1].set_ylabel('NUM_STEPS')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 参数影响趋势图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # NUM_STEPS的影响
    for max_t in MAX_T_LIST:
        subset = df[df['max_t'] == max_t]
        if 'tm_score' in subset.columns and subset['tm_score'].notna().any():
            axes[0, 0].plot(subset['num_steps'], subset['tm_score'], 
                          marker='o', label=f'MAX_T={max_t}')
    axes[0, 0].set_xlabel('NUM_STEPS')
    axes[0, 0].set_ylabel('TM-Score')
    axes[0, 0].set_title('TM-Score vs NUM_STEPS')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True)
    
    # MAX_T的影响
    for num_steps in NUM_STEPS_LIST:
        subset = df[df['num_steps'] == num_steps]
        if 'tm_score' in subset.columns and subset['tm_score'].notna().any():
            axes[0, 1].plot(subset['max_t'], subset['tm_score'], 
                          marker='s', label=f'STEPS={num_steps}')
    axes[0, 1].set_xlabel('MAX_T')
    axes[0, 1].set_ylabel('TM-Score')
    axes[0, 1].set_title('TM-Score vs MAX_T')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # RMSD vs NUM_STEPS
    for max_t in MAX_T_LIST:
        subset = df[df['max_t'] == max_t]
        axes[1, 0].plot(subset['num_steps'], subset['rmsd'], 
                       marker='o', label=f'MAX_T={max_t}')
    axes[1, 0].set_xlabel('NUM_STEPS')
    axes[1, 0].set_ylabel('RMSD (Å)')
    axes[1, 0].set_title('RMSD vs NUM_STEPS')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True)
    
    # Score Norms比较
    x = np.arange(len(df))
    width = 0.35
    axes[1, 1].bar(x - width/2, df['rot_score_norm'], width, label='Rotation', alpha=0.8)
    axes[1, 1].bar(x + width/2, df['trans_score_norm'], width, label='Translation', alpha=0.8)
    axes[1, 1].set_xlabel('Experiment Index')
    axes[1, 1].set_ylabel('Score Norm')
    axes[1, 1].set_title('Score Norms Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(df, output_dir):
    """生成分析报告"""
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("参数扫描分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PDB文件: {PDB_PATH}\n")
        f.write(f"链ID: {CHAIN_ID}\n")
        f.write(f"实验总数: {len(df)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("参数设置\n")
        f.write("=" * 80 + "\n")
        f.write(f"NUM_STEPS范围: {NUM_STEPS_LIST}\n")
        f.write(f"MAX_T范围: {MAX_T_LIST}\n")
        f.write(f"MIN_T (固定): {MIN_T}\n")
        f.write(f"NOISE_SCALE (固定): {NOISE_SCALE}\n")
        f.write(f"SELF_CONDITIONING: {ENABLE_SELF_CONDITIONING}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("整体统计\n")
        f.write("=" * 80 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\n")
        
        if 'tm_score' in df.columns and df['tm_score'].notna().any():
            f.write("=" * 80 + "\n")
            f.write("最佳结果 (按TM-Score)\n")
            f.write("=" * 80 + "\n")
            best_idx = df['tm_score'].idxmax()
            best_row = df.loc[best_idx]
            f.write(f"NUM_STEPS: {best_row['num_steps']}\n")
            f.write(f"MAX_T: {best_row['max_t']}\n")
            f.write(f"TM-Score: {best_row['tm_score']:.4f}\n")
            f.write(f"RMSD: {best_row['rmsd']:.4f} Å\n")
            f.write(f"Rotation Score Norm: {best_row['rot_score_norm']:.4f}\n")
            f.write(f"Translation Score Norm: {best_row['trans_score_norm']:.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("最差结果 (按TM-Score)\n")
            f.write("=" * 80 + "\n")
            worst_idx = df['tm_score'].idxmin()
            worst_row = df.loc[worst_idx]
            f.write(f"NUM_STEPS: {worst_row['num_steps']}\n")
            f.write(f"MAX_T: {worst_row['max_t']}\n")
            f.write(f"TM-Score: {worst_row['tm_score']:.4f}\n")
            f.write(f"RMSD: {worst_row['rmsd']:.4f} Å\n")
            f.write(f"Rotation Score Norm: {worst_row['rot_score_norm']:.4f}\n")
            f.write(f"Translation Score Norm: {worst_row['trans_score_norm']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("按NUM_STEPS分组统计\n")
        f.write("=" * 80 + "\n")
        grouped_steps = df.groupby('num_steps').agg({
            'tm_score': ['mean', 'std', 'min', 'max'] if 'tm_score' in df.columns else [],
            'rmsd': ['mean', 'std', 'min', 'max'],
            'rot_score_norm': ['mean', 'std'],
            'trans_score_norm': ['mean', 'std']
        })
        f.write(grouped_steps.to_string())
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("按MAX_T分组统计\n")
        f.write("=" * 80 + "\n")
        grouped_maxt = df.groupby('max_t').agg({
            'tm_score': ['mean', 'std', 'min', 'max'] if 'tm_score' in df.columns else [],
            'rmsd': ['mean', 'std', 'min', 'max'],
            'rot_score_norm': ['mean', 'std'],
            'trans_score_norm': ['mean', 'std']
        })
        f.write(grouped_maxt.to_string())
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("详细结果表\n")
        f.write("=" * 80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
    
    print(f"✅ 分析报告已保存: {report_path}")


def main():
    print("=" * 80)
    print("参数扫描分析器")
    print("=" * 80)
    print(f"PDB: {PDB_PATH}")
    print(f"链ID: {CHAIN_ID}")
    print(f"输出目录: {RUN_OUTPUT_DIR}")
    print(f"参数组合数: {len(NUM_STEPS_LIST)} × {len(MAX_T_LIST)} = {len(NUM_STEPS_LIST) * len(MAX_T_LIST)}")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    
    # 保存配置
    config_dict = {
        'pdb_path': PDB_PATH,
        'chain_id': CHAIN_ID,
        'num_steps_list': NUM_STEPS_LIST,
        'max_t_list': MAX_T_LIST,
        'min_t': MIN_T,
        'noise_scale': NOISE_SCALE,
        'enable_self_conditioning': ENABLE_SELF_CONDITIONING,
        'timestamp': TIMESTAMP,
    }
    with open(os.path.join(RUN_OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 加载配置和模型
    print("\n加载配置和模型...")
    conf = OmegaConf.load(CONF_PATH)
    
    # 加载PDB
    print(f"加载PDB文件...")
    pdb_feats = du.parse_pdb_feats('query', PDB_PATH, chain_id=CHAIN_ID)
    chain_feats = process_chain_feats(pdb_feats)
    bb_mask = np.array(pdb_feats['bb_mask']).astype(bool)
    num_res = int(np.sum(bb_mask))
    print(f"残基数: {num_res}")
    
    # 初始化模型
    print("初始化模型...")
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    model = score_network.ScoreNetwork(conf.model, diffuser)
    model.to(device)
    model.eval()
    
    # 加载权重
    print(f"加载权重: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # 运行参数扫描
    print("\n" + "=" * 80)
    print("开始参数扫描")
    print("=" * 80)
    
    results = []
    total_experiments = len(NUM_STEPS_LIST) * len(MAX_T_LIST)
    
    with tqdm(total=total_experiments, desc="参数扫描进度") as pbar:
        for num_steps in NUM_STEPS_LIST:
            for max_t in MAX_T_LIST:
                print(f"\n运行实验: NUM_STEPS={num_steps}, MAX_T={max_t}")
                
                output_subdir = os.path.join(RUN_OUTPUT_DIR, 
                                            f'steps{num_steps}_maxT{max_t:.2f}')
                
                try:
                    result = run_single_experiment(
                        model=model,
                        diffuser=diffuser,
                        chain_feats=chain_feats,
                        pdb_feats=pdb_feats,
                        bb_mask=bb_mask,
                        num_steps=num_steps,
                        max_t=max_t,
                        device=device,
                        output_subdir=output_subdir
                    )
                    
                    results.append(result)
                    
                    tm_str = f"{result['tm_score']:.4f}" if result['tm_score'] is not None else "N/A"
                    print(f"  ✓ TM-Score: {tm_str}")
                    print(f"  ✓ RMSD: {result['rmsd']:.4f} Å")
                    print(f"  ✓ 文件已保存到: {output_subdir}")
                    
                except Exception as e:
                    print(f"  ✗ 实验失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                pbar.update(1)
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果CSV
    csv_path = os.path.join(RUN_OUTPUT_DIR, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 结果CSV已保存: {csv_path}")
    
    # 生成可视化
    print("\n生成可视化图表...")
    try:
        create_visualizations(df, RUN_OUTPUT_DIR)
        print("✅ 可视化图表已生成")
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")
    
    # 生成报告
    print("\n生成分析报告...")
    try:
        generate_report(df, RUN_OUTPUT_DIR)
        print("✅ 分析报告已生成")
    except Exception as e:
        print(f"⚠️ 报告生成失败: {e}")
    
    print("\n" + "=" * 80)
    print("参数扫描完成！")
    print("=" * 80)
    print(f"📁 所有结果保存在: {RUN_OUTPUT_DIR}")
    print(f"📊 结果CSV: results.csv")
    print(f"📈 可视化图表: heatmaps.png, trend_plots.png")
    print(f"📄 分析报告: analysis_report.txt")
    print("=" * 80)


if __name__ == '__main__':
    main()
