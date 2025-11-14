#!/usr/bin/env python3
"""
逐步去噪分析器

功能:
1. 与direct_denoising_predictor.py相同的去噪功能
2. 每进行一步去噪，就保存该步的旋转分数和平移分数为.npy格式
3. 去噪结束后，自动调用score_distance_analyzer.py进行分析

基于: direct_denoising_predictor.py
"""
import os
import sys
import torch
import numpy as np
import traceback
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
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

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 输入参数
PDB_PATH = str(PROJECT_ROOT / 'test' / 'pdb_dir' / '1TFU.pdb')
CHAIN_ID = 'A'
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'output_dir_stepwise_denoising')
WEIGHTS_PATH = str(PROJECT_ROOT / 'weights' / 'best_weights.pth')
CONF_PATH = str(PROJECT_ROOT / 'config' / 'base.yaml')

# 直接去噪参数
NUM_DENOISING_STEPS = 5     # 去噪步数
MIN_T = 0.01                 # 最小时间步
MAX_T = 0.05                 # 最大时间步（从很小的噪声开始）
NOISE_SCALE = 0              # 极小的噪声缩放因子
ENABLE_SELF_CONDITIONING = True  # 启用自条件
SAVE_STEPWISE_SCORES = True  # 保存每步的分数


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
        rigid_tensor = rigids_t.detach().cpu().float()
    elif isinstance(rigids_t, ru.Rigid):
        rigid_tensor = rigids_t.to_tensor_7().detach().cpu().float()
    else:
        raise ValueError(f"不支持的rigids_t类型: {type(rigids_t)}")

    if rigid_tensor.ndim == 2:
        rigid_tensor = rigid_tensor.unsqueeze(0)
    elif rigid_tensor.ndim != 3:
        raise ValueError(f"rigids_t维度错误: {rigid_tensor.ndim}")

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
        atom37_pos = atom37_pos.detach().cpu().numpy()
    if isinstance(atom37_mask, torch.Tensor):
        atom37_mask = atom37_mask.detach().cpu().numpy()

    if isinstance(aatype, torch.Tensor):
        aatype = aatype.detach().cpu().numpy()
    if isinstance(residue_index, torch.Tensor):
        residue_index = residue_index.detach().cpu().numpy()

    if atom37_pos.shape[0] != len(aatype):
        raise ValueError(f"原子位置数量({atom37_pos.shape[0]})与序列长度({len(aatype)})不匹配")

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
    """
    计算TM-score，支持指定链ID。
    
    Args:
        pdb_path1: 参考PDB路径 (原始结构)
        pdb_path2: 查询PDB路径 (去噪后结构)
        chain_id1: 参考PDB中要读取的链ID (如 'B')
        chain_id2: 查询PDB中要读取的链ID (如果不指定，默认读取第一条链)
    """
    if not TMTOOLS_AVAILABLE:
        return None
    
    try:
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure('ref', pdb_path1)
        structure2 = parser.get_structure('query', pdb_path2)
        
        def get_coords_and_seq(structure, target_chain_id=None):
            coords = []
            seq = []
            # 只读取第一个模型
            model = next(iter(structure))
            
            for chain in model:
                if target_chain_id is not None and chain.id != target_chain_id:
                    continue
                for residue in chain:
                    if residue.id[0] == ' ':
                        if 'CA' in residue:
                            coords.append(residue['CA'].get_coord())
                            seq.append(residue.get_resname())
                if target_chain_id is not None:
                    break
            
            return np.array(coords), seq

        # 提取坐标和序列
        # 原始结构：必须指定链ID (如 'B')
        coords1, seq1 = get_coords_and_seq(structure1, chain_id1)
        # 去噪后结构：通常只有一条链，可以不指定，或者指定为默认的 'A'
        coords2, seq2 = get_coords_and_seq(structure2, chain_id2)
        
        if len(coords1) == 0 or len(coords2) == 0:
            print(f"警告: 未能在指定链中找到CA原子 (Chain1: {chain_id1}, Chain2: {chain_id2})")
            return None

        from data.residue_constants import restype_3to1
        seq1_str = ''.join([restype_3to1.get(res, 'X') for res in seq1])
        seq2_str = ''.join([restype_3to1.get(res, 'X') for res in seq2])
        
        result = tmtools.tm_align(coords1, coords2, seq1_str, seq2_str)
        return result.tm_norm_chain1
        
    except Exception as e:
        print(f"计算TM-score失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def move_to_device(obj, device):
    """改进的设备转移函数"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, ru.Rigid):
        tensor_7 = obj.to_tensor_7()
        if torch.is_tensor(tensor_7):
            tensor_7_device = tensor_7.to(device)
            return ru.Rigid.from_tensor_7(tensor_7_device)
        return obj
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    else:
        return obj


def stepwise_denoising(
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
        output_dir=None,
        output_prefix=None,
    ):
    """
    逐步去噪过程：从原始结构开始，进行轻微的去噪优化，每步保存分数
    
    返回:
        dict: 包含最终结构、score历史、以及所有保存的分数文件路径
    """
    print(f"开始逐步去噪过程...")
    print(f"  去噪步数: {num_steps}")
    print(f"  时间范围: {min_t} -> {max_t}")
    print(f"  噪声缩放: {noise_scale}")
    print(f"  每步保存分数: {SAVE_STEPWISE_SCORES}")
    
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
    dt = float((max_t - min_t) / max(num_steps - 1, 1))  # dt为正数，表示步长大小
    
    all_rot_scores = []
    all_trans_scores = []
    saved_score_files = []
    
    # 准备numpy版本的mask用于diffuser.reverse
    diffuse_mask = ((1 - sample_feats['fixed_mask']) * sample_feats['res_mask']).detach().cpu().numpy()
    fixed_mask_np = (sample_feats['fixed_mask'] * sample_feats['res_mask']).detach().cpu().numpy()
    diffuse_mask_np = diffuse_mask  # 保持引用一致性
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

            # 保存每步的分数到文件
            if SAVE_STEPWISE_SCORES and output_dir is not None and output_prefix is not None:
                step_rot_path = os.path.join(output_dir, f'{output_prefix}_step{step_idx:03d}_rot_score.npy')
                step_trans_path = os.path.join(output_dir, f'{output_prefix}_step{step_idx:03d}_trans_score.npy')
                
                rot_score_np = du.move_to_np(rot_score)
                trans_score_np = du.move_to_np(trans_score)
                
                np.save(step_rot_path, rot_score_np)
                np.save(step_trans_path, trans_score_np)
                
                saved_score_files.append({
                    'step': step_idx,
                    't': float(t),
                    'rot_score_path': step_rot_path,
                    'trans_score_path': step_trans_path,
                })
                
                if step_idx == 0 or step_idx == len(denoising_steps) - 1:
                    print(f"  步骤 {step_idx}: t={t:.4f}, 已保存分数")

            # 如果是最后一步，保存score
            if step_idx == len(denoising_steps) - 1:
                final_rot_score = du.move_to_np(rot_score)
                final_trans_score = du.move_to_np(trans_score)

            # 执行去噪步骤
            if t > min_t:
                # 转换为numpy数组进行计算
                rot_score_np = du.move_to_np(rot_score)
                trans_score_np = du.move_to_np(trans_score)
                
                perturb_rot_score = diffuse_mask[..., None] * rot_score_np
                perturb_trans_score = diffuse_mask[..., None] * trans_score_np

                rigids_t = ru.Rigid.from_tensor_7(sample_feats['rigids_t'])
                if noise_scale > 0:
                    gt_rot_score, gt_trans_score = diffuser.score(
                        rigids_t, sample_feats['t'], use_torch=False
                    )
                    perturb_rot_score += noise_scale * gt_rot_score
                    perturb_trans_score += noise_scale * gt_trans_score

                # diffuser.reverse需要numpy数组和Python标量
                rigids_t_next = diffuser.reverse(
                    rigid_t=rigids_t,
                    rot_score=perturb_rot_score,
                    trans_score=perturb_trans_score,
                    diffuse_mask=diffuse_mask,
                    t=float(t),
                    dt=float(dt),
                    center=True,
                    noise_scale=0.0,
                )
                
                # 确保结果移回GPU，并保持批次维度
                rigids_t_tensor = rigids_t_next.to_tensor_7()
                if rigids_t_tensor.ndim == 2:
                    rigids_t_tensor = rigids_t_tensor.unsqueeze(0)
                sample_feats['rigids_t'] = rigids_t_tensor.to(device)

                if embed_self_conditioning:
                    sample_feats['sc_ca_t'] = model_out['rigids'][..., 4:]
            else:
                rigids_t = ru.Rigid.from_tensor_7(sample_feats['rigids_t'])

    print(f"去噪完成！共保存 {len(saved_score_files)} 步的分数")
    
    return {
        'final_rigids': rigids_t,
        'all_rot_scores': all_rot_scores,
        'all_trans_scores': all_trans_scores,
        'final_rot_score': final_rot_score,
        'final_trans_score': final_trans_score,
        'fixed_mask': fixed_mask_np,
        'diffuse_mask': diffuse_mask_np,
        'saved_score_files': saved_score_files,
    }



def main():
    print("=" * 80)
    print("逐步去噪分析器")
    print("=" * 80)
    print("功能:")
    print("  1. 直接对原始PDB结构进行去噪")
    print("  2. 每一步去噪后保存旋转分数和平移分数")
    print("  3. 去噪结束后自动分析所有分数")
    
    # 加载配置
    conf = OmegaConf.load(CONF_PATH)
    
    # 设备
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用设备: GPU ({torch.cuda.get_device_name()})")
    else:
        device = 'cpu'
        print("使用设备: CPU")
    
    # 加载PDB
    print(f"\n加载原始PDB: {PDB_PATH}")
    pdb_feats = du.parse_pdb_feats('query', PDB_PATH, chain_id=CHAIN_ID)
    chain_feats = process_chain_feats(pdb_feats)
    bb_mask = np.array(pdb_feats['bb_mask']).astype(bool)
    num_res = int(np.sum(bb_mask))
    print(f"残基数: {num_res}")
    
    # 初始化模型
    print("\n初始化模型...")
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
    
    # 准备特征
    mask_tensor = torch.from_numpy(bb_mask).to(torch.bool)
    torsion_angles = chain_feats['torsion_angles_sin_cos'].detach().cpu().numpy()[bb_mask]
    if torsion_angles.dtype == np.object_:
        torsion_angles_list = []
        for i in range(torsion_angles.shape[0]):
            torsion_angles_list.append(torsion_angles[i].astype(np.float32))
        torsion_angles = np.stack(torsion_angles_list, axis=0)
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
    
    # 提取aatype和residue_index用于PDB生成
    aatype = pdb_feats['aatype'][bb_mask]
    residue_index = pdb_feats['residue_index'][bb_mask]
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdb_name = os.path.splitext(os.path.basename(PDB_PATH))[0]
    # 构建包含链ID的输出文件名前缀
    output_prefix = f'{pdb_name}_{CHAIN_ID}'
    
    print(f"\n{'='*80}")
    print("开始逐步去噪过程")
    print(f"{'='*80}")
    print(f"去噪参数:")
    print(f"  步数: {NUM_DENOISING_STEPS}")
    print(f"  时间范围: {MIN_T} -> {MAX_T}")
    print(f"  噪声缩放: {NOISE_SCALE}")
    print(f"  自条件: {ENABLE_SELF_CONDITIONING}")
    
    # 执行逐步去噪
    denoising_result = stepwise_denoising(
        model=model,
        diffuser=diffuser,
        original_rigids=rigids_0,
        res_mask=res_mask_tensor,
        seq_idx=seq_idx_tensor,
        fixed_mask=fixed_mask_tensor,
        torsion_angles=torsion_tensor,
        sc_ca=sc_ca_tensor,
        num_steps=NUM_DENOISING_STEPS,
        min_t=MIN_T,
        max_t=MAX_T,
        device=device,
        noise_scale=NOISE_SCALE,
        enable_self_conditioning=ENABLE_SELF_CONDITIONING,
        output_dir=OUTPUT_DIR,
        output_prefix=output_prefix,
    )
    
    # 转换为PDB并保存
    print("\n转换去噪结果为PDB格式...")
    final_rigids = denoising_result['final_rigids']
    try:
        denoised_protein = rigids_to_protein(final_rigids, aatype, residue_index)
        
        # 保存去噪后的PDB
        output_pdb = os.path.join(OUTPUT_DIR, f'{output_prefix}_denoised.pdb')
        save_protein_to_pdb(denoised_protein, output_pdb)
        print(f"去噪后PDB已保存: {output_pdb}")
        
        # 计算TM-score
        print("\n计算TM-score...")
        tm_score = calculate_tm_score(PDB_PATH, output_pdb, chain_id1=CHAIN_ID, chain_id2=None)
        if tm_score is not None:
            print(f"TM-score (去噪后 vs 原始): {tm_score:.4f}")
        else:
            print("TM-score计算失败")
            tm_score = 0.0
        
        # 保存详细结果摘要
        summary_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_stepwise_denoising_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"逐步去噪结果摘要\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"输入PDB: {PDB_PATH}\n")
            f.write(f"链ID: {CHAIN_ID}\n")
            f.write(f"残基数: {num_res}\n")
            f.write(f"输出目录: {OUTPUT_DIR}\n\n")
            
            f.write(f"去噪参数:\n")
            f.write(f"  步数: {NUM_DENOISING_STEPS}\n")
            f.write(f"  时间范围: {MIN_T} -> {MAX_T}\n")
            f.write(f"  噪声缩放: {NOISE_SCALE}\n")
            f.write(f"  自条件: {ENABLE_SELF_CONDITIONING}\n\n")
            
            f.write(f"TM-score (去噪后 vs 原始): {tm_score:.4f}\n\n")
            
            f.write(f"保存的分数文件:\n")
            f.write(f"{'='*60}\n")
            for file_info in denoising_result['saved_score_files']:
                f.write(f"步骤 {file_info['step']:03d} (t={file_info['t']:.4f}):\n")
                f.write(f"  旋转分数: {os.path.basename(file_info['rot_score_path'])}\n")
                f.write(f"  平移分数: {os.path.basename(file_info['trans_score_path'])}\n")
        
        print(f"\n详细摘要已保存: {summary_path}")
        
    except Exception as e:
        print(f"生成去噪结构失败: {e}")
        traceback.print_exc()
        return
    
    # 运行分数距离分析
    print(f"\n{'='*80}")
    print("准备运行分数距离分析...")
    print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print("逐步去噪完成！")
    print(f"{'='*80}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 去噪后PDB: {os.path.basename(output_pdb)}")
    print(f"📊 TM-score: {tm_score:.4f}")
    print(f"💾 共保存 {len(denoising_result['saved_score_files'])} 步的分数文件")
    print(f"📋 摘要文件: {os.path.basename(summary_path)}")
    print(f"\n提示: 要分析保存的分数，请运行score_distance_analyzer.py")
    print(f"      并将INPUT_DIR设置为: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
