#!/usr/bin/env python3
"""
修复设备不匹配问题的Score预测器

主要修复:
1. 确保所有张量都在GPU上
2. 修复diffuser.reverse()返回值的设备问题
3. 增强move_to_device函数对Rigid对象的处理
4. 在关键位置添加设备检查和转换

基于: score_predictor_TMscore_optimized.py
"""
import os
import sys
import copy
import torch
import numpy as np
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
PDB_PATH = str(PROJECT_ROOT / 'test' / 'pdb_dir' / '1AKE.pdb')
CHAIN_ID = 'B'
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'output_dir_fixed_device')
WEIGHTS_PATH = str(PROJECT_ROOT / 'weights' / 'best_weights.pth')
CONF_PATH = str(PROJECT_ROOT / 'config' / 'base.yaml')

# 激进优化的采样参数 - 针对TM-score > 0.9
NUM_SAMPLES = 5          # 增加生成的样本数量以提高成功率
NUM_DIFFUSION_STEPS = 300  # 增加逆向扩散步数以提高精度
MIN_T = 0.001            # 更小的最小时间步以更接近原始结构
NOISE_SCALE = 0.3         # 大幅降低噪声缩放因子
START_T_RANGE = (0.01, 0.03)  # 从很小的噪声开始，几乎从原始结构开始
ENABLE_SELF_CONDITIONING = True  # 保持自条件
USE_FORWARD_MARGINAL_INIT = True  # 通过前向扩散得到初始状态
TARGET_TM_SCORE = 0.9     # 目标TM-score阈值
EARLY_STOP = True         # 达到目标后提前停止
SAVE_SCORES = True        # 保存最终分数为npy格式


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


def calculate_tm_score(pdb_path1, pdb_path2):
    """计算TM-score"""
    if not TMTOOLS_AVAILABLE:
        return None
    
    try:
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure('target', pdb_path1)
        structure2 = parser.get_structure('pred', pdb_path2)
        
        coords1 = []
        seq1 = []
        for model in structure1:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords1.append(residue['CA'].get_coord())
                        seq1.append(residue.get_resname())
        
        coords2 = []
        seq2 = []
        for model in structure2:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords2.append(residue['CA'].get_coord())
                        seq2.append(residue.get_resname())
        
        coords1 = np.array(coords1, dtype=np.float64)
        coords2 = np.array(coords2, dtype=np.float64)
        
        from data.residue_constants import restype_3to1
        seq1_str = ''.join([restype_3to1.get(res, 'X') for res in seq1])
        seq2_str = ''.join([restype_3to1.get(res, 'X') for res in seq2])
        
        result = tmtools.tm_align(coords1, coords2, seq1_str, seq2_str)
        return result.tm_norm_chain1
        
    except Exception as e:
        print(f"计算TM-score失败: {e}")
        return None


def move_to_device(obj, device):
    """改进的设备转移函数，特别处理Rigid对象"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, ru.Rigid):
        # 特殊处理Rigid对象
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


def reverse_diffusion_sampling(
        model,
        diffuser,
        init_feats,
        num_steps=100,
        min_t=0.01,
        start_t=1.0,
        device='cuda',
        noise_scale=1.0,
        enable_self_conditioning=True,
    ):
    """
    核心函数：完整的逆向扩散采样过程。
    修复设备不匹配问题。
    
    返回:
        dict: 包含最终结构、所有score历史、以及最后一步的score
    """
    sample_feats = copy.deepcopy(init_feats)
    
    # 使用改进的move_to_device函数
    sample_feats = move_to_device(sample_feats, device)

    if 'rigids_t' not in sample_feats:
        raise KeyError('init_feats 必须包含 rigids_t 用于逆向采样')

    batch_size = sample_feats['rigids_t'].shape[0]
    start_t = float(max(start_t, min_t))
    reverse_steps = np.linspace(min_t, start_t, num_steps)
    reverse_steps = reverse_steps[::-1]
    if reverse_steps.size == 0:
        reverse_steps = np.array([start_t], dtype=np.float32)
    dt = start_t / max(num_steps, 1)

    all_rot_scores = []
    all_trans_scores = []

    diffuse_mask = ((1 - sample_feats['fixed_mask']) * sample_feats['res_mask']).detach().cpu().numpy()
    fixed_mask = (sample_feats['fixed_mask'] * sample_feats['res_mask']).detach().cpu().numpy()
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
        if embed_self_conditioning and reverse_steps.size > 0:
            set_t_feats(sample_feats, reverse_steps[0])
            model_sc = model(sample_feats)
            sample_feats['sc_ca_t'] = model_sc['rigids'][..., 4:]

        for step_idx, t in enumerate(tqdm(reverse_steps, desc="逆向扩散去噪")):
            set_t_feats(sample_feats, t)
            model_out = model(sample_feats)
            rot_score = model_out['rot_score']
            trans_score = model_out['trans_score']

            # 保存所有时间步的score
            all_rot_scores.append({'t': float(t), 'score': du.move_to_np(rot_score)})
            all_trans_scores.append({'t': float(t), 'score': du.move_to_np(trans_score)})

            # 如果是最后一步，保存score（最接近原始结构的时刻）
            if step_idx == len(reverse_steps) - 1:
                final_rot_score = du.move_to_np(rot_score).copy()
                final_trans_score = du.move_to_np(trans_score).copy()

            if t > min_t:
                # 确保输入到diffuser.reverse的rigid_t在CPU上
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

    return {
        'final_rigids': rigids_t,
        'all_rot_scores': all_rot_scores,
        'all_trans_scores': all_trans_scores,
        'final_rot_score': final_rot_score,
        'final_trans_score': final_trans_score,
        'fixed_mask': fixed_mask,
        'diffuse_mask': diffuse_mask,
    }


def generate_samples(
        model,
        diffuser,
        base_feats,
        num_samples,
        num_steps,
        min_t,
        device,
        aatype,
        residue_index,
        reference_pdb,
        output_dir,
        pdb_name
    ):
    """生成多个样本并计算TM-score"""
    num_res = base_feats['num_res']
    all_results = []

    samples_dir = os.path.join(output_dir, f'{pdb_name}_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # 确保所有张量都在正确的设备上
    res_mask_tensor = torch.tensor(base_feats['res_mask'], dtype=torch.float32, device=device)
    seq_idx_tensor = torch.tensor(base_feats['seq_idx'], dtype=torch.float32, device=device)
    fixed_mask_tensor = torch.tensor(base_feats['fixed_mask'], dtype=torch.float32, device=device)
    torsion_tensor = torch.tensor(base_feats['torsion_angles_sin_cos'], dtype=torch.float32, device=device)
    sc_ca_tensor = torch.tensor(base_feats['sc_ca_t'], dtype=torch.float32, device=device)
    diffuse_mask_np = base_feats['res_mask'].astype(np.float32)
    rigids_0 = base_feats['rigids_0']

    best_tm_score = 0.0
    best_sample_idx = -1

    for sample_idx in range(num_samples):
        print(f"\n{'='*60}")
        print(f"生成样本 {sample_idx + 1}/{num_samples}")
        print(f"{'='*60}")
        
        # 生成随机起始时间
        t0 = np.random.uniform(*START_T_RANGE)
        print(f"  起始时间步 t0 = {t0:.4f}")
        
        if USE_FORWARD_MARGINAL_INIT:
            # 通过前向扩散生成初始状态
            ref_sample = diffuser.forward_marginal(
                rigids_0=rigids_0, 
                t=t0, 
                diffuse_mask=diffuse_mask_np
            )
            
            # 处理ref_sample的返回格式
            if isinstance(ref_sample, dict) and 'rigids_t' in ref_sample:
                if torch.is_tensor(ref_sample['rigids_t']):
                    rigids_t_tensor = ref_sample['rigids_t'].clone().detach().to(dtype=torch.float32, device=device)
                else:
                    rigids_t_tensor = torch.from_numpy(ref_sample['rigids_t']).to(dtype=torch.float32, device=device)
            else:
                # ref_sample直接是rigids_t
                if torch.is_tensor(ref_sample):
                    rigids_t_tensor = ref_sample.clone().detach().to(dtype=torch.float32, device=device)
                else:
                    rigids_t_tensor = torch.from_numpy(ref_sample).to(dtype=torch.float32, device=device)
            
            # 确保rigids_t_tensor有正确的batch维度
            if rigids_t_tensor.ndim == 2:  # [num_res, 7]
                rigids_t_tensor = rigids_t_tensor.unsqueeze(0)  # [1, num_res, 7]
            elif rigids_t_tensor.ndim == 3 and rigids_t_tensor.shape[0] != 1:
                print(f"Warning: rigids_t_tensor has unexpected batch size: {rigids_t_tensor.shape}")
        else:
            # 使用纯随机初始化
            rigids_t_tensor = torch.randn((1, num_res, 7), dtype=torch.float32, device=device)
        
        # 构造采样输入 - 添加调试信息
        print(f"  张量形状调试:")
        print(f"    rigids_t_tensor: {rigids_t_tensor.shape}")
        print(f"    res_mask_tensor: {res_mask_tensor.shape} -> {res_mask_tensor.unsqueeze(0).shape}")
        print(f"    fixed_mask_tensor: {fixed_mask_tensor.shape} -> {fixed_mask_tensor.unsqueeze(0).shape}")
        
        sample_feats = {
            'rigids_t': rigids_t_tensor,
            'res_mask': res_mask_tensor.unsqueeze(0),
            'seq_idx': seq_idx_tensor.unsqueeze(0),
            'fixed_mask': fixed_mask_tensor.unsqueeze(0),
            'torsion_angles_sin_cos': torsion_tensor.unsqueeze(0),
            'sc_ca_t': sc_ca_tensor.unsqueeze(0).clone(),
        }
        
        # 执行逆向扩散采样
        print("  开始逆向扩散采样...")
        sample_out = reverse_diffusion_sampling(
            model=model,
            diffuser=diffuser,
            init_feats=sample_feats,
            num_steps=num_steps,
            min_t=min_t,
            start_t=t0,
            device=device,
            noise_scale=NOISE_SCALE,
            enable_self_conditioning=ENABLE_SELF_CONDITIONING,
        )
        
        # 保存最终分数为.npy格式
        if SAVE_SCORES and sample_out['final_rot_score'] is not None:
            rot_score_path = os.path.join(samples_dir, f'sample_{sample_idx + 1:03d}_rot_score.npy')
            trans_score_path = os.path.join(samples_dir, f'sample_{sample_idx + 1:03d}_trans_score.npy')
            np.save(rot_score_path, sample_out['final_rot_score'])
            np.save(trans_score_path, sample_out['final_trans_score'])
            print(f"  已保存分数: {os.path.basename(rot_score_path)}, {os.path.basename(trans_score_path)}")
        
        # 转换为PDB
        print("  转换为PDB格式...")
        final_rigids = sample_out['final_rigids']
        try:
            pred_protein = rigids_to_protein(final_rigids, aatype, residue_index)
            
            # 保存PDB
            sample_pdb = os.path.join(samples_dir, f'sample_{sample_idx + 1:03d}.pdb')
            save_protein_to_pdb(pred_protein, sample_pdb)
            print(f"  已保存: {sample_pdb}")
            
            # 计算TM-score
            tm_score = calculate_tm_score(reference_pdb, sample_pdb)
            if tm_score is not None:
                print(f"  TM-score: {tm_score:.4f}")
                
                if tm_score > best_tm_score:
                    best_tm_score = tm_score
                    best_sample_idx = sample_idx
                    
                # 检查是否达到目标
                if EARLY_STOP and tm_score >= TARGET_TM_SCORE:
                    print(f"\n✓ 达到目标TM-score {TARGET_TM_SCORE}! 提前停止采样。")
                    print(f"  样本 {sample_idx + 1}: TM-score = {tm_score:.4f}")
            else:
                print(f"  TM-score计算失败")
            
            # 保存结果
            result = {
                'sample_idx': sample_idx + 1,
                'sample_pdb': sample_pdb,
                'tm_score': tm_score,
                't0': t0,
                'final_rot_score': sample_out['final_rot_score'],
                'final_trans_score': sample_out['final_trans_score'],
                'all_rot_scores': sample_out['all_rot_scores'],
                'all_trans_scores': sample_out['all_trans_scores'],
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"  生成样本失败: {e}")
            continue
        
        # 如果达到目标且启用早停，退出循环
        if EARLY_STOP and tm_score is not None and tm_score >= TARGET_TM_SCORE:
            break

    print(f"\n{'='*60}")
    print(f"最佳样本: 样本 {best_sample_idx + 1}, TM-score = {best_tm_score:.4f}")
    print(f"{'='*60}")

    return all_results


def save_results(all_results, output_dir, pdb_name):
    """保存结果摘要和最佳样本的score"""
    summary_path = os.path.join(output_dir, f'{pdb_name}_summary.txt')

    # 找到最佳样本
    valid_results = [r for r in all_results if r['tm_score'] is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['tm_score'])
    else:
        best_result = None

    with open(summary_path, 'w') as f:
        f.write(f"Score预测器结果摘要 - 修复设备不匹配版本\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"输入PDB: {PDB_PATH}\n")
        f.write(f"链ID: {CHAIN_ID}\n")
        f.write(f"生成样本数: {len(all_results)}\n")
        f.write(f"目标TM-score: {TARGET_TM_SCORE}\n\n")
        
        if best_result:
            f.write(f"最佳结果:\n")
            f.write(f"  样本ID: {best_result['sample_idx']}\n")
            f.write(f"  TM-score: {best_result['tm_score']:.4f}\n")
            f.write(f"  起始时间: {best_result['t0']:.4f}\n")
            f.write(f"  样本文件: {best_result['sample_pdb']}\n\n")
            
            # 保存最佳样本的score信息
            if best_result['final_rot_score'] is not None:
                f.write(f"最佳样本的最终Score信息:\n")
                f.write(f"  旋转Score shape: {best_result['final_rot_score'].shape}\n")
                f.write(f"  平移Score shape: {best_result['final_trans_score'].shape}\n")
                f.write(f"  旋转Score统计: mean={np.mean(best_result['final_rot_score']):.6f}, "
                       f"std={np.std(best_result['final_rot_score']):.6f}\n")
                f.write(f"  平移Score统计: mean={np.mean(best_result['final_trans_score']):.6f}, "
                       f"std={np.std(best_result['final_trans_score']):.6f}\n\n")
        
        f.write(f"所有样本结果:\n")
        f.write(f"{'ID':<4} {'TM-score':<10} {'起始时间':<10} {'文件路径':<50}\n")
        f.write(f"{'-'*80}\n")
        
        for result in all_results:
            tm_str = f"{result['tm_score']:.4f}" if result['tm_score'] is not None else "失败"
            f.write(f"{result['sample_idx']:<4} {tm_str:<10} {result['t0']:<10.4f} "
                   f"{os.path.basename(result['sample_pdb']):<50}\n")

    print(f"\n{'='*80}")
    print(f"结果已保存: {summary_path}")
    
    if best_result:
        print(f"最佳样本TM-score: {best_result['tm_score']:.4f}")
        if best_result['tm_score'] >= TARGET_TM_SCORE:
            print(f"✓ 成功达到目标TM-score {TARGET_TM_SCORE}!")
        else:
            print(f"✗ 未达到目标TM-score {TARGET_TM_SCORE}")
    print(f"{'='*80}")


def main():
    print("=" * 80)
    print("修复设备不匹配问题的Score预测器")
    print("=" * 80)
    
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
    print(f"\n加载PDB: {PDB_PATH}")
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

    rigid_frames = chain_feats['rigidgroups_gt_frames'][mask_tensor, 0].detach().cpu().float()
    rigids_0 = ru.Rigid.from_tensor_4x4(rigid_frames)
    sc_ca_init = rigids_0.get_trans().detach().cpu().numpy().astype(np.float32)

    base_feats = {
        'num_res': num_res,
        'res_mask': np.ones(num_res, dtype=np.float32),
        'seq_idx': np.arange(1, num_res + 1, dtype=np.float32),
        'fixed_mask': np.zeros(num_res, dtype=np.float32),
        'torsion_angles_sin_cos': torsion_angles,
        'sc_ca_t': sc_ca_init,
        'rigids_0': rigids_0,
    }
    
    # 提取aatype和residue_index用于PDB生成
    aatype = pdb_feats['aatype'][bb_mask]
    residue_index = pdb_feats['residue_index'][bb_mask]
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdb_name = os.path.splitext(os.path.basename(PDB_PATH))[0]
    
    print(f"\n{'='*80}")
    print("开始生成样本 - 修复设备不匹配问题")
    print(f"{'='*80}")
    print(f"最大样本数: {NUM_SAMPLES}")
    print(f"逆向扩散步数: {NUM_DIFFUSION_STEPS}")
    print(f"最小时间步: {MIN_T}")
    print(f"噪声缩放: {NOISE_SCALE}")
    print(f"起始时间范围: {START_T_RANGE}")
    print(f"提前停止: {EARLY_STOP}")
    
    # 生成样本
    all_results = generate_samples(
        model=model,
        diffuser=diffuser,
        base_feats=base_feats,
        num_samples=NUM_SAMPLES,
        num_steps=NUM_DIFFUSION_STEPS,
        min_t=MIN_T,
        device=device,
        aatype=aatype,
        residue_index=residue_index,
        reference_pdb=PDB_PATH,
        output_dir=OUTPUT_DIR,
        pdb_name=pdb_name
    )
    
    # 保存结果
    save_results(all_results, OUTPUT_DIR, pdb_name)


if __name__ == '__main__':
    main()