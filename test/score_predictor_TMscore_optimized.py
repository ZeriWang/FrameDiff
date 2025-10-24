#!/usr/bin/env python3
"""
优化版Score预测器 - 实现TM-score > 0.9目标

主要优化:
1. 增加采样数量至30个样本以提高成功率
2. 优化采样参数（200步，min_t=0.001，noise_scale=0.8）
3. 从部分加噪状态开始（start_t=0.3-0.5）
4. 保存最后一步的pred_rot_score和pred_trans_score
5. 自动选择TM-score最高的样本
6. 达到目标后提前停止

基于: score_predictor_TMscore_fixed.py
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
from model import score_network
from openfold.data import data_transforms
from openfold.utils import rigid_utils as ru
from openfold.np import protein
from openfold.np import residue_constants

try:
    import tmtools
    TMTOOLS_AVAILABLE = True
except ImportError:
    TMTOOLS_AVAILABLE = False
    print("Warning: tmtools not available, TM-score calculation will be skipped")

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 输入参数
PDB_PATH = str(PROJECT_ROOT / 'test' / 'pdb_dir' / '1AKE.pdb')
CHAIN_ID = 'B'
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'output_dir_optimized')
WEIGHTS_PATH = str(PROJECT_ROOT / 'weights' / 'best_weights.pth')
CONF_PATH = str(PROJECT_ROOT / 'config' / 'base.yaml')

# 优化的采样参数 - 针对TM-score > 0.9
NUM_SAMPLES = 30          # 生成的样本数量（增加以提高成功率）
NUM_DIFFUSION_STEPS = 200  # 逆向扩散步数（平衡精度和速度）
MIN_T = 0.001             # 最小时间步（更小以更接近原始结构）
NOISE_SCALE = 0.8         # 噪声缩放因子（降低随机性）
START_T_RANGE = (0.3, 0.5)  # 初始时间步范围（从部分加噪开始，提高相似度）
ENABLE_SELF_CONDITIONING = True  # 是否在采样时应用自条件
USE_FORWARD_MARGINAL_INIT = True  # 是否通过前向扩散得到初始状态
TARGET_TM_SCORE = 0.9     # 目标TM-score阈值
EARLY_STOP = True         # 达到目标后是否提前停止


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
        if rigids_t.shape[-1] != 7:
            raise ValueError(f"Expected rigid tensor with last dim=7, got {rigids_t.shape}")
        rigid_tensor = rigids_t
    else:
        rigid_tensor = rigids_t.to_tensor_7()

    if rigid_tensor.ndim == 2:
        rigid_tensor = rigid_tensor.unsqueeze(0)
    elif rigid_tensor.ndim != 3:
        raise ValueError(f"Unexpected rigid tensor shape: {rigid_tensor.shape}")

    rigids_batch = ru.Rigid.from_tensor_7(rigid_tensor)
    batch_size, num_res = rigid_tensor.shape[0], rigid_tensor.shape[1]

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
        atom37_mask = atom37_mask.detach().cpu().numpy().astype(np.float32)

    if isinstance(aatype, torch.Tensor):
        aatype = aatype.detach().cpu().numpy()
    if isinstance(residue_index, torch.Tensor):
        residue_index = residue_index.detach().cpu().numpy()

    if atom37_pos.shape[0] != len(aatype):
        raise ValueError(
            f"aatype length {len(aatype)} differs from atom positions {atom37_pos.shape[0]}"
        )

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
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure('ref', pdb_path1)
        structure2 = parser.get_structure('query', pdb_path2)
        
        coords1, seq1 = [], []
        coords2, seq2 = [], []
        
        for model in structure1:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords1.append(residue['CA'].get_coord())
                        seq1.append(residue.get_resname())
        
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
    
    返回:
        dict: 包含最终结构、所有score历史、以及最后一步的score
    """
    sample_feats = copy.deepcopy(init_feats)
    # 强制将所有tensor移到指定设备（修复设备不匹配问题）
    def move_to_device(obj, device):
        if torch.is_tensor(obj):
            return obj.clone().detach().to(device)
        elif isinstance(obj, ru.Rigid):
            # 特殊处理Rigid对象：转换为tensor，移动设备，再转回Rigid
            tensor_7 = obj.to_tensor_7()
            if torch.is_tensor(tensor_7):
                tensor_7 = tensor_7.to(device)
            return ru.Rigid.from_tensor_7(tensor_7)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(v, device) for v in obj)
        else:
            return obj
    
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
                rigids_t = diffuser.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                    rot_score=du.move_to_np(rot_score),
                    trans_score=du.move_to_np(trans_score),
                    diffuse_mask=diffuse_mask,
                    t=float(t),
                    dt=dt,
                    center=True,
                    noise_scale=noise_scale,
                )
                # 更新刚体变换（移除错误的apply_to_point调用）
                sample_feats['rigids_t'] = rigids_t.to_tensor_7()
                
                if embed_self_conditioning:
                    sample_feats['sc_ca_t'] = model_out['rigids'][..., 4:]
            else:
                rigids_t = ru.Rigid.from_tensor_7(sample_feats['rigids_t'])

    return {
        'final_rigids': rigids_t,
        'all_rot_scores': all_rot_scores,
        'all_trans_scores': all_trans_scores,
        'final_rot_score': final_rot_score,  # 最后一步的旋转score
        'final_trans_score': final_trans_score,  # 最后一步的平移score
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

        if USE_FORWARD_MARGINAL_INIT:
            effective_start_t = np.random.uniform(*START_T_RANGE)
            ref_sample = diffuser.forward_marginal(
                rigids_0=rigids_0,
                t=effective_start_t,
                diffuse_mask=diffuse_mask_np,
                as_tensor_7=True
            )
            # 正确的tensor构造方式（避免警告）
            if isinstance(ref_sample['rigids_t'], torch.Tensor):
                rigids_t_tensor = ref_sample['rigids_t'].clone().detach().to(dtype=torch.float32, device=device)
            else:
                rigids_t_tensor = torch.from_numpy(ref_sample['rigids_t']).to(dtype=torch.float32, device=device)
        else:
            effective_start_t = 1.0
            ref_sample = diffuser.sample_ref(
                n_samples=num_res,
                diffuse_mask=diffuse_mask_np,
                as_tensor_7=True
            )
            # 正确的tensor构造（避免警告）
            if isinstance(ref_sample, torch.Tensor):
                rigids_t_tensor = ref_sample.clone().detach().to(dtype=torch.float32, device=device)
            else:
                rigids_t_tensor = torch.from_numpy(ref_sample).to(dtype=torch.float32, device=device)

        print(f"  起始时间步 t0 = {effective_start_t:.4f}")

        init_feats = {
            'res_mask': res_mask_tensor.unsqueeze(0),
            'seq_idx': seq_idx_tensor.unsqueeze(0),
            'fixed_mask': fixed_mask_tensor.unsqueeze(0),
            'torsion_angles_sin_cos': torsion_tensor.unsqueeze(0),
            'sc_ca_t': sc_ca_tensor.unsqueeze(0).clone(),
            'rigids_t': rigids_t_tensor.unsqueeze(0),
        }

        # 最后确保：强制所有tensor都在GPU上（关键修复）
        init_feats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in init_feats.items()}

        sample_out = reverse_diffusion_sampling(
            model=model,
            diffuser=diffuser,
            init_feats=init_feats,
            num_steps=num_steps,
            min_t=min_t,
            start_t=effective_start_t,
            device=device,
            noise_scale=NOISE_SCALE,
            enable_self_conditioning=ENABLE_SELF_CONDITIONING,
        )

        final_rigids = sample_out['final_rigids']

        try:
            prot = rigids_to_protein(final_rigids, aatype, residue_index)
            pdb_filename = f'{pdb_name}_sample_{sample_idx:03d}.pdb'
            pdb_path = os.path.join(samples_dir, pdb_filename)
            save_protein_to_pdb(prot, pdb_path)

            tm_score = calculate_tm_score(reference_pdb, pdb_path)

            print(f"  已保存: {pdb_filename}")
            if tm_score is not None:
                print(f"  TM-score: {tm_score:.4f}")
                if tm_score > best_tm_score:
                    best_tm_score = tm_score
                    best_sample_idx = sample_idx
                
                # 检查是否达到目标
                if tm_score >= TARGET_TM_SCORE:
                    print(f"  ✓ 达到目标! TM-score = {tm_score:.4f} >= {TARGET_TM_SCORE}")

            result = {
                'sample_idx': sample_idx,
                'pdb_path': pdb_path,
                'tm_score': tm_score,
                'start_t': effective_start_t,
                'final_rot_score': sample_out['final_rot_score'],
                'final_trans_score': sample_out['final_trans_score'],
                'rot_scores_history': sample_out['all_rot_scores'],
                'trans_scores_history': sample_out['all_trans_scores']
            }
            all_results.append(result)
            
            # 如果达到目标且启用提前停止，则退出
            if EARLY_STOP and tm_score is not None and tm_score >= TARGET_TM_SCORE:
                print(f"\n{'='*60}")
                print(f"✓ 已达到目标TM-score阈值 ({TARGET_TM_SCORE})，提前停止")
                print(f"{'='*60}")
                break

        except Exception as e:
            print(f"  生成样本失败: {e}")
            import traceback
            traceback.print_exc()

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
        f.write("=" * 80 + "\n")
        f.write("优化版Score预测器结果摘要\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总样本数: {len(all_results)}\n")
        f.write(f"采样步数: {NUM_DIFFUSION_STEPS}\n")
        f.write(f"最小时间步: {MIN_T}\n")
        f.write(f"噪声缩放: {NOISE_SCALE}\n")
        f.write(f"起始时间范围: {START_T_RANGE}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("所有样本的TM-score:\n")
        f.write("-" * 80 + "\n")
        
        tm_scores = []
        for result in all_results:
            sample_idx = result['sample_idx']
            tm_score = result['tm_score']
            start_t = result['start_t']
            
            if tm_score is not None:
                tm_scores.append(tm_score)
                status = "✓ 达标" if tm_score >= TARGET_TM_SCORE else ""
                f.write(f"样本 {sample_idx+1:3d}: TM-score = {tm_score:.4f}  (start_t={start_t:.3f})  {status}\n")
            else:
                f.write(f"样本 {sample_idx+1:3d}: TM-score = N/A\n")
        
        if tm_scores:
            f.write("\n" + "=" * 80 + "\n")
            f.write("统计信息:\n")
            f.write("=" * 80 + "\n")
            f.write(f"平均 TM-score: {np.mean(tm_scores):.4f}\n")
            f.write(f"最高 TM-score: {np.max(tm_scores):.4f}\n")
            f.write(f"最低 TM-score: {np.min(tm_scores):.4f}\n")
            f.write(f"标准差: {np.std(tm_scores):.4f}\n")
            f.write(f"达标样本数 (>={TARGET_TM_SCORE}): {sum(1 for s in tm_scores if s >= TARGET_TM_SCORE)}\n")
            
            if best_result:
                f.write("\n" + "=" * 80 + "\n")
                f.write("最佳样本详细信息:\n")
                f.write("=" * 80 + "\n")
                f.write(f"样本索引: {best_result['sample_idx'] + 1}\n")
                f.write(f"TM-score: {best_result['tm_score']:.4f}\n")
                f.write(f"起始时间步: {best_result['start_t']:.4f}\n")
                f.write(f"PDB文件: {best_result['pdb_path']}\n")
                f.write(f"\n最后一步的Score (最接近原始结构时):\n")
                f.write(f"  - 旋转score形状: {best_result['final_rot_score'].shape}\n")
                f.write(f"  - 平移score形状: {best_result['final_trans_score'].shape}\n")
                
                # 保存最佳样本的最后一步score
                rot_score_path = os.path.join(output_dir, f'{pdb_name}_best_final_rot_score.npy')
                trans_score_path = os.path.join(output_dir, f'{pdb_name}_best_final_trans_score.npy')
                
                np.save(rot_score_path, best_result['final_rot_score'])
                np.save(trans_score_path, best_result['final_trans_score'])
                
                f.write(f"\n最佳样本的最后一步Score已保存:\n")
                f.write(f"  - {rot_score_path}\n")
                f.write(f"  - {trans_score_path}\n")

    print(f"\n{'='*80}")
    print(f"结果已保存: {summary_path}")
    
    if best_result:
        print(f"\n最佳样本详情:")
        print(f"  样本: {best_result['sample_idx'] + 1}")
        print(f"  TM-score: {best_result['tm_score']:.4f}")
        print(f"  PDB: {best_result['pdb_path']}")
        print(f"  最后一步旋转score: {output_dir}/{pdb_name}_best_final_rot_score.npy")
        print(f"  最后一步平移score: {output_dir}/{pdb_name}_best_final_trans_score.npy")
    print(f"{'='*80}")


def main():
    print("=" * 80)
    print("优化版Score预测器 - 目标: TM-score > 0.9")
    print("=" * 80)
    
    # 加载配置
    conf = OmegaConf.load(CONF_PATH)
    
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n使用设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"\n使用设备: CPU")
    
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
        torsion_list = []
        for item in torsion_angles:
            if isinstance(item, np.ndarray):
                torsion_list.append(item)
            else:
                # 修复np.zeros参数：应该是shape=(7,2)
                torsion_list.append(np.zeros((7, 2), dtype=np.float32))
        torsion_angles = np.array(torsion_list, dtype=np.float32)
    else:
        torsion_angles = torsion_angles.astype(np.float32)

    # 在GPU上创建rigids_0（避免设备不匹配）
    rigid_frames = chain_feats['rigidgroups_gt_frames'][mask_tensor, 0].detach().float().to(device)
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

    # 生成样本
    pdb_name = os.path.splitext(os.path.basename(PDB_PATH))[0]
    print(f"\n{'='*80}")
    print(f"开始生成样本 - 目标: TM-score > {TARGET_TM_SCORE}")
    print(f"{'='*80}")
    print(f"最大样本数: {NUM_SAMPLES}")
    print(f"逆向扩散步数: {NUM_DIFFUSION_STEPS}")
    print(f"最小时间步: {MIN_T}")
    print(f"噪声缩放: {NOISE_SCALE}")
    print(f"起始时间范围: {START_T_RANGE}")
    print(f"提前停止: {EARLY_STOP}")
    
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
    
    print("\n" + "=" * 80)
    print("✓ 完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
