#!/usr/bin/env python3
"""
重构后的Score预测器 - 修复扩散模型使用错误

主要修复:
1. 实现完整的逆向扩散采样过程
2. 使用diffuser.reverse()进行去噪
3. 保存去噪后的结构而非随机噪声
4. 保留原有的score输出和TM-score计算功能

参考: experiments/inference_se3_diffusion.py
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
    TMTOOLS_AVAILABLE = True
except ImportError:
    TMTOOLS_AVAILABLE = False
    print("Warning: tmtools not available, TM-score calculation will be skipped")

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 输入参数
PDB_PATH = str(PROJECT_ROOT / 'test' / 'pdb_dir' / '1AKE.pdb')
CHAIN_ID = 'B'
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'output_dir_fixed')
WEIGHTS_PATH = str(PROJECT_ROOT / 'weights' / 'best_weights.pth')
CONF_PATH = str(PROJECT_ROOT / 'config' / 'base.yaml')

# 采样参数
NUM_SAMPLES = 10           # 生成的样本数量
NUM_DIFFUSION_STEPS = 500  # 逆向扩散步数
MIN_T = 0.01              # 最小时间步
NOISE_SCALE = 0.1         # 噪声缩放因子
START_T_RANGE = (0.1, 0.35)  # 初始时间步范围（越小表示加入的噪声越少）
ENABLE_SELF_CONDITIONING = True  # 是否在采样时应用自条件
USE_FORWARD_MARGINAL_INIT = True  # 是否通过前向扩散得到初始状态


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
            raise ValueError(f"Unexpected rigids_t tensor shape: {rigids_t.shape}")
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
    允许通过可控的起始时间步和自条件策略，从较低噪声状态逐步复原蛋白质骨架。
    """
    sample_feats = copy.deepcopy(init_feats)
    sample_feats = {
        k: v.clone().to(device) if torch.is_tensor(v) else v
        for k, v in sample_feats.items()
    }

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

    with torch.no_grad():
        if embed_self_conditioning and reverse_steps.size > 0:
            set_t_feats(sample_feats, reverse_steps[0])
            sc_out = model(sample_feats)
            sample_feats['sc_ca_t'] = sc_out['rigids'][..., 4:].detach()

        for step_idx, t in enumerate(tqdm(reverse_steps, desc="逆向扩散去噪")):
            set_t_feats(sample_feats, t)
            model_out = model(sample_feats)
            rot_score = model_out['rot_score']
            trans_score = model_out['trans_score']

            all_rot_scores.append({'t': float(t), 'score': du.move_to_np(rot_score)})
            all_trans_scores.append({'t': float(t), 'score': du.move_to_np(trans_score)})

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
            else:
                rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])

            sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
            if embed_self_conditioning:
                sample_feats['sc_ca_t'] = model_out['rigids'][..., 4:].detach()

    return {
        'final_rigids': rigids_t,
        'all_rot_scores': all_rot_scores,
        'all_trans_scores': all_trans_scores,
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

    for sample_idx in range(num_samples):
        print(f"\n生成样本 {sample_idx + 1}/{num_samples}")

        if USE_FORWARD_MARGINAL_INIT:
            start_t = float(np.random.uniform(*START_T_RANGE)) if START_T_RANGE else 1.0
            forward_out = diffuser.forward_marginal(
                rigids_0=rigids_0,
                t=start_t,
                diffuse_mask=diffuse_mask_np,
                as_tensor_7=True,
            )
            rigids_t_tensor = torch.tensor(forward_out['rigids_t'], dtype=torch.float32, device=device)
            effective_start_t = start_t
        else:
            ref_sample = diffuser.sample_ref(
                n_samples=num_res,
                as_tensor_7=True,
            )
            rigids_t_tensor = torch.tensor(ref_sample['rigids_t'], dtype=torch.float32, device=device)
            effective_start_t = 1.0

        print(f"  起始时间步 t0 = {effective_start_t:.4f}")

        init_feats = {
            'res_mask': res_mask_tensor.unsqueeze(0),
            'seq_idx': seq_idx_tensor.unsqueeze(0),
            'fixed_mask': fixed_mask_tensor.unsqueeze(0),
            'torsion_angles_sin_cos': torsion_tensor.unsqueeze(0),
            'sc_ca_t': sc_ca_tensor.unsqueeze(0).clone(),
            'rigids_t': rigids_t_tensor.unsqueeze(0),
        }

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

            result = {
                'sample_idx': sample_idx,
                'pdb_path': pdb_path,
                'tm_score': tm_score,
                'rot_scores': sample_out['all_rot_scores'],
                'trans_scores': sample_out['all_trans_scores'],
                'start_t': effective_start_t,
            }
            all_results.append(result)

        except Exception as e:
            print(f"  生成失败: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def save_results(all_results, output_dir, pdb_name):
    """保存结果摘要"""
    summary_path = os.path.join(output_dir, f'{pdb_name}_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("样本生成结果汇总（修复后）\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总样本数: {len(all_results)}\n")
        f.write(f"逆向扩散步数: {NUM_DIFFUSION_STEPS}\n")
        f.write(f"最小时间步: {MIN_T}\n\n")

        tm_scores = [r['tm_score'] for r in all_results if r['tm_score'] is not None]
        if tm_scores:
            f.write("TM-score统计:\n")
            f.write(f"  平均值: {np.mean(tm_scores):.4f}\n")
            f.write(f"  标准差: {np.std(tm_scores):.4f}\n")
            f.write(f"  最小值: {np.min(tm_scores):.4f}\n")
            f.write(f"  最大值: {np.max(tm_scores):.4f}\n\n")

        start_ts = [r.get('start_t') for r in all_results if r.get('start_t') is not None]
        if start_ts:
            f.write("起始时间步统计:\n")
            f.write(f"  平均值: {np.mean(start_ts):.4f}\n")
            f.write(f"  最小值: {np.min(start_ts):.4f}\n")
            f.write(f"  最大值: {np.max(start_ts):.4f}\n\n")

        f.write("各样本详情:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            start_t = result.get('start_t')
            tm_text = f"TM-score = {result['tm_score']:.4f}" if result['tm_score'] else "TM-score = N/A"
            if start_t is not None:
                f.write(f"样本 {result['sample_idx']:03d}: t0 = {start_t:.4f}, {tm_text}\n")
            else:
                f.write(f"样本 {result['sample_idx']:03d}: {tm_text}\n")

    print(f"\n结果已保存: {summary_path}")




def main():
    print("=" * 80)
    print("重构后的Score预测器 - 修复扩散模型使用错误")
    print("=" * 80)
    
    # 加载配置
    conf = OmegaConf.load(CONF_PATH)
    
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
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
        torsion_angles_fixed = np.zeros((num_res, 7, 2), dtype=np.float32)
        for i, ta in enumerate(torsion_angles):
            if isinstance(ta, np.ndarray):
                torsion_angles_fixed[i] = ta
        torsion_angles = torsion_angles_fixed
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

    # 生成样本
    pdb_name = os.path.splitext(os.path.basename(PDB_PATH))[0]
    print(f"\n开始生成 {NUM_SAMPLES} 个样本...")
    print(f"逆向扩散步数: {NUM_DIFFUSION_STEPS}")
    
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
    print("完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
