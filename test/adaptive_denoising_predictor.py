#!/usr/bin/env python3
"""
自适应去噪版Score预测器

功能:
1. 固定时间步范围(MIN_T, MAX_T),取消去噪步数限制
2. 根据旋转分数和平移分数大小自适应停止去噪
3. 当score足够小时保存去噪后结构并计算TM-score
4. 输出最后一步的旋转分数和平移分数为.npy格式

基于: direct_denoising_predictor.py
"""
import os
import sys
import torch
import numpy as np
import traceback
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
PDB_PATH = str(PROJECT_ROOT / 'test' / 'pdb_dir' / '4AKE.pdb')
CHAIN_ID = 'B'
OUTPUT_DIR = str(PROJECT_ROOT / 'test' / 'output_dir_adaptive_denoising')
WEIGHTS_PATH = str(PROJECT_ROOT / 'weights' / 'best_weights.pth')
CONF_PATH = str(PROJECT_ROOT / 'config' / 'base.yaml')

# 自适应去噪参数
MIN_T = 0.01                 # 固定最小时间步
MAX_T = 0.05                 # 固定最大时间步
NOISE_SCALE = 0.01            # 噪声缩放因子
ENABLE_SELF_CONDITIONING = True  # 启用自条件
SAVE_SCORES = True           # 保存最终分数

# 停止条件阈值
ROT_SCORE_THRESHOLD = 0.01   # 旋转分数阈值
TRANS_SCORE_THRESHOLD = 0.01 # 平移分数阈值
MAX_ITERATIONS = 10000       # 最大迭代次数(防止无限循环)
CHECK_INTERVAL = 10          # 每多少步检查一次停止条件


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
            for model in structure:
                for chain in model:
                    if target_chain_id is not None and chain.id != target_chain_id:
                        continue
                    for residue in chain:
                        if residue.id[0] == ' ':
                            try:
                                ca = residue['CA']
                                coords.append(ca.coord)
                                seq.append(residue.resname)
                            except KeyError:
                                continue
                    if target_chain_id is not None and chain.id == target_chain_id:
                        break
                break
            return np.array(coords), seq

        # 提取坐标和序列
        coords1, seq1 = get_coords_and_seq(structure1, chain_id1)
        coords2, seq2 = get_coords_and_seq(structure2, chain_id2)
        
        if len(coords1) == 0 or len(coords2) == 0:
            print(f"警告: 无法从PDB中提取坐标")
            return None

        from data.residue_constants import restype_3to1
        seq1_str = ''.join([restype_3to1.get(res, 'X') for res in seq1])
        seq2_str = ''.join([restype_3to1.get(res, 'X') for res in seq2])
        
        result = tmtools.tm_align(coords1, coords2, seq1_str, seq2_str)
        return result.tm_norm_chain1
        
    except Exception as e:
        print(f"计算TM-score失败: {e}")
        traceback.print_exc()
        return None


def move_to_device(obj, device):
    """改进的设备转移函数"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, ru.Rigid):
        tensor_7 = obj.to_tensor_7()
        if torch.is_tensor(tensor_7):
            return ru.Rigid.from_tensor_7(tensor_7.to(device))
        return obj
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    else:
        return obj


def adaptive_denoising(
        model,
        diffuser,
        original_rigids,
        res_mask,
        seq_idx,
        fixed_mask,
        torsion_angles,
        sc_ca,
        min_t=0.001,
        max_t=0.05,
        device='cuda',
        noise_scale=0.1,
        enable_self_conditioning=True,
        rot_threshold=0.01,
        trans_threshold=0.01,
        max_iterations=10000,
        check_interval=10,
    ):
    """
    自适应去噪过程：固定时间步范围，根据score大小自动停止
    
    Args:
        model: 模型
        diffuser: 扩散器
        original_rigids: 原始刚体变换
        res_mask: 残基mask
        seq_idx: 序列索引
        fixed_mask: 固定mask
        torsion_angles: 扭转角
        sc_ca: 侧链Ca坐标
        min_t: 最小时间步
        max_t: 最大时间步
        device: 设备
        noise_scale: 噪声缩放因子
        enable_self_conditioning: 是否启用自条件
        rot_threshold: 旋转分数停止阈值
        trans_threshold: 平移分数停止阈值
        max_iterations: 最大迭代次数
        check_interval: 检查停止条件的间隔
    
    返回:
        dict: 包含最终结构、score历史、迭代次数等信息
    """
    print(f"开始自适应去噪过程...")
    print(f"  时间步范围: {min_t} <-> {max_t} (固定)")
    print(f"  噪声缩放: {noise_scale}")
    print(f"  停止阈值: rot={rot_threshold}, trans={trans_threshold}")
    print(f"  最大迭代次数: {max_iterations}")
    print(f"  检查间隔: 每{check_interval}步")
    
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
    
    # 计算固定的dt
    dt = (max_t - min_t) / 2.0  # 使用固定的步长
    
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
    
    # 当前时间步(从max_t开始向min_t移动)
    current_t = max_t
    rigids_t = ru.Rigid.from_tensor_7(sample_feats['rigids_t'][0])
    
    converged = False
    iteration = 0

    with torch.no_grad():
        # 自条件初始化
        if embed_self_conditioning:
            temp_feats = {k: v.clone() if torch.is_tensor(v) else v for k, v in sample_feats.items()}
            temp_feats = set_t_feats(temp_feats, current_t)
            temp_feats['sc_aa_t'] = None
            model_out = model(temp_feats)
            sample_feats['sc_aa_t'] = model_out['rigids'][..., :5, :].detach()

        # 自适应去噪循环
        pbar = tqdm(desc="自适应去噪中", total=max_iterations)
        
        while iteration < max_iterations:
            # 在min_t和max_t之间振荡
            if iteration % 2 == 0:
                t = max_t - (iteration // 2) * dt
                if t < min_t:
                    t = min_t
            else:
                t = min_t + ((iteration - 1) // 2) * dt
                if t > max_t:
                    t = max_t
            
            # 准备当前步特征
            sample_feats['rigids_t'] = rigids_t.to_tensor_7().unsqueeze(0).to(device)
            sample_feats = set_t_feats(sample_feats, t)
            
            # 模型预测
            model_out = model(sample_feats)
            rot_score = model_out['rot_score'].detach()
            trans_score = model_out['trans_score'].detach()
            rigid_pred = model_out['rigids']
            
            # 更新自条件
            if embed_self_conditioning:
                sample_feats['sc_aa_t'] = rigid_pred[..., :5, :].detach()
            
            # 计算score的范数
            rot_score_norm = torch.norm(rot_score, dim=-1).mean().item()
            trans_score_norm = torch.norm(trans_score, dim=-1).mean().item()
            
            all_rot_scores.append(rot_score_norm)
            all_trans_scores.append(trans_score_norm)
            
            # 去噪步骤
            # 提取当前刚体
            if isinstance(rigids_t, ru.Rigid):
                rigids_tensor = rigids_t.to_tensor_7()
            else:
                rigids_tensor = rigids_t
            
            if rigids_tensor.ndim == 2:
                rigids_tensor = rigids_tensor.unsqueeze(0)
            
            rigids_batch = ru.Rigid.from_tensor_7(rigids_tensor.to(device))
            
            # 应用score进行去噪
            dt_step = abs(dt) * noise_scale
            
            # 旋转更新
            scaled_rot_score = rot_score * diffuser.score_scaling(t)[0] * dt_step
            perturb_rot_vec = scaled_rot_score.squeeze(0).cpu().numpy()
            
            # 平移更新
            scaled_trans_score = trans_score * diffuser.score_scaling(t)[1] * dt_step
            perturb_trans = scaled_trans_score.squeeze(0).cpu().numpy()
            
            # 应用扰动
            curr_rots = rigids_batch.get_rots().get_rot_mats()[0].cpu().numpy()
            curr_trans = rigids_batch.get_trans()[0].cpu().numpy()
            
            # 更新刚体
            from scipy.spatial.transform import Rotation as R
            new_rots = []
            for i in range(len(curr_rots)):
                if diffuse_mask[0, i]:
                    rot_update = R.from_rotvec(perturb_rot_vec[i]).as_matrix()
                    new_rot = rot_update @ curr_rots[i]
                    new_rots.append(new_rot)
                else:
                    new_rots.append(curr_rots[i])
            
            new_trans = curr_trans + perturb_trans * diffuse_mask[0][:, None]
            
            # 构建新的刚体
            new_rots = torch.tensor(np.stack(new_rots), dtype=torch.float32)
            new_trans = torch.tensor(new_trans, dtype=torch.float32)
            rigids_t = ru.Rigid(
                rots=ru.Rotation(rot_mats=new_rots),
                trans=new_trans
            )
            
            # 每CHECK_INTERVAL步检查一次停止条件
            if (iteration + 1) % check_interval == 0:
                if rot_score_norm < rot_threshold and trans_score_norm < trans_threshold:
                    print(f"\n收敛！第{iteration+1}步: rot_score={rot_score_norm:.6f}, trans_score={trans_score_norm:.6f}")
                    converged = True
                    final_rot_score = rot_score.squeeze(0).detach().cpu().numpy()
                    final_trans_score = trans_score.squeeze(0).detach().cpu().numpy()
                    break
                else:
                    pbar.set_postfix({
                        'rot': f'{rot_score_norm:.6f}',
                        'trans': f'{trans_score_norm:.6f}',
                        't': f'{t:.4f}'
                    })
            
            iteration += 1
            pbar.update(1)
        
        pbar.close()
        
        # 如果未收敛，保存最后一步的score
        if not converged:
            print(f"\n达到最大迭代次数({max_iterations})，未完全收敛")
            print(f"最终: rot_score={rot_score_norm:.6f}, trans_score={trans_score_norm:.6f}")
            final_rot_score = rot_score.squeeze(0).detach().cpu().numpy()
            final_trans_score = trans_score.squeeze(0).detach().cpu().numpy()

    print(f"去噪完成！总迭代次数: {iteration}")
    
    return {
        'final_rigids': rigids_t,
        'all_rot_scores': all_rot_scores,
        'all_trans_scores': all_trans_scores,
        'final_rot_score': final_rot_score,
        'final_trans_score': final_trans_score,
        'fixed_mask': fixed_mask_np,
        'diffuse_mask': diffuse_mask,
        'converged': converged,
        'num_iterations': iteration,
    }


def main():
    print("=" * 80)
    print("自适应去噪版Score预测器")
    print("=" * 80)
    print("功能: 固定时间步范围，根据score大小自适应停止去噪")
    
    # 加载配置
    conf = OmegaConf.load(CONF_PATH)
    
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"使用设备: {device}")
    else:
        device = torch.device('cpu')
        print(f"使用设备: CPU")
    
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
        torsion_list = []
        for arr in torsion_angles:
            torsion_list.append(arr if isinstance(arr, np.ndarray) else np.array(arr))
        torsion_angles = np.stack(torsion_list, axis=0).astype(np.float32)
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
    output_prefix = f'{pdb_name}_{CHAIN_ID}'
    
    print(f"\n{'='*80}")
    print("开始自适应去噪过程")
    print(f"{'='*80}")
    print(f"去噪参数:")
    print(f"  时间步范围: {MIN_T} <-> {MAX_T} (固定)")
    print(f"  噪声缩放: {NOISE_SCALE}")
    print(f"  自条件: {ENABLE_SELF_CONDITIONING}")
    print(f"  停止阈值: rot={ROT_SCORE_THRESHOLD}, trans={TRANS_SCORE_THRESHOLD}")
    print(f"  最大迭代: {MAX_ITERATIONS}")
    
    # 执行自适应去噪
    denoising_result = adaptive_denoising(
        model=model,
        diffuser=diffuser,
        original_rigids=rigids_0,
        res_mask=res_mask_tensor,
        seq_idx=seq_idx_tensor,
        fixed_mask=fixed_mask_tensor,
        torsion_angles=torsion_tensor,
        sc_ca=sc_ca_tensor,
        min_t=MIN_T,
        max_t=MAX_T,
        device=device,
        noise_scale=NOISE_SCALE,
        enable_self_conditioning=ENABLE_SELF_CONDITIONING,
        rot_threshold=ROT_SCORE_THRESHOLD,
        trans_threshold=TRANS_SCORE_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
        check_interval=CHECK_INTERVAL,
    )
    
    # 转换为PDB并保存
    print("\n转换去噪结果为PDB格式...")
    final_rigids = denoising_result['final_rigids']
    try:
        prot = rigids_to_protein(final_rigids, aatype, residue_index)
        output_pdb = os.path.join(OUTPUT_DIR, f'{output_prefix}_denoised.pdb')
        save_protein_to_pdb(prot, output_pdb)
        print(f"✓ 去噪后结构已保存: {output_pdb}")
        
        # 保存最终分数
        if SAVE_SCORES and denoising_result['final_rot_score'] is not None:
            rot_score_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_final_rot_score.npy')
            trans_score_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_final_trans_score.npy')
            np.save(rot_score_path, denoising_result['final_rot_score'])
            np.save(trans_score_path, denoising_result['final_trans_score'])
            print(f"✓ 旋转分数已保存: {rot_score_path}")
            print(f"✓ 平移分数已保存: {trans_score_path}")
        
        # 计算TM-score
        print("\n计算TM-score...")
        tm_score = calculate_tm_score(PDB_PATH, output_pdb, chain_id1=CHAIN_ID, chain_id2=None)
        if tm_score is not None:
            print(f"✓ TM-score: {tm_score:.4f}")
        else:
            print("✗ TM-score计算失败")
            tm_score = 0.0
        
        # 保存摘要信息
        summary_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_denoising_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("自适应去噪摘要\n")
            f.write("="*80 + "\n\n")
            f.write(f"输入PDB: {PDB_PATH}\n")
            f.write(f"链ID: {CHAIN_ID}\n")
            f.write(f"残基数: {num_res}\n\n")
            f.write(f"去噪参数:\n")
            f.write(f"  时间步范围: {MIN_T} <-> {MAX_T}\n")
            f.write(f"  噪声缩放: {NOISE_SCALE}\n")
            f.write(f"  自条件: {ENABLE_SELF_CONDITIONING}\n")
            f.write(f"  旋转阈值: {ROT_SCORE_THRESHOLD}\n")
            f.write(f"  平移阈值: {TRANS_SCORE_THRESHOLD}\n")
            f.write(f"  最大迭代: {MAX_ITERATIONS}\n\n")
            f.write(f"结果:\n")
            f.write(f"  收敛状态: {'已收敛' if denoising_result['converged'] else '未完全收敛'}\n")
            f.write(f"  实际迭代次数: {denoising_result['num_iterations']}\n")
            f.write(f"  最终旋转分数范数: {denoising_result['all_rot_scores'][-1]:.6f}\n")
            f.write(f"  最终平移分数范数: {denoising_result['all_trans_scores'][-1]:.6f}\n")
            f.write(f"  TM-score: {tm_score:.4f}\n\n")
            f.write(f"输出文件:\n")
            f.write(f"  去噪PDB: {output_pdb}\n")
            f.write(f"  旋转分数: {rot_score_path}\n")
            f.write(f"  平移分数: {trans_score_path}\n")
        print(f"✓ 摘要已保存: {summary_path}")
        
    except Exception as e:
        print(f"✗ 保存结果时出错: {e}")
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("自适应去噪完成！")
    print(f"{'='*80}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 去噪后PDB: {os.path.basename(output_pdb)}")
    print(f"📊 TM-score: {tm_score:.4f}")
    print(f"🔄 迭代次数: {denoising_result['num_iterations']}")
    print(f"✓ 收敛状态: {'已收敛' if denoising_result['converged'] else '未完全收敛'}")
    print(f"💾 分数文件: {output_prefix}_final_rot_score.npy, {output_prefix}_final_trans_score.npy")
    print(f"📋 摘要文件: {output_prefix}_denoising_summary.txt")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
