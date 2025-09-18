import os
import torch
import numpy as np
import tree
from omegaconf import OmegaConf
from data import utils as du
from data import se3_diffuser
from model import score_network
from openfold.data import data_transforms
from openfold.utils import rigid_utils as ru

# 输入参数
PDB_PATH = '/home/zeriwang/lab/FrameDiff/test/pdb_dir/4AKE.pdb'
OUTPUT_DIR = '/home/zeriwang/lab/FrameDiff/test/output_dir'
WEIGHTS_PATH = '/home/zeriwang/lab/FrameDiff/weights/best_weights.pth'
CONF_PATH = '/home/zeriwang/lab/FrameDiff/config/base.yaml'

def process_chain_feats(pdb_feats):
    """处理PDB特征，转换为适合模型的格式"""
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

def main():
    # 加载配置
    conf = OmegaConf.load(CONF_PATH)
    
    # 优先使用GPU，并进行详细的设备检查
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # 明确指定第一个GPU
        torch.cuda.set_device(0)  # 设置当前设备
        print(f"使用设备: {device}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前可用GPU内存: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print(f"CUDA不可用，使用设备: {device}")
        print("警告: 使用CPU可能会显著增加推理时间")

    # 加载PDB并预处理
    print(f"加载PDB文件: {PDB_PATH}")
    pdb_feats = du.parse_pdb_feats('query', PDB_PATH, chain_id='A')
    
    # 处理链特征
    chain_feats = process_chain_feats(pdb_feats)
    bb_mask = pdb_feats['bb_mask'].astype(bool)
    num_res = int(np.sum(bb_mask))
    print(f"蛋白质长度: {num_res} 个残基")

    # 构造diffuser
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)

    # 构造模型
    model = score_network.ScoreNetwork(conf.model, diffuser)
    model.to(device)
    model.eval()
    
    # 如果使用GPU，启用一些优化
    if device.type == 'cuda':
        # 启用cudnn基准模式以优化性能
        torch.backends.cudnn.benchmark = True
        # 启用内存高效的注意力机制（如果支持）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("已启用GPU优化设置")

    # 加载权重
    print(f"加载模型权重: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # 去除module前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # 组装输入特征（全部裁剪到有效残基）
    res_mask = np.ones(num_res, dtype=np.float32)
    fixed_mask = np.zeros_like(res_mask)
    seq_idx = np.arange(1, num_res+1)
    torsion_angles_sin_cos = chain_feats['torsion_angles_sin_cos'].numpy()[bb_mask]
    sc_ca_t = np.zeros((num_res, 3))
    
    # 从diffuser采样参考结构
    ref_sample = diffuser.sample_ref(
        n_samples=num_res,
        as_tensor_7=True,
    )
    
    # 设置时间相关特征
    t = 0  # 中间时间步
    rot_score_scaling, trans_score_scaling = diffuser.score_scaling(t)
    
    # 构造输入字典，包含所有必要特征
    input_feats = {
        'res_mask': res_mask,
        'seq_idx': seq_idx,
        'fixed_mask': fixed_mask,
        'torsion_angles_sin_cos': torsion_angles_sin_cos,
        'sc_ca_t': sc_ca_t,
        'rot_score_scaling': np.array([rot_score_scaling]),
        'trans_score_scaling': np.array([trans_score_scaling]),
        **ref_sample,
    }
    
    # 单独处理时间步，不添加batch维度
    t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
    
    # 转换为tensor并添加batch维度
    input_feats_tensors = tree.map_structure(
        lambda x: x if torch.is_tensor(x) else torch.tensor(x), input_feats)
    input_feats_batched = tree.map_structure(
        lambda x: x[None].to(device), input_feats_tensors)
    
    # 重新构造为普通字典并添加时间步
    final_input_feats = {}
    for key, value in input_feats_batched.items():
        final_input_feats[key] = value
    final_input_feats['t'] = t_tensor
    input_feats = final_input_feats

    # 推理
    print("开始预测对数概率密度梯度...")
    
    # 如果使用GPU，显示推理前的内存使用情况
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # 清理缓存
        print(f"推理前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    with torch.no_grad():
        output = model(input_feats)
    
    # 推理后的内存情况
    if device.type == 'cuda':
        print(f"推理后GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存score预测结果
    rot_score = output['rot_score'].cpu().numpy().squeeze()
    trans_score = output['trans_score'].cpu().numpy().squeeze()
    
    np.save(os.path.join(OUTPUT_DIR, 'rot_score.npy'), rot_score)
    np.save(os.path.join(OUTPUT_DIR, 'trans_score.npy'), trans_score)
    
    print(f"预测完成！结果已保存到 {OUTPUT_DIR}")
    print(f"- 旋转score形状: {rot_score.shape}")
    print(f"- 平移score形状: {trans_score.shape}")

if __name__ == '__main__':
	main()
