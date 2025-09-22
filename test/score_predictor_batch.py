import os
import torch
import numpy as np
import tree
from tqdm import tqdm
from omegaconf import OmegaConf
from data import utils as du
from data import se3_diffuser
from model import score_network
from openfold.data import data_transforms
from openfold.utils import rigid_utils as ru

# 输入参数
PDB_PATH = '/home/zeriwang/lab/FrameDiff/test/pdb_dir/1AKE.pdb'
CHAIN_ID = 'A'  # 蛋白质链ID
OUTPUT_DIR = '/home/zeriwang/lab/FrameDiff/test/output_dir_batch'
WEIGHTS_PATH = '/home/zeriwang/lab/FrameDiff/weights/best_weights.pth'
CONF_PATH = '/home/zeriwang/lab/FrameDiff/config/base.yaml'

# 批处理参数
TIME_RANGE = (0.01, 0.99)  # 时间步范围
NUM_SAMPLES = 500          # 总样本数量
BATCH_SIZE = 20            # 批处理大小
NUM_TIME_STEPS = 25        # 时间步数量

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

def create_batch_inputs(base_feats, time_steps, diffuser, device, batch_size):
    """
    创建批处理输入
    
    Args:
        base_feats: 基础特征字典
        time_steps: 时间步列表
        diffuser: SE3扩散器
        device: 计算设备
        batch_size: 批处理大小
    
    Returns:
        批处理输入字典列表
    """
    batch_inputs = []
    num_res = base_feats['num_res']
    
    # 将时间步分批处理
    for i in range(0, len(time_steps), batch_size):
        batch_time_steps = time_steps[i:i+batch_size]
        current_batch_size = len(batch_time_steps)
        
        # 为每个时间步生成参考结构样本和score缩放
        batch_ref_samples = []
        batch_scalings = []
        
        for t in batch_time_steps:
            # 采样参考结构
            ref_sample = diffuser.sample_ref(
                n_samples=num_res,
                as_tensor_7=True,
            )
            batch_ref_samples.append(ref_sample)
            
            # 计算score缩放
            rot_score_scaling, trans_score_scaling = diffuser.score_scaling(t)
            batch_scalings.append({
                'rot_score_scaling': rot_score_scaling,
                'trans_score_scaling': trans_score_scaling
            })
        
        # 构建批量输入特征
        batch_input_feats = {}
        
        # 复制基础特征到批维度
        for key in ['res_mask', 'seq_idx', 'fixed_mask', 'torsion_angles_sin_cos', 'sc_ca_t']:
            if key in base_feats:
                feat = base_feats[key]
                # 扩展到批维度 [batch_size, ...]
                batch_input_feats[key] = np.tile(feat[None], (current_batch_size, *([1] * feat.ndim)))
        
        # 处理缩放因子
        rot_scaling_batch = []
        trans_scaling_batch = []
        
        for scaling in batch_scalings:
            rot_scaling_batch.append(scaling['rot_score_scaling'])
            trans_scaling_batch.append(scaling['trans_score_scaling'])
        
        # 现在直接将参考样本字典合并到批输入中
        # 因为原始代码使用 **ref_sample 来展开参考样本字典
        # 我们需要将每个ref_sample的所有key合并到批输入中
        
        # 收集所有ref_sample的key
        all_ref_keys = set()
        for ref_sample in batch_ref_samples:
            if isinstance(ref_sample, dict):
                all_ref_keys.update(ref_sample.keys())
        
        # 为每个ref key创建批量数据
        for ref_key in all_ref_keys:
            batch_ref_values = []
            for ref_sample in batch_ref_samples:
                if isinstance(ref_sample, dict) and ref_key in ref_sample:
                    ref_value = ref_sample[ref_key]
                    if torch.is_tensor(ref_value):
                        batch_ref_values.append(ref_value.cpu().numpy())
                    elif isinstance(ref_value, np.ndarray):
                        batch_ref_values.append(ref_value)
                    else:
                        # 如果不是tensor或numpy数组，尝试转换
                        batch_ref_values.append(np.array(ref_value, dtype=np.float32))
                else:
                    # 如果某个样本缺少这个key，使用默认值
                    if ref_key in ['rigids_t', 'rigids_0']:
                        cpu_device = torch.device('cpu')
                        identity_rigid = ru.Rigid.identity((num_res,), dtype=torch.float32, device=cpu_device)
                        default_value = identity_rigid.to_tensor_7().numpy()
                        batch_ref_values.append(default_value)
                    else:
                        # 其他key使用零值（形状需要从其他样本推断）
                        if batch_ref_values:
                            default_value = np.zeros_like(batch_ref_values[0])
                        else:
                            default_value = np.array(0.0, dtype=np.float32)
                        batch_ref_values.append(default_value)
            
            # 堆叠到批维度
            try:
                batch_input_feats[ref_key] = np.stack(batch_ref_values, axis=0)
            except Exception as e:
                print(f"Error stacking {ref_key}: {e}")
                print(f"shapes: {[v.shape for v in batch_ref_values]}")
                # 如果堆叠失败，使用默认值
                if ref_key in ['rigids_t', 'rigids_0']:
                    cpu_device = torch.device('cpu')
                    identity_rigid = ru.Rigid.identity((num_res,), dtype=torch.float32, device=cpu_device)
                    rigids_tensor = identity_rigid.to_tensor_7().numpy()
                    batch_input_feats[ref_key] = np.tile(rigids_tensor[None], (current_batch_size, 1, 1))
                else:
                    batch_input_feats[ref_key] = np.zeros((current_batch_size,), dtype=np.float32)
        
        batch_input_feats['rot_score_scaling'] = np.array(rot_scaling_batch)
        batch_input_feats['trans_score_scaling'] = np.array(trans_scaling_batch)
        
        # 转换为tensor并移动到设备
        batch_input_tensors = {}
        for key, value in batch_input_feats.items():
            if isinstance(value, np.ndarray):
                # 确保数据类型为数值类型，避免object类型
                if value.dtype == np.object_:
                    print(f"警告: 特征 '{key}' 包含object类型数据，尝试转换...")
                    try:
                        # 对于object数组，尝试递归提取数值数据
                        if key in ['rigids_t', 'rigids_0']:
                            # 特殊处理rigids数据
                            print(f"特殊处理rigids数据，形状: {value.shape}")
                            # 如果是object数组但包含tensor，尝试提取
                            if value.size > 0:
                                first_elem = value.flat[0]
                                if torch.is_tensor(first_elem):
                                    # 如果包含tensor，转换所有元素为numpy
                                    value_list = []
                                    for item in value.flat:
                                        if torch.is_tensor(item):
                                            value_list.append(item.cpu().numpy())
                                        else:
                                            value_list.append(np.array(item, dtype=np.float32))
                                    value = np.array(value_list).reshape(value.shape + value_list[0].shape)
                                else:
                                    value = np.array(value, dtype=np.float32)
                            else:
                                value = np.zeros_like(value, dtype=np.float32)
                        else:
                            # 其他特征的常规处理
                            value = np.array(value, dtype=np.float32)
                    except (ValueError, TypeError) as e:
                        print(f"无法转换特征 '{key}': {e}")
                        # 创建相同形状的零数组作为替代
                        if key in ['rigids_t', 'rigids_0']:
                            # rigids应该是 [batch_size, num_res, 7]
                            batch_size = current_batch_size
                            value = np.zeros((batch_size, num_res, 7), dtype=np.float32)
                        elif len(value.shape) > 0:
                            value = np.zeros_like(value, dtype=np.float32)
                        else:
                            value = np.array(0.0, dtype=np.float32)
                
                # 确保数据类型为float32
                if value.dtype not in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
                    value = value.astype(np.float32)
                
                batch_input_tensors[key] = torch.tensor(value, dtype=torch.float32).to(device)
            else:
                batch_input_tensors[key] = value
        
        # 添加时间步tensor
        batch_input_tensors['t'] = torch.tensor(batch_time_steps, dtype=torch.float32).to(device)
        
        batch_inputs.append((batch_input_tensors, batch_time_steps))
    
    return batch_inputs

def batch_inference(model, batch_inputs, device):
    """
    执行批处理推理
    
    Args:
        model: 预训练模型
        batch_inputs: 批输入列表
        device: 计算设备
    
    Returns:
        所有批次的推理结果
    """
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (batch_input, time_steps) in enumerate(tqdm(batch_inputs, desc="批处理推理")):
            if device.type == 'cuda':
                # 监控内存使用
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                
                memory_before = torch.cuda.memory_allocated(0) / 1024**3
                if memory_before > 5.0:
                    torch.cuda.empty_cache()
                    print(f"批次 {batch_idx}: GPU内存清理前 {memory_before:.2f} GB")
            
            try:
                # 执行推理
                output = model(batch_input)
                
                # 提取结果并立即移动到CPU
                rot_score = output['rot_score'].detach().cpu().numpy()
                trans_score = output['trans_score'].detach().cpu().numpy()
                
                # 删除GPU上的输出tensor
                del output
                
                # 保存每个时间步的结果
                for i, t in enumerate(time_steps):
                    all_results.append({
                        'time_step': t,
                        'rot_score': rot_score[i],
                        'trans_score': trans_score[i]
                    })
                
                # 显示进度
                if device.type == 'cuda' and batch_idx % 5 == 0:
                    memory_after = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"批次 {batch_idx + 1}/{len(batch_inputs)}: GPU内存 {memory_after:.2f} GB")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"批次 {batch_idx} GPU内存不足，跳过...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    return all_results

def save_batch_results(all_samples, output_dir, time_steps, prefix=""):
    """
    保存批处理结果
    
    Args:
        all_samples: 所有样本的结果列表
        output_dir: 输出目录
        time_steps: 时间步数组
        prefix: 文件名前缀（如 "1AKE_A_"）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有有效样本
    if len(all_samples) == 0:
        print("警告: 没有成功的样本，保存空结果...")
        # 创建空的结果文件
        np.save(os.path.join(output_dir, f'{prefix}all_rot_scores.npy'), np.array([]))
        np.save(os.path.join(output_dir, f'{prefix}all_trans_scores.npy'), np.array([]))
        np.save(os.path.join(output_dir, f'{prefix}time_steps.npy'), np.array([]))
        
        # 保存错误报告
        error_file = os.path.join(output_dir, f'{prefix}error_report.txt')
        with open(error_file, 'w') as f:
            f.write("错误报告: 没有成功的样本\n")
            f.write("所有时间步的处理都失败了\n")
            f.write("请检查数据类型和模型配置\n")
        
        print(f"空结果已保存到 {output_dir}")
        return
    
    # 按时间步组织结果
    results_by_time = {}
    for sample in all_samples:
        t = sample['time_step']
        if t not in results_by_time:
            results_by_time[t] = []
        results_by_time[t].append(sample)
    
    print(f"保存结果到 {output_dir}")
    print(f"共 {len(results_by_time)} 个时间步，{len(all_samples)} 个总样本")
    
    # 保存每个时间步的结果
    all_rot_scores = []
    all_trans_scores = []
    time_step_list = []
    
    for t in sorted(results_by_time.keys()):
        samples = results_by_time[t]
        time_step_list.append(t)
        
        # 收集当前时间步的所有score
        rot_scores = np.array([s['rot_score'] for s in samples])
        trans_scores = np.array([s['trans_score'] for s in samples])
        
        all_rot_scores.append(rot_scores)
        all_trans_scores.append(trans_scores)
        
        # 计算统计信息
        rot_mean = np.mean(rot_scores, axis=0)
        rot_std = np.std(rot_scores, axis=0)
        trans_mean = np.mean(trans_scores, axis=0)
        trans_std = np.std(trans_scores, axis=0)
        
        # 保存单个时间步的详细结果
        time_output_dir = os.path.join(output_dir, f'{prefix}time_step_{t:.4f}')
        os.makedirs(time_output_dir, exist_ok=True)
        
        np.save(os.path.join(time_output_dir, f'{prefix}rot_scores.npy'), rot_scores)
        np.save(os.path.join(time_output_dir, f'{prefix}trans_scores.npy'), trans_scores)
        np.save(os.path.join(time_output_dir, f'{prefix}rot_score_mean.npy'), rot_mean)
        np.save(os.path.join(time_output_dir, f'{prefix}rot_score_std.npy'), rot_std)
        np.save(os.path.join(time_output_dir, f'{prefix}trans_score_mean.npy'), trans_mean)
        np.save(os.path.join(time_output_dir, f'{prefix}trans_score_std.npy'), trans_std)
    
    # 保存总体结果汇总
    all_rot_scores = np.array(all_rot_scores)  # shape: (num_time_steps, num_samples_per_time, num_res, 3)
    all_trans_scores = np.array(all_trans_scores)
    
    # 保存完整数据
    np.save(os.path.join(output_dir, f'{prefix}all_rot_scores.npy'), all_rot_scores)
    np.save(os.path.join(output_dir, f'{prefix}all_trans_scores.npy'), all_trans_scores)
    np.save(os.path.join(output_dir, f'{prefix}time_steps.npy'), np.array(time_step_list))
    
    # 计算并保存跨时间步的统计信息
    if all_rot_scores.size > 0 and len(all_rot_scores.shape) > 1:
        time_mean_rot_scores = np.mean(all_rot_scores, axis=1)  # (num_time_steps, num_res, 3)
        time_mean_trans_scores = np.mean(all_trans_scores, axis=1)
        
        np.save(os.path.join(output_dir, f'{prefix}time_evolution_rot_mean.npy'), time_mean_rot_scores)
        np.save(os.path.join(output_dir, f'{prefix}time_evolution_trans_mean.npy'), time_mean_trans_scores)
    else:
        print("警告: 数据不足，跳过时间演化统计")
    
    # 保存汇总统计信息到文本文件
    summary_file = os.path.join(output_dir, f'{prefix}summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"批处理Score预测结果汇总\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"时间步范围: {TIME_RANGE[0]} - {TIME_RANGE[1]}\n")
        f.write(f"时间步数量: {len(time_step_list)}\n")
        f.write(f"总采样数量: {len(all_samples)}\n")
        f.write(f"批处理大小: {BATCH_SIZE}\n")
        f.write(f"每个时间步的平均样本数: {len(all_samples) / len(time_step_list):.1f}\n")
        f.write(f"\n")
        
        f.write("各时间步统计信息:\n")
        f.write("-" * 50 + "\n")
        for i, (t, rot_scores) in enumerate(zip(time_step_list, all_rot_scores)):
            num_samples = rot_scores.shape[0]
            
            # 计算范数，如果time_mean_scores可用的话
            if all_rot_scores.size > 0 and len(all_rot_scores.shape) > 1:
                # 重新计算局部均值（因为time_mean_scores可能未定义）
                local_rot_mean = np.mean(rot_scores, axis=0)
                local_trans_mean = np.mean(all_trans_scores[i], axis=0)
                rot_mean_norm = np.linalg.norm(local_rot_mean, axis=-1).mean()
                trans_mean_norm = np.linalg.norm(local_trans_mean, axis=-1).mean()
                
                f.write(f"时间步 {t:.4f}: {num_samples} 个样本, ")
                f.write(f"旋转score平均范数: {rot_mean_norm:.4f}, ")
                f.write(f"平移score平均范数: {trans_mean_norm:.4f}\n")
            else:
                f.write(f"时间步 {t:.4f}: {num_samples} 个样本\n")
    
    print(f"结果保存完成!")
    print(f"- 总体结果: {prefix}all_rot_scores.npy, {prefix}all_trans_scores.npy")
    print(f"- 时间演化: {prefix}time_evolution_rot_mean.npy, {prefix}time_evolution_trans_mean.npy") 
    print(f"- 详细统计: {prefix}summary.txt")
    print(f"- 各时间步详细结果保存在对应的子目录中，文件名前缀: {prefix}")

def main():
    # 加载配置
    conf = OmegaConf.load(CONF_PATH)
    
    # 设备配置
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        print(f"使用设备: {device}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print(f"使用设备: {device}")

    # 加载PDB并预处理
    print(f"加载PDB文件: {PDB_PATH}")
    pdb_feats = du.parse_pdb_feats('query', PDB_PATH, chain_id=CHAIN_ID)
    
    # 处理链特征
    chain_feats = process_chain_feats(pdb_feats)
    bb_mask = np.array(pdb_feats['bb_mask']).astype(bool)
    num_res = int(np.sum(bb_mask))
    print(f"蛋白质长度: {num_res} 个残基")

    # 构造diffuser和模型
    diffuser = se3_diffuser.SE3Diffuser(conf.diffuser)
    model = score_network.ScoreNetwork(conf.model, diffuser)
    model.to(device)
    model.eval()
    
    # GPU优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("已启用GPU优化设置")

    # 加载权重
    print(f"加载模型权重: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # 生成时间步序列
    min_t, max_t = TIME_RANGE
    time_steps = np.linspace(min_t, max_t, NUM_TIME_STEPS)
    print(f"时间步范围: {min_t} - {max_t}, 共 {NUM_TIME_STEPS} 个时间步")
    print(f"目标总样本数: {NUM_SAMPLES}, 批处理大小: {BATCH_SIZE}")
    
    # 准备基础特征（全部裁剪到有效残基）
    # 确保torsion_angles_sin_cos数据类型正确
    torsion_angles = chain_feats['torsion_angles_sin_cos'].numpy()[bb_mask]
    if torsion_angles.dtype == np.object_:
        print("警告: torsion_angles_sin_cos 包含object类型，尝试转换...")
        try:
            torsion_angles = np.array(torsion_angles, dtype=np.float32)
        except (ValueError, TypeError):
            print("无法转换，使用零数组替代")
            torsion_angles = np.zeros((num_res, 7, 2), dtype=np.float32)
    else:
        torsion_angles = torsion_angles.astype(np.float32)
    
    base_feats = {
        'num_res': num_res,
        'res_mask': np.ones(num_res, dtype=np.float32),
        'seq_idx': np.arange(1, num_res+1, dtype=np.float32),
        'fixed_mask': np.zeros(num_res, dtype=np.float32),
        'torsion_angles_sin_cos': torsion_angles,
        'sc_ca_t': np.zeros((num_res, 3), dtype=np.float32),
    }
    
    # 收集所有结果
    all_samples = []
    
    # 计算每个时间步的样本数
    samples_per_time = NUM_SAMPLES // NUM_TIME_STEPS
    remaining_samples = NUM_SAMPLES % NUM_TIME_STEPS
    
    print("开始批处理采样...")
    print(f"每个时间步将生成约 {samples_per_time} 个样本")
    
    for time_idx, time_step in enumerate(tqdm(time_steps, desc="时间步进度")):
        try:
            # 为当前时间步确定样本数
            current_samples = samples_per_time
            if time_idx < remaining_samples:
                current_samples += 1
            
            # 创建当前时间步的时间步列表
            current_time_steps = [time_step] * current_samples
            
            # 创建批输入
            batch_inputs = create_batch_inputs(
                base_feats, current_time_steps, diffuser, device, BATCH_SIZE
            )
            
            # 执行批推理
            results = batch_inference(model, batch_inputs, device)
            all_samples.extend(results)
            
            # 显示进度
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated(0) / 1024**3
                print(f"时间步 {time_step:.4f} ({time_idx + 1}/{len(time_steps)}) 完成, " +
                      f"样本数: {len(results)}, GPU内存: {current_memory:.2f} GB")
            else:
                print(f"时间步 {time_step:.4f} ({time_idx + 1}/{len(time_steps)}) 完成, " +
                      f"样本数: {len(results)}")
                      
        except Exception as e:
            print(f"时间步 {time_step:.4f} 处理失败: {e}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    print(f"总共完成 {len(all_samples)} 个样本的预测")
    
    # 生成文件名前缀（从PDB路径提取蛋白质名称）
    pdb_name = os.path.splitext(os.path.basename(PDB_PATH))[0]  # 提取文件名（不含扩展名）
    file_prefix = f"{pdb_name}_{CHAIN_ID}_"
    print(f"使用文件前缀: {file_prefix}")
    
    # 保存结果
    save_batch_results(all_samples, OUTPUT_DIR, time_steps, prefix=file_prefix)

if __name__ == '__main__':
    main()