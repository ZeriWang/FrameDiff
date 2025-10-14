"""Conditioned sampling entry point for FrameDiff.

Example usage:
    python experiments/inference_se3_conditioned.py \
        inference.condition.pdb_path=/home/wangzeli/Lab/FrameDiff/test/pdb_dir/split_output/1AKE_B.pdb \
        inference.condition.chain_id=B \
        inference.condition.pad_to_length=200
"""

import os
import time
import logging
import shutil
from typing import Iterable, List, Optional, Tuple

import hydra
import numpy as np
import torch
import tree
from omegaconf import DictConfig, ListConfig
from openfold.utils import rigid_utils as ru

from data import utils as du
from experiments.inference_se3_diffusion import Sampler, process_chain


def _to_python_list(value: Optional[Iterable]) -> Optional[List]:
    if value is None:
        return None
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _parse_range_spec(range_spec: Optional[Iterable]) -> List[Tuple[int, int]]:
    parsed: List[Tuple[int, int]] = []
    raw_items = _to_python_list(range_spec)
    if not raw_items:
        return parsed
    for item in raw_items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            text = str(item).strip()
            if not text:
                continue
            if text.isdigit():
                start = end = int(text)
            else:
                if '-' in text:
                    delim = '-'
                elif ':' in text:
                    delim = ':'
                else:
                    raise ValueError(f'Invalid range specification "{text}".')
                start_str, end_str = text.split(delim, 1)
                start, end = int(start_str), int(end_str)
        if start <= 0 or end <= 0:
            raise ValueError('Ranges must use 1-based positive indices.')
        if end < start:
            raise ValueError('Range end must be >= start.')
        parsed.append((start, end))
    return parsed


class ConditionedSampler(Sampler):

    def __init__(self, conf: DictConfig, conf_overrides: Optional[dict] = None):
        super().__init__(conf, conf_overrides)
        if 'condition' not in self._infer_conf:
            raise ValueError('Please specify inference.condition in the config.')
        self._condition_conf = self._infer_conf.condition
        if self._condition_conf.pdb_path is None:
            raise ValueError('inference.condition.pdb_path must be provided.')
        self._condition_state = self._prepare_condition_state()

    def _prepare_condition_state(self) -> dict:
        pdb_path = os.path.abspath(self._condition_conf.pdb_path)
        chain_id = self._condition_conf.get('chain_id', 'A')
        freeze_motif = bool(self._condition_conf.get('freeze_motif', True))
        pad_to_length = self._condition_conf.get('pad_to_length', None)
        fixed_ranges = _parse_range_spec(self._condition_conf.get('fixed_ranges'))
        diffuse_ranges = _parse_range_spec(self._condition_conf.get('diffuse_ranges'))

        design_feats = du.parse_pdb_feats('condition', pdb_path, chain_id=chain_id)
        chain_feats = process_chain(design_feats)

        rigid_tensor = chain_feats['rigidgroups_gt_frames'].to(torch.float32)
        rigids_motif = ru.Rigid.from_tensor_4x4(rigid_tensor)[:, 0]
        torsion_motif = chain_feats['torsion_angles_sin_cos'].to(torch.float32)
        res_mask_motif = torch.tensor(chain_feats['res_mask'], dtype=torch.float32)
        motif_len = int(res_mask_motif.shape[0])

        if pad_to_length is not None:
            total_len = int(pad_to_length)
            if total_len < motif_len:
                raise ValueError('pad_to_length must be >= motif length.')
        else:
            total_len = motif_len

        pad_amt = total_len - motif_len
        if pad_amt > 0:
            rigids_pad = ru.Rigid.identity(
                (pad_amt,),
                dtype=rigids_motif.get_trans().dtype,
                device=rigids_motif.get_trans().device,
                requires_grad=False,
            )
            rigids_impute = ru.Rigid.cat([rigids_motif, rigids_pad], dim=0)
            torsion_pad = torch.zeros(
                (pad_amt, torsion_motif.shape[1], torsion_motif.shape[2]),
                dtype=torsion_motif.dtype,
            )
            torsion_impute = torch.cat([torsion_motif, torsion_pad], dim=0)
            res_mask_pad = torch.ones(pad_amt, dtype=res_mask_motif.dtype)
            res_mask = torch.cat([res_mask_motif, res_mask_pad], dim=0)
            motif_mask = torch.cat(
                [res_mask_motif, torch.zeros(pad_amt, dtype=res_mask_motif.dtype)],
                dim=0,
            )
        else:
            rigids_impute = rigids_motif
            torsion_impute = torsion_motif
            res_mask = res_mask_motif
            motif_mask = res_mask_motif.clone()

        seq_idx = torch.arange(1, total_len + 1)
        fixed_mask = torch.zeros(total_len, dtype=torch.float32)
        if freeze_motif:
            fixed_mask[:motif_len] = res_mask[:motif_len]

        for start, end in fixed_ranges:
            if end > total_len:
                raise ValueError('fixed range exceeds sequence length.')
            fixed_mask[start - 1:end] = 1.0
        for start, end in diffuse_ranges:
            if end > total_len:
                raise ValueError('diffuse range exceeds sequence length.')
            fixed_mask[start - 1:end] = 0.0

        fixed_mask = fixed_mask * res_mask
        diffuse_mask = (1.0 - fixed_mask) * res_mask

        diffuse_mask_np = diffuse_mask.cpu().numpy().astype(np.float32)
        motif_mask_np = motif_mask.cpu().numpy().astype(bool)

        sc_ca_zero = torch.zeros_like(rigids_impute.get_trans())
        self._log.info(
            'Conditioned sampling: motif_len=%d, total_len=%d, diffused=%d',
            motif_len,
            total_len,
            int(diffuse_mask.sum().item()),
        )

        return {
            'pdb_path': pdb_path,
            'chain_id': chain_id,
            'motif_len': motif_len,
            'total_len': total_len,
            'rigids_impute': rigids_impute,
            'torsion_impute': torsion_impute,
            'res_mask': res_mask,
            'fixed_mask': fixed_mask,
            'diffuse_mask': diffuse_mask,
            'diffuse_mask_np': diffuse_mask_np,
            'motif_mask_np': motif_mask_np,
            'seq_idx': seq_idx,
            'sc_ca_zero': sc_ca_zero,
        }

    def _build_init_feats(self):
        state = self._condition_state
        ref_sample = self.diffuser.sample_ref(
            n_samples=state['total_len'],
            impute=state['rigids_impute'],
            diffuse_mask=state['diffuse_mask_np'],
            as_tensor_7=True,
        )
        init_feats = {
            'res_mask': state['res_mask'],
            'seq_idx': state['seq_idx'],
            'fixed_mask': state['fixed_mask'],
            'torsion_angles_sin_cos': state['torsion_impute'],
            'sc_ca_t': state['sc_ca_zero'],
            **ref_sample,
        }
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x),
            init_feats,
        )
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device),
            init_feats,
        )
        return init_feats

    def _run_single_sample(self):
        init_feats = self._build_init_feats()
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t,
            aux_traj=True,
            noise_scale=self._diff_conf.noise_scale,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)

    def run_sampling(self):
        sample_root = os.path.join(self._output_dir, 'conditioned')
        os.makedirs(sample_root, exist_ok=True)
        shutil.copy2(
            self._condition_state['pdb_path'],
            os.path.join(sample_root, os.path.basename(self._condition_state['pdb_path'])),
        )
        for sample_idx in range(self._sample_conf.samples_per_length):
            sample_dir = os.path.join(sample_root, f'sample_{sample_idx:03d}')
            if os.path.isdir(sample_dir):
                self._log.info('Skipping existing sample at %s', sample_dir)
                continue
            os.makedirs(sample_dir, exist_ok=True)
            sample_output = self._run_single_sample()
            traj_paths = self.save_traj(
                sample_output['prot_traj'],
                sample_output['rigid_0_traj'],
                self._condition_state['diffuse_mask_np'],
                output_dir=sample_dir,
            )
            sc_dir = os.path.join(sample_dir, 'self_consistency')
            os.makedirs(sc_dir, exist_ok=True)
            shutil.copy2(
                traj_paths['sample_path'],
                os.path.join(sc_dir, os.path.basename(traj_paths['sample_path'])),
            )
            _ = self.run_self_consistency(
                sc_dir,
                traj_paths['sample_path'],
                motif_mask=self._condition_state['motif_mask_np'],
            )
            self._log.info('Finished conditioned sample %d at %s', sample_idx, sample_dir)


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:
    print('Starting conditioned inference')
    start_time = time.time()
    sampler = ConditionedSampler(conf)
    sampler.run_sampling()
    elapsed = time.time() - start_time
    print(f'Finished in {elapsed:.2f}s')


if __name__ == '__main__':
    run()
