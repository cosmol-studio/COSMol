import copy
import os
import sys
import traceback
from copy import deepcopy

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from torch_geometric.data import Batch

from data.utils import get_confs_from_coords, save_sdf
from models.model import AbstractModel

model = None


max_total_atoms = 16000


def pred_confs(cfg, data, device, num_confs=1, flow_temperature=1.0, mol=None):
    return pred_confs_batch(
        cfg,
        [data],
        device,
        num_confs=num_confs,
        flow_temperature=flow_temperature,
    )[0]


def pred_confs_batch(
    cfg,
    data_items,
    device,
    num_confs=1,
    flow_temperature=1.0,
    atom_budget=None,
):
    """Generate conformers for multiple variable-size ligand graphs.

    Graph replicas from different ligands share model forward passes up to the
    same ``max_total_atoms`` budget used by ``pred_confs``.  Returned tensors
    retain input-ligand order and have shape ``(num_confs, atoms, 3)``.
    """
    budget = max_total_atoms if atom_budget is None else int(atom_budget)
    if budget < 1:
        raise ValueError("atom_budget must be positive")
    global model
    if model is None:
        model = AbstractModel.load_from_checkpoint(
            cfg.paths.checkpoint, cfg=cfg, map_location=device
        )
        model.eval()

    samples = []
    owners = []
    for owner, data in enumerate(data_items):
        for _ in range(num_confs):
            sample = copy.deepcopy(data)
            sample = sample.sample_conformer()
            x0, _ = sample.sample_base_distribution_harmonic()
            sample.pos = x0
            samples.append(sample)
            owners.append(owner)

    outputs = [[] for _ in data_items]
    start = 0
    while start < len(samples):
        stop = start
        atom_total = 0
        while stop < len(samples):
            atom_count = int(samples[stop].pos.shape[0])
            if stop > start and atom_total + atom_count > budget:
                break
            atom_total += atom_count
            stop += 1

        batch = Batch.from_data_list(samples[start:stop])
        batch.to(model.device)
        pointers = batch.ptr.detach().cpu().tolist()
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            coordinates = run_batch(
                batch,
                cfg,
                flow_temperature=flow_temperature,
            )
        for local_index, owner in enumerate(owners[start:stop]):
            begin, end = pointers[local_index : local_index + 2]
            outputs[owner].append(coordinates[begin:end])
        start = stop

    return [torch.stack(values, dim=0) for values in outputs]


def _compute_delta_t(t_schedule: torch.Tensor, t: torch.Tensor):
    if t + 1 >= t_schedule.size(0):
        return 0.0

    t_curr, t_next = t_schedule[t : t + 2]
    return t_next - t_curr


def run_batch(batch, cfg, save_each_frame=False, flow_temperature=1.2, mol=None):
    n_steps = 50
    steps = n_steps + 1
    # t_schedule = torch.cos(torch.linspace(0, math.pi/2, steps)).flip(0)
    t_schedule = torch.linspace(0, 1, steps)
    for i in tqdm(range(t_schedule.size(0) - 1)):
        delta_t = _compute_delta_t(t_schedule, t=i)
        t = t_schedule[i]
        # batch.update_graph(cfg.dataset.radius_cutoff, t)
        batch.update_graph(cfg.dataset.conf.radius_cutoff, t)
        batch.ligand_mask = torch.ones(
            batch.x.shape[0], dtype=torch.long, device=batch.x.device
        )
        with torch.no_grad():
            v_t = model(batch, task="conformation")

        if t < 0.8:
            batch.pos = batch.pos + v_t * (delta_t / (1 - t) * flow_temperature)
        else:
            batch.pos = batch.pos + v_t * 1.0

        if save_each_frame:
            num_mols = batch.batch.max().item() + 1

            for j in range(num_mols):
                # 选出属于第 j 个分子的节点索引
                keep_nodes = (batch.batch == j).nonzero(as_tuple=False).view(-1)

                # 提取子图（节点、边、边属性）
                edge_index_sub, edge_types_sub, edge_mask = subgraph(
                    subset=keep_nodes,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    num_nodes=batch.pos.size(0),
                    relabel_nodes=True,
                    return_edge_mask=True,
                )

                # exit()

                # 对应节点坐标和类型
                pos_sub = batch.pos[keep_nodes]
                node_types_sub = batch.x[keep_nodes, 0]

                # print(node_types_sub)
                # exit()
                # print(edge_index_sub.shape, edge_types_sub.shape)

                # 保存单个分子
                # graph_to_sdf_v3000(
                #     pos_sub,
                #     node_types_sub,
                #     edge_index_sub,
                #     edge_types_sub.flatten(),
                #     sdf_path=f"../../frames/frame_{j:03d}_{i:03d}.sdf",
                # )

            pred_mols = get_confs_from_coords(
                mol, batch.pos.view(batch.batch.max() + 1, -1, 3)
            )

            for idx, mol in enumerate(pred_mols):
                save_sdf(
                    mol,
                    sdf_path=f"../../frames/frame_{idx:03d}_{i:03d}.sdf",
                )

    # exit()

    return batch.pos


def calc_rmsd(mol, ref_mol):
    from data.utils import aligned_rmsd
    try:
        rmsd = aligned_rmsd(mol, ref_mol)
    except:
        traceback.print_exc()
        rmsd = None
    return rmsd


def pairwise_conformer_rmsds(sampled_mols, gt_mols):
    n_confs = len(gt_mols)
    n_samples = len(sampled_mols)
    rmsd_confusion_mat = np.nan * np.ones([n_confs, n_samples], dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_confs):
            mol_ref = deepcopy(gt_mols[j])
            mol_sampled = deepcopy(sampled_mols[i])
            rmsd_confusion_mat[j, i] = calc_rmsd(mol_sampled, mol_ref)

    return rmsd_confusion_mat
