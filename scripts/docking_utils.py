import math
import os
import signal
import sys
import traceback
from collections.abc import Mapping
from copy import copy

import numpy as np
import torch
from spyrmsd.rmsd import symmrmsd
from torch.nn import functional as F
from torch_geometric.utils import scatter, subgraph

from data.utils import (
    RADIUS_EDGE_TYPE,
    CPIdata,
    graph_to_sdf_v3000,
    keep_fields,
    rotation_fraction,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tempfile
from contextlib import contextmanager
from importlib.metadata import version

from posebusters import PoseBusters
from rdkit import Chem
from torch_geometric.data import Batch

from models.model import AbstractModel
from models.utils import batch_chirality_loss_no_gt

assert version("posebusters") == "0.4.5", (
    f"PoseBusters version mismatch: expected 0.4.5, got {version('posebusters')}"
)
from pathlib import Path

docking_model = None
pocket_model = None
pocket_score_model = None
score_model = None
_model_cache = {}


def _as_device(device):
    return device if isinstance(device, torch.device) else torch.device(device)


def _to_device_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _freeze_config(value):
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_config(item)) for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _model_cache_key(cfg, device, strict):
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return (
        device.type,
        device_index,
        bool(strict),
        _freeze_config(cfg),
    )


def _get_cached_model(cfg, device):
    strict = cfg.paths.get("strict_load", True)
    cache_key = _model_cache_key(cfg, device, strict)
    model = _model_cache.get(cache_key)
    if model is None:
        model = AbstractModel.load_from_checkpoint(
            cfg.paths.checkpoint,
            map_location=device,
            cfg=cfg,
            strict=strict,
        )
        model.eval()
        _model_cache[cache_key] = model
    return model


def _sample_pocket_centers(
    centers: torch.Tensor,
    confidences: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if centers.size(0) == 0:
        raise ValueError("cannot sample from empty pocket center list")

    confidences = confidences.view(-1)
    confidence_mean = confidences.mean()
    weights = (confidences - confidence_mean) / (1 - confidence_mean + 1e-8)
    weights = torch.nan_to_num(weights.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, min=0.0)
    weights = torch.where(weights.sum() > 0, weights, torch.ones_like(weights))
    indices = torch.multinomial(weights, num_samples=num_samples, replacement=True)
    return centers[indices]


def _load_models_once(
    device,
    pocket,
    docking_cfg,
    pocket_cfg,
    pocket_ranker_cfg,
    calculate_score,
    pose_ranker_cfg,
):
    global docking_model
    global pocket_model
    global pocket_score_model
    global score_model

    docking_model = _get_cached_model(docking_cfg, device)

    if pocket == "blind":
        assert pocket_cfg is not None, (
            "pocket_cfg must be provided when pocket is 'blind'"
        )
        assert pocket_ranker_cfg is not None, (
            "pocket_ranker_cfg must be provided when pocket is 'blind'"
        )
        pocket_model = _get_cached_model(pocket_cfg, device)
        pocket_score_model = _get_cached_model(pocket_ranker_cfg, device)

    if calculate_score:
        assert pose_ranker_cfg is not None, (
            "score_cfg must be provided when calculate_score is True"
        )
        score_model = _get_cached_model(pose_ranker_cfg, device)


def pred_confs(
    data_or_datalist,
    device,
    num_confs=20,
    pocket="blind",
    docking_cfg=None,
    pocket_cfg=None,
    pocket_ranker_cfg=None,
    calculate_score=False,
    pose_ranker_cfg=None,
    pocket_pos_list=None,
    precision=torch.float32,
    initial_shift_range=1.0,
    optimized=False,
    # pocket_ranker_sdf_dir="pocket_ranker_sdf.sdf",
):
    device = _as_device(device)
    assert pocket in ["blind", "pocket_center", "ligand_center", "given"], (
        f"pocket must be one of ['blind', 'pocket_center', 'ligand_center', 'given'], got {pocket}"
    )

    input_is_single = isinstance(data_or_datalist, tuple)
    datalist = [data_or_datalist] if input_is_single else data_or_datalist

    _load_models_once(
        device=device,
        pocket=pocket,
        docking_cfg=docking_cfg,
        pocket_cfg=pocket_cfg,
        pocket_ranker_cfg=pocket_ranker_cfg,
        calculate_score=calculate_score,
        pose_ranker_cfg=pose_ranker_cfg,
    )

    autocast_enabled = device.type == "cuda" and precision == torch.float16
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=autocast_enabled,
    ):

        if pocket == "blind":
            backbone_list = [
                data[1]
                .to_backbone_graph()
                .update_graph(
                    pocket_cfg.dataset.pocket.radius_cutoff,
                    pocket_cfg.dataset.pocket.get("radius_min", 0.0),
                    scale=8.0 / pocket_cfg.dataset.pocket.radius_cutoff,
                )
                for data in datalist
            ]
            backbone_batch = Batch.from_data_list(backbone_list).to(device)
            with torch.no_grad():
                vec_to_pocket_all = pocket_model(backbone_batch, task="pocket_pred")

            cb_batch_all = backbone_batch.batch[backbone_batch.CB_mask]
            points_all = backbone_batch.pos[backbone_batch.CB_mask] + vec_to_pocket_all
            points_per_data = []
            for data_idx in range(len(datalist)):
                points_i = points_all[cb_batch_all == data_idx]
                points_per_data.append(points_i)

            ranker_batch = Batch.from_data_list(
                [
                    datalist[data_idx][1].prepare_pocket_ranker_graph(
                        points_per_data[data_idx],
                        pocket_cfg.dataset.pocket_ranker.radius_cutoff,
                        inplace=False,
                        device=device,
                    )
                    for data_idx in range(len(datalist))
                ]
            )

            ranker_batch = keep_fields(
                ranker_batch,
                [
                    "x",
                    "y",
                    "pos",
                    "edge_index",
                    "edge_types",
                    "edge_vector",
                ],
            ).to(device)

            with torch.no_grad():
                score_all = pocket_score_model(ranker_batch, task="pocket_ranker")

        datalist = [
            (ligand.to(device), protein.to(device))
            for ligand, protein in datalist
        ]
        dock_batch = []
        ligand_atom_counts = []
        for data_idx, (ligand_data, protein_data) in enumerate(datalist):

            if pocket == "blind":
                pred_pocket_centers = points_all[cb_batch_all == data_idx]
                pocket_confidences = torch.relu(
                    score_all[cb_batch_all == data_idx]
                ).view(-1)
                sampled_pocket_centers = _sample_pocket_centers(
                    pred_pocket_centers,
                    pocket_confidences,
                    num_confs,
                )
            elif pocket == "pocket_center":
                pocket_center = _to_device_tensor(protein_data.pocket_center, device)
            elif pocket == "ligand_center":
                pocket_center = ligand_data.get_center().to(device, non_blocking=True)
            else:
                pocket_center = _to_device_tensor(pocket_pos_list[data_idx], device)

            for conf_idx in range(num_confs):
                if pocket == "blind":
                    pocket_center = sampled_pocket_centers[conf_idx]

                mol_x0 = ligand_data.sample_base_distribution_docking().to(
                    device, non_blocking=True
                )
                mol_pos = (
                    mol_x0
                    + pocket_center
                    + torch.randn(3, device=mol_x0.device) * initial_shift_range
                )
                sample = CPIdata.from_pair(
                    ligand_data,
                    protein_data,
                    mol_pos,
                )
                sample = keep_fields(
                    sample,
                    [
                        "pos",
                        "x",
                        "chirality",
                        "chirality_ligand",
                        "chiral_neighbors",
                        "chiral_neighbors_ligand",
                        "ligand_mask",
                        "chem_bond_index",
                        "chem_bond_attr",
                    ],
                ).to(device)
                dock_batch.append(sample)
                ligand_atom_counts.append(int(ligand_data.x.size(0)))

        dock_batch = Batch.from_data_list(dock_batch)

        with torch.no_grad():
            run_kwargs = {}
            if optimized:
                run_kwargs["optimized"] = True
            pred_coords_flat = run_batch(dock_batch, docking_cfg, **run_kwargs)

        pred_confs_flat = torch.split(pred_coords_flat, ligand_atom_counts, dim=0)
        per_data_pred = [
            torch.stack(pred_confs_flat[i : i + num_confs])
            for i in range(0, len(pred_confs_flat), num_confs)
        ]

        if not calculate_score:
            outputs = [
                {"pred_coords": pred_coords.detach().cpu()}
                for pred_coords in per_data_pred
            ]
            return outputs[0] if input_is_single else outputs

        score_batch = []
        chirality_losses = []
        for (ligand_data, protein_data), pred_conf_set in zip(datalist, per_data_pred):
            chiral_neighbors = getattr(ligand_data, "chiral_neighbors", None)
            chirality = getattr(ligand_data, "chirality", None)

            for pred_conf in pred_conf_set:
                sample = CPIdata.from_pair(
                    ligand_data,
                    protein_data,
                    pred_conf,
                )
                sample = keep_fields(
                    sample,
                    [
                        "x",
                        "pos",
                        "ligand_mask",
                        "chem_bond_index",
                        "chem_bond_attr",
                    ],
                ).to(device)
                score_batch.append(sample)

                if chiral_neighbors is None or chirality is None:
                    chirality_losses.append(torch.tensor(0.0, device=device))
                else:
                    chirality_losses.append(
                        batch_chirality_loss_no_gt(
                            pred_conf,
                            chiral_neighbors,
                            chirality,
                        )
                    )

        score_batch = Batch.from_data_list(score_batch)
        score_batch = score_batch.extract_pocket_and_update_graph(
            cutoff=score_model.model.radius_cutoff,
            max_num_neighbors=999,
        )
        with torch.no_grad():
            scores_flat = F.softplus(score_model(score_batch).view(-1), beta=10)

        chirality_losses = torch.stack(chirality_losses)
        scores_flat = torch.where(
            chirality_losses > 0.3,
            scores_flat * 0.7,
            scores_flat,
        )
        score_tensors = scores_flat.view(len(datalist), num_confs)

        if device.type == "cuda" and not optimized:
            torch.cuda.empty_cache()

        outputs = []
        for pred_conf_set, scores in zip(per_data_pred, score_tensors):
            outputs.append(
                {
                    "pred_coords": pred_conf_set.detach().cpu(),
                    "scores": scores.detach().cpu(),
                }
            )
        return outputs[0] if input_is_single else outputs


def _compute_delta_t(t_schedule: torch.Tensor, t: torch.Tensor):
    if t + 1 >= t_schedule.size(0):
        return 0.0

    t_curr, t_next = t_schedule[t : t + 2]
    return t_next - t_curr


def run_batch(batch, docking_cfg, save_each_frame=False, optimized=False):
    n_steps = 50
    terminal_updates = 20
    terminal_count = 0
    rotation_scale = 1.5
    shift_scale = 3.0
    conformation_scale = 2.5
    radius_cutoff = docking_cfg.dataset.dock.radius_cutoff
    steps = n_steps + 1
    device = batch.pos.device
    t_schedule = torch.cos(torch.linspace(0, math.pi / 2, steps, device=device)).flip(0)
    if optimized:
        # Preserve device-generated schedule values; synchronize only once.
        t_host = t_schedule.tolist()
        ligand_indices = batch.ligand_mask.nonzero(as_tuple=False).view(-1)
        node_batch = batch.batch[ligand_indices]
        graph_count = batch.num_graphs
    for i in range(t_schedule.size(0) - 1):
        delta_t = _compute_delta_t(t_schedule, t=i)
        t = t_host[i] if optimized else t_schedule[i]
        if t >= 0.8:
            if terminal_count >= terminal_updates:
                break
            terminal_count += 1

        batch_t = copy(batch)
        batch_t.extract_pocket_and_update_graph(
            radius_cutoff,
            (1.0 if optimized else torch.tensor(1.0, device=device)) if t > 0.9 else t,
        )
        with torch.no_grad():
            if docking_cfg.task == "JointDockingConformation":
                batch_t.task_id = torch.ones_like(batch_t.x[:, 0], dtype=torch.long)
            v_t, R_pred, shift = docking_model(batch_t, task="docking")
            shift *= docking_cfg.dataset.dock.max_translation_shift

        if optimized:
            pos = batch.pos.index_select(0, ligand_indices)
        else:
            pos = batch.pos[batch.ligand_mask]
            node_batch = batch.batch[batch.ligand_mask]

        mile_stone = 0.4
        if t > mile_stone:
            R_pred = (
                torch.eye(3, device=R_pred.device)
                .unsqueeze(0)
                .repeat(R_pred.size(0), 1, 1)
            )
            shift = torch.zeros_like(shift)

        t_float = float(t)
        rotation_fraction_t = delta_t.to(R_pred.device) / max((mile_stone - t_float), 0.01) * rotation_scale
        shift = shift * delta_t.to(shift.device) / max((mile_stone - t_float), 0.01) * shift_scale
        R_pred = rotation_fraction(R_pred, rotation_fraction_t)
        R_graph = R_pred[node_batch]

        centers = scatter(pos, node_batch, dim=0, reduce="mean",
                          **({"dim_size": graph_count} if optimized else {}))
        center_node = centers[node_batch]

        if t < 0.8:
            pos_centered = pos - center_node + v_t * (delta_t / (1 - t_float) * conformation_scale)
        else:
            pos_centered = pos - center_node + v_t

        pos_rot = torch.einsum("ni,nij->nj", pos_centered, R_graph)
        pos_rot = pos_rot + center_node + shift[node_batch]

        if optimized:
            batch.pos.index_copy_(0, ligand_indices, pos_rot)
        else:
            batch.pos[batch.ligand_mask] = pos_rot
        if save_each_frame:
            num_mols = batch.batch.max().item() + 1
            for j in range(num_mols):
                keep_nodes = (batch_t.batch == j).nonzero(as_tuple=False).view(-1)

                edge_index_sub, edge_types_sub, edge_mask = subgraph(
                    subset=keep_nodes,
                    edge_index=batch_t.edge_index,
                    edge_attr=batch_t.edge_attr[:, 0],
                    num_nodes=batch_t.pos.size(0),
                    relabel_nodes=True,
                    return_edge_mask=True,
                )

                pos_sub = batch_t.pos[keep_nodes]
                node_types_sub = batch_t.x[:, 0][keep_nodes]

                edge_index_sub = edge_index_sub[:, edge_types_sub != RADIUS_EDGE_TYPE]
                edge_types_sub = edge_types_sub[edge_types_sub != RADIUS_EDGE_TYPE]

                assert edge_index_sub.max() < pos_sub.shape[0], (
                    "edge_index_sub 超出范围"
                )

                graph_to_sdf_v3000(
                    pos_sub,
                    node_types_sub,
                    edge_index_sub,
                    edge_types_sub,
                    sdf_path=f"../../frames/frame_{j:03d}_{i:03d}.sdf",
                )

    if optimized:
        return batch.pos.index_select(0, ligand_indices)
    return batch.pos[torch.nonzero(batch.ligand_mask, as_tuple=False).view(-1)]


def calc_rmsd(mol, ref_mol, remove_Hs=True):
    if mol is None or ref_mol is None:
        print("Molecule is None")
        return torch.tensor(100.0)

    from data.utils import to_rdkit
    mol, ref_mol = to_rdkit(mol), to_rdkit(ref_mol)
    if remove_Hs:
        mol, ref_mol = Chem.RemoveHs(mol), Chem.RemoveHs(ref_mol)
    coords1, coords2 = mol.GetConformer().GetPositions(), ref_mol.GetConformer().GetPositions()
    numbers1 = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    numbers2 = np.asarray([atom.GetAtomicNum() for atom in ref_mol.GetAtoms()])
    adjacency1, adjacency2 = Chem.GetAdjacencyMatrix(mol), Chem.GetAdjacencyMatrix(ref_mol)
    try:
        with time_limit(2):
            # Preserve docking frame and the established element/connectivity symmetry.
            rmsd = symmrmsd(coords1, coords2, numbers1, numbers2, adjacency1, adjacency2)
    except TimeoutError:
        rmsd = np.nan
    return torch.tensor(rmsd)


@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def remove_hoh_from_pdb(pdb_in: Path, pdb_out: Path):
    with open(pdb_in, "r") as fin, open(pdb_out, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname == "HOH":
                    continue
            fout.write(line)


buster = PoseBusters(config="redock")


def PB_validate_test(pred_mol, gt_mol, path):
    orig_cond_file = Path(path)

    from data.utils import to_rdkit
    pred_mol = Chem.RemoveHs(to_rdkit(pred_mol))
    gt_mol = Chem.RemoveHs(to_rdkit(gt_mol))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            pred_file = tmpdir / "pred.sdf"
            true_file = tmpdir / "gt.sdf"
            cond_file = tmpdir / "protein_nowater.pdb"

            Chem.SDWriter(str(pred_file)).write(pred_mol)
            Chem.SDWriter(str(true_file)).write(gt_mol)

            remove_hoh_from_pdb(orig_cond_file, cond_file)

            df = buster.bust([str(pred_file)], true_file, cond_file)

            bad_cols = df.columns[~df.iloc[0].astype(bool)]
            pb_tensor = torch.tensor(df.iloc[0].values, dtype=torch.bool)

            if len(bad_cols):
                print(f"❌ PoseBusters failed checks:")
                for i, flag in enumerate(bad_cols):
                    print(f" - flag {flag}")

        return pb_tensor

    except Exception as e:
        traceback.print_exc()
        print(f"❌ PoseBusters failed with error: {e}")
        return torch.tensor([False] + [True] * 26)


FAIL_RULES = [
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "tetrahedral_chirality",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
    "rmsd_≤_2å",
]


def dbscan_pockets(points, weights, eps=6.0, min_samples=5, quantile=0.9, center_power=1):
    """
    points: (N, 3) numpy array
    weights: (N,) numpy array
    eps: DBSCAN radius in Å (推荐 5~8)
    min_samples: minimum points to form a pocket
    quantile: confidence filtering quantile
    center_power: center-only weighting exponent; ranking still uses raw score sums

    Returns:
        centers: (K, 3) numpy array
        confidences: (K,) torch.Tensor
    """
    import numpy as np
    import torch
    from sklearn.cluster import DBSCAN

    # 1. 先按 confidence 过滤
    mask = weights > np.quantile(weights, quantile)
    points_f = points[mask]
    weights_f = weights[mask]

    if len(points_f) == 0:
        return np.zeros((0, 3)), torch.zeros(0)

    # 2. DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_f)

    centers = []
    confidences = []

    # 3. 对每个 cluster 计算口袋中心和置信度
    for lab in np.unique(labels):
        if lab == -1:
            continue  # noise

        cluster_points = points_f[labels == lab]
        cluster_weights = weights_f[labels == lab]

        # center：加权平均（物理意义最好）
        center_weights = (
            cluster_weights if center_power == 1
            else np.maximum(cluster_weights, 0) ** center_power
        )
        if center_power != 1 and center_weights.sum() <= 0:
            continue
        center = np.average(cluster_points, axis=0, weights=center_weights)

        # confidence：max / mean / sum 都可以
        # confidence = cluster_weights.max()
        # confidence = cluster_weights.mean() * np.log(len(cluster_weights) + 1)
        confidence = cluster_weights.sum()

        centers.append(center)
        confidences.append(confidence)

    if len(centers) == 0:
        return np.zeros((0, 3)), torch.zeros(0)

    centers = np.asarray(centers)
    confidences = np.asarray(confidences)

    # 4. 按 confidence 排序（和你原来一致）
    order = np.argsort(confidences)[::-1]
    centers = centers[order]
    confidences = confidences[order]

    return centers, torch.tensor(confidences)


VDW_RADII = {
    1: 1.20,  # H
    6: 1.70,  # C
    7: 1.55,  # N
    8: 1.52,  # O
    9: 1.47,  # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}


def evaluate_single_protein_strict(
    centers,
    confidences,
    gt_centers,
    gt_coords,
    gt_ligand_ids,
    threshold=4.0,
):
    """
    Strict evaluation:
    n = number of actual binding sites (M)
    """

    if len(centers) == 0:
        return 0, 0  # DCC_hits, DCA_hits

    centers = np.array(centers)
    confidences = np.array(confidences)

    rank_idx = np.argsort(confidences)[::-1]

    num_ligs = len(gt_centers)

    # 关键：只取 top-M
    top_k = rank_idx[: min(num_ligs, len(rank_idx))]
    pred_centers = centers[top_k]

    # ==========================
    # DCC
    # ==========================
    dcc_success = 0

    for gt_center in gt_centers:
        dist = np.linalg.norm(pred_centers - gt_center, axis=1)
        if (dist <= threshold).any():
            dcc_success += 1

    # ==========================
    # DCA
    # ==========================
    dca_success = 0
    unique_ligs = np.unique(gt_ligand_ids)

    for lig_id in unique_ligs:
        lig_atoms = gt_coords[gt_ligand_ids == lig_id]

        dist = np.linalg.norm(
            lig_atoms[:, None, :] - pred_centers[None, :, :],
            axis=-1,
        )

        if (dist <= threshold).any():
            dca_success += 1

    return dcc_success, dca_success
