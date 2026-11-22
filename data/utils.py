import copy
import os
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List

import Bio.PDB
import cosmolkit as ck
import numpy as np
import torch
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_cluster import radius, radius_graph
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, scatter, subgraph

x_map: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "is_in_ring": [False, True],
}

NUM_ATOM_TYPES = len(x_map["atomic_num"])

e_map: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
        "RADIUS",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}

RADIUS_EDGE_TYPE = 22

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
aa_to_id = {}
for i in range(0, 25):
    aa_to_id["ACDEFGHIKLMNPQRSTVWYXBZUO"[i]] = i


def radius_graph_with_edge_attr_batch(
    pos, edge_index, edge_attr, radius, batch, loop=False
):
    # 创建一个 mask，保留不等于 4 的边
    mask = edge_attr[:, 0] != 4

    # 应用 mask 过滤
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    # === 使用 batch-aware 构图 ===
    edge_index_radius = radius_graph(
        pos, r=radius, loop=loop, batch=batch, max_num_neighbors=999
    )
    edge_attr_radius = (
        torch.tensor([RADIUS_EDGE_TYPE])
        .tile((edge_index_radius.shape[1], 1))
        .to(pos.device)
    )
    all_edge_index = torch.cat([edge_index_radius, edge_index], dim=1)
    all_edge_attr = torch.cat([edge_attr_radius, edge_attr], dim=0)
    # edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce="min", sort_by_row=False)
    edge_index_, edge_attr_ = coalesce(
        all_edge_index, all_edge_attr[:, 0], reduce="min", sort_by_row=False
    )
    edge_index, edge_attr = dedup_edges_prefer_bond(
        all_edge_index, all_edge_attr, sort_by_row=False
    )
    assert torch.all(edge_index_ == edge_index) and torch.all(
        edge_attr_ == edge_attr[:, 0]
    )
    return edge_index, edge_attr


def radius_graph_with_edge_attr_for_ligand_batch(
    pos, edge_index, edge_attr, radius_cutoff, ligand_mask, batch=None, loop=False
):
    """
    Build:
      - ligand↔pocket radius edges
      - ligand↔ligand internal radius edges (low priority)
      - keep chemical bonds highest priority
    """

    device = pos.device
    existing_index = edge_index
    existing_attr = edge_attr

    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=device)

    all_edge_indices = []
    all_edge_attrs = []

    # 1. 提取全局索引和掩码
    mol_idx_all = torch.where(ligand_mask)[0]
    pocket_idx_all = torch.where(~ligand_mask)[0]

    # ---------------------------------------------------------
    # PART A: Ligand ↔ Ligand (使用内置 batch 并行)
    # ---------------------------------------------------------
    # 仅保留配体节点
    pos_mol = pos[ligand_mask]
    batch_mol = batch[ligand_mask]

    edge_index_ll_local = radius_graph(
        pos_mol, r=radius_cutoff, batch=batch_mol, loop=loop
    )
    # 将局部索引映射回全局索引
    edge_index_ll = mol_idx_all[edge_index_ll_local]

    # ---------------------------------------------------------
    # PART B: Ligand ↔ Pocket (跨类交互)
    # ---------------------------------------------------------
    # 使用 torch_cluster.radius 寻找从配体到受体的邻居
    # radius(x, y, r, batch_x, batch_y) -> 寻找 y 在 x 周围 r 范围内的点
    # 这里 x 是 pocket, y 是 mol
    edge_index_lp_raw = radius(
        x=pos[~ligand_mask],  # 受体位置
        y=pos[ligand_mask],  # 配体位置
        r=radius_cutoff,
        batch_x=batch[~ligand_mask],
        batch_y=batch[ligand_mask],
        max_num_neighbors=9999,  # 可选：限制邻居数以防显存峰值
    )

    # 映射回全局索引
    # edge_index_lp_raw[0] 是 mol 侧的局部索引，[1] 是 pocket 侧的局部索引
    src_l = mol_idx_all[edge_index_lp_raw[0]]
    dst_p = pocket_idx_all[edge_index_lp_raw[1]]

    # 构建双向边
    edge_index_lp = torch.stack(
        [torch.cat([src_l, dst_p]), torch.cat([dst_p, src_l])], dim=0
    )

    all_edge_indices = torch.cat([edge_index_ll, edge_index_lp], dim=1)
    num_edges = all_edge_indices.size(1)
    all_edge_attrs = torch.full(
        (num_edges,), fill_value=RADIUS_EDGE_TYPE, device=device, dtype=edge_attr.dtype
    )

    # print(all_edge_indices, all_edge_indices.shape)
    # print(all_edge_attrs, all_edge_attrs.shape)
    # T2 =time.perf_counter()
    # print('radius 运行时间:%s毫秒\n' % (T2 - T1))

    # exit()

    # -------------------------
    # Merge collected edges
    # -------------------------
    if len(all_edge_indices) > 0:
        new_edges = all_edge_indices
        new_attrs = all_edge_attrs
        # new_edges = torch.cat(all_edge_indices, dim=1)
        # new_attrs = torch.cat(all_edge_attrs, dim=0)
    else:
        new_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        new_attrs = torch.empty((0,), dtype=edge_attr.dtype, device=device)

    # -------------------------
    # Combine with existing edges
    # -------------------------
    edge_index = torch.cat([existing_index, new_edges], dim=1)
    edge_attr = torch.cat([existing_attr, new_attrs], dim=0)

    # chemical bonds win over radius‐edges because reduce="min"
    edge_index, edge_attr = coalesce(
        edge_index, edge_attr, reduce="min", sort_by_row=False
    )

    return edge_index, edge_attr


def map_index_with_default(src, index, default=-1):
    """
    src:   (N,)
    index: (M,)
    return:
      mapped: (N,)  每个 src[i] 在 index 中的位置，不存在则为 default
      mask:   (N,)  是否命中
    """
    # (N, 1) == (1, M) -> (N, M)
    match = src.unsqueeze(1) == index.unsqueeze(0)

    mask = match.any(dim=1)
    mapped = torch.full_like(src, default)

    # argmax 只在 mask=True 的位置使用
    mapped[mask] = match[mask].long().argmax(dim=1)

    return mapped, mask


def protein_edge_types(edge_index, max_offset=10):
    """
    Assigns edge types based on the difference between connected node indices.
    Edge type IDs start from 1; 0 means not in the offset range.

    Args:
        edge_index (Tensor): Shape [2, num_edges].
        max_offset (int): Max absolute offset to consider.

    Returns:
        Tensor: Edge types for each edge (0 if outside range).
    """
    import torch

    edge_types = torch.zeros(
        edge_index.size(1), dtype=torch.long, device=edge_index.device
    )
    diff = edge_index[0] - edge_index[1]

    for i, offset in enumerate(range(-max_offset, max_offset + 1), start=1):
        edge_types[diff == offset] = i

    return edge_types


def extract_chiral_neighbors(chirality, edge_index, pos, atomic_number):
    """
    Extract chiral neighbors considering only non-H atoms.
    For centers with 4 non-H neighbors, only first 3 are used to compute oriented volume and returned.

    Args:
        chirality: (N,) LongTensor, nonzero indicates chiral center candidate
        edge_index: (2, E) LongTensor
        pos: (N, 3) FloatTensor
        atomic_number: (N,) LongTensor, 1 for H, >1 for other atoms

    Returns:
        new_chirality: (N,) LongTensor
        chiral_neighbors: (M, 3) LongTensor, indices of 3 non-H neighbors
    """
    from collections import defaultdict

    import torch

    device = chirality.device
    N = chirality.size(0)

    # --- build adjacency list ---
    neighbors = defaultdict(list)
    row, col = edge_index
    for r, c in zip(row.tolist(), col.tolist()):
        neighbors[r].append(c)

    chiral_neighbors = []
    valid_mask = torch.zeros(N, dtype=torch.bool, device=device)

    for i in range(N):
        if chirality[i].item() == 0:
            continue

        # --- keep only non-H neighbors ---
        nbrs_all = [j for j in neighbors[i] if atomic_number[j] != 1]
        deg = len(nbrs_all)

        if deg < 3:
            # skip centers with fewer than 3 non-H neighbors
            continue

        # pick first 3 neighbors for oriented volume
        nbrs = nbrs_all[:3]

        c = pos[i]
        nbrs_tensor = torch.tensor(nbrs, device=device, dtype=torch.long)
        vecs = pos[nbrs_tensor] - c  # (3, 3)

        # --- compute oriented volume ---
        v1, v2, v3 = vecs[0], vecs[1], vecs[2]
        det = torch.dot(v1, torch.cross(v2, v3, dim=0))

        # --- if orientation negative, swap two neighbors ---
        if det < 0:
            nbrs[2], nbrs[1] = nbrs[1], nbrs[2]

        chiral_neighbors.append(nbrs)
        valid_mask[i] = True

    new_chirality = valid_mask.long()

    if len(chiral_neighbors) == 0:
        return new_chirality, torch.empty((0, 3), dtype=torch.long, device=device)

    assert torch.sum(new_chirality != 0) == len(chiral_neighbors)
    return new_chirality, torch.tensor(
        chiral_neighbors, device=device, dtype=torch.long
    )


def normalized_chirality_loss_no_gt(
    pos: torch.Tensor,
    chiral_neighbors: torch.Tensor | None,
    chirality: torch.Tensor | None,
    margin: float = 0.02,
) -> torch.Tensor:
    """Measure whether a coordinate tensor preserves the stored chirality order.

    The chirality representation stores direction in ``chiral_neighbors`` order:
    for each valid chiral center, the three neighbor vectors should have
    positive oriented volume.  We normalize by vector lengths so the score is
    comparable across differently scaled prior samples.
    """

    device = pos.device
    if chirality is None or chiral_neighbors is None:
        return torch.tensor(0.0, device=device)
    if chiral_neighbors.numel() == 0:
        return torch.tensor(0.0, device=device)

    chirality = chirality.to(device)
    chiral_neighbors = chiral_neighbors.to(device)
    center_idx = torch.nonzero(chirality > 0, as_tuple=False).view(-1)
    if center_idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    nbr_idx = chiral_neighbors[:, :3].long()
    n_pairs = min(center_idx.numel(), nbr_idx.size(0))
    if n_pairs == 0:
        return torch.tensor(0.0, device=device)

    center_idx = center_idx[:n_pairs]
    nbr_idx = nbr_idx[:n_pairs]

    valid_rows = (nbr_idx >= 0).all(dim=1) & (nbr_idx < pos.size(0)).all(dim=1)
    if not valid_rows.any():
        return torch.tensor(0.0, device=device)

    center_idx = center_idx[valid_rows]
    nbr_idx = nbr_idx[valid_rows]

    centers = pos[center_idx]
    nbr_pos = pos[nbr_idx]
    vecs = nbr_pos - centers.unsqueeze(1)
    v1, v2, v3 = vecs[:, 0], vecs[:, 1], vecs[:, 2]

    det = torch.sum(v1 * torch.cross(v2, v3, dim=1), dim=1)
    denom = (v1.norm(dim=1) * v2.norm(dim=1) * v3.norm(dim=1)).clamp_min(1.0e-8)
    signed_volume = det / denom

    return torch.relu(margin - signed_volume).mean()


def mirror_prior_coordinates(pos: torch.Tensor) -> torch.Tensor:
    """Reflect a prior sample around its centroid to flip all handedness signs."""

    center = pos.mean(dim=0, keepdim=True)
    mirrored = pos - center
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored + center


def select_chiral_prior_sample(
    sample_once,
    chiral_neighbors: torch.Tensor | None,
    chirality: torch.Tensor | None,
    max_attempts: int = 32,
    margin: float = 0.02,
    accept_loss: float = 0.0,
) -> torch.Tensor:
    """Sample x0 repeatedly and keep the candidate with best chirality.

    ``sample_once`` must preserve the original x0 construction path.  This
    helper only evaluates the generated sample and its global mirror image.
    If no candidate is fully acceptable within ``max_attempts``, the lowest-loss
    candidate is returned.
    """

    if chirality is None or chiral_neighbors is None or chiral_neighbors.numel() == 0:
        return sample_once()

    max_attempts = max(1, int(max_attempts))
    best_x0 = None
    best_loss = None

    for _ in range(max_attempts):
        x0 = sample_once()

        normal_loss = normalized_chirality_loss_no_gt(
            x0, chiral_neighbors, chirality, margin=margin
        )
        mirrored_x0 = mirror_prior_coordinates(x0)
        mirrored_loss = normalized_chirality_loss_no_gt(
            mirrored_x0, chiral_neighbors, chirality, margin=margin
        )

        if mirrored_loss < normal_loss:
            candidate = mirrored_x0
            candidate_loss = mirrored_loss
        else:
            candidate = x0
            candidate_loss = normal_loss

        if best_loss is None or candidate_loss < best_loss:
            best_x0 = candidate
            best_loss = candidate_loss

        if candidate_loss <= accept_loss:
            return candidate

    return best_x0


from torch_geometric.utils import get_laplacian, to_dense_adj


class HarmonicSampler:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.eig_val_cache = {}
        self.eig_vec_cache = {}

    def diagonalize(self, n_nodes, edges=[], batch=None, smiles=None):
        a = self.alpha * torch.ones((edges.shape[0],), device=edges.device)
        edge_index, edge_weight = get_laplacian(
            edges.T,
            a,
            num_nodes=n_nodes,
        )

        H = to_dense_adj(
            edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=n_nodes
        ).squeeze()

        if batch is None:
            D, P = torch.linalg.eigh(H)
            return D, P

        Ds, Ps = [], []
        batch_size = batch.max() + 1

        for i in range(batch_size):
            idx = torch.where(batch == i)[0]
            start = idx.min()
            end = idx.max() + 1

            D, P = None, None
            if smiles is not None:
                D, P = self.check_cache(smiles[i])

                if (D is not None) and (P is not None):
                    D = D.to(edge_index.device)
                    P = P.to(edge_index.device)

            if (D is None) or (P is None):
                D, P = torch.linalg.eigh(H[start:end, start:end])

                if smiles is not None:
                    self.eig_val_cache[smiles[i]] = D.cpu()
                    self.eig_vec_cache[smiles[i]] = P.cpu()

            Ds.append(D)
            Ps.append(P)

        return torch.cat(Ds), torch.block_diag(*Ps)

    def check_cache(self, smiles):
        D = self.eig_val_cache.get(smiles, None)
        P = self.eig_vec_cache.get(smiles, None)
        return D, P

    def sample(self, size, edge_index, batch=None, smiles=None):
        # transpose if (2, n_edges)
        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        n_nodes = size[0]
        D, P = self.diagonalize(
            n_nodes=n_nodes, edges=edge_index, batch=batch, smiles=smiles
        )

        # get starting index per sample in batch
        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            cum_sum = counts.cumsum(0)[:-1].to(D.device)
            zero = torch.zeros(1).to(D.device)
            start_index = torch.concat((zero, cum_sum)).long()

        std = 1.0 / torch.sqrt(D)
        std[start_index] = 0.0

        noise = torch.randn(size).to(D.device)
        noise = std[:, None] * noise
        noise[noise.isnan()] = 0.0
        sample = P @ (noise)

        return sample

    def energy(self, x, edge_index, batch=None, smiles=None):
        n_nodes = x.size(0)
        x = center_of_mass(x)

        if batch is None:
            batch = torch.zeros(n_nodes).to(x.device).long()

        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        D, P = self.diagonalize(n_nodes, edges=edge_index, batch=batch, smiles=smiles)

        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            cum_sum = counts.cumsum(0)[:-1]
            zero = torch.zeros(1).to(D.device)
            start_index = torch.concat((zero, cum_sum)).long()

        energy_unpooled = D[:, None] * (P.T @ x) ** 2
        energy_unpooled[start_index] = 0.0
        energy_unpooled = energy_unpooled.sum(-1)
        energy = 0.5 * scatter(energy_unpooled, batch)

        return energy.view(-1, 1)


def center_of_mass(x, dim=0, batch=None):
    num_nodes = x.size(0)

    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

    x_com = scatter(x, batch, dim=dim, reduce="mean")[batch]
    return x - x_com


def rodrigues_rotation_matrix_():
    # Randomly generate a uniformly distributed rotation axis
    axis = np.random.normal(0, 1, 3)  # Sample from a normal distribution
    axis = axis / np.linalg.norm(
        axis
    )  # Normalize to distribute uniformly on the sphere

    # Randomly generate a uniformly distributed rotation angle
    theta = np.random.uniform(-1, 1) * np.pi  # Uniformly distribute the angle

    # Construct the skew-symmetric matrix K
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    # Calculate the Rodrigues rotation matrix
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


class Molecule(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "chiral_neighbors" or key == "chiral_neighbors_CW_CCW":
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def sample_conformer(self):
        """随机采样一个构象，并进行中心化（可选镜像）"""
        if hasattr(self, "num_confs"):
            conf_idx = torch.randint(0, self.num_confs, (1,))
            pos = self.confs_pos[conf_idx].squeeze(0)
            if hasattr(self, "confs_totalenergy"):
                self.totalenergy = self.confs_totalenergy[conf_idx].squeeze(0)
        else:
            pos = self.pos
        self.y = self.pos = pos
        del self.confs_pos
        if hasattr(self, "confs_totalenergy"):
            del self.confs_totalenergy
        return self

    def get_train_sample(self):
        del self.mol
        pos = self.pos
        self.y = self.pos = pos
        return self

    def get_center(self):
        return self.pos[self.x[:, 0] != 1].mean(dim=0)

    def sample_base_distribution_harmonic(
        self,
        ligand_center=None,
        align=False,
        chirality_prior_max_attempts=32,
        chirality_prior_margin=0.02,
    ):
        """Sample base distribution (x0) and align if necessary."""

        batch = torch.zeros(self.pos.shape[0], dtype=torch.long)

        def sample_once():
            # Original harmonic prior path: sample x0 from the harmonic base
            # distribution, then apply the same optional alignment/translation.
            x0 = sample_base_dist(
                size=self.pos.shape,
                edge_index=self.chem_bond_index,
                smiles=[self.smiles if hasattr(self, "smiles") else None],
                batch=batch,
            )

            if align:
                x0 = rmsd_align(pos=x0, ref_pos=self.pos, batch=batch)
            if ligand_center is not None:
                x0 = x0 + ligand_center
            return x0

        # x0 = select_chiral_prior_sample(
        #     sample_once,
        #     getattr(self, "chiral_neighbors", None),
        #     getattr(self, "chirality", None),
        #     max_attempts=chirality_prior_max_attempts,
        #     margin=chirality_prior_margin,
        # )
        x0 = sample_once()
        return x0, batch

    def sample_base_distribution_docking(
        self,
        chirality_prior_max_attempts=32,
        chirality_prior_margin=0.02,
    ):
        """Sample base distribution (x0) and align if necessary."""

        batch = torch.zeros(self.x.shape[0], dtype=torch.long)

        def sample_once():
            # Original docking prior path: harmonic base sample followed by a
            # random proper rotation.  The chirality selector wraps this path
            # without modifying the harmonic sampler itself.
            x0 = sample_base_dist(
                size=(self.x.shape[0], 3),
                edge_index=self.chem_bond_index,
                smiles=[self.smiles],
                batch=batch,
            )

            R_random = (
                torch.from_numpy(rodrigues_rotation_matrix_()).float().to(x0.device)
            )
            return x0 @ R_random

        # return sample_once()

        x0 = select_chiral_prior_sample(
            sample_once,
            getattr(self, "chiral_neighbors", None),
            getattr(self, "chirality", None),
            max_attempts=chirality_prior_max_attempts,
            margin=chirality_prior_margin,
        )

        return x0

    def sample_base_distribution_docking_train(
        self,
        ligand_center,
        shift_range,
        chirality_prior_max_attempts=32,
        chirality_prior_margin=0.02,
        prior_extra_noise_std=0.0,
    ):
        """Sample base distribution (x0) and align if necessary."""

        batch = torch.zeros(self.pos.shape[0], dtype=torch.long)

        def sample_once():
            # Original docking-train prior path: harmonic base sample followed
            # by a random proper rotation.  We keep this untouched and only
            # select/mirror the resulting x0 based on the stored chirality.
            x0 = sample_base_dist(
                size=self.pos.shape,
                edge_index=self.chem_bond_index,
                smiles=[self.smiles],
                batch=batch,
            )

            R_random = torch.from_numpy(rodrigues_rotation_matrix_()).float()
            return x0 @ R_random

        x0 = sample_once()
        if prior_extra_noise_std > 0:
            noise = torch.randn_like(x0) * prior_extra_noise_std
            x0 = x0 + noise - noise.mean(dim=0, keepdim=True)
        # x0 = select_chiral_prior_sample(
        #     sample_once,
        #     getattr(self, "chiral_neighbors", None),
        #     getattr(self, "chirality", None),
        #     max_attempts=chirality_prior_max_attempts,
        #     margin=chirality_prior_margin,
        # )
        R, _ = rmsd_align_transformation(
            pos=x0, ref_pos=self.pos - ligand_center, batch=batch
        )

        if shift_range is not None:
            rand_shift = torch.randn(3)
            shift = rand_shift / rand_shift.norm() * shift_range * torch.rand(1)
        else:
            shift = torch.zeros_like(x0[0])

        return x0, R, shift, batch

    def compute_flow_field_docking(
        self, x0, R, t, ligand_center, shift, train_shift_range
    ):
        """
        Two-stage docking flow:
        - Early stage: global rigid-body motion (rotation + translation),
            guiding the ligand from pocket entrance to pocket center.
        - Late stage: fine-grained stabilization via residual flow.
        """

        flow = compute_docking_flow_fields(
            ligand_pos=self.pos,
            x0=x0,
            R=R,
            t=t,
            ligand_center=ligand_center,
            shift=shift,
            train_shift_range=train_shift_range,
        )
        # graph_to_sdf_v3000(
        #     x0_aligned,
        #     self.node_types,
        #     self.edge_index,
        #     self.edge_types,
        #     sdf_path=f"./x0_aligned.sdf"
        # )

        # =========================
        # 6. Bookkeeping
        # =========================
        self.y = flow.y
        self.pos = flow.pos
        self.u_t = flow.u_t
        self.t = torch.full_like(self.x[:, 0], flow.t.item(), dtype=torch.float)
        self.R_pred = flow.R_pred
        self.shift = flow.shift

        return self

    def update_pos(self, pos):
        self.pos = pos

    def remove_hydrogens(self):
        keep_nodes = self.x[:, 0] != 1

        # 提取子图（节点、边、边属性）
        edge_index_sub, edge_types_sub, edge_mask = subgraph(
            subset=keep_nodes,
            edge_index=self.chem_bond_index,
            edge_attr=self.chem_bond_attr,
            num_nodes=self.x.size(0),
            relabel_nodes=True,
            return_edge_mask=True,
        )

        if hasattr(self, 'pos') and self.pos is not None:
            self.pos = self.pos[keep_nodes]
        if hasattr(self, "confs_pos"):
            self.confs_pos = self.confs_pos[:, keep_nodes]
        self.x = self.x[keep_nodes]
        self.chirality = self.chirality[keep_nodes]
        self.keep_nodes = keep_nodes
        self.chem_bond_index = edge_index_sub
        self.chem_bond_attr = edge_types_sub

        keep_nodes = torch.nonzero(torch.ones_like(keep_nodes), as_tuple=False).view(
            -1
        )[keep_nodes]

        self.chiral_neighbors = map_index_with_default(
            self.chiral_neighbors.view(-1),
            keep_nodes,
            default=-1,
        )[0].view(-1, 3)

    def virtual_node(self, center=None):
        if center is None:
            center = self.get_center()
        self.pos = center.unsqueeze(0)
        self.node_types = torch.cat([torch.tensor([99])], dim=0)
        self.edge_index = torch.tensor([], dtype=torch.long).view(2, 0)
        self.edge_types = torch.tensor([], dtype=torch.long)

        self.chirality = None

        return self

    def sample_base_distribution_gen(self, x0_std=0.1):
        """Sample base distribution (x0) and align if necessary."""

        N = self.node_types.shape[0]  # 原子节点数
        add_N = 2  # 额外添加的节点数
        total_N = N + add_N

        # 1) batch 信息（假设所有节点同属一个分子）
        batch = torch.zeros(total_N, dtype=torch.long)

        # 2) x0 初始坐标采样
        # x0 = torch.randn(total_N, 3) * torch.tensor([x0_std,x0_std,x0_std]).squeeze(0)
        x0 = torch.randn(total_N, 3) * torch.tensor(
            [x0_std, x0_std, x0_std * 2.0]
        ).squeeze(0)
        x1_add = x0[N:].to(self.pos.device)
        self.real_pos = torch.cat([self.pos, x1_add], dim=0)

        # 3) node_attr 初始化
        node_attr = torch.randn(total_N, 101) * 100
        node_attr = torch.softmax(node_attr, dim=-1)

        real_attr = torch.zeros((total_N, 101))
        real_attr[torch.arange(N), self.node_types] = 1.0
        real_attr[N:, 100] = 1.0
        self.real_attr = real_attr

        return x0, batch, node_attr

    def remove_H(self, keep_chiral_H: bool = True):
        """
        移除非必要氢，可选保留参与手性中心的氢。

        参数:
            keep_chiral_H (bool): 是否保留参与手性中心的氢
            chiral_neighbors (torch.Tensor, optional): (n_chiral, 4)，
                每个手性中心的4个邻居索引。如果提供且keep_chiral_H=True，
                将保留出现在该列表中的氢。
        """

        # 基础 mask：非氢原子
        # non_H_mask = self.node_types > 99
        non_H_mask = self.node_types != 1

        if keep_chiral_H and self.chiral_neighbors is not None:
            # chiral_neighbors: (N, K)
            # 将 chiral_neighbors 展平成一维索引
            chiral_H_indices = self.chiral_neighbors.view(-1)

            # 找出哪些节点是氢
            is_H = self.node_types == 1

            # 找出哪些节点在 chiral_H_indices 里
            # torch.isin 在新版本 PyTorch 里支持直接比较
            is_chiral_H = torch.isin(
                torch.arange(len(self.node_types), device=self.node_types.device),
                chiral_H_indices,
            )

            # 保留手性氢：对 non_H_mask 进行更新
            non_H_mask = non_H_mask | (is_H & is_chiral_H)
            # non_H_mask = non_H_mask | is_chiral_H

        print(self.chiral_neighbors, non_H_mask)

        # 应用 mask
        self.pos = self.pos[non_H_mask]
        self.y = self.y[non_H_mask]
        # self.node_types[is_chiral_H] = 14
        self.node_types = self.node_types[non_H_mask]
        if hasattr(self, "chirality"):
            self.chirality = self.chirality[non_H_mask]

        # 获取保留节点索引
        keep_nodes = torch.nonzero(non_H_mask, as_tuple=False).view(-1)

        # 更新子图（重新映射节点索引）
        self.edge_index_origin, self.edge_attr_origin, mask = subgraph(
            subset=keep_nodes,
            edge_index=self.edge_index_origin,
            edge_attr=self.edge_attr_origin,
            relabel_nodes=True,
            return_edge_mask=True,
        )

        # 同步更新当前边信息
        self.edge_index = self.edge_index_origin
        self.edge_types = self.edge_attr_origin

    def compute_flow_field(
        self,
        x0,
        t,
        terminal_time_threshold: float = 0.8,
        terminal_x0_noise_std: float = 0.0,
    ):
        """Compute conditional flow field and update self state."""
        x1 = self.pos
        if t > terminal_time_threshold and terminal_x0_noise_std > 0:
            x0 = x0 + torch.randn_like(x0) * terminal_x0_noise_std
        x_t = (1.0 - t) * x0 + t * x1

        # =========================
        # 5. Residual (contractive) flow
        # =========================
        u_t = x1 - x_t

        # =========================
        # 6. Bookkeeping
        # =========================
        self.y = self.pos
        self.pos = x_t
        self.u_t = u_t
        self.t = torch.full_like(self.x[:, 0], t.item(), dtype=torch.float)

        return self

    def update_graph(self, radius_cutoff, t=None):
        batch = getattr(self, "batch", None)
        if batch is None:
            batch = torch.zeros(
                self.pos.shape[0], dtype=torch.long, device=self.pos.device
            )

        self.edge_index, self.edge_attr = radius_graph_with_edge_attr_batch(
            self.pos,
            self.chem_bond_index,
            edge_attr=self.chem_bond_attr,
            radius=radius_cutoff,
            batch=batch,
            loop=False,
        )

        if t is not None:
            self.t = torch.full_like(self.x[:, 0], t.item(), dtype=torch.float)

        if hasattr(self, "chirality"):
            self.edge_chirality_feat = generate_chirality_edge_features_vectorized(
                self.edge_index, self.chirality, self.chiral_neighbors
            )
        self.edge_vector = self.pos[self.edge_index[0]] - self.pos[self.edge_index[1]]

        self.to(self.pos.device)

        return self


# Molecular graph conversion and chemistry shared by training and inference.


def get_or_generate_conformer(mol):
    # 如果没有构象
    if mol.GetNumConformers() == 0:
        # print("No conformer found, generating...")
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # 可复现（可选）

        # 生成3D构象
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            raise ValueError("Embedding failed")

    return mol.GetConformer()


def _cosmolkit_graph(mol: Any, keep_chirality=False):
    # An unassigned configuration has no target orientation. Within an
    # assigned configuration, None denotes hydrogen or a lone pair.
    stereo = []
    if keep_chirality:
        stereo = [
            (center, [i for i in configuration if i is not None])
            for center, configuration in mol.tetrahedral_stereo(include_unassigned=False)
            if configuration is not None
        ]
    if any(len(neighbors) < 3 for _, neighbors in stereo):
        raise ValueError("Assigned tetrahedral center has fewer than three explicit neighbors")
    chiral_idx_set = {center for center, _ in stereo}

    xs = []
    for atom in mol.atoms:
        idx = atom.id
        xs.append(
            [
                x_map["atomic_num"].index(atom.atomic_number),
                int(idx in chiral_idx_set),
            ]
        )
    x = torch.tensor(xs, dtype=torch.long).view(-1, 2)

    edge_indices, edge_attrs = [], []
    for bond in mol.bonds:
        i = bond.begin
        j = bond.end
        bond_type = bond.order
        bond_type_name = getattr(bond_type, "name", str(bond_type)).upper()
        edge_attr = [e_map["bond_type"].index(bond_type_name)]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [edge_attr, edge_attr]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).reshape(-1, 2).t()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).reshape(-1, 1)
    return x, edge_index, edge_attr, stereo


def molecule_to_graph(
    mol: ck.Molecule,
    keep_mol=False,
    keep_smiles=False,
    keep_chirality=False,
):
    x, edge_index, edge_attr, stereo = _cosmolkit_graph(
        mol, keep_chirality=keep_chirality
    )

    if keep_chirality:
        chirality = x[:, 1]
        if len(stereo) == 0:
            chiral_neighbors = torch.empty((0, 3), dtype=torch.long)
        else:
            chiral_neighbors = torch.tensor(
                [neighbors[:3] for _, neighbors in stereo], dtype=torch.long
            ).view(-1, 3)
    else:
        chirality = None
        chiral_neighbors = None

    smiles = mol.to_smiles(isomeric_smiles=True) if keep_smiles else None
    try:
        pos = torch.tensor(mol.coordinates_3d(), dtype=torch.float)
    except Exception:
        pos = None
    return Molecule(
        x=x,
        chem_bond_index=edge_index,
        chem_bond_attr=edge_attr,
        pos=pos,
        chirality=chirality,
        chiral_neighbors=chiral_neighbors,
        smiles=smiles,
        mol=mol if keep_mol else None,
    )


def rdkit_molecule_to_graph(
    mol,
    keep_mol=False,
    keep_smiles=False,
    keep_chirality=False,
    includeUnassigned=True,
):
    assert isinstance(mol, Chem.Mol)

    if includeUnassigned:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    else:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    chiral_idx_set = {idx for idx, _ in chiral_centers}

    xs: List[List[int]] = []
    for i, atom in enumerate(mol.GetAtoms()):
        row: List[int] = []
        row.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
        row.append(int(i in chiral_idx_set))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 2)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map["bond_type"].index(str(bond.GetBondType())))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 1)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    conf = get_or_generate_conformer(mol)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

    if keep_chirality:
        chirality, chiral_neighbors = extract_chiral_neighbors(
            x[:, 1], edge_index, pos, x[:, 0]
        )
    else:
        chirality = None
        chiral_neighbors = None

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if keep_smiles else None
    return Molecule(
        x=x,
        pos=pos,
        chem_bond_index=edge_index,
        chem_bond_attr=edge_attr,
        chirality=chirality,
        chiral_neighbors=chiral_neighbors,
        smiles=smiles,
        mol=mol if keep_mol else None,
    )


def read_ligand(description):
    path = os.path.expanduser(description)
    if path.lower().endswith('.sdf'):
        return ck.Molecule.read_sdf(path)
    if path.lower().endswith('.mol2'):
        return ck.Molecule.read_mol2(path)
    return ck.Molecule.from_smiles(description)


def remove_all_hydrogens(mol: ck.Molecule):
    """Remove every explicit H when coordinates contain heavy atoms only."""
    return mol.without_hydrogens(
        remove_degree_zero=True,
        remove_higher_degrees=True,
        remove_only_h_neighbors=True,
        remove_isotopes=True,
        remove_dummy_neighbors=True,
        remove_defining_bond_stereo=True,
        remove_with_wedged_bond=True,
        remove_with_query=True,
        remove_mapped=True,
        remove_in_sgroups=True,
        remove_nonimplicit=True,
        remove_hydrides=True,
        remove_nontetrahedral_neighbors=True,
        show_warnings=False,
    )


def get_confs_from_coords(mol: ck.Molecule, pos, has_hydrogens=False):
    """Build conformers with the explicitly requested hydrogen representation.

    When True, retain all atoms and coordinates, including explicit H.
    When False, coordinates must follow the hydrogen-stripped molecule.
    """
    if hasattr(pos, 'detach'):
        pos = pos.detach().cpu().numpy()
    pos = np.asarray(pos, dtype=float)
    if not has_hydrogens:
        mol = remove_all_hydrogens(mol)
    if pos.ndim != 3 or pos.shape[1:] != (len(mol.atoms), 3):
        raise ValueError(
            f'Coordinate/atom shape mismatch: {pos.shape}; template has {len(mol.atoms)} atoms'
        )
    if not np.isfinite(pos).all():
        raise ValueError('Non-finite conformer coordinates')
    return [mol.with_only_3d_conformer(coords) for coords in pos]


def to_rdkit(mol):
    """Convert a molecule for PoseBusters and RDKit-based evaluation."""
    from rdkit import Chem
    if isinstance(mol, Chem.Mol):
        return Chem.Mol(mol)
    result = Chem.MolFromMolBlock(mol.to_sdf_3d(), removeHs=False)
    if result is None:
        raise ValueError('Cannot convert molecule for RDKit evaluation')
    return result


def save_sdf(mol, path, has_hydrogens=False):
    if not has_hydrogens:
        mol = remove_all_hydrogens(mol)
    mol.write_sdf(str(path))


def aligned_rmsd(mol, reference):
    return mol.without_hydrogens().best_rmsd_to(
        reference.without_hydrogens())


def molecular_graph(mol: ck.Molecule):
    atoms = mol.atoms
    numbers = np.array([a.atomic_number for a in atoms], dtype=int)
    adjacency = np.zeros((len(atoms), len(atoms)), dtype=int)
    for bond in mol.bonds:
        adjacency[bond.begin, bond.end] = adjacency[bond.end, bond.begin] = 1
    return numbers, adjacency


def rotatable_quartets(mol):
    # Same non-strict Lipinski definition and terminal tie-break as trained scoring.
    query = ck.parse_smarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    numbers, adjacency = molecular_graph(mol)
    seen, quartets = set(), []
    for match in mol.substruct_matches(query):
        a, b = match.atom_mapping
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        def terminal(center, other):
            candidates = [int(i) for i in np.flatnonzero(adjacency[center])
                          if i != other and numbers[i] > 1]
            return max(candidates, key=lambda i: (numbers[i], adjacency[i].sum(), -i)) if candidates else None
        left, right = terminal(a, b), terminal(b, a)
        if left is not None and right is not None:
            quartets.append((left, a, b, right))
    return quartets


CACHE_INDEX_PROPERTY = 'cosmol_cache_index'


def cache_mapping(mol, numbers, bond_index):
    """Read validated cache indices from atom properties, or bind them once.

    Mutates only atom metadata. Properties survive CK hydrogen/conformer
    operations, but are not automatically persisted by our HDF5 caches or
    plain SMILES. Such inputs still require initial graph matching.
    """
    numbers = np.asarray(numbers, dtype=int)
    atomic_numbers, adjacency = molecular_graph(mol)
    heavy = np.flatnonzero(atomic_numbers != 1)
    source = adjacency[np.ix_(heavy, heavy)]
    target = np.zeros((len(numbers), len(numbers)), dtype=int)
    target[tuple(np.asarray(bond_index, dtype=int))] = 1
    if len(heavy) != len(numbers):
        raise ValueError('CosmolKit/cache heavy atom count mismatch')
    mapping = np.full(len(atomic_numbers), -1, dtype=int)
    stored = [mol.atom_property_int(int(i), CACHE_INDEX_PROPERTY) for i in heavy]
    if any(value is not None for value in stored):
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in stored):
            raise ValueError('Incomplete or invalid cache-index atom properties')
        indices = np.asarray(stored, dtype=int)
        if (not np.array_equal(np.sort(indices), np.arange(len(heavy)))
                or not np.array_equal(atomic_numbers[heavy], numbers[indices])
                or not np.array_equal(source, target[np.ix_(indices, indices)])):
            raise ValueError('Cache-index atom properties do not match this cache')
        mapping[heavy] = indices
        return mapping

    def bind():
        for i in heavy:
            mol.set_atom_property_int_(int(i), CACHE_INDEX_PROPERTY, int(mapping[i]))
        return mapping

    if np.array_equal(atomic_numbers[heavy], numbers) and np.array_equal(source, target):
        mapping[heavy] = np.arange(len(heavy))
        return bind()
    import rustworkx as rx
    from spyrmsd import graph
    matches = rx.vf2_mapping(graph.graph_from_adjacency_matrix(source, atomic_numbers[heavy]),
                             graph.graph_from_adjacency_matrix(target, numbers), lambda a, b: a == b)
    match = next(matches, None)
    if match is None:
        raise ValueError('CosmolKit/cache molecular graphs are not isomorphic')
    for a, b in match.items():
        mapping[heavy[a]] = b
    return bind()


harmonic_sampler = HarmonicSampler(alpha=1.0)


def check_disconnected_components(n_nodes, edge_index):
    """Check for disconnected components using Union-Find algorithm."""
    # Initialize parent array for union-find
    parent = list(range(n_nodes))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process all edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        union(src, dst)

    # Count unique components
    components = {}
    for node in range(n_nodes):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)

    return list(components.values())


def sample_base_dist(
    size: torch.Size, edge_index=None, batch=None, smiles=None, prior_type="harmonic"
):
    """Sample from prior distribution (Either harmonic or gaussian)"""
    if size[0] == 1:
        return torch.zeros(size=size)

    if prior_type == "harmonic":
        assert (edge_index is not None) and (batch is not None)
        x0 = harmonic_sampler.sample(
            size=size, edge_index=edge_index, batch=batch, smiles=smiles
        )

        # check if x0 is nan
        if torch.isnan(x0).any():
            print(check_disconnected_components(batch.shape[0], edge_index))
            raise ValueError(
                "x0 is NaN. Check edge_index for disconnected graphs!",
                edge_index,
                batch.shape,
                smiles,
            )

        return x0

    # gaussian prior if not harmonic
    return torch.randn(size=size)


def sample_time(
    low: float = 1e-4,
    high: float = 0.9999,
):
    return torch.zeros(size=(1,)).uniform_(low, high)


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor(
            [[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float
        )
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Ensure R is a proper rotation matrix
    if torch.det(R) < 0:  # reflection
        V[:, -1] *= -1  # flip the sign of the last column of V
        R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def rmsd_align(pos, ref_pos, batch):
    aligned_pos = []
    batch_size = batch.max() + 1
    for i in range(batch_size):
        index = torch.where(batch == i)[0]
        pos_i = pos[index]
        ref_pos_i = ref_pos[index]
        R, t = find_rigid_alignment(pos_i, ref_pos_i)

        pos_i = (R @ pos_i.T).T + t
        aligned_pos.append(pos_i)

    return torch.concat(aligned_pos, dim=0)


def rmsd_align_transformation(pos, ref_pos, batch):
    R, t = find_rigid_alignment(pos, ref_pos)
    return R.T, t


import torch.nn.functional as F


def rotation_fraction(R: torch.Tensor, fraction: float | torch.Tensor) -> torch.Tensor:
    """
    Compute fractional rotation R ** fraction (batch supported).
    Args:
        R: rotation matrix of shape [3, 3] or [B, 3, 3]
        fraction: scalar or tensor [B] representing rotation fraction
    Returns:
        R_frac: fractional rotation matrix of shape [3, 3] or [B, 3, 3]
    """
    # 保证 batch 维度存在
    if R.ndim == 2:
        R = R.unsqueeze(0)  # [1, 3, 3]
        single = True
    else:
        single = False

    # 确保 fraction 是 tensor
    if not torch.is_tensor(fraction):
        fraction = torch.tensor(fraction, dtype=R.dtype, device=R.device)
    if fraction.ndim == 0:
        fraction = fraction.expand(R.shape[0])
    fraction = fraction.view(-1, 1, 1)  # broadcast shape [B, 1, 1]

    # trace 计算角度
    cos_theta = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # [B]

    # sin(theta) 可能接近 0，防止除零
    sin_theta = torch.sin(theta)
    sin_theta = torch.where(
        sin_theta.abs() < 1e-8, torch.ones_like(sin_theta), sin_theta
    )

    # 计算旋转轴 (反对称部分)
    w_hat = (R - R.transpose(-1, -2)) / (2 * sin_theta.view(-1, 1, 1))
    axis = torch.stack(
        [w_hat[:, 2, 1], w_hat[:, 0, 2], w_hat[:, 1, 0]], dim=-1
    )  # [B, 3]
    axis = F.normalize(axis, dim=-1)

    # 缩放角度
    theta_scaled = theta.view(-1, 1, 1) * fraction

    # 构造 K 矩阵 (batch)
    zeros = torch.zeros_like(axis[:, 0])
    K = torch.stack(
        [
            torch.stack([zeros, -axis[:, 2], axis[:, 1]], dim=-1),
            torch.stack([axis[:, 2], zeros, -axis[:, 0]], dim=-1),
            torch.stack([-axis[:, 1], axis[:, 0], zeros], dim=-1),
        ],
        dim=1,
    )  # [B, 3, 3]

    # 罗德里格斯公式
    I = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
    R_frac = I + torch.sin(theta_scaled) * K + (1 - torch.cos(theta_scaled)) * (K @ K)

    # 对于角度极小的情况直接返回单位矩阵
    small_angle = theta.abs() < 1e-6
    if small_angle.any():
        R_frac[small_angle] = torch.eye(3, device=R.device, dtype=R.dtype)

    return R_frac[0] if single else R_frac


@dataclass(frozen=True)
class DockingFlowFields:
    """Training fields for docking flow without mutating Molecule."""

    pos: torch.Tensor
    y: torch.Tensor
    u_t: torch.Tensor
    t: torch.Tensor
    R_pred: torch.Tensor
    shift: torch.Tensor


def compute_docking_flow_fields(
    ligand_pos: torch.Tensor,
    x0: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    ligand_center: torch.Tensor,
    shift: torch.Tensor,
    train_shift_range: float,
    terminal_time_threshold: float = 0.8,
    terminal_x0_noise_std: float = 2.0,
) -> DockingFlowFields:
    """Build docking-flow supervision tensors as a pure function.

    This mirrors Molecule.compute_flow_field_docking, but returns the runtime
    fields explicitly instead of writing them onto the Molecule object.
    """

    if t > terminal_time_threshold and terminal_x0_noise_std > 0:
        x0 = x0 + torch.randn_like(x0) * terminal_x0_noise_std

    stage_gate = torch.clamp(1 - t / 0.4, min=0.0, max=1.0)
    R_pred = rotation_fraction(R, stage_gate)
    normalized_shift = (shift * stage_gate / train_shift_range).unsqueeze(0)

    shift_frac = shift * stage_gate

    if t < 0.4:
        x0_aligned = (
            x0 @ rotation_fraction(R, 1 - stage_gate) + ligand_center - shift_frac
        )
    else:
        x0_aligned = (
            x0
            @ rotation_fraction(
                R,
                1 - stage_gate - torch.abs(torch.randn(1, device=x0.device) * 0.5),
            )
            + ligand_center
            - shift_frac
            + torch.randn(3, device=x0.device) * 1.0
        )

    x1_aligned = (
        (ligand_pos - ligand_center) @ rotation_fraction(R, stage_gate).T
        + ligand_center
        - shift_frac
    )

    x_t = (1.0 - t) * x0_aligned + t * x1_aligned
    u_t = x1_aligned - x_t

    return DockingFlowFields(
        pos=x_t,
        y=ligand_pos,
        u_t=u_t,
        t=t,
        R_pred=R_pred,
        shift=normalized_shift,
    )


def generate_chirality_edge_features(
    edge_index: torch.Tensor,
    chirality: torch.Tensor,
    chiral_neighbors: torch.Tensor,
    flip: bool = True,
) -> torch.Tensor:
    # 只需要计算一次就够了吧？

    """
    给定 chiral_neighbors 和 chirality，生成 edge-level 手性特征。

    参数:
        edge_index: (2, E), 全图的边
        chirality: (N,), 节点的手性标签 (0 = 非手性, >0 = 手性)
        chiral_neighbors: (n, 4), 每个手性中心的 4 个邻居
        flip: 如果 True，反转方向匹配 (即 nbr->center 而不是 center->nbr)

    返回:
        edge_feat: (E,), 每条边的标签
            - 0: 非手性
            - 1~4: 手性中心对应的4条边
    """
    E = edge_index.shape[1]
    edge_feat = torch.zeros(E, dtype=torch.long)

    edge_index_T = edge_index.t()  # (E, 2)

    # 遍历手性中心
    chiral_idx = torch.nonzero(chirality > 0, as_tuple=True)[0]

    for center_idx, nbrs in zip(chiral_idx.tolist(), chiral_neighbors.tolist()):
        for pos, nbr in enumerate(nbrs):  # pos: 0~3
            if flip:
                # 匹配 nbr -> center
                mask = (edge_index_T[:, 0] == nbr) & (edge_index_T[:, 1] == center_idx)
            else:
                # 匹配 center -> nbr
                mask = (edge_index_T[:, 0] == center_idx) & (edge_index_T[:, 1] == nbr)
            idx = torch.nonzero(mask, as_tuple=True)[0]
            if len(idx) > 0:
                edge_feat[idx] = pos + 1  # 编号 1~4

    return edge_feat


def generate_chirality_edge_features_vectorized(
    edge_index: torch.Tensor,
    chirality: torch.Tensor,
    chiral_neighbors: torch.Tensor,
    flip: bool = True,
) -> torch.Tensor:
    """
    Vectorized generation of chiral edge features for a ligand/molecule graph.
    Only considers 3 neighbors per chiral center.

    Args:
        edge_index: (2, E) tensor of graph edges
        chirality: (N,) tensor, non-zero indicates chiral centers
        chiral_neighbors: (M, 3) tensor of 3 neighbor indices for each chiral center
        flip: if True, edge direction is neighbor -> center; else center -> neighbor

    Returns:
        edge_feat: (E,) tensor, 0 if not chiral edge, else 1/2/3 indicating neighbor position
    """
    device = edge_index.device
    num_nodes = chirality.size(0)
    num_edges = edge_index.size(1)

    # Choose which end is center and which is neighbor
    center_col = 1 if flip else 0
    nbr_col = 0 if flip else 1
    edge_centers = edge_index[center_col]
    edge_nbrs = edge_index[nbr_col]

    # Expand chiral_neighbors to full node list
    full_chiral_neighbors = torch.zeros((num_nodes, 3), dtype=torch.long, device=device)
    full_chiral_neighbors[chirality > 0] = chiral_neighbors

    # Get 3 candidate neighbors for each edge's center
    candidates = full_chiral_neighbors[edge_centers]

    # Check if edge neighbor matches one of the 3 chiral neighbors
    match_mask = edge_nbrs.unsqueeze(1) == candidates
    has_match = match_mask.any(dim=1)

    # Get the position index (1,2,3) of the neighbor
    pos_indices = match_mask.long().argmax(dim=1) + 1

    # Only keep edges where center is chiral and neighbor matches
    final_mask = has_match & (chirality[edge_centers] > 0)
    edge_feat = torch.zeros(num_edges, dtype=torch.long, device=device)
    edge_feat[final_mask] = pos_indices[final_mask]

    return edge_feat


def pdb_without_waters(pdb_path: str):
    # 直接读取原始PDB文本
    with open(pdb_path, "r") as f:
        pdb_lines = f.readlines()

    # 过滤掉 HOH 或 WAT 的残基
    filtered_lines = [
        line
        for line in pdb_lines
        if not (line.startswith("HETATM") and "HOH" in line)  # 或 "WAT" 取决于命名
    ]

    # 拼接成新的 PDB block
    pdb_block = "".join(filtered_lines)

    return StringIO(pdb_block), pdb_block


def pdb_protein_atom_block(pdb_block: str):
    lines = [line for line in pdb_block.splitlines() if line.startswith("ATOM  ")]
    return "\n".join(lines) + ("\n" if lines else "")


def pdb_block_with_titlecase_elements(pdb_block: str):
    lines = []
    for line in pdb_block.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 78:
            element = line[76:78]
            stripped = element.strip()
            if len(stripped) == 2 and stripped.isupper():
                fixed = stripped.title().rjust(2)
                line = line[:76] + fixed + line[78:]
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def protein_to_graph_rdkit(
    pdb_file,
    type="cif",
):
    pdb_file, pdb_block = pdb_without_waters(pdb_file)

    if type == "cif":
        parser = Bio.PDB.MMCIFParser()
    else:
        parser = Bio.PDB.PDBParser()

    structure = parser.get_structure("X", pdb_file)
    model = structure[0]

    residues = [
        residue for chain in model.get_chains() for residue in chain.get_residues()
    ]

    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    protein_data = ProteinData(**rdkit_molecule_to_graph(mol))

    backbone_pos = []
    backbone_atom_types = []
    backbone_atom_residual_types = []
    seq = []
    residual_pos = []
    pdb_residues_index = []  # to keep track of the residue index in the pdb file

    # Clean the structure from non-standard residues
    for residue in residues:
        aa_id = seq1(residue.get_resname())
        if residue.get_id()[0] == " " and aa_id in aa_to_id.keys():
            for atom in residue:
                if atom.get_id() in ["N", "C", "O", "CA", "CB"]:
                    backbone_pos.append(atom.get_coord())
                    backbone_atom_types.append(
                        {"N": 0, "C": 1, "O": 2, "CA": 3, "CB": 4}[atom.get_id()]
                    )
                    backbone_atom_residual_types.append(aa_to_id[aa_id])

                if atom.get_id() == "CA":
                    seq.append(aa_to_id[aa_id])
                    pdb_residues_index.append(residue.get_id()[1])
                    residual_pos.append(torch.from_numpy(atom.get_coord()))
                    continue

    residual_pos = torch.tensor(np.array(residual_pos)).float()
    residual_node_types = torch.tensor(seq).long()

    assert residual_pos.shape[0] == residual_node_types.shape[0], (
        "The number of nodes and edge types do not match"
    )

    protein_data.residual_pos = residual_pos
    protein_data.residual_node_types = residual_node_types

    assert (
        len(backbone_pos)
        == len(backbone_atom_types)
        == len(backbone_atom_residual_types)
    ), "The number of backbone atoms and edge types do not match"

    protein_data.backbone_pos = torch.tensor(np.array(backbone_pos)).float()
    protein_data.backbone_atom_types = torch.tensor(backbone_atom_types)
    protein_data.backbone_atom_residual_types = torch.tensor(
        backbone_atom_residual_types
    )

    return protein_data


def protein_to_graph(
    pdb_file,
    type="cif",
):
    import cosmolkit as ck

    if type == "cif":
        protein = ck.Protein.read(str(pdb_file))
        structure = ck.BioStructure.read(str(pdb_file))
    else:
        _, pdb_block = pdb_without_waters(pdb_file)
        molecule_block = pdb_block_with_titlecase_elements(pdb_block)
        protein = ck.Protein.from_pdb(pdb_protein_atom_block(pdb_block))
        structure = ck.BioStructure.from_pdb(molecule_block)
    mol = structure.to_molecule(sanitize=False, remove_hs=False, proximity_bonding=True)

    protein_data = ProteinData(**molecule_to_graph(mol))
    protein_data.pos = torch.tensor(mol.coordinates_3d(), dtype=torch.float)

    backbone_pos = []
    backbone_atom_types = []
    backbone_atom_residual_types = []
    seq = []
    residual_pos = []

    for residue in protein.residues:
        aa_id = residue.one_letter_code
        if residue.is_standard and aa_id in aa_to_id.keys():
            for atom in residue.atoms:
                atom_name = atom.name.as_str().strip()
                if atom_name in ["N", "C", "O", "CA", "CB"]:
                    backbone_pos.append(atom.position)
                    backbone_atom_types.append(
                        {"N": 0, "C": 1, "O": 2, "CA": 3, "CB": 4}[atom_name]
                    )
                    backbone_atom_residual_types.append(aa_to_id[aa_id])

                if atom_name == "CA":
                    seq.append(aa_to_id[aa_id])
                    residual_pos.append(atom.position)
                    continue

    protein_data.residual_pos = torch.tensor(np.array(residual_pos)).float()
    protein_data.residual_node_types = torch.tensor(seq).long()

    assert (
        protein_data.residual_pos.shape[0] == protein_data.residual_node_types.shape[0]
    ), "The number of nodes and edge types do not match"

    assert (
        len(backbone_pos)
        == len(backbone_atom_types)
        == len(backbone_atom_residual_types)
    ), "The number of backbone atoms and edge types do not match"

    protein_data.backbone_pos = torch.tensor(np.array(backbone_pos)).float()
    protein_data.backbone_atom_types = torch.tensor(backbone_atom_types)
    protein_data.backbone_atom_residual_types = torch.tensor(
        backbone_atom_residual_types
    )

    return protein_data


def get_pocket_center(pocket_description):
    import cosmolkit as ck
    if pocket_description.endswith(".pdb"):
        _, block = pdb_without_waters(pocket_description)
        mol = ck.BioStructure.from_pdb(block).to_molecule(sanitize=False, remove_hs=False)
    elif pocket_description.endswith(".sdf"):
        mol = ck.Molecule.read_sdf(pocket_description, sanitize=False, remove_hs=False)
    elif pocket_description.endswith(".mol2"):
        mol = ck.Molecule.read_mol2(pocket_description, sanitize=False, remove_hs=False)
    else:
        raise ValueError(f"Unsupported pocket file format: {pocket_description}")
    positions = torch.tensor(mol.coordinates_3d(), dtype=torch.float)
    heavy = torch.tensor([a.atomic_number != 1 for a in mol.atoms])
    if not heavy.any():
        raise ValueError("Pocket reference contains no heavy atoms")
    return positions[heavy].mean(dim=0)


def keep_fields(data, fields):
    for key in list(data.keys()):
        if key not in fields:
            delattr(data, key)
    return data


def remove_fields(data, fields):
    for key in fields:
        if hasattr(data, key):
            delattr(data, key)
    return data


class ProteinData(Data):
    def update_graph(self, radius_cutoff):
        batch = self.batch if self.batch is not None else None
        print(batch)

        self.edge_index = radius_graph(
            self.pos, r=radius_cutoff, loop=False, batch=batch
        )
        self.edge_types = protein_edge_types(self.edge_index)
        self.edge_vector = self.pos[self.edge_index[0]] - self.pos[self.edge_index[1]]

        return self

    def extract_pocket_all_atoms(
        self, ligand_positions: torch.Tensor, cutoff: float = 10.0
    ):
        """
        Extract pocket atoms within `cutoff` Å of any ligand atom.

        Args:
            ligand_positions (torch.Tensor): shape [N_ligand, 3], 3D coordinates of ligand atoms
            cutoff (float): distance threshold (Å)

        Returns:
            pocket_data (Data): new Data object containing pocket atoms & edges
        """
        # 1. Compute distances from each pocket atom to each ligand atom
        # self.pos: [N_pocket, 3], ligand_positions: [N_ligand, 3]
        diff = self.pos.unsqueeze(1) - ligand_positions.unsqueeze(
            0
        )  # [N_pocket, N_ligand, 3]
        dist2 = torch.norm(diff, dim=-1)  # [N_pocket, N_ligand]
        cutoff = 3.0

        # 2. Keep atoms that are within cutoff to ANY ligand atom
        mask = torch.any(dist2 <= cutoff**2, dim=1)
        keep_indices = torch.nonzero(mask, as_tuple=False).view(-1)

        # if keep_indices.numel() == 0:
        #     raise ValueError("No atoms found within cutoff of any ligand atom — check ligand positions or cutoff.")

        # 3. Filter edge_index to only keep edges within the pocket
        edge_index, _, edge_mask = subgraph(
            subset=keep_indices,
            edge_index=self.edge_index,
            num_nodes=self.pos.size(0),
            relabel_nodes=True,
            return_edge_mask=True,
        )

        # 4. Create a new Data object (or modify in-place)
        self.pos = self.pos[keep_indices]
        self.node_types = self.node_types[keep_indices]
        self.edge_index = edge_index
        self.edge_types = self.edge_types[edge_mask]

        return self

    def remove_hydrogens(self):
        keep_nodes = self.x[:, 0] != 1

        # 提取子图（节点、边、边属性）
        edge_index_sub, edge_types_sub, edge_mask = subgraph(
            subset=keep_nodes,
            edge_index=self.chem_bond_index,
            edge_attr=self.chem_bond_attr,
            num_nodes=self.pos.size(0),
            relabel_nodes=True,
            return_edge_mask=True,
        )

        self.x = self.x[keep_nodes]
        self.pos = self.pos[keep_nodes]
        self.chem_bond_index = edge_index_sub
        self.chem_bond_attr = edge_types_sub

    def to_residue_graph(self):
        self.pos = self.residual_pos
        self.node_types = self.residual_node_types + 100

        num_nodes = self.pos.size(0)

        # 顺序连接 n <-> n+1
        src = torch.arange(num_nodes - 1, dtype=torch.long)
        dst = src + 1

        # 双向 edge index
        edge_index = torch.stack(
            [
                torch.cat([src, dst]),  # from
                torch.cat([dst, src]),  # to
            ],
            dim=0,
        )

        # edge types
        edge_types = torch.cat(
            [
                torch.full((num_nodes - 1,), 5, dtype=torch.long),  # n -> n+1
                torch.full((num_nodes - 1,), 6, dtype=torch.long),  # n+1 -> n
            ]
        )

        self.edge_index = edge_index
        self.edge_types = edge_types

        return self

    def to_backbone_graph(self):
        # ["N", "C", "O", "CA", "CB"]
        mask = (self.backbone_atom_types == 3) | (self.backbone_atom_types == 4)
        # mask = (self.backbone_atom_types == 3) | (self.backbone_atom_types == 4) | (self.backbone_atom_types == 2)
        # mask = (self.backbone_atom_types == 3) | (self.backbone_atom_types == 4) | (self.backbone_atom_types == 1)

        pos = self.backbone_pos[mask]
        node_types_atom = self.backbone_atom_types[mask]
        node_types_residue = self.backbone_atom_residual_types[mask]
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_types = torch.tensor([], dtype=torch.long)

        return ProteinBackboneData(
            pos=pos,
            node_types_atom=node_types_atom,
            node_types_residue=node_types_residue,
            edge_index=edge_index,
            edge_types=edge_types,
        )

    def prepare_pocket_ranker_graph(
        self, centers, cutoff, gt_pockets=None, max_centers=9999,
        *, inplace=True, device=None, max_num_neighbors=1024,
    ):
        """Build the pocket scoring graph; inplace=False preserves the protein.

        The returned graph includes edge vectors and edge types. Without reference
        pockets, y is a zero placeholder for inference.
        """
        device = self.pos.device if device is None else device
        pos = self.pos.to(device, non_blocking=True)
        x = self.x.to(device, non_blocking=True)
        centers = centers.to(device, non_blocking=True)
        result = self if inplace else ProteinData()

        if centers.size(0) > max_centers:
            idx = torch.randperm(centers.size(0), device=centers.device)[:max_centers]
            centers = centers[idx]

        device = pos.device

        # --------------------------------
        # 1 radius search
        # --------------------------------

        edge = radius(x=pos, y=centers, r=cutoff, max_num_neighbors=max_num_neighbors)

        atom_idx = edge[0]
        pocket_idx = edge[1]

        E = atom_idx.size(0)
        K = centers.size(0)

        # --------------------------------
        # 2 构建 atom copy nodes
        # --------------------------------

        pocket_atom_pos = pos[pocket_idx]
        pocket_atom_x = x[pocket_idx]

        # --------------------------------
        # 3 center nodes
        # --------------------------------

        center_pos = centers
        center_attr = torch.zeros((K, x.shape[1]), device=device, dtype=x.dtype)
        center_attr[:, 0] = NUM_ATOM_TYPES

        # --------------------------------
        # 4 合并节点
        # --------------------------------
        #
        new_pos = torch.cat([pocket_atom_pos, center_pos], dim=0)
        new_x = torch.cat([pocket_atom_x, center_attr], dim=0)

        # --------------------------------
        # 5 构建边
        # --------------------------------

        atom_nodes = torch.arange(E, device=device)
        center_nodes = E + atom_idx

        edge_src = torch.cat([atom_nodes, center_nodes])
        edge_dst = torch.cat([center_nodes, atom_nodes])

        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        result.pos = new_pos
        result.x = new_x
        result.edge_index = edge_index
        result.edge_vector = new_pos[edge_index[0]] - new_pos[edge_index[1]]
        result.edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        result.y = torch.zeros(K, dtype=torch.float32, device=device)

        if gt_pockets is None:
            return result

        # (N, L, 3)
        gt_pockets = gt_pockets.to(device, non_blocking=True)
        diff = centers.unsqueeze(1) - gt_pockets.unsqueeze(0)

        # (N, L)
        dists = diff.norm(dim=2)

        # 找最近 ligand
        min_dist, min_idx = dists.min(dim=1)  # (N,), (N,)

        # 取对应的最近 ligand center
        nearest_centers = gt_pockets[min_idx]  # (N, 3)
        # === 计算 score ===
        score = 1 - torch.tanh((centers - nearest_centers).norm(dim=1) / 10)

        result.y = score
        return result


class ProteinBackboneData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_pocket_vec_multiple_ligands(self, ligand_centers):
        """
        ligand_centers: Tensor of shape (L, 3)
        """

        CB_mask = self.node_types_atom == 4

        # (N, L, 3)
        diff = self.pos.unsqueeze(1)[CB_mask] - ligand_centers.unsqueeze(0)

        # (N, L)
        dists = diff.norm(dim=2)

        # 找最近 ligand
        min_dist, min_idx = dists.min(dim=1)  # (N,), (N,)

        # 取对应的最近 ligand center
        nearest_centers = ligand_centers[min_idx]  # (N, 3)

        # === 计算 pocket_vec ===
        dist = min_dist  # (N,)
        direction = nearest_centers - self.pos[CB_mask]  # (N, 3)

        pocket_vec = direction * (dist.clamp(max=10) / dist).unsqueeze(1)

        # === 计算 score ===
        score = 1 - torch.tanh(
            (self.pos[CB_mask] - nearest_centers + pocket_vec).norm(dim=1) / 10
        )

        self.y = score
        self.pocket_vec = pocket_vec

        return self

    def update_graph(self, radius_cutoff, radius_min=0.0, scale=1.0):
        """
        Return a copied object with updated radius-range graph.

        Keeps edges whose distance satisfies:

            radius_min < distance <= radius_cutoff

        Args:
            radius_cutoff: outer radius, r_max
            radius_min: inner radius, r_min. Default 0.0 means normal radius_graph.
        """
        self = copy.copy(self)

        batch = self.batch if self.batch is not None else None

        edge_index = radius_graph(
            self.pos,
            r=radius_cutoff,
            loop=False,
            batch=batch,
            max_num_neighbors=64,
        )

        src, dst = edge_index

        edge_vector = self.pos[src] - self.pos[dst]
        edge_dist = edge_vector.norm(dim=-1)

        if radius_min is not None and radius_min > 0:
            mask = edge_dist > radius_min
            edge_index = edge_index[:, mask]
            edge_vector = edge_vector[mask]

        self.edge_index = edge_index
        self.edge_vector = edge_vector * scale

        self.edge_types = torch.zeros(
            self.edge_index.shape[1],
            dtype=torch.long,
            device=self.pos.device,
        )

        self.CB_mask = self.node_types_atom == 4

        return self


class ProteinPocketData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.edge_index is not None:
            self.edge_index_origin = self.edge_index
            self.edge_attr_origin = self.edge_types

    @staticmethod
    def from_pair(mol_data: Molecule, pocket_data: ProteinData):
        num_mol_atoms = mol_data.pos.size(0)
        node_types = torch.cat([mol_data.node_types, pocket_data.node_types], dim=0)
        pos = torch.cat([mol_data.pos, pocket_data.pos], dim=0)
        edge_index = torch.cat(
            [mol_data.edge_index, pocket_data.edge_index + num_mol_atoms], dim=1
        )
        edge_types = torch.cat([mol_data.edge_types, pocket_data.edge_types], dim=0)

        ligand_mask = torch.zeros(node_types.size(0), dtype=torch.bool)
        ligand_mask[:num_mol_atoms] = True

        data = ProteinPocketData(
            pos=pos,
            edge_index=edge_index,
            edge_types=edge_types,
            node_types=node_types,
            ligand_mask=ligand_mask,
            y=mol_data.y,
            R=mol_data.R.unsqueeze(0) if hasattr(mol_data, "R") else None,
            u_t=mol_data.u_t if hasattr(mol_data, "u_t") else None,
        )

        return data

    def update_graph(self, radius_cutoff, t, stage=None):
        self.edge_index, self.edge_types = radius_graph_with_edge_attr_for_ligand_batch(
            self.pos,
            self.edge_index_origin,
            edge_attr=self.edge_attr_origin,
            radius_cutoff=radius_cutoff,
            ligand_mask=self.ligand_mask,
            batch=self.batch,
            loop=False,
        )

        if stage is not None:
            t = (t / 2) + stage * 0.5

        self.t = torch.full_like(self.node_types, t, dtype=torch.float)

        if hasattr(self, "chirality"):
            self.edge_chirality_feat = generate_chirality_edge_features(
                self.edge_index, self.chirality, self.chiral_neighbors
            )
        self.edge_vector = self.pos[self.edge_index[0]] - self.pos[self.edge_index[1]]

        self.to(self.pos.device)

        return self


SENTINEL = torch.iinfo(torch.long).min


class CPIdata(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if self.edge_index is not None:
        #     self.chem_bond_index = self.edge_index
        #     self.chem_bond_attr = self.edge_types

    def __inc__(self, key, value, *args, **kwargs):
        if key in ["chiral_neighbors"]:
            return self.pos.size(0)
        if key in ["chiral_neighbors_ligand"]:
            return self.ligand_mask.sum()
        return super().__inc__(key, value, *args, **kwargs)

    @staticmethod
    def from_pair(
        mol_data: Molecule,
        protein_data: ProteinData,
        mol_pos=None,
        y=None,
        R=None,
        R_pred=None,
        u_t=None,
        shift=None,
    ):
        mol_pos = mol_data.pos if mol_pos is None else mol_pos
        num_mol_atoms = mol_pos.size(0)
        pos = torch.cat([mol_pos, protein_data.pos], dim=0)
        chem_bond_index = torch.cat(
            [mol_data.chem_bond_index, protein_data.chem_bond_index + num_mol_atoms],
            dim=1,
        )
        chem_bond_attr = torch.cat(
            [mol_data.chem_bond_attr, protein_data.chem_bond_attr], dim=0
        )
        x = torch.cat([mol_data.x, protein_data.x], dim=0)

        ligand_mask = torch.zeros(pos.size(0), dtype=torch.bool, device=pos.device)
        ligand_mask[:num_mol_atoms] = True

        if hasattr(mol_data, "chirality"):
            chiral_neighbors_ligand = mol_data.chiral_neighbors
            chirality_ligand = mol_data.chirality
            chirality = torch.cat(
                [
                    mol_data.chirality,
                    torch.zeros(protein_data.pos.size(0), dtype=torch.long).to(
                        mol_data.x.device
                    ),
                ],
                dim=0,
            )
            chiral_neighbors = mol_data.chiral_neighbors
            chiral_neighbors = torch.where(
                chiral_neighbors == -1,
                torch.full_like(chiral_neighbors, SENTINEL),
                chiral_neighbors,
            )
            chiral_neighbors_ligand = torch.where(
                chiral_neighbors_ligand == -1,
                torch.full_like(chiral_neighbors_ligand, SENTINEL),
                chiral_neighbors_ligand,
            )
        else:
            chirality = torch.zeros_like(x[:, 0]).to(mol_data.pos.device)
            chiral_neighbors = (
                torch.tensor([], dtype=torch.long).view(0, 4).to(mol_data.pos.device)
            )
            chiral_neighbors_ligand = chiral_neighbors
            chirality_ligand = torch.zeros_like(mol_data.x[:, 0])

        if y is None:
            y = mol_data.y if hasattr(mol_data, "y") else mol_data.pos
        if R is None:
            R = mol_data.R.unsqueeze(0) if hasattr(mol_data, "R") else None
        elif R.dim() == 2:
            R = R.unsqueeze(0)
        if R_pred is None:
            R_pred = (
                mol_data.R_pred.unsqueeze(0) if hasattr(mol_data, "R_pred") else None
            )
        elif R_pred.dim() == 2:
            R_pred = R_pred.unsqueeze(0)
        if u_t is None:
            u_t = mol_data.u_t if hasattr(mol_data, "u_t") else None
        if shift is None:
            shift = mol_data.shift if hasattr(mol_data, "shift") else None

        data = CPIdata(
            pos=pos,
            chem_bond_index=chem_bond_index,
            x=x,
            chem_bond_attr=chem_bond_attr,
            ligand_mask=ligand_mask,
            y=y,
            R=R,
            R_pred=R_pred,
            u_t=u_t,
            shift=shift,
            chirality=chirality,
            chirality_ligand=chirality_ligand,
            chiral_neighbors=chiral_neighbors,
            chiral_neighbors_ligand=chiral_neighbors_ligand,
            pdb_id=protein_data.pdb_id if hasattr(protein_data, "pdb_id") else None,
        )

        return data

    def update_ligand_pos(self, pos):
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos, dtype=self.pos.dtype, device=self.pos.device)
        else:
            pos = pos.to(device=self.pos.device, dtype=self.pos.dtype)
        expected_shape = self.pos[self.ligand_mask].shape
        if tuple(pos.shape) != tuple(expected_shape):
            raise ValueError(
                f"ligand pos shape mismatch: got {tuple(pos.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        self.pos[self.ligand_mask] = pos
        return self

    def extract_pocket_and_update_graph(
        self,
        cutoff: float = 3.0,
        t=None,
        compute_edge_mindist=False,
        max_num_neighbors=9999,
    ):
        assert hasattr(self, "ligand_mask"), "ligand_mask required."

        device = self.pos.device
        batch = getattr(self, "batch", None)
        ligand_mask = self.ligand_mask

        mol_idx_all = torch.where(ligand_mask)[0]
        pocket_idx_all = torch.where(~ligand_mask)[0]

        # -------------------------
        # Step 1: ligand-protein radius
        # radius(x=protein, y=ligand)
        # edge_index_lp_raw[0]: ligand local index
        # edge_index_lp_raw[1]: protein local index
        # -------------------------
        if batch is not None:
            edge_index_lp_raw = radius(
                x=self.pos[~ligand_mask],
                y=self.pos[ligand_mask],
                r=cutoff,
                batch_x=batch[~ligand_mask],
                batch_y=batch[ligand_mask],
                max_num_neighbors=max_num_neighbors,
            )
        else:
            edge_index_lp_raw = radius(
                x=self.pos[~ligand_mask],
                y=self.pos[ligand_mask],
                r=cutoff,
                max_num_neighbors=max_num_neighbors,
            )

        protein_indices = pocket_idx_all[edge_index_lp_raw[1].unique()]
        ligand_indices = mol_idx_all

        # Keep sorted order for result consistency.
        keep_indices = torch.cat([protein_indices, ligand_indices]).unique().sort()[0]

        # -------------------------
        # Step 2: ligand-pocket bidirectional edges
        # -------------------------
        src_l = mol_idx_all[edge_index_lp_raw[0]]
        dst_p = pocket_idx_all[edge_index_lp_raw[1]]

        edge_index_lp = torch.cat(
            [
                torch.stack([src_l, dst_p], dim=0),
                torch.stack([dst_p, src_l], dim=0),
            ],
            dim=1,
        )

        # -------------------------
        # Step 3: ligand-ligand radius edges
        # -------------------------
        pos_mol = self.pos[ligand_mask]
        batch_mol = batch[ligand_mask] if batch is not None else None

        edge_index_ll_local = radius_graph(
            pos_mol, r=cutoff, batch=batch_mol, loop=False, max_num_neighbors=999
        )
        edge_index_ll = mol_idx_all[edge_index_ll_local]

        # -------------------------
        # Step 4: merge chemical bonds + radius edges
        # -------------------------
        all_edge_index = torch.cat(
            [self.chem_bond_index, edge_index_lp, edge_index_ll],
            dim=1,
        )

        num_chem_edges = self.chem_bond_index.size(1)
        num_radius_edges = all_edge_index.size(1) - num_chem_edges

        radius_edge_attr = torch.full(
            (num_radius_edges, 1),
            RADIUS_EDGE_TYPE,
            dtype=self.chem_bond_attr.dtype,
            device=self.chem_bond_attr.device,
        )

        all_edge_attr = torch.cat(
            [self.chem_bond_attr, radius_edge_attr],
            dim=0,
        )

        # Chemical bonds win if bond edge type < radius edge type.
        edge_index, edge_attr = dedup_edges_prefer_bond(
            all_edge_index,
            all_edge_attr,
            sort_by_row=False,
        )

        # -------------------------
        # Step 5: manual subgraph + relabel
        # equivalent to subgraph(..., relabel_nodes=True)
        # -------------------------
        mapping = torch.full(
            (self.pos.size(0),),
            -1,
            dtype=torch.long,
            device=device,
        )
        mapping[keep_indices] = torch.arange(
            keep_indices.numel(),
            dtype=torch.long,
            device=device,
        )

        src_new = mapping[edge_index[0]]
        dst_new = mapping[edge_index[1]]

        edge_mask = (src_new >= 0) & (dst_new >= 0)

        edge_index = torch.stack(
            [src_new[edge_mask], dst_new[edge_mask]],
            dim=0,
        )
        edge_attr = edge_attr[edge_mask]

        # -------------------------
        # Step 6: update node features
        # -------------------------
        self.pos = self.pos[keep_indices]
        self.x = self.x[keep_indices]
        self.ligand_mask = self.ligand_mask[keep_indices]

        if hasattr(self, "batch") and self.batch is not None:
            self.batch = self.batch[keep_indices]

        if hasattr(self, "chirality"):
            self.chirality = self.chirality[keep_indices]
            self.chiral_neighbors = map_index_with_default(
                self.chiral_neighbors.view(-1),
                keep_indices,
                default=-1,
            )[0].view(-1, 3)

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # -------------------------
        # Step 7: optional features
        # -------------------------
        if compute_edge_mindist:
            self.edge_mindist = compute_pb_min_distance_per_edge(
                self.pos,
                self.edge_index,
                self.edge_attr[:, 0],
                self.x[:, 0],
                self.ligand_mask,
                clash_cutoff=0.77,
            )

        if t is not None:
            t_value = t.item() if hasattr(t, "item") else t
            self.t = torch.full(
                (self.x.size(0),),
                t_value,
                dtype=torch.float,
                device=device,
            )

        if hasattr(self, "chirality"):
            self.edge_chirality_feat = generate_chirality_edge_features_vectorized(
                self.edge_index,
                self.chirality,
                self.chiral_neighbors,
            )

        self.edge_vector = self.pos[self.edge_index[0]] - self.pos[self.edge_index[1]]

        return self


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

# Function: compute_pb_min_distance_per_edge
# Purpose: Calculate the minimum allowed distance between atoms in a molecule
#          based on the Pauling Bond (PB) theory.


def compute_pb_min_distance_per_edge(
    pos: torch.Tensor,  # (N, 3) tensor representing the positions of atoms
    edge_index: torch.Tensor,  # (2, E) tensor representing the edges between atoms
    edge_attr: torch.Tensor,  # (E,) tensor representing the attributes of edges
    node_types: torch.Tensor,  # (N,) tensor representing the types of atoms
    ligand_mask: torch.Tensor,  # (N,) tensor representing the mask of ligand atoms
    clash_cutoff: float = 0.75,  # clash cutoff value (default: 0.75)
):
    """
    Returns:
        edge_min_distance: (E,) tensor representing the minimum allowed distance
            between atoms for each edge.
    """

    # Get the device (GPU or CPU) where the tensors are stored
    device = edge_index.device

    # Get the number of edges
    E = edge_index.size(1)

    # Initialize the edge_min_distance tensor with zeros
    edge_min_distance = torch.zeros(E, device=device)

    # Find the radius edges (edges with attribute RADIUS_EDGE_TYPE)
    radius_edge_mask = edge_attr == RADIUS_EDGE_TYPE
    if not radius_edge_mask.any():
        # If no radius edges are found, return the edge_min_distance tensor
        return edge_min_distance

    # Get the source and destination nodes for the radius edges
    src = edge_index[0, radius_edge_mask]
    dst = edge_index[1, radius_edge_mask]

    # Filter the edges to only include ligand-protein interactions
    ligand_to_protein = ligand_mask[src] & (~ligand_mask[dst])
    protein_to_ligand = (~ligand_mask[src]) & ligand_mask[dst]
    lp_mask = ligand_to_protein | protein_to_ligand
    if not lp_mask.any():
        # If no ligand-protein interactions are found, return the edge_min_distance tensor
        return edge_min_distance

    # Get the source and destination nodes for the ligand-protein interactions
    src_lp = src[lp_mask]
    dst_lp = dst[lp_mask]

    # Get the atomic numbers of the source and destination nodes
    z_src = node_types[src_lp]
    z_dst = node_types[dst_lp]

    # Ignore edges involving hydrogen atoms
    non_h_mask = (z_src != 1) & (z_dst != 1)
    if not non_h_mask.any():
        # If no non-hydrogen edges are found, return the edge_min_distance tensor
        return edge_min_distance

    # Get the source and destination nodes for the non-hydrogen edges
    src_lp = src_lp[non_h_mask]
    dst_lp = dst_lp[non_h_mask]

    # Define a function to look up the van der Waals radius for each atomic number
    def lookup(z):
        r = torch.empty_like(z, dtype=torch.float, device=device)
        for i, zi in enumerate(z.tolist()):
            r[i] = VDW_RADII.get(int(zi), 0)
        return r

    # Look up the van der Waals radii for the source and destination nodes
    r_src = lookup(node_types[src_lp])
    r_dst = lookup(node_types[dst_lp])

    # Calculate the minimum allowed distance between the atoms
    min_dist = clash_cutoff * (r_src + r_dst)

    # Write the minimum allowed distance to the edge_min_distance tensor
    idx_radius = torch.where(radius_edge_mask)[0]
    idx_lp = idx_radius[lp_mask]
    idx_valid = idx_lp[non_h_mask]
    edge_min_distance[idx_valid] = min_dist

    # Return the edge_min_distance tensor
    return edge_min_distance


def graph_to_sdf_v3000(
    pos,
    node_types,
    edge_index,
    edge_types,
    atom_map=None,
    sdf_path="output_v3000.sdf",
    dummy_node_pos=None,
    dummy_node_weight=None,
):
    """
    Build a V3000 SDF from graph data, optionally with weighted dummy/virtual points.

    Real atoms are written as normal V3000 atoms.
    Dummy nodes are written as '*' atoms for better CTFile/V3000 compatibility.
    Dummy weights are stored in an SDF data field <DUMMY_WEIGHTS>, one value per line,
    matching the dummy atom order.

    Dummy nodes do not participate in bonds.
    """

    from rdkit import Chem

    # ---- 1. Move to CPU ----
    pos = pos.detach().cpu()
    node_types = node_types.detach().cpu()
    edge_index = edge_index.detach().cpu()
    edge_types = edge_types.detach().cpu()

    if atom_map is not None:
        atom_map = atom_map.detach().cpu()

    if dummy_node_pos is not None:
        dummy_node_pos = dummy_node_pos.detach().cpu()
        if dummy_node_weight is None:
            raise ValueError(
                "dummy_node_weight must not be None when dummy_node_pos is provided"
            )
        dummy_node_weight = dummy_node_weight.detach().cpu()

    # ---- 2. Basic shape checks ----
    if pos.ndim != 2 or pos.size(1) != 3:
        raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")

    if node_types.ndim != 1:
        node_types = node_types.reshape(-1)

    if node_types.size(0) != pos.size(0):
        raise ValueError(
            f"node_types size mismatch: got {node_types.size(0)}, expected {pos.size(0)}"
        )

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        )

    if edge_types.ndim != 1:
        edge_types = edge_types.reshape(-1)

    if edge_types.size(0) != edge_index.size(1):
        raise ValueError(
            f"edge_types size mismatch: got {edge_types.size(0)}, expected {edge_index.size(1)}"
        )

    if atom_map is not None:
        if atom_map.ndim != 1:
            atom_map = atom_map.reshape(-1)
        if atom_map.size(0) != pos.size(0):
            raise ValueError(
                f"atom_map size mismatch: got {atom_map.size(0)}, expected {pos.size(0)}"
            )

    pt = Chem.GetPeriodicTable()

    # ---- 3. Atom construction ----
    atom_lines = []
    node_to_v3k = []
    dummy_v3k_indices = []
    dummy_weight_lines = []

    v3k_idx = 1

    # ---- real atoms ----
    for i in range(node_types.size(0)):
        at_num = int(node_types[i])

        if at_num <= 0:
            symbol = "*"
        elif at_num <= 118:
            symbol = pt.GetElementSymbol(at_num)
        else:
            symbol = "*"

        x, y, z = pos[i].tolist()

        atom_map_value = int(atom_map[i]) if atom_map is not None else 0

        atom_lines.append(
            f"M  V30 {v3k_idx} {symbol} {x:.4f} {y:.4f} {z:.4f} {atom_map_value}"
        )

        node_to_v3k.append(v3k_idx)
        v3k_idx += 1

    # ---- dummy / virtual atoms ----
    if dummy_node_pos is not None:
        if dummy_node_pos.ndim != 2 or dummy_node_pos.size(1) != 3:
            raise ValueError(
                f"dummy_node_pos must have shape (K, 3), got {tuple(dummy_node_pos.shape)}"
            )

        if dummy_node_weight.ndim != 1:
            dummy_node_weight = dummy_node_weight.reshape(-1)

        K = dummy_node_pos.size(0)

        if dummy_node_weight.size(0) != K:
            raise ValueError(
                f"dummy_node_weight size mismatch: "
                f"got {dummy_node_weight.size(0)}, expected {K}"
            )

        for i in range(K):
            x, y, z = dummy_node_pos[i].tolist()
            w = float(dummy_node_weight[i])

            # Use '*' instead of 'Du' for broader V3000 compatibility.
            atom_lines.append(f"M  V30 {v3k_idx} * {x:.4f} {y:.4f} {z:.4f} 0")

            dummy_v3k_indices.append(v3k_idx)
            dummy_weight_lines.append(f"{w:.6f}")

            v3k_idx += 1

    num_atoms = len(atom_lines)

    # ---- 4. Bonds: only original real graph ----
    bond_lines = []
    v3k_bond_idx = 1

    # V3000 bond types:
    # 1 = single, 2 = double, 3 = triple, 4 = aromatic
    bond_type_dict = {
        1: 1,
        2: 2,
        3: 3,
        12: 4,
    }

    E = edge_index.size(1)
    num_real_atoms = len(node_to_v3k)

    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])
        t = int(edge_types[e])

        # Skip duplicated undirected edge.
        if u >= v:
            continue

        # Skip invalid or dummy-related edges.
        # This function assumes edge_index only refers to real atoms.
        if u < 0 or v < 0 or u >= num_real_atoms or v >= num_real_atoms:
            continue

        b_type = bond_type_dict.get(t)

        # Skip unknown bond types rather than writing invalid V3000 bond type.
        if b_type is None:
            continue

        bond_lines.append(
            f"M  V30 {v3k_bond_idx} {b_type} {node_to_v3k[u]} {node_to_v3k[v]}"
        )

        v3k_bond_idx += 1

    num_bonds = len(bond_lines)

    # ---- 5. Assemble V3000 SDF ----
    lines = [
        "GraphWithDummyPoints",
        "  COSMolKit",
        "",
        "  0  0  0     0  0            999 V3000",
        "M  V30 BEGIN CTAB",
        f"M  V30 COUNTS {num_atoms} {num_bonds} 0 0 0",
        "M  V30 BEGIN ATOM",
        *atom_lines,
        "M  V30 END ATOM",
    ]

    if num_bonds > 0:
        lines.extend(
            [
                "M  V30 BEGIN BOND",
                *bond_lines,
                "M  V30 END BOND",
            ]
        )

    lines.extend(
        [
            "M  V30 END CTAB",
            "M  END",
        ]
    )

    # ---- 6. SDF data fields for dummy metadata ----
    if dummy_node_pos is not None and len(dummy_weight_lines) > 0:
        lines.extend(
            [
                ">  <DUMMY_ATOM_INDICES>",
                *[str(i) for i in dummy_v3k_indices],
                "",
                ">  <DUMMY_WEIGHTS>",
                *dummy_weight_lines,
                "",
            ]
        )

    lines.append("$$$$")

    sdf_str = "\n".join(lines) + "\n"

    with open(sdf_path, "w", encoding="utf-8") as f:
        f.write(sdf_str)

    return sdf_str


def axis_angle_to_matrix(axis, theta):
    """
    axis: (B, 3, 1) rotation axis
    theta: (B, 1) rotation angle (radians)
    return: (B, 3, 3) rotation matrix
    """
    axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)  # normalize axis
    x, y, z = axis[:, 0, 0], axis[:, 1, 0], axis[:, 2, 0]

    # build skew-symmetric matrix [r]_x
    K = torch.zeros(axis.size(0), 3, 3, device=axis.device)
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x

    I = torch.eye(3, device=axis.device).unsqueeze(0).expand_as(K)
    sin_t = torch.sin(theta).view(-1, 1, 1)
    cos_t = torch.cos(theta).view(-1, 1, 1)

    R = I + sin_t * K + (1 - cos_t) * (K @ K)
    return R


def dedup_edges_prefer_bond(
    edge_index,
    edge_attr,
    num_nodes: int = None,
    sort_by_row: bool = True,
):
    """
    Mimic coalesce ordering exactly, but for duplicate (src, dst) edges,
    keep the whole edge whose edge_attr[:, 0] is minimal.

    Args:
        edge_index: LongTensor [2, E]
        edge_attr:  Tensor [E, F]
        num_nodes:  optional
        sort_by_row:
            True  -> sort like coalesce(..., sort_by_row=True)
            False -> sort like coalesce(..., sort_by_row=False)

    Returns:
        edge_index_new, edge_attr_new
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    assert edge_attr.dim() >= 1
    assert edge_index.size(1) == edge_attr.size(0)

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    row = edge_index[0]
    col = edge_index[1]

    # ---- exactly the same pair-order key as coalesce ----
    primary = edge_index[
        1 - int(sort_by_row)
    ]  # col if sort_by_row=True, row if False? wait
    secondary = edge_index[int(sort_by_row)]

    # Actually matching coalesce source:
    # idx = edge_index[1 - int(sort_by_row)] * num_nodes + edge_index[int(sort_by_row)]
    pair_key = primary * num_nodes + secondary

    # priority inside duplicate pair: smaller edge_attr[:,0] wins
    edge_type = edge_attr[:, 0].long()

    # stable lexicographic behavior:
    # first sort by edge_type, then stable-sort by pair_key
    perm = torch.argsort(edge_type, stable=True)
    pair_key_1 = pair_key[perm]

    perm2 = torch.argsort(pair_key_1, stable=True)
    perm = perm[perm2]

    edge_index_sorted = edge_index[:, perm]
    edge_attr_sorted = edge_attr[perm]
    pair_key_sorted = pair_key[perm]

    # keep first occurrence in each duplicate group
    keep = torch.ones(
        pair_key_sorted.numel(), dtype=torch.bool, device=edge_index.device
    )
    keep[1:] = pair_key_sorted[1:] != pair_key_sorted[:-1]

    return edge_index_sorted[:, keep], edge_attr_sorted[keep]
