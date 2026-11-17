"""Read prepared V5 pose-ranker training records from LMDB."""

import copy
import os
import pickle
import sys
from typing import List

import lmdb
import torch
from torch_geometric.data import Batch, Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.utils import CPIdata, Molecule, ProteinData, keep_fields


def sample_rmsd_stratified_2233(rmsds):
    """Draw up to ten unique poses: <=1, (1,2], (2,5], >5 A.

    Quotas are 2/2/3/3. Fill shortages uniformly from remaining valid
    poses, without replacement. Uses the worker's seeded torch RNG.
    """
    valid = torch.isfinite(rmsds) & (rmsds >= 0)
    masks = [rmsds <= 1, (rmsds > 1) & (rmsds <= 2),
             (rmsds > 2) & (rmsds <= 5), rmsds > 5]
    selected = torch.zeros_like(valid)
    parts = []
    for mask, quota in zip(masks, (2, 2, 3, 3)):
        indices = torch.where(valid & mask)[0]
        indices = indices[torch.randperm(indices.numel(), device=indices.device)[:quota]]
        parts.append(indices)
        selected[indices] = True
    chosen = torch.cat(parts)
    remaining = torch.where(valid & ~selected)[0]
    fill = remaining[torch.randperm(remaining.numel(), device=remaining.device)[:10-chosen.numel()]]
    chosen = torch.cat([chosen, fill])
    return chosen[torch.randperm(chosen.numel(), device=chosen.device)]


class PDBbind_score(Dataset):
    def __init__(
        self,
        root,
        split="train",
        radius_cutoff=4.0,
        excluded_indices=None,
        pose_sampling="uniform",
    ) -> None:
        assert split in ["train", "valid", "test"], (
            f"Unsupported split: {split}. Must be 'train', 'valid', or 'test'."
        )
        self.split = split
        self.radius_cutoff = radius_cutoff
        if pose_sampling not in ("uniform", "rmsd_stratified_2233"):
            raise ValueError(f"Unknown pose sampling policy: {pose_sampling}")
        self.pose_sampling = pose_sampling if split == "train" else "uniform"

        print(f"Loading PDBbind pose ranker {split} dataset.")
        super().__init__(root)

        env = lmdb.open(
            self.processed_paths[0],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin() as txn:
            cache_length = int(txn.get(b"__len__").decode())
            excluded = set(pickle.loads(txn.get(b"__rmsd_timeout_indices__") or pickle.dumps([])))
        excluded.update(excluded_indices or [])
        self._cache_indices = [i for i in range(cache_length) if i not in excluded]
        self._length = len(self._cache_indices)
        if excluded:
            print(f"Excluded {len(excluded)} timeout/configured records from {self.split} training/evaluation", flush=True)
        env.close()

    def len(self):
        return self._length

    @property
    def raw_file_names(self) -> List[str]:
        return ["./"]

    @property
    def processed_file_names(self):
        return [f"PDBbind2020_{self.split}_pose_ranker_v5.lmdb"]

    def download(self) -> None:
        pass

    def _init_env(self):
        self.env = lmdb.open(
            self.processed_paths[0],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )

    def get(self, idx: int):
        if not hasattr(self, "env"):
            self._init_env()

        with self.env.begin() as txn:
            byteflow = txn.get(f"{self._cache_indices[idx]:08d}".encode())
            data = pickle.loads(byteflow)

        ligand_data = Molecule(**data.ligand_data)
        protein_data = ProteinData(**data.protein_data)

        ligand_data.remove_hydrogens()
        protein_data.remove_hydrogens()

        data_list = []
        subset_size = 10
        valid_indices = torch.where(torch.isfinite(data.rmsds) & (data.rmsds >= 0))[0]
        if valid_indices.numel() == 0:
            raise ValueError(f"No valid RMSD labels for cache record {self._cache_indices[idx]}")
        if self.pose_sampling == "rmsd_stratified_2233":
            sampled_indices = sample_rmsd_stratified_2233(data.rmsds)
        else:
            sampled_indices = valid_indices[torch.randperm(len(valid_indices))[:subset_size]]
        for conf_idx in sampled_indices:
            cpi_data = CPIdata.from_pair(ligand_data, protein_data)
            cpi_data.update_ligand_pos(ligand_data.confs_pos[conf_idx])
            cpi_data.y = 1 - torch.tanh(data.rmsds[conf_idx].unsqueeze(-1) / 3)
            data_list.append(copy.copy(cpi_data))

        batch = Batch.from_data_list(data_list)

        # batch = batch.extract_pocket_and_update_graph(
        #     cutoff=self.radius_cutoff, max_num_neighbors=999
        # )

        batch = keep_fields(
            batch,
            [
                "x",
                "pos",
                "chem_bond_index",
                "chem_bond_attr",
                "edge_index",
                "edge_vector",
                "edge_attr",
                "ligand_mask",
                "batch",
                "y",
            ],
        )

        return batch

    def process(self) -> None:
        raise FileNotFoundError(
            f"Prepared V5 pose-ranker cache is required: {self.processed_paths[0]}"
        )
