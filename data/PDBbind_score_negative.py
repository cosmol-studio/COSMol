"""Read prepared V5 pose-ranker training records from LMDB."""

import os
import pickle
import sys
from typing import List

import lmdb
import torch
from torch_geometric.data import Batch, Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.utils import CPIdata, Molecule, ProteinData, keep_fields, remove_fields


def normalize_pred_coords(pred_coords, num_confs: int, num_atoms: int):
    pred_coords = pred_coords.detach().cpu() if torch.is_tensor(pred_coords) else torch.as_tensor(pred_coords)
    if pred_coords.ndim == 3 and pred_coords.shape == (num_confs, num_atoms, 3):
        return pred_coords.float()
    if pred_coords.ndim == 2 and pred_coords.shape == (num_confs * num_atoms, 3):
        return pred_coords.view(num_confs, num_atoms, 3).float()
    raise ValueError(
        f"Invalid negative pred_coords shape {tuple(pred_coords.shape)}; "
        f"expected ({num_confs}, {num_atoms}, 3) or ({num_confs * num_atoms}, 3)"
    )


class PDBbind_score_negative(Dataset):
    def __init__(
        self,
        root,
        split="train",
        radius_cutoff=4.0,
        force_reload=False,
        excluded_indices=None,
    ) -> None:
        assert split in ["train", "valid", "test"], (
            f"Unsupported split: {split}. Must be 'train', 'valid', or 'test'."
        )
        self.split = split
        self.radius_cutoff = radius_cutoff
        self.excluded_indices = set(excluded_indices or [])

        print(f"Loading PDBbind pose negative {split} dataset.")
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
        self._cache_indices = [i for i in range(cache_length) if i not in self.excluded_indices]
        self._length = len(self._cache_indices)
        env.close()

    def len(self):
        return self._length

    @property
    def raw_file_names(self) -> List[str]:
        return ["./"]

    @property
    def processed_file_names(self):
        return [f"PDBbind2020_{self.split}_v5_pose_ranker_negative.lmdb"]

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

        protein_data = ProteinData(**data.protein_data)
        protein_data.remove_hydrogens()

        data_list = []
        subset_size = 10
        candidates = [(int(ligand_idx), pose_idx)
                      for ligand_idx in data.negative_complex
                      if int(ligand_idx) not in self.excluded_indices
                      for pose_idx in range(10)]
        if len(candidates) < subset_size:
            raise ValueError("Too few eligible negative poses after configured exclusions")
        for selected in torch.randperm(len(candidates))[:subset_size]:
            ligand_idx, conf_sub_idx = candidates[int(selected)]

            with self.env.begin() as txn:
                byteflow = txn.get(f"{ligand_idx:08d}".encode())
                data_tmp = pickle.loads(byteflow)
                ligand_data_negative = Molecule(**data_tmp.ligand_data)
                ligand_data_negative.remove_hydrogens()
                ligand_data_negative = remove_fields(
                    ligand_data_negative,
                    [
                        "bounds",
                        "chirality",
                        "chirality_ligand",
                        "chiral_neighbors",
                        "chiral_neighbors_ligand",
                    ],
                )

            pred_coords = normalize_pred_coords(
                data.negative_complex[ligand_idx]["pred_coords"],
                num_confs=10,
                num_atoms=int(ligand_data_negative.x.size(0)),
            )
            cpi_data = CPIdata.from_pair(ligand_data_negative, protein_data)
            cpi_data.update_ligand_pos(pred_coords[conf_sub_idx])
            cpi_data.y = torch.tensor(0.0)
            data_list.append(cpi_data)

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
