"""
python ./data/PDBbind2020_pocket.py
"""

import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import lmdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.PDBbind2020 import read_mol_cosmolkit, read_pdbbind_data
from data.utils import (
    molecule_to_graph,
    ProteinData,
    keep_fields,
    protein_to_graph,
)

torch.multiprocessing.set_sharing_strategy("file_system")


class PDBbind2020_pocket(Dataset):
    SUPPORTED_SPLITS = {"train", "valid", "test", "test_144"}
    SUPPORTED_MODES = {"train", "inference"}
    SUPPORTED_TASKS = {"pocket", "pocket_ranker"}

    def __init__(
        self,
        root,
        split="train",
        radius_cutoff=8.0,
        train_noise_std=0.0,
        mode="train",
        task="pocket",
        ranker_num_centers=300,
        ranker_noise_std=6.0,
        ranker_oversample_factor=5,
    ) -> None:
        assert split in self.SUPPORTED_SPLITS, f"Unsupported split: {split}"
        assert mode in self.SUPPORTED_MODES, f"Unsupported mode: {mode}"
        assert task in self.SUPPORTED_TASKS, f"Unsupported task: {task}"
        self.split = split
        self.radius_cutoff = radius_cutoff
        self.train_noise_std = train_noise_std
        self.mode = mode
        self.task = task
        self.ranker_num_centers = ranker_num_centers
        self.ranker_noise_std = ranker_noise_std
        self.ranker_oversample_factor = ranker_oversample_factor

        print(f"Loading PDBbind2020_pocket {split} dataset for {task}.")
        super().__init__(root)

        env = lmdb.open(
            self.processed_paths[0],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin() as txn:
            self._length = int(txn.get(b"__len__").decode())
        env.close()
        self._num_nodes_cache = {}

    def len(self):
        return self._length

    def __getstate__(self):
        # Spawned data workers open their own LMDB environment in get().
        state = self.__dict__.copy()
        state.pop("env", None)
        return state

    def get_num_nodes(self, idx: int) -> int:
        """Return the transformed graph node count without building edges.

        ``NodeBudgetBatchSampler`` calls this in the main process.  For the
        pocket prediction task, the final node count is the CA/CB backbone atom
        count used by ``to_backbone_graph``; reading that metadata avoids
        running the full ``get`` path before DataLoader workers load the sample.
        """

        idx = int(idx)
        if idx not in self._num_nodes_cache:
            if not hasattr(self, "env"):
                self._init_env()
            with self.env.begin() as txn:
                byteflow = txn.get(f"{idx:08d}".encode())
                protein_data_raw = pickle.loads(byteflow)
            backbone_atom_types = protein_data_raw.backbone_atom_types
            num_nodes = int(
                ((backbone_atom_types == 3) | (backbone_atom_types == 4)).sum().item()
            )
            self._num_nodes_cache[idx] = num_nodes
        return self._num_nodes_cache[idx]

    @property
    def raw_file_names(self) -> List[str]:
        return ["./"]

    @property
    def processed_file_names(self):
        return [f"PDBbind2020_pocket_{self.split}_v5.lmdb"]

    def download(self) -> None:
        pass

    def _init_env(self) -> None:
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
            byteflow = txn.get(f"{idx:08d}".encode())
            data = pickle.loads(byteflow)

            protein_data_raw = data
            ligand_center = protein_data_raw.ligand_centers.clone()
            protein_data = ProteinData(**protein_data_raw)
            protein_data.remove_hydrogens()

        if self.mode == "inference":
            return protein_data

        if self.task in {"pocket_ranker"}:
            gt_pockets = ligand_center
            if self.split == "train" and self.train_noise_std > 0.0:
                gt_pockets += torch.randn_like(gt_pockets) * self.train_noise_std

            num_fake = max(
                self.ranker_num_centers * self.ranker_oversample_factor,
                self.ranker_num_centers,
            )
            center_idx = torch.randint(gt_pockets.size(0), (num_fake,))
            fake_centers = (
                gt_pockets[center_idx]
                + torch.randn(
                    num_fake,
                    3,
                    dtype=gt_pockets.dtype,
                    device=gt_pockets.device,
                )
                * self.ranker_noise_std
            )

            prot_backbone_data = protein_data.to_backbone_graph()
            cb_pos = prot_backbone_data.pos[prot_backbone_data.node_types_atom == 4]
            if cb_pos.numel() > 0:
                min_cb_dist = torch.cdist(fake_centers, cb_pos).min(dim=1).values
                fake_centers = fake_centers[min_cb_dist <= self.ranker_noise_std]

            num_extra = max(self.ranker_num_centers - gt_pockets.size(0), 0)
            if fake_centers.size(0) > num_extra:
                fake_centers = fake_centers[
                    torch.randperm(fake_centers.size(0))[:num_extra]
                ]
            protein_data.predicted_pocket = torch.cat(
                [gt_pockets, fake_centers], dim=0
            )[: self.ranker_num_centers]

            protein_data.prepare_pocket_ranker_graph(
                protein_data.predicted_pocket,
                self.radius_cutoff,
                gt_pockets,
                max_centers=300,
            )
            protein_data.edge_vector = (
                protein_data.pos[protein_data.edge_index[0]]
                - protein_data.pos[protein_data.edge_index[1]]
            )
            protein_data.edge_types = torch.zeros(
                protein_data.edge_index.shape[1],
                dtype=torch.long,
                device=protein_data.pos.device,
            )

            return keep_fields(
                protein_data,
                ["x", "pos", "edge_index", "edge_types", "edge_vector", "y"],
            )

        if self.split == "train" and self.train_noise_std > 0.0:
            ligand_center += torch.randn_like(ligand_center) * self.train_noise_std

        prot_backbone_data = protein_data.to_backbone_graph()
        prot_backbone_data = prot_backbone_data.add_pocket_vec_multiple_ligands(
            ligand_center
        )
        prot_backbone_data = prot_backbone_data.update_graph(self.radius_cutoff)

        return prot_backbone_data

    def process_data(self, pdb_id, sdf_filename, mol2_filename, pdb_file):
        try:
            mol = read_mol_cosmolkit(sdf_filename, mol2_filename)
            ligand_data = molecule_to_graph(mol)
            protein_data = protein_to_graph(pdb_file, type="pdb")
            ligand_center = ligand_data.get_center()

            protein_data.pos = protein_data.pos.float()
            protein_data.residual_pos = protein_data.residual_pos.float()
            protein_data.backbone_pos = protein_data.backbone_pos.float()
            protein_data.ligand_centers = ligand_center.unsqueeze(0)
            protein_data.pdb_id = pdb_id
            protein_data.id = pdb_id
            return pickle.dumps(protein_data)

        except Exception as e:
            pid = os.getpid()
            traceback.print_exc()
            print(f"[Worker {pid}] Error in process_data:\n{e}")
            return None

    def _load_split_metadata(self, raw_path: str) -> pd.DataFrame:
        df_pdb_id = pd.read_csv(
            os.path.join(raw_path, "index/INDEX_general_PL_name.2020"),
            sep="  ",
            comment="#",
            header=None,
            names=[
                "pdb",
                "year",
                "uid",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
            ],
            engine="python",
        )[["pdb", "uid"]]

        data = read_pdbbind_data(
            os.path.join(raw_path, "index/INDEX_general_PL_data.2020")
        )
        data = data.merge(df_pdb_id, on=["pdb"])
        return data

    def _filter_by_split(self, data: pd.DataFrame, raw_path: str) -> pd.DataFrame:
        valid = np.loadtxt(
            os.path.join(raw_path, "timesplit_no_lig_overlap_val"), dtype=str
        )
        test = np.loadtxt(os.path.join(raw_path, "timesplit_test"), dtype=str)
        test_144 = np.loadtxt(
            os.path.join(raw_path, "timesplit_test_no_rec_overlap"), dtype=str
        )

        def assign_standard_split(pdb):
            if pdb in valid:
                return "valid"
            if pdb in test:
                return "test"
            return "train"

        def assign_test_144_split(pdb):
            if pdb in test_144:
                return "test_144"
            return "train"

        if self.split == "test_144":
            return data[data["pdb"].apply(assign_test_144_split) == "test_144"]

        return data[data["pdb"].apply(assign_standard_split) == self.split]

    def _build_tasks(self, data: pd.DataFrame, raw_path: str):
        tasks = []

        for _, row in tqdm(data.iterrows(), total=len(data)):
            pdb_id = row["pdb"]
            pdb_dir = os.path.join(raw_path, f"v2020-other-PL/{pdb_id}")

            if not os.path.exists(pdb_dir):
                continue

            sdf_filename = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
            mol2_filename = os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2")
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")

            tasks.append(
                (
                    pdb_id,
                    sdf_filename,
                    mol2_filename,
                    pdb_file,
                )
            )

        return tasks

    def _process_tasks_parallel(self, tasks, max_workers: int = 16):
        data_list = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_data, *task) for task in tasks]

            for future in tqdm(as_completed(futures), total=len(futures)):
                result_bytes = future.result()
                if result_bytes is not None:
                    data_list.append(result_bytes)

        return data_list

    def _write_lmdb(self, data_list) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)

        env = lmdb.open(self.processed_paths[0], map_size=1024**4)  # 1 TB
        with env.begin(write=True) as txn:
            for idx, data in enumerate(tqdm(data_list)):
                txn.put(f"{idx:08d}".encode(), data)
            txn.put(b"__len__", str(len(data_list)).encode())

        env.sync()
        env.close()

    def process(self) -> None:
        print(f"Processing PDBbind2020_pocket {self.split} dataset...")
        raw_path = self.raw_paths[0]

        data = self._load_split_metadata(raw_path)
        data = self._filter_by_split(data, raw_path)
        tasks = self._build_tasks(data, raw_path)
        data_list = self._process_tasks_parallel(tasks)
        self._write_lmdb(data_list)
        print(f"Data saved to {self.processed_paths[0]}")


if __name__ == "__main__":
    # dataset = PDBbind2020_pocket("./datasets/PDBbind2020", "train")
    # dataset = PDBbind2020_pocket("./datasets/PDBbind2020", "valid")
    dataset = PDBbind2020_pocket("./datasets/PDBbind2020", "test", task="pocket_ranker")
    for data in tqdm(dataset):
        print(data)
        break
