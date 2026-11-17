"""
python ./data/GEOM_SDEGen.py
"""

import copy
import os
import pickle
import sys
from typing import List

import cosmolkit as ck
import lmdb
import torch
from rdkit import Chem
from torch_geometric.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tqdm import tqdm

from data.utils import (
    molecule_to_graph,
    Molecule,
    check_disconnected_components,
    rdkit_molecule_to_graph,
    sample_time,
)


class GEOM_SDEGen(Dataset):
    def __init__(
        self,
        root,
        split="train",
        radius_cutoff=3.0,
        force_reload=False,
        mode="train",
        remove_hydrogens=False,
        prior_type="harmonic",
        time_high=0.9999,
        terminal_time_threshold=0.8,
        terminal_x0_noise_std=0.0,
        molecule_backend="cosmolkit",
    ) -> None:
        assert split in ["train", "valid", "test"], (
            f"Unsupported split: {split}. Must be 'train', 'valid', or 'test'."
        )
        assert mode in ["train", "inference"]
        assert molecule_backend in {"rdkit", "cosmolkit"}, (
            f"Unsupported molecule_backend: {molecule_backend}"
        )
        print(f"Loading GEOM_SDEGen_drugs {split} dataset")
        self.file_name = f"{split}.pt"
        self.split = split
        self.prior_type = prior_type
        self.mode = mode
        self.radius_cutoff = radius_cutoff
        self.remove_hydrogens = remove_hydrogens
        self.time_high = time_high
        self.terminal_time_threshold = terminal_time_threshold
        self.terminal_x0_noise_std = terminal_x0_noise_std
        self.molecule_backend = molecule_backend
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

    def len(self):
        return self._length

    @property
    def raw_file_names(self) -> List[str]:
        match self.split:
            case "train":
                return ["train_data_39k.pkl"]
            case "valid":
                return ["val_data_5k.pkl"]
            case "test":
                return ["test_data_200.pkl"]

    @property
    def processed_file_names(self):
        suffix = (
            "" if self.molecule_backend == "rdkit" else "_cosmolkit"
        )
        return [f"geom_{self.split}_v5{suffix}.lmdb"]

    def download(self) -> None:
        0

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
            byteflow = txn.get(f"{idx:08d}".encode())
            data = pickle.loads(byteflow)

        data = Molecule(**data)
        if self.remove_hydrogens:
            data.remove_hydrogens()

        if self.mode == "inference":
            return data

        if self.split == "test":
            sample = data.sample_conformer()
        else:
            sample = data.get_train_sample()

        t = sample_time(high=self.time_high)
        x0, batch = sample.sample_base_distribution_harmonic(align=True)
        data_xt = sample.compute_flow_field(
            x0,
            t,
            terminal_time_threshold=self.terminal_time_threshold,
            terminal_x0_noise_std=self.terminal_x0_noise_std,
        )
        data_xt = data_xt.update_graph(self.radius_cutoff, t)
        data_xt.ligand_mask = torch.ones(data_xt.x.shape[0], dtype=torch.bool)

        return data_xt

    def process_data(self, data):
        totalenergy = torch.as_tensor(data.totalenergy).clone().float().reshape(-1)
        rdmol = data.rdmol
        if self.molecule_backend == "cosmolkit":
            mol = ck.Molecule.from_rdkit(rdmol, sanitize=False)
            data = molecule_to_graph(
                mol,
                keep_mol=True,
                keep_smiles=True,
                keep_chirality=True,
            )
        else:
            data = rdkit_molecule_to_graph(
                rdmol,
                keep_smiles=True,
                keep_chirality=True,
            )

        data.totalenergy = totalenergy

        # Check for disconnected components
        components = check_disconnected_components(
            data.x.shape[0], data.chem_bond_index
        )
        if len(components) > 1:
            print("Disconnected components detected!")
            return None

        if data is None:
            return None

        return pickle.dumps(data)

    def process_test_data(self, confs):
        data = None
        confs_pos = []
        confs_totalenergy = []
        for i, conf in enumerate(confs):
            if self.molecule_backend == "cosmolkit":
                mol = ck.Molecule.from_rdkit(conf.rdmol, sanitize=False)
                data_ = molecule_to_graph(
                    mol,
                    keep_mol=(i == 0),
                    keep_smiles=False,
                    keep_chirality=True,
                )
                data_.smiles = Chem.MolToSmiles(conf.rdmol, isomericSmiles=True)
            else:
                data_ = rdkit_molecule_to_graph(
                    conf.rdmol,
                    keep_mol=(i == 0),
                    keep_smiles=True,
                    keep_chirality=True,
                )
            confs_pos.append(data_.pos)
            confs_totalenergy.append(
                torch.as_tensor(conf.totalenergy).clone().float().reshape(-1)
            )
            if data is None:
                data = copy.copy(data_)

        components = check_disconnected_components(
            data.x.shape[0], data.chem_bond_index
        )
        if len(components) > 1:
            print("Disconnected components detected!")
            return None

        if data is None:
            return None

        data.confs_pos = torch.stack(confs_pos, dim=0)
        data.confs_totalenergy = torch.stack(confs_totalenergy, dim=0)
        data.num_confs = len(confs_pos)

        return data

    def process(self) -> None:
        print(f"Processing GEOM_SDEGen_drugs {self.split} split")

        if self.split != "test":
            with open(self.raw_paths[0], "rb") as f:
                datas = pickle.load(f)

            filtered_datas = []
            for data in tqdm(datas):
                mol_o = data.rdmol
                if "." in Chem.MolToSmiles(mol_o):
                    print("Disconnected components detected :", Chem.MolToSmiles(mol_o))
                    continue
                filtered_datas.append(data)

            print("Filtered data size:", len(filtered_datas))

            data_list = []

            for data in tqdm(filtered_datas):
                result_bytes = self.process_data(data)
                if result_bytes is not None:
                    data_list.append(pickle.loads(result_bytes))

        else:
            with open(self.raw_paths[0], "rb") as f:
                datas = pickle.load(f)

            filtered_datas = {}
            for data in tqdm(datas):
                mol_o = data.rdmol
                if "." in Chem.MolToSmiles(mol_o):
                    print("Disconnected components detected :", Chem.MolToSmiles(mol_o))
                    continue
                if data.smiles not in filtered_datas:
                    filtered_datas[data.smiles] = []
                filtered_datas[data.smiles].append(data)

            print("Filtered molecules:", len(filtered_datas))

            data_list = []
            for key in tqdm(filtered_datas.keys()):
                result = self.process_test_data(filtered_datas[key])
                if result is not None:
                    data_list.append(result)

            num = 0
            for data in data_list:
                num += data.num_confs

            print("Total molecules:", num)

        # ======================= LMDB 保存 =======================
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(self.processed_paths[0], map_size=1024**4)  # 1TB
        with env.begin(write=True) as txn:
            for idx, data in enumerate(tqdm(data_list)):
                txn.put(f"{idx:08d}".encode(), pickle.dumps(data))
            txn.put(b"__len__", str(len(data_list)).encode())
        env.sync()
        env.close()
        print(f"Data saved to {self.processed_paths[0]}")
        print(f"{len(data_list)} data saved to {self.processed_paths[0]}")


if __name__ == "__main__":
    dataset = GEOM_SDEGen("./datasets/GEOM_SDEGen", "test", radius_cutoff=4.0)
    dataset = GEOM_SDEGen("./datasets/GEOM_SDEGen", "train", radius_cutoff=4.0)
    dataset = GEOM_SDEGen("./datasets/GEOM_SDEGen", "valid", radius_cutoff=4.0)

    for data in tqdm(dataset):
        print(data)
        pass
