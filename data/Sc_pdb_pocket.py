"""scPDB training data and VN-EGNN binding-site benchmark datasets."""

import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import lmdb
import torch
from rdkit import Chem
from torch_geometric.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.utils import (
    ProteinData,
    keep_fields,
    protein_to_graph_rdkit,
    rdkit_molecule_to_graph,
)


def load_if_exists(path: Path) -> list[str]:
    if path.exists():
        with open(path, "r") as f:
            return f.read().splitlines()
    return []


# ======================= Dataset =======================
class Sc_pdb_pocket(Dataset):
    SUPPORTED_SPLITS = {"train", "valid", "coach420", "holo4k", "pdbbind2020"}
    SUPPORTED_MODES = {"train", "inference"}

    def __init__(
        self,
        root,
        split="train",
        radius_cutoff=10.0,
        radius_min=6.0,
        train_noise_std=0.0,
        mode="train",
        task="pocket",
        ranker_num_centers=300,
        ranker_noise_std=6.0,
        # ranker_cb_filter_cutoff=12.0,
        ranker_oversample_factor=5,
    ) -> None:
        split = "valid" if split == "test" else split
        assert split in self.SUPPORTED_SPLITS, f"Unsupported split: {split}"
        assert mode in self.SUPPORTED_MODES, f"Unsupported mode: {mode}"
        assert task in {"pocket", "ranker", "pocket_ranker"}, f"Unsupported task: {task}"
        self.split = split
        self.radius_cutoff = radius_cutoff
        self.radius_min = radius_min
        self.train_noise_std = train_noise_std
        self.mode = mode
        self.task = task
        self.ranker_num_centers = ranker_num_centers
        self.ranker_noise_std = ranker_noise_std
        # self.ranker_cb_filter_cutoff = ranker_cb_filter_cutoff
        self.ranker_oversample_factor = ranker_oversample_factor

        print(f"Loading sc_pdb_pocket {split} dataset for {task}")
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
        return ["./"]

    @property
    def processed_file_names(self):
        return [f"sc-pdb_{self.split}_v5.lmdb"]

    def download(self) -> None:
        # https://zenodo.org/records/17365855
        pass

    def _init_env(self) -> None:
        """Initialize LMDB environment lazily for worker-safe reading."""
        self.env = lmdb.open(
            self.processed_paths[0],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )

    def get(self, idx: int):
        """Load one processed sample and return the corresponding training/inference object."""
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

        if self.task in {"ranker", "pocket_ranker"}:
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

            # print(protein_data.predicted_pocket.shape)
            # print(protein_data)
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

            # print(protein_data)

            # graph_to_sdf_v3000(
            #     protein_data.pos,
            #     protein_data.x[:, 0],
            #     protein_data.edge_index,
            #     torch.ones_like(protein_data.edge_index[0]),
            #     # protein_data.edge_index,
            #     # protein_data.edge_types,
            #     sdf_path="test_p.cosdf",
            #     # dummy_node_pos=protein_data.predicted_pocket,
            #     # dummy_node_weight=torch.ones_like(protein_data.predicted_pocket[:, 0]),
            # )

            # exit()

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
        # print(self.radius_cutoff, self.radius_min, 8.0 / self.radius_cutoff)
        prot_backbone_data = prot_backbone_data.update_graph(
            self.radius_cutoff, self.radius_min, scale=8.0 / self.radius_cutoff
        )

        return prot_backbone_data

    # ======================= 数据生成 =======================
    def process_complex(self, complex_path):
        ligand_paths = [
            os.path.join(complex_path, f)
            for f in os.listdir(complex_path)
            if f.startswith("ligand")
        ]
        protein_path = os.path.join(complex_path, "protein.pdb")
        if not ligand_paths:
            print(f"Warning: No ligand files found in {complex_path}")
            return None

        try:
            ligand_mols = [
                Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
                for molecule_file in ligand_paths
            ]
            ligand_datas = [rdkit_molecule_to_graph(Chem.Mol(mol)) for mol in ligand_mols]
            ligand_centers = [
                ligand_data.pos.mean(dim=0) for ligand_data in ligand_datas
            ]
            ligand_pos = []
            ligand_id = []
            for i, ligand_data in enumerate(ligand_datas):
                ligand_pos.append(ligand_data.pos)
                ligand_id.extend([i] * ligand_data.pos.shape[0])

            protein_data = protein_to_graph_rdkit(protein_path, type="pdb")
            protein_data.ligand_centers = torch.stack(ligand_centers)
            protein_data.ligand_pos = torch.concat(ligand_pos, dim=0)
            protein_data.ligand_id = torch.tensor(ligand_id)
            protein_data.id = complex_path.split("/")[-1]
            return pickle.dumps(protein_data)
        except Exception as e:
            pid = os.getpid()
            print(f"[Worker {pid}] Error in generate_data:")
            traceback.print_exc()
            return None

    # ======================= 处理原始数据 =======================
    def process(self) -> None:
        print(f"Processing sc_pdb_pocket {self.split} dataset...")
        raw_path = self.raw_paths[0]
        match self.split:
            case "train" | "valid":
                dataset_path = os.path.join(raw_path, "sc-pdb")
                complex_names_path = os.path.join(
                    dataset_path, "splits", f"{self.split}_ids_scpdb"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()

                blacklist_path = Path(
                    os.path.join(dataset_path, "splits", "scPDB_blacklist.txt")
                )
                leakage_path = Path(
                    os.path.join(dataset_path, "splits", "scPDB_leakage.txt")
                )
                blacklist = load_if_exists(blacklist_path)
                leakage = load_if_exists(leakage_path)

                complex_paths = [
                    os.path.join(dataset_path, "raw", c)
                    for c in complex_names
                    if c not in blacklist and c not in leakage
                ]
            case "coach420" | "holo4k" | "pdbbind2020":
                dataset_path = os.path.join(raw_path, self.split)
                complex_names_path = os.path.join(
                    dataset_path, "splits", f"test_ids_{self.split}"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()

                raw_file_names = os.listdir(os.path.join(dataset_path, "raw"))

                if self.split == "pdbbind2020" or self.split == "holo4k":
                    complex_paths = [
                        os.path.join(dataset_path, "raw", f)
                        for f in raw_file_names
                        if f[:4] in complex_names
                    ]

                else:
                    complex_paths = [
                        os.path.join(dataset_path, "raw", f)
                        for f in raw_file_names
                        if f in complex_names
                    ]

                print(raw_file_names[:10], "\n\n\n\n", complex_names[:10])

                if len(complex_paths) == 0:
                    raise ValueError(f"No complex files found for split {self.split}")

            case _:
                raise ValueError(f"Invalid mode: {self.split}")

        # ======================= 并行处理 =======================
        data_list = []
        with ProcessPoolExecutor(max_workers=48) as executor:
            futures = [
                executor.submit(self.process_complex, complex_path)
                for complex_path in complex_paths
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result_bytes = future.result()
                if result_bytes is not None:
                    data_list.append(pickle.loads(result_bytes))

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
