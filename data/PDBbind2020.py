"""
python ./data/PDBbind2020.py
"""

import os
import pickle
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import cosmolkit as ck
import lmdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.utils import (
    molecule_to_graph,
    CPIdata,
    Molecule,
    ProteinData,
    compute_docking_flow_fields,
    get_pocket_center,
    keep_fields,
    protein_to_graph,
    sample_time,
)

torch.multiprocessing.set_sharing_strategy("file_system")

def read_pdbbind_data(fileName):
    with open(fileName) as f:
        lines = f.readlines()
    info = []
    for line in lines:
        if line[0] == "#":
            continue
        lines_part, ligand = line.split("//")
        pdb, resolution, year, affinity, raw = lines_part.strip().split("  ")
        ligand = ligand.strip().split("(")[1].split(")")[0]
        info.append([pdb, resolution, year, affinity, raw, ligand])
    df = pd.DataFrame(
        info, columns=["pdb", "resolution", "year", "affinity", "raw", "ligand"]
    )
    df.year = df.year.astype(int)
    df.affinity = df.affinity.astype(float)
    return df


def read_mol_cosmolkit(sdf_fileName, mol2_fileName):
    errors = []
    try:
        return ck.Molecule.read_mol(sdf_fileName, sanitize=True)
    except Exception as exc:
        errors.append(f"SDF {type(exc).__name__}: {exc}")

    try:
        return ck.Molecule.read_mol2(
            mol2_fileName, sanitize=True, remove_hs=True
        )
    except Exception as exc:
        errors.append(f"MOL2 {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "cosmolkit ligand read failed for "
        f"sdf={sdf_fileName}, mol2={mol2_fileName}; " + "; ".join(errors)
    )


class PDBbind2020(Dataset):
    """PDBbind 2020 dataset backed by LMDB.

    This dataset supports two usage modes:

    - ``train``: load processed complexes and generate docking training samples.
    - ``inference``: load ligand/protein pairs directly for inference.

    Parameters
    ----------
    root : str
        Dataset root directory.
    split : str, default="train"
        Dataset split. Supported values: ``train``, ``valid``, ``test``, ``test_144``.
    radius_cutoff : float, default=3.0
        Pocket extraction radius cutoff.
    mode : str, default="train"
        Usage mode. Supported values: ``train`` and ``inference``.
    max_translation_shift : float, default=10.0
        Maximum ligand translation shift used during docking training sample generation.
    chirality_prior_max_attempts : int, default=32
        Maximum number of harmonic prior samples to try before using the best
        chirality-matching candidate.
    chirality_prior_margin : float, default=0.02
        Normalized oriented-volume margin used when accepting a chiral prior sample.
    """

    SUPPORTED_SPLITS = {"train", "valid", "test", "test_144"}
    SUPPORTED_MODES = {"train", "inference"}

    def __init__(
        self,
        root,
        split="train",
        radius_cutoff=3.0,
        mode="train",
        max_translation_shift=10.0,
        chirality_prior_max_attempts=32,
        chirality_prior_margin=0.02,
        time_high=0.9999,
        terminal_time_threshold=0.8,
        terminal_x0_noise_std=2.0,
        prior_extra_noise_std=0.0,
        eval_seed=95028,
    ) -> None:
        assert split in self.SUPPORTED_SPLITS, f"Unsupported split: {split}"
        assert mode in self.SUPPORTED_MODES, f"Unsupported mode: {mode}"

        self.split = split
        self.radius_cutoff = radius_cutoff
        self.mode = mode
        self.max_translation_shift = max_translation_shift
        self.chirality_prior_max_attempts = chirality_prior_max_attempts
        self.chirality_prior_margin = chirality_prior_margin
        self.time_high = time_high
        self.terminal_time_threshold = terminal_time_threshold
        self.terminal_x0_noise_std = terminal_x0_noise_std
        self.prior_extra_noise_std = prior_extra_noise_std if split == "train" else 0.0
        self.eval_seed = int(eval_seed)

        # print(f"Loading PDBbind2020 {split} dataset.")
        if split != "train" and mode != "inference":
            print(f"Fixed docking evaluation sampling: seed={self.eval_seed} + cache index")
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
        return [f"PDBbind2020_{self.split}_v5.lmdb"]

    def download(self) -> None:
        """No download logic is provided for this dataset."""
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
        # The same cached validation record always receives the same flow state,
        # independently of epoch, DDP rank, worker count and access order.
        if self.split == "train" or self.mode == "inference":
            return self._get_sample(idx)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        seed = (self.eval_seed + int(idx)) % (2**32)
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Seed CPU only: dataset workers must not initialize CUDA.
            torch.random.default_generator.manual_seed(seed)
            return self._get_sample(idx)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def _get_sample(self, idx: int):
        """Load one processed sample and return the corresponding training/inference object."""
        if not hasattr(self, "env"):
            self._init_env()

        with self.env.begin() as txn:
            byteflow = txn.get(f"{idx:08d}".encode())
            data = pickle.loads(byteflow)

        ligand_data_raw, protein_data_raw = data
        ligand_data = Molecule(**ligand_data_raw)
        protein_data = ProteinData(**protein_data_raw)

        protein_data.path = os.path.join(
            self.raw_paths[0],
            f"v2020-other-PL/{protein_data.pdb_id}/{protein_data.pdb_id}_protein.pdb",
        )

        ligand_data.remove_hydrogens()
        protein_data.remove_hydrogens()
        ligand_data.pos = ligand_data.pos.float()
        protein_data.pos = protein_data.pos.float()
        protein_data.backbone_pos = protein_data.backbone_pos.float()
        protein_data.residual_pos = protein_data.residual_pos.float()
        protein_data.pocket_center = protein_data.pocket_center.float()

        if self.mode == "inference":
            return ligand_data, protein_data

        t = sample_time(high=self.time_high)
        ligand_center = ligand_data.get_center()

        mol_x0_pos, R, shift, batch = (
            ligand_data.sample_base_distribution_docking_train(
                ligand_center,
                self.max_translation_shift,
                chirality_prior_max_attempts=self.chirality_prior_max_attempts,
                chirality_prior_margin=self.chirality_prior_margin,
                prior_extra_noise_std=self.prior_extra_noise_std,
            )
        )
        flow = compute_docking_flow_fields(
            ligand_pos=ligand_data.pos,
            x0=mol_x0_pos,
            R=R,
            t=t,
            ligand_center=ligand_center,
            shift=shift,
            train_shift_range=self.max_translation_shift,
            terminal_time_threshold=self.terminal_time_threshold,
            terminal_x0_noise_std=self.terminal_x0_noise_std,
        )

        cpi_data = CPIdata.from_pair(
            ligand_data,
            protein_data,
            mol_pos=flow.pos,
            y=flow.y,
            R_pred=flow.R_pred,
            u_t=flow.u_t,
            shift=flow.shift,
        )
        ligand_pocket = cpi_data.extract_pocket_and_update_graph(
            self.radius_cutoff,
            flow.t,
            compute_edge_mindist=True,
        )
        ligand_pocket = keep_fields(
            ligand_pocket,
            [
                "x",
                "y",
                "t",
                "u_t",
                "pos",
                "shift",
                "R_pred",
                "edge_attr",
                "chirality",
                "edge_index",
                "ligand_mask",
                "edge_vector",
                "edge_mindist",
                "chirality_ligand",
                "chiral_neighbors",
                "edge_chirality_feat",
                "chiral_neighbors_ligand",
            ],
        )

        return ligand_pocket

    def process_data(
        self,
        pdb_id,
        sdf_filename,
        mol2_filename,
        pdb_file,
        pdb_pocket_file,
    ):
        """Process one raw complex into a serialized training sample."""
        try:
            mol = read_mol_cosmolkit(sdf_filename, mol2_filename)
            ligand_data = molecule_to_graph(
                mol,
                keep_mol=True,
                keep_smiles=True,
                keep_chirality=True,
            )
            protein_data = protein_to_graph(pdb_file, type="pdb")

            protein_data.pocket_center = get_pocket_center(pdb_pocket_file)
            protein_data.pdb_id = pdb_id
            return pickle.dumps((ligand_data, protein_data))

        except Exception as e:
            pid = os.getpid()
            traceback.print_exc()
            print(f"[Worker {pid}] Error in process_data:\n{e}")
            return None

    def _load_split_metadata(self, raw_path: str) -> pd.DataFrame:
        """Load and merge PDBbind index metadata."""
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
        """Filter metadata table according to the requested split."""
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
        """Build raw-file processing tasks for the selected split."""
        tasks = []

        for _, row in tqdm(data.iterrows(), total=len(data)):
            pdb_id = row["pdb"]
            pdb_dir = os.path.join(raw_path, f"v2020-other-PL/{pdb_id}")

            if not os.path.exists(pdb_dir):
                continue

            sdf_filename = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
            mol2_filename = os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2")
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
            pdb_pocket_file = os.path.join(pdb_dir, f"{pdb_id}_pocket.pdb")

            tasks.append(
                (
                    pdb_id,
                    sdf_filename,
                    mol2_filename,
                    pdb_file,
                    pdb_pocket_file,
                )
            )

        return tasks

    def _process_tasks_parallel(self, tasks, max_workers: int = 32):
        """Process all tasks in parallel and collect valid results."""
        data_list = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_data, *task) for task in tasks]

            for future in tqdm(as_completed(futures), total=len(futures)):
                result_bytes = future.result()
                if result_bytes is not None:
                    data_list.append(result_bytes)

        return data_list

    def _write_lmdb(self, data_list) -> None:
        """Write processed samples to LMDB."""
        os.makedirs(self.processed_dir, exist_ok=True)

        env = lmdb.open(self.processed_paths[0], map_size=1024**4)  # 1 TB
        with env.begin(write=True) as txn:
            for idx, data in enumerate(tqdm(data_list)):
                txn.put(f"{idx:08d}".encode(), data)
            txn.put(b"__len__", str(len(data_list)).encode())

        env.sync()
        env.close()

    def process(self) -> None:
        """Process raw PDBbind files into an LMDB dataset."""
        print(f"Processing PDBbind2020 {self.split} dataset...")
        raw_path = self.raw_paths[0]

        data = self._load_split_metadata(raw_path)
        data = self._filter_by_split(data, raw_path)
        tasks = self._build_tasks(data, raw_path)
        data_list = self._process_tasks_parallel(tasks)
        self._write_lmdb(data_list)
        print(f"Data saved to {self.processed_paths[0]}")


# ======================= 测试示例 =======================
if __name__ == "__main__":
    dataset = PDBbind2020("./datasets/PDBbind2020", "test", mode="inference")
    dataset = PDBbind2020("./datasets/PDBbind2020", "train", mode="inference")
    dataset = PDBbind2020("./datasets/PDBbind2020", "valid", mode="inference")
    for data in tqdm(dataset):
        pass
