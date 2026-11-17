import multiprocessing as mp
import os
import re
import sys
import tempfile
import traceback
from argparse import ArgumentParser
from pathlib import Path

import easydict
import polars as pl
import torch
from importlib.metadata import version
from posebusters import PoseBusters
from rdkit import Chem
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.docking_utils import calc_rmsd, FAIL_RULES
from models.utils import resolve_path



assert version("posebusters") == "0.4.5", (
    f"PoseBusters version mismatch: "
    f"expected 0.4.5, got {version('posebusters')}"
)

buster = PoseBusters(config="redock")


def remove_hoh_from_pdb(pdb_in: Path, pdb_out: Path):
    with open(pdb_in, "r") as fin, open(pdb_out, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname == "HOH":
                    continue
            fout.write(line)


def pb_validate_test_local(pred_mol_block, gt_mol_block, protein_path):
    pred_mol = Chem.MolFromMolBlock(pred_mol_block, sanitize=True, removeHs=False)
    gt_mol = Chem.MolFromMolBlock(gt_mol_block, sanitize=True, removeHs=False)

    if pred_mol is None or gt_mol is None:
        return torch.tensor([False] + [True] * 26)

    pred_mol = Chem.RemoveHs(pred_mol)
    gt_mol = Chem.RemoveHs(gt_mol)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            pred_file = tmpdir / "pred.sdf"
            true_file = tmpdir / "gt.sdf"
            cond_file = tmpdir / "protein_nowater.pdb"

            pred_writer = Chem.SDWriter(str(pred_file))
            pred_writer.write(pred_mol)
            pred_writer.close()

            true_writer = Chem.SDWriter(str(true_file))
            true_writer.write(gt_mol)
            true_writer.close()

            remove_hoh_from_pdb(Path(protein_path), cond_file)

            df = buster.bust(
                [str(pred_file)],
                true_file,
                cond_file,
            )

            bad_cols = df.columns[~df.iloc[0].astype(bool)]
            pb_tensor = torch.tensor(df.iloc[0].values, dtype=torch.bool)

            if len(bad_cols):
                print(f"❌ PoseBusters failed checks:")
                for i, flag in enumerate(bad_cols):
                    print(f" - flag {flag}")

            return torch.tensor(df.iloc[0].values, dtype=torch.bool)
    except Exception:
        traceback.print_exc()
        return torch.tensor([False] + [True] * 26)


def pb_worker(task_queue, result_queue):
    while True:
        item = task_queue.get()
        if item is None:
            break

        index, job = item
        try:
            pb_tensor = pb_validate_test_local(
                job["pred_mol_block"],
                job["gt_mol_block"],
                job["protein_path"],
            )
            result_queue.put((index, pb_tensor.tolist(), None))
        except Exception:
            traceback.print_exc()
            result_queue.put((index, [False] + [True] * 26, job["complex_name"]))


def load_gt_mol(ligand_description):
    ligand_description = os.path.expanduser(ligand_description)
    if ligand_description.endswith(".sdf"):
        return Chem.MolFromMolFile(ligand_description, sanitize=True)
    if ligand_description.endswith(".mol2"):
        return Chem.MolFromMol2File(ligand_description, sanitize=True)
    raise ValueError(f"Unsupported ligand format: {ligand_description}")


def collect_ranked_predictions(out_dir, complex_name, perfect_selection):
    ranked = {}
    for path in Path(out_dir).expanduser().joinpath(complex_name).glob("rank*.sdf"):
        match = re.match(r"rank(\d+)_", path.name)
        if match is None:
            continue
        rank = int(match[1])
        if rank in ranked:
            raise ValueError(f"Duplicate rank {rank} in {path.parent}; regenerate into a clean directory")
        ranked[rank] = str(path)
    ranks = sorted(ranked) if perfect_selection else [rank for rank in sorted(ranked) if rank <= 5]
    return [ranked[rank] for rank in ranks]


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--protein_ligand_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    # parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 4))
    args = easydict.EasyDict(vars(parser.parse_args()))
    args.input_dir = resolve_path(args.input_dir)
    args.out_dir = resolve_path(args.out_dir, output=True)

    out_dir = args.out_dir
    perfect_selection = False

    df = pl.read_csv(args.protein_ligand_csv)
    tasks = [
        {
            "complex_name": row["complex_name"] or f"complex_{i}",
            "protein_path": row["protein_path"],
            "ligand_description": row["ligand_description"],
        }
        for i, row in enumerate(df.iter_rows(named=True))
    ]

    for task in tasks:
        for field in ("protein_path", "ligand_description"):
            path = os.path.expanduser(task[field])
            if args.input_dir is not None:
                path = os.path.join(args.input_dir, path)
            task[field] = resolve_path(path)

    rmsd_top1 = []
    rmsd_top5 = []
    rmsd_top5_min = []
    pb_jobs = []
    failed_complexes = []

    for task in tqdm(tasks, desc="Prepare metrics"):
        complex_name = task["complex_name"]
        gt_mol = load_gt_mol(task["ligand_description"])

        mol_pred_paths = collect_ranked_predictions(out_dir, complex_name, perfect_selection)
        if len(mol_pred_paths) < 1:
            failed_complexes.append(complex_name)
            print(f"{complex_name}: no ranked prediction found.")
            continue

        mol_preds = [Chem.MolFromMolFile(mol_pred_path) for mol_pred_path in mol_pred_paths]
        rmsds = torch.stack(
            [calc_rmsd(mol_pred, gt_mol, remove_Hs=True) for mol_pred in mol_preds],
            dim=0,
        )

        if perfect_selection:
            _, indices = torch.sort(rmsds, descending=False)
            rmsds = rmsds[indices]
            pb_pred_mol = mol_preds[indices[0].item()]
            rmsd_top5.append(rmsds[:5])
            rmsd_top5_min.append(torch.min(rmsds[:5]))
        else:
            pb_pred_mol = mol_preds[0]
            rmsd_top5.append(rmsds)
            rmsd_top5_min.append(torch.min(rmsds))

        rmsd_top1.append(rmsds[0])

        pb_jobs.append(
            {
                "complex_name": complex_name,
                "pred_mol_block": Chem.MolToMolBlock(pb_pred_mol),
                "gt_mol_block": Chem.MolToMolBlock(gt_mol),
                "protein_path": os.path.expanduser(task["protein_path"]),
            }
        )

    pb_results = [None] * len(pb_jobs)
    if pb_jobs:
        num_workers = max(1, args.num_workers)
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        workers = []

        for _ in range(num_workers):
            p = mp.Process(target=pb_worker, args=(task_queue, result_queue))
            p.start()
            workers.append(p)

        for index, job in enumerate(pb_jobs):
            task_queue.put((index, job))

        for _ in workers:
            task_queue.put(None)

        for _ in tqdm(range(len(pb_jobs)), total=len(pb_jobs), desc="PoseBusters"):
            index, pb_values, failed_name = result_queue.get()
            pb_results[index] = torch.tensor(pb_values, dtype=torch.bool)
            if failed_name is not None:
                failed_complexes.append(failed_name)

        for p in workers:
            p.join()

    rmsd_top1 = torch.tensor(rmsd_top1)
    rmsd_top5 = torch.nn.utils.rnn.pad_sequence(
        rmsd_top5, batch_first=True, padding_value=float("nan")
    ) if rmsd_top5 else torch.empty((0, 0))
    rmsd_top5_min = torch.tensor(rmsd_top5_min)
    PB_valids = torch.stack(pb_results) if pb_results else torch.empty((0, len(FAIL_RULES)), dtype=torch.bool)

    print(f"Failed complexes: {len(failed_complexes)}")

    if len(rmsd_top1) > 0:
        print("the top1 rmsd below 1.0:", torch.sum(rmsd_top1 < 1.0).item(), "/", len(rmsd_top1), "(", torch.sum(rmsd_top1 < 1.0).item() / len(rmsd_top1) * 100, "%)")
        print("the top1 rmsd below 2.0:", torch.sum(rmsd_top1 < 2.0).item(), "/", len(rmsd_top1), "(", torch.sum(rmsd_top1 < 2.0).item() / len(rmsd_top1) * 100, "%)")
        print("the top1 rmsd median:", torch.nanmedian(rmsd_top1))
        print("the top5 rmsd below 1.0:", torch.sum(rmsd_top5_min < 1.0).item(), "/", len(rmsd_top5_min), "(", torch.sum(rmsd_top5_min < 1.0).item() / len(rmsd_top5_min) * 100, "%)")
        print("the top5 rmsd below 2.0:", torch.sum(rmsd_top5_min < 2.0).item(), "/", len(rmsd_top5_min), "(", torch.sum(rmsd_top5_min < 2.0).item() / len(rmsd_top5_min) * 100, "%)")
        print("the top5 rmsd median:", torch.nanmedian(rmsd_top5_min))

    if PB_valids.shape[0] > 0:
        print("PoseBusters valid:", torch.sum(PB_valids.all(dim=-1)).item(), "/", PB_valids.shape[0], "(", torch.sum(PB_valids.all(dim=-1)).item() / PB_valids.shape[0] * 100, "%)")
        PB_failed_count = torch.sum(~PB_valids, dim=0)
        for i, flag in enumerate(PB_failed_count):
            print(f"{flag} - {FAIL_RULES[i]}")
