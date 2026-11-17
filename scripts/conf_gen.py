"""Generate conformers for one ligand or a CSV of ligands.

python scripts/conf_gen.py -c config/Joint/Joint_all.toml
python scripts/conf_gen.py -c config/Joint/Joint_all.toml --ligand_csv input/ligands.csv
"""

import csv
import gc
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cosmolkit as ck
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.utils import (
    get_confs_from_coords,
    read_ligand,
    molecule_to_graph,
    save_sdf,
)
from models.utils import load_toml_cfg, resolve_path
from scripts.conformer_utils import max_total_atoms, pred_confs_batch

DEFAULT_SMILES = r"CC1=CC(F)=CC=C1S(N/C1=N/C=N\C2=C1C(C1=CC=C(F)C=C1)=CS2)(=O)=O"


def align_molecules(mols):
    if not mols:
        return mols

    ref_mol = mols[0]
    atom_map = [ck.AlignmentAtomMap(i, i) for i in range(len(ref_mol.atoms))]
    for idx, prb_mol in enumerate(mols[1:], start=1):
        try:
            prb_mol.align_to_(ref_mol, atom_map=atom_map)
        except Exception as exc:
            print(f"Failed to align conformer {idx}: {exc}")
    return mols


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--ligand_csv", "--ligand-csv", type=Path)
    inputs.add_argument("--ligand_description", "--ligand-description")
    parser.add_argument("--out_dir", "--out-dir", type=Path, default=Path("output/conformers"))
    parser.add_argument("--num_confs", "--num-confs", type=int, default=40)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32,
                        help="Maximum number of ligands passed to batched generation")
    parser.add_argument("--atom_budget", "--atom-budget", type=int, default=max_total_atoms,
                        help="Maximum atoms per model batch, including conformer copies")
    parser.add_argument("--flow_temperature", "--flow-temperature", type=float, default=1.0)
    parser.add_argument("--device", help="Torch device, for example cuda:0 or cpu")
    args = parser.parse_args()
    if min(args.num_confs, args.batch_size, args.atom_budget) < 1:
        parser.error("Conformer count, batch size and atom budget must be positive")
    if not np.isfinite(args.flow_temperature) or args.flow_temperature <= 0:
        parser.error("Flow temperature must be finite and positive")

    records = []
    if args.ligand_csv:
        with args.ligand_csv.expanduser().open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "ligand_description" not in reader.fieldnames:
                parser.error("CSV must contain a ligand_description column")
            for row_number, row in enumerate(reader, start=1):
                name = (row.get("ligand_name") or f"ligand_{row_number}").strip()
                description = (row.get("ligand_description") or "").strip()
                if not description:
                    parser.error(f"Empty ligand_description at CSV row {row_number}")
                if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
                    parser.error(f"ligand_name must be a directory name at CSV row {row_number}")
                records.append((name, description))
        if not records:
            parser.error("CSV contains no ligands")
        if len({name for name, _ in records}) != len(records):
            parser.error("CSV ligand_name values must be unique")
    else:
        records = [("ligand", args.ligand_description or DEFAULT_SMILES)]

    cfg = load_toml_cfg(args.config)
    if args.device:
        device = torch.device(args.device)
    elif cfg.device.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.device.gpus[0]}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device: {device}")

    output_root = Path(resolve_path(args.out_dir, output=True))
    atom_budget = args.atom_budget
    failures = []
    for start in range(0, len(records), args.batch_size):
        names, data_items = [], []
        for name, description in records[start:start + args.batch_size]:
            try:
                mol = read_ligand(description).with_hydrogens().sanitize()
                mol = mol.with_only_3d_conformer(np.zeros((len(mol.atoms), 3)))
                data = molecule_to_graph(mol, keep_mol=True, keep_smiles=True, keep_chirality=True)
            except Exception as exc:
                failures.append(name)
                print(f"Failed to prepare {name}: {exc}", file=sys.stderr)
                continue
            names.append(name)
            data_items.append(data)
        if not data_items:
            continue

        largest_ligand = max(len(data.mol.atoms) for data in data_items)
        while True:
            try:
                with torch.inference_mode():
                    predictions = pred_confs_batch(
                        cfg, data_items, device, num_confs=args.num_confs,
                        flow_temperature=args.flow_temperature, atom_budget=atom_budget,
                    )
            except torch.cuda.OutOfMemoryError:
                smaller_budget = max(largest_ligand, atom_budget // 2)
                if smaller_budget >= atom_budget:
                    raise
            else:
                break
            gc.collect()
            torch.cuda.empty_cache()
            atom_budget = smaller_budget
            print(f"Retrying generation with atom budget {atom_budget}", file=sys.stderr)

        if len(predictions) != len(data_items):
            raise ValueError("Generated ligand count does not match the input batch")
        for name, data, coords in zip(names, data_items, predictions):
            pred_mols = get_confs_from_coords(data.mol, coords, has_hydrogens=True)
            if len(pred_mols) != args.num_confs:
                raise ValueError(f"Incorrect conformer count for {name}")
            pred_mols = align_molecules(pred_mols)
            out_dir = output_root / name if args.ligand_csv else output_root
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, mol in enumerate(pred_mols):
                save_sdf(mol, str(out_dir / f"{idx}.sdf"), has_hydrogens=True)
            print(f"Saved {len(pred_mols)} conformers to {out_dir.resolve()}")
        del predictions

    if failures:
        raise SystemExit(f"Failed to prepare {len(failures)} ligands: {', '.join(failures)}")


if __name__ == "__main__":
    main()
