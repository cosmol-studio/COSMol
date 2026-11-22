"""
python scripts/virtual_screening.py -c config/dock_config.toml --data-root ./datasets/DEKOIS2 --out_dir ./vs_results --target pdk1
python scripts/virtual_screening.py -c config/dock_config.toml --input-format dudez --data-root ./datasets/DUDE-Z --out_dir ./vs_results_dudez

Add --unbound-support to save free conformers for scripts/evaluate_virtual_screening.py.
"""

import csv
import gc
import hashlib
import json
import os
import pickle
import random
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
from queue import Empty
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.utils import (
    read_ligand,
    cache_mapping,
    molecular_graph,
    molecule_to_graph,
    protein_to_graph,
)
from models.utils import load_toml_cfg
from scripts.docking_utils import pred_confs

def find_target_files(target_dir: str):
    protein_dir = os.path.join(target_dir, "protein")
    if not os.path.isdir(protein_dir):
        raise FileNotFoundError(f"protein folder not found: {protein_dir}")

    protein_path = None
    ligand_mol2_path = None

    for f in os.listdir(protein_dir):
        full = os.path.join(protein_dir, f)
        if f.endswith("_protein.pdb"):
            protein_path = full
        elif f.endswith("_ligand.mol2"):
            ligand_mol2_path = full

    if protein_path is None:
        raise FileNotFoundError(f"Cannot find *_protein.pdb in {protein_dir}")
    if ligand_mol2_path is None:
        raise FileNotFoundError(f"Cannot find *_ligand.mol2 in {protein_dir}")

    return protein_path, ligand_mol2_path


def get_mol2_center_raw(mol2_path: str):
    coords = []
    in_atom_block = False

    with open(mol2_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            elif line.startswith("@<TRIPOS>") and in_atom_block:
                break

            if in_atom_block and line:
                parts = line.split()
                if len(parts) < 6:
                    continue

                # mol2 atom line:
                # atom_id atom_name x y z atom_type ...
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                if parts[5].split(".", 1)[0].upper() != "H":
                    coords.append((x, y, z))

    if not coords:
        raise ValueError(f"No atom coordinates found in {mol2_path}")

    center = [sum(c[i] for c in coords) / len(coords) for i in range(3)]
    return torch.tensor(center, dtype=torch.float32)


def read_active_decoys_smi(smi_path: str):
    records = []
    with open(smi_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad line in {smi_path} at {line_idx + 1}: {line}")

            smiles = parts[0]
            mol_id = parts[1]
            extra = parts[2:] if len(parts) > 2 else []

            records.append(
                {
                    "smiles": smiles,
                    "mol_id": mol_id,
                    "extra": extra,
                    "source_index": len(records),
                    "label": 1 if mol_id.startswith("BDB") else (
                        0 if mol_id.startswith("ZINC") else None
                    ),
                }
            )
    return records


def prepare_protein_data(protein_path: str):
    if protein_path.endswith(".pdb"):
        protein_data = protein_to_graph(
            os.path.expanduser(protein_path), type="pdb"
        )
    elif protein_path.endswith(".cif"):
        protein_data = protein_to_graph(
            os.path.expanduser(protein_path), type="cif"
        )
    else:
        raise ValueError(f"Unsupported protein file format: {protein_path}")

    protein_data.remove_hydrogens()
    return protein_data


def load_target(target_dir, input_format="dekois"):
    """Read the original DEKOIS layout or a DUDE-Z raw target directory."""
    root = Path(target_dir)
    if input_format == "dekois":
        protein, reference = find_target_files(str(root))
        return protein, get_mol2_center_raw(reference), read_active_decoys_smi(
            str(root / "active_decoys.smi")
        )
    protein, reference = root / "rec.crg.pdb", root / "xtal-lig.pdb"
    if not protein.is_file():
        raise FileNotFoundError(protein)
    coords = []
    for line in reference.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            element = line[76:78].strip().upper()
            if element in ("H", "D") or (
                not element and line[12:16].strip().lstrip("0123456789").startswith(("H", "D"))
            ):
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No heavy atoms in {reference}")
    records = []
    for filename, label in [("ligands.smi", 1), ("decoys.smi", 0)]:
        for line_no, line in enumerate((root / filename).read_text().splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            records.append(dict(
                smiles=parts[0], mol_id=f"{root.name}:{label}:{line_no}",
                original_id=parts[1] if len(parts) > 1 else str(line_no),
                label=label, source_index=len(records),
            ))
    return str(protein), torch.tensor(coords, dtype=torch.float32).mean(0), records


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def score_ligand_batch(
    records: List[dict],
    protein_data,
    ligand_center: torch.Tensor,
    device,
    args,
):
    if len(records) == 0:
        return []

    datalist = []
    valid_records = []
    failed_results = []

    for rec in records:
        smiles = rec["smiles"]
        mol_id = rec["mol_id"]
        try:
            mol = read_ligand(smiles).without_hydrogens()
            ligand_data = molecule_to_graph(
                mol, keep_mol=True, keep_smiles=True, keep_chirality=True
            )
            ligand_data.remove_hydrogens()
            datalist.append((ligand_data, protein_data))
            valid_records.append(rec)
        except Exception as e:
            failed_results.append(
                {
                    "mol_id": mol_id,
                    "smiles": smiles,
                    "ranker_score": "",
                    "ranker_scores": "",
                    "status": f"failed: {repr(e)}",
                    "source_index": rec["source_index"],
                    "label": rec.get("label"),
                    "original_id": rec.get("original_id", mol_id),
                }
            )

    if len(valid_records) == 0:
        return failed_results

    pocket_pos_list = [ligand_center] * len(valid_records)
    random.seed(args.seed + records[0]["source_index"])
    np.random.seed(args.seed + records[0]["source_index"])
    torch.manual_seed(args.seed + records[0]["source_index"])
    with torch.inference_mode():
        results = pred_confs(
            datalist,
            device,
            num_confs=args.num_confs,
            pocket="given",
            pocket_pos_list=pocket_pos_list,
            docking_cfg=args.docking_cfg,
            calculate_score=True,
            pose_ranker_cfg=args.score_cfg,
            precision=torch.float16,
            inference_steps=args.inference_steps,
        )
    if isinstance(results, dict):
        results = [results]
    if len(results) != len(valid_records):
        raise ValueError("Docking output count mismatch")

    free_data, free_coords = [], []
    if args.unbound_support:
        from scripts.conformer_utils import pred_confs_batch

        for ligand, _ in datalist:
            mol = ligand.mol.with_hydrogens().sanitize()
            mol = mol.with_only_3d_conformer(np.zeros((len(mol.atoms), 3)))
            free_data.append(molecule_to_graph(
                mol, keep_mol=True, keep_smiles=True, keep_chirality=True
            ))
        with torch.inference_mode():
            free_coords = pred_confs_batch(
                args.docking_cfg, free_data, device,
                num_confs=args.unbound_num_confs,
                flow_temperature=args.unbound_scale,
                atom_budget=args.unbound_atom_budget,
            )

    ok_results = []
    for i, (rec, pred) in enumerate(zip(valid_records, results)):
        scores = pred["scores"]
        if isinstance(scores, torch.Tensor):
            scores_list = scores.detach().cpu().tolist()
        else:
            scores_list = list(scores)
        if len(scores_list) != args.num_confs or not np.isfinite(scores_list).all():
            raise ValueError(f"Invalid Ranker scores for {rec['mol_id']}")
        conformer_file = ""
        if args.unbound_support:
            ligand = datalist[i][0]
            numbers = ligand.x[:, 0].detach().cpu().numpy()
            bonds = ligand.chem_bond_index.detach().cpu().numpy()
            free_numbers, _ = molecular_graph(free_data[i].mol)
            heavy = np.flatnonzero(free_numbers != 1)
            heavy_mol = free_data[i].mol.without_hydrogens()
            # Explicitly map free heavy atoms to the docking graph atom order.
            mapping = cache_mapping(heavy_mol, numbers, bonds)
            free_heavy = free_coords[i].detach().cpu().numpy()[:, heavy]
            free_ordered = np.empty_like(free_heavy)
            free_ordered[:, mapping] = free_heavy
            conformer_file = f"conformations/{rec['source_index']:08d}.npz"
            path = Path(args.target_out) / conformer_file
            path.parent.mkdir(parents=True, exist_ok=True)
            dock_coords = pred["pred_coords"].detach().cpu().numpy()
            free_all = free_coords[i].detach().cpu().numpy()
            if not np.isfinite(dock_coords).all() or not np.isfinite(free_all).all():
                raise ValueError(f"Non-finite coordinates for {rec['mol_id']}")
            with path.with_suffix(".tmp").open("wb") as handle:
                np.savez_compressed(
                    handle, mol_id=rec["mol_id"], smiles=rec["smiles"],
                    atomic_numbers=numbers, bond_index=bonds,
                    docking_coords=dock_coords, ranker_scores=np.asarray(scores_list),
                    free_heavy_coords=free_ordered, free_coords=free_all,
                    free_atomic_numbers=free_numbers,
                )
            os.replace(path.with_suffix(".tmp"), path)
        ok_results.append(
            {
                "mol_id": rec["mol_id"],
                "smiles": rec["smiles"],
                "ranker_score": max(scores_list),
                "ranker_scores": scores_list,
                "status": "ok",
                "source_index": rec["source_index"],
                "label": rec.get("label"),
                "original_id": rec.get("original_id", rec["mol_id"]),
                "conformer_file": conformer_file,
            }
        )

    return ok_results + failed_results


def run_ligand_batch(records, protein_data, ligand_center, device, args):
    """Split failed batches to fit memory and isolate invalid ligands."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        result = score_ligand_batch(records, protein_data, ligand_center, device, args)
    except Exception as exc:
        error = repr(exc)
        traceback.print_exc()
    else:
        with (Path(args.target_out) / f"batches_{str(device).replace(':', '_')}.jsonl").open("a") as log:
            log.write(json.dumps(dict(
                batch_size=len(records), seconds=time.perf_counter() - started,
                peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            )) + "\n")
        return result
    # Leave the except block before retrying so failed tensors can be released.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(records) > 1:
        mid = len(records) // 2
        return (run_ligand_batch(records[:mid], protein_data, ligand_center, device, args)
                + run_ligand_batch(records[mid:], protein_data, ligand_center, device, args))
    rec = records[0]
    return [dict(mol_id=rec["mol_id"], smiles=rec["smiles"], source_index=rec["source_index"],
                 label=rec.get("label"), original_id=rec.get("original_id", rec["mol_id"]),
                 ranker_score="", ranker_scores="", status=f"failed: {error}")]


def write_results(results, out_csv: str):
    ok_results = [x for x in results if x["ranker_score"] != ""]
    failed_results = [x for x in results if x["ranker_score"] == ""]

    ok_results = sorted(ok_results, key=lambda x: x["ranker_score"], reverse=True)

    for rank, row in enumerate(ok_results, start=1):
        row["rank"] = rank
    for row in failed_results:
        row["rank"] = ""

    final_results = ok_results + failed_results

    with open(out_csv + ".tmp", "w", newline="", encoding="utf-8") as f:
        for row in final_results:
            if "ranker_scores" in row and isinstance(row["ranker_scores"], list):
                row["ranker_scores"] = ";".join(
                    repr(float(x)) for x in row["ranker_scores"]
                )
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "mol_id",
                "smiles",
                "ranker_score",
                "ranker_scores",
                "status",
                "source_index",
                "label",
                "original_id",
                "conformer_file",
            ],
        )
        writer.writeheader()
        writer.writerows(final_results)
    os.replace(out_csv + ".tmp", out_csv)


def prepare_target_output(target_dir, protein_path, center, records, args):
    if not records:
        raise ValueError(f"No ligand records in {target_dir}")
    if len({record["mol_id"] for record in records}) != len(records):
        raise ValueError(f"Duplicate ligand IDs in {target_dir}")
    args.target_out = str(Path(args.out_dir).expanduser() / Path(target_dir).name)
    root = Path(args.target_out)
    root.mkdir(parents=True, exist_ok=True)
    metadata = dict(args.run_metadata)
    metadata.update(
        protein_sha256=hashlib.sha256(Path(protein_path).read_bytes()).hexdigest(),
        pocket_center=center.tolist(),
        ligands_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        num_ligands=len(records),
    )
    manifest = root / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError(f"Run settings changed; choose a new --out_dir: {root}")
    scores = root / "scores.csv"
    if scores.exists():
        if not manifest.exists():
            raise ValueError(f"Existing scores have no run.json; choose a new --out_dir: {root}")
        with scores.open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) == len(records) and all(row["status"] == "ok" for row in rows):
            if not args.unbound_support or all(
                row.get("conformer_file") and (root / row["conformer_file"]).is_file()
                for row in rows
            ):
                print(f"[SKIP] {root.name} already complete")
                return True
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return False


def ligand_worker(device, ligand_task_queue, result_queue, worker_args):
    """
    每个 GPU worker:
    - 只服务当前 target
    - 启动时加载一次 protein 和 ligand_center
    - 不断从队列中取 ligand 做打分
    """
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.set_num_threads(1)
        protein_data = prepare_protein_data(worker_args["protein_path"]).to(device)
        ligand_center = worker_args["ligand_center"]

        while True:
            task = ligand_task_queue.get()
            if task is None:
                break

            try:
                batch_results = run_ligand_batch(
                    records=task["records"],
                    protein_data=protein_data,
                    ligand_center=ligand_center,
                    device=device,
                    args=worker_args["args"],
                )
                result = {
                    "batch_id": task["batch_id"],
                    "results": batch_results,
                }
            except Exception as e:
                traceback.print_exc()
                result = {
                    "batch_id": task["batch_id"],
                    "results": [
                        {
                            "mol_id": rec["mol_id"],
                            "smiles": rec["smiles"],
                            "ranker_score": "",
                            "ranker_scores": "",
                            "status": f"failed: {repr(e)}",
                            "source_index": rec["source_index"],
                            "label": rec.get("label"),
                            "original_id": rec.get("original_id", rec["mol_id"]),
                        }
                        for rec in task["records"]
                    ],
                }

            result_queue.put(pickle.dumps(result))

    except Exception as e:
        traceback.print_exc()
        result_queue.put(
            pickle.dumps(
                {
                    "__worker_init_failed__": True,
                    "ranker_score": "",
                    "status": f"worker init failed on {device}: {repr(e)}",
                }
            )
        )


def process_target_single_gpu(target_dir: str, device, args):
    target_name = os.path.basename(os.path.abspath(target_dir))

    protein_path, ligand_center, mol_records = load_target(target_dir, args.input_format)
    if args.max_ligands is not None:
        mol_records = mol_records[:args.max_ligands]
    if prepare_target_output(target_dir, protein_path, ligand_center, mol_records, args):
        return
    print(f"Scoring {target_name}: {len(mol_records)} ligands on {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    protein_data = prepare_protein_data(protein_path).to(device)

    out_dir = os.path.join(args.out_dir, target_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "scores.csv")

    batch_size = max(1, int(getattr(args, "ligand_batch_size", 64)))
    batches = list(chunked(mol_records, batch_size))
    results = []
    for batch in tqdm(batches, desc=f"Scoring {target_name}"):
        batch_results = run_ligand_batch(
            records=batch,
            protein_data=protein_data,
            ligand_center=ligand_center,
            device=device,
            args=args,
        )
        results.extend(batch_results)

    write_results(results, out_csv)
    print(f"Saved: {out_csv}")
    return sum(row["status"] != "ok" for row in results)


def process_target_multi_gpu(target_dir: str, args):
    """
    一个 target 一个 target 地执行；
    对于当前 target，所有 GPU 一起处理这个 target 的所有 ligands。
    """
    target_name = os.path.basename(os.path.abspath(target_dir))

    protein_path, ligand_center, mol_records = load_target(target_dir, args.input_format)
    if args.max_ligands is not None:
        mol_records = mol_records[:args.max_ligands]
    if prepare_target_output(target_dir, protein_path, ligand_center, mol_records, args):
        return
    print(f"Scoring {target_name}: {len(mol_records)} ligands on {args.device.gpus}")
    out_csv = str(Path(args.target_out) / "scores.csv")

    gpu_ids = args.device.gpus
    ligand_task_queue = mp.Queue()
    result_queue = mp.Queue()

    worker_args = {
        "protein_path": protein_path,
        "ligand_center": ligand_center,
        "args": args,
    }

    gpu_processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=ligand_worker,
            args=(
                torch.device(f"cuda:{gpu_id}"),
                ligand_task_queue,
                result_queue,
                worker_args,
            ),
        )
        p.start()
        gpu_processes.append(p)

    batch_size = max(1, int(getattr(args, "ligand_batch_size", 64)))
    ligand_batches = list(chunked(mol_records, batch_size))
    for batch_id, batch in enumerate(ligand_batches):
        ligand_task_queue.put(
            {
                "batch_id": batch_id,
                "records": batch,
            }
        )

    for _ in gpu_processes:
        ligand_task_queue.put(None)

    results = []
    received = 0
    try:
        with tqdm(total=len(ligand_batches), desc=f"Collecting {target_name}") as progress:
            while received < len(ligand_batches):
                try:
                    result = pickle.loads(result_queue.get(timeout=1))
                except Empty:
                    if any(p.exitcode not in (None, 0) for p in gpu_processes) or not any(p.is_alive() for p in gpu_processes):
                        raise RuntimeError(f"GPU workers exited before completing {target_name}")
                    continue
                if result.get("__worker_init_failed__", False):
                    raise RuntimeError(result["status"])
                results.extend(result["results"])
                received += 1
                progress.update(1)
        for p in gpu_processes:
            p.join()
    finally:
        for p in gpu_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        ligand_task_queue.cancel_join_thread()
        ligand_task_queue.close()
        result_queue.close()

    write_results(results, out_csv)
    print(f"Saved: {out_csv}")
    return sum(row["status"] != "ok" for row in results)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--dekois_root", "--data-root", type=str, required=True,
        help="Dataset root, for example ./datasets/DEKOIS2 or ./datasets/DUDE-Z",
    )
    parser.add_argument("--input-format", "--input_format", choices=["dekois", "dudez"], default="dekois")
    parser.add_argument("--unbound-support", "--unbound_support", action="store_true")
    parser.add_argument("--unbound-num-confs", type=int, default=20)
    parser.add_argument("--unbound-scale", type=float, default=2.0)
    parser.add_argument("--unbound-atom-budget", type=int, default=128000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ligands", type=int, default=None, help="Limit each target for smoke tests")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="If set, only run one target, e.g. pdk1",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_confs", type=int, default=40)
    parser.add_argument(
        "--ligand_batch_size",
        type=int,
        default=32,
        help="Number of ligands scored together for one pocket on one GPU",
    )

    args_cli = parser.parse_args()
    if min(args_cli.num_confs, args_cli.ligand_batch_size, args_cli.unbound_num_confs,
           args_cli.unbound_atom_budget) < 1:
        parser.error("Conformer counts, batch size and atom budget must be positive")
    if args_cli.max_ligands is not None and args_cli.max_ligands < 1:
        parser.error("--max-ligands must be positive")
    if not np.isfinite(args_cli.unbound_scale) or args_cli.unbound_scale <= 0:
        parser.error("--unbound-scale must be finite and positive")
    if args_cli.unbound_support and args_cli.num_confs < 5:
        parser.error("Unbound support requires --num_confs >= 5")

    if args_cli.config == "dock_config.toml" and not Path(args_cli.config).exists():
        args_cli.config = "config/dock_config.toml"
    args = Namespace(**load_toml_cfg(args_cli.config, overrides=vars(args_cli)))

    docking_section = getattr(args, "docking_joint", None) or getattr(args, "docking", None)
    if docking_section is None:
        parser.error("Configuration requires [docking] or [docking_joint]")
    args.inference_steps = docking_section.get("inference_steps", 50)
    args.docking_cfg = load_toml_cfg(docking_section.config)
    args.score_cfg = load_toml_cfg(args.score.config)
    args.out_dir = str(Path(args.out_dir).expanduser())
    args.run_metadata = dict(
        input_format=args.input_format, unbound_support=args.unbound_support,
        num_confs=args.num_confs, inference_steps=args.inference_steps,
        seed=args.seed, ligand_batch_size=args.ligand_batch_size,
        unbound_num_confs=args.unbound_num_confs, unbound_scale=args.unbound_scale,
        unbound_atom_budget=args.unbound_atom_budget, precision="float16",
        pocket_center_definition="reference heavy-atom coordinate mean",
    )
    for name, cfg in [("docking", args.docking_cfg), ("ranker", args.score_cfg)]:
        checkpoint = Path(cfg.paths.checkpoint).expanduser()
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        args.run_metadata[name] = dict(checkpoint=str(checkpoint), sha256=digest, config=dict(cfg))
        print(f"{name}: {checkpoint} SHA256={digest}")

    data_root = args.dekois_root

    if args.target is not None:
        target_dirs = [os.path.join(data_root, args.target)]
    else:
        target_dirs = [
            os.path.join(data_root, d)
            for d in sorted(os.listdir(data_root))
            if os.path.isdir(os.path.join(data_root, d))
            and (Path(data_root) / d / (
                "active_decoys.smi" if args.input_format == "dekois" else "ligands.smi"
            )).is_file()
        ]
        # random.seed(20)
        # target_dirs = random.sample(target_dirs, 20)
        target_dirs = list(reversed(target_dirs))

    if not target_dirs:
        parser.error("No target directories found")

    failed_ligands = 0
    if args.device.use_cpu or len(args.device.gpus) == 1:
        device = torch.device(
            "cpu" if args.device.use_cpu else f"cuda:{args.device.gpus[0]}"
        )
        print(f"Using device: {device}")

        for target_dir in tqdm(target_dirs, desc="Targets"):
            failed_ligands += process_target_single_gpu(target_dir, device, args) or 0

    else:
        mp.set_start_method("spawn", force=True)
        print(f"Using multi-GPU target-by-target mode: {args.device.gpus}")

        for target_dir in tqdm(target_dirs, desc="Targets"):
            failed_ligands += process_target_multi_gpu(target_dir, args) or 0

    if failed_ligands:
        raise RuntimeError(f"{failed_ligands} ligands failed; details are retained in scores.csv")
    print(f"All screening results have been saved to {Path(args.out_dir).resolve()}")
