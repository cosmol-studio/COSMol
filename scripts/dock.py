"""
python ./scripts/dock.py -c config/dock_config.toml --protein_path input/6fe5_protein.pdb --ligand_description input/6fe5_ligand.sdf
"""

import os
import pickle
import re
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path
from queue import Empty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import polars as pl
import toml_rs
import torch
import torch.multiprocessing as mp
from data.utils import (
    get_confs_from_coords,
    read_ligand,
    molecule_to_graph,
    get_pocket_center,
    protein_to_graph,
    save_sdf,
)
from tqdm import tqdm

from models.utils import load_toml_cfg
from scripts.docking_utils import pred_confs


def run_task(task, args, device, precision):
    docking_joint_cfg = getattr(args, "docking_joint_cfg", None)
    score_cfg = getattr(args, "score_cfg", None)
    use_pocket = getattr(args, "use_pocket") or getattr(
        args, "use_ligand_center_as_pocket"
    )

    mol = read_ligand(task["ligand_description"])
    ligand_data = molecule_to_graph(
        mol,
        keep_mol=True,
        keep_smiles=True,
        keep_chirality=True
    )
    if task["protein_path"].endswith(".pdb"):
        protein_data = protein_to_graph(
            os.path.expanduser(task["protein_path"]), type="pdb"
        )
    elif task["protein_path"].endswith(".cif"):
        protein_data = protein_to_graph(
            os.path.expanduser(task["protein_path"]), type="cif"
        )
    else:
        raise ValueError(f"Unsupported protein file format: {task['protein_path']}")

    ligand_data.remove_hydrogens()
    protein_data.remove_hydrogens()

    data = (ligand_data, protein_data)

    if use_pocket:
        pocket_pos = get_pocket_center(task["pocket_description"])

    result = pred_confs(
        data,
        device,
        num_confs=args.num_conformations,
        pocket="given" if use_pocket else "blind",
        docking_cfg=docking_joint_cfg,
        pocket_cfg=docking_joint_cfg,
        pocket_ranker_cfg=docking_joint_cfg,
        calculate_score=True,
        pose_ranker_cfg=score_cfg,
        pocket_pos_list=[pocket_pos] if use_pocket else None,
        precision=precision,
        optimized=True,
        inference_steps=args.docking_joint.get("inference_steps", 50),
    )

    pred_coords = result["pred_coords"].view(args.num_conformations, -1, 3)
    scores = result.get("scores", None)
    if scores is None:
        scores = torch.zeros(pred_coords.shape[0], device=pred_coords.device)

    sorted_indices = scores.argsort(descending=True)
    pred_mols = get_confs_from_coords(ligand_data.mol, pred_coords)

    out_dir = os.path.join(args.out_dir, task["complex_name"])
    os.makedirs(out_dir, exist_ok=True)

    n = len(sorted_indices)
    width = max(2, len(str(n)))
    previous_files = {
        path for path in Path(out_dir).glob("rank*.sdf")
        if re.match(r"rank\d+_", path.name)
    }
    written_files = set()

    for i, idx in enumerate(sorted_indices):
        rank_str = str(i + 1).zfill(width)
        path = Path(out_dir) / f"rank{rank_str}_{scores[idx]:.3f}.sdf"
        save_sdf(pred_mols[idx], path)
        written_files.add(path)
    for path in previous_files - written_files:
        path.unlink()
    return {"complex_name": task["complex_name"], "out_dir": out_dir}


def gpu_worker(device, task_queue, result_queue, args):
    """
    Each GPU worker:
    - loads model once
    - receives index
    - runs inference
    - sends result to CPU queue
    """
    precision = torch.float16 if args.precision == "float16" else torch.float32
    while True:
        task = task_queue.get()

        if task is None:
            break

        try:
            result = run_task(task, args, device, precision)
        except Exception as exc:
            traceback.print_exc()
            result = {"complex_name": task["complex_name"], "error": str(exc)}
        result_queue.put(pickle.dumps(result))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="config/dock_config.toml"
    )
    parser.add_argument("--protein_ligand_csv", type=str, default=None)
    parser.add_argument("--use_pocket", action="store_true", default=None)
    parser.add_argument("--use_ligand_center_as_pocket", action="store_true", default=None)
    parser.add_argument("--complex_name", type=str, default=None)
    parser.add_argument("--protein_path", type=str, default=None)
    parser.add_argument("--ligand_description", type=str, default=None)
    parser.add_argument("--pocket_description", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    args_cli = parser.parse_args()
    cli_dict = {k: v for k, v in vars(args_cli).items() if v is not None}
    args = load_toml_cfg(args_cli.config, overrides=cli_dict)
    args.use_pocket = args.get("use_pocket", False)
    args.use_ligand_center_as_pocket = args.get("use_ligand_center_as_pocket", False)
    args.out_dir = args.get("out_dir", "./output")
    if not args.get("protein_ligand_csv") and not (
        args.get("protein_path") and args.get("ligand_description")
    ):
        parser.error("Provide --protein_ligand_csv or both --protein_path and --ligand_description")

    print("=" * 40)
    print(" Final Configuration ")
    print("=" * 40)
    print(toml_rs.dumps(vars(args)))
    print("=" * 40)

    args.docking_joint_cfg = load_toml_cfg(args.docking_joint.config)
    args.score_cfg = load_toml_cfg(args.score.config)

    tasks = []

    # ------------------------
    # Build tasks
    # ------------------------
    if args.get("protein_ligand_csv", None) is not None:
        df = pl.read_csv(args.protein_ligand_csv)

        tasks = [
            {
                "complex_name": row["complex_name"] or f"complex_{i}",
                "protein_path": row["protein_path"],
                "ligand_description": row["ligand_description"],
                "pocket_description": (
                    row["ligand_description"]
                    if args.use_ligand_center_as_pocket
                    else row.get("pocket_description")
                ),
            }
            for i, row in enumerate(df.iter_rows(named=True))
        ]

    else:
        tasks = [
            {
                "complex_name": args.get("complex_name") or "complex_0",
                "protein_path": args.protein_path,
                "ligand_description": args.ligand_description,
                "pocket_description": (
                    args.ligand_description
                    if args.use_ligand_center_as_pocket
                    else args.get("pocket_description")
                ),
            }
        ]

    # ------------------------
    # Apply input_dir
    # ------------------------
    if args.get("input_dir", None) is not None:
        base_dir = args.input_dir

        for task in tasks:
            task["protein_path"] = os.path.join(base_dir, task["protein_path"])
            if Path(task["ligand_description"]).suffix.lower() in (".sdf", ".mol2", ".mol"):
                task["ligand_description"] = os.path.join(base_dir, task["ligand_description"])
            task["pocket_description"] = (
                os.path.join(base_dir, task["pocket_description"])
                if task["pocket_description"]
                else None
            )

    precision = torch.float16 if args.precision == "float16" else torch.float32
    print(f"Using precision: {precision}")

    if args.device.use_cpu or len(args.device.gpus) == 1 or len(tasks) == 1:
        device = torch.device(
            "cpu" if args.device.use_cpu else f"cuda:{args.device.gpus[0]}"
        )

        for task in tqdm(tasks):
            run_task(task, args, device, precision)

    else:
        mp.set_start_method("spawn", force=True)
        device = torch.device("cpu")

        GPU_IDS = args.device.gpus
        NUM_GPUS = len(GPU_IDS)

        gpu_task_queue = mp.Queue()
        result_queue = mp.Queue()

        gpu_processes = []
        for gpu_id in tqdm(GPU_IDS, desc="Starting GPU workers"):
            p = mp.Process(
                args=(
                    torch.device(f"cuda:{gpu_id}"),
                    gpu_task_queue,
                    result_queue,
                    args,
                ),
                target=gpu_worker,
            )
            p.start()
            gpu_processes.append(p)

        # ---------------------------------
        # Send all tasks
        # ---------------------------------
        for task in tqdm(tasks, desc="Sending tasks"):
            gpu_task_queue.put(task)

        # send stop signal to GPU
        for _ in gpu_processes:
            gpu_task_queue.put(None)

        # ---------------------------------
        # Collect results
        # ---------------------------------
        results = []
        try:
            for _ in tqdm(range(len(tasks)), desc="Collecting results"):
                while True:
                    try:
                        results.append(pickle.loads(result_queue.get(timeout=1)))
                        break
                    except Empty:
                        if any(p.exitcode not in (None, 0) for p in gpu_processes) or all(
                            not p.is_alive() for p in gpu_processes
                        ):
                            raise RuntimeError("Docking workers exited before all results arrived")
            for p in gpu_processes:
                p.join()
        finally:
            for p in gpu_processes:
                if p.is_alive():
                    p.terminate()
                p.join()
        failed = [result for result in results if "error" in result]
        if failed:
            raise RuntimeError(f"Docking failed for {len(failed)} complexes: {failed}")
    print(f"All results have been saved to {os.path.abspath(args.out_dir)}")
