"""
python ./scripts/docking_eval.py
"""

import os
import pickle
import sys
import traceback
import json
import random
import hashlib
import time
from pathlib import Path
from queue import Empty

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.utils import load_toml_cfg
from data.utils import get_confs_from_coords, remove_all_hydrogens
from scripts.docking_utils import (
    FAIL_RULES,
    PB_validate_test,
    calc_rmsd,
    pred_confs,
)


docking_cfg = load_toml_cfg("config/Joint/Joint_all.toml")

pocket_cfg = docking_cfg
pocket_ranker_cfg = pocket_cfg

score_cfg = load_toml_cfg("config/PDBbind/Pose_ranker.toml")
GPU_IDS = docking_cfg.device.gpus
OUTPUT_DIR = Path("./output/pdbbind_docking")
EVAL_SEED = None
perfect_selection = False

from data.PDBbind2020 import PDBbind2020

test_dataset = PDBbind2020(
    "./datasets/PDBbind2020",
    split="test",
    mode="inference",
)

# =========================
# GPU worker
# =========================


def gpu_worker(gpu_id, task_queue, result_queue):
    """
    Each GPU worker:
    - loads model once
    - receives index
    - runs inference
    - sends result to CPU queue
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    while True:
        idx = task_queue.get()

        # stop signal
        if idx is None:
            break

        try:
            if EVAL_SEED is not None:
                random.seed(EVAL_SEED + idx)
                np.random.seed(EVAL_SEED + idx)
                torch.manual_seed(EVAL_SEED + idx)
                torch.cuda.manual_seed(EVAL_SEED + idx)
            data = test_dataset[idx]

            with torch.inference_mode():
                result = pred_confs(
                    data,
                    device,
                    num_confs=40,
                    pocket="pocket_center",
                    # pocket="blind",
                    docking_cfg=docking_cfg,
                    pocket_cfg=pocket_cfg,
                    pocket_ranker_cfg=pocket_ranker_cfg,
                    calculate_score=not perfect_selection,
                    pose_ranker_cfg=score_cfg,
                    initial_shift_range=1.0,
                    precision=torch.float16,
                )

            result_queue.put((idx, result))

        except Exception as e:
            print(f"Error in GPU worker {gpu_id}: {e}")
            traceback.print_exc()
            result_queue.put((idx, {"error": str(e)}))


# =========================
# CPU worker
# =========================


def cpu_worker(result_queue, final_queue):
    """
    Each CPU worker:
    - receives prediction
    - calculates RMSD
    - sends final small result
    """

    while True:
        item = result_queue.get()

        # stop signal
        if item is None:
            break

        idx, pred = item

        # error from GPU
        if "error" in pred:
            final_queue.put(pickle.dumps({"idx": idx, "error": pred["error"]}))
            continue

        # read dataset again (shared memory)
        ligand_data, protein_data = test_dataset[idx]
        gt_mol = ligand_data.mol
        gt_mol = remove_all_hydrogens(gt_mol)

        pred_mols = get_confs_from_coords(
            gt_mol,
            pred["pred_coords"].view(40, -1, 3),
        )

        rmsds = torch.tensor(
            [
                calc_rmsd(pred_mols[i], gt_mol, remove_Hs=True)
                for i in range(len(pred_mols))
            ]
        )

        if perfect_selection:
            print(rmsds)
            sorted_idx = torch.argsort(rmsds)
        else:
            scores_flat = pred["scores"]
            sorted_idx = torch.argsort(scores_flat, descending=True)
        sorted_rmsds = rmsds[sorted_idx.cpu()]

        PB_valid = PB_validate_test(
            pred_mols[sorted_idx[0]],
            gt_mol,
            protein_data.path,
        )
        print(f"PB_valid: {PB_valid}")

        final_queue.put(
            pickle.dumps(
                {
                    "idx": idx,
                    "protein_path": str(protein_data.path),
                    "rmsd_min": torch.min(rmsds).item(),
                    "rmsd_top1": sorted_rmsds[0].item(),
                    "rmsd_top5": sorted_rmsds[:5].cpu(),
                    "rmsd_top5_min": torch.min(sorted_rmsds[:5]).item(),
                    "rmsd_top5_perfect": torch.sort(rmsds)[0][:5].cpu(),
                    "rmsd_all": rmsds.cpu(),
                    "PB_valid": PB_valid.cpu(),
                }
            )
        )


# =========================
# Main
# =========================

if __name__ == "__main__":
    print(test_dataset)
    mp.set_start_method("spawn", force=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metadata = {
        "checkpoint": str(docking_cfg.paths.checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(docking_cfg.paths.checkpoint).read_bytes()).hexdigest(),
        "config": docking_cfg, "gpu_ids": GPU_IDS, "seed": EVAL_SEED,
        "num_complexes": len(test_dataset), "num_confs": 40,
        "perfect_selection": perfect_selection, "pocket": "pocket_center",
        "precision": "float16 autocast",
        "inference_mode": True,
        "started_unix": started,
        "initial_shift_std_A": 1.0,
        "terminal_updates": 20,
        "integration_schedule_steps": 50,
        "actual_model_calls_per_pose": 30 + 20,
        "inference_scales": {
            "rotation": 1.5,
            "shift": 3.0,
            "conformation": 2.5,
        },
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ---------------------------------
    # Create dataset ONCE
    # ---------------------------------

    NUM_GPUS = len(GPU_IDS)
    NUM_CPU_WORKERS = 16

    gpu_task_queue = mp.Queue()
    gpu_to_cpu_queue = mp.Queue(maxsize=NUM_GPUS * 8)
    final_queue = mp.Queue()

    # ---------------------------------
    # Start GPU workers
    # ---------------------------------
    gpu_processes = []
    for gpu_id in tqdm(GPU_IDS, desc="Starting GPU workers"):
        p = mp.Process(
            target=gpu_worker, args=(gpu_id, gpu_task_queue, gpu_to_cpu_queue)
        )
        p.start()
        gpu_processes.append(p)

    # ---------------------------------
    # Start CPU workers
    # ---------------------------------
    cpu_processes = []
    for _ in tqdm(range(NUM_CPU_WORKERS), desc="Starting CPU workers"):
        p = mp.Process(target=cpu_worker, args=(gpu_to_cpu_queue, final_queue))
        p.start()
        cpu_processes.append(p)

    # ---------------------------------
    # Send all tasks
    # ---------------------------------
    for idx in tqdm(range(len(test_dataset)), desc="Sending tasks"):
        gpu_task_queue.put(idx)

    # send stop signal to GPU
    for _ in gpu_processes:
        gpu_task_queue.put(None)

    # ---------------------------------
    # Collect results
    # ---------------------------------
    results = []
    for _ in tqdm(range(len(test_dataset)), desc="Collecting results"):
        while True:
            try:
                result = pickle.loads(final_queue.get(timeout=30))
                break
            except Empty:
                if any(p.exitcode not in (None, 0) for p in gpu_processes + cpu_processes):
                    for p in gpu_processes + cpu_processes:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError("Evaluation worker failed; partial results retained")
        results.append(result)
        torch.save(result, OUTPUT_DIR / f"result_{result['idx']:04d}.pt")

    # ---------------------------------
    # Wait GPU finish
    # ---------------------------------
    for p in gpu_processes:
        p.join()

    # ---------------------------------
    # Stop CPU workers
    # ---------------------------------
    for _ in cpu_processes:
        gpu_to_cpu_queue.put(None)

    for p in cpu_processes:
        p.join()

    # ============================================================
    # 结果统计（主进程）
    # ============================================================

    valid_results = [r for r in results if "error" not in r]
    torch.save(sorted(results, key=lambda r: r["idx"]), OUTPUT_DIR / "results.pt")
    metadata.update(elapsed_seconds=time.time() - started, successful=len(valid_results), failed=len(results)-len(valid_results))
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rmsd_mins = torch.tensor([r["rmsd_min"] for r in valid_results])
    rmsd_top1 = torch.tensor([r["rmsd_top1"] for r in valid_results])
    rmsd_top5 = torch.stack([torch.tensor(r["rmsd_top5"]) for r in valid_results])
    rmsd_top5_min = torch.tensor([r["rmsd_top5_min"] for r in valid_results])
    rmsd_all = torch.stack([r["rmsd_all"] for r in valid_results])
    rmsd_top5_perfect = torch.stack([r["rmsd_top5_perfect"] for r in valid_results])
    # pdb_ids = [r["pdb_id"] for r in valid_results]

    print(
        "the rmsd below 1.0:",
        torch.sum(rmsd_mins < 1.0).item(),
        "/",
        len(rmsd_mins),
        "(",
        torch.sum(rmsd_mins < 1.0).item() / len(rmsd_mins) * 100,
        "%)",
    )
    print(
        "the rmsd below 2.0:",
        torch.sum(rmsd_mins < 2.0).item(),
        "/",
        len(rmsd_mins),
        "(",
        torch.sum(rmsd_mins < 2.0).item() / len(rmsd_mins) * 100,
        "%)",
    )
    print(
        "the top1 rmsd below 1.0:",
        torch.sum(rmsd_top1 < 1.0).item(),
        "/",
        len(rmsd_top1),
        "(",
        torch.sum(rmsd_top1 < 1.0).item() / len(rmsd_top1) * 100,
        "%)",
    )
    print(
        "the top1 rmsd below 2.0:",
        torch.sum(rmsd_top1 < 2.0).item(),
        "/",
        len(rmsd_top1),
        "(",
        torch.sum(rmsd_top1 < 2.0).item() / len(rmsd_top1) * 100,
        "%)",
    )
    print("the top1 rmsd median:", torch.nanmedian(rmsd_top1))
    print(
        "the top5 rmsd below 1.0:",
        torch.sum(rmsd_top5_min < 1.0).item(),
        "/",
        len(rmsd_top5_min),
        "(",
        torch.sum(rmsd_top5_min < 1.0).item() / len(rmsd_top5_min) * 100,
        "%)",
    )
    print(
        "the top5 rmsd below 2.0:",
        torch.sum(rmsd_top5_min < 2.0).item(),
        "/",
        len(rmsd_top5_min),
        "(",
        torch.sum(rmsd_top5_min < 2.0).item() / len(rmsd_top5_min) * 100,
        "%)",
    )
    print("the top5 rmsd median:", torch.nanmedian(rmsd_top5_min))
    print("mean_rmsd:", torch.nanmean(rmsd_all))
    print("the top5 rmsd perfect:", torch.nanmean(rmsd_top5_perfect))
    PB_valids = torch.stack([r["PB_valid"] for r in valid_results])
    print(
        "PoseBusters valid:",
        torch.sum(PB_valids.all(dim=-1)).item(),
        "/",
        PB_valids.shape[0],
        "(",
        torch.sum(PB_valids.all(dim=-1)).item() / PB_valids.shape[0] * 100,
        "%)",
    )

    PB_failed_count = torch.sum(~PB_valids, dim=-0)
    for i, flag in enumerate(PB_failed_count):
        print(f"{flag} - {FAIL_RULES[i]}")

    # 只考虑 0~40 范围
    rmsd_top1_clipped = torch.clamp(rmsd_top1, max=40).numpy()

    # 设置柱状图的 bin，这里每 1 Å 一个 bin
    bins = range(0, 41, 1)

    plt.figure(figsize=(10, 6))
    plt.hist(rmsd_top1_clipped, bins=bins, color="skyblue", edgecolor="black")
    plt.title("Top1 RMSD Distribution (0~40 Å)")
    plt.xlabel("RMSD (Å)")
    plt.ylabel("Count")
    plt.xticks(range(0, 41, 2))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 保存图
    plt.savefig(OUTPUT_DIR / "rmsd_top1_distribution.png", dpi=300)
    plt.close()

    print("RMSD top1 distribution plot saved as rmsd_top1_distribution_7.png")
