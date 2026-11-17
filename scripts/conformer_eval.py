"""
python ./scripts/conformer_eval.py -c config/Joint/Joint_all.toml
"""

import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from scripts.conformer_utils import pairwise_conformer_rmsds, pred_confs

from data.GEOM_SDEGen import GEOM_SDEGen
from data.utils import get_confs_from_coords
from models.utils import (
    chirality_orientation_loss,
    parse_args,
)


cfg = parse_args()
if cfg.task == "JointDockingConformation" and cfg.dataset.get("conf", None) is not None:
    for key, value in cfg.dataset.conf.items():
        cfg.dataset[key] = value
dataset_test = GEOM_SDEGen(
    cfg.dataset.conf.path,
    split="test",
    # radius_cutoff=cfg.dataset.radius_cutoff,
    radius_cutoff=cfg.dataset.conf.radius_cutoff,
    mode="inference",
)

device = torch.device("cpu" if cfg.device.use_cpu else f"cuda:{cfg.device.gpus[0]}")

rmsd_confusion_matrices = []
covr_scores = []
matr_scores = []
covp_scores = []
matp_scores = []

per_mol_metrics = []

threshold = 0.75

# 随机选择 1000 个不重复的样本
random.seed(42)
random_indices = random.sample(range(len(dataset_test)), len(dataset_test))
# random_indices = random.sample(range(len(dataset_test)), 3)

chirality_error_count = 0
chirality_error_count_all = 0

for idx in tqdm(random_indices):
    data = dataset_test[idx]

    gt_mol = data.mol
    gt_mols = get_confs_from_coords(
        gt_mol, data.confs_pos.view(data.num_confs, -1, 3), has_hydrogens=True
    )

    gt_coords = data.confs_pos.view(data.num_confs, -1, 3)[0]

    pred_coords = pred_confs(
        cfg, data, device, data.num_confs * 2, flow_temperature=0.9, mol=data.mol
    )

    chirality_error_count_all += len(pred_coords)
    for idx, pred_coords_ in enumerate(pred_coords):
        chirality_loss = chirality_orientation_loss(
            pred_coords_, data.chiral_neighbors, data.chirality
        )
        if chirality_loss > 0.3:
            chirality_error_count += 1
            print("chirality error!", chirality_loss)

    pred_mols = get_confs_from_coords(data.mol, pred_coords, has_hydrogens=True)

    rmsd_conf_mat = pairwise_conformer_rmsds(pred_mols, gt_mols)
    print(rmsd_conf_mat.shape)

    if np.isnan(np.sum(rmsd_conf_mat)):
        print("NAN in RMSD confusion matrix!")

    rmsd_ref_min = np.nanmin(rmsd_conf_mat, axis=-1)  # np (num_ref, )
    rmsd_gen_min = np.nanmin(rmsd_conf_mat, axis=0)  # np (num_gen, )
    rmsd_cov_thres = rmsd_ref_min.reshape(-1, 1) <= threshold  # np (num_ref, )
    rmsd_jnk_thres = rmsd_gen_min.reshape(-1, 1) <= threshold  # np (num_gen, )

    matr_scores.append(rmsd_ref_min.mean())
    covr_scores.append(rmsd_cov_thres.mean())
    matp_scores.append(rmsd_gen_min.mean())
    covp_scores.append(rmsd_jnk_thres.mean())

    per_mol_metrics.append(
        {
            "covr": rmsd_cov_thres.mean(),
            "covp": rmsd_jnk_thres.mean(),
            "matr": rmsd_ref_min.mean(),
            "matp": rmsd_gen_min.mean(),
        }
    )


covr_scores = np.array(covr_scores)  # np (num_mols, )
matr_scores = np.array(matr_scores)  # np (num_mols, )
covp_scores = np.array(covp_scores)  # np (num_mols, )
matp_scores = np.array(matp_scores)  # np (num_mols, )

covr_mean = np.mean(covr_scores)
covr_median = np.median(covr_scores)
matr_mean = np.nanmean(matr_scores)
matr_median = np.nanmedian(matr_scores)
covp_mean = np.mean(covp_scores)
covp_median = np.median(covp_scores)
matp_mean = np.nanmean(matp_scores)
matp_median = np.nanmedian(matp_scores)
print(
    f"chirality_error_count: {chirality_error_count}/{chirality_error_count_all}, threshold: {threshold}"
)
print("covr-mean:", covr_mean * 100, "covr-median:", covr_median * 100)
print("matr-mean:", matr_mean, "matr-median:", matr_median)
print("covp-mean:", covp_mean * 100, "covp-median:", covp_median * 100)
print("matp-mean:", matp_mean, "matp-median:", matp_median)
output_dir = Path("./output/conformer_eval")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "per_mol_metrics.csv"
pd.DataFrame(per_mol_metrics).to_csv(output_path, index=False)
print(f"Per-molecule metrics saved to {output_path.resolve()}")
