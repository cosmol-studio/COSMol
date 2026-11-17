from argparse import ArgumentParser
from rdkit import Chem
import polars as pl
import glob
import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import align_poses_to_reference_site

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.utils import resolve_path
from reproduce.evaluate_docking import collect_ranked_predictions
from scripts.docking_utils import (
    calc_rmsd,
    PB_validate_test,
    FAIL_RULES,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--protein_ligand_csv_apo', type=str, required=True)
    parser.add_argument('--protein_ligand_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default=None)
    args = parser.parse_args()
    args.input_dir = resolve_path(args.input_dir)
    args.out_dir = resolve_path(args.out_dir, output=True)

    out_dir = args.out_dir

    df_apo = pl.read_csv(args.protein_ligand_csv_apo)
    df = pl.read_csv(args.protein_ligand_csv)

    tasks = [
        {
            "complex_name": row["complex_name"] or f"complex_{i}",
            "protein_path": row["protein_path"],
            "protein_path_apo": df_apo.filter(
                df_apo["complex_name"] == row["complex_name"]
            )["protein_path"].item(),
            "ligand_description": row["ligand_description"],
        }
        for i, row in enumerate(df.iter_rows(named=True))
    ]

    for task in tasks:
        for field in ("protein_path", "protein_path_apo", "ligand_description"):
            path = os.path.expanduser(task[field])
            if args.input_dir is not None:
                path = os.path.join(args.input_dir, path)
            task[field] = resolve_path(path)

    perfect_selection = False

    rmsd_top1 = []
    rmsd_top5 = []
    rmsd_top5_min = []
    PB_valids = []
    failed_complexes = []
    for task in tqdm(tasks):
        complex_name = task["complex_name"]
        print(complex_name)
        gt_mol = Chem.MolFromMolFile(os.path.expanduser(task["ligand_description"]))

        if not perfect_selection:
            mol_pred_paths = collect_ranked_predictions(out_dir, complex_name, False)

            if len(mol_pred_paths) < 1:
                failed_complexes.append(complex_name)
                print("not ranked prediction found.")
                continue

            mol_preds_aligned, aligned_protein_path = align_poses_to_reference_site(
                task["protein_path"],
                task["ligand_description"],
                task["protein_path_apo"],
                mol_pred_paths
            )

            rmsds = torch.stack([calc_rmsd(mol_pred, gt_mol, remove_Hs=True) for mol_pred in mol_preds_aligned], dim=0)

            # from concurrent.futures import ThreadPoolExecutor
            # from functools import partial

            # func = partial(calc_rmsd, gt_mol=gt_mol, remove_Hs=True)

            # with ThreadPoolExecutor() as executor:
            #     rmsds = list(executor.map(func, mol_preds_aligned))

            # rmsds = torch.stack(rmsds, dim=0)

            print(rmsds)

            rmsd_top1.append(rmsds[0])
            rmsd_top5.append(rmsds[:5])
            rmsd_top5_min.append(torch.min(rmsds[:5]))

            PB_valid = PB_validate_test(
                mol_preds_aligned[0],
                gt_mol,
                aligned_protein_path,
            )
            PB_valids.append(PB_valid)

        else:
            mol_pred_paths = []

            mol_pred_paths = glob.glob(os.path.expanduser(
                    f"{out_dir}/{complex_name}/rank*.sdf"
                ))

            if len(mol_pred_paths) < 1:
                failed_complexes.append(complex_name)
                print("not ranked prediction found.")
                continue

            mol_preds_aligned, aligned_protein_path = align_poses_to_reference_site(
                task["protein_path"],
                task["ligand_description"],
                task["protein_path_apo"],
                mol_pred_paths
            )

            rmsds = torch.stack([calc_rmsd(mol_pred, gt_mol, remove_Hs=True) for mol_pred in mol_preds_aligned], dim=0)

            values, indices = torch.sort(rmsds, descending=False)
            rmsds = rmsds[indices]

            print(rmsds)

            rmsd_top1.append(rmsds[0])
            rmsd_top5.append(rmsds[:5])
            rmsd_top5_min.append(torch.min(rmsds[:5]))

            PB_valid = PB_validate_test(
                mol_preds_aligned[indices[0]],
                gt_mol,
                aligned_protein_path,
            )
            PB_valids.append(PB_valid)

    if not rmsd_top1:
        raise SystemExit("No predictions were evaluated; check the input and prediction paths.")
    rmsd_top1 = torch.tensor(rmsd_top1)
    rmsd_top5 = torch.nn.utils.rnn.pad_sequence(
        rmsd_top5, batch_first=True, padding_value=float("nan")
    )
    rmsd_top5_min = torch.tensor(rmsd_top5_min)
    PB_valids = torch.stack(PB_valids)

    print(f"Failed complexes: {len(failed_complexes)}")

    print("the top1 rmsd below 1.0:", torch.sum(rmsd_top1 < 1.0).item() ,"/" ,len(rmsd_top1), "(" ,torch.sum(rmsd_top1 < 1.0).item() / len(rmsd_top1) * 100, "%)")
    print("the top1 rmsd below 2.0:", torch.sum(rmsd_top1 < 2.0).item() ,"/" ,len(rmsd_top1), "(" ,torch.sum(rmsd_top1 < 2.0).item() / len(rmsd_top1) * 100, "%)")
    print("the top1 rmsd median:", torch.nanmedian(rmsd_top1))
    print("the top5 rmsd below 1.0:", torch.sum(rmsd_top5_min < 1.0).item() ,"/" ,len(rmsd_top5_min), "(" ,torch.sum(rmsd_top5_min < 1.0).item() / len(rmsd_top5_min) * 100, "%)")
    print("the top5 rmsd below 2.0:", torch.sum(rmsd_top5_min < 2.0).item() ,"/" ,len(rmsd_top5_min), "(" ,torch.sum(rmsd_top5_min < 2.0).item() / len(rmsd_top5_min) * 100, "%)")
    print("the top5 rmsd median:", torch.nanmedian(rmsd_top5_min))
    print("PoseBusters valid:", torch.sum(PB_valids.all(dim=-1)).item() ,"/" ,PB_valids.shape[0], "(" ,torch.sum(PB_valids.all(dim=-1)).item() / PB_valids.shape[0] * 100, "%)")

    PB_failed_count = torch.sum(~PB_valids, dim=-0)
    for i, flag in enumerate(PB_failed_count):
        print(f"{flag} - {FAIL_RULES[i]}")

    # 只考虑 0~40 范围
    rmsd_top1_clipped = torch.clamp(rmsd_top1, max=40).numpy()

    # 设置柱状图的 bin，这里每 1 Å 一个 bin
    bins = range(0, 41, 1)

    plt.figure(figsize=(10,6))
    plt.hist(rmsd_top1_clipped, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Top1 RMSD Distribution (0~40 Å)')
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Count')
    plt.xticks(range(0, 41, 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存图
    plot_path = os.path.join(out_dir, 'rmsd_top1_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"RMSD top1 distribution plot saved as {plot_path}")
