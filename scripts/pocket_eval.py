"""
python ./scripts/pocket_eval.py
"""

import os
import random
import sys
import json
import argparse
import hashlib
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def reference_centers(protein_path, ligand_coords, ligand_ids, threshold=4.0):
    """Compute DCC targets from raw structure coordinates, without model outputs.

    Implements VN-EGNN scripts/extract_binding_info.py (commit 1d1f922):
    1. Parse protein.pdb with Bio.PDB; iterate all models/chains/residues.
    2. Keep residues with residue.id[0] == " " and the official seq1 check.
       Include all atoms returned by Bio.PDB (its normal altloc selection).
    3. For each ligand (np.unique(ligand_ids) order), select protein atoms
       within <= threshold Angstrom of ANY atom of that ligand.
    4. Average the selected protein atom coordinates, unweighted. This is
       NOT the ligand centroid and NOT a mean over residue/CA coordinates.

    Ligand coordinates are the same dataset coordinates used by DCA, in the
    same coordinate frame as protein.pdb. No alignment/centering is applied.
    No contact atoms => NaN center => DCC miss; keep that ligand in the
    denominator. Missing/unreadable PDBs raise, rather than changing targets.
    """
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    from scipy.spatial import cKDTree

    protein_path = Path(protein_path).expanduser()
    if not protein_path.is_file():
        raise FileNotFoundError(f'DCC requires the original protein structure: {protein_path}')
    structure = PDBParser(QUIET=True).get_structure(protein_path.stem, str(protein_path))
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                try:
                    seq1(residue.resname)
                except KeyError:
                    continue
                atoms.extend(atom.get_coord() for atom in residue)
    protein_coords = np.asarray(atoms).reshape(-1, 3)
    ligand_coords = np.asarray(ligand_coords)
    ligand_ids = np.asarray(ligand_ids)
    if ligand_coords.shape != (len(ligand_ids), 3) or not np.isfinite(ligand_coords).all():
        raise ValueError('Expected finite ligand coordinates (N, 3) and N ligand IDs')
    centers = []
    for ligand_id in np.unique(ligand_ids):
        # Nearest-ligand-atom distance is equivalent to (pairwise_dist <= 4).any(axis=1).
        distances = cKDTree(ligand_coords[ligand_ids == ligand_id]).query(protein_coords)[0]
        contacts = protein_coords[distances <= threshold]
        centers.append(contacts.mean(axis=0) if len(contacts) else np.full(3, np.nan))
    return np.asarray(centers).reshape(-1, 3)


def ligand_key(x):
    a = np.rint(x * 1000).astype(np.int64)
    a = a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
    return hashlib.sha256(a.tobytes()).hexdigest()


def summarize_predictions(records, split, evaluate_single):
    """Top-M over PDBs; round each protein's ratio to 2 decimals before macro mean.

    HOLO4K merges chain predictions, deduplicates ligand coordinate sets, and
    retains the first chain's reference for identical ligands (stable ID order).
    """
    groups = {}
    for r in sorted(records, key=lambda r: r['id']):
        ident = r['id'][:4] if split == 'holo4k' else r['id']
        g = groups.setdefault(ident, dict(points=[], scores=[], ligands=[], keys=set()))
        g['points'].extend(r['centers'])
        g['scores'].extend(r['scores'])
        for j, center in zip(np.unique(r['ids']), r['gt']):
            x = r['xyz'][r['ids'] == j]
            key = ligand_key(x)
            if split == 'holo4k' and key in g['keys']:
                continue
            g['keys'].add(key)
            g['ligands'].append((x, center))
    rows = []
    for ident, g in sorted(groups.items()):
        ligands = g['ligands']
        gt = np.asarray([l[1] for l in ligands])
        xyz = np.concatenate([l[0] for l in ligands])
        ids = np.concatenate([np.full(len(l[0]), i) for i, l in enumerate(ligands)])
        centers = np.asarray(g['points']).reshape(-1, 3)
        dcc, dca = evaluate_single(centers, np.asarray(g['scores']), gt, xyz, ids)
        rows.append(dict(protein_name=ident, num_ligs=len(ligands),
                         n_rank_dcc_0=dcc, n_rank_dca_0=dca,
                         no_prediction=int(not len(centers)),
                         empty_reference_centers=int(np.isnan(gt).any(axis=1).sum())))
    frame = pd.DataFrame(rows)
    for metric in ('dcc', 'dca'):
        frame[f'{metric}_ratio'] = (frame[f'n_rank_{metric}_0'] / frame.num_ligs).round(2)
    summary = dict(dataset=split, proteins=len(frame), sites=int(frame.num_ligs.sum()),
                   DCC=float(frame.dcc_ratio.mean()), DCA=float(frame.dca_ratio.mean()),
                   no_prediction=int(frame.no_prediction.sum()),
                   empty_reference_centers=int(frame.empty_reference_centers.sum()))
    return frame, summary


def main():
    from scripts.docking_utils import dbscan_pockets, evaluate_single_protein_strict
    from data.utils import keep_fields
    from models.model import AbstractModel
    from models.utils import load_toml_cfg

    parser = argparse.ArgumentParser(description="Pocket prediction with VN-EGNN-aligned evaluation")
    parser.add_argument('--config', default='config/PDBbind/Sc_pdb_pocket_joint.toml')
    parser.add_argument('--checkpoint')
    parser.add_argument('--data-root', default='./datasets/Sc_pdb_pocket', help='scPDB benchmark dataset root')
    parser.add_argument('--split', choices=['coach420', 'holo4k', 'pdbbind2020'], default='coach420')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    pocket_cfg = load_toml_cfg(args.config)
    if args.checkpoint:
        pocket_cfg.paths.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    pocket_ranker_cfg = pocket_cfg

    if pocket_cfg.task == "JointPocketPrediction" or pocket_cfg.task == "Joint_all":
        pocket_radius_cutoff = pocket_cfg.dataset.pocket.radius_cutoff
        radius_min = pocket_ranker_cfg.dataset.pocket.get('radius_min', 0.0)
        ranker_radius_cutoff = pocket_ranker_cfg.dataset.pocket_ranker.radius_cutoff
    else:
        pocket_radius_cutoff = pocket_cfg.dataset.radius_cutoff
        ranker_radius_cutoff = pocket_ranker_cfg.dataset.radius_cutoff
        radius_min = pocket_cfg.dataset.get('radius_min', 0.0)

    from data.Sc_pdb_pocket import Sc_pdb_pocket

    split = args.split
    dataset_test = Sc_pdb_pocket(
        args.data_root,
        split=split,
        mode="inference",
    )

    # Original proteins in the same coordinate frame as the dataset ligand coordinates.
    # Layout: <raw_protein_root>/<sample_id>/protein.pdb. No report/JSON dependency.
    raw_protein_root = Path(dataset_test.raw_dir).expanduser() / split / "raw"
    if not raw_protein_root.is_dir():
        raise FileNotFoundError(f"Set raw_protein_root to the original {split} proteins: {raw_protein_root}")

    device = torch.device(args.device)
    out_dir = (args.out_dir.expanduser() if args.out_dir is not None else
               Path('./output') / 'pocket_eval' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out_dir.mkdir(parents=True, exist_ok=False)

    random_indices = random.sample(range(len(dataset_test)), len(dataset_test))

    model = AbstractModel.load_from_checkpoint(
        pocket_cfg.paths.checkpoint, map_location=device, cfg=pocket_cfg
    )
    model.to(device)
    model.eval()

    evaluation_records = []
    for idx in tqdm(random_indices):
        protein_data = dataset_test[idx]
        prot_data_residue = protein_data.to_backbone_graph()
        prot_data_residue = prot_data_residue.update_graph(
            pocket_radius_cutoff, radius_min, scale=8.0 / pocket_radius_cutoff
        ).to(device)

        with torch.no_grad():
            vec_to_pocket = model(prot_data_residue, task="pocket_pred")

        points = prot_data_residue.pos[prot_data_residue.CB_mask] + vec_to_pocket  # [N,3]

        protein_data.prepare_pocket_ranker_graph(points.cpu(), ranker_radius_cutoff)
        protein_data.edge_vector = (
            protein_data.pos[protein_data.edge_index[0]]
            - protein_data.pos[protein_data.edge_index[1]]
        )
        protein_data.edge_types = torch.zeros(
            protein_data.edge_index.shape[1],
            dtype=torch.long,
            device=protein_data.pos.device,
        )

        protein_id = str(protein_data.id)
        gt_coords = protein_data.ligand_pos.cpu().numpy()
        gt_ligand_ids = protein_data.ligand_id.cpu().numpy()
        gt_centers = reference_centers(
            raw_protein_root / protein_id / "protein.pdb", gt_coords, gt_ligand_ids,
        )

        protein_data = keep_fields(
            protein_data,
            [
                "x",
                "y",
                "pos",
                "edge_index",
                "edge_types",
                "edge_vector",
            ],
        ).to(device)

        with torch.no_grad():
            score = model(protein_data, task="pocket_ranker")

        centers, confidences = dbscan_pockets(
            points.cpu().numpy(), score.cpu().detach().numpy().reshape(-1),
            eps=4.5, min_samples=1, quantile=0.9, center_power=2,
        )

        print(centers, confidences)

        evaluation_records.append(dict(
            id=protein_id, centers=centers, scores=confidences.cpu().numpy(),
            gt=gt_centers, xyz=gt_coords, ids=gt_ligand_ids,
        ))
        (out_dir / 'progress.json').write_text(json.dumps(dict(
            split=split, completed=len(evaluation_records), total=len(dataset_test),
            last_id=protein_id,
        )))

        prot_data_residue = prot_data_residue.update_graph(4.0).to("cpu")

        print(prot_data_residue)

        score = score / score.max()

    if len(evaluation_records) != len(dataset_test):
        raise ValueError("Evaluation did not cover the entire test dataset")
    per_protein, summary = summarize_predictions(
        evaluation_records, split, evaluate_single_protein_strict,
    )
    per_protein.to_csv(out_dir / f'{split}_per_protein.csv', index=False)
    if args.save_predictions:
        with (out_dir / 'predictions.pkl').open('wb') as handle:
            pickle.dump(evaluation_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary.update(checkpoint=str(pocket_cfg.paths.checkpoint), precision='FP32',
                   eps=4.5, min_samples=1, quantile=0.9, center_power=2,
                   aggregation='per-protein mean after rounding each ratio to 2 decimals',
                   reference='mean of protein atoms within <=4 A of any ligand atom',
                   raw_protein_root=str(raw_protein_root), config=args.config,
                   checkpoint_sha256=hashlib.sha256(Path(pocket_cfg.paths.checkpoint).expanduser().read_bytes()).hexdigest(),
                   pocket_radius_cutoff=pocket_radius_cutoff, radius_min=radius_min,
                   ranker_radius_cutoff=ranker_radius_cutoff)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    print("===== FINAL RESULTS (Strict Top-M) =====")
    print(f"DCC: {summary['DCC']:.4f}")
    print(f"DCA: {summary['DCA']:.4f}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
