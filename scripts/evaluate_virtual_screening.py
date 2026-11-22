"""Evaluate raw scripts/virtual_screening.py outputs for DEKOIS2.0 or DUDE-Z.

python scripts/evaluate_virtual_screening.py --vs_dir ./vs_results
python scripts/evaluate_virtual_screening.py --vs_dir ./vs_results --unbound-support --agg topk_mean --topk 10

Compute ligand scores and screening metrics from saved docking poses and
optional unbound conformers.
"""

import argparse
import json
import signal
import sys
from pathlib import Path

import cosmolkit as ck
import numpy as np
import pandas as pd
import rustworkx as rx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from spyrmsd import graph as spyrmsd_graph

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.utils import cache_mapping, rotatable_quartets
from models.utils import resolve_path


def get_label(mol_id: str) -> int:
    if mol_id.startswith("BDB"):
        return 1
    elif mol_id.startswith("ZINC"):
        return 0
    else:
        raise ValueError(f"Unknown mol_id: {mol_id}")

def parse_ranker_scores(x):
    if pd.isna(x):
        return []

    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]

    s = str(x).strip()
    if not s:
        return []

    parts = [p.strip() for p in s.split(";") if p.strip()]
    return [float(p) for p in parts]

def aggregate_scores(
    scores,
    agg="max",
    topk=5,
):
    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        raise ValueError("Empty ranker score list")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite ranker score list")

    if agg == "max":
        return float(np.max(scores))

    elif agg == "mean":
        return float(np.mean(scores))

    elif agg == "median":
        return float(np.median(scores))

    elif agg == "topk_mean":
        k = min(topk, scores.size)
        top_scores = np.sort(scores)[-k:]
        return float(np.mean(top_scores))

    elif agg == "topk_median":
        k = min(topk, scores.size)
        top_scores = np.sort(scores)[-k:]
        return float(np.median(top_scores))

    else:
        raise ValueError(f"Unknown aggregation method: {agg}")

def enrichment_factor(labels, scores, top_percent):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    N = len(labels)
    n_top = max(1, int(np.ceil(N * top_percent / 100.0)))

    idx = np.argsort(scores)[::-1]
    top_idx = idx[:n_top]

    n_actives_total = np.sum(labels)
    n_actives_top = np.sum(labels[top_idx])

    if n_actives_total == 0:
        return 0.0

    ef = (n_actives_top / n_top) / (n_actives_total / N)
    return float(ef)

def bedroc(labels, scores, alpha=20.0):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    N = len(labels)
    n = int(labels.sum())

    if n == 0 or n == N:
        return np.nan

    # Rank ligands by descending score.
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]

    # Common RIE normalization cancels. Shift ranks by one to avoid underflow
    # for small smoke-test libraries at large alpha; this leaves BEDROC unchanged.
    ranks = np.flatnonzero(labels_sorted == 1)
    observed = np.exp(-alpha * ranks / N).sum()
    best = np.exp(-alpha * np.arange(n) / N).sum()
    worst = np.exp(-alpha * np.arange(N - n, N) / N).sum()
    if best == worst:
        return np.nan

    bedroc_value = (observed - worst) / (best - worst)
    return float(bedroc_value)

def fuse(rows, exponent):
    r = np.asarray([x['ranker_top10_mean'] for x in rows], dtype=float)
    qt = rankdata(-np.asarray([x['torsion_surprisal'] for x in rows]), method='average') / len(rows)
    qt[np.asarray([x['num_rotatable_bonds'] for x in rows]) == 0] = 0.5
    qd = rankdata([x['basin_fraction'] for x in rows], method='average') / len(rows)
    qc = 2 * qt * qd / np.maximum(qt + qd, 1e-12)
    return r * np.maximum(qc, 1e-6) ** exponent, qt, qd


def support_features(path, expected_id, expected_smiles, expected_scores,
                     cluster_threshold=3.0, bandwidth=15.0):
    """Compute support using top-five poses; threshold is in Å, bandwidth in degrees."""
    with np.load(path, allow_pickle=False) as raw:
        if str(raw["mol_id"]) != expected_id or str(raw["smiles"]) != expected_smiles:
            raise ValueError(f"Ligand identity mismatch: {path}")
        scores = raw["ranker_scores"]
        np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=0)
        numbers, bonds = raw["atomic_numbers"], raw["bond_index"]
        free = raw["free_heavy_coords"].astype(np.float64)
        dock = raw["docking_coords"].astype(np.float64)
    if len(scores) < 5:
        raise ValueError("Unbound support requires at least 5 docking poses")
    for coords in (free, dock):
        if coords.ndim != 3 or coords.shape[1:] != (len(numbers), 3) or not np.isfinite(coords).all():
            raise ValueError(f"Invalid coordinates: {path}")
    if not len(free):
        raise ValueError(f"Empty free ensemble: {path}")
    order = np.argsort(scores)[::-1][:5]
    dock = dock[order]

    # Graph automorphisms correct atom symmetry without rigidly aligning poses.
    adjacency = np.zeros((len(numbers), len(numbers)), dtype=int)
    adjacency[bonds[0], bonds[1]] = 1
    graph = spyrmsd_graph.graph_from_adjacency_matrix(adjacency, numbers)
    identity = tuple(range(len(numbers)))
    permutations, seen = [np.asarray(identity)], {identity}
    def timed_out(signum, frame):
        raise TimeoutError("Graph symmetry enumeration timed out")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(2)
    try:
        for mapping in rx.vf2_mapping(graph, graph, lambda a, b: a == b):
            permutation = tuple(int(mapping[i]) for i in range(len(numbers)))
            if permutation not in seen:
                permutations.append(np.asarray(permutation))
                seen.add(permutation)
            if len(permutations) >= 256:
                break
    except TimeoutError:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    permutations = np.stack(permutations)
    distances = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1, 5):
            squared = np.mean(np.sum((dock[j][permutations] - dock[i]) ** 2, axis=2), axis=1)
            distances[i, j] = distances[j, i] = np.sqrt(squared.min())
    labels = fcluster(linkage(squareform(distances, checks=False), method="complete"),
                      cluster_threshold, criterion="distance")
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    size = max(map(len, clusters))
    basin = max((c for c in clusters if len(c) == size),
                key=lambda c: float(np.mean(scores[order][c])))

    mol = ck.Molecule.from_smiles(expected_smiles)
    mapping = cache_mapping(mol, numbers, bonds)
    quartets = [tuple(int(mapping[i]) for i in q) for q in rotatable_quartets(mol)]

    def angles(coords):
        values = []
        for a, b, c, d in quartets:
            v0, axis, v2 = coords[:, a] - coords[:, b], coords[:, c] - coords[:, b], coords[:, d] - coords[:, c]
            axis = axis / np.linalg.norm(axis, axis=1, keepdims=True).clip(min=1e-12)
            p0 = v0 - np.sum(v0 * axis, axis=1, keepdims=True) * axis
            p2 = v2 - np.sum(v2 * axis, axis=1, keepdims=True) * axis
            values.append(np.arctan2(np.sum(np.cross(axis, p0) * p2, axis=1), np.sum(p0 * p2, axis=1)))
        return np.stack(values, axis=1)

    surprisal = 0.0
    if quartets:
        delta = angles(dock)[:, None, :] - angles(free)[None, :, :]
        delta = np.abs((delta + np.pi) % (2 * np.pi) - np.pi)
        support = np.exp(-delta ** 2 / (2 * np.deg2rad(bandwidth) ** 2)).mean(axis=1)
        support = (len(free) * support + 0.05) / (len(free) + 1)
        surprisal = float((-np.log(np.clip(support, 1e-8, None)))[basin].mean())
    return dict(num_rotatable_bonds=len(quartets), torsion_surprisal=surprisal,
                basin_fraction=len(basin) / 5)


def evaluate_target(directory, args):
    df = pd.read_csv(directory / "scores.csv", dtype={"mol_id": str, "smiles": str})
    failed = df["status"] != "ok"
    if failed.any() and not args.allow_failed:
        raise ValueError(f"{directory.name}: {int(failed.sum())} failed ligands; rerun or use --allow_failed")
    df = df[~failed].copy().sort_values("source_index") if "source_index" in df else df[~failed].copy()
    if df.empty or df["mol_id"].duplicated().any():
        raise ValueError(f"Empty or duplicate ligand IDs: {directory}")
    parsed = [parse_ranker_scores(value) for value in df["ranker_scores"]]
    ranker = np.asarray([aggregate_scores(scores, args.agg, args.topk) for scores in parsed])
    if "label" in df and df["label"].notna().all():
        labels = df["label"].to_numpy(dtype=float)
    else:
        labels = df["mol_id"].map(get_label).to_numpy()
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError(f"Target needs binary labels and both classes: {directory}")
    labels = labels.astype(int)
    methods = {"Ranker": ranker}
    df["Ranker"] = ranker
    if args.unbound_support:
        rows = []
        for (_, row), scores, r in zip(df.iterrows(), parsed, ranker):
            if not isinstance(row.get("conformer_file"), str) or not row["conformer_file"]:
                raise ValueError(f"Missing conformations for {row['mol_id']}; generate with --unbound-support")
            features = support_features(
                directory / row["conformer_file"], row["mol_id"], row["smiles"], scores,
                cluster_threshold=args.cluster_threshold, bandwidth=args.bandwidth,
            )
            rows.append(dict(ranker_top10_mean=float(r), **features))
        # All percentile calculations use this target's evaluated ligand library.
        combined, qt, qd = fuse(rows, args.support_exponent)
        qc = 2 * qt * qd / np.maximum(qt + qd, 1e-12)
        methods.update({
            "Ranker + Q_D": ranker * np.maximum(qd, 1e-6) ** args.support_exponent,
            "Ranker + Q_T": ranker * np.maximum(qt, 1e-6) ** args.support_exponent,
            "Ranker + Q_C": combined,
        })
        for key in ("num_rotatable_bonds", "torsion_surprisal", "basin_fraction"):
            df[key] = [row[key] for row in rows]
        df["Q_D"], df["Q_T"], df["Q_C"] = qd, qt, qc
    metrics = []
    for method, scores in methods.items():
        df[method] = scores
        metrics.append(dict(
            target_id=directory.name, method=method, n_ligands=len(df), n_actives=int(labels.sum()),
            n_failed=int(failed.sum()), AUROC=roc_auc_score(labels, scores),
            BEDROC=bedroc(labels, scores, args.bedroc_alpha),
            EF05=enrichment_factor(labels, scores, 0.5),
            EF1=enrichment_factor(labels, scores, 1), EF5=enrichment_factor(labels, scores, 5),
        ))
    output = args.output_dir / directory.name
    output.mkdir(parents=True, exist_ok=True)
    df.sort_values("Ranker", ascending=False).to_csv(output / "evaluated_scores.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs_dir", "--vs-dir", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--unbound-support", "--unbound_support", action="store_true")
    parser.add_argument("--agg", choices=["max", "mean", "median", "topk_mean", "topk_median"], default="max")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--support-exponent", type=float, default=0.10)
    parser.add_argument("--cluster-threshold", type=float, default=3.0,
                        help="Complete-linkage pose clustering threshold in Å (default: 3)")
    parser.add_argument("--bandwidth", type=float, default=15.0,
                        help="Periodic torsion kernel bandwidth in degrees (default: 15)")
    parser.add_argument("--bedroc_alpha", "--bedroc-alpha", type=float, default=80.5)
    parser.add_argument("--allow_failed", "--allow-failed", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.topk < 1 or not np.isfinite(args.support_exponent) or args.support_exponent < 0:
        parser.error("topk must be positive and support exponent finite and nonnegative")
    if not np.isfinite(args.bedroc_alpha) or args.bedroc_alpha <= 0:
        parser.error("BEDROC alpha must be finite and positive")
    if any(not np.isfinite(value) or value <= 0
           for value in (args.cluster_threshold, args.bandwidth)):
        parser.error("cluster threshold and bandwidth must be finite and positive")
    args.vs_dir = Path(resolve_path(args.vs_dir, output=True))
    args.output_dir = Path(resolve_path(args.output_dir, output=True)) if args.output_dir else args.vs_dir / "evaluation"
    targets = [args.vs_dir / args.target] if args.target else sorted(
        p for p in args.vs_dir.iterdir() if p.is_dir() and (p / "scores.csv").is_file()
    )
    if not targets:
        parser.error("No scores.csv files found")
    metrics = []
    for target in targets:
        metrics.extend(evaluate_target(target, args))
        print(f"Evaluated {target.name}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_target = pd.DataFrame(metrics)
    per_target.to_csv(args.output_dir / "per_target_metrics.csv", index=False)
    summary = per_target.groupby("method", sort=False)[["AUROC", "BEDROC", "EF05", "EF1", "EF5"]].agg(["mean", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(args.output_dir / "summary.csv")
    (args.output_dir / "evaluation.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
    ) + "\n")
    print(summary.to_string())
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
