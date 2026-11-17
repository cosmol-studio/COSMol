# Reproduce COSMol experiments

This guide describes COSMol docking and virtual-screening experiments on the supplied benchmark manifests. For environment setup, model weights, and general inference, see the [usage guide](../README.md).

Run all commands from the repository root. Configure the model checkpoint paths and GPUs in the configurations linked from the usage guide. The commands below use 40 docking poses, 50 updates, FP16 inference, and pose-ranker ordering.

## Docking datasets

Keep the complex lists in the supplied CSV files and prepare the corresponding receptor and reference-ligand structures.

| Dataset | Manifest | Complexes | Input directory |
| --- | --- | ---: | --- |
| PDBbind2020 test | [PDBBind_test_set.csv](PDBBind_test_set.csv) | 363 | `./datasets/PDBbind2020/raw/v2020-other-PL` |
| Astex Diverse | [astex_diverse_set.csv](astex_diverse_set.csv) | 85 | `./datasets/Astex_Diverse/astex_diverse_set` |
| Astex Diverse (apo) | [astex_diverse_set_apo.csv](astex_diverse_set_apo.csv) | 85 | `./datasets/Astex_Diverse/astex_diverse_set` |
| PoseBusters | [posebusters_test_set.csv](posebusters_test_set.csv) | 308 | `./datasets/PoseBusters/posebusters_benchmark_set` |

The counts describe the manifests supplied in this repository. Paths inside each manifest are relative to its input directory. Each reference ligand must retain its crystallographic coordinates in the receptor's coordinate frame.

Docking evaluation requires a rank-1 prediction for every complex in the input CSV. Missing predictions stop evaluation with an error; success rates use the full CSV size as the denominator.

For **ligand-center docking**, `--use_ligand_center_as_pocket` sets the docking center to the reference ligand's heavy-atom centroid. For **blind docking**, the model predicts candidate sites from the full receptor. Both settings use the pose ranker to order the generated poses.

### PDBbind2020

Generate ligand-center predictions.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv reproduce/PDBBind_test_set.csv \
  --input_dir ./datasets/PDBbind2020/raw/v2020-other-PL \
  --use_ligand_center_as_pocket --out_dir ./reproduce/results/pdbbind_center
```

Generate blind-docking predictions.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv reproduce/PDBBind_test_set.csv \
  --input_dir ./datasets/PDBbind2020/raw/v2020-other-PL \
  --out_dir ./reproduce/results/pdbbind_blind
```

Evaluate each output directory.

```bash
python reproduce/evaluate_docking.py \
  --protein_ligand_csv reproduce/PDBBind_test_set.csv \
  --input_dir ./datasets/PDBbind2020/raw/v2020-other-PL \
  --out_dir ./reproduce/results/pdbbind_center

python reproduce/evaluate_docking.py \
  --protein_ligand_csv reproduce/PDBBind_test_set.csv \
  --input_dir ./datasets/PDBbind2020/raw/v2020-other-PL \
  --out_dir ./reproduce/results/pdbbind_blind
```

### Astex Diverse

Generate and evaluate ligand-center predictions.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv reproduce/astex_diverse_set.csv \
  --input_dir ./datasets/Astex_Diverse/astex_diverse_set \
  --use_ligand_center_as_pocket --out_dir ./reproduce/results/astex_center

python reproduce/evaluate_docking.py \
  --protein_ligand_csv reproduce/astex_diverse_set.csv \
  --input_dir ./datasets/Astex_Diverse/astex_diverse_set \
  --out_dir ./reproduce/results/astex_center
```

Generate and evaluate blind-docking predictions.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv reproduce/astex_diverse_set.csv \
  --input_dir ./datasets/Astex_Diverse/astex_diverse_set \
  --out_dir ./reproduce/results/astex_blind

python reproduce/evaluate_docking.py \
  --protein_ligand_csv reproduce/astex_diverse_set.csv \
  --input_dir ./datasets/Astex_Diverse/astex_diverse_set \
  --out_dir ./reproduce/results/astex_blind
```

#### COSMol apo docking and evaluation

Use the receptors listed in `astex_diverse_set_apo.csv` for blind docking with COSMol. Place these receptor files under `astex_diverse_predicted_structures/` inside the Astex input directory. The evaluation aligns the receptor and predicted ligand poses to the holo reference before calculating RMSD and PoseBusters validity.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv reproduce/astex_diverse_set_apo.csv \
  --input_dir ./datasets/Astex_Diverse/astex_diverse_set \
  --out_dir ./reproduce/astex_diverse_set_apo_predictions

python reproduce/evaluate_apo_docking.py \
  --protein_ligand_csv reproduce/astex_diverse_set.csv \
  --protein_ligand_csv_apo reproduce/astex_diverse_set_apo.csv \
  --input_dir ./datasets/Astex_Diverse/astex_diverse_set \
  --out_dir ./reproduce/astex_diverse_set_apo_predictions
```

### PoseBusters

Generate and evaluate ligand-center predictions.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv reproduce/posebusters_test_set.csv \
  --input_dir ./datasets/PoseBusters/posebusters_benchmark_set \
  --use_ligand_center_as_pocket --out_dir ./reproduce/results/posebusters_center

python reproduce/evaluate_docking.py \
  --protein_ligand_csv reproduce/posebusters_test_set.csv \
  --input_dir ./datasets/PoseBusters/posebusters_benchmark_set \
  --out_dir ./reproduce/results/posebusters_center
```

Generate and evaluate blind-docking predictions.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv reproduce/posebusters_test_set.csv \
  --input_dir ./datasets/PoseBusters/posebusters_benchmark_set \
  --out_dir ./reproduce/results/posebusters_blind

python reproduce/evaluate_docking.py \
  --protein_ligand_csv reproduce/posebusters_test_set.csv \
  --input_dir ./datasets/PoseBusters/posebusters_benchmark_set \
  --out_dir ./reproduce/results/posebusters_blind
```

### Docking evaluation

The evaluator reads the five highest-ranked SDF files per complex. It reports Top-1 and Top-5 RMSD success rates at 1 Å and 2 Å, median RMSD, and the Top-1 PoseBusters pass rate. The current evaluator uses strict `<` comparisons for the RMSD thresholds. Review the reported failed-complex count alongside the evaluation denominator. PoseBusters is evaluated using version 0.4.5 with its `redock` configuration.

## Virtual screening

### Screening inputs

Prepare one directory per target under `./datasets/DEKOIS2` or `./datasets/DUDE-Z`.

```text
datasets/
├── DEKOIS2/
│   └── <target>/
│       ├── active_decoys.smi
│       └── protein/
│           ├── <target>_protein.pdb
│           └── <target>_ligand.mol2
└── DUDE-Z/
    └── <target>/
        ├── rec.crg.pdb
        ├── xtal-lig.pdb
        ├── ligands.smi
        └── decoys.smi
```

Each SMILES file contains one molecule per line, followed by its identifier. DEKOIS2 combines actives and decoys in `active_decoys.smi`, with IDs beginning with `BDB` for actives and `ZINC` for decoys. DUDE-Z labels follow the source file, with actives in `ligands.smi` and decoys in `decoys.smi`.

The reference ligand's heavy-atom centroid defines the docking center. Keep its coordinates in the receptor's coordinate frame.

### Generate docking poses and unbound conformers

```bash
python scripts/virtual_screening.py -c config/dock_config.toml \
  --data-root ./datasets/DEKOIS2 --out_dir ./reproduce/results/dekois2 \
  --unbound-support --num_confs 40 --unbound-num-confs 20 --seed 42

python scripts/virtual_screening.py -c config/dock_config.toml \
  --input-format dudez --data-root ./datasets/DUDE-Z \
  --out_dir ./reproduce/results/dudez \
  --unbound-support --num_confs 40 --unbound-num-confs 20 --seed 42
```

These commands generate the pose scores and conformations needed to evaluate the Ranker baseline and all three support variants from the same screening run. The unbound update scale defaults to 2.0. Select `--ligand_batch_size` for the available GPU memory. Failed batches are split into smaller batches, and the actual batch sizes and peak memory are recorded per GPU.

### Evaluate the baseline and support variants

```bash
python scripts/evaluate_virtual_screening.py --vs-dir ./reproduce/results/dekois2 \
  --unbound-support --agg topk_mean --topk 10 \
  --support-exponent 0.10 --bedroc-alpha 80.5

python scripts/evaluate_virtual_screening.py --vs-dir ./reproduce/results/dudez \
  --unbound-support --agg topk_mean --topk 10 \
  --support-exponent 0.10 --bedroc-alpha 80.5
```

The evaluator reports `Ranker`, `Ranker + Q_D`, `Ranker + Q_T`, and `Ranker + Q_C`. The baseline uses the mean of the ten highest pose scores. Support uses the five highest-ranked docking poses, complete-linkage clustering at 3 Å, and a periodic torsion kernel with a 15° bandwidth. Percentile ranks are calculated within each target library.

The reported metrics are AUROC, BEDROC with α = 80.5, and EF at 0.5%, 1%, and 5%. Targets require both active and decoy labels. Failed ligands cause evaluation to stop by default.

| Output under each screening directory | Contents |
| --- | --- |
| `evaluation/<target>/evaluated_scores.csv` | Per-ligand Ranker scores, support features, and combined scores |
| `evaluation/per_target_metrics.csv` | Metrics and evaluated ligand counts for each target and method |
| `evaluation/summary.csv` | Mean and median metrics across targets |
| `evaluation/evaluation.json` | Evaluation parameters |

To evaluate the Ranker baseline alone from an existing screening run, use

```bash
python scripts/evaluate_virtual_screening.py --vs-dir ./reproduce/results/dekois2 \
  --agg topk_mean --topk 10 --bedroc-alpha 80.5 \
  --output-dir ./reproduce/results/dekois2/evaluation_ranker
```

## Pocket prediction

Use the model trained on scPDB and the corresponding prepared benchmark caches. See [training data and cache layout](../data/README.md) for input requirements.

```bash
python scripts/pocket_eval.py --config config/PDBbind/Sc_pdb_pocket_joint.toml \
  --data-root ./datasets/Sc_pdb_pocket --split coach420 --device cuda:0
python scripts/pocket_eval.py --config config/PDBbind/Sc_pdb_pocket_joint.toml \
  --data-root ./datasets/Sc_pdb_pocket --split holo4k --device cuda:0
python scripts/pocket_eval.py --config config/PDBbind/Sc_pdb_pocket_joint.toml \
  --data-root ./datasets/Sc_pdb_pocket --split pdbbind2020 --device cuda:0
```

The evaluator reports DCC and DCA under the VN-EGNN-aligned protocol. Results are written under `./output/pocket_eval/`; use `--out-dir` to choose another directory. Model training and required caches are documented in [data/README.md](../data/README.md).
