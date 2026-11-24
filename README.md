# COSMol

COSMol provides protein–ligand docking, ligand conformer generation, and virtual screening with optional unbound conformer support. See the [reproduction guide](reproduce/README.md) for dataset-specific experiments and evaluation.

## Setup

Use Python 3.11 on Linux with an NVIDIA GPU.

```bash
git clone https://github.com/cosmol-studio/COSMol.git
cd COSMol
python -m pip install -r requirements.in
curl -fL "https://obj.cosmol.net/cosmol/checkpoints_v6.gz" -o ckpt/checkpoints.tar.gz
tar -xzf ckpt/checkpoints.tar.gz -C ckpt
```

Select your GPUs in [config/dock_config.toml](config/dock_config.toml). The default is GPU 0 with 40 poses per ligand.

## Dock one complex

Provide a receptor as a PDB (`.pdb`) or mmCIF (`.cif`) file and a ligand as an SDF file, a MOL2 file, or a SMILES string.

```bash
python scripts/dock.py -c config/dock_config.toml \
  --complex_name 6fe5 \
  --protein_path input/6fe5_protein.pdb \
  --ligand_description input/6fe5_ligand.sdf \
  --out_dir ./output/single
```

The model predicts candidate binding sites and generates docking poses. The pose ranker orders the output SDF files under `./output/single/6fe5/`. Filenames contain the rank and score, such as `rank01_0.850.sdf`.

To use the center of a reference ligand as the docking center, add `--use_ligand_center_as_pocket`. The ligand input must contain coordinates in the receptor's coordinate frame.

## Dock multiple complexes

Use [input/protein_ligand.csv](input/protein_ligand.csv) as an example, with one complex per row.

```csv
complex_name,protein_path,ligand_description
6fe5,input/6fe5_protein.pdb,input/6fe5_ligand.sdf
1sre,input/1sre_clean.pdb,input/1sre_ligand.sdf
```

Run

```bash
python scripts/dock.py -c config/dock_config.toml \
  --protein_ligand_csv input/protein_ligand.csv --out_dir ./output/batch
```

Each complex has its own output directory. Tasks are distributed across the GPUs selected in the configuration. Use `--input_dir` to prepend a common directory to relative input paths in the CSV.

For supplied pockets, add a `pocket_description` column containing a PDB, SDF, or MOL2 path, then pass `--use_pocket`. Its heavy-atom centroid defines the docking center. Pass `--use_ligand_center_as_pocket` to use each input ligand's center instead.

## Generate ligand conformers

Generate free-ligand conformers from a SMILES string, an SDF file, or a MOL2 file.

```bash
python scripts/conf_gen.py -c config/Joint/Joint_all.toml \
  --ligand_description "CCO" --num_confs 40 --device cuda:0 \
  --out_dir ./output/conformers
```

For multiple ligands, use [input/ligands.csv](input/ligands.csv), with one molecule per row.

```csv
ligand_name,ligand_description
ethanol,CCO
lactic_acid,C[C@H](O)C(=O)O
6fe5,input/6fe5_ligand.sdf
```

```bash
python scripts/conf_gen.py -c config/Joint/Joint_all.toml \
  --ligand_csv input/ligands.csv --num_confs 40 --batch_size 32 --device cuda:0 \
  --out_dir ./output/conformers_batch
```

Relative ligand file paths are resolved from the working directory. Conformers are saved as `0.sdf`, `1.sdf`, and so on, with explicit hydrogens retained. Single-ligand output goes directly under `<out_dir>/`; CSV input creates one `<out_dir>/<ligand_name>/` directory per ligand. `--num_confs` sets the number of conformers per ligand. `--batch_size` controls the number of ligands processed together, and `--flow_temperature` sets the flow update scale. The GPU defaults to the first GPU selected in the model configuration when `--device` is omitted.

## Virtual screening

Screen a ligand library against a receptor using a reference ligand to define the docking center. Prepare one directory per target using either supported layout.

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

SMILES files contain one molecule per line as `SMILES ligand_id`. For DEKOIS2, combine actives and decoys in `active_decoys.smi`, using IDs beginning with `BDB` for actives and `ZINC` for decoys. For DUDE-Z, place actives in `ligands.smi` and decoys in `decoys.smi`. These labels are required by the evaluator.

Keep the reference ligand coordinates in the receptor's coordinate frame. Its heavy-atom centroid defines the docking center.

Run screening, then calculate the final scores.

```bash
python scripts/virtual_screening.py -c config/dock_config.toml \
  --data-root ./datasets/DEKOIS2 --out_dir ./vs_results --unbound-support

python scripts/evaluate_virtual_screening.py --vs-dir ./vs_results \
  --unbound-support --agg topk_mean --topk 10
```

`--unbound-support` enables free-conformer support scoring. Omit this option from both commands to use pose-ranker scores alone. For DUDE-Z input, use `--input-format dudez --data-root ./datasets/DUDE-Z` in the screening command. Add `--target <name>` to select one target, and use `--ligand_batch_size` to control the number of ligands processed together.

Raw pose scores are saved to `<out_dir>/<target>/scores.csv`. Final ligand scores are saved to `<out_dir>/evaluation/<target>/evaluated_scores.csv`. Use the `Ranker + Q_C` column for support scoring or `Ranker` for the baseline, sorting from highest to lowest.
