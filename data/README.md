# Training data and caches

Training datasets use `./datasets/` by default. Set the dataset `path` fields in the model TOML files to use another location. Benchmark CSV inputs and virtual-screening libraries are described in [reproduce/README.md](../reproduce/README.md).

## PDBbind2020

Prepare the original PDBbind2020 files and the supplied benchmark split lists under:

```text
datasets/PDBbind2020/
├── raw/
│   ├── index/
│   │   ├── INDEX_general_PL_name.2020
│   │   └── INDEX_general_PL_data.2020
│   ├── v2020-other-PL/<pdb_id>/
│   │   ├── <pdb_id>_protein.pdb
│   │   ├── <pdb_id>_ligand.sdf
│   │   └── <pdb_id>_ligand.mol2
│   ├── timesplit_no_lig_overlap_val
│   ├── timesplit_test
│   └── timesplit_test_no_rec_overlap
└── processed/
```

Copy the provided [split lists](../reproduce/splits/PDBbind2020) into the raw dataset directory:

```bash
cp reproduce/splits/PDBbind2020/* datasets/PDBbind2020/raw/
```

`data/PDBbind2020.py` and `data/PDBbind2020_pocket.py` construct docking and pocket caches from these inputs when the corresponding cache is absent. Their cache filenames are `PDBbind2020_<split>_v5.lmdb` and `PDBbind2020_pocket_<split>_v5.lmdb`, with `split` equal to `train`, `valid`, or `test`.

Pose-ranker training reads prepared V5 positive and negative pose caches under the same `processed/` directory:

```text
PDBbind2020_<split>_pose_ranker_v5.lmdb
PDBbind2020_<split>_v5_pose_ranker_negative.lmdb
```

These contain docked coordinates and their training labels. Raw receptor and ligand files alone are insufficient for the pose-ranker training loader.

## GEOM conformers

Place the GEOM/SDEGen raw files `train_data_39k.pkl`, `val_data_5k.pkl`, and `test_data_200.pkl` under `datasets/GEOM_SDEGen/raw/`. The dataset constructs `processed/geom_<split>_v5_cosmolkit.lmdb`. Explicit hydrogens are retained for conformer generation. Construct these caches with the cosmolkit version specified in `requirements.in`.

## scPDB and pocket benchmarks

The standalone pocket model uses `datasets/Sc_pdb_pocket/`. Its prepared caches are `processed/sc-pdb_<split>_v5.lmdb`, where `split` is `train`, `valid`, `coach420`, `holo4k`, or `pdbbind2020`.

Pocket evaluation also requires the original receptor files at `raw/<split>/raw/<sample_id>/protein.pdb`. These receptors must share the coordinate frame of the ligands stored in the cache.

## Training

Select GPUs and dataset paths in the configuration before running:

```bash
python scripts/train.py -c config/Joint/Joint_all.toml
python scripts/train.py -c config/PDBbind/Pose_ranker.toml
python scripts/train.py -c config/PDBbind/Sc_pdb_pocket_joint.toml
```

Training logs are written to `./logs/`. Checkpoints are saved every five epochs and all periodic checkpoints are retained. After training, the final checkpoint is exported to `paths.checkpoint`; if that file already exists, the export receives a `-vN` suffix. Set `paths.resume_from` to a checkpoint path to resume training. Use `WANDB_MODE=offline` to save W&B logs locally.
