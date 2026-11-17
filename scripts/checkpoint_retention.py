"""Project checkpoint retention policy: periodic archives are never disposable."""
from pathlib import Path
import re
from string import Formatter

from lightning.pytorch.callbacks import ModelCheckpoint


def validate_checkpoint_retention(trainer_cfg):
    if trainer_cfg.get("save_top_k", -1) != -1:
        raise ValueError("Checkpoint retention gate: save_top_k must be -1. Deleting old epoch checkpoints is forbidden.")
    interval = trainer_cfg.get("save_every_n_epochs", 5)
    if type(interval) is not int or interval < 1:
        raise ValueError("Checkpoint retention gate: save_every_n_epochs must be a positive integer (default 5).")
    filename = trainer_cfg.get("checkpoint_filename", "epoch-{epoch:03d}")
    fields = [field for _, field, _, _ in Formatter().parse(filename) if field is not None]
    if "epoch" not in fields:
        raise ValueError("Checkpoint retention gate: checkpoint_filename must contain {epoch} to preserve history.")


class RetainAllCheckpoints(ModelCheckpoint):
    def __init__(self, **kwargs):
        validate_checkpoint_retention({
            "save_top_k": kwargs.get("save_top_k", -1),
            "save_every_n_epochs": kwargs.get("every_n_epochs", 5),
            "checkpoint_filename": kwargs.get("filename", "epoch-{epoch:03d}"),
        })
        if kwargs.get("enable_version_counter", True) is not True:
            raise ValueError("Checkpoint retention gate: version counter cannot be disabled.")
        kwargs.setdefault("filename", "epoch-{epoch:03d}")
        kwargs.setdefault("every_n_epochs", 5)
        kwargs.setdefault("auto_insert_metric_name", False)
        kwargs["save_top_k"] = -1
        kwargs["enable_version_counter"] = True
        super().__init__(**kwargs)

    def _remove_checkpoint(self, trainer, filepath):
        # Retention is unconditional; a cleanup request must not stop training.
        if trainer.is_global_zero:
            print(f"Checkpoint retention: kept {filepath} (automatic cleanup skipped)", flush=True)

    def _save_checkpoint(self, trainer, filepath):
        # Lightning can version the moving resume pointer when resuming into
        # an existing directory. Only its registered last path may be updated.
        is_last = (
            filepath == self.last_model_path
            and re.fullmatch(r"last(?:-v\d+)?\.ckpt", Path(filepath).name) is not None
        )
        if not is_last and self.file_exists(filepath, trainer):
            original = Path(filepath)
            version = 1
            while self.file_exists(filepath, trainer):
                filepath = str(original.with_name(f"{original.stem}-v{version}{original.suffix}"))
                version += 1
            if trainer.is_global_zero:
                print(f"Checkpoint retention: preserved {original}; saving {filepath}", flush=True)
        super()._save_checkpoint(trainer, filepath)
