import os
import json
import sys
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import traceback

import lightning as L
import torch
import torch.multiprocessing
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from data.dataLoader import dataLoader
from models.model import AbstractModel
from models.utils import parse_args, save_args, save_checkpoint
from scripts.checkpoint_retention import RetainAllCheckpoints, validate_checkpoint_retention

torch.multiprocessing.set_sharing_strategy("file_system")


def load_conf_pretrained(model, path):
    checkpoint = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=True)
    source = checkpoint["state_dict"]
    target = model.state_dict()
    missing = set(target) - set(source)
    expected = {key for key in target if key.startswith("model.dock_output.")}
    if not expected or missing != expected or set(source) - set(target):
        raise ValueError("Conformation pretraining must omit only the new docking head")
    mismatched = [key for key in source if source[key].shape != target[key].shape]
    if mismatched:
        raise ValueError(f"Pretrained parameter shape mismatch: {mismatched}")
    target.update(source)
    model.load_state_dict(target, strict=True)
    print(
        f"Conformation pretrained: {path}, epoch={checkpoint.get('epoch')}, "
        f"loaded={len(source)}, new docking-head tensors={len(missing)}",
        flush=True,
    )


if __name__ == "__main__":
    cfg = parse_args()
    validate_checkpoint_retention(cfg.trainer)
    torch.autograd.set_detect_anomaly(bool(cfg.trainer.get("detect_anomaly", False)))

    if cfg.trainer.get("torch_num_threads", None) is not None:
        torch.set_num_threads(cfg.trainer.torch_num_threads)
    if cfg.trainer.get("torch_num_interop_threads", None) is not None:
        torch.set_num_interop_threads(cfg.trainer.torch_num_interop_threads)

    torch.set_float32_matmul_precision("medium")
    L.seed_everything(95028)

    dataset_train, y_mean, y_std = dataLoader(
        cfg.dataset.name, "train", cfg.dataset.path, cfg.dataset
    )
    validation_split = cfg.trainer.get("validation_split", "valid")
    if validation_split not in ["valid", "test"]:
        raise ValueError(
            f"validation_split must be 'valid' or 'test', got {validation_split!r}"
        )
    dataset_valid, _, _ = dataLoader(
        cfg.dataset.name, validation_split, cfg.dataset.path, cfg.dataset
    )

    # Apply after dataset imports, before fork workers start: some datasets set
    # their own default sharing strategy at import time.
    sharing_strategy = cfg.trainer.get("sharing_strategy", "file_system")
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    print(f"DataLoader sharing strategy: {sharing_strategy}", flush=True)

    model = AbstractModel(cfg, y_mean, y_std)

    if cfg.paths.get("conf_pretrained", "") and not cfg.paths.get("resume_from"):
        if cfg.task not in ["JointDockingConformation", "Docking"] or cfg.paths.get("try_load", ""):
            raise ValueError("conf_pretrained requires a fresh docking or docking/conformation run")
        load_conf_pretrained(model, cfg.paths.conf_pretrained)

    if cfg.paths.get("try_load", "") != "":
        checkpoint = torch.load(
            os.path.expanduser(cfg.paths.try_load), map_location=model.device
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"checkpoint loaded strictly: {cfg.paths.try_load}")

    precision = cfg.trainer.get("precision", "32-true")
    max_epochs = cfg.trainer.max_epochs
    gradient_clip_val = cfg.trainer.get("gradient_clip_val", 0)
    print("max epochs:", max_epochs, "precision:", precision)
    logger = TensorBoardLogger(
        cfg.paths.save_dir,
        name=cfg.paths.name,
        version=cfg.paths.get("version"),
    )
    save_args(cfg, logger.log_dir)
    wandb_logger = WandbLogger(
        project="cosmol",
        name=cfg.paths.name,
        save_dir=logger.log_dir,
        log_model=False,
    )

    devices = 1 if cfg.device.use_cpu else cfg.device.gpus

    checkpoint_callback = RetainAllCheckpoints(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename=cfg.trainer.get("checkpoint_filename", "epoch-{epoch:03d}"),
        monitor=cfg.trainer.get("checkpoint_monitor", None),
        mode=cfg.trainer.get("checkpoint_mode", "min"),
        every_n_epochs=cfg.trainer.get(
            "save_every_n_epochs", 5
        ),
        save_top_k=cfg.trainer.get("save_top_k", -1),
        save_last=cfg.trainer.get("save_last", True),
        auto_insert_metric_name=False,
    )

    # trainer = L.Trainer(max_epochs=max_epochs, precision=precision, enable_checkpointing=True, logger=logger, devices=devices, strategy="ddp", accelerator="cpu" if cfg.device.use_cpu else "gpu", callbacks=[RichProgressBar(leave=True)])
    if len(cfg.device.gpus) > 1:
        strategy = DDPStrategy(
            find_unused_parameters=cfg.trainer.get("find_unused_parameters", True),
            timeout=timedelta(seconds=cfg.trainer.get("ddp_timeout_seconds", 1800)),
        )
    else:
        strategy = "auto"
    trainer = L.Trainer(
        max_epochs=max_epochs,
        max_steps=cfg.trainer.get("max_steps", -1),
        limit_train_batches=cfg.trainer.get("limit_train_batches", None),
        limit_val_batches=cfg.trainer.get("limit_val_batches", None),
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", None),
        precision=precision,
        enable_checkpointing=True,
        logger=[logger, wandb_logger],
        devices=devices,
        strategy=strategy,
        accelerator="cpu" if cfg.device.use_cpu else "gpu",
        callbacks=[RichProgressBar(leave=True), checkpoint_callback],
        gradient_clip_val=gradient_clip_val,
    )

    fit_ok = False
    try:
        if cfg.trainer.get("validate_before_fit", False) and not cfg.paths.get("resume_from"):
            initial_metrics = trainer.validate(model=model, dataloaders=dataset_valid)
            if trainer.is_global_zero:
                with open(os.path.join(logger.log_dir, "validation_initial.json"), "w") as fout:
                    json.dump(initial_metrics, fout, indent=2)
            # Match the fresh training RNG seed after the diagnostic validation.
            L.seed_everything(95028)
        trainer.fit(
            model=model,
            train_dataloaders=dataset_train,
            val_dataloaders=dataset_valid,
            ckpt_path=os.path.expanduser(cfg.paths.resume_from)
            if cfg.paths.get("resume_from")
            else None,
        )
        if trainer.is_global_zero:
            save_checkpoint(
                logger.log_dir, cfg.paths.checkpoint,
                checkpoint_path=checkpoint_callback.last_model_path or None,
            )
        fit_ok = True
    except:
        traceback.print_exc()
        raise

    if fit_ok and cfg.trainer.get("run_test_after_fit", True):
        dataset_test, _, _ = dataLoader(
            cfg.dataset.name, "test", cfg.dataset.path, cfg.dataset
        )
        print("Testing dataset")
        trainer.test(
            model=model,
            dataloaders=dataset_test,
            ckpt_path=cfg.trainer.get("test_ckpt_path", None),
        )
