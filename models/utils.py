import argparse
import math
import os
from pathlib import Path

import easydict
import toml_rs
import torch
from torch.nn import functional as F
from torch_geometric.utils import scatter

raw_config_file = None


def resolve_path(path, *, output=False):
    """Resolve a user-supplied path relative to the working directory."""
    if path is None or os.fspath(path) == "":
        return path
    return str(Path(path).expanduser().resolve())


_OUTPUT_PATH_FIELDS = {"out_dir", "output_dir"}
_INPUT_PATH_FIELDS = {"input_dir", "data_root", "dekois_root"}


def _expand_paths(obj):
    if isinstance(obj, dict):
        return {
            key: resolve_path(value) if key in _OUTPUT_PATH_FIELDS | _INPUT_PATH_FIELDS
            else _expand_paths(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_expand_paths(value) for value in obj]
    if isinstance(obj, str) and obj.startswith(("./", "~/")):
        return str(Path(obj).expanduser().resolve())
    return obj


def _load_config_raw(path):
    global raw_config_file
    with open(Path(path).expanduser()) as handle:
        raw_config_file = handle.read()
    return easydict.EasyDict(_expand_paths(toml_rs.loads(raw_config_file)))


def parse_args(flag="-c"):
    parser = argparse.ArgumentParser()
    parser.add_argument(flag, "--config", help="TOML configuration file", required=True)
    return _load_config_raw(parser.parse_args().config)


def save_args(cfg, log_dir):
    global raw_config_file

    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "hparams.toml"), "w") as fout:
        fout.write(raw_config_file)


def load_toml_cfg(path, *, overrides=None):
    """Load a TOML configuration and apply command-line overrides."""
    cfg = _load_config_raw(path)
    if overrides:
        for key, value in overrides.items():
            if key in _OUTPUT_PATH_FIELDS | _INPUT_PATH_FIELDS:
                value = resolve_path(value)
            cfg[key] = value
    return cfg


import shutil


def save_checkpoint(log_dir: str, path: str, checkpoint_path=None):
    """Export the final checkpoint, preserving any existing destination."""
    src_dir = Path(log_dir) / "checkpoints"

    if not src_dir.is_dir():
        print(f"Directory not found: {src_dir}")
        return

    ckpt_files = list(src_dir.glob("*.ckpt"))

    if not ckpt_files:
        print(f"No .ckpt files found in: {src_dir}")
        return

    src_path = (Path(checkpoint_path).expanduser() if checkpoint_path else
                max(ckpt_files, key=lambda item: item.stat().st_mtime_ns))
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    dst_path = Path(path).expanduser().resolve()
    if dst_path.is_dir():
        dst_path = dst_path / src_path.name
    if dst_path == src_path.resolve():
        return str(dst_path)
    original = dst_path
    version = 1
    while dst_path.exists():
        dst_path = original.with_name(f"{original.stem}-v{version}{original.suffix}")
        version += 1

    os.makedirs(dst_path.parent, exist_ok=True)

    while True:
        try:
            with dst_path.open("xb") as destination, src_path.open("rb") as source:
                shutil.copyfileobj(source, destination)
            shutil.copystat(src_path, dst_path)
            break
        except FileExistsError:
            dst_path = original.with_name(f"{original.stem}-v{version}{original.suffix}")
            version += 1
    print(f"Copied: {src_path} → {dst_path}")
    return str(dst_path)


from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to 0.0 on a cosine schedule over
    the remaining ``num_training_steps-num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def batch_chirality_loss_no_gt(
    pred_pos, chiral_neighbors, chirality, margin: float = 0.05
):
    """
    pred_pos: (N, 3)
    chiral_neighbors: (M, 4)
    chirality: (N,)
    """

    device = pred_pos.device

    if chiral_neighbors.numel() == 0:
        return torch.tensor(0.0, device=device)

    # center indices (must align with chiral_neighbors)
    center_idx = torch.nonzero(chirality, as_tuple=False).squeeze(-1)

    # neighbor indices
    nbr_idx = chiral_neighbors[:, :3]  # only first 3 used

    # gather positions
    centers = pred_pos[center_idx]  # (M, 3)
    nbr_pos = pred_pos[nbr_idx]  # (M, 3, 3)

    # compute vectors
    vecs = nbr_pos - centers.unsqueeze(1)  # (M, 3, 3)

    def normalize(v):
        return F.normalize(v, dim=-1)

    v1 = normalize(vecs[:, 0])
    v2 = normalize(vecs[:, 1])
    v3 = normalize(vecs[:, 2])

    # Normalized oriented volume. The raw triple product scales with bond
    # lengths, while chirality only depends on signed local orientation.
    # det = torch.sum(v1 * torch.cross(v2, v3, dim=1), dim=1)  # (M,)
    # denom = (v1.norm(dim=1) * v2.norm(dim=1) * v3.norm(dim=1)).clamp_min(1e-8)
    # det = det / denom

    # det = torch.sum(v1 * torch.cross(v2, v3, dim=1), dim=1)  # (M,)

    det = torch.mean(v1 * torch.cross(v2, v3, dim=1), dim=1)  # (M,)

    # Keep a positive margin from the planar chirality boundary.
    loss_all = torch.relu(margin - det)

    return loss_all.mean()


def batch_chirality_loss_no_gt_before_terminal(
    pred_pos,
    chiral_neighbors,
    chirality,
    t=None,
    margin: float = 0.05,
    terminal: float = 1.0,
):
    """
    pred_pos: (N, 3)
    chiral_neighbors: (M, 4)
    chirality: (N,)
    t: (N,)
    """

    if t is not None:
        keep = (t <= terminal)[chirality != 0]
        chiral_neighbors = chiral_neighbors[keep]
        chirality = chirality.clone()
        chirality[t > terminal] = 0

    device = pred_pos.device

    if chiral_neighbors.numel() == 0:
        return torch.tensor(0.0, device=device)

    # center indices (must align with chiral_neighbors)
    center_idx = torch.nonzero(chirality, as_tuple=False).squeeze(-1)

    # neighbor indices
    nbr_idx = chiral_neighbors[:, :3]  # only first 3 used

    # gather positions
    centers = pred_pos[center_idx]  # (M, 3)
    nbr_pos = pred_pos[nbr_idx]  # (M, 3, 3)

    # compute vectors
    vecs = nbr_pos - centers.unsqueeze(1)  # (M, 3, 3)

    def normalize(v):
        return F.normalize(v, dim=-1)

    v1 = normalize(vecs[:, 0])
    v2 = normalize(vecs[:, 1])
    v3 = normalize(vecs[:, 2])

    # Normalized oriented volume. The raw triple product scales with bond
    # lengths, while chirality only depends on signed local orientation.
    # det = torch.sum(v1 * torch.cross(v2, v3, dim=1), dim=1)  # (M,)
    # denom = (v1.norm(dim=1) * v2.norm(dim=1) * v3.norm(dim=1)).clamp_min(1e-8)
    # det = det / denom

    # det = torch.sum(v1 * torch.cross(v2, v3, dim=1), dim=1)  # (M,)

    det = torch.mean(v1 * torch.cross(v2, v3, dim=1), dim=1)  # (M,)

    # Keep a positive margin from the planar chirality boundary.
    loss_all = torch.relu(margin - det)

    return loss_all.mean()


def chirality_orientation_loss(pred_pos, chiral_neighbors, chirality):
    """
    pred_pos: (N, 3)
    chiral_neighbors: (M, 4)
    chirality: (N,)
    """

    device = pred_pos.device

    if chiral_neighbors.numel() == 0:
        return torch.tensor(0.0, device=device)

    # center indices (must align with chiral_neighbors)
    center_idx = torch.nonzero(chirality, as_tuple=False).squeeze(-1)

    # neighbor indices
    nbr_idx = chiral_neighbors[:, :3]  # only first 3 used

    # gather positions
    centers = pred_pos[center_idx]  # (M, 3)
    nbr_pos = pred_pos[nbr_idx]  # (M, 3, 3)

    # compute vectors
    vecs = nbr_pos - centers.unsqueeze(1)  # (M, 3, 3)

    v1 = vecs[:, 0]
    v2 = vecs[:, 1]
    v3 = vecs[:, 2]

    # oriented volume
    det = torch.sum(v1 * torch.cross(v2, v3, dim=1), dim=1)  # (M,)

    # enforce positive orientation
    loss_all = torch.relu(-det)

    return loss_all.mean()


def rotation_geodesic_loss(R_pred, R_true, loss_factor_batch, reduction="mean"):
    """
    R_pred, R_true: (B, 3, 3)
    """
    R_diff = torch.matmul(R_pred.transpose(1, 2), R_true)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    angle = (
        torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
        * loss_factor_batch
    )
    return angle if reduction == "none" else angle.mean()


def min_dist_barrier_loss(
    pos_pred: torch.Tensor,  # (N, 3)
    edge_index: torch.Tensor,  # (2, E)
    mindist: torch.Tensor,  # (E,)
    t_graph: torch.Tensor,  # (N,)
    batch: torch.Tensor = None,
    t_threshold: float = 0.8,
    reduction: str = "mean",
):
    """
    Late-stage minimum distance barrier loss
    """

    active = (t_graph >= t_threshold).float()

    # ---------- edge distances ----------
    src, dst = edge_index
    diff = pos_pred[src] - pos_pred[dst]
    dist = torch.norm(diff, dim=1)  # (E,)

    loss_per_edge = F.relu(mindist - dist)

    # ---------- aggregate ----------
    loss = (
        scatter(
            scatter(loss_per_edge, edge_index[1], dim=0, reduce="max"),
            batch,
            dim=0,
            reduce="mean",
        )
        * active
    )
    return loss if reduction == "none" else loss.mean()
