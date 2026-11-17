import torch


def pocket_vector_l1_loss(prediction, target, proximity_weights=None):
    """Mean coordinate L1, optionally weighted by ground-truth pocket proximity.

    Keep the mean over all nodes (not a sum-of-weights normalization), matching
    the historical proximity weighting while retaining the Joint L1 metric.
    """
    if proximity_weights is None:
        return torch.nn.functional.l1_loss(prediction, target)
    weights = proximity_weights.detach().reshape(-1).to(prediction)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Pocket vectors must have matching [nodes, coordinates] shapes")
    if weights.numel() != prediction.shape[0]:
        raise ValueError("One ground-truth proximity weight is required per pocket vector")
    return ((prediction - target).abs().mean(dim=-1) * weights).mean()
