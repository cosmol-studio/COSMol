import torch
import scipy.stats as stats

def count_f1_max(pred, target):
	"""
	    F1 score with the optimal threshold, Copied from TorchDrug.

	    This function first enumerates all possible thresholds for deciding positive and negative
	    samples, and then pick the threshold with the maximal F1 score.

	    Parameters:
	        pred (Tensor): predictions of shape :math:`(B, N)`
	        target (Tensor): binary targets of shape :math:`(B, N)`
    """

	order = pred.argsort(descending=True, dim=1)
	target = target.gather(1, order)
	precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
	recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
	is_start = torch.zeros_like(target).bool()
	is_start[:, 0] = 1
	is_start = torch.scatter(is_start, 1, order, is_start)

	all_order = pred.flatten().argsort(descending=True)
	order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
	order = order.flatten()
	inv_order = torch.zeros_like(order)
	inv_order[order] = torch.arange(order.shape[0], device=order.device)
	is_start = is_start.flatten()[all_order]
	all_order = inv_order[all_order]
	precision = precision.flatten()
	recall = recall.flatten()
	all_precision = precision[all_order] - \
	                torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
	all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
	all_recall = recall[all_order] - \
	             torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
	all_recall = all_recall.cumsum(0) / pred.shape[0]
	all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
	return all_f1.max()

def accuracy(pred, target):
	"""
	Classification accuracy.

	Suppose there are :math:`N` sets and :math:`C` categories.

	Parameters:
		pred (Tensor): prediction of shape :math:`(N, C)`
		target (Tensor): target of shape :math:`(N,)`
	"""
	# print(pred)
	# print(pred.max(1)[1].eq(target).sum().item() / target.size(0))
	a = (pred.argmax(dim=-1) == target).float()
	# print(a.mean())
	return a.mean()

def recovery(pred, target):
	"""
	Recovery rate.

	Suppose there are :math:`N` sets and :math:`C` categories.

	Parameters:
		pred (Tensor): prediction of shape :math:`(N, C)`
		target (Tensor): target of shape :math:`(N,)`
	"""
	return (pred.argmax(dim=-1) == target).float().mean()

def AUPRC(pred, target):
	"""
	Area under precision-recall curve (PRC).

	Parameters:
		pred (Tensor): predictions of shape :math:`(n,)`
		target (Tensor): binary targets of shape :math:`(n,)`
	"""
	pred, target = pred.flatten(), target.flatten()
	order = pred.argsort(descending=True)
	target = target[order]
	precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
	auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
	return auprc

def RMSE(pred, target):
	"""
	Root mean squared error.

	Parameters:
		pred (Tensor): predictions of shape :math:`(n,)`
		target (Tensor): targets of shape :math:`(n,)`
	"""
	return ((pred - target) ** 2).mean().sqrt()

def pearson(pred, target):
	"""
	Pearson correlation coefficient.

	Parameters:
		pred (Tensor): predictions of shape :math:`(n,)`
		target (Tensor): targets of shape :math:`(n,)`
	"""
	device = pred.device
	pred, target = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
	r = stats.pearsonr(target, pred)[0]
	return torch.tensor(r, device=device)

def spearman(pred, target):
	"""
	Spearman correlation coefficient.
	"""
	device = pred.device
	pred, target = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
	rho = stats.spearmanr(target, pred)[0]
	return torch.tensor(rho , device=device)
