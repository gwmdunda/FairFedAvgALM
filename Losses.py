import torch
import torch.nn.functional as F

def weighted_loss(logits, targets, weights, mean = True):
    acc_loss = F.cross_entropy(logits, targets, reduction = 'none')
    if mean:
        weights_sum = weights.sum().item() + 1e-10
        acc_loss = torch.sum(acc_loss * weights / weights_sum)
    else:
        acc_loss = torch.sum(acc_loss * weights)
    return acc_loss