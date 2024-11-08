import torch
import torch.nn.functional as F


def element_weighted_loss(y_hat, y, weights):
    loss = F.cross_entropy(y_hat, y, reduction='none')
    return loss @ weights / weights.sum()

def calculate_item_freq(item_num, item_id_tensor):
    return torch.bincount(item_id_tensor, minlength=item_num)

def weight_decay(item_freq, alpha, beta):
    return alpha - torch.tanh(beta * (item_freq-1))