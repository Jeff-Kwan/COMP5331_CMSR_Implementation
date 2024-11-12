import torch
import torch.nn.functional as F

def element_weighted_loss(y_hat, y, weights):
    loss = F.cross_entropy(y_hat, y, reduction='none')
    loss = loss.to(torch.float32)
    weights = weights.to(torch.float32)
    return loss @ weights / weights.sum()
def calculate_item_freq(item_num, item_id_tensor):
    item_freq_tensor = torch.bincount(item_id_tensor, minlength=item_num)
    return {i: freq.item() for i, freq in enumerate(item_freq_tensor)}

def weight_decay(item_freq, alpha, beta):
    item_freq_tensor = torch.tensor(item_freq) if not isinstance(item_freq, torch.Tensor) else item_freq
    return alpha - torch.tanh(beta * (item_freq_tensor - 1))
