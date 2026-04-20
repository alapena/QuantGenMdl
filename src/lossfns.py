import torch

def quantum_mean_infidelity(pred, true):
    # pred and true are [n_data, n_features] complex tensors
    inner_product = torch.linalg.vecdot(true, pred, dim=-1) # [n_data]
    fidelity = inner_product.abs().pow(2)
    infidelity = 1 - fidelity
    return infidelity.mean()