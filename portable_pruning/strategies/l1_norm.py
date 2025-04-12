# portable_pruning/strategies/l1_norm.py

import torch
import torch.nn as nn

def prune_model(model, compression=0.5):
    print("[L1] Pruning with L1-norm...")
    # Placeholder: iterate over conv layers and zero out lowest-L1 channels
    # NOTE: This is a stub implementation
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data.abs().sum(dim=(1, 2, 3))
            num_channels = weight.size(0)
            keep = int((1 - compression) * num_channels)
            threshold = torch.topk(weight, keep, largest=True).values.min()
            mask = (weight >= threshold).float().view(-1, 1, 1, 1)
            module.weight.data *= mask
    return model
