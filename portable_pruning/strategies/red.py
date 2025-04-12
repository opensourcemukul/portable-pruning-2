import torch
import torch.nn as nn
import numpy as np
from portable_pruning.utils.kde_utils import kde_quantize_weights
from portable_pruning.utils.merge_utils import merge_similar_filters

class Pruner:
    def __init__(self, model, compression_ratio=0.5, alpha=0.3, tau=0.05, device='cpu'):
        self.model = model.to(device)
        self.compression_ratio = compression_ratio
        self.device = device
        self.alpha = alpha
        self.tau = tau

    def prune(self):
        print(f"[RED] Applying RED with α={self.alpha}, τ={self.tau}")

        # Stage 1: KDE-based scalar quantization
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.requires_grad:
                original_weights = module.weight.data.clone().cpu().numpy()
                quantized_weights = kde_quantize_weights(original_weights, tau=self.tau)
                module.weight.data = torch.tensor(quantized_weights).to(module.weight.device)

        # Stage 2: Merge similar filters
        self.model = merge_similar_filters(self.model, alpha=self.alpha)

        return self.model