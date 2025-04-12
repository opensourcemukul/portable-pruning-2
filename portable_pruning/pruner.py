import torch
import torch.nn as nn

from portable_pruning.strategies import l1_norm, batchnorm_gamma, deep_inversion

class PortablePruner:
    def __init__(self, model, method="l1", compression=0.5):
        """
        Initialize the pruner with the model and pruning method.

        Args:
            model (torch.nn.Module): The CNN model to prune.
            method (str): Pruning strategy: 'l1', 'batchnorm', 'deepinv'.
            compression (float): Fraction of FLOPs to remove (e.g., 0.5 for 50% pruning).
        """
        self.model = model
        self.method = method
        self.compression = compression

    def prune(self):
        """
        Apply the selected pruning method to the model.

        Returns:
            torch.nn.Module: The pruned model.
        """
        if self.method == "l1":
            return l1_norm.prune_model(self.model, self.compression)
        elif self.method == "batchnorm":
            return batchnorm_gamma.prune_model(self.model, self.compression)
        elif self.method == "deepinv":
            return deep_inversion.prune_model(self.model, self.compression)
        else:
            raise ValueError(f"[ERROR] Unsupported pruning method: {self.method}")