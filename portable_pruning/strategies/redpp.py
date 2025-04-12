import torch
import torch.nn as nn
import numpy as np
from portable_pruning.utils.kde_utils import estimate_kde
from portable_pruning.utils.merge_utils import merge_channels

class REDPPPruner:
    def __init__(self, model, compression=0.5, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device
        self.compression = compression
        self.prev_keep_idxs = None  # Track output channels to align with next conv's input

    def prune(self):
        print(f"[RED++] 🔧 Pruning with RED++ at {int(self.compression * 100)}% compression")
        prev_keep_idxs = None  # 👈 Initialize here
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Skip conv1 to avoid bn1 mismatch
                if name == "conv1":
                    print(f"[RED++] ⚠️ Skipping {name} to avoid mismatch with bn1")
                    self.prev_keep_idxs = None
                    continue

                W = module.weight.data.cpu().numpy()
                C_out = W.shape[0]
                if C_out < 4:
                    self.prev_keep_idxs = None
                    continue

                print(f"[RED++] → Processing {name}")
                flattened = W.reshape(C_out, -1)
                kde_scores = estimate_kde(flattened)

                sim_matrix = np.zeros((C_out, C_out))
                for i in range(C_out):
                    for j in range(i + 1, C_out):
                        sim = 1.0 - np.linalg.norm(flattened[i] - flattened[j]) / (np.linalg.norm(flattened[i]) + 1e-8)
                        sim_matrix[i, j] = sim_matrix[j, i] = sim

                merged_weights, keep_idxs = merge_channels(W, kde_scores, sim_matrix, self.compression)
                keep_idxs = torch.tensor(keep_idxs).long()

                new_C_out = merged_weights.shape[0]
                print(f"[RED++] 🧩 Reduced {C_out} → {new_C_out} filters in {name}")

                # Fix input channels if needed
                if self.prev_keep_idxs is not None and merged_weights.shape[1] != len(self.prev_keep_idxs):
                    if merged_weights.shape[1] >= self.prev_keep_idxs.max().item() + 1:
                        merged_weights = merged_weights[:, self.prev_keep_idxs.numpy(), :, :]
                        print(f"[RED++] 🔁 Fixed input channels of {name}")
                    else:
                        print(f"[RED++] ⚠️ Skipping input fix for {name} (in_channels={merged_weights.shape[1]})")

                merged_weights_tensor = torch.tensor(merged_weights, dtype=module.weight.dtype)
                # Update input channels BEFORE assignment
                if prev_keep_idxs is not None:
                    merged_weights_tensor = merged_weights_tensor[:, prev_keep_idxs]

                # Assign weights
                module.weight.data[:new_C_out] = merged_weights_tensor
                module.weight.data[new_C_out:] = 0.0
                module.out_channels = new_C_out

                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_idxs]

                self._prune_next_bn(name, keep_idxs)
                self.prev_keep_idxs = keep_idxs

        return self.model

    def _prune_next_bn(self, conv_name, keep_idxs):
        # Avoid pruning bn1 — tied to conv1 which we skipped
        if conv_name == "conv1":
            print(f"[RED++] ⚠️ Skipping BN pruning after {conv_name} (bn1)")
            return

        found = False
        modules = list(self.model.named_modules())
        for i, (name, module) in enumerate(modules):
            if name == conv_name and i + 1 < len(modules):
                next_name, next_module = modules[i + 1]
                if isinstance(next_module, nn.BatchNorm2d):
                    print(f"[RED++] 🩺 Pruning BN: {next_name}")
                    next_module.weight.data = next_module.weight.data[keep_idxs]
                    next_module.bias.data = next_module.bias.data[keep_idxs]
                    next_module.running_mean = next_module.running_mean[keep_idxs]
                    next_module.running_var = next_module.running_var[keep_idxs]
                    next_module.num_features = len(keep_idxs)
                    found = True
                    break
        if not found:
            print(f"[RED++] ⚠️ No BN found after {conv_name}")