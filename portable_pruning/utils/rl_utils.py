import torch
import torch.nn as nn  
import torch.nn.functional as F
import numpy as np

class ChannelSimilarity:
    def __call__(self, weight_tensor):
        """
        Compute cosine similarity between output channels.
        Shape: (C_out, C_in, kH, kW) → flatten to (C_out, -1)
        Returns: [C_out x C_out] similarity matrix
        """
        W = weight_tensor.view(weight_tensor.size(0), -1)  # Flatten
        W = F.normalize(W, dim=1)
        sim_matrix = torch.matmul(W, W.T)
        return sim_matrix


class RLController:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def sample_mask(self, similarity_matrix, compression_ratio=0.5):
        """
        Sample a binary pruning mask based on similarity.
        Higher similarity → higher chance to prune one of the channels.
        """
        num_channels = similarity_matrix.size(0)
        num_keep = int((1 - compression_ratio) * num_channels)

        # Summed similarity per channel → less unique → prune more
        redundancy_score = similarity_matrix.sum(dim=1)
        prob = F.softmax(-redundancy_score / self.temperature, dim=0)
        keep_indices = torch.multinomial(prob, num_keep, replacement=False)

        mask = torch.zeros(num_channels, dtype=torch.bool)
        mask[keep_indices] = True
        return mask


import copy
def compute_reward(model, layer_name, mask, input_tensor, reward_temperature=1.0):
    """
    Evaluate reward after applying pruning mask to the target layer and its downstream consumer.
    """
    model_orig = copy.deepcopy(model).eval()
    test_model = copy.deepcopy(model).eval()

    # Locate target conv
    layer_parts = layer_name.split('.')
    module = test_model
    for part in layer_parts:
        module = getattr(module, part)

    keep_idxs = torch.nonzero(mask).squeeze(1)

    # Apply pruning to output channels
    # module.weight.data = module.weight.data[keep_idxs]
    # if module.bias is not None:
    #     module.bias.data = module.bias.data[keep_idxs]
    # module.out_channels = len(keep_idxs)
    if keep_idxs.max().item() < module.weight.shape[0]:
        module.weight.data = module.weight.data[keep_idxs]
        if module.bias is not None:
            module.bias.data = module.bias.data[keep_idxs]
        module.out_channels = len(keep_idxs)
    else:
        print(f"[Reward] ⚠️ Skipping layer {layer_name} — keep_idxs exceed out_channels ({module.weight.shape[0]})")
        return -1.0  # Invalid reward

    # Find parent and index
    def find_parent_and_index(root, target):
        for name, mod in root.named_modules():
            for idx, child in enumerate(mod.children()):
                if child is target:
                    return mod, idx
        return None, None

    parent, idx = find_parent_and_index(test_model, module)

    # Update next BN
    if parent and idx + 1 < len(list(parent.children())):
        next_layer = list(parent.children())[idx + 1]
        if isinstance(next_layer, nn.BatchNorm2d):
            next_layer.weight.data = next_layer.weight.data[keep_idxs]
            next_layer.bias.data = next_layer.bias.data[keep_idxs]
            next_layer.running_mean = next_layer.running_mean[keep_idxs]
            next_layer.running_var = next_layer.running_var[keep_idxs]
            next_layer.num_features = len(keep_idxs)

    # Update consumer layer’s input channels
    # Special logic for ResNet: if inside a BasicBlock, patch conv2 input
    # Update consumer layer’s input channels (conv2 in BasicBlock)
    if 'layer' in layer_name and '.0.conv1' in layer_name:
        block_name = layer_name.rsplit('.', 1)[0]
        block = dict(test_model.named_modules())[block_name]
        if hasattr(block, 'conv2') and isinstance(block.conv2, nn.Conv2d):
            conv2 = block.conv2
            # if conv2.weight.shape[1] >= keep_idxs.max().item() + 1:
            #     conv2.weight.data = conv2.weight.data[:, keep_idxs]
            #     conv2.in_channels = len(keep_idxs)
            # else:
            #     print(f"[Reward] ⚠️ Skipping fix for {layer_name}.conv2 — keep_idxs exceed in_channels ({conv2.weight.shape[1]})")
            if keep_idxs.max().item() < conv2.in_channels:
                conv2.weight.data = conv2.weight.data[:, keep_idxs]
                conv2.in_channels = len(keep_idxs)
            else:
                print(f"[Reward] ⚠️ Skipping fix for {layer_name}.conv2 — keep_idxs exceed in_channels ({conv2.in_channels})")

    # Inference and reward
    with torch.no_grad():
        try:
            out_orig = model_orig(input_tensor)
            out_test = test_model(input_tensor)
            mse_loss = F.mse_loss(out_orig, out_test)
            reward = torch.exp(-mse_loss / reward_temperature).item()
        except Exception as e:
            print(f"[Reward] Model failed: {e}")
            reward = -1.0

    return reward