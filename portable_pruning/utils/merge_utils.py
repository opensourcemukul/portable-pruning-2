import torch
import torch.nn as nn
import numpy as np
def merge_similar_filters(model, alpha=0.3):
    """
    Merge similar filters in Conv2D layers using L2 distance threshold.
    Args:
        model: torch.nn.Module
        alpha: L2 threshold for similarity
    Returns:
        Modified model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            W = module.weight.data
            out_channels = W.size(0)
            used = set()
            merged = torch.zeros_like(W)
            count = 0

            for i in range(out_channels):
                if i in used:
                    continue
                group = [W[i]]
                used.add(i)
                for j in range(i + 1, out_channels):
                    if j in used:
                        continue
                    if torch.norm(W[i] - W[j]) < alpha:
                        group.append(W[j])
                        used.add(j)
                avg = torch.stack(group).mean(dim=0)
                merged[count] = avg
                count += 1

            module.weight.data[:count] = merged[:count]
            if count < out_channels:
                module.weight.data[count:] = 0.0

    return model

def merge_channels(weights, kde_scores, sim_matrix, compression_ratio):
    """
    Merge redundant channels based on similarity and KDE scores.
    Args:
        weights: np.ndarray of shape (C_out, C_in, kH, kW)
        kde_scores: np.ndarray of shape (C_out,)
        sim_matrix: np.ndarray of shape (C_out, C_out)
        compression_ratio: float (e.g., 0.5 for 50% pruning)
    Returns:
        merged_weights: np.ndarray of shape (new_C_out, C_in, kH, kW)
        keep_idxs: list of indices of kept channels
    """
    C_out = weights.shape[0]
    num_keep = int(C_out * (1 - compression_ratio))
    if num_keep < 1:
        num_keep = 1

    scores = -kde_scores + sim_matrix.mean(axis=1)  # simple score heuristic
    keep_idxs = np.argsort(scores)[-num_keep:]

    merged_weights = weights[keep_idxs]
    return merged_weights, keep_idxs