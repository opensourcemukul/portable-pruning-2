import numpy as np
from scipy.stats import gaussian_kde

def kde_quantize_weights(weights, tau=0.05):
    """
    Quantize weights by KDE-based clustering of similar values.
    Args:
        weights: numpy array of shape (C_out, C_in, kH, kW)
        tau: max distance between a weight and KDE peak to quantize
    """
    flat = weights.flatten()
    kde = gaussian_kde(flat)
    x_vals = np.linspace(flat.min(), flat.max(), 512)
    density = kde(x_vals)

    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peaks.append(x_vals[i])

    quantized = np.copy(flat)
    for i in range(len(flat)):
        for peak in peaks:
            if abs(flat[i] - peak) < tau:
                quantized[i] = peak
                break

    return quantized.reshape(weights.shape)

def estimate_kde(X, bandwidth='scott'):
    """
    Estimate KDE score for each channel.
    Args:
        X: np.ndarray of shape (C, N) where C is number of channels, N is flattened activations
        bandwidth: str or float, KDE bandwidth (passed to gaussian_kde)
    Returns:
        np.ndarray of shape (C,) representing the entropy of each channel
    """
    kde_scores = []
    for c in range(X.shape[0]):
        try:
            kde = gaussian_kde(X[c], bw_method=bandwidth)
            samples = np.linspace(X[c].min(), X[c].max(), 512)
            density = kde(samples)
            density /= density.sum() + 1e-8
            entropy = -np.sum(density * np.log(density + 1e-8))
        except Exception:
            entropy = float('inf')
        kde_scores.append(entropy)
    return np.array(kde_scores)