import pandas as pd
import matplotlib.pyplot as plt
import os

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def plot_flops_vs_latency(df):
    plt.figure()
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        x = subset['flops'].str.replace(" GMac", "").astype(float)
        y = subset['latency_ms'].astype(float)
        plt.scatter(x, y, label=method)
    plt.xlabel("FLOPs (GMac)")
    plt.ylabel("Latency (ms/image)")
    plt.title("FLOPs vs Latency")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/flops_vs_latency.png")

def plot_compression_vs_ram(df):
    plt.figure()
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        x = subset['compression'].astype(float) * 100
        y = subset['ram_mb'].astype(float)
        plt.plot(x, y, marker='o', label=method)
    plt.xlabel("Compression (%)")
    plt.ylabel("RAM Usage (MB)")
    plt.title("Compression vs RAM Usage")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/compression_vs_ram.png")

def plot_comparison_bars(df):
    plt.figure(figsize=(8, 5))
    metrics = ["baseline_latency", "pruned_latency", "baseline_ram", "pruned_ram"]
    for i, metric in enumerate(metrics):
        plt.bar(i, df[metric].astype(float).mean(), label=metric)
    plt.xticks(range(len(metrics)), metrics, rotation=45)
    plt.title("Baseline vs Pruned (Avg across runs)")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("plots/baseline_vs_pruned_bar.png")

def main():
    os.makedirs("plots", exist_ok=True)

    results = load_csv("benchmarks/results.csv")
    comparison = load_csv("benchmarks/comparison.csv")

    plot_flops_vs_latency(results)
    plot_compression_vs_ram(results)
    plot_comparison_bars(comparison)

    print("✅ Plots saved in 'plots/' folder.")

if __name__ == "__main__":
    main()