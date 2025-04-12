# portable_pruning/utils/eval_utils.py

import time
import torch
import psutil
import os

def measure_latency(model, device="cpu", input_size=(1, 3, 224, 224), runs=30):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)

    # Warm-up
    for _ in range(5):
        _ = model(dummy_input)

    torch.cuda.empty_cache() if device == "cuda" else None
    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy_input)
    end = time.time()

    avg_latency = (end - start) / runs * 1000  # in ms
    return avg_latency

def measure_memory(model, device='cpu'):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / 1024 / 1024
    return mem_mb