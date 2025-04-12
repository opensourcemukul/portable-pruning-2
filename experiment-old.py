#!/usr/bin/env python
import subprocess
import csv
import os
import re
import itertools
import sys
import time

# Define experiment grid
models = ["resnet18", "mobilenet_v2"]
methods = ["builtin"]  # Extend with other methods if desired
modes = ["l1_unstructured", "ln_structured", "random_structured"]
compressions = [0.1, 0.3, 0.5, 0.7, 0.9]
states = ["baseline", "pruned"]  # baseline = unpruned, pruned = with pruning

# For CIFAR-10 evaluation, we simply pass "cifar10" as the dataset indicator.
VAL_DATASET = "cifar10"

# Output CSV file
output_csv = "experiment_results.csv"

# CSV fieldnames
fieldnames = [
    "model", "method", "mode", "compression", "state",
    "pytorch_latency", "pytorch_ram", "flops", "params",
    "onnx_latency", "onnx_ram", "pytorch_accuracy", "onnx_accuracy"
]

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()

    for model, method, mode, compression, state in itertools.product(models, methods, modes, compressions, states):
        print(f"\n\n===== Running experiment: Model={model} | Method={method} | Mode={mode} | Compression={compression} | State={state} =====")
        
        # Build command for run_pruning.py; append --baseline if state is "baseline"
        cmd = [sys.executable, "run_pruning.py",
               "--model", model,
               "--method", method,
               "--compression", str(compression),
               "--mode", mode,
               "--onnx"]
        if state == "baseline":
            cmd.append("--baseline")
        print("Executing:", " ".join(cmd))
        run_result = subprocess.run(cmd, capture_output=True, text=True)
        time.sleep(1)
        output = run_result.stdout

        # Parse metrics from run_pruning.py output.
        flops_params = re.findall(r"\[RESULT\] FLOPs:\s*([\d\.]+).*?Parameters:\s*([\d\.]+)", output)
        if flops_params:
            flops, params = flops_params[-1]
        else:
            flops = params = ""
        latencies = re.findall(r"\[RESULT\] Latency:\s*([\d\.]+)\s*ms/image", output)
        pytorch_latency = latencies[-1] if latencies else ""
        ram_usages = re.findall(r"\[RESULT\] Peak RAM usage:\s*([\d\.]+)\s*MB", output)
        pytorch_ram = ram_usages[-1] if ram_usages else ""
        
        state_suffix = "_baseline" if state == "baseline" else "_pruned"
        checkpoint = f"{model}_{method}{state_suffix}.pth"
        onnx_file = f"{model}_{method}{state_suffix}_pruned.onnx"

        # Evaluate PyTorch accuracy using CIFAR-10.
        if os.path.exists(checkpoint):
            cmd_acc_pt = [sys.executable, "eval_accuracy.py",
                          "--model_path", checkpoint,
                          "--dataset", VAL_DATASET,
                          "--arch", model,
                          "--device", "cpu"]
            acc_pt_res = subprocess.run(cmd_acc_pt, capture_output=True, text=True)
            acc_pt_out = acc_pt_res.stdout
            acc_pt_match = re.search(r"\[RESULT\] Accuracy:\s*([\d\.]+)%", acc_pt_out)
            pytorch_accuracy = acc_pt_match.group(1) if acc_pt_match else ""
        else:
            print(f"[WARNING] Checkpoint {checkpoint} not found.")
            pytorch_accuracy = ""

        # Evaluate ONNX accuracy using eval_onnx.py with --eval_accuracy.
        if os.path.exists(onnx_file):
            cmd_acc_onnx = [sys.executable, "eval_onnx.py",
                            "--onnx_path", onnx_file,
                            "--input_shape", "1,3,224,224",
                            "--eval_accuracy",
                            "--dataset", VAL_DATASET,
                            "--batch_size", "32"]
            acc_onnx_res = subprocess.run(cmd_acc_onnx, capture_output=True, text=True)
            acc_onnx_out = acc_onnx_res.stdout
            acc_onnx_match = re.search(r"\[RESULT\] ONNX Accuracy:\s*([\d\.]+)%", acc_onnx_out)
            onnx_accuracy = acc_onnx_match.group(1) if acc_onnx_match else ""
        else:
            print(f"[WARNING] ONNX file {onnx_file} not found.")
            onnx_accuracy = ""
        
        # Evaluate ONNX latency and RAM (non-accuracy run)
        if os.path.exists(onnx_file):
            cmd_onnx = [sys.executable, "eval_onnx.py",
                        "--onnx_path", onnx_file,
                        "--input_shape", "1,3,224,224"]
            onnx_res = subprocess.run(cmd_onnx, capture_output=True, text=True)
            onnx_out = onnx_res.stdout
            latency_match = re.search(r"\[RESULT\] ONNX Latency:\s*([\d\.]+)\s*ms/image", onnx_out)
            onnx_latency = latency_match.group(1) if latency_match else ""
            ram_match = re.search(r"\[RESULT\] ONNX Peak RAM Usage:\s*([\d\.]+)\s*MB", onnx_out)
            onnx_ram = ram_match.group(1) if ram_match else ""
        else:
            onnx_latency = onnx_ram = ""
        
        row = {
            "model": model,
            "method": method,
            "mode": mode,
            "compression": compression,
            "state": state,
            "pytorch_latency": pytorch_latency,
            "pytorch_ram": pytorch_ram,
            "flops": flops,
            "params": params,
            "onnx_latency": onnx_latency,
            "onnx_ram": onnx_ram,
            "pytorch_accuracy": pytorch_accuracy,
            "onnx_accuracy": onnx_accuracy
        }
        
        writer.writerow(row)
        csvfile.flush()
        print("Experiment complete:", row)
        time.sleep(1)

print("\nAll experiments finished. Results saved in", output_csv)