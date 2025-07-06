import argparse
import subprocess
import csv
import os
import datetime
def run_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.stdout

def extract_metric(output, key):
    for line in output.splitlines():
        if key in line and line.startswith("[RESULT]"):
            try:
                value_str = line.split(":", 1)[1].strip()
                for unit in ["GMac", "M", "ms/image", "MB", "%"]:
                    value_str = value_str.replace(unit, "")
                return float(value_str.split()[0])
            except Exception as e:
                print(f"[DEBUG] Error parsing line for {key}: {e}")
                return ""
    return ""

def get_model_filename(model, method, mode, state):
    # For baseline state, we add "_baseline"; for pruned, we use "_pruned"
    suffix = f"_{state}" if state == "baseline" else "_pruned"
    return f"{model}_{method}_{mode}{suffix}.pth"

def get_onnx_filename(model, method, mode, state):
    suffix = f"_{state}" if state == "baseline" else "_pruned"
    return f"{model}_{method}_{mode}{suffix}.onnx"

def main():
    # Create timestamped folder for this experiment run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(".outputs", f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=100, help="Number of images to evaluate on")
    parser.add_argument("--results_file", type=str, default="experiment_results.csv", help="CSV file to store results")
    parser.add_argument("--output_dir", type=str, default=".outputs/latest")
    args = parser.parse_args()

    # Define your experiment grid
    models = ["resnet18","resnet34","mobilenet_v2"]
    # models = ["resnet34"]
    methods = ["builtin"]
    modes = ["l1_unstructured", "ln_structured", "random_structured","random_unstructured"]
    compressions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # compressions = [0.1, 0.3, 0.5]

    # Open the CSV file in write mode initially to write header
    with open(args.results_file, "w", newline="") as f:
        fieldnames = ["model", "method", "mode", "compression", "state",
                      "flops", "params", "pytorch_latency", "pytorch_ram", "pytorch_accuracy",
                      "onnx_accuracy", "onnx_latency", "onnx_ram"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Iterate through your experiment grid
    for model in models:
        for method in methods:
            for mode in modes:
                for compression in compressions:
                    for state in ["baseline", "pruned"]:
                        print(f"\n===== Running: Model={model} | Method={method} | Mode={mode} | Compression={compression} | State={state} =====")
                        is_baseline = (state == "baseline")
                        # model_file = get_model_filename(model, method, mode, state)
                        # onnx_file = get_onnx_filename(model, method, mode, state)
                        model_file = os.path.join(run_output_dir, get_model_filename(model, method, mode, state))
                        onnx_file = os.path.join(run_output_dir, get_onnx_filename(model, method, mode, state))
                        print(f"[DEBUG-toii] Using model_file: {model_file}")
                        print(f"[DEBUG] Using onnx_file: {onnx_file}")
                        # --- Step 1: Run Pruning (or Baseline) ---
                        prune_cmd = [
                            "python", "run_pruning.py",
                            "--model", model,
                            "--method", method,
                            "--compression", str(compression),
                            "--mode", mode,
                            "--onnx",
                            "--output_dir", run_output_dir
                        ]
                        if is_baseline:
                            prune_cmd.append("--baseline")

                        print("→ Running:", " ".join(prune_cmd))
                        prune_output = run_command(prune_cmd)
                        print("[PRUNE OUTPUT]\n", prune_output)

                        result = {
                            "model": model,
                            "method": method,
                            "mode": mode,
                            "compression": compression,
                            "state": state,
                            "flops": extract_metric(prune_output, "FLOPs"),
                            "params": extract_metric(prune_output, "Parameters"),
                            "pytorch_latency": extract_metric(prune_output, "Latency:"),
                            "pytorch_ram": extract_metric(prune_output, "Peak RAM usage:"),
                        }

                        # --- Step 2: Evaluate PyTorch Accuracy ---
                        acc_cmd = [
                            "python", "eval_accuracy.py",
                            "--model_path", model_file,
                            "--dataset", "imagenette2-320",
                            "--subset_size", str(args.subset_size),
                            "--arch", model  # Pass the correct architecture flag
                        ]
                        print("→ Evaluating PyTorch Accuracy:", " ".join(acc_cmd))
                        acc_output = run_command(acc_cmd)
                        print("[PYTORCH ACC OUTPUT]\n", acc_output)
                        result["pytorch_accuracy"] = extract_metric(acc_output, "Accuracy")

                        # --- Step 3: Evaluate ONNX Accuracy ---
                        onnx_acc_cmd = [
                            "python", "eval_onnx.py",
                            "--onnx_path", onnx_file,
                            "--dataset", "imagenette2-320",
                            "--eval_accuracy",
                            "--subset_size", str(args.subset_size)
                        ]
                        print("→ Evaluating ONNX Accuracy:", " ".join(onnx_acc_cmd))
                        onnx_acc_output = run_command(onnx_acc_cmd)
                        print("[ONNX ACC OUTPUT]\n", onnx_acc_output)
                        result["onnx_accuracy"] = extract_metric(onnx_acc_output, "ONNX Accuracy")

                        # --- Step 4: Evaluate ONNX Latency ---
                        onnx_latency_cmd = [
                            "python", "eval_onnx.py",
                            "--onnx_path", onnx_file,
                            "--input_shape", "1,3,224,224"
                        ]
                        print("→ Evaluating ONNX Latency:", " ".join(onnx_latency_cmd))
                        onnx_latency_output = run_command(onnx_latency_cmd)
                        print("[ONNX LATENCY OUTPUT]\n", onnx_latency_output)
                        result["onnx_latency"] = extract_metric(onnx_latency_output, "ONNX Latency")
                        result["onnx_ram"] = extract_metric(onnx_latency_output, "ONNX Peak RAM Usage")

                        print("✅ Experiment complete:", result)

                        # Append result immediately to CSV so that you don't lose it if the process stops
                        with open(args.results_file, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=result.keys())
                            writer.writerow(result)
                            
    print(f"\n✅ All results saved to {args.results_file}")

if __name__ == "__main__":
    main()
