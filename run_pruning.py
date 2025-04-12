#!/usr/bin/env python
import sys
import os
import argparse
import torch
import torchvision.models as models
from portable_pruning.utils.eval_utils import measure_latency, measure_memory
from portable_pruning.utils.flops_counter import compute_flops

# Import pruning strategies (if needed)
from portable_pruning.strategies import l1_norm, batchnorm_gamma, red, autodfp

def export_to_onnx(model, dummy_input, export_path):
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[EXPORT] ONNX model saved to {export_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'mobilenet_v2'])
    parser.add_argument('--method', type=str, required=True,
                        choices=['l1', 'batchnorm', 'red', 'autodfp', 'redpp', 'builtin'])
    parser.add_argument('--compression', type=float, required=True,
                        help="Fraction of compression (e.g., 0.5 for 50% pruning)")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--onnx', action='store_true', help="Export model to ONNX")
    parser.add_argument('--baseline', action='store_true',
                        help="If set, run baseline evaluation (no pruning applied)")

    # Additional parameters for specific methods
    parser.add_argument('--alpha', type=float, default=0.3, help='RED: filter similarity threshold')
    parser.add_argument('--tau', type=float, default=0.05, help='RED: KDE quantization threshold')
    parser.add_argument('--max_iters', type=int, default=100, help='AutoDFP: Number of RL samples per layer')
    parser.add_argument('--reward_temp', type=float, default=1.0, help='AutoDFP: Reward temperature for sampling')

    # For torch_builtin
    parser.add_argument('--mode', type=str, default='l1_unstructured',
                        choices=['l1_unstructured', 'ln_structured', 'random_structured'],
                        help='Pruning mode for torch_builtin')
    parser.add_argument("--state", type=str, default="pruned", choices=["baseline", "pruned"], help="Whether baseline or pruned")

    args = parser.parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = getattr(models, args.model)(pretrained=True).to(args.device)

    if args.baseline:
        print("[INFO] Running baseline evaluation (no pruning applied)")
        pruned_model = model
    else:
        print(f"[INFO] Applying {args.method} pruning with {args.compression*100:.0f}% compression")
        if args.method == 'l1':
            pruner = l1_norm.Pruner(model, compression_ratio=args.compression, device=args.device)
        elif args.method == 'batchnorm':
            pruner = batchnorm_gamma.Pruner(model, compression_ratio=args.compression, device=args.device)
        elif args.method == 'red':
            pruner = red.Pruner(model, compression_ratio=args.compression, alpha=args.alpha, tau=args.tau, device=args.device)
        elif args.method == 'redpp':
            from portable_pruning.strategies.redpp import REDPPPruner
            pruner = REDPPPruner(model, compression=args.compression, device=args.device)
            pruned_model = pruner.prune()
        elif args.method == 'builtin':
            from portable_pruning.strategies import torch_builtin
            pruner = torch_builtin.Pruner(model, compression_ratio=args.compression, device=args.device, mode=args.mode)
        elif args.method == 'autodfp':
            pruner = autodfp.Pruner(model, compression_ratio=args.compression, device=args.device,
                                    max_iters=args.max_iters, reward_temperature=args.reward_temp)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        if args.method != 'redpp':
            pruned_model = pruner.prune()
        if hasattr(pruner, "finalize"):
            pruner.finalize()

    # Measure metrics
    print("[INFO] Measuring FLOPs and Params...")
    macs, params = compute_flops(pruned_model, input_res=(3,224,224))
    # print(f"[RESULT] FLOPs: {macs}, Parameters: {params}")
    print(f"[RESULT] FLOPs: {macs}")
    print(f"[RESULT] Parameters: {params}")
    print("[INFO] Measuring latency and memory usage...")
    latency = measure_latency(pruned_model, device=args.device)
    memory = measure_memory(pruned_model, device=args.device)
    print(f"[RESULT] Latency: {latency:.2f} ms/image")
    print(f"[RESULT] Peak RAM usage: {memory:.2f} MB")

    state_suffix = "_baseline" if args.baseline else "_pruned"
    # Construct name suffix
    # Construct proper suffix including mode (for builtin methods)
    mode_suffix = f"_{args.mode}" if hasattr(args, "mode") and args.method == "builtin" else ""

    # Save ONNX
    if args.onnx:
        dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
        onnx_filename = f"{args.model}_{args.method}{mode_suffix}{state_suffix}.onnx"
        export_to_onnx(pruned_model, dummy_input, onnx_filename)

    # Save PyTorch model
    model_filename = f"{args.model}_{args.method}{mode_suffix}{state_suffix}.pth"
    torch.save(pruned_model.state_dict(), model_filename)
    print(f"[INFO] Model saved to {model_filename}")
    # if args.onnx:
    #     dummy_input = torch.randn(1,3,224,224).to(args.device)
    #     onnx_filename = f"{args.model}_{args.method}{state_suffix}_pruned.onnx"
    #     export_to_onnx(pruned_model, dummy_input, onnx_filename)

    # checkpoint = f"{args.model}_{args.method}{state_suffix}.pth"
    # torch.save(pruned_model.state_dict(), checkpoint)
    # print(f"[INFO] Model saved to {checkpoint}")

if __name__ == "__main__":
    main()