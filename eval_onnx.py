import argparse
import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import psutil

# Hardcoded ImageNet class indices for Imagenette (sorted)
imagenette_indices = sorted([
    0,    # Tench
    217,  # English springer
    482,  # Cassette player
    491,  # Chain saw
    497,  # Church
    566,  # French horn
    569,  # Garbage truck
    571,  # Gas pump
    574,  # Golf ball
    701   # Parachute
])

# Map ImageNet class index → dataset label index (0 to 9)
imagenette_index_to_label = {v: i for i, v in enumerate(imagenette_indices)}

def measure_latency_onnx(session, input_tensor):
    start = time.time()
    _ = session.run(None, {"input": input_tensor})
    end = time.time()
    return (end - start) * 1000  # in milliseconds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--eval_accuracy", action="store_true")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_shape", type=str, default="1,3,224,224")
    args = parser.parse_args()

    # Load ONNX model
    print(f"[INFO] Loading ONNX model from {args.onnx_path}")
    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    shape = tuple(map(int, args.input_shape.split(",")))

    if args.eval_accuracy:
        if not args.dataset:
            raise ValueError("--dataset is required for accuracy evaluation")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(args.dataset, transform=transform)

        if args.subset_size:
            dataset = torch.utils.data.Subset(dataset, range(args.subset_size))

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        is_imagenette = "imagenette" in args.dataset.lower()

        for images, labels in dataloader:
            ort_inputs = {input_name: images.numpy()}
            ort_outputs = session.run(None, ort_inputs)
            logits = torch.tensor(ort_outputs[0])

            if is_imagenette:
                # Filter logits to only 10 classes
                filtered_logits = logits[:, imagenette_indices]
                _, preds_local = torch.max(filtered_logits, 1)
                # Convert back to dataset label indices
                preds = torch.tensor(
                    [imagenette_index_to_label[imagenette_indices[i.item()]] for i in preds_local]
                )
            else:
                _, preds = torch.max(logits, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        print(f"[RESULT] ONNX Accuracy: {acc:.2f}%")

    else:
        # Latency measurement only
        print("[INFO] Measuring ONNX inference latency...")

        dummy_input = np.random.randn(*shape).astype(np.float32)
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024

        latency = measure_latency_onnx(session, dummy_input)

        end_mem = process.memory_info().rss / 1024 / 1024
        peak_ram = end_mem - start_mem

        print(f"[RESULT] ONNX Latency: {latency:.2f} ms/image")
        print(f"[RESULT] ONNX Peak RAM Usage: {peak_ram:.2f} MB")

if __name__ == "__main__":
    main()