import argparse
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# For clean multiprocessing on macOS
import torch.multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# Imagenette class synsets (these are the folder names in imagenette)
imagenette_synsets = [
    "n01440764",  # Tench
    "n02102040",  # English springer
    "n02979186",  # Cassette player
    "n03000684",  # Chain saw
    "n03028079",  # Church
    "n03394916",  # French horn
    "n03417042",  # Garbage truck
    "n03425413",  # Gas pump
    "n03445777",  # Golf ball
    "n03888257"   # Parachute
]

def get_imagenette_indices():
    # Hardcoded indices from ImageNet corresponding to the imagenette synsets.
    # These are based on the ImageNet class ordering used by pretrained models.
    return sorted([
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint (.pth file)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset (for ImageFolder) or 'cifar10' etc.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--subset_size", type=int, default=None, help="If set, evaluate only this many samples")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2"], help="Architecture of the model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the proper model architecture
    if args.arch == "resnet18":
        model = models.resnet18(weights=None)
    elif args.arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    # Load model checkpoint (the file should be saved as state_dict)
    # We use weights_only=False to ensure full state_dict is loaded properly.
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    # Define transforms (using ImageNet standard normalization)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset. If the dataset argument is a directory, assume it's in ImageFolder format.
    if os.path.isdir(args.dataset):
        dataset = datasets.ImageFolder(args.dataset, transform=transform)
    else:
        raise ValueError("Unsupported dataset format; please provide a directory in ImageFolder format.")

    if args.subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(args.subset_size))

    # Use num_workers=0 for clean exit on macOS
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # If the dataset folder name (or argument) contains "imagenette", restrict predictions to those 10 classes.
    if "imagenette" in args.dataset.lower():
        imagenette_indices = get_imagenette_indices()
    else:
        imagenette_indices = None

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            if imagenette_indices is not None:
                # Restrict outputs to the 10 imagenette class columns.
                restricted_logits = outputs[:, imagenette_indices]
                # Take argmax over these restricted logits.
                _, preds = torch.max(restricted_logits, 1)
                # Now preds are in the 0...9 range, matching ImageFolder ordering (which sorts the folder names).
            else:
                _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"[RESULT] Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()