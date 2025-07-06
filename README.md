
# Portable Pruning

**Data-Free. Device-Agnostic. Simple CNN Pruning for Edge Devices.**

This is a modular and clean framework to help you experiment with pruning deep learning models like `ResNet18`, `ResNet34`, and `MobileNetV2`. You don’t need any training data for pruning. It’s built for people deploying compressed models on edge devices like Raspberry Pi, Jetson Nano, or Android phones.

---

## What it Can Do

- Works without any training data  
- Device-agnostic — models run on various edge hardware  
- Easy-to-plug pruning strategies  
- Measures: FLOPs, Params, Latency, RAM, Accuracy  
- Exports `.pth` and `.onnx` for deployment  
- Logs every experiment in a CSV automatically  

---

## Installation & Setup

### Step 1: Clone the Repo

```bash
git clone https://github.com/yourusername/portable-pruning.git
cd portable-pruning
````

### Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

If you have issues importing the `portable_pruning` module, add it to your path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Step 3: Dataset

By default, the code uses `imagenette2-320`. You don’t need to train anything — just prune and evaluate.

---

## Models Available

* `resnet18` — lightweight and fast
* `resnet34` — deeper but still efficient
* `mobilenet_v2` — made for mobile deployment

All are from `torchvision.models` and come pretrained.

---

## Pruning Methods

| Method      | Description                                |
| ----------- | ------------------------------------------ |
| `l1`        | Unstructured pruning (L1 norm)             |
| `batchnorm` | Based on BatchNorm gamma values            |
| `red`       | Redundancy-based with KDE (`alpha`, `tau`) |
| `redpp`     | Lighter version of RED                     |
| `builtin`   | PyTorch’s pruning API                      |
| `autodfp`   | RL-based (Auto Deep Filter Pruning)        |

---

## Pruning Modes (For `builtin` Method Only)

Use these modes with the `builtin` method:

* `l1_unstructured` – Removes individual weights with the lowest L1 norm
* `ln_structured` – Drops entire filters or channels using structured norms
* `random_structured` – Randomly removes full structures (like channels)
* `random_unstructured` – Randomly removes individual weights

Choose structured modes for speed-up and unstructured for compression.

---

## Run It

### Prune a Model

```bash
python run_pruning.py \
  --model resnet18 \
  --method builtin \
  --mode l1_unstructured \
  --compression 0.5 \
  --onnx \
  --output_dir .outputs/my_run
```

### Just Evaluate Baseline (No Pruning)

```bash
python run_pruning.py \
  --model resnet18 \
  --method builtin \
  --mode l1_unstructured \
  --compression 0.5 \
  --baseline \
  --onnx \
  --output_dir .outputs/my_run
```

---

## Run a Full Experiment Grid

```bash
python experiment.py --subset_size 100 --results_file results.csv
```

This will go through all models, compressions, and modes in one shot.

Results saved to:

```
.outputs/run_<timestamp>/
results.csv
```

---

## License

MIT License © 2025 Purnendu Prabhat

```

---

Just copy and paste this into your `README.md`. Let me know if you want a version that includes badges (like PyTorch version, build status, etc.) or a table of contents for navigation.
```
