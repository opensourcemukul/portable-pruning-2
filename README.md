# 🔧 Portable Pruning

**Data-Free. Device-Agnostic. Simple CNN Pruning for Edge Devices.**

This is a modular and clean framework to help you experiment with pruning deep learning models like `ResNet18`, `ResNet34`, and `MobileNetV2`. The best part — you don’t need any training data for pruning. It’s built for people who want to deploy compressed models on edge devices like Raspberry Pi, Jetson Nano, or Android phones.

---

## 🔍 What it Can Do

- ✅ Works without any training data
- 🧱 Device-agnostic — models run on various edge hardware
- 🔄 Easy-to-plug pruning strategies
- 📏 Measures: FLOPs, Params, Latency, RAM, Accuracy
- 📤 Exports `.pth` and `.onnx` for deployment
- 📊 Logs every experiment in a CSV automatically

---

## ⚙️ Installation & Setup

### 🔹 Step 1: Clone the Repo

```bash
git clone https://github.com/yourusername/portable-pruning.git
cd portable-pruning
```

### 🔹 Step 2: Install Requirements

> This will install PyTorch, ONNX, and any helper libraries.

```bash
pip install -r requirements.txt
```

> If `portable_pruning/` is not being found as a module, you can either install it or add it to your PYTHONPATH.

```bash
# Optional
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 🔹 Step 3: Dataset

By default, it uses the `imagenette2-320` dataset. You can change this in the script if needed. Make sure the dataset is downloaded or accessible.

> No need to train anything — just prune and evaluate.

---

## 🧠 Models Available

- `resnet18` — lightweight and fast
- `resnet34` — deeper but still efficient
- `mobilenet_v2` — made for mobile deployment

All models load from `torchvision.models` with pretrained weights.

---

## ✂️ Pruning Methods

| Method      | Description |
|-------------|-------------|
| `l1`        | Unstructured weight pruning based on L1 norm |
| `batchnorm` | Prunes based on BatchNorm gamma values |
| `red`       | KDE-based redundancy filtering (tune `alpha`, `tau`) |
| `redpp`     | A simplified variant of RED |
| `builtin`   | PyTorch’s own `nn.utils.prune` interface |
| `autodfp`   | Auto Deep Filter Pruning (RL-based) |

Modes like `l1_unstructured`, `ln_structured`, etc. are available under the `builtin` method.

---

## 🚀 Run It (Your Way)

### 🔸 Prune a Model

```bash
python run_pruning.py   --model resnet18   --method builtin   --mode l1_unstructured   --compression 0.5   --onnx   --output_dir .outputs/my_run
```

### 🔸 Just Evaluate Baseline (No Pruning)

```bash
python run_pruning.py   --model resnet18   --method builtin   --mode l1_unstructured   --compression 0.5   --baseline   --onnx   --output_dir .outputs/my_run
```

You’ll get `.pth`, `.onnx`, and a bunch of console logs like FLOPs, latency, RAM, etc.

---

## 🧪 Run a Full Experiment Grid

Run everything in one shot (all models, compressions, etc.).

```bash
python experiment.py --subset_size 100 --results_file results.csv
```

> This is useful when you want a proper benchmark with all configurations.

Outputs go to:

- `.outputs/run_<timestamp>/`
- `results.csv`

---

## 📄 License

MIT License © 2025 **Purnendu Prabhat**

---

If you're looking to do quick, clean, and practical CNN pruning for deployment — this repo is a good starting point.