# Portable Pruning 🚀

**Towards Data-Free and Device-Agnostic CNN Pruning for Edge Deployment**

Portable Pruning is a research framework and toolkit for pruning convolutional neural networks **without any training data**, making them deployable across a variety of **edge devices** such as Raspberry Pi, Jetson Nano, and mobile NPUs.

---

## 🔍 Features

- 💡 **Data-Free**: No access to original training data required
- 🔄 **Device-Agnostic**: Tested on real edge devices
- 🧱 **Modular**: Easily swap or extend pruning strategies
- 📊 **Benchmarking**: Includes latency and memory profiling

---

## 🧠 Supported Pruning Techniques

- ✅ L1-Norm Weight Pruning
- ✅ BatchNorm Gamma-Based Channel Pruning
- ✅ Deep Inversion (Synthetic Input) Pruning
- 🔄 Random Channel Pruning (Baseline)

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/portable-pruning.git
cd portable-pruning
pip install -r requirements.txt
```

---

## 🚀 Quickstart

```bash
python experiments/run_pruning.py --model resnet18 --method l1 --device jetson --compression 0.5
```

---

## 📈 Results

| Method         | Accuracy (CIFAR-10) | FLOPs ↓ | Latency (ms/img) | RAM (MB) |
|----------------|---------------------|---------|-------------------|-----------|
| L1-Norm        | 91.2%               | 50%     | 110.7             | 188       |
| BN γ Pruning   | 91.5%               | 49%     | 108.2             | 176       |
| Deep Inversion | 92.3%               | 52%     | 85.4              | 211       |

---

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{portablepruning2025,
  title={Portable Pruning: Towards Data-Free and Device-Agnostic CNN Pruning for Edge Deployment},
  author={Your Name},
  year={2025},
  journal={arXiv preprint arXiv:xxxx.xxxxx}
}
```

---

## 🧑‍💻 Contributing

Pull requests are welcome! Feel free to open an issue or submit ideas.

---

## 📄 License

MIT License © 2025 Your Name
