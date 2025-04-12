import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class Pruner:
    def __init__(self, model, compression_ratio=0.5, device="cpu", mode="l1_unstructured"):
        self.model = model.to(device)
        self.amount = compression_ratio
        self.device = device
        self.mode = mode  # 'l1_unstructured', 'ln_structured', 'random_structured'
        self._pruned_modules = []

    def prune(self):
        print(f"[TorchBuiltin] Applying '{self.mode}' pruning with amount={self.amount}")

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                print(f"  → Pruning {name} ({type(module).__name__})")

                if self.mode == "l1_unstructured":
                    prune.l1_unstructured(module, name="weight", amount=self.amount)

                elif self.mode == "ln_structured":
                    prune.ln_structured(module, name="weight", amount=self.amount, n=2, dim=0)

                elif self.mode == "random_structured":
                    prune.random_structured(module, name="weight", amount=self.amount, dim=0)

                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                self._pruned_modules.append(module)

        return self.model

    def finalize(self):
        for module in self._pruned_modules:
            prune.remove(module, "weight")
        print("[TorchBuiltin] Pruning masks removed — weights are now permanently pruned.")