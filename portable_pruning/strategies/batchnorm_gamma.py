import torch
import torch.nn as nn

def prune_model(model, compression=0.5):
    print("[BN] Pruning using BatchNorm gamma values...")

    # This placeholder demonstrates selecting top channels by γ (BN weight)
    # NOTE: This doesn't actually zero weights – it's a mock structure to log pruning decisions

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            gamma = module.weight.data.abs()
            num_channels = gamma.size(0)

            keep = int((1 - compression) * num_channels)
            threshold = torch.topk(gamma, keep, largest=True).values.min()
            mask = (gamma >= threshold).float()

            print(f"[BN] {name}: keeping {keep}/{num_channels} channels")

            # Here we only print what we would keep. You can connect this to Conv2d mask pruning logic.

    return model