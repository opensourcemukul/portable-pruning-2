import torch
import torch.nn as nn
import copy
from portable_pruning.utils.rl_utils import ChannelSimilarity, RLController, compute_reward

class Pruner:
    def __init__(self, model, compression_ratio=0.5, device='cpu', max_iters=20, reward_temperature=1.0):
        self.model = model.to(device)
        self.device = device
        self.controller = RLController()
        self.similarity_computer = ChannelSimilarity()
        self.max_iters = max_iters
        self.reward_temperature = reward_temperature
        self.compression_ratio = compression_ratio

    def generate_synthetic_input(self, shape=(1, 3, 224, 224)):
        return torch.randn(*shape).to(self.device)

    def _find_parent(self, root, target_module):
        for name, module in root.named_modules():
            for child_name, child in module.named_children():
                if child is target_module:
                    return module
        return None

    def _inject_downsample(self, model, layer_name, out_channels):
        """
        Adds a 1x1 conv downsample if identity and output don't match in residual blocks.
        """
        parent = dict(model.named_modules())[layer_name.rsplit('.', 1)[0]]
        block = parent
        block.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=block.conv1.in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        print(f"[AutoDFP] Injecting downsample in {layer_name} to fix residual mismatch.")

    def prune(self):
        print("[AutoDFP] Starting data-free pruning via RL...")

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Conv2d) and module.weight.requires_grad:
                if name == "conv1":
                    print(f"  [SKIP] Skipping {name} (input layer)")
                    continue

                print(f"  → Pruning Layer: {name}")

                # Step 1: Compute similarity matrix
                sim_matrix = self.similarity_computer(module.weight.data)

                # Step 2: Use RL to select pruning mask
                best_mask, best_reward = None, -float('inf')
                for _ in range(self.max_iters):
                    mask = self.controller.sample_mask(sim_matrix, compression_ratio=self.compression_ratio)

                    test_model = copy.deepcopy(self.model)
                    reward = compute_reward(
                        model=test_model,
                        layer_name=name,
                        mask=mask,
                        input_tensor=self.generate_synthetic_input(),
                        reward_temperature=self.reward_temperature
                    )

                    if reward > best_reward:
                        best_reward = reward
                        best_mask = mask

                # Step 3: Apply best mask to the real model
                self.apply_mask(self.model, name, module, best_mask)

        return self.model
    # def fix_next_conv_input_channels(self, model, layer_name, keep_idxs):
    #     """
    #     Fix the input channels of the next convolution layer after pruning output channels of the current layer.
    #     """
    #     found = False
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Conv2d) and name.startswith(layer_name.rsplit('.', 1)[0]):
    #             # Skip the current layer
    #             continue

    #         if isinstance(module, nn.Conv2d):
    #             weight_shape = module.weight.data.shape  # [out_channels, in_channels, kH, kW]

    #             # Sanity check before slicing
    #             if weight_shape[1] < keep_idxs.max().item() + 1:
    #                 print(f"[AutoDFP] ⚠️ Skipping fix for {name} — keep_idxs exceed in_channels ({weight_shape[1]})")
    #                 continue

    #             try:
    #                 module.weight.data = module.weight.data[:, keep_idxs]
    #                 module.in_channels = len(keep_idxs)
    #                 print(f"[AutoDFP] → Fixing input channels of: {name}")
    #                 found = True
    #                 break
    #             except IndexError as e:
    #                 print(f"[AutoDFP] ⚠️ IndexError while fixing {name}: {e}")
    #                 continue

    #     if not found:
    #         print(f"[AutoDFP] ⚠️ No next Conv2d layer found to fix after {layer_name}")
    def fix_next_conv_input_channels(self, model, layer_name, keep_idxs):
        """
        Fix the input channels of the next convolution layer after pruning output channels of the current layer.
        """
        found = False
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue

            # ✅ Skip only the exact same layer
            if name == layer_name:
                continue

            weight_shape = module.weight.data.shape  # [out_channels, in_channels, kH, kW]

            if weight_shape[1] < keep_idxs.max().item() + 1:
                print(f"[AutoDFP] ⚠️ Skipping fix for {name} — keep_idxs exceed in_channels ({weight_shape[1]})")
                continue

            try:
                module.weight.data = module.weight.data[:, keep_idxs]
                module.in_channels = len(keep_idxs)
                print(f"[AutoDFP] → Fixing input channels of: {name}")
                found = True
                break
            except IndexError as e:
                print(f"[AutoDFP] ⚠️ IndexError while fixing {name}: {e}")
                continue

        if not found:
            print(f"[AutoDFP] ⚠️ No next Conv2d layer found to fix after {layer_name}")

    def apply_mask(self, model, layer_name, conv_layer, mask):
        keep_idxs = torch.nonzero(mask).squeeze(1)

        # Prune Conv2D output channels
        conv_layer.weight.data = conv_layer.weight.data[keep_idxs]
        if conv_layer.bias is not None:
            conv_layer.bias.data = conv_layer.bias.data[keep_idxs]
        conv_layer.out_channels = len(keep_idxs)

        # Prune BatchNorm if present next
        parent = self._find_parent(model, conv_layer)
        if parent:
            modules = list(parent.children())
            for i, m in enumerate(modules):
                if m is conv_layer:
                    if i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm2d):
                        bn = modules[i + 1]
                        bn.weight.data = bn.weight.data[keep_idxs]
                        bn.bias.data = bn.bias.data[keep_idxs]
                        bn.running_mean = bn.running_mean[keep_idxs]
                        bn.running_var = bn.running_var[keep_idxs]
                        bn.num_features = len(keep_idxs)

        # Special fix: if this is conv1 inside a ResNet BasicBlock, fix conv2 input
        if "layer" in layer_name and ".0.conv1" in layer_name:
            block_name = layer_name.rsplit(".", 1)[0]
            block = dict(model.named_modules())[block_name]
            next_conv = getattr(block, "conv2", None)
            if isinstance(next_conv, nn.Conv2d):
                if next_conv.weight.shape[1] >= keep_idxs.max().item() + 1:
                    next_conv.weight.data = next_conv.weight.data[:, keep_idxs]
                    next_conv.in_channels = len(keep_idxs)
                else:
                    print(f"[AutoDFP] ⚠️ Skipping fix for {layer_name} — keep_idxs exceed in_channels ({next_conv.weight.shape[1]})")
                # next_conv.weight.data = next_conv.weight.data[:, keep_idxs]
                # next_conv.in_channels = len(keep_idxs)

        # Residual mismatch fix (for first blocks)
        if "layer" in layer_name and ".0.conv1" in layer_name:
            identity_channels = conv_layer.in_channels
            out_channels = conv_layer.out_channels
            if identity_channels != out_channels:
                self._inject_downsample(model, layer_name, out_channels)
        
        # ✅ Fix next Conv2d’s input channels *anywhere* in model
        self.fix_next_conv_input_channels(model, layer_name, keep_idxs)