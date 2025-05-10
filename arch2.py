from arch_with_hybrid_peft import VisionMamba
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=1.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.lora_down = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, base_layer.out_features, bias=False)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_up(self.lora_down(x))


# Patch VisionMamba to inject LoRA into `self.head` if requested
_original_init = VisionMamba.__init__

def _patched_init(self, *args, lora_out_proj=False, lora_r=8, lora_alpha=1.0, **kwargs):
    _original_init(self, *args, **kwargs)

    if lora_out_proj:
        print(f"✅ Injecting LoRA into classifier head with r={lora_r}, alpha={lora_alpha}")
        self.head = LoRALinear(self.head, r=lora_r, alpha=lora_alpha)

VisionMamba.__init__ = _patched_init
