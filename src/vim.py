# src/model/vim.py

import torch
import torch.nn as nn
import sys
import os
import argparse
from huggingface_hub import hf_hub_download

# Add the Vim repo to the path
VIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Vim"))
if VIM_ROOT not in sys.path:
    sys.path.insert(0, VIM_ROOT)

# Import the correct Vim-Tiny architecture
from VimOffical.vim.models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

class VimTinyClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self):
        print("📦 Downloading Vim-Tiny pretrained weights from Hugging Face...")
        weight_path = hf_hub_download("hustvl/Vim-tiny-midclstok", "vim_t_midclstok_76p1acc.pth")

        # Safe loading for PyTorch 2.6+ with argparse.Namespace
        with torch.serialization.safe_globals([argparse.Namespace]):
            state_dict = torch.load(weight_path, map_location="cpu")

        incompatible = self.load_state_dict(state_dict.get("model", state_dict), strict=False)
        print("✅ Pretrained weights loaded.")
        
