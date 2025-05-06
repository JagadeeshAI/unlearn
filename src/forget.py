# src/forget/forget.py

import os
import logging
from tqdm import tqdm
from itertools import cycle

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from huggingface_hub import snapshot_download

from config import Config
from src.data import prepare_data_loaders
from src.vim import VimTinyClassifier
from VimOffical.vim.models_mamba import VisionMamba

def retention_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def forgetting_loss(logits, labels):
    ce = F.cross_entropy(logits, labels)
    return F.relu(Config.FORGET.BND - ce)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(pixel_values=images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def load_model():
    VIM_REPO = "hustvl/Vim-small-midclstok"
    pretrained_dir = snapshot_download(
        repo_id=VIM_REPO,
        local_files_only=True,
        resume_download=True,
    )

    ckpt_path = PurePath(pretrained_dir, "vim_s_midclstok_ft_81p6acc.pth")

    model = VisionMamba(
        patch_size=16,
        stride=8,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224,
    )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, device

def train_forget():
    print(f"Device: {Config.DEVICE}")
    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)

    print("🧠 Initializing Vim-Tiny classifier...")
    model = VimTinyClassifier(num_classes=Config.FORGET.NUM_LABELS)
    model.load_pretrained()
    model.to(Config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=Config.FORGET.LR, weight_decay=Config.FORGET.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.FORGET.EPOCHS)

    data = prepare_data_loaders(Config.DATA_DIR, image_size=(224, 224), num_workers=4)
    loader_r = data['forgetting']['train']['retain']
    loader_f = data['forgetting']['train']['forget']
    val_r    = data['forgetting']['val']['retain']
    val_f    = data['forgetting']['val']['forget']

    print("🚀 Starting forgetting fine-tuning…")
    for epoch in range(1, Config.FORGET.EPOCHS + 1):
        model.train()
        sum_ret = sum_for = steps = 0
        forget_cycle = cycle(loader_f)

        loop = tqdm(loader_r, total=len(loader_r), desc=f"Epoch {epoch}/{Config.FORGET.EPOCHS}", unit="batch")

        for xr, yr in loop:
            xf, yf = next(forget_cycle)
            xr, yr = xr.to(Config.DEVICE), yr.to(Config.DEVICE)
            xf, yf = xf.to(Config.DEVICE), yf.to(Config.DEVICE)

            logits_r = model(pixel_values=xr)
            loss_r = retention_loss(logits_r, yr)

            logits_f = model(pixel_values=xf)
            loss_f = forgetting_loss(logits_f, yf)

            loss = loss_r + Config.FORGET.BETA * loss_f

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sum_ret += loss_r.item()
            sum_for += loss_f.item()
            steps += 1

            loop.set_postfix({
                "Ret": f"{loss_r:.4f}",
                "For": f"{loss_f:.4f}"
            })

        scheduler.step()
        print(f"📊 Epoch {epoch:2d} — Avg Ret={sum_ret/steps:.4f}  Avg For={sum_for/steps:.4f}")

    print("✅ Forgetting complete!")

    acc_f = evaluate(model, val_f, Config.DEVICE)
    acc_r = evaluate(model, val_r, Config.DEVICE)
    print(f"\n🎯 Forgotten Acc: {acc_f:.2f}%   🎯 Retained Acc: {acc_r:.2f}%")

    final_path = Config.FORGET.model_path()
    torch.save(model.state_dict(), final_path)
    print(f"💾 Model saved to {final_path}")

if __name__ == "__main__":
    train_forget()


