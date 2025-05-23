import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import torch
import logging
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path, PurePath

from arch2 import VisionMamba
from config import Config
from src.data import prepare_data_loaders
from huggingface_hub import snapshot_download

# Silence unnecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def print_lora_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)

    percent_trainable = 100 * trainable_params / total_params
    percent_lora = 100 * lora_params / total_params

    print(f"\n📊 Model Parameters Summary:")
    print(f"🔹 Total Parameters:     {total_params:,}")
    print(f"🔹 Trainable Parameters: {trainable_params:,} ({percent_trainable:.4f}%)")
    print(f"🔹 LoRA Parameters Only: {lora_params:,} ({percent_lora:.6f}%)")

def load_model():
    if Config.FORGET.RESUME:
        ckpt_path = "results/unlearned/recent.pth"
    else:  
        VIM_REPO = "hustvl/Vim-small-midclstok"
        pretrained_dir = snapshot_download(
            repo_id=VIM_REPO,
            local_files_only=True,
            resume_download=True,
        )
        ckpt_path = PurePath(pretrained_dir, "vim_s_midclstok_ft_81p6acc.pth")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

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
        lora_out_proj=True,
        lora_r=96,
        lora_alpha=0.1
    )

    # 🔒 Freeze all params
    for param in model.parameters():
        param.requires_grad = False
    # ✅ Enable only LoRA layers
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    # 🧠 Handle both LoRA and Hugging Face formats
    if "head.base.weight" in state_dict:
        # Already LoRA format
        model.load_state_dict(state_dict, strict=True)
    elif "head.weight" in state_dict:
        # Hugging Face format, needs remapping
        print("🔁 Remapping Hugging Face weights for LoRA structure...")
        new_state_dict = model.state_dict()
        new_state_dict["head.base.weight"] = state_dict["head.weight"]
        new_state_dict["head.base.bias"] = state_dict["head.bias"]
        for k, v in state_dict.items():
            if k not in ["head.weight", "head.bias"]:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        raise ValueError("❌ Unknown checkpoint structure")

    print_lora_stats(model)
    model.to(Config.DEVICE).eval()
    return model, Config.DEVICE

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
            outputs = model(images)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def train_forget():
    print(f"📟 Device: {Config.DEVICE}")
    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)

    print("🧠 Loading VisionMamba model...")
    model, device = load_model()
    print_lora_stats(model)

    data = prepare_data_loaders(Config.DATA_DIR, image_size=(224, 224), num_workers=4)
    loader_r = data['forgetting']['train']['retain']
    loader_f = data['forgetting']['train']['forget']
    val_r = data['forgetting']['val']['retain']
    val_f = data['forgetting']['val']['forget']

    print("🚀 Starting forgetting fine-tuning…")
    optimizer = AdamW(model.parameters(), lr=Config.FORGET.LR, weight_decay=Config.FORGET.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.FORGET.EPOCHS)

    global_step = 0
    for epoch in range(1, Config.FORGET.EPOCHS + 1):
        model.train()
        sum_ret = sum_for = steps = 0
        forget_cycle = cycle(loader_f)

        loop = tqdm(loader_r, total=len(loader_r), desc=f"Epoch {epoch}/{Config.FORGET.EPOCHS}", unit="batch")

        for xr, yr in loop:
            xf, yf = next(forget_cycle)
            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)

            logits_r = model(xr)
            loss_r = retention_loss(logits_r, yr)

            logits_f = model(xf)
            loss_f = forgetting_loss(logits_f, yf)

            loss = loss_r + Config.FORGET.BETA * loss_f

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sum_ret += loss_r.item()
            sum_for += loss_f.item()
            steps += 1
            global_step += 1

            loop.set_postfix({
                "Ret": f"{loss_r:.4f}",
                "For": f"{loss_f:.4f}"
            })

        scheduler.step()
        print(f"📊 Epoch {epoch:2d} — Avg Ret={sum_ret/steps:.4f}  Avg For={sum_for/steps:.4f}")

        # 🔍 Validation after epoch
        acc_f = evaluate(model, val_f, device)
        acc_r = evaluate(model, val_r, device)
        print(f"🧪 Epoch {epoch} Validation — Forgotten Accuracy: {acc_f:.2f}% | Retained Accuracy: {acc_r:.2f}%")

        # 💾 Save checkpoint
        checkpoint_path = os.path.join(Config.FORGET.OUT_DIR, f"recent.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved model checkpoint to {checkpoint_path}")

    print("✅ Forgetting complete!")

    acc_f = evaluate(model, val_f, device)
    acc_r = evaluate(model, val_r, device)
    print(f"\n🎯 Final Forgotten Accuracy: {acc_f:.2f}%")
    print(f"🎯 Final Retained Accuracy: {acc_r:.2f}%")

    final_path = Config.FORGET.model_path()
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final model saved to {final_path}")

    results = {
        "forgotten_accuracy": acc_f,
        "retained_accuracy": acc_r
    }
    result_path = os.path.join(Config.FORGET.OUT_DIR, "forgetting_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📁 Results saved to {result_path}")

if __name__ == "__main__":
    train_forget()
