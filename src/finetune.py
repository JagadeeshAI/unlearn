# src/forget/forget.py

import os
import logging
from tqdm import tqdm
from itertools import cycle

# suppress TF/HF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

from config import Config
from src.model.data import prepare_data_loaders


def retention_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def forgetting_loss(logits, labels):
    ce = F.cross_entropy(logits, labels)
    return F.relu(Config.FORGET.BND - ce)


def group_sparsity_loss(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            loss = loss + param.norm()
    return loss


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def train_forget():
    # 1) Load Stage-1 fine-tuned ViT
    ckpt = Config.FINETUNE.model_path()
    if not os.path.exists(ckpt):
        print("❌ Run Stage 1 finetuning first.")
        return

    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print("🧠 Loading ViT + injecting LoRA adapters…")
    base = ViTForImageClassification.from_pretrained(
        Config.FINETUNE.VIT_MODEL,
        num_labels=Config.FINETUNE.NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    base.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

    # 2) Attach LoRA
    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=Config.FORGET.LORA_RANK,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=["intermediate.dense", "output.dense"]
    )
    model = get_peft_model(base, peft_cfg)

    # 3) Freeze LoRA only
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    adapters  = [p for p in model.parameters() if p.requires_grad]
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in adapters)
    print(f"Trainable params: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    model.to(Config.DEVICE)

    # 4) Data loaders
    data     = prepare_data_loaders(Config.DATA_DIR, image_size=(224,224), num_workers=4)
    loader_r = data['forgetting']['train']['retain']
    loader_f = data['forgetting']['train']['forget']
    val_r    = data['forgetting']['val']['retain']
    val_f    = data['forgetting']['val']['forget']

    # 5) Optimizer & scheduler
    optimizer = AdamW(adapters, lr=Config.FORGET.LR, weight_decay=Config.FORGET.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.FORGET.EPOCHS)

    # 6) GS-LoRA forgetting loop
    print("🚀 Starting GS-LoRA forgetting…")
    for epoch in range(1, Config.FORGET.EPOCHS + 1):
        model.train()
        sum_ret = sum_for = sum_struc = steps = 0
        alpha = 0.0 if epoch <= Config.FORGET.WARMUP_EPOCHS else Config.FORGET.ALPHA

        forget_cycle = cycle(loader_f)
        loop = tqdm(loader_r, total=len(loader_r),
                    desc=f"Epoch {epoch}/{Config.FORGET.EPOCHS}", unit="batch")

        for xr, yr in loop:
            xf, yf = next(forget_cycle)
            xr, yr = xr.to(Config.DEVICE), yr.to(Config.DEVICE)
            xf, yf = xf.to(Config.DEVICE), yf.to(Config.DEVICE)

            out_r = model(pixel_values=xr).logits
            loss_r = retention_loss(out_r, yr)

            out_f = model(pixel_values=xf).logits
            loss_f = forgetting_loss(out_f, yf)

            loss_s = group_sparsity_loss(model)
            loss   = loss_r + Config.FORGET.BETA * loss_f + alpha * loss_s

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sum_ret   += loss_r.item()
            sum_for   += loss_f.item()
            sum_struc += loss_s.item()
            steps     += 1

            loop.set_postfix({
                "Ret":   f"{loss_r:.4f}",
                "For":   f"{loss_f:.4f}",
                "Struc": f"{loss_s:.4f}"
            })

        scheduler.step()

        # ——— SAVE ONE .pth PER EPOCH ———
        epoch_path = os.path.join(Config.FORGET.OUT_DIR, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"💾 Saved epoch {epoch} state_dict to {epoch_path}")

        print(f"📊 Epoch {epoch} — Avg Ret={sum_ret/steps:.4f}  Avg For={sum_for/steps:.4f}  Avg Struc={sum_struc:.2f}")

    print("✅ GS-LoRA forgetting complete!")

    # 7) Final evaluation
    acc_f = evaluate(model, val_f, Config.DEVICE)
    acc_r = evaluate(model, val_r, Config.DEVICE)
    print(f"\n🎯 Forgotten Acc: {acc_f:.2f}%   🎯 Retained Acc: {acc_r:.2f}%")

    # 8) Save the **one** final .pth
    final_path = Config.FORGET.model_path()
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final state_dict saved to {final_path}")

if __name__ == "__main__":
    train_forget()
