import os
import logging

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformers import ViTForImageClassification
from config import Config
from src.finetune.data import prepare_data_loaders

# Hyperparameters (now properly defined globally)
BND = 1.0   
beta = 0.15
num_epochs = 50

def retention_loss(logits, labels):
    """Cross-entropy loss for retained classes."""
    return F.cross_entropy(logits, labels)

def forgetting_loss(logits, labels):
    """Forgetting loss to make model uncertain about forgotten class."""
    ce = F.cross_entropy(logits, labels)
    return F.relu(BND - ce)

def evaluate(model, loader, device):
    """Simple evaluation: returns accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits  # ✅ fix here
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def train_forget():
    """Main forgetting training loop."""
    if not os.path.exists(Config.MODEL_PATH):
        print("❌ Error: Pretrained model not found. Run finetuning first!")
        return

    print("🧠 Loading pre-trained ViT model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=37,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    # Ensure all parameters are trainable
    for p in model.parameters():
        p.requires_grad = True

    device = Config.DEVICE
    model = model.to(device)

    # Load datasets
    data = prepare_data_loaders(
        data_dir=Config.DATA_DIR,
        image_size=(224, 224),
        batch_size=Config.BATCH_SIZE,
        num_workers=4
    )

    forget_train_loader = data['forgetting']['train']['forget']
    retain_train_loader = data['forgetting']['train']['retain']
    forget_val_loader   = data['forgetting']['val']['forget']
    retain_val_loader   = data['forgetting']['val']['retain']

    loader_f = forget_train_loader
    loader_r = retain_train_loader

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("🚀 Starting forgetting training...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        sum_ret, sum_for = 0.0, 0.0
        count = 0

        loop = tqdm(zip(loader_r, loader_f), total=min(len(loader_r), len(loader_f)), desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        
        for (xr, yr), (xf, yf) in loop:
            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)

            # Retention step
            outputs_r = model(xr)
            logits_r = outputs_r.logits  # ✅ fix here
            loss_r = retention_loss(logits_r, yr)

            # Forgetting step
            outputs_f = model(xf)
            logits_f = outputs_f.logits  # ✅ fix here
            loss_f = forgetting_loss(logits_f, yf)

            # Combine
            loss = loss_r + beta * loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_ret += loss_r.item()
            sum_for += loss_f.item()
            count += 1

            loop.set_postfix({
                "Ret_Loss": loss_r.item(),
                "Forget_Loss": loss_f.item(),
                "Total_Loss": loss.item()
            })

        scheduler.step()

        print(f"📊 Epoch {epoch:2d}  Ret_Loss={sum_ret/count:.4f}  Forget_Loss={sum_for/count:.4f}")

        # Save model after each epoch
        save_path = os.path.join(Config.MODEL_DIR, f"{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model checkpoint saved to: {save_path}")


    print("✅ Forgetting training complete!")

    # Final Evaluation
    forget_acc = evaluate(model, forget_val_loader, device)
    retain_acc = evaluate(model, retain_val_loader, device)

    print(f"\n🎯 Forget Class Validation Accuracy: {forget_acc:.2f}% (should be low)")
    print(f"🎯 Retain Classes Validation Accuracy: {retain_acc:.2f}% (should stay high)")

    # Save the final model if you want
    save_path = os.path.join(Config.MODEL_DIR, "forget_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Forgetting model saved to: {save_path}")

if __name__ == "__main__":
    train_forget()
