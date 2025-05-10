import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from tqdm import tqdm
from data import prepare_data_loaders
from pathlib import Path
import json
import os

from arch import VisionMamba
from config import Config

def load_model():
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

    ckpt_path = Path("results/unlearned/recent.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)

    device = Config.DEVICE
    model.to(device).eval()
    return model, device

@torch.no_grad()
def validate(model, device, dataloader):
    total = 0
    correct = 0
    class_correct = {}
    class_total = {}

    target_label = 60
    misclassified_log = []

    pbar = tqdm(dataloader, desc="🔍 Validating", dynamic_ncols=True)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = preds[i].item()
            conf = confs[i].item()

            if label not in class_total:
                class_total[label] = 0
                class_correct[label] = 0
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

            # 🔎 Log if ground truth is 60
            if label == target_label:
                misclassified_log.append(f"True: {label}, Pred: {pred}, Confidence: {conf:.4f}")

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        acc = 100.0 * correct / total
        pbar.set_postfix(acc=f"{acc:.2f}%")

    print(f"\n✅ Final Accuracy: {acc:.2f}%")

    results = {
        "overall_accuracy": acc,
        "class_accuracy": {
            str(cls): 100.0 * class_correct[cls] / class_total[cls]
            for cls in class_total
        }
    }

    # 📁 Save accuracy results
    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)
    out_path = os.path.join(Config.FORGET.OUT_DIR, "validation_scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📁 Saved results to {out_path}")

    # 📝 Save predictions where label == 60
    log_path = os.path.join(Config.FORGET.OUT_DIR, "label_60_predictions.txt")
    with open(log_path, "w") as f:
        for entry in misclassified_log:
            f.write(entry + "\n")
    print(f"📝 Logged predictions for label 60 to {log_path}")

def main():
    model, device = load_model()
    data = prepare_data_loaders()
    val_loader = data["finetune"]["val"]
    validate(model, device, val_loader)

if __name__ == "__main__":
    main()
