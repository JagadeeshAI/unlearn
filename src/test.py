import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from tqdm import tqdm
from data import prepare_data_loaders
from pathlib import Path, PurePath
import json
import os
from huggingface_hub import snapshot_download

from arch2 import VisionMamba
from config import Config


def print_trainable_lora_stats(model):
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
    VIM_REPO = "hustvl/Vim-small-midclstok"
    pretrained_dir = snapshot_download(
        repo_id=VIM_REPO,
        local_files_only=True,
        resume_download=True,
    )

    ckpt_path = PurePath(pretrained_dir, "vim_s_midclstok_ft_81p6acc.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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

    # Remap classifier weights for LoRA
    new_state_dict = model.state_dict()
    new_state_dict["head.base.weight"] = state_dict["head.weight"]
    new_state_dict["head.base.bias"] = state_dict["head.bias"]

    for k, v in state_dict.items():
        if k not in ["head.weight", "head.bias"]:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    print_trainable_lora_stats(model)

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

    # Load label names
    with open("data/imagenet_class_index.json") as f:
        idx_to_label = json.load(f)
        idx_to_name = {int(k): v[1] for k, v in idx_to_label.items()}

    # Sort top 10
    sorted_classes = sorted(
        results["class_accuracy"].items(),
        key=lambda item: item[1],
        reverse=True
    )

    print("\n🏆 Top 10 Classes by Accuracy:")
    print(f"{'Rank':<5} {'Label':<6} {'Name':<30} {'Accuracy (%)':>12}")
    print("-" * 60)
    for i, (cls_str, acc) in enumerate(sorted_classes[:10], 1):
        cls_id = int(cls_str)
        cls_name = idx_to_name.get(cls_id, "Unknown")
        print(f"{i:<5} {cls_id:<6} {cls_name:<30} {acc:>12.2f}")

    # Save results
    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)
    out_path = os.path.join(Config.FORGET.OUT_DIR, "validation_scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📁 Saved results to {out_path}")

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
