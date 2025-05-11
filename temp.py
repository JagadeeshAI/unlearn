import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from tqdm import tqdm
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms.functional import to_pil_image
import numpy as np
from collections import defaultdict

from huggingface_hub import snapshot_download
from pathlib import PurePath
from arch2 import VisionMamba
from config import Config


# === Load Vision Mamba as per your instructions ===
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

    checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
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

    # Freeze and unfreeze LoRA
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    if "head.base.weight" in state_dict:
        model.load_state_dict(state_dict, strict=True)
    elif "head.weight" in state_dict:
        new_state_dict = model.state_dict()
        new_state_dict["head.base.weight"] = state_dict["head.weight"]
        new_state_dict["head.base.bias"] = state_dict["head.bias"]
        for k, v in state_dict.items():
            if k not in ["head.weight", "head.bias"]:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        raise ValueError("Checkpoint format not recognized")

    print_trainable_lora_stats(model)
    model.to(Config.DEVICE).eval()
    return model, Config.DEVICE


# === Evaluation logic ===

def main():
    # 🔧 Settings
    truncation = 0.4
    n_per_class = 100
    selected_classes = {
        102: "golden_retriever",
        11: "flamingo",
        129: "king_penguin",
        13: "revolver",
        139: "golf_ball",
        148: "running_shoe",
        169: "screwdriver",
        225: "spatula",
        258: "toilet_tissue",
        275: "volcano"
    }

    # 🎨 Preprocessing
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        lambda x: torch.tensor(np.array(x)).permute(2, 0, 1).float() / 255,
        Normalize([0.5]*3, [0.5]*3)
    ])

    # 🧠 Load models
    model, device = load_model()
    biggan = BigGAN.from_pretrained('biggan-deep-512').to(device).eval()

    # 📊 Stats
    total_correct = 0
    total_samples = 0
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # 🔁 Loop
    with torch.no_grad():
        for class_id, class_name in tqdm(selected_classes.items(), desc="Evaluating Classes"):
            for _ in range(n_per_class):
                z = torch.from_numpy(truncated_noise_sample(batch_size=1, truncation=truncation)).to(device)
                c = torch.from_numpy(one_hot_from_int(class_id, 1)).to(device)
                img = biggan(z, c, truncation)
                img = (img + 1) / 2
                pil_img = to_pil_image(img.squeeze(0).cpu())
                x = transform(pil_img).unsqueeze(0).to(device)

                logits = model(x)
                pred = logits.argmax(dim=1).item()

                total_samples += 1
                class_stats[class_id]["total"] += 1
                if pred == class_id:
                    total_correct += 1
                    class_stats[class_id]["correct"] += 1

    # 📈 Results
    print(f"\n✅ Overall Accuracy: {100 * total_correct / total_samples:.2f}%")
    print(f"{'Class ID':<9} {'Class Name':<20} {'Accuracy (%)':>12}")
    print("-" * 45)
    for cid, stats in class_stats.items():
        acc = 100 * stats["correct"] / stats["total"]
        print(f"{cid:<9} {selected_classes[cid]:<20} {acc:>12.2f}")


if __name__ == "__main__":
    main()
