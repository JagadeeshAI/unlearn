# test.py

import torch
from tqdm import tqdm
from data import prepare_data_loaders
from pathlib import PurePath
from PIL import Image
from huggingface_hub import snapshot_download
from VimOffical.vim.models_mamba import VisionMamba
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

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

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, device

@torch.no_grad()
def validate(model, device, dataloader):
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc="🔍 Validating", dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100.0 * correct / total
        pbar.set_postfix(acc=f"{acc:.2f}%")

    print(f"\n✅ Final Accuracy: {acc:.2f}%")

def main():
    model, device = load_model()
    data = prepare_data_loaders()
    val_loader = data["finetune"]["val"]

    validate(model, device, val_loader)

if __name__ == "__main__":
    main()
