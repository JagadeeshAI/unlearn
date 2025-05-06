
import sys
from pathlib import PurePath

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from VimOffical.vim.models_mamba import VisionMamba


def load_model() -> torch.nn.Module:
    """Download the checkpoint (if needed), build the model, load weights."""
    VIM_REPO = "hustvl/Vim-small-midclstok"

    # download once, then reuse the local cache next time
    pretrained_dir = snapshot_download(
        repo_id=VIM_REPO,
        local_files_only=True,  
        resume_download=True,
    )

    ckpt_path = PurePath(pretrained_dir, "vim_s_midclstok_ft_81p6acc.pth")

    # Model hyper‑params must match those in the checkpoint
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


def preprocess(image_path: str, device: str) -> torch.Tensor:
    """Load and normalize an image to 224×224."""
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    tensor = transforms.ToTensor()(image)
    tensor = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(tensor)
    return tensor.unsqueeze(0).to(device)


def main(image_path: str):
    model, device = load_model()
    x = preprocess(image_path, device)

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax())

    print(f"Input image: {image_path}")
    print(f"Predicted class index: {pred_idx}")

if __name__ == "__main__":
    default_img = "/home/jag/codes/unlearn/data/val/n13052670/ILSVRC2012_val_00002575.jpg"
    img_path = sys.argv[1] if len(sys.argv) > 1 else default_img
    main(img_path)
