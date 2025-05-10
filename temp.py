import torch
from pathlib import PurePath
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms
import torch.nn as nn

from arch import VisionMamba
from config import Config  # Your Config class

# Constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGE_PATH = "/home/jag/codes/unlearn/data/val/n04536866/ILSVRC2012_val_00036991.jpg"


def load_model():
    VIM_REPO = "hustvl/Vim-small-midclstok"
    pretrained_dir = snapshot_download(
        repo_id=VIM_REPO,
        local_files_only=True,
        resume_download=True,
    )

    ckpt_path = "results/unlearned/recent.pth"

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

    device = Config.DEVICE
    model.to(device).eval()
    return model, device


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_and_print(model, device, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top1_prob, top1_idx = torch.topk(probabilities, 1)
        print(f"🧠 Predicted class index: {top1_idx.item()}, confidence: {top1_prob.item():.4f}")


def main():
    model, device = load_model()
    image_tensor = preprocess_image(IMAGE_PATH)
    predict_and_print(model, device, image_tensor)


if __name__ == "__main__":
    main()
