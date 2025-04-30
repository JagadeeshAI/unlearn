# src/model/test.py

import os
import json
import logging
from tqdm import tqdm

# suppress TF/HF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from transformers import SegformerForSemanticSegmentation
from config import Config
from src.model.data import prepare_segmentation_loader

def visualize_predictions(images, preds, masks, save_dir="/media/jagadeesh/volD/unlearn/vis", num_samples=5):

    os.makedirs(save_dir, exist_ok=True)

    for idx in range(min(num_samples, images.size(0))):
        img = images[idx].cpu()
        mask = masks[idx].cpu()
        pred = preds[idx].cpu()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        axs[1].imshow(mask)
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')

        axs[2].imshow(pred)
        axs[2].set_title("Prediction")
        axs[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{idx}.png")
        plt.savefig(save_path)
        print(f"✅ Saved: {save_path}")  # <-- Add this line
        plt.close()


def evaluate_model():
    # 1) Load Pretrained SegFormer
    print("🔍 Loading SegFormer architecture...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        Config.FINETUNE.SEGFORMER_MODEL,
        num_labels=150,
        ignore_mismatched_sizes=True,
    )
    
    checkpoint_path = "results/learned/best_segformer_lora.pth"
    
    state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded weights from {checkpoint_path}")

    model.to(Config.DEVICE).eval()
    print("✅ Model ready for evaluation.")

    # 2) Prepare val data
    val_loader = prepare_segmentation_loader(
        data_dir=Config.DATA_DIR,
        split="val",
        batch_size=Config.FINETUNE.BATCH_SIZE,
        num_workers=2,
    )

    # 3) Run evaluation
    total_pixels = 0
    correct_pixels = 0

    iou_sum = 0
    iou_count = 0

    preds_list = []
    masks_list = []
    images_list = []

    print("🚀 Evaluating on validation set…")
    with torch.no_grad():
        for images, masks, _, _ in tqdm(val_loader, desc="Validating", unit="batch"):
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
            outputs = model(pixel_values=images)

            if hasattr(outputs, 'logits'):
                preds = outputs.logits
            else:
                preds = outputs

            preds = torch.nn.functional.interpolate(
                preds,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            preds_classes = preds.argmax(dim=1)

            correct_pixels += (preds_classes == masks).sum().item()
            total_pixels += masks.numel()

            # IoU calculation
            for cls in range(Config.FINETUNE.NUM_LABELS):
                pred_inds = (preds_classes == cls)
                target_inds = (masks == cls)
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()

                if union > 0:
                    iou_sum += intersection / union
                    iou_count += 1

            if len(images_list) < 10:
                images_list.append(images.cpu())
                masks_list.append(masks.cpu())
                preds_list.append(preds_classes.cpu())

    overall_pixel_acc = 100 * correct_pixels / total_pixels
    mean_iou = 100 * iou_sum / iou_count if iou_count > 0 else 0

    print(f"\n✅ Pixel Accuracy: {overall_pixel_acc:.2f}%")
    print(f"✅ Mean IoU: {mean_iou:.2f}%")

    results = {
        "pixel_accuracy": overall_pixel_acc,
        "mean_iou": mean_iou,
    }

    # 4) Save metrics
    os.makedirs(Config.FINETUNE.OUT_DIR, exist_ok=True)
    out_path = os.path.join(Config.FINETUNE.OUT_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n📝 Results saved to {out_path}")

    # 5) Visualize a few predictions
    print("🖼️ Visualizing predictions...")
    visualize_predictions(
        images=torch.cat(images_list, dim=0),
        preds=torch.cat(preds_list, dim=0),
        masks=torch.cat(masks_list, dim=0),
        save_dir=os.path.join(Config.FINETUNE.OUT_DIR, "visualizations"),
        num_samples=5
    )
    print(f"✅ Visualization samples saved.")

if __name__ == "__main__":
    evaluate_model()
