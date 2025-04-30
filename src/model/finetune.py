# train.py

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from config import Config
from src.model.data import prepare_segmentation_loader

def compute_metrics(preds, masks, num_classes):
    preds = preds.view(-1)
    masks = masks.view(-1)

    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0

def inject_lora(model, rank=8):
    """
    Adds LoRA adapters to FFN (MLP) layers inside SegFormer Transformer blocks.
    """
    for name, module in model.named_modules():
        if "encoder.block" in name and hasattr(module, "intermediate"):
            # Inject LoRA into intermediate dense (FFN first layer)
            if hasattr(module.intermediate, "dense"):
                dense_layer = module.intermediate.dense

                # Wrap dense layer with LoRA
                lora = torch.nn.Linear(dense_layer.in_features, dense_layer.out_features, bias=False)
                torch.nn.init.kaiming_uniform_(lora.weight, a=math.sqrt(5))
                module.intermediate.dense_lora = lora  # add a lora dense layer
                module.intermediate.lora_scale = torch.nn.Parameter(torch.zeros(1))  # learnable scale

            # Inject LoRA into output dense (FFN second layer)
            if hasattr(module.output, "dense"):
                dense_layer = module.output.dense

                lora = torch.nn.Linear(dense_layer.in_features, dense_layer.out_features, bias=False)
                torch.nn.init.kaiming_uniform_(lora.weight, a=math.sqrt(5))
                module.output.dense_lora = lora
                module.output.lora_scale = torch.nn.Parameter(torch.zeros(1))

    print("✅ LoRA injected into SegFormer FFN layers.")

def apply_lora_forward(module, x):
    """
    Modify forward pass: output = normal_dense(x) + scale * lora_dense(x)
    """
    normal_out = module.dense(x)
    if hasattr(module, 'dense_lora') and hasattr(module, 'lora_scale'):
        lora_out = module.dense_lora(x)
        return normal_out + module.lora_scale * lora_out
    else:
        return normal_out

def patch_model_with_lora(model):
    """
    Patch SegFormer MLP forward methods to include LoRA
    """
    for name, module in model.named_modules():
        if "encoder.block" in name and hasattr(module, "intermediate"):
            if hasattr(module.intermediate, "dense_lora"):
                module.intermediate.forward = lambda x, module=module.intermediate: apply_lora_forward(module, x)
            if hasattr(module.output, "dense_lora"):
                module.output.forward = lambda x, module=module.output: apply_lora_forward(module, x)

def train():
    # 1. Load pretrained SegFormer model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=Config.FINETUNE.NUM_LABELS,
        ignore_mismatched_sizes=True,
    ).to(Config.DEVICE)

   
    


    # 1b. Inject LoRA
    inject_lora(model, rank=Config.FINETUNE.LORA_RANK)
    patch_model_with_lora(model)

    # 2. Load datasets
    train_loader = prepare_segmentation_loader(
        data_dir=Config.DATA_DIR,
        split="train",
        batch_size=Config.FINETUNE.BATCH_SIZE,
        num_workers=2,
    )
    val_loader = prepare_segmentation_loader(
        data_dir=Config.DATA_DIR,
        split="val",
        batch_size=Config.FINETUNE.BATCH_SIZE,
        num_workers=2,
    )

    checkpoint_path = "results/learned/best_segformer_lora.pth"
    
    state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded weights from {checkpoint_path}")

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only LoRA params will be trained if needed
        lr=Config.FINETUNE.LR,
        weight_decay=Config.FINETUNE.WEIGHT_DECAY,
    )

    best_iou = 0.0
    os.makedirs(Config.FINETUNE.OUT_DIR, exist_ok=True)

    for epoch in range(Config.FINETUNE.EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.FINETUNE.EPOCHS}", unit="batch")

        for images, masks, _, _ in loop:
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = F.cross_entropy(logits, masks, ignore_index=150)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())  # Show live batch loss in tqdm bar

        avg_train_loss = total_loss / len(train_loader)
        print(f"🔁 Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

        # 4. Evaluation
        model.eval()
        iou_total = 0.0
        with torch.no_grad():
            for images, masks, _, _ in tqdm(val_loader, desc="Validating", unit="batch"):
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = model(pixel_values=images)
                preds = outputs.logits

                preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                preds = preds.argmax(dim=1)

                iou = compute_metrics(preds, masks, Config.FINETUNE.NUM_LABELS)
                iou_total += iou

        avg_iou = iou_total / len(val_loader)
        print(f"📊 Val Mean IoU: {avg_iou:.4f}")

        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_model_path = os.path.join(Config.FINETUNE.OUT_DIR, "best_segformer_lora.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model to {best_model_path} (IoU: {best_iou:.4f})")

if __name__ == "__main__":
    train()
