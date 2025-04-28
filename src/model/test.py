# src/forget/test.py
import os
import json
import logging
from tqdm import tqdm

# suppress TF/HF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import torch
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

from config import Config
from src.model.data import prepare_data_loaders

def evaluate_model():
    device = Config.DEVICE
    print(f"🔍 Device: {device}")

    # 1) Load base ViT and wrap with LoRA (inference_mode=True)
    print("🧠 Loading ViT and injecting LoRA adapters for inference…")
    base = ViTForImageClassification.from_pretrained(
        Config.FINETUNE.VIT_MODEL,
        num_labels=37,
        ignore_mismatched_sizes=True
    )
    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=Config.FORGET.LORA_RANK,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=["intermediate.dense", "output.dense"]
    )
    model = get_peft_model(base, peft_cfg)
    # load the forgotten model
    model.load_state_dict(torch.load(Config.FORGET.model_path(), map_location=device), strict=False)
    model.to(device).eval()
    print("✅ Model loaded!")

    # 2) Prepare test DataLoader (use FINETUNE batch size)
    data = prepare_data_loaders(
        data_dir=Config.DATA_DIR,
        image_size=(224, 224),
        num_workers=2
    )
    test_loader = data['finetune']['test']

    # 3) Run evaluation
    correct = 0
    total = 0
    class_correct = [0] * 37
    class_total   = [0] * 37
    class_names   = test_loader.dataset.classes

    print("🚀 Evaluating on Oxford-IIIT Pet Test set…")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            # forward through the base_model to avoid unwanted kwargs
            outputs = model.base_model(pixel_values=images)
            preds = outputs.logits.argmax(dim=-1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for i, lab in enumerate(labels):
                class_total[lab] += 1
                if preds[i] == lab:
                    class_correct[lab] += 1

    overall_acc = 100 * correct / total
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.2f}%")

    # 4) Per-class accuracies
    print("\n📊 Per-Class Accuracy:")
    results = {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": {}
    }
    for idx, name in enumerate(class_names):
        if class_total[idx] > 0:
            acc = 100 * class_correct[idx] / class_total[idx]
        else:
            acc = 0.0
        print(f"  {name}: {acc:.2f}%")
        results["per_class_accuracy"][name] = acc

    # 5) Save to JSON
    out_path = os.path.join(Config.FORGET.OUT_DIR, "after_forgetting_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n📝 Results saved to {out_path}")

if __name__ == "__main__":
    evaluate_model()
