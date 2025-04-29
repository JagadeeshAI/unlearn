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
from transformers import ViTForImageClassification
from config import Config
from src.model.data import prepare_data_loaders

def evaluate_model():
    # 1) Path to full merged model
    ckpt = os.path.join(Config.FORGET.OUT_DIR, "merged_model.pth")
    if not os.path.isfile(ckpt):
        print(f"❌ Merged model not found at {ckpt}")
        return

    # 2) Load the base ViT model
    print("🔍 Loading ViT architecture...")
    model = ViTForImageClassification.from_pretrained(
        Config.FINETUNE.VIT_MODEL,
        num_labels=Config.FINETUNE.NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    print("🧠 Loading merged weights from:", ckpt)
    state = torch.load(ckpt, map_location=Config.DEVICE)
    model.load_state_dict(state, strict=True)

    model.to(Config.DEVICE).eval()
    print("✅ Model ready for evaluation.")

    # 3) Prepare your test data
    data = prepare_data_loaders(
        data_dir=Config.DATA_DIR,
        image_size=(224, 224),
        num_workers=2
    )
    test_loader = data['finetune']['test']

    # 4) Run evaluation
    correct = total = 0
    per_class_correct = [0] * Config.FINETUNE.NUM_LABELS
    per_class_total   = [0] * Config.FINETUNE.NUM_LABELS

    print("🚀 Evaluating on test set…")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            logits = model(pixel_values=images).logits
            preds  = logits.argmax(dim=-1)

            total  += labels.size(0)
            correct += (preds == labels).sum().item()

            for i in range(labels.size(0)):
                lbl = labels[i].item()
                if preds[i].item() == lbl:
                    per_class_correct[lbl] += 1
                per_class_total[lbl] += 1

    overall_acc = 100 * correct / total
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.2f}%")

    # 5) Per-class breakdown
    print("\n📊 Per-Class Accuracy:")
    class_names = test_loader.dataset.classes
    results = {"overall_accuracy": overall_acc, "per_class_accuracy": {}}

    for idx, name in enumerate(class_names):
        if per_class_total[idx] > 0:
            acc = 100 * per_class_correct[idx] / per_class_total[idx]
        else:
            acc = 0.0
        print(f"  {name}: {acc:.2f}%")
        results["per_class_accuracy"][name] = acc

    # 6) Dump JSON report
    out_path = os.path.join(Config.FORGET.OUT_DIR, "after.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n📝 Results saved to {out_path}")

if __name__ == "__main__":
    evaluate_model()
