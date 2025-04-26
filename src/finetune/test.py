import os
import json

# Suppress TensorFlow logs if unnecessary
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch
from tqdm import tqdm
from transformers import ViTForImageClassification
from config import Config
from src.data import load_oxford_pets_dataset

def evaluate_model():
    """Loads the trained ViT model and evaluates accuracy on Oxford-IIIT Pets Test set."""
    if not os.path.exists(Config.MODEL_PATH):
        print("❌ Error: Model weights not found. Make sure training has completed.")
        return

    print("🔍 Loading model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=37,  # 37 classes
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")

    # Load test data
    _, _, test_loader = load_oxford_pets_dataset(
        data_dir=Config.DATA_DIR,
        image_size=(224, 224),
        batch_size=Config.BATCH_SIZE,
        num_workers=2,
    )

    class_correct = [0] * 37
    class_total = [0] * 37

    correct = 0
    total = 0

    print("🚀 Evaluating model on Oxford-IIIT Pets Test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f"\n✅ Overall Test Accuracy: {accuracy:.2f}%")

    # Now calculate per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    test_results = {}

    # You can also map class indices to class names if available
    class_names = test_loader.dataset.classes  # ImageFolder gives this automatically

    for i in range(37):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
        else:
            acc = 0.0
        class_name = class_names[i]
        print(f"  {class_name}: {acc:.2f}%")
        test_results[class_name] = acc

    # Save per-class results to JSON
    result_path = os.path.join(Config.MODEL_DIR, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"\n📝 Per-class accuracies saved to {result_path}")

if __name__ == "__main__":
    evaluate_model()
