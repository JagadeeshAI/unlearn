import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image
from transformers import ViTForImageClassification

# Paths
DATA_DIR = "./data"
MODEL_PATH = "./model/vit_pets_best.pth"  # FIXED PATH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load class names from dataset
dataset = OxfordIIITPet(root=DATA_DIR, split="test", download=False)
class_names = dataset.classes

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", num_labels=37, ignore_mismatched_sizes=True
)

# Load fine-tuned weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(image_path):
    """Predicts label for a given image path."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image).logits
        predicted_class = torch.argmax(outputs, dim=1).item()

    predicted_label = class_names[predicted_class]
    print(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    if os.path.exists(image_path):
        predict(image_path)
    else:
        print("Invalid image path!")
