import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from tqdm import tqdm  # Import tqdm for progress bar

DATA_DIR = "./data"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "cairs100_vit.pth")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_test_data():
    """Loads CIFAR-100 test dataset and returns a DataLoader."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ViT input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, transform=transform, download=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return test_loader

def evaluate_model():
    """Loads the trained ViT model and evaluates accuracy on CIFAR-100 test set."""
    if not os.path.exists(MODEL_PATH):
        print("Error: Model weights not found. Run `process.py` first.")
        return
    
    print("Loading model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=100,
        ignore_mismatched_sizes=True  # Ensure correct classifier layer
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_loader = load_test_data()
    correct = 0
    total = 0

    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).logits  # Get raw logits
            _, predicted = torch.max(outputs, 1)  # Get highest probability class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
