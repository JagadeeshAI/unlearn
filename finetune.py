import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from torch import nn, optim
from tqdm import tqdm

# Directories
DATA_DIR = "./data"
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "vit_pets_best.pth")

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloaders():
    """Prepares train and test DataLoaders."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.OxfordIIITPet(root=DATA_DIR, split="trainval", transform=transform, download=False)
    test_dataset = datasets.OxfordIIITPet(root=DATA_DIR, split="test", transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset.classes

def evaluate(model, test_loader):
    """Evaluates the model and returns accuracy."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def train():
    """Fine-tunes ViT on Oxford-IIIT Pets dataset."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading ViT model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=37,  # Oxford Pets has 37 classes
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    train_loader, test_loader, _ = get_dataloaders()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0  # Track best accuracy

    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct, total = 0, 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for batch_idx, (images, labels) in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update tqdm display
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

            # Print accuracy every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}: Accuracy = {100 * correct / total:.2f}%")

        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader)
        print(f"Epoch {epoch+1} Test Accuracy: {test_accuracy:.2f}%")

        # Save model only if accuracy improves
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    print("Training complete!")

if __name__ == "__main__":
    train()
