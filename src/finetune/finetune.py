import os
import logging

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from tqdm import tqdm

from config import Config
from src.finetune.data import prepare_data_loaders  # updated import!

def evaluate(model, data_loader):
    """Evaluates the model and returns accuracy."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def train():
    """Fine-tunes ViT on Oxford-IIIT Pets dataset."""
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print("Loading ViT model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=37,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    ).to(Config.DEVICE)

    # Load datasets using new prepare_data_loaders
    data = prepare_data_loaders(
        data_dir=Config.DATA_DIR,
        image_size=(224, 224),
        batch_size=Config.BATCH_SIZE,
        num_workers=4
    )

    train_loader = data['finetune']['train']
    val_loader   = data['finetune']['val']
    test_loader  = data['finetune']['test']

    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
        eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        loop = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{Config.EPOCHS}",
            unit="batch",
        )

        for batch_idx, (images, labels) in loop:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()

            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        scheduler.step()

        # Evaluate on validation set
        val_accuracy = evaluate(model, val_loader)
        print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}%")

        # Save model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    print("Training complete!")

    # Final evaluation on test set
    test_accuracy = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    train()
