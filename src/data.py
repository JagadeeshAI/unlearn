import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import Config

def get_transforms(image_size=(224, 224)):
    """
    Returns transformation pipeline for images.
    """
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

def get_val_test_transforms(image_size=(224, 224)):
    """Transforms for validation and test set."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    

def load_oxford_pets_dataset(data_dir=Config.DATA_DIR, image_size=(224, 224), batch_size=32, num_workers=20):
    """
    Loads and returns train, val, and test DataLoaders from folder structure.
    """
    train_transform = get_transforms(image_size)
    val_test_transform = get_val_test_transforms(image_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = load_oxford_pets_dataset(
        data_dir=Config.DATA_DIR,
        image_size=(224, 224),
        batch_size=32,
        num_workers=2
    )

    print("📦 Sample batch from Train Loader:")
    for images, labels in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break

    print("\n📦 Sample batch from Validation Loader:")
    for images, labels in val_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break

    print("\n📦 Sample batch from Test Loader:")
    for images, labels in test_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break

if __name__ == "__main__":
    main()
