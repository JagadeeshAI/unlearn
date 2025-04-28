import os
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from config import Config

def get_transforms(image_size=(224, 224)):
    """Transforms for training set."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def get_val_test_transforms(image_size=(224, 224)):
    """Transforms for validation and test set."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def split_forget_retain(dataset, forget_class_idx):
    """Splits dataset into forget subset and retain subset based on label."""
    idx_forget = [i for i, (_, label) in enumerate(dataset) if label == forget_class_idx]
    idx_retain = [i for i, (_, label) in enumerate(dataset) if label != forget_class_idx]

    forget_set = Subset(dataset, idx_forget)
    retain_set = Subset(dataset, idx_retain)

    return forget_set, retain_set

def load_datasets(data_dir, image_size=(224, 224)):
    """Loads train, val, test datasets."""
    train_transform = get_transforms(image_size)
    val_test_transform = get_val_test_transforms(image_size)

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)

    return train_dataset, val_dataset, test_dataset

def prepare_data_loaders(data_dir=Config.DATA_DIR, image_size=(224, 224), num_workers=2):
    """
    Returns a nested dictionary matching the updated structure:
    
    data = {
        'finetune': {
            'train': ..., 'val': ..., 'test': ...
        },
        'forgetting': {
            'train': {'forget': ..., 'retain': ...},
            'val': {'forget': ..., 'retain': ...},
            'test': {'forget': ..., 'retain': ...},
        }
    }
    """
    train_dataset, val_dataset, test_dataset = load_datasets(data_dir, image_size)

    # Map class names to indices
    class_to_idx = train_dataset.class_to_idx
    # Get the class name from Config.FORGET.CLASS_TO_FORGET list (first element)
    forget_class_name = Config.FORGET.CLASS_TO_FORGET[0]
    forget_class_idx = class_to_idx[forget_class_name]

    print(f"🧠 Forgetting class '{forget_class_name}' with label index {forget_class_idx}")

    # Split forget/retain sets
    forget_train, retain_train = split_forget_retain(train_dataset, forget_class_idx)
    forget_val, retain_val = split_forget_retain(val_dataset, forget_class_idx)
    forget_test, retain_test = split_forget_retain(test_dataset, forget_class_idx)

    # Build the loader tree
    data = {
        'finetune': {
            'train': DataLoader(train_dataset, batch_size=Config.FINETUNE.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True),
            'val':   DataLoader(val_dataset, batch_size=Config.FINETUNE.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True),
            'test':  DataLoader(test_dataset, batch_size=Config.FINETUNE.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True),
        },
        'forgetting': {
            'train': {
                'forget': DataLoader(forget_train, batch_size=Config.FORGET.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True),
                'retain': DataLoader(retain_train, batch_size=Config.FORGET.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True),
            },
            'val': {
                'forget': DataLoader(forget_val, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True),
                'retain': DataLoader(retain_val, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True),
            },
            'test': {
                'forget': DataLoader(forget_test, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True),
                'retain': DataLoader(retain_test, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True),
            }
        }
    }

    return data

def main():
    data = prepare_data_loaders()

    print("\n📦 Finetune Loaders:")
    print(f"Train Set Size: {len(data['finetune']['train'].dataset)}")
    print(f"Val Set Size  : {len(data['finetune']['val'].dataset)}")
    print(f"Test Set Size : {len(data['finetune']['test'].dataset)}")

    print("\n📦 Forgetting - Train Split:")
    print(f"Forget Train Size: {len(data['forgetting']['train']['forget'].dataset)}")
    print(f"Retain Train Size: {len(data['forgetting']['train']['retain'].dataset)}")

    print("\n📦 Forgetting - Val Split:")
    print(f"Forget Val Size: {len(data['forgetting']['val']['forget'].dataset)}")
    print(f"Retain Val Size: {len(data['forgetting']['val']['retain'].dataset)}")

    print("\n📦 Forgetting - Test Split:")
    print(f"Forget Test Size: {len(data['forgetting']['test']['forget'].dataset)}")
    print(f"Retain Test Size: {len(data['forgetting']['test']['retain'].dataset)}")

if __name__ == "__main__":
    main()
