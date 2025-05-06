# src/data.py

import os
import json
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from config import Config

def get_transforms(image_size=(224, 224)):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

def get_val_test_transforms(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

class JsonlImageDataset(Dataset):
    def __init__(self, jsonl_path, transform=None, tag_filter=None):
        self.samples = []
        print(f"📖 Reading {jsonl_path} with tag_filter={tag_filter}...")
        with open(jsonl_path, "r") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(jsonl_path)}"):
                item = json.loads(line)
                if tag_filter and item["tag"] != tag_filter:
                    continue
                self.samples.append(item)

        self.transform = transform
        print(f"✅ Loaded {len(self.samples)} items.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        try:
            image = Image.open(entry["path"]).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"⚠️ Skipping unreadable image: {entry['path']} ({e})")
            return self.__getitem__((idx + 1) % len(self.samples))  # Try next item

        if self.transform:
            image = self.transform(image)
        return image, entry["label"]

def prepare_data_loaders(data_dir=Config.DATA_DIR, image_size=(224, 224), num_workers=8):
    index_dir = os.path.join(data_dir, "index")
    train_jsonl = os.path.join(index_dir, "train.jsonl")
    val_jsonl   = os.path.join(index_dir, "val.jsonl")

    print("🔧 Initializing DataLoaders...\n")

    train_forget = JsonlImageDataset(train_jsonl, transform=get_transforms(image_size), tag_filter="forget")
    train_retain = JsonlImageDataset(train_jsonl, transform=get_transforms(image_size), tag_filter="retain")

    val_forget = JsonlImageDataset(val_jsonl, transform=get_val_test_transforms(image_size), tag_filter="forget")
    val_retain = JsonlImageDataset(val_jsonl, transform=get_val_test_transforms(image_size), tag_filter="retain")

    combined_val = ConcatDataset([val_forget, val_retain])

    data = {
        'forgetting': {
            'train': {
                'forget': DataLoader(train_forget, batch_size=Config.FORGET.BATCH_SIZE, shuffle=True,
                                     num_workers=num_workers, pin_memory=True),
                'retain': DataLoader(train_retain, batch_size=Config.FORGET.BATCH_SIZE, shuffle=True,
                                     num_workers=num_workers, pin_memory=True),
            },
            'val': {
                'forget': DataLoader(val_forget, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False,
                                     num_workers=num_workers, pin_memory=True),
                'retain': DataLoader(val_retain, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False,
                                     num_workers=num_workers, pin_memory=True),
            },
            'test': {
                'forget': DataLoader(val_forget, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False,
                                     num_workers=num_workers, pin_memory=True),
                'retain': DataLoader(val_retain, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False,
                                     num_workers=num_workers, pin_memory=True),
            }
        },
        'finetune': {
            'val': DataLoader(combined_val, batch_size=Config.FORGET.BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        }
    }

    print("\n✅ All DataLoaders prepared.")
    return data

def main():
    data = prepare_data_loaders()

    print("\n📦 Forgetting - Train Split:")
    print(f"Forget Train Size: {len(data['forgetting']['train']['forget'].dataset)}")
    print(f"Retain Train Size: {len(data['forgetting']['train']['retain'].dataset)}")

    print("\n📦 Forgetting - Val Split:")
    print(f"Forget Val Size: {len(data['forgetting']['val']['forget'].dataset)}")
    print(f"Retain Val Size: {len(data['forgetting']['val']['retain'].dataset)}")

    print("\n📦 Combined Finetune Val Split:")
    combined = data['finetune']['val'].dataset
    print(f"Combined Val Size: {sum(len(d) for d in combined.datasets)}")

if __name__ == "__main__":
    main()
