# data.py

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define the correct ADE20K vehicle classes to forget
FORGET_CLASS_IDS = [21, 77, 81, 84, 103, 104, 91, 128]  # Correct based on your table (car, bus, truck, etc.)

class ADE20KForgetDataset(Dataset):
    """
    ADE20K Dataset with dynamic Df/Dr mask creation for selective forgetting.
    """
    def __init__(self, image_dir, mask_dir, image_size=(512, 512)):
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask)).long()

        # Create Df and Dr masks
        df_mask = torch.isin(mask, torch.tensor(FORGET_CLASS_IDS)).long()
        dr_mask = 1 - df_mask

        return image, mask, df_mask, dr_mask


def get_transforms(image_size=(512, 512)):
    """Transformations for images."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def prepare_segmentation_loader(data_dir, split="train", batch_size=8, num_workers=2):
    """
    Prepares a DataLoader for segmentation task.
    :param data_dir: Base data directory (should have train/val folders)
    :param split: 'train' or 'val'
    :return: DataLoader
    """
    image_root = os.path.join(data_dir, split, "images")
    mask_root = os.path.join(data_dir, split, "annotations")

    dataset = ADE20KForgetDataset(
        image_dir=image_root,
        mask_dir=mask_root,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader

def main():
    data_dir = "data"  # Your root directory (containing 'train' and 'val')
    batch_size = 4

    train_loader = prepare_segmentation_loader(data_dir=data_dir, split="train", batch_size=batch_size)
    val_loader = prepare_segmentation_loader(data_dir=data_dir, split="val", batch_size=batch_size)

    images, masks, df_masks, dr_masks = next(iter(train_loader))
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Df_masks shape: {df_masks.shape}")
    print(f"Dr_masks shape: {dr_masks.shape}")

if __name__ == "__main__":
    main()
