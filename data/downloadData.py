import os
import shutil
import random
from torchvision.datasets import OxfordIIITPet
from config import Config

def download_and_prepare_data():
    """Downloads Oxford-IIIT Pets dataset and splits into train, val, and test folders."""
    dataset = OxfordIIITPet(root=Config.DATA_DIR, split="trainval", download=True)

    # Paths
    images_dir = os.path.join(Config.DATA_DIR, "oxford-iiit-pet", "images")
    train_dir = os.path.join(Config.DATA_DIR, "train")
    val_dir = os.path.join(Config.DATA_DIR, "val")
    test_dir = os.path.join(Config.DATA_DIR, "test")

    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(folder, exist_ok=True)

    # Get class names
    class_names = dataset.classes

    # Prepare (image_filename, class_name) list
    data = []
    for i in range(len(dataset)):
        image_path, label = dataset._images[i], dataset[i][1]  # label is 0-based
        class_name = class_names[label]
        image_name = image_path.with_suffix('.jpg').name  # e.g., "Abyssinian_12.jpg"
        data.append((image_name, class_name))

    # Shuffle and split
    random.shuffle(data)
    total = len(data)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    splits = {
        train_dir: data[:train_end],
        val_dir: data[train_end:val_end],
        test_dir: data[val_end:]
    }

    # Copy images
    for split_dir, split_data in splits.items():
        for img_name, class_name in split_data:
            src = os.path.join(images_dir, img_name)
            dst_dir = os.path.join(split_dir, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, img_name))

    shutil.move("data/oxford-iiit-pet/test", "data/test")
    shutil.move("data/oxford-iiit-pet/train", "data/train")
    shutil.move("data/oxford-iiit-pet/val", "data/val")
    shutil.rmtree("data/oxford-iiit-pet")
    print("✅ Dataset successfully split into train, val, and test folders.")

if __name__ == "__main__":
    download_and_prepare_data()
