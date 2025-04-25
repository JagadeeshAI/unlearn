import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

def download_data():
    """Downloads Oxford-IIIT Pets dataset if not already present."""
    if not os.path.exists(DATA_DIR):
        print(f"Creating {DATA_DIR} and downloading dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        datasets.OxfordIIITPet(root=DATA_DIR, split="trainval", download=True, transform=transform)
        datasets.OxfordIIITPet(root=DATA_DIR, split="test", download=True, transform=transform)

        print(f"Dataset saved in {DATA_DIR}/train and {DATA_DIR}/test.")
    else:
        print("Data directory already exists. Skipping download.")

if __name__ == "__main__":
    download_data()
