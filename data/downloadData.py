import os
import zipfile
import requests
import shutil
from tqdm import tqdm

def download_ade20k(root="data"):
    os.makedirs(root, exist_ok=True)
    url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    zip_path = os.path.join(root, "ADEChallengeData2016.zip")

    # Download
    if not os.path.exists(zip_path):
        with requests.get(url, stream=True) as r, open(zip_path, 'wb') as f, tqdm(
            unit='B', unit_scale=True, unit_divisor=1024, total=int(r.headers.get('content-length', 0))
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print("✅ Download complete.")
    else:
        print("📦 Zip file already exists. Skipping download.")

    # Extract
    extracted_dir = os.path.join(root, "ADEChallengeData2016")
    if not os.path.exists(os.path.join(extracted_dir, "images")):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        print("✅ Extraction complete.")
    else:
        print("📁 Data already extracted. Skipping extraction.")

    # Setup output folders (train/val based on ADE20K official split)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "annotations"), exist_ok=True)

    # Copy images and annotations for official train split
    train_img_dir = os.path.join(extracted_dir, "images/training")
    train_ann_dir = os.path.join(extracted_dir, "annotations/training")
    for fname in os.listdir(train_img_dir):
        if fname.endswith(".jpg"):
            ann_name = fname.replace(".jpg", ".png")
            img_src = os.path.join(train_img_dir, fname)
            ann_src = os.path.join(train_ann_dir, ann_name)
            img_dst = os.path.join(root, "train", "images", fname)
            ann_dst = os.path.join(root, "train", "annotations", ann_name)
            shutil.copy(img_src, img_dst)
            shutil.copy(ann_src, ann_dst)

    # Copy images and annotations for official val split
    val_img_dir = os.path.join(extracted_dir, "images/validation")
    val_ann_dir = os.path.join(extracted_dir, "annotations/validation")
    for fname in os.listdir(val_img_dir):
        if fname.endswith(".jpg"):
            ann_name = fname.replace(".jpg", ".png")
            img_src = os.path.join(val_img_dir, fname)
            ann_src = os.path.join(val_ann_dir, ann_name)
            img_dst = os.path.join(root, "val", "images", fname)
            ann_dst = os.path.join(root, "val", "annotations", ann_name)
            shutil.copy(img_src, img_dst)
            shutil.copy(ann_src, ann_dst)

    print("✅ Dataset organized with official ADE20K train/val split.")

if __name__ == "__main__":
    download_ade20k()
