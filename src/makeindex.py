import os
import json
from tqdm import tqdm
from config import Config

def load_synset_to_label_map(json_path="imagenet_class_index.json"):
    print(f"📄 Loading synset mapping from: {json_path}")
    with open(json_path) as f:
        idx_to_data = json.load(f)

    synset_to_idx = {v[0]: int(k) for k, v in idx_to_data.items()}
    synset_to_name = {v[0]: v[1] for k, v in idx_to_data.items()}

    print(f"✅ Loaded {len(synset_to_idx)} synsets from imagenet_class_index.json")
    return synset_to_idx, synset_to_name

def index_split(split_name, split_path, forget_class, synset_to_idx, synset_to_name):
    index = []
    print(f"📂 Indexing {split_name} set...")

    # Create numeric label to synset map
    idx_to_synset = {v: k for k, v in synset_to_idx.items()}

    for class_name in sorted(os.listdir(split_path)):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        try:
            label = int(class_name)
            synset = idx_to_synset[label]
        except (ValueError, KeyError):
            print(f"⚠️ Skipping unknown or invalid class: {class_name}")
            continue

        label_name = synset_to_name[synset]
        tag = "forget" if label == forget_class else "retain"

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in tqdm(image_files, desc=f"{split_name}/{class_name}", leave=False):
            img_path = os.path.join(class_dir, img_name)
            index.append({
                "path": img_path,
                "label": label,
                "label_name": label_name,
                "class": synset,
                "tag": tag
            })

    print(f"✅ Indexed {split_name}: {len(index)} samples.")
    return index


def main():
    data_dir = Config.DATA_DIR
    forget_class = Config.FORGET.CLASS_TO_FORGET[0]
    output_dir = os.path.join(data_dir, "index")
    os.makedirs(output_dir, exist_ok=True)

    synset_to_idx, synset_to_name = load_synset_to_label_map("data/imagenet_class_index.json")

    for split in ["train", "val"]:
        split_path = os.path.join(data_dir, split)
        out_path = os.path.join(output_dir, f"{split}.jsonl")

        index = index_split(split, split_path, forget_class, synset_to_idx, synset_to_name)
        with open(out_path, "w") as f:
            for item in index:
                f.write(json.dumps(item) + "\n")

    print("📁 JSONL indexing complete. Saved to:", output_dir)

if __name__ == "__main__":
    main()
