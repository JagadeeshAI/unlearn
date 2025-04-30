import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch

class Config:
    DATA_DIR = "./data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class FINETUNE:
        OUT_DIR = "./results/learned"
        SEGFORMER_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
        NUM_LABELS = 150
        BATCH_SIZE = 4
        EPOCHS = 10
        LR = 3e-5
        WEIGHT_DECAY = 1e-2
        LORA_RANK = 8  # Added this for LoRA rank

for dir in [Config.DATA_DIR, Config.FINETUNE.OUT_DIR]:
    os.makedirs(dir, exist_ok=True)

print(f"✅ Device selected: {Config.DEVICE}")
