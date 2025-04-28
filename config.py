# config.py
import os
import torch
import math
# Suppress TensorFlow warnings if TF ever gets imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class Config:
    # General settings
    DATA_DIR = "./data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Finetuning Hyperparameters
    class FINETUNE:
        OUT_DIR = "./results/learned"
        VIT_MODEL = "google/vit-base-patch16-224"  # Huggingface model name
        CUSTOM_PRETRAINED_PATH = os.path.join(OUT_DIR, "vit_custom_pretrained.pth")
        BATCH_SIZE = 16
        EPOCHS = 20
        LR = 1e-5
        WEIGHT_DECAY = 5e-4
        NUM_LABELS=37
        
        @staticmethod
        def model_path():
            return os.path.join(Config.FINETUNE.OUT_DIR, "finetuned.pth")

    # Forgetting Hyperparameters
    class FORGET:
        OUT_DIR = "./results/unlearned"
        BATCH_SIZE = 16
        EPOCHS = 100
        LR = 1e-4
        WEIGHT_DECAY = 1e-2
        BND = math.log(37)          # CE threshold for forgetting
        BETA = 0.15        # Weight on forgetting loss
        ALPHA = 0.001       # Group sparsity loss weight after warmup
        WARMUP_EPOCHS = 10 # Epochs before applying sparsity
        LORA_RANK = 8      # Rank of LoRA adapters
        CLASS_TO_FORGET = ["Bombay"]
        NUM_LABELS=37
        @staticmethod
        def model_path():
            return os.path.join(Config.FORGET.OUT_DIR, "after_forgetting.pth")

# Make sure all necessary directories exist
for dir in [Config.DATA_DIR, Config.FINETUNE.OUT_DIR, Config.FORGET.OUT_DIR]:
    os.makedirs(dir, exist_ok=True)



print(f"Device used: {Config.DEVICE}")
