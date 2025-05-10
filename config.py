# config.py
import os
import torch
import math
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class Config:
    DATA_DIR = "/home/jag/codes/unlearn/data"  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class FORGET:
        OUT_DIR = "./results/unlearned"
        BATCH_SIZE = 8 
        EPOCHS = 300       
        LR = 1e-3         
        WEIGHT_DECAY = 0.05  
        BND = math.log(1000) 
        BETA = 0.15
        ALPHA = 0.001
        WARMUP_EPOCHS = 10
        LORA_RANK = 8
        CLASS_TO_FORGET = [60]  # Example class
        NUM_LABELS = 1000

        @staticmethod
        def model_path():
            return os.path.join(Config.FORGET.OUT_DIR, "after_forgetting.pth")

# Ensure directories exist
for dir in [Config.DATA_DIR, Config.FORGET.OUT_DIR]:
    os.makedirs(dir, exist_ok=True)

print(f"Device used: {Config.DEVICE}")
