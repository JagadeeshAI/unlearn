import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch


class Config:
    # Directories
    DATA_DIR = "./data"
    MODEL_DIR = "./results"
    MODEL_PATH = os.path.join(MODEL_DIR, "vit_pets_best.pth")

    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 5e-4 

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device used is {Config.DEVICE}")
