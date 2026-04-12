"""
Central configuration for training and inference.
All hyperparameters and paths live here.
"""
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_WEIGHTS_PATH  = os.path.join(MODELS_DIR, "best_model_weights.pth")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Classes ────────────────────────────────────────────────────────────────────
# Alphabetical order — must match the order used during training
CLASS_NAMES = ["airplane", "bicycle", "car", "dog"]
NUM_CLASSES  = len(CLASS_NAMES)

# ── Dataset source mappings ────────────────────────────────────────────────────
# CIFAR-10 class indices we want (run cifar10.classes to verify)
CIFAR10_WANTED = {
    0: 0,   # airplane  → our class 0
    1: 2,   # automobile → our class 2 (car)
    5: 3,   # dog       → our class 3
}

# CIFAR-100 class index for bicycle (alphabetical order in CIFAR-100: index 8)
CIFAR100_BICYCLE_IDX = 8
CIFAR100_BICYCLE_TARGET = 1  # our class 1

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE    = 64
NUM_EPOCHS    = 20
PHASE1_EPOCHS = 5          # Frozen backbone (only train head)
LR_HEAD       = 1e-3       # Learning rate for the classification head
LR_BACKBONE   = 1e-4       # Learning rate for the backbone (phase 2)
WEIGHT_DECAY  = 1e-4
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA     = 0.1

# ── Class weights for imbalanced data ─────────────────────────────────────────
# CIFAR-10 has 5000 samples/class; CIFAR-100 has 500 for bicycle → weight 10x
CLASS_WEIGHTS = torch.tensor([1.0, 10.0, 1.0, 1.0], dtype=torch.float)

# ── Image settings ─────────────────────────────────────────────────────────────
IMAGE_SIZE = 224   # ResNet18 expects 224×224
# ImageNet normalisation (used because we start from ImageNet pre-trained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
