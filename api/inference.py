"""
Model loading and inference logic.

The model is loaded ONCE at module import time (FastAPI startup),
not per-request. This avoids ~400ms latency per call.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    DEVICE, CLASS_NAMES, MODEL_WEIGHTS_PATH, MODEL_METADATA_PATH, NUM_CLASSES
)
from src.model import get_model
from src.transforms import get_val_transforms


# ── Load model at import time ──────────────────────────────────────────────────

def _load_model() -> tuple[torch.nn.Module, list[str]]:
    """
    Load the trained model from disk.
    Raises FileNotFoundError with a helpful message if the weights are missing.
    """
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_WEIGHTS_PATH}'.\n"
            "Run  python train_pipeline.py  first to train and save the model."
        )

    model = get_model(pretrained=False)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Read class names from metadata if available, fall back to config
    class_names = CLASS_NAMES
    if os.path.exists(MODEL_METADATA_PATH):
        with open(MODEL_METADATA_PATH) as f:
            meta = json.load(f)
        class_names = meta.get("class_names", CLASS_NAMES)

    return model, class_names


# Module-level singleton — loaded once, reused for every request
_MODEL, _CLASS_NAMES = _load_model()
_TRANSFORM = get_val_transforms()


# ── Public interface ───────────────────────────────────────────────────────────

def predict_bytes(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes (JPEG / PNG / WebP / …).

    Returns:
        {
          "predictions": [{"class": str, "confidence": float}, …],  # sorted by confidence desc
          "top_class": str
        }
    """
    # Decode image — convert to RGB to handle PNG+alpha (RGBA) gracefully
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]

    with torch.no_grad():
        logits = _MODEL(tensor)                          # [1, NUM_CLASSES]
        probs  = F.softmax(logits, dim=1).squeeze(0)     # [NUM_CLASSES]

    results = [
        {"class": name, "confidence": round(prob.item(), 4)}
        for name, prob in zip(_CLASS_NAMES, probs)
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "predictions": results,
        "top_class": results[0]["class"],
    }
