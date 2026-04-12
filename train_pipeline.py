"""
Entry point — run this file to train the model end-to-end.

  python train_pipeline.py

Steps
─────
  1. Download CIFAR-10 and CIFAR-100 (automatically, first run only)
  2. Build merged 4-class datasets for train and validation
  3. Build ResNet18 with a 4-class head (ImageNet pre-trained)
  4. Phase 1: train head only for PHASE1_EPOCHS epochs
  5. Phase 2: fine-tune all layers for remaining epochs
  6. Save best weights to models/best_model_weights.pth
  7. Save metadata to models/model_metadata.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE
from src.dataset import build_dataset
from src.model import get_model
from src.transforms import get_train_transforms, get_val_transforms
from src.train import train


def main():
    print("=" * 60)
    print("  Image Classifier — End-to-End Training Pipeline")
    print("=" * 60)

    # ── 1. Datasets ────────────────────────────────────────────────
    print("\nBuilding datasets (downloading if needed)…")
    train_ds = build_dataset(train=True,  transform=get_train_transforms())
    val_ds   = build_dataset(train=False, transform=get_val_transforms())
    print(f"  Train samples : {len(train_ds):,}")
    print(f"  Val   samples : {len(val_ds):,}")

    # ── 2. Model ───────────────────────────────────────────────────
    model = get_model(pretrained=True).to(DEVICE)
    print(f"\nModel loaded on {DEVICE}")

    # ── 3. Train ───────────────────────────────────────────────────
    best_acc = train(model, train_ds, val_ds)

    print("\n" + "=" * 60)
    print(f"  Training complete. Best val accuracy: {best_acc:.4f}")
    print("  Run the backend with:")
    print("    uvicorn api.main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
