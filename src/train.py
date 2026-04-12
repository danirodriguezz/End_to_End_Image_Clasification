"""
Training and evaluation loop.

Exports:
  train_one_epoch(model, loader, criterion, optimizer, device) → (loss, acc)
  evaluate(model, loader, criterion, device) → (loss, acc, class_report)
  train(model, train_ds, val_ds) → best_val_acc
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from datetime import date
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import (
    BATCH_SIZE, NUM_EPOCHS, PHASE1_EPOCHS,
    CLASS_WEIGHTS, DEVICE, CLASS_NAMES,
    MODEL_WEIGHTS_PATH, MODEL_METADATA_PATH,
    SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA,
)
from src.model import (
    freeze_backbone, unfreeze_all,
    get_phase1_optimizer, get_phase2_optimizer,
    count_parameters,
)
from src.dataset import build_weighted_sampler


# ── Single epoch ───────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, str]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="    val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    return total_loss / total, correct / total, report


# ── Full training pipeline ─────────────────────────────────────────────────────

def train(model: nn.Module, train_ds, val_ds) -> float:
    """
    Two-phase fine-tuning:
      Phase 1 (epochs 1–PHASE1_EPOCHS):  frozen backbone, train head only
      Phase 2 (epochs PHASE1_EPOCHS+1–NUM_EPOCHS): full fine-tuning with differential LRs

    Returns the best validation accuracy achieved.
    """
    # ── Data loaders ───────────────────────────────────────────────────────────
    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Weighted loss to further boost bicycle gradient
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))

    # ── Phase 1 setup ──────────────────────────────────────────────────────────
    freeze_backbone(model)
    optimizer = get_phase1_optimizer(model)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nDevice: {DEVICE}")
    trainable, total = count_parameters(model)
    print(f"Phase 1 — trainable params: {trainable:,} / {total:,}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Switch to phase 2
        if epoch == PHASE1_EPOCHS + 1:
            print("\n── Switching to Phase 2: full fine-tuning ──\n")
            unfreeze_all(model)
            optimizer = get_phase2_optimizer(model)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
            )
            trainable, _ = count_parameters(model)
            print(f"Phase 2 — trainable params: {trainable:,} / {total:,}\n")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc, report = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        flag = " ← best" if val_acc > best_val_acc else ""
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({time.time()-t0:.1f}s){flag}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"  ✓ Saved best model → {MODEL_WEIGHTS_PATH}")

        if epoch == NUM_EPOCHS:
            print("\nFinal per-class report:\n", report)

    # Save metadata alongside the weights
    metadata = {
        "class_names": CLASS_NAMES,
        "input_size": 224,
        "architecture": "resnet18",
        "val_accuracy": round(best_val_acc, 4),
        "trained_date": str(date.today()),
        "num_epochs": NUM_EPOCHS,
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved → {MODEL_METADATA_PATH}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return best_val_acc
