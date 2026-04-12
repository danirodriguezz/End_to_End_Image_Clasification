"""
ResNet18 model with a custom classification head for 4 classes.

Two-phase fine-tuning helpers:
  Phase 1 — freeze_backbone()  → only train the FC head
  Phase 2 — unfreeze_all()     → fine-tune everything with differential LRs
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config import NUM_CLASSES, LR_HEAD, LR_BACKBONE, WEIGHT_DECAY


def get_model(pretrained: bool = True) -> nn.Module:
    """Build ResNet18 with the last FC replaced by a NUM_CLASSES head."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: freeze everything except the new FC head."""
    for name, param in model.named_parameters():
        param.requires_grad = ("fc" in name)


def unfreeze_all(model: nn.Module) -> None:
    """Phase 2: unfreeze all parameters for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_phase1_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Optimizer for phase 1 — only the FC head parameters."""
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )


def get_phase2_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Optimizer for phase 2 — differential LRs: lower for backbone, higher for head."""
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = list(model.fc.parameters())

    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total
