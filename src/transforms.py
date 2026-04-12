"""
Data augmentation and pre-processing pipelines.

CIFAR images are 32×32 — we upscale to 224×224 for ResNet18.
ImageNet normalisation is kept because we use ImageNet pre-trained weights.
"""
import torchvision.transforms as T
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms() -> T.Compose:
    """Aggressive augmentation for training to compensate for the small CIFAR size."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> T.Compose:
    """Deterministic pipeline for validation / inference."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
