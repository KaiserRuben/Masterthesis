"""
Training Script for Multi-Task Encoder

Finetunes a small model on 100 anchor scenes to predict semantic keys.
The learned backbone features are then used for latent navigation on the full dataset.

Usage:
    # Train with all keys
    python train.py --key-mode all --epochs 50

    # Train with top keys only
    python train.py --key-mode top --epochs 50

    # Different backbone
    python train.py --backbone mobilenetv3_small_100 --key-mode all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_anchor_classifications, get_anchor_image_paths
from sub_finetuned_encoder.model import (
    MultiTaskEncoder,
    MultiTaskLoss,
    create_label_encoder,
    get_key_config,
    KEY_VALUES,
)


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(__file__).parents[3] / "data" / "EMB-001" / "finetuned_models"


# =============================================================================
# DATASET
# =============================================================================

class AnchorDataset(Dataset):
    """Dataset for anchor scenes with semantic labels."""

    def __init__(
        self,
        scene_ids: list[str],
        image_paths: dict[str, Path],
        classifications: dict[str, dict],
        keys: list[str],
        encoders: dict[str, dict],
        transform: transforms.Compose | None = None,
    ):
        self.scene_ids = scene_ids
        self.image_paths = image_paths
        self.classifications = classifications
        self.keys = keys
        self.encoders = encoders
        self.transform = transform

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int], str]:
        scene_id = self.scene_ids[idx]

        # Load image
        img_path = self.image_paths[scene_id]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Encode labels
        labels = {}
        scene_labels = self.classifications.get(scene_id, {})
        for key in self.keys:
            raw_value = scene_labels.get(key)

            # Extract actual value from nested dict structure
            # e.g., weather: {"reasoning": "...", "weather": "foggy"} -> "foggy"
            # or traffic_situation: {"points": {...}, "category": "complex"} -> "complex"
            value = None
            if isinstance(raw_value, dict):
                # Try the key name first (e.g., weather['weather'])
                if key in raw_value:
                    value = raw_value[key]
                # Special case for traffic_situation which uses 'category'
                elif 'category' in raw_value:
                    value = raw_value['category']
            elif raw_value is not None:
                value = raw_value

            # Handle boolean keys: missing/None means False
            is_boolean_key = key in [
                'pedestrians_present', 'cyclists_present', 'construction_activity',
                'traffic_signals_visible', 'similar_object_confusion'
            ]

            if value is None:
                if is_boolean_key:
                    value = False  # Missing boolean = False
                else:
                    labels[key] = -1  # Missing categorical
                    continue

            if not isinstance(value, (str, bool, int, float)):
                # Skip complex values (dicts, lists)
                labels[key] = -1
            elif value in self.encoders[key]:
                labels[key] = self.encoders[key][value]
            else:
                labels[key] = -1  # Unknown value not in our predefined set

        return image, labels, scene_id


def get_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    """Get transforms for training/validation."""
    if train:
        # Light augmentation - avoid heavy distortion for driving scenes
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),  # Less frequent flip
            transforms.ColorJitter(
                brightness=0.15,  # Reduced
                contrast=0.15,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.RandomAffine(
                degrees=5,  # Reduced rotation
                translate=(0.05, 0.05),  # Reduced translation
                scale=(0.95, 1.05),  # Reduced scale
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def collate_fn(batch):
    """Custom collate to handle label dicts."""
    images, labels_list, scene_ids = zip(*batch)
    images = torch.stack(images)

    # Stack labels per key
    keys = labels_list[0].keys()
    labels = {
        key: torch.tensor([l[key] for l in labels_list])
        for key in keys
    }

    return images, labels, list(scene_ids)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    per_key_losses = {key: 0.0 for key in criterion.keys}
    num_batches = 0

    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        logits = model(images)
        loss, key_losses = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for key, kl in key_losses.items():
            per_key_losses[key] += kl.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "per_key": {k: v / num_batches for k, v in per_key_losses.items()},
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    per_key_losses = {key: 0.0 for key in criterion.keys}
    per_key_correct = {key: 0 for key in criterion.keys}
    per_key_total = {key: 0 for key in criterion.keys}
    num_batches = 0

    for images, labels, _ in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        logits = model(images)
        loss, key_losses = criterion(logits, labels)

        total_loss += loss.item()
        for key, kl in key_losses.items():
            per_key_losses[key] += kl.item()

        # Accuracy per key
        for key in criterion.keys:
            if key not in logits:
                continue
            mask = labels[key] >= 0
            if mask.sum() == 0:
                continue
            preds = logits[key][mask].argmax(dim=1)
            correct = (preds == labels[key][mask]).sum().item()
            per_key_correct[key] += correct
            per_key_total[key] += mask.sum().item()

        num_batches += 1

    # Compute accuracies
    per_key_acc = {}
    for key in criterion.keys:
        if per_key_total[key] > 0:
            per_key_acc[key] = per_key_correct[key] / per_key_total[key]
        else:
            per_key_acc[key] = 0.0

    return {
        "loss": total_loss / max(num_batches, 1),
        "per_key_loss": {k: v / max(num_batches, 1) for k, v in per_key_losses.items()},
        "per_key_acc": per_key_acc,
        "mean_acc": np.mean(list(per_key_acc.values())),
    }


def train(
    backbone: str = "efficientnet_b0",
    key_mode: str = "all",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    freeze_backbone_epochs: int = 5,
) -> Path:
    """
    Train multi-task encoder.

    Args:
        backbone: timm backbone name
        key_mode: 'all' or 'top' keys
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        val_split: Validation split ratio
        seed: Random seed
        device: Device ('cuda', 'mps', 'cpu', or None for auto)
        freeze_backbone_epochs: Epochs to freeze backbone (warmup heads)

    Returns:
        Path to saved model directory
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load data
    print("Loading anchor data...")
    classifications = load_anchor_classifications()
    image_paths = get_anchor_image_paths()

    # Filter to scenes with images
    scene_ids = [sid for sid in classifications if sid in image_paths]
    print(f"Found {len(scene_ids)} scenes with images")

    # Get keys
    cat_keys, bool_keys = get_key_config(key_mode)
    all_keys = cat_keys + bool_keys
    print(f"Training on {len(all_keys)} keys ({key_mode} mode)")

    # Create encoders
    encoders = create_label_encoder(all_keys)

    # Create dataset
    full_dataset = AnchorDataset(
        scene_ids=scene_ids,
        image_paths=image_paths,
        classifications=classifications,
        keys=all_keys,
        encoders=encoders,
        transform=None,  # Set per split
    )

    # Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_ids = [scene_ids[i] for i in train_indices]
    val_ids = [scene_ids[i] for i in val_indices]

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create split datasets with transforms
    train_dataset = AnchorDataset(
        scene_ids=train_ids,
        image_paths=image_paths,
        classifications=classifications,
        keys=all_keys,
        encoders=encoders,
        transform=get_transforms(train=True),
    )

    val_dataset = AnchorDataset(
        scene_ids=val_ids,
        image_paths=image_paths,
        classifications=classifications,
        keys=all_keys,
        encoders=encoders,
        transform=get_transforms(train=False),
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model = MultiTaskEncoder(
        backbone_name=backbone,
        key_mode=key_mode,
        pretrained=True,
        freeze_backbone=True,  # Start frozen
    ).to(device)

    params = model.get_num_params()
    print(f"Model params: {params['total']:,} ({params['trainable']:,} trainable)")

    # Loss
    criterion = MultiTaskLoss(keys=all_keys, label_smoothing=0.1)

    # Optimizer (only heads initially)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{backbone}_{key_mode}_{timestamp}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "config": {
            "backbone": backbone,
            "key_mode": key_mode,
            "keys": all_keys,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_split": val_split,
            "seed": seed,
            "freeze_backbone_epochs": freeze_backbone_epochs,
        },
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "per_key_acc": [],
    }

    best_val_acc = -1.0  # Start at -1 so we save first model

    print(f"\nTraining for {epochs} epochs...")
    print(f"Backbone frozen for first {freeze_backbone_epochs} epochs")

    for epoch in range(epochs):
        # Unfreeze backbone after warmup
        if epoch == freeze_backbone_epochs:
            print(f"\nUnfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Re-create optimizer with all params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr / 10,  # Lower LR for finetuning
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=lr / 1000
            )

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_metrics["loss"])

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["mean_acc"])
        history["per_key_acc"].append(val_metrics["per_key_acc"])

        scheduler.step()

        # Log
        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['mean_acc']:.3f}"
        )

        # Save best
        if val_metrics["mean_acc"] > best_val_acc:
            best_val_acc = val_metrics["mean_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": history["config"],
            }, run_dir / "best_model.pt")
            print(f"  -> New best model (acc: {best_val_acc:.3f})")

    # Save final model
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "config": history["config"],
    }, run_dir / "final_model.pt")

    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save encoders
    with open(run_dir / "encoders.json", "w") as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()} for k, v in encoders.items()}, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {run_dir}")

    return run_dir


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-task encoder")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Backbone model")
    parser.add_argument("--key-mode", choices=["all", "top"], default="all", help="Keys to train on")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Epochs to freeze backbone")

    args = parser.parse_args()

    train(
        backbone=args.backbone,
        key_mode=args.key_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        freeze_backbone_epochs=args.freeze_epochs,
    )
