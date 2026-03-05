"""
Visualize Training Results

Creates plots for:
- Training/validation loss curves
- Per-key accuracy over epochs
- Model comparison
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = Path(__file__).parents[3] / "data" / "EMB-001" / "finetuned_models"
OUTPUT_DIR = Path(__file__).parent / "figures"


def load_history(model_dir: Path) -> dict:
    """Load training history from model directory."""
    history_path = model_dir / "history.json"
    with open(history_path) as f:
        return json.load(f)


def plot_loss_curves(histories: dict[str, dict], save_path: Path | None = None):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, hist), color in zip(histories.items(), colors):
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Training loss
        axes[0].plot(epochs, hist["train_loss"], label=name, color=color, linewidth=2)

        # Validation loss
        axes[1].plot(epochs, hist["val_loss"], label=name, color=color, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_accuracy_curves(histories: dict[str, dict], save_path: Path | None = None):
    """Plot validation accuracy curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, hist), color in zip(histories.items(), colors):
        epochs = range(1, len(hist["val_acc"]) + 1)
        ax.plot(epochs, hist["val_acc"], label=name, color=color, linewidth=2, marker='o', markersize=3)

        # Mark best accuracy
        best_idx = np.argmax(hist["val_acc"])
        best_acc = hist["val_acc"][best_idx]
        ax.scatter([best_idx + 1], [best_acc], color=color, s=100, zorder=5, edgecolors='black')
        ax.annotate(f"{best_acc:.1%}", (best_idx + 1, best_acc), textcoords="offset points",
                   xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy Over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_per_key_accuracy(history: dict, model_name: str, save_path: Path | None = None):
    """Plot per-key accuracy heatmap over epochs."""
    per_key_acc = history.get("per_key_acc", [])
    if not per_key_acc:
        print("No per-key accuracy data available")
        return

    # Get all keys
    keys = list(per_key_acc[0].keys())
    epochs = len(per_key_acc)

    # Build matrix
    matrix = np.zeros((len(keys), epochs))
    for e, epoch_acc in enumerate(per_key_acc):
        for k, key in enumerate(keys):
            matrix[k, e] = epoch_acc.get(key, 0)

    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Semantic Key")
    ax.set_title(f"Per-Key Validation Accuracy: {model_name}")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    # Add final accuracy text
    for k, key in enumerate(keys):
        final_acc = matrix[k, -1]
        ax.text(epochs - 1 + 0.5, k, f"{final_acc:.0%}", va="center", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def plot_comparison_bar(histories: dict[str, dict], save_path: Path | None = None):
    """Bar chart comparing final accuracies."""
    models = list(histories.keys())
    best_accs = [max(h["val_acc"]) for h in histories.values()]
    final_accs = [h["val_acc"][-1] for h in histories.values()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, best_accs, width, label="Best", color="steelblue")
    bars2 = ax.bar(x + width/2, final_accs, width, label="Final", color="lightsteelblue")

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison: Best vs Final Validation Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1%}", xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--models", nargs="+", help="Model directory names to visualize")
    parser.add_argument("--save", action="store_true", help="Save figures")
    parser.add_argument("--all", action="store_true", help="Visualize all available models")
    args = parser.parse_args()

    # Find models
    if args.all:
        model_dirs = [d for d in MODEL_DIR.iterdir() if d.is_dir() and (d / "history.json").exists()]
    elif args.models:
        model_dirs = [MODEL_DIR / m for m in args.models]
    else:
        # Default: latest models
        model_dirs = sorted(
            [d for d in MODEL_DIR.iterdir() if d.is_dir() and (d / "history.json").exists()],
            key=lambda x: x.name,
            reverse=True
        )[:4]

    if not model_dirs:
        print("No models found!")
        return

    print(f"Visualizing {len(model_dirs)} models:")
    for d in model_dirs:
        print(f"  - {d.name}")

    # Load histories
    histories = {}
    for model_dir in model_dirs:
        try:
            hist = load_history(model_dir)
            # Create short name
            config = hist.get("config", {})
            name = f"{config.get('backbone', 'unknown')}_{config.get('key_mode', 'unknown')}"
            histories[name] = hist
        except Exception as e:
            print(f"Error loading {model_dir}: {e}")

    if not histories:
        print("No histories loaded!")
        return

    # Create output dir
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)

    # Plot
    print("\nPlotting loss curves...")
    plot_loss_curves(
        histories,
        OUTPUT_DIR / "loss_curves.png" if args.save else None
    )

    print("\nPlotting accuracy curves...")
    plot_accuracy_curves(
        histories,
        OUTPUT_DIR / "accuracy_curves.png" if args.save else None
    )

    print("\nPlotting comparison...")
    plot_comparison_bar(
        histories,
        OUTPUT_DIR / "comparison.png" if args.save else None
    )

    # Per-key accuracy for each model
    for name, hist in histories.items():
        print(f"\nPlotting per-key accuracy for {name}...")
        plot_per_key_accuracy(
            hist, name,
            OUTPUT_DIR / f"per_key_{name}.png" if args.save else None
        )


if __name__ == "__main__":
    main()
