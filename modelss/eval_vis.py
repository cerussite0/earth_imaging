#!/usr/bin/env python3
"""
Evaluation and Visualization Script

Computes metrics on the validation set using the best trained model,
and generates side-by-side visualizations: [Input RGB] | [Target] | [Prediction].
Saves the result to an image.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from config import (
    DEVICE, NUM_INPUT_CHANNELS, NUM_CLASSES,
    CHECKPOINT_DIR, OUTPUT_DIR, ESRI_CLASSES, INDEX_TO_CLASS, CLASS_NAMES
)
from dataset import create_dataloaders
from model import LULCSegmentationNet
from metrics import SegmentationMetrics

def stretch_percentile(channel: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Enhance image contrast using percentile stretching for visualization."""
    channel = channel.astype(np.float32)
    p_low, p_high = np.percentile(channel, [low, high])
    if p_high <= p_low:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - p_low) / (p_high - p_low), 0, 1)

def main():
    print("Loading best model checkpoint...")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}. Please train the model first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    model = LULCSegmentationNet(in_channels=NUM_INPUT_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    dataset_info = ckpt.get("dataset_info", {})
    channel_means = dataset_info.get("channel_means")
    channel_stds = dataset_info.get("channel_stds")
    
    print("\nLoading validation dataset...")
    _, val_loader, _ = create_dataloaders()
    
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    print("\nEvaluating model over validation set...")
    images_to_plot = []
    
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            logits = model(x_batch)
            preds = logits.argmax(dim=1)
            
            metrics.update(preds, y_batch)
            
            # Save some samples from the first batch for visualization
            if i == 0:
                num_samples = min(5, x_batch.shape[0])
                for j in range(num_samples):
                    x_cpu = x_batch[j].cpu()
                    # Un-normalize back to raw values for proper RGB display
                    if channel_means is not None and channel_stds is not None:
                        x_raw = x_cpu * channel_stds.view(-1, 1, 1).cpu() + channel_means.view(-1, 1, 1).cpu()
                    else:
                        x_raw = x_cpu
                    
                    images_to_plot.append((
                        x_raw.numpy(), 
                        y_batch[j].cpu().numpy(), 
                        preds[j].cpu().numpy()
                    ))
                    
    print("\n" + "="*60)
    print("  VALIDATION METRICS")
    print("="*60)
    print(metrics.summary())
    
    print("\nGenerating visualizations...")
    
    # Configure the colormap from config.py definitions
    max_class = max(ESRI_CLASSES.keys())
    color_list = ["#000000"] * (max_class + 1)
    for class_val, (_, hex_color) in ESRI_CLASSES.items():
        color_list[class_val] = hex_color

    label_cmap = ListedColormap(color_list)
    label_norm = BoundaryNorm(np.arange(-0.5, max_class + 1.5, 1), label_cmap.N)
    legend_patches = [
        mpatches.Patch(color=hex_color, label=f"{class_val}: {name}")
        for class_val, (name, hex_color) in sorted(ESRI_CLASSES.items())
        if class_val != 0
    ]

    # Create figure plot
    fig, axes = plt.subplots(len(images_to_plot), 3, figsize=(15, 5 * len(images_to_plot)))
    if len(images_to_plot) == 1:
        axes = [axes]
        
    for i, (x_np, y_np, p_np) in enumerate(images_to_plot):
        # We need to map y_np and p_np back from contiguous IDs to raw ESRI classes
        y_raw = np.zeros_like(y_np)
        p_raw = np.zeros_like(p_np)
        for class_idx, raw_id in INDEX_TO_CLASS.items():
            y_raw[y_np == class_idx] = raw_id
            p_raw[p_np == class_idx] = raw_id
        
        # B4, B3, B2 correspond to channels 2, 1, 0 respectively (Landsat indices)
        rgb = np.dstack([
            stretch_percentile(x_np[2]),
            stretch_percentile(x_np[1]),
            stretch_percentile(x_np[0]),
        ])
        
        axes[i][0].imshow(rgb)
        axes[i][0].set_title(f"Sample {i+1}: Input RGB")
        axes[i][0].axis("off")
        
        axes[i][1].imshow(y_raw, cmap=label_cmap, norm=label_norm, interpolation='nearest')
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")
        
        axes[i][2].imshow(p_raw, cmap=label_cmap, norm=label_norm, interpolation='nearest')
        axes[i][2].set_title("Prediction")
        axes[i][2].axis("off")
        
    # Place legend on the right side of the first sequence
    axes[0][2].legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        title="ESRI Classes",
    )
    
    plt.tight_layout()
    vis_path = os.path.join(OUTPUT_DIR, "validation_vis.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved successfully to: {vis_path}")

if __name__ == "__main__":
    main()
