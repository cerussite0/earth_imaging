
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from config import DEVICE, NUM_INPUT_CHANNELS, NUM_CLASSES, CHECKPOINT_DIR, OUTPUT_DIR, ESRI_CLASSES, INDEX_TO_CLASS, CLASS_NAMES
from dataset import create_dataloaders
from model import LULCSegmentationNet
from metrics import SegmentationMetrics

def stretch_percentile(ch, low=2.0, high=98.0):
    ch = ch.astype(np.float32)
    (p_lo, p_hi) = np.percentile(ch, [low, high])
    if (p_hi <= p_lo):
        return np.zeros_like(ch)
    return np.clip(((ch - p_lo) / (p_hi - p_lo)), 0, 1)

def main():
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if (not os.path.exists(ckpt_path)):
        print(f'Checkpoint not found: {ckpt_path}')
        return
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = LULCSegmentationNet(NUM_INPUT_CHANNELS, NUM_CLASSES)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    (_, val_loader, _) = create_dataloaders()
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    images_to_plot = []
    with torch.no_grad():
        for (i, (x_batch, y_batch)) in enumerate(val_loader):
            (x_batch, y_batch) = (x_batch.to(DEVICE), y_batch.to(DEVICE))
            logits = model(x_batch)
            preds = logits.argmax(dim=1)
            metrics.update(preds, y_batch)
            if (i == 0):
                for j in range(min(5, x_batch.shape[0])):
                    images_to_plot.append((x_batch[j].cpu().numpy(), y_batch[j].cpu().numpy(), preds[j].cpu().numpy()))
    print(metrics.summary())
    max_class = max(ESRI_CLASSES.keys())
    color_list = (['#000000'] * (max_class + 1))
    for (v, (_, hex_c)) in ESRI_CLASSES.items():
        color_list[v] = hex_c
    label_cmap = ListedColormap(color_list)
    label_norm = BoundaryNorm(np.arange((- 0.5), (max_class + 1.5), 1), label_cmap.N)
    legend_patches = [mpatches.Patch(color=hc, label=f'{v}: {n}') for (v, (n, hc)) in sorted(ESRI_CLASSES.items()) if (v != 0)]
    (fig, axes) = plt.subplots(len(images_to_plot), 3, figsize=(15, (5 * len(images_to_plot))))
    if (len(images_to_plot) == 1):
        axes = [axes]
    for (i, (x_np, y_np, p_np)) in enumerate(images_to_plot):
        y_raw = np.zeros_like(y_np)
        p_raw = np.zeros_like(p_np)
        for (ci, raw_id) in INDEX_TO_CLASS.items():
            y_raw[y_np == ci] = raw_id
            p_raw[p_np == ci] = raw_id
        rgb = np.dstack([stretch_percentile(x_np[2]), stretch_percentile(x_np[1]), stretch_percentile(x_np[0])])
        axes[i][0].imshow(rgb)
        axes[i][0].set_title(f'Sample {(i + 1)}: RGB')
        axes[i][0].axis('off')
        axes[i][1].imshow(y_raw, cmap=label_cmap, norm=label_norm, interpolation='nearest')
        axes[i][1].set_title('Ground Truth')
        axes[i][1].axis('off')
        axes[i][2].imshow(p_raw, cmap=label_cmap, norm=label_norm, interpolation='nearest')
        axes[i][2].set_title('Prediction')
        axes[i][2].axis('off')
    axes[0][2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title='ESRI Classes')
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis_path = os.path.join(OUTPUT_DIR, 'validation_vis.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {vis_path}')
if (__name__ == '__main__'):
    main()
