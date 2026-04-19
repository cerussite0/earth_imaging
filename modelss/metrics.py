"""
Evaluation metrics for semantic segmentation.

Computes:
  - Overall Accuracy (OA)
  - Per-class IoU (Intersection over Union)
  - Mean IoU (mIoU)
  - Per-class F1 score
  - Mean F1
  - Confusion Matrix
"""

import torch
import numpy as np
from config import NUM_CLASSES, CLASS_NAMES, NODATA_INDEX


class SegmentationMetrics:
    """
    Accumulates a confusion matrix across batches and computes
    standard segmentation metrics.
    """

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=NODATA_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the confusion matrix with a batch of predictions and targets.

        Parameters
        ----------
        preds : Tensor of shape (B, H, W), int64 — predicted class indices
        targets : Tensor of shape (B, H, W), int64 — ground truth class indices
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Mask out ignored pixels
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        # Build confusion matrix
        for t, p in zip(targets, preds):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def overall_accuracy(self):
        """Overall pixel accuracy."""
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        correct = np.diag(self.confusion_matrix).sum()
        return correct / total

    def per_class_iou(self):
        """Per-class IoU."""
        ious = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            denom = tp + fp + fn
            ious[c] = tp / denom if denom > 0 else float('nan')
        return ious

    def mean_iou(self):
        """Mean IoU (ignoring classes with NaN)."""
        ious = self.per_class_iou()
        valid = ~np.isnan(ious)
        if self.ignore_index is not None:
            valid[self.ignore_index] = False
        return np.nanmean(ious[valid]) if valid.any() else 0.0

    def per_class_f1(self):
        """Per-class F1 score."""
        f1s = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s[c] = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        return f1s

    def mean_f1(self):
        """Mean F1 score (ignoring classes with 0 support)."""
        f1s = self.per_class_f1()
        valid = f1s > 0
        if self.ignore_index is not None:
            valid[self.ignore_index] = False
        return f1s[valid].mean() if valid.any() else 0.0

    def summary(self):
        """Return a formatted summary string of all metrics."""
        oa = self.overall_accuracy()
        miou = self.mean_iou()
        mf1 = self.mean_f1()
        ious = self.per_class_iou()
        f1s = self.per_class_f1()

        lines = [
            f"  Overall Accuracy: {oa:.4f}",
            f"  Mean IoU:         {miou:.4f}",
            f"  Mean F1:          {mf1:.4f}",
            f"",
            f"  {'Class':<25s}  {'IoU':>8s}  {'F1':>8s}  {'Support':>10s}",
            f"  {'-'*60}",
        ]

        for c in range(self.num_classes):
            support = self.confusion_matrix[c, :].sum()
            iou_str = f"{ious[c]:.4f}" if not np.isnan(ious[c]) else "  N/A  "
            f1_str = f"{f1s[c]:.4f}"
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
            marker = " *" if c == self.ignore_index else ""
            lines.append(
                f"  {name + marker:<25s}  {iou_str:>8s}  {f1_str:>8s}  {support:>10d}"
            )

        if self.ignore_index is not None:
            lines.append(f"\n  * = ignored in mIoU/mF1 computation")

        return "\n".join(lines)
