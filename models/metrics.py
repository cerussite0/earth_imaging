
import torch
import numpy as np
from config import NUM_CLASSES, CLASS_NAMES, NODATA_INDEX

class SegmentationMetrics():

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=NODATA_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, preds, targets):
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        if (self.ignore_index is not None):
            mask = (targets != self.ignore_index)
            preds = preds[mask]
            targets = targets[mask]
        for (t, p) in zip(targets, preds):
            if ((0 <= t < self.num_classes) and (0 <= p < self.num_classes)):
                self.confusion_matrix[t, p] += 1

    def overall_accuracy(self):
        total = self.confusion_matrix.sum()
        if (total == 0):
            return 0.0
        return (np.diag(self.confusion_matrix).sum() / total)

    def per_class_iou(self):
        ious = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = (self.confusion_matrix[:, c].sum() - tp)
            fn = (self.confusion_matrix[c, :].sum() - tp)
            denom = ((tp + fp) + fn)
            ious[c] = ((tp / denom) if (denom > 0) else float('nan'))
        return ious

    def mean_iou(self):
        ious = self.per_class_iou()
        valid = (~ np.isnan(ious))
        if (self.ignore_index is not None):
            valid[self.ignore_index] = False
        return (np.nanmean(ious[valid]) if valid.any() else 0.0)

    def per_class_f1(self):
        f1s = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = (self.confusion_matrix[:, c].sum() - tp)
            fn = (self.confusion_matrix[c, :].sum() - tp)
            prec = ((tp / (tp + fp)) if ((tp + fp) > 0) else 0.0)
            rec = ((tp / (tp + fn)) if ((tp + fn) > 0) else 0.0)
            f1s[c] = ((((2 * prec) * rec) / (prec + rec)) if ((prec + rec) > 0) else 0.0)
        return f1s

    def mean_f1(self):
        f1s = self.per_class_f1()
        valid = (f1s > 0)
        if (self.ignore_index is not None):
            valid[self.ignore_index] = False
        return (f1s[valid].mean() if valid.any() else 0.0)

    def summary(self):
        oa = self.overall_accuracy()
        miou = self.mean_iou()
        mf1 = self.mean_f1()
        ious = self.per_class_iou()
        f1s = self.per_class_f1()
        lines = [f'  Overall Accuracy: {oa:.4f}', f'  Mean IoU:         {miou:.4f}', f'  Mean F1:          {mf1:.4f}', '', f"  {'Class':<25s}  {'IoU':>8s}  {'F1':>8s}  {'Support':>10s}", f"  {('-' * 60)}"]
        for c in range(self.num_classes):
            support = self.confusion_matrix[c, :].sum()
            iou_str = (f'{ious[c]:.4f}' if (not np.isnan(ious[c])) else '  N/A  ')
            name = (CLASS_NAMES[c] if (c < len(CLASS_NAMES)) else f'Class {c}')
            marker = (' *' if (c == self.ignore_index) else '')
            lines.append(f'  {(name + marker):<25s}  {iou_str:>8s}  {f1s[c]:.4f:>8s}  {support:>10d}')
        if (self.ignore_index is not None):
            lines.append(f'''
  * = ignored in mIoU/mF1''')
        return '\n'.join(lines)
