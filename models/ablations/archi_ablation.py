
import os, sys, time, argparse, json, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from architectures import build_unet
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from config import DEVICE, NUM_CLASSES, NUM_INPUT_CHANNELS, LEARNING_RATE, WEIGHT_DECAY, NODATA_INDEX, CLASS_NAMES
from dataset import create_dataloaders
from metrics import SegmentationMetrics
from train import train_one_epoch, validate, compute_class_weights

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, ignore_index=(- 1)):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp((- ce))
        focal = ((self.alpha[targets] * ((1 - pt) ** self.gamma)) * ce)
        return focal.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'''Device: {DEVICE} | Epochs: {args.epochs}
''')
    (orig_train_loader, orig_val_loader, _) = create_dataloaders()
    class_weights = compute_class_weights(orig_train_loader, NUM_CLASSES, NODATA_INDEX, DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=NODATA_INDEX).to(DEVICE)
    batch_sizes = {'resnet18': 256, 'efficientnet-b0': 224, 'mobilenet_v2': 256}
    architectures = ['resnet18', 'efficientnet-b0', 'mobilenet_v2']
    for arch in architectures:
        bs = batch_sizes[arch]
        train_loader = DataLoader(orig_train_loader.dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(orig_val_loader.dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        print(f'''
--- Training: {arch} ---''')
        model = build_unet(arch, NUM_INPUT_CHANNELS, NUM_CLASSES, dropout_rate=0.2).to(DEVICE)
        n_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
        print(f'Params: {n_params:,}')
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-06)
        train_metrics = SegmentationMetrics()
        val_metrics = SegmentationMetrics()
        history = []
        best_miou = 0.0
        for epoch in range(args.epochs):
            t0 = time.time()
            lr = optimizer.param_groups[0]['lr']
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, train_metrics)
            val_loss = validate(model, val_loader, criterion, DEVICE, val_metrics)
            val_miou = val_metrics.mean_iou()
            scheduler.step()
            elapsed = (time.time() - t0)
            history.append({'epoch': epoch, 'train_loss': train_loss, 'train_oa': train_metrics.overall_accuracy(), 'train_miou': train_metrics.mean_iou(), 'val_loss': val_loss, 'val_oa': val_metrics.overall_accuracy(), 'val_miou': val_miou, 'val_mf1': val_metrics.mean_f1(), 'lr': lr, 'time': elapsed})
            print(f'Ep {(epoch + 1):3d}/{args.epochs} | TrL:{train_loss:.4f} mIoU:{train_metrics.mean_iou():.4f} | VL:{val_loss:.4f} mIoU:{val_miou:.4f} | {elapsed:.1f}s')
            if (val_miou > best_miou):
                best_miou = val_miou
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{arch}_best.pth'))
        with open(os.path.join(log_dir, f'{arch}_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        print(f'''Done {arch}. Best mIoU: {best_miou:.4f}
''')
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
if (__name__ == '__main__'):
    main()
