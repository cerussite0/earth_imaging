#!/usr/bin/env python3
"""
Architecture Ablation for LULC Segmentation
Uses segmentation_models_pytorch to compare 3 backbones on 100-epoch runs:
1) resnet18
2) efficientnet-b0
3) mobilenet_v2
"""

import os
import sys
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Add parent dir to path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import (
    DEVICE, NUM_CLASSES, NUM_INPUT_CHANNELS, LEARNING_RATE, WEIGHT_DECAY,
    NODATA_INDEX, CLASS_NAMES
)
from dataset import create_dataloaders
from metrics import SegmentationMetrics
from train import train_one_epoch, validate, compute_class_weights

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="Architecture Ablation")
    parser.add_argument("--epochs", type=int, default=25)
    # Use BS 256 optimally for 16GB GPU with 32x32 patches
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    ablations_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    ablations_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ablations_log_dir, exist_ok=True)
    os.makedirs(ablations_ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("  ARCHITECTURE ABLATION PIPELINE")
    print("=" * 60)
    print(f"  Device:       {DEVICE}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print()

    # Data
    print("── Loading Dataset ...")
    orig_train_loader, orig_val_loader, dataset_info = create_dataloaders()
    
    # Re-wrap dataloaders with our custom batch size
    train_loader = DataLoader(
        orig_train_loader.dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        orig_val_loader.dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print("\n── Computing class weights ...")
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, NODATA_INDEX, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=NODATA_INDEX)

    architectures = ["resnet18", "efficientnet-b0", "mobilenet_v2"]

    for arch in architectures:
        print("\n" + "="*60)
        print(f"  TRAINING BACKBONE: {arch}")
        print("="*60)
        
        # Build SMP Model
        model = smp.Unet(
            encoder_name=arch,
            encoder_weights="imagenet",
            in_channels=NUM_INPUT_CHANNELS,
            classes=NUM_CLASSES 
        ).to(DEVICE)
        
        print(f"  Params: {count_parameters(model):,} trainable\n")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        train_metrics = SegmentationMetrics()
        val_metrics = SegmentationMetrics()
        history = []
        
        best_miou = 0.0

        for epoch in range(args.epochs):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]

            # Train
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, train_metrics)
            train_oa = train_metrics.overall_accuracy()
            train_miou = train_metrics.mean_iou()

            # Val
            val_loss = validate(model, val_loader, criterion, DEVICE, val_metrics)
            val_oa = val_metrics.overall_accuracy()
            val_miou = val_metrics.mean_iou()
            val_mf1 = val_metrics.mean_f1()

            scheduler.step()
            elapsed = time.time() - t0

            history.append({
                "epoch": epoch, "train_loss": train_loss, "train_oa": train_oa, "train_miou": train_miou,
                "val_loss": val_loss, "val_oa": val_oa, "val_miou": val_miou, "val_mf1": val_mf1,
                "lr": lr, "time": elapsed
            })

            print(f"  Ep {epoch+1:3d}/{args.epochs} | "
                  f"TrLoss:{train_loss:.4f} OA:{train_oa:.4f} mIoU:{train_miou:.4f} | "
                  f"VaLoss:{val_loss:.4f} OA:{val_oa:.4f} mIoU:{val_miou:.4f} | {elapsed:.1f}s")
            
            if val_miou > best_miou:
                best_miou = val_miou
                ckpt_path = os.path.join(ablations_ckpt_dir, f"{arch}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                
        # Save history log after training finishes for current arch
        log_path = os.path.join(ablations_log_dir, f"{arch}_history.json")
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
            
        print(f"\n  ✓ Finished {arch}. Best mIoU: {best_miou:.4f}")
        print(f"  Logs saved to: {log_path}")

if __name__ == "__main__":
    main()
