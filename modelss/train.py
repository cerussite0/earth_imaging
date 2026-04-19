#!/usr/bin/env python3
"""
Training pipeline for the LULC Segmentation Model.

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 64 --lr 5e-4
"""

import os, sys, time, argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from config import (
    DEVICE, NUM_CLASSES, NUM_INPUT_CHANNELS,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, NODATA_INDEX,
    CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR,
    LR_SCHEDULER, LR_STEP_SIZE, LR_GAMMA, CLASS_NAMES,
)
from dataset import create_dataloaders
from model import LULCSegmentationNet, count_parameters
from metrics import SegmentationMetrics


def train_one_epoch(model, loader, criterion, optimizer, device, metrics):
    model.train()
    metrics.reset()
    running_loss, num_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
        metrics.update(logits.argmax(dim=1), y)
    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    running_loss, num_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()
        num_batches += 1
        metrics.update(logits.argmax(dim=1), y)
    return running_loss / max(num_batches, 1)


def compute_class_weights(loader, num_classes, ignore_index, device):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, y in loader:
        for c in range(num_classes):
            if c != ignore_index:
                counts[c] += (y == c).sum().item()
    total = counts.sum()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        weights[c] = total / (num_classes * counts[c]) if counts[c] > 0 else 0.0
    if ignore_index is not None:
        weights[ignore_index] = 0.0
    max_w = weights.max()
    if max_w > 10:
        weights = weights * (10.0 / max_w)
    return weights.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train LULC Segmentation Model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 70)
    print("  LULC SEGMENTATION — TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Device: {DEVICE} | Epochs: {args.epochs} | BS: {args.batch_size}")
    print(f"  LR: {args.lr} | WD: {args.weight_decay} | Patience: {args.patience}\n")

    # Data
    print("── Loading Dataset ──")
    train_loader, val_loader, dataset_info = create_dataloaders()

    # Model
    print("\n── Building Model ──")
    model = LULCSegmentationNet(NUM_INPUT_CHANNELS, NUM_CLASSES).to(DEVICE)
    total_p, train_p = count_parameters(model)
    print(f"  Params: {total_p:,} total, {train_p:,} trainable\n")

    # Class weights & loss
    print("── Computing class weights ──")
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, NODATA_INDEX, DEVICE)
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {n:<25s}: {class_weights[i]:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=NODATA_INDEX)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
                 if LR_SCHEDULER == "cosine"
                 else StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA))

    start_epoch, best_miou = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"  Resumed epoch {start_epoch}, best mIoU={best_miou:.4f}")

    # Training loop
    train_m, val_m = SegmentationMetrics(), SegmentationMetrics()
    history, patience_counter = [], 0

    print("\n" + "=" * 70 + "\n  TRAINING\n" + "=" * 70)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, train_m)
        v_loss = validate(model, val_loader, criterion, DEVICE, val_m)
        scheduler.step()

        t_oa, t_miou = train_m.overall_accuracy(), train_m.mean_iou()
        v_oa, v_miou, v_mf1 = val_m.overall_accuracy(), val_m.mean_iou(), val_m.mean_f1()
        elapsed = time.time() - t0

        history.append(dict(epoch=epoch, train_loss=t_loss, train_oa=t_oa,
                            train_miou=t_miou, val_loss=v_loss, val_oa=v_oa,
                            val_miou=v_miou, val_mf1=v_mf1, lr=lr, time=elapsed))

        print(f"  Ep {epoch+1:3d}/{args.epochs} | "
              f"TrL:{t_loss:.4f} OA:{t_oa:.4f} mIoU:{t_miou:.4f} | "
              f"VL:{v_loss:.4f} OA:{v_oa:.4f} mIoU:{v_miou:.4f} mF1:{v_mf1:.4f} | "
              f"LR:{lr:.2e} | {elapsed:.1f}s")

        if v_miou > best_miou:
            best_miou = v_miou
            patience_counter = 0
            p = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            best_miou=best_miou, dataset_info=dataset_info), p)
            print(f"    ★ New best mIoU: {best_miou:.4f} — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Save last & history
    torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    best_miou=best_miou, dataset_info=dataset_info),
               os.path.join(CHECKPOINT_DIR, "last_model.pth"))
    with open(os.path.join(LOG_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final eval
    print("\n" + "=" * 70 + "\n  FINAL EVALUATION (Best Model)\n" + "=" * 70)
    best_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"),
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_m = SegmentationMetrics()
    v_loss = validate(model, val_loader, criterion, DEVICE, final_m)
    print(f"\n  Val Loss: {v_loss:.4f}")
    print(final_m.summary())
    print(f"\n  Best mIoU: {best_miou:.4f} (epoch {best_ckpt['epoch']+1})")
    print("=" * 70 + "\n  TRAINING COMPLETE\n" + "=" * 70)


if __name__ == "__main__":
    main()
