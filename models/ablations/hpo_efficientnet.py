
import os, sys, time, json, gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import optuna
import argparse
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from config import DEVICE, NUM_CLASSES, NUM_INPUT_CHANNELS, NODATA_INDEX
from dataset import create_dataloaders
from metrics import SegmentationMetrics
from train import train_one_epoch, validate, compute_class_weights
from architectures import build_unet

def objective(trial, orig_train_loader, orig_val_loader, criterion, epochs):
    lr = trial.suggest_float('learning_rate', 1e-05, 0.01, log=True)
    wd = trial.suggest_float('weight_decay', 1e-06, 0.01, log=True)
    bs = trial.suggest_categorical('batch_size', [64, 128, 192, 224])
    dropout = trial.suggest_float('dropout_rate', 0.0, 0.5)
    train_loader = DataLoader(orig_train_loader.dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(orig_val_loader.dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    model = build_unet('efficientnet-b0', NUM_INPUT_CHANNELS, NUM_CLASSES, dropout_rate=dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-06)
    best_miou = 0.0
    val_metrics = SegmentationMetrics()
    train_metrics = SegmentationMetrics()
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, train_metrics)
        validate(model, val_loader, criterion, DEVICE, val_metrics)
        val_miou = val_metrics.mean_iou()
        scheduler.step()
        best_miou = max(best_miou, val_miou)
        trial.report(val_miou, epoch)
        if trial.should_prune():
            del model, optimizer, scheduler, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return best_miou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f'''Device: {DEVICE} | Trials: {args.n_trials} | Epochs/trial: {args.epochs}
''')
    (orig_train_loader, orig_val_loader, _) = create_dataloaders()
    class_weights = compute_class_weights(orig_train_loader, NUM_CLASSES, NODATA_INDEX, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=NODATA_INDEX)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize((lambda trial: objective(trial, orig_train_loader, orig_val_loader, criterion, args.epochs)), n_trials=args.n_trials)
    pruned = [t for t in study.trials if (t.state == optuna.trial.TrialState.PRUNED)]
    complete = [t for t in study.trials if (t.state == optuna.trial.TrialState.COMPLETE)]
    print(f'''
Trials: {len(study.trials)} total, {len(complete)} complete, {len(pruned)} pruned''')
    best = study.best_trial
    print(f'Best mIoU: {best.value:.4f}')
    for (k, v) in best.params.items():
        print(f'  {k}: {v}')
    with open(os.path.join(log_dir, 'best_hpo_params.json'), 'w') as f:
        json.dump({'best_value': best.value, 'best_params': best.params}, f, indent=4)
if (__name__ == '__main__'):
    main()
