#!/usr/bin/env python3
"""
Plot Ablation Comparisons
Reads the JSON logs from archi_ablation.py and plots comparative graphs.
"""
import os
import json
import matplotlib.pyplot as plt

def main():
    ablations_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    architectures = ["resnet18", "efficientnet-b0", "mobilenet_v2"]
    colors = {"resnet18": "tomato", "efficientnet-b0": "mediumseagreen", "mobilenet_v2": "royalblue"}
    
    histories = {}
    for arch in architectures:
        log_path = os.path.join(ablations_log_dir, f"{arch}_history.json")
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found. Ensure archi_ablation.py was run for '{arch}'.")
            continue
        with open(log_path, "r") as f:
            histories[arch] = json.load(f)
            
    if not histories:
        print("No logs found to plot.")
        return
        
    # Create a 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Architecture Ablation: 100-Epoch Comparisons", fontsize=16)
    
    metrics = [
        ("val_miou", "Validation mean IoU", axes[0, 0]),
        ("val_loss", "Validation Loss", axes[0, 1]),
        ("val_oa", "Validation Overall Accuracy", axes[1, 0]),
        ("val_mf1", "Validation mean F1 Score", axes[1, 1])
    ]
    
    for metric_key, title, ax in metrics:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(metric_key, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        for arch, hist in histories.items():
            epochs = [step["epoch"] for step in hist]
            values = [step[metric_key] for step in hist]
            
            ax.plot(epochs, values, label=arch, color=colors[arch], linewidth=2)
            
        ax.legend()
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plots to: {save_path}")

if __name__ == "__main__":
    main()
