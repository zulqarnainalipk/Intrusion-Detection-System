"""
Training visualization utilities for IDS Framework
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from typing import Dict, Any
import os

from ..utils.logger import Logger


class TrainingVisualizer:
    """Visualization utilities for training progress"""
    
    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        self.plot_dpi = config.get("evaluation.plot_dpi", 300)
    
    def create_training_dashboard(self, history: Any, epoch_df: pd.DataFrame, output_dir: str) -> None:
        """Create comprehensive training dashboard"""
        self.logger.subsection("CREATING TRAINING DASHBOARD")
        
        epochs_run = len(history.history["loss"])
        
        fig = plt.figure(figsize=(22, 14))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
        
        metric_pairs = [
            ("accuracy", "val_accuracy", "Accuracy", "royalblue", "tomato"),
            ("loss", "val_loss", "Loss", "darkorange", "purple"),
            ("auc", "val_auc", "AUC", "seagreen", "crimson"),
            ("precision", "val_precision", "Precision", "steelblue", "darkorange"),
            ("recall", "val_recall", "Recall", "mediumvioletred", "teal"),
        ]
        
        for idx, (train_key, val_key, title, c_train, c_val) in enumerate(metric_pairs):
            row, col = divmod(idx, 3)
            ax = fig.add_subplot(gs[row, col])
            
            if train_key in history.history:
                epochs_x = range(1, epochs_run + 1)
                ax.plot(epochs_x, history.history[train_key], label=f"Train {title}",
                       color=c_train, linewidth=2.0)
                ax.plot(epochs_x, history.history[val_key], label=f"Val {title}",
                       color=c_val, linewidth=2.0, linestyle="--")
                
                # Shade gap between train/val (overfitting zone)
                ax.fill_between(
                    epochs_x,
                    history.history[train_key],
                    history.history[val_key],
                    alpha=0.12,
                    color="gray",
                    label="Train-Val Gap"
                )
                
                # Mark best epoch
                if "loss" in train_key:
                    best_ep = np.argmin(history.history[val_key]) + 1
                else:
                    best_ep = np.argmax(history.history[val_key]) + 1
                
                ax.axvline(x=best_ep, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
                ax.annotate(f"Best\nEp.{best_ep}", xy=(best_ep, ax.get_ylim()[0]),
                          fontsize=7.5, ha="center", color="black", alpha=0.7)
                
                ax.set_title(f"Model {title}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.4)
        
        # Learning Rate subplot
        ax_lr = fig.add_subplot(gs[1, 2])
        if "lr" in epoch_df.columns:
            ax_lr.plot(epoch_df["epoch"] + 1, epoch_df["lr"],
                      color="darkorange", linewidth=2.0, marker="o", markersize=4)
            ax_lr.set_yscale("log")
            ax_lr.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
            ax_lr.set_xlabel("Epoch", fontsize=10)
            ax_lr.set_ylabel("Learning Rate (log scale)", fontsize=10)
            ax_lr.grid(True, alpha=0.4)
        
        plt.suptitle("Training Dashboard — NSL-KDD LSTM Model", fontsize=18, fontweight="bold", y=1.01)
        
        path = os.path.join(output_dir, "training_dashboard.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Training dashboard saved: {path}")
    
    def create_overfitting_analysis(self, history: Any, output_dir: str) -> None:
        """Create overfitting analysis plots"""
        self.logger.subsection("CREATING OVERFITTING ANALYSIS")
        
        epochs_run = len(history.history["loss"])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Train-Val Accuracy gap
        acc_gap = np.array(history.history["accuracy"]) - np.array(history.history["val_accuracy"])
        axes[0].fill_between(
            range(1, epochs_run + 1), acc_gap, 0,
            where=(acc_gap > 0), alpha=0.5, color="#e74c3c", label="Overfit Zone (Train > Val)"
        )
        axes[0].fill_between(
            range(1, epochs_run + 1), acc_gap, 0,
            where=(acc_gap <= 0), alpha=0.5, color="#2ecc71", label="Underfit Zone (Val > Train)"
        )
        axes[0].axhline(y=0, color="black", linewidth=1.2, linestyle="--")
        axes[0].set_title("Accuracy Gap (Train − Validation)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy Gap")
        axes[0].legend(fontsize=9)
        
        # Train-Val Loss gap
        loss_gap = np.array(history.history["val_loss"]) - np.array(history.history["loss"])
        axes[1].fill_between(
            range(1, epochs_run + 1), loss_gap, 0,
            where=(loss_gap > 0), alpha=0.5, color="#e74c3c", label="Overfit Zone (Val Loss > Train)"
        )
        axes[1].fill_between(
            range(1, epochs_run + 1), loss_gap, 0,
            where=(loss_gap <= 0), alpha=0.5, color="#2ecc71", label="Underfit Zone"
        )
        axes[1].axhline(y=0, color="black", linewidth=1.2, linestyle="--")
        axes[1].set_title("Loss Gap (Validation − Training)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss Gap")
        axes[1].legend(fontsize=9)
        
        plt.suptitle("Overfitting / Underfitting Analysis", fontsize=15, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "overfitting_analysis.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Overfitting analysis saved: {path}")
    
    def create_learning_curve_analysis(self, history: Any, output_dir: str) -> None:
        """Create learning curve analysis"""
        self.logger.subsection("CREATING LEARNING CURVE ANALYSIS")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(history.history["loss"]) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history.history["loss"], "b-", label="Training Loss", linewidth=2)
        axes[0, 0].plot(epochs, history.history["val_loss"], "r-", label="Validation Loss", linewidth=2)
        axes[0, 0].set_title("Loss Curves", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history.history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
        axes[0, 1].plot(epochs, history.history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2)
        axes[0, 1].set_title("Accuracy Curves", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC curves
        axes[1, 0].plot(epochs, history.history["auc"], "b-", label="Training AUC", linewidth=2)
        axes[1, 0].plot(epochs, history.history["val_auc"], "r-", label="Validation AUC", linewidth=2)
        axes[1, 0].set_title("AUC Curves", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUC")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall curves
        axes[1, 1].plot(epochs, history.history["precision"], "g-", label="Training Precision", linewidth=2)
        axes[1, 1].plot(epochs, history.history["val_precision"], "r-", label="Validation Precision", linewidth=2)
        axes[1, 1].plot(epochs, history.history["recall"], "b-", label="Training Recall", linewidth=2)
        axes[1, 1].plot(epochs, history.history["val_recall"], "orange", label="Validation Recall", linewidth=2)
        axes[1, 1].set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, "learning_curves.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Learning curves saved: {path}")
    
    def create_metrics_comparison(self, history: Any, output_dir: str) -> None:
        """Create metrics comparison plot"""
        self.logger.subsection("CREATING METRICS COMPARISON")
        
        epochs = range(1, len(history.history["loss"]) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot all metrics on the same scale
        metrics = ["accuracy", "val_accuracy", "auc", "val_auc", "precision", "val_precision", "recall", "val_recall"]
        colors = ["blue", "lightblue", "green", "lightgreen", "red", "lightcoral", "orange", "lightsalmon"]
        labels = ["Train Acc", "Val Acc", "Train AUC", "Val AUC", "Train Prec", "Val Prec", "Train Rec", "Val Rec"]
        
        for metric, color, label in zip(metrics, colors, labels):
            if metric in history.history:
                ax.plot(epochs, history.history[metric], color=color, label=label, linewidth=2)
        
        ax.set_title("All Metrics Comparison", fontsize=16, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim([0, 1])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, "metrics_comparison.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Metrics comparison saved: {path}")
