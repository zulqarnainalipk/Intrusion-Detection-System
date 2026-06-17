"""
Evaluation visualization utilities for IDS Framework
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from typing import Dict, Any
import os

from ..utils.logger import Logger


class EvaluationVisualizer:
    """Visualization utilities for model evaluation"""
    
    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        self.plot_dpi = config.get("evaluation.plot_dpi", 300)
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> None:
        """Create confusion matrix visualization"""
        cm_raw = confusion_matrix(y_true, y_pred)
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw confusion matrix
        sns.heatmap(
            cm_raw, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"],
            ax=axes[0],
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 16, "fontweight": "bold"}
        )
        axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Predicted Label", fontsize=11)
        axes[0].set_ylabel("True Label", fontsize=11)
        
        # Normalized confusion matrix
        sns.heatmap(
            cm_norm, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"],
            ax=axes[1],
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 16, "fontweight": "bold"},
            vmin=0, vmax=1
        )
        axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Predicted Label", fontsize=11)
        axes[1].set_ylabel("True Label", fontsize=11)
        
        plt.suptitle("Confusion Matrix Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Confusion matrix saved: {path}")
    
    def create_roc_curve(self, y_true: np.ndarray, y_pred_prob: np.ndarray, output_dir: str) -> None:
        """Create ROC curve visualization"""
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_pred_prob)
        auc_roc = auc(fpr_arr, tpr_arr)
        
        # Calculate operating point
        threshold = self.config.get("evaluation.threshold", 0.5)
        y_pred = (y_pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr_metric = fp / (fp + tn) if (fp + tn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fig, ax = plt.subplots(figsize=(9, 7))
        
        ax.plot(fpr_arr, tpr_arr, color="royalblue", linewidth=2.5, 
               label=f"LSTM Model (AUC = {auc_roc:.4f})")
        ax.fill_between(fpr_arr, tpr_arr, alpha=0.12, color="royalblue")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier (AUC = 0.5)")
        ax.plot([0, 0, 1], [0, 1, 1], "g:", linewidth=1.5, label="Perfect Classifier (AUC = 1.0)")
        
        # Operating point
        ax.scatter(fpr_metric, recall, color="red", s=100, zorder=5, label="Operating Point (0.5)")
        ax.annotate(
            f"  Op. Point\n  FPR={fpr_metric:.3f}\n  TPR={recall:.3f}",
            xy=(fpr_metric, recall), fontsize=9, color="red"
        )
        
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title("ROC Curve — Intrusion Detection", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.4)
        
        plt.tight_layout()
        path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"ROC curve saved: {path}")
    
    def create_precision_recall_curve(self, y_true: np.ndarray, y_pred_prob: np.ndarray, output_dir: str) -> None:
        """Create precision-recall curve visualization"""
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_pred_prob)
        baseline = y_true.mean()
        auc_pr = auc(rec_arr, prec_arr)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # PR Curve
        axes[0].plot(rec_arr, prec_arr, color="darkorange", linewidth=2.5,
                    label=f"LSTM (AUC-PR = {auc_pr:.4f})")
        axes[0].fill_between(rec_arr, prec_arr, alpha=0.12, color="darkorange")
        axes[0].axhline(y=baseline, color="navy", linestyle="--", linewidth=1.5,
                       label=f"Baseline (attack rate = {baseline:.3f})")
        axes[0].set_xlabel("Recall", fontsize=12)
        axes[0].set_ylabel("Precision", fontsize=12)
        axes[0].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1.02])
        axes[0].grid(True, alpha=0.4)
        
        # Threshold analysis
        thresh_plot = thresh_arr[::max(1, len(thresh_arr)//200)]
        prec_plot = prec_arr[::max(1, len(thresh_arr)//200)]
        rec_plot = rec_arr[::max(1, len(thresh_arr)//200)]
        
        axes[1].plot(thresh_plot, prec_plot[:-1] if len(prec_plot) > len(thresh_plot) else prec_plot,
                    color="steelblue", linewidth=2, label="Precision")
        axes[1].plot(thresh_plot, rec_plot[:-1] if len(rec_plot) > len(thresh_plot) else rec_plot,
                    color="crimson", linewidth=2, label="Recall")
        axes[1].axvline(x=0.5, color="black", linestyle=":", linewidth=1.5, label="Default Threshold (0.5)")
        axes[1].set_xlabel("Classification Threshold", fontsize=12)
        axes[1].set_ylabel("Score", fontsize=12)
        axes[1].set_title("Precision & Recall vs Threshold", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=11)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.02])
        axes[1].grid(True, alpha=0.4)
        
        plt.suptitle("Precision-Recall Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "precision_recall_curve.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Precision-recall curve saved: {path}")
    
    def create_prediction_distribution(self, y_true: np.ndarray, y_pred_prob: np.ndarray, output_dir: str) -> None:
        """Create prediction probability distribution visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # By class
        for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Attack")]:
            mask = y_true == label
            axes[0].hist(
                y_pred_prob[mask], bins=60, alpha=0.65, density=True,
                color=color, label=f"{name} (n={mask.sum():,})", edgecolor="white"
            )
        
        axes[0].axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Boundary")
        axes[0].set_xlabel("Predicted Probability (Attack)", fontsize=11)
        axes[0].set_ylabel("Density", fontsize=11)
        axes[0].set_title("Prediction Probability Distribution by True Class", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=10)
        
        # Overall distribution
        axes[1].hist(y_pred_prob, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
        axes[1].axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.5")
        axes[1].set_xlabel("Predicted Probability", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title("Overall Prediction Probability Histogram", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=10)
        
        plt.suptitle("Model Confidence Analysis", fontsize=15, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "prediction_probability_distribution.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Prediction distribution saved: {path}")
    
    def create_metrics_dashboard(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Create comprehensive metrics dashboard"""
        # Extract and normalize metrics
        score_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "AUC-ROC", "AUC-PR", "MCC*"]
        score_values = [
            metrics["Accuracy"], metrics["Precision"], metrics["Recall (TPR)"], 
            metrics["Specificity (TNR)"], metrics["F1 Score"], metrics["AUC-ROC"], 
            metrics["AUC-PR"], (metrics["MCC"] + 1) / 2  # Normalize MCC to [0,1]
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Bar chart of metrics
        colors_bar = sns.color_palette("husl", len(score_labels))
        bars = axes[0].bar(score_labels, score_values, color=colors_bar, edgecolor="white", linewidth=0.6)
        axes[0].set_ylim(0, 1.08)
        axes[0].set_title("Model Performance Metrics Summary\n(*MCC normalized to [0,1])",
                         fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Score", fontsize=11)
        axes[0].tick_params(axis="x", rotation=30)
        
        for bar, val in zip(bars, score_values):
            color = "#2ecc71" if val >= 0.9 else "#f39c12" if val >= 0.75 else "#e74c3c"
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=color)
        
        # Confusion matrix details table
        cm_details = {
            "TP (Attack correctly detected)": f"{metrics['True Positives']:,}",
            "TN (Normal correctly classified)": f"{metrics['True Negatives']:,}",
            "FP (Normal misclassified as Attack)": f"{metrics['False Positives']:,}",
            "FN (Attack missed)": f"{metrics['False Negatives']:,}",
            "Total Test Samples": f"{metrics['Total Samples']:,}",
            "Detection Rate (Recall)": f"{metrics['Recall (TPR)']:.4f}",
            "False Alarm Rate (FPR)": f"{metrics['False Positive Rate']:.4f}",
        }
        
        axes[1].axis("off")
        table_data = [[k, v] for k, v in cm_details.items()]
        table = axes[1].table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.5, 2.2)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#ecf0f1")
        
        axes[1].set_title("Detailed Classification Statistics", fontsize=13, fontweight="bold", pad=20)
        
        plt.suptitle("Complete Evaluation Report", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        path = os.path.join(output_dir, "metrics_dashboard.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Metrics dashboard saved: {path}")
    
    def create_threshold_analysis(self, threshold_df: pd.DataFrame, output_dir: str) -> None:
        """Create threshold analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Metrics vs Threshold
        axes[0, 0].plot(threshold_df["threshold"], threshold_df["accuracy"], 
                       "b-", label="Accuracy", linewidth=2)
        axes[0, 0].plot(threshold_df["threshold"], threshold_df["precision"], 
                       "r-", label="Precision", linewidth=2)
        axes[0, 0].plot(threshold_df["threshold"], threshold_df["recall"], 
                       "g-", label="Recall", linewidth=2)
        axes[0, 0].plot(threshold_df["threshold"], threshold_df["f1_score"], 
                       "orange", label="F1 Score", linewidth=2)
        axes[0, 0].set_xlabel("Threshold")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Metrics vs Classification Threshold")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Trade-off
        axes[0, 1].plot(threshold_df["threshold"], threshold_df["precision"], 
                       "r-", label="Precision", linewidth=2)
        axes[0, 1].plot(threshold_df["threshold"], threshold_df["recall"], 
                       "g-", label="Recall", linewidth=2)
        axes[0, 1].set_xlabel("Threshold")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Precision-Recall Trade-off")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score focus
        best_idx = threshold_df["f1_score"].idxmax()
        axes[1, 0].plot(threshold_df["threshold"], threshold_df["f1_score"], 
                       "orange", linewidth=2)
        axes[1, 0].scatter(threshold_df.loc[best_idx, "threshold"], 
                          threshold_df.loc[best_idx, "f1_score"], 
                          color="red", s=100, zorder=5)
        axes[1, 0].annotate(f"Best: {threshold_df.loc[best_idx, 'threshold']:.2f}",
                          xy=(threshold_df.loc[best_idx, "threshold"], 
                              threshold_df.loc[best_idx, "f1_score"]),
                          xytext=(10, 10), textcoords="offset points")
        axes[1, 0].set_xlabel("Threshold")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].set_title("F1 Score vs Threshold")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Threshold comparison table
        axes[1, 1].axis("off")
        table_data = threshold_df.round(4).values.tolist()
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=threshold_df.columns,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#34495e")
                cell.set_text_props(color="white", fontweight="bold")
        
        axes[1, 1].set_title("Threshold Analysis Table", fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, "threshold_analysis.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Threshold analysis saved: {path}")
    
    def create_feature_importance_plot(self, importance_df: pd.DataFrame, output_dir: str, top_n: int = 20) -> None:
        """Create feature importance visualization"""
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
        bars = ax.barh(
            range(top_n),
            top_features["importance_mean"],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            xerr=top_features["importance_std"],
            capsize=3
        )
        
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(
            [f.replace("_", " ").title() for f in top_features["feature"]],
            fontsize=9
        )
        ax.invert_yaxis()
        ax.set_xlabel("Permutation Importance Score", fontsize=11)
        ax.set_title(f"Top {top_n} Feature Importances (Permutation Importance)",
                    fontsize=14, fontweight="bold", pad=12)
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Feature importance plot saved: {path}")
    
    def create_error_analysis_plots(self, error_df: pd.DataFrame, output_dir: str) -> None:
        """Create error analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confidence distribution by error type
        for error_type in error_df["error_type"].unique():
            subset = error_df[error_df["error_type"] == error_type]
            axes[0, 0].hist(subset["confidence"], bins=30, alpha=0.6, 
                           label=error_type, density=True)
        
        axes[0, 0].set_xlabel("Prediction Confidence")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Confidence Distribution by Prediction Type")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot of confidence by error type
        error_types = []
        confidences = []
        for error_type in error_df["error_type"].unique():
            subset = error_df[error_df["error_type"] == error_type]
            error_types.extend([error_type] * len(subset))
            confidences.extend(subset["confidence"].tolist())
        
        axes[0, 1].boxplot([error_df[error_df["error_type"] == et]["confidence"].tolist() 
                           for et in error_df["error_type"].unique()],
                          labels=error_df["error_type"].unique())
        axes[0, 1].set_ylabel("Prediction Confidence")
        axes[0, 1].set_title("Confidence by Prediction Type")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error type counts
        error_counts = error_df["error_type"].value_counts()
        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
        axes[1, 0].pie(error_counts.values, labels=error_counts.index, autopct="%1.1f%%",
                      colors=colors[:len(error_counts)])
        axes[1, 0].set_title("Prediction Type Distribution")
        
        # Confidence threshold analysis
        thresholds = np.arange(0.1, 1.0, 0.1)
        accuracy_by_threshold = []
        for threshold in thresholds:
            high_conf = error_df[error_df["confidence"] >= threshold]
            if len(high_conf) > 0:
                correct = len(high_conf[high_conf["error_type"].isin(["True Positive", "True Negative"])])
                accuracy_by_threshold.append(correct / len(high_conf))
            else:
                accuracy_by_threshold.append(0)
        
        axes[1, 1].plot(thresholds, accuracy_by_threshold, "bo-", linewidth=2, markersize=8)
        axes[1, 1].set_xlabel("Confidence Threshold")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_title("Accuracy vs Confidence Threshold")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, "error_analysis.png")
        plt.savefig(path, bbox_inches="tight", dpi=self.plot_dpi)
        plt.show()
        self.logger.info(f"Error analysis plots saved: {path}")
