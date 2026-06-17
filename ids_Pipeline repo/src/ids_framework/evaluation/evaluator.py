"""
Model evaluation utilities for IDS Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, classification_report,
    confusion_matrix, roc_curve, auc, average_precision_score,
    precision_recall_curve
)
from typing import Dict, Any, Tuple, List
import os
import json

from ..utils.logger import Logger
from ..visualization.evaluation_plots import EvaluationVisualizer


class ModelEvaluator:
    """Comprehensive model evaluation for IDS Framework"""
    
    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        self.visualizer = EvaluationVisualizer(config, logger)
        self.threshold = config.get("evaluation.threshold", 0.5)
        
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      output_dirs: Dict[str, str]) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        self.logger.section("MODEL EVALUATION")
        
        # Get model predictions
        y_pred_prob = self._get_predictions(model, X_test)
        y_pred = self._apply_threshold(y_pred_prob, self.threshold)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_prob)
        
        # Generate classification report
        class_report = self._generate_classification_report(y_test, y_pred)
        
        # Create visualizations
        self.visualizer.create_confusion_matrix(y_test, y_pred, output_dirs["evaluation"])
        self.visualizer.create_roc_curve(y_test, y_pred_prob, output_dirs["evaluation"])
        self.visualizer.create_precision_recall_curve(y_test, y_pred_prob, output_dirs["evaluation"])
        self.visualizer.create_prediction_distribution(y_test, y_pred_prob, output_dirs["evaluation"])
        self.visualizer.create_metrics_dashboard(metrics, output_dirs["evaluation"])
        
        # Save evaluation results
        evaluation_results = {
            "metrics": metrics,
            "classification_report": class_report,
            "threshold": self.threshold,
            "predictions_saved": self.config.get("evaluation.save_predictions", True)
        }
        
        # Save predictions if configured
        if self.config.get("evaluation.save_predictions", True):
            self._save_predictions(y_test, y_pred, y_pred_prob, output_dirs["reports"])
        
        # Save metrics to CSV
        self._save_metrics(metrics, output_dirs["reports"])
        
        self.logger.evaluation_results(metrics)
        
        return evaluation_results
    
    def _get_predictions(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        self.logger.subsection("GENERATING PREDICTIONS")
        
        y_pred_prob = model.predict(X_test, batch_size=512, verbose=1).flatten()
        self.logger.info("Predictions completed")
        
        return y_pred_prob
    
    def _apply_threshold(self, y_pred_prob: np.ndarray, threshold: float) -> np.ndarray:
        """Apply classification threshold"""
        return (y_pred_prob >= threshold).astype(int)
    
    def _calculate_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, 
                          y_pred_prob: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # ROC and PR metrics
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_prob)
        auc_roc = auc(fpr_arr, tpr_arr)
        auc_pr = average_precision_score(y_test, y_pred_prob)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        fpr_metric = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Additional metrics
        balanced_accuracy = (recall + specificity) / 2
        youden_j = recall + specificity - 1
        
        metrics_dict = {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall (TPR)": round(recall, 4),
            "Specificity (TNR)": round(specificity, 4),
            "F1 Score": round(f1, 4),
            "MCC": round(mcc, 4),
            "Cohen Kappa": round(kappa, 4),
            "AUC-ROC": round(auc_roc, 4),
            "AUC-PR": round(auc_pr, 4),
            "False Positive Rate": round(fpr_metric, 4),
            "Negative Predictive Value": round(npv, 4),
            "Balanced Accuracy": round(balanced_accuracy, 4),
            "Youden J Index": round(youden_j, 4),
            "True Positives": int(tp),
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "Total Samples": len(y_test)
        }
        
        return metrics_dict
    
    def _generate_classification_report(self, y_test: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report"""
        self.logger.subsection("CLASSIFICATION REPORT")
        
        report = classification_report(y_test, y_pred, target_names=["Normal", "Attack"])
        self.logger.info(f"\n{report}")
        
        return report
    
    def _save_predictions(self, y_test: np.ndarray, y_pred: np.ndarray, 
                          y_pred_prob: np.ndarray, report_dir: str) -> None:
        """Save predictions to file"""
        predictions_df = pd.DataFrame({
            "true_label": y_test,
            "predicted_label": y_pred,
            "prediction_probability": y_pred_prob
        })
        
        predictions_path = os.path.join(report_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Predictions saved: {predictions_path}")
    
    def _save_metrics(self, metrics: Dict[str, Any], report_dir: str) -> None:
        """Save metrics to CSV file"""
        metrics_df = pd.DataFrame(
            metrics.items(),
            columns=["Metric", "Value"]
        )
        
        metrics_path = os.path.join(report_dir, "evaluation_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        self.logger.info(f"Metrics saved: {metrics_path}")
    
    def threshold_analysis(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                          output_dirs: Dict[str, str]) -> Dict[str, Any]:
        """Analyze different classification thresholds"""
        self.logger.subsection("THRESHOLD ANALYSIS")
        
        y_pred_prob = model.predict(X_test, batch_size=512, verbose=1).flatten()
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            threshold_results.append({
                "threshold": round(threshold, 2),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            })
        
        # Convert to DataFrame and save
        threshold_df = pd.DataFrame(threshold_results)
        threshold_path = os.path.join(output_dirs["reports"], "threshold_analysis.csv")
        threshold_df.to_csv(threshold_path, index=False)
        
        # Find optimal threshold (max F1)
        optimal_idx = threshold_df["f1_score"].idxmax()
        optimal_threshold = threshold_df.loc[optimal_idx, "threshold"]
        
        self.logger.info(f"Optimal threshold (max F1): {optimal_threshold}")
        self.logger.info(f"Threshold analysis saved: {threshold_path}")
        
        # Create threshold analysis plot
        self.visualizer.create_threshold_analysis(threshold_df, output_dirs["evaluation"])
        
        return {
            "optimal_threshold": float(optimal_threshold),
            "threshold_results": threshold_results,
            "threshold_path": threshold_path
        }
    
    def feature_importance_analysis(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                  feature_names: List[str], output_dirs: Dict[str, str]) -> Dict[str, Any]:
        """Analyze feature importance using permutation importance"""
        self.logger.subsection("FEATURE IMPORTANCE ANALYSIS")
        
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X_test.reshape(X_test.shape[0], X_test.shape[1]), 
            y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values("importance_mean", ascending=False)
        
        # Save importance results
        importance_path = os.path.join(output_dirs["reports"], "permutation_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        
        # Create importance plot
        self.visualizer.create_feature_importance_plot(importance_df, output_dirs["evaluation"])
        
        self.logger.info(f"Permutation importance saved: {importance_path}")
        
        return {
            "importance_path": importance_path,
            "top_features": importance_df.head(10)["feature"].tolist(),
            "importance_scores": importance_df.head(10)["importance_mean"].tolist()
        }
    
    def error_analysis(self, y_test: np.ndarray, y_pred: np.ndarray, 
                      y_pred_prob: np.ndarray, X_test: np.ndarray,
                      feature_names: List[str], output_dirs: Dict[str, str]) -> Dict[str, Any]:
        """Perform detailed error analysis"""
        self.logger.subsection("ERROR ANALYSIS")
        
        # Identify different types of errors
        false_positives = (y_test == 0) & (y_pred == 1)
        false_negatives = (y_test == 1) & (y_pred == 0)
        true_positives = (y_test == 1) & (y_pred == 1)
        true_negatives = (y_test == 0) & (y_pred == 0)
        
        error_analysis = {
            "false_positive_count": int(false_positives.sum()),
            "false_negative_count": int(false_negatives.sum()),
            "true_positive_count": int(true_positives.sum()),
            "true_negative_count": int(true_negatives.sum()),
            "false_positive_rate": float(false_positives.mean()),
            "false_negative_rate": float(false_negatives[y_test == 1].mean() if y_test.sum() > 0 else 0),
            "avg_confidence_correct": float(y_pred_prob[y_pred == y_test].mean()),
            "avg_confidence_incorrect": float(y_pred_prob[y_pred != y_test].mean())
        }
        
        # Analyze confidence distribution for errors
        error_confidence_df = pd.DataFrame({
            "confidence": y_pred_prob,
            "error_type": np.where(false_positives, "False Positive",
                         np.where(false_negatives, "False Negative",
                         np.where(true_positives, "True Positive", "True Negative")))
        })
        
        # Save error analysis
        error_path = os.path.join(output_dirs["reports"], "error_analysis.json")
        with open(error_path, "w") as f:
            json.dump(error_analysis, f, indent=4)
        
        # Create error analysis plots
        self.visualizer.create_error_analysis_plots(error_confidence_df, output_dirs["evaluation"])
        
        self.logger.info(f"Error analysis saved: {error_path}")
        
        return error_analysis
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                  output_dirs: Dict[str, str]) -> str:
        """Generate comprehensive evaluation report"""
        self.logger.subsection("GENERATING EVALUATION REPORT")
        
        metrics = evaluation_results["metrics"]
        
        html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #3498db; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
        .high {{ border-left-color: #27ae60; }}
        .medium {{ border-left-color: #f39c12; }}
        .low {{ border-left-color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #34495e; color: white; }}
        .classification-report {{ background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: monospace; white-space: pre-line; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Evaluation Report</h1>
        
        <h2>Performance Metrics</h2>
        <div class="metric-grid">
            <div class="metric-box high">
                <div class="metric-value">{metrics['Accuracy']}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-box high">
                <div class="metric-value">{metrics['Precision']}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-box high">
                <div class="metric-value">{metrics['Recall (TPR)']}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-box high">
                <div class="metric-value">{metrics['F1 Score']}</div>
                <div class="metric-label">F1 Score</div>
            </div>
            <div class="metric-box high">
                <div class="metric-value">{metrics['AUC-ROC']}</div>
                <div class="metric-label">AUC-ROC</div>
            </div>
            <div class="metric-box high">
                <div class="metric-value">{metrics['AUC-PR']}</div>
                <div class="metric-label">AUC-PR</div>
            </div>
            <div class="metric-box medium">
                <div class="metric-value">{metrics['MCC']}</div>
                <div class="metric-label">MCC</div>
            </div>
            <div class="metric-box medium">
                <div class="metric-value">{metrics['Cohen Kappa']}</div>
                <div class="metric-label">Cohen's Kappa</div>
            </div>
        </div>
        
        <h2>Confusion Matrix</h2>
        <table>
            <tr>
                <th></th>
                <th>Predicted Normal</th>
                <th>Predicted Attack</th>
            </tr>
            <tr>
                <td><strong>Actual Normal</strong></td>
                <td>{metrics['True Negatives']}</td>
                <td>{metrics['False Positives']}</td>
            </tr>
            <tr>
                <td><strong>Actual Attack</strong></td>
                <td>{metrics['False Negatives']}</td>
                <td>{metrics['True Positives']}</td>
            </tr>
        </table>
        
        <h2>Detailed Classification Report</h2>
        <div class="classification-report">{evaluation_results['classification_report']}</div>
        
        <h2>Additional Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Specificity (TNR)</td><td>{metrics['Specificity (TNR)']}</td></tr>
            <tr><td>False Positive Rate</td><td>{metrics['False Positive Rate']}</td></tr>
            <tr><td>Negative Predictive Value</td><td>{metrics['Negative Predictive Value']}</td></tr>
            <tr><td>Balanced Accuracy</td><td>{metrics['Balanced Accuracy']}</td></tr>
            <tr><td>Youden J Index</td><td>{metrics['Youden J Index']}</td></tr>
            <tr><td>Total Samples</td><td>{metrics['Total Samples']:,}</td></tr>
        </table>
    </div>
</body>
</html>
        """
        
        report_path = os.path.join(output_dirs["reports"], "evaluation_report.html")
        with open(report_path, "w") as f:
            f.write(html_report)
        
        self.logger.info(f"Evaluation report saved: {report_path}")
        return report_path
