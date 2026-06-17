"""
Model training utilities for IDS Framework
"""

import time
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..utils.logger import Logger
from ..visualization.training_plots import TrainingVisualizer


class ModelTrainer:
    """Model training class for IDS Framework"""
    
    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        self.visualizer = TrainingVisualizer(config, logger)
        self.training_history = None
        
    def train_model(self, model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   output_dirs: Dict[str, str]) -> Dict[str, Any]:
        """Train the model with comprehensive monitoring"""
        self.logger.section("TRAINING MODEL")
        
        # Get training configuration
        epochs = self.config.get("training.epochs", 50)
        batch_size = self.config.get("data.batch_size", 256)
        validation_split = self.config.get("data.validation_split", 0.20)
        class_weights = self.config.get("training.class_weights", {})
        
        # Get callbacks
        callbacks, epoch_log_path, best_model_path = model.get_callbacks(output_dirs)
        
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Validation split: {validation_split}")
        self.logger.info(f"  Class weights: {class_weights}")
        
        # Start training
        train_start = time()
        
        try:
            history = model.fit(
                X_train,
                y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            train_end = time()
            train_time = round(train_end - train_start, 2)
            
            # Store training history
            self.training_history = history
            
            # Calculate training statistics
            epochs_run = len(history.history["loss"])
            best_val_auc = max(history.history["val_auc"])
            
            training_info = {
                "epochs_run": epochs_run,
                "training_time_s": train_time,
                "best_val_auc": round(float(best_val_auc), 4),
                "batch_size": batch_size,
                "validation_split": validation_split,
                "class_weights_used": bool(class_weights),
                "epoch_log_path": epoch_log_path,
                "best_model_path": best_model_path
            }
            
            self.logger.info(f"Training completed successfully")
            self.logger.info(f"Epochs run: {epochs_run}/{epochs}")
            self.logger.info(f"Training time: {train_time:.1f}s")
            self.logger.info(f"Best validation AUC: {best_val_auc:.4f}")
            
            return training_info
            
        except Exception as e:
            self.logger.error_report(e, "Model Training")
            raise
    
    def evaluate_training_progress(self, output_dirs: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate and visualize training progress"""
        if self.training_history is None:
            raise ValueError("No training history available. Train a model first.")
        
        self.logger.section("TRAINING EVALUATION")
        
        # Load epoch log
        epoch_log_path = f"{output_dirs['logs']}/training_epoch_log.csv"
        epoch_df = pd.read_csv(epoch_log_path)
        
        self.logger.info("Per-Epoch Metrics Log:")
        self.logger.info(f"Latest 10 epochs:\n{epoch_df.tail(10).to_string()}")
        
        # Generate training visualizations
        self.visualizer.create_training_dashboard(
            self.training_history, epoch_df, output_dirs["training"]
        )
        
        self.visualizer.create_overfitting_analysis(
            self.training_history, output_dirs["training"]
        )
        
        # Calculate training statistics
        training_stats = self._calculate_training_statistics(self.training_history, epoch_df)
        
        return training_stats
    
    def _calculate_training_statistics(self, history: tf.keras.callbacks.History, 
                                     epoch_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive training statistics"""
        epochs_run = len(history.history["loss"])
        
        # Best metrics
        best_val_loss = min(history.history["val_loss"])
        best_val_acc = max(history.history["val_accuracy"])
        best_val_auc = max(history.history["val_auc"])
        best_val_precision = max(history.history["val_precision"])
        best_val_recall = max(history.history["val_recall"])
        
        # Final metrics
        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        
        # Overfitting indicators
        train_val_acc_gap = np.array(history.history["accuracy"]) - np.array(history.history["val_accuracy"])
        train_val_loss_gap = np.array(history.history["val_loss"]) - np.array(history.history["loss"])
        
        avg_acc_gap = np.mean(np.abs(train_val_acc_gap))
        avg_loss_gap = np.mean(np.abs(train_val_loss_gap))
        
        # Learning rate statistics
        if "lr" in epoch_df.columns:
            initial_lr = epoch_df["lr"].iloc[0]
            final_lr = epoch_df["lr"].iloc[-1]
            lr_reductions = len(epoch_df[epoch_df["lr"].diff() < 0])
        else:
            initial_lr = final_lr = lr_reductions = 0
        
        return {
            "epochs_run": epochs_run,
            "best_metrics": {
                "val_loss": round(float(best_val_loss), 4),
                "val_accuracy": round(float(best_val_acc), 4),
                "val_auc": round(float(best_val_auc), 4),
                "val_precision": round(float(best_val_precision), 4),
                "val_recall": round(float(best_val_recall), 4)
            },
            "final_metrics": {
                "train_loss": round(float(final_train_loss), 4),
                "val_loss": round(float(final_val_loss), 4),
                "train_accuracy": round(float(final_train_acc), 4),
                "val_accuracy": round(float(final_val_acc), 4)
            },
            "overfitting_indicators": {
                "avg_accuracy_gap": round(float(avg_acc_gap), 4),
                "avg_loss_gap": round(float(avg_loss_gap), 4),
                "max_accuracy_gap": round(float(np.max(train_val_acc_gap)), 4),
                "max_loss_gap": round(float(np.max(train_val_loss_gap)), 4)
            },
            "learning_rate_stats": {
                "initial_lr": float(initial_lr),
                "final_lr": float(final_lr),
                "lr_reductions": int(lr_reductions)
            }
        }
    
    def load_training_history(self, history_path: str) -> tf.keras.callbacks.History:
        """Load training history from file"""
        # This would require implementing a custom history loading mechanism
        # For now, we'll create a simple history object from CSV
        epoch_df = pd.read_csv(history_path)
        
        # Create a mock history object
        class MockHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        history_dict = {}
        for col in epoch_df.columns:
            if col != "epoch":
                history_dict[col] = epoch_df[col].tolist()
        
        self.training_history = MockHistory(history_dict)
        return self.training_history
    
    def get_best_epoch(self) -> int:
        """Get the epoch with best validation performance"""
        if self.training_history is None:
            raise ValueError("No training history available")
        
        val_auc = self.training_history.history["val_auc"]
        best_epoch = np.argmax(val_auc) + 1
        return best_epoch
    
    def should_continue_training(self, patience: int = 5) -> bool:
        """Determine if training should continue based on recent performance"""
        if self.training_history is None:
            return True
        
        val_loss = self.training_history.history["val_loss"]
        if len(val_loss) < patience + 1:
            return True
        
        # Check if validation loss has improved in the last 'patience' epochs
        recent_losses = val_loss[-patience:]
        best_recent = min(recent_losses)
        best_overall = min(val_loss[:-patience])
        
        return best_recent < best_overall
    
    def get_training_recommendations(self) -> List[str]:
        """Get recommendations for improving training"""
        if self.training_history is None:
            return ["No training history available"]
        
        recommendations = []
        stats = self._calculate_training_statistics(
            self.training_history, 
            pd.read_csv(f"{self.config.get('output.log_dir', 'logs')}/training_epoch_log.csv")
        )
        
        # Check for overfitting
        if stats["overfitting_indicators"]["avg_accuracy_gap"] > 0.1:
            recommendations.append("Consider increasing dropout or regularization to reduce overfitting")
        
        # Check for underfitting
        if stats["final_metrics"]["val_accuracy"] < 0.8:
            recommendations.append("Model may be underfitting. Consider increasing model capacity")
        
        # Check learning rate
        if stats["learning_rate_stats"]["lr_reductions"] > 3:
            recommendations.append("Learning rate reduced multiple times. Consider adjusting initial LR")
        
        # Check training time
        if stats["epochs_run"] < self.config.get("training.epochs", 50) * 0.5:
            recommendations.append("Training stopped early. Consider increasing early stopping patience")
        
        if not recommendations:
            recommendations.append("Training appears to have proceeded normally")
        
        return recommendations
