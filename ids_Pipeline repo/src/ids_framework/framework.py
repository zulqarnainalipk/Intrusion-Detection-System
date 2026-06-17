"""
Main IDS Framework class that orchestrates the complete pipeline
"""

import os
import json
import time
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import tensorflow as tf

from .data import DataLoader, DataPreprocessor, DataExplorer
from .models import LSTMModel, ModelBuilder
from .training import ModelTrainer
from .evaluation import ModelEvaluator
from .blockchain import AlertSystem
from .visualization import TrainingVisualizer, EvaluationVisualizer, EDAVisualizer
from .utils import Config, Logger, create_run_id, create_master_log


class IDSFramework:
    """Main IDS Framework class for complete intrusion detection pipeline"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.run_id = create_run_id()
        
        # Initialize components
        self.data_loader = DataLoader(config, logger)
        self.data_preprocessor = DataPreprocessor(config, logger)
        self.data_explorer = DataExplorer(config, logger)
        self.model_builder = ModelBuilder(config, logger)
        self.model_trainer = ModelTrainer(config, logger)
        self.model_evaluator = ModelEvaluator(config, logger)
        self.alert_system = AlertSystem(config, logger)
        
        # Initialize master log
        self.master_log = create_master_log()
        self.master_log["run_id"] = self.run_id
        
        # Create output directories
        self.output_dirs = config.create_output_directories()
        
        self.logger.info(f"IDS Framework initialized - Run ID: {self.run_id}")
    
    def load_data(self) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Any]]:
        """Load and validate dataset"""
        train_df, test_df, data_info = self.data_loader.load_and_validate()
        
        # Update master log
        self.master_log["dataset"] = data_info
        
        return (train_df, test_df), data_info
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float], list
    ]:
        """Preprocess data for model training"""
        X_train_lstm, X_test_lstm, y_train, y_test, class_weights, feature_names = \
            self.data_preprocessor.preprocess_data(train_df, test_df)
        
        # Save preprocessing objects
        saved_paths = self.data_preprocessor.save_preprocessors(self.output_dirs["models"])
        self.master_log["artifacts"].extend(saved_paths.values())
        
        # Update master log
        self.master_log["preprocessing"] = self.data_preprocessor.get_preprocessing_info()
        
        return X_train_lstm, X_test_lstm, y_train, y_test, class_weights, feature_names
    
    def perform_eda(self, train_df: pd.DataFrame, X_train_scaled: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        eda_results = self.data_explorer.perform_eda(
            train_df, X_train_scaled, y_train, self.output_dirs
        )
        
        # Update master log with EDA results
        self.master_log["eda"] = eda_results
        
        return eda_results
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build the LSTM model"""
        model_type = self.config.get("model.architecture", "bidirectional_lstm")
        model = self.model_builder.build_model(model_type, input_shape)
        
        # Get model instance for training
        model_instance = self.model_builder.get_model_instance()
        
        # Update master log
        self.master_log["model"] = model_instance.get_model_info()
        
        return model
    
    def train_model(self, train_data: Tuple[np.ndarray, np.ndarray], 
                   model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
        """Train the model"""
        X_train, y_train = train_data
        
        if model is None:
            # Build model if not provided
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self.build_model(input_shape)
        
        # Update config with class weights
        class_weights = self._calculate_class_weights(y_train)
        self.config.set("training.class_weights", class_weights)
        
        # Train model
        training_info = self.model_trainer.train_model(model, X_train, y_train, self.output_dirs)
        
        # Evaluate training progress
        training_stats = self.model_trainer.evaluate_training_progress(self.output_dirs)
        
        # Update master log
        self.master_log["training"] = {**training_info, **training_stats}
        
        return model
    
    def evaluate_model(self, model: tf.keras.Model, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Evaluate the model"""
        X_test, y_test = test_data
        
        # Comprehensive evaluation
        evaluation_results = self.model_evaluator.evaluate_model(model, X_test, y_test, self.output_dirs)
        
        # Additional analyses
        threshold_results = self.model_evaluator.threshold_analysis(model, X_test, y_test, self.output_dirs)
        
        # Feature importance analysis
        feature_names = self.master_log["preprocessing"]["feature_names"]
        feature_importance = self.model_evaluator.feature_importance_analysis(
            model, X_test, y_test, feature_names, self.output_dirs
        )
        
        # Error analysis
        y_pred_prob = model.predict(X_test, batch_size=512, verbose=0).flatten()
        y_pred = self.model_evaluator._apply_threshold(y_pred_prob, self.config.get("evaluation.threshold", 0.5))
        error_analysis = self.model_evaluator.error_analysis(
            y_test, y_pred, y_pred_prob, X_test, feature_names, self.output_dirs
        )
        
        # Generate evaluation report
        report_path = self.model_evaluator.generate_evaluation_report(evaluation_results, self.output_dirs)
        
        # Update master log
        self.master_log["evaluation"] = {
            "metrics": evaluation_results["metrics"],
            "threshold_analysis": threshold_results,
            "feature_importance": feature_importance,
            "error_analysis": error_analysis,
            "report_path": report_path
        }
        
        return evaluation_results
    
    def log_attack_alerts(self, model: tf.keras.Model, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Log attack alerts to blockchain"""
        X_test, y_test = test_data
        
        # Initialize blockchain
        blockchain = self.alert_system.initialize_blockchain(self.run_id)
        
        # Get predictions
        y_pred_prob = model.predict(X_test, batch_size=512, verbose=0).flatten()
        y_pred = self.model_evaluator._apply_threshold(y_pred_prob, self.config.get("evaluation.threshold", 0.5))
        
        # Log alerts
        alert_results = self.alert_system.log_attack_alerts(y_pred, y_pred_prob, y_test, self.output_dirs)
        
        # Create additional blockchain visualizations
        self.alert_system.create_alert_timeline(self.output_dirs["evaluation"])
        
        # Export alert report
        alert_report_path = self.alert_system.export_alert_report(self.output_dirs["reports"])
        
        # Update master log
        self.master_log["blockchain"] = {
            "stats": alert_results["blockchain_stats"],
            "total_attacks": alert_results["total_attacks_detected"],
            "alerts_logged": alert_results["alerts_logged"],
            "alert_report_path": alert_report_path
        }
        
        return alert_results
    
    def save_model(self, model: tf.keras.Model, filepath: str) -> None:
        """Save the trained model"""
        model.save(filepath)
        self.master_log["artifacts"].append(filepath)
        self.logger.info(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str) -> tf.keras.Model:
        """Load a trained model"""
        model = tf.keras.models.load_model(filepath)
        self.logger.info(f"Model loaded: {filepath}")
        return model
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete IDS pipeline"""
        self.logger.section("RUNNING FULL IDS PIPELINE")
        
        start_time = time.time()
        
        try:
            # 1. Load data
            (train_df, test_df), data_info = self.load_data()
            
            # 2. Preprocess data
            X_train_lstm, X_test_lstm, y_train, y_test, class_weights, feature_names = \
                self.preprocess_data(train_df, test_df)
            
            # 3. Perform EDA
            self.perform_eda(train_df, X_train_lstm.reshape(X_train_lstm.shape[0], -1), y_train)
            
            # 4. Build and train model
            model = self.train_model((X_train_lstm, y_train))
            
            # 5. Evaluate model
            evaluation_results = self.evaluate_model(model, (X_test_lstm, y_test))
            
            # 6. Log blockchain alerts
            alert_results = self.log_attack_alerts(model, (X_test_lstm, y_test))
            
            # 7. Save final model
            final_model_path = os.path.join(self.output_dirs["models"], "final_model.h5")
            self.save_model(model, final_model_path)
            
            # 8. Generate comprehensive report
            self.generate_comprehensive_report(evaluation_results, alert_results, self.output_dirs)
            
            end_time = time.time()
            total_time = round(end_time - start_time, 2)
            
            # Update master log
            self.master_log["total_runtime_seconds"] = total_time
            self.master_log["completed_at"] = time.time()
            
            # Save master log
            self._save_master_log()
            
            self.logger.section("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total runtime: {total_time:.2f} seconds")
            self.logger.info(f"Output directory: {self.output_dirs['base']}")
            
            return {
                "run_id": self.run_id,
                "total_runtime": total_time,
                "output_dirs": self.output_dirs,
                "model_path": final_model_path,
                "evaluation_results": evaluation_results,
                "alert_results": alert_results
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def generate_comprehensive_report(self, evaluation_results: Dict[str, Any], 
                                   alert_results: Dict[str, Any], 
                                   output_dirs: Dict[str, str]) -> str:
        """Generate comprehensive HTML report"""
        report_html = self._create_html_report(evaluation_results, alert_results)
        
        report_path = os.path.join(output_dirs["reports"], f"comprehensive_report_{self.run_id}.html")
        with open(report_path, "w") as f:
            f.write(report_html)
        
        self.master_log["artifacts"].append(report_path)
        self.logger.info(f"Comprehensive report saved: {report_path}")
        
        return report_path
    
    def _create_html_report(self, evaluation_results: Dict[str, Any], 
                           alert_results: Dict[str, Any]) -> str:
        """Create HTML report content"""
        metrics = evaluation_results["metrics"]
        blockchain_stats = alert_results["blockchain_stats"]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IDS Framework Comprehensive Report - {self.run_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #f5f6fa; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 40px 60px; }}
        .container {{ max-width: 1200px; margin: 30px auto; padding: 0 30px; }}
        .card {{ background: white; border-radius: 10px; padding: 25px; margin-bottom: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
        .card h2 {{ margin-top: 0; color: #2980b9; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric-box {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; border-left: 4px solid #3498db; }}
        .metric-box .val {{ font-size: 26px; font-weight: bold; color: #2c3e50; }}
        .metric-box .lbl {{ font-size: 12px; color: #7f8c8d; margin-top: 4px; }}
        .status {{ padding: 5px 10px; border-radius: 15px; color: white; font-size: 12px; font-weight: bold; }}
        .valid {{ background: #27ae60; }}
        .invalid {{ background: #e74c3c; }}
        .footer {{ text-align: center; padding: 20px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Advanced Intrusion Detection System</h1>
        <h2>Comprehensive Report - Run ID: {self.run_id}</h2>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>📊 Model Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-box"><div class="val">{metrics['Accuracy']}</div><div class="lbl">Accuracy</div></div>
                <div class="metric-box"><div class="val">{metrics['Precision']}</div><div class="lbl">Precision</div></div>
                <div class="metric-box"><div class="val">{metrics['Recall (TPR)']}</div><div class="lbl">Recall (TPR)</div></div>
                <div class="metric-box"><div class="val">{metrics['F1 Score']}</div><div class="lbl">F1 Score</div></div>
                <div class="metric-box"><div class="val">{metrics['AUC-ROC']}</div><div class="lbl">AUC-ROC</div></div>
                <div class="metric-box"><div class="val">{metrics['AUC-PR']}</div><div class="lbl">AUC-PR</div></div>
                <div class="metric-box"><div class="val">{metrics['MCC']}</div><div class="lbl">MCC</div></div>
                <div class="metric-box"><div class="val">{metrics['Cohen Kappa']}</div><div class="lbl">Cohen Kappa</div></div>
            </div>
        </div>
        
        <div class="card">
            <h2>🔗 Blockchain Alert System</h2>
            <div class="metric-grid">
                <div class="metric-box"><div class="val">{blockchain_stats['total_blocks']}</div><div class="lbl">Total Blocks</div></div>
                <div class="metric-box"><div class="val">{alert_results['total_attacks_detected']}</div><div class="lbl">Attacks Detected</div></div>
                <div class="metric-box"><div class="val">{alert_results['alerts_logged']}</div><div class="lbl">Alerts Logged</div></div>
                <div class="metric-box"><div class="val"><span class="status {'valid' if blockchain_stats['is_valid'] else 'invalid'}">{'Valid' if blockchain_stats['is_valid'] else 'Invalid'}</span></div><div class="lbl">Chain Integrity</div></div>
            </div>
        </div>
        
        <div class="card">
            <h2>📁 Generated Artifacts</h2>
            <p>Total artifacts generated: {len(self.master_log['artifacts'])}</p>
            <p>All artifacts saved in: {self.output_dirs['base']}</p>
        </div>
    </div>
    
    <div class="footer">
        <p>Generated by IDS Framework v2.0 • Run {self.run_id}</p>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _save_master_log(self) -> None:
        """Save master log to file"""
        log_path = os.path.join(self.output_dirs["reports"], f"master_log_{self.run_id}.json")
        
        with open(log_path, "w") as f:
            json.dump(self.master_log, f, indent=4, default=str)
        
        self.logger.info(f"Master log saved: {log_path}")
    
    def _calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.array([0, 1])
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        
        return {0: class_weights_arr[0], 1: class_weights_arr[1]}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get framework summary"""
        return {
            "run_id": self.run_id,
            "config": self.config.get(""),
            "output_dirs": self.output_dirs,
            "master_log": self.master_log
        }
