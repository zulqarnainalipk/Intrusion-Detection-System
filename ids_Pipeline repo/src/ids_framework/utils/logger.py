"""
Logging utilities for IDS Framework
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path


class Logger:
    """Enhanced logging utility for IDS Framework"""
    
    def __init__(self, name: str = "IDS_Framework", log_level: str = "INFO", 
                 log_file: Optional[str] = None, console_output: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)
    
    def section(self, title: str) -> None:
        """Log section header"""
        separator = "=" * 70
        self.info(f"\n{separator}")
        self.info(f"  {title}")
        self.info(f"{separator}\n")
    
    def subsection(self, title: str) -> None:
        """Log subsection header"""
        separator = "-" * 50
        self.info(f"\n{separator}")
        self.info(f"  {title}")
        self.info(f"{separator}\n")
    
    def metrics(self, metrics_dict: dict, title: str = "Metrics") -> None:
        """Log metrics in a formatted way"""
        self.subsection(title)
        for key, value in metrics_dict.items():
            self.info(f"  {key:<25} : {value}")
    
    def model_summary(self, model) -> None:
        """Log model summary"""
        self.section("MODEL ARCHITECTURE")
        self.info(f"Total Parameters: {model.count_params():,}")
        self.info(f"Model Layers: {len(model.layers)}")
        
        for i, layer in enumerate(model.layers):
            self.info(f"  Layer {i+1}: {layer.name} - {layer.output_shape}")
    
    def training_progress(self, epoch: int, total_epochs: int, 
                         metrics: dict, is_best: bool = False) -> None:
        """Log training progress"""
        status = " [BEST]" if is_best else ""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch+1}/{total_epochs}{status} - {metrics_str}")
    
    def evaluation_results(self, metrics_dict: dict) -> None:
        """Log evaluation results"""
        self.section("EVALUATION RESULTS")
        for metric, value in metrics_dict.items():
            if isinstance(value, float):
                self.info(f"  {metric:<25} : {value:.4f}")
            else:
                self.info(f"  {metric:<25} : {value}")
    
    def blockchain_summary(self, blockchain_stats: dict) -> None:
        """Log blockchain summary"""
        self.section("BLOCKCHAIN SUMMARY")
        self.info(f"Chain ID: {blockchain_stats.get('chain_id', 'N/A')}")
        self.info(f"Total Blocks: {blockchain_stats.get('total_blocks', 0)}")
        self.info(f"Chain Valid: {blockchain_stats.get('is_valid', False)}")
        
        alert_counts = blockchain_stats.get('alert_counts', {})
        if alert_counts:
            self.info("Alert Breakdown:")
            for level, count in alert_counts.items():
                self.info(f"  {level:<12} : {count}")
    
    def artifact_saved(self, artifact_path: str, description: str = "") -> None:
        """Log artifact save"""
        desc_str = f" ({description})" if description else ""
        self.info(f"Artifact Saved{desc_str}: {artifact_path}")
    
    def error_report(self, error: Exception, context: str = "") -> None:
        """Log error with context"""
        self.error(f"Error in {context}: {str(error)}")
        self.debug(f"Error type: {type(error).__name__}")
    
    @staticmethod
    def get_default_log_path(base_dir: str, run_id: str) -> str:
        """Get default log file path"""
        return os.path.join(base_dir, "logs", f"ids_framework_{run_id}.log")
