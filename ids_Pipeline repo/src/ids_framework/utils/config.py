"""
Configuration management for IDS Framework
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration management class for the IDS Framework"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "data": {
                "dataset_path": "/kaggle/input/datasets/hassan06/nslkdd",
                "train_file": "KDDTrain+.txt",
                "test_file": "KDDTest+.txt",
                "validation_split": 0.20,
                "batch_size": 256,
                "random_seed": 42
            },
            "model": {
                "architecture": "bidirectional_lstm",
                "l2_regularization": 1e-4,
                "dropout_rates": [0.35, 0.30, 0.25, 0.20, 0.15],
                "lstm_units": [128, 96, 64],
                "dense_units": [128, 64],
                "learning_rate": 1e-3,
                "gradient_clip_norm": 1.0
            },
            "training": {
                "epochs": 50,
                "early_stopping_patience": 7,
                "reduce_lr_patience": 3,
                "reduce_lr_factor": 0.5,
                "min_lr": 1e-6,
                "class_weights": "balanced",
                "mixed_precision": True
            },
            "evaluation": {
                "threshold": 0.5,
                "save_predictions": True,
                "generate_plots": True,
                "plot_dpi": 300
            },
            "blockchain": {
                "chain_id": "IDS-CHAIN-01",
                "max_blocks": 300,
                "threat_levels": {
                    "critical": 0.95,
                    "high": 0.85,
                    "medium": 0.70,
                    "low": 0.0
                }
            },
            "output": {
                "base_dir": "project_outputs",
                "model_dir": "models",
                "figure_dir": "figures",
                "report_dir": "reports",
                "blockchain_dir": "blockchain",
                "log_dir": "logs"
            },
            "gpu": {
                "memory_growth": True,
                "mixed_precision": True
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        self._deep_update(self.config, user_config)
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def create_output_directories(self) -> Dict[str, str]:
        """Create output directories and return paths"""
        base_dir = self.get("output.base_dir")
        directories = {
            "base": base_dir,
            "figures": os.path.join(base_dir, self.get("output.figure_dir")),
            "models": os.path.join(base_dir, self.get("output.model_dir")),
            "reports": os.path.join(base_dir, self.get("output.report_dir")),
            "blockchain": os.path.join(base_dir, self.get("output.blockchain_dir")),
            "logs": os.path.join(base_dir, self.get("output.log_dir")),
            "eda": os.path.join(base_dir, self.get("output.figure_dir"), "eda"),
            "evaluation": os.path.join(base_dir, self.get("output.figure_dir"), "evaluation"),
            "training": os.path.join(base_dir, self.get("output.figure_dir"), "training")
        }
        
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return directories
