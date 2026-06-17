"""
Helper utilities for IDS Framework
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    import random
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_run_id() -> str:
    """Create unique run ID based on timestamp"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def get_feature_columns() -> List[str]:
    """Get NSL-KDD feature column names"""
    return [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "attack", "difficulty"
    ]


def get_categorical_columns() -> List[str]:
    """Get categorical column names"""
    return ["protocol_type", "service", "flag"]


def get_numerical_columns() -> List[str]:
    """Get numerical column names (excluding binary and target)"""
    all_cols = get_feature_columns()
    categorical = get_categorical_columns()
    exclude = ["attack", "difficulty", "land", "logged_in", "root_shell", 
               "su_attempted", "is_host_login", "is_guest_login", "num_outbound_cmds"]
    
    numerical = [col for col in all_cols 
                if col not in categorical and col not in exclude]
    return numerical


def classify_threat_level(confidence: float, thresholds: Dict[str, float]) -> str:
    """Classify threat level based on confidence score"""
    if confidence >= thresholds["critical"]:
        return "CRITICAL"
    elif confidence >= thresholds["high"]:
        return "HIGH"
    elif confidence >= thresholds["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


def format_metrics_dict(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Format metrics dictionary for display"""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
        elif isinstance(value, int):
            formatted[key] = f"{value:,}"
        else:
            formatted[key] = str(value)
    return formatted


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced dataset"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.array([0, 1])
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    
    return {0: class_weights_arr[0], 1: class_weights_arr[1]}


def validate_data_shape(X: np.ndarray, y: np.ndarray, 
                        expected_features: int) -> Tuple[bool, str]:
    """Validate data shapes"""
    if X.shape[1] != expected_features:
        return False, f"Expected {expected_features} features, got {X.shape[1]}"
    
    if X.shape[0] != y.shape[0]:
        return False, f"X and y have mismatched sample counts: {X.shape[0]} vs {y.shape[0]}"
    
    if len(y.shape) != 1:
        return False, f"Expected 1D target array, got shape {y.shape}"
    
    return True, "Data validation passed"


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices("GPU")
    
    return {
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "gpu_names": [gpu.name for gpu in gpus] if gpus else [],
        "logical_devices": len(tf.config.list_logical_devices())
    }


def setup_gpu_memory_growth() -> bool:
    """Setup GPU memory growth"""
    import tensorflow as tf
    
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision if available
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                return True
            except:
                return True
        return False
    except:
        return False


def create_master_log() -> Dict[str, Any]:
    """Create master log dictionary"""
    run_id = create_run_id()
    gpu_info = get_gpu_info()
    
    return {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu_available": gpu_info["gpu_available"],
        "gpu_count": gpu_info["gpu_count"],
        "dataset": {},
        "preprocessing": {},
        "model": {},
        "training": {},
        "evaluation": {},
        "blockchain": {},
        "artifacts": []
    }


def save_artifact_list(artifacts: List[str], filepath: str) -> None:
    """Save list of generated artifacts"""
    ensure_dir(os.path.dirname(filepath))
    
    artifact_data = {
        "run_id": create_run_id(),
        "timestamp": datetime.datetime.now().isoformat(),
        "total_artifacts": len(artifacts),
        "artifacts": artifacts
    }
    
    save_json(artifact_data, filepath)


def load_artifact_list(filepath: str) -> List[str]:
    """Load list of generated artifacts"""
    if os.path.exists(filepath):
        data = load_json(filepath)
        return data.get("artifacts", [])
    return []


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": format_file_size(psutil.virtual_memory().total),
        "memory_available": format_file_size(psutil.virtual_memory().available)
    }


def validate_file_paths(paths: List[str]) -> Dict[str, bool]:
    """Validate if file paths exist"""
    return {path: os.path.exists(path) for path in paths}
