"""
Advanced Intrusion Detection System Framework

A comprehensive deep learning based intrusion detection system
with blockchain integrated alert logging for NSL-KDD dataset.
"""

__version__ = "2.0.0"
__author__ = "IDS Framework Team"
__email__ = "contact@ids-framework.com"
__description__ = "Advanced Deep Learning Based Intrusion Detection System with Blockchain Integration"

from .data import DataLoader, DataPreprocessor
from .models import LSTMModel
from .training import ModelTrainer
from .evaluation import ModelEvaluator
from .blockchain import Blockchain, Block
from .utils import Config, Logger
from .visualization import Visualizer

__all__ = [
    "DataLoader",
    "DataPreprocessor", 
    "LSTMModel",
    "ModelTrainer",
    "ModelEvaluator",
    "Blockchain",
    "Block",
    "Config",
    "Logger",
    "Visualizer"
]
