# Advanced Intrusion Detection System Framework

A comprehensive deep learning-based intrusion detection system with blockchain-integrated secure alert logging for the NSL-KDD dataset.

## 🚀 Features

- **Deep Learning Model**: Stacked Bidirectional LSTM with attention mechanism
- **Blockchain Integration**: SHA-256 based immutable alert logging
- **Comprehensive Evaluation**: Advanced metrics including MCC, Cohen's Kappa, AUC-PR
- **Exploratory Data Analysis**: Statistical analysis, PCA, t-SNE visualizations
- **GPU Optimization**: Mixed precision training and memory management
- **Professional Visualizations**: High-resolution plots for reports and publications
- **Modular Architecture**: Clean, extensible codebase with proper separation of concerns
- **Configuration Management**: JSON-based configuration system
- **Comprehensive Logging**: Structured logging with multiple output formats

## 📊 Dataset

This framework is designed for the **NSL-KDD dataset**, a benchmark dataset for network intrusion detection research.

- **Training samples**: 125,973
- **Testing samples**: 22,544
- **Features**: 41 network traffic features
- **Attack types**: 23 different attack categories
- **Classes**: Binary classification (Normal vs Attack)

## 🏗️ Architecture

```
ids_framework/
├── src/
│   └── ids_framework/
│       ├── data/           # Data loading and preprocessing
│       ├── models/         # LSTM model architecture
│       ├── training/       # Model training utilities
│       ├── evaluation/     # Model evaluation and metrics
│       ├── blockchain/     # Blockchain alert system
│       ├── visualization/  # Plotting and visualization
│       └── utils/          # Configuration and utilities
├── configs/                # Configuration files
├── examples/               # Usage examples
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/ids-framework/ids-framework.git
cd ids-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Quick Install

```bash
pip install ids-framework
```

## 🚀 Quick Start

### Basic Usage

```python
from ids_framework import IDSFramework

# Initialize the framework
ids = IDSFramework(config_path="configs/default.json")

# Load and preprocess data
train_data, test_data = ids.load_data()

# Train the model
model = ids.train_model(train_data)

# Evaluate the model
results = ids.evaluate_model(model, test_data)

# Generate blockchain alerts
alerts = ids.log_attack_alerts(model, test_data)
```

### Command Line Interface

```bash
# Train model
ids-train --config configs/default.json --data-path /path/to/nslkdd

# Evaluate model
ids-evaluate --model-path models/best_model.h5 --test-path /path/to/test_data

# Make predictions
ids-predict --model-path models/best_model.h5 --input-file new_data.csv
```

## 📖 Detailed Usage

### 1. Configuration

Create a configuration file `configs/my_config.json`:

```json
{
    "data": {
        "dataset_path": "/path/to/nslkdd",
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
        "learning_rate": 1e-3
    },
    "training": {
        "epochs": 50,
        "early_stopping_patience": 7,
        "class_weights": "balanced"
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
    }
}
```

### 2. Data Loading and Preprocessing

```python
from ids_framework.data import DataLoader, DataPreprocessor
from ids_framework.utils import Config, Logger

# Load configuration
config = Config("configs/my_config.json")
logger = Logger("IDS_Framework")

# Load data
loader = DataLoader(config, logger)
train_df, test_df, data_info = loader.load_and_validate()

# Preprocess data
preprocessor = DataPreprocessor(config, logger)
X_train, X_test, y_train, y_test, class_weights, feature_names = preprocessor.preprocess_data(train_df, test_df)
```

### 3. Model Training

```python
from ids_framework.models import LSTMModel
from ids_framework.training import ModelTrainer

# Build model
lstm_model = LSTMModel(config, logger)
model = lstm_model.build_model(input_shape=(X_train.shape[1], 1))

# Train model
trainer = ModelTrainer(config, logger)
training_info = trainer.train_model(model, X_train, y_train, output_dirs)
```

### 4. Model Evaluation

```python
from ids_framework.evaluation import ModelEvaluator

# Evaluate model
evaluator = ModelEvaluator(config, logger)
results = evaluator.evaluate_model(model, X_test, y_test, output_dirs)

# Threshold analysis
threshold_results = evaluator.threshold_analysis(model, X_test, y_test, output_dirs)
```

### 5. Blockchain Alert System

```python
from ids_framework.blockchain import AlertSystem

# Initialize alert system
alert_system = AlertSystem(config, logger)
blockchain = alert_system.initialize_blockchain(run_id="20231201_120000")

# Get predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Log alerts to blockchain
alert_results = alert_system.log_attack_alerts(y_pred, y_pred_prob, y_test, output_dirs)
```

## 📊 Performance Metrics

The framework provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive prediction accuracy
- **Recall (TPR)**: Attack detection rate
- **Specificity (TNR)**: Normal traffic classification rate
- **F1 Score**: Harmonic mean of precision and recall
- **MCC**: Matthews correlation coefficient
- **Cohen's Kappa**: Inter-rater reliability
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve

## 🔧 Advanced Features

### GPU Optimization

```python
# Enable mixed precision training
config.set("training.mixed_precision", True)

# Configure GPU memory growth
config.set("gpu.memory_growth", True)
```

### Custom Model Architectures

```python
from ids_framework.models import ModelBuilder

# Build custom model
builder = ModelBuilder(config, logger)
model = builder.build_model("custom_lstm", input_shape=(41, 1))

# Register new model type
builder.register_model("custom_model", CustomModelClass)
```

### Ensemble Learning

```python
# Create model ensemble
models = builder.create_ensemble(
    ["bidirectional_lstm", "lstm"], 
    input_shape=(41, 1)
)
```

## 📈 Visualization Examples

The framework generates comprehensive visualizations:

- **Training Dashboard**: Loss, accuracy, AUC curves
- **Confusion Matrix**: Raw and normalized
- **ROC Curves**: With operating points
- **Precision-Recall Curves**: With threshold analysis
- **Feature Distributions**: By class comparison
- **Correlation Heatmaps**: Feature relationships
- **PCA/t-SNE Plots**: Dimensionality reduction
- **Blockchain Analytics**: Alert distribution and timeline

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ids_framework --cov-report=html

# Run specific test
pytest tests/test_models.py
```

## 📚 Documentation

- **API Documentation**: [ids-framework.readthedocs.io](https://ids-framework.readthedocs.io/)
- **Examples**: Check the `examples/` directory
- **Configuration Guide**: See `docs/configuration.md`
- **Model Architecture**: See `docs/architecture.md`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/ids-framework/ids-framework.git
cd ids-framework

# Install development dependencies
pip install -e ".[dev,docs]"

# Run pre-commit checks
pre-commit run --all-files

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NSL-KDD Dataset**: University of New Brunswick
- **TensorFlow**: Google Brain Team
- **Scikit-learn**: Scikit-learn developers
- **Blockchain Inspiration**: Bitcoin and Ethereum communities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ids-framework/ids-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ids-framework/ids-framework/discussions)
- **Email**: contact@ids-framework.com

## 🏆 Citation

If you use this framework in your research, please cite:

```bibtex
@software{ids_framework,
  title={Advanced Intrusion Detection System Framework},
  author={IDS Framework Team},
  year={2023},
  url={https://github.com/ids-framework/ids-framework}
}
```

---

**Built with ❤️ for the cybersecurity research community**
