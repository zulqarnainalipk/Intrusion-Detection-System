# IDS Framework API Documentation

## Overview

The IDS Framework provides a comprehensive API for building, training, and evaluating deep learning-based intrusion detection systems with blockchain-integrated alert logging.

## Core Classes

### IDSFramework

The main orchestrator class that manages the complete pipeline.

```python
from ids_framework import IDSFramework
from ids_framework.utils import Config, Logger

# Initialize framework
config = Config("configs/default.json")
logger = Logger("IDS_Framework")
ids = IDSFramework(config, logger)

# Run complete pipeline
results = ids.run_full_pipeline()
```

#### Methods

- `load_data()` - Load and validate NSL-KDD dataset
- `preprocess_data(train_df, test_df)` - Preprocess data for model training
- `build_model(input_shape)` - Build LSTM model architecture
- `train_model(train_data, model=None)` - Train the model
- `evaluate_model(model, test_data)` - Comprehensive model evaluation
- `log_attack_alerts(model, test_data)` - Log alerts to blockchain
- `run_full_pipeline()` - Execute complete pipeline

### Config

Configuration management class for loading and managing settings.

```python
from ids_framework.utils import Config

# Load configuration
config = Config("configs/default.json")

# Get configuration values
batch_size = config.get("data.batch_size", 256)

# Set configuration values
config.set("model.learning_rate", 0.001)

# Create output directories
output_dirs = config.create_output_directories()
```

#### Methods

- `get(key, default=None)` - Get configuration value using dot notation
- `set(key, value)` - Set configuration value using dot notation
- `load_config(config_path)` - Load configuration from JSON file
- `save_config(config_path)` - Save configuration to JSON file
- `create_output_directories()` - Create output directory structure

### Logger

Enhanced logging utility with structured output.

```python
from ids_framework.utils import Logger

# Initialize logger
logger = Logger("MyComponent", log_level="INFO")

# Log messages
logger.info("Processing started")
logger.error("Error occurred", exc_info=True)

# Log sections and metrics
logger.section("TRAINING PHASE")
logger.metrics({"accuracy": 0.95, "loss": 0.05})
```

#### Methods

- `info(message)` - Log info message
- `debug(message)` - Log debug message
- `warning(message)` - Log warning message
- `error(message)` - Log error message
- `section(title)` - Log section header
- `metrics(metrics_dict, title)` - Log metrics in formatted way

## Data Processing

### DataLoader

Handles loading and validation of NSL-KDD dataset.

```python
from ids_framework.data import DataLoader

loader = DataLoader(config, logger)
train_df, test_df, data_info = loader.load_and_validate()
```

#### Methods

- `validate_paths()` - Validate data file paths exist
- `load_data()` - Load training and test data
- `get_data_info(train_df, test_df)` - Get dataset information

### DataPreprocessor

Preprocesses data for model training.

```python
from ids_framework.data import DataPreprocessor

preprocessor = DataPreprocessor(config, logger)
X_train, X_test, y_train, y_test, class_weights, feature_names = \
    preprocessor.preprocess_data(train_df, test_df)
```

#### Methods

- `preprocess_data(train_df, test_df)` - Complete preprocessing pipeline
- `save_preprocessors(model_dir)` - Save preprocessing objects
- `load_preprocessors(model_dir)` - Load preprocessing objects

### DataExplorer

Performs exploratory data analysis.

```python
from ids_framework.data import DataExplorer

explorer = DataExplorer(config, logger)
eda_results = explorer.perform_eda(train_df, X_train_scaled, y_train, output_dirs)
```

#### Methods

- `perform_eda(train_df, X_train_scaled, y_train, output_dirs)` - Comprehensive EDA
- `_generate_statistical_summary(train_df, report_dir)` - Statistical analysis
- `_analyze_attack_distribution(train_df, output_dirs)` - Attack distribution analysis

## Model Components

### LSTMModel

Bidirectional LSTM model for intrusion detection.

```python
from ids_framework.models import LSTMModel

lstm_model = LSTMModel(config, logger)
model = lstm_model.build_model(input_shape=(41, 1))
```

#### Methods

- `build_model(input_shape)` - Build LSTM model architecture
- `get_model()` - Get the built model
- `save_model(filepath)` - Save the model
- `load_model(filepath)` - Load a saved model
- `predict(X, batch_size=512)` - Make predictions
- `evaluate(X, y, batch_size=512)` - Evaluate the model

### ModelBuilder

Factory class for building different model architectures.

```python
from ids_framework.models import ModelBuilder

builder = ModelBuilder(config, logger)
model = builder.build_model("bidirectional_lstm", input_shape=(41, 1))
```

#### Methods

- `build_model(model_type, input_shape)` - Build model of specified type
- `get_supported_models()` - Get list of supported model types
- `register_model(name, model_class)` - Register new model type
- `create_ensemble(model_types, input_shape)` - Create model ensemble

## Training

### ModelTrainer

Handles model training with comprehensive monitoring.

```python
from ids_framework.training import ModelTrainer

trainer = ModelTrainer(config, logger)
training_info = trainer.train_model(model, X_train, y_train, output_dirs)
```

#### Methods

- `train_model(model, X_train, y_train, output_dirs)` - Train the model
- `evaluate_training_progress(output_dirs)` - Evaluate training progress
- `get_best_epoch()` - Get epoch with best validation performance
- `get_training_recommendations()` - Get training improvement recommendations

## Evaluation

### ModelEvaluator

Comprehensive model evaluation.

```python
from ids_framework.evaluation import ModelEvaluator

evaluator = ModelEvaluator(config, logger)
results = evaluator.evaluate_model(model, X_test, y_test, output_dirs)
```

#### Methods

- `evaluate_model(model, X_test, y_test, output_dirs)` - Comprehensive evaluation
- `threshold_analysis(model, X_test, y_test, output_dirs)` - Threshold optimization
- `feature_importance_analysis(model, X_test, y_test, feature_names, output_dirs)` - Feature importance
- `error_analysis(y_test, y_pred, y_pred_prob, X_test, feature_names, output_dirs)` - Error analysis
- `generate_evaluation_report(evaluation_results, output_dirs)` - Generate HTML report

## Blockchain

### Blockchain

SHA-256 based blockchain for secure alert logging.

```python
from ids_framework.blockchain import Blockchain

blockchain = Blockchain(chain_id="IDS-CHAIN-01")
block = blockchain.add_block(alert_data)
is_valid, msg = blockchain.is_chain_valid()
```

#### Methods

- `add_block(data)` - Add new block to chain
- `is_chain_valid()` - Validate chain integrity
- `get_stats()` - Get blockchain statistics
- `get_alert_blocks()` - Get all alert blocks
- `export_to_json(filepath)` - Export blockchain to JSON

### AlertSystem

Manages blockchain-based alert logging.

```python
from ids_framework.blockchain import AlertSystem

alert_system = AlertSystem(config, logger)
blockchain = alert_system.initialize_blockchain(run_id)
alert_results = alert_system.log_attack_alerts(y_pred, y_pred_prob, y_test, output_dirs)
```

#### Methods

- `initialize_blockchain(run_id)` - Initialize blockchain for logging
- `log_attack_alerts(y_pred, y_pred_prob, y_test, output_dirs)` - Log alerts to blockchain
- `get_alert_analysis()` - Get comprehensive alert analysis
- `create_alert_timeline(output_dir)` - Create alert timeline visualization
- `export_alert_report(output_dir)` - Export alert report

## Visualization

### TrainingVisualizer

Creates training progress visualizations.

```python
from ids_framework.visualization import TrainingVisualizer

visualizer = TrainingVisualizer(config, logger)
visualizer.create_training_dashboard(history, epoch_df, output_dir)
```

#### Methods

- `create_training_dashboard(history, epoch_df, output_dir)` - Training dashboard
- `create_overfitting_analysis(history, output_dir)` - Overfitting analysis
- `create_learning_curve_analysis(history, output_dir)` - Learning curves

### EvaluationVisualizer

Creates evaluation visualizations.

```python
from ids_framework.visualization import EvaluationVisualizer

visualizer = EvaluationVisualizer(config, logger)
visualizer.create_confusion_matrix(y_test, y_pred, output_dir)
```

#### Methods

- `create_confusion_matrix(y_test, y_pred, output_dir)` - Confusion matrix
- `create_roc_curve(y_test, y_pred_prob, output_dir)` - ROC curve
- `create_precision_recall_curve(y_test, y_pred_prob, output_dir)` - PR curve
- `create_metrics_dashboard(metrics, output_dir)` - Metrics dashboard

## Utility Functions

### Helper Functions

```python
from ids_framework.utils.helpers import (
    set_random_seeds, create_run_id, get_feature_columns,
    calculate_class_weights, validate_data_shape
)

# Set random seeds for reproducibility
set_random_seeds(42)

# Create unique run ID
run_id = create_run_id()

# Get NSL-KDD feature columns
features = get_feature_columns()

# Calculate class weights
weights = calculate_class_weights(y_train)
```

## Configuration Options

### Data Configuration

```json
{
    "data": {
        "dataset_path": "/path/to/nslkdd",
        "train_file": "KDDTrain+.txt",
        "test_file": "KDDTest+.txt",
        "validation_split": 0.20,
        "batch_size": 256,
        "random_seed": 42
    }
}
```

### Model Configuration

```json
{
    "model": {
        "architecture": "bidirectional_lstm",
        "l2_regularization": 0.0001,
        "dropout_rates": [0.35, 0.30, 0.25, 0.20, 0.15],
        "lstm_units": [128, 96, 64],
        "dense_units": [128, 64],
        "learning_rate": 0.001,
        "gradient_clip_norm": 1.0
    }
}
```

### Training Configuration

```json
{
    "training": {
        "epochs": 50,
        "early_stopping_patience": 7,
        "reduce_lr_patience": 3,
        "reduce_lr_factor": 0.5,
        "min_lr": 0.000001,
        "class_weights": "balanced",
        "mixed_precision": true
    }
}
```

### Blockchain Configuration

```json
{
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

## Error Handling

The framework uses structured error handling with specific exception types:

```python
try:
    results = ids.run_full_pipeline()
except FileNotFoundError as e:
    logger.error(f"Data files not found: {e}")
except ValueError as e:
    logger.error(f"Invalid configuration: {e}")
except RuntimeError as e:
    logger.error(f"Training failed: {e}")
```

## Performance Optimization

### GPU Configuration

```python
# Enable GPU memory growth
config.set("gpu.memory_growth", True)

# Enable mixed precision
config.set("training.mixed_precision", True)
```

### Batch Processing

```python
# Configure batch size for memory efficiency
config.set("data.batch_size", 512)

# Process large datasets in batches
batch_size = 1000
for i in range(0, len(X_test), batch_size):
    batch_X = X_test[i:i+batch_size]
    predictions = model.predict(batch_X, batch_size=256)
```

## Extending the Framework

### Custom Models

```python
from ids_framework.models.lstm_model import LSTMModel

class CustomModel(LSTMModel):
    def build_model(self, input_shape):
        # Custom architecture implementation
        pass

# Register custom model
builder.register_model("custom_model", CustomModel)
```

### Custom Evaluators

```python
from ids_framework.evaluation.evaluator import ModelEvaluator

class CustomEvaluator(ModelEvaluator):
    def custom_metric(self, y_true, y_pred):
        # Custom metric implementation
        pass
```

## Best Practices

1. **Configuration Management**: Use JSON configuration files for all parameters
2. **Logging**: Use structured logging with appropriate log levels
3. **Error Handling**: Handle exceptions gracefully with specific error types
4. **Memory Management**: Use batch processing for large datasets
5. **Reproducibility**: Set random seeds and save configurations
6. **Documentation**: Include comprehensive docstrings and examples
7. **Testing**: Write unit tests for all components
8. **Validation**: Validate inputs and data integrity
9. **Performance**: Monitor GPU memory usage and optimize batch sizes
10. **Security**: Never log sensitive information or credentials
