"""
Advanced usage example for IDS Framework
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ids_framework import IDSFramework
from ids_framework.utils import Config, Logger
from ids_framework.data import DataLoader, DataPreprocessor
from ids_framework.models import ModelBuilder
from ids_framework.training import ModelTrainer
from ids_framework.evaluation import ModelEvaluator
from ids_framework.blockchain import AlertSystem


def advanced_training_example():
    """Advanced training with custom configurations"""
    
    logger = Logger("IDS_Advanced_Example")
    
    # Load and customize configuration
    config = Config("configs/default.json")
    
    # Custom model configuration
    config.set("model.lstm_units", [256, 128, 64])
    config.set("model.dropout_rates", [0.4, 0.35, 0.3, 0.25, 0.2])
    config.set("training.epochs", 100)
    config.set("training.early_stopping_patience", 10)
    
    # Initialize framework
    ids = IDSFramework(config, logger)
    
    # Step-by-step pipeline
    logger.section("ADVANCED STEP-BY-STEP PIPELINE")
    
    # 1. Load data
    (train_df, test_df), data_info = ids.load_data()
    logger.info(f"Data loaded: {data_info['train_samples']} train, {data_info['test_samples']} test samples")
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, class_weights, feature_names = ids.preprocess_data(train_df, test_df)
    logger.info(f"Data preprocessed: {X_train.shape} train, {X_test.shape} test")
    
    # 3. Custom model building
    model_builder = ModelBuilder(config, logger)
    model = model_builder.build_model("bidirectional_lstm", (X_train.shape[1], 1))
    
    # 4. Advanced training with monitoring
    trainer = ModelTrainer(config, logger)
    training_info = trainer.train_model(model, X_train, y_train, ids.output_dirs)
    
    # 5. Comprehensive evaluation
    evaluator = ModelEvaluator(config, logger)
    results = evaluator.evaluate_model(model, X_test, y_test, ids.output_dirs)
    
    # 6. Threshold optimization
    threshold_results = evaluator.threshold_analysis(model, X_test, y_test, ids.output_dirs)
    optimal_threshold = threshold_results["optimal_threshold"]
    logger.info(f"Optimal threshold: {optimal_threshold}")
    
    # 7. Feature importance analysis
    feature_importance = evaluator.feature_importance_analysis(
        model, X_test, y_test, feature_names, ids.output_dirs
    )
    
    # 8. Blockchain alert system
    alert_system = AlertSystem(config, logger)
    blockchain = alert_system.initialize_blockchain(ids.run_id)
    
    # Get predictions with optimal threshold
    y_pred_prob = model.predict(X_test, batch_size=512, verbose=0).flatten()
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
    
    alert_results = alert_system.log_attack_alerts(y_pred_optimal, y_pred_prob, y_test, ids.output_dirs)
    
    # 9. Generate detailed reports
    comprehensive_report = ids.generate_comprehensive_report(results, alert_results, ids.output_dirs)
    
    logger.section("ADVANCED PIPELINE COMPLETED")
    logger.info(f"Comprehensive report: {comprehensive_report}")
    
    return results, alert_results


def ensemble_learning_example():
    """Example of ensemble learning"""
    
    logger = Logger("IDS_Ensemble_Example")
    config = Config("configs/default.json")
    
    # Initialize model builder
    model_builder = ModelBuilder(config, logger)
    
    # Create ensemble of models
    input_shape = (41, 1)  # NSL-KDD feature shape
    ensemble_models = model_builder.create_ensemble(
        ["bidirectional_lstm"], 
        input_shape
    )
    
    logger.info(f"Created ensemble with {len(ensemble_models)} models")
    
    # Note: In practice, you would train each model separately
    # and combine their predictions using voting or averaging
    
    return ensemble_models


def custom_model_example():
    """Example of creating a custom model"""
    
    logger = Logger("IDS_Custom_Model_Example")
    config = Config("configs/default.json")
    
    # Register custom model
    from ids_framework.models.lstm_model import LSTMModel
    
    class CustomLSTMModel(LSTMModel):
        """Custom LSTM model with attention mechanism"""
        
        def build_model(self, input_shape):
            """Build custom LSTM with attention"""
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
            from tensorflow.keras.layers import Multiply, Permute, Reshape, Lambda
            from tensorflow.keras.regularizers import l2
            
            # Input layer
            inputs = Input(shape=input_shape)
            
            # LSTM layers
            x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Simple attention mechanism
            attention = Dense(1, activation='tanh')(x)
            attention = Flatten()(attention)
            attention = Dense(x.shape[1], activation='softmax')(attention)
            attention = Reshape((x.shape[1], 1))(attention)
            
            # Apply attention
            attended = Multiply()([x, attention])
            attended = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
            
            # Dense layers
            x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(attended)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            outputs = Dense(1, activation='sigmoid')(x)
            
            # Compile model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC']
            )
            
            self.model = model
            return model
    
    # Register custom model
    model_builder = ModelBuilder(config, logger)
    model_builder.register_model("custom_lstm_attention", CustomLSTMModel)
    
    # Build custom model
    custom_model = model_builder.build_model("custom_lstm_attention", (41, 1))
    
    logger.info("Custom LSTM model with attention built successfully")
    
    return custom_model


def batch_processing_example():
    """Example of batch processing for large datasets"""
    
    logger = Logger("IDS_Batch_Example")
    config = Config("configs/default.json")
    
    # Configure for batch processing
    config.set("data.batch_size", 512)
    config.set("training.validation_split", 0.15)
    
    ids = IDSFramework(config, logger)
    
    # Load and preprocess data
    (train_df, test_df), data_info = ids.load_data()
    X_train, X_test, y_train, y_test, class_weights, feature_names = ids.preprocess_data(train_df, test_df)
    
    # Train with batch processing
    model = ids.train_model((X_train, y_train))
    
    # Batch evaluation
    logger.section("BATCH EVALUATION")
    
    # Process in batches to manage memory
    batch_size = 1000
    n_batches = len(X_test) // batch_size + 1
    
    all_predictions = []
    all_probabilities = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))
        
        if start_idx >= len(X_test):
            break
        
        batch_X = X_test[start_idx:end_idx]
        batch_probs = model.predict(batch_X, batch_size=256, verbose=0)
        batch_preds = (batch_probs >= 0.5).astype(int)
        
        all_probabilities.extend(batch_probs.flatten())
        all_predictions.extend(batch_preds.flatten())
        
        logger.info(f"Processed batch {i+1}/{n_batches} ({end_idx-start_idx} samples)")
    
    # Convert to arrays
    import numpy as np
    y_pred_prob = np.array(all_probabilities)
    y_pred = np.array(all_predictions)
    
    # Evaluate batch results
    evaluator = ModelEvaluator(config, logger)
    batch_results = evaluator._calculate_metrics(y_test, y_pred, y_pred_prob)
    
    logger.info(f"Batch evaluation completed")
    logger.info(f"Accuracy: {batch_results['Accuracy']}")
    logger.info(f"F1 Score: {batch_results['F1 Score']}")
    
    return batch_results


def cross_validation_example():
    """Example of cross-validation approach"""
    
    logger = Logger("IDS_CV_Example")
    config = Config("configs/default.json")
    
    # Note: This is a simplified example
    # In practice, you would implement proper k-fold cross-validation
    
    from sklearn.model_selection import StratifiedKFold
    
    # Load data
    data_loader = DataLoader(config, logger)
    train_df, test_df, data_info = data_loader.load_and_validate()
    
    # Preprocess
    preprocessor = DataPreprocessor(config, logger)
    X_train, X_test, y_train, y_test, class_weights, feature_names = preprocessor.preprocess_data(train_df, test_df)
    
    # Combine for cross-validation
    X_combined = np.concatenate([X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)])
    y_combined = np.concatenate([y_train, y_test])
    
    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
        logger.info(f"Training fold {fold + 1}/5")
        
        # Split data
        X_train_fold = X_combined[train_idx].reshape(-1, X_train.shape[1], 1)
        X_val_fold = X_combined[val_idx].reshape(-1, X_train.shape[1], 1)
        y_train_fold = y_combined[train_idx]
        y_val_fold = y_combined[val_idx]
        
        # Build and train model
        ids = IDSFramework(config, logger)
        model = ids.build_model((X_train_fold.shape[1], 1))
        
        # Train for fewer epochs in CV
        config.set("training.epochs", 20)
        training_info = ids.model_trainer.train_model(model, X_train_fold, y_train_fold, ids.output_dirs)
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(val_acc)
        
        logger.info(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}")
    
    # Calculate mean and std
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    logger.info(f"Cross-validation results:")
    logger.info(f"Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
    
    return cv_scores


def main():
    """Run all advanced examples"""
    
    print("Running Advanced IDS Framework Examples...")
    
    try:
        # Example 1: Advanced training
        print("\n1. Advanced Training Example")
        results, alerts = advanced_training_example()
        
        # Example 2: Ensemble learning
        print("\n2. Ensemble Learning Example")
        ensemble_models = ensemble_learning_example()
        
        # Example 3: Custom model
        print("\n3. Custom Model Example")
        custom_model = custom_model_example()
        
        # Example 4: Batch processing
        print("\n4. Batch Processing Example")
        batch_results = batch_processing_example()
        
        # Example 5: Cross-validation
        print("\n5. Cross-Validation Example")
        cv_scores = cross_validation_example()
        
        print("\nAll advanced examples completed successfully!")
        
    except Exception as e:
        print(f"Advanced examples failed: {e}")
        raise


if __name__ == "__main__":
    main()
