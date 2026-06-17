"""
Main entry point for IDS Framework
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ids_framework import IDSFramework
from ids_framework.utils import Config, Logger
from ids_framework.utils.helpers import create_run_id, setup_gpu_memory_growth


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Intrusion Detection System Framework")
    parser.add_argument("--config", type=str, default="configs/default.json",
                       help="Path to configuration file")
    parser.add_argument("--data-path", type=str, 
                       help="Override dataset path")
    parser.add_argument("--output-dir", type=str, default="project_outputs",
                       help="Output directory")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict", "full"],
                       default="full", help="Operation mode")
    parser.add_argument("--model-path", type=str,
                       help="Path to trained model (for evaluate/predict)")
    parser.add_argument("--gpu", action="store_true", default=True,
                       help="Use GPU if available")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup GPU
    if args.gpu:
        setup_gpu_memory_growth()
    
    # Initialize logger
    log_level = "DEBUG" if args.verbose else "INFO"
    run_id = create_run_id()
    logger = Logger(f"IDS_Framework_{run_id}", log_level=log_level)
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Override config with command line arguments
        if args.data_path:
            config.set("data.dataset_path", args.data_path)
        if args.output_dir:
            config.set("output.base_dir", args.output_dir)
        
        # Create output directories
        output_dirs = config.create_output_directories()
        
        # Initialize IDS Framework
        ids = IDSFramework(config, logger)
        
        if args.mode == "train":
            logger.section("TRAINING MODE")
            train_data, test_data = ids.load_data()
            model = ids.train_model(train_data)
            ids.save_model(model, os.path.join(output_dirs["models"], "final_model.h5"))
            
        elif args.mode == "evaluate":
            logger.section("EVALUATION MODE")
            if not args.model_path:
                raise ValueError("Model path required for evaluation mode")
            
            train_data, test_data = ids.load_data()
            model = ids.load_model(args.model_path)
            results = ids.evaluate_model(model, test_data)
            
        elif args.mode == "predict":
            logger.section("PREDICTION MODE")
            if not args.model_path:
                raise ValueError("Model path required for prediction mode")
            
            model = ids.load_model(args.model_path)
            # Implementation for prediction mode
            logger.info("Prediction mode not yet implemented")
            
        else:  # full mode
            logger.section("FULL PIPELINE")
            train_data, test_data = ids.load_data()
            model = ids.train_model(train_data)
            results = ids.evaluate_model(model, test_data)
            alerts = ids.log_attack_alerts(model, test_data)
            
            # Generate final report
            ids.generate_comprehensive_report(results, alerts, output_dirs)
        
        logger.section("COMPLETED SUCCESSFULLY")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {output_dirs['base']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
