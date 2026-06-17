"""
Basic usage example for IDS Framework
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ids_framework import IDSFramework
from ids_framework.utils import Config, Logger


def main():
    """Basic usage example"""
    
    # Initialize logger
    logger = Logger("IDS_Example", log_level="INFO")
    
    # Load configuration
    config = Config("configs/default.json")
    
    # Update configuration for your environment
    config.set("data.dataset_path", "/path/to/nslkdd")  # Update this path
    config.set("output.base_dir", "example_outputs")
    
    # Initialize IDS Framework
    ids = IDSFramework(config, logger)
    
    try:
        # Run the complete pipeline
        results = ids.run_full_pipeline()
        
        print("Pipeline completed successfully!")
        print(f"Run ID: {results['run_id']}")
        print(f"Output directory: {results['output_dirs']['base']}")
        print(f"Model saved at: {results['model_path']}")
        print(f"Total runtime: {results['total_runtime']:.2f} seconds")
        
        # Print key metrics
        metrics = results['evaluation_results']['metrics']
        print(f"\nKey Performance Metrics:")
        print(f"  Accuracy: {metrics['Accuracy']}")
        print(f"  Precision: {metrics['Precision']}")
        print(f"  Recall: {metrics['Recall (TPR)']}")
        print(f"  F1 Score: {metrics['F1 Score']}")
        print(f"  AUC-ROC: {metrics['AUC-ROC']}")
        
        # Print blockchain stats
        blockchain_stats = results['alert_results']['blockchain_stats']
        print(f"\nBlockchain Alert System:")
        print(f"  Total blocks: {blockchain_stats['total_blocks']}")
        print(f"  Alerts logged: {results['alert_results']['alerts_logged']}")
        print(f"  Chain valid: {blockchain_stats['is_valid']}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
