#!/usr/bin/env python3
"""
Environment setup script for IDS Framework
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Error: Python 3.8+ required, found Python {version.major}.{version.minor}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_gpu():
    """Check GPU availability"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU detected: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True
        else:
            print("⚠ No GPU detected - training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠ TensorFlow not installed - cannot check GPU")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models",
        "outputs",
        "logs",
        "reports",
        "figures"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def setup_gitignore():
    """Create .gitignore file if not exists"""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
project_outputs/
*.h5
*.pkl
*.joblib
models/
logs/
figures/
reports/
blockchain/

# Data
data/
*.txt
*.arff

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
"""
        with open(gitignore_path, "w") as f:
            f.write(gitignore_content.strip())
        print("✓ Created .gitignore file")


def verify_installation():
    """Verify installation by importing key modules"""
    modules_to_test = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "TQDM"),
        ("joblib", "Joblib")
    ]
    
    failed_imports = []
    
    for module, name in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError:
            print(f"✗ {name} import failed")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    print("\n✓ All modules imported successfully")
    return True


def download_sample_data():
    """Download sample NSL-KDD data (placeholder)"""
    print("Note: NSL-KDD dataset must be downloaded manually")
    print("Download from: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html")
    print("Extract to: data/ directory")
    print("Files needed: KDDTrain+.txt, KDDTest+.txt")


def run_tests():
    """Run basic tests to verify installation"""
    print("Running basic tests...")
    
    try:
        # Test configuration
        from src.ids_framework.utils.config import Config
        config = Config()
        print("✓ Configuration module working")
        
        # Test logger
        from src.ids_framework.utils.logger import Logger
        logger = Logger("test")
        print("✓ Logger module working")
        
        # Test helper functions
        from src.ids_framework.utils.helpers import create_run_id
        run_id = create_run_id()
        print(f"✓ Helper functions working (run_id: {run_id})")
        
        print("✓ All basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup IDS Framework environment")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--gpu-only", action="store_true", help="Only check GPU availability")
    parser.add_argument("--test-only", action="store_true", help="Only run verification tests")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IDS Framework Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU if requested
    if args.gpu_only:
        check_gpu()
        return
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("Dependency installation failed. Please install manually.")
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup gitignore
    setup_gitignore()
    
    # Check GPU
    check_gpu()
    
    # Verify installation
    if not verify_installation():
        print("Installation verification failed. Please check error messages above.")
        sys.exit(1)
    
    # Run tests if not test-only mode
    if not args.test_only:
        if not run_tests():
            print("Basic tests failed. Please check installation.")
            sys.exit(1)
    
    # Download instructions
    download_sample_data()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download NSL-KDD dataset to data/ directory")
    print("2. Update config file with dataset path")
    print("3. Run: python -m ids_framework --help")
    print("4. Try examples: python examples/basic_usage.py")


if __name__ == "__main__":
    main()
