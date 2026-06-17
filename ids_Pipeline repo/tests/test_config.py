"""
Test configuration management
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ids_framework.utils.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test configuration
        self.test_config = {
            "data": {
                "dataset_path": "/test/path",
                "batch_size": 128,
                "random_seed": 42
            },
            "model": {
                "architecture": "lstm",
                "learning_rate": 0.001
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_default_config(self):
        """Test loading default configuration"""
        config = Config()
        
        self.assertIsInstance(config.config, dict)
        self.assertIn("data", config.config)
        self.assertIn("model", config.config)
        self.assertEqual(config.get("data.batch_size"), 256)
    
    def test_load_config_from_file(self):
        """Test loading configuration from file"""
        config = Config(self.config_path)
        
        self.assertEqual(config.get("data.dataset_path"), "/test/path")
        self.assertEqual(config.get("data.batch_size"), 128)
        self.assertEqual(config.get("model.architecture"), "lstm")
    
    def test_get_config_value(self):
        """Test getting configuration values"""
        config = Config(self.config_path)
        
        # Test existing key
        self.assertEqual(config.get("data.batch_size"), 128)
        
        # Test nested key
        self.assertEqual(config.get("model.learning_rate"), 0.001)
        
        # Test non-existent key with default
        self.assertEqual(config.get("non.existent.key", "default"), "default")
        
        # Test non-existent key without default
        self.assertIsNone(config.get("non.existent.key"))
    
    def test_set_config_value(self):
        """Test setting configuration values"""
        config = Config()
        
        # Set new value
        config.set("data.batch_size", 512)
        self.assertEqual(config.get("data.batch_size"), 512)
        
        # Set nested value
        config.set("new.nested.key", "value")
        self.assertEqual(config.get("new.nested.key"), "value")
    
    def test_save_config(self):
        """Test saving configuration"""
        config = Config()
        config.set("test.key", "test_value")
        
        save_path = os.path.join(self.temp_dir, "saved_config.json")
        config.save_config(save_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Load and verify content
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["test"]["key"], "test_value")
    
    def test_create_output_directories(self):
        """Test creating output directories"""
        config = Config()
        config.set("output.base_dir", self.temp_dir)
        
        dirs = config.create_output_directories()
        
        # Check all directories were created
        expected_dirs = ["base", "figures", "models", "reports", "blockchain", "logs", "eda", "evaluation", "training"]
        for dir_key in expected_dirs:
            self.assertIn(dir_key, dirs)
            self.assertTrue(os.path.exists(dirs[dir_key]))


if __name__ == "__main__":
    unittest.main()
