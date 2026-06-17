# Contributing to IDS Framework

Thank you for your interest in contributing to the Advanced Intrusion Detection System Framework! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/ids-framework/ids-framework/issues) page
- Search existing issues before creating a new one
- Provide clear, descriptive titles and detailed descriptions
- Include steps to reproduce, expected behavior, and actual behavior
- Add relevant logs, screenshots, or code examples

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request with a clear description

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/ids-framework.git
cd ids-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ids_framework --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## 📝 Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstrings
- Include parameter types and return types
- Add usage examples for complex functions

```python
def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> tf.keras.Model:
    """Train the LSTM model on the provided data.
    
    Args:
        X_train: Training features of shape (n_samples, n_features, 1)
        y_train: Training labels of shape (n_samples,)
        
    Returns:
        Trained Keras model
        
    Raises:
        ValueError: If input data is invalid
        
    Example:
        >>> model = trainer.train_model(X_train, y_train)
        >>> print(f"Model trained with {len(X_train)} samples")
    """
```

### Naming Conventions

- **Classes**: PascalCase (`ModelTrainer`, `DataLoader`)
- **Functions/Variables**: snake_case (`train_model`, `batch_size`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_BATCH_SIZE`, `MAX_EPOCHS`)
- **Private members**: underscore prefix (`_internal_method`, `_private_var`)

### Error Handling

- Use specific exception types
- Include meaningful error messages
- Log errors appropriately
- Handle exceptions gracefully

```python
try:
    result = some_operation()
except ValueError as e:
    self.logger.error(f"Invalid input in operation: {e}")
    raise
except Exception as e:
    self.logger.error(f"Unexpected error: {e}")
    raise RuntimeError("Operation failed") from e
```

## 🧪 Testing Guidelines

### Test Structure

- Unit tests in `tests/` directory
- Test files named `test_*.py`
- Test classes named `Test*`
- Test methods named `test_*`

### Writing Tests

```python
import unittest
from ids_framework.utils import Config

class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
    
    def test_get_config_value(self):
        """Test getting configuration values."""
        value = self.config.get("data.batch_size")
        self.assertEqual(value, 256)
    
    def test_set_config_value(self):
        """Test setting configuration values."""
        self.config.set("data.batch_size", 512)
        value = self.config.get("data.batch_size")
        self.assertEqual(value, 512)
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Use mock objects for external dependencies
- Test both success and failure paths

## 📚 Documentation

### API Documentation

- Use docstrings for all public APIs
- Include parameter descriptions
- Provide return value information
- Add usage examples

### User Documentation

- Update README.md for new features
- Add examples to `examples/` directory
- Update configuration documentation
- Include troubleshooting guides

## 🔄 Pull Request Process

### Before Submitting

1. Run all tests and ensure they pass
2. Check code formatting with black
3. Verify code style with flake8
4. Update documentation if needed
5. Add tests for new functionality

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

1. Automated checks (tests, linting)
2. Code review by maintainers
3. Discussion and feedback
4. Approval and merge

## 🏗️ Architecture Guidelines

### Module Structure

- Keep modules focused and single-purpose
- Use clear interfaces between components
- Minimize circular dependencies
- Follow dependency injection patterns

### Configuration

- Use configuration files for parameters
- Provide sensible defaults
- Validate configuration values
- Document all configuration options

### Logging

- Use structured logging
- Include relevant context
- Use appropriate log levels
- Avoid sensitive information in logs

## 🚀 Release Process

### Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version numbers in setup.py
- Create release notes
- Tag releases in Git

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version numbers updated
- [ ] Changelog updated
- [ ] Release notes prepared

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain professional discussions

### Communication

- Use GitHub issues for bug reports
- Use discussions for questions
- Be patient with response times
- Provide clear and concise messages

## 📌 Getting Help

### Resources

- [Documentation](https://ids-framework.readthedocs.io/)
- [GitHub Issues](https://github.com/ids-framework/ids-framework/issues)
- [GitHub Discussions](https://github.com/ids-framework/ids-framework/discussions)

### Contact

- Email: contact@ids-framework.com
- Discord: [Join our community](https://discord.gg/ids-framework)

## 🙏 Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes
- Documentation acknowledgments

Thank you for contributing to the IDS Framework! 🎉
