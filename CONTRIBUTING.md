# Contributing to Strategy Forge

Thank you for your interest in contributing to Strategy Forge! This document provides guidelines for contributing to the project.

## 🎯 How to Contribute

### Reporting Issues
- Use the [GitHub Issues](https://github.com/srinidhim3/strategy-forge/issues) page
- Provide clear description of the problem
- Include steps to reproduce
- Mention your environment (OS, Python version, etc.)

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Provide use cases and examples

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Follow coding standards
7. Update documentation
8. Submit a pull request

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small

### Testing
- Write unit tests for new features
- Maintain test coverage above 80%
- Use pytest for testing framework
- Include integration tests for critical paths

### Documentation
- Update README.md for new features
- Add docstrings with examples
- Update API documentation
- Include inline comments for complex logic

## 🛠️ Development Setup

### Prerequisites
- Python 3.12+
- Git
- Virtual environment tools

### Setup Steps
```bash
# Clone your fork
git clone https://github.com/your-username/strategy-forge.git
cd strategy-forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Run tests
pytest

# Run linting
flake8 src/
black src/
```

## 🎨 Coding Standards

### Python Code
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio for a return series.
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sharpe ratio as a float
        
    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
```

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense
- Keep first line under 50 characters
- Add detailed description if needed

Examples:
```
Add portfolio rebalancing functionality
Fix data alignment bug in financial calculator
Update README with installation instructions
```

## 🧪 Testing Guidelines

### Unit Tests
```python
import pytest
from src.models.financial_calculator import FinancialCalculator

class TestFinancialCalculator:
    def test_calculate_pe_ratio(self):
        calculator = FinancialCalculator()
        result = calculator.calculate_pe_ratio(price=100, eps=5)
        assert result == 20
        
    def test_calculate_pe_ratio_zero_eps(self):
        calculator = FinancialCalculator()
        result = calculator.calculate_pe_ratio(price=100, eps=0)
        assert result is None
```

### Integration Tests
```python
def test_full_pipeline():
    """Test the complete data processing pipeline."""
    pipeline = DataPipeline()
    result = pipeline.process_stock("AAPL", "2023-01-01", "2023-12-31")
    
    assert not result.empty
    assert "pe_ratio" in result.columns
    assert result.index.is_monotonic_increasing
```

## 📚 Documentation

### Function Documentation
```python
def backtest_strategy(data: pd.DataFrame, strategy: Strategy, 
                     initial_capital: float = 100000) -> BacktestResults:
    """
    Run a backtest on historical data using the specified strategy.
    
    Args:
        data: Historical price and fundamental data
        strategy: Trading strategy to test
        initial_capital: Starting capital in currency units
        
    Returns:
        BacktestResults object containing performance metrics
        
    Raises:
        ValueError: If data is empty or invalid
        
    Example:
        >>> data = get_stock_data("AAPL", "2020-01-01", "2023-12-31")
        >>> strategy = PERatioStrategy(threshold=15)
        >>> results = backtest_strategy(data, strategy)
        >>> print(f"Total Return: {results.total_return:.2%}")
    """
```

## 🚀 Release Process

### Version Numbers
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Release notes prepared

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## 📞 Questions?

- Open an issue for technical questions
- Email maintainers for private matters
- Join our Discord community for discussions

Thank you for contributing to Strategy Forge! 🎯