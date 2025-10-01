# Development Setup Guide

## 🔧 Prerequisites

- **Python 3.8+** (Recommended: Python 3.12)
- **Git** for version control
- **Internet connection** for Yahoo Finance data

## 🚀 Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/srinidhim3/strategy-forge.git
cd strategy-forge
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux  
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```python
# Test the financial calculator
from src.data.financial_fetcher import FinancialFetcher
from src.models.financial_calculator import FinancialCalculator

# Fetch sample data
fetcher = FinancialFetcher()
statements = fetcher.fetch_all_statements('AAPL')

# Calculate metrics
calc = FinancialCalculator()
metrics = calc.calculate_all_metrics(statements)
print(f"ROE: {metrics.loc['ROE'].iloc[0]:.2%}")
```

## 📁 Project Structure

```
strategy-forge/
├── src/
│   ├── data/           # Data fetching and processing
│   │   ├── financial_fetcher.py
│   │   ├── price_fetcher.py
│   │   └── data_aligner.py      # Task 12 (Coming)
│   ├── models/         # Business logic and algorithms
│   │   ├── financial_calculator.py
│   │   ├── ratio_calculator.py  # Task 13 (Coming)
│   │   └── strategies.py        # Task 15 (Coming)
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── docs/               # Documentation
├── requirements.txt    # Python dependencies
└── README.md          # Project overview
```

## 🧪 Testing

### Run Financial Calculator Tests
```bash
cd tests/test_models/
python -m pytest test_financial_calculator.py -v
```

### Manual Testing with Real Data
```python
# Test with different stocks
symbols = ['AAPL', 'MSFT', 'GOOGL']
for symbol in symbols:
    statements = fetcher.fetch_all_statements(symbol)
    metrics = calc.calculate_all_metrics(statements)
    print(f"{symbol} ROE: {metrics.loc['ROE'].iloc[0]:.2%}")
```

## 🔍 Development Workflow

### 1. Feature Development
- Create feature branch: `git checkout -b feature/task-12-data-aligner`
- Implement functionality
- Add tests
- Update documentation

### 2. Testing & Validation
- Run unit tests: `pytest`
- Test with real data
- Verify accuracy against benchmarks

### 3. Integration
- Update TASK_PROGRESS.md
- Commit with descriptive message
- Push to remote repository

## 📊 Current Capabilities

### ✅ What Works Now
- **Financial Data Fetching**: Yahoo Finance integration
- **Financial Metrics**: 12 calculated ratios with TTM support
- **High Accuracy**: 6/7 metrics within ±5% industry tolerance

### 🚧 What's Coming (Task 12)
- **Data Alignment Pipeline**: Point-in-time accurate datasets
- **Ratio Calculator**: Daily P/E, P/B, PEG calculations
- **Strategy Framework**: Trading signal generation

## 🐛 Troubleshooting

### Common Issues

**1. yfinance Connection Errors**
```python
# Retry mechanism
import time
for attempt in range(3):
    try:
        data = yf.download('AAPL')
        break
    except:
        time.sleep(2)
```

**2. Missing Financial Data**
```python
# Check data availability
statements = fetcher.fetch_all_statements('AAPL')
print(f"Income Statement Shape: {statements['income_statement'].shape}")
```

**3. Virtual Environment Issues**
```bash
# Recreate environment
deactivate
rm -rf venv
python -m venv venv
```

## 🔧 IDE Setup

### VS Code Extensions (Recommended)
- Python
- Pylance  
- Python Docstring Generator
- GitLens

### PyCharm Configuration
- Enable virtual environment in Project Settings
- Configure pytest as test runner
- Set up Code Style (PEP 8)

---

**Need Help?** Check the [Task Progress](TASK_PROGRESS.md) for current status or review [Architecture](ARCHITECTURE.md) for system overview.