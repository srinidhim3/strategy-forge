# Strategy Forge CLI Examples

This file contains comprehensive examples demonstrating all features of the Strategy Forge CLI (Task 18).

## 🚀 Basic Usage Examples

### 1. List Available Strategies
```bash
python main.py --list-strategies
```
**Output:**
```
Available Trading Strategies
==================================================

pe_threshold
   Description: Buy when P/E ratio below threshold, sell when above
   Default Config:
     buy_pe_threshold: 15.0
     sell_pe_threshold: 25.0
     min_signal_strength: 0.1

moving_average
   Description: Buy/sell based on moving average crossovers
   Default Config:
     short_window: 20
     long_window: 50
     min_signal_strength: 0.2
     ma_type: SMA
     price_column: Close

combined
   Description: Combines P/E and moving average signals
   Default Config:
     buy_pe_threshold: 18.0
     sell_pe_threshold: 28.0
     short_window: 20
     long_window: 50
     min_signal_strength: 0.15
```

### 2. Get Help Documentation
```bash
python main.py --help
```

### 3. Simple Backtest
```bash
python main.py --symbol AAPL --strategy pe_threshold
```

## 🎯 Strategy-Specific Examples

### P/E Threshold Strategy
```bash
# Basic P/E strategy with default parameters
python main.py --symbol AAPL --strategy pe_threshold

# Custom P/E thresholds
python main.py --symbol AAPL --strategy pe_threshold \
               --start-date 2022-01-01 --end-date 2023-12-31

# Conservative P/E approach
python main.py --symbol MSFT --strategy pe_threshold \
               --initial-capital 500000 --position-size 0.10
```

### Moving Average Strategy
```bash
# Default moving average crossover
python main.py --symbol GOOGL --strategy moving_average \
               --start-date 2022-01-01 --end-date 2023-06-30

# Long-term trend following
python main.py --symbol TSLA --strategy moving_average \
               --initial-capital 1000000 --position-size 0.20
```

### Combined Strategy
```bash
# Fundamental + Technical analysis
python main.py --symbol NVDA --strategy combined \
               --start-date 2022-01-01 --end-date 2023-12-31

# Advanced combined strategy with metrics
python main.py --symbol AMZN --strategy combined \
               --advanced-metrics --benchmark
```

## 📊 Output Format Examples

### Console Output (Default)
```bash
python main.py --symbol AAPL --strategy pe_threshold
```

### JSON Export
```bash
# Export to JSON file
python main.py --symbol AAPL --strategy pe_threshold \
               --output-format json --output-file aapl_results.json

# JSON output to console
python main.py --symbol AAPL --strategy pe_threshold \
               --output-format json
```

### CSV Export
```bash
# Export to CSV file
python main.py --symbol MSFT --strategy moving_average \
               --output-format csv --output-file msft_analysis.csv
```

### Multiple Formats
```bash
# Generate all formats (console + JSON + CSV)
python main.py --symbol GOOGL --strategy combined \
               --output-format all --output-file googl_comprehensive
```

## ⚙️ Configuration Examples

### Custom Trading Parameters
```bash
# High-frequency trading setup
python main.py --symbol AAPL --strategy pe_threshold \
               --position-size 0.05 --commission 0.001 \
               --slippage 0.0001

# Conservative large position
python main.py --symbol MSFT --strategy pe_threshold \
               --initial-capital 5000000 --position-size 0.30 \
               --commission 0.01
```

### Risk Management
```bash
# Tight risk controls
python main.py --symbol TSLA --strategy moving_average \
               --stop-loss 0.03 --take-profit 0.10 \
               --position-size 0.08

# Loose risk controls for long-term holding
python main.py --symbol AAPL --strategy pe_threshold \
               --stop-loss 0.15 --take-profit 0.40 \
               --position-size 0.25
```

### Date Range Specifications
```bash
# Short-term analysis
python main.py --symbol NVDA --strategy combined \
               --start-date 2023-01-01 --end-date 2023-06-30

# Long-term analysis
python main.py --symbol GOOGL --strategy pe_threshold \
               --start-date 2020-01-01 --end-date 2023-12-31

# Recent performance
python main.py --symbol AMZN --strategy moving_average \
               --start-date 2023-06-01 --end-date 2023-12-31
```

## 🔬 Advanced Analytics Examples

### Comprehensive Analysis
```bash
# Full analysis with all advanced features
python main.py --symbol AAPL --strategy combined \
               --start-date 2022-01-01 --end-date 2023-12-31 \
               --initial-capital 1000000 --position-size 0.20 \
               --advanced-metrics --benchmark \
               --output-format all --output-file aapl_full_analysis \
               --verbose
```

### Performance Benchmarking
```bash
# Compare strategy vs buy-and-hold
python main.py --symbol MSFT --strategy pe_threshold \
               --benchmark --advanced-metrics \
               --output-format json --output-file msft_vs_benchmark.json

# Rolling performance analysis
python main.py --symbol GOOGL --strategy moving_average \
               --advanced-metrics --verbose
```

### Risk Analysis
```bash
# Value at Risk analysis
python main.py --symbol TSLA --strategy combined \
               --advanced-metrics --output-format json

# Stress testing with tight stops
python main.py --symbol NVDA --strategy pe_threshold \
               --stop-loss 0.02 --advanced-metrics --benchmark
```

## 🛠️ Configuration File Examples

### Create Strategy Configuration File
Create `conservative_pe.json`:
```json
{
  "buy_pe_threshold": 12.0,
  "sell_pe_threshold": 20.0,
  "min_signal_strength": 0.15
}
```

Use with CLI:
```bash
python main.py --symbol AAPL --strategy pe_threshold \
               --strategy-config conservative_pe.json
```

### Moving Average Configuration
Create `fast_ma.json`:
```json
{
  "short_window": 5,
  "long_window": 15,
  "min_signal_strength": 0.1,
  "ma_type": "SMA",
  "price_column": "Close"
}
```

Use with CLI:
```bash
python main.py --symbol TSLA --strategy moving_average \
               --strategy-config fast_ma.json
```

### Combined Strategy Configuration
Create `balanced_combined.json`:
```json
{
  "buy_pe_threshold": 15.0,
  "sell_pe_threshold": 25.0,
  "short_window": 10,
  "long_window": 30,
  "min_signal_strength": 0.2
}
```

Use with CLI:
```bash
python main.py --symbol GOOGL --strategy combined \
               --strategy-config balanced_combined.json \
               --advanced-metrics --benchmark
```

## 🔄 Batch Processing Examples

### Multiple Stock Analysis Script
Create `batch_analysis.sh` (Linux/Mac) or `batch_analysis.bat` (Windows):

**Linux/Mac:**
```bash
#!/bin/bash
STOCKS=("AAPL" "MSFT" "GOOGL" "TSLA" "NVDA")
for stock in "${STOCKS[@]}"; do
    echo "Analyzing $stock..."
    python main.py --symbol $stock --strategy combined \
                   --advanced-metrics --benchmark \
                   --output-format json --output-file "${stock}_analysis.json"
done
```

**Windows:**
```batch
@echo off
for %%S in (AAPL MSFT GOOGL TSLA NVDA) do (
    echo Analyzing %%S...
    python main.py --symbol %%S --strategy combined ^
                   --advanced-metrics --benchmark ^
                   --output-format json --output-file "%%S_analysis.json"
)
```

### Strategy Comparison
```bash
# Compare all strategies on same stock
python main.py --symbol AAPL --strategy pe_threshold \
               --output-format json --output-file aapl_pe.json

python main.py --symbol AAPL --strategy moving_average \
               --output-format json --output-file aapl_ma.json

python main.py --symbol AAPL --strategy combined \
               --output-format json --output-file aapl_combined.json
```

## 🚨 Error Handling Examples

### Common Error Scenarios

#### Missing Required Arguments
```bash
# This will fail - missing strategy
python main.py --symbol AAPL
# Output: ERROR: Strategy is required for backtesting operations
```

#### Invalid Date Format
```bash
# This will fail - wrong date format
python main.py --symbol AAPL --strategy pe_threshold --start-date 01/01/2022
# Output: ERROR: Start date must be in YYYY-MM-DD format
```

#### Invalid Parameters
```bash
# This will fail - position size > 1
python main.py --symbol AAPL --strategy pe_threshold --position-size 1.5
# Output: ERROR: Position size must be between 0 and 1
```

### Debugging Commands
```bash
# Enable verbose logging for debugging
python main.py --symbol AAPL --strategy pe_threshold --verbose

# Check available strategies
python main.py --list-strategies

# Get comprehensive help
python main.py --help
```

## 📈 Performance Testing Examples

### Quick Test (Short Time Period)
```bash
python main.py --symbol AAPL --strategy pe_threshold \
               --start-date 2023-10-01 --end-date 2023-12-31
```

### Comprehensive Test (Long Time Period)
```bash
python main.py --symbol AAPL --strategy combined \
               --start-date 2020-01-01 --end-date 2023-12-31 \
               --advanced-metrics --benchmark --verbose
```

### Memory Efficient Test
```bash
python main.py --symbol AAPL --strategy pe_threshold \
               --start-date 2023-01-01 --end-date 2023-03-31 \
               --output-format csv
```

## 🎯 Real-World Use Cases

### Portfolio Manager Analysis
```bash
# Analyze large-cap growth stock
python main.py --symbol AAPL --strategy combined \
               --initial-capital 10000000 --position-size 0.15 \
               --advanced-metrics --benchmark \
               --output-format json --output-file portfolio_aapl.json
```

### Quantitative Research
```bash
# Research P/E effectiveness
python main.py --symbol MSFT --strategy pe_threshold \
               --start-date 2015-01-01 --end-date 2023-12-31 \
               --advanced-metrics --benchmark \
               --output-format all --output-file pe_research
```

### Risk Management Testing
```bash
# Test strategy under tight risk controls
python main.py --symbol TSLA --strategy moving_average \
               --stop-loss 0.02 --take-profit 0.08 \
               --position-size 0.05 --advanced-metrics \
               --output-format json --output-file risk_test.json
```

### Client Reporting
```bash
# Generate client-ready analysis
python main.py --symbol GOOGL --strategy combined \
               --start-date 2022-01-01 --end-date 2023-12-31 \
               --advanced-metrics --benchmark \
               --output-format all --output-file client_googl_report
```

---

## 💡 Tips and Best Practices

### 1. Start Simple
- Begin with `--list-strategies` to understand available options
- Use default parameters first, then customize
- Test with short date ranges initially

### 2. Progressive Enhancement
- Start with console output, then move to JSON/CSV
- Add `--verbose` for debugging
- Use `--advanced-metrics` for detailed analysis

### 3. Performance Optimization
- Use shorter date ranges for initial testing
- Avoid very small position sizes that might cause issues
- Consider network latency for data fetching

### 4. Output Management
- Use descriptive filenames for `--output-file`
- Organize outputs in dated folders
- Use `--output-format all` for comprehensive analysis

### 5. Error Prevention
- Always use YYYY-MM-DD date format
- Keep position sizes between 0.01 and 1.0
- Ensure symbol exists before running analysis

**Strategy Forge CLI** - Your comprehensive tool for quantitative trading strategy analysis! 🚀