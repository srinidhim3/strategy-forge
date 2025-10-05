# Strategy Forge CLI Usage Guide

## 📋 Overview

The Strategy Forge CLI provides a comprehensive command-line interface for single-stock backtesting operations. This guide covers all features, usage patterns, and examples.

## 🚀 Quick Start

### Basic Commands
```bash
# List available strategies
python main.py --list-strategies

# Get help and documentation
python main.py --help

# Run a simple backtest
python main.py --symbol AAPL --strategy pe_threshold
```

## 📖 Command Reference

### Required Arguments
- `--symbol` / `-s`: Stock symbol to analyze (e.g., AAPL, MSFT, GOOGL)
- `--strategy` / `-st`: Trading strategy to use

### Optional Arguments

#### Date Range
- `--start-date`: Start date for analysis (YYYY-MM-DD format, default: 2 years ago)
- `--end-date`: End date for analysis (YYYY-MM-DD format, default: today)

#### Trading Parameters
- `--initial-capital`: Initial capital for backtesting (default: $1,000,000)
- `--position-size`: Position size as fraction of capital (default: 0.15 = 15%)
- `--commission`: Commission per share in currency units (default: 0.005)
- `--slippage`: Slippage as fraction of price (default: 0.0003 = 3 basis points)

#### Risk Management
- `--stop-loss`: Stop loss threshold as fraction (default: 0.08 = 8%)
- `--take-profit`: Take profit threshold as fraction (default: 0.20 = 20%)

#### Output Control
- `--output-format`: Output format - console, json, csv, all (default: console)
- `--output-file`: File path for saving results (auto-generated if not specified)

#### Advanced Options
- `--strategy-config`: JSON string or file path for strategy configuration
- `--advanced-metrics`: Calculate advanced performance metrics (VaR, Information Ratio, etc.)
- `--benchmark`: Include benchmark (buy & hold) comparison
- `--verbose` / `-v`: Enable verbose logging

#### Utility Commands
- `--list-strategies`: List available strategies and exit
- `--help` / `-h`: Show help message and exit

## 🎯 Available Strategies

### 1. P/E Threshold Strategy (`pe_threshold`)
**Description**: Buy when P/E ratio below threshold, sell when above

**Default Configuration**:
```json
{
  "buy_pe_threshold": 15.0,
  "sell_pe_threshold": 25.0,
  "min_signal_strength": 0.1
}
```

**Example**:
```bash
python main.py --symbol AAPL --strategy pe_threshold \
               --strategy-config '{"buy_pe_threshold": 12.0, "sell_pe_threshold": 22.0}'
```

### 2. Moving Average Strategy (`moving_average`)
**Description**: Buy/sell based on moving average crossovers

**Default Configuration**:
```json
{
  "short_window": 20,
  "long_window": 50,
  "min_signal_strength": 0.2,
  "ma_type": "SMA",
  "price_column": "Close"
}
```

**Example**:
```bash
python main.py --symbol MSFT --strategy moving_average \
               --start-date 2022-01-01 --end-date 2023-12-31
```

### 3. Combined Strategy (`combined`)
**Description**: Combines P/E and moving average signals

**Default Configuration**:
```json
{
  "buy_pe_threshold": 18.0,
  "sell_pe_threshold": 28.0,
  "short_window": 20,
  "long_window": 50,
  "min_signal_strength": 0.15
}
```

**Example**:
```bash
python main.py --symbol GOOGL --strategy combined \
               --advanced-metrics --benchmark
```

## 📊 Output Formats

### Console Output (Default)
Formatted table output with key performance metrics:
```
🎯 STRATEGY FORGE - BACKTEST RESULTS
============================================================
📊 Symbol: AAPL
🎯 Strategy: PE Threshold Strategy
📅 Period: 2022-01-01 to 2023-12-31
💰 Initial Capital: $1,000,000.00
💎 Final Value: $1,150,000.00

📈 PERFORMANCE SUMMARY
------------------------------
Total Return: 15.00%
Annualized Return: 15.00%
Sharpe Ratio: 1.200
Max Drawdown: 8.50%
Volatility: 12.30%

📊 TRADING STATISTICS
------------------------------
Total Trades: 25
Winning Trades: 15
Win Rate: 60.0%
Average Win: 2.50%
Average Loss: -1.50%
Profit Factor: 1.67
```

### JSON Output
Structured data export for programmatic use:
```json
{
  "metadata": {
    "symbol": "AAPL",
    "strategy": "PE Threshold Strategy",
    "start_date": "2022-01-01",
    "end_date": "2023-12-31",
    "generated_at": "2025-10-02T21:30:00"
  },
  "performance": {
    "initial_capital": 1000000.0,
    "final_value": 1150000.0,
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": 0.085
  },
  "trading": {
    "total_trades": 25,
    "winning_trades": 15,
    "win_rate": 0.6,
    "profit_factor": 1.67
  }
}
```

### CSV Output
Tabular data for spreadsheet analysis:
```csv
metric,value
symbol,AAPL
strategy,PE Threshold Strategy
total_return,0.15
sharpe_ratio,1.2
max_drawdown,0.085
total_trades,25
win_rate,0.6
profit_factor,1.67
```

## 🔧 Configuration Examples

### Using JSON String Configuration
```bash
python main.py --symbol AAPL --strategy pe_threshold \
               --strategy-config '{"buy_pe_threshold": 10.0, "sell_pe_threshold": 30.0}'
```

### Using Configuration File
Create a file `my_strategy.json`:
```json
{
  "buy_pe_threshold": 12.0,
  "sell_pe_threshold": 25.0,
  "min_signal_strength": 0.15
}
```

Run with file:
```bash
python main.py --symbol AAPL --strategy pe_threshold \
               --strategy-config my_strategy.json
```

## 📈 Advanced Usage Examples

### Comprehensive Analysis
```bash
python main.py --symbol AAPL --strategy combined \
               --start-date 2022-01-01 --end-date 2023-12-31 \
               --initial-capital 1000000 --position-size 0.20 \
               --commission 0.01 --slippage 0.0005 \
               --stop-loss 0.05 --take-profit 0.25 \
               --advanced-metrics --benchmark \
               --output-format all --output-file aapl_analysis \
               --verbose
```

### Multiple Output Formats
```bash
# Generate JSON and CSV files
python main.py --symbol MSFT --strategy moving_average \
               --output-format all --output-file msft_results
```

### High-Frequency Strategy Testing
```bash
python main.py --symbol TSLA --strategy pe_threshold \
               --position-size 0.05 --commission 0.001 \
               --stop-loss 0.03 --take-profit 0.10
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Strategy Configuration Errors
**Error**: `Failed to parse strategy config`
**Solution**: Ensure JSON is properly formatted with double quotes
```bash
# Correct
--strategy-config '{"buy_pe_threshold": 15.0}'
# Incorrect
--strategy-config '{buy_pe_threshold: 15.0}'
```

#### 2. Date Format Errors
**Error**: `Start date must be in YYYY-MM-DD format`
**Solution**: Use proper date format
```bash
# Correct
--start-date 2022-01-01
# Incorrect
--start-date 01/01/2022
```

#### 3. Missing Required Arguments
**Error**: `Symbol is required for backtesting operations`
**Solution**: Provide both symbol and strategy
```bash
# Correct
python main.py --symbol AAPL --strategy pe_threshold
# Incorrect
python main.py --symbol AAPL
```

#### 4. Insufficient Data
**Error**: `Insufficient data for moving average crossover signals`
**Solution**: Use longer time period or smaller moving average windows
```bash
# Use longer period
--start-date 2021-01-01 --end-date 2023-12-31
# Or adjust strategy config for smaller windows
--strategy-config '{"short_window": 5, "long_window": 15}'
```

### Debugging Tips

#### Enable Verbose Logging
```bash
python main.py --symbol AAPL --strategy pe_threshold --verbose
```

#### Check Available Strategies
```bash
python main.py --list-strategies
```

#### Validate Arguments
```bash
python main.py --help
```

## 📁 Output File Management

### Default File Naming
- Console output: No file generated
- JSON output: `backtest_results_YYYYMMDD_HHMMSS.json`
- CSV output: `backtest_results_YYYYMMDD_HHMMSS.csv`

### Custom File Naming
```bash
# Specify base filename (extension auto-added)
--output-file my_results

# Generates:
# my_results.json (for JSON format)
# my_results.csv (for CSV format)
```

### Output Directory
All files are saved to the `output/` directory (created automatically)

## 🚀 Performance Tips

### 1. Optimize Date Ranges
- Use shorter periods for initial testing
- Extend ranges for comprehensive analysis

### 2. Strategy Selection
- Use `pe_threshold` for fundamental analysis
- Use `moving_average` for technical analysis
- Use `combined` for comprehensive signals

### 3. Parameter Tuning
- Start with default parameters
- Adjust gradually based on results
- Use configuration files for complex setups

## 📞 Support

### Getting Help
```bash
# CLI help
python main.py --help

# Strategy information
python main.py --list-strategies

# Documentation
# Check README.md and docs/ directory
```

### Common Commands Summary
```bash
# Quick start
python main.py --list-strategies
python main.py --symbol AAPL --strategy pe_threshold

# Full analysis
python main.py --symbol AAPL --strategy combined \
               --advanced-metrics --benchmark \
               --output-format json

# Custom configuration
python main.py --symbol AAPL --strategy pe_threshold \
               --initial-capital 500000 --position-size 0.10
```

---

**Strategy Forge CLI** - Production-ready command-line interface for quantitative trading strategy backtesting.