"""
Strategy Forge CLI Demo Script

This script demonstrates various usage patterns of the Strategy Forge CLI
for single-stock backtesting. It showcases different strategies, parameters,
and output formats.

Author: Strategy Forge Development Team
Version: 1.0
"""

import subprocess
import sys
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

def run_cli_command(args, description=""):
    """Run a CLI command and capture output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: python main.py {' '.join(args)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "main.py"] + args,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def create_temp_config_file(config_dict):
    """Create a temporary configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f, indent=2)
        return f.name

def main():
    """Main demo function"""
    print("🎯 Strategy Forge CLI Demonstration")
    print("=" * 60)
    print("This demo showcases various CLI usage patterns")
    print("Note: Some commands may take time due to data fetching")
    
    # Demo 1: List available strategies
    run_cli_command(
        ["--list-strategies"],
        "Demo 1: List Available Strategies"
    )
    
    # Demo 2: Help command
    run_cli_command(
        ["--help"],
        "Demo 2: Show Help Documentation"
    )
    
    # Demo 3: Basic P/E threshold strategy (short time period)
    run_cli_command([
        "--symbol", "AAPL",
        "--strategy", "pe_threshold",
        "--start-date", "2022-06-01",
        "--end-date", "2023-06-30",
        "--initial-capital", "500000",
        "--position-size", "0.20"
    ], "Demo 3: Basic P/E Threshold Strategy")
    
    # Demo 4: Moving average strategy with custom config file
    config_file = None
    try:
        ma_config = {
            "short_window": 5,
            "long_window": 15,
            "min_signal_strength": 0.1,
            "ma_type": "SMA",
            "price_column": "Close"
        }
        config_file = create_temp_config_file(ma_config)
        
        run_cli_command([
            "--symbol", "MSFT",
            "--strategy", "moving_average",
            "--start-date", "2022-01-01",
            "--end-date", "2023-06-30",
            "--strategy-config", config_file,
            "--output-format", "json",
            "--output-file", "msft_results.json"
        ], "Demo 4: Moving Average Strategy with Config File")
        
    finally:
        if config_file and os.path.exists(config_file):
            os.unlink(config_file)
    
    # Demo 5: Combined strategy with advanced metrics
    run_cli_command([
        "--symbol", "GOOGL",
        "--strategy", "combined",
        "--start-date", "2022-01-01",
        "--end-date", "2023-03-31",
        "--advanced-metrics",
        "--benchmark",
        "--output-format", "all",
        "--verbose"
    ], "Demo 5: Combined Strategy with Advanced Analytics")
    
    # Demo 6: Custom parameters demonstration
    run_cli_command([
        "--symbol", "TSLA",
        "--strategy", "pe_threshold",
        "--start-date", "2022-01-01",
        "--end-date", "2022-12-31",
        "--initial-capital", "1000000",
        "--position-size", "0.10",
        "--commission", "0.01",
        "--slippage", "0.0005",
        "--stop-loss", "0.05",
        "--take-profit", "0.25",
        "--output-format", "csv",
        "--output-file", "tesla_backtest.csv"
    ], "Demo 6: Custom Parameters with Risk Management")
    
    # Demo 7: Error handling - invalid symbol
    run_cli_command([
        "--symbol", "INVALID_SYMBOL",
        "--strategy", "pe_threshold",
        "--start-date", "2023-01-01",
        "--end-date", "2023-06-30"
    ], "Demo 7: Error Handling - Invalid Symbol")
    
    # Demo 8: Error handling - invalid date range
    run_cli_command([
        "--symbol", "AAPL",
        "--strategy", "moving_average",
        "--start-date", "2023-12-31",
        "--end-date", "2023-01-01"  # End before start
    ], "Demo 8: Error Handling - Invalid Date Range")
    
    print(f"\n{'='*60}")
    print("🎉 CLI Demo Complete!")
    print("="*60)
    print("Check the output/ directory for generated files:")
    print("- msft_results.json")
    print("- googl_results.json (if 'all' format was used)")
    print("- googl_results.csv (if 'all' format was used)")
    print("- tesla_backtest.csv")
    print("\nStrategy Forge CLI is ready for production use!")

if __name__ == "__main__":
    main()