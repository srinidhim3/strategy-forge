#!/usr/bin/env python3
"""
Strategy Forge - Main CLI Entry Point

This script provides the primary command-line interface for Strategy Forge,
a comprehensive quantitative trading strategy backtesting framework.

Usage:
    python main.py --symbol AAPL --strategy pe_threshold
    python main.py --help
    python main.py --list-strategies

Author: Strategy Forge Development Team
Version: 1.0
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import and run the CLI
from cli.single_stock_runner import main

if __name__ == "__main__":
    sys.exit(main())