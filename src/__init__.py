"""
Strategy Forge - Quantitative Trading Strategy Backtesting Platform

A comprehensive platform for designing, backtesting, and analyzing quantitative 
stock trading strategies with accurate point-in-time data.

Modules:
    data: Data fetching and processing utilities
    models: Core business logic and algorithms  
    utils: Utility functions and helpers
"""

__version__ = "0.1.0"
__author__ = "Strategy Forge Contributors"
__email__ = "your.email@example.com"

# Core imports for convenience
from . import data
from . import models
from . import utils

__all__ = [
    "data",
    "models", 
    "utils",
    "__version__",
]