"""
CLI package for Strategy Forge

This package provides command-line interfaces for various Strategy Forge operations.
"""

from .single_stock_runner import main as run_single_stock_cli

__version__ = "1.0.0"
__all__ = ["run_single_stock_cli"]