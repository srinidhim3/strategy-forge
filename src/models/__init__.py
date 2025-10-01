"""
Models Module - Core Business Logic and Algorithms

This module contains the core business logic for Strategy Forge including:
- Financial metrics calculation (EPS, BVPS, ROE, etc.)
- Point-in-time ratio calculations (P/E, P/B, PEG)
- Trading strategy implementations
- Backtesting engine
- Portfolio management
- Stock screening logic

Key Components:
    financial_calculator: Financial metrics computation
    ratio_calculator: Point-in-time ratio calculations  
    strategies: Trading strategy implementations
    backtester: Single and multi-asset backtesting engine
    portfolio: Portfolio management and rebalancing
    screener: Flexible stock screening with rule engine
"""

# Current imports
from .financial_calculator import FinancialCalculator, calculate_financial_metrics

# Future imports will be added as modules are implemented
# from .ratio_calculator import RatioCalculator
# from .strategies import Strategy, PERatioStrategy, MovingAverageStrategy
# from .backtester import Backtester, BacktestResults
# from .portfolio import Portfolio, PortfolioManager
# from .screener import Screener, ScreenerRule

__all__ = [
    'FinancialCalculator',
    'calculate_financial_metrics',
    # Will be populated as modules are implemented
]