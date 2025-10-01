"""
Data Module - Financial Data Fetching and Processing

This module handles all data-related operations including:
- Fetching daily price data from Yahoo Finance
- Downloading quarterly financial statements
- Processing and aligning data for point-in-time accuracy
- Data pipeline orchestration

Key Components:
    price_fetcher: Daily OHLCV data retrieval
    financial_fetcher: Quarterly financial statements
    data_aligner: Point-in-time data alignment with reporting lag
    pipeline: Complete data processing orchestration
    universe: Stock universe management
"""

# Import implemented modules
from .price_fetcher import PriceFetcher, fetch_single_stock
from .financial_fetcher import FinancialFetcher, fetch_financial_statements

# Future imports will be added as modules are implemented
# from .data_aligner import DataAligner
# from .pipeline import DataPipeline
# from .universe import StockUniverse

__all__ = [
    "PriceFetcher",
    "fetch_single_stock",
    "FinancialFetcher", 
    "fetch_financial_statements",
    # Will be populated as more modules are implemented
]