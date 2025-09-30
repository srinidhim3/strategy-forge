# pytest configuration for Strategy Forge

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_date_range():
    """Provide a standard date range for testing."""
    return {
        'start': '2024-09-01',
        'end': '2024-09-15',
        'start_dt': datetime(2024, 9, 1),
        'end_dt': datetime(2024, 9, 15)
    }

@pytest.fixture
def test_symbols():
    """Provide test symbols for different markets."""
    return {
        'us': 'AAPL',
        'india': 'RELIANCE.NS',
        'invalid': 'INVALID_SYMBOL_12345'
    }

@pytest.fixture
def price_fetcher():
    """Provide a PriceFetcher instance for testing."""
    from src.data.price_fetcher import PriceFetcher
    return PriceFetcher()

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network connection"
    )