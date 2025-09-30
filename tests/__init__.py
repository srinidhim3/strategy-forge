"""
Tests Module - Test Suite for Strategy Forge

This module contains the comprehensive test suite for Strategy Forge including:
- Unit tests for individual components
- Integration tests for complete workflows  
- Performance tests for backtesting engine
- Data validation tests

Test Structure:
    test_data/: Tests for data fetching and processing
    test_models/: Tests for business logic and algorithms
    test_utils/: Tests for utility functions
    test_integration/: End-to-end integration tests
    fixtures/: Test data and fixtures
    conftest.py: Pytest configuration and shared fixtures

Usage:
    Run all tests: pytest
    Run specific module: pytest tests/test_data/
    Run with coverage: pytest --cov=src
"""

# Test configuration and shared utilities
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Common test fixtures and utilities will be added here
# from .fixtures import sample_price_data, sample_financial_data
# from .conftest import setup_test_environment

__all__ = [
    # Will be populated as test utilities are implemented
]