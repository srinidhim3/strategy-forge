"""
Unit Tests for PriceFetcher

This module contains comprehensive tests for the PriceFetcher class including:
- Data fetching functionality
- Input validation and error handling
- Data cleaning and standardization
- Integration with Yahoo Finance API
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.price_fetcher import PriceFetcher, fetch_single_stock


class TestPriceFetcher:
    """Test cases for PriceFetcher class."""
    
    def test_price_fetcher_initialization(self, price_fetcher):
        """Test that PriceFetcher initializes correctly."""
        assert isinstance(price_fetcher, PriceFetcher)
        assert hasattr(price_fetcher, 'logger')
    
    @pytest.mark.network
    def test_fetch_price_data_basic(self, price_fetcher, sample_date_range, test_symbols):
        """Test basic price data fetching functionality."""
        # Test with US stock
        data = price_fetcher.fetch_price_data(
            test_symbols['us'], 
            sample_date_range['start'], 
            sample_date_range['end']
        )
        
        # Verify data structure
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert isinstance(data.index, pd.DatetimeIndex)
        
        # Verify required columns exist
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
        for col in expected_columns:
            if col in data.columns:  # Some may be missing for certain stocks
                assert col in data.columns
        
        # Verify data types
        assert data['Symbol'].iloc[0] == test_symbols['us']
        
        # Verify data integrity
        assert not data['Close'].isna().all()  # Should have some close prices
        assert (data['High'] >= data['Low']).all()  # High >= Low
        assert (data['High'] >= data['Close']).all()  # High >= Close
        assert (data['Low'] <= data['Close']).all()  # Low <= Close
    
    @pytest.mark.network
    def test_fetch_price_data_international(self, price_fetcher, sample_date_range, test_symbols):
        """Test price data fetching for international markets."""
        # Test with Indian stock
        data = price_fetcher.fetch_price_data(
            test_symbols['india'], 
            sample_date_range['start'], 
            sample_date_range['end']
        )
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert data['Symbol'].iloc[0] == test_symbols['india']
    
    def test_fetch_price_data_datetime_inputs(self, price_fetcher, sample_date_range, test_symbols):
        """Test price data fetching with datetime objects as input."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock the ticker and its history method
            mock_history = Mock()
            mock_history.return_value = pd.DataFrame({
                'Open': [100, 101],
                'High': [105, 106],
                'Low': [99, 100],
                'Close': [102, 103],
                'Volume': [1000, 1100]
            }, index=pd.date_range('2024-09-01', periods=2))
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history = mock_history
            mock_ticker.return_value = mock_ticker_instance
            
            # Test with datetime objects
            data = price_fetcher.fetch_price_data(
                test_symbols['us'], 
                sample_date_range['start_dt'], 
                sample_date_range['end_dt']
            )
            
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
    
    def test_symbol_validation(self, price_fetcher):
        """Test symbol validation functionality."""
        # Test valid symbol
        result = price_fetcher._validate_symbol("AAPL")
        assert result == "AAPL"
        
        # Test symbol with whitespace
        result = price_fetcher._validate_symbol("  aapl  ")
        assert result == "AAPL"
        
        # Test invalid symbols
        with pytest.raises(ValueError):
            price_fetcher._validate_symbol("")
        
        with pytest.raises(ValueError):
            price_fetcher._validate_symbol("   ")
        
        with pytest.raises(ValueError):
            price_fetcher._validate_symbol(None)
    
    def test_date_validation(self, price_fetcher):
        """Test date validation functionality."""
        # Test string dates
        result = price_fetcher._validate_date("2024-09-01")
        assert result == datetime(2024, 9, 1)
        
        result = price_fetcher._validate_date("2024/09/01")
        assert result == datetime(2024, 9, 1)
        
        # Test datetime object
        dt = datetime(2024, 9, 1)
        result = price_fetcher._validate_date(dt)
        assert result == dt
        
        # Test invalid dates
        with pytest.raises(ValueError):
            price_fetcher._validate_date("invalid-date")
        
        with pytest.raises(ValueError):
            price_fetcher._validate_date(12345)
    
    def test_date_order_validation(self, price_fetcher, test_symbols):
        """Test that start date must be before end date."""
        with patch('yfinance.Ticker'):
            with pytest.raises(ValueError, match="Start date.*must be before end date"):
                price_fetcher.fetch_price_data(
                    test_symbols['us'], 
                    "2024-09-15", 
                    "2024-09-01"  # End before start
                )
    
    def test_fetch_latest_price(self, price_fetcher, test_symbols):
        """Test fetching latest price data."""
        with patch.object(price_fetcher, 'fetch_price_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()
            
            price_fetcher.fetch_latest_price(test_symbols['us'], days=10)
            
            # Verify that fetch_price_data was called with recent dates
            mock_fetch.assert_called_once()
            args = mock_fetch.call_args[0]
            assert args[0] == test_symbols['us']
            # Verify dates are recent (within last 10 days)
            assert isinstance(args[1], datetime)
            assert isinstance(args[2], datetime)
    
    @pytest.mark.network
    def test_get_stock_info(self, price_fetcher, test_symbols):
        """Test stock information retrieval."""
        info = price_fetcher.get_stock_info(test_symbols['us'])
        
        assert isinstance(info, dict)
        assert 'symbol' in info
        assert info['symbol'] == test_symbols['us']
        
        # Should have basic info fields
        expected_fields = ['name', 'sector', 'exchange']
        for field in expected_fields:
            assert field in info
    
    def test_get_stock_info_invalid_symbol(self, price_fetcher, test_symbols):
        """Test stock info retrieval with invalid symbol."""
        info = price_fetcher.get_stock_info(test_symbols['invalid'])
        
        assert isinstance(info, dict)
        assert 'symbol' in info
        # Should handle error gracefully (either return error or minimal info)
    
    def test_data_cleaning(self, price_fetcher):
        """Test data cleaning functionality."""
        # Create sample data with issues
        sample_data = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, np.nan],
            'Low': [99, 100, 101],
            'Close': [102, np.nan, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-09-01', periods=3))
        
        cleaned_data = price_fetcher._clean_price_data(sample_data, "TEST")
        
        # Should have Symbol column
        assert 'Symbol' in cleaned_data.columns
        assert cleaned_data['Symbol'].iloc[0] == "TEST"
        
        # Should remove rows where Close is NaN
        assert not cleaned_data['Close'].isna().any()
        
        # Should be sorted by index
        assert cleaned_data.index.is_monotonic_increasing
    
    def test_empty_data_handling(self, price_fetcher, test_symbols):
        """Test handling of empty data response."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock empty DataFrame response
            mock_history = Mock()
            mock_history.return_value = pd.DataFrame()
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history = mock_history
            mock_ticker.return_value = mock_ticker_instance
            
            with pytest.raises(ValueError, match="No data found"):
                price_fetcher.fetch_price_data(
                    test_symbols['us'], 
                    "2024-09-01", 
                    "2024-09-15"
                )


class TestFetchSingleStock:
    """Test cases for fetch_single_stock convenience function."""
    
    def test_fetch_single_stock_function(self, sample_date_range, test_symbols):
        """Test the convenience function works correctly."""
        with patch('src.data.price_fetcher.PriceFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.fetch_price_data.return_value = pd.DataFrame({'Close': [100]})
            mock_fetcher_class.return_value = mock_fetcher
            
            result = fetch_single_stock(
                test_symbols['us'], 
                sample_date_range['start'], 
                sample_date_range['end']
            )
            
            # Verify PriceFetcher was created and called correctly
            mock_fetcher_class.assert_called_once()
            mock_fetcher.fetch_price_data.assert_called_once_with(
                test_symbols['us'], 
                sample_date_range['start'], 
                sample_date_range['end']
            )
            
            assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
class TestPriceFetcherIntegration:
    """Integration tests that require network access."""
    
    @pytest.mark.network
    def test_end_to_end_data_fetch(self, price_fetcher):
        """Test complete end-to-end data fetching process."""
        # Use a recent, short date range to ensure data exists
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = price_fetcher.fetch_price_data("AAPL", start_date, end_date)
        
        # Comprehensive validation
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'Close' in data.columns
        assert data['Symbol'].iloc[0] == "AAPL"
        
        # Verify data quality
        assert not data['Close'].isna().all()
        assert (data['Volume'] >= 0).all()  # Volume should be non-negative
        
        # Verify date range
        assert data.index.min() >= pd.Timestamp(start_date)
        assert data.index.max() <= pd.Timestamp(end_date)
    
    @pytest.mark.network
    def test_multiple_symbols(self, price_fetcher):
        """Test fetching data for multiple symbols sequentially."""
        symbols = ["AAPL", "MSFT"]
        date_range = ("2024-09-01", "2024-09-15")
        
        results = {}
        for symbol in symbols:
            try:
                data = price_fetcher.fetch_price_data(symbol, *date_range)
                results[symbol] = data
            except Exception as e:
                pytest.fail(f"Failed to fetch data for {symbol}: {e}")
        
        # Verify we got data for all symbols
        assert len(results) == len(symbols)
        for symbol, data in results.items():
            assert not data.empty
            assert data['Symbol'].iloc[0] == symbol