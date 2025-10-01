"""
Price Data Fetcher Module

This module handles fetching daily OHLCV (Open, High, Low, Close, Volume) price data
from Yahoo Finance using the yfinance library. It provides functionality to download
historical price data for single stocks with proper error handling and data validation.

Classes:
    PriceFetcher: Main class for fetching and processing price data

Functions:
    fetch_single_stock: Convenience function for single stock data fetching
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceFetcher:
    """
    A class to fetch daily OHLCV price data from Yahoo Finance.
    
    This class provides methods to download historical stock price data with
    proper error handling, data validation, and standardized output format.
    """
    
    def __init__(self):
        """Initialize the PriceFetcher."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_price_data(
        self, 
        symbol: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime],
        interval: str = "1d",
        auto_adjust: bool = True,
        prepost: bool = False
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a single stock from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            start_date: Start date for data fetch (YYYY-MM-DD format or datetime)
            end_date: End date for data fetch (YYYY-MM-DD format or datetime)
            interval: Data interval ('1d', '1wk', '1mo'). Default: '1d'
            auto_adjust: Whether to adjust for splits and dividends. Default: True
            prepost: Include pre and post market data. Default: False
            
        Returns:
            pandas.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume]
                             and DatetimeIndex
                             
        Raises:
            ValueError: If symbol is invalid or dates are in wrong format
            ConnectionError: If unable to connect to Yahoo Finance
            Exception: For other data fetching errors
            
        Example:
            >>> fetcher = PriceFetcher()
            >>> data = fetcher.fetch_price_data("AAPL", "2023-01-01", "2023-12-31")
            >>> print(data.head())
        """
        try:
            # Validate inputs
            symbol = self._validate_symbol(symbol)
            start_date = self._validate_date(start_date)
            end_date = self._validate_date(end_date)
            
            # Check date order
            if start_date >= end_date:
                raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
            
            self.logger.info(f"Fetching price data for {symbol} from {start_date} to {end_date}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch the data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost
            )
            
            # Validate returned data
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol} between {start_date} and {end_date}")
            
            # Clean and standardize the data
            data = self._clean_price_data(data, symbol)
            
            self.logger.info(f"Successfully fetched {len(data)} rows of data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_latest_price(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch the latest price data for a stock.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of recent data to fetch. Default: 30
            
        Returns:
            pandas.DataFrame: Recent price data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_price_data(symbol, start_date, end_date)
    
    def get_stock_info(self, symbol: str) -> dict:
        """
        Get basic information about a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            dict: Stock information including name, sector, market cap, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            
            return stock_info
            
        except Exception as e:
            self.logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _validate_symbol(self, symbol: str) -> str:
        """Validate and clean stock symbol."""
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string")
        
        return symbol.strip().upper()
    
    def _validate_date(self, date: Union[str, datetime]) -> datetime:
        """Validate and convert date to datetime object."""
        if isinstance(date, datetime):
            return date
        
        if isinstance(date, str):
            try:
                return datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                try:
                    return datetime.strptime(date, "%Y/%m/%d")
                except ValueError:
                    raise ValueError(f"Date must be in YYYY-MM-DD or YYYY/MM/DD format, got: {date}")
        
        raise ValueError(f"Date must be string or datetime object, got: {type(date)}")
    
    def _clean_price_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize price data."""
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        # Ensure we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for {symbol}: {missing_columns}")
        
        # Keep only the expected columns that exist
        available_columns = [col for col in expected_columns if col in data.columns]
        data = data[available_columns]
        
        # Add symbol column for identification
        data['Symbol'] = symbol
        
        # Ensure numeric data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows where Close price is NaN (invalid data)
        if 'Close' in data.columns:
            data = data.dropna(subset=['Close'])
        
        # Sort by date to ensure chronological order
        data = data.sort_index()
        
        return data


def fetch_single_stock(
    symbol: str, 
    start_date: Union[str, datetime], 
    end_date: Union[str, datetime],
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to fetch price data for a single stock.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data fetch
        end_date: End date for data fetch
        **kwargs: Additional arguments passed to PriceFetcher.fetch_price_data()
        
    Returns:
        pandas.DataFrame: Price data
        
    Example:
        >>> data = fetch_single_stock("AAPL", "2023-01-01", "2023-12-31")
        >>> print(data.head())
    """
    fetcher = PriceFetcher()
    return fetcher.fetch_price_data(symbol, start_date, end_date, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    fetcher = PriceFetcher()
    
    try:
        # Fetch Apple stock data for 2023
        print("Fetching AAPL data...")
        aapl_data = fetcher.fetch_price_data("AAPL", "2023-01-01", "2023-12-31")
        print(f"AAPL data shape: {aapl_data.shape}")
        print(aapl_data.head())
        
        # Get stock info
        print("\nFetching AAPL info...")
        aapl_info = fetcher.get_stock_info("AAPL")
        print(aapl_info)
        
        # Test Indian stock
        print("\nFetching RELIANCE.NS data...")
        reliance_data = fetcher.fetch_price_data("RELIANCE.NS", "2023-01-01", "2023-06-30")
        print(f"RELIANCE data shape: {reliance_data.shape}")
        print(reliance_data.head())
        
    except Exception as e:
        print(f"Error in example: {e}")