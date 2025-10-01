"""
Data Alignment Pipeline Module

This module handles the critical task of aligning quarterly financial data with daily price data
while maintaining point-in-time accuracy. It ensures that backtesting results are realistic
by preventing lookahead bias.

Key Features:
- Applies reporting lag to quarterly financial data (default: 45 days)
- Forward-fills financial metrics to create daily frequency data
- Merges price and fundamental data into aligned dataset
- Maintains data integrity and handles missing values
- Supports configurable reporting lag periods

Classes:
    DataAligner: Main class for aligning financial and price data
    
Functions:
    align_data: Convenience function for simple alignment tasks
    
Example:
    >>> from src.data.data_aligner import DataAligner
    >>> from src.data.price_fetcher import PriceFetcher
    >>> from src.data.financial_fetcher import FinancialFetcher
    >>> from src.models.financial_calculator import FinancialCalculator
    >>> 
    >>> # Fetch raw data
    >>> price_fetcher = PriceFetcher()
    >>> financial_fetcher = FinancialFetcher()
    >>> calculator = FinancialCalculator()
    >>> 
    >>> prices = price_fetcher.fetch_price_data("AAPL", "2020-01-01", "2024-12-31")
    >>> statements = financial_fetcher.fetch_all_statements("AAPL")
    >>> metrics = calculator.calculate_all_metrics(statements)
    >>> 
    >>> # Align data with point-in-time accuracy
    >>> aligner = DataAligner(reporting_lag_days=45)
    >>> aligned_data = aligner.align(prices, metrics)
    >>> 
    >>> # Result: Every trading day has correct fundamental data available on that date
    >>> print(aligned_data.head())
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings

class DataAligner:
    """
    Aligns quarterly financial metrics with daily price data while maintaining point-in-time accuracy.
    
    This class handles the complex task of merging different frequency data (quarterly fundamentals
    with daily prices) while ensuring that no future information is used. It applies reporting lag
    to simulate real-world conditions where fundamental data isn't immediately available.
    
    Attributes:
        reporting_lag_days (int): Number of days to shift fundamental data
        logger (logging.Logger): Logger instance for tracking operations
        
    Example:
        >>> aligner = DataAligner(reporting_lag_days=45)
        >>> aligned_data = aligner.align(price_data, financial_metrics)
    """
    
    def __init__(self, reporting_lag_days: int = 45):
        """
        Initialize DataAligner with configurable reporting lag.
        
        Args:
            reporting_lag_days: Number of days to shift fundamental data to simulate
                              reporting lag. Default is 45 days (typical for quarterly reports).
        """
        self.reporting_lag_days = reporting_lag_days
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"DataAligner initialized with {reporting_lag_days}-day reporting lag")
    
    def align(
        self,
        price_data: pd.DataFrame,
        financial_metrics: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Align price data with financial metrics using point-in-time methodology.
        
        Args:
            price_data: DataFrame with daily price data (OHLCV format)
            financial_metrics: DataFrame with financial metrics by quarter
            start_date: Optional start date for alignment (YYYY-MM-DD)
            end_date: Optional end date for alignment (YYYY-MM-DD)
            
        Returns:
            DataFrame with aligned daily data containing both price and fundamental data
            
        Raises:
            ValueError: If input data is invalid or incompatible
            
        Example:
            >>> aligned = aligner.align(prices, metrics, "2020-01-01", "2023-12-31")
        """
        try:
            self.logger.info("Starting data alignment process")
            
            # Validate input data
            self._validate_input_data(price_data, financial_metrics)
            
            # Apply reporting lag to financial data
            shifted_metrics = self._apply_reporting_lag(financial_metrics)
            
            # Forward-fill metrics to daily frequency
            daily_metrics = self._forward_fill_metrics(shifted_metrics, price_data.index)
            
            # Merge price and fundamental data
            aligned_data = self._merge_data(price_data, daily_metrics)
            
            # Filter by date range if specified
            if start_date or end_date:
                aligned_data = self._filter_date_range(aligned_data, start_date, end_date)
            
            # Clean and validate final dataset
            aligned_data = self._clean_final_data(aligned_data)
            
            self.logger.info(f"Data alignment completed. Final dataset shape: {aligned_data.shape}")
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"Error in data alignment: {str(e)}")
            raise
    
    def _validate_input_data(self, price_data: pd.DataFrame, financial_metrics: pd.DataFrame) -> None:
        """
        Validate input data for alignment process.
        
        Args:
            price_data: DataFrame with daily price data
            financial_metrics: DataFrame with financial metrics
            
        Raises:
            ValueError: If data validation fails
        """
        if price_data.empty:
            raise ValueError("Price data cannot be empty")
        
        if financial_metrics.empty:
            raise ValueError("Financial metrics cannot be empty")
        
        # Check if price data has datetime index
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise ValueError("Price data must have datetime index")
        
        # Check if financial metrics has datetime columns
        datetime_cols = 0
        for col in financial_metrics.columns:
            if isinstance(col, (pd.Timestamp, datetime)) or pd.api.types.is_datetime64_any_dtype(pd.to_datetime(col, errors='coerce')):
                datetime_cols += 1
        
        if datetime_cols == 0:
            raise ValueError("Financial metrics must have datetime columns")
        
        # Check for required price columns
        required_price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_price_cols if col not in price_data.columns]
        if missing_cols:
            self.logger.warning(f"Missing price columns: {missing_cols}")
        
        self.logger.debug("Input data validation completed successfully")
    
    def _apply_reporting_lag(self, financial_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Apply reporting lag to financial metrics to simulate real-world delays.
        
        In reality, quarterly financial statements are published 45-90 days after
        the quarter end. This function shifts the availability of fundamental data
        to reflect this lag.
        
        Args:
            financial_metrics: DataFrame with financial metrics by quarter
            
        Returns:
            DataFrame with reporting lag applied to column dates
            
        Example:
            Q2 2024 data (June 30) becomes available on Aug 14 (45 days later)
        """
        try:
            shifted_metrics = financial_metrics.copy()
            
            # Shift column dates by reporting lag
            lag_delta = timedelta(days=self.reporting_lag_days)
            shifted_columns = []
            
            for col in financial_metrics.columns:
                if isinstance(col, (pd.Timestamp, datetime)):
                    # Handle timezone-aware timestamps
                    if hasattr(col, 'tz') and col.tz is not None:
                        shifted_col = col.tz_localize(None) + lag_delta
                    else:
                        shifted_col = col + lag_delta
                    shifted_columns.append(shifted_col)
                else:
                    # Try to convert to datetime
                    try:
                        col_dt = pd.to_datetime(col)
                        if hasattr(col_dt, 'tz') and col_dt.tz is not None:
                            shifted_col = col_dt.tz_localize(None) + lag_delta
                        else:
                            shifted_col = col_dt + lag_delta
                        shifted_columns.append(shifted_col)
                    except:
                        # If conversion fails, keep original
                        shifted_columns.append(col)
            
            shifted_metrics.columns = shifted_columns
            
            self.logger.debug(f"Applied {self.reporting_lag_days}-day reporting lag to financial metrics")
            
            return shifted_metrics
            
        except Exception as e:
            self.logger.error(f"Error applying reporting lag: {str(e)}")
            raise
    
    def _forward_fill_metrics(
        self, 
        shifted_metrics: pd.DataFrame, 
        price_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Forward-fill quarterly metrics to daily frequency.
        
        This creates a daily time series where each day has the most recent
        fundamental data available on that date. Handles gaps between quarters
        and ensures no lookahead bias.
        
        Args:
            shifted_metrics: DataFrame with lag-adjusted financial metrics
            price_index: DatetimeIndex from price data for reindexing
            
        Returns:
            DataFrame with daily fundamental data
        """
        try:
            # Ensure price_index is timezone-naive for compatibility
            if price_index.tz is not None:
                price_index_clean = price_index.tz_localize(None)
            else:
                price_index_clean = price_index.copy()
            
            # Create daily metrics DataFrame with price index
            daily_metrics = pd.DataFrame(index=price_index_clean)
            
            # For each metric, forward-fill from quarterly to daily
            for metric_name in shifted_metrics.index:
                metric_series = shifted_metrics.loc[metric_name]
                
                # Ensure quarterly dates are timezone-naive
                quarterly_index = metric_series.index
                if hasattr(quarterly_index, 'tz') and quarterly_index.tz is not None:
                    quarterly_index = quarterly_index.tz_localize(None)
                
                # Create a series with quarterly dates and values
                quarterly_series = pd.Series(
                    data=metric_series.values,
                    index=quarterly_index,
                    name=metric_name
                )
                
                # Remove NaN values
                quarterly_series = quarterly_series.dropna()
                
                if not quarterly_series.empty:
                    # Create combined index and reindex
                    combined_index = price_index_clean.union(quarterly_series.index)
                    
                    # Reindex to combined frequency and forward-fill
                    daily_series = quarterly_series.reindex(combined_index).sort_index().ffill()
                    
                    # Keep only the dates in price_index
                    daily_metrics[metric_name] = daily_series.reindex(price_index_clean)
            
            self.logger.debug(f"Forward-filled {len(shifted_metrics)} metrics to daily frequency")
            
            return daily_metrics
            
        except Exception as e:
            self.logger.error(f"Error in forward-filling metrics: {str(e)}")
            raise
    
    def _merge_data(
        self, 
        price_data: pd.DataFrame, 
        daily_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge price data with daily fundamental metrics.
        
        Args:
            price_data: DataFrame with daily price data
            daily_metrics: DataFrame with daily fundamental metrics
            
        Returns:
            DataFrame with combined price and fundamental data
        """
        try:
            # Handle timezone differences between price_data and daily_metrics
            price_data_clean = price_data.copy()
            
            # Ensure both DataFrames have timezone-naive indexes
            if price_data_clean.index.tz is not None:
                price_data_clean.index = price_data_clean.index.tz_localize(None)
            
            if daily_metrics.index.tz is not None:
                daily_metrics.index = daily_metrics.index.tz_localize(None)
            
            # Merge on datetime index
            aligned_data = price_data_clean.join(daily_metrics, how='left')
            
            self.logger.debug(f"Merged price and fundamental data. Combined shape: {aligned_data.shape}")
            
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"Error merging data: {str(e)}")
            raise
    
    def _filter_date_range(
        self, 
        data: pd.DataFrame, 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Filter data by specified date range.
        
        Args:
            data: DataFrame to filter
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            Filtered DataFrame
        """
        try:
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
            
            self.logger.debug(f"Filtered data to date range: {start_date} to {end_date}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error filtering date range: {str(e)}")
            raise
    
    def _clean_final_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the final aligned dataset.
        
        Args:
            data: Raw aligned DataFrame
            
        Returns:
            Cleaned DataFrame ready for analysis
        """
        try:
            # Remove rows where all fundamental data is NaN
            fundamental_cols = [col for col in data.columns 
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            
            if fundamental_cols:
                # Keep rows where at least one fundamental metric is available
                data = data.dropna(subset=fundamental_cols, how='all')
            
            # Sort by date
            data = data.sort_index()
            
            # Add metadata columns
            data['data_date'] = data.index
            data['has_fundamental_data'] = ~data[fundamental_cols].isna().all(axis=1) if fundamental_cols else False
            
            self.logger.debug(f"Final cleaned dataset shape: {data.shape}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning final data: {str(e)}")
            raise
    
    def get_alignment_summary(self, aligned_data: pd.DataFrame) -> Dict[str, Union[int, float, str]]:
        """
        Generate summary statistics for the aligned dataset.
        
        Args:
            aligned_data: Aligned DataFrame from align() method
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            fundamental_cols = [col for col in aligned_data.columns 
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                                           'data_date', 'has_fundamental_data']]
            
            summary = {
                'total_rows': len(aligned_data),
                'date_range_start': aligned_data.index.min().strftime('%Y-%m-%d'),
                'date_range_end': aligned_data.index.max().strftime('%Y-%m-%d'),
                'fundamental_metrics_count': len(fundamental_cols),
                'rows_with_fundamental_data': aligned_data['has_fundamental_data'].sum() if 'has_fundamental_data' in aligned_data.columns else 0,
                'coverage_percentage': (aligned_data['has_fundamental_data'].sum() / len(aligned_data) * 100) if 'has_fundamental_data' in aligned_data.columns else 0,
                'reporting_lag_days': self.reporting_lag_days
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating alignment summary: {str(e)}")
            return {}


def align_data(
    price_data: pd.DataFrame,
    financial_metrics: pd.DataFrame,
    reporting_lag_days: int = 45,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function for simple data alignment tasks.
    
    Args:
        price_data: DataFrame with daily price data
        financial_metrics: DataFrame with financial metrics by quarter
        reporting_lag_days: Days to shift fundamental data (default: 45)
        start_date: Optional start date for alignment
        end_date: Optional end date for alignment
        
    Returns:
        Aligned DataFrame with both price and fundamental data
        
    Example:
        >>> aligned = align_data(prices, metrics, reporting_lag_days=60)
    """
    aligner = DataAligner(reporting_lag_days=reporting_lag_days)
    return aligner.align(price_data, financial_metrics, start_date, end_date)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Data Alignment Pipeline - Example Usage")
    print("="*50)
    
    # This would typically be run with real data:
    # from src.data.price_fetcher import PriceFetcher
    # from src.data.financial_fetcher import FinancialFetcher  
    # from src.models.financial_calculator import FinancialCalculator
    #
    # price_fetcher = PriceFetcher()
    # financial_fetcher = FinancialFetcher()
    # calculator = FinancialCalculator()
    #
    # prices = price_fetcher.fetch_price_data("AAPL", "2020-01-01", "2024-12-31")
    # statements = financial_fetcher.fetch_all_statements("AAPL")
    # metrics = calculator.calculate_all_metrics(statements)
    #
    # aligner = DataAligner(reporting_lag_days=45)
    # aligned_data = aligner.align(prices, metrics)
    #
    # print(f"Aligned data shape: {aligned_data.shape}")
    # print(aligner.get_alignment_summary(aligned_data))
    
    print("Module loaded successfully. Ready for data alignment tasks!")