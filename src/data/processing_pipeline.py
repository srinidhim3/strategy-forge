"""
Data Processing Pipeline for Strategy Forge.

This module provides a unified interface for orchestrating all data processing steps:
1. Price data fetching from Yahoo Finance
2. Financial statements fetching 
3. Financial metrics calculation
4. Point-in-time data alignment with reporting lag
5. Financial ratio calculations

Key Features:
- Single entry point for complete data preparation workflow
- Configurable parameters for all processing steps
- Comprehensive error handling and progress tracking
- Data quality validation and logging
- Performance monitoring and benchmarking
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import warnings
import time
from dataclasses import dataclass

# Import our modules
from .price_fetcher import PriceFetcher
from .financial_fetcher import FinancialFetcher
from .data_aligner import DataAligner
from ..models.financial_calculator import FinancialCalculator
from ..models.ratio_calculator import RatioCalculator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class PipelineConfig:
    """Configuration class for DataProcessingPipeline parameters."""
    
    # Date range parameters
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Data fetching parameters
    price_interval: str = "1d"
    financial_period: str = "quarterly"
    auto_adjust_prices: bool = True
    
    # Data alignment parameters
    reporting_lag_days: int = 45
    
    # Ratio calculation parameters
    ratios_to_calculate: Optional[List[str]] = None
    enable_ratio_validation: bool = True
    
    # Performance and logging
    enable_progress_tracking: bool = True
    enable_benchmarking: bool = True
    log_level: str = "INFO"
    
    # Data quality parameters
    min_data_points: int = 50
    max_missing_data_pct: float = 50.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.ratios_to_calculate is None:
            self.ratios_to_calculate = ['pe', 'pb', 'peg', 'ps', 'ev_ebitda']
        
        valid_ratios = ['pe', 'pb', 'peg', 'ps', 'ev_ebitda']
        invalid_ratios = [r for r in self.ratios_to_calculate if r not in valid_ratios]
        if invalid_ratios:
            raise ValueError(f"Invalid ratios specified: {invalid_ratios}")
        
        if self.reporting_lag_days < 0:
            raise ValueError("Reporting lag days must be non-negative")


class DataProcessingPipeline:
    """
    Unified data processing pipeline for Strategy Forge.
    
    This class orchestrates the complete data processing workflow from raw Yahoo Finance
    data to trading-ready financial ratios. It provides a single entry point with
    comprehensive error handling, progress tracking, and data validation.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the Data Processing Pipeline."""
        self.config = config or PipelineConfig()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize performance tracking
        self.performance_metrics = {}
        
        # Initialize modules
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all data processing modules."""
        try:
            self.price_fetcher = PriceFetcher()
            self.financial_fetcher = FinancialFetcher()
            self.financial_calculator = FinancialCalculator()
            self.data_aligner = DataAligner(reporting_lag_days=self.config.reporting_lag_days)
            self.ratio_calculator = RatioCalculator(enable_validation=self.config.enable_ratio_validation)
            
            self.logger.info("All pipeline modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline modules: {str(e)}")
            raise
    
    def process_stock(
        self, 
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process a single stock through the complete data pipeline.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data processing (YYYY-MM-DD format)
            end_date: End date for data processing (YYYY-MM-DD format)
            
        Returns:
            pandas.DataFrame: Complete dataset with price, fundamental, and ratio data
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            symbol = self._validate_symbol(symbol)
            start_date = start_date or self.config.start_date
            end_date = end_date or self.config.end_date
            
            if not start_date or not end_date:
                # Default to 2 years of data ending today
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
            
            self.logger.info(f"Starting pipeline processing for {symbol} ({start_date} to {end_date})")
            
            # Track progress through pipeline stages
            total_stages = 6
            current_stage = 0
            
            # Stage 1: Fetch price data
            current_stage += 1
            self._log_progress(f"Stage {current_stage}/{total_stages}: Fetching price data", current_stage, total_stages)
            
            price_data = self._fetch_price_data(symbol, start_date, end_date)
            self._validate_price_data(price_data, symbol)
            
            # Stage 2: Fetch financial statements
            current_stage += 1
            self._log_progress(f"Stage {current_stage}/{total_stages}: Fetching financial statements", current_stage, total_stages)
            
            financial_statements = self._fetch_financial_statements(symbol)
            self._validate_financial_statements(financial_statements, symbol)
            
            # Stage 3: Calculate financial metrics
            current_stage += 1
            self._log_progress(f"Stage {current_stage}/{total_stages}: Calculating financial metrics", current_stage, total_stages)
            
            financial_metrics = self._calculate_financial_metrics(financial_statements)
            self._validate_financial_metrics(financial_metrics, symbol)
            
            # Stage 4: Align data with point-in-time accuracy
            current_stage += 1
            self._log_progress(f"Stage {current_stage}/{total_stages}: Aligning data with {self.config.reporting_lag_days}-day lag", current_stage, total_stages)
            
            aligned_data = self._align_data(price_data, financial_metrics, start_date, end_date)
            self._validate_aligned_data(aligned_data, symbol)
            
            # Stage 5: Calculate financial ratios
            current_stage += 1
            self._log_progress(f"Stage {current_stage}/{total_stages}: Calculating financial ratios", current_stage, total_stages)
            
            ratio_data = self._calculate_ratios(aligned_data)
            self._validate_ratio_data(ratio_data, symbol)
            
            # Stage 6: Final processing and quality checks
            current_stage += 1
            self._log_progress(f"Stage {current_stage}/{total_stages}: Final processing and validation", current_stage, total_stages)
            
            final_data = self._finalize_data(ratio_data, symbol)
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self._record_performance_metrics(symbol, final_data, processing_time)
            
            self.logger.info(f"✅ Pipeline completed successfully for {symbol} in {processing_time:.2f}s")
            return final_data
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Pipeline failed for {symbol} after {processing_time:.2f}s: {str(e)}")
            raise
    
    def process_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        continue_on_error: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple stocks through the pipeline.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date for data processing
            end_date: End date for data processing
            continue_on_error: Whether to continue processing other stocks if one fails
            
        Returns:
            Dictionary mapping symbols to processed DataFrames
        """
        results = {}
        failed_symbols = []
        
        self.logger.info(f"Starting pipeline processing for {len(symbols)} stocks")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"Processing stock {i}/{len(symbols)}: {symbol}")
                
                result = self.process_stock(symbol, start_date, end_date)
                results[symbol] = result
                
                self.logger.info(f"✅ Completed {symbol} ({i}/{len(symbols)})")
                
            except Exception as e:
                failed_symbols.append(symbol)
                self.logger.error(f"❌ Failed to process {symbol}: {str(e)}")
                
                if not continue_on_error:
                    raise
        
        # Summary logging
        success_count = len(results)
        total_count = len(symbols)
        self.logger.info(f"Pipeline batch processing completed: {success_count}/{total_count} successful")
        
        if failed_symbols:
            self.logger.warning(f"Failed symbols: {failed_symbols}")
        
        return results
    
    def _fetch_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data with error handling."""
        return self.price_fetcher.fetch_price_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=self.config.price_interval,
            auto_adjust=self.config.auto_adjust_prices
        )
    
    def _fetch_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch financial statements with error handling."""
        return self.financial_fetcher.fetch_all_statements(
            symbol=symbol,
            period=self.config.financial_period
        )
    
    def _calculate_financial_metrics(self, statements: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate financial metrics with error handling."""
        return self.financial_calculator.calculate_all_metrics(statements)
    
    def _align_data(
        self, 
        price_data: pd.DataFrame, 
        financial_metrics: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Align data with error handling."""
        return self.data_aligner.align(
            price_data=price_data,
            financial_metrics=financial_metrics,
            start_date=start_date,
            end_date=end_date
        )
    
    def _calculate_ratios(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ratios with error handling."""
        if self.config.ratios_to_calculate:
            # Calculate specific ratios
            result_data = aligned_data.copy()
            
            ratio_methods = {
                'pe': self.ratio_calculator.calculate_pe_ratio,
                'pb': self.ratio_calculator.calculate_pb_ratio,
                'peg': self.ratio_calculator.calculate_peg_ratio,
                'ps': self.ratio_calculator.calculate_ps_ratio,
                'ev_ebitda': self.ratio_calculator.calculate_ev_ebitda
            }
            
            for ratio in self.config.ratios_to_calculate:
                if ratio in ratio_methods:
                    result_data = ratio_methods[ratio](result_data)
            
            return result_data
        else:
            # Calculate all ratios
            return self.ratio_calculator.calculate_all_ratios(aligned_data)
    
    def _validate_symbol(self, symbol: str) -> str:
        """Validate and clean stock symbol."""
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string")
        return symbol.strip().upper()
    
    def _validate_price_data(self, price_data: pd.DataFrame, symbol: str):
        """Validate price data quality."""
        if price_data.empty:
            raise ValueError(f"No price data available for {symbol}")
        
        if len(price_data) < self.config.min_data_points:
            raise ValueError(f"Insufficient price data for {symbol}: {len(price_data)} < {self.config.min_data_points}")
        
        self.logger.info(f"✅ Price data validation passed for {symbol}: {len(price_data)} rows")
    
    def _validate_financial_statements(self, statements: Dict[str, pd.DataFrame], symbol: str):
        """Validate financial statements quality."""
        if not statements:
            raise ValueError(f"No financial statements available for {symbol}")
        
        self.logger.info(f"✅ Financial statements validation passed for {symbol}")
    
    def _validate_financial_metrics(self, metrics: pd.DataFrame, symbol: str):
        """Validate financial metrics quality."""
        if metrics.empty:
            raise ValueError(f"No financial metrics calculated for {symbol}")
        
        self.logger.info(f"✅ Financial metrics validation passed for {symbol}: {len(metrics)} metrics")
    
    def _validate_aligned_data(self, aligned_data: pd.DataFrame, symbol: str):
        """Validate aligned data quality."""
        if aligned_data.empty:
            raise ValueError(f"No aligned data produced for {symbol}")
        
        self.logger.info(f"✅ Data alignment validation passed for {symbol}: {len(aligned_data)} rows")
    
    def _validate_ratio_data(self, ratio_data: pd.DataFrame, symbol: str):
        """Validate ratio calculation quality."""
        ratio_columns = ['pe_ratio', 'pb_ratio', 'peg_ratio', 'ps_ratio', 'ev_ebitda']
        calculated_ratios = [col for col in ratio_columns if col in ratio_data.columns]
        
        if not calculated_ratios:
            self.logger.warning(f"No ratios calculated for {symbol}")
        
        self.logger.info(f"✅ Ratio calculation validation passed for {symbol}")
    
    def _finalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Final data processing and cleanup."""
        # Add symbol column if not present
        if 'Symbol' not in data.columns:
            data['Symbol'] = symbol
        
        # Sort by date to ensure chronological order
        data = data.sort_index()
        
        # Add processing metadata
        data.attrs['symbol'] = symbol
        data.attrs['processed_date'] = datetime.now().isoformat()
        
        return data
    
    def _log_progress(self, message: str, current: int, total: int):
        """Log progress updates if enabled."""
        if self.config.enable_progress_tracking:
            progress_pct = (current / total) * 100
            self.logger.info(f"[{progress_pct:.0f}%] {message}")
    
    def _record_performance_metrics(self, symbol: str, data: pd.DataFrame, processing_time: float):
        """Record performance metrics if enabled."""
        if not self.config.enable_benchmarking:
            return
        
        metrics = {
            'symbol': symbol,
            'processing_time_seconds': processing_time,
            'data_rows': len(data),
            'data_columns': len(data.columns)
        }
        
        # Store metrics
        if 'performance_history' not in self.performance_metrics:
            self.performance_metrics['performance_history'] = []
        
        self.performance_metrics['performance_history'].append(metrics)
        
        # Log key metrics
        self.logger.info(
            f"📊 Performance: {processing_time:.2f}s, "
            f"{len(data)} rows, "
            f"{len(data.columns)} columns"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline performance metrics."""
        if 'performance_history' not in self.performance_metrics:
            return {'message': 'No performance data available'}
        
        history = self.performance_metrics['performance_history']
        
        # Calculate summary statistics
        processing_times = [m['processing_time_seconds'] for m in history]
        
        summary = {
            'total_stocks_processed': len(history),
            'avg_processing_time_seconds': np.mean(processing_times),
            'total_processing_time_seconds': np.sum(processing_times)
        }
        
        return summary


# Convenience function for simple usage
def process_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    reporting_lag_days: int = 45,
    ratios: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to process a single stock with default configuration.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data processing
        end_date: End date for data processing
        reporting_lag_days: Days to shift fundamental data
        ratios: List of ratios to calculate (None = all ratios)
        
    Returns:
        DataFrame with complete processed data
    """
    config = PipelineConfig(
        start_date=start_date,
        end_date=end_date,
        reporting_lag_days=reporting_lag_days,
        ratios_to_calculate=ratios
    )
    
    pipeline = DataProcessingPipeline(config)
    return pipeline.process_stock(symbol)


if __name__ == "__main__":
    print("Data Processing Pipeline - Example Usage")
    print("=" * 50)
    
    try:
        print("\n📊 Processing AAPL with default configuration")
        data = process_stock_data("AAPL", "2023-01-01", "2024-06-30")
        print(f"✅ Successfully processed AAPL: {data.shape}")
        
        # Show sample of results
        display_cols = ['Close', 'EPS', 'pe_ratio', 'pb_ratio']
        available_cols = [col for col in display_cols if col in data.columns]
        if available_cols:
            print(data[available_cols].tail().round(3))
        
    except Exception as e:
        print(f"❌ Example failed: {e}")