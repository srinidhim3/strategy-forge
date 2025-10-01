"""
Point-in-Time Ratio Calculator for Strategy Forge.

This module calculates financial ratios from aligned price and fundamental data.
It takes the output from DataAligner (Task 12) and computes daily ratios that
can be used by trading strategies.

Key Features:
- Daily ratio calculations: P/E, P/B, PEG, P/S, EV/EBITDA
- Point-in-time accuracy with forward-filled fundamental data
- Comprehensive edge case handling (negative values, zeros, missing data)
- Ratio validation against market-reasonable ranges
- Integration with Data Alignment Pipeline output

Classes:
    RatioCalculator: Main class for calculating financial ratios from aligned data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List, Tuple
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class RatioCalculator:
    """
    Calculate financial ratios from aligned price and fundamental data.
    
    This class takes the output from DataAligner and calculates daily financial
    ratios that are critical for trading strategy development. It handles edge
    cases and provides validation to ensure ratio quality.
    
    Attributes:
        logger: Logger instance for tracking calculations
        validation_ranges: Dictionary of reasonable ratio ranges for validation
    """
    
    def __init__(self, enable_validation: bool = True):
        """
        Initialize the Ratio Calculator.
        
        Args:
            enable_validation: Whether to enable ratio validation against reasonable ranges
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.enable_validation = enable_validation
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:%(name)s:%(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Define reasonable ratio ranges for validation
        self.validation_ranges = {
            'pe_ratio': {'min': 0, 'max': 500, 'typical_min': 5, 'typical_max': 100},
            'pb_ratio': {'min': 0, 'max': 50, 'typical_min': 0.5, 'typical_max': 10},
            'peg_ratio': {'min': 0, 'max': 10, 'typical_min': 0.1, 'typical_max': 3},
            'ps_ratio': {'min': 0, 'max': 100, 'typical_min': 0.5, 'typical_max': 20},
            'ev_ebitda': {'min': 0, 'max': 1000, 'typical_min': 5, 'typical_max': 50}
        }
    
    def calculate_all_ratios(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all supported financial ratios from aligned data.
        
        Args:
            aligned_data: DataFrame from DataAligner containing price and fundamental data
            
        Returns:
            DataFrame with original data plus calculated ratio columns
            
        Example:
            >>> calculator = RatioCalculator()
            >>> ratios_data = calculator.calculate_all_ratios(aligned_data)
            >>> print(ratios_data[['Close', 'pe_ratio', 'pb_ratio']].head())
        """
        try:
            self.logger.info("Starting calculation of all financial ratios")
            
            # Create a copy to avoid modifying original data
            result_data = aligned_data.copy()
            
            # Calculate each ratio
            result_data = self.calculate_pe_ratio(result_data)
            result_data = self.calculate_pb_ratio(result_data)
            result_data = self.calculate_peg_ratio(result_data)
            result_data = self.calculate_ps_ratio(result_data)
            result_data = self.calculate_ev_ebitda(result_data)
            
            # Generate summary statistics
            summary = self._generate_ratio_summary(result_data)
            self.logger.info(f"Ratio calculation completed. Summary: {summary}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Error calculating ratios: {str(e)}")
            raise
    
    def calculate_pe_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price-to-Earnings (P/E) ratio.
        
        Formula: P/E = Price per Share / Earnings per Share
        
        Args:
            data: DataFrame with 'Close' price and 'eps' columns
            
        Returns:
            DataFrame with added 'pe_ratio' column
        """
        try:
            if 'Close' not in data.columns or 'EPS' not in data.columns:
                self.logger.warning("Missing required columns for P/E calculation: 'Close' and 'EPS'")
                data['pe_ratio'] = np.nan
                return data
            
            # Calculate P/E ratio
            # Handle edge cases: negative or zero EPS
            eps_positive = data['EPS'] > 0
            
            data['pe_ratio'] = np.where(
                eps_positive,
                data['Close'] / data['EPS'],
                np.nan  # Set to NaN for negative or zero EPS
            )
            
            # Apply validation if enabled
            if self.enable_validation:
                data['pe_ratio'] = self._validate_ratio(data['pe_ratio'], 'pe_ratio')
            
            # Log calculation statistics
            valid_ratios = data['pe_ratio'].dropna()
            if len(valid_ratios) > 0:
                self.logger.info(f"P/E Ratio - Calculated: {len(valid_ratios)} values, "
                               f"Range: {valid_ratios.min():.2f} - {valid_ratios.max():.2f}, "
                               f"Median: {valid_ratios.median():.2f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating P/E ratio: {str(e)}")
            data['pe_ratio'] = np.nan
            return data
    
    def calculate_pb_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price-to-Book (P/B) ratio.
        
        Formula: P/B = Price per Share / Book Value per Share
        
        Args:
            data: DataFrame with 'Close' price and 'book_value_per_share' columns
            
        Returns:
            DataFrame with added 'pb_ratio' column
        """
        try:
            if 'Close' not in data.columns or 'BVPS' not in data.columns:
                self.logger.warning("Missing required columns for P/B calculation: 'Close' and 'BVPS'")
                data['pb_ratio'] = np.nan
                return data
            
            # Calculate P/B ratio
            # Handle edge cases: negative or zero book value
            bvps_positive = data['BVPS'] > 0
            
            data['pb_ratio'] = np.where(
                bvps_positive,
                data['Close'] / data['BVPS'],
                np.nan  # Set to NaN for negative or zero book value
            )
            
            # Apply validation if enabled
            if self.enable_validation:
                data['pb_ratio'] = self._validate_ratio(data['pb_ratio'], 'pb_ratio')
            
            # Log calculation statistics
            valid_ratios = data['pb_ratio'].dropna()
            if len(valid_ratios) > 0:
                self.logger.info(f"P/B Ratio - Calculated: {len(valid_ratios)} values, "
                               f"Range: {valid_ratios.min():.2f} - {valid_ratios.max():.2f}, "
                               f"Median: {valid_ratios.median():.2f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating P/B ratio: {str(e)}")
            data['pb_ratio'] = np.nan
            return data
    
    def calculate_peg_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price/Earnings to Growth (PEG) ratio.
        
        Formula: PEG = P/E Ratio / Earnings Growth Rate (%)
        
        Note: Requires 'pe_ratio' to be calculated first and earnings growth data
        
        Args:
            data: DataFrame with 'pe_ratio' and earnings growth data
            
        Returns:
            DataFrame with added 'peg_ratio' column
        """
        try:
            # Check if P/E ratio exists (should be calculated first)
            if 'pe_ratio' not in data.columns:
                # Calculate P/E ratio first
                data = self.calculate_pe_ratio(data)
            
            # For PEG calculation, we need earnings growth rate
            # This would typically come from comparing current EPS to previous year EPS
            # For now, we'll implement a simplified version using available EPS data
            if 'EPS' not in data.columns:
                self.logger.warning("Missing EPS data for PEG calculation")
                data['peg_ratio'] = np.nan
                return data
            
            # Calculate year-over-year EPS growth rate (simplified approach)
            # We'll use a rolling 4-quarter (252 trading days) comparison
            eps_1y_ago = data['EPS'].shift(252)  # Approximately 1 year ago
            eps_growth_rate = ((data['EPS'] - eps_1y_ago) / eps_1y_ago.abs()) * 100
            
            # Calculate PEG ratio
            # Handle edge cases: zero or negative growth rate, invalid P/E
            valid_conditions = (
                (data['pe_ratio'] > 0) & 
                (eps_growth_rate > 0.1) &  # Minimum 0.1% growth to avoid division issues
                (data['pe_ratio'].notna()) &
                (eps_growth_rate.notna())
            )
            
            data['peg_ratio'] = np.where(
                valid_conditions,
                data['pe_ratio'] / eps_growth_rate,
                np.nan
            )
            
            # Apply validation if enabled
            if self.enable_validation:
                data['peg_ratio'] = self._validate_ratio(data['peg_ratio'], 'peg_ratio')
            
            # Log calculation statistics
            valid_ratios = data['peg_ratio'].dropna()
            if len(valid_ratios) > 0:
                self.logger.info(f"PEG Ratio - Calculated: {len(valid_ratios)} values, "
                               f"Range: {valid_ratios.min():.2f} - {valid_ratios.max():.2f}, "
                               f"Median: {valid_ratios.median():.2f}")
            else:
                self.logger.warning("PEG Ratio - No valid values calculated (requires historical EPS data)")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating PEG ratio: {str(e)}")
            data['peg_ratio'] = np.nan
            return data
    
    def calculate_ps_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price-to-Sales (P/S) ratio.
        
        Formula: P/S = Price per Share / Sales per Share
        
        Args:
            data: DataFrame with 'Close' price and revenue per share data
            
        Returns:
            DataFrame with added 'ps_ratio' column
        """
        try:
            # Check for required columns - we need sales/revenue per share
            # This should be calculated from total revenue and shares outstanding
            required_cols = ['Close']
            revenue_cols = ['Revenue_Per_Share', 'revenue_per_share', 'sales_per_share']
            
            # Look for revenue per share column
            revenue_col = None
            for col in revenue_cols:
                if col in data.columns:
                    revenue_col = col
                    break
            
            if revenue_col is None:
                # Try to calculate from total revenue and shares outstanding
                if 'total_revenue' in data.columns and 'shares_outstanding' in data.columns:
                    shares_outstanding = data['shares_outstanding'].replace(0, np.nan)
                    data['revenue_per_share'] = data['total_revenue'] / shares_outstanding
                    revenue_col = 'revenue_per_share'
                else:
                    self.logger.warning("Missing required data for P/S calculation: revenue per share or total revenue + shares outstanding")
                    data['ps_ratio'] = np.nan
                    return data
            
            # Calculate P/S ratio
            # Handle edge cases: negative or zero revenue per share
            revenue_positive = data[revenue_col] > 0
            
            data['ps_ratio'] = np.where(
                revenue_positive,
                data['Close'] / data[revenue_col],
                np.nan  # Set to NaN for negative or zero revenue
            )
            
            # Apply validation if enabled
            if self.enable_validation:
                data['ps_ratio'] = self._validate_ratio(data['ps_ratio'], 'ps_ratio')
            
            # Log calculation statistics
            valid_ratios = data['ps_ratio'].dropna()
            if len(valid_ratios) > 0:
                self.logger.info(f"P/S Ratio - Calculated: {len(valid_ratios)} values, "
                               f"Range: {valid_ratios.min():.2f} - {valid_ratios.max():.2f}, "
                               f"Median: {valid_ratios.median():.2f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating P/S ratio: {str(e)}")
            data['ps_ratio'] = np.nan
            return data
    
    def calculate_ev_ebitda(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Enterprise Value to EBITDA (EV/EBITDA) ratio.
        
        Formula: EV/EBITDA = Enterprise Value / EBITDA
        Enterprise Value = Market Cap + Total Debt - Cash
        
        Args:
            data: DataFrame with required financial data
            
        Returns:
            DataFrame with added 'ev_ebitda' column
        """
        try:
            # Required data: shares outstanding, total debt, cash, EBITDA
            required_for_ev = ['shares_outstanding', 'Close']
            optional_for_ev = ['total_debt', 'cash_and_equivalents', 'ebitda']
            
            missing_required = [col for col in required_for_ev if col not in data.columns]
            if missing_required:
                self.logger.warning(f"Missing required columns for EV/EBITDA calculation: {missing_required}")
                data['ev_ebitda'] = np.nan
                return data
            
            # Calculate Market Cap
            shares_outstanding = data['shares_outstanding'].replace(0, np.nan)
            market_cap = data['Close'] * shares_outstanding
            
            # Calculate Enterprise Value
            # Start with market cap, add debt, subtract cash
            enterprise_value = market_cap.copy()
            
            if 'total_debt' in data.columns:
                enterprise_value += data['total_debt'].fillna(0)
            
            if 'cash_and_equivalents' in data.columns:
                enterprise_value -= data['cash_and_equivalents'].fillna(0)
            
            # Check for EBITDA
            if 'ebitda' not in data.columns:
                self.logger.warning("Missing EBITDA data for EV/EBITDA calculation")
                data['ev_ebitda'] = np.nan
                return data
            
            # Calculate EV/EBITDA ratio
            # Handle edge cases: negative or zero EBITDA
            ebitda_positive = data['ebitda'] > 0
            
            data['ev_ebitda'] = np.where(
                ebitda_positive & enterprise_value.notna(),
                enterprise_value / data['ebitda'],
                np.nan  # Set to NaN for negative/zero EBITDA or missing EV
            )
            
            # Apply validation if enabled
            if self.enable_validation:
                data['ev_ebitda'] = self._validate_ratio(data['ev_ebitda'], 'ev_ebitda')
            
            # Log calculation statistics
            valid_ratios = data['ev_ebitda'].dropna()
            if len(valid_ratios) > 0:
                self.logger.info(f"EV/EBITDA Ratio - Calculated: {len(valid_ratios)} values, "
                               f"Range: {valid_ratios.min():.2f} - {valid_ratios.max():.2f}, "
                               f"Median: {valid_ratios.median():.2f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating EV/EBITDA ratio: {str(e)}")
            data['ev_ebitda'] = np.nan
            return data
    
    def _validate_ratio(self, ratio_series: pd.Series, ratio_name: str) -> pd.Series:
        """
        Validate ratio values against reasonable ranges.
        
        Args:
            ratio_series: Series containing ratio values
            ratio_name: Name of the ratio for validation ranges
            
        Returns:
            Series with outliers set to NaN
        """
        if ratio_name not in self.validation_ranges:
            return ratio_series
        
        ranges = self.validation_ranges[ratio_name]
        
        # Apply validation: set extreme outliers to NaN
        validated = ratio_series.where(
            (ratio_series >= ranges['min']) & (ratio_series <= ranges['max']),
            np.nan
        )
        
        # Count outliers for logging
        outliers = (ratio_series.notna() & validated.isna()).sum()
        if outliers > 0:
            self.logger.warning(f"{ratio_name}: {outliers} outlier values set to NaN "
                              f"(outside range {ranges['min']}-{ranges['max']})")
        
        return validated
    
    def _generate_ratio_summary(self, data: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for calculated ratios.
        
        Args:
            data: DataFrame with calculated ratios
            
        Returns:
            Dictionary with summary statistics
        """
        ratio_columns = ['pe_ratio', 'pb_ratio', 'peg_ratio', 'ps_ratio', 'ev_ebitda']
        existing_ratios = [col for col in ratio_columns if col in data.columns]
        
        summary = {
            'total_rows': len(data),
            'ratios_calculated': len(existing_ratios),
            'ratio_coverage': {}
        }
        
        for ratio in existing_ratios:
            valid_count = data[ratio].notna().sum()
            coverage_pct = (valid_count / len(data)) * 100
            
            if valid_count > 0:
                ratio_values = data[ratio].dropna()
                summary['ratio_coverage'][ratio] = {
                    'valid_values': valid_count,
                    'coverage_percentage': round(coverage_pct, 2),
                    'median': round(ratio_values.median(), 3),
                    'mean': round(ratio_values.mean(), 3),
                    'std': round(ratio_values.std(), 3)
                }
            else:
                summary['ratio_coverage'][ratio] = {
                    'valid_values': 0,
                    'coverage_percentage': 0,
                    'median': None,
                    'mean': None,
                    'std': None
                }
        
        return summary
    
    def get_ratio_benchmarks(self, data: pd.DataFrame, ratio_name: str) -> Dict:
        """
        Get benchmark statistics for a specific ratio.
        
        Args:
            data: DataFrame with calculated ratios
            ratio_name: Name of the ratio to benchmark
            
        Returns:
            Dictionary with benchmark statistics
        """
        if ratio_name not in data.columns:
            return {'error': f'Ratio {ratio_name} not found in data'}
        
        ratio_values = data[ratio_name].dropna()
        
        if len(ratio_values) == 0:
            return {'error': f'No valid values for {ratio_name}'}
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        benchmarks = {
            'count': len(ratio_values),
            'mean': ratio_values.mean(),
            'median': ratio_values.median(),
            'std': ratio_values.std(),
            'min': ratio_values.min(),
            'max': ratio_values.max(),
            'percentiles': {f'p{p}': ratio_values.quantile(p/100) for p in percentiles}
        }
        
        # Add typical range information if available
        if ratio_name in self.validation_ranges:
            ranges = self.validation_ranges[ratio_name]
            benchmarks['typical_range'] = {
                'min': ranges['typical_min'],
                'max': ranges['typical_max']
            }
        
        return benchmarks


def calculate_ratios(
    aligned_data: pd.DataFrame,
    ratios: Optional[List[str]] = None,
    enable_validation: bool = True
) -> pd.DataFrame:
    """
    Convenience function to calculate financial ratios from aligned data.
    
    Args:
        aligned_data: DataFrame from DataAligner with price and fundamental data
        ratios: List of ratios to calculate. If None, calculates all ratios.
                Options: ['pe', 'pb', 'peg', 'ps', 'ev_ebitda']
        enable_validation: Whether to enable ratio validation
        
    Returns:
        DataFrame with original data plus calculated ratio columns
        
    Example:
        >>> ratios_data = calculate_ratios(aligned_data, ['pe', 'pb'])
        >>> print(ratios_data[['Close', 'pe_ratio', 'pb_ratio']].head())
    """
    calculator = RatioCalculator(enable_validation=enable_validation)
    
    if ratios is None:
        # Calculate all ratios
        return calculator.calculate_all_ratios(aligned_data)
    
    # Calculate specific ratios
    result_data = aligned_data.copy()
    
    ratio_methods = {
        'pe': calculator.calculate_pe_ratio,
        'pb': calculator.calculate_pb_ratio,
        'peg': calculator.calculate_peg_ratio,
        'ps': calculator.calculate_ps_ratio,
        'ev_ebitda': calculator.calculate_ev_ebitda
    }
    
    for ratio in ratios:
        if ratio in ratio_methods:
            result_data = ratio_methods[ratio](result_data)
        else:
            logging.warning(f"Unknown ratio: {ratio}. Skipping.")
    
    return result_data


# Example usage and testing
if __name__ == "__main__":
    print("Ratio Calculator Module - Ready for testing")
    print("Available ratios: P/E, P/B, PEG, P/S, EV/EBITDA")
    print("Use calculate_ratios() function or RatioCalculator class directly")