"""
Financial Metrics Calculator for Strategy Forge.

This module provides comprehensive calculation of key financial ratios and metrics
from raw financial statements. Designed to be extensible for future CFA Level 1
ratio expansion.

Key Features:
- Core financial metrics: EPS, BVPS, ROE, ROA, Debt-to-Equity
- Per-share calculations with proper share count handling
- Robust error handling and data validation
- Extensible framework for additional ratio categories
- Point-in-time calculations with proper date alignment
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, List
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class FinancialCalculator:
    """
    Calculate financial metrics and ratios from raw financial statements.
    
    This class provides methods to calculate key financial metrics including
    profitability ratios, per-share metrics, and leverage ratios. Designed
    to work with data from FinancialFetcher.
    
    Attributes:
        logger: Logger instance for tracking calculations
    """
    
    def __init__(self):
        """Initialize the Financial Calculator."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:%(name)s:%(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def calculate_all_metrics(
        self, 
        statements: Dict[str, pd.DataFrame],
        shares_outstanding: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate all available financial metrics from complete statement set.
        
        Args:
            statements: Dictionary containing 'income_statement', 'balance_sheet', 
                       and 'cash_flow' DataFrames
            shares_outstanding: Optional series of shares outstanding by period.
                              If None, will extract from balance sheet.
        
        Returns:
            pandas.DataFrame: Comprehensive metrics with periods as columns
            
        Example:
            >>> calc = FinancialCalculator()
            >>> statements = fetcher.fetch_all_statements("AAPL")
            >>> metrics = calc.calculate_all_metrics(statements)
            >>> print(metrics.head())
        """
        try:
            self.logger.info("Calculating comprehensive financial metrics")
            
            # Validate input data
            self._validate_statements(statements)
            
            # Extract individual statements
            income_stmt = statements.get('income_statement')
            balance_sheet = statements.get('balance_sheet')
            cash_flow = statements.get('cash_flow')
            
            # Initialize metrics dictionary
            all_metrics = {}
            
            # Get shares outstanding
            if shares_outstanding is None:
                shares_outstanding = self._get_shares_outstanding(balance_sheet)
            
            # Calculate profitability metrics
            profitability = self.calculate_profitability_metrics(
                income_stmt, balance_sheet
            )
            all_metrics.update(profitability)
            
            # Calculate per-share metrics
            per_share = self.calculate_per_share_metrics(
                income_stmt, balance_sheet, shares_outstanding
            )
            all_metrics.update(per_share)
            
            # Calculate leverage metrics
            leverage = self.calculate_leverage_metrics(
                income_stmt, balance_sheet
            )
            all_metrics.update(leverage)
            
            # Calculate efficiency metrics
            efficiency = self.calculate_efficiency_metrics(
                income_stmt, balance_sheet
            )
            all_metrics.update(efficiency)
            
            # Convert to DataFrame and align periods
            metrics_df = pd.DataFrame(all_metrics)
            
            # Transpose so metrics are rows and periods are columns
            metrics_df = metrics_df.T
            
            # Sort columns by date (newest first)
            if len(metrics_df.columns) > 0:
                try:
                    date_cols = pd.to_datetime(metrics_df.columns)
                    sorted_cols = date_cols.sort_values(ascending=False)
                    metrics_df = metrics_df[sorted_cols.strftime('%Y-%m-%d %H:%M:%S')]
                except:
                    # If date parsing fails, just return as-is
                    pass
            
            self.logger.info(f"Successfully calculated {len(metrics_df)} metrics across {len(metrics_df.columns)} periods")
            return metrics_df
            
        except Exception as e:
            self.logger.error(f"Error calculating financial metrics: {str(e)}")
            raise
    
    def calculate_profitability_metrics(
        self, 
        income_stmt: pd.DataFrame, 
        balance_sheet: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Calculate profitability ratios: ROE, ROA, ROI, profit margins.
        
        Args:
            income_stmt: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
            
        Returns:
            Dict[str, pd.Series]: Profitability metrics by period
        """
        try:
            metrics = {}
            
            if income_stmt is None or balance_sheet is None:
                self.logger.warning("Missing statements for profitability calculations")
                return metrics
            
            # Get common periods
            common_periods = self._get_common_periods([income_stmt, balance_sheet])
            
            for period in common_periods:
                period_metrics = {}
                
                # Extract key values
                net_income = self._get_value(income_stmt, 'Net Income', period)
                total_revenue = self._get_value(income_stmt, 'Total Revenue', period)
                operating_income = self._get_value(income_stmt, 'Operating Income', period)
                total_assets = self._get_value(balance_sheet, 'Total Assets', period)
                total_equity = self._get_value(
                    balance_sheet, 
                    ['Total Equity Gross Minority Interest', 'Common Stock Equity', 'Total Stockholder Equity'], 
                    period
                )
                
                # Calculate Return on Equity (ROE)
                if net_income is not None and total_equity is not None and total_equity != 0:
                    period_metrics['ROE'] = net_income / total_equity
                
                # Calculate Return on Assets (ROA)
                if net_income is not None and total_assets is not None and total_assets != 0:
                    period_metrics['ROA'] = net_income / total_assets
                
                # Calculate Net Profit Margin
                if net_income is not None and total_revenue is not None and total_revenue != 0:
                    period_metrics['Net_Profit_Margin'] = net_income / total_revenue
                
                # Calculate Operating Margin
                if operating_income is not None and total_revenue is not None and total_revenue != 0:
                    period_metrics['Operating_Margin'] = operating_income / total_revenue
                
                # Calculate Asset Turnover
                if total_revenue is not None and total_assets is not None and total_assets != 0:
                    period_metrics['Asset_Turnover'] = total_revenue / total_assets
                
                # Store period metrics
                for metric_name, value in period_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = pd.Series(dtype=float)
                    metrics[metric_name][period] = value
            
            self.logger.info(f"Calculated {len(metrics)} profitability metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating profitability metrics: {str(e)}")
            return {}
    
    def calculate_per_share_metrics(
        self, 
        income_stmt: pd.DataFrame, 
        balance_sheet: pd.DataFrame,
        shares_outstanding: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate per-share metrics: EPS, BVPS, Revenue per Share.
        
        Args:
            income_stmt: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
            shares_outstanding: Series of shares outstanding by period
            
        Returns:
            Dict[str, pd.Series]: Per-share metrics by period
        """
        try:
            metrics = {}
            
            if income_stmt is None or balance_sheet is None:
                self.logger.warning("Missing statements for per-share calculations")
                return metrics
            
            # Get common periods
            common_periods = self._get_common_periods([income_stmt, balance_sheet])
            
            for period in common_periods:
                period_metrics = {}
                
                # Get shares outstanding for this period
                shares = self._get_shares_for_period(shares_outstanding, period)
                
                if shares is None or shares == 0:
                    continue
                
                # Extract key values
                net_income = self._get_value(income_stmt, 'Net Income', period)
                total_revenue = self._get_value(income_stmt, 'Total Revenue', period)
                total_equity = self._get_value(
                    balance_sheet, 
                    ['Total Equity Gross Minority Interest', 'Common Stock Equity', 'Total Stockholder Equity'], 
                    period
                )
                
                # Calculate Earnings per Share (EPS)
                if net_income is not None:
                    period_metrics['EPS'] = net_income / shares
                
                # Calculate Book Value per Share (BVPS)
                if total_equity is not None:
                    period_metrics['BVPS'] = total_equity / shares
                
                # Calculate Revenue per Share
                if total_revenue is not None:
                    period_metrics['Revenue_Per_Share'] = total_revenue / shares
                
                # Store period metrics
                for metric_name, value in period_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = pd.Series(dtype=float)
                    metrics[metric_name][period] = value
            
            self.logger.info(f"Calculated {len(metrics)} per-share metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating per-share metrics: {str(e)}")
            return {}
    
    def calculate_leverage_metrics(
        self, 
        income_stmt: pd.DataFrame, 
        balance_sheet: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Calculate leverage ratios: Debt-to-Equity, Debt-to-Assets, Interest Coverage.
        
        Args:
            income_stmt: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
            
        Returns:
            Dict[str, pd.Series]: Leverage metrics by period
        """
        try:
            metrics = {}
            
            if income_stmt is None or balance_sheet is None:
                self.logger.warning("Missing statements for leverage calculations")
                return metrics
            
            # Get common periods
            common_periods = self._get_common_periods([income_stmt, balance_sheet])
            
            for period in common_periods:
                period_metrics = {}
                
                # Extract key values
                total_debt = self._get_value(balance_sheet, 'Total Debt', period)
                total_assets = self._get_value(balance_sheet, 'Total Assets', period)
                total_equity = self._get_value(
                    balance_sheet, 
                    ['Total Equity Gross Minority Interest', 'Common Stock Equity', 'Total Stockholder Equity'], 
                    period
                )
                operating_income = self._get_value(income_stmt, 'Operating Income', period)
                interest_expense = self._get_value(
                    income_stmt, 
                    ['Interest Expense', 'Interest Expense Non Operating'], 
                    period
                )
                
                # Calculate Debt-to-Equity Ratio
                if total_debt is not None and total_equity is not None and total_equity != 0:
                    period_metrics['Debt_to_Equity'] = total_debt / total_equity
                
                # Calculate Debt-to-Assets Ratio
                if total_debt is not None and total_assets is not None and total_assets != 0:
                    period_metrics['Debt_to_Assets'] = total_debt / total_assets
                
                # Calculate Equity Multiplier (Financial Leverage)
                if total_assets is not None and total_equity is not None and total_equity != 0:
                    period_metrics['Equity_Multiplier'] = total_assets / total_equity
                
                # Calculate Interest Coverage Ratio
                if (operating_income is not None and interest_expense is not None 
                    and interest_expense != 0):
                    period_metrics['Interest_Coverage'] = operating_income / abs(interest_expense)
                
                # Store period metrics
                for metric_name, value in period_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = pd.Series(dtype=float)
                    metrics[metric_name][period] = value
            
            self.logger.info(f"Calculated {len(metrics)} leverage metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage metrics: {str(e)}")
            return {}
    
    def calculate_efficiency_metrics(
        self, 
        income_stmt: pd.DataFrame, 
        balance_sheet: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Calculate efficiency ratios: Asset Turnover, Equity Turnover.
        
        Args:
            income_stmt: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
            
        Returns:
            Dict[str, pd.Series]: Efficiency metrics by period
        """
        try:
            metrics = {}
            
            if income_stmt is None or balance_sheet is None:
                self.logger.warning("Missing statements for efficiency calculations")
                return metrics
            
            # Get common periods
            common_periods = self._get_common_periods([income_stmt, balance_sheet])
            
            for period in common_periods:
                period_metrics = {}
                
                # Extract key values
                total_revenue = self._get_value(income_stmt, 'Total Revenue', period)
                total_assets = self._get_value(balance_sheet, 'Total Assets', period)
                total_equity = self._get_value(
                    balance_sheet, 
                    ['Total Equity Gross Minority Interest', 'Common Stock Equity', 'Total Stockholder Equity'], 
                    period
                )
                
                # Calculate Asset Turnover (already calculated in profitability, but included for completeness)
                if total_revenue is not None and total_assets is not None and total_assets != 0:
                    period_metrics['Asset_Turnover'] = total_revenue / total_assets
                
                # Calculate Equity Turnover
                if total_revenue is not None and total_equity is not None and total_equity != 0:
                    period_metrics['Equity_Turnover'] = total_revenue / total_equity
                
                # Store period metrics
                for metric_name, value in period_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = pd.Series(dtype=float)
                    metrics[metric_name][period] = value
            
            self.logger.info(f"Calculated {len(metrics)} efficiency metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency metrics: {str(e)}")
            return {}
    
    def calculate_single_metric(
        self, 
        metric_name: str, 
        statements: Dict[str, pd.DataFrame],
        **kwargs
    ) -> pd.Series:
        """
        Calculate a single financial metric across all available periods.
        
        Args:
            metric_name: Name of the metric to calculate
            statements: Dictionary of financial statements
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            pd.Series: Metric values by period
        """
        try:
            # Map metric names to calculation methods
            metric_map = {
                'ROE': self._calculate_roe,
                'ROA': self._calculate_roa,
                'EPS': self._calculate_eps,
                'BVPS': self._calculate_bvps,
                'Debt_to_Equity': self._calculate_debt_to_equity,
                'Net_Profit_Margin': self._calculate_net_profit_margin,
                'Operating_Margin': self._calculate_operating_margin,
                'Asset_Turnover': self._calculate_asset_turnover,
                'Interest_Coverage': self._calculate_interest_coverage
            }
            
            if metric_name not in metric_map:
                raise ValueError(f"Unknown metric: {metric_name}")
            
            return metric_map[metric_name](statements, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error calculating {metric_name}: {str(e)}")
            return pd.Series(dtype=float)
    
    # Helper methods
    def _validate_statements(self, statements: Dict[str, pd.DataFrame]) -> None:
        """Validate input financial statements."""
        if not isinstance(statements, dict):
            raise ValueError("Statements must be a dictionary")
        
        required_keys = ['income_statement', 'balance_sheet', 'cash_flow']
        for key in required_keys:
            if key not in statements:
                raise ValueError(f"Missing required statement: {key}")
    
    def _get_common_periods(self, dataframes: List[pd.DataFrame]) -> List:
        """Get common periods across multiple DataFrames."""
        if not dataframes or any(df is None for df in dataframes):
            return []
        
        # Get intersection of all columns
        common_periods = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_periods = common_periods.intersection(df.columns)
        
        return sorted(list(common_periods))
    
    def _get_value(
        self, 
        df: pd.DataFrame, 
        metric_names: Union[str, List[str]], 
        period: str
    ) -> Optional[float]:
        """Get value from DataFrame with multiple possible metric names."""
        if df is None or period not in df.columns:
            return None
        
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        
        for metric_name in metric_names:
            if metric_name in df.index:
                value = df.loc[metric_name, period]
                if pd.notna(value):
                    return float(value)
        
        return None
    
    def _get_shares_outstanding(self, balance_sheet: pd.DataFrame) -> pd.Series:
        """Extract shares outstanding from balance sheet."""
        if balance_sheet is None:
            return pd.Series(dtype=float)
        
        # Try different possible field names for shares outstanding
        share_fields = [
            'Ordinary Shares Number',
            'Share Issued',
            'Common Shares Outstanding',
            'Shares Outstanding'
        ]
        
        for field in share_fields:
            if field in balance_sheet.index:
                shares = balance_sheet.loc[field]
                # Convert to numeric and remove NaN values
                shares = pd.to_numeric(shares, errors='coerce')
                return shares.dropna()
        
        self.logger.warning("Could not find shares outstanding in balance sheet")
        return pd.Series(dtype=float)
    
    def _get_shares_for_period(self, shares_outstanding: pd.Series, period: str) -> Optional[float]:
        """Get shares outstanding for a specific period."""
        if shares_outstanding is None or len(shares_outstanding) == 0:
            return None
        
        if period in shares_outstanding.index:
            shares = shares_outstanding[period]
            if pd.notna(shares) and shares > 0:
                return float(shares)
        
        # If exact period not found, try to find the closest period
        try:
            period_dt = pd.to_datetime(period)
            shares_dt = pd.to_datetime(shares_outstanding.index)
            closest_idx = np.argmin(np.abs(shares_dt - period_dt))
            closest_shares = shares_outstanding.iloc[closest_idx]
            
            if pd.notna(closest_shares) and closest_shares > 0:
                return float(closest_shares)
        except:
            pass
        
        return None
    
    # Individual metric calculation methods
    def _calculate_roe(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Return on Equity."""
        income_stmt = statements.get('income_statement')
        balance_sheet = statements.get('balance_sheet')
        
        if income_stmt is None or balance_sheet is None:
            return pd.Series(dtype=float)
        
        roe_series = pd.Series(dtype=float)
        common_periods = self._get_common_periods([income_stmt, balance_sheet])
        
        for period in common_periods:
            net_income = self._get_value(income_stmt, 'Net Income', period)
            total_equity = self._get_value(
                balance_sheet, 
                ['Total Equity Gross Minority Interest', 'Common Stock Equity'], 
                period
            )
            
            if net_income is not None and total_equity is not None and total_equity != 0:
                roe_series[period] = net_income / total_equity
        
        return roe_series
    
    def _calculate_roa(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Return on Assets."""
        income_stmt = statements.get('income_statement')
        balance_sheet = statements.get('balance_sheet')
        
        if income_stmt is None or balance_sheet is None:
            return pd.Series(dtype=float)
        
        roa_series = pd.Series(dtype=float)
        common_periods = self._get_common_periods([income_stmt, balance_sheet])
        
        for period in common_periods:
            net_income = self._get_value(income_stmt, 'Net Income', period)
            total_assets = self._get_value(balance_sheet, 'Total Assets', period)
            
            if net_income is not None and total_assets is not None and total_assets != 0:
                roa_series[period] = net_income / total_assets
        
        return roa_series
    
    def _calculate_eps(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Earnings per Share."""
        income_stmt = statements.get('income_statement')
        balance_sheet = statements.get('balance_sheet')
        shares_outstanding = kwargs.get('shares_outstanding')
        
        if income_stmt is None:
            return pd.Series(dtype=float)
        
        if shares_outstanding is None:
            shares_outstanding = self._get_shares_outstanding(balance_sheet)
        
        eps_series = pd.Series(dtype=float)
        
        for period in income_stmt.columns:
            net_income = self._get_value(income_stmt, 'Net Income', period)
            shares = self._get_shares_for_period(shares_outstanding, period)
            
            if net_income is not None and shares is not None and shares != 0:
                eps_series[period] = net_income / shares
        
        return eps_series
    
    def _calculate_bvps(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Book Value per Share."""
        balance_sheet = statements.get('balance_sheet')
        shares_outstanding = kwargs.get('shares_outstanding')
        
        if balance_sheet is None:
            return pd.Series(dtype=float)
        
        if shares_outstanding is None:
            shares_outstanding = self._get_shares_outstanding(balance_sheet)
        
        bvps_series = pd.Series(dtype=float)
        
        for period in balance_sheet.columns:
            total_equity = self._get_value(
                balance_sheet, 
                ['Total Equity Gross Minority Interest', 'Common Stock Equity'], 
                period
            )
            shares = self._get_shares_for_period(shares_outstanding, period)
            
            if total_equity is not None and shares is not None and shares != 0:
                bvps_series[period] = total_equity / shares
        
        return bvps_series
    
    def _calculate_debt_to_equity(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Debt-to-Equity ratio."""
        balance_sheet = statements.get('balance_sheet')
        
        if balance_sheet is None:
            return pd.Series(dtype=float)
        
        dte_series = pd.Series(dtype=float)
        
        for period in balance_sheet.columns:
            total_debt = self._get_value(balance_sheet, 'Total Debt', period)
            total_equity = self._get_value(
                balance_sheet, 
                ['Total Equity Gross Minority Interest', 'Common Stock Equity'], 
                period
            )
            
            if total_debt is not None and total_equity is not None and total_equity != 0:
                dte_series[period] = total_debt / total_equity
        
        return dte_series
    
    def _calculate_net_profit_margin(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Net Profit Margin."""
        income_stmt = statements.get('income_statement')
        
        if income_stmt is None:
            return pd.Series(dtype=float)
        
        npm_series = pd.Series(dtype=float)
        
        for period in income_stmt.columns:
            net_income = self._get_value(income_stmt, 'Net Income', period)
            total_revenue = self._get_value(income_stmt, 'Total Revenue', period)
            
            if net_income is not None and total_revenue is not None and total_revenue != 0:
                npm_series[period] = net_income / total_revenue
        
        return npm_series
    
    def _calculate_operating_margin(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Operating Margin."""
        income_stmt = statements.get('income_statement')
        
        if income_stmt is None:
            return pd.Series(dtype=float)
        
        om_series = pd.Series(dtype=float)
        
        for period in income_stmt.columns:
            operating_income = self._get_value(income_stmt, 'Operating Income', period)
            total_revenue = self._get_value(income_stmt, 'Total Revenue', period)
            
            if operating_income is not None and total_revenue is not None and total_revenue != 0:
                om_series[period] = operating_income / total_revenue
        
        return om_series
    
    def _calculate_asset_turnover(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Asset Turnover."""
        income_stmt = statements.get('income_statement')
        balance_sheet = statements.get('balance_sheet')
        
        if income_stmt is None or balance_sheet is None:
            return pd.Series(dtype=float)
        
        at_series = pd.Series(dtype=float)
        common_periods = self._get_common_periods([income_stmt, balance_sheet])
        
        for period in common_periods:
            total_revenue = self._get_value(income_stmt, 'Total Revenue', period)
            total_assets = self._get_value(balance_sheet, 'Total Assets', period)
            
            if total_revenue is not None and total_assets is not None and total_assets != 0:
                at_series[period] = total_revenue / total_assets
        
        return at_series
    
    def _calculate_interest_coverage(self, statements: Dict[str, pd.DataFrame], **kwargs) -> pd.Series:
        """Calculate Interest Coverage Ratio."""
        income_stmt = statements.get('income_statement')
        
        if income_stmt is None:
            return pd.Series(dtype=float)
        
        ic_series = pd.Series(dtype=float)
        
        for period in income_stmt.columns:
            operating_income = self._get_value(income_stmt, 'Operating Income', period)
            interest_expense = self._get_value(
                income_stmt, 
                ['Interest Expense', 'Interest Expense Non Operating'], 
                period
            )
            
            if (operating_income is not None and interest_expense is not None 
                and interest_expense != 0):
                ic_series[period] = operating_income / abs(interest_expense)
        
        return ic_series


# Convenience function for quick metric calculation
def calculate_financial_metrics(statements: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convenience function to calculate all financial metrics.
    
    Args:
        statements: Dictionary containing financial statements
        
    Returns:
        pd.DataFrame: All calculated metrics
    """
    calculator = FinancialCalculator()
    return calculator.calculate_all_metrics(statements)