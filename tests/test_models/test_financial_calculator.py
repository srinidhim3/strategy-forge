"""
Test module for financial metrics calculation functionality.

This module contains comprehensive tests for the FinancialCalculator class,
including unit tests, integration tests, and validation of calculations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
import warnings

from src.models.financial_calculator import FinancialCalculator, calculate_financial_metrics


class TestFinancialCalculator:
    """Test suite for the FinancialCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.calculator = FinancialCalculator()
        self.test_symbol = "AAPL"
        
        # Create sample data for testing
        self.sample_periods = [
            pd.Timestamp('2024-06-30'),
            pd.Timestamp('2024-03-31'),
            pd.Timestamp('2023-12-31')
        ]
        
        # Sample income statement data
        self.sample_income = pd.DataFrame({
            self.sample_periods[0]: [100000, 50000, 20000, 80000],
            self.sample_periods[1]: [95000, 48000, 19000, 76000],
            self.sample_periods[2]: [90000, 45000, 18000, 72000]
        }, index=[
            'Total Revenue', 'Operating Income', 'Net Income', 'Gross Profit'
        ])
        
        # Sample balance sheet data
        self.sample_balance = pd.DataFrame({
            self.sample_periods[0]: [500000, 200000, 50000, 15000000000],
            self.sample_periods[1]: [480000, 190000, 48000, 14500000000],
            self.sample_periods[2]: [460000, 180000, 46000, 14000000000]
        }, index=[
            'Total Assets', 'Total Equity Gross Minority Interest', 'Total Debt', 'Ordinary Shares Number'
        ])
        
        # Sample cash flow data
        self.sample_cashflow = pd.DataFrame({
            self.sample_periods[0]: [25000, 22000, -3000],
            self.sample_periods[1]: [24000, 21000, -3000],
            self.sample_periods[2]: [23000, 20000, -3000]
        }, index=[
            'Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure'
        ])
        
        self.sample_statements = {
            'income_statement': self.sample_income,
            'balance_sheet': self.sample_balance,
            'cash_flow': self.sample_cashflow
        }
    
    def test_initialization(self):
        """Test FinancialCalculator initialization."""
        assert self.calculator is not None
        assert hasattr(self.calculator, 'calculate_all_metrics')
        assert hasattr(self.calculator, 'calculate_profitability_metrics')
        assert hasattr(self.calculator, 'calculate_per_share_metrics')
        assert hasattr(self.calculator, 'calculate_leverage_metrics')
        assert hasattr(self.calculator, 'calculate_single_metric')
    
    def test_calculate_all_metrics_with_sample_data(self):
        """Test calculating all metrics with sample data."""
        result = self.calculator.calculate_all_metrics(self.sample_statements)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result.columns) == 3  # Three periods
        
        # Check that key metrics are present
        expected_metrics = [
            'ROE', 'ROA', 'EPS', 'BVPS', 'Debt_to_Equity',
            'Net_Profit_Margin', 'Operating_Margin'
        ]
        available_metrics = [metric for metric in expected_metrics 
                           if metric in result.index]
        assert len(available_metrics) > 0
    
    def test_profitability_metrics_calculation(self):
        """Test profitability metrics calculation."""
        metrics = self.calculator.calculate_profitability_metrics(
            self.sample_income, self.sample_balance
        )
        
        assert isinstance(metrics, dict)
        assert 'ROE' in metrics
        assert 'ROA' in metrics
        assert 'Net_Profit_Margin' in metrics
        assert 'Operating_Margin' in metrics
        
        # Test specific calculations
        roe = metrics['ROE'][self.sample_periods[0]]
        expected_roe = 20000 / 200000  # Net Income / Total Equity
        assert abs(roe - expected_roe) < 0.001
        
        roa = metrics['ROA'][self.sample_periods[0]]
        expected_roa = 20000 / 500000  # Net Income / Total Assets
        assert abs(roa - expected_roa) < 0.001
    
    def test_per_share_metrics_calculation(self):
        """Test per-share metrics calculation."""
        shares_outstanding = self.sample_balance.loc['Ordinary Shares Number']
        
        metrics = self.calculator.calculate_per_share_metrics(
            self.sample_income, self.sample_balance, shares_outstanding
        )
        
        assert isinstance(metrics, dict)
        assert 'EPS' in metrics
        assert 'BVPS' in metrics
        
        # Test EPS calculation
        eps = metrics['EPS'][self.sample_periods[0]]
        expected_eps = 20000 / 15000000000  # Net Income / Shares
        assert abs(eps - expected_eps) < 0.001
        
        # Test BVPS calculation
        bvps = metrics['BVPS'][self.sample_periods[0]]
        expected_bvps = 200000 / 15000000000  # Total Equity / Shares
        assert abs(bvps - expected_bvps) < 0.001
    
    def test_leverage_metrics_calculation(self):
        """Test leverage metrics calculation."""
        metrics = self.calculator.calculate_leverage_metrics(
            self.sample_income, self.sample_balance
        )
        
        assert isinstance(metrics, dict)
        assert 'Debt_to_Equity' in metrics
        assert 'Debt_to_Assets' in metrics
        assert 'Equity_Multiplier' in metrics
        
        # Test Debt-to-Equity calculation
        dte = metrics['Debt_to_Equity'][self.sample_periods[0]]
        expected_dte = 50000 / 200000  # Total Debt / Total Equity
        assert abs(dte - expected_dte) < 0.001
        
        # Test Equity Multiplier calculation
        em = metrics['Equity_Multiplier'][self.sample_periods[0]]
        expected_em = 500000 / 200000  # Total Assets / Total Equity
        assert abs(em - expected_em) < 0.001
    
    def test_single_metric_calculation(self):
        """Test calculating individual metrics."""
        # Test ROE calculation
        roe_series = self.calculator.calculate_single_metric(
            'ROE', self.sample_statements
        )
        assert isinstance(roe_series, pd.Series)
        assert len(roe_series) > 0
        
        # Test EPS calculation
        eps_series = self.calculator.calculate_single_metric(
            'EPS', self.sample_statements
        )
        assert isinstance(eps_series, pd.Series)
        assert len(eps_series) > 0
    
    def test_invalid_metric_name(self):
        """Test error handling for invalid metric names."""
        with pytest.raises(ValueError, match="Unknown metric"):
            self.calculator.calculate_single_metric(
                'INVALID_METRIC', self.sample_statements
            )
    
    def test_missing_statements_validation(self):
        """Test validation of missing financial statements."""
        invalid_statements = {
            'income_statement': self.sample_income,
            'balance_sheet': None,  # Missing balance sheet
            'cash_flow': self.sample_cashflow
        }
        
        with pytest.raises(ValueError, match="Missing required statement"):
            self.calculator._validate_statements(invalid_statements)
    
    def test_empty_statements_handling(self):
        """Test handling of empty financial statements."""
        empty_statements = {
            'income_statement': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame()
        }
        
        result = self.calculator.calculate_all_metrics(empty_statements)
        assert isinstance(result, pd.DataFrame)
        # Should return empty DataFrame for empty input
    
    def test_get_common_periods(self):
        """Test finding common periods across DataFrames."""
        common_periods = self.calculator._get_common_periods([
            self.sample_income, self.sample_balance
        ])
        
        assert len(common_periods) == 3
        assert all(period in common_periods for period in self.sample_periods)
    
    def test_get_value_with_multiple_names(self):
        """Test getting values with multiple possible field names."""
        # Test with exact match
        value = self.calculator._get_value(
            self.sample_income, 'Net Income', self.sample_periods[0]
        )
        assert value == 20000
        
        # Test with multiple possible names
        value = self.calculator._get_value(
            self.sample_balance, 
            ['Total Equity', 'Total Equity Gross Minority Interest'], 
            self.sample_periods[0]
        )
        assert value == 200000
        
        # Test with non-existent field
        value = self.calculator._get_value(
            self.sample_income, 'Non Existent Field', self.sample_periods[0]
        )
        assert value is None
    
    def test_convenience_function(self):
        """Test the convenience function for calculating metrics."""
        result = calculate_financial_metrics(self.sample_statements)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        
        # Should be identical to using the class directly
        calc_result = self.calculator.calculate_all_metrics(self.sample_statements)
        pd.testing.assert_frame_equal(result, calc_result)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])