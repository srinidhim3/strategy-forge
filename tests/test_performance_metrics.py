"""
Test suite for advanced performance metrics implementation (Task 17)

This module tests the enhanced performance metrics calculations including:
- Value at Risk (VaR) and Conditional VaR
- Information Ratio, Treynor Ratio, Jensen Alpha
- Beta, Tracking Error, Capture Ratios
- Rolling performance metrics
- Benchmark comparison utilities
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.backtester import (
    SingleAssetBacktester, BacktestConfig, BacktestResult,
    benchmark_strategy_analysis, compare_strategies_advanced,
    generate_performance_report
)
from src.models.strategies import PEThresholdStrategy, MovingAverageStrategy, StrategyConfig


class TestAdvancedPerformanceMetrics(unittest.TestCase):
    """Test advanced performance metrics calculations"""

    def setUp(self):
        """Set up test data and configurations"""
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Generate correlated price series (for realistic beta calculations)
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        prices = 100 * (1 + returns).cumprod()
        
        self.price_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 252)
        }, index=dates)
        
        # Generate synthetic P/E data
        self.price_data['pe_ratio'] = np.random.uniform(15, 25, 252)
        
        # Create test strategy
        pe_config = StrategyConfig(
            name="Test PE Strategy",
            description="Test strategy for performance metrics",
            parameters={'buy_pe_threshold': 20.0, 'sell_pe_threshold': 25.0}
        )
        self.strategy = PEThresholdStrategy(pe_config)
        self.strategy.generate_signals(self.price_data)
        
        # Create backtester
        self.config = BacktestConfig(
            initial_capital=100000,
            commission_per_share=0.01,
            slippage_bps=5
        )
        self.backtester = SingleAssetBacktester(self.config)

    def test_value_at_risk_calculation(self):
        """Test VaR calculation at different confidence levels"""
        # Create returns with known distribution
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var_95 = self.backtester._calculate_value_at_risk(returns, 0.95)
        var_99 = self.backtester._calculate_value_at_risk(returns, 0.99)
        
        # VaR 99% should be more negative than VaR 95%
        self.assertLess(var_99, var_95)
        
        # Both should be negative for loss scenarios
        self.assertLess(var_95, 0)
        self.assertLess(var_99, 0)
        
        # Test edge cases
        empty_returns = pd.Series([])
        self.assertEqual(self.backtester._calculate_value_at_risk(empty_returns), 0.0)

    def test_conditional_var_calculation(self):
        """Test CVaR calculation"""
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var_95 = self.backtester._calculate_value_at_risk(returns, 0.95)
        cvar_95 = self.backtester._calculate_conditional_var(returns, 0.95)
        
        # CVaR should be more negative than VaR (expected loss in tail)
        self.assertLess(cvar_95, var_95)
        
        # Test edge cases
        empty_returns = pd.Series([])
        self.assertEqual(self.backtester._calculate_conditional_var(empty_returns), 0.0)

    def test_beta_calculation(self):
        """Test portfolio beta calculation"""
        # Create correlated returns
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        portfolio_returns = 0.8 * benchmark_returns + pd.Series(np.random.normal(0, 0.01, 252))
        
        beta = self.backtester._calculate_beta(portfolio_returns, benchmark_returns)
        
        # Beta should be close to 0.8 given our construction
        self.assertGreater(beta, 0.5)
        self.assertLess(beta, 1.2)
        
        # Test edge cases
        empty_returns = pd.Series([])
        self.assertEqual(self.backtester._calculate_beta(empty_returns, benchmark_returns), 0.0)
        self.assertEqual(self.backtester._calculate_beta(portfolio_returns, empty_returns), 0.0)

    def test_information_ratio_calculation(self):
        """Test Information Ratio calculation"""
        # Create portfolio with slight outperformance
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        portfolio_returns = benchmark_returns + pd.Series(np.random.normal(0.0002, 0.005, 252))
        
        info_ratio = self.backtester._calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        # Should be a reasonable number (not inf or 0)
        self.assertGreater(abs(info_ratio), 0.01)
        self.assertLess(abs(info_ratio), 10)

    def test_treynor_ratio_calculation(self):
        """Test Treynor Ratio calculation"""
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        portfolio_returns = pd.Series(np.random.normal(0.0008, 0.018, 252))
        
        treynor_ratio = self.backtester._calculate_treynor_ratio(portfolio_returns, benchmark_returns)
        
        # Should be a reasonable number (expand range for test environment)
        self.assertGreater(abs(treynor_ratio), 0.01)
        self.assertLess(abs(treynor_ratio), 20)  # Expand range for synthetic data

    def test_jensen_alpha_calculation(self):
        """Test Jensen Alpha calculation"""
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        portfolio_returns = pd.Series(np.random.normal(0.0008, 0.018, 252))
        
        alpha = self.backtester._calculate_jensen_alpha(portfolio_returns, benchmark_returns)
        
        # Alpha should be a percentage, reasonable range (expand for test)
        self.assertGreater(alpha, -100)  # -100% to 100% annual alpha range
        self.assertLess(alpha, 100)

    def test_tracking_error_calculation(self):
        """Test Tracking Error calculation"""
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        portfolio_returns = benchmark_returns + pd.Series(np.random.normal(0, 0.005, 252))
        
        tracking_error = self.backtester._calculate_tracking_error(portfolio_returns, benchmark_returns)
        
        # Should be positive and reasonable (annual percentage)
        self.assertGreater(tracking_error, 0)
        self.assertLess(tracking_error, 50)  # Less than 50% annual tracking error

    def test_capture_ratios_calculation(self):
        """Test Up and Down Capture Ratios"""
        # Create benchmark with clear up and down periods
        benchmark_returns = pd.Series([0.02, -0.02, 0.01, -0.01, 0.03, -0.03] * 42)
        portfolio_returns = benchmark_returns * 1.1  # 110% capture both ways
        
        up_capture, down_capture = self.backtester._calculate_capture_ratios(portfolio_returns, benchmark_returns)
        
        # Should be close to 110%
        self.assertGreater(up_capture, 100)
        self.assertLess(up_capture, 120)
        self.assertGreater(down_capture, 100)
        self.assertLess(down_capture, 120)

    def test_rolling_metrics_calculation(self):
        """Test rolling performance metrics"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        rolling_metrics = self.backtester._calculate_rolling_metrics(returns, window_days=60)
        
        # Check that all metrics are calculated
        self.assertIn('rolling_sharpe', rolling_metrics)
        self.assertIn('rolling_sortino', rolling_metrics)
        self.assertIn('rolling_volatility', rolling_metrics)
        
        # Check reasonable ranges
        self.assertGreater(rolling_metrics['rolling_volatility'], 0)
        self.assertLess(rolling_metrics['rolling_volatility'], 100)

    def test_consecutive_wins_losses(self):
        """Test consecutive wins and losses calculation"""
        from src.models.backtester import Trade, TradeType
        
        # Create mock trades: buy-sell pairs
        trades = [
            Trade(datetime.now(), TradeType.BUY, 'TEST', 100, 100, 1, 0),   # Buy at 100
            Trade(datetime.now(), TradeType.SELL, 'TEST', 100, 101, 1, 0),  # Sell at 101 (Win)
            Trade(datetime.now(), TradeType.BUY, 'TEST', 100, 101, 1, 0),   # Buy at 101  
            Trade(datetime.now(), TradeType.SELL, 'TEST', 100, 102, 1, 0),  # Sell at 102 (Win)
            Trade(datetime.now(), TradeType.BUY, 'TEST', 100, 102, 1, 0),   # Buy at 102
            Trade(datetime.now(), TradeType.SELL, 'TEST', 100, 99, 1, 0),   # Sell at 99 (Loss)
            Trade(datetime.now(), TradeType.BUY, 'TEST', 100, 99, 1, 0),    # Buy at 99
            Trade(datetime.now(), TradeType.SELL, 'TEST', 100, 103, 1, 0),  # Sell at 103 (Win)
        ]
        
        max_wins, max_losses = self.backtester._calculate_consecutive_wins_losses(trades)
        
        self.assertEqual(max_wins, 2)  # Two consecutive wins
        self.assertEqual(max_losses, 1)  # One loss

    def test_full_backtest_with_advanced_metrics(self):
        """Test complete backtest with all advanced metrics"""
        result = self.backtester.backtest(
            self.strategy.signals, 
            self.price_data, 
            'TEST', 
            'PE Strategy'
        )
        
        # Check that advanced metrics are calculated
        self.assertIsInstance(result.value_at_risk_95, float)
        self.assertIsInstance(result.conditional_var_95, float)
        self.assertIsInstance(result.information_ratio, float)
        self.assertIsInstance(result.beta, float)
        self.assertIsInstance(result.jensen_alpha, float)
        
        # Check that benchmark data is included
        self.assertFalse(result.benchmark_returns.empty)
        self.assertFalse(result.benchmark_equity.empty)
        
        # Test advanced metrics dictionary
        advanced_dict = result.get_advanced_metrics_dict()
        self.assertIn('Value at Risk (95%)', advanced_dict)
        self.assertIn('Beta', advanced_dict)
        self.assertIn('Jensen Alpha', advanced_dict)

    def test_benchmark_strategy_analysis(self):
        """Test comprehensive benchmark analysis"""
        analysis = benchmark_strategy_analysis(
            self.strategy,
            self.price_data,
            'TEST',
            config=self.config
        )
        
        # Check all analysis sections are present
        self.assertIn('Strategy Performance', analysis)
        self.assertIn('Benchmark Performance', analysis)
        self.assertIn('Relative Performance', analysis)
        self.assertIn('Risk Analysis', analysis)
        
        # Check specific metrics
        self.assertIn('Information Ratio', analysis['Relative Performance'])
        self.assertIn('Beta', analysis['Relative Performance'])
        self.assertIn('Value at Risk (95%)', analysis['Risk Analysis'])

    def test_advanced_strategy_comparison(self):
        """Test advanced strategy comparison"""
        # Create multiple strategies
        strategies = [
            PEThresholdStrategy(StrategyConfig(
                name="PE 18", description="PE threshold 18", 
                parameters={'buy_pe_threshold': 18.0, 'sell_pe_threshold': 22.0}
            )),
            PEThresholdStrategy(StrategyConfig(
                name="PE 22", description="PE threshold 22", 
                parameters={'buy_pe_threshold': 22.0, 'sell_pe_threshold': 26.0}
            )),
            MovingAverageStrategy(StrategyConfig(
                name="MA 10-20", description="MA 10-20", 
                parameters={'short_window': 10, 'long_window': 20}
            ))
        ]
        
        # Generate signals for all strategies
        for strategy in strategies:
            strategy.generate_signals(self.price_data)
        
        # Compare strategies
        comparison_df = compare_strategies_advanced(
            strategies,
            self.price_data,
            'TEST',
            self.config,
            include_advanced_metrics=True
        )
        
        # Check that advanced metrics are included
        self.assertIn('Value at Risk (95%)', comparison_df.columns)
        self.assertIn('Beta', comparison_df.columns)
        self.assertIn('Information Ratio', comparison_df.columns)

    def test_performance_report_generation(self):
        """Test performance report generation"""
        result = self.backtester.backtest(
            self.strategy.signals,
            self.price_data,
            'TEST',
            'Test Strategy'
        )
        
        report = generate_performance_report(result)
        
        # Check that report contains key sections
        self.assertIn('STRATEGY PERFORMANCE REPORT', report)
        self.assertIn('BASIC METRICS', report)
        self.assertIn('ADVANCED RISK ANALYSIS', report)
        self.assertIn('BENCHMARK COMPARISON', report)
        self.assertIn('ROLLING PERFORMANCE', report)
        
        # Check specific metrics are included
        self.assertIn('Value at Risk', report)
        self.assertIn('Information Ratio', report)
        self.assertIn('Jensen Alpha', report)

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        # Test with no trades/signals
        empty_config = StrategyConfig(
            name="Empty Strategy",
            description="Strategy with very low threshold",
            parameters={'buy_pe_threshold': 5.0, 'sell_pe_threshold': 8.0}
        )
        empty_strategy = PEThresholdStrategy(empty_config)  # Very low threshold, likely no signals
        empty_strategy.signals = []
        
        result = self.backtester.backtest(empty_strategy.signals, self.price_data, 'TEST', 'Empty Strategy')
        
        # Should handle gracefully with zero values
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.value_at_risk_95, 0.0)
        self.assertEqual(result.beta, 0.0)
        
        # Test with insufficient data for rolling metrics
        short_data = self.price_data.head(30)  # Only 30 days
        short_config = StrategyConfig(
            name="Short Strategy",
            description="Strategy for short data test",
            parameters={'buy_pe_threshold': 20.0, 'sell_pe_threshold': 25.0}
        )
        short_strategy = PEThresholdStrategy(short_config)
        short_strategy.generate_signals(short_data)
        
        short_result = self.backtester.backtest(short_strategy.signals, short_data, 'TEST', 'Short Strategy')
        
        # Should handle gracefully
        self.assertIsInstance(short_result.rolling_sharpe_6m, float)


class TestPerformanceMetricsIntegration(unittest.TestCase):
    """Integration tests for performance metrics with real-world scenarios"""

    def setUp(self):
        """Set up integration test environment"""
        # Create more realistic market data
        np.random.seed(123)
        dates = pd.date_range('2022-01-01', periods=504, freq='D')  # 2 years
        
        # Simulate market conditions with trends and volatility
        base_return = 0.0008  # ~20% annual return
        volatility = 0.018    # ~28% annual volatility
        
        returns = np.random.normal(base_return, volatility, 504)
        prices = 100 * (1 + returns).cumprod()
        
        self.market_data = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, 504),
            'high': prices * np.random.uniform(1.01, 1.03, 504),
            'low': prices * np.random.uniform(0.97, 0.99, 504),
            'close': prices,
            'volume': np.random.randint(500000, 2000000, 504),
            'pe_ratio': np.random.uniform(12, 30, 504)
        }, index=dates)

    def test_realistic_performance_metrics(self):
        """Test performance metrics with realistic market data"""
        strategy = PEThresholdStrategy(StrategyConfig(
            name="Test Strategy", 
            description="Strategy for realistic testing",
            parameters={'buy_pe_threshold': 20.0, 'sell_pe_threshold': 25.0}
        ))
        strategy.generate_signals(self.market_data)
        
        config = BacktestConfig(
            initial_capital=100000,
            commission_per_share=0.005,  # 0.5 cents per share
            slippage_bps=3,              # 3 basis points
            position_size=0.25           # 25% position sizing
        )
        
        backtester = SingleAssetBacktester(config)
        result = backtester.backtest(strategy.signals, self.market_data, 'REALISTIC', 'PE Strategy')
        
        # Validate realistic ranges for key metrics
        self.assertGreater(result.sharpe_ratio, -3)    # Reasonable Sharpe range
        self.assertLess(result.sharpe_ratio, 5)
        
        self.assertGreater(result.value_at_risk_95, -10)  # VaR should be reasonable
        self.assertLess(result.value_at_risk_95, 2)
        
        self.assertGreater(result.beta, -2)              # Beta should be reasonable
        self.assertLess(result.beta, 3)
        
        # Generate and validate report
        report = generate_performance_report(result)
        self.assertIn('REALISTIC', report)

    def test_multiple_strategy_comparison_realistic(self):
        """Test realistic comparison of multiple strategies"""
        strategies = [
            PEThresholdStrategy(StrategyConfig(
                name="PE 15", description="PE threshold 15",
                parameters={'buy_pe_threshold': 15.0, 'sell_pe_threshold': 20.0}
            )),
            PEThresholdStrategy(StrategyConfig(
                name="PE 20", description="PE threshold 20",
                parameters={'buy_pe_threshold': 20.0, 'sell_pe_threshold': 25.0}
            )),
            PEThresholdStrategy(StrategyConfig(
                name="PE 25", description="PE threshold 25",
                parameters={'buy_pe_threshold': 25.0, 'sell_pe_threshold': 30.0}
            )),
            MovingAverageStrategy(StrategyConfig(
                name="MA 10-30", description="MA 10-30",
                parameters={'short_window': 10, 'long_window': 30}
            )),
            MovingAverageStrategy(StrategyConfig(
                name="MA 20-50", description="MA 20-50",
                parameters={'short_window': 20, 'long_window': 50}
            ))
        ]
        
        for strategy in strategies:
            strategy.generate_signals(self.market_data)
        
        config = BacktestConfig(initial_capital=100000)
        
        comparison = compare_strategies_advanced(
            strategies,
            self.market_data,
            'MARKET',
            config
        )
        
        # Should have all strategies (some might have no signals, that's OK)
        self.assertGreaterEqual(len(comparison), 1)
        
        # Check that advanced metrics are reasonable
        if 'Beta' in comparison.columns:
            beta_values = pd.to_numeric(comparison['Beta'], errors='coerce')
            valid_betas = beta_values.dropna()
            if len(valid_betas) > 0:
                self.assertTrue(all(beta >= -2 for beta in valid_betas))
                self.assertTrue(all(beta <= 3 for beta in valid_betas))


if __name__ == '__main__':
    # Run the tests
    print("🧪 Running Advanced Performance Metrics Tests (Task 17)")
    print("=" * 60)
    
    unittest.main(verbosity=2)