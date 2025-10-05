"""
Test Suite for Single Stock CLI Runner

This module provides comprehensive tests for the CLI functionality including
argument parsing, validation, data integration, strategy execution, and output formatting.

Test Coverage:
- Argument parsing and validation
- Strategy configuration
- Data integration
- Output formatting (console, JSON, CSV)
- Error handling scenarios
- Integration testing

Author: Strategy Forge Development Team
Version: 1.0
"""

import pytest
import sys
import os
import json
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from io import StringIO
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.single_stock_runner import SingleStockCLI
from src.models.backtester import BacktestResult


class TestArgumentParsing:
    """Test argument parsing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
        self.parser = self.cli.create_argument_parser()
    
    def test_required_arguments(self):
        """Test that required arguments are enforced"""
        # Test that validation catches missing required args for normal operation
        cli = SingleStockCLI()
        
        # Missing symbol should fail validation
        args = Mock(
            symbol=None, 
            strategy='pe_threshold', 
            list_strategies=False,
            start_date=None,
            end_date=None,
            initial_capital=1000000.0,
            position_size=0.15,
            commission=0.005,
            slippage=0.0003,
            stop_loss=0.08,
            take_profit=0.20
        )
        assert cli.validate_arguments(args) == False
        
        # Missing strategy should fail validation
        args = Mock(
            symbol='AAPL', 
            strategy=None, 
            list_strategies=False,
            start_date=None,
            end_date=None,
            initial_capital=1000000.0,
            position_size=0.15,
            commission=0.005,
            slippage=0.0003,
            stop_loss=0.08,
            take_profit=0.20
        )
        assert cli.validate_arguments(args) == False
        
        # Both required arguments should pass validation
        args = Mock(
            symbol='AAPL', 
            strategy='pe_threshold', 
            list_strategies=False,
            start_date=None,
            end_date=None,
            initial_capital=1000000.0,
            position_size=0.15,
            commission=0.005,
            slippage=0.0003,
            stop_loss=0.08,
            take_profit=0.20
        )
        assert cli.validate_arguments(args) == True
    
    def test_default_values(self):
        """Test default argument values"""
        args = self.parser.parse_args(['--symbol', 'AAPL', '--strategy', 'pe_threshold'])
        
        assert args.initial_capital == 1000000.0
        assert args.position_size == 0.15
        assert args.commission == 0.005
        assert args.slippage == 0.0003
        assert args.stop_loss == 0.08
        assert args.take_profit == 0.20
        assert args.output_format == 'console'
        assert args.verbose == False
        assert args.benchmark == False
        assert args.advanced_metrics == False
    
    def test_optional_arguments(self):
        """Test optional argument parsing"""
        args = self.parser.parse_args([
            '--symbol', 'MSFT',
            '--strategy', 'moving_average',
            '--start-date', '2023-01-01',
            '--end-date', '2023-12-31',
            '--initial-capital', '5000000',
            '--position-size', '0.10',
            '--commission', '0.01',
            '--slippage', '0.0005',
            '--stop-loss', '0.05',
            '--take-profit', '0.25',
            '--output-format', 'json',
            '--verbose',
            '--benchmark',
            '--advanced-metrics'
        ])
        
        assert args.symbol == 'MSFT'
        assert args.strategy == 'moving_average'
        assert args.start_date == '2023-01-01'
        assert args.end_date == '2023-12-31'
        assert args.initial_capital == 5000000.0
        assert args.position_size == 0.10
        assert args.commission == 0.01
        assert args.slippage == 0.0005
        assert args.stop_loss == 0.05
        assert args.take_profit == 0.25
        assert args.output_format == 'json'
        assert args.verbose == True
        assert args.benchmark == True
        assert args.advanced_metrics == True
    
    def test_strategy_choices(self):
        """Test strategy choice validation"""
        # Valid strategy should work
        args = self.parser.parse_args(['--symbol', 'AAPL', '--strategy', 'pe_threshold'])
        assert args.strategy == 'pe_threshold'
        
        # Invalid strategy should fail
        with pytest.raises(SystemExit):
            self.parser.parse_args(['--symbol', 'AAPL', '--strategy', 'invalid_strategy'])
    
    def test_output_format_choices(self):
        """Test output format validation"""
        valid_formats = ['console', 'json', 'csv', 'all']
        
        for fmt in valid_formats:
            args = self.parser.parse_args(['--symbol', 'AAPL', '--strategy', 'pe_threshold', '--output-format', fmt])
            assert args.output_format == fmt
        
        # Invalid format should fail
        with pytest.raises(SystemExit):
            self.parser.parse_args(['--symbol', 'AAPL', '--strategy', 'pe_threshold', '--output-format', 'invalid'])


class TestArgumentValidation:
    """Test argument validation logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
    
    def create_mock_args(self, **kwargs):
        """Create mock arguments with defaults"""
        defaults = {
            'symbol': 'AAPL',
            'strategy': 'pe_threshold',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 1000000.0,
            'position_size': 0.15,
            'commission': 0.005,
            'slippage': 0.0003,
            'stop_loss': 0.08,
            'take_profit': 0.20
        }
        defaults.update(kwargs)
        return Mock(**defaults)
    
    def test_valid_arguments(self):
        """Test validation with valid arguments"""
        args = self.create_mock_args()
        assert self.cli.validate_arguments(args) == True
    
    def test_invalid_symbol(self):
        """Test symbol validation"""
        args = self.create_mock_args(symbol='')
        assert self.cli.validate_arguments(args) == False
        
        args = self.create_mock_args(symbol=None)
        assert self.cli.validate_arguments(args) == False
    
    def test_invalid_dates(self):
        """Test date format validation"""
        args = self.create_mock_args(start_date='invalid-date')
        assert self.cli.validate_arguments(args) == False
        
        args = self.create_mock_args(end_date='2023-13-45')  # Invalid date
        assert self.cli.validate_arguments(args) == False
    
    def test_invalid_numerical_parameters(self):
        """Test numerical parameter validation"""
        # Negative capital
        args = self.create_mock_args(initial_capital=-1000)
        assert self.cli.validate_arguments(args) == False
        
        # Invalid position size
        args = self.create_mock_args(position_size=1.5)  # > 1
        assert self.cli.validate_arguments(args) == False
        
        args = self.create_mock_args(position_size=0)  # = 0
        assert self.cli.validate_arguments(args) == False
        
        # Negative commission
        args = self.create_mock_args(commission=-0.01)
        assert self.cli.validate_arguments(args) == False
        
        # Negative slippage
        args = self.create_mock_args(slippage=-0.001)
        assert self.cli.validate_arguments(args) == False
        
        # Invalid stop loss
        args = self.create_mock_args(stop_loss=1.5)  # > 1
        assert self.cli.validate_arguments(args) == False
        
        # Invalid take profit
        args = self.create_mock_args(take_profit=0)  # = 0
        assert self.cli.validate_arguments(args) == False


class TestStrategyConfiguration:
    """Test strategy configuration parsing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
    
    def test_default_configuration(self):
        """Test default configuration loading"""
        config = self.cli.parse_strategy_config(None, 'pe_threshold')
        expected = self.cli.available_strategies['pe_threshold']['default_config']
        assert config == expected
    
    def test_json_string_configuration(self):
        """Test JSON string configuration parsing"""
        json_config = '{"buy_pe_threshold": 12.0, "sell_pe_threshold": 20.0}'
        config = self.cli.parse_strategy_config(json_config, 'pe_threshold')
        
        assert config['buy_pe_threshold'] == 12.0
        assert config['sell_pe_threshold'] == 20.0
        # Should preserve other defaults
        assert 'min_signal_strength' in config
    
    def test_file_configuration(self):
        """Test configuration file loading"""
        config_data = {
            "buy_pe_threshold": 10.0,
            "sell_pe_threshold": 30.0,
            "min_signal_strength": 0.2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = self.cli.parse_strategy_config(config_file, 'pe_threshold')
            assert config['buy_pe_threshold'] == 10.0
            assert config['sell_pe_threshold'] == 30.0
            assert config['min_signal_strength'] == 0.2
        finally:
            os.unlink(config_file)
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        # Invalid JSON should fall back to defaults
        config = self.cli.parse_strategy_config('invalid json', 'pe_threshold')
        expected = self.cli.available_strategies['pe_threshold']['default_config']
        assert config == expected
        
        # Non-existent file should fall back to defaults
        config = self.cli.parse_strategy_config('/non/existent/file.json', 'pe_threshold')
        assert config == expected


class TestDataIntegration:
    """Test data fetching and processing integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
    
    @patch('src.cli.single_stock_runner.DataProcessingPipeline')
    def test_successful_data_fetch(self, mock_pipeline_class):
        """Test successful data fetching"""
        # Create mock data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'EPS': [5, 5, 5, 5, 5],
            'pe_ratio': [20, 20.2, 20.4, 20.6, 20.8]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process_stock.return_value = mock_data
        mock_pipeline_class.return_value = mock_pipeline
        
        # Test data fetching
        result = self.cli.fetch_and_process_data('AAPL', '2023-01-01', '2023-01-05')
        
        assert result is not None
        assert len(result) == 5
        mock_pipeline.process_stock.assert_called_once_with('AAPL', '2023-01-01', '2023-01-05')
    
    @patch('src.cli.single_stock_runner.DataProcessingPipeline')
    def test_failed_data_fetch(self, mock_pipeline_class):
        """Test handling of data fetch failures"""
        # Setup mock pipeline to raise exception
        mock_pipeline = Mock()
        mock_pipeline.process_stock.side_effect = Exception("Data fetch failed")
        mock_pipeline_class.return_value = mock_pipeline
        
        # Test error handling
        result = self.cli.fetch_and_process_data('INVALID', '2023-01-01', '2023-01-05')
        assert result is None
    
    @patch('src.cli.single_stock_runner.DataProcessingPipeline')
    def test_empty_data_handling(self, mock_pipeline_class):
        """Test handling of empty data"""
        # Setup mock pipeline to return empty DataFrame
        mock_pipeline = Mock()
        mock_pipeline.process_stock.return_value = pd.DataFrame()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Test empty data handling
        result = self.cli.fetch_and_process_data('EMPTY', '2023-01-01', '2023-01-05')
        assert result is None


class TestStrategyCreation:
    """Test strategy creation and configuration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
    
    def test_pe_threshold_strategy_creation(self):
        """Test P/E threshold strategy creation"""
        config = {
            'buy_pe_threshold': 15.0,
            'sell_pe_threshold': 25.0,
            'min_signal_strength': 0.1
        }
        
        strategy = self.cli.create_strategy('pe_threshold', config)
        assert strategy is not None
        assert strategy.config.buy_pe_threshold == 15.0
        assert strategy.config.sell_pe_threshold == 25.0
    
    def test_moving_average_strategy_creation(self):
        """Test moving average strategy creation"""
        config = {
            'short_window': 20,
            'long_window': 50,
            'min_signal_strength': 0.2
        }
        
        strategy = self.cli.create_strategy('moving_average', config)
        assert strategy is not None
        assert strategy.config.short_window == 20
        assert strategy.config.long_window == 50
    
    def test_combined_strategy_creation(self):
        """Test combined strategy creation"""
        config = {
            'buy_pe_threshold': 18.0,
            'sell_pe_threshold': 28.0,
            'short_window': 20,
            'long_window': 50,
            'min_signal_strength': 0.15
        }
        
        strategy = self.cli.create_strategy('combined', config)
        assert strategy is not None
        assert hasattr(strategy.config, 'buy_pe_threshold')
        assert hasattr(strategy.config, 'short_window')


class TestOutputFormatting:
    """Test output formatting functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
        
        # Create mock backtest result
        self.mock_result = Mock(spec=BacktestResult)
        self.mock_result.strategy_name = "Test Strategy"
        self.mock_result.start_date = "2023-01-01"
        self.mock_result.end_date = "2023-12-31"
        self.mock_result.initial_capital = 1000000.0
        self.mock_result.final_value = 1150000.0
        self.mock_result.total_return = 0.15
        self.mock_result.annualized_return = 0.15
        self.mock_result.sharpe_ratio = 1.2
        self.mock_result.max_drawdown = 0.08
        self.mock_result.volatility = 0.12
        self.mock_result.total_trades = 25
        self.mock_result.winning_trades = 15
        self.mock_result.win_rate = 0.6
        self.mock_result.avg_win = 0.025
        self.mock_result.avg_loss = -0.015
        self.mock_result.profit_factor = 1.67
        self.mock_result.total_commission = 125.0
        self.mock_result.total_slippage = 345.0
    
    def test_console_output_format(self):
        """Test console output formatting"""
        output = self.cli.format_output(self.mock_result, 'AAPL', 'console')
        
        assert 'STRATEGY FORGE - BACKTEST RESULTS' in output
        assert 'AAPL' in output
        assert 'Test Strategy' in output
        assert '15.00%' in output  # Total return
        assert '1.200' in output  # Sharpe ratio
        assert '25' in output     # Total trades
        assert '60.0%' in output  # Win rate
    
    def test_json_output_format(self):
        """Test JSON output formatting"""
        output = self.cli.format_output(self.mock_result, 'AAPL', 'json')
        
        # Parse JSON to verify structure
        data = json.loads(output)
        
        assert data['metadata']['symbol'] == 'AAPL'
        assert data['metadata']['strategy'] == 'Test Strategy'
        assert data['performance']['total_return'] == 0.15
        assert data['performance']['sharpe_ratio'] == 1.2
        assert data['trading']['total_trades'] == 25
        assert data['trading']['win_rate'] == 0.6
        assert data['costs']['total_commission'] == 125.0
    
    def test_csv_output_format(self):
        """Test CSV output formatting"""
        output = self.cli.format_output(self.mock_result, 'AAPL', 'csv')
        
        lines = output.strip().split('\n')
        assert lines[0] == 'metric,value'
        assert 'symbol,AAPL' in lines
        assert 'strategy,Test Strategy' in lines
        assert 'total_return,0.15' in lines
    
    def test_advanced_metrics_output(self):
        """Test output with advanced metrics"""
        # Add advanced metrics to mock result
        self.mock_result.sortino_ratio = 1.5
        self.mock_result.calmar_ratio = 1.8
        self.mock_result.value_at_risk_95 = 0.025
        self.mock_result.information_ratio = 0.8
        
        output = self.cli.format_output(
            self.mock_result, 'AAPL', 'console', 
            include_advanced=True
        )
        
        assert 'ADVANCED RISK METRICS' in output
        assert 'Sortino Ratio: 1.500' in output
        assert 'Calmar Ratio: 1.800' in output
        assert 'Value at Risk (95%): 2.50%' in output


class TestDateHandling:
    """Test date range setup and validation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
    
    def test_explicit_date_range(self):
        """Test explicit date range setup"""
        start, end = self.cli.setup_date_range('2023-01-01', '2023-12-31')
        assert start == '2023-01-01'
        assert end == '2023-12-31'
    
    def test_default_end_date(self):
        """Test default end date (today)"""
        today = datetime.now().strftime('%Y-%m-%d')
        start, end = self.cli.setup_date_range('2023-01-01', None)
        assert start == '2023-01-01'
        assert end == today
    
    def test_default_start_date(self):
        """Test default start date (2 years ago)"""
        expected_start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        
        start, end = self.cli.setup_date_range(None, None)
        assert start == expected_start
        assert end == today


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
    
    def test_strategy_list_command(self):
        """Test strategy listing functionality"""
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            self.cli.list_strategies()
            output = captured_output.getvalue()
            
            assert 'Available Trading Strategies' in output
            assert 'pe_threshold' in output
            assert 'moving_average' in output
            assert 'combined' in output
        finally:
            sys.stdout = old_stdout


class TestIntegration:
    """Integration tests for complete CLI workflows"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cli = SingleStockCLI()
    
    def create_mock_data(self):
        """Create realistic mock data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        
        return pd.DataFrame({
            'Close': prices,
            'EPS': [5.0] * 100,
            'pe_ratio': prices / 5.0
        }, index=dates)
    
    @patch('src.cli.single_stock_runner.DataProcessingPipeline')
    @patch('src.cli.single_stock_runner.SingleAssetBacktester')
    def test_complete_workflow(self, mock_backtester_class, mock_pipeline_class):
        """Test complete CLI workflow"""
        # Setup mock data
        mock_data = self.create_mock_data()
        mock_pipeline = Mock()
        mock_pipeline.process_stock.return_value = mock_data
        mock_pipeline_class.return_value = mock_pipeline
        
        # Setup mock backtester
        mock_result = Mock(spec=BacktestResult)
        mock_result.strategy_name = "PE Threshold Strategy"
        mock_result.total_return = 0.12
        mock_result.sharpe_ratio = 1.1
        mock_result.total_trades = 20
        
        mock_backtester = Mock()
        mock_backtester.backtest.return_value = mock_result
        mock_backtester_class.return_value = mock_backtester
        
        # Create mock strategy
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = [Mock()] * 10
        mock_strategy.name = "PE Threshold Strategy"
        
        # Mock strategy creation
        with patch.object(self.cli, 'create_strategy', return_value=mock_strategy):
            # Create mock arguments
            args = Mock()
            args.symbol = 'AAPL'
            args.strategy = 'pe_threshold'
            args.start_date = '2023-01-01'
            args.end_date = '2023-12-31'
            args.strategy_config = None
            args.initial_capital = 1000000.0
            args.commission = 0.005
            args.slippage = 0.0003
            args.position_size = 0.15
            args.stop_loss = 0.08
            args.take_profit = 0.20
            args.output_format = 'console'
            args.output_file = None
            args.verbose = False
            args.benchmark = False
            args.advanced_metrics = False
            
            # Run CLI
            result = self.cli.run(args)
            
            # Verify success
            assert result == 0
            
            # Verify method calls
            mock_pipeline.process_stock.assert_called_once()
            mock_strategy.generate_signals.assert_called_once()
            mock_backtester.backtest.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])