"""
Strategy Forge - Single Stock CLI Runner

This module provides a command-line interface for running complete single-stock
backtesting workflows. It orchestrates the entire pipeline from data fetching
to strategy execution and performance reporting.

Usage:
    python -m src.cli.single_stock_runner --symbol AAPL --strategy pe_threshold
    python main.py --symbol RELIANCE.NS --strategy moving_average --start-date 2023-01-01
    
Features:
    - Complete data pipeline integration
    - Multiple strategy options with configuration
    - Comprehensive performance reporting
    - Export capabilities (JSON, CSV)
    - Advanced performance metrics
    - Benchmark comparison

Author: Strategy Forge Development Team
Version: 1.0
"""

import argparse
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Strategy Forge imports
from src.data.processing_pipeline import DataProcessingPipeline, PipelineConfig
from src.models.strategies import (
    StrategyConfig, PEThresholdStrategy, MovingAverageStrategy,
    CombinedStrategy, create_strategy
)
from src.models.backtester import SingleAssetBacktester, BacktestConfig, BacktestResult
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SingleStockCLI:
    """
    Command-line interface for single stock backtesting operations
    """
    
    def __init__(self):
        """Initialize CLI with available strategies and default configurations"""
        self.available_strategies = {
            'pe_threshold': {
                'class': PEThresholdStrategy,
                'description': 'Buy when P/E ratio below threshold, sell when above',
                'default_config': {
                    'buy_pe_threshold': 15.0,
                    'sell_pe_threshold': 25.0,
                    'min_signal_strength': 0.1
                }
            },
            'moving_average': {
                'class': MovingAverageStrategy,
                'description': 'Buy/sell based on moving average crossovers',
                'default_config': {
                    'short_window': 20,
                    'long_window': 50,
                    'min_signal_strength': 0.2,
                    'ma_type': 'SMA',
                    'price_column': 'Close'  # Changed from 'close' to 'Close'
                }
            },
            'combined': {
                'class': CombinedStrategy,
                'description': 'Combines P/E and moving average signals',
                'default_config': {
                    'buy_pe_threshold': 18.0,
                    'sell_pe_threshold': 28.0,
                    'short_window': 20,
                    'long_window': 50,
                    'min_signal_strength': 0.15
                }
            }
        }
        
        self.pipeline = None
        self.backtester = None
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(
            description="Strategy Forge - Single Stock Backtesting CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )
        
        # Required arguments (unless using special commands)
        parser.add_argument(
            '--symbol', '-s',
            type=str,
            required=False,  # Will be validated later if not using special commands
            help='Stock symbol to analyze (e.g., AAPL, RELIANCE.NS)'
        )
        
        parser.add_argument(
            '--strategy', '-st',
            type=str,
            required=False,  # Will be validated later if not using special commands
            choices=list(self.available_strategies.keys()),
            help='Trading strategy to use'
        )
        
        # Date range arguments
        parser.add_argument(
            '--start-date',
            type=str,
            default=None,
            help='Start date for analysis (YYYY-MM-DD format, default: 2 years ago)'
        )
        
        parser.add_argument(
            '--end-date',
            type=str,
            default=None,
            help='End date for analysis (YYYY-MM-DD format, default: today)'
        )
        
        # Strategy configuration
        parser.add_argument(
            '--strategy-config',
            type=str,
            default=None,
            help='JSON string or file path for strategy configuration'
        )
        
        # Backtesting parameters
        parser.add_argument(
            '--initial-capital',
            type=float,
            default=1000000.0,
            help='Initial capital for backtesting (default: $1,000,000)'
        )
        
        parser.add_argument(
            '--position-size',
            type=float,
            default=0.15,
            help='Position size as fraction of capital (default: 0.15 = 15%%)'
        )
        
        parser.add_argument(
            '--commission',
            type=float,
            default=0.005,
            help='Commission per share in currency units (default: 0.005)'
        )
        
        parser.add_argument(
            '--slippage',
            type=float,
            default=0.0003,
            help='Slippage as fraction of price (default: 0.0003 = 3 basis points)'
        )
        
        # Risk management
        parser.add_argument(
            '--stop-loss',
            type=float,
            default=0.08,
            help='Stop loss threshold as fraction (default: 0.08 = 8%%)'
        )
        
        parser.add_argument(
            '--take-profit',
            type=float,
            default=0.20,
            help='Take profit threshold as fraction (default: 0.20 = 20%%)'
        )
        
        # Output options
        parser.add_argument(
            '--output-format',
            type=str,
            choices=['console', 'json', 'csv', 'all'],
            default='console',
            help='Output format for results'
        )
        
        parser.add_argument(
            '--output-file',
            type=str,
            default=None,
            help='File path for saving results (auto-generated if not specified)'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        parser.add_argument(
            '--list-strategies',
            action='store_true',
            help='List available strategies and exit'
        )
        
        parser.add_argument(
            '--benchmark',
            action='store_true',
            help='Include benchmark (buy & hold) comparison'
        )
        
        parser.add_argument(
            '--advanced-metrics',
            action='store_true',
            help='Calculate advanced performance metrics (VaR, Information Ratio, etc.)'
        )
        
        return parser
    
    def _get_usage_examples(self) -> str:
        """Get usage examples for help text"""
        return """
Examples:
  # Basic backtest with P/E threshold strategy
  python main.py --symbol AAPL --strategy pe_threshold
  
  # Custom date range and parameters
  python main.py --symbol RELIANCE.NS --strategy moving_average \\
                 --start-date 2023-01-01 --end-date 2023-12-31 \\
                 --initial-capital 5000000 --position-size 0.10
  
  # Advanced analysis with benchmark comparison
  python main.py --symbol MSFT --strategy combined \\
                 --benchmark --advanced-metrics \\
                 --output-format json --output-file results.json
  
  # Custom strategy configuration
  python main.py --symbol GOOGL --strategy pe_threshold \\
                 --strategy-config '{"buy_pe_threshold": 12.0, "sell_pe_threshold": 22.0}'
  
  # List available strategies
  python main.py --list-strategies

Available Strategies:
  pe_threshold    - Buy when P/E below threshold, sell when above
  moving_average  - Buy/sell based on moving average crossovers  
  combined       - Combines P/E and moving average signals
        """
    
    def list_strategies(self):
        """List available strategies with descriptions"""
        print("\nAvailable Trading Strategies")
        print("=" * 50)
        
        for name, info in self.available_strategies.items():
            print(f"\n{name}")
            print(f"   Description: {info['description']}")
            print(f"   Default Config:")
            for key, value in info['default_config'].items():
                print(f"     {key}: {value}")
    
    def parse_strategy_config(self, config_str: Optional[str], strategy_name: str) -> Dict[str, Any]:
        """Parse strategy configuration from string or file"""
        if not config_str:
            return self.available_strategies[strategy_name]['default_config'].copy()
        
        try:
            # Try to parse as JSON string first
            if config_str.startswith('{'):
                config = json.loads(config_str)
            else:
                # Try to read as file
                with open(config_str, 'r') as f:
                    config = json.load(f)
            
            # Merge with defaults
            default_config = self.available_strategies[strategy_name]['default_config'].copy()
            default_config.update(config)
            return default_config
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to parse strategy config: {e}")
            logger.info("Using default configuration")
            return self.available_strategies[strategy_name]['default_config'].copy()
    
    def validate_arguments(self, args) -> bool:
        """Validate command line arguments"""
        errors = []
        
        # Skip validation for special commands
        if hasattr(args, 'list_strategies') and args.list_strategies:
            return True
        
        # Validate required arguments for normal operation
        if not args.symbol:
            errors.append("Symbol is required for backtesting operations")
        
        if not args.strategy:
            errors.append("Strategy is required for backtesting operations")
        
        # Validate symbol format
        if args.symbol and len(args.symbol) < 1:
            errors.append("Symbol must be a valid stock ticker")
        
        # Validate date formats
        date_format = "%Y-%m-%d"
        if args.start_date:
            try:
                datetime.strptime(args.start_date, date_format)
            except ValueError:
                errors.append("Start date must be in YYYY-MM-DD format")
        
        if args.end_date:
            try:
                datetime.strptime(args.end_date, date_format)
            except ValueError:
                errors.append("End date must be in YYYY-MM-DD format")
        
        # Validate numerical parameters
        if args.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if not 0 < args.position_size <= 1:
            errors.append("Position size must be between 0 and 1")
        
        if args.commission < 0:
            errors.append("Commission cannot be negative")
        
        if args.slippage < 0:
            errors.append("Slippage cannot be negative")
        
        if not 0 < args.stop_loss < 1:
            errors.append("Stop loss must be between 0 and 1")
        
        if not 0 < args.take_profit < 1:
            errors.append("Take profit must be between 0 and 1")
        
        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return False
        
        return True
    
    def setup_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> tuple:
        """Setup and validate date range"""
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            # Default to 2 years ago
            start_dt = datetime.now() - timedelta(days=730)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"Analysis period: {start_date} to {end_date}")
        return start_date, end_date
    
    def fetch_and_process_data(self, symbol: str, start_date: str, end_date: str) -> Optional[object]:
        """Fetch and process stock data"""
        try:
            logger.info(f"🔄 Fetching and processing data for {symbol}")
            
            # Initialize pipeline if not already done
            if not self.pipeline:
                config = PipelineConfig(
                    start_date=start_date,
                    end_date=end_date,
                    reporting_lag_days=45
                )
                self.pipeline = DataProcessingPipeline(config)
            
            # Process the stock data
            data = self.pipeline.process_stock(symbol, start_date, end_date)
            
            if data is None or data.empty:
                logger.error(f"No data available for {symbol}")
                return None
            
            logger.info(f"✅ Successfully processed {len(data)} trading days")
            logger.info(f"   Data columns: {list(data.columns)}")
            logger.info(f"   Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch/process data for {symbol}: {e}")
            return None
    
    def create_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]) -> Optional[object]:
        """Create and configure strategy instance"""
        try:
            logger.info(f"🎯 Creating {strategy_name} strategy")
            
            # Get strategy class
            strategy_info = self.available_strategies[strategy_name]
            strategy_class = strategy_info['class']
            
            # Create strategy configuration with proper structure
            config = StrategyConfig(
                name=f"{strategy_name.title()} Strategy",
                description=strategy_info['description'],
                parameters=strategy_config  # Pass strategy-specific params in parameters dict
            )
            
            # Instantiate strategy
            strategy = strategy_class(config)
            
            logger.info(f"✅ Strategy created with config: {strategy_config}")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to create strategy: {e}")
            return None
    
    def run_backtest(
        self, 
        strategy: object, 
        data: object, 
        symbol: str, 
        backtest_config: Dict[str, Any]
    ) -> Optional[BacktestResult]:
        """Run backtest with given strategy and data"""
        try:
            logger.info(f"📈 Running backtest for {symbol}")
            
            # Create backtest configuration
            config = BacktestConfig(
                initial_capital=backtest_config['initial_capital'],
                commission_per_share=backtest_config['commission'],
                slippage_bps=int(backtest_config['slippage'] * 10000),  # Convert to basis points
                position_size=backtest_config['position_size'],
                stop_loss_pct=backtest_config['stop_loss'],
                take_profit_pct=backtest_config['take_profit']
            )
            
            # Initialize backtester
            if not self.backtester:
                self.backtester = SingleAssetBacktester(config)
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            if not signals:
                logger.warning("No trading signals generated")
                return None
            
            logger.info(f"Generated {len(signals)} trading signals")
            
            # Run backtest
            result = self.backtester.backtest(signals, data, symbol, strategy.name)
            
            logger.info(f"✅ Backtest completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None
    
    def format_output(
        self, 
        result: BacktestResult, 
        symbol: str, 
        format_type: str,
        include_advanced: bool = False,
        include_benchmark: bool = False
    ) -> str:
        """Format backtest results for output"""
        if format_type == 'json':
            return self._format_json_output(result, symbol, include_advanced, include_benchmark)
        elif format_type == 'csv':
            return self._format_csv_output(result, symbol)
        else:  # console
            return self._format_console_output(result, symbol, include_advanced, include_benchmark)
    
    def _format_console_output(
        self, 
        result: BacktestResult, 
        symbol: str,
        include_advanced: bool = False,
        include_benchmark: bool = False
    ) -> str:
        """Format results for console display"""
        output = []
        output.append(f"\n🎯 STRATEGY FORGE - BACKTEST RESULTS")
        output.append("=" * 60)
        output.append(f"📊 Symbol: {symbol}")
        output.append(f"🎯 Strategy: {result.strategy_name}")
        output.append(f"📅 Period: {result.start_date} to {result.end_date}")
        output.append(f"💰 Initial Capital: ${result.initial_capital:,.2f}")
        output.append(f"💎 Final Value: ${result.final_value:,.2f}")
        output.append("")
        
        # Performance metrics
        output.append("📈 PERFORMANCE SUMMARY")
        output.append("-" * 30)
        output.append(f"Total Return: {result.total_return:.2%}")
        output.append(f"Annualized Return: {result.annualized_return:.2%}")
        output.append(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        output.append(f"Max Drawdown: {result.max_drawdown:.2%}")
        output.append(f"Volatility: {result.volatility:.2%}")
        output.append("")
        
        # Trading statistics
        output.append("📊 TRADING STATISTICS")
        output.append("-" * 30)
        output.append(f"Total Trades: {result.total_trades}")
        output.append(f"Winning Trades: {result.winning_trades}")
        output.append(f"Win Rate: {result.win_rate:.1%}")
        output.append(f"Average Win: {result.avg_win:.2%}")
        output.append(f"Average Loss: {result.avg_loss:.2%}")
        output.append(f"Profit Factor: {result.profit_factor:.2f}")
        output.append("")
        
        # Advanced metrics if requested
        if include_advanced and hasattr(result, 'sortino_ratio'):
            output.append("🔬 ADVANCED RISK METRICS")
            output.append("-" * 30)
            output.append(f"Sortino Ratio: {getattr(result, 'sortino_ratio', 'N/A'):.3f}")
            output.append(f"Calmar Ratio: {getattr(result, 'calmar_ratio', 'N/A'):.3f}")
            output.append(f"Value at Risk (95%): {getattr(result, 'value_at_risk_95', 'N/A'):.2%}")
            output.append(f"Conditional VaR (95%): {getattr(result, 'conditional_var_95', 'N/A'):.2%}")
            output.append(f"Information Ratio: {getattr(result, 'information_ratio', 'N/A'):.3f}")
            output.append(f"Jensen Alpha: {getattr(result, 'jensen_alpha', 'N/A'):.2%}")
            output.append("")
        
        # Benchmark comparison if requested
        if include_benchmark:
            output.append("🎯 BENCHMARK COMPARISON")
            output.append("-" * 30)
            output.append("(Buy & Hold Strategy)")
            # This would require implementing benchmark calculation
            output.append("Benchmark analysis not yet implemented")
            output.append("")
        
        # Cost analysis
        output.append("💸 COST ANALYSIS")
        output.append("-" * 30)
        output.append(f"Total Commission: ${result.total_commission:.2f}")
        output.append(f"Total Slippage: ${result.total_slippage:.2f}")
        output.append(f"Total Costs: ${result.total_commission + result.total_slippage:.2f}")
        output.append("")
        
        output.append("=" * 60)
        output.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(output)
    
    def _format_json_output(
        self, 
        result: BacktestResult, 
        symbol: str,
        include_advanced: bool = False,
        include_benchmark: bool = False
    ) -> str:
        """Format results as JSON"""
        output_data = {
            'metadata': {
                'symbol': symbol,
                'strategy': result.strategy_name,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'generated_at': datetime.now().isoformat()
            },
            'performance': {
                'initial_capital': result.initial_capital,
                'final_value': result.final_value,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility
            },
            'trading': {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'win_rate': result.win_rate,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
                'profit_factor': result.profit_factor
            },
            'costs': {
                'total_commission': result.total_commission,
                'total_slippage': result.total_slippage,
                'total_costs': result.total_commission + result.total_slippage
            }
        }
        
        # Add advanced metrics if available and requested
        if include_advanced:
            advanced_metrics = {}
            for attr in ['sortino_ratio', 'calmar_ratio', 'value_at_risk_95', 
                        'conditional_var_95', 'information_ratio', 'jensen_alpha']:
                if hasattr(result, attr):
                    advanced_metrics[attr] = getattr(result, attr)
            
            if advanced_metrics:
                output_data['advanced_metrics'] = advanced_metrics
        
        return json.dumps(output_data, indent=2, default=str)
    
    def _format_csv_output(self, result: BacktestResult, symbol: str) -> str:
        """Format results as CSV"""
        # This would create a CSV with trade-by-trade results
        # For now, return a summary CSV
        csv_lines = [
            "metric,value",
            f"symbol,{symbol}",
            f"strategy,{result.strategy_name}",
            f"total_return,{result.total_return}",
            f"sharpe_ratio,{result.sharpe_ratio}",
            f"max_drawdown,{result.max_drawdown}",
            f"total_trades,{result.total_trades}",
            f"win_rate,{result.win_rate}",
            f"profit_factor,{result.profit_factor}"
        ]
        return "\n".join(csv_lines)
    
    def save_output(self, output: str, filename: str, format_type: str):
        """Save output to file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = 'json' if format_type == 'json' else 'csv' if format_type == 'csv' else 'txt'
                filename = f"backtest_results_{timestamp}.{extension}"
            
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(output)
            
            logger.info(f"✅ Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
    
    def run(self, args):
        """Main execution method"""
        try:
            # Set up logging level
            if args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            
            logger.info("Starting Strategy Forge CLI")
            
            # Validate arguments
            if not self.validate_arguments(args):
                return 1
            
            # Setup date range
            start_date, end_date = self.setup_date_range(args.start_date, args.end_date)
            
            # Fetch and process data
            data = self.fetch_and_process_data(args.symbol, start_date, end_date)
            if data is None:
                logger.error("Failed to fetch data. Exiting.")
                return 1
            
            # Parse strategy configuration
            strategy_config = self.parse_strategy_config(args.strategy_config, args.strategy)
            
            # Create strategy
            strategy = self.create_strategy(args.strategy, strategy_config)
            if strategy is None:
                logger.error("Failed to create strategy. Exiting.")
                return 1
            
            # Prepare backtest configuration
            backtest_config = {
                'initial_capital': args.initial_capital,
                'commission': args.commission,
                'slippage': args.slippage,
                'position_size': args.position_size,
                'stop_loss': args.stop_loss,
                'take_profit': args.take_profit
            }
            
            # Run backtest
            result = self.run_backtest(strategy, data, args.symbol, backtest_config)
            if result is None:
                logger.error("Backtest failed. Exiting.")
                return 1
            
            # Format and display results
            if args.output_format in ['console', 'all']:
                console_output = self.format_output(
                    result, args.symbol, 'console', 
                    args.advanced_metrics, args.benchmark
                )
                print(console_output)
            
            # Save to file if requested
            if args.output_format in ['json', 'csv', 'all']:
                for fmt in (['json', 'csv'] if args.output_format == 'all' else [args.output_format]):
                    if fmt != 'console':
                        output_content = self.format_output(
                            result, args.symbol, fmt,
                            args.advanced_metrics, args.benchmark
                        )
                        
                        # Generate filename for this format
                        if args.output_file:
                            base_name = Path(args.output_file).stem
                            extension = 'json' if fmt == 'json' else 'csv'
                            filename = f"{base_name}.{extension}"
                        else:
                            filename = None
                        
                        self.save_output(output_content, filename, fmt)
            
            logger.info("CLI execution completed successfully")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main entry point for CLI"""
    cli = SingleStockCLI()
    parser = cli.create_argument_parser()
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_strategies:
        cli.list_strategies()
        return 0
    
    # Run main CLI
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())