"""
Strategy Forge - End-to-End Trading Strategy Demonstration

This script demonstrates the complete workflow from raw data to trading signals:
1. Fetch and process stock data using DataProcessingPipeline
2. Generate trading signals using various strategies
3. Analyze signal quality and performance
4. Display comprehensive results and insights

This serves as a practical example of how to use the Strategy Forge framework
for real-world quantitative trading strategy development.

Author: Strategy Forge Development Team
Version: 1.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import Strategy Forge modules
from src.data.processing_pipeline import DataProcessingPipeline, PipelineConfig
from src.models.strategies import (
    StrategyConfig, PEThresholdStrategy, MovingAverageStrategy, 
    CombinedStrategy, create_strategy
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyDemonstrator:
    """
    Demonstrates end-to-end Strategy Forge workflow
    """
    
    def __init__(self):
        self.results = {}
        print("🚀 Strategy Forge - Trading Strategy Demonstration")
        print("=" * 60)
    
    def run_complete_demonstration(self, symbol='AAPL', start_date='2023-01-01', end_date='2023-12-31'):
        """
        Run complete demonstration for a given stock symbol
        
        Args:
            symbol: Stock symbol to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
        """
        print(f"\n📈 Running Strategy Analysis for {symbol}")
        print(f"📅 Period: {start_date} to {end_date}")
        print("-" * 60)
        
        try:
            # Step 1: Process data through pipeline
            processed_data = self._process_data(symbol, start_date, end_date)
            
            if processed_data is None or processed_data.empty:
                print(f"❌ Failed to process data for {symbol}. Trying with sample data...")
                processed_data = self._create_sample_data()
                symbol = "SAMPLE_DATA"
            
            # Step 2: Run all strategy types
            strategies_results = self._run_all_strategies(processed_data, symbol)
            
            # Step 3: Analyze and compare results
            self._analyze_strategy_performance(strategies_results, symbol)
            
            # Step 4: Display comprehensive summary
            self._display_comprehensive_summary(strategies_results, symbol)
            
            return strategies_results
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            print(f"❌ Demonstration failed: {e}")
            return None
    
    def _process_data(self, symbol, start_date, end_date):
        """Process data through DataProcessingPipeline"""
        print(f"\n🔄 Step 1: Processing {symbol} data through pipeline...")
        
        try:
            # Note: Adjusting for the correct PipelineConfig parameters
            pipeline_config = PipelineConfig(
                stock_symbol=symbol,  # Use stock_symbol instead of symbol
                start_date=start_date,
                end_date=end_date,
                reporting_lag_days=45,
                ratios_to_calculate=['pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'roa']
            )
            
            pipeline = DataProcessingPipeline(pipeline_config)
            result = pipeline.process_stock()
            
            if result and 'processed_data' in result:
                data = result['processed_data']
                print(f"✅ Successfully processed {len(data)} data points")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                return data
            else:
                print("❌ Pipeline processing failed")
                return None
                
        except Exception as e:
            print(f"⚠️ Pipeline error: {e}")
            return None
    
    def _create_sample_data(self):
        """Create realistic sample data for demonstration"""
        print("🔄 Creating sample data for demonstration...")
        
        # Create 252 trading days
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        dates = dates[dates.weekday < 5][:252]  # Remove weekends, limit to 252 days
        
        # Generate realistic price movement
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns ~20% annual vol
        prices = 150 * np.exp(np.cumsum(returns))  # Starting at $150
        
        # Generate realistic P/E ratios with some correlation to price movement
        base_pe = 22
        pe_noise = np.random.normal(0, 2, len(dates))
        price_effect = (prices / prices[0] - 1) * -5  # P/E tends to fall as price rises
        pe_ratios = base_pe + pe_noise + price_effect
        pe_ratios = np.clip(pe_ratios, 8, 40)  # Keep in reasonable range
        
        data = pd.DataFrame({
            'open': prices * 0.998,
            'high': prices * 1.015,
            'low': prices * 0.985,
            'close': prices,
            'volume': np.random.randint(50000000, 150000000, len(dates)),
            'pe_ratio': pe_ratios,
            'pb_ratio': pe_ratios * 0.3 + np.random.normal(0, 0.2, len(dates)),
            'debt_to_equity': np.random.normal(0.4, 0.1, len(dates)),
            'roe': np.random.normal(0.15, 0.03, len(dates)),
            'roa': np.random.normal(0.08, 0.02, len(dates))
        }, index=dates)
        
        print(f"✅ Generated sample data: {len(data)} points")
        return data
    
    def _run_all_strategies(self, data, symbol):
        """Run all strategy types on the processed data"""
        print(f"\n📊 Step 2: Running all strategy types on {symbol} data...")
        
        strategies_configs = {
            'PE_Conservative': StrategyConfig(
                name="Conservative P/E Strategy",
                description="Buy when P/E < 18, sell when P/E > 28",
                parameters={
                    'buy_pe_threshold': 18.0,
                    'sell_pe_threshold': 28.0,
                    'min_pe': 5.0,
                    'max_pe': 45.0
                }
            ),
            
            'PE_Aggressive': StrategyConfig(
                name="Aggressive P/E Strategy", 
                description="Buy when P/E < 15, sell when P/E > 25",
                parameters={
                    'buy_pe_threshold': 15.0,
                    'sell_pe_threshold': 25.0,
                    'min_pe': 5.0,
                    'max_pe': 45.0
                }
            ),
            
            'MA_Classic': StrategyConfig(
                name="Classic 20/50 SMA",
                description="20-day vs 50-day Simple Moving Average crossover",
                parameters={
                    'short_window': 20,
                    'long_window': 50,
                    'ma_type': 'SMA'
                }
            ),
            
            'MA_Fast': StrategyConfig(
                name="Fast 10/20 EMA",
                description="10-day vs 20-day Exponential Moving Average crossover",
                parameters={
                    'short_window': 10,
                    'long_window': 20,
                    'ma_type': 'EMA'
                }
            ),
            
            'Combined_Balanced': StrategyConfig(
                name="Balanced Multi-Factor",
                description="50/50 blend of P/E and MA signals",
                parameters={
                    'weight_fundamental': 0.5,
                    'weight_technical': 0.5,
                    'min_signal_strength': 0.25,
                    'buy_pe_threshold': 20.0,
                    'sell_pe_threshold': 30.0,
                    'short_window': 15,
                    'long_window': 35
                }
            ),
            
            'Combined_FundFocus': StrategyConfig(
                name="Fundamental-Focused Multi-Factor",
                description="70/30 blend favoring fundamental analysis",
                parameters={
                    'weight_fundamental': 0.7,
                    'weight_technical': 0.3,
                    'min_signal_strength': 0.2,
                    'buy_pe_threshold': 18.0,
                    'sell_pe_threshold': 28.0,
                    'short_window': 20,
                    'long_window': 50
                }
            )
        }
        
        results = {}
        
        for strategy_name, config in strategies_configs.items():
            try:
                print(f"   Running {strategy_name}...")
                
                # Determine strategy type
                if strategy_name.startswith('PE_'):
                    strategy = PEThresholdStrategy(config)
                elif strategy_name.startswith('MA_'):
                    strategy = MovingAverageStrategy(config)
                elif strategy_name.startswith('Combined_'):
                    strategy = CombinedStrategy(config)
                
                # Generate signals
                signals = strategy.generate_signals(data)
                positions = strategy.calculate_positions(signals)
                
                results[strategy_name] = {
                    'strategy': strategy,
                    'signals': signals,
                    'positions': positions,
                    'config': config,
                    'metrics': strategy.performance_metrics
                }
                
                print(f"      ✅ {len(signals)} signals generated")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[strategy_name] = None
        
        return results
    
    def _analyze_strategy_performance(self, strategies_results, symbol):
        """Analyze and compare strategy performance"""
        print(f"\n🔍 Step 3: Analyzing strategy performance for {symbol}...")
        
        analysis = {}
        
        for strategy_name, result in strategies_results.items():
            if result is None:
                continue
                
            signals = result['signals']
            metrics = result['metrics']
            
            # Count signal types
            buy_signals = [s for s in signals if s.signal == 'BUY']
            sell_signals = [s for s in signals if s.signal == 'SELL']
            
            # Calculate signal frequency
            total_days = len(signals) if signals else 0
            signal_frequency = len([s for s in signals if s.signal in ['BUY', 'SELL']]) / max(total_days, 1)
            
            # Calculate average signal strength
            action_signals = [s for s in signals if s.signal in ['BUY', 'SELL']]
            avg_strength = np.mean([s.strength for s in action_signals]) if action_signals else 0
            
            analysis[strategy_name] = {
                'total_signals': len(signals),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'signal_frequency': signal_frequency,
                'avg_signal_strength': avg_strength,
                'first_signal': signals[0].date if signals else None,
                'last_signal': signals[-1].date if signals else None
            }
        
        # Display analysis
        print("\n📈 Strategy Performance Analysis:")
        print("-" * 60)
        
        for strategy_name, analysis_data in analysis.items():
            print(f"\n{strategy_name}:")
            print(f"   Total Signals: {analysis_data['total_signals']}")
            print(f"   Buy Signals: {analysis_data['buy_signals']}")
            print(f"   Sell Signals: {analysis_data['sell_signals']}")
            print(f"   Signal Frequency: {analysis_data['signal_frequency']:.2%}")
            print(f"   Avg Signal Strength: {analysis_data['avg_signal_strength']:.3f}")
        
        self.results[symbol] = analysis
        return analysis
    
    def _display_comprehensive_summary(self, strategies_results, symbol):
        """Display comprehensive summary and insights"""
        print(f"\n📋 Step 4: Comprehensive Summary for {symbol}")
        print("=" * 60)
        
        # Strategy comparison table
        print("\n🏆 Strategy Comparison:")
        print("-" * 60)
        
        comparison_data = []
        for strategy_name, result in strategies_results.items():
            if result is None:
                continue
                
            signals = result['signals']
            buy_signals = len([s for s in signals if s.signal == 'BUY'])
            sell_signals = len([s for s in signals if s.signal == 'SELL'])
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Signals': len(signals),
                'Buy': buy_signals,
                'Sell': sell_signals,
                'Activity': f"{((buy_signals + sell_signals) / max(len(signals), 1)):.1%}"
            })
        
        # Display as formatted table
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
        
        # Key insights
        print(f"\n💡 Key Insights:")
        print("-" * 30)
        
        if comparison_data:
            # Most active strategy
            most_active = max(comparison_data, key=lambda x: x['Buy'] + x['Sell'])
            print(f"• Most Active Strategy: {most_active['Strategy']}")
            
            # Most conservative strategy  
            least_active = min(comparison_data, key=lambda x: x['Buy'] + x['Sell'])
            print(f"• Most Conservative Strategy: {least_active['Strategy']}")
            
            # Average signals per strategy
            avg_signals = np.mean([x['Total Signals'] for x in comparison_data])
            print(f"• Average Signals per Strategy: {avg_signals:.1f}")
        
        # Sample signals from best performing strategy
        if strategies_results:
            sample_strategy = list(strategies_results.keys())[0]
            sample_result = strategies_results[sample_strategy]
            
            if sample_result and sample_result['signals']:
                print(f"\n🔍 Sample Signals from {sample_strategy}:")
                print("-" * 50)
                
                sample_signals = sample_result['signals'][:5]  # First 5 signals
                for i, signal in enumerate(sample_signals, 1):
                    print(f"{i}. {signal.date}: {signal.signal} at ${signal.price:.2f}")
                    print(f"   Strength: {signal.strength:.3f}, Rationale: {signal.rationale}")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        print("-" * 30)
        
        if len([r for r in strategies_results.values() if r]) >= 3:
            print("• Multiple strategies successfully generated signals")
            print("• Consider combining fundamental and technical approaches")
            print("• Backtest strategies with historical performance analysis")
            print("• Monitor signal frequency vs. transaction costs")
        else:
            print("• Limited strategy success - review data quality")
            print("• Consider adjusting strategy parameters")
            print("• Verify data pipeline completeness")
        
        print(f"\n✅ Strategy analysis complete for {symbol}!")
    
    def run_multiple_stocks_demo(self, symbols=['AAPL', 'MSFT', 'GOOGL']):
        """Demonstrate strategy analysis across multiple stocks"""
        print(f"\n🌐 Multi-Stock Strategy Analysis")
        print("=" * 60)
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n🔄 Processing {symbol}...")
            try:
                results = self.run_complete_demonstration(
                    symbol=symbol,
                    start_date='2023-01-01',
                    end_date='2023-06-30'  # Shorter period for demo
                )
                all_results[symbol] = results
            except Exception as e:
                print(f"❌ Failed to process {symbol}: {e}")
                all_results[symbol] = None
        
        # Cross-stock analysis
        self._cross_stock_analysis(all_results)
        
        return all_results
    
    def _cross_stock_analysis(self, all_results):
        """Analyze strategy performance across multiple stocks"""
        print(f"\n📊 Cross-Stock Strategy Analysis")
        print("=" * 60)
        
        # Aggregate statistics
        strategy_performance = {}
        
        for stock, results in all_results.items():
            if results is None:
                continue
                
            for strategy_name, result in results.items():
                if result is None:
                    continue
                    
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = []
                
                signals = result['signals']
                signal_count = len([s for s in signals if s.signal in ['BUY', 'SELL']])
                strategy_performance[strategy_name].append(signal_count)
        
        print("Strategy Performance Across Stocks:")
        print("-" * 40)
        
        for strategy_name, signal_counts in strategy_performance.items():
            if signal_counts:
                avg_signals = np.mean(signal_counts)
                std_signals = np.std(signal_counts)
                print(f"{strategy_name}:")
                print(f"   Avg Signals: {avg_signals:.1f} ± {std_signals:.1f}")
                print(f"   Consistency: {(1 - std_signals/max(avg_signals, 1)):.2%}")


def main():
    """Main demonstration function"""
    demonstrator = StrategyDemonstrator()
    
    # Run single stock demonstration
    print("Starting single-stock demonstration...")
    single_result = demonstrator.run_complete_demonstration(
        symbol='AAPL',
        start_date='2023-01-01', 
        end_date='2023-12-31'
    )
    
    # Uncomment for multi-stock demonstration (takes longer)
    # print("\n" + "="*60)
    # print("Starting multi-stock demonstration...")
    # multi_result = demonstrator.run_multiple_stocks_demo(['AAPL', 'MSFT'])
    
    print(f"\n🎉 Strategy Forge demonstration complete!")
    print("Ready for Task 16: Backtester Implementation")


if __name__ == "__main__":
    main()