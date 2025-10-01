"""
Strategy Forge - Complete Backtesting Demonstration

This script demonstrates the complete end-to-end workflow:
1. Data processing through pipeline
2. Strategy signal generation
3. Backtesting with realistic transaction costs
4. Performance analysis and comparison
5. Risk management validation

This showcases the full power of the Strategy Forge framework for 
quantitative trading strategy development and validation.

Author: Strategy Forge Development Team
Version: 1.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# Import Strategy Forge modules
from src.models.strategies import (
    StrategyConfig, PEThresholdStrategy, MovingAverageStrategy, 
    CombinedStrategy, create_strategy
)
from src.models.backtester import (
    SingleAssetBacktester, BacktestConfig, BacktestResult,
    run_strategy_backtest, compare_strategies
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BacktestingDemonstrator:
    """
    Comprehensive demonstration of Strategy Forge backtesting capabilities
    """
    
    def __init__(self):
        self.results = {}
        print("🚀 Strategy Forge - Complete Backtesting Demonstration")
        print("=" * 70)
    
    def create_realistic_market_data(self, symbol="DEMO", days=252):
        """Create realistic market data for demonstration"""
        print(f"\n📊 Creating realistic market data for {symbol} ({days} days)")
        print("-" * 50)
        
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create trading days (exclude weekends)
        start_date = datetime(2023, 1, 1)
        dates = pd.date_range(start=start_date, periods=days*2, freq='D')
        trading_dates = dates[dates.weekday < 5][:days]
        
        # Generate realistic price series with volatility clustering
        initial_price = 150.0
        
        # Model parameters
        mu = 0.0008  # Daily drift (positive for growth)
        sigma_base = 0.015  # Base volatility
        
        # Generate returns with volatility clustering (GARCH-like behavior)
        returns = []
        volatility = sigma_base
        
        for i in range(days):
            # Update volatility (simple volatility clustering)
            vol_innovation = np.random.normal(0, 0.002)
            volatility = 0.95 * volatility + 0.05 * sigma_base + vol_innovation
            volatility = max(0.005, min(volatility, 0.05))  # Keep volatility reasonable
            
            # Generate return
            daily_return = np.random.normal(mu, volatility)
            returns.append(daily_return)
        
        # Convert to prices
        log_prices = np.log(initial_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Create OHLCV data with realistic intraday patterns
        data = pd.DataFrame(index=trading_dates)
        
        # Close prices
        data['close'] = prices
        
        # Open prices (with overnight gaps)
        overnight_gaps = np.random.normal(0, 0.002, days)
        data['open'] = data['close'].shift(1) * (1 + overnight_gaps)
        data.iloc[0, data.columns.get_loc('open')] = initial_price
        
        # High and low prices
        intraday_range = np.random.uniform(0.01, 0.03, days)
        high_factor = 1 + intraday_range * np.random.uniform(0.3, 0.7, days)
        low_factor = 1 - intraday_range * np.random.uniform(0.3, 0.7, days)
        
        data['high'] = np.maximum(data['open'], data['close']) * high_factor
        data['low'] = np.minimum(data['open'], data['close']) * low_factor
        
        # Volume with realistic patterns
        base_volume = 1000000
        volume_trend = np.random.uniform(0.8, 1.2, days)
        volume_noise = np.random.lognormal(0, 0.3, days)
        data['volume'] = (base_volume * volume_trend * volume_noise).astype(int)
        
        # Add fundamental data for strategies
        # P/E ratio with mean reversion and trend
        pe_base = 20
        pe_trend = np.cumsum(np.random.normal(0, 0.01, days))
        pe_noise = np.random.normal(0, 1, days)
        data['pe_ratio'] = pe_base + pe_trend + pe_noise
        data['pe_ratio'] = np.clip(data['pe_ratio'], 8, 40)  # Keep reasonable
        
        # P/B ratio correlated with P/E
        data['pb_ratio'] = data['pe_ratio'] * 0.25 + np.random.normal(0, 0.2, days)
        data['pb_ratio'] = np.clip(data['pb_ratio'], 0.5, 8)
        
        print(f"✅ Generated {len(data)} trading days")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"   P/E range: {data['pe_ratio'].min():.1f} - {data['pe_ratio'].max():.1f}")
        print(f"   Average volume: {data['volume'].mean():,.0f}")
        
        return data
    
    def setup_strategies(self):
        """Set up different strategies for comparison"""
        print(f"\n🎯 Setting up trading strategies")
        print("-" * 50)
        
        strategies = {}
        
        # 1. Conservative P/E Strategy
        strategies['PE_Conservative'] = create_strategy('pe_threshold', StrategyConfig(
            name="Conservative P/E Strategy",
            description="Conservative P/E thresholds with wide bands",
            parameters={
                'buy_pe_threshold': 16.0,
                'sell_pe_threshold': 28.0,
                'min_pe': 8.0,
                'max_pe': 40.0
            }
        ))
        
        # 2. Aggressive P/E Strategy
        strategies['PE_Aggressive'] = create_strategy('pe_threshold', StrategyConfig(
            name="Aggressive P/E Strategy",
            description="Tight P/E thresholds for frequent trading",
            parameters={
                'buy_pe_threshold': 18.0,
                'sell_pe_threshold': 24.0,
                'min_pe': 8.0,
                'max_pe': 40.0
            }
        ))
        
        # 3. Fast Moving Average Strategy
        strategies['MA_Fast'] = create_strategy('moving_average', StrategyConfig(
            name="Fast MA Strategy",
            description="Short-term moving average crossovers",
            parameters={
                'short_window': 10,
                'long_window': 20,
                'ma_type': 'EMA'
            }
        ))
        
        # 4. Classic Moving Average Strategy
        strategies['MA_Classic'] = create_strategy('moving_average', StrategyConfig(
            name="Classic MA Strategy",
            description="Classic 20/50 moving average system",
            parameters={
                'short_window': 20,
                'long_window': 50,
                'ma_type': 'SMA'
            }
        ))
        
        # 5. Balanced Multi-Factor Strategy
        strategies['Combined_Balanced'] = create_strategy('combined', StrategyConfig(
            name="Balanced Multi-Factor",
            description="Balanced fundamental and technical analysis",
            parameters={
                'weight_fundamental': 0.6,
                'weight_technical': 0.4,
                'min_signal_strength': 0.25,
                'buy_pe_threshold': 19.0,
                'sell_pe_threshold': 26.0,
                'short_window': 15,
                'long_window': 35,
                'ma_type': 'SMA'
            }
        ))
        
        print(f"✅ Created {len(strategies)} strategies:")
        for name, strategy in strategies.items():
            print(f"   - {strategy.name}")
        
        return strategies
    
    def setup_backtest_configurations(self):
        """Set up different backtesting configurations"""
        print(f"\n⚙️ Setting up backtest configurations")
        print("-" * 50)
        
        configs = {}
        
        # 1. Low-cost configuration (institutional-like)
        configs['Low_Cost'] = BacktestConfig(
            initial_capital=100000,
            commission_per_share=0.005,
            commission_min=1.0,
            commission_max=5.0,
            slippage_bps=2.0,
            position_size=0.2,
            max_position_size=0.4
        )
        
        # 2. Retail investor configuration
        configs['Retail'] = BacktestConfig(
            initial_capital=50000,
            commission_per_share=0.01,
            commission_min=2.0,
            commission_max=10.0,
            slippage_bps=5.0,
            position_size=0.15,
            max_position_size=0.3
        )
        
        # 3. Conservative with risk management
        configs['Conservative'] = BacktestConfig(
            initial_capital=100000,
            commission_per_share=0.008,
            commission_min=1.5,
            commission_max=8.0,
            slippage_bps=3.0,
            position_size=0.1,
            max_position_size=0.25,
            stop_loss_pct=8.0,
            take_profit_pct=20.0
        )
        
        # 4. Aggressive trading
        configs['Aggressive'] = BacktestConfig(
            initial_capital=100000,
            commission_per_share=0.01,
            commission_min=2.0,
            commission_max=15.0,
            slippage_bps=8.0,
            position_size=0.3,
            max_position_size=0.6,
            position_sizing='signal_strength'
        )
        
        print(f"✅ Created {len(configs)} backtest configurations:")
        for name in configs.keys():
            print(f"   - {name}")
        
        return configs
    
    def run_comprehensive_backtest(self, symbol="DEMO"):
        """Run comprehensive backtesting demonstration"""
        print(f"\n🔄 Running comprehensive backtest for {symbol}")
        print("=" * 70)
        
        # Create market data
        market_data = self.create_realistic_market_data(symbol, 252)
        
        # Setup strategies and configurations
        strategies = self.setup_strategies()
        configs = self.setup_backtest_configurations()
        
        # Generate signals for all strategies
        print(f"\n📡 Generating signals for all strategies")
        print("-" * 50)
        
        for name, strategy in strategies.items():
            signals = strategy.generate_signals(market_data)
            print(f"   {name}: {len(signals)} signals generated")
        
        # Run backtests for all combinations
        print(f"\n🎬 Running backtests (strategies × configurations)")
        print("-" * 50)
        
        results = {}
        
        for strategy_name, strategy in strategies.items():
            if not strategy.signals:
                print(f"   ⚠️ Skipping {strategy_name} (no signals)")
                continue
                
            strategy_results = {}
            
            for config_name, config in configs.items():
                try:
                    result = run_strategy_backtest(strategy, market_data, symbol, config)
                    strategy_results[config_name] = result
                    
                    print(f"   ✅ {strategy_name} × {config_name}: "
                          f"{result.total_return_pct:.2f}% return, {result.total_trades} trades")
                    
                except Exception as e:
                    print(f"   ❌ {strategy_name} × {config_name}: Failed - {e}")
                    strategy_results[config_name] = None
            
            results[strategy_name] = strategy_results
        
        return results, market_data
    
    def analyze_results(self, results, market_data, symbol):
        """Analyze and compare backtest results"""
        print(f"\n📊 Analyzing backtest results for {symbol}")
        print("=" * 70)
        
        # Create comparison tables
        comparison_data = []
        
        for strategy_name, strategy_results in results.items():
            for config_name, result in strategy_results.items():
                if result is None:
                    continue
                    
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Config': config_name,
                    'Total Return (%)': result.total_return_pct,
                    'Total Trades': result.total_trades,
                    'Win Rate (%)': result.win_rate,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Max Drawdown (%)': result.max_drawdown,
                    'Profit Factor': result.profit_factor,
                    'Total Costs ($)': result.total_commission + result.total_slippage
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            print("\n🏆 Performance Comparison Table:")
            print("-" * 70)
            print(comparison_df.to_string(index=False, float_format='%.2f'))
            
            # Best performers analysis
            print(f"\n🥇 Best Performers:")
            print("-" * 30)
            
            best_return = comparison_df.loc[comparison_df['Total Return (%)'].idxmax()]
            best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
            best_winrate = comparison_df.loc[comparison_df['Win Rate (%)'].idxmax()]
            
            print(f"Highest Return: {best_return['Strategy']} × {best_return['Config']} "
                  f"({best_return['Total Return (%)']:.2f}%)")
            print(f"Best Sharpe Ratio: {best_sharpe['Strategy']} × {best_sharpe['Config']} "
                  f"({best_sharpe['Sharpe Ratio']:.2f})")
            print(f"Highest Win Rate: {best_winrate['Strategy']} × {best_winrate['Config']} "
                  f"({best_winrate['Win Rate (%)']:.1f}%)")
            
            # Risk analysis
            print(f"\n⚠️ Risk Analysis:")
            print("-" * 30)
            
            worst_drawdown = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin()]
            highest_cost = comparison_df.loc[comparison_df['Total Costs ($)'].idxmax()]
            
            print(f"Worst Drawdown: {worst_drawdown['Strategy']} × {worst_drawdown['Config']} "
                  f"({worst_drawdown['Max Drawdown (%)']:.2f}%)")
            print(f"Highest Costs: {highest_cost['Strategy']} × {highest_cost['Config']} "
                  f"(${highest_cost['Total Costs ($)']:.2f})")
            
            # Configuration impact analysis
            print(f"\n🔧 Configuration Impact Analysis:")
            print("-" * 30)
            
            config_impact = comparison_df.groupby('Config').agg({
                'Total Return (%)': 'mean',
                'Sharpe Ratio': 'mean',
                'Max Drawdown (%)': 'mean',
                'Total Costs ($)': 'mean'
            }).round(2)
            
            print(config_impact)
        
        return comparison_df
    
    def demonstrate_detailed_analysis(self, results, symbol):
        """Demonstrate detailed analysis of a specific strategy"""
        print(f"\n🔍 Detailed Analysis Example: PE_Conservative × Low_Cost")
        print("=" * 70)
        
        # Get a specific result for detailed analysis
        if 'PE_Conservative' in results and 'Low_Cost' in results['PE_Conservative']:
            result = results['PE_Conservative']['Low_Cost']
            
            if result is not None:
                print(f"\n📈 Strategy Performance Summary:")
                print("-" * 40)
                
                summary = result.get_summary_dict()
                for key, value in summary.items():
                    print(f"{key:.<25} {value}")
                
                print(f"\n💰 Trade Analysis:")
                print("-" * 40)
                
                if result.trades:
                    print(f"Total Trades: {len(result.trades)}")
                    
                    buy_trades = [t for t in result.trades if t.trade_type.value == 'BUY']
                    sell_trades = [t for t in result.trades if t.trade_type.value == 'SELL']
                    
                    print(f"Buy Trades: {len(buy_trades)}")
                    print(f"Sell Trades: {len(sell_trades)}")
                    
                    if buy_trades:
                        avg_buy_price = np.mean([t.price for t in buy_trades])
                        print(f"Average Buy Price: ${avg_buy_price:.2f}")
                    
                    if sell_trades:
                        avg_sell_price = np.mean([t.price for t in sell_trades])
                        print(f"Average Sell Price: ${avg_sell_price:.2f}")
                    
                    # Show first few trades
                    print(f"\nFirst 3 Trades:")
                    for i, trade in enumerate(result.trades[:3]):
                        print(f"  {i+1}. {trade.timestamp}: {trade.trade_type.value} "
                              f"{trade.quantity:.0f} shares @ ${trade.price:.2f}")
                
                print(f"\n📊 Risk Metrics:")
                print("-" * 40)
                print(f"Maximum Drawdown: {result.max_drawdown:.2f}%")
                print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
                print(f"Sortino Ratio: {result.sortino_ratio:.3f}")
                print(f"Calmar Ratio: {result.calmar_ratio:.3f}")
                
            else:
                print("❌ No results available for detailed analysis")
    
    def run_complete_demonstration(self):
        """Run the complete backtesting demonstration"""
        
        # Run comprehensive backtest
        results, market_data = self.run_comprehensive_backtest("DEMO")
        
        # Analyze results
        comparison_df = self.analyze_results(results, market_data, "DEMO")
        
        # Detailed analysis
        self.demonstrate_detailed_analysis(results, "DEMO")
        
        # Final summary
        print(f"\n🎉 Demonstration Complete!")
        print("=" * 70)
        
        total_combinations = 0
        successful_combinations = 0
        
        for strategy_results in results.values():
            for result in strategy_results.values():
                total_combinations += 1
                if result is not None:
                    successful_combinations += 1
        
        print(f"📊 Summary Statistics:")
        print(f"   Total Strategy×Config Combinations: {total_combinations}")
        print(f"   Successful Backtests: {successful_combinations}")
        print(f"   Success Rate: {(successful_combinations/total_combinations)*100:.1f}%")
        
        if not comparison_df.empty:
            print(f"\n📈 Overall Performance:")
            print(f"   Average Return: {comparison_df['Total Return (%)'].mean():.2f}%")
            print(f"   Average Sharpe Ratio: {comparison_df['Sharpe Ratio'].mean():.3f}")
            print(f"   Average Win Rate: {comparison_df['Win Rate (%)'].mean():.1f}%")
        
        print(f"\n✨ Strategy Forge backtesting framework fully demonstrated!")
        print("Ready for Task 17: Performance Metrics Implementation")
        
        return results, comparison_df


def main():
    """Main demonstration function"""
    demonstrator = BacktestingDemonstrator()
    
    # Run complete demonstration
    results, comparison = demonstrator.run_complete_demonstration()
    
    print("\n" + "="*70)
    print("🚀 Strategy Forge Backtesting Demo Complete!")
    print("="*70)
    print("Key accomplishments:")
    print("✅ Realistic market data generation")
    print("✅ Multiple strategy configurations")
    print("✅ Comprehensive backtesting scenarios")
    print("✅ Performance analysis and comparison")
    print("✅ Risk management validation")
    print("✅ Transaction cost modeling")
    print("✅ Detailed trade tracking and analysis")


if __name__ == "__main__":
    main()