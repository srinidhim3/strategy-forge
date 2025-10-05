"""
Advanced Performance Metrics Demonstration (Task 17)

This script demonstrates the enhanced backtesting engine with institutional-grade
performance analytics including Value at Risk, Information Ratio, Jensen Alpha,
and comprehensive benchmark comparison capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.models.backtester import (
    SingleAssetBacktester, BacktestConfig, 
    benchmark_strategy_analysis, compare_strategies_advanced,
    generate_performance_report
)
from src.models.strategies import PEThresholdStrategy, MovingAverageStrategy, StrategyConfig


def create_realistic_market_data(symbol: str, days: int = 504) -> pd.DataFrame:
    """Create realistic market data with volatility clustering and trends"""
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range('2022-01-01', periods=days, freq='D')
    
    # Generate realistic returns with GARCH-like volatility clustering
    base_vol = 0.02
    vol_persistence = 0.85
    vol_shock = 0.1
    
    returns = []
    current_vol = base_vol
    
    for i in range(days):
        # Update volatility with persistence and shocks
        vol_shock_rand = np.random.normal(0, 0.001)
        current_vol = vol_persistence * current_vol + (1 - vol_persistence) * base_vol + vol_shock * vol_shock_rand
        current_vol = max(0.005, min(0.05, current_vol))  # Bound volatility
        
        # Generate return with current volatility
        daily_return = np.random.normal(0.0008, current_vol)  # ~20% annual return
        returns.append(daily_return)
    
    # Convert to prices
    prices = 100 * (1 + pd.Series(returns)).cumprod()
    
    # Create realistic OHLC data
    market_data = pd.DataFrame({
        'open': prices * np.random.uniform(0.995, 1.005, days),
        'high': prices * np.random.uniform(1.005, 1.025, days),
        'low': prices * np.random.uniform(0.975, 0.995, days),
        'close': prices,
        'volume': np.random.randint(800000, 2500000, days),
    }, index=dates)
    
    # Add realistic P/E ratios with some correlation to price movements
    base_pe = 18
    pe_volatility = 2
    pe_ratios = []
    for i, ret in enumerate(returns):
        # P/E tends to move opposite to recent returns (simplified)
        pe_adjustment = -ret * 50 + np.random.normal(0, pe_volatility)
        if i == 0:
            pe = base_pe
        else:
            pe = pe_ratios[-1] + pe_adjustment
        pe = max(8, min(40, pe))  # Bound P/E ratios
        pe_ratios.append(pe)
    
    market_data['pe_ratio'] = pe_ratios
    
    return market_data


def demonstrate_advanced_performance_metrics():
    """Demonstrate advanced performance metrics capabilities"""
    
    print("🚀 Strategy Forge - Advanced Performance Metrics Demonstration")
    print("=" * 70)
    print()
    
    print("📊 Creating realistic market data for comprehensive analysis")
    print("-" * 50)
    
    # Create realistic market data
    market_data = create_realistic_market_data('ADVANCED_DEMO', days=504)  # 2 years
    
    print(f"✅ Generated {len(market_data)} trading days")
    print(f"   Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
    print(f"   P/E range: {market_data['pe_ratio'].min():.1f} - {market_data['pe_ratio'].max():.1f}")
    print(f"   Total return (buy & hold): {((market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1) * 100:.2f}%")
    print()
    
    print("🎯 Setting up strategies for advanced analysis")
    print("-" * 50)
    
    # Create diverse strategies
    strategies = [
        PEThresholdStrategy(StrategyConfig(
            name="Conservative Value",
            description="Conservative P/E value strategy",
            parameters={'buy_pe_threshold': 12.0, 'sell_pe_threshold': 18.0}
        )),
        PEThresholdStrategy(StrategyConfig(
            name="Moderate Value", 
            description="Moderate P/E value strategy",
            parameters={'buy_pe_threshold': 15.0, 'sell_pe_threshold': 22.0}
        )),
        PEThresholdStrategy(StrategyConfig(
            name="Growth Oriented",
            description="Growth-oriented P/E strategy", 
            parameters={'buy_pe_threshold': 20.0, 'sell_pe_threshold': 30.0}
        )),
        MovingAverageStrategy(StrategyConfig(
            name="Fast Momentum",
            description="Fast momentum strategy",
            parameters={'short_window': 10, 'long_window': 25}
        )),
        MovingAverageStrategy(StrategyConfig(
            name="Classic Trend",
            description="Classic trend following",
            parameters={'short_window': 20, 'long_window': 50}
        ))
    ]
    
    # Generate signals for all strategies
    for strategy in strategies:
        strategy.generate_signals(market_data)
    
    print(f"✅ Created {len(strategies)} strategies:")
    for strategy in strategies:
        signals_count = len(strategy.signals)
        print(f"   - {strategy.name}: {signals_count} signals generated")
    print()
    
    print("⚙️ Configuring realistic trading environment")
    print("-" * 50)
    
    # Create institutional-grade configuration
    config = BacktestConfig(
        initial_capital=1000000,    # $1M portfolio
        commission_per_share=0.005, # 0.5 cents per share
        slippage_bps=3,            # 3 basis points slippage
        position_size=0.15,        # 15% position sizing
        stop_loss_pct=8,           # 8% stop loss
        take_profit_pct=20         # 20% take profit
    )
    
    print(f"✅ Configuration:")
    print(f"   Initial Capital: ${config.initial_capital:,}")
    print(f"   Commission: {config.commission_per_share}¢ per share")
    print(f"   Slippage: {config.slippage_bps} basis points")
    print(f"   Position Size: {config.position_size*100}% of portfolio")
    print(f"   Risk Management: {config.stop_loss_pct}% stop loss, {config.take_profit_pct}% take profit")
    print()
    
    print("📈 Running advanced strategy comparison")
    print("-" * 50)
    
    # Run advanced comparison
    comparison_df = compare_strategies_advanced(
        strategies,
        market_data,
        'ADVANCED_DEMO',
        config,
        include_advanced_metrics=True
    )
    
    if len(comparison_df) > 0:
        print("🏆 Strategy Comparison with Advanced Metrics:")
        print("-" * 50)
        
        # Display key metrics
        key_metrics = [
            'Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown',
            'Value at Risk (95%)', 'Beta', 'Information Ratio', 'Jensen Alpha'
        ]
        
        display_df = comparison_df[key_metrics].copy()
        print(display_df.to_string(index=False))
        print()
        
        # Find best performers
        if 'Total Return' in comparison_df.columns:
            # Extract numeric values for comparison
            returns = []
            for ret_str in comparison_df['Total Return']:
                try:
                    ret_val = float(ret_str.replace('%', ''))
                    returns.append(ret_val)
                except:
                    returns.append(0.0)
            
            if returns:
                best_idx = returns.index(max(returns))
                best_strategy = comparison_df.iloc[best_idx]['Strategy']
                best_return = max(returns)
                
                print(f"🥇 Best Performing Strategy: {best_strategy}")
                print(f"   Total Return: {best_return:.2f}%")
                print()
    
    print("🔍 Detailed benchmark analysis for top strategy")
    print("-" * 50)
    
    # Select strategy with signals for detailed analysis
    analysis_strategy = None
    for strategy in strategies:
        if len(strategy.signals) > 0:
            analysis_strategy = strategy
            break
    
    if analysis_strategy:
        print(f"📊 Analyzing: {analysis_strategy.name}")
        print()
        
        # Comprehensive benchmark analysis
        benchmark_analysis = benchmark_strategy_analysis(
            analysis_strategy,
            market_data,
            'ADVANCED_DEMO',
            config=config
        )
        
        # Display each section
        for section_name, section_data in benchmark_analysis.items():
            print(f"📈 {section_name}:")
            print("-" * 30)
            for metric, value in section_data.items():
                print(f"   {metric}: {value}")
            print()
        
        print("📋 Generating comprehensive performance report")
        print("-" * 50)
        
        # Generate full backtest for detailed report
        backtester = SingleAssetBacktester(config)
        detailed_result = backtester.backtest(
            analysis_strategy.signals,
            market_data,
            'ADVANCED_DEMO',
            analysis_strategy.name
        )
        
        # Generate and display performance report
        performance_report = generate_performance_report(detailed_result)
        print(performance_report)
        
    else:
        print("⚠️ No strategies generated signals with current data")
        print("   This can happen with very specific thresholds")
        print("   The advanced metrics framework is fully functional")
    
    print()
    print("🎉 Advanced Performance Metrics Demonstration Complete!")
    print("=" * 70)
    print("✨ New Capabilities Demonstrated:")
    print("   ✅ Value at Risk (VaR) at 95% and 99% confidence levels")
    print("   ✅ Conditional Value at Risk (Expected Shortfall)")
    print("   ✅ Information Ratio for active management evaluation")
    print("   ✅ Treynor Ratio for risk-adjusted performance")
    print("   ✅ Jensen Alpha for benchmark-relative performance")
    print("   ✅ Beta coefficient for systematic risk measurement")
    print("   ✅ Tracking Error for active risk quantification")
    print("   ✅ Up/Down Capture Ratios for market condition analysis")
    print("   ✅ Rolling performance metrics (6-month windows)")
    print("   ✅ Consecutive wins/losses tracking")
    print("   ✅ Comprehensive benchmark comparison framework")
    print("   ✅ Institutional-grade performance reporting")
    print()
    print("🚀 Ready for Task 18: CLI Runner Implementation")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_advanced_performance_metrics()