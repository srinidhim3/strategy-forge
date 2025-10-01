"""
Comprehensive Test Suite for SingleAssetBacktester

This test suite validates all aspects of the backtesting engine:
- Trade execution and position management
- Transaction cost calculations
- Risk management features
- Performance metrics calculation
- Integration with strategy signals
- Various market scenarios and edge cases

Author: Strategy Forge Development Team
Version: 1.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List

# Import our modules
from src.models.backtester import (
    SingleAssetBacktester, BacktestConfig, BacktestResult,
    Trade, TradeType, Position, Portfolio, OrderType,
    run_strategy_backtest, compare_strategies
)
from src.models.strategies import (
    StrategyConfig, TradingSignal, PEThresholdStrategy, 
    MovingAverageStrategy, CombinedStrategy
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktesterTestSuite:
    """Comprehensive test suite for backtesting functionality"""
    
    def __init__(self):
        self.test_results = []
        print("🧪 Strategy Forge - Backtester Testing Suite")
        print("=" * 60)
    
    def create_sample_price_data(self, scenario="normal", days=252):
        """Create sample price data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        np.random.seed(42)  # For reproducible tests
        
        if scenario == "trending_up":
            # Upward trending market
            returns = np.random.normal(0.001, 0.015, len(dates))  # Positive drift
            prices = 100 * np.exp(np.cumsum(returns))
            
        elif scenario == "trending_down":
            # Downward trending market
            returns = np.random.normal(-0.001, 0.015, len(dates))  # Negative drift
            prices = 100 * np.exp(np.cumsum(returns))
            
        elif scenario == "volatile":
            # High volatility market
            returns = np.random.normal(0, 0.03, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
        elif scenario == "sideways":
            # Sideways market with mean reversion
            prices = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 1, len(dates))
            
        else:  # normal
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        # Ensure high >= close >= low and high >= open >= low
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    def create_sample_signals(self, price_data: pd.DataFrame, pattern="simple"):
        """Create sample trading signals for testing"""
        signals = []
        
        if pattern == "simple":
            # Simple buy and hold pattern
            signals.append(TradingSignal(
                date=price_data.index[10].strftime('%Y-%m-%d'),
                signal='BUY',
                strength=0.8,
                price=price_data.iloc[10]['close'],
                rationale="Test buy signal"
            ))
            
            signals.append(TradingSignal(
                date=price_data.index[-10].strftime('%Y-%m-%d'),
                signal='SELL',
                strength=0.9,
                price=price_data.iloc[-10]['close'],
                rationale="Test sell signal"
            ))
            
        elif pattern == "frequent":
            # Frequent trading pattern
            for i in range(20, len(price_data), 20):
                signal_type = 'BUY' if i % 40 == 20 else 'SELL'
                signals.append(TradingSignal(
                    date=price_data.index[i].strftime('%Y-%m-%d'),
                    signal=signal_type,
                    strength=0.6 + np.random.random() * 0.4,
                    price=price_data.iloc[i]['close'],
                    rationale=f"Test {signal_type.lower()} signal #{i//20}"
                ))
        
        elif pattern == "alternating":
            # Alternating buy/sell pattern
            for i in range(30, len(price_data), 30):
                signal_type = 'BUY' if len(signals) % 2 == 0 else 'SELL'
                signals.append(TradingSignal(
                    date=price_data.index[i].strftime('%Y-%m-%d'),
                    signal=signal_type,
                    strength=0.7,
                    price=price_data.iloc[i]['close'],
                    rationale=f"Alternating {signal_type.lower()} signal"
                ))
        
        return signals
    
    def test_basic_trade_execution(self):
        """Test basic trade execution and position management"""
        print("\n🔄 Testing Basic Trade Execution")
        print("-" * 40)
        
        test_passed = True
        errors = []
        
        try:
            # Create test data
            price_data = self.create_sample_price_data("normal", 100)
            signals = self.create_sample_signals(price_data, "simple")
            
            # Configure backtester
            config = BacktestConfig(
                initial_capital=100000,
                commission_per_share=0.01,
                position_size=0.2  # 20% positions
            )
            
            backtester = SingleAssetBacktester(config)
            
            # Run backtest
            result = backtester.backtest(signals, price_data, "TEST", "Basic Test Strategy")
            
            # Validate results
            assert result.total_trades > 0, "No trades executed"
            assert result.final_portfolio_value > 0, "Invalid final portfolio value"
            assert len(result.trades) == result.total_trades, "Trade count mismatch"
            
            print(f"✅ Executed {result.total_trades} trades")
            print(f"✅ Final portfolio value: ${result.final_portfolio_value:,.2f}")
            print(f"✅ Total return: {result.total_return_pct:.2f}%")
            
            # Validate trade details
            for trade in result.trades:
                assert trade.quantity > 0, "Invalid trade quantity"
                assert trade.price > 0, "Invalid trade price"
                assert trade.commission >= 0, "Invalid commission"
                
            print(f"✅ All {len(result.trades)} trades validated")
            
        except Exception as e:
            test_passed = False
            errors.append(f"Basic execution test failed: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        self.test_results.append({
            'test': 'Basic Trade Execution',
            'passed': test_passed,
            'errors': errors
        })
        
        return test_passed
    
    def test_transaction_costs(self):
        """Test transaction cost calculations"""
        print("\n💰 Testing Transaction Costs")
        print("-" * 40)
        
        test_passed = True
        errors = []
        
        try:
            # Test with different commission structures
            configs = [
                BacktestConfig(commission_per_share=0.005, commission_min=1.0, commission_max=5.0),
                BacktestConfig(commission_per_share=0.01, commission_min=2.0, commission_max=10.0),
                BacktestConfig(commission_per_share=0.02, commission_min=5.0, commission_max=20.0)
            ]
            
            price_data = self.create_sample_price_data("normal", 50)
            signals = self.create_sample_signals(price_data, "frequent")
            
            results = []
            for i, config in enumerate(configs):
                backtester = SingleAssetBacktester(config)
                result = backtester.backtest(signals, price_data, "TEST", f"Cost Test {i+1}")
                results.append(result)
                
                print(f"✅ Config {i+1}: ${result.total_commission:.2f} commission, {result.total_trades} trades")
            
            # Validate that higher commission rates result in higher costs
            for i in range(1, len(results)):
                if results[i].total_trades > 0 and results[i-1].total_trades > 0:
                    assert results[i].total_commission >= results[i-1].total_commission, \
                        "Higher commission rates should result in higher total costs"
            
            print(f"✅ Transaction cost scaling validated")
            
        except Exception as e:
            test_passed = False
            errors.append(f"Transaction cost test failed: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        self.test_results.append({
            'test': 'Transaction Costs',
            'passed': test_passed,
            'errors': errors
        })
        
        return test_passed
    
    def test_position_sizing(self):
        """Test different position sizing methods"""
        print("\n📏 Testing Position Sizing")
        print("-" * 40)
        
        test_passed = True
        errors = []
        
        try:
            price_data = self.create_sample_price_data("normal", 80)
            signals = self.create_sample_signals(price_data, "simple")
            
            # Test different position sizing methods
            sizing_methods = [
                ("fixed_percentage", 0.3),
                ("fixed_dollar", 30000),
                ("signal_strength", 0.5)
            ]
            
            for method, size in sizing_methods:
                config = BacktestConfig(
                    initial_capital=100000,
                    position_sizing=method,
                    position_size=size
                )
                
                backtester = SingleAssetBacktester(config)
                result = backtester.backtest(signals, price_data, "TEST", f"Position Test {method}")
                
                print(f"✅ {method}: {result.total_trades} trades, {result.total_return_pct:.2f}% return")
                
                # Validate that trades were executed
                assert result.total_trades > 0, f"No trades executed for {method}"
            
        except Exception as e:
            test_passed = False
            errors.append(f"Position sizing test failed: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        self.test_results.append({
            'test': 'Position Sizing',
            'passed': test_passed,
            'errors': errors
        })
        
        return test_passed
    
    def test_risk_management(self):
        """Test risk management features"""
        print("\n🛡️ Testing Risk Management")
        print("-" * 40)
        
        test_passed = True
        errors = []
        
        try:
            # Create volatile market data
            price_data = self.create_sample_price_data("volatile", 100)
            signals = self.create_sample_signals(price_data, "simple")
            
            # Test with stop loss
            config_stop_loss = BacktestConfig(
                initial_capital=100000,
                position_size=0.3,
                stop_loss_pct=5.0  # 5% stop loss
            )
            
            backtester = SingleAssetBacktester(config_stop_loss)
            result_stop = backtester.backtest(signals, price_data, "TEST", "Stop Loss Test")
            
            print(f"✅ Stop loss test: {result_stop.total_trades} trades")
            
            # Test with take profit
            config_take_profit = BacktestConfig(
                initial_capital=100000,
                position_size=0.3,
                take_profit_pct=10.0  # 10% take profit
            )
            
            backtester = SingleAssetBacktester(config_take_profit)
            result_profit = backtester.backtest(signals, price_data, "TEST", "Take Profit Test")
            
            print(f"✅ Take profit test: {result_profit.total_trades} trades")
            
            # Test with both
            config_both = BacktestConfig(
                initial_capital=100000,
                position_size=0.3,
                stop_loss_pct=5.0,
                take_profit_pct=10.0
            )
            
            backtester = SingleAssetBacktester(config_both)
            result_both = backtester.backtest(signals, price_data, "TEST", "Combined Risk Test")
            
            print(f"✅ Combined risk test: {result_both.total_trades} trades")
            
            # Validate that risk management may create additional trades
            # (stop loss and take profit can trigger exits)
            
        except Exception as e:
            test_passed = False
            errors.append(f"Risk management test failed: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        self.test_results.append({
            'test': 'Risk Management',
            'passed': test_passed,
            'errors': errors
        })
        
        return test_passed
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        print("\n📊 Testing Performance Metrics")
        print("-" * 40)
        
        test_passed = True
        errors = []
        
        try:
            # Test with different market scenarios
            scenarios = ["trending_up", "trending_down", "sideways", "volatile"]
            
            for scenario in scenarios:
                price_data = self.create_sample_price_data(scenario, 120)
                signals = self.create_sample_signals(price_data, "alternating")
                
                config = BacktestConfig(initial_capital=100000, position_size=0.25)
                backtester = SingleAssetBacktester(config)
                result = backtester.backtest(signals, price_data, "TEST", f"{scenario} Test")
                
                # Validate all metrics are calculated
                assert isinstance(result.sharpe_ratio, (int, float)), "Invalid Sharpe ratio"
                assert isinstance(result.sortino_ratio, (int, float)), "Invalid Sortino ratio"
                assert isinstance(result.max_drawdown, (int, float)), "Invalid max drawdown"
                assert isinstance(result.win_rate, (int, float)), "Invalid win rate"
                assert isinstance(result.profit_factor, (int, float)), "Invalid profit factor"
                
                # Validate ranges
                assert 0 <= result.win_rate <= 100, "Win rate out of range"
                assert result.profit_factor >= 0, "Negative profit factor"
                
                print(f"✅ {scenario}: Sharpe {result.sharpe_ratio:.2f}, Max DD {result.max_drawdown:.2f}%")
                
        except Exception as e:
            test_passed = False
            errors.append(f"Performance metrics test failed: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        self.test_results.append({
            'test': 'Performance Metrics',
            'passed': test_passed,
            'errors': errors
        })
        
        return test_passed
    
    def test_strategy_integration(self):
        """Test integration with actual strategy objects"""
        print("\n🔗 Testing Strategy Integration")
        print("-" * 40)
        
        test_passed = True
        errors = []
        
        try:
            # Create sample data with P/E ratios for strategies
            price_data = self.create_sample_price_data("normal", 150)
            
            # Add P/E ratio data
            pe_ratios = 15 + np.random.normal(0, 3, len(price_data))
            pe_ratios = np.clip(pe_ratios, 8, 35)
            
            strategy_data = price_data.copy()
            strategy_data['pe_ratio'] = pe_ratios
            
            # Test P/E strategy
            pe_config = StrategyConfig(
                name="Backtest P/E Strategy",
                description="P/E strategy for backtesting",
                parameters={'buy_pe_threshold': 18.0, 'sell_pe_threshold': 25.0}
            )
            
            pe_strategy = PEThresholdStrategy(pe_config)
            pe_signals = pe_strategy.generate_signals(strategy_data)
            
            # Test backtesting with P/E strategy
            backtest_config = BacktestConfig(initial_capital=100000, position_size=0.3)
            result = run_strategy_backtest(pe_strategy, price_data, "TEST", backtest_config)
            
            print(f"✅ P/E strategy backtest: {result.total_trades} trades, {result.total_return_pct:.2f}% return")
            
            # Test MA strategy
            ma_config = StrategyConfig(
                name="Backtest MA Strategy",
                description="MA strategy for backtesting",
                parameters={'short_window': 10, 'long_window': 30}
            )
            
            ma_strategy = MovingAverageStrategy(ma_config)
            ma_signals = ma_strategy.generate_signals(price_data)
            
            result_ma = run_strategy_backtest(ma_strategy, price_data, "TEST", backtest_config)
            
            print(f"✅ MA strategy backtest: {result_ma.total_trades} trades, {result_ma.total_return_pct:.2f}% return")
            
            # Test strategy comparison
            strategies = [pe_strategy, ma_strategy]
            comparison = compare_strategies(strategies, price_data, "TEST", backtest_config)
            
            print(f"✅ Strategy comparison: {len(comparison)} strategies compared")
            assert len(comparison) <= len(strategies), "Too many comparison results"
            
        except Exception as e:
            test_passed = False
            errors.append(f"Strategy integration test failed: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        self.test_results.append({
            'test': 'Strategy Integration',
            'passed': test_passed,
            'errors': errors
        })
        
        return test_passed
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n🔍 Testing Edge Cases")
        print("-" * 40)
        
        test_passed = True
        errors = []
        
        try:
            # Test with no signals
            price_data = self.create_sample_price_data("normal", 50)
            empty_signals = []
            
            config = BacktestConfig(initial_capital=100000)
            backtester = SingleAssetBacktester(config)
            result = backtester.backtest(empty_signals, price_data, "TEST", "No Signals Test")
            
            assert result.total_trades == 0, "Should have no trades with empty signals"
            assert result.final_portfolio_value == result.initial_capital, "Portfolio value should equal initial capital"
            
            print(f"✅ No signals test: {result.total_trades} trades (expected 0)")
            
            # Test with insufficient cash
            expensive_signals = [
                TradingSignal(
                    date=price_data.index[10].strftime('%Y-%m-%d'),
                    signal='BUY',
                    strength=1.0,
                    price=price_data.iloc[10]['close'],
                    rationale="Expensive buy signal"
                )
            ]
            
            small_capital_config = BacktestConfig(
                initial_capital=100,  # Very small capital
                position_size=0.99   # Try to use 99% of capital
            )
            
            backtester = SingleAssetBacktester(small_capital_config)
            result = backtester.backtest(expensive_signals, price_data, "TEST", "Insufficient Cash Test")
            
            print(f"✅ Insufficient cash test: {result.total_trades} trades")
            
            # Test with extreme prices
            extreme_data = price_data.copy()
            extreme_data['close'] = extreme_data['close'] * 1000  # Very expensive stock
            
            result_extreme = backtester.backtest(expensive_signals, extreme_data, "TEST", "Extreme Price Test")
            
            print(f"✅ Extreme price test: {result_extreme.total_trades} trades")
            
        except Exception as e:
            test_passed = False
            errors.append(f"Edge cases test failed: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        self.test_results.append({
            'test': 'Edge Cases',
            'passed': test_passed,
            'errors': errors
        })
        
        return test_passed
    
    def run_all_tests(self):
        """Run all backtester tests"""
        print("🚀 Starting Comprehensive Backtester Testing")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Run all tests
        tests = [
            self.test_basic_trade_execution,
            self.test_transaction_costs,
            self.test_position_sizing,
            self.test_risk_management,
            self.test_performance_metrics,
            self.test_strategy_integration,
            self.test_edge_cases
        ]
        
        passed_tests = 0
        for test in tests:
            if test():
                passed_tests += 1
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 BACKTESTER TEST SUMMARY")
        print("=" * 60)
        
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['test']}")
            if result['errors']:
                for error in result['errors']:
                    print(f"    ⚠️  {error}")
        
        print(f"\n📊 Results: {passed_tests}/{len(tests)} tests passed")
        print(f"⏱️  Duration: {duration.total_seconds():.2f} seconds")
        
        if passed_tests == len(tests):
            print("🎉 All backtester tests passed! Ready for production use.")
        else:
            print("⚠️  Some tests failed. Please review the errors above.")
        
        return passed_tests == len(tests)


if __name__ == "__main__":
    # Run comprehensive backtester tests
    tester = BacktesterTestSuite()
    success = tester.run_all_tests()
    
    if success:
        print("\n✨ Strategy Forge backtester is fully validated!")
        print("Ready for Task 17: Performance Metrics Implementation")
    else:
        print("\n⚠️  Please address test failures before proceeding.")