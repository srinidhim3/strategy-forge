"""
Single-Asset Backtester for Strategy Forge

This module implements a comprehensive backtesting engine for single-stock trading strategies.
It simulates realistic trading conditions including transaction costs, slippage, and position
management to provide accurate performance measurement.

Key Components:
- Trade: Individual transaction record
- Position: Current holdings and exposure tracking
- Portfolio: Overall portfolio state and cash management
- SingleAssetBacktester: Core backtesting engine
- BacktestResult: Comprehensive results container

Features:
- Realistic transaction cost modeling
- Position sizing and risk management
- Detailed trade and performance tracking
- Integration with Strategy Forge signal framework
- Comprehensive performance metrics calculation

Author: Strategy Forge Development Team
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
import warnings
from enum import Enum

# Import our modules
from .strategies import TradingSignal, BaseStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class TradeType(Enum):
    """Enumeration for trade types"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Enumeration for order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class Trade:
    """
    Represents a single trade transaction
    """
    timestamp: str
    trade_type: TradeType
    symbol: str
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    order_type: OrderType = OrderType.MARKET
    signal_strength: float = 0.0
    rationale: str = ""
    
    @property
    def gross_amount(self) -> float:
        """Gross trade amount before costs"""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> float:
        """Net trade amount after costs"""
        if self.trade_type == TradeType.BUY:
            return -(self.gross_amount + self.commission)
        else:  # SELL
            return self.gross_amount - self.commission
    
    @property
    def total_cost(self) -> float:
        """Total transaction costs"""
        return self.commission + (self.quantity * self.slippage)


@dataclass
class Position:
    """
    Represents current position in a security
    """
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    total_cost: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.last_price
    
    @property
    def is_long(self) -> bool:
        """True if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """True if position is short"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """True if no position"""
        return abs(self.quantity) < 1e-6
    
    def update_price(self, new_price: float):
        """Update last price and unrealized P&L"""
        self.last_price = new_price
        if not self.is_flat:
            self.unrealized_pnl = (new_price - self.avg_cost) * self.quantity
    
    def add_trade(self, trade: Trade):
        """Add a trade to this position"""
        old_quantity = self.quantity
        
        if trade.trade_type == TradeType.BUY:
            new_quantity = self.quantity + trade.quantity
            if self.quantity >= 0:  # Adding to long or starting long
                self.avg_cost = ((self.avg_cost * self.quantity) + 
                               (trade.price * trade.quantity)) / new_quantity
            else:  # Covering short
                if new_quantity >= 0:  # Partial or full cover
                    cover_quantity = min(abs(self.quantity), trade.quantity)
                    self.realized_pnl += (self.avg_cost - trade.price) * cover_quantity
                    if new_quantity > 0:  # Still have long position
                        remaining_quantity = trade.quantity - cover_quantity
                        self.avg_cost = trade.price
                
        else:  # SELL
            new_quantity = self.quantity - trade.quantity
            if self.quantity <= 0:  # Adding to short or starting short
                self.avg_cost = ((abs(self.avg_cost * self.quantity)) + 
                               (trade.price * trade.quantity)) / abs(new_quantity)
            else:  # Selling long
                if new_quantity <= 0:  # Partial or full sell
                    sell_quantity = min(self.quantity, trade.quantity)
                    self.realized_pnl += (trade.price - self.avg_cost) * sell_quantity
                    if new_quantity < 0:  # Now short
                        remaining_quantity = trade.quantity - sell_quantity
                        self.avg_cost = trade.price
        
        self.quantity = new_quantity
        self.total_cost += trade.commission


@dataclass
class Portfolio:
    """
    Represents overall portfolio state
    """
    initial_capital: float
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital
    
    @property
    def total_market_value(self) -> float:
        """Total market value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_portfolio_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + self.total_market_value
    
    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L across all positions"""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_return(self) -> float:
        """Total return as percentage of initial capital"""
        return ((self.total_portfolio_value - self.initial_capital) / 
                self.initial_capital) * 100
    
    def get_position(self, symbol: str) -> Position:
        """Get position for symbol, create if doesn't exist"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade and update portfolio
        
        Returns:
            bool: True if trade was executed successfully
        """
        # Check if we have enough cash for buy orders
        if trade.trade_type == TradeType.BUY:
            required_cash = trade.gross_amount + trade.commission
            if required_cash > self.cash:
                logger.warning(f"Insufficient cash for trade: need ${required_cash:.2f}, have ${self.cash:.2f}")
                return False
        
        # Execute the trade
        position = self.get_position(trade.symbol)
        position.add_trade(trade)
        
        # Update cash
        self.cash -= trade.net_amount
        
        # Update portfolio costs
        self.total_commission += trade.commission
        self.total_slippage += trade.quantity * trade.slippage
        
        return True


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_capital: float = 100000.0
    commission_per_share: float = 0.01
    commission_min: float = 1.0
    commission_max: float = 10.0
    slippage_bps: float = 5.0  # Basis points
    position_sizing: str = "fixed_percentage"  # "fixed_percentage", "fixed_dollar", "signal_strength"
    position_size: float = 0.1  # 10% of portfolio for fixed_percentage
    max_position_size: float = 0.5  # Maximum 50% in any position
    min_position_size: float = 0.01  # Minimum 1% position
    enable_short_selling: bool = False
    margin_requirement: float = 0.5  # 50% margin for short selling
    stop_loss_pct: Optional[float] = None  # Stop loss percentage
    take_profit_pct: Optional[float] = None  # Take profit percentage
    
    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade"""
        commission = quantity * self.commission_per_share
        return max(self.commission_min, min(commission, self.commission_max))
    
    def calculate_slippage(self, price: float) -> float:
        """Calculate slippage for a trade"""
        return price * (self.slippage_bps / 10000.0)


@dataclass
class BacktestResult:
    """
    Comprehensive backtesting results container
    """
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_portfolio_value: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_commission: float
    total_slippage: float
    max_drawdown: float
    max_drawdown_duration: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    position_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics as dictionary"""
        return {
            'Symbol': self.symbol,
            'Strategy': self.strategy_name,
            'Period': f"{self.start_date} to {self.end_date}",
            'Initial Capital': f"${self.initial_capital:,.2f}",
            'Final Value': f"${self.final_portfolio_value:,.2f}",
            'Total Return': f"{self.total_return_pct:.2f}%",
            'Total Trades': self.total_trades,
            'Win Rate': f"{self.win_rate:.1f}%",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Max Drawdown': f"{self.max_drawdown:.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Total Costs': f"${self.total_commission + self.total_slippage:,.2f}"
        }


class SingleAssetBacktester:
    """
    Comprehensive single-asset backtesting engine
    
    This backtester simulates realistic trading of a single stock based on strategy signals.
    It includes transaction costs, slippage, position management, and comprehensive
    performance measurement capabilities.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize the backtester with configuration"""
        self.config = config or BacktestConfig()
        self.portfolio = None
        self.trades = []
        self.equity_curve = []
        self.position_history = []
        self.daily_returns = []
        
        logger.info(f"Initialized SingleAssetBacktester with ${self.config.initial_capital:,.0f} capital")
    
    def backtest(
        self,
        signals: List[TradingSignal],
        price_data: pd.DataFrame,
        symbol: str,
        strategy_name: str = "Unknown Strategy"
    ) -> BacktestResult:
        """
        Run a comprehensive backtest on the provided signals and price data
        
        Args:
            signals: List of trading signals from a strategy
            price_data: DataFrame with OHLCV price data
            symbol: Stock symbol being traded
            strategy_name: Name of the strategy for reporting
            
        Returns:
            BacktestResult: Comprehensive results object
        """
        logger.info(f"Starting backtest for {symbol} using {strategy_name}")
        logger.info(f"Signals: {len(signals)}, Price data: {len(price_data)} days")
        
        # Initialize portfolio
        self.portfolio = Portfolio(initial_capital=self.config.initial_capital)
        self.trades = []
        self.equity_curve = []
        self.position_history = []
        
        # Convert signals to dictionary for faster lookup
        signal_dict = {signal.date: signal for signal in signals}
        
        # Process each day
        for date_idx, (date, row) in enumerate(price_data.iterrows()):
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            
            # Update portfolio with current prices
            current_price = row['close']
            self.portfolio.update_prices({symbol: current_price})
            
            # Check for signals on this date
            if date_str in signal_dict:
                signal = signal_dict[date_str]
                self._process_signal(signal, symbol, current_price, date_str)
            
            # Check for stop loss or take profit
            self._check_risk_management(symbol, current_price, date_str)
            
            # Record daily portfolio state
            self._record_daily_state(date_str, current_price, symbol)
            
            # Log progress periodically
            if date_idx % 50 == 0:
                logger.debug(f"Processed day {date_idx}: {date_str}, Portfolio: ${self.portfolio.total_portfolio_value:,.2f}")
        
        # Calculate final results
        result = self._calculate_results(symbol, strategy_name, price_data)
        
        logger.info(f"Backtest complete: {result.total_return_pct:.2f}% return, {result.total_trades} trades")
        
        return result
    
    def _process_signal(self, signal: TradingSignal, symbol: str, price: float, date: str):
        """Process a trading signal and execute trades if appropriate"""
        current_position = self.portfolio.get_position(symbol)
        
        if signal.signal == 'BUY':
            # Calculate position size
            target_value = self._calculate_position_size(signal.strength, 'BUY')
            current_value = current_position.market_value
            
            if target_value > current_value:
                # Need to buy more
                additional_value = target_value - current_value
                quantity = additional_value / price
                
                if quantity > 0:
                    self._execute_trade(
                        symbol=symbol,
                        trade_type=TradeType.BUY,
                        quantity=quantity,
                        price=price,
                        date=date,
                        signal=signal
                    )
        
        elif signal.signal == 'SELL':
            # Close position if we have one
            if not current_position.is_flat:
                self._execute_trade(
                    symbol=symbol,
                    trade_type=TradeType.SELL,
                    quantity=abs(current_position.quantity),
                    price=price,
                    date=date,
                    signal=signal
                )
            
            # For short selling (if enabled)
            if self.config.enable_short_selling:
                target_value = self._calculate_position_size(signal.strength, 'SELL')
                if target_value > 0:
                    quantity = target_value / price
                    self._execute_trade(
                        symbol=symbol,
                        trade_type=TradeType.SELL,
                        quantity=quantity,
                        price=price,
                        date=date,
                        signal=signal
                    )
    
    def _calculate_position_size(self, signal_strength: float, signal_type: str) -> float:
        """Calculate position size based on configuration and signal strength"""
        if self.config.position_sizing == "fixed_percentage":
            base_size = self.config.position_size
        elif self.config.position_sizing == "fixed_dollar":
            base_size = self.config.position_size / self.portfolio.total_portfolio_value
        elif self.config.position_sizing == "signal_strength":
            base_size = self.config.position_size * signal_strength
        else:
            base_size = self.config.position_size
        
        # Apply limits
        base_size = max(self.config.min_position_size, 
                       min(base_size, self.config.max_position_size))
        
        return base_size * self.portfolio.total_portfolio_value
    
    def _execute_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: float,
        price: float,
        date: str,
        signal: TradingSignal
    ):
        """Execute a trade with realistic costs and constraints"""
        # Calculate costs
        commission = self.config.calculate_commission(quantity, price)
        slippage = self.config.calculate_slippage(price)
        
        # Adjust price for slippage
        if trade_type == TradeType.BUY:
            execution_price = price + slippage
        else:
            execution_price = price - slippage
        
        # Create trade
        trade = Trade(
            timestamp=date,
            trade_type=trade_type,
            symbol=symbol,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            signal_strength=signal.strength,
            rationale=signal.rationale
        )
        
        # Execute trade
        if self.portfolio.execute_trade(trade):
            self.trades.append(trade)
            logger.debug(f"Executed {trade_type.value}: {quantity:.2f} shares at ${execution_price:.2f}")
        else:
            logger.warning(f"Failed to execute trade: {trade_type.value} {quantity:.2f} shares")
    
    def _check_risk_management(self, symbol: str, current_price: float, date: str):
        """Check and execute risk management rules like stop loss and take profit"""
        position = self.portfolio.get_position(symbol)
        
        if position.is_flat:
            return
        
        # Stop loss check
        if self.config.stop_loss_pct:
            if position.is_long:
                stop_price = position.avg_cost * (1 - self.config.stop_loss_pct / 100)
                if current_price <= stop_price:
                    self._execute_risk_trade(symbol, position, current_price, date, "Stop Loss")
            elif position.is_short:
                stop_price = position.avg_cost * (1 + self.config.stop_loss_pct / 100)
                if current_price >= stop_price:
                    self._execute_risk_trade(symbol, position, current_price, date, "Stop Loss")
        
        # Take profit check
        if self.config.take_profit_pct:
            if position.is_long:
                profit_price = position.avg_cost * (1 + self.config.take_profit_pct / 100)
                if current_price >= profit_price:
                    self._execute_risk_trade(symbol, position, current_price, date, "Take Profit")
            elif position.is_short:
                profit_price = position.avg_cost * (1 - self.config.take_profit_pct / 100)
                if current_price <= profit_price:
                    self._execute_risk_trade(symbol, position, current_price, date, "Take Profit")
    
    def _execute_risk_trade(self, symbol: str, position: Position, price: float, date: str, reason: str):
        """Execute a risk management trade"""
        trade_type = TradeType.SELL if position.is_long else TradeType.BUY
        
        # Create dummy signal for risk management
        risk_signal = TradingSignal(
            date=date,
            signal='SELL' if position.is_long else 'BUY',
            strength=1.0,
            price=price,
            rationale=reason
        )
        
        self._execute_trade(
            symbol=symbol,
            trade_type=trade_type,
            quantity=abs(position.quantity),
            price=price,
            date=date,
            signal=risk_signal
        )
    
    def _record_daily_state(self, date: str, price: float, symbol: str):
        """Record daily portfolio state for analysis"""
        position = self.portfolio.get_position(symbol)
        
        equity_value = self.portfolio.total_portfolio_value
        self.equity_curve.append({
            'date': date,
            'portfolio_value': equity_value,
            'cash': self.portfolio.cash,
            'position_value': position.market_value,
            'position_quantity': position.quantity,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl
        })
    
    def _calculate_results(self, symbol: str, strategy_name: str, price_data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        # Calculate daily returns
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        # Calculate drawdown
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate drawdown duration
        drawdown_duration = self._calculate_max_drawdown_duration(equity_df)
        
        # Trade statistics
        winning_trades = len([t for t in self.trades if self._trade_pnl(t, symbol) > 0])
        losing_trades = len([t for t in self.trades if self._trade_pnl(t, symbol) < 0])
        win_rate = (winning_trades / max(len(self.trades), 1)) * 100
        
        # Calculate average wins and losses
        trade_pnls = [self._trade_pnl(t, symbol) for t in self.trades]
        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        calmar_ratio = self._calculate_calmar_ratio(daily_returns, max_drawdown)
        
        # Create result object
        result = BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=price_data.index[0].strftime('%Y-%m-%d'),
            end_date=price_data.index[-1].strftime('%Y-%m-%d'),
            initial_capital=self.config.initial_capital,
            final_portfolio_value=self.portfolio.total_portfolio_value,
            total_return_pct=self.portfolio.total_return,
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_commission=self.portfolio.total_commission,
            total_slippage=self.portfolio.total_slippage,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades.copy(),
            daily_returns=daily_returns,
            equity_curve=equity_df['portfolio_value'],
            position_history=equity_df
        )
        
        return result
    
    def _trade_pnl(self, trade: Trade, symbol: str) -> float:
        """Calculate P&L for a trade (simplified)"""
        # This is a simplified calculation - in reality would need to match buys/sells
        position = self.portfolio.get_position(symbol)
        if trade.trade_type == TradeType.SELL:
            return (trade.price - position.avg_cost) * trade.quantity
        return 0  # Buy trades don't have immediate P&L
    
    def _calculate_max_drawdown_duration(self, equity_df: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        drawdowns = equity_df['drawdown'] < 0
        if not drawdowns.any():
            return 0
        
        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = 0
        
        for is_drawdown in drawdowns:
            if is_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0.0
        
        annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        return (annual_return - risk_free_rate) / annual_volatility
    
    def _calculate_sortino_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(daily_returns) == 0:
            return 0.0
        
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        annual_return = daily_returns.mean() * 252
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        return (annual_return - risk_free_rate) / downside_deviation
    
    def _calculate_calmar_ratio(self, daily_returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0 or len(daily_returns) == 0:
            return 0.0
        
        annual_return = daily_returns.mean() * 252
        return annual_return / abs(max_drawdown / 100)


# Utility functions for backtesting
def run_strategy_backtest(
    strategy: BaseStrategy,
    price_data: pd.DataFrame,
    symbol: str,
    config: Optional[BacktestConfig] = None
) -> BacktestResult:
    """
    Convenience function to run a complete backtest for a strategy
    
    Args:
        strategy: Strategy instance with generated signals
        price_data: Price data DataFrame
        symbol: Stock symbol
        config: Backtesting configuration
        
    Returns:
        BacktestResult: Complete backtest results
    """
    backtester = SingleAssetBacktester(config)
    return backtester.backtest(strategy.signals, price_data, symbol, strategy.name)


def compare_strategies(
    strategies: List[BaseStrategy],
    price_data: pd.DataFrame,
    symbol: str,
    config: Optional[BacktestConfig] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies using backtesting
    
    Args:
        strategies: List of strategy instances
        price_data: Price data DataFrame
        symbol: Stock symbol
        config: Backtesting configuration
        
    Returns:
        DataFrame: Comparison results
    """
    results = []
    
    for strategy in strategies:
        if not strategy.signals:
            logger.warning(f"Strategy {strategy.name} has no signals, skipping")
            continue
            
        result = run_strategy_backtest(strategy, price_data, symbol, config)
        results.append(result.get_summary_dict())
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Strategy Forge - Single Asset Backtester")
    print("=======================================")
    print("Comprehensive backtesting engine ready for integration!")
    print("\nKey Features:")
    print("- Realistic transaction cost modeling")
    print("- Position management and risk controls")
    print("- Comprehensive performance metrics")
    print("- Integration with Strategy Forge signals")
    print("\nReady for testing and demonstration!")