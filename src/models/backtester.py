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
    Comprehensive backtesting results container with advanced performance metrics
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
    
    # Advanced Performance Metrics (Task 17)
    value_at_risk_95: float = 0.0
    value_at_risk_99: float = 0.0
    conditional_var_95: float = 0.0
    conditional_var_99: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    up_capture_ratio: float = 0.0
    down_capture_ratio: float = 0.0
    
    # Rolling Performance Metrics
    rolling_sharpe_6m: float = 0.0
    rolling_sortino_6m: float = 0.0
    rolling_volatility_6m: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    trades: List[Trade] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    position_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Benchmark comparison data
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_equity: pd.Series = field(default_factory=pd.Series)
    
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
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}",
            'Total Costs': f"${self.total_commission + self.total_slippage:,.2f}"
        }
    
    def get_advanced_metrics_dict(self) -> Dict[str, Any]:
        """Get advanced performance metrics as dictionary"""
        return {
            'Value at Risk (95%)': f"{self.value_at_risk_95:.2f}%",
            'Value at Risk (99%)': f"{self.value_at_risk_99:.2f}%",
            'Conditional VaR (95%)': f"{self.conditional_var_95:.2f}%",
            'Conditional VaR (99%)': f"{self.conditional_var_99:.2f}%",
            'Information Ratio': f"{self.information_ratio:.2f}",
            'Treynor Ratio': f"{self.treynor_ratio:.2f}",
            'Jensen Alpha': f"{self.jensen_alpha:.2f}%",
            'Beta': f"{self.beta:.2f}",
            'Tracking Error': f"{self.tracking_error:.2f}%",
            'Up Capture Ratio': f"{self.up_capture_ratio:.2f}%",
            'Down Capture Ratio': f"{self.down_capture_ratio:.2f}%",
            'Rolling Sharpe (6M)': f"{self.rolling_sharpe_6m:.2f}",
            'Rolling Sortino (6M)': f"{self.rolling_sortino_6m:.2f}",
            'Rolling Volatility (6M)': f"{self.rolling_volatility_6m:.2f}%",
            'Max Consecutive Wins': self.max_consecutive_wins,
            'Max Consecutive Losses': self.max_consecutive_losses
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
            current_price = row.get('Close', row.get('close', 0))  # Handle both cases
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
        
        # Advanced performance metrics (Task 17)
        var_95 = self._calculate_value_at_risk(daily_returns, 0.95)
        var_99 = self._calculate_value_at_risk(daily_returns, 0.99)
        cvar_95 = self._calculate_conditional_var(daily_returns, 0.95)
        cvar_99 = self._calculate_conditional_var(daily_returns, 0.99)
        
        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(daily_returns)
        
        # Consecutive wins/losses
        max_wins, max_losses = self._calculate_consecutive_wins_losses(self.trades)
        
        # Generate benchmark returns (simple buy-and-hold strategy)
        close_col = 'Close' if 'Close' in price_data.columns else 'close'
        benchmark_returns = price_data[close_col].pct_change().fillna(0)
        benchmark_equity = (1 + benchmark_returns).cumprod() * self.config.initial_capital
        
        # Benchmark-relative metrics
        info_ratio = self._calculate_information_ratio(daily_returns, benchmark_returns)
        treynor_ratio = self._calculate_treynor_ratio(daily_returns, benchmark_returns)
        jensen_alpha = self._calculate_jensen_alpha(daily_returns, benchmark_returns)
        beta = self._calculate_beta(daily_returns, benchmark_returns)
        tracking_error = self._calculate_tracking_error(daily_returns, benchmark_returns)
        up_capture, down_capture = self._calculate_capture_ratios(daily_returns, benchmark_returns)
        
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
            
            # Advanced metrics (Task 17)
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            conditional_var_95=cvar_95,
            conditional_var_99=cvar_99,
            information_ratio=info_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            beta=beta,
            tracking_error=tracking_error,
            up_capture_ratio=up_capture,
            down_capture_ratio=down_capture,
            rolling_sharpe_6m=rolling_metrics['rolling_sharpe'],
            rolling_sortino_6m=rolling_metrics['rolling_sortino'],
            rolling_volatility_6m=rolling_metrics['rolling_volatility'],
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            
            trades=self.trades.copy(),
            daily_returns=daily_returns,
            equity_curve=equity_df['portfolio_value'],
            position_history=equity_df,
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity
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
    
    # Advanced Performance Metrics (Task 17)
    
    def _calculate_value_at_risk(self, daily_returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level
        
        Args:
            daily_returns: Series of daily returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR as percentage
        """
        if len(daily_returns) == 0:
            return 0.0
        
        percentile = (1 - confidence_level) * 100
        var_daily = np.percentile(daily_returns, percentile)
        return var_daily * 100  # Convert to percentage
    
    def _calculate_conditional_var(self, daily_returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR
        
        Args:
            daily_returns: Series of daily returns
            confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)
            
        Returns:
            CVaR as percentage
        """
        if len(daily_returns) == 0:
            return 0.0
        
        percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(daily_returns, percentile)
        tail_losses = daily_returns[daily_returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        cvar = tail_losses.mean()
        return cvar * 100  # Convert to percentage
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio (active return / tracking error)
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            Information ratio
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align the series by date
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_portfolio) == 0:
            return 0.0
        
        active_returns = aligned_portfolio - aligned_benchmark
        active_return_annual = active_returns.mean() * 252
        tracking_error_annual = active_returns.std() * np.sqrt(252)
        
        if tracking_error_annual == 0:
            return 0.0
        
        return active_return_annual / tracking_error_annual
    
    def _calculate_treynor_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                                 risk_free_rate: float = 0.02) -> float:
        """
        Calculate Treynor Ratio (excess return / beta)
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Treynor ratio
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        if beta == 0:
            return 0.0
        
        annual_return = portfolio_returns.mean() * 252
        excess_return = annual_return - risk_free_rate
        
        return excess_return / beta
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to benchmark
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            Beta coefficient
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align the series by date
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_portfolio) < 2:
            return 0.0
        
        covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance
    
    def _calculate_jensen_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                                risk_free_rate: float = 0.02) -> float:
        """
        Calculate Jensen's Alpha (portfolio return - expected return based on CAPM)
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Alpha as annual percentage
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        portfolio_annual_return = portfolio_returns.mean() * 252
        benchmark_annual_return = benchmark_returns.mean() * 252
        
        expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        alpha = portfolio_annual_return - expected_return
        
        return alpha * 100  # Convert to percentage
    
    def _calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error (standard deviation of active returns)
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            Tracking error as annual percentage
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align the series by date
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_portfolio) == 0:
            return 0.0
        
        active_returns = aligned_portfolio - aligned_benchmark
        tracking_error_annual = active_returns.std() * np.sqrt(252)
        
        return tracking_error_annual * 100  # Convert to percentage
    
    def _calculate_capture_ratios(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> tuple:
        """
        Calculate up and down capture ratios
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            Tuple of (up_capture_ratio, down_capture_ratio) as percentages
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0, 0.0
        
        # Align the series by date
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_portfolio) == 0:
            return 0.0, 0.0
        
        # Separate up and down periods based on benchmark
        up_periods = aligned_benchmark > 0
        down_periods = aligned_benchmark < 0
        
        up_portfolio = aligned_portfolio[up_periods]
        up_benchmark = aligned_benchmark[up_periods]
        down_portfolio = aligned_portfolio[down_periods]
        down_benchmark = aligned_benchmark[down_periods]
        
        # Calculate capture ratios
        up_capture = 0.0
        down_capture = 0.0
        
        if len(up_portfolio) > 0 and up_benchmark.mean() != 0:
            up_capture = (up_portfolio.mean() / up_benchmark.mean()) * 100
        
        if len(down_portfolio) > 0 and down_benchmark.mean() != 0:
            down_capture = (down_portfolio.mean() / down_benchmark.mean()) * 100
        
        return up_capture, down_capture
    
    def _calculate_rolling_metrics(self, daily_returns: pd.Series, window_days: int = 126) -> dict:
        """
        Calculate rolling performance metrics (default 6 months = 126 trading days)
        
        Args:
            daily_returns: Series of daily returns
            window_days: Rolling window size in days
            
        Returns:
            Dictionary with rolling metrics
        """
        if len(daily_returns) < window_days:
            return {
                'rolling_sharpe': 0.0,
                'rolling_sortino': 0.0,
                'rolling_volatility': 0.0
            }
        
        # Calculate rolling Sharpe ratio
        rolling_returns = daily_returns.rolling(window_days).mean() * 252
        rolling_volatility = daily_returns.rolling(window_days).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns - 0.02) / rolling_volatility  # Assuming 2% risk-free rate
        
        # Calculate rolling Sortino ratio
        rolling_downside_std = daily_returns.rolling(window_days).apply(
            lambda x: x[x < 0].std() * np.sqrt(252) if len(x[x < 0]) > 0 else 0
        )
        rolling_sortino = (rolling_returns - 0.02) / rolling_downside_std
        
        # Use the most recent rolling values
        latest_sharpe = rolling_sharpe.iloc[-1] if not rolling_sharpe.isna().iloc[-1] else 0.0
        latest_sortino = rolling_sortino.iloc[-1] if not rolling_sortino.isna().iloc[-1] else 0.0
        latest_volatility = rolling_volatility.iloc[-1] if not rolling_volatility.isna().iloc[-1] else 0.0
        
        return {
            'rolling_sharpe': latest_sharpe,
            'rolling_sortino': latest_sortino,
            'rolling_volatility': latest_volatility * 100  # Convert to percentage
        }
    
    def _calculate_consecutive_wins_losses(self, trades: List[Trade]) -> tuple:
        """
        Calculate maximum consecutive wins and losses
        
        Args:
            trades: List of executed trades
            
        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if not trades:
            return 0, 0
        
        # For simplicity, we'll consider a trade profitable if it's a sell at higher price
        # than the previous buy, or a buy at lower price than the previous sell
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        buy_prices = []
        sell_prices = []
        
        for trade in trades:
            if trade.trade_type == TradeType.BUY:
                buy_prices.append(trade.price)
            elif trade.trade_type == TradeType.SELL:
                sell_prices.append(trade.price)
                
                # Check if we have corresponding buy price to compare
                if buy_prices:
                    buy_price = buy_prices[-1]  # Use most recent buy price
                    if trade.price > buy_price:  # Profitable sell
                        current_wins += 1
                        current_losses = 0
                        max_wins = max(max_wins, current_wins)
                    else:  # Loss on sell
                        current_losses += 1
                        current_wins = 0
                        max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses


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


def compare_strategies_advanced(
    strategies: List[BaseStrategy],
    price_data: pd.DataFrame,
    symbol: str,
    config: Optional[BacktestConfig] = None,
    include_advanced_metrics: bool = True
) -> pd.DataFrame:
    """
    Compare multiple strategies with advanced performance metrics
    
    Args:
        strategies: List of strategy instances
        price_data: Price data DataFrame
        symbol: Stock symbol
        config: Backtesting configuration
        include_advanced_metrics: Whether to include advanced metrics
        
    Returns:
        DataFrame: Detailed comparison results with advanced metrics
    """
    results = []
    
    for strategy in strategies:
        if not strategy.signals:
            logger.warning(f"Strategy {strategy.name} has no signals, skipping")
            continue
            
        result = run_strategy_backtest(strategy, price_data, symbol, config)
        
        # Combine basic and advanced metrics
        basic_metrics = result.get_summary_dict()
        if include_advanced_metrics:
            advanced_metrics = result.get_advanced_metrics_dict()
            combined_metrics = {**basic_metrics, **advanced_metrics}
        else:
            combined_metrics = basic_metrics
            
        results.append(combined_metrics)
    
    return pd.DataFrame(results)


def benchmark_strategy_analysis(
    strategy: BaseStrategy,
    price_data: pd.DataFrame,
    symbol: str,
    benchmark_data: Optional[pd.DataFrame] = None,
    config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Comprehensive strategy analysis against benchmark
    
    Args:
        strategy: Strategy instance
        price_data: Price data DataFrame
        symbol: Stock symbol
        benchmark_data: Benchmark price data (if None, uses buy-and-hold of same asset)
        config: Backtesting configuration
        
    Returns:
        Dictionary with comprehensive analysis
    """
    if not strategy.signals:
        raise ValueError(f"Strategy {strategy.name} has no signals")
    
    # Run strategy backtest
    strategy_result = run_strategy_backtest(strategy, price_data, symbol, config)
    
    # Calculate benchmark performance (buy-and-hold if no benchmark data provided)
    if benchmark_data is None:
        close_col = 'Close' if 'Close' in price_data.columns else 'close'
        benchmark_returns = price_data[close_col].pct_change().fillna(0)
        benchmark_total_return = ((price_data[close_col].iloc[-1] / price_data[close_col].iloc[0]) - 1) * 100
    else:
        close_col = 'Close' if 'Close' in benchmark_data.columns else 'close'
        benchmark_returns = benchmark_data[close_col].pct_change().fillna(0)
        benchmark_total_return = ((benchmark_data[close_col].iloc[-1] / benchmark_data[close_col].iloc[0]) - 1) * 100
    
    # Performance comparison
    analysis = {
        'Strategy Performance': {
            'Strategy Name': strategy_result.strategy_name,
            'Total Return': f"{strategy_result.total_return_pct:.2f}%",
            'Sharpe Ratio': f"{strategy_result.sharpe_ratio:.2f}",
            'Max Drawdown': f"{strategy_result.max_drawdown:.2f}%",
            'Win Rate': f"{strategy_result.win_rate:.1f}%",
            'Total Trades': strategy_result.total_trades,
        },
        
        'Benchmark Performance': {
            'Benchmark': 'Buy & Hold' if benchmark_data is None else 'Custom Benchmark',
            'Total Return': f"{benchmark_total_return:.2f}%",
            'Volatility': f"{benchmark_returns.std() * np.sqrt(252) * 100:.2f}%",
            'Sharpe Ratio': f"{(benchmark_returns.mean() * 252 - 0.02) / (benchmark_returns.std() * np.sqrt(252)):.2f}",
        },
        
        'Relative Performance': {
            'Excess Return': f"{strategy_result.total_return_pct - benchmark_total_return:.2f}%",
            'Information Ratio': f"{strategy_result.information_ratio:.2f}",
            'Tracking Error': f"{strategy_result.tracking_error:.2f}%",
            'Beta': f"{strategy_result.beta:.2f}",
            'Jensen Alpha': f"{strategy_result.jensen_alpha:.2f}%",
            'Up Capture Ratio': f"{strategy_result.up_capture_ratio:.2f}%",
            'Down Capture Ratio': f"{strategy_result.down_capture_ratio:.2f}%",
        },
        
        'Risk Analysis': {
            'Value at Risk (95%)': f"{strategy_result.value_at_risk_95:.2f}%",
            'Conditional VaR (95%)': f"{strategy_result.conditional_var_95:.2f}%",
            'Treynor Ratio': f"{strategy_result.treynor_ratio:.2f}",
            'Calmar Ratio': f"{strategy_result.calmar_ratio:.2f}",
            'Rolling Sharpe (6M)': f"{strategy_result.rolling_sharpe_6m:.2f}",
        }
    }
    
    return analysis


def generate_performance_report(result: BacktestResult) -> str:
    """
    Generate a comprehensive performance report
    
    Args:
        result: BacktestResult instance
        
    Returns:
        Formatted string report
    """
    report = f"""
🎯 STRATEGY PERFORMANCE REPORT
{'=' * 50}

📊 BASIC METRICS
{'-' * 20}
Strategy: {result.strategy_name}
Symbol: {result.symbol}
Period: {result.start_date} to {result.end_date}
Initial Capital: ${result.initial_capital:,.2f}
Final Value: ${result.final_portfolio_value:,.2f}
Total Return: {result.total_return_pct:.2f}%

📈 RETURN METRICS
{'-' * 20}
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Win Rate: {result.win_rate:.1f}%
Average Win: {result.avg_win:.2f}%
Average Loss: {result.avg_loss:.2f}%
Profit Factor: {result.profit_factor:.2f}

🎯 RISK METRICS
{'-' * 20}
Maximum Drawdown: {result.max_drawdown:.2f}%
Drawdown Duration: {result.max_drawdown_duration} days
Sharpe Ratio: {result.sharpe_ratio:.2f}
Sortino Ratio: {result.sortino_ratio:.2f}
Calmar Ratio: {result.calmar_ratio:.2f}

📊 ADVANCED RISK ANALYSIS
{'-' * 20}
Value at Risk (95%): {result.value_at_risk_95:.2f}%
Value at Risk (99%): {result.value_at_risk_99:.2f}%
Conditional VaR (95%): {result.conditional_var_95:.2f}%
Conditional VaR (99%): {result.conditional_var_99:.2f}%

🔄 BENCHMARK COMPARISON
{'-' * 20}
Information Ratio: {result.information_ratio:.2f}
Treynor Ratio: {result.treynor_ratio:.2f}
Jensen Alpha: {result.jensen_alpha:.2f}%
Beta: {result.beta:.2f}
Tracking Error: {result.tracking_error:.2f}%
Up Capture Ratio: {result.up_capture_ratio:.2f}%
Down Capture Ratio: {result.down_capture_ratio:.2f}%

📈 ROLLING PERFORMANCE (6 Months)
{'-' * 20}
Rolling Sharpe: {result.rolling_sharpe_6m:.2f}
Rolling Sortino: {result.rolling_sortino_6m:.2f}
Rolling Volatility: {result.rolling_volatility_6m:.2f}%

🎲 TRADE ANALYSIS
{'-' * 20}
Max Consecutive Wins: {result.max_consecutive_wins}
Max Consecutive Losses: {result.max_consecutive_losses}
Total Commission: ${result.total_commission:.2f}
Total Slippage: ${result.total_slippage:.2f}
Total Costs: ${result.total_commission + result.total_slippage:.2f}

{'=' * 50}
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report


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