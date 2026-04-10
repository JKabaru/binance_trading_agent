"""
Backtesting engine for Binance Futures trading strategies.
Supports long/short positions with leverage.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    leverage: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    
    def close(self, exit_price: float, exit_time: datetime, fee_rate: float = 0.0004):
        """Close the trade and calculate PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        # Calculate PnL based on position side
        if self.side == 'long':
            price_change = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            price_change = (self.entry_price - exit_price) / self.entry_price
        
        # Apply leverage
        leveraged_return = price_change * self.leverage
        
        # Calculate fees (entry + exit)
        self.fees = (self.entry_price + exit_price) * self.quantity * fee_rate
        
        # Net PnL
        gross_pnl = leveraged_return * self.entry_price * self.quantity
        self.pnl = gross_pnl - self.fees
        self.pnl_pct = (self.pnl / (self.entry_price * self.quantity)) * 100
        
        return self.pnl


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trades: List[Trade]
    equity_curve: pd.Series
    config_hash: str = ""


class Backtester:
    """
    Backtesting engine for futures trading strategies.
    
    Features:
    - Long and short positions
    - Leverage support
    - Fee calculation
    - Performance metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 1,
        fee_rate: float = 0.0004,  # 0.04% per trade
        slippage: float = 0.0001,  # 0.01% slippage
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage = slippage
        
    def run(self, df: pd.DataFrame, strategy) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy object with generate_signals() method
        
        Returns:
            BacktestResult with performance metrics
        """
        # Initialize tracking variables
        capital = self.initial_capital
        position = None  # Current position (Trade object)
        trades = []
        equity = [capital]
        
        # Generate trading signals
        signals = strategy.generate_signals(df)
        
        # Iterate through bars
        for i in range(1, len(df)):
            row = df.iloc[i]
            signal = signals.iloc[i] if hasattr(signals, 'iloc') else signals[i]
            
            # Close position if exit signal
            if position is not None:
                if signal.get('action') == 'exit' or signal.get('side') != position.side:
                    # Apply slippage to exit price
                    if position.side == 'long':
                        exit_price = row['close'] * (1 - self.slippage)
                    else:
                        exit_price = row['close'] * (1 + self.slippage)
                    
                    position.close(exit_price, row.name, self.fee_rate)
                    capital += position.pnl
                    trades.append(position)
                    position = None
            
            # Open position if entry signal and no current position
            if position is None and signal.get('action') == 'entry':
                # Apply slippage to entry price
                if signal['side'] == 'long':
                    entry_price = row['close'] * (1 + self.slippage)
                else:
                    entry_price = row['close'] * (1 - self.slippage)
                
                # Calculate position size (use full capital with leverage)
                quantity = (capital * self.leverage) / entry_price
                
                position = Trade(
                    entry_time=row.name,
                    exit_time=None,
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    side=signal['side'],
                    entry_price=entry_price,
                    exit_price=None,
                    quantity=quantity,
                    leverage=self.leverage,
                )
            
            # Track equity
            if position is not None:
                # Unrealized PnL
                if position.side == 'long':
                    unrealized_pnl = (row['close'] - position.entry_price) * position.quantity
                else:
                    unrealized_pnl = (position.entry_price - row['close']) * position.quantity
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            equity.append(current_equity)
        
        # Close any open position at the end
        if position is not None:
            exit_price = df.iloc[-1]['close']
            position.close(exit_price, df.index[-1], self.fee_rate)
            capital += position.pnl
            trades.append(position)
            equity[-1] = capital
        
        # Calculate metrics
        equity_series = pd.Series(equity, index=df.index)
        result = self.calculate_metrics(trades, equity_series)
        
        return result
    
    def calculate_metrics(self, trades: List[Trade], equity: pd.Series) -> BacktestResult:
        """Calculate performance metrics from trades and equity curve."""
        
        # Total return
        total_return = ((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]) * 100
        
        # Daily returns for Sharpe ratio
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
        
        # Win/Loss statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = (num_winning / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        result = BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=num_winning,
            losing_trades=num_losing,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades=trades,
            equity_curve=equity,
        )
        
        return result
