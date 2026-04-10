"""
Momentum-based trading strategies for Binance Futures.
Includes RSI, MACD, and Moving Average crossover strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) momentum strategy.
    
    Entry signals:
    - Long: RSI crosses above oversold threshold (e.g., 30)
    - Short: RSI crosses below overbought threshold (e.g., 70)
    
    Exit signals:
    - RSI reaches neutral zone (e.g., 50)
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
    ):
        super().__init__(name="rsi_momentum")
        self.config = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            'exit_level': exit_level,
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        
        rsi_period = self.config['rsi_period']
        oversold = self.config['oversold']
        overbought = self.config['overbought']
        exit_level = self.config['exit_level']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Initialize signals
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['side'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            current_rsi = rsi.iloc[i]
            prev_rsi = rsi.iloc[i-1]
            
            if pd.isna(current_rsi) or pd.isna(prev_rsi):
                continue
            
            # Long entry: RSI crosses above oversold
            if prev_rsi < oversold and current_rsi >= oversold:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'long'
            
            # Short entry: RSI crosses below overbought
            elif prev_rsi > overbought and current_rsi <= overbought:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'short'
            
            # Exit long: RSI reaches exit level from below
            elif prev_rsi < exit_level and current_rsi >= exit_level:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
            
            # Exit short: RSI reaches exit level from above
            elif prev_rsi > exit_level and current_rsi <= exit_level:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
        
        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) strategy.
    
    Entry signals:
    - Long: MACD line crosses above signal line (bullish crossover)
    - Short: MACD line crosses below signal line (bearish crossover)
    
    Exit signals:
    - Opposite crossover
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        super().__init__(name="macd_crossover")
        self.config = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        
        fast = self.config['fast_period']
        slow = self.config['slow_period']
        signal_period = self.config['signal_period']
        
        # Calculate MACD
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        macd_hist = macd_line - signal_line
        
        # Initialize signals
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['side'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            current_macd = macd_line.iloc[i]
            prev_macd = macd_line.iloc[i-1]
            current_signal = signal_line.iloc[i]
            prev_signal = signal_line.iloc[i-1]
            
            if pd.isna(current_macd) or pd.isna(prev_macd):
                continue
            
            # Long entry: MACD crosses above signal line
            if prev_macd <= prev_signal and current_macd > current_signal:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'long'
            
            # Short entry: MACD crosses below signal line
            elif prev_macd >= prev_signal and current_macd < current_signal:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'short'
            
            # Exit long: MACD crosses below signal line
            elif prev_macd >= prev_signal and current_macd < current_signal:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
            
            # Exit short: MACD crosses above signal line
            elif prev_macd <= prev_signal and current_macd > current_signal:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
        
        return signals


class MAStrategy(BaseStrategy):
    """
    Moving Average crossover strategy.
    
    Entry signals:
    - Long: Fast MA crosses above slow MA (golden cross)
    - Short: Fast MA crosses below slow MA (death cross)
    
    Exit signals:
    - Opposite crossover
    """
    
    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
    ):
        super().__init__(name="ma_crossover")
        self.config = {
            'fast_period': fast_period,
            'slow_period': slow_period,
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        
        fast_period = self.config['fast_period']
        slow_period = self.config['slow_period']
        
        # Calculate moving averages
        fast_ma = df['close'].rolling(window=fast_period).mean()
        slow_ma = df['close'].rolling(window=slow_period).mean()
        
        # Initialize signals
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['side'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            current_fast = fast_ma.iloc[i]
            prev_fast = fast_ma.iloc[i-1]
            current_slow = slow_ma.iloc[i]
            prev_slow = slow_ma.iloc[i-1]
            
            if pd.isna(current_fast) or pd.isna(prev_fast):
                continue
            
            # Long entry: Fast MA crosses above slow MA
            if prev_fast <= prev_slow and current_fast > current_slow:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'long'
            
            # Short entry: Fast MA crosses below slow MA
            elif prev_fast >= prev_slow and current_fast < current_slow:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'short'
            
            # Exit long: Fast MA crosses below slow MA
            elif prev_fast >= prev_slow and current_fast < current_slow:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
            
            # Exit short: Fast MA crosses above slow MA
            elif prev_fast <= prev_slow and current_fast > current_slow:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
        
        return signals


# Strategy registry for easy access
STRATEGIES = {
    'rsi': RSIStrategy,
    'macd': MACDStrategy,
    'ma': MAStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Get a strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)
