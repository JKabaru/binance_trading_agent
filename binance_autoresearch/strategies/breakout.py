"""
Breakout trading strategies for Binance Futures.
Profits from price breaking through support/resistance levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseStrategy


class DonchianChannelStrategy(BaseStrategy):
    """
    Donchian Channel breakout strategy.
    
    Entry signals:
    - Long: Price breaks above upper channel (N-period high)
    - Short: Price breaks below lower channel (N-period low)
    
    Exit signals:
    - Opposite breakout or middle line cross
    """
    
    def __init__(
        self,
        period: int = 20,
    ):
        super().__init__(name="donchian_breakout")
        self.config = {
            'period': period,
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.config['period']
        
        # Calculate Donchian Channel
        upper = df['high'].rolling(window=period).max()
        lower = df['low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        # Initialize signals
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['side'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_upper = upper.iloc[i]
            current_lower = lower.iloc[i]
            
            if pd.isna(current_upper) or pd.isna(current_lower):
                continue
            
            # Long entry: Price breaks above upper channel
            if current_high >= current_upper:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'long'
            
            # Short entry: Price breaks below lower channel
            elif current_low <= current_lower:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'short'
            
            # Exit long: Price crosses below middle
            elif df['close'].iloc[i] <= middle.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
            
            # Exit short: Price crosses above middle
            elif df['close'].iloc[i] >= middle.iloc[i]:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
        
        return signals


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume-confirmed breakout strategy.
    
    Entry signals:
    - Long: Price breaks resistance with volume > threshold
    - Short: Price breaks support with volume > threshold
    
    Exit signals:
    - Price returns to breakout level
    """
    
    def __init__(
        self,
        lookback: int = 20,
        volume_multiplier: float = 1.5,
    ):
        super().__init__(name="volume_breakout")
        self.config = {
            'lookback': lookback,
            'volume_multiplier': volume_multiplier,
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = self.config['lookback']
        vol_mult = self.config['volume_multiplier']
        
        # Calculate resistance and support levels
        resistance = df['high'].rolling(window=lookback).max()
        support = df['low'].rolling(window=lookback).min()
        
        # Calculate volume threshold
        volume_sma = df['volume'].rolling(window=lookback).mean()
        volume_threshold = volume_sma * vol_mult
        
        # Initialize signals
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['side'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_volume = df['volume'].iloc[i]
            current_resistance = resistance.iloc[i]
            current_support = support.iloc[i]
            current_vol_threshold = volume_threshold.iloc[i]
            
            if pd.isna(current_resistance) or pd.isna(current_support):
                continue
            
            # Check if volume confirms breakout
            volume_confirmed = current_volume >= current_vol_threshold
            
            if not volume_confirmed:
                continue
            
            # Long entry: Price breaks resistance with volume
            if current_high >= current_resistance:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'long'
            
            # Short entry: Price breaks support with volume
            elif current_low <= current_support:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'short'
            
            # Exit long: Price falls below entry level
            elif current_close < df['open'].iloc[i]:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
            
            # Exit short: Price rises above entry level
            elif current_close > df['open'].iloc[i]:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
        
        return signals


# Strategy registry
BREAKOUT_STRATEGIES = {
    'donchian': DonchianChannelStrategy,
    'volume': VolumeBreakoutStrategy,
}


def get_strategy(name: str, **kwargs):
    """Get a breakout strategy by name."""
    if name not in BREAKOUT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(BREAKOUT_STRATEGIES.keys())}")
    return BREAKOUT_STRATEGIES[name](**kwargs)
