"""
Mean reversion trading strategies for Binance Futures.
Profits from price returning to average levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion strategy.
    
    Entry signals:
    - Long: Price touches or crosses below lower band
    - Short: Price touches or crosses above upper band
    
    Exit signals:
    - Price returns to middle band (SMA)
    """
    
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ):
        super().__init__(name="bollinger_bands")
        self.config = {
            'period': period,
            'std_dev': std_dev,
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        
        period = self.config['period']
        std_dev = self.config['std_dev']
        
        # Calculate Bollinger Bands
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        # Initialize signals
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['side'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            current_close = df['close'].iloc[i]
            current_upper = upper.iloc[i]
            current_lower = lower.iloc[i]
            current_middle = middle.iloc[i]
            
            if pd.isna(current_upper) or pd.isna(current_lower):
                continue
            
            # Long entry: Price crosses below lower band
            if current_close <= current_lower:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'long'
            
            # Short entry: Price crosses above upper band
            elif current_close >= current_upper:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'short'
            
            # Exit long: Price crosses above middle band
            elif current_close >= current_middle:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
            
            # Exit short: Price crosses below middle band
            elif current_close <= current_middle:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
        
        return signals


class ZScoreStrategy(BaseStrategy):
    """
    Z-Score mean reversion strategy.
    
    Uses statistical z-score to identify overextended prices.
    
    Entry signals:
    - Long: Z-score < -threshold (price significantly below mean)
    - Short: Z-score > threshold (price significantly above mean)
    
    Exit signals:
    - Z-score returns to neutral zone
    """
    
    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ):
        super().__init__(name="zscore_reversion")
        self.config = {
            'lookback': lookback,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = self.config['lookback']
        entry_thresh = self.config['entry_threshold']
        exit_thresh = self.config['exit_threshold']
        
        # Calculate rolling mean and std
        rolling_mean = df['close'].rolling(window=lookback).mean()
        rolling_std = df['close'].rolling(window=lookback).std()
        
        # Calculate z-score
        zscore = (df['close'] - rolling_mean) / rolling_std
        
        # Initialize signals
        signals = pd.DataFrame(index=df.index)
        signals['action'] = 'hold'
        signals['side'] = None
        
        # Generate signals
        for i in range(1, len(df)):
            current_z = zscore.iloc[i]
            prev_z = zscore.iloc[i-1]
            
            if pd.isna(current_z) or pd.isna(prev_z):
                continue
            
            # Long entry: Z-score crosses below -threshold
            if prev_z >= -entry_thresh and current_z < -entry_thresh:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'long'
            
            # Short entry: Z-score crosses above +threshold
            elif prev_z <= entry_thresh and current_z > entry_thresh:
                signals.iloc[i, signals.columns.get_loc('action')] = 'entry'
                signals.iloc[i, signals.columns.get_loc('side')] = 'short'
            
            # Exit long: Z-score returns above -exit_threshold
            elif prev_z < -exit_thresh and current_z >= -exit_thresh:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
            
            # Exit short: Z-score returns below +exit_threshold
            elif prev_z > exit_thresh and current_z <= exit_thresh:
                signals.iloc[i, signals.columns.get_loc('action')] = 'exit'
        
        return signals


# Strategy registry
MEAN_REVERSION_STRATEGIES = {
    'bollinger': BollingerBandsStrategy,
    'zscore': ZScoreStrategy,
}


def get_strategy(name: str, **kwargs):
    """Get a mean reversion strategy by name."""
    if name not in MEAN_REVERSION_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(MEAN_REVERSION_STRATEGIES.keys())}")
    return MEAN_REVERSION_STRATEGIES[name](**kwargs)
