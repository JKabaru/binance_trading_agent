"""
Base class for trading strategies.
All strategies should inherit from this class.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All custom strategies must implement:
    - generate_signals(): Generate entry/exit signals
    - get_config(): Return current strategy configuration
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.config = {}
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with 'action' and 'side' columns
            - action: 'entry', 'exit', or 'hold'
            - side: 'long', 'short', or None
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return current strategy configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Update strategy configuration."""
        self.config.update(config)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators.
        Can be overridden by subclasses.
        """
        df = df.copy()
        
        # Moving averages
        if 'close' in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        return df
