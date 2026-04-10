"""
Binance Futures data collector.
Downloads historical kline (candlestick) data for backtesting.
"""

import os
import time
import argparse
from datetime import datetime, timedelta

import ccxt
import pandas as pd


class BinanceDataCollector:
    """Collects historical futures data from Binance."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize Binance Futures exchange
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',  # Use futures market
            },
            'rateLimit': 100,  # Respect rate limits
        })
    
    def fetch_klines(self, symbol, timeframe='1h', days=30, limit=None):
        """
        Fetch historical kline data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            days: Number of days to fetch
            limit: Optional limit on number of candles
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {symbol} {timeframe} data for last {days} days...")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)  # Convert to milliseconds
        
        all_ohlcvs = []
        
        while True:
            try:
                # Fetch klines
                ohlcvs = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                
                if not ohlcvs:
                    break
                
                all_ohlcvs.extend(ohlcvs)
                
                # Update since to last fetched timestamp
                since = ohlcvs[-1][0] + 1
                
                # Check if we've reached the end
                if ohlcvs[-1][0] > end_time.timestamp() * 1000:
                    break
                
                # Optional limit
                if limit and len(all_ohlcvs) >= limit:
                    all_ohlcvs = all_ohlcvs[:limit]
                    break
                
                # Rate limit handling
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Fetched {len(df)} candles")
        return df
    
    def save_data(self, df, symbol, timeframe):
        """Save data to CSV file."""
        # Clean symbol name for filename
        safe_symbol = symbol.replace('/', '_').replace(':', '_')
        filename = f"{safe_symbol}_{timeframe}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        df.to_csv(filepath)
        print(f"Saved data to {filepath}")
        return filepath
    
    def load_data(self, symbol, timeframe):
        """Load previously saved data."""
        safe_symbol = symbol.replace('/', '_').replace(':', '_')
        filename = f"{safe_symbol}_{timeframe}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        print(f"Loaded {len(df)} candles from {filepath}")
        return df


def main():
    parser = argparse.ArgumentParser(description="Download Binance Futures data")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", 
                       choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
                       help="Candlestick timeframe")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    collector = BinanceDataCollector(data_dir=args.output_dir)
    df = collector.fetch_klines(args.symbol, args.timeframe, args.days)
    collector.save_data(df, args.symbol, args.timeframe)
    
    print("\nData summary:")
    print(df.describe())


if __name__ == "__main__":
    main()
