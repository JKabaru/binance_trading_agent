"""
Main experiment runner for Binance Futures Auto-Research.
Adapts the autoresearch framework for trading strategy optimization.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from binance_data import BinanceDataCollector
from backtester import Backtester, BacktestResult
from strategies.momentum import get_strategy as get_momentum_strategy
from strategies.mean_reversion import get_strategy as get_mr_strategy
from strategies.breakout import get_strategy as get_breakout_strategy


class ExperimentRunner:
    """
    Runs trading strategy experiments with autonomous improvement.
    
    Similar to karpathy/autoresearch but for trading strategies:
    - Fixed time budget per experiment (e.g., 4 hours)
    - Measures strategy performance (Sharpe ratio, profit factor)
    - Keeps improvements, discards regressions
    - Logs all changes for review
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        results_dir: str = "results",
        logs_dir: str = "logs",
        initial_capital: float = 10000.0,
        leverage: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.initial_capital = initial_capital
        self.leverage = leverage
        
        # Data collector
        self.data_collector = BinanceDataCollector(data_dir=str(self.data_dir))
        
        # Best result tracking
        self.best_result = None
        self.best_config = None
    
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load or download historical data."""
        try:
            df = self.data_collector.load_data(symbol, timeframe)
        except FileNotFoundError:
            print(f"Data not found, downloading...")
            df = self.data_collector.fetch_klines(symbol, timeframe, days=90)
            self.data_collector.save_data(df, symbol, timeframe)
        return df
    
    def get_strategy(self, strategy_type: str, config: dict):
        """Get strategy instance by type."""
        if strategy_type == 'rsi' or strategy_type == 'macd' or strategy_type == 'ma':
            return get_momentum_strategy(strategy_type, **config)
        elif strategy_type == 'bollinger' or strategy_type == 'zscore':
            return get_mr_strategy(strategy_type, **config)
        elif strategy_type == 'donchian' or strategy_type == 'volume':
            return get_breakout_strategy(strategy_type, **config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def run_experiment(
        self,
        symbol: str,
        timeframe: str,
        strategy_type: str,
        config: dict,
        time_budget: int = 300,  # seconds
    ) -> tuple[BacktestResult, dict]:
        """
        Run a single strategy experiment.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe
            strategy_type: Strategy name
            config: Strategy configuration
            time_budget: Time budget for experiment (seconds)
        
        Returns:
            Tuple of (BacktestResult, config_dict)
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {strategy_type} on {symbol} ({timeframe})")
        print(f"Config: {config}")
        print(f"Time budget: {time_budget}s")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Load data
        df = self.load_data(symbol, timeframe)
        df.attrs['symbol'] = symbol
        
        # Create strategy
        strategy = self.get_strategy(strategy_type, config)
        
        # Run backtest
        backtester = Backtester(
            initial_capital=self.initial_capital,
            leverage=self.leverage,
        )
        
        result = backtester.run(df, strategy)
        
        # Calculate config hash for tracking
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8]
        result.config_hash = config_hash
        
        elapsed = time.time() - start_time
        
        # Log results
        self.log_experiment({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy_type,
            'config': config,
            'config_hash': config_hash,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'profit_factor': result.profit_factor,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'elapsed_seconds': elapsed,
        })
        
        # Print summary
        print(f"\nResults ({elapsed:.2f}s):")
        print(f"  Total Return: {result.total_return:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Total Trades: {result.total_trades}")
        
        # Check if this is an improvement
        is_improvement = False
        if self.best_result is None:
            is_improvement = True
            self.best_result = result
            self.best_config = config.copy()
        elif result.sharpe_ratio > self.best_result.sharpe_ratio:
            is_improvement = True
            print(f"\n✓ NEW BEST! Sharpe improved from {self.best_result.sharpe_ratio:.2f} to {result.sharpe_ratio:.2f}")
            self.best_result = result
            self.best_config = config.copy()
        
        # Save detailed results
        self.save_results(result, symbol, strategy_type, config_hash)
        
        return result, config, is_improvement
    
    def log_experiment(self, data: dict):
        """Log experiment results to JSON file."""
        log_file = self.logs_dir / f"{datetime.now().strftime('%Y%m%d')}_experiments.json"
        
        # Load existing logs
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(data)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def save_results(self, result: BacktestResult, symbol: str, strategy: str, config_hash: str):
        """Save detailed backtest results."""
        filename = f"{symbol.replace('/', '_')}_{strategy}_{config_hash}.json"
        filepath = self.results_dir / filename
        
        # Convert trades to serializable format
        trades_data = []
        for trade in result.trades:
            trades_data.append({
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
            })
        
        # Convert equity curve to list
        equity_data = {
            'index': result.equity_curve.index.astype(str).tolist(),
            'values': result.equity_curve.values.tolist(),
        }
        
        results_dict = {
            'symbol': symbol,
            'strategy': strategy,
            'config_hash': config_hash,
            'metrics': {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'profit_factor': result.profit_factor,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
            },
            'trades': trades_data,
            'equity_curve': equity_data,
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def optimize_strategy(
        self,
        symbol: str,
        timeframe: str,
        strategy_type: str,
        base_config: dict,
        param_grid: dict,
        num_iterations: int = 10,
    ):
        """
        Optimize strategy parameters through grid search.
        
        This simulates the autonomous research process where
        the system tries different parameter combinations.
        """
        print(f"\nStarting optimization: {strategy_type}")
        print(f"Base config: {base_config}")
        print(f"Parameter grid: {param_grid}")
        print(f"Iterations: {num_iterations}\n")
        
        best_sharpe = float('-inf')
        best_config = None
        best_result = None
        
        for i in range(num_iterations):
            # Generate random config from grid
            config = {}
            for param, values in param_grid.items():
                if isinstance(values[0], (int, float)):
                    val = np.random.choice(values)
                    config[param] = int(val) if isinstance(val, (np.integer, np.int64)) else val
                elif isinstance(values[0], (list, tuple)):
                    val = np.random.uniform(values[0], values[1])
                    config[param] = float(val) if isinstance(val, (np.floating, np.float64)) else val
            
            # Add base config items not in param_grid
            for k, v in base_config.items():
                if k not in config:
                    config[k] = v
            
            # Run experiment
            result, _, _ = self.run_experiment(
                symbol=symbol,
                timeframe=timeframe,
                strategy_type=strategy_type,
                config=config,
                time_budget=60,  # Short budget for optimization
            )
            
            # Track best
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_config = config.copy()
                best_result = result
            
            print(f"\nIteration {i+1}/{num_iterations} - Best Sharpe so far: {best_sharpe:.2f}")
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"Best Config: {best_config}")
        print(f"Best Sharpe: {best_sharpe:.2f}")
        print(f"{'='*60}\n")
        
        return best_config, best_result


def main():
    parser = argparse.ArgumentParser(description="Run trading strategy experiments")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candlestick timeframe")
    parser.add_argument("--strategy", type=str, default="rsi", 
                       choices=['rsi', 'macd', 'ma', 'bollinger', 'zscore', 'donchian', 'volume'],
                       help="Strategy type")
    parser.add_argument("--leverage", type=int, default=1, help="Leverage multiplier")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    parser.add_argument("--iterations", type=int, default=10, help="Number of optimization iterations")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner(leverage=args.leverage)
    
    # Default configs for each strategy
    default_configs = {
        'rsi': {'rsi_period': 14, 'oversold': 30.0, 'overbought': 70.0},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'ma': {'fast_period': 20, 'slow_period': 50},
        'bollinger': {'period': 20, 'std_dev': 2.0},
        'zscore': {'lookback': 20, 'entry_threshold': 2.0, 'exit_threshold': 0.5},
        'donchian': {'period': 20},
        'volume': {'lookback': 20, 'volume_multiplier': 1.5},
    }
    
    base_config = default_configs.get(args.strategy, {})
    
    if args.optimize:
        # Define parameter grid for optimization
        param_grids = {
            'rsi': {
                'rsi_period': [10, 14, 20, 25],
                'oversold': [25.0, 30.0, 35.0],
                'overbought': [65.0, 70.0, 75.0],
            },
            'macd': {
                'fast_period': [8, 12, 16],
                'slow_period': [20, 26, 30],
                'signal_period': [7, 9, 12],
            },
            'ma': {
                'fast_period': [10, 20, 30],
                'slow_period': [40, 50, 60],
            },
            'bollinger': {
                'period': [15, 20, 25],
                'std_dev': [1.5, 2.0, 2.5],
            },
        }
        
        param_grid = param_grids.get(args.strategy, {})
        
        runner.optimize_strategy(
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategy_type=args.strategy,
            base_config=base_config,
            param_grid=param_grid,
            num_iterations=args.iterations,
        )
    else:
        # Single experiment
        runner.run_experiment(
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategy_type=args.strategy,
            config=base_config,
            time_budget=300,
        )


if __name__ == "__main__":
    main()
