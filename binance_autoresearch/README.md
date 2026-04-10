# Binance Futures Auto-Research System

This project adapts the [autoresearch](https://github.com/karpathy/autoresearch) framework for **Binance Futures trading strategy optimization**. The system autonomously researches, tests, and improves trading strategies over time.

## Overview

The system works by:
1. **Downloading historical Binance Futures data** (klines/candlestick data)
2. **Implementing trading strategies** that can be backtested
3. **Running 4-hour experiment cycles** where AI agents modify strategy parameters
4. **Measuring performance** using Sharpe ratio, profit factor, max drawdown
5. **Self-improving** by keeping strategies that perform better than baseline
6. **Visualizing results** in a Streamlit dashboard

## Project Structure

```
binance_autoresearch/
├── strategies/           # Trading strategy implementations
│   ├── base.py          # Base strategy class
│   ├── momentum.py      # Momentum-based strategies
│   ├── mean_reversion.py # Mean reversion strategies
│   └── breakout.py      # Breakout strategies
├── data/                # Historical price data
├── logs/                # Experiment logs
├── results/             # Backtest results
├── dashboard/           # Streamlit dashboard
│   └── app.py
├── binance_data.py      # Data collection from Binance API
├── backtester.py        # Backtesting engine
├── program.md           # AI agent instructions
└── run_experiment.py    # Main experiment runner
```

## Quick Start

### 1. Install Dependencies

```bash
pip install ccxt pandas numpy torch streamlit plotly ta-lib
```

### 2. Download Historical Data

```bash
python binance_data.py --symbol BTCUSDT --timeframe 1h --days 30
```

### 3. Run Baseline Strategy

```bash
python run_experiment.py --strategy momentum --symbol BTCUSDT
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Trading Strategies

### Momentum Strategies
- RSI-based entry/exit
- MACD crossover
- Moving average trends

### Mean Reversion
- Bollinger Bands bounce
- Statistical arbitrage

### Breakout Strategies
- Support/Resistance breakouts
- Volume confirmation

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Gross profit / Gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Cumulative P&L

## Autonomous Improvement Cycle

Every 4 hours, the system:
1. Analyzes previous experiment results
2. Modifies strategy parameters or logic
3. Runs backtest on historical data
4. Compares against baseline
5. Keeps improvements, discards regressions
6. Logs all changes for review

## Configuration

Edit `program.md` to customize:
- Which symbols to trade
- Timeframes to analyze
- Risk management rules
- Strategy constraints

## Disclaimer

This is for **research and educational purposes only**. Trading futures involves substantial risk. Past performance does not guarantee future results.
