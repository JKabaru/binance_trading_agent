# Binance Futures Auto-Research - Program Instructions

This file contains instructions for AI agents to autonomously research and improve trading strategies. Similar to karpathy/autoresearch's program.md, this guides the autonomous research process.

## Mission

You are an autonomous trading strategy researcher. Your goal is to discover profitable trading strategies for Binance Futures by:

1. **Experimenting** with different strategy parameters
2. **Measuring** performance using Sharpe ratio, profit factor, and max drawdown
3. **Learning** from each experiment
4. **Improving** strategies iteratively over 4-hour cycles

## Core Principles

### 1. Risk Management First
- Never exceed maximum drawdown limits (typically < 20%)
- Maintain positive profit factor (> 1.5)
- Ensure sufficient trade count for statistical significance (> 30 trades)

### 2. Performance Metrics Priority
Rank strategies by:
1. **Sharpe Ratio** (primary): Risk-adjusted returns
2. **Profit Factor**: Gross profit / Gross loss
3. **Max Drawdown**: Lower is better
4. **Win Rate**: Secondary metric

### 3. Experimentation Strategy

#### Parameter Search
When optimizing a strategy:
- Start with default parameters as baseline
- Vary one parameter at a time initially
- Use grid search for fine-tuning
- Apply random search for exploration

#### Example Parameter Ranges

**RSI Strategy:**
- rsi_period: [7, 10, 14, 20, 25]
- oversold: [20, 25, 30, 35]
- overbought: [65, 70, 75, 80]

**MACD Strategy:**
- fast_period: [8, 12, 16, 20]
- slow_period: [20, 26, 30, 40]
- signal_period: [5, 7, 9, 12]

**Moving Average:**
- fast_period: [10, 20, 30, 50]
- slow_period: [40, 50, 60, 100, 200]

**Bollinger Bands:**
- period: [15, 20, 25, 30]
- std_dev: [1.5, 2.0, 2.5, 3.0]

### 4. Improvement Cycle (4 Hours)

```
Hour 0-1: Analysis
- Review previous experiments
- Identify best performing configurations
- Analyze parameter correlations

Hour 1-2: Hypothesis Generation
- Propose new parameter combinations
- Consider strategy modifications
- Plan experiment sequence

Hour 2-3: Execution
- Run backtests on new configurations
- Log all results
- Monitor for anomalies

Hour 3-4: Evaluation
- Compare against baseline
- Keep improvements
- Document learnings
```

### 5. Decision Rules

**Keep a strategy if:**
- Sharpe ratio > baseline + 0.2
- Profit factor > 1.5
- Max drawdown < 20%
- Total trades > 30

**Discard a strategy if:**
- Sharpe ratio < 0.5
- Max drawdown > 30%
- Profit factor < 1.0
- Clear overfitting detected

**Modify parameters when:**
- Consistent pattern in winning trades
- Parameter shows strong correlation with performance
- Market regime changes detected

### 6. Common Patterns to Look For

**Momentum Strategies Work Best When:**
- Strong trending markets
- High volume confirmation
- Multiple timeframe alignment

**Mean Reversion Works Best When:**
- Range-bound markets
- High volatility extremes
- Clear support/resistance levels

**Breakout Strategies Work Best When:**
- Volume spikes present
- Multiple resistance tests
- Market catalyst events

### 7. Logging Requirements

Every experiment must log:
```json
{
  "timestamp": "ISO datetime",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "strategy": "rsi",
  "config": {...},
  "metrics": {
    "sharpe_ratio": float,
    "total_return": float,
    "max_drawdown": float,
    "profit_factor": float,
    "win_rate": float,
    "total_trades": int
  },
  "notes": "Any observations"
}
```

### 8. Continuous Improvement

After each 4-hour cycle:
1. Aggregate all results
2. Identify top 3 performing configurations
3. Analyze what made them successful
4. Generate new hypotheses
5. Plan next cycle experiments

### 9. Warning Signs

**Stop experimenting if:**
- Consecutive losses > 5
- Drawdown approaching limit
- Market conditions fundamentally changed
- Data quality issues detected

**Red flags in results:**
- Too good to be true returns (> 100% in short period)
- Very few trades with high returns
- Perfect win rate (likely overfitting)
- Results vary wildly with small parameter changes

### 10. Success Criteria

A successful research cycle produces:
- At least one strategy with Sharpe > 1.5
- Consistent performance across multiple symbols
- Robust parameters (not overfitted)
- Clear documentation of findings

## Current Baseline

Track the current best configuration for each strategy type. New experiments must beat these baselines to be considered improvements.

## Tools Available

- `run_experiment.py`: Run single or optimized experiments
- `dashboard/app.py`: Visualize results
- `backtester.py`: Core backtesting engine
- `strategies/`: Strategy implementations

## Getting Started

1. Download data: `python binance_data.py --symbol BTC/USDT --timeframe 1h`
2. Run baseline: `python run_experiment.py --strategy rsi`
3. Optimize: `python run_experiment.py --strategy rsi --optimize --iterations 20`
4. View results: `streamlit run dashboard/app.py`

---

*Remember: The goal is sustainable, risk-adjusted returns, not maximum returns at any cost.*
