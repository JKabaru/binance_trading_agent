"""
Streamlit Dashboard for Binance Futures Auto-Research.
Visualizes trading strategy performance and experiment results.
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="Binance Auto-Research Dashboard",
    page_icon="📈",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
.best-metric {
    background-color: #d4edda;
    border: 2px solid #28a745;
}
.worst-metric {
    background-color: #f8d7da;
    border: 2px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_experiment_logs(logs_dir="logs"):
    """Load all experiment logs."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []
    
    all_logs = []
    for log_file in logs_path.glob("*_experiments.json"):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
                all_logs.extend(logs)
        except Exception as e:
            st.error(f"Error loading {log_file}: {e}")
    
    # Sort by timestamp
    all_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return all_logs


@st.cache_data
def load_results(results_dir="results"):
    """Load detailed backtest results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return {}
    
    results = {}
    for result_file in results_path.glob("*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[result_file.stem] = data
        except Exception as e:
            st.error(f"Error loading {result_file}: {e}")
    
    return results


def create_equity_curve_chart(equity_data):
    """Create interactive equity curve chart."""
    df = pd.DataFrame({
        'Date': pd.to_datetime(equity_data['index']),
        'Equity': equity_data['values'],
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)',
    ))
    
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
    )
    
    return fig


def create_drawdown_chart(equity_data):
    """Create drawdown chart."""
    df = pd.DataFrame({
        'Date': pd.to_datetime(equity_data['index']),
        'Equity': equity_data['values'],
    })
    
    # Calculate drawdown
    df['Peak'] = df['Equity'].expanding().max()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Drawdown'],
        mode='lines',
        name='Drawdown',
        line=dict(color='#A23B72', width=2),
        fill='tozeroy',
        fillcolor='rgba(162, 59, 114, 0.3)',
    ))
    
    fig.update_layout(
        title='Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown %',
        hovermode='x unified',
        template='plotly_white',
        height=300,
    )
    
    return fig


def create_metrics_summary(logs):
    """Create summary metrics from experiment logs."""
    if not logs:
        return {}
    
    df = pd.DataFrame(logs)
    
    summary = {
        'total_experiments': len(df),
        'avg_sharpe': df['sharpe_ratio'].mean(),
        'best_sharpe': df['sharpe_ratio'].max(),
        'avg_return': df['total_return'].mean(),
        'best_return': df['total_return'].max(),
        'avg_win_rate': df['win_rate'].mean(),
        'total_trades': df['total_trades'].sum(),
    }
    
    return summary


def main():
    st.title("📈 Binance Futures Auto-Research Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    view = st.sidebar.radio(
        "Select View",
        ["Overview", "Experiment Results", "Strategy Comparison", "Detailed Analysis"],
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    
    # Load data
    logs = load_experiment_logs()
    results = load_results()
    
    if not logs:
        st.warning("No experiment logs found. Run some experiments first!")
        st.info("Use `python run_experiment.py` to run trading strategy experiments.")
        return
    
    # Filter options
    symbols = list(set(log['symbol'] for log in logs))
    strategies = list(set(log['strategy'] for log in logs))
    
    selected_symbol = st.sidebar.multiselect("Symbol", symbols, default=symbols)
    selected_strategy = st.sidebar.multiselect("Strategy", strategies, default=strategies)
    
    # Filter logs
    filtered_logs = [
        log for log in logs
        if log['symbol'] in selected_symbol and log['strategy'] in selected_strategy
    ]
    
    if not filtered_logs:
        st.warning("No experiments match the selected filters.")
        return
    
    # Overview Page
    if view == "Overview":
        st.header("🎯 Overview")
        
        # Summary metrics
        summary = create_metrics_summary(filtered_logs)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Experiments",
                value=summary['total_experiments'],
            )
        
        with col2:
            st.metric(
                label="Best Sharpe Ratio",
                value=f"{summary['best_sharpe']:.2f}",
                delta=f"Avg: {summary['avg_sharpe']:.2f}",
            )
        
        with col3:
            st.metric(
                label="Best Return",
                value=f"{summary['best_return']:.2f}%",
                delta=f"Avg: {summary['avg_return']:.2f}%",
            )
        
        with col4:
            st.metric(
                label="Total Trades",
                value=summary['total_trades'],
            )
        
        st.markdown("---")
        
        # Recent experiments table
        st.subheader("Recent Experiments")
        
        df_logs = pd.DataFrame(filtered_logs)
        
        # Format dataframe for display
        display_df = df_logs[[
            'timestamp', 'symbol', 'timeframe', 'strategy',
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]].copy()
        
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Time', 'Symbol', 'Timeframe', 'Strategy', 
                             'Return (%)', 'Sharpe', 'Max DD (%)', 'Win Rate (%)']
        
        st.dataframe(
            display_df.head(20),
            use_container_width=True,
            hide_index=True,
        )
        
        # Strategy performance distribution
        st.subheader("Strategy Performance Distribution")
        
        strategy_perf = df_logs.groupby('strategy').agg({
            'sharpe_ratio': ['mean', 'std', 'max'],
            'total_return': ['mean', 'std', 'max'],
            'win_rate': 'mean',
        }).round(2)
        
        st.dataframe(strategy_perf, use_container_width=True)
    
    # Experiment Results Page
    elif view == "Experiment Results":
        st.header("📊 Experiment Results")
        
        # Select specific result to view
        result_names = list(results.keys())
        if result_names:
            selected_result = st.selectbox("Select Result", result_names)
            
            if selected_result:
                result_data = results[selected_result]
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = result_data['metrics']
                
                with col1:
                    st.metric("Total Return", f"{metrics['total_return']:.2f}%")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                
                with col4:
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                
                st.markdown("---")
                
                # Equity curve
                if 'equity_curve' in result_data:
                    fig = create_equity_curve_chart(result_data['equity_curve'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Drawdown chart
                    fig_dd = create_drawdown_chart(result_data['equity_curve'])
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                # Trade statistics
                st.subheader("Trade Statistics")
                
                trade_stats = {
                    'Total Trades': metrics['total_trades'],
                    'Winning Trades': metrics['winning_trades'],
                    'Losing Trades': metrics['losing_trades'],
                    'Average Win': f"${metrics['avg_win']:.2f}",
                    'Average Loss': f"${metrics['avg_loss']:.2f}",
                    'Profit Factor': f"{metrics['profit_factor']:.2f}",
                }
                
                st.json(trade_stats)
                
                # Recent trades table
                if 'trades' in result_data and result_data['trades']:
                    st.subheader("Recent Trades")
                    
                    trades_df = pd.DataFrame(result_data['trades'][-10:])  # Last 10 trades
                    
                    if not trades_df.empty:
                        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
                        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
                        
                        # Color code PnL
                        def color_pnl(val):
                            return 'color: green' if val > 0 else 'color: red'
                        
                        st.dataframe(
                            trades_df.style.applymap(color_pnl, subset=['pnl']),
                            use_container_width=True,
                            hide_index=True,
                        )
    
    # Strategy Comparison Page
    elif view == "Strategy Comparison":
        st.header("⚖️ Strategy Comparison")
        
        df_logs = pd.DataFrame(filtered_logs)
        
        # Compare strategies
        if len(df_logs['strategy'].unique()) > 1:
            # Box plot of Sharpe ratios
            fig = px.box(
                df_logs,
                x='strategy',
                y='sharpe_ratio',
                title='Sharpe Ratio by Strategy',
                labels={'strategy': 'Strategy', 'sharpe_ratio': 'Sharpe Ratio'},
                color='strategy',
            )
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare returns
            fig2 = px.bar(
                df_logs.groupby('strategy')['total_return'].mean().reset_index(),
                x='strategy',
                y='total_return',
                title='Average Total Return by Strategy',
                labels={'strategy': 'Strategy', 'total_return': 'Return (%)'},
                color='total_return',
                color_continuous_scale='RdYlGn',
            )
            fig2.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Win rate comparison
            fig3 = px.bar(
                df_logs.groupby('strategy')['win_rate'].mean().reset_index(),
                x='strategy',
                y='win_rate',
                title='Average Win Rate by Strategy',
                labels={'strategy': 'Strategy', 'win_rate': 'Win Rate (%)'},
                color='win_rate',
                color_continuous_scale='Blues',
            )
            fig3.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Select multiple strategies to compare performance.")
    
    # Detailed Analysis Page
    elif view == "Detailed Analysis":
        st.header("🔍 Detailed Analysis")
        
        df_logs = pd.DataFrame(filtered_logs)
        
        # Time series of experiments
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        df_logs = df_logs.sort_values('timestamp')
        
        # Experiment timeline
        fig = go.Figure()
        
        for strategy in df_logs['strategy'].unique():
            strategy_df = df_logs[df_logs['strategy'] == strategy]
            fig.add_trace(go.Scatter(
                x=strategy_df['timestamp'],
                y=strategy_df['sharpe_ratio'],
                mode='markers+lines',
                name=strategy,
                marker=dict(size=10),
            ))
        
        fig.update_layout(
            title='Sharpe Ratio Over Time',
            xaxis_title='Time',
            yaxis_title='Sharpe Ratio',
            template='plotly_white',
            height=500,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Metric Correlations")
        
        corr_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']
        corr_matrix = df_logs[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Metric Correlation Heatmap',
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Parameter analysis (if config is available)
        st.subheader("Parameter Analysis")
        
        if 'config' in df_logs.columns:
            # Extract common parameters
            all_params = set()
            for config in df_logs['config']:
                if isinstance(config, dict):
                    all_params.update(config.keys())
            
            if all_params:
                selected_param = st.selectbox("Select Parameter", list(all_params))
                
                # Analyze parameter impact
                param_values = []
                sharpe_values = []
                
                for _, row in df_logs.iterrows():
                    if isinstance(row['config'], dict) and selected_param in row['config']:
                        param_values.append(row['config'][selected_param])
                        sharpe_values.append(row['sharpe_ratio'])
                
                if param_values:
                    fig_param = px.scatter(
                        x=param_values,
                        y=sharpe_values,
                        labels={'x': selected_param, 'y': 'Sharpe Ratio'},
                        title=f'Impact of {selected_param} on Sharpe Ratio',
                        trendline='ols',
                    )
                    fig_param.update_layout(template='plotly_white', height=500)
                    st.plotly_chart(fig_param, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption(
        "Dashboard updates automatically when new experiments are run. "
        "Data is loaded from the `logs/` and `results/` directories."
    )


if __name__ == "__main__":
    main()
