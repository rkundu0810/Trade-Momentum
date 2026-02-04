"""
Utility functions for performance metrics and visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import json
import os


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_cagr(equity_series: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        equity_series: Time series of portfolio values with datetime index

    Returns:
        CAGR as a decimal (e.g., 0.52 for 52%)
    """
    if len(equity_series) < 2:
        return 0.0

    start_value = equity_series.iloc[0]
    end_value = equity_series.iloc[-1]

    # Calculate years between first and last date
    years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25

    if years <= 0 or start_value <= 0:
        return 0.0

    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr


def calculate_total_return(equity_series: pd.Series) -> float:
    """
    Calculate total return over the period.

    Returns:
        Total return as a decimal (e.g., 43.57 for 4357%)
    """
    if len(equity_series) < 2:
        return 0.0

    return (equity_series.iloc[-1] / equity_series.iloc[0]) - 1


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    # Daily risk-free rate
    daily_rf = risk_free_rate / 252

    excess_returns = returns - daily_rf
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sortino Ratio (downside deviation only).

    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_std = downside_returns.std()
    sortino = excess_returns.mean() / downside_std * np.sqrt(252)

    return sortino


def calculate_max_drawdown(equity_series: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.42 for 42% drawdown)
    """
    if len(equity_series) < 2:
        return 0.0

    # Calculate running maximum
    running_max = equity_series.expanding().max()

    # Calculate drawdowns
    drawdowns = (equity_series - running_max) / running_max

    # Maximum drawdown (most negative value)
    max_dd = drawdowns.min()

    return abs(max_dd)


def calculate_drawdown_series(equity_series: pd.Series) -> pd.Series:
    """
    Calculate drawdown series over time.

    Returns:
        Series of drawdown percentages (negative values)
    """
    running_max = equity_series.expanding().max()
    drawdowns = (equity_series - running_max) / running_max
    return drawdowns


def calculate_calmar_ratio(equity_series: pd.Series) -> float:
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).

    Returns:
        Calmar Ratio
    """
    cagr = calculate_cagr(equity_series)
    max_dd = calculate_max_drawdown(equity_series)

    if max_dd == 0:
        return float('inf') if cagr > 0 else 0.0

    return cagr / max_dd


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        trades_df: DataFrame with 'pnl' column

    Returns:
        Profit factor
    """
    if 'pnl' not in trades_df.columns or len(trades_df) == 0:
        return 0.0

    gross_profits = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0

    return gross_profits / gross_losses


def calculate_win_rate(trades_df: pd.DataFrame) -> float:
    """
    Calculate win rate (% of profitable trades).

    Args:
        trades_df: DataFrame with 'pnl' column

    Returns:
        Win rate as decimal (e.g., 0.39 for 39%)
    """
    if 'pnl' not in trades_df.columns or len(trades_df) == 0:
        return 0.0

    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    total_trades = len(trades_df)

    return winning_trades / total_trades


def calculate_avg_trade(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate average trade statistics.

    Returns:
        Dictionary with avg_win, avg_loss, avg_trade, expectancy
    """
    if 'pnl' not in trades_df.columns or len(trades_df) == 0:
        return {'avg_win': 0, 'avg_loss': 0, 'avg_trade': 0, 'expectancy': 0}

    winners = trades_df[trades_df['pnl'] > 0]['pnl']
    losers = trades_df[trades_df['pnl'] < 0]['pnl']

    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0
    avg_trade = trades_df['pnl'].mean()

    # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
    win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade': avg_trade,
        'expectancy': expectancy
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_equity_curve(equity_series: pd.Series, benchmark_series: Optional[pd.Series] = None,
                      title: str = "Equity Curve", log_scale: bool = True,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot equity curve with optional benchmark comparison.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Normalize to start at same value for comparison
    equity_norm = equity_series / equity_series.iloc[0] * 100
    ax.plot(equity_norm.index, equity_norm.values, label='Strategy', linewidth=2, color='blue')

    if benchmark_series is not None:
        benchmark_norm = benchmark_series / benchmark_series.iloc[0] * 100
        ax.plot(benchmark_norm.index, benchmark_norm.values, label='Benchmark (SPY)',
                linewidth=1.5, color='gray', alpha=0.7)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (Normalized to 100)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_drawdown(equity_series: pd.Series, title: str = "Drawdown",
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot drawdown chart.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    drawdowns = calculate_drawdown_series(equity_series)

    ax.fill_between(drawdowns.index, drawdowns.values * 100, 0,
                    color='red', alpha=0.3, label='Drawdown')
    ax.plot(drawdowns.index, drawdowns.values * 100, color='red', linewidth=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate max drawdown
    max_dd_idx = drawdowns.idxmin()
    max_dd_val = drawdowns.min() * 100
    ax.annotate(f'Max DD: {max_dd_val:.1f}%', xy=(max_dd_idx, max_dd_val),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_monthly_returns_heatmap(equity_series: pd.Series, title: str = "Monthly Returns Heatmap",
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot monthly returns as a heatmap (years × months).
    """
    # Calculate monthly returns
    monthly_equity = equity_series.resample('ME').last()
    monthly_returns = monthly_equity.pct_change() * 100

    # Create pivot table (years as rows, months as columns)
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(pivot, annot=True, fmt='.1f', center=0,
                cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Return (%)'},
                linewidths=0.5, annot_kws={'size': 9})

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_annual_returns(equity_series: pd.Series, benchmark_series: Optional[pd.Series] = None,
                        title: str = "Annual Returns", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot annual returns as a bar chart.
    """
    # Calculate annual returns
    annual_equity = equity_series.resample('YE').last()
    annual_returns = annual_equity.pct_change() * 100
    annual_returns = annual_returns.dropna()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(annual_returns))
    width = 0.35

    bars1 = ax.bar(x - width/2, annual_returns.values, width, label='Strategy', color='blue', alpha=0.7)

    if benchmark_series is not None:
        bench_annual = benchmark_series.resample('YE').last()
        bench_returns = bench_annual.pct_change() * 100
        bench_returns = bench_returns.dropna()

        # Align with strategy years
        common_years = annual_returns.index.intersection(bench_returns.index)
        bench_returns = bench_returns.loc[common_years]

        bars2 = ax.bar(x + width/2, bench_returns.values, width, label='Benchmark (SPY)', color='gray', alpha=0.7)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.year for d in annual_returns.index], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# ============================================================================
# REPORTING
# ============================================================================

def calculate_all_metrics(equity_series: pd.Series, trades_df: pd.DataFrame,
                          benchmark_series: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Calculate all performance metrics.

    Returns:
        Dictionary of all metrics
    """
    returns = equity_series.pct_change().dropna()

    metrics = {
        'total_return_pct': calculate_total_return(equity_series) * 100,
        'cagr_pct': calculate_cagr(equity_series) * 100,
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown_pct': calculate_max_drawdown(equity_series) * 100,
        'calmar_ratio': calculate_calmar_ratio(equity_series),
        'volatility_pct': returns.std() * np.sqrt(252) * 100,
        'total_trades': len(trades_df),
        'win_rate_pct': calculate_win_rate(trades_df) * 100,
        'profit_factor': calculate_profit_factor(trades_df),
        'start_value': equity_series.iloc[0],
        'end_value': equity_series.iloc[-1],
        'start_date': str(equity_series.index[0].date()),
        'end_date': str(equity_series.index[-1].date()),
    }

    # Add average trade stats
    avg_stats = calculate_avg_trade(trades_df)
    metrics.update({
        'avg_win': avg_stats['avg_win'],
        'avg_loss': avg_stats['avg_loss'],
        'avg_trade': avg_stats['avg_trade'],
        'expectancy': avg_stats['expectancy'],
    })

    # Add benchmark comparison if available
    if benchmark_series is not None:
        bench_returns = benchmark_series.pct_change().dropna()
        metrics['benchmark_cagr_pct'] = calculate_cagr(benchmark_series) * 100
        metrics['benchmark_sharpe'] = calculate_sharpe_ratio(bench_returns)
        metrics['benchmark_max_dd_pct'] = calculate_max_drawdown(benchmark_series) * 100
        metrics['alpha_pct'] = metrics['cagr_pct'] - metrics['benchmark_cagr_pct']

    return metrics


def print_metrics_report(metrics: Dict[str, Any], title: str = "BACKTEST RESULTS") -> None:
    """
    Print formatted metrics report to console.
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

    print(f"\n{'PERFORMANCE':^60}")
    print("-" * 60)
    print(f"  Total Return:        {metrics['total_return_pct']:>10.1f}%")
    print(f"  CAGR:                {metrics['cagr_pct']:>10.1f}%")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>10.1f}%")
    print(f"  Volatility (Ann.):   {metrics['volatility_pct']:>10.1f}%")

    print(f"\n{'RISK-ADJUSTED':^60}")
    print("-" * 60)
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

    print(f"\n{'TRADE STATISTICS':^60}")
    print("-" * 60)
    print(f"  Total Trades:        {metrics['total_trades']:>10}")
    print(f"  Win Rate:            {metrics['win_rate_pct']:>10.1f}%")
    print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Winner:          ${metrics['avg_win']:>10,.2f}")
    print(f"  Avg Loser:           ${metrics['avg_loss']:>10,.2f}")
    print(f"  Expectancy:          ${metrics['expectancy']:>10,.2f}")

    if 'benchmark_cagr_pct' in metrics:
        print(f"\n{'VS BENCHMARK (SPY)':^60}")
        print("-" * 60)
        print(f"  Benchmark CAGR:      {metrics['benchmark_cagr_pct']:>10.1f}%")
        print(f"  Benchmark Sharpe:    {metrics['benchmark_sharpe']:>10.2f}")
        print(f"  Benchmark Max DD:    {metrics['benchmark_max_dd_pct']:>10.1f}%")
        print(f"  Alpha:               {metrics['alpha_pct']:>10.1f}%")

    print(f"\n{'PERIOD':^60}")
    print("-" * 60)
    print(f"  Start Date:          {metrics['start_date']:>10}")
    print(f"  End Date:            {metrics['end_date']:>10}")
    print(f"  Start Value:         ${metrics['start_value']:>10,.2f}")
    print(f"  End Value:           ${metrics['end_value']:>10,.2f}")

    print("\n" + "=" * 60)


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics to JSON file."""
    # Convert any non-serializable types
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean_metrics[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = v

    with open(filepath, 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"  Saved: {filepath}")


def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
