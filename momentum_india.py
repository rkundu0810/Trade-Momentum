"""
Momentum Strategy for Indian Stocks (NSE)

Goal: Rs.30 Lakhs -> Rs.4 Crores using momentum strategy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import os

from config_india import IndiaStrategyConfig, NIFTY50_UNIVERSE, QUALITY_MIDCAP_UNIVERSE
from utils import (
    calculate_all_metrics,
    print_metrics_report,
    save_metrics_json,
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns_heatmap,
    plot_annual_returns,
    ensure_output_dir,
    calculate_cagr,
    calculate_max_drawdown,
)

warnings.filterwarnings('ignore')


class IndianMomentumBacktest:
    """Momentum backtest engine for Indian stocks."""

    def __init__(self, config: IndiaStrategyConfig):
        self.config = config
        self.prices = None
        self.benchmark_prices = None

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch price data from Yahoo Finance."""
        buffer_days = self.config.lookback_days + 30
        start_with_buffer = pd.to_datetime(self.config.start_date) - timedelta(days=buffer_days * 1.5)

        print(f"\nFetching Indian stock data...")
        print(f"  Universe: {len(self.config.universe)} stocks")
        print(f"  Period: {start_with_buffer.date()} to {self.config.end_date}")

        tickers = self.config.universe + [self.config.benchmark_ticker]

        data = yf.download(
            tickers=tickers,
            start=start_with_buffer,
            end=self.config.end_date,
            progress=False,
            auto_adjust=True,
        )

        if len(tickers) == 1:
            prices = data['Close'].to_frame(name=tickers[0])
        else:
            prices = data['Close']

        benchmark_prices = prices[self.config.benchmark_ticker].copy()
        universe_prices = prices[[t for t in self.config.universe if t in prices.columns]].copy()

        universe_prices = universe_prices.ffill()
        benchmark_prices = benchmark_prices.ffill()

        print(f"  Trading days: {len(universe_prices)}")
        print(f"  Stocks with data: {universe_prices.notna().any().sum()} / {len(self.config.universe)}")

        self.prices = universe_prices
        self.benchmark_prices = benchmark_prices

        return universe_prices, benchmark_prices

    def calculate_momentum(self, date: pd.Timestamp) -> pd.Series:
        """Calculate 12-month momentum for all stocks."""
        prices_to_date = self.prices.loc[:date]

        if len(prices_to_date) < self.config.lookback_days:
            return pd.Series(dtype=float)

        current = prices_to_date.iloc[-1]
        past = prices_to_date.iloc[-self.config.lookback_days]

        momentum = (current / past) - 1
        return momentum.dropna()

    def get_top_stocks(self, date: pd.Timestamp) -> List[str]:
        """Get top N momentum stocks."""
        momentum = self.calculate_momentum(date)
        if len(momentum) == 0:
            return []
        return momentum.sort_values(ascending=False).head(self.config.top_n).index.tolist()

    def is_risk_on(self, date: pd.Timestamp) -> bool:
        """
        Check if market is in risk-on mode (Nifty 50 > 200 SMA).
        Returns True if should be invested, False if should go to cash.
        """
        if not self.config.use_regime_filter:
            return True

        prices_to_date = self.benchmark_prices.loc[:date]

        if len(prices_to_date) < self.config.regime_sma_period:
            return True  # Default to invested if not enough data

        sma_200 = prices_to_date.iloc[-self.config.regime_sma_period:].mean()
        current_price = prices_to_date.iloc[-1]

        return current_price > sma_200

    def run_backtest(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Run the momentum backtest."""
        self.fetch_data()

        cash = self.config.initial_capital
        holdings = {}
        entry_prices = {}
        equity_history = []
        trades = []
        cash_periods = 0  # Track how many months in cash

        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        trading_days = self.prices.index[(self.prices.index >= start) & (self.prices.index <= end)]

        # Get rebalance dates (first trading day of each month)
        monthly_groups = trading_days.to_series().groupby([trading_days.year, trading_days.month])
        rebalance_dates = set(group.iloc[0] for _, group in monthly_groups)

        print(f"\nRunning backtest...")
        print(f"  Initial capital: Rs.{cash:,.0f}")
        print(f"  Rebalance dates: {len(rebalance_dates)}")
        print(f"  Regime filter: {'ON (Nifty > 200 SMA)' if self.config.use_regime_filter else 'OFF'}")

        for date in trading_days:
            if date in rebalance_dates:
                # Check regime filter first
                risk_on = self.is_risk_on(date)

                if risk_on:
                    new_stocks = self.get_top_stocks(date)
                else:
                    new_stocks = []  # Go to cash
                    cash_periods += 1
                prices_today = self.prices.loc[date]

                # Sell current holdings
                for ticker, shares in list(holdings.items()):
                    if ticker in prices_today and not pd.isna(prices_today[ticker]):
                        sell_price = prices_today[ticker]
                        sell_value = shares * sell_price
                        cost = sell_value * self.config.transaction_cost_pct
                        pnl = (sell_price - entry_prices.get(ticker, sell_price)) * shares - cost

                        trades.append({
                            'date': date, 'ticker': ticker.replace('.NS', ''),
                            'action': 'SELL', 'shares': shares,
                            'price': sell_price, 'value': sell_value,
                            'cost': cost, 'pnl': pnl
                        })
                        cash += sell_value - cost

                holdings = {}
                entry_prices = {}

                # Buy new stocks
                if len(new_stocks) > 0:
                    weight = 1.0 / len(new_stocks)
                    for ticker in new_stocks:
                        if ticker in prices_today and not pd.isna(prices_today[ticker]):
                            target = cash * weight
                            buy_price = prices_today[ticker]
                            effective = target / (1 + self.config.transaction_cost_pct)
                            shares = effective / buy_price
                            actual_value = shares * buy_price
                            cost = actual_value * self.config.transaction_cost_pct

                            holdings[ticker] = shares
                            entry_prices[ticker] = buy_price
                            cash -= (actual_value + cost)

                            trades.append({
                                'date': date, 'ticker': ticker.replace('.NS', ''),
                                'action': 'BUY', 'shares': shares,
                                'price': buy_price, 'value': actual_value,
                                'cost': cost, 'pnl': 0
                            })

            # Calculate daily equity
            holdings_value = sum(
                shares * self.prices.loc[date].get(ticker, 0)
                for ticker, shares in holdings.items()
                if not pd.isna(self.prices.loc[date].get(ticker, np.nan))
            )
            equity_history.append((date, cash + holdings_value))

        equity_series = pd.Series(dict(equity_history), name='equity')
        trades_df = pd.DataFrame(trades)

        print(f"  Final equity: Rs.{equity_series.iloc[-1]:,.0f}")
        print(f"  Total trades: {len(trades_df)}")
        if self.config.use_regime_filter:
            print(f"  Months in cash (regime off): {cash_periods}")

        self.cash_periods = cash_periods
        return equity_series, trades_df

    def generate_reports(self, equity_series: pd.Series, trades_df: pd.DataFrame) -> Dict:
        """Generate reports and projections."""
        ensure_output_dir(self.config.output_dir)

        start = equity_series.index[0]
        end = equity_series.index[-1]
        benchmark_aligned = self.benchmark_prices.loc[start:end]

        metrics = calculate_all_metrics(equity_series, trades_df, benchmark_aligned)

        # Custom print for INR
        print_india_report(metrics, equity_series, self.config.initial_capital)

        # Save outputs
        print("\nSaving outputs...")

        equity_df = pd.DataFrame({
            'date': equity_series.index,
            'equity': equity_series.values,
            'benchmark': benchmark_aligned.values
        })
        equity_df.to_csv(f"{self.config.output_dir}/equity_curve.csv", index=False)
        print(f"  Saved: {self.config.output_dir}/equity_curve.csv")

        trades_df.to_csv(f"{self.config.output_dir}/trades.csv", index=False)
        print(f"  Saved: {self.config.output_dir}/trades.csv")

        save_metrics_json(metrics, f"{self.config.output_dir}/metrics.json")

        print("\nGenerating charts...")
        plot_equity_curve(equity_series, benchmark_aligned,
                          title="Momentum Strategy vs Nifty 50 (Log Scale)",
                          save_path=f"{self.config.output_dir}/equity_curve.png")

        plot_drawdown(equity_series, title="Strategy Drawdown",
                      save_path=f"{self.config.output_dir}/drawdown.png")

        plot_monthly_returns_heatmap(equity_series,
                                     save_path=f"{self.config.output_dir}/monthly_returns.png")

        plot_annual_returns(equity_series, benchmark_aligned,
                            save_path=f"{self.config.output_dir}/annual_returns.png")

        return metrics


def print_india_report(metrics: Dict, equity_series: pd.Series, initial_capital: float):
    """Print report formatted for Indian context."""
    print("\n" + "=" * 70)
    print(" MOMENTUM STRATEGY BACKTEST - INDIAN STOCKS (NSE)")
    print("=" * 70)

    final_value = equity_series.iloc[-1]
    multiple = final_value / initial_capital

    print(f"\n{'PORTFOLIO GROWTH':^70}")
    print("-" * 70)
    print(f"  Initial Investment:    Rs.{initial_capital/100000:>10.2f} Lakhs")
    print(f"  Final Value:           Rs.{final_value/100000:>10.2f} Lakhs")
    print(f"  Growth Multiple:       {multiple:>10.1f}x")
    print(f"  Total Return:          {metrics['total_return_pct']:>10.1f}%")

    print(f"\n{'PERFORMANCE METRICS':^70}")
    print("-" * 70)
    print(f"  CAGR:                  {metrics['cagr_pct']:>10.1f}%")
    print(f"  Max Drawdown:          {metrics['max_drawdown_pct']:>10.1f}%")
    print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}")
    print(f"  Volatility (Ann.):     {metrics['volatility_pct']:>10.1f}%")

    print(f"\n{'TRADE STATISTICS':^70}")
    print("-" * 70)
    print(f"  Total Trades:          {metrics['total_trades']:>10}")
    print(f"  Win Rate:              {metrics['win_rate_pct']:>10.1f}%")
    print(f"  Profit Factor:         {metrics['profit_factor']:>10.2f}")

    if 'benchmark_cagr_pct' in metrics:
        print(f"\n{'VS NIFTY 50 BENCHMARK':^70}")
        print("-" * 70)
        print(f"  Nifty 50 CAGR:         {metrics['benchmark_cagr_pct']:>10.1f}%")
        print(f"  Strategy Alpha:        {metrics['alpha_pct']:>10.1f}%")

    print("\n" + "=" * 70)


def project_future_growth(cagr: float, initial: float, target: float):
    """Project years to reach target and show milestones."""
    print("\n" + "=" * 70)
    print(" PROJECTION: Rs.30 LAKHS -> Rs.4 CRORES")
    print("=" * 70)

    years_needed = np.log(target / initial) / np.log(1 + cagr)

    print(f"\n  Based on backtest CAGR of {cagr*100:.1f}%:")
    print(f"  Years to reach Rs.4 Crores: {years_needed:.1f} years")

    print(f"\n{'YEAR-BY-YEAR PROJECTION':^70}")
    print("-" * 70)
    print(f"  {'Year':<8} {'Portfolio Value':<20} {'Milestone':<25}")
    print("-" * 70)

    value = initial
    for year in range(int(years_needed) + 2):
        value = initial * ((1 + cagr) ** year)

        if value >= 10000000:  # 1 Cr+
            value_str = f"Rs.{value/10000000:.2f} Cr"
        else:
            value_str = f"Rs.{value/100000:.2f} L"

        milestone = ""
        if year == 0:
            milestone = "Starting point"
        elif value >= 40000000 and initial * ((1 + cagr) ** (year-1)) < 40000000:
            milestone = "*** TARGET REACHED! ***"
        elif value >= 10000000 and initial * ((1 + cagr) ** (year-1)) < 10000000:
            milestone = "First Crore!"
        elif value >= 20000000 and initial * ((1 + cagr) ** (year-1)) < 20000000:
            milestone = "2 Crores!"
        elif value >= 30000000 and initial * ((1 + cagr) ** (year-1)) < 30000000:
            milestone = "3 Crores!"

        print(f"  {year:<8} {value_str:<20} {milestone:<25}")

        if value >= 50000000:  # Stop at 5 Cr
            break

    print("-" * 70)

    # Compare with different CAGR scenarios
    print(f"\n{'SENSITIVITY ANALYSIS':^70}")
    print("-" * 70)
    print(f"  {'CAGR':<12} {'Years to 4 Cr':<18} {'Risk Level':<20}")
    print("-" * 70)

    scenarios = [
        (0.12, "Low (Index funds)"),
        (0.15, "Moderate (Target)"),
        (0.20, "High (Active momentum)"),
        (0.25, "Very High (Aggressive)"),
        (cagr, f"Backtest result"),
    ]

    for rate, risk in scenarios:
        years = np.log(40000000 / 3000000) / np.log(1 + rate)
        marker = " <-- YOUR TARGET" if rate == 0.15 else ""
        if rate == cagr:
            marker = " <-- BACKTEST"
        print(f"  {rate*100:<12.1f}% {years:<18.1f} {risk:<20}{marker}")

    print("-" * 70)
    print("\n" + "=" * 70)


def print_monthly_stock_picks(trades_df: pd.DataFrame):
    """Show recent monthly stock picks."""
    if len(trades_df) == 0:
        return

    print("\n" + "=" * 70)
    print(" RECENT MONTHLY STOCK PICKS (Last 12 Months)")
    print("=" * 70)

    buys = trades_df[trades_df['action'] == 'BUY'].copy()
    buys['month'] = pd.to_datetime(buys['date']).dt.to_period('M')

    recent_months = buys['month'].unique()[-12:]

    print(f"\n  {'Month':<12} {'Stock 1':<15} {'Stock 2':<15} {'Stock 3':<15}")
    print("-" * 70)

    for month in recent_months:
        month_buys = buys[buys['month'] == month]['ticker'].tolist()
        stocks = month_buys + [''] * (3 - len(month_buys))
        print(f"  {str(month):<12} {stocks[0]:<15} {stocks[1]:<15} {stocks[2]:<15}")

    print("-" * 70)


def main():
    """Run Indian momentum strategy backtest."""
    print("\n" + "=" * 70)
    print(" INDIAN MOMENTUM STRATEGY")
    print(" Goal: Rs.30 Lakhs -> Rs.4 Crores")
    print("=" * 70)

    # Survivorship bias warning
    print("""
+--------------------------------------------------------------------+
|                    IMPORTANT DISCLAIMERS                           |
+--------------------------------------------------------------------+
| 1. SURVIVORSHIP BIAS: Universe selected with hindsight             |
| 2. PAST PERFORMANCE != FUTURE RETURNS                              |
| 3. Actual returns may be 30-50% lower than backtest                |
| 4. Requires discipline to follow during drawdowns                  |
| 5. Tax implications not included (LTCG @ 12.5%)                    |
+--------------------------------------------------------------------+
    """)

    # Configuration - WITH REGIME FILTER
    config = IndiaStrategyConfig(
        top_n=3,
        lookback_days=252,
        initial_capital=30_00_000,  # Rs.30 Lakhs
        transaction_cost_pct=0.001,
        start_date="2015-01-01",
        end_date="2024-12-31",
        use_regime_filter=True,  # Go to cash when Nifty < 200 SMA
        output_dir="output_india",
    )

    print(f"\nStrategy Configuration:")
    print(f"  Universe: {len(config.universe)} Indian stocks")
    print(f"  Top N: {config.top_n} stocks (equal weight)")
    print(f"  Lookback: {config.lookback_days} days (12 months)")
    print(f"  Initial Capital: Rs.{config.initial_capital/100000:.0f} Lakhs")
    print(f"  Transaction Cost: {config.transaction_cost_pct*100:.2f}%")

    # Run backtest
    engine = IndianMomentumBacktest(config)
    equity_series, trades_df = engine.run_backtest()

    # Generate reports
    metrics = engine.generate_reports(equity_series, trades_df)

    # Project future growth
    cagr = metrics['cagr_pct'] / 100
    project_future_growth(cagr, 30_00_000, 4_00_00_000)

    # Show recent picks
    print_monthly_stock_picks(trades_df)

    print("\n" + "=" * 70)
    print(" BACKTEST COMPLETE")
    print("=" * 70)
    print(f"\nOutput files: {config.output_dir}/")

    return equity_series, trades_df, metrics


if __name__ == "__main__":
    equity, trades, metrics = main()
