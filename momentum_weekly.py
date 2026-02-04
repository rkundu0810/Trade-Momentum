"""
Weekly Momentum Strategy - Backtest & Alert System

Features:
- 15 stock focused universe
- Weekly rebalancing (every Monday)
- Top 3 stocks by 12-month momentum
- Regime filter (Nifty > 200 SMA)
- Can run standalone for weekly alerts
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import os
import json

from config_weekly import WeeklyStrategyConfig, MOMENTUM_15_UNIVERSE
from utils import (
    calculate_all_metrics,
    save_metrics_json,
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns_heatmap,
    plot_annual_returns,
    ensure_output_dir,
)

warnings.filterwarnings('ignore')


class WeeklyMomentumStrategy:
    """Weekly momentum strategy with 15-stock universe."""

    def __init__(self, config: WeeklyStrategyConfig):
        self.config = config
        self.prices = None
        self.benchmark_prices = None

    def fetch_data(self, start_date=None, end_date=None) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch price data from Yahoo Finance."""
        if start_date is None:
            start_date = self.config.start_date
        if end_date is None:
            end_date = self.config.end_date

        buffer_days = self.config.lookback_days + 30
        start_with_buffer = pd.to_datetime(start_date) - timedelta(days=buffer_days * 1.5)

        print(f"\nFetching data...")
        print(f"  Universe: {len(self.config.universe)} stocks")

        tickers = self.config.universe + [self.config.benchmark_ticker]

        data = yf.download(
            tickers=tickers,
            start=start_with_buffer,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )

        if len(tickers) == 1:
            prices = data['Close'].to_frame(name=tickers[0])
        else:
            prices = data['Close']

        benchmark_prices = prices[self.config.benchmark_ticker].copy()
        universe_cols = [t for t in self.config.universe if t in prices.columns]
        universe_prices = prices[universe_cols].copy()

        universe_prices = universe_prices.ffill()
        benchmark_prices = benchmark_prices.ffill()

        print(f"  Stocks with data: {len(universe_cols)} / {len(self.config.universe)}")

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
        """Check if market is in risk-on mode (Nifty > 200 SMA)."""
        if not self.config.use_regime_filter:
            return True

        prices_to_date = self.benchmark_prices.loc[:date]

        if len(prices_to_date) < self.config.regime_sma_period:
            return True

        sma_200 = prices_to_date.iloc[-self.config.regime_sma_period:].mean()
        current_price = prices_to_date.iloc[-1]

        return current_price > sma_200

    def get_weekly_rebalance_dates(self, trading_days: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Get Monday of each week (or first trading day if Monday is holiday)."""
        rebalance_dates = []

        # Group by year and week number
        weekly_groups = trading_days.to_series().groupby(
            [trading_days.isocalendar().year, trading_days.isocalendar().week]
        )

        for _, group in weekly_groups:
            # Take first trading day of each week
            rebalance_dates.append(group.iloc[0])

        return rebalance_dates

    def run_backtest(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Run the weekly momentum backtest."""
        self.fetch_data()

        cash = self.config.initial_capital
        holdings = {}
        entry_prices = {}
        equity_history = []
        trades = []
        cash_weeks = 0

        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        trading_days = self.prices.index[(self.prices.index >= start) & (self.prices.index <= end)]

        # Get weekly rebalance dates
        rebalance_dates = self.get_weekly_rebalance_dates(trading_days)
        rebalance_set = set(rebalance_dates)

        print(f"\nRunning weekly backtest...")
        print(f"  Initial capital: Rs.{cash:,.0f}")
        print(f"  Rebalance weeks: {len(rebalance_dates)}")
        print(f"  Regime filter: {'ON' if self.config.use_regime_filter else 'OFF'}")

        current_holdings_set = set()

        for date in trading_days:
            if date in rebalance_set:
                risk_on = self.is_risk_on(date)

                if risk_on:
                    new_stocks = self.get_top_stocks(date)
                else:
                    new_stocks = []
                    cash_weeks += 1

                new_stocks_set = set(new_stocks)

                # SMART REBALANCE: Only trade if holdings changed
                if new_stocks_set == current_holdings_set:
                    # No change needed, skip trading
                    pass
                else:
                    prices_today = self.prices.loc[date]

                    # Sell current holdings
                    for ticker, shares in list(holdings.items()):
                    if ticker in prices_today and not pd.isna(prices_today[ticker]):
                        sell_price = prices_today[ticker]
                        sell_value = shares * sell_price
                        cost = sell_value * self.config.transaction_cost_pct
                        pnl = (sell_price - entry_prices.get(ticker, sell_price)) * shares - cost

                        trades.append({
                            'date': date,
                            'ticker': ticker.replace('.NS', ''),
                            'action': 'SELL',
                            'shares': round(shares, 2),
                            'price': round(sell_price, 2),
                            'value': round(sell_value, 2),
                            'cost': round(cost, 2),
                            'pnl': round(pnl, 2)
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
                                'date': date,
                                'ticker': ticker.replace('.NS', ''),
                                'action': 'BUY',
                                'shares': round(shares, 2),
                                'price': round(buy_price, 2),
                                'value': round(actual_value, 2),
                                'cost': round(cost, 2),
                                'pnl': 0
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
            print(f"  Weeks in cash: {cash_weeks}")

        self.cash_weeks = cash_weeks
        return equity_series, trades_df

    def generate_reports(self, equity_series: pd.Series, trades_df: pd.DataFrame) -> Dict:
        """Generate reports and charts."""
        ensure_output_dir(self.config.output_dir)

        start = equity_series.index[0]
        end = equity_series.index[-1]
        benchmark_aligned = self.benchmark_prices.loc[start:end]

        metrics = calculate_all_metrics(equity_series, trades_df, benchmark_aligned)

        # Print report
        self._print_report(metrics, equity_series)

        # Save files
        print("\nSaving outputs...")

        equity_df = pd.DataFrame({
            'date': equity_series.index,
            'equity': equity_series.values,
        })
        equity_df.to_csv(f"{self.config.output_dir}/equity_curve.csv", index=False)

        trades_df.to_csv(f"{self.config.output_dir}/trades.csv", index=False)
        save_metrics_json(metrics, f"{self.config.output_dir}/metrics.json")

        print("\nGenerating charts...")
        plot_equity_curve(equity_series, benchmark_aligned,
                          title="Weekly Momentum vs Nifty 50",
                          save_path=f"{self.config.output_dir}/equity_curve.png")

        plot_drawdown(equity_series,
                      save_path=f"{self.config.output_dir}/drawdown.png")

        plot_monthly_returns_heatmap(equity_series,
                                     save_path=f"{self.config.output_dir}/monthly_returns.png")

        return metrics

    def _print_report(self, metrics: Dict, equity_series: pd.Series):
        """Print formatted report."""
        print("\n" + "=" * 70)
        print(" WEEKLY MOMENTUM STRATEGY - BACKTEST RESULTS")
        print("=" * 70)

        final = equity_series.iloc[-1]
        initial = self.config.initial_capital
        multiple = final / initial

        print(f"\n  Initial:     Rs.{initial/100000:>8.2f} Lakhs")
        print(f"  Final:       Rs.{final/100000:>8.2f} Lakhs")
        print(f"  Multiple:    {multiple:>8.1f}x")
        print(f"  CAGR:        {metrics['cagr_pct']:>8.1f}%")
        print(f"  Max DD:      {metrics['max_drawdown_pct']:>8.1f}%")
        print(f"  Sharpe:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Trades:      {metrics['total_trades']:>8}")
        print(f"  Win Rate:    {metrics['win_rate_pct']:>8.1f}%")

        if 'benchmark_cagr_pct' in metrics:
            print(f"\n  Nifty CAGR:  {metrics['benchmark_cagr_pct']:>8.1f}%")
            print(f"  Alpha:       {metrics['alpha_pct']:>8.1f}%")

        print("\n" + "=" * 70)


def get_weekly_alert(capital: float = 30_00_000) -> Dict:
    """
    Get this week's momentum picks.
    Run this every Monday to know what to buy/sell.
    """
    print("\n" + "=" * 70)
    print(" WEEKLY MOMENTUM ALERT")
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    config = WeeklyStrategyConfig(initial_capital=capital)
    strategy = WeeklyMomentumStrategy(config)

    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)

    strategy.fetch_data(start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'))

    # Get latest trading day
    latest_date = strategy.prices.index[-1]
    print(f"\n  Latest data: {latest_date.strftime('%Y-%m-%d')}")

    # Check regime
    risk_on = strategy.is_risk_on(latest_date)
    nifty_price = strategy.benchmark_prices.loc[latest_date]
    nifty_sma = strategy.benchmark_prices.iloc[-200:].mean()

    print(f"\n  Nifty 50:    {nifty_price:,.0f}")
    print(f"  200-day SMA: {nifty_sma:,.0f}")
    print(f"  Regime:      {'RISK ON (Invest)' if risk_on else 'RISK OFF (Cash)'}")

    if not risk_on:
        print("\n" + "-" * 70)
        print("  ACTION: GO TO CASH - Nifty below 200 SMA")
        print("  Sell all holdings and wait for regime to turn positive")
        print("-" * 70)
        return {'action': 'CASH', 'stocks': [], 'regime': 'OFF'}

    # Calculate momentum for all stocks
    momentum = strategy.calculate_momentum(latest_date)
    momentum_sorted = momentum.sort_values(ascending=False)

    print(f"\n{'='*70}")
    print(" MOMENTUM RANKINGS (12-Month Returns)")
    print("=" * 70)
    print(f"\n  {'Rank':<6} {'Stock':<15} {'Return':<12} {'Action'}")
    print("-" * 70)

    top_stocks = []
    for i, (ticker, ret) in enumerate(momentum_sorted.items(), 1):
        clean_ticker = ticker.replace('.NS', '')
        action = "*** BUY ***" if i <= config.top_n else ""
        print(f"  {i:<6} {clean_ticker:<15} {ret*100:>8.1f}%    {action}")

        if i <= config.top_n:
            current_price = strategy.prices.loc[latest_date, ticker]
            top_stocks.append({
                'rank': i,
                'ticker': clean_ticker,
                'ticker_ns': ticker,
                'momentum_pct': round(ret * 100, 2),
                'current_price': round(current_price, 2),
            })

    print("-" * 70)

    # Calculate position sizes
    print(f"\n{'='*70}")
    print(" THIS WEEK'S TRADES")
    print("=" * 70)

    position_value = capital / len(top_stocks)

    print(f"\n  Capital: Rs.{capital/100000:.2f} Lakhs")
    print(f"  Positions: {len(top_stocks)} stocks @ Rs.{position_value/100000:.2f} Lakhs each")
    print(f"\n  {'Stock':<12} {'Price':<12} {'Qty':<10} {'Value':<15}")
    print("-" * 70)

    for stock in top_stocks:
        qty = int(position_value / stock['current_price'])
        value = qty * stock['current_price']
        stock['quantity'] = qty
        stock['position_value'] = value
        print(f"  {stock['ticker']:<12} Rs.{stock['current_price']:<10,.0f} {qty:<10} Rs.{value:>12,.0f}")

    print("-" * 70)
    total_invested = sum(s['position_value'] for s in top_stocks)
    print(f"  {'TOTAL':<12} {'':<12} {'':<10} Rs.{total_invested:>12,.0f}")
    print(f"  {'Cash Left':<12} {'':<12} {'':<10} Rs.{capital - total_invested:>12,.0f}")

    # Save alert to JSON
    alert = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'data_date': latest_date.strftime('%Y-%m-%d'),
        'regime': 'ON' if risk_on else 'OFF',
        'nifty_price': float(nifty_price),
        'nifty_sma_200': float(nifty_sma),
        'capital': capital,
        'stocks': top_stocks,
        'action': 'BUY' if risk_on else 'CASH',
    }

    ensure_output_dir('alerts')
    alert_file = f"alerts/weekly_alert_{datetime.now().strftime('%Y%m%d')}.json"
    with open(alert_file, 'w') as f:
        json.dump(alert, f, indent=2)
    print(f"\n  Alert saved: {alert_file}")

    print("\n" + "=" * 70)

    return alert


def run_backtest():
    """Run full backtest."""
    print("\n" + "=" * 70)
    print(" WEEKLY MOMENTUM STRATEGY BACKTEST")
    print(" 15 Stock Universe | Weekly Rebalancing")
    print("=" * 70)

    config = WeeklyStrategyConfig()

    print(f"\nUniverse ({len(config.universe)} stocks):")
    for i, ticker in enumerate(config.universe, 1):
        print(f"  {i:2}. {ticker.replace('.NS', '')}")

    strategy = WeeklyMomentumStrategy(config)
    equity_series, trades_df = strategy.run_backtest()
    metrics = strategy.generate_reports(equity_series, trades_df)

    # Projection
    cagr = metrics['cagr_pct'] / 100
    initial = config.initial_capital
    target = 4_00_00_000

    years = np.log(target / initial) / np.log(1 + cagr)

    print(f"\n{'='*70}")
    print(" PROJECTION: Rs.30 Lakhs -> Rs.4 Crores")
    print(f"{'='*70}")
    print(f"\n  At {cagr*100:.1f}% CAGR: {years:.1f} years to reach target")

    print("\n" + "=" * 70)
    print(" BACKTEST COMPLETE")
    print("=" * 70)

    return equity_series, trades_df, metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--alert':
        # Run weekly alert only
        capital = float(sys.argv[2]) if len(sys.argv) > 2 else 30_00_000
        get_weekly_alert(capital)
    else:
        # Run full backtest
        run_backtest()
        print("\n  To get weekly alerts, run:")
        print("  python momentum_weekly.py --alert")
        print("  python momentum_weekly.py --alert 5000000  # with custom capital")
