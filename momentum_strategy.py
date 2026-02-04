"""
Momentum Strategy Backtest Engine

This implements a momentum strategy that:
- Selects top N stocks by 12-month trailing return
- Rebalances monthly (first trading day)
- Equal-weights positions
- Includes optional market regime filter (SPY > 200 SMA)

WARNING: This backtest uses a fixed universe of stocks that were selected with
the benefit of hindsight (survivorship bias). The results are NOT indicative
of what could have been achieved in real-time.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

from config import StrategyConfig, DEFAULT_CONFIG
from utils import (
    calculate_all_metrics,
    print_metrics_report,
    save_metrics_json,
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns_heatmap,
    plot_annual_returns,
    ensure_output_dir,
)

warnings.filterwarnings('ignore')


class DataFetcher:
    """Fetches and prepares price data from Yahoo Finance."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.price_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.Series] = None

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fetch adjusted close prices for universe and benchmark.

        Returns:
            Tuple of (universe_prices DataFrame, benchmark_prices Series)
        """
        # Add buffer for lookback period
        buffer_days = self.config.lookback_days + 30
        start_with_buffer = pd.to_datetime(self.config.start_date) - timedelta(days=buffer_days * 1.5)

        print(f"\nFetching data from Yahoo Finance...")
        print(f"  Universe: {len(self.config.universe)} stocks")
        print(f"  Period: {start_with_buffer.date()} to {self.config.end_date}")

        # Fetch universe data
        tickers = self.config.universe + [self.config.benchmark_ticker]
        if self.config.use_regime_filter and self.config.regime_benchmark not in tickers:
            tickers.append(self.config.regime_benchmark)

        data = yf.download(
            tickers=tickers,
            start=start_with_buffer,
            end=self.config.end_date,
            progress=False,
            auto_adjust=True,  # Use adjusted prices
        )

        # Handle single vs multiple tickers
        if len(tickers) == 1:
            prices = data['Close'].to_frame(name=tickers[0])
        else:
            prices = data['Close']

        # Separate benchmark
        benchmark_prices = prices[self.config.benchmark_ticker].copy()
        universe_prices = prices[self.config.universe].copy()

        # Forward fill then drop remaining NaNs
        universe_prices = universe_prices.ffill()
        benchmark_prices = benchmark_prices.ffill()

        print(f"  Fetched {len(universe_prices)} trading days")
        print(f"  Stocks with data: {universe_prices.notna().any().sum()} / {len(self.config.universe)}")

        self.price_data = universe_prices
        self.benchmark_data = benchmark_prices

        return universe_prices, benchmark_prices


class MomentumCalculator:
    """Calculates momentum scores and rankings."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def calculate_momentum(self, prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """
        Calculate 12-month momentum for all stocks on a given date.

        Args:
            prices: DataFrame of adjusted close prices
            date: Date to calculate momentum for

        Returns:
            Series of momentum scores (returns) indexed by ticker
        """
        lookback = self.config.lookback_days

        # Get data up to this date
        prices_to_date = prices.loc[:date]

        if len(prices_to_date) < lookback:
            return pd.Series(dtype=float)

        # Current price and price lookback_days ago
        current_prices = prices_to_date.iloc[-1]
        past_prices = prices_to_date.iloc[-lookback]

        # Calculate momentum (simple return)
        momentum = (current_prices / past_prices) - 1

        # Remove stocks with NaN momentum
        momentum = momentum.dropna()

        return momentum

    def get_top_stocks(self, prices: pd.DataFrame, date: pd.Timestamp) -> List[str]:
        """
        Get top N stocks by momentum on a given date.

        Returns:
            List of ticker symbols for top momentum stocks
        """
        momentum = self.calculate_momentum(prices, date)

        if len(momentum) == 0:
            return []

        # Sort descending and take top N
        top_momentum = momentum.sort_values(ascending=False).head(self.config.top_n)

        return top_momentum.index.tolist()


class MarketRegimeFilter:
    """Optional market regime filter based on SPY vs 200 SMA."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.sma_cache: Dict[pd.Timestamp, float] = {}

    def is_risk_on(self, benchmark_prices: pd.Series, date: pd.Timestamp) -> bool:
        """
        Check if market is in risk-on mode (benchmark > 200 SMA).

        Returns:
            True if should be invested, False if should go to cash
        """
        if not self.config.use_regime_filter:
            return True

        prices_to_date = benchmark_prices.loc[:date]

        if len(prices_to_date) < self.config.regime_sma_period:
            return True  # Default to invested if not enough data

        sma = prices_to_date.iloc[-self.config.regime_sma_period:].mean()
        current_price = prices_to_date.iloc[-1]

        return current_price > sma


class BacktestEngine:
    """Main backtest simulation engine."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.momentum_calc = MomentumCalculator(config)
        self.regime_filter = MarketRegimeFilter(config)

        # State variables
        self.prices: Optional[pd.DataFrame] = None
        self.benchmark_prices: Optional[pd.Series] = None
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.trades: List[Dict] = []
        self.holdings: Dict[str, float] = {}  # ticker -> shares

    def get_rebalance_dates(self, prices: pd.DataFrame) -> List[pd.Timestamp]:
        """
        Get first trading day of each month for rebalancing.

        Returns:
            List of rebalance dates
        """
        # Filter to backtest period
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)

        dates_in_range = prices.index[(prices.index >= start) & (prices.index <= end)]

        if len(dates_in_range) == 0:
            return []

        # Group by year-month and take first date of each group
        monthly_groups = dates_in_range.to_series().groupby(
            [dates_in_range.year, dates_in_range.month]
        )

        rebalance_dates = [group.iloc[0] for _, group in monthly_groups]

        return rebalance_dates

    def get_portfolio_value(self, date: pd.Timestamp) -> float:
        """Calculate current portfolio value based on holdings."""
        if not self.holdings:
            return self.cash

        prices_on_date = self.prices.loc[date]
        holdings_value = sum(
            shares * prices_on_date.get(ticker, 0)
            for ticker, shares in self.holdings.items()
            if not pd.isna(prices_on_date.get(ticker, np.nan))
        )

        return self.cash + holdings_value

    def rebalance_portfolio(self, date: pd.Timestamp, new_holdings: List[str]) -> None:
        """
        Rebalance portfolio to new holdings with equal weights.

        Args:
            date: Rebalance date
            new_holdings: List of tickers to hold
        """
        prices_on_date = self.prices.loc[date]
        current_value = self.get_portfolio_value(date)

        # Sell current holdings
        for ticker, shares in list(self.holdings.items()):
            if ticker in prices_on_date and not pd.isna(prices_on_date[ticker]):
                sell_price = prices_on_date[ticker]
                sell_value = shares * sell_price

                # Transaction cost
                cost = sell_value * self.config.transaction_cost_pct
                proceeds = sell_value - cost

                # Record trade
                buy_price = self.entry_prices.get(ticker, sell_price)
                pnl = (sell_price - buy_price) * shares - cost

                self.trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': sell_price,
                    'value': sell_value,
                    'cost': cost,
                    'pnl': pnl,
                })

                self.cash += proceeds

        # Clear holdings
        self.holdings = {}
        self.entry_prices = {}

        # Buy new holdings with equal weight
        if len(new_holdings) > 0:
            available_cash = self.cash
            weight = 1.0 / len(new_holdings)

            for ticker in new_holdings:
                if ticker in prices_on_date and not pd.isna(prices_on_date[ticker]):
                    target_value = available_cash * weight
                    buy_price = prices_on_date[ticker]

                    # Account for transaction cost
                    effective_value = target_value / (1 + self.config.transaction_cost_pct)
                    shares = effective_value / buy_price
                    actual_value = shares * buy_price
                    cost = actual_value * self.config.transaction_cost_pct

                    self.holdings[ticker] = shares
                    self.entry_prices[ticker] = buy_price
                    self.cash -= (actual_value + cost)

                    self.trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares,
                        'price': buy_price,
                        'value': actual_value,
                        'cost': cost,
                        'pnl': 0,  # PnL calculated on sell
                    })

    def run_backtest(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run the momentum backtest simulation.

        Returns:
            Tuple of (equity_series, trades_dataframe)
        """
        # Fetch data
        self.prices, self.benchmark_prices = self.data_fetcher.fetch_data()

        # Initialize state
        self.cash = self.config.initial_capital
        self.holdings = {}
        self.entry_prices = {}
        self.equity_history = []
        self.trades = []

        # Get rebalance dates
        rebalance_dates = self.get_rebalance_dates(self.prices)
        print(f"\nRunning backtest...")
        print(f"  Rebalance dates: {len(rebalance_dates)}")

        # Get all trading days in backtest period
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        trading_days = self.prices.index[(self.prices.index >= start) & (self.prices.index <= end)]

        rebalance_set = set(rebalance_dates)
        current_holdings = []

        # Main simulation loop
        for date in trading_days:
            # Check if rebalance day
            if date in rebalance_set:
                # Check regime filter
                if self.regime_filter.is_risk_on(self.benchmark_prices, date):
                    # Get top momentum stocks
                    new_holdings = self.momentum_calc.get_top_stocks(self.prices, date)
                else:
                    # Risk-off: go to cash
                    new_holdings = []

                # Only rebalance if holdings changed
                if set(new_holdings) != set(self.holdings.keys()):
                    self.rebalance_portfolio(date, new_holdings)
                    current_holdings = new_holdings

            # Record daily equity
            equity = self.get_portfolio_value(date)
            self.equity_history.append((date, equity))

        # Convert to Series and DataFrame
        equity_series = pd.Series(
            dict(self.equity_history),
            name='equity'
        )
        trades_df = pd.DataFrame(self.trades)

        print(f"  Final equity: ${equity_series.iloc[-1]:,.2f}")
        print(f"  Total trades: {len(trades_df)}")

        return equity_series, trades_df

    def generate_reports(self, equity_series: pd.Series, trades_df: pd.DataFrame) -> Dict:
        """Generate all reports and charts."""
        ensure_output_dir(self.config.output_dir)

        # Align benchmark to backtest period
        start = equity_series.index[0]
        end = equity_series.index[-1]
        benchmark_aligned = self.benchmark_prices.loc[start:end]

        # Calculate metrics
        print("\nCalculating performance metrics...")
        metrics = calculate_all_metrics(equity_series, trades_df, benchmark_aligned)

        # Print report
        print_metrics_report(metrics, "MOMENTUM STRATEGY BACKTEST RESULTS")

        # Save outputs
        print("\nSaving outputs...")

        if self.config.save_equity_csv:
            equity_path = f"{self.config.output_dir}/equity_curve.csv"
            equity_df = pd.DataFrame({
                'date': equity_series.index,
                'equity': equity_series.values,
                'benchmark': benchmark_aligned.values
            })
            equity_df.to_csv(equity_path, index=False)
            print(f"  Saved: {equity_path}")

        if self.config.save_trades_csv:
            trades_path = f"{self.config.output_dir}/trades.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"  Saved: {trades_path}")

        # Save metrics JSON
        metrics_path = f"{self.config.output_dir}/metrics.json"
        save_metrics_json(metrics, metrics_path)

        if self.config.save_charts:
            print("\nGenerating charts...")

            plot_equity_curve(
                equity_series, benchmark_aligned,
                title="Momentum Strategy vs SPY (Log Scale)",
                log_scale=True,
                save_path=f"{self.config.output_dir}/equity_curve.png"
            )

            plot_drawdown(
                equity_series,
                title="Strategy Drawdown",
                save_path=f"{self.config.output_dir}/drawdown.png"
            )

            plot_monthly_returns_heatmap(
                equity_series,
                title="Monthly Returns Heatmap (%)",
                save_path=f"{self.config.output_dir}/monthly_returns.png"
            )

            plot_annual_returns(
                equity_series, benchmark_aligned,
                title="Annual Returns: Strategy vs SPY",
                save_path=f"{self.config.output_dir}/annual_returns.png"
            )

        return metrics


def print_survivorship_bias_warning():
    """Print warning about survivorship bias."""
    warning = """
================================================================================
                         SURVIVORSHIP BIAS WARNING
================================================================================

This backtest uses a FIXED universe of 30 stocks that were selected with the
benefit of HINDSIGHT. Many of these stocks (TSLA, NVDA, etc.) were chosen
because they are known to have performed exceptionally well.

LIMITATIONS:
  1. Stocks that went bankrupt, were delisted, or underperformed are excluded
  2. IPO dates are not enforced (e.g., COIN IPO'd in 2021)
  3. This is NOT a valid test of the momentum strategy
  4. Real-world results would likely be significantly lower

For a proper momentum backtest, use:
  - An index (e.g., S&P 500 constituents) with historical membership
  - Point-in-time data that respects IPO/delisting dates
  - A survivorship-bias-free database

================================================================================
"""
    print(warning)


def main():
    """Main entry point for the momentum strategy backtest."""
    print("\n" + "=" * 60)
    print(" MOMENTUM STRATEGY BACKTEST")
    print("=" * 60)

    # Print survivorship bias warning
    print_survivorship_bias_warning()

    # Create configuration
    config = StrategyConfig(
        # Can customize parameters here
        top_n=3,
        lookback_days=252,
        initial_capital=100_000,
        transaction_cost_pct=0.001,
        start_date="2015-01-01",
        end_date="2024-12-31",
        use_regime_filter=False,
        output_dir="output",
    )

    print(f"\nStrategy Configuration:")
    print(f"  Universe size: {len(config.universe)} stocks")
    print(f"  Top N positions: {config.top_n}")
    print(f"  Lookback period: {config.lookback_days} days (~12 months)")
    print(f"  Transaction cost: {config.transaction_cost_pct * 100:.2f}%")
    print(f"  Initial capital: ${config.initial_capital:,.2f}")
    print(f"  Market regime filter: {'ON' if config.use_regime_filter else 'OFF'}")

    # Run backtest
    engine = BacktestEngine(config)
    equity_series, trades_df = engine.run_backtest()

    # Generate reports
    metrics = engine.generate_reports(equity_series, trades_df)

    print("\n" + "=" * 60)
    print(" BACKTEST COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {config.output_dir}/")
    print("  - equity_curve.csv")
    print("  - trades.csv")
    print("  - metrics.json")
    print("  - equity_curve.png")
    print("  - drawdown.png")
    print("  - monthly_returns.png")
    print("  - annual_returns.png")

    return equity_series, trades_df, metrics


if __name__ == "__main__":
    equity, trades, metrics = main()
