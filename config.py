"""
Configuration for Momentum Strategy Backtest
"""
from dataclasses import dataclass, field
from typing import List
from datetime import date


@dataclass
class StrategyConfig:
    """Configuration parameters for the momentum strategy."""

    # Universe of stocks (30 tech-heavy stocks from the original backtest)
    universe: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "ADBE",
        "CRM", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "AVGO", "NOW", "SHOP",
        "SQ", "PYPL", "UBER", "ABNB", "SNOW", "PLTR", "COIN", "DDOG", "ZS", "CRWD"
    ])

    # Strategy parameters
    top_n: int = 3                          # Number of stocks to hold
    lookback_days: int = 252                # 12-month momentum (trading days)
    rebalance_frequency: str = "monthly"    # Rebalance on first trading day of month

    # Capital and costs
    initial_capital: float = 100_000.0      # Starting capital
    transaction_cost_pct: float = 0.001     # 0.1% per trade (10 bps)

    # Backtest period
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # Market regime filter (optional)
    use_regime_filter: bool = False         # Go to cash when SPY < 200 SMA
    regime_benchmark: str = "SPY"
    regime_sma_period: int = 200

    # Benchmark for comparison
    benchmark_ticker: str = "SPY"

    # Output settings
    output_dir: str = "output"
    save_trades_csv: bool = True
    save_equity_csv: bool = True
    save_charts: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.top_n > len(self.universe):
            raise ValueError(f"top_n ({self.top_n}) cannot exceed universe size ({len(self.universe)})")
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive")
        if self.transaction_cost_pct < 0:
            raise ValueError("transaction_cost_pct cannot be negative")


# Default configuration instance
DEFAULT_CONFIG = StrategyConfig()
