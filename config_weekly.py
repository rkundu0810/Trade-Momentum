"""
Configuration for Weekly Momentum Strategy
- 15 stock universe (focused, high liquidity)
- Weekly rebalancing (every Monday)
- Top 3 stocks by momentum
"""
from dataclasses import dataclass, field
from typing import List


# Focused universe of 15 high-momentum potential stocks
# Selection criteria:
# 1. High liquidity (Nifty 50/Next 50 members)
# 2. Historically strong momentum characteristics
# 3. Sector diversification
# 4. No PSU banks (too volatile, governance issues)

MOMENTUM_15_UNIVERSE = [
    # Finance (3) - High beta, momentum leaders
    "BAJFINANCE.NS",      # Bajaj Finance - retail lending leader
    "SHRIRAMFIN.NS",      # Shriram Finance - vehicle financing
    "CHOLAFIN.NS",        # Chola Finance - strong momentum

    # IT (2) - Global tech exposure
    "INFY.NS",            # Infosys - stable large cap
    "LTIM.NS",            # LTI Mindtree - mid-cap IT momentum

    # Consumer/Retail (2) - Domestic consumption
    "TITAN.NS",           # Titan - jewelry & watches
    "TRENT.NS",           # Trent - Zudio retail momentum

    # Auto (2) - Cyclical momentum
    "M&M.NS",             # Mahindra - SUV + EV + farm
    "TVSMOTOR.NS",        # TVS Motor - 2-wheeler momentum

    # Industrial/Infra (2) - Capex cycle
    "LT.NS",              # L&T - infrastructure bellwether
    "ADANIPORTS.NS",      # Adani Ports - logistics momentum

    # Metals (1) - Commodity cycle
    "HINDALCO.NS",        # Hindalco - aluminum + Novelis

    # Pharma (1) - Defensive with momentum
    "SUNPHARMA.NS",       # Sun Pharma - specialty drugs

    # Power/Energy (1) - New theme
    "TATAPOWER.NS",       # Tata Power - renewable energy

    # Defence (1) - Govt spending theme
    "HAL.NS",             # HAL - defence momentum
]


@dataclass
class WeeklyStrategyConfig:
    """Configuration for weekly momentum strategy."""

    # Focused 15-stock universe
    universe: List[str] = field(default_factory=lambda: MOMENTUM_15_UNIVERSE)

    # Strategy parameters
    top_n: int = 3                          # Hold top 3 momentum stocks
    lookback_days: int = 63                 # 3-month momentum (better for weekly)
    rebalance_frequency: str = "weekly"     # Every Monday

    # Capital
    initial_capital: float = 30_00_000.0    # Rs.30 Lakhs
    transaction_cost_pct: float = 0.0005    # 0.05% (use discount broker)

    # Backtest period
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # Regime filter
    use_regime_filter: bool = True          # Go to cash when Nifty < 200 SMA
    regime_benchmark: str = "^NSEI"
    regime_sma_period: int = 200

    # Benchmark
    benchmark_ticker: str = "^NSEI"

    # Output
    output_dir: str = "output_weekly"
    save_trades_csv: bool = True
    save_equity_csv: bool = True
    save_charts: bool = True


# Alternative: More aggressive 15 stocks (higher risk)
AGGRESSIVE_15_UNIVERSE = [
    "BAJFINANCE.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS",  # Finance
    "TRENT.NS", "TITAN.NS",                            # Consumer
    "M&M.NS", "TVSMOTOR.NS",                           # Auto
    "HAL.NS", "BEL.NS",                                # Defence
    "TATAPOWER.NS", "ADANIPORTS.NS",                   # Infra
    "HINDALCO.NS", "VEDL.NS",                          # Metals
    "RECLTD.NS", "PFC.NS",                             # Power Finance
]

# Conservative: Large cap only
CONSERVATIVE_15_UNIVERSE = [
    "HDFCBANK.NS", "ICICIBANK.NS", "BAJFINANCE.NS",   # Banks
    "TCS.NS", "INFY.NS",                               # IT
    "RELIANCE.NS", "LT.NS",                            # Industrial
    "TITAN.NS", "HINDUNILVR.NS",                       # Consumer
    "SUNPHARMA.NS", "DRREDDY.NS",                      # Pharma
    "M&M.NS", "MARUTI.NS",                             # Auto
    "BHARTIARTL.NS", "POWERGRID.NS",                   # Telecom/Utility
]
