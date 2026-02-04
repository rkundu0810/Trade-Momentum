"""
Configuration for Momentum Strategy - Indian Stocks (NSE)
Target: 30 Lakhs -> 4 Crores with 15%+ CAGR

UPDATED: Using Nifty 50 + Nifty Next 50 (100 stocks) for more objective universe
"""
from dataclasses import dataclass, field
from typing import List


# Nifty 50 - Large Cap Blue Chips (as of Jan 2025)
NIFTY_50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "SHRIRAMFIN.NS",
]

# Nifty Next 50 - Large-Mid Cap (higher growth potential)
NIFTY_NEXT_50 = [
    "ABB.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS", "BOSCHLTD.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DLF.NS", "DABUR.NS",
    "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "IGL.NS", "IOC.NS", "INDIGO.NS",
    "JINDALSTEL.NS", "LICI.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS",
    "NAUKRI.NS", "NHPC.NS", "OFSS.NS", "PIDILITIND.NS", "PFC.NS",
    "PIIND.NS", "PNB.NS", "RECLTD.NS", "SIEMENS.NS", "SRF.NS",
    "SHREECEM.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS",
    "VBL.NS", "VEDL.NS", "ZYDUSLIFE.NS", "MAXHEALTH.NS",
]

# Combined Nifty 100 Universe (Nifty 50 + Next 50)
NIFTY_100_UNIVERSE = list(set(NIFTY_50 + NIFTY_NEXT_50))


@dataclass
class IndiaStrategyConfig:
    """Configuration for Indian momentum strategy."""

    # OBJECTIVE Universe: Nifty 50 + Nifty Next 50 (~100 stocks)
    # This is more systematic than manual selection
    # Still has survivorship bias (using current members, not historical)
    universe: List[str] = field(default_factory=lambda: NIFTY_100_UNIVERSE)

    # Strategy parameters
    top_n: int = 3                          # Hold top 3 momentum stocks
    lookback_days: int = 252                # 12-month momentum
    rebalance_frequency: str = "monthly"

    # Capital (30 Lakhs in INR)
    initial_capital: float = 30_00_000.0    # Rs.30,00,000
    transaction_cost_pct: float = 0.001     # 0.1% (brokerage + STT + charges)

    # Backtest period (10 years)
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # REGIME FILTER: Go to cash when Nifty 50 < 200 SMA
    use_regime_filter: bool = True          # ENABLED
    regime_benchmark: str = "^NSEI"         # Nifty 50 index
    regime_sma_period: int = 200

    # Benchmark
    benchmark_ticker: str = "^NSEI"         # Nifty 50

    # Output
    output_dir: str = "output_india"
    save_trades_csv: bool = True
    save_equity_csv: bool = True
    save_charts: bool = True


# Alternative Universe: Quality Midcaps (Higher Risk/Higher Reward)
QUALITY_MIDCAP_UNIVERSE = [
    # Large Cap Anchors (10)
    "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS",
    "BHARTIARTL.NS", "LT.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS",

    # Quality Midcaps - IT (5)
    "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "TATAELXSI.NS", "LTTS.NS",

    # Quality Midcaps - Consumer (5)
    "TRENT.NS", "POLYCAB.NS", "ASTRAL.NS", "PAGEIND.NS", "VOLTAS.NS",

    # Quality Midcaps - Finance (5)
    "CHOLAFIN.NS", "MUTHOOTFIN.NS", "ICICIGI.NS", "HDFCLIFE.NS", "SBILIFE.NS",

    # Quality Midcaps - Industrial/Others (5)
    "PIIND.NS", "DEEPAKNTR.NS", "HAL.NS", "BEL.NS", "IRCTC.NS",
]


# Conservative Universe: Nifty 50 only
NIFTY50_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
    "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "TECHM.NS", "M&M.NS", "POWERGRID.NS", "NTPC.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "ONGC.NS", "COALINDIA.NS",
]
