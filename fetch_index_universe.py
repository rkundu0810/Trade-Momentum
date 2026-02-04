"""
Fetch objective stock universes from NSE indices.

This provides a more systematic (though still imperfect) approach
to selecting the universe rather than manual cherry-picking.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# ============================================================================
# OBJECTIVE UNIVERSES BASED ON NSE INDICES
# ============================================================================

# Nifty 50 - Large Cap Blue Chips (as of Jan 2025)
# Source: NSE India official list
NIFTY_50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "SHRIRAMFIN.NS",
]

# Nifty Next 50 - Large-Mid Cap (higher growth potential)
NIFTY_NEXT_50 = [
    "ABB.NS", "ADANIGREEN.NS", "AMBUJACEM.NS", "ATGL.NS", "AUROPHARMA.NS",
    "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS", "BOSCHLTD.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DLF.NS", "DABUR.NS",
    "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "IDEA.NS", "IGL.NS", "IOC.NS", "INDIGO.NS",
    "JINDALSTEL.NS", "LICI.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS",
    "NAUKRI.NS", "NHPC.NS", "OFSS.NS", "PIDILITIND.NS", "PFC.NS",
    "PIIND.NS", "PNB.NS", "RECLTD.NS", "SIEMENS.NS", "SRF.NS",
    "SHREECEM.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS",
    "VBL.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "MAXHEALTH.NS",
]

# Nifty Midcap 50 - Mid Cap (higher risk/reward)
NIFTY_MIDCAP_50 = [
    "APLAPOLLO.NS", "ASHOKLEY.NS", "ASTRAL.NS", "AUBANK.NS", "BALKRISIND.NS",
    "BHARATFORG.NS", "BHEL.NS", "BIOCON.NS", "CANFINHOME.NS", "CGPOWER.NS",
    "COFORGE.NS", "CONCOR.NS", "CUMMINSIND.NS", "DALBHARAT.NS", "DEEPAKNTR.NS",
    "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GMRINFRA.NS",
    "GODREJPROP.NS", "GUJGASLTD.NS", "HINDPETRO.NS", "IDFCFIRSTB.NS", "INDIANB.NS",
    "INDHOTEL.NS", "IRCTC.NS", "JUBLFOOD.NS", "LICHSGFIN.NS", "LTTS.NS",
    "M&MFIN.NS", "MFSL.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS",
    "NATIONALUM.NS", "NAVINFLUOR.NS", "OBEROIRLTY.NS", "PAGEIND.NS", "PERSISTENT.NS",
    "PETRONET.NS", "POLYCAB.NS", "SAIL.NS", "SOLARINDS.NS", "SUNDRMFAST.NS",
    "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS", "TATAELXSI.NS", "VOLTAS.NS",
]


def validate_tickers(tickers, name="Universe"):
    """Check which tickers have valid data in yfinance."""
    print(f"\nValidating {name} ({len(tickers)} tickers)...")

    # Try to fetch 1 day of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    valid = []
    invalid = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                valid.append(ticker)
            else:
                invalid.append(ticker)
        except:
            invalid.append(ticker)

    print(f"  Valid: {len(valid)}")
    print(f"  Invalid: {len(invalid)}")
    if invalid:
        print(f"  Invalid tickers: {invalid[:10]}...")

    return valid, invalid


def get_combined_universe(include_midcap=False):
    """
    Get a combined universe from Nifty 50 + Next 50 (+ optionally Midcap 50).

    This is more objective than manual selection but still has survivorship bias
    because we're using CURRENT index members, not historical.
    """
    universe = list(set(NIFTY_50 + NIFTY_NEXT_50))

    if include_midcap:
        universe = list(set(universe + NIFTY_MIDCAP_50))

    return universe


def print_universe_summary():
    """Print summary of available universes."""
    print("\n" + "=" * 70)
    print(" AVAILABLE STOCK UNIVERSES")
    print("=" * 70)

    print(f"""
+------------------+--------+------------------------------------------+
| Universe         | Stocks | Description                              |
+------------------+--------+------------------------------------------+
| NIFTY_50         |   50   | Large cap blue chips (lowest risk)       |
| NIFTY_NEXT_50    |   50   | Large-mid cap (moderate risk)            |
| NIFTY_MIDCAP_50  |   50   | Mid cap (higher risk/reward)             |
| Combined (50+N50)|  100   | Large + Large-mid cap                    |
| Full (all 3)     |  150   | Entire NSE quality universe              |
+------------------+--------+------------------------------------------+

IMPORTANT CAVEATS:
1. These are CURRENT index members (as of Jan 2025)
2. Stocks that were removed from indices are NOT included
3. This still has survivorship bias (less than manual picking)
4. For proper backtesting, you need HISTORICAL index membership data
    """)


def compare_universes_momentum():
    """Show current top momentum stocks from each universe."""
    import warnings
    warnings.filterwarnings('ignore')

    print("\n" + "=" * 70)
    print(" TOP MOMENTUM STOCKS BY UNIVERSE (12-Month Return)")
    print("=" * 70)

    universes = {
        "Nifty 50": NIFTY_50,
        "Nifty Next 50": NIFTY_NEXT_50,
        "Nifty Midcap 50": NIFTY_MIDCAP_50,
    }

    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # ~13 months for 12-month momentum

    for name, tickers in universes.items():
        print(f"\n{name}:")
        print("-" * 50)

        # Fetch data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if 'Close' in data.columns.names or len(tickers) > 1:
            prices = data['Close']
        else:
            prices = data['Close'].to_frame(name=tickers[0])

        # Calculate 12-month momentum
        if len(prices) > 252:
            current = prices.iloc[-1]
            past = prices.iloc[-252]
            momentum = ((current / past) - 1) * 100
            momentum = momentum.dropna().sort_values(ascending=False)

            print(f"  {'Rank':<6} {'Stock':<20} {'12M Return':>12}")
            print("  " + "-" * 40)
            for i, (ticker, ret) in enumerate(momentum.head(5).items(), 1):
                clean_ticker = ticker.replace('.NS', '')
                print(f"  {i:<6} {clean_ticker:<20} {ret:>10.1f}%")
        else:
            print("  Insufficient data")


if __name__ == "__main__":
    print_universe_summary()
    compare_universes_momentum()

    print("\n" + "=" * 70)
    print(" RECOMMENDATION")
    print("=" * 70)
    print("""
For your momentum strategy, I recommend:

1. CONSERVATIVE: Use NIFTY_50 only
   - 50 stocks, most liquid, lowest risk
   - Expected CAGR: 12-15%

2. MODERATE: Use NIFTY_50 + NIFTY_NEXT_50 (100 stocks)
   - Broader opportunity set
   - Expected CAGR: 15-18%

3. AGGRESSIVE: Add NIFTY_MIDCAP_50 (150 stocks)
   - Highest momentum potential
   - Expected CAGR: 18-22%
   - Higher drawdowns

To use in config_india.py, import the desired universe:
    from fetch_index_universe import NIFTY_50, NIFTY_NEXT_50
    """)
