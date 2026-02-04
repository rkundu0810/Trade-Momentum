"""
Monthly Momentum Alert System

Run this on the 1st trading day of each month to get:
- Top 3 momentum stocks to BUY
- Current holdings to SELL
- Position sizes based on your capital

Universe: Nifty 50 + Nifty Next 50 (100 stocks)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# UNIVERSE: Nifty 50 + Nifty Next 50
# ============================================================================

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

NIFTY_NEXT_50 = [
    "ABB.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BAJAJHLDNG.NS", "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS", "BOSCHLTD.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DLF.NS", "DABUR.NS",
    "GAIL.NS", "GODREJCP.NS", "HAL.NS", "HAVELLS.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "IGL.NS", "IOC.NS", "INDIGO.NS",
    "JINDALSTEL.NS", "LICI.NS", "LTIM.NS", "LUPIN.NS", "MARICO.NS",
    "NAUKRI.NS", "NHPC.NS", "PIDILITIND.NS", "PFC.NS",
    "PIIND.NS", "PNB.NS", "RECLTD.NS", "SIEMENS.NS", "SRF.NS",
    "SHREECEM.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS",
    "VBL.NS", "VEDL.NS", "ZYDUSLIFE.NS", "MAXHEALTH.NS",
]

UNIVERSE = list(set(NIFTY_50 + NIFTY_NEXT_50))


def get_monthly_picks(capital: float = 30_00_000, top_n: int = 3):
    """
    Get this month's momentum stock picks.

    Args:
        capital: Your investment capital in Rs.
        top_n: Number of stocks to hold (default 3)
    """
    print("\n" + "=" * 70)
    print(" MONTHLY MOMENTUM ALERT")
    print(f" {datetime.now().strftime('%d %B %Y, %H:%M')}")
    print("=" * 70)

    # Fetch data
    print("\nFetching data for ~95 stocks...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)

    tickers = UNIVERSE + ["^NSEI"]

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )

    prices = data['Close']
    prices = prices.ffill()

    benchmark = prices["^NSEI"]
    universe_prices = prices[[t for t in UNIVERSE if t in prices.columns]]

    latest_date = universe_prices.index[-1]
    print(f"  Data as of: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  Stocks with data: {len(universe_prices.columns)}")

    # ========================================================================
    # REGIME CHECK: Nifty 50 vs 200 SMA
    # ========================================================================
    nifty_price = benchmark.iloc[-1]
    nifty_sma_200 = benchmark.iloc[-200:].mean()
    risk_on = nifty_price > nifty_sma_200

    print(f"\n" + "-" * 70)
    print(" MARKET REGIME CHECK")
    print("-" * 70)
    print(f"  Nifty 50:      {nifty_price:>12,.0f}")
    print(f"  200-day SMA:   {nifty_sma_200:>12,.0f}")
    print(f"  Difference:    {((nifty_price/nifty_sma_200)-1)*100:>12.1f}%")
    print(f"\n  REGIME: {'RISK ON - OK to invest' if risk_on else 'RISK OFF - Stay in CASH'}")
    print("-" * 70)

    if not risk_on:
        print("""
  +------------------------------------------------------------------+
  |  ACTION: SELL ALL HOLDINGS AND MOVE TO CASH                      |
  |                                                                  |
  |  Nifty is BELOW its 200-day moving average.                      |
  |  Wait for Nifty to close above 200 SMA before re-entering.       |
  +------------------------------------------------------------------+
        """)
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'regime': 'OFF',
            'action': 'SELL ALL - GO TO CASH',
            'stocks': []
        }

    # ========================================================================
    # CALCULATE 12-MONTH MOMENTUM
    # ========================================================================
    lookback = 252  # 12 months

    if len(universe_prices) < lookback:
        print("ERROR: Not enough data for momentum calculation")
        return None

    current_prices = universe_prices.iloc[-1]
    past_prices = universe_prices.iloc[-lookback]

    momentum = ((current_prices / past_prices) - 1) * 100  # Convert to percentage
    momentum = momentum.dropna().sort_values(ascending=False)

    # ========================================================================
    # TOP PICKS
    # ========================================================================
    print(f"\n" + "=" * 70)
    print(" 12-MONTH MOMENTUM RANKINGS")
    print("=" * 70)

    print(f"\n  {'Rank':<6}{'Stock':<15}{'12M Return':<12}{'Price':<12}{'Signal'}")
    print("-" * 70)

    top_stocks = []
    for i, (ticker, ret) in enumerate(momentum.items(), 1):
        clean = ticker.replace('.NS', '')
        price = current_prices[ticker]
        signal = ">>> BUY <<<" if i <= top_n else ""

        print(f"  {i:<6}{clean:<15}{ret:>8.1f}%   Rs.{price:>8,.0f}  {signal}")

        if i <= top_n:
            top_stocks.append({
                'rank': i,
                'ticker': clean,
                'ticker_ns': ticker,
                'momentum_12m': round(ret, 2),
                'price': round(price, 2)
            })

        if i == top_n:
            print("-" * 70)

        if i >= 15:  # Show top 15
            remaining = len(momentum) - 15
            print(f"  ... and {remaining} more stocks")
            break

    # ========================================================================
    # POSITION SIZING
    # ========================================================================
    print(f"\n" + "=" * 70)
    print(" TRADE EXECUTION PLAN")
    print("=" * 70)

    position_value = capital / top_n

    print(f"\n  Your Capital:     Rs. {capital:>12,.0f}")
    print(f"  Positions:        {top_n} stocks")
    print(f"  Per Position:     Rs. {position_value:>12,.0f}")

    print(f"\n  {'Stock':<12}{'Price':<14}{'Quantity':<12}{'Investment':<15}")
    print("-" * 70)

    total_invested = 0
    for stock in top_stocks:
        qty = int(position_value / stock['price'])
        investment = qty * stock['price']
        stock['quantity'] = qty
        stock['investment'] = investment
        total_invested += investment

        print(f"  {stock['ticker']:<12}Rs. {stock['price']:<10,.0f}{qty:<12}Rs. {investment:>12,.0f}")

    print("-" * 70)
    print(f"  {'TOTAL':<12}{'':<14}{'':<12}Rs. {total_invested:>12,.0f}")
    cash_remaining = capital - total_invested
    print(f"  {'Cash Left':<12}{'':<14}{'':<12}Rs. {cash_remaining:>12,.0f}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n" + "=" * 70)
    print(" ACTION SUMMARY")
    print("=" * 70)
    print(f"""
  1. SELL all existing holdings (if different from below)

  2. BUY these {top_n} stocks:
""")
    for stock in top_stocks:
        print(f"     - {stock['ticker']:<12} {stock['quantity']} shares @ Rs.{stock['price']:,.0f}")

    print(f"""
  3. HOLD until next month's 1st trading day

  4. NEXT REBALANCE: ~{(datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1).strftime('%d %B %Y')}
""")
    print("=" * 70)

    # Save to JSON
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'data_date': latest_date.strftime('%Y-%m-%d'),
        'regime': 'ON',
        'nifty_price': float(nifty_price),
        'nifty_sma_200': float(nifty_sma_200),
        'capital': capital,
        'top_n': top_n,
        'stocks': top_stocks,
        'total_invested': total_invested,
        'cash_remaining': cash_remaining,
    }

    os.makedirs('alerts', exist_ok=True)
    filename = f"alerts/monthly_{datetime.now().strftime('%Y%m%d')}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Alert saved: {filename}")

    return result


if __name__ == "__main__":
    import sys

    # Default capital Rs.30 Lakhs
    capital = 30_00_000

    if len(sys.argv) > 1:
        try:
            capital = float(sys.argv[1])
        except:
            print(f"Usage: python monthly_alert.py [capital]")
            print(f"Example: python monthly_alert.py 5000000")
            sys.exit(1)

    get_monthly_picks(capital=capital)
