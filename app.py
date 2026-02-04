"""
Momentum Strategy Web App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Momentum Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# UNIVERSE
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


# ============================================================================
# DATA FETCHING (Cached)
# ============================================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(start_date, end_date):
    """Fetch price data from Yahoo Finance."""
    tickers = UNIVERSE + ["^NSEI"]

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )

    prices = data['Close'].ffill()
    return prices


# ============================================================================
# CALCULATIONS
# ============================================================================
def calculate_momentum(prices, lookback=252):
    """Calculate momentum for all stocks."""
    if len(prices) < lookback:
        return pd.Series(dtype=float)

    current = prices.iloc[-1]
    past = prices.iloc[-lookback]

    momentum = ((current / past) - 1) * 100
    return momentum.dropna().sort_values(ascending=False)


def check_regime(benchmark_prices, sma_period=200):
    """Check if market is in risk-on mode."""
    if len(benchmark_prices) < sma_period:
        return True, 0, 0

    current = benchmark_prices.iloc[-1]
    sma = benchmark_prices.iloc[-sma_period:].mean()

    return current > sma, current, sma


def run_backtest(prices, benchmark, config):
    """Run momentum backtest."""
    start = pd.to_datetime(config['start_date'])
    end = pd.to_datetime(config['end_date'])

    trading_days = prices.index[(prices.index >= start) & (prices.index <= end)]

    if len(trading_days) == 0:
        return None, None

    # Get monthly rebalance dates
    monthly_groups = trading_days.to_series().groupby([trading_days.year, trading_days.month])
    rebalance_dates = set(group.iloc[0] for _, group in monthly_groups)

    cash = config['capital']
    holdings = {}
    entry_prices = {}
    equity_history = []
    trades = []

    for date in trading_days:
        if date in rebalance_dates:
            # Check regime
            if config['use_regime_filter']:
                bench_to_date = benchmark.loc[:date]
                risk_on = bench_to_date.iloc[-1] > bench_to_date.iloc[-200:].mean() if len(bench_to_date) >= 200 else True
            else:
                risk_on = True

            # Get momentum
            prices_to_date = prices.loc[:date]
            if len(prices_to_date) >= config['lookback']:
                momentum = calculate_momentum(prices_to_date, config['lookback'])
                new_stocks = momentum.head(config['top_n']).index.tolist() if risk_on else []
            else:
                new_stocks = []

            prices_today = prices.loc[date]

            # Sell holdings
            for ticker, shares in list(holdings.items()):
                if ticker in prices_today and not pd.isna(prices_today[ticker]):
                    sell_price = prices_today[ticker]
                    sell_value = shares * sell_price
                    cost = sell_value * config['txn_cost']
                    pnl = (sell_price - entry_prices.get(ticker, sell_price)) * shares - cost

                    trades.append({
                        'date': date,
                        'ticker': ticker.replace('.NS', ''),
                        'action': 'SELL',
                        'price': sell_price,
                        'pnl': pnl
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
                        effective = target / (1 + config['txn_cost'])
                        shares = effective / buy_price
                        actual_value = shares * buy_price
                        cost = actual_value * config['txn_cost']

                        holdings[ticker] = shares
                        entry_prices[ticker] = buy_price
                        cash -= (actual_value + cost)

                        trades.append({
                            'date': date,
                            'ticker': ticker.replace('.NS', ''),
                            'action': 'BUY',
                            'price': buy_price,
                            'pnl': 0
                        })

        # Daily equity
        holdings_value = sum(
            shares * prices.loc[date].get(ticker, 0)
            for ticker, shares in holdings.items()
            if not pd.isna(prices.loc[date].get(ticker, np.nan))
        )
        equity_history.append({'date': date, 'equity': cash + holdings_value})

    equity_df = pd.DataFrame(equity_history).set_index('date')
    trades_df = pd.DataFrame(trades)

    return equity_df, trades_df


def calculate_metrics(equity_df, benchmark, initial_capital):
    """Calculate performance metrics."""
    equity = equity_df['equity']
    returns = equity.pct_change().dropna()

    # Years
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    # CAGR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1

    # Max Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Sharpe
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Benchmark metrics
    bench_aligned = benchmark.loc[equity.index[0]:equity.index[-1]]
    bench_returns = bench_aligned.pct_change().dropna()
    bench_cagr = (bench_aligned.iloc[-1] / bench_aligned.iloc[0]) ** (1/years) - 1

    return {
        'total_return': (equity.iloc[-1] / initial_capital - 1) * 100,
        'cagr': cagr * 100,
        'max_drawdown': max_dd * 100,
        'sharpe': sharpe,
        'volatility': returns.std() * np.sqrt(252) * 100,
        'benchmark_cagr': bench_cagr * 100,
        'alpha': (cagr - bench_cagr) * 100,
        'final_value': equity.iloc[-1],
    }


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar():
    """Render sidebar with configuration."""
    st.sidebar.header("⚙️ Strategy Settings")

    capital = st.sidebar.number_input(
        "Investment Capital (₹)",
        min_value=100000,
        max_value=100000000,
        value=3000000,
        step=100000,
        format="%d"
    )

    top_n = st.sidebar.slider("Number of Stocks to Hold", 1, 10, 3)

    lookback = st.sidebar.selectbox(
        "Momentum Lookback",
        options=[63, 126, 189, 252],
        index=3,
        format_func=lambda x: f"{x} days (~{x//21} months)"
    )

    use_regime = st.sidebar.checkbox("Use Regime Filter (Nifty > 200 SMA)", value=True)

    txn_cost = st.sidebar.slider(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05
    ) / 100

    st.sidebar.header("📅 Backtest Period")

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", datetime(2015, 1, 1))
    end_date = col2.date_input("End", datetime.now())

    return {
        'capital': capital,
        'top_n': top_n,
        'lookback': lookback,
        'use_regime_filter': use_regime,
        'txn_cost': txn_cost,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
    }


def render_current_picks(prices, benchmark, config):
    """Render current month's stock picks."""
    st.header("📊 Current Month's Picks")

    # Regime check
    risk_on, nifty_price, nifty_sma = check_regime(benchmark)

    col1, col2, col3 = st.columns(3)
    col1.metric("Nifty 50", f"₹{nifty_price:,.0f}")
    col2.metric("200-day SMA", f"₹{nifty_sma:,.0f}")
    col3.metric("Regime", "🟢 RISK ON" if risk_on else "🔴 RISK OFF")

    if not risk_on:
        st.warning("⚠️ Market is below 200 SMA. Strategy recommends staying in CASH.")
        return

    # Calculate momentum
    momentum = calculate_momentum(prices.drop(columns=['^NSEI'], errors='ignore'), config['lookback'])

    # Top picks
    top_picks = momentum.head(config['top_n'])

    st.subheader(f"Top {config['top_n']} Momentum Stocks")

    # Create picks table
    picks_data = []
    position_value = config['capital'] / config['top_n']

    for rank, (ticker, mom) in enumerate(top_picks.items(), 1):
        price = prices[ticker].iloc[-1]
        qty = int(position_value / price)
        investment = qty * price

        picks_data.append({
            'Rank': rank,
            'Stock': ticker.replace('.NS', ''),
            '12M Return': f"{mom:.1f}%",
            'Price': f"₹{price:,.0f}",
            'Quantity': qty,
            'Investment': f"₹{investment:,.0f}"
        })

    picks_df = pd.DataFrame(picks_data)
    st.dataframe(picks_df, use_container_width=True, hide_index=True)

    # Full rankings
    with st.expander("📋 View All Momentum Rankings"):
        all_data = []
        for rank, (ticker, mom) in enumerate(momentum.items(), 1):
            price = prices[ticker].iloc[-1]
            all_data.append({
                'Rank': rank,
                'Stock': ticker.replace('.NS', ''),
                '12M Return (%)': round(mom, 2),
                'Price': round(price, 2),
            })

        all_df = pd.DataFrame(all_data)
        st.dataframe(all_df, use_container_width=True, hide_index=True, height=400)


def render_backtest_results(equity_df, trades_df, benchmark, config):
    """Render backtest results."""
    st.header("📈 Backtest Results")

    metrics = calculate_metrics(equity_df, benchmark, config['capital'])

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics['total_return']:.1f}%")
    col2.metric("CAGR", f"{metrics['cagr']:.1f}%")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")
    col4.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Value", f"₹{metrics['final_value']:,.0f}")
    col2.metric("Volatility", f"{metrics['volatility']:.1f}%")
    col3.metric("Nifty CAGR", f"{metrics['benchmark_cagr']:.1f}%")
    col4.metric("Alpha", f"{metrics['alpha']:.1f}%")

    # Equity curve
    st.subheader("Equity Curve")

    bench_aligned = benchmark.loc[equity_df.index[0]:equity_df.index[-1]]

    fig = go.Figure()

    # Normalize to 100
    equity_norm = equity_df['equity'] / equity_df['equity'].iloc[0] * 100
    bench_norm = bench_aligned / bench_aligned.iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=equity_norm.index,
        y=equity_norm.values,
        name='Strategy',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=bench_norm.index,
        y=bench_norm.values,
        name='Nifty 50',
        line=dict(color='gray', width=1.5, dash='dash')
    ))

    fig.update_layout(
        title='Strategy vs Nifty 50 (Normalized to 100)',
        yaxis_title='Value',
        yaxis_type='log',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    st.subheader("Drawdown")

    running_max = equity_df['equity'].expanding().max()
    drawdown = (equity_df['equity'] - running_max) / running_max * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=1),
        name='Drawdown'
    ))

    fig_dd.update_layout(
        title='Portfolio Drawdown',
        yaxis_title='Drawdown (%)',
        height=300
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    # Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")

    monthly_equity = equity_df['equity'].resample('ME').last()
    monthly_returns = monthly_equity.pct_change() * 100

    monthly_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    }).dropna()

    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        aspect='auto',
        labels=dict(color='Return (%)')
    )
    fig_hm.update_layout(height=400)

    st.plotly_chart(fig_hm, use_container_width=True)

    # Trades table
    if len(trades_df) > 0:
        with st.expander("📋 View All Trades"):
            trades_display = trades_df.copy()
            trades_display['date'] = pd.to_datetime(trades_display['date']).dt.strftime('%Y-%m-%d')
            trades_display['price'] = trades_display['price'].round(2)
            trades_display['pnl'] = trades_display['pnl'].round(2)
            st.dataframe(trades_display, use_container_width=True, hide_index=True, height=400)


def render_projection(metrics, config):
    """Render projection to target."""
    st.header("🎯 Projection: ₹30L to ₹4Cr")

    cagr = metrics['cagr'] / 100
    initial = config['capital']
    target = 4_00_00_000

    if cagr > 0:
        years_to_target = np.log(target / initial) / np.log(1 + cagr)
    else:
        years_to_target = float('inf')

    col1, col2, col3 = st.columns(3)
    col1.metric("Your CAGR", f"{cagr*100:.1f}%")
    col2.metric("Years to ₹4 Cr", f"{years_to_target:.1f}" if years_to_target < 100 else "N/A")
    col3.metric("Target", "₹4,00,00,000")

    # Projection table
    projections = []
    value = initial
    for year in range(int(min(years_to_target + 2, 25))):
        value = initial * ((1 + cagr) ** year)
        milestone = ""
        if value >= 4_00_00_000 and initial * ((1 + cagr) ** (year-1)) < 4_00_00_000:
            milestone = "🎯 TARGET!"
        elif value >= 1_00_00_000 and initial * ((1 + cagr) ** (year-1)) < 1_00_00_000:
            milestone = "1 Crore!"

        projections.append({
            'Year': year,
            'Value': f"₹{value/100000:.1f}L" if value < 1_00_00_000 else f"₹{value/10000000:.2f}Cr",
            'Milestone': milestone
        })

        if value >= 5_00_00_000:
            break

    proj_df = pd.DataFrame(projections)
    st.dataframe(proj_df, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📈 Momentum Strategy Simulator")
    st.markdown("*Nifty 100 Universe | Monthly Rebalancing | Top 3 Momentum Stocks*")

    # Sidebar
    config = render_sidebar()

    # Fetch data
    with st.spinner("Fetching market data..."):
        buffer_start = (datetime.strptime(config['start_date'], '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
        prices = fetch_data(buffer_start, config['end_date'])

    if prices is None or len(prices) == 0:
        st.error("Failed to fetch data. Please try again.")
        return

    benchmark = prices['^NSEI']
    stock_prices = prices.drop(columns=['^NSEI'], errors='ignore')

    st.success(f"✅ Loaded data for {len(stock_prices.columns)} stocks")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Current Picks", "📊 Backtest", "🔮 Projection"])

    with tab1:
        render_current_picks(prices, benchmark, config)

    with tab2:
        if st.button("▶️ Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                equity_df, trades_df = run_backtest(stock_prices, benchmark, config)

            if equity_df is not None:
                st.session_state['equity_df'] = equity_df
                st.session_state['trades_df'] = trades_df
                st.session_state['config'] = config

        if 'equity_df' in st.session_state:
            render_backtest_results(
                st.session_state['equity_df'],
                st.session_state['trades_df'],
                benchmark,
                st.session_state['config']
            )

    with tab3:
        if 'equity_df' in st.session_state:
            metrics = calculate_metrics(
                st.session_state['equity_df'],
                benchmark,
                st.session_state['config']['capital']
            )
            render_projection(metrics, st.session_state['config'])
        else:
            st.info("Run a backtest first to see projections.")


if __name__ == "__main__":
    main()
