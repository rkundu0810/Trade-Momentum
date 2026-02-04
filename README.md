# Momentum Trading Strategy

A momentum-based stock selection strategy for Indian markets (NSE).

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| Universe | Nifty 50 + Nifty Next 50 (~95 stocks) |
| Selection | Top 3 by 12-month momentum |
| Rebalancing | Monthly (1st trading day) |
| Position Sizing | Equal weight (33.3% each) |
| Regime Filter | Nifty 50 > 200 SMA |

## Backtest Results (2015-2024)

| Metric | Value |
|--------|-------|
| CAGR | 18.4% |
| Max Drawdown | 19.5% |
| Sharpe Ratio | 0.90 |
| Alpha (vs Nifty) | +7.3% |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Monthly Alert (Use This!)
```bash
python monthly_alert.py 3000000  # Rs.30 Lakhs capital
```

### 3. Run Full Backtest
```bash
python momentum_india.py
```

## Files

| File | Purpose |
|------|---------|
| `monthly_alert.py` | **Run monthly for stock picks** |
| `momentum_india.py` | Full backtest engine |
| `config_india.py` | Universe & configuration |
| `utils.py` | Metrics & charting |
| `requirements.txt` | Dependencies |

## How It Works

1. **Monthly Rebalancing**: On 1st trading day of each month
2. **Momentum Calculation**: `(Price Today / Price 252 days ago) - 1`
3. **Stock Selection**: Buy top 3 stocks by momentum
4. **Regime Filter**: Go to cash if Nifty 50 < 200 SMA

## Disclaimer

- Past performance does not guarantee future results
- Backtest has survivorship bias (uses current index members)
- Realistic returns likely 30-50% lower than backtest
- This is for educational purposes only

## License

MIT
