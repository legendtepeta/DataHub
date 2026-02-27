# Nasdaq Data Hub

This directory contains historical daily candle data for Nasdaq instruments from 2015 to 2025.

## Data Files

- **`nasdaq_futures_micro_mnq.csv`**: Micro E-mini Nasdaq 100 Futures (`MNQ=F`).
  - *Note*: Data starts from **2019-05-03** (MNQ launch date).
- **`nasdaq_futures_emini_nq.csv`**: E-mini Nasdaq 100 Futures (`NQ=F`).
  - Contains data from 2015-01-01 to 2025-12-31. Use this for futures analysis prior to 2019.
- **`nasdaq_100_index_cfd_proxy.csv`**: Nasdaq 100 Index (`^NDX`).
  - This is the underlying index for most Nasdaq CFDs.
- **`nasdaq_composite_index.csv`**: Nasdaq Composite Index (`^IXIC`).

## Data Source
- Data downloaded via Yahoo Finance using the `yfinance` library.
- Resolution: Daily (1d) candles.
- Timeframe: 2015-01-01 to 2025-12-30.

## How to update
You can run the script in `scripts/download_nasdaq.py` using the provided virtual environment:
```bash
./venv/bin/python3 scripts/download_nasdaq.py
```
