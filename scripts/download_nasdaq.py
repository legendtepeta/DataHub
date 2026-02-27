import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def download_data(ticker, start_date, end_date, filename):
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if data.empty:
        print(f"No data found for {ticker}")
        return False
    
    # Flatten multi-index columns if they exist (yfinance 0.2.x behavior)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    data.to_csv(filename)
    print(f"Saved {ticker} to {filename}")
    return True

def main():
    start_date = "2015-01-01"
    end_date = "2025-12-31"  # User asked for 2015 to 2025
    
    # Output directory
    output_dir = "data/daily"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tickers to download
    # MNQ=F: Micro Nasdaq 100 Futures
    # NQ=F: E-mini Nasdaq 100 Futures (better history before 2019)
    # ^NDX: Nasdaq 100 Index (CFD proxy)
    # ^IXIC: Nasdaq Composite
    
    tickers = {
        "MNQ=F": "nasdaq_futures_micro_mnq.csv",
        "NQ=F": "nasdaq_futures_emini_nq.csv",
        "^NDX": "nasdaq_100_index_cfd_proxy.csv",
        "^IXIC": "nasdaq_composite_index.csv"
    }
    
    for ticker, filename in tickers.items():
        filepath = os.path.join(output_dir, filename)
        download_data(ticker, start_date, end_date, filepath)

if __name__ == "__main__":
    main()
