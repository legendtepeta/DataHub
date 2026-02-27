import requests
import pandas as pd
import os
from io import StringIO

def download_fred_csv(series_id, filename):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Downloading {series_id} from {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        df.to_csv(filename, index=False)
        print(f"Saved {series_id} to {filename}")
        return True
    else:
        print(f"Failed to download {series_id}: {response.status_code}")
        return False

def main():
    series = {
        "FEDFUNDS": "fed_funds_rate.csv",
        "CPIAUCSL": "cpi.csv",
        "UNRATE": "unemployment_rate.csv",
        "T10Y2Y": "yield_curve_spread.csv"
    }
    
    output_dir = "data/daily"
    os.makedirs(output_dir, exist_ok=True)
    
    for series_id, filename in series.items():
        download_fred_csv(series_id, os.path.join(output_dir, filename))

if __name__ == "__main__":
    main()
