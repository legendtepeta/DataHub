import pandas as pd
import glob

files = glob.glob('data/daily/*nasdaq*.csv') + glob.glob('data/daily/*nq*.csv')
for f in files:
    try:
        df = pd.read_csv(f)
        print(f"\n--- {f} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        date_col = 'Date' if 'Date' in df.columns else 'date' if 'date' in df.columns else None
        if date_col:
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            print(f"Date range: {min_date} to {max_date}")
            print(f"Missing values:\n{df.isnull().sum().to_string()}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
