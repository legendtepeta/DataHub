import pandas as pd
import glob
import json

def analyze():
    files = glob.glob('data/daily/*nasdaq*.csv') + glob.glob('data/daily/*nq*.csv')
    res = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            date_col = 'Date' if 'Date' in df.columns else 'date' if 'date' in df.columns else None
            rng = f"{df[date_col].min()} to {df[date_col].max()}" if date_col else "No date col"
            res[f] = {
                'shape': df.shape,
                'range': rng,
                'nas': df.isnull().sum().sum()
            }
        except Exception as e:
            res[f] = {'error': str(e)}
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    analyze()
