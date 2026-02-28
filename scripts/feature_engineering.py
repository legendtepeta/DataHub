import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.math_utils import get_hurst_exponent, calculate_entropy, calculate_fractal_dimension

def load_and_merge_datasets(base_df):
    """
    Merges local Nasdaq data with external macro datasets.
    """
    data_dir = "data/daily"
    external_files = {
        "VIX": "vix_volatility_index.csv",
        "TNX_10Y": "10y_treasury_yield.csv",
        "DXY": "usd_index.csv",
        "SPX": "sp500_index.csv",
        "Oil": "crude_oil_futures.csv",
        "Gold": "gold_futures.csv",
        "Yield_Curve": "yield_curve_spread.csv",
        "Fed_Funds": "fed_funds_rate.csv",
        "CPI": "cpi.csv"
    }
    
    merged_df = base_df.copy()
    
    for name, filename in external_files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            ext_df = pd.read_csv(path)
            # Normalize date column case
            ext_df.columns = [c.upper() for c in ext_df.columns]
            date_col = next((c for c in ext_df.columns if 'DATE' in c), None)
            if date_col:
                ext_df = ext_df.rename(columns={date_col: 'Date'})
                ext_df['Date'] = pd.to_datetime(ext_df['Date'])
                ext_df = ext_df.set_index('Date')
            
            # Take relevant column (usually the last/only value column)
            # In FRED it's the 1st column (index 1 if 0 is Date), in Yahoo it's index 4 (Close)
            if 'CLOSE' in ext_df.columns:
                ext_col = ext_df['CLOSE'].rename(f'Ext_{name}')
            else:
                # Use the first column that isn't 'Date'
                val_col = [c for c in ext_df.columns if c != 'Date'][0]
                ext_col = ext_df[val_col].rename(f'Ext_{name}')
            
            # Left join and ONLY ffill the newly joined column to prevent data leakage
            merged_df = merged_df.join(ext_col, how='left')
            merged_df[f'Ext_{name}'] = merged_df[f'Ext_{name}'].ffill()
            
    return merged_df

def engineer_complex_features(df):
    """
    Calculates advanced statistical features (Hurst, Entropy, etc.)
    """
    print("Calculating complex statistical features...")
    window = 63 # Quarter lookback for stable stats
    
    closes = df['Close'].values
    log_rets = df['Log_Ret'].values
    
    hurst_vals = [np.nan] * window
    entropy_vals = [np.nan] * window
    
    # O(N) Loop optimized using lists instead of expensive pd.DataFrame.iloc assignment
    for i in range(window, len(df)):
        window_closes = closes[i-window:i]
        window_rets = log_rets[i-window:i]
        
        hurst_vals.append(get_hurst_exponent(window_closes))
        entropy_vals.append(calculate_entropy(window_rets))
        
    df['Hurst_63'] = hurst_vals
    df['Entropy_63'] = entropy_vals
        
    return df

def create_target_and_baseline(df):
    df = df.copy()
    df['Fwd_Ret_1d'] = df['Close'].pct_change().shift(-1)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    
    df['Regime_Baseline'] = 0 
    bull_mask = (df['Close'] > df['SMA_20']) & (df['Close'] > df['SMA_50'])
    df.loc[bull_mask, 'Regime_Baseline'] = 1
    bear_mask = (df['Close'] < df['SMA_20']) & (df['Close'] < df['SMA_50'])
    df.loc[bear_mask, 'Regime_Baseline'] = -1
    return df

def engineer_features(df):
    df = df.copy()
    
    # 1. Internal Technicals (Subset for clarity)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [1, 2, 3, 5]:
        df[f'Log_Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)
    
    # Volatility
    df['Vol_21d'] = df['Log_Ret'].rolling(21).std() * np.sqrt(252)
    df['NATR_14'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)
    
    # Trend distances
    df['Dist_SMA_200'] = (df['Close'] - ta.sma(df['Close'], length=200)) / ta.sma(df['Close'], length=200)
    
    # ADX / RSI
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None:
        df['ADX_14'] = adx['ADX_14']
    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    # Volume Profile: VWMA Distance and OBV
    vwma_21 = ta.vwma(df['Close'], df['Volume'], length=21)
    df['Dist_VWMA_21'] = (df['Close'] - vwma_21) / vwma_21
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_ROC_21'] = df['OBV'].pct_change(21)
    
    # Higher-Order Moments (Skewness & Kurtosis)
    df['Log_Ret_Skew_63'] = df['Log_Ret'].rolling(63).skew()
    df['Log_Ret_Kurt_63'] = df['Log_Ret'].rolling(63).kurt()
    
    # 2. External Macro Interactions
    # VIX Ratio (Current vs Mean) - High ratio = fear spike
    if 'Ext_VIX' in df.columns:
        df['VIX_Relative'] = df['Ext_VIX'] / df['Ext_VIX'].rolling(252).mean()
        df['VIX_Relative_ROC_21'] = df['VIX_Relative'].pct_change(21)
    
    # Yield Curve Slope (10Y - 3M approximation)
    if 'Ext_Yield_Curve' in df.columns:
        df['Macro_Yield_Curve'] = df['Ext_Yield_Curve']
    
    # Correlation and Beta to SPX
    if 'Ext_SPX' in df.columns:
        spx_log_ret = np.log(df['Ext_SPX'] / df['Ext_SPX'].shift(1))
        df['Corr_SPX_21'] = df['Log_Ret'].rolling(21).corr(spx_log_ret)
        
        cov_spx = df['Log_Ret'].rolling(21).cov(spx_log_ret)
        var_spx = spx_log_ret.rolling(21).var()
        df['Beta_SPX_21'] = cov_spx / var_spx
    
    # Commodity Ratios
    if 'Ext_Gold' in df.columns and 'Ext_Oil' in df.columns:
        df['Gold_Oil_Ratio'] = df['Ext_Gold'] / df['Ext_Oil'] # Safe haven vs Energy
        df['Gold_Oil_Ratio_ROC_21'] = df['Gold_Oil_Ratio'].pct_change(21)
    
    # 4. Tail Risk & Market Microstructure
    df['Ret_ZScore_252'] = (df['Log_Ret'] - df['Log_Ret'].rolling(252).mean()) / df['Log_Ret'].rolling(252).std()
    df['Vol_ZScore_21'] = (df['Volume'] - df['Volume'].rolling(21).mean()) / df['Volume'].rolling(21).std()
    
    # 5. Macro "Event" Flags
    if 'Ext_DXY' in df.columns:
        df['DXY_Change'] = df['Ext_DXY'].pct_change()
        df['Macro_Volatility_Event'] = (df['DXY_Change'].abs() > df['DXY_Change'].abs().rolling(252).quantile(0.95)).astype(int)

    return df

def main():
    input_file = "data/daily/nasdaq_futures_emini_nq.csv"
    output_file = "data/daily/nq_extensive_features.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find input file at {input_file}")
        return

    # 1. Load Data
    df = pd.read_csv(input_file, index_col='Date', parse_dates=True)
    
    # 2. Merge External
    print("Merging external macro datasets...")
    df = load_and_merge_datasets(df)
    
    # 3. Base Features
    print("Engineering technical and macro interaction features...")
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1)) # Required for complex stats
    df = engineer_features(df)
    
    # 4. Complex Stats (Takes a moment)
    df = engineer_complex_features(df)
    
    # 5. Targets & Labels
    df = create_target_and_baseline(df)
    
    # 6. Clean and Save
    print(f"Cleaning dataset (Initial shape: {df.shape})...")
    df = df.dropna()
    print(f"Final shape: {df.shape}")
    
    df.to_csv(output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
