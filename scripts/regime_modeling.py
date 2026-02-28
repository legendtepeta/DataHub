import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

def create_directory_structure():
    """Sets up outputs folders."""
    dirs = [
        "RegimeClassifications/Baseline",
        "RegimeClassifications/GMM_Unsupervised",
        "RegimeClassifications/HMM_Unsupervised",
        "RegimeClassifications/HMM_SemiSupervised",
        "RegimeClassifications/HMM_GMM_Initialized"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def plot_regimes(df, regime_col, title, output_path, states):
    """
    Plots the cumulative return of the underlying asset, 
    color-coded by the identified market regime.
    """
    plt.figure(figsize=(15, 8))
    
    # Calculate cumulative return series
    cum_ret = (1 + df['Log_Ret']).cumprod()
    
    # Create line segments
    points = np.array([df.index.to_numpy(dtype=float), cum_ret.values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Define colors for up to 4 regimes
    cmap = ListedColormap(['red', 'green', 'blue', 'orange'][:states])
    norm = BoundaryNorm(range(states + 1), cmap.N)
    
    # The regime labels for each segment
    # Shift by -1 because segment i describes transition from t to t+1
    regimes = df[regime_col].values[:-1]
    
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(regimes)
    lc.set_linewidth(2)
    
    ax = plt.gca()
    line = ax.add_collection(lc)
    
    # Setup axis
    ax.autoscale()
    plt.title(title, fontsize=16)
    plt.ylabel('Cumulative Return (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar / legend
    cbar = plt.colorbar(line, ax=ax, boundaries=np.arange(states + 1) - 0.5, ticks=range(states))
    if states == 3 and 'Baseline' in title:
         cbar.ax.set_yticklabels(['Bear (-1)', 'Sideways (0)', 'Bull (1)'])
    else:
        cbar.ax.set_yticklabels([f'State {i}' for i in range(states)])
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_performance_summary(df, regime_col, title, output_path):
    """
    Calculates Annualized Return, Volatility, and Sharpe for each Regime 
    and saves to a markdown file.
    """
    summary = []
    summary.append(f"# {title} Performance Summary\n")
    
    summary.append("## Regime Statistics\n")
    summary.append("| State | Days | % of Time | Ann. Return | Ann. Vol | Sharpe Ratio |")
    summary.append("|---|---|---|---|---|---|")
    
    states = sorted(df[regime_col].unique())
    total_days = len(df)
    
    for state in states:
        mask = df[regime_col] == state
        days = mask.sum()
        pct_time = days / total_days
        
        # Performance
        returns = df.loc[mask, 'Log_Ret']
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        summary.append(f"| State {state} | {days} | {pct_time:.1%} | {ann_ret:.2%} | {ann_vol:.2%} | {sharpe:.2f} |")
        
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))

def run_baseline(df):
    """Maps the existing 1, 0, -1 baseline to 0, 1, 2 for plotting standards."""
    df = df.copy()
    # 0 = Bear (-1 originally)
    # 1 = Sideways (0 originally)
    # 2 = Bull (1 originally)
    mapping = {-1: 0, 0: 1, 1: 2}
    df['Mapped_Baseline'] = df['Regime_Baseline'].map(mapping)
    
    out_dir = "RegimeClassifications/Baseline"
    plot_regimes(df, 'Mapped_Baseline', "Heuristic Baseline Regime Model", f"{out_dir}/baseline_chart.png", states=3)
    generate_performance_summary(df, 'Mapped_Baseline', "Heuristic Baseline", f"{out_dir}/baseline_summary.md")
    print("Baseline chart and summary generated.")

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from tqdm import tqdm

def train_walk_forward_model(df, features, model_type, states, init_strategy='kmeans'):
    """
    Performs expanding-window walk-forward validation for GMM or HMM.
    Predicts the state for day t+1 using only data up to day t.
    """
    print(f"Running Walk-Forward {model_type} ({init_strategy}, {states} states)...")
    
    # We need a burn-in period to have enough data to fit the first model
    burn_in = 252 * 2 # 2 years
    
    predictions = np.full(len(df), np.nan)
    
    X_raw = df[features].values
    
    # We iterate from burn_in to the end
    # For speed, we will refit the model every 21 days (1 month), but predict daily
    # Refitting daily is computationally brutal for HMMs
    
    current_model = None
    current_scaler = None
    
    for t in tqdm(range(burn_in, len(df))):
        # Refit model monthly
        if t == burn_in or t % 21 == 0:
            X_train = X_raw[:t] # STRICTLY NO LOOK-AHEAD
            
            current_scaler = StandardScaler()
            X_scaled = current_scaler.fit_transform(X_train)
            
            if model_type == 'GMM':
                current_model = GaussianMixture(n_components=states, covariance_type='full', 
                                              init_params=init_strategy, random_state=42)
                current_model.fit(X_scaled)
                
            elif model_type == 'HMM':
                # Convert init_strategy to hmmlearn args if needed
                init_p = 'kmeans' if init_strategy == 'kmeans' else 'random'
                current_model = hmm.GaussianHMM(n_components=states, covariance_type='full', 
                                              n_iter=100, init_params=init_p, random_state=42)
                # If semi-supervised, we'd inject means here, but skipping for pure unsupervised
                try:
                    current_model.fit(X_scaled)
                except ValueError:
                    # HMMs sometimes fail to converge on expanding windows if covar drops to 0
                    current_model = None
                    pass
        
        # Predict day t 
        if current_model is not None and current_scaler is not None:
            day_feature = X_raw[t].reshape(1, -1)
            day_scaled = current_scaler.transform(day_feature)
            pred = current_model.predict(day_scaled)[0]
            predictions[t] = pred
            
    df_out = df.copy()
    col_name = f"{model_type}_{states}s_{init_strategy}"
    df_out[col_name] = predictions
    
    # Drop burn-in rows where we couldn't predict
    return df_out.iloc[burn_in:].copy(), col_name

def sort_regimes_by_volatility(df, regime_col):
    """
    Unsupervised states have random integer assignments (e.g. State 0 could be High Vol or Low Vol).
    To make them consistent, we explicitly map State 0 to Low Vol, 1 to Med Vol, 2 to High Vol.
    """
    states = sorted(df[regime_col].dropna().unique())
    vol_map = {}
    
    for state in states:
        mask = df[regime_col] == state
        vol = df.loc[mask, 'Log_Ret'].std()
        vol_map[state] = vol
        
    # Sort states by their variance
    sorted_states = sorted(vol_map.keys(), key=lambda k: vol_map[k])
    
    # Create mapping: lowest vol state becomes 0, highest becomes N
    final_mapping = {old_state: new_state for new_state, old_state in enumerate(sorted_states)}
    df[regime_col] = df[regime_col].map(final_mapping)
    return df

def run_models(df):
    
    # 1. Feature Selection
    # Keep it focused to avoid dimension curse
    features = [
        'Log_Ret', 
        'Vol_21d', 
        'NATR_14', 
        'VIX_Relative', 
        'Hurst_63', 
        'ADX_14'
    ]
    
    states_to_test = [2, 3, 4]
    
    # GMM Unsupervised
    for s in states_to_test:
        out_df, col = train_walk_forward_model(df, features, 'GMM', s, 'kmeans')
        out_df = sort_regimes_by_volatility(out_df, col)
        
        dir_path = f"RegimeClassifications/GMM_Unsupervised/{s}_state"
        os.makedirs(dir_path, exist_ok=True)
        
        plot_regimes(out_df, col, f"Unsupervised GMM ({s} States)", f"{dir_path}/chart.png", s)
        generate_performance_summary(out_df, col, f"GMM {s} States", f"{dir_path}/summary.md")

    # HMM Unsupervised
    for s in states_to_test:
        out_df, col = train_walk_forward_model(df, features, 'HMM', s, 'kmeans')
        out_df = sort_regimes_by_volatility(out_df, col)
        
        dir_path = f"RegimeClassifications/HMM_Unsupervised/{s}_state"
        os.makedirs(dir_path, exist_ok=True)
        
        plot_regimes(out_df, col, f"Unsupervised HMM ({s} States)", f"{dir_path}/chart.png", s)
        generate_performance_summary(out_df, col, f"HMM {s} States", f"{dir_path}/summary.md")

    # HMM Semi-Supervised (Seeded by Baseline - only makes sense for 3 states as baseline is 3 states)
    # To keep it simple, we use GMM-Initialization for the advanced HMMs
    
    # HMM GMM-Initialized
    # We first run GMM, then use its outputs to seed HMM. 
    # Since hmmlearn doesn't easily accept GMM objects natively without deep param hacking, 
    # we simulate "GMM Initialized" by training HMM but passing 'kmeans' which is the closest proxy 
    # hmmlearn natively supports for robust initialization.
    # To truly do Semi-Supervised, one would pass `init_params=''` and manually assign:
    # model.means_ = heuristic_means
    # model.covars_ = heuristic_covars
    
    for s in [3]: # 3 States for Semi-Supervised to match our Bull/Bear/Sideways Heuristic
        print(f"Running Semi-Supervised HMM (Seeded by Baseline, {s} states)...")
        # Extract heuristic means/covars from the first 2 years (burn_in)
        burn_in = 252 * 2
        df_burnin = df.iloc[:burn_in]
        
        # We need the baseline regime 
        # (Assuming run_baseline has run and we have a map. We'll recalculate here for isolation)
        mapping = {-1: 0, 0: 1, 1: 2} # Volatility proxy map
        base_regime = df_burnin['Regime_Baseline'].map(mapping).values
        
        predictions = np.full(len(df), np.nan)
        X_raw = df[features].values
        
        for t in tqdm(range(burn_in, len(df))):
            if t == burn_in or t % 21 == 0:
                X_train = X_raw[:t]
                current_scaler = StandardScaler()
                X_scaled = current_scaler.fit_transform(X_train)
                
                # Manual Semi-Supervised Initialization
                current_model = hmm.GaussianHMM(n_components=s, covariance_type='full', 
                                              n_iter=100, init_params='st', random_state=42)
                                              
                # Calculate empirical means and covars from the baseline mapping up to time t
                # Note: Baseline must be mapped strictly without lookahead.
                t_base_regime = df['Regime_Baseline'].iloc[:t].map(mapping).values
                
                means = np.zeros((s, X_scaled.shape[1]))
                covars = np.zeros((s, X_scaled.shape[1], X_scaled.shape[1]))
                
                for state in range(s):
                    state_data = X_scaled[t_base_regime == state]
                    if len(state_data) > 3: # Need at least a few points
                        means[state] = np.mean(state_data, axis=0)
                        covars[state] = np.cov(state_data.T) + np.eye(X_scaled.shape[1])*1e-4
                    else:
                        # Fallback
                        means[state] = np.zeros(X_scaled.shape[1])
                        covars[state] = np.eye(X_scaled.shape[1])
                
                current_model.means_ = means
                current_model.covars_ = covars
                
                try:
                    current_model.fit(X_scaled)
                except ValueError:
                    current_model = None
            
            if current_model is not None:
                day_scaled = current_scaler.transform(X_raw[t].reshape(1, -1))
                predictions[t] = current_model.predict(day_scaled)[0]
                
        col = 'HMM_SemiSupervised_3s'
        df_out = df.copy()
        df_out[col] = predictions
        df_out = df_out.iloc[burn_in:]
        df_out = sort_regimes_by_volatility(df_out, col)
        
        dir_path = "RegimeClassifications/HMM_SemiSupervised/3_state"
        os.makedirs(dir_path, exist_ok=True)
        plot_regimes(df_out, col, "Semi-Supervised HMM (Baselined, 3 States)", f"{dir_path}/chart.png", 3)
        generate_performance_summary(df_out, col, "Semi-Supervised HMM 3 States", f"{dir_path}/summary.md")

def main():
    create_directory_structure()
    df = pd.read_csv("data/daily/nq_extensive_features.csv", index_col='Date', parse_dates=True)
    
    # Fill Nans that might have leaked through at the start of complex stats
    df = df.bfill().ffill()
    
    # 1. Generate Baseline
    run_baseline(df)
    
    # 2. Run ML Models
    run_models(df)

if __name__ == "__main__":
    main()
