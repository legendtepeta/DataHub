import numpy as np
from scipy.stats import entropy
from numba import jit

# @jit(nopython=True) # np.polyfit not supported in numba
def get_hurst_exponent(time_series, max_lag=20):
    """
    Calculates the Hurst Exponent of a time series.
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk
    H > 0.5: Trending
    """
    lags = range(2, max_lag)
    # Corrected: convert range to a numpy array for numba and math operations
    lags_arr = np.array([float(l) for l in lags])
    
    # Calculate the variance of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    
    # Store tau in a numpy array for polyfit
    tau_arr = np.array(tau)
    
    # Fit to a linear regression to find the Hurst exponent
    poly = np.polyfit(np.log(lags_arr), np.log(tau_arr), 1)
    
    return poly[0] * 2.0

def calculate_entropy(data, window=21):
    """
    Calculates Shannon Entropy on the distribution of returns.
    """
    if np.any(np.isnan(data)):
        return np.nan
    # Histogram of returns to get probabilities
    hist, _ = np.histogram(data, bins=10, density=True)
    hist = hist[hist > 0] # Remove zeros
    return entropy(hist)

def calculate_fractal_dimension(Z, threshold=0.9):
    """
    Estimates the fractal dimension of a price series (Box-counting idea).
    Simplified version for 1D time series.
    """
    # Only for 2d? Let's use a simpler Higuchi Fractal Dimension or similar
    # For now, let's use a proxy: (log(L) / log(d))
    # where L is total path length and d is displacement.
    L = np.sum(np.abs(np.diff(Z)))
    d = np.abs(Z[-1] - Z[0])
    if d == 0: return 2.0
    return np.log(L) / np.log(d)
