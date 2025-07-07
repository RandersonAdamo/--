import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller, acf, pacf#, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf#, plot_ccf
import warnings
warnings.filterwarnings("ignore")
def zerosearch_autosarimax(series, seasonal_period=None, alpha=0.05, thresholds=[0.3, 6, 0.3, 0, 6, 0], verbose=1):
    """
    Calculates the SARIMAX (p, d, q)(P, D, Q, S) parameters automatically through statistical tests.

    Args:
        series (pd.Series): The time series data.
        seasonal_period (int): The seasonal period (e.g., 12 for monthly, 4 for quarterly).
                               If None, only non-seasonal ARIMA parameters will be determined.
        alpha (float): The significance level for the ADF test and for determining p, q, P, Q confidence intervals.
        thresholds (list): The threshold for each parameter (p_th, d_th, q_th, P_th, D_th, Q_th).
                            p_th: Correlation limit for selecting p lags in PACF.
                            d_th: Limit of differencing times (to avoid endless loop).
                            q_th: Correlation limit for selecting q lags in ACF.
                            P_th: Correlation limit for selecting (seasonal part) P lags in PACF.
                            D_th: Limit of differencing times (seasonal part) (to avoid endless loop).
                            Q_th: Correlation limit for selecting (seasonal part) Q lags in ACF.
                            Note:
                            For p_th, q_th, P_th, and Q_th, if the threshold used is None or less than the confidence interval for ACF/PACF, 
                            the confidence interval will be used.
        verbose (int):  verbose=0:  only the parameters tuple is returned.
                        verbose=1:  the statistical tests are printed.
                        verbose>=2: the ACF and PACF plots are exhibited.

    Returns:
        tuple: A tuple containing the determined (p, d, q, P, D, Q, S) values.
    """
    # Initialize seasonal orders
    P, D, Q, S = 0, 0, 0, seasonal_period
    p_th, d_th, q_th, P_th, D_th, Q_th = thresholds
    # --- 1. Find the Order of Non-Seasonal Differencing (d) ---
    d = 0
    if d_th < 5 or d_th == None: d_th = 5
    if D_th < 5 or D_th == None: D_th = 5
    d_th = int(d_th)
    D_th = int(D_th)
    temp_series_non_seasonal_diff = series.copy()
    while True:
        # Perform Augmented Dickey-Fuller test
        # adfuller requires at least 2 observations; ensure temp_series is not empty
        if len(temp_series_non_seasonal_diff.dropna()) < 2:
            if verbose > 0: print(f"Not enough data to perform ADF test after {d} non-seasonal differencing. Stopping.")
            break

        adf_test = adfuller(temp_series_non_seasonal_diff.dropna())
        p_value = adf_test[1]
        if verbose > 0: print(f"ADF Test on non-seasonal differenced series (d={d}): p-value = {p_value:.4f}")

        if p_value < alpha:
            if verbose > 0: print(f"Non-seasonal series is stationary at d={d}. Stopping.\n")
            break
        else:
            if verbose > 0: print(f"p-value > {alpha}. Differencing non-seasonally again.")
            d += 1
            if d > d_th: # Safety break to avoid infinite loops
                if verbose > 0: print("Exceeded max non-seasonal differencing order. Stopping.")
                break
            temp_series_non_seasonal_diff = temp_series_non_seasonal_diff.diff().dropna()
    
    current_stationary_series = temp_series_non_seasonal_diff.copy()

    # --- 2. Find the Order of Seasonal Differencing (D) if seasonal_period is provided ---
    if seasonal_period is not None and seasonal_period > 1:
        if verbose > 0: print(f"--- Finding Order of Seasonal Differencing (D) for S={seasonal_period} ---")
        temp_series_seasonal_diff = current_stationary_series.copy()
        D = 0
        while True:
            # Check if enough data for seasonal differencing and ADF test
            if len(temp_series_seasonal_diff.dropna()) < seasonal_period + 2: # Need enough data for diff and ADF
                if verbose > 0: print(f"Not enough data for seasonal differencing (D={D}) with period S={seasonal_period}. Stopping.")
                break

            # Apply seasonal differencing
            differenced_series = temp_series_seasonal_diff.diff(periods=seasonal_period).dropna()
            
            if len(differenced_series) < 2:
                if verbose > 0: print(f"Series too short after seasonal differencing (D={D}) for ADF test. Stopping.")
                break

            adf_test = adfuller(differenced_series)
            p_value = adf_test[1]
            if verbose > 0: print(f"ADF Test on seasonal differenced series (D={D}): p-value = {p_value:.4f}")

            if p_value < alpha:
                if verbose > 0: print(f"Seasonal series is stationary at D={D}. Stopping.\n")
                current_stationary_series = differenced_series
                break
            else:
                if verbose > 0: print(f"p-value > {alpha}. Differencing seasonally again.")
                D += 1
                if D > D_th: # Safety break for seasonal differencing
                    if verbose > 0: print("Exceeded max seasonal differencing order. Stopping.")
                    current_stationary_series = differenced_series # Use the last differenced series
                    break
                temp_series_seasonal_diff = differenced_series
    
    # Ensure the series is not empty after all differencing steps
    if current_stationary_series.empty:
        if verbose > 0: print("Final stationary series is empty after differencing. Cannot determine AR/MA orders. Returning (0, d, 0, 0, D, 0, S).")
        return (0, d, 0, 0, D, 0, S)

    # --- 3. Find the Autoregressive Order (p) and Seasonal Autoregressive Order (P) ---
    if verbose > 0: print("\n--- Finding Autoregressive Order (p) and Seasonal Autoregressive Order (P) ---")
    
    if verbose > 0: print(f"Length of current_stationary_series: {len(current_stationary_series)}")
    # Calculate the absolute maximum lag allowed by the sample size
    max_lags_allowed_by_sample = len(current_stationary_series) // 2 - 1

    # Determine max lags for ACF/PACF plots and calculations
    # Start with a sensible default of 40, but cap it at the sample size limit
    max_lags_for_plots = min(40, max_lags_allowed_by_sample)

    # If seasonal, we want to ensure we capture seasonal patterns up to 2*seasonal_period
    # However, this still must be capped by the overall sample size limit.
    if seasonal_period is not None and seasonal_period > 1:
        # We take the maximum between the non-seasonal heuristic and the desired seasonal lag,
        # but *then* we ensure this combined value does not exceed the sample-size limit.
        seasonal_consideration = 2 * seasonal_period + 5
        max_lags_for_plots = min(max(max_lags_for_plots, seasonal_consideration), max_lags_allowed_by_sample)

    # Ensure max_lags_for_plots is at least 1, provided there's enough data for at least one lag.
    if max_lags_allowed_by_sample < 1:
        max_lags_for_plots = 0 # Cannot compute any lags if series is too short
    
    if verbose > 0: print(f"Calculated max_lags_for_plots: {max_lags_for_plots}")

    # Calculate threshold '_th' values for significance, centered around zero
    # This is based on the standard error of the autocorrelation/partial autocorrelation: 1/sqrt(N)
    # where N is the effective number of observations (length of the stationary series)
    num_observations = len(current_stationary_series)
    if num_observations > 1: # Need at least 2 observations to calculate std err
        # Z-score for the given alpha level (two-tailed test)
        z_score = norm.ppf(1 - alpha / 2)
        for th in [p_th, q_th, P_th, Q_th]:
            th = z_score / np.sqrt(num_observations) if th == None or th < z_score / np.sqrt(num_observations) else th

    if None in [p_th, q_th, P_th, Q_th] or 0 in [p_th, q_th, P_th, Q_th]:
        if verbose > 0: print(f"Confidence band critical value (centered at zero): +/- {p_th:.4f}")


    if max_lags_for_plots < 1:
        if verbose > 0: print("Not enough data for PACF calculation. Setting p = 0, P = 0.\n")
        p = 0
        P = 0
    else:
        # Plot PACF
        if verbose > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_pacf(current_stationary_series, ax=ax, alpha=alpha, lags=max_lags_for_plots)
            plt.title(f'Partial Autocorrelation (PACF) of Fully Stationary Series (d={d}, D={D})')
            plt.xlabel('Lag')
            plt.ylabel('Partial Correlation')
            plt.grid(True)
            plt.show()

        # Programmatically determine p and P
        # Using method='ywm' for consistency with plot_pacf's default
        # No alpha here, as we are calculating critical value manually
        pacf_vals = pacf(current_stationary_series, nlags=max_lags_for_plots, method='ywm')
        
        if verbose > 0: print(f"PACF values (first 5): {pacf_vals[:5]}")
        # Note: 'confint' is not directly used from pacf/acf returns for significance checking here
        # Instead, we use the manually calculated 'critical_value'

        # Determine non-seasonal p
        # Look for the last significant non-seasonal lag before the first seasonal lag
        p_candidates = []
        for i in range(1, len(pacf_vals)):
            # Check if the PACF value is outside the manually calculated critical band
            is_significant = (pacf_vals[i] < -p_th or pacf_vals[i] > p_th)
            if is_significant:
                if seasonal_period and seasonal_period > 1 and i % seasonal_period == 0:
                    # This is a seasonal lag, handled by P
                    pass
                else:
                    p_candidates.append(i)
        p = p_candidates[-1] if p_candidates else 0
        if seasonal_period is not None and seasonal_period > 1:
            if p > seasonal_period:
                p = seasonal_period - 1
        if verbose > 0: print(f"Last significant non-seasonal lag in PACF: {p}. Setting p = {p}.")

        # Determine seasonal P
        P_candidates = []
        if seasonal_period is not None and seasonal_period > 1:
            for i in range(1, int(max_lags_for_plots / seasonal_period) + 1):
                seasonal_lag = i * seasonal_period
                if seasonal_lag < len(pacf_vals): # Ensure lag is within bounds
                    is_significant = (pacf_vals[seasonal_lag] < -P_th or pacf_vals[seasonal_lag] > P_th)
                    if is_significant:
                        P_candidates.append(i)
        P = P_candidates[-1] if P_candidates else 0
        if verbose > 0: print(f"Last significant seasonal lag in PACF (multiples of S={seasonal_period}): {P}. Setting P = {P}.\n")

    # --- 4. Find the Moving Average Order (q) and Seasonal Moving Average Order (Q) ---
    if verbose > 0: print("--- Finding Moving Average Order (q) and Seasonal Moving Average Order (Q) ---")

    if max_lags_for_plots < 1:
        if verbose > 0: print("Not enough data for ACF calculation. Setting q = 0, Q = 0.\n")
        q = 0
        Q = 0
    else:
        # Plot ACF
        if verbose > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_acf(current_stationary_series, ax=ax, alpha=alpha, lags=max_lags_for_plots)
            plt.title(f'Autocorrelation (ACF) of Fully Stationary Series (d={d}, D={D})')
            plt.xlabel('Lag')
            plt.ylabel('Correlation')
            plt.grid(True)
            plt.show()

        # Programmatically determine q and Q
        # No alpha here, as we are calculating critical value manually
        acf_vals = acf(current_stationary_series, nlags=max_lags_for_plots)
        
        if verbose > 0: print(f"ACF values (first 5): {acf_vals[:5]}")
        # Note: 'confint' is not directly used from pacf/acf returns for significance checking here
        # Instead, we use the manually calculated '_th'

        # Determine non-seasonal q
        q_candidates = []
        for i in range(1, len(acf_vals)):
            # Check if the ACF value is outside the manually calculated threshold band
            is_significant = (acf_vals[i] < -q_th or acf_vals[i] > q_th)
            if is_significant:
                if seasonal_period and seasonal_period > 1 and i % seasonal_period == 0:
                    # This is a seasonal lag, handled by Q
                    pass
                else:
                    q_candidates.append(i)
        q = q_candidates[-1] if q_candidates else 0
        if seasonal_period is not None and seasonal_period > 1:
            if q > seasonal_period:
                q = seasonal_period - 1
        if verbose > 0: print(f"Last significant non-seasonal lag in ACF: {q}. Setting q = {q}.\n")

        # Determine seasonal Q
        Q_candidates = []
        if seasonal_period is not None and seasonal_period > 1:
            for i in range(1, int(max_lags_for_plots / seasonal_period) + 1):
                seasonal_lag = i * seasonal_period
                if seasonal_lag < len(acf_vals): # Ensure lag is within bounds
                    is_significant = (acf_vals[seasonal_lag] < -Q_th or acf_vals[seasonal_lag] > Q_th)
                    if is_significant:
                        Q_candidates.append(i)
        Q = Q_candidates[-1] if Q_candidates else 0
        if verbose > 0: print(f"Last significant seasonal lag in ACF (multiples of S={seasonal_period}): {Q}. Setting Q = {Q}.\n")

    return (p, d, q), (P, D, Q, S)