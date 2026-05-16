# ============================================================
# Week 2: Data Preprocessing & Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import os

print("=" * 50)
print("Week 2 - Data Preprocessing & Feature Engineering")
print("=" * 50)


# ============================================================
# 1. Load Data (from Week 1)
# ============================================================
print("\n[1] Loading raw dataset...")

# Ensure the path matches your Week 1 output
data = pd.read_csv("data/raw_dataset.csv")

# Convert Date to datetime objects for time-series operations
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date to maintain chronological order for rolling windows
data = data.sort_values("Date")

print(f" Loaded dataset shape: {data.shape}")


# ============================================================
# 2. Data Cleaning
# ============================================================
print("\n[2] Data Cleaning...")

# ---- Missing Values ----
# Using linear interpolation for price/rates and forward-fill for remaining gaps
print(" Handling missing values...")
data = data.interpolate(method="linear")
data = data.ffill()

# ---- Outlier Removal (IQR) ----
print(" Removing outliers (IQR)...")

# We apply IQR to price and rates, but EXCLUDE 'Dividend' 
# because dividends are sparse (mostly 0) and will be flagged as outliers incorrectly.
cols_to_filter = ["JPM_Close", "VIX_Close", "Treasury_3M", "Treasury_10Y"]

Q1 = data[cols_to_filter].quantile(0.25)
Q3 = data[cols_to_filter].quantile(0.75)
IQR = Q3 - Q1

# Define bounds: 1.5 * IQR is the standard statistical fence
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataframe
mask = ~((data[cols_to_filter] < lower_bound) | (data[cols_to_filter] > upper_bound)).any(axis=1)
data = data[mask]

print(f" After cleaning shape: {data.shape}")


# ============================================================
# 3. Feature Engineering
# ============================================================
print("\n[3] Feature Engineering...")

# ---- 3.1 Traditional Financial Features ----

# Log Returns: More statistically normalized than simple percentage change
data["log_return"] = np.log(data["JPM_Close"] / data["JPM_Close"].shift(1))

# Rolling Volatility (21 trading days ~ 1 month): Standard deviation of log returns
data["rolling_vol_21"] = data["log_return"].rolling(window=21).std()

# Dividend Features: Since Dividends are quarterly (sparse), we calculate the 1-year rolling sum
# This represents the total dividends paid over the trailing 12 months (252 trading days)
data["annual_dividend"] = data["Dividend"].rolling(window=252, min_periods=1).sum()

# Dividend Yield: Annual Dividend / Current Price
data["div_yield"] = data["annual_dividend"] / data["JPM_Close"]


# ---- 3.2 Macro & Sentiment Features ----

# Yield Curve Slope: 10Y minus 3M (a classic recession predictor)
data["yield_slope"] = data["Treasury_10Y"] - data["Treasury_3M"]

# Interest Rate Momentum: Change in the 10Y Treasury rate from previous day
data["rate_momentum"] = data["Treasury_10Y"].diff()

# VIX-JPM Correlation: 21-day rolling correlation between fear index and stock price
data["vix_jpm_corr"] = data["JPM_Close"].rolling(21).corr(data["VIX_Close"])

# Sentiment Score: Min-Max Normalization of VIX (scaled between 0 and 1)
vix_min = data["VIX_Close"].min()
vix_max = data["VIX_Close"].max()
data["sentiment_score"] = (data["VIX_Close"] - vix_min) / (vix_max - vix_min)


# ============================================================
# 4. Final Cleanup
# ============================================================
print("\n[4] Final cleaning (drop NaN from rolling windows)...")

# Rolling windows (21 and 252 days) create NaNs at the beginning of the dataset
data = data.dropna()

print(f" Final dataset shape: {data.shape}")


# ============================================================
# 5. Save Clean Dataset
# ============================================================
print("\n[5] Saving cleaned dataset...")

os.makedirs("data", exist_ok=True)
data.to_csv("data/cleaned_dataset.csv", index=False)

print(" Saved to data/cleaned_dataset.csv")


# ============================================================
# Done
# ============================================================
print("\n" + "=" * 50)
print(" Week 2 Completed Successfully")
print("=" * 50)