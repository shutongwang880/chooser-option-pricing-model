# ============================================================
# Week 1: Data Collection Script
# ============================================================

import yfinance as yf
import pandas as pd
from fredapi import Fred
from config import ALPHA_VANTAGE_KEY, FRED_API_KEY
import requests
import os

# Define the timeframe for data collection
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

print("=" * 50)
print("Week 1 - Data Collection")
print("=" * 50)


# ============================================================
# 1. Yahoo Finance - JPM (with Dividends) & VIX
# ============================================================
print("\n[1] Downloading JPM data (including Dividends) and VIX...")

# Download JPM using Ticker object to access 'actions' (Dividends/Splits)
jpm_ticker = yf.Ticker("JPM")
jpm_history = jpm_ticker.history(start=START_DATE, end=END_DATE, actions=True)

# Select only Close and Dividends, then reset index to move 'Date' to a column
jpm = jpm_history[["Close", "Dividends"]].reset_index()
jpm.columns = ["Date", "JPM_Close", "Dividend"]

# Crucial: Remove timezone info to ensure seamless merging with FRED data
jpm['Date'] = jpm['Date'].dt.tz_localize(None)

# Download VIX Index data
vix_raw = yf.download("^VIX", start=START_DATE, end=END_DATE)
vix = vix_raw[["Close"]].reset_index()
vix.columns = ["Date", "VIX_Close"]

print(" Yahoo Finance data ready")


# ============================================================
# 2. FRED - Treasury Rates
# ============================================================
print("\n[2] Downloading Treasury rates from FRED...")

fred = Fred(api_key=FRED_API_KEY)

# Fetch 3-Month and 10-Year Treasury series
treasury_3m = fred.get_series("DTB3").to_frame(name="Treasury_3M")
treasury_10y = fred.get_series("DGS10").to_frame(name="Treasury_10Y")

# Slice data to match our specific timeframe
treasury_3m = treasury_3m.loc[START_DATE:END_DATE]
treasury_10y = treasury_10y.loc[START_DATE:END_DATE]

print(" FRED data ready")


# ============================================================
# 3. Alpha Vantage - Simple API Test
# ============================================================
print("\n[3] Testing Alpha Vantage API...")

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=JPM&apikey={ALPHA_VANTAGE_KEY}"

try:
    response = requests.get(url)
    api_data = response.json()
    if "Time Series (Daily)" in api_data:
        print(" Alpha Vantage API working")
    else:
        print(" Alpha Vantage response received but check structure/limit")
except Exception as e:
    print(f" Alpha Vantage failed: {e}")


# ============================================================
# 4. Merge Dataset
# ============================================================
print("\n[4] Merging datasets...")

# Reset index for FRED data to prepare for merging
treasury_3m = treasury_3m.reset_index()
treasury_10y = treasury_10y.reset_index()

# Rename columns to ensure consistency across all DataFrames
treasury_3m.columns = ["Date", "Treasury_3M"]
treasury_10y.columns = ["Date", "Treasury_10Y"]

# Convert all 'Date' columns to datetime objects to ensure alignment
for df in [jpm, vix, treasury_3m, treasury_10y]:
    df["Date"] = pd.to_datetime(df["Date"])

# Perform Left Merges using JPM as the primary reference table
data = jpm.merge(vix, on="Date", how="left")
data = data.merge(treasury_3m, on="Date", how="left")
data = data.merge(treasury_10y, on="Date", how="left")

print(f" Final dataset shape: {data.shape}")
print(f" Total Dividends found: {data['Dividend'].sum()}")


# ============================================================
# 5. Save Data
# ============================================================
print("\n[5] Saving dataset...")

# Create 'data' directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save to CSV without the default index column
data.to_csv("data/raw_dataset.csv", index=False)

print(" Saved to data/raw_dataset.csv")


# ============================================================
# Done
# ============================================================
print("\n" + "=" * 50)
print(" Week 1 Completed")
print("=" * 50)