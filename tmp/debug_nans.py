import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["ZC=F", "ZS=F", "ZW=F", "LE=F", "HE=F", "ETH=F", "NG=F", "CT=F", "DBA"]

print("📥 Downloading...")
data = yf.download(tickers, start="2020-01-01", end="2024-04-01")['Close']
print(f"Total Rows: {len(data)}")

# Check Nulls per ticker
print("\nNull Counts per Ticker:")
print(data.isnull().sum())

# How many rows have ANY null?
any_null = data.isnull().any(axis=1).sum()
print(f"\nRows with ANY NULL: {any_null} out of {len(data)}")

# Let's perform the cleaning we have in market_data.py
clean_data = data.ffill().dropna()
print(f"\nRows after ffill + dropna: {len(clean_data)}")

# Check target_horizon = 30
returns = clean_data.pct_change(periods=30).shift(-30)
print(f"Returns Rows with NULL: {returns.isnull().any(axis=1).sum()}")

# FINAL CLEAN
final_returns = returns.dropna()
print(f"Final Count with 0 NaNs: {len(final_returns)}")
