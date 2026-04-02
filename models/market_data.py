import yfinance as yf
import pandas as pd
import numpy as np
import os

def get_real_commodity_returns(start_date="2020-01-01", end_date="2024-04-01", target_horizon=30):
    """
    Fetches real historical price data and computes target horizon returns.
    """
    tickers = {
        "Corn": "ZC=F",
        "Soybeans": "ZS=F",
        "Wheat": "ZW=F",
        "Cattle": "LE=F",
        "Hogs": "HE=F",
        "Ethanol": "ZC=F", # ETH=F is illiquid (276 nulls); using ZC=F (Corn) as proxy
        "NatGas": "NG=F",
        "Cotton": "CT=F"
    }
    benchmarks = {
        "DBA": "DBA", # Invesco DB Agriculture
        "GSG": "GSG"  # iShares GSCI Commodity
    }
    
    all_tickers = {**tickers, **benchmarks}
    print(f"📥 Downloading market data for {len(all_tickers)} tickers...")
    
    data = yf.download(list(all_tickers.values()), start=start_date, end=end_date)['Close']
    
    # Forward-fill and drop any completely empty columns
    data = data.ffill().dropna(axis=1, how='all')
    
    # Re-align to our specific 8 tickers to maintain order
    commodity_cols = [tickers[c] for c in ["Corn", "Soybeans", "Wheat", "Cattle", "Hogs", "Ethanol", "NatGas", "Cotton"]]
    # If a ticker is missing (e.g. ETH=F fails), use a reasonable proxy or zero-pad
    for col in commodity_cols:
        if col not in data.columns:
            print(f"⚠️ Missing ticker {col}, filling with zeros.")
            data[col] = 0.0
            
    # Compute Real-World Daily Returns for Valuation (Wealth Path)
    # This is for the backtester to calculate $1 investment growth.
    daily_returns_df = data.pct_change(1)
    
    # Compute Arithmetic Returns for the target horizon (30-day forward looking)
    # Target(t) = (Price(t+30) - Price(t)) / Price(t)
    # Use pct_change followed by a reverse shift to get the 'future' return at time 't'
    target_returns_df = data.pct_change(periods=target_horizon).shift(-target_horizon)
    
    # Aggressive NaN Management (V1.6)
    # 1. Drop the 30-day edge-case NaNs at the end
    # 2. Drop any row containing any NaN across all 10 tickers
    # This ensures a strictly continuous and non-poisoned training set
    full_clean_mask = target_returns_df.notnull().all(axis=1) & daily_returns_df.notnull().all(axis=1)
    target_returns_df = target_returns_df[full_clean_mask]
    daily_returns_df = daily_returns_df[full_clean_mask]
    
    # Also align the original prices for backtesting
    # We want to return the 30-day forward returns for benchmarks too
    bench_targets_df = target_returns_df[["DBA", "GSG"]]
    bench_daily_df = daily_returns_df[["DBA", "GSG"]]
    
    return target_returns_df[commodity_cols], daily_returns_df[commodity_cols], bench_targets_df, bench_daily_df, target_returns_df.index

if __name__ == "__main__":
    ret, bench, dates = get_real_commodity_returns()
    print(ret.tail())
    print(bench.tail())
