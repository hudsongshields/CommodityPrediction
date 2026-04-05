import yfinance as yf
import pandas as pd
import numpy as np
import os
import time


class YahooFetchError(RuntimeError):
    """Raised when Yahoo data is incomplete after a fetch attempt."""

def get_real_commodity_returns(start_date="2020-01-01", end_date="2024-04-01", target_horizon=30):
    """
    Fetches historical market data from Yahoo Finance and calculates percentage returns.
    
    Args:
        start_date: The beginning of the historical data window.
        end_date: The end of the historical data window.
        target_horizon: The number of days used to calculate forward-looking returns.
        
    Returns:
        A tuple containing (Target Returns, Daily Returns, Benchmark Targets, Benchmark Daily, Index).
    """
    # Mapping of human-readable labels to Yahoo Finance tickers.
    tickers = {
        "Corn": "ZC=F",
        "Soybeans": "ZS=F",
        "Wheat": "ZW=F",
        "Cattle": "LE=F",
        "Hogs": "HE=F",
        "Ethanol": "ZC=F", # Using Corn as a liquid proxy for Ethanol.
        "NatGas": "NG=F",
        "Cotton": "CT=F"
    }
    benchmarks = {
        "DBA": "DBA", # Invesco DB Agriculture Fund
        "GSG": "GSG"  # iShares GSCI Commodity ETF
    }
    
    all_tickers = {**tickers, **benchmarks}
    print(f"Downloading market records for {len(all_tickers)} tickers...")

    commodity_cols = [tickers[c] for c in ["Corn", "Soybeans", "Wheat", "Cattle", "Hogs", "Ethanol", "NatGas", "Cotton"]]
    benchmark_cols = ["DBA", "GSG"]
    required_cols = list(dict.fromkeys(commodity_cols + benchmark_cols))

    # Use the same retry structure as weather fetching: retry_count + success flag.
    data = None
    max_retries = 4

    def _download_close_frame():
        raw = yf.download(
            list(all_tickers.values()),
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        frame = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()

        if frame is None or frame.empty:
            raise YahooFetchError("Yahoo returned an empty price frame.")

        frame = frame.ffill().dropna(axis=1, how='all')
        missing_cols = [col for col in required_cols if col not in frame.columns]
        if missing_cols:
            raise YahooFetchError(f"Yahoo response missing tickers: {missing_cols}")

        return frame

    retry_count = 0
    success = False
    last_error = None
    while retry_count < max_retries and not success:
        try:
            data = _download_close_frame()
            success = True
        except Exception as exc:
            last_error = exc
            retry_count += 1
            if retry_count >= max_retries:
                break

            wait_time = 2 ** retry_count
            print(
                f"Yahoo fetch failed (attempt {retry_count}/{max_retries}): {exc}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    if not success:
        raise RuntimeError(
            f"Yahoo fetch failed after {max_retries} attempts. Last error: {last_error}"
        )
    
    # Enforce strict completeness after retries.
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise RuntimeError(
            "Missing required Yahoo tickers after retries: "
            f"{missing_cols}. Refusing to continue with synthetic placeholders. "
            "Please retry later or reduce request frequency."
        )
            
    # Calculate daily percentage returns for wealth path and valuation metrics.
    daily_returns_df = data.pct_change(1)
    
    # Calculate forward-looking arithmetic returns for the target horizon.
    # Formula: Return(t) = [Price(t + n) - Price(t)] / Price(t)
    target_returns_df = data.pct_change(periods=target_horizon).shift(-target_horizon)
    
    # Remove overlapping windows at the end of the time series and filter for consistency.
    # This prevents training on incomplete return windows.
    full_clean_mask = target_returns_df.notnull().all(axis=1) & daily_returns_df.notnull().all(axis=1)
    target_returns_df = target_returns_df[full_clean_mask]
    daily_returns_df = daily_returns_df[full_clean_mask]
    
    # Synchronize benchmark targets for outperformance (Alpha) calculation.
    bench_targets_df = target_returns_df[benchmark_cols]
    bench_daily_df = daily_returns_df[benchmark_cols]
    
    return (target_returns_df[commodity_cols], 
            daily_returns_df[commodity_cols], 
            bench_targets_df, 
            bench_daily_df, 
            target_returns_df.index)

if __name__ == "__main__":
    ret, daily, bench_t, bench_d, dates = get_real_commodity_returns()
    print("Latest Target Returns (30-day forward):")
    print(ret.tail())
