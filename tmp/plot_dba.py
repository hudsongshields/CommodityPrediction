import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_dba_price_index():
    results_path = 'results/ds_tgnn_detailed_results.npz'
    if not os.path.exists(results_path):
        print(f"❌ Results file not found at {results_path}")
        return

    data = np.load(results_path, allow_pickle=True)
    test_dates = data.get('test_dates', None)

    if test_dates is None:
        print("❌ Test dates missing from the results file.")
        return

    # Convert to datetime and get start/end for downloading
    test_dates = pd.to_datetime(test_dates)
    start_date = test_dates.min().strftime('%Y-%m-%d')
    end_date = (test_dates.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"📥 Downloading literal daily prices for DBA from {start_date} to {end_date}...")
    # Fetching only the 'Close' column explicitly
    dba_data = yf.download("DBA", start=start_date, end=end_date)['Close']
    
    # Ensure it's a 1D Series for indexing/division
    if isinstance(dba_data, pd.DataFrame):
        dba_data = dba_data.iloc[:, 0]
        
    dba_prices = dba_data.reindex(test_dates).ffill()
    
    # Normalize: $1 invested at the start of the test set
    price_index = dba_prices / dba_prices.iloc[0]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, price_index, color='orange', linewidth=2.5, label='Invesco DBA Fund Price Index')
    plt.fill_between(test_dates, 1.0, price_index, color='orange', alpha=0.08)
    
    plt.title('Institutional Asset Value: Historical $1 Investment in Invesco DBA Fund', fontsize=14, fontweight='bold')
    plt.xlabel('Date (2023-2024)', fontsize=12)
    plt.ylabel('Asset Value (Starting at $1.00)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.3) # Baseline
    plt.legend()
    
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    save_path = 'plots/dba_price_index.png'
    plt.savefig(save_path)
    print(f"✅ Standalone DBA Price-Index Plot saved to {save_path}")
    print(f"💰 Final Value: ${price_index.iloc[-1]:.4f}")

if __name__ == "__main__":
    plot_dba_price_index()
