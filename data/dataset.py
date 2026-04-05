import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import os
try:
    from .market_data import get_real_commodity_returns
except (ImportError, ValueError):
    from data.market_data import get_real_commodity_returns
except Exception:
    import market_data
    get_real_commodity_returns = market_data.get_real_commodity_returns

class CommodityWeatherDataset(Dataset):
    """
    Chronological dataset for commodity return prediction using meteorological features.
    
    This class synchronizes daily price returns with historical weather data 
    from a specified set of global hubs. It calculates excess returns (Alpha) 
    relative to the Invesco DB Agriculture (DBA) fund.
    """
    def __init__(self, target_horizon=30):
        # 1. Load commodity price data and the DBA benchmark.
        m_targets, m_daily, bench_t, bench_d, m_dates = get_real_commodity_returns(target_horizon=target_horizon)
        
        # Calculate Alpha: Excess Return = Realized Return - DBA Benchmark Return.
        dba_targets = bench_t["DBA"].values.reshape(-1, 1)
        self.global_returns = torch.tensor(m_targets.values - dba_targets, dtype=torch.float32)
        self.daily_valuations = torch.tensor(m_daily.values, dtype=torch.float32)
        self.dates = m_dates
        
        # 2. Synchronize historical weather data.
        # Required features: Maximum Temperature, Minimum Temperature, Shortwave Radiation, and Precipitation.
        weather_file = os.path.join(os.path.dirname(__file__), 'global_daily_weather.csv')
        weather_df = pd.read_csv(weather_file, parse_dates=['time'])
        feat_cols = ['temperature_2m_max', 'temperature_2m_min', 'shortwave_radiation_sum', 'precipitation_sum']
        
        # Selection of primary meteorological hubs for global commodity markets.
        core_hubs = [
            'Beijing', 'Chicago', 'Cuiaba', 'Dubai', 'Moscow', 
            'Paris', 'Pittsburgh', 'Rostov-on-Don', 'Singapore'
        ]
        weather_df = weather_df[weather_df['city'].isin(core_hubs)]
        
        # Pivot the data to create a consistent [Time x (Hubs * Features)] matrix.
        pivoted = weather_df.pivot(index='time', columns='city', values=feat_cols)
        pivoted = pivoted.reindex(columns=core_hubs, level=1) # Ensure consistent hub ordering.
        pivoted = pivoted.dropna() # Maintain a continuous time series.
        
        # Align weather data timeline with the commodity return dates.
        common_dates = pivoted.index.intersection(m_dates)
        weather_sync = pivoted.loc[common_dates].values 
        
        self.n_hubs = pivoted.shape[1] // 4
        self.global_timeline = torch.tensor(weather_sync, dtype=torch.float32).view(-1, self.n_hubs, 4)
        
        # Synchronize return targets to match the weather data availability.
        mask = m_dates.isin(common_dates)
        self.global_returns = self.global_returns[mask]
        self.daily_valuations = self.daily_valuations[mask]
        self.dates = m_dates[mask]
        
        self.n_samples = len(self.dates)
        self.history_len = 180 # Uses a 6-month lookback window.

    def __len__(self):
        return self.n_samples - self.history_len

    def __getitem__(self, idx):
        start, end = idx, idx + self.history_len
        return (
            self.global_timeline[start:end], 
            self.global_returns[end-1],      
            self.daily_valuations[end-1],    
            self.dates[end-1].strftime('%Y-%m-%d')
        )

def get_dataloaders(batch_size=16):
    """
    Splits the dataset into Training, Validation, and Testing sets using an embargo gap.
    
    The 'gap' ensures that data from the end of the training set does not 
    overlap with the start of the validation/test sets, preventing look-ahead leakage.
    """
    ds = CommodityWeatherDataset()
    total = len(ds)
    tr_end, vl_end = int(total * 0.6), int(total * 0.8)
    gap = 30 # Mandatory gap based on the 30-day target horizon.
    
    return DataLoader(Subset(ds, range(0, tr_end - gap)), batch_size=batch_size, shuffle=True), \
           DataLoader(Subset(ds, range(tr_end, vl_end - gap)), batch_size=batch_size), \
           DataLoader(Subset(ds, range(vl_end, total)), batch_size=batch_size)

def get_walk_forward_dataloaders(batch_size=16, target_horizon=30, n_folds=5):
    """
    Implements a Walk-Forward validation strategy with an embargo gap between sets.
    """
    ds = CommodityWeatherDataset(target_horizon=target_horizon)
    total = len(ds)
    initial_tr = int(total * 0.5)
    step = int((total - initial_tr) / n_folds)
    gap = 30
    
    folds = []
    for i in range(n_folds):
        tr_end = initial_tr + i * step
        vl_end, ts_end = tr_end + (step // 2), min(tr_end + step, total)
        
        folds.append({
            "fold": i + 1,
            "train": DataLoader(Subset(ds, range(0, tr_end - gap)), batch_size=batch_size, shuffle=True),
            "val": DataLoader(Subset(ds, range(tr_end, min(vl_end - gap, total))), batch_size=batch_size),
            "test": DataLoader(Subset(ds, range(vl_end, ts_end)), batch_size=batch_size)
        })
    return folds
