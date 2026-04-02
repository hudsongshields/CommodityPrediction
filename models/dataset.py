import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from market_data import get_real_commodity_returns

class CommodityWeatherDataset(Dataset):
    def __init__(self, target_horizon=30):
        """
        V2.1 Core Fidelity: Finalizing for 14 available High-Impact Meteorological Hubs.
        Target: Excess 30-day Return Alpha over the DBA hurdle rate.
        Features: 14 Hubs (Des Moines to Odessa) x 4 Core Features.
        """
        # 1. Fetch Price Matrix & Benchmark
        m_targets, m_daily, bench_t, bench_d, m_dates = get_real_commodity_returns(target_horizon=target_horizon)
        
        # Calculate Alpha (Excess Return vs DBA)
        dba_targets = bench_t["DBA"].values.reshape(-1, 1)
        self.global_returns = torch.tensor(m_targets.values - dba_targets, dtype=torch.float32)
        self.daily_valuations = torch.tensor(m_daily.values, dtype=torch.float32)
        self.dates = m_dates
        
        # 2. Synchronize Real Weather Dataset (N=14 Hubs)
        weather_df = pd.read_csv('global_daily_weather.csv', parse_dates=['time'])
        
        # High-Fidelity Features: MaxTemp, MinTemp, Shortwave, Precip
        feat_cols = ['temperature_2m_max', 'temperature_2m_min', 'shortwave_radiation_sum', 'precipitation_sum']
        
        # Pivot by City (N=14)
        pivoted = weather_df.pivot(index='time', columns='city', values=feat_cols)
        
        # Synchronize Time & Ensure Matrix Integrity
        common_dates = pivoted.index.intersection(m_dates)
        weather_sync = pivoted.loc[common_dates].values # [T, 14*4]
        
        # Final Tensor: [T, 14, 4]
        self.global_timeline = torch.tensor(weather_sync, dtype=torch.float32).view(-1, 14, 4)
        
        # Re-align alpha targets
        mask = m_dates.isin(common_dates)
        self.global_returns = self.global_returns[mask]
        self.daily_valuations = self.daily_valuations[mask]
        self.dates = m_dates[mask]
        
        self.n_samples = len(self.dates)
        self.history_len = 180

    def __len__(self):
        return self.n_samples - self.history_len

    def __getitem__(self, idx):
        start, end = idx, idx + self.history_len
        return (
            self.global_timeline[start:end], # [180, 14, 4]
            self.global_returns[end-1],      # [8] Alpha Target
            self.daily_valuations[end-1],    # [8] Valuation context
            self.dates[end-1].strftime('%Y-%m-%d')
        )

def get_dataloaders(batch_size=16, use_embargo=True):
    ds = CommodityWeatherDataset()
    total = len(ds)
    tr_end, vl_end = int(total * 0.6), int(total * 0.8)
    gap = 30 # Standard Embargo
    
    return DataLoader(Subset(ds, range(0, tr_end - gap)), batch_size=batch_size, shuffle=True), \
           DataLoader(Subset(ds, range(tr_end, vl_end - gap)), batch_size=batch_size), \
           DataLoader(Subset(ds, range(vl_end, total)), batch_size=batch_size)

def get_walk_forward_dataloaders(batch_size=16, target_horizon=30, n_folds=5):
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
