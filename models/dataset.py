import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
try:
    from market_data import get_real_commodity_returns
except ImportError:
    get_real_commodity_returns = None

class CommodityWeatherDataset(Dataset):
    """
    Mock Dataset for Commodity Weather data simulating a continuous temporal sequence.
    """
    def __init__(self, continuous_steps=1000, N=8, window_size=180, F=5, target_horizon=30):
        super().__init__()
        self.continuous_steps = continuous_steps
        self.N = N
        self.window_size = window_size
        self.F = F
        self.target_horizon = target_horizon
        
        # Simulate continuous global weather timeline: [Days, N, Features]
        self.global_timeline = torch.randn(continuous_steps, N, F)
        
        # Real Market Returns integration (V1.5)
        if get_real_commodity_returns:
            print("🚀 Fetching REAL historical market returns for backtesting...")
            # Fetch for the entire simulation period
            # v1.11: Now using both targets (30d) and valuations (1d)
            m_targets, m_daily, B_targets, B_daily, m_dates = get_real_commodity_returns(target_horizon=target_horizon)
            
            # Align simulation length to available market data
            available_steps = len(m_targets)
            self.continuous_steps = min(continuous_steps, available_steps)
            
            # Use real returns (convert to torch)
            self.global_returns = torch.tensor(m_targets.values[:available_steps], dtype=torch.float32)
            self.global_daily_returns = torch.tensor(m_daily.values[:available_steps], dtype=torch.float32)
            self.dates = m_dates[:available_steps]
            
            # Master Sync: Anchor Weather Signal to the Trading Calendar (approx 252 days/yr)
            self.global_timeline = self.global_timeline[:available_steps]
            self.continuous_steps = available_steps
        else:
            # Fallback to Mock simulation
            self.global_returns = torch.randn(continuous_steps, N) * 0.05
            self.global_daily_returns = torch.randn(continuous_steps, N) * 0.005 # Smaller daily vol
            self.dates = pd.date_range(start='2020-01-01', periods=continuous_steps, freq='D')
            
        self.valid_indices = self.continuous_steps - window_size - target_horizon
        
    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        # The 180-day window ends at t = idx + window_size
        start_t = idx
        end_t = idx + self.window_size
        
        weather_history = self.global_timeline[start_t:end_t] # [180, N, F] -> requires permuting to [N, 180, F]
        weather_history = weather_history.transpose(0, 1)     # Now [N, 180, F]
        
        # Target is the cumulative return exactly `target_horizon` days after the window ends
        # (30-day forward return for training)
        target_returns = self.global_returns[end_t] # We want the target at the precise moment the window closes
        
        # Valuation is the 1-day arithmetic return for the date the window ends
        daily_valuation = self.global_daily_returns[end_t]
        
        # Also return the date at which the window ends (for backtesting x-axis)
        end_date = str(self.dates[end_t])
        
        return weather_history, target_returns, daily_valuation, end_date

def get_dataloaders(batch_size=16, N=8, window_size=180, F=5, target_horizon=30, use_embargo=True):
    # Total days simulated: ~ 4 years
    total_days = 1500  
    base_ds = CommodityWeatherDataset(continuous_steps=total_days, N=N, window_size=window_size, F=F, target_horizon=target_horizon)
    
    # We have valid_indices corresponding to the sequential ends of the 180-day windows.
    # We must chronologically split them.
    valid_len = len(base_ds)
    
    # E.g., 60% Train, 20% Val, 20% Test
    train_end = int(valid_len * 0.6)
    
    # If use_embargo=True, shift the start of Val/Test by the target horizon (30 days)
    gap = target_horizon if use_embargo else 0
    
    val_start = train_end + gap
    val_end = val_start + int(valid_len * 0.2)
    
    test_start = val_end + gap
    test_end = valid_len
    
    train_idx = list(range(0, train_end))
    val_idx = list(range(val_start, val_end))
    test_idx = list(range(test_start, test_end))

    # Overlapping windows are allowed within Train (shuffled SGD batch is fine),
    # but strictly evaluating out-of-embargo for Val/Test.
    train_loader = DataLoader(Subset(base_ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(base_ds, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(base_ds, test_idx), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_l, val_l, test_l = get_dataloaders()
    x, y = next(iter(train_l))
    print(f"Weather Batch: {x.shape}")    
    print(f"Target Returns: {y.shape}")
