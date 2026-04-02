import torch
from torch.utils.data import Dataset, DataLoader, Subset

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
        
        # Simulate continuous realized return timeline
        # A real dataset would align `returns[t+30]` as the target for the window ending at `t`
        self.global_returns = torch.randn(continuous_steps, N) * 0.05

        # Total valid ending indices for a 180-day window
        # We need enough space at the end to evaluate 30 days into the future.
        self.valid_indices = continuous_steps - window_size - target_horizon
        
    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        # The 180-day window ends at t = idx + window_size
        start_t = idx
        end_t = idx + self.window_size
        
        weather_history = self.global_timeline[start_t:end_t] # [180, N, F] -> requires permuting to [N, 180, F]
        weather_history = weather_history.transpose(0, 1)     # Now [N, 180, F]
        
        # Target is the cumulative return exactly `target_horizon` days after the window ends
        # (Mock representation of cumulative sum over the horizon)
        target_returns = self.global_returns[end_t : end_t + self.target_horizon].sum(dim=0)
        
        return weather_history, target_returns

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
