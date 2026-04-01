import torch
from torch.utils.data import Dataset, DataLoader

class CommodityWeatherDataset(Dataset):
    """
    Mock Dataset for Commodity Weather data.
    Generates dummy random data matching the shapes requested.
    Real implementation should load the actual CSVs/Parquet, window the 180 days, 
    and compute the cumulative 30-day excess returns.
    """
    def __init__(self, num_samples=100, N=8, T=180, F=5, target_horizon=30):
        super().__init__()
        self.num_samples = num_samples
        self.N = N
        self.T = T
        self.F = F
        self.target_horizon = target_horizon
        
        # [num_samples, N, T, F]
        # E.g. normalized weather features per region mapped to commodity node
        self.weather_history = torch.randn(num_samples, N, T, F)
        
        # [num_samples, N]
        # Target: 30-day cumulative excess return over benchmark
        self.excess_returns = torch.randn(num_samples, N) * 0.05  # roughly 5% vol

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.weather_history[idx], self.excess_returns[idx]


def get_dataloaders(batch_size=16, N=8, T=180, F=5):
    train_ds = CommodityWeatherDataset(num_samples=200, N=N, T=T, F=F)
    val_ds = CommodityWeatherDataset(num_samples=50, N=N, T=T, F=F)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    loader, _ = get_dataloaders()
    x, y = next(iter(loader))
    print(f"Weather Batch: {x.shape}")    # Expected: [B, 8, 180, 5]
    print(f"Target Returns: {y.shape}")   # Expected: [B, 8]
