import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DiffusionReturnPrediction(nn.Module):
    def __init__(self, score_net, input_dim, lstm_hidden, gnn_hidden, n_hubs=14, n_out=8, use_diffusion=True, include_denoised=False):
        """
        Deep Spatiotemporal Graph Neural Network with Score-Augmented features.
        
        Args:
            score_net: The diffusion score model used to estimate the gradient of the log-density.
            input_dim: Dimension of the raw meteorological features (e.g., 4).
            lstm_hidden: Hidden dimension for the temporal LSTM.
            gnn_hidden: Hidden dimension for the spatial GCN.
            include_denoised: If True, uses Tweedie's Formula to include cleaned state features.
        """
        super().__init__()
        self.score_net = score_net
        self.use_diffusion = use_diffusion
        self.n_hubs = n_hubs
        self.n_out = n_out
        self.include_denoised = include_denoised
        
        # Combined Input Dimension: [Raw + Score + Optional Denoised]
        # Multiplier depends on whether denoised (Tweedie) features are active.
        self.combined_in_dim = input_dim * (3 if include_denoised else 2)
        
        # Temporal Component (recalibrated for the concatenated input dimension)
        self.lstm = nn.LSTM(input_size=self.combined_in_dim, hidden_size=lstm_hidden, batch_first=True)
        
        # Spatial Component
        self.gnn = GCNConv(in_channels=lstm_hidden, out_channels=gnn_hidden)
        
        # Prediction Head (MC-Dropout enabled)
        self.shared_mlp = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.Dropout(p=0.2), 
            nn.SiLU(),
            nn.Linear(gnn_hidden // 2, 1) 
        )
        
        # Map 14 Hubs -> 8 Commodities
        self.spatial_pool = nn.Linear(n_hubs, n_out)

    def enable_dropout(self):
        """Enable MC-Dropout during inference."""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def extract_features(self, x_combined, edge_index):
        """
        x_combined: [B, N, T, F_total=12]
        """
        B, N, T, F_total = x_combined.shape
        # Process all hubs in parallel: [B*N, T, 12]
        x_in = x_combined.reshape(B * N, T, F_total)
        _, (h_n, _) = self.lstm(x_in)
        h_t = h_n[-1].reshape(B, N, -1) # [B, N, H_lstm]
        
        gnn_outs = []
        for i in range(B):
            gnn_outs.append(self.gnn(h_t[i], edge_index))
        return torch.stack(gnn_outs, dim=0) # [B, N, H_gnn]

    def forward(self, x, edge_index):
        """
        x: Raw exogenous weather [B, T=180, N=14, F=4]
        """
        # Transpose to canonical [B, N, T, F]
        x = x.transpose(1, 2)
        B, N, T, F = x.shape
        
        if self.use_diffusion:
            # define state
            x_flat = x.reshape(B, -1)
            sigma_low = 0.1 # High resolution
            sigma = torch.full((B,), sigma_low, device=x.device)
            
            # generate scores
            scores_flat = self.score_net(x_flat, sigma)
            scores = scores_flat.reshape(B, N, T, F)
            
            # concatenate features
            if self.include_denoised:
                # Tweedie's Formula (Empirical Bayes): x_0 ≈ x + sigma^2 * ∇ log p(x)
                # This approximates the most likely 'clean' version of the noisy weather data.
                denoised = x + (sigma_low**2) * scores
                x_combined = torch.cat([x, denoised, scores], dim=-1)
            else:
                # Direct Score-Augmented path: Concatenate raw and score features.
                x_combined = torch.cat([x, scores], dim=-1)
        else:
            # Baseline compatibility: Zero-padded pseudo-signals
            x_combined = torch.cat([x, torch.zeros_like(x)], dim=-1)

        # Spatiotemporal Information Processing
        features = self.extract_features(x_combined, edge_index)
        
        # Alpha Mapping
        out_hubs = self.shared_mlp(features).squeeze(-1) 
        return self.spatial_pool(out_hubs)
