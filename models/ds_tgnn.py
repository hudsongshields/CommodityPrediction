import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DiffusionReturnPrediction(nn.Module):
    def __init__(self, score_net, input_dim, lstm_hidden, gnn_hidden, n_hubs=14, n_out=8, use_diffusion=True, include_denoised=False):
        """
        V2.3 Score-Hardened DS-TGNN.
        Supports 8-dim (Raw+Score) and 12-dim (Raw+Clean+Score) docking.
        """
        super().__init__()
        self.score_net = score_net 
        self.use_diffusion = use_diffusion
        self.n_hubs = n_hubs
        self.n_out = n_out
        self.include_denoised = include_denoised
        
        # Combined Input Dimension: [Raw(4) + Score(4) + Optional Denoised(4)]
        self.combined_in_dim = input_dim * (3 if include_denoised else 2)
        
        # Temporal Component (Recalibrated for dynamic input dimension)
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
            # 1. State Estimation
            x_flat = x.reshape(B, -1)
            sigma_low = 0.1 # High-fidelity signal resolution
            t_const = torch.full((B, 1), sigma_low, device=x.device)
            
            # 2. Score Signal Generation: Gradient of log-density \nabla_x log p(x)
            # The ScoreNetwork is trained to output the score s_theta(x, sigma) directly.
            scores_flat = self.score_net(x_flat, t_const)
            scores = scores_flat.reshape(B, N, T, F)
            
            # 3. Input Signal Docking: [B, N, T, combined_dim]
            if self.include_denoised:
                # V2.3 Hardened: Include Tweedie-style denoised projection (Empirical Bayes)
                # Formula: \hat{x}_0 = x + sigma^2 * \nabla log p(x)
                denoised = x + (sigma_low**2) * scores
                x_combined = torch.cat([x, denoised, scores], dim=-1)
            else:
                # V2.3 Score-Only path (Direct Score Features)
                x_combined = torch.cat([x, scores], dim=-1)
        else:
            # Baseline compatibility: Zero-padded pseudo-signals
            x_combined = torch.cat([x, torch.zeros_like(x)], dim=-1)

        # Spatiotemporal Information Processing
        features = self.extract_features(x_combined, edge_index)
        
        # Alpha Mapping
        out_hubs = self.shared_mlp(features).squeeze(-1) 
        return self.spatial_pool(out_hubs)
