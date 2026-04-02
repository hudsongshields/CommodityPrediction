import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DiffusionReturnPrediction(nn.Module):
    def __init__(self, denoiser, input_dim, lstm_hidden, gnn_hidden, n_hubs=14, n_out=8, use_diffusion=True):
        """
        V2.1.1 Triple-Signal Score-Conditioned Upgrade.
        Architecture: Triple-Input Encoder [Raw, Clean, Score].
        Target: Excess Return (Alpha) over the DBA benchmark.
        """
        super().__init__()
        self.denoiser = denoiser 
        self.use_diffusion = use_diffusion
        self.n_hubs = n_hubs
        self.n_out = n_out
        
        # Triple-Input Pathway (Combined_Dim = 3 * F = 12)
        # 1. Raw Weather (4)
        # 2. Denoised Weather (4)
        # 3. Manifold Score (4)
        self.combined_in_dim = input_dim * 3
        
        # Temporal Component (Recalibrated for 12-dim input)
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
            sigma_low = 0.1
            t_const = torch.full((B, 1), sigma_low, device=x.device)
            z_pred_flat = self.denoiser(x_flat, t_const)
            
            # 2. Reshape Denoising Signal
            z_pred = z_pred_flat.reshape(B, N, T, F)
            
            # 3. Component Generation
            # x_clean: The underlying "Normal" weather manifold projection
            # scores: The "Off-Manifold" surprise gradient (Residual)
            scores = z_pred * sigma_low
            x_clean = x - scores
            
            # 4. Triple-Signal Concatenation: [B, N, T, 12 features]
            x_combined = torch.cat([x, x_clean, scores], dim=-1)
        else:
            # Baseline compatibility: Zero-padded pseudo-signals
            zero_signal = torch.zeros_like(x)
            x_combined = torch.cat([x, zero_signal, zero_signal], dim=-1)

        # Spatiotemporal Information Processing
        features = self.extract_features(x_combined, edge_index)
        
        # Alpha Mapping
        out_hubs = self.shared_mlp(features).squeeze(-1) 
        return self.spatial_pool(out_hubs)
