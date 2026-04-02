import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DiffusionReturnPrediction(nn.Module):
    def __init__(self, denoiser, input_dim, lstm_hidden, gnn_hidden, use_diffusion=True, use_lstm=True, use_gnn=True):
        super().__init__()
        self.denoiser = denoiser 
        self.use_diffusion = use_diffusion
        self.use_lstm = use_lstm
        self.use_gnn = use_gnn
        
        # Temporal Encoder (LSTM)
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, batch_first=True)
        else:
            # If no LSTM, we use a simple linear projection to match the expected GNN/MLP input dim
            self.feature_proj = nn.Linear(input_dim, lstm_hidden)
        
        # Spatial Encoder (GNN)
        if self.use_gnn:
            self.gnn = GCNConv(in_channels=lstm_hidden, out_channels=gnn_hidden)
            mlp_in_dim = gnn_hidden
        else:
            mlp_in_dim = lstm_hidden
        
        # Shared MLP Prediction Head
        self.shared_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_in_dim // 2),
            nn.Dropout(p=0.2), # MC-Dropout injected here
            nn.SiLU(),
            nn.Linear(mlp_in_dim // 2, 1) # Outputs scalar excess return
        )

    def enable_dropout(self):
        """
        Helper method to force ONLY dropout layers into train mode during inference
        to support Monte-Carlo Uncertainty Sampling.
        """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def extract_features(self, x_cond, edge_index):
        """
        x_cond: meteorological sequence [B, N, T, F] (possibly denoised)
        edge_index: [2, E] constant graph
        """
        B, N, T, F = x_cond.shape
        
        # --- PHASE 1: TEMPORAL AGGREGATION ---
        if self.use_lstm:
            # Flatten B and N to run LSTM in parallel: [B*N, T, F]
            x_lstm_in = x_cond.view(B * N, T, F)
            lstm_out, (h_n, c_n) = self.lstm(x_lstm_in)
            # Take final hidden state per sequence: [B*N, H_lstm]
            h_temporal = h_n[-1] 
            node_features = h_temporal.view(B, N, -1) # [B, N, H_lstm]
        else:
            # Simple temporal pooling: Mean over time [B, N, F]
            x_mean = x_cond.mean(dim=2)
            node_features = self.feature_proj(x_mean) # Project to match dim: [B, N, H_lstm]

        # --- PHASE 2: SPATIAL COUPLING (GNN) ---
        if self.use_gnn:
            gnn_outs = []
            for i in range(B):
                # [N, H] -> [N, H_out]
                out_i = self.gnn(node_features[i], edge_index)
                gnn_outs.append(out_i)
            return torch.stack(gnn_outs, dim=0) # [B, N, H_gnn]
        else:
            # Skip GNN, return node features directly
            return node_features # [B, N, H_lstm]

    def forward(self, x, edge_index):
        """
        x: Input meteorological tensor [B, N, T, F]
        edge_index: Graph connectivity [2, E]
        """
        if self.use_diffusion:
            # Objective: Get single-step denoised expectation (t ~ 0.1)
            # This logic mirrors the transformation in the training controller
            B, N, T, F = x.shape
            x_flat = x.view(B, -1)
            
            # Using a fixed low noise level for deterministic expectation during prediction
            sigma_low = 0.1 
            
            # Prediction of noise component
            # Note: sigma_low is passed as a constant tensor for the denoiser's time-embedding
            t_const = torch.full((B, 1), sigma_low, device=x.device)
            z_pred = self.denoiser(x_flat, t_const)
            
            # Denoised representation (x_0 estimate)
            x_cond_flat = x_flat - z_pred * sigma_low
            x_input = x_cond_flat.view(B, N, T, F)
        else:
            # Use raw weather features directly
            x_input = x

        # [B, N, H_gnn]
        features = self.extract_features(x_input, edge_index)
        
        # --- Shared MLP ---
        out = self.shared_mlp(features)
        
        return out.squeeze(-1) # -> [B, N]
