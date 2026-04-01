import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DiffusionReturnPrediction(nn.Module):
    def __init__(self, denoiser, input_dim, lstm_hidden, gnn_hidden):
        super().__init__()
        self.denoiser = denoiser 
        
        # Temporal Encoder (LSTM)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, batch_first=True)
        
        # Spatial Encoder (GNN)
        self.gnn = GCNConv(in_channels=lstm_hidden, out_channels=gnn_hidden)
        
        # Shared MLP Prediction Head
        self.shared_mlp = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden // 2),
            nn.Dropout(p=0.2), # MC-Dropout injected here
            nn.SiLU(),
            nn.Linear(gnn_hidden // 2, 1) # Outputs scalar excess return
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
        x_cond: single-step denoised weather expectation [B, N, T, F]
        edge_index: [2, E] constant graph
        """
        B, N, T, F = x_cond.shape
        
        # --- LSTM ---
        # Flatten B and N to run LSTM in parallel
        # [B*N, T, F]
        x_lstm_in = x_cond.view(B * N, T, F)
        
        # Output shape: [B*N, T, H]
        lstm_out, (h_n, c_n) = self.lstm(x_lstm_in)
        
        # Take final hidden state per sequence: [B*N, H]
        h_final = h_n[-1] 
        
        # --- GNN ---
        # Reshape to [B, N, H] 
        H = h_final.shape[-1]
        node_features = h_final.view(B, N, H)
        
        gnn_outs = []
        for i in range(B):
            # [N, H] -> [N, H_out]
            out_i = self.gnn(node_features[i], edge_index)
            gnn_outs.append(out_i)
            
        return torch.stack(gnn_outs, dim=0) # [B, N, H_gnn]

    def forward(self, x_cond, edge_index):
        # [B, N, H_gnn]
        gnn_out = self.extract_features(x_cond, edge_index)
        
        # --- Shared MLP ---
        # [B, N, H_gnn] -> [B, N, 1]
        out = self.shared_mlp(gnn_out)
        
        return out.squeeze(-1) # -> [B, N]
