import torch
import torch.nn as nn
import torch.nn.functional as F
from base_mlps import MLP, ConvolutionalMLP

class ScoreNetwork(nn.Module):
    """
    Predicts the score gradient s_theta(x, sigma) for a given noise scale.
    
    The network approximates the gradient of the log-density of the data distribution, 
    providing structural features derived from the underlying data manifold.
    """
    def __init__(self, input_dim, mlp_hidden, conv_hidden, t_hidden_dim, output_dim, use_conv=False):
        super().__init__()
        # Time-conditional embedding to inform the model of the current noise scale.
        self.time_embed = nn.Sequential(
            nn.Linear(1, t_hidden_dim),
            nn.SiLU(),
            nn.Linear(t_hidden_dim, t_hidden_dim)
        )
        self.use_conv = use_conv
        
        if use_conv:
            self.score_regression = ConvolutionalMLP(
                input_channels=1, 
                hidden_channels=conv_hidden, 
                t_hidden_dim=t_hidden_dim
            )
        else:
            # Linear projector for integrating time embeddings into the input space.
            self.time_to_input = nn.Linear(t_hidden_dim, input_dim)
            self.score_regression = MLP(input_dim, mlp_hidden, output_dim)
    
    def forward(self, x, t: torch.Tensor):
        """
        Calculates the score prediction for input x at noise level t.
        
        Args:
            x: Input observations (batch_size, input_dim).
            t: Noise scale parameters (batch_size, 1).
            
        Returns:
            The predicted score gradient (batch_size, output_dim).
        """
        t = t.view(-1, 1)
        t_embed = self.time_embed(t)
            
        if not self.use_conv:
            x = x.reshape(x.size(0), -1)
            t_embed_proj = self.time_to_input(t_embed)
            # Additive integration of time-conditional signals.
            score = self.score_regression(x + t_embed_proj)
        else:
            # Spatial dimensions are derived assuming a square input grid (N_hubs x T_steps).
            spatial = int(x.size(1)**0.5)
            x = x.view(x.size(0), 1, spatial, spatial)
            score = self.score_regression(x, t_embed)

        return score