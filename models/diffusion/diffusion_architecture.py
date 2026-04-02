from pyexpat import model

import torch.nn as nn
import torch
from reversal import reverse_sde
import torch.nn.functional as F
from base_mlps import MLP, ConvolutionalMLP

        
    
# Takes in a noisy input and predicts the score to remove the noise
class Diffusion(nn.Module):
    def __init__(self, input_dim, mlp_hidden, conv_hidden, t_hidden_dim, output_dim, use_conv=False):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, t_hidden_dim),
            nn.SiLU(),
            nn.Linear(t_hidden_dim, t_hidden_dim)
        )
        self.time_to_input = nn.Linear(t_hidden_dim, input_dim)

        self.use_conv = use_conv
        if use_conv:
            self.score_regression = ConvolutionalMLP(
                input_channels=1, 
                hidden_channels=conv_hidden, 
                t_hidden_dim=t_hidden_dim
            )
        else:
            self.score_regression = MLP(input_dim, mlp_hidden, output_dim)
    
    def forward(self, x, t: torch.Tensor):
        t = t.view(-1, 1)
        t_embed = self.time_embed(t)
            
        if not self.use_conv:
            x = x.reshape(x.size(0), -1)
            t_embed = self.time_to_input(t_embed)
            score = self.score_regression(x + t_embed)

        if self.use_conv:
            spatial = int(x.size(1)**0.5)
            assert spatial * spatial == x.size(1)
            x = x.view(x.size(0), 1, spatial, spatial)
            score = self.score_regression(x, t_embed)

        return score