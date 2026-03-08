from pyexpat import model

import torch.nn as nn
import torch
from reversal import reverse_sde
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class ConvolutionalMLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, t_hidden_dim):
        super().__init__()
        conv_layers = []
        time_layers = []
        prev_channels = input_channels
        for h in hidden_channels:
            conv_layers.append(nn.Conv2d(prev_channels, h, kernel_size=3, padding=1))
            time_layers.append(nn.Linear(t_hidden_dim, h))
            prev_channels = h

        self.conv_layers = nn.ModuleList(conv_layers)
        self.time_layers = nn.ModuleList(time_layers)

        self.final_conv_layer = nn.Conv2d(prev_channels, input_channels, kernel_size=3, padding=1)


    def forward(self, x, t_embed):
        # x: (B, C, H, W)
        # t_embed: (B, t_hidden_dim)
        
        for layer in range(len(self.conv_layers)):
            x = self.conv_layers[layer](x) + self.time_layers[layer](t_embed).view(t_embed.size(0), -1, 1, 1)
            x = F.silu(x)
        x = self.final_conv_layer(x)
        return x

        
    
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
            x = x.view(x.size(0), -1)
            t_embed = self.time_to_input(t_embed)
            score = self.score_regression(x + t_embed)

        if self.use_conv:
            spatial = int(x.size(1)**0.5)
            assert spatial * spatial == x.size(1)
            x = x.view(x.size(0), 1, spatial, spatial)
            score = self.score_regression(x, t_embed)

        return score

    
    

class ReturnPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.return_regression = MLP(input_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        #lstm_out, _ = self.lstm(x)
        # return_value = self.return_regression(lstm_out[:, -1, :])
        return_value = self.return_regression(x)
        return return_value
    

class DiffusionReturnPrediction(nn.Module):
    def __init__(self, input_dim, mlp_hidden, conv_hidden, t_hidden_dim, output_dim, use_conv=False):
        super().__init__()
        self.diffusion = Diffusion(input_dim, mlp_hidden, conv_hidden, t_hidden_dim, output_dim, use_conv)
        self.return_prediction = ReturnPrediction(input_dim, hidden_dim, output_dim)

    def forward(self, x, t):
        cleaned_x = reversal(self.diffusion, x, t)
        return self.return_prediction(cleaned_x)
