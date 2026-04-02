import torch.nn as nn
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
            x = self.conv_layers[layer](x) + self.time_layers[layer](t_embed).reshape(t_embed.size(0), -1, 1, 1)
            x = F.silu(x)
        x = self.final_conv_layer(x)
        return x