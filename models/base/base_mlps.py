import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP) for high-dimensional feature projection.
    
    This module provides the core non-linear transformation layers for both 
    the ScoreNetwork and the ReturnPrediction head.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU()) # Using Sigmoid Linear Unit for smoother gradients.
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Projects the input x into the target dimension through the MLP stack."""
        return self.net(x)
    
class ConvolutionalMLP(nn.Module):
    """
    Spatiotemporal Convolutional MLP with time-conditional feature modulation.
    
    This architecture integrates noise-scale embeddings (t_embed) into a 
    convolutional stack, allowing the model to adapt its score prediction 
    based on the current diffusion step.
    """
    def __init__(self, input_channels, hidden_channels, t_hidden_dim):
        super().__init__()
        conv_layers = []
        time_layers = []
        prev_channels = input_channels
        for h in hidden_channels:
            conv_layers.append(nn.Conv2d(prev_channels, h, kernel_size=3, padding=1))
            # Linear modulator for integrating time-scale embeddings into the channel dimension.
            time_layers.append(nn.Linear(t_hidden_dim, h))
            prev_channels = h

        self.conv_layers = nn.ModuleList(conv_layers)
        self.time_layers = nn.ModuleList(time_layers)
        self.final_conv_layer = nn.Conv2d(prev_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x, t_embed):
        """
        Performs modulated convolutions across the spatiotemporal grid.
        
        Args:
            x: Input feature grid (batch_size, channels, height, width).
            t_embed: Noise scale embedding (batch_size, t_hidden_dim).
        """
        for layer in range(len(self.conv_layers)):
            # Modulate feature maps using the time-conditional embedding.
            x = self.conv_layers[layer](x) + self.time_layers[layer](t_embed).reshape(t_embed.size(0), -1, 1, 1)
            x = F.silu(x)
        x = self.final_conv_layer(x)
        return x