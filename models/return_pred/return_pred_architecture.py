from .base_mlps import MLP, ConvolutionalMLP
import torch.nn as nn

class ReturnPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.return_regression = MLP(input_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        #lstm_out, _ = self.lstm(x)
        # return_value = self.return_regression(lstm_out[:, -1, :])
        return_value = self.return_regression(x)
        return return_value
    
"""
class DiffusionReturnPrediction(nn.Module):
    def __init__(self, input_dim, mlp_hidden, conv_hidden, t_hidden_dim, output_dim, use_conv=False):
        super().__init__()
        self.diffusion = Diffusion(input_dim, mlp_hidden, conv_hidden, t_hidden_dim, output_dim, use_conv)
        self.return_prediction = ReturnPrediction(input_dim, mlp_hidden, output_dim)

    def forward(self, x, t):
        cleaned_x = reversal(self.diffusion, x, t)
        return self.return_prediction(cleaned_x)"""