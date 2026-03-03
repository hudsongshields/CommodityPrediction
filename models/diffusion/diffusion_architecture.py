from pyexpat import model

import torch.nn as nn
import torch
from reversal import reversal


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
# Takes in a noisy input and predicts the score to remove the noise
class Diffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.score_regression = MLP(input_dim, hidden_dim, output_dim)
    
    def forward(self, x, t: torch.Tensor):

        t = t.view(-1, 1)
        t_embed = self.time_embedding(t)

        score = self.score_regression(x + t_embed)
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.diffusion = Diffusion(input_dim, hidden_dim, output_dim)
        self.return_prediction = ReturnPrediction(input_dim, hidden_dim, output_dim)

    def forward(self, x, t):
        cleaned_x = reversal(self.diffusion, x, t)
        return self.return_prediction(cleaned_x)
