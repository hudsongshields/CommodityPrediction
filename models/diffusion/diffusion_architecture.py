from pyexpat import model

import torch.nn as nn
import torch
from reversal import reversal


# Takes in a noisy input and predicts the score to remove the noise
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

        t = t.to(device=x.device, dtype=x.dtype).view(-1, 1)
        t_embed = self.time_embedding(t)

        score = self.score_regression(x + t_embed)
        return score
    
    

class ReturnPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.return_regression = MLP(hidden_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return_value = self.return_regression(lstm_out[:, -1, :])
        return return_value
    


class DiffusionReturnPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, return_regression):
        super().__init__()
        self.diffusion = Diffusion(input_dim, hidden_dim, output_dim)
        self.return_prediction = return_regression

    def forward(self, x_noisy, x_orig):
        batch_scores = self.diffusion(x_noisy)
        x_clean = reversal(x_orig)
        batch_returns = self.return_prediction()
        

        return batch_scores, batch_returns

diffusion = DiffusionReturnPrediction(return_regression)