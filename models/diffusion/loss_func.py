import torch
import torch.nn as nn

__all__ = ['ScoreDiffusionLoss']

class ScoreDiffusionLoss(nn.Module):
    def __init__(self, sigma_min=1e-4, sigma_max=1.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, denoiser, x_0):
        """
        x_0 shape: [B, N, T, F] (Clean weather data)
        Returns: scalar loss
        """
        B, N, T, F = x_0.shape
        
        # 1. Sample random continuous timesteps t ~ U[0, 1]
        t = torch.rand(B, device=x_0.device)
        
        # 2. Derive sigma(t) based on exponential schedule
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        
        # Reshape for broadcasting [B, 1, 1, 1]
        sigma_t_view = sigma_t.view(B, 1, 1, 1)
        
        # 3. Add noise
        z = torch.randn_like(x_0)
        x_t = x_0 + z * sigma_t_view
        
        # 4. Predict score/noise using the denoiser
        # Denoiser should take x_t [B, N, T, F] and sigma_t (or t) [B]
        predicted = denoiser(x_t, sigma_t)
        
        # 5. Compute loss based on parameterization (e.g. predicting z)
        # Weighting by sigma_t^2 is common in score matching
        loss = torch.mean((predicted - z) ** 2)
        return loss