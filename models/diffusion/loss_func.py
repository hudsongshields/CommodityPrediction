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
        V2.1 Robust Multi-Task Loss: Synchronizing Score-Matching and Alpha-Targeting.
        x_0 shape: [B, N, T, F] 
        """
        B, N, T, F = x_0.shape
        
        # 1. Continuous timestep sampling
        t = torch.rand(B, device=x_0.device)
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        sigma_t_view = sigma_t.reshape(B, 1, 1, 1)
        
        # 2. Score perturbation
        z = torch.randn_like(x_0)
        x_t = x_0 + z * sigma_t_view
        
        # 3. Model score prediction (Denoising)
        # Denoiser returns flattened score [B, N*T*F]
        predicted_flat = denoiser(x_t, sigma_t)
        
        # 4. Symmetric comparison (Flatten both for loss computation)
        z_flat = z.reshape(B, -1)
        
        # Consistent shape matching to prevent broadcasting errors
        loss = torch.mean((predicted_flat - z_flat) ** 2)
        return loss
