import torch
import torch.nn as nn

__all__ = ['ScoreDiffusionLoss']

class ScoreDiffusionLoss(nn.Module):
    def __init__(self, sigma_min=1e-4, sigma_max=1.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, score_net, x_0):
        """
        V2.3 Score-Hardened Loss: Denoising Score Matching (DSM).
        This objective minimizes the Fisher Divergence between the model s_theta
        and the true score \nabla log p(x) of the perturbed data distribution.
        
        Math: Loss = E[ \sigma_t^2 || s_theta(x_t, \sigma_t) + z/\sigma_t ||^2 ]
                   = E[ || \sigma_t * s_theta(x_t, \sigma_t) + z ||^2 ]
        """
        B, N, T, F = x_0.shape
        
        # 1. Continuous timestep sampling (NCSN/VE-SDE standard)
        t = torch.rand(B, device=x_0.device)
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        sigma_t_view = sigma_t.reshape(B, 1, 1, 1)
        
        # 2. Score perturbation: x_t = x_0 + \sigma_t * z
        z = torch.randn_like(x_0)
        x_t = x_0 + z * sigma_t_view
        
        # 3. Model score prediction
        # The score_net is trained to output the score s_theta(x_t, sigma_t)
        score_pred_flat = score_net(x_t, sigma_t)
        
        # 4. Theory-Aligned Objective
        # For VE-SDE, we scale the score by \sigma_t to match the unit-variance noise z.
        z_flat = z.reshape(B, -1)
        sigma_t_flat = sigma_t.view(B, 1)
        loss = torch.mean((sigma_t_flat * score_pred_flat + z_flat) ** 2)
        return loss
