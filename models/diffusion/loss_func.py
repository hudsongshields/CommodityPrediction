import torch
import torch.nn as nn

__all__ = ['ScoreDiffusionLoss']

class ScoreDiffusionLoss(nn.Module):
    def __init__(self, sigma_min=1e-4, sigma_max=1.0):
        """
        Calculates the Denoising Score Matching (DSM) loss for Variance-Exploding SDEs.
        
        The objective minimizes the Fisher Divergence between the model score 
        s_theta(x_t, sigma_t) and the true score of the perturbed data distribution 
        p(x_t|x_0).
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, score_net, x_0):
        """
        Computes the weighted Fisher Divergence loss using the DSM objective.
        
        The loss is defined as:
        Loss = E[ 0.5 * sigma_t^2 * || s_theta(x_t, sigma_t) - \nabla_{x_t} log p(x_t|x_0) ||^2 ]
        where \nabla_{x_t} log p(x_t|x_0) = -z / sigma_t.
        """
        B = x_0.shape[0]
        
        # 1. Continuous timestep sampling across the noise schedule.
        t = torch.rand(B, device=x_0.device)
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        sigma_t_view = sigma_t.view(B, 1, 1, 1)
        
        # 2. Score perturbation: x_t = x_0 + sigma_t * z
        z = torch.randn_like(x_0)
        x_t = x_0 + z * sigma_t_view
        
        # 3. Model score prediction s_theta(x_t, sigma_t)
        # The ScoreNetwork identifies the structural density gradient of the data.
        x_t_flat = x_t.reshape(B, -1)
        score_pred_flat = score_net(x_t_flat, sigma_t.view(B, 1))
        
        # 4. Objective Minimization
        # For VE-SDE, we scale the score by sigma_t to match the unit-variance noise z.
        z_flat = z.reshape(B, -1)
        sigma_t_flat = sigma_t.view(B, 1)
        
        # Unified Fisher loss: sigma_t^2 * || score_pred + z/sigma_t ||^2
        loss = torch.mean((sigma_t_flat * score_pred_flat + z_flat) ** 2)
        
        return loss
