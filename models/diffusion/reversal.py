import torch
from forward import noise_schedule

def reverse_sde(model, x, t_start, T, noise_schedule, t_all=None, beta_all=None, beta_cumsum=None):
    model.eval()
    with torch.no_grad():
        # ONLY keep this for testing
        if t_all is None and beta_all is  None and beta_cumsum is None:
            t_all = torch.arange(T, dtype=torch.float32)
            beta_all = noise_schedule(t_all, T)
            beta_cumsum = torch.cumsum(beta_all, dim=0) * (1.0 / T)

        timesteps = torch.arange(t_start, 0, -1, device=x.device, dtype=torch.long)
        delta_t = 1.0 / T
        x_t = x.float().reshape(x.size(0), -1).to(x.device)

        # V2.1.2 VE-SDE Reverse Diffusion Step
        # sigma(t) = sigma_min * (sigma_max/sigma_min)**t
        sigma_min, sigma_max = 1e-4, 1.0
        
        for t_idx in range(T, 0, -1):
            t_curr = float(t_idx) / T
            t_prev = float(t_idx - 1) / T
            
            sigma_curr = sigma_min * (sigma_max / sigma_min) ** t_curr
            sigma_prev = sigma_min * (sigma_max / sigma_min) ** t_prev
            
            t_batch = torch.full((x_t.size(0),), t_curr, dtype=x_t.dtype, device=x_t.device)
            # Model prediction is the pure score s_theta(x, sigma)
            score = model(x_t, sigma_curr * torch.ones_like(t_batch)).reshape(x_t.size(0), -1)

            # Variational step: x_{t-1} = x_t + (sigma_curr^2 - sigma_prev^2) * score + noise
            sigma_diff = sigma_curr**2 - sigma_prev**2
            
            noise = torch.randn_like(x_t) if t_idx > 1 else 0.0
            x_t = x_t + sigma_diff * score + torch.sqrt(sigma_diff) * noise

        return x_t