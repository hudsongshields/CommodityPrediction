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
        x_t = x.float().view(x.size(0), -1).to(x.device)

        for t_val in timesteps:
            t_batch = torch.full((x_t.size(0),), float(t_val.item()) / T, dtype=x_t.dtype, device=x_t.device)
            
            beta_t = beta_cumsum[t_val].to(x.device)
            score = model(x_t, t_batch).view(x_t.size(0), -1)

            noise = 0.0
            if t_val > 0:
                noise = torch.sqrt(beta_t * delta_t) * torch.randn_like(x_t)
            drift = -0.5 * beta_t * x_t + beta_t * score
            x_t = x_t + drift * delta_t + noise

        return x_t