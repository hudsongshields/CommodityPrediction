import torch

def reverse_sde(model, x, t_start, T):
    """
    Implements a stochastic reverse-time SDE sampler for Variance-Exploding (VE) SDEs.
    
    This function progressively denoises a sample x_t starting from t_start down 
    to t=0 using the learned score s_theta(x, sigma).
    
    Args:
        model: The ScoreNetwork providing structural density gradients.
        x: Noisy input observations (batch_size, input_dim).
        t_start: Initial noise level (integer index).
        T: Total number of discretization steps.
        
    Returns:
        The progressively denoised sample at t=0.
    """
    model.eval()
    with torch.no_grad():
        # Variance Exploding (VE-SDE) parameters.
        sigma_min, sigma_max = 1e-4, 1.0
        
        x_t = x.float().reshape(x.size(0), -1).to(x.device)
        
        # Backward chronological iteration across the noise schedule.
        for t_idx in range(T, 0, -1):
            t_curr = float(t_idx) / T
            t_prev = float(t_idx - 1) / T
            
            # Map discrete indices to continuous sigma scales.
            sigma_curr = sigma_min * (sigma_max / sigma_min) ** t_curr
            sigma_prev = sigma_min * (sigma_max / sigma_min) ** t_prev
            
            t_batch = torch.full((x_t.size(0),), t_curr, dtype=x_t.dtype, device=x_t.device)
            # The model predicts the pure score s_theta(x, sigma).
            score = model(x_t, sigma_curr * torch.ones_like(t_batch)).reshape(x_t.size(0), -1)

            # SDE Discretization step: x_{t-1} = x_t + (sigma_curr^2 - sigma_prev^2) * score + noise.
            sigma_diff = sigma_curr**2 - sigma_prev**2
            
            # Add Langevin-style noise at each step except the final reconstruction.
            z = torch.randn_like(x_t) if t_idx > 1 else 0.0
            x_t = x_t + sigma_diff * score + torch.sqrt(sigma_diff) * z

        return x_t