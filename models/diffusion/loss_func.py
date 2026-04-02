import torch


def loss_func_diffusion(score, x_t, x_0, beta_integral):
    mean_coeff = torch.exp(-0.5 * beta_integral)
    var_t = 1.0 - torch.exp(-beta_integral)
    var_t = torch.clamp(var_t, min=1e-5)

    target_score = (mean_coeff * x_0 - x_t) / var_t
    return torch.mean(var_t *(score - target_score) ** 2)