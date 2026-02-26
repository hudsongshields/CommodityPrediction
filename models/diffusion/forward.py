import torch

def noise_schedule(t: int, T: int, beta_start=1e-4, beta_end=0.02) -> float:
    return beta_start + (beta_end - beta_start) * (t / (T - 1))

def forward(x, T, t_max, noise_schedule: callable[[int, int], float]):
    # T_max = int(pct_T * T) -> stop prematurely
    device = x.device

    # Build beta schedule
    betas = torch.tensor(
        [noise_schedule(t, T) for t in range(T)],
        device=device
    )

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # direct sampling
    noise = torch.randn_like(x)

    sqrt_alpha_bar = torch.sqrt(alpha_bar[t_max])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t_max])

    x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    return x_t


if __name__ == '__main__':
    pass