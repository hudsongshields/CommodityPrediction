import torch
from forward import noise_schedule

def reversal(model, x, t_max, T=1000):
    model.eval()

    with torch.no_grad():
        # loop through consecutive pairs of images
        timesteps = torch.arange(start=t_max, end=0, steps=-1) # reversed timesteps

        # convert timesteps to noise using scheduler
        var_tensor = torch.tensor(noise_schedule(timesteps, T))
        delta_t = 1.0 / T

        x_t = torch.float(x)
        for t in timesteps:
            var_t = var_tensor[t]
            score = model(x_t, var_t)
            
            drift_coeff = 0.5 * (var_t ** 2) * x_t
            drift_term = drift_coeff - 0.5 * (var_t ** 2) * score

            noise_term = torch.sqrt(var_t * (delta_t)) * torch.randn_like(x_t)

            x_t += drift_term * delta_t + noise_term

    return x_t

if __name__ == '__main__':
    # TODO: test the reversal function with a noisy image (see how image is denoised with decreasing time steps)
    pass