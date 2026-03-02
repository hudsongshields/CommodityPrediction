import torch
from forward import noise_schedule

def reversal(model, x, t_max, T=1000, steps=100, device='cpu'):
    model.eval()

    with torch.no_grad():
        x_t = x.clone().to(device)

        # equal size steps
        timesteps = torch.linspace(t_max, 0, steps, dtype=torch.float32)

        # loop through consecutive pairs of images
        for i in range(len(timesteps) - 1):
            t_now = timesteps[i].item()
            t_next = timesteps[i + 1].item()

            # convert timesteps to noise using scheduler
            sigma_now = noise_schedule(t_now, T)
            sigma_next = noise_schedule(t_next, T)
            sigma_tensor = torch.tensor([sigma_now], device=device, dtype=torch.float32)

            # model predicts noise and gives score (score = -noise / sigma)
            noise_pred = model(x_t, sigma_tensor.expand(x_t.shape[0]))
            score = -noise_pred / sigma_now

            # how much variance to remove in this step
            d_sigma_sq = sigma_now ** 2 - sigma_next ** 2

            # actual denoising
            x_t =  x_t + d_sigma_sq * score

            if sigma_next > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + (d_sigma_sq ** 0.5) * noise

    return x_t

if __name__ == '__main__':
    # TODO: test the reversal function with a noisy image (see how image is denoised with decreasing time steps)
    pass