import torch
import matplotlib.pyplot as plt
from typing import Callable
import torchvision
import torchvision.transforms as transforms

def noise_schedule(t: int, T: int, beta_start=1e-4, beta_end=0.02) -> float:
    return beta_start + (beta_end - beta_start) * (t / (T - 1))

def forward(x, T, t_max, noise_schedule: Callable[[int, int], float]):
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

# test data
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    x, label = next(iter(loader))
    print(x.shape)


    T = 1000

    for t in [10, 100, 300, 700, 999]:
        x_noisy = forward(x, T, t, noise_schedule)

        img = x_noisy.squeeze().detach().numpy()
        img = (img + 1) / 2  # back to [0,1]

        plt.imshow(img, cmap='gray')
        plt.title(f"t = {t}")
        plt.axis("off")
        plt.show()