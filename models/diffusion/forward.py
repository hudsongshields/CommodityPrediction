import torch
import matplotlib.pyplot as plt
from typing import Callable
import torchvision
import torchvision.transforms as transforms
import typing
from pathlib import Path

## continuous sigma(t) schedule, exponential growth
def noise_schedule(t, T: int, sigma_min=0.01, sigma_max=1.0):
    return sigma_min * (sigma_max / sigma_min) ** (t / T)


def forward(x, t, T, noise_schedule, oneshot=True):
    # t_max = int(pct_T * T) -> stop prematurely

    # noise based on the timestep
    var_t = noise_schedule(t, T)
    var_t = torch.as_tensor(var_t, dtype=x.dtype)
    delta_t = torch.as_tensor(1.0 / T, dtype=x.dtype)

    if oneshot:
        t_grid = torch.arange(0, t, dtype=x.dtype)
        beta_vals = noise_schedule(t_grid, T)
        integral_beta_t = torch.sum(beta_vals) * delta_t

        decay = torch.exp(-0.5 * integral_beta_t)
        noise_scale = torch.sqrt(torch.clamp(1 - torch.exp(-integral_beta_t), min=0.0))
        x_t = decay * x + noise_scale * torch.randn_like(x)

    else:
        # random noise, scale, then add to image
        noise = torch.randn_like(x)
        x_t = x + torch.sqrt(var_t * delta_t) * noise

    return x_t


# test data
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_root = Path("./data")
    raw_dir = data_root / "FashionMNIST" / "raw"
    required_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    should_download = not all((raw_dir / file_name).exists() for file_name in required_files)

    dataset = torchvision.datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=should_download,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    x, label = next(iter(loader))
    print(x.shape)

    T = 1000
    t_max = 0.8 * T

    for t in torch.linspace(0, t_max, steps=5):
        x_noisy = forward(x, t, T, noise_schedule)

        # scaling
        img = x_noisy.squeeze().detach().numpy()
        img = ((img.clip(-1.0, 1.0)) + 1.0) / 2.0

        plt.imshow(img, cmap='gray')
        plt.title(f"t = {t}")
        plt.axis("off")
        plt.show()