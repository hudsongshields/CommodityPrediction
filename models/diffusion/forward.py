import torch
import matplotlib.pyplot as plt
from typing import Callable
import torchvision
import torchvision.transforms as transforms
import typing
from pathlib import Path

## continuous sigma(t) schedule, exponential growth
def noise_schedule(t, T: int, sigma_min=0.1, sigma_max=20.0 , exponential=False):
    if not exponential:
        return sigma_min + (sigma_max - sigma_min) * (t / T)

    if exponential:
        return sigma_min * (sigma_max / sigma_min) ** (t / T)


def forward_process(x_0, t, T, noise_schedule, batch_betas, device, oneshot=True):
    # t_max = int(pct_T * T) -> stop prematurely

    # noise based on the timestep
    x_0 = x_0.view(x_0.size(0), -1)  # flatten the image
    var_t = noise_schedule(t, T)
    var_t = torch.as_tensor(var_t, dtype=x_0.dtype)
    delta_t = torch.as_tensor(1.0 / T, dtype=x_0.dtype)

    if oneshot:
        decay = torch.exp(-0.5 * batch_betas).view(-1, 1).to(device)
        noise_scale = torch.sqrt(torch.clamp(1 - torch.exp(-batch_betas), min=0.0)).view(-1, 1).to(device)
        x_t = decay * x_0 + noise_scale * torch.randn_like(x_0).to(device)
        return x_t

    else:
        # random noise, scale, then add to image
        noise = torch.randn_like(x_0).to(device)
        x_t = x_0 + torch.sqrt(var_t * delta_t) * noise

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
        t_tensor = torch.tensor([t_max], dtype=torch.long)
        beta_cumsum = torch.cumsum(noise_schedule(torch.arange(t_max, dtype=torch.float32), T), dim=0) * (1.0 / T)
        batch_betas = beta_cumsum[t_tensor].view(-1, 1)

        x_noisy = forward_process(x, t, T, noise_schedule, batch_betas)

        # scaling
        img = x_noisy.squeeze().detach().numpy()
        img = ((img.clip(-1.0, 1.0)) + 1.0) / 2.0

        plt.imshow(img, cmap='gray')
        plt.title(f"t = {t}")
        plt.axis("off")
        plt.show()