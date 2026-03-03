import torch
import matplotlib.pyplot as plt
from typing import Callable
import torchvision
import torchvision.transforms as transforms

## continuous sigma(t) schedule, exponential growth
def noise_schedule(t: int, T: int, sigma_min=0.01, sigma_max=250.0) -> float:
    return sigma_min * (sigma_max - sigma_min) ** (t / T)

def forward(x, T, t_max, noise_schedule: Callable[[int, int], float]):
    # t_max = int(pct_T * T) -> stop prematurely
    device = x.device

    # noise based on the timestep
    sigma_t = noise_schedule(t_max, T)
    sigma_t = torch.tensor(sigma_t, device=device, dtype=x.dtype)

    # random noise, scale, then add to image
    noise = torch.randn_like(x)
    x_t = x + sigma_t * noise
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

    for t in [10, 100, 300, 500, 700, 999]:
        x_noisy = forward(x, T, t, noise_schedule)

        # scaling
        img = x_noisy.squeeze().detach().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img, cmap='gray')
        plt.title(f"t = {t}")
        plt.axis("off")
        plt.show()