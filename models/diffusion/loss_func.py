import torch
import torch.nn as nn
from forward import noise_schedule, forward


class ScoreDiffusionLoss(nn.Module):

    def __init__(
        self,
        T:           int   = 1000,
        sigma_min:   float = 0.01,
        sigma_max:   float = 1.0,
        eps:         float = 1e-5,
    ):
        super().__init__()
        self.T         = T
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps       = eps

    def _marginal_params(self, x0: torch.Tensor, t: torch.Tensor):
        delta_t = 1.0 / self.T
        device  = x0.device

        t_int  = t.long().clamp(1, self.T)
        t_grid = torch.arange(0, t_int.max().item(), dtype=x0.dtype, device=device)

        beta_vals = torch.tensor(
            noise_schedule(t_grid.cpu().numpy(), self.T, self.sigma_min, self.sigma_max),
            dtype=x0.dtype, device=device
        )

        integral = torch.zeros(x0.shape[0], dtype=x0.dtype, device=device)
        for i, ti in enumerate(t_int):
            integral[i] = beta_vals[:ti].sum() * delta_t

        while integral.dim() < x0.dim():
            integral = integral.unsqueeze(-1)

        decay       = torch.exp(-0.5 * integral)
        noise_scale = torch.sqrt((1 - torch.exp(-integral)).clamp(min=self.eps))

        return decay, noise_scale

    def forward(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        B      = x0.shape[0]
        device = x0.device

        # pick a random point in the diffusion process for each image in the batch
        # each image gets a different noise level so the model learns all of them
        t = torch.randint(1, self.T, (B,), device=device)

        # figure out how decayed and how noisy each image should be at its chosen t
        decay, noise_scale = self._marginal_params(x0, t)

        # actually corrupt the images 
        noise = torch.randn_like(x0)
        x_t   = decay * x0 + noise_scale * noise

        # convert timestep to noise magnitude — this is what the reversal
        # passes to the model so we keep it consistent here
        var_t = torch.tensor(
            noise_schedule(t.cpu().numpy(), self.T, self.sigma_min, self.sigma_max),
            dtype=x0.dtype, device=device
        )

        # given this noisy image and this noise level,
        # which direction points toward the clean version?
        score_pred = model(x_t, var_t.view(B))

        # we know the right answer because we added the noise ourselves
        # the correct direction is just the negative noise scaled down by how loud it was
        target_score = -noise / (noise_scale + self.eps)

        # noisier timesteps get more weight in the loss
        # this matches the var_t² coefficient in the reversal's drift term
        # and ensures the loss upper bounds the negative log likelihood
        lambda_t = var_t.view(B, *([1] * (x0.dim() - 1))) ** 2

        # how wrong was the model's direction vs the true direction, weighted by noise level
        loss = lambda_t * (score_pred - target_score).pow(2)

        return 0.5 * loss.view(B, -1).mean()


if __name__ == '__main__':
    import torchvision
    import torchvision.transforms as transforms
    from pathlib import Path

    class DummyScoreNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Conv2d(1, 1, 1)

        def forward(self, x, var_t):
            return self.net(x)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_root = Path("./data")
    dataset   = torchvision.datasets.FashionMNIST(
        root=str(data_root), train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    x0, _ = next(iter(loader))

    model   = DummyScoreNet()
    loss_fn = ScoreDiffusionLoss(T=1000)
    loss    = loss_fn(model, x0)
    print(f"loss = {loss.item():.4f}")