import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Sigma conditioning ---

class SinusoidalEmbedding(nn.Module):
    """Encodes a scalar sigma into a sinusoidal vector via log-scale frequencies."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
        # sigma: (batch,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=sigma.device) / (half - 1)
        )
        log_sigma = torch.log(sigma).unsqueeze(1)  # (batch, 1)
        args = log_sigma * freqs.unsqueeze(0)       # (batch, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (batch, dim)


# --- Building blocks ---

class ResBlock1D(nn.Module):
    """
    1D residual block with FiLM sigma conditioning.
    GroupNorm → SiLU → Conv1d → FiLM(scale, shift) → GroupNorm → SiLU → Conv1d → + residual
    """
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film  = nn.Linear(emb_dim, out_ch * 2)  # projects emb → (scale, shift)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act   = nn.SiLU()
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # FiLM conditioning: scale and shift from sigma embedding
        film = self.film(emb).unsqueeze(-1)          # (batch, out_ch*2, 1)
        scale, shift = film.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_res=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock1D(in_ch if i == 0 else out_ch, out_ch, emb_dim)
            for i in range(num_res)
        ])
        self.downsample = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, emb):
        for block in self.res_blocks:
            x = block(x, emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim, num_res=2):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.res_blocks = nn.ModuleList([
            ResBlock1D(in_ch + skip_ch if i == 0 else out_ch, out_ch, emb_dim)
            for i in range(num_res)
        ])

    def forward(self, x, skip, emb):
        x = self.upsample(x)

        # pad if length mismatch due to odd input lengths
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))

        x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, emb)
        return x


# --- Main model ---

class UNet1D(nn.Module):
    """
    1D-UNet score network for score-based diffusion on time series.

    Args:
        in_channels:   number of input channels (1 for univariate, N for multivariate)
        base_channels: channel count at the first encoder level
        channel_mults: multiplier per encoder level, e.g. (1, 2, 4)
        emb_dim:       dimension of the sigma embedding MLP
        num_res_blocks: residual blocks per encoder/decoder level

    Input:
        x:     (batch, length) or (batch, channels, length)
        sigma: (batch,) noise level tensor

    Output: same shape as x
    """
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        emb_dim=128,
        num_res_blocks=2,
    ):
        super().__init__()
        self.in_channels = in_channels

        # sigma embedding: sinusoidal → 2-layer MLP
        self.sigma_emb = SinusoidalEmbedding(emb_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        channels = [base_channels * m for m in channel_mults]

        # stem
        self.stem = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)

        # encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i + 1], emb_dim, num_res_blocks)
            )

        # bottleneck
        bot_ch = channels[-1]
        self.bottleneck = nn.ModuleList([
            ResBlock1D(bot_ch, bot_ch, emb_dim),
            ResBlock1D(bot_ch, bot_ch, emb_dim),
        ])

        # decoder (mirrors encoder in reverse)
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.up_blocks.append(
                UpBlock(channels[i + 1], channels[i + 1], channels[i], emb_dim, num_res_blocks)
            )

        # head
        self.head = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, sigma):
        # handle (batch, length) input
        squeezed = x.ndim == 2
        if squeezed:
            x = x.unsqueeze(1)  # → (batch, 1, length)

        # sigma embedding
        emb = self.emb_mlp(self.sigma_emb(sigma))  # (batch, emb_dim)

        # stem
        h = self.stem(x)

        # encoder — collect skips
        skips = []
        for block in self.down_blocks:
            h, skip = block(h, emb)
            skips.append(skip)

        # bottleneck
        for block in self.bottleneck:
            h = block(h, emb)

        # decoder — consume skips in reverse
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = block(h, skip, emb)

        # head
        out = self.head(h)

        if squeezed:
            out = out.squeeze(1)  # → (batch, length)
        return out


class Diffusion(nn.Module):
    """
    Wrapper around UNet1D for backwards compatibility with existing notebooks/scripts.

    The old interface accepted MLP/conv topology args; those are ignored here since
    the architecture is now fully UNet-based. `in_channels` can be set explicitly if
    working with multivariate data.
    """
    def __init__(self, t_hidden_dim=128,
                 in_channels=1, base_channels=64, channel_mults=(1, 2, 4), num_res_blocks=2,
                 **kwargs):  # absorb legacy MLP/conv topology args
        super().__init__()
        self.unet = UNet1D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            emb_dim=t_hidden_dim,
            num_res_blocks=num_res_blocks,
        )

    def forward(self, x, sigma):
        return self.unet(x, sigma)
