import os
import numpy as np
import torch

from models.diffusion.loss_func import ScoreDiffusionLoss


def pretrain_diffusion_model(
    diffusion_model,
    train_loader,
    device,
    epochs=10,
    learning_rate=1e-3,
    save_path=None,
    print_every=1,
):
    """
    Pretrain diffusion score model on raw weather tensors before return prediction.

    Args:
        diffusion_model: model from `models.diffusion.diffusion_architecture.Diffusion`.
        train_loader: dataloader yielding `(weather, target, valuation, date)` tuples.
        device: torch device.
        epochs: pretraining epochs.
        learning_rate: optimizer learning rate.
        save_path: optional path to persist pretrained weights.
        print_every: print interval for loss logs.

    Returns:
        dict with history and optional save path.
    """
    diffusion_model = diffusion_model.to(device)
    score_loss_fn = ScoreDiffusionLoss().to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=learning_rate)

    history = {"diffusion_pretrain_loss": []}
    diffusion_model.train()

    for epoch_idx in range(epochs):
        epoch_losses = []
        for weather_batch, _, _, _ in train_loader:
            weather_batch = weather_batch.to(device)
            optimizer.zero_grad()

            # CommodityWeatherDataset yields [B, T, N, F]; score loss expects [B, N, T, F].
            score_loss = score_loss_fn(diffusion_model, weather_batch.transpose(1, 2))
            score_loss.backward()
            optimizer.step()

            epoch_losses.append(score_loss.item())

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["diffusion_pretrain_loss"].append(epoch_loss)
        if (epoch_idx + 1) % max(print_every, 1) == 0:
            print(f"  Diffusion Pretrain | Epoch {epoch_idx + 1}/{epochs} | Loss: {epoch_loss:.6f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(diffusion_model.state_dict(), save_path)
        print(f"Saved pretrained diffusion weights to {save_path}")

    return {"history": history, "save_path": save_path}


def train_diffusion(
    diffusion_model,
    train_loader,
    device,
    epochs=10,
    learning_rate=1e-3,
    save_path=None,
    print_every=1,
):
    """Public diffusion training entrypoint used by experiment scripts."""
    return pretrain_diffusion_model(
        diffusion_model=diffusion_model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=save_path,
        print_every=print_every,
    )

