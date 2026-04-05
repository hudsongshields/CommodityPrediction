import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.return_prediction.ds_tgnn import DiffusionReturnPrediction
from models.diffusion.diffusion_architecture import Diffusion
from models.diffusion.loss_func import ScoreDiffusionLoss


class _ScoreModelAdapter(nn.Module):
    """Adapter to keep legacy score_net(x, sigma) calls compatible with UNet diffusion."""

    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x, sigma):
        # New UNet diffusion expects sigma shape [B], but legacy code sometimes passes [B, 1].
        if sigma.ndim > 1:
            sigma = sigma.view(-1)
        return self.diffusion_model(x, sigma)


def _build_fully_connected_edge_index(num_hubs, device):
    """Create an undirected fully connected graph over weather hubs."""
    edge_pairs = []
    for src in range(num_hubs):
        for dst in range(src + 1, num_hubs):
            edge_pairs.append([src, dst])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous().to(device)
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
    return edge_index


def train_dstgnn(config, folds, device):
    """
    Train and evaluate DS-TGNN across one or more folds.

    Returns a dictionary with concatenated predictions/targets, uncertainty,
    valuations, timestamps, fold markers, loss history, and the final model.
    """
    epochs = config.get("epochs", 5)
    learning_rate = config.get("lr", 0.001)
    monte_carlo_samples = config.get("mc_samples", 1)
    magnitude_weighted_loss = config.get("mag_weighted", True)
    include_denoised = config.get("include_denoised", False)
    score_loss_weight = 0.1

    score_loss_fn = ScoreDiffusionLoss().to(device)
    history = {"train_loss": [], "val_loss": []}

    all_predictions = []
    all_targets = []
    all_uncertainty = []
    all_valuations = []
    all_dates = []
    fold_markers = []
    model = None

    for fold_data in folds:
        fold_number = fold_data["fold"]
        print(f"  Processing Fold {fold_number}/{len(folds)}...")
        train_loader = fold_data["train"]
        test_loader = fold_data["test"]

        num_hubs = train_loader.dataset.dataset.n_hubs
        time_steps = 180
        feature_dim = 4
        edge_index = _build_fully_connected_edge_index(num_hubs, device)

        score_network = _ScoreModelAdapter(
            Diffusion(
                t_hidden_dim=config.get("t_hidden_dim", 16),
                in_channels=1,
                base_channels=config.get("base_channels", 64),
                channel_mults=config.get("channel_mults", (1, 2, 4)),
                num_res_blocks=config.get("num_res_blocks", 2),
            )
        )
        model = DiffusionReturnPrediction(
            score_network,
            input_dim=feature_dim,
            lstm_hidden=32,
            gnn_hidden=32,
            n_hubs=num_hubs,
            include_denoised=include_denoised,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch_index in range(epochs):
            model.train()
            epoch_losses = []
            for weather_batch, return_batch, _, _ in train_loader:
                weather_batch = weather_batch.to(device)
                return_batch = return_batch.to(device)
                optimizer.zero_grad()

                score_loss = score_loss_fn(model.score_net, weather_batch.transpose(1, 2))
                predicted_returns = model(weather_batch, edge_index)
                regression_loss = (predicted_returns - return_batch) ** 2

                if magnitude_weighted_loss:
                    regression_loss = regression_loss * (1.0 + torch.abs(return_batch) * 10.0)

                total_loss = regression_loss.mean() + score_loss_weight * score_loss
                total_loss.backward()
                optimizer.step()
                epoch_losses.append(total_loss.item())

            history["train_loss"].append(np.mean(epoch_losses))
            if (epoch_index + 1) % 5 == 0 or epochs <= 5:
                print(
                    f"  Fold {fold_number} | Epoch {epoch_index + 1}/{epochs} | "
                    f"Total Loss: {history['train_loss'][-1]:.6f}"
                )

        model.eval()
        if monte_carlo_samples > 1:
            model.enable_dropout()

        fold_predictions = []
        fold_targets = []
        fold_uncertainty = []
        fold_valuations = []
        fold_dates = []

        for weather_batch, target_batch, valuation_batch, date_batch in test_loader:
            weather_batch = weather_batch.to(device)
            with torch.no_grad():
                if monte_carlo_samples > 1:
                    sampled_predictions = torch.stack(
                        [model(weather_batch, edge_index).cpu() for _ in range(monte_carlo_samples)]
                    )
                    pred_mean = sampled_predictions.mean(0)
                    pred_std = sampled_predictions.std(0)
                else:
                    pred_mean = model(weather_batch, edge_index).cpu()
                    pred_std = torch.zeros_like(target_batch)

            fold_predictions.append(pred_mean)
            fold_targets.append(target_batch)
            fold_uncertainty.append(pred_std)
            fold_valuations.append(valuation_batch)
            fold_dates.extend(date_batch)

        all_predictions.append(torch.cat(fold_predictions).numpy())
        all_targets.append(torch.cat(fold_targets).numpy())
        all_uncertainty.append(torch.cat(fold_uncertainty).numpy())
        all_valuations.append(torch.cat(fold_valuations).numpy())
        all_dates.extend(fold_dates)
        fold_markers.append(pd.to_datetime(fold_dates[0]))

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    uncertainty = np.concatenate(all_uncertainty)
    valuations = np.concatenate(all_valuations)

    dates = pd.to_datetime(all_dates)
    sort_index = np.argsort(dates)

    return {
        "predictions": predictions[sort_index],
        "targets": targets[sort_index],
        "uncertainty": uncertainty[sort_index],
        "valuations": valuations[sort_index],
        "dates": dates[sort_index],
        "fold_markers": fold_markers,
        "history": history,
        "model": model,
    }
