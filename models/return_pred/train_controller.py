import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_controller_return(
    model,
    train_loader,
    epochs,
    optimizer,
    loss_func_MSE,
    every_n_epochs=10,
    device="cpu",
    t_score=None,
    val_loader=None,
    patience=None,
):
    """
    Train return predictor using diffusion score features.
    Batch formats:
      - (features, targets)
      - (features, targets, season_idx)
    """
    model.train()
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            features = batch[0].to(device)
            targets = batch[1].to(device)
            season_idx = batch[2].to(device) if len(batch) > 2 else None

            results = model(features, season_idx=season_idx, t_score=t_score)
            batch_loss = loss_func_MSE(results, targets)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

        if (epoch % every_n_epochs == 0) or (epoch == epochs - 1):
            msg = f"{epoch}: {epoch_loss}"
            if val_loader is not None:
                val_loss = eval_mse(model, val_loader, loss_func_MSE, device, t_score)
                msg += f" | val: {val_loss:.6f}"
            print(msg)

        if val_loader is not None and patience is not None:
            val_loss = eval_mse(model, val_loader, loss_func_MSE, device, t_score)
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


def eval_mse(model, val_loader, loss_func_MSE, device="cpu", t_score=None):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            features = batch[0].to(device)
            targets = batch[1].to(device)
            season_idx = batch[2].to(device) if len(batch) > 2 else None
            preds = model(features, season_idx=season_idx, t_score=t_score)
            loss = loss_func_MSE(preds, targets)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


# Recommended arguments:
# python3.12 models/return_pred/train_controller.py --target-col Return_Corn --use-season --seq-len 60 --horizons 5 --t-score 0.2,0.5,0.8 --seeds 1,2,3,4,5

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from models.diffusion.diffusion_architecture import Diffusion
    from models.diffusion.forward import noise_schedule, forward_process
    from models.diffusion.loss_func import loss_func_diffusion
    from models.return_pred.return_pred_architecture import ReturnPrediction

    def train_diffusion(diffusion_model, data_loader, epochs, t_max, T, lr, device):
        diffusion_model.to(device)
        diffusion_model.train()
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

        delta_t = 1.0 / T
        t_all = torch.arange(t_max, dtype=torch.float32, device=device)
        beta_all = noise_schedule(t_all, T)
        beta_cumsum = torch.cumsum(beta_all, dim=0) * delta_t

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (x0,) in data_loader:
                x0 = x0.to(device)
                optimizer.zero_grad()

                t = torch.randint(1, t_max, (x0.size(0),), device=device, dtype=torch.long)
                batch_betas = beta_cumsum[t].view(-1, 1)
                x_t = forward_process(x0, t, T, noise_schedule, batch_betas, device=device).view(x0.size(0), -1)

                scores = diffusion_model(x_t, t.to(x_t.dtype) / T).view(x0.size(0), -1)
                loss = loss_func_diffusion(scores, x_t, x0.view(x0.size(0), -1), batch_betas)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"[Diffusion] {epoch}: {epoch_loss:.6f}")

        diffusion_model.eval()

    parser = argparse.ArgumentParser(description="Train return predictor on CSV data.")
    parser.add_argument("--data-csv", type=str, default="models/diffusion/testing/data/total_merged_data.csv")
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--horizons", type=str, default="1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--t-score", type=str, default="0.5")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--use-season", action="store_true")

    parser.add_argument("--diffusion-checkpoint", type=str, default="")
    parser.add_argument("--diff-epochs", type=int, default=50)
    parser.add_argument("--diff-lr", type=float, default=1e-3)
    parser.add_argument("--diff-t-max", type=int, default=200)
    parser.add_argument("--diff-T", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,1337,2024")
    args = parser.parse_args()

    if "," in args.t_score:
        t_score = [float(v.strip()) for v in args.t_score.split(",") if v.strip()]
    else:
        t_score = float(args.t_score)

    horizons = [int(v.strip()) for v in args.horizons.split(",") if v.strip()]
    if any(h < 0 for h in horizons):
        raise ValueError("horizons must be >= 0")
    max_h = max(horizons)

    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if not seeds:
        seeds = [42]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.data_csv)
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"])

    return_cols = [c for c in df.columns if c.startswith("Return_")]
    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not found in CSV.")

    weather_cols = [c for c in df.columns if c not in return_cols and c != "time"]
    df = df.copy()
    doy = df["time"].dt.dayofyear.astype(float)
    df = df.assign(
        doy_sin=np.sin(2 * np.pi * doy / 365.25),
        doy_cos=np.cos(2 * np.pi * doy / 365.25),
    )

    feature_cols = weather_cols + ["doy_sin", "doy_cos"]
    df = df.dropna(subset=feature_cols + [args.target_col]).reset_index(drop=True)

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[args.target_col].to_numpy(dtype=np.float32)

    X_seq, y_seq, season_seq, start_idx = [], [], [], []
    for i in range(len(df) - args.seq_len - max_h + 1):
        target_idx = i + args.seq_len - 1 + max_h
        X_seq.append(X_all[i:i + args.seq_len])
        y_seq.append([y_all[i + args.seq_len - 1 + h] for h in horizons])
        season_seq.append(df.loc[target_idx, "time"].month)
        start_idx.append(i)

    X_arr = np.stack(X_seq)
    y_arr = np.array(y_seq, dtype=np.float32)
    start_idx = np.array(start_idx)

    split_row = int(len(df) * (1.0 - args.val_split))
    train_mask = (start_idx + args.seq_len - 1 + max_h) < split_row
    val_mask = start_idx >= split_row

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Not enough data for non-overlapping time split.")

    X_train = torch.tensor(X_arr[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y_arr[train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_arr[val_mask], dtype=torch.float32)
    y_val = torch.tensor(y_arr[val_mask], dtype=torch.float32)

    if args.use_season:
        def month_to_season(m):
            if m in (12, 1, 2):
                return 0
            if m in (3, 4, 5):
                return 1
            if m in (6, 7, 8):
                return 2
            return 3

        season_arr = np.array([month_to_season(m) for m in season_seq], dtype=np.int64)
        season_train = torch.tensor(season_arr[train_mask], dtype=torch.long)
        season_val = torch.tensor(season_arr[val_mask], dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train, season_train)
        val_dataset = TensorDataset(X_val, y_val, season_val)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

    feature_dim = X_train.shape[2]
    loss_fn = torch.nn.MSELoss()
    accuracies = []

    for seed in seeds:
        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        diffusion = Diffusion(
            input_dim=feature_dim,
            mlp_hidden=(64, 32),
            conv_hidden=(32, 64, 32),
            t_hidden_dim=64,
            output_dim=feature_dim,
            use_conv=False,
        ).to(device)

        if args.diffusion_checkpoint:
            diffusion.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))

        diff_t_max = min(args.diff_t_max, args.diff_T - 1)
        if diff_t_max < 2:
            raise ValueError("diff_t_max must be >= 2 and less than diff_T.")

        diff_rows = df.iloc[:split_row]
        diff_data = diff_rows[feature_cols].to_numpy(dtype=np.float32)
        diff_loader = DataLoader(TensorDataset(torch.tensor(diff_data)), batch_size=args.batch_size, shuffle=True)

        train_diffusion(
            diffusion_model=diffusion,
            data_loader=diff_loader,
            epochs=args.diff_epochs,
            t_max=diff_t_max,
            T=args.diff_T,
            lr=args.diff_lr,
            device=device,
        )

        model = ReturnPrediction(
            diffusion=diffusion,
            feature_dim=feature_dim,
            lstm_hidden=64,
            mlp_hidden=[64, 32],
            output_dim=len(horizons),
            use_season=args.use_season,
            t_score=t_score,
            lstm_layers=1,
            dropout=0.0,
            freeze_diffusion=True,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_controller_return(
            model=model,
            train_loader=train_loader,
            epochs=args.epochs,
            optimizer=optimizer,
            loss_func_MSE=loss_fn,
            every_n_epochs=1,
            device=device,
            t_score=t_score,
            val_loader=val_loader,
            patience=args.patience if args.patience > 0 else None,
        )

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                features = batch[0].to(device)
                targets = batch[1].to(device)
                season_idx = batch[2].to(device) if len(batch) > 2 else None
                preds = model(features, season_idx=season_idx, t_score=t_score)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        preds = torch.cat(all_preds).view(-1)
        targets = torch.cat(all_targets).view(-1)
        errors = preds - targets

        mse = torch.mean(errors ** 2).item()
        accuracy = (preds > 0).eq(targets > 0).float().mean().item()
        bias = errors.mean().item()
        variance = errors.var(unbiased=False).item()
        accuracies.append(accuracy)

        print(f"Seed {seed} | Eval MSE: {mse:.6f}")
        print(f"Seed {seed} | Eval Accuracy (sign): {accuracy:.4f}")
        print(f"Seed {seed} | Eval Bias (mean error): {bias:.6f}")
        print(f"Seed {seed} | Eval Variance (error var): {variance:.6f}")
