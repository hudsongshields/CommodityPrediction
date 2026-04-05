import os
import sys
import numpy as np
import pandas as pd
import random
import torch

# Avoid forcing extra DLL search paths for torch on Windows; this can load a
# second OpenMP runtime (libiomp5md.dll) in conda/pip mixed environments.

# Ensure repository root is on the import path when running this file directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data import get_dataloaders, get_walk_forward_dataloaders, get_real_commodity_returns
from models.return_prediction import train_dstgnn
from models.diffusion.diffusion_architecture import Diffusion
from models.diffusion.train_diffusion import train_diffusion

def set_seed(seed=42):
    """Sets global seeds to ensure experimental reproducibility."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False; os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to {seed}")

def compute_strategy_returns(preds, targets, k=2):
    """
    Simulates a Long/Short commodity strategy based on model predictions.
    
    The strategy selects the top 'k' assets to Buy (Long) and the bottom 'k'
    assets to Sell (Short) at each time step.
    """
    B, N = preds.shape
    strategy_pnls = []
    for i in range(B):
        p, t = preds[i], targets[i]
        idx = torch.argsort(p)
        s_idx, l_idx = idx[:k], idx[-k:]
        # Calculate the return spread: Long Average - Short Average.
        pnl = t[l_idx].mean() - t[s_idx].mean()
        strategy_pnls.append(pnl.item())
    return np.array(strategy_pnls)

def calculate_metrics(preds, targets, strategy_pnls):
    """
    Computes key performance indicators for model evaluation.
    
    Metrics:
        RMSE: Root Mean Squared Error of the raw return predictions.
        Information Ratio (IR): The mean strategic return divided by its 
                               standard deviation (volatility).
    """
    p_f, t_f = preds.ravel(), targets.ravel()
    rmse = np.sqrt(np.mean((p_f - t_f)**2))
    ir = np.mean(strategy_pnls) / np.std(strategy_pnls) if np.std(strategy_pnls) > 0 else 0.0
    p_rmse = [np.sqrt(np.mean((preds[:, j] - targets[:, j])**2)) for j in range(preds.shape[1])]
    return {"Excess_RMSE": float(rmse), "Excess_IR": float(ir), "per_asset_rmse": p_rmse}

def run_experiment(config):
    """
    Trains and evaluates a specific model configuration.
    """
    name = config['name']
    print(f"\nRunning Experiment: {name}")
    num_epochs = config.get('epochs', 5)
    learning_rate = config.get('lr', 0.001)
    batch_size = config.get('batch_size', 16)
    mc_samples = config.get('mc_samples', 1)
    magnitude_weighted = config.get('mag_weighted', True)
    walk_forward = config.get('walk_forward', False)
    use_diffusion = config.get('use_diffusion', True)
    pretrain_diffusion = config.get('pretrain_diffusion', use_diffusion)
    diffusion_pretrain_epochs = config.get('diffusion_pretrain_epochs', max(1, num_epochs // 2))
    diffusion_pretrain_lr = config.get('diffusion_pretrain_lr', learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if walk_forward:
        folds = get_walk_forward_dataloaders(batch_size=batch_size)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
        folds = [{"fold": 1, "train": train_loader, "val": val_loader, "test": test_loader}]

    train_cfg = {
        **config,
        "epochs": num_epochs,
        "lr": learning_rate,
        "mc_samples": mc_samples,
        "mag_weighted": magnitude_weighted,
        "pretrain_diffusion": False,
        "diffusion_pretrain_epochs": diffusion_pretrain_epochs,
        "diffusion_pretrain_lr": diffusion_pretrain_lr,
    }

    if use_diffusion and pretrain_diffusion:
        diffusion_model = Diffusion(
            t_hidden_dim=config.get("t_hidden_dim", 16),
            in_channels=1,
            base_channels=config.get("base_channels", 64),
            channel_mults=config.get("channel_mults", (1, 2, 4)),
            num_res_blocks=config.get("num_res_blocks", 2),
        ).to(device)

        diffusion_checkpoint = config.get(
            "diffusion_pretrained_path",
            os.path.join("results", f"{name}_diffusion_pretrained.pt"),
        )

        print(
            f"  Stage 1: training diffusion model for {diffusion_pretrain_epochs} "
            f"epochs at lr={diffusion_pretrain_lr}."
        )
        train_diffusion(
            diffusion_model=diffusion_model,
            train_loader=folds[0]["train"],
            device=device,
            epochs=diffusion_pretrain_epochs,
            learning_rate=diffusion_pretrain_lr,
            save_path=diffusion_checkpoint,
            print_every=config.get("diffusion_pretrain_print_every", 1),
        )
        train_cfg["diffusion_pretrained_path"] = diffusion_checkpoint

    if use_diffusion and not pretrain_diffusion and config.get("diffusion_pretrained_path"):
        train_cfg["diffusion_pretrained_path"] = config["diffusion_pretrained_path"]

    training_outputs = train_dstgnn(config=train_cfg, folds=folds, device=device)

    predictions = training_outputs["predictions"]
    targets = training_outputs["targets"]
    uncertainty = training_outputs["uncertainty"]
    valuations = training_outputs["valuations"]
    dates = training_outputs["dates"]
    fold_markers = training_outputs["fold_markers"]
    history = training_outputs["history"]
    model = training_outputs["model"]
    
    # Performance assessment relative to benchmarks.
    strategy_returns = compute_strategy_returns(torch.tensor(predictions), torch.tensor(valuations))
    benchmark_etf = None
    if get_real_commodity_returns:
        _, _, _, benchmark_df, _ = get_real_commodity_returns()
        benchmark_etf = benchmark_df["DBA"].reindex(dates).ffill().values
    
    res = {**config, **calculate_metrics(predictions, targets, strategy_returns)}
    
    # Uncertainty Correlation: Measures the alignment between model confidence and prediction accuracy.
    if mc_samples > 1: 
        res["Uncertainty_Corr"] = float(np.corrcoef(np.abs(predictions - targets).ravel(), uncertainty.ravel())[0, 1])
    else: 
        res["Uncertainty_Corr"] = None
        
    # Store the final model weights for production deployment.
    os.makedirs('results', exist_ok=True)
    model_path = f"results/{config['name']}_weights.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

    return res, history, predictions, targets, uncertainty, strategy_returns, benchmark_etf, dates, fold_markers

def run_standard_suite(fast_dev=False, walk_forward=False):
    """Executes the standardized research suite and generates visual reports."""
    os.makedirs('results', exist_ok=True); os.makedirs('plots', exist_ok=True)
    epochs = 2 if fast_dev else 50 
    prefix = "final_" if walk_forward else ""
    configs = [
        {
            "name": "Base_A_LSTM",
            "use_diffusion": False,
            "pretrain_diffusion": False,
            "epochs": epochs,
            "walk_forward": walk_forward,
        },
        {
            "name": "DS-TGNN_V2.3_Triple",
            "use_diffusion": True,
            "pretrain_diffusion": True,
            "diffusion_pretrain_epochs": epochs,
            "include_denoised": True,
            "mc_samples": 50,
            "epochs": epochs,
            "walk_forward": walk_forward,
        },
    ]
    sum_data = []
    for cfg in configs:
        res, hist, p, t, s, s_r, b_etf, dates, m_list = run_experiment(cfg)
        sum_data.append(res)
        np.savez(f"results/{prefix}{cfg['name']}_results.npz", preds=p, targets=t, stds=s, strat_returns=s_r, dates=dates)
        
    # Generate administrative performance summary.
    df = pd.DataFrame(sum_data)
    df.to_csv(f'results/{prefix}summary.csv', index=False)
    print("\n" + "="*50 + "\nFinal Performance Summary\n" + "="*50)
    print(df[['name', 'Excess_RMSE', 'Excess_IR', 'Uncertainty_Corr']].to_string(index=False))

if __name__ == "__main__":
    set_seed(42)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--walk_forward', action='store_true')
    args = parser.parse_args()
    run_standard_suite(fast_dev=args.fast_dev_run, walk_forward=args.walk_forward)
