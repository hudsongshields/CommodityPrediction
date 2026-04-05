import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import random
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend for server-side stability.
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Path synchronization for local modules.
prefix = sys.prefix
dll_path = os.path.join(prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)

sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'diffusion'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'base'))

from models.dataset import get_dataloaders, get_walk_forward_dataloaders
from models.market_data import get_real_commodity_returns
from models.ds_tgnn import DiffusionReturnPrediction
from models.diffusion.diffusion_architecture import ScoreNetwork
from models.diffusion.loss_func import ScoreDiffusionLoss

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

def compute_benchmark_returns(targets):
    """Calculates the equal-weighted return across all commodities in the dataset."""
    return targets.mean(dim=1).numpy()

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
    epochs = config.get('epochs', 5)
    lr = config.get('lr', 0.001)
    batch_size = config.get('batch_size', 16)
    mc_samples = config.get('mc_samples', 1)
    mag_weighted = config.get('mag_weighted', True)
    walk_forward = config.get('walk_forward', False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if walk_forward:
        folds = get_walk_forward_dataloaders(batch_size=batch_size)
    else:
        tr, vl, ts = get_dataloaders(batch_size=batch_size)
        folds = [{"fold": 1, "train": tr, "val": vl, "test": ts}]

    score_loss_fn = ScoreDiffusionLoss().to(device)
    gamma = 0.1 # Scaling constant for the Score-Matching objective.

    all_p, all_t, all_s, all_v, all_d, fold_markers = [], [], [], [], [], []
    history = {"train_loss": [], "val_loss": []}

    for fold_data in folds:
        f_idx = fold_data['fold']
        print(f"  Processing Fold {f_idx}/{len(folds)}...")
        train_l, val_l, test_l = fold_data['train'], fold_data['val'], fold_data['test']
        
        # Grid dimensions based on the synchronized 14-hub dataset.
        N, T, F = train_l.dataset.dataset.n_hubs, 180, 4
        d_in = N * T * F
        
        # Spatial graph prior (fully connected network between weather hubs).
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                edges.append([i, j])
        e_idx = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        if e_idx.numel() > 0: 
            e_idx = torch.cat([e_idx, e_idx.flip(0)], dim=1)
        else: 
            e_idx = torch.zeros((2, 0), dtype=torch.long).to(device)

        score_net = ScoreNetwork(input_dim=d_in, mlp_hidden=[128], conv_hidden=32, t_hidden_dim=16, output_dim=d_in, use_conv=False)
        inc_den = config.get('include_denoised', False)
        model = DiffusionReturnPrediction(score_net, input_dim=F, lstm_hidden=32, gnn_hidden=32, n_hubs=N, include_denoised=inc_den).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            t_losses = []
            for batch_x, batch_y, batch_v, batch_d in train_l:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                
                # Multi-Task Optimization: Joint training of Score Matching and Alpha Prediction.
                # 1. Score matching objective ensures structural feature integrity.
                score_l = score_loss_fn(model.score_net, batch_x.transpose(1, 2)) 
                
                # 2. Regression objective minimizes prediction error for return alpha.
                p = model(batch_x, e_idx)
                reg_l = (p - batch_y)**2
                
                # Apply magnitude weighting to prioritize extreme market events.
                if mag_weighted: 
                    reg_l = reg_l * (1.0 + torch.abs(batch_y) * 10.0)
                
                # Combined Gradient Signal.
                total_l = reg_l.mean() + gamma * score_l
                total_l.backward()
                optimizer.step()
                t_losses.append(total_l.item())
                
            history["train_loss"].append(np.mean(t_losses))
            if (epoch+1) % 5 == 0 or epochs <= 5:
                print(f"  Fold {f_idx} | Epoch {epoch+1}/{epochs} | Total Loss: {history['train_loss'][-1]:.6f}")

        model.eval()
        if mc_samples > 1: model.enable_dropout()
        f_p, f_t, f_s, f_v, f_d = [], [], [], [], []
        for x, target, v_d, b_dt in test_l:
            x = x.to(device)
            with torch.no_grad():
                if mc_samples > 1:
                    preds = torch.stack([model(x, e_idx).cpu() for _ in range(mc_samples)])
                    pm, ps = preds.mean(0), preds.std(0)
                else:
                    pm = model(x, e_idx).cpu()
                    ps = torch.zeros_like(target)
            f_p.append(pm); f_t.append(target); f_s.append(ps); f_v.append(v_d); f_d.extend(b_dt)
            
        all_p.append(torch.cat(f_p).numpy()); all_t.append(torch.cat(f_t).numpy())
        all_s.append(torch.cat(f_s).numpy()); all_v.append(torch.cat(f_v).numpy()); all_d.extend(f_d)
        fold_markers.append(pd.to_datetime(f_d[0]))

    p_c, t_c, s_c, v_c = np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_s), np.concatenate(all_v)
    dates = pd.to_datetime(all_d)
    s_idx = np.argsort(dates)
    dates, p_c, t_c, s_c, v_c = dates[s_idx], p_c[s_idx], t_c[s_idx], s_c[s_idx], v_c[s_idx]
    
    # Performance assessment relative to benchmarks.
    strat_r = compute_strategy_returns(torch.tensor(p_c), torch.tensor(v_c))
    b_etf = None
    if get_real_commodity_returns:
        _, _, _, m_b, _ = get_real_commodity_returns()
        b_etf = m_b["DBA"].reindex(dates).ffill().values
    
    res = {**config, **calculate_metrics(p_c, t_c, strat_r)}
    
    # Uncertainty Correlation: Measures the alignment between model confidence and prediction accuracy.
    if mc_samples > 1: 
        res["Uncertainty_Corr"] = float(np.corrcoef(np.abs(p_c - t_c).ravel(), s_c.ravel())[0, 1])
    else: 
        res["Uncertainty_Corr"] = None
        
    # Store the final model weights for production deployment.
    os.makedirs('results', exist_ok=True)
    m_path = f"results/{config['name']}_weights.pt"
    torch.save(model.state_dict(), m_path)
    print(f"Model parameters saved to {m_path}")

    return res, history, p_c, t_c, s_c, strat_r, b_etf, dates, fold_markers

def run_standard_suite(fast_dev=False, walk_forward=False):
    """Executes the standardized research suite and generates visual reports."""
    os.makedirs('results', exist_ok=True); os.makedirs('plots', exist_ok=True)
    epochs = 2 if fast_dev else 50 
    prefix = "final_" if walk_forward else ""
    configs = [
        {"name": "Base_A_LSTM", "use_diffusion": False, "epochs": epochs, "walk_forward": walk_forward},
        {"name": "DS-TGNN_V2.3_Triple", "use_diffusion": True, "include_denoised": True, "mc_samples": 50, "epochs": epochs, "walk_forward": walk_forward},
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
