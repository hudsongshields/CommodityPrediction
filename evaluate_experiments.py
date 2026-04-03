import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import random
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for headless stability
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- OS-LEVEL DLL FIX ---
prefix = sys.prefix
dll_path = os.path.join(prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)

# Add local modules path
sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'diffusion'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'base'))

from dataset import get_dataloaders, get_walk_forward_dataloaders
from market_data import get_real_commodity_returns
from ds_tgnn import DiffusionReturnPrediction
from diffusion.diffusion_architecture import Diffusion
from diffusion.loss_func import ScoreDiffusionLoss

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False; os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Global Seed Set: {seed}")

def compute_strategy_returns(preds, targets, k=2):
    """Long/Short Strategy simulation for Excess Returns."""
    B, N = preds.shape; strategy_pnls = []
    for i in range(B):
        p, t = preds[i], targets[i]; idx = torch.argsort(p)
        s_idx, l_idx = idx[:k], idx[-k:]
        pnl = t[l_idx].mean() - t[s_idx].mean(); strategy_pnls.append(pnl.item())
    return np.array(strategy_pnls)

def compute_benchmark_returns(targets): return targets.mean(dim=1).numpy()

def calculate_metrics(preds, targets, strategy_pnls):
    p_f, t_f = preds.ravel(), targets.ravel()
    rmse = np.sqrt(np.mean((p_f - t_f)**2))
    ir = np.mean(strategy_pnls) / np.std(strategy_pnls) if np.std(strategy_pnls) > 0 else 0.0
    p_rmse = [np.sqrt(np.mean((preds[:, j] - targets[:, j])**2)) for j in range(preds.shape[1])]
    return {"Excess_RMSE": float(rmse), "Excess_IR": float(ir), "per_asset_rmse": p_rmse}

def run_experiment(config):
    name = config['name']; print(f"\n>>> Running Experiment: {name}")
    epochs = config.get('epochs', 5); lr = config.get('lr', 0.001); batch_size = config.get('batch_size', 16)
    mc_samples = config.get('mc_samples', 1); mag_weighted = config.get('mag_weighted', True)
    walk_forward = config.get('walk_forward', False); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Final 14-Hub Global Supply Network
    # US (0-7), SA (8-10), EUR (11-13)
    edges = []
    us_i, sa_i, eur_i = [0,1,2,3,4,5,6,7], [8,9,10], [11,12,13]
    for zone in [us_i, sa_i, eur_i]:
        for i in zone:
            for j in zone:
                if i < j: edges.append([i, j])
    edges.extend([[7, 11], [0, 8], [5, 11]]) # Global Connectivity
    e_idx = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    e_idx = torch.cat([e_idx, e_idx.flip(0)], dim=1)

    if walk_forward:
        folds = get_walk_forward_dataloaders(batch_size=batch_size)
    else:
        tr, vl, ts = get_dataloaders(batch_size=batch_size); folds = [{"fold": 1, "train": tr, "val": vl, "test": ts}]

    score_loss_fn = ScoreDiffusionLoss().to(device)
    gamma = 0.1 # Diffusion Penalty weight

    all_p, all_t, all_s, all_v, all_d, fold_markers = [], [], [], [], [], []; history = {"train_loss": [], "val_loss": []}

    for fold_data in folds:
        f_idx = fold_data['fold']; print(f"  --- Processing Fold {f_idx}/{len(folds)} ---")
        train_l, val_l, test_l = fold_data['train'], fold_data['val'], fold_data['test']
        
        N, T, F = 14, 180, 4; d_in = N * T * F
        den = Diffusion(input_dim=d_in, mlp_hidden=[128], conv_hidden=32, t_hidden_dim=16, output_dim=d_in, use_conv=False)
        inc_den = config.get('include_denoised', False)
        model = DiffusionReturnPrediction(den, input_dim=F, lstm_hidden=32, gnn_hidden=32, n_hubs=N, include_denoised=inc_den).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train(); t_l = []
            for x, y, v, d in train_l:
                x, y = x.to(device), y.to(device); opt.zero_grad()
                diff_l = score_loss_fn(model.denoiser, x.transpose(1, 2)) # Canonical [B, N, T, F]
                p = model(x, e_idx)
                reg_l = (p - y)**2
                if mag_weighted: reg_l = reg_l * (1.0 + torch.abs(y) * 10.0)
                total_l = reg_l.mean() + gamma * diff_l
                total_l.backward(); opt.step(); t_l.append(total_l.item())
            history["train_loss"].append(np.mean(t_l))
            if (epoch+1) % 5 == 0 or epochs <= 5:
                print(f"  Fold {f_idx} | Ep {epoch+1}/{epochs} | Total-Loss: {history['train_loss'][-1]:.6f}")

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
                    pm = model(x, e_idx).cpu(); ps = torch.zeros_like(target)
            f_p.append(pm); f_t.append(target); f_s.append(ps); f_v.append(v_d); f_d.extend(b_dt)
        all_p.append(torch.cat(f_p).numpy()); all_t.append(torch.cat(f_t).numpy())
        all_s.append(torch.cat(f_s).numpy()); all_v.append(torch.cat(f_v).numpy()); all_d.extend(f_d)
        fold_markers.append(pd.to_datetime(f_d[0]))

    p_c, t_c, s_c, v_c = np.concatenate(all_p), np.concatenate(all_t), np.concatenate(all_s), np.concatenate(all_v)
    dates = pd.to_datetime(all_d); s_idx = np.argsort(dates)
    dates, p_c, t_c, s_c, v_c = dates[s_idx], p_c[s_idx], t_c[s_idx], s_c[s_idx], v_c[s_idx]
    
    strat_r = compute_strategy_returns(torch.tensor(p_c), torch.tensor(v_c))
    b_ew = compute_benchmark_returns(torch.tensor(v_c))
    b_etf = None
    if get_real_commodity_returns:
        _, _, _, m_b, _ = get_real_commodity_returns(); b_etf = m_b["DBA"].reindex(dates).ffill().values
    
    res = {**config, **calculate_metrics(p_c, t_c, strat_r)}
    if mc_samples > 1: res["Uncertainty_Corr"] = float(np.corrcoef(np.abs(p_c - t_c).ravel(), s_c.ravel())[0, 1])
    else: res["Uncertainty_Corr"] = None
    return res, history, p_c, t_c, s_c, strat_r, b_ew, b_etf, dates, fold_markers

def run_standard_suite(fast_dev=False, walk_forward=False):
    os.makedirs('results', exist_ok=True); os.makedirs('plots', exist_ok=True)
    epochs = 2 if fast_dev else 50 # Final Score-Only Research Spec (V2.1.2)
    c_names = ["Corn", "Soybeans", "Wheat", "Cattle", "Hogs", "Ethanol", "NatGas", "Cotton"]
    prefix = "v2_" if walk_forward else ""
    configs = [
        {"name": "Base_A_LSTM", "use_diffusion": False, "use_gnn": False, "use_lstm": True, "epochs": epochs, "walk_forward": walk_forward},
        {"name": "DS-TGNN_V2.1.1_Triple", "use_diffusion": True, "use_gnn": True, "use_lstm": True, "include_denoised": True, "mc_samples": 50, "epochs": epochs, "walk_forward": walk_forward},
        {"name": "DS-TGNN_V2.1.2_Score", "use_diffusion": True, "use_gnn": True, "use_lstm": True, "include_denoised": False, "mc_samples": 50, "epochs": epochs, "walk_forward": walk_forward},
    ]
    sum_data = []; d_name = f"{prefix}alpha_dashboard.png"
    for cfg in configs:
        res, hist, p, t, s, s_r, b_ew, b_etf, dates, m_list = run_experiment(cfg)
        sum_data.append(res)
        np.savez(f"results/{prefix}{cfg['name']}_results.npz", preds=p, targets=t, stds=s, strat_returns=s_r, dates=dates)
        if "V2" in cfg['name']:
            fig, axarr = plt.subplots(2, 2, figsize=(20, 14))
            ax = axarr[0, 0]
            def to_pct(r): return (np.cumprod(1 + np.nan_to_num(r)) - 1.0) * 100
            ax.plot(dates, to_pct(s_r), label='V2 Strategy (Alpha)', color='blue', lw=2)
            ax.plot(dates, to_pct(b_ew), label='Market EW', color='gray', alpha=0.5, ls='--')
            if walk_forward:
                for m in m_list: ax.axvline(m, color='green', alpha=0.3, ls=':', lw=1)
            ax.axhline(0, color='black', alpha=0.3); ax.set_title(f"Cumulative Excess Return ({cfg['name']})"); ax.legend(); ax.grid(True, alpha=0.3)
            ax = axarr[0, 1]; sw, mw = np.cumprod(1 + np.nan_to_num(s_r)), np.cumprod(1 + np.nan_to_num(b_ew))
            ax.plot(dates, (sw - mw) * 100, label='Pure Alpha vs Bench', color='blue', lw=2.5)
            ax.fill_between(dates, 0, (sw - mw)*100, color='blue', alpha=0.08)
            ax.set_title('Exogenous Outperformance (%)'); ax.legend(); ax.grid(True, alpha=0.3)
            e = np.std(t, 0) / (np.array(res['per_asset_rmse']) + 1e-6)
            axarr[1, 0].bar(c_names, e, color='teal', alpha=0.7); axarr[1, 0].axhline(1.0, color='red', ls='--')
            axarr[1, 0].set_title('Alpha Edge (Vol / RMSE)'); axarr[1, 0].tick_params(axis='x', rotation=45)
            sns.histplot((p - t).ravel(), bins=50, kde=True, ax=axarr[1, 1], color='purple')
            axarr[1, 1].set_title('Excess Error Distribution'); plt.tight_layout(); plt.savefig(f"plots/{d_name}")
    df = pd.DataFrame(sum_data); df.to_csv(f'results/{prefix}summary.csv', index=False)
    print(f"\n" + "="*50 + f"\n      V2.1 RESEARCH SUMMARY (N=14 Hubs)\n" + "="*50)
    print(df[['name', 'Excess_RMSE', 'Excess_IR', 'Uncertainty_Corr']].to_string(index=False))

if __name__ == "__main__":
    set_seed(42); import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--walk_forward', action='store_true')
    args = parser.parse_args(); run_standard_suite(fast_dev=args.fast_dev_run, walk_forward=args.walk_forward)
