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

def set_seed(seed=42):
    """
    Forces all random number generators to use a fixed seed
    to ensure perfectly reproducible research results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Global Seed Set: {seed} (Deterministic Mode Active)")

# --- OS-LEVEL DLL FIX ---
prefix = sys.prefix
dll_path = os.path.join(prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path): os.add_dll_directory(dll_path)

# Add local modules path
sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'diffusion'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'base'))

from dataset import get_dataloaders
try:
    from market_data import get_real_commodity_returns
except ImportError:
    get_real_commodity_returns = None
from ds_tgnn import DiffusionReturnPrediction
from diffusion.diffusion_architecture import Diffusion
from diffusion.loss_func import ScoreDiffusionLoss

def compute_strategy_returns(preds, targets, k=2):
    """
    Simulates a long/short strategy:
    Go long on top-k predicted returns, short on bottom-k predicted returns.
    Calculation: (Realized Return of Longs) - (Realized Return of Shorts)
    """
    B, N = preds.shape
    strategy_pnls = []
    
    for i in range(B):
        p = preds[i]
        t = targets[i]
        
        # Get indices of top-k and bottom-k predictions
        indices = torch.argsort(p)
        short_idx = indices[:k]
        long_idx = indices[-k:]
        
        # Mean realized return of longs - mean realized return of shorts
        pnl = t[long_idx].mean() - t[short_idx].mean()
        strategy_pnls.append(pnl.item())
        
    return np.array(strategy_pnls)

def compute_benchmark_returns(targets):
    """
    Benchmark 1: Equal-Weighted (EW) basket of all commodities.
    """
    return targets.mean(dim=1).numpy()

def compute_weighted_benchmark_returns(targets):
    """
    Benchmark 2: Production-Weighted Index (BCOM Proxy).
    Weights prioritize critical commodities like Corn, Soybeans, and Gas.
    """
    # Weights for the 8 commodities: Corn, Soy, Wheat, Cattle, Hogs, Ethanol, Gas, Cotton
    weights = torch.tensor([0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.15, 0.10])
    return (targets * weights).sum(dim=1).numpy()

def calculate_metrics(preds, targets, strategy_pnls):
    """
    Computes regression metrics (RMSE, R2) and strategy metrics (IR).
    """
    preds_flat = preds.ravel()
    targets_flat = targets.ravel()
    
    rmse = np.sqrt(np.mean((preds_flat - targets_flat)**2))
    
    # R2
    ss_res = np.sum((targets_flat - preds_flat)**2)
    ss_tot = np.sum((targets_flat - np.mean(targets_flat))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Tail RMSE (Top 10% magnitude realized returns)
    threshold = np.percentile(np.abs(targets_flat), 90)
    tail_mask = np.abs(targets_flat) > threshold
    if np.any(tail_mask):
        tail_rmse = np.sqrt(np.mean((preds_flat[tail_mask] - targets_flat[tail_mask])**2))
    else:
        tail_rmse = 0.0
        
    # Strategy IR
    ir = np.mean(strategy_pnls) / np.std(strategy_pnls) if np.std(strategy_pnls) > 0 else 0.0
    
    # Per-Asset RMSE (granular break-down)
    B, N = preds.shape
    per_asset_rmse = []
    for j in range(N):
        per_asset_rmse.append(np.sqrt(np.mean((preds[:, j] - targets[:, j])**2)))
        
    return {
        "RMSE": float(rmse),
        "R2": float(r2),
        "Tail_RMSE": float(tail_rmse),
        "Strategy_IR": float(ir),
        "per_asset_rmse": per_asset_rmse
    }

def run_experiment(config):
    """
    Runs a single experiment configuration.
    """
    name = config['name']
    print(f"\n>>> Running Experiment: {name}")
    
    # Unpack hyperparams
    epochs = config.get('epochs', 5)
    lr = config.get('lr', 0.001)
    batch_size = config.get('batch_size', 16)
    mc_samples = config.get('mc_samples', 1)
    use_embargo = config.get('use_embargo', True)
    mag_weighted = config.get('mag_weighted', True)
    
    # Model components
    use_diffusion = config.get('use_diffusion', True)
    use_lstm = config.get('use_lstm', True)
    use_gnn = config.get('use_gnn', True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = torch.tensor([[0,1,2,3,4,0,1,5],[3,4,0,1,2,5,6,7]], dtype=torch.long).to(device)
    
    train_l, val_l, test_l = get_dataloaders(batch_size=batch_size, use_embargo=use_embargo)
    
    # Setup model
    # Note: input_dim = N * T * F for the flattened denoiser
    N, T, F = 8, 180, 5
    d_input_dim = N * T * F
    denoiser = Diffusion(
        input_dim=d_input_dim,
        mlp_hidden=[64],
        conv_hidden=32,
        t_hidden_dim=16,
        output_dim=d_input_dim,
        use_conv=False
    )
    
    model = DiffusionReturnPrediction(
        denoiser, 
        input_dim=F, 
        lstm_hidden=32, 
        gnn_hidden=32,
        use_diffusion=use_diffusion,
        use_lstm=use_lstm,
        use_gnn=use_gnn
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y, v, d in train_l:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            p = model(x, edge_index)
            loss = (p - y)**2
            if mag_weighted:
                loss = loss * (1.0 + torch.abs(y) * 10.0)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, v, d in val_l:
                x, y = x.to(device), y.to(device)
                p = model(x, edge_index)
                v_loss = torch.mean((p - y)**2).item()
                val_losses.append(v_loss)
        
        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        if (epoch+1) % 5 == 0 or epochs <= 5:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.6f} | Val Loss: {history['val_loss'][-1]:.6f}")

    # Final Evaluation on Test Set
    model.eval()
    if mc_samples > 1:
        model.enable_dropout()
    
    all_preds, all_stds, all_targets, all_daily_val, all_dates = [], [], [], [], []
    for batch in test_l:
        x, target, daily_val, batch_dates = batch
        x = x.to(device)
        
        with torch.no_grad():
            if mc_samples > 1:
                model.enable_dropout()
                preds = [model(x, edge_index).cpu() for _ in range(mc_samples)]
                preds = torch.stack(preds) # [MC, B, N]
                pred_mean = preds.mean(dim=0)
                pred_std = preds.std(dim=0)
            else:
                pred_mean = model(x, edge_index).cpu()
                pred_std = torch.zeros_like(pred_mean)
        
        all_preds.append(pred_mean)
        all_targets.append(target)
        all_stds.append(pred_std)
        all_daily_val.append(daily_val)
        all_dates.extend(batch_dates)

    p_conc = torch.cat(all_preds).numpy()
    t_conc = torch.cat(all_targets).numpy()
    s_conc = torch.cat(all_stds).numpy()
    v_conc = torch.cat(all_daily_val).numpy()
    test_dates = pd.to_datetime(all_dates)
    
    # Strat returns using DAILY valuations (Arithmetic mean of top/bottom)
    strat_daily = compute_strategy_returns(torch.tensor(p_conc), torch.tensor(v_conc))
    
    # Benchmarks using DAILY valuations
    bench_daily_ew = compute_benchmark_returns(torch.tensor(v_conc))
    bench_daily_w = compute_weighted_benchmark_returns(torch.tensor(v_conc))
    
    # Fetch real ETF benchmark DAILY return (DBA) if available (V1.11)
    bench_returns_etf = None
    if get_real_commodity_returns:
        # v1.9/V1.11: Institutional Valuation Sync
        test_dates_dt = pd.to_datetime(test_dates)
        _, _, _, m_bench_daily, _ = get_real_commodity_returns()
        bench_returns_etf = m_bench_daily["DBA"].reindex(test_dates_dt).ffill().values
    
    metrics = calculate_metrics(p_conc, t_conc, strat_daily)
    
    # --- CHRONOLOGICAL SORTING (V1.6) ---
    # Ensure all test-set results are ordered by Date to prevent "spaghetti" plots
    sort_idx = np.argsort(test_dates)
    test_dates = test_dates[sort_idx]
    strat_daily = strat_daily[sort_idx]
    bench_daily_ew = bench_daily_ew[sort_idx]
    bench_daily_w = bench_daily_w[sort_idx]
    if bench_returns_etf is not None:
        bench_returns_etf = bench_returns_etf[sort_idx]
    
    # Store results
    result_entry = {**config, **metrics}
    # Calculate uncertainty correlation if MC-Dropout was used
    if mc_samples > 1:
        abs_err = np.abs(p_conc - t_conc).ravel()
        uncertainty = s_conc.ravel()
        corr = np.corrcoef(abs_err, uncertainty)[0, 1]
        result_entry["Uncertainty_Corr"] = float(corr)
    else:
        result_entry["Uncertainty_Corr"] = None
        
    return result_entry, history, p_conc, t_conc, s_conc, strat_daily, bench_daily_ew, bench_daily_w, bench_returns_etf, test_dates

def run_standard_suite(fast_dev=False):
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    epochs = 2 if fast_dev else 20
    commodity_names = ["Corn", "Soybeans", "Wheat", "Cattle", "Hogs", "Ethanol", "NatGas", "Cotton"]
    
    configs = [
        {"name": "Base_A_LSTM", "use_diffusion": False, "use_gnn": False, "use_lstm": True, "epochs": epochs},
        {"name": "Base_B_GNN", "use_diffusion": False, "use_gnn": True, "use_lstm": False, "epochs": epochs},
        {"name": "Ablation_NoDiff", "use_diffusion": False, "use_gnn": True, "use_lstm": True, "epochs": epochs},
        {"name": "Ablation_NoGNN", "use_diffusion": True, "use_gnn": False, "use_lstm": True, "epochs": epochs},
        {"name": "Ablation_NoWeighted", "use_diffusion": True, "use_gnn": True, "use_lstm": True, "mag_weighted": False, "epochs": epochs},
        {"name": "DS-TGNN_Full", "use_diffusion": True, "use_gnn": True, "use_lstm": True, "mag_weighted": True, "mc_samples": 50, "epochs": epochs},
    ]
    
    summary_data = []
    
    for cfg in configs:
        res, hist, p, t, s, strat_r, b_ew, b_w, b_etf, dates = run_experiment(cfg)
        summary_data.append(res)
        
        # Plot single experiment results (Loss Curves)
        plt.figure()
        plt.plot(hist['train_loss'], label='Train')
        plt.plot(hist['val_loss'], label='Val')
        plt.title(f"Learning Curves: {cfg['name']}")
        plt.legend()
        plt.savefig(f"plots/loss_{cfg['name']}.png")
        plt.close()

        # If it's the Full model, generate detailed Dashboard and save raw results
        if "Full" in cfg['name']:
            np.savez('results/ds_tgnn_detailed_results.npz', 
                     preds=p, targets=t, stds=s, 
                     strat_returns=strat_r, 
                     test_dates=dates,
                     bench_returns_etf=b_etf)
            
            # --- COMPREHENSIVE PERFORMANCE DASHBOARD ---
            fig, axes = plt.subplots(2, 2, figsize=(20, 14))
            
            # 1. Equity Curves (Cumulative Return)
            def to_percentage_return(daily_returns):
                # Standard Wealth Index -> Percentage Return
                wealth_index = np.cumprod(1 + np.nan_to_num(daily_returns))
                return (wealth_index - 1.0) * 100

            axis = axes[0, 0]
            axis.plot(dates, to_percentage_return(strat_r), label='DS-TGNN Strategy', color='blue', linewidth=2)
            axis.plot(dates, to_percentage_return(b_ew), label='Market (Equal-Weighted)', color='gray', alpha=0.5, linestyle='--')
            axis.plot(dates, to_percentage_return(b_w), label='Market (BCOM-Weighted Proxy)', color='red', alpha=0.5, linestyle=':')
            if b_etf is not None:
                axis.plot(dates, to_percentage_return(b_etf), label='Real Market (Invesco DBA Fund)', color='orange', linewidth=2)
            
            axis.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1) # Zero Baseline
            axis.set_title('Total Cumulative Return (%)', fontsize=12)
            axis.set_ylabel('Return (%)', fontsize=10)
            axis.legend(fontsize=9)
            axis.grid(True, alpha=0.3)
            
            # 2. Cumulative Alphas (Percentage Points Spread)
            axis = axes[0, 1]
            def to_wealth_only(daily_returns):
                return np.cumprod(1 + np.nan_to_num(daily_returns))

            s_wealth = to_wealth_only(strat_r)
            axis.plot(dates, (s_wealth - to_wealth_only(b_ew)) * 100, label='vs. Equal-Weighted', color='gray', alpha=0.3)
            axis.plot(dates, (s_wealth - to_wealth_only(b_w)) * 100, label='vs. Institutional-Weighted', color='red', alpha=0.3)
            
            if b_etf is not None:
                alpha_etf = (s_wealth - to_wealth_only(b_etf)) * 100
                axis.plot(dates, alpha_etf, label='vs. REAL Real Market (DBA Fund)', color='orange', linewidth=2.5)
                axis.fill_between(dates, 0, alpha_etf, color='orange', alpha=0.05)
                
            axis.set_title('Strategic Wealth Alpha (%)', fontsize=12)
            axis.set_ylabel('Alpha (Percentage Points)', fontsize=10)
            axis.legend()
            axis.grid(True, alpha=0.3)
            
            # 3. Standardized Asset-Wise Accuracy Score (Predictive Edge)
            # Normalizing RMSE by the Historical Volatility (Std Dev) of each asset
            # This makes the accuracy comparable across high-vol (NatGas) and low-vol (Soybeans) assets.
            
            # targets 't' has shape [Batch, N]
            asset_volatility = np.std(t, axis=0) # Natural volatility of each asset
            asset_rmse = np.array(res['per_asset_rmse'])
            
            # Predictive Edge = Volatility / RMSE 
            # > 1.0 means the model is better than a 'zero-knowledge' volatility baseline
            predictive_edge = asset_volatility / (asset_rmse + 1e-6)
            
            axes[1, 0].bar(commodity_names, predictive_edge, color='teal', alpha=0.7)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
            axes[1, 0].set_title('Standardized Predictive Edge (Volatility / RMSE)')
            axes[1, 0].set_ylabel('Edge Index (>1.0 is Good)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
            
            # 4. Error Distribution (Residuals)
            residuals = (p - t).ravel()
            sns.histplot(residuals, bins=50, kde=True, ax=axes[1, 1], color='purple')
            axes[1, 1].vlines(0, 0, axes[1, 1].get_ylim()[1], color='red', linestyle='--') # zero error line
            axes[1, 1].set_title('Prediction Error Distribution (Residual Histogram)')
            axes[1, 1].set_xlabel('Absolute Error (Pred - Target)')
            
            plt.tight_layout()
            plt.savefig("plots/performance_dashboard.png")
            print("🚀 Deep Integrity Performance Dashboard saved to plots/performance_dashboard.png")
        
    # Save Summary Table
    df = pd.DataFrame(summary_data)
    df.to_csv('results/experiment_summary.csv', index=False)
    print("\n✅ Results saved to results/experiment_summary.csv")
    
    # Generate Comparison Plot: Overall RMSE vs Tail RMSE
    plt.figure(figsize=(10, 6))
    df.plot(x='name', y=['RMSE', 'Tail_RMSE'], kind='bar')
    plt.title("Performance Comparison: Overall vs Tail RMSE")
    plt.ylabel("Loss (MSE-like)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/tail_performance_comparison.png")
    
    print("📈 Comparison plots saved to plots/")
    
    # Generate Uncertainty Calibration Plot (Hexbin)
    full_res_path = 'results/ds_tgnn_detailed_results.npz'
    if os.path.exists(full_res_path):
        data = np.load(full_res_path)
        p, s, t = data['preds'].ravel(), data['stds'].ravel(), data['targets'].ravel()
        abs_err = np.abs(p - t)
        
        plt.figure(figsize=(10, 8))
        plt.hexbin(s, abs_err, gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(label='Sample Density')
        plt.xlabel('Model Uncertainty (MC-Dropout Std)')
        plt.ylabel('Absolute Prediction Error')
        plt.title('Uncertainty Calibration: Error vs. Confidence (Full Model)')
        
        # Add regression trend line
        sns.regplot(x=s, y=abs_err, scatter=False, color='blue', label='Calibration Trend')
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/uncertainty_calibration_hexbin.png")
        print("📊 Uncertainty calibration hexbin saved to plots/uncertainty_calibration_hexbin.png")
        
    # Final Result Table to Console
    print("\n" + "="*50)
    print("      DEEP SPATIOTEMPORAL RESEARCH SUMMARY")
    print("="*50)
    print(df[['name', 'RMSE', 'Tail_RMSE', 'Strategy_IR', 'Uncertainty_Corr']].to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    # 1. Force absolute reproducibility
    set_seed(42)
    
    # Check for fast dev run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', action='store_true')
    args = parser.parse_args()
    
    # The suite itself handles the epoch logic based on fast_dev_run
    run_standard_suite(fast_dev=args.fast_dev_run)
