import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- OS-LEVEL DLL FIX ---
prefix = sys.prefix
dll_path = os.path.join(prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path): os.add_dll_directory(dll_path)

# Add local modules path
sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'diffusion'))
sys.path.append(os.path.join(os.getcwd(), 'models', 'base'))

from dataset import get_dataloaders
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
    
    return {
        "RMSE": float(rmse),
        "R2": float(r2),
        "Tail_RMSE": float(tail_rmse),
        "Strategy_IR": float(ir)
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
        for x, y in train_l:
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
            for x, y in val_l:
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
        
    all_preds, all_stds, all_targets = [], [], []
    with torch.no_grad():
        for x, y in test_l:
            x, y = x.to(device), y.to(device)
            # MC-Dropout sampling
            mc_outputs = torch.stack([model(x, edge_index) for _ in range(mc_samples)])
            mean_p = mc_outputs.mean(dim=0)
            std_p = mc_outputs.std(dim=0)
            
            all_preds.append(mean_p.cpu())
            all_stds.append(std_p.cpu())
            all_targets.append(y.cpu())
            
    p_conc = torch.cat(all_preds).numpy()
    t_conc = torch.cat(all_targets).numpy()
    s_conc = torch.cat(all_stds).numpy()
    
    strat_returns = compute_strategy_returns(torch.tensor(p_conc), torch.tensor(t_conc))
    metrics = calculate_metrics(p_conc, t_conc, strat_returns)
    
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
        
    return result_entry, history, p_conc, t_conc, s_conc

def run_standard_suite(fast_dev=False):
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    epochs = 2 if fast_dev else 20
    
    # Configurations to run
    configs = [
        {"name": "Base_A_LSTM", "use_diffusion": False, "use_gnn": False, "use_lstm": True, "epochs": epochs},
        {"name": "Base_B_GNN", "use_diffusion": False, "use_gnn": True, "use_lstm": False, "epochs": epochs},
        {"name": "Ablation_NoDiff", "use_diffusion": False, "use_gnn": True, "use_lstm": True, "epochs": epochs},
        {"name": "Ablation_NoGNN", "use_diffusion": True, "use_gnn": False, "use_lstm": True, "epochs": epochs},
        {"name": "Ablation_NoWeighted", "use_diffusion": True, "use_gnn": True, "use_lstm": True, "mag_weighted": False, "epochs": epochs},
        {"name": "Full_V1.1", "use_diffusion": True, "use_gnn": True, "use_lstm": True, "mag_weighted": True, "mc_samples": 50, "epochs": epochs},
    ]
    
    summary_data = []
    
    for cfg in configs:
        res, hist, p, t, s = run_experiment(cfg)
        summary_data.append(res)
        
        # Plot single experiment results (Loss Curves)
        plt.figure()
        plt.plot(hist['train_loss'], label='Train')
        plt.plot(hist['val_loss'], label='Val')
        plt.title(f"Learning Curves: {cfg['name']}")
        plt.legend()
        plt.savefig(f"plots/loss_{cfg['name']}.png")
        plt.close()

        # If it's the Full_V1.1 model, save raw data for detailed notebook analysis
        if cfg['name'] == "Full_V1.1":
            np.savez('results/v1_1_detailed_results.npz', preds=p, targets=t, stds=s)
        
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
    full_res_path = 'results/v1_1_detailed_results.npz'
    if os.path.exists(full_res_path):
        data = np.load(full_res_path)
        p, s, t = data['preds'].ravel(), data['stds'].ravel(), data['targets'].ravel()
        abs_err = np.abs(p - t)
        
        plt.figure(figsize=(10, 8))
        plt.hexbin(s, abs_err, gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(label='Sample Density')
        plt.xlabel('Model Uncertainty (MC-Dropout Std)')
        plt.ylabel('Absolute Prediction Error')
        plt.title('Uncertainty Calibration: Error vs. Confidence (Full_V1.1)')
        
        # Add regression trend line
        sns.regplot(x=s, y=abs_err, scatter=False, color='blue', label='Calibration Trend')
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/uncertainty_calibration_hexbin.png")
        print("📊 Uncertainty calibration hexbin saved to plots/uncertainty_calibration_hexbin.png")
        
    # Final Result Table to Console
    print("\n" + "="*50)
    print("      FINAL EXPERIMENT SUMMARY (V1.2)")
    print("="*50)
    print(df[['name', 'RMSE', 'Tail_RMSE', 'Strategy_IR', 'Uncertainty_Corr']].to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', action='store_true')
    args = parser.parse_args()
    
    run_standard_suite(fast_dev=args.fast_dev_run)
