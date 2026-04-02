import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json

# --- 🛠️ OS-LEVEL DLL FIX (REQUIRED FOR WINDOWS TORCH) ---
prefix = sys.prefix
dll_path = os.path.join(prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(dll_path): os.add_dll_directory(dll_path)

# Import local modules
sys.path.append(os.path.join(os.getcwd(), 'models'))
try:
    from dataset import get_dataloaders
    from ds_tgnn_train import train_controller_ds_tgnn, build_supply_chain_graph
    from ds_tgnn import DiffusionReturnPrediction
    from diffusion.diffusion_architecture import Diffusion
    from diffusion.loss_func import ScoreDiffusionLoss
except ImportError:
    print("⚠️ Local imports failed; using fallback definitions...")
    # Add minimalist fallbacks here if necessary

def run_evaluation_suite():
    print("🚀 Initializing DS-TGNN V1.1 Automated Evaluation Suite...")
    edge_index = torch.tensor([[0,1,2,3,4,0,1,5],[3,4,0,1,2,5,6,7]], dtype=torch.long)
    
    results = {}
    configs = [
        ("A: Baseline", False, False, 1),
        ("B: Embargo", True, False, 1),
        ("C: MC-Dropout", True, False, 20),
        ("D: Full V1.1", True, True, 20)
    ]
    
    for name, embargo, weight, mc in configs:
        print(f"\\n--- Running {name} ---")
        # For this script we assume 10 epochs
        train_l, val_l, test_l = get_dataloaders(use_embargo=embargo)
        model = DiffusionReturnPrediction(Diffusion()).train()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(5):
            for x, y in train_l:
                opt.zero_grad(); p = model(x, edge_index); loss = (p-y)**2
                if weight: loss = loss * (1.0 + torch.abs(y)*10.0)
                loss.mean().backward(); opt.step()
        
        # Evaluate
        model.eval(); if mc > 1: model.enable_dropout()
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in test_l:
                bp = torch.stack([model(x, edge_index) for _ in range(mc)]).mean(dim=0)
                all_p.append(bp); all_y.append(y)
        p_conc = torch.cat(all_p).numpy().ravel(); y_conc = torch.cat(all_y).numpy().ravel()
        results[name] = {"RMSE": float(np.sqrt(np.mean((p_conc-y_conc)**2)))}
        print(f"Result: {results[name]}")
        
    with open('artifacts/v1_vs_v1_1/experiment_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\\n🎯 Metrics saved to artifacts/v1_vs_v1_1/experiment_metrics.json")

if __name__ == "__main__":
    run_evaluation_suite()
