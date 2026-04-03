import numpy as np
import pandas as pd

def analyze_npz(path):
    data = np.load(path)
    # strat_returns is (T,)
    # market_returns (we need to assume fixed or load if available, but IR is often calculated on strat_returns)
    # Actually, the script saves strat_returns as the strategy's excess returns if it's coded that way.
    # Let's check Excess_IR logic: mean / std * sqrt(252) usually
    s_r = np.nan_to_num(data['strat_returns'])
    ir = np.mean(s_r) / (np.std(s_r) + 1e-6) * np.sqrt(252)
    
    # RMSE: mean(sqrt(mean((p-t)**2, axis=0)))
    p = data['preds']
    t = data['targets']
    rmse = np.mean(np.sqrt(np.mean((p - t)**2, axis=0)))
    
    return rmse, ir

# Load Triple (V2.1.1)
t_rmse, t_ir = analyze_npz('results/v2_DS-TGNN_V2.1.1_Triple_results.npz')

# Load Score (V2.1.2)
s_rmse, s_ir = analyze_npz('results/v2_DS-TGNN_V2.1.2_Score_results.npz')

print(f"| Metric | Triple-Signal (12-dim) | Score-Only (8-dim) | Delta |")
print(f"| :--- | :---: | :---: | :---: |")
print(f"| **Excess IR** | {t_ir:.4f} | {s_ir:.4f} | {s_ir - t_ir:+.4f} |")
print(f"| **Excess RMSE** | {t_rmse:.4f} | {s_rmse:.4f} | {s_rmse - t_rmse:+.4f} |")
