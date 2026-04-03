import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_analyze(file_path, label):
    data = np.load(file_path, allow_pickle=True)
    p = data['preds']
    t = data['targets']
    s_r = data['strat_returns']
    dates = data['dates']
    
    # Reconstructed split points based on 5-fold WalkForward gaps
    split_indices = [0, 43, 86, 129, 172, len(dates)]
    
    fold_irs = []
    fold_rmses = []
    for i in range(len(split_indices) - 1):
        s, e = split_indices[i], split_indices[i+1]
        f_sr = s_r[s:e]
        f_p = p[s:e]
        f_t = t[s:e]
        
        # Calculate per-fold IR and RMSE
        ir = np.mean(f_sr) / (np.std(f_sr) + 1e-6) * np.sqrt(252)
        rmse = np.sqrt(np.mean((f_p - f_t)**2))
        fold_irs.append(ir)
        fold_rmses.append(rmse)
        
    df = pd.DataFrame({
        'Fold': range(1, len(fold_irs) + 1),
        'Excess_IR': fold_irs,
        'Excess_RMSE': fold_rmses,
        'Model': label
    })
    return df

results_dir = 'results'
files = {
    'Baseline': 'v2_Base_A_LSTM_results.npz',
    'Score-Only': 'v2_DS-TGNN_V2.1.2_Score_results.npz',
    'Triple-Signal': 'v2_DS-TGNN_V2.1.1_Triple_results.npz'
}

all_dfs = []
for label, filename in files.items():
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        all_dfs.append(load_and_analyze(path, label))

full_df = pd.concat(all_dfs)

# Compute Statistics
stability_stats = full_df.groupby('Model')['Excess_IR'].agg(['mean', 'std', 'min', 'max']).reset_index()
stability_stats['Stability_Ratio'] = stability_stats['mean'] / (stability_stats['std'] + 1e-6)

print("\n=== GENERALIZATION AUDIT: STABILITY ACROSS 5 FOLDS ===")
print(stability_stats.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
for label in files.keys():
    subset = full_df[full_df['Model'] == label]
    plt.plot(subset['Fold'], subset['Excess_IR'], marker='o', label=label, linewidth=2)

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.title('Architectural Stability: Excess IR across 5 Walk-Forward Folds', fontsize=14)
plt.xlabel('Fold (Time Window)', fontsize=12)
plt.ylabel('Excess Information Ratio', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('plots/stability_audit_fold_variance.png')
print(f"\n✅ Stability Dashboard saved to plots/stability_audit_fold_variance.png")
