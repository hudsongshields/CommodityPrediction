import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_res(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found.")
        return None, None, None, None
    d = np.load(path)
    return d['strat_returns'], d['dates'], d['preds'], d['targets']

# Load Datasets
tr_r, tr_d, tr_p, tr_t = load_res('results/v2_DS-TGNN_V2.1.1_Triple_results.npz')
sc_r, sc_d, sc_p, sc_t = load_res('results/v2_DS-TGNN_V2.1.2_Score_results.npz')
bs_r, bs_d, bs_p, bs_t = load_res('results/v2_Base_A_LSTM_results.npz')

# Fix dates
tr_d = pd.to_datetime(tr_d) if tr_d is not None else None
sc_d = pd.to_datetime(sc_d) if sc_d is not None else None
bs_d = pd.to_datetime(bs_d) if bs_d is not None else None

def to_wealth(r): return (np.cumprod(1 + np.nan_to_num(r)) - 1.0) * 100

fig, axarr = plt.subplots(2, 2, figsize=(22, 14))
sns.set_style("darkgrid")

# 1. Cumulative Performance
ax = axarr[0, 0]
if tr_r is not None: ax.plot(tr_d, to_wealth(tr_r), label='V2.1.1 Triple (12-dim)', color='blue', lw=2.5)
if sc_r is not None: ax.plot(sc_d, to_wealth(sc_r), label='V2.1.2 Score (8-dim)', color='orange', lw=2.5, ls='--')
if bs_r is not None: ax.plot(bs_d, to_wealth(bs_r), label='V1.0 Baseline (8-dim)', color='red', lw=2.0, alpha=0.7)
ax.set_title("Strategy Cumulative Excess Return (%)", weight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 2. Alpha Delta Index (Triple vs Others)
ax = axarr[0, 1]
if tr_r is not None and sc_r is not None:
    delta_sc = (np.cumprod(1 + np.nan_to_num(tr_r)) - np.cumprod(1 + np.nan_to_num(sc_r))) * 100
    ax.plot(tr_d, delta_sc, color='teal', lw=2, label='vs Score-Only')
if tr_r is not None and bs_r is not None:
    delta_bs = (np.cumprod(1 + np.nan_to_num(tr_r)) - np.cumprod(1 + np.nan_to_num(bs_r))) * 100
    ax.plot(tr_d, delta_bs, color='purple', lw=2, label='vs Baseline (Diffusion Lift)')
    ax.fill_between(tr_d, 0, delta_bs, color='purple', alpha=0.05)

ax.set_title("Alpha Advantage: Triple vs Competitors (bps)", weight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 3. Rolling Sharpe/IR Analysis (6m Rolling)
ax = axarr[1, 0]
def roll_ir(r):
    if r is None: return None
    w = 126 # 6 months
    ser = pd.Series(r)
    return (ser.rolling(w).mean() / (ser.rolling(w).std() + 1e-6)) * np.sqrt(252)

if tr_r is not None: ax.plot(tr_d, roll_ir(tr_r), label='Triple IR', color='blue', alpha=0.6)
if sc_r is not None: ax.plot(sc_d, roll_ir(sc_r), label='Score IR', color='orange', alpha=0.6)
if bs_r is not None: ax.plot(bs_d, roll_ir(bs_r), label='Baseline IR', color='red', alpha=0.6)
ax.set_title("Rolling Information Ratio (6m Window)", weight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. Error Dispersion
ax = axarr[1, 1]
if tr_p is not None: sns.kdeplot((tr_p - tr_t).ravel(), ax=ax, label='Triple Error', color='blue', fill=True, alpha=0.1)
if sc_p is not None: sns.kdeplot((sc_p - sc_t).ravel(), ax=ax, label='Score Error', color='orange', fill=True, alpha=0.1)
if bs_p is not None: sns.kdeplot((bs_p - bs_t).ravel(), ax=ax, label='Baseline Error', color='red', fill=True, alpha=0.1)
ax.set_title("Predictive Error Distribution", weight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/v2_3way_architecture_comparison.png', dpi=300)
print("SUCCESS: 3-Way Comparison Dashboard written to plots/v2_3way_architecture_comparison.png")
