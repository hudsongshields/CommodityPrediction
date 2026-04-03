import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_res(path):
    d = np.load(path)
    return d['strat_returns'], d['dates'], d['preds'], d['targets']

# Load Datasets
tr_r, tr_d, tr_p, tr_t = load_res('results/v2_DS-TGNN_V2.1.1_Triple_results.npz')
sc_r, sc_d, sc_p, sc_t = load_res('results/v2_DS-TGNN_V2.1.2_Score_results.npz')

# Fix dates
tr_d = pd.to_datetime(tr_d)
sc_d = pd.to_datetime(sc_d)

def to_wealth(r): return (np.cumprod(1 + np.nan_to_num(r)) - 1.0) * 100

fig, axarr = plt.subplots(2, 2, figsize=(22, 14))
sns.set_style("darkgrid")

# 1. Cumulative Performance
ax = axarr[0, 0]
ax.plot(tr_d, to_wealth(tr_r), label='V2.1.1 Triple (12-dim)', color='blue', lw=2.5)
ax.plot(sc_d, to_wealth(sc_r), label='V2.1.2 Score (8-dim)', color='orange', lw=2.5, ls='--')
ax.set_title("Strategy Cumulative Excess Return (%)", weight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 2. Alpha Delta Index (Triple - Score)
ax = axarr[0, 1]
delta = (np.cumprod(1 + np.nan_to_num(tr_r)) - np.cumprod(1 + np.nan_to_num(sc_r))) * 100
ax.plot(tr_d, delta, color='teal', lw=2.5)
ax.fill_between(tr_d, 0, delta, color='teal', alpha=0.1)
ax.set_title("Alpha Advantage: Triple vs Score (bps)", weight='bold')
ax.grid(True, alpha=0.3)

# 3. Rolling Sharpe/IR Analysis (6m Rolling)
ax = axarr[1, 0]
def roll_ir(r):
    w = 126 # 6 months
    ser = pd.Series(r)
    return (ser.rolling(w).mean() / (ser.rolling(w).std() + 1e-6)) * np.sqrt(252)

ax.plot(tr_d, roll_ir(tr_r), label='Triple IR (Roll)', color='blue', alpha=0.6)
ax.plot(sc_d, roll_ir(sc_r), label='Score IR (Roll)', color='orange', alpha=0.6)
ax.set_title("Rolling Information Ratio (6m Window)", weight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. Error Dispersion
ax = axarr[1, 1]
tr_err = (tr_p - tr_t).ravel()
sc_err = (sc_p - sc_t).ravel()
sns.kdeplot(tr_err, ax=ax, label='Triple Error', color='blue', fill=True, alpha=0.2)
sns.kdeplot(sc_err, ax=ax, label='Score Error', color='orange', fill=True, alpha=0.2)
ax.set_title("Predictive Error Distribution", weight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/v2_architecture_comparison.png', dpi=300)
print("SUCCESS: Comparison Dashboard written to plots/v2_architecture_comparison.png")
