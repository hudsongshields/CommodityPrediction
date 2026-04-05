# DS-TGNN V1.1: Experimental Methodology & Notes

This document provides the formal specification for the DS-TGNN V1.1 experimental pipeline, designed for reproduction in a research context.

## 1. Mathematical Framework

### Forecasting Objective
We predict the cumulative excess return $R_{i, [t+1, t+H]}$ for commodity $i$ over a future horizon $H=30$ days, given a 180-day window of meteorological features $X_{i, [t-W+1, t]}$.

### Architecture Components
1.  **Diffusion Denoiser**: An MLP-based score matching model that estimates the clean meteorological expectation from noisy inputs.
2.  **Temporal Encoder (LSTM)**: Processes the 180-day sequence into a 32-dimensional latent state $h_{i,t}$.
3.  **Spatial Encoder (GNN)**: A Graph Convolutional Network (GCN) that models supply-chain dependencies between commodities using a static adjacency matrix.
4.  **Prediction Head (MLP)**: A shared prediction head using SiLU activations and MC-Dropout for epistemic uncertainty estimation.

---

## 2. Experimental Setup

### Data Splitting & Embargo
To prevent temporal data leakage, we employ a chronological split (60/20/20) with a **30-day embargo gap**. The embargo ensures that no information from the future (overlapping windows) is present in the training set during validation or testing.

| Split | Percentage | Notes |
| :--- | :--- | :--- |
| Training | 60% | Continuous samples with overlapping windows. |
| **Embargo** | -- | **30-day gap** to clear the target horizon. |
| Validation | 20% | Out-of-sample tuning. |
| **Embargo** | -- | **30-day gap**. |
| Testing | 20% | Final out-of-sample benchmark. |

### Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| Window Size ($W$) | 180 days | Historical lookback. |
| Horizon ($H$) | 30 days | Target return accumulation period. |
| Batch Size | 16 | Number of sequence samples per batch. |
| Learning Rate | 1e-3 | Adam optimizer step size. |
| Magnitude Weight | 10.0 | Penalty multiplier for extreme return events. |
| MC-Samples | 50 | Forward passes for uncertainty calculation. |
| LSTM Hidden Dim | 32 | Temporal latent feature size. |
| GNN Output Dim | 32 | Spatial latent feature size. |

---

## 3. Evaluation Metrics

### Prediction Metrics (Regression)
*   **RMSE**: Root Mean Squared Error across all test samples.
*   **Tail RMSE**: RMSE restricted to samples where the realized return magnitude is in the top 10th percentile (90% quantile).
*   **R²**: Coefficient of determination for return predictability.

### Strategy Metrics (Trading)
*   **Information Ratio (IR)**: Calculated on a toy long/short strategy.
    *   **Strategy**: At each timestep, go long on the Top-2 commodities by predicted return and short on the Bottom-2.
    *   **Formula**: $IR = \frac{\mathbb{E}[R_{strat}]}{\sigma(R_{strat})}$.

### Uncertainty Calibration
*   **Uncertainty Correlation**: Pearson correlation between the standard deviation of MC-Dropout predictions and the absolute realized prediction error. High correlation indicates reliable "confidence" signaling.
