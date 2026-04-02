# DS-TGNN Research Note: Signal Representation Strategy 📈

## 1. The Strategy Trade-Off

| Feature | **Triple-Signal** (Current V2.1.1) | **Score-Only** (Proposed V2.2) |
| :--- | :--- | :--- |
| **Logic** | Raw + Denoised + Score | Raw + Score |
| **Feature Space** | 12-Dimensional | 8-Dimensional |
| **Philosophy** | "Safety First": Keep the clean data as a fallback. | "Alpha First": The score is the only truth of the signal. |
| **Risk** | High dimensionality can lead to overfitting on noise. | High sensitivity to score calibration. |

---

## 2. Why "Score-Only" is Theoretically Superior

In traditional commodity prediction, **Denoised Data** essentially tells the model what "normal" weather looks like. However, in trading, we don't care about normal weather; we care about **anomalies**.

### A. Information Density
The **Manifold Score** ($z_{pred} \cdot \sigma_{low}$) is the mathematical derivative of the feature space. It represents the "Level of Surprise" in the data. By providing *only* Raw and Score, you are telling the model:
1.  **Raw**: "This is what happened."
2.  **Score**: "This is exactly how much you should care about each feature." 
Adding **Denoised Weather** in the middle can actually "dilute" the signal by adding features that are highly correlated with the Raw data.

### B. Overfitting Mitigation (Occam's Razor)
A 12-dimensional feature space ($3 \times 4$ channels) per hub is significantly harder to regularize than an 8-dimensional space. If the **Score** correctly represents which features are signals, the **Denoised** data becomes redundant. Redundancy in deep learning often leads to "vanishing gradients" where the model ignores the signal (Score) in favor of the easy-to-learn correlation (Denoised).

---

## 3. Recommendation: The "Staged" Approach

### Why we are running Triple-Signal NOW (V2.1.1):
We are currently in **Validation Mode**. We need to prove that the `Score` is actually contributing more than the `Denoised` data. By having both in the same model, we can visualize the **Signal Importance** across the encoder weights.

### Why we should move to Score-Only (V2.2):
Once our current **5-Fold Walk-Forward** confirms that `Uncertainty_Corr` is high (> 0.05), we should strip away the "Cleaned" features. This will:
1.  **Speed up inference** by 15-20%.
2.  **Force the model** to rely purely on the Manifold Score for its predictive edge.
3.  **Sharpen the Alpha Dashboard**: Eliminating redundant data usually results in more stable Information Ratios (IR).

---

## 4. Final Verdict

**Current V2.1.1 (Triple-Signal)** is better for **Research & Verification**. 
**V2.2 (Score-Only)** is better for **Production & Deployment**.

If the current run shows a significant IR improvement over the LSTM baseline, we have effectively "proven" the Score's value. In the next session, we can prune the Denoised features to create the leaner, more aggressive V2.2 Alpha model.
