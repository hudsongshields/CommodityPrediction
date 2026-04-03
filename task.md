# DS-TGNN V2.1.1 Research Task List 🚀

- [x] **Phase 8: Multi-Task Architecture Restoration** (V2.1.1)
    - [x] Restore `ScoreDiffusionLoss` in `loss_func.py` (V2.1.1)
    - [x] Implement 12-dim feature encoder in `ds_tgnn.py` (Raw + Denoised + Score)
- [x] **Phase 9: Stride Stability & Tensor Debugging** (V2.1.1)
    - [x] Trace and replace all `.view()` with `.reshape()` across `models/` (V2.1.1)
    - [x] Validate Triple-Signal (12-dim) gradient flow in isolation test (V2.1.1)
    - [x] Full Scale Research Run V2.1.1: Partial (3/5) check with IR +0.029 (V2.1.1)
- [x] **Phase 11: Score-Only & Theory Alignment** (V2.1.2)
    - [x] Refactor `ScoreDiffusionLoss` to normalized score-matching objective
    - [x] Prune "Denoised Weather" from `ds_tgnn.py` (8-dim Docking)
    - [x] Align SDE `reversal.py` drift with new score definition
    - [x] Reset Walk-Forward Benchmark (5-fold, 50 epochs)

