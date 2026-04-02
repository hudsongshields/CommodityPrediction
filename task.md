# DS-TGNN V2.1.1 Research Task List 🚀

- [x] **Phase 8: Multi-Task Architecture Restoration** (V2.1.1)
    - [x] Restore `ScoreDiffusionLoss` in `loss_func.py` (V2.1.1)
    - [x] Implement 12-dim feature encoder in `ds_tgnn.py` (Raw + Denoised + Score)
- [x] **Phase 9: Stride Stability & Tensor Debugging** (V2.1.1)
    - [x] Trace and replace all `.view()` with `.reshape()` across `models/` (V2.1.1)
    - [x] Validate Triple-Signal (12-dim) gradient flow in isolation test (V2.1.1)
- [/] **Phase 10: Full Scale Research Execution** (V2.1.1)
    - [/] Run 5-fold Walk-Forward @ 25 epochs per fold (Benchmark Run)
    - [ ] Generate comparative Alpha Dashboard (`v2_alpha_dashboard.png`)
    - [ ] Calculate 1,000-walk Monte Carlo confidence intervals for PnL
    - [ ] Final Presentation: Documenting DS-TGNN V2.1.1 performance edge
