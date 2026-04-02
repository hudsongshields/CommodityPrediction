import torch
import sys
import os

# Robust path handling for current project structure
curr_dir = os.getcwd()
sys.path.append(os.path.join(curr_dir, 'models'))
sys.path.append(os.path.join(curr_dir, 'models', 'diffusion'))
sys.path.append(os.path.join(curr_dir, 'models', 'base'))

from ds_tgnn import DiffusionReturnPrediction
from diffusion.diffusion_architecture import Diffusion
from diffusion.loss_func import ScoreDiffusionLoss

def test_v2_1_1_triple_signal():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Triple-Signal V2.1.1 on {device}")
    
    # Spec V2.1: 14 Hubs, 180 Days, 4 Features
    B, N, T, F = 16, 14, 180, 4
    d_in = N * T * F
    
    print(f"Initializing Denoiser with input_dim={d_in}...")
    den = Diffusion(input_dim=d_in, mlp_hidden=[128], conv_hidden=32, t_hidden_dim=16, output_dim=d_in)
    
    print(f"Initializing 3-Channel Model with n_hubs={N}, input_dim={F}...")
    # Internally maps input_dim (4) to combined_in_dim (12)
    model = DiffusionReturnPrediction(den, input_dim=F, lstm_hidden=32, gnn_hidden=32, n_hubs=N).to(device)
    
    # Fake Hub Network
    edges = [[0, 1], [1, 2], [2, 13]] 
    e_idx = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    
    # Fake Dataloader Output (Weather [B, T, N, F])
    x = torch.randn(B, T, N, F).to(device)
    y = torch.randn(B, 8).to(device)
    
    print("--- Testing Model Forward Pass (Triple-Signal) ---")
    try:
        print("  Calling model.forward...")
        preds = model(x, e_idx)
        print(f"  Success! Preds shape: {preds.shape}")
    except Exception as e:
        print(f"  FAILED Model Forward: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Testing Multi-Task Loss Stability ---")
    score_loss_fn = ScoreDiffusionLoss().to(device)
    try:
        print("  Calling score_loss_fn...")
        x_canonical = x.transpose(1, 2)
        diff_loss = score_loss_fn(model.denoiser, x_canonical)
        print(f"  Success! Score Loss: {diff_loss.item()}")
        
        print("  Calculating Total Loss & Backward...")
        reg_loss = torch.mean((preds - y)**2)
        total_loss = reg_loss + 0.1 * diff_loss
        
        total_loss.backward()
        print("  Success! Triple-Signal Gradient flow verified.")
    except Exception as e:
        print(f"FAILED Loss/Backward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_v2_1_1_triple_signal()
