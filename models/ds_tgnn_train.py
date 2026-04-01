import argparse
import sys
import torch
import torch.nn as nn
from diffusion.loss_func import ScoreDiffusionLoss
from diffusion.diffusion_architecture import Diffusion
from ds_tgnn import DiffusionReturnPrediction
from dataset import get_dataloaders

# Define commodity IDs and adjacency
def build_supply_chain_graph():
    # 0: Corn, 1: Soybeans, 2: Wheat, 3: Cattle, 4: Hogs, 5: Ethanol, 6: Natural Gas, 7: Cotton
    edges = [
        (0, 3), (0, 4), (0, 5),  # Corn -> Cattle, Hogs, Ethanol
        (1, 3), (1, 4),          # Soybeans -> Cattle, Hogs
        (2, 0),                  # Wheat -> Corn
        (5, 6),                  # Ethanol -> Natural Gas
    ]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def train_controller_ds_tgnn(model, train_loader, epochs, opt_diff, opt_pred, score_loss_fn, device='cpu', every_n_epochs=10):
    model.train()
    model.to(device)

    # Static graph edge index
    baseline_edge_index = build_supply_chain_graph().to(device)

    for epoch in range(epochs):
        epoch_diff_loss = 0.0
        epoch_pred_loss = 0.0

        for batch_x, batch_targets in train_loader:
            x_clean = batch_x.to(device)   # [B, N, T, F]
            target_excess_returns = batch_targets.to(device)
            B, N, Time, F = x_clean.shape

            # To use the existing MLP Diffusion, we must flatten the spatial/temporal dimension for the denoiser
            # so x_clean becomes [B, N*T*F]
            x_clean_flat = x_clean.view(B, -1)

            # ========================================
            # Objective 1: Score Matching (Denoiser)
            # ========================================
            opt_diff.zero_grad()
            
            # Train the denoiser to reconstruct clean weather from random noise steps
            # score_loss_fn will handle sampling t, adding noise, and predicting.
            loss_diff = score_loss_fn(model.denoiser, x_clean_flat)
            loss_diff.backward()
            opt_diff.step()
            
            # ========================================
            # Objective 2: Supervised Prediction
            # ========================================
            opt_pred.zero_grad()
            
            with torch.no_grad():
                # Inject small deterministic noise (t ~ 0.1)
                t_low = torch.full((B,), 0.1, device=device)
                sigma_low = score_loss_fn.sigma_min * (score_loss_fn.sigma_max / score_loss_fn.sigma_min) ** t_low
                sigma_low_view = sigma_low.view(B, 1) # for broadcasting with [B, Flat_Dim]
                x_noisy_flat = x_clean_flat + torch.randn_like(x_clean_flat) * sigma_low_view
                
            # One step expectation of x_0:
            z_pred_flat = model.denoiser(x_noisy_flat, sigma_low.view(-1, 1))

            # Since the denoiser predicted the added noise Z:
            denoised_repr_flat = x_noisy_flat - z_pred_flat * sigma_low_view 
            
            # Reshape back to [B, N, T, F]
            denoised_repr = denoised_repr_flat.view(B, N, Time, F)

            # Pass through the predictive pipeline: LSTM -> GNN -> Shared MLP
            predicted_returns = model(denoised_repr, baseline_edge_index) # [B, N]
            
            # Compute MSE on the 30-day target vs predictions
            # Magnitude-Weighted MSE (Exp 3)
            # weights scale the loss penalty proportional to the rarity/extremity of the return
            raw_loss = (predicted_returns - target_excess_returns) ** 2
            
            # Scale factor 10.0: If target is 0.05 (+5%), weight is 1.5x. 
            # If target is 0.20 (+20%), weight is 3.0x.
            weights = 1.0 + torch.abs(target_excess_returns) * 10.0
            
            loss_pred = torch.mean(raw_loss * weights)
            loss_pred.backward()
            opt_pred.step()
            
            epoch_diff_loss += loss_diff.item()
            epoch_pred_loss += loss_pred.item()

        if epoch % every_n_epochs == 0:
            print(f'Epoch {epoch} | Diff Loss: {epoch_diff_loss:.4f} | Pred Loss: {epoch_pred_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train end-to-end DS-TGNN pipeline')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_diff', type=float, default=1e-3)
    parser.add_argument('--lr_pred', type=float, default=1e-3)

    args = parser.parse_args()

    # Hyperparams 
    N = 8
    T = 180
    F = 5
    input_dim = N * T * F  # Flattened dim for MLP denoiser
    
    lstm_hidden = 32
    gnn_hidden = 32
    mlp_hidden = 64
    conv_hidden = 32
    t_hidden_dim = 16
    output_dim = input_dim # Denoiser outputs same dim as input
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Denoiser setup
    denoiser = Diffusion(
        input_dim=input_dim, 
        mlp_hidden=mlp_hidden, 
        conv_hidden=conv_hidden, 
        t_hidden_dim=t_hidden_dim, 
        output_dim=output_dim, 
        use_conv=False
    )
    
    # Composite Model setup
    model = DiffusionReturnPrediction(
        denoiser=denoiser,
        input_dim=F,         # The LSTM takes F features per step
        lstm_hidden=lstm_hidden,
        gnn_hidden=gnn_hidden
    ).to(device)

    # Losses & Optimizers
    score_loss_fn = ScoreDiffusionLoss(sigma_min=1e-4, sigma_max=1.0).to(device)
    
    opt_diff = torch.optim.Adam(model.denoiser.parameters(), lr=args.lr_diff)
    opt_pred = torch.optim.Adam(
        list(model.lstm.parameters()) + list(model.gnn.parameters()) + list(model.shared_mlp.parameters()),
        lr=args.lr_pred
    )

    # Dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=16, N=N, T=T, F=F)
    
    print("DS-TGNN instantiated and ready for training!")
    train_controller_ds_tgnn(model, train_loader, args.epochs, opt_diff, opt_pred, score_loss_fn, device=device, every_n_epochs=1)
