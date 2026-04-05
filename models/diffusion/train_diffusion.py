import argparse

import loss_func
import forward
from forward import noise_schedule
from forward import forward_process
import torch
from reversal import *
from diffusion_architecture import DiffusionReturnPrediction, Diffusion, ReturnPrediction

# model represent score of the distribution in this parameter
# train_loader allows us to loop over segments of the dataset
# optimizer takes in hyperparameters to increase the capture and learning rate. How and how fast?
# loss function represents diff between predicted and actual. 
# gradient of loss is sum of derivatives wrt each weight (nueron) in the model
# epochs will be defined as an iterable previously

'''
First thing is to go forward in each epoch, then take the score w the noisy data
Then find the loss w the score and noisy data using loss_func
Then go loss.backward() to calculate gradient
optim.step()
'''

def train_controller_diffusion(model, train_loader, optimizer, loss_func, epochs, t_max, T, every_n_epochs=10, noise_schedule=noise_schedule, device='cpu'): 
    model.train()

    delta_t = 1.0 / T
    t_all = torch.arange(t_max, dtype=torch.float32)
    beta_all = noise_schedule(t_all, T)
    beta_cumsum = torch.cumsum(beta_all, dim=0) * delta_t
    beta_cumsum = beta_cumsum.to(device)

    print("beta_integral final:", beta_cumsum[-1].item())
    print("mean coeff final:", torch.exp(-0.5 * beta_cumsum[-1]).item())
    print("var final:", (1 - torch.exp(-beta_cumsum[-1])).item())

    for epoch in range(epochs):
        epoch_loss = 0
        for batch, _ in train_loader:
            batch = batch.view(batch.size(0), -1).to(device)
            optimizer.zero_grad()

            # Sample t weighted by time step to encourage learning on earlier steps
            weights = 1.0 / torch.arange(1, t_max, device=device, dtype=torch.float32)
            weights = weights / weights.sum()
            t = torch.multinomial(weights, batch.size(0), replacement=True)

            batch_betas = beta_cumsum[t].view(-1, 1).to(device)

            batch_noisy = forward_process(batch, t, T, noise_schedule, batch_betas, device=device).view(batch.size(0), -1)
            batch_scores = model(batch_noisy, t.to(batch_noisy.dtype) / T).view(batch_noisy.size(0), -1)

            batch_loss = loss_func(batch_scores, batch_noisy, batch, batch_betas)
            batch_loss.backward()

            optimizer.step()

            epoch_loss += batch_loss.item()

        if epoch % every_n_epochs == 0:
            print(f'{epoch}: {epoch_loss}')



def train_controller_main(model, train_loader, epochs, optimizer, loss_func_diffusion, loss_func_MSE, t_max, T, every_n_epochs=10, noise_schedule=noise_schedule, device='cpu'):

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        for batch_x, batch_y in train_loader:
            features = batch_x.to(device)
            targets = batch_y.to(device)

            # forward diffusion
            t = torch.randint(0, t_max, (features.shape[0],), dtype=features.dtype).to(device)
            features_noisy = forward(features, T, t, device=device)

            # score prediction
            scores = model.diffusion(features_noisy, t)

            # loss for diffusion
            betas = noise_schedule(t, T).to(device)
            loss_diffusion = loss_func_diffusion(scores, betas, features_noisy)
            

            """ 
            Compute Return Prediction Loss

            Use stochastic sampling withing batch for: 
                1. computational efficiency
                2. encourage generalization
            """
            # stochastic sampling from batch
            sampled_features = features_noisy[torch.randperm(features_noisy.size(0))]
            sampled_targets = targets[torch.randperm(targets.size(0))]

            # generate return preds from clean version of sampled features
            sampled_features_clean = reversal(model.diffusion, sampled_features, t_max, T, device=device)
            return_preds = model.return_regression(sampled_features_clean)
            loss_return = loss_func_MSE(return_preds, sampled_targets)


            total_loss = loss_diffusion + loss_return

            total_loss.backward()
            optimizer.step()

        if epoch % every_n_epochs == 0:
            print(f'{epoch}: {total_loss.item()}')


if __name__ == '__main__':
    import argsparse
    import sys
    
    parser = argparse.ArgumentParser(description='Hypertuning diffusion model for commodity price prediction')

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()

    model = DiffusionReturnPrediction(input_dim=..., hidden_dim=..., output_dim=..., return_regression=...)
    loss_func_MSE = torch.nn.MSELoss()
    loss_func_diffusion = loss_func.ScoreDiffusionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=...)

    train_controller_main(
        model, 
        train_loader=..., 
        epochs=args.epochs, 
        optimizer=optimizer, 
        loss_func_diffusion=loss_func_diffusion, 
        loss_func_MSE=loss_func_MSE, 
        t_max=..., T=...
    )

