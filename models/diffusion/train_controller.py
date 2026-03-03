import loss_func
import forward
from forward import noise_schedule
import torch
from reversal import *
from diffusion_architecture import DiffusionReturnPrediction, Diffusion, ReturnPrediction

import argsparse
import sys

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

def train_controller_diffusion(model, train_loader, optimizer, loss_func, epochs, t_max, T): 
    model.train()
    # this is pseudocode, I need to understand how each parameter is used and how to access member functions
    for epoch in range(epochs):
        epoch_loss = 0
        # this loop assumes a lot of things about the architecture of the params by the team
        for batch in train_loader:
            optimizer.zero_grad() # how does this interact w the rest of the process?

            t = torch.randint(0, t_max, (batch.shape[0],))

            batch_noisy = forward(batch, T, t)
            batch_scores = model(batch_noisy, t)
            betas = noise_schedule(t, T)

            batch_loss = loss_func(batch_scores, betas, batch_noisy)
            batch_loss.backward()

            optimizer.step() # the link between the optimizer and the model has to be built by the rest of the team previously

            epoch_loss += batch_loss.item()

        if epoch % 100 == 0:
            print(f'{epoch}: {epoch_loss}')


def train_controller_return(model, train_loader, epochs, optimizer, loss_func_MSE):
    # TODO: train return regression with similar logic
    # Compare predicted excess return to actual excess return MSE
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            features = batch[0]
            targets = batch[1]

            results = model(features)

            batch_loss = loss_func_MSE(results, targets)
            batch_loss.backward()

            optimizer.step() # the link between the optimizer and the model has to be built by the rest of the team previously

            epoch_loss += batch_loss.item()
            
        if epoch % 100 == 0:
            print(f'{epoch}: {epoch_loss}')


def train_controller_main(model, train_loader, epochs, optimizer, loss_func_diffusion, loss_func_MSE, t_max, T):

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        for batch_x, batch_y in train_loader:
            features = batch_x
            targets = batch_y

            # forward diffusion
            t = torch.randint(0, t_max, (features.shape[0],))
            features_noisy = forward(features, T, t)

            # score prediction
            scores = model.diffusion(features_noisy, t)

            # loss for diffusion
            betas = noise_schedule(t, T)
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
            sampled_features_clean = reversal(model.diffusion, sampled_features, t_max, T)
            return_preds = model.return_regression(sampled_features_clean)
            loss_return = loss_func_MSE(return_preds, sampled_targets)

    
            total_loss = loss_diffusion + loss_return

            total_loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'{epoch}: {total_loss.item()}')


if __name__ == '__main__':
    parser = argsparse.ArgumentParser(description='Hypertuning diffusion model for commodity price prediction')

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

