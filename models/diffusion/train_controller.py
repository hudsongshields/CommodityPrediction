import loss_func
import forward
from forward import noise_schedule
import torch

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

def train_controller(model, train_loader, optimizer, loss_func, epochs, t_max, T): 
    model.train()
    # TODO: implement training loop for score-based diffusion model
    # this is pseudocode, I need to understand how each parameter is used and how to access member functions
    for epoch in range(epochs):
        epoch_loss = 0
        # this loop assumes a lot of things about the architecture of the params by the team
        for batch in train_loader:
            optimizer.zero_grad() # how does this interact w the rest of the process?

            t = torch.randint(0, t_max, (batch.shape[0],))

            batch_noisy = forward(batch, T, t_max)
            batch_scores = model(batch_noisy, t)
            betas = noise_schedule(t, T)

            batch_loss = loss_func(batch_scores, betas)
            batch_loss.backward()


            optimizer.step() # the link between the optimizer and the model has to be built by the rest of the team previously

            epoch_loss += batch_loss

        if epoch % 100 == 0:
            print(f'{epoch}: {epoch_loss}')

if __name__ == 'main':
    # TODO: test vectorized version with tensors 
    pass


