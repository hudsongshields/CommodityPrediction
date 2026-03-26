
def train_controller_return(model, train_loader, epochs, optimizer, loss_func_MSE, every_n_epochs=10, device='cpu'):
    # Compare predicted excess return to actual excess return MSE

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            features = batch[0].to(device)
            targets = batch[1].to(device)

            results = model(features)

            batch_loss = loss_func_MSE(results, targets)
            batch_loss.backward()

            optimizer.step()

            epoch_loss += batch_loss.item()
            
        if epoch % every_n_epochs == 0:
            print(f'{epoch}: {epoch_loss}')