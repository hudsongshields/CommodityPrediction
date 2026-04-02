import torch


def evaluate_model(model, test_loader, loss_func):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x)
            batch_loss = loss_func(predictions, batch_y)
            total_loss += batch_loss.item()
    average_loss = total_loss / len(test_loader)
    return average_loss