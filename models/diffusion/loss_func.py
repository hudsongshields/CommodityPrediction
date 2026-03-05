import torch

__all__ = ['ScoreDiffusionLoss']


def ScoreDiffusionLoss(score, x_t, x_0, var_t):
    loss = torch.mean((score - (x_t - x_0) / var_t) ** 2)
    return loss


if __name__ == '__main__':
    score = torch.tensor([0.5, 0.3, 0.2])
    x_t = torch.tensor([1.0, 0.8, 0.6])
    x_0 = torch.tensor([0.0, 0.0, 0.0])
    var_t = torch.tensor([0.1, 0.1, 0.1])
    loss = ScoreDiffusionLoss(score, x_t, x_0, var_t)
    print(loss)