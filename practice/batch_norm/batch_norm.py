import torch 
from torch import nn 

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()

        # Parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running mean and variance for inference
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        # Hyperparameters
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        out = self.gamma * x_hat + self.beta
        return out