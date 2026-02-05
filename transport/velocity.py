import torch

class VelocityScale:
    def __init__(self, n=1.0):
        self.n = float(n)

    def __call__(self, t):
        return torch.log((torch.exp(torch.tensor(self.n)) - 1) * t + 1) / self.n

    def der(self, t):
        return (torch.exp(torch.tensor(self.n)) - 1)/(self.n * ((torch.exp(torch.tensor(self.n)) - 1) * t + 1))