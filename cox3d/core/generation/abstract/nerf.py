import torch

class NeRF(torch.nn.Module):
    """
    Paper-level NeRF skeleton.
    """
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4)  # sigma + rgb
        )

    def forward(self, x):
        return self.mlp(x)
