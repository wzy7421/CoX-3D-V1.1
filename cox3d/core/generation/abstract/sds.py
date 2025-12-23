import torch

class SDSLoss(torch.nn.Module):
    """
    Paper-level SDS loss skeleton (Eq. 8).
    Kept as an abstract module for SI/GitHub completeness.
    """
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion = diffusion_model

    def forward(self, x_theta, y, t):
        eps = torch.randn_like(x_theta)
        eps_pred = self.diffusion(x_theta, y, t)
        return torch.mean((eps_pred - eps) ** 2)
