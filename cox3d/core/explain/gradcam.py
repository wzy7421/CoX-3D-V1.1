import torch

class GradCAM:
    """
    Minimal Grad-CAM helper (placeholder).
    """
    def compute(self, grads: torch.Tensor, activations: torch.Tensor):
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(0)
        return torch.relu(cam)
