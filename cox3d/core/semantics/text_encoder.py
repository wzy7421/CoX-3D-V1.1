import torch

class FrozenSentenceEncoder(torch.nn.Module):
    """
    Frozen text encoder placeholder.

    For full fidelity, replace with SentenceTransformer.
    Here we use a deterministic hashing-based embedding to keep the repo runnable
    without extra heavy dependencies.
    """

    def __init__(self, device="cuda", dim=768):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dim = dim

    @torch.no_grad()
    def forward(self, texts):
        embs = []
        for t in texts:
            v = torch.zeros(self.dim, device=self.device)
            for ch in t:
                idx = (ord(ch) * 1315423911) % self.dim
                v[idx] += 1.0
            v = v / (v.norm() + 1e-6)
            embs.append(v)
        return torch.stack(embs, dim=0)
