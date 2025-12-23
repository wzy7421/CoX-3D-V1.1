import torch
from PIL import Image

try:
    import clip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False


class CLIPAligner:
    """
    Optional CLIP alignment module (kept for methodological completeness).
    The minimal runnable instantiation does not depend on this module.
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        if not _HAS_CLIP:
            self.model = None
            self.preprocess = None
            return
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, text: str):
        if self.model is None:
            raise RuntimeError("CLIP is not installed.")
        tokens = clip.tokenize([text]).to(self.device)
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_image_path(self, image_path: str):
        if self.model is None:
            raise RuntimeError("CLIP is not installed.")
        img = Image.open(image_path).convert("RGB")
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        return self.model.encode_image(img_t)

    @staticmethod
    def cosine(a, b):
        a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
        return (a * b).sum(dim=-1)
