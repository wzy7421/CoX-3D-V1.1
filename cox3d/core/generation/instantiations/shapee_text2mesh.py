import numpy as np
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh


class ShapeEText2Mesh:
    """
    Minimal runnable instantiation of CoX-3D generation module.
    Produces a triangle mesh with vertex colors, then downstream stages
    materialize it to OBJ+MTL+PNG.

    Runtime notes:
    - First run may download Shap-E weights.
    - GPU recommended.
    """

    def __init__(self, device="cuda", steps=64, guidance_scale=15.0):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)

        self.transmitter = load_model("transmitter", device=self.device)
        self.text_model = load_model("text300M", device=self.device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    @torch.no_grad()
    def __call__(self, prompt: str):
        latents = sample_latents(
            batch_size=1,
            model=self.text_model,
            diffusion=self.diffusion,
            guidance_scale=self.guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=(self.device == "cuda"),
            device=self.device,
            num_steps=self.steps,
        )

        tri = decode_latent_mesh(self.transmitter, latents[0]).tri_mesh()

        verts = tri.verts.detach().cpu().numpy().astype(np.float32)
        faces = tri.faces.detach().cpu().numpy().astype(np.int32)

        if hasattr(tri, "vertex_channels") and "RGB" in tri.vertex_channels:
            rgb = tri.vertex_channels["RGB"].detach().cpu().numpy()
            vcolors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            vcolors = np.ones((verts.shape[0], 3), dtype=np.uint8) * 200

        return {"verts": verts, "faces": faces, "vcolors": vcolors}
