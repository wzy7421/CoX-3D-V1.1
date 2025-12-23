import os

from core.datasets.ugc_dataset import UGCDataset
from core.semantics.bertopic_llm import ELLM
from core.semantics.text_encoder import FrozenSentenceEncoder
from core.semantics.shap_mapper import SHAPMapper
from core.alignment.clip_alignment import CLIPAligner

from core.generation.instantiations.shapee_text2mesh import ShapeEText2Mesh
from core.postprocess.uv_unwrap import unwrap_uv_xatlas
from core.postprocess.bake_texture import bake_vertex_colors_to_texture
from core.postprocess.export_obj import export_obj_mtl_png


class CoX3DPipeline:
    """
    End-to-end pipeline aligned with:
    - Algorithm 1: dataset construction (here uses JSONL as the constructed dataset)
    - Algorithm 2: ELLM -> alignment -> generation -> postprocess -> export
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")

        # Dataset
        self.ds = UGCDataset(
            path_jsonl=cfg["dataset"]["path_jsonl"],
            image_root=cfg["dataset"]["image_root"],
        )

        # Stage I: ELLM semantic modeling
        self.text_encoder = FrozenSentenceEncoder(device=self.device)
        self.ellm = ELLM(
            text_encoder=self.text_encoder,
            topic_k=cfg["semantic"]["topic_k"],
            use_bertopic=cfg["semantic"]["use_bertopic"],
            llm_filter=cfg["semantic"]["llm_filter"],
            lambda_kl=cfg["semantic"]["lambda_kl"],
            device=self.device,
        )
        self.shap_mapper = SHAPMapper(num_factors=cfg["semantic"]["topic_k"])

        # Stage II: CLIP alignment (optional)
        self.clip = CLIPAligner(device=self.device)

        # Stage III-IV: Generation (minimal runnable instantiation)
        impl = cfg["generation"]["impl"]
        if impl == "shapee":
            self.generator = ShapeEText2Mesh(
                device=self.device,
                steps=cfg["generation"]["shapee_steps"],
                guidance_scale=cfg["generation"]["guidance_scale"],
            )
        else:
            raise ValueError(f"Unknown generation impl: {impl}")

        # Postprocess configs
        self.tex_res = cfg["postprocess"]["tex_res"]
        self.out_dir = cfg["postprocess"]["out_dir"]
        self.export_name = cfg["postprocess"]["export_name"]
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self):
        # Minimal demo: generate 1–2 objects from dataset texts
        items = [self.ds[i] for i in range(min(2, len(self.ds)))]

        texts = [it["text"] for it in items]
        times = [it["timestamp"] for it in items]

        # Stage I: ELLM
        sem = self.ellm(texts, times)
        _ = self.shap_mapper.map_topics_to_factors(sem["topic_ids"])  # placeholder

        for idx, it in enumerate(items):
            prompt = sem["prompts"][idx]
            print(f"\n[CoX-3D] Sample {idx} prompt:\n  {prompt}")

            tri_mesh = self.generator(prompt)  # verts, faces, vcolors

            verts, faces, vcolors = tri_mesh["verts"], tri_mesh["faces"], tri_mesh["vcolors"]

            # UV unwrap
            new_verts, new_faces, uvs, vmapping = unwrap_uv_xatlas(verts, faces)
            new_vcolors = vcolors[vmapping]

            # Bake vertex colors -> UV texture
            tex = bake_vertex_colors_to_texture(
                new_verts, new_faces, new_vcolors, uvs, tex_res=self.tex_res
            )

            name = f"{self.export_name}_{idx}"
            obj_path = export_obj_mtl_png(
                out_dir=self.out_dir,
                name=name,
                verts=new_verts,
                faces=new_faces,
                uvs=uvs,
                texture=tex,
            )

            print(f"[Done] Exported Blender-ready asset:\n  {obj_path}")
