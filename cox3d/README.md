# CoX-3D — Explainable Generative 3D Design Driven by Social Media Semantics

This repository packages a **GitHub-ready** reference implementation of the **CoX-3D** framework (Figure 7 / Algorithm 1–2 alignment), plus a **minimal runnable instantiation** that can generate **Blender-ready textured 3D assets**.

## What you can run end-to-end (text → textured OBJ)
A minimal runnable instantiation is provided to **materialize** the semantic-to-3D pipeline into explicit assets:

- Text prompt (from ELLM abstraction)
- Text-to-3D mesh generation (**Shap-E instantiation**)
- UV unwrap (**xatlas**)
- Vertex-color baking → UV texture (PNG)
- Export `OBJ + MTL + PNG` (importable in Blender)

Outputs are written to `outputs/`:
- `cox3d_asset_0.obj`
- `cox3d_asset_0.mtl`
- `cox3d_asset_0_albedo.png`

## Methodological positioning (NC/SI friendly)
> This repo is designed for **methodological reproducibility** (module boundaries, variables, losses, algorithmic flow).  
> The runnable generator is provided as a **concrete instantiation** of the abstract generation module, enabling asset-level verification.

## Repository layout
```text
cox3d/
├── main.py
├── configs/default.yaml
├── data/examples.jsonl
├── core/
│   ├── datasets/        # CoX-3D dataset structuring
│   ├── semantics/       # ELLM (BERTopic-style + design relevance filtering)
│   ├── alignment/       # CLIP alignment (optional for runnable path)
│   ├── generation/
│   │   ├── abstract/    # SDS/NeRF skeletons (paper-level abstraction)
│   │   └── instantiations/shapee_text2mesh.py  # runnable instantiation
│   ├── postprocess/     # UV unwrap + baking + OBJ export
│   ├── explain/         # SHAP / Grad-CAM placeholders
│   └── engine/pipeline.py
└── outputs/
```

## Install
1) Install PyTorch for your CUDA first (recommended), then:
```bash
pip install -r requirements.txt
```

> Notes:
> - `bertopic` is optional. If not installed, ELLM will fall back to deterministic KMeans clustering.
> - The minimal runnable generator uses `shap-e` (OpenAI) as an instantiation.

## Run
```bash
python main.py --config configs/default.yaml
```

## Data format (JSONL)
Each line is a JSON object containing:
- `text` (required)
- `timestamp` (required, ISO 8601)
- `image_path` (optional; can be empty for minimal demo)

See `data/examples.jsonl`.

## Blender import
Blender → `File → Import → Wavefront (.obj)` → select the exported `.obj`.

## Citation
If you use this code, please cite the associated paper:
```bibtex
@article{cox3d2025,
  title={Explainable generative 3D design driven by social media semantics},
  author={...},
  journal={Nature Communications},
  year={2025}
}
```
