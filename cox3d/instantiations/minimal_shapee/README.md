# Minimal runnable instantiation (text → textured OBJ)

This folder provides a minimal runnable instantiation of the CoX-3D generation module:

- Input: a semantic prompt (produced by ELLM abstraction)
- Generator: Shap-E text-to-3D mesh (triangle mesh with vertex colors)
- Postprocess: UV unwrap (xatlas) + bake vertex colors → UV texture
- Output: `OBJ + MTL + PNG` (importable in Blender)

Run from repo root:
```bash
python main.py --config configs/default.yaml
```
