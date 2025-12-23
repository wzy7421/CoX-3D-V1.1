# CoX-3D-V1.1
CoX-3D

Explainable Generative 3D Design Driven by Social Media Semantics

This repository provides a reference PyTorch implementation of the CoX-3D framework, an explainable, semantic-driven approach for generative 3D design based on large-scale social media data.

CoX-3D is designed to transform noisy, unstructured user-generated content (UGC) into interpretable design semantics, and further map these semantics into controllable 3D geometry and material representations through an explainable generation pipeline.

1. Repository Contents

This repository provides a complete, self-contained implementation of the CoX-3D framework.

All modules required for semantic modeling, cross-modal alignment, 3D generation, post-processing, and asset export are explicitly included to support methodological reproducibility and executable verification.

The repository is organized as follows:

1.1 core/datasets/

UGC dataset construction and preprocessing (Algorithm 1 alignment)

This module implements the data construction pipeline described in Algorithm 1 of the paper.

It includes:

JSONL-based social media data loader (text / image / timestamp)

Text preprocessing, denoising, and semantic normalization

Deterministic and reproducible dataset construction interface

Support for multimodal UGC samples with temporal information

This module formalizes social media expressions as demand cues, rather than direct design requirements.

1.2 core/semantics/

Explainable Large Language Model (ELLM) for semantic abstraction

This module implements the semantic abstraction and validation layer of CoX-3D.

It includes:

BERTopic-based topic discovery (optional, with automatic fallback if unavailable)

Deterministic text embedding encoder (dependency-free fallback)

Topic-level semantic relevance filtering using KL-divergence constraints

Mapping from semantic topics to explainable design semantic factors (SHAP-style features)

The entire module is fully reproducible without external APIs and does not require LLM fine-tuning.

1.3 core/alignment/

Cross-modal text–image semantic alignment (optional)

This module provides optional cross-modal alignment functionality.

It includes:

CLIP-based text–image embedding

Cosine similarity–based semantic alignment

Modular design:

Enabled for full methodological completeness

Optional for minimal runnable generation

This module does not block execution of the minimal runnable pipeline.

1.4 core/generation/

3D generation modules

1.4.1 core/generation/abstract/

Paper-level algorithmic skeletons (for SI / GitHub reference)

This submodule provides abstract implementations aligned with the paper.

It includes:

SDS (Score Distillation Sampling) loss formulation

NeRF network abstraction

Variable definitions consistent with equations in the paper

These components are intended for methodological clarity and SI documentation, not direct execution.

1.4.2 core/generation/instantiations/shapee_text2mesh.py

Runnable instantiation of the CoX-3D generation module

This file provides a concrete, executable instantiation of the CoX-3D generation stage.

Features:

Text → 3D triangle mesh generation using Shap-E

Explicit mesh geometry with vertex colors

Serves as a concrete execution path of the abstract CoX-3D framework

This is the core runnable path that produces real 3D assets.

1.5 core/postprocess/

Mesh post-processing and asset materialization

This module converts generated meshes into standard 3D assets.

It includes:

UV unwrapping using xatlas

Vertex-color to UV-texture baking (PNG)

Export to standard OBJ + MTL + PNG format

All exported assets are directly importable into Blender without additional tools.

1.6 core/engine/pipeline.py

End-to-end CoX-3D pipeline (Algorithm 2 alignment)

This file orchestrates the entire framework and corresponds to Algorithm 2 in the paper.

It integrates:

Semantic modeling (ELLM)

Optional cross-modal alignment

3D generation instantiation

Post-processing and asset export

This file represents the computational backbone of the CoX-3D framework.

1.7 configs/default.yaml

Centralized hyperparameter configuration

This file contains all key hyperparameters, including:

Semantic modeling parameters

3D generation parameters

Post-processing and export settings

It ensures consistent and reproducible experiments across runs.

1.8 data/examples.jsonl

Example multimodal social media data

This file demonstrates the expected JSONL format, including:

text

image_path (optional)

timestamp

It can be directly replaced with real or simulated social media data.

1.9 README.md

Nature Communications–style reproducibility documentation

This document provides:

Methodological positioning

Reproducibility scope

Execution instructions

Clear separation between:

Abstract framework

Concrete instantiation

2. Summary

✔ All core modules required by the CoX-3D framework are included

✔ Both paper-level abstractions and runnable implementations are provided

✔ The repository supports:

Algorithmic inspection

Executable verification

Blender-level asset validation

3. Citation

If you use this code, please cite the associated paper:

@article{cox3d2025,
  title={Explainable generative 3D design driven by social media semantics},
  author={...},
  journal={Nature Communications},
  year={2025}
}

4. License

This project is released for research and academic use only.
Please refer to individual third-party libraries for their respective licenses.

5. Contact

For questions regarding the methodology or implementation, please contact the corresponding author listed in the paper.
