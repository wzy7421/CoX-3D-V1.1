import numpy as np
from tqdm import tqdm

def bake_vertex_colors_to_texture(
    verts: np.ndarray,
    faces: np.ndarray,
    vcolors: np.ndarray,
    uvs: np.ndarray,
    tex_res: int = 1024,
):
    """
    Minimal UV raster bake: per-vertex RGB -> UV texture PNG.
    Deterministic and fully self-contained (no Blender required).
    """
    tex = np.zeros((tex_res, tex_res, 3), dtype=np.uint8)
    mask = np.zeros((tex_res, tex_res), dtype=np.uint8)

    uv_pix = np.empty_like(uvs)
    uv_pix[:, 0] = uvs[:, 0] * (tex_res - 1)
    uv_pix[:, 1] = (1.0 - uvs[:, 1]) * (tex_res - 1)

    def bbox(p0, p1, p2):
        xs = [p0[0], p1[0], p2[0]]
        ys = [p0[1], p1[1], p2[1]]
        x0 = max(int(np.floor(min(xs))), 0)
        x1 = min(int(np.ceil(max(xs))), tex_res - 1)
        y0 = max(int(np.floor(min(ys))), 0)
        y1 = min(int(np.ceil(max(ys))), tex_res - 1)
        return x0, x1, y0, y1

    def bary(p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = v0.dot(v0); d01 = v0.dot(v1); d11 = v1.dot(v1)
        d20 = v2.dot(v0); d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return None
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    for fi in tqdm(range(len(faces)), desc="Baking texture"):
        f = faces[fi]
        a = uv_pix[f[0]].astype(np.float64)
        b = uv_pix[f[1]].astype(np.float64)
        c = uv_pix[f[2]].astype(np.float64)

        ca = vcolors[f[0]].astype(np.float64)
        cb = vcolors[f[1]].astype(np.float64)
        cc = vcolors[f[2]].astype(np.float64)

        x0, x1, y0, y1 = bbox(a, b, c)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                p = np.array([x + 0.5, y + 0.5], dtype=np.float64)
                bc = bary(p, a, b, c)
                if bc is None:
                    continue
                w0, w1, w2 = bc
                if (w0 >= -1e-6) and (w1 >= -1e-6) and (w2 >= -1e-6):
                    col = w0 * ca + w1 * cb + w2 * cc
                    tex[y, x] = np.clip(col, 0, 255).astype(np.uint8)
                    mask[y, x] = 255

    # hole fill
    for _ in range(8):
        holes = (mask == 0)
        if not holes.any():
            break
        padded = np.pad(tex, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_m = np.pad(mask, ((1, 1), (1, 1)), mode="edge")
        neigh = (
            padded_m[0:-2,0:-2] + padded_m[0:-2,1:-1] + padded_m[0:-2,2:] +
            padded_m[1:-1,0:-2] + padded_m[1:-1,1:-1] + padded_m[1:-1,2:] +
            padded_m[2:,0:-2] + padded_m[2:,1:-1] + padded_m[2:,2:]
        )
        has = (neigh > 0) & holes
        if not has.any():
            break
        sum_col = (
            padded[0:-2,0:-2] + padded[0:-2,1:-1] + padded[0:-2,2:] +
            padded[1:-1,0:-2] + padded[1:-1,1:-1] + padded[1:-1,2:] +
            padded[2:,0:-2] + padded[2:,1:-1] + padded[2:,2:]
        )
        tex[has] = (sum_col[has] / 9.0).astype(np.uint8)
        mask[has] = 255

    return tex
