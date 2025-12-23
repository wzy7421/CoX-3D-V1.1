import numpy as np
import xatlas

def unwrap_uv_xatlas(verts: np.ndarray, faces: np.ndarray):
    """
    UV unwrapping using xatlas.

    Returns:
      new_verts [V',3], new_faces [F,3], uvs [V',2], vmapping [V'] -> old vertex index
    """
    vmapping, indices, uvs = xatlas.parametrize(verts, faces)
    new_verts = verts[vmapping]
    new_faces = indices.astype(np.int32)
    uvs = uvs.astype(np.float32)
    return new_verts, new_faces, uvs, vmapping
