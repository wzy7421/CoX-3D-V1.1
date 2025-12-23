import os
from PIL import Image

def export_obj_mtl_png(out_dir, name, verts, faces, uvs, texture):
    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, f"{name}_albedo.png")
    mtl_path = os.path.join(out_dir, f"{name}.mtl")
    obj_path = os.path.join(out_dir, f"{name}.obj")

    Image.fromarray(texture).save(tex_path)

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl material0\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("Ns 10.000\n")
        f.write("d 1.0\n")
        f.write(f"map_Kd {os.path.basename(tex_path)}\n")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        f.write("usemtl material0\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for vt in uvs:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")

        for tri in faces:
            a, b, c = tri
            f.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")

    return obj_path
