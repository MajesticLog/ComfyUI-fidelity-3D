"""
inspect_model.py — Imports a model and prints a report to judge whether it suits the project:
number of objects and polygons, materials (with their textures and resolution), dimensions.

Usage:
    blender -b --factory-startup --python inspect_model.py -- path/model.glb [report.json]
"""

import json
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
path = os.path.abspath(argv[0])
json_out = argv[1] if len(argv) > 1 else None

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

ext = os.path.splitext(path)[1].lower()
if ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=path)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=path)
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=path)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
deps = bpy.context.evaluated_depsgraph_get()
faces = sum(len(o.evaluated_get(deps).data.polygons) for o in meshes)
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
quads = sum(sum(1 for p in o.data.polygons if len(p.vertices) == 4) for o in meshes)

corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
dims = [max(c[i] for c in corners) - min(c[i] for c in corners) for i in range(3)] if corners else [0, 0, 0]

materials = {}
for o in meshes:
    for slot in o.material_slots:
        m = slot.material
        if not m or m.name in materials:
            continue
        textures = []
        if m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image:
                    # Which BSDF input is the texture (directly) connected to?
                    targets = sorted({l.to_socket.name for l in n.outputs[0].links} |
                                     {l.to_socket.name for s in n.outputs for l in s.links})
                    textures.append({"image": n.image.name, "size": list(n.image.size), "to": targets})
        materials[m.name] = {"textures": textures}

report = {
    "file": os.path.basename(path),
    "mesh_objects": len(meshes),
    "faces": faces,
    "triangle_equivalent": tris,
    "quad_ratio": round(quads / max(faces, 1), 2),
    "dimensions_m": [round(d, 3) for d in dims],
    "num_materials": len(materials),
    "materials": materials,
}
print("@@REPORT@@" + json.dumps(report, ensure_ascii=False))
if json_out:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
