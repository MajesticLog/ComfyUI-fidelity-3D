"""
animate_drop.py — Video demo, "animated depth": the bag drops onto a round table and changes
colour AT CONTACT (powder pink + gold hardware → lemon yellow + silver hardware).

3D sets the shape, the motion, and WHERE / WHEN the colours change. The AI (Wan VACE) renders
the material, the light and the set, and is left free to "animate" the transition.

FOR EACH FRAME, in --out:
    simple/NNNN.png        simple render (Cycles 16 samples, correct colours, white table,
                           transparent background → flattened on grey) → start video for the AI
    depth/NNNN.png         disparity depth, table included → shape control
    beauty/NNNN.png        Cycles render of the bag alone, transparent background → QA / re-injection reference
    mask_object/NNNN.png   bag silhouette
    mask_metal/NNNN.png    charms and hardware
    mask_leather/NNNN.png  the rest of the bag
plus anim.json (timing, colours, camera).

FIXED camera (same direction as cam05, pulled back to see the drop).
RIGID animation: the strap follows the bag (no cloth simulation yet).

Usage:
    blender -b --factory-startup --python animate_drop.py -- --import bag.glb --out ../renders/drop
"""

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_passes as ep  # noqa: E402

COLORS = {
    # (leather, metal) in sRGB, converted to linear below.
    "before": {"leather": (0.92, 0.73, 0.74), "metal": (1.00, 0.78, 0.40), "name": "powder pink + gold"},
    "after": {"leather": (0.98, 0.90, 0.22), "metal": (0.86, 0.87, 0.89), "name": "lemon yellow + silver"},
}


def srgb_to_linear(c):
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def read_arguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--import", dest="import_file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hdri", default=None)
    p.add_argument("--res", type=int, default=640)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--contact", type=int, default=10, help="frame of first contact with the table")
    p.add_argument("--height", type=float, default=0.45,
                   help="start height, as a multiple of the bag's height")
    p.add_argument("--pullback", type=float, default=1.15, help="camera pull-back relative to cam05")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--passes", default="simple,depth,beauty,masks")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    return p.parse_args(argv)


def height_and_tilt(f, contact, h0):
    """Constant-acceleration drop (slow motion), small bounce, then rest.
    Returns (height above the table, tilt in radians)."""
    if f <= contact:
        t = f / contact
        z = h0 * (1 - t * t)  # parabola: speed increases until contact
        tilt = math.radians(6) * (1 - t)  # slight tilt that straightens out
        return z, tilt
    r = f - contact
    bounce = 4  # frames
    if r <= bounce:
        x = r / bounce
        return h0 * 0.06 * 4 * x * (1 - x), 0.0  # small bump
    return 0.0, 0.0


def leather_material(color_srgb):
    m = ep.principled_material("_anim_leather", srgb_to_linear(color_srgb), metal=0.0, roughness=0.45)
    m.diffuse_color = (*color_srgb, 1.0)  # colour used by Workbench
    return m


def metal_material(color_srgb):
    m = ep.principled_material("_anim_metal", srgb_to_linear(color_srgb), metal=1.0, roughness=0.25)
    m.diffuse_color = (*color_srgb, 1.0)
    m.metallic = 1.0
    return m


def main():
    a = read_arguments()
    passes = {x.strip() for x in a.passes.split(",")}
    out = os.path.abspath(a.out)
    for d in ("simple", "depth", "beauty", "mask_object", "mask_metal", "mask_leather"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    ep.import_model(os.path.abspath(a.import_file))
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = a.res
    scene.render.resolution_percentage = 100
    ep.enable_gpu(scene)
    if a.hdri:
        ep.prepare_world(scene, os.path.abspath(a.hdri))

    objects = ep.rendered_objects()
    corners, center, _ = ep.bounding_box(objects)
    bag_height = max(c.z for c in corners) - min(c.z for c in corners)

    # Metal = objects with a material whose name contains "metal"; everything else = leather.
    def is_metal(o):
        return any(s.material and "metal" in s.material.name.lower() for s in o.material_slots)
    metal = [o for o in objects if is_metal(o)]
    leather = [o for o in objects if not is_metal(o)]
    print(f"[bag] metal: {[o.name for o in metal]} | leather: {[o.name for o in leather]}")

    # Camera: same direction as cam05 ("back three-quarter"), pulled back to see the drop.
    cam = ep.create_cameras(scene, center, corners, 5, 85.0)[-1]
    direction = (cam.location - center).normalized()
    distance = (cam.location - center).length * a.pullback
    target = center + Vector((0, 0, bag_height * a.height * 0.25))
    cam.location = target + direction * distance
    cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_end = distance * 10
    scene.camera = cam

    # The whole bag under a pivot: we animate the pivot (rigid animation).
    pivot = bpy.data.objects.new("_bag_pivot", None)
    scene.collection.objects.link(pivot)
    pivot.location = Vector((center.x, center.y, min(c.z for c in corners)))
    bpy.context.view_layer.update()
    for o in objects:
        mw = o.matrix_world.copy()
        o.parent = pivot
        o.matrix_parent_inverse = pivot.matrix_world.inverted()
        o.matrix_world = mw

    table, table_rim = ep.create_table(corners, 2.0, 0.1)
    white_table = ep.principled_material("_anim_table", (0.85, 0.85, 0.83), roughness=0.35)
    white_table.diffuse_color = (0.92, 0.92, 0.9, 1.0)
    table.data.materials.append(white_table)
    bounce = ep.create_bounce(corners)
    bounce.hide_render = True

    mats = {k: (leather_material(v["leather"]), metal_material(v["metal"])) for k, v in COLORS.items()}
    m_depth = ep.disparity_depth_material()
    m_depth.node_tree.nodes["gamma"].inputs[1].default_value = 1.0
    m_white, m_black = ep.flat_material("_anim_white", 1.0), ep.flat_material("_anim_black", 0.0)

    # Fixed depth range for the whole sequence (otherwise the depth would "breathe" from frame to frame).
    highest = [c + Vector((0, 0, bag_height * a.height)) for c in corners]
    near, _ = ep.depth_range(cam, corners + highest)
    _, far = ep.depth_range(cam, table_rim)
    remap = m_depth.node_tree.nodes["remap"]
    remap.inputs["From Min"].default_value = 1.0 / far
    remap.inputs["From Max"].default_value = 1.0 / near

    settings = ep.RenderSettings(scene)  # original state (AgX, etc.), restored after each pass
    swap = ep.MaterialSwap(objects)
    swap_t = ep.MaterialSwap(objects + [table])
    end = a.frames if a.end is None else min(a.end, a.frames)
    timeline = []
    for f in range(a.start, end):
        z, tilt = height_and_tilt(f, a.contact, bag_height * a.height)
        pivot.location.z = min(c.z for c in corners) + z
        pivot.rotation_euler = Euler((tilt * 0.6, -tilt, 0))
        bpy.context.view_layer.update()
        phase = "before" if f < a.contact else "after"
        m_leather, m_metal = mats[phase]
        name = f"{f:04d}.png"
        timeline.append({"frame": f, "height": z, "phase": phase})

        def colors(m):
            return m_metal if (m and "metal" in m.name.lower()) else m_leather

        if "simple" in passes:
            # "Simple" render = quick Cycles, WITH the table, transparent background (flattened on a
            # neutral grey afterwards by flatten_simple.py). Workbench gave dull, wrong colours
            # (pink → mauve, lemon → olive), which would have misled the AI about the target colour.
            scene.render.engine = "CYCLES"
            settings.beauty_mode(16)
            ep.set_world_strength(scene, 0.8)
            table.hide_render = False
            swap.apply(colors)
            try:
                ep.render(scene, os.path.join(out, "simple", name), "RGBA")
            finally:
                swap.restore()
                table.hide_render = True
            settings.restore()

        if "beauty" in passes:
            scene.render.engine = "CYCLES"
            settings.beauty_mode(a.samples)
            ep.set_world_strength(scene, 0.8)
            bounce.hide_render = False
            swap.apply(colors)
            try:
                ep.render(scene, os.path.join(out, "beauty", name), "RGBA")
            finally:
                swap.restore()
                bounce.hide_render = True

        scene.render.engine = "CYCLES"
        settings.data_mode()
        if "depth" in passes:
            table.hide_render = False
            swap_t.apply(lambda m: m_depth)
            try:
                ep.render(scene, os.path.join(out, "depth", name), "BW")
            finally:
                swap_t.restore()
                table.hide_render = True
        if "masks" in passes:
            for folder, choose in (("mask_object", lambda m: m_white),
                                   ("mask_metal", lambda m: m_white if (m and "metal" in m.name.lower()) else m_black),
                                   ("mask_leather", lambda m: m_black if (m and "metal" in m.name.lower()) else m_white)):
                swap.apply(choose)
                try:
                    ep.render(scene, os.path.join(out, folder, name), "BW")
                finally:
                    swap.restore()
        settings.restore()
        print(f"[frame {f}] z={z:.3f} phase={phase}")

    with open(os.path.join(out, "anim.json"), "w", encoding="utf-8") as fp:
        json.dump({"frames": a.frames, "fps": 16, "contact": a.contact, "resolution": a.res,
                   "colors": COLORS, "camera": cam.name, "pullback": a.pullback,
                   "start_height": a.height, "timeline": timeline}, fp, indent=2, ensure_ascii=False)
    print("[done]")


if __name__ == "__main__":
    main()
