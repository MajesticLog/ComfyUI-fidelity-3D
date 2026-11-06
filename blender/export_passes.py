"""
export_passes.py — Renders the beauty and the data passes needed by the
"the render is the ground truth" pipeline, the fidelity QA, and the video demo.

FOR EACH CAMERA the script writes:
    camXX_beauty.png         final render, transparent background (RGBA)
    camXX_depth.png          depth: white = near, black = far / background
    camXX_normal.png         CAMERA-SPACE normals (ControlNet convention:
                             a surface facing the camera is lavender blue)
    camXX_mask_object.png    silhouette of the whole object (white on black)
    camXX_mask_<mat>.png     one mask per MATERIAL (leather, metal, logo…)
and a cameras.json file describing everything (cameras, depth ranges, files).

WHY NOT THE COMPOSITOR?
Blender's "native" passes (Z, Normal, Cryptomatte) go through the compositor,
whose API changed a lot in Blender 5. This script is simpler and more robust:
it re-renders quickly while temporarily swapping every material for an
"emission" material that outputs the wanted data directly (depth, normal,
white/black). Original materials are always restored, even on error.

USAGE (in a terminal):
    # On a prepared .blend file:
    blender -b my_bag.blend --python export_passes.py -- --out path/to/renders

    # Importing a .glb / .gltf / .fbx / .obj directly:
    blender -b --factory-startup --python export_passes.py -- --import bag.glb --out renders

    # Demo scene (sphere + clasp + logo) to test the script:
    blender -b --factory-startup --python export_passes.py -- --demo --out renders_demo

USEFUL OPTIONS:
    --res 1024          resolution (square)
    --samples 256       Cycles samples for the beauty
    --num-cams 5        number of automatic cameras (if no "CAM…" camera exists)
    --focal 85          focal length of automatic cameras, in mm
    --hdri path.exr     HDRI lighting for the beauty
    --passes beauty,depth,normal,masks,depth_table   to render only some passes

TIP: rename your materials in Blender BEFORE exporting ("leather", "metal",
"logo"…). Masks are named after the materials.

CAMERAS: if the scene contains cameras whose name starts with "CAM", they are
used as is. Otherwise the script places cameras automatically around the
object (front, three-quarter, profile, high angle, back).
"""

import argparse
import json
import math
import os
import re
import sys

import bpy
from mathutils import Vector


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def read_arguments():
    # Blender ignores everything after "--": that is where our options go.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Render beauty and data passes")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--import", dest="import_file", help="model to import (.glb, .gltf, .fbx, .obj)")
    p.add_argument("--demo", action="store_true", help="build a test scene")
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--num-cams", type=int, default=5)
    p.add_argument("--focal", type=float, default=85.0)
    p.add_argument("--hdri", help="HDRI image (.exr/.hdr) to light the beauty")
    p.add_argument("--passes", default="beauty,depth,normal,masks",
                   help="list among beauty,depth,normal,masks,depth_table")
    p.add_argument("--table", type=float, default=2.0, metavar="RATIO",
                   help="depth_table pass: diameter of the round table, as a multiple of the object's largest "
                        "horizontal dimension (2.0 ≈ 60 cm for a 30 cm bag)")
    p.add_argument("--table-thickness", type=float, default=0.1, metavar="RATIO",
                   help="table top thickness, same unit as --table (0.1 ≈ 3 cm)")
    p.add_argument("--cams", help="only process these cameras, e.g. 'cam05' or 'cam02,cam05'")
    p.add_argument("--sun", metavar="AZ,EL",
                   help="add a sun RELATIVE TO THE CAMERA. AZ: 0 = behind the camera (front light), "
                        "180 = behind the object (backlight), positive = image left. EL: elevation in degrees")
    p.add_argument("--sun-color", default="1.0,0.72,0.45", metavar="R,G,B")
    p.add_argument("--sun-strength", type=float, default=5.0, help="sun irradiance (W/m²)")
    p.add_argument("--sun-angle", type=float, default=2.0,
                   help="apparent diameter in degrees (larger = softer shadows)")
    p.add_argument("--hdri-strength", type=float, default=1.0, help="multiplies the world (HDRI) lighting")
    p.add_argument("--bounce", action="store_true",
                   help="white plane under the object, invisible to the camera, bouncing light (a pale table)")
    p.add_argument("--depth-gamma", type=float, default=1.0,
                   help="depth_table pass: value**gamma curve applied after normalization. "
                        ">1 = more contrast in bright (near) areas, i.e. the bag; "
                        "<1 = more contrast in dark (far) areas, i.e. the back of the table")
    p.add_argument("--zone-uv", action="append", default=[], metavar="NAME:MATERIAL:u0,v0,u1,v1",
                   help="extra mask: the part of a material whose UVs fall inside this rectangle "
                        "(useful when a logo/text is painted in the same texture as the rest)")
    return p.parse_args(argv)


def read_uv_zones(specs):
    zones = []
    for s in specs:
        name, material, rect = s.split(":")
        u0, v0, u1, v1 = (float(x) for x in rect.split(","))
        zones.append((name, material, (u0, v0, u1, v1)))
    return zones


# ---------------------------------------------------------------------------
# Scene preparation
# ---------------------------------------------------------------------------

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def import_model(path):
    clear_scene()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    print(f"[import] {path}")


def principled_material(name, color, metal=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def build_demo():
    """Small test scene: a "bag" (flattened sphere), a gold clasp and an embossed logo."""
    clear_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32)
    bag = bpy.context.active_object
    bag.name = "bag"
    bag.scale = (1.0, 0.45, 0.8)
    bpy.ops.object.shade_smooth()
    bag.data.materials.append(principled_material("leather", (0.35, 0.05, 0.04), roughness=0.45))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.47, 0.15))
    clasp = bpy.context.active_object
    clasp.name = "clasp"
    clasp.scale = (0.45, 0.06, 0.28)
    clasp.data.materials.append(principled_material("metal", (1.0, 0.77, 0.34), metal=1.0, roughness=0.2))

    bpy.ops.object.text_add(location=(0, -0.505, 0.12), rotation=(math.radians(90), 0, 0))
    text = bpy.context.active_object
    text.data.body = "MAISON"
    text.data.align_x = "CENTER"
    text.data.size = 0.13
    text.data.extrude = 0.004
    bpy.ops.object.convert(target="MESH")
    text = bpy.context.active_object
    text.name = "logo"
    text.data.materials.clear()
    text.data.materials.append(principled_material("logo", (0.9, 0.85, 0.75), roughness=0.3))

    bpy.ops.object.light_add(type="AREA", location=(2.5, -3.0, 3.0))
    light = bpy.context.active_object
    light.data.energy = 600
    light.data.size = 2.0
    light.rotation_euler = (math.radians(50), 0, math.radians(40))
    print("[demo] test scene built")


def prepare_world(scene, hdri):
    """Lighting for the beauty: the HDRI if given, otherwise neutral grey if no world exists."""
    if hdri:
        world = bpy.data.worlds.new("world_hdri")
        world.use_nodes = True
        nt = world.node_tree
        background = nt.nodes.get("Background")
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri)
        nt.links.new(env.outputs["Color"], background.inputs["Color"])
        scene.world = world
    elif scene.world is None:
        world = bpy.data.worlds.new("world_neutral")
        world.use_nodes = True
        world.node_tree.nodes.get("Background").inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        scene.world = world


def set_world_strength(scene, strength):
    if scene.world and scene.world.use_nodes:
        background = scene.world.node_tree.nodes.get("Background")
        if background:
            background.inputs["Strength"].default_value = strength


def create_sun(color, strength, angle_deg):
    data = bpy.data.lights.new("_QA_sun", type="SUN")
    data.color = color
    data.energy = strength
    data.angle = math.radians(angle_deg)
    sun = bpy.data.objects.new("_QA_sun", data)
    bpy.context.scene.collection.objects.link(sun)
    return sun


def orient_sun(sun, cam, az_deg, el_deg):
    """Places the sun relative to the camera (see --sun). We compute where the light COMES FROM in
    the camera frame, then orient the sun (which shines along its local -Z)."""
    q = cam.matrix_world.to_quaternion()
    forward, right = q @ Vector((0, 0, -1)), q @ Vector((1, 0, 0))
    az, el = math.radians(az_deg), math.radians(el_deg)
    horizontal = (math.cos(az) * -forward + math.sin(az) * -right)
    horizontal.z = 0  # elevation is measured from the ground, not from the camera axis
    horizontal.normalize()
    to_light = horizontal * math.cos(el) + Vector((0, 0, 1)) * math.sin(el)
    sun.rotation_euler = (-to_light).to_track_quat("-Z", "Y").to_euler()


def create_bounce(corners):
    """Matte white plane under the object: it bounces light like a pale table but stays invisible
    to the camera (the beauty keeps its transparent background)."""
    min_z = min(c.z for c in corners)
    width = max(max(c.x for c in corners) - min(c.x for c in corners),
                max(c.y for c in corners) - min(c.y for c in corners))
    cx = (max(c.x for c in corners) + min(c.x for c in corners)) / 2
    cy = (max(c.y for c in corners) + min(c.y for c in corners)) / 2
    bpy.ops.mesh.primitive_plane_add(size=width * 4, location=(cx, cy, min_z - width * 0.001))
    plane = bpy.context.active_object
    plane.name = "_QA_bounce"
    plane.visible_camera = False
    plane.data.materials.append(principled_material("_QA_bounce_mat", (0.8, 0.8, 0.78), roughness=0.6))
    return plane


def enable_gpu(scene):
    """Tries to use the graphics card (OptiX, then CUDA) for Cycles."""
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for gpu_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = gpu_type
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == gpu_type]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == gpu_type
                    scene.cycles.device = "GPU"
                    print(f"[gpu] {gpu_type}: {', '.join(d.name for d in gpus)}")
                    return
            except TypeError:
                continue
    except Exception as e:  # not blocking: render on the CPU
        print(f"[gpu] unavailable ({e}), rendering on CPU")
    scene.cycles.device = "CPU"


# ---------------------------------------------------------------------------
# Geometry: objects, bounding box, cameras
# ---------------------------------------------------------------------------

def rendered_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_render]


def bounding_box(objects):
    corners = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return corners, (lo + hi) / 2, (hi - lo).length / 2


# (name, azimuth in degrees around Z, elevation in degrees). Azimuth 0 = front view (from -Y).
AUTO_VIEWS = [
    ("front", 0, 8),
    ("three_quarter", 35, 15),
    ("profile", 90, 5),
    ("high_angle", 20, 45),
    ("back_three_quarter", 150, 15),
    ("three_quarter_left", -35, 15),
    ("low_angle", 25, -12),
    ("top", 0, 80),
]


def framing_distance(corners, center, direction, fov, fill=0.9):
    """Smallest distance at which every bounding-box corner fits in the frame
    (using at most `fill` of the half field of view). Binary search."""
    rot = (-direction).to_track_quat("-Z", "Y")
    right, up, forward = rot @ Vector((1, 0, 0)), rot @ Vector((0, 1, 0)), -direction
    limit = math.tan(fov / 2) * fill

    def fits(d):
        pos = center + direction * d
        for c in corners:
            v = c - pos
            z = v.dot(forward)
            if z <= 0 or max(abs(v.dot(right)), abs(v.dot(up))) / z > limit:
                return False
        return True

    low, high = 0.0, 1.0
    while not fits(high):
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        low, high = (low, mid) if fits(mid) else (mid, high)
    return high


def create_cameras(scene, center, corners, count, focal):
    cams = []
    for i, (name, az, el) in enumerate(AUTO_VIEWS[:count], start=1):
        data = bpy.data.cameras.new(f"CAM{i:02d}_{name}")
        data.lens = focal
        data.sensor_width = 36.0
        cam = bpy.data.objects.new(f"CAM{i:02d}_{name}", data)
        scene.collection.objects.link(cam)
        fov = 2 * math.atan(data.sensor_width / 2 / focal)
        a, e = math.radians(az), math.radians(el)
        direction = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        # Tight framing: the object fills ~90 % of the frame along its largest dimension.
        distance = framing_distance(corners, center, direction, fov)
        cam.location = center + direction * distance
        cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        data.clip_start = max(0.001, distance * 0.01)
        data.clip_end = distance * 10
        cams.append(cam)
    # Without this update, matrix_world of the new cameras is not computed yet.
    bpy.context.view_layer.update()
    print(f"[cameras] {len(cams)} cameras created automatically")
    return cams


def depth_range(cam, corners):
    """Minimum and maximum distance of the object along the view axis (to normalize the depth)."""
    forward = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    z = [(c - cam.matrix_world.translation).dot(forward) for c in corners]
    margin = (max(z) - min(z)) * 0.02
    return max(1e-4, min(z) - margin), max(z) + margin


# ---------------------------------------------------------------------------
# "Data" materials (pure emission)
# ---------------------------------------------------------------------------

def emission_material(name, build):
    """Creates a material whose output is an emission driven by build(nt) -> color socket."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    output = nt.nodes.new("ShaderNodeOutputMaterial")
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    socket = build(nt)
    if isinstance(socket, tuple):
        emission.inputs["Color"].default_value = socket
    else:
        nt.links.new(socket, emission.inputs["Color"])
    nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def flat_material(name, value):
    return emission_material(name, lambda nt: (value, value, value, 1.0))


def depth_material():
    def build(nt):
        cam = nt.nodes.new("ShaderNodeCameraData")
        remap = nt.nodes.new("ShaderNodeMapRange")
        remap.name = "remap"
        remap.clamp = True
        remap.inputs["To Min"].default_value = 1.0  # near = white
        remap.inputs["To Max"].default_value = 0.0  # far = black
        nt.links.new(cam.outputs["View Z Depth"], remap.inputs["Value"])
        return remap.outputs["Result"]
    return emission_material("_QA_depth", build)


def disparity_depth_material():
    """Depth "the Depth Anything way": we encode 1/z (disparity), not z.
    Why: depth ControlNets are used with estimator maps (MiDaS, Depth Anything) that output disparity.
    It naturally gives more nuance to the foreground (the bag) and compresses the background (the back of
    the table), like a real photo run through the estimator.
    value = ((1/z - 1/far) / (1/near - 1/far)) ** gamma, clamped to [0, 1]."""
    def build(nt):
        cam = nt.nodes.new("ShaderNodeCameraData")
        inverse = nt.nodes.new("ShaderNodeMath")
        inverse.operation = "DIVIDE"
        inverse.inputs[0].default_value = 1.0
        nt.links.new(cam.outputs["View Z Depth"], inverse.inputs[1])
        remap = nt.nodes.new("ShaderNodeMapRange")
        remap.name = "remap"
        remap.clamp = True
        remap.inputs["To Min"].default_value = 0.0
        remap.inputs["To Max"].default_value = 1.0
        nt.links.new(inverse.outputs[0], remap.inputs["Value"])
        curve = nt.nodes.new("ShaderNodeMath")
        curve.name = "gamma"
        curve.operation = "POWER"
        curve.use_clamp = True
        nt.links.new(remap.outputs["Result"], curve.inputs[0])
        return curve.outputs[0]
    return emission_material("_QA_depth_disparity", build)


def create_table(corners, diameter_ratio, thickness_ratio):
    """Round table top under the object, visible ONLY in the depth_table pass.
    It gives the ControlNet the table plane, hence the perspective and scale of the scene."""
    min_z = min(c.z for c in corners)
    width = max(max(c.x for c in corners) - min(c.x for c in corners),
                max(c.y for c in corners) - min(c.y for c in corners))
    radius = width * diameter_ratio / 2
    thickness = width * thickness_ratio
    cx = (max(c.x for c in corners) + min(c.x for c in corners)) / 2
    cy = (max(c.y for c in corners) + min(c.y for c in corners)) / 2
    # The table top sits 1 mm below the object's lowest point: the bag "rests" on it
    # without the two surfaces overlapping (z-fighting).
    top = min_z - width * 0.001
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=radius, depth=thickness,
                                        location=(cx, cy, top - thickness / 2))
    table = bpy.context.active_object
    table.name = "_QA_table"
    table.hide_render = True
    print(f"[table] diameter {2 * radius:.3f}, thickness {thickness:.3f}, top at z={top:.3f}")
    # Rim points, used to compute the depth range.
    rim = [Vector((cx + radius * math.cos(a), cy + radius * math.sin(a), top))
           for a in (i * 2 * math.pi / 64 for i in range(64))]
    return table, rim


def normal_material():
    def build(nt):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        transform = nt.nodes.new("ShaderNodeVectorTransform")
        transform.vector_type = "NORMAL"
        transform.convert_from = "WORLD"
        transform.convert_to = "CAMERA"
        # Blender camera space: X right, Y up, Z forward (into the scene).
        # ControlNet convention: X right, Y up, Z towards the camera -> flip Z.
        # Then map [-1, 1] to [0, 1]: n * 0.5 + 0.5.
        remap = nt.nodes.new("ShaderNodeVectorMath")
        remap.name = "remap_normals"
        remap.operation = "MULTIPLY_ADD"
        remap.inputs[1].default_value = (0.5, 0.5, -0.5)
        remap.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(geo.outputs["Normal"], transform.inputs["Vector"])
        nt.links.new(transform.outputs["Vector"], remap.inputs[0])
        return remap.outputs["Vector"]
    return emission_material("_QA_normal", build)


def uv_zone_material(name, rect):
    """White where UV coordinates fall inside the rectangle (u0,v0)-(u1,v1), black elsewhere."""
    u0, v0, u1, v1 = rect

    def build(nt):
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(coord.outputs["UV"], sep.inputs["Vector"])
        result = None
        for component, lo, hi in (("X", u0, u1), ("Y", v0, v1)):
            for operation, threshold in (("GREATER_THAN", lo), ("LESS_THAN", hi)):
                test = nt.nodes.new("ShaderNodeMath")
                test.operation = operation
                test.inputs[1].default_value = threshold
                nt.links.new(sep.outputs[component], test.inputs[0])
                if result is None:
                    result = test.outputs[0]
                else:
                    product = nt.nodes.new("ShaderNodeMath")
                    product.operation = "MULTIPLY"  # logical "and": 1 * 1 = 1
                    nt.links.new(result, product.inputs[0])
                    nt.links.new(test.outputs[0], product.inputs[1])
                    result = product.outputs[0]
        return result
    return emission_material(f"_QA_zone_{name}", build)


class MaterialSwap:
    """Temporarily replaces the material of every slot, then restores everything (even on error)."""

    def __init__(self, objects):
        self.objects = objects
        self.saved = []
        self.added_slots = []

    def apply(self, choose):
        """choose(original_material) -> replacement material."""
        for obj in self.objects:
            if len(obj.material_slots) == 0:
                obj.data.materials.append(None)
                self.added_slots.append(obj)
            for i, slot in enumerate(obj.material_slots):
                self.saved.append((obj, i, slot.material))
                slot.material = choose(slot.material)

    def restore(self):
        for obj, i, mat in reversed(self.saved):
            obj.material_slots[i].material = mat
        for obj in self.added_slots:
            obj.data.materials.pop()
        self.saved.clear()
        self.added_slots.clear()


def safe_filename(name):
    name = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_").lower()
    return name or "unnamed"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class RenderSettings:
    """Saves / restores the settings we change for the data passes."""

    def __init__(self, scene):
        self.scene = scene
        vs = scene.view_settings
        self.state = dict(
            samples=scene.cycles.samples,
            denoise=scene.cycles.use_denoising,
            transparent=scene.render.film_transparent,
            view=vs.view_transform, look=vs.look, exposure=vs.exposure, gamma=vs.gamma,
        )

    def data_mode(self):
        s = self.scene
        s.cycles.samples = 16
        s.cycles.use_denoising = False  # denoising would damage the sharp edges of the masks
        s.render.film_transparent = True  # black background once alpha is dropped
        try:
            s.view_settings.view_transform = "Raw"  # values written as is, no "look"
        except TypeError:
            s.view_settings.view_transform = "Standard"
        s.view_settings.look = "None"
        s.view_settings.exposure = 0.0
        s.view_settings.gamma = 1.0

    def beauty_mode(self, samples):
        self.restore()
        self.scene.cycles.samples = samples
        self.scene.cycles.use_denoising = True
        self.scene.render.film_transparent = True

    def restore(self):
        s, e = self.scene, self.state
        s.cycles.samples = e["samples"]
        s.cycles.use_denoising = e["denoise"]
        s.render.film_transparent = e["transparent"]
        s.view_settings.view_transform = e["view"]
        s.view_settings.look = e["look"]
        s.view_settings.exposure = e["exposure"]
        s.view_settings.gamma = e["gamma"]


def render(scene, path, color_mode):
    # 8-bit PNG everywhere: what ComfyUI loads most reliably.
    settings = scene.render.image_settings
    settings.file_format = "PNG"
    settings.color_mode = color_mode
    settings.color_depth = "8"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"  -> {os.path.basename(path)}")


def main():
    args = read_arguments()
    passes = {p.strip() for p in args.passes.split(",")}
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    if args.demo:
        build_demo()
    elif args.import_file:
        import_model(os.path.abspath(args.import_file))

    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    enable_gpu(scene)
    prepare_world(scene, args.hdri)

    objects = rendered_objects()
    if not objects:
        raise RuntimeError("No renderable mesh object in the scene.")
    corners, center, radius = bounding_box(objects)

    cams = sorted([o for o in scene.objects if o.type == "CAMERA" and o.name.upper().startswith("CAM")],
                  key=lambda o: o.name)
    if cams:
        print(f"[cameras] {len(cams)} 'CAM…' cameras found in the scene")
    else:
        cams = create_cameras(scene, center, corners, args.num_cams, args.focal)
    if args.cams:
        wanted = {c.strip().lower() for c in args.cams.split(",")}
        cams = [c for c in cams if c.name.split("_")[0].lower() in wanted]
        print(f"[cameras] --cams filter: {[c.name for c in cams]}")

    set_world_strength(scene, args.hdri_strength)
    sun = create_sun(tuple(float(x) for x in args.sun_color.split(",")),
                     args.sun_strength, args.sun_angle) if args.sun else None
    bounce = create_bounce(corners) if args.bounce else None
    # The bounce plane only exists for the beauty: hide it during the data passes.
    if bounce:
        bounce.hide_render = True

    # Original materials on the objects (one mask per material).
    materials = []
    for obj in objects:
        for slot in obj.material_slots:
            if slot.material and slot.material not in materials:
                materials.append(slot.material)
    mask_names = {}
    for m in materials:
        name = safe_filename(m.name)
        while name in mask_names.values() or name == "object":
            name += "_bis"
        mask_names[m.name] = name

    m_depth, m_normal = depth_material(), normal_material()
    m_white, m_black = flat_material("_QA_white", 1.0), flat_material("_QA_black", 0.0)
    uv_zones = []
    for name, target, rect in read_uv_zones(args.zone_uv):
        if target not in mask_names.values():
            raise ValueError(f"--zone-uv: unknown material '{target}'. Choices: {sorted(mask_names.values())}")
        uv_zones.append((safe_filename(name), target, uv_zone_material(name, rect)))
    table = table_rim = m_depth_table = None
    if "depth_table" in passes:
        table, table_rim = create_table(corners, args.table, args.table_thickness)
        m_depth_table = disparity_depth_material()
        m_depth_table.node_tree.nodes["gamma"].inputs[1].default_value = args.depth_gamma
    settings = RenderSettings(scene)
    swap = MaterialSwap(objects)
    info = {"resolution": args.res, "materials": mask_names, "cameras": [],
            "uv_zones": {n: {"material": c, "rect": r} for n, c, r in read_uv_zones(args.zone_uv)}}

    try:
        for cam in cams:
            scene.camera = cam
            prefix = os.path.join(out, cam.name.split("_")[0].lower())  # e.g. "cam01"
            near, far = depth_range(cam, corners)
            print(f"[{cam.name}] depth {near:.3f} -> {far:.3f}")
            files = {}

            if sun:
                az, el = (float(x) for x in args.sun.split(","))
                orient_sun(sun, cam, az, el)
            if "beauty" in passes:
                settings.beauty_mode(args.samples)
                if bounce:
                    bounce.hide_render = False
                try:
                    render(scene, prefix + "_beauty.png", "RGBA")
                finally:
                    if bounce:
                        bounce.hide_render = True
                files["beauty"] = os.path.basename(prefix + "_beauty.png")

            settings.data_mode()
            if "depth" in passes:
                remap = m_depth.node_tree.nodes["remap"]
                remap.inputs["From Min"].default_value = near
                remap.inputs["From Max"].default_value = far
                swap.apply(lambda m: m_depth)
                try:
                    render(scene, prefix + "_depth.png", "BW")
                finally:
                    swap.restore()
                files["depth"] = os.path.basename(prefix + "_depth.png")

            if "depth_table" in passes:
                # Range: from the nearest point of the OBJECT (anything nearer, front edge of the table
                # included, saturates to white) to the farthest edge of the TABLE (the background
                # beyond stays black = far).
                near_t, _ = depth_range(cam, corners)
                _, far_t = depth_range(cam, table_rim)
                remap = m_depth_table.node_tree.nodes["remap"]
                remap.inputs["From Min"].default_value = 1.0 / far_t
                remap.inputs["From Max"].default_value = 1.0 / near_t
                swap_t = MaterialSwap(objects + [table])
                table.hide_render = False
                swap_t.apply(lambda m: m_depth_table)
                try:
                    render(scene, prefix + "_depth_table.png", "BW")
                finally:
                    swap_t.restore()
                    table.hide_render = True
                files["depth_table"] = os.path.basename(prefix + "_depth_table.png")
                print(f"  depth_table: disparity from {near_t:.3f} to {far_t:.3f}, gamma {args.depth_gamma}")

            if "normal" in passes:
                swap.apply(lambda m: m_normal)
                try:
                    render(scene, prefix + "_normal.png", "RGB")
                finally:
                    swap.restore()
                files["normal"] = os.path.basename(prefix + "_normal.png")

            if "masks" in passes:
                masks = {}
                swap.apply(lambda m: m_white)
                try:
                    render(scene, prefix + "_mask_object.png", "BW")
                finally:
                    swap.restore()
                masks["object"] = os.path.basename(prefix + "_mask_object.png")
                for mat in materials:
                    name = mask_names[mat.name]
                    # Target material in white, everything else in black: occlusions are respected.
                    swap.apply(lambda m, target=mat: m_white if m == target else m_black)
                    try:
                        render(scene, f"{prefix}_mask_{name}.png", "BW")
                    finally:
                        swap.restore()
                    masks[name] = os.path.basename(f"{prefix}_mask_{name}.png")
                for name, target_name, zone_mat in uv_zones:
                    swap.apply(lambda m, c=target_name, z=zone_mat:
                               z if m is not None and mask_names.get(m.name) == c else m_black)
                    try:
                        render(scene, f"{prefix}_mask_{name}.png", "BW")
                    finally:
                        swap.restore()
                    masks[name] = os.path.basename(f"{prefix}_mask_{name}.png")
                files["masks"] = masks

            info["cameras"].append({
                "name": cam.name,
                "prefix": os.path.basename(prefix),
                "focal_mm": cam.data.lens,
                "sensor_mm": cam.data.sensor_width,
                "world_matrix": [list(r) for r in cam.matrix_world],
                "depth_near": near,
                "depth_far": far,
                "files": files,
            })
    finally:
        swap.restore()
        settings.restore()
        for m in (m_depth, m_normal, m_white, m_black, m_depth_table, *(z[2] for z in uv_zones)):
            if m is not None:
                bpy.data.materials.remove(m)
        if table is not None:
            bpy.data.objects.remove(table, do_unlink=True)

    with open(os.path.join(out, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[done] {len(cams)} cameras exported to {out}")


if __name__ == "__main__":
    main()
