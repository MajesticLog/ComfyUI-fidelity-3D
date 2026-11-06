"""
fidelity_qa.py — Product fidelity quality control.

QUESTION ASKED: "does the produced image (by the AI or the pipeline) respect the product as it is
in the reference 3D render?", zone by zone.

INPUTS
    --ref    the reference render (camXX_beauty.png, output of the Blender script)
    --gen    the image to check, PIXEL-ALIGNED with the reference
             (same camera: true for the still and video demos)
    --masks  (optional) glob pattern of the masks; default camXX_mask_*.png next to --ref
    --out    report folder

OUTPUT
    report.html      pass/fail table per zone + images
    results.json     every number
    heatmap.png      map of structural differences
    edges.png        red = lost detail, cyan = invented detail

THE THREE MEASURES (per zone)
1. STRUCTURE (SSIM, 0 to 1): did shapes, patterns and lettering stay the same? The overall light
   difference of the zone is neutralized first (same mean and contrast), because harmonization is
   ALLOWED to change the lighting, but not the product's drawing.
2. EDGES (F1 score, 0 to 1): sharp lines (Canny) are detected in both images and must match within
   ±2 pixels. Very sensitive to stitches, engravings and logo letters.
3. HUE (chromatic ΔE): perceived colour difference, ignoring lightness (the reference lightness is
   kept and only hue/saturation are compared). Red leather turning brown or gold turning yellow is
   caught, a zone simply darker because of a shadow is not.

THRESHOLDS: starting values, TO BE CALIBRATED on real examples.
"""

import argparse
import base64
import glob
import html
import json
import os
import sys

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

DEFAULT_THRESHOLDS = {
    "ssim_min": 0.80,          # structure
    "edges_f1_min": 0.75,      # edges
    "delta_e_chroma_max": 5.0,  # hue (median over the zone); < 2 for a brand colour
}
EDGE_TOLERANCE_PX = 2
MIN_ZONE_PIXELS = 50        # below this, the zone is too small to be judged
MIN_EDGE_PIXELS = 150       # absolute minimum of edge pixels (and at least 0.5 % of the zone)
OBJECT_MASK_NAMES = ("object", "objet")  # "objet" = masks exported by older versions


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_rgb(path):
    """Reads an image as float RGB in [0,1]. An RGBA image is composited over black."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Cannot read: {path}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def read_mask(path, size):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        sys.exit(f"Cannot read mask: {path}")
    if m.shape[:2] != size:
        m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


def find_masks(ref, pattern):
    if pattern is None:
        base = ref.replace("_beauty.png", "")
        pattern = base + "_mask_*.png"
    masks = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path).split("_mask_", 1)[-1].rsplit(".", 1)[0]
        masks[name] = path
    return masks


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def grey(rgb):
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255


def equalize_light(grey_gen, grey_ref, mask):
    """Gives the zone of the produced image the same mean and contrast as the reference."""
    r, g = grey_ref[mask], grey_gen[mask]
    std_g = g.std() if g.std() > 1e-4 else 1.0
    out = grey_gen.copy()
    out[mask] = (g - g.mean()) / std_g * r.std() + r.mean()
    return np.clip(out, 0, 1)


def analysis_region(mask, object_inside):
    """Slightly enlarged zone (to include its own edges, e.g. the outline of logo letters),
    but limited to the inside of the object: the outer silhouette touches the background, which is
    necessarily different (transparent background vs generated set) and would bias the measures."""
    enlarged = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    region = enlarged & object_inside
    return region if region.sum() >= MIN_ZONE_PIXELS else mask


def measure_ssim(grey_ref, grey_gen_eq, region):
    _, ssim_map = structural_similarity(
        grey_ref, grey_gen_eq, data_range=1.0, full=True,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    return float(ssim_map[region].mean()), ssim_map


def canny_thresholds(grey_ref, object_mask):
    """Thresholds computed ONCE on the reference and applied to both images:
    otherwise a different background would change the thresholds, hence the detected edges."""
    gx = cv2.Sobel(grey_ref, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(grey_ref, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)[object_mask] * 255
    high = float(np.clip(np.percentile(mag, 90), 40, 200))
    return high * 0.5, high


def edges(grey_img, thresholds):
    u8 = cv2.GaussianBlur((grey_img * 255).astype(np.uint8), (3, 3), 0)
    return cv2.Canny(u8, thresholds[0], thresholds[1]) > 0


def measure_edges(e_ref, e_gen, region):
    """F1 score: precision (no invented detail) and recall (no lost detail)."""
    r, g = e_ref & region, e_gen & region
    # Plain zone (e.g. smooth leather): too few edges to judge, the score would be noise.
    if r.sum() < max(MIN_EDGE_PIXELS, 0.005 * region.sum()):
        return None
    kernel = np.ones((2 * EDGE_TOLERANCE_PX + 1,) * 2, np.uint8)
    r_dil = cv2.dilate(r.astype(np.uint8), kernel).astype(bool)
    g_dil = cv2.dilate(g.astype(np.uint8), kernel).astype(bool)
    precision = (g & r_dil).sum() / max(g.sum(), 1)
    recall = (r & g_dil).sum() / max(r.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def measure_hue(lab_ref, lab_gen, mask):
    """ΔE2000 with the reference lightness (L) imposed: only hue and saturation are judged."""
    lab_gen_same_l = lab_gen.copy()
    lab_gen_same_l[..., 0] = lab_ref[..., 0]
    d = deltaE_ciede2000(lab_ref[mask], lab_gen_same_l[mask])
    return float(np.median(d))


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

def heatmap(gen_rgb, ssim_map, object_mask):
    diff = np.clip((1 - ssim_map) / 0.5, 0, 1)  # 0 = identical, 1 = very different
    diff[~object_mask] = 0
    color = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.cvtColor((gen_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    blend = cv2.addWeighted(base, 0.45, color, 0.55, 0)
    blend[~object_mask] = (base[~object_mask] * 0.35).astype(np.uint8)
    return blend


def edge_image(gen_rgb, e_ref, e_gen, object_mask):
    kernel = np.ones((2 * EDGE_TOLERANCE_PX + 1,) * 2, np.uint8)
    lost = e_ref & ~cv2.dilate(e_gen.astype(np.uint8), kernel).astype(bool) & object_mask
    invented = e_gen & ~cv2.dilate(e_ref.astype(np.uint8), kernel).astype(bool) & object_mask
    background = (grey(gen_rgb) * 255 * 0.5).astype(np.uint8)
    img = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    img[lost] = (40, 40, 255)       # red (BGR): reference detail that disappeared
    img[invented] = (255, 255, 0)   # cyan: detail that does not exist in the reference
    return img


def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def write_html(folder, ref, gen, results, thresholds):
    rows = []
    for z in results["zones"]:
        def cell(value, ok, fmt):
            if value is None:
                return "<td class='na'>n/a</td>"
            return f"<td class='{'ok' if ok else 'ko'}'>{fmt.format(value)}</td>"
        rows.append(
            f"<tr><td><b>{html.escape(z['zone'])}</b></td><td>{z['pixels']}</td>"
            + cell(z["ssim"], z["ok_ssim"], "{:.3f}")
            + cell(z["edges_f1"], z["ok_edges"], "{:.3f}")
            + cell(z["delta_e_chroma"], z["ok_hue"], "{:.2f}")
            + f"<td style='font-size:1.3em'>{'✅' if z['verdict'] else '❌'}</td></tr>"
        )
    verdict = "✅ PASS" if results["overall_verdict"] else "❌ FAIL"
    imgs = [("Reference (3D render)", ref), ("Checked image", gen),
            ("Heatmap (structure)", os.path.join(folder, "heatmap.png")),
            ("Edges: red = lost, cyan = invented", os.path.join(folder, "edges.png"))]
    figures = "".join(
        f"<figure><img src='data:image/png;base64,{to_base64(p)}'><figcaption>{t}</figcaption></figure>"
        for t, p in imgs)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Fidelity QA report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;margin:16px 0}} td,th{{border:1px solid #ccc;padding:6px 12px;text-align:center}}
.ok{{background:#e3f6e3}} .ko{{background:#fbe0e0}} .na{{color:#999}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
figure{{margin:0}} img{{width:100%;border:1px solid #ddd}} figcaption{{font-size:.9em;color:#555}}
</style>
<h1>Fidelity QA: {verdict}</h1>
<p>Reference: <code>{html.escape(ref)}</code><br>Image: <code>{html.escape(gen)}</code></p>
<table><tr><th>Zone</th><th>Pixels</th><th>Structure SSIM<br>≥ {thresholds['ssim_min']}</th>
<th>Edges F1<br>≥ {thresholds['edges_f1_min']}</th><th>Hue ΔE<br>≤ {thresholds['delta_e_chroma_max']}</th><th>Verdict</th></tr>
{''.join(rows)}</table>
<div class="grid">{figures}</div>
<p style="color:#777;font-size:.85em">Starting thresholds, to be calibrated on real examples.</p>
</html>"""
    with open(os.path.join(folder, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def white_balance(gen_rgb, ref_rgb, neutral_zone):
    """Removes the colour cast of the lighting (e.g. warm light) using a zone known to keep its
    colour (ideally a neutral one: white, grey, steel).
    Per-channel gain = reference median / image median, measured on that zone only."""
    gains = [np.median(ref_rgb[..., c][neutral_zone]) / max(np.median(gen_rgb[..., c][neutral_zone]), 1e-4)
             for c in range(3)]
    gains = np.array(gains) / np.mean(gains)  # correct the hue, not the brightness
    print(f"[white balance] R,G,B gains = {np.round(gains, 3).tolist()}")
    return np.clip(gen_rgb * gains, 0, 1)


def check(ref, gen, mask_pattern=None, folder="qa_report", thresholds=None, balance_zone=None):
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    os.makedirs(folder, exist_ok=True)
    ref_rgb, gen_rgb = read_rgb(ref), read_rgb(gen)
    if gen_rgb.shape != ref_rgb.shape:
        print(f"[warning] different sizes {gen_rgb.shape[:2]} vs {ref_rgb.shape[:2]}: "
              "the checked image is resized. Make sure it is aligned.")
        gen_rgb = cv2.resize(gen_rgb, (ref_rgb.shape[1], ref_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    size = ref_rgb.shape[:2]

    paths = find_masks(ref, mask_pattern)
    if not paths:
        sys.exit("No mask found (camXX_mask_*.png). Run export_passes.py first.")
    masks = {name: read_mask(p, size) for name, p in paths.items()}
    object_key = next((k for k in OBJECT_MASK_NAMES if k in masks), None)
    object_mask = masks[object_key] if object_key else np.logical_or.reduce(list(masks.values()))

    if balance_zone:
        if balance_zone not in masks:
            sys.exit(f"Unknown white-balance zone: {balance_zone}")
        gen_rgb = white_balance(gen_rgb, ref_rgb, masks[balance_zone])

    object_inside = cv2.erode(object_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    g_ref, g_gen = grey(ref_rgb), grey(gen_rgb)
    lab_ref, lab_gen = rgb2lab(ref_rgb), rgb2lab(gen_rgb)
    thr_c = canny_thresholds(g_ref, object_mask)
    e_ref = edges(g_ref, thr_c)
    g_gen_object = equalize_light(g_gen, g_ref, object_mask)
    e_gen = edges(g_gen_object, thr_c)

    zones, global_map = [], None
    for name, m in masks.items():
        n = int(m.sum())
        if n < MIN_ZONE_PIXELS:
            print(f"[info] zone '{name}' skipped ({n} px, not visible from this camera)")
            continue
        region = analysis_region(m, object_inside)
        g_eq = equalize_light(g_gen, g_ref, region)
        s, ssim_map = measure_ssim(g_ref, g_eq, region)
        if name == object_key:
            global_map = ssim_map
        f1 = measure_edges(e_ref, edges(g_eq, thr_c), region)
        de = measure_hue(lab_ref, lab_gen, m)
        z = {
            "zone": name, "pixels": n, "ssim": s, "edges_f1": f1, "delta_e_chroma": de,
            "ok_ssim": s >= thresholds["ssim_min"],
            "ok_edges": True if f1 is None else f1 >= thresholds["edges_f1_min"],
            "ok_hue": de <= thresholds["delta_e_chroma_max"],
        }
        z["verdict"] = z["ok_ssim"] and z["ok_edges"] and z["ok_hue"]
        zones.append(z)

    if global_map is None:
        _, global_map = measure_ssim(g_ref, g_gen_object, object_inside)
    cv2.imwrite(os.path.join(folder, "heatmap.png"), heatmap(gen_rgb, global_map, object_mask))
    cv2.imwrite(os.path.join(folder, "edges.png"), edge_image(gen_rgb, e_ref, e_gen, object_mask))

    results = {"reference": ref, "image": gen, "thresholds": thresholds, "zones": zones,
               "overall_verdict": all(z["verdict"] for z in zones)}
    with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_html(folder, ref, gen, results, thresholds)
    return results


def main():
    p = argparse.ArgumentParser(description="Product fidelity check, zone by zone")
    p.add_argument("--ref", required=True)
    p.add_argument("--gen", required=True)
    p.add_argument("--masks", help="glob pattern of the masks (default: camXX_mask_*.png next to --ref)")
    p.add_argument("--out", default="qa_report")
    p.add_argument("--thresholds", help="JSON file overriding the default thresholds")
    p.add_argument("--white-balance", metavar="ZONE",
                   help="compensate the lighting colour cast using this zone (e.g. metal)")
    a = p.parse_args()
    thresholds = None
    if a.thresholds:
        with open(a.thresholds, encoding="utf-8") as f:
            thresholds = json.load(f)
    r = check(a.ref, a.gen, a.masks, a.out, thresholds, a.white_balance)
    print(f"\n{'Zone':<20}{'SSIM':>8}{'Edges':>8}{'Hue ΔE':>9}  Verdict")
    for z in r["zones"]:
        f1 = "n/a" if z["edges_f1"] is None else f"{z['edges_f1']:.3f}"
        print(f"{z['zone']:<20}{z['ssim']:>8.3f}{f1:>8}{z['delta_e_chroma']:>9.2f}  "
              f"{'PASS' if z['verdict'] else 'FAIL'}")
    print(f"\nOverall verdict: {'PASS' if r['overall_verdict'] else 'FAIL'}")
    print(f"Report: {os.path.abspath(os.path.join(a.out, 'report.html'))}")


if __name__ == "__main__":
    main()
