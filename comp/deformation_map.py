"""
deformation_map.py — Measure, frame by frame, how far the AI DEFORMED the product.

IDEA: in the video demo, Blender drives the video. For every frame we therefore have the EXACT
3D render of the same pose, seen by the same camera. Any shape difference between that render and
the generated frame is a deformation invented by the AI. No pose estimation needed: it is known.

TWO COMPLEMENTARY MEASUREMENTS

1. EDGES (main measurement): at every point of the 3D contour of the bag (outer edges AND holes,
   such as the inside of the strap), we look for the nearest edge in the AI frame, along the
   perpendicular to the contour (± 12 px). The offset, in pixels, says how far the silhouette
   moved at that spot. It is an oriented "chamfer distance".
   Safeguards:
     - the AI edge must be parallel to the 3D contour (otherwise it belongs to something else);
     - the "inside" of the AI edge must look like the bag (colour distance < BAG_COLOR_MAX_DIST), and
       the "outside" less so. This rejects the edges of cast shadows on the table, often more
       contrasted than the bag.
   It does not depend on texture: it works even on plain leather.
   Three edge types are measured:
     type 0  SILHOUETTE   contour of the object mask;
     type 1  OCCLUSION    internal edges where one part passes IN FRONT of another (strap in front
                          of the body, front handle, charms), found as JUMPS in the depth pass;
     type 2  CREASE       folds and box corners: no depth jump, but the surface ORIENTATION changes
                          abruptly, found in the normal pass.

2. INTERIOR (secondary measurement): optical flow between the render and the AI frame, on a
   "locally normalized" version of both (only contours and patterns remain, not light or colour).
   We only keep anchor points (corners, patterns) and, on edges, only the perpendicular component
   (the "aperture problem"). It covers little surface (smooth leather has nothing to track) but
   looks at the metal, the handle, the stitching.

PLACEMENT AND DEFORMATION: in both cases we first remove a global affine fit (translation,
rotation, scale, shear): that is the "placement" of the bag. What remains (the RESIDUAL) is the
local deformation: a wider strap, a swelling handle.

TWO MODES
    measure      compares an AI video with the 3D renders, frame by frame
    calibrate    builds fake videos from the renders: light, colour, background, noise and
                 compression changed, BUT geometry intact (→ noise floor), then with "bumps" of
                 known size (→ sensitivity). Without this step, nobody knows what a number means.

Expected render folder (--refs), as written by blender/animate_drop.py:
    beauty/NNNN.png (RGBA)   depth/   mask_object/   mask_metal/   normal/ (optional)
    ids/ + ids.json (optional, per-part statistics)

Usage:
    python deformation_map.py calibrate --refs ../renders/drop --background-video X.mp4 --out ../renders/deformation_calibration
    python deformation_map.py measure --video X.mp4 --refs ../renders/drop \\
        --calibration ../renders/deformation_calibration/calibration.json --out ../renders/deformation/X
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

# Maximum colour distance (OpenCV 8-bit Lab: L, a, b on 0-255) between the inside of an AI edge and the bag colour
# read 5 px inside the 3D contour. Chosen on frame 30 of the original-mask video: at 45, false detections on the
# cast shadow (cyan table / navy shadow boundary) disappear, while silhouette coverage only drops from 84 to 82 %.
# None = previous behaviour (relative test only).
BAG_COLOR_MAX_DIST = 45.0

# ---------------------------------------------------------------------------------------------
# Reading / writing


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)  # BGR uint8
    cap.release()
    if not frames:
        sys.exit(f"No frame read from {path}")
    return frames, fps


def read_refs(folder, w, h):
    """Per-frame beauty renders (RGBA): returns the BGR images, the bag masks and the extra passes
    (depth = disparity, white = near; metal mask; normals; part ids) when they exist, else None."""
    files = sorted(glob.glob(os.path.join(folder, "beauty", "*.png")))
    if not files:
        sys.exit(f"No renders in {folder}/beauty")
    bgr, masks, passes = [], [], []
    part_names = {}
    if os.path.exists(os.path.join(folder, "ids.json")):
        with open(os.path.join(folder, "ids.json"), encoding="utf-8") as fj:
            part_names = {int(k): v for k, v in json.load(fj).items()}
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        bgr.append(im[..., :3])
        if im.shape[2] == 4:
            masks.append(im[..., 3] > 127)
        else:
            m = cv2.imread(os.path.join(folder, "mask_object", os.path.basename(f)), cv2.IMREAD_GRAYSCALE)
            masks.append(cv2.resize(m, (w, h)) > 127)
        fd = os.path.join(folder, "depth", os.path.basename(f))
        d = cv2.imread(fd, cv2.IMREAD_UNCHANGED) if os.path.exists(fd) else None
        if d is not None and d.ndim == 3:
            d = d[..., 0]
        if d is not None:
            d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            fm = os.path.join(folder, "mask_metal", os.path.basename(f))
            mm = cv2.imread(fm, cv2.IMREAD_GRAYSCALE) if os.path.exists(fm) else None
            fn = os.path.join(folder, "normal", os.path.basename(f))
            nn = cv2.imread(fn, cv2.IMREAD_COLOR) if os.path.exists(fn) else None
            fi = os.path.join(folder, "ids", os.path.basename(f))
            ii = cv2.imread(fi, cv2.IMREAD_GRAYSCALE) if os.path.exists(fi) else None
            d = {"depth": d, "metal": None if mm is None else cv2.resize(mm, (w, h)) > 127,
                 "ids": None if ii is None else cv2.resize(ii, (w, h), interpolation=cv2.INTER_NEAREST),
                 "part_names": part_names,
                 "normal": None if nn is None else cv2.resize(nn, (w, h), interpolation=cv2.INTER_AREA)}
        passes.append(d)
    return bgr, masks, passes


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for im in frames:
        vw.write(im)
    vw.release()


def write_csv(path, rows):
    keys = sorted({c for r in rows for c in r}, key=lambda c: (c != "frame", c))
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def placement_px(A, mask):
    """Largest displacement of the 4 corners of the bag's bounding box by the affine part A (3×2)."""
    ys, xs = np.nonzero(mask)
    corners = np.array([[xs.min(), ys.min(), 1], [xs.max(), ys.min(), 1],
                        [xs.min(), ys.max(), 1], [xs.max(), ys.max(), 1]], np.float64)
    return float(np.linalg.norm(corners @ A, axis=1).max())


# ---------------------------------------------------------------------------------------------
# Measurement 1: the edges


def ref_contours(mask, step=2):
    """Points of the 3D contour (outer + holes), every `step` px, with their OUTWARD normal."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    P, N = [], []
    h, w = mask.shape
    for c in cs:
        c = c[:, 0, :].astype(np.float32)
        if len(c) < 40:
            continue
        t = np.roll(c, -3, 0) - np.roll(c, 3, 0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
        n = np.stack([t[:, 1], -t[:, 0]], 1)
        q = np.clip((c + 3 * n).round().astype(int), 0, [w - 1, h - 1])
        n[mask[q[:, 1], q[:, 0]]] *= -1  # if +3 px lands inside the bag, the normal pointed inward
        P.append(c[::step])
        N.append(n[::step])
    if not P:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.concatenate(P), np.concatenate(N)


def sample_bilinear(img, X, Y):
    """Bilinear read of `img` (2D) at float positions X, Y (arrays of the same shape)."""
    h, w = img.shape
    X = np.clip(X, 0, w - 1.001)
    Y = np.clip(Y, 0, h - 1.001)
    x0, y0 = X.astype(int), Y.astype(int)
    fx, fy = X - x0, Y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def occlusion_contours(passes, mask, min_jump=4.0, margin=3, min_size=8, metal_margin=4):
    """Internal OCCLUSION edges: where one part of the bag passes IN FRONT of another (the strap in
    front of the body, the front handle, the charms). The object mask cannot see them; the depth
    pass can: depth JUMPS there.

    How a jump is told apart from a merely slanted surface (seen at a grazing angle, depth also
    changes fast): the depth difference across the edge (at ±3 px) is compared with the slopes
    measured on each side (between 3 and 6 px). A regular slope gives a "jump" equal to the sum of
    the slopes; a real step gives a jump much larger than the slopes.

    Creases (a fold, the corner of the bag's box) have NO depth jump: they are not found here
    (see crease_contours, which reads the normal pass).

    Returns the points (sub-pixel) and the normal pointing from the FRONT object to the one behind,
    i.e. the same convention as the "outward" normal of the silhouette.

    `passes` is the dict returned by read_refs: {"depth": disparity, "metal": metal mask, ...}.
    Edges closer than `metal_margin` px to the metal are dropped: rings and clasp are small, full of
    close edges and reflections, and the measurement latched onto the wrong edge there."""
    metal = passes.get("metal")
    d = cv2.GaussianBlur(passes["depth"].astype(np.float32), (0, 0), 0.7)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8
    g = np.hypot(gx, gy)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)
    if metal is not None:
        k = 2 * metal_margin + 1
        inner &= ~cv2.dilate(metal.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    ys, xs = np.nonzero(inner & (g > min_jump / 3))
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    X, Y = xs.astype(np.float32), ys.astype(np.float32)
    ux, uy = gx[ys, xs] / g[ys, xs], gy[ys, xs] / g[ys, xs]  # towards NEAR (increasing disparity)

    def D(t, img=d):
        return sample_bilinear(img, X + t * ux, Y + t * uy)
    g0, gp, gm = g[ys, xs], D(1, g), D(-1, g)
    jump = D(3) - D(-3)
    slopes = np.abs(D(3) - D(6)) + np.abs(D(-3) - D(-6))
    behind_in_bag = sample_bilinear(mask.astype(np.float32), X - 5 * ux, Y - 5 * uy) > 0.5
    ok = (g0 >= gp) & (g0 >= gm) & (jump > min_jump) & (jump > 1.5 * slopes) & behind_in_bag
    # Small isolated fragments = depth noise: removed.
    img = np.zeros(mask.shape, np.uint8)
    img[ys[ok], xs[ok]] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    ok &= st[lab[ys, xs], cv2.CC_STAT_AREA] >= min_size
    # Sub-pixel position: vertex of the parabola through |∇d| at -1, 0, +1 px.
    den = gm - 2 * g0 + gp
    off = np.clip(np.where(np.abs(den) > 1e-6, 0.5 * (gm - gp) / np.where(np.abs(den) > 1e-6, den, 1), 0), -0.5, 0.5)
    P = np.stack([X + off * ux, Y + off * uy], 1)[ok]
    N = np.stack([-ux, -uy], 1)[ok]
    return P.astype(np.float32), N.astype(np.float32)


def crease_contours(passes, mask, other_P, min_angle=25.0, margin=3, min_size=8, metal_margin=4, edge_guard=4):
    """CREASES: a fold, the corner of the bag's box. No depth jump, but the surface ORIENTATION
    changes abruptly. We read the normal pass (camera space, rendered by
    animate_drop.py --passes normal) and reason as for jumps:
      - direction u of the strongest change (structure tensor of the 3 channels of the normal);
      - angle between the normals at ±3 px along u: it must exceed `min_angle`;
      - a merely CURVED surface (the round handle) also turns, but steadily: we require the angle
        "across" to be > 1.5 × the sum of the angles measured on each side (3 to 6 px).
    Points close to an already known edge (silhouette, occlusion) or to the metal are dropped.
    Normal: u, with no notion of front/back (a crease has no "object" side).

    TOLERANCE BAND: a bag's crease is ROUNDED. The orientation turns over about ten px, and the
    light/dark line seen in the image can fall anywhere in that rounding depending on the LIGHT
    (checked: same undeformed bag, two suns → offsets of 5 to 8 px). We therefore measure the
    half-width of the rounding (between 10 % and 90 % of the total rotation, over ±10 px) and only
    the offset that goes BEYOND it counts as deformation. These half-widths are returned too."""
    nb = passes.get("normal") if passes else None
    if nb is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    n = cv2.GaussianBlur(nb.astype(np.float32) / 255 * 2 - 1, (0, 0), 0.7)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
    jxx = jyy = jxy = 0
    for c in range(3):
        gx = cv2.Sobel(n[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(n[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8
        jxx, jyy, jxy = jxx + gx * gx, jyy + gy * gy, jxy + gx * gy
    tr, det = jxx + jyy, jxx * jyy - jxy ** 2
    l1 = tr / 2 + np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    strength = np.sqrt(l1)  # change of the normal per pixel (roughly in radians)
    zone = cv2.erode(mask.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)
    known = np.zeros(mask.shape, np.uint8)
    if len(other_P):
        known[np.clip(other_P[:, 1].round().astype(int), 0, mask.shape[0] - 1),
              np.clip(other_P[:, 0].round().astype(int), 0, mask.shape[1] - 1)] = 1
    if passes.get("metal") is not None:
        known |= passes["metal"].astype(np.uint8)
    known = cv2.dilate(known, np.ones((2 * edge_guard + 1,) * 2, np.uint8)).astype(bool)
    ys, xs = np.nonzero(zone & ~known & (strength > np.radians(min_angle) / 6))
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    X, Y = xs.astype(np.float32), ys.astype(np.float32)
    ux, uy = np.cos(ang[ys, xs]), np.sin(ang[ys, xs])

    def Nt(t):
        return np.stack([sample_bilinear(n[..., c], X + t * ux, Y + t * uy) for c in range(3)], 1)

    def angle(a, b):
        return np.degrees(np.arccos(np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6), -1, 1)))
    f0, fp, fm = strength[ys, xs], sample_bilinear(strength, X + ux, Y + uy), sample_bilinear(strength, X - ux, Y - uy)
    across = angle(Nt(3), Nt(-3))
    sides = angle(Nt(3), Nt(6)) + angle(Nt(-3), Nt(-6))
    both_in_bag = (sample_bilinear(mask.astype(np.float32), X + 6 * ux, Y + 6 * uy) > 0.5) & \
                  (sample_bilinear(mask.astype(np.float32), X - 6 * ux, Y - 6 * uy) > 0.5)
    ok = (f0 >= fp) & (f0 >= fm) & (across > min_angle) & (across > 1.5 * sides) & both_in_bag
    img = np.zeros(mask.shape, np.uint8)
    img[ys[ok], xs[ok]] = 1
    _, lab, st, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    ok &= st[lab[ys, xs], cv2.CC_STAT_AREA] >= min_size
    den = fm - 2 * f0 + fp
    off = np.clip(np.where(np.abs(den) > 1e-6, 0.5 * (fm - fp) / np.where(np.abs(den) > 1e-6, den, 1), 0), -0.5, 0.5)
    P = np.stack([X + off * ux, Y + off * uy], 1)[ok]
    N = np.stack([ux, uy], 1)[ok]
    # Half-width of the rounding: rotation of the normal from t = -10 px, between 10 % and 90 % of the total.
    ts = np.arange(-10, 10.5, 0.5)
    X, Y, ux, uy = X[ok], Y[ok], ux[ok], uy[ok]
    n0 = np.stack([sample_bilinear(n[..., c], X - 10 * ux, Y - 10 * uy) for c in range(3)], 1)
    prof = np.stack([angle(np.stack([sample_bilinear(n[..., c], X + t * ux, Y + t * uy) for c in range(3)], 1), n0)
                     for t in ts], 1)
    tot = prof[:, -1:] + 1e-6
    t10 = ts[(prof < 0.1 * tot).sum(1).clip(0, len(ts) - 1)]
    t90 = ts[(prof < 0.9 * tot).sum(1).clip(0, len(ts) - 1)]
    half = np.clip((t90 - t10) / 2, 0.5, 10)
    return P.astype(np.float32), N.astype(np.float32), half.astype(np.float32)


def analyze_edges(mask, ai_bgr, passes=None, R=12, cos_min=0.7, rel_threshold=0.25, bag_color_max_dist="default"):
    """Edge measurement. Type 0 = silhouette (mask), type 1 = internal occlusion (depth),
    type 2 = crease (normal pass).
    The placement (global affine) is fitted on the SILHOUETTE only, so that numbers stay comparable
    with a silhouette-only measurement; the other edges are then read with that same placement."""
    P, N = ref_contours(mask)
    T = np.zeros(len(P), np.int8)
    if passes is not None:
        P2, N2 = occlusion_contours(passes, mask)
        P, N, T = np.concatenate([P, P2]), np.concatenate([N, N2]), np.concatenate([T, np.ones(len(P2), np.int8)])
        P3, N3, half3 = crease_contours(passes, mask, P)
        band = np.concatenate([np.zeros(len(P), np.float32), half3])
        P, N, T = np.concatenate([P, P3]), np.concatenate([N, N3]), np.concatenate([T, np.full(len(P3), 2, np.int8)])
    else:
        band = np.zeros(len(P), np.float32)
    h, w = mask.shape
    if (T == 0).sum() < 20:
        return {}, None
    g = cv2.GaussianBlur(cv2.cvtColor(ai_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    lab = cv2.cvtColor(cv2.GaussianBlur(ai_bgr, (0, 0), 1.5), cv2.COLOR_BGR2LAB).astype(np.float32)

    def read(img, X, Y):
        return img[np.clip(Y.round().astype(int), 0, h - 1), np.clip(X.round().astype(int), 0, w - 1)]

    ts = np.arange(-R, R + 0.5, 0.5)
    X = P[:, None, 0] + N[:, None, 0] * ts[None]
    Y = P[:, None, 1] + N[:, None, 1] * ts[None]
    # ADAPTIVE search radius: if another 3D edge (silhouette or occlusion) crosses the perpendicular
    # at |t| px, we do not search beyond |t|/2. Otherwise, near a junction (the handle meeting the
    # body, an attachment), the measurement latches onto the NEIGHBOURING edge.
    edge_map = np.zeros((h, w), np.uint8)
    edge_map[np.clip(P[:, 1].round().astype(int), 0, h - 1), np.clip(P[:, 0].round().astype(int), 0, w - 1)] = 1
    if passes is not None and passes.get("metal") is not None:
        mt = passes["metal"].astype(np.uint8)
        edge_map |= cv2.morphologyEx(mt, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))  # metal contour
    edge_map = cv2.dilate(edge_map, np.ones((2, 2), np.uint8))
    other = (read(edge_map, X, Y) > 0) & (np.abs(ts)[None] > 3)
    dist = np.where(other, np.abs(ts)[None], np.inf).min(1)
    r_eff = np.minimum(R, dist / 2)
    if passes is None:
        r_eff[:] = R  # without depth: silhouette-only behaviour unchanged (comparable calibration)
    in_radius = np.abs(ts)[None] <= r_eff[:, None]
    M = read(mag, X, Y)
    parallel = np.abs(read(gx, X, Y) * N[:, None, 0] + read(gy, X, Y) * N[:, None, 1]) / (M + 1e-6) > cos_min
    # Inside / outside test: the bag's colour is read 5 px inside the 3D contour.
    bag = read(lab, P[:, 0] - 5 * N[:, 0], P[:, 1] - 5 * N[:, 1])[:, None]
    inside = read(lab, X - 2.5 * N[:, None, 0], Y - 2.5 * N[:, None, 1])
    outside = read(lab, X + 2.5 * N[:, None, 0], Y + 2.5 * N[:, None, 1])
    d_inside = np.linalg.norm(inside - bag, axis=2)
    bag_side = d_inside < np.linalg.norm(outside - bag, axis=2)
    # ABSOLUTE condition: the inside must LOOK LIKE the bag, not merely be less far from it than the outside.
    # Otherwise a cyan table / navy shadow boundary passed the test (cyan is "closer" to yellow than navy):
    # false detections on the cast shadow, and bumps in the edge correction.
    if bag_color_max_dist == "default":
        bag_color_max_dist = BAG_COLOR_MAX_DIST
    if bag_color_max_dist is not None:
        bag_side &= d_inside < bag_color_max_dist
    # Internal edge: front and back are often the same leather. If, in the AI frame, both sides have
    # almost the same colour, the inside/outside test means nothing → ignored.
    behind = read(lab, P[:, 0] + 5 * N[:, 0], P[:, 1] + 5 * N[:, 1])
    indistinct = ((T == 1) & (np.linalg.norm(bag[:, 0] - behind, axis=1) < 8)) | (T == 2)  # crease: same material on both sides
    bag_side |= indistinct[:, None]
    score = M * parallel * bag_side * in_radius * np.exp(-0.5 * (ts[None] / R) ** 2)
    j = score.argmax(1)
    found = (score[np.arange(len(P)), j] > rel_threshold * np.quantile(mag[mask], 0.99)) & (r_eff >= 3)
    d = ts[j]
    # Placement: affine A such that n·(A^T [x y 1]) ≈ d (only the normal component is visible).
    Xh = np.c_[P, np.ones(len(P))]
    Mx = np.c_[Xh * N[:, :1], Xh * N[:, 1:2]]
    sil = found & (T == 0)
    keep = sil.copy()
    if keep.sum() < 20:
        return {"silhouette_coverage": float(sil.sum() / max((T == 0).sum(), 1))}, None
    for _ in range(2):
        a, *_ = np.linalg.lstsq(Mx[keep], d[keep], rcond=None)
        r = d - Mx @ a
        keep = sil & (np.abs(r) <= np.quantile(np.abs(r[sil]), 0.9))
    A = np.stack([a[:3], a[3:]], 1)
    meas = {"placement_px": placement_px(A, mask)}
    # "Out-of-band" offset: for a crease, only what goes beyond the rounding counts (0 for other edges).
    r_ob = np.sign(r) * np.maximum(np.abs(r) - band, 0)
    for name, sel, rr in (("silhouette", T == 0, r), ("occlusion", T == 1, r), ("crease_raw", T == 2, r),
                          ("crease", T == 2, r_ob), ("edges", T >= 0, r_ob)):
        if sel.sum() == 0:
            continue
        v = np.abs(rr[found & sel])
        meas[f"{name}_coverage"] = float((found & sel).sum() / sel.sum())
        if len(v) >= 5:
            meas.update({f"{name}_median_px": float(np.median(v)), f"{name}_p95_px": float(np.quantile(v, 0.95)),
                         f"{name}_share_3px": float((v > 3).mean())})
    if (T == 2).any():
        meas["crease_half_width_px"] = float(np.median(band[T == 2]))
    return meas, {"P": P, "N": N, "T": T, "d": d, "found": found, "residual": r_ob, "residual_raw": r, "band": band}


# ---------------------------------------------------------------------------------------------
# Per-edge-point confidence (basis lights, see edge_confidence.py) and part label


def parts_of_points(sil, passes):
    """Name of the model part (ids pass) of each edge point, read 2 px on the object side
    ("front" side for an occlusion). Mixed values (on edges) are snapped to the nearest id."""
    if not passes or passes.get("ids") is None or not passes.get("part_names"):
        return None
    ids, names = passes["ids"], passes["part_names"]
    h, w = ids.shape
    Q = (sil["P"] - 2 * sil["N"]).round().astype(int)
    v = ids[np.clip(Q[:, 1], 0, h - 1), np.clip(Q[:, 0], 0, w - 1)].astype(int)
    keys = np.array(sorted(names))
    nearest = keys[np.abs(v[:, None] - keys[None]).argmin(1)]
    nearest[v < keys.min() / 2] = -1  # fell into the background
    return np.array([names.get(int(c), "?") for c in nearest])


def confidence_metrics(sil, conf, passes, min_vis=0.5):
    """Applies one frame's confidence map: a point only counts if it is VISIBLE under at least
    `min_vis` of the training lights, and only the offset that EXCEEDS its own tolerance (how far
    light alone moves this point, p90) counts as deformation."""
    if sil is None or conf is None or len(conf["vis"]) != len(sil["P"]):
        return {}
    r, T = sil["residual_raw"], sil["T"]
    tol = np.fmax(conf["tol"], sil["band"])
    reliable = sil["found"] & (conf["vis"] >= min_vis) & np.isfinite(conf["tol"])
    rc = np.sign(r) * np.maximum(np.abs(r) - np.nan_to_num(tol, nan=99), 0)
    sil["reliable"], sil["residual_certain"] = reliable, rc
    meas = {"reliable_coverage": float(reliable.mean())}
    groups = [("silhouette", T == 0), ("occlusion", T == 1), ("crease", T == 2), ("all", T >= 0)]
    parts = parts_of_points(sil, passes)
    if parts is not None:
        sil["part"] = parts
        groups += [(f"part_{p}", parts == p) for p in sorted(set(parts)) if p not in ("?",)]
    for name, sel in groups:
        v = np.abs(rc[reliable & sel])
        key = name if name.startswith("part_") else f"reliable_{name}"
        if sel.sum():
            meas[f"{key}_coverage"] = float((reliable & sel).sum() / sel.sum())
        if len(v) >= 5:
            meas.update({f"{key}_median_px": float(np.median(v)), f"{key}_p95_px": float(np.quantile(v, 0.95)),
                         f"{key}_share_1px": float((v > 1).mean())})
    return meas


def read_confidence(folder, k):
    f = os.path.join(folder, f"{k:04d}.npz") if folder else None
    if not f or not os.path.exists(f):
        return None
    z = np.load(f)
    return {"vis": z["vis"], "tol": z["tol"]}


# ---------------------------------------------------------------------------------------------
# Measurement 2: the interior (optical flow on structure)


def structure(bgr, sigma=4.0, floor=0.06):
    """Local contrast normalization: (g - local mean) / local standard deviation.
    `floor` avoids amplifying noise and compression blocks in smooth areas
    (with 0.02, plain leather turned into a noise "pattern", and the flow followed it)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    mu = cv2.GaussianBlur(g, (0, 0), sigma)
    sd = np.sqrt(cv2.GaussianBlur((g - mu) ** 2, (0, 0), sigma)) + floor
    n = np.clip((g - mu) / sd, -3, 3)
    return ((n + 3) / 6 * 255).astype(np.uint8)


def anchors(ref_bgr, mask, q_edge=0.5, q_corner=0.9):
    """Structure tensor of the render: two eigenvalues l1 ≥ l2 per point.
        large l2            → CORNER or pattern: motion visible in every direction
        large l1, small l2  → EDGE: only the perpendicular motion is visible
        both small          → SMOOTH area: nothing measurable, ignored."""
    g = cv2.GaussianBlur(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, (0, 0), 1)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), 2) for v in (gx * gx, gy * gy, gx * gy))
    tr, det = jxx + jyy, jxx * jyy - jxy ** 2
    d = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + d, tr / 2 - d
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    corner = l2 > np.quantile(l2[mask], q_corner)
    edge = (l1 > np.quantile(l1[mask], q_edge)) & ~corner
    return corner, edge, np.cos(ang), np.sin(ang)


_DIS = None


def flow(a, b):
    """DIS optical flow (OpenCV): a(x) ≈ b(x + flow(x))."""
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS.calc(a, b, None)


def sample_field(field, f):
    h, w = f.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(field, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def fit_affine(f, zone):
    """Robust affine fit of the flow (least squares, then without the 15 % farthest points)."""
    ys, xs = np.nonzero(zone)
    if len(xs) < 50:
        return None
    X = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64)
    U = f[ys, xs].astype(np.float64)
    keep = np.ones(len(xs), bool)
    for _ in range(2):
        A, *_ = np.linalg.lstsq(X[keep], U[keep], rcond=None)
        err = np.linalg.norm(X @ A - U, axis=1)
        keep = err <= np.quantile(err, 0.85)
    return A


def analyze_interior(ref_bgr, ai_bgr, mask, consistency_threshold=1.0):
    h, w = mask.shape
    sr, si = structure(ref_bgr), structure(ai_bgr)
    forward, backward = flow(sr, si), flow(si, sr)
    # Confidence: the round trip must come back to the start point within 1 px.
    inconsistency = np.linalg.norm(forward + sample_field(backward, forward), axis=2)
    zone = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    corner, edge, nx, ny = anchors(ref_bgr, zone)
    reliable = zone & (inconsistency < consistency_threshold) & (corner | edge)
    meas = {"interior_coverage": float(reliable.sum() / max(zone.sum(), 1))}
    A = fit_affine(forward, reliable)
    if A is None:
        return meas, np.full((h, w), np.nan, np.float32)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    residual = forward - (np.stack([gx, gy, np.ones_like(gx)], -1) @ A).astype(np.float32)
    defo = np.where(corner, np.linalg.norm(residual, axis=2), np.abs(residual[..., 0] * nx + residual[..., 1] * ny))
    v = defo[reliable]
    meas.update({"interior_median_px": float(np.median(v)), "interior_p95_px": float(np.quantile(v, 0.95))})
    return meas, np.where(reliable, defo, np.nan).astype(np.float32)


def analyze(ref_bgr, ai_bgr, mask, passes=None):
    m1, sil = analyze_edges(mask, ai_bgr, passes)
    m2, interior_map = analyze_interior(ref_bgr, ai_bgr, mask)
    return {**m1, **m2}, sil, interior_map


# ---------------------------------------------------------------------------------------------
# Visualization (OpenCV only: no matplotlib needed)


def draw(ai_bgr, sil, interior_map, vmax=6.0):
    """Darkened AI frame + interior flow (inferno palette) + 3D contour coloured by the edge offset
    (turbo palette, blue = 0, red = vmax px). Beyond 3 px, a white line joins the 3D contour to the
    edge found in the AI frame. Grey = contour not found.
    Markers: circles = silhouette, squares = occlusion, diamonds = crease."""
    out = (ai_bgr.astype(np.float32) * 0.5).astype(np.uint8)
    if interior_map is not None:
        ok = ~np.isnan(interior_map)
        c = cv2.applyColorMap((np.clip(np.nan_to_num(interior_map) / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        out[ok] = (0.3 * out[ok] + 0.7 * c[ok]).astype(np.uint8)
    if sil is not None:
        for p, n, d, t, r, ty in zip(sil["P"], sil["N"], sil["d"], sil.get("reliable", sil["found"]),
                                     sil.get("residual_certain", sil["residual"]), sil["T"]):
            pi = tuple(int(v) for v in p.round())
            if not t:
                cv2.circle(out, pi, 1, (130, 130, 130), -1)
                continue
            col = cv2.applyColorMap(np.uint8([[min(abs(r) / vmax, 1) * 255]]), cv2.COLORMAP_TURBO)[0, 0].tolist()
            if abs(r) > 3:
                cv2.line(out, pi, tuple(int(v) for v in (p + n * d).round()), (255, 255, 255), 1, cv2.LINE_AA)
            if ty == 0:
                cv2.circle(out, pi, 2, col, -1)  # silhouette: circles
            elif ty == 2:
                cv2.drawMarker(out, pi, col, cv2.MARKER_DIAMOND, 4, 1)  # crease: diamonds
            else:
                cv2.rectangle(out, (pi[0] - 1, pi[1] - 1), (pi[0] + 1, pi[1] + 1), col, -1)  # occlusion: squares
    h = out.shape[0]
    bar = cv2.applyColorMap(np.linspace(0, 255, 200).astype(np.uint8)[None].repeat(10, 0), cv2.COLORMAP_TURBO)
    out[h - 30:h - 20, 20:220] = bar
    for x, txt in ((16, "0"), (190, f"{vmax:g} px")):
        cv2.putText(out, txt, (x, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def plot_curves(path, rows, floors):
    W, H, M = 1200, 1000, 70
    img = np.full((H, W, 3), 250, np.uint8)
    n = len(rows)
    panels = [("Edges: offset from the 3D contour after placement (px)",
               [("silhouette_p95_px", (60, 70, 200), "p95"), ("silhouette_median_px", (60, 150, 60), "median"),
                ("occlusion_p95_px", (200, 80, 160), "p95 occl."), ("occlusion_median_px", (40, 160, 200), "med. occl."),
                ("crease_p95_px", (0, 140, 255), "p95 crease")],
               0, 10, floors.get("silhouette_p95_px")),
              ("Interior: optical flow on anchor points (px)",
               [("interior_p95_px", (60, 70, 200), "p95"), ("interior_median_px", (60, 150, 60), "median")],
               0, 6, floors.get("interior_p95_px")),
              ("Placement: global offset of the bag (px)", [("placement_px", (180, 110, 40), "corners")],
               0, 12, floors.get("placement_px"))]
    ph = (H - 4 * M) // 3
    for i, (title, series, lo, hi, floor) in enumerate(panels):
        y0 = M + i * (ph + M)
        cv2.rectangle(img, (M, y0), (W - 30, y0 + ph), (200, 200, 200), 1)
        cv2.putText(img, title, (M, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)

        def xy(k, v):
            return (M + int(k / max(n - 1, 1) * (W - 30 - M)),
                    y0 + ph - int((np.clip(v, lo, hi) - lo) / (hi - lo) * ph))
        for v in (lo, hi):
            cv2.putText(img, f"{v:g}", (25, xy(0, v)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        if floor is not None:
            yp = xy(0, floor)[1]
            cv2.line(img, (M, yp), (W - 30, yp), (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(img, f"noise floor (p95) {floor:.2f}", (W - 290, yp - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
        for j, (key, colour, label) in enumerate(series):
            pts = [xy(k, r[key]) for k, r in enumerate(rows) if key in r]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, colour, 2, cv2.LINE_AA)
            cv2.putText(img, label, (M + 10 + 120 * j, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.putText(img, "horizontal axis: frame number", (M, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------------------------
# "measure" mode


def mode_measure(a):
    os.makedirs(a.out, exist_ok=True)
    ai, fps = read_video(a.video)
    h, w = ai[0].shape[:2]
    refs, masks, passes = read_refs(a.refs, w, h)
    n = min(len(ai), len(refs))
    if len(ai) != len(refs):
        print(f"⚠️ {len(ai)} AI frames vs {len(refs)} renders: comparing the first {n}.")
    floors = {}
    if a.calibration and os.path.exists(a.calibration):
        with open(a.calibration, encoding="utf-8") as f:
            floors = {c: v["p95_over_frames"] for c, v in json.load(f)["noise_floor"].items()
                      if isinstance(v, dict)}
    rows, views = [], []
    for k in range(n):
        meas, sil, interior_map = analyze(refs[k], ai[k], masks[k], passes[k])
        meas.update(confidence_metrics(sil, read_confidence(a.confidence, k), passes[k]))
        rows.append({"frame": k, **meas})
        views.append(np.hstack([ai[k], draw(ai[k], sil, interior_map, a.vmax)]))
        print(f"frame {k:3d}  silhouette med {meas.get('silhouette_median_px', np.nan):4.2f} "
              f"p95 {meas.get('silhouette_p95_px', np.nan):5.2f} px (contour found "
              f"{meas.get('silhouette_coverage', 0):.0%})   occlusion med {meas.get('occlusion_median_px', np.nan):4.2f} "
              f"p95 {meas.get('occlusion_p95_px', np.nan):5.2f} px ({meas.get('occlusion_coverage', 0):.0%})   crease med "
              f"{meas.get('crease_median_px', np.nan):4.2f} p95 {meas.get('crease_p95_px', np.nan):5.2f} px   "
              f"interior p95 {meas.get('interior_p95_px', np.nan):4.2f} px   "
              f"placement {meas.get('placement_px', np.nan):5.2f} px")
    write_csv(os.path.join(a.out, "measurements.csv"), rows)
    plot_curves(os.path.join(a.out, "curves.png"), rows, floors)
    write_video(os.path.join(a.out, "deformation_map.mp4"), views, fps)
    valid = [r for r in rows if "silhouette_p95_px" in r]
    worst = sorted(sorted(valid, key=lambda r: -r["silhouette_p95_px"])[:4], key=lambda r: r["frame"])
    cv2.imwrite(os.path.join(a.out, "worst_frames.jpg"), np.vstack([
        cv2.putText(views[r["frame"]].copy(), f"frame {r['frame']}  silhouette p95 {r['silhouette_p95_px']:.1f} px",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) for r in worst]))

    def stats(c):
        v = np.array([r[c] for r in rows if c in r])
        if len(v) == 0:
            return None
        return {"mean": float(v.mean()), "max": float(v.max()),
                "frame_max": int([r["frame"] for r in rows if c in r][int(v.argmax())])}
    summary = {c: stats(c) for c in ("silhouette_median_px", "silhouette_p95_px", "silhouette_share_3px",
                                     "silhouette_coverage", "occlusion_median_px", "occlusion_p95_px",
                                     "occlusion_share_3px", "occlusion_coverage", "crease_median_px",
                                     "crease_p95_px", "crease_share_3px", "crease_coverage", "edges_p95_px",
                                     "edges_share_3px", "interior_p95_px", "interior_coverage", "placement_px")}
    for c in sorted({c for r in rows for c in r if c.startswith(("reliable_", "part_"))}):
        summary[c] = stats(c)
    summary["noise_floor_p95"] = floors
    summary["video"], summary["frames"] = os.path.basename(a.video), n
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------
# "calibrate" mode: fake videos built from the renders


def degrade(ref_bgr, mask, background_bgr, rng):
    """Changes everything the AI changes EXCEPT the geometry: colour, light, background, noise, blur."""
    im = ref_bgr.astype(np.float32) / 255
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(40, 120)) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1.2, 2.0), 0, 1)
    im = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0, 1) ** rng.uniform(0.6, 1.5)
    im = im * np.linspace(rng.uniform(0.6, 1.0), rng.uniform(1.0, 1.4), im.shape[0])[:, None, None]
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)[..., None]
    im = im * a + background_bgr.astype(np.float32) / 255 * (1 - a)
    im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.3, 0.9)) + rng.normal(0, 0.015, im.shape)
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)


def bump(h, w, cx, cy, radius, amplitude, angle):
    """Known local displacement field: a Gaussian "bump" of `amplitude` px."""
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * radius ** 2))
    return np.stack([np.cos(angle) * amplitude * g, np.sin(angle) * amplitude * g], -1).astype(np.float32), g


def warp(img, field):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx - field[..., 0], gy - field[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compress(frames, fps, path):
    write_video(path, frames, fps)
    return read_video(path)[0]


def mode_calibrate(a):
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1312)
    bg_src, fps = read_video(a.background_video)
    h, w = bg_src[0].shape[:2]
    refs, masks, passes = read_refs(a.refs, w, h)
    n = min(len(refs), len(bg_src))
    # Realistic background: an AI frame with the bag erased (OpenCV inpainting).
    backgrounds = [cv2.inpaint(bg_src[k], cv2.dilate(masks[k].astype(np.uint8) * 255, np.ones((25, 25), np.uint8)),
                               5, cv2.INPAINT_TELEA) for k in range(n)]

    # 1) Noise floor: no deformation.
    fakes = compress([degrade(refs[k], masks[k], backgrounds[k], rng) for k in range(n)], fps,
                     os.path.join(a.out, "fake_no_deformation.mp4"))
    rows = [{"frame": k, **analyze(refs[k], fakes[k], masks[k], passes[k])[0]} for k in range(n)]
    write_csv(os.path.join(a.out, "noise_floor.csv"), rows)
    noise = {}
    for c in ("silhouette_median_px", "silhouette_p95_px", "occlusion_median_px", "occlusion_p95_px",
              "crease_median_px", "crease_p95_px", "crease_coverage",
              "interior_p95_px", "placement_px", "silhouette_coverage", "occlusion_coverage",
              "interior_coverage"):
        v = np.array([r[c] for r in rows if c in r])
        if len(v) == 0:
            continue
        noise[c] = {"mean": float(v.mean()), "p95_over_frames": float(np.quantile(v, 0.95))}
    print("Noise floor:", json.dumps(noise, indent=2, ensure_ascii=False))

    # 2) Sensitivity: known bumps, placed ON an edge, along its normal. One series per edge type:
    #    0 = silhouette, 1 = internal occlusion edge (depth), 2 = crease (normals).
    sensitivity = {}
    for ty, name in ((0, "silhouette"), (1, "occlusion"), (2, "crease")):
        key = f"{name}_p95_px"
        if key not in noise:
            continue
        sensitivity[name] = []
        for amp in (0.5, 1.0, 2.0, 4.0):
            measured = []
            for k in range(0, n, 2):
                if ty == 0:
                    P, N = ref_contours(masks[k])
                elif ty == 1:
                    P, N = occlusion_contours(passes[k], masks[k])
                else:
                    P0, _ = ref_contours(masks[k])
                    P1, _ = occlusion_contours(passes[k], masks[k])
                    P, N, _ = crease_contours(passes[k], masks[k], np.concatenate([P0, P1]))
                if len(P) < 20:
                    continue
                i = rng.integers(len(P))
                field, weight = bump(h, w, P[i, 0], P[i, 1], 15, amp, np.arctan2(N[i, 1], N[i, 0]))
                fake = degrade(warp(refs[k], field), warp(masks[k].astype(np.float32), field) > 0.5,
                               backgrounds[k], rng)
                fake = compress([fake], fps, os.path.join(a.out, "_tmp.mp4"))[0]
                _, sil = analyze_edges(masks[k], fake, passes[k])
                if sil is None:
                    continue
                wts = sample_bilinear(weight, sil["P"][:, 0], sil["P"][:, 1])
                core = sil["found"] & (wts > 0.8) & (sil["T"] == ty)
                if core.sum() >= 2:
                    measured.append(float(np.median(sil["residual_raw"][core])))
            m = np.array(measured)
            if len(m) == 0:
                continue
            threshold = noise[key]["mean"]
            sensitivity[name].append({"true_amplitude_px": amp, "trials": len(m), "measured_median_px": float(np.median(m)),
                                      "std_px": float(m.std()),
                                      "share_above_noise_p95": float((m > threshold).mean())})
            print(f"bump of {amp} px on {name}: measured {np.median(m):.2f} ± {m.std():.2f} px "
                  f"({len(m)} trials), above the noise p95 ({threshold:.2f}): {(m > threshold).mean():.0%}")
    if os.path.exists(os.path.join(a.out, "_tmp.mp4")):
        os.remove(os.path.join(a.out, "_tmp.mp4"))
    with open(os.path.join(a.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"noise_floor": noise, "sensitivity": sensitivity}, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="mode", required=True)
    m = sp.add_parser("measure")
    m.add_argument("--video", required=True)
    m.add_argument("--refs", required=True, help="folder containing beauty/ (RGBA, one image per frame)")
    m.add_argument("--out", required=True)
    m.add_argument("--calibration", help="calibration.json, to draw the noise floor")
    m.add_argument("--vmax", type=float, default=6.0, help="px mapped to the red end of the scale")
    m.add_argument("--confidence", help="folder of confidence maps (NNNN.npz) written by edge_confidence.py")
    c = sp.add_parser("calibrate")
    c.add_argument("--refs", required=True)
    c.add_argument("--background-video", required=True, help="an AI video whose set is reused")
    c.add_argument("--out", required=True)
    a = p.parse_args()
    mode_measure(a) if a.mode == "measure" else mode_calibrate(a)


if __name__ == "__main__":
    main()
