"""
light_estimation.py — Recover the AI frame's lighting from the basis lights, to sharpen the edge measurement.

STATUS: EXPERIMENT TESTED, PARTIAL / NEGATIVE RESULT (2026-11-11). Kept for reference; see "Knowing the light:
tried, partial result" in the README. In short: the light is recovered well in luminance, but that does not
predict WHERE the edges fall to the pixel, and on real AI frames the material mismatch (AI vs Blender) biases
the estimate.

IDEA
  The deformation measurement (deformation_map.py) is limited by light: a rounded edge "moves" with the light
  direction, so each edge point gets a lighting tolerance (edge_confidence.py), about 0.7 px on the silhouette
  and 2 px on internal edges. If we KNEW the AI's light, we could predict that move and subtract it.

METHOD
  - Light adds up: the basis-light renders (animate_drop.py --passes lights, linear 16-bit) can be mixed into
    any lighting. ESTIMATE = the non-negative weights whose mix best reproduces the luminance of the leather
    in the frame (non-negative least squares, NNLS). Pixels: leather mask eroded by 4 px (no edges), clipped
    and near-black pixels excluded. The tone curve (light → image) is unknown: several curves are tried
    (gamma 1.0-2.8 and a "filmic" curve) and the best fit is kept.
  - PREDICT = re-render the UNDEFORMED bag under the estimated light, put it on the set, compress it, and
    measure its edges with the same detector: that is the light-induced offset r_pred.
  - NEW TOLERANCE of a point = p90 of |r - r_pred| (instead of |r|) over the training lightings; tested on
    unseen lightings.

MODES
  validate : on fake videos made with the KNOWN lightings of edge_confidence.py (confidence.json), estimate the
             light with a basis (--lights-dir; --leave-one-out removes the true dominant sun), and compare the
             tolerances with and without the estimated light.
  estimate : estimate the light of a real AI frame and write the re-rendered bag next to it.

RESULTS (Lady D-Lite drop, frame 30)
  | Median tolerance         | no light | 11 lights, ideal | 11 lights, true sun removed | 38 lights (--dense-lights) |
  | Silhouette               | 0.75 px  | 0.24 px          | 0.79 px                     | 0.70 px                    |
  | Internal edges           | 1.99 px  | 0.49 px          | 1.34 px                     | 0.95 px                    |
  The 38-light basis recovers the light almost as well as the ideal case (fit residual 4.2 % vs 3.6 %), yet
  the silhouette tolerance barely moves. On real AI frames the fit residual is 10-18 % and the "light" found is
  a mix of 9-10 lights that compensates for the AI's different material (contrast, highlights, hue).

Usage:
    python light_estimation.py validate --refs ../renders/drop --lights-dir ../renders/drop_lights38 --frame 30 \
        --background-video ai.mp4 --confidence ../renders/confidence/confidence.json --out ../reports/light
    python light_estimation.py estimate --refs ../renders/drop --lights-dir ../renders/drop_lights38 --frame 30 \
        --video ai.mp4 --out ../reports/light
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import nnls

import deformation_map as dm
import edge_confidence as ec

CURVES = {f"gamma{g:.1f}": (lambda s, g=g: s ** g) for g in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8)}
CURVES["filmic"] = lambda s: -np.log(np.clip(1 - s ** 2.2, 1e-3, 1))
INVERSES = {f"gamma{g:.1f}": (lambda x, g=g: np.clip(x, 0, None) ** (1 / g)) for g in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8)}
INVERSES["filmic"] = lambda x: (1 - np.exp(-np.clip(x, 0, None))) ** (1 / 2.2)


def luminance(bgr):
    return 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]


def read_lights_frame(folder, k, w, h):
    """Basis lights of ONE frame, as {name: {k: image}} (same indexing as edge_confidence.read_lights)."""
    basis = {}
    for name in sorted(os.listdir(os.path.join(folder, "lights"))):
        im = cv2.imread(os.path.join(folder, "lights", name, f"{k:04d}.png"), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 65535
        im[..., :3] *= ec.RENDER_EXPOSURE
        basis[name] = {k: im}
    return basis


def leather_zone(refs, k, w, h, erosion=4):
    m = cv2.imread(os.path.join(refs, "mask_leather", f"{k:04d}.png"), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) > 127
    return cv2.erode(m.astype(np.uint8), np.ones((2 * erosion + 1,) * 2, np.uint8)) > 0


def estimate(img_bgr, basis, k, zone, n_pixels=6000, seed=0):
    """Returns (weights, tone curve, relative residual)."""
    s = luminance(img_bgr.astype(np.float32) / 255)
    idx = np.flatnonzero(zone & (s < 0.97) & (s > 0.03))
    if len(idx) > n_pixels:
        idx = np.random.default_rng(seed).choice(idx, n_pixels, replace=False)
    names = list(basis)
    A = np.stack([luminance(basis[n][k][..., :3]).ravel()[idx] for n in names], 1)
    best = None
    for curve, f in CURVES.items():
        y = f(s.ravel()[idx])
        wts, res = nnls(A, y)
        rel = res / np.linalg.norm(y)
        if best is None or rel < best[2]:
            best = (dict(zip(names, wts)), curve, rel)
    return best


def predict(basis, k, weights, curve, background_bgr):
    """Re-render the UNDEFORMED bag under the estimated light, in the estimated tone space, on the set."""
    lin = sum(p * basis[n][k][..., :3] for n, p in weights.items() if p > 0)
    alpha = basis[next(iter(basis))][k][..., 3:]
    s = np.clip(INVERSES[curve](lin), 0, 1)
    return np.clip(s * 255 * alpha + background_bgr.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)


def background(frame, mask):
    return cv2.inpaint(frame, cv2.dilate(mask.astype(np.uint8) * 255, np.ones((25, 25), np.uint8)), 5, cv2.INPAINT_TELEA)


def mode_validate(a):
    os.makedirs(a.out, exist_ok=True)
    conf = json.load(open(a.confidence, encoding="utf-8"))
    src, fps = dm.read_video(a.background_video)
    h, w = src[0].shape[:2]
    _, masks, passes = dm.read_refs(a.refs, w, h)
    K = a.frame
    truth_basis = read_lights_frame(a.refs, K, w, h)          # the basis the known lightings were drawn from
    est_basis = read_lights_frame(a.lights_dir or a.refs, K, w, h)
    bg = background(src[K], masks[K])
    zone = leather_zone(a.refs, K, w, h)
    rng = np.random.default_rng(7)
    lightings, n_train = conf["lightings"], conf["training"]
    R, F, RP, FP, fit = [], [], [], [], []
    for j, wts in enumerate(lightings):
        lum = sum(p * truth_basis[n][K][..., :3] for n, p in wts.items()).mean(2)
        expo = rng.uniform(0.6, 1.1) / max(np.quantile(lum[masks[K]], 0.9), 1e-4)
        img = dm.compress([ec.compose(truth_basis, wts, K, bg, expo)], fps, os.path.join(a.out, "_tmp.mp4"))[0]
        _, sil = dm.analyze_edges(masks[K], img, passes[K])
        b = {n: v for n, v in est_basis.items() if not (a.leave_one_out and n == max(wts, key=wts.get))}
        p, curve, rel = estimate(img, b, K, zone)
        pred = dm.compress([predict(b, K, p, curve, bg)], fps, os.path.join(a.out, "_tmp.mp4"))[0]
        _, sil_p = dm.analyze_edges(masks[K], pred, passes[K])
        R.append(sil["residual_raw"]); F.append(sil["found"]); RP.append(sil_p["residual_raw"]); FP.append(sil_p["found"])
        fit.append(rel)
        T = sil["T"]
        print(f"lighting {j + 1}/{len(lightings)}: fit residual {rel:.3f}", flush=True)
    os.remove(os.path.join(a.out, "_tmp.mp4"))
    res = {"frame": K, "fit_residual_mean": float(np.mean(fit))}
    for name, ty in (("silhouette", 0), ("internal", 1), ("crease", 2)):
        res[name] = {}
        for method in ("no_light", "estimated_light"):
            use = method == "estimated_light"
            Ft = np.stack([F[j] & (FP[j] if use else True) for j in range(n_train)])
            Rt = np.stack([R[j] - (RP[j] if use else 0) for j in range(n_train)])
            with np.errstate(all="ignore"):
                tol = np.where(Ft.sum(0) >= 3, np.nanquantile(np.where(Ft, np.abs(Rt), np.nan), 0.9, axis=0), np.nan)
            sel = (T == ty) & (Ft.mean(0) >= 0.5) & np.isfinite(tol)
            if sel.sum() < 5:
                continue
            false_pos = []
            for j in range(n_train, len(lightings)):
                ok = sel & F[j] & (FP[j] if use else True)
                e = np.abs(R[j] - (RP[j] if use else 0))[ok]
                false_pos.append(np.maximum(e - tol[ok], 0) > 1)
            res[name][method] = {"median_tolerance_px": float(np.median(tol[sel])), "p90_tolerance_px": float(np.quantile(tol[sel], 0.9)),
                                 "test_false_positives": float(np.concatenate(false_pos).mean()), "points": int(sel.sum())}
    with open(os.path.join(a.out, "light_validation.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def mode_estimate(a):
    os.makedirs(a.out, exist_ok=True)
    src, _ = dm.read_video(a.video)
    h, w = src[0].shape[:2]
    _, masks, _ = dm.read_refs(a.refs, w, h)
    K = a.frame
    basis = read_lights_frame(a.lights_dir or a.refs, K, w, h)
    p, curve, rel = estimate(src[K], basis, K, leather_zone(a.refs, K, w, h))
    total = sum(p.values())
    res = {"frame": K, "tone_curve": curve, "fit_residual": float(rel),
           "weights": {n: round(float(v / total), 3) for n, v in sorted(p.items(), key=lambda x: -x[1]) if v > 0.02 * total}}
    pred = predict(basis, K, p, curve, background(src[K], masks[K]))
    ys, xs = np.where(masks[K])
    y0, y1, x0, x1 = max(ys.min() - 10, 0), ys.max() + 10, max(xs.min() - 10, 0), xs.max() + 10
    cv2.imwrite(os.path.join(a.out, "ai_vs_relit.jpg"), np.hstack([src[K][y0:y1, x0:x1], pred[y0:y1, x0:x1]]))
    with open(os.path.join(a.out, "light_estimate.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("validate", "estimate"):
        s = sub.add_parser(name)
        s.add_argument("--refs", required=True, help="output folder of animate_drop.py (beauty, masks, passes, lights)")
        s.add_argument("--lights-dir", default=None, help="folder whose lights/ is the ESTIMATION basis (default: --refs)")
        s.add_argument("--frame", type=int, default=30)
        s.add_argument("--out", required=True)
    v = sub.choices["validate"]
    v.add_argument("--background-video", required=True)
    v.add_argument("--confidence", required=True, help="confidence.json written by edge_confidence.py (known lightings)")
    v.add_argument("--leave-one-out", action="store_true", help="remove the true dominant sun from the estimation basis")
    sub.choices["estimate"].add_argument("--video", required=True)
    a = p.parse_args()
    {"validate": mode_validate, "estimate": mode_estimate}[a.mode](a)


if __name__ == "__main__":
    main()
