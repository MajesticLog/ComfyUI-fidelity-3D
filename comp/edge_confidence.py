"""
edge_confidence.py — Which edge points can be trusted, and how far does LIGHT alone move them?

PRINCIPLE (inspired by the multi-flash camera of Raskar et al., MERL 2004): take the UNDEFORMED
bag and change only the lighting. A geometric edge stays in place; a shading boundary (a rounded
crease, a shadow) moves or disappears.

TRICK: light adds up. animate_drop.py --passes lights renders, once and for all, 11 "basis
lights" (10 suns + 1 sky) as LINEAR 16-bit PNG. Any lighting is a weighted sum of these renders:
we can make as many lightings as we want without re-running Blender.

STEPS
  1. Draw N random lightings (a dominant sun, a weaker second one, some sky), apply a tone curve,
     put the bag on the set of an AI video and compress to MP4, as in the calibration.
  2. Measure each version with deformation_map.analyze_edges. The edge points are the same from
     one lighting to the next (they come from the 3D passes), so they can be tracked.
  3. TRAINING (the first N lightings): for each point,
        visibility = share of lightings where the edge is found in the image;
        tolerance  = offset caused by light alone (90th percentile).
  4. TEST (the next M lightings, never seen): apply the confidence map and measure the noise that
     REMAINS. This is cross-validation: evaluating on the lightings used to set the tolerances
     would be too optimistic.

Outputs: <out>/NNNN.npz (vis, tol per point), <out>/confidence.json (remaining noise, coverage).

Usage:
    python edge_confidence.py --refs ../renders/drop --background-video V.mp4 --out ../renders/deformation_confidence
"""

import argparse
import json
import os

import cv2
import numpy as np

import deformation_map as dm

RENDER_EXPOSURE = 4.0  # the PNGs were written with an exposure of -2 (× 0.25)


def read_lights(folder, n, w, h):
    names = sorted(os.listdir(os.path.join(folder, "lights")))
    basis = {}
    for name in names:
        imgs = []
        for k in range(n):
            im = cv2.imread(os.path.join(folder, "lights", name, f"{k:04d}.png"), cv2.IMREAD_UNCHANGED)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 65535
            im[..., :3] *= RENDER_EXPOSURE
            imgs.append(im)
        basis[name] = imgs
    return basis


def draw_lighting(names, rng):
    suns = [n for n in names if n != "sky"]
    i, j = rng.choice(len(suns), 2, replace=False)
    weights = {suns[i]: rng.uniform(0.6, 1.4), suns[j]: rng.uniform(0.0, 0.5)}
    if "sky" in names:
        weights["sky"] = rng.uniform(0.15, 0.6)
    return weights


def compose(basis, weights, k, background_bgr, expo):
    rgb = sum(p * basis[n][k][..., :3] for n, p in weights.items())
    alpha = basis[next(iter(weights))][k][..., 3:]
    tone = 1 - np.exp(-rgb * expo)  # soft tone curve (no hard clipping)
    srgb = np.clip(tone, 0, 1) ** (1 / 2.2)
    out = srgb * 255 * alpha + background_bgr.astype(np.float32) * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)  # Blender's PNGs are read as BGR by OpenCV


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", required=True)
    p.add_argument("--background-video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--training", type=int, default=12)
    p.add_argument("--test", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    src, fps = dm.read_video(a.background_video)
    h, w = src[0].shape[:2]
    refs, masks, passes = dm.read_refs(a.refs, w, h)
    n = min(len(refs), len(src))
    basis = read_lights(a.refs, n, w, h)
    backgrounds = [cv2.inpaint(src[k], cv2.dilate(masks[k].astype(np.uint8) * 255, np.ones((25, 25), np.uint8)),
                               5, cv2.INPAINT_TELEA) for k in range(n)]

    N = a.training + a.test
    lightings = [draw_lighting(list(basis), rng) for _ in range(N)]
    R = [[None] * N for _ in range(n)]  # raw residuals per frame and per lighting
    F = [[None] * N for _ in range(n)]  # edge found?
    SIL = [[None] * N for _ in range(n)]
    for j, weights in enumerate(lightings):
        # Exposure set on the middle frame, then kept for the whole video (consistent lighting).
        km = n // 2
        lum = sum(pp * basis[nn][km][..., :3] for nn, pp in weights.items()).mean(2)
        expo = rng.uniform(0.6, 1.1) / max(np.quantile(lum[masks[km]], 0.9), 1e-4)
        video = [compose(basis, weights, k, backgrounds[k], expo) for k in range(n)]
        video = dm.compress(video, fps, os.path.join(a.out, "_tmp.mp4"))
        if j < 3:
            cv2.imwrite(os.path.join(a.out, f"preview_lighting_{j}.jpg"), video[km])
        for k in range(n):
            _, sil = dm.analyze_edges(masks[k], video[k], passes[k])
            if sil is None:
                continue
            R[k][j], F[k][j], SIL[k][j] = sil["residual_raw"], sil["found"], sil
        print(f"lighting {j + 1}/{N} ({'training' if j < a.training else 'TEST'}): "
              + ", ".join(f"{k_} × {v:.2f}" for k_, v in weights.items()))
    if os.path.exists(os.path.join(a.out, "_tmp.mp4")):
        os.remove(os.path.join(a.out, "_tmp.mp4"))

    keys = ("reliable_coverage", "reliable_silhouette_median_px", "reliable_silhouette_p95_px",
            "reliable_silhouette_coverage", "reliable_occlusion_median_px", "reliable_occlusion_p95_px",
            "reliable_occlusion_coverage", "reliable_crease_median_px", "reliable_crease_p95_px",
            "reliable_crease_coverage", "reliable_all_median_px", "reliable_all_p95_px", "reliable_all_share_1px")
    test, raw, tol_types = [], [], {}
    for k in range(n):
        train = [j for j in range(a.training) if R[k][j] is not None]
        if not train:
            continue
        Rk = np.stack([R[k][j] for j in train])
        Fk = np.stack([F[k][j] for j in train])
        vis = Fk.mean(0)
        absR = np.where(Fk, np.abs(Rk), np.nan)
        with np.errstate(all="ignore"):
            tol = np.where(Fk.sum(0) >= 3, np.nanquantile(absR, 0.9, axis=0), np.nan)
        np.savez_compressed(os.path.join(a.out, f"{k:04d}.npz"), vis=vis.astype(np.float32),
                            tol=tol.astype(np.float32), P=SIL[k][train[0]]["P"])
        T = SIL[k][train[0]]["T"]
        for ty, name in ((0, "silhouette"), (1, "occlusion"), (2, "crease")):
            ok = np.isfinite(tol) & (T == ty)
            tol_types.setdefault(name, []).append(tol[ok])
        for j in range(a.training, N):
            sil = SIL[k][j]
            if sil is None:
                continue
            test.append(dm.confidence_metrics(sil, {"vis": vis, "tol": tol}, passes[k]))
            # For comparison: the same noise WITHOUT the confidence map.
            r, t = sil["residual_raw"], sil["found"]
            raw.append({f"raw_{name}_{q}_px": float(f(np.abs(r[t & (sil["T"] == ty)])))
                        for ty, name in ((0, "silhouette"), (1, "occlusion"), (2, "crease"))
                        for q, f in (("median", np.median), ("p95", lambda v: np.quantile(v, 0.95)))
                        if (t & (sil["T"] == ty)).sum() >= 5})

    def mean(rows, c):
        v = [r[c] for r in rows if c in r]
        return float(np.mean(v)) if v else None
    res = {"lightings": [{k_: round(float(v), 3) for k_, v in e.items()} for e in lightings],
           "training": a.training, "test": a.test,
           "remaining_noise_on_test_lightings": {c: mean(test, c) for c in keys},
           "raw_noise_on_test_lightings": {c: mean(raw, c) for c in sorted({c for r in raw for c in r})},
           "median_tolerance_px": {name: float(np.median(np.concatenate(v))) for name, v in tol_types.items() if v},
           "parts_test": {c: mean(test, c) for c in sorted({c for r in test for c in r if c.startswith("part_")})}}
    with open(os.path.join(a.out, "confidence.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
