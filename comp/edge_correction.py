"""
edge_correction.py — CORRECT the bag's shape in the AI video, by pulling its edges back onto the 3D edges.

STATUS: EXPERIMENT TESTED AND ABANDONED (2026-11-11). Kept for reference; see "Correcting the deformation:
tried, then abandoned" in the README. In short: it passes its synthetic validation, but on real AI videos the
result looks like a Photoshop Liquify brush (lumpy edges, dented strap where leather passes over leather), and
an independent colour-based measurement shows only a 12-19 % reduction of the edge error. Use
deformation_map.py to MEASURE the deformation, and the exact pass-2 mask to PREVENT it.

PRINCIPLE
  deformation_map.py measures, for each 3D edge point, where the matching edge lies in the AI frame: an offset
  r along the normal, RAW, i.e. without the global placement (see offset_to_correct). The AI frame is warped to
  cancel that offset:
    - "backward" field u: output pixel x fetches input pixel x + u(x) (cv2.remap);
    - at edge point P, u(P) = s · N, where s = the offset TO CORRECT: the AI edge is at P + r·N, we want it
      back at P;
    - between points, u is interpolated by NORMALIZED CONVOLUTION: each point pushes around it with a Gaussian
      influence (sigma), divided by the influence density. Far from the edges, u falls back to 0: the set and
      the inside of the bag do not move;
    - two passes: the corrected frame is measured again and the remainder corrected (smoothing slightly
      weakens the 1st pass). The fields are COMPOSED, so the frame is resampled only once (less blur).

  WHICH POINTS: only the RELIABLE silhouette and internal-edge points (confidence map from edge_confidence.py).
  Creases are excluded: their position in the image depends on the light.

  HOW MUCH (--mode):
    certain : s = the CERTAIN offset, i.e. what exceeds the point's tolerance. Cautious: we never correct what
              light alone could explain. But part of the deformation remains;
    full    : s = r. The whole measured offset is corrected, at the risk of "correcting" a lighting effect.

  STABILITY: while the 3D bag is still (frames --still-start to the end), each point's offset is replaced by its
  temporal median (the AI's deformation is stable over time). A single field is applied to the whole range, so
  the image does not wobble.

  SMOOTHING (--smoothing, default 3 px): the correction is smoothed along each edge, since the per-point lighting
  tolerance is noisy and would otherwise make the corrected edge jitter.

  SAFEGUARDS (without them, one aberrant measurement is enough to tear the image):
    1. neighbour consistency (filter_consistency): an AI deformation is SMOOTH along an edge; a point whose
       offset differs by more than 1.5 px from the median of its neighbours is dropped, as are isolated points;
       |s| is capped at 6 px;
    2. anti-fold (anti_fold): the Jacobian determinant of x → x + u(x) must stay above 0.5, otherwise two
       neighbouring pixels would fetch their colour out of order and the image would "fold".

MODES
  validate : on the SYNTHETIC bag (basis lights, unseen test lightings), known bumps are applied and we check that
             the correction brings the part boundaries (ids pass) back to their true place. This is an
             independent check: it does not go through the edge detector of deformation_map.py.
  correct  : applies the correction to a real video and writes the corrected video, a side-by-side and the field.

Usage:
    python edge_correction.py validate --refs ../renders/drop --background-video V.mp4 --confidence ../renders/deformation_confidence --out ...
    python edge_correction.py correct  --refs ../renders/drop --video V.mp4 --confidence ../renders/deformation_confidence --out ...
"""

import argparse
import json
import os

import cv2
import numpy as np

import deformation_map as dm
import edge_confidence as ec


# ---------------------------------------------------------------------------------------------
# The correction field


def offset_to_correct(sil, conf, mode, min_vis=0.5):
    """s per point (nan = unknown / left untouched).

    TARGET = THE 3D CONTOUR ITSELF. We correct the RAW offset d (AI edge at P + d·N), not the residual r left after
    deformation_map's global affine placement. That placement is useful for measuring shape independently of
    position, but it absorbs part of the real deformations (handles...) and, in compensation, creates fictitious
    offsets on correct edges (inner strap edge: d = +0.6 px, r = -1.5 px). Correcting r created bumps there. In
    this pipeline the video is generated from the render, so the right position is known: it is P."""
    r, T = sil["d"], sil["T"]
    tol = np.fmax(conf["tol"], sil["band"])
    ok = sil["found"] & (conf["vis"] >= min_vis) & np.isfinite(conf["tol"]) & (T <= 1)
    if mode == "certain":
        s = np.sign(r) * np.maximum(np.abs(r) - np.nan_to_num(tol, nan=99), 0)
    else:
        s = r.copy()
    return np.where(ok, s, np.nan)


def filter_consistency(P, T, s, radius=8.0, max_dev=1.5, s_max=6.0, min_neighbours=3):
    """Safeguard no. 1. An AI deformation is SMOOTH along an edge: a point whose offset differs by more than
    max_dev px from the median of its neighbours (same edge type, closer than `radius` px) is probably a detection
    error (the detector latched onto another edge, a shadow). It is dropped, as are isolated points. |s| is also
    capped at s_max px."""
    from scipy.spatial import cKDTree
    s = s.copy()
    ok = np.isfinite(s)
    for ty in np.unique(T):
        idx = np.where(ok & (T == ty))[0]
        if len(idx) < min_neighbours + 1:
            s[idx] = np.nan
            continue
        tree = cKDTree(P[idx])
        neigh = tree.query_ball_point(P[idx], radius)
        keep = np.zeros(len(idx), bool)
        for i, v in enumerate(neigh):
            v = [x for x in v if x != i]
            if len(v) >= min_neighbours:
                keep[i] = abs(s[idx[i]] - np.median(s[idx[v]])) <= max_dev
        s[idx[~keep]] = np.nan
    return np.where(np.abs(s) <= s_max, s, np.nan)


def smooth_along_edges(P, N, T, s, sigma=3.0, cos_min=0.5):
    """Against jitter: after the consistency filter, two neighbouring points can still get slightly different
    corrections (the per-point lighting tolerance jumps from 0.2 to 1.1 px between neighbours). An AI deformation is
    smooth along an edge, so each s is replaced by the Gaussian-weighted mean (sigma px) of its neighbours: same edge
    type, and normal pointing the same way (cos > cos_min), so that the two sides of a narrow strap are not mixed.
    The points lie on curves, so this smooths ALONG the contour. sigma = 0: disabled."""
    if sigma <= 0:
        return s
    from scipy.spatial import cKDTree
    s_s = s.copy()
    ok = np.isfinite(s)
    for ty in np.unique(T):
        idx = np.where(ok & (T == ty))[0]
        if len(idx) < 2:
            continue
        tree = cKDTree(P[idx])
        for i, v in enumerate(tree.query_ball_point(P[idx], 3 * sigma)):
            v = np.array(v)
            v = v[(N[idx[v]] @ N[idx[i]]) > cos_min]
            d2 = ((P[idx[v]] - P[idx[i]]) ** 2).sum(1)
            wts = np.exp(-d2 / (2 * sigma ** 2))
            s_s[idx[i]] = (wts * s[idx[v]]).sum() / wts.sum()
    return s_s


def ripple(sil, r_max=3.0):
    """Ripple of an edge: along the silhouette (points in contour order, every 2 px), the second difference
    |r[i-1] - 2 r[i] + r[i+1]| of the measured offset. A straight but shifted edge gives 0; a wavy edge gives a large
    value. Only triplets of found, consecutive points (less than r_max px apart, so on the same contour) are kept."""
    P, r, T, f = sil["P"], sil["residual_raw"], sil["T"], sil["found"]
    i = np.where(T == 0)[0]
    i = i[(i >= 1) & (i < len(P) - 1)]
    ok = (T[i - 1] == 0) & (T[i + 1] == 0) & f[i - 1] & f[i] & f[i + 1]
    ok &= (np.linalg.norm(P[i] - P[i - 1], axis=1) < r_max) & (np.linalg.norm(P[i + 1] - P[i], axis=1) < r_max)
    i = i[ok]
    return np.abs(r[i - 1] - 2 * r[i] + r[i + 1])


def anti_fold(u, min_det=0.5, attempts=8):
    """Safeguard no. 2. The field must not FOLD the image (two output pixels fetching the same input pixel out of
    order, hence tears). Checked with the Jacobian determinant of x → x + u(x): it must stay > min_det. Where it
    does not, the field is attenuated locally."""
    for _ in range(attempts):
        dux = np.gradient(u[..., 0])
        duy = np.gradient(u[..., 1])
        det = (1 + dux[1]) * (1 + duy[0]) - dux[0] * duy[1]
        bad = det < min_det
        if not bad.any():
            break
        a = cv2.GaussianBlur(cv2.dilate(bad.astype(np.float32), np.ones((7, 7), np.uint8)), (0, 0), 3)
        u = u * (1 - 0.5 * np.clip(a / max(a.max(), 1e-6), 0, 1))[..., None]
    return u.astype(np.float32)


def field_from_points(P, N, s, h, w, sigma=4.0):
    """Normalized convolution: u(x) = Σ g(x-Pi) si Ni / max(Σ g(x-Pi), floor)."""
    ok = np.isfinite(s)
    num = np.zeros((h, w, 2), np.float32)
    den = np.zeros((h, w), np.float32)
    q = np.clip(P[ok].round().astype(int), 0, [w - 1, h - 1])
    np.add.at(num, (q[:, 1], q[:, 0]), (s[ok, None] * N[ok]).astype(np.float32))
    np.add.at(den, (q[:, 1], q[:, 0]), 1.0)
    # Points every 2 px along an edge: on the edge, the density is about 1 / (2·√(2π)·σ).
    floor = 0.5 / (2 * np.sqrt(2 * np.pi) * sigma)
    num = np.stack([cv2.GaussianBlur(num[..., c], (0, 0), sigma) for c in range(2)], -1)
    den = cv2.GaussianBlur(den, (0, 0), sigma)
    return (num / np.maximum(den, np.float32(floor))[..., None]).astype(np.float32)


def compose_fields(u1, u2):
    """Applying u1 then u2 (on the already corrected image) = a single field u(x) = u2(x) + u1(x + u2(x))."""
    h, w = u1.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = (gx + u2[..., 0]).astype(np.float32), (gy + u2[..., 1]).astype(np.float32)
    u1_read = np.stack([cv2.remap(np.ascontiguousarray(u1[..., c]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for c in range(2)], -1)
    return u2 + u1_read


def apply_field(img, u, interp=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, (gx + u[..., 0]).astype(np.float32), (gy + u[..., 1]).astype(np.float32), interp,
                     borderMode=cv2.BORDER_REPLICATE)


def measure_offsets(img, mask, passes, conf, mode):
    _, sil = dm.analyze_edges(mask, img, passes)
    if sil is None or conf is None or len(conf["vis"]) != len(sil["P"]):
        return None, None
    return sil, offset_to_correct(sil, conf, mode)


def correct_frames(frames, masks, passes, confs, mode, still_start, correction_passes=2, sigma=4.0, smoothing=3.0):
    """Returns the corrected frames and the fields. Still range: a single field (temporal median)."""
    h, w = frames[0].shape[:2]
    n = len(frames)
    fields = [np.zeros((h, w, 2), np.float32) for _ in range(n)]
    for it in range(correction_passes):
        current = [apply_field(frames[k], fields[k]) for k in range(n)]
        S, sils = [None] * n, [None] * n
        for k in range(n):
            sils[k], S[k] = measure_offsets(current[k], masks[k], passes[k], confs[k], mode)
        still = [k for k in range(still_start, n) if S[k] is not None]
        if still and len({len(S[k]) for k in still}) == 1:
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.stack([S[k] for k in still]), axis=0)
            for k in still:
                S[k] = med
        for k in range(n):
            if S[k] is None:
                continue
            s = filter_consistency(sils[k]["P"], sils[k]["T"], S[k])
            s = smooth_along_edges(sils[k]["P"], sils[k]["N"], sils[k]["T"], s, smoothing)
            u = field_from_points(sils[k]["P"], sils[k]["N"], s, h, w, sigma)
            fields[k] = anti_fold(compose_fields(fields[k], u))
    return [apply_field(frames[k], fields[k]) for k in range(n)], fields


# ---------------------------------------------------------------------------------------------
# Independent check: part boundaries (ids pass)


def boundaries(ids):
    g = cv2.morphologyEx(ids, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return g > 0


def mode_validate(a):
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    src, fps = dm.read_video(a.background_video)
    h, w = src[0].shape[:2]
    refs, masks, passes = dm.read_refs(a.refs, w, h)
    n = min(len(refs), len(src))
    basis = ec.read_lights(a.refs, n, w, h)
    backgrounds = [cv2.inpaint(src[k], cv2.dilate(masks[k].astype(np.uint8) * 255, np.ones((25, 25), np.uint8)),
                               5, cv2.INPAINT_TELEA) for k in range(n)]
    confs = [dm.read_confidence(a.confidence, k) for k in range(n)]
    # Same lightings as edge_confidence.py; only the TEST ones (never seen) are used.
    lightings = [ec.draw_lighting(list(basis), rng) for _ in range(16)]
    expos = []
    for weights in lightings:
        lum = sum(pp * basis[nn][n // 2][..., :3] for nn, pp in weights.items()).mean(2)
        expos.append(rng.uniform(0.6, 1.1) / max(np.quantile(lum[masks[n // 2]], 0.9), 1e-4))
    val_frames = list(range(a.still_start, n, 5))
    rows = []
    for j in range(12, 16):
        for with_bumps in (False, True):
            video, truth, sil_ref = [], [], {}
            for k in range(n):
                im = ec.compose(basis, lightings[j], k, backgrounds[k], expos[j])
                field = np.zeros((h, w, 2), np.float32)
                if k in val_frames:
                    _, sil_ref[k] = dm.analyze_edges(masks[k], im, passes[k])
                if with_bumps and k in val_frames:
                    # Known bumps on RELIABLE points (silhouette and internal edges), 1 to 4 px, outward or
                    # inward, more than 50 px apart.
                    _, sil = dm.analyze_edges(masks[k], im, passes[k])
                    c = confs[k]
                    cand = np.where((c["vis"] >= 0.5) & np.isfinite(c["tol"]) & (sil["T"] <= 1))[0]
                    chosen = []
                    for i in rng.permutation(cand):
                        if all(np.hypot(*(sil["P"][i] - sil["P"][o])) > 50 for o in chosen):
                            chosen.append(i)
                        if len(chosen) == a.bumps:
                            break
                    for i in chosen:
                        amp = rng.uniform(1, 4) * rng.choice([-1, 1])
                        fb, _ = dm.bump(h, w, *sil["P"][i], 12, amp, np.arctan2(sil["N"][i, 1], sil["N"][i, 0]))
                        field += fb
                    im = dm.warp(im, field)
                video.append(im)
                truth.append(field)
            video = dm.compress(video, fps, os.path.join(a.out, "_tmp.mp4"))
            for mode in ("certain", "full"):
                # Validation frames only, each with its own bumps: no temporal median.
                sel = val_frames
                _, fl = correct_frames([video[k] for k in sel], [masks[k] for k in sel], [passes[k] for k in sel],
                                       [confs[k] for k in sel], mode, len(sel), smoothing=a.smoothing)
                fields = dict(zip(sel, fl))
                for k in val_frames:
                    # EXACT residual error: deformed image D(x) = I(x - c(x)), corrected C(x) = D(x + u(x))
                    # = I(x + u(x) - c(x + u(x))). Error e(x) = u(x) - c(x + u(x)); before correction, e = -c.
                    # Read on the true part boundaries (ids pass, silhouette included).
                    c_true, u = truth[k], fields[k]
                    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                    c_u = np.stack([cv2.remap(np.ascontiguousarray(c_true[..., i]), (gx + u[..., 0]).astype(np.float32),
                                              (gy + u[..., 1]).astype(np.float32), cv2.INTER_LINEAR) for i in range(2)], -1)
                    e_after = np.linalg.norm(u - c_u, axis=2)
                    e_before = np.linalg.norm(c_true, axis=2)
                    bd = boundaries(passes[k]["ids"])
                    on_bump = bd & (e_before > 0.5)
                    elsewhere = bd & (e_before < 0.1)
                    row = {"lighting": j, "bumps": with_bumps, "mode": mode, "frame": k,
                           "elsewhere_after_mean": float(e_after[elsewhere].mean()),
                           "elsewhere_after_p95": float(np.quantile(e_after[elsewhere], 0.95)),
                           "elsewhere_share_1px": float((e_after[elsewhere] > 1).mean()),
                           "max_displacement": float(np.linalg.norm(u, axis=2).max())}
                    if on_bump.sum() > 20:
                        row.update({"bump_before_mean": float(e_before[on_bump].mean()),
                                    "bump_after_mean": float(e_after[on_bump].mean()),
                                    "bump_before_p95": float(np.quantile(e_before[on_bump], 0.95)),
                                    "bump_after_p95": float(np.quantile(e_after[on_bump], 0.95))})
                    # At the RELIABLE edge points themselves (what the method claims to correct), along the normal.
                    sr, cf = sil_ref[k], confs[k]
                    rel = (cf["vis"] >= 0.5) & np.isfinite(cf["tol"]) & (sr["T"] <= 1)
                    Pf, Nf, Tf = sr["P"][rel], sr["N"][rel], sr["T"][rel]
                    cP = np.stack([dm.sample_bilinear(np.ascontiguousarray(c_true[..., i]), Pf[:, 0], Pf[:, 1]) for i in range(2)], 1)
                    euP = np.stack([dm.sample_bilinear(np.ascontiguousarray((u - c_u)[..., i]), Pf[:, 0], Pf[:, 1]) for i in range(2)], 1)
                    bef, aft = np.abs((cP * Nf).sum(1)), np.abs((euP * Nf).sum(1))
                    for ty, name in ((0, "sil"), (1, "int")):
                        for lo, hi in ((0, 0.1), (0.5, 1.5), (1.5, 2.5), (2.5, 9)):
                            m = (Tf == ty) & (bef >= lo) & (bef < hi)
                            if m.sum():
                                row[f"pts_{name}_{lo}-{hi}_before"] = float(bef[m].mean())
                                row[f"pts_{name}_{lo}-{hi}_after"] = float(aft[m].mean())
                                row[f"pts_{name}_{lo}-{hi}_n"] = int(m.sum())
                    # EXACT ripple: residual error along the normal, at the silhouette points taken in contour order;
                    # second difference between consecutive points (before / after correction).
                    i0 = np.where(sr["T"] == 0)[0]
                    Ps, Ns = sr["P"][i0], sr["N"][i0]
                    e_n = {name: (np.stack([dm.sample_bilinear(np.ascontiguousarray(f_[..., i]), Ps[:, 0], Ps[:, 1]) for i in range(2)], 1) * Ns).sum(1)
                           for name, f_ in (("before", -c_true), ("after", u - c_u))}
                    cons = (np.linalg.norm(Ps[1:-1] - Ps[:-2], axis=1) < 3) & (np.linalg.norm(Ps[2:] - Ps[1:-1], axis=1) < 3)
                    for name, e in e_n.items():
                        row[f"ripple_sil_{name}"] = float(np.mean(np.abs(e[:-2] - 2 * e[1:-1] + e[2:])[cons]))
                    rows.append(row)
            print(f"lighting {j} bumps={with_bumps} done")
    if os.path.exists(os.path.join(a.out, "_tmp.mp4")):
        os.remove(os.path.join(a.out, "_tmp.mp4"))
    res = {}
    for b in (False, True):
        for mode in ("certain", "full"):
            L = [r for r in rows if r["bumps"] == b and r["mode"] == mode]
            keys = sorted({c for r in L for c in r if c.startswith(("bump_", "elsewhere_", "max_displacement", "pts_", "ripple"))})
            res[f"{'with' if b else 'without'}_bumps_{mode}"] = {
                c: float(np.nanmean([r[c] for r in L if c in r])) for c in keys}
    with open(os.path.join(a.out, "validation.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": res, "detail": rows}, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def mode_correct(a):
    os.makedirs(a.out, exist_ok=True)
    video, fps = dm.read_video(a.video)
    h, w = video[0].shape[:2]
    refs, masks, passes = dm.read_refs(a.refs, w, h)
    n = min(len(video), len(refs))
    confs = [dm.read_confidence(a.confidence, k) for k in range(n)]
    corr, fields = correct_frames(video[:n], masks[:n], passes[:n], confs, a.mode, a.still_start, smoothing=a.smoothing)
    dm.write_video(os.path.join(a.out, "corrected.mp4"), corr, fps)
    side = []
    for k in range(n):
        left = video[k].copy()
        right = corr[k].copy()
        cs, _ = cv2.findContours(masks[k].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for im in (left, right):
            cv2.drawContours(im, cs, -1, (255, 0, 255), 1)
        amp = np.linalg.norm(fields[k], axis=2)
        field_map = cv2.applyColorMap(np.clip(amp / 4 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        side.append(np.hstack([left, right, field_map]))
    dm.write_video(os.path.join(a.out, "before_after_field.mp4"), side, fps)
    np.savez_compressed(os.path.join(a.out, "fields.npz"), fields=np.stack(fields).astype(np.float16))
    amp = [float(np.linalg.norm(c, axis=2).max()) for c in fields]
    # Silhouette ripple, before / after, on the still frames (same detector: indicative only).
    rip = {}
    for name, imgs in (("before", video), ("after", corr)):
        v = [ripple(s_) for k in range(a.still_start, n)
             for s_ in [dm.analyze_edges(masks[k], imgs[k], passes[k])[1]] if s_ is not None]
        rip[f"ripple_sil_{name}_px"] = float(np.mean(np.concatenate(v)))
    print(json.dumps({"max_displacement_px": max(amp), "median_max_displacement_px": float(np.median(amp)), **rip}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="action", required=True)
    v = sp.add_parser("validate")
    v.add_argument("--background-video", required=True)
    v.add_argument("--bumps", type=int, default=8)
    v.add_argument("--seed", type=int, default=2026)
    c = sp.add_parser("correct")
    c.add_argument("--video", required=True)
    c.add_argument("--mode", choices=("certain", "full"), default="certain")
    for s in (v, c):
        s.add_argument("--refs", required=True)
        s.add_argument("--confidence", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--still-start", type=int, default=14)
        s.add_argument("--smoothing", type=float, default=3.0,
                       help="sigma (px) of the smoothing of the correction along the edges, against jitter (0 = none)")
    a = p.parse_args()
    mode_validate(a) if a.action == "validate" else mode_correct(a)


if __name__ == "__main__":
    main()
