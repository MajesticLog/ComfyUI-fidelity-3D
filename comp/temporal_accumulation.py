"""
temporal_accumulation.py — Accumulate the edge measurement OVER TIME while the bag is still.

IDEA: in the drop video, the 3D bag no longer moves from frame 14 to frame 48 (35 frames). Its edge
points are then exactly the same from one frame to the next, so for each point we can take the
MEDIAN of its offset over the 35 frames. Noise that changes from frame to frame (compression, AI
flicker, small detection errors) averages out; a STABLE deformation remains.

WHAT IT DOES NOT FIX: a bias that does not change over time. If the AI's lighting is fixed, the
offset it imposes on a rounded crease is the same on all 35 frames: accumulation does not remove
it. This is why the confidence map (visibility, tolerance) is rebuilt IN ACCUMULATED MODE, with the
same basis lights as edge_confidence.py: the accumulated tolerance only keeps the part of the
lighting effect that is a true bias.

RESULT ON OUR DATA (negative, and worth knowing): the AI's deformation turned out to be stable
over time, so accumulation barely changes the measurement. The limit is the lighting bias, not
the frame-to-frame noise.

STEPS
  1. Lightings drawn as in edge_confidence.py (same seed, hence the same ones): 12 training, 4 test.
     For each, the UNDEFORMED bag is composited on the set, compressed, and measured on the still frames.
  2. Two confidence maps, computed on the SAME data:
        "per_frame"   : tolerance = p90 over lightings of |offset of the frame|;
        "accumulated" : tolerance = p90 over lightings of |temporal median of the offset|.
  3. Test on the 4 unseen lightings: remaining noise in both modes.
  4. Sensitivity: KNOWN bumps (1, 2, 4 px), constant over time, are placed on reliable points under
     the test lightings, to see whether they stand out of the noise (threshold = p95 of the mode's
     remaining noise).
  5. Applied to the real videos (--videos): accumulated certain offset, per edge type and per part.

Outputs (--out): lighting_cache.npz, still_confidence.npz (vis, tol, P of the accumulated mode),
                 accumulation.json, videos.md (comparison table).

Usage:
    python temporal_accumulation.py --refs ../renders/drop --background-video V.mp4 \\
        --out ../renders/deformation_accumulation --videos A.mp4 B.mp4 ...
"""

import argparse
import json
import os

import cv2
import numpy as np

import deformation_map as dm
import edge_confidence as ec

TYPES = ((0, "silhouette"), (1, "occlusion"), (2, "crease"))


def measure(video, masks, passes, frames):
    """Raw offset and "found" flag of each point, for each still frame. Returns (R, F, sil_ref)."""
    R, F, ref = [], [], None
    for k in frames:
        _, sil = dm.analyze_edges(masks[k], video[k], passes[k])
        if sil is None:
            R.append(None), F.append(None)
            continue
        if ref is None:
            ref = sil
        if len(sil["P"]) != len(ref["P"]) or np.abs(sil["P"] - ref["P"]).max() > 1e-3:
            raise RuntimeError(f"frame {k}: edge points differ, the bag is not still")
        R.append(sil["residual_raw"]), F.append(sil["found"])
    return R, F, ref


def accumulate(R, F, min_share=0.5):
    """Temporal median of each point's offset, if it is found on at least min_share of the frames."""
    ok = [i for i, r in enumerate(R) if r is not None]
    Rs = np.stack([R[i] for i in ok])
    Fs = np.stack([F[i] for i in ok])
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(Fs, Rs, np.nan), axis=0)
    return np.where(Fs.mean(0) >= min_share, med, np.nan)


def confidence_map(samples):
    """samples: (n_lightings, n_points), nan = not found. → visibility, tolerance (p90 of |offset|)."""
    E = np.asarray(samples)
    seen = np.isfinite(E)
    with np.errstate(all="ignore"):
        tol = np.where(seen.sum(0) >= 3, np.nanquantile(np.abs(E), 0.9, axis=0), np.nan)
    return seen.mean(0), tol


def certain(r, vis, tol, band, min_vis=0.5):
    reliable = np.isfinite(r) & (vis >= min_vis) & np.isfinite(tol)
    rc = np.where(reliable, np.maximum(np.abs(np.nan_to_num(r)) - np.fmax(np.nan_to_num(tol, nan=99), band), 0), np.nan)
    return reliable, rc


def stats(rc, reliable, sel):
    v = rc[reliable & sel]
    if len(v) < 5:
        return {}
    return {"share_1px": float((v > 1).mean()), "p95_px": float(np.quantile(v, 0.95)),
            "median_px": float(np.median(v)), "coverage": float((reliable & sel).sum() / max(sel.sum(), 1))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", required=True)
    p.add_argument("--background-video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, default=14, help="first still frame")
    p.add_argument("--training", type=int, default=12)
    p.add_argument("--test", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bumps-per-video", type=int, default=6)
    p.add_argument("--videos", nargs="*", default=[])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    src, fps = dm.read_video(a.background_video)
    h, w = src[0].shape[:2]
    refs, masks, passes = dm.read_refs(a.refs, w, h)
    n = min(len(refs), len(src))
    still = list(range(a.start, n))
    k0 = still[len(still) // 2]
    basis = ec.read_lights(a.refs, n, w, h)
    backgrounds = [cv2.inpaint(src[k], cv2.dilate(masks[k].astype(np.uint8) * 255, np.ones((25, 25), np.uint8)),
                               5, cv2.INPAINT_TELEA) for k in range(n)]
    tmp = os.path.join(a.out, "_tmp.mp4")

    # 1) Lightings: same draw as edge_confidence.py (lightings first, then one exposure per lighting).
    N = a.training + a.test
    lightings = [ec.draw_lighting(list(basis), rng) for _ in range(N)]
    R_lit, F_lit, sil0 = [], [], None
    cache = os.path.join(a.out, "lighting_cache.npz")
    expos = []
    for j, weights in enumerate(lightings):
        km = n // 2
        lum = sum(pp * basis[nn][km][..., :3] for nn, pp in weights.items()).mean(2)
        expos.append(rng.uniform(0.6, 1.1) / max(np.quantile(lum[masks[km]], 0.9), 1e-4))

    def lighting_video(j, fields=None):
        v = [ec.compose(basis, lightings[j], k, backgrounds[k], expos[j]) for k in range(n)]
        if fields is not None:
            v = [dm.warp(im, fields) if k >= a.start else im for k, im in enumerate(v)]
        return dm.compress(v, fps, tmp)

    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        R_lit, F_lit = list(z["R"]), list(z["F"])
        _, sil0 = dm.analyze_edges(masks[k0], lighting_video(0)[k0], passes[k0])
        print("lightings read back from the cache")
    else:
        for j in range(N):
            R, F, ref = measure(lighting_video(j), masks, passes, still)
            sil0 = sil0 or ref
            R_lit.append(np.stack(R)), F_lit.append(np.stack(F))
            print(f"lighting {j + 1}/{N} measured")
        np.savez_compressed(cache, R=np.array(R_lit), F=np.array(F_lit))
    T, band, P = sil0["T"], sil0["band"], sil0["P"]

    # 2) The two confidence maps.
    train, test = range(a.training), range(a.training, N)
    acc = [accumulate(list(R_lit[j]), list(F_lit[j])) for j in range(N)]
    vis_a, tol_a = confidence_map([acc[j] for j in train])
    per_frame = []  # (vis, tol) for each still frame
    for i in range(len(still)):
        per_frame.append(confidence_map([np.where(F_lit[j][i], R_lit[j][i], np.nan) for j in train]))
    np.savez_compressed(os.path.join(a.out, "still_confidence.npz"), vis=vis_a, tol=tol_a, P=P, T=T, band=band,
                        frames=np.array(still))

    # 3) Remaining noise on the test lightings.
    res = {"still_frames": [still[0], still[-1]], "lightings": len(lightings), "test_noise": {}, "reliable": {},
           "median_tolerance_px": {}, "sensitivity": {}}
    for mode in ("per_frame", "accumulated"):
        vals = {name: [] for _, name in TYPES}
        vals["all"] = []
        for j in test:
            if mode == "accumulated":
                batches = [certain(acc[j], vis_a, tol_a, band)]
            else:
                batches = [certain(np.where(F_lit[j][i], R_lit[j][i], np.nan), *per_frame[i], band)
                           for i in range(len(still))]
            for rel, rc in batches:
                for ty, name in TYPES:
                    vals[name].append(rc[rel & (T == ty)])
                vals["all"].append(rc[rel])
        res["test_noise"][mode] = {name: {"p95_px": float(np.quantile(np.concatenate(v), 0.95)),
                                          "share_1px": float((np.concatenate(v) > 1).mean())}
                                   for name, v in vals.items() if sum(map(len, v)) >= 5}
    rel_a = (vis_a >= 0.5) & np.isfinite(tol_a)
    for ty, name in TYPES:
        sel = T == ty
        rel_i = np.mean([((v >= 0.5) & np.isfinite(t))[sel].mean() for v, t in per_frame])
        res["reliable"][name] = {"per_frame": float(rel_i), "accumulated": float(rel_a[sel].mean())}
        res["median_tolerance_px"][name] = {
            "per_frame": float(np.nanmedian(np.concatenate([t[sel & (v >= 0.5)] for v, t in per_frame]))),
            "accumulated": float(np.nanmedian(tol_a[sel & rel_a]))}
    print(json.dumps({k: res[k] for k in ("test_noise", "reliable", "median_tolerance_px")}, indent=1, ensure_ascii=False))

    # 4) Sensitivity: known bumps, constant over time, on reliable points (in BOTH modes).
    thresholds = {m: res["test_noise"][m]["all"]["p95_px"] for m in res["test_noise"]}
    rel_i_all = np.all([(v >= 0.5) & np.isfinite(t) for v, t in per_frame], axis=0)
    for ty, name in TYPES:
        candidates = np.where(rel_a & rel_i_all & (T == ty))[0]
        if len(candidates) < 5:
            continue
        res["sensitivity"][name] = []
        for amp in (1.0, 2.0, 4.0):
            det = {"per_frame": [], "accumulated": []}
            measured = {"per_frame": [], "accumulated": []}
            for j in test:
                # Several bumps per video, more than 60 px apart.
                chosen = []
                for c in rng.permutation(candidates):
                    if all(np.hypot(*(P[c] - P[o])) > 60 for o in chosen):
                        chosen.append(c)
                    if len(chosen) == a.bumps_per_video:
                        break
                field = np.zeros((h, w, 2), np.float32)
                weight = np.zeros((h, w), np.float32)
                N_ = sil0["N"]
                for c in chosen:
                    fb, g = dm.bump(h, w, P[c, 0], P[c, 1], 15, amp, np.arctan2(N_[c, 1], N_[c, 0]))
                    field += fb
                    weight = np.maximum(weight, g)
                R, F, _ = measure(lighting_video(j, field), masks, passes, still)
                wts = dm.sample_bilinear(weight, P[:, 0], P[:, 1])
                for c in chosen:
                    core = (wts > 0.8) & (T == ty) & (np.hypot(*(P - P[c]).T) < 15)
                    ra, rca = certain(accumulate(R, F), vis_a, tol_a, band)
                    v = rca[core & ra]
                    if len(v):
                        measured["accumulated"].append(float(np.median(v)))
                        det["accumulated"].append(float(np.median(v)) > thresholds["accumulated"])
                    for i in range(len(still)):
                        ri, rci = certain(np.where(F[i], R[i], np.nan), *per_frame[i], band)
                        v = rci[core & ri]
                        if len(v):
                            measured["per_frame"].append(float(np.median(v)))
                            det["per_frame"].append(float(np.median(v)) > thresholds["per_frame"])
            row = {"amplitude_px": amp}
            for m in det:
                if det[m]:
                    row[m] = {"detected": float(np.mean(det[m])), "median_certain_offset_px": float(np.median(measured[m])),
                              "trials": len(det[m])}
            res["sensitivity"][name].append(row)
            print(name, json.dumps(row, ensure_ascii=False))
    if os.path.exists(tmp):
        os.remove(tmp)

    # 5) The real videos.
    parts = dm.parts_of_points(sil0, passes[k0])
    groups = [("all", T >= 0)] + [(name, T == ty) for ty, name in TYPES]
    if parts is not None:
        groups += [(pc, parts == pc) for pc in sorted(set(parts)) if pc != "?"]
    res["videos"] = {}
    for path in a.videos:
        v, _ = dm.read_video(path)
        R, F, _ = measure(v, masks, passes, still)
        ra, rca = certain(accumulate(R, F), vis_a, tol_a, band)
        res["videos"][os.path.basename(path)] = {g: stats(rca, ra, sel) for g, sel in groups}
        np.savez_compressed(os.path.join(a.out, os.path.splitext(os.path.basename(path))[0] + "_offset.npz"),
                            accumulated_offset=accumulate(R, F), certain_offset=rca, reliable=ra, P=P, T=T)
        print("video measured:", os.path.basename(path))
    with open(os.path.join(a.out, "accumulation.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    if res["videos"]:
        names = list(res["videos"])
        lines = ["| Group | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for g, _ in groups:
            cells = []
            for nv in names:
                s = res["videos"][nv].get(g, {})
                cells.append(f"{s['share_1px']:.0%} / {s['p95_px']:.1f}" if s else "–")
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        with open(os.path.join(a.out, "videos.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
