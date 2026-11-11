"""
temporal_noise.py — How much of the deformation measured frame by frame is just FLICKER?

On the frames where the bag is still, each video is measured two ways:
  - PER FRAME: certain offset of each frame, then statistics;
  - ACCUMULATED: temporal median of each point's offset, then certain offset.
Both use the SAME confidence map (still_confidence.npz, from temporal_accumulation.py), so the only
difference is the noise that changes from frame to frame.

This noise is also measured directly: for each reliable point, the median absolute deviation from
its own temporal median (MAD). Reference: the same computation on the undeformed bag under a
synthetic lighting (lighting cache).

Usage:
    python temporal_noise.py --refs ../renders/drop --folder ../renders/deformation_accumulation V1.mp4 V2.mp4 ...
"""

import argparse
import os

import numpy as np

import deformation_map as dm
import temporal_accumulation as ta


def temporal_mad(R, F, reliable):
    Rs = np.stack(R)
    Fs = np.stack(F)
    X = np.where(Fs, Rs, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
    return float(np.nanmedian(mad[reliable]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", required=True)
    p.add_argument("--folder", required=True, help="output folder of temporal_accumulation.py")
    p.add_argument("videos", nargs="+")
    a = p.parse_args()
    z = np.load(os.path.join(a.folder, "still_confidence.npz"))
    vis, tol, T, band, still = z["vis"], z["tol"], z["T"], z["band"], list(z["frames"])
    c = np.load(os.path.join(a.folder, "lighting_cache.npz"))
    rel0 = (vis >= 0.5) & np.isfinite(tol)
    mads_synth = [temporal_mad(list(c["R"][j]), list(c["F"][j]), rel0) for j in range(len(c["R"]))]
    print(f"Undeformed bag, synthetic lightings: median temporal MAD {np.median(mads_synth):.2f} px\n")
    lines = ["| Video | Temporal MAD (px) | Per frame: share > 1 px / p95 | Accumulated: share > 1 px / p95 |",
             "|---|---|---|---|"]
    refs = masks = passes = None
    for path in a.videos:
        v, _ = dm.read_video(path)
        if refs is None:
            h, w = v[0].shape[:2]
            refs, masks, passes = dm.read_refs(a.refs, w, h)
        R, F, _ = ta.measure(v, masks, passes, still)
        vals = []
        for r, f in zip(R, F):
            rel, rc = ta.certain(np.where(f, r, np.nan), vis, tol, band)
            vals.append(rc[rel])
        vi = np.concatenate(vals)
        ra, rca = ta.certain(ta.accumulate(R, F), vis, tol, band)
        va = rca[ra]
        lines.append(f"| {os.path.basename(path)} | {temporal_mad(R, F, rel0):.2f} | {(vi > 1).mean():.0%} / "
                     f"{np.quantile(vi, 0.95):.1f} | {(va > 1).mean():.0%} / {np.quantile(va, 0.95):.1f} |")
        print(lines[-1])
    with open(os.path.join(a.folder, "temporal_noise.md"), "w", encoding="utf-8") as f:
        f.write(f"Temporal MAD of the undeformed bag (synthetic): {np.median(mads_synth):.2f} px\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
