"""
compare_measurements.py — Puts several measurements of deformation_map.py (measure mode with --confidence) side by side.

Mean per video, over a chosen frame range (by default from the contact, frame 10, since pass 2 changes nothing
before), of:
    share_1px = share of RELIABLE points that exceed their tolerance by more than 1 px (floor: about 1.5 %)
    p95       = 95th percentile of the certain offset (px)
for all edges and for each part of the model.

The part names below are those of the "Lady D-Lite" model used in this project (ids.json); edit PARTS for yours.

Usage:
    python compare_measurements.py --start 10 FOLDER1 FOLDER2 ...
"""

import argparse
import csv
import os

import numpy as np

PARTS = [("all", "All edges"), ("handle", "Handles, charm, strap"), ("SideFabeic", "Piping"),
         ("pattern.001", "Body"), ("fabric", "Strap lining"), ("stitches.001", "Stitching"),
         ("Metal.001", "Metal")]


def read(folder, start, end):
    with open(os.path.join(folder, "measurements.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if start <= int(r["frame"]) <= end]
    res = {}
    for key, _ in PARTS:
        prefix = "reliable_all" if key == "all" else f"part_{key}"
        for q in ("share_1px", "p95_px"):
            v = [float(r[f"{prefix}_{q}"]) for r in rows if r.get(f"{prefix}_{q}") not in (None, "", "nan")]
            res[(key, q)] = float(np.mean(v)) if v else float("nan")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+")
    p.add_argument("--start", type=int, default=10)
    p.add_argument("--end", type=int, default=10000)
    a = p.parse_args()
    names = [os.path.basename(os.path.normpath(d)) for d in a.folders]
    R = [read(d, a.start, a.end) for d in a.folders]
    print(f"Frames {a.start} to {a.end if a.end < 10000 else 'end'} — share > 1 px / p95 (px), reliable points\n")
    print("| Part | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for key, label in PARTS:
        print(f"| {label} | " + " | ".join(f"{r[(key, 'share_1px')]:.0%} / {r[(key, 'p95_px')]:.1f}" for r in R) + " |")


if __name__ == "__main__":
    main()
