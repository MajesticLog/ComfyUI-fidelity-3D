"""
lighting_noise_test.py — How much "noise" does LIGHT alone create in the deformation map?

The calibration of deformation_map.py changes hue, contrast, background, noise and compression, but keeps the
lighting of the reference render. Yet the AI relights the bag, and where an edge lands in the image (especially a
rounded crease) depends on the light.

Here: the SAME, UNDEFORMED bag, rendered by Blender under other suns (animate_drop.py --passes beauty --sun ...),
composited on the set of an AI video. Any measured offset is a bias caused by lighting: it is the real floor below
which nothing can be claimed.

Finding: this floor is higher than the synthetic calibration suggested (creases move 5 to 8 px with the light).
This is what led to the per-point confidence map (edge_confidence.py).

Usage:
    blender -b --factory-startup --python ../blender/animate_drop.py -- --import bag.glb --out X/light_sunR \\
        --passes beauty --start 20 --end 27 --sun=-140,45 --sun-strength 5 --hdri-strength 0.4
    python lighting_noise_test.py --refs ../renders/drop --background-video V.mp4 --renders X/light_sunR X/light_sunL \\
        --out ../renders/deformation_calibration/lighting.json
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

import deformation_map as dm

KEYS = ("silhouette_median_px", "silhouette_p95_px", "occlusion_median_px", "occlusion_p95_px",
        "crease_raw_median_px", "crease_raw_p95_px", "crease_median_px", "crease_p95_px", "crease_coverage",
        "edges_median_px", "edges_p95_px", "edges_share_3px")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", required=True)
    p.add_argument("--background-video", required=True)
    p.add_argument("--renders", nargs="+", required=True, help="folders containing beauty/NNNN.png (RGBA)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src, _ = dm.read_video(a.background_video)
    h, w = src[0].shape[:2]
    refs, masks, passes = dm.read_refs(a.refs, w, h)
    res = {}
    for folder in a.renders:
        rows = []
        for f in sorted(glob.glob(os.path.join(folder, "beauty", "*.png"))):
            k = int(os.path.splitext(os.path.basename(f))[0])
            im = cv2.resize(cv2.imread(f, cv2.IMREAD_UNCHANGED), (w, h), interpolation=cv2.INTER_AREA)
            al = im[..., 3:].astype(np.float32) / 255
            background = cv2.inpaint(src[k], cv2.dilate(masks[k].astype(np.uint8) * 255, np.ones((25, 25), np.uint8)),
                                     5, cv2.INPAINT_TELEA)
            comp = (im[..., :3] * al + background * (1 - al)).astype(np.uint8)
            rows.append(dm.analyze_edges(masks[k], comp, passes[k])[0])
        res[os.path.basename(folder.rstrip("/\\"))] = {
            c: float(np.nanmean([r.get(c, np.nan) for r in rows])) for c in KEYS} | {"frames": len(rows)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
