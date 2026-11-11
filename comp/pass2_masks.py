"""
pass2_masks.py — Builds the temporal mask of pass 2 (the colour-change repaint), with a chosen margin.

Recipe of the original mask (drop_mask_after_contact.mp4): bag silhouette (mask_object > 127), dilated by a 9×9
square (= +4 px), black before the contact frame (10). This script generalizes it: --margin 4 gives the original,
--margin 0 the exact silhouette, a negative margin erodes it.

Why: deformation_map.py showed that pass 2 made the edge deformation worse. Hypothesis (confirmed): inside the
grey hole, VACE only knows the bag's shape from the mask contour; if the mask is 4 px wider, it fills a bigger bag.
With --margin 0, the share of deformed edges drops from 30-35 % to 17-19 %.

Requires ffmpeg on the PATH.

Usage:
    python pass2_masks.py --refs ../renders/drop --margin 0 --out <ComfyUI>/input/drop_mask_after_contact_m0.mp4
"""

import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np


def make_mask(refs, k, margin, contact):
    m = (cv2.imread(os.path.join(refs, "mask_object", f"{k:04d}.png"), 0) > 127).astype(np.uint8)
    if k < contact:
        return np.zeros_like(m)
    if margin > 0:
        m = cv2.dilate(m, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    elif margin < 0:
        m = cv2.erode(m, np.ones((-2 * margin + 1, -2 * margin + 1), np.uint8))
    return m * 255


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", required=True)
    p.add_argument("--margin", type=int, required=True, help="px: +4 = original, 0 = exact silhouette")
    p.add_argument("--contact", type=int, default=10)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as d:
        for k in range(a.frames):
            m = make_mask(a.refs, k, a.margin, a.contact)
            cv2.imwrite(os.path.join(d, f"{k:04d}.png"), cv2.merge([m, m, m]))
        # Lossless (qp 0): the graph re-binarizes at 0.5 anyway.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "16", "-i", os.path.join(d, "%04d.png"),
                        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", a.out], check=True)
    print("written:", a.out)


if __name__ == "__main__":
    main()
