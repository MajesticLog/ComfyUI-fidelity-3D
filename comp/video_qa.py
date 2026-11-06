"""
video_qa.py — Measures product drift in an AI video, frame by frame, then corrects it.

IDEA: when the camera is fixed (or its motion is known), the 3D render (beauty + masks) is the EXACT
reference for every frame. A regular AI product video has no such reference: nobody can say how far
the product drifted. Here it is measured.

FOR EACH FRAME, inside the bag mask:
    - SSIM after light equalization        → did the drawing (embroidery, lettering) move?
    - chromatic ΔE (median)                → did the colour drift?
    - displacement (affine ECC alignment)  → did the bag move? (the render is re-aligned before use)
    - temporal stability: SSIM between a frame and the previous one → product "flicker"
Then frequency re-injection (comp/frequency_reinjection.py) is applied to every frame, and
everything is measured again.

OUTPUTS (in --out):
    measures.csv          every value, frame by frame, raw and corrected
    curves.png            SSIM and ΔE over time, raw vs corrected
    corrected.mp4         the corrected video
    side_by_side.mp4      raw | corrected
    summary.json          means, worst values

Usage:
    python video_qa.py --video wan.mp4 --render ../renders/cam05_beauty.png --out ../renders/video_qa
"""

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qa"))
from frequency_reinjection import reinject, restore_chroma  # noqa: E402
from fidelity_qa import equalize_light, measure_ssim, measure_hue  # noqa: E402
from skimage.color import rgb2lab  # noqa: E402


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    cap.release()
    if not frames:
        sys.exit(f"No frame read from {path}")
    return frames, fps


def read_images(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    return [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 for f in files], 16


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for im in frames:
        vw.write(cv2.cvtColor((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


def grey(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255


def measure(img, ref_grey, ref_lab, region):
    g = grey(img)
    ssim, _ = measure_ssim(ref_grey, equalize_light(g, ref_grey, region), region)
    de = measure_hue(ref_lab, rgb2lab(np.clip(img, 0, 1)), region)
    return ssim, de


def align(ref_grey, g, region, W_init):
    """Affine transform (translation, rotation, scale, shear) bringing the render onto the bag in
    the frame, estimated by ECC (maximized correlation) inside the bag mask.
    NB: a first version used phase correlation on images multiplied by the mask. The mask edges,
    identical in both images, dominated the computation and always returned "0 px". That is what had
    hidden the drift."""
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, W = cv2.findTransformECC(cv2.GaussianBlur(ref_grey, (5, 5), 0), cv2.GaussianBlur(g, (5, 5), 0),
                                    W_init.copy(), cv2.MOTION_AFFINE, crit, region.astype(np.uint8), 5)
    except cv2.error:
        W = W_init.copy()
    return W


def max_displacement(W, region):
    ys, xs = np.where(region)
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]],
                       np.float32)
    return float(np.linalg.norm(corners @ W[:, :2].T + W[:, 2] - corners, axis=1).max())


def warp(img, W, w, h, interp=cv2.INTER_LINEAR):
    # ECC gives W such that frame(W·x) ≈ render(x): warpAffine(render, W) therefore places the
    # render in the frame's coordinates.
    return cv2.warpAffine(img, W, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT)


def curves(path, rows, n):
    """Simple chart drawn with OpenCV (no matplotlib dependency)."""
    W, H, M = 1200, 700, 70
    img = np.full((H, W, 3), 250, np.uint8)
    panels = [("SSIM structure (higher = faithful)", "ssim", 0.5, 1.0, 0.80),
              ("Delta E hue (lower = faithful)", "de", 0.0, 12.0, 5.0)]
    ph = (H - 3 * M) // 2
    for i, (title, key, lo, hi, threshold) in enumerate(panels):
        y0 = M + i * (ph + M)
        cv2.rectangle(img, (M, y0), (W - 30, y0 + ph), (200, 200, 200), 1)
        cv2.putText(img, title, (M, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)

        def xy(k, v):
            x = M + int(k / max(n - 1, 1) * (W - 30 - M))
            y = y0 + ph - int((np.clip(v, lo, hi) - lo) / (hi - lo) * ph)
            return x, y
        ys = xy(0, threshold)[1]
        cv2.line(img, (M, ys), (W - 30, ys), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, f"threshold {threshold}", (W - 170, ys - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        for v in (lo, hi):
            cv2.putText(img, f"{v:g}", (20, xy(0, v)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        for suffix, color in (("_raw", (60, 70, 200)), ("_fixed", (60, 150, 60))):
            pts = [xy(k, r[key + suffix]) for k, r in enumerate(rows)]
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.putText(img, "red: raw AI video    green: after re-injection", (M, H - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images", help="folder of numbered PNG frames")
    p.add_argument("--render", required=True, help="RGBA beauty of the camera")
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--chroma", type=float, default=0.5)
    p.add_argument("--no-align", dest="align", action="store_false",
                   help="do not re-align the render on each frame (assumes a perfectly still bag)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames, fps = read_video(a.video) if a.video else read_images(a.images)
    h, w = frames[0].shape[:2]
    beauty = cv2.imread(a.render, cv2.IMREAD_UNCHANGED)
    if abs(beauty.shape[1] / beauty.shape[0] - w / h) > 0.01:
        sys.exit(f"The video framing ({w}x{h}) does not have the render's aspect ratio "
                 f"({beauty.shape[1]}x{beauty.shape[0]}): generate the video at the same (square) format.")
    beauty = cv2.resize(beauty, (w, h), interpolation=cv2.INTER_AREA)
    alpha = beauty[..., 3].astype(np.float32) / 255
    ref_rgb = cv2.cvtColor(beauty[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    # Inside of the bag: 3 px of edge are removed, where the background mixes with the product.
    region = cv2.erode((alpha > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ref_grey = grey(ref_rgb * alpha[..., None])

    rows, fixed, previous = [], [], {}
    W = np.eye(2, 3, dtype=np.float32)
    for k, img in enumerate(frames):
        if a.align:
            # Start from the previous frame's transform: the motion is continuous.
            W = align(ref_grey, grey(img), region, W)
        else:
            W = np.eye(2, 3, dtype=np.float32)
        rgb_k = warp(ref_rgb, W, w, h)
        alpha_k = warp(alpha, W, w, h)
        mask_k = (alpha_k > 0.5).astype(np.float32)
        region_k = cv2.erode((alpha_k > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        grey_k, lab_k = grey(rgb_k * alpha_k[..., None]), rgb2lab(rgb_k)
        corr = reinject(img, rgb_k, mask_k, a.sigma)
        if a.chroma > 0:
            corr = restore_chroma(corr, rgb_k, mask_k, a.chroma)
        fixed.append(corr)
        row = {"frame": k, "displacement_px": max_displacement(W, region)}
        for name, im in (("raw", img), ("fixed", corr)):
            ssim, de = measure(im, grey_k, lab_k, region_k)
            row[f"ssim_{name}"], row[f"de_{name}"] = ssim, de
            g = grey(im)
            if name in previous:
                s, _ = measure_ssim(previous[name], g, region_k)
                row[f"stability_{name}"] = s
            else:
                row[f"stability_{name}"] = None
            previous[name] = g
        rows.append(row)
        print(f"frame {k:3d}  SSIM {row['ssim_raw']:.3f} -> {row['ssim_fixed']:.3f}   "
              f"dE {row['de_raw']:.2f} -> {row['de_fixed']:.2f}   displacement {row['displacement_px']:.2f} px")

    with open(os.path.join(a.out, "measures.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    curves(os.path.join(a.out, "curves.png"), rows, len(rows))
    write_video(os.path.join(a.out, "corrected.mp4"), fixed, fps)
    write_video(os.path.join(a.out, "side_by_side.mp4"),
                [np.hstack([b, c]) for b, c in zip(frames, fixed)], fps)

    def stats(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"mean": float(np.mean(v)), "worst": float(min(v) if "ssim" in key or "stab" in key else max(v))}
    summary = {c: stats(c) for c in ("ssim_raw", "ssim_fixed", "de_raw", "de_fixed",
                                     "stability_raw", "stability_fixed")}
    summary["max_displacement_px"] = max(r["displacement_px"] for r in rows)
    summary["frames"], summary["fps"] = len(rows), fps
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["max_displacement_px"] > 1.5 and not a.align:
        print("Warning: the bag moved by more than 1.5 px; without alignment, re-injection creates doubled edges.")


if __name__ == "__main__":
    main()
