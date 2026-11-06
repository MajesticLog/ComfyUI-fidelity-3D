"""
video_brand_color.py — Enforces the EXACT brand colours in a VACE video without destroying the
transition animated by the AI.

PROBLEM: the AI animates the pink → yellow transition well, but its colours drift (mustard yellow,
salmon pink, hardware still gold). Bluntly replacing the colour with the 3D render's would erase the
AI's transition (the render switches abruptly at the contact frame).

IDEA: keep the AI's "WHERE and WHEN", correct the "WHAT".
For each material (leather, metal), in Lab space and for each pixel of the 3D mask:
    1. Measure the AI's two colours: before (an early frame) and after (a late frame).
    2. Compute the pixel's progress t through the AI's transition: 0 = still "before",
       1 = already "after" (projection of its colour onto the before→after segment).
    3. New colour = blend of the TARGET colours (3D render) with the same t,
       plus an attenuated share of the residual (the AI's fine colour variations).
    4. Lightness L is kept: the AI's light, shadows and material stay intact.
The transition keeps its shape and tempo, but starts from the right pink and lands on the right yellow.

Usage:
    python video_brand_color.py --video vace_final.mp4 --renders ../renders/drop --out ../renders/brand_color
"""

import argparse
import json
import os

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab


def read_video(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    return frames


def write_video(path, frames, fps=16):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for im in frames:
        vw.write(cv2.cvtColor((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


def mask(folder, material, k, erosion=0):
    m = cv2.imread(os.path.join(folder, f"mask_{material}", f"{k:04d}.png"), 0) > 127
    if erosion:
        m = cv2.erode(m.astype(np.uint8), np.ones((erosion, erosion), np.uint8)) > 0
    return m


def render_lab(folder, k):
    b = cv2.imread(os.path.join(folder, "beauty", f"{k:04d}.png"), cv2.IMREAD_UNCHANGED)
    return rgb2lab(cv2.cvtColor(b[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255)


def ab_median(lab, m):
    return np.median(lab[m][:, 1:], axis=0)


def delta_e(lab_ref, lab_img, m):
    a = lab_img[m].copy()
    a[:, 0] = lab_ref[m][:, 0]  # judge only the hue, not the lightness
    return float(np.median(deltaE_ciede2000(lab_ref[m], a)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--renders", required=True, help="output folder of animate_drop.py (beauty/, mask_leather/, mask_metal/)")
    p.add_argument("--before", type=int, default=5, help="frame where the AI shows the 'before' colour")
    p.add_argument("--after", type=int, default=40, help="frame where the AI shows the 'after' colour")
    p.add_argument("--residual", type=float, default=0.5, help="share of the AI's fine colour variations kept")
    p.add_argument("--margin", type=int, default=6, help="px of overflow tolerated around the 3D mask")
    p.add_argument("--tolerance", type=float, default=14.0, help="colour distance (a*b*) to accept a margin pixel")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames = read_video(a.video)
    n = len(frames)
    labs = [rgb2lab(im) for im in frames]

    # AI colours and target colours (3D render), before and after, per material.
    anchors = {}
    for mat in ("leather", "metal"):
        m_before, m_after = mask(a.renders, mat, a.before, 5), mask(a.renders, mat, a.after, 5)
        anchors[mat] = {
            "ai_before": ab_median(labs[a.before], m_before), "ai_after": ab_median(labs[a.after], m_after),
            "target_before": ab_median(render_lab(a.renders, a.before), m_before),
            "target_after": ab_median(render_lab(a.renders, a.after), m_after),
        }
        print(mat, {k: np.round(v, 1).tolist() for k, v in anchors[mat].items()})

    outputs, measures = [], []
    for k in range(n):
        lab = labs[k].copy()
        row = {"frame": k}
        lab_r = render_lab(a.renders, k)
        for mat, r in anchors.items():
            m3d = mask(a.renders, mat, k)
            if not m3d.any():
                continue
            # The generated bag can overflow the 3D mask by a few pixels (a fringe left uncorrected).
            # The mask is enlarged, but inside the margin only pixels whose colour is close to the
            # AI bag's colour are corrected (colour keying), so the set is not tinted.
            seg = r["ai_after"] - r["ai_before"]
            # Margin for leather only: around the small charms, a margin would catch leather.
            margin = a.margin if mat == "leather" else 0
            wide = cv2.dilate(m3d.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
            if mat == "leather":
                wide &= ~cv2.dilate(mask(a.renders, "metal", k).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            ab_w = lab[..., 1:][wide]
            t_w = np.clip(((ab_w - r["ai_before"]) @ seg) / max(seg @ seg, 1e-6), 0, 1)[:, None]
            dist = np.linalg.norm(ab_w - (r["ai_before"] + t_w * seg), axis=1)
            keep = m3d[wide] | ((dist < a.tolerance) if margin else False)
            m = np.zeros_like(m3d)
            m[wide] = keep
            ab = lab[..., 1:][m]
            t = np.clip(((ab - r["ai_before"]) @ seg) / max(seg @ seg, 1e-6), 0, 1)[:, None]
            ai_base = r["ai_before"] + t * seg
            target = r["target_before"] + t * (r["target_after"] - r["target_before"])
            new = target + a.residual * (ab - ai_base)
            # 1 px feather at the mask edge
            feather = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)[m][:, None]
            lab[..., 1:][m] = new * feather + ab * (1 - feather)
            m_eval = mask(a.renders, mat, k, 5)
            if m_eval.any():
                row[f"de_{mat}_raw"] = delta_e(lab_r, labs[k], m_eval)
                row[f"de_{mat}_fixed"] = delta_e(lab_r, lab, m_eval)
                row[f"t_{mat}"] = float(np.median(t))
        outputs.append(np.clip(lab2rgb(lab), 0, 1).astype(np.float32))
        measures.append(row)
        if k % 6 == 0 or k in (9, 10, 11, 12):
            print(k, {c: round(v, 2) for c, v in row.items() if c != "frame"})

    write_video(os.path.join(a.out, "corrected.mp4"), outputs)
    write_video(os.path.join(a.out, "side_by_side.mp4"), [np.hstack([b, c]) for b, c in zip(frames, outputs)])
    with open(os.path.join(a.out, "measures.json"), "w", encoding="utf-8") as f:
        json.dump({"anchors": {m: {k: v.tolist() for k, v in r.items()} for m, r in anchors.items()},
                   "frames": measures}, f, indent=1, ensure_ascii=False)
    # Contact sheet: frames 8, 10, 12, 14, 20, 40; top = raw AI, bottom = corrected.
    ks = [k for k in (8, 10, 12, 14, 20, 40) if k < n]
    top = np.hstack([cv2.resize(frames[k], (320, 320)) for k in ks])
    bottom = np.hstack([cv2.resize(outputs[k], (320, 320)) for k in ks])
    cv2.imwrite(os.path.join(a.out, "contact_sheet.jpg"),
                cv2.cvtColor((np.vstack([top, bottom]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("ok", a.out)


if __name__ == "__main__":
    main()
