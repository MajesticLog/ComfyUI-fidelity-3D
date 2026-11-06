"""
frequency_reinjection.py — Puts the REAL product material back into an AI-harmonized image,
without losing the light the AI added.

PRINCIPLE: frequency separation (a classic retouching technique)
    - low frequencies (the blur) = light, shadow, ambient colour
      → taken from the HARMONIZED image (the AI's work)
    - high frequencies (the detail) = raffia texture, embroidery threads, edges
      → taken from the 3D RENDER (the product's ground truth)

MULTIPLICATIVE VERSION: detail is a RATIO, detail = render / blur(render).
A thread 10 % brighter than its neighbourhood stays 10 % brighter, whether in sunlight or in
shadow. With a plain subtraction, the detail would keep the amplitude of the studio lighting and
"float" over shadowed areas.

    result = blur(harmonized) × render / blur(render)        (inside the object mask)
    result = harmonized                                      (outside the mask)

SIGMA: the size (in pixels) separating "light" from "detail".
    small sigma (1-2): only the finest grain is replaced, the AI keeps almost everything
    large sigma (8+):  medium patterns are restored too; more faithful, but the AI's lighting
                       fades in favour of the studio's
It is a fidelity / integration trade-off: measure it with the QA.

"MASKED" BLUR: we blur render × mask, then divide by the blurred mask. Otherwise, near the
silhouette, the blur would mix the bag with the transparent (black) background and the bag's edge
would get a bright fringe.

Usage:
    python frequency_reinjection.py --harmonized harmo.png --render cam05_beauty.png --sigma 4 --chroma 0.8 --out fixed.png
"""

import argparse

import cv2
import numpy as np


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    alpha = img[..., 3].astype(np.float32) / 255 if img.shape[2] == 4 else None
    rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return rgb, alpha


def srgb_to_linear(c):
    # The "detail" ratio must be computed on real (linear) light, not on gamma-compressed sRGB values.
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def masked_blur(img, mask, sigma):
    m = mask[..., None]
    num = cv2.GaussianBlur(img * m, (0, 0), sigma)
    den = cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-4)


def reinject(harmonized, render, mask, sigma, eps=1e-3):
    h, r = srgb_to_linear(harmonized), srgb_to_linear(render)
    detail = (r + eps) / (masked_blur(r, mask, sigma) + eps)
    light = masked_blur(h, mask, sigma)
    fixed = light * detail
    # 1 px feather at the silhouette edge, to avoid a hard cut.
    feather = cv2.GaussianBlur(mask, (0, 0), 0.8)[..., None]
    out = fixed * feather + h * (1 - feather)
    return linear_to_srgb(out)


def restore_chroma(img, render, mask, amount):
    """Inside the mask, replaces hue/saturation (a*, b*) with the render's, keeping the image's
    lightness L* (the light added by the AI). The QA showed the AI drifts the PRODUCT's colour
    (not only the scene's): this is what is corrected here."""
    lab_i = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    lab_r = cv2.cvtColor(render.astype(np.float32), cv2.COLOR_RGB2LAB)
    m = cv2.GaussianBlur(mask, (0, 0), 0.8)[..., None] * amount
    lab_i[..., 1:] = lab_r[..., 1:] * m + lab_i[..., 1:] * (1 - m)
    return np.clip(cv2.cvtColor(lab_i, cv2.COLOR_LAB2RGB), 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harmonized", required=True, help="image harmonized by the AI")
    p.add_argument("--render", required=True, help="RGBA beauty of the 3D render (same camera)")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--chroma", type=float, default=0.0,
                   help="0 to 1: share of the product COLOUR (a*, b* in Lab) taken from the render. "
                        "Lightness (L*, i.e. the AI's light) is kept. 1 = exact product colour")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    harmonized, _ = read_rgb(a.harmonized)
    render, alpha = read_rgb(a.render)
    if alpha is None:
        raise ValueError("the render must have an alpha channel (RGBA beauty)")
    if harmonized.shape != render.shape:
        raise ValueError(f"different sizes: {harmonized.shape} vs {render.shape}")
    # Only the solid inside of the bag is used: a half-transparent edge pixel mixes bag and
    # background, its "detail" would be wrong.
    mask = (alpha > 0.5).astype(np.float32)

    out = reinject(harmonized, render, mask, a.sigma)
    if a.chroma > 0:
        out = restore_chroma(out, render, mask, a.chroma)
    cv2.imwrite(a.out, cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"[ok] {a.out} (sigma {a.sigma})")


if __name__ == "__main__":
    main()
