# ComfyUI-fidelity-3D

**Use generative AI on a luxury product without losing the product — and prove it with numbers.**

A 3D render is used as the *ground truth*. Generative AI (Flux, Qwen Image Edit, Wan 2.2 VACE in ComfyUI) creates the set, the light and the motion. Python tools measure, zone by zone, how far the product drifted, and restore it.

> **Disclaimer.** Personal, non-commercial test project, **not affiliated with, endorsed by or connected to** the fashion house whose bag design inspired the 3D model used here. The 3D model "Lady D-Lite" is by **Roberto Domínguez** (CC-BY), the HDRI "Studio Small 09" by **Sergej Majboroda / Poly Haven** (CC0). Any trademark visible in the renders belongs to its owner.

![Final image](docs/images/final.jpg)

---

## Why

Generative AI makes beautiful images, but it **quietly changes the product**: embroidery is re-invented, colours drift, hardware melts. To the eye the result often looks intact. For a luxury house, a thread or a shade that moved is a refused image.

| | Structure (SSIM) | Colour (ΔE2000) |
|---|---|---|
| AI harmonized output | **0.72** ❌ | **6.4** ❌ |
| After frequency re-injection + colour restore | **0.92** ✅ | **1.3** ✅ |

![Heatmap](docs/images/heatmap.jpg)

*Blue = preserved, red = changed. Flowers and lettering survive; the raffia weave was re-invented and the hue drifted — invisible at a glance, obvious in the measurement.*

---

## The method

```
 Blender (3D = truth)            ComfyUI (AI = creative)                 Python (script = the boundary)
 ─────────────────────           ───────────────────────────             ─────────────────────────────
 beauty RGBA (product)  ──┐      1. Set: Flux.1 dev + ControlNet depth   4. Frequency re-injection:
 depth (with table)     ──┼────► 2. Composite: beauty over set      ┌──►    AI light + 3D texture & colour
 masks per material     ──┐ │    3. Harmonize: Qwen Image Edit 2511 ─┘    5. Fidelity QA per zone
                          │ └──────────────────────────────────────────►    (SSIM, edges F1, ΔE2000)
                          └────────────────────────────────────────────►
```

1. **3D is the source of truth.** The exact product, plus everything the AI cannot do alone: depth, per-material masks, and a pixel-aligned reference to measure against.
2. **AI does the creative part.** Set, light, atmosphere, motion.
3. **Scripts hold the line.** Deterministic, reproducible steps: restore the product, then measure it.

### Demo A — still image

| Step | Image |
|---|---|
| 3D render (reference) | ![](docs/images/render_3d.jpg) |
| Depth with an invisible round table (disparity-encoded) — without it the bag floats | ![](docs/images/depth_object_only.jpg) ![](docs/images/depth_with_table.jpg) |
| Flux + ControlNet depth: set generated *around* the bag's exact silhouette | ![](docs/images/decor_no_table.jpg) ![](docs/images/decor_with_table.jpg) |
| Composite (looks pasted: studio light on a sunset set) | ![](docs/images/composite.jpg) |
| Harmonized by Qwen Image Edit 2511 (Lightning 4 steps / 4 steps / 20 steps) | ![](docs/images/harmonize_lightning.jpg) ![](docs/images/harmonize_4steps.jpg) ![](docs/images/harmonize_20steps.jpg) |
| Detail ×2: composite · AI harmonized · restored | ![](docs/images/detail_zoom.jpg) |

### Demo C — video (work in progress)

**When the AI invents the motion, the product drifts.** Wan 2.2 image-to-video with a "static camera" prompt still slid the bag by up to 15 px in 3 s, with parallax (no 2D re-alignment can fix it), and flickered every 4 frames — the period of Wan's video VAE.

![Parallax](docs/images/wan_i2v_parallax.jpg)

**So 3D dictates the motion, and the AI follows.** A Blender animation (the bag drops on a table and changes colour at contact) provides per-frame depth, masks and references. Wan 2.2 Fun VACE is guided by the animated depth: shape and motion follow the 3D by construction, no tracking needed.

![3D drop sequence](docs/images/drop_sequence_3d.jpg)
![VACE depth control](docs/images/vace_depth_control.jpg)

**A colour transition animated by the AI, at the moment set by 3D.** Pass 1 renders a blush-pink bag. Pass 2 is a VACE inpainting with a mask that only exists *after* contact, plus a lemon-yellow reference image rendered from 3D: the AI repaints the bag over ~10 frames. A script then keeps the AI's per-pixel transition timing but enforces the exact brand colours (leather ΔE 8.3 → 1.3).

![Colour transition](docs/images/vace_color_transition.jpg)

*Top: AI output (salmon pink, mustard yellow). Bottom: brand colours restored, same animated transition. Metal hardware is not solved yet (ΔE 8.6).*

---

## Repository

| Path | What it does |
|---|---|
| `blender/export_passes.py` | Renders beauty (RGBA), depth, camera-space normals and **one mask per material** for each camera. Emission-material swap (robust to Blender 5's compositor changes). Options: `--passes depth_table` (invisible round table in the depth only, disparity encoding), `--sun` / `--bounce` (relight the beauty to match a generated set). |
| `blender/animate_drop.py` | Drop animation with colour switch at contact: per-frame simple render, depth (fixed range), beauty, masks (object / metal / leather). |
| `blender/flatten_simple.py`, `blender/inspect_model.py` | Helpers. |
| `qa/fidelity_qa.py` | **Per-zone fidelity QA**: SSIM after light equalization, edge F1 (Canny, ±2 px), ΔE2000 with reference lightness. HTML report, heatmap, edge map. |
| `qa/make_test_case.py` | Builds a known-good and a known-bad image to self-test the QA. |
| `comp/frequency_reinjection.py` | Keeps the AI's low frequencies (light) and re-injects the render's high frequencies (texture), multiplicatively in linear light; optional chroma restore in Lab. |
| `comp/video_qa.py` | Per-frame QA + re-injection for video, with affine re-alignment (ECC) and temporal stability. |
| `comp/video_brand_color.py` | Enforces exact brand colours in a video while keeping the AI's transition timing. |
| `comp/vace_api.py`, `comp/vace_two_pass.py` | Build / queue Wan 2.2 Fun VACE graphs through the ComfyUI API. |
| `workflows/` | ComfyUI workflows (UI format, plus one API-format graph). |

### Models used (not included)

Flux.1 dev (fp8) · FLUX.1-dev ControlNet Union Pro 2.0 (fp8) · Qwen-Image-Edit-2511 (GGUF Q4_K_M) + Lightning 4-step LoRA · Wan 2.2 I2V 14B + lightx2v · Wan 2.2 Fun VACE 14B (fp8) + T2V lightx2v 4-step LoRAs · Wan 2.1 VAE · umt5-xxl. Tested on an RTX 5080 Laptop (16 GB VRAM), ComfyUI 0.29, Blender 5.2.

**Tip:** verify every downloaded model with its SHA-256 (`X-Linked-ETag` on Hugging Face). Three browser-downloaded files here had the right size but corrupted content; a corrupted VAE produced uniform grey images.

### Install

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows (use .venv/bin/pip elsewhere)
```
Blender scripts run inside Blender 5.x (`blender -b --factory-startup --python <script> -- <options>`).

### Quick start

```bash
# 0. Self-test the QA on a demo scene: the "ok" image must PASS, the "ko" image must FAIL
blender -b --factory-startup --python blender/export_passes.py -- --demo --out renders_demo --num-cams 1
python qa/make_test_case.py renders_demo/cam01
python qa/fidelity_qa.py --ref renders_demo/cam01_beauty.png --gen renders_demo/cam01_gen_ok.png --out reports/ok --white-balance logo
python qa/fidelity_qa.py --ref renders_demo/cam01_beauty.png --gen renders_demo/cam01_gen_ko.png --out reports/ko --white-balance logo

# 1. Passes for your own model (beauty, depth with a round table, normals, masks per material)
blender -b --factory-startup --python blender/export_passes.py -- --import model.glb --out renders         --hdri studio.hdr --passes beauty,depth,depth_table,normal,masks

# 2. In ComfyUI: workflows/01 (set), 02 (composite), 03 (harmonize)

# 3. Restore texture and colour, then measure
python comp/frequency_reinjection.py --harmonized harmonized.png --render renders/cam05_beauty.png        --sigma 4 --chroma 0.8 --out fixed.png
python qa/fidelity_qa.py --ref renders/cam05_beauty.png --gen fixed.png --out reports/fixed
```

**Video:** `blender/animate_drop.py` renders the per-frame passes; encode `depth/` and the after-contact mask as MP4 (`drop_depth.mp4`, `drop_mask_after_contact.mp4`) into ComfyUI's input folder with a yellow reference image (`drop_ref_yellow.png`); run `workflows/06_*`; then `comp/video_brand_color.py` and `comp/video_qa.py`.

---

## Known limits

- **Partly circular after correction.** Re-injected texture and colour come from the render, so their scores rise partly by construction. What is proven: the raw AI output drifted, and the correction did not damage the AI's light. The tool's value is in **detection**.
- **Thresholds are starting values** (SSIM ≥ 0.80, F1 ≥ 0.75, ΔE ≤ 5; < 2 for a brand colour). They should be calibrated with a house, on its own approved and rejected references.
- **1024 px only**, one product, one material family (embroidered raffia). Not yet tested: 4K, polished metal, glass, patent leather.
- **Video is a first result**, not a finished pipeline.

## Next

- Estimate the generated set's lighting and **re-render the product in Blender** with it — exact by construction, no correction needed.
- Organic transition masks ("ink in water") driven from the impact point.
- A blind, independent evaluation (perceptual metrics, OCR of lettering, vision-language checks) to break the circularity.
- A per-zone pass/fail report an art director can sign off.

## License

Code: MIT (see `LICENSE`). Images in `docs/` are renders and AI outputs of a third-party CC-BY model, shown for documentation only.
