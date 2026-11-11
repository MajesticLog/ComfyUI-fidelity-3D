"""
vace_two_pass.py — Writes the ComfyUI graph (API format) of the TWO-PASS colour transition, with Lightning.

PASS 1: animated depth → VACE → POWDER PINK bag + gold hardware, set, light, motion.
PASS 2: VACE inpainting IN TIME. Pass 1's video, with the masked area painted grey, + a mask that
        only exists AFTER contact, + a reference image of the new colourway rendered from 3D → the
        AI repaints the bag LEMON YELLOW + silver, at the exact moment set by the 3D animation.
FINAL:  pass 2 is composited over pass 1 through the mask (outside the mask, pass 1 stays
        pixel-identical).

Three videos are saved: pass1, pass2 (raw) and final.

Input files expected in ComfyUI's input folder:
    drop_depth.mp4               animated depth (blender/animate_drop.py)
    drop_mask_after_contact.mp4  black before contact, dilated bag silhouette after
    drop_ref_yellow.png          the bag in the new colourway (3D render on white)

Usage:
    python vace_two_pass.py --json two_pass_api.json             (writes the file)
    python vace_two_pass.py --json two_pass_api.json --queue     (and queues it)
To open it in ComfyUI: drag the .json onto the canvas (API format is accepted).

Lesson learned: without the reference image, pass 2 repainted the bag PINK again. VACE propagates the
identity of the unmasked frames 0-9, and at cfg 1 (Lightning) the "lemon yellow" prompt weighs too
little against that visual context.
"""

import argparse
import json
import urllib.request

PROMPT_1 = ("A small structured leather handbag in blush pink with polished gold hardware falls slowly onto a "
            "pastel blue round table on a sunny Parisian café terrace. Slow motion, photorealistic luxury "
            "commercial, static camera, warm late afternoon light, shallow depth of field.")
PROMPT_2 = ("A small structured lemon yellow leather handbag with polished silver hardware resting on a pastel "
            "blue round table on a sunny Parisian café terrace. Photorealistic luxury commercial, static camera, "
            "warm late afternoon light, shallow depth of field.")
NEGATIVE = ("blurry, distorted handbag, deformed handles, extra straps, melting, text, watermark, low quality, "
            "flicker, camera shake, people")
GREY = 0x7F7F7F  # VACE inpainting convention: the area to regenerate is filled with neutral grey


def add_pass(g, prefix, positive_txt, control, mask, a, reference=None):
    """Adds a VACE pass to g (high/low models + Lightning, 2 KSamplers). Returns the VAE Decode id."""
    n = lambda s: f"{prefix}{s}"
    g[n("pos")] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": positive_txt}}
    vace = {"positive": [n("pos"), 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": a.size,
            "height": a.size, "length": a.frames, "batch_size": 1, "strength": 1.0, "control_video": control}
    if mask:
        vace["control_masks"] = mask
    if reference:
        vace["reference_image"] = reference
    g[n("vace")] = {"class_type": "WanVaceToVideo", "inputs": vace}
    common = {"cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "steps": 4,
              "positive": [n("vace"), 0], "negative": [n("vace"), 1]}
    g[n("ks1")] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": ["ms_hi", 0], "add_noise": "enable",
                   "noise_seed": a.seed, "latent_image": [n("vace"), 2], "start_at_step": 0, "end_at_step": 2,
                   "return_with_leftover_noise": "enable"}}
    g[n("ks2")] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": ["ms_lo", 0], "add_noise": "disable",
                   "noise_seed": a.seed, "latent_image": [n("ks1"), 0], "start_at_step": 2, "end_at_step": 10000,
                   "return_with_leftover_noise": "disable"}}
    g[n("trim")] = {"class_type": "TrimVideoLatent", "inputs": {"samples": [n("ks2"), 0], "trim_amount": [n("vace"), 3]}}
    g[n("dec")] = {"class_type": "VAEDecode", "inputs": {"samples": [n("trim"), 0], "vae": ["vae", 0]}}
    return n("dec")


def save(g, id_, images, prefix):
    g[id_ + "_cv"] = {"class_type": "CreateVideo", "inputs": {"images": images, "fps": 16}}
    g[id_ + "_sv"] = {"class_type": "SaveVideo", "inputs": {"video": [id_ + "_cv", 0], "filename_prefix": prefix,
                                                           "format": "mp4", "codec": "h264"}}


def build_graph(a):
    g = {
        "unet_hi": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "unet_lo": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "lora_hi": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet_hi", 0], "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "strength_model": 1.0}},
        "lora_lo": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet_lo", 0], "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", "strength_model": 1.0}},
        "ms_hi": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_hi", 0], "shift": 5.0}},
        "ms_lo": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_lo", 0], "shift": 5.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEGATIVE}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        # Pass 1 control: the animated depth
        "depth_vid": {"class_type": "LoadVideo", "inputs": {"file": a.depth}},
        "depth_img": {"class_type": "GetVideoComponents", "inputs": {"video": ["depth_vid", 0]}},
        # Temporal mask: black before contact, (dilated) silhouette after
        "mask_vid": {"class_type": "LoadVideo", "inputs": {"file": a.mask}},
        "mask_img": {"class_type": "GetVideoComponents", "inputs": {"video": ["mask_vid", 0]}},
        "mask_raw": {"class_type": "ImageToMask", "inputs": {"image": ["mask_img", 0], "channel": "red"}},
        # MP4 compression softens the edges: re-binarize
        "mask": {"class_type": "ThresholdMask", "inputs": {"mask": ["mask_raw", 0], "value": 0.5}},
        "grey": {"class_type": "EmptyImage", "inputs": {"width": a.size, "height": a.size, "batch_size": a.frames, "color": GREY}},
    }
    # Optional pass 1 reference image (e.g. the pink bag rendered from 3D in the target light).
    ref1 = None
    if a.reference1:
        g["ref_pink"] = {"class_type": "LoadImage", "inputs": {"image": a.reference1}}
        ref1 = ["ref_pink", 0]
    d1 = add_pass(g, "p1_", a.prompt1, ["depth_img", 0], None, a, ref1)
    save(g, "p1", [d1, 0], "video/vace_pass1_pink")
    # Pass 2: pass 1 with a grey hole inside the mask
    g["p2_input"] = {"class_type": "ImageCompositeMasked", "inputs": {"destination": [d1, 0], "source": ["grey", 0],
                     "x": 0, "y": 0, "resize_source": False, "mask": ["mask", 0]}}
    # Reference image of the new colourway (3D render): without it, VACE copies the pink identity of the
    # unmasked frames and the prompt alone is not enough at cfg 1.
    ref = None
    if a.reference2:
        g["ref_yellow"] = {"class_type": "LoadImage", "inputs": {"image": a.reference2}}
        ref = ["ref_yellow", 0]
    d2 = add_pass(g, "p2_", a.prompt2, ["p2_input", 0], ["mask", 0], a, ref)
    save(g, "p2", [d2, 0], "video/vace_pass2_yellow_raw")
    # Final: pass 2 kept only inside the mask, pass 1 everywhere else
    g["final"] = {"class_type": "ImageCompositeMasked", "inputs": {"destination": [d1, 0], "source": [d2, 0],
                  "x": 0, "y": 0, "resize_source": False, "mask": ["mask", 0]}}
    save(g, "fin", ["final", 0], "video/vace_final_transition")
    return number_nodes(g)


def number_nodes(g):
    """Replaces readable ids with numbers (ComfyUI's loader prefers them); names are kept as titles."""
    ids = {k: str(i + 1) for i, k in enumerate(g)}
    out = {}
    for k, node in g.items():
        inputs = {c: ([ids[v[0]], v[1]] if isinstance(v, list) and len(v) == 2 and v[0] in ids else v)
                  for c, v in node["inputs"].items()}
        out[ids[k]] = {"class_type": node["class_type"], "inputs": inputs, "_meta": {"title": k}}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--queue", action="store_true", help="also queue it in ComfyUI (http://127.0.0.1:8000)")
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=1312)
    p.add_argument("--depth", default="drop_depth.mp4")
    p.add_argument("--mask", default="drop_mask_after_contact.mp4")
    p.add_argument("--prompt1", default=PROMPT_1)
    p.add_argument("--prompt2", default=PROMPT_2)
    p.add_argument("--reference1", default="", help="pass 1 reference image (optional)")
    p.add_argument("--reference2", default="drop_ref_yellow.png", help="pass 2 reference image ('' for none)")
    a = p.parse_args()
    g = build_graph(a)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1, ensure_ascii=False)
    print("written:", a.json)
    if a.queue:
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/prompt",
                                   data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"}))
        print(r.read().decode())


if __name__ == "__main__":
    main()
