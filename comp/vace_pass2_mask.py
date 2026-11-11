"""
vace_pass2_mask.py — Re-runs ONLY pass 2 of the two-pass video (yellow repaint), on an already rendered pass 1,
with the mask of your choice. Used for the "tighter mask" experiment.

Why start from a saved pass 1: so that the only difference between two runs is the mask (and the seed, which is
varied on purpose to separate the effect of the mask from chance).

Settings: CLEAN Lightning (lightx2v LoRA at 1.0, 4 steps, cfg 1, high→low switch at step 2, shift 5). This is the
configuration the LoRA is distilled for: adding cfg > 1 on top of it "overcooks" the image (blown highlights,
crushed blacks, doubled saturation).

Finding: with the EXACT silhouette mask (pass2_masks.py --margin 0) instead of the original +4 px one, the share
of deformed edges is roughly halved (deformation_map.py). Inside the grey hole, VACE only knows the bag's shape
from the mask contour: a mask 4 px too wide makes it fill a bigger bag.

Outputs (in ComfyUI's output/): <prefix>_raw (pass 2 alone) and <prefix>_final (composited inside the mask).

Usage:
    python vace_pass2_mask.py --mask drop_mask_after_contact_m0.mp4 --seed 1101480555102272 \\
        --prefix video/mask_m0_s1 --queue
"""

import argparse
import json
import urllib.request

import vace_two_pass as v2

PROMPT_2 = ("A small structured bright lemon yellow leather handbag with polished silver hardware resting on a pastel "
            "blue round table on a sunny Parisian café terrace. The handbag is fully lit by bright direct sunlight from "
            "behind on the right, with crisp highlights. Photorealistic luxury commercial, static camera, shallow depth "
            "of field.")


def build_graph(a):
    g = {
        "unet_hi": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "unet_lo": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "lora_hi": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet_hi", 0], "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "strength_model": 1.0}},
        "lora_lo": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet_lo", 0], "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", "strength_model": 1.0}},
        "ms_hi": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_hi", 0], "shift": 5.0}},
        "ms_lo": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_lo", 0], "shift": 5.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": v2.NEGATIVE}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "p1_vid": {"class_type": "LoadVideo", "inputs": {"file": a.pass1}},
        "p1_img": {"class_type": "GetVideoComponents", "inputs": {"video": ["p1_vid", 0]}},
        "mask_vid": {"class_type": "LoadVideo", "inputs": {"file": a.mask}},
        "mask_img": {"class_type": "GetVideoComponents", "inputs": {"video": ["mask_vid", 0]}},
        "mask_raw": {"class_type": "ImageToMask", "inputs": {"image": ["mask_img", 0], "channel": "red"}},
        "mask": {"class_type": "ThresholdMask", "inputs": {"mask": ["mask_raw", 0], "value": 0.5}},
        "grey": {"class_type": "EmptyImage", "inputs": {"width": a.size, "height": a.size, "batch_size": a.frames, "color": v2.GREY}},
        "ref_yellow": {"class_type": "LoadImage", "inputs": {"image": a.reference}},
        "p2_input": {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["p1_img", 0], "source": ["grey", 0],
                     "x": 0, "y": 0, "resize_source": False, "mask": ["mask", 0]}},
    }
    d2 = v2.add_pass(g, "p2_", PROMPT_2, ["p2_input", 0], ["mask", 0], a, ["ref_yellow", 0])
    v2.save(g, "p2", [d2, 0], a.prefix + "_raw")
    g["final"] = {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["p1_img", 0], "source": [d2, 0],
                  "x": 0, "y": 0, "resize_source": False, "mask": ["mask", 0]}}
    v2.save(g, "fin", ["final", 0], a.prefix + "_final")
    return v2.number_nodes(g)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", required=True, help="mask video in ComfyUI's input/ folder")
    p.add_argument("--pass1", default="drop_pass1_pink.mp4", help="pass 1 video in ComfyUI's input/ folder")
    p.add_argument("--reference", default="drop_ref_yellow_sun.png")
    p.add_argument("--seed", type=int, default=1101480555102272)
    p.add_argument("--prefix", required=True)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--json", help="also write the graph here")
    p.add_argument("--queue", action="store_true", help="queue it in ComfyUI (http://127.0.0.1:8000)")
    a = p.parse_args()
    g = build_graph(a)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1, ensure_ascii=False)
    if a.queue:
        r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/prompt",
                                   data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"}))
        print(r.read().decode())


if __name__ == "__main__":
    main()
