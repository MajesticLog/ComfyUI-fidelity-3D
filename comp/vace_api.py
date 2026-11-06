"""
vace_api.py — Queues a Wan 2.2 Fun VACE render in ComfyUI through its API (http://127.0.0.1:8000).

Graph (native ComfyUI nodes, two stages "high noise" then "low noise", like Wan 2.2 A14B):
    control video (animated depth) ──┐
    reference image (optional) ──────┼─► WanVaceToVideo ─► KSamplerAdvanced (high model, steps 0→k)
    positive / negative prompts ─────┘                    └► KSamplerAdvanced (low model, steps k→n)
                                                            ─► TrimVideoLatent ─► VAEDecode ─► MP4 video

Usage:
    python vace_api.py --control drop_depth.mp4 --frames 17 --size 480 --steps 6 --prefix video/vace_test
    python vace_api.py ... --lightning --steps 4 --cfg 1   (T2V lightx2v 4-step LoRAs)

The graph JSON can also be written (--json) to open and edit it in ComfyUI.
Input files (control video, reference image) must be in ComfyUI's input folder.
"""

import argparse
import json
import urllib.request

PROMPT = ("A small structured leather handbag in blush pink with polished gold hardware falls slowly onto a "
          "pastel blue round table on a sunny Parisian café terrace. Slow motion, photorealistic luxury "
          "commercial, static camera, warm late afternoon light, shallow depth of field.")
NEGATIVE = ("blurry, distorted handbag, deformed handles, extra straps, melting, text, watermark, low quality, "
            "flicker, camera shake, people")


def graph(a):
    lo = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
    hi = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
    split = a.steps // 2
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": a.shift}},
        "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": a.shift}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": a.prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": NEGATIVE}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "9": {"class_type": "LoadVideo", "inputs": {"file": a.control}},
        "10": {"class_type": "GetVideoComponents", "inputs": {"video": ["9", 0]}},
        "11": {"class_type": "WanVaceToVideo", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0], "width": a.size, "height": a.size,
            "length": a.frames, "batch_size": 1, "strength": a.strength, "control_video": ["10", 0]}},
        "12": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["3", 0], "add_noise": "enable", "noise_seed": a.seed, "steps": a.steps, "cfg": a.cfg,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["11", 0], "negative": ["11", 1],
            "latent_image": ["11", 2], "start_at_step": 0, "end_at_step": split,
            "return_with_leftover_noise": "enable"}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["4", 0], "add_noise": "disable", "noise_seed": a.seed, "steps": a.steps, "cfg": a.cfg,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["11", 0], "negative": ["11", 1],
            "latent_image": ["12", 0], "start_at_step": split, "end_at_step": 10000,
            "return_with_leftover_noise": "disable"}},
        "14": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["13", 0], "trim_amount": ["11", 3]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["8", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": 16}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": a.prefix, "format": "mp4", "codec": "h264"}},
    }
    if a.reference:
        g["18"] = {"class_type": "LoadImage", "inputs": {"image": a.reference}}
        g["11"]["inputs"]["reference_image"] = ["18", 0]
    if a.lightning:
        g["19"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": hi, "strength_model": 1.0}}
        g["20"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": lo, "strength_model": 1.0}}
        g["3"]["inputs"]["model"] = ["19", 0]
        g["4"]["inputs"]["model"] = ["20", 0]
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True, help="control video in ComfyUI's input folder")
    p.add_argument("--reference", help="reference image in ComfyUI's input folder (optional)")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--frames", type=int, default=49, help="4n+1 (Wan's VAE encodes time in blocks of 4)")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--strength", type=float, default=1.0, help="VACE control strength")
    p.add_argument("--seed", type=int, default=1312)
    p.add_argument("--lightning", action="store_true", help="4-step LoRAs (use --steps 4 --cfg 1)")
    p.add_argument("--prefix", default="video/vace")
    p.add_argument("--json", help="write the API graph here")
    p.add_argument("--no-queue", action="store_true", help="only write the JSON, do not queue it")
    a = p.parse_args()
    g = graph(a)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1, ensure_ascii=False)
    if not a.no_queue:
        r = urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8000/prompt", data=json.dumps({"prompt": g}).encode(),
            headers={"Content-Type": "application/json"}))
        print(r.read().decode())


if __name__ == "__main__":
    main()
