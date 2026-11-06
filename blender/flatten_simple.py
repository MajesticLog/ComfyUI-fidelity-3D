"""flatten_simple.py — Composites the "simple" render frames (RGBA) over a flat neutral grey.
The AI then receives a bag and a table on a neutral background, which it replaces with the prompted set.
Usage: python flatten_simple.py ../renders/drop"""
import glob
import os
import sys

from PIL import Image

folder = sys.argv[1]
os.makedirs(os.path.join(folder, "simple_grey"), exist_ok=True)
for f in sorted(glob.glob(os.path.join(folder, "simple", "*.png"))):
    im = Image.open(f).convert("RGBA")
    background = Image.new("RGBA", im.size, (110, 110, 110, 255))
    background.alpha_composite(im)
    background.convert("RGB").save(os.path.join(folder, "simple_grey", os.path.basename(f)))
print("ok", folder)
