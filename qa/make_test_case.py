"""
make_test_case.py — Builds two fake "generated images" from a render, to check that the QA
tool reacts as expected:

    gen_ok.png : different background + modified light (warmer, darker)
                 -> must PASS (harmonization is allowed to do that)
    gen_ko.png : same + warped logo + leather shifting hue
                 -> must FAIL on the "logo" and "leather" zones

Usage: python make_test_case.py ../renders_demo/cam01
(the demo scene of export_passes.py --demo has "leather", "metal" and "logo" materials)
"""

import sys

import cv2
import numpy as np

prefix = sys.argv[1]
beauty = cv2.imread(prefix + "_beauty.png", cv2.IMREAD_UNCHANGED)
alpha = beauty[:, :, 3:4].astype(np.float32) / 255
obj = beauty[:, :, :3].astype(np.float32)
h, w = obj.shape[:2]

# Background: dark "marble" gradient + light noise
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
background = np.dstack([40 + 30 * yy / h, 45 + 25 * yy / h, 55 + 20 * yy / h])
background += np.random.default_rng(0).normal(0, 4, background.shape)

# Simulated relighting: warm light gradient from left to right + global dimming
light = (0.75 + 0.25 * xx / w)[..., None] * np.array([1.0, 0.95, 0.88])  # BGR -> warmer
lit_obj = np.clip(obj * light, 0, 255)

gen_ok = lit_obj * alpha + background * (1 - alpha)
cv2.imwrite(prefix + "_gen_ok.png", gen_ok.astype(np.uint8))

# Defects: 1) warped logo (ripple), 2) leather shifting to another hue
logo = cv2.imread(prefix + "_mask_logo.png", cv2.IMREAD_GRAYSCALE) > 127
leather = cv2.imread(prefix + "_mask_leather.png", cv2.IMREAD_GRAYSCALE) > 127
map_x = (xx + 3 * np.sin(yy / 2.5)).astype(np.float32)
map_y = (yy + 2 * np.cos(xx / 3.0)).astype(np.float32)
warped = cv2.remap(gen_ok.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
logo_zone = cv2.dilate(logo.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

gen_ko = gen_ok.copy()
gen_ko[logo_zone] = warped[logo_zone]
hsv = cv2.cvtColor(np.clip(gen_ko, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
hsv[..., 0][leather] = (hsv[..., 0][leather].astype(int) + 12) % 180  # hue shift
gen_ko = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite(prefix + "_gen_ko.png", gen_ko)
print("Written:", prefix + "_gen_ok.png", "and", prefix + "_gen_ko.png")
