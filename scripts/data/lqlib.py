"""Shared helpers: degradations and image I/O (ROITOP patches are 8-bit grayscale .bmp, 256x256)."""
import hashlib
import os

import cv2
import numpy as np

import noisemodel  # calibrated noise study conditions NL8/NL4/NL2/NL1 and NA2/NA4/NA6/NA8

# Speckle: number of looks L (lower = stronger). Tune so input PSNR lands near ~30 and ~25 dB.
SPECKLE_LOOKS = {"SP1": 16, "SP2": 4}
# OCT B-scans are usually stored log-compressed; assumed displayed dynamic range in dB. CHECK.
LOG_DYNAMIC_RANGE_DB = 40.0


def seed_for(name, cond):
    return int(hashlib.md5(f"{name}|{cond}".encode()).hexdigest()[:8], 16)


def down_up(x, factor):
    """Proposal definition: (256x256) -> (256/f x 256/f) area-downsample -> bicubic back to (256x256)."""
    h, w = x.shape[:2]
    small = cv2.resize(x, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def speckle(x01, looks, rng):
    lin = 10.0 ** (x01 * LOG_DYNAMIC_RANGE_DB / 10.0)
    noisy = lin * rng.gamma(shape=looks, scale=1.0 / looks, size=x01.shape)  # mean-1 multiplicative
    return np.clip(10.0 * np.log10(np.maximum(noisy, 1e-12)) / LOG_DYNAMIC_RANGE_DB, 0, 1)


def degrade(img, cond, name):
    """img: uint8 HxW (or HxWxC). cond in X2, X4, SP1, SP2, or a noisemodel condition. Same dtype/shape out."""
    x = img.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed_for(name, cond))
    chans = [x] if x.ndim == 2 else [x[..., c] for c in range(x.shape[2])]
    out = []
    for ch in chans:
        if cond == "X2":
            y = down_up(ch, 2)
        elif cond == "X4":
            y = down_up(ch, 4)
        elif cond in SPECKLE_LOOKS:
            y = speckle(ch, SPECKLE_LOOKS[cond], rng)
        elif cond in noisemodel.CONDS:
            y = noisemodel.apply(ch * 255.0, cond, rng) / 255.0
        else:
            raise ValueError(cond)
        out.append(np.clip(y, 0, 1))
    y = out[0] if x.ndim == 2 else np.stack(out, 2)
    return np.round(y * 255).astype(np.uint8)


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    if img.ndim == 3:  # keep as-is if truly 3-channel; ROITOP patches are 2D
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_img(path, img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, img):
        raise IOError(path)


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".bmp", ".png", ".jpg", ".tif", ".tiff")))
