from collections import Counter
from typing import Dict, Optional

import numpy as np
from PIL import Image


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc
    h = np.zeros_like(maxc)

    r_mask = (maxc == r) & (delta > 0)
    g_mask = (maxc == g) & (delta > 0)
    b_mask = (maxc == b) & (delta > 0)

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
    h *= 60

    s = np.zeros_like(maxc)
    np.divide(delta, maxc, out=s, where=maxc != 0)
    v = maxc
    return np.stack([h, s, v], axis=1)


def _labels_from_hsv(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    labels = np.full(h.shape, "unknown", dtype=object)

    labels[v < 0.16] = "black"
    labels[(v >= 0.16) & (s < 0.14) & (v > 0.82)] = "white"
    labels[(v >= 0.16) & (s < 0.14) & (v <= 0.82)] = "gray"
    labels[(s >= 0.14) & (h < 15)] = "red"
    labels[(s >= 0.14) & (h >= 345)] = "red"
    labels[(s >= 0.14) & (h >= 15) & (h < 38) & (v < 0.65)] = "brown"
    labels[(s >= 0.14) & (h >= 15) & (h < 38) & (v >= 0.65)] = "orange"
    labels[(s >= 0.14) & (h >= 38) & (h < 68) & (s < 0.38) & (v > 0.65)] = "beige"
    labels[(s >= 0.14) & (h >= 38) & (h < 68) & ~((s < 0.38) & (v > 0.65))] = "yellow"
    labels[(s >= 0.14) & (h >= 68) & (h < 165)] = "green"
    labels[(s >= 0.14) & (h >= 165) & (h < 195)] = "teal"
    labels[(s >= 0.14) & (h >= 195) & (h < 255) & (v < 0.42)] = "navy"
    labels[(s >= 0.14) & (h >= 195) & (h < 255) & (v >= 0.42)] = "blue"
    labels[(s >= 0.14) & (h >= 255) & (h < 290)] = "purple"
    labels[(s >= 0.14) & (h >= 290) & (h < 345)] = "pink"
    return labels


def extract_color_profile(image: Image.Image, mask: Optional[Image.Image] = None, max_pixels: int = 7000) -> Dict[str, float]:
    arr = np.asarray(image.convert("RGB"))

    if mask is not None:
        mask_arr = np.asarray(mask.resize(image.size).convert("L")) > 0
        pixels = arr[mask_arr]
    else:
        pixels = arr.reshape(-1, 3)

    if len(pixels) == 0:
        return {}

    if len(pixels) > max_pixels:
        step = max(1, len(pixels) // max_pixels)
        pixels = pixels[::step]

    hsv = _rgb_to_hsv(pixels)
    labels = _labels_from_hsv(hsv)
    labels = [label for label in labels.tolist() if label != "unknown"]

    if not labels:
        return {}

    counts = Counter(labels)
    total = sum(counts.values())
    return {k: round(v / total, 4) for k, v in counts.most_common()}


def best_color(profile: Dict[str, float]) -> str:
    if not profile:
        return "unknown"
    return max(profile.items(), key=lambda x: x[1])[0]
