"""The synthetic frames of Sowilo's benchmarks.

bench_sowilo.py draws them with numpy when it starts; bench_sowilo.ml draws
the same palettes with Nx, whose noise has the same distribution and other
values.
"""

from __future__ import annotations

import numpy as np


def _make_image(width: int, height: int, seed: int, *, palette: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    radial = np.sqrt(xv**2 + yv**2)
    angle = np.arctan2(yv, xv)

    if palette == "sunset":
        base_r = 0.6 + 0.4 * np.cos(angle)
        base_g = 0.4 + 0.35 * np.sin(radial * np.pi)
        base_b = 0.2 + 0.6 * np.exp(-radial * 1.5)
    else:  # "forest"
        base_r = 0.2 + 0.3 * np.exp(-radial * 2.0)
        base_g = 0.4 + 0.5 * np.cos(angle * 2.0)
        base_b = 0.3 + 0.4 * np.sin(radial * np.pi)

    noise = rng.normal(loc=0.0, scale=0.05, size=(height, width)).astype(np.float32)
    channels = [base_r, base_g, base_b]
    stacked = []
    for channel in channels:
        layer = channel + noise
        layer = np.clip(layer, 0.0, 1.0)
        stacked.append((layer * 255.0).astype(np.uint8))

    return np.stack(stacked, axis=-1)


# The frames the benchmarks run on: width, height, numpy seed and palette.
SPECS = {
    "1080p": (1920, 1080, 7, "sunset"),
    "720p": (1280, 720, 19, "forest"),
}


def images() -> dict[str, np.ndarray]:
    """Each frame of SPECS, by name, as RGB uint8 of shape (height, width, 3)."""
    return {name: _make_image(w, h, seed, palette=p) for name, (w, h, seed, p) in SPECS.items()}
