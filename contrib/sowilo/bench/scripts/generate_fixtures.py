"""Generate synthetic image fixtures for Sowilo benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


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
    elif palette == "forest":
        base_r = 0.2 + 0.3 * np.exp(-radial * 2.0)
        base_g = 0.4 + 0.5 * np.cos(angle * 2.0)
        base_b = 0.3 + 0.4 * np.sin(radial * np.pi)
    else:  # "nebula"
        base_r = 0.45 + 0.5 * np.sin(3.0 * angle)
        base_g = 0.35 + 0.45 * np.cos(5.0 * radial)
        base_b = 0.55 + 0.35 * np.sin(4.0 * angle + radial)

    noise = rng.normal(loc=0.0, scale=0.05, size=(height, width)).astype(np.float32)
    channels = [base_r, base_g, base_b]
    stacked = []
    for channel in channels:
        layer = channel + noise
        layer = np.clip(layer, 0.0, 1.0)
        stacked.append((layer * 255.0).astype(np.uint8))

    return np.stack(stacked, axis=-1)


def _write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(path, format="PNG", optimize=True)


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (1920, 1080, 7, "img_1920x1080.png", "sunset"),
        (1280, 720, 19, "img_1280x720.png", "forest"),
        (512, 512, 29, "img_512x512.png", "nebula"),
    ]

    for width, height, seed, filename, palette in specs:
        img = _make_image(width, height, seed, palette=palette)
        _write_image(data_dir / filename, img)

    print(f"Generated Sowilo fixtures in {data_dir}")


if __name__ == "__main__":
    main()
