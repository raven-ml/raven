from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_UBENCH = _ROOT / "vendor" / "ubench"
if str(_UBENCH) not in sys.path:
    sys.path.insert(0, str(_UBENCH))

import ubench  # type: ignore


DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_images() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    color_images: dict[str, np.ndarray] = {}
    gray_images: dict[str, np.ndarray] = {}

    for name in ["img_1920x1080.png", "img_1280x720.png", "img_512x512.png"]:
        path = DATA_DIR / name
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        color_images[name] = img_rgb
        gray_images[name] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return color_images, gray_images


COLOR_IMAGES, GRAY_IMAGES = _load_images()


def build_benchmarks() -> List[Any]:
    benches: List[Any] = []

    color_1080 = COLOR_IMAGES["img_1920x1080.png"]
    gray_1080 = GRAY_IMAGES["img_1920x1080.png"]
    gray_720 = GRAY_IMAGES["img_1280x720.png"]

    def bench_grayscale() -> None:
        gray = cv2.cvtColor(color_1080, cv2.COLOR_RGB2GRAY)
        int(gray.sum())

    benches.append(ubench.bench("ToGrayscale/1080p (OpenCV)", bench_grayscale))

    def bench_gaussian() -> None:
        blurred = cv2.GaussianBlur(color_1080, ksize=(5, 5), sigmaX=1.2, sigmaY=1.2)
        int(blurred.sum())

    benches.append(ubench.bench("GaussianBlur/1080p (OpenCV)", bench_gaussian))

    def bench_sobel() -> None:
        sobel_x = cv2.Sobel(gray_720, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
        int(sobel_x.sum())

    benches.append(ubench.bench("Sobel/720p (OpenCV)", bench_sobel))

    def bench_canny() -> None:
        edges = cv2.Canny(gray_1080, threshold1=55.0, threshold2=120.0)
        int(edges.sum())

    benches.append(ubench.bench("Canny/1080p (OpenCV)", bench_canny))

    return benches


def default_config() -> ubench.Config:
    return ubench.Config.default().build()


def main() -> None:
    benchmarks = build_benchmarks()
    config = default_config()
    ubench.run(benchmarks, config=config, output_format="pretty", verbose=False)


if __name__ == "__main__":
    main()
