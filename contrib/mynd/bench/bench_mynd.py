# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==4.14.0.94"]
# ///
"""OpenCV's side of mynd's bench: each row's time per call.

The frames are bench_mynd.ml's, drawn with numpy: the same palettes and
noise distribution, other noise values. Images are float32 in [0, 1], as
mynd's rows read them, but for Canny, which OpenCV takes on uint8 only.

OpenCV computes other things on some rows: its linear, cubic and Lanczos
resizes do not band-limit a shrink (INTER_AREA does, for Box), its Lanczos
has 4 lobes, its median reads past the border by replication, and its
connected components and NMS are sequential scans.

usage: uv run contrib/mynd/bench/bench_mynd.py
"""

from __future__ import annotations

import gc
import timeit
from collections.abc import Callable, Sequence

import cv2
import numpy as np

REPEATS = 5

Row = tuple[str, Callable[[], object]]


# Fixtures


def image(width: int, height: int, seed: int, palette: str) -> np.ndarray:
    """The RGB frame of bench_mynd.ml as uint8 (height, width, 3)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    radial = np.sqrt(xv**2 + yv**2)
    angle = np.arctan2(yv, xv)
    if palette == "sunset":
        channels = [
            0.6 + 0.4 * np.cos(angle),
            0.4 + 0.35 * np.sin(radial * np.pi),
            0.2 + 0.6 * np.exp(-radial * 1.5),
        ]
    else:  # "forest"
        channels = [
            0.2 + 0.3 * np.exp(-radial * 2.0),
            0.4 + 0.5 * np.cos(angle * 2.0),
            0.3 + 0.4 * np.sin(radial * np.pi),
        ]
    noise = rng.normal(0.0, 0.05, size=(height, width)).astype(np.float32)
    layers = [(np.clip(c + noise, 0.0, 1.0) * 255.0).astype(np.uint8) for c in channels]
    return np.stack(layers, axis=-1)


def floats(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def luma(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)


def detections(objects: int, per: int, seed: int = 3) -> tuple[list, list]:
    """bench_mynd.ml's detections: [objects] objects of a 1080p frame, each
    detected by [per] boxes jittered by up to a tenth of its size, as OpenCV's
    (x, y, w, h) boxes and scores."""
    rng = np.random.default_rng(seed)
    boxes, scores = [], []
    for _ in range(objects):
        cx, cy = rng.uniform(0, 1920), rng.uniform(0, 1080)
        bw, bh = 40 + rng.uniform(0, 200), 40 + rng.uniform(0, 200)
        for _ in range(per):
            x = cx + rng.uniform(-0.1, 0.1) * bw
            y = cy + rng.uniform(-0.1, 0.1) * bh
            w = bw + rng.uniform(-0.1, 0.1) * bw
            h = bh + rng.uniform(-0.1, 0.1) * bh
            boxes.append([x - w / 2, y - h / 2, w, h])
            scores.append(float(rng.uniform(0, 1)))
    return boxes, scores


def disk(r: int) -> np.ndarray:
    d = np.arange(-r, r + 1)
    return ((d[:, None] ** 2 + d[None, :] ** 2) <= r * r).astype(np.uint8)


# Timing


def time_per_call(fn: Callable[[], object]) -> float:
    """Seconds per call of fn: the fastest of REPEATS batches, each of as many
    calls as take 0.2 s. The collector stays on, as in a program."""
    timer = timeit.Timer(fn, setup=gc.enable)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=REPEATS, number=number)) / number


def format_time(seconds: float) -> str:
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if seconds >= scale:
            return f"{seconds / scale:.2f}{unit}"
    return f"{seconds * 1e9:.2f}ns"


def run(rows: Sequence[Row]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, fn in rows:
        print(f"{name:<{width}}  {format_time(time_per_call(fn))}", flush=True)


# Rows


def build_rows() -> list[Row]:
    rgb_1080 = image(1920, 1080, 7, "sunset")
    rgb_720 = image(1280, 720, 19, "forest")
    img_1080, img_720 = floats(rgb_1080), floats(rgb_720)
    gray_1080_u8 = luma(rgb_1080)
    gray_720 = luma(img_720)
    mask_720 = (gray_720 > 0.5).astype(np.uint8)
    boxes, scores = detections(96, 32)
    square, round_ = np.ones((3, 3), np.uint8), disk(2)
    turn = cv2.getRotationMatrix2D((640.0, 360.0), -10.0, 1.0)
    # Canny's thresholds: mynd's strengths 0.025 and 0.075, in OpenCV's raw
    # Sobel on uint8 codes, 8 times 255 as large.
    low, high = 0.025 * 8 * 255, 0.075 * 8 * 255

    def canny() -> np.ndarray:
        blurred = cv2.GaussianBlur(gray_1080_u8, (11, 11), 1.4)
        return cv2.Canny(blurred, low, high, L2gradient=True)

    def resize(interpolation: int) -> Callable[[], np.ndarray]:
        return lambda: cv2.resize(img_1080, (1280, 720), interpolation=interpolation)

    def sobel() -> tuple[np.ndarray, np.ndarray]:
        gx = cv2.Sobel(gray_720, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        gy = cv2.Sobel(gray_720, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        return gx, gy

    rows: list[Row] = [
        ("ToGrayscale/1080p", lambda: luma(img_1080)),
        (
            "GaussianBlur/1080p",
            lambda: cv2.GaussianBlur(img_1080, (5, 5), 1.2, borderType=cv2.BORDER_CONSTANT),
        ),
        ("Sobel/720p", sobel),
        ("Canny/1080p", canny),
        ("Resize/Box/1080p", resize(cv2.INTER_AREA)),
        ("Resize/Triangle/1080p", resize(cv2.INTER_LINEAR)),
        ("Resize/Cubic/1080p", resize(cv2.INTER_CUBIC)),
        ("Resize/Lanczos3/1080p", resize(cv2.INTER_LANCZOS4)),
        (
            "SampleAt/Triangle/720p",
            lambda: cv2.warpAffine(
                img_720, turn, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            ),
        ),
        ("Components/720p", lambda: cv2.connectedComponents(mask_720, connectivity=8)),
        ("NMS/3072", lambda: cv2.dnn.NMSBoxes(boxes, scores, 0.0, 0.5)),
        ("Median/3x3/720p", lambda: cv2.medianBlur(gray_720, 3)),
        ("Median/5x5/720p", lambda: cv2.medianBlur(gray_720, 5)),
        ("Erode/3x3/720p", lambda: cv2.erode(gray_720, square)),
        ("Erode/Disk2/720p", lambda: cv2.erode(gray_720, round_)),
    ]
    return [(f"{name} (OpenCV)", fn) for name, fn in rows]


if __name__ == "__main__":
    run(build_rows())
