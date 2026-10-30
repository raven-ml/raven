# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==4.14.0.94"]
# ///
"""OpenCV's side of Sowilo's bench: each row's time per call.

usage: uv run contrib/sowilo/bench/bench_sowilo.py
"""

from __future__ import annotations

import gc
import sys
import timeit
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import generate_fixtures  # noqa: E402

REPEATS = 5

Row = Tuple[str, Callable[[], object]]


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


def build_rows() -> List[Row]:
    color = generate_fixtures.images()
    color_1080 = color["1080p"]
    gray_1080 = cv2.cvtColor(color_1080, cv2.COLOR_RGB2GRAY)
    gray_720 = cv2.cvtColor(color["720p"], cv2.COLOR_RGB2GRAY)

    def bench_grayscale() -> None:
        gray = cv2.cvtColor(color_1080, cv2.COLOR_RGB2GRAY)
        int(gray.sum())

    def bench_gaussian() -> None:
        blurred = cv2.GaussianBlur(color_1080, ksize=(5, 5), sigmaX=1.2, sigmaY=1.2)
        int(blurred.sum())

    def bench_sobel() -> None:
        sobel_x = cv2.Sobel(gray_720, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
        int(sobel_x.sum())

    def bench_canny() -> None:
        edges = cv2.Canny(gray_1080, threshold1=55.0, threshold2=120.0)
        int(edges.sum())

    return [
        ("ToGrayscale/1080p (OpenCV)", bench_grayscale),
        ("GaussianBlur/1080p (OpenCV)", bench_gaussian),
        ("Sobel/720p (OpenCV)", bench_sobel),
        ("Canny/1080p (OpenCV)", bench_canny),
    ]


if __name__ == "__main__":
    run(build_rows())
