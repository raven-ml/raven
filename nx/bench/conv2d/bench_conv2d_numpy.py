from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np
from scipy import signal

_ROOT = Path(__file__).resolve().parents[3]
_UBENCH_DIR = _ROOT / "vendor" / "ubench"
if str(_UBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_UBENCH_DIR))

import ubench  # type: ignore


# Common CNN layer sizes: (batch, in_channels, out_channels, input_size, kernel_size)
CONFIGS: Sequence[Tuple[int, int, int, int, int]] = (
    (1, 3, 32, 64, 3),    # Small: first conv layer, single image
    (8, 32, 64, 32, 3),   # Medium: mid-layer, small batch
    (16, 64, 128, 16, 3), # Large: deep layer, larger batch
)

DTYPES: Sequence[np.dtype] = (np.float32, np.float64)
BACKEND_NAME = "NumPy"
_RNG = np.random.default_rng(seed=0)


def _dtype_label(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    return str(dtype)


def _benchmark_name(
    op_name: str,
    batch: int,
    in_ch: int,
    out_ch: int,
    img_size: int,
    kernel_size: int,
    dtype: np.dtype,
) -> str:
    return (
        f"{op_name} B{batch} C{in_ch}->{out_ch} {img_size}x{img_size} "
        f"K{kernel_size} {_dtype_label(dtype)} ({BACKEND_NAME})"
    )


class ConvSpec:
    """Conv2d operation specification."""

    def __init__(
        self,
        name: str,
        batch: int,
        in_channels: int,
        out_channels: int,
        img_size: int,
        kernel_size: int,
    ):
        self.name = name
        self.batch = batch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.kernel_size = kernel_size


def create_conv_specs() -> List[ConvSpec]:
    """Create conv2d specs from configs."""
    return [
        ConvSpec("Conv2d", batch, in_ch, out_ch, img_size, kernel_size)
        for batch, in_ch, out_ch, img_size, kernel_size in CONFIGS
    ]


def conv2d_numpy(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform 2D convolution using scipy for each batch and output channel."""
    batch, in_ch, h, w = input.shape
    out_ch, _, kh, kw = kernel.shape

    # Calculate output size (valid convolution)
    out_h = h - kh + 1
    out_w = w - kw + 1

    output = np.zeros((batch, out_ch, out_h, out_w), dtype=input.dtype)

    for b in range(batch):
        for o in range(out_ch):
            for i in range(in_ch):
                output[b, o] += signal.convolve2d(
                    input[b, i], kernel[o, i], mode='valid'
                )

    return output


def build_benchmarks() -> List[Any]:
    """Build all conv2d benchmarks."""
    benchmarks: List[Any] = []
    specs = create_conv_specs()

    for spec in specs:
        for dtype in DTYPES:
            input_shape = (spec.batch, spec.in_channels, spec.img_size, spec.img_size)
            kernel_shape = (spec.out_channels, spec.in_channels, spec.kernel_size, spec.kernel_size)

            input_arr = _RNG.random(input_shape, dtype=dtype)
            kernel_arr = _RNG.random(kernel_shape, dtype=dtype)

            bench_name = _benchmark_name(
                spec.name, spec.batch, spec.in_channels, spec.out_channels,
                spec.img_size, spec.kernel_size, dtype
            )

            def make_fn(inp: np.ndarray, kern: np.ndarray) -> Callable[[], None]:
                return lambda: conv2d_numpy(inp, kern)

            benchmarks.append(ubench.bench(bench_name, make_fn(input_arr, kernel_arr)))

    return benchmarks


def default_config() -> ubench.Config:
    """Create default benchmark configuration."""
    return (
        ubench.Config.default()
        .time_limit(1.0)
        .warmup(1)
        .min_measurements(5)
        .min_cpu(0.01)
        .geometric_scale(1.3)
        .gc_stabilization(False)
        .build()
    )


def main() -> None:
    """Main entry point."""
    benchmarks = build_benchmarks()
    config = default_config()
    ubench.run(benchmarks, config=config, output_format="pretty", verbose=False)


if __name__ == "__main__":
    main()
