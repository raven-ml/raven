from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_UBENCH_DIR = _ROOT / "vendor" / "ubench"
if str(_UBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_UBENCH_DIR))

import ubench  # type: ignore


SIZES: Sequence[int] = (50, 100, 200, 500)
DTYPES: Sequence[np.dtype] = (np.float32, np.float64)
BACKEND_NAME = "NumPy"
_RNG = np.random.default_rng(seed=0)


def _dtype_label(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    return str(dtype)


def _benchmark_name(op_name: str, size: int, dtype: np.dtype) -> str:
    return f"{op_name} {size}x{size} {_dtype_label(dtype)} ({BACKEND_NAME})"


def _numpy_operations(
    size: int, dtype: np.dtype
) -> Iterable[Tuple[str, Callable[[], None]]]:
    a = _RNG.random((size, size), dtype=dtype)
    b = _RNG.random((size, size), dtype=dtype)

    ops: List[Tuple[str, Callable[[], None]]] = [
        ("Add", lambda a=a, b=b: np.add(a, b)),
        ("Mul", lambda a=a, b=b: np.multiply(a, b)),
    ]

    ops.extend(
        [
            ("Sum", lambda a=a: np.sum(a)),
            ("Transpose", lambda a=a: np.transpose(a)),
        ]
    )
    return ops


def build_benchmarks() -> List[Any]:
    benchmarks: List[Any] = []
    for size in SIZES:
        for dtype in DTYPES:
            for op_name, fn in _numpy_operations(size, dtype):
                bench_name = _benchmark_name(op_name, size, dtype)
                benchmarks.append(ubench.bench(bench_name, fn))
    return benchmarks


def default_config() -> ubench.Config:
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
    benchmarks = build_benchmarks()
    # Mirror the OCaml defaults for fair comparisons with Nx benchmarks.
    config = default_config()
    ubench.run(benchmarks, config=config, output_format="pretty", verbose=False)


if __name__ == "__main__":
    main()
