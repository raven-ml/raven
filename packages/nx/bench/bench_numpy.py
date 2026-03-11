from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_ROOT = Path(__file__).resolve().parents[3]

_UBENCH_PATH = _ROOT / "vendor" / "ubench" / "ubench.py"
_UBENCH_NAME = "ubench"
_UBENCH_SPEC = importlib.util.spec_from_file_location(_UBENCH_NAME, _UBENCH_PATH)
if _UBENCH_SPEC is None or _UBENCH_SPEC.loader is None:
    raise ImportError(f"Failed to load ubench from {_UBENCH_PATH}")
ubench = importlib.util.module_from_spec(_UBENCH_SPEC)
sys.modules[_UBENCH_NAME] = ubench
_UBENCH_SPEC.loader.exec_module(ubench)  # type: ignore[assignment]


SIZES: Sequence[int] = (500, 1000)
DTYPES: Sequence[np.dtype] = (np.float32, np.float64)
BACKEND_NAME = "NumPy"
_RNG = np.random.default_rng(seed=0)
try:
    _ERF = np.erf
except AttributeError:  # pragma: no cover - compatibility with older numpy
    import math

    def _ERF(x: np.ndarray) -> np.ndarray:
        return np.vectorize(math.erf, otypes=[np.float64])(x)


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
    cond = np.less(a, b)
    indices = np.arange(size, dtype=np.int32)
    scatter_template = np.zeros_like(a)
    scatter_updates = np.arange(size, dtype=dtype)
    cast_dtype = np.float64 if dtype == np.float32 else np.float32
    pad_config = ((1, 1), (1, 1))
    kernel_size = (3, 3)
    stride = (1, 1)

    def _cumsum_flat(x: np.ndarray) -> np.ndarray:
        return np.cumsum(x, axis=None).reshape(x.shape)

    def _cumprod_flat(x: np.ndarray) -> np.ndarray:
        return np.cumprod(x, axis=None).reshape(x.shape)

    def _cummax_flat(x: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(x.reshape(-1)).reshape(x.shape)

    def _cummin_flat(x: np.ndarray) -> np.ndarray:
        return np.minimum.accumulate(x.reshape(-1)).reshape(x.shape)

    ops: List[Tuple[str, Callable[[], None]]] = [
        ("Add", lambda a=a, b=b: np.add(a, b)),
        ("Matmul", lambda a=a, b=b: np.matmul(a, b)),
        ("Sub", lambda a=a, b=b: np.subtract(a, b)),
        ("Mul", lambda a=a, b=b: np.multiply(a, b)),
        ("Div", lambda a=a, b=b: np.divide(a, b)),
        ("Mod", lambda a=a, b=b: np.mod(a, b)),
        ("Pow", lambda a=a, b=b: np.power(a, b)),
        ("Atan2", lambda a=a, b=b: np.arctan2(a, b)),
        ("Max", lambda a=a, b=b: np.maximum(a, b)),
        ("Min", lambda a=a, b=b: np.minimum(a, b)),
        ("Cmp_eq", lambda a=a, b=b: np.equal(a, b)),
        ("Cmp_ne", lambda a=a, b=b: np.not_equal(a, b)),
        ("Cmp_lt", lambda a=a, b=b: np.less(a, b)),
        ("Cmp_le", lambda a=a, b=b: np.less_equal(a, b)),
        ("Where", lambda c=cond, a=a, b=b: np.where(c, a, b)),
        ("Neg", lambda a=a: np.negative(a)),
        ("Abs", lambda a=a: np.abs(a)),
        ("Recip", lambda a=a: np.reciprocal(a)),
        ("Sqrt", lambda a=a: np.sqrt(a)),
        ("Exp", lambda a=a: np.exp(a)),
        ("Log", lambda a=a: np.log(a)),
        ("Sign", lambda a=a: np.sign(a)),
        ("Sin", lambda a=a: np.sin(a)),
        ("Cos", lambda a=a: np.cos(a)),
        ("Tan", lambda a=a: np.tan(a)),
        ("Asin", lambda a=a: np.arcsin(a)),
        ("Acos", lambda a=a: np.arccos(a)),
        ("Atan", lambda a=a: np.arctan(a)),
        ("Sinh", lambda a=a: np.sinh(a)),
        ("Cosh", lambda a=a: np.cosh(a)),
        ("Tanh", lambda a=a: np.tanh(a)),
        ("Trunc", lambda a=a: np.trunc(a)),
        ("Ceil", lambda a=a: np.ceil(a)),
        ("Floor", lambda a=a: np.floor(a)),
        ("Round", lambda a=a: np.round(a)),
        ("Erf", lambda a=a: _ERF(a)),
        ("Reduce_sum", lambda a=a: np.sum(a)),
        ("Reduce_prod", lambda a=a: np.prod(a)),
        ("Reduce_max", lambda a=a: np.max(a)),
        ("Reduce_min", lambda a=a: np.min(a)),
        ("Cum_sum", lambda a=a: _cumsum_flat(a)),
        ("Scan_sum_axis1", lambda a=a: np.cumsum(a, axis=1)),
        ("Cum_prod", lambda a=a: _cumprod_flat(a)),
        ("Scan_prod_axis1", lambda a=a: np.cumprod(a, axis=1)),
        ("Cum_max", lambda a=a: _cummax_flat(a)),
        ("Scan_max_axis1", lambda a=a: np.maximum.accumulate(a, axis=1)),
        ("Cum_min", lambda a=a: _cummin_flat(a)),
        ("Scan_min_axis1", lambda a=a: np.minimum.accumulate(a, axis=1)),
        ("Argmax", lambda a=a: np.argmax(a)),
        ("Argmin", lambda a=a: np.argmin(a)),
        ("Sort", lambda a=a: np.sort(a)),
        ("Argsort", lambda a=a: np.argsort(a)),
        ("Cat", lambda a=a, b=b: np.concatenate([a, b], axis=0)),
        (
            "Pad",
            lambda a=a, pad_config=pad_config: np.pad(
                a, pad_config, mode="constant"
            ),
        ),
        ("Shrink", lambda a=a: a[1:-1, 1:-1]),
        ("Flip", lambda a=a: np.flip(a, axis=0)),
        ("Cast", lambda a=a, dt=cast_dtype: a.astype(dt, copy=False)),
        ("Gather", lambda a=a, idx=indices: np.take(a, idx)),
        (
            "Scatter",
            lambda t=scatter_template, idx=indices, upd=scatter_updates: np.put(
                t, idx, upd
            ),
        ),
        (
            "Threefry_rand",
            lambda size=size, dtype=dtype: _RNG.random((size, size)).astype(
                dtype, copy=False
            ),
        ),
        (
            "Threefry_randn",
            lambda size=size, dtype=dtype: _RNG.standard_normal((size, size)).astype(
                dtype, copy=False
            ),
        ),
    ]
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
    ubench.run(
        benchmarks,
        config=config,
        output_format="pretty",
        sort_by_wall=False,
        verbose=False,
    )


if __name__ == "__main__":
    main()
