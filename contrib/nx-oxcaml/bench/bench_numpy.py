from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
while not (_SCRIPTS_DIR / "dune-project").exists():
    _SCRIPTS_DIR = _SCRIPTS_DIR.parent
_SCRIPTS_DIR = _SCRIPTS_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import ubench  # type: ignore


SIZES: Sequence[int] = (512,)
DTYPES: Sequence[np.dtype] = (np.dtype(np.float32), np.dtype(np.float64))
_RNG = np.random.default_rng(seed=0)


def _dtype_name(dtype: np.dtype) -> str:
    return "f32" if dtype == np.float32 else "f64"


def _operations(
    size: int, dtype: np.dtype
) -> List[Tuple[str, str, Callable[[], Any]]]:
    shape = (size, size)
    a = _RNG.random(shape, dtype=dtype)
    b = _RNG.random(shape, dtype=dtype)
    cond = np.less(a, b)
    transposed_a = a.T
    transposed_b = b.T
    offset_a = a[1:-1, :]
    offset_b = b[1:-1, :]
    indices = (np.arange(size, dtype=np.int32) * 37) % size
    cast_dtype = np.float64 if dtype == np.float32 else np.float32
    cast_name = "Cast f32 to f64" if dtype == np.float32 else "Cast f64 to f32"

    return [
        ("elementwise", "Add", lambda: np.add(a, b)),
        ("elementwise", "Sub", lambda: np.subtract(a, b)),
        ("elementwise", "Mul", lambda: np.multiply(a, b)),
        ("elementwise", "Div", lambda: np.divide(a, b)),
        ("elementwise", "Maximum", lambda: np.maximum(a, b)),
        ("elementwise", "Minimum", lambda: np.minimum(a, b)),
        ("elementwise", "Less", lambda: np.less(a, b)),
        ("elementwise", "Where", lambda: np.where(cond, a, b)),
        ("unary", "Neg", lambda: np.negative(a)),
        ("unary", "Abs", lambda: np.abs(a)),
        ("unary", "Sqrt", lambda: np.sqrt(a)),
        ("unary", "Exp", lambda: np.exp(a)),
        ("unary", "Log", lambda: np.log(a)),
        ("unary", "Sin", lambda: np.sin(a)),
        ("unary", "Cos", lambda: np.cos(a)),
        ("reduction and scan", "Sum", lambda: np.sum(a)),
        ("reduction and scan", "Sum axis 0", lambda: np.sum(a, axis=0)),
        ("reduction and scan", "Sum axis 1", lambda: np.sum(a, axis=1)),
        ("reduction and scan", "Max axis 1", lambda: np.max(a, axis=1)),
        (
            "reduction and scan",
            "Cumsum axis 1",
            lambda: np.cumsum(a, axis=1),
        ),
        (
            "reduction and scan",
            "Argmax axis 1",
            lambda: np.argmax(a, axis=1),
        ),
        ("structural", "Matmul", lambda: np.matmul(a, b)),
        (
            "structural",
            "Cat axis 0",
            lambda: np.concatenate([a, b], axis=0),
        ),
        (
            "structural",
            "Cat axis 1",
            lambda: np.concatenate([a, b], axis=1),
        ),
        (
            "structural",
            "Cat offset views axis 0",
            lambda: np.concatenate([offset_a, offset_b], axis=0),
        ),
        (
            "structural",
            "Cat transposed views axis 1",
            lambda: np.concatenate([transposed_a, transposed_b], axis=1),
        ),
        (
            "structural",
            "Contiguous transpose",
            lambda: np.ascontiguousarray(transposed_a),
        ),
        (
            "structural",
            "Pad",
            lambda: np.pad(a, ((1, 1), (1, 1)), mode="constant"),
        ),
        ("structural", cast_name, lambda: a.astype(cast_dtype)),
        ("structural", "Take rows", lambda: np.take(a, indices, axis=0)),
        ("structural", "Sort rows", lambda: np.sort(a, axis=1)),
        ("structural", "Rand", lambda: _RNG.random(shape, dtype=dtype)),
        ("structural", "Randn", lambda: _RNG.standard_normal(shape, dtype=dtype)),
    ]


def build_benchmarks() -> List[Any]:
    benchmarks: List[Any] = []
    for size in SIZES:
        for dtype in DTYPES:
            dtype_name = _dtype_name(dtype)
            for category, operation, fn in _operations(size, dtype):
                name = f"{category} / {size}x{size} {dtype_name} / {operation}"
                benchmarks.append(ubench.bench(name, fn))
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


if __name__ == "__main__":
    ubench.run(
        build_benchmarks(),
        config=default_config(),
        output_format="pretty",
        sort_by_wall=False,
        verbose=False,
    )
