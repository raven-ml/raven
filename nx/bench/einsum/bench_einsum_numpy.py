from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, List, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_UBENCH_DIR = _ROOT / "vendor" / "ubench"
if str(_UBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_UBENCH_DIR))

import ubench  # type: ignore


SIZES: Sequence[int] = (50, 100, 200)
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


class EinsumOp:
    """Einsum operation specification."""

    def __init__(
        self,
        name: str,
        subscripts: str,
        setup: Callable[[int, np.dtype], List[np.ndarray]],
    ):
        self.name = name
        self.subscripts = subscripts
        self.setup = setup


# Define common einsum operations to benchmark - covering key use cases
EINSUM_OPS = [
    EinsumOp(
        "MatMul",
        "ij,jk->ik",
        lambda size, dtype: [
            _RNG.random((size, size), dtype=dtype),
            _RNG.random((size, size), dtype=dtype),
        ],
    ),
    EinsumOp(
        "BatchMatMul",
        "bij,bjk->bik",
        lambda size, dtype: [
            _RNG.random((4, size, size), dtype=dtype),
            _RNG.random((4, size, size), dtype=dtype),
        ],
    ),
    EinsumOp(
        "InnerProduct",
        "i,i->",
        lambda size, dtype: [
            _RNG.random(size, dtype=dtype),
            _RNG.random(size, dtype=dtype),
        ],
    ),
    # Critical contraction-reduction patterns (known to be slow in Raven)
    EinsumOp(
        "ContractReduce1",
        "ij,kj->",
        lambda size, dtype: [
            _RNG.random((size, size), dtype=dtype),
            _RNG.random((size, size), dtype=dtype),
        ],
    ),
    EinsumOp(
        "ContractReduce2",
        "ij,jk->",
        lambda size, dtype: [
            _RNG.random((size, size), dtype=dtype),
            _RNG.random((size, size), dtype=dtype),
        ],
    ),
]


def build_benchmarks() -> List[Any]:
    """Build all einsum benchmarks."""
    benchmarks: List[Any] = []

    for size in SIZES:
        for dtype in DTYPES:
            for op in EINSUM_OPS:
                operands = op.setup(size, dtype)
                bench_name = _benchmark_name(op.name, size, dtype)

                # Capture operands in closure
                def make_fn(subscripts: str, arrays: List[np.ndarray]) -> Callable[[], None]:
                    return lambda: np.einsum(subscripts, *arrays)

                benchmarks.append(ubench.bench(bench_name, make_fn(op.subscripts, operands)))

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
