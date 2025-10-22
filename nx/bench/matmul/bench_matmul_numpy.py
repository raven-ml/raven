from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_UBENCH_DIR = _ROOT / "vendor" / "ubench"
if str(_UBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_UBENCH_DIR))

import ubench  # type: ignore


BACKEND_NAME = "NumPy"
DTYPES: Sequence[np.dtype] = (np.float32, np.float64)


@dataclass(frozen=True)
class MatmulCase:
    """Specification for a matrix multiplication benchmark."""

    name: str
    m: int
    k: int
    n: int
    seed: int


CASES: Sequence[MatmulCase] = (
    MatmulCase("SquareSmall", m=64, k=64, n=64, seed=11),
    MatmulCase("TallSkinny", m=256, k=64, n=256, seed=17),
    MatmulCase("Wide", m=128, k=256, n=64, seed=23),
    MatmulCase("SquareLarge", m=512, k=512, n=512, seed=29),
)


def _dtype_label(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    return str(dtype)


def _benchmark_name(case: MatmulCase, dtype: np.dtype) -> str:
    return f"MatMul {case.name} {case.m}x{case.k} @ {case.k}x{case.n} {_dtype_label(dtype)} ({BACKEND_NAME})"


def _make_operands(case: MatmulCase, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    lhs_rng = np.random.default_rng(seed=case.seed)
    rhs_rng = np.random.default_rng(seed=case.seed + 1)
    lhs = lhs_rng.random((case.m, case.k), dtype=dtype)
    rhs = rhs_rng.random((case.k, case.n), dtype=dtype)
    return lhs, rhs


def build_benchmarks() -> List[Any]:
    """Build benchmarks for NumPy matmul."""
    benchmarks: List[Any] = []

    for case in CASES:
        for dtype in DTYPES:
            lhs, rhs = _make_operands(case, dtype)

            def make_fn(a: np.ndarray, b: np.ndarray) -> Callable[[], None]:
                return lambda: np.matmul(a, b)

            benchmarks.append(ubench.bench(_benchmark_name(case, dtype), make_fn(lhs, rhs)))

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
