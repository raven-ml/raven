from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import torch

_ROOT = Path(__file__).resolve().parents[2]
_UBENCH_DIR = _ROOT / "vendor" / "ubench"
if str(_UBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_UBENCH_DIR))

import ubench  # type: ignore


# Benchmark sizes - focus on realistic ML workload sizes
SIZES = [
    ("Small", 100),    # Small batch/feature size
    ("Medium", 500),   # Medium neural network layer
    ("Large", 1000),   # Large neural network layer
]

BACKEND_NAME = "PyTorch"


def benchmark_name(op_name: str, size_name: str) -> str:
    """Create benchmark name."""
    return f"{op_name} {size_name} ({BACKEND_NAME})"


class ScalarGradBenchmarks:
    """Scalar→Scalar gradient: f(x) = x^2"""

    @staticmethod
    def build() -> List[Any]:
        benchmarks = []
        for size_name, _ in SIZES:
            # Create tensor outside benchmark - matching Rune's approach
            x = torch.tensor(5.0, requires_grad=True)

            def bench_fn(x_input=x):
                # Reset gradient from previous run
                if x_input.grad is not None:
                    x_input.grad.zero_()
                y = x_input ** 2
                y.backward()
                return x_input.grad

            bench_name = benchmark_name("ScalarGrad", size_name)
            benchmarks.append(ubench.bench(bench_name, bench_fn))

        return benchmarks


class VectorScalarGradBenchmarks:
    """Vector→Scalar gradient: f(x) = sum(x^2) (L2 norm squared)"""

    @staticmethod
    def build() -> List[Any]:
        benchmarks = []
        torch.manual_seed(0)

        for size_name, size in SIZES:
            # Create tensor outside benchmark - matching Rune's approach
            x = torch.randn(size, requires_grad=True)

            def bench_fn(x_input=x):
                # Reset gradient from previous run
                if x_input.grad is not None:
                    x_input.grad.zero_()
                y = torch.sum(x_input ** 2)
                y.backward()
                return x_input.grad

            bench_name = benchmark_name("VectorGrad", size_name)
            benchmarks.append(ubench.bench(bench_name, bench_fn))

        return benchmarks


class MatMulGradBenchmarks:
    """MatMul gradient: f(x) = sum(matmul(x, W))"""

    @staticmethod
    def build() -> List[Any]:
        benchmarks = []
        torch.manual_seed(1)

        for size_name, size in SIZES:
            # Create tensors outside benchmark - matching Rune's approach
            x = torch.randn(size, size, requires_grad=True)
            w = torch.randn(size, size)

            def bench_fn(x_input=x, w_input=w):
                # Reset gradient from previous run
                if x_input.grad is not None:
                    x_input.grad.zero_()
                y = torch.sum(torch.matmul(x_input, w_input))
                y.backward()
                return x_input.grad

            bench_name = benchmark_name("MatMulGrad", size_name)
            benchmarks.append(ubench.bench(bench_name, bench_fn))

        return benchmarks


class ChainGradBenchmarks:
    """Chain of operations: f(x) = sum(exp(tanh(x^2)))"""

    @staticmethod
    def build() -> List[Any]:
        benchmarks = []
        torch.manual_seed(2)

        for size_name, size in SIZES:
            # Create tensor outside benchmark - matching Rune's approach
            x = torch.randn(size, size, requires_grad=True)

            def bench_fn(x_input=x):
                # Reset gradient from previous run
                if x_input.grad is not None:
                    x_input.grad.zero_()
                y = torch.sum(torch.exp(torch.tanh(x_input ** 2)))
                y.backward()
                return x_input.grad

            bench_name = benchmark_name("ChainGrad", size_name)
            benchmarks.append(ubench.bench(bench_name, bench_fn))

        return benchmarks


class HigherOrderGradBenchmarks:
    """Higher-order gradient: grad(grad(f)) where f(x) = sum(x^3)"""

    @staticmethod
    def build() -> List[Any]:
        benchmarks = []
        torch.manual_seed(3)

        for size_name, size in SIZES:
            # Create tensor outside benchmark - matching Rune's approach
            x = torch.randn(size, requires_grad=True)

            def bench_fn(x_input=x):
                # Reset gradient from previous run
                if x_input.grad is not None:
                    x_input.grad.zero_()

                # First grad: grad(f)
                y = torch.sum(x_input ** 3)
                grad_outputs = torch.ones_like(y)
                first_grad = torch.autograd.grad(y, x_input, grad_outputs=grad_outputs, create_graph=True)[0]

                # Second grad: grad(grad(f))
                grad_sum = torch.sum(first_grad)
                second_grad = torch.autograd.grad(grad_sum, x_input)[0]

                return second_grad

            bench_name = benchmark_name("HigherOrderGrad", size_name)
            benchmarks.append(ubench.bench(bench_name, bench_fn))

        return benchmarks


def build_benchmarks() -> List[Any]:
    """Build all gradient benchmarks."""
    benchmarks = []
    benchmarks.extend(ScalarGradBenchmarks.build())
    benchmarks.extend(VectorScalarGradBenchmarks.build())
    benchmarks.extend(MatMulGradBenchmarks.build())
    benchmarks.extend(ChainGradBenchmarks.build())
    benchmarks.extend(HigherOrderGradBenchmarks.build())
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
