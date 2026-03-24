#!/usr/bin/env python3
"""Generate tinygrad reference .expected files for grad+JIT golden tests.

Uses tinygrad's Tensor.gradient() + Tensor.schedule() to produce the
scheduled gradient kernel, then renders via the clang renderer.

Usage:
    uv run packages/rune/test/golden/jit_grad/generate_expected.py

After running, commit the generated .expected files.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "_tinygrad"
    ),
)

from tinygrad import Tensor
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.renderer.cstyle import ClangRenderer

OUT_DIR = os.path.dirname(__file__)

renderer = ClangRenderer()


def write_expected(name, content):
    path = os.path.join(OUT_DIR, f"{name}.expected")
    with open(path, "w") as f:
        f.write(content + "\n")
    print(f"  wrote {path}")


def gradient_source(f, x_shape):
    """Compute gradient of f w.r.t. x and return rendered source.

    Uses Tensor.gradient() + Tensor.schedule() for proper scheduling,
    then renders each kernel via the clang renderer.
    """
    x = Tensor.empty(*x_shape).requires_grad_(True)
    y = f(x)
    (grad_x,) = y.gradient(x)
    sched = grad_x.schedule()
    sources = []
    for item in sched:
        ast = item.ast
        rewritten = full_rewrite_to_sink(ast, renderer, optimize=True)
        lst = linearize(rewritten)
        lst = line_rewrite(lst, pm_linearize_cleanups)
        sources.append(renderer.render(lst).strip())
    return "\n---\n".join(sources)


# ── Test cases ──


def build_grad_square():
    """grad(sum(x*x)) = 2*x, shape [4]."""
    return gradient_source(lambda x: (x * x).sum(), (4,))


def build_grad_sin():
    """grad(sum(sin(x))) = cos(x), shape [4]."""
    return gradient_source(lambda x: x.sin().sum(), (4,))


def build_grad_polynomial():
    """grad(sum((x+1)*x)) = 2x+1, shape [4]."""
    return gradient_source(lambda x: ((x + 1) * x).sum(), (4,))


def build_grad_cube():
    """grad(sum(x*x*x)) = 3*x^2, shape [4]."""
    return gradient_source(lambda x: (x * x * x).sum(), (4,))


def build_grad_sum():
    """grad(sum(x)) = ones, shape [4]."""
    return gradient_source(lambda x: x.sum(), (4,))


TEST_CASES = [
    ("grad_square", build_grad_square),
    ("grad_sin", build_grad_sin),
    ("grad_polynomial", build_grad_polynomial),
    ("grad_cube", build_grad_cube),
    ("grad_sum", build_grad_sum),
]


def main():
    total = 0
    for case_name, builder in TEST_CASES:
        print(f"\n{case_name}:")
        try:
            src = builder()
            write_expected(case_name, src)
            total += 1
        except Exception as e:
            print(f"  FAIL {case_name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nDone. Generated {total} .expected files in {OUT_DIR}")


if __name__ == "__main__":
    main()
