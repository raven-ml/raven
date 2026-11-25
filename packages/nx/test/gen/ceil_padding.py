# /// script
# requires-python = "==3.13.*"
# dependencies = ["jax==0.12.1", "jaxlib==0.12.1"]
# ///
"""Generate the paddings nx's ceil_padding is held to, from JAX.

    uv run packages/nx/test/gen/ceil_padding.py [--check]

Writes `packages/nx/test/golden/ceil_padding.golden`. With --check nothing
is written, and the run fails if the golden would change.

Each case is one axis: an extent, a kernel size, a stride and a dilation.
The golden holds the (lo, hi) padding that `lax.padtype_to_pads` gives for
"SAME" over the dilated kernel, `(kernel - 1) * dilation + 1` taps, as
`lax.conv_general_dilated` passes it. Flax's and TensorFlow's SAME padding
place windows by this rule, so a model trained under either expects it.
The cases cross extents at zero, one, small, both parities and Whisper's
and ResNet's sizes with every kernel size, stride and dilation below.

A golden is a comment line, then lines of six integers:
`extent kernel stride dilation lo hi`.
"""

import argparse
import difflib
import itertools
import sys
from pathlib import Path

from jax import lax

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "ceil_padding.golden"
HEADER = "# extent kernel stride dilation lo hi, from lax.padtype_to_pads\n"

EXTENTS = [0, 1, 2, 3, 5, 8, 56, 57, 224, 3000, 3001]
KERNELS = [1, 2, 3, 4, 5, 7]
STRIDES = [1, 2, 3, 4]
DILATIONS = [1, 2, 3]


def render():
    rows = [HEADER]
    for n, k, s, d in itertools.product(EXTENTS, KERNELS, STRIDES, DILATIONS):
        taps = (k - 1) * d + 1
        ((lo, hi),) = lax.padtype_to_pads((n,), (taps,), (s,), "SAME")
        rows.append(f"{n} {k} {s} {d} {lo} {hi}\n")
    return "".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render()
    if args.check:
        old = GOLDEN.read_text() if GOLDEN.exists() else ""
        if old != text:
            diff = difflib.unified_diff(old.splitlines(), text.splitlines(), lineterm="", n=0)
            sys.stdout.write("\n".join(list(diff)[:20]) + "\n")
            sys.exit(f"{GOLDEN} would change")
        return
    GOLDEN.write_text(text)


if __name__ == "__main__":
    main()
