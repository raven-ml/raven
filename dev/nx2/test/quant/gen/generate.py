#!/usr/bin/env python3
"""Generate golden/: block-quantised tensors and ggml's values of them.

For each of Q8_0, Q4_K, Q6_K and MXFP4, golden/<type>.blocks holds the bytes of
a tensor of shape (4, 512) stored in that type, row by row, and
golden/<type>.values the float32 values, little-endian, that gguf-py,
llama.cpp's own Python package, dequantises from it. Rows hold a
zero block, a block at the largest float16 scale, blocks of negative and
subnormal scales, and blocks of random bytes under finite scales. MXFP4's
blocks hold scale bytes 0, 1, 127, 253 and 254 and random ones, never 255,
which ggml reads as 2^128 and the OCP format as NaN.

Usage:
    uv run --no-project --with gguf --with numpy \\
      python dev/nx2/test/quant/gen/generate.py
"""

import os

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

ROWS, COLS = 4, 512
OUT = os.path.join(os.path.dirname(__file__), "..", "golden")

rng = np.random.default_rng(20261002)


def half(x):
    return np.frombuffer(np.float16(x).tobytes(), dtype=np.uint8)


def scales(count, special):
    """Float16 scales: the special ones first, then random finite ones."""
    out = list(special)
    while len(out) < count:
        out.append(rng.uniform(-2.0, 2.0) * 2.0 ** rng.integers(-12, 4))
    return out[:count]


def blocks(qtype, nbytes, fields):
    """`ROWS * COLS / values` blocks of random bytes, each block's float16
    fields at `fields` (byte offsets) set from `scales`, the first block
    zero."""
    values = 32 if qtype == GGMLQuantizationType.Q8_0 else 256
    n = ROWS * COLS // values
    b = rng.integers(0, 256, size=(n, nbytes), dtype=np.uint8)
    special = [0.0, 65504.0, -65504.0, -1.5, 2.0**-24, -(2.0**-20)]
    for off in fields:
        for i, s in enumerate(scales(n, special)):
            b[i, off : off + 2] = half(s)
        # The second field of a block reads its scales in another order, so
        # that a block pairs a negative scale with a positive min.
        special = special[::-1]
    b[0] = 0
    return b.reshape(ROWS, -1)


def mxfp4_blocks():
    """MXFP4 blocks: a scale byte, the special ones first, then 16 random code
    bytes; the first block zero."""
    n = ROWS * COLS // 32
    b = rng.integers(0, 256, size=(n, 17), dtype=np.uint8)
    special = [0, 1, 127, 253, 254]
    b[:, 0] = special + list(rng.integers(100, 151, size=n - len(special)))
    b[0] = 0
    return b.reshape(ROWS, -1)


def write(name, raw, qtype):
    values = dequantize(raw, qtype).astype("<f4")
    assert values.shape == (ROWS, COLS)
    with open(os.path.join(OUT, name + ".blocks"), "wb") as f:
        f.write(raw.astype(np.uint8).tobytes())
    with open(os.path.join(OUT, name + ".values"), "wb") as f:
        f.write(values.tobytes())


def main():
    cases = [
        ("q8_0", GGMLQuantizationType.Q8_0, 34, [0]),
        ("q4_k", GGMLQuantizationType.Q4_K, 144, [0, 2]),
        ("q6_k", GGMLQuantizationType.Q6_K, 210, [208]),
    ]
    for name, qtype, nbytes, fields in cases:
        write(name, blocks(qtype, nbytes, fields), qtype)
    write("mxfp4", mxfp4_blocks(), GGMLQuantizationType.MXFP4)


if __name__ == "__main__":
    main()
