# /// script
# requires-python = "==3.13.*"
# dependencies = ["torch==2.15.0", "numpy==2.5.5"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Generate kaun's convolution goldens from PyTorch.

    uv run packages/kaun/test/gen/conv.py [--check]

Writes `packages/kaun/test/golden/conv.npz`, which test_conv.ml and
test_conv_transpose.ml read. With --check nothing is written: the script
exits 1 unless the golden it would write equals the committed one byte for
byte.

Inputs, weights and biases are multiples of 1/4 in [-2, 2], so every
product and sum below is exact in float64 and PyTorch's results are the
exact values, stored as float32, which holds them. Activations are stored
channels last, [n; s...; c], as kaun takes them; weights in PyTorch's
layout, [o; c/g; k...] for a convolution and [in; out/g; k...] for a
transpose, so the tests convert them as an importer does, by
Nx.moveaxis 1 (-1).
"""
import io
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "golden" / "conv.npz"

rng = np.random.default_rng(20261118)
out = {}


def quarters(*shape):
    return rng.integers(-8, 9, size=shape).astype(np.float64) / 4


def last(t):
    """[n; c; s...] to [n; s...; c]."""
    return np.ascontiguousarray(np.moveaxis(t.detach().numpy(), 1, -1))


def first(x):
    """[n; s...; c] to [n; c; s...] as a torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, 1)))


def conv(x, w, b, stride, pads, dilation, groups):
    """PyTorch's convolution of channels-last x padded by (lo, hi) pairs."""
    flat = [p for lo_hi in reversed(pads) for p in lo_hi]
    xt = F.pad(first(x), flat)
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[len(pads)]
    y = fn(xt, torch.from_numpy(w), None if b is None else torch.from_numpy(b),
           stride=stride, dilation=dilation, groups=groups)
    return last(y)


def put(name, **arrays):
    """Floats stored as float32, which holds each of them exactly."""
    for k, v in arrays.items():
        if v.dtype == np.float64:
            v32 = v.astype(np.float32)
            assert (v32.astype(np.float64) == v).all(), name + "." + k
            v = v32
        out[f"{name}.{k}"] = np.ascontiguousarray(v)


# Whisper's stem: k3 p1, then k3 s2 p1, at an even and an odd extent.
w1, b1 = quarters(3, 2, 3), quarters(3)
w2, b2 = quarters(3, 3, 3), quarters(3)
for t in (3000, 3001):
    x = quarters(1, t, 2)
    y1 = conv(x, w1, b1, (1,), [(1, 1)], (1,), 1)
    y2 = conv(y1, w2, b2, (2,), [(1, 1)], (1,), 1)
    put(f"stem{t}", x=x, w1=w1, b1=b1, w2=w2, b2=b2, y1=y1, y2=y2)

# Grouped and dilated 2-D, asymmetric pads.
x = quarters(2, 7, 9, 4)
w, b = quarters(6, 2, 3, 2), quarters(6)
y = conv(x, w, b, (2, 1), [(1, 2), (0, 1)], (2, 1), 2)
put("grouped", x=x, w=w, b=b, y=y)

# 3-D, no bias.
x = quarters(1, 4, 5, 6, 2)
w = quarters(3, 2, 2, 3, 2)
y = conv(x, w, None, (1, 2, 1), [(1, 1), (0, 0), (1, 1)], (1, 1, 1), 1)
put("volume", x=x, w=w, y=y)

# SAM 2's upscaling: a k2 s2 transpose.
x = quarters(1, 4, 4, 4)
w, b = quarters(4, 2, 2, 2), quarters(2)
y = F.conv_transpose2d(first(x), torch.from_numpy(w), torch.from_numpy(b),
                       stride=2)
put("upscale", x=x, w=w, b=b, y=last(y))

# A transpose with output_padding 1 beside padding 1: kaun's pair (1, 0).
x = quarters(1, 5, 3)
w, b = quarters(3, 2, 3), quarters(2)
y = F.conv_transpose1d(first(x), torch.from_numpy(w), torch.from_numpy(b),
                       stride=2, padding=1, output_padding=1)
put("outpad", x=x, w=w, b=b, y=last(y))

# Pooling: ResNet's stem max pool over negative values, so that a padded
# element read as 0 would show; average pools counting the padding, of
# windows a power of two so that each mean is exact.
x = -np.abs(quarters(1, 7, 8, 3)) - 0.25
y = F.max_pool2d(first(x), 3, stride=2, padding=1)
put("maxpool", x=x, y=last(y))
x = quarters(1, 7, 8, 3)
y = F.avg_pool2d(first(x), 2, stride=2, padding=1, count_include_pad=True)
put("avgpool", x=x, y=last(y))
x = quarters(1, 3, 4, 5, 2)
y = F.avg_pool3d(first(x), (2, 2, 2), stride=(1, 2, 2))
put("avgpool3", x=x, y=last(y))

buf = io.BytesIO()
np.savez(buf, **out)
data = buf.getvalue()
if "--check" in sys.argv[1:]:
    same = GOLDEN.read_bytes() == data
    print(f"{GOLDEN}: {'unchanged' if same else 'differs'}")
    sys.exit(0 if same else 1)
GOLDEN.write_bytes(data)
print(f"{len(out)} arrays to {GOLDEN}")
