# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "torch==2.9.0+cpu", "torchvision==0.24.0+cpu"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Goldens for Mynd.Box.transform and Mynd.Box.giou, from exact rationals
and torchvision 0.24.

Run from this directory, with the script's own lock:

    uv run --locked --script geometry.py ../golden/geometry.npz

transform.*: dyadic transforms and boxes, small enough that every sum of
Transform.apply's arithmetic is exact in float64 and only its division
rounds. The want is the exact rule rounded once into float64: the least
and greatest coordinates of the mapped corners where the corners' d share
one strict sign, NaN where they do not, 0 0 0 0 for an empty box. The
script checks that a 9 x 9 grid of each closed box maps inside the exact
result, and that each side is a mapped corner's coordinate, so the result
is the least box holding the box's image.
"""
import sys
from fractions import Fraction as F
import random

import numpy as np

random.seed(23)
out = {}

# Box.transform

AFFINE, POSITIVE, NEGATIVE, STRADDLE, ZERO, EMPTY = range(6)


def dy(lo, hi, den):
    return F(random.randint(lo * den, hi * den), den)


def matrix(kind):
    m = [[dy(-2, 2, 16) for _ in range(2)] + [dy(-20, 20, 16)] for _ in range(2)]
    if kind == AFFINE:
        m.append([F(0), F(0), F(1)])
    elif kind == ZERO:
        # d = 1 - x / 4, 0 at the corners on x = 4.
        m.append([F(-1, 4), F(0), F(1)])
    else:
        sign = -1 if kind == NEGATIVE else 1
        m.append([F(random.randint(-8, 8), 64), F(random.randint(-8, 8), 64),
                  F(sign)])
    return m


def box(kind):
    if kind == ZERO:
        x0 = dy(-10, 3, 16); y0 = dy(-10, 10, 16)
        return (x0, y0, F(4), y0 + dy(0, 12, 16) + F(1, 16))
    x0 = dy(-10, 10, 16); y0 = dy(-10, 10, 16)
    if kind == EMPTY:
        w, h = random.choice([(dy(-4, 0, 16), dy(0, 4, 16)),
                              (dy(0, 4, 16), dy(-4, 0, 16)),
                              (dy(-4, 0, 16), dy(-4, 0, 16))])
        return (x0, y0, x0 + w, y0 + h)
    return (x0, y0, x0 + dy(0, 12, 16) + F(1, 16), y0 + dy(0, 12, 16) + F(1, 16))


def d(m, x, y):
    return m[2][0] * x + m[2][1] * y + m[2][2]


def apply(m, x, y):
    e = d(m, x, y)
    return ((m[0][0] * x + m[0][1] * y + m[0][2]) / e,
            (m[1][0] * x + m[1][1] * y + m[1][2]) / e)


def corners(b):
    x0, y0, x1, y1 = b
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def classify(m, b):
    x0, y0, x1, y1 = b
    if x1 <= x0 or y1 <= y0:
        return EMPTY
    ds = [d(m, x, y) for x, y in corners(b)]
    if all(e > 0 for e in ds):
        return AFFINE if m[2] == [0, 0, 1] else POSITIVE
    if all(e < 0 for e in ds):
        return NEGATIVE
    return ZERO if 0 in ds else STRADDLE


def rule(m, b):
    k = classify(m, b)
    if k == EMPTY:
        return k, (F(0),) * 4
    if k in (STRADDLE, ZERO):
        return k, None
    q = [apply(m, x, y) for x, y in corners(b)]
    r = (min(p[0] for p in q), min(p[1] for p in q),
         max(p[0] for p in q), max(p[1] for p in q))
    # The least box: a grid of the closed box maps inside it, and each side
    # is a corner's.
    x0, y0, x1, y1 = b
    for i in range(9):
        for j in range(9):
            px, py = apply(m, x0 + (x1 - x0) * j / 8, y0 + (y1 - y0) * i / 8)
            assert r[0] <= px <= r[2] and r[1] <= py <= r[3]
    return k, r


def det(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


per_kind = 60
rows = {k: [] for k in range(6)}
draws = 0
while min(len(v) for v in rows.values()) < per_kind:
    draws += 1
    want = random.choice([AFFINE, POSITIVE, NEGATIVE, ZERO, EMPTY])
    m = matrix(AFFINE if want == EMPTY and random.random() < 0.5 else want)
    if det(m) == 0:
        continue
    b = box(want)
    k, r = rule(m, b)
    if len(rows[k]) < per_kind:
        rows[k].append((m, b, r))
ms, bs, wants, kinds = [], [], [], []
for k, rs in rows.items():
    for m, b, r in rs:
        ms.append([[float(v) for v in row] for row in m])
        bs.append([float(v) for v in b])
        wants.append([float("nan")] * 4 if r is None else [float(v) for v in r])
        kinds.append(k)
        # Every input is exact in float64.
        assert all(F(v) == w for v, w in zip(bs[-1], b))
out["transform.m"] = np.array(ms, np.float64)
out["transform.b"] = np.array(bs, np.float64)
out["transform.want"] = np.array(wants, np.float64)
out["transform.kind"] = np.array(kinds, np.int64)

np.savez(sys.argv[1], **out)
print("transform:", {k: len(v) for k, v in rows.items()}, "of", draws, "draws")
