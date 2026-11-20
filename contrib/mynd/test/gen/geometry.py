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

giou.p, giou.g, giou.exact: the paper's Algorithm 2 (Rezatofighi et al.,
"Generalized Intersection over Union", CVPR 2019) in exact rationals, on
the pairs it admits without ordering the prediction's coordinates: the
ground truth g not empty, the prediction p with x1 >= x0 and y1 >= y0.
Coordinates are quarters, so every area is exact in float64. giou.kind
labels each pair (KINDS below).

giou.{a,b,want}{32,64}: torchvision's generalized_box_iou on boxes that
are not empty, pairwise, in float32 and float64. Where its c - u rounds
below 0 the want is the IoU, which Box.giou's clamp of c - u at 0 gives;
giou.clamped{32,64} counts those entries.
"""
import sys
from fractions import Fraction as F
import random

import numpy as np
import torch
from torchvision.ops import generalized_box_iou
from torchvision.ops.boxes import _box_inter_union

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

# Box.giou

KINDS = ["overlapping", "disjoint", "nested", "touching", "zero width",
         "zero height", "point"]


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1]) if b[2] > b[0] and b[3] > b[1] else 0


def algorithm2(p, g):
    # Step 1 orders p's coordinates; the pairs drawn need no ordering.
    assert p[2] >= p[0] and p[3] >= p[1] and area(g) > 0
    ap = (p[2] - p[0]) * (p[3] - p[1])
    ag = (g[2] - g[0]) * (g[3] - g[1])
    xi1, xi2 = max(p[0], g[0]), min(p[2], g[2])
    yi1, yi2 = max(p[1], g[1]), min(p[3], g[3])
    i = (xi2 - xi1) * (yi2 - yi1) if xi2 > xi1 and yi2 > yi1 else 0
    ac = ((max(p[2], g[2]) - min(p[0], g[0]))
          * (max(p[3], g[3]) - min(p[1], g[1])))
    u = ap + ag - i
    return i / u - (ac - u) / ac


def kind(p, g):
    w, h = p[2] - p[0], p[3] - p[1]
    if w == 0 and h == 0:
        return "point"
    if w == 0:
        return "zero width"
    if h == 0:
        return "zero height"
    if p[0] >= g[0] and p[1] >= g[1] and p[2] <= g[2] and p[3] <= g[3]:
        return "nested"
    iw = min(p[2], g[2]) - max(p[0], g[0])
    ih = min(p[3], g[3]) - max(p[1], g[1])
    if iw > 0 and ih > 0:
        return "overlapping"
    if iw == 0 or ih == 0:
        return "touching" if iw >= 0 and ih >= 0 else "disjoint"
    return "disjoint"


def quarter(lo, hi):
    return F(random.randint(lo, hi), 4)


per_kind = 40
pairs = {k: [] for k in KINDS}
while min(len(v) for v in pairs.values()) < per_kind:
    gx, gy = quarter(-24, 24), quarter(-24, 24)
    g = (gx, gy, gx + quarter(1, 24), gy + quarter(1, 24))
    shape = random.choice(["any", "zero", "inside", "beside"])
    if shape == "inside":
        px = g[0] + quarter(0, int((g[2] - g[0]) * 4))
        py = g[1] + quarter(0, int((g[3] - g[1]) * 4))
        p = (px, py, min(px + quarter(0, 12), g[2]),
             min(py + quarter(0, 12), g[3]))
    elif shape == "beside":
        px = random.choice([g[2], g[0] - quarter(1, 12)])
        py = g[1] + quarter(-8, 8)
        right = px + quarter(1, 12) if px == g[2] else g[0]
        p = (px, py, right, py + quarter(1, 12))
    else:
        px, py = quarter(-24, 24), quarter(-24, 24)
        hi = 0 if shape == "zero" else 24
        p = (px, py, px + quarter(0, hi),
             py + quarter(0, random.choice([0, hi])))
    k = kind(p, g)
    if len(pairs[k]) < per_kind:
        pairs[k].append((p, g))
ps, gs, exact, kinds = [], [], [], []
for k, rs in pairs.items():
    for p, g in rs:
        ps.append([float(v) for v in p])
        gs.append([float(v) for v in g])
        exact.append(float(algorithm2(p, g)))
        kinds.append(KINDS.index(k))
out["giou.p"] = np.array(ps, np.float64)
out["giou.g"] = np.array(gs, np.float64)
out["giou.exact"] = np.array(exact, np.float64)
out["giou.kind"] = np.array(kinds, np.int64)

np_rng = np.random.default_rng(29)


def nonempty(n):
    # Boxes anywhere, and tiny boxes inside large ones far from the origin,
    # where c - u can round below 0.
    xy = np_rng.uniform(0, 600, (n, 2))
    wh = np_rng.uniform(0.5, 300, (n, 2))
    b = np.concatenate([xy, xy + wh], 1)
    k = n // 3
    outer = np_rng.uniform(1000, 4000, (k, 2))
    owh = np_rng.uniform(100, 400, (k, 2))
    inner = outer + np_rng.uniform(0, 1, (k, 2)) * (owh - 1)
    iwh = np_rng.uniform(1e-3, 1, (k, 2))
    b[:k] = np.concatenate([outer, outer + owh], 1)
    b[k:2 * k] = np.concatenate([inner, inner + iwh], 1)
    return b


for bits, dt in ((32, np.float32), (64, np.float64)):
    a = nonempty(96).astype(dt)
    b = nonempty(90).astype(dt)
    ta, tb = torch.tensor(a), torch.tensor(b)
    got = generalized_box_iou(ta, tb)
    inter, union = _box_inter_union(ta, tb)
    iou = inter / union
    lt = torch.min(ta[:, None, :2], tb[None, :, :2])
    rb = torch.max(ta[:, None, 2:], tb[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    c = wh[..., 0] * wh[..., 1]
    assert torch.equal(got, iou - (c - union) / c)
    clamped = (c - union) < 0
    out[f"giou.a{bits}"], out[f"giou.b{bits}"] = a, b
    out[f"giou.want{bits}"] = torch.where(clamped, iou, got).numpy()
    out[f"giou.clamped{bits}"] = np.array(int(clamped.sum()), np.int64)
    print(f"giou float{bits}: {int(clamped.sum())} of {clamped.numel()}",
          "entries clamped")

np.savez(sys.argv[1], **out)
print("transform:", {k: len(v) for k, v in rows.items()}, "of", draws, "draws")
print("giou exact:", {k: len(v) for k, v in pairs.items()})
