# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.15.0", "mlx==0.32.4"]
# ///
"""nx.metal against PyTorch MPS and MLX on this Mac's GPU: a table of each
row's time under nx.metal, under PyTorch eager, torch.compile, MLX eager
and mx.compile, each MLX mode evaluated per call and lazily; the copy
of the row's bytes by nx.metal and by PyTorch; the bar in us, the
higher of the fastest mode's median and the faster copy's, over 0.95,
and what it comes from; the ratio, that higher median over nx.metal's
(above 1, nx.metal is faster); and the band: "meets" from 0.95, where
nx.metal is within the bar, "gap" from 0.5, "blocks" below. Not on any
alias.

The copy bounds the bar from below: the bar never asks nx.metal to run
faster than the faster copy of the row's bytes, a call's fixed cost and
the bytes at the copy's bandwidth.

nx.metal's copy is a call of nx.metal's that only moves the row's bytes:
its fixed cost and its copy's bandwidth, no bound on what the GPU can
do. PyTorch's copy of the same bytes is the same measure through
PyTorch.

usage: uv run packages/nx/bench/metal/compare.py [--exe EXE] [PATTERN...]

Run it under the Mac's timing lock, nothing else building or on the GPU.

The rows are bench_metal.exe's, selected by PATTERN as its compare
selects them. Each side is a process that makes one computation and
times its calls on request: `bench_metal.exe compare ROW row` nx.metal's
call, `bench_metal.exe compare ROW floor` its copy, and a worker of this
script the row's spec under each mode in turn. A row runs in phases,
each of two sides at once: nx.metal's copy against PyTorch's, then
nx.metal against each mode. A phase gives each side four unmeasured
calls, compiling where the mode compiles; 150 ms of the sides'
measurements, alternating, bring up the GPU's clock; then a side's time
over its next measurements gives the calls each of its measurements
makes, spanning 10 ms; then 8 to 12 rounds, about 0.5 s, of one
measurement a side, in order and in reverse by turns (ABBA), so that
the machine's drift weighs on both sides alike. A measurement is the
host's time from an idle GPU to the end of its last call, over its
calls. Each cell is the median [min max] of a side's measurements;
nx.metal's pool its four phases.

Calls are chained (bench_metal.ml's header): each reads the previous
call's output, so a time is one call's, never calls overlapping. Where a
row's output has the shape and dtype of an operand, it takes that
operand's place in the next call; a mode fails unless its first output
has that operand's strides too (PyTorch; MLX shows none), so that every
call does the first's work. A product's or a transform's output takes
its operand's place whatever its shape: a larger output's first
elements, a view, in the operand's shape; a smaller output written over
the operand's first elements by a copy (PyTorch) or joined to the first
operand's tail (MLX), the glue. A glued mode's calls are timed in the
same rounds with the glue run twice, as a third side; each round's time
of a call is its time with one glue less the second glue's cost, twice
the first time less the second, and the cell says "less glue" and that
cost's median. The cost is taken in place, beside the product: a glue
timed alone pays its host's issue, which a chained call overlaps with
the product's GPU work (Q8_0 Rows' MLX concatenate took 29 us a call
alone, 3 to 8 in place). It holds where one more glue costs what the
first does. Where joining would cost more than the call, or would feed
the call other data than the first (a sort of sorted values), MLX's
calls are chained by a dependence instead: the next call's operand is
the operand, chosen against the operand plus one by whether the output's
first element is NaN, so the same bits where it is finite: a one-element
read and a pass over the operand, taken out as a glue. An r2c's and a
c2r's calls, whose output is of another dtype than their operand, are
chained by that dependence in both libraries. Two measurements
make each round's time, so a glued cell's spread is wider than its
mode's: under load, one Q8_0 Rows round read 1.6 us. Where an output
takes no operand's place (a reduction, a cast, a transform of a view),
calls are timed as issued, and whether that is one call's time depends
on the library. PyTorch runs them one at a time, as rig.metal runs
nx.metal's: on an M1 Max under torch 2.15.0, argmax, logsumexp, moments
and sqdiff over 1024x1024, window2 over 1x16x256x256, the bfloat16 sum
over 2048x1024 and the cast of 1M elements took 0.94-1.10 as long issued
as when each call waited on the one before through a one-element copy of
its output into its operand, that copy's own time taken out. MLX may
overlap them: its encoder runs kernels with no dependence between them
concurrently, and its unchained maps at 64M took 0.85-0.98 of its
chained ones. Its gathers issued took 0.76-0.92 as long as chained by a
dependence (gather_emb, take_along, ranges, ranges_elem, float32, 04:29
on 2026-11-20). No probe has timed its reductions, so its time there,
marked "*", is no bar; leaving it out can only raise a ratio. No probe
has timed PyTorch's transform of a view issued against waited. A mode
that fails prints its reason, and a row neither library computes takes
its own bound (below).

MLX's modes run two ways. `mlx` and `mx.compile` evaluate each call as
it is issued (async_eval). `mlx-lazy` and `mx.compile-lazy` issue a
chain's calls without evaluating them and evaluate the last output at
the measurement's end, or once the calls pending would hold
WINDOW_BYTES, so that MLX encodes the chain in one pass. A lazy mode
takes chained rows only: of unchained calls, it would compute the last.
The bar takes each reference in its faster way.

The references: a contraction is a matmul, b stored n x k read through
its transposed view; a Rows call is MLX's gather_mm; a decode is MLX's
quantized_matmul, or gather_qmm over the row's experts, MXFP4 by its
mxfp4 form and Q8_0 by its 8-bit affine one (QUANT); a transform is
fftn over the row's axes; an attention is scaled_dot_product_attention
at scale 1/sqrt(d), of no lse, a window or lengths a boolean mask,
PyTorch's causal mask aligned top-left, Nx.causal's at tq = tk alone. A
paged row's references read its keys dense: neither library reads a
page table. pad1 is a pad with the far row and column cut;
maxchain1024 the same 1024 maximums, one call each where nothing
compiles. A strided copy (copy_rows, copy_step2) is PyTorch's clone and
MLX's contiguous of the view. An LU is PyTorch's lu_factor with
lu_unpack's L and U, and MLX's lu on its CPU stream, each the factors
nx.metal writes: L, U and the rows' order (PyTorch's as pivots, which
lu_unpack would expand to an m x m matrix). eigh, eigvalsh and svdvals
are torch.linalg's on MPS and MLX's eigh, eigvalsh and svd without
vectors on its CPU stream, since MLX's linalg has no GPU kernels: their
bar is the faster of PyTorch on the GPU and MLX on the CPU. Each runs
unchained: an output fed back as the operand would change the spectrum
the row times. A move is the library's gather, take along an axis,
concatenation or pad. Its indices are int64 in PyTorch, as in nx.metal's
rows, and int32 in MLX, its default, so MLX's reads fewer bytes. The
gathers whose output is smaller than their operand run as issued in
PyTorch and chained by a dependence through their indices in MLX:
joining the output to the operand would cost more than the call. A sort
runs as issued in PyTorch and chained by a dependence through its
operand in MLX, so every call sorts the same unsorted values: MLX's sort
of 16M sorted floats took 0.60 as long as of random ones. The ranges are
drawn here, as rows.ml draws them but by another generator, so their
size differs from the row's by up to about 1%. Three are the composites
a user writes, where neither library has the operation: a range gather
is each position's cell (a row, or an element in ranges_elem), built
from the starts and ends, then the gather; a product of mixed dtypes,
which nx.metal converts in its loads, is each operand cast to the
output's dtype and then the product; a cumulative moments scan is the
running sums of the values and their squares, a cheaper and less exact
algorithm than nx.metal's. Neither library has a gathered product or a
group-wise quantized product in PyTorch, Q6_K, uint4 or float4_e2m1fn,
or float8_e4m3fn in MLX, and MLX copies no contiguous view (copy_at1):
those modes fail and say so. A row neither library computes (`bounded`)
is held to its own bound, its bar the faster copy of its bytes or its
product's operations at PEAK_FLOPS, whichever takes longer, named "copy
bound" or "FLOP bound". Any other row whose references all fail, or a
reference that fails for another reason than its library's lacking the
operation, is an error: its bar names the failures, and the run exits
nonzero.

Where the work differs, the reference does more or the difference is
noise: PyTorch's cummax also writes int64 positions, its bfloat16 sum
may cast its operand to float32 first, MLX eager's pad9 is a pad and
eight adds, and nx.metal's argmax writes values beside positions,
4 bytes a row more than PyTorch's. MLX's 8-bit affine weight holds a
scale and a bias a group, 40 bytes a group at float32 against Q8_0's 34.
MLX quantizes a weight stored k x n along n, where nx.metal's blocks run
along k: the same bytes, grouped otherwise. MLX accumulates float16 and
bfloat16 products in float32, as nx.metal does; PyTorch's MPS matmul
does not state its accumulator.

A chained reference computes on its own outputs, so exp, fused4,
sqcumsum, expcumsum_axis0 and pad9 overflow to infinities and NaN within
a few to about 40 calls, where nx.metal's calls read the same finite
operands every time. The GPU's arithmetic is taken to cost the same on
any value; no time here tests it.

torch.compile compiles each row for its shapes alone (dynamic=False,
after a reset): a shape change otherwise recompiles a generic kernel.
mx.compile compiles for the row's shapes. Each row's processes start
fresh, so the memory of one row's operands never pages another's.

Memory stays bounded. A measurement ends once its calls are done, so a
mode holds its operands and at most its calls' arrays. MLX's host runs
far ahead of the GPU and MLX keeps a call's arrays until its command
buffer completes, so a call waits for the oldest output once the calls
in flight hold 512 MiB (WINDOW_BYTES), their size taken from the first
call; MLX's limit (LIVE_BYTES) stops it past 4 GiB, should a mode hold
more. Operands are drawn in parts and the worker releases a mode before
it makes the next. "Peak MiB" is the most memory a row held at once:
over its phases, the largest sum of its two sides' peak footprints, GPU
buffers included, as the kernel counts them against the machine's
memory. The run stops before a row unless the kernel reports normal
memory pressure.
"""

import collections
import ctypes
import math
import os
import platform
import select
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
EXE = os.path.join(ROOT,
                   "_build/default/packages/nx/bench/metal/bench_metal.exe")

# The protocol. A side's measurement is its calls back to back, enough to
# span SPAN_S, counted once RAMP_S of measurements, unmeasured, have
# brought up the GPU's clock: an idle M1 Max takes about 1.2 s of work to
# reach it, and a timed hold starts idle. Then rounds of one measurement
# a side, about PHASE_S of them, a multiple of four. A clock still rising
# would slow both sides' first rounds alike, and show in their spreads.
WARM = 4
SPAN_S = 10e-3
RAMP_S = 1.5
PHASE_S = 0.5
MIN_ROUNDS = 8
MAX_ROUNDS = 12
MAX_RUNS = 10_000

COPIES = ("nx.metal copy", "torch copy")
MODES = ("torch", "torch.compile", "mlx", "mlx-lazy", "mx.compile",
         "mx.compile-lazy")

# The elements an operand is drawn by at a time: each library draws
# through float32 temporaries of the operand's size.
DRAW = 1 << 22

# The arrays of MLX's calls in flight: MLX keeps a scheduled call's
# arrays until its command buffer completes, and its host runs far ahead
# of the GPU. Waiting on the oldest output cost MLX's add about 1 %, at
# 1M and at 128M.
WINDOW_BYTES = 1 << 29

# The memory MLX may keep live, and in its cache of freed buffers: past
# it, MLX waits for its oldest command buffer before it allocates more.
# Without it MLX schedules until 1.5 times the GPU's recommended working
# set, 30 GiB on a 32 GiB M1 Max. No row reaches it. A limit a row
# reaches stalls MLX: at 1 GiB its eager sqdiff 8192x8192, whose call
# holds about 1 GiB, took 3.4 times as long, and a cache smaller than the
# outputs in flight made it allocate afresh each call, its 64M maps at
# 1.5 to 5 times their time.
LIVE_BYTES = 4 << 30


class Unsupported(Exception):
    pass


def shape(text):
    """A row's dims: `scalar`, or extents joined by `x`."""
    return () if text == "scalar" else tuple(int(e) for e in text.split("x"))


def dtype(spec):
    """The operands' dtype: `loop OP DTYPE SHAPE` and `move OP DTYPE`, else
    `KIND DTYPE ...`."""
    return spec[2] if spec[0] in ("loop", "move") else spec[1]


def ids(text):
    return [int(i) for i in text.split(",")]


def contraction(spec):
    """`contract DTYPE BDTYPE ACC OUT BATCH M N K LALB`'s dtypes of a, b
    and the output, a's shape, b's shape as stored, and whether b is
    stored n x k. Operands of mixed dtypes take the composite a user
    writes: each cast to the output's dtype, then the product, where that
    dtype is the accumulator's. Raises Unsupported where neither library
    computes the row's product: another accumulator or output, or a
    stored k x m."""
    _, da, db, acc, out, batch, m, n, k, lalb = spec
    if acc != ("complex64" if da == "complex64" else "float32"):
        raise Unsupported(f"no {acc} accumulator")
    if not da == db == out and out != acc:
        raise Unsupported(f"no {da} by {db} product into {out}")
    if lalb[0] != "k":
        raise Unsupported("no a stored k x m")
    batch, m, n, k = int(batch), int(m), int(n), int(k)
    lead = (batch,) if batch > 1 else ()
    nk = lalb[1] == "k"
    return da, db, out, lead + (m, k), lead + ((n, k) if nk else (k, n)), nk


# MLX's products by a quantized weight, by the format the row's weight is
# stored in. Q8_0's nearest is MLX's 8-bit affine form: a scale and a bias
# a group of 32, in x's dtype, so 40 bytes a group at float32 against
# Q8_0's 34. Neither library has Q6_K.
QUANT = {"mxfp4": dict(group_size=32, bits=4, mode="mxfp4"),
         "q8_0": dict(group_size=32, bits=8, mode="affine")}


# A row as (f, operands, chain): f the computation, of the operands; chain
# the index of the operand the output takes the place of in the next call
# (see `link`), or None.


# The bounds an integer or boolean operand's elements are drawn between,
# uniformly, the upper one excluded: its dtype's range, int64's cut to
# int32's, the draws' temporaries.
INT_RANGE = {"bool": (0, 2), "uint8": (0, 1 << 8), "int8": (-(1 << 7), 1 << 7),
             "uint16": (0, 1 << 16), "int32": (-(1 << 31), 1 << 31),
             "int64": (-(1 << 31), 1 << 31)}

# A cast's target dtype by the row's op, plain or of a transposed view
# (`t_` and the cast's name).
CASTS = {"cast_bf16": "bfloat16", "cast_f16": "float16", "cast_f32": "float32",
         "cast_i8": "int8", "cast_e4m3": "float8_e4m3fn", "cast_i32": "int32",
         "cast_i64": "int64", "cast_u8": "uint8"}


# The moves' operands, as rows.ml's moves make them (their specs, `move
# OP DTYPE`, carry no shapes): a gather of 16384 random rows of an
# embedding [65536; 1024]; a random gather of each of 16M elements; 64
# random positions of each row of [4096; 4096]; 8192 runs of 0 to 31 rows
# of [262144; 256], and 2^20 runs of 0 to 31 elements of [2^24]
# (ranges_elem); 8 blocks of [2048; 4096] stacked; [8192; 8192] padded by
# 4 on each side with 1.5; the fold of 3 x 3 windows, step 1, padded by
# 1, of [32; 64; 64; 64; 3; 3] into [32; 64; 64; 64]; the fold of
# 16 x 16 patches at step 16 into [16; 64; 256; 256] (fold_patch); and the
# fold of 3 x 3 windows, step 2, padded by 1, of [32; 64; 64; 64; 3; 3]
# into [32; 64; 128; 128] (fold_s2).
EMB_ROWS, EMB_COLS, EMB_IDS = 65536, 1024, 16384
RAND_N = 1 << 24
TAKE_ROWS, TAKE_COLS, TAKE_K = 4096, 4096, 64
RUN_MOST = 32
RANGES = {"ranges": (8192, (262144, 256)),
          "ranges_elem": (1 << 20, (1 << 24,))}
CAT_PIECES, CAT_ROWS, CAT_COLS = 8, 2048, 4096
PAD_N, PAD_BY, PAD_FILL = 8192, 4, 1.5
FOLD_N, FOLD_C, FOLD_HW, FOLD_K = 32, 64, 64, 3
PATCH_N, PATCH_C, PATCH_HW, PATCH_K = 16, 64, 256, 16

# The spectral routines' specs, `KIND DTYPE MxN`.
SPECTRAL = ("eigh", "eigvalsh", "svdvals")


def range_runs(runs, n):
    """RUNS ranges' starts, their ends (the running sums of their lengths)
    and their size, drawn on the host: lengths below RUN_MOST, starts
    whose runs stay inside N cells."""
    import random

    r = random.Random(3)
    lens = [r.randrange(RUN_MOST) for _ in range(runs)]
    starts = [r.randrange(n - RUN_MOST) for _ in range(runs)]
    ends = list(__import__("itertools").accumulate(lens))
    return starts, ends, ends[-1]


def dtype_name(dt):
    """A library's dtype's name: `torch.uint8` and `mlx.core.uint8` are
    `uint8`."""
    return str(dt).rsplit(".", 1)[1]


def torch_dtype(name):
    import torch

    dts = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16, "complex64": torch.complex64,
           "float8_e4m3fn": torch.float8_e4m3fn, "bool": torch.bool,
           "uint8": torch.uint8, "uint16": torch.uint16, "int8": torch.int8,
           "int32": torch.int32, "int64": torch.int64}
    if name not in dts:
        raise Unsupported(f"no {name}")
    return dts[name]


def torch_draw(s, dt):
    """A fresh array of shape S and dtype DT on the GPU, drawn DRAW at a
    time: a float's elements, a complex one's parts, standard normal, an
    integer's or a boolean's uniform in INT_RANGE."""
    import torch

    x = torch.empty(s, device="mps", dtype=dt)
    flat = torch.view_as_real(x) if dt.is_complex else x
    bounds = INT_RANGE.get(dtype_name(dt))
    for part in flat.view(-1).split(DRAW):
        if bounds:
            part.copy_(torch.randint(*bounds, part.shape, device="mps"))
        elif part.dtype in (torch.float32, torch.float16, torch.bfloat16):
            part.normal_()
        else:
            part.copy_(torch.randn(part.shape, device="mps"))
    return x


def torch_contract(spec):
    da, db, out, sa, sb, nk = contraction(spec)
    a, b = torch_draw(sa, torch_dtype(da)), torch_draw(sb, torch_dtype(db))
    if da == db == out:
        if nk:
            return (lambda a, b: a @ b.mT), [a, b], 0
        return (lambda a, b: a @ b), [a, b], 0
    # The output's dtype is neither operand's: no output takes a place.
    to = torch_dtype(out)
    if nk:
        return (lambda a, b: a.to(to) @ b.to(to).mT), [a, b], None
    return (lambda a, b: a.to(to) @ b.to(to)), [a, b], None


def torch_move(op, dt):
    import torch
    import torch.nn.functional as F

    def ints(n, hi):
        return torch.randint(0, hi, (n,), device="mps")

    if op == "gather_emb":
        x = torch_draw((EMB_ROWS, EMB_COLS), dt)
        return (lambda x, i: x[i]), [x, ints(EMB_IDS, EMB_ROWS)], None
    if op == "gather_rand":
        return ((lambda x, i: x[i]),
                [torch_draw((RAND_N,), dt), ints(RAND_N, RAND_N)], 0)
    if op == "take_along":
        i = ints(TAKE_ROWS * TAKE_K, TAKE_COLS).view(TAKE_ROWS, TAKE_K)
        return ((lambda x, i: torch.take_along_dim(x, i, dim=1)),
                [torch_draw((TAKE_ROWS, TAKE_COLS), dt), i], None)
    if op in RANGES:
        # The composite a user writes from starts and ends: each position's
        # cell, its run's start less the run's first position plus the
        # position, then the gather.
        runs, xs = RANGES[op]
        starts, ends, size = range_runs(runs, xs[0])
        first = [0] + ends[:-1]
        base = torch.tensor([s - f for s, f in zip(starts, first)],
                            device="mps")
        lens = torch.tensor([e - f for e, f in zip(ends, first)],
                            device="mps")
        at = torch.arange(size, device="mps")

        def ranges(x, base, lens):
            rows = torch.repeat_interleave(base, lens, output_size=size) + at
            return x[rows]

        return ranges, [torch_draw(xs, dt), base, lens], None
    if op == "assemble_cat":
        pieces = [torch_draw((CAT_ROWS, CAT_COLS), dt)
                  for _ in range(CAT_PIECES)]
        return (lambda *ps: torch.cat(ps)), pieces, 0
    if op == "assemble_pad":
        b = PAD_BY
        return ((lambda x: F.pad(x, (b, b, b, b), value=PAD_FILL)),
                [torch_draw((PAD_N, PAD_N), dt)], 0)
    if op == "fold_col2im":
        # The windows' columns in the layout F.fold reads, (N, C k k, L),
        # one column a window position.
        k = FOLD_K
        cols = torch_draw((FOLD_N, FOLD_C * k * k, FOLD_HW * FOLD_HW), dt)
        return ((lambda c: F.fold(c, (FOLD_HW, FOLD_HW), k, padding=1)),
                [cols], None)
    if op == "fold_patch":
        k, n = PATCH_K, PATCH_HW // PATCH_K
        cols = torch_draw((PATCH_N, PATCH_C * k * k, n * n), dt)
        return ((lambda c: F.fold(c, (PATCH_HW, PATCH_HW), k, stride=k)),
                [cols], None)
    if op == "fold_s2":
        k, hw = FOLD_K, 2 * FOLD_HW
        cols = torch_draw((FOLD_N, FOLD_C * k * k, FOLD_HW * FOLD_HW), dt)
        return ((lambda c: F.fold(c, (hw, hw), k, padding=1, stride=2)),
                [cols], None)
    raise Unsupported(f"no {op}")


def torch_fft(spec):
    import torch

    kind, name, dims, axes, *held = spec
    dt, s, axes = torch_dtype(name), shape(dims), ids(axes)
    # An r2c's or a c2r's output is of another dtype than its operand, so
    # it reaches the next call through a dependence.
    if kind == "rfft":
        return ((lambda x: torch.fft.rfftn(x, dim=axes)), [torch_draw(s, dt)],
                Dep(0))
    if kind == "irfft":
        n = int(held[0])
        return ((lambda x: torch.fft.irfft(x, n=n, dim=axes[-1])),
                [torch_draw(s, dt)], Dep(0))

    def f(x):
        return torch.fft.fftn(x, dim=axes)

    if not held:
        return f, [torch_draw(s, dt)], 0
    # No output has the view's layout: calls run as issued.
    whole = torch_draw(shape(held[0]), dt)
    return f, [whole[tuple(slice(0, e) for e in s)]], None


def torch_sort(spec):
    import torch

    _, name, dims, axis, keep = spec
    s, axis, keep = shape(dims), int(axis), int(keep)
    ints = {"int64": torch.int64, "int32": torch.int32, "int16": torch.int16,
            "int8": torch.int8, "uint8": torch.uint8}
    if name in ints:
        x = torch.empty(*s, device="mps", dtype=ints[name]).random_()
    else:
        x = torch_draw(s, torch_dtype(name))

    # A stable sort's values and int64 indices, or the [keep] least in
    # order, as nx.metal writes them. Calls run as issued: a sorted
    # output read again would sort sorted data.
    def f(x):
        if keep == s[axis]:
            return torch.sort(x, dim=axis, stable=True)
        return torch.topk(x, keep, dim=axis, largest=False, sorted=True)

    return f, [x], None
def attention_fields(spec):
    """`attention DTYPE B HQ HK TQ TK D MASK`'s dtype, its extents
    (b, hq, hk, tq, tk, d) and its mask."""
    _, name, *extents, mask = spec
    return name, tuple(int(e) for e in extents), mask


def key_lengths(b, tk):
    """Each batch element's keys under the mask `lengths`."""
    return [(i + 1) * tk // b for i in range(b)]


def torch_attention(spec):
    import torch
    import torch.nn.functional as F

    name, (b, hq, hk, tq, tk, d), mask = attention_fields(spec)
    dt = torch_dtype(name)
    if mask == "causal" and tq != tk:
        raise Unsupported("no causal mask aligned bottom-right")
    q = torch_draw((b, hq, tq, d), dt)
    k, v = torch_draw((b, hk, tk, d), dt), torch_draw((b, hk, tk, d), dt)
    m = None
    if mask == "lengths":
        lens = torch.tensor(key_lengths(b, tk), device="mps")
        m = (torch.arange(tk, device="mps") < lens[:, None]).view(b, 1, 1, tk)
    if mask == "window128":
        pos = torch.arange(tq, device="mps")[:, None] + (tk - tq)
        keys = torch.arange(tk, device="mps")[None, :]
        m = ((keys <= pos) & (pos < keys + 128)).view(1, 1, tq, tk)
    causal, gqa, scale = mask == "causal", hq != hk, 1 / math.sqrt(d)

    def f(q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=m, is_causal=causal, scale=scale,
            enable_gqa=gqa)

    return f, [q, k, v], 0


def torch_row(spec):
    import torch
    import torch.nn.functional as F

    kind, *fields = spec
    if kind == "contract":
        return torch_contract(spec)
    if kind == "attention":
        return torch_attention(spec)
    if kind == "rows":
        raise Unsupported("no gathered product")
    if kind == "decode":
        raise Unsupported(f"no {fields[0]} product")
    if kind in ("fft", "rfft", "irfft"):
        return torch_fft(spec)
    if kind == "sort":
        return torch_sort(spec)
    dt = torch_dtype(dtype(spec))
    if kind == "lu":
        # L and U unpacked from the packed factors, and the pivots; the
        # chain factors the last call's L, which has the operand's shape
        # where it has no more columns than rows, and its row-major
        # strides. The packed factors come column-major, so torch copies
        # the row-major operand to that order first, as nx.metal does.
        a = torch_draw(shape(fields[1]), dt)

        def lu(a):
            f = torch.linalg.lu_factor(a)
            return torch.lu_unpack(f.LU, f.pivots, unpack_pivots=False)[1]

        return lu, [a], 0
    if kind in SPECTRAL:
        # eigh and eigvalsh read the operand's lower triangle, as nx.metal's
        # rows do.
        f = {"eigh": torch.linalg.eigh, "eigvalsh": torch.linalg.eigvalsh,
             "svdvals": torch.linalg.svdvals}[kind]
        return f, [torch_draw(shape(fields[1]), dt)], None
    if kind == "move":
        return torch_move(fields[0], dt)

    def draw(s):
        return torch_draw(s, dt)

    if kind in ("reduce", "scan"):
        _, monoid, dims, axes = fields
        s, axis = shape(dims), int(axes.split(",")[0])
        x = draw(s)
        if kind == "reduce" and monoid == "sum" and "," not in axes:
            acc = torch.float32
            return (lambda x: torch.sum(x, dim=axis, dtype=acc)), [x], None
        if kind == "scan" and monoid == "max":
            return (lambda x: torch.cummax(x, dim=axis).values), [x], 0
        raise Unsupported(f"no {monoid} {kind}")
    op, _, dims = fields
    s = shape(dims)
    if op == "copy_at1":
        # The elements from position 1 on, into a fresh array.
        return (lambda x: x[1:].clone()), [draw((s[0] + 1,))], None
    if op == "copy_rows":
        # The first s[1] elements of each row of 256, into a fresh array.
        return (lambda x: x[:, :s[1]].clone()), [draw((s[0], 256))], None
    if op == "copy_step2":
        # Every other element, into a fresh array.
        return (lambda x: x[::2].clone()), [draw((2 * s[0],))], None
    if op == "maxchain1024":
        def chain(x):
            y = x
            for _ in range(1024):
                y = torch.maximum(y, x)
            return y

        return chain, [draw(s)], 0
    x, y = draw(s), draw(s)
    if op == "add":
        return (lambda x, y: x + y), [x, y], 0
    if op == "exp":
        return torch.exp, [x], 0
    if op in CASTS:
        to = torch_dtype(CASTS[op])
        return (lambda x: x.to(to)), [x], None
    if op == "bcast_add":
        return (lambda x, v: x + v), [x, draw(s[-1:])], 0
    if op == "t_add":
        # y first: an output takes its first operand's layout, so the
        # chained output keeps y's and every call reads t transposed.
        return (lambda t, y: y + t.T), [draw(s[::-1]), y], 1
    if op[2:] in CASTS and op.startswith("t_"):
        # In C order: .to keeps a view's strides by default.
        to = torch_dtype(CASTS[op[2:]])
        cast = (lambda t: t.T.to(to, memory_format=torch.contiguous_format))
        return cast, [draw(s[::-1])], None
    if op == "fused4":
        return (lambda x, y: x * y + torch.exp(x) - 0.5), [x, y], 0
    if op == "pad1":
        # Shifted a row and a column, zeros in front: a negative pad crops.
        return (lambda x: F.pad(x, (1, -1, 1, -1))), [x], 0
    if op == "pad9" and dt.is_floating_point:
        # The 3x3 sum at padding 1: zeros count, nothing divides.
        return (lambda x: F.avg_pool2d(x, 3, 1, 1, divisor_override=1)), [x], 0
    if op == "pad9":
        # No pooling of integers: the pad and the sum of its 9 taps.
        h, w = s[-2:]

        def pad9(x):
            p = F.pad(x, (1, 1, 1, 1))
            taps = [p[..., i:i + h, j:j + w] for i in range(3) for j in range(3)]
            out = taps[0]
            for t in taps[1:]:
                out = out + t
            return out

        return pad9, [x], 0
    if op == "where_lt":
        return (lambda x, y: torch.where(x < y, x, y)), [x, y], 0
    if op == "argmax":
        return (lambda x: torch.argmax(x, -1)), [x], None
    if op == "logsumexp":
        return (lambda x: torch.logsumexp(x, -1)), [x], None
    if op == "moments":
        return (lambda x: torch.var_mean(x, -1, correction=0)), [x], None
    if op == "window2":
        return (lambda x: F.avg_pool2d(x, 2, divisor_override=1)), [x], None
    if op == "sqdiff":
        return (lambda x, y: ((x - y) ** 2).sum(-1)), [x, y], None
    if op == "logcumsumexp":
        return (lambda x: torch.logcumsumexp(x, -1)), [x], 0
    if op == "sqcumsum":
        return (lambda x: torch.cumsum(x * x, -1)), [x], 0
    if op == "expcumsum_axis0":
        return (lambda x: torch.cumsum(torch.exp(x), 0)), [x], 0
    if op == "cummoments_explog16":
        # No moments scan: the population mean and variance from the
        # running sums of y and y², y the 16 exps each followed by a log.
        count = torch.arange(1, s[-1] + 1, device="mps", dtype=dt)

        def cummoments(x):
            y = x
            for _ in range(16):
                y = torch.log(torch.exp(y))
            mean = torch.cumsum(y, -1) / count
            return mean, torch.cumsum(y * y, -1) / count - mean * mean

        return cummoments, [x], None
    raise Unsupported(f"no {op}")


def mlx_dtype(name):
    import mlx.core as mx

    dts = {"float32": mx.float32, "float16": mx.float16,
           "bfloat16": mx.bfloat16, "complex64": mx.complex64,
           "bool": mx.bool_, "uint8": mx.uint8, "uint16": mx.uint16,
           "int8": mx.int8, "int32": mx.int32, "int64": mx.int64}
    if name not in dts:
        raise Unsupported(f"no {name}")
    return dts[name]


def mlx_draw(s, dt, key):
    """As torch_draw, from the random key in KEY, which it advances."""
    import mlx.core as mx

    pairs = dt == mx.complex64
    bounds = INT_RANGE.get(dtype_name(dt))
    n = math.prod(s) * (2 if pairs else 1)
    parts = []
    for i in range(0, n, DRAW):
        key[0], k = mx.random.split(key[0])
        m = min(DRAW, n - i)
        if bounds:
            part = mx.random.randint(*bounds, (m,), key=k).astype(dt)
        else:
            real = mx.float32 if pairs else dt
            part = mx.random.normal((m,), dtype=real, key=k)
        parts.append(part)
        mx.eval(parts[-1])
    x = mx.concatenate(parts)
    if pairs:
        x = mx.view(x, mx.complex64)
    x = x.reshape(s)
    mx.eval(x)
    return x


def mlx_contract(spec, key):
    import mlx.core as mx

    da, db, out, sa, sb, nk = contraction(spec)
    a, b = mlx_draw(sa, mlx_dtype(da), key), mlx_draw(sb, mlx_dtype(db), key)

    def t(b):
        return mx.swapaxes(b, -1, -2) if nk else b

    if da == db == out:
        return (lambda a, b: a @ t(b)), [a, b], 0
    to = mlx_dtype(out)
    return (lambda a, b: a.astype(to) @ t(b.astype(to))), [a, b], None


def mlx_rows(spec, key):
    """`rows DTYPE G T N K LALB IDS`: MLX's gather_mm, a's row i, as a
    1 x K matrix, by the group IDS names."""
    import mlx.core as mx

    _, name, g, t, n, k, lalb, groups = spec
    if lalb[0] != "k":
        raise Unsupported("no a stored k x t")
    dt = mlx_dtype(name)
    g, t, n, k = int(g), int(t), int(n), int(k)
    idx = mx.array(ids(groups), dtype=mx.uint32)
    a = mlx_draw((t, 1, k), dt, key)
    if lalb[1] == "k":
        b = mx.swapaxes(mlx_draw((g, n, k), dt, key), -1, -2)
    else:
        b = mlx_draw((g, k, n), dt, key)
    return (lambda a, b: mx.gather_mm(a, b, rhs_indices=idx)), [a, b], 0


def mlx_quantized(lead, s, dt, q, key):
    """Weights of shape LEAD + S, standard normal at DT, quantized by Q
    along their last axis, (quants, scales, biases or None), drawn and
    quantized DRAW elements at a time."""
    import mlx.core as mx

    rows, cols = math.prod(lead) * s[0], s[1]
    step = max(1, DRAW // cols)
    parts = []
    for i in range(0, rows, step):
        key[0], k = mx.random.split(key[0])
        w = mx.random.normal((min(step, rows - i), cols), key=k).astype(dt)
        parts.append(mx.quantize(w, **q))
        mx.eval(*parts[-1])
    whole = [mx.concatenate([p[j] for p in parts]).reshape(*lead, s[0], -1)
             for j in range(len(parts[0]))]
    mx.eval(*whole)
    return tuple(whole) + (None,) * (3 - len(whole))


def mlx_decode(spec, key):
    """`decode FORMAT DTYPE EXPERTS ROWS N K LALB [IDS]`: MLX's
    quantized_matmul, or of experts gather_qmm, x's row i, as a 1 x K
    matrix, by the expert IDS names. A weight stored k x n is quantized
    along n, the only axis MLX quantizes it along."""
    import mlx.core as mx

    _, fmt, name, experts, rows, n, k, lalb, *named = spec
    if fmt not in QUANT:
        raise Unsupported(f"no {fmt} product")
    if lalb[0] != "k":
        raise Unsupported("no x stored k x rows")
    q, dt = QUANT[fmt], mlx_dtype(name)
    experts, rows, n, k = int(experts), int(rows), int(n), int(k)
    nk = lalb[1] == "k"
    s = (n, k) if nk else (k, n)
    if not experts:
        wq, scales, biases = mlx_quantized((), s, dt, q, key)

        def f(x):
            return mx.quantized_matmul(x, wq, scales, biases, transpose=nk,
                                       **q)

        return f, [mlx_draw((rows, k), dt, key)], 0
    wq, scales, biases = mlx_quantized((experts,), s, dt, q, key)
    idx = mx.array(ids(named[0]), dtype=mx.uint32)

    def f(x):
        return mx.gather_qmm(x, wq, scales, biases, rhs_indices=idx,
                             transpose=nk, **q)

    return f, [mlx_draw((rows, 1, k), dt, key)], 0


def mlx_fft(spec, key):
    import mlx.core as mx

    kind, name, dims, axes, *held = spec
    dt, s, axes = mlx_dtype(name), shape(dims), ids(axes)
    if kind == "rfft":
        return ((lambda x: mx.fft.rfftn(x, axes=axes)),
                [mlx_draw(s, dt, key)], Dep(0))
    if kind == "irfft":
        n = int(held[0])
        return ((lambda x: mx.fft.irfft(x, n=n, axis=axes[-1])),
                [mlx_draw(s, dt, key)], Dep(0))

    def f(x):
        return mx.fft.fftn(x, axes=axes)

    if not held:
        return f, [mlx_draw(s, dt, key)], 0
    whole = mlx_draw(shape(held[0]), dt, key)
    return f, [whole[tuple(slice(0, e) for e in s)]], None


def mlx_move(op, dt, key):
    import mlx.core as mx

    def ints(n, hi):
        key[0], k = mx.random.split(key[0])
        i = mx.random.randint(0, hi, (n,), key=k)
        mx.eval(i)
        return i

    def draw(s):
        return mlx_draw(s, dt, key)

    if op == "gather_emb":
        return ((lambda x, i: mx.take(x, i, axis=0)),
                [draw((EMB_ROWS, EMB_COLS)), ints(EMB_IDS, EMB_ROWS)], Dep(1))
    if op == "gather_rand":
        return ((lambda x, i: mx.take(x, i)),
                [draw((RAND_N,)), ints(RAND_N, RAND_N)], 0)
    if op == "take_along":
        i = ints(TAKE_ROWS * TAKE_K, TAKE_COLS).reshape(TAKE_ROWS, TAKE_K)
        return ((lambda x, i: mx.take_along_axis(x, i, axis=1)),
                [draw((TAKE_ROWS, TAKE_COLS)), i], Dep(1))
    if op in RANGES:
        # As PyTorch's, with no repeat by counts: a 1 at each run's first
        # position after the first run's, summed, numbers each position's
        # run.
        runs, xs = RANGES[op]
        starts, ends, size = range_runs(runs, xs[0])
        first = [0] + ends[:-1]
        base = mx.array([s - f for s, f in zip(starts, first)])
        # Runs of no cells at the end start past the last position.
        marks = mx.array([f for f in first[1:] if f < size])
        at = mx.arange(size)

        def ranges(x, base, marks):
            run = mx.cumsum(mx.zeros(size, dtype=mx.int32).at[marks].add(1))
            return mx.take(x, base[run] + at, axis=0)

        return ranges, [draw(xs), base, marks], Dep(1)
    if op == "assemble_cat":
        pieces = [draw((CAT_ROWS, CAT_COLS)) for _ in range(CAT_PIECES)]
        return (lambda *ps: mx.concatenate(ps)), pieces, 0
    if op == "assemble_pad":
        b = PAD_BY
        return ((lambda x: mx.pad(x, [(b, b), (b, b)],
                                  constant_values=PAD_FILL)),
                [draw((PAD_N, PAD_N))], 0)
    if op == "fold_patch":
        # A composite: patches that do not overlap are put back in place by
        # a transpose, (N, C, rows, columns, k, k) to (N, C, rows, k,
        # columns, k), copied into the image's shape.
        k, n = PATCH_K, PATCH_HW // PATCH_K
        s = (PATCH_N, PATCH_C, PATCH_HW, PATCH_HW)
        return ((lambda x: mx.reshape(mx.transpose(x, (0, 1, 2, 4, 3, 5)), s)),
                [draw((PATCH_N, PATCH_C, n, n, k, k))], 0)
    raise Unsupported(f"no {op}")


def mlx_sort(spec, key):
    import mlx.core as mx

    _, name, dims, axis, keep = spec
    s, axis, keep = shape(dims), int(axis), int(keep)
    ints = {"int64": mx.int64, "int32": mx.int32, "int16": mx.int16,
            "int8": mx.int8, "uint8": mx.uint8, "uint16": mx.uint16,
            "uint32": mx.uint32, "uint64": mx.uint64}
    if name in ints:
        key[0], k = mx.random.split(key[0])
        info = mx.iinfo(ints[name])
        lo, hi = max(info.min, -(2**31)), min(info.max, 2**31 - 1)
        x = mx.random.randint(lo, hi, s, dtype=ints[name], key=k)
        mx.eval(x)
    else:
        x = mlx_draw(s, mlx_dtype(name), key)

    # The values and the indices they take, as nx.metal writes them; the
    # [keep] first of each slice. MLX's topk leaves its k unordered, so a
    # kept sort sorts whole slices too. The values need the indices, so a
    # chain through the values computes both.
    def f(x):
        i = mx.argsort(x, axis=axis)
        if keep < s[axis]:
            i = mx.take(i, mx.arange(keep), axis=axis)
        return mx.take_along_axis(x, i, axis=axis)

    return f, [x], Dep(0)
def mlx_attention(spec, key):
    import mlx.core as mx

    name, (b, hq, hk, tq, tk, d), mask = attention_fields(spec)
    dt = mlx_dtype(name)
    q = mlx_draw((b, hq, tq, d), dt, key)
    k, v = mlx_draw((b, hk, tk, d), dt, key), mlx_draw((b, hk, tk, d), dt, key)
    m = "causal" if mask == "causal" else None
    if mask == "lengths":
        lens = mx.array(key_lengths(b, tk))
        m = (mx.arange(tk)[None, :] < lens[:, None]).reshape(b, 1, 1, tk)
    if mask == "window128":
        pos = mx.arange(tq)[:, None] + (tk - tq)
        keys = mx.arange(tk)[None, :]
        m = mx.logical_and(keys <= pos, pos < keys + 128).reshape(1, 1, tq, tk)
    if m is not None and not isinstance(m, str):
        mx.eval(m)
    scale = 1 / math.sqrt(d)

    def f(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                    mask=m)

    return f, [q, k, v], 0


def mlx_row(spec):
    import mlx.core as mx

    kind, *fields = spec
    key = [mx.random.key(7)]
    builders = {"contract": mlx_contract, "rows": mlx_rows,
                "decode": mlx_decode, "fft": mlx_fft, "rfft": mlx_fft,
                "irfft": mlx_fft, "sort": mlx_sort,
                "attention": mlx_attention}
    if kind in builders:
        return builders[kind](spec, key)
    dt = mlx_dtype(dtype(spec))
    if kind == "lu":
        # MLX factors on its CPU stream alone. L takes the operand's place:
        # it has its shape where the operand has no more columns than rows.
        return ((lambda a: mx.linalg.lu(a, stream=mx.cpu)[1]),
                [mlx_draw(shape(fields[1]), dt, key)], 0)
    if kind in SPECTRAL:
        # MLX's linalg has no GPU kernels: these run on its CPU stream,
        # and the bar is the faster of PyTorch's GPU and MLX's CPU.
        fs = {"eigh": lambda a: mx.linalg.eigh(a, stream=mx.cpu),
              "eigvalsh": lambda a: mx.linalg.eigvalsh(a, stream=mx.cpu),
              "svdvals": lambda a: mx.linalg.svd(a, compute_uv=False,
                                                 stream=mx.cpu)}
        return fs[kind], [mlx_draw(shape(fields[1]), dt, key)], None
    if kind == "move":
        return mlx_move(fields[0], dt, key)

    def draw(s):
        return mlx_draw(s, dt, key)

    if kind in ("reduce", "scan"):
        _, monoid, dims, axes = fields
        s, axis = shape(dims), int(axes.split(",")[0])
        x = draw(s)
        # MLX sums bfloat16 in float32 and rounds the sums to bfloat16.
        if kind == "reduce" and monoid == "sum" and "," not in axes:
            return (lambda x: mx.sum(x, axis=axis)), [x], None
        if kind == "scan" and monoid == "max":
            return (lambda x: mx.cummax(x, axis=axis)), [x], 0
        raise Unsupported(f"no {monoid} {kind}")
    op, _, dims = fields
    s = shape(dims)
    if op == "copy_at1":
        # A slice shares its operand's buffer and MLX copies no
        # contiguous array: no call of MLX's moves the elements.
        raise Unsupported("no copy of a contiguous view")
    if op == "copy_rows":
        # The first s[1] elements of each row of 256, into a fresh array.
        return (lambda x: mx.contiguous(x[:, :s[1]])), [draw((s[0], 256))], \
            None
    if op == "copy_step2":
        # Every other element, into a fresh array.
        return (lambda x: mx.contiguous(x[::2])), [draw((2 * s[0],))], None
    if op == "maxchain1024":
        def chain(x):
            y = x
            for _ in range(1024):
                y = mx.maximum(y, x)
            return y

        return chain, [draw(s)], 0
    x, y = draw(s), draw(s)
    if op == "add":
        return (lambda x, y: x + y), [x, y], 0
    if op == "exp":
        return mx.exp, [x], 0
    if op in CASTS:
        to = mlx_dtype(CASTS[op])
        return (lambda x: x.astype(to)), [x], None
    if op == "bcast_add":
        return (lambda x, v: x + v), [x, draw(s[-1:])], 0
    if op == "t_add":
        return (lambda t, y: y + t.T), [draw(s[::-1]), y], 1
    if op[2:] in CASTS and op.startswith("t_"):
        # astype keeps a transposed view's strides, so it would time a
        # cast without the transpose; forcing C order adds a pass.
        raise Unsupported("no C-order cast of a view in one pass")
    if op == "fused4":
        return (lambda x, y: x * y + mx.exp(x) - 0.5), [x, y], 0
    lead = [(0, 0)] * (len(s) - 2)
    if op == "pad1":
        return (lambda x: mx.pad(x[..., :-1, :-1], lead + [(1, 0), (1, 0)])), \
            [x], 0
    if op == "pad9":
        h, w = s[-2:]

        def pad9(x):
            p = mx.pad(x, lead + [(1, 1), (1, 1)])
            taps = [p[..., i:i + h, j:j + w] for i in range(3) for j in range(3)]
            out = taps[0]
            for t in taps[1:]:
                out = out + t
            return out

        return pad9, [x], 0
    if op == "where_lt":
        return (lambda x, y: mx.where(x < y, x, y)), [x, y], 0
    if op == "argmax":
        return (lambda x: mx.argmax(x, -1)), [x], None
    if op == "logsumexp":
        return (lambda x: mx.logsumexp(x, -1)), [x], None
    if op == "moments":
        return (lambda x: (mx.var(x, -1), mx.mean(x, -1))), [x], None
    if op == "window2":
        n, c, h, w = s
        return (lambda x: x.reshape(n, c, h // 2, 2, w // 2, 2).sum((3, 5))), \
            [x], None
    if op == "sqdiff":
        return (lambda x, y: mx.sum(mx.square(x - y), -1)), [x, y], None
    if op == "logcumsumexp":
        return (lambda x: mx.logcumsumexp(x, axis=-1)), [x], 0
    if op == "sqcumsum":
        return (lambda x: mx.cumsum(x * x, axis=-1)), [x], 0
    if op == "expcumsum_axis0":
        return (lambda x: mx.cumsum(mx.exp(x), axis=0)), [x], 0
    if op == "cummoments_explog16":
        # As PyTorch's.
        count = mx.arange(1, s[-1] + 1, dtype=dt)

        def cummoments(x):
            y = x
            for _ in range(16):
                y = mx.log(mx.exp(y))
            mean = mx.cumsum(y, axis=-1) / count
            return mean, mx.cumsum(y * y, axis=-1) / count - mean * mean

        return cummoments, [x], None
    raise Unsupported(f"no {op}")


def same_layout(y, x, strides):
    """Raises Unsupported unless output [y] can stand for operand [x]: the
    same shape and dtype and, where [strides] gives them, strides, so that
    the next call does the same work."""
    got = (tuple(y.shape), y.dtype) + ((strides(y),) if strides else ())
    want = (tuple(x.shape), x.dtype) + ((strides(x),) if strides else ())
    if got != want:
        raise Unsupported(f"chained output {got} is not its operand's {want}")


class Dep(int):
    """A row's chain where its output reaches the next call through a
    dependence (see `dep_link` and `torch_dep_link`), not in an operand's
    place: the index of the operand the dependence goes through."""


def dep_link():
    """MLX's link of a dependence: the operand, chosen against the operand
    plus one by whether the output's first element differs from itself;
    its second glue the same on the operand alone. Where that element is
    finite, as every row's draws make it, the next operand has the
    operand's bits, -0 included; a NaN would add one to each element."""
    import mlx.core as mx

    def dep(y, x):
        one = y.reshape(-1)[0]
        return mx.where(one != one, x + 1, x)

    return dep, (lambda x: dep(x, x))


def torch_dep_link():
    """PyTorch's link of a dependence, as `dep_link`'s."""
    import torch

    def dep(y, x):
        one = y.reshape(-1)[0]
        return torch.where(one != one, x + 1, x)

    return dep, (lambda x: dep(x, x))


# How an output takes its operand's place in the next call, by a link
# (output, operand) -> next operand, and a second glue, operand -> next
# operand, which repeats the link's work on the operand it gave, or None
# where the link is free:
# - the output itself where it has the operand's shape;
# - a larger output's first elements, a view, as the operand's shape;
# - a smaller output written over the operand's first elements, or
#   joined to the first operand's tail where the library writes nothing
#   in place: a kernel, the glue.


def torch_link(y, x):
    if y.dtype != x.dtype:
        raise Unsupported(f"output {y.dtype} is not its operand's {x.dtype}")
    if y.shape == x.shape:
        return None, None
    ny, nx = y.numel(), x.numel()
    if ny >= nx:
        if not y.is_contiguous():
            raise Unsupported("a strided output cuts by a copy")
        return (lambda y, x: y.view(-1)[:nx].view(x.shape)), None
    if not x.is_contiguous():
        raise Unsupported("a strided operand takes no output")

    def link(y, x):
        x.view(-1)[:ny].copy_(y.reshape(-1))
        return x

    # A copy of an array onto itself is skipped: the second copies another.
    fixed = x.reshape(-1)[:ny].clone()
    return link, (lambda x: link(fixed, x))


def mlx_link(y, x):
    import mlx.core as mx

    if y.dtype != x.dtype:
        raise Unsupported(f"output {y.dtype} is not its operand's {x.dtype}")
    if y.shape == x.shape:
        return None, None
    ny, nx = y.size, x.size
    if ny >= nx:
        return (lambda y, x: y.reshape(-1)[:nx].reshape(x.shape)), None

    # MLX writes no array in place: the output joined to the first
    # operand's tail, which a slice update would copy whole as well.
    tail = mx.array(x.reshape(-1)[ny:])
    mx.eval(tail)

    def link(y, x):
        return mx.concatenate([y.reshape(-1), tail]).reshape(x.shape)

    return link, (lambda x: link(x.reshape(-1)[:ny], x))


def cast_target(spec):
    """The dtype a cast row writes, or None."""
    kind, *fields = spec
    op = fields[0].removeprefix("t_") if fields else ""
    return CASTS.get(op) if kind == "loop" else None


def torch_mode(spec, compiled):
    import torch

    f, xs, chain = torch_row(spec)
    if compiled and (torch.float8_e4m3fn in [x.dtype for x in xs]
                     or cast_target(spec) == "float8_e4m3fn"):
        # torch 2.15's compiler raises KeyError on float8 for MPS, read or
        # written.
        raise Unsupported("no float8_e4m3fn in torch.compile on MPS")
    if compiled:
        torch._dynamo.reset()
        f = torch.compile(f, dynamic=False, fullgraph=True)
    link = again = None
    if isinstance(chain, Dep):
        link, again = torch_dep_link()
    elif chain is not None:
        y = f(*xs)
        link, again = torch_link(y, xs[chain])
        same_layout(link(y, xs[chain]) if link else y, xs[chain],
                    lambda t: t.stride())

    def call():
        y = f(*xs)
        if isinstance(chain, Dep):
            xs[chain] = link(y, xs[chain])
        elif chain is not None:
            xs[chain] = link(y, xs[chain]) if link else y

    def twice():
        call()
        xs[chain] = again(xs[chain])

    torch.mps.synchronize()
    return call, again and twice, torch.mps.synchronize, True


def mlx_mode(spec, compiled, lazy):
    import mlx.core as mx

    f, xs, chain = mlx_row(spec)
    mx.eval(*xs)
    if compiled:
        f = mx.compile(f)
    mx.reset_peak_memory()
    held = mx.get_active_memory()
    y = f(*xs)
    mx.eval(y)
    link = again = None
    # MLX exposes no strides: the check stops at shape and dtype.
    if isinstance(chain, Dep):
        link, again = dep_link()
    elif chain is not None:
        link, again = mlx_link(y, xs[chain])
        same_layout(link(y, xs[chain]) if link else y, xs[chain], None)
    # The calls in flight, their arrays at most WINDOW_BYTES: past it, a
    # call waits for the oldest's output while the GPU runs the others.
    window = max(2, WINDOW_BYTES // max(1, mx.get_peak_memory() - held))
    del y
    live = collections.deque()

    def one():
        y = f(*xs)
        if isinstance(chain, Dep):
            xs[chain] = link(y, xs[chain])
        elif chain is not None:
            y = xs[chain] = link(y, xs[chain]) if link else y
        return y

    def twice():
        one()
        xs[chain] = again(xs[chain])
        return xs[chain]

    if lazy:
        if chain is None:
            raise Unsupported("lazy evaluation computes unchained calls' last")
        return lazy_calls(one, twice if again else None, xs, chain, window)

    def windowed(issue):
        def call():
            y = issue()
            mx.async_eval(y)
            live.append(y)
            if len(live) > window:
                mx.eval(live.popleft())

        return call

    def sync():
        # Evaluating the calls in flight waits for them on any stream, the
        # CPU's included (lu), which the default stream's sync does not.
        mx.eval(*live)
        mx.synchronize()
        live.clear()

    return (windowed(one), again and windowed(twice), sync,
            chain is not None)


def lazy_calls(one, twice, xs, chain, window):
    """A chain's calls issued unevaluated, its last output evaluated once
    WINDOW calls are pending and by the sync."""
    import mlx.core as mx

    pending = 0

    def lazily(issue):
        def call():
            nonlocal pending
            issue()
            pending += 1
            if pending >= window:
                mx.eval(xs[chain])
                pending = 0

        return call

    def sync():
        nonlocal pending
        mx.eval(xs[chain])
        mx.synchronize()
        pending = 0

    return lazily(one), twice and lazily(twice), sync, True


RUSAGE_INFO_V4 = 4
RLIMIT_FOOTPRINT_INTERVAL = 4
FOOTPRINT_INTERVAL_RESET = 1


def take_peak(pid):
    """The most physical memory process [pid] has held since it started or
    since the last take, GPU buffers included, as the kernel counts it
    against the machine's memory; the next take counts from now."""
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    info = (ctypes.c_uint64 * 37)()  # rusage_info_v4: a 16-byte uuid, 35 fields
    if libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, info) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage {pid}")
    if libc.proc_rlimit_control(pid, RLIMIT_FOOTPRINT_INTERVAL,
                                ctypes.c_void_p(FOOTPRINT_INTERVAL_RESET)):
        raise OSError(ctypes.get_errno(), f"proc_rlimit_control {pid}")
    return info[2 + 33]  # ri_interval_max_phys_footprint


def pressure():
    """The kernel's memory pressure: 1 normal, 2 warning, 4 critical."""
    return int(subprocess.run(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        stdout=subprocess.PIPE, text=True, check=True).stdout)


def torch_copy(nbytes):
    """PyTorch's copy of NBYTES, half read and half written: the bytes
    nx.metal's copy moves."""
    import torch

    src = torch.randn(nbytes // 8, device="mps")
    dst = torch.empty_like(src)
    torch.mps.synchronize()
    return (lambda: dst.copy_(src)), None, torch.mps.synchronize, True


def make(mode, nbytes, spec):
    """MODE's call of the spec, or `copy`'s of NBYTES; the call with a
    second glue, or None (see `link`); their sync; and whether a time is
    one call's."""
    if mode == "copy":
        return torch_copy(nbytes)
    if mode.startswith("torch"):
        return torch_mode(spec, compiled="compile" in mode)
    return mlx_mode(spec, compiled="compile" in mode, lazy="lazy" in mode)


def reason(e):
    return str(e).splitlines()[0][:60] if str(e) else type(e).__name__


def reply(*fields):
    print("\t".join(str(f) for f in fields), flush=True)


def worker(nbytes, spec):
    """`--ref BYTES SPEC...`: the references' side. `mode MODE` makes the
    spec under MODE, or PyTorch's copy of BYTES under `copy`, after
    releasing the mode before, and answers
    `ready<TAB>CHAINED<TAB>GLUE`, CHAINED `True` where a time is one
    call's, GLUE `True` where the mode has a glue, or
    `none<TAB>REASON`. `COUNT` issues COUNT calls back to back from an idle
    GPU and answers `s<TAB>SECONDS`, or `error<TAB>REASON`; `glue COUNT`
    likewise the calls with a second glue. It prints `ready` once it has loaded
    both libraries."""
    import gc

    import mlx.core as mx
    import torch

    mx.set_memory_limit(LIVE_BYTES)
    mx.set_cache_limit(LIVE_BYTES)
    reply("ready")
    call = glue = sync = None
    for line in sys.stdin:
        cmd = line.split()
        if cmd[0] == "mode":
            call = glue = sync = None
            gc.collect()
            torch.mps.empty_cache()
            mx.clear_cache()
            try:
                call, glue, sync, chained = make(cmd[1], nbytes, spec)
                torch.mps.empty_cache()
                mx.clear_cache()
                reply("ready", chained, glue is not None)
            except Unsupported as e:  # a mode the library lacks says why
                reply("none", reason(e))
            except Exception as e:  # any other failure is the row's error
                reply("error", f"{type(e).__name__}: {reason(e)}")
            continue
        issue, count = (glue, cmd[1]) if cmd[0] == "glue" else (call, cmd[0])
        try:
            t0 = time.perf_counter()
            for _ in range(int(count)):
                issue()
            sync()
            reply("s", repr(time.perf_counter() - t0))
        except Exception as e:
            reply("error", f"{type(e).__name__}: {reason(e)}")


def describe():
    import mlx.core as mx
    import torch

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          stdout=subprocess.PIPE, text=True).stdout.strip()
    print(f"# {chip}, macOS {platform.mac_ver()[0]}, torch {torch.__version__}"
          f", mlx {mx.__version__}", flush=True)


# Sides

REPLY_S = 120  # the longest a side may take to answer: a compile


class Failed(Exception):
    def __init__(self, side, why):
        super().__init__(f"{side.name}: {why}")
        self.side, self.why = side, why


class Side:
    """A process that makes one computation and times its calls on
    request (bench_metal.exe compare ROW row|floor, worker)."""

    def __init__(self, name, cmd):
        self.name = name
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)

    def answer(self):
        r, _, _ = select.select([self.p.stdout], [], [], REPLY_S)
        line = self.p.stdout.readline() if r else ""
        if not line:
            self.close()
            raise Failed(self, "died" if r else "no answer")
        return line.rstrip("\n").split("\t")

    def ask(self, line):
        if self.p.poll() is not None:
            raise Failed(self, "died")
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return self.answer()

    def time(self, count, what=""):
        """Seconds a call over COUNT calls back to back from an idle GPU,
        of WHAT where given (`glue `)."""
        got, *rest = self.ask(f"{what}{count}")
        if got != "s":
            raise Failed(self, rest[0] if rest else got)
        return float(rest[0]) / count

    def take_peak(self):
        return take_peak(self.p.pid) if self.p.poll() is None else 0

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


class Glue:
    """The references' calls with a second glue, timed by their process as
    a side of its own."""

    def __init__(self, side):
        self.side, self.name = side, side.name + " glue"

    def time(self, count):
        return self.side.time(count, "glue ")


def runs(t):
    return max(3, min(MAX_RUNS, math.ceil(SPAN_S / t)))


def calls(side, n=1):
    """The calls a measurement of SIDE makes, enough to span SPAN_S and at
    least three, and their time a call. The count starts from a
    measurement of N calls and is taken again from a measurement of the
    calls it gives. One call's time holds the wait for an idle GPU and
    for the host to see the GPU end, and gives too few calls for a
    short one; the second measurement spreads that wait over its
    calls."""
    t = side.time(n)
    n = runs(t)
    if n > 3:
        t = side.time(n)
        n = runs(t)
    return n, t


def interleave(sides):
    """Each side's measurements in us a call. The ramp's calls are
    counted before it, after WARM unmeasured calls; a measurement's are
    counted after it, since a count taken on a GPU whose clock is still
    low is too small, and a measurement's start and end then weigh more
    on each of its calls. After the ramp, a round measures each side
    once, in order on rounds 0 and 3 of four and in reverse on 1 and
    2, so that a drift over the phase weighs on every side alike."""
    for s in sides:
        s.time(WARM)
    plan = [(s, *calls(s)) for s in sides]
    spent = 0.0
    while spent < RAMP_S:
        spent += sum(s.time(n) * n for s, n, _ in plan)
    plan = [(s, *calls(s, n)) for s, n, _ in plan]
    span = sum(n * t for _, n, t in plan)
    rounds = max(MIN_ROUNDS, min(MAX_ROUNDS, int(PHASE_S / span) // 4 * 4))
    got = [[] for _ in plan]
    for i in range(rounds):
        order = list(enumerate(plan))
        if i % 4 in (1, 2):
            order.reverse()
        for k, (s, n, _) in order:
            got[k].append(s.time(n) * 1e6)
    return got


# A cell's prefix where a reference failed for another reason than the
# library's lacking the operation: the row is in error, never rated.
ERROR = "error: "


def failure(what, why):
    """The cell of a mode the worker answered WHAT (`none` or `error`)."""
    return why if what == "none" else ERROR + why


def row(exe, name, nbytes, spec):
    """The row's measurements in us a call, by column, or a mode's reason,
    a glued mode's each less its round's cost of a glue; whether each
    mode's time is one call's; a glued mode's costs of a glue; and the most
    memory the row held at once, the largest over its phases of its two
    sides' peaks. Raises Failed where nx.metal fails."""
    me = os.path.abspath(__file__)
    ref = Side("references", [sys.executable, me, "--ref", nbytes, *spec])
    sides = [ref]
    try:
        floor = Side("nx.metal copy", [exe, "compare", name, "floor"])
        sides.append(floor)
        floor.answer()
        ref.answer()
        got, chained, glues = {}, {}, {}
        what, *why = ref.ask("mode copy")
        if what == "ready":
            got["nx.metal copy"], got["torch copy"] = interleave([floor, ref])
        else:
            got["torch copy"] = failure(what, why[0])
            (got["nx.metal copy"],) = interleave([floor])
        peak = floor.take_peak() + ref.take_peak()
        floor.close()
        nx = Side("nx.metal", [exe, "compare", name, "row"])
        sides.append(nx)
        nx.answer()
        got["nx.metal"] = []
        for m in MODES:
            try:
                what, *why = ref.ask("mode " + m)
                if what != "ready":
                    got[m] = failure(what, why[0])
                    continue
                chained[m] = why[0] == "True"
                glued = why[1] == "True"
                mine, theirs, *glue = interleave(
                    [nx, ref] + ([Glue(ref)] if glued else []))
                if glued:
                    glues[m] = [g - t for t, g in zip(theirs, glue[0])]
                    theirs = [t - c for t, c in zip(theirs, glues[m])]
            except Failed as e:
                if e.side is not ref:
                    raise
                got[m] = ERROR + e.why
                continue
            finally:
                peak = max(peak, nx.take_peak() + ref.take_peak())
            got["nx.metal"] += mine
            got[m] = theirs
        if not got["nx.metal"]:
            (got["nx.metal"],) = interleave([nx])
            peak = max(peak, nx.take_peak() + ref.take_peak())
        return got, chained, glues, peak
    finally:
        for s in sides:
            s.close()


# Gate bands of a ratio
MEETS = 0.95
BLOCKS = 0.5

# The products' rate, in operations a second, that bounds a row neither
# library computes, on the M1 Max: the fastest reference of
# contract/float16-4096x4096x4096, 2 x 4096^3 operations in 15.21 ms
# (mx.compile-lazy, gate 8 of 2026-11-19), the highest any contraction
# row measured.
PEAK_FLOPS = 2 * 4096**3 / 15.21e-3


def bounded(spec):
    """Whether the row is one neither library computes, held to its own
    bound: an operand of uint4 or float4_e2m1fn, or a Q6_K product. Any
    other row whose references all fail is in error."""
    kind, *fields = spec
    if kind == "loop":
        return fields[1] in ("uint4", "float4_e2m1fn")
    return kind == "decode" and fields[0].startswith("q6_k")


def composite(spec):
    """The note of a row whose reference is a composite of the library's
    calls, or None."""
    kind, *fields = spec
    if kind == "contract" and not fields[0] == fields[1] == fields[3]:
        return "composite: each operand cast to the output's dtype, then the product"
    if kind == "loop" and fields[0] == "cummoments_explog16":
        return "composite: cumsum of y and y², less accurate than nx's scan"
    if kind == "move" and fields[0] in RANGES:
        return "composite: each position's cell from starts and ends, then the gather"
    if kind == "move" and fields[0] == "fold_patch":
        return "composite in MLX: the patches transposed into place, copied"
    return None


def flops(spec):
    """The operations of a row's product, an add and a multiply a term,
    four of each at complex64; 0 for a row of no product."""
    kind, *fields = spec
    if kind == "contract":
        batch, m, n, k = (int(f) for f in fields[4:8])
        return (8 if fields[0] == "complex64" else 2) * batch * m * n * k
    if kind == "decode":
        rows, n, k = (int(f) for f in fields[3:6])
        return 2 * rows * n * k
    return 0


def band(ratio):
    if ratio >= MEETS:
        return "meets"
    return "gap" if ratio >= BLOCKS else "blocks"


def spread(us):
    """Median [min max] of measurements in us."""
    return (f"{statistics.median(us):.2f} [{min(us):.2f} "
            f"{max(us):.2f}]")


def cell(got, chained, glues, column):
    v = got[column]
    if isinstance(v, str):
        return v
    glue = (f" less glue {statistics.median(glues[column]):.2f}"
            if column in glues else "")
    return spread(v) + ("" if chained.get(column, True) else "*") + glue


def main(argv):
    exe = EXE
    if argv[:1] == ["--exe"]:
        exe, argv = argv[1], argv[2:]
    out = subprocess.run([exe, "compare", "--specs", *argv], check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    rows = [line.split("\t") for line in out.splitlines()]
    subprocess.run([sys.executable, os.path.abspath(__file__), "--describe"],
                   check=True)
    print()
    columns = ("nx.metal",) + COPIES + MODES
    print("| Row | " + " | ".join(columns)
          + " | Bar | Ratio | Band | Peak MiB |")
    print("|---|" + "---:|" * len(columns) + "---|---:|---|---:|")
    start = time.perf_counter()
    failures = []
    for name, nbytes, spec in rows:
        level = pressure()
        if level != 1:
            sys.exit(f"compare.py: memory pressure level {level} (1 is "
                     f"normal) before {name}; stopped")
        try:
            got, chained, glues, peak = row(exe, name, nbytes, spec.split())
        except Failed as e:
            print(f"| {name} | failed: {e} |" + " |" * (len(columns) + 3),
                  flush=True)
            failures.append(name)
            continue
        nx_us = statistics.median(got["nx.metal"])
        bars = [(statistics.median(got[m]), m) for m in MODES
                if not isinstance(got[m], str) and chained[m]]
        copies = [(statistics.median(got[c]), c) for c in COPIES
                  if not isinstance(got[c], str)]
        errors = [c for c in columns
                  if isinstance(got[c], str) and got[c].startswith(ERROR)]
        own = bounded(spec.split())
        bar, ratio, gate = "", "", ""
        if errors or not (bars or own and copies):
            bar = "error: " + ("; ".join(f"{c} {got[c][len(ERROR):]}"
                                         for c in errors)
                               or "no reference rates it")
            failures.append(name)
        else:
            if bars:
                t, m = max(min(bars), min(copies, default=min(bars)))
            else:
                # Neither library computes it: its own bound, the faster
                # copy of its bytes or its product's operations at
                # PEAK_FLOPS.
                t, m = max((min(copies)[0], "copy bound"),
                           (flops(spec.split()) / PEAK_FLOPS * 1e6,
                            "FLOP bound"))
            note = composite(spec.split())
            bar = f"{t / MEETS:.2f} {m}" + (f" ({note})" if note else "")
            ratio, gate = f"{t / nx_us:.3f}", band(t / nx_us)
        print(f"| {name} | "
              + " | ".join(cell(got, chained, glues, c) for c in columns)
              + f" | {bar} | {ratio} | {gate} | {peak / 2**20:.0f} |",
              flush=True)
    print(f"\n{len(rows)} rows in {time.perf_counter() - start:.0f} s")
    if failures:
        sys.exit(f"compare.py: {len(failures)} rows failed or in error: "
                 + ", ".join(failures))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--ref"]:
        worker(int(args[1]), args[2:])
    elif args[:1] == ["--describe"]:
        describe()
    else:
        main(args)
