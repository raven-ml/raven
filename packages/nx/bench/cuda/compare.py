# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.15.0", "numpy==2.5.4"]
# ///
"""nx.cuda against PyTorch on this machine's GPU: a table of each row's time
under both and their ratio, PyTorch's time over nx.cuda's (above 1, nx.cuda
is faster). Not on any alias.

usage: uv run packages/nx/bench/cuda/compare.py [--exe EXE] [PATTERN...]

The rows are bench_cuda.exe's, selected by PATTERN as its gate selects them.
`bench_cuda.exe compare` times nx.cuda's side and prints each row's spec;
this script times the spec under PyTorch with the same protocol: runs back
to back spanning 10 ms of the GPU's time, after it idled 20 ms, the median
of 30 such measurements, after the clock settled on the row; a call
longer than 100 ms runs alone in a span, measured as often as 3 s of
calls allow, at least 5 times. PyTorch's runs
replay a CUDA graph, so no host work falls between them, as nx.cuda's runs
queue behind a hold. Float products sum in float32 or wider, as nx.cuda's
do: TF32 and reduced-precision half sums are off. A row PyTorch does not
compute prints its reason in place of a time. An attention row names the
backend scaled_dot_product_attention picks.

EXE defaults to bench_cuda.exe in the checkout's _build, which
`dune build packages/nx/bench/cuda/bench_cuda.exe` makes.
"""

import os
import statistics
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
EXE = os.path.join(ROOT, "_build/default/packages/nx/bench/cuda/bench_cuda.exe")

SPAN_MS = 10.0
IDLE_S = 0.02
SETTLE_MS = 100.0
MEASUREMENTS = 30
LONG_MS = 3000.0
MAX_RUNS = 2000


def nx_times(exe, patterns):
    """Each selected row's name, nx.cuda's idle time in us, and spec; a
    time from the host, with its submission, names its row "(wall)"."""
    out = subprocess.run([exe, "compare", *patterns], check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    rows = []
    for line in out.splitlines():
        name, us, spec, *note = line.split("\t")
        name = name + "".join(f" ({n})" for n in note)
        rows.append((name, float(us), spec.split()))
    return rows


# PyTorch's side

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
DEV = "cuda"


class Unsupported(Exception):
    pass


def span_ms(f, n):
    """GPU time of n calls of f, in ms."""
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        f()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


def graphed(f, n):
    """A graph of n calls of f, after three on a side stream."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            f()
    return g


def first_ms(f):
    """One call of f in ms, or the mean of three for a call under 1 ms."""
    t = span_ms(f, 1)
    return t if t >= 1.0 else span_ms(f, 3) / 3


def runs(t):
    """Calls a span takes: one for a call longer than the settling time,
    else enough for SPAN_MS, at least three."""
    if t > SETTLE_MS:
        return 1
    return max(3, min(MAX_RUNS, int(SPAN_MS / t) + 1))


def measurements(t):
    """MEASUREMENTS, or for a call longer than the settling time as many as
    fit LONG_MS of calls, at least five."""
    if t <= SETTLE_MS:
        return MEASUREMENTS
    return max(5, min(MEASUREMENTS, int(LONG_MS / t)))


def eager_us(f):
    """f's time by calls back to back, for an f that syncs, which a graph
    cannot capture."""
    t = first_ms(f)
    n = runs(t)
    span_ms(f, n)
    ts = []
    for _ in range(measurements(t)):
        time.sleep(IDLE_S)
        ts.append(span_ms(f, n) / n)
    return statistics.median(ts) * 1e3


def time_us(f):
    t = first_ms(f)
    n = runs(t)
    try:
        g = graphed(f, n)
    except RuntimeError:
        torch.cuda.synchronize()
        return eager_us(f)
    t = span_ms(g.replay, 1) / n
    n2 = runs(t)
    if n2 > n:
        n = n2
        g = graphed(f, n)
    span_ms(g.replay, max(1, int(SETTLE_MS / (t * n))))
    ts = []
    for _ in range(measurements(t)):
        time.sleep(IDLE_S)
        ts.append(span_ms(g.replay, 1) / n)
    return statistics.median(ts) * 1e3


def dtype(name):
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise Unsupported(f"no torch.{name}")
    return dt


def draw(shape, dt):
    if dt.is_complex:
        return torch.randn(*shape, device=DEV, dtype=dt)
    if dt.is_floating_point:
        return torch.randn(*shape, device=DEV).to(dt)
    if dt == torch.int32:
        info = torch.iinfo(dt)
        return torch.randint(info.min, info.max, shape, device=DEV, dtype=dt)
    return torch.randint(-8, 8, shape, device=DEV, dtype=dt)


def operand(batch, rows, k, layout, dt):
    """[batch, rows, k] with k contiguous, or a view of [batch, k, rows]."""
    if layout == "k":
        x = draw((batch, rows, k), dt)
    else:
        x = draw((batch, k, rows), dt).mT
    return x[0] if batch == 1 else x


def contract(dt, acc, out, batch, m, n, k, layouts):
    dt, acc, out = dtype(dt), dtype(acc), dtype(out)
    batch, m, n, k = int(batch), int(m), int(n), int(k)
    a = operand(batch, m, k, layouts[0], dt)
    b = operand(batch, n, k, layouts[1], dt)
    wide = (torch.float32, torch.float64, torch.complex64, torch.complex128)
    if (dt.is_floating_point or dt.is_complex) and dt.itemsize >= 2 and out == dt:
        if acc not in wide or acc.itemsize < dt.itemsize:
            raise Unsupported(f"matmul sums {dt} in its own width")
        return lambda: torch.matmul(a, b.mT)
    if dt == torch.int8 and acc == torch.int32 and out == torch.int32:
        if batch != 1 or m <= 16 or k % 8 or n % 8:
            raise Unsupported("_int_mm takes m > 16, k and n of 8s")
        return lambda: torch._int_mm(a, b.mT)
    if dt == torch.float8_e4m3fn and out == torch.bfloat16:
        one = torch.ones((), device=DEV)
        if batch != 1 or m % 16 or n % 16 or k % 16:
            raise Unsupported("_scaled_mm takes extents of 16s")
        return lambda: torch._scaled_mm(a, b.mT, scale_a=one, scale_b=one,
                                        out_dtype=out)
    raise Unsupported(f"no {dt} -> {out} product")


def shape(text):
    return tuple(int(e) for e in text.split("x"))


def reduce(dt, monoid, dims, axes):
    dt = dtype(dt)
    x = draw(shape(dims), dt)
    axes = tuple(int(a) for a in axes.split(","))
    if monoid == "sum":
        return lambda: torch.sum(x, dim=axes, dtype=dt)
    if monoid == "max":
        return lambda: torch.amax(x, dim=axes)
    if monoid == "min":
        return lambda: torch.amin(x, dim=axes)
    if monoid == "prod" and len(axes) == 1:
        return lambda: torch.prod(x, dim=axes[0], dtype=dt)
    raise Unsupported(f"no {monoid} over {len(axes)} axes")


def scan(dt, monoid, dims, axis):
    dt = dtype(dt)
    x = draw(shape(dims), dt)
    axis = int(axis)
    if monoid == "sum":
        return lambda: torch.cumsum(x, dim=axis, dtype=dt)
    if monoid == "prod":
        return lambda: torch.cumprod(x, dim=axis, dtype=dt)
    if monoid == "max":
        return lambda: torch.cummax(x, dim=axis).values
    if monoid == "min":
        return lambda: torch.cummin(x, dim=axis).values
    raise Unsupported(f"no {monoid} scan")


def loop(op, dt, dims):
    """The loop rows.ml's loop_op names op, as PyTorch computes it."""
    dt, s = dtype(dt), shape(dims)
    x, y = draw(s, dt), draw(s, dt)
    if op == "add":
        return lambda: x + y
    if op == "exp":
        return lambda: torch.exp(x)
    if op == "cast_bf16":
        return lambda: x.to(torch.bfloat16)
    if op == "bcast_add":
        v = draw(s[-1:], dt)
        return lambda: x + v
    if op == "t_add":
        t = draw(s[::-1], dt)
        return lambda: t.T + y
    if op == "fused4":
        return lambda: x * y + torch.exp(x) - 0.5
    if op == "pad9":
        n, c, h, w = s
        f = torch.nn.functional.unfold
        return lambda: f(x, 3, padding=1).view(n, c, 9, h * w).sum(2)
    if op == "where_lt":
        return lambda: torch.where(x < y, x, y)
    if op == "argmax":
        return lambda: x.argmax(-1)
    if op == "logsumexp":
        return lambda: torch.logsumexp(x, -1)
    if op == "moments":
        return lambda: torch.var_mean(x, -1, correction=0)
    if op == "window2":
        pool = torch.nn.functional.avg_pool2d
        return lambda: pool(x, 2) * 4
    if op == "sqdiff":
        return lambda: ((x - y) ** 2).sum(-1)
    if op == "logcumsumexp":
        return lambda: torch.logcumsumexp(x, -1)
    if op == "sqcumsum":
        return lambda: torch.cumsum(x * x, -1)
    raise Unsupported(f"no loop {op}")


def keys(s, dt, rng):
    """Floats normal; integers in [0, rng), or over the dtype for rng 0."""
    if dt.is_floating_point or dt.is_complex:
        return torch.randn(*s, device=DEV, dtype=dt)
    if rng:
        return torch.randint(0, rng, s, device=DEV, dtype=dt)
    info = torch.iinfo(dt)
    return torch.randint(info.min, info.max, s, device=DEV, dtype=dt)


def sort(dt, dims, axis, descending, k, rng):
    dt, axis, k, rng = dtype(dt), int(axis), int(k), int(rng)
    x = keys(shape(dims), dt, rng)
    if k >= 0:
        return lambda: torch.topk(x, k, dim=axis, largest=descending == "true")
    return lambda: torch.sort(x, dim=axis, stable=True,
                              descending=descending == "true")


def unique(n, w, rng):
    n, w, rng = int(n), int(w), int(rng)
    if rng:
        x = torch.randint(0, rng, (n, w), device=DEV)
    else:
        x = torch.randint(-(1 << 62), 1 << 62, (n, w), device=DEV)
    if w == 1:
        v = x.view(-1)
        return lambda: torch.unique(v, return_inverse=True, return_counts=True)
    return lambda: torch.unique(x, dim=0, return_inverse=True,
                                return_counts=True)


def generator():
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    return g


def move(op, dt):
    """The move rows.ml's moves name op, as the moves' torch_rows.py
    computes it; PyTorch's scatter sums by atomics, in no fixed order."""
    dt, g, m = dtype(dt), generator(), 1 << 24
    rand = lambda n, hi: torch.randint(0, hi, (n,), device=DEV, generator=g)
    if op == "gather_emb":
        x = draw((65536, 1024), dt)
        ids = rand(16384, 65536)
        return lambda: torch.index_select(x, 0, ids)
    if op == "gather_rand":
        x, i = draw((m,), dt), rand(m, m)
        return lambda: torch.gather(x, 0, i)
    if op == "ranges":
        # PyTorch has no ranged gather: index_select of the rows the runs
        # name, their expansion made once, not timed
        x, k = draw((262144, 256), dt), 8192
        lens, starts = rand(k, 32), rand(k, 262144 - 32)
        off = torch.repeat_interleave(starts - (torch.cumsum(lens, 0) - lens),
                                      lens)
        rows = off + torch.arange(off.numel(), device=DEV)
        return lambda: torch.index_select(x, 0, rows)
    src = draw((m,), dt)
    if op in ("scatter_add_perm", "scatter_set_perm"):
        into = draw((m,), dt)
        perm = torch.randperm(m, device=DEV, generator=g)
        if op == "scatter_set_perm":
            return lambda: into.scatter(0, perm, src)
        return lambda: into.scatter_add(0, perm, src)
    if op == "scatter_add_one":
        into = draw((1024,), dt)
        one = torch.full((m,), 7, device=DEV, dtype=torch.int64)
        return lambda: into.scatter_add(0, one, src)
    if op == "scatter_add_rand":
        into, i = draw((m,), dt), rand(m, m)
        return lambda: into.scatter_add(0, i, src)
    if op in ("scatter_add_rand1m", "scatter_max_rand1m"):
        into, i = draw((1 << 20,), dt), rand(m, 1 << 20)
        if op == "scatter_max_rand1m":
            return lambda: into.scatter_reduce(0, i, src, "amax")
        return lambda: into.scatter_add(0, i, src)
    if op == "scatter_add_emb":
        emb, grads = draw((32768, 1024), dt), draw((16384, 1024), dt)
        ids = rand(16384, 32768)
        return lambda: emb.index_add(0, ids, grads)
    if op == "assemble_cat":
        parts = [draw((2048, 4096), dt) for _ in range(8)]
        return lambda: torch.cat(parts, 0)
    if op == "assemble_pad":
        x = draw((8192, 8192), dt)
        return lambda: torch.nn.functional.pad(x, (4, 4, 4, 4), value=1.5)
    if op == "fold_col2im":
        # the patches of a 3x3 window at padding 1 over [32, 64, 64, 64]
        cols = draw((32, 64 * 9, 64 * 64), dt)
        f = torch.nn.functional.fold
        return lambda: f(cols, (64, 64), 3, padding=1)
    raise Unsupported(f"no move {op}")


def fft(kind, dt, dims, axes):
    """A forward transform along axes; c2r the inverse of r2c, unscaled,
    into a real result of dims."""
    cd = dtype(dt)
    rd = torch.float32 if cd == torch.complex64 else torch.float64
    s, axes = shape(dims), tuple(int(a) for a in axes.split(","))
    if kind == "c2c":
        x = draw(s, cd)
        return lambda: torch.fft.fftn(x, dim=axes)
    if kind == "r2c":
        x = draw(s, rd)
        return lambda: torch.fft.rfftn(x, dim=axes)
    if kind == "c2r":
        x = draw(s[:-1] + (s[-1] // 2 + 1,), cd)
        n = [s[a] for a in axes]
        return lambda: torch.fft.irfftn(x, s=n, dim=axes, norm="forward")
    raise Unsupported(f"no {kind} transform")


def matrix(routine, dt, dims, mode):
    """The probes' torch_rows.py: factorisations of a batch of shape dims,
    a single spectral matrix as a 2-d one; uniform in [-1, 1)."""
    d, s = dtype(dt), shape(dims)
    g = torch.Generator(device=DEV).manual_seed(1)
    rnd = lambda *s: torch.rand(*s, device=DEV, dtype=d, generator=g) * 2 - 1
    eye = lambda n: torch.eye(n, device=DEV, dtype=d)
    if routine == "lu":
        a = rnd(*s)
        return lambda: torch.linalg.lu_factor_ex(a)
    if routine == "qr":
        a = rnd(*s)
        return lambda: torch.linalg.qr(a)
    if routine == "chol":
        b, n = s
        a = rnd(b, n, n)
        a = (a + a.mH) / 2 + n * eye(n)
        return lambda: torch.linalg.cholesky_ex(a)
    if routine == "solve":
        b, n, r = s
        a = torch.tril(rnd(b, n, n)) / n + 2 * eye(n)
        rhs = rnd(b, n, r)
        return lambda: torch.linalg.solve_triangular(a, rhs, upper=False)
    a = rnd(*s)
    if s[0] == 1:
        a = a[0]
    values = mode == "values"
    if routine == "eigh":
        f = torch.linalg.eigvalsh if values else torch.linalg.eigh
        return lambda: f(a)
    if routine == "svd":
        if values:
            return lambda: torch.linalg.svdvals(a)
        full = mode == "complete"
        return lambda: torch.linalg.svd(a, full_matrices=full)
    if routine == "eig":
        f = torch.linalg.eigvals if values else torch.linalg.eig
        return lambda: f(a)
    raise Unsupported(f"no {routine}")


SDP_BACKENDS = {int(b): b.name.lower() for b in
                torch.nn.attention.SDPBackend.__members__.values()}


def attention(dt, b, hq, hk, tq, tk, d, mask):
    """scaled_dot_product_attention of rows.ml's attention row, by the
    backend PyTorch picks, which the function's note names. Its causal
    mask is top-left aligned, Nx.causal's at tq = tk; "lengths" masks
    each batch element's keys past (i + 1) tk / b."""
    dt = dtype(dt)
    b, hq, hk, tq, tk, d = (int(x) for x in (b, hq, hk, tq, tk, d))
    q = draw((b, hq, tq, d), dt)
    k, v = draw((b, hk, tk, d), dt), draw((b, hk, tk, d), dt)
    causal, gqa, m = mask == "causal", hq != hk, None
    if mask == "lengths":
        lens = torch.tensor([(i + 1) * tk // b for i in range(b)], device=DEV)
        m = (torch.arange(tk, device=DEV) < lens[:, None]).view(b, 1, 1, tk)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    f = lambda: sdpa(q, k, v, attn_mask=m, is_causal=causal, enable_gqa=gqa)
    choice = torch._fused_sdp_choice(q, k, v, attn_mask=m, is_causal=causal,
                                     enable_gqa=gqa)
    f.note = SDP_BACKENDS.get(choice, str(choice))
    return f


KINDS = {"contract": contract, "reduce": reduce, "scan": scan, "loop": loop,
         "sort": sort, "unique": unique, "move": move, "fft": fft,
         "matrix": matrix, "attention": attention}


def torch_time(spec):
    """PyTorch's time of spec in us, or the reason it has none, and the
    notes its function carries."""
    kind, *fields = spec
    try:
        f = KINDS[kind](*fields)
        f()
        note = getattr(f, "note", None)
        return time_us(f), [note] if note else []
    except Unsupported as e:
        return str(e), []
    except (RuntimeError, TypeError) as e:
        return str(e).splitlines()[0][:60], []
    finally:
        torch.cuda.empty_cache()


def main(argv):
    exe = EXE
    if argv[:1] == ["--exe"]:
        exe, argv = argv[1], argv[2:]
    rows = nx_times(exe, argv)
    props = torch.cuda.get_device_properties(0)
    print(f"# {props.name}, torch {torch.__version__} (CUDA {torch.version.cuda})")
    print()
    print("| Row | nx.cuda us | PyTorch us | Ratio |")
    print("|---|---:|---:|---:|")
    for name, nx_us, spec in rows:
        t, note = torch_time(spec)
        name = name + "".join(f" ({n})" for n in note)
        if isinstance(t, str):
            print(f"| {name} | {nx_us:.2f} | {t} | |", flush=True)
        else:
            print(f"| {name} | {nx_us:.2f} | {t:.2f} | {t / nx_us:.2f} |",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
