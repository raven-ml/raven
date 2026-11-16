# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1+rocm7.2"]
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/rocm7.2"
# ///
"""nx.amd against PyTorch ROCm on this machine's GPU: a table of each row's
time under both, their ratio, PyTorch's time over nx.amd's (above 1,
nx.amd is faster), the row's floor kernel's time and the ratio's band:
"meets" from 0.95, "gap" from 0.5, "blocks" below. Not on any alias.

usage: uv run packages/nx/bench/amd/compare.py [--exe EXE] [PATTERN...]

On a machine without uv, run the script with the python3 of an
environment that holds torch 2.14.1+rocm7.2; it needs no NumPy.

The rows are bench_amd.exe's, selected by PATTERN as its gate selects
them. `bench_amd.exe compare` times nx.amd's side and prints each row's
spec; this script times the spec under PyTorch with the same protocol:
runs back to back spanning 10 ms of the GPU's time, after it idled 20 ms
and then ran the row 300 ms unmeasured, which brings back the clock that
idling lowers, the median of 30 such measurements, after the clock
settled on the row; a call longer than 100 ms runs alone in a span,
after one unmeasured call, measured as often as 3 s of calls allow, at
least 5 times. PyTorch's runs replay a HIP graph behind
torch.cuda._sleep, which spins the GPU while the host places the graph's
calls, as nx.amd's runs queue behind a hold: unheld, a graph of 4 us
kernels ran 8% slower on gfx1201. A span replays a graph of at most
256 calls as often as it takes. Float products sum in float32 or
wider, as nx.amd's do: reduced-precision half sums are off. Each nx.amd
call is a submission of its own, whose start writes back and invalidates
the L2, where PyTorch's graph keeps the L2 between its kernels: on a row
whose operands fit the L2, part of a gap may be that start, which
floor/submission times. The 64 MiB MALL keeps them across it. A row
PyTorch does not compute prints its reason in place of a time, and a row
nx.amd declines prints "declined" in place of its own, beside PyTorch's.
An attention row names the backend scaled_dot_product_attention picks.

Each row's PyTorch side runs in a process of its own, run once more if
it dies; a second death prints in place of the time, and the table ends
with the count of both. A float scatter sum is also timed under
torch.use_deterministic_algorithms(True), and that time is its ratio's
reference: nx.amd fixes the sum's bits, PyTorch's atomics do not.
"raises" says PyTorch refuses the row so, and its ratio is then against
the default, marked so.

EXE defaults to bench_amd.exe in the checkout's _build, which
`dune build packages/nx/bench/amd/bench_amd.exe` makes.
"""

import os
import signal
import statistics
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
EXE = os.path.join(ROOT, "_build/default/packages/nx/bench/amd/bench_amd.exe")

SPAN_MS = 10.0
IDLE_S = 0.02
SETTLE_MS = 100.0
RAMP_MS = 300.0
HOLD_CYCLES = 50_000_000
MEASUREMENTS = 30
LONG_MS = 3000.0
MAX_RUNS = 2000
# The most calls a graph holds: under ROCm 7.2 on gfx1201, replaying a
# graph of 2000 small matmuls segfaults, where one of 1000 does not.
GRAPH_CALLS = 256


def nx_times(exe, patterns):
    """Each selected row's name, nx.amd's idle time in us or None where it
    declines the row, its floor kernel's idle time in us or None, and spec;
    a time from the host, with its submission, names its row "(wall)"."""
    out = subprocess.run([exe, "compare", *patterns], check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    rows = []
    for line in out.splitlines():
        name, us, floor, spec, *note = line.split("\t")
        name = name + "".join(f" ({n})" for n in note)
        rows.append((name, None if us == "declined" else float(us),
                     None if floor == "-" else float(floor), spec.split()))
    return rows


# PyTorch's side

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
DEV = "cuda"


class Unsupported(Exception):
    pass


def span_ms(f, n, held=False):
    """GPU time of n calls of f, in ms; held, behind a spin of the
    stream's GPU, so that the host places every call before the first
    runs."""
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    if held:
        torch.cuda._sleep(HOLD_CYCLES)
    a.record()
    for _ in range(n):
        f()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


def graphed(f, n):
    """A graph of n calls of f, after three on a side stream: a HIP graph
    under ROCm, which torch.cuda names as CUDA's."""
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


def warm_calls(t):
    """Calls of t ms that span the ramp, unmeasured, at least one."""
    return max(1, int(RAMP_MS / t))


def eager_us(f):
    """f's time by calls back to back, for an f that syncs, which a graph
    cannot capture."""
    t = first_ms(f)
    n = runs(t)
    span_ms(f, n)
    ts = []
    for _ in range(measurements(t)):
        time.sleep(IDLE_S)
        for _ in range(warm_calls(t)):
            f()
        ts.append(span_ms(f, n) / n)
    return statistics.median(ts) * 1e3


def time_us(f):
    """f's time: a span's calls replay a graph of at most GRAPH_CALLS
    calls, as often as the span takes."""
    t = first_ms(f)
    c = min(runs(t), GRAPH_CALLS)
    try:
        g = graphed(f, c)
    except RuntimeError:
        torch.cuda.synchronize()
        return eager_us(f)
    t = span_ms(g.replay, 1) / c
    if runs(t) < c:
        c = runs(t)
        g.reset()
        g = graphed(f, c)
    reps = -(-runs(t) // c)
    span_ms(g.replay, max(1, int(SETTLE_MS / (t * c))))
    ts = []
    for _ in range(measurements(t)):
        time.sleep(IDLE_S)
        for _ in range(warm_calls(t * c)):
            g.replay()
        ts.append(span_ms(g.replay, reps, held=True) / (c * reps))
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
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    if dt in fp8 and out == torch.bfloat16:
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
    """The move rows.ml's moves name op; PyTorch's scatter sums by
    atomics, in no fixed order."""
    dt, g, m = dtype(dt), generator(), 1 << 24
    rand = lambda n, hi: torch.randint(0, hi, (n,), device=DEV, generator=g)
    if op == "gather_emb":
        x = draw((65536, 1024), dt)
        ids = rand(16384, 65536)
        return lambda: torch.index_select(x, 0, ids)
    if op in ("gather_rand", "gather_rand64k"):
        n = m if op == "gather_rand" else 65536
        x, i = draw((n,), dt), rand(n, n)
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
    if op in ("scatter_add_perm", "scatter_set_perm", "scatter_max_perm"):
        into = draw((m,), dt)
        perm = torch.randperm(m, device=DEV, generator=g)
        if op == "scatter_set_perm":
            return lambda: into.scatter(0, perm, src)
        if op == "scatter_max_perm":
            return lambda: into.scatter_reduce(0, perm, src, "amax")
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
    if op in ("scatter_max_rand4m", "scatter_max_rand8m"):
        t = (4 if op == "scatter_max_rand4m" else 8) << 20
        into, i = draw((t,), dt), rand(m, t)
        return lambda: into.scatter_reduce(0, i, src, "amax")
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
    """Factorisations of a batch of shape dims, a single spectral matrix as
    a 2-d one; uniform in [-1, 1)."""
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
        text = str(e)
        if torch.are_deterministic_algorithms_enabled() and \
                "deterministic" in text:
            return "raises", []
        return text.splitlines()[0][:60], []
    finally:
        torch.cuda.empty_cache()


def worker(argv):
    """`--torch [--deterministic] SPEC...`: one row's PyTorch time, printed
    as `us<TAB>TIME<TAB>NOTE...` or `none<TAB>REASON`; `--describe`: the
    device's line."""
    if argv[:1] == ["--describe"]:
        props = torch.cuda.get_device_properties(0)
        print(f"# {props.name} ({props.gcnArchName}), torch"
              f" {torch.__version__} (HIP {torch.version.hip})", flush=True)
        return
    if argv[:1] == ["--deterministic"]:
        # Uninitialized memory stays as it is: filling it with NaN would
        # time a write that determinism does not need.
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False
        argv = argv[1:]
    t, note = torch_time(argv)
    if isinstance(t, str):
        print(f"none\t{t}", flush=True)
    else:
        print("\t".join(["us", repr(t), *note]), flush=True)


# One row a process

# A row's process that dies, as torch.cuda._sleep has done on gfx1201 at
# random, runs once more; a second death names its signal in place of a
# time. A process past WORKER_S is killed and counts as a death.
WORKER_S = 600
SIGNALS = {signal.SIGSEGV: "segfault", signal.SIGABRT: "abort",
           signal.SIGBUS: "bus error", signal.SIGKILL: "killed"}


def death(rc):
    if rc is None:
        return "PyTorch timeout"
    if rc < 0:
        return "PyTorch " + SIGNALS.get(-rc, f"signal {-rc}")
    return f"PyTorch exit {rc}"


def spawn(args):
    """The result line of a worker run with args, or its death."""
    cmd = [sys.executable, os.path.abspath(__file__), *args]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True,
                           timeout=WORKER_S)
    except subprocess.TimeoutExpired:
        return None, death(None)
    lines = p.stdout.splitlines()
    if p.returncode != 0 or not lines:
        return None, death(p.returncode)
    return lines[-1].split("\t"), None


class Retries:
    def __init__(self):
        self.rows, self.retried, self.died = 0, 0, 0


def row_time(spec, retries, deterministic=False):
    """PyTorch's time of spec in us, or the reason it has none, and its
    notes, from a process of its own."""
    args = ["--torch", *(["--deterministic"] if deterministic else []), *spec]
    retries.rows += 1
    out, dead = spawn(args)
    if dead is not None:
        retries.retried += 1
        out, dead = spawn(args)
    if dead is not None:
        retries.died += 1
        return dead, []
    if out[0] == "none":
        return out[1], []
    return float(out[1]), out[2:]


def wants_deterministic(spec):
    """A float scatter sum: its bits follow PyTorch's atomics by default,
    where nx.amd fixes them, so the deterministic time is the bar."""
    return (spec[0] == "move" and spec[1].startswith("scatter_add")
            and spec[2].startswith(("float", "bfloat")))


# Gate bands of a ratio
MEETS = 0.95
BLOCKS = 0.5


def band(ratio):
    if ratio >= MEETS:
        return "meets"
    return "gap" if ratio >= BLOCKS else "blocks"


def cell(t):
    return t if isinstance(t, str) else f"{t:.2f}"


def main(argv):
    exe = EXE
    if argv[:1] == ["--exe"]:
        exe, argv = argv[1], argv[2:]
    rows = nx_times(exe, argv)
    out, dead = spawn(["--describe"])
    print(dead if out is None else out[0])
    print()
    print("| Row | nx.amd us | PyTorch us | PyTorch det. us | Ratio |"
          " Floor us | Band |")
    print("|---|---:|---:|---:|---:|---:|---|")
    retries = Retries()
    for name, nx_us, floor, spec in rows:
        t, note = row_time(spec, retries)
        det = ""
        ref, against = t, ""
        if wants_deterministic(spec):
            det, _ = row_time(spec, retries, deterministic=True)
            if isinstance(det, str):
                against = " (default)"
            else:
                ref = det
        name = name + "".join(f" ({n})" for n in note)
        nx = "declined" if nx_us is None else f"{nx_us:.2f}"
        fl = "" if floor is None else f"{floor:.2f}"
        ratio, gate = "", ""
        if nx_us is not None and not isinstance(ref, str):
            ratio, gate = f"{ref / nx_us:.2f}{against}", band(ref / nx_us)
        print(f"| {name} | {nx} | {cell(t)} | {cell(det)} | {ratio} | {fl} |"
              f" {gate} |", flush=True)
    print()
    print(f"# PyTorch processes: {retries.rows} runs, {retries.retried}"
          f" retried, {retries.died} died twice")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] in (["--torch"], ["--describe"]):
        worker(args[1:] if args[0] == "--torch" else args)
    else:
        main(args)
