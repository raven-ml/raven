# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.3", "pandas==3.0.6", "polars==1.44.2"]
# ///
"""The numpy, pandas and polars twins of nx's primitives bench.

Each twin computes what the nx row of the same id computes, on inputs of the
same distributions.

    uv run packages/nx/bench/primitives/bench_twins.py run [-f PAT]...
    uv run packages/nx/bench/primitives/bench_twins.py compare

`run` measures every twin and writes twins.tsv next to this script. With
`-f PAT`, it measures the twins whose id contains a PAT and prints them,
leaving twins.tsv as it is: the file holds one run, on one machine, on one day.
`compare` prints, for every row of nx_primitives.thumper, nx's median, each
twin's median and the ratio nx / twin.

The protocol: inputs are built outside timing; each twin is called 3 times to
warm up, then in batches of k calls, k the smallest power of two whose batch
takes at least 20 ms; 21 batches are timed with the collector disabled, and
the median time per call is kept. A twin whose library offers no stable order
is recorded under `<library>-unstable`.
"""

import gc
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
TWINS = HERE / "twins.tsv"
THUMPER = HERE / "nx_primitives.thumper"
SEED = 15
S = 40_000
L = 10_000_000


def rng():
    return np.random.default_rng(SEED)


def uniform(n):
    return rng().random(n)


def ints(n, bound, dtype=np.int64):
    return rng().integers(0, bound, n, dtype=dtype)


def words(n):
    return rng().integers(0, 2**64, n, dtype=np.uint64, endpoint=False)


# Twins: (id, library, setup, f). [setup ()] builds the inputs; [f inputs] is
# timed.

TWINS_ROWS = []


def twin(id, library, setup, f):
    TWINS_ROWS.append((id, library, setup, f))


def add_at(base, i, u):
    o = base.copy()
    np.add.at(o, i, u)
    return o


def set_at(base, i, u):
    o = base.copy()
    o[i] = u
    return o


def max_at(base, i, u):
    o = base.copy()
    np.maximum.at(o, i, u)
    return o


for n, name in [(S, "4e4"), (L, "1e7")]:
    twin(f"arange/int64-{name}", "numpy", lambda n=n: n,
         lambda n: np.arange(n, dtype=np.int64))
    twin(f"arange/int64-{name}", "polars", lambda n=n: n,
         lambda n: pl.int_range(0, n, eager=True))
twin("arange/int32-1e7", "numpy", lambda: L,
     lambda n: np.arange(n, dtype=np.int32))
twin("arange/int32-1e7", "polars", lambda: L,
     lambda n: pl.int_range(0, n, dtype=pl.Int32, eager=True))

for n, name in [(S, "4e4"), (L, "1e7")]:
    for dtype, make in [("float64", uniform), ("uint64", words)]:
        id = f"argsort/{dtype}-{name}"
        twin(id, "numpy", lambda n=n, make=make: make(n),
             lambda x: np.argsort(x, kind="stable"))
        twin(id, "pandas", lambda n=n, make=make: pd.Series(make(n)),
             lambda s: s.argsort(kind="stable"))
        twin(id, "polars-unstable", lambda n=n, make=make: pl.Series(make(n)),
             lambda s: s.arg_sort())

for id, make in [("cumsum/float64-4e4", lambda: uniform(S)),
                 ("cumsum/float64-1e7", lambda: uniform(L)),
                 ("cumsum/int64-1e7", lambda: ints(L, 2**20))]:
    twin(id, "numpy", make, np.cumsum)
    twin(id, "pandas", lambda make=make: pd.Series(make()), lambda s: s.cumsum())
    twin(id, "polars", lambda make=make: pl.Series(make()), lambda s: s.cum_sum())

for id, f, n, m in [("add-float64-4e4-into-1e2", add_at, S, 100),
                    ("add-float64-1e7-into-1e2", add_at, L, 100),
                    ("add-float64-1e7-into-1e6", add_at, L, 1_000_000),
                    ("set-float64-4e4-into-4e4", set_at, S, S),
                    ("set-float64-1e7-into-1e6", set_at, L, 1_000_000),
                    ("max-float64-4e4-into-1e2", max_at, S, 100),
                    ("max-float64-1e7-into-1e2", max_at, L, 100),
                    ("max-float64-1e7-into-1e6", max_at, L, 1_000_000)]:
    twin(f"scatter/{id}", "numpy",
         lambda n=n, m=m: (np.zeros(m), ints(n, m), uniform(n)),
         lambda a, f=f: f(*a))
twin("scatter/add-float16-1e7-into-1e6", "numpy",
     lambda: (np.zeros(1_000_000, np.float16), ints(L, 1_000_000),
              uniform(L).astype(np.float16)),
     lambda a: add_at(*a))


def gathered(make_x, make_i):
    return lambda: (make_x(), make_i())


for id, make in [
        ("gather/float32-4e4-monotone",
         gathered(lambda: uniform(S).astype(np.float32),
                  lambda: np.sort(ints(S, S)))),
        ("gather/float32-1e7-monotone",
         gathered(lambda: uniform(L).astype(np.float32),
                  lambda: np.sort(ints(L, L)))),
        ("gather/float64-1e7-random",
         gathered(lambda: uniform(L), lambda: ints(L, L)))]:
    twin(id, "numpy", make, lambda a: a[0][a[1]])
    twin(id, "polars",
         lambda make=make: tuple(pl.Series(v) for v in make()),
         lambda a: a[0].gather(a[1]))
for id, n in [("gather/float32-rows-4e4x8", S),
              ("gather/float32-rows-1.25e6x8", 1_250_000)]:
    twin(id, "numpy",
         gathered(lambda n=n: uniform(8 * n).astype(np.float32).reshape(n, 8),
                  lambda n=n: ints(n, n)),
         lambda a: a[0][a[1]])


# Measurement

WARMUP = 3
SAMPLES = 21
BATCH_FLOOR_NS = 20_000_000


def measure(setup, f):
    x = setup()
    for _ in range(WARMUP):
        f(x)
    gc.collect()
    gc.disable()
    try:
        k = 1
        while True:
            t = time.perf_counter_ns()
            for _ in range(k):
                f(x)
            if time.perf_counter_ns() - t >= BATCH_FLOOR_NS:
                break
            k *= 2
        samples = []
        for _ in range(SAMPLES):
            t = time.perf_counter_ns()
            for _ in range(k):
                f(x)
            samples.append((time.perf_counter_ns() - t) / k / 1e9)
    finally:
        gc.enable()
    del x
    gc.collect()
    return sorted(samples)


def median(samples):
    s = sorted(samples)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def load():
    return " ".join(f"{v:.2f}" for v in os.getloadavg())


def run(patterns):
    rows = [r for r in TWINS_ROWS
            if not patterns or any(p in r[0] for p in patterns)]
    if not rows:
        sys.exit(f"no twin matches {' '.join(patterns)}")
    before = load()
    lines = []
    for id, library, setup, f in rows:
        samples = measure(setup, f)
        print(f"{id}\t{library}\t{median(samples):.3e}", flush=True)
        lines.append("\t".join([id, library, f"{median(samples):.3e}",
                                " ".join(f"{s:.3e}" for s in samples)]))
    if patterns:
        return
    u = platform.uname()
    header = [
        "# suite: nx_primitives twins",
        f"# host: {u.node} · {platform.processor()} · {u.machine} · {u.system}",
        f"# python: {platform.python_version()}; numpy {np.__version__}; "
        f"pandas {pd.__version__}; polars {pl.__version__}",
        f"# date: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# load: {before} -> {load()}",
        "# id\tlibrary\tmedian_s\tsorted samples_s",
    ]
    TWINS.write_text("\n".join(header + lines) + "\n")


def thumper_medians():
    """nx's wall-time medians, from this machine's section of the baseline."""
    sections = {}
    host = None
    for line in THUMPER.read_text().splitlines():
        if line.startswith("# host: "):
            host = line[len("# host: "):]
            sections[host] = {}
        elif line and not line.startswith("#") and host is not None:
            fields = line.split("\t")
            if fields[1] == "wall_time":
                samples = [float(v) for v in fields[-1].split()]
                sections[host][fields[0]] = median(samples)
    node = platform.uname().node
    mine = [h for h in sections if h.startswith(node)]
    if mine:
        return sections[mine[0]]
    if len(sections) == 1:
        return next(iter(sections.values()))
    sys.exit(f"{THUMPER.name} has no section for {node}")


def compare():
    twins = {}
    for line in TWINS.read_text().splitlines():
        if not line.startswith("#"):
            id, library, median_s, _ = line.split("\t")
            twins.setdefault(id, []).append((library, float(median_s)))
    print("id\tnx_s\tlibrary\ttwin_s\tnx/twin")
    for id, nx in sorted(thumper_medians().items()):
        for library, t in twins.get(id, [("-", None)]):
            ratio = "-" if t is None else f"{nx / t:.2f}"
            twin_s = "-" if t is None else f"{t:.3e}"
            print(f"{id}\t{nx:.3e}\t{library}\t{twin_s}\t{ratio}")


def main(argv):
    if argv[:1] == ["run"]:
        rest = argv[1:]
        if len(rest) % 2 or any(f != "-f" for f in rest[::2]):
            sys.exit("usage: bench_twins.py run [-f PAT]...")
        run(rest[1::2])
    elif argv == ["compare"]:
        compare()
    else:
        sys.exit("usage: bench_twins.py run [-f PAT]... | compare")


if __name__ == "__main__":
    main(sys.argv[1:])
