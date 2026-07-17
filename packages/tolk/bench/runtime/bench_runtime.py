#!/usr/bin/env python3
"""Reference-side runtime throughput for the indicative runtime benches, and
the merge step that prints the context table.

Mirrors the four OCaml workloads (see bench_runtime.ml) on the reference's CPU
device and times execution-only replays: matmul and the elementwise/reduce
buffer kernels through the reference JIT, and the host->device copy through the
allocator's copyin primitive. Warm once (capture), then median and min of an
adaptively sized replay run on a monotonic clock.

Then reads <out>/tolk_runtime.json (written by the OCaml exe), joins on
(bench, size), prints the human table `bench size backend tolk tg unit`, and
writes <out>/runtime.tsv.

These numbers are indicative context, not a target: a runtime gap would only
close by changing compiler semantics, which is out of scope for this suite.

Run from the repo root, after the OCaml exe has written its JSON:
  uv run packages/tolk/bench/runtime/bench_runtime.py [out_dir]
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "_tinygrad"))

# Force the reference onto CPU so the context column matches the tolk CPU
# default, and silence ANSI so nothing leaks into stdout.
os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("NO_COLOR", "1")

from tinygrad import Device, Tensor, TinyJit  # noqa: E402
from tinygrad.device import Buffer  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

F32_BYTES = 4
BUF_ELEMS = 16 * 1024 * 1024
TARGET_S = 1.5
MIN_K = 5
MAX_K = 5000


def sync():
    Device["CPU"].synchronize()


def time_replay(call):
    """Adaptive execution-only timing: one estimate call sizes K, then median
    and min per-replay nanoseconds over K samples."""
    t0 = time.perf_counter_ns()
    call()
    est_s = (time.perf_counter_ns() - t0) / 1e9
    k = int(max(1.0, TARGET_S / max(est_s, 1e-9)))
    k = max(MIN_K, min(MAX_K, k))
    samples = []
    for _ in range(k):
        t0 = time.perf_counter_ns()
        call()
        samples.append(float(time.perf_counter_ns() - t0))
    samples.sort()
    return samples[k // 2], samples[0], k


def time_compute(build, inputs):
    jf = TinyJit(build)

    def call():
        jf(*inputs)
        sync()

    # Warm and capture: the reference JIT records on the second call and
    # replays afterward, so three calls guarantee a compiled program.
    call()
    call()
    call()
    return time_replay(call)


def matmul_bench(n):
    a = Tensor.ones(n, n, device="CPU").contiguous().realize()
    b = Tensor.ones(n, n, device="CPU").contiguous().realize()
    median, minimum, k = time_compute(lambda a, b: (a @ b).realize(), (a, b))
    flops = 2.0 * n * n * n
    return {"bench": "matmul", "size": str(n), "unit": "GFLOP/s",
            "amount": flops, "median_ns": median, "min_ns": minimum, "k": k}


def elementwise_bench(n):
    a = Tensor.ones(n, device="CPU").contiguous().realize()
    b = Tensor.ones(n, device="CPU").contiguous().realize()
    c = Tensor.ones(n, device="CPU").contiguous().realize()
    median, minimum, k = time_compute(
        lambda a, b, c: (a + b * c).realize(), (a, b, c))
    return {"bench": "elementwise", "size": "16M", "unit": "GB/s",
            "amount": 4.0 * n * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k}


def reduce_bench(n):
    x = Tensor.ones(n, device="CPU").contiguous().realize()
    median, minimum, k = time_compute(lambda x: x.sum().realize(), (x,))
    return {"bench": "reduce", "size": "16M", "unit": "GB/s",
            "amount": float(n) * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k}


def copy_bench(n):
    dev = Device["CPU"]
    buf = Buffer("CPU", n, dtypes.float32).allocate()
    host = memoryview(bytearray(n * F32_BYTES))

    def call():
        dev.allocator._copyin(buf._buf, host)
        dev.synchronize()

    call()
    median, minimum, k = time_replay(call)
    return {"bench": "copy", "size": "16M", "unit": "GB/s",
            "amount": float(n) * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k}


BENCHES = [
    lambda: matmul_bench(512),
    lambda: matmul_bench(1024),
    lambda: elementwise_bench(BUF_ELEMS),
    lambda: reduce_bench(BUF_ELEMS),
    lambda: copy_bench(BUF_ELEMS),
]


def per_ns(amount, ns):
    # One flop per ns is a GFLOP/s; one byte per ns is a GB/s.
    return amount / ns if ns > 0 else 0.0


def run_reference():
    rows = {}
    for bench in BENCHES:
        try:
            r = bench()
            rows[(r["bench"], r["size"])] = r
        except Exception as e:  # noqa: BLE001 - context column, keep going
            print(f"WARNING: reference bench failed: {e}", file=sys.stderr)
    return rows


BANNER = (
    "=" * 72 + "\n"
    "INDICATIVE runtime throughput — context only, NOT a target.\n"
    "The reference column is provided for orientation. Closing a runtime gap\n"
    "would require changing compiler semantics, which is out of scope here.\n"
    "Both sides time execution-only replays (compile/schedule held outside\n"
    "the timed loop); values are the median-replay throughput.\n"
    + "=" * 72
)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    ref = run_reference()

    with open(os.path.join(out_dir, "tolk_runtime.json")) as f:
        tolk_rows = json.load(f)

    header = ["bench", "size", "backend", "tolk", "tg", "unit"]
    table = []
    for t in tolk_rows:
        key = (t["bench"], t["size"])
        r = ref.get(key)
        tg = per_ns(r["amount"], r["median_ns"]) if r else None
        table.append({
            "bench": t["bench"], "size": t["size"], "backend": t["backend"],
            "tolk": t["median"], "tolk_peak": t["peak"],
            "tg": tg,
            "tg_peak": per_ns(r["amount"], r["min_ns"]) if r else None,
            "unit": t["unit"],
        })

    print(BANNER)
    rows = [header] + [[
        r["bench"], r["size"], r["backend"], f"{r['tolk']:.2f}",
        f"{r['tg']:.2f}" if r["tg"] is not None else "-", r["unit"],
    ] for r in table]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for i, row in enumerate(rows):
        print("  ".join(c.ljust(widths[j]) for j, c in enumerate(row)))
        if i == 0:
            print("  ".join("-" * widths[j] for j in range(len(header))))

    tsv_path = os.path.join(out_dir, "runtime.tsv")
    cols = ["bench", "size", "backend", "tolk_median", "tolk_peak",
            "tg_median", "tg_peak", "unit"]
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in table:
            f.write("\t".join([
                r["bench"], r["size"], r["backend"],
                f"{r['tolk']:.6f}", f"{r['tolk_peak']:.6f}",
                f"{r['tg']:.6f}" if r["tg"] is not None else "",
                f"{r['tg_peak']:.6f}" if r["tg_peak"] is not None else "",
                r["unit"],
            ]) + "\n")
    print(f"\nwrote {tsv_path}")


if __name__ == "__main__":
    main()
