#!/usr/bin/env python3
"""Compare execution-only CPU throughput with an explicit tinygrad checkout.

Run the OCaml executable first, then:
  python packages/tolk/bench/runtime/bench_runtime.py OUT --reference _tinygrad_target

Both sides use realized zero-filled inputs, two JIT warmup/capture calls and
adaptive replay timing. Compilation and allocation are outside the timed
region. Repeat alternating runs on a quiet host before drawing conclusions.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out_dir", nargs="?", default=".")
parser.add_argument("--reference", type=Path,
                    default=Path(__file__).resolve().parents[4] / "_tinygrad")
args = parser.parse_args()
sys.path.insert(0, str(args.reference.resolve()))

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

    # Warm and capture; the estimate call below is the first replay.
    call()
    call()
    return time_replay(call)


def input_tensor(shape):
    return Tensor(bytes(math.prod(shape) * F32_BYTES), dtype=dtypes.float32,
                  device="CPU").reshape(shape).realize()


def matmul_bench(n):
    a = input_tensor((n, n))
    b = input_tensor((n, n))
    median, minimum, k = time_compute(lambda a, b: (a @ b).realize(), (a, b))
    flops = 2.0 * n * n * n
    return {"bench": "matmul", "size": str(n), "unit": "GFLOP/s",
            "amount": flops, "median_ns": median, "min_ns": minimum, "k": k}


def elementwise_bench(n):
    a = input_tensor((n,))
    b = input_tensor((n,))
    c = input_tensor((n,))
    median, minimum, k = time_compute(
        lambda a, b, c: (a + b * c).realize(), (a, b, c))
    return {"bench": "elementwise", "size": "16M", "unit": "GB/s",
            "amount": 4.0 * n * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k}


def reduce_bench(n):
    x = input_tensor((n,))
    median, minimum, k = time_compute(lambda x: x.sum().realize(), (x,))
    return {"bench": "reduce", "size": "16M", "unit": "GB/s",
            "amount": float(n) * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k}


def copy_bench(n):
    dev = Device["CPU"]
    buf = Buffer("CPU", n, dtypes.float32).allocate()
    host = memoryview(bytearray(n * F32_BYTES))

    def call():
        buf.host[:] = host
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
        row = bench()
        rows[(row["bench"], row["size"])] = row
    return rows


def main():
    out_dir = args.out_dir

    with open(os.path.join(out_dir, "tolk_runtime.json")) as f:
        tolk_rows = json.load(f)

    if any(row["backend"] != "CPU" for row in tolk_rows):
        raise ValueError("the reference CPU benchmark requires DEV=CPU on both sides")
    reference = args.reference.resolve()
    revision = (subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
        if (reference / ".git").exists() else None)
    digest = hashlib.sha256()
    for source in sorted((reference / "tinygrad").rglob("*.py")):
        digest.update(str(source.relative_to(reference)).encode() + b"\0")
        digest.update(source.read_bytes() + b"\0")
    ref = run_reference()
    provenance = {"reference": str(reference), "revision": revision,
                  "python_source_sha256": digest.hexdigest(), "python": sys.version,
                  "target": str(Device["CPU"].renderer.target),
                  "rows": list(ref.values())}
    Path(out_dir, "tinygrad_runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")

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

    print(f"CPU execution-only throughput against tinygrad {revision or digest.hexdigest()}")
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
