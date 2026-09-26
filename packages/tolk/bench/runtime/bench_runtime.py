#!/usr/bin/env python3
"""Compare execution-only throughput with an explicit tinygrad checkout.

Run either companion first, using the same DEV for both:
  python packages/tolk/bench/runtime/bench_runtime.py OUT --reference PATH_TO_REFERENCE

Both sides use realized zero-filled inputs, two JIT warmup/capture calls and
adaptive replay timing. Compilation and input-buffer allocation are outside
the timed region. Repeat alternating runs on a quiet host before drawing conclusions.
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
parser.add_argument("--reference", type=Path, required=True)
args = parser.parse_args()
reference = args.reference.resolve()
reference_package = (reference / "tinygrad").resolve()
if not (reference_package / "__init__.py").is_file():
    parser.error("--reference must contain the tinygrad Python package")
sys.path.insert(0, str(reference))

# Match the OCaml runner: CPU by default, or the explicitly selected backend.
os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("NO_COLOR", "1")

import tinygrad  # noqa: E402

if Path(tinygrad.__file__).resolve().parent != reference_package:
    raise RuntimeError("imported tinygrad does not belong to --reference")

from tinygrad import Device, Tensor, TinyJit  # noqa: E402
from tinygrad.device import Buffer  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.helpers import GlobalCounters  # noqa: E402

BACKEND = Device.DEFAULT
F32_BYTES = 4
BUF_ELEMS = 16 * 1024 * 1024
TARGET_S = 1.5
MIN_K = 5
MAX_K = 5000


def sync():
    Device[BACKEND].synchronize()


def time_replay(call):
    """Adaptive execution-only timing: one estimate call sizes K, then median
    and min per-replay nanoseconds over K samples."""
    t0 = time.perf_counter_ns()
    call()
    est_s = (time.perf_counter_ns() - t0) / 1e9
    k = int(max(1.0, TARGET_S / max(est_s, 1e-9)))
    k = max(MIN_K, min(MAX_K, k))
    samples = []
    storage_before = GlobalCounters.mem_used
    for _ in range(k):
        t0 = time.perf_counter_ns()
        call()
        samples.append(float(time.perf_counter_ns() - t0))
    storage_after = GlobalCounters.mem_used
    samples.sort()
    return samples[k // 2], samples[0], k, storage_before, storage_after


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
                  device=BACKEND).reshape(shape).realize()


def matmul_bench(n):
    a = input_tensor((n, n))
    b = input_tensor((n, n))
    median, minimum, k, storage_before, storage_after = time_compute(lambda a, b: (a @ b).realize(), (a, b))
    flops = 2.0 * n * n * n
    return {"bench": "matmul", "size": str(n), "unit": "GFLOP/s",
            "amount": flops, "median_ns": median, "min_ns": minimum, "k": k,
            "storage_bytes_before": storage_before, "storage_bytes_after": storage_after}


def elementwise_bench(n):
    a = input_tensor((n,))
    b = input_tensor((n,))
    c = input_tensor((n,))
    median, minimum, k, storage_before, storage_after = time_compute(
        lambda a, b, c: (a + b * c).realize(), (a, b, c))
    return {"bench": "elementwise", "size": "16M", "unit": "GB/s",
            "amount": 4.0 * n * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k,
            "storage_bytes_before": storage_before, "storage_bytes_after": storage_after}


def reduce_bench(n):
    x = input_tensor((n,))
    median, minimum, k, storage_before, storage_after = time_compute(lambda x: x.sum().realize(), (x,))
    return {"bench": "reduce", "size": "16M", "unit": "GB/s",
            "amount": float(n) * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k,
            "storage_bytes_before": storage_before, "storage_bytes_after": storage_after}


def copy_bench(n):
    dev = Device[BACKEND]
    buf = Buffer(BACKEND, n, dtypes.float32).allocate()
    host = memoryview(bytearray(n * F32_BYTES))

    def call():
        buf.host[:] = host
        dev.synchronize()

    call()
    median, minimum, k, storage_before, storage_after = time_replay(call)
    return {"bench": "copy", "size": "16M", "unit": "GB/s",
            "amount": float(n) * F32_BYTES,
            "median_ns": median, "min_ns": minimum, "k": k,
            "storage_bytes_before": storage_before, "storage_bytes_after": storage_after}


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

    tolk_file = Path(out_dir, "tolk_runtime.json")
    tolk_rows = json.loads(tolk_file.read_text()) if tolk_file.exists() else []

    if any(row["backend"] != BACKEND for row in tolk_rows):
        raise ValueError("both runtime benchmarks must use the same DEV backend")
    revision = (subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
        if (reference / ".git").exists() else None)
    digest = hashlib.sha256()
    for source in sorted((reference / "tinygrad").rglob("*.py")):
        digest.update(str(source.relative_to(reference)).encode() + b"\0")
        digest.update(source.read_bytes() + b"\0")
    ref = run_reference()
    provenance = {"reference": str(reference), "imported_package": str(reference_package),
                  "revision": revision,
                  "python_source_sha256": digest.hexdigest(), "python": sys.version,
                  "target": str(Device[BACKEND].renderer.target),
                  "rows": list(ref.values())}
    Path(out_dir, "tinygrad_runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")

    if not tolk_rows:
        print(f"wrote {len(ref)} reference rows for {BACKEND} to {out_dir}")
        return

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

    print(f"{BACKEND} execution-only throughput against tinygrad {revision or digest.hexdigest()}")
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
