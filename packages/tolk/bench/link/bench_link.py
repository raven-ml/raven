#!/usr/bin/env python3
"""Companion for bench_link.ml, using an explicit frozen tinygrad checkout.

This measures linking, not model throughput. GC counters are Python collection
counts, not OCaml heap words. Linked storage roots include shared ring capacity;
neither implementation reports a native allocation count.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--calls", type=int, default=64)
parser.add_argument("--samples", type=int, default=11)
args = parser.parse_args()
sys.path.insert(0, str(args.reference.resolve()))

from tinygrad import Device  # noqa: E402
from tinygrad.codegen import to_program  # noqa: E402
from tinygrad.device import Buffer  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.engine.realize import compile_linear, link_linear, run_linear  # noqa: E402
from tinygrad.helpers import Context  # noqa: E402
from tinygrad.runtime.support.hcq2 import HCQ_CACHE_THRESH  # noqa: E402
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp  # noqa: E402

ELEMENTS = 1024
if args.calls < max(2, HCQ_CACHE_THRESH.value) or args.samples < 1:
  raise ValueError("calls must reach HCQ_CACHE_THRESH, and samples must be positive")
device = Device["METAL"]
inputs = [Buffer("METAL", ELEMENTS, dtypes.int32).allocate() for _ in range(2)]
input_uops = [UOp.from_buffer(b) for b in inputs]
dst, src = [UOp.param(i, dtypes.int32, ELEMENTS) for i in range(2)]
r = UOp.range(ELEMENTS, 0, AxisType.GLOBAL)
sink = dst.index(r).store(src.index(r).load() + 1).end(r).sink(arg=KernelInfo(name="link_chain"))
with Context(NOOPT=1):
  program = to_program(sink, device.renderer)
reference = args.reference.resolve()
git_root = subprocess.run(["git", "-C", str(reference), "rev-parse", "--show-toplevel"],
                          text=True, capture_output=True, check=False)
revision = None
if git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == reference:
  revision = subprocess.check_output(
    ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
# Hash sorted relative names and exact contents with explicit lengths. Archives
# have no verifiable Git revision; parent repositories must never stand in.
source_digest = hashlib.sha256()
source_files = sorted((reference / "tinygrad").rglob("*.py"))
if not source_files:
  raise ValueError("reference contains no tinygrad Python sources")
for path in source_files:
  name = path.relative_to(reference).as_posix().encode()
  contents = path.read_bytes()
  for part in (name, contents):
    source_digest.update(len(part).to_bytes(8, "big"))
    source_digest.update(part)


def roots(linked):
  backing = {}
  for node in linked.toposort():
    if node.op is Ops.BUFFER and isinstance(node.arg.buffer, Buffer):
      b = node.arg.buffer.base
      backing[id(b)] = b
  return list(backing.values())


def replay(linked):
  for b in inputs:
    b.host.view(fmt="B")[:] = bytes(ELEMENTS * 4)
  run_linear(linked, input_uops=input_uops, update_stats=False, jit=True, wait=True)
  device.synchronize()
  for slot, b in enumerate(inputs):
    expected = args.calls if slot == (args.calls - 1) % 2 else args.calls - 1
    if any(value != expected for value in b.host.view(fmt="i")):
      raise AssertionError("linked chain produced an incorrect result")


def run(eager):
  params = [UOp.param(i, dtypes.int32, ELEMENTS, "METAL") for i in range(2)]
  if eager:
    params = [p.replace(tag="lt_input") for p in params]
  calls = tuple(program.call(params[i % 2], params[1 - i % 2], name=str(i)) for i in range(args.calls))
  compiled = compile_linear(UOp(Ops.LINEAR, src=calls), profile=False)
  if eager and not any(u.tag == "lt_input" for u in compiled.toposort()):
    raise AssertionError("benchmark did not preserve lt_input bindings")
  retained = None if eager else link_linear(compiled, input_uops=input_uops, allow_cache=False)
  if retained is not None:
    replay(retained)

  def sample():
    gc.collect()
    before = [s["collections"] for s in gc.get_stats()]
    start = time.perf_counter_ns()
    linked = link_linear(compiled, input_uops=input_uops, allow_cache=eager)
    link_ns = time.perf_counter_ns() - start
    after = [s["collections"] for s in gc.get_stats()]
    backing = roots(linked)
    start = time.perf_counter_ns()
    for b in backing:
      b.ensure_allocated()
    materialize_ns = time.perf_counter_ns() - start
    backing = [b for b in backing if all(b is not inp for inp in inputs)]
    row = {"link_ns": link_ns, "materialize_ns": materialize_ns,
           "linked_storage_roots": len(backing), "linked_storage_bytes": sum(b.nbytes for b in backing)}
    row.update({f"python_gen{i}_collections": a - b for i, (a, b) in enumerate(zip(after, before))})
    replay(linked)
    if retained is not None:
      replay(retained)
    return row

  for _ in range(3):
    sample()
  rows = [sample() for _ in range(args.samples)]
  print(json.dumps({"backend": "METAL", "mode": "eager_lt_input" if eager else "retained_independent",
                    "calls": args.calls, "elements": ELEMENTS, "samples": args.samples,
                    "hcq_cache_thresh": HCQ_CACHE_THRESH.value, "python": sys.version,
                    "reference": revision, "reference_path": str(reference),
                    "reference_python_sha256": source_digest.hexdigest(),
                    "reference_python_files": len(source_files),
                    **{k: statistics.median(row[k] for row in rows) for k in rows[0]}}), flush=True)


run(True)
run(False)
