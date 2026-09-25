Metal linker measurement
=======================

This standalone microbenchmark measures the linker for chains of 64 or 128
compiled 1024-element integer additions. It exercises real Metal command
encoding and execution. It is not a model-throughput benchmark.

Build once with the release profile, then run directly on a quiet machine:

```
dune build --profile release packages/tolk/bench/link/bench_link.exe
_build/default/packages/tolk/bench/link/bench_link.exe --calls 64
python packages/tolk/bench/link/bench_link.py --reference _tinygrad_target --calls 64
```

Repeat with `--calls 128`, alternating implementations. Record the Raven
revision and release-build configuration alongside the JSON lines. Both
harnesses record runtime versions and `HCQ_CACHE_THRESH`. The companion records
the reference's absolute path and a deterministic SHA-256 digest of all Python
sources under its `tinygrad/` directory: sorted relative paths and contents,
each prefixed by its eight-byte big-endian length. It records a Git revision
only when the checkout's Git root is the reference directory itself. Archives
report `reference: null`, never a parent repository's revision. Their claimed
upstream revision must be verified separately against the original Git object;
the source digest identifies the measured tree but does not prove its origin.
The digest is also recorded for checkouts, whose working files may differ from
their reported commit. Keep `DEBUG=0`, `PROFILE=0`, and the same threshold
on both sides. The harness rejects call counts below that threshold.

Each mode discards three warmups and reports medians over eleven samples
(`--samples` changes this). Kernel compilation and queue compilation precede
sampling. Collection happens before each sample. Only `link_linear` is timed
by `link_ns`; `materialize_ns` separately measures ensuring all linked backing
is allocated. Execution and full output verification follow every sample,
outside both timings.

`eager_lt_input` explicitly tags input parameters and enables linking's cache
policy. These tags disable linked-schedule caching and select tinygrad's eager
ring path. `retained_independent` uses ordinary parameters and disables link
caching; an older link stays alive and is replayed after every subsequent link.
These modes intentionally compare different ownership contracts, rather than
pretending `allow_cache=false` means one-shot execution.

`linked_storage_roots` and `linked_storage_bytes` describe backing reachable
from each linked graph, excluding the two user inputs. They include shared
backend storage and full ring capacity where applicable. They are **not native
allocation counts or newly allocated bytes**: LRU reuse, native pipelines, and
indirect command objects are not separately observable through the public
interfaces. OCaml word deltas use `Gc.counters`, including the current domain's
uncollected minor allocations. `Gc.quick_stat` alone reports collection-boundary
snapshots on OCaml 5 and can misleadingly report zero allocation for a link.
It is used only for collection counts here. These measurements cover linking
plus the small clock/counter sampling overhead; they exclude graph inspection,
materialization, execution, and validation. Major words include promoted words.
Python collection counts and OCaml heap words are different metrics and must
not be compared as equivalents. The harness does not measure allocation on
other OCaml domains or native driver threads.

The benchmark establishes the scale of linking and owned backing. A faster
reference result alone does not identify ring allocation as the cause: link
patching, native API bindings, and backend object creation also differ. There
is no machine-specific pass/fail threshold and no automatic promotion.
