# tolk benchmarks

Tolk turns a tensor-level graph into runnable kernels through a fixed pipeline:
range analysis and kernel splitting (**rangeify**), kernel scheduling and memory
planning (**schedule**), per-kernel optimization and lowering (**codegen**),
linearization to an SSA program (**linearize**), backend source emission
(**render**), and host compilation (**compile**). These benchmarks measure how
fast that pipeline runs — the cost of *compiling* a graph, not of executing the
kernels it produces — with one small runtime-throughput suite for context.

Every workload is built once, in one place: `graphs/graphs.ml` holds the graph
builders (`elementwise`, `reduce`, `matmul_small`, `attention`, and the
`lorenz` / `rnn` scaling ladders). Both the gate and the comparison link that
library, so they always time the identical graph.

## The three executables

| Path | Role | Gated? |
|---|---|---|
| `bench_tolk.ml` → `bench_tolk.exe` | **Gate.** Per-stage microbenchmarks with a committed baseline; the regression tripwire the lab loop runs each iteration. | yes (`tolk.thumper`) |
| `compare/` | **Diagnostic.** Times each tolk stage against its counterpart in the pinned reference implementation and prints a per-stage tolk-vs-reference table. Run on demand. | no |
| `runtime/` | **Indicative.** Runtime throughput (GFLOP/s, GB/s) of a few compiled kernels, for context only. Run on demand. | no |

Run every executable **directly from `_build`**, never through `dune exec`:
`dune exec` sets `INSIDE_DUNE`, which redirects `--bless` output and forces a
`--` separator that swallows the tool's own flags.

## The gate: `bench_tolk`

Each fixed-size workload is timed one pipeline stage at a time. The stage input
is built once in `setup`; the timed closure runs the single pass, so a
regression or a superlinear cost localizes to one stage (case path
`<workload>/<stage>`). Metrics are wall time (primary) and allocated words —
allocation is deterministic, so it catches an O(n²) list copy before wall-time
noise does.

Build and run the lab subset:

```
dune build packages/tolk/bench/bench_tolk.exe
SCACHE=0 ./_build/default/packages/tolk/bench/bench_tolk.exe --tag lab
```

`SCACHE=0` disables the schedule engine's semantic-key cache so repeated passes
measure cold work; it is read once at process start, so it must be in the
environment before launch (the `@bench` dune rule sets it, and `CCACHE=0`, for
you). Useful flags: `--explore` (measure, no baseline check), `--bless` (write a
new baseline), `--list`, `--case <id>`, `--tag lab`, `--baseline <file>`.

**What `--tag lab` covers** — the tight, ~2-minute iteration gate:

- the graph-transform stages (rangeify, schedule, codegen, linearize, render) of
  the small fixed workloads `elementwise`, `reduce`, `matmul_small`;
- the same stages of `attention`, **except** its `codegen` — that pass is the
  single priciest case in the suite (four kernels), so it is left untagged to
  keep the gate under budget;
- `rangeify` of the `lorenz` and `rnn` scaling ladders up to a bounded size,
  where the historical superlinearity risk lives and where allocation growth is
  the detector.

**What stays untagged** (present in the suite, run on demand without `--tag`):

- `attention/codegen`, and every workload's `schedule` / `codegen` at the larger
  scaling-ladder sizes — the expensive passes;
- the stage-7 `compile` case of each fixed workload (`<workload>/compile`) — it
  shells out to `clang`, is wall-time-only, and is disk-cached, so run it with
  `CCACHE=0` to measure a cold compile:

```
CCACHE=0 SCACHE=0 ./_build/default/packages/tolk/bench/bench_tolk.exe -f compile --explore
```

## The diagnostic: `compare/`

Answers "where is tolk slower than the reference, and by how much, per stage?"
Three pieces, joined on `(workload, size, stage)`:

- `bench_compare.ml` → `bench_compare.exe` — times the tolk stages, writes
  `tolk.json` and `tolk.verify.json` to an output directory.
- `bench_compare.py` — builds the *same* graph with the pinned reference's
  primitives (mirroring `graphs.ml` node for node) and times the counterpart
  stages, writing `tinygrad.json` and `tinygrad.verify.json`.
- `report.py` — joins the two, prints the comparison table, writes
  `compare.tsv` (and `scaling.tsv` for the ladders), and runs the same-graph
  cross-check.

```
OUT=/tmp/tolk-cmp && mkdir -p "$OUT"
dune build packages/tolk/bench/compare/bench_compare.exe
SCACHE=0 CCACHE=0 ./_build/default/packages/tolk/bench/compare/bench_compare.exe "$OUT"
uv run packages/tolk/bench/compare/bench_compare.py "$OUT"
uv run packages/tolk/bench/compare/report.py "$OUT"
```

Both drivers warm the exact call once, then report the median and min of N warm
samples on a monotonic clock, in one long-lived process — so interpreter startup
and library-import cost are excluded and the comparison is algorithm-vs-algorithm
on an identical graph. Both disable every cache (`SCACHE=0 CCACHE=0` on the tolk
side; the Python side sets the equivalents before import) so both measure cold
work. The `compare/` outputs are scratch files; they are not committed.

**Same-graph cross-check.** `report.py` verifies that both sides emit the same
number of kernels and a byte-identical first kernel; a mismatch means the two
sides are not compiling the same graph, and the report fails loudly rather than
present a meaningless table. This is what makes each row a like-for-like
comparison — a stage-transform benchmark is only honest if both stages consume
the identical graph.

## The context suite: `runtime/`

Runtime throughput of a few compiled kernels (matmul GFLOP/s; elementwise,
reduce, and host→device copy GB/s), execution-only, with compilation held
outside the timed loop. These numbers are context, not a target — closing a
runtime gap would mean changing compiler semantics, which is out of scope.

```
dune build packages/tolk/bench/runtime/bench_runtime.exe
./_build/default/packages/tolk/bench/runtime/bench_runtime.exe
uv run packages/tolk/bench/runtime/bench_runtime.py
```

The backend defaults to CPU; set `DEV=METAL` (etc.) to run on another device.

## Baseline and blessing

The gate's baseline is `tolk.thumper`, committed and partitioned per machine.
The lab loop ratchets it on confirmed improvements and re-blesses only through
the loop's promote step. The full target contract — build and correctness-gate
commands, the in-scope hot paths, the keep rule, and the stage seam map — lives
in `lab/targets/tolk.md`; follow the bless and ratchet procedure there rather
than re-blessing by hand.

When adding a new lab-tagged workload, bless it **additively**: bless to a
throwaway baseline file, append only the new rows to the committed
`tolk.thumper`, and confirm every existing row is byte-unchanged
(`git diff --numstat` shows added rows only). A whole-section re-bless would
silently move unrelated cases.
