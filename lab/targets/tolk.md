# Target: tolk

tolk's compiler, which turns a tensor graph into kernels' source: preparing
the graph (prepare), ranges and kernel splitting (kernel_graph), the order of
the kernels (schedule), per-kernel optimization and lowering (codegen),
linearization to instructions (linearize), and source emission (render). The
benchmark times each stage on its own over graphs recorded from tinygrad,
among them a block of gpt-oss prefilling and decoding, so a regression
localizes to one stage of one program. A model's first compiled call spends
most of its time in these stages.

## Commands

- **Build:** `dune build packages/tolk/bench/bench_tolk.exe`.
- **Test (correctness gate):** `dune build @packages/tolk/runtest`. This is
  the ruler: the goldens recorded from tinygrad pin every stage's output, the
  kernel graphs, schedules, lowered sinks and rendered sources included, so a
  change that alters what a stage emits **must fail** it. That is the guard,
  not a nuisance. It does not run the benchmark.
- **Baseline:** `packages/tolk/bench/tolk.thumper`, partitioned per machine;
  the session refreshes it at setup and ratchets on keeps.
- **BENCH:** the built executable, run **directly**, never through `dune exec`:
  `<WT>/_build/default/packages/tolk/bench/bench_tolk.exe`.
  The suite is `tolk`; the lab subset is `--tag lab`, every compiler stage
  (the `kernels` cases, which time compiled kernels, are outside it). Example
  gate run:

  ```
  rm -f <RESULTS>/verdict.json <WT>/<BASELINE>.corrected
  <BENCH> --tag lab \
    --baseline <WT>/<BASELINE> \
    --json <RESULTS>/verdict.json
  ```

## In scope (may edit)

- `packages/tolk/lib/**`, chiefly the stages the benchmark times:
  - `schedule/prepare.ml`, `schedule/indexing.ml` and `schedule/rangeify.ml`:
    preparing and the kernel graph, the second costliest stage;
  - `schedule/schedule.ml`: the order of the kernels;
  - `codegen/**`: optimization and lowering (`codegen/codegen.ml`,
    `codegen/opt/**`, `codegen/simplify.ml`, `codegen/decomp/**`,
    `codegen/late/**`), the costliest stage;
  - `uop/ops.ml`, `uop/symbolic.ml`: graph rewriting and the symbolic rules
    every stage runs;
  - `renderer/**`: source emission.

tolk is a port of tinygrad, file for file. A speedup keeps each function the
port of its tinygrad function: it may change how the OCaml computes a result,
never which result, nor do something tinygrad does not. A change that would
needs an entry in `packages/tolk/DIVERGENCES.md`, whose reasons a speedup is
not, so it is out of scope.

## Read-only / never touch

- `packages/tolk/bench/**` (sources and `tolk.thumper`) and
  `packages/tolk/test/**` (the goldens, their generators and every suite).
  They are the ruler. Optimizing the ruler is cheating; such a change is void.
- The op set, `packages/tolk/lib/uop/op.ml`.
- Anything outside `packages/tolk/lib/`.

## Keep rule

The standard lab pair-based rule from `lab/program.md` applies verbatim:

- the target tests pass; and
- no case has a `wall_time` `regressed` relation in **both** runs (a real
  regression reproduces on the same case; a one-run blip on a case the change
  cannot affect is noise); and
- no case has an `alloc_words` `regressed` relation in **either** run;
  allocation is deterministic, so one alloc regression discards immediately;
  and
- at least one case is `improved` (`wall_time` or `alloc_words`) in **both**
  runs, *or* the change is a strict-LOC-decrease simplification with no
  reproduced wall regression and no alloc regression.

`alloc_words` is the sharpest tool here: every stage is a pure graph rewrite,
so its allocation reproduces exactly, and an O(n²) list copy shows up as an
allocation jump before wall-time noise matters.

## Perf context

### Stage seam map

The benchmark times these functions, each fed its input built in `setup`:

| Stage | Function | Owner | In → out |
|---|---|---|---|
| prepare | `Prepare.prepare_rangeify` | `schedule/prepare.ml` | tensor graph → prepared graph |
| kernel_graph | `Rangeify.get_kernel_graph` | `schedule/rangeify.ml`, `schedule/indexing.ml` | prepared graph → kernel graph |
| schedule | `Schedule.create_schedule` | `schedule/schedule.ml` | kernel graph → `Linear` of calls |
| codegen | `Codegen.full_rewrite_to_sink` | `codegen/**` | each kernel → lowered sink |
| linearize | `Linearizer.linearize`, then `Codegen.pm_linearize_cleanups` | `codegen/late/linearizer.ml`, `codegen/codegen.ml` | lowered sink → instructions |
| render | the renderer's `render` | `renderer/cstyle.ml` | instructions → C source |

Kernels are lowered and rendered for the CPU's C renderer
(`Cstyle.clang`), on a fixed architecture, so every machine times the same
work; nothing is compiled. Codegen dominates: gpt-oss's block spends about
130-210 ms there and 30-37 ms in kernel_graph on an M1 Max, and every other
stage is at most a few milliseconds.

### Caches

The benchmark calls each stage directly, so neither the schedule cache of
`Schedule.create_linear_with_vars` (`SCACHE`) nor the program cache of
`Codegen.to_program` serves a timed case. The hash-consing of nodes is
process-wide: a case's setup builds its input once, and the timed closure
runs only the stage, so repeated batches time the stage, not graph
construction.
