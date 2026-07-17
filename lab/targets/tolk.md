# Target: tolk

Compile pipeline that turns a tensor graph into runnable kernels — the
tinygrad-analog lowering stack. A tensor-level `Sink` is lowered in stages:
range analysis and kernel splitting (rangeify), kernel scheduling and memory
planning (schedule), per-kernel optimization and lowering (codegen),
linearization to an SSA program (linearize), and backend source emission
(render). The benchmark times each stage in isolation on fixed-size workloads,
so a regression localizes to one stage.

## Commands

- **Build:** `dune build packages/tolk/bench/bench_tolk.exe` — building the
  bench exe pulls in `packages/tolk/lib` transitively.
- **Test (correctness gate):** `dune build @packages/tolk/test/runtest` —
  scoped to the test dir. This is the ruler: the parity goldens diff emitted
  source byte-for-byte against the reference (~70 cases through stage 5 and
  stage 7), plus the unit battery. The whole point of this target is to speed
  the pipeline up *without changing its output*, so a change that alters
  emitted source **must fail parity** — that is the guard, not a nuisance. It
  does not run the benches.
- **Baseline:** `packages/tolk/bench/tolk.thumper` — the committed baseline,
  partitioned per machine; the session refreshes it at setup and ratchets on
  keeps.
- **BENCH:** the built executable, run **directly**, never through `dune exec`:
  `<WT>/_build/default/packages/tolk/bench/bench_tolk.exe`
  Suite name is `tolk`; lab subset is `--tag lab`. Run it with `SCACHE=0` in
  the environment (see Perf context). Running the exe directly is required, not
  stylistic: `dune exec` sets `INSIDE_DUNE`, which makes `--bless --baseline
  PATH` write `PATH.corrected` instead of `PATH` and suppresses the
  corrected-file write the ratchet needs, and it forces a `--` separator that
  would swallow thumper's own flags. Example gate run:

  ```
  rm -f <RESULTS>/verdict.json <WT>/<BASELINE>.corrected
  SCACHE=0 <BENCH> --tag lab -q \
    --baseline <WT>/<BASELINE> \
    --json <RESULTS>/verdict.json
  ```

## In scope (may edit)

- `packages/tolk/lib/**` — chiefly the compile-pipeline hot paths:
  - `schedule/rangeify.ml` — range analysis and kernel splitting (the heaviest
    stage; see the O(n²) suspects below).
  - `engine/schedule.ml` — kernel scheduling and buffer memory planning.
  - `callify.ml` — call resolution feeding rangeify.
  - `codegen/**` — the optimize + lower passes (`codegen/codegen.ml`,
    `codegen/opt/**`, `codegen/simplify.ml`, `codegen/decomp/**`).
  - `codegen/late/linearizer.ml` — linearization to an SSA program.
  - `renderer/**`, `renderer.ml` — backend source emission.

## Read-only / never touch

- `packages/tolk/bench/**` (sources, `graphs/`, and every `*.thumper`) and
  `packages/tolk/test/**` (parity goldens *and* any scaling assertion). They
  are the ruler. Optimizing the ruler is cheating; such a change is void.
- The uop op set and backend interfaces — `packages/tolk/lib/uop/**`. Adding or
  changing a uop op is off-limits per house rule.
- Anything outside `packages/tolk/lib/`.

## Keep rule

The standard lab pair-based rule from `lab/program.md` applies verbatim:

- the target tests pass; and
- no case has a `wall_time` `regressed` relation in **both** runs (a real
  regression reproduces on the same case; a one-run blip on a case the change
  cannot affect is noise); and
- no case has an `alloc_words` `regressed` relation in **either** run —
  allocation is deterministic; one alloc regression discards immediately; and
- at least one case is `improved` (`wall_time` or `alloc_words`) in **both**
  runs, *or* the change is a strict-LOC-decrease simplification with no
  reproduced wall regression and no alloc regression.

`alloc_words` is the sharpest tool here: every stage is a pure graph transform,
so its allocation is exactly reproducible, and an O(n²) list copy shows up as a
super-linear allocation jump before wall-time noise matters.

## Perf context

### Stage seam map

The bench times these functions, each fed its stage input built in `setup`:

| Stage | Function | Owner | In → out |
|---|---|---|---|
| rangeify | `Rangeify.get_kernel_graph` | `schedule/rangeify.ml` | tensor `Sink` → kernel graph |
| schedule | `Schedule.create_schedule` then `Schedule.memory_plan_rewrite` | `engine/schedule.ml` | kernel graph → planned `Linear` |
| codegen | `Codegen.full_rewrite_to_sink ~optimize ren k` | `codegen/codegen.ml` (+ `opt/`, `simplify.ml`) | per-kernel AST → lowered sink |
| linearize | `Linearizer.linearize` | `codegen/late/linearizer.ml` | lowered sink → program |
| render | `Renderer.render ren ~name program` | `renderer.ml`, `renderer/cstyle.ml` | program → backend source |

The bench uses the CPU (clang) renderer (`Cstyle.clang_no_abi X86_64`) — the
same renderer the parity `cpu` goldens bless through, so the render stage stays
on a proven path. Stage 7 (device compile) is out of scope: it shells out to
the toolchain, is disk-cached and noisy, and does not belong in the tight gate.

The current lab subset (15 cases) shows codegen as the dominant cost —
`matmul_small/codegen` is ~14 ms and ~6.9 Mw, an order of magnitude above every
other case — and rangeify second (~80–170 µs). Schedule/linearize/render are
µs-scale. Optimization effort is best aimed at codegen and rangeify.

### Byte-identical output invariant

Emitted source must not change. The correctness gate diffs stage-5 and stage-7
output against goldens byte-for-byte; a pipeline speedup that alters the uop
dump or the rendered source fails the gate. Treat any output diff as a bug in
the change, not a golden to update.

### O(n²) rangeify suspects

`schedule/rangeify.ml` has two range-accumulation helpers that append to a list
inside a per-node fold, giving O(n²) behavior on straight-line graphs:

- `add_unique acc r = ... else acc @ [ r ]` (~line 1372), folded over kernel
  ranges (~lines 1381/1389/1394).
- `add_unique xs x = if List.exists ((==) x) xs then xs else xs @ [ x ]`
  (~line 1482), folded over graph nodes (~line 1490).

Both are `xs @ [x]` (append-copies the whole list) plus a linear membership
scan. On a large straight-line graph (e.g. a long fold) this is the leading
superlinearity suspect. The Phase-1 workloads are small and fixed-size, so they
do not exercise this — but it is the known hot spot a scaling workload would hit,
and the natural first target. (Line numbers drift as the file is edited; match
on the `add_unique` / `@ [` pattern.)

### Caches to keep disabled during measurement

- **`SCACHE=0`** — the schedule engine's semantic-key cache
  (`engine/schedule.ml`, read once at module init). The Phase-1 stage benches
  call `create_schedule` directly and never consult it, but keep it disabled so
  later comparative runs (which go through `lower_sink_to_linear`) measure cold
  work.
- **`CCACHE=0`** — the compiler disk cache. Only relevant to the out-of-scope
  compile stage.
- The in-process compiled-program cache and the global uop hash-cons table are
  process-global. The benches sidestep them by timing the *passes* (input built
  in `setup`), never graph construction — repeating an identical graph build
  would be cheapened by hash-consing and would not measure a cold pass.
