# Target: nx

N-dimensional arrays — the numpy analog. Most performance lives in the C backend
kernels and in how the frontend composes operations (avoiding copies and
materializations).

## Commands

- **Build:** `dune build packages/nx/bench/bench_nx.exe` — building the bench exe
  pulls in `packages/nx/lib` transitively, so this one command builds both.
- **Test (correctness gate):** `dune build @packages/nx/test/runtest` — scoped to
  the test dir; it does not run the benches.
- **Baseline:** `packages/nx/bench/nx.thumper` — the committed baseline the
  session refreshes at setup and ratchets on keeps. The subdir suites use the
  `.thumper` next to their exe (`matmul/nx_matmul.thumper`, etc.).
- **BENCH:** the built executable, run **directly**, never through `dune exec`:
  `<WT>/_build/default/packages/nx/bench/bench_nx.exe`
  Suite name is `nx`; lab subset is `--tag lab`. Running the exe directly is
  required, not stylistic: `dune exec` sets `INSIDE_DUNE`, which makes
  `--bless --baseline PATH` write `PATH.corrected` instead of `PATH` and
  suppresses the corrected-file write the ratchet needs, and it forces a `--`
  separator that would swallow thumper's own flags. Build with the dune command
  above, then invoke the exe path with thumper's flags appended (no `--`).

One executable per session. Other nx suites you may target instead (each its own
session, same exe-path rule):
`<WT>/_build/default/packages/nx/bench/matmul/bench_matmul_nx.exe`,
`.../conv2d/bench_conv2d_nx.exe`, `.../einsum/bench_einsum_nx.exe`.

## In scope (may edit)

- `packages/nx/lib/core/frontend.ml` — op composition, views, broadcasting,
  contiguity decisions.
- `packages/nx/lib/backend_c/**` — the C backend: OCaml dispatch and the C
  kernels. Most elementwise / reduction / matmul wins are here.
- `packages/nx/lib/core/**` **except** `backend.mli`.

## Read-only / never touch

- `packages/nx/lib/core/backend.mli` — never add or change a backend op (hard
  project rule).
- `packages/nx/bench/**`, `packages/nx/test/**`, every `*.thumper`.
- Anything outside `packages/nx/lib/`.

## Perf context

- The backend is multithreaded: cpu_time ≫ wall_time. Decide on wall_time and
  alloc_words; ignore cpu_time.
- Known slow paths worth attacking first: reductions along the non-contiguous
  axis (`sum ~axes:[1]` on a C-contiguous array), materializing a transposed
  view (`contiguous` of a transpose), broadcasting binary ops, and the einsum
  reduce / independent-sum cases (already flagged slow in that suite).
- Documented anomalies, present in the lab subset as cases to attack: `sum
  128x128` sits near a parallelization threshold (the full sum was ~27× slower
  at 100×100 than at 50×50), and `add/mul 100x100 f64` are far slower than their
  f32 twins (a dtype cliff, not a bandwidth effect).
- Elementwise ops allocate a fresh output each call. Reducing per-op allocation
  or avoiding an intermediate copy is a common, mechanism-clear win. Never
  change an op's result — the tests pin every value.
