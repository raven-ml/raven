# AGENTS.md

raven brings modern numerical computing and machine learning to OCaml: arrays,
autodiff and compilation, neural networks, dataframes, plotting, tokenizers
and notebooks, each a small library that does one thing well.

## Hard rules

These protect the maintainer's work and other sessions on the same machine.
Breaking one causes real damage.

- NEVER stage or commit unless asked. Never push.
- NEVER pass `--force` to git or dune. Never run `dune clean`, never pass
  `--build-dir` or `DUNE_CACHE=disabled`, never delete or relock `dune.lock`
  without being asked.
- NEVER kill a dune build, dune may be running in watch mode.
- NEVER silence a warning or prefix a variable with `_` to hide it. A warning
  is a bug in the change.
- NEVER add an nx backend operation without being asked.
- NEVER add a dependency to the project, raven is purposefully zero-dependency.
- NEVER cite an RFC or a ledger from code or an `.mli`. The `.mli` and the code
  are the source of truth.

## Packages

| Package | Role |
|---|---|
| `packages/nx` | n-dimensional arrays. Libraries: `nx.dtype`, `nx.device` (host, Metal, CUDA, AMD, NV, remote, disk runtimes), `nx.array`, `nx.cpu` (C kernels), `nx`, `nx.io`, `nx.quant`, `nx.bits`, `nx.ragged` |
| `packages/rune` | autodiff, vmap and compilation over nx |
| `packages/tolk` | the compiler, a port of tinygrad. Departures from tinygrad are recorded in `packages/tolk/DIVERGENCES.md` |
| `packages/kaun` | layers, optimizers and training on rune. Models live in `examples/`, never in the library |
| `packages/vega` | optimizers as values |
| `packages/talon`, `hugin`, `brot`, `quill`, `munin` | dataframes, plotting, tokenizers, notebooks, run monitoring |
| `contrib/` | fehu, sowilo, norn: own `dune-project`, own `CHANGES.md`, public core libraries only |

Accepted designs live in `doc/rfc/`.

## Parallel agents

When several agents work several streams on one machine, each works in its
own worktree and never changes the maintainer checkout's working tree (no
`git stash`, `checkout`, `reset`, `restore` or `clean` there). Use
`git -C <path>`, never `cd <path> && git …`.

## Commands

- Build or test one package: `dune build @packages/<pkg>/runtest`. Don't
  wait for the machine to be quiet; only timing needs that.
- Rerun a cached test: run its executable from `_build/default/…`, from the
  directory holding its goldens, with `CACHEDB=cache`. Never use `--force`.
- tolk goldens: `uv run packages/tolk/test/gen/generate.py --check`.
  Regenerate goldens with the generator; never edit them by hand.
- Python: always through `uv run`.

## How we design

The bar is the design a careful library author would still defend in ten
years: few concepts, each with something a caller can point at; total
functions over plain data; illegal states unrepresentable; no knobs and no
modes; call sites that read as prose. Judge a design by what a production 1.0
needs, never by "nothing uses it yet". When a question has a principled
answer, decide it and say why; ask only for the maintainer's own calls.

- **Make the right change.** Every change moves the code toward the one true
  design, however much it touches. Diff size is never the measure; elegant
  designs usually take fewer lines, and that follows from the design.
- **Change every consumer in the same sweep.** No compatibility shims, bridge
  modules or dead fields, even if the build is broken in between.
- **Ask whether the layer is right before the second fix.** A mechanism that
  needs a second patch for the same class of bug is usually in the wrong
  place.
- **Lines are a signal.** A large diff or module usually means duplicated
  machinery or broken design. Look for it before adding more.
- **Dependencies.** No external OCaml packages and no system libraries;
  implement the small subset you need. Copy a small helper rather than create a
  shared library.
- **Library boundaries.** Each library has one job and a minimal set of dependencies.
  Always prefer to solve problems through composition rather than dependencies and
  abstractions.
- **Layers.** Each layer talks only to the one directly below it. Low-level
  mechanics (drivers, raw I/O, byte layouts) stay behind their layer; callers
  work with domain values.
- **Visibility.** Everything stays out of the `.mli` unless the design needs it
  exported. Adding to a public `.mli` is a design change that should be weighted
  as such.
- **Fix upstream.** A capability missing in nx, rune or tolk is built there,
  never worked around downstream.

## Code and docs

- Modules and variants are `Capitalized_snake_case`; values `snake_case`.
- Doc comments live in `.mli` only and start `(** [f x] …`. Operations that
  match on dtypes take explicit type annotations, as in
  `let f (type a b) (t : (a, b) t) =`.
- A block that isn't obvious gets a short comment: what it does and why, with
  an example when it helps. ASCII diagrams for whole systems. Don't narrate
  the code's arrangement. Section headers are plain `(* Name *)`.
- Don't rename during a tidy unless the name gets clearly better.
- Shallow code: early returns, no arrow-shaped nesting. Blank lines between
  logical blocks.
- Short function names, under 30 characters. A variant, not a `bool`, for an
  argument that selects behaviour.
- Name meaningful or recurring values as constants; keep a self-explanatory
  one-off inline. A value from a spec is always named.
- Hot paths: allocate outside loops, prefer loops to closures, use `unsafe_`
  accessors where bounds are already checked.
- User-facing docs explain behaviour from first principles, without comparing
  to tinygrad, JAX or NumPy.
- Prose, in docs, comments, commits and replies: as few words as possible,
  each chosen. Short sentences, no "X, not Y" constructions, no rhetorical
  triples, no punchy closing lines. No superlatives or praise; state facts,
  including bad news.

## Tests

- Write a test where a user would get a wrong answer or a crash.
- Fixing a bug: write the test first, watch it fail, then fix and watch it
  pass.
- Layout: `test/test_*.ml` in one `dune` file; only `support/`, `golden/` and
  `gen/` subdirectories.
- Laws and invariants get property tests. Concurrency tests are deterministic:
  hold the interleaving, don't loop and hope.
- No mutation runs, no after-the-fact fail-without proofs, no race loops.
- Memory tests warm the device with one uncounted run, then collect and
  synchronise a fixed number of rounds before reading the allocated count:
  a chain of finalisers frees its memory one round late, so the count can
  stop changing before it settles.
- GPU tests run on real hardware and skip without it. No mock drivers.

## Performance

- Every package with a bench keeps per-machine baselines (`*.thumper`). A
  commit that changes performance re-records its bench, in the same commit or
  a `bench(…)` follow-up. A bench going the wrong way blocks landing until it
  is explained.
- Time only on a quiet machine. Correctness runs need no quiet.

## Commits

- Subject: `type(scope): Imperative summary` (`feat`, `fix`, `perf`,
  `refactor`, `test`, `docs`, `bench`, `chore`), capitalised, no period, about
  50 characters (72 at most). It completes "If applied, this commit will …".
- Blank line after the subject; body wrapped at 72.
- The diff says what changed. The body says why: what was wrong, why it
  matters to a user, why this approach and not another, what it costs.
- Describe the change, never the process: no reviews, agents, phases or
  attempts.

## Changelog

Every user-visible change gets an entry in `CHANGES.md` (contrib packages: their
own), under the unreleased version, at the top of its `### Package` section.
Lead with what changed for the user, say why when a fix isn't obvious, name the
functions or types, and keep it to 1–3 lines with backticks for identifiers.
No entries for internal refactors, style or test-only changes.
