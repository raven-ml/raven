# AGENTS.md

raven brings modern numerical computing and machine learning to OCaml: arrays,
autodiff and compilation, neural networks, dataframes, plotting, tokenizers
and notebooks, each a small library that does one thing well.

## Hard rules

These protect the maintainer's work and other sessions on the same machine.
Breaking one causes real damage.

- NEVER push.
- NEVER pass `--force` to git or dune. Never run `dune clean`, never pass
  `--build-dir` or `DUNE_CACHE=disabled`, never delete or relock `dune.lock`
  without being asked.
- NEVER kill a dune you didn't start, unless its session has ended: the
  maintainer and other agents run dune in watch mode.
- NEVER silence a warning or prefix a variable with `_` to hide it. A warning
  is a bug in the change.
- NEVER add an nx backend operation without being asked.
- NEVER add a dependency to the project, raven is purposefully zero-dependency.
- NEVER cite an RFC or a ledger from code or an `.mli`. The `.mli` and the code
  are the source of truth.

## Packages

| Package | Role |
|---|---|
| `packages/nx` | n-dimensional arrays. Libraries: `nx.dtype`, `nx.device` (host, Metal, CUDA, AMD, NV, remote, disk runtimes), `nx.array`, `nx.cpu` (C kernels), `nx`, `nx.io`, `nx.quant`, `nx.ragged` |
| `packages/rune` | autodiff, vmap and compilation over nx |
| `packages/tolk` | the compiler, a port of tinygrad. Departures from tinygrad are recorded in `packages/tolk/DIVERGENCES.md` |
| `packages/kaun` | layers, optimizers and training on rune. Models live in `examples/`, never in the library |
| `packages/vega` | optimizers as values |
| `packages/jera` | numerical methods: quadrature, interpolation, differential equations, splittings |
| `packages/norn` | probabilistic inference over structures: distributions, bijectors, samplers, diagnostics; the model language is `norn.model` |
| `packages/talon`, `hugin`, `brot`, `quill`, `munin` | dataframes, plotting, tokenizers, notebooks, run monitoring |
| `contrib/` | fehu, sowilo: own `dune-project`, own `CHANGES.md`, public core libraries only |

A package's main library lives at `lib/`, each sub-library at
`lib/<name>/`, named without the package prefix. tolk keeps its
`include_subdirs` module groups.

Accepted designs live in `doc/rfc/`.

## Parallel agents

Agents work in a shared pool of worktrees, `../raven-pool/<n>` next to the
main checkout, and never change the main checkout's working tree (no
`git stash`, `checkout`, `reset`, `restore` or `clean` there). Worktrees
are reused, never removed: their warm `_build` is the point. Use
`git -C <path>`, never `cd <path> && git …`.

Don't burn tokens. Give a new stream to a fresh agent with a short brief
instead of a long-running one.

- **Claim.** A worktree is taken while git holds it locked. Claim one with
  `git worktree lock --reason "<session> <agent> <task>" <path>`, which
  fails if it is already locked; `git worktree list --verbose` shows each
  lock's reason. If none is free, add the next number with
  `git worktree add --detach --lock --reason "…" <path> main`, then copy
  the checkout's `dune.lock` into it. Touch nothing in a worktree before
  its lock is yours. Work on a branch named for the task.
- **Keep.** The claimant keeps its branch rebased on main and copies the
  checkout's `dune.lock` when it changes.
- **Build.** Run one `dune build --passive-watch-mode` in the background;
  `dune build` and `dune runtest` forward to it. Before each, run
  `timeout 5 dune rpc ping`: no answer means the server is stuck, so kill
  its pid and start another. A server that answers but builds files
  older than the tree (its errors name code you changed) is stuck too:
  before killing it, save `lsof -p <pid>`, `sample <pid> 5` and the time
  of the last change it noticed to `_plans/dune-stale-watch/`, so the
  cause can be found. The server watches every `$PATH` directory,
  and for one that doesn't exist, its nearest existing parent: a missing
  entry can make it watch `$HOME` and rebuild on every change there.
- **Release.** When your task is done, kill your server, leave the tree
  clean and detached at main, then `git worktree unlock <path>`. A lock
  whose session has ended is stale; the lead clears it.

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
- **Argue for the most elegant shape.** Each new concept (a module type,
  functor, scope, cache, mode or optional argument) weighs its alternatives —
  plain data, an existing noun, composition, not exporting it — and shows the
  chosen one captures the problem's structure with the fewest concepts and each
  fact in one place. The simplest mechanism often wins, but not when it
  scatters a fact across callers. A design that can't make the case is not
  ready.
- **Review the shape before landing.** A change to a public `.mli` lands only
  after a review of its API shape against this section. Passing tests and a
  correctness review don't make it ready.
- **Briefs state problems.** A brief to an agent gives the problem, its
  constraints and the decisions already made, never a mechanism nobody
  weighed.

## Code and docs

- Modules and variants are `Capitalized_snake_case`; values `snake_case`.
- Doc comments live in `.mli` only and start `(** [f x] …`. Operations that
  match on dtypes take explicit type annotations, as in
  `let f (type a b) (t : (a, b) t) =`.
- A private module's `.mli` is its contract with its siblings: a `(**`
  preamble naming what it holds, then its invariants, concurrency, errors
  and, for C, ownership, where they exist. A value gets a doc only for what
  its name, type and the public docs don't say: preconditions, raises,
  blocking, ownership, units, effects. A module implementing a public one
  points to it and documents only what differs. The representation of an
  abstract type is commented at the type in the `.ml`.
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

Tests use windtrap, whose `.mli` (under `_build/_private/default/.pkg/`) is
the contract, and cram tests for executables. A test states what the `.mli`
promises, through the public interface, never what the code does today.

- Every test runs in its package's `runtest`, which runs within 2 minutes on
  a warm build. A slow test is made fast at its cause (the library, shared
  setup, a smaller input whose `cover` shows it still reaches its cases),
  never moved behind a tag, an opt-in alias or a scheduled job. Only a test
  that needs what a test run cannot hold, such as a real model's weights,
  lives outside `runtest`, and it says why.
- Use the strongest tool the claim allows: `prop` for a law (agreement with
  a simpler reference, a round trip, an identity, `jit` equal to eager,
  `grad` against finite differences), with a `Law` verb when one names it;
  `stateful` against a model for state across calls (pools, caches,
  stores); `cases` for values the spec states; `expect` for long text.
- Expected values come from the spec. One learned by running the code is a
  baseline, written as `expect`.
- Draw inputs that break the code: zero-size and one-element shapes, bounds
  and their neighbours, every dtype, strided and broadcast views, NaN,
  `-0.`, infinities, int extremes. `cover` the cases a law needs.
- Assert with verbs that print the data (`equal`, `less ~than`, `raises`)
  and floats with a stated tolerance, never `is_true`.
- Write a test where a user would get a wrong answer or a crash. Fixing a
  bug: write the test first, watch it fail, then fix and watch it pass.
- Layout: `test/` mirrors `lib/`, with the main library's suites as
  `test/test_*.ml` in one `dune` file, each sub-library's in `test/<name>/`,
  and `support/`, `golden/` and `gen/` at the level that needs them.
- A structure shared between domains gets `stateful ~domains:2`, which
  explains every result by some order of the calls. A known interleaving is
  held deterministically: don't loop and hope. No mutation runs, no
  after-the-fact fail-without proofs.
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
- A performance fix lands with a row that would catch its regression, unless
  one already does. The row times what a user calls (an Nx operation, a
  compiled step, a search), never the internals the fix changed, so it
  survives their rewrite. Its baseline is recorded on every machine the fix
  targets.
- Each package's bench suite runs within 2 minutes warm on every machine. A
  row that needs more is shrunk to the smallest size that still shows its
  regression, never moved out of the suite. Only a run that a suite cannot
  hold, such as gpt-oss's decode step on real weights, lives outside it, and
  it says why. Benches never run in `dune build`.
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
