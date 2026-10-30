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
  maintainer and other agents run dune in watch mode. Kill by pid, never by
  pattern (`pkill`, `killall`).
- NEVER edit a file under `_build`. dune hardlinks it into the shared cache,
  so the edit corrupts every tree that reads that entry.
- NEVER exhaust a system resource (threads, processes, descriptors, memory,
  disk), in a test or otherwise: it takes down every session on the
  machine. A failure path that needs a limit sets it on a forked child.
- NEVER silence a warning or prefix a variable with `_` to hide it. A warning
  is a bug in the change.
- NEVER add an nx backend operation without being asked.
- NEVER add a dependency without being asked. raven depends only on OCaml
  libraries whose types its public interfaces expose for composition
  (`doc/dev/design.md`).
- NEVER cite an RFC or a ledger from code or an `.mli`. The `.mli` and the code
  are the source of truth.
- NEVER copy sources between directories (`copy_files`, copy rules, an
  `#include` of another directory's `.c`), and never reach into another
  library's private files from a test. Share code through a library.

## Packages

| Package | Role |
|---|---|
| `packages/nx` | n-dimensional arrays on sets of rig's devices. Libraries: `nx.array` (dtypes, layouts, arrays over device buffers), `nx.kernel` (the kernel contract), `nx.cpu` (C kernels for host memory), `nx`, `nx.io`, `nx.quant`, `nx.ragged`, `nx.wide`. No eager GPU kernels: a GPU set computes under `Rune.jit` |
| `packages/rig` | devices: Metal, CUDA, NVIDIA and AMD GPUs through their kernel drivers or over PCI, files and remote machines; the order of work across them |
| `packages/rune` | autodiff, vmap, loops and compilation over nx. Departures from the semantics it follows are recorded in `packages/rune/DIVERGENCES.md` |
| `packages/tolk` | the compiler, a port of tinygrad. Departures from tinygrad are recorded in `packages/tolk/DIVERGENCES.md` |
| `packages/kaun` | layers, losses and data batching over nx structures. Models live in `examples/`, never in the library |
| `packages/vega` | optimizers as values |
| `packages/jera` | numerical methods: quadrature, interpolation, differential equations, splittings |
| `packages/norn` | probabilistic inference over structures: distributions, bijectors, samplers, diagnostics; the model language is `norn.model` |
| `packages/lz` | compression between bigarrays: deflate, zlib, gzip, Zstandard, LZ4, Snappy and CRC-32, in raven's own C kernels; `lz.bytesrw` filters |
| `packages/talon`, `hugin`, `brot`, `quill`, `munin` | dataframes, plotting, tokenizers, notebooks, run monitoring |
| `packages/ppx_ptree` | derives a structure's `walk` from its type, for users; no raven library depends on it |
| `contrib/` | fehu, sowilo, ymir: own `dune-project`, own `CHANGES.md`, public core libraries only |

A package's main library lives at `lib/`, each sub-library at
`lib/<name>/`, named without the package prefix. tolk keeps its
`include_subdirs` module groups. A package's own `AGENTS.md` holds its
boundaries, decisions and traps.

Accepted designs live in `doc/rfc/`. RFCs from 0010 on are not committed.

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
  `dune build` and `dune runtest` forward to it, always under `timeout`,
  since a stalled server hangs its clients forever. Before each, run
  `timeout 5 dune rpc ping`: no answer means the server is stuck, so kill
  its pid and start another. A build sent right after an edit can build
  the old file: let the edit settle a couple of seconds, and check the
  `_build` copy before trusting a result. A server that answers but builds files
  older than the tree (its errors name code you changed) is stuck too:
  before killing it, save `lsof -p <pid>`, `sample <pid> 5` and the time
  of the last change it noticed to `_plans/dune-stale-watch/`, so the
  cause can be found. The server watches every `$PATH` directory,
  and for one that doesn't exist, its nearest existing parent: a missing
  entry can make it watch `$HOME` and rebuild on every change there.
- **Release.** When your task is done, kill your server, leave the tree
  clean and detached at main, then `git worktree unlock <path>`. A lock
  whose session has ended is stale; the lead clears it.
- **Remote machines.** Never end a turn while a job on another machine
  runs: nothing wakes an agent when it finishes. Run it in the foreground
  with a timeout, or wait on a local command that blocks until it ends,
  and if word is late, check its pid or log yourself.

## Commands

- Build or test one package: `dune build @packages/<pkg>/runtest`. Don't
  wait for the machine to be quiet; only timing needs that.
- Rerun a cached test: run its executable from `_build/default/…`, from the
  directory holding its goldens, with `CACHEDB=cache`. Never use `--force`.
- tolk goldens: `uv run packages/tolk/test/gen/generate.py --check`.
  Regenerate goldens with the generator; never edit them by hand.
- Python: always through `uv run`.

## How we design

Before any design or semantic decision, work through `doc/dev/design.md`:
the constraint we design for, how to trade correctness, understanding and
speed, the hard constraints, how to shape a design, how to become certain
of it from references, and how to review it.

## Code and docs

- Modules and variants are `Capitalized_snake_case`; values `snake_case`.
- Doc comments live in `.mli` only and start `(** [f x] …`. Operations that
  match on dtypes take explicit type annotations, as in
  `let f (type a b) (t : (a, b) t) =`.
- API docs (`.mli` doc comments, private modules' included, and guide
  pages under `doc/`) follow `doc/dev/documentation.md`.
- A block that isn't obvious gets a short comment: what it does and why, with
  an example when it helps. ASCII diagrams for whole systems. Don't narrate
  the code's arrangement. Section headers are plain `(* Name *)`.
- Don't rename during a tidy unless the name gets clearly better.
- Shallow code: early returns, no arrow-shaped nesting. Blank lines between
  logical blocks.
- Short function names, under 30 characters. An argument that selects
  behaviour is a variant, or a labelled `bool` where that reads better at
  the call site.
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
When a test needs what the interface hides, the library exposes the minimum
behind a `(**/**)` stop (in C, a section marked for its tests at the end of
the installed header). No test reads META, dune files or source text: dune
files and review hold the structure.

A failing test is diagnosed from the failure, the code it runs and that
code's recent diffs, never by bisecting.

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
  `-0.`, infinities, int extremes. `cover` the cases a law needs, and draw
  each covered case often enough that every run reaches it: a cover that
  some seeds miss is a flaky test.
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
  it says why. A bench runs only through the `bench` alias, never in
  `dune build` or `dune runtest`.
- Time only on a quiet machine. Correctness runs need no quiet.

## Reviewing

A review reads `REVIEW.md` and the `REVIEW.md` of each package the change
touches, and reports as they say.

A change that touches a public `.mli`, library code, C or a contract gets a
review by an agent that did not write it, and lands once each finding is
fixed or answered with a reason. A design finding means redesigning before
landing. An author rejects a finding only after reading the whole function,
its callers and its history, and gives that evidence.

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
