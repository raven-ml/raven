# Testing rig

For a contributor who adds a driver or changes rig. Paths are relative to
`packages/rig/`.

## What a test states

Each library's `.mli` is its specification. A test states one promise of it,
through the public interface, and its name is the promise: "a fault two
domains' sleeps find loses the device once". The windtrap `.mli` is the
contract for every verb below.

| Promise | Example | Verb |
|---|---|---|
| A value computed from a spec | a GPFIFO entry's fields, a QMD's bits | `cases` from the vendor header, `prop` over the field ranges |
| A reader agrees with a writer | an ELF object reads back as written | `prop` round trip |
| A refusal is total | a corrupted object reads or is refused | `prop` over corruptions, `cases` at each bound |
| State across calls | stamps, rings, windows, page tables | `stateful` against a model |
| Calls from several domains | free, unmap and unload end a value once | `stateful ~domains:2` |
| A known interleaving | a worker parking as a job is published | a held schedule: the pool's hook points, Polled's gate |
| A bounded cost | a submit that does not wait allocates nothing | words counted with `Gc.minor_words` |

- Expected values come from the spec. A value learned from the code is a
  baseline, written as `expect`.
- Draw at the edges: 0, 1, the largest legal value and its neighbours,
  `min_int`, `max_int`, the last address, a ring write that ends exactly at
  the ring's end. A size at a bound is drawn by construction at a fixed
  weight, since a uniform draw reaches it by luck. `cover` names outcomes the
  generator decides, never the scheduler's.
- A test waits on its helpers' signals (`Rig_support.await`), never a fixed
  sleep. Where a correct library gives none, it samples a short window and
  its name says "(sampled)".
- A failure path that needs a limit (files, address space, processes) sets
  it on a forked child alone; no test exhausts a shared resource. A test that
  forks lives in a suite of its own, since a process that spawned domains
  cannot fork.
- Fixing a bug: write the test, watch it fail, then fix. A bug not yet fixed
  lands as an `xfail` stating the wrong behaviour.

Layout mirrors `lib/`: suites in `test/<lib>/test_<module>.ml`, helpers
shared by two suites in `test/<lib>/support/`, fixtures in
`test/<lib>/fixtures/` with the `gen.sh` that makes them. A fixture that a
second suite opens by its own path lives once, in the `fixtures/` of the
suites' common parent; one read through a driver's support library stays.
Every top-level group sets `~timeout`; `dune` sets `WINDTRAP_TIMEOUT` for the
rest.

## Polled

rig's own suites run on `Rig_support.Polled`, a driver over host memory
whose queue runs only when the test runs it or a wait sleeps, and which logs
every driver call. Its options stand for the drivers rig must handle, such
as a full queue (`~capacity`), a stop with work in flight
(``~answer:`Unknown``) or memory the host cannot address (`~host_visible`).

`fault`, `fail`, `interrupt` and `stall` inject failures; `gate`,
`open_gate` and `sleepers` hold a wait at a known point. `test_walk.ml` fails
each fallible driver call of each operation in turn (`fail_at`), then checks
the outcome is one the `.mli` states, `outstanding` is empty, and the C heap
and descriptors are back where they were. Polled tests rig, never a driver.

## GPUs

Drivers are tested on their GPUs, with no mock drivers. A GPU test skips only
when the machine has no such GPU. A test that faults or hangs a GPU joins a
suite after one guarded run (one process, nothing else on the GPU, killed
after 60 s, the GPU idle afterwards), in a suite of its own if the fault
outlives the device in its process.

Within one build, the suites and benches that act on a GPU take turns
through dune's lock `%{workspace_root}/gpu`. A suite passes on a GPU that
other processes use. A timing needs the GPU to itself and a quiet machine
([bench/README.md](../bench/README.md)).

## Sanitizers

Each sanitizer is a workspace file at the repository's root, run the same
way:

```sh
dune build @packages/rig/runtest --workspace dune-workspace.asan
dune build @packages/rig/runtest --workspace dune-workspace.tsan
```

`dune-workspace.asan` builds with the sanitize profile of the root `dune`,
which holds its options: the C under AddressSanitizer and
UndefinedBehaviorSanitizer, and the debug runtime with a stressed collector.
`dune-workspace.tsan` builds the OCaml and the C under ThreadSanitizer, with
a compiler from its own lock, made once:

```sh
dune pkg lock --workspace dune-workspace.tsan dune-tsan.lock
```

On Linux that compiler needs libunwind's headers (`libunwind-dev`). A report
fails the test that made it. Both build in `_build/default`, so run them in
a tree with no watch server, whose build they would replace.

The gate runs on Linux, with Clang, whose AddressSanitizer takes its
allocator's locks across fork. OCaml compiles and links C with the `gcc` on
`PATH`, so the gate puts Clang there:

```sh
mkdir -p /tmp/rig-clang && ln -sf "$(command -v clang-19)" /tmp/rig-clang/gcc
PATH=/tmp/rig-clang:$PATH dune build @packages/rig/runtest --workspace dune-workspace.asan
```

Under GCC's AddressSanitizer, which holds no allocator lock across fork, a
test that forks a process running more than one thread skips. A GPU test
that skips checks nothing, so read a sanitizer run's skip count.

## A new driver's checklist

- Support matching `Rig_gpu_support.Conformance`, listed in
  `test/conformance/test_conformance.ml`, which then runs every driver's
  laws on it.
- A suite that opens the GPU through `Rig_gpu_support.Make` and states its
  facts and refusals at each bound its `.mli` states; work its rings cannot
  hold refused by `Rig.submit`; ring wraps at exact fits and streams longer
  than the rings; what a loss does to running and queued work; its fault
  path, after a guarded run.
- A walk of its real resources under limits on a forked child, as
  `test/nv/nvidia/test_walk.ml`, and a sanitize run on its host, with its
  vendor library's environment in its own `dune`.
