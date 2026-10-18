# rig

rig runs work on GPUs, NICs, files and remote machines. A program allocates
memory and submits work; rig orders the work by the memory it reads and
writes, so a program never names an event or a fence.

## Structure

- Drivers share no code: `rig.amd`, `rig.nv`, `rig.cuda`, `rig.metal`,
  `rig.mlx5`, `rig.bnxt`, `rig.pci`, `rig.disk`, `rig.remote`. When two need
  the same shape, each implements it. Never propose a module or helper
  shared between drivers.
- Drivers depend on `rig.edge`, never on `rig`.
- rig exists to run work on the hardware a user has, from OCaml, with
  nothing else installed. Each capability reaches hardware or a setting
  that no other path reaches: GPUs with no vendor driver installed, RDMA
  NICs across machines, remote machines, CUDA graphs and Metal ICBs where
  launch cost bounds a step, counters and traces to see where time goes,
  VF/XGMI, fork, the memory device. Removing one removes the users who
  need it, and reach is rig's purpose, so it is never traded for lines.
  Callers on main say nothing about users outside the repository, and a
  path not yet run on hardware needs a run.
- To make rig smaller, build a capability more simply. tinygrad's runtime
  has most of these in fewer lines: a datapoint for how small each can be.
- A promise about failure or misuse is no capability. Judge it as any
  design: whether raven should make it at all.

## API

- An error handled at the call (open, load) is a value. One that can arise
  anywhere (`Lost`, `Out_of_memory`) is an exception. A caller's mistake
  raises `Invalid_argument` at the call that made it.
- No correctness depends on caller discipline (keeping values alive,
  ordering, locks), except a value labelled expert or unsafe with a warning.
- Submit, launch, copy and claim allocate nothing where the `.mli` promises
  it.
- No `CHANGES.md` entries before the first release.

## C and generated code

- C builds with GCC `-Werror` on Linux as well as clang on macOS.
- Generated files come from their `gen.sh`, never by hand.
- No GPL-only header excerpt is committed. Write the values by hand, with
  one comment giving the reason, and a Linux test that checks them against
  the system header.

## Tests

`test/README.md` is the contract: Polled, GPU suites, sanitizers and the new
driver checklist. Traps:

- windtrap's `stateful` runs each program 50 times and may call its start
  twice. Each start opens its own device; keep `~steps` small.
- Polled runs its queue only when a test runs it or a wait sleeps. A wait on
  lazy Polled work through another device spins 200 ms first, and CUDA can
  deadlock on it in C. Use `Polled.make ~runs:`Itself` there.
- A slow or hung model run is first checked for those two, then for another
  process holding the GPU lock.
