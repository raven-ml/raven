# Divergences from tinygrad

tolk.next follows tinygrad `79af1ca70e7021f504919c4ff5631245acc33ed6`. This
ledger lists every place where it does something else. An entry is admitted
for one of three reasons only:

- **(a) an OCaml constraint:** acyclic modules, static types, the GC, domains;
- **(b) a named consumer:** a raven call site that fails without it;
- **(c) an nx.device contract.**

Taste, speed without a measurement, and "the old tolk did it" are not reasons.
A difference in numerics also needs a failing rune test, since nx semantics
belong in rune's lowering.

Each entry gives the tinygrad file and line, the tolk.next file and line, what
differs, its reason, and the test that pins it. An entry whose test does not
exist yet names the layer that brings it; the entry is rejected at that
layer's review if the test is still missing. An entry goes when its reason
goes. Keeping only part of a file is scope, recorded under Exclusions in
`README.md`, not a divergence.

## D1. Timeline values are parameters

- **tinygrad:** `runtime/support/hcq2.py:276,432`.
- **tolk.next:** waiting for L7.
- **Differs:** timeline values are parameters, and only the runtime writes
  the submitted word.
- **Reason:** (c).
- **Pinned by:** waiting for L7.

## D2. The pattern matcher matches directly

- **tinygrad:** `uop/upat.py:177-185` (`upat_compile` generates Python source
  and runs it with `exec`).
- **tolk.next:** waiting for L1.
- **Differs:** a pattern is matched by walking it, without generating code.
- **Reason:** (a).
- **Pinned by:** waiting for L1.

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk.next:** waiting for L6 and L7.
- **Differs:** the registry, the lazy `Buffer`, `Compiler` and `Compiled`, and
  the IR half of each `ops_*.py` (queues, `pm_encode`, program data) are in
  tolk.next; allocators, programs, drivers and profile events are in
  nx.device.
- **Reason:** (c).
- **Pinned by:** waiting for L6 and L7.

## D4. Import cycles are broken

- **tinygrad:** the imports inside functions that Python uses to hide a cycle:
  - `uop/ops.py:15,204,270,517,935,1167-1190,1572` calls `symbolic`, `render`,
    `spec`, `upat`, `schedule.prepare`, `mixin.rand` and `renderer`;
  - `uop/spec.py:287-289` against `codegen.opt`, `schedule.rangeify` and
    `renderer`;
  - `renderer/__init__.py` imports `device.Compiler`, while `device.Compiled`
    holds renderers;
  - `codegen/opt/postrange.py:267,273` against `search` and `heuristic`, with
    `search` importing `engine.realize`;
  - `schedule/__init__.py` against `engine.realize`, `engine/realize.py:262`
    against `hcq2`, and `tensor.py` against `engine.jit` and `engine.realize`;
  - `mixin/creation.py` imports `canonicalize_device`;
  - `renderer/cstyle.py` imports the compilers and `ops_metal`.
- **tolk.next:** waiting for L1 through L8, each break with its layer.
- **Differs:**
  - the `UOp` methods that call a later module become functions of that
    module (`Symbolic.simplify u`);
  - the small types that `ops` and `spec` name (`Estimates`, `BufferizeOpts`,
    `Opt`, `OptOps`) are defined in the earliest module that needs them;
  - `Compiler` is defined ahead of the renderers, and `Device` follows
    `Renderer`;
  - `apply_opts` takes the optimiser as an argument, and `Search` lands with
    the engine;
  - the engine has one order (schedule, hcq2 helpers, realize, tensor, jit),
    and each late binding is passed as a function argument, never a global
    reference;
  - `canonicalize_device` moves to `Helpers`;
  - the compiler modules precede `Cstyle`.
- **Reason:** (a). Each layer's review checks that its breaks are the
  smallest possible.
- **Pinned by:** waiting for L1 through L8.

## D5. Compilation workers are domains

- **tinygrad:** `engine/worker.py:1-2` (`multiprocessing` spawn workers).
- **tolk.next:** waiting for L7.
- **Differs:** compilation runs on domains, not processes.
- **Reason:** (a).
- **Pinned by:** waiting for L7.

## D6. Devices are named, never parsed

- **tinygrad:** `device.py:26,36,395,491` (device strings split at `:`).
- **tolk.next:** waiting for L6.
- **Differs:** a device is named by `Nx_device.name` and looked up; its name
  is never parsed.
- **Reason:** (c).
- **Pinned by:** waiting for L6.

## D7. Stamp slots follow `Submission.record`

- **tinygrad:** `runtime/support/hcq2.py:231,240`.
- **tolk.next:** waiting for L7.
- **Differs:** profiling stamp slots follow the amended `Submission.record`.
- **Reason:** (c).
- **Pinned by:** waiting for L7.
