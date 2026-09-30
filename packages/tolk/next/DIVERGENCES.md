# Divergences from tinygrad

tolk.next follows tinygrad `79af1ca70e7021f504919c4ff5631245acc33ed6`. This
ledger lists every place where it does something else. An entry is admitted
for one of three reasons only:

- **(a) an OCaml constraint:** acyclic modules, static types, the GC, domains;
- **(b) a named consumer:** a raven call site that fails without it;
- **(c) the executor's contract:** nx.device's submission protocol, which
  rune drives with tolk.next's output.

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
- **Differs:** timeline values are parameters of the host program, and only
  the runtime writes the submitted word.
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
- **Differs:** tolk.next holds the compiler half: `Compiler`, the renderer
  and compiler selection of `Compiled`, and the IR half of each `ops_*.py`
  (queues, `pm_encode`, program data), all returning data. The registry, the
  lazy `Buffer` and running a schedule are rune's; allocators, programs,
  drivers and profile events are nx.device's.
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
- **Differs:** the caller gives a target, a device name and its `arch`;
  tolk.next picks the renderer and compiler from the `arch` and never parses
  the name.
- **Reason:** (c).
- **Pinned by:** waiting for L6.

## D7. Stamp slots follow `Submission.record`

- **tinygrad:** `runtime/support/hcq2.py:231,240`.
- **tolk.next:** waiting for L7.
- **Differs:** profiling stamp slots follow the amended `Submission.record`.
- **Reason:** (c).
- **Pinned by:** waiting for L7.

## D8. The disk cache maps strings to strings

- **tinygrad:** `helpers.py:398-447` (an SQLite database of pickled values,
  keyed by strings, integers or dictionaries of columns).
- **tolk.next:** `lib/helpers.ml:586-697` (`Diskcache`).
- **Differs:** keys and values are strings, which callers encode. Each entry
  is a file under `CACHEDB`, a directory, written aside and renamed into
  place; tables are versioned by tolk.next's own version.
- **Reason:** (a). OCaml has no pickle, no type-safe serialization of
  arbitrary values: a cache returning any type would be `Marshal`, which
  crashes when a rebuilt program reads a value of a changed type. Without an
  SQLite transaction, writing aside and renaming is what keeps an entry whole
  when a writer crashes or races another.
- **Pinned by:** `test/helpers`: `Helpers › Diskcache › get reads back any
  key and value put` and `Helpers › Diskcache › behaves as a table of entries
  per table`.

## D9. bfloat16 rounds once from the double

- **tinygrad:** `dtype.py:230-234` (`float_to_bf16` rounds to float32 with
  `truncate[dtypes.float]`, then to bfloat16).
- **tolk.next:** `lib/dtype.ml:345-364` (`encode_format`, one rounding for
  every narrow float).
- **Differs:** a double rounds to bfloat16 once, to nearest even. The two
  differ when the float32 lands on a bfloat16 tie: `1 + 2^-8 + 2^-40` is
  `1.0078125` here and `1.0` in tinygrad.
- **Reason:** (b). rune folds bfloat16 constants from OCaml floats, and a
  folded constant must be the value nx's eager cast gives, which rounds once.
  The codegen layer's bfloat16 cast must compute the same, or get a row of its
  own.
- **Pinned by:** `Dtype › truncate › truncation.golden` (the near-tie rows,
  stated in code) and `Dtype › truncate › bfloat16 rounds once, to the
  nearest, ties to even`; and a rune test at L9.

## D10. A float8 NaN keeps its sign when decoded

- **tinygrad:** `dtype.py:279` (`fp8_to_float` returns `math.nan` for
  e4m3's NaN codes, whatever their sign bit).
- **tolk.next:** `lib/dtype.ml:308-324` (`decode_format`).
- **Differs:** e4m3's `0xff` decodes to a negative NaN, so it encodes back to
  `0xff`, where tinygrad gives `0x7f`. e5m2 already kept the sign.
- **Reason:** (b). nx's decoder keeps the sign, so a value read eagerly and a
  folded constant agree, and every NaN code keeps its sign through a round
  trip.
- **Pinned by:** `Dtype › storage › a NaN decodes with the sign of its bits`;
  and a rune test at L9.

## D11. Kernel optimisations are typed

- **tinygrad:** `codegen/opt/__init__.py:9-15` (`Opt(op, axis, arg)`, with an
  `arg` of any type), checked by `codegen/opt/postrange.py:109-164`.
- **tolk.next:** `lib/codegen/opt/opt.ml` (`t`, `target`).
- **Differs:** each kind of optimisation is a constructor with its own
  fields, which stand for `OptOps` too, and a split's target is one of the
  three axis types a split can make (`split_targets`, `postrange.py:14`). The
  malformed arguments that `postrange.py` refuses at run time cannot be
  built.
- **Reason:** (a): static types.
- **Pinned by:** the type itself, whose printing the `Ops` suite checks
  against tinygrad's `Opt` repr (`reprs.golden`, the row "kernel with opts");
  `Postrange`'s suite (L4) drops tinygrad's malformed-argument cases with
  this entry as the reason.
