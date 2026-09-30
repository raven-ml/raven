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

## D2. Withdrawn

tinygrad matches patterns without generating code when `UPAT_COMPILE` is 0
(`uop/ops.py:1545`, `upat_interpret`), and `UPat`, `PatternMatcher` and
`graph_rewrite` live in `uop/ops.py`. tolk.next ports that path in
`Ops`, so leaving out `uop/upat.py`, the pattern compiler, is scope: see
the Exclusions of `README.md`.

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk.next:** waiting for L6 and L7; `lib/uop/ops.ml:227` (`param_arg`),
  `:3005` (`new_buffer`).
- **Differs:** tolk.next holds the compiler half: `Compiler`, the renderer
  and compiler selection of `Compiled`, and the IR half of each `ops_*.py`
  (queues, `pm_encode`, program data), all returning data. The registry, the
  lazy `Buffer` and running a schedule are rune's; allocators, programs,
  drivers and profile events are nx.device's. So a node holds no runtime
  state: `ParamArg` has no `buffer`, a `BUFFER` is named by its slot for rune
  to bind, and `UOp.buffer`, `realized`, `is_realized`, `_buffer_view`,
  `_base_buffer_is_realized`, `from_buffer` and `_frompy`
  (`uop/ops.py:856-993`) are rune's. Whether storage is bound is the
  operation itself, `BUFFER` or `ALLOC`, so the specification's checks of
  `ParamArg.buffer` (`uop/spec.py:149,153,266,268`) hold by construction.
- **Reason:** (c).
- **Pinned by:** waiting for L6 and L7; for `Ops`, its suite
  (`test/uop/ops`): `storage › new_buffer takes the next slot without one`
  and `reprs.golden`, where a `BUFFER` prints and interns by its slot alone.

## D4. Import cycles are broken

- **tinygrad:** the imports inside functions that Python uses to hide a cycle:
  - `uop/ops.py:15,204,270,517,935,1167-1190,1572` calls `symbolic`, `render`,
    `spec`, `upat`, `schedule.prepare`, `mixin.rand` and `renderer`;
  - `uop/spec.py:7,287-289` against `device`, `codegen.opt`,
    `schedule.rangeify` and `renderer`;
  - `renderer/__init__.py` imports `device.Compiler`, while `device.Compiled`
    holds renderers;
  - `codegen/opt/postrange.py:267,273` against `search` and `heuristic`, with
    `search` importing `engine.realize`;
  - `schedule/__init__.py` against `engine.realize`, `engine/realize.py:262`
    against `hcq2`, and `tensor.py` against `engine.jit` and `engine.realize`;
  - `renderer/cstyle.py` imports the compilers and `ops_metal`.
- **tolk.next:** waiting for L2 through L8, each break with its layer; for
  `uop/ops.py`, `lib/uop/ops.ml:572` (`construction_check`), `:1651`
  (`simplify_hook`), `:4317` (`Private`), `:1188` (`Make_elementwise`),
  `:815` (`repr`), `:241` (`bufferize_opts`), `:259` (`Calls`);
  `lib/uop/render.ml:202` (`render`), `:212` (`srender`); and
  `lib/renderer/renderer.ml` (`Compiler`).
- **Differs:**
  - the `UOp` methods that call a later module become functions of that
    module: `contiguous_view` and its matcher go to `Schedule.Prepare`,
    `to_elf` to `Device`, and `render` and `srender` to `Render`;
  - `simplify` stays in `Ops`, since reshaping, `resolve` and shapes call
    it: it rewrites with the `symbolic` matcher that `Symbolic` installs when
    the library is initialised, and the `SPEC` check at construction runs the
    matcher that `Spec` installs. Each is set once; `lib/dune` links the
    library whole (`-linkall`), so both are set before any program runs, and
    `simplify` raises if its rules are missing. A sink of constants and stacks
    of constants is itself without the rules, which leave it as it is, so
    shapes are built before they are installed;
  - `CallInfo.aux`, `hcq2.py`'s `HCQInfo`, is the record `hcq_info` of
    `Ops`, since call arguments hold it;
  - `render.py`'s `pretty_print` is `Ops.pp`: it prints arguments
    (`argstr`), and arguments print the nodes they hold, so the two recurse;
  - the kept methods of `mixin/*.py` are functions of `Ops`, since
    `ops.py` calls them and a module cannot call a later one; the elementwise
    ones, which patterns share with nodes, are one functor applied to both;
  - the small types that `ops` and `spec` name from later files
    (`Estimates`, `BufferizeOpts`), and `device.py`'s `is_disk_device`, are
    defined in the earliest module that needs them; `Opt`, whose file
    depends on nothing, stays in its own module;
  - `spec.py`'s imports of `codegen.opt`, `schedule.rangeify` and
    `renderer` serve only `pyrender_globals`, which is not ported (see the
    README), so they need no break;
  - `device.py`'s `Compiler` and `CompileError` are `Renderer.Compiler`,
    since a renderer holds its compiler and `Device` follows `Renderer`;
  - `apply_opts` takes the optimiser as an argument, and `Search` lands with
    the engine;
  - the engine has one order (schedule, hcq2 helpers, realize, tensor, jit),
    and each late binding is passed as a function argument, never a global
    reference;
  - the compiler modules precede `Cstyle`.
- **Reason:** (a). Each layer's review checks that its breaks are the
  smallest possible.
- **Pinned by:** waiting for L4 through L8; for `Ops`, its suite
  (`test/uop/ops`): `resolve › simplify rejects a graph other than constants
  while the symbolic rules are not installed` (before L3), and at L3 the law
  that `simplify` returns a sink of constants and stacks of constants itself;
  `printing › pretty.golden`; `elementwise patterns › the pattern operators
  are the named pattern operations`; `queue calls › pp_hcq_info formats every
  field, as the record's repr`.

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
  the name, with one exception: a name starting with `DISK` is a disk, as
  tinygrad reserves it and nx.device names its disk devices
  (`Ops.on_disk`, `is_disk_device`, `copy_to_device`, `clone`).
- **Reason:** (c).
- **Pinned by:** waiting for L6 for targets; for the disk, the `Ops`
  suite: `several devices › on_disk holds for one disk device`, `several
  devices › copy_to_device rejects a disk and a weak type` and `storage ›
  clone rejects a disk`.

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

## D12. A node's key is a BLAKE2 digest

- **tinygrad:** `uop/ops.py:266-268` (`key`, the SHA-256 of
  `str((op, dtype, arg))` followed by the keys of the sources).
- **tolk.next:** `lib/uop/ops.ml:1063` (`key`).
- **Differs:** the same text is digested with BLAKE2b-256. No key is ever
  compared with a key tinygrad computed: keys name compiled programs in caches
  that tolk.next alone writes.
- **Reason:** (a): OCaml's standard library has MD5 and BLAKE2, not SHA-256.
- **Pinned by:** the `Ops` suite: `key › ignores tags` and `key › tells
  arguments apart`.

## D13. Folding reads committed constants at their width

- **tinygrad:** `uop/symbolic.py:29` (`fold_const_alu`) and `:154-155` (the
  collapse of committed const conversions), which read a constant with
  `UOp.val` (`uop/ops.py:259-263`), unwrapped. tinygrad's own
  `TestModularWraparound` expects the wrapped results and is marked
  `xfail_broken_const_wraparound`.
- **tolk.next:** `lib/uop/symbolic.ml:94` (`fold_const_alu`) and
  `:438`.
- **Differs:** a committed constant, a cast of a literal to a type of known
  width, is read wrapped to that width: by an operation that folds, and by a
  cast of it that collapses. The folded result is still kept mathematical,
  for emission to wrap. tinygrad reads the unwrapped value, so a fold that
  reads high bits gives what no machine computes: `(uint32 0xFFFFFFFF + 1) >> 1`
  folds to `2147483648` where the machine gives `0`, `threefry2x32(5, 10)`
  folds to another key than the unfolded graph computes, and the int64 cast
  of the int32 constant `2^31` folds to `2^31` where the machine gives
  `-2^31`.
- **Reason:** (b): rune's `Nx.Rng` (Threefry), jitted with constant keys,
  must draw the numbers eager nx draws.
- **Pinned by:** the `Symbolic` suite: `symbolic_simple › constants › an
  operation reads committed constants at their width` (a fold, a cast of a
  committed constant, a uint8 remainder), and `tinygrad › tests.golden ›
  TestModularWraparound.<test>` and `TestThreefryConstFolding.test_threefry`,
  which check the machine value.

## D14. Folded comparisons treat NaN as IEEE does

- **tinygrad:** `uop/symbolic.py:29` (`fold_const_alu`), whose operands come
  from `UOp.val` (`uop/ops.py:259-263`) as `ConstFloat`s (`dtype.py:8-22`).
  `ConstFloat` makes NaN equal to NaN on purpose, so that NaN constants
  intern as one node, and `exec_alu` compares with it: `nan != nan` and
  `nan < nan` fold to `False`, where `exec_alu` on floats gives `True` and
  `False`.
- **tolk.next:** `lib/uop/symbolic.ml:94` (`fold_const_alu`); constants are
  interned by `Dtype.equal_const`, and `exec_alu` compares floats.
- **Differs:** a folded comparison of NaN constants follows IEEE: `nan <> nan`
  is `true`.
- **Reason:** (b): `Nx.not_equal x x` on a NaN is `true` in eager nx, and
  rune's jitted graph, whose constants fold here, must agree.
- **Pinned by:** the `Symbolic` suite: `symbolic_simple › constants › NaN is
  unequal to itself when constants fold, as IEEE says`.
