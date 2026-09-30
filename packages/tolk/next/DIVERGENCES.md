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
- **tolk.next:** `lib/device.ml` (the compiler half of `device.py`), waiting
  for L7 for the `ops_*.py` files; `lib/uop/ops.ml:227` (`param_arg`),
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
  (`test/uop/ops`): `resolve › simplify leaves a constant, and a sink of
  constants and stacks of constants, alone`, and the `resolve` tests that
  simplify with `Symbolic`'s rules;
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
- **tolk.next:** `lib/device.ml:122` (`renderers`), `:145` (`renderer`).
- **Differs:** the caller gives a target, a device name and its `arch`;
  tolk.next picks the renderer and compiler from the `arch` and never parses
  the name, with one exception: a name starting with `DISK` is a disk, as
  tinygrad reserves it and nx.device names its disk devices
  (`Ops.on_disk`, `is_disk_device`, `copy_to_device`, `clone`).
- **Reason:** (c).
- **Pinned by:** for targets, the `Device` suite: `renderer takes a
  device's name as it is (D6)` (a name with an index, in lower case, or a
  disk's, is no device, and a name gives no renderer or architecture) and
  `renderer › picks a device's renderer as tinygrad does` (the architecture
  comes from DEV or the caller); for the disk, the `Ops`
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

## D9. Narrow float conversions are IEEE conversions

- **tinygrad:**
  - `dtype.py:230-234` (`float_to_bf16` rounds to float32 with
    `truncate[dtypes.float]`, then to bfloat16), and `dtype.py:84` (`const`
    converts an integer to a float through a double, `float(val)`);
  - `codegen/decomp/dtype.py:36-38,101-134,198-199`: `l2i` converts a long
    to a float32 word by word; an emulated cast narrows its source to float32
    first; `f2f` flushes subnormals to zero both ways; `f2f_clamp` sends a
    finite value above the greatest one to infinity in the 16-bit floats, and
    saturates infinities in the 8-bit floats.
- **tolk.next:** `lib/dtype.ml:469-490` (`encode_format`, one rounding for
  every narrow float; `float_of_integer`); `lib/codegen/decomp/decomp_dtype.ml`
  (`long_to_float`, `f2f_clamp`, `narrow`, `f2f`).
- **Differs:** converting to a narrow float rounds once, to nearest with ties
  to even, from any source, and treats every value as IEEE does, folded or
  emulated:
  - a double rounds to bfloat16 once: `1 + 2^-8 + 2^-40` is `1.0078125` here
    and `1.0` in tinygrad. An integer converts once, from its exact value:
    `9042383626829825` is bfloat16 `0x5a01` here and `0x5a00` in tinygrad,
    whose double rounds it onto a tie. An emulated cast narrows a double, or
    an integer more precise than a float32, to float32 by rounding to odd, so
    that the store's rounding is the one rounding; an emulated 64-bit integer
    converts to a float32 once, where tinygrad's word arithmetic rounds each
    word and their sum;
  - emulation keeps subnormals, both ways;
  - a finite value becomes an infinity in a 16-bit float from the greatest
    finite value plus half an ulp, the tie rounding to even; an 8-bit float
    saturates to its greatest finite value;
  - an infinity stays one in e5m2 and becomes the NaN of e4m3 and the `fnuz`
    formats, of its sign where the format has one (D10).
- **Reason:** (b). rune folds constants with `Dtype`, and nx converts eagerly
  with one rounding. An emulated kernel exists only because its target lacks
  the type, so it must give the bits a native one gives.
- **Pinned by:** `Dtype › truncate › truncation.golden` (the near-tie rows,
  stated in code), `Dtype › truncate › bfloat16 rounds once, to the nearest,
  ties to even` and `Dtype › truncate › an integer rounds to a narrower float
  once, from its value`; for emulation, the `Decomp_dtype` suite
  (`test/codegen/decomp_dtype`), one test per facet; and a rune test at L9.

## D10. A float8 NaN keeps its sign when decoded

- **tinygrad:** `dtype.py:279` (`fp8_to_float` returns `math.nan` for
  e4m3's NaN codes, whatever their sign bit).
- **tolk.next:** `lib/dtype.ml:430-447` (`decode_format`).
- **Differs:** e4m3's `0xff` decodes to a negative NaN, so it encodes back to
  `0xff`, where tinygrad gives `0x7f`. e5m2 already kept the sign.
- **Reason:** (b). nx's decoder keeps the sign, so a value read eagerly and a
  folded constant agree, and every NaN code keeps its sign through a round
  trip.
- **Pinned by:** `Dtype › storage › a NaN decodes with the sign of its bits`;
  and a rune test at L9.

## D16. CUDA keeps a float8 infinity special

- **tinygrad:** `renderer/cstyle.py:33,42` (a cast to `__nv_fp8_e4m3` or
  `__nv_fp8_e5m2` is the constructor, which converts with
  `__NV_SATFINITE`), `:25-26` (an infinite constant is cast the same way).
- **tolk.next:** `lib/renderer/cstyle.ml:961-974` (`fp8_infinity`,
  `cuda_fp8_guard`, `is_fp8_guarded`), `:1018-1036` (the two rules of
  `cuda_lang`) and `:1144` (the helper in the prefix).
- **Differs:** the saturating conversion turns ±inf into ±max. tolk.next
  keeps an infinity special, as `Dtype.truncate` does: it stays an infinity
  in e5m2 and becomes a NaN of its sign in e4m3, which has no infinity. A
  cast of a float value calls `tg_fp8`, a helper the kernel declares only
  when it has such a cast, which converts with the constructor and then
  writes the bits of the special value if the input was infinite. An
  infinite constant is its bits, through `tg_bitcast`. Finite values,
  including those that overflow the format, still saturate. HIP needs no
  guard: on gfx950, `f32_to_fp8` clamps only finite values, and the
  non-saturating `cvt_pk_{fp8,bf8}_f32` it then calls keeps an infinity in
  bf8 and makes it NaN in fp8, which is unverified on hardware.
- **Reason:** (b). nx's float8 encoder keeps infinities special, and a
  jitted kernel must store what eager nx stores for the same cast.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`):
  `float8 infinities on CUDA (D16)`, which checks where the guard is declared,
  the byte each infinity writes and the bits of infinite e5m2 constants;
  `sources › by default › cuda_dtype_float8_e4m3`, `cuda_dtype_float8_e5m2`,
  `cuda_inf_nan_float8_e4m3` and `cuda_inf_nan_float8_e5m2`, which compare
  with tinygrad's source once the guard is written back as tinygrad writes
  it; and `every GPU kernel compiles with its target's toolchain › cuda_*`
  (slow, skipped without NVRTC). A rune test at L9 checks the stored values.

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
  `UOp.val` (`uop/ops.py:259-263`), unwrapped; `uop/weak.py:82-87`
  (`uncast_const`), which leaves the unwrapped literal bare. tinygrad's own
  `TestModularWraparound` expects the wrapped results and is marked
  `xfail_broken_const_wraparound`.
- **tolk.next:** `lib/uop/symbolic.ml:94` (`fold_const_alu`) and `:438`;
  `lib/uop/uop_weak.ml:203` (`uncast_const`).
- **Differs:** a committed constant, a cast of a literal to a type of known
  width, is read wrapped to that width: by an operation that folds, by a cast
  of it that collapses, and where its cast is dropped for a bare literal. The folded result is still kept mathematical,
  for emission to wrap. tinygrad reads the unwrapped value, so a fold that
  reads high bits gives what no machine computes: `(uint32 0xFFFFFFFF + 1) >> 1`
  folds to `2147483648` where the machine gives `0`, `threefry2x32(5, 10)`
  folds to another key than the unfolded graph computes, and the int64 cast
  of the int32 constant `2^31` folds to `2^31` where the machine gives
  `-2^31`, and `x < uint8 300` compares against a bare `300`, which folds to
  `true`, where the machine compares against `44`.
- **Reason:** (b): rune's `Nx.Rng` (Threefry), jitted with constant keys,
  must draw the numbers eager nx draws.
- **Pinned by:** the `Symbolic` suite: `symbolic_simple › constants › an
  operation reads committed constants at their width` (a fold, a cast of a
  committed constant, a uint8 remainder), `symbolic_simple › constants › a
  comparison reads a committed constant at its width` (the uncast), and
  `tinygrad › tests.golden › TestModularWraparound.<test>` and
  `TestThreefryConstFolding.test_threefry`, which check the machine value.

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

## D15. Compilers load their library at the first compile

- **tinygrad:** `runtime/support/compiler_cuda.py:62` (`NVRTCCompiler`
  calls `nvrtcVersion`), `runtime/support/compiler_amd.py:80` (`HIPCompiler`
  asserts comgr is loaded) and `runtime/ops_metal.py:39` (`MetalCompiler`
  creates its code generation service), each when the compiler is made, so
  making the renderer that holds it fails without the library.
- **tolk.next:** `lib/runtime/support/compiler_cuda.ml` (`nvrtc`),
  `lib/runtime/support/compiler_amd.ml` (`hip`), `lib/runtime/ops_metal.ml`
  (`compiler`).
- **Differs:** making a compiler loads nothing. The library is loaded at the
  first compile, once per process, and a compile without it raises
  `Compile_error` with the reason.
- **Reason:** (b): `Cstyle`'s CUDA, HIP and Metal renderers are made, and
  render, on machines without NVRTC, comgr or MTLCompiler: the source goldens
  of the `Cstyle` suite and rune's rendering of a kernel for inspection.
- **Pinned by:** `Tolk_next.Compiler_cuda › a library that does not load`,
  `Tolk_next.Compiler_amd › a library that does not load` and
  `Tolk_next.Ops_metal › a library that does not load`, run where the library's
  variable names a file that is no library: making the compiler succeeds, each
  compile raises `Compile_error`, and a cached binary is served without a load.
  Where the library is absent, `› without NVRTC on the machine`,
  `› without comgr on the machine` and `› without MTLCompiler on the machine`
  pin the error that names the library and its variable.

## D17. Each operation on a narrow scalar is narrowed in the source

- **tinygrad:** `renderer/cstyle.py:245-250` (an ALU used once is inlined into
  its user, whatever its type) with `:66-67` (the operator text).
- **tolk.next:** `lib/renderer/cstyle.ml:86,565,771` (`promoted`), `:622-632`
  (`narrowed`), `:714-718` (the cast) and `:348` (an operand's parentheses).
- **Differs:** C, C++ and Metal compute an operation on a `char`,
  `unsigned char`, `short` or `unsigned short` in `int`, and Clang computes
  one on an `__fp16`, a storage format, in `float`. Only an assignment
  narrows the result back, so an inlined operation neither wraps nor rounds
  until the kernel stores: `(uchar)16 + 86` times `3`, cast to float, gives
  `306` where each operation wrapping gives `50`. The source is tinygrad's
  for the kernel with a cast to its own type after each inlined scalar
  operation of one of these types that its user does not store. Vectors are not promoted, and CUDA's `__half` and
  `nv_bfloat16`, HIP's `_Float16` and Metal's `half` and `bfloat` compute
  natively, so they keep tinygrad's source; so does every kernel without such
  an inlined operation. Clang's, HIP's and Metal's rules were checked with
  their compilers; CUDA's rests on the operators `cuda_fp16.hpp` and
  `cuda_bf16.hpp` declare, and is on the hardware checks of
  `test/README.md`.
- **Reason:** (b). Eager nx and compiled code agree: nx computes each
  operation in its own type, as the folder (`Ops.exec_alu`) and tinygrad's
  interpreter do.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `narrowing
  (D17)`, which checks for every kernel that the kernel whose tinygrad source
  is its source is the kernel with those casts, and `sources`, which compares
  the 21 sources it changes with tinygrad's for that kernel; `execution on
  the host › wraps each operation on unsigned chars (D17)` and `rounds each
  operation on halves to a half (D17)`; and the slow `a kernel over a narrow
  type wraps and rounds as the interpreter (D17)`, over each type's whole
  range.

## D18. Metal computes a bfloat trunc in float

- **tinygrad:** `renderer/cstyle.py:368-372` (`MetalRenderer.extra_matcher`
  computes `SQRT`, `EXP2`, `LOG2` and `SIN` of a bfloat16 in float32).
- **tolk.next:** `lib/renderer/cstyle.ml:883` (`metal_extra_matcher`).
- **Differs:** `TRUNC` of a bfloat16 is computed in float32 as well. Metal
  has no `trunc` of a `bfloat`: `trunc(x)` converts `x` to `float` and
  returns a `float`, which does not convert to a `bfloat` implicitly, so a
  kernel that stores or adds it to a bfloat does not compile. The graph
  handed to the renderer has a cast to float32 and back around each such
  `TRUNC`.
- **Reason:** (b). rune runs bfloat16 models on Metal (gpt-oss), where
  `Nx.trunc`, `floor`, `ceil` and `round` of a bfloat16 lower to `TRUNC`.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `bfloat16
  truncation on Metal (D18) › truncates a bfloat16 in float32` and `leaves it
  to CUDA, which truncates a bfloat16 with htrunc`, and the slow `compiles a
  kernel that truncates a bfloat16`; `every GPU kernel compiles with its
  target's toolchain › metal_transcendental_bf16`, tinygrad's graph, is an
  expected failure.

## D19. A hierarchical allreduce to one device lands there

- **tinygrad:** `schedule/allreduce.py:28-34`, the hierarchical branch of
  `handle_allreduce`, which returns before it reads the target device
  (`:11`), so its result is on every device of the source even when the
  allreduce targets one. `create_allreduce_function` (`:68-75`) then stores
  that value into storage on the one target device. Repro: with
  `ALLREDUCE_NODE_NDEVS=2`,
  `UOp.new_buffer(("CPU:0","CPU:1","CPU:2","CPU:3"), 16, dtypes.float).allreduce(Ops.ADD, "CPU:0")`
  handled by `handle_allreduce` gives an `MSTACK` on the four devices.
- **tolk.next:** `lib/schedule/allreduce.ml:104` (`handle_allreduce`).
- **Differs:** when the target is one device, the hierarchical branch copies
  each reduced chunk there from the device of its rank in the first node, as
  the ring and all-to-all branches copy their reduced chunks, and the result
  is on that device.
- **Reason:** (b): rune's multi-device reductions. A sharded sum reduced to
  one device under `ALLREDUCE_NODE_NDEVS`, the multi-node setting, would
  otherwise compute a value on every device and store it into storage on
  one.
- **Pinned by:** the `Allreduce` suite: `handle_allreduce › recorded ›
  nodes_to_one_device_handled.golden, landed on its device (D19)`, which
  states tinygrad's golden in code with each gather of a chunk replaced by
  its copy to the target, and `rules › a hierarchical allreduce to one device
  lands there (D19)`, which evaluates the value and the function on two
  devices and finds the reduction on the target alone.

## D20. Storage keeps an e5m2 NaN's payload

- **tinygrad:** `dtype.py:251,278` (`float_to_fp8` stores every e5m2 NaN as
  `0x7f` of its sign, and `fp8_to_float` decodes every NaN code as `math.nan`
  of its sign).
- **tolk.next:** `lib/dtype.ml:411-447` (`nan_of_payload`, `nan_payload` and
  `decode_format`) and `:469-490` (`encode_format`).
- **Differs:** storage moves an e5m2 NaN's two payload bits through the double,
  as it moves the payload of the 16-bit formats, so `bitcast` keeps every e5m2
  code: `0x7d` and `0x7e` come back as themselves, where tinygrad gives `0x7f`,
  and the canonical NaN stores as `0x7e`, its quiet code. A conversion still
  gives `0x7f` of its sign, as nx's encoder does.
- **Reason:** (b). A kernel's bitcast and nx's are byte reinterpretations, so
  a bitcast that rune folds must keep the bits as they do.
- **Pinned by:** the `Dtype` suite (`test/dtype`): the bitcast round trip on
  every e5m2 code, and the `reencode.golden` and `truncation.golden` NaN rows,
  stated in code.

## D22. An emulated long converts no float past its words' range

- **tinygrad:** `codegen/decomp/dtype.py:33-34` (`l2i` makes a long's low
  word by casting the float to an int32, and its high word by casting the
  float over 2^32).
- **tolk.next:** `lib/codegen/decomp/decomp_dtype.ml` (`l2i`, the cast of a
  float to a long).
- **Differs:** the words are the quotient and remainder of the truncated
  float's magnitude by 2^32, each converted from a float the word holds, and
  negated as a long for a negative float. tinygrad converts a float past
  2^31 to an int32 word, which C leaves undefined: ARM saturates and x86 gives
  `0x80000000`, so the low word of `2^32 + 5` is `0x7fffffff` on ARM.
- **Reason:** (b): rune's kernels on a target without 64-bit integers cast
  floats to longs through this emulation, and must give the long nx gives.
- **Pinned by:** the `Decomp_dtype` suite: `emulated 64-bit integers › an
  emulated cast of a float32 to a 64-bit integer converts no float to a word
  that cannot hold it`.
