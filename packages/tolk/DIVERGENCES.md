# Divergences from tinygrad

tolk follows tinygrad `f1d8f0f0def436ee45b3907fc12e30bb1a17d387`. This
ledger lists every place where it does something else. An entry is admitted
for one of five reasons only:

- **(a) an OCaml constraint:** acyclic modules, static types, the GC, domains;
- **(b) a named consumer:** a raven call site that fails without it;
- **(c) the executor's contract:** rig's devices, submissions and programs,
  which `tolk.engine` drives with the compiler's output;
- **(d) measured speed:** of a named raven workload, pinned by a bench row on
  fudge whose baseline would catch the regression;
- **(e) correctness:** a divergence that keeps correctness where tinygrad
  computes a wrong value or does not terminate, pinned by a test, stays
  without a consumer (`doc/dev/design.md`: correctness is never traded).

Taste, speed without such a measurement, "the old tolk did it" and the
maintainer's call alone are not reasons. A speed measured only on another
machine is marked "to re-measure on fudge" until a fudge row holds it. An
entry whose reason has lapsed says "Reason: none" until it is withdrawn or
redesigned. A difference in numerics also needs a failing rune test, since
nx semantics belong in rune's lowering, unless it keeps correctness (e).

NaNs follow nx's rule. A NaN result is some NaN, its sign and payload
unspecified, and it is signalling only where an operand is a signalling NaN.
A move (a load, a store, a copy, a selection, a bitcast) and a negation keep a
NaN's bits. A NaN's bits are a reason for an entry only where a move or a
negation fixes them.

Each entry gives the tinygrad file and line at the pinned commit, the tolk
module and symbol (`Rangeify.remove_bufferize`), what differs, its reason,
and the test that pins it. tolk, and any other raven code, is cited by module
and symbol, never by line, since lines move with every change. An entry
whose test does not exist yet names the layer that brings it; the entry is
rejected at that layer's review if the test is still missing. An entry goes
when its reason goes. Entries are in the order of their numbers, and a
number is never reused, since entries and documents cite them: an entry
merged into another is withdrawn and names the entry that holds its rule,
where each merged part keeps its own reason and pins.
Keeping only part of a file is scope, recorded under Exclusions in
`README.md`, not a divergence.

## D1. Withdrawn

A device's timeline was its signal word alone, whose values a batch's host
program took as parameters. rig numbers each device's work and orders it by the
memory it uses, and tolk encodes no queue (D146).

## D2. Withdrawn

tinygrad matches patterns without generating code when `UPAT_COMPILE` is 0
(`uop/ops.py:1524`, `upat_interpret`), and `UPat`, `PatternMatcher` and
`graph_rewrite` live in `uop/ops.py`. tolk ports that path in
`Ops`, so leaving out `uop/upat.py`, the pattern compiler, is scope: see
the Exclusions of `README.md`.

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk:** `Device` (the compiler half of `device.py`);
  `Tolk_engine.target` and `Tolk_engine.link`; `Ops.param_arg` (the type and
  the function) and `Ops.new_buffer`.
- **Differs:** tolk holds the compiler half: `Compiler` and the renderer
  and compiler selection of `Compiled`, returning data. The lazy `Buffer`
  and running a schedule are `tolk.engine`'s, which has no registry of
  devices by name: its caller maps each name to a device. The IR half of
  each `ops_*.py` (queues, `pm_encode`, program data), allocators, programs,
  drivers and profile events are rig's (D146). So a node of the compiler
  holds no runtime state:
  `ParamArg` has no `buffer`, a `BUFFER` is named by its slot for the engine
  to bind, and `UOp.buffer`, `realized`, `is_realized`, `_buffer_view`,
  `_base_buffer_is_realized`, `from_buffer` and `_frompy`
  (`uop/ops.py:813-953`) are the engine's. Whether storage is bound is the
  operation itself, `BUFFER` or `ALLOC`, so the specification's checks of
  `ParamArg.buffer` (`uop/spec.py:151,155,267,269`) hold by construction.
- **Reason:** (c).
- **Pinned by:** for `Ops`: `Tolk.Ops › storage › new_buffer takes the
  next slot without one` and `Tolk.Ops › arguments › reprs.golden`, where
  a `BUFFER` prints and interns by its slot alone; for the engine: the
  Engine suite (`test/engine/tolk_engine`): `refusals › link refuses a
  device the map does not hold`, which a registry would resolve, and
  `refusals › run refuses a parameter it binds no buffers`, which a `BUFFER`
  holding its buffer would run.

## D4. Import cycles are broken

- **tinygrad:** the imports inside functions that Python uses to hide a cycle:
  - `uop/ops.py:15,197,257,491,896,1123-1146,1540` calls `symbolic`, `render`,
    `spec`, `upat`, `schedule.prepare`, `mixin.rand` and `renderer`;
  - `uop/spec.py:6` against `device`;
  - `renderer/__init__.py` imports `device.Compiler`, while `device.Compiled`
    holds renderers;
  - `codegen/opt/postrange.py:246,252` against `search` and `heuristic`, with
    `search` importing `engine.realize`;
  - `schedule/__init__.py` against `engine.realize`, and `tensor.py` against
    `engine.jit` and `engine.realize`;
  - `renderer/cstyle.py` imports the compilers and `ops_metal`.
- **tolk:** for `uop/ops.py`, `Shape.simplify_rules` and the line after
  them, which sets them; `Ops.Make_elementwise`, `Ops.repr`,
  `Ops.bufferize_opts`; `Render.render`, `Render.srender`;
  `Renderer.Compiler`; `Prepare.contiguous_view`;
  `Schedule.pm_flatten_linear`; `Codegen.apply_opts`;
  `Realize.lower_and_compile`; and `Compiler_metal`.
- **Differs:**
  - the `UOp` methods that call a later module become functions of that
    module: `contiguous_view` goes to `Prepare`,
    `to_elf` to `Device`, and `render` and `srender` to `Render`;
  - `engine/realize.py`'s `pm_flatten_linear`, which `schedule/__init__.py`
    imports, is `Schedule.pm_flatten_linear`, since `Realize` follows
    `Schedule`;
  - `simplify` is `Shape`'s, with the functions that call it (reshaping,
    `resolve`, shapes) and the rules it rewrites with, which read shapes:
    `symbolic.py`'s `symbolic_simple`, `commutative` and `symbolic`,
    `divandmod.py`'s `div_and_mod_symbolic`, `movement.py`'s `mop_cleanup`
    and `weak.py`'s `pm_uncast_const`. `simplify` precedes the rules and reads
    them through one reference that the line after them sets, so a
    program that links `Shape` has them. `Symbolic` keeps `sym` and the rules
    on conditions, which call `simplify`. The other `UOp` methods that read
    shapes, of storage, variables, calls and programs, are
    `Shape`'s too;
  - `render.py`'s `pretty_print` is `Ops.pp`: it prints arguments
    (`argstr`), and arguments print the nodes they hold, so the two recurse;
  - the kept methods of `mixin/*.py` are functions of `Ops`, or of `Shape`
    for those that read shapes, since `ops.py` calls them and a module cannot
    call a later one; the elementwise ones, which patterns share with nodes,
    are one functor applied to both;
  - the small types that `ops` and `spec` name from later files
    (`Estimates`, `BufferizeOpts`), and `device.py`'s `is_disk_device`, are
    defined in the earliest module that needs them; `Opt`, whose file
    depends on nothing, stays in its own module;
  - `spec.py`'s imports of `codegen.opt`, `schedule.rangeify` and
    `renderer` serve only `pyrender_globals`, which is not ported (see the
    README), so they need no break;
  - `device.py`'s `Compiler` and `CompileError` are `Renderer.Compiler`,
    since a renderer holds its compiler and `Device` follows `Renderer`;
  - `apply_opts` takes the optimiser as an argument, and `Search` follows
    `Postrange` and takes how to link and time a program as its `link`
    and `time` arguments, where `beam_search` takes `rawbufs` and
    `var_vals`: its caller makes the buffers `args_from_ast` describes
    (`Tolk_engine.slots`), since `tolk` cannot name rig's buffers,
    and `get_kernel_actions` returns each candidate with its action, without
    tinygrad's `include_0` and its positions in the action table;
    `Codegen.full_rewrite_to_sink` and `Codegen.to_program` take
    the beam search as their `beam` argument, a function of the width the
    kernel asks for, and raise when a kernel asks for one and none is given;
  - the engine has one order (schedule, realize, jit), and each late
    binding is passed as a function argument, never a global reference:
    `Realize.compile_linear` takes the target of each device from its
    caller;
  - the compiler modules precede `Cstyle`, and `ops_metal.py`'s
    `MetalCompiler`, which `cstyle.py` imports inside `MetalRenderer`, is
    `Compiler_metal`, one of them; the rest of `ops_metal.py` is rig's
    (D146).
- **Reason:** (a). Each layer's review checks that its breaks are the
  smallest possible.
- **Pinned by:** for `Codegen`:
  `Tolk.Codegen › beam search › a kernel that asks for a beam of
  width w is optimised by beam w` and `› raises Invalid_argument when a kernel
  asks for a beam and none is given`; for `Ops`: `Tolk.Ops › resolve ›
  simplify leaves a constant, and a sink of constants and stacks of
  constants, alone`, and the `Tolk.Ops › resolve` tests that simplify with
  `Symbolic`'s rules; `Tolk.Ops › printing › pretty.golden`;
  `Tolk.Ops › elementwise patterns › the pattern operators are the named
  pattern operations`; for `Renderer`: `Tolk.Renderer ›
  Compiler › compile raises what the toolchain rejects`, its
  `Compile_error`; for `Prepare`: `Tolk.Prepare › contiguous_view › a
  reshape keeps the order`; for `Schedule`: `Tolk.Schedule ›
  pm_flatten_linear › Linears nested in Linears are inlined in order`; for
  the engine's order: `Tolk.Realize › lower_and_compile with a beam
  search › raises Invalid_argument for a kernel that asks for one when none is
  given`, and `Jit JITBEAM › JITBEAM unset › a kernel is searched with
  BEAM's width`, which passes the search through `Realize.compile_linear`;
  for `Compiler_metal`:
  `Tolk.Compiler_metal › MTLCompiler › compiles a kernel to a Metal
  library`. Each calls the function where its break puts it, or passes the
  late binding as an argument, so none compiles without the break.

## D5. Compilation workers are domains

- **tinygrad:** `engine/worker.py:1-2` (`multiprocessing` spawn workers);
  `helpers.py:170-187` (`Context` and `ContextVar`, one value per process);
  `codegen/__init__.py:510-518` (`to_program_context`, the settings a worker
  process is started with, and `to_program_cache`, which the parent fills).
- **tolk:** `Worker.spawned` and `Worker.map`; `Setting.t` and
  `Setting.context`; `Codegen.to_program_cache` (`to_program`'s cache);
  `Compiler_metal.build`; `Compiler_worker.run`, and the C files
  `lib/runtime/support/compiler_worker.h`, `compiler_amd_worker.c` and
  `compiler_cuda_worker.c`.
- **Differs:** compilation runs on domains, not processes. `Worker.map`
  spawns its domains for the call and joins them before it returns, where
  tinygrad keeps a pool: an idle domain still takes part in every minor
  collection. So the process machinery goes with the pool: the hidden
  `__main__`, the copied environment, the recycling of a worker after 16
  tasks, and `terminate_worker_pool`. A tinygrad worker ignores SIGINT,
  which a domain cannot, since a handler is the process's: the domains
  block every signal but a fault's instead. The workers' context
  (`ALLOW_DEVICE_USAGE`, `VIZ`, `TRACK_MATCH_STATS`) goes too: the compiler
  opens no device and has neither of the other settings. A setting holds one
  value per domain, where tinygrad holds one per process: a domain starts with
  the values of the domain that spawns it, and a `context` override is seen by
  its own domain only. A tinygrad worker process has its own settings, so a
  `Context` entered while it compiles (the construction check's `CHECK_OOB=0`)
  leaves the others alone; process-wide settings shared by compiling domains
  would let one domain's override and restore clobber another's. A spawned
  domain starts with its spawner's settings, so `to_program_context` goes
  too. `to_program` keeps its programs, and `lower_sink_to_linear` its
  schedules, in a table that every domain reads and fills under a lock
  (`Helpers.Memo`), and a domain that asks for one another is making waits
  for it, so each program is compiled once, as tinygrad's parent compiles
  each key once, and each body is scheduled once. MTLCompiler, which a tinygrad worker loads
  in its own process, is loaded once for every domain, and it re-parses its
  options into LLVM's global option registry on every build, which is not
  thread-safe: its builds run one at a time. comgr serialises the compiles
  of a process on a mutex of its own, and NVRTC 13.3 and 13.4's assembler
  crashes on some valid sources, so comgr and NVRTC compile in processes
  of their own, spawned by the compiling domains, where a tinygrad worker
  process loads the library once and compiles in it. Making the compiler
  finds the library, and those processes load it, so a file found that is
  no library raises `Compile_error` at each compile, where it fails making
  tinygrad's compiler. Each process is a small C program per
  library that tolk carries, which runs from memory on Linux and from a file
  in `Helpers.cache_dir` elsewhere. A comgr process runs one compile,
  through files. An NVRTC process serves compiles in turn over a socket (two
  pipes on Windows), idle between them in a list every domain takes from,
  since a process's first compile costs about 100 ms, most of it NVRTC
  starting CUDA's driver; a compile that finds none idle starts another. A compile whose process ends
  without a reply raises `Compile_error` naming the signal or exit status,
  where it would end tinygrad's worker, and that process serves no other.
  The NVRTC process disables NVRTC's own cache of binaries
  (`CUDA_CACHE_DISABLE=1`), whose key misses the math options
  (`--use_fast_math`, `--prec-sqrt`, `--prec-div`, `--ftz`), `-I`, `-D` and
  `--std`, so it could serve tolk a binary built under other options or
  against other headers.
- **Reason:** (a).
- **Pinned by:** `Tolk.Setting › context › is not seen by the other
  domains` and `Tolk.Setting › context › binds for the domains spawned
  while it runs`; `Tolk.Worker` (every test); `Tolk.Codegen ›
  programs are kept › calls from several domains at once make one program,
  compiled once`; `Tolk.Compiler_metal › MTLCompiler › compiles
  from several domains at once`; `Tolk.Compiler_amd › a compile in a process
  of its own` (every test) and `› a process that cannot start`;
  `Tolk.Compiler_cuda › compiles in processes of their own` (every test),
  `› a library that does not load` (every test), and `› a process that cannot
  start`; `Tolk.Compiler_amd › a library that does not load` (every test); `Tolk.Compiler_cuda (fork) ›
  a forked process` (every test).

## D6. Devices are named, never parsed

- **tinygrad:** `device.py:28,38,389,468` (device strings split at `:`).
- **tolk:** `Device.renderers`, `Device.renderer`.
- **Differs:** the caller gives a target, whose device is a kind and whose
  `arch` is the device's; tolk picks the renderer and compiler from the
  target and never parses the name, with one exception: a name starting with `DISK` is a disk, as
  tinygrad reserves it and rig names its disk device
  (`Ops.on_disk`, `is_disk_device`, `copy_to_device`, `clone`).
- **Reason:** (c).
- **Pinned by:** for targets, `Tolk.Device › renderer takes a device's
  name as it is` (a name with an index, in lower case, or a disk's, is no
  device, and a name gives no renderer or architecture) and `Tolk.Device ›
  renderer › picks a device's renderer as tinygrad does` (the architecture
  comes from the target); for the disk, `Tolk.Ops › several
  devices › on_disk holds for one disk device`, `Tolk.Ops › several
  devices › copy_to_device rejects a disk and a weak type` and `Tolk.Ops ›
  storage › clone rejects a disk`.

## D7. Withdrawn

A device's stamp slots followed nx.device's `Submission.record`, so a batch's
kernels were spans of a profile. rig times each submission itself, and tolk
stamps nothing (D146).

## D8. The disk cache maps strings to strings

- **tinygrad:** `helpers.py:393-442` (an SQLite database of pickled values,
  keyed by strings, integers or dictionaries of columns).
- **tolk:** `Helpers.Diskcache`; `Diskcache.generation`, `Diskcache.renew`
  and `Diskcache.clear`.
- **Differs:** keys and values are strings, which callers encode. Each entry
  is a file under `CACHEDB`, a directory, written aside and renamed into
  place; tables are versioned by tolk's own version. Entries belong to a
  generation, an empty file named by a count and a unique name; a clear
  makes one past the newest, which empties every table at one step, then
  removes the old files. tinygrad's clear (`helpers.py:411-414`) drops the tables one by one,
  each drop committed on its own, so a reader between two drops sees one
  table cleared and another not. A tolk from before generations reads an
  entry of one as a write in progress: its clear and trim remove it only
  once an hour old, and its trim does not count it.
- **Reason:** (a). OCaml has no pickle, no type-safe serialization of
  arbitrary values: a cache returning any type would be `Marshal`, which
  crashes when a rebuilt program reads a value of a changed type. Without an
  SQLite transaction, writing aside and renaming is what keeps an entry whole
  when a writer crashes or races another. Without SQLite's drop, a clear
  that removes files one by one orders against no call, and a generation per
  table would cost what one costs while promising less.
- **Pinned by:** `Tolk.Helpers › Diskcache › get reads back any key and
  value put`, `Tolk.Helpers › Diskcache › behaves as a table of entries per
  table`, `› a put and a get on another domain come before or after a
  clear`, `› a put and a get in another process come before or after a
  clear` and `Tolk.Helpers › Diskcache on two domains › behaves as a table
  of entries, whatever the order of the calls`, which run with `CACHEDB`
  set, as dune runs them.

## D9. Narrow float conversions are IEEE conversions

- **tinygrad:**
  - `dtype.py:227-231` (`float_to_bf16` rounds to float32 with
    `truncate[dtypes.float]`, then to bfloat16), and `dtype.py:87` (`const`
    converts an integer to a float through a double, `float(val)`);
  - `codegen/decomp/dtype.py:35-37,98-124,183-184`: `l2i` converts a long
    to a float32 word by word; an emulated cast narrows its source to float32
    first; `f2f` flushes subnormals to zero both ways, but for those within
    half a subnormal ulp of the least normal, which round up to it, and
    narrows every NaN to its format's one quiet NaN of its sign.
- **tolk:** `Dtype.encode_format` (one rounding for every narrow float) and
  `Dtype.float_of_integer`; `Decomp_dtype.long_to_float`,
  `Decomp_dtype.narrow` and `Decomp_dtype.f2f`.
- **Differs:** converting to a narrow float rounds once, to nearest with ties
  to even, from any source, and treats every value as IEEE does, folded or
  emulated:
  - a double rounds to bfloat16 once: `1 + 2^-8 + 2^-40` is `1.0078125` here
    and `1.0` in tinygrad. An integer converts once, from its exact value:
    `9042383626829825` is bfloat16 `0x5a01` here and `0x5a00` in tinygrad,
    whose double rounds it onto a tie. An emulated cast narrows a double, or
    an integer more precise than a float32, to float32 by rounding to odd, so
    that the cast's rounding to the narrow float is the one rounding; an
    emulated 64-bit integer
    converts to a float32 once, where tinygrad's word arithmetic rounds each
    word and their sum;
  - emulation keeps subnormals, both ways.

  A NaN converts as the NaN rule allows, which no golden distinguishes:
  `Dtype.truncate`, the folder, quiets it and, in a 16-bit or wider float,
  keeps its sign and the top of its payload (an 8-bit float's is its format's
  NaN, D10 and D20); the bfloat16 cast of Clang and HIP
  (`Cstyle.pm_manual_bf16_cast`) shifts a float32's bits, so a signalling NaN
  may stay signalling; `f2f` narrows every NaN to its format's quiet NaN of
  its sign, as tinygrad's does. Each gives a NaN, signalling only from a
  signalling one, so a folded NaN and the same NaN computed in a kernel may
  differ in bits. A copy or a selection, which converts nothing, keeps a
  signalling one (D62).
- **Reason:** (b). rune folds constants with `Dtype`, and nx converts eagerly
  with one rounding. An emulated kernel exists only because its target lacks
  the type, so it must give the values a native one gives. The emulated
  cast's rounding to odd: (e). rune hands tolk only a float32 to cast to a
  narrow float, so no raven call site reaches it, but without it the cast
  rounds twice, first to a float32 to nearest: `1 + 2^-8 + 2^-40` becomes
  bfloat16 `0x3f80` (1.0) where IEEE's one rounding, which `Dtype.truncate`
  folds, gives `0x3f81` (1.0078125). Repro on the CPU with
  `EMULATED_DTYPES=bfloat16`: `Tensor([1 + 2**-8 + 2**-40],
  dtype=dtypes.float64).cast(dtypes.bfloat16).bitcast(dtypes.uint16)`
  gives `0x3f80`. The 8-bit facets: (e). rune hands tolk no 8-bit float,
  since it stores each as `Uint8` and converts it itself (rune's
  `Lower_arith.dtype`), but without them an emulated 8-bit float flushes its
  subnormals both ways, where tinygrad's own folder (`dtype.py:258-263`,
  `float_to_fp8`) rounds to them: on the CPU with
  `EMULATED_DTYPES=fp8e4m3,fp8e5m2`, `Tensor([2**-9]).contiguous().cast(
  dtypes.fp8e4m3).bitcast(dtypes.uint8)` gives `0x00` where the folder gives
  `0x01`, e5m2's `2**-16` likewise, and the code `0x01` widens to `0.0`; and a
  double rounds twice, through a float32: the float64 `1 + 2**-4 + 2**-40`
  becomes e4m3 `0x38` (1.0) where the folder gives `0x39` (1.125).
- **Pinned by:** `Tolk.Dtype › truncate › truncation.golden` (the
  near-tie rows, stated in code), `Tolk.Dtype › truncate › bfloat16 rounds
  once, to the nearest, ties to even` and `Tolk.Dtype › truncate › an
  integer rounds to a narrower float once, from its value`; for emulation, one
  test per facet in `Tolk.Decomp_dtype › narrow float conversions are IEEE
  conversions`: `an emulated narrow float keeps its subnormals, both ways`, `a
  double, or an integer more precise than a float32, rounds once` (the
  rounding to odd: `1 + 2^-8 + 2^-40` is `0x3f81`), `an emulated 64-bit integer converts to a float32 once`, `an fnuz format stores
  an underflow to negative zero as positive zero`; `Tolk.Decomp_dtype › NaNs ›
  f2f narrows a quiet NaN to a quiet NaN` and `› an emulated cast of a quiet
  float32 or double NaN is a quiet NaN`, which state the rule;
  `Tolk.Decomp_dtype › goldens`, which are tinygrad's
  graphs with `f2f` replaced by these conversions. The 8-bit facets' values:
  the e4m3, e5m2 and fnuz cases of `Tolk.Decomp_dtype › narrow float
  conversions are IEEE conversions › an emulated narrow float keeps its
  subnormals, both ways`, and `Tolk.Decomp_dtype › emulated narrow floats ›
  an emulated cast of doubles near every <format> tie rounds once` for each
  8-bit format.

## D10. Withdrawn

e4m3's `0xff` decoded to a negative NaN. Under the NaN rule a decoded NaN's
sign is unspecified, and rune hands tolk no 8-bit float, so
`Dtype.decode_format` decodes every e4m3 NaN as tinygrad's positive
`math.nan` (`dtype.py:276`).

## D11. Kernel optimisations are typed

- **tinygrad:** `codegen/opt/__init__.py:10-15` (`Opt(op, axis, arg)`, with an
  `arg` of any type), checked by `codegen/opt/postrange.py:93-150`.
- **tolk:** `Opt.t`, `Opt.target`.
- **Differs:** each kind of optimisation is a constructor with its own
  fields, which stand for `OptOps` too, and a split's target is one of the
  two axis types a split can make (`split_targets`, `postrange.py:14-15`),
  `Local` and `Upcast`, in their axis types' order. The malformed arguments
  that `postrange.py` refuses at run time cannot be built.
- **Reason:** (a): static types.
- **Pinned by:** the type itself, whose printing `Tolk.Ops › arguments ›
  reprs.golden` checks against tinygrad's `Opt` repr (the row "kernel with
  opts"); `Tolk.Postrange` drops tinygrad's malformed-argument cases with
  this entry as the reason.

## D12. A node's key is a BLAKE2 digest

- **tinygrad:** `uop/ops.py:253-255` (`key`, the SHA-256 of
  `str((op, dtype, arg))` followed by the keys of the sources).
- **tolk:** `Ops.key`.
- **Differs:** the same text is digested with BLAKE2b-256, but for a string
  or bytes argument, which enters as its own digest (D193).
- **Reason:** (a). No key is ever compared with a key tinygrad computed: keys
  name compiled programs in caches that tolk alone writes, so the digest
  is free to differ, and tolk takes the one OCaml's standard library
  has, which has MD5 and BLAKE2 but not SHA-256.
- **Pinned by:** `Tolk.Ops › key › tells arguments apart`.

## D13. Folding reads and writes committed constants at their width

- **tinygrad:** `uop/symbolic.py:30` (`fold_const_alu`), which writes a
  folded integer result at its type's width; `:160-162` (the collapse of
  committed const conversions), and `fold_const_alu` and `:170-173` (two
  stage ALU folding), which read a weak operand of an operation on a
  committed integer as written;
  `uop/ops.py:1053-1119` (`_min_max`), which bounds such an operand by its
  written value; `uop/weak.py:90-95` (`uncast_const`), which leaves the
  literal bare. tinygrad's own `TestModularWraparound` expects the wrapped
  results and is marked `xfail_broken_const_wraparound`: its cases fold to
  them at this commit, but for the signed one of `test_div`, whose `//`
  floors where the test truncates.
- **tolk:** `Shape.fold_const_alu`; the two stage fold of a maximum in
  `Shape.symbolic_simple`; `Ops.at_width` and
  `Ops.operand_bounds`; `Shape.uncast_const`.
- **Differs:** a fold reads a cast of a literal to a type of known width wrapped to that width,
  and so does a cast of it that collapses; it reads a weak integer operand of
  an operation on a committed integer at that operation's width, as compiled
  code commits it; and it writes a committed integer result wrapped. The
  bounds read operands the same way. So `uncast_const` only ever drops a cast
  whose literal fits. tinygrad reads a weak operand of an operation as
  written, which a sum, a product and a bitwise operation fold the same
  before or after wrapping, but an order and a division do not: on uint8,
  `max(-3, a)` becomes `a` where the machine computes `max(253, a)`, and on
  uint32, `max(1, b) // -2` folds to `-1`, which is `4294967295`, where the
  machine divides by `4294967294` and gives `0`. A maximum orders the
  wrapped values, so the two stage fold reads a maximum's weak constants at
  its width: on uint8, `max(max(w, 1), -3)` would become `max(w, 1)` where
  the machine computes `max(w, 253)`. A maximum by bounds that keeps a weak
  operand commits it to the maximum's type, so the operations that read it
  do not lose their width. Reading at the width also keeps
  `c0 + x < c1 → x < c1 - c0`, which both apply to signed `x` only, from a
  wrong answer where the offset wraps; D24 keeps that rule to offsets that
  do not wrap.
- **Reason:** (b): RFC 0012's Law 1: every integer expression's compiled
  value is eager nx's modular one, signed included. rune's `rune jit ›
  classes › integer constants keep their bits and wrap` and `› a square of a
  product whose constant squares to 0`, at every signed and unsigned width,
  compile constants whose arithmetic wraps against eager nx.
- **Pinned by:** `Tolk.Symbolic › symbolic_simple › constants › an
  operation reads committed constants at their width` (a fold, a cast of a
  committed constant, a uint8 remainder), `Tolk.Symbolic ›
  symbolic_simple › constants › a comparison reads a committed constant at its
  width` (the uncast), `Tolk.Symbolic › committed constants hold their
  type's value` (each case above, evaluated before and after), the law
  `Tolk.Symbolic › laws › sym keeps the value of an integer expression at
  a committed width, wrapping included`, and `Tolk.Symbolic › tinygrad ›
  tests.golden › TestModularWraparound.<test>`, whose goldens, generated with
  the same change in `test/gen/tinygrad.patch`, hold the machine values;
  the offset's interaction with D24: `Tolk.Symbolic › integers wrap ›
  an offset crosses a comparison only where neither side wraps` (the
  committed case).

## D14. Withdrawn

Folded comparisons of NaN constants followed IEEE here, where tinygrad compared
them as `ConstFloat`s, which make NaN equal to NaN. At
`uop/symbolic.py:30,74-76` `fold_const_alu` reads each operand through
`const_arg`, which turns a `ConstFloat` into a plain float, so `nan != nan`
folds to `True` there as in `Shape.fold_const_alu`; tinygrad's
`test_const_folding.py::TestWeakConstFolding::test_nan_compare` states it.

## D15. Withdrawn

Making a compiler loaded nothing: its library was loaded at the first compile,
so a renderer for CUDA, HIP or Metal was made on a machine without NVRTC, comgr
or MTLCompiler. No raven call site made such a renderer without compiling with
it, so the entry had no reason; tolk's suites and the bench's `search` rows
rendered for those targets without the libraries. As tinygrad's
`runtime/support/compiler_cuda.py:52` (`nvrtcVersion`),
`runtime/support/compiler_amd.py:81` (`HIPCompiler` asserts comgr is loaded) and
`runtime/ops_metal.py:39` (`MetalCompiler`) do when the compiler is made,
`Compiler_cuda.nvrtc` and `Compiler_amd.hip` find their library and
`Compiler_metal.compiler` loads MTLCompiler, raising `Failure` without it, so
`Cstyle.cuda`, `Cstyle.hip`, `Cstyle.metal` and `Device.renderer` raise it too.
The tests that render for those targets skip without the library, as tolk's
GPU tests do, except HIP's: no machine here has comgr, so the suites that
render for HIP name a file that stands in for it in `COMGR_PATH`, as
tinygrad's goldens' generator marks comgr loaded. NVRTC and comgr are loaded by the processes that compile (D5),
so a file found that is no library fails each compile.

## D16. Withdrawn

A CUDA conversion to an 8-bit float kept an infinity special: an infinity in
e5m2 and a NaN of its sign in e4m3, where nx stores ±448. rune hands tolk no
8-bit float, so `Cstyle.cuda` converts with the saturating constructor, as
tinygrad's `renderer/cstyle.py:25-26,33,40` do: an infinity becomes the
greatest finite value of its sign in both formats. This departs from nx in
e5m2, which keeps an infinity, and from the folder, where `Dtype.encode_format`
keeps one in e5m2 and makes it a NaN in e4m3, as tinygrad's `float_to_fp8`
(`dtype.py:246-247`) does beside its own CUDA kernels.

## D17. Each operation on a narrow signed scalar is narrowed in the source

- **tinygrad:** `renderer/cstyle.py:260-265` (an ALU used once is inlined into
  its user, whatever its type) with `:65-66` (the operator text);
  `:85-87` (`pm_narrow_uint`), which computes the operations on an unsigned
  char or short that wrap in an unsigned int and casts them back, for these
  two types only.
- **tolk:** `Cstyle.lang`'s field `promoted`, set in `Cstyle.cstyle` and
  `Cstyle.clang_lang`; `Cstyle.render_uops` (`narrowed`, and the cast in
  `visit`); `Cstyle.base_rewrite` (an operand's parentheses).
- **Differs:** C, C++ and Metal compute an operation on a `char` or `short`
  in `int`, and Clang computes one on an `__fp16`, a storage format, in
  `float`. Only an assignment narrows the result back, so an inlined
  operation neither wraps nor rounds until the kernel stores: `(char)100 +
  100`, halved, gives `100` where each operation wrapping gives `-28`. The
  source is tinygrad's for the kernel with a cast to its own type after each
  inlined scalar operation of one of these types that its user does not
  store. The unsigned types take tinygrad's rule, whose unsigned int computes
  each wrapping operation without the signed overflow of `int` that
  `65535 * 65535` would be. Vectors are not promoted, and CUDA's `__half` and
  `nv_bfloat16`, HIP's `_Float16` and Metal's `half` and `bfloat` compute
  natively, so they keep tinygrad's source; so does every kernel without such
  an inlined operation. Clang's, HIP's and Metal's rules were checked with
  their compilers; CUDA's rests on the operators `cuda_fp16.hpp` and
  `cuda_bf16.hpp` declare, and is on the hardware checks of
  `test/README.md`.
- **Reason:** (b). Eager nx and compiled code agree: nx computes each
  operation in its own type, as the folder (`Ops.exec_alu`) and tinygrad's
  interpreter do.
- **Pinned by:** `Tolk.Cstyle › narrowing`, which checks for every
  kernel that the kernel whose tinygrad source is its source is the kernel
  with those casts, and `Tolk.Cstyle › sources`, which compares the
  sources it changes with tinygrad's for that kernel; `Tolk.Cstyle ›
  execution on the host › rounds each operation on halves to a half`; and
  the slow `Tolk.Cstyle › execution on the host › a kernel over a narrow
  type wraps and rounds as the interpreter`, over each type's whole range.
  The unsigned types' rule: `› wraps each operation on unsigned chars`.

## D18. Withdrawn

Metal computed a bfloat16 `TRUNC` in float32 here, where tinygrad truncated the
bfloat, which Metal has no `trunc` of. At `renderer/cstyle.py:390`
`MetalRenderer.extra_matcher` computes `TRUNC` in float32 with `SQRT`, `EXP2`,
`LOG2` and `SIN`, as `Cstyle.metal_extra_matcher` does.

## D19. Withdrawn

A hierarchical allreduce to one device landed there, where tinygrad's stayed on
every device of its source. Withdrawn: multi-device is out of scope, see
README.

## D20. Withdrawn

Storage kept an e5m2 NaN's two payload bits, so a folded bitcast kept every
e5m2 code. rune hands tolk no 8-bit float, so no consumer folds such a
bitcast: `Dtype.encode_format` stores every e5m2 NaN as `0x7f` of its sign and
`Dtype.decode_format` reads it as the NaN of its sign, as tinygrad's
`dtype.py:248,275` do.

## D21. Withdrawn

A reshape of a value sharded on two axes divided each by its own shard count,
where tinygrad's divided both by the last one's. Withdrawn: multi-device is out
of scope, see README.

## D22. An emulated long converts no float past its words' range

- **tinygrad:** `codegen/decomp/dtype.py:32-33` (`l2i` makes a long's low
  word by casting the float to an int32, and its high word by casting the
  float over 2^32).
- **tolk:** `Decomp_dtype.l2i` (the cast of a float to a long).
- **Differs:** the words are the quotient and remainder of the truncated
  float's magnitude by 2^32, each converted from a float the word holds, and
  negated as a long for a negative float. tinygrad converts a float past
  2^31 to an int32 word, which C leaves undefined: ARM saturates and x86 gives
  `0x80000000`, so the low word of `2^32 + 5` is `0x7fffffff` on ARM.
- **Reason:** (b): rune's kernels on a target without 64-bit integers cast
  floats to longs through this emulation, and must give the long nx gives.
- **Pinned by:** `Tolk.Decomp_dtype › emulated 64-bit integers › an
  emulated cast of a float32 to a 64-bit integer converts no float to a word
  that cannot hold it`.

## D23. Withdrawn

A shard selected through a movement read its own shard, the movement's device
range taking the selected shard's position. Withdrawn: multi-device is out of
scope, see README.

## D24. Rewrites keep IEEE and modular values

- **tinygrad:** the rules and bounds each facet names below.
- **tolk:** each facet's rules below; `test/gen/tinygrad.patch`, the same
  restrictions applied to tinygrad, which `test/gen/generate.py` applies to a
  copy of the checkout before generating the goldens and which is re-applied
  when the pin moves.
- **Differs:** tinygrad's rewrites assume real arithmetic: integers that never
  wrap and floats that neither round, overflow, nor carry signed zeros and NaN.
  A rewrite here keeps the value compiled code computes, bit for bit: an
  integer of a committed type wraps at its width, as C's unsigned arithmetic,
  Metal, CUDA and nx do, and the lowering computes signed arithmetic on the
  unsigned bit pattern; a float follows IEEE. A rewrite that holds only in
  real arithmetic applies to integers and booleans, and to a committed integer
  only where no value it computes wraps (`Ops.exact`). Index arithmetic is weak
  (`Weak_int`), never wraps, and keeps every fold tinygrad makes, so the
  restrictions touch float and committed-integer values only. The one rounding
  that stays is that of float powers, a constant one computed by products and
  square roots and any other as `exp2 (y * log2 x)` (`Transcendental.xpow`).
  - **Integer bounds wrap.** tinygrad: `uop/ops.py:58-60` (`wrapped`),
    which `_min_max` (`:1053-1119`) applies to an unsigned sum, difference,
    product, left shift and complement, `:1109-1119` (a cast), `:1102-1104`
    (a constant table). tolk: `Ops.at_width`, `Ops.min_max`,
    `Ops.cast_bounds`, and the constant table's case of
    `Ops.compute_min_max`. The bounds of a committed integer that leave its
    type are one value wrapped, or the type's, signed or not, whatever the
    operation: tinygrad wraps those of the unsigned operations alone, and to
    the type's, one value included; an integer cast to an integer keeps its
    interval, which then wraps, where tinygrad gives a cast that leaves its
    type the type's bounds, one value included (69c2c1bce); a constant table holding a NaN has its type's bounds, as a NaN
    constant has. tinygrad folds int8 `(y + 100) < 0` for `y` in `[0, 50]` to
    `false`, which is `true` at 40, and so `y * 4 < 0` and `y << 2 < 0`,
    `true` at 40 too, and `cast (cast (y + 100) uint8) int32` to one cast,
    `-116` where the machine gives `140`.
  - **Wrapping rules.** tinygrad: `uop/symbolic.py:298-300` (`(x // c1) //
    c2`), `:302-304` (`c0 + x < c1`), `codegen/simplify.py:99-104` (`x + y <
    c`, `x * y < c`) and `:124-126` (`x + y = c` under a cast),
    `uop/symbolic.py:402-412` (`reduce_mul_chain`, a factor out of a maximum),
    `uop/symbolic.py:315-316` (`-x < -y`, `-x < c`). tolk:
    `Shape.symbolic_simple`, `Shape.symbolic`, `Simplify.solves_sum`,
    `Simplify.pm_reduce_collapse`, `Simplify.pm_reduce_load_collapse`,
    `Symbolic.reduce_mul_chain`. tinygrad keeps `c0 + x < c1`, the comparisons
    of `Simplify` but the last, and a factor out of a maximum from unsigned
    integers, which wrap, and applies them to signed ones, whose overflow it
    takes to be undefined, checking only that the constants it makes fit the
    type (`fits`, `uop/symbolic.py:32`); tolk applies each to a signed integer
    only where every value it computes fits the type, and so `x + y = c` to any
    committed integer, and `-x < -y` and `-x < c` to a signed one only where no
    negation wraps. The comparisons of `Simplify` apply to integers only, since
    moving a float term rounds, and `x + y = c` only under a cast that does not
    narrow, where tinygrad takes any cast whose operand's bounds fit its type
    (7e36b6735). `where (a < b) c (a + (c - b))` is a maximum for a signed `a`
    only where `a + (c - b)` does not wrap (`uop/symbolic.py:292-294`), where
    tinygrad takes signed overflow to be undefined. tinygrad folds the int8 maximum of `r * p * q` over `r < 2` to `p *
    q`, `-112` at `12, 12` where the machine gives `0`, and int8 `-x < 3` to
    `-3 < x`, `false` at `x = -128` where the machine gives `true`.
  - **Float folds.** tinygrad: `uop/symbolic.py:126` (`x + 0`), `:181-186`
    (`x / x`, `x * 0`). tolk: `Shape.symbolic_simple`. A float `x + 0` is `x`
    only for `-0.` (`-0. + +0.` is `+0.`); `x * 0` is `0` for integers and
    booleans only (a float product by zero is NaN at an infinity or a NaN
    and `-0.` at a negative `x`); `x / x` is gone, since true division is
    always a float, and `x / x` is NaN at 0.
  - **Signed zeros.** tinygrad: `uop/symbolic.py:284-285` (complementary
    selections). tolk: `Shape.symbolic`. For integers and booleans only:
    `where c t 0 + where c 0 f` at `t = -0.` is `+0.`, where `where c t f` is
    `-0.`.
  - **Reassociation.** tinygrad: `uop/symbolic.py:170-173` (two constants
    of an associative operation), `:251-264` (like terms), `:281-283` (a sum
    of two selections), `:308-311` (constants to the end), `:402-412,482`
    (`reduce_mul_chain`). tolk: `Shape.fold_where_closure`,
    `Shape.symbolic_simple`, `Shape.symbolic`, `Symbolic.reduce_mul_chain`.
    Sums, products and maxima regroup for integers and booleans only, and a
    factor leaves a float reduction nowhere: `(x + 1e8) + -1e8` at `x = 1` is
    `0.`, where `x + 0.` is `1.`; `(y + x) + x` at `y = 1`, `x = 2^-24` is
    `1.`, where `y + x * 2` is `1.0000001`; `x * 1.1 + x * 2.2` at `1.` is
    `3.3000002`, where `x * 3.3` is `3.2999999`; a sum over `r < 3` of
    `(r * 2e38) * 0.25` is `inf`, where `0.25 * 2e38` times the sum of `r` is
    finite; and a maximum keeps a NaN only as its first operand, so
    `max (NaN, max (x, 0.))` is NaN, where `max (x, max (0., NaN))` is `x`.
    `x + x` is still `x * 2`, which is exact, and `(x * -1) * y` is `(x * y) *
    -1`, since a negation commutes with rounding. tinygrad restricts these
    rules to integers too (7e36b6735), the two stage fold of a sum since
    5e826e373, but not that of a product or a maximum, nor `x + x * -1`,
    which it makes `x * 0`: at `x = -1.` that is `-0.`, where the sum is
    `+0.`. It checks that an intermediate it makes does not overflow
    (`no_overflow`, `uop/symbolic.py:34`), which holds wherever tolk would
    check it, since the bounds of a committed integer wrap (`Ops.at_width`),
    so tolk makes none.
  - **Maxima.** tinygrad: `uop/symbolic.py:290-295`. tolk:
    `Shape.symbolic`. A maximum by bounds applies to integers
    only, since a float's bounds leave out NaN and the order of zeros:
    `max (x, inf)` at NaN is NaN, where the fold gives `inf`. A selection that
    computes a maximum becomes one for integers only: rune builds a float
    maximum that follows IEEE from selections, which the fold would turn back
    into a maximum that renders as the target's own, and it took `-0.` and
    `+0.` for one constant: `where (x < -0.) 0. x` at `-1e-45` is `+0.`,
    where `max (x, -0.)` is `-0.`. A float ReLU stays a selection.
  - **Sigmoid forms.** tinygrad: `uop/symbolic.py:478-480`. tolk: the rules
    are left out of `sym`. `x * (1 / (1 + x))` at `1e-8` is `1e-8`, where
    `1 - 1 / (1 + x)` is `0.`, and at `inf` is NaN, where it is `1.`.
  - **Pow.** tinygrad: `uop/symbolic.py:16-22` (`simplify_pow`), `:199-201`
    (`c ** x`). tolk: `Shape.simplify_pow` and the `c ** x` rule of
    `Shape.symbolic_simple`. The reciprocal of the
    base is taken for exponents of magnitude at least 1 only, where the power
    overflows whenever the reciprocal does: `1e-40 ** -0.8` is `1e32`, where
    `(1 / 1e-40) ** 0.8` is `inf`. A float half-integer power selects `+0.`
    at `-0.` and `+inf` at `-inf`, where `sqrt` gives `-0.` and NaN. `c ** x`
    goes to `xpow` for `c = inf`: `inf ** 0` is `1.`, where
    `exp2 (0 * log2 inf)` is NaN. `1 ** x` is `1` of the power's type, as
    IEEE's `pow (1, x)` is at every `x`, NaN included: tinygrad no longer
    folds it (89ce9c494, which fixed the type of the `1` it gave), and leaves
    a float `x` to `exp2 (x * 0)`, NaN at an infinity.
  - **Simplify's reductions.** tinygrad: `codegen/simplify.py:105-112`
    (`sum_between`), `:116-117` (a parameter lifted out of a sum) and `:119`
    (a product by a boolean cast). tolk: `Simplify.sum_between`,
    `Simplify.pm_reduce_collapse`.
    A float sum counted in closed form is `+0.` over an empty part of its
    range, where `0 * inf` is NaN; a product by a boolean cast is a selection
    for integers only (`-1. * 0.` is `-0.` and `inf * 0.` NaN, where the
    selection gives `+0.`); a parameter conjunct is lifted out of a sum on a
    selection as a selection: tinygrad lifts it as a product by its cast,
    which its product rule turns back into a selection, and tolk's product
    rule does so for integers only. A sum over a range its value does not
    use, `x * n`, and a sum of a sum split in two stay: each is
    the sum of the same terms in another association, which RFC 0012's rounded
    sum class allows; a sum of `-0.` terms computed as `-0. * n` is `+0.` once
    the lowering adds `+0.` to every float sum's result, which the `x + 0` fold
    now keeps.
  - **Transcendental polynomials.** tinygrad: `helpers.py:154` (`polyN`, from
    `0.0`). tolk: `Transcendental.poly_n`. A
    polynomial starts from its first coefficient, so the decompositions no
    longer rely on the float folds of `0. * x` and `x + 0` to be what tinygrad
    renders.
  - **Tensor-core accumulators.** tinygrad: `codegen/__init__.py:101-103`
    (`pm_wmma_add`, which adds the running sum to a WMMA's accumulator
    operand). tolk: `Codegen.wmma_accumulate`. A WMMA built for a sum starts
    from a zero accumulator, and the running sum used to be added to it,
    `+0. + acc`, which only the float `x + 0` fold removed: with that fold kept to `-0.`, every tensor-core
    loop added `+0.` to each accumulator element on every iteration. The
    running sum now replaces a zero accumulator: the sum's identity is the
    accumulator's initial value, set once before the loop, and the
    lowering adds `+0.` to the result, so a partial sum's zero sign, which
    the rounded sum class leaves open, is all that differs. A nonzero
    accumulator keeps tinygrad's rule, which regroups its sum as
    `(c + acc) + P`: a tensor-core accumulation is part of a contraction,
    whose rounded sum class permits any association of the terms, and no
    exact class reaches it.
- **Reason:** (b). RFC 0012's Law 1: every constructor's compiled result, alone
  or fused, meets its class against eager nx, whose integer arithmetic is
  modular and whose floats follow IEEE. rune's `rune jit › classes › jit gives
  eager's bits for exact functions` and `› … at every dtype` compile drawn
  compositions of sums, differences, products, quotients, maxima, minima and
  selections against eager nx; `Lower_real › exact` compiles each float kind
  at NaN, `-0.`, infinities and subnormals; `Lower_real › transcendental › pow
  within 2 ulps at float32, 2 at float64` checks a power's special values; and
  `rune jit › jit › an integer maximum of a running sum wraps` checks a
  compiled maximum of a wrapping sum against the value nx states. They replace
  the old rune's `check_wrapping_comparisons`, `check_float_identities`,
  `check_signed_zeros` and `check_pow`, which went with its rewrite.
- **Pinned by:** each facet by value tests, which evaluate a graph with the
  reference interpreter before and after the rewrite at bindings where the
  restricted rewrite changes the value:
  - integer bounds: `Tolk.Ops › bounds › a committed integer that can
    leave its type has its bounds`, `› an integer cast to a signed type
    it leaves wraps`, `› a constant table holding a NaN has its type's
    bounds` and the laws `› bounds hold the value a committed integer
    wraps to` and `› bounds hold every value a cast takes`;
  - wrapping rules: `Tolk.Symbolic › integers wrap` (every test);
    `Tolk.Simplify › keeping values`, its comparison tests;
  - float folds, signed zeros, reassociation, maxima, reciprocal and sigmoid
    forms, pow: `Tolk.Symbolic › floats keep IEEE values`, one test
    per facet, with `› a Dekker split and product keep their low parts` and
    `› a float selection that computes a maximum stays`, the structural tests
    marked D24 in `symbolic_simple`, `symbolic` and `sym`, and the law
    `Tolk.Symbolic › laws › sym keeps a float expression's value bit for
    bit at special values`;
  - Simplify's reductions: `Tolk.Simplify › keeping values › a
    float sum over an empty part of a range is +0., whatever the value`,
    `› a float product by a boolean mask keeps its value` and `› a float sum
    lifted past a parameter is +0. where it is false`, and
    `Tolk.Simplify › pm_reduce_collapse › a float product by a comparison
    cast from a boolean stays`;
  - transcendental polynomials: the `Tolk.Transcendental` goldens,
    generated from the patched tinygrad, whose value tables are unchanged;
  - tensor-core accumulators: the sources of every tensor core,
    `Tolk.Cstyle › sources › by default › metal_tc_*`, `hip_tc_*` and
    `metal_matmul`, whose loops add nothing to the accumulator, as tinygrad's;
    `Tolk.Codegen › tensor-core accumulators › the running sum
    replaces a zero accumulator` and `› the lowering keeps a tensor core's value, apart from a
    zero's sign`, which draws accumulators at `-0.`;
  - the goldens, generated from the patched tinygrad.

## D25. A product and a sum round apart, except into a sum

- **tinygrad:** `runtime/support/compiler_cpu.py` (Clang's arguments),
  `runtime/support/compiler_amd.py` (HIP's options), `runtime/support/
  compiler_cuda.py` (NVRTC's options), `runtime/ops_metal.py:60` (Metal's
  parameters): none turns floating-point contraction off.
  `codegen/__init__.py:183-195` (`lower_reduce`: a sum adds its products
  unfused),
  `codegen/decomp/op.py:116-119` (`a*b + c` becomes `MULACC` for a renderer
  that writes it), `renderer/cstyle.py:153-164` (no C-style language writes
  `MULACC`).
- **tolk:** `Compiler_cpu` (`-ffp-contract=off`), `Compiler_amd`
  (`-ffp-contract=off`), `Compiler_cuda` (`--fmad=false`), `Compiler_metal`
  (`#pragma METAL fp contract(off)` before the source);
  `Helpers.Diskcache.version` 2, since a cached binary compiled with
  contraction answers the same key; `Codegen.fuses`,
  `Codegen.lower_reduce`, and the
  operations `Codegen.full_rewrite_to_sink` decomposes with, without `Mulacc`;
  `Cstyle.fma` (`fma` in C-style and Metal, `__builtin_fmaf` and
  `__builtin_fma` in Clang and HIP, `__fmaf_rn` and `__fma_rn` in CUDA);
  `Ops.mulacc`, and its arm in `Ops.exec_alu`; `Shape.pm_data_invalid`'s
  `Mulacc` rule.
- **Differs:**
  - Each compiler would fuse a product and a sum that a rendered expression
    holds together, `a*b + c`, into one multiply-add with one rounding: Clang
    at `-O2` (`-ffp-contract=on`), HIP (`fast`), NVRTC (`--fmad=true`) and
    Metal, whose `-ffp-contract=off` does not reach the code where its pragma
    does. No compiler contracts.
  - A sum of products of float32 or float64, on a renderer that writes a
    multiply-add and has the type natively, adds each product into its running
    sum as one multiply-add (`Op.Mulacc`), rounded once, in the order it adds
    them unfused: an accumulator's `acc + a*b` is `fma(a, b, acc)`, and a
    horizontal reduce's `a0*b0 + a1*b1 + ...` is `fma(a1, b1, a0*b0)` and so
    on, which the accumulator then adds. A horizontal reduce of unrolled lanes
    beside upcast ones sums a permuted view of the products, which the
    expander makes to bring the unrolled axes first; its products fuse too
    (`fuses` looks under the source's movements, as the generator's `fuses`
    looks under `x.base`).
  - `Decomp_op`'s `a * b + c` rule never applies: decomposition takes the
    renderer's operations without `Mulacc`. Every other product and sum keeps
    the two roundings the graph states.
  - A float `Mulacc` that a graph states, as rune's lowering of `Nx.fma` does,
    folds and evaluates rounded once, where `python_alu` (`uop/ops.py`)
    computes `x*y + z` with two roundings: at float64 by `Float.fma`, and below
    by a sum rounded to odd at double precision, which the dtype's truncation
    rounds once. A `Mulacc` moves inside the Invalid gate of each operand, and
    an Invalid operand makes it Invalid, as `pm_data_invalid` does for binary
    operations; no tinygrad tensor graph holds `MULACC`, so it has no such
    rule.
- **Reason:** (b). RFC 0012's Law 1: a
  reduction's result is in the rounded-sum class, whose error bound a fused
  multiply-add stays within; every other compiled result meets eager nx, which
  rounds a product and a sum apart, and rune's accurate compositions
  (two-part products and sums) are exact only where each operation rounds as
  written. The fusion is the IR's, so that only a sum's products fuse, where a
  compiler's contraction fuses any `a*b + c` of an expression. `Nx.fma`
  (RFC 0015's D3) states a `Mulacc` whose compiled bits must be eager's, so its
  fold rounds once too, and `Nx.ewma`, built on it, compiles. Fusing a sum's
  products is (d), on fudge (medians of 11, two interleaved runs each, fused
  then unfused): rune's `Jit/matmul-64` takes 7.3 and 4.9 us against 49 and
  46 us, and the tolk bench's `lorenz/r_128_16_3_4_4_3_3_400_400/heuristic`,
  a kernel of lorenz_simple's step, 0.79 and 0.63 ms against 1.47 and 1.84
  ms.
- **Pinned by:** `Tolk.Compiler_cpu › execution on the host › a product
  and a sum round twice, never fused` (`(1 + 2^-12)^2 - (1 + 2^-11)` is 0,
  where a fused multiply-add gives `2^-24`); `Tolk.Compiler_metal ›
  MTLCompiler › a product and a sum compile under the no-contraction pragma`. On
  an Apple GPU, the same kernel compiled by MTLCompiler gives
  `2^-24` without the pragma and 0 with it, and still `2^-24` with
  `-ffp-contract=off` alone. NVRTC and HIP are README's hardware checks.
  `Tolk.Codegen › multiply-adds › a sum of products adds each into
  its running sum rounded once, of 2` and `› of 64` (the sum of `-(1 +
  2^-11)` and `(1 + 2^-12)^2` is `2^-24`, with a multiply-add in the source)
  and `› a sum of products unrolled beside upcast lanes adds each into its
  running sum rounded once` (four running sums of a loop unrolled by 4, three
  multiply-adds each, and the same `2^-24`) and `› a product and a sum outside a reduction are not
  fused`; the Codegen and C-style goldens, from tinygrad with D25 applied by
  its generator.
  `Tolk.Ops › exec_alu › a float multiply-add folds rounded once` (its
  float32 case past a double's rounding ties to the wrong float32 when the sum
  is not rounded to odd); `Tolk.Symbolic › invalid values › a multiply-add
  moves inside the gate of each operand` and `› a multiply-add of
  invalid is invalid`. In rune, `Lower_arith › exact binary kinds,
  comparisons, where and fma give eager's bits` and `Lower_real › exact ›
  fma`, a compiled `Nx.fma` against eager's bits, and `rune jit › classes ›
  jit gives eager's bits for exact functions`, whose products and sums outside
  a reduction round apart. On rune: missing for a sum's products and for
  `Nx.ewma` (the old rune's `Test_lower_arith › multiply-adds › a product
  that the sum cancels is kept whole` and `Test_compiled_rules › compositions
  › a compiled ewma is its eager value, bit for bit, and so is its gradient`
  went with its rewrite).

## D26. A minus never meets a minus in C-style source

- **tinygrad:** `renderer/cstyle.py:154` (`Ops.NEG` is `-{x}`), `:158`
  (`Ops.SUB` is `({a}-{b})`).
- **tolk:** `Cstyle.joined`, `Cstyle.infix`, and `Neg` in
  `Cstyle.code_for_op`.
- **Differs:** a negation or a subtraction whose operand starts with a minus
  sign, itself a negation or a negative constant, is written with a space
  between the two signs: `(a- -b)`, `- -b`. tinygrad writes `(a--b)`, which C,
  Metal, CUDA and HIP read as the decrement operator and reject. Every other
  source is written as tinygrad writes it.
- **Reason:** (b). rune lowers `Nx.sub` to `Ops.sub` (rune's
  `Lower_real.binary`) and its FFT's twiddles with `Ops.neg` (rune's
  `Lower_fft`), so a difference of a negative constant, as `Nx.sub_s x (-1.)`
  states, or a negation of a negation can reach the renderer; a source the
  compiler rejects is a failed program. No rune test is shown to reach it.
- **Pinned by:** `Tolk.Cstyle › negation › a minus before a minus is
  apart › *` (every renderer), `› Clang compiles and runs a difference with a
  negated operand` and `› … with a negative constant`.

## D27. Constants keep a NaN's bits

- **tinygrad:** `dtype.py:79-85` (`DType.const` makes every NaN `math.nan`),
  `dtype.py:16-19` (`ConstFloat` compares every NaN equal to every other), and
  `uop/symbolic.py:24-27` (`fold_bitcast` converts the constant with
  `truncate` before reading its bits).
- **tolk:** `Dtype.equal_const`, `Dtype.hash_const` and `Dtype.const`;
  `Ops.arg_key`, which `Ops.key` reads; `Shape.fold_bitcast`;
  `test/gen/tinygrad.patch`, which applies the same rules to tinygrad before
  the goldens are generated.
- **Differs:** a float constant is its bits. `Dtype.const` keeps a NaN's sign
  and the payload its type holds, a signalling NaN staying one, in a float of
  16 bits or more (an 8-bit float's NaN is the one its storage holds, as the
  patched tinygrad's is, D10 and D20), and
  constants compare and hash by their bits, so NaNs of different bits are
  different nodes, with different keys: a schedule cached under one NaN's key is never
  the schedule of another's. No golden reads a key, so the patch leaves
  tinygrad's, which writes every NaN `nan`. `fold_bitcast` reads a float
  constant as its type stores it, where tinygrad's conversion quiets a
  signalling NaN and makes an 8-bit NaN canonical. So folding `bitcast(bitcast(0x7d01, half), uint16)` gives
  `0x7d01`, where tinygrad gives `0x7e00`, and a negative float32 NaN's bits
  survive a fold, where tinygrad makes them `0x7fc00000`. A NaN an operation
  makes from other values is `Dtype.nan`, tinygrad's `math.nan`, whatever the
  host's FPU gives, so that folding does not depend on the host: `exec_alu`
  makes the NaN of an invalid operation (`inf - inf`, `0 * inf`, `0 / 0`, a
  square root or logarithm of a negative) canonical, where x86 gives a
  negative NaN, and so does the patched tinygrad's. The NaN rule leaves that
  NaN's bits free, so this part follows tinygrad.
- **Reason:** (b): under the NaN rule a bitcast keeps a NaN's bits, as a
  kernel's and nx's do, so a bitcast that rune folds must keep them, which a
  constant can only if it holds them; a rewrite must keep a graph's value.
- **Pinned by:** `Tolk.Ops › identity › NaN constants of different bits
  are different nodes`; `Tolk.Ops › key › tells apart NaN constants of
  different bits`; `Tolk.Ops › exec_alu › an invalid operation's
  NaN is the canonical positive quiet NaN, whatever the host gives`,
  which compares bits and fails on an x86 host without the canonical NaN; `Tolk.Symbolic › symbolic_simple › casts › a
  bitcast round trip of every 8- and 16-bit word folds to its value`, which evaluates each graph
  with the reference interpreter before and after `simplify`;
  `Tolk.Dtype › const › const keeps the bits of every 8- and 16-bit float
  word but an 8-bit float's NaN codes`.

## D28. A host program has rig's entry

- **tinygrad:** `codegen/__init__.py:422-427` (`pack_args`: a kernel of a
  renderer with `packed_args` loads its arguments from one byte parameter at
  the offsets of `TinyELF.iter_sig`), `:461-465` (`do_compile`: the binary is
  the source compiled), `runtime/ops_cpu.py:47-57` (`CPUProgram.__call__`,
  which packs the arguments at those offsets and calls the kernel once with
  their address).
- **tolk:** `Codegen.host_entry`, `Codegen.do_compile`.
- **Differs:** a program for a CPU target is compiled from its source with
  the kernel renamed `NAME_` and an entry `NAME` appended:
  `void NAME(const void *p, long long first, long long end)`. `p` points to
  the bytes of the kernel's arguments, the ones the kernel takes, and the
  entry passes them on. A split program (D80) reads its bounds from the
  bytes, where a launch passes those of the whole loop, `0` and its
  iterations; its entry runs the iterations `first` to `end - 1`, `end` at
  most the iterations, through a copy of the bytes that holds `first` and
  `end` as its bounds, made with `__builtin_memcpy_inline`. The program's
  source is the renderer's; only its binary differs.
- **Reason:** (c). rig's host calls a function of linked code with one ABI,
  `void f(const void *params, int64_t first, int64_t end)`, places each
  buffer's address among the parameters, and names a submission's span after
  the entry. It runs a launch's blocks of threads on several cores at once,
  each call with its own `first` and `end`, which the parameters every call
  shares cannot hold. The entry keeps the kernel's name in profiles, and on
  x86_64 Windows is declared System V's (D178). Clang compiles a copy of many
  bytes as a call of `memcpy`, which rig takes from the process, and a remote
  Windows host's target names no system (D178), so its object would call that
  `memcpy` by System V: the inline copy calls nothing.
- **Pinned by:** the `Codegen` suite, `programs › a host program's binary
  holds rig's host entry after the kernel` and `› a host program runs as a
  launch of rig's host, with ten scalars past the argument registers` and
  `› a split host program calls no function it does not define`; the
  Tolk_engine suite, `host programs in blocks › a program writes the same
  bits in 1, 3 and 32 blocks` and `link and run › custom kernels with
  variables` (four tests).

## D29. A float32 tensor core takes operands widened from its narrow input

- **tinygrad:** `codegen/opt/postrange.py:166` (`_apply_tc_opt`: a core
  applies only when both operands of the multiply have its input dtype),
  `:203` (the core multiplies the multiply's own operands).
- **tolk:** `Postrange`'s `tc_operand`, `try_core` and `use_wmma`;
  `test/gen/tinygrad.patch`, which gives tinygrad's matcher the
  same rule before the goldens are generated.
- **Differs:** a core whose input is a narrow float (`half`, `bfloat16`, an
  8-bit float) and whose output is `float32` also takes a `float32` operand
  that is a cast of its input dtype, and multiplies the narrow value. The core
  and the widened products need not agree bit for bit: a product of two
  bfloat16s below `2^-126` may round in `float32`, and Ada's `m16n8k16` cores
  sum their products in pairs and truncate. Both are within `Nx.contract`'s
  bound, `γ(K + 1, 2u) (|init| + Σ|a||b|)` for `float32`'s `u`: measured on
  Ada at `K = 2880`, the core errs by `2868u`, inside even `γ(K + 1, u)`,
  about `2881.5u`. A core shown to exceed the bound is disabled. A core with
  a narrow output still takes no widened operand, since its sum rounds to the
  narrow type. rune never gives tolk an 8-bit float (rune's
  `Lower_arith.dtype` stores each as `Uint8`), so no contraction of rune's
  reaches an 8-bit core. Cores are tried in the renderer's order, so Metal,
  whose `float32` core comes first, keeps it. tinygrad never builds the
  widened form, which its `dot` does not make, so no golden it made before
  changes: `generate.py --check` matches every one with the patch applied.
- **Reason:** (b). rune's contraction lowering (`Lower_reduce.entered`), which
  casts each operand to the accumulator's type, computes nx's contraction of
  narrow floats as `MUL(CAST f32 a, CAST f32 b)` summed at `float32` (RFC
  0012, the `Matmul` row); without this rule it takes no tensor core on CUDA
  or HIP, where the `float32` core is TF32 or absent, and
  gpt-oss's `float16` and `bfloat16` products on CUDA lose theirs.
- **Pinned by:** the Postrange suite, `apply_opts optimises a kernel as
  tinygrad does › tc_{cuda,amd}_widened_{half,bfloat}.golden` (the core
  applies on CUDA sm_89 and HIP gfx1100) and
  `› tc_{cuda,amd}_widened_{half,bfloat}_shaped.golden`, with `optimising
  keeps a kernel's writes › small kernels › tc_*_widened_*_shaped` (shaped for
  the core, the kernel keeps its values). On rune, which widens a `half` or
  `bfloat16` operand by tolk's cast (`Lower_arith.decode`), so that its
  contractions and products of operands a program casts first take this
  form: `test_jit_gpu`'s `products › a float16 or bfloat16 matmul is within
  its bound of the exact product`, on the machine's first GPU, at extents the
  core takes and extents it refuses, with `› sixteen terms under float32's
  unit hold the bound` and `a contraction on matrix units holds its bound ›
  *`. Each checks `Nx.contract`'s bound, `γ(K + 1, 2u) Σ|a||b|` for
  `float32`'s `u`, with the result's rounding to its dtype. They pin the
  bound only: whether a core was taken is not observable through rune's
  interface.

## D30. Withdrawn

A range around calls on devices with queues was a loop of their batch, its
commands repeated per trip. A range is a loop of rig's program, one submission
per trip of each submission under it (D146).

## D31. Payne-Hanek reduces exactly, to the nearest quadrant

- **tinygrad:** `codegen/decomp/transcendental.py:66-113`
  (`payne_hanek_reduction`) and `:164-166` (`sin_poly_large`).
- **tolk:** `Transcendental.one_over_two_pi`, `Transcendental.turn_fraction`,
  `Transcendental.radians` and `Transcendental.payne_hanek`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same
  reduction before the goldens are generated.
- **Differs:** tinygrad rounds the quotient on `f < 0.5`, where `f` is
  frexp's mantissa, always at least 0.5, so it always returns the fraction
  less a quarter turn and the next quadrant: `|r|` reaches `pi/2`, and a tiny
  remainder loses its low bits in the subtraction (4.4e-8 absolute in
  float32). It also multiplies only 32 bits of the mantissa by 96 bits of
  `2/pi` from a table of 190 bits, so a float64 is reduced wrong from about 30
  up, and beyond the table's bits for large exponents. Here the quotient
  rounds to the nearest quadrant on the fraction's top bit, so `|r| <= pi/4`;
  the whole mantissa, 24 bits or 53 in two words, is multiplied by the bits of
  `1/(2pi)` at its exponent, from a table of 1312 bits that covers a
  float64's greatest exponent, to 128 bits of the fraction, carried word by
  word, its words moved into place by one selection per bit of their index,
  as a barrel shifter does, where tinygrad compares the index with each
  position of the table (`_take`); and the remainder is the signed fraction,
  summed exactly in two floats from parts that convert exactly and
  multiplied by `2pi/2^64` in two floats (D126). The remainder is within an
  ulp of the exact one in float32 and float64, `6381956970095103 * 2^797`,
  the float64 nearest a multiple of `pi/2`, included.
- **Reason:** (b): rune's `sin` and `cos` reduce by `pi/2` with this
  reduction beyond `2^12` in float32 and `2^22` in float64, where their exact
  Cody-Waite parts stop (RFC 0012), and tolk's own `xsin` uses it beyond its
  switch-over; both must give nx's sine there. Selecting the table's words
  by bits is (d), compile time: every compiled sine or cosine of an
  unbounded angle builds the reduction, and a kernel's compile time grows
  with its nodes. Selected as tinygrad's `_take` does, over this 40-word
  table, the float64 words would take 345 selections and 43 comparisons of
  728 nodes; by bits they take 111 and 6 of 428. No fudge bench row times
  it.
- **Pinned by:** `Tolk.Transcendental › reductions ›
  payne_hanek_near_multiples.golden`, remainders and quadrants of float32 and
  float64 angles near multiples of `pi/2`, from pi to 1400 bits, which the
  old reduction fails; `› payne_hanek_reduction removes quarter turns`; and
  the `payne_hanek_*`, `xsin_*` and `values` goldens, from the equally
  patched tinygrad.

## D34. Withdrawn

A profiled Metal command buffer waited in its stamps retained. rig's Metal
driver commits and times its own command buffers (D146).

## D35. Withdrawn

A selection of a value on no device was that value, where tinygrad raised.
Withdrawn: multi-device is out of scope, see README.

## D36. Withdrawn

A CUDA host program read a function's address from a word of the device. rig's
CUDA driver launches each function (D146).

## D37. Withdrawn

AMD's queues waited on a signal word for equality and wrote it whole. rig's AMD
driver encodes its own waits and releases (D146).

## D38. Withdrawn

A program's code object was loaded by the engine for a batch's commands to
name. rig loads each image (`Rig.Image.load`, D146).

## D39. Withdrawn

A submission wrote at most half of each AMD ring. rig's AMD driver sizes its
own submissions (D146).

## D40. Withdrawn

NV's copy engine signalled the high word of a value too. rig's NV driver
encodes its own releases (D146).

## D41. The memory plan sees a range's calls and the buffers views reach, and holds no copy past its last call

- **tinygrad:** `schedule/memory.py:27-32` (`memory_plan_rewrite` takes each
  entry's buffers from its sources after the first, through `_collect_bufs`,
  which passes buffers, `MSELECT` and `MSTACK` only) and `:36-43`
  (`buf_hold`: a copy's buffer lives `last - first + 1` calls past its last);
  `schedule/__init__.py:295` (`create_linear_with_vars` holds the buffers
  among its call's sources).
- **tolk:** `Memory.viewed`, `Memory.uses` and
  `Memory.memory_plan_rewrite`'s lifetimes (`call_bufs`, the events) and
  lanes (`key`, `per_run`);
  `Schedule.create_linear_with_vars`'s `held_bufs`;
  `test/gen/tinygrad.patch`, which drops `buf_hold` before the goldens are
  recorded.
- **Differs:** an `END` of ranges around calls counts as one entry that takes
  the buffers of each call it runs, and a call takes the buffers its
  arguments view (`SHRINK`, `BITCAST`, `AFTER`) as well as those they are.
  tinygrad takes an `END`'s sources after the first, its ranges, so a buffer
  used inside a range lives only where it is used outside it; it leaves a
  buffer only views reach where it is, and a buffer reached both directly and
  through a view lives only where it is reached directly, so the plan can
  place it over bytes another buffer still needs. A schedule's call holds the
  buffers its sources are or view, where tinygrad's holds those they are: its
  planner never moves a viewed one. No recorded schedule plans a buffer a view
  reaches, so the patch leaves tinygrad's collection as it is. A planned
  buffer takes whole blocks of 256 bytes of its arena, so one placed past the
  arena's other buffers can cost what its own storage did not: rune's
  `compile/scan/checks-48` row's program holds 75 520 bytes, where it held
  75 328 with a buffer of 320 bytes, reached through a view, apart.

  A copy's buffer is free from the call after its last, as a kernel's is,
  where tinygrad holds it as many calls again, lest a later copy overwrite it
  while the first may still run. Copies keep a lane of their own on each
  device, as tinygrad's do, except in a schedule linked per run (D173), whose
  copies and kernels share their device's lane: its arenas live through the
  run, so two lanes add their peaks, and rune's fully sharded steps then hold
  more than RFC 0005's bound.
- **Reason:** (b) for ranges and views: `Rune.scan`'s staged loop (RFC 0012)
  schedules calls inside a range, whose arguments are views that move with
  it, of buffers the calls after it read whole; tinygrad's scheduler hands
  the planner neither. rune's compiled programs over one device store into
  runs of a buffer through views of it (D184); left out of the plan, such a
  buffer would hold its own memory for the whole schedule. (c) for the hold:
  an arena is one rig allocation, and rig orders work that writes an
  allocation after every use of it by any device (`Rig.section-order`, rule
  3, Memory), its device's work by device order (rule 1) and the parts of a
  submission that share memory (rule 2); `tolk.engine` passes no unordered
  ref. A later call that reuses a copy's bytes therefore follows the copy,
  and the hold guards nothing: it only costs memory, as the copy lane of
  tinygrad's `test_deferred_copy_frees_chain` schedule, which needs 512
  bytes with it and 256 without.
- **Pinned by:** the Engine suite (`test/engine/tolk_engine`): `link and
  run › a planned buffer a range writes is not placed over one it leaves` and
  `› a buffer a range writes through a view is not placed over another`, each
  of which fails without its half; the Memory suite
  (`test/schedule/memory`): `memory_plan_rewrite › the buffers a call takes ›
  a buffer a call views is planned` and `› a buffer viewed after another's
  life is not placed over it`, `› rules › a copy's buffer is free from the
  call after its last` and `› per run, a copy's buffer and a call's of one
  device share bytes once their lives are apart`, the recorded `copy_chain`,
  `random_5` and
  `random_11` plans of the equally patched tinygrad, and the law over drawn
  schedules, whose calls view some buffers and whose copies live no longer
  than their last call; the Schedule suite's recorded `copy_view` schedule,
  whose copy reads a view of a buffer its call is given.

## D42. Withdrawn

A batch's command buffers, and its placeholders where a device had no window,
were pinned memory. tolk makes no command buffer, and a program's memory is
rig's (D146).

## D43. Withdrawn

A queue bufferized its own commands, and a region kept its alignment. tolk
encodes no queue (D146).

## D44. An integer cast of a weak expression computes in integers

- **tinygrad:** `uop/weak.py:28-34` (`cast_weak_srcs`), `dtype.py:184-198`
  (`promo_lattice`, `least_upper_dtype`).
- **tolk:** `Uop_weak.cast_weak_srcs`.
- **Differs:** a cast to a committed type over a weak expression commits the
  expression at the least upper type of the cast's type and the committed
  types of the expression and its weak sources. In the lattice, a 64-bit
  unsigned integer and any signed one meet only at a float, so tinygrad
  computes such an expression, `(b * 3 + 1).cast(dtypes.uint64)` for one, in
  float32, and a value past 2^24 rounds. Where that least upper type is a
  float and the cast's type an integer, the expression commits at `Int64`
  here, or at `Uint64` where one of the committed types is: an integer cast
  computes in integers, and converting the 64-bit result to the cast's type is
  C's conversion of the integer. Every other cast commits as tinygrad's does.
- **Reason:** (b): rune's scatter kernel (`Lower_index.walk`) casts the first
  position of a run, `g * len` over a weak range `g`, to `Uint64`; computed in
  float, a first position past 2^24 rounds, and updates land in the wrong
  run.
- **Pinned by:** the Uop_weak suite (`test/uop/uop_weak`): `pm_commit_weak ›
  a 64-bit unsigned cast of a weak expression keeps its value past a float's
  precision` and `laws › pm_commit_weak computes an integer cast in
  integers`. On rune: missing (no test scatters past 2^24 positions).

## D45. Withdrawn

Staging memory was a placeholder of the host. rig stages the copies it cannot
run directly (`Rig.Buffer.copy`, D146).

## D46. Withdrawn

Whether a device's queues reached memory was described, not tried. rig answers
it (`Rig.reaches`), and tolk compiles no copy for it (D146).

## D48. Withdrawn

A kernel's dispatch packet in its arguments was words. rig's AMD driver writes
its dispatch packets (D146).

## D50. Every C-style renderer writes a division; a product by a reciprocal stays one

- **tinygrad:** `renderer/cstyle.py:153-161` (`CStyleLanguage.code_for_op` has
  no `FDIV`) and `:301-304` (Clang's adds it); `codegen/decomp/op.py:120-123` (a
  target that lists `FDIV` gets its reciprocals as divisions, and a product by a
  reciprocal, `a * (1/b)`, as the quotient `a/b`).
- **tolk:** the `FDIV` rule of `Cstyle.base_rewrite`; the `FDIV` rules of
  `Decomp_op.late_patterns`, which turn a reciprocal into `1/x` and nothing
  else.
- **Differs:** Metal, CUDA and HIP write an `FDIV` as `(a/b)`, as Clang does,
  where tinygrad's renderers fail on it. Their tables still leave `FDIV` out,
  so code generation keeps their reciprocals, and every kernel built from
  tinygrad's operations keeps tinygrad's source. On a target that lists
  `FDIV`, a product by a reciprocal stays a product: `a/b` rounds once where
  `a * (1/b)` rounds twice, so the rewrite changed the value nx computes
  eagerly, as at `a = 0x0.000000016db99p-1022`, `b = 0x1.5d24f36473bb3p-998`.
  `gen/tinygrad.patch` removes tinygrad's rule, so the goldens state tolk's
  graphs.
- **Reason:** (b). rune lowers nx's float division, and the power and the
  arc tangent built on it, to `FDIV` (RFC 0012): eager nx divides as IEEE
  does, rounding once, which a product by the reciprocal does not.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `sources › by
  default › <target>_fdiv_<type>`, tinygrad's source once its renderer lists
  `FDIV` as Clang does, for Clang, Metal, CUDA and HIP in each float type
  they have; `division › the operands tell a quotient from a product by
  the reciprocal` and the slow `› Metal divides as IEEE does, rounding once`.
  The `decomp_op` golden `late_division_fdiv`, whose products by a reciprocal
  stay products; rune's `Lower_real › exact › fdiv` and `› recip`, a compiled
  quotient and reciprocal against eager's bits, and `rune jit › classes › jit
  gives eager's bits for exact functions`, whose drawn compositions divide.
  On rune: missing for a product by a reciprocal (the old rune's `Rune.jit ›
  division`, at the value above, and `test_jit_programs` went with its
  rewrite).
  CUDA's and HIP's `/` are correctly rounded by their compilers' defaults,
  which is on the hardware checks of `test/README.md`.

## D51. Withdrawn

An NV launch read its local memory size from a word of the device. rig's NV
driver sizes local memory as it loads a function (D146).

## D52. A selection of -0. stays off a gated load

- **tinygrad:** `uop/symbolic.py:435-439` (`pm_move_where_on_load`), whose
  pattern constant `0` matches `-0.` since `-0. == 0`.
- **tolk:** `Symbolic.pm_move_where_on_load`.
- **Differs:** a selection between a load and `-0.` stays a selection.
  tinygrad moves its condition into the load's gate, and a gated load reads
  `+0.` where its gate fails, so a pad of `-0.`, lowered as a selection of
  `-0.` off the padded load, computes `+0.` on the padding.
- **Reason:** (b). `Nx.pad` fills with the value given, `-0.` included, and
  rune's compiled pad must compute eager nx's bits.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `signed zeros › a pad with -0. fill renders and computes -0.` and `› a pad and a
  selection of signed zeros keep the interpreter's bits`; in rune, `Lower ›
  padded loads give eager's bits`, whose examples fill with `-0.`.

## D54. Storage says what it knows of where it starts within 16 bytes

- **tinygrad:** `uop/ops.py:25-36` (`ParamArg`, which has no such field) and
  `codegen/late/coalesce.py:152` (`memory_coalescing` merges a run of `l`
  elements where `l` divides its element offset, taking every buffer to start
  on a 16-byte boundary). tinygrad folds a view's offset into a kernel's
  index, except for a contiguous slice that it stages without a copy
  (`schedule/__init__.py:195`, `contiguous_mops_to_view`): a later kernel
  reads that view with vector loads that take it to start aligned, as
  `x[1:].contiguous() + 1` loads `float4`s from 4 bytes past a boundary on
  Metal.
- **tolk:** the fields `phase` and `align` of the type `Ops.param_arg`, and
  the function `Ops.param_arg`, which checks them; `Shape.view_start` and
  `Shape.storage_phase`, which `Shape.param_like` gives a parameter;
  `Rangeify.debuf`, which gives them a kernel's parameter;
  `Coalesce.memory_coalescing` (the merge).
- **Differs:** a parameter or buffer carries a congruence for its start: its
  first element lies `phase` bytes past a multiple of `align`, a power of two
  up to 16, the width of the widest vector access off the CPU device, and
  `phase` is a multiple of its element's size or of `align`, whichever is
  less. They default to `0`
  and `16`, tinygrad's assumption, so no graph of tinygrad's changes. Off the
  CPU device, a run of `l` elements merges where `l` divides the element's
  count from that boundary and the run is no wider than `align`, so every
  vector access is
  aligned to its width; an access through a bitcast to elements of a size the
  phase is not a multiple of merges nothing. A parameter made from storage
  keeps the storage's congruence, moved by the bytes a shrink of the storage
  seen whole skips. A shrink by a symbolic start is known only modulo the
  largest power of two its moving bytes are a multiple of: a range's window
  `r · (2^24 + 1)` floats in is known modulo 4 bytes, and is read a float at a
  time, where one `r · 4` floats in stays known modulo 16. A symbolic start
  into a view that reorders or pads the storage is known only to its
  element's size. Any other view of a view that reorders the storage keeps
  its storage's, as tinygrad takes every view to start aligned, and storage on
  a disk, which no vector access reads, keeps the default. A stage of a view
  of a buffer through movements and bitcasts is that view when the schedule
  finds it contiguous (`Schedule.contiguous_mops_to_view`), and storage the
  schedule allocates, on a boundary, otherwise: it is known at phase 0 modulo
  the largest power of two up to the buffer's alignment that the byte of the
  view's first element is a multiple of, which holds of both, and modulo 1
  byte when a size is symbolic. A view whose first or
  last element is padding, or whose ends lie farther apart or closer than a
  run of its size, is never made the view, and its stage keeps the
  boundary. rune's staged loops give each body parameter the alignment and
  phase of what the call passes it. The congruence is part of the graph, so
  of a program's cache key.
- **Reason:** (b). rune's jit runs a program over the storage it is given,
  each buffer at its phase (rune's `Compile.storage`), and mapped weights put
  a tensor at any byte offset of its file: a vector access from an address
  that is not a multiple of its width faults on CUDA, and is
  undefined in Metal, which requires device pointers aligned to their type:
  there it reads the wrong elements once Metal merges adjacent vector loads
  at constant offsets into one wider load, which drops the low bits of the
  address. Single loads at computed indices are not merged, and read
  correctly. A batched range (D30, since withdrawn) binds each trip's window
  at its own address, so a kernel vectorised for a window that starts aligned faulted on
  x86, where Clang emits `movaps`, on the trips that moved it by 4 bytes. A
  rune loop (`Loop.cut`) passes its body a stage of what it reads that does
  not change, and the schedule makes a stage of a contiguous view the view:
  norn's NUTS step on a posterior whose data the model reads from its sixth
  double failed verification, its body compiled for storage on a boundary
  and passed the data 8 bytes past one. Those two failures were on the host,
  in Clang's vector types aligned to their size; on the CPU device they are
  now aligned to their element (`renderer/cstyle.py:312`), so an access there
  is defined from any element's address: the merge there takes every buffer
  to start on a boundary of its widest access, as tinygrad's takes it on 16
  bytes (`Coalesce.memory_coalescing`; the width, D187).
- **Pinned by:** the `Coalesce` suite (`test/codegen/late/coalesce`): `phase › one float past a boundary, eight loads are of one, two, four and
  one`, `› three floats past a boundary, a load of four starts at element 1`,
  `› stores start where loads would`, `› two halves past a boundary, a load of
  four starts at element 2`, `› floats known to start on 4 bytes alone are
  not merged`, `› floats known to start on 8 bytes merge in twos` and `›
  halves three past 8 bytes merge no wider than 8 bytes, from element 1` and
  `› on the CPU device, floats past a boundary merge as from one`, and
  the laws `every vector access is aligned to its width and within the
  alignment, for every phase and alignment` and `coalescing preserves
  the kernel's writes, for every phase and alignment`; the `Ops` suite: `param_arg takes a
  phase that is a multiple of the element size below 16`, `param_arg
  takes an alignment that is a power of two up to 16, and a phase below it`,
  `pp_arg writes a parameter's phase that is not 0 and alignment that
  is not 16`, `equal_arg tells apart parameters that differ in phase` and
  `param_like keeps its storage's phase, moved by a shrink, and
  knows it modulo what a symbolic start moves by`;  the graph format's round trip of `a buffer past
  a 16-byte boundary`; the `Schedule` suite's `disk_view_to_linear.golden`,
  a view 8 bytes into a disk file whose parameter has no phase. In rune,
  `rune jit › jit › an argument off 16 bytes`, which binds an argument 0, 4, 8
  and 12 bytes into a window through a program compiled for its phase; the
  `Schedule`
  suite's `contiguous_mops_to_view › a stage of doubles one past a boundary is
  known to start on 8 bytes`, `› a stage of rows padded apart, starting
  one double past a boundary, is storage of its own on one` and `› a
  stage's alignment and phase hold of the storage it gets, a view or its own`;
  jera's `Split`, `Ode` and `Sde` suites' compiled marches, whose rows are
  padded apart. On rune: missing for a scan and on Metal (the old rune's
  `Scan › compiled › a scan under jit reads a copy of a slice that starts off
  a 16-byte boundary`, `› … reads the bits …`, `› … reads rows …`, the
  `Lower` suite's phase tests and `Compiled › edges › an operand whose buffer
  starts 2 bytes into its memory is read where it is` went with its
  rewrite).

## D55. Metal names a vector after its element's one-word name

- **tinygrad:** `renderer/cstyle.py:200` (`_render_dtype` names a
  vector after its element's name in `type_map` with spaces made
  underscores) and `:382` (`MetalRenderer.type_map` renames only
  `uint` and `bfloat` of `CStyleLanguage.type_map`'s C names).
- **tolk:** `Cstyle.lang`'s field `vector_names`, which
  `Cstyle.render_dtype` reads, and Metal's in `Cstyle.metal_lang`.
- **Differs:** a Metal vector of `signed char`, `unsigned char`,
  `unsigned short` or `unsigned long` is a `char`, `uchar`, `ushort` or
  `ulong` vector, as Metal names them; tinygrad writes `signed_char4`,
  `unsigned_char4`, `unsigned_short4` and `unsigned_long4`, which Metal does
  not have, so such a kernel does not compile. Scalars keep tinygrad's names.
  Metal's vectors have 2, 3 or 4 lanes; none has 8.
- **Reason:** (b). rune's jit on Metal computes `Int8` and `Uint8` arrays
  (rune's `Lower_arith.dtype`, which also stores booleans and narrow floats as
  `Uint8`), whose upcast kernels hold `char4` and `uchar4` values.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `sources › by
  default › metal_vector_<element>_<lanes>`, tinygrad's source with the D55
  names, for each element and 2, 3 and 4 lanes, and `metal_vector_cast_char`;
  and the slow `every GPU kernel compiles with its target's toolchain ›
  metal_vector_*`, which compiles each with MTLCompiler. On rune: missing (the
  old rune's `Compiled › fold of integers` went with its rewrite).

## D56. The logarithm of a negative subnormal is NaN

- **tinygrad:** `codegen/decomp/transcendental.py:250-253` (`xlog2` selects
  NaN where `d < -0.0`, then `-inf` where the reciprocal of `d` is `-inf`,
  which it means for `-0.0`).
- **tolk:** `Transcendental.xlog2`,
  and `test/gen/tinygrad.patch`, which reorders tinygrad's selects the same
  way.
- **Differs:** the reciprocal of a negative number of magnitude below
  `2^-128` in float32 (`2^-1024` in float64, about `2^-16` in float16)
  overflows to `-inf` too, so
  tinygrad's select of `-inf`, after the NaN select of a negative number,
  turns the NaN of such a number's logarithm into `-inf`. The `-inf` select now comes first and the NaN select after it:
  `-0.0` still gives `-inf`, and every negative number gives NaN.
- **Reason:** (b). rune lowers `Nx.log2` to tolk's `Log2` (rune's
  `Lower_real.unary`), which the Clang renderer leaves out
  (`Cstyle.clang_lang`), so every compiled `Nx.log2` on the host runs `xlog2`;
  nx gives NaN for the logarithm of every negative number, subnormals
  included. rune's `Nx.log` reads no `log2`.
- **Pinned by:** the `Transcendental` suite
  (`test/codegen/decomp/transcendental`): `special values › xlog2 of a
  negative number whose reciprocal overflows is NaN, and of -0. is -inf`, and
  the negative float16 rows of `values.golden` of magnitude below
  `2^-16`, from the patched tinygrad; in rune, `Lower_real › transcendental ›
  log2 within 4 ulps at float32, 4 at float64`, whose golden holds negative
  float64 subnormals at NaN.

## D58. Withdrawn

`Spec.program` rejected an elementwise operation of a program on a vector
(the host's renderer, which computes on vectors, was checked against a second
specification), so that a vector left after devectorize failed at lowering.
No consumer got a wrong value or a failure without it: CUDA rejects such a
kernel at NVRTC, Metal and the host compute it. `Spec.program` is
`spec_program` (`uop/spec.py:205`) on every renderer.

## D60. A loop of a call stays in the schedule

- **tinygrad:** `schedule/rangeify.py:160-163` (`pm_no_views`, which strips
  every view of storage), `:321-323` (`split_store`, which makes a kernel of
  an end of a call), `:319` (`pm_renumber_range`, which renumbers the ranges
  of that kernel), `schedule/__init__.py:71-74` (`create_schedule`, which
  schedules the call of an end and drops the end), and `uop/spec.py:254-281`
  (`spec_kernel_graph`, which admits device ranges only, and no end).
- **tolk:** `Rangeify.loop_range`, `Rangeify.pm_no_views` (which keeps a view
  that moves with a loop's range) and `Rangeify.split_store`;
  `Schedule.create_schedule` (`loops`, `argument` and the kept end);
  `Spec.loop`, `Spec.loop_bound` and `Spec.kernel_graph`; and
  `test/gen/tinygrad.patch`, which gives tinygrad's `spec_kernel_graph` the
  same rules.
- **Differs:** an end of a precompiled call over loop ranges
  (`Axis_type.Loop`), `AFTER(buf, END(CALL(fn, args), r))`, is an effect of
  the kernel graph and no kernel. The arguments that move with `r` stay
  views, `r` keeps its number, since `split_store` renumbers only the
  ranges of the kernel it makes, and `create_schedule` keeps the end around
  the call, ordered by the call's reads and writes.
  Once the call's body is scheduled and resolved, the schedule is `LINEAR
  [END(LINEAR [CALL k1; CALL k2; ...], r)]`, the form that the memory plan of
  a range (D41) and `Tolk_engine.run`, a loop of rig's program (D146), run.
  The kernel graph spec admits loop ranges, ends of calls over them, and
  views that move with them, whose bounds are constants, loop ranges, and
  weak integer sums and products of them, which a call may also pass as a
  scalar argument (D140); a range of any other kind outside
  a kernel is still refused, so a range rangeify leaks never runs as a loop.
  tinygrad has no such loop: its `split_store` wraps the end in a kernel of
  its own, and its `create_schedule` drops the end.
- **Reason:** (b). rune's `Rune.scan` compiles its body once and runs it
  once per trip, in place over its carry and over rows of its inputs and
  outputs, as such a loop.
- **Pinned by:** the `Rangeify` suite (`test/schedule/rangeify`):
  `get_kernel_graph › loops of calls › a loop of a precompiled call is
  no kernel` and `› a loop's call reads views that move with its range`; the
  `Schedule` suite (`test/schedule/schedule`): `create_schedule › rules ›
  an end of a call over a loop schedules the call in its loop`, `› a loop runs
  after the call that writes what it reads`, `› a loop runs before
  the call that overwrites what it reads`, and
  `create_linear_with_vars › loops of calls › a loop of a precompiled
  call is a loop of its body's calls` and `› a loop inside a loop's body
  keeps its own end`; and the `Tolk_engine` suite
  (`test/engine/tolk_engine`): `link and run › a scan runs its body
  once per trip, carrying in place`, against the unrolled loop on the
  host, and `› a scan's merged calls carry in place as its calls apart do`
  (D199), on the host and each GPU present. The `Spec` suite (`test/uop/spec`): `kernel_graph › loops of calls ›
  accepts a loop of a call over a loop range`, `› refuses a loop of a call
  over a range of any other kind` and `› refuses a weak sum of a weak integer
  variable`; its
  `verdicts.golden` records the patched spec on ends and shrinks.

## D61. Withdrawn

A stack took the per-shard sub-view of a whole source beside sharded ones.
Withdrawn: multi-device is out of scope, see README.

## D62. A move of an emulated float keeps its bits

- **tinygrad:** `codegen/decomp/dtype.py:167-191` (`pm_float_decomp`, which
  widens every load of an emulated float to the emulating float and narrows
  every store back).
- **tolk:** `Decomp_dtype.moved` and the store rule of
  `Decomp_dtype.pm_float_decomp` that stores it, and
  `test/gen/tinygrad.patch`, which gives tinygrad the same rule.
- **Differs:** a store of a value that moves stored bits without arithmetic
  (a load, a constant the float holds exactly, a selection between such
  values, and stacks and lanes of them) stores those bits, as the storage's
  unsigned integers. tinygrad widens and narrows each such value, which keeps
  its value but makes every NaN its format's quiet one: a copy of e5m2's
  `0x7d`, half's `0x7c01` or bfloat16's `0x7f81` stores `0x7f`, `0x7e00` or
  `0x7fc0`. Arithmetic still converts, and gives a NaN as the NaN rule allows
  (D9).
- **Reason:** (b). Under the NaN rule a move keeps a NaN's bits, and nx's do:
  an eager copy, movement or `where` keeps a signalling NaN, so a compiled
  function computes what it computes eagerly only if the emulated moves keep
  the bits too. rune converts `float16` and `bfloat16` by tolk's casts, and
  stores the 8-bit floats and `float4_e2m1fn` as `uint8`, which it decodes
  and encodes by integer formulas on their bits (`Lower_arith.of_byte`,
  `Lower_arith.to_byte`), so tolk no longer emulates an 8-bit float for
  it. What remains is a move of `bfloat16` or `float16` storage on a target
  that emulates it: `bfloat16` on a riscv64 host, on an Apple GPU before
  family 6 and on CUDA before sm_80.
  rune stores both in their own types (`Lower_arith.dtype`), so a compiled
  copy of one is such a move.
- **Pinned by:** the `Decomp_dtype` suite (`test/codegen/decomp/decomp_dtype`): `NaNs › an
  emulated copy keeps every NaN code, a signalling one's included`
  and `› an emulated selection keeps every NaN code, a signalling one's
  included`, on every emulated float, and the goldens of the
  `where`, `flip`, `gather` and `pad` kernels, from the patched tinygrad.
  On rune: missing (the old rune's `Compiled › host › 8-bit floats › a copy
  of every 8-bit float code keeps its bits` and `› a selection of every
  8-bit float code keeps its bits` went with its rewrite).


## D63. Programs and schedules are kept on disk

- **tinygrad:** `codegen/__init__.py:510-518` (`to_program_config`,
  `to_program_key` and `to_program_cache`, a dictionary of the process),
  `schedule/__init__.py:124-134` (`lower_sink_to_linear`'s schedule cache,
  on disk only from `SCACHE=2`), `helpers.py:288` (`SCACHE`, `1` by
  default), and `codegen/opt/postrange.py:48-49` (a kernel's name, coloured).
- **tolk:** `Codegen.program_key`, `Codegen.kept` and
  `Codegen.made_program`; `Schedule.in_order`, `Schedule.schedule_key` and
  `Schedule.lower_sink_to_linear`; `Graph.cached`;
  `Postrange.Scheduler.get_optimized_ast`'s name; `Setting.scache` (`2` by
  default), `Setting.shaping` and `Setting.cc`; and
  `lib/dune`'s rule for `source_digest.ml`, written by
  `tools/source_digest.ml`.
- **Differs:** the program `to_program` makes of a kernel, up to its
  source, and by default the schedule `lower_sink_to_linear` makes of a
  function, are also put in the disk cache (tables `to_program` and
  `schedule_cache`) as their graphs' text (`Graph`), and a later process
  reads them back instead of making them. A program read back takes its
  binary from its compiler's table, under its source, so each binary is
  stored once. tinygrad keeps programs for its process only, and schedules on disk
  only when asked. Each key is the kernel or function, a function with its
  ranges and its storage replaced by placeholders numbered in order (whoever
  makes a loop numbers its range, and whoever makes a staged loop's carry or
  inlines call-local storage numbers it, from a counter whose value depends
  on what the process did before); the schedule is kept with the same
  placeholders and given back each function's own ranges and storage, for programs
  the renderer and its target, the value of every setting and environment
  variable that shapes what compilation makes, and the digest of the
  library's sources, which dune computes when it builds the library: an
  entry is a function of the code that made it. Every variable is a setting
  declared with its reach (D114), among them those tinygrad reads with
  `getenv` where it compiles (`ALLOW_HALF8`, `BEAM_ESTIMATE`,
  `BEAM_LOCAL_MAX`, `BEAM_MIN_PROGRESS`, `BEAM_PADTO`, `BEAM_UOPS_MAX`,
  `BEAM_UPCAST_MAX`, `DMC`, `EXPAND_SSA`, `HCQ_NUM_SDMA`, `JITBEAM`, and the
  `MV` and `REDUCEOP_SPLIT` variables), and the keys take every one that reaches
  output (`Setting.shaping`), so one declared later, by tolk or by its
  caller, is keyed without being listed; a setting of the process alone,
  such as `DEBUG`, `PARALLEL` or `CC`, is not. One key serves memory and
  disk, so a setting changed within the process by `context` makes the
  result again, where tinygrad's in-memory keys leave out `TUPLE_ORDER` and,
  for schedules, every setting. The compile cache's table names the compiler
  `CC` runs, where tinygrad's names Clang alone. An entry that does not read
  as a program or a schedule is a miss, and is replaced. The program of a
  kernel that asks for a beam search is not kept, since it is what the
  search found. `SCACHE=1` keeps schedules in memory only, as tinygrad's
  default does. A kernel's name holds no colour, where tinygrad colours it
  unless `NO_COLOR` is set: a kept program does not depend on the display of
  the process that made it. A program read back shows its source and
  instructions at `DEBUG` 4 and 7, as a compiled one does.
  `lower_and_compile` reads the programs the disk keeps on the calling
  domain (`Codegen.read`), and hands only the others to `Worker.map`, where
  a tinygrad worker reads its compile cache: a read took 0.4 to 0.5 ms on
  fudge (eight schools, gpt-oss's 12-token prefill), less than a domain's
  start and the collections it joins.
- **Reason:** (d): a later process's first compiled call of a model. With
  binaries alone on disk, a warm process schedules and lowers every kernel
  again. On fudge at `PARALLEL=1`, rune's compile bench rows
  `warm/model/schools-start` and `warm/linalg/eigh-8x8`, a fresh process's
  first call over the disk cache an earlier one filled, take 101 and 98 ms,
  and 70 and 69 ms, against 263 and 274 ms, and 221 and 200 ms with binaries
  alone kept and schedules in memory (`SCACHE=1`) (medians of 3, two
  interleaved runs each). In one timed run on fudge, read on 14 domains, the
  40 programs of `warm/model/schools-start` took 161 ms of its 328 ms first
  call, and 14 ms of 96 ms on one. Its other fudge rows: the tolk bench's
  `*/warm` rows, which time reading a schedule and its programs back against
  the stages they replace, and rune's `WarmStart/chain-100`.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `programs are
  kept › a program is made again under another value of ›` each setting,
  `programs are kept on disk › a program made by one process is read back by
  the next`, `› read reads back a program and its binary, and is none
  without`, `› a program made under one setting is not read back under another
  ›` each setting, `› a damaged entry is made anew, and replaced ›
  truncated` and `› holding no program`, `› an entry of another build of the
  library is not read back`, and `› processes making one program at once all
  get it`; and the `Schedule` suite (`test/schedule/schedule`):
  `create_linear_with_vars › cache › a body scheduled under one setting
  misses under another ›` each setting, a setting its caller declares
  among them, `create_linear_with_vars ›
  schedules are kept on disk ›` the same as programs' for schedules, and `›
  with SCACHE at 1, nothing is kept on disk`, and `create_linear_with_vars ›
  loops of calls › a loop whose range has another number is the same body`
  and `› a body that hits under other numbers keeps its own loops`, and
  `› cache › storage numbered in another counter state is the same body`;
  the `Helpers` suite's
  `shaping ›` tests; and the `Compiler_cpu` suite's `without Clang › an
  object cached by the default compiler is not served under CC`; the
  `Codegen` suite's `programs are kept › read waits for no program being
  made`, and the `Realize` suite's `lower_and_compile in parallel › reads
  the programs the disk keeps without spawning a domain`.

## D64. A cast to a narrow float through a float32 rounds once

- **tinygrad:** `renderer/cstyle.py:100-101` (`create_non_native_float_pats`,
  which casts a source of any type but float32 to a float32, then to the
  narrow float).
- **tolk:** the cast rule of `Cstyle.create_non_native_float_pats`, with
  `Decomp_dtype.narrow` (D9's).
- **Differs:** a renderer without arithmetic on a narrow float, Clang's on
  bfloat16 and HIP's on bfloat16 and the 8-bit floats, casts to it through a
  float32. A source more precise than a float32 (a float64, or an integer of
  32 or 64 bits) reaches the float32 rounded to odd, as an emulated cast's
  does (D9), so that the cast rounds once, from the exact value, as the folder
  (`Dtype.truncate`) does. tinygrad rounds to the float32 to nearest, then
  again: on the host, the bfloat16 of the double `1 + 2^-8 + 2^-52` is `1.0`
  there, where its correctly rounded value is `1.0078125`, and of the int64
  `2^40 + 2^32 + 1` is `2^40` where it is `2^40 + 2^33`. No golden holds such
  a cast.
- **Reason:** (e). A cast is IEEE's conversion, one rounding, in the IR's own
  types; the second rounding gives the wrong neighbour where the float32 lands
  on a tie of the narrow float. rune hands tolk only a float32 to cast to a
  narrow float, rounding a more precise source to odd itself
  (`Lower_arith.store_real`, `Lower_arith.float_odd`), so no raven call site
  reaches the rule.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `casts to
  bfloat16 › an integer or a double rounds once on the host`, for int64,
  int32, uint32, uint64 and float64 sources, among them doubles just past a
  tie, one of them subnormal. On rune, which rounds itself: `Lower_arith ›
  casts give eager's values`, whose float64 operands include the doubles one
  unit each side of the bfloat16 tie `1 + 2^-8`.

## D65. Withdrawn

An operation on an emulated narrow float rounded its result to the float in
the kernel, so that a chain of operations rounded after each, as nx does. rune
computes each narrow-float operation in float32 and encodes its result itself
(`Lower_arith.narrowed`), so no operation on an emulated float reaches tolk
from rune. The operation rule of `Decomp_dtype.pm_float_decomp` computes in
the emulating float and rounds at the store, as `codegen/decomp/dtype.py:185-186`
does; a cast to the emulated float rounds, as `:183-184` does.

## D66. Withdrawn

A counted run's log entry was its kernel descriptor's address and its times,
for the counters an AMD batch read. tolk encodes no counter yet (D146).

## D67. Withdrawn

A long range ran as chunks of its trips, so that a batch held a bounded number
of commands. A range is a loop of rig's program, which holds one trip's work
(D146).

## D68. Withdrawn

The scheduler recognised a one-hot sum, rune's lowering of a gather, and
inlined it as a load where it was read. Inlining values that read state
written elsewhere in the graph left buffers with two definitions, and it
was withdrawn; an explicit gather, which rune emits as an indexed load and
tolk lowers as one, replaces it.

## D69. Withdrawn

A store whose destination moved through a pad wrote only within the pad's
source, its validity carried on the index into the storage, so that rune
could drop a row it wrote out of range into a padding row. Rune now writes
those rows with one store through a gather whose index is Invalid where a
row is dropped, which tinygrad schedules as a gated store, and nothing
stores through a pad.

## D70. Withdrawn

A batch reached host memory through nx.device's staging. rig's borrows and
staged copies replace it (D146).

## D71. Withdrawn

A launch stated a constant bank's size in 16-byte units. rig's NV driver writes
its launch descriptors (D146).

## D72. Withdrawn

Only a view of storage had a contiguous view, since tinygrad's rewrite of a
constant's index marked the constant and `symbolic` rebuilt it unmarked, without
end. At `uop/ops.py:895-914` the rewrite marks nothing and finds a
view of any value, a constant or a computed one included; `Prepare.contiguous_view`
does the same.

## D73. A gather is an INDEX of a tensor by a value with axes

- **tinygrad:** `uop/ops.py:335-338` (INDEX's shape: its index sources'
  shapes, then the rest of the source's), which no tinygrad pass puts in a
  tensor graph; `schedule/indexing.py:28-43` (`pm_generate_realize_map` and
  its rules), `:68` (the sources an INDEX's ranges reach),
  `:129` and `:137` (`pm_apply_rangeify`) and `:222` (`run_rangeify`'s
  consumer ranges); `schedule/prepare.py:65` (`_mop_index`), `:75`
  (`pm_mops`) and `:257-259` (`prepare_rangeify`).
- **tolk:** `Indexing.storage`, `Indexing.is_gather`,
  `Indexing.realize_gathered`, `Indexing.data_srcs`, `Indexing.convert_gather`
  and its rule in `Indexing.pm_apply_rangeify`, and
  `Indexing.assign_ranges`'s consumer ranges of a gather;
  `Prepare.move_index`, `Prepare.mops` and `Prepare.pm_tensor_mops`; the
  same rules in
  `test/gen/tinygrad.patch`, which the goldens are recorded with.
- **Differs:** an `INDEX` whose one index source has axes, `INDEX(x, L)`,
  reads `x` at the row each element of `L` holds, and its shape is `L`'s
  followed by the rest of `x`'s. The scheduler takes it as a load:
  - `x` is stored whole unless it is storage, so that a view of storage is
    read through its movements;
  - `L` takes the gather's leading ranges, one per axis of `L`, by its
    position among the gather's sources, and the gather reads `x` at `L`'s
    element followed by its trailing ranges;
  - an index that loads from a storage state, as `L` does once ranged, moves
    through a movement with each load held as a parameter of its bounds, so
    that the index's simplification does not rebuild the stores the state is
    ordered after, which gave the storage a second definition;
  - in the tensor graph, prepare's movement rules leave the gather whole,
    since `L`'s shape would multiply the movement's index; `pm_mops`, which
    kernels use with vector indices, is tinygrad's;
  - refused: a gather by several indices (rune emits one).
  The index is in range: rune clamps it and selects `0` where it was not.
- **Reason:** (b): rune's gather lowering (`Lower_index`) emits this INDEX
  for `Nx.take` and `Nx.take_along_axis`. Without it rune would lower them to
  a one-hot sum over the whole axis, which costs a reduction per element and
  made gpt-oss's decode step schedule embedding and cache reads as their own
  kernels: that cost is (d), with no fudge bench row. D68, which recognised
  the sum in the scheduler, was withdrawn. The sharded clauses are
  withdrawn: multi-device is out of scope, see README.
- **Pinned by:** the Rangeify suite's `gathers` programs (`index_rows`,
  `index_rows_computed`, `index_read_twice`, `index_under_reduce`,
  `index_broadcast`, `index_view_source`, `index_zip`, `index_of_index`,
  `index_zero_fill`, `index_assign_self`) and their kernel counts,
  `gather_kernel_counts`; the Prepare suite's `gather_of_self`,
  `gather_by_self` and `gather_of_reshape`, and `pm_mops › rules › an index
  that loads from a storage state moves with the state as it is`; rune's
  `Lower_index` suite: `gathers and scatters › gathers give eager's bits`,
  compiled on the host.

## D74. Float arithmetic has bounds, and a bounded sine takes the short reduction

- **tinygrad:** `uop/ops.py:1053-1119` (`UOp._min_max`), which bounds binary
  operations on integers only (`:1054`), has no case for `TRUNC` or `NEG`, and
  bounds a `WHERE` by both its branches whatever selects them (`:1093`),
  and gives a value it cannot bound, or a cast it cannot, its type's
  `min` and `max` (`:1119`), finite in the 8-bit formats without infinities
  (`dtype.py:74-78`);
  `codegen/decomp/transcendental.py:170-191` (`xsin`), which builds the
  Payne-Hanek reduction unless its caller passes `fast`, and chooses between
  the reductions with `x_abs < switch_over` (`:187`), a comparison that the
  decompositions' rewrite (`symbolic_simple`) does not fold.
- **tolk:** `Ops.unbounded`, `Ops.compute_min_max` (its `Where`, `Trunc`
  and `Neg` cases), `Ops.selected`, `Ops.float_bounds` and
  `Ops.cast_bounds`; `Transcendental.xsin`; `test/gen/tinygrad.patch`,
  which gives tinygrad the same bounds and the same `xsin` before the goldens
  are generated.
- **Differs:** a float sum, difference or product of operands with finite
  bounds has the bounds of its corners, computed in double and widened by a
  relative `2^-m` and the smallest normal of its type, `m` its mantissa's bits,
  so that they hold the result rounded at its type, or flushed to zero; an
  operand's end within the subnormals counts as 0, for a target that flushes
  the operand. A
  result that may overflow its type, and one of a weak float, has the type's
  bounds. A truncation's bounds are its operand's, truncated, and a float
  negation's are its operand's, negated and swapped. A float selection by a
  comparison `a < b` narrows the branch it selects where the comparison
  holds, so where neither operand is NaN: `a` to below `b`'s greatest value,
  and `b` to above `a`'s least. Finite float bounds state that a value is
  not NaN, which every float type holds: a float that nothing bounds has
  infinite bounds, in `float8_e4m3` and the `fnuz` formats too, and a cast
  into a float keeps its source's rounded bounds only where they are values
  of its type, so a value cast from an unbounded float stays unbounded. A
  cast into a weak float rounds its bounds to doubles, as it rounds its
  value: tinygrad's bounds of a `uint64` cast to one held `2^64 - 1`, past
  the value `2^64`. With
  the type's finite greatest value instead, `a < inf` of an 8-bit NaN folded
  true and a compiled sine, cosine or tangent of it gave -1, 0 and -452.
  `xsin` of an angle whose
  bounds lie strictly within `switch_over` is its `fast` form, the Cody-Waite
  reduction alone.
- **Reason:** (d): the sine or cosine of a bounded value on the CPU, such as
  a Gaussian draw (`Nx.Rng.normal`), whose angle is `2 pi u` for a uniform `u`
  in `[0, 1)` fused into the draw. rune's trigonometric reduction
  (`Lower_real.reduce`) reads tolk's bounds (`Ops.vmin`, `Ops.vmax`) and
  builds Cody-Waite's reduction alone for an angle bounded below
  `Lower_real.trig_switch`, and holds the remainder to `[-1, 1]`
  (`Lower_real.reduced`), since the cancellation in `a - q pi/2` hides that
  bound from interval arithmetic. The host's renderer has no sine, so `xsin`
  reduces what remains. Without this entry every compiled normal draw built
  the Payne-Hanek reduction for angles below `2 pi`. On fudge, with float
  operations unbounded and `xsin` choosing at run time as tinygrad's do
  (two interleaved runs each, with the entry then without), rune's
  `Jit/normal-float32-1M`, `2^20` draws, takes 2.8 and 2.6 ms against 4.3
  and 4.1 ms (medians of 11), and the compile bench's first calls of a draw
  of 1024, `rng/normal-f32` and `rng/normal-f64`, take 0.75 and 0.30 s, and
  1.15 and 0.44 s, against 0.78 and 0.59 s, and 2.10 and 0.90 s (medians of
  3). `Jit/sin-bounded-float32-64K`, a sine of angles a `tanh` bounds, shows
  no difference beyond its ranges. The 8-bit half of the
  bounds (a float without infinities is unbounded) has no rune consumer
  since rune stores the 8-bit floats as `uint8` (D62).
- **Pinned by:** the Ops suite (`test/uop/ops`): `bounds › bounds hold every
  value a float operation rounds to`, sums, differences, products and
  selections, `› a float operation of bounded operands is bounded`,
  `› a float operation of an operand within the subnormals bounds a flushed
  operand too`, `› a float operation of an unbounded operand, or that can
  overflow, has its type's bounds`, `› a float selection by a
  comparison narrows what it selects`, `› a float truncation and
  negation map their operand's bounds`, `› a value of a float type
  without infinities, which may be NaN, has no finite bounds` and `› bounds
  hold every value a cast takes`; the Transcendental suite: `graphs › a sine of an
  angle bounded below the switch-over is its fast form`; and rune's
  `Lower_real` suite: `bounded arguments › cos of an argument bounded below
  the switch-over` and `› tan of an argument bounded below the switch-over`.
  On rune: missing (the old rune's `lower_arith › long reductions › a normal
  draw takes none`, `› the sine of an angle read from a buffer takes its own
  only` and `› the cosine …`, which counted the long reductions built, and
  `Compiled › 8-bit floats › the sine, cosine and tangent of an 8-bit float
  NaN are NaN`, went with its rewrite).

## D75. A selection by a constant condition

- **tinygrad:** `uop/symbolic.py:209`, which folds a `WHERE` whose condition is
  a `CONST` (`UPat.cvar("gate")`), and `:240-245` (`fold_where_closure`),
  which substitutes `True` for a selection's condition in its true branch and
  `False` in its false branch, whatever the condition is.
- **tolk:** `Shape.broadcast_const`, its rule in `Shape.symbolic_simple`,
  and `Shape.fold_where_closure`; `test/gen/tinygrad.patch`, which gives
  tinygrad the same.
- **Differs:** a selection whose condition is a constant through movements
  that keep each element's value (reshape, expand, permute, shrink) is
  its branch, as one by a `CONST` is. A padded constant is not folded, since
  its padding holds `False`. `fold_where_closure` leaves a condition that is
  a constant, broadcast, padded or not, in the branches: a broadcast constant
  is one node for every use of that constant, so substituting it is no
  assumption about the selection. In `where F (where F x (where F y x)) y`,
  with `F` a broadcast `False`, the outer selection makes the inner `F`s
  `True`, the inner selection makes its false branch's `True` back into
  `False`, and the rewrite does not terminate; tinygrad raises "infinite loop
  in graph_rewrite" on that graph.
- **Reason:** (b): D74's float bounds fold the comparisons of rune's sine and
  cosine (`Lower_real.reduce`) to broadcast constants in the tensor graph.
  Two symbolic passes over such graphs did not terminate:
  `Prepare.contiguous_view`'s, on the old rune's staged scan reading a draw on
  Metal, and the multi-device rewrite's, since out of scope (see README), on
  a sharded model's rope. Folding the selection at its constant also drops
  the branch never taken, such as the long reduction of a bounded angle.
  Unverified that current rune builds the first graph: rune still stages
  scans (`Loop`), but the Metal test of a scan reading a draw went with the
  old rune.
- **Pinned by:** the Symbolic suite: `symbolic_simple › selections › a
  selection by a broadcast constant is the branch it picks` and `› a selection
  by a padded constant is no constant's`, and `symbolic › selections › a
  padded constant condition is not folded in the branches`. On rune: missing
  (the old rune's `Rune_internals.Jit › metal › staged scans › stage a scan
  whose step reads a draw and another scan's result made before it` went with
  its rewrite).

## D76. Withdrawn

A stack of `Invalid` lanes stayed a stack, since tinygrad folded it to one
`Invalid` and the lanes' reshape, select and reduce lost their width.
tinygrad now folds it to an `Invalid` expanded to the stack's shape
(`uop/symbolic.py:89`, `Shape.pm_data_invalid`), which keeps the width, and
devectorizes no expanded scalar into a stack. The kernels that pinned the
entry, whose upcast lanes all read an `Invalid` index, compile and sum to zero
with tinygrad's rule (`Tolk.Codegen › lanes all Invalid`, `lanes of an
unrolled reduce`).

## D77. A broadcast value that runs a transcendental stays stored

- **tinygrad:** `schedule/indexing.py:262-268` (`run_rangeify`, which
  stores an elementwise value or reduction whose ranges a broadcast ends),
  `:58` (`BufferizeOpts.removable`) and `:88` (its value), and
  `schedule/rangeify.py:23` (`cleanup_dead_axes`) and `:48-90`
  (`remove_bufferize`, which inlines that store again when its value reads
  at most three buffers and no reduction reads a buffer).
- **tolk:** `Indexing.assign_ranges` (which records the values a broadcast
  stores) and `Indexing.bufferize_and_index` (which marks their stages
  `Broadcast`); `Ops.keep`, in place of `removable`;
  `Rangeify.remove_bufferize`, which keeps such a stage;
  `test/gen/tinygrad.patch`, which gives tinygrad the
  same rule through a field `broadcast` of `BufferizeOpts`.
- **Differs:** where a broadcast ends ranges below a value, its store is
  marked, and the cost check keeps it if computing it runs `EXP2`, `LOG2`,
  `SIN` or `POW`: the value is computed once per element and read where it
  is broadcast. The cost check sees the value with its producers already
  stored or inlined, so it counts exactly what inlining would compute: a
  value that reads a stored exponential computes none, and is inlined as
  before. A marked store still loses the axes its value does not vary
  along. tinygrad's cost check counts buffers and
  reductions only, so it inlines such a value into its consumer, which
  computes it again for every element of the ranges it does not vary along:
  a SwiGLU's sigmoid once per output column of the down projection, a
  softmax's exponentials once per column of the product with the values, a
  RoPE's sines and cosines once per head. A value read once per element,
  as a rotated query reads its rows of a stored sine table, is not marked.
- **Reason:** (d): gpt-oss-20b (`kaun/examples/06-gpt-oss`). On the host,
  where the transcendental functions are polynomials, recomputing them
  dominates. On an M-series Mac under load
  (provisional), one MoE block of gpt-oss-20b took 182 ms a token and
  64 ms with the rule, and 1415 ms and 555 ms for 8 tokens; causal
  attention with sinks at gpt-oss's shapes took 2726 ms and 614 ms at 512
  tokens, and 39.0 s and 10.9 s at 2048. On Metal the MoE block took
  18.4 ms and 17.1 ms a token, and attention 23.5 ms and 16.8 ms at 512
  tokens, 187 ms and 183 ms at 2048. On CUDA, gpt-oss-20b's prefill took
  9.4 s and 3.1 s warm, and its attention-score kernel went from 29k nodes
  to 479; symo's tutorial step on kimchi took 962 ms and 161 ms. The cost is
  memory: attention keeps its exponentials, one more buffer the size of the
  scores (tokens by tokens by heads) while the call runs, at every length;
  at 2048 tokens 1.10 GB and 2.17 GB were allocated. On fudge's RTX 5000 Ada
  (gpt-oss's bench, `kaun/examples/06-gpt-oss/bench`, medians of 11, two
  interleaved runs each, with the rule then without), `GptOss/prefill-512`
  takes 185 ms against 228 ms in both runs, and
  `GptOss/decode-step-cache-1024` 10.2 ms against 14.1 and 14.2 ms.
- **Pinned by:** the Rangeify suite's `swiglu_down`, `attention`,
  `exp_dead_axis` (a kept exponential of a row expanded to a matrix is
  stored as the row), `exp_cheap_consumer` (a value that reads a kept
  exponential is inlined) and `rope_decode` (a decode step's rotated query
  reads its gathered rows of a cosine table and is not stored) rows of
  `kernel_counts.golden` and their kernel goldens, from the equally patched
  tinygrad.

## D79. A chain of one operation keeps its right operand's grouping

- **tinygrad:** `renderer/cstyle.py:66` (`base_rewrite`'s ALU rule), which
  drops the parentheses of any operand that is the same associative operation
  (`ADD`, `MUL`, `XOR`, `OR`, `AND`), on either side.
- **tolk:** `Cstyle.base_rewrite`'s ALU rule;
  `test/gen/tinygrad.patch`, which gives tinygrad the same.
- **Differs:** only the first operand drops them. C groups `a*b*c` as
  `(a*b)*c`, so writing `a*(b*c)` without parentheses regroups it, and a float
  sum or product rounds and overflows by its grouping. An upcast reduction
  `acc + (v0+v1+v2+v3)` was written `acc+v0+v1+v2+v3` and added in another
  order than the kernel's; `4 * (2^126 * 0.25)` was written `4*2^126*0.25`,
  which overflows.
- **Reason:** (b): nx.quant's product, `Nx_quant.apply`, decodes each weight
  in the kernel that multiplies it, `x * (value * scale)`, and gave infinity
  where the eager product's infinities of both signs give NaN.
- **Pinned by:** the Cstyle suite: `grouping › Clang computes a product of a
  product as it is grouped`; and the rendered sources of the `codegen`,
  `renderer/cstyle` and `engine/jit` goldens, from the
  equally patched tinygrad. On rune: missing (the old rune's `quant › values ›
  compiled, a product is eager's › the largest scales` went with its
  rewrite).

## D78. Withdrawn

The CUDA device's kernels compiled to a cubin, so that a process's first
load would not wait for the driver to translate PTX. On fudge (RTX 5000 Ada),
rune's compile bench row `cuda/sin-nested-8`, a fresh process's first call
of eight nested float64 sines over a driver cache of its own, took 455 and
517 ms with cubins and 390 and 340 ms with PTX (medians of 3, interleaved
runs), and `warm/cuda/sin-nested-8`, the same call over a disk cache an
earlier process filled, 340 and 329 ms against 218 and 178 ms: no gain from
cubins. As tinygrad's `renderer/cstyle.py:426`,
`Cstyle.cuda` compiles PTX for the CUDA device and a cubin for the NV device.

## D80. A host program runs its output loop in blocks on the host's cores

- **tinygrad:** `runtime/ops_cpu.py:47-57` (`CPUProgram.__call__`),
  which calls a kernel once per launch on the calling thread;
  `codegen/__init__.py:352` (the pipeline after `pm_cast_const`),
  `:34-39,363-365` (`do_number_param` and its pass) and `:446`
  (`do_estimates`); `uop/ops.py:1295` (`KernelInfo`) and `:1322`
  (`ProgramInfo.from_sink`); `engine/realize.py:17`
  (`get_call_kernel_args`) and `:157` (`exec_kernel`).
- **tolk:** `Codegen.split_blocks`, its place in
  `Codegen.full_rewrite_to_sink`, `Codegen.pm_number_params` and
  `Codegen.whole_loop`; `Ops.kernel_info`'s
  `split`; `Shape.program_info_of_sink`; `Realize.get_call_kernel_args`;
  `Tolk_engine.block_ops`, `Tolk_engine.blocks` and `Tolk_engine.split_grid`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same split, estimates,
  launch dimensions and variable values, and runs a split program as one
  block; `test/gen/renderer/renderer.py`, whose estimate tables count a split
  kernel's whole loop.
- **Differs:** for a CPU target, once the kernel is final (after
  `pm_cast_const`), the largest range of axis type `WEAK` whose end reads no
  range, of at least two iterations, and that the address of every store to
  global memory reads, runs `block_lo + r` for `r` below
  `block_hi - block_lo`. `block_lo` and `block_hi` are new variables of the
  range's type, bounded by `[0, n]` for a loop of `n` iterations, in the
  first two slots past those the kernel's parameters took, a loop's trip
  among them (D140), and named afresh
  where the kernel has a variable of either name. The variables numbered
  after them take the slots past every slot a parameter takes, the bounds'
  included, one slot per name. The kernel's `KernelInfo`
  records `split = (n, lo, hi)`, the iterations and the two slots, by which
  each launch finds the bounds; `n` is its program's first global size, and
  its estimates count the whole loop, exactly when they read no other
  variable: counted in the loop's 32-bit type, a count past 2^31 would wrap. A host launch cuts `[0, n)` into `min(n, ops / 2^18)` blocks, at least
  one, each a group of rig's host of as many threads as iterations, which
  its pool runs on the performance cores, and the entry (D28) runs the
  iterations its threads number; tinygrad's own launch (`whole_loop` in
  `engine/realize.py`) runs one block, the whole loop.
- **Reason:** (d): sofo's and symo's training steps on the CPU (outside the
  repository), which ran each kernel on one core. Their hottest kernels,
  split by hand into 32 blocks on kimchi's 8 E-cores (`taskset -c 6-13`, best
  of 30), took 6.17 ms and 0.77 ms (r_256_8_4_4_32_4), 3.38 and 0.42
  (r_32_32_4_4_128_16), 0.28 and 0.04 (r_256_3_3_100_4) and 17.8 and 2.83
  (r_128_64_3_4_400_100_4). On fudge, against one block (medians of 11, two
  interleaved runs each), the tolk bench's
  `lorenz/r_128_16_3_4_4_3_3_400_400/heuristic` takes 0.79 and 0.63 ms
  against 5.85 and 6.04 ms, and rune's `Jit/sin-float32-64K` 82 and 74 us
  against 360 and 356 us, `Jit/normal-float32-1M` 2.8 and 2.6 ms against
  16.9 ms. tinygrad deleted its CPU threading (`OptOps.THREAD`)
  at 4197f7423, before the pin.
- **Pinned by:** the Codegen suite: `host programs in blocks › a host program
  splits the loop each store's address reads`, `› a split loop's estimates
  count its whole loop past 2^31`, `› stores of separate loops
  split none`, `› a store of no loop splits none`, `› a reduction's loop is
  never split`, `› a serial loop is never split`, `› a loop shrunk by its
  guard splits at its shrunk end, unguarded`, `› a split loop of 2^25 iterations keeps 32-bit indices` and `› a split's bounds and a variable take slots of their own past buffers that skip one`; the
  Tolk_engine suite: `host programs in blocks › a program writes the same bits in 1, 3 and 32
  blocks` and `› a run of a large kernel computes its sums`, `a variable
  named as a block's bounds is a variable like any other`, and `a kernel
  whose estimates fit no int › splits and runs`.

## D81. Withdrawn

The matrix-vector layout read its operands through conversions and decoding;
merged into D89.

## D82. Withdrawn

A sharded value was not stored into a destination replicated on several
devices. Withdrawn: multi-device is out of scope, see README.

## D83. Withdrawn

A sharded value joined on several devices kept its bits, by a bitwise or of the
devices' values in place of a sum. Withdrawn: multi-device is out of scope, see
README.

## D84. Withdrawn

A dispatch's thread-trace marker numbered its queue's dispatches. tolk encodes
no thread trace yet (D146).

## D85. Withdrawn

A GFX11 trace set `TTRACE_EXEC` by name. tolk encodes no thread trace yet
(D146).

## D86. Withdrawn

`Spec.program` rejected an index of a vector value at a lane that is not a
constant. No pass makes one, and NVRTC rejects a hand-built kernel that does,
so no consumer depended on the check. `Spec.program` accepts it, as
`shared_spec` (`uop/spec.py:83`) does.

## D87. Withdrawn

A store through a gather of a sharded value stored each shard's rows.
Withdrawn: multi-device is out of scope, see README.

## D88. A float zero a host comparison reads reaches C as a value the backend cannot see

- **tinygrad:** `renderer/cstyle.py:287` (`ClangRenderer`, whose
  `string_rewrite` writes a float zero as a literal, `0.0f`).
- **tolk:** `Cstyle.float_zero`, `Cstyle.opaque`, `Cstyle.opaque_zeros`
  and `Cstyle.clang` (which adds it for each host target with its types);
  `test/gen/tinygrad.patch`, which
  gives tinygrad's `ClangRenderer` the same rule.
- **Differs:** a float zero constant that a comparison reads renders in the
  comparison as an empty `asm` statement over a register holding it,
  `({float z = 0.0f; __asm__("" : "+w"(z)); z;})`, in a float for a narrow
  type and cast back to it: on arm64 at float16, bfloat16, float32 and
  float64, on riscv64 at float16, float32 and float64, and on x86_64 at
  float16. The register class is the target's for a float: `w` on arm64, `x`
  on x86_64, `f` on riscv64. Every other use of the zero, and every other
  zero, keeps the literal.
- **Reason:** (b): backends lower a select of a value and a zero, by a
  comparison of the two, to an instruction or a call that may keep the
  value's zero where the select returns the literal's: `(v < 0.0f) ? v :
  0.0f` is `-0.0` at `v = -0.0`. clang's AArch64 backend makes it `fminnm` or
  `fmaxnm` from `-O1` up, in clang 19, Homebrew clang 22.1.7 and Apple clang
  17; its RISC-V backend `fmin` or `fmax`, in clang 19; its x86 backend, at
  float16, a call to `fminf` or `fmaxf`, whose zero at a tie is the C
  library's, in clang 19. At float32 and float64 the x86 backend's `minss`
  and `maxss` pick as the select does. A maximum against a zero, rendered as
  such a select, meets it too, and an add of `+0.0` after it is then
  dropped. Under `Rune.jit`, `Nx.where (Nx.less x z) x z` gave `-0.` on arm64
  where eager gives `0.`, and a sum of a minimum against a folded zero gave
  `-0.`, which a float sum never is. The lowering needs the select's
  operands to be the comparison's, so a zero hidden in the comparison alone
  avoids it. Hidden everywhere, the zero also kept the backend from making
  `c ? v : 0.0f`, a select by any other condition, a mask: a routed MXFP4
  product on the host, whose rows are selected by their validity before the
  product, ran 1.17 times as long as the same product without the select,
  and runs 1.05 times as long with the literal there (single-threaded
  kernel, paired). `-ffp-exception-behavior=maytrap` or `-frounding-math`
  also avoid the lowering, at 2.7× to 4.7× the time of a matmul or an
  element-wise kernel.
- **Open:** the GPU renderers keep the literal, and NVRTC 13.4 lowers `(v <
  0.0f) ? v : 0.0f` to `min.f32` for sm_89: under `Rune.jit` on the RTX 5000,
  the selection gives `-0.` at `-0.`. NVRTC has no statement expressions, so
  CUDA needs a helper function around the `asm`; Metal has no inline `asm`.
  HIP and Metal are untested.
- **Pinned by:** the Cstyle suite (`test/renderer/cstyle`): `zeros a
  comparison reads › a float zero a comparison reads is opaque to a backend
  that lowers it to an extreme, and a literal elsewhere › *` and `› a select
  of a value and a zero picks the zero at -0., on the host › *`; rune's `rune
  jit` suite: `classes › a selection of a minimum against its constant is
  exact`, a selection on `minimum c x < c` at `c = 0.`, whose inputs hold
  `-0.`, in float16, float32 and float64.

## D89. A contraction of little reuse streams its operand of least reuse

- **tinygrad:** `codegen/opt/heuristic.py:60-78` (`hand_coded_optimizations`'
  matrix-vector case, which applies only when the first reduce's product
  has two loads as operands, `mulop.src[0].op is Ops.INDEX and
  mulop.src[1].op is Ops.INDEX`, the vector's index has the first weak
  reduce range as a term of its sum and its ranges are among the matrix's:
  that axis split into `MV_THREADS_PER_ROW` local threads, 8, the first
  global axis that 16 divides, whichever operand reads it, into
  `MV_BLOCKSIZE` local threads, 4, then `MV_ROWS_PER_THREAD` upcast lanes,
  4, whatever the matrix's layout, size and reuse), `:9-44` (the tensor
  cores, tried first), `:111-137` (the upcasts of output axes, which find
  reuse only on an axis some access does not read and take 3 or 4 of it),
  and `codegen/opt/postrange.py:111-115` (`apply_opt`, which refuses a local
  split of a reduce axis whose reduction runs inside a weak range that a
  reduction closes, a serial reduce).
- **tolk:** `Heuristic.lanes`, `Heuristic.rows`, `Heuristic.columns`,
  `Heuristic.sector_lanes`, `Heuristic.stream_threads`, `Heuristic.in_flight`,
  `Heuristic.in_flight_held`, `Heuristic.stream_first`,
  `Heuristic.upcast_max`, `Heuristic.summed`, `Renderer.t.per_byte_first`,
  `Heuristic.per_byte`, `Heuristic.run`,
  `Heuristic.reuse_ranges`, `Heuristic.reuse`, `Heuristic.stride`,
  `Heuristic.accesses`, `Heuristic.contraction`, `Heuristic.contracted`,
  `Heuristic.units`, `Heuristic.main_access`, `Heuristic.layout`,
  `Heuristic.streamed`, `Heuristic.early`, `Heuristic.stream`,
  `Heuristic.streams` and their order in
  `Heuristic.hand_coded_optimizations`; `Postrange.Scheduler.apply`'s check
  of a group inside a reduce; `test/gen/tinygrad.patch`, which gives tinygrad
  the same before the goldens are recorded.
- **Differs:**
  - the contraction: a sum over the kernel's first reduction of a product of
    two computations of accesses with no reduce, on a renderer with local
    indices and shared memory; `MV=0` turns it off. An operand may read its
    loads through conversions (`CAST`, `BITCAST`), decode codes and scale a
    vector (D81's part);
  - the streamed operand: an operand's reuse `R` is the product of the sizes
    of the global axes it does not read, times `d` for each it reads only as
    its quotient by `d`. The operand of least `R` is streamed, `W`; of two of
    equal reuse, the one laid out along a contracted range. A gathered
    operand reads every row: `R = 1`;
  - the order against the tensor cores: up to `R = 32`, a contraction that
    sums at least 256 values an output, or a number not known, and does at
    most 64 multiply-adds per byte of the buffers `W` reads (the product of
    its output and contracted ranges' sizes against each buffer's bytes
    once), or a number not known, streams before the cores are tried; a
    shorter sum, or more multiply-adds a byte, tries a core first and streams
    where none applies. Past `R = 32` a core takes it where one applies, and
    the other rules where none does;
  - the layout: `W`'s main access is the one with a range of unit stride
    whose terms of contracted ranges span the most bytes. When that range `u`
    is contracted (rows): `W`'s reuse axes are upcast, each by its share of
    the largest divisor of `R` up to 16 (D92's runs, as a term of `R`); `u`
    splits into up to 32 local threads, and when they take `u` whole and are
    fewer, the contracted range of the next stride into as many more as make
    32; 4 rows, of a global axis `W` reads and the other operand does not
    (D96's part), else one both read, go to 4 SIMD groups at `R = 1` and to
    each thread's registers at `R > 1`; what the threads leave of `u`, or
    else the range of the next stride, is unrolled by its largest divisor up
    to 8, or 4 with rows held. When that range is a global axis (columns),
    after the reuse upcast it splits into 32 local threads and 2 upcast
    lanes, and the first contracted range into the threads, up to 32, that
    make 32768 threads; with more than 32768 outputs the kernel takes the
    other rules. `MV_BLOCKSIZE`, `MV_THREADS_PER_ROW` and `MV_ROWS_PER_THREAD`
    are gone;
  - few outputs (D110's part): when the outputs are too few for the columns
    layout to run `stream_threads`, 32768 threads, a workgroup of 1024
    threads takes fewer threads across them, down to 8, and more along the
    reduce, so that more workgroups share the outputs. 8 threads of 2
    bfloat16 columns read a 32-byte sector. 32768 is the bound of sm_89's
    parts, as D175's floor is: at 16384, from the L2, the decode step's key
    and value projection (`gpt_oss_kv`, 2880 to 512, `stream/decode-kv`) is
    1.16x slower, its query projection (`gpt_oss_qkv`) 1.10x and `vecmat`
    1.52x, and from memory equal; 65536 changes the layout's shape, 8
    threads across, and `vecmat` is 3.5x slower;
  - a group inside a reduce: a reduce axis whose reduction runs inside a
    serial reduce, a weak range that a reduction closes, splits into local
    threads, each iteration of the outer reduce storing and reading the shared
    buffer between barriers. One inside an unrolled reduce, an upcast range
    that a reduction closes, is refused, since each of its lanes would need a
    buffer. tinygrad, which checks only for a weak range, refuses that one too
    for the weak quotient that the unroll's split leaves its reduction; the
    lanes of a tensor core's K, which the core contracts and no reduction
    closes, take a group in both;
  - the constants: `lanes`, `in_flight`, `in_flight_held`,
    `stream_threads`, `stream_first`, `summed`, `per_byte_first` and D175's
    `core_threads` each move the order of a sum. They are fixed per target,
    never read from a device: a value read from the part would give two
    parts of one target different bits, and the program key names the
    target alone (D170). `per_byte_first` is a field of the renderer, since
    a second target measured another crossover: nonnormal's R9700 (gfx1201)
    streams gpt-oss-20b's routed blocks at 149 a byte faster than its cores
    (gate and up 20.6-22.1 against 26.7-28.0 ms a layer, down 10.2-11.1
    against 13.6-14.4, the gather stored as the model stores it, under every
    routing measured) where sm_89's cores win 4-6x at the same intensity.
    Every target keeps 64 until its own crossover is placed: gfx1201's lies
    above 149, and the rows that place it (routed blocks at 1024 and 2048
    tokens, 298 and 597 a byte, both paths forced) are pending on
    nonnormal. Routing does not order the two paths: across eight routings
    no kernel moves by more than 5% on AMD or 10% on CUDA, and a path chosen
    by data would make a row's bits depend on other rows' routes. A gather
    fused into the streamed product, as when its rows are an input, costs
    1.3x on CUDA (the streamed gate and up product 12.3-13.9 against
    9.4-10.8 ms), its rows gathered again in each of 1440 workgroups, which
    the order does not weigh: an open cost.
- **Reason:** (d): gpt-oss-20b (`kaun/examples/06-gpt-oss`) on CUDA, and
  nx.quant's products (`Nx_quant.apply`, `Nx.map_segments` over it)
  compiled by rune. A contraction of little reuse reads its streamed operand
  from memory more than it multiplies: tinygrad's layout puts 8 threads on a
  row whatever the matrix, finds no layout for a reuse past 1, and reads only
  the first reduce range for lanes, which a quantised block's values are not.
  Measured on fudge's RTX 5000 Ada (576 GB/s, 64 MiB of L2), the rule against
  the layouts before it (D89's matrix-vector layout and D92), interleaved,
  ranges of three runs:
  - lanes by stride: tolk's bench `stream/q8_0-1`, Q8_0 16384 by 8192 at one
    token, 1230-1240 to 291 us; kaun's `Quant/cuda/q8_0-product-1-token`
    408-410 to 27-28 us, `q6_k` 3.40-3.41 ms to 571 us, `q4_k` 255-257 to
    53-55 us, `mxfp4-blocks` (GGUF's 17-byte blocks) 86 to 24-26 us; the
    gpt-oss `--q8_0` decode step 22.8 to 8.0-8.1 ms;
  - reuse in registers, rows held: `stream/dense-bf16-12` 837-838 to
    275-276 us, `stream/q8_0-12` 1030 to 431-438 us, `stream/mxfp4-12` 5.35-5.39
    to 3.09-3.10 ms (its decode stored as float32, D163), the routed blocks
    of 16 rows of a 512-token prompt `stream/routed-512-bf16` 33.2 to 10.2 ms
    (from a tensor core) and `stream/routed-512-f32` 57.8-58.0 to 20.6 ms,
    kaun's `Quant/cuda/routed-product-512-tokens` 82.5 to 44.4-44.6 ms. Rows
    held per thread against 4 SIMD groups: at `R = 16` 10.2 against 22.6 ms
    and 20.6 against 28.6 ms; at `R = 2` (dense bfloat16, 16384 by 4096)
    264.6-265.0 against 265.1-265.2 us;
  - the order against the cores: `stream/dense-bf16-16` 805-807 us on a core
    to 288-289 us streamed, `stream/dense-bf16-32` 839-844 to 548-550 us; at
    `R = 64` the core keeps it (`stream/dense-bf16-64`, 1.55 ms both). The
    decode step's attention values, `gpt_oss/gpt_oss_decode_values` (`R = 8`,
    a sum of 1024 slots), 25.2-26.6 us on a core to 14.6-15.3 us streamed;
    its keys, `stream/decode-keys` (`R = 8`, a sum of 64), 8.4 us on a core
    and 13.9-14.6 us streamed, where 32 lanes leave 2 loads each and a sum
    across them that a core takes in one step: a sum under 256 values tries
    the cores first, a constant between the sums measured. Floats are upcast
    as decoded operands are: streamed without the upcast of their 8 heads the
    keys took 16.6 us against 13.9-14.6, and the float32 routed blocks hold
    their 16 rows in 20.6 ms against 28.6 ms in SIMD groups;
  - the order by multiply-adds per byte, forced each way (kernel times under
    `DEBUG=2`, 17 calls a run, load about 60): a core's tiles read a matrix
    from memory at 137-167 GB/s where the streamed threads reach 465-512, so
    a core wins where blocks of rows share a matrix the cache holds, or where
    the arithmetic per byte binds the streamed threads. Under 64 a byte
    streams faster: an MXFP4 matrix of 16384 by 16384 at 16 rows, 30 a byte,
    0.69-0.71 ms against 0.88-1.06 on a core, at 32 rows, 60 a byte,
    1.50-1.52 against 1.68; blocks of 16 rows each of its own bfloat16
    matrix (`stream/routed-512-bf16`, 8 a byte) 10.2 against 33.2 ms;
    gpt-oss-20b's experts with bfloat16 matrices at 128 and 256 tokens, 12
    and 24 a byte, 4.9 and 7.7 ms a layer against 11.2 and 11.1. Past it the
    cores: its MXFP4 experts at 224, 256 and 512 tokens, 66, 88 and 149 a
    byte, 7.0, 2.6 and 3.7 ms a layer against 7.9, 9.9 and 17.5 streamed,
    and bfloat16 experts at 1024 tokens, 72 a byte, 12.7 against 22.8. The
    constant lies between 60 and 66; blocks sharing their matrices at 40 to
    58 a byte stay streamed, 1.1 to 1.55 times slower than a core: the MXFP4
    experts at 128 to 192 tokens (43-58 a byte, 5.0-6.6 ms against 4.1-4.3)
    and bfloat16 ones at 512 (40 a byte, 12.8 against 11.6). The constant is a
    crossover of achieved rates: at 60 a byte the streamed threads reach 18%
    of the GPU's peak arithmetic and a core's tiles 15% of its bandwidth.
    What separates the shared blocks from the dense matrix at 60 is how
    much of the blocks' matrices the cache holds, and sm_89's parts hold
    from 12 MB to 96 MB: no bound of the target tells 8.8 MB working sets
    from 142 MB ones, so the 40-58 a byte band, 1.1-1.55x slower on fudge,
    is the cost of one program per target. End to end, interleaved, three rounds, ranges
    apart: kaun's `GptOss/prefill-512` 446-452 to 186-187 ms and
    `GptOss/experts-at-prefill-512` 18.5-18.7 to 4.66-4.68 ms; unchanged
    within noise, `GptOss/decode-step-Q8_0-cache-1024` (8.1-8.3 ms), kaun's
    `Quant/cuda/*` rows and tolk's `stream/*` and `gpt_oss/*` rows. The rule
    applies on AMD's and Metal's cores too, where the constant is not
    measured: nonnormal's R9700 rows (`GptOss/experts-at-prefill-512`,
    `GptOss/prefill-512`) and Metal's are pending;
  - the bound past `R = 32`: a streamed operand is read once a pass, where
    the other rules' tiles share loads among a workgroup's threads. Float32
    streamed loses from `R = 64` (2107-2171 against 1909-1987 us for a matrix
    of 16384 by 4096) to `R = 1024` (a 1024-cube product, 1054 against 398
    us). Narrower operands streamed faster than tiled to `R = 256` in a model
    of the kernel on tinygrad at the pin, on matrices past L2, which no tolk
    row confirms: they take the other rules past 32 until one does. A
    decoded operand streams at any reuse (D230);
  - unchanged: `stream/dense-bf16-1`, `stream/mxfp4-1`, `stream/decode-head`,
    `stream/q8_0-64`, kaun's `mxfp4-product-1-token` and
    `routed-product-1-tokens`;
  - the parts merged from D81, D96 and D110, measured before fudge's rows:
    the decode step from 45.5 to 11.1 ms on an RTX 5000 Ada, its expert
    products reading MXFP4 codes and a widened bfloat16 activation, which
    tinygrad declines; a workgroup's rows on the matrix's own axis took a
    512-token gate and up product in blocks of 4 from 56.6 to 21.9 ms; few
    outputs took the key and value projections from 9.05 to 7.8 us and the
    router from 6.7 to 3.7 us.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `matvec_*`, `vecmat_*`, `gpt_oss_*`,
  `experts_down_*`, `routed_blocks_*`, `routed_own_*`, `routed_shared_*`
  (blocks sharing their experts' matrices, on a core),
  `routed_at_bound_*` and `routed_past_bound_*` (63.9 a byte streamed, 71.0
  on a core), `shared_keys_*`,
  `rows_m{4,12,32}_*`,
  `q8_0_blocks_*`, `gguf_mxfp4_*`, `mxfp4_m12_*` and `rows_short_*` (a sum of
  64 on the cores),
  recorded from the equally patched tinygrad; `streaming › the streamed
  operand's reuse outside a thread is its reuse over its tile`, `› a SIMD
  group's load of an operand streamed along the reduce reads whole
  sectors`, `› the unroll
  is the largest divisor of its loop up to 8, 4 with reuse` and `› a
  contraction streams up to a reuse of 32, before a tensor core` and `› a
  recorded contraction streams before a tensor core up to 64 multiply-adds
  per byte of its streamed operand` (rune's kernels of one token's products,
  of blocks of their own matrices and of gpt-oss-20b's experts at a 512-token
  prompt), `› each renderer streams before a core up to 64 multiply-adds a
  byte` (the renderers' fields, from the equally patched tinygrad) and `› a
  contraction streams before a core up to the renderer's bound a byte`; `the
  hand-coded optimisations keep a kernel's writes › small kernels ›
  experts_down_metal` (a group inside another reduce); the Postrange suite's
  cases `double_sum_group`, `double_sum_group_twice` and
  `double_sum_unrolled_group`; on fudge, tolk's bench `stream/*` and
  `gpt_oss/*` rows and kaun's `Quant/cuda/*` rows.

## D90. A linked schedule resolves its launches at link

- **tinygrad:** `engine/realize.py:132` (`resolve_params`) and `:157`
  (`exec_kernel`), which find a call's buffers, launch dimensions and variable
  values on every run, and `uop/ops.py:1122-1135` (`UOp._sym_fxn`, a cached
  property that simplifies a symbolic integer once and compiles it to a
  Python function, which `UOp.sym_infer` calls on each evaluation).
- **tolk:** `Tolk_engine.affine`, `Tolk_engine.value`, `Tolk_engine.locate`
  and `Tolk_engine.launch`; `Shape.sym_compile`, and
  `Shape.sym_infer`, which is `Shape.sym_compile` applied once.
- **Differs:** linking a schedule turns each call into a launch per lane of
  rig's program (D146), whose buffers, offsets, scalars and grid are words of
  the run's ints or constants: each variable of the run is a word, a value
  computed from them a word the run computes once
  (`Shape.sym_compile`, which simplifies once and computes on `int`s, exactly
  where a value does not fit), and a value that moves with a loop's trips a
  hole of rig's program, the sum of trip words times constant factors.
  `Shape.sym_compile` is the port of `_sym_fxn`, and `Shape.sym_infer` of
  `sym_infer` (`uop/ops.py:94`), which evaluates once; a caller that
  evaluates an integer many times keeps the function `Shape.sym_compile`
  returns, where tinygrad caches it on the node. A `DEBUG` report of a call lists the run's
  variables that the schedule reads, where it listed every variable of the
  run.
- **Reason:** (c): rig's program takes words and holes (D146), which a link
  resolves once. Speed is not a reason: tinygrad's resolution on each run
  has no path through rig's program to time against, so no fudge row can
  show what resolving at link saves.
- **Pinned by:** the Tolk_engine suite (`test/engine/tolk_engine`): `link and
  run › a linked scan carries its storage across runs on each run's
  parameters`, `› a scan runs its body once per trip, carrying in place`, `› a
  schedule runs with each binding of its variables` and `runs › runs of one
  schedule from two domains each compute their own`; the Ops suite
  (`test/uop/ops`): `sym_compile › computes what exact arithmetic does,
  variables within 50`, `› within 2147483648`, `› at the int extremes`,
  `› computes exactly where int arithmetic overflows` and `› raises as exact
  arithmetic does where a value fits no int`.

## D91. Withdrawn

The host's upcast lanes stayed within 32; merged into D186, which counts the
host's accumulators in its vector registers.

## D92. On the host, a decoded operand shared by a run of an output axis is decoded once a run

- **tinygrad:** `codegen/opt/heuristic.py:111-137` (`hand_coded_optimizations`'
  upcasts of output axes, which find reuse only on an axis that some access
  does not read, and take 3 or 4 values of it).
- **tolk:** `Heuristic.run_cap`, `Heuristic.run`, `Heuristic.decoded`,
  `Heuristic.upcast_runs` and its place in `Heuristic.general`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** on the host (`target.device = "CPU"`), before the other
  rules, an output axis that an operand of the summed product of the
  contraction (D89) reads only as `r / d` is upcast by the largest amount up
  to 4 that divides both `d` and the axis, when that operand is decoded: its
  value converts integers it reads into floats, by a cast or a bitcast,
  outside any access's address. The axis's lanes then share the operand's
  value. With several such axes, each is upcast. On a GPU a streamed
  contraction takes the run as a term of the operand's reuse (D89),
  whatever the operand's type; one that does not stream takes no run.
- **Reason:** (d): `Nx.map_segments` over `Nx_quant.apply` on a prompt, on
  the host, which sorts positions by expert into blocks of 16 rows that read
  their block's matrix, `owner[row / 16]`: every row decoded the matrix's
  MXFP4 codes and scales again, the row axis reaching the matrix through the
  gather. On fudge, kaun's `Quant/host/routed-product-64-tokens` takes
  195-226 ms a call upcast by 4 (`r_88_1920_4_3_180_2_8`); without it its
  product kernel (`r_352_1920_3_180_2_8`) alone takes 6.0-6.9 s.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `routed_blocks_cpu` and
  `routed_tiles_cpu`, recorded from the equally patched tinygrad, and `the
  hand-coded optimisations keep a kernel's writes › large kernels ›
  routed_blocks_*` (slow); on fudge, kaun's `Quant/host/routed-product-*`
  rows.

## D93. The where-closure rule leaves loads alone, and its slice queries walk no more than they need

- **tinygrad:** `uop/symbolic.py:240-244` (`fold_where_closure`, which asks
  whether the condition is in `t.bool_slice` or `f.bool_slice`, and whether
  the condition or a branch reaches an `INDEX`); `uop/ops.py:275-279`
  (`_bool_slice`, a `recursive_property`: each node keeps the set of the
  boolean nodes it reaches) and `:263-273` (`backward_slice`, a
  `cached_property` holding the whole slice of each node asked, and
  `op_in_backward_slice_with_self`, which walks it);
  `codegen/late/coalesce.py:156` (`memory_coalescing`, which addresses a
  load of several elements by a `SHRINK` of the buffer where one element's
  address is an `INDEX`).
- **tolk:** `Shape.fold_where_closure`, `Ops.reaches`, `Ops.ops_reached` and
  `Ops.op_in_backward_slice_with_self`; `test/gen/tinygrad.patch`, which
  gives tinygrad the load gate.
- **Differs:**
  - the load gate: a selection whose condition or branch reaches a `LOAD` is
    left alone, as one that reaches an `INDEX` is. A load's address is an
    INDEX until memory coalescing makes a run of loads one load through a
    SHRINK, valid included; the INDEX gate then misses it, and the rule
    assumes a condition read from memory in every branch below it. The gate
    also leaves alone a selection over a register's read, the load of a
    reduction's accumulator, whose address is the register alone. The
    goldens do not change;
  - the condition: the rule asks whether `t` or `f` reaches the condition, a
    scalar boolean, with `Ops.reaches`: a walk from the branch that enters no
    node built before the condition, since a node is built after its
    sources. `Ops.bool_slice` is gone. The search runs last, after the INDEX
    and load gate, where tinygrad's lookup runs first: a branch that reaches
    an INDEX or a LOAD is rejected without a walk;
  - the gate's sets: each node keeps, as a property filled from its sources,
    the set of the operations of itself and of the nodes it reaches outside
    call bodies; a node whose set is a source's shares it.
    `op_in_backward_slice_with_self` reads that set and no longer builds the
    node's `backward_slice`.

  But for loads, the rule folds the same selections, and the answers are
  the same.
- **Reason:** (d): compile time. A host kernel of 4 values upcasts them
  whole, so its inputs load as vectors. `Nx.betainc` selects by conditions on
  its shapes, and in a jitted chain of `Nx.betainc` over the same shapes those
  conditions are one node in every link: the rule assumed each in a later
  link's branches down through all the earlier links, copying them. Chains of
  1, 2 and 3 float32 links rendered 3918, 9464 and 17811 lines of C and
  compiled cold on fudge (load 30) in 1.7 s, 12.9 s and 77 s; with the load
  gate, 1713, 3411 and 5102 lines, as a chain over 1 value, whose loads are
  INDEXes. 4 links compiled cold in 172 s, and compile in 2.9 s with the
  gate (load 2, rune's compile bench row `special/betainc-chain-f32`). rune
  lowers `Nx.sin` to a reduction full of comparisons and selections
  (`Lower_real.reduce`), and in a jitted chain of `Nx.sin` every link's
  selections are asked about. Each node's boolean set holds every
  comparison of the links before it, and each slice the whole chain below,
  so work and memory grow with the square of the chain.
  Compiling a chain of 80 `Nx.sin` cold on an M1 Max took 63.0 s and a heap
  of 618M words with the boolean sets, and 16.6 s and 76M words with the
  search, whose walks stay inside the link that built the condition; with
  the slices it took 28.0 s and 218M words, and 16.4 s and 73M words with
  the operation sets. A condition built before a long chain of selections
  still makes each search walk the chain below it, which the gate
  spares when the branches read memory. On fudge, with each node's boolean
  set and whole slice kept as tinygrad keeps them, the load gate kept, rune's
  compile bench row `special/betainc-chain-f32` compiles in 2.99 and 2.70 s
  against 8.05 and 7.75 s (medians of 3, two interleaved runs each);
  `shaped-4k` and `trig/sin-nested-8` show no difference beyond their
  ranges.
- **Pinned by:** rune's compile bench row `special/betainc-chain-f32`; the
  Symbolic suite (`test/uop/symbolic`): `symbolic › selections › a condition
  over a load of several elements is not folded in the branches`, `› a
  condition is not folded in branches over a load of several elements`,
  `cost › sym's work on a chain under one condition over a load of several
  elements is linear in its length`, `› sym's work on a chain of selections
  is linear in its length`, `› sym's work on a chain under one far
  condition is linear in its length` and tinygrad's
  `test_where_closure_folding*` goldens; the Ops suite (`test/uop/ops`):
  `graphs › reaches is membership in the node's toposort`, `› reaches
  ~calls:Enter enters call bodies`, `› op_in_backward_slice_with_self is an
  operation of the slice` and `› op_in_backward_slice_with_self enters call
  bodies only under Enter`.

## D94. Withdrawn

A node kept the operations it reaches, so that the where-closure rule's INDEX
gate read a set; merged into D93.

## D95. A device's target follows from the device alone

- **tinygrad:** `helpers.py:222-237` (`_DEV`, the `DEV` setting, and
  `DEV.target`); `device.py:460-465` (`_select_renderer`, which renders for
  `DEV.target` of the device's kind).
- **tolk:** `Tolk_engine.target`; `Device.renderer`; `Helpers.Target`.
- **Differs:** there is no `DEV` setting. `Tolk_engine.target` is a function
  of the device: its vendor's kind, which its driver declares
  (`Rig.capability`), and its `arch`, or the host's CPU target.
  `Device.renderer` takes the target and renders for it with the renderer
  of its kind. `Helpers.Target` keeps the
  record and its syntax, which tests and the goldens' generator write.
- **Reason:** (c). rig places values on devices, and the compiler
  compiles for the device the values live on. A target read from the
  environment applies to every device of the process, so it compiles for a
  device the values do not live on, and a malformed `DEV`, meant for another
  program, failed every program linking tolk when it started.
- **Pinned by:** `Tolk.Setting › startup › starts whatever DEV holds`;
  `Tolk.Device › renderer › picks a device's renderer as tinygrad does`
  (each row renders for the target tinygrad's `DEV` gives the device) and
  `renders for the target it is given`.

## D96. Withdrawn

A matrix-vector workgroup's rows were the matrix's own; merged into D89.

## D100. A reduction of no axes that a broadcast ends is stored

- **tinygrad:** `schedule/indexing.py:263-268` (`run_rangeify`, which stores
  an elementwise value or reduction whose ranges a broadcast ends by
  realizing each of its axes, so a value of no axes is not stored).
- **tolk:** `Indexing.assign_ranges`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the
  goldens are recorded.
- **Differs:** a value of no axes whose consumers broadcast it, and whose
  computation holds a reduction, is stored as one element, as a value with
  axes is. The cost check (`remove_bufferize`) then inlines it again where
  no reduction of it reads a buffer and it reads at most three. A value of
  no axes without a reduction, such as an offset, stays inlined.
  tinygrad computes such a reduction inside its consumer, once for each
  element: `x - x.sum()` sums the whole of `x` for each element.
- **Reason:** (d): `Vega.Loss_scale.step`, whose `finite`, the conjunction
  of `Nx.all (Nx.isfinite g)` over every gradient, selects each update.
  In GPT-2 124M's float16 step (`kaun/examples/04-gpt2`) on an M1 Max's Metal,
  five kernels each reduced all 148 gradients again: the scale and its
  counters, and the embedding updates, about 14 ms a step. Stored, one kernel
  reduces them. On fudge's RTX 5000 Ada, rune's
  `Cuda/finite-selects-148-float16`, the check of 148 float16 gradients
  selecting each, takes 295 and 302 us against 5.0 ms in both runs (medians
  of 11, two interleaved runs each). On the host, vega's `Loss_scale/step/1024`, whose
  check of one large gradient splits its reduction, shows no difference
  beyond its ranges.
- **Pinned by:** the Rangeify suite's `sum_all_broadcast` and
  `finite_checks` rows of `kernel_counts.golden` and their kernel goldens,
  and its `reduce_expand_child` and `preserve_multistage_reduce` rows; the
  Jit suite's `nonzero` and `masked_select` goldens; all from the equally
  patched tinygrad. On rune: missing (the old rune's `lower_linalg` goldens
  `qr_q` and `qr_r`, whose reflections' norms were stored, went with its
  rewrite).

## D101. A group shares its threads with the kernel's independent reductions

- **tinygrad:** `codegen/opt/postrange.py:102-116` (`apply_opt`: a local
  split of a reduce axis splits that axis alone, and sizes the shared
  memory by the first reduction's type).
- **tolk:** `Postrange.Scheduler.siblings`, `Postrange.Scheduler.smem` and
  its use in `Postrange.Scheduler.apply`; `test/gen/tinygrad.patch`,
  which gives tinygrad the same before the goldens are recorded.
- **Differs:** a local split of an axis of reductions that no reduce axis
  encloses also splits, by the same threads, one axis of each other such
  reduction: its first weak axis that the reduction closes and the amount
  divides, when only such reductions close it, when it neither reads a
  grouped reduction nor is read by one, and while every grouped reduction's
  buffer fits in shared memory. The shared memory checked is the sum of the grouped reductions'
  buffers. A reduction that reads another stays out, since the final
  reductions of shared buffers that one thread range closes run as one
  loop.
- **Reason:** (d): the kernel of D100's `finite`. Grouped on its first
  reduction, each of its 16 threads ran the other 147 whole. In GPT-2
  124M's float16 step on an M1 Max's Metal the kernel took 3.1 ms and takes
  0.77 ms; 200 checks of float16 vectors, from 3.6 ms to 0.90 ms. On fudge's
  RTX 5000 Ada, rune's `Cuda/finite-of-148-float16`, the check of 148
  float16 gradients of 256 to 2048 values, takes 174 and 165 us against 1.45
  ms in both runs with the group on the first reduction alone (medians of
  11, two interleaved runs each).
- **Pinned by:** the Postrange suite's cases `sibling_sums_group` (a
  reduction the amount does not divide stays out), `ten_sums_group` (eight
  of ten buffers fit) and `single_kernel_softmax_group` (the sum that reads
  a maximum stays out); the Codegen suite's `two_grouped_stores_local`
  goldens and its claim that a barrier follows each local store; all from
  the equally patched tinygrad.

## D102. Withdrawn

An empty argument of a precompiled call, which the size-0 rule made an
expanded constant, reached the kernel graph as its scalar, so that a call could
take an argument with no element. Its consumer was rune's staged scan with an
empty carry, which rune now replaces by zeros in the step (`Loop`), and no
other raven call passes an empty argument to a precompiled call: rune's suite
passes with the rule gone. `Rangeify.no_indexing_calls` is
`schedule/rangeify.py:138`, and such a call fails the kernel graph's
verification on the expand, as there.

## D103. An integer constant is written as the value its type holds

- **tinygrad:** `renderer/cstyle.py:26-35` (the constant rules: an `int64` is
  its value with `l`, a `uint32` or a `uint64` its truncated value with `u` or
  `ul`, an 8- or 16-bit integer a cast of its value, any other its value).
- **tolk:** `Cstyle.int_const`; `test/gen/tinygrad.patch`,
  which gives tinygrad the same before the goldens are recorded.
- **Differs:** a constant of an integer type is written as the value the type
  holds: the integer part of a float, wrapped to the type's width, with the
  suffix or cast of its type. The least `int64` is written
  `(-9223372036854775807l-1)`. An infinity or a NaN, which has no integer
  value, is a conversion of the float as the program runs. tinygrad writes a
  signed or 8- or 16-bit constant unwrapped, so a value past 64 bits is no C
  literal, and the least `int64` is an unsigned literal negated.
- **Reason:** (b): rune compiles integer constants at every width's
  extremes, the least `int64` among them. The old rune's `test_jit_programs`
  property converted a float padding constant to an integer, written as a
  literal past 64 bits, which C refuses.
- **Pinned by:** the Cstyle suite: `execution on the host › stores each
  integer constant as its type holds it` and `› converts an infinity or a NaN
  to an integer type`; `clang_transcendental_f16.golden`, recorded from the
  equally patched tinygrad; rune's `rune jit` suite: `classes › integer
  constants keep their bits and wrap`. On rune, a float converted to an
  integer past 64 bits: missing (the old rune's `test_jit_programs` went with
  its rewrite).

## D104. A value read widened from a narrower type is stored narrow

- **tinygrad:** `schedule/rangeify.py:205-207` (`bufferize_to_store`, which
  stores a stage's value at its committed dtype).
- **tolk:** `Rangeify.widens`, `Rangeify.bufferize_to_store` and
  `Rangeify.pm_add_buffers`' rule for the index of a narrow store through its
  widening; `test/gen/tinygrad.patch`, which gives tinygrad the same before
  the goldens are recorded.
- **Differs:** a stage that is not stored whole, whose value is a cast to a
  wider type that holds every value of its source (`Dtype.can_lossless_cast`,
  `bool` and the weak types excepted), stores the source at its own type, and
  each index of the buffer widens the value it reads. Where the stage is
  placed and how it is laid out do not change.
- **Reason:** (d): rune widens a product's `bfloat16` operands to `float32`
  before multiplying them (D29), and a stage sits where its consumers end
  their ranges, after the widening. gpt-oss-20b's attention probabilities, its
  attention output and the rows of its experts were stored as `float32`
  values of `bfloat16`: twice the bytes, and a `float32` load, which no CUDA
  tensor core multiplies. Stored narrow, the output projection of a 512-token
  prefill on an RTX 5000 Ada runs on the tensor cores in 0.30 ms where it took
  2.34 ms, and the scores in 0.29 ms where they took 0.87 ms. On fudge,
  gpt-oss's bench row `GptOss/prefill-512` takes 185 ms against 955 and
  958 ms with each stage stored at its committed type (medians of 11, two
  interleaved runs each).
- **Pinned by:** the Heuristic suite's `projection_of_stored` kernel, whose
  golden reads the stored attention output as `bfloat16`, and its cases
  `projection_of_stored_{metal,cuda,amd}` (the tensor cores apply), recorded
  from the equally patched tinygrad.

## D105. A tensor core takes rows that share the other operand's tile

- **tinygrad:** `codegen/opt/postrange.py:168-180` (`_apply_tc_opt`: M runs
  along a range of `in0` that `in1` does not read, N along one of `in1` that
  `in0` does not read), `:203` (the core multiplies the multiply's own
  operands), and `codegen/opt/heuristic.py:28` (`hand_coded_optimizations`
  tries the cores on a kernel of one reduce axis only, below `TC_OPT=1`).
- **tolk:** `Postrange.Scheduler.in_tiles`, `Postrange.Scheduler.try_core`'s
  `roles` and `Postrange.Scheduler.use_wmma`'s `own`;
  `Heuristic.decoded_product` and `Heuristic.tensor_cores`;
  `test/gen/tinygrad.patch`, which gives tinygrad
  the same before the goldens are recorded.
- **Differs:** a range both operands read is the core's M (or N) range of one
  of them when the other reads it only as `r / d`, `d` a multiple of the
  core's M (or N): that operand reads one value over each tile of the core,
  and the core reads its bits of the range at 0. Either operand may be the
  core's A: the choices with `in0` as A come first, as tinygrad orders them,
  then those with `in1`. A summed product with a decoded operand
  (`Heuristic.decoded`) tries the cores whatever its number of reduce axes.
- **Reason:** (d): `Nx.map_segments` over `Nx_quant.apply` on a prompt, which
  sorts positions by expert into blocks of up to 16 rows, each multiplying its
  expert's matrix, `W[owner[row / 16]]`. The rows read the row axis and the
  matrix reads it by blocks, so tinygrad finds no M range; the decoded bytes
  split the reduce into bytes and a byte's two values (D89), which tinygrad's
  heuristic leaves to the hand-coded path. A block of 8, which fewer positions
  take, fits CUDA's N of 8 and not its M of 16, so its rows are the core's B.
  gpt-oss-20b's 512-token gate and up product on an RTX 5000 Ada takes 2.9 ms on
  the tensor cores, from 13.3 ms, and the down product 1.5 ms, from 6.9 ms; the
  prefill takes 0.22 s, from 0.71 s.
  Blocks of up to 32 rows stream before the cores unless they do more than
  64 multiply-adds per byte of their matrices (D89): blocks each of its own
  bfloat16 matrix stream, 10.2 ms against 33.2 on a core for the gate and up
  product on fudge; gpt-oss-20b's blocks of 16 rows sharing 32 experts' MXFP4
  matrices at a 512-token prompt, 149 a byte, take the cores, and kaun's
  `GptOss/experts-at-prefill-512` runs 4.66-4.68 ms against 18.5-18.7
  streamed, `GptOss/prefill-512` 186-187 ms against 446-452. Blocks sharing
  their matrices at 40 to 58 a byte stay streamed, up to 1.55x slower than a
  core on fudge: D89's band, the cost of one program per target.
- **Pinned by:** the Heuristic suite's cases `routed_tiles_{metal,cuda,amd}`
  (the cores apply to a product in blocks of 64 rows) and
  `routed_shared_{metal,cuda,amd}` (blocks of 16 rows sharing their
  experts' matrices), recorded from the equally patched tinygrad, and its
  `streaming › a recorded contraction streams before a tensor core up to 64
  multiply-adds per byte of its streamed operand` over gpt-oss-20b's expert
  kernels; on fudge, kaun's `GptOss/experts-at-prefill-512` and
  `GptOss/prefill-512`; the Postrange suite's `tc_operands_swapped`
  (the second choice of a matmul makes its `in1` the core's A). On rune, the
  values on a GPU: missing (the old rune's `Rune.quant › gpus › CUDA › on
  CUDA, a prompt's routes in blocks are eager's` went with its rewrite).

## D106. A value every output of a reduction reads stays stored

- **tinygrad:** `schedule/rangeify.py:48-97` (`remove_bufferize`, which
  inlines a stored value that reads at most three buffers and no reduction
  reading a buffer, whoever reads it) and `schedule/indexing.py:259`
  (`run_rangeify`, which stores a value its consumers read at different
  indices without marking it broadcast).
- **tolk:** `Rangeify.remove_bufferize`'s `read_by_every_output`, with
  `Rangeify.reduced` (a range a reduction of the graph closes), and
  `Indexing.assign_ranges`, which marks such a value `Broadcast` when its
  consumers broadcast it;
  `test/gen/tinygrad.patch`, which gives tinygrad the same rule.
- **Differs:** a stored value that is broadcast, read only through the
  reduce ranges of its consumer, those its reductions close, and reads more
  than one buffer stays stored. Its consumer is a reduction that every output of reads it, as a
  matrix-vector product reads its vector, so inlined it would be computed,
  and each of its buffers loaded, once per output. A value read through an
  output range, as a matrix's rows are, is inlined as before: a tiled
  product computes it once for the outputs a thread holds. A value stored
  for consumers that read it at different indices, as the query, key and
  value projections read one normalised activation, is marked broadcast
  when they broadcast it, as D77 marks a value that a broadcast ends.
- **Reason:** (d): gpt-oss-20b's decode on CUDA (RTX 5000 Ada). The
  normalised activation, the residual divided by its root mean square and
  scaled by its gain, reads three buffers and divides, and every projection
  of a decode step computed it again for each of its outputs: the router's
  32, the key and value projections' 512, the experts' 23040 gate and up
  rows. Stored, one 2880-element kernel per normalisation (1.2 us), the
  router went from 37.1 us to 6.8 us, the key and value projections from
  22.1 us to 9.4 us, the gate and up product from 113.9 us to 73.1 us (54%
  to 84% of the bandwidth), and the step's kernels from 11.94 ms to 9.73 ms.
  The 512-token prefill keeps its kernels and its 0.71 s. On fudge,
  gpt-oss's bench row `GptOss/decode-step-cache-1024` takes 10.2 ms against
  11.3 and 11.2 ms without the rule (medians of 11, two interleaved runs
  each).
- **Pinned by:** the Rangeify suite's `normed_vecmat` (a normalised vector
  read by a product), `normed_vecmats` (by two products, stored once) and
  `normed_matmul` (a normalised matrix, inlined) rows of
  `kernel_counts.golden` and their kernel goldens, and `variable_read_twice`
  (a value two full reductions read, inlined), from the equally patched
  tinygrad.

## D107. Withdrawn

A batch submitted a streamed queue after the queues it waited for. rig orders
submissions across queues (D146).

## D108. Withdrawn

On the host, a load a reduce reads filled the lanes by 2; merged into D186,
where the fill is a candidate within the register budget.

## D109. Withdrawn

On the host, a kernel of several reduces unrolled within 32 lanes; merged
into D186, whose unroll clauses hold what an unroll keeps across a loop.

## D110. Withdrawn

A matrix-vector product with few outputs took fewer threads across them;
merged into D89.

## D111. A hung AM device is lost, never recovered in the process

- **tinygrad:** `runtime/ops_amd.py:759-779` (`_collect_interrupts`, whose
  `reset` resets the compute engines through `AMDev.recover`, rewinds the
  compute queue and moves the timeline past the hung value, and
  `on_device_hang`) and `:830` (`can_recover`, true under AM off a virtual
  function); `runtime/support/am/amdev.py:282-291` (`recover`).
- **tolk:** none: rig's AMD driver owns the GPU, and loses it on a hang,
  driver-less after its hang limit and under the kernel's driver when the
  driver reports one.
- **Differs:** a hang or fault loses the device under AM as under KFD. Every
  later operation on memory it can reach raises `Lost`, and the process does
  not reset the GPU. An open of the GPU then gives a fresh device under KFD,
  with new queues, and refuses it under AM; the next process that opens it
  resets it.
- **Reason:** (c): after a reset nothing the device held can be trusted. Work
  in flight is dropped, its queues restart, and its buffers and programs hold
  whatever the hung work left. nx's values are immutable, so a value must fail
  loudly rather than silently outlive its storage. Recovery sits behind
  device-lost semantics: after a fault every operation on the device's values
  raises, naming the device and the fault, and reopening gives a fresh device
  with a new identity. An in-process reset would only be how AM reopens.
- **Pinned by:** rig's AMD fault suite (`packages/rig/test/amd/test_fault.ml`),
  on a GPU.

## D112. Withdrawn

An AMD memory barrier left the host data path to the host. rig's AMD driver
encodes its barriers (D146).

## D113. A compiled binary is kept under its compiler's identity

- **tinygrad:** `device.py:331-337` (`Compiler.__init__`, which drops
  `cachekey` when `CCACHE` is 0, and `compile_cached`: a binary is kept under
  its source in the table `cachekey`);
  `runtime/support/compiler_cpu.py:16`, `runtime/ops_metal.py:40`,
  `runtime/support/compiler_cuda.py:54` and
  `runtime/support/compiler_amd.py:83` (the tables, named by the compiler,
  the architecture and a key); `codegen/opt/search.py:106` (the beam search's
  key: the kernel, the width, `allow_test_size`, the device and the suffix).
- **tolk:** `Renderer.Compiler.v`'s `cachekey` (a function asked once) and
  `Renderer.Compiler.compile_cached`; `Compiler_cpu.statement`, and the
  tables: `Compiler_cpu.clang`'s, `Compiler_metal.table`,
  `Compiler_cuda.nvrtc`'s and `Compiler_amd.hip`'s; `C.identity`;
  `Codegen.program_key` and `Codegen.kept`; the search's settings
  (`Setting.beam_padto` to `Setting.beam_estimate`) and `Search.beam_search`'s
  key.
- **Differs:** a table also names everything besides the source that
  determines a binary: for Clang, the digest of what `clang -###` states it
  runs (its version and installation, the processor and features `native`
  resolves to, every option) for a short and a long source, and the length
  between them (D167), with `-ffile-compilation-dir=.` keeping the
  working directory out; for MTLCompiler, NVRTC and comgr, the build of macOS
  or the library's file, size and modification time, and the options, PTX or
  cubin included. The table is the compiler's identity whatever `CCACHE`
  holds: the keys of programs and searches take it, and `CCACHE`, read when a
  binary or a program is compiled, decides only whether the disk keeps it. A
  program is kept on disk under its compiler's table. The beam search's key
  adds the
  library's sources, the renderer, its compiler's table, and the settings
  that shape compilation (`Setting.shaping`), among which those that pick
  the candidates (`TC` and `BEAM_PADTO`, D114) and those
  the search declares, `BEAM_UOPS_MAX`, `BEAM_UPCAST_MAX`,
  `BEAM_LOCAL_MAX`, `BEAM_MIN_PROGRESS` and `BEAM_ESTIMATE`.
  `BEAM_STRICT_MODE` is not keyed: a strict search that completes finds what
  another finds.
- **Reason:** (b): raven's test suites and rune share the default cache, so a
  hit must answer the compilation it stands for. tinygrad's key answers a
  changed flag in `compiler_cpu.py`, an upgraded toolchain, a program of
  another compiler of the same renderer name, or a search under other
  settings, with a binary or a search made for something else; with
  `CCACHE=0` its compilers have no table, so two that differ in their options
  alone share programs.
- **Pinned by:** `Tolk.Compiler_cpu › cache`, `Tolk.Compiler_metal › cache`,
  `Tolk.Compiler_cuda › cache`, `Tolk.Compiler_amd › cache`,
  `Tolk.Renderer › Compiler › the table is asked for once, when first
  needed`, `› ccache is read when compiling` and `› compile_cached compiles
  every time with ccache off`, `Tolk.C › identity`, `Tolk.Codegen › programs
  are kept › a program is not returned for a compiler of other options, with
  CCACHE=0`, `Tolk.Codegen › programs are kept on disk › a program is not
  read back for a compiler of another table`, and
  `Tolk.Search › a search kept under one setting measures again under
  another`, `› a search kept under one value of a setting declared by its
  caller measures again under another` and `› the search's settings shape
  compilation`.

## D114. Every environment variable is one declared setting

- **tinygrad:** `helpers.py:158-163` (`getenv`, read where it is called,
  with that call's default) and `:180` (`ContextVar`); `:242` (`TC`, default
  `1`, and `TC_OPT`, default `0`); `codegen/opt/search.py:17-21` (the
  actions, reading `getenv("TC", 1)` and `getenv("TC_OPT", 2)`);
  `codegen/opt/heuristic.py:21` (`TC_OPT`'s default, `2` during a search and
  `0` otherwise); `engine/jit.py:35` (`getenv("JITBEAM", BEAM.value)`);
  `engine/realize.py:279` (`compile_linear`'s `profile`, `PROFILE or
  DEBUG >= 2` unless given, for `hcq_compile`); `codegen/__init__.py:370`
  (`os.environ.get("DBGTV")`, read on each failure).
- **tolk:** `Setting.reach`, `Setting.shaping`, `Setting.t`,
  `Setting.jitbeam` and `Setting.tc_opt`; `Search.tc_opt_all` and
  `Search.actions`; `Heuristic.tensor_cores`; `Setting.dbgtv`;
  `Realize.compile_linear`; rune's `Compile.settings`.
- **Differs:** tolk reads no variable outside a setting: no module exports
  `getenv`. Each variable is declared once (`Setting`), with its type,
  default and reach. A setting reaches `Output` if a change of its value
  alone can change what a compilation that returns makes from its graph,
  renderer and compiler, for the same measurements, and `Process`
  otherwise. The caches key on every
  `Output` setting (`Setting.shaping`, D63). `BEAM` and `JITBEAM` reach
  output: `compile_linear` writes the width into the kernels it searches.
  `CC`, `CUDA_PATH` and `ROCM_PATH` reach the process: they are read when a
  compiler is made, and its table names the tools and options they pick
  (D113). tinygrad reads a variable with `getenv` at any call site, so one
  variable can have two readers with two defaults: `TC_OPT` is a
  `ContextVar` of default `0` and a `getenv` of default `2` in the search.
  In tolk `TC_OPT` is the level of hand-coded optimizations alone, default
  `0`, and the search's candidates take the level `2`, a constant
  (`Search.tc_opt_all`): it admits every kernel the lower levels admit, and
  the search measures what it admits, so a lower level only shrinks the
  search. `TC_OPT` in the environment no longer reaches the search. `TC` and
  `BEAM_PADTO` are read when the search makes its candidates (`actions` is a
  function), so a `context` binding of `TC` reaches the search, where
  tinygrad's search sees the environment alone. `JITBEAM` is a setting
  whose `None` stands for `BEAM`'s current value. `compile_linear` reads neither `DEBUG` nor `PROFILE`, since it makes no
  batch to stamp (D146): rig's profiles record each submission's span, and
  the engine makes each kernel a submission of its own from `DEBUG` 2, when
  it prints each kernel's time. There is no `PROFILE`. A variable tinygrad reads on each use, such as
  `DBGTV`, is read when the program starts. A library that compiles with
  tolk declares its own variables the same way, so rune's jit keys on its
  `Output` settings too, and beyond them only on what is not a setting of tolk. A function compiled with an explicit width keys on
  that width in `BEAM`'s entry, without `JITBEAM`'s, since the width
  overrides both. `ALLOW_TF32`, default `false` as in tinygrad, is an
  explicit choice of precision: a TF32 core rounds each `float32` operand to
  10 mantissa bits, so a contraction compiled under it may err past
  `Nx.contract`'s bound.
- **Reason:** (b): rune's jit memo and tolk's program, schedule and search
  caches key on `Setting.shaping`. A variable read outside a declaration on
  a compile path is missing from every key, and a cache returns what was
  made under another value: the search's `TC_OPT`, read raw with its own
  default, kept a search made with `TC_OPT` unset for a process with
  `TC_OPT=0`. A setting read by compilation but classed `Process` must be
  keyed again by each caller that keeps what it compiles, as rune's jit
  did for `BEAM`, `JITBEAM` and `DEBUG`; `DEBUG` read by `compile_linear`
  would reach output and key every cache on verbosity.
- **Pinned by:** `Tolk.Setting › declarations` and `› shaping` (every
  test, among them `› holds the library's settings of output, and those
  alone` and `› changes a library setting's entry when the setting
  changes`); `Tolk.Search › the tensor cores' actions ›` every test and `›
  a search kept under one value of a setting declared by its caller
  measures again under another`; `Tolk.Codegen › programs are kept › a
  program is made again under another value of ›` each setting, a setting
  its caller declares among them; `Tolk.Schedule › create_linear_with_vars
  › cache › a body scheduled under one setting misses under another › a
  setting its caller declares`; rune's `search.t`, a function given a
  width that compiles once under another `BEAM` or `JITBEAM`. On rune:
  missing (the old rune's `Rune.jit › keys › a setting that shapes
  compilation ›` each setting, `› a call under DEBUG=2, which reports kernel
  times, retraces once` and `› a call under DEBUG=1 replays the program` went
  with its rewrite).

## D115. A search linearizes a candidate before it compiles it

- **tinygrad:** `codegen/opt/search.py:63-69` (`_try_compile`: `to_program`,
  then the count of `prg.src[1].src` against `BEAM_UOPS_MAX`).
- **tolk:** `Search.try_compile` and `Codegen.linearize`.
- **Differs:** a candidate is linearized (`Codegen.linearize`), dropped if it
  has `BEAM_UOPS_MAX` instructions or more, and only then rendered and
  compiled (`Codegen.to_program` completes the linearized program). tinygrad
  renders and compiles every candidate, then drops the one past the cap. The
  search keeps and drops the same candidates.
- **Reason:** (d): a beam search of lorenz_simple's step (`Rune.jit ~beam`)
  on AMD. The candidates past the cap are the largest unrolls, the slowest to
  compile, and compiling them took 154 s of a search, 30 s once they were
  dropped before compiling. On fudge, the tolk bench's
  `search/beam/r_128_16_3_4_4_3_3_400_400`, a whole search of lorenz's kernel
  on the host whose candidates Clang compiles uncached, over rounds a timing
  of their optimisations fixes, whose last rounds drop candidates past the
  cap, takes 0.47 to 0.59 s in five of six runs and 1.26 s in one under load
  (medians of 11, interleaved runs), against 0.71 to 1.58 s in seven when
  each candidate compiles before the cap drops it.
- **Pinned by:** `Tolk.Search › BEAM_PADTO=1 ... › a candidate of
  BEAM_UOPS_MAX instructions or more is not compiled`; `Tolk.Codegen ›
  linearize and compile › compile completes linearize into the program
  to_program makes` and `› linearize compiles nothing`.

## D116. A search keeps no candidate's program

- **tinygrad:** `codegen/opt/search.py:63` (`_try_compile` calls
  `to_program`); `codegen/__init__.py:477-505` (`do_to_program` takes a sink
  or a program at any stage), `:516-518` (`to_program_cache`) and `:463`
  (`do_compile` calls `compile_cached`); `engine/realize.py:234`
  (`_get_call_to_compile` takes a program not yet compiled).
- **tolk:** `Search.try_compile`, `Codegen.compile` and `Codegen.to_program`,
  and `Realize.get_call_to_compile`.
- **Differs:** a candidate is compiled by `Codegen.linearize` then
  `Codegen.compile`, through its compiler's disk cache
  (`Renderer.Compiler.compile_cached`), as tinygrad's `compile_cached` does;
  its program enters neither of `to_program`'s tables, where tinygrad's
  `to_program` keeps it in the memory of the process that compiled it. A
  search keeps its choice, and the kernel compiled with it keeps its own
  program and binary, as in tinygrad; that kernel is compiled apart from the
  candidate it was, since a candidate is named `"test"`. `to_program` takes
  only a kernel's sink, and `compile` only what `linearize` returns:
  `to_program ast ren` is `compile (linearize ast ren) ren`, kept. tinygrad's
  `do_to_program`, and `lower_and_compile` with it, also resume a program from
  any stage.
- **Reason:** (a): tinygrad compiles candidates in worker processes
  (`get_worker_pool`), which exit after `BEAM_MAX_TASKS_PER_CHILD` (16) tasks
  and free the programs they kept, where tolk compiles them on domains of the
  calling process (D5), whose `to_program` table would hold every candidate of
  every search for the life of the process. A stage resumed by `to_program` is
  a stage reached by `Codegen.linearize` or `Codegen.compile`, so no caller
  loses one. The entry also kept memos of each kernel and each source a search
  compiled, for duplicate candidates measured on AMD; they are withdrawn: on
  fudge, the tolk bench's `search/beam/r_128_16_3_4_4_3_3_400_400` took 0.52,
  0.50 and 1.39 s with them against 0.47 and 1.26 s without, in the same
  rounds (medians of 11, interleaved runs), and a first call of `Rune.jit
  ~beam:2` of a 64 by 64 float32 product and its tanh on the host 9.4, 7.3
  and 7.0 s with them against 3.7, 7.6 and 10.0 s (medians of 3 cold
  processes). As tinygrad's `codegen/opt/search.py:133-143`, every candidate
  compiles, and a binary timed before is dropped after its compilation.
- **Pinned by:** `Tolk.Search › a search keeps each candidate's binary in its
  compiler's table, as a kernel compiled with it keeps its own`; an absence:
  `Search.try_compile` calls `Codegen.linearize` and `Codegen.compile`, which
  keep nothing; `Tolk.Codegen › linearize and compile` (every test).

## D117. Withdrawn

A search timed its kernel before its first round and started the beam from
its samples, and a round progressed only when every sample of its fastest
candidate beat every sample of the beam's first kernel by
`BEAM_MIN_PROGRESS`, so that it would not run rounds on gains within the
noise, as searches on AMD did. On fudge, a first call of `Rune.jit ~beam:2`
of a 64 by 64 float32 product and its tanh on the host took 9.4, 7.3 and
7.0 s with the rule and 2.4, 4.7 and 4.6 s with tinygrad's (medians of 3
cold processes, interleaved runs), and the
kernel each search chose, replayed after a search over an empty cache, ran
in 43, 61, 40 and 40 us against 104, 53, 55 and 36 us (medians of 11): no
gain beyond the ranges. As tinygrad's
`codegen/opt/search.py:112,161-164`, `Search.beam_search` starts from its
kernel untimed, ends a round on no timing, a fastest time under
`BEAM_MIN_PROGRESS` or a gain under it, and then keeps the fastest alone if it
beat the beam's first.

## D118. Withdrawn

A program's launch read its own kernel of its cubin. rig's NV driver loads each
kernel of a cubin (D146).

## D119. Withdrawn

AMD's compute queue stamped the GPU's clock as it reached the packet. rig's AMD
driver times its submissions (D146).

## D120. Withdrawn

An AMD program's descriptor was its code object's one kernel. rig's AMD driver
reads code objects (D146).

## D121. Withdrawn

AMD's packet layouts were nx.amd.packet's. tolk encodes no packet (D146).

## D122. A stored value keeps its bounds

- **tinygrad:** `schedule/rangeify.py:227` (`bufferize_to_store`'s `ALLOC`),
  `:283` (`debuf`) and `schedule/prepare.py:153,161` (`copy_to_anon_store`,
  `stage_to_anon_store`), whose storage carries no bounds, so a read of it
  has its type's; `uop/ops.py:1119` (`_min_max`, which bounds a `BITCAST` by
  its type).
- **tolk:** `Ops.stored_bounds` and `Ops.bitcast_bounds`;
  `Rangeify.bufferize_to_store` (a stage's storage) and `Rangeify.debuf`;
  `Prepare.copy_to_anon_store` and `Prepare.stage_to_anon_store`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the
  goldens are recorded.
- **Differs:** storage that a stage or a copy fills carries the bounds of the
  value stored when they are narrower than its type's, and the parameter a
  kernel reads it through keeps them, so each read has them. Bounds of one
  value are not kept: a parameter whose bounds are one value folds to that
  constant, in place of the storage. A bitcast between integer types of one
  width keeps the bounds of a value that both types hold, from 0 to the
  lesser of their greatest values.
- **Reason:** (d): rune lowers `Nx.take` to a gather at the clamped index
  and a check of the index, one unsigned comparison, `bitcast i < n`. A
  gather by indices that other kernels compute, clamp and store read them
  at their type's bounds, so the check stayed, and `pm_move_where_on_load`
  moved it onto each load the gather feeds. With the stored range known, the
  gather keeps no per-load check. A routed MXFP4 product whose experts are
  sorted and clamped in kernels of their own masked its 40 code and scale
  byte loads: gpt-oss-20b's gate and up product of 512 tokens on an M1
  Max's Metal took 42.6 ms and takes 36.4 ms, as fast as a gather of the
  decoded matrices. Within one kernel, the bitcast hid the clamp's bounds
  from the check. On fudge's RTX 5000 Ada, gpt-oss's bench row
  `GptOss/experts-at-prefill-512`, one layer's routed MXFP4 experts at 512
  tokens, takes 4.55 and 4.60 ms against 5.61 ms in both runs without the
  stored bounds, and `GptOss/prefill-512` 185 ms against 198 and 199 ms
  (medians of 11, two interleaved runs each).
- **Pinned by:** the Rangeify suite's `index_stored_zero_fill` gather, whose
  `_kernels` golden reads the stored indices with bounds `(0, 7)` and keeps
  no check; the Ops suite's `› bounds › bounds hold every value a bitcast
  between integers takes` and `› storage
  keeps the bounds of its values where they are narrower than its type's
  and hold more than one value`; the memory goldens `convs` and
  `matmul_chain`, whose stored activations after a ReLU carry
  `(0.0, inf)`, recorded from the equally patched tinygrad.

## D123. The engine runs a loop of calls while a flag holds

- **tinygrad:** `uop/spec.py:88` (a `BACKEDGE` closes a bound-less range,
  inside a kernel), `engine/realize.py:284-291` (`run_linear` runs every
  call of a schedule once, in order).
- **tolk:** `Spec.shared` and `Spec.kernel_graph` (a `Backedge` around
  calls, in the tensor and kernel-graph specs), `Schedule.is_loop` and
  `Schedule.linearize` (`create_schedule` keeps a back edge of a call),
  `Memory.uses` (the plan counts its calls and flag), and
  `Tolk_engine.loop`.
- **Differs:** an `Op.Backedge` stands around a call of a schedule, or a
  linear of calls, with a loop range of at most as many trips and a flag, one
  boolean of storage the calls write. The schedule keeps it as an entry, and the engine makes it a loop of rig's
  program (D146) that reads the flag before each trip and runs the calls
  while it holds, its trip in a word of the run's ints, which a call's scalar
  argument reads too (D140). The flag is storage the program makes or a
  buffer a caller binds. tinygrad's schedules run each call once, and stop on
  no value. A range of one trip around a call folds to a constant, which
  `Shape.symbolic` drops from the end, so that the call runs once; tinygrad
  deleted that rule (07efd3877), since its ends close only a kernel's ranges,
  where code generation drops a constant (`codegen/simplify.py:9-11`,
  `flatten_range`).
- **Reason:** (b). rune's `Rune.iterate` under `Rune.jit` stages a loop that
  stops on a condition of its carry as one body compiled once, which no
  schedule tinygrad makes can run.
- **Pinned by:** the Engine suite: `link and run › a loop runs its call while
  its flag holds, at most its trips`, `› a loop of one trip passes its call the
  trip as a scalar argument`, and `Tolk.Symbolic › symbolic › ordering › an end
  drops the ranges that became constants`, `› a loop's flag that views wider
  storage is refused`, `› a planned buffer a loop writes is not placed over
  one it leaves`, `› a loop's planned flag is not placed under a buffer before
  it`, `› a back edge whose flag is bound storage reads that storage` and
  `GPUs › Metal › a loop runs its call while its flag holds, at most its
  trips`; rune's `rune jit › constructs › an iterate stops at the program's
  first failure` and the `rune iterate` suite.

## D124. A search times no compilation

- **tinygrad:** `codegen/opt/search.py:49-59,76` (`timeout_handler`, and the
  alarm of `BEAM_TIMEOUT_SEC` that `_try_compile` sets and clears).
- **tolk:** `Search.try_compile`.
- **Differs:** a candidate's lowering and compilation run to their end.
  tinygrad sets an alarm of `BEAM_TIMEOUT_SEC` (10 s) around each candidate
  in its worker process and drops the candidate when it rings. The alarm
  stops Python code and a Clang child. comgr, NVRTC and MTLCompiler run in
  the worker through ctypes, and Python runs a signal handler only once the
  call returns, so tinygrad drops such a candidate after its compile ends,
  and a compile that hangs hangs its search. tolk bounds a candidate by its
  size instead: one of `BEAM_UOPS_MAX` instructions or more is dropped
  before it is compiled (D115), and lowering raises past
  `REWRITE_STACK_LIMIT`.
- **Reason:** (a): tinygrad bounds a candidate as one unit in a worker
  process. A tolk candidate is lowered on a domain (D5), which no signal
  stops, and a process with domains cannot fork. MTLCompiler compiles
  in this process and cannot be stopped either. A limit would reach only
  Clang, comgr and NVRTC, the toolchains that run as processes, and give the
  search a promise that depends on the backend.
- **Pinned by:** an absence; the bound tolk keeps is pinned by D115's
  `Tolk.Search › BEAM_PADTO=1 ... › a candidate of BEAM_UOPS_MAX
  instructions or more is not compiled`.

## D125. Withdrawn

A launch descriptor's field held all of its value. rig's NV driver writes its
launch descriptors (D146).

## D126. The transcendental functions are within one ulp

- **tinygrad:** `codegen/decomp/transcendental.py:7`
  (`TRANSCENDENTAL_DTYPES`, float16 included), `:115-148`
  (`cody_waite_reduction`, by half turns, whose float32 parts are the first
  three of Sleef's split for FMA and the fourth of its other split),
  `:150-166` (`sin_poly`, `sin_poly_small`, `sin_poly_large`), `:170-191`
  (`xsin`), and `:219-255` (`xlog2`).
- **tolk:** `Transcendental.transcendental_dtypes`, `Transcendental.radians`,
  `Transcendental.cody_waite`, `Transcendental.sin_kernel`,
  `Transcendental.cos_kernel`, `Transcendental.xsin` and
  `Transcendental.xlog2`; `test/gen/tinygrad.patch`, which gives tinygrad
  the same functions before the goldens are generated.
- **Differs:** `xexp2`, `xlog2` and `xsin` are defined in float32 and float64;
  float16 computes at float32 and rounds once, as bfloat16 and the 8-bit
  floats did. Both reductions remove quarter turns and return the remainder
  in two floats, the rounded remainder and the rest: Cody-Waite subtracts
  `q pi/2` in four parts whose products by a quotient below `2^15` are exact,
  and carries the rounding of the third difference to the last, in place of
  tinygrad's parts, whose fourth term belongs to another split of pi and
  leaves `1.2e-10 q` behind; Payne-Hanek's remainder is summed and scaled in
  two floats (D31). The sine and the cosine of the remainder are fdlibm's
  kernels, which take the rest to first order, in place of one polynomial on
  half turns evaluated as `d p(d^2)`. `xsin` returns a zero of either sign
  unchanged. `xlog2` is fdlibm's `log2` (`e_log2.c`): `m` in
  `[1/sqrt 2, sqrt 2)`, `log (1 + f) = f - h + s (h + R)`, `f - h` split so
  that its product by the leading part of `1/ln 2` is exact, the exponent
  added last with its rounding carried, in place of a polynomial in
  `(m - 1)/(m + 1)` whose rounding scales into the result. `xexp2` is
  unchanged. Measured against correctly rounded results, every float32 input
  and every float16 and bfloat16 input, and drawn float64 inputs, the largest
  errors were 1, 5 and 205266 units in the last place in float32 (`exp2`,
  `log2`, `sin`), 4, 3 and 2 in float16, and 1, 3 and 2 in float64; they are 1
  in every case. On one core of the M1 Max, without vectorization, a float32
  `xlog2` costs 8.0 ns from 5.0, `xsin` below its switch-over 11.4 ns from
  6.2 and beyond it 28.5 from 21.0; float64 `xlog2` 9.8 ns from 6.4, `xsin`
  13.3 from 11.4 and 69.2 from 73.4.
- **Reason:** (b). rune's `exp`, `sin`, `cos` and the kinds built on them
  compute with tolk's `exp2` and `sin`, and its `Log2` kind is tolk's `log2`
  (rune's `Lower_real.unary`); rune's `log` reads no `log2`. A target's
  function one unit off leaves rune's compositions within their stated
  bounds; tinygrad's `xsin` left the sine of a float32 near a multiple of pi
  below 30 with no correct bit, and its `xlog2` was 5 units off in float32.
  On the host, where Clang renders none of these functions, every one is
  tolk's.
- **Pinned by:** the Transcendental suite (`test/codegen/decomp/transcendental`):
  `accuracy › accuracy.golden`, which holds `exp2`, `log2` and `sin` of
  float16, bfloat16, float32 and float64 values correctly rounded by mpmath
  (`test/gen/codegen/decomp/transcendental.py`) and checks each function, and
  `xsin ~fast` below 30, within an ulp of them; `special values › sin <type> -0x0p+0 is -0x0p+0` and the subnormal rows;
  `types › <function> refuses a node of another type › half`; the graph
  goldens and `values.golden`, from the patched tinygrad; and rune's
  `Lower_real › transcendental › exp within 2 ulps at float32, 9 at
  float64`, `› exp2 within 4 ulps …`, `› log2 within 4 ulps …`, `› sin within
  2 ulps …` and `› cos within 2 ulps …`, on the host.

## D127. Withdrawn

A selection moved into a converted gated load only when the load's type held
its other value, since tinygrad converted that value to the load's type and
back. tinygrad no longer moves a selection through a conversion
(`codegen/late/gater.py:5,14-15`): `Gater.move_where_load` moves it only into
a load of the selection's own type, where converting the other value changes
nothing, so `Gater.holds` went.

## D128. A committed float operation reads a weak float operand rounded

- **tinygrad:** `uop/ops.py:1053` (`UOp._min_max`), which bounds a weak float
  operand of a committed float operation by its value as written, at no
  width.
- **tolk:** `Ops.rounded` and `Ops.operand_bounds`;
  `test/gen/tinygrad.patch`, which gives tinygrad the
  same bounds (`rounded`, `UOp._operand`).
- **Differs:** a committed float operation that reads a weak float operand
  bounds it rounded to its own type, as the code it compiles to computes it,
  as a committed integer operation already wraps a weak integer one. Rounding
  keeps the order of values, so the bounds rounded hold the values rounded; a
  bound that rounds to NaN, past a type without infinities, leaves the type's
  bounds. tinygrad bounded `where c x 5e-324` at float32 by `5e-324`, which
  rounds to 0, so the bounds excluded 0 and `== 0` of the selection folded to
  false.
- **Reason:** (b). rune's pad of a float32 tensor by a value that rounds to 0
  (`Nx.pad ~value:5e-324`) compiled to such a selection, whose comparison with
  0 folded to false where eager compares 0 with 0.
- **Pinned by:** the Ops suite (`test/uop/ops`): `bounds › a committed float
  selection holds a weak constant at its type, where it may round to zero`.
  On rune: missing (the old rune's `a compiled program › computes eager's
  bits`, whose examples held the pad, went with its rewrite).

## D129. Withdrawn

A search kept nothing of its candidates; merged into D116, which holds what
a search keeps.

## D130. Withdrawn

A round's candidates were timed in their own order, each once its compilation
ended, and on the host's clock all compiled before any was timed, so that a
search chose as a serial one does. Nothing but a golden needed the order, and
no measurement showed the host's searches choosing worse for compilations on
other cores. `Search.beam_search` times each candidate as its compilation ends
(`Worker.iter`), on every device, as `codegen/opt/search.py:133`'s
`pool.imap_unordered` does; on one domain, as without a pool, in their order.

## D132. A contiguous view's rewrite stops at the effects its storage waits on

- **tinygrad:** `uop/ops.py:906-914` (`UOp.contiguous_view`), whose
  rewrite of the index with `pm_mops` and `symbolic` visits every node the
  index reaches, the effects an `AFTER` orders its storage after and the graphs
  they compute included, and returns the node it reaches as rewritten
  (`out.src[0]`).
- **tolk:** `Prepare.pm_stop_at_effects` and `Prepare.contiguous_view`.
- **Differs:** the rewrite stops at void nodes, the effects storage is ordered
  after, and leaves them and what they compute as they are, so that the
  storage it returns is the node of the view's graph. The effects decide
  nothing of where the view's elements lie. The graphs they compute hold
  rune's gathers (D73), which `pm_mops`, a rule for a view's index, moves
  through a reshape into an index whose shape repeats the gather's, and
  arithmetic on weak-integer tensors, on which `Divandmod`'s `divide_by_gcd`
  raises. tinygrad's tensor graphs hold neither.
- **Reason:** (b): the kernel split asks of each copy, stage and store of
  movements or bitcasts of storage whether it is a view of that storage
  (`Schedule.view_of`, through `Schedule.pm_callify_ctx_collect` and
  `Schedule.contiguous_mops_to_view`), which calls `Prepare.contiguous_view`;
  rune calls neither. norn's NUTS warmup and sampling of a posterior over
  sixteen chains, compiled as one function, has results ordered after loops
  whose steps gather by weak-integer positions, and did not compile: it
  raised in `Divandmod`'s `divide_by_gcd`, and, past it, on a reshape of a
  gather's index of four axes.
- **Pinned by:** the `Prepare` suite: `contiguous_view › a view of storage
  after effects is a view of that storage, whatever the effects compute`.

## D133. A greatest common divisor's coefficient is a scalar

- **tinygrad:** `uop/ops.py:1032-1037` (`UOp.gcd`, whose coefficient is
  `uops[0].const_like(...)`, a constant broadcast to the shape of the first
  term) and `uop/divandmod.py:72-76` (`divide_by_gcd`, which divides every
  term by a divisor that is not the `CONST` 1). Over terms with a shape, a
  vector of index expressions or a weak-integer tensor, a divisor of 1 is a
  broadcast 1, which is not the `CONST` 1 and divides no term:
  `UOp.stack(a, b) // 3` raises in `unwrap`.
- **tolk:** `Shape.gcd`.
- **Differs:** the coefficient is a scalar constant of the first term's type,
  whatever the terms' shape, so a divisor of 1 is the `CONST` 1, and a larger
  one divides each term as `divides` does.
- **Reason:** (b): the norn compile D132 names reached `divide_by_gcd`, before
  D132, on a weak-integer tensor of rune's gather positions, and raised there.
  The rule applies to any vector of index arithmetic.
- **Pinned by:** the `Divandmod` suite: `values › each rewrite of a random
  division of vectors keeps each lane's value`.

## D134. Withdrawn

A widening cast of an unsigned `x land y`, or `x lsr k` with `k` within
the operand's width, was the operation on the widened operands, so that
nx.quant's host MXFP4 products unpacked their uint4 codes at uint32, which
Clang vectorises better on arm64. On fudge, kaun's decode bench rows show no
gain beyond their ranges (medians of 11, two interleaved runs each, with the
rule then without): `Quant/host/mxfp4-product-1-token` 0.91 and 0.84 ms
against 0.68 and 0.88 ms, `Quant/host/routed-product-1-tokens` 5.01 and
4.07 ms against 3.85 and 5.28 ms, `Quant/host/routed-product-64-tokens` 130
and 122 ms against 114 and 145 ms. As tinygrad's `uop/symbolic.py:322-325`,
`Shape.symbolic` keeps such a cast where it is.

## D135. Withdrawn

A Metal command buffer signalled through the device's signaler. rig's Metal
driver signals its own (D146).

## D137. Consumers of one domain share their loops

- **tinygrad:** `schedule/indexing.py:225` (new ranges for every stored value)
  and `:245-268` (a node its consumers index apart is stored);
  `schedule/rangeify.py:48` (`remove_bufferize`, which inlines such a store
  back into each consumer) and `:321-336` (`split_store`, one kernel per
  store). Upstream calls scheduling several outputs into one kernel pending:
  `test/runtime/test_assign.py:356`, `:385` and `:398` skip "multi output not
  supported anymore", and `test/runtime/test_jit.py:485` skips "Pending
  multioutput implementation #3607". Codegen already compiles a kernel of one
  end over a group of stores (`test/runtime/test_custom_kernel.py:139`). For
  reductions, `schedule/indexing.py:293-295` (`run_rangeify`: each reduction
  takes new ranges for the axes it reduces); `codegen/__init__.py:163`
  (`merge_reduce_ends`, which already makes one loop of the reductions that
  close one range).
- **tolk:** for stores, `Indexing.region`, `Indexing.views`,
  `Indexing.waits_over`, `Indexing.share_loops` (its `crossed`),
  `Indexing.loops_of` and `Indexing.end_shared` (which
  `Indexing.bufferize_and_index` returns for each store of a group), with
  `Indexing.data_srcs`, `Indexing.create_bufferize_and_index_based_on_ranges`
  and `Rangeify.handle_after`, which take a group (`Ops.is_group`) as its
  stores and no state of a buffer; for reductions, `Indexing.reaches`,
  `Indexing.reduce_loops` and its use in `Indexing.assign_ranges`;
  `Codegen.merge_reduce_ends`; `test/gen/tinygrad.patch`, which gives tinygrad
  the same before the goldens are recorded.
- **Differs:** consumers of one domain take one set of ranges, so a node they
  share is indexed alike and computed once, where tinygrad indexes each
  consumer apart, stores what they share and inlines it back into each. Stores
  and reductions are its two cases.

  Stores: before ranges are assigned, the stores of the tensor graph that have
  a loop are grouped. Two join when their values have one shape and one
  device, they write different storage, neither reads the other or the storage
  the other writes, through anything, neither waits for a write over storage
  that the other reads in a state before it, which one kernel of both could
  run neither before nor after, the computed nodes each reaches before storage
  or another stored value share one, and no computed node that both read
  through views (movements) is read through different views. A node behind a
  view is read at the view's indices, so two stores index it alike only
  through the same views: two slices of one reduction, each stored, stay two
  kernels, each holding its slice alone, whatever else they share. Two groups
  join only when neither reads the other through the stores of a third: where
  `b` reads `c` and `d` reads `a`, a kernel of `a` and `b` and one of `c` and
  `d` would each run before and after the other. Neither may reach a reduction
  the other does not: a kernel of two reductions can lose the optimisations
  each would take. A group's stores take one set of loops, so their shared
  nodes are indexed alike and computed once, and they end as one end over a
  group of the stores, which the kernel split makes one kernel writing them
  all. Outputs that share only loads stay apart; finished calls of one
  domain may still share a launch (D199).

  Reductions: a reduction whose output is indexed by the same ranges as an
  earlier one's in the walk, and that reduces axes of the same sizes, takes
  that reduction's reduce ranges when no reduction of the group reaches it,
  through anything. The walk meets a reduction before those it reads, so only
  a reduction already in the group can read it. The sources are then indexed
  alike: a node they share is one node, computed once per element of the loop,
  where tinygrad indexes it by each reduction's ranges, stores it, and inlines
  the store back into each loop. A group of a loop that several reductions
  close takes a buffer of shared memory for each (D101). Two ranges that
  reductions close merge in every reduction that closes them or not at all,
  as tinygrad merges ranges since df93a6cb8. A reduction can stop closing its group's ranges: a sum
  over a range its source does not read drops it (`pm_reduce_unparented`), so
  a sum over `i` and `k` of a source that reads `k` alone closes `k` alone,
  where a maximum beside it closes `k` within `i`'s loop. Both merge every
  end, a store's as a reduction's. tinygrad gives an end fresh ranges only
  when it closes the same ranges as another at a different depth; tolk gives
  them to every end that closes a range an earlier end closes, so each range
  has one end. No recorded kernel has such ends, so the patch leaves
  tinygrad's `merge_reduce_ends` as it is.
- **Reason:** for stores, (d): `Nx_wide`, whose operations return a double
  word's high and low parts, the two ends of one chain, and rune's gradients
  of several arguments, which share their chain. tinygrad schedules each part
  as a kernel that computes the whole chain: a compiled double-word add ran
  six kernels and runs one, and the old rune's `compile/wide/sum-1e6` row
  compiled 17 kernels. Without the grouping, rune's
  `compile/special/betainccinv-grad-f32` row, the gradient of a float32
  `Nx.betainccinv` in its three arguments, ran 593 kernels in place of one,
  its first call took 29.7 s against 6.1 s, under load on fudge, and it held
  1.3 MB of scratch against none. Through views, (b): rune's fully sharded
  training (RFC 0005's memory bound), whose closing slices one partial into a
  piece for each device; grouped, every slice was live at once. The check
  costs kernel graphs up to 2% more words (tolk's `*/kernel_graph` rows). For
  reductions, (d): rune's reverse mode. The cotangent of a value that
  operations broadcast sums one reduction per operation, each over the same
  cotangent computation: the gradient of an aperture's weights over a 104 x
  104 grid in its radius, which 28 operations broadcast, was one kernel of 28
  loops, each computing the whole cell, and took 647-672 ms compiled against
  93-171 ms eagerly on an M1 Max under load 11-15; one loop takes 30-31 ms.
  On fudge, rune's `Jit/grad-broadcast-radius-float64-104x104`, the gradient
  in a radius that 28 operations broadcast over a grid, takes 215 and 224 us
  as one loop against 383 and 358 us as 28 (medians of 11, two interleaved
  runs each).
- **Pinned by:** for stores, the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › outputs that share a computation` (every test, `slices
  of one product through two views run as two kernels`, `slices that also
  share a computation run as two kernels` and `outputs that read a product
  through one view run as one kernel` among them); its recorded
  `read_around_assign_kernels`, `crossed_pairs_kernels` and `kernel_counts`,
  and the Postrange suite's `where_max_multioutput`, `padto_where_multioutput`
  and `axes` cases, recorded from the equally patched tinygrad. On rune: the
  `compile/special/betainccinv-grad-f32` bench row, and `rune jit › fully
  sharded` for the views (the old rune's `Rune nx.wide › kernels` went with
  its rewrite). For reductions, the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › reductions that share a domain` (every test), and its
  law over generated programs, whose `sum k + max k` and `x - sum k` steps
  cover reductions that share a loop; its recorded `multireduce_*` and
  `ugly_reduceop_pairing` kernel graphs and `kernel_counts`; the Indexing
  suite's recorded `two_consumers` graph; the Postrange suite's
  `sum_and_max_grouped`, `ten_shared_sums_group` and
  `ten_shared_sums_group_unfit` cases, and the Codegen suite's
  `two_grouped_stores_local`, whose upstream axes count one loop for the two
  maximums; all from the equally patched tinygrad. The Codegen suite's
  `reductions that share a range › a range two reductions close at two depths
  runs as two loops`. On rune: missing (went with the old rune's rewrite):
  its Packed suite's bit example of `a compiled call computes eager's values`,
  two views a bit apart whose sums share their loops; its Jit suite's
  `transformations › a compiled gradient of a parameter broadcast over a grid
  is its gradient`; and its bench row
  `reverse/jit-of-grad-clipped-disc-float64-104x104-host`.

## D138. A load's index is simplified without the loads in it assuming its gate

- **tinygrad:** `codegen/late/coalesce.py:41-43` (`simplify_valid_load`),
  which simplifies a gated index with `uop/symbolic.py:354`
  (`uop_given_valid`) down into the loads the index reads.
- **tolk:** `Coalesce.simplify_valid_load`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** each load in the index stands as a variable of its bounds while
  the gate simplifies the arithmetic around it, and is put back after, so a
  load keeps its own index and gate. tinygrad substitutes the gate's bounds
  into the loads too. A load's index whose gate implies the outer gate, such
  as a pad's, loses that gate, and the load, which runs whatever the outer
  gate, reads outside its buffer: the index of a gather through a pad is read
  before the pad's first element. tinygrad's own `gated_given_valid`
  (`uop/symbolic.py:441`) refuses an index that loads for the same reason.
- **Reason:** (b): `Nx.concatenate` of a buffer and an `Nx.take` under
  `Rune.jit`, as nested sampling's evidence concatenates its dead points and
  its live points in likelihood order: the compiled kernel read up to the
  buffer's length before the start of the indices and faulted.
- **Pinned by:** the Coalesce suite (`test/codegen/late/coalesce`):
  `indexing_simplify › a gather through a pad reads its indices only inside
  the pad`.

## D139. Construction checks no specification

- **tinygrad:** `uop/ops.py:196-202` (`UOpMetaClass.__call__`, which checks each
  node it creates against `spec_full` when `SPEC` is above 1, and computes its
  shape when `SPEC` is above 2), and `schedule/rangeify.py`'s
  `Context(SPEC=min(SPEC.value, 2))` around `pm_apply_rangeify`.
- **tolk:** `Ops.v`, `Setting.spec` and `Indexing.run_rangeify`.
- **Differs:** `Ops.v` checks nothing and computes no shape, whatever `SPEC`.
  `SPEC=0` checks nothing, and any other value checks the graphs passed
  between stages, as `SPEC=1` does in tinygrad. A malformed node is reported at the
  end of its stage rather than by the rule that built it;
  `Spec.type_verify ~calls:Enter Spec.full` checks a graph after a suspect
  rewrite.
- **Reason:** (a). Construction is the bottom of the module stack, and the
  full specification reads shapes, which read `simplify`, which builds nodes:
  the check can only be defined above the nodes it checks. Running it from
  construction needs a hook that a later module installs when it is
  initialised, which holds only while every module of the library is linked.
- **Pinned by:** the `Spec` suite: `construction › builds a node that breaks
  the full specification, whatever SPEC`.

## D140. A call of a loop takes its trip as a scalar argument

- **tinygrad:** `schedule/__init__.py:100-112` (`resolve_linear_call`, which
  binds a nameless scalar parameter only to a variable, and substitutes it into
  each kernel that reads it), `engine/realize.py:16-22` (`get_call_arg_uops`,
  which leaves out every variable, and `get_call_kernel_args`, which reads each
  scalar parameter by its name).
- **tolk:** `Schedule.scalar_argument`, `Schedule.pass_values` and
  `Schedule.resolve_linear_call`; `Realize.get_call_arg_uops` and
  `Realize.get_call_kernel_args`; `Tolk_engine.affine`.
- **Differs:** a precompiled call inside a loop may pass a scalar parameter of
  its body a weak integer expression of the loop's ranges, such as `r`,
  `2r + 1` or `n - 1 - r`. The schedule keeps it as the call's argument, no
  state. Each kernel of the body that reads the parameter keeps it, unnamed,
  in a slot past its call's arguments, bounded by its type, and its call
  passes the expression in that slot: `get_call_kernel_args` reads a nameless
  scalar parameter of a program from its call's slot, and `get_call_arg_uops`
  leaves out every scalar argument, an expression as well as a variable. The value reaches the program
  as any variable's does, converted to the parameter's type at the ABI (D28):
  a launch inside a loop of rig's program reads it from a hole of its
  parameters, the sum of the loops' trip words times constant factors
  (D90). The kernel's program depends on neither the range nor its trips, so
  that loops of any range and length compile one. tinygrad has no loop of
  calls, and its calls bind a scalar only to a variable, by name.
  Beam search samples positional scalars by slot and named variables by name.
  Its `link` callback takes a complete call, with the scalar samples passed in
  their slots and the launch sizes evaluated from the same samples; the
  candidate's compiled body keeps its parameters. `Tolk_engine.link_call`
  links that call. tinygrad's `codegen/opt/postrange.py:235` (`args_from_ast`)
  reads every scalar's `expr` and constructs only buffer arguments.
- **Reason:** (b). rune's `Loop.repeat` and staged scans give each step its
  trip's index, which a kernel or a buffer of indices would otherwise compute.
- **Pinned by:** the `Schedule` suite: `create_linear_with_vars › loops of
  calls › a loop's call passes its kernel the trip as a scalar argument`; the
  `Spec` suite: `kernel_graph › loops of calls › accepts a loop's call whose
  scalar argument is an expression of its range`; the `Realize` suite:
  `get_call_arg_uops › is a call's storage arguments in order, without its
  scalar ones` and `get_call_kernel_args › is the call's argument in the slot
  of a scalar parameter`; the `Tolk_engine` suite: `link and run › a loop's call
  reads its trip as a scalar argument on each trip`, `› a back edge's call
  reads its trip on each trip, at most its trips`, `› a loop of one trip passes
  its call the trip as a scalar argument`, `› loops of any range and trips
  around one call compile one program`, and the same two on each GPU;
  `Tolk.Search › a search binds a positional size in its call and launch`;
  `Tolk.Search on the host › a search runs candidates with named and
  positional scalars`. On rune: missing (the old rune's `Rune.jit › keys › a
  scan searched with beam 2 draws as eager does` went with its rewrite;
  `rune jit › jit › a function searched at a width` searches no loop).

## D141. The host computes upcast lanes as vectors

- **tinygrad:** `codegen/__init__.py:116` (`do_devectorize`, which splits
  every elementwise operation, load and store of several lanes into one per
  lane, on every target);
  `renderer/cstyle.py:40`, which writes every conversion as a C cast, for a
  vector a reinterpretation of its bits.
- **tolk:** `Renderer.t`'s `vector_alu`; `Codegen.constant_lanes`,
  `Codegen.in_order`, `Codegen.vector_operation`, `Codegen.distinct_lanes`,
  `Codegen.vector_dtypes`, `Codegen.do_devectorize`,
  `Codegen.address_lanes`, `Codegen.pm_vector_constants`,
  `Codegen.product_operands` and `Codegen.first_lane`;
  `Cstyle.mask_widths`, `Cstyle.clang_vectors` and `Cstyle.clang_kernel`;
  `Decomp_dtype.emulates`; `Uop_weak.pm_commit_weak`'s cast over a stack;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded, and makes its compiled patterns match a repeated source
  lazily, as its interpreted ones do: a stack whose first lane is a constant
  and whose others are lanes of a vector raised in a bind of the first lane's
  source that the match never needed.
- **Differs:** a renderer may compute on vectors (`Renderer.vector_alu`;
  Clang does). There an elementwise operation of several lanes on a type the
  target computes natively stays one operation on a vector of its lanes, on
  one axis, when each source holds them in order: a vector
  operation, constants in any view (a literal), a stack of loads of memory, a
  load in as many lanes as read it, and of lanes of vectors, each in one
  lane, or a scalar that every lane reads, and when no two of its lanes read
  the same sources: a lane of a vector operation is its own, a lane of a
  stack its element. A source that holds computed lanes in
  another order, a broadcast or a permutation across lanes, or apart in
  registers, as an accumulator's lanes, costs a vector shuffles that a scalar
  reads for free, and lanes that repeat are computed once each apart: such an
  operation computes each lane apart. Loads, stores, tensor-core products,
  calls, and operations on a weak type (an address) or an emulated one still
  compute each lane apart; an address is computed lane by lane through the vector
  operations it reads, so that accesses still merge. A sum fuses each lane of
  a vector product into its running sum, and computes its first lane, which
  it adds to nothing, as a product of scalars, so that no vector product is
  left; a selection by constant conditions picks each lane. Clang writes the vectors as `ext_vector_type`
  values: a conversion is `__builtin_convertvector`, square roots,
  truncations and multiply-adds are `__builtin_elementwise_*`, a scalar
  operand of a builtin or a selection is splat, and a vector of booleans is a
  mask of -1 and 0 lanes as wide as the values it compares, converted to the
  width of the masks or values it meets; a lane of a mask is read as a
  `_Bool`, and a mask of one-byte lanes, which C types as plain chars, as
  signed chars. A cast of a stack of weak constants commits them at the
  cast's type, as tinygrad commits the weak sources of a cast of an
  operation, so that the vector is of constants of that type, as each lane
  would be alone (`uop/weak.py:41`). Which lanes a host kernel takes is
  D188's.
- **Reason:** (d): RFC 0025's special functions on the host. Lanes copied as
  scalars made Clang compile each lane's copy of the body, in a time that
  grows faster than the copies, and kept its loop vectorizer off a float32
  loop: float64 `log_betainc` compiled in 36 s and its gradient in `a` in
  174 s, and float32 `exp`, `erf` and `sin` replayed at 0.23-0.71 of their
  speed without the upcast. As vectors, Clang compiles the body once and runs
  its lanes at once, and each lane computes what the scalar kernel computes,
  bit for bit. Timed on one core of an M1 Max (10^5 elements, kernels
  alternated), against no upcast: float32 `exp`, `erf`, `erfc`, `erfinv`,
  `ndtri` and `tanh` replay 1.03-1.50 times as fast, float64 `exp`, `erf`,
  `erfc`, `erfinv`, `ndtri` and `tanh` 1.14-1.56 times, and float32 `sin`,
  whose reduction multiplies 64-bit integers, which NEON lacks, 0.96 times.
  Rune's compile bench rows `shaped-4k` and `shaped-4k-3-lanes` time host
  compiles that the vectors decide. A stack of weak constants committed
  apart from its cast took the narrowest type that holds them, and a
  complex128 signalling NaN stored as two doubles of a vector became a
  conversion of floats, which quiets it (e).

  nx.quant's MXFP4 product unpacks a byte of uint4 codes into two lanes, a
  load each lane of the pair reads, shifted by a broadcast of the constants 0
  and 4. Refused as a vector, the shift and every operation after it went lane
  by lane: Clang compiled each code's e2m1 decode as a scalar select, a branch
  on x86 that random codes, as a checkpoint's are, mispredict, which made the
  product 2.5-4.5 times slower on random codes than on zero codes (kaun's
  decode bench, kimchi). A load fills each lane that reads it, as a literal
  fills its lanes, at no cost beyond the stack of distinct loads already
  taken, and the decode, a vector, selects without branches. A load that
  fills lanes the other sources fill alike repeats work instead: Q6_K's scale
  byte serves 16 values, and converting it in each of their lanes made the
  dense product's x86 kernel 45% longer and 33% slower on kimchi, so such
  lanes stay apart, where each is computed once. With D188, the
  host's products of a checkpoint's codes run 1.7-3.6 times as fast for one
  token's routed product and 2.4-3.3 times for a dense one, on an M1 Max
  (alternated under load). On fudge (medians of 11, two interleaved runs
  each), kaun's `Quant/host/mxfp4-product-1-token` takes 0.91 and 0.84 ms
  against 5.03 and 4.53 ms with a load filling one lane, and 6.20 and 5.62 ms
  with every lane apart; `Quant/host/routed-product-1-tokens` 5.0 and 4.1 ms
  against 15.3 and 21.0 ms, and 38.6 and 27.2 ms. Every lane apart, rune's
  `Jit/sin-float32-64K` takes 806 and 139 us against 82 and 74 us, the tolk
  bench's `lorenz/r_128_16_3_4_4_3_3_400_400/heuristic` 3.2 and 1.8 ms
  against 0.79 and 0.63 ms, and the compile bench's `trig/sin-nested-8` 1.98
  and 1.91 s against 0.74 and 0.46 s (medians of 3).
  A matmul's upcast products read their loads, or
  the lanes of a loaded vector, in each lane's multiply-add
  (`clang_matmul_upcasted*`). They computed a vector product of their shared
  loads for its first lane alone, the one the sum adds to nothing, and read
  the other lanes back from the vectors of those loads; Clang compiled both
  to the same instructions, and building them cost tolk's passes: as one
  vector, a conv kernel's 9 unrolled lanes made the conv rows' linearize,
  render and warm stages 19%, 26% and 17% slower.

  Any number of lanes is one vector, which LLVM computes in the next power of
  two. Computed lane by lane, Clang compiles each lane's copy of a body, in a
  time that grows faster than the copies: jera's compiled Milstein march over
  3 elements fuses into one kernel of 3 lanes, whose test compiled in 57 s
  cold, and in 14 s as one vector.
- **Rejected:** host vectors of a power of two of lanes only, made for Ubuntu
  24.04's Clang 18.1.3, which crashed on x86 compiling rune's sort of a uint4
  [2; 3] view (D167): m·2^k lanes as m vectors of 2^k, 6 lanes as 3 of 2, and
  an odd number each lane apart. The crash was not from its 6-lane vector,
  and the split's other reason, speed, does not hold: on fudge, quiet, taskset
  0-5, lorenz/step-batch-128-host replays in 7.23 and 7.25 ms with its 48
  lanes as 3 vectors of 16, and 7.23 and 7.23 ms as one vector. An odd
  number as vectors of the powers of two that sum to all but its last lane,
  and that lane apart, 5 lanes as 4 and 1, made eight_schools' kernel
  `E_8_5_2` take Clang 8.9 s, all in LLVM's loop strength reduction, against
  0.2 s, and its first call 13.6 s against 8.8 s.

  Compile time is paid once and replay on every call, so
  neither trade below was taken. `#pragma clang loop vectorize(disable)`
  changes nothing on vector kernels (Clang 0.24 s with and without it on
  float64 `erfinv`, replay 0.99-1.00 times on 9 kernels); `-fno-vectorize`
  on scalar kernels cuts the gradient of `log_betainc` from 7.68 s to 1.36 s
  of Clang but slows its replay from 204 ms to 475 ms (D167 takes it for a
  source of more than 400 000 bytes). One 16-byte register
  of lanes at 5 operations or more saves at most 2.1 times of Clang (that
  gradient, 6.35 s to 3.05 s) and replays 1.17-1.79 times slower (the
  gradient 1.17, float64 `sin` 1.28, float32 `lgamma` 1.79).
- **Pinned by:** the Cstyle suite's `sources › by default › clang_vector_*`,
  from the patched tinygrad, and `execution on the host › every kernel the
  interpreter runs writes what it computes`; the Codegen suite's `vectors in
  programs › a host program computes on vectors of its lanes` (every host
  case), `› a host program of 6 lanes computes one vector`, `› a host program
  of 48 lanes computes one vector`, `› a host program of 24 lanes computes
  one vector`, `› a host program of n lanes computes one vector` and `› a
  host sum of a product's unrolled lanes multiplies no vectors` (every case),
  `› a host selection of 8, 16 or 32 lanes by a
  comparison of integers takes each lane's arm` and `› a host selection of 3,
  6, 12, 24 or 48 lanes by a comparison of integers takes each lane's arm`
  (every case), `› a weak constant stored into four lanes of half on the host
  converts them` and `multiply-adds` (every test);
  `Tolk.Uop_weak › pm_commit_weak › a cast of a stack of weak constants
  commits them at its type, signalling NaN bits kept`;
  rune's `Lower › padded loads give eager's bits` (a complex128 signalling NaN
  fill); rune's `rune jit › classes › a uint4 sort along the last axis of every
  other row and column, flipped, gives eager's codes` and `› nx.quant's
  decodes compile`; rune's compile bench: `shaped-4k-3-lanes`. On rune,
  missing (went with the old rune's rewrite): its Rune.jit suite's `lanes`,
  `stated values › a window set into bits at an odd offset keeps the bits
  around it`, and lower_index's `quantised products › a product decodes a
  checkpoint's codes 16 at once on the host` (with D188) and `› a product
  over GGUF's Q6_K blocks converts each scale once`. The
  Codegen suite's `invalid_lanes_clang` and `invalid_lanes_int8_clang`, whose
  lanes of one constant compute once.

## D142. Withdrawn

The host ran a run of its calls as one host program, which called each through
nx.device's entry. Each call is a launch of rig's host, consecutive ones one
submission, and a range a loop of rig's program (D146).

## D143. Withdrawn

The host unrolled a reduce beside a small unrolled axis to a vector; merged
into D188, which holds the host's vector width.

## D144. Withdrawn

Reductions of one domain that read none of each other shared their loop;
merged into D137, which holds the loops that consumers of one domain share.

## D145. An integer division on padding divides by 1 there

- **tinygrad:** `schedule/indexing.py:98-103`
  (`convert_pad_to_where_to_keep_behavior_local`) and `:169`
  (`apply_movement_op` on a `PAD`, which makes the source's ranges invalid on
  the padding); `codegen/opt/postrange.py:127-143` (`PADTO`, which gates the
  loads of the padded axis). Either way the source is computed on every lane,
  and its loads read zeros on the padding.
- **tolk:** `Indexing.constant`, `Indexing.guard_division_on_padding` and
  its rule in `Indexing.pm_apply_rangeify`; `Postrange.Scheduler.apply` (its
  `pad_division`, in `Padto`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** a `TRUNCDIV`, `TRUNCMOD`, `DIV` or `MOD` of integers divides by
  `where(valid, d, 1)` in two places. Rangeify guards a division whose ranges
  are invalid somewhere, unless its divisor is a nonzero constant through casts
  and movements: a stored divisor's bounds say nothing of the zeros its padding
  loads. A range's validity is read after the symbolic rules lift its gates to
  the top and merge them: a shrink after a pad leaves the gate below an
  addition, and a pad of a pad nests one gate in another. `PADTO` guards a
  division whose divisor reads the padded range. Where the divisor is a gated
  load, the selection becomes the load's alternative, so the padding loads 1.
  tinygrad divides by the zeros its padding loads.
- **Reason:** (b). A division by zero traps on x86 (arm64 gives 0), and C
  leaves it undefined everywhere. tinygrad's scalar C evaluates such a division
  lazily only while it has one use, inlined into the selection that drops it;
  with more uses it computes the quotient into a variable before the selection
  or the store's gate, and traps too. The host's vector lanes (D141) divide on
  every lane. `Norn.Nuts`'s compiled transition concatenates `parents + next /
  spans` between two constants, and its kernel divided the padding's lanes by
  zero: `test_nuts`, `test_calibration` and `test_posteriordb` died with SIGFPE.
- **Pinned by:** the `Tolk_engine` suite (`test/engine/tolk_engine`):
  `link and run › a pad of integer divisions writes what its tensors compute`,
  over `pad_idiv`, `pad_idiv_stored` (a divisor of at least 1, stored),
  `pad_idiv_nested` (a quotient padded twice) and `pad_idiv_shrunk` (a quotient
  shrunk, then padded); the `Codegen` suite's `on the host, a program writes
  what its kernel writes › padto_idiv_upcast`. Each traps on x86 without the
  rule; arm64 does not trap. The Schedule suite's recorded `pad_idiv*` graphs
  and the Codegen suite's `padto_idiv_upcast` goldens, from the equally patched
  tinygrad; norn's `Norn.Nuts › reproducibility › a compiled transition is
  the eager one` on x86.

## D146. The runtime is rig: no command queues, batches or host batches

- **tinygrad:** `runtime/support/hcq2.py` (`hcq_compile`, `sched_batches`,
  the queue encoders it calls through `Compiled.pm_encode`, the host
  programs that submit them, the staging copies and `hcq_link`);
  `runtime/ops_*.py` (`pm_encode` and the queues);
  `engine/realize.py:275` (`compile_linear`, which ends in `hcq_compile`)
  and `:284` (`run_linear`).
- **tolk:** `Realize.compile_linear`; `Tolk_engine.link`, `Tolk_engine.run`
  and `Tolk_engine.time`; `test/gen/engine/jit.py`,
  which stops tinygrad's `compile_linear` before `hcq_compile`.
- **Differs:** `compile_linear` ends once the kernels are compiled: it batches
  no call, encodes no queue, makes no host program, splits no call into
  lanes and stages no copy. `Tolk_engine.link` turns the compiled calls into
  a program of rig (`Rig_program`): a kernel is a launch on each lane's
  device, a host program's on its host, its parameters laid out as
  `hcq2.py`'s `layout_args` lays them out; a copy is a copy part of a device
  that copies between its two sides, or else a move, a queued copy of rig's
  (`Rig.Buffer.submit_copy`); a range or a back edge is a loop; and
  consecutive calls of one device are one submission. rig's drivers encode each vendor's queues, order the work
  across queues and devices by the memory it uses, stage what no device
  copies directly, and time the submissions where the driver can. A host
  program's entry is rig's (D28). Storage the schedule makes is the
  program's, or an input each run makes where it is linked per run (D173),
  and storage a caller binds is an input of each run.
- **Reason:** (c). rig owns each GPU's queues: two runtimes cannot share a
  GPU, and memory belongs to the runtime that made it.
- **Pinned by:** the Tolk_engine suite: `link and run › recorded`, `GPUs`
  (each recorded program, loop and refusal on Metal, CUDA, NV and AMD where
  the machine has one), `lanes and words` and `runs`; the jit goldens
  (`test/engine/jit/*_lowered.golden`), which `gen/engine/jit.py` makes
  without batches.

## D147. The x86 host holds bfloat16 as its bits

- **tinygrad:** `renderer/cstyle.py:300` (`ClangRenderer.type_map`,
  which keeps `CStyleLanguage.type_map`'s bfloat16, Clang's `__bf16`
  (`:146-150`), on every machine).
- **tolk:** `Cstyle.clang` (the Clang renderer's language,
  which on x86 maps bfloat16 to `unsigned short` and renders its constants as
  their bits with `pm_bf16_ushort_const`).
- **Differs:** on x86, a bfloat16 value, pointer or vector is written as
  `unsigned short`, and a bfloat16 constant as its bits, as OpenCL writes
  them (`renderer/cstyle.py:342,345`: its type map's `ushort` and
  `pm_bf16_ushort_const`). The
  kernel's graph is tinygrad's; only its C
  types change. Every bfloat16 operation already computes in float32 through
  `create_non_native_float_pats` and `pm_manual_bf16_cast`, so a bfloat16
  value is only loaded, stored, selected and bitcast, which act on its bits.
  `gen/tinygrad.patch` makes the same choice in tinygrad's Clang renderer on
  x86, so the goldens state tolk's sources.
- **Reason:** (b). LLVM's x86 backend without AVX512-BF16 legalises a
  bfloat16 value it only moves through float32. It aborts on a selection
  between vectors ("Cannot select: v8bf16 = X86ISD::BLENDV"), and it truncates
  a merge of a gated load with its alternative, and a bitcast of stacked
  lanes, with compiler-rt's `__truncsfbf2`, which rig's host linker finds in
  no library of the process. rune's bfloat16 suites failed on both x86 hosts:
  `compiled dtypes › bfloat16, compiled, carries hold each lane's bits`, the
  bfloat16 cases of `Rune.map_segments`' `dtypes`, `Rune.quant`'s routes in
  blocks and `transcendental functions on the host › pow › bfloat16`.
- **Pinned by:** the `Codegen` suite's `a selection between bfloat16 vectors
  takes each lane's arm` and `a gated bfloat16 load takes its alternative
  where its gate is false`, which run on the host: on an x86 host with clang
  19, the first aborts the compile and the second fails to link (on a CPU
  without AVX512-BF16) under `__bf16`. The x86 Clang goldens of bfloat16
  kernels (`bf16_clang`, `clang_*_bf16`). On rune: missing (the cases above
  went with the old rune's rewrite).

## D148. A selection keeps its guard of a conversion or a division

- **tinygrad:** `uop/symbolic.py:240-247` (`fold_where_closure`, which
  assumes a selection's condition true in its true branch and false in the
  other, unless an `INDEX` is involved).
- **tolk:** `Shape.guards_partial` and `Shape.fold_where_closure`.
- **Differs:** the condition is not assumed in the branches where it reaches
  the operand of a conversion of a float to an integer or of an integer
  divisor in one of them. tinygrad folds `where c (cast (where c q 0)) f` to
  `where c (cast q) f`, and `where c (x // where c d 1) 0` to
  `where c (x // d) 0`. `gen/tinygrad.patch` makes the same exception, and no
  golden changes.
- **Reason:** (b). C leaves a conversion of a float outside the integer's
  range undefined, and a division by zero traps on x86. Clang computes an
  operation in every lane of a vector and selects between vectors with masks,
  which keep an undefined lane the selection discards; LLVM then folds it into
  any value. rune guards such an operand, as the quadrant of a sine's
  reduction, by a selection on the condition that discards it, and the fold
  removed the guard: the compiled sine of `[0; inf]` on x86 read
  `0x1.c9e8dep+97` for `sin 0`.
- **Pinned by:** the `Symbolic` suite's `selections › a guard of a conversion
  to an integer is kept in the branches` and `› a guard of an integer divisor
  is kept in the branches`. On rune: missing (the old rune's `a compiled
  program › computes eager's values within the stated rounding`, whose third
  example was that sine, went with its rewrite).

## D149. Withdrawn

The Clang renderer wrote no calling convention on Windows, where tinygrad wrote
`__attribute__((ms_abi))` on its kernels and the function pointers its calls
cast to. tinygrad no longer supports Windows: `ClangRenderer`
(`renderer/cstyle.py:287`) keeps the empty `abi` and plain `kernel_typedef`
of `CStyleLanguage` (`:131-132`) on every system, as `Cstyle.clang_lang`
does. A Windows host's code is compiled for Windows' convention by D178.

## D150. The folder's sine, logarithm and powers are correctly rounded

- **tinygrad:** `uop/ops.py:1366-1379` (`python_alu`'s `SIN`, `math.sin`;
  `LOG2`, `math.log2`; `EXP2`, `safe_exp2`'s `2 ** x`; `POW`, `safe_pow`'s
  `pow`: the C library's), and `uop/symbolic.py:201` (`c ** x` as
  `exp2 (x * math.log2 c)`).
- **tolk:** `Ops.sine`, `Ops.py_log2`, `Ops.py_exp2` and `Ops.py_pow`, which
  `python_alu` calls, and the `c ** x` rule of `Shape.symbolic_simple`, which
  takes `log2 c` from `Ops.exec_alu`; `test/gen/tinygrad.patch`, which gives
  tinygrad's `python_alu` a correctly rounded sine, logarithm and powers
  (`cr_sin`, `cr_log2`, `safe_exp2`, `safe_pow`: mpmath's, at a precision
  doubled until it rounds to one double, a power that is a fraction exact) and
  its `c ** x` rule `cr_log2`, before the goldens are generated.
- **Differs:** `exec_alu` computes each on integers and rounds once. A sine's
  argument is reduced exactly by pi/2, held to 1200 bits, and the sine or
  cosine of the remainder is summed in fixed point to 192 bits. A logarithm,
  a power of two and a power are summed in fixed point (`atanh` and `exp`
  series) with a bound on their error, at 64 bits, then 128, 256, ..., until
  both ends of the bound round to one double; a power of two `2^(e y)` and a
  power that a double or a midpoint between two can equal are computed
  exactly. Each result is correctly rounded on every host, where tinygrad's is
  the C library's: on an M1 Mac about 3% of drawn float64 sines are one unit
  off (`sin 2.5`, `sin 4.0`), glibc's about 0.1% (`sin 0.259`), and mingw-w64's
  reduces by x87's 66-bit pi; glibc 2.41's `log2` is one unit off for about
  one operand in 320 of [0.5, 2), its `pow (2, x)` for one in 1,250 of
  [-1, 1), and its `pow` for one in 1,100 of [0.5, 8) to powers in [-2, 2)
  (200,000 uniform draws each; `log2 0.86`, `2^-0.338`, `5.4 ** 0.8`). A NaN
  operand of the sine gives `Dtype.nan`.
- **Reason:** for the sine, (b). rune's `sin` and `cos` are `SIN` nodes
  (rune's `Lower_real.sin_`), which fold through `exec_alu`, and a fold must
  not depend on the host: on Windows `sin 1e300` folded to `0.979`, of the
  wrong sign. For the logarithm and the powers, (e): `exec_alu` computes its
  operation exactly and rounds once, and a C library's value one unit off is
  the wrong value, which also differs between hosts. rune's `exp`, `log` and
  their kin compose `EXP2` and `LOG2` nodes (rune's `Lower_real`), within one
  unit of the correctly rounded result either way.
- **Pinned by:** `Tolk.Ops › exec_alu › exec_alu_values.golden`, compared
  exactly: its `SIN` rows include `1e300`, `3.4e38`, pi, floats near multiples
  of pi and floats whose sine glibc rounds to the wrong neighbour, which fail
  with the C library's sine on macOS, glibc and Windows; its float64 `LOG2`,
  `EXP2` and `POW` rows include operands whose results glibc rounds to the
  wrong neighbour, which fail with glibc's, logarithms near 1, of subnormals
  and of an integer past the greatest double, powers of two that round to a
  subnormal or to 0, powers with exponents near `2^60`, and exact powers, one
  of them halfway between two doubles (`68718952449 ** 1.5`, `262143^3`);
  `Tolk.Symbolic › tinygrad › tests.golden ›
  TestWeakConstFolding.test_float_unaries`, which folds `sin 4.0`; and
  `Tolk.Symbolic › symbolic_simple › powers › c ** x multiplies x by log2 c
  correctly rounded`, at `c = 0.86`.

## D151. An x86 kernel declares that it uses 512-bit vectors

- **tinygrad:** `renderer/cstyle.py:132` (`kernel_typedef`, `void`) and
  `:327` (`ClangRenderer.__init__`).
- **tolk:** `Cstyle.clang` (the Clang renderer's language on
  x86, whose `kernel_typedef` is
  `__attribute__((min_vector_width(512))) void`); `gen/tinygrad.patch` makes
  the same choice in tinygrad's Clang renderer on x86, so the goldens state
  tolk's sources.
- **Differs:** on x86, every kernel is marked as a function that uses
  512-bit vectors. On a processor with AVX-512, 512-bit types are then legal
  even where the processor prefers 256-bit vectors, as Intel's do: LLVM
  holds a vector of up to 512 bits in one register and splits a wider one
  into 512-bit parts. A kernel whose vectors are no wider than 256 bits,
  scalar loops among them, compiles to the same instructions, and on a
  processor without AVX-512 the mark changes nothing.
- **Reason:** (b). The renderer writes vectors wider than 256 bits: 32 halves
  for an elementwise float16 kernel (D141). Clang 18 crashes (a segfault in
  `DAGTypeLegalizer::SplitVectorResult`) splitting a negation, an absolute
  value or a copysign of more than 16 halves for a processor with AVX-512
  that prefers 256-bit vectors, with or without AVX512-FP16. Such a vector is
  what `Rune.jit` writes for `Nx.neg` and `Nx.abs` of float16 and for a
  float16 minimum, which negates its lanes: rune's `Rune.jit programs › computes
  eager's values within the stated rounding` crashed Ubuntu 24.04's Clang
  18.1.3 under `-march=native` on a CI runner. Clang 17 and Clang 19 and later
  compile them without the mark. Rendering the negation otherwise does not
  hold: LLVM folds a round trip through float32, a subtraction from `-0.`, a
  product by `-1` and a flip of the sign bit back into the same negation of
  halves. tinygrad computes each lane apart and writes no such vector.
- **Pinned by:** `Tolk.Compiler_cpu › features › a kernel over more than 16
  halves compiles for AVX-512`, which renders `half_neg`, `half_abs`,
  `half_copysign` and `half_min` (32 lanes each) for
  `skylake-avx512`, `icelake-server` and `sapphirerapids` and fails under
  `CC` naming Clang 18 without the mark; the x86 Clang goldens, whose kernels
  carry it. On rune: missing (the old rune's `Rune.jit programs › computes
  eager's values within the stated rounding`, whose examples held the program
  found, went with its rewrite).

## D152. The disk cache holds at most CACHESIZE bytes

- **tinygrad:** `helpers.py:411-442` (`diskcache_clear`, `diskcache_get`,
  `diskcache_put`): a put inserts or replaces a row and nothing removes one
  but `diskcache_clear`.
- **tolk:** `Helpers.Diskcache.mark`, `Helpers.Diskcache.size_file`,
  `Helpers.Diskcache.count`, `Helpers.Diskcache.trim`,
  `Helpers.Diskcache.make_room`, `Helpers.Diskcache.put` and
  `Helpers.Diskcache.clear`; `Setting.cachesize` (5 GiB by default, `0` for
  no bound).
- **Differs:** a put that would take the cache's files past `CACHESIZE`
  bytes first removes entries of every table and version, the least recently
  used by file time first, until they would hold four fifths of it with its
  entry; it removes them before it writes, so that on a full disk the write
  finds their space. A put of an entry larger than the bound writes nothing.
  It also removes the writes that killed processes left: those last written
  an hour or more before, the age at which ccache removes its temporary
  files, so that another process's write in progress stays. A hit sets its
  entry's time through a descriptor shared for deletion
  (`lib/helpers_stubs.c`, `futimens` or `SetFileTime`), and on Windows a put
  renames its entry with POSIX semantics (`tolk_rename`), which replace a
  file open shared for deletion, so that a put over an entry a read or a
  mark holds meanwhile still wins. Processes count each other's puts in the
  length of one file, to which each put appends a byte per KiB. `diskcache_clear`
  removes every entry; `clear` keeps a write in progress, an entry or size
  it cannot remove, as on Windows one another program holds open without
  sharing deletion, and the tables' directories, which a put meanwhile
  writes in.
- **Reason:** (b). Programs, schedules and beam results are keyed by the
  digest of tolk's sources (D63), so each change of the sources leaves the
  previous build's entries unread, about 0.6 GB for the tests of tolk and
  rune. Every raven process shares the cache under `~/Library/Caches/tolk` or
  `~/.cache/tolk`, which reached 96 GB on one machine; a full disk fails the
  put of a compiled binary (`Renderer.Compiler.compile_cached`) and so the
  compile.
- **Pinned by:** `Tolk.Helpers › Diskcache › the files of the cache never
  hold more than cachesize`, `› a put past cachesize removes entries used
  least recently first`, `› a put past cachesize removes the writes a killed
  process left`, `› a put past cachesize keeps another process's write in
  progress`, `› a put that cannot write removes entries past cachesize
  first`, `› a put that cannot count the cache's size stores its entry`,
  `› clear leaves the entries it cannot remove`, `› clear keeps the
  directories of the tables`, `› a put replaces an entry that a read holds
  open`, `› clear keeps another
  process's write in progress`, `› processes putting in turn keep the cache
  within cachesize`, `› a cachesize of 0 keeps every entry` and `› a
  negative cachesize is refused`; `Tolk.Helpers › Diskcache on two domains ›
  behaves as a table of entries, whatever the order of the calls`.

## D154. A range whose size may be 0 runs no time

- **tinygrad:** `codegen/late/linearizer.py:20` (`run_count`, the product of
  `vmax + 1` over a node's ranges) and `:33` (the key, by run count, then
  depth) and `uop/symbolic.py:402-412` (`reduce_mul_chain`, an integer
  maximum of a product times the factor that ignores its ranges).
- **tolk:** `Linearizer.runs` and `Symbolic.reduce_mul_chain`;
  `gen/tinygrad.patch` makes the two changes in tinygrad, so the goldens
  state tolk's order.
- **Differs:** the linearizer counts a range as at least one run. Both
  order nodes by their run count, then by their depth, the number of ranges
  they run inside, so a loop's body sorts after the nodes around the loop
  that run as often. tinygrad counts a range whose size is at most 0 as no
  run, its body sorts before the nodes around the loop, and a sum's first
  store of its accumulator is placed inside the loop: where the loop runs no
  time, the sum reads its accumulator unwritten. An integer maximum of a
  product by
  a factor that ignores its ranges, over a range whose size may be 0, is its
  identity there too: tinygrad's is the identity times the factor, which
  wraps. The order of the goldens'
  `empty_range_runs_none` changes: the nodes outside its range of size 0 come
  before it.
- **Reason:** (c). `Tolk_engine.run` binds a variable to any value of its
  range, its lower bound included, and runs each range's trips; a size of 0
  is a binding the schedule must compute. With concrete shapes the scheduler
  folds an empty value first (`Prepare`), so only a variable reaches it.
- **Pinned by:** `Tolk.Codegen › reductions over a variable size`, which runs
  sums and maxima over sizes of at most 1 and 3 on the host at every size,
  and an int32 maximum of `2 k` among them;
  `Tolk.Linearizer › laws › after splitting and chaining, the rest runs
  outside each loop`, whose kernels hold ranges of constant sizes from 0 and
  of variable sizes; the linearizer's `orders_output` goldens.

## D155. A pad by -0. fills its padding with -0.

- **tinygrad:** `mixin/op.py:312` (`_pad_constant`) and `:318` (`pad_to`),
  which return the plain pad when `value == 0`, true of `-0.`.
- **tolk:** `Shape.is_zero`, used by `Shape.pad` and `Shape.pad_to`.
- **Differs:** a fill is left to the pad's own zeros only when its bits are
  all zero. A `-0.` fill takes the selection off the padded mask, so the
  padding holds `-0.` where tinygrad's holds `+0.`.
- **Reason:** (b). nx's `pad` fills with the value given, `-0.` included, and
  rune lowers a padded load (rune's `Lower.padded`) through
  `Shape.pad` with that fill, so its compiled pad must compute eager nx's
  bits.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `signed zeros ›
  Shape.pad by -0. computes -0. on the padding`.

## D156. A NaN constant is written from its bits

- **tinygrad:** `renderer/cstyle.py:25-26` (`base_rewrite` writes every NaN
  constant as the language's `nan` cast to its type), `:152` (`NAN`), `:295`
  (Clang's `__builtin_nanf("")`), `:462` (CUDA's
  `#define NAN (__int_as_float(0x7fffffff))`) and `:524-526` (HIP's 8-bit
  floats, `f32_to_fp8(NAN, …)`).
- **tolk:** `Cstyle.nan_const`, the `bitcast` of each language
  (`Cstyle.cstyle`, `Cstyle.metal_lang`, `Cstyle.cuda_lang`) and
  `Cstyle.cdna_rewrite`;
  `test/gen/tinygrad.patch`, which renders tinygrad's NaN constants the same
  way before the goldens are generated.
- **Differs:** a NaN constant (D27) is written as the language's bitcast of
  its bits, an unsigned literal of its width:
  `__builtin_bit_cast(float, (unsigned int)(0xffc00005u))` in C and HIP,
  `as_type<float>(…)` in Metal, `tg_bitcast<float>(…)` in CUDA. So a program
  stores the constant's sign and payload, a signalling NaN staying one, where
  tinygrad's stores the conversion of one NaN of the language: a float64 NaN
  with bits `0x7ff0000020000000` is stored as `0x7ff8000000000000` by Clang,
  and every float32 NaN as `0x7fffffff` by CUDA. An infinity is written as
  before.
- **Reason:** (b). Under the NaN rule a move keeps a NaN's bits. rune lowers
  every float constant from the bits nx stores for it (rune's `Lower.padded`
  fills a padded load with its fill's bits through `Shape.pad`), and padding
  moves those bits into the result, so a compiled kernel must store the bits
  eager nx stores, as it must store the bits of a constant folded from a
  bitcast (D27).
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `a NaN constant
  is stored with its sign and payload › *`, which runs a store of NaNs of both
  signs, quiet and signalling, of half, float and double on the host; and the
  `*_inf_nan_*` goldens of the `Cstyle` suite, for every renderer.

## D157. A C-style expression nests at most 64 brackets

- **tinygrad:** `renderer/cstyle.py:259-264` (`_render`'s rule that writes a
  value read once into its reader).
- **tolk:** `Cstyle.max_nesting`, `Cstyle.nesting`, and `inlined` in
  `Cstyle.render_uops`, with its call there.
- **Differs:** a value read once, which both write into its reader, is
  declared as a local variable when its expression nests more than 64
  brackets, counting `(`, `[` and `{`. tinygrad bounds no depth, so a chain of
  single-use values nests without limit, and Clang and Metal reject a statement
  past 256 levels: "bracket nesting level exceeded maximum of 256". The limit
  is Clang's `BracketDepth` (`LangOptions.def`, default 256); HIP compiles
  through comgr, which is Clang, and its limit is unverified on hardware;
  NVRTC 13 compiles 1000 levels. tinygrad met the error and removed
  parentheses of associative chains instead, which a chain of `<<` and `%`
  cannot use. No golden nests past 24 levels, so none changes and
  `gen/tinygrad.patch` keeps tinygrad's rule.
- **Reason:** (b). rune's float64 `fmod` (rune's `Lower_real.fmod`)
  computes `r := (r << left) % my` 187 times, which written as one
  expression nests about 374 levels, so its kernel did not compile on the
  host: rune's `binary kinds, comparisons, where and fma give eager's bits`
  (float64) and `binary › mod` failed.
- **Pinned by:** the Cstyle suite: `parentheses › a chain of any length nests
  a bounded number of brackets`, which renders chains of 1 to 300 steps with
  each renderer, and `execution on the host › computes a chain too deep for
  one expression`, which runs 300 steps on the host against a fold in OCaml.

## D158. A dead read of a staged value is 0

- **tinygrad:** `schedule/rangeify.py:95-97` (`remove_bufferize`), which
  inlines a stage's value at a read and leaves each range whose index is
  `Invalid` in place.
- **tolk:** `Rangeify.remove_bufferize`;
  `test/gen/tinygrad.patch`, which makes tinygrad's rule the same before
  the goldens are generated.
- **Differs:** a read of a stage that would be inlined, at indices of which
  one is `Invalid`, is 0, as a load at an `Invalid` index is. tinygrad inlines
  the stage's value there with the stage's own ranges, which no kernel ends,
  so a value that a gate around the read does not fold away leaves a free
  range in the kernel graph, which its specification refuses: "UOp
  verification failed … RANGE … AxisType.WEAK".
- **Reason:** (b). rune's compiled Fold (rune's `Lower_index.fold`) of a
  complex128 `[2; 3; 1]` by `lo [0; 1; 2]`, `interior [0; 1;
  0]` and a window of size 2 and step 2 on its last axis, whose every term
  the crop drops, stages its operand to read its two parts and raised
  there: rune's `a fold wraps integers and holds its float bound` failed.
- **Pinned by:** the `Rangeify` suite (`test/schedule/rangeify`):
  `get_kernel_graph › dead reads › a staged value read only in padding reads
  0 there`; rune's `Lower_index › folds and assemblies › a fold wraps
  integers and holds its float bound`.

## D159. A captured schedule tells the settings it compiles under

- **tinygrad:** `engine/jit.py:24-37` (`jit_lower`), which reads `JITBEAM`,
  or else `BEAM`, at `:35` and passes the width to `compile_linear` as its
  `beam`, binding no setting; its search cache keys on no setting
  (`codegen/opt/search.py:106`).
- **tolk:** `Jit.width`, `Jit.bindings`, `Jit.shaping`, and `Jit.jit_lower`,
  which compiles under `bindings`; `Setting.shaping_under`.
- **Differs:** `jit_lower ~beam` compiles with `BEAM` at its width and
  `JITBEAM`, read by then, unset, so the caches of what it compiles key on
  the width alone. `Jit.shaping ~beam` is `Setting.shaping` under those
  bindings, read without binding them (`Setting.shaping_under`): two
  lowerings whose `shaping` is equal compile alike, whatever `BEAM` and
  `JITBEAM` were. tinygrad has no such value. tolk's caches key on every
  setting that shapes compilation (`Setting.shaping`), so they keyed on
  `JITBEAM` after the lowering had read it; unset, they key on the width
  alone.
- **Reason:** (b). rune's `Rune.jit ~beam` (rune's `Compile.shaping`) keys
  a compiled function's variants on the settings that shape
  its programs, and with an explicit width neither `BEAM` nor `JITBEAM`
  does; it read the lowering's rule by repeating it.
- **Pinned by:** the `Jit JITBEAM` suite (`test/engine/jit`):
  `JITBEAM unset › a width given and BEAM's alike compile alike` and
  `JITBEAM=3 › a width given and JITBEAM's alike compile alike`; and the
  `Tolk.Setting` suite: `shaping › under bindings is shaping in their
  context, and binds nothing`.

## D160. A kernel that waits for the overwrite of a state it reads reads a copy

- **tinygrad:** `schedule/__init__.py:56-64` (`create_schedule`), which adds a
  write-after-read edge from a kernel to every overwrite of a state it reads,
  so a kernel that also waits for the overwrite, through a value computed
  after it, makes a cycle, and the schedule raises "cycle detected in assign
  graph" (`:78`). Repro: `a = Tensor([1., 2., 3.]).contiguous().realize();
  b = a * 2; a.assign(a + 1); (b + (a * 3).contiguous()).tolist()` raises;
  `test/runtime/test_assign.py::TestAssignOrdering::
  test_war_reader_already_depends_on_write` states the raise, with the
  values `[2, 4, 15]` "once that read is staged" as its TODO.
- **tolk:** `Schedule.copy_state`, `Schedule.copy_before_overwrites` and
  `Schedule.schedule`; `test/gen/tinygrad.patch`, which
  makes tinygrad copy the same way before the goldens are generated.
- **Differs:** no order runs such a kernel both before and after the
  overwrite, so it reads instead a copy of the state it read: a kernel of its
  own copies the whole state into call-local storage before the overwrite,
  and the schedule is made again from the graph with the copy. This holds for
  any overwrite of the storage, whichever elements each kernel touches, so a
  partial overwrite of elements the kernel does not read copies too, at a
  kernel and the state's size in storage.
- **Reason:** none. jera's compiled gradient of an adaptive solve reaches it,
  the only kernel of rune's, jera's, norn's and kaun's suites to:
  `Ode › solve derivatives › compiled grad equals eager grad` (jera's
  `test_ode.ml`). With tinygrad's refusal, rune's `Compile.declined` takes
  the `Invalid_argument` and evaluates the program, with the same values, so
  no call site fails without the copy: what it keeps is the gradient's
  compilation, a speed no bench row measures. Staging only the stale read in
  rangeify, as tinygrad's TODO says, would cost the read's size where the
  copy costs the state's.
- **Pinned by:** the `Tolk.Schedule` suite: `create_schedule › rules › a
  kernel that waits for the overwrite of a state it reads reads it`, on the
  kernel graph `read_overwritten_through_kernels.golden`, `› a kernel reading
  storage before and after a write reads both` and `› reads it before`, and
  the rules after them, for a copy two kernels share, a kernel with two
  copies, a copy of a state a rebuilt kernel writes and the copy's phase;
  and the program `read_overwritten_through`, whose recorded schedule and
  linear and whose values (`create_linear_with_vars › values`) are
  tinygrad's, patched.

## D161. A folded weak constant wraps, and a power of two is read at its width

- **tinygrad:** `uop/symbolic.py:170-173` (two stage ALU folding), which folds
  `c1 op c2` as weak constants, unbounded, under an operation on a committed
  integer, unless a signed result does not fit (`fits`, `:32`);
  `codegen/decomp/op.py:64-96` (`powers_of_two`), which turns a product,
  quotient or remainder by `2^i`, `i < 64`, into a shift or a mask by the
  constant's exact value, where the operation reads a weak constant at its
  operand's width. On int32, `(y * 65536) * (y * 65536)` becomes `(y * y) <<
  32`, a shift by the full width, which C leaves undefined: clang gives
  poison. Repro: `y = Tensor.arange(8,
  dtype=dtypes.int32).contiguous().realize(); p = y * c; (p * p).tolist()`
  renders `<< 32` and `<< 36` on the CPU for `c = 65536`, and for `c = -2**31`
  renders `<< 62` and the literal `73786976294838206464`, which clang refuses.
  A quotient's rule computes a value its own folder does not: on int8 the
  folder gives `1 // 128` as -1 and `1 % 128` as -127, reading 128 as -128,
  while the rules lower `x // 128` to `x >> 7` and `x % 128` to `x & 127`, 0
  and 1 at `x = 1`; on int32 `x // 2**31` becomes `x >> 31`, 0 where the
  folder gives -1. Repro: `x = UOp.variable("x", -128, 127, dtypes.int8);
  graph_rewrite(x.alu(Ops.DIV, UOp.const(128)),
  get_simplifying_rewrite_patterns((Ops.SHR, Ops.AND, Ops.CMPLT,
  Ops.CMPNE)))` against `UOp.const(1, dtypes.int8).alu(Ops.DIV,
  UOp.const(128)).simplify()`.
- **tolk:** `Shape.wrap_folded` and its use in `Shape.symbolic_simple`
  (the two stage fold); `Decomp_op.at_width`, `Decomp_op.power_of_two` and
  `Decomp_op.shift_of`; `test/gen/tinygrad.patch`, which makes
  tinygrad's rules the same before the goldens are generated.
- **Differs:** a weak result of folding two weak constants under an
  operation on an unsigned integer, past its width, is wrapped to it, as the
  operation reads it: `(x * 65536) * 65536` on uint32 is `x * 0`; on a signed
  integer the fold is refused, as tinygrad refuses it. A product by
  a power of two reads the constant modulo `2^bits`, and a quotient or
  remainder reads it in the operand's type, before it becomes a shift or a
  mask: no shift reaches the width, and on int32 a division by `2^31`, the
  least int32, is no shift.
- **Reason:** the fold and the product: (b). rune's compiled `Nx.square
  (Nx.mul c y)` at int32 and uint32, with `c` 65536 or the least int32, gave
  values that varied by run where eager gives 0: rune's `jit gives eager's
  bits for exact functions at every dtype` left such functions out. The
  quotient and remainder: (e). Without the reading, an int8 `x // 128` gives
  0 where -1 is due for a positive `x`, and `x % 128` gives `x` where `x -
  128` is, with no error; no raven call site is known to reach it. D162 reads
  every other divisor past the width so, and its `Decomp_op` pin fails without
  this rule.
- **Pinned by:** the fold and the product: the `Symbolic` suite: `integers
  wrap › a square of a product keeps its value, its constants folded at an
  unsigned width`; the `Decomp_op` suite: `late_patterns › a product by a
  power of two shifts by less than its width`; the `Codegen` suite:
  `constants past the width › a square of a product whose constant squares to
  0 is 0`. The quotient and remainder: the `Decomp_op` suite: `late_patterns
  › a quotient by a power of two reads it in its operand's type`, whose
  values at the width's power, int8 `1 // 128` among them, fail without it.

## D162. A divisor past its operand's width is read in its operand's type

- **tinygrad:** `codegen/decomp/op.py:54-62` (`floordiv_to_idiv`,
  `floormod_to_mod`) and `97-103` (the `fast_idiv` rules), which read a weak
  constant divisor's exact value where the division reads it in its operand's
  type: the sign test skips the correction and `fast_idiv` divides by the
  wrong number. On int8 the rules lower `x // 200` to a truncating
  division whose sign test reads 200 as positive: 0 at `x = 1`, where the folder
  gives `1 // 200` as -1 (a power of two, as 128, is D161's). Repro: `x =
  UOp.variable("x", -128, 127, dtypes.int8); graph_rewrite(x.alu(Ops.DIV,
  UOp.const(200)), get_simplifying_rewrite_patterns((Ops.SHR, Ops.AND,
  Ops.CMPLT, Ops.CMPNE)))`. On the CPU, `x = Tensor([0, 1, 7, 100, 127],
  dtype=dtypes.uint8).contiguous().realize()` gives `x // 259` as `[0, 0, 0,
  0, 0]` where `x % 259` gives `[0, 1, 1, 1, 1]`.
- **tolk:** `Decomp_op.divisor` and its
  uses in `floordiv_to_idiv`, `floormod_to_mod` and the `fast_idiv` rules;
  `test/gen/tinygrad.patch`, which makes tinygrad's rules the same before the
  goldens are generated.
- **Differs:** a weak integer divisor past a committed integer operand's
  width is wrapped to the operand's type before a division or remainder
  lowers: on int8, 128 is -128 and 200 is -56, and on uint8 259 is 3.
- **Reason:** (e). Without the wrap, an int8 `x // 200` gives 0 where -1 is
  due for a positive `x`, with no error. No raven call site is known to reach
  it.
- **Pinned by:** the `Codegen` suite: `constants past the width › a weak
  divisor past the width divides as its value there`, which compiles and runs
  the divisors past each width; the `Decomp_op` suite:
  `late_patterns › a quotient by a weak divisor past its width reads it in its
  type`.

## D163. An inlined value computes at most 360 nodes again

- **tinygrad:** `schedule/rangeify.py:48-97` (`remove_bufferize`, which
  inlines a stored value that reads at most three buffers and no reduction
  reading a buffer, at every position and every trip it is read at) and
  `:339-346` (`get_kernel_graph`); `schedule/indexing.py:218-269`
  (`run_rangeify`, which stores a value its consumers index differently, or
  one a reduction or an elementwise operation reads past a broadcast, the
  stores `remove_bufferize` weighs).
- **tolk:** `Rangeify.extent`, `Rangeify.read_ranges` and
  `Rangeify.read_copies` (the copies a read makes of a stage's value), with
  `Rangeify.digit_of` (a range read through floor divisions and remainders);
  `Rangeify.collapses`, `Rangeify.outer` and `Rangeify.collapsed` (a sum
  that selects one trip, asked only where the answer changes a stage's
  decision), and `Rangeify.storage` (a stage of no storage read in such a
  sum), through `Simplify.load_collapse`, which
  `Simplify.pm_load_collapse` applies; `Rangeify.reads`,
  `Rangeify.stage_reads`, `Rangeify.reduced`, `Rangeify.stage_copies` and
  `Rangeify.stage_bounds` (the copies of each stage);
  `Rangeify.inline_budget` and `Rangeify.copied`;
  `Rangeify.inlined`; `copies` in `Rangeify.remove_bufferize`;
  `Rangeify.pm_cleanups` and `Rangeify.get_kernel_graph`, which counts the
  reads before the cleanups; across kernels, `Rangeify.store_shared`,
  `Rangeify.candidates`, `Rangeify.kept`, `Rangeify.computed_nodes`,
  `Rangeify.union` and `Rangeify.is_kernel`; `test/gen/tinygrad.patch`, which
  gives tinygrad the same rules.
- **Differs:** a stage whose reads make `c` copies of its value stays stored
  when `n * (c - 1) > 360`, the nodes inlining computes again past the copy
  storing computes, where `n` counts the nodes of its value, its producers
  already inlined, that its ranges reach: those an index copies, since the
  substitution leaves the others shared. `c` sums over the stage's reads. A
  read makes one copy at each distinct index; in a reduction it makes one
  per trip `w` of the reduction ranges its index runs in, those a reduction
  of the graph closes, at most its trips per element of the stage, `t / e`,
  since a trip that computes no element again computes nothing that storing
  saves. A running sum's windows of `w` count `w`; a sum that reads each
  element once counts 1; a read whose index runs in no reduction range, as a
  broadcast through an expand, a reshape or padding, counts 1, its outputs
  each computing the value once, as an elementwise kernel does. Trips in
  padding count: inlined there, the value's loads are gated and its
  arithmetic runs. `e` and `t` leave out the axes the value does not vary
  along, whose index the inlining substitutes into nothing, and take a
  symbolic size at its largest.

  A read's copies leave out the reuse of a dropped digit. An output range of
  constant size that each component of the index reading it reads through
  one chain of floor divisions and remainders by positive constants, each
  dividing the size of what it is applied to, in components that read no
  other range, repeats the value across the outputs
  that differ only in the digits the chains drop: those outputs read it at
  one index. gpt-oss's experts read their decoded weights at
  `[(r // 16); (a * 2 + b); k]` for blocks of 16 rows, and the down
  projection at `[(r // 1024); (r % 64); k]`; a dense product whose output
  merges its rows with its `n` columns reads the decoded matrix at
  `[(r % n); k]` and the rows' cast at `[(r // n); k]`. The codegen splits a
  range at a remainder by a constant dividing it (`Simplify.pm_split_ranges`)
  and shares a quotient's repeats on a tensor core's `M` or a streamed
  operand's reuse, so a tile of those outputs shares the value as it shares a
  broadcast through a range the index does not name, which D77 and D106
  weigh. That reuse, the extent of such ranges over the extent of the stage
  axes their components index, divides `t / e`. A range that another
  component also reads, or that a sum selecting one trip runs in, stays
  counted; so do a window (`r // 16 + k`), a gather (a component reading a
  buffer), a framed read `(r // L) * h + r % L` and a digit sum
  `(r % m) + (r // m)`, which read a range twice in one component, a
  constant that does not divide its operand (`r // 16` of 50 rows, which no
  tile of a divisor of 50 shares whole), and a range of symbolic size, which
  is never split. Reduction ranges keep their count:
  the MXFP4 scales' `k // 32` counts 32. Rangeify counts without the target:
  what a tile shares is the codegen's, and a target that shares fewer lanes
  computes the value again more often than counted, as the host does
  (Reason).
  On a GPU the codegen owes that sharing, and states it: D230's bound of
  `ceil(R / 16)` decodes a value.

  A sum over one range that selects one trip, where an index equals the
  range, as a routed product's rows select their expert's matrix, counts
  that range as the ranges the sum runs in: the kernel computes it at the
  selected index alone (`Simplify.load_collapse`), once for each point of
  those, so a row decodes only its own expert's weights, `p / e` times a
  weight for `p` rows over `e` experts, where the trips count `p`. The test
  is the collapse's own, run on the sum as its kernel holds it: reads of
  stages whose values read no storage, as a range's positions do, are seen
  as those values, simplified as the cleanups simplify them, since the
  cleanups inline them before the kernel compares the range with its index.
  Counting the work after the substitution follows Mullapudi et al. 2016,
  §4.3: "a more accurate measure of cost because bounds analysis can
  overestimate the actual number of values required by a consumer".

  A read of a stage whose value reads no storage, in a sum the test finds
  selecting one trip, counts that sum's own trips, as a read in a sum that
  selects none: the test saw it as its value, which tinygrad inlines, and
  stored, it would leave the sum comparing a load with its range, which no
  kernel computes at one trip. A one-hot gather compares each row's index
  with the positions of the gathered axis, `arange`, a stage of a running
  sum whose ranges reach two of its nodes; counted at the sum's outer
  ranges, it made `k` copies in a product's sum over `k`, past the budget
  from `k = 182`, and was stored, so that every row multiplied every
  expert's matrix while the decoded matrices were counted, and stored, as
  read once a row: `Nx_quant.apply_experts` of gpt-oss-20b's gate and up
  projection at 512 tokens on fudge's RTX 5000 Ada took 1.49 s, and takes
  26.4-27.6 ms, against 3.14-3.23 ms through blocks of rows
  (`Nx.map_segments`). Its reads elsewhere count as before, so such a value
  past the budget in other reads stays stored, and the sum then runs every
  trip.

  The bound weighs work: an index copies the value into the code of each
  output, and a window's trips run it again in each output's loop, each copy
  past the first computing what storing computes once. Inlined, the copies
  compound: a value whose readers are inlined at several positions in turn
  is copied again with them, so `k` values each read at two positions are
  `2^k` copies of the first, in one kernel. tinygrad has the same
  exponential behaviour, and computes a value read in a window again at
  every trip. The bound also holds the code: a copy at a distinct index adds
  the `n` nodes a trip computes, and every read counts at least once, so the
  code inlining adds, `n` times the indices past the first, is at most the
  work, at most 360 nodes, under the 1024 a kernel of a thousand nodes costs
  to compile (about 50 ms on fudge, growing faster than its nodes; XLA
  bounds the same duplication by the copies alone, at 15,
  `FusionNodeIndexingEvaluation::kAllowedCodeDuplication`). Several windowed
  reads of one stage add up, as several indices do.

  Across kernels, the same count applies to a value of no range, which
  `run_rangeify` never stages: every consumer indexes it alike, so each
  kernel that reads it computes it, once. Once the cleanups have run, the
  kernels are known (no later pass removes a stage), and a computed value of
  no range that `k` kernels compute is staged in one element and read by
  each when `(k - 1) n > 360`, `n` its computed nodes. One walk, kernels
  first, gives each node the kernels its consumers are computed in, a
  staged value being a kernel of its own; a value is weighed where some
  consumer is computed in fewer kernels than it, so the values below it
  move with it. A weak value is never staged, since it takes its readers'
  widths and has no storage of its own; below it, as below a load several
  kernels share or a value with ranges, the consumers' fewest kernels pass
  down, and the value under it is weighed. Its `n` is then counted again down
  to the values kept below it, consumers first, and a value within the
  budget is computed in its kernels after all, the values below it counting
  those kernels too: the cost after the substitution, as Mullapudi et al.
  2016, §4.3, decide one function for all its consumers: "Either a function
  is inlined into all of its consumers (essentially removing the function
  from the program DAG), or to none of its consumers", weighing "the cost
  of inlining a function into all consumers". The walk is linear: a node's
  kernels are counted exactly to 362, past which any computed node passes
  the budget, and each weighing stops at the `360 / (k - 1) + 1` computed
  nodes that decide it. A value with ranges is computed in several kernels
  only through a range they share; it is left as it is, since storing it
  would hold its extent alive between its kernels (RFC 0005's bound).

  So a program of `N` nodes makes kernels of at most `N` nodes plus 360 for
  each stage inlined and for each value of no range left in several
  kernels, the reads the cleanups make aside; nodes with ranges that several
  kernels share, which no rule bounds, aside too.

  360 is placed by the cases whose stage it decides, each run with the stage
  stored and inlined (the bound at either side of the stage's count),
  alternating, 7 rounds of each under fudge's exclusive timing lock at a
  1-minute load of 24 to 51 (median 29), taking the median ratio of the
  rounds, stored over inlined; a case decides only where the two sides'
  times do not overlap:
  - gpt-oss-20b's decode step with Q8_0 projections, CUDA
    (`GptOss/decode-step-Q8_0-cache-1024`): each MXFP4 block's scale, 341
    nodes past the first copy, 24.05 ms stored against 23.80 ms inlined,
    1.011 (1.010 to 1.014), apart: the bound is at least 341. The MXFP4-only
    step gives 1.014 (1.001 to 1.046), overlapping.
  - jera's `root-newton-kepler-1k/compiled`, a float64 sine read by two
    kernels, 371 nodes: 0.47 ms stored against 0.75 ms inlined, 0.64 (0.58
    to 1.30), overlapping in one round of high load, so it does not decide;
    its median asks for a bound under 371.
  - rune's `Jit/experts-mxfp4-16x4of32-256`, the decoded weights at 2 rows
    an expert, 49 nodes: 0.91 (0.60 to 1.15); jera's
    `root-bracket-kepler-1k/compiled`, stages of up to 60 nodes: 1.14 (0.61
    to 2.42); `Cuda/sin-read-by-two-float64-1000`: 1.00; the compile of
    `scan/tridiagonal-100` against 1024 nodes of copies: 1.10 (1.00 to
    1.45): all overlapping.

  360 keeps the decided case and kepler's median. Wall-clock baselines of
  these rows on fudge are pending the scheduled re-record.

  D77 and D106 weigh the other recomputation inlining makes, where the value
  is broadcast across ranges its read's index does not run in: the ranges
  around the read, which an index does not name, so they stay separate
  rules.

  The reads are counted once, in a walk of the graph the cleanups start
  from, at each axis of a stage's ranges; a stage's count is the least over
  its axes, since stages over one range share that axis, so stages with no
  other range pool their reads and may stay stored when read once. A read
  the cleanups make is not counted, and a read through a stack is not one:
  every stage has a device by then, so no cleanup turns it into a read of
  the stage.
- **Reason:** (d). rune's compiled `Nx.associative_scan`, whose prefix
  circuit reads each level's pairs and results at two or three positions:
  jera's tridiagonal solve of 100 equations, three scans one of them over
  4-tuples, compiled cold in 62 s and compiles in 2.4 s (fudge, under load),
  as a scan of 1000 4-tuples does; a scan of 4 is 0.05 s either way.
  tinygrad compiles the same graph as slowly, 3.4 s at 32 elements. norn's
  nested sampling, whose error is a running sum over 100 × 1213 negated logs
  of uniform draws: inlined, the draw and its log ran once per window
  element, and the kernel took 0.87 s a call against 0.18 s with the draws
  stored. Compiled running sums over 100 × 400 values on fudge, in windows
  of 400 before D194, took 0.59 ms stored against 1.24 ms inlined for `2x`,
  and 0.57 ms against 1.25 ms for `x * x + x`; in windows of 16 `2x` is
  inlined. gpt-oss-20b's decode step with Q8_0 projections on fudge's
  RTX 5000 keeps each block's scale inlined, 1.1% faster, as above.
  Counting a dropped digit's reuse as copies stored every decoded weight a
  quantised product of many rows reads, which the product's tiles share
  once inlined: on fudge's RTX 5000 Ada (load 33 to 76, interleaved, one
  process under the timing lock per run, medians of 3 rounds, the ranges
  apart), gpt-oss-20b's `GptOss/prefill-512` takes 0.44-0.45 s against
  1.18-1.19 s, its `GptOss/experts-at-prefill-512` 18.5-21.8 ms against
  47.1-47.5 ms, and MXFP4 products of 16384 by 16384 past L2, of bfloat16
  rows (`Quant/cuda/mxfp4-product-of-16384-*`), 0.65 ms against 3.2 ms at
  12 rows, 1.49-1.50 against 6.0 at 64 and 1.81 against 6.1-6.2 at 128. Of
  float32 rows, which no tensor core multiplies, 64 rows take 12.5-12.6 ms
  against 9.6-9.7, 1.3 times slower. The count is the same on fudge's RTX
  5000 Ada and nonnormal's R9700, read from the inlined kernels' sources (1,
  2, 1, 2 and 4 decodes a value at 16 to 256 bfloat16 rows, 16 at 64 float32
  rows), with opposite outcomes: inlined, every bfloat16 row is faster on
  CUDA and slower on AMD (64 rows 8.2-8.8 ms against 5.0-5.1 stored), where
  the loss lies in the kernels the heuristic makes (256 single-wave
  workgroups, 48 spilled registers, a byte's values loaded 2 bytes apart),
  and a search over tolk's own optimisations finds inlined kernels faster
  than stored ones on both (64 rows 1.15 against 3.58 ms on CUDA, 128 rows
  3.31-3.47 against 6.57-6.68 ms on AMD): the count stands, and the codegen
  owes its sharing (D230). One MoE layer at 512
  tokens runs 18.6-18.8 ms against 47.4-47.9; its inlined products are
  streamed (`Setting.mv`), and on the tensor cores, with streaming off, the
  layer takes 4.5 ms against 79.5; D89's order by multiply-adds per byte now
  takes them to the cores (4.66-4.68 ms). Of float32 rows, which no core
  takes on CUDA, the general rules shared only their upcast of 4 rows and
  decoded each value 16 times at 64 rows; streamed past a reuse of 32
  (D230), 64 rows take 4.98-6.68 ms, under the stored decode's 9.6. The
  cost is the host's, whose tiles
  share 3 or 4 lanes of a digit or none, so an inlined decode runs about 4
  times a value per 16-row block and 10 to 15 times in a dense product: at
  a load of 67 to 76, MXFP4 products of 1024 by 1024 take 1.5-2.5 ms
  against 0.28-0.48 ms at 10 rows and 5.9-11.1 ms against 1.5-2.3 ms at 64
  (ranges overlapping), and a MoE layer 256 wide at 512 tokens 60-67 ms
  against 30-36 ms (apart in two rounds). Deciding for the host needs the
  target's operations per byte, which the scheduler does not take.
  Across kernels, (d) too: ymir's aperture gradient (`aperture-104/
  compiled-grad`) computes the transform of the aperture's centre, a value
  of no range of about 3,000 nodes, in 18 kernels. Stored where its copies
  pass the budget, its cold first call takes 22.7-25.9 CPU-s (the process
  and Clang) against 31.8-36.9 s, compiling 211 programs against 196, with
  the same bits (fudge, three interleaved rounds under the timing lock at a
  1-minute load of 36 to 76). A chain of scalars each read by a kernel of
  its own, as rune's chain of outputs reads it, was quadratic in its length
  and is linear (the Rangeify suite's chain test). The row's wall-clock
  baseline is pending the scheduled re-record.
- **Pinned by:** the Rangeify suite's rows of `kernel_counts.golden` and
  their kernel goldens, from the equally patched tinygrad: across kernels,
  `scalar_read_by_two` (a value of no range of some forty nodes read by two
  kernels, computed in each), `long_scalar_read_by_two` (of some four
  hundred, stored) and `scalar_chain` (40 values of no range, each read by a
  kernel of its own and the next, one stored); `read_at_offsets`
  (a value of some forty nodes read at 32 offsets, stored), `windowed_chain`
  (in a running sum of 300, stored), `windowed_pair` (in two running sums of
  16, stored), `windowed_scaled` (a product in a running sum of 16, its
  copies past the first 90 nodes, inlined), `windowed_scaled_long` (a
  product in a running sum of 400, inlined into its windows of 16), `read_by_two` (some forty nodes read by two
  kernels, inlined), `long_read_by_two` (some four hundred, stored),
  `routed_product_2` (a decoded matrix gathered by 2 rows an expert over 64
  experts, inlined), `routed_product_64` (64 rows an expert, stored) and
  `routed_product_wide` (16 rows an expert in a product's sum of 192, the
  gather's positions inlined),
  `blocked_product_16` (rows in blocks of 16 reading their block's matrix
  of some forty nodes at `r // 16`, inlined) and `dense_decoded_64` (64 rows
  merged with a matrix's 4 columns, read at `r % 4`, inlined),
  `tiled_chain` (tiled 256 times, inlined) and `padded_pair` (read through
  two paddings to 1024, inlined); `cost › levels each read twice make
  kernels that stop growing with depth`; the `copies` group: `a value of
  1200 nodes read at one index is inlined`, `› a value read twice whose
  second copy is the budget is inlined`, `› a value read twice whose second
  copy passes the budget is stored`, `› a value of 600 nodes outside its
  ranges read twice is inlined` and `› reads of a stage sharing a range are
  not the other's`; the `recomputation` group: `a value of 1200 nodes
  reduced once an element is inlined`, `› a cheap value reduced in windows
  is inlined`, `› a value reduced in windows whose work is under the budget
  is inlined`, `› a value reduced in windows whose work passes the budget is
  stored`, `› a value of no storage reduced in windows past the budget is
  stored`, `› two windowed reads of a stage add up`, `› a value of symbolic
  size counts at its largest`, `› a value reduced in windows counts no axis
  it does not vary along`, `› a read counts at most the trips of its
  reductions`, `› a value reduced through padding counts the padding's
  trips`, `› a value inlined into a windowed stage counts in its copies`, `›
  a value broadcast through a reshape is inlined` and `› a value read
  through padding is inlined`; `work bound › a stage is stored when its
  copies beyond the first pass the budget`, over values read once, twice, in
  windows and by routed rows; the `dropped digits` group: `a dropped digit's
  reuse is no copy`, over quotients, remainders, a middle digit and a chain
  inlined, and windows, a reduction range's digit, a gather, frames, digit
  sums, a range read by two components, constants not dividing and a
  symbolic size stored, `› a read of merged outputs decides as one of them
  split` and `› a digit a selection runs in stays counted`; rune's
  `Lower_reduce › work › a Rows contraction's work is one slice's over 256
  contracted` and `› a scaled float4 Rows contraction's work is one slice's
  over 256`; rune's `rune
  jit › classes › an associative scan over a tuple gives eager's bits` and
  `› a value stored or computed again in its readers gives eager's bits`,
  which reads the value in blocks of rows and in a product of merged rows
  too; on fudge, the
  `scan/tridiagonal-100` row of rune's compile bench, the
  `Jit/cumsum-neg-log-uniform-8x1024`, `Jit/experts-mxfp4-16x4of32-256` and
  `Cuda/sin-read-by-two-float64-1000` rows of its bench, jera's
  `root-newton-kepler-1k/compiled` and `root-bracket-kepler-1k/compiled`,
  gpt-oss's `GptOss/prefill-512` and `GptOss/experts-at-prefill-512`, and
  kaun's `Quant/cuda/mxfp4-product-of-16384-*-tokens` and
  `Quant/host/mxfp4-product-of-1024-*-tokens`. Across kernels, the
  Rangeify suite's `get_kernel_graph › values that several kernels compute`
  (every test: the budget's two sides for two and three kernels, a weak
  value, a value read through a load the kernels share, a value cheap above
  a stored one, and the chain of outputs whose
  work is linear in its length and whose kernels stop growing with it), and
  ymir's `aperture-104/compiled-grad` row.

## D164. Withdrawn

The read of an accumulator after its loop waited on the initial store as well
as the loop's end, since `sym` folded a store of a loaded accumulator to a bare
`NOOP` and the loop could end holding no store, so nothing ordered the read
after the initial store. At `uop/symbolic.py:467,474` a folded store is
`NOOP.after(*x.src)`, which keeps its sources and so the initial store before
the read, and `codegen/__init__.py:183-195` (`lower_reduce`) orders the read
after the end alone, as `Codegen.lower_reduce` does. The
`Codegen` suite's `an accumulator no lane adds to › is stored before it is
read` and `› sums to zero`, and rune's `Lower_index › folds and assemblies › a
fold wraps integers and holds its float bound`, hold.

## D165. A weak maximum's identity is its accumulator's least value

- **tinygrad:** `uop/ops.py:57` (`identity_element`), whose `Ops.MAX` identity
  is `dt.min`: for `weakint`, of bitsize 800 (`dtype.py:122`), `-2**799`.
- **tolk:** `Ops.identity_element`;
  `test/gen/tinygrad.patch`, which makes tinygrad's the same before the
  goldens are generated.
- **Differs:** a maximum over a weak type has as identity the least value of
  `Dtype.strong` of the type, the width its accumulator is allocated at. In
  tinygrad a weak integer's maximum, gated by an Invalid, takes `-2**799`:
  `lift_reduce_gate` (`uop/symbolic.py:78-86`) selects it where the gate
  fails, and committing the
  selection's width (`cast_weak_srcs`, `commit_int`) raises "does not fit any
  int". A weak maximum arises where a pad's bare fill is all that remains of a
  committed maximum's operand.
- **Reason:** (b). rune's compiled Fold (rune's `Lower_index.fold`) of a
  float64 `[2; 2; 2]` by `lo [2; 2; 0]`, `hi [0; -1; -1]`,
  `interior [1; 0; 0]` and three windows maximises the uint8 marks of each
  element's terms; where no term reaches an element, the mark is the pad's
  fill, a weak `0`, and compiling raised: rune's `a fold wraps integers and
  holds its float bound` failed.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `a maximum of
  weak integers › is the greatest value it reduces`; `Tolk.Ops › data types ›
  identities.golden`; rune's `Lower_index › folds and assemblies › a fold
  wraps integers and holds its float bound`.

## D166. A minimum of a minimum against one value is one selection

- **tinygrad:** `codegen/decomp/op.py:81` (`get_late_rewrite_patterns`),
  which has no rule for a nest of selections.
- **tolk:** `Decomp_op.extreme_of`, `Decomp_op.nested_extreme`,
  `Decomp_op.unwrapped`, `Decomp_op.holds`, `Decomp_op.keeps` and
  `Decomp_op.late_patterns`; `test/gen/tinygrad.patch`, which gives
  tinygrad's late patterns the same rule but for casts, which no golden's
  nest holds.
- **Differs:** a selection of a value and `c` by their less-than comparison,
  a minimum or a maximum against `c`, whose value is such a selection of the
  same kind against the same `c`, is rewritten to one selection: the inner
  one where it picks `c` where its comparison is false, at a tie or a NaN,
  or where the two pick alike, and otherwise `where (v < c) v c` or `where (c
  < v) v c` of the inner one's other operand `v`, cast to the outer one's
  type. Both are what the nest computes at every `v`, NaN and both zeros
  included. Values are matched through the casts that keep them
  (`Decomp_op.keeps`): a widening to a float type that holds the operand's,
  and a narrowing of a value its type holds (`Decomp_op.holds`), a constant,
  a selection of such values at any depth, or such a value widened. LLVM's
  InstCombine narrows a `float32` selection of widened `float16` values to a
  `float16` selection, which keeps every value, then folds the plain nest.
  rune computes a `float16` minimum as a `float32` selection of widened
  operands, narrowed back, so a nest of them reaches that fold.
- **Reason:** (b): LLVM's InstCombine (`foldSPFofSPF`) folds a minimum of a
  minimum, or a maximum of a maximum, against one constant into the inner
  selection, though the two pick apart at a tie or a NaN: with `m = (0.0 <
  v) ? 0.0 : v`, `(m < 0.0) ? m : 0.0` becomes `m`, `-0.0` at `v = -0.0`
  where the selection gives `0.0`, and with `1.0` for `0.0` NaN at a NaN
  where it gives `1.0`. It runs on every target from `-O1` in clang 19, at
  `-O3` for amdgcn, which HIP's comgr runs, and in NVRTC 13.4, and LLVM's
  main of 2026-10 keeps it; tinygrad on x86_64 computes these `-0.0` and NaN.
  Under `Rune.jit`, `Nx.where (Nx.less m c) m c` of `m = Nx.minimum c x` gave
  `-0.` where eager gives `0.`, NaN where eager gives `c`, and a reciprocal
  of it `-inf` for `inf`: rune's `a step over donated leaves keeps the
  values` failed. Made one selection before rendering, a single selection
  keeps its lowering to one instruction: hiding the constant from LLVM in
  every comparison instead cost a maximum before a 64 × 64 matmul 1.10 times
  its time and a 4096-point FFT 1.15 times on fudge (paired). LLVM can still
  form such a nest itself, as when it vectorises scalar minima of a vector
  minimum's lanes; no tolk kernel is known to.
- **Pinned by:** the `Decomp_op` suite (`test/codegen/decomp/decomp_op`):
  `late_patterns › late_nests_cmplt.golden`, `› a nest of extremes against
  one value keeps its values › *` and `› a nest of 16-bit extremes through
  float32 keeps its values › *`, which also holds the rule back where a
  narrowing rounds; rune's `rune jit › classes › a selection of a minimum
  against its constant is exact`, at `float16` and `bfloat16` too.

## D167. Clang of LLVM 18 compiles without its straight-line vectorizer, and a long source without its loop vectorizer

- **tinygrad:** `runtime/support/compiler_cpu.py:21` (`ClangCompiler.compile`,
  Clang's arguments at `-O2`, which runs LLVM's SLP and loop vectorizers).
- **tolk:** `Compiler_cpu.llvm_major` (the LLVM version a Clang's `-###`
  statement names), `Compiler_cpu.long_source`, and `Compiler_cpu.clang` (its
  `args`, which adds `-fno-slp-vectorize` when it is 18 and `-fno-vectorize`
  for a source past `long_source`, and its `table`, which names both
  statements and the length).
- **Differs:** a Clang that states LLVM 18 compiles every kernel with
  `-fno-slp-vectorize`, after `-O2`, which would turn the vectorizer back on.
  Every Clang compiles a source of more than 400 000 bytes with
  `-fno-vectorize`, after `-O2`. Otherwise every Clang, Apple's included,
  compiles with the arguments tinygrad passes. The table of objects names
  what Clang states it runs for either length, and the length (D113).
- **Reason:** (b): rune's packed suite, `a compiled call computes eager's
  values › a uint4 sort along the last axis of every other row and column,
  flipped` (now `rune jit › classes › a uint4 sort along the last axis of
  every other row and column, flipped, gives eager's codes`), failed on CI
  runners with AVX-512 under Ubuntu 24.04's Clang
  18.1.3 ("SmallVector unable to grow"), and passed on runners without it.
  The kernel `r_3_2` loads 6 nibbles at clamped indices, `2 - min(i & 3, 2)`
  or `7 - min(i & 3, 2)`, the minimum written `~max(~x, -3)`. Before the
  vectorizer, LLVM holds four of these indices as `sext` of `2 - x` and `zext
  nneg` of `x ^ 7`; for `x86-64-v4`, `skylake-avx512` or `znver4` the
  vectorizer packs them into one vector, narrows it to 16 bits, and writes
  `sext <4 x i32> ... to <4 x i16>`, invalid IR, on which X86 instruction
  selection crashes. Clang 18.1.8 does the same; Clang 19.1.7 compiles the
  source, and without the vectorizer so does 18. `x86-64-v3` packs nothing
  there. Of the 6048 kernels rune's suites compile, this is the only one whose
  IR Clang 19 rejects when Clang 18 writes it for `x86-64-v4`. D141's split of
  6 lanes into 3 pairs was not the cure: the suite failed the same way on CI
  after it. Rewriting the source does not hold either: writing that minimum as
  `(-3<x)?x:-3` or `__builtin_elementwise_max`, or selecting with scalar
  comparisons, lets this kernel compile, but each only moves what the
  vectorizer meets, for a rendering of every kernel tinygrad also writes.
  Disabling the vectorizer for every Clang would change the objects of 44%
  of a sample of those kernels (265 of 604, Clang 18 for the host): it stays
  on wherever it is correct.

  (d), the loop vectorizer: rune's test_special timed out at 60 s on Ubuntu
  CI compiling float64 `betainccinv`'s gradients. Each is one scalar loop
  kernel, `E_1021`, of 1.3 MB of C; the loop vectorizer weighs every width
  against the whole body, and its time grows faster than the body (about
  n^2.4 on 4-lane kernels). On fudge at load 40-55, Clang 19.1.7 took 19.7 s
  on it with the vectorizer and 5.7 s without (medians of 3, interleaved),
  and the first call 38.9 s against 13.6 s; replay, timed only loaded, was
  1.5-5.5 ms against 4.5-22 ms for 1021 values, as D141 found 204 ms against
  475 ms for `log_betainc`'s gradient. Of the 13 096 kernels tolk's bench,
  rune's compile bench and rune's suites but test_special compile, the
  longest is 136 KB and the longest with a loop 125 KB; test_special's
  gradients of special functions run from 51 KB to 1.3 MB, and those past
  400 KB (432 KB to 1.3 MB, 12 of 58) took Clang 4.1-29.4 s, against 0.7-1.4
  s for those of 158-279 KB. The one other kernel past it, the compile
  bench's `special/betainc-chain-f32` (530 KB), has no loop, and its object
  is the same byte for byte. A `#pragma clang loop vectorize(disable)` on the
  loop saves a third (medians of 5: 11.5 s against 17.8 s), since the pass
  still weighs interleaving; with `interleave(disable)` it saves what the
  flag saves (5.5 s), but the renderer would write it before each loop of a
  source tinygrad writes without it, and the goldens with it, while the flag
  leaves every source as it is and the table keys the objects.
- **Pinned by:** `Tolk.Compiler_cpu › features › four clamped gathers compile
  for AVX-512 on` (`x86_64,x86-64-v4`, `x86_64,skylake-avx512`,
  `x86_64,znver4`), which compiles the indices of `r_3_2` and fails under
  Clang 18 without the flag; rune's `rune jit › classes › a uint4 sort along
  the last axis of every other row and column, flipped, gives eager's
  codes`, under Clang 18 on an x86 host with AVX-512. The loop vectorizer:
  `Tolk.Compiler_cpu › features › a loop over a count it takes is packed
  into vectors iff its source is at most 400000 bytes long:` (rows 400000 and
  400001 among them); rune's compile bench `special/betainccinv-grad-f32`, the float32
  gradient of test_special (914 KB), whose first call takes 3.88 and 4.48 s
  on fudge against 6.57 and 6.66 s with the loop vectorizer (medians of 3,
  two interleaved runs each).

## D168. The renderer finds a program's callees by a scan of its instructions

- **tinygrad:** `renderer/cstyle.py:278-285` (`CStyleLanguage.render`, which
  lists the callees by a toposort of `UOp.sink(*uops)`, entering calls where
  the renderer inlines their bodies, then renders each body, then the
  program).
- **tolk:** `Cstyle.callees`, which `Cstyle.render` asks.
- **Differs:** a scan of the instructions replaces the toposort: each
  call's callee is met at the call, and a body not met before is scanned
  through its own instructions, then listed. The instructions are in order,
  each after its sources, and hold every node but the bodies of calls, which
  a toposort of their sink enters at their call: the externs and the bodies
  come in the toposort's order, and the source is the same.
- **Reason:** (d): compile time of every kernel, most of which call nothing.
  The toposort of the whole program, call bodies entered, took 16 to 22% of
  the tolk bench's `*/render` rows on fudge and 7 to 13% of their
  allocation, `gpt_oss_decode/render` and `gpt_oss_prefill/render` among
  them. On fudge, rendering `conv`'s kernels took 1.33M instructions with
  it and 1.12M with the scan, against 1.06M before calls were rendered.
- **Pinned by:** `Tolk.Codegen › calls in a kernel ›` `externs and functions
  are declared as a toposort of the instructions meets them`, `a body calls
  an external function the kernel declares once`, `bodies of one name traced
  apart are two functions` and `one body called twice is one function`; the
  tolk bench's `*/render` rows on fudge, whose time and exact allocation
  would catch the toposort's return.

## D169. A scheduler reads its kernel's axes once, and keeps a flat kernel

- **tinygrad:** `codegen/opt/postrange.py:18-19` (`Scheduler.__init__`,
  which rewrites its kernel by `pm_flatten_range`), `:23-34`
  (`Scheduler.rngs` and `full_shape`, properties computed at each access),
  `:38-41` (`copy`, which makes a scheduler of the kernel, rewriting it
  again), `:75-80` (`axes_of`, which computes `reduce_ranges` and `rngs`,
  itself computing `reduce_ranges`) and `:227` (`reduceops`).
- **tolk:** `Postrange.Scheduler.view` (`view_of`), which `slice`, `rngs`,
  `full_shape`, `reduce_ranges` and `reduceops` read;
  `Postrange.Scheduler.flat` and `Postrange.Scheduler.v`.
- **Differs:**
  - the kernel's slice, its reductions, the ranges they close, its axes and
    their sizes are computed when first asked after the kernel changes, and
    kept until it changes. tinygrad computes them at each query;
  - a kernel whose reductions and ends each close distinct ranges, which the
    rewrite would leave unchanged, is kept without it; any other is
    rewritten. tinygrad rewrites every kernel, a copy's included.

  The kernels and the answers are the same.
- **Reason:** (d): a beam search's rounds and the hand-coded optimisations
  of every kernel. With a range's reduce role read from the reductions, each
  axis query listed the kernel's slice up to five times, and each copy of a
  scheduler rewrote its kernel. Measured on fudge by instruction counts:
  making the candidates of `matmul`'s kernel for CUDA took 14.0M
  instructions with the role and 6.6M with both changes (9.5M before the
  role, whose search also had unroll actions); lowering `matmul`'s kernel
  took 14.96M with the role, 14.57M with both, 14.31M before. With both, the
  tolk bench's `search/actions/*` rows (the three lorenz kernels) run 25 to
  27% faster than before the role, and `matmul/codegen` within 1%, where it
  had taken 12% longer.
- **Pinned by:** `Tolk.Postrange › a scheduler's axes answer as a fresh
  copy's, whatever came before`, which interleaves optimisations, copies and
  queries; `› the scheduler's axes after optimising are tinygrad's`;
  `› Scheduler ›` `v's kernel has the ranges its reductions and ends
  close flattened`, `v's kernel of a recorded kernel is it flattened` and `a
  copy's kernel after optimising is the kernel flattened`; the tolk bench's
  `search/actions/*` rows on fudge, which time making a search's candidates,
  and its `*/codegen` rows.

## D170. A program serves every device of its target

- **tinygrad:** `codegen/__init__.py:513` (`to_program_key`): the key is
  `ast.key`, whose `PARAM`s carry their buffer's device
  (`schedule/rangeify.py:284`, `debuf`), so each device makes its own
  program of one kernel.
- **tolk:** `Codegen.program_key` (`Codegen.params`) and
  `Codegen.to_program` (`Codegen.rename`); `Realize.lower_and_compile`
  compiles one kernel of each key, as tinygrad's does
  (`engine/realize.py:244`).
- **Differs:** the key of a program is its kernel with each parameter that
  names a device naming its target's (`CPU`, `CUDA`), so a kernel on the
  target's first device is its own key. A kernel for one device takes the
  program made for another of its target, in memory and on disk, its
  parameters renamed to the kernel's devices: its source and binary name no
  device, and a link binds the program to its call's buffers and launches it
  on its call's devices.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.make`, `Pieces.split`) lower each device's operations on that
  device, so a program over `n` memory devices of one host makes `n` copies
  of each kernel. With the devices in the key, a data-parallel step's first
  call over 16 memory devices took 0.92 s on fudge, 52 ms per device past
  the first; with them erased, 0.24 s.
- **Pinned by:** `Tolk.Codegen › programs are kept › a program for one device
  serves another of its target`; rune's `Rune.jit over several devices`.


## D171. A store's check for reading its destination walks what follows it

- **tinygrad:** `schedule/indexing.py:32` (`realize_store_after_src`): each
  store walks its value's backward slice, entering no other buffer's state
  (`AFTER`), for its destination's storage.
- **tolk:** `Indexing.realize_store_after_src`, over the order
  `Indexing.run_rangeify` gives the graph.
- **Differs:** once the stores' walks have entered as many nodes as the
  graph holds, each walk enters only the nodes that come after the
  destination's storage in one walk of the graph that finishes each node
  after its sources: a node before it cannot reach it. The values stored
  are tinygrad's.
- **Reason:** (d): rune's data-parallel step over 16 memory devices
  (`Rune.jit` over `Nx.Placement.split`) has a store per move between
  devices, about 1500, each of whose values reaches its device's whole step.
  Their walks covered 1.84 million nodes, 235 ms of a 1.05 s schedule on
  fudge; pruned, 38 thousand in 8 ms. Pinned by rune's `devices/dp-step-16`
  row on fudge.
- **Pinned by:** `Tolk.Indexing › run_rangeify › recorded graphs`
  (`assign`, `assign_permuted`, `assign_double_diamond`), whose stores read
  their destinations.

## D172. An index of an Invalid value folds after construction

- **tinygrad:** `uop/ops.py:549-553` (`UOp.index`, which returns
  `UOp.invalid().expand(ret.shape)` when the indexed value's base is
  `Invalid`).
- **tolk:** `Ops.index`, which builds the `Index`; `Prepare.mop_index`, which
  moves it through the expansion to an `Index` of `Invalid`, and
  `Codegen.devectorizer2`'s rule for an `Index` without indices, which drops
  it; `Codegen.do_devectorize`, which reads an `Invalid` source's base in each
  lane.
- **Differs:** an index of a value whose base is `Invalid` is an `Index` node
  until the movement and devectorize rules fold it, where tinygrad folds it to
  an `Invalid` of the index's shape when it builds it. The kernels are the
  same: the rangeified graphs, schedules and values of a selection no lane
  takes match tinygrad's.
- **Reason:** (a): `Ops` builds nodes without shapes, which `Shape` computes
  over `Ops`, so `Ops.index` cannot know the shape of the `Invalid` it would
  return.
- **Pinned by:** `Tolk.Indexing › run_rangeify › recorded graphs ›
  select_no_lane_rangeified.golden` and `select_no_lane_rows_rangeified.golden`;
  `Tolk.Schedule › create_schedule › recorded › select_no_lane_kernels.golden`,
  `select_one_lane_kernels.golden`, `select_no_lane_rows_kernels.golden`; and
  `Tolk_engine › link and run › recorded › select_no_lane writes what its
  tensors compute` and its `select_one_lane` and `select_no_lane_rows`
  versions, which run the kernels on the host.

## D173. A schedule linked per run makes its arenas at each run

- **tinygrad:** `engine/jit.py:50-80` (`CapturedJit`): the linked linear of a
  captured schedule holds its intermediates' buffers from its first call
  until `free_intermediates` drops it, so they outlive every run, and
  `engine/jit.py:32` (`jit_lower`) places them in arenas
  (`memory_plan_rewrite`), copies' and kernels' apart
  (`schedule/memory.py:35-37`).
- **tolk:** `Tolk_engine.link`'s `storage`, `Tolk_engine.per_run_storage`,
  `Tolk_engine.run_with`; `Jit.jit_lower`'s `storage` and
  `Memory.memory_plan_rewrite`'s `per_run`.
- **Differs:** a schedule linked `Per_run` is planned into arenas, one lane
  per device (D41), in the order by storage (D183, D185), and makes each
  storage node no buffer is bound to before its run's first call and frees
  it after its last, its memory coming back once the work on it is done. A
  run is one program of rig's, which loads each image once per device, and
  whose submissions are those of the same schedule linked `Linked`: the order
  by storage interleaves devices, so a data-parallel step over 16 memory
  devices makes 1411 `Submit` steps where tinygrad's order makes 84. Linked
  `Linked`, the default, its storage is the program's, as tinygrad's. A run
  of either keeps none of the buffers it was given once it returns.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.variant`, for a split program) promise that a device holds
  between calls only the values its caller keeps, and RFC 0005 bounds what a
  fully sharded step holds within a call. Linked `Linked`, the programs of
  fully sharded training over two memory devices, its gathers compiled, held
  4.75 layers on each device beside its share after a step (`rune jit ›
  fully sharded`, d = 128): 3.75 of scratch, and one of the buffers the
  frames kept from their last runs. Each buffer made at its first call and
  freed after its last, as tolk did before, met the bound on memory devices,
  whose work is done when it is submitted, but not on a GPU, which reuses
  freed memory only once its work is done while a run returns before it
  is: on fudge's RTX 5000, eight chained products of 64 MiB, two live at
  once, held 640 MiB at the call's return, where planned into an arena they
  hold 320 MiB with the arguments and the result. That design was also one
  program of rig's per stretch between makes and frees: the data-parallel
  step over 16 memory devices was 1168 programs, loaded 656 images where one
  program loads 11, and made 1584 submissions a call, allocating 153 580
  words where one program allocates 23 015 (rune's `Jit/devices/dp-step-16`
  row). Planned
  per run, rune's five `fully sharded` cases held 2.128, 1.692, 1.473, 4.001
  and 2.626 layers against bounds of 2.188, 1.719, 1.484, 4.250 and 2.812,
  as each buffer of its own did; with copies in a lane of their own, 3.068,
  2.538, 2.273, 4.756 and 2.944, each over its bound, since two arenas made
  for the whole run hold both lanes' peaks at once.
- **Pinned by:** `Tolk_engine › runs › a schedule linked per run holds no
  storage between runs`, `› a run holds its storage whole, no less than its
  buffers live at once` (the least budget a run fits is its plan's bytes,
  which hold what a model of each buffer's first and last use has live at
  once), `› a
  schedule linked per run does the work it does linked` and `› a run keeps
  none of the buffers it was given`; `Tolk.Memory › memory_plan_rewrite ›
  rules › per run, a copy's buffer and a call's of one device share bytes
  once their lives are apart` and the plan law; rune's `fully sharded`
  tests, whose shares are exact after a step, and `Rune.jit on a GPU ›
  memory › a split program holds at its return at most its result and its
  plan`.

## D174. A kernel's call bodies are functions of its own source on every target

- **tinygrad:** `codegen/__init__.py:403-416` (`LowerCallsCtx`, `lower_call`), which lowers a
  call's body to a program of its own (`to_program`, marked
  `KernelInfo.kernel=False`, so its arguments stay parameters) unless the
  renderer has an `inline_prefix`, and `renderer/cstyle.py:73-75,278-285`
  (`render`), which declares such a program `extern` and calls it with its
  arguments; `engine/realize.py:106-112` (`get_runtime`), which links it at
  load time by name (`runtime/support/elf.py`, `link_syms`). Only HIP has an
  `inline_prefix` (`renderer/cstyle.py:507`).
- **tolk:** `Codegen.lower_call`, which lowers every body to the lines of a
  function of the calling kernel and refuses a call of a compiled program, and
  `Cstyle.render`, which writes the function `static inline` with the
  language's `inline_prefix` (`""` for Clang and Metal, `__device__` for
  CUDA, HIP's for HIP).
- **Differs:** a body called inside a kernel is inlined into the kernel's
  source on every target, as tinygrad does on HIP alone. tinygrad compiles it
  apart on Clang, Metal and CUDA and links the two binaries when it loads the
  kernel, and so calls a compiled program too. A body is never a program, so
  no kernel information tolk makes is marked as called by another kernel
  (`kernel_info.kernel` stays `true`).
- **Reason:** (c): rig links a host object only against symbols of the process
  (`Rig.section-host_linking`: each symbol the object refers to and does not
  define is defined by the process), and loads each GPU image alone, so a
  kernel that refers to another program's symbol does not load.
- **Pinned by:** `Tolk.Codegen › calls in a kernel`, whose kernels run on the
  host with their bodies inlined, and `› a CUDA kernel defines the functions
  it calls, as device functions`, `› a Metal kernel defines the functions it
  calls` and `› a call of a compiled program is refused`.

## D175. The splits after a tensor core keep the target's threads in flight

- **tinygrad:** `codegen/opt/heuristic.py:40-43` (`hand_coded_optimizations`,
  without `TC_MIN_GLOBALS`: after the core, M and N are upcast and N split
  into local threads, by the first size that divides them, whatever the
  workgroups left; the same at `6b7b018da`).
- **tolk:** `Heuristic.core_threads` and `Heuristic.tensor_cores`;
  `test/gen/tinygrad.patch`, which gives tinygrad's heuristic the same rule
  before the goldens are generated.
- **Differs:** each of the three splits applies only while the kernel keeps
  at least `core_threads / Tc.threads` workgroups after it, the product of
  its global axes, or when that product is not known: 16384 threads, 512
  workgroups of a CUDA, RDNA or Metal core of 32 threads and 256 of a CDNA
  core of 64. An upcast leaves fewer threads to keep memory and the L2 busy
  (concurrency is latency times throughput, Volkov, UCB/EECS-2016-143 §3.1);
  the local split keeps the threads and leaves fewer workgroups to share
  among the compute units, so it takes the same floor. 16384 is the bound
  of sm_89's parts, and every target takes it until one measures another;
  then `core_threads` becomes a field of `Renderer.t` set per target. Under
  `TC_MIN_GLOBALS` the splits are tinygrad's.
- **Reason:** (d): gpt-oss-20b on fudge's RTX 5000 Ada, its products at
  bfloat16 widened by a cast (D29). GPU time, three interleaved rounds under
  the timing lock, warm and from a clean, cold cache:
  - against 100 workgroups (4096 threads on these kernels), us: the decode
    step's keys (`stream/decode-keys`) 5.37-5.86 to 4.15 warm and 9.03 to
    6.10 cold; a prefill's key projection, 512 by 512 over 2880
    (`stream/key-projection`), 132.8-132.9 to 114.8 and 250.9-251.0 to
    115.7; `matmul_half_512` 19.0-19.3 to 17.3 and 33.7-34.7 to 17.3;
    `matmul_half_256` 6.10 both warm, 8.06 to 6.10 cold; `matmul_half_1024`
    44.2-44.7 to 40.8-41.0 warm, 62.5-66.5 and 62.5-65.7 cold;
    `stream/dense-bf16-64`'s core kernel 472-487 and 483-485 warm, 469-483
    and 479-481 cold; the attention values, `stream/decode-kv` and
    `stream/decode-router` keep their programs. End to end, wall time,
    kaun's `GptOss/decode-step-Q8_0-cache-1024` 7.98-8.02 and 7.96-7.99 ms,
    124,448 words a call each, and `GptOss/prefill-512` 185.0-185.9 and
    183.0-184.0 ms;
  - past 16384 threads the splits lose reuse: at 65536 `dense-bf16-64`'s
    third kernel is 1.42x slower and `matmul_half_1024` 1.97x, so fudge's
    window is (4096, 16384]. The lower edge shows from the L2 as well as
    from memory, so it moves with SMs times clock and with bandwidth; the
    RTX 4090 and L40 lie within 4x of fudge by both (the Ada whitepaper's
    tables);
  - the local split as tinygrad's, unfloored: a decode step's attention
    values 1.19x slower, the key projection 1.34x warm and 1.6-1.7x cold,
    the routed products 1.14x, for 1-4% elsewhere. Its cost follows the
    warps on the busiest SM, a function of the part's units: 512 is the
    smallest floor that refuses every harmful case of fudge's rows;
  - kept from before: a decode step's attention values, 64 by 64 outputs
    over 1024 slots, take 32 workgroups and 23.6 us where tinygrad's splits
    leave one and 68.9 us; a prefill's router, 512 tokens by 32 experts over
    2880, 128 workgroups where tinygrad's leave 8.
  The band where the bound is wrong: on parts the size of an RTX 4060 (24
  SMs, 272 GB/s) the model puts the window at about (1000, 3900] threads,
  where 16384 may cost 1.4-2x. Unmeasured; a row on a small part is owed.
  The floor moves the order of a sum (97-98% of the key projection's and
  `matmul_half_512`'s outputs differ from 100 workgroups', by at most 1.2e-5
  on values of 3.5), so it is fixed per target: a floor read from the part
  would give two parts of one target different bits (D89's constants).
  Metal and AMD take the same value; theirs are to measure on a Mac's GPU
  and on nonnormal's R9700, which fudge lacks.
- **Pinned by:** `Tolk.Heuristic`'s cases, on CUDA: `matmul_half_1024`,
  `matmul_half_512`, `matmul_half_256`, `attention_values` and
  `matmul_half_variable` (a variable counts the workgroups), with the
  goldens from tinygrad with D175 applied by its generator; its `the tensor
  core's floor › the splits after a tensor core are those that leave its
  floor of workgroups` and `› a larger floor after a tensor core never
  leaves fewer workgroups`, over every recorded core kernel, CDNA's
  included; the tolk bench's `stream/decode-keys`, `stream/dense-bf16-64`
  and `stream/key-projection` rows on fudge.

## D176. A folded selection or negation keeps a NaN's bits

- **tinygrad:** `uop/ops.py:1386-1392` (`exec_alu`), which truncates every
  result to its type through `dtype.py:293-297` (`truncate`), whose
  conversions to `float`, `half` and `bfloat16` quiet a signalling NaN; the
  patch's `exec_alu` (`test/gen/tinygrad.patch`) applies tolk's rule before
  the goldens are generated.
- **tolk:** `Ops.exec_alu`, whose `Where` and `Neg` give a NaN as
  `Dtype.const` holds it, and `Shape.fold_const_alu`, which reads a float
  constant operand as `Dtype.const` holds it, its bits (D27).
- **Differs:** a folded selection is the NaN it picks, and a folded negation
  that NaN with its sign flipped, a signalling NaN staying one, in a float of
  16 bits or more; tinygrad's quiets a signalling one. Every other operation
  still truncates its result. tolk has no absolute value or sign copy among
  its operations: rune computes both on a float's bits (`Lower_bits`).
- **Reason:** (b). Under the NaN rule a selection and a negation keep a NaN's
  bits, as LLVM's LangRef states ("Behavior of Floating-Point NaN values"):
  "if there is no floating-point operation being performed, NaN signs, quiet
  bits, and payloads are preserved", and `fneg`, `llvm.fabs` and
  `llvm.copysign` "act directly on the underlying bit representation and
  never change anything except possibly for the sign bit". nx's `where` keeps
  a signalling NaN (D62), and rune lowers it to `Ops.where`
  (`Lower_arith`), so a selection tolk folds must give the bits the kernel's
  selection gives. The `Neg` half has no rune call site: rune negates a float
  on its bits (`Lower_bits.negated`), and `Op.Neg` reaches the folder only
  from `Decomp_op`'s rewrite of `x * -1`. It rests on D27's rule that a
  rewrite keeps a graph's value: a `Neg` the folder leaves renders as C's
  `-x`, which flips the sign bit alone, so a folded one must too.
- **Pinned by:** `Tolk.Ops › exec_alu › a selection keeps a NaN's bits and a
  negation flips only its sign, a signalling NaN's included`, over NaNs of
  every sign and payload of `half`, `bfloat16`, `float` and `double`; and
  `Tolk.Symbolic › symbolic_simple › constants › a folded selection keeps a
  signalling NaN's bits, and a negation flips only its sign`. On rune:
  `Lower_arith › a folded selection keeps a NaN's bits, and a negation or an
  absolute value sets only its sign`, compiled selections by a constant and by
  a folded comparison between a NaN constant and 1, and its negations and
  absolute values, over signalling and quiet NaNs of both signs of `float16`,
  `bfloat16`, `float32` and `float64`.

## D177. A name bound to several values is passed by each call that reads it

- **tinygrad:** `schedule/__init__.py:282-288` (`create_linear_with_vars`),
  which raises "bind mismatch" when a schedule's kernels read one variable
  name bound to two values, and `:100-112` (`resolve_linear_call`), which
  substitutes a bound variable into its kernels unbound, read by name from
  the run's `var_vals`.
- **tolk:** `Schedule.create_linear_with_vars` (`bound_several`),
  `Schedule.resolve_linear_call` and `Schedule.pass_values`;
  `Tolk_engine.var_word`, and the grid and split size of
  `Tolk_engine.launch`, which read a nameless scalar from its call's slot.
- **Differs:** where the graph binds variables of one name to several values,
  a kernel that reads one takes it as a nameless parameter of its range, in a
  slot past its arguments, and its call passes the bound variable there, as a
  loop's call passes its trip (D140). The name is not among the schedule's
  variables, and the engine reads each bound variable apart, so each launch
  computes with its call's value, unless a run binds the name. A kernel
  takes such parameters in the order its body reaches them, so kernels that
  differ in their values alone, or in the order the schedule met those
  values, are one program. A name bound to one value is read by name, as
  tinygrad reads it.
- **Reason:** (b): rune's compiled programs over several devices
  (`Pieces.split`, `Compile.moving`) bind each device's window start, of a
  move or of a loop's coordinates, to a variable named by its bound, so that
  every device's kernel is one program (D170) and the data-parallel step
  takes as many programs over 2 to 16 memory devices.
- **Pinned by:** `Tolk.Schedule › create_linear_with_vars › rules › two
  variables of one name bound to two values are passed by the calls that read
  them`, `› kernels that read two names bound to several values alike are
  one program, whatever order the schedule met the values in` and `› loops
  of calls › loops whose calls read one name at two values each pass their
  own`; `Tolk_engine › link and run › one program launched
  on several devices computes each device's window at its start` and `›
  kernels whose sizes are one name of two values run each its own`, and the
  same on each GPU; rune's `Rune.jit over
  several devices › maps over a cut axis › per-device code takes as many
  programs over 2 to 16`, `› programs › a data-parallel step takes as many
  over 2, 4, 8 and 16`, `› loops › a step that reads its coordinates gives
  eager's bits`.

## D178. A Windows host's code is compiled for Windows' calling convention

- **tinygrad:** `runtime/ops_cpu.py:68` (the CPU's arch, its machine and
  `native`); `runtime/support/compiler_cpu.py:18-22` (`ClangCompiler.compile`,
  which compiles for `<machine>-none-unknown-elf` on every system);
  `renderer/cstyle.py:28` (an `int64` literal ends in `l`), `:131`
  (`CStyleLanguage.abi`, `""`, which `ClangRenderer` at `:287` keeps) and
  `:146-149` (`CStyleLanguage.type_map`, where an `int64` is a `long`).
- **tolk:** `Tolk_engine.target`, whose host arch on this machine under Windows
  is `x86_64-windows,native` (or `arm64-windows,native`);
  `Compiler_cpu.clang` and `Compiler_cpu.win64`; `Cstyle.clang`, whose
  language under `x86_64-windows` names a 64-bit integer `long long`, writes
  its literals with `ll`, and declares the kernel `sysv_abi`;
  `Codegen.host_entry`, which declares the entry `sysv_abi` there.
- **Differs:** a host target names Windows after its machine. Under
  `x86_64-windows` the object is compiled for
  `x86_64-w64-windows-gnu-elf` with `-mno-stack-arg-probe` and
  `-mno-red-zone`, so that it calls every function by Windows' convention,
  and its kernel and entry are declared System V's. Windows is LLP64, so a 64-bit integer is a
  `long long` and its literals end in `ll`. tinygrad compiles for no system
  everywhere, and its CPU arch names none. Every other target renders and
  compiles as before.
- **Reason:** (c). rig calls linked code by System V and takes each symbol the
  object does not define from the process, whose functions follow the
  platform's convention (`Rig.section-host_abi`, `Rig.section-host_symbols`).
  Compiled for no system, the object called them by System V: `frexp` read
  its pointer from `rdx` where the kernel put it in `rdi`, and wrote the 32
  bytes above its return address that Windows lets a callee use, so `CG ›
  calls in a kernel › an external function writes through its pointer
  argument` crashed on windows-2025. Clang also calls functions the source
  does not name: `fma` and `trunc` without FMA or SSE4.1, `fminf` and
  `fmaxf` for a half selected against a constant, half conversions without
  F16C, `memcpy` for a large copy. Compiled for Windows, every such call
  follows Windows' convention, as `Rig.section-host_abi` states such code.
  rig commits a thread's stack before it runs linked code, so the code
  probes none, and Windows may write below the stack pointer. The system comes from the device
  (D95): this machine's host runs this process's system. A remote Windows
  host's target names none, so its programs call by System V, until rig
  states a machine's system. A program's key holds its target (D113), so it
  holds the system.
- **Pinned by:** `Tolk.Cstyle › x86_64 Windows`: every x86_64 Clang case
  renders with no LP64 `long` and as its golden but for its 64-bit types,
  literals and the kernel's `sysv_abi`; `› the kernel is declared System
  V's`; `› a 64-bit integer is a long long and its literals end in ll`; `›
  an object is compiled for Windows' triple and calls no stack probe`; and,
  on an x86_64 host, every Clang case compiled for Windows loads and every
  interpreted one that calls nothing of the process runs through rig's host
  and writes what the interpreter computes, which a System V entry's
  absence breaks. `Tolk_engine › target › the host compiles C for Clang on
  its own processor, naming its system under Windows, whose functions
  follow a convention of their own`. On windows-2025 CI, `CG › calls in a
  kernel`.

## D179. An index of a reshape's added leading axes is its source

- **tinygrad:** `schedule/prepare.py:65-73` (`_mop_index`), whose shortcut for
  a reshape whose added leading axes are indexed whole tinygrad 8b4d8c026
  deleted as dead code: the index is then its source's index by nothing.
- **tolk:** `Prepare.mop_index`.
- **Differs:** an index of a reshape that adds leading axes, by indices of
  those axes alone, is the reshape's source, where tinygrad makes it an index
  of the source by nothing.
- **Reason:** (b). rune's lowerings index such reshapes (`Compile`'s reads at
  an offset, `Lower_index`, `Lower_linalg`), which tinygrad's tensor graphs
  never build: rangeify leaves an index by nothing of a value it stores
  unstored (`Indexing.pm_apply_rangeify` rewrites no index), and
  `Rangeify.cleanup_dead_axes` then raises on rune's complex contractions
  ("cannot reshape (2,) to (1,)").
- **Pinned by:** `Tolk.Prepare › pm_mops › rules › an index of a reshape's
  added leading axis is its source`; rune's `contractions › complex
  contractions hold their bound`.

## D180. A stored value of no device lives on each device that reads it

- **tinygrad:** `schedule/indexing.py:142-145` (`pm_fix_deviceless`) and
  `:312-313` (`run_rangeify`), which place every stage of no device on the
  sink's device, the first device among its sources.
- **tolk:** `Indexing.fix_deviceless`.
- **Differs:** a stage of no device, an arange or a table computed from
  constants that rangeify stores, takes the device of the nearest node above
  it that has one, the device of the kernel that reads it. Read by kernels on
  several devices, it is one stage per device, each computed there. A stage
  that no node with a device reads, or that only a store to a disk reads,
  lives on the sink's device. On a sink of one device the two rules agree.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.lower` of `Pieces.split`) schedule one sink whose kernels each
  compute on one device. A constant table their kernels share, as the shifts
  that unpack a sub-byte bitcast (`Compile.unpack`, `Lower_arith`'s
  bitcast) or an `exp2` of an arange, was stored on the first device and
  read by the others' kernels, which `Schedule` refuses, so the call
  evaluated.
- **Pinned by:** `Tolk.Schedule › create_linear_with_vars › rules › a stored
  value of no device lives on each device whose kernels read it`; rune's
  `Rune.jit over several devices › values computed from constants`.

## D181. A tensor core's addend broadcasts before it moves past a permute

- **tinygrad:** `codegen/__init__.py:105-108` (`pm_wmma_add`: the sum of a
  permuted WMMA, or of a permuted reshape of one, and an addend moves the
  addend through the inverse permute, which takes the permute's axes).
- **tolk:** `Codegen.pm_wmma_add`; `test/gen/tinygrad.patch`, which gives
  tinygrad's rules the same broadcast before the goldens are generated.
- **Differs:** the addend is first broadcast to the permute's shape, and the
  rules leave a sum whose addend does not broadcast there. Before
  `expand_broadcast` runs an addend may have fewer axes: a product that one
  step of a core sums, with no loop left to accumulate it, reaches the rules
  as a permute of the core plus the scalar the sum ends with, and tinygrad
  raises (`IndexError: dim=1 out of range [-1, 0]`). At the pin, tinygrad
  raises so on a float16 product of 16 by 16 summed into float32 plus `1.0` on
  CUDA sm_89, with or without its splits after the core; 32 by 32 compiles.
- **Reason:** (b). rune's contraction lowering (`Lower_reduce.close`) adds
  `+0.` or the initial value to every float sum's result, a scalar, which D24
  keeps. rune widens float16 and bfloat16 operands to float32 on their bits
  today, which no core takes; once it converts them by tolk's casts, from the
  change that lands after this entry, its products reach CUDA's cores through
  D29, and a 16 by 16 by 16 product raised in lowering and was evaluated.
- **Pinned by:** the Codegen suite, `stages › … › matmul_{half,bf16}_tile_*`
  (a widened 16 by 16 product plus `0.`, on every target), and `tensor cores
  on CUDA`: a product of one tile, one of 256 by 16 by 256, which the splits
  after the core reshape, and the law over 1 to 4 tiles a side, of float16 and
  bfloat16, equal to the exact product on fudge's RTX 5000 Ada.

## D182. A copy stored into a run of storage lands in place

- **tinygrad:** `schedule/prepare.py:206-208` (a copy stored into storage on
  its device is the store, only where the destination has buffer identity)
  and `:211` (`copy_to_anon_store`, any other copy lands in storage of its
  own); `schedule/__init__.py:170-180` (`pm_copy_from_store`, which lowers a
  copy kernel that reads and writes both buffers at its range alone).
- **tolk:** `Prepare.storage_run` and its use in `Prepare.earliest_rewrites`;
  `Schedule.run_start` and `Schedule.copy_runs` in
  `Schedule.pm_copy_from_store`.
- **Differs:** a copy stored into a run of consecutive elements of storage on
  its device that shrinks and reshapes view (`Prepare.contiguous_view`, the
  predicate that already turns a copy of a run into a copy of a view) is the
  store, so the copy lands in the
  run. Its kernel then reads its source at its range and writes the
  destination at its range plus the run's first element, and lowers to a copy
  between views of the call's arguments, each the run it copies. tinygrad
  copies into storage of its own and stores that into the run with a kernel,
  so both are live at once. A copy stored into a run of a bitcast of storage
  is a copy into a view of it (D184): the copy moves bytes, so it lands in
  place too. A copy of one element indexes each side at its element and
  lowers the same way. No recorded program stores a copy into a part of
  storage, so the patch leaves tinygrad's rules as they are.
- **Reason:** (b): rune's split assembly of a gathered value (RFC 0005),
  which stores each device's piece into its run of the whole. With that
  assembly and kernels ordered as D183 says, fully sharded training's
  backward call of one layer at d = 512 held 2.062, 1.594 and
  1.422 layers beside its arguments over 2, 4 and 8 memory devices with each
  piece copied apart, and 1.812, 1.531 and 1.391 landing in place, against
  RFC 0005's bound of 1.875, 1.562 and 1.406 (`rune jit › fully sharded`).
- **Pinned by:** the Schedule suite (`test/schedule/schedule`): `copies into
  runs › a copy stored into rows of storage on its device copies into them`,
  `› a copy stored into a first row copies into it`, `› a copy stored
  into columns lands in storage of its own`, `› a copy stored into a run of
  a bitcast copies into a view of it` and `› a copy of one element stored
  into filled storage copies into it`; the Engine suite's `link and run › a
  copy stored into rows of a buffer writes them alone`.

## D183. In a schedule made per run, ready kernels run by their number, and one that frees storage first

- **tinygrad:** `schedule/__init__.py:66-78` (`create_schedule`'s linearize:
  Kahn's algorithm over a FIFO `deque`, so ready kernels run in the order
  they became ready).
- **tolk:** `Schedule.storage_order`, chosen by `create_schedule
  ~by_storage` and `create_linear_with_vars ~by_storage`, and the storage
  `Schedule.schedule` sees written; `Memory.taken`, the storage an entry
  takes as the memory plan counts it; rune's `Compile.link`, which asks for
  it for a schedule linked per run.
- **Differs:** in a schedule whose storage is made at each run (D173), a ready
  kernel runs by its number, the order the walk of the sink meets it, so a
  value is consumed soon after it is made, as the program wrote it. A schedule
  linked once, as rune links every program it does not split, keeps tinygrad's
  order, and scheduling it costs what it did. A ready kernel that frees
  storage, being the last to take storage an earlier kernel made, and that on
  no device makes more than it frees, runs before the others. The storage a
  kernel writes lives from the first kernel that takes it to the last; storage
  only kernels read lives throughout. tinygrad runs every kernel whose inputs
  are ready breadth first: a closing slices its whole partial into every piece
  before any is copied or summed, and computes a partial gradient before the
  last reader of the gathered weights. Picking costs one pass over each ready
  kernel's storage, so a schedule of `k` kernels costs `O(k²)` at worst.
- **Reason:** (b): rune's fully sharded training (RFC 0005's memory bound),
  whose split programs' storage is planned by each buffer's life (D173). With
  the split assembly as storage and copies landing in place (D182), the
  backward call of one layer at d = 512 held 2.625, 2.312 and 2.156 layers
  beside its arguments over 2, 4 and 8 memory devices in tinygrad's order,
  3.625, 3.312 and 3.156 by number alone, and 1.812, 1.531 and 1.391 in this
  order, as much as with each buffer of its own, against the bound of 1.875,
  1.562 and 1.406 (`rune jit › fully sharded`). The order interleaves
  devices, so a data-parallel step over 16 memory devices makes 1411
  `Submit` steps where tinygrad's makes 84; preferring, among each kind of
  ready kernel, one on the last kernel's device made 1016, and raised four
  of rune's five `fully sharded` cases over their bounds.
- **Pinned by:** the Schedule suite (`test/schedule/schedule`):
  `create_schedule › order by storage › a kernel that frees storage runs
  before one that only makes it`, `› pieces of one value are each read
  before the next is made` and `› without it, ready kernels run as they
  become ready`; rune's `fully sharded` tests. Any other schedule keeps
  tinygrad's order, as every recorded schedule does.

## D184. A store into a run of a buffer stores into a view of it

- **tinygrad:** `schedule/__init__.py:222-223` (`pm_callify_ctx_collect`):
  `contiguous_mops_to_view` makes a view of a buffer for a copy or a stage
  of movements of it, and for a store only into a bitcast of it. A store into
  a shrink of a buffer indexes the buffer at the shrink's offset in its
  kernel, so each offset is a program of its own.
- **tolk:** `Schedule.view_of`, through `Schedule.pm_callify_ctx_collect`.
- **Differs:** a store into movements or a bitcast of a buffer that select a
  contiguous run of its elements stores into a view of that run: the kernel
  takes the run as its buffer, and a copy stored there is a copy into it
  (`Prepare`'s rule for a copy stored into storage on its device). Stores of
  one shape into runs of one buffer at other offsets are one program. A store
  into a buffer's state after earlier stores keeps indexing it, since a view
  of the buffer would not be ordered after them, and so does a store whose
  value reads the buffer in its own kernel, short of the stored states of
  other storage, which the kernel split stages by the buffer's identity. A
  run may start at bound variables, whose values its call knows
  (`Prepare.contiguous_view` checks the run at those values): a store into
  it stores into a view at that start, whose kernel knows the start modulo
  what the variables move it by (D54), so stores of one shape at starts of
  other alignments are one program too. A copy or a stage of a view still
  needs a constant start.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.assembled`) hold an assembly as a buffer filled by a store per
  piece, each device's own piece at its device's offset. Indexed at the
  offset, a data-parallel step over 2, 4, 8 and 16 memory devices made 19,
  31, 55 and 103 programs; through views, 11 each, and 11, 13, 13 and 13
  once rune reads a closing's own slot from storage of its own, against
  rune's promise that they do not grow with the devices (D170). A view at a
  constant start keeps that start's alignment, and a closing's parts lie
  at starts of several: a sum over an axis cut across 2, 4, 8 and 16 memory
  devices, its 2 + n wide result's parts stored at their starts, made 4, 7, 9
  and 9 programs, and 4, 7, 7 and 7 with each device's own part stored at a
  bound start. Indexing the whole buffer at that start instead held 3.02 MB
  per device in fully sharded training over two devices, against the 2.29 MB
  its bound allows and views keep.
- **Pinned by:** `Tolk.Schedule › contiguous_mops_to_view › stores into runs
  of a buffer at a bound start store into views of one alignment, whatever
  the start`, `› stores of one shape into two runs of a buffer share a
  kernel`, `› a store into contiguous rows of a buffer stores into a view of
  them`, `› a store into rows of a buffer after a store has no view`, `› a
  store into rows of a buffer of a value that reads it has no view`, `› a
  store into rows of a buffer of a value that reads it through other storage
  stores into a view` and `› a store into columns of storage has no view`;
  `Tolk_engine › link and run › a store into rows of a buffer of a value
  read through other storage writes what it computes`; `Tolk.Schedule ›
  create_linear_with_vars › merged calls › the work on a schedule of many
  outputs is linear in their number`; rune's `Rune.jit over several devices ›
  programs` and `› maps over a cut axis › partials give eager's
  bits, their programs do not grow`, and `rune jit › fully sharded`.

## D185. A ready kernel that only frees runs before one that also makes storage

- **tinygrad:** `schedule/__init__.py:66-78` (`create_schedule`'s linearize,
  Kahn's algorithm over a FIFO `deque`).
- **tolk:** `Schedule.storage_order`'s `pick`, for a schedule made per run
  (D183).
- **Differs:** of the ready kernels that free storage (D183), one that makes
  no storage runs first, by its number; only where none does, one that on no
  device makes more than it frees. A kernel holds what it makes beside what
  it reads, so one that frees much but makes some raises what a schedule
  holds at once where one that only frees does not. tinygrad has no such
  order.
- **Reason:** (b): rune's fully sharded training (RFC 0005's memory bound),
  whose split programs' storage is planned by each buffer's life (D173).
  At the end of a closing, the copy of a device's own slot of the last piece
  is the last reader of the partial and makes the slot's storage; the sum of
  the piece before it only frees. Run first, the copy held three slots
  beside the partial: the backward call of one layer at d = 512 held 1.938,
  1.594 and 1.422 layers over 2, 4 and 8 memory devices, and 1.812, 1.531
  and 1.391 in this order, against the bound of 1.875, 1.562 and 1.406.
- **Pinned by:** the Schedule suite (`test/schedule/schedule`):
  `create_schedule › order by storage › a kernel that only frees runs before
  one that also makes storage`.

## D186. The host's accumulators take half its vector registers

- **tinygrad:** `codegen/opt/heuristic.py:111-137` (`hand_coded_optimizations`'
  upcasts of output axes: by 3 or 4, of an axis that some access does not
  index while it indexes every upcast axis, while 1024 elements remain and
  the kernel has fewer than 32 lanes) and `:139-150` (the unroll of the last
  reduce axis, whole when it is at most 32, by 4 otherwise, while the kernel
  has fewer than 64 lanes).
- **tolk:** `Heuristic.held_registers` (`held`, `held_share`, `short_trips`,
  `computed_bytes`), `Heuristic.fits`, `Heuristic.stored_contiguous`,
  `Heuristic.upcast_contiguous`, and `Heuristic.upcast_more`'s and
  `Heuristic.unroll`'s use of them; `Renderer.registers`, which
  `Cstyle.clang` sets from `Compiler_cpu.registers`; `test/gen/tinygrad.patch`,
  which gives tinygrad the same before the goldens are recorded.
- **Differs:** on the host (the `CPU` device of a renderer that knows its
  vector registers: 16 of 32 bytes on x86-64 with AVX, 16 of 16 without, 32
  with AVX-512, 32 of 16 on arm64), a reduce kernel's lanes are counted in
  registers, a register holding as many lanes of a reduce's type as fit, at
  4 bytes for a narrower type, and no more than its stores write at
  consecutive addresses. The accumulators of the kernel's hottest loop
  (`Heuristic.held_registers` states which) take at most half the
  registers. The upcasts after the masked and shared ones (D92) and the
  unroll take the budget in place of tinygrad's lane counts:
  - the first output upcast, while 1024 elements are left, takes the axis
    the stores write at unit stride, when every access that reads it reads it
    at unit stride, to a host vector of the accumulators' type (D188), the
    largest power of two that divides it and fits;
  - the upcasts by 3 or 4 follow while they fit, without tinygrad's 32
    lanes, while 1024 elements are left as tinygrad's do;
  - when none is left, an output axis not yet upcast is upcast by 2 if it
    fits and some access that indexes a serial reduce axis does not index it
    (the fill; among such axes, tinygrad's order: fewest accesses indexing
    the axis, then the least sum of their strides);
  - the unroll takes what fits, while the lanes upcast fill fewer registers
    than the target has, where tinygrad's take fewer than 64 lanes, 16
    registers of 4: a partial unroll holds nothing, a whole one its lanes
    while another reduce of the loop still loops.
  Other devices, and a host renderer that knows no registers (riscv64's),
  upcast and unroll as tinygrad does. The register file is the compile target's,
  which the compiler's cache key holds, so a program follows the
  instruction set it is compiled for and no fact of the part it runs on.
- **Reason:** (d): lorenz_simple's step (sofo-raven) and the host's
  contractions. Each accumulator carried around a loop is a dependency chain:
  Intel's optimization manual (248966-051, §15.15.1, Rule 31; Golden Cove's FMA
  on ports 0 and 1 at latency 4) asks for latency × pipes independent chains, 8
  vector accumulators, and Rule 38 for loop-carried values in registers before
  invariants; BLIS (Low et al., TOMS 2016, §4.2.2-4.2.3) takes the smallest tile
  above that floor with the contiguous axis a multiple of a register's elements,
  as its Haswell kernels hold 6 × 16 floats in 12 of 16 registers. Clang packs a
  register only as wide as the stores it reads (D187), so a register holds no
  more lanes than the stores write at consecutive addresses. Timed on fudge, one
  P-core (`taskset -c 4`, the variants interleaved under the timing lock,
  loaded, three runs), q1-q3 against origin/main: a float32 matmul of 128³
  (`host/matmul/heuristic`) 69.8-71.9 µs to 30.0-30.5, its float64 one 74.6-75.8
  to 67.4-69.1, `conv` 140.2-142.2 to 82.2-83.4, `conv_half` 1147-1151 to
  980-986, `tangent_products` 338.6-375.0 to 49.9-50.5, `paired_products`
  85.9-86.1 and 85.5-85.6 (the same optimisations); lorenz_simple's four-sum
  kernel (`lorenz/r_128_16_3_4_4_3_3_400_400/heuristic`) 6.87-7.02 ms to
  6.16-6.25, its two-sum kernel 3.31-3.37 ms to 2.51-2.65; `r_32_3_2_4_3_400`,
  `routed_blocks`, `routed_tiles` and `vecmat_decoded` keep their optimisations
  and times, `experts_down` 0.6 to 0.5 µs, the stack and pad kernels unchanged.
  The fill, re-timed at this render: the four-sum kernel takes 6.23-6.68 ms with
  it and 8.01-8.40 without, the two-sum kernel 2.50-2.73 ms and 3.13-3.32. The
  float32 matmul's unroll by 4 at 64 lanes, which tinygrad's 64 lanes refuse,
  takes it from 50.0-50.1 µs to 34.1-36.2 (one run). D91's 32 lanes were this
  budget at a render that filled half of each float32 register: AVX2's 8
  registers of 4 floats. The whole-unroll clause, timed at that render on the
  real path (this campaign's audit and its skeptic, two runs each): the
  two-sum kernel at 16 lanes takes 3.31-3.42 ms, and 7.92-8.32 with one of
  its sums unrolled whole while the other loops, D109's case, which the
  clause refuses; `tangent_products` at 16 lanes takes 277-291 µs, and 183-194
  with its sums' one axis unrolled whole, which removes every loop, and which
  the clause takes.
- **Pinned by:** the Heuristic suite: `the host's budget › on AVX2 a
  kernel's accumulators hold` (every case), `the host's budget › the register
  budget › the accumulators of the hottest loop take at most half the
  registers` and `› the kernels the budget is checked on have each of its
  shapes`, `the host's budget › on the host a larger register file never
  holds fewer output lanes` and `› on the host a reduce of doubles over a
  stored axis of a multiple of 32 takes half the lanes of floats, in the
  registers floats fill`, over every recorded kernel and lorenz_simple's,
  on files of 16 × 16, 16 × 32, 32 × 16 and 32 × 64; `the optimisations
  chosen are tinygrad's › … › the optimised kernel`, the `cpu` (16 × 16) and
  `avx2` (16 × 32) cases, recorded from the equally patched tinygrad; the
  Compiler_cpu suite's `the vector registers, their number and bytes, of`
  (every case) and `without Clang › an x86_64 processor has SSE's 16
  registers of 16 bytes`; the tolk bench's `host/*/heuristic` and
  `lorenz/*/heuristic` rows on fudge.

## D187. The host's stores take a full vector register

- **tinygrad:** `codegen/late/coalesce.py:136-145` (`memory_coalescing`'s
  lengths: 4 or 2 elements of float32, float16, int32, uint32 and the 8-bit
  floats, 8 of float16 under `ALLOW_HALF8`, none of the others).
- **tolk:** `Coalesce.memory_coalescing`, through `Renderer.registers`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the
  goldens are recorded.
- **Differs:** where the renderer knows its vector registers, loads and stores
  of float64, float32, float16, int32, uint32 and the 8-bit floats merge into
  accesses of a power of two of elements up to one register: 8 floats or 4
  doubles of 32 bytes on AVX2, 4 floats or 2 doubles on SSE and NEON. A store of
  doubles merges so only when each value is a load, as an accumulator's read is;
  doubles computed lane by lane are stored one by one, as
  tinygrad stores them. On the CPU device a buffer is taken to start on a
  boundary of that width (D54). Other renderers merge as tinygrad does.
- **Reason:** (d): the host's contractions and lorenz_simple's step. Clang's
  straight-line vectorizer packs values into registers as wide as the stores it
  reads them from: a float32 matmul's inner loop with its 16 stores as `float4`
  takes 65 instructions and spills 11 values, as `float8` 31 instructions and
  none (`clang -S` with tolk's flags), and merging the loads alone changes
  nothing. A load as wide as a register feeds it whole: `tangent_products`
  replays in 51.0-55.6 µs with its loads of one register against 190.5-193.5 µs
  with loads of 16 bytes, `conv` in 84.4-90.4 µs against 109.0-110.0, and a
  float32 selection of zero over 4M elements (`host/select_zero/heuristic`),
  whose lanes stay as they were, in 737-834 µs against 862-953 on origin/main.
  Cost: a kernel whose loads of a register are read lane by lane into a reduce
  along them takes longer: `max_rows` (`host/max_rows/heuristic`) 2.8 µs to
  3.4-3.5, its loads of 8 floats split back into the four rows it reduces,
  gpt-oss's MXFP4 gate and up product 12.65-12.72 ms to 13.32-13.42, and
  lorenz_simple's four-sum kernel with the beam search's 48 lanes
  (`lorenz/r_128_16_3_4_4_3_3_400_400/searched`) 6.53-6.60 ms to 7.22-7.28,
  where the wider stores pack accumulators past the registers; kept because the
  gains above are larger, and the heuristic's own four-sum kernel runs in
  6.16-6.25 ms. Merged, doubles computed lane by lane made Clang pack the
  computations behind them: rune's float64 normal sampler
  (`host/normal_f64/heuristic`, rune's compile bench `rng/normal-f64`), one
  kernel of 8 lanes that hash and convert 32- and 64-bit integers, replayed
  in 26.8-28.2 µs against 13.4-13.9 on origin/main, its Clang compile took
  203-220 ms against 182-196 and the row 574-593 against 472-519; stored one
  by one, it runs in 13.8-14.1 µs, and eight nested float64 sines, whose
  lanes are a vector, in 117.9-120.8 against 115.4-118.4. Floats computed
  lane by lane keep the merge: lorenz_simple's two-sum kernel stores them,
  and runs 2.99-3.15 ms with it against 3.50-3.62 without. Timed as in D186.
- **Pinned by:** the Coalesce suite's `lengths › eight consecutive stores fill
  registers of` (every case), `› eight consecutive stores of lanes computed
  apart, on 32 bytes, are` (every case) and `› without ALLOW_HALF8 or known
  registers, eight halves are two loads of four`, and its kernels' goldens,
  recorded from the equally patched tinygrad for SSE's 16 bytes; the tolk
  bench's `host/*/heuristic` rows on fudge.

## D188. A host vector spans 64 bytes

- **tinygrad:** `codegen/opt/heuristic.py:111-137` (the upcasts of output
  axes by 3 or 4), `:139-150` (the unroll of the last reduce axis, and of a
  second one when both are at most 3) and `:152-154` (the upcast by 4 of a
  kernel that has none).
- **tolk:** `Heuristic.host_vector_bytes`, `Heuristic.parallel_operations`,
  `Heuristic.upcast_vectors`, `Heuristic.upcast_contiguous` and
  `Heuristic.unroll_vector`; `test/gen/tinygrad.patch`, which gives tinygrad
  the same before the goldens are recorded.
- **Differs:** on the host a group of lanes computed as one vector spans 64
  bytes of its widest type, two registers of AVX2. A kernel with no reduce
  takes, after its masked axes, one upcast of its last output axis that 2
  divides, by the largest power of two, at most 16, that keeps its lanes
  within 64 bytes of the widest type it reads, writes or computes, and again
  while they have room: 16 float32 lanes, 8 float64, 32 float16. It takes
  none when its stored values hold 5 arithmetic operations or more for each
  of their longest chain, or when it computes a type the target emulates.
  A reduce kernel's first upcast takes its stored axis to 64 bytes of its
  accumulators (D186). When the last reduce axis is at most 3 and wholly
  unrolled and the next one is larger than 3, the next is unrolled by the
  largest power of two that divides it and keeps the unrolled lanes within
  64 bytes of the reduce's type, if the budget holds: 8 beside a pair of
  float32 lanes, 4 beside 3. tinygrad upcasts output axes by 3 or 4 while
  1024 elements remain and fewer than 32 lanes, by 4 a kernel that has none,
  and unrolls the next reduce axis only when it is at most 3 too.
- **Reason:** (d). Elementwise kernels (RFC 0025's special functions, from
  D141): lanes give the core independent chains while each waits on its latency,
  and computed as vectors (D141) they compile once. Eight nested float64 sines
  over 1024 elements replay 1.25 times as fast at 64 bytes as at 32 and as with
  none (fudge, one P-core, two runs); 128 bytes replay a further 1.05 times at
  1.6 times Clang's time, and Clang takes 0.34-0.45 s at 64 bytes against
  0.35-0.54 s with no lanes. A kernel whose operations already run 5 at a time
  gains nothing from lanes and pays for them in compile time: `lgamma`,
  `log_betainc` and its gradients take none and compile as fast as without the
  upcast (2.7 s and 4.9-8.9 s cold, on an M1 Max, where 5 was measured; a sine
  is one operation when the heuristic runs, so float64 `sin` takes lanes). A
  reduce kernel's stored axis at 64 bytes runs a float64 matmul in 70.8-73.7 µs
  against 96.9-97.3 µs at one register and a float32 one in 30.3-33.7 against
  37.1-37.4, the 6 × 16 floats of BLIS's Haswell kernels; on 16-byte registers,
  SSE's and NEON's, a vector is four of them, unmeasured. From D143, nx.quant's
  MXFP4 product reduces over a byte axis and the pair of codes each byte holds:
  the pair alone left each iteration two decoded codes per row and its loop, its
  scale decode and its loads outweighed them; unrolled to a vector, the dense
  product of a checkpoint's codes ran 2.6-3.0 ms against 6.7-9.8 ms and one
  token's routed product 11-21 ms against 37-55 ms on an M1 Max. Timed on fudge
  as in D186: `routed_blocks` 12.7-13.4 µs, `routed_tiles` 38.5-40.2,
  `experts_down` 0.5, the nested sines 101.6-105.7 µs and a float32 selection of
  zero over 4M elements 737-834 µs. kaun's `Quant/host/*` rows hold the MXFP4
  products; the tolk bench's `host/sin_nested_8/heuristic` and
  `host/select_zero/heuristic` the elementwise kernels, and rune's compile bench
  `trig/sin-nested-8` their compile. Whether memory-bound kernels take lanes at
  all is open (TODO).
- **Pinned by:** the Heuristic suite's `the optimisations chosen are tinygrad's
  › … › the optimised kernel`: the host cases `exp_float`, `exp_double`,
  `add_half`, `add_uchar`, `cast_to_double`, `compare`, `add_24`, `add_17`,
  `wide_tree`, `long_chain` and every host case without a reduce, and
  `vecmat_decoded`, `gpt_oss_gate_up`, `routed_blocks`, `routed_tiles` and
  `experts_down`, on `cpu` and `avx2`, recorded from the equally patched
  tinygrad; the tolk bench's `host/*/heuristic` rows and rune's compile bench
  `trig/sin-nested-8` on fudge. On rune: missing (the old rune's `lower_index ›
  quantised products › a product decodes a checkpoint's codes 16 at once on the
  host` went with its rewrite).
## D189. A sum gated on an index read at its range is not collapsed

- **tinygrad:** `codegen/simplify.py:127-131` (`pm_reduce_load_collapse`'s
  rule for a sum of a value gated on an index equal to the range), which
  substitutes the index for the range in the value whatever the index
  depends on.
- **tolk:** `Simplify.pm_reduce_load_collapse`, through
  `Simplify.load_collapse` and `Rangeify.collapses`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same.
- **Differs:** the rule declines an index that depends on the range. Such an
  index, as `b[r]` in `sum over r of (b[r] == r ? x[r] : 0)`, selects any
  number of trips, so the sum is no value at one index, and substituting an
  expression of the range for the range does not terminate. tinygrad raises
  "graph_rewrite stalled" on that program (`(b == Tensor.arange(n)).where(x,
  0).sum()`); tolk raised `Invalid_argument`.
- **Reason:** (b): rune's lowering of a contraction cut into blocks
  (`Lower_reduce`), whose sum `Rangeify.collapses` asks about selects by a
  stage read at the summed range.
- **Pinned by:** the Simplify suite: `pm_load_collapse › a sum whose index is
  read at its range adds each trip it selects`; on rune, `Lower_reduce ›
  contractions › contracting pairs in blocks give eager's integers and hold
  their bound`.

## D190. Clang of LLVM 18 compiles for x86_64 without AVX10.1 and avx512fp16

- **tinygrad:** `runtime/support/compiler_cpu.py:10` (`ClangCompiler`'s
  x86_64 arguments, `-march=<cpu>` and its features); `runtime/ops_cpu.py:68`
  (the CPU's arch, its machine and `native`).
- **tolk:** `Compiler_cpu.flags` (`fixed` and `llvm18_x86_off`), which
  `Compiler_cpu.clang` and `Compiler_cpu.registers` take, with the LLVM
  version `Compiler_cpu.llvm_major` reads from the statement
  `Compiler_cpu.version` asks for.
- **Differs:** under a Clang that states LLVM 18, an x86_64 arch's arguments
  end with `-mno-avx10.1-512 -mno-avx10.1-256 -mno-avx512fp16`, after
  `-march` and the arch's features, so they win over what `native` or the
  arch enables. A Clang host's half is an `__fp16`, which computes in
  `float` on every processor (D17), so values are unchanged: only the
  instructions converting halves differ, as on a processor without AVX-512
  FP16. Every other Clang, and every other machine,
  takes tinygrad's arguments.
- **Reason:** (b): rune's `Lower_arith › casts give eager's values` at
  `(float16, 11)` and `rune vmap › compositions › under jit` at float16
  failed on GitHub's ubuntu-latest under Ubuntu 24.04's Clang 18.1.3 with "clang
  frontend command failed with exit code 139", after "invalid feature
  combination: +avx10.1-256; will be promoted to avx10.1-512", on runners
  whose processor states AVX10.1, and passed on others. The cast kernel
  `E_21_230` loads a half and stores its double bits in one row and the half
  converted to a saturated short in another; Clang 18.1.3 and 18.1.8 crash on
  it in machine loop-invariant code motion for `sapphirerapids`,
  `graniterapids`, `graniterapids` with `-mavx10.1-256`, `x86-64-v3` with
  `-mavx10.1-256` and `x86-64-v4` with `-mavx512fp16`, and compile it for
  `x86-64-v4`, `znver4` and each of these with avx512fp16 off. AVX10.1
  implies avx512fp16, and under LLVM 18 disabling avx512fp16 or avx10.1-256
  alone leaves it on: avx10.1-512, then avx10.1-256, then avx512fp16 must go.
  Clang 17.0.6 and 19.1.7 compile the kernel for each of these processors,
  so every other Clang keeps the features.
- **Pinned by:** `Tolk.Compiler_cpu › flags › under LLVM 18, x86_64 ends by
  disabling AVX10.1 and avx512fp16:` (every row), `› LLVM's own release of
  18 is LLVM 18`, `› x86_64,native takes its own arguments alone under`
  (every row) and `› another machine under LLVM 18 takes its own arguments
  alone:`; `Tolk.Compiler_cpu › features › a loop of half casts compiles
  with AVX-512's half arithmetic on` (`x86_64,sapphirerapids`,
  `x86_64,x86-64-v4,avx512fp16`), which crashes Clang 18 without the flags;
  rune's two tests above, under Clang 18 on an x86 host with avx512fp16.

## D191. A loop bounded by a load outside any range counts its greatest trips

- **tinygrad:** `renderer/__init__.py:47-51` (`Estimates.from_uops`'s range
  multiplier), which counts a bound that reads no range as the bound
  itself.
- **tolk:** `Renderer.Estimates.of_uops`.
- **Differs:** a bound that reads a load and no range, as `j < x[0]`,
  counts at its greatest value, as one that reads a range does; tinygrad's
  count then reads the load. Every consumer evaluates a count from the run's
  variables alone, before the kernel reads memory: the search's compute
  filter, the DEBUG=2 report, and a host launch's blocks (D80). On a count
  that reads a load, tinygrad's `sym_infer` raises, and so did tolk's
  `Shape.sym_compile`.
- **Reason:** (c): a host launch takes its blocks from the count (D80),
  when it is linked, before the kernel reads memory.
- **Pinned by:** the Renderer suite: `estimates of hand-written kernels ›
  counts a range bounded by a value loaded outside any range at its greatest
  trips`; the Tolk_engine suite: `a kernel whose estimates fit no int ›
  splits and runs, its loop bounded by a value loaded outside it`; the Search
  suite: `a search measures candidates whose estimates fit no int`.

## D192. A call that accesses LOCAL memory is ordered by barriers

- **tinygrad:** `codegen/__init__.py:228-247` (`_is_local_store`,
  `add_raw_barrier` and `add_war_barrier`), whose walks of an AFTER's
  dependencies and of a loop's body do not enter call bodies.
- **tolk:** `Codegen.local_writes` and `Codegen.reads`, which
  `Codegen.add_raw_barrier` and `Codegen.add_war_barrier` take.
- **Differs:** a call whose body, or a body that body calls, stores to LOCAL
  memory writes each of its LOCAL arguments' buffers, and one whose body
  loads from LOCAL memory reads them. A read of LOCAL memory ordered after
  such a writing call waits at a workgroup barrier, and a loop that writes a
  LOCAL buffer and reads it, by a store and a load of its own or by such
  calls, ends each trip with one. A body lowered alone orders only its own
  accesses. tinygrad's kernel reads before other threads' calls have written,
  and a trip writes before other threads' calls of the trip before have read.
- **Reason:** (e): a thread that reads an element another warp's call
  writes reads it unwritten; without the barrier, one run in three on
  fudge's RTX 5000 read wrong values.
- **Pinned by:** the Codegen suite: `calls in a kernel › a kernel reads
  what a call writes to local memory after a barrier`, `› a loop whose call
  writes local memory the loop reads ends with a barrier`, `› a loop whose
  call reads local memory the loop writes ends with a barrier`, `› threads
  read what other warps' calls wrote to local memory, on CUDA` and `› calls
  read what other warps wrote to local memory, on CUDA` (both skipped
  without a CUDA GPU).

## D193. A string or bytes argument enters a key as its digest

- **tinygrad:** `uop/ops.py:253-255` (`key`, the SHA-256 of
  `str((op, dtype, arg))`, which quotes a string or bytes argument whole,
  followed by the keys of the sources) and `:1277` (`to_elf`, whose profile
  key is the program's `key`).
- **tolk:** `Ops.key_of_arg`, which `Ops.key` and `Ops.key_with` read, and
  so `Device.Tiny_elf.of_program`.
- **Differs:** a `String` or `Bytes` argument enters the digested text as
  `String(h)` or `Bytes(h)`, `h` the BLAKE2b-256 digest of its contents in
  hexadecimal, where tinygrad writes its literal. Keys stay equal for equal
  graphs and different for different ones; no key is compared with one
  tinygrad computed (D12).
- **Reason:** (d): a compiled program's profile key digests its source and
  binary, megabytes for rune's special-function gradients. Writing the
  binary's literal, a `\xNN` for each byte past ASCII, took 22 M of the 28 M
  words that keying the program took in rune's
  `compile/special/betaincinv-grad-f64` row, whose cold compile now
  allocates 357 M words against 382 M, counted at the process's exit. Every
  rune compile row allocates 0.6% to 23% fewer words. A literal written
  without the formatter would still be several times the binary's size,
  built once to be digested and held by the node after. The row's baseline on
  fudge is pending a quiet run, and the compile bench records time alone.
- **Pinned by:** the Ops suite: `key › tells apart sources and binaries by
  their text` and `key_with › is the key of a source and a binary of the
  texts given`; the Device suite: `Tiny_elf.of_program › keys equal programs
  alike and programs that differ apart`; rune's
  `compile/special/betaincinv-grad-f64` row, once recorded on fudge.

## D194. A running operation scans chunks of 16 at every level

- **tinygrad:** `mixin/op.py:781-791` (`_split_cumalu`: past 512 elements,
  chunks of 256, the first padded before, scanned by windows, then the
  chunks' last elements scanned by one window each).
- **tolk:** `Shape.split_cumalu` and `Shape.cumalu`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same.
- **Differs:** past 16 elements an axis runs in chunks of 16 from its first
  element, and the chunks' last elements run through the same scan again,
  chunked in turn past 16 of them. The first stage runs over the axis's full
  chunks alone, whose loads are those of whole rows, and a tail of fewer
  than 16 elements runs as a chunk of its own, padded after, joined to them
  by the operation with its identity; the later stages, 16 times smaller,
  pad their last chunk after. An element's chunk at each stage, and so the
  order of its operations, depends on its index alone, but for operations
  with the identity, which change only the sign of a zero sum or a float
  maximum's result at a NaN (`Op.Max` gives it in its operands' order): the
  running operation of a prefix of the axis is the prefix of the axis's, bit
  for bit, as nx.mli promises of `Nx.cumsum`, whose sum of `-0` terms rune
  makes `+0`; rune computes extremes over integer keys. Padded before, an
  element's chunk moves with the axis's length, and a float sum of a prefix
  rounds otherwise. Padded after at the first stage, a chunk's loads are
  gated each by its own element, and the host kernel loads them one at a
  time: float64 over 100 × 50500 took twice as long in that stage. Each
  element reduces a window of at most 16 elements and combines once per
  stage, a tail's elements once more in the first, so the work is linear in
  the axis: estimated (`Realize.estimate_uop`) at 6 operations an element
  over 65536 float32 elements against tinygrad's 285, whose window is 256
  elements, or the whole axis below 513. An axis of `n` elements takes about
  `log₁₆ n` stages. Values are a reassociation of the same terms: integers,
  booleans and extremes are unchanged, and a float sum or product of `n`
  terms keeps rune's bound of `γ(n - 1)`.
- **Reason:** (d): compiled running sums, `Nx.cumsum` under `Rune.jit`, as
  norn's nested sampling's error estimate takes, and rune's lowering of an
  extreme's position and of segments' starts (`Lower_reduce`). The aligned
  chunks against chunks padded before, with one window up to 32 elements
  (fudge, five interleaved rounds, one process under the timing lock per
  run, load 15 to 43): float64 over 100 × 50500 5.4-13.5 ms against
  5.4-16.9 ms, 3326 words against 3322; float32 over 4096 × 32 80-127 µs
  against 203-466 µs, and 2281 words against 1268, its three kernels
  against one, the cost of the windows of 16 the alignment needs; float32
  over 100 × 400 44-59 µs against 44-47 µs, 2804 words against 2281, a
  stage more for its 25 chunks. The chunks padded before against
  tinygrad's (three interleaved rounds, load 60 to 92): float64 over
  100 × 50500 8.5-35.6 ms against 44.8-64.9 ms; float32 over 100 × 400
  55-66 µs against 2.75-5.00 ms; the nested sampling estimate over
  8 × 1024 100-200 µs against 305-465 µs. Each row allocates the same
  words. On fudge's RTX 5000 Ada, float32 over 100 × 50500 206-220
  µs against 736-770 µs, and over 8 × 1024, 11.5 µs against 13.0-13.4 µs
  (three rounds, one that shared the GPU left out); over 100 × 400, three
  stages against one kernel, 10.7 µs against 19.6-20.2 µs (two rounds
  before, one after). The block of 16 was chosen among 8, 16, 32, 64 and
  256 by an interleaved sweep over running sums of `2x`, three rounds of one
  process per block size: over float64 100 × 50500, 8, 16 and 32 overlapped
  at 6.1-13.4 ms against 11.1-19.1 at 64 and 26.3-38.4 at 256; over float32
  64 × 64, 8 and 16 took 3-5 µs against 20-25 µs at 32 and above, which scan
  it in one window; 16 takes one stage fewer than 8.
- **Pinned by:** rune's `Lower_reduce › work › a running sum's work per term
  at most doubles from 32 to 65536 terms` and `› scans › running scans along
  long axes give eager's values`; `rune jit › classes › a running
  operation's prefix is the prefix of its result`, on the host and, in
  `Rune.jit on a GPU`, on the GPU; the Rangeify suite's `long_cumsum`,
  `windowed_chain` and `windowed_scaled_long` rows of `kernel_counts.golden`
  and their goldens, and the Codegen and Heuristic suites' `cumsum` kernels,
  from the equally patched tinygrad; on fudge, rune's bench rows
  `Jit/cumsum-float64-100x50500`, `Jit/cumsum-scaled-float32-100x400`,
  `Jit/cumsum-neg-log-uniform-8x1024`, `Cuda/cumsum-float32-100x400`,
  `Cuda/cumsum-float32-8x1024` and `Cuda/cumsum-float32-100x50500`.

## D195. Clang reinterprets a vector's bits by a vector cast

- **tinygrad:** `renderer/cstyle.py:43` (`base_rewrite`'s `BITCAST`,
  `__builtin_bit_cast(T, (S)(x))` for every value).
- **tolk:** `Cstyle.clang_vectors`'s bitcast rule and `Cstyle.byte_mask`;
  `test/gen/tinygrad.patch`, which gives tinygrad's `clang_vectors` the same
  (`_vector_bitcast`, `_byte_mask`) before the goldens are recorded.
- **Differs:** on Clang, a bitcast between vectors, the host's lanes of
  D141, is the C cast `((T)((S)(x)))`, which reinterprets the bits of a
  vector of one size as another's, and a mask of one-byte lanes is read as
  signed chars by that cast. A bitcast with a scalar side stays
  `__builtin_bit_cast`. Each lane holds the same bits either way: Clang
  writes `__builtin_bit_cast` of a vector as a store and a load of a
  temporary, and the cast as one `bitcast` instruction. Of the 339 kernels
  of rune's compile bench, the 61 with a vector bitcast compile to the same
  object but two, whose vectors of 3 lanes Clang pads to 4 in memory: the
  temporary copies the padding lane, and their instructions differ in order
  and by up to 7 of 4520.
- **Reason:** (d): rune's compile bench row `special/betaincinv-grad-f64`.
  Its one kernel, 1.0 MB of C with 2495 vector bitcasts, spent 5.2 of
  Clang's 6.2 s of CPU in scalar replacement of aggregates, removing the
  temporaries, and 0.08 s with casts. Clang took 6.8-11.9 s against
  2.7-4.8 s for the same object (fudge, loaded, three interleaved pairs).
- **Pinned by:** the Cstyle suite's `sources › by default ›
  clang_vector_bitcast` and `› clang_vector_byte_mask`, from the patched
  tinygrad, and every host golden with a vector bitcast; rune's compile
  bench row `special/betaincinv-grad-f64`, whose fudge baseline the
  scheduled re-record records.

## D196. A schedule's host kernels compile in one run of Clang per domain

- **tinygrad:** `engine/realize.py:238` (`lower_and_compile`, which makes
  each kernel's program, `to_program`, apart on its pool's workers);
  `codegen/__init__.py:461` (`do_compile`) and `device.py:333`
  (`Compiler.compile_cached`, one source at a time);
  `runtime/support/compiler_cpu.py:18` (`ClangCompiler.compile`, a `clang`
  process per source, from standard input).
- **tolk:** `Codegen.to_programs`, which `Realize.lower_and_compile` calls
  for every kernel; `Renderer.Compiler.batch` and
  `Renderer.Compiler.compile_all`; `Compiler_cpu.clang`'s `compile`, with
  `Compiler_cpu.cc1` and `Compiler_cpu.compile_files`.
- **Differs:** up to `PARALLEL` workers on `Worker.map` each take the
  kernels a schedule lacks, those `Codegen.read` does not find, one at a
  time, to lower and render until none is left, then compile their sources,
  each compiler's in one call (`Renderer.Compiler.compile_all`), which Clang
  serves with one run of its compiler proper (`clang -cc1`, the command
  Clang states for the options with `-###`), each source the file `-` of a
  directory of its own, after a line marker that names it `<stdin>`, the
  files named in a response file. Each object is the one Clang's driver
  makes of the source alone from standard input, so the cache keys, tables
  and entries are unchanged. When a kernel asks for a beam search, each
  kernel compiles alone, in order. Clang's driver runs the compiler proper
  in its own process for one source, and starts a process for each of
  several (`clang -c a.c b.c` costs as many starts as `clang -c` per
  source), so only the compiler proper's own command saves the starts.
- **Reason:** (d): rune's compile bench rows `scan/checks-48`,
  `scan/tridiagonal-100`, `devices/partials-16` and `model/schools-start`,
  which compile 111, 52, 60 and 39 host kernels, most under 8 KB. On fudge,
  Clang took 30 ms of wall for a kernel under 2 KB and 41 ms for one of 2-8
  KB, and 25-33 ms for an empty source (111 runs, loaded): the start
  dominates. checks-48's 111 kernels took 2.4-3.5 s of CPU as 111 runs and
  0.54-0.65 s as one run of the compiler proper (three interleaved pairs).
- **Pinned by:** the Compiler_cpu suite's `several sources › each object is
  the one its source compiles to alone` and `› a rejected source among
  others raises its own diagnostics`; the Renderer suite's `Compiler ›
  compile_all compiles the sources the table lacks in one call, each once`,
  `› compile_all calls nothing when the table holds every source`, `›
  compile_all raises for the first source the toolchain rejects` and `›
  compile_cached behaves as a table of binaries`; the Codegen suite's
  `programs are kept › to_programs makes each kernel's program, compiled in
  at most a list per domain` and `› to_programs on one domain compiles every
  source in one call`; rune's compile bench rows above.

## D197. A program key orders commutative index operands by the kernel's ranges

- **tinygrad:** `uop/symbolic.py:235-237` (commutative flipping, which
  orders the two operands of a commutative index operation by `tuplize`,
  a range's number among it); `schedule/rangeify.py:318-329`
  (`pm_renumber_range` in `split_store`, which renumbers a kernel's ranges
  from 0 after the symbolic rules ran over the whole graph, "so that kernel
  deduping works"); `codegen/__init__.py:513-518` (`to_program_key`, which
  keys a program by its kernel's `ast.key`).
- **tolk:** `Codegen.params_on`, which `Codegen.program_key`,
  `Codegen.to_program` and `Codegen.read` read.
- **Differs:** codegen orders the operands of each commutative index
  operation of a kernel by the kernel's own range numbers, as the symbolic
  rules order them, then keys that kernel and makes its program from it.
  Two kernels whose ranges the whole graph numbered in different orders,
  and which so differ only in that order, are one kernel there: one key,
  one source, one program, which computes either. The key cannot map two
  sources to one binary: it reads the very kernel the source is rendered
  from, so kernels of one key render one source, and the disk cache keeps a
  binary by its source. The schedule's kernel itself is tinygrad's.
- **Reason:** (b): rune's compiled programs over several devices
  (`Pieces.split`) put every device's kernels in one graph, where one device's
  scan of its tile numbered its ranges in the other order: a cumulative sum
  over an axis cut across 2, 4, 8 and 16 memory devices made 5, 7, 8 and 7
  programs, and makes 5, 7, 7 and 7, against rune's promise that they do not
  grow with the devices (D170).
- **Pinned by:** the Codegen suite: `programs are kept › kernels share a
  key where they render one source, as a sum of ranges in either order
  does`, which compares the rendered sources and keys of both orders and of
  another index; rune's
  `Rune.jit over several devices › maps over a cut axis › partials give
  eager's bits, their programs do not grow`.

## D198. A store that reads only its own element is not staged

- **tinygrad:** `schedule/indexing.py:31-34` (`realize_store_after_src`,
  which stores the value of every store whose value reaches its
  destination's buffer: "you don't usually have to do this for assign unless
  there's a WAR hazard like TestAssign.test_assign_double_diamond_reduce").
- **tolk:** `Indexing.reads_own_element`, `Indexing.keeps_elements` and
  `Indexing.readers`, which `Indexing.realize_store_after_src` asks.
- **Differs:** a store into a view of its storage that expands nothing,
  whose value reaches that storage only through the destination itself and
  elementwise operations, a bitcast only between types of one size, keeps
  its value unstaged when no other store or call reaches that storage but
  through a later state (`After`). Each element is then loaded by the value
  that stores it, a data dependency of the store, so the kernel loads it
  before it overwrites it, with no order relied on; no other kernel reads
  the old state, so there is no write after read to break. Through an
  expand, two elements of the destination are one of the storage, and one
  would load what another stored. tinygrad stages the value, and
  `remove_bufferize` inlines the stage back unless it stays stored for its
  own reasons (more than three buffers, a reduction reading one, D163's
  copies): then a kernel computes the value into a stage and another copies
  it into the destination. A value read at another element, through a
  movement or a reduction, or read by another store, is staged as tinygrad
  stages it.
- **Reason:** (d). rune's carries, which a loop's trip updates in place:
  NUTS on eight schools (`norn` bench `nuts/schools/4`), whose doubling
  cascade's trip ran 17 kernels and runs 10, its main body's 189 and 182,
  and whose cold compile makes 145 programs against 147, with the same bits
  (fudge, counted). The bench row's wall-clock baseline is pending the
  scheduled re-record.
- **Pinned by:** the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › stores that read their destination` (`a value that
  reads its destination at the element written is one kernel`, `› a value
  of five storages that reads its destination at the element written is one
  kernel`, `› a value that reads a slice of its destination at the element
  written is one kernel`, `› a value that reads its destination through a
  bitcast of one size is one kernel`, `› a value that reads its destination
  at another element is staged`, `› a value that reduces its destination is
  staged`, `› a value stored into a broadcast of its destination is staged`
  and `› a value whose destination another store reads is staged`), each
  against the values its tensors write; the norn bench's `nuts/schools/4`.

## D199. Independent calls of one domain run as one call where no lifetime grows past one element

- **tinygrad:** `schedule/__init__.py:71-74` (`create_schedule`, one call
  per kernel of the kernel graph, in the linear as scheduled) and
  `schedule/rangeify.py:321-336` (`split_store`, one kernel per store or
  end).
- **tolk:** `Schedule.merge_calls`, `Schedule.merge_linear`,
  `Schedule.member`, `Schedule.merged` and `Schedule.storage_taken`, which
  `Schedule.create_linear_with_vars` runs after `pm_copy_from_store`, after
  the schedule cache.
- **Differs:** in a schedule, two kernel calls run as one call when each is
  a kernel of the default kind that reduces nothing and ends its stores over
  its ranges, all their storage lies on one device, their domains are the
  same ranges by position (constant sizes and kinds), neither reads or
  writes storage the other writes (storage compared by `Ops.storage_base`,
  so two views of one buffer conflict), the later moves up to the group's
  first call only past calls it does not conflict with, and the merged call
  passes at most `MAX_KERNEL_BUFFERS` buffers when that is set. A call moves
  earlier only where every storage whose life that starts earlier is storage
  the schedule is given, holds one element, or lives for the whole loop
  around both: in a loop's body every buffer already lives for the whole
  loop (a loop counts as one entry, D41, and `Tolk_engine` makes storage per
  run, D173), so calls join freely; at the top level, only calls of one element
  join, since the order of calls of many elements is the schedule's, chosen
  for the storage they make (RFC 0005's bound). Arguments that are one node
  are passed once, storage first, so loads and arithmetic the members share
  are one node of the merged kernel. Each call is weighed once, in order,
  against the open groups of its device and domain, the earliest that
  starts past the last call it conflicts with and past the first use of any
  storage whose life may not start earlier: the merge is linear in the
  calls. The decision reads devices only to compare them, after the cache,
  so the schedule kept stays free of device names.

  The merged kernel takes one set of optimisations for the union of its
  members. On the host, two independent `[1024; 1024]` float32 carries in a
  scan lose one member's 16-lane upcast, the `sqrt` member's `E_65536_16`
  becoming the union's `E_131072_8`: measured at 2-3% a call when the merge
  was designed, and 36.7-37.1 ms against 36.8-37.3 ms on this tree (best of
  5 × 20 calls, two interleaved rounds); the same program on CUDA gained
  10-25%. A mode that merges by target is not taken; the heuristic's choice
  for a union is the host lane budget's question.
- **Reason:** (d). On CUDA a launch costs about 1.3-2 µs, so a trip's
  kernels are its cost: NUTS on eight schools, its density written by hand,
  4 chains, on fudge's RTX 5000, launches 498 kernels a transition against
  1,305 and takes 1.36-1.88 ms a warm step against 2.36-2.46 ms, compiling
  131 programs against 155, with the same bits (three interleaved rounds,
  each run under the timing and GPU locks, load 4 to 12; with D198 and
  D163's rule across kernels). On the host a launch costs about 20 ns: the
  model-language program's schedule holds 162 entries against 368 (top
  level 78 against 134, main body 69 against 189, cascade 6 against 17)
  and compiles 114 programs against 147, its cold first call 3.8-4.0 CPU-s
  either way since Clang runs once per domain (D196), its warm step
  unchanged; ymir's aperture gradient compiles cold in 13.7-15.1 CPU-s
  against 22.1-28.3, 180 programs against 196, with the same bits. Pinned
  by norn's `nuts/schools/4` on the host; a CUDA row of the hand-written
  density is pending.
- **Pinned by:** the Schedule suite (`test/schedule/schedule`):
  `create_linear_with_vars › merged calls` (every test: independent calls in
  a loop and of one element at the top level run as one, calls of many
  elements at the top level, a reader of another's output, an overwriter of
  what another reads, two runs of one buffer, a reduction and two devices
  run apart, a call moves up only past calls it does not conflict with, the
  buffer limit, shared arguments and work, pairs of calls, the merge's
  linear work, the lifetime of storage of many elements, and the laws that
  outputs run as one write what their tensors compute, bit for bit, and that
  calls of a loop run as one, reading and overwriting one another's slots
  and the trip, write what their tensors compute); the Tolk_engine suite
  (`test/engine/tolk_engine`): `link and run › a scan's merged calls carry
  in place as its calls apart do`, and the same on each GPU present.

## D200. A tensor core's row upcast keeps a warp within a run of rows

- **tinygrad:** `codegen/opt/heuristic.py:29-43` (`hand_coded_optimizations`,
  whose tensor core upcasts the core's M range by the first of 5, 4, 3 and
  2 that divides it).
- **tolk:** `Heuristic.tensor_cores`, through `Heuristic.within_run` and
  `Heuristic.owner_run`; `test/gen/tinygrad.patch`, which gives tinygrad the
  same.
- **Differs:** where the operands of the core's product read its M range
  only as its quotient by `d`, as blocks of `d` rows read their expert's
  matrix at `owner[r / d]`, and the first candidate dividing M would make a
  warp of tiles cross a run of `d` rows, M takes the first candidate
  dividing it whose warp, the core's tile times the upcast, divides `d`.
  Where none does, the operands read M otherwise, or the core padded M,
  tinygrad's candidate stands, and where tinygrad takes none, none. The
  rule takes M alone: rows routed onto the core's N cross runs as before.
- **Reason:** (d): a warp whose rows cross runs loads a tile of each run's
  matrix. gpt-oss-20b's experts layer of MXFP4 matrices at bfloat16 on
  fudge's RTX 5000 Ada, its rows in blocks of `c` of one expert (tile 16,
  three interleaved rounds, ranges apart): at 2048 tokens and `c = 64` an
  upcast of 5, warps of 80 rows over two blocks, takes 19.25-19.73 ms a layer
  against 10.62-10.85 ms at 4; at 512 tokens and `c = 32` 6.60-6.64 ms at 5
  against 4.44-4.52 ms at 2; at 128 tokens and `c = 64` 4.96-5.01 ms against
  3.21-3.24 ms. At `c = 16` no candidate keeps a warp of 16-row tiles within
  a block, so tinygrad's candidate stands: there its upcasts of 2 and 4 over
  two to four blocks were 1.2 to 1.4 times faster than none; an upcast of 5
  over blocks of 16 is not measured. rune lowers no routed product in blocks past 16 rows yet; its
  blocked lowering of 64 rows is the consumer, and its rows the pins.
- **Pinned by:** the Heuristic suite: the `routed_run_16`, `routed_run_32`,
  `routed_run_64` and `routed_run_128` cases on each renderer, from the
  equally patched tinygrad, and `a core's warp holds rows of one run where an
  upcast of the core's rows keeps it in one`; on fudge, once rune lowers
  routed products in blocks of 64, gpt-oss-20b's experts rows at 128, 512
  and 2048 tokens.

## D201. A search's samples start from a clean, cold cache

- **tinygrad:** `codegen/opt/search.py:145` (`_time_program` is asked to
  clear the L2 only where the device has `invalidate_caches`, NV's) and
  `engine/realize.py:292-299` (`time_call`, which invalidates NV's caches
  before each sample, and elsewhere realizes a tensor of 4 MB, which the
  search never asks for): CUDA, AMD, Metal and the host time warm.
- **tolk:** `Tolk_engine.scrub` (`device_scrub`, `scrub_kernel`,
  `cache_bytes`) and `Tolk_engine.time` (`read_scrub`); rune's
  `Compile.search`; `Rig.cache_bytes`.
- **Differs:** a search makes, beside the slots it times on, a scrub: for
  each device that runs the kernel and whose driver states its last-level
  cache's bytes, a buffer of twice those bytes, written once, and a kernel
  that reads it. Before each sample, outside its time, the kernel reads it
  and the time waits for the read. Where the scrub does not fit in memory,
  rune's search times warm. A read leaves clean lines, so the timed kernel finds none of its
  buffers cached and no line to write back. On every driver that states the
  size, CUDA, NV, AMD through amdgpu and the host; a Metal device and an
  AMD GPU over PCI are timed warm.
- **Reason:** (c): `Search.beam_search` promises each sample from cold
  caches where the device can, and rig gives no flush on CUDA, its default
  path on fudge, while AMD's L2 is invalidated on every submission and its
  memory-side cache is not; a read reaches every driver. Measured on fudge's
  RTX 5000 Ada (64 MiB of L2, libcuda's figure), GPU time, three
  interleaved rounds:
  - warm, a search times a state a model's run rarely gives a kernel whose
    operands other kernels evicted. On `matmul_half_1024`, six rounds at
    width 4, the warm search chose one program each round, 43.1-44.0 us warm
    and 65.7-66.5 cold; the cold search chose one of 44.2-44.4 and 44.9-45.0
    us in three rounds, 50.2 and 51.0 in one, and in two, programs no faster
    cold (67.6) or slower (84.0), where its greedy beam kept a worse first
    level. Before this change's floor, cold searches picked programs
    1.09-1.85x faster from memory on `matmul_half_1024` and 1.33-1.41x on a
    prefill's key projection;
  - a write of the cache's size leaves the cache dirty, and the beam's
    first ranking follows the write-back: a search timed after a fill picks
    a `matmul_half_1024` program 1.48-1.51x slower from a clean cache and
    1.13-1.20x slower even in its own dirty state, 3 of 3 rounds;
  - a fill of the cache's own 64 MiB left 4 of 19 rows warm in every round,
    by where their buffers lay; 128 and 256 MiB evicted every row;
  - the read costs about 0.25 ms a sample on fudge, 0.07-0.26 s a search.
- **Pinned by:** the Tolk_engine suite's `timing › a time with a scrub
  reads it before the run` (the host, a memory device, and CUDA, NV and AMD
  where present); rig's tests of `Rig.cache_bytes`. Whether the read leaves
  the cache cold is the timing above: on a GPU other processes share, a
  warm sample is not reliably warm, so no test times it. A bench row of a searched program would flap with the
  search's pick, as the rounds above do, so none holds it.

## D220. A loop's bound may be loaded, and no axis it reads runs as lanes or launches

- **tinygrad:** `uop/spec.py:103-104` and `205-228` (`spec_shared`, an
  AFTER of a scalar ALU, and `spec_program`, neither of a LOAD), which
  `codegen/__init__.py:370-375` checks under `SPEC`, 1 by default; `codegen/late/linearizer.py:83-86`
  (`pm_add_control_flow`, which wraps a nested loop's bound in an AFTER of
  the range or loop end it follows); `codegen/opt/postrange.py:52-60`
  (`convert_loop_to_global`), `84-85` (`upcastable_dims`) and `87-125`
  (`apply_opt`).
- **tolk:** `Spec.program`; `Postrange.Scheduler.read_by_bounds`,
  `upcastable_dims`, `apply_opt` and `convert_loop_to_global`;
  `Ops.known_at_launch`, which `Codegen.split_blocks` (D80) and
  `Renderer.Estimates.of_uops` (D191) read; `Heuristic.streams`; `test/gen/tinygrad.patch`, which gives tinygrad the
  same rules.
- **Differs:** a loop bounded by a load, as `j < x[i]`, compiles: a
  program's Spec accepts an AFTER of a scalar LOAD whose dependencies are
  ranges and loop ends, the linearizer's edges, where tinygrad refuses it;
  a graph's Spec still refuses one, and an AFTER of a load and a store
  stays refused, since neither orders the load. An
  axis that a range's bound reads, through a load or not (`j < i + 1`), is
  not upcastable, and an upcast of it is refused, where tinygrad makes the
  bound a vector that C compilers reject: a loop has one count of trips
  per thread. A group and a tensor core are refused on a kernel where a
  range's bound reads an axis, since a group's barriers need every thread
  of the workgroup to run the same trips, and the heuristic streams no
  contraction there. A weak range whose bound reads a range or memory is
  neither made global nor split into host blocks, where tinygrad makes it
  global: a launch computes its counts before the kernel reads memory. The
  estimates (D191) take the same test of a bound.
- **Reason:** (c): tolk's own interface builds such a loop, since
  `Ops.range` takes any bound and `Shape.shrink` any size, a loaded one
  included, and a graph that interface accepts must compute, where
  without this one with a loaded or row-dependent bound fails at Spec, at
  the C compiler or at launch.
- **Pinned by:** the Tolk_engine suite: `loops bounded by a loaded value`,
  on the host and each GPU present (CUDA and NV on fudge): serial and weak
  loops and sums bounded by each row's or the first row's clamped or bare
  count, or by their row, unoptimized, optimized and upcast (refused where
  the bound reads the row), a weak loop of no row, a loaded bound after a
  sibling loop, and a sum of products along the rows, which the heuristic
  would stream. The Ops suite: `ranges › a launch knows a bound of
  constants and variables, ...`. The Postrange suite: `loops bounded by an
  axis`, and `axes.golden`'s `padto_reduction_bound` from the patched tinygrad. The
  Spec suite: `verdicts.golden`'s `an int32 load after a range`, `a range
  bounded by a load after a range` and `a load after a store`.

## D230. A GPU kernel decodes a value R rows read at most ceil(R / 16) times

- **tinygrad:** `codegen/opt/heuristic.py:9-44` (the tensor cores) and
  `:111-137` (the upcasts of output axes by 3 or 4); it streams nothing
  (D89), and states no bound on how often a kernel computes a value its
  outputs share.
- **tolk:** `Heuristic.stream`'s upcast of the reuse ranges and
  `Heuristic.hand_coded_with`'s late stream of a decoded operand;
  `test/gen/tinygrad.patch`, which gives tinygrad the same.
- **Differs:** rangeify inlines a decoded value that rows read at one index
  as one copy (D163), so the codegen owes the sharing that count assumes:
  on a GPU, a decoded operand that reads every output axis but one, along
  which its `R` outputs read it at one index, is decoded at most
  `ceil(R / 16)` times a value where the kernel streams or takes a tensor
  core whose upcast of those rows applies, times the lanes its core's
  fragment holds each value in (two on RDNA3, whose `B` lanes carry an `M`
  bit). A core whose floor (D175) refuses the upcast decodes it once a tile
  of its rows, `ceil(R / 8)` times on Metal's core of 8, and an operand that
  several output axes reuse takes the largest divisor of their product up
  to 16, as before. The streamed layout upcasts such an axis of `f` outputs into
  `ceil(f / 16)` passes of as many outputs, padding the axis where they do
  not divide it, where it took the largest divisor of `f` up to 16: 17 rows
  take 2 passes of 9, padded to 18, where they took 17 of 1. Past a reuse
  of 32, where no core applies, a decoded operand streams, where it took
  the other rules, which share only their upcast of 4 rows: 64 float32 rows
  decoded each value 16 times, and take 4. The host is exempt: its tiles
  share 3 or 4 lanes (D163's Reason).
- **Reason:** (d): MXFP4 products of 16384 by 16384 on fudge's RTX 5000 Ada
  (kaun's `Quant/cuda/mxfp4-product-of-16384-*`, three interleaved rounds
  under the timing lock, GPU shared with other jobs): 17 bfloat16 rows
  4.58-11.91 ms to 1.07-3.28 ms; 64 float32 rows, streamed, 14.8-20.0 ms to
  4.98-6.68 ms (12.5-12.9 in quieter rounds before), past the stored
  decode's 9.6; 12, 16, 32, 64, 128 and 256 bfloat16 rows unchanged within
  noise. Larger float32 counts are not benched. nonnormal's R9700 rows are
  pending, among them 64 float32 rows, where a model of its decode cost
  puts 4 decodes a value at about 43 ms against 31 stored.
- **Pinned by:** the Heuristic suite's `a GPU kernel decodes a value R rows
  read at one index at most ceil(R / 16) times its core's copies`, over
  decoded products of 1 to 512 rows of bfloat16, half and float32 on the
  CUDA, RDNA3, RDNA4, CDNA and Metal renderers, from the equally patched
  tinygrad, and its `a decoded product's optimisations keep its writes`
  (17 and 19 rows padded, 64 float32 rows streamed, evaluated); kaun's
  `Quant/cuda/mxfp4-product-of-16384-17-bfloat16-tokens` and
  `-64-float32-tokens`.

## D235. A tensor core's lanes are a warp of its threads

- **tinygrad:** `codegen/gpudims.py:44-97` (`group_gpudims`), which lays a
  kernel's warp and local ranges on the thread axes whatever its tensor cores
  need, and `codegen/simplify.py:68-77` (`mark_range_mod` and
  `pm_split_ranges`), which split a local range, never a warp (`:70`), by
  its divisions, the most significant part taking the first thread axis;
  `test/gen/tinygrad.patch`'s `group_gpudims` raises as tolk does.
- **tolk:** `Gpudims.group_gpudims`, through `check_lanes`.
- **Differs:** a kernel holding a tensor core of more than one thread raises
  `Invalid_argument` unless its first thread axis is a warp range of that many
  threads. A product's fragments are laid out by the hardware's lane, the
  thread's index within its warp, so the kernel's own lane index must be it,
  which only a warp range on the first axis gives. tinygrad's optimiser makes
  that range; a kernel written by hand may number its lanes with a local
  range instead. tinygrad then computes another product: a local range of 128
  threads read as `t // 32` and `t % 32` splits into those two parts, and the
  warp's index takes `threadIdx.x`, so the hardware's lane is not `t % 32` and
  each lane reads another lane's elements.
- **Reason:** (e).
- **Pinned by:** `Tolk.Gpudims › group_gpudims › a tensor core's lanes › are
  refused from other ranges` and `› are a warp of its threads on the first
  thread axis`; on CUDA, `Tolk.Codegen › tensor cores written by hand › a
  warp per product, lanes a warp range, gives each warp's exact product`,
  whose four warps each compute their own product.

## D236. A bitcast that changes a vector's lane count stays whole

- **tinygrad:** `codegen/__init__.py:125-134` (`do_stack_wmma`), which
  replaces a tensor core's operand by the stack of its lanes, and
  `uop/movement.py:16-18` (`mop_cleanup`), which folds that stack back to the
  operand, so that a bitcast operand that changes the lane count, such as
  four words read as eight bfloat16, rewrites forever ("infinite loop in
  graph_rewrite"); `renderer/cstyle.py:451` (CUDA's `string_rewrite`), which
  renders a bitcast with its scalar types, so that a bitcast of a vector does
  not compile ("no suitable conversion function from uint4 to uint");
  `test/gen/tinygrad.patch` gives tinygrad tolk's rules.
- **tolk:** `Codegen.do_stack_wmma`, which keeps such a bitcast whole;
  `Cstyle`'s base rewrite renders a bitcast with each side's vector type on
  CUDA, which has no rule of its own.
- **Differs:** a tensor core takes such a bitcast as the vector it is, and
  CUDA renders it `tg_bitcast<nv_bfloat168>((uint4)(...))`. A scalar bitcast
  renders as before. tinygrad does not terminate on these kernels; tolk
  computes the right product.
- **Reason:** (e).
- **Pinned by:** `Tolk.Codegen › tensor cores written by hand › a tensor
  core's operand bitcast from words lowers for CUDA`, which needs no GPU; on
  CUDA, `› a warp per product, lanes a warp range, gives each warp's exact
  product`, whose rows with words read each fragment as words bitcast to
  bfloat16, and `› a bitcast of words to bfloat16 lanes takes each word's low
  half first, on CUDA`, and on the host.

## D237. Ends of one range that reach each other are not merged

- **tinygrad:** `codegen/__init__.py:163-181` (`merge_reduce_ends`), which
  merges the ends that close the same ranges in one context. When one end
  reads what another stored, as a copy of a tile reads, through the scores,
  what the copy before it staged, the merged end contains the end it
  replaces, and the substitution stalls ("graph_rewrite stalled: unresolved
  rewrite dependencies"). Ends over upcast ranges alone close no range once
  they are expanded, so any two such copies of one context meet this;
  `test/gen/tinygrad.patch` gives tinygrad tolk's rule.
- **tolk:** `Codegen.merge_reduce_ends`.
- **Differs:** ends of one context that reach each other go apart, each a
  loop of its own: the later one takes cloned ranges, as an end of another
  context does, and every other end that closes one of those ranges, along
  other ranges too, stands aside while its ranges are renamed, so the loop
  it reads runs once. Ends that do not reach each other merge as before.
  tinygrad does not terminate on these kernels; tolk computes the right
  values.
- **Reason:** (e).
- **Pinned by:** `Tolk.Codegen › effects run as often as written › a copy
  reads the copy ended before it`, over upcast and loop ranges, apart and
  shared, and `› a copy that reads an end of its range along another range
  runs that end once`, whose second end takes cloned ranges as D137 gives
  them, which tinygrad does not (its linearizer then refuses the kernel).

## D202. A join keeps its dependencies, and holds the state under it

- **tinygrad:** `uop/symbolic.py:333-336` (`symbolic`'s ordering rule):
  an `AFTER` keeps among its dependencies only effects, ranges, ends, back
  edges, linears and stages, and replaces any other by its sources, an
  `AFTER` included. `x.after(y')`, where `y'` is `y` once assigned, becomes
  `x` after `y`'s kernel, which `create_schedule` (`schedule/__init__.py:
  31-78`) reads as a state of `x` that `y`'s kernel writes. The rule came
  with `AFTER` itself, for a kernel's register initialisation (`b8a9cce78`,
  `d1e2c393f`), and no commit states a reason for the graph of kernels.
- **tolk:** `Shape.symbolic`'s ordering rule; `Ops.is_join`;
  `Schedule.value_state`, `Schedule.swap` and the waits of
  `Schedule.schedule`; `Rangeify.unjoin` and `Rangeify.handle_after`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same rules.
- **Differs:** an `AFTER` of global storage stays among an `AFTER`'s
  dependencies. An `AFTER` whose dependencies are all `AFTER`s is a join:
  it names no state of its own. A source read through joins reads the
  state under them, and its reader waits for that state's writers and for
  what each join waits for; a write over a join is a write over that
  state; the copy of a read through a join goes under the join, which keeps
  its waits; a kernel's parameter for storage read through a join is the
  state's, its call's argument waiting for the join's dependencies. Afters
  of registers and local memory still flatten, so no kernel's source
  changes: the goldens of every module regenerate unchanged.
- **Reason:** (e). Flattened, a reader of `x` once assigned, joined after
  `y`'s assign, did not wait for `x`'s assign and read `x` before it; a
  reader of `x` as it was, joined after a kernel that reads `x` once
  assigned, was not copied and read `x` after the assign. Both gave wrong
  values, and a kernel storing into `x` a value read through a join of `x`
  raised, as did one reading two joins over one state.
- **Pinned by:** the `Tolk.Schedule` suite: `create_linear_with_vars ›
  joins › a reader of a join waits for the writer of the state it holds`,
  `› a reader of a join of an overwritten state reads it as it was`, `› a
  write over a join runs after the readers of the state it holds`, `› two
  joins over one state are no two writes` and `› a store reading its
  destination through a join takes it once`; and the programs
  `read_join_of_assigned` and `read_join_of_overwritten`, whose recorded
  schedules and linears and whose values (`create_linear_with_vars ›
  values`) are tinygrad's, patched.
