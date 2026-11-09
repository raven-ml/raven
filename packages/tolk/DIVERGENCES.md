# Divergences from tinygrad

tolk follows the tinygrad commit `TINYGRAD` in `test/gen/generate.py` pins. This
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

Each entry gives the tinygrad file and symbol at the pinned commit, the tolk
module and symbol (`Rangeify.remove_bufferize`), what differs, its reason, and
the test that pins it. tolk, and any other raven code, is cited by module and
symbol, tinygrad by file and symbol, never by line, since lines move with every
change. An entry whose test does not exist yet names the layer that brings it;
the entry is rejected at that layer's review if the test is still missing. An
entry goes when its reason goes. Entries are in the order of their numbers, and
a number is never reused, since entries and documents cite them: an entry merged
into another is withdrawn and names the entry that holds its rule, where each
merged part keeps its own reason and pins. Keeping only part of a file is scope,
recorded under Exclusions in `README.md`, not a divergence.

## D1. Withdrawn

A device's timeline was its signal word alone, whose values a batch's host
program took as parameters. rig numbers each device's work and orders it by the
memory it uses, and tolk encodes no queue (D146).

## D2. Withdrawn

tinygrad compiles a rule's pattern on its first try (`uop/ops.py`
`upat_deferred_compile`; `uop/upat.py` `upat_compile`) into one test per field
the pattern has and one call of the rule's function per alternative, as Python
source. tolk compiles it the same way into closures (`Ops.Upat.compile`,
`Ops.Pattern_matcher.rule_ctx`), so leaving out the source generation and
`UPAT_COMPILE` is scope: see the Exclusions of `README.md`. A pattern that
`upat_compile` refuses, a repeat over a pattern with several namings or too many
alternatives, has the namings `UPat.match` gives (`uop/ops.py` `UPat.match`) in
tolk, which `Upat.match_` promises for every pattern. A repeat that names nodes
matches no sources as `UPat.match` does, its names unbound, where `upat_compile`
requires a source (`uop/upat.py` `_get_clause`).

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk:** `Device` (the compiler half of `device.py`);
  `Tolk_engine.target` and `Tolk_engine.link`; `Ops.param_arg` (the type and
  the function) and `Ops.new_buffer`.
- **Differs:** tolk holds the compiler half: `Compiler` and the renderer and
  compiler selection of `Compiled`, returning data. The lazy `Buffer` and
  running a schedule are `tolk.engine`'s, which has no registry of devices by
  name: its caller maps each name to a device. The IR half of each `ops_*.py`
  (queues, `pm_encode`, program data), allocators, programs, drivers and profile
  events are rig's (D146). So a node of the compiler holds no runtime state:
  `ParamArg` has no `buffer`, a `BUFFER` is named by its slot for the engine to
  bind, and `UOp.buffer`, `realized`, `is_realized`, `_buffer_view`,
  `_base_buffer_is_realized`, `from_buffer` and `_frompy` (`uop/ops.py`) are the
  engine's. Whether storage is bound is the operation itself, `BUFFER` or
  `ALLOC`, so the specification's checks of `ParamArg.buffer` (`uop/spec.py`
  `spec_tensor`, `spec_kernel_graph`) hold by construction.
- **Reason:** (c).
- **Pinned by:** for `Ops`: `Tolk.Ops › storage › new_buffer takes the
  next slot without one` and `Tolk.Ops › arguments › reprs.golden`, where
  a `BUFFER` prints and interns by its slot alone; for the engine: the
  Engine suite (`test/engine/tolk_engine`): `refusals › link refuses a
  device the map does not hold`, which a registry would resolve, and
  `refusals › run refuses a parameter it binds no buffer`, which a `BUFFER`
  holding its buffer would run.

## D4. Import cycles are broken

- **tinygrad:** the imports inside functions that Python uses to hide a cycle:
  - `uop/ops.py` calls `symbolic`, `render`,
    `spec`, `upat`, `schedule.prepare`, `mixin.rand` and `renderer`;
  - `uop/spec.py` against `device`;
  - `renderer/__init__.py` imports `device.Compiler`, while `device.Compiled`
    holds renderers;
  - `codegen/opt/postrange.py` `apply_opts` against `search` and `heuristic`,
    with `search` importing `engine.realize`;
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

- **tinygrad:** `engine/worker.py` (`multiprocessing` spawn workers);
  `helpers.py` (`Context` and `ContextVar`, one value per process);
  `codegen/__init__.py` (`to_program_context`, the settings a worker
  process is started with, and `to_program_cache`, which the parent fills).
- **tolk:** `Worker.spawned` and `Worker.map`; `Setting.t` and
  `Setting.context`; `Codegen.to_program_cache` (`to_program`'s cache);
  `Compiler_metal.build`; `Compiler_worker.call`, and the C files
  `lib/runtime/support/compiler_worker.h`, `compiler_amd_worker.c` and
  `compiler_cuda_worker.c`.
- **Differs:** compilation runs on domains, not processes. `Worker.map` spawns
  its domains for the call and joins them before it returns, where tinygrad
  keeps a pool: an idle domain still takes part in every minor collection. So
  the process machinery goes with the pool: the hidden `__main__`, the copied
  environment, the recycling of a worker after 16 tasks, and
  `terminate_worker_pool`. A tinygrad worker ignores SIGINT, which a domain
  cannot, since a handler is the process's: the domains block every signal but a
  fault's instead. The workers' context (`ALLOW_DEVICE_USAGE`, `VIZ`,
  `TRACK_MATCH_STATS`) goes too: the compiler opens no device and has neither of
  the other settings. A setting holds one value per domain, where tinygrad holds
  one per process: a domain starts with the values of the domain that spawns it,
  and a `context` override is seen by its own domain only. A tinygrad worker
  process has its own settings, so a `Context` entered while it compiles (the
  construction check's `CHECK_OOB=0`) leaves the others alone; process-wide
  settings shared by compiling domains would let one domain's override and
  restore clobber another's. A spawned domain starts with its spawner's
  settings, so `to_program_context` goes too. `to_program` keeps its programs,
  and `lower_sink_to_linear` its schedules, in a table that every domain reads
  and fills under a lock (`Helpers.Memo`), and a domain that asks for one
  another is making waits for it, so each program is compiled once, as
  tinygrad's parent compiles each key once, and each body is scheduled once. The
  matchers that a tinygrad worker keeps per process (`codegen/decomp/op.py`
  `get_simplifying_rewrite_patterns`, `get_late_rewrite_patterns` and
  `codegen/decomp/transcendental.py` `get_transcendental_patterns`,
  `functools.cache`) are made once and shared by every domain
  (`Decomp_op.simplifying_patterns`, `Decomp_op.late_patterns`,
  `Transcendental.patterns`), and a rule compiles on the domain that tries it
  first, or on each that tries it at once. MTLCompiler, which a tinygrad worker
  loads in its own process, is loaded once for every domain, and it re-parses
  its options into LLVM's global option registry on every build, which is not
  thread-safe: its builds run one at a time. comgr serialises the compiles of a
  process on a mutex of its own, and NVRTC 13.3 and 13.4's assembler crashes on
  some valid sources, so comgr and NVRTC compile in processes of their own,
  spawned by the compiling domains, where a tinygrad worker process loads the
  library once and compiles in it. Making the compiler finds the library, and
  those processes load it, so a file found that is no library raises
  `Compile_error` at each compile, where it fails making tinygrad's compiler.
  Each process is a small C program per library that tolk carries, which runs
  from memory on Linux and from a file in `Helpers.cache_dir` elsewhere. A
  process serves compiles in turn over a socket (two pipes on Windows), idle
  between them in a list every domain takes from, since a process's first
  compile pays the library's setup: about 100 ms for NVRTC, most of it starting
  CUDA's driver, and about 9 ms for comgr, of which loading it is 4 ms
  (nonnormal: a small kernel's compile takes 23-28 ms in a new process, 15 ms in
  a serving one); a compile that finds none idle starts another. A compile whose
  process ends without a reply raises `Compile_error` naming the signal or exit
  status, where it would end tinygrad's worker, and that process serves no
  other. The NVRTC process disables NVRTC's own cache of binaries
  (`CUDA_CACHE_DISABLE=1`), whose key misses the math options
  (`--use_fast_math`, `--prec-sqrt`, `--prec-div`, `--ftz`), `-I`, `-D` and
  `--std`, so it could serve tolk a binary built under other options or against
  other headers. The comgr process disables comgr's own cache of generated code
  (`AMD_COMGR_CACHE=0`), on by default from ROCm 7.0 under `~/.cache/comgr` (2
  GB on nonnormal), which tinygrad leaves on: tolk's table is the one cache it
  keeps, and comgr's served no measurable time over a cold table (nonnormal,
  gpt-oss-20b's first prefill and step: 0.886 and 0.587 s from a filled comgr
  cache, 0.892 and 0.546 s with it off; rune's `amd/mlp-step` first call before
  serving: 1.04 and 1.01 s with it off, 1.03 and 1.02 s on, against 0.91 s
  served with the verifier).
- **Reason:** (a).
- **Pinned by:** `Tolk.Setting › context › is not seen by the other
  domains` and `Tolk.Setting › context › binds for the domains spawned
  while it runs`; `Tolk.Worker` (every test); `Tolk.Codegen ›
  programs are kept › calls from several domains at once make one program,
  compiled once`; `Tolk.Decomp_dtype, compiled in parallel › targets lowered
  on several domains at once get their own rules`; `Tolk.Compiler_metal ›
  MTLCompiler › compiles
  from several domains at once`; `Tolk.Compiler_amd › compiles in processes
  of their own` (every test) and `› a process that cannot start`;
  `Tolk.Compiler_cuda › compiles in processes of their own` (every test),
  `› a library that does not load` (every test), and `› a process that cannot
  start`; `Tolk.Compiler_amd › a library that does not load` (every test); `Tolk.Compiler_cuda (fork) ›
  a forked process` (every test).

## D6. Devices are named, never parsed

- **tinygrad:** `device.py` `_Device._canonicalize`, `_Device.get_class`,
  `Compiled.__init__`, `Compiled._select_iface` (device strings split at `:`).
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
  devices › on_disk holds for a disk device`, `Tolk.Ops › several
  devices › copy_to_device rejects a disk and a weak type` and `Tolk.Ops ›
  storage › clone rejects a disk`.

## D7. Withdrawn

A device's stamp slots followed nx.device's `Submission.record`, so a batch's
kernels were spans of a profile. rig times each submission itself, and tolk
stamps nothing (D146).

## D8. The disk cache maps strings to strings

- **tinygrad:** `helpers.py` `diskcache_get`, `diskcache_put` (an SQLite
  database of pickled values, keyed by strings, integers or dictionaries of
  columns).
- **tolk:** `Helpers.Diskcache`; `Diskcache.generation`, `Diskcache.renew`
  and `Diskcache.clear`.
- **Differs:** keys and values are strings, which callers encode. Each entry is
  a file under `CACHEDB`, a directory, written aside and renamed into place,
  holding the digest of its key and value: an entry damaged on disk raises
  instead of reading as another value, and a graph's reader makes it anew.
  Tables are versioned by tolk's own version. Entries belong to a generation, an
  empty file named by a count and a unique name; a clear makes one past the
  newest, which empties every table at one step, then removes the old files.
  tinygrad's clear (`helpers.py` `diskcache_clear`) drops the tables one by one,
  each drop committed on its own, so a reader between two drops sees one table
  cleared and another not. A tolk from before generations reads an entry of one
  as a write in progress: its clear and trim remove it only once an hour old,
  and its trim does not count it.
- **Reason:** (a). OCaml has no pickle, no type-safe serialization of
  arbitrary values: a cache returning any type would be `Marshal`, which
  crashes when a rebuilt program reads a value of a changed type. Without an
  SQLite transaction, writing aside and renaming is what keeps an entry whole
  when a writer crashes or races another. A file's contents carry no check
  of their own, and a byte changed in a program or a binary would run as
  another kernel. Without SQLite's drop, a clear
  that removes files one by one orders against no call, and a generation per
  table would cost what one costs while promising less.
- **Pinned by:** `Tolk.Helpers › Diskcache › get reads back any key and
  value put`, `Tolk.Helpers › Diskcache › behaves as a table of entries per
  table`, `› a put and a get on another domain come before or after a
  clear`, `› a put and a get in another process come before or after a
  clear` and `Tolk.Helpers › Diskcache on two domains › behaves as a table
  of entries, whatever the order of the calls`, which run with `CACHEDB`
  set, as dune runs them, and `Graph › a damaged entry reads back as its
  graph, or misses`.

## D9. Narrow float conversions are IEEE conversions

- **tinygrad:**
  - `dtype.py` (`float_to_bf16` rounds to float32 with
    `truncate[dtypes.float]`, then to bfloat16), and `dtype.py` (`const`
    converts an integer to a float through a double, `float(val)`);
  - `codegen/decomp/dtype.py` `l2i`, `f2f`, `pm_float_decomp`: `l2i` converts a
    long to a float32 word by word; an emulated cast narrows its source to
    float32 first; `f2f` flushes subnormals to zero both ways, but for those
    within half a subnormal ulp of the least normal, which round up to it, and
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
  with one rounding. An emulated kernel exists only because its target lacks the
  type, so it must give the values a native one gives. The emulated cast's
  rounding to odd: (e). rune hands tolk only a float32 to cast to a narrow
  float, so no raven call site reaches it, but without it the cast rounds twice,
  first to a float32 to nearest: `1 + 2^-8 + 2^-40` becomes bfloat16 `0x3f80`
  (1.0) where IEEE's one rounding, which `Dtype.truncate` folds, gives `0x3f81`
  (1.0078125). Repro on the CPU with `EMULATED_DTYPES=bfloat16`: `Tensor([1 +
  2**-8 + 2**-40],
  dtype=dtypes.float64).cast(dtypes.bfloat16).bitcast(dtypes.uint16)` gives
  `0x3f80`. The 8-bit facets: (e). rune hands tolk no 8-bit float, since it
  stores each as `Uint8` and converts it itself (rune's `Lower_arith.dtype`),
  but without them an emulated 8-bit float flushes its subnormals both ways,
  where tinygrad's own folder (`dtype.py` `float_to_fp8`) rounds to them: on the
  CPU with `EMULATED_DTYPES=fp8e4m3,fp8e5m2`,
  `Tensor([2**-9]).contiguous().cast( dtypes.fp8e4m3).bitcast(dtypes.uint8)`
  gives `0x00` where the folder gives `0x01`, e5m2's `2**-16` likewise, and the
  code `0x01` widens to `0.0`; and a double rounds twice, through a float32: the
  float64 `1 + 2**-4 + 2**-40` becomes e4m3 `0x38` (1.0) where the folder gives
  `0x39` (1.125).
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
`math.nan` (`dtype.py` `fp8_to_float`).

## D11. Kernel optimisations are typed

- **tinygrad:** `codegen/opt/__init__.py` `Opt` (`Opt(op, axis, arg)`, with an
  `arg` of any type), checked by `codegen/opt/postrange.py`
  `Scheduler.apply_opt`.
- **tolk:** `Opt.t`, `Opt.target`.
- **Differs:** each kind of optimisation is a constructor with its own fields,
  which stand for `OptOps` too, and a split's target is one of the two axis
  types a split can make (`split_targets`, `postrange.py` `split_targets`),
  `Local` and `Upcast`, in their axis types' order. The malformed arguments that
  `postrange.py` refuses at run time cannot be built.
- **Reason:** (a): static types.
- **Pinned by:** the type itself, whose printing `Tolk.Ops › arguments ›
  reprs.golden` checks against tinygrad's `Opt` repr (the row "kernel with
  opts"); `Tolk.Postrange` drops tinygrad's malformed-argument cases with
  this entry as the reason.

## D12. A node's key is a BLAKE2 digest

- **tinygrad:** `uop/ops.py` (`key`, the SHA-256 of
  `str((op, dtype, arg))` followed by the keys of the sources).
- **tolk:** `Ops.key`.
- **Differs:** the same text is digested with BLAKE2b-256.
- **Reason:** (a). No key is ever compared with a key tinygrad computed: keys
  name compiled programs in caches that tolk alone writes, so the digest
  is free to differ, and tolk takes the one OCaml's standard library
  has, which has MD5 and BLAKE2 but not SHA-256.
- **Pinned by:** `Tolk.Ops › key › tells arguments apart`.

## D13. Folding reads and writes committed constants at their width

- **tinygrad:** `uop/symbolic.py` (`fold_const_alu`), which writes a
  folded integer result at its type's width; `symbolic_simple` (the collapse of
  committed const conversions); `uop/ops.py` (`_min_max`), which
  bounds an operand by its written value; `uop/weak.py` (`uncast_const`),
  which leaves the literal bare. tinygrad's own `TestModularWraparound`
  expects the wrapped results and is marked `xfail_broken_const_wraparound`:
  its cases fold to them at this commit, but for the signed one of `test_div`,
  whose `//` floors where the test truncates.
- **tolk:** `Shape.fold_const_alu`; `Ops.at_width` and `Ops.operand_bounds`;
  `Shape.uncast_const`.
- **Differs:** a fold reads a cast of a literal to a type of known width wrapped
  to that width, and so does a cast of it that collapses, and it writes a
  committed integer result wrapped. The bounds read a weak operand at the width
  of an operation on a committed integer, as compiled code commits it (a literal
  operand already holds that value, D413). So `uncast_const` only ever drops a
  cast whose literal fits, and a maximum by bounds that keeps a weak operand
  commits it to the maximum's type, so the operations that read it do not lose
  their width. Reading at the width also keeps `c0 + x < c1 → x < c1 - c0`,
  which both apply to signed `x` only, from a wrong answer where the offset
  wraps; D24 keeps that rule to offsets that do not wrap.
- **Reason:** (b): RFC 0012's Law 1: every integer expression's compiled
  value is eager nx's modular one, signed included. rune's `rune jit ›
  classes › integer constants keep their bits and wrap` and `› a square of a
  product whose constant squares to 0`, at every signed and unsigned width,
  compile constants whose arithmetic wraps against eager nx.
- **Pinned by:** `Tolk.Symbolic › symbolic_simple › constants › an operation
  reads committed constants at their width` (a fold, a cast of a committed
  constant, a uint8 remainder), `Tolk.Symbolic › symbolic_simple › constants › a
  comparison reads a committed constant at its width` (the uncast),
  `Tolk.Symbolic › committed constants hold their type's value` (each case
  above, evaluated before and after), the law `Tolk.Symbolic › laws › sym keeps
  the value of an integer expression at a committed width, wrapping included`,
  and `Tolk.Symbolic › tinygrad › tests.golden › TestModularWraparound.<test>`,
  whose goldens hold the machine values; the offset's interaction with D24:
  `Tolk.Symbolic › integers wrap › an offset crosses a comparison only where
  neither side wraps` (the committed case).

## D14. Withdrawn

Folded comparisons of NaN constants followed IEEE here, where tinygrad compared
them as `ConstFloat`s, which make NaN equal to NaN. At
`uop/symbolic.py` `fold_const_alu` reads each operand through
`const_arg`, which turns a `ConstFloat` into a plain float, so `nan != nan`
folds to `True` there as in `Shape.fold_const_alu`; tinygrad's
`test_const_folding.py::TestWeakConstFolding::test_nan_compare` states it.

## D15. Withdrawn

Making a compiler loaded nothing: its library was loaded at the first compile,
so a renderer for CUDA, HIP or Metal was made on a machine without NVRTC, comgr
or MTLCompiler. No raven call site made such a renderer without compiling with
it, so the entry had no reason; tolk's suites and the bench's `search` rows
rendered for those targets without the libraries. As tinygrad's
`runtime/support/compiler_cuda.py` `NVRTCCompiler.__init__` (`nvrtcVersion`),
`runtime/support/compiler_amd.py` (`HIPCompiler` asserts comgr is loaded) and
`runtime/ops_metal.py` (`MetalCompiler`) do when the compiler is made,
`Compiler_cuda.nvrtc` and `Compiler_amd.hip` find their library and
`Compiler_metal.compiler` loads MTLCompiler, raising `Failure` without it, so
`Cstyle.cuda`, `Cstyle.hip`, `Cstyle.metal` and `Device.renderer` raise it too.
The tests that render for those targets skip without the library, as tolk's GPU
tests do, except HIP's: no machine here has comgr, so the suites that render for
HIP name a file that stands in for it in `COMGR_PATH`, as tinygrad's goldens'
generator marks comgr loaded. NVRTC and comgr are loaded by the processes that
compile (D5), so a file found that is no library fails each compile.

## D16. Withdrawn

A CUDA conversion to an 8-bit float kept an infinity special: an infinity in
e5m2 and a NaN of its sign in e4m3, where nx stores ±448. rune hands tolk no
8-bit float, so `Cstyle.cuda` converts with the saturating constructor, as
tinygrad's `renderer/cstyle.py` `base_rewrite` does: an infinity becomes the
greatest finite value of its sign in both formats. This departs from nx in
e5m2, which keeps an infinity, and from the folder, where `Dtype.encode_format`
keeps one in e5m2 and makes it a NaN in e4m3, as tinygrad's `float_to_fp8`
(`dtype.py` `float_to_fp8`) does beside its own CUDA kernels.

## D17. Each operation on a narrow signed scalar is narrowed in the source

- **tinygrad:** `renderer/cstyle.py` `CStyleLanguage._render` (an ALU used once
  is inlined into its user, whatever its type) with `base_rewrite` (the operator
  text); `pm_narrow_uint`, which computes the operations on an unsigned char or
  short that wrap in an unsigned int and casts them back, for these two types
  only.
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
bfloat, which Metal has no `trunc` of. At `renderer/cstyle.py`
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
`dtype.py` `float_to_fp8`, `fp8_to_float` do.

## D21. Withdrawn

A reshape of a value sharded on two axes divided each by its own shard count,
where tinygrad's divided both by the last one's. Withdrawn: multi-device is out
of scope, see README.

## D22. An emulated long converts no float past its words' range

- **tinygrad:** `codegen/decomp/dtype.py` (`l2i` makes a long's low
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
- **tolk:** each facet's rules below.
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
  - **Integer bounds wrap.** tinygrad: `uop/ops.py` (`wrapped`), which
    `_min_max` applies to an unsigned sum, difference, product, left shift and
    complement, `UOp._min_max` (a cast), `UOp._min_max` (a constant table).
    tolk: `Ops.at_width`, `Ops.min_max`, `Ops.cast_bounds`, and the constant
    table's case of `Ops.compute_min_max`. The bounds of a committed integer
    that leave its type are one value wrapped, or the type's, signed or not,
    whatever the operation: tinygrad wraps those of the unsigned operations
    alone, and to the type's, one value included; an integer cast to an integer
    keeps its interval, which then wraps, where tinygrad gives a cast that
    leaves its type the type's bounds, one value included (69c2c1bce); a
    constant table holding a NaN has its type's bounds, as a NaN constant has.
    tinygrad folds int8 `(y + 100) < 0` for `y` in `[0, 50]` to `false`, which
    is `true` at 40, and so `y * 4 < 0` and `y << 2 < 0`, `true` at 40 too, and
    `cast (cast (y + 100) uint8) int32` to one cast, `-116` where the machine
    gives `140`.
  - **Wrapping rules.** tinygrad: `uop/symbolic.py` `symbolic` (`(x // c1) //
    c2`), `symbolic` (`c0 + x < c1`), `codegen/simplify.py` `pm_reduce_collapse`
    (`x + y < c`, `x * y < c`) and `pm_reduce_load_collapse` (`x + y = c` under
    a cast), `uop/symbolic.py` (`reduce_mul_chain`, a factor out of a maximum),
    `uop/symbolic.py` `symbolic` (`-x < -y`, `-x < c`). tolk:
    `Shape.symbolic_simple`, `Shape.symbolic`, `Simplify.solves_sum`,
    `Simplify.pm_reduce_collapse`, `Simplify.pm_reduce_load_collapse`,
    `Symbolic.reduce_mul_chain`. tinygrad keeps `c0 + x < c1`, the comparisons
    of `Simplify` but the last, and a factor out of a maximum from unsigned
    integers, which wrap, and applies them to signed ones, whose overflow it
    takes to be undefined, checking only that the constants it makes fit the
    type (`fits`, `uop/symbolic.py` `fits`); tolk applies each to a signed
    integer only where every value it computes fits the type, and so `x + y = c`
    to any committed integer, and `-x < -y` and `-x < c` to a signed one only
    where no negation wraps. The comparisons of `Simplify` apply to integers
    only, since moving a float term rounds, and `x + y = c` only under a cast
    that does not narrow, where tinygrad takes any cast whose operand's bounds
    fit its type (7e36b6735). `where (a < b) c (a + (c - b))` is a maximum for a
    signed `a` only where `a + (c - b)` does not wrap (`uop/symbolic.py`
    `symbolic`), where tinygrad takes signed overflow to be undefined. tinygrad
    folds the int8 maximum of `r * p * q` over `r < 2` to `p * q`, `-112` at
    `12, 12` where the machine gives `0`, and int8 `-x < 3` to `-3 < x`, `false`
    at `x = -128` where the machine gives `true`.
  - **Float folds.** tinygrad: `uop/symbolic.py` `symbolic_simple` (`x + 0`),
    `symbolic_simple` (`x / x`, `x * 0`). tolk: `Shape.symbolic_simple`. A float
    `x + 0` is `x` only for `-0.` (`-0. + +0.` is `+0.`); `x * 0` is `0` for
    integers and booleans only (a float product by zero is NaN at an infinity or
    a NaN and `-0.` at a negative `x`); `x / x` is gone, since true division is
    always a float, and `x / x` is NaN at 0.
  - **Signed zeros.** tinygrad: `uop/symbolic.py` `symbolic` (complementary
    selections). tolk: `Shape.symbolic`. For integers and booleans only:
    `where c t 0 + where c 0 f` at `t = -0.` is `+0.`, where `where c t f` is
    `-0.`.
  - **Reassociation.** tinygrad: `uop/symbolic.py` `symbolic_simple` (two
    constants of an associative operation), `symbolic` (like terms), `symbolic`
    (a sum of two selections), `symbolic` (constants to the end),
    `reduce_mul_chain`. tolk: `Shape.fold_where_closure`,
    `Shape.symbolic_simple`, `Shape.symbolic`, `Symbolic.reduce_mul_chain`.
    Sums, products and maxima regroup for integers and booleans only, and a
    factor leaves a float reduction nowhere: `(x + 1e8) + -1e8` at `x = 1` is
    `0.`, where `x + 0.` is `1.`; `(y + x) + x` at `y = 1`, `x = 2^-24` is `1.`,
    where `y + x * 2` is `1.0000001`; `x * 1.1 + x * 2.2` at `1.` is
    `3.3000002`, where `x * 3.3` is `3.2999999`; a sum over `r < 3` of `(r *
    2e38) * 0.25` is `inf`, where `0.25 * 2e38` times the sum of `r` is finite;
    and a maximum keeps a NaN only as its first operand, so `max (NaN, max (x,
    0.))` is NaN, where `max (x, max (0., NaN))` is `x`. `x + x` is still `x *
    2`, which is exact, and `(x * -1) * y` is `(x * y) * -1`, since a negation
    commutes with rounding. tinygrad restricts these rules to integers too
    (7e36b6735), the two stage fold of a sum since 5e826e373, but not that of a
    product or a maximum, nor `x + x * -1`, which it makes `x * 0`: at `x = -1.`
    that is `-0.`, where the sum is `+0.`. It checks that an intermediate it
    makes does not overflow (`no_overflow`, `uop/symbolic.py` `no_overflow`),
    which holds wherever tolk would check it, since the bounds of a committed
    integer wrap (`Ops.at_width`), so tolk makes none.
  - **Maxima.** tinygrad: `uop/symbolic.py` `symbolic`. tolk:
    `Shape.symbolic`. A maximum by bounds applies to integers
    only, since a float's bounds leave out NaN and the order of zeros:
    `max (x, inf)` at NaN is NaN, where the fold gives `inf`. A selection that
    computes a maximum becomes one for integers only: rune builds a float
    maximum that follows IEEE from selections, which the fold would turn back
    into a maximum that renders as the target's own, and it took `-0.` and
    `+0.` for one constant: `where (x < -0.) 0. x` at `-1e-45` is `+0.`,
    where `max (x, -0.)` is `-0.`. A float ReLU stays a selection.
  - **Sigmoid forms.** tinygrad: `uop/symbolic.py` `sym`. tolk: the rules
    are left out of `sym`. `x * (1 / (1 + x))` at `1e-8` is `1e-8`, where
    `1 - 1 / (1 + x)` is `0.`, and at `inf` is NaN, where it is `1.`.
  - **Pow.** tinygrad: `uop/symbolic.py` (`simplify_pow`), `symbolic_simple`
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
  - **Simplify's reductions.** tinygrad: `codegen/simplify.py`
    `pm_reduce_collapse` (`sum_between`), `pm_reduce_collapse` (a parameter
    lifted out of a sum) and `pm_reduce_collapse` (a product by a boolean cast).
    tolk: `Simplify.sum_between`, `Simplify.pm_reduce_collapse`. A float sum
    counted in closed form is `+0.` over an empty part of its range, where `0 *
    inf` is NaN; a product by a boolean cast is a selection for integers only
    (`-1. * 0.` is `-0.` and `inf * 0.` NaN, where the selection gives `+0.`); a
    parameter conjunct is lifted out of a sum on a selection as a selection:
    tinygrad lifts it as a product by its cast, which its product rule turns
    back into a selection, and tolk's product rule does so for integers only. A
    sum over a range its value does not use, `x * n`, and a sum of a sum split
    in two stay: each is the sum of the same terms in another association, which
    RFC 0012's rounded sum class allows; a sum of `-0.` terms computed as `-0. *
    n` is `+0.` once the lowering adds `+0.` to every float sum's result, which
    the `x + 0` fold now keeps.
  - **Transcendental polynomials.** tinygrad: `helpers.py` (`polyN`, from
    `0.0`). tolk: `Transcendental.poly_n`. A
    polynomial starts from its first coefficient, so the decompositions no
    longer rely on the float folds of `0. * x` and `x + 0` to be what tinygrad
    renders.
  - **Tensor-core accumulators.** tinygrad: `codegen/__init__.py`
    (`pm_wmma_add`, which adds the running sum to a WMMA's accumulator operand).
    tolk: `Codegen.wmma_accumulate`, which the fold at the reduce calls (D260).
    A WMMA built for a sum starts from a zero accumulator, and the running sum
    used to be added to it, `+0. + acc`, which only the float `x + 0` fold
    removed: with that fold kept to `-0.`, every tensor-core loop added `+0.` to
    each accumulator element on every iteration. The running sum now replaces a
    zero accumulator: the sum's identity is the accumulator's initial value, set
    once before the loop, and the lowering adds `+0.` to the result, so a
    partial sum's zero sign, which the rounded sum class leaves open, is all
    that differs. A core of another accumulator, which only a core built by hand
    has, since the optimiser builds every core from zero, keeps it, and the sum
    adds its value as it adds any term: tinygrad's rule regrouped it as `(c +
    acc) + P`.
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
  - transcendental polynomials: the `Tolk.Transcendental` goldens, whose value
    tables are unchanged;
  - tensor-core accumulators: the sources of every tensor core,
    `Tolk.Cstyle › sources › by default › metal_tc_*`, `hip_tc_*` and
    `metal_matmul`, whose loops add nothing to the accumulator, as tinygrad's;
    `Tolk.Codegen › tensor-core accumulators › the running sum
    replaces a zero accumulator`, which lowers a contraction on each
    target's tensor cores, one or two steps of K a trip; and the values of
    folded sums on CUDA's cores, `Tolk.Codegen › tensor cores on CUDA › a
    float16 or bfloat16 product of 1 to 4 tiles a side, summed in float32
    from 0, is the exact product`;
  - the goldens.

## D25. A product and a sum round apart, except into a sum

- **tinygrad:** `runtime/support/compiler_cpu.py` (Clang's arguments),
  `runtime/support/compiler_amd.py` (HIP's options), `runtime/support/
  compiler_cuda.py` (NVRTC's options), `runtime/ops_metal.py`
  `MetalCompiler.compile` (Metal's parameters): none turns floating-point
  contraction off. `codegen/__init__.py` (`lower_reduce`: a sum adds its
  products unfused), `codegen/decomp/op.py` `get_late_rewrite_patterns` (`a*b +
  c` becomes `MULACC` for a renderer that writes it), `renderer/cstyle.py`
  `CStyleLanguage.code_for_op` (no C-style language writes `MULACC`).
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
  fused`; the Codegen and C-style goldens, with tolk's output naming D25 in `.tolk`
  files beside them.
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

- **tinygrad:** `renderer/cstyle.py` `CStyleLanguage.code_for_op` (`Ops.NEG` is
  `-{x}`), `CStyleLanguage.code_for_op` (`Ops.SUB` is `({a}-{b})`).
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

## D27. A constant or a move keeps a NaN's bits

Each part is one layer of the head's NaN rule: a float constant is its bits
(D27), and a move or a negation keeps them.

### Constants keep a NaN's bits

- **tinygrad:** `dtype.py` (`DType.const` makes every NaN `math.nan`),
  `dtype.py` (`ConstFloat` compares every NaN equal to every other), and
  `uop/symbolic.py` (`fold_bitcast` converts the constant with
  `truncate` before reading its bits).
- **tolk:** `Dtype.equal_const`, `Dtype.hash_const` and `Dtype.const`;
  `Ops.arg_key`, which `Ops.key` reads; `Shape.fold_bitcast`.
- **Differs:** a float constant is its bits. `Dtype.const` keeps a NaN's sign
  and the payload its type holds, a signalling NaN staying one, in a float of 16
  bits or more (an 8-bit float's NaN is the one its storage holds, D10 and D20),
  and constants compare and hash by their bits, so NaNs of different bits are
  different nodes, with different keys: a schedule cached under one NaN's key is
  never the schedule of another's. No golden reads a key; tinygrad's writes
  every NaN `nan`. `fold_bitcast` reads a float constant as its type stores it,
  where tinygrad's conversion quiets a signalling NaN and makes an 8-bit NaN
  canonical. So folding `bitcast(bitcast(0x7d01, half), uint16)` gives `0x7d01`,
  where tinygrad gives `0x7e00`, and a negative float32 NaN's bits survive a
  fold, where tinygrad makes them `0x7fc00000`. A NaN an operation makes from
  other values is `Dtype.nan`, tinygrad's `math.nan`, whatever the host's FPU
  gives, so that folding does not depend on the host: `exec_alu` makes the NaN
  of an invalid operation (`inf - inf`, `0 * inf`, `0 / 0`, a square root or
  logarithm of a negative) canonical, where x86 gives a negative NaN. The NaN
  rule leaves that NaN's bits free, so this part follows tinygrad.
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

### A move of an emulated float keeps its bits (D62)

- **tinygrad:** `codegen/decomp/dtype.py` (`pm_float_decomp`, which
  widens every load of an emulated float to the emulating float and narrows
  every store back); `renderer/cstyle.py` (`CUDARenderer.render_kernel`, which
  declares vectors of `half`, `bfloat16` and the 8-bit floats only).
- **tolk:** `Decomp_dtype.moved` and the store rule of
  `Decomp_dtype.pm_float_decomp` that stores it; `Cstyle.cuda_kernel`.
- **Differs:** a store of a value that moves stored bits without arithmetic
  (a load, a constant the float holds exactly, a selection between such
  values, and stacks and lanes of them) stores those bits, as the storage's
  unsigned integers. tinygrad widens and narrows each such value, which keeps
  its value but makes every NaN its format's quiet one: a copy of e5m2's
  `0x7d`, half's `0x7c01` or bfloat16's `0x7f81` stores `0x7f`, `0x7e00` or
  `0x7fc0`. Arithmetic still converts, and gives a NaN as the NaN rule allows
  (D9). A move keeps a merged access's lanes together, so it loads and stores
  vectors of the storage, which CUDA does not name as the renderer writes them
  (`unsigned_short4`, `unsigned_char4`): the CUDA renderer declares a vector of
  `unsigned char` or `unsigned short` as it declares one of `half`.
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
- **Pinned by:** the `Decomp_dtype` suite (`test/codegen/decomp/decomp_dtype`):
  `NaNs › an emulated copy keeps every NaN code, a signalling one's included`
  and `› an emulated selection keeps every NaN code, a signalling one's
  included`, on every emulated float, and the goldens of the `where`, `flip`,
  `gather` and `pad` kernels; the Codegen suite's `emulated floats' vectors on
  CUDA › a copy of every code, a thread's lanes moved together, keeps it`
  (`bfloat16` on sm_75, `fp8e4m3` on sm_80 and `fp8e4m3fnuz` on sm_89, run on
  the machine's first CUDA GPU) and `› writes an sm_75 copy of bfloat16 lanes
  with its vector declared`. On rune: missing (the old rune's `Compiled › host › 8-bit floats › a
  copy of every 8-bit float code keeps its bits` and `› a selection of every
  8-bit float code keeps its bits` went with its rewrite).

### A NaN constant is written from its bits (D156)

- **tinygrad:** `renderer/cstyle.py` (`base_rewrite` writes every NaN constant
  as the language's `nan` cast to its type), `CStyleLanguage.nan` (`NAN`),
  `ClangRenderer.nan` (Clang's `__builtin_nanf("")`),
  `CUDARenderer.render_kernel` (CUDA's `#define NAN
  (__int_as_float(0x7fffffff))`) and `HIPRenderer.__init__` (HIP's 8-bit floats,
  `f32_to_fp8(NAN, …)`).
- **tolk:** `Cstyle.nan_const`, the `bitcast` of each language
  (`Cstyle.cstyle`, `Cstyle.metal_lang`, `Cstyle.cuda_lang`) and
  `Cstyle.cdna_rewrite`.
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

### A folded selection or negation keeps a NaN's bits (D176)

- **tinygrad:** `uop/ops.py` (`exec_alu`), which truncates every
  result to its type through `dtype.py` (`truncate`), whose
  conversions to `float`, `half` and `bfloat16` quiet a signalling NaN.
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

### Clang selects a 16-bit float by its bits (D325)

- **tinygrad:** `renderer/cstyle.py` (`CStyleLanguage.code_for_op`'s
  `Ops.WHERE`, `(c?a:b)`) and `base_rewrite`'s gated `Ops.LOAD`,
  `(gate?*p:alt)`, under `ClangRenderer`'s `type_map` of
  `ClangRenderer.type_map`, which renders a half as `__fp16`; a bfloat16 is a
  `__bf16` off x86.
- **tolk:** `Cstyle.bit_where`, Clang's `code_for_op` of `Op.Where`, and
  `Cstyle.bit_loads`, in its rewrite, for `Float16`, and for `Bfloat16` where
  it is a `__bf16`.
- **Differs:** a scalar selection of such a value, and a gated load's merge
  with its alternative, choose between the arms' bits:
  `__builtin_bit_cast(__fp16, (unsigned short)((c?A:B)))`, each arm read as
  an `unsigned short`. Clang computes `c?a:b` of two `__fp16`s in float, a
  storage format's promotion, and narrows the result back; on x86 the
  conversions are `vcvtph2ps` and `vcvtps2ph`, which quiet a signalling
  NaN, wherever LLVM's folds leave the round trip. A `__bf16` is not
  promoted, but riscv64's backend, in clang 19, moves a selected one
  through float32 and narrows it with `__truncsfbf2`; arm64's keeps its
  bits, and takes the rule for one form across targets. Vector selections
  are not promoted and are unchanged. Clang's `code_for_op` writes the
  scalar selection, as `written.tolk`'s row of a half `Ops.WHERE` shows.
- **Reason:** (e): a move keeps a NaN's bits. rune's scatter selects an
  element no update reaches from its target, and for a half `Add` on the
  host, in the kernel its comparison form makes, `0x7c01` came back
  `0x7e01`.
- **Pinned by:** the Cstyle suite's `selections of 16-bit floats` (the
  source of each Clang target); the goldens `clang_selected_half` and
  `written`; rune's `Lower_index › gathers and scatters › a scatter keeps the
  bits of each element no update lands on` and `Rune.jit`'s `a scatter keeps
  the bits of each element no update lands on`, on the host and on a GPU.

## D28. A host program has rig's entry

- **tinygrad:** `codegen/__init__.py` (`pack_args`: a kernel of a
  renderer with `packed_args` loads its arguments from one byte parameter at
  the offsets of `TinyELF.iter_sig`), `do_compile` (the binary is
  the source compiled), `runtime/ops_cpu.py` (`CPUProgram.__call__`,
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
  bits in any blocks` and `link and run › custom kernels with
  variables` (four tests).

## D29. A float32 tensor core takes operands widened from its narrow input

- **tinygrad:** `codegen/opt/postrange.py` (`_apply_tc_opt`: a core
  applies only when both operands of the multiply have its input dtype),
  `Scheduler._apply_tc_opt` (the core multiplies the multiply's own operands).
- **tolk:** `Postrange`'s `tc_operand`, `try_core` and `use_wmma`.
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
  narrow type. A `float32` operand that is not such a cast takes no core, and
  tolk never narrows one by a cast of its own: a `bfloat16` core would round
  a value the program did not round, outside `Nx.contract`'s bound for
  `float32`'s `u`. Attention's values, `float32` weights by `bfloat16` values,
  so run without a core; they are bound by reading the weights (at 2048 slots
  of 8 heads, 128 MiB, 233 us at fudge's 576 GB/s), not by arithmetic. A
  program that wants the core rounds its operand itself, and this rule then
  applies. rune never gives tolk an 8-bit float (rune's
  `Lower_arith.dtype` stores each as `Uint8`), so no contraction of rune's
  reaches an 8-bit core. Cores are tried in the renderer's order, so Metal,
  whose `float32` core comes first, keeps it. tinygrad never builds the
  widened form, which its `dot` does not make, so no golden it made before
  changes.
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
  interface. The Codegen suite's `tensor cores on CUDA › a float32 operand by
  a widened bfloat16 takes no core, and is the exact product` pins the
  refusal.

## D30. Withdrawn

A range around calls on devices with queues was a loop of their batch, its
commands repeated per trip. A range is a loop of rig's program, one submission
per trip of each submission under it (D146).

## D31. Payne-Hanek reduces exactly, to the nearest quadrant

- **tinygrad:** `codegen/decomp/transcendental.py`
  (`payne_hanek_reduction`) and `sin_poly_large`.
- **tolk:** `Transcendental.one_over_two_pi`, `Transcendental.turn_fraction`,
  `Transcendental.radians` and `Transcendental.payne_hanek`.
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
- **Reason:** (b): rune's `cos` and `tan` reduce by `pi/2` with this
  reduction from 30 in magnitude, in float32 and float64, and with
  Cody-Waite's below (`Lower_real.reduce`, `trig_switch`); its `sin` is
  tolk's `Sin`, whose `xsin` uses it beyond its switch-over, 30 by default.
  Both must give nx's sine and cosine there. Selecting the table's words
  by bits is (d), compile time: every compiled sine or cosine of an
  unbounded angle builds the reduction. Selected as tinygrad's `_take` does,
  over this 40-word table, the float64 words take 345 selections and 43
  comparisons of 728 nodes; by bits 111 and 6 of 428. rune's compile bench
  row `trig/sin`, one float64 sine of an unbounded angle, makes the first
  call 0.14 s of CPU against 0.17 s and 1.41 M minor words against 1.95 M
  (fudge, 2026-11-08, Clang included).
- **Pinned by:** `Tolk.Transcendental › reductions ›
  payne_hanek_near_multiples.golden`, remainders and quadrants of float32 and
  float64 angles near multiples of `pi/2`, from pi to 1400 bits, which the old
  reduction fails; `› payne_hanek_reduction removes quarter turns`; and the
  `payne_hanek_*`, `xsin_*` and `values` goldens; and rune's compile bench row
  `trig/sin`.

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

## D41. The memory plan sees a range's calls and the buffers views reach, holds no copy past its last call, and plans a device's copies with its kernels

- **tinygrad:** `schedule/memory.py` (`memory_plan_rewrite` takes each
  entry's buffers from its sources after the first, through `_collect_bufs`,
  which passes buffers, `MSELECT` and `MSTACK` only) and `memory_plan_rewrite`
  (`buf_hold`: a copy's buffer lives `last - first + 1` calls past its last);
  `schedule/__init__.py` (`create_linear_with_vars` holds the buffers
  among its call's sources).
- **tolk:** `Memory.viewed`, `Memory.uses` and
  `Memory.memory_plan_rewrite`'s lifetimes (`call_bufs`, the events) and
  lanes (`key`);
  `Schedule.create_linear_with_vars`'s `held_bufs`.
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
  reaches. A planned
  buffer takes whole blocks of 256 bytes of its arena, so one placed past the
  arena's other buffers can cost what its own storage did not: rune's
  `compile/scan/checks-48` row's program holds 75 520 bytes, where it held
  75 328 with a buffer of 320 bytes, reached through a view, apart.

  A copy's buffer is free from the call after its last, as a kernel's is,
  where tinygrad holds it as many calls again, lest a later copy overwrite it
  while the first may still run. Copies and kernels of a device share one
  lane in every schedule, where tinygrad keeps copies in a lane of their
  own. tinygrad's copy lane keeps reuse from ordering a copy against a
  kernel, which rig's device order makes moot on one device: on fudge's
  RTX 5000 beside a memory device, a value moved into the GPU mid-chain is
  a submission of the GPU's copy queue, ordered after its earlier products
  with either lane count, and `run` returned in 12.2 ms with one lane
  against 11.8 ms with two. Across devices rig orders by allocation, which
  a lane avoids only in part, and two lanes add their peaks, past RFC
  0005's bound for a schedule linked per run (D173). Where a host-run
  device's work reuses bytes a GPU copy touched, the call waits for the
  GPU: a memory device's sum placed over a value just moved into the GPU
  waited for all eight products before it, `run` returning in 168-200 ms
  against 13 ms where that order does not reuse the bytes. A copy lane
  avoided that case (51 ms) but not host work that writes the lane after a
  GPU copy used it; removing the wait is rig's to do.
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
  bytes with it and 256 without. (b) for one lane: the same order makes a
  copy lane guard nothing on one device, and rune's fully sharded steps
  hold within RFC 0005's bound only with one.
- **Pinned by:** the Engine suite (`test/engine/tolk_engine`): `link and run › a
  planned buffer a range writes is not placed over one it leaves` and `› a
  buffer a range writes through a view is not placed over another`, each of
  which fails without its half; the Memory suite (`test/schedule/memory`):
  `memory_plan_rewrite › the buffers a call takes › a buffer a call views is
  planned` and `› a buffer viewed after another's life is not placed over it`,
  `› rules › a copy's buffer is free from the call after its last` and `› a
  copy's buffer and a call's of one device share bytes once their lives are
  apart`, the recorded `copy_chain`, `copies_share`, `random_*` plans, and the
  law over drawn schedules, whose calls view some buffers, whose copies live no
  longer than their last call and whose device has one arena of at least its
  live bytes; the Jit suite's `jit_lower › arenas › a device's buffers share one
  arena, a copy's among them` and its recorded `weights_copy_lowered`,
  `held_constant_lowered`, `split_multidev_xfer_lowered` and
  `split_multidev_copy_lowered`; the Schedule suite's recorded `copy_view`
  schedule, whose copy reads a view of a buffer its call is given.

## D42. Withdrawn

A batch's command buffers, and its placeholders where a device had no window,
were pinned memory. tolk makes no command buffer, and a program's memory is
rig's (D146).

## D43. Withdrawn

A queue bufferized its own commands, and a region kept its alignment. tolk
encodes no queue (D146).

## D44. An integer cast of a weak expression computes in integers

- **tinygrad:** `uop/weak.py` (`cast_weak_srcs`), `dtype.py`
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

- **tinygrad:** `renderer/cstyle.py` (`CStyleLanguage.code_for_op` has no
  `FDIV`) and `ClangRenderer.code_for_op` (Clang's adds it);
  `codegen/decomp/op.py` `get_late_rewrite_patterns` (a target that lists `FDIV`
  gets its reciprocals as divisions, and a product by a reciprocal, `a * (1/b)`,
  as the quotient `a/b`).
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

## D52. A -0. fill or alternative is not +0.

A selection or a pad by `-0.` keeps `-0.`: tinygrad's tests for 0 take
`-0. == 0` as true.

### A selection of -0. stays off a gated load

- **tinygrad:** `uop/symbolic.py` (`pm_move_where_on_load`), whose
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

### A pad by -0. fills its padding with -0. (D155)

- **tinygrad:** `mixin/op.py` (`_pad_constant`) and `pad_to`,
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

## D54. Storage says what it knows of where it starts within 16 bytes

- **tinygrad:** `uop/ops.py` (`ParamArg`, which has no such field) and
  `codegen/late/coalesce.py` (`memory_coalescing` merges a run of `l`
  elements where `l` divides its element offset, taking every buffer to start
  on a 16-byte boundary). tinygrad folds a view's offset into a kernel's
  index, except for a contiguous slice that it stages without a copy
  (`schedule/__init__.py` `contiguous_mops_to_view`): a later kernel
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
- **Reason:** (b). rune's jit runs a program over the storage it is given, each
  buffer at its phase (rune's `Compile.storage`), and mapped weights put a
  tensor at any byte offset of its file: a vector access from an address that is
  not a multiple of its width faults on CUDA, and is undefined in Metal, which
  requires device pointers aligned to their type: there it reads the wrong
  elements once Metal merges adjacent vector loads at constant offsets into one
  wider load, which drops the low bits of the address. Single loads at computed
  indices are not merged, and read correctly. A batched range (D30, since
  withdrawn) binds each trip's window at its own address, so a kernel vectorised
  for a window that starts aligned faulted on x86, where Clang emits `movaps`,
  on the trips that moved it by 4 bytes. A rune loop (`Loop.cut`) passes its
  body a stage of what it reads that does not change, and the schedule makes a
  stage of a contiguous view the view: norn's NUTS step on a posterior whose
  data the model reads from its sixth double failed verification, its body
  compiled for storage on a boundary and passed the data 8 bytes past one. Those
  two failures were on the host, in Clang's vector types aligned to their size;
  on the CPU device they are now aligned to their element (`renderer/cstyle.py`
  `ClangRenderer.render_vector_prefix`), so an access there is defined from any
  element's address: the merge there takes every buffer to start on a boundary
  of its widest access, as tinygrad's takes it on 16 bytes
  (`Coalesce.memory_coalescing`; the width, D187).
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

- **tinygrad:** `renderer/cstyle.py` (`_render_dtype` names a
  vector after its element's name in `type_map` with spaces made
  underscores) and `MetalRenderer.type_map` (renames only
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

- **tinygrad:** `codegen/decomp/transcendental.py` (`xlog2` selects
  NaN where `d < -0.0`, then `-inf` where the reciprocal of `d` is `-inf`,
  which it means for `-0.0`).
- **tolk:** `Transcendental.xlog2`.
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
  the negative float16 rows of `values.tolk` of magnitude below
  `2^-16`; in rune, `Lower_real › transcendental ›
  log2 within 4 ulps at float32, 4 at float64`, whose golden holds negative
  float64 subnormals at NaN.

## D58. Withdrawn

`Spec.program` rejected an elementwise operation of a program on a vector
(the host's renderer, which computes on vectors, was checked against a second
specification), so that a vector left after devectorize failed at lowering.
No consumer got a wrong value or a failure without it: CUDA rejects such a
kernel at NVRTC, Metal and the host compute it. `Spec.program` is
`spec_program` (`uop/spec.py` `spec_program`) on every renderer.

## D60. A loop of a call stays in the schedule

- **tinygrad:** `schedule/rangeify.py` (`pm_no_views`, which strips
  every view of storage), `split_store` (which makes a kernel of
  an end of a call), `pm_renumber_range` (which renumbers the ranges
  of that kernel), `schedule/__init__.py` (`create_schedule`, which
  schedules the call of an end and drops the end), and `uop/spec.py`
  (`spec_kernel_graph`, which admits device ranges only, and no end).
- **tolk:** `Rangeify.loop_range`, `Rangeify.pm_no_views` (which keeps a view
  that moves with a loop's range) and `Rangeify.split_store`;
  `Schedule.create_schedule` (`loops`, `argument` and the kept end);
  `Spec.loop`, `Spec.loop_bound` and `Spec.kernel_graph`.
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
  `get_kernel_graph › loops of calls › a loop of a precompiled call is no
  kernel` and `› a loop's call reads views that move with its range`; the
  `Schedule` suite (`test/schedule/schedule`): `create_schedule › rules › an end
  of a call over a loop schedules the call in its loop`, `› a loop runs after
  the call that writes what it reads`, `› a loop runs before the call that
  overwrites what it reads`, and `create_linear_with_vars › loops of calls › a
  loop of a precompiled call is a loop of its body's calls` and `› a loop inside
  a loop's body keeps its own end`; and the `Tolk_engine` suite
  (`test/engine/tolk_engine`): `link and run › a scan runs its body once per
  trip, carrying in place`, against the unrolled loop on the host, and `› a
  scan's merged calls carry in place as its calls apart do` (D404), on the host
  and each GPU present. The `Spec` suite (`test/uop/spec`): `kernel_graph ›
  loops of calls › accepts a loop of a call over a loop range`, `› refuses a
  loop of a call over a range of any other kind` and `› refuses a weak sum of a
  weak integer variable`; its `verdicts.tolk` records tolk's spec on ends and
  shrinks.

## D61. Withdrawn

A stack took the per-shard sub-view of a whole source beside sharded ones.
Withdrawn: multi-device is out of scope, see README.

## D62. Withdrawn

A move of an emulated float kept its bits; merged into D27, which holds its
rule, its reason and its pins.

## D63. Programs and schedules are kept on disk

- **tinygrad:** `codegen/__init__.py` (`to_program_config`, `to_program_key` and
  `to_program_cache`, a dictionary of the process), `schedule/__init__.py`
  (`lower_sink_to_linear`'s schedule cache, on disk only from `SCACHE=2`),
  `helpers.py` (`SCACHE`, `1` by default), and `codegen/opt/postrange.py`
  `Scheduler.get_optimized_ast` (a kernel's name, coloured).
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
  `getenv` where it compiles (`BEAM_ESTIMATE`,
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

- **tinygrad:** `renderer/cstyle.py` (`create_non_native_float_pats`,
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
  narrow float, rounding a more precise source to odd first with
  `Decomp_dtype.narrow` (`Lower_arith.store_real`), so no raven call site
  reaches the rule.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `casts to
  bfloat16 › an integer or a double rounds once on the host`, for int64,
  int32, uint32, uint64 and float64 sources, among them doubles just past a
  tie, one of them subnormal. On rune, which rounds itself: `Lower_arith ›
  casts give eager's values`, whose float64 operands include the doubles one
  unit each side of the bfloat16 tie `1 + 2^-8`.

## D65. Withdrawn

An operation on an emulated narrow float rounded its result to the float in the
kernel, so that a chain of operations rounded after each, as nx does. rune
computes each narrow-float operation in float32 and encodes its result itself
(`Lower_arith.narrowed`), so no operation on an emulated float reaches tolk from
rune. The operation rule of `Decomp_dtype.pm_float_decomp` computes in the
emulating float and rounds at the store, as `codegen/decomp/dtype.py`
`pm_float_decomp` does; a cast to the emulated float rounds, as
`pm_float_decomp` does.

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
end. At `uop/ops.py` `UOp.contiguous_view` the rewrite marks nothing and finds a
view of any value, a constant or a computed one included;
`Prepare.contiguous_view` does the same.

## D73. A gather is an INDEX of a tensor by a value with axes

- **tinygrad:** `uop/ops.py` `UOp._shape` (INDEX's shape: its index sources'
  shapes, then the rest of the source's), which no tinygrad pass puts in a
  tensor graph; `schedule/indexing.py` (`pm_generate_realize_map` and its
  rules), `data_srcs` (the sources an INDEX's ranges reach), `pm_apply_rangeify`
  and `pm_apply_rangeify` and `run_rangeify`'s consumer ranges;
  `schedule/prepare.py` (`_mop_index`), `pm_mops` and `prepare_rangeify`.
- **tolk:** `Indexing.storage`, `Indexing.is_gather`,
  `Indexing.realize_gathered`, `Indexing.data_srcs`, `Indexing.convert_gather`
  and its rule in `Indexing.pm_apply_rangeify`, and
  `Indexing.assign_ranges`'s consumer ranges of a gather;
  `Prepare.move_index`, `Prepare.mops` and `Prepare.pm_tensor_mops`.
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
  for `Nx.take` and `Nx.take_along_axis`. D68, which recognised a one-hot sum
  over the axis in the scheduler, was withdrawn. The sharded clauses are
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

- **tinygrad:** `uop/ops.py` (`UOp._min_max`), which bounds binary
  operations on integers only, has no case for `TRUNC` or `NEG`, and
  bounds a `WHERE` by both its branches whatever selects them,
  and gives a value it cannot bound, or a cast it cannot, its type's
  `min` and `max`, finite in the 8-bit formats without infinities
  (`dtype.py` `DType.max`);
  `codegen/decomp/transcendental.py` (`xsin`), which builds the
  Payne-Hanek reduction unless its caller passes `fast`, and chooses between
  the reductions with `x_abs < switch_over`, a comparison that the
  decompositions' rewrite (`symbolic_simple`) does not fold.
- **tolk:** `Ops.unbounded`, `Ops.compute_min_max` (its `Where`, `Trunc`
  and `Neg` cases), `Ops.selected`, `Ops.float_bounds` and
  `Ops.cast_bounds`; `Transcendental.xsin`.
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
  value, as tinygrad does since d94ec221b (`uop/ops.py` `UOp._min_max`), whose
  cast of an infinite bound into an 8-bit format without infinities is NaN
  and falls back to the type's finite bounds, where tolk's stays
  unbounded. With
  the type's finite greatest value instead, a compiled sine, cosine or
  tangent of an 8-bit NaN gave -1, 0 and -452. Neither bounds a float
  comparison to `true`, since a NaN compares false (d94ec221b).
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

## D75. A constant condition assumes nothing in its branches

- **tinygrad:** `uop/symbolic.py` (`fold_where_closure`), which
  substitutes `True` for a selection's condition in its true branch and
  `False` in its false branch, whatever the condition is.
- **tolk:** `Shape.fold_where_closure`.
- **Differs:** `fold_where_closure` leaves a condition that is a constant,
  broadcast, padded or not, in the branches: a broadcast constant is one node
  for every use of that constant, so substituting it is no assumption about
  the selection. In `where F (where F x (where F y x)) y`, with `F` a
  broadcast `False`, the outer selection makes the inner `F`s `True`, the
  inner selection makes its false branch's `True` back into `False`, and the
  rewrite does not terminate; tinygrad raises "infinite loop in
  graph_rewrite" on that graph, and tolk's rewrite, without this rule,
  "graph_rewrite does not terminate".
- **Reason:** (e): a graph tinygrad's rewrite does not terminate on, valid
  by construction. The fold of a selection whose condition is a constant
  through movements, this entry's other part, is withdrawn: its witnesses,
  the old rune's staged scan on Metal and the multi-device rewrite, are gone,
  and rune's suites pass without it. A selection by a `CONST` still folds, as
  in tinygrad.
- **Pinned by:** the Symbolic suite: `symbolic › selections › a broadcast
  constant condition is not folded in the branches`, which raises without
  the rule, and `› a padded constant condition is not folded in the
  branches`.

## D76. Withdrawn

A stack of `Invalid` lanes stayed a stack, since tinygrad folded it to one
`Invalid` and the lanes' reshape, select and reduce lost their width.
tinygrad now folds it to an `Invalid` expanded to the stack's shape
(`uop/symbolic.py` `Shape.pm_data_invalid`), which keeps the width, and
devectorizes no expanded scalar into a stack. The kernels that pinned the
entry, whose upcast lanes all read an `Invalid` index, compile and sum to zero
with tinygrad's rule (`Tolk.Codegen › lanes all Invalid`, `lanes of an
unrolled reduce`).

## D77. A broadcast value that runs a transcendental stays stored

- **tinygrad:** `schedule/indexing.py` (`run_rangeify`, which stores an
  elementwise value or reduction whose ranges a broadcast ends),
  `BufferizeOpts.removable` and `create_bufferize_and_index_srcs` (its value),
  and `schedule/rangeify.py` (`cleanup_dead_axes`) and `remove_bufferize` (which
  inlines that store again when its value reads at most three buffers and no
  reduction reads a buffer).
- **tolk:** `Indexing.assign_ranges` (which records the values a broadcast
  stores) and `Indexing.bufferize_and_index` (which marks their stages
  `Broadcast`); `Ops.keep`, in place of `removable`;
  `Rangeify.remove_bufferize`, which keeps such a stage.
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
  `kernel_counts.golden` and their kernel goldens.

## D79. A chain of one operation keeps its right operand's grouping

- **tinygrad:** `renderer/cstyle.py` (`base_rewrite`'s ALU rule), which
  drops the parentheses of any operand that is the same associative operation
  (`ADD`, `MUL`, `XOR`, `OR`, `AND`), on either side, but the right operand
  of a signed integer operation since 845977979, where regrouping can
  overflow.
- **tolk:** `Cstyle.base_rewrite`'s ALU rule.
- **Differs:** only the first operand drops them, of floats and unsigned
  integers too. C groups `a*b*c` as
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
  `renderer/cstyle` and `engine/jit` goldens. On rune: missing (the old rune's
  `quant › values › compiled, a product is eager's › the largest scales` went
  with its rewrite).

## D78. Withdrawn

The CUDA device's kernels compiled to a cubin, so that a process's first
load would not wait for the driver to translate PTX. On fudge (RTX 5000 Ada),
rune's compile bench row `cuda/sin-nested-8`, a fresh process's first call
of eight nested float64 sines over a driver cache of its own, took 455 and
517 ms with cubins and 390 and 340 ms with PTX (medians of 3, interleaved
runs), and `warm/cuda/sin-nested-8`, the same call over a disk cache an
earlier process filled, 340 and 329 ms against 218 and 178 ms: no gain from
cubins. As tinygrad's `renderer/cstyle.py` `CUDARenderer.__init__`,
`Cstyle.cuda` compiles PTX for the CUDA device and a cubin for the NV device.
D320 compiles a cubin again, measured with the driver's cache missing.

## D80. A host program runs its output loop in blocks on the host's cores

- **tinygrad:** `runtime/ops_cpu.py` (`CPUProgram.__call__`), which calls a
  kernel once per launch on the calling thread; `codegen/__init__.py`
  `full_rewrite_to_sink` (the pipeline after `pm_cast_const`), `do_number_param`
  (and its pass) and `do_to_program`'s estimates; `uop/ops.py` (`KernelInfo`)
  and `ProgramInfo.from_sink`; `engine/realize.py` (`get_call_kernel_args`) and
  `exec_kernel`.
- **tolk:** `Codegen.split_blocks`, its place in
  `Codegen.full_rewrite_to_sink`, `Postrange.misses`,
  `Codegen.pm_number_params` and
  `Codegen.whole_loop`; `Ops.kernel_info`'s
  `split`; `Shape.program_info_of_sink`; `Realize.get_call_kernel_args`;
  `Tolk_engine.compute_work`, `Tolk_engine.memory_work`,
  `Tolk_engine.memory_blocks`, `Tolk_engine.blocks` and
  `Tolk_engine.split_grid`.
- **Differs:** for a CPU target, once the kernel is final (after
  `pm_cast_const`), the largest range of axis type `WEAK` whose end reads no
  range, of at least two iterations, and on which every store to global
  memory, by its address, and every call and piece of custom code, by its
  sources, depend, runs `block_lo + r` for `r` below
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
  variable: counted in the loop's 32-bit type, a count past 2^31 would wrap.
  A host launch cuts `[0, n)` into `min(n, ops / 2^16)` blocks where
  `ops > mem`, and `min(n, max(ops / 2^18, min(8, mem / 2^18)))` otherwise,
  at least one, `ops` and `mem` the estimates' operations and bytes of
  memory, each a group of
  rig's host of as many threads as iterations, which its pool runs on the
  performance cores, and the entry (D28) runs the
  iterations its threads number; tinygrad's own launch (`whole_loop` in
  `engine/realize.py`) runs one block, the whole loop.
- **Reason:** (d): sofo's and symo's training steps on the CPU (outside the
  repository), which ran each kernel on one core. Their hottest kernels,
  split by hand into 32 blocks on kimchi's 8 E-cores (`taskset -c 6-13`,
  best of 30), took 6.17 ms and 0.77 ms (r_256_8_4_4_32_4), 3.38 and 0.42
  (r_32_32_4_4_128_16), 0.28 and 0.04 (r_256_3_3_100_4) and 17.8 and 2.83
  (r_128_64_3_4_400_100_4). On fudge, against one block (medians of 11, two
  interleaved runs each), the tolk bench's
  `lorenz/r_128_16_3_4_4_3_3_400_400/heuristic` takes 0.79 and 0.63 ms
  against 5.85 and 6.04 ms, and rune's `Jit/sin-float32-64K` 82 and 74 us
  against 360 and 356 us, `Jit/normal-float32-1M` 2.8 and 2.6 ms against
  16.9 ms. A block costs its threads' start and join: rig's pool bench on
  fudge, workers spinning, starts and joins an empty job on all 14 cores in
  1.3-1.5 us, and a job of 2 us a core ends 2.1-3.7 us past its work
  (`launch/empty-all-cores`, `launch/2us-chunks-all-cores`, three runs at
  loads 5.1-8.5). At the 19 operations a nanosecond one of fudge's cores
  does (one block of `sin` over [8; 1024]), 2^16 operations take 3.4 us,
  about twice that cost. At 2^17 rows gained less (`Jit/matmul-64` 0.57 of
  2^18's against 0.36 at 2^16); at 2^15 some gained more
  (`JitFootprint/rnn-fwd-h20-replay` 0.34 against 0.57) and
  `WarmStart/transposed-matmul-64` took 1.28 times as long as at 2^18
  (medians, 4 interleaved runs at loads 3.9-9.9), so 2^16 is the smallest at
  which no row lost. A kernel of more operations than bytes takes a block
  for each 2^16 of them. Against 2^18, every other kernel as before, 6
  interleaved runs each at loads 3.7-9.9: the tolk bench's
  `lorenz/r_32_3_2_4_3_400/heuristic` (1.03 M operations, 3 blocks to 15)
  takes 5.7-40.6 us against 15.2-35.8 (median 0.55 of 2^18's), rune's
  `JitFootprint/rnn-fwd-h20-replay` 189-200 against 340-352, `Jit/matmul-64`
  8.5-8.9 against 23.9-24.4 and `Jit/grad-of-jit-mlp-float32` 65-68 against
  112-139. No row of either bench whose split moves is slower beyond the
  spread of the unchanged `host/conv/heuristic` in the same runs (62.8-89.2
  against 60.2-82.9); `host/tangent_products/heuristic`, 11 blocks to 32,
  takes 18.7-112.0 against 18.2-93.5 (median 1.34, the control's 1.12), and
  0.81 of 2^18's where the same 32 blocks ran with every kernel at 2^16. A
  kernel that moves as many bytes as it does operations shares memory's
  bandwidth among its blocks, so a further block speeds it less: a block for
  each 2^16 operations made the concatenations of rune's
  `Jit/cat-3-views-scaled-4096x64` and `Jit/kv-concat-decode`, which move 2
  and 4 MiB (25 and 24 blocks), slower than at 2^18 (medians 1.61 and 1.50,
  4 interleaved runs at loads 3.9-9.9), and a block for each 2^16 bytes made
  `JitLoops/newton-grad-1024-replay`, whose 512 KiB copy takes 8 blocks for
  2, slower in 7 of 8 interleaved runs (median 1.10, 27.6-54.6 us against
  26.8-29.2, loads 5.1-9.8) and `Jit/devices/fsdp-backward-8` in 6 of 8
  (1.08), while `host/max_rows/heuristic` (0.67),
  `Jit/cumsum-scaled-float32-100x400` (0.57) and
  `JitLoops/iterate-64-rows-into-4096x64-replay` (0.90) ran faster: the
  gains are larger, but bytes keep 2^18 so that no row loses. At 2^15
  operations or bytes 64 KiB copies split in 2 made
  `Jit/sort-with-positions-4096` and `Jit/eigh-float32-64x8x8` 1.5-1.6 times
  as slow. Such a kernel keeps a block for each 2^18 operations or bytes.
  A block for each 2^17 or 2^16 bytes made a row slower, as did a cap on a
  launch's threads apart from its blocks (6 interleaved runs each at loads
  3.4-6.0, over every row of the rune and tolk benches whose split moves). At
  2^17 bytes rune's `Jit/cumsum-scaled-float32-100x400` takes 27.0-30.2 us
  against 35.0-38.0 and `JitLoops/iterate-64-rows-into-4096x64-replay`
  77.0-83.8 against 83.8-91.3, but `JitLoops/newton-grad-1024-replay`, whose
  512 KiB copy takes 4 blocks for 2, is slower in 6 of 6 runs (paired median
  1.04; 1.12 at 2^16 bytes). A block for each 2^16 bytes up to 64, run on the
  threads 2^18 bytes ask and the others queued on them, gained no row beyond
  the spread of the unchanged rows and made `newton-grad` slower in 6 of 6
  runs (1.16), `Jit/cumsum-neg-log-uniform-8x1024` in 5 of 6 (1.12) and
  `host/max_rows/heuristic` in 6 of 6 (1.01). So a launch runs a thread for
  each block, up to the performance cores.
  One of fudge's cores moves about a byte in the time of an operation (14-17
  bytes a nanosecond, one block of `x * 2` over [4096; 64]): a copy or a
  cheap elementwise kernel, counted by its operations alone, ran on one
  core. Rounding the blocks past 14 down to a multiple of fudge's 14 cores,
  so that no wave runs short, gained nothing (`host/sin_nested_8` 15.4-34.4
  against 17.8-29.7, `host/vecmat_1000` 145.7-448.6 against 117.2-229.6), so
  a launch does not read the host's cores. Bytes ask at most 8 blocks,
  measured: a memory-bound kernel gains to about 8 blocks, and on a loaded
  host each further thread can be held up (rune's `Jit/kv-concat-decode`, 4
  MiB, 4 interleaved runs at load 4.6-6.6: 6 blocks 62-69 us, 8 57-95, 16
  83-143, 32 89-185). Against the operations alone, 5 interleaved runs each
  at load 4.8-5.8, rune's `Jit/cumsum-float32-4096x32` takes 39.6-50.5 us
  against 72.1-80.0, `JitLoops/iterate-64-rows-into-4096x64-replay`
  83.9-91.2 against 114.7-123.7 and `JitLoops/newton-grad-1024-replay`
  26.0-26.7 against 30.2-31.5, and no row of the rune or tolk bench whose
  split moves is slower. In the assembly prototype's cases, 4 interleaved
  runs each at load 4.6-5.4, `x * 2` over [4096; 64] takes 36.1-42.2 against
  124.0-277.0, and a decode step whose key and value concatenations are
  stores that attention reads 126.8-128.4 against 243.6-256.7. tinygrad
  deleted its CPU threading (`OptOps.THREAD`) at 4197f7423, before the pin.
- **Pinned by:** the Codegen suite: `host programs in blocks › a host program
  splits the loop each store's address reads`, `› a split loop's estimates
  count its whole loop past 2^31`, `› stores of separate loops
  split none`, `› a store of no loop splits none`, `› a reduction's loop is
  never split`, `› a serial loop is never split`, `› a loop shrunk by its
  guard splits at its shrunk end, unguarded`, `› a split loop of 2^25 iterations keeps 32-bit indices` and `› a split's bounds and a variable take slots of their own past buffers that skip one`;
  `calls and custom code beside host blocks › a loop beside one that does
  not depend on it does not split`, `› one beside a loop runs once` (a
  call and code, counting atomically, which counted 8 for 1 when the loop
  split), `› a loop that one depends on splits, beside no store` and `› a
  loop that code depends on splits`; the
  Tolk_engine suite: `host programs in blocks › a launch of more operations
  than bytes runs a block per 2^16 operations, another per 2^18 operations
  or bytes, bytes up to 8`, `› a program writes the
  same bits in any blocks`, `› a kernel that moves many bytes for few
  operations writes its values` and `› a run of a large kernel computes its
  sums`, `a variable named as a block's bounds is a variable like any
  other`, and `a kernel whose estimates fit no int › splits and runs`.

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
`shared_spec` (`uop/spec.py` `spec_shared`) does.

## D87. Withdrawn

A store through a gather of a sharded value stored each shard's rows.
Withdrawn: multi-device is out of scope, see README.

## D88. A float zero a host comparison reads reaches C as a value the backend cannot see

- **tinygrad:** `renderer/cstyle.py` (`ClangRenderer`, whose
  `string_rewrite` writes a float zero as a literal, `0.0f`).
- **tolk:** `Cstyle.float_zero`, `Cstyle.opaque`, `Cstyle.opaque_zeros`
  and `Cstyle.clang` (which adds it for each host target with its types).
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

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  matrix-vector case, which applies only when the first reduce's product has two
  loads as operands, `mulop.src[0].op is Ops.INDEX and mulop.src[1].op is
  Ops.INDEX`, the vector's index has the first weak reduce range as a term of
  its sum and its ranges are among the matrix's: that axis split into
  `MV_THREADS_PER_ROW` local threads, 8, the first global axis that 16 divides,
  whichever operand reads it, into `MV_BLOCKSIZE` local threads, 4, then
  `MV_ROWS_PER_THREAD` upcast lanes, 4, whatever the matrix's layout, size and
  reuse), `hand_coded_optimizations` (the tensor cores, tried first),
  `hand_coded_optimizations` (the upcasts of output axes, which find reuse only
  on an axis some access does not read and take 3 or 4 of it), and
  `codegen/opt/postrange.py` (`apply_opt`, which refuses a local split of a
  reduce axis whose reduction runs inside a weak range that a reduction closes,
  a serial reduce).
- **tolk:** `Heuristic.lanes`, `Heuristic.rows`, `Heuristic.columns`,
  `Heuristic.sector_lanes`, `Renderer.t.stream_threads`, `Heuristic.in_flight`,
  `Heuristic.in_flight_held`, `Heuristic.stream_first`,
  `Heuristic_common.upcast_max`, `Heuristic.summed`, `Renderer.t.per_byte_first`,
  `Heuristic.per_byte`, `Heuristic_common.run`,
  `Heuristic.reuse_ranges`, `Heuristic.reuse`, `Heuristic_common.stride`,
  `Heuristic.accesses`, `Heuristic.reduces_in`, `Heuristic.contraction`,
  `Heuristic.contracted`,
  `Heuristic.units`, `Heuristic.main_access`, `Heuristic.layout`,
  `Heuristic.streamed`, `Heuristic.early`, `Heuristic.stream`,
  `Heuristic.streams` and their order in
  `Heuristic.hand_coded_optimizations`; `Postrange.Scheduler.apply`'s check
  of a group inside a reduce.
- **Differs:**
  - the contraction: the kernel's first reduction that sums a product of
    two computations of accesses, neither reading a reduction that runs in
    a range the sum closes, on a renderer with local indices and shared memory;
    `MV=0` turns it off. An operand may read its loads through conversions
    (`CAST`, `BITCAST`), decode codes and scale a vector (D81's part), and
    read a reduction of the outputs alone, which comes before the sum: a
    row's expert that an argmax over the experts finds is computed once a
    row;
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
    target alone (D170). `stream_threads` and D175's `local_workgroups` are
    fields of the renderer, since a second reader, rune's attention
    lowering, sizes its grid by them. `per_byte_first` is one, since
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
  - reuse in registers, rows held: `stream/dense-bf16-12` 889-914 to
    267-271 us, `stream/q8_0-12` 1030 to 431-438 us, `stream/mxfp4-12` 5.35-5.39
    to 3.09-3.10 ms (its decode stored as float32, D163), the routed blocks
    of 16 rows of a 512-token prompt `stream/routed-512-bf16` 33.2 to 10.2 ms
    (from a tensor core) and `stream/routed-512-f32` 57.8-58.0 to 20.6 ms,
    kaun's `Quant/cuda/routed-product-512-tokens` 82.5 to 44.4-44.6 ms. Rows
    held per thread against 4 SIMD groups: at `R = 16` 10.2 against 22.6 ms
    and 20.6 against 28.6 ms; at `R = 2` (dense bfloat16, 16384 by 4096)
    264.6-265.0 against 265.1-265.2 us;
  - the order against the cores, on the rows' kernels as rune compiles them
    today: `stream/dense-bf16-16` 834-844 us on a core to 281-283 us
    streamed; `stream/dense-bf16-32`, where a thread's 16 rows leave two
    passes over the operand, 429-431 us on a staged core and 541-545 us
    streamed, the one row the order loses, kept for gpt-oss's prefill
    (`Heuristic.stream_first`); at `R = 64` the core keeps it
    (`stream/dense-bf16-64`, 433-436 us with and without `MV`). The
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
    is the cost of one program per target. When the order was introduced, end
    to end: kaun's `GptOss/prefill-512` 446-452 to 186-187 ms and
    `GptOss/experts-at-prefill-512` 18.5-18.7 to 4.66-4.68 ms, as the cores
    took back what the layouts had streamed. Those two rows no longer measure
    the layouts: their kernels are the cores', so turning the contraction off
    (`MV=0`) leaves them at 122.0 against 123.1 ms and 2.77 against 2.79 ms
    (minima of four interleaved rounds). The rows that hold the layouts are
    tolk's `stream/*` and `gpt_oss/*` and kaun's `Quant/cuda/*`: few outputs
    and decoded products, 1.65-8.7x and 3.7-65x. The rule
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
  - a routed product whose rows' experts an argmax in the kernel finds,
    gpt-oss-20b's shapes of 32 experts' `[2880; 2880]` matrices
    (`stream/routed-argmax-16-bf16`), the general rules against streamed,
    GPU time of a call, three interleaved processes each: bfloat16 at 1
    row 138-142 to 35-51 us, 16 rows 1.235-1.251 to 0.502-0.519 ms (the
    matrices read at 91% of the bandwidth), 64 rows 3.10-3.18 to 1.99-2.01
    ms; float32 at 16 rows 2.74-2.77 to 1.00-1.01 ms. Such a kernel arises
    only where the reduction reads no storage, a function of positions: one
    that reads a buffer, as a router's top-k over its logits, is stored
    (`remove_bufferize`), and its product reads the ids;
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
- **Pinned by:** the Heuristic suite: `the optimisations chosen are tinygrad's ›
  <environment> › the optimised kernel`, cases `matvec_*`, `vecmat_*`,
  `gpt_oss_*`, `experts_down_*`, `routed_blocks_*`, `routed_own_*`,
  `routed_shared_*` (blocks sharing their experts' matrices, on a core),
  `routed_at_bound_*` and `routed_past_bound_*` (63.9 a byte streamed, 71.0 on a
  core), `shared_keys_*`, `rows_m{4,12,32}_*`, `q8_0_blocks_*`, `gguf_mxfp4_*`,
  `mxfp4_m12_*`, `rows_short_*` (a sum of 64 on the cores) and
  `routed_by_argmax_*` (rows whose experts an argmax in the kernel finds,
  streamed); `streaming › the streamed operand's reuse outside a thread is its
  reuse over its tile`, `› a SIMD group's load of an operand streamed along the
  reduce reads whole sectors`, `› the unroll is the largest divisor of its loop
  up to 8, 4 with reuse` (covering a contracted loop left beside an unroll), `›
  a contraction streams up to a reuse of 32, before a tensor core` and `› a
  recorded contraction streams before a tensor core up to 64 multiply-adds per
  byte of its streamed operand` (rune's kernels of one token's products, of
  blocks of their own matrices, of rows routed by an argmax in the kernel and of
  gpt-oss-20b's experts at a 512-token prompt), `› each renderer streams before
  a core up to 64 multiply-adds a byte` (the renderers' fields) and `› a
  contraction streams before a core up to the renderer's bound a byte`; `the
  hand-coded optimisations keep a kernel's writes › small kernels ›
  experts_down_metal` (a group inside another reduce); the Postrange suite's
  cases `double_sum_group`, `double_sum_group_twice` and
  `double_sum_unrolled_group`; on fudge, tolk's bench `stream/*` and `gpt_oss/*`
  rows and kaun's `Quant/cuda/*` rows.

## D90. A linked schedule resolves its launches at link

- **tinygrad:** `engine/realize.py` (`resolve_params`) and `exec_kernel`, which
  find a call's buffers, launch dimensions and variable values on every run, and
  `uop/ops.py` (`UOp._sym_fxn`, a cached property that simplifies a symbolic
  integer once and compiles it to a Python function, which `UOp.sym_infer` calls
  on each evaluation).
- **tolk:** `Tolk_engine.affine`, `Tolk_engine.value`, `Tolk_engine.locate`
  and `Tolk_engine.launch`; `Shape.sym_compile`, and
  `Shape.sym_infer`, which is `Shape.sym_compile` applied once.
- **Differs:** linking a schedule turns each call into a launch per lane of
  rig's program (D146), whose buffers, offsets, scalars and grid are words of
  the run's ints or constants: each variable of the run is a word, a value
  computed from them a word the run computes once (`Shape.sym_compile`, which
  simplifies once and computes on `int`s, exactly where a value does not fit),
  and a value that moves with a loop's trips a hole of rig's program, the sum of
  trip words times constant factors. `Shape.sym_compile` is the port of
  `_sym_fxn`, and `Shape.sym_infer` of `sym_infer` (`uop/ops.py` `sym_infer`),
  which evaluates once; a caller that evaluates an integer many times keeps the
  function `Shape.sym_compile` returns, where tinygrad caches it on the node. A
  `DEBUG` report of a call lists the run's variables that the schedule reads,
  where it listed every variable of the run.
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

## D92. On the host, a decoded operand's rows share its decodes as the registers allow

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  upcasts of output axes, which find reuse only on an axis that some access
  does not read, and take 3 or 4 values of it).
- **tolk:** `Heuristic_host.share_decodes`, `Heuristic_host.unbudgeted_rows`,
  `Heuristic_common.run`, `Stage.decoded` and `share_decodes`' place in
  `Heuristic.general`.
- **Differs:** on the host (`target.device = "CPU"`), before the other
  rules, each output axis along which `f > 1` outputs read each value of a
  decoded operand of the summed product of the contraction (D89) at one
  index, `f` the axis's size where the operand does not read it and `d`
  where it reads it only as `r / d`, is upcast, later axes first, by the
  largest amount up to 16 that divides the axis and one such operand's `f`
  and keeps the kernel within the host's budget (D186's registers and D241's
  conversions and iteration). An operand is decoded when its value converts integers it
  reads into floats, by a cast or a bitcast, outside any access's address.
  The axis's lanes then share one decode of each value. A host renderer
  that does not know its registers takes at most 4. On a GPU a streamed contraction takes the
  same reuse as a term of the operand's reuse (D89), whatever the operand's
  type; one that does not stream takes no run.
- **Reason:** (d): a product with a decoded matrix decodes each value once
  for each tile of the rows that read it, and on the host, where D163
  inlines the decode into the product, tinygrad's 4 rows decoded a value 16
  times at 64 rows. Upcast first and bounded by the budget, the rows take 8
  on a file of 16 registers (one register a row: their stores lie apart),
  and the columns' upcast of 4 still fits beside them: BLIS's fusing factor,
  "as large ... as possible, given the constraints of the floating-point
  datatype and the size of the architecture's register set" (Van Zee and
  van de Geijn, TOMS, §4.3). On fudge's AVX2, one P-core, interleaved, the
  kernels of kaun's `Quant/host/mxfp4-product-of-1024-{10,64}-bfloat16-tokens`
  took 4.29 and 18.6 ms with 4 rows and 2.90 and 15.1 ms with 8; float32
  rows 3.46 and 16.3 ms, then 2.27 and 12.05 ms, against 2.6 and 8.5 ms for
  the decode stored and the 4-row product, a gap D348's panel closes;
  rune's `Jit/experts-mxfp4-512x4of32-256`
  64.3-64.7 ms, now 48-53 ms. Past the budget, 16 rows by 4 columns took
  13.7 and 54-56 ms at 64 rows and 512 tokens, 8 rows by 8 columns 13.3 and
  54-58, 16 rows by 2 columns 14.6 and 65. Files of 32 registers (NEON,
  AVX-512) take 16 rows unmeasured: fudge has neither. Before D163's dropped digit,
  `Nx.map_segments` over `Nx_quant.apply` on a prompt took 195-226 ms with
  blocks of 16 rows upcast by 4 against 6.0-6.9 s without.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › in the default environment › the optimised kernel`, cases
  `routed_*_cpu` and `routed_*_avx2`; `the host's decodes`: the
  decodes of products of 1 to 512 rows on four register files and on a host
  without registers, blocks of 16 to 128 rows on four files, and a larger
  file decoding no more times; `the hand-coded optimisations keep a
  kernel's writes › large kernels › routed_blocks_*` (slow); on fudge,
  kaun's `Quant/host/mxfp4-product-of-1024-*` and
  `Quant/host/routed-product-64-tokens`, and rune's
  `Jit/experts-mxfp4-512x4of32-256`.

## D93. The where-closure rule leaves loads alone, and its slice queries walk no more than they need

- **tinygrad:** `uop/symbolic.py` (`fold_where_closure`, which asks
  whether the condition is in `t.bool_slice` or `f.bool_slice`, and whether
  the condition or a branch reaches an `INDEX`); `uop/ops.py`
  (`_bool_slice`, a `recursive_property`: each node keeps the set of the
  boolean nodes it reaches) and `backward_slice` (a
  `cached_property` holding the whole slice of each node asked, and
  `op_in_backward_slice_with_self`, which walks it);
  `codegen/late/coalesce.py` (`memory_coalescing`, which addresses a
  load of several elements by a `SHRINK` of the buffer where one element's
  address is an `INDEX`).
- **tolk:** `Shape.fold_where_closure`, `Ops.reaches`, `Ops.ops_reached` and
  `Ops.op_in_backward_slice_with_self`.
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

- **tinygrad:** `helpers.py` (`_DEV`, the `DEV` setting, and
  `DEV.target`); `device.py` (`_select_renderer`, which renders for
  `DEV.target` of the device's kind); `runtime/ops_cpu.py`
  (`CPUDevice`, whose arch, remote or not, is this machine's
  `platform.machine()` and `native`).
- **tolk:** `Tolk_engine.target`; `Device.renderer`; `Helpers.Target`.
- **Differs:** there is no `DEV` setting. `Tolk_engine.target` is a function
  of the device: its vendor's kind, which its driver declares
  (`Rig.capability`), and its `arch`, or the host's CPU target.
  `Device.renderer` takes the target and renders for it with the renderer
  of its kind. `Helpers.Target` keeps the
  record and its syntax, which tests and the goldens' generator write.
  A host compiles for its `Rig.arch`: on x86-64 its psABI level, such as
  `x86_64,x86-64-v3`, and on arm64 `generic`, the base architecture; this
  machine's host also tunes for its processor, as in
  `x86_64,x86-64-v3,tune=native`, where tinygrad's CPU, remote or not,
  compiles for this machine's instruction set and processor, `native`.
- **Reason:** (c). rig places values on devices, and the compiler
  compiles for the device the values live on. A target read from the
  environment applies to every device of the process, so it compiles for a
  device the values do not live on, and a malformed `DEV`, meant for another
  program, failed every program linking tolk when it started. Another
  machine's processor may lack instructions this machine's has, and a
  program records the arch it was compiled for, which `Rig_program.load`
  checks against the device's: `native` would use instructions beyond the
  level it records. Tuning chooses among the level's instructions and adds
  none.
- **Pinned by:** `Tolk.Setting › startup › starts whatever DEV holds`;
  `Rune.jit on another machine's host › another machine's host compiles for
  its arch` and `› gives eager's bits`; `Tolk_engine › target › the host
  compiles C for Clang at its arch, tuned for its own processor, …`;
  `Tolk.Compiler_cpu › a program for x86-64's baseline runs here`;
  `Tolk.Device › renderer › picks a device's renderer as tinygrad does`
  (each row renders for the target tinygrad's `DEV` gives the device) and
  `renders for the target it is given`.

## D96. Withdrawn

A matrix-vector workgroup's rows were the matrix's own; merged into D89.

## D100. A reduction of no axes that a broadcast ends is stored

- **tinygrad:** `schedule/indexing.py` (`run_rangeify`, which stores
  an elementwise value or reduction whose ranges a broadcast ends by
  realizing each of its axes, so a value of no axes is not stored).
- **tolk:** `Indexing.assign_ranges`.
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
- **Pinned by:** the Rangeify suite's `sum_all_broadcast` and `finite_checks`
  rows of `kernel_counts.golden` and their kernel goldens, and its
  `reduce_expand_child` and `preserve_multistage_reduce` rows; the Jit suite's
  `nonzero` and `masked_select` goldens. On rune: missing (the old rune's
  `lower_linalg` goldens `qr_q` and `qr_r`, whose reflections' norms were
  stored, went with its rewrite).

## D101. A group shares its threads with the kernel's independent reductions

- **tinygrad:** `codegen/opt/postrange.py` (`apply_opt`: a local
  split of a reduce axis splits that axis alone, and sizes the shared
  memory by the first reduction's type).
- **tolk:** `Postrange.Scheduler.siblings`, `Postrange.Scheduler.smem` and
  its use in `Postrange.Scheduler.apply`.
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
  goldens and its claim that a barrier follows each local store.

## D102. Withdrawn

An empty argument of a precompiled call, which the size-0 rule made an expanded
constant, reached the kernel graph as its scalar, so that a call could take an
argument with no element. Its consumer was rune's staged scan with an empty
carry, which rune now replaces by zeros in the step (`Loop`), and no other raven
call passes an empty argument to a precompiled call: rune's suite passes with
the rule gone. `Rangeify.no_indexing_calls` is `schedule/rangeify.py`
`no_indexing_calls`, and such a call fails the kernel graph's verification on
the expand, as there.

## D103. An integer constant is written as the value its type holds

- **tinygrad:** `renderer/cstyle.py` `base_rewrite` (the constant rules: an
  `int64` is its value with `l`, a `uint32` or a `uint64` its truncated value
  with `u` or `ul`, an 8- or 16-bit integer a cast of its value, any other its
  value).
- **tolk:** `Cstyle.int_const`.
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
- **Pinned by:** the Cstyle suite: `execution on the host › stores each integer
  constant as its type holds it` and `› converts an infinity or a NaN to an
  integer type`; `clang_transcendental_f16.golden`; rune's `rune jit` suite:
  `classes › integer constants keep their bits and wrap`. On rune, a float
  converted to an integer past 64 bits: missing (the old rune's
  `test_jit_programs` went with its rewrite).

## D104. A value read widened from a narrower type is stored narrow

- **tinygrad:** `schedule/rangeify.py` (`bufferize_to_store`, which
  stores a stage's value at its committed dtype).
- **tolk:** `Rangeify.widens`, `Rangeify.bufferize_to_store` and
  `Rangeify.pm_add_buffers`' rule for the index of a narrow store through its
  widening.
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
  `projection_of_stored_{metal,cuda,amd}` (the tensor cores apply).

## D105. A tensor core takes rows that share the other operand's tile

- **tinygrad:** `codegen/opt/postrange.py` (`_apply_tc_opt`: M runs along a
  range of `in0` that `in1` does not read, N along one of `in1` that `in0` does
  not read), `Scheduler._apply_tc_opt` (the core multiplies the multiply's own
  operands), and `codegen/opt/heuristic.py` (`hand_coded_optimizations` tries
  the cores on a kernel of one reduce axis only, below `TC_OPT=1`).
- **tolk:** `Postrange.Scheduler.in_tiles`, `Postrange.Scheduler.try_core`'s
  `roles` and `Postrange.Scheduler.use_wmma`'s `own`;
  `Heuristic.decoded_product` and `Heuristic.tensor_cores`.
- **Differs:** a range both operands read is the core's M (or N) range of one
  of them when the other reads it only as `r / d`, `d` a multiple of the
  core's M (or N): that operand reads one value over each tile of the core,
  and the core reads its bits of the range at 0. Either operand may be the
  core's A: the choices with `in0` as A come first, as tinygrad orders them,
  then those with `in1`. A summed product with a decoded operand
  (`Heuristic.decoded_product`) tries the cores whatever its number of reduce axes.
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
  experts' matrices), and its
  `streaming › a recorded contraction streams before a tensor core up to 64
  multiply-adds per byte of its streamed operand` over gpt-oss-20b's expert
  kernels; on fudge, kaun's `GptOss/experts-at-prefill-512` and
  `GptOss/prefill-512`; the Postrange suite's `tc_operands_swapped`
  (the second choice of a matmul makes its `in1` the core's A). On rune, the
  values on a GPU: missing (the old rune's `Rune.quant › gpus › CUDA › on
  CUDA, a prompt's routes in blocks are eager's` went with its rewrite).

## D106. A value every output of a reduction reads stays stored

- **tinygrad:** `schedule/rangeify.py` (`remove_bufferize`, which
  inlines a stored value that reads at most three buffers and no reduction
  reading a buffer, whoever reads it) and `schedule/indexing.py`
  (`run_rangeify`, which stores a value its consumers read at different
  indices without marking it broadcast).
- **tolk:** `Rangeify.remove_bufferize`'s `read_by_every_output`, with
  `Rangeify.reduced` (a range a reduction of the graph closes), and
  `Indexing.assign_ranges`, which marks such a value `Broadcast` when its
  consumers broadcast it.
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
  (a value two full reductions read, inlined).

## D107. Withdrawn

A batch submitted a streamed queue after the queues it waited for. rig orders
submissions across queues (D146).

## D108. Withdrawn

On the host, a load a reduce reads filled the lanes by 2; merged into D186,
where the fill is a candidate within the register budget.

## D109. Withdrawn

On the host, a kernel of several reduces unrolled within 32 lanes; merged
into D186, whose budget counts an unroll by the lanes it holds across a
loop no more: only an upcast of the outputs adds accumulators.

## D110. Withdrawn

A matrix-vector product with few outputs took fewer threads across them;
merged into D89.

## D111. A hung AM device is lost, never recovered in the process

- **tinygrad:** `runtime/ops_amd.py` (`_collect_interrupts`, whose `reset`
  resets the compute engines through `AMDev.recover`, rewinds the compute queue
  and moves the timeline past the hung value, and `on_device_hang`) and
  `AMDDevice.__init__` (`can_recover`, true under AM off a virtual function);
  `runtime/support/am/amdev.py` (`recover`).
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

- **tinygrad:** `device.py` (`Compiler.__init__`, which drops `cachekey` when
  `CCACHE` is 0, and `compile_cached`: a binary is kept under its source in the
  table `cachekey`); `runtime/support/compiler_cpu.py` `ClangCompiler.__init__`,
  `runtime/ops_metal.py` `MetalCompiler.__init__`,
  `runtime/support/compiler_cuda.py` `NVRTCCompiler.__init__` and
  `runtime/support/compiler_amd.py` `HIPCompiler.__init__` (the tables, named by
  the compiler, the architecture and a key); `codegen/opt/search.py`
  `beam_search` (the beam search's key: the kernel, the width,
  `allow_test_size`, the device and the suffix).
- **tolk:** `Renderer.Compiler.v`'s `cachekey` (a function asked once) and
  `Renderer.Compiler.compile_cached`; `Compiler_cpu.statement`, and the
  tables: `Compiler_cpu.clang`'s, `Compiler_metal.table`,
  `Compiler_cuda.nvrtc`'s and `Compiler_amd.hip`'s; `C.identity`;
  `Codegen.program_key` and `Codegen.kept`; the search's settings
  (`Setting.beam_padto` to `Setting.beam_estimate`) and `Search.beam_search`'s
  key.
- **Differs:** a table also names everything besides the source that
  determines a binary: for Clang, the digest of what `clang -###` states it
  runs for a source, its version and installation first, of the command of
  its compiler proper that compiles a short and a long source, with the IR
  verifier on (D275: the processor and features `native` resolves to, every
  option), and of the length between them (D167), with
  `-ffile-compilation-dir=.` keeping the
  working directory out; for MTLCompiler, NVRTC and comgr, the build of macOS
  or the library's file, size and modification time, and the options, PTX or
  cubin included. Clang runs once per process to state its command: the
  options depend on the LLVM release it states (D167), and a process asks
  first for those of the release the last process found, a hint kept in the
  disk cache's table `clang_llvm` while `CCACHE` holds, and again when Clang
  states another. The
  hint picks options only, and the table digests what Clang states for
  them, so a stale hint costs a second run and never an object of another
  compiler. A key on the binary's path, size and modification time alone,
  ccache's default `compiler_check`, would miss the processor `native`
  resolves to, Clang's configuration files and the environment its driver
  reads. The table is the compiler's identity whatever `CCACHE`
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

- **tinygrad:** `helpers.py` (`getenv`, read where it is called, with that
  call's default) and `ContextVar`; `USE_TC` and `TC_OPT` (`TC`, default `1`,
  and `TC_OPT`, default `0`); `codegen/opt/search.py` `actions` (the actions,
  reading `getenv("TC", 1)` and `getenv("TC_OPT", 2)`);
  `codegen/opt/heuristic.py` `hand_coded_optimizations` (`TC_OPT`'s default, `2`
  during a search and `0` otherwise); `engine/jit.py` `jit_lower`
  (`getenv("JITBEAM", BEAM.value)`); `engine/realize.py` (`compile_linear`'s
  `profile`, `PROFILE or DEBUG >= 2` unless given, for `hcq_compile`);
  `codegen/__init__.py` `full_rewrite_to_sink` (`os.environ.get("DBGTV")`, read
  on each failure).
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

- **tinygrad:** `codegen/opt/search.py` (`_try_compile`: `to_program`,
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

- **tinygrad:** `codegen/opt/search.py` (`_try_compile` calls `to_program`);
  `codegen/__init__.py` (`do_to_program` takes a sink or a program at any
  stage), `to_program_cache` and `do_compile` (calls `compile_cached`);
  `engine/realize.py` (`_get_call_to_compile` takes a program not yet compiled).
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
  every search for the life of the process. A stage resumed by `to_program` is a
  stage reached by `Codegen.linearize` or `Codegen.compile`, so no caller loses
  one. The entry also kept memos of each kernel and each source a search
  compiled, for duplicate candidates measured on AMD; they are withdrawn: on
  fudge, the tolk bench's `search/beam/r_128_16_3_4_4_3_3_400_400` took 0.52,
  0.50 and 1.39 s with them against 0.47 and 1.26 s without, in the same rounds
  (medians of 11, interleaved runs), and a first call of `Rune.jit ~beam:2` of a
  64 by 64 float32 product and its tanh on the host 9.4, 7.3 and 7.0 s with them
  against 3.7, 7.6 and 10.0 s (medians of 3 cold processes). As tinygrad's
  `codegen/opt/search.py` `beam_search`, every candidate compiles, and a binary
  timed before is dropped after its compilation.
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
`codegen/opt/search.py` `Search.beam_search` starts from its
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

## D122. Withdrawn

Storage that a stage or a copy fills carried the bounds of its value, and the
parameter a kernel read it through kept them, so a gather read its clamped,
stored indices within their bounds; a bitcast between integers of one width
kept the bounds of a value both types hold. rune's check of a gather's index
then folded away where the index is clamped and stored. On fudge's RTX 5000
Ada, with D211 withdrawn, the bounds change nothing end to end: minima of four
interleaved rounds, one layer's experts 2.77 ms with the bounds and 2.80 ms
without, the 512-token prefill 121.8 ms and 121.7 ms. Its measured gain on an
M1 Max (a routed MXFP4 product 42.6 to 36.4 ms) is not re-measured, and speed
held only on another machine is no reason.

## D123. The engine runs a loop of calls while a flag holds

- **tinygrad:** `uop/spec.py` `spec_shared` (a `BACKEDGE` closes a bound-less
  range, inside a kernel), `engine/realize.py` (`run_linear` runs every call of
  a schedule once, in order).
- **tolk:** `Spec.shared` and `Spec.kernel_graph` (a `Backedge` around
  calls, in the tensor and kernel-graph specs), `Schedule.is_loop` and
  `Schedule.linearize` (`create_schedule` keeps a back edge of a call),
  `Memory.uses` (the plan counts its calls and flag), and
  `Tolk_engine.loop`.
- **Differs:** an `Op.Backedge` stands around a call of a schedule, or a linear
  of calls, with a loop range of at most as many trips and a flag, one boolean
  of storage the calls write. The schedule keeps it as an entry, and the engine
  makes it a loop of rig's program (D146) that reads the flag before each trip
  and runs the calls while it holds, its trip in a word of the run's ints, which
  a call's scalar argument reads too (D140). The flag is storage the program
  makes or a buffer a caller binds. tinygrad's schedules run each call once, and
  stop on no value. A range of one trip around a call folds to a constant, which
  `Shape.symbolic` drops from the end, so that the call runs once; tinygrad
  deleted that rule (07efd3877), since its ends close only a kernel's ranges,
  where code generation drops a constant (`codegen/simplify.py`
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

- **tinygrad:** `codegen/opt/search.py` (`timeout_handler`, and the
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

- **tinygrad:** `codegen/decomp/transcendental.py` (`TRANSCENDENTAL_DTYPES`,
  float16 included), `cody_waite_reduction` (by half turns, whose float32 parts
  are the first three of Sleef's split for FMA and the fourth of its other
  split), `sin_poly`, `sin_poly_small` and `sin_poly_large`, `xsin`, and
  `xlog2`.
- **tolk:** `Transcendental.transcendental_dtypes`, `Transcendental.radians`,
  `Transcendental.cody_waite`, `Transcendental.sin_kernel`,
  `Transcendental.cos_kernel`, `Transcendental.xsin` and
  `Transcendental.xlog2`.
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
- **Pinned by:** the Transcendental suite
  (`test/codegen/decomp/transcendental`): `accuracy › accuracy.golden`, which
  holds `exp2`, `log2` and `sin` of float16, bfloat16, float32 and float64
  values correctly rounded by mpmath
  (`test/gen/codegen/decomp/transcendental.py`) and checks each function, and
  `xsin ~fast` below 30, within an ulp of them; `special values › sin <type>
  -0x0p+0 is -0x0p+0` and the subnormal rows; `types › <function> refuses a node
  of another type › half`; the graph goldens and `values.tolk`; and rune's
  `Lower_real › transcendental › exp within 2 ulps at float32, 9 at float64`, `›
  exp2 within 4 ulps …`, `› log2 within 4 ulps …`, `› sin within 2 ulps …` and
  `› cos within 2 ulps …`, on the host.

## D127. Withdrawn

A selection moved into a converted gated load only when the load's type held its
other value, since tinygrad converted that value to the load's type and back.
tinygrad no longer moves a selection through a conversion
(`codegen/late/gater.py` `move_where_load`, `pm_move_gates_from_index`):
`Gater.move_where_load` moves it only into a load of the selection's own type,
where converting the other value changes nothing, so `Gater.holds` went.

## D128. A committed float operation reads a weak float operand rounded

- **tinygrad:** `uop/ops.py` (`UOp._min_max`), which bounds a weak float
  operand of a committed float operation by its value as written, at no
  width.
- **tolk:** `Ops.rounded` and `Ops.operand_bounds`.
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
(`Worker.iter`), on every device, as `codegen/opt/search.py` `beam_search`'s
`pool.imap_unordered` does; on one domain, as without a pool, in their order.

## D132. A contiguous view's rewrite stops at the effects its storage waits on

- **tinygrad:** `uop/ops.py` (`UOp.contiguous_view`), whose
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

- **tinygrad:** `uop/ops.py` (`UOp.gcd`, whose coefficient is
  `uops[0].const_like(...)`, a constant broadcast to the shape of the first
  term) and `uop/divandmod.py` `fold_divmod_general` (`divide_by_gcd`, which
  divides every term by a divisor that is not the `CONST` 1). Over terms with a
  shape, a vector of index expressions or a weak-integer tensor, a divisor of 1
  is a broadcast 1, which is not the `CONST` 1 and divides no term:
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

A widening cast of an unsigned `x land y`, or `x lsr k` with `k` within the
operand's width, was the operation on the widened operands, so that nx.quant's
host MXFP4 products unpacked their uint4 codes at uint32, which Clang vectorises
better on arm64. On fudge, kaun's decode bench rows show no gain beyond their
ranges (medians of 11, two interleaved runs each, with the rule then without):
`Quant/host/mxfp4-product-1-token` 0.91 and 0.84 ms against 0.68 and 0.88 ms,
`Quant/host/routed-product-1-tokens` 5.01 and 4.07 ms against 3.85 and 5.28 ms,
`Quant/host/routed-product-64-tokens` 130 and 122 ms against 114 and 145 ms. As
tinygrad's `uop/symbolic.py` `Shape.symbolic` keeps such a cast where it is.

## D135. Withdrawn

A Metal command buffer signalled through the device's signaler. rig's Metal
driver signals its own (D146).

## D137. Consumers of one domain share their loops

- **tinygrad:** `schedule/indexing.py` `run_rangeify` (new ranges for every
  stored value) and `run_rangeify` (a node its consumers index apart is stored);
  `schedule/rangeify.py` (`remove_bufferize`, which inlines such a store back
  into each consumer) and `split_store` (one kernel per store). Upstream calls
  scheduling several outputs into one kernel pending:
  `test/runtime/test_assign.py` `TestAssign.test_simple_assignment_multioutput`,
  `TestAssign.test_permuted_reduceop_multioutput_dual_use` and
  `TestAssign.test_permuted_reduceop_multioutput_dual_use_possible` skip "multi
  output not supported anymore", and `test/runtime/test_jit.py`
  `TestMultioutputJit` skips "Pending multioutput implementation #3607". Codegen
  already compiles a kernel of one end over a group of stores
  (`test/runtime/test_custom_kernel.py` `flip_contract_kernel`). For reductions,
  `schedule/indexing.py` (`run_rangeify`: each reduction takes new ranges for
  the axes it reduces); `codegen/__init__.py` (`merge_reduce_ends`, which
  already makes one loop of the reductions that close one range).
- **tolk:** for stores, `Indexing.region`, `Indexing.views`,
  `Indexing.waits_over`, `Indexing.share_loops` (its `crossed`),
  `Indexing.loops_of` and `Indexing.end_shared` (which
  `Indexing.bufferize_and_index` returns for each store of a group), with
  `Indexing.data_srcs`, `Indexing.create_bufferize_and_index_based_on_ranges`
  and `Rangeify.handle_after`, which take a group (`Ops.is_group`) as its
  stores and no state of a buffer, and `Rangeify.is_noop_after_dep`, which
  takes a group emptied of its stores as no effect; for reductions,
  `Indexing.reaches`, `Indexing.reduce_loops` and its use in
  `Indexing.assign_ranges`;
  `Codegen.merge_reduce_ends`.
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
  all. A group whose every store is dropped, as a store of a value into itself
  is, leaves the empty stack, which waits on nothing and makes no kernel,
  where tinygrad, whose stores end apart, drops a noop and the end of one.
  Outputs that share only loads stay apart; finished calls may still share a
  launch (D404).

  Reductions: a reduction whose output is indexed by the same ranges as an
  earlier one's in the walk, and that reduces axes of the same sizes, takes that
  reduction's reduce ranges when no reduction of the group reaches it, through
  anything. The walk meets a reduction before those it reads, so only a
  reduction already in the group can read it. The sources are then indexed
  alike: a node they share is one node, computed once per element of the loop,
  where tinygrad indexes it by each reduction's ranges, stores it, and inlines
  the store back into each loop. A group of a loop that several reductions close
  takes a buffer of shared memory for each (D101). Two ranges that reductions
  close merge in every reduction that closes them or not at all, as tinygrad
  merges ranges since df93a6cb8. A reduction can stop closing its group's
  ranges: a sum over a range its source does not read drops it
  (`pm_reduce_unparented`), so a sum over `i` and `k` of a source that reads `k`
  alone closes `k` alone, where a maximum beside it closes `k` within `i`'s
  loop. Both merge every end, a store's as a reduction's. tinygrad gives an end
  fresh ranges only when it closes the same ranges as another at a different
  depth; tolk gives them to every end that closes a range an earlier end closes,
  so each range has one end. No recorded kernel has such ends.
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
  costs kernel graphs up to 2% more words (tolk's `*/kernel_graph` rows). A
  group emptied of its stores, (b): `Rune.jit` of `Rune.jvp` of a scan whose
  step gives back its carry, whose carry and tangent each store a value into
  themselves through a shared sign mask; the group's end was a kernel of no
  storage, which no device runs, and the call was evaluated, which on a set of
  GPUs raises. For reductions, (d): rune's reverse mode. The cotangent of a
  value that operations broadcast sums one reduction per operation, each over
  the same cotangent computation: the gradient of an aperture's weights over a
  104 x 104 grid in its radius, which 28 operations broadcast, was one kernel of
  28 loops, each computing the whole cell, and took 647-672 ms compiled against
  93-171 ms eagerly on an M1 Max under load 11-15; one loop takes 30-31 ms.
  On fudge, rune's `Jit/grad-broadcast-radius-float64-104x104`, the gradient
  in a radius that 28 operations broadcast over a grid, takes 215 and 224 us
  as one loop against 383 and 358 us as 28 (medians of 11, two interleaved
  runs each).
- **Pinned by:** for stores, the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › outputs that share a computation` (every test, `slices
  of one product through two views run as two kernels`, `slices that also
  share a computation run as two kernels`, `outputs that read a product
  through one view run as one kernel` and `outputs that store their own
  storage back are no kernel` among them); its recorded
  `read_around_assign_kernels`, `crossed_pairs_kernels` and `kernel_counts`,
  and the Postrange suite's `where_max_multioutput`, `padto_where_multioutput`
  and `axes` cases. On rune: the
  `compile/special/betainccinv-grad-f32` bench row, and `rune jit › fully
  sharded` for the views and `rune scan › transformations › derivatives and
  maps over views are the fold's` for an emptied group (the old rune's `Rune
  nx.wide › kernels` went with its rewrite). For reductions, the Rangeify
  suite (`test/schedule/rangeify`):
  `get_kernel_graph › reductions that share a domain` (every test), and its
  law over generated programs, whose `sum k + max k` and `x - sum k` steps
  cover reductions that share a loop; its recorded `multireduce_*` and
  `ugly_reduceop_pairing` kernel graphs and `kernel_counts`; the Indexing
  suite's recorded `two_consumers` graph; the Postrange suite's
  `sum_and_max_grouped`, `ten_shared_sums_group` and
  `ten_shared_sums_group_unfit` cases, and the Codegen suite's
  `two_grouped_stores_local`, whose upstream axes count one loop for the two
  maximums. The Codegen suite's
  `reductions that share a range › a range two reductions close at two depths
  runs as two loops`. On rune: missing (went with the old rune's rewrite):
  its Packed suite's bit example of `a compiled call computes eager's values`,
  two views a bit apart whose sums share their loops; its Jit suite's
  `transformations › a compiled gradient of a parameter broadcast over a grid
  is its gradient`; and its bench row
  `reverse/jit-of-grad-clipped-disc-float64-104x104-host`.

## D138. A load's index is simplified without the loads in it assuming its gate

- **tinygrad:** `codegen/late/coalesce.py` (`simplify_valid_load`),
  which simplifies a gated index with `uop/symbolic.py`
  (`uop_given_valid`) down into the loads the index reads.
- **tolk:** `Coalesce.simplify_valid_load`.
- **Differs:** each load in the index stands as a variable of its bounds while
  the gate simplifies the arithmetic around it, and is put back after, so a load
  keeps its own index and gate. tinygrad substitutes the gate's bounds into the
  loads too. A load's index whose gate implies the outer gate, such as a pad's,
  loses that gate, and the load, which runs whatever the outer gate, reads
  outside its buffer: the index of a gather through a pad is read before the
  pad's first element. tinygrad's own `gated_given_valid` (`uop/symbolic.py`
  `gated_given_valid`) refuses an index that loads for the same reason.
- **Reason:** (b): `Nx.concatenate` of a buffer and an `Nx.take` under
  `Rune.jit`, as nested sampling's evidence concatenates its dead points and
  its live points in likelihood order: the compiled kernel read up to the
  buffer's length before the start of the indices and faulted.
- **Pinned by:** the Coalesce suite (`test/codegen/late/coalesce`):
  `indexing_simplify › a gather through a pad reads its indices only inside
  the pad`.

## D139. Construction checks no specification

- **tinygrad:** `uop/ops.py` (`UOpMetaClass.__call__`, which checks each
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

- **tinygrad:** `schedule/__init__.py` (`resolve_linear_call`, which
  binds a nameless scalar parameter only to a variable, and substitutes it into
  each kernel that reads it), `engine/realize.py` (`get_call_arg_uops`,
  which leaves out every variable, and `get_call_kernel_args`, which reads each
  scalar parameter by its name).
- **tolk:** `Schedule.scalar_argument`, `Schedule.pass_values` and
  `Schedule.resolve_linear_call`; `Realize.get_call_arg_uops` and
  `Realize.get_call_kernel_args`; `Tolk_engine.affine`.
- **Differs:** a precompiled call inside a loop may pass a scalar parameter of
  its body a weak integer expression of the loop's ranges, such as `r`, `2r + 1`
  or `n - 1 - r`. The schedule keeps it as the call's argument, no state. Each
  kernel of the body that reads the parameter keeps it, unnamed, in a slot past
  its call's arguments, bounded by its type, and its call passes the expression
  in that slot: `get_call_kernel_args` reads a nameless scalar parameter of a
  program from its call's slot, and `get_call_arg_uops` leaves out every scalar
  argument, an expression as well as a variable. The value reaches the program
  as any variable's does, converted to the parameter's type at the ABI (D28): a
  launch inside a loop of rig's program reads it from a hole of its parameters,
  the sum of the loops' trip words times constant factors (D90). The kernel's
  program depends on neither the range nor its trips, so that loops of any range
  and length compile one. tinygrad has no loop of calls, and its calls bind a
  scalar only to a variable, by name. Beam search samples positional scalars by
  slot and named variables by name. Its `link` callback takes a complete call,
  with the scalar samples passed in their slots and the launch sizes evaluated
  from the same samples; the candidate's compiled body keeps its parameters.
  `Tolk_engine.link_call` links that call. tinygrad's `codegen/opt/postrange.py`
  (`args_from_ast`) reads every scalar's `expr` and constructs only buffer
  arguments.
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

- **tinygrad:** `codegen/__init__.py` (`do_devectorize`, which splits every
  elementwise operation, load and store of several lanes into one per lane, on
  every target); `renderer/cstyle.py` `base_rewrite`, which writes every
  conversion as a C cast, for a vector a reinterpretation of its bits.
- **tolk:** `Renderer.t`'s `vector_lanes`; `Codegen.lanes_of`,
  `Codegen.in_order`, `Codegen.vector_operation`, `Codegen.lanes_at`,
  `Codegen.split_widths`, `Codegen.split_lanes`, `Codegen.distinct_lanes`,
  `Codegen.vector_dtypes`, `Codegen.do_devectorize`,
  `Codegen.address_lanes`, `Codegen.pm_vector_constants`,
  `Codegen.product_operands` and `Codegen.first_lane`;
  `Cstyle.mask_widths`, `Cstyle.clang_lanes`, `Cstyle.clang_vectors` and
  `Cstyle.clang_kernel`;
  `Decomp_dtype.emulates`; `Uop_weak.pm_commit_weak`'s cast over a stack.
- **Differs:** a renderer may compute on vectors (`Renderer.vector_lanes` past
  1; Clang's is 64). There an elementwise operation of several lanes on a type
  the target computes natively stays one operation on a vector of its lanes, on
  one axis, or past `vector_lanes` lanes vectors of that many and a last of the
  rest, the last `vector_lanes + 1` as two vectors where one lane would be left
  (D241), when each source holds them in order: a vector operation, or a view
  (reshapes, broadcasts, permutations, stacks) of constants (a literal), of
  loads of memory, of lanes of vectors and of vector operations, a load or a
  lane in as many lanes as read it (a shuffle), or a scalar that every lane
  reads, and when no two of its lanes read the same sources: a lane of a vector
  operation is its own, a lane of a stack its element. A source that holds lanes
  computed apart, or apart in registers, as an accumulator's lanes, would be put
  together lane by lane, and lanes that repeat are computed once each apart:
  such an operation computes each lane apart. Loads, stores, tensor-core
  products, calls, and operations on a weak type (an address) or an emulated one
  still compute each lane apart; an address is computed lane by lane through the
  vector operations it reads, so that accesses still merge. A sum fuses each
  lane of a vector product into its running sum, and computes its first lane,
  which it adds to nothing, as a product of scalars, so that no vector product
  is left; a selection by constant conditions picks each lane. Clang writes the
  vectors as `ext_vector_type` values: a conversion is
  `__builtin_convertvector`, square roots, truncations and multiply-adds are
  `__builtin_elementwise_*`, a scalar operand of a builtin or a selection is
  splat, and a vector of booleans is a mask of -1 and 0 lanes as wide as the
  values it compares, converted to the width of the masks or values it meets; a
  lane of a mask is read as a `_Bool`, and a mask of one-byte lanes, which C
  types as plain chars, as signed chars. A cast of a stack of weak constants
  commits them at the cast's type, as tinygrad commits the weak sources of a
  cast of an operation, so that the vector is of constants of that type, as each
  lane would be alone (`uop/weak.py` `pm_commit_weak`). Which lanes a host
  kernel takes is D188's.
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

  Any number of lanes up to 64 is one vector, which LLVM computes in the next
  power of two. Computed lane by lane, Clang compiles each lane's copy of a
  body, in a time that grows faster than the copies: jera's compiled
  Milstein march over 3 elements fuses into one kernel of 3 lanes, whose test
  compiled in 57 s cold, and in 14 s as one vector.

  A broadcast or a permutation of a vector operation's lanes is a shuffle, as
  a load in several lanes is. Computed apart, an operation that read one went
  lane by lane, and so did every operation after it, since each read lanes
  computed apart: a map of 3 lanes over a loop whose lanes stop apart selects
  each step's values by a lane's flag, a comparison of 3 lanes broadcast
  across the lane's values, and rune computes each float8 operation in float32
  between a decode and an encode. rune's test_iterate (`maps › compiled, lanes
  stop apart at every dtype`, seed `s1:a9af9f06d362c663`) compiled one such
  kernel of 36 lanes to 828 KB of C, which Clang took 2.4-4.5 s on, and to 38
  KB, 0.22-0.28 s, as vectors; the test ran in 16.8 s against 3.1 s. The
  compile bench's `lanes/held-fp8-36`, a step of that shape whose kernel is
  705 KB of C against 35 KB, takes 3.2-7.2 s against 0.49-1.04 s (fudge at
  load 24-31, three interleaved runs, the timing lock held per run).
  Elsewhere the rule turns kernels whose lanes were read apart into vectors
  read back lane by lane: GGUF's q8_0, q4_k and q6_k host products multiply
  128 code lanes by a scale broadcast from 4 lanes as one vector, which each
  lane's multiply-add then reads. On fudge at load 30-60, kaun's
  `Quant/host/q8_0-product-1-token` took 2.6-3.9 ms against 1.8-3.1 ms
  unpinned, and 1.2-5.7 ms against 1.8-4.5 ms on its E-cores (six
  interleaved runs each), and its kernel alone on one E-core 4.6-5.6 ms
  against 3.9-5.3 ms (four runs, minima of 15 rounds), the q4_k and q6_k rows
  overlapping as well; no range stands apart. The cascade cost 2-7 s of
  Clang per kernel at each first call, which outweighs a replay difference
  no range shows.
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
- **Pinned by:** the Cstyle suite's `sources › by default › clang_vector_*`, and
  `execution on the host › every kernel the interpreter runs writes what it
  computes`; the Codegen suite's `vectors in programs › a host program computes
  on vectors of its lanes` (every host case), `› a host program of 6 lanes
  computes one vector`, `› a host program of 48 lanes computes one vector`, `› a
  host program of 24 lanes computes one vector`, `› a host program of n lanes
  computes one vector`, `› a host program of 128 lanes computes vectors of 64`,
  `› a host program of 80 lanes computes vectors of 64 and 16`, `› a host
  program of 65 lanes computes vectors of 33 and 32`, `› a host operation of any
  lanes computes vectors of at most 64 and writes what its kernel writes` (every
  case), `› a host selection by a broadcast of computed lanes computes no lane
  apart` and `› a host sum of a product's unrolled lanes multiplies no vectors`
  (every case), `› a host selection of 8, 16 or 32 lanes by a comparison of
  integers takes each lane's arm` and `› a host selection of 3, 6, 12, 24 or 48
  lanes by a comparison of integers takes each lane's arm` (every case), `› a
  weak constant stored into four lanes of half on the host converts them` and
  `multiply-adds` (every test); `Tolk.Uop_weak › pm_commit_weak › a cast of a
  stack of weak constants commits them at its type, signalling NaN bits kept`;
  rune's `Lower › padded loads give eager's bits` (a complex128 signalling NaN
  fill); rune's `rune jit › classes › a uint4 sort along the last axis of every
  other row and column, flipped, gives eager's codes` and `› nx.quant's decodes
  compile`; rune's compile bench: `shaped-4k-3-lanes` and `lanes/held-fp8-36`.
  On rune, missing (went with the old rune's rewrite): its Rune.jit suite's
  `lanes`, `stated values › a window set into bits at an odd offset keeps the
  bits around it`, and lower_index's `quantised products › a product decodes a
  checkpoint's codes 16 at once on the host` (with D188) and `› a product over
  GGUF's Q6_K blocks converts each scale once`. The Codegen suite's
  `invalid_lanes_clang` and `invalid_lanes_int8_clang`, whose lanes of one
  constant compute once.

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

## D145. Withdrawn

Rangeify and `PADTO` divided an integer division by `where(valid, d, 1)` on
padding, whose loads read zeros, since x86 traps on a division by zero. D355
defines every division whose divisor's bounds reach 0, and the bounds of a load
on padding hold the zero it reads: a load of storage is bounded by its type,
and a gated load of a constant table by its alternative too (D355). The padding
divides as tinygrad's interpreter does, without the guard. The Tolk_engine
suite's `a pad of integer divisions writes what its tensors compute` and `a pad
of a quotient by a constant table writes what it computes` (on the host and
each GPU) and the Codegen suite's `padto_idiv_upcast` keep the values.

## D146. The runtime is rig: no command queues, batches or host batches

- **tinygrad:** `runtime/support/hcq2.py` (`hcq_compile`, `sched_batches`,
  the queue encoders it calls through `Compiled.pm_encode`, the host
  programs that submit them, the staging copies and `hcq_link`);
  `runtime/ops_*.py` (`pm_encode` and the queues);
  `engine/realize.py` (`compile_linear`, which ends in `hcq_compile`)
  and `run_linear`.
- **tolk:** `Realize.compile_linear`; `Tolk_engine.link`, `Tolk_engine.run`
  and `Tolk_engine.time`; `test/gen/engine/jit.py`,
  which stops tinygrad's `compile_linear` before `hcq_compile`.
- **Differs:** `compile_linear` ends once the kernels are compiled: it batches
  no call, encodes no queue, makes no host program, splits no call into lanes
  and stages no copy. `Tolk_engine.link` turns the compiled calls into a program
  of rig (`Rig_program`): a kernel is a launch on each lane's device, a host
  program's on its host, its parameters laid out as `hcq2.py`'s `layout_args`
  lays them out; a copy is a copy part of a device that copies between its two
  sides, or else a move, a queued copy of rig's (`Rig.Buffer.submit_copy`); a
  range or a back edge is a loop; and consecutive calls of one device are one
  submission. rig's drivers encode each vendor's queues, order the work across
  queues and devices by the memory it uses, stage what no device copies
  directly, and time the submissions where the driver can. A host program's
  entry is rig's (D28). Storage the schedule makes is the program's, or an input
  each run makes where it is linked per run (D173), and storage a caller binds
  is an input of each run. Each linked program loads its own images
  (`Rig.Image.load`), where `CUDADevice.function` (`runtime/ops_cuda.py`
  `CUDADevice.function`) loads a binary once per device and process: two
  programs that run one kernel load its binary twice, since `Rig.Image.load`
  loads each call's binary again, so that each image's module variables are its
  own, and a program shares an image among its own launches alone
  (`Tolk_engine.link`). Sharing a binary without module variables across
  programs would save 1-1.5 ms of a 70 ms gpt-oss-20b first call on CUDA, 2.5 ms
  of 60 ms on NV, and 3.3-4.2 ms (1-4%) of its prefill and first step on an
  R9700, against a driver function and an image table (about 80 lines).
- **Reason:** (c). rig owns each GPU's queues: two runtimes cannot share a
  GPU, and memory belongs to the runtime that made it.
- **Pinned by:** the Tolk_engine suite: `link and run › recorded`, `GPUs`
  (each recorded program, loop and refusal on Metal, CUDA, NV and AMD where
  the machine has one), `lanes and words` and `runs`; the jit goldens
  (`test/engine/jit/*_lowered.golden`), which `gen/engine/jit.py` makes
  without batches.

## D147. The x86 host holds bfloat16 as its bits

- **tinygrad:** `renderer/cstyle.py` (`ClangRenderer.type_map`,
  which keeps `CStyleLanguage.type_map`'s bfloat16, Clang's `__bf16`
  (`CStyleLanguage.type_map`), on every machine).
- **tolk:** `Cstyle.clang` (the Clang renderer's language,
  which on x86 maps bfloat16 to `unsigned short` and renders its constants as
  their bits with `pm_bf16_ushort_const`).
- **Differs:** on x86, a bfloat16 value, pointer or vector is written as
  `unsigned short`, and a bfloat16 constant as its bits, as OpenCL writes them
  (`renderer/cstyle.py` `OpenCLRenderer.type_map`,
  `OpenCLRenderer.string_rewrite`: its type map's `ushort` and
  `pm_bf16_ushort_const`). The kernel's graph is tinygrad's; only its C types
  change. Every bfloat16 operation already computes in float32 through
  `create_non_native_float_pats` and `pm_manual_bf16_cast`, so a bfloat16 value
  is only loaded, stored, selected and bitcast, which act on its bits.
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

## D148. A selection keeps its guard of a conversion

- **tinygrad:** `uop/symbolic.py` (`fold_where_closure`, which
  assumes a selection's condition true in its true branch and false in the
  other, unless an `INDEX` is involved).
- **tolk:** `Shape.guards_partial` and `Shape.fold_where_closure`.
- **Differs:** the condition is not assumed in the branches where it reaches
  the operand of a conversion of a float to an integer in one of them.
  tinygrad folds `where c (cast (where c q 0)) f` to `where c (cast q) f`. A
  guard of an integer divisor, `where c (x // where c d 1) 0`, folds as
  tinygrad folds it: D355 defines the division by any divisor whose bounds
  reach 0.
- **Reason:** (b). C leaves a conversion of a float outside the integer's
  range undefined. Clang computes an operation in every lane of a vector and
  selects between vectors with masks, which keep an undefined lane the
  selection discards; LLVM then folds it into any value. rune guards such an
  operand, as the quadrant of a sine's reduction, by a selection on the
  condition that discards it, and the fold removed the guard: the compiled
  sine of `[0; inf]` on x86 read `0x1.c9e8dep+97` for `sin 0`.
- **Pinned by:** the `Symbolic` suite's `selections › a guard of a conversion
  to an integer is kept in the branches`. On rune: missing (the old rune's `a
  compiled program › computes eager's values within the stated rounding`,
  whose third example was that sine, went with its rewrite).

## D149. Withdrawn

The Clang renderer wrote no calling convention on Windows, where tinygrad wrote
`__attribute__((ms_abi))` on its kernels and the function pointers its calls
cast to. tinygrad no longer supports Windows: `ClangRenderer`
(`renderer/cstyle.py` `ClangRenderer`) keeps the empty `abi` and plain
`kernel_typedef` of `CStyleLanguage` (`CStyleLanguage.abi`,
`CStyleLanguage.kernel_typedef`) on every system, as `Cstyle.clang_lang` does. A
Windows host's code is compiled for Windows' convention by D178.

## D150. The folder's sine, logarithm and powers are correctly rounded

- **tinygrad:** `uop/ops.py` (`python_alu`'s `SIN`, `math.sin`;
  `LOG2`, `math.log2`; `EXP2`, `safe_exp2`'s `2 ** x`; `POW`, `safe_pow`'s
  `pow`: the C library's), and `uop/symbolic.py` `symbolic_simple` (`c ** x` as
  `exp2 (x * math.log2 c)`).
- **tolk:** `Ops.sine`, `Ops.py_log2`, `Ops.py_exp2` and `Ops.py_pow`, which
  `python_alu` calls, and the `c ** x` rule of `Shape.symbolic_simple`, which
  takes `log2 c` from `Ops.exec_alu`; `test/gen/generate.py`, whose child
  computes these folds correctly rounded with mpmath, so the goldens do not
  depend on the host's C library.
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

- **tinygrad:** `renderer/cstyle.py` (`kernel_typedef`, `void`) and
  `ClangRenderer.__init__`.
- **tolk:** `Cstyle.clang` (the Clang renderer's language on
  x86, whose `kernel_typedef` is
  `__attribute__((min_vector_width(512))) void`).
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

- **tinygrad:** `helpers.py` (`diskcache_clear`, `diskcache_get`,
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

## D154. A maximum over a range whose size may be 0 is its identity

- **tinygrad:** `uop/symbolic.py` (`reduce_mul_chain`, an integer
  maximum of a product times the factor that ignores its ranges).
- **tolk:** `Symbolic.reduce_mul_chain`.
- **Differs:** an integer maximum of a product by a factor that ignores its
  ranges, over a range whose size may be 0, is its identity there:
  tinygrad's is the identity times the factor, which wraps. The entry's other
  part, the linearizer counting a range as at least one run, went when
  tinygrad dropped run counts from its order (cf335e26b): a loop's body sorts
  after the nodes around it, whatever its size.
- **Reason:** (c). `Tolk_engine.run` binds a variable to any value of its
  range, its lower bound included, and runs each range's trips; a size of 0
  is a binding the schedule must compute. With concrete shapes the scheduler
  folds an empty value first (`Prepare`), so only a variable reaches it.
- **Pinned by:** `Tolk.Codegen › reductions over a variable size`, which runs
  sums and maxima over sizes of at most 1 and 3 on the host at every size,
  and an int32 maximum of `2 k` among them.

## D155. Withdrawn

A pad by `-0.` filled its padding with `-0.`; merged into D52, which holds its
rule, its reason and its pins.

## D156. Withdrawn

A NaN constant was written from its bits; merged into D27, which holds its
rule, its reason and its pins.

## D157. A C-style expression nests at most 64 brackets

- **tinygrad:** `renderer/cstyle.py` (`_render`'s rule that writes a
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
  cannot use. No golden nests past 24 levels, so none changes.
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

- **tinygrad:** `schedule/rangeify.py` (`remove_bufferize`), which
  inlines a stage's value at a read and leaves each range whose index is
  `Invalid` in place.
- **tolk:** `Rangeify.remove_bufferize`.
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

- **tinygrad:** `engine/jit.py` (`jit_lower`), which reads `JITBEAM`,
  or else `BEAM`, at `jit_lower` and passes the width to `compile_linear` as its
  `beam`, binding no setting; its search cache keys on no setting
  (`codegen/opt/search.py` `beam_search`).
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

## D160. A read no order keeps reads a copy of its state

- **tinygrad:** `schedule/__init__.py` (`create_schedule`), which adds a
  write-after-read edge from a kernel to every other kernel writing over a state
  it reads, and raises "cycle detected in assign graph" where those edges and
  the reads after writes form a cycle; `schedule/rangeify.py`
  (`check_buf_states`), which raises "cycle detected while indexing" where one
  kernel reads a storage in two states. Repro: `a = Tensor([1., 2.,
  3.]).contiguous().realize(); b = a * 2; a.assign(a + 1); (b + (a *
  3).contiguous()).tolist()` raises, and so does a swap `a, b = b * 1, a * 1`
  through assigns; `test/runtime/test_assign.py::
  TestAssignOrdering::test_war_reader_already_depends_on_write` states the
  raise, with the values `[2, 4, 15]` "once that read is staged" as its TODO. A
  call given one buffer at two arguments, writing one, is ordered against no
  other kernel and races with itself. Two AFTERs over the state each names first
  raise "sibling AFTERs detected in assign graph" (`create_schedule`,
  c00c9ada9).
- **tolk:** `Schedule.schedule`, `Schedule.cuts`, `Schedule.copy_call`,
  `Schedule.copy_before_overwrites` and `Schedule.aliases`;
  `Schedule.written_over`; `Rangeify.stand_in`; `Shape.written_params` and
  `Shape.custom_kernel`.
- **Differs:** one relation orders every kernel of the graph, a call of a
  kernel, of a copy, of a precompiled body or a custom kernel, and a loop of a
  call. A kernel reads the states its arguments name, a join being the state
  under it (D202), and writes the states the Afters that list it are over, but
  for a kernel already on an earlier state of that storage, which orders the
  state and writes nothing over it, as tinygrad's (`create_schedule`,
  3bdafaedf). It runs after the kernels that write the states it reads, and what
  the states under them wait for, as tinygrad orders a state after the state
  under it (413ecebe6): a kernel listed after a state its own arguments do not
  reach, as `first` in `x.after(second).after(first)`, would otherwise let a
  reader run before `second`. It runs before the kernels of every write over
  them of which it is no part (D330), whichever elements each touches. Where
  these orders form a cycle, the cycle is cut: one read on it reads a copy of
  its whole state, which a kernel of its own makes into call-local storage
  before the state's overwrite, and the schedule is made again from the graph
  with the copy. Kahn's order finds the cycle where no kernel is ready, walking
  back through the kernels not run; the read cut is one whose state is copied
  already, else the one of fewest bytes, of storage that can be copied first.
  Fewest copies is NP-hard (Sethi, by Rideau, Serpette and Leroy, "Tilting at
  windmills with Coq", §1); the walk is greedy. A cycle in the graph made again
  raises as an internal error. A graph whose kernels form no cycle gets no copy;
  grouping stores into one kernel can make one, so the promise is of the kernel
  graph `create_schedule` is given. Swaps, rotations and crossed reads compute,
  with a copy for each cycle. A kernel reading one storage in two states takes a
  parameter for each, and lies on a cycle, so no uncut kernel binds one buffer
  at two parameters. A call given one state at an argument its body writes and
  at one it does not reads, as `memmove` reads its source (C11 §7.24.2.2), a
  copy of the state made before it at the argument it only reads
  (`Shape.written_params`, at any depth through the calls the body makes); in a
  loop, the copy is made in the loop as each trip starts. A custom kernel orders
  after it only the arguments its body writes. A write's prior state is the
  latest state of the After's storage its calls are given, else the state the
  After is after: rune's loops name their carry's storage after the loop and
  pass the carry's first state: beside `a = x.after(x.store(1))`,
  `x.after(a.store(2))` writes over `a`, where tinygrad raises on two AFTERs
  named after `x`. Two writes over one state, which no storage holds, raise
  `Invalid_argument` naming the storage, and so does a call whose body does not
  show its writes given one state twice. A partial overwrite of elements a
  kernel does not read copies too, at a kernel and the state's size in storage.
  The relation costs allocation in every schedule, cycle or not: an edge list, 7
  words an edge, beside the order's own children, and tables of the kernels met
  and of the joins' waits. Measured in exact words on fudge, `matmul/schedule`
  allocates 780 words where it allocated 613, and `gpt_oss_decode/schedule`
  about a third more (15.4 thousand against 11.6); the `*/kernel_graph` rows are
  unchanged within 1%. Waiting on the state under a state and listing each
  kernel per storage (413ecebe6, 3bdafaedf) add 3-7% to these rows:
  `matmul/schedule` 707 words against 684. Cheaper shapes were tried and kept:
  the order runs first and the cut's parent tables are built only where it finds
  a cycle, and a state is resolved without allocating; with parent and child
  tables built for every graph, `matmul/schedule` allocated 909 words.
- **Reason:** (e). tinygrad raises on valid graphs (the swap, the stale read
  beside its overwrite), and computes wrong values where it orders silently:
  a call given one buffer twice, and the join cases of D202. rune's jit
  promises eager's values, and a program refused on a set of GPUs, which
  have no eager kernels, raises. No raven suite schedules a cycle today:
  jera's compiled gradient of an adaptive solve made the one copy they reach
  (`Ode › solve derivatives › compiled grad equals eager grad`).
- **Pinned by:** the `Tolk.Schedule` suite: `create_schedule › laws › a schedule
  copies a state iff its kernels form a cycle` and `create_linear_with_vars ›
  laws › a schedule writes what its tensors compute`; `create_linear_with_vars ›
  orders and copies`, `› writes` and `› calls given one state twice`;
  `create_schedule › rules › kernels that overwrite each other's inputs read one
  copy`, `› a kernel that waits for the overwrite of a state it reads reads it`
  and the rules beside it, for a copy two kernels share, a kernel with two
  copies, a copy of a state a rebuilt kernel writes and the copy's phase; the
  programs `read_overwritten_through`, `read_beside_overwrite`,
  `swap_through_values`, `call_given_one_state_twice`, `after_older_write`,
  `readers_after_older_write`, `after_older_and_new_write` and
  `write_over_written`, whose recorded schedules and linears and whose values
  (`create_linear_with_vars › values`); the kernel graph of
  `after_part_of_older_write`, whose schedule and linear differ by D184; and the
  `Tolk.Rangeify` suite's `get_kernel_graph › buffer states`.

## D161. Withdrawn

A folded weak constant wrapped at its operation's width, and a power of two was
read at the width of the operation that reads it (`Shape.wrap_folded`,
`Decomp_op.at_width`). An integer literal now holds the value of the type that
reads it at construction (D413), so the fold and the power of two read it as
written; the pins move there.

## D162. Withdrawn

A divisor past its operand's width was read in its operand's type
(`Decomp_op.divisor`). An integer literal now holds that value at construction
(D413); the pins move there.

## D163. An inlined value computes at most 360 nodes again

- **tinygrad:** `schedule/rangeify.py` (`remove_bufferize`, which
  inlines a stored value that reads at most three buffers and no reduction
  reading a buffer, at every position and every trip it is read at) and
  `get_kernel_graph`; `schedule/indexing.py`
  (`run_rangeify`, which stores a value its consumers index differently, or
  one a reduction or an elementwise operation reads past a broadcast, the
  stores `remove_bufferize` weighs).
- **tolk:** `Rangeify.extent`, `Rangeify.read_ranges` and
  `Rangeify.read_copies` (the copies a read makes of a stage's value), with
  `Rangeify.digit_of` (a range read through floor divisions and remainders);
  `Rangeify.collapses`, `Rangeify.selected` and `Rangeify.collapsed` (a sum
  that selects one trip, asked only where the answer changes a stage's
  decision), and `Rangeify.storage` (a stage of no storage read in such a
  sum), through `Simplify.load_collapse`, which
  `Simplify.pm_load_collapse` applies; `Rangeify.reads`,
  `Rangeify.stage_reads`, `Rangeify.reduced`, `Rangeify.stage_copies` and
  `Rangeify.stage_bounds` (the copies of each stage);
  `Rangeify.inline_budget` and `Rangeify.copied`;
  `Rangeify.inlined`; `copies` and `computes` in
  `Rangeify.remove_bufferize`; `Rangeify.destination`, which
  `Rangeify.pm_remove_bufferize` applies to a store's destination;
  `Rangeify.pm_cleanups` and `Rangeify.get_kernel_graph`, which counts the
  reads before the cleanups; across kernels, `Rangeify.store_shared`,
  `Rangeify.candidates`, `Rangeify.kept`, `Rangeify.computed_nodes`,
  `Rangeify.union` and `Rangeify.is_kernel`.
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

  A view of storage, a read whose index computes only positions, is
  inlined at any count, as tinygrad inlines it: stored, it would copy
  storage its readers load in place, saving no work. Positions are weak
  integers and their comparisons, on no device. An elementwise operation, a
  reduction or a call on a device, as a position loaded from storage,
  or of another type computes; so does any value that is not a read, a
  mask of positions included, and each is weighed as above. A stage a store
  writes through is inlined whatever it costs, past the budget and past
  tinygrad's own checks too (more than three buffers, a reduction reading
  one): kept, it would take the stores, and the storage would hold what it
  held. A user's materialisation is kept, and written as asked.
  47 stores of one state into the rows of a `[2; 47; 9]` buffer read its
  view at 47 indices, 8 copied nodes 46 times past the first, 368 in all,
  and wrote a copy of it.

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
  computes the value again more often than counted.
  On a GPU the codegen owes that sharing, and states it: D230's bound of
  `ceil(R / 16)` decodes a value. On the host it shares a decode over as
  many rows as its registers hold (D92).

  A sum over one range that selects one trip, where an index equals the
  range, as a routed product's rows select their expert's matrix, counts
  that range as the ranges its selecting conditions read besides it, from
  which the selected index is computed: the kernel computes it at that index
  alone (`Simplify.load_collapse`), once for each point of those, so a row
  decodes only its own expert's weights, `p / e` times a weight for `p` rows
  over `e` experts, where the trips count `p`. A factor over another range
  inside the sum, `where (ids (j) = e) (v (e, d) * y (m)) 0` summed over
  `e`, broadcasts the selected value along `m` as it would a read of `v` at
  `ids (j)`, which counts `m` neither, and the codegen shares it as it
  shares a broadcast; a sum with no such condition counts every range it
  runs in. The test
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
  its axes, since stages over one range share that axis. Indexing gives
  each stage a later pass may remove ranges new to it, except on an axis
  whose size is a range, which that range indexes: a stage with any other
  range has one of its own, where its reads are counted alone, and the least
  is its own count. A stage over such axes alone, as one value of each of a
  device's lanes, shares them and pools its reads; traced with each stage's
  own reads beside the pooled count, the Rangeify suite, rune's (its
  multi-device programs among them) and jera's decide every stage alike, so
  no program shows a wrong store, and hand-built graphs whose stages share
  their only range pool. A read
  the cleanups make is not counted, and a read through a stack is not one:
  every stage has a device by then, so no cleanup turns it into a read of
  the stage.
- **Reason:** (e), `Rangeify.destination`: kept, a stage a store writes
  through takes the stores, and the storage holds what it held. (d) the rest.
  rune's compiled `Nx.associative_scan`, whose prefix circuit reads each
  level's pairs and results at two or three positions: jera's tridiagonal
  solve of 100 equations (`scan/tridiagonal-100`), three scans one of them
  over 4-tuples, compiles cold in 1.25 CPU-s and 34.4 M words against 2.08 s
  and 100.3 M with no budget, and at 1000 equations in 5.3 s and 45 M words
  against 33.6 s and 780 M, 94.5 s and 1.09 G with the across-kernels store
  off too (fudge, 2026-11-08, load 25 to 60). norn's
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
  (D230), 64 rows take 4.98-6.68 ms, under the stored decode's 9.6. On
  the host, whose tiles then shared 4 rows of a digit, an inlined decode ran
  about 4 times a value per 16-row block and 10 to 16 times in a dense
  product, and lost to the stored decode; its tile now shares the rows its
  registers hold (D92), with the scheduler kept target-free.
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
- **Pinned by:** the Rangeify suite's rows of `kernel_counts.tolk` and their
  kernel goldens: across kernels, `scalar_read_by_two` (a value of no range of
  some forty nodes read by two kernels, computed in each),
  `long_scalar_read_by_two` (of some four hundred, stored) and `scalar_chain`
  (40 values of no range, each read by a kernel of its own and the next, one
  stored); `read_at_offsets` (a value of some forty nodes read at 32 offsets,
  stored), `windowed_chain` (in a running sum of 300, stored), `windowed_pair`
  (in two running sums of 16, stored), `windowed_scaled` (a product in a running
  sum of 16, its copies past the first 90 nodes, inlined),
  `windowed_scaled_long` (a product in a running sum of 400, inlined into its
  windows of 16), `read_by_two` (some forty nodes read by two kernels, inlined),
  `long_read_by_two` (some four hundred, stored), `routed_product_2` (a decoded
  matrix gathered by 2 rows an expert over 64 experts, inlined),
  `routed_product_64` (64 rows an expert, stored) and `routed_product_wide` (16
  rows an expert in a product's sum of 192, the gather's positions inlined),
  `selected_scaled` (a matrix of some 160 nodes selected by a sum that
  multiplies a factor over 4 columns inside it, read once a row, inlined),
  `blocked_product_16` (rows in blocks of 16 reading their block's matrix of
  some forty nodes at `r // 16`, inlined) and `dense_decoded_64` (64 rows merged
  with a matrix's 4 columns, read at `r % 4`, inlined), `tiled_chain` (tiled 256
  times, inlined) and `padded_pair` (read through two paddings to 1024,
  inlined); `cost › levels each read twice make kernels that stop growing with
  depth`; the `copies` group: `a value of 1200 nodes read at one index is
  inlined`, `› a value read twice whose second copy is the budget is inlined`,
  `› a value read twice whose second copy passes the budget is stored`, `› a
  value of 600 nodes outside its ranges read twice is inlined` and `› reads of a
  stage sharing a range are not the other's`; the `recomputation` group: `a
  value of 1200 nodes reduced once an element is inlined`, `› a cheap value
  reduced in windows is inlined`, `› a value reduced in windows whose work is
  under the budget is inlined`, `› a value reduced in windows whose work passes
  the budget is stored`, `› a value of no storage reduced in windows past the
  budget is stored`, `› two windowed reads of a stage add up`, `› a value of
  symbolic size counts at its largest`, `› a value reduced in windows counts no
  axis it does not vary along`, `› a read counts at most the trips of its
  reductions`, `› a value reduced through padding counts the padding's trips`,
  `› a value inlined into a windowed stage counts in its copies`, `› a value
  broadcast through a reshape is inlined`, `› a value read through padding is
  inlined`, `› a view of storage read past the budget is inlined`, `› a gather
  at loaded positions read past the budget is stored`, `› a mask of positions
  read past the budget is stored` and `› stores through a stage of their
  destination write the destination`; the Launch suite's `launches › on the
  host, the stores of one state write each piece's values` and `› on CUDA, the
  stores of one state write each piece's values`, up to 160 stores; `work bound
  › a stage is stored when its copies beyond the first pass the budget`, over
  values read once, twice, in windows and by routed rows; the `dropped digits`
  group: `a dropped digit's reuse is no copy`, over quotients, remainders, a
  middle digit and a chain inlined, and windows, a reduction range's digit, a
  gather, frames, digit sums, a range read by two components, constants not
  dividing and a symbolic size stored, `› a read of merged outputs decides as
  one of them split` and `› a digit a selection runs in stays counted`;
  `selections › a read a sum selects decides as one at the selected index`,
  which a factor over another range inside the sum made 4 copies, stored past
  120 nodes; `own ranges › a removable stage has a range no other stage has`;
  rune's `Lower_reduce › work › a Rows contraction's work is one slice's over
  256 contracted` and `› a scaled float4 Rows contraction's work is one slice's
  over 256`; rune's `rune jit › classes › an associative scan over a tuple gives
  eager's bits` and `› a value stored or computed again in its readers gives
  eager's bits`, which reads the value in blocks of rows and in a product of
  merged rows too; on fudge, the `scan/tridiagonal-100` row of rune's compile
  bench, the `Jit/cumsum-neg-log-uniform-8x1024`,
  `Jit/experts-mxfp4-16x4of32-256` and `Cuda/sin-read-by-two-float64-1000` rows
  of its bench, jera's `root-newton-kepler-1k/compiled` and
  `root-bracket-kepler-1k/compiled`, gpt-oss's `GptOss/prefill-512` and
  `GptOss/experts-at-prefill-512`, and kaun's
  `Quant/cuda/mxfp4-product-of-16384-*-tokens` and
  `Quant/host/mxfp4-product-of-1024-*-tokens`. Across kernels, the Rangeify
  suite's `get_kernel_graph › values that several kernels compute` (every test:
  the budget's two sides for two and three kernels, a weak value, a value read
  through a load the kernels share, a value cheap above a stored one, and the
  chain of outputs whose work is linear in its length and whose kernels stop
  growing with it), and ymir's `aperture-104/compiled-grad` row.

## D164. Withdrawn

The read of an accumulator after its loop waited on the initial store as well as
the loop's end, since `sym` folded a store of a loaded accumulator to a bare
`NOOP` and the loop could end holding no store, so nothing ordered the read
after the initial store. At `uop/symbolic.py` `sym` a folded store of a load of
its own index is `NOOP.after(index)`, and at `sym` one of an invalid value
`NOOP.after(*x.src)`: each keeps the accumulator's index, ordered after the
initial store, before the read, and `codegen/__init__.py` (`lower_reduce`)
orders the read after the end alone, as `Codegen.lower_reduce` does. The
`Codegen` suite's `an accumulator no lane adds to › is stored before it is read`
and `› sums to zero`, and rune's `Lower_index › folds and assemblies › a fold
wraps integers and holds its float bound`, hold.

## D165. A weak maximum's identity is its accumulator's least value

- **tinygrad:** `uop/ops.py` (`identity_element`), whose `Ops.MAX` identity is
  `dt.min`: for `weakint`, of bitsize 800 (`dtype.py` `DTypes.weakint`),
  `-2**799`.
- **tolk:** `Ops.identity_element`.
- **Differs:** a maximum over a weak type has as identity the least value of
  `Dtype.strong` of the type, the width its accumulator is allocated at. In
  tinygrad a weak integer's maximum, gated by an Invalid, takes `-2**799`:
  `lift_reduce_gate` (`uop/symbolic.py` `lift_reduce_gate`) selects it where the
  gate fails, and committing the selection's width (`cast_weak_srcs`,
  `commit_int`) raises "does not fit any int". A weak maximum arises where a
  pad's bare fill is all that remains of a committed maximum's operand.
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

- **tinygrad:** `codegen/decomp/op.py` (`get_late_rewrite_patterns`),
  which has no rule for a nest of selections.
- **tolk:** `Decomp_op.extreme_of`, `Decomp_op.nested_extreme`,
  `Decomp_op.unwrapped`, `Decomp_op.holds`, `Decomp_op.keeps` and
  `Decomp_op.late_patterns`.
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

## D167. Clang's arguments: LLVM 18, 19 or 20 and Apple's Clang 17 for x86_64 without its straight-line vectorizer, a long source without its loop vectorizer, arm64 without the interleaved load combine, every kernel without loop idiom recognition

- **tinygrad:** `runtime/support/compiler_cpu.py` (`ClangCompiler.compile`,
  Clang's arguments at `-O2`, which runs LLVM's SLP and loop vectorizers).
- **tolk:** `Compiler_cpu.stated_release` (the release a Clang's `-###`
  statement names, LLVM's or Apple's), `Compiler_cpu.long_source`,
  `Compiler_cpu.compile_args`, which adds `-fno-slp-vectorize` when it is
  LLVM 18, 19 or 20, or Apple's 17 for x86_64,
  `-mllvm -disable-interleaved-load-combine` for arm64 and
  `-mllvm -disable-loop-idiom-all` for every Clang,
  `Compiler_cpu.without_loop_vectorizer`, and `Compiler_cpu.clang`
  (its `table`, which names both commands and the length);
  `Compiler_amd.hip`, whose code generation's options end with
  `-mllvm -disable-loop-idiom-all`.
- **Differs:** a Clang that states LLVM 18, 19 or 20 compiles every kernel
  with `-fno-slp-vectorize`, after `-O2`, which would turn the vectorizer
  back on, and so does Apple's Clang 17 for x86_64.
  Every Clang compiles a source of more than 400 000 bytes without the loop
  vectorizer: its compiler proper runs without `-vectorize-loops`, as the
  driver runs it under `-fno-vectorize` (Clang 18.1.8 and 19.1.7 state the
  same command but for that flag). Every Clang compiles for arm64 without
  LLVM's interleaved load combine. Every Clang compiles without LLVM's loop
  idiom recognition (`-mllvm -disable-loop-idiom-all`, which every LLVM
  release from 12 defines), and so does comgr's code generation for AMD
  GPUs. Otherwise every Clang, Apple's included,
  compiles with the arguments tinygrad passes. The table of objects names
  the command for either length, and the length (D113).
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

  Clang 19 has the same defect without the crash: rune's `Lower_reduce`
  test `rows in blocks give eager's integers and hold their bound` failed on
  fudge (Clang 19.1.7, Intel Core Ultra 5 235) at one seed of three, five
  float32 products of 96 reading `0` where `0x1p-4` was expected, and so did
  `Nx.matmul_segments` and `Nx_quant.apply_experts` under `vmap` and `jit`,
  rows of one expert in one block of 64 at each of two batch indices. The
  kernel `E_16_2_4` searches each slot's row by halving, four slots and two
  batch indices a thread, its bounds selected longs and their midpoint an
  int. For `x86-64-v3`, `x86-64-v4`, `arrowlake` or `znver4` (not
  `x86-64-v2` or arm64's `apple-m1`), the vectorizer packs four lanes'
  bounds, narrows them to 8 bits, and writes `trunc <4 x i8> ... to <4 x
  i32>`, invalid IR that a release Clang does not verify and compiles into
  garbage rows (202116109); `opt-19 -passes=verify` rejects it, and
  `-opt-bisect-limit` names `SLPVectorizerPass` as the pass that breaks the
  kernel. `llvm-reduce` brings it to 40 instructions of selects, truncations
  to i32, `or`, `lshr` and `zext`. Without the vectorizer the kernel gives
  every row. With the IR verifier on (D275), invalid IR from it is a compile
  error naming the kernel.

  Apple's Clangs state their own release, not LLVM's. Apple clang 17.0.0
  (clang-1700.6.4.2, macOS 26.3) writes the same invalid IR for `E_16_2_4`
  for `x86-64-v3`, `x86-64-v4` and `znver4` (`trunc nsw <4 x i8> ... to <4 x
  i32>` for the kernel rune renders, `<2 x i8>` for the reduced one), and
  compiles it without the vectorizer: it compiles for x86_64 on an Intel
  Mac and for another machine's x86-64 host. For arm64 it compiles both
  kernels, rune's `Lower_reduce` and attention cases above pass on an M1
  Max, and of the 195 kernels of those attention cases none converts a
  vector of 8 or 16-bit lanes to float, where Homebrew's Clang 19.1.7
  writes `uitofp <2 x i8>` in one of them for arm64 and another for
  `x86-64-v3`. For arm64 the vectorizer stays on: of 466 kernels of tolk's
  heuristic suite and those rune cases, Apple Clang 17 changes 269 objects
  without it and adds 9.5% instructions (247 691 to 271 190), up to 5
  times in one kernel.

  LLVM 19 and 20 also write valid IR of wrong values, which no verifier
  sees: rune's attention test `jit within the bounds of eager › float32`
  failed on the Windows runner, `lse` off by 4249261054.8, at tq 2, tk 3,
  dk 1 under a captured mask and ALiBi, and so did its compiled gradient.
  The host kernel converts two int64 differences of row and key, `q - 1`
  and `q - 2`, to float32 from their 32-bit halves, `sitofp` of the high
  half times 2^32 plus `uitofp` of the low half (`Lower_arith.odd64`).
  For `x86-64-v3` (not `x86-64-v2`) the vectorizer packs the two low halves,
  narrows them to 8 bits, since the high halves need only their sign, and
  writes `uitofp <2 x i8>`: a low half of 2^32 - 1 converts to 255, and the
  score is off by 2^32 times its slope (`-opt-bisect-limit` names
  `SLPVectorizerPass`). Clang 18.1.3, 21.1.0 and 22.1.8 compile the kernel
  with the vectorizer; 19.1.7 and 20.1.2 compile it wrong with it and right
  without it. With the verifier on, 20.1.2, 21.1.0 and 22.1.8 compile
  `E_16_2_4` for `x86-64-v3`, which 19.1.7 rejects.

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

  (b), the interleaved load combine: rune's `Lower_reduce › work › a
  Contracting contraction of 8 groups works in blocks` timed out at 60 s on
  macOS CI (Apple Clang 17, LLVM 19) after D240's upcasts, where it had
  taken 2.3-5.7 s, and ran 60-164 s. Its kernel `r_23_4_16_4_16` upcasts its
  outputs by 16 and 4 and unrolls its 16 rows, each row loaded at an
  index read from memory: each iteration multiplies two vectors of 1024
  lanes put together by shuffles of 32 loaded rows. Clang 19.1.7 for
  `apple-m1` took 293 s on it, 274 s of them in `InterleavedLoadCombinePass`
  (`-ftime-report`), an AArch64 pass that looks for loads that shuffles put
  together into interleaved vectors, to read them with one `ldN`; without it
  8 s, against 5.1 s for `x86-64-v3`, which has no such pass. The kernel
  unrolled by 4 instead (256 lanes) took 6.2 s and 0.27 s. Of the 1241
  distinct kernels the rest of that suite compiles, rendered on fudge and
  compiled for `apple-m1`, the objects are the same byte for byte without
  the pass, and their compiles take the same time (40.7 s and 40.8 s in
  all).

  (b), loop idiom recognition: rune's `Nx.eigh` under `Rune.jit` of a
  float64 matrix of 24 rows took 8.5 s to compile under Clang 23, and of 32
  rows did not finish in minutes (SOFO's subspace solve, which runs it on
  the host instead), against 0.3 s under Clang 19. The kernel is
  `Lower_linalg.exponent`'s, the largest magnitude of `where(lower, a,
  transpose a)`: a loop of 32 trips, each folding 32 lanes into a chain of
  float maxima `(m < x) ? x : m`, each reading the last twice. From LLVM 22
  the pass runs its CRC recogniser (`HashRecognize`) on every countable
  loop; for an innermost loop of one block whose constant trip count is a
  multiple of 8, it walks the operands of the loop's running selection
  without marking those it has seen, 2^31 paths here. LLVM 23.1.5 spent
  99.9% of 14 s in the pass at 24 lanes and did not finish at 32 in 300 s;
  without it, 0.01 s. The 20-row kernel (20 trips) and those of 64 and 128
  rows (chains of 4 lanes) compile in under 0.15 s. svd's exponent of its
  triangle escapes only because its masked loads render as branches, which
  give its loop 45 blocks: loaded unconditionally, its 24-row kernel took
  LLVM 23.1.5 15.35 s. Upstream
  fixed the walk on main (llvm/llvm-project#211711, #211887) after the 23
  branch; `release/22.x` and `release/23.x` do not have it. The pass has
  nothing to do in a kernel: under `-ffreestanding` Clang marks its
  functions `"no-builtins"`, so LLVM offers it no `memset` or `memcpy` to
  call, its other idioms need a loop that exits on its data, which no
  kernel producer writes today, and no lowering renders a CRC. The objects
  are the same byte for byte without it: of 8695 and 10 213 kernels of
  rune's linear algebra and suites under Clang 19.1.7 for x86_64, of 8695
  under Clang 18.1.3 and 8694 under LLVM 23.1.5, where it finishes, and of
  1167 for arm64 and 389 for riscv64 under Clang 19.1.7. Every Clang
  compiles without it, Apple's included, since Apple's release is not
  LLVM's: Apple clang 17 is based mainly on LLVM 19 and Apple clang 21
  (Xcode 26.4) on LLVM 21, with no Apple 18 to 20, and Swift's `main` builds
  on swiftlang's `stable/23.x`, which has the walk, while the CRC recogniser's own option is an error before LLVM
  22. A gate to LLVM 22 and 23 would differ only from LLVM 24, where a
  future superlinear addition to the pass would show silently, as a
  compile that never ends; the blanket option instead takes a rename of the hidden
  option, which would fail every host compile with Clang's message at the
  call that compiles. The item is revisited when a kernel loop exits on its
  data (`Cstyle` renders a void range as `for (;;)`), and goes when tolk
  supports no Clang of LLVM 22 or 23 and Apple's bases are past 23.

  comgr runs the pass too, when it generates code, under the same
  reasoning. ROCm 7.0.0's comgr, AMD's LLVM 20.0.0git, has no CRC
  recogniser. Its pass turns a loop counting a word's bits into `s_bcnt1`,
  and no kernel producer writes such a loop today: on nonnormal's R9700,
  with tolk's and comgr's caches off, the 1384 code objects that rune's GPU
  suite (three runs) and `Rune.jit` of `Nx.eigh` on float64 matrices of 8,
  24 and 32 rows compile for gfx1201 are the same byte for byte without
  it. A ROCm built on LLVM 22 or 23 would carry the walk, so comgr compiles
  without the pass, from every release.
- **Pinned by:** `Tolk.Compiler_cpu › features › four clamped gathers compile
  for AVX-512 on` (`x86_64,x86-64-v4`, `x86_64,skylake-avx512`,
  `x86_64,znver4`), which compiles the indices of `r_3_2` and fails under
  Clang 18 without the flag; `Tolk.Compiler_cpu › features › a binary
  search of four lanes compiles for AVX2 and AVX-512 on` (`x86_64,x86-64-v3`,
  `x86_64,x86-64-v4`, `x86_64,znver4`), over `E_16_2_4` reduced, which fails
  under Clang 19 and Apple's Clang 17 without the flag; rune's `rune jit ›
  classes › a uint4 sort along
  the last axis of every other row and column, flipped, gives eager's
  codes`, under Clang 18 on an x86 host with AVX-512; rune's `Lower_reduce ›
  cases › rows of one group in a block at each batch index give eager's
  values`, under Clang 19 on an x86 host with AVX2; rune's `rune attention
  › jit within the bounds of eager › float32` and `› the gradient compiled
  and in forward mode › float32`, whose first example fails under Clang 20
  on an x86 host with AVX2. The loop vectorizer:
  `Tolk.Compiler_cpu › features › a loop over a count it takes is packed
  into vectors iff its source is at most 400000 bytes long:` (rows 400000 and
  400001 among them); rune's compile bench `special/betainccinv-grad-f32`, the float32
  gradient of test_special (914 KB), whose first call takes 3.88 and 4.48 s
  on fudge against 6.57 and 6.66 s with the loop vectorizer (medians of 3,
  two interleaved runs each). The interleaved load combine: `Tolk.Compiler_cpu
  › compile time › an arm64 kernel of 1024 lanes gathered from 32 rows
  compiles within a minute`, over that kernel's source, which times out at
  60 s on fudge with the pass and compiles in 5.4 s without it; rune's
  `Lower_reduce` test above, on macOS CI. Loop idiom recognition:
  `Tolk.Compiler_cpu › features › a loop that exits on its data stays a
  loop: idioms are off`, a population count by clearing bits, which Clang
  19 turns into `popcnt` with the pass; `Tolk.Compiler_cpu › compile time › a host maximum of 32 float64 lanes in
  a loop of 32 trips compiles within a minute for` (`x86_64,x86-64-v3`,
  `arm64,apple-m1`), over `r_32_32`'s source; rune's `Lower_linalg › eigh ›
  compiled for 32 rows of float64 gives eager's eigenvalues`; rune's
  compile bench `linalg/eigh-f64-32x32`. These three fail without the fix
  only under a Clang of LLVM 22 or 23. comgr: `Tolk.Compiler_amd › comgr ›
  a loop that exits on its data stays a loop: idioms are off`, which ROCm
  7.0.0's comgr turns into `s_bcnt1` without the option.

## D168. Withdrawn

The renderer listed a program's callees by a scan of its instructions, in
place of tinygrad's toposort of the program, for 7 to 13% of render
allocation. On fudge that gain is 2% of render words and 0.4% of compile
instructions: it did not pay for the scan.

## D169. A scheduler reads its kernel's axes once, and keeps a flat kernel

- **tinygrad:** `codegen/opt/postrange.py` (`Scheduler.__init__`, which rewrites
  its kernel by `pm_flatten_range`), `Scheduler.rngs` (and `full_shape`,
  properties computed at each access), `copy` (which makes a scheduler of the
  kernel, rewriting it again), `axes_of` (which computes `reduce_ranges` and
  `rngs`, itself computing `reduce_ranges`) and `reduceops`.
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
  queries; `› the scheduler's axes after optimising are as recorded`;
  `› Scheduler ›` `v's kernel has the ranges its reductions and ends
  close flattened`, `v's kernel of a recorded kernel is it flattened` and `a
  copy's kernel after optimising is the kernel flattened`; the tolk bench's
  `search/actions/*` rows on fudge, which time making a search's candidates,
  and its `*/codegen` rows.

## D170. A kernel's parameters name no device

- **tinygrad:** `schedule/rangeify.py` (`debuf`) gives each kernel parameter its
  buffer's device, and `codegen/__init__.py` (`to_program_key`) keys a program
  by `ast.key`, so each device makes its own program of one kernel. Copy
  detection and the check that a kernel spans one device read the kernel's
  parameters (`schedule/__init__.py` `assert_all_same_devices`,
  `copy_kernel_to_store`, `simplify_copy_kernel`), the renderer is picked from
  the call's device, which a call takes from its body (`engine/realize.py`
  `_get_call_to_compile`), and the search names its parameters' device itself
  (`codegen/opt/search.py` `_try_compile`).
- **tolk:** `Rangeify.debuf`; `Schedule.is_copy`, `copy_runs`,
  `assert_all_same_devices` and `copy_call`; `Codegen.program_key`
  (`Codegen.params`); `Realize.get_call_to_compile`; `Search.timed_call`;
  `Spec.program`.
- **Differs:** a kernel's parameters name no device: the device is its
  call's arguments', where a kernel launches (`engine/realize.py` `exec_kernel`
  launches there too). Kernels of one expression on several devices of one
  target are one node, one key and one program, in memory and on disk. Copy
  detection and the one-device check read the call's arguments, the renderer
  is the target's of the call's first storage argument, where the engine
  launches the kernel, and `Spec.program` refuses a
  parameter that names a device. A search's timed call names its target's
  device, and its caller places it on a device of the target (rune's
  `search`). `test/gen/graph.py`
  (`unplaced_kernels`) writes a kernel body's parameters without a device, so
  the goldens carry this whatever tinygrad computes.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.make`, `Pieces.split`) lower each device's operations on that
  device, so a program over `n` memory devices of one host makes `n` copies
  of each kernel. With the devices in the key, a data-parallel step's first
  call over 16 memory devices took 0.92 s on fudge, 52 ms per device past
  the first; with one program per target, 0.24 s. A device kept in the
  kernel and erased for the key held one fact in two places, kept equal by
  renaming each program to its call's devices.
- **Pinned by:** `Tolk.Schedule › create_linear_with_vars › rules › one
  expression on two devices of a target is one kernel`, `… › a kernel on
  buffers of two devices is refused` and `copies into runs`; `Tolk.Search ›
  a search kept for one device serves another of its target`; `Tolk.Spec ›
  program › parameters`; rune's `Rune.jit over several devices`, `searches`
  among them.


## D171. Withdrawn

A store's check for reading its destination walked only the nodes after the
destination in the graph's order. The check is gone: a store that reads its
destination is ordered across kernels by D160's relation and staged within
its kernel by D198's rules. rune's `devices/dp-step-16` row keeps guarding
the walk of `Rangeify.fix_store_hazard`.

## D172. An index of an Invalid value folds after construction

- **tinygrad:** `uop/ops.py` (`UOp.index`, which returns
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
  `Tolk.Schedule › create_schedule › recorded › select_no_lane_schedule.golden`,
  `select_one_lane_schedule.golden`, `select_no_lane_rows_schedule.golden`; and
  `Tolk_engine › link and run › recorded › select_no_lane writes what its
  tensors compute` and its `select_one_lane` and `select_no_lane_rows`
  versions, which run the kernels on the host.

## D173. A schedule linked per run makes its arenas at each run

- **tinygrad:** `engine/jit.py` (`CapturedJit`): the linked linear of a
  captured schedule holds its intermediates' buffers from its first call
  until `free_intermediates` drops it, so they outlive every run, and
  `engine/jit.py` (`jit_lower`) places them in arenas
  (`memory_plan_rewrite`).
- **tolk:** `Tolk_engine.link`'s `storage`, `Tolk_engine.per_run_storage`,
  `Tolk_engine.run_with`.
- **Differs:** a schedule linked `Per_run` is planned into arenas (D41), in
  the order by storage (D183, D185), and makes each storage node no buffer
  is bound to before its run's first call and frees it after its last, its
  memory coming back once the work on it is done. A run is one program of
  rig's, which loads each image once per device. A run's submissions are
  those of the same schedule linked `Linked`: the order by storage
  interleaves devices, 1411 `Submit` steps for a data-parallel step over 16
  memory devices where tinygrad's order makes 84, and under rig's one point
  per submission they are what cross-device overlap costs: grouping a
  device's work into fewer submissions lengthens a GPU-faithful critical
  path 8 to 11 times. Linked `Linked`, the default, its storage is the
  program's, as tinygrad's. A run of either keeps none of the buffers it
  was given once it returns. In a set of a GPU and a host-run device, where
  the plan places host work over bytes a GPU copy touched, the run waits
  for the GPU's earlier work before it returns (D41: 168-200 ms against
  13 ms): the plan keeps the bound and gives up that call's asynchrony.
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
  rules › a copy's buffer and a call's of one device share bytes once their
  lives are apart` and the plan law; rune's `fully sharded`
  tests, whose shares are exact after a step, and `Rune.jit on a GPU ›
  memory › a split program holds at its return at most its result and its
  plan`.

## D174. A kernel's call bodies are functions of its own source on every target

- **tinygrad:** `codegen/__init__.py` (`LowerCallsCtx`, `lower_call`), which
  lowers a call's body to a program of its own (`to_program`, marked
  `KernelInfo.kernel=False`, so its arguments stay parameters) unless the
  renderer has an `inline_prefix`, and `renderer/cstyle.py` (`render`), which
  declares such a program `extern` and calls it with its arguments;
  `engine/realize.py` (`get_runtime`), which links it at load time by name
  (`runtime/support/elf.py`, `link_syms`). Only HIP has an `inline_prefix`
  (`renderer/cstyle.py` `HIPRenderer.inline_prefix`).
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

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`,
  without `TC_MIN_GLOBALS`: after the core, M and N are upcast and N split
  into local threads, by the first size that divides them, whatever the
  workgroups left; the same at `6b7b018da`).
- **tolk:** `Heuristic.core_threads`, `Renderer.t.local_workgroups` and
  `Heuristic.tensor_cores`.
- **Differs:** each upcast applies only while the kernel keeps at least
  `core_threads / Tc.threads` workgroups after it, the product of its global
  axes, or when that product is not known: 16384 threads, 512 workgroups of a
  CUDA, RDNA or Metal core of 32 threads and 256 of a CDNA core of 64. An
  upcast leaves fewer threads to keep memory and the L2 busy (concurrency is
  latency times throughput, Volkov, UCB/EECS-2016-143 §3.1). The local split
  keeps the threads and leaves fewer workgroups to share among the compute
  units, so threads do not govern it: it takes the largest of 4 and 2 that
  divides N and leaves at least `local_workgroups` = 256 workgroups, or an
  unknown count, where tinygrad takes the first that divides. 16384 is the bound
  of sm_89's parts, and every target takes it until one measures another;
  then `core_threads` becomes a field of `Renderer.t` set per target. Under
  `TC_MIN_GLOBALS` the splits are tinygrad's.
  A core whose operands are staged through shared memory takes neither
  these splits nor their floors but a tile of its own (D262), with the
  floor `core_threads / stages`.
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
  - the local split as tinygrad's, unfloored: a decode step's attention values
    1.19x slower, the key projection 1.34x warm and 1.6-1.7x cold, the routed
    products 1.14x, for 1-4% elsewhere. Its cost follows the warps on the
    busiest SM, a function of the part's units: a split is harmless where it
    does not raise them, `ceil(W / (L U)) L <= ceil(W / U)` for `W` one-warp
    workgroups, a split `L` and `U` units, which needs `U`, a fact of the
    part. 256 workgroups with the fall back to 2 is the floor fudge's rows
    admit (between 1.44 and 2.56 times its 100 SMs): it takes a split by 2
    where 512 refused every split, on fudge, three interleaved rounds, against
    512 with D211: the key projection 88-115 to 78-96 us (its bench row 89.1
    to 73.0 us at best), the output projection 186-195 to 169-186 us,
    `matmul_half_512` 17.4 to 16.3 us at best, the decode step's keys
    unchanged, the prefill 167.5-181.1 to 168.3-169.0 ms; the local split by 4
    to 128 workgroups that cost the key projection 1.34x stays refused. At
    2048 cubed the floor of 512 left one warp a workgroup, and 256 takes a
    split by 4 (195-219 to 199-203 us). On parts of 24 to 142 SMs the floor
    is a fit, not the test: unmeasured;
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
  Since staging (D262) the staged tile's rule of `core_threads / stages`
  replaces these floors for a staged core, so `stream/key-projection` and
  `stream/dense-bf16-64` no longer reach them (29.8 against 29.5 us, four
  interleaved rounds, and 433-436 us both on the row's kernel as rune compiles
  it today). gpt-oss-20b's routed experts take the
  unstaged path and reach the floors, which refuse nothing there:
  `GptOss/experts-at-prefill-512` 2.77 against 2.80 ms and
  `GptOss/prefill-512` 122.0 against 121.2 ms with and without them. The row
  that still gains is `stream/decode-keys`, 7.4 against 7.8 us without the floors, 1.05x
  for about 15 lines.
- **Pinned by:** `Tolk.Heuristic`'s cases, on CUDA: `matmul_half_1024`,
  `matmul_half_512`, `matmul_half_256`, `attention_values` and
  `matmul_half_variable` (a variable counts the workgroups), with the
  layouts in the suite's `.tolk` files; its `the tensor
  core's floor › the splits after a tensor core are those that leave its
  floor of workgroups`, over every recorded core kernel, CDNA's included,
  with the local split's floor of 256 workgroups; the `routed_run` cases of
  D200 on CUDA, whose local split by 2 the floor of 256 admits; the tolk
  bench's `stream/decode-keys` row on fudge.

## D176. Withdrawn

A folded selection or negation kept a NaN's bits; merged into D27, which holds
its rule, its reason and its pins.

## D177. A name bound to several values is passed by each call that reads it

- **tinygrad:** `schedule/__init__.py` (`create_linear_with_vars`), which raises
  "bind mismatch" when a schedule's kernels read one variable name bound to two
  values, and `resolve_linear_call`, which substitutes a bound variable into its
  kernels unbound, read by name from the run's `var_vals`; `function.py`
  (`is_implicit_storage`, `pm_ctx`), where a function traced into a call
  captures each variable its body reads as an argument, and
  `schedule/__init__.py` (`pm_replace_buf`), which passes the schedule's own the
  same way.
- **tolk:** `Schedule.create_linear_with_vars` (`bound_several`, and its
  `used_vars`, which reads the kernels of loops),
  `Schedule.canonicalize_call_body` and `Schedule.pm_pass_bound`,
  `Schedule.resolve_linear_call` and `Schedule.pass_values`;
  `Tolk_engine.var_word`, and `Realize.call_value`, through which the grid
  and split size of `Tolk_engine.launch` read a nameless scalar from its
  call's slot.
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

  A call passes the variables its body binds, as the schedule passes its
  own, by the same rule (`pm_pass_bound`), where the schedule gives each
  call's call-local storage its slots: each becomes a nameless parameter of
  the call in a slot past its arguments, one per value, and the call passes
  the bound variable there. tinygrad's function does the same as it traces
  a body into a call; tolk takes a call's body as built (`Ops.call`), so the
  schedule does it, once, for every builder. No body the schedule splits
  into kernels then binds a name: stores of one kernel (D137) and calls
  run as one launch (D404) read each value from a slot of their own. A kernel
  the body calls, such as a custom kernel whose code binds a variable,
  reads it from the body's slots as the kernels split from the body do
  (`Schedule.pm_pass_kernel_bound`), which resolving the scope fills
  (`Schedule.resolve_linear_call`). A name of one value that a loop's
  kernels read is among the schedule's variables, which tinygrad, without
  loops in a schedule, reads from its top-level kernels alone.
- **Reason:** (b): rune's compiled programs over several devices
  (`Pieces.split`, `Compile.moving`) bind each device's window start, of a
  move or of a loop's coordinates, to a variable named by its bound, so that
  every device's kernel is one program (D170) and the data-parallel step
  takes as many programs over 2 to 16 memory devices. A body's variables,
  (b): rune's loops (`Loop.cut`) keep the variables their steps bind in the
  loop's body, such as the start of each piece of an assembly
  (`Compile.region`), which binds one name to each piece's start; named in
  the body, two stores or calls of a trip that one kernel joined passed the
  name one value.
- **Pinned by:** `Tolk.Schedule › create_linear_with_vars › rules › two
  variables of one name bound to two values are passed by the calls that read
  them`, `› kernels that read two names bound to several values alike are one
  program, whatever order the schedule met the values in` and `› loops of
  calls › loops whose calls read one name at two values each pass their own`
  and `› a loop inside a loop's body passes the values of one name it binds
  apart`; `› merged calls › calls that bind one name to two values run as
  one`, `› loop bodies that differ in their bound values alone are one` and
  its law `calls and stores of a loop that bind one name to several values
  join, and no kernel binds a name to two`; `Tolk_engine › link and run › a
  scan's calls that read one name at two values carry each its own`, `› a
  scan's stores that read one name at two values store each its own` and
  `› a scan's custom kernel that closes over a bound variable runs as one
  with the scan's own and reads its value`, and `› custom kernels with
  variables › a custom kernel that closes over a bound variable reads its
  value`, and the same on each GPU;
  `Tolk_engine › link and run › one program launched on
  several devices computes each device's window at its start` and `› kernels
  whose sizes are one name of two values run each its own`, and the same on
  each GPU; rune's `rune jit › jit › two assemblies of alike pieces compile`,
  `Rune.jit over several devices › maps over a cut axis › per-device code
  takes as many programs over 2 to 16`, `› programs › a data-parallel step
  takes as many over 2, 4, 8 and 16`, `› loops › a step that reads its
  coordinates gives eager's bits`.

## D178. A Windows host's code is compiled for Windows' calling convention

- **tinygrad:** `runtime/ops_cpu.py` `CPUDevice.__init__` (the CPU's arch, its
  machine and `native`); `runtime/support/compiler_cpu.py`
  (`ClangCompiler.compile`, which compiles for `<machine>-none-unknown-elf` on
  every system); `renderer/cstyle.py` `base_rewrite` (an `int64` literal ends in
  `l`), `CStyleLanguage.abi` (`""`, which `ClangRenderer` at `ClangRenderer`
  keeps) and `CStyleLanguage.type_map` (where an `int64` is a `long`).
- **tolk:** `Tolk_engine.target`, whose host arch on this machine under Windows
  names the system after its machine, as in
  `x86_64-windows,x86-64-v3,tune=native` (or
  `arm64-windows,generic,tune=native`);
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
  absence breaks. `Tolk_engine › target › the host compiles C for Clang at
  its arch, tuned for its own processor, naming its system under Windows,
  whose functions follow a convention of their own`. On windows-2025 CI, `CG › calls in a
  kernel`.

## D179. An index of a reshape's added leading axes is its source

- **tinygrad:** `schedule/prepare.py` (`_mop_index`), whose shortcut for
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

- **tinygrad:** `schedule/indexing.py` (`pm_fix_deviceless`) and
  `run_rangeify`, which place every stage of no device on the
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

## D181. Withdrawn

A tensor core's addend was broadcast before tinygrad's rules moved it past a
permute, where a one-step core's sum plus its scalar initial value raised. No
rule moves an addend: tolk folds a core's running sum where it builds the sum,
and an initial value is added after it (D260).

## D182. A copy or a store into a run of storage lands in it

### A copy stored into a run of storage lands in place

- **tinygrad:** `schedule/prepare.py` `earliest_rewrites` (a copy stored into
  storage on its device is the store, only where the destination has buffer
  identity) and `copy_to_anon_store` (any other copy lands in storage of its
  own); `schedule/__init__.py` (`pm_copy_kernel`, which lowers a copy kernel
  that reads and writes both buffers at its range alone).
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
  lowers the same way. No recorded program stores a copy into a part of storage.
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

### A store into a run of a buffer stores into a view of it (D184)

- **tinygrad:** `schedule/__init__.py` (`pm_callify_ctx_collect`):
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
  other storage, which the kernel split stages by the buffer's identity. A copy or a stage of a run of a buffer that a
  store of the call writes keeps reading the run itself, which the kernel
  split stages: a view, a parameter of its own in the call's body that
  nothing writes, would be read when its reader runs, in whatever state the
  buffer then holds, and not in the state it was taken of. A run may start
  at bound variables, whose values its call knows
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
  its bound allows and views keep. The refusal of a view of a buffer the
  call writes is (e): `x[0:2].contiguous()` read after `x.assign(...)` in
  one realize read the assigned `x` (tinygrad's
  `test/runtime/test_assign.py::test_swap_slices`, an expected failure, is
  that bug), at the cost of a stage kernel per such view.
  The program `after_part_of_older_write`, whose two stores go into one
  element each of one buffer, is recorded as a kernel graph alone: its linear
  here is one launch (D404) over two one-element views of the buffer, where
  tinygrad's indexes the whole buffer in a kernel per store.
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
  outputs is linear in their number` and `› views of written buffers › a
  stage of a run of a buffer the call overwrites reads it as it was` and
  `› halves of a buffer swapped through stages swap`, and the programs
  `stage_of_overwritten` and `copy_of_overwritten`; rune's `Rune.jit over
  several devices › programs` and `› maps over a cut axis › partials give
  eager's bits, their programs do not grow`, and `rune jit › fully sharded`.

## D183. In a schedule made per run, ready kernels run by their number, those that free storage first

`Schedule.storage_order` keeps both parts.

### In a schedule made per run, ready kernels run by their number, and one that frees storage first

- **tinygrad:** `schedule/__init__.py` (`create_schedule`'s linearize:
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
  of rune's five `fully sharded` cases over their bounds. Grouping a
  device's work at link instead keeps the bounds but lengthens a
  GPU-faithful critical path 8 times under rig's one point per submission.
- **Pinned by:** the Schedule suite (`test/schedule/schedule`):
  `create_schedule › order by storage › a kernel that frees storage runs
  before one that only makes it`, `› pieces of one value are each read
  before the next is made` and `› without it, ready kernels run as they
  become ready`; rune's `fully sharded` tests. Any other schedule keeps
  tinygrad's order, as every recorded schedule does.

### A ready kernel that only frees runs before one that also makes storage (D185)

- **tinygrad:** `schedule/__init__.py` (`create_schedule`'s linearize,
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

## D184. Withdrawn

A store into a run of a buffer stored into a view of it; merged into D182,
which holds its rule, its reason and its pins.

## D185. Withdrawn

A ready kernel that only frees ran before one that also makes storage; merged
into D183, which holds its rule, its reason and its pins.

## D186. The host's accumulators take half its vector registers

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`' upcasts
  of output axes: by 3 or 4, of an axis that some access does not index while it
  indexes every upcast axis, while 1024 elements remain and the kernel has fewer
  than 32 lanes) and `hand_coded_optimizations` (the unroll of the last reduce
  axis, whole when it is at most 32, by 4 otherwise, while the kernel has fewer
  than 64 lanes).
- **tolk:** `Heuristic.held_registers` (`held`, `held_share`,
  `computed_bytes`), `Heuristic_host.fits`, `Heuristic_host.stored_contiguous`,
  `Heuristic_host.upcast_contiguous`, and `Heuristic.upcast_more`'s and
  `Heuristic.unroll`'s use of them; `Renderer.registers`, which
  `Cstyle.clang` sets from `Compiler_cpu.registers`.
- **Differs:** on the host (the `CPU` device of a renderer that knows its
  vector registers: 16 of 32 bytes on x86-64 with AVX, 16 of 16 without, 32
  with AVX-512, 32 of 16 on arm64), a reduce kernel's lanes are counted in
  registers, a register holding as many lanes of a reduce's type as fit, at
  4 bytes for a narrower type, and no more than its stores write at
  consecutive addresses. Every reduce's accumulators, one for each output
  lane, take at most half the registers, judged before each split: a later
  split that nests a stored term, `(a * 4 + b) * 3 + c`, reads as lanes
  apart afterwards, so `held_registers` can then exceed the half
  (`r_128_16_3_4_4_3_3_400_400` on AVX2, 16 of 16). Reading through the nest
  admits more unrolls, and one rune kernel ran 1.7 times slower. The upcasts
  after the masked and shared ones (D92) and the
  unroll take the budget in place of tinygrad's lane counts:
  - the first output upcast takes the axis the stores write at unit
    stride, when every access that reads it reads it at unit stride, to a
    host vector of the accumulators' type (D188), the largest power of two
    that divides it and fits;
  - the upcasts by 3 or 4 follow while they fit, without tinygrad's 32
    lanes and 1024 elements (D240);
  - when none is left, an output axis not yet upcast is upcast by 2 if it
    fits and some access that indexes a serial reduce axis does not index it
    (the fill; among such axes, tinygrad's order: fewest accesses indexing
    the axis, then the least sum of their strides);
  - the unroll takes what fits, while the lanes upcast fill fewer registers
    than the target has, where tinygrad's take fewer than 64 lanes, 16
    registers of 4.
  Other devices, and a host renderer that knows no registers (riscv64's),
  upcast and unroll as tinygrad does, after D92's rows of a decoded operand,
  at most 4. The register file is the compile target's,
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
  registers of 4 floats.

  The budget counts every reduce the same: one accumulator for each output
  lane, summed over the kernel, with no notion of which loop is hottest. A
  model of which reduces share the hottest loop, and which unrolled lanes
  count, took 82 lines and changed 3 of 864 recorded kernels, none on a GPU;
  against it, lorenz's `r_128_16_3_4_4_3_3_400_400` runs 1.27-1.37 times
  slower and its `r_32_256_50_4_4_2_4_4` about 1.2 times faster (minima of
  six interleaved rounds, one core, 151 samples, `lorenz/*/heuristic`). The
  model refused to unroll one sum whole while another looped, measured at an
  earlier render as 7.92-8.32 ms against 3.31-3.42 on the two-sum kernel; at
  this render `r_32_256_50_4_4_2_4_4` takes that unroll and runs 15.8 ms
  against 18.9 (one core, one run). The
  rune host kernel that changes, a triangular solve's gradient in
  `Jit/gaussian-process-grad-float64-8x11`, ran 14.8-16.0 µs before and 11.0
  µs after (four interleaved rounds, 201 samples, one core); its row takes
  72.3-83.7 µs against 76.0-86.2 (eight interleaved runs).
- **Pinned by:** the Heuristic suite: `the host's budget › on AVX2 a
  kernel's accumulators hold` (every case), `the host's budget › the register
  budget › a kernel's accumulators take at most half the registers` and
  `› the kernels the budget is checked on have each of its shapes`, `the host's budget › on the host a larger register file never
  holds fewer output lanes` and `› on the host a reduce of doubles over a
  stored axis of a multiple of 32 takes half the lanes of floats, in the
  registers floats fill`, over every recorded kernel and lorenz_simple's,
  on files of 16 × 16, 16 × 32, 32 × 16 and 32 × 64; `the optimisations
  chosen are tinygrad's › … › the optimised kernel`, the `cpu` (16 × 16) and
  `avx2` (16 × 32) cases; the
  Compiler_cpu suite's `the vector registers, their number and bytes, of`
  (every case) and `without Clang › an x86_64 processor has SSE's 16
  registers of 16 bytes`; the tolk bench's `host/*/heuristic` and
  `lorenz/*/heuristic` rows on fudge.

## D187. The host's stores take a full vector register

- **tinygrad:** `codegen/late/coalesce.py` (`memory_coalescing`'s
  lengths: 4 or 2 elements of float32, float16, int32, uint32 and the 8-bit
  floats, 8 of float16 under `ALLOW_HALF8`, none of the others, bytes
  included; D261 makes
  that 8 the renderer's, for copies into local memory).
- **tolk:** `Coalesce.memory_coalescing`, through `Renderer.registers`.
- **Differs:** where the renderer knows its vector registers, loads and stores
  of float64, float32, float16, int32, uint32, the 8-bit floats, int8 and
  uint8 merge into accesses of a power of two of elements up to one register:
  8 floats, 4 doubles or 32 bytes in AVX2's 32-byte registers, 4 floats, 2
  doubles or 16 bytes on SSE and NEON. A store of
  doubles or bytes merges so only when each value is a load, as an
  accumulator's read is; doubles and bytes computed lane by lane are stored one
  by one, as tinygrad stores them. A group of byte loads any of which a
  computation reads alone is not merged, as tinygrad merges no bytes: LLVM's
  load combine folds an or of bytes in descending addresses, from vector
  loads, into a byte-swapped load at the address of the first byte's vector,
  the wrong one where that byte is not its vector's first, so a compiled
  running sum of a widening bitcast of flipped bytes read a byte off; Clang
  merges bytes that feed one lane each itself.
  On the CPU device a buffer is taken to start on a
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
  Bytes: a decoded product reads its codes as bytes, and an MXFP4 code's byte
  feeds two lanes, so its loads, one a byte, reach Clang as a vector built lane
  by lane, which it builds with one `vpinsrb` a lane: the 1-row MXFP4 product,
  4 of the matrix's rows by 16 values a trip, took 63 `vpinsrb` a trip, 3 once
  the 32 bytes it reads load 8 at a time, and 534 instructions against 644.
  Consecutive bytes that feed one lane each Clang merges itself, and GGUF's
  MXFP4 blocks, Q4_K and Q6_K kept their kernels' loads. On fudge, one P-core,
  interleaved, five rounds, kaun's `Quant/host/mxfp4-product-1-token` took
  9.89-10.45 ms and takes 6.48-6.81, the gpt-oss expert product at one token
  (`Quant/host/routed-product-1-tokens`) 43.8-44.9 ms and 32.1-33.4, and
  bfloat16 rows of 10 and 64
  (`Quant/host/mxfp4-product-of-1024-{10,64}-bfloat16-tokens`) 1.98-2.06 ms
  and 1.71-1.78, 8.94-9.33 ms and 8.68-9.02; the Q8_0, Q4_K, Q6_K and GGUF
  MXFP4 rows kept their times, and every row its 698 or 822 words a call.
  Only the loads were measured, so bytes computed lane by lane are stored one
  by one, as doubles are.
- **Pinned by:** `rune jit › classes › a running sum of a widening bitcast gives
  eager's bits`; the Coalesce suite's `lengths › thirty-two byte loads each read
  by a computation stay apart, on` and `› eight consecutive stores fill
  registers of` (every case), `› eight consecutive stores of lanes computed
  apart, on 32 bytes, are` (every case, bytes included), `› thirty-two
  consecutive loads of bytes fill registers of` (every case), `› without known
  registers, byte loads stay apart, on` (every case) and `› without vectors of
  eight halves or known registers, eight halves are two loads of four`, and its
  kernels' goldens for SSE's 16 bytes; the Stage suite's `decoded panels › on
  the host a decoded product loads its codes a register at a time` and `› on the
  host a decoded product computes the product of the decoded codes`; the tolk
  bench's `host/*/heuristic` rows and kaun's `Quant/host/mxfp4-product-1-token`
  and `Quant/host/routed-product-1-tokens` on fudge.

## D188. A host vector spans 64 bytes

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (the
  upcasts of output axes by 3 or 4), `hand_coded_optimizations` (the unroll of
  the last reduce axis, and of a second one when both are at most 3) and
  `hand_coded_optimizations` (the upcast by 4 of a kernel that has none).
- **tolk:** `Heuristic_host.host_vector_bytes`, `Heuristic_host.parallel_operations`,
  `Heuristic_host.upcast_vectors`, `Heuristic_host.upcast_contiguous` and
  `Heuristic_host.unroll_vector`.
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
  `experts_down`, on `cpu` and `avx2`; the tolk bench's `host/*/heuristic` rows
  and rune's compile bench `trig/sin-nested-8` on fudge. On rune: missing (the
  old rune's `lower_index › quantised products › a product decodes a
  checkpoint's codes 16 at once on the host` went with its rewrite).

## D189. A sum gated on an index read at its range is not collapsed

- **tinygrad:** `codegen/simplify.py` (`pm_reduce_load_collapse`'s
  rule for a sum of a value gated on an index equal to the range), which
  substitutes the index for the range in the value whatever the index
  depends on.
- **tolk:** `Simplify.pm_reduce_load_collapse`, through
  `Simplify.load_collapse` and `Rangeify.collapses`.
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

- **tinygrad:** `runtime/support/compiler_cpu.py` (`ClangCompiler`'s x86_64
  arguments, `-march=<cpu>` and its features); `runtime/ops_cpu.py`
  `CPUDevice.__init__` (the CPU's arch, its machine and `native`).
- **tolk:** `Compiler_cpu.flags` (`fixed` and `llvm18_x86_off`), which
  `Compiler_cpu.clang` and `Compiler_cpu.registers` take, with the
  release `Compiler_cpu.stated_release` reads from the statement
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

- **tinygrad:** `renderer/__init__.py` (`Estimates.from_uops`'s range
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

- **tinygrad:** `codegen/__init__.py` (`_is_local_store`,
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

## D193. Withdrawn

A string or bytes argument entered a key as its digest, for the profile key
`Device.Tiny_elf.of_program` computed over a program's source and binary. Its
one consumer was gone: nothing read that key, and the digest changed no
compile's words or instructions. `Ops.key` writes every argument as its repr,
as tinygrad does, and `of_program` and `Tiny_elf.t` went with it.

## D194. A running operation scans chunks of 16 at every level

- **tinygrad:** `mixin/op.py` (`_split_cumalu`: past 512 elements,
  chunks of 256, the first padded before, scanned by windows, then the
  chunks' last elements scanned by one window each).
- **tolk:** `Shape.split_cumalu` and `Shape.cumalu`.
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
- **Reason:** (e), the chunks aligned from element 0: a running operation's
  prefix is the prefix of the axis's, bit for bit, as `Nx.cumsum` promises;
  padded before, a float sum of a prefix rounds otherwise. (d), the chunks of
  16 at every level: compiled running sums, `Nx.cumsum` under `Rune.jit`, as
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
- **Pinned by:** rune's `Lower_reduce › work › a running sum's work per term at
  most doubles from 32 to 65536 terms` and `› scans › running scans along long
  axes give eager's values`; `rune jit › classes › a running operation's prefix
  is the prefix of its result`, on the host and, in `Rune.jit on a GPU`, on the
  GPU; the Rangeify suite's `long_cumsum`, `windowed_chain` and
  `windowed_scaled_long` rows of `kernel_counts.golden` and their goldens, and
  the Codegen and Heuristic suites' `cumsum` kernels; on fudge, rune's bench
  rows `Jit/cumsum-float64-100x50500`, `Jit/cumsum-scaled-float32-100x400`,
  `Jit/cumsum-neg-log-uniform-8x1024`, `Cuda/cumsum-float32-100x400`,
  `Cuda/cumsum-float32-8x1024` and `Cuda/cumsum-float32-100x50500`.

## D195. Clang reinterprets a vector's bits by a vector cast

- **tinygrad:** `renderer/cstyle.py` (`base_rewrite`'s `BITCAST`,
  `__builtin_bit_cast(T, (S)(x))` for every value).
- **tolk:** `Cstyle.clang_vectors`'s bitcast rule and `Cstyle.byte_mask`.
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
  clang_vector_bitcast` and `› clang_vector_byte_mask`, and every host golden
  with a vector bitcast; rune's compile bench row `special/betaincinv-grad-f64`,
  whose fudge baseline the scheduled re-record records.

## D196. A schedule's host kernels compile in one run of Clang per domain

- **tinygrad:** `engine/realize.py` (`lower_and_compile`, which makes
  each kernel's program, `to_program`, apart on its pool's workers);
  `codegen/__init__.py` (`do_compile`) and `device.py`
  (`Compiler.compile_cached`, one source at a time);
  `runtime/support/compiler_cpu.py` (`ClangCompiler.compile`, a `clang`
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
  Clang states for the options with `-###`, its IR verifier on, D275), each
  source the file `-` of a
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

- **tinygrad:** `uop/symbolic.py` `commutative` (commutative flipping, which
  orders the two operands of a commutative index operation by `tuplize`,
  a range's number among it); `schedule/rangeify.py`
  (`pm_renumber_range` in `split_store`, which renumbers a kernel's ranges
  from 0 after the symbolic rules ran over the whole graph, "so that kernel
  deduping works"); `codegen/__init__.py` (`to_program_key`, which
  keys a program by its kernel's `ast.key`).
- **tolk:** `Codegen.params`, which `Codegen.program_key`,
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

## D198. A store that reads its destination is staged only where it reads it elsewhere

- **tinygrad:** `schedule/indexing.py` (`realize_store_after_src`,
  which stores the value of every store whose value reaches its
  destination's buffer: "you don't usually have to do this for assign unless
  there's a WAR hazard like TestAssign.test_assign_double_diamond_reduce").
- **tolk:** no `realize_store_after_src`; `Rangeify.fix_store_hazard`
  decides within a kernel, and `Schedule.create_schedule` across kernels
  (D160).
- **Differs:** a store whose value reads its destination's storage is not
  staged for that alone. Within its kernel, whose iterations run in
  parallel, the value is staged where it reads the destination at another
  index than it writes (`fix_store_hazard`, as tinygrad); a store into a
  broadcast, which writes an element on several trips, is refused (D345).
  A value that reads its destination only at the element it writes, as an
  optimizer step `p := p - lr * g` or a loop's carry `c := c * 2 + exp2 c`
  does, loads each element before it stores it, a data dependency of the
  store: one kernel. Across kernels, another kernel reading the destination's old state
  runs before the store, or reads a copy where no order does (D160): the
  double diamond, each store reading through a sum what the other
  overwrites, is a cycle that one copy breaks.
- **Reason:** (d). Kernels per optimizer step drop with bit-equal values:
  in tinygrad, measured, an Adam step's 13 kernels to 7, an SGD step over a
  convolution's 5 to 3, a convolution, batch norm and Adam's 30 to 18; in
  tolk, rune's carries, which a loop's trip updates in place: NUTS on eight
  schools (`norn` bench `nuts/schools/4`), whose doubling cascade's trip ran
  17 kernels and runs 10, its main body's 189 and 182, and whose cold
  compile makes 145 programs against 147, with the same bits (fudge,
  counted). Measured on fudge in tinygrad only for the optimizer steps until
  vega's Adam row, rune's MLP step row and tolk's `adam_mlp` rows are
  re-recorded; the wall-clock baselines are pending the scheduled
  re-record.
- **Pinned by:** the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › stores that read their destination` (`a value that reads
  its destination at the element written is one kernel`, `› a value of five
  storages that reads its destination at the element written is one kernel`, `›
  a value that reads a slice of its destination at the element written is one
  kernel`, `› a value that reads its destination through a bitcast of one size
  is one kernel`, `› a value that reads its destination at another element is
  staged`, `› a value that reduces its destination reads the sums first` and `›
  a value whose destination another store reads runs after it`), each against
  the values its tensors write; the Indexing suite's `run_rangeify › stores › a
  value stored into storage it reads is not stored whole here`; the recorded
  graphs of the program `assign_double_diamond`; the norn bench's
  `nuts/schools/4`, tolk's `adam_mlp` rows, vega's Adam row and rune's MLP step
  row.

## D199. Withdrawn

D199 fused independent calls of one domain into one kernel. A launch of bodies (D404) runs the same grouping and more without one set of optimisations for the union: 1.65 against 1.68 ms a NUTS step on CUDA, equal grouping, inside the row's spread, and equal on the host and in compile time.

## D200. A tensor core's row upcast keeps a warp within a run of rows

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`,
  whose tensor core upcasts the core's M range by the first of 5, 4, 3 and
  2 that divides it).
- **tolk:** `Heuristic.tensor_cores`, through `Heuristic.within_run` and
  `Heuristic.owner_run`.
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
  over blocks of 16 is not measured. rune's routed product (`Lower_reduce`)
  now takes blocks of 64 rows, so the rule fires where the block count is a
  multiple of 5 and leaves no other first candidate. gpt-oss-20b's experts at
  a prefill of 536 tokens, 65 blocks and 4160 rows on the unstaged path, kaun's
  `GptOss/experts-at-prefill-536`, take 3.02 ms a layer against 6.96 ms
  without the rule (upcast 4 against 5), and the prefill 117 ms against 207 ms;
  minima of four interleaved rounds with D211 and D122 withdrawn. 512 tokens, 63 blocks, already take 4 and
  do not reach it.
- **Pinned by:** the Heuristic suite: the `routed_run_16`, `routed_run_32`,
  `routed_run_64` and `routed_run_128` cases on each renderer, and `a core's
  warp holds rows of one run where an upcast of the core's rows keeps it in
  one`; on fudge, kaun's
  `GptOss/experts-at-prefill-536` row, its baseline pending the scheduled
  re-record.

## D201. A search's samples start from a clean, cold cache

- **tinygrad:** `codegen/opt/search.py` (`_time_program` is asked to
  clear the L2 only where the device has `invalidate_caches`, NV's) and
  `engine/realize.py` (`time_call`, which invalidates NV's caches
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

- **tinygrad:** `uop/spec.py` (`spec_shared`, an AFTER of a scalar or a computed
  value, a LOAD's among them, after any dependencies, which `spec_program`
  includes), which `codegen/__init__.py` `full_rewrite_to_sink` checks under
  `SPEC`, 1 by default; `codegen/late/linearizer.py` (`pm_add_control_flow`,
  which wraps a nested loop's bound in an AFTER of the range or loop end it
  follows); `codegen/opt/postrange.py` `convert_loop_to_global`,
  `upcastable_dims` and `apply_opt`.
- **tolk:** `Spec.program`; `Postrange.Scheduler.read_by_bounds`,
  `upcastable_dims`, `apply_opt` and `convert_loop_to_global`;
  `Ops.known_at_launch`, which `Codegen.split_blocks` (D80) and
  `Renderer.Estimates.of_uops` (D191) read; `Heuristic.streams`.
- **Differs:** a loop bounded by a load, as `j < x[i]`, compiles: a
  program's Spec accepts an AFTER of a scalar LOAD whose dependencies are
  ranges and loop ends, the linearizer's edges, or loads, as a lane selected
  from loaded lanes is; a graph's Spec refuses one, as tinygrad's does. An
  AFTER of a load and a store is refused, where tinygrad's Spec accepts an
  AFTER of any scalar value since b63e6cef8: the linearizer orders the AFTER,
  not the load it holds, which would read before the store. An
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
  axis`, and `axes.tolk`'s `padto_reduction_bound`. The
  Spec suite: `verdicts.tolk`'s `an int32 load after a range`, `a range
  bounded by a load after a range` and `a load after a store`. The Codegen
  suite: `hand-written kernels on the host › a lane of stored lanes keeps
  every lane's store`.

## D230. A GPU kernel decodes a value R rows read at most ceil(R / 16) times

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (the
  tensor cores) and `hand_coded_optimizations` (the upcasts of output axes by 3
  or 4); it streams nothing (D89), and states no bound on how often a kernel
  computes a value its outputs share.
- **tolk:** `Heuristic.stream`'s upcast of the reuse ranges and
  `Heuristic.hand_coded_with`'s late stream of a decoded operand.
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
  decoded each value 16 times, and take 4. The host shares its decodes as
  its registers allow (D92).
- **Reason:** (d): MXFP4 products of 16384 by 16384 on fudge's RTX 5000 Ada
  (kaun's `Quant/cuda/mxfp4-product-of-16384-*`, three interleaved rounds
  under the timing lock, GPU shared with other jobs): 17 bfloat16 rows
  4.58-11.91 ms to 1.07-3.28 ms; 64 float32 rows, streamed, 14.8-20.0 ms to
  4.98-6.68 ms (12.5-12.9 in quieter rounds before), past the stored
  decode's 9.6; 12, 16, 32, 64, 128 and 256 bfloat16 rows unchanged within
  noise. Larger float32 counts are not benched. nonnormal's R9700 rows are
  pending, among them 64 float32 rows, where a model of its decode cost
  puts 4 decodes a value at about 43 ms against 31 stored.
- **Pinned by:** the Heuristic suite's `a GPU kernel decodes a value R rows read
  at one index at most ceil(R / 16) times its core's copies`, over decoded
  products of 1 to 512 rows of bfloat16, half and float32 on the CUDA, RDNA3,
  RDNA4, CDNA and Metal renderers, and its `a decoded product's optimisations
  keep its writes` (17 and 19 rows padded, 64 float32 rows streamed, evaluated);
  kaun's `Quant/cuda/mxfp4-product-of-16384-17-bfloat16-tokens` and
  `-64-float32-tokens`.

## D235. A tensor core's lanes are a warp of its threads

- **tinygrad:** `codegen/gpudims.py` (`group_gpudims`), which lays a
  kernel's warp and local ranges on the thread axes whatever its tensor cores
  need, and `codegen/simplify.py` (`mark_range_mod` and
  `pm_split_ranges`), which split a local range, never a warp, by
  its divisions, the most significant part taking the first thread axis.
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

- **tinygrad:** `codegen/__init__.py` (`do_stack_wmma`), which replaces a tensor
  core's operand by the stack of its lanes, and `uop/movement.py`
  (`mop_cleanup`), which folds that stack back to the operand, so that a bitcast
  operand that changes the lane count, such as four words read as eight
  bfloat16, rewrites forever ("infinite loop in graph_rewrite");
  `renderer/cstyle.py` `CUDARenderer.string_rewrite` (CUDA's `string_rewrite`),
  which renders a bitcast with its scalar types, so that a bitcast of a vector
  does not compile ("no suitable conversion function from uint4 to uint").
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

## D237. Ends of one range that reach each other run apart, each a loop

- **tinygrad:** `codegen/__init__.py` (`merge_reduce_ends`), which
  merges the ends that close the same ranges in one context, and leaves
  every end of those ranges unmerged when one reaches another (5229d642d):
  a copy of a tile reads, through the scores, what the copy before it
  staged, and merged would contain itself. Unmerged, the ends share their
  ranges, and over a loop range the linearizer refuses the kernel (an
  assertion, `codegen/late/linearizer.py` `CFGContext.__init__`).
- **tolk:** `Codegen.merge_reduce_ends`.
- **Differs:** ends of one context that reach each other go apart, each a
  loop of its own: the later one takes cloned ranges, as an end of another
  context does, and every other end that closes one of those ranges, along
  other ranges too, stands aside while its ranges are renamed, so the loop
  it reads runs once. Ends that do not reach each other merge as before,
  beside those that do. tinygrad refuses these kernels over loop ranges;
  tolk computes the right values.
- **Reason:** (e).
- **Pinned by:** `Tolk.Codegen › effects run as often as written › a copy
  reads the copy ended before it`, over upcast and loop ranges, apart and
  shared, and `› a copy that reads an end of its range along another range
  runs that end once`, whose second end takes cloned ranges as D137 gives
  them, which tinygrad does not (its linearizer then refuses the kernel);
  over threads, `lanes and threads › an end reading the stores of an end of
  its threads runs after it on CUDA`.

## D238. A tensor core says whether its accumulator chains

- **tinygrad:** `renderer/tc.py` (`TensorCore`), whose fragments hold
  the fact and whose comment states that "the k bits may be permuted,
  identically in A and B", with no function that answers it.
- **tolk:** `Tc.chain`, `Tc.transpose`.
- **Differs:** `Tc.chain x y op` says whether `x`'s accumulator is `y`'s
  operand `op` without leaving its lanes, the permutation of `K` that takes,
  and whether both accumulators hold the same lines in each lane.
  `Tc.transpose tc` is the core of `Dᵀ = Bᵀ Aᵀ + Cᵀ`, which a program
  computes by giving `tc` its operands in the other order: an accumulator
  that chains only as `B`, as RDNA 4's and CDNA's, chains as `A` in the
  transposes.
- **Reason:** (b): rune's `Lower_attention`, the fused attention lowering,
  reads it to chain the scores' accumulator into the values' product, and
  takes a core that chains as `B` transposed.
- **Pinned by:** `Tolk.Tc › chain › a core's accumulator as its own operand`,
  per shipped core, and `› an accumulator chains iff some permutation of K
  reads it in place, and the chain reads it so`, against every map of the
  moved bits over every pair of each target's cores; `Tolk.Tc › transpose`,
  over every shipped core and every pair of each target's cores.

## D202. A join keeps its dependencies, and holds the state under it

- **tinygrad:** `uop/symbolic.py` (`symbolic`'s ordering rule): an `AFTER` keeps
  among its dependencies only effects, ranges, ends, back edges, linears and
  stages, and replaces any other by its sources, an `AFTER` included.
  `x.after(y')`, where `y'` is `y` once assigned, becomes `x` after `y`'s
  kernel, which `schedule/__init__.py` `create_schedule` reads as a state of `x`
  that `y`'s kernel writes. The rule came with `AFTER` itself, for a kernel's
  register initialisation (`b8a9cce78`, `d1e2c393f`), and no commit states a
  reason for the graph of kernels.
- **tolk:** `Shape.symbolic`'s ordering rule; `Ops.is_join`;
  `Schedule.value_state`, `Schedule.swap` and the waits of
  `Schedule.schedule`; `Rangeify.unjoin` and `Rangeify.handle_after`.
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
- **Pinned by:** the `Tolk.Schedule` suite: `create_linear_with_vars › joins › a
  reader of a join waits for the writer of the state it holds`, `› a reader of a
  join of an overwritten state reads it as it was`, `› a write over a join runs
  after the readers of the state it holds`, `› two joins over one state are no
  two writes` and `› a store reading its destination through a join takes it
  once`; and the programs `read_join_of_assigned` and
  `read_join_of_overwritten`, whose recorded schedules and linears and whose
  values (`create_linear_with_vars › values`).

## D250. Two stores of one value into one storage are two states

- **tinygrad:** `schedule/prepare.py` (`earliest_rewrites`, "remove
  two STOREs that store the same thing to the same place", for
  `TestSchedule.test_dedup_assign`), which rewrites `a1.after(a1.store(v))`,
  where `a1` is `buf.after(buf.store(v))`, to `a1`: the program's second state
  becomes its first. A loop of a call that writes the second state and reads
  the first then takes one state at both arguments, and its schedule passes
  the call the storage at both, so the loop would read what each trip left
  (tinygrad renders no loop of a call, and runs none). Without the rule, the
  loop runs after the second store and before the overwrite of the first, a
  cycle, which tinygrad refuses ("cycle detected in assign graph",
  `schedule/__init__.py` `create_schedule`).
- **tolk:** `Prepare.earliest_rewrites`, which has no such rule.
- **Differs:** a store of the value the store before it stored into its
  storage runs, and the state it makes is a state of its own, after the one
  it overwrites. A call or a loop given both states reads each as the program
  names it. The cost is a kernel for each repeated store: the program of
  `test_dedup_assign` runs two stores where tinygrad runs one.
- **Reason:** (e). With the two states one, the loop above is given one state
  at the argument it writes and at one it reads, and reads at the second, as
  such a loop does (D160), the state each trip leaves, where the program reads
  the first state as it was: tolk computed `[24, -8, 48, 16]` for `[18, -6,
  36, 12]`.
- **Pinned by:** the `Tolk.Schedule` suite: `create_linear_with_vars › writes
  › a loop given two states of one value reads the older as it was`, and `›
  laws › a schedule writes what its tensors compute`, whose programs repeat
  their statements, constants included; the `Tolk.Prepare` suite:
  `prepare_rangeify › earliest rewrites › two equal stores into one storage
  are two states`.

## D255. A kernel whose loop's bound loads from memory is not searched

- **tinygrad:** `codegen/opt/postrange.py` (`apply_opts`), which searches every
  kernel under BEAM, timing it on `args_from_ast`'s new buffers
  (`args_from_ast`), their contents unspecified, and each variable at the middle
  of its bounds.
- **tolk:** `Postrange.apply_opts` (`bound_by_memory`), `Ops.bound_loads`.
- **Differs:** a kernel with a loop bounded by a load, as `j < x[i]` (D220),
  takes the heuristic's optimisations when a search is asked for, as
  without one. tinygrad refuses such loops.
- **Reason:** (e), following D220: the trips such a loop runs are what
  memory holds, which no sample the search makes knows. On unspecified
  bytes a search timed up to 2^31 - 1 trips a thread for an `int32` bound,
  each candidate for minutes or more, and measured whatever the memory's
  last user left; it then keeps its choice in the disk cache keyed by the
  kernel alone. A call's own values are not the search's: it runs while
  the program compiles, and its choice serves every later call. Timing on
  zero bytes gives a stated count, but usually no trips, so the search
  would rank candidates by the kernel without its loop and keep that
  ranking as measured. No stated count fits every bound: the middle of an
  `int64` bound clamped at 0 is 2^62.
- **Pinned by:** `Tolk.Search on the host`: `a kernel whose loop's bound
  loads from memory is not searched, and takes the heuristic's
  optimisations`.
## D210. A GPU merges bfloat16 accesses as float16's

- **tinygrad:** `codegen/late/coalesce.py` (`memory_coalescing`: a GPU
  merges adjacent accesses of `float`, `half`, `int`, `uint` and the 8-bit
  floats into vectors of 4 and 2, and leaves `bfloat16` scalar; the same at
  `6b7b018da`, `memory_coalescing`); `renderer/cstyle.py`
  (`CUDARenderer.render_kernel`, whose 2-lane `bfloat16` vector, never merged
  there, has no constructor: it renders `make_nv_bfloat162`, which
  `cuda_bf16.h` names `make_bfloat162`).
- **tolk:** `Coalesce.memory_coalescing`; `Cstyle.cuda_kernel`.
- **Differs:** on a renderer other than the CPU device's, adjacent `bfloat16`
  accesses merge into vectors of 4 and 2 as `float16`'s do, and the CUDA
  renderer defines `make_nv_bfloat162` as `make_bfloat162` when a kernel holds
  a 2-lane `bfloat16` vector. The host is unchanged: its `bfloat16` accesses
  stay scalar. tinygrad states no reason for leaving `bfloat16` out, and its
  CUDA kernels could not build its 2-lane vector, a defect of its own.
- **Reason:** (d): a tensor core's `bfloat16` operands, each loaded as a
  2-byte scalar, made every bfloat16 product on fudge's RTX 5000 Ada about 1.5
  times slower than its float16 product of the same optimisations. GPU time,
  three interleaved rounds, before and after: 2048 cubed 480-482 to 310-325
  us, 4096 cubed 2908-2920 to 1824-1848 us, gpt-oss-20b's query projection at
  a 512-token prompt (512 by 4096 over 2880) 274-277 to 191-201 us, its key
  projection (512 by 512) 124-125 to 88-110 us, its output projection (512 by
  2880 over 4096) 243-245 to 185-199 us, a 512 by 5760 product over 2880
  308-312 to 249-260 us, attention's values at 2048 slots (float32 weights by
  bfloat16 values) 1060-1081 to 913-931 us; the model's 512-token prefill
  183.0-184.9 to 176.3-176.9 ms (two rounds). One layer's experts, which
  decode their MXFP4 weights in place, overlap (4380-4497 and 4337-4441 us),
  and so do attention's bfloat16 scores. Metal (`bfloat2`) and AMD (an
  `ext_vector_type` of `__bf16`) render the same merges; unmeasured, as fudge
  has neither.
- **Pinned by:** the Coalesce suite's `lengths › on a GPU, eight consecutive
  loads are two loads of four, of bfloat16` and `› on the host, bfloat16 loads
  stay apart`; the Codegen suite's `bfloat16 vectors on CUDA › a bfloat16 sum
  of lanes loaded and stored together is exact` (2 and 4 lanes, on the
  machine's first CUDA GPU); the goldens generated from tinygrad with this
  rule; the tolk bench's `contract/bf16-2048`, `contract/bf16-4096` and
  `contract/bf16-512x5760x2880` rows on fudge, their baselines pending the
  scheduled re-record.

## D211. Withdrawn

N took one more upcast by 2 within 160 outputs a lane, after the core's upcasts
of M and N. Staging (D262) took the kernels it was measured on, the contraction
rows, which no longer reach it; what it still touched was gpt-oss-20b's routed
experts, where a warp's larger tile costs: on fudge's RTX 5000 Ada, minima of
four interleaved rounds, one layer's experts 3.54 ms with the rule and 2.77 ms
without, and the 512-token prefill 141.0 ms and 121.8 ms.

## D212. A contraction on a GPU's registers holds 8 by 8 outputs a thread

- **tinygrad:** `codegen/opt/heuristic.py`
  (`hand_coded_optimizations`: the upcast loop takes each axis once, by 3 or
  4, while fewer than 32 lanes are upcast; the unroll needs fewer than 64; the
  locals stop at 128 threads; the same at `6b7b018da`).
- **tolk:** `Heuristic.block`, `Heuristic.upcast_more`, `Heuristic.locals` and
  `Heuristic.general`.
- **Differs:** on a renderer with local indices, a contraction summed in 4
  bytes or fewer whose upcast loop took two axes, each by 4, each unread by
  some operand, takes a second upcast of each by 2 when 2 divides both and
  what is left of them holds a SIMD group of 32 threads in powers of two; its
  last unrollable axis is then unrolled by 4 where 4 divides it, and its
  workgroup takes at most 32 threads. A thread then holds 8 by 8 outputs, and
  each element it loads feeds 8 of them.
- **Reason:** (d): float32 contractions on fudge's RTX 5000 Ada, which take no
  tensor core (TF32 is off), ran at 6 TFLOP/s of the part's 65 with 4 by 4
  outputs a thread in workgroups of 128 threads. More outputs a thread cover
  the loads' latency with fewer threads in flight (Volkov, "Better Performance
  at Lower Occupancy", GTC 2010, slide 48: "This often can be accomplished by
  computing multiple outputs per thread"); CUTLASS's SIMT default holds 8 by 8
  a thread. GPU time, three interleaved rounds, against the rules before it:
  1024 cubed 361-375 to 142-161 us, 2048 cubed 2799-2810 to 846-855 us, 4096
  cubed 19.3-19.8 to 6.0-6.7 ms, 8 products of 512 cubed 416-433 to 119-141
  us, attention's scores at 2048 positions 651-660 to 543-548 us; attention's
  values (`h i j, h j d -> h i d`, float32 weights) 842-843 and 837-854 us,
  read-bound. No reference fixes the threads of a workgroup: 32 is measured.
  Attention's scores (`h i d, h j d -> h i j`, 8 heads of 64 at 2048
  positions) ran faster at 64 threads (365-373 against 527-534 us, three
  rounds on main before device facts landed), the products of 1024 to 4096 and attention's
  values at 32; that row is the band. Metal and AMD take the same rule,
  unmeasured, as fudge has neither. Under `--fmad=false` (D25) the unrolled
  reduce renders a multiply and three fused multiply-adds per four products,
  which caps these kernels below the part's peak whatever their layout.
- **Pinned by:** `Tolk.Heuristic`'s `a contraction in registers` cases
  (`matmul`, `matmul_256` and `scores` blocked on CUDA, `matmul_256` on AMD;
  `matmul_1000`, whose halved axes hold one thread, not blocked), with the
  layouts in the suite's `.tolk` files; rune's
  `test_jit_gpu` `products › a float32 matmul is within its bound of the exact
  product` at extents of 64 and 128 summed over 13 and 52, on the machine's
  first GPU; the tolk bench's `contract/f32-2048` and `contract/qk-f32-2048`
  rows on fudge, their baselines pending the scheduled re-record. A sum in 8
  bytes is not blocked: 64 accumulators of 8 bytes and their operands would
  pass a thread's 255 registers.

## D221. A part of a pattern that names nothing gives one naming

- **tinygrad:** `uop/ops.py` (`UPat.match`, which extends its
  result with a store for each ordering of each source pattern, so a store
  comes back once for each way a part that names nothing matches).
- **tolk:** `Ops.Upat.first`, `Ops.Upat.alternatives` and so
  `Ops.Upat.match_` and `Ops.Pattern_matcher.rewrite`.
- **Differs:** ways of matching that differ only within a part of the
  pattern that names nothing give one naming, where tinygrad gives the same
  store once for each. A rule's function is called once for that naming,
  where tinygrad calls it again after it declines. A rule's result is the
  same when its function's result depends only on the naming.
- **Reason:** (d): constant folding's rules, repeated over every source of
  an operation whose sources may each be a constant or a stack of them,
  enumerated orderings that named nothing through continuations. Testing
  such a part once cut rune's `compile/special/betaincinv-grad-f64` row from
  261 M to 235 M words, counted at the process's exit at `s=64M`. The row's
  baseline on fudge is pending a quiet run, and the compile bench records
  time alone.
- **Pinned by:** the Ops suite: `Pattern_matcher › a part of a pattern that
  names nothing gives one naming`; rune's
  `compile/special/betaincinv-grad-f64` row, once recorded on fudge.
## D240. The host's output upcasts take no floor of elements

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  upcasts of output axes while the upcastable axes hold 1024 elements).
- **tolk:** `Heuristic.enough_work` and `Heuristic.upcast_more`.
- **Differs:** on the host (a renderer that knows its vector registers,
  D186), the contiguous upcast and the upcasts by 3, 4 or 2 apply however
  few elements the upcastable axes hold, bounded by the register budget
  (D186) and the conversions' and the iteration's lanes (D241) alone. Other
  devices, and a host renderer that knows no registers, keep tinygrad's
  1024 elements: on a GPU the outputs are its threads.
- **Reason:** (d): rune's host products of few outputs over long sums,
  gpt-oss's decode projections and lorenz_simple's step. On the host an
  upcast takes lanes that share a load and run independent chains, where a
  launch's blocks (D80) would each repeat the load. A floor counted in the
  split loop's iterations, at least 16 or the blocks the work fills, was
  slower than none on every kernel it bound (fudge, loaded, both orders, all
  cores): 64 sums of 2^15 bfloat16 values took 0.17-0.19 ms with none and
  0.65-1.6 ms with it, the sums of 64 columns over 2^17 rows 2.6-3.5 ms and
  5.4-9.1 ms, 8 products of 2^20 values 1.04-1.25 ms in 2 blocks against
  1.18-1.32 in 8. Against the 1024 floor, one P-core (`taskset -c 5`):
  gpt_oss_kv 1508-2293 µs to 89-98, lorenz's `r_32_3_2_4_3_400` 48-50 µs
  to 25.6-26.2, `rows_m4` 8.3-8.5 µs to 3.0-3.1, `attention_values`
  1356-2118 µs to 277-283; all cores, the column sums 17.9-20.3 ms to
  3.1-3.4, the bfloat16 sums 0.95-1.8 ms to 0.10-0.17, the 8 products
  2.26-3.02 ms to 2.28-2.89. Measured under load only: whether a
  memory-bound product of few outputs loses its blocks on a quiet host is
  for the scheduled quiet re-record. The compiler's own passes do more work
  on the larger programs: the tolk bench's `linearize`, `render` and `warm`
  rows of its recorded programs allocate 1.3-2.9 times the words,
  `gpt_oss_decode/codegen` 1.7 times, `cumsum`'s rows 0.6-0.7 times.
- **Pinned by:** the Heuristic suite's `the host's lanes › on AVX2 a
  recorded host kernel of few outputs takes its lanes` and `the
  optimisations chosen are tinygrad's › … › the optimised kernel`, the
  `cpu` and `avx2` cases, among them `matvec_tall`, `vecmat_long` and
  `column_sums`; the tolk
  bench's `host/gpt_oss_kv/heuristic`, `host/vecmat_1000/heuristic`,
  `host/matvec_tall/heuristic` and `lorenz/r_32_3_2_4_3_400/heuristic`,
  whose fudge baselines the scheduled re-record records.

## D241. A host conversion takes at most 64 lanes, an iteration 512

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (the
  upcasts while fewer than 32 lanes and the unroll while fewer than 64, counting
  every lane of the kernel).
- **tolk:** `Heuristic_host.converted_lanes`, `Heuristic_host.conversion_lanes`,
  `Heuristic_host.iteration_lanes`, `Heuristic_host.unroll_lanes`,
  `Heuristic_host.lanes_after` and `Heuristic_host.fits`; `Cstyle.clang_lanes`
  and `Codegen.split_lanes`.
- **Differs:** on the host, the upcasts after the masked and shared ones
  and the unroll (D186) apply only while each conversion (`CAST`) computes
  at most 64 lanes: of the upcast axes its value reads, the lanes holding
  distinct values, `n / gcd n d` of `n` lanes read as a quotient by `d`,
  and none of a reduce's closed axes, whose lanes it sums first. They also
  keep an iteration within 512 lanes, the product of the upcast axes, the
  unrolled ones among them; an output upcast counts too the lanes an unroll
  of the last unrollable axis would take, the axis whole when it holds at
  most 32, 4 otherwise, as tinygrad's 32 output lanes leave room below its
  64 for the unroll.
- **Reason:** (d): Clang's straight-line vectorizer, which packs a
  conversion's vector from the lanes it converts, takes time much faster
  than linear in them where they are loads gathered from apart: 98% of
  gpt_oss_kv's 0.9-1.2 s at 128 lanes is that pass (`-ftime-report`), 0.15
  s at 64 and 0.03 s at 32. Against the unbounded lanes, Clang's CPU on
  fudge: `conv_half` 6.5-6.6 s to 52-66 ms (its run time tied),
  `matmul_half` 137-144 ms to 29-32, `gpt_oss_qkv` 174-178 ms to 28-37; and
  the run time of the narrower vectors, one P-core: `matmul_half` 351-352
  µs to 123-124, `batched_matmul_half` 177-215 to 61-63, `gpt_oss_qkv`
  6.4-20.5 ms to 3.2-7.2 (loaded). gpt_oss_kv takes 32 output lanes and no
  unroll, 89-98 µs, where 32 by 4 took 332-336 and 16 by 4 338-344. The
  routed products read their block's expert as a row's quotient and keep
  their unroll by 8, which ran 1.35 times as fast as by 2.

  The iteration's lanes: Clang's time grows much faster than them where
  they put together vectors from many loaded rows, 97% of it in InstCombine
  and VectorCombine (`-ftime-report`). rune's `Lower_reduce › work › a
  Contracting contraction of 8 groups works in blocks` builds
  `r_23_4_16_4_16`, 16 by 4 output lanes by its 16 rows unrolled whole,
  each row loaded at an index read from memory. Clang 19 for fudge's
  processor (CPU time) took 5.3-5.6 s on it, and over its other upcasts and
  unrolls 0.04-0.05 s at 64 lanes, 0.07-0.09 at 128, 0.17-0.23 at 256 and
  0.9-1.1 at 512. It now takes 32 output lanes by its rows, 512 lanes, in
  1.1 s; it runs as fast (22.2-22.8 µs against 22.4, one P-core) and keeps
  its rows' loads out of its output loop, which an unroll by 4 or none
  would load four times (73,600 estimated bytes against 153,088, past that
  test's bound). Of the 15,220 host kernels the tolk, rune and jera suites
  optimise, 23 change, all rune's: of the 17 past 512 lanes, a product of
  64 output lanes summed over 32 values unrolled whole (2048 lanes) ran
  605-629 µs and 517-533 with 16 output lanes, one over 16 values (1024)
  700-726 and 670-719, and the rest tie; past 256 lanes the next slowest
  compiles in 0.24 s, where 0.44 s was. The cost: the Heuristic suite's
  `conv` on AVX2 unrolls its 3 by 3 window by 3 alone, 192 lanes against
  576, and runs 1.12 times as long (620 µs against 554, one P-core), though
  Clang takes 0.08 s on it at 576: the lanes stand for Clang's time only
  as a bound. 256 lanes would hold every kernel within 0.25 s, but take
  the routed MXFP4 decodes (`decoded_rows_*`, gpt-oss's experts) from their
  unroll by 8 to 4, and `matmul_1000` ran 1.02-1.06 times as long there.
  Computing a vector operation wider than 256 lanes lane by lane instead,
  in the renderer (D141), compiled the contraction in 0.26 s but ran it in
  25.5 µs against 21.4 and `paired_products` in 177 µs against 113.

  Clang computes at most 64 lanes of a vector operation at once (D141's
  `Renderer.vector_lanes`): an operation of more lanes is vectors of 64 and a
  last of the rest. Of the 15,220 host kernels, 54 hold one; Clang's CPU time
  on them falls from 5.8 s to 3.8 s in all, the contraction's from 1.06 s
  to 0.11 s, `paired_products`' from 0.19 s to 0.06 s, and the slowest's
  from 1.06 s to 0.19 s, none slower (two compiles each, interleaved, one
  P-core); rune's compile bench row `products/segments-grad` takes 0.53-0.56
  s cold against 1.45-1.46 s. Run times hold: the tolk bench's
  `host/paired_products/heuristic` 110-119 µs against 113-120 (16 runs,
  interleaved, one P-core), the contraction 21.5 µs against 23.2-24.2, and
  rune's bench rows allocate the same words. Vectors of 16 cut Clang further
  past 64 lanes (the contraction 0.07 s) but not below: on the 589 kernels of
  17 to 64 lanes it took 19.6 s against 19.9, `conv_half` and `gpt_oss_kv`
  ran 1.5% and 1.2% longer in each of ten interleaved pairs, and the compile
  bench's `lanes/held-fp8-36` 1.30-1.36 times as long. A rest of one lane
  would be computed apart, which made Clang's loop strength reduction take
  seconds (D141): the last 65 lanes are vectors of 33 and 32. In the tolk
  bench's stages, the split makes `gpt_oss_decode/codegen` allocate 0.7%
  more words, building the vectors of 64 for operations whose lanes later
  fold apart (its instructions are the same), and `adam_mlp`'s linearize,
  render and warm 4.5-6% fewer. `conv` holds no vector that wide, so its
  lanes are the bound's alone: at 576 it ran 529-555 µs against 596-604,
  but a bound of 1024 took three of rune's products from 16 output lanes
  unrolled whole to 32 lanes and no unroll, 2724-2726 µs against 2301-2310,
  2416-2417 against 2029-2078 and 2948-3004 against 2754-2848, and another
  to 1024 lanes, 672-685 µs against 659-669.

  The bound stays since those vectors of 64 lanes held the compile time.
  Without it, of 5,240 recorded (kernel, target) pairs 4 change: `conv` on
  AVX2 unrolls both axes of its window, 570 µs against 672, and rune's
  contraction of 8 groups runs in the same 21.0 µs, but a float32 product of
  64 by 128 rows over 32 values, a layer's matmul, goes from 16 output lanes
  unrolled whole to 64 by the sum's 32, 78 µs against 62, and Clang's
  0.33-0.5 s against 0.13-0.2; its row `Jit/grad-of-jit-mlp-float32` takes
  215-226 µs against 201-204, and `products/segments-grad` and
  `linalg/eigh-f64-32x32` compile 1.12 times as long, on a row users run.
- **Pinned by:** the Heuristic suite's `the host's lanes › on AVX2 a
  recorded host kernel's conversions take their lanes` and `› on the host a
  conversion computes at most 64 lanes`, over every recorded kernel and
  lorenz_simple's on four register files, and the host cases of `the
  optimisations chosen are tinygrad's`; rune's compile bench row
  `products/vecmat-bf16`, whose fudge baseline the scheduled re-record
  records. The iteration: the Heuristic suite's `the host's lanes › on the
  host an iteration computes at most 512 lanes`, over the same kernels and
  rune's contraction of 8 groups (`contracting_groups.graph`), `› on AVX2
  an output upcast leaves room for the whole unroll of an axis of at most
  32` and `› an output upcast leaves 4 lanes for the unroll of an axis past
  32`; rune's `Lower_reduce › work › a Contracting contraction of 8 groups
  works in blocks`; rune's compile bench row `products/segments-grad` (5.5
  s cold before, 1.5 s after, 0.55 s with vectors of at most 64 lanes, on
  fudge), whose baseline, with the tolk bench's `host/conv/heuristic`, the
  scheduled re-record records. The vectors: the Heuristic suite's `the
  host's lanes › on AVX2 a recorded host kernel's program declares vectors
  of at most 64 lanes`, and the Codegen suite's `vectors in programs › a
  host operation of any lanes computes vectors of at most 64 and writes what
  its kernel writes`.

## D260. A tensor core accumulates its running sum at the reduce

- **tinygrad:** `codegen/__init__.py` (`pm_wmma_add`: an add of a WMMA, of a
  permute of one, or of a permute of a reshape of one, folds the addend into the
  core's accumulator) and its use in `pm_expand_broadcast` and
  `pm_reduce_local`.
- **tolk:** `Codegen.lower_reduce`, with `Codegen.into_core` and
  `Codegen.core_at`.
- **Differs:** the reduce that builds a sum of cores' values feeds its
  running sum to each core as its accumulator, `D = A B + C`, through any
  chain of reshapes and permutes the expander leaves around the core, and
  chains the cores of a K axis unrolled past the core in K order:
  `wmma(a1, b1, wmma(a0, b0, acc))`. Only the running sum enters a core: an
  add after the sum, as rune's initial value, stays an add. tinygrad folds
  any addend by pattern, in three forms of four: a core behind a reshape
  alone adds the core's value to the sum, so whether a core accumulates
  depended on its lanes' shape, its upcasts and its unroll, and an unrolled
  K summed its cores apart before adding them.
- **Reason:** (d) and (e)'s bits: a core's sum is one sequence of the core's
  steps at any split, so a contraction's bits no longer change with its
  opts' unroll or with shared-memory staging, which moves only operands. On
  Ada's cores, which truncate (Khattak and Mikaitis, arXiv 2512.07004v4,
  §4.1.2), the fold is D29's measured error: 0.497 of `Nx.contract`'s bound
  on rune's adversarial rows at `K = 2880` and `2048`, against 0.013 to
  0.028 unfolded; both within the bound. A core accumulating two steps a
  trip (K unrolled by 2) beat the same kernel unfolded by 1.06 to 1.14 on
  bfloat16 products of 512 by 512 by 2880, 512 by 4096 by 2880 and 2048
  cubed (paired medians on fudge's RTX 5000 Ada, staged as the
  shared-memory staging design runs them).
- **Pinned by:** the Codegen suite, `tensor cores on CUDA › a product summed
  in float32 has the same bits whether the core takes one or two steps of K
  a trip` (fails unfolded), `tensor-core accumulators › the running sum
  replaces a zero accumulator` (fails unfolded, on every target's cores),
  and `stages › … › matmul_{half,half_big,half_tile,bf16_tile,tc}_{cuda,hip}`
  and `matmul_tc_unroll_metal`; the cstyle and renderer goldens of each core.

## D261. 16-bit floats copied into shared memory merge to 16 bytes

- **tinygrad:** `codegen/late/coalesce.py` (`memory_coalescing`: a GPU
  merges `half` accesses into vectors of 8 under `ALLOW_HALF8`, an
  environment setting off by default, and of 4 and 2 otherwise).
- **tolk:** `Coalesce.memory_coalescing`; `Renderer.t.half8`, set by
  `Cstyle.cuda` and `Cstyle.hip`. `Setting.allow_half8` is withdrawn.
- **Differs:** on a renderer that writes vectors of eight 16-bit floats (CUDA
  and HIP; Metal's vectors hold at most four), adjacent `float16` or
  `bfloat16` accesses of a group every access of which copies from global
  into local memory (loads of global memory whose only use is the value of a
  store into local memory, or those stores) merge into vectors of 8, 4 and 2,
  16 bytes at most; every other access of them merges into 4 and 2, as
  float16's do in tinygrad without `ALLOW_HALF8` (bfloat16 by D210). No
  setting changes it.
- **Reason:** (d): a tensor core's operands staged through shared memory
  (D262) are copied 16 bytes a thread, which PTX's `cp.async` copies in one
  instruction (D263); 8 bytes at a time they take twice the copies and no
  asynchronous one. Elsewhere 16 bytes gained as often as they lost. Merged
  everywhere, they changed three kernels of gpt-oss-20b's prefill of 512
  tokens and no staged contraction (2048 cubed and the projections compile to
  the same source either way). GPU time on fudge's RTX 5000 Ada through nsys,
  three interleaved rounds, us a launch, 16 bytes against 8: the full
  attention layers' product of probabilities by values (12 a prefill)
  785-791 against 562-564, its sliding-window twin 769-772 against 798-800,
  the router 27.8-28.0 against 28.9-29.0; kaun's `GptOss/prefill-512`
  174.0-176.7 ms against 171.8-172.9. In the full product both forms hold
  128 registers, and ptxas issues 6 of its loop's 12 loads at the top with
  16-byte value loads, against 12 of 16 with 8-byte ones, the rest of the
  probabilities' loads one at a time between the multiply-adds. Run alone,
  with its operands moved by 0 to 3072 bytes, the 16-byte form takes
  737-773 us and the 8-byte form 709-870: more loads in flight pay where the
  probabilities' addresses spread over the memory's channels, as they do in
  the prefill.

  On RDNA4 the rule leaves unstaged operands alone. A lane's operand of the
  16-by-16-by-16 core is two runs of four 16-bit values, eight elements
  apart, so 8 bytes is the widest contiguous load. On nonnormal's R9700
  (gfx1201), float16 and bfloat16 products of 2048 cubed, weights in either
  layout, staged and unstaged, compile to the same HIP source and the same
  comgr instructions with every access merged to 16 bytes. RDNA3 and CDNA
  are unmeasured.
- **Pinned by:** the Coalesce suite's `vectors of eight 16-bit floats` group
  (`the runs of eight 16-bit floats' loads and stores`, of `float16` and
  `bfloat16`; `half_cuda`, `half8_cuda` and `accesses_cuda`, generated from
  tinygrad with this rule); the Stage suite's `sources` (`cp.async`); kaun's
  `GptOss/prefill-512`.

## D262. A tensor core's operands are staged through shared memory

- **tinygrad:** none: no pass stages an operand. Its `OptOps.LDS`, one opt a
  buffer, was reverted (#9669, #9691), and its hand-written kernels stage by a
  scheduler `STAGE`, which skips the heuristic. `uop/spec.py`
  (`spec_shared`: an AFTER of a scalar or a computed value only).
- **tolk:** `Stage.operands`, which `Codegen.full_rewrite_to_sink` runs after
  the opts; `Stage.depth` and `Stage.layout`; `Heuristic.staged_tile`;
  `Renderer.t.async_copy`; `Setting.stage`.
- **Differs:** after the opts, each core operand that reads global memory
  through an index whose terms reading the workgroup's threads and lanes form
  a dense box of strided dimensions is copied by every thread of the
  workgroup into shared memory, 16 bytes a copy where its innermost
  dimension is contiguous, in rows whose 16-byte chunks are permuted by an
  XOR with the row (`line` 128 bytes of 32 banks, `copy` 16 bytes: CuTe's
  `Swizzle` and Triton's `perPhase`, generalized to rows of any even count of
  chunks); the core reads its fragments back through the same layout. Two
  stages where the renderer copies asynchronously and they fit
  `shared_max`, the next trip's copy issued before this trip's reads and
  ordered so by the slot's offset, an AFTER of the loop's range at two trips;
  one stage otherwise. The barriers are those the implicit rules place. A
  gated operand (PADTO's validity, or a core's selection of 0) copies 0
  where its gate fails, its thread ranges rebuilt from the tile's
  coordinates. A staged core takes its own tile: 2 by 2 warps of 64 by 64,
  the warp tile halved, M then N, to 32 by 32 while the kernel keeps fewer
  than `core_threads / stages` threads, and K unrolled to chunks of 32; its
  rows stay within a run of rows that read one value of an operand, and the
  heuristic takes it only where `Stage.admits` the kernel as the core leaves
  it, so a core staging refuses keeps the unstaged splits. An operand whose
  terms mix the box's ranges with others, a core whose operands step through
  different loops, a kernel with other shared memory, and `Setting.stage`
  off, as the goldens compare with tinygrad, stay unstaged; a tile whose
  address gathers copies in one stage, so that no gather reads past the
  loop's bound.
- **Reason:** (d): GPU time on fudge's RTX 5000 Ada, bfloat16, three
  interleaved rounds, us, before and after: 2048 cubed 187.4-189.4 to
  137.2-137.3, 4096 cubed 1391-1444 to 917-936, gpt-oss-20b's projections at a
  512-token prompt, query (512 by 4096 over 2880) 163.8-165.7 to 120.1-121.7,
  key and value (512 by 512) 77.8-79.9 to 27.5-45.1, gate and up (512 by
  5760) 196.6-197.6 to 130.0-134.2, output (512 by 2880 over 4096)
  167.8-170.1 to 104.4-105.5, and its experts' blocks of 64 rows, gate and up
  (64 by 5760 over 2880) 91.3-92.2 to 29.7-39.9, down (64 by 2880) 57.2-61.4
  to 26.5-26.6. The warp tile's halving is a fit at two stages on fudge: the
  window lies between 5760 and 11520 threads, below the 100 SMs of fudge's
  part, so a part of 142 SMs may leave SMs idle and one of 24 halve tiles it
  could keep. End to end, kaun's `GptOss/prefill-512` ran 199.7-201.3 ms
  on main and 174.3-175.6 ms staged, and `GptOss/experts-at-prefill-512`
  4.17 and 3.81-3.83 ms (three interleaved rounds). AMD gfx1201 and Metal stage in one stage
  of register copies with CUDA's bank constants; correct, unmeasured.
- **Pinned by:** the Stage suite: `values › a staged kernel computes the bits
  of the same optimisations unstaged` (CUDA's renderer against itself
  without shared memory, both tiles, one and two stages, padded, batched,
  `[N; K]` operands, a refused operand), `barriers › …` (every store and load
  of an element separated by a barrier over two trips, on CUDA and on HIP's
  RDNA3 and RDNA4, and every barrier of a kernel with `cp.async` waiting for
  it), `races` (`compute-sanitizer`'s race checker in hazard mode on a
  kernel of each depth), `stages` (`Stage.depth`), `a staged core's
  workgroups, threads and trips`, `the layout` and `sources`; the tolk
  bench's `contract/bf16-*` rows, `bf16-512x4096x2880` and
  `bf16-512x512x2880` among them, their baselines pending the scheduled
  re-record.

## D263. CUDA copies 16 bytes from global to shared memory asynchronously

- **tinygrad:** `renderer/cstyle.py` (`CUDARenderer.barrier`,
  `__syncthreads()`), `base_rewrite` (a store is an assignment).
- **tolk:** `Cstyle.cuda` (`copies_async`, `cuda_async_lang`, `cp_async`);
  `Renderer.t.async_copy`, from sm_80.
- **Differs:** a store into shared memory of a 16-byte load from global
  memory prints as `cp.async.cg.shared.global`, a gated load with
  `ignore-src` where its gate fails, which copies zeros (PTX ISA
  §9.7.10.28.3.1), and every barrier of a kernel holding one prints as
  `cp.async.wait_all; __syncthreads()`: a thread waits only for its own
  copies (CUDA C++ Programming Guide §4.12.1). A kernel without one renders
  as tinygrad's. With at most one stage in flight, waiting for all copies
  waits for none too many, so no group is counted.
- **Reason:** (d): the second stage of D262 overlaps the next trip's copy
  with this trip's cores only where the copy does not wait; its rows are
  D262's.
- **Pinned by:** the Stage suite's `barriers` (its second clause), `races`
  and `sources`.

## D264. Withdrawn

A read of 16-bit elements of an operand staged in two stages was a load marked
`ldmatrix` or `ldmatrix.trans`, lowered after the gater to the instruction
CUDA prints, reading a warp's 8-by-8 matrices in one instruction where it now
reads each element narrow. It gained on staged bfloat16 contractions and
nothing end to end. On fudge's RTX 5000 Ada, minima of four interleaved rounds
of each commit against the one before it, its withdrawal costs 1.04-1.18x on
`contract/bf16-*`, 1.18x and 1.22x on `gated/*` and 1.03x on
`stream/key-projection`, while the 512-token prefill takes 122.9 ms with it
gone and 121.9 ms with it and one layer's experts 2.79 ms and 2.80 ms. About
350 lines for no gain on the workload it named did not pay; the loss on those
staged rows is the stated cost.

## D265. No axis a call reads runs as lanes

- **tinygrad:** `codegen/opt/postrange.py` `upcastable_dims` and `apply_opt`,
  which upcast an axis that a call's argument reads, of a body or of code in the
  target's language (a `CUSTOM_FUNCTION` with `fmt`); `codegen/opt/heuristic.py`
  `hand_coded_optimizations`, whose matrix-vector layout upcasts output axes;
  `test/runtime/test_custom_kernel.py` (`test_call_loop_mini_opts`,
  `test_call_loop_split`, expected failures).
- **tolk:** `Postrange.Scheduler.read_by_opaque`, `upcastable_dims` and
  `apply_opt`; `Heuristic.streams`.
- **Differs:** an axis that a source of a call reads, as `i` in `f(&x[i])`,
  `f(i)`, the statement `{0}[0] += 1;` on `&C[i]` or the expression
  `__builtin_popcount({0})` on `x[i]`, is not upcastable, and an upcast of
  it is refused, as is a tensor core in a kernel with a call; the heuristic
  streams no contraction there. tinygrad upcasts such an axis and renders
  the call with a vector where it takes one pointer or value, which C
  compilers reject. The entry's pad, refused on such an axis, went when
  tinygrad gated a pad's effects (18a86504f) and made code in the target's
  language a call, an effect (b63ad7ead): a pad of an axis a call runs along
  is refused there too. So went its expressions, which tolk kept apart as no
  effect: an expression is a call that its reader writes inline, an effect
  like any call, so D271 applies to it.
- **Reason:** (e): under the heuristic, `test_call_loop_mini`'s kernel and
  that code's over 16 `i` fail at the C compiler, as does
  `out[i] = __builtin_popcount(x[i])` as an expression over 16 `i` upcast by
  4 (Clang: "passing 'int4' to parameter of incompatible type 'unsigned
  int'"). An expression's text states no lanes: the target's builtins take
  one value. Each takes one pointer or value for each source; one per lane,
  gated on its point, would keep the upcast, for a kernel no raven library
  builds yet.
- **Pinned by:** the Codegen suite: `calls in a kernel › a call in an
  optimised kernel runs once per element`, `effects run as often as written
  › code in the target's language in an optimised kernel runs at each
  step`. The Postrange suite: `axes a call or custom code reads`, each case
  a call of a body, a statement and an expression, the axes it lists,
  upcastable and not, an upcast refused on such an axis and applied on
  another, a local split applied, and a tensor core refused. The Heuristic
  suite: `streaming › a kernel whose call reads its output axes is not
  streamed`.

## D270. A range of a typed integer splits and pads as a weak one

- **tinygrad:** `uop/ops.py` (`divides`), which divides a `CONST` and no `CAST`
  of one, though `val` (`uop/ops.py` `UOp.val`) reads a cast constant as one
  constant; `codegen/opt/postrange.py` `Scheduler.apply_opt`, which pads only a
  range whose bound is a `CONST`; `codegen/opt/heuristic.py`
  `hand_coded_optimizations`, which gives threads only to such ranges and sums
  only a `CONST` stride; `codegen/simplify.py` (`mark_range_mod`), which splits
  only such ranges; `codegen/simplify.py` (`mark_gated`), which shrinks a range
  only to a `CONST` guard; `codegen/late/coalesce.py` (`memory_coalescing`),
  which reads only a `CONST` index or term as an access's offset.
- **tolk:** `Ops.is_literal`; `Shape.divides`; `Postrange.Scheduler.apply_opt`
  and its tensor core's tiles; `Heuristic.hand_coded_optimizations`, its
  strides, tiles and divisors; `Simplify.pm_split_ranges`;
  `Simplify.pm_simplify_ranges`'s guards; `Coalesce.memory_coalescing`'s
  offsets.
- **Differs:** a constant of a type other than the weak ones is a cast of
  one, as a range of `Int32` has for its bound. tolk reads it as a constant
  wherever these read a range's bound or a stride, tile or divisor of an
  index, so such a range is split, padded, given threads and split by a
  remainder as a range of the weak integer is, and a kernel indexed in
  `Int32` takes the optimisations of the same kernel in the weak integer.
  A range shrinks to a typed guard `r < c`, and accesses at typed constant
  offsets merge into vectors, as with weak ones.
  tinygrad refuses every split of it, and sums no typed stride.
- **Reason:** (e): tinygrad's heuristic raises on a kernel whose ranges are
  of `Int32`, `out[16i + j] = a[16i + j] + 1` over 64 `i` and 16 `j`: its
  checks read the bound through the cast and pass a split that `apply_opt`
  then refuses ("16 can't divide" the bound 16 on the host, "4 can't divide"
  it on CUDA), so the kernel does not compile. It computes no wrong value: the
  failure is a valid kernel refused, which no setting avoids but disabling the
  heuristic. The strides and tiles follow from the same reading: once a cast
  constant is a constant, one read as unknown chose by the index's type, and
  119 of the heuristic's 431 recorded cases, indexed in `Int32`, took other
  optimisations, a matmul upcasting its rows where it upcasts its columns, a
  routed product refusing its tensor core. A typed guard left its range
  whole, and loads at typed constant indices loaded one element each.
- **Pinned by:** the Heuristic suite: `a range of a typed integer is split
  as one of the weak integer` (`Int32`, `Int64` and `Uint32`, on the host
  and on CUDA), `a kernel indexed in a typed integer is optimised as in the
  weak one` (each recorded case). The Postrange suite: `apply_opt pads a range of a typed integer as
  a weak one`, and the golden case `split_int32_axis`, which applies. The
  Simplify suite: `pm_split_ranges › a range of a typed integer splits as a
  weak one` and `pm_simplify_ranges › shrinking › a range of a typed integer
  shrinks to its guard as a weak one`. The Coalesce suite:
  `memory_coalescing › lengths › loads at typed offsets merge as at weak
  ones, for`.

## D271. No axis a call does not read takes threads

- **tinygrad:** `codegen/opt/postrange.py` `convert_loop_to_global`, which makes
  global every output range that each `STAGE` reads, and `Scheduler.apply_opt`,
  which splits any axis into threads and takes a tensor core in a kernel with a
  `CALL`; `codegen/opt/heuristic.py` `hand_coded_optimizations`, which ranks
  such axes for threads; `codegen/gpudims.py` `group_gpudims`, which masks only
  a global store, to the threads whose local indices it misses are 0.
- **tolk:** `Postrange.Scheduler.opaque`, `unread_by_opaque`,
  `convert_loop_to_global` and `apply_opt`; `Heuristic.streams` and the
  heuristic's local splits.
- **Differs:** an axis that the sources of some call, of a body or of code in
  the target's language, do not read stays a loop: it is not made global, a local split of it (a group,
  for a reduce axis) is refused, and the heuristic gives it no threads. A
  tensor core, whose warp is threads along axes no call reads, is refused in
  a kernel with a call, and the heuristic streams no contraction there.
  tinygrad gives such an axis threads, and every thread runs the call. An
  expression in the target's language is a call too (b63ad7ead), whose
  reader writes it inline.
- **Reason:** (e): `n[0] += 1` by a call or by code, beside `out[i] = a[i] +
  1` over 64 `i` on CUDA, runs in each of the 16 threads the heuristic
  gives `i`, and beside a sum over 4096 `r` in each of the 16 threads of the
  group it makes: `n` counts 16 for 1. A call can write anything, so masking
  it to one thread of each such axis, as a global store is, would keep the
  threads only where no other thread reads what it writes, and drop what it
  writes to a thread's own registers in the others.
- **Pinned by:** the Codegen suite: `calls, custom code and stores on CUDA ›
  one beside stores across threads runs once`, `› one beside a sum runs
  once` (a call and code, counting atomically; they skip without a GPU). The
  Postrange suite, each case a call of a body, a statement and an
  expression: `axes a call or custom code reads ›
  unread_by_opaque is the axes its sources do not read`, `›
  convert_loop_to_global gives workgroups only to an axis its sources read`,
  `› apply_opt refuses a local split of an axis its sources do not read`,
  `› a group is refused over a reduce axis its sources do not read`, `› a
  tensor core is refused in a kernel with one`. The Heuristic suite: `a
  kernel whose call reads no axis takes no threads`, on CUDA's renderer.

## D275. Clang compiles with LLVM's IR verifier on

- **tinygrad:** `runtime/support/compiler_cpu.py`
  (`ClangCompiler.compile`, Clang's driver with its defaults, which in a
  release build pass `-disable-llvm-verifier` to the compiler proper).
- **tolk:** `Compiler_cpu.cc1`, which drops `-disable-llvm-verifier` from
  the command Clang states, and `Compiler_cpu.clang`'s `table`, which
  digests the command run (D113).
- **Differs:** the compiler proper runs with the verifier on: it checks the
  IR Clang makes of the source before the optimizations, and again before
  instruction selection, and invalid IR stops the compile with a diagnostic
  that names the function, which `Renderer.Compiler.Compile_error` carries.
  Clang's release notes put the verifier's cost at "a few percent". On
  fudge (Clang 19.1.7, the host's arguments, cachegrind's instruction
  counts) it costs 0.5-2.2% of a kernel's compile: 1.1% for `trig/sin`'s
  kernel, 2.2% for a `scan/checks-48` kernel, 0.5% for a
  `model/schools-start` kernel, 1.5% for `rng/normal-f64`'s, 1.9% for
  `special/betainc-chain-f32`'s and 1.5% for
  `special/betaincinv-grad-f64`'s, of 600 M to 25 G instructions. The
  verifier after every pass (`-llvm-verify-each`) cost 6.6-36% on the same
  kernels and names the pass, not the function; both cases below survive
  to instruction selection, where the verifier above stops them.
- **Reason:** (e): an optimization that writes invalid IR goes unseen by a
  release Clang. LLVM 19's straight-line vectorizer wrote `trunc <4 x i8> ... to
  <4 x i32>` for rune's routed rows (D167), which compiled into garbage row
  numbers; LLVM 18's wrote `sext <4 x i32> to <4 x i16>` for `r_3_2`, which
  crashed instruction selection. D167 turns that vectorizer off for those
  releases and LLVM 20, whose wrong values were valid IR; other Clangs and
  Apple's are unchecked, and with the verifier on, a defect of their passes of
  that kind is a compile error instead of wrong values. With the verifier, Clang
  18 rejects `r_3_2` naming it, and Clang 19 `E_16_2_4`. tinygrad's own LLVM
  compiler verifies its input and runs the verifier after every pass
  (`runtime/support/compiler_llvm.py` `LLVMCompiler.__init__`,
  `LLVMCompiler.compile`).
- **Pinned by:** `Tolk.Compiler_cpu › invalid IR › a source that LLVM 19's
  straight-line vectorizer makes invalid raises Compile_error naming its
  function`, under a Clang of LLVM 19 with that vectorizer turned back on.
## D280. An effect keeps its ranges when its index folds

- **tinygrad:** `codegen/__init__.py` (`full_rewrite_to_sink`, which simplifies
  a kernel's indices and ranges with nothing holding an effect's ranges),
  `uop/symbolic.py` `symbolic` (a value whose bounds meet folds to a constant)
  and `uop/ops.py` `UOp._ranges`, `UOp.ranges` (a node depends on the ranges its
  sources depend on).
- **tolk:** `Simplify.pin_effects`, called by
  `Codegen.full_rewrite_to_sink` after the movements are lowered, and
  `Simplify.pm_unpin_effects`, in its symbolic pass before the expander.
- **Differs:** before its indices are simplified, an index of storage that
  an effect (a store, a call, code in the target's language) indexes reads
  it through an `After` of the ranges its offsets depend on, and an effect
  that depends on ranges its index does not, through an `After` of those
  too. Once opts and range shrinking are done, before the expander, such an
  `After` of ranges alone goes where the index or effect depends on them
  without it, so it stays only where a rewrite removed a range: a fold
  (`C[r // 8] += 1` over `r < 8`), or a range shrunk to a guard that then
  folds (`C[r // 8]` where `r < 8`, over `r < 32`). tinygrad folds the
  index to `0`, and the effect, which no longer depends on `r`, runs once:
  it counts 1 where 0, 5 or 8 is wanted, and the store writes in a loop
  that runs no trip (run at `6b7b018da` and at the pinned commit). An
  `After` of ranges alone that a builder wrote goes too where the effect
  depends on them without it, which places nothing differently:
  `reg_placeholder_then_reduce`'s golden reads its register directly where
  its store's value depends on the range. Not covered: an effect with no
  index, and a rewrite after the expander. The expander drops an `After`'s
  upcast ranges, so lanes of a range an effect reads only through the
  `After` would run it once: such a range, if weak, breaks its promise,
  which a store whose cells show it is refused for before any optimisation
  (D333), and a serial one takes no lanes.
- **Reason:** (e): an effect that runs once per step of its loops as built
  runs once, or runs in a loop of no trip.
- **Pinned by:** the Codegen suite's `effects at an index that folds run as
  often as written` (the counts of `C[r // 8] += 1` at 0, 5 and 8 trips, a
  store in a loop of no trip, a variable's and a flag's, a stored value
  that folds, a guarded quotient whose range shrinks, code in the target's
  language, and a law over loop nests of indices that fold);
  `reg_placeholder_then_reduce`'s goldens, and the kernels the Heuristic,
  Postrange, Search and Simplify suites record where their passes receive
  them, which carry the `After`s.

## D285. A loop holding a barrier or a matrix instruction has a uniform bound

- **tinygrad:** `codegen/__init__.py` (`full_rewrite_to_sink`),
  which places the implicit barriers and lowers the ranges to hardware
  indices with no check of the loops a barrier runs in;
  `codegen/opt/postrange.py` (`apply_opts`), which runs the
  heuristic, or BEAM, on any kernel without a `STAGE`.
- **tolk:** `Uniform.varying`, `Uniform.check` and its call in
  `Codegen.full_rewrite_to_sink` after `pm_implicit_barriers`;
  `Postrange.apply_opts` (`threads_fixed`).
- **Differs:** a kernel whose barrier, matrix multiply-accumulate, or call
  of a body holding one, runs in a serial loop whose trips may differ
  between a workgroup's threads is refused when lowered, naming the kernel,
  the operation and the loop, under `SPEC=0` too. Neither the heuristic nor
  a search optimises a kernel that holds, or calls a body that holds,
  workgroup or register storage, a barrier, a matrix multiply-accumulate or
  code in the target's language; the optimisations such a kernel lists still
  apply.
- **Reason:** (e): a barrier some threads skip, or reach fewer times, hangs
  the workgroup or reads workgroup memory not yet written (CUDA C++
  Programming Guide §7.6, PTX §9.7.15.1 `bar`; non-termination is (e)), and
  tinygrad compiles a barrier in a loop bounded by a thread index without a
  word. Its heuristic splits a hand-made kernel's axes into threads and
  lanes while the kernel's workgroup and register buffers stay sized for
  one workgroup and one thread: on CUDA, a kernel reading its neighbour's
  staged value computed 224 of 256 values wrong, and one whose staging ran
  in a loop of one trip per flagged workgroup got that loop's bound from
  `threadIdx.y`, around a barrier, which once hung the GPU.
- **Pinned by:** `Tolk.Uniform` (`varying`, `check`); `Tolk.Postrange`'s
  `apply_opts › neither hand-optimises nor searches a kernel whose builder
  fixed its threads with` and `› applies the optimisations a kernel whose
  builder fixed its threads asks for`; `Tolk.Codegen`'s `kernels whose
  builder fixed their threads`, on CUDA; and the tolk, rune, jera and kaun
  suites, which lower every kernel through the check.

## D300. A streamed thread loads a pair of float32 values at once

- **tinygrad:** streams nothing (D89); its `codegen/opt/heuristic.py`
  `hand_coded_optimizations` unrolls the last reduce range whole when it is at
  most 32 and few lanes are upcast, which the stream's own unroll leaves no room
  for.
- **tolk:** `Heuristic.stream`'s rows layout (`unroll_together`,
  `Heuristic.vector_bytes`, `Heuristic.line_bytes`).
- **Differs:** after the lanes and the unroll of the loads in flight, each
  contracted range that no split has taken, other than the lanes', and
  along which the other operand is read at unit stride, is unrolled whole
  where its values there fit one 16-byte load and 32 lanes, stepping the
  other operand by its stride along the lanes' range, load across more
  than a 128-byte line. A byte's two MXFP4 values multiply the other
  operand at `2 b` and `2 b + 1`: in a loop of 2 trips 32 lanes load
  float32 values 8 bytes apart, over two lines a load with half of each
  unused; unrolled, each thread loads its pair as one 8-byte vector.
  bfloat16 activations, whose 32 loads 4 bytes apart span one line, keep
  the loop. A refused split is skipped, as the in-flight unroll's is.
- **Reason:** (d): GPU time on fudge's RTX 5000 Ada, three interleaved
  rounds under the timing lock: gpt-oss-20b's gate and up projection,
  float32 activations routed in blocks of 64 rows at 512 tokens (kaun's
  `Quant/cuda/routed-product-512-tokens`), 24.4-25.9 to 12.6-12.9 ms; 64
  float32 rows of a 16384-square MXFP4 matrix
  (`Quant/cuda/mxfp4-product-of-16384-64-float32-tokens`), 4.8-5.0 to
  2.7 ms; bfloat16 rows' kernels unchanged. Unrolled for bfloat16 too,
  12 and 17 rows ran 1.1-1.15 times faster and 16 and 32 rows 1.1-1.2
  times slower, with no spills either way, so bfloat16 keeps the loop
  until a rule separates them. AMD and Metal take the same rule,
  unmeasured.
- **Pinned by:** the Heuristic suite's `streaming › no loop is left over
  the other operand's values of one load`, over the recorded streamed
  kernels, with the layouts in the suite's `.tolk` files (`gpt_oss_gate_up`, `gpt_oss_down`, `mxfp4_m12`,
  `routed_blocks`, `vecmat_decoded`); kaun's
  `Quant/cuda/mxfp4-product-of-16384-64-float32-tokens` and
  `Quant/cuda/routed-product-512-tokens`, whose fudge baselines (the first
  recorded before D300, the second not yet recorded) wait for the
  scheduled re-record.

## D320. The CUDA device's kernels compile to a cubin

- **tinygrad:** `renderer/cstyle.py` (`CUDARenderer.__init__`), which
  gives `NVRTCCompiler` `ptx=True` for the device `CUDA`;
  `runtime/support/compiler_cuda.py` (`NVRTCCompiler.compile`), which
  then returns PTX; `runtime/ops_cuda.py` (`CUDADevice.function`),
  whose `cuModuleLoadData` translates it for the GPU.
- **tolk:** `Cstyle.cuda`, which makes `Compiler_cuda.nvrtc ~ptx:false` for
  every device.
- **Differs:** NVRTC compiles the CUDA device's kernels to a cubin of the
  device's architecture, as it does the NV device's, and the disk cache
  keeps the cubin. The driver loads it as it is, where it would translate
  PTX at each load and keep its work only in its own cache, which a later
  process may miss: the cache is bounded and per user, keyed by the driver,
  and off under `CUDA_CACHE_DISABLE`. NVRTC is given the device's
  architecture in either form, and the disk cache keys binaries by it, so a
  binary is only loaded on a GPU of that architecture; one NVRTC does not
  know is refused in both. The GPU's code is that of NVRTC's assembler
  rather than the driver's, which differ where their versions do; PTX's
  semantics and `--fmad=false` fix the values either way.
- **Reason:** (d): a later process's first call, every binary read back
  from the disk cache and the driver's cache missing, on fudge's RTX 5000
  Ada: a two-layer gpt-oss-20b at its own widths (vocabulary 4096), its
  12-token prefill, 710-1050 ms with PTX against 58-131 ms with cubins (3
  interleaved runs each), the driver translating 43 kernels; rune's
  `cuda/sin-nested-8` kernel loads in 40.7 ms as PTX and 0.19 ms as a
  cubin. With the driver's cache warm,
  the prefill's 74 loads take 6.3 ms of CUDA's time as PTX and 4.2 ms as
  cubins, and its call allocates 6.97 M words against 6.95 M. A cold
  compile of `cuda/sin-nested-8`, interleaved, takes 293-408 ms with PTX
  and 255-299 ms with cubins (medians of 5: 345 and 294 ms): the cubins
  D78 found slower, in whole calls timed under load, were not.
- **Pinned by:** the Cstyle suite's `CUDA's binaries › the CUDA device's
  kernels compile to a cubin`; rune's compile bench rows
  `cuda/sin-nested-8` and `warm/cuda/sin-nested-8`, whose processes run
  with the driver's cache off, and where PTX adds the driver's 40 ms;
  their wall-clock baselines are missing until fudge's scheduled
  recording of every suite.

## D305. A reduce under a uniform gate runs its loop only where the gate holds

- **tinygrad:** `codegen/__init__.py` (`full_rewrite_to_sink`),
  which runs the post-optimisation symbolic pass and then the expander: a
  reduce whose result a gate selects, or whose store a gate guards, runs
  every trip of its loops in every workgroup, where the gate fails too.
- **tolk:** `Codegen.lift_uniform_gates`, after the post-optimisation
  symbolic pass, through `Uniform.varying`; `Shape.fold_where_closure`,
  whose assumption leaves each range whole (`Ops.assuming`).
- **Differs:** a value computed by casts and arithmetic from one reduce,
  which `where (g, v, e)` selects where `g` holds (or where it fails) or a
  store through `where (g, i, Invalid)` writes, gets the reduce's outermost
  serial loop bounded by `where (c, n, 0)`: `c` the conjunction of the
  clauses of `g` that are the same for every thread of a workgroup and read
  only ranges the reduce runs inside. A workgroup whose clauses fail skips
  the loop and its loads; the reduce's identity it then holds is never
  read. Each arm of a selection is bounded alone. The rewrite declines when
  no clause is uniform, when the reduce has no bounded serial loop, or when a
  node of the reduce that computes from the loop is read outside it, as by a
  second reduce over the loop.
  rune's routed product (`Lower_reduce`, blocks of 64 rows of one group)
  states a block of padding as `Invalid`, which the gated store carries.
  A selection's condition, assumed within its arms, stops at a range: the
  lifted bound holds the condition, and rewriting the bound inside an arm
  would fork a second loop of the range's identity.
- **Reason:** (d) rune's routed product, whose blocks past the groups' hold
  only padding rows that no row reads. Measured on the kernels as they now
  stand, minima of four interleaved rounds: kaun's
  `GptOss/experts-at-prefill-512`, 3.29 ms without the rewrite and 2.80 ms
  with it, and `GptOss/prefill-512` 133.5 and 121.7 ms; these end-to-end rows
  are what earn its code. And the tolk bench's
  `gated/bf16-2048-5-of-16-blocks-off`, a 2048-cubed bfloat16 product on a
  staged tensor core whose gate leaves 5 of its 16 blocks of 128 rows
  without a value, 119.5-120.0 µs to 82.8-83.5 µs, and
  `gated/bf16-2048-all-on` 120.8-121.9 µs to 119.8-120.7 µs; fudge's RTX
  5000 Ada, three interleaved rounds each. The rows' wall-clock baselines
  wait for the scheduled re-record. Leaving a range whole in
  `fold_where_closure` is part of that rewrite: tinygrad makes no bound that
  holds a condition, and without it a lifted kernel with a load gated by the
  loop's own clause does not compile.
- **Pinned by:** `Tolk.Codegen`'s `reduces under a uniform gate` (values
  equal to the reference on the host and CUDA with every block off, every
  block on, one off and any counts; the bounded loop in the lowered
  program and its source; two sums over one loop and a gate that varies
  between threads keep the loop whole; a sum gated by a clause its loads are
  gated by too, either side of it; a loop the gate of its loads bounds stays
  one loop on every target); `Tolk.Symbolic`'s selections (`the condition is
  not assumed in the bound of a loop in a branch`); the goldens `uniform_gate_reduce`, `uniform_gate_store`
  and `varying_gate_reduce`; rune's `rows in blocks give eager's integers and
  hold their bound`; the bench rows above.

## D315. A global store is masked by the workgroup indices it does not use

- **tinygrad:** `codegen/gpudims.py` (`group_gpudims`), which masks a
  global store to the threads whose local indices its index misses are 0,
  and runs it in every workgroup.
- **tolk:** `Gpudims.group_gpudims`.
- **Differs:** a store to global memory whose index, its validity included,
  does not use a workgroup index is masked to the workgroups whose unused
  indices are 0, as it is to such threads. tinygrad runs it in every
  workgroup.
- **Reason:** (e): `n[0] = n[0] + 1`, which no range holds, beside `out[i] =
  a[i] + 1` over 2^20 `i` on CUDA, runs in each of the 8192 workgroups the
  heuristic makes, and a workgroup that starts after another's write reads
  it: `n` counts 10 to 13 for 1. A store to global memory whose index does
  not read a thread range a builder wrote runs in the thread where that
  range is 0, as `Ops.Axis_type` defines it; under a weak range made
  global, such a store breaks the range's promise (D333).
- **Pinned by:** the Codegen suite: `calls, custom code and stores on CUDA ›
  a plain store beside stores across workgroups runs once`, `› a store
  whose folded index is bound to a loop runs once a step`, its storage read
  through an `After` of the loop (D280), `› a store valid at one point
  runs there` (they skip without a GPU). The Gpudims suite:
  `group_gpudims_masks_a_store_by_its_missing_global.tolk`.

## D325. Withdrawn

Clang selected a 16-bit float by its bits; merged into D27, which holds its
rule, its reason and its pins.

## D330. The kernels of one write are unordered whichever Afters list them

- **tinygrad:** `schedule/__init__.py` (`create_schedule`'s
  write-after-read edges), which orders a kernel reading a state before the
  other kernels of every `AFTER` over that state but the `AFTER` the read
  was listed under (`if a is u: continue`), and raises "cycle detected in
  assign graph" where those edges form a cycle. One kernel per store
  (`schedule/rangeify.py` `split_store`) lists each kernel under
  one `AFTER`, so its stores never meet the case.
- **tolk:** `Schedule.schedule`.
- **Differs:** a kernel reading a state runs before the kernels of every
  write over it of which it is no part. A kernel that two `AFTER`s list, as one whose stores
  share their loops (D137) into `p` and `q`, reads `q` as the store it makes
  there, beside the other kernels of `q`'s write, and is ordered before none
  of them, as it would be listed under that write alone. tinygrad orders it,
  through its read listed under `p`'s write, before them.
  A launch of the kernels of one write holds them apart unless the write is
  marked disjoint (D405).
- **Reason:** (b): rune's split program `[a; b] - [b; a]`, two assemblies of
  the same computed pieces stored into their regions of two buffers, makes
  two kernels, each storing a column of both. Each was ordered before the
  other, and no copy cuts the cycle, since each read is the kernel's own
  store; the program was declined, and over a set of GPUs, which have no
  eager kernels, it raises.
- **Pinned by:** the `Tolk.Schedule` suite: `create_schedule › laws › a
  schedule copies a state iff its kernels form a cycle`, whose writes may
  have two kernels that two Afters list, and `create_linear_with_vars ›
  writes › stores crossed into two buffers write both`; rune's `Jit_devices
  › assemblies give eager's bits › crossed assemblies of computed pieces`.

## D331. A reduce of one trip or none, and a range of no trip, compile

- **tinygrad:** `codegen/simplify.py` (`reduce_unparented`), which
  asserts that every source of a reduce past its value is a range, where
  `uop/symbolic.py` `symbolic` folds a range of one trip to its one value, a
  constant; `codegen/simplify.py` (`merge_ranges`), which merges a
  range of size 0 into a range of the size it has, the same node, and
  substitutes its quotient for it; `codegen/opt/heuristic.py`
  (`hand_coded_optimizations`), which optimises a kernel that runs no trip
  of a range as any other.
- **tolk:** `Simplify.pm_reduce_unparented`, `Simplify.pm_simplify_ranges`
  and `Heuristic.hand_coded_optimizations`.
- **Differs:** a constant source of a reduce, a range of one trip folded,
  binds nothing, and the reduce drops it, a typed integer's constant (a
  cast of one) too: a sum over one trip is its one value. Two ranges merge
  only into a range other than the first, so a first range of no trip stays
  whole. A kernel holding a range of no trip
  takes no hand-coded optimisation.
- **Reason:** (e): tinygrad raises on kernels built by hand that compute a
  value. `out[i, j] = Σ_k a[i, k] b[k, j]` over one `k` raises "some
  reduce srcs aren't ranges", with or without optimisations; over no row
  `i`, `graph_rewrite` stalls, its substitution reaching itself; over no
  `k`, the heuristic's upcast of `j` by 4 merges loads of `b`, which holds
  no element, into a vector past its end ("invalid shrink"). tolk raised the
  same, and on CUDA its streamed product found no contracted range, and on
  the host over no column an upcast of rows put two lanes' stores at one
  address ("attempting multiple stores"). A kernel that runs no trip of a
  range does no work along it to share among lanes or threads: an
  unoptimised one writes the same values, a sum over no trip its identity.
- **Pinned by:** the Codegen suite's `kernels of few trips › on the host,
  write what they compute node by node` and `› on CUDA, ...`, over products
  of 0 to 8 rows and columns and of 0 to 16 trips, sums, maxima and
  products, gated or not, with and without the heuristic, against the
  interpreter, their indices in the weak integer or `Int32`; the Simplify
  suite's `pm_reduce_simplify › a sum over a range of size 1 is its value
  where the range is 0` and `› a sum over a typed range of size 1 is its
  value where it is 0`.

## D332. A call's arguments lose their indices once ranges are applied

- **tinygrad:** `schedule/rangeify.py` (`no_indexing_calls`: "sometimes
  if call srcs have children the call will get an INDEX. we remove it
  here"), applied at `get_kernel_graph`, after `pm_limit_bufs` has staged
  values: `stage` and `fix_store_hazard` index a
  stage by `s.ranges`.
- **tolk:** `Rangeify.get_kernel_graph`, which applies
  `pm_no_indexing_calls` right after `Indexing.run_rangeify` too.
- **Differs:** a movement of storage that a call takes and another kernel
  reads is one node, which applying ranges indexes by that kernel's ranges,
  for the call too. Until the call drops the index, those ranges are among
  the ranges of every value that reads the call's results. A store that
  reads its destination at another index stages its value over its
  ranges, so the stage was indexed by the other kernel's ranges, which no
  kernel ends, and tinygrad's kernel-graph check refuses the graph. tolk
  drops the index before any pass reads the ranges.
- **Reason:** (b): norn's `evidence › a conjugate normal`, a jit of a vmap
  of SMC, which stores a value gathered from its destination beside a
  loop's call whose argument another kernel reads, failed the check and
  raised.
- **Pinned by:** `Tolk.Rangeify › get_kernel_graph › recorded ›
  assign_flipped_beside_call_1_kernels.golden` and the kernel-graph laws
  over `assign_flipped_beside_call_1`: `x.assign(x.flip(1) + f(a))` beside
  `a * 3`, `f` a function compiled on its own; norn's `Norn.Smc › evidence ›
  a conjugate normal`.

## D333. A weak range is its builder's promise, refused where a store breaks it

- **tinygrad:** `uop/ops.py` (`AxisType`, whose `WEAK` states
  nothing of its trips' independence); `codegen/opt/postrange.py`
  (`apply_opts`, which optimises a kernel whatever its stores);
  `shift_to` (which splits a range whatever its bound reads, by
  any amount that divides it).
- **tolk:** `Ops.Axis_type.Weak`, which states the promise;
  `Postrange.apply_opts`, which refutes it before any optimisation
  (`refute_weak`); `Postrange.Scheduler.split_size`.
- **Differs:** a weak range is its builder's promise that two points of the
  kernel whose values of it differ never meet at a cell of global or
  workgroup memory that one of them writes, so its trips may become lanes,
  threads or host blocks; a value carried from trip to trip needs a `Loop`.
  Before any optimisation, a store to global or workgroup memory raises
  where it names fewer cells than it has points along one of its weak
  ranges, or along all of them together, its validity and gate reading none
  of them: an offset's cells are the multiples of the greatest common
  divisor of the constant factors of its terms that read the ranges,
  between those terms' least and greatest sums, and a range's points are
  counted over every value of the variables. Only a range whose bound is
  a constant or a variable is counted, its points at the bound's greatest
  value, which some launch reaches for every such bound at once. A range
  whose bound reads memory or another range is not known at launch, and no
  split makes it lanes or threads. tinygrad optimises such kernels as any
  other. Silent, as broken promises whose
  result is unspecified: a store whose validity or gate reads the ranges;
  two stores whose cells meet across points (`A[r] = 1; A[r + 1] = 2`); a
  read at a cell another point writes (`C[r + 1] = C[r] + 1`); cells that
  memory names; the effects of calls and custom code; a store whose cells
  are few only at some value of the variables (`O[r // m]`). A range whose
  points another range's bound restricts can be refused falsely
  (`O[max(i, 1) + 4 j]` over `j < i`).
- **Reason:** (e): at the pinned commit, `C[r // 8] += 1` over a weak
  `r < 16` upcast by 4 counts 2 a cell where 8 is wanted, and on a GPU its
  threads race; `C[i] += 1` over `j < 4 (i + 1)`, `j` upcast by 4, counts
  `[1; 2]` for `[4; 8]`.
- **Pinned by:** the Codegen suite's `weak ranges` group: the refusals of
  `C[r // 8] += 1` over a weak `r < 16` on lanes, in host blocks and on
  threads, and over `r < 8`, of `C[r + s] += 1`, `C[0] += 1` after `r`, `O[0] = r`,
  `O[r // 8] = 5`, `C[16 (r // 8)] += 1`, `O[2i + 2j] = 1` and workgroup
  memory staged under a weak range; the counts of the same stores over
  serial ranges; the kernels it does not refuse (a packed triangle, a store
  gated or valid at one point, loaded and variable bounds, bounds that share
  a variable, a range of no trip, a scatter to cells memory names); and the
  refused upcast of `j < 4 (i + 1)`.

## D335. A long arange is laid out in rows, typed by its last element

- **tinygrad:** `mixin/op.py` (`arange`): its type is checked and
  chosen from `start` and `stop - step`, and its elements are
  the running sum (`_cumalu`, `OpMixin._cumalu`) of `n` steps, whose pool
  (`mixin/movement.py` `_pool`) repeats the input, `2n - 1` elements
  padded, to `(n + 1)(2n - 1)` elements before it cuts the windows.
- **tolk:** `Shape.arange` and its `run`.
- **Differs:** up to 1518500249 elements, the most whose pool's views fit
  an OCaml `int`, an arange is tinygrad's running sum. A longer one of `n`
  elements lays them out in rows of `b`, the power of two with
  `b² <= n < 4b²`: an element is its row's first element, an arange by
  `b * step`, plus its offset along the row, an arange by `step`, each
  built the same way. A float arange
  that long is laid out in `Int64` and cast, so each element is its integer
  rounded once. An arange of more than `max_int - 2^30` elements raises.
  The type of a nonempty arange is checked and chosen from `start` and its
  last element, `start + (n - 1) * step`, where tinygrad reads
  `stop - step`, and its length and bounds are computed exactly.
- **Reason:** rows: (a): a shape is an OCaml `int`, 63 bits, where
  tinygrad's is a Python integer, and the pool's views past `max_int`
  elements raised. (b): rune's `Unique`, which numbers its rows by an
  arange, lowers up to `2^31` rows and raised past 1518500249. A row and a
  column of about `sqrt n` elements keep both aranges within an int's
  views, and their kernel is elementwise. Rows summed in a float type would
  round each part, and `Float32` elements were then off by hundreds near
  `10^10`. Bounds: (e): when `stop - start` is not a multiple of `step`,
  the last element passes `stop - step`, and `arange ~dtype:Int8 ~step:2
  129` wrapped its last element, 128, to -128, as tinygrad does.
- **Pinned by:** the Ops suite's `aranges › an arange's element i is start +
  i * step at any length`, over integer and float types; `movement ›
  arange rejects more than max_int - 2^30 elements`, `› arange counts its
  elements past an int's range` and `› arange's type holds its last
  element, past [stop - step]`; the Tolk_engine suite's `aranges › an
  arange past 2^31 elements holds start + i * step`, on the host and on
  each GPU; rune's `Lower_index › numbering › a Unique of 2^31 rows is
  lowered`.

## D338. Withdrawn

Replaced by D404, one pass over the finished schedule, and D405, the mark an assembly's maker states.

## D340. Lanes keep registers apart and threads keep the code around them

- **tinygrad:** `codegen/opt/postrange.py` (`convert_loop_to_global`,
  which makes a weak output range global whatever the code around it);
  `shift_to` (which splits a range into lanes whatever its register
  stores read); `upcastable_dims`.
- **tolk:** `Postrange.Scheduler.threadless`, which
  `convert_loop_to_global`, `shift_to` (and the splits of `apply_opt`) and
  `Heuristic`'s local splits read; `shift_to` and `upcastable_dims` for
  lanes; `Shape.written_params`.
- **Differs:** the lanes of a range share one thread's registers, so a range
  takes no lanes where a register store's cells, value or gate, or the
  condition a backedge repeats on, depends on it in data, a register load by
  its offsets and gate alone, an `After` by its first source, a reduction by
  its value less the ranges it closes. Under the weak promise a step reads a
  register cell only after writing it, or one no step writes, so a register
  store that depends on no lane writes what each step would. Threads run
  unordered with the code outside their range, and an effect that does not
  depend on a thread range runs in one of its threads, so a weak range takes
  no threads (it stays weak, and a local split of it is refused) where an
  effect depending on it and one not depending on it access one storage of
  global or workgroup memory, one of them writing it, or the second reads a
  register the first writes. Effects are stores, calls, which write the
  arguments their body writes, and code, which writes every pointer it is
  passed; a body lowered to lines inside a kernel writes what its stores
  write, where tolk read it as writing nothing. tinygrad splits and makes
  global such ranges as any other.
- **Reason:** (e): over a weak `r < 8`, `R[0] = A[r] + 1; D[r] = R[0]`
  upcast by 4 stores `[4; 4; 4; 4; 8; 8; 8; 8]` for `[1; …; 8]`, on CUDA
  too once `r` is global, and `R[r // 4]` the same; `S[0] = T[r]` over a
  table each step fills stores `[0; 0; 0; 0; 4; 4; 4; 4]` for `[0; …; 7]`;
  a counter repeated while below `r + 1` or `T[r]` fails verification. On
  CUDA, `G[r] = A[r]` over 65536 steps then `D[0] = G[65535]` reads 0 for
  65535, by a store, a call or code; `R[0] = A[r]` then `D[0] = R[0]` reads
  0 for 7; `G[0] = 5` before `r`, read at every step, reads 0 in some
  workgroups.
- **Pinned by:** the Codegen suite's `lanes and threads` group: the register
  kernels under a split, against their serial values, and the refusals of
  `R[0]`, `R[r // 4]`, `R[2x + y]` and `R[r]; R[r + 1]` scratch, a table
  read at `T[r]` or through an index valid where `r < 4`, a store valid
  where `r < 4`, counters to `r + 1` and `T[r]`, and a register written per
  step beside a reduction, upcast; the axes `upcastable_dims` leaves out;
  scratch over a range made global; the
  splits kept (a counter to 3, a register written from another, a register
  ordered after a store, a register beside or holding a reduction, its
  range unrolled); the ranges given no threads on CUDA under the heuristic
  and their values (reads after, a register read after, a write before,
  writes by a call and by code), the local splits refused, the range a
  read-only call keeps global, and a lowered body's writes.

## D341. A range states its axis type

- **tinygrad:** `uop/ops.py` (`UOp.range`, whose `axis_type` defaults
  to `AxisType.WEAK`).
- **tolk:** `Ops.range`.
- **Differs:** `Ops.range` takes `~axis_type` with no default, so each
  builder states whether a range's trips are independent (`Weak`, D333) or
  run in turn (`Loop`). tinygrad makes a range weak where its builder names
  no type.
- **Reason:** (e): a default promises independence that the builder never
  stated. At the pinned commit `C[0] += 1`, read through `After (C, [r])`
  over an untyped `r < 8` upcast by 4, counts 2 for 8; tinygrad's kernels
  `slice_sum` and `simple_qkv` carry a register across untyped ranges,
  which tinygrad's own tests type `LOOP` from `78d482262`.
- **Pinned by:** the type of `Ops.range`: a call without `~axis_type` does
  not compile.

## D342. An integer operation is defined where C leaves it undefined

Three parts of one rule, `Cstyle.pm_wrap_signed` and
`Cstyle.pm_defined_div_shift`: tolk's integers wrap at their width,
`Ops.exec_alu` states every value, and none of these operations is left to
C's undefined behaviour.

### A signed operation that may overflow computes in its unsigned type

- **tinygrad:** `renderer/cstyle.py` (`code_for_op`, which writes `+`, `-`, `*`,
  a unary `-` and `<<` whatever the integer type); `pm_narrow_uint` (for an
  unsigned char or short alone); `runtime/support/compiler_cpu.py`
  `ClangCompiler.compile` (Clang, without `-fwrapv`); `codegen/decomp/op.py`
  `get_late_rewrite_patterns` (a product by a power of two becomes a left shift,
  of a signed operand too).
- **tolk:** `Cstyle.pm_wrap_signed`, in the extra matchers of
  `Cstyle.clang`, `Cstyle.cuda`, `Cstyle.hip` and `Cstyle.metal`.
- **Differs:** C computes an operation on a signed integer in `int`, or in
  `long long` for a 64-bit one, and leaves its overflow undefined, as it
  does a left shift of a negative value; CUDA, HIP and Metal follow C++,
  which does the same. A compiler takes either for impossible: NVRTC
  compiles `(x + y) < x` of int32 as `y < 0`. tolk's integers wrap at their
  width (`Ops.exec_alu`). An `Add`, `Sub`, `Mul`, `Neg`, `Shl` or `Mulacc`
  of a signed type computes in the unsigned type of `int`'s or `long
  long`'s width, its operands converted, and converts back, unless its
  operands' bounds (`Ops.operand_bounds`) keep its exact result in that
  type and, for a shift, the shifted value at or above 0 and the count
  below the width. An index, whose bounds chose its width, keeps tinygrad's
  source; a value loaded, or decomposed from a float's bits, does not. The
  conversion back of a value past the signed type keeps its low bits on
  every compiler tolk targets, as D17's narrowing does. `-fwrapv` would
  cover Clang alone: NVRTC's library holds no such option. On fudge,
  the kernels of the bench rows it changes (`host/sin_nested_8`,
  `host/normal_f64`, `gpt_oss/gpt_oss_decode_values`, `stream/decode-keys`,
  `stream/routed-argmax-16-bf16`) compile to the same machine code, host
  and SASS, as before; the stages that linearize and render a kernel with
  such operations allocate up to 14% more words (`embedding/render`), for
  the casts.
- **Reason:** (e): at the pinned commit, a CUDA kernel computes
  `x + y < x` of int32 as `y < 0`, true where `x + y` wraps below `x`.
- **Pinned by:** the Tolk_engine suite's `signed arithmetic wraps as the
  interpreter`, on the host (`kernels on the host`) and on each GPU
  (`GPUs ›`, run on fudge's CUDA and NV): each operation of each signed type
  over its extremes' pairs, its comparison with its first operand, a
  product by 4, which becomes a shift, and a sum and a shift of a value
  masked to its non-negative bits, whose bounds are narrowed but still let
  them overflow; rune's `jit › integer arithmetic at its type's edges gives
  eager's values`, shifts included, and its GPU suite's `integers ›` the
  same law.

### A division that may overflow and a shift past the width are defined (D346)

- **tinygrad:** `renderer/cstyle.py` (`code_for_op`, which writes
  `/`, `%`, `<<` and `>>` whatever their operands).
- **tolk:** `Cstyle.pm_defined_div_shift`, in the extra matchers of
  `Cstyle.clang`, `Cstyle.cuda`, `Cstyle.hip` and `Cstyle.metal`.
- **Differs:** C leaves undefined a signed division of the least value by -1,
  its quotient's and its remainder's, and a shift by a count at or past the
  width of the promoted operand; CUDA, HIP and Metal follow C++, which does the
  same, and a vector lane of a narrow type, which C does not promote, is
  undefined from its own width. x86 traps on that division, scalar or in a lane.
  tolk's integers wrap at their width (`Ops.exec_alu`), as nx states its values:
  the quotient is the least value, the remainder 0, a left shift and an unsigned
  right shift past the width 0, and a signed right shift 0 or -1. A `Truncdiv`
  or `Truncmod` of a signed type whose operands' bounds (`Ops.operand_bounds`)
  reach the least value and -1 divides the magnitudes in the unsigned type of
  its width and takes the quotient's sign from the operands' or the remainder's
  from the dividend's. A `Shl` or `Shr` whose count's bounds, in the count's own
  type (a `Uint32` count of a narrow value included), leave 0 to the width less
  one shifts by the count's low bits and selects 0 past the width, or for a
  signed right shift shifts by the count capped at the width less one. D342's
  shift reads its count so too. An index's division by a positive constant and a
  shift by a constant count keep tinygrad's source. A zero divisor is D355's. A
  shift's count is a number of bits, read at no width. `Ops.operand_bounds`
  keeps a count's own bounds, and `Ops.exec_alu` takes a count of any size: a
  left shift's value is 0 at or past the width of a committed type, a right
  shift the floor at any size. A left shift nothing truncates, by more than
  `Ops.max_exact_shift` bits, has no value, where tinygrad's Python integer
  would exhaust memory. A count at or past the width keeps the `(x & mask) >> k`
  rule, which tinygrad applies, from dropping a mask that clears the sign.
  tinygrad's fold of a constant and a bare count (`uop/symbolic.py`) commits the
  count at the operand's type, which wraps it; `Symbolic`'s fold reads it as it
  is. tinygrad's `uop/weak.py` `commit_srcs_at` commits a weak count at the
  operand's type too: a uint8 shifted left by 256 is itself. tolk's
  `Uop_weak.commit_srcs_at` saturates a literal count at or past the width at
  the width, which every integer type holds, as it does a weak expression count
  always past it, and a weak expression count the operand's type does not hold
  at its own committed type, then converts it to the operand's; one whose known
  bounds no 64-bit integer type holds (`Ops.fits_no_int`), and that is not
  always past the width, is refused, since wrapped it would shift by the wrong
  number. A shift of a weak operand has no such type and commits as any weak
  expression does. A weak count does not widen the type its shift computes at.
  tinygrad's `codegen/decomp/dtype.py` `l2i` emulates a 64-bit shift from the
  count's low word: a count of 2^32 shifts by 0, a count of 64 or more by its
  low 6 bits, and a `Uint32` count fails. tolk's reads both words of the count,
  applies this rule past the width (0, or the sign fill for a signed right
  shift), and takes the low word of a `Uint32` count.
- **Reason:** (e): at the pinned commit, the host's kernel of an int32
  division of the least value by -1 dies with SIGFPE, on the host's vector
  lanes a uint8 shifted right by 8 is itself, where it is 0, a masked
  negative int32 shifted right by 32 is -1, where it is 0, a uint16
  constant 1 shifted left by 65536 folds to 1, where it is 0, a uint8
  shifted left by a weak count of 256 is itself, where it is 0, and on a
  target emulating 64-bit integers a shift by a count of 2^32 is by 0,
  where it is past the width.
- **Pinned by:** the Tolk_engine suite's `divisions and shifts at the
  extremes compute as the interpreter`, on the host (`kernels on the
  host`) and on each GPU (`GPUs ›`, run on fudge's CUDA and NV): every
  integer type's quotient, remainder and shifts of operands drawn at its
  extremes, -1 to 2, counts either side of its width, 32 and 64, and a
  constant count of its width, in scalars and in vectors of 4 lanes,
  against `Ops.exec_alu`; `a narrow
  type shifted by a uint32 count computes as the interpreter`, by constant
  counts of 8 to 65537; `a shift by a literal or weak count of any size
  computes the shift`, compiled and through the interpreter, on the host
  and each GPU, against the shift's meaning written in the test, for
  literal and weak expression counts around the width and at and past
  2^8, 2^16, 2^32 and 2^63; `an emulated 64-bit shift reads the whole of
  its count`, compiled with 64-bit integers emulated, by counts of the
  operand's type, of `Uint32`, weak literals and weak expressions past the
  word and the width; and the Symbolic suite's `a shift of constants folds
  by its count, of any size` and `a masked right shift past the width
  keeps its mask's sign`.

### A division by zero is defined (D355)

- **tinygrad:** `renderer/cstyle.py` (`code_for_op`, which writes `/` and
  `%` whatever the divisor); `uop/symbolic.py` `symbolic_simple` (`x//x -> 1`);
  `codegen/decomp/op.py` (`floordiv_to_idiv`, which corrects a
  negative dividend's quotient whatever the divisor);
  `codegen/decomp/dtype.py` `l2i` (an emulated 64-bit quotient, each bit of
  which a divisor of 0 sets); `helpers.py` (`cdiv` and `floordiv`,
  whose quotient by 0 is 0, which `uop/ops.py` `python_alu` computes);
  `uop/ops.py` (`_min_max`, which bounds a load of a constant table
  by the table's values, whatever its gate).
- **tolk:** `Cstyle.pm_defined_div_shift`, in the extra matchers of
  `Cstyle.clang`, `Cstyle.cuda`, `Cstyle.hip` and `Cstyle.metal`;
  `Shape.symbolic_simple` (`x // x`); `Decomp_op.floordiv_to_idiv`;
  `Decomp_dtype`'s `long_division`; `Ops.vmin` and `Ops.vmax` of a gated
  load of a constant table.
- **Differs:** C leaves an integer division by zero undefined, its
  quotient's and its remainder's; CUDA, HIP and Metal follow C++, which does
  the same. x86 traps on it, scalar or in a lane. tolk's integer quotient by
  0 is 0 and its remainder the dividend (`Ops.exec_alu`), as nx states its
  values and as tinygrad's interpreter computes them. A `Truncdiv` or
  `Truncmod` of an integer type whose divisor's bounds
  (`Ops.operand_bounds`) reach 0 divides by `where (d <> 0) d 1` and
  selects 0, or the dividend, where the divisor is 0; where D346 divides
  magnitudes, it divides by the divisor's magnitude made so. The rule reads
  such a divisor as nonzero, which its bounds cannot show. `x // x` folds
  to 1 only where `x`'s bounds hold no 0; a floored division's correction
  of a negative dividend skips a divisor of 0; an emulated 64-bit quotient
  selects 0 where both words of the divisor are 0. A division whose
  divisor's bounds hold no 0, as an index's by a positive constant, keeps
  tinygrad's source. rune divides as it lowers any divisor
  (`Lower_arith.int_div`). A gated load of a constant table is bounded by
  its alternative too, which it reads where its gate fails: a pad's padding
  reads 0 there, which the table's values alone left out, so a division by
  such a load went unguarded.
- **Reason:** (e): at the pinned commit, the host's kernel of an integer
  division by a loaded 0 dies with SIGFPE, and CUDA's uint8 quotient of 0
  by 0 is 255, where tinygrad's interpreter gives 0.
- **Pinned by:** the Tolk_engine suite's `divisions and shifts at the
  extremes compute as the interpreter`, on the host (`kernels on the host`)
  and on each GPU (`GPUs ›`, run on fudge's CUDA and NV): every integer
  type's truncated and floored quotients and remainders of operands drawn
  at its extremes, -1 to 2 and 0 included as divisors, `a // a`, and a
  quotient and remainder by a constant 0, in scalars and in vectors of 4
  lanes, against `Ops.exec_alu`; the Symbolic suite's `identities › x // x
  is 1 where x's bounds hold no 0`; the Decomp_dtype suite's `emulated
  64-bit integers › i64 quotient by 0, which is 0` and its neighbours, for
  both types and the remainder; rune's `jit › integer arithmetic at its
  type's edges gives eager's values`, its GPU suite's `integers ›` the same
  law, and `Lower_arith › divisions by constants give eager's bits`; the
  Symbolic suite's `bounds › a gated load of a constant table may read its
  alternative`; the Tolk_engine suite's `a pad of integer divisions writes
  what its tensors compute`, `a pad of a quotient by a constant table writes
  what it computes` on the host and on each GPU, and the Codegen suite's
  `padto_idiv_upcast`, whose padding loads the zeros they divide by.

## D348. On the host, a decoded operand is decoded once for every row tile

- **tinygrad:** none: no pass stages an operand on the host, and
  `codegen/opt/heuristic.py` `hand_coded_optimizations` shares a decoded value
  only between the rows a tile upcasts; `codegen/late/linearizer.py` nests a
  kernel's output loops in the order of their axes.
- **tolk:** `Stage.decodes`, `Stage.panel`, `Stage.panel_bytes` and
  `Stage.decoded` (moved from `Heuristic`); `Codegen.full_rewrite_to_sink`,
  which runs it after `Stage.operands`; `Setting.stage`.
- **Differs:** on the host (`target.device = "CPU"`), after the opts, a
  kernel where exactly one operand of its sums of products is decoded
  (D92's `decoded`), whose output loops all have constant bounds and one of at
  least two trips does not read the operand, and whose operand's other
  ranges are its sum's ranges and upcast lanes of constant extents taking at
  most 64 KiB of values, stages the operand's values over those ranges, its
  panel, into a buffer of the register address space: a loop over fresh
  copies of the sum's ranges, the lanes upcast, stores them, and the product
  reads them back after it. The output loops the operand reads take the
  first of the output loops' axes, in order, so that they nest outside the
  others and the panel is decoded once for all those loops' trips. The
  goldens run with `STAGE=0`, as for D262, and do not move.
- **Reason:** (d): D92's tile shares a decode between the rows it upcasts,
  8 on AVX2, so 64 rows decoded each value 8 times, and the decode, about
  2.5 cycles a value on fudge's P-core, was the whole gap to a stored
  decode: kaun's float32 rows by an MXFP4 matrix of 1024 by 1024 took 12.05
  ms against 8.5 ms stored. A packed panel of one matrix is reused across
  the rows of the other: "the micro-panel Br is reused in every iteration of
  Loop 5. As such, it is desirable for Br to remain in the L1 cache while
  Loop 5 is being executed" (Low et al., "Analytical Modeling is Enough for
  High Performance BLIS", ACM TOMS, §4.3.2), and BLIS casts while it packs
  ("casting may be performed as needed when the matrices are packed", Van
  Zee and van de Geijn, TOMS, §5.1.3). The panel stays below an L1 only
  where it fits: on fudge's P-core 64 float32 rows by matrices of 2880, 4096
  and 8192 (panels of 46, 64 and 128 KiB, past the 48 KiB L1) took 98.4,
  189 and 749 ms decoded per tile and 69.3, 145 and 532 ms staged, so the
  bound is the stack the panel lies on: rig's pool workers have 8 MiB, the
  calling thread its own, 512 KiB for a secondary thread on macOS. The
  scheduler stays target-free (D92's ruling): the panel is the host's
  codegen. On fudge, one P-core, interleaved, load 15-17, two undisturbed
  rounds of 41 runs, medians: float32 rows of 10 2.37-2.38 to 1.66-1.67 ms,
  of 64 13.2-13.3 to 9.8-10.4 ms, beside 10.2-10.3 ms for the stored decode;
  kaun's bfloat16 rows (`Quant/host/mxfp4-product-of-1024-{10,64}-bfloat16-tokens`)
  3.01-3.03 to 2.15-2.17 ms and 16.3-16.5 to 10.5-10.6 ms, and in kaun's
  bench itself (`check --quick`, two rounds) 2.90-2.92 to 2.05 ms and 15.11
  to 9.17-9.35 ms, 698 words allocated a call on both; one row (0.66,
  0.82 ms) and rune's `Jit/experts-mxfp4-512x4of32-256` (35-37 ms), whose
  operand reads the rows' blocks, keep their kernels. Files of 32 registers
  (NEON, AVX-512) take 16 rows a tile (D92), so 64 rows take 4 tiles and
  share one panel: unmeasured, fudge has neither.
- **Pinned by:** the Stage suite's `decoded panels`: `on the host a staged
  decoded product computes the bits of the same optimisations unstaged`
  (compiled and run, panels staged, refused at one row tile and past 64
  KiB), `on the host a decoded operand read by several row tiles is decoded
  once` (products of 1 to 128 rows by 32 to 8192 values on files of 16 and
  32 registers, the decodes of each code counted in the linearized
  program, against the rule) and `staging off decodes once a row tile`; on
  fudge, kaun's `Quant/host/mxfp4-product-of-1024-*-bfloat16-tokens`.

## D345. A store through a broadcast is refused

- **tinygrad:** `uop/ops.py` (`UOp.store`, which stores through
  any view); `tensor.py` (`Tensor.assign`, which stores into the
  view it is called on, an expand included); `codegen/opt/postrange.py`
  (`convert_loop_to_global`, which makes the store's rows threads).
- **tolk:** `Prepare.prepare_rangeify` (`refuse_broadcast_store`, among its
  earliest rewrites, after inline calls are inlined); no `fix_repeated_store` in
  `Rangeify`.
- **Differs:** preparing a tensor graph raises `Invalid_argument` where a
  store's destination is an `Op.Expand` that stretches an axis of one
  element to more, or a movement, index, bitcast, `Op.After`, `Op.Detach`
  or `Op.Contiguous_backward` of one, whether the builder wrote it or an
  inline call bound a parameter its body stores into to such an argument.
  Several of the destination's elements are one cell of its storage, so
  the store would write that cell once per element, in no order. A view of
  the broadcast that reaches each cell once, such as one row of it, is
  refused too; an expand that only adds axes of one, or empties one, is no
  broadcast.
  tinygrad stores through it: the kernel of
  `row.reshape(1, 4).expand(4, 4).assign(y)` loops over `y`'s rows into
  `row`. So `fix_repeated_store`, which staged such a store's value where
  it read its destination, has no store left to stage, and goes.
- **Reason:** (e): at the pinned commit that kernel keeps `y`'s last row
  on the host, where its rows run in turn; on a GPU its rows become
  threads, which race for each cell. No raven library stores through a
  broadcast: nx's in-place writes refuse an array that reaches an element
  twice (`Nx_array.set`, `Nx_array.transfer`, the door's
  `Repeated_elements`), and nx and rune write a donated value in place only
  where it is C-contiguous.
- **Pinned by:** the Prepare suite's `prepare_rangeify › broadcast stores`:
  the refusals of the broadcast, its transpose, bitcast, detach and
  gradient marker, a gated store, a broadcast of a state and a state of a
  broadcast, one row of it, an index of its first axis, and an inline call
  that writes a broadcast argument; the stores through an expand that only
  adds axes of one and through a transpose, against the values they write;
  and an expand that empties an axis, not refused.

## D346. Withdrawn

A division that may overflow and a shift past the width were defined; merged
into D342, which holds its rule, its reason and its pins.

## D347. Compiles run on as many domains as the cores the process may occupy

- **tinygrad:** `helpers.py` (`_get_cpu_count`, the CPUs of the
  process's affinity, at most the quota of `/sys/fs/cgroup/cpu.max` rounded
  down) and `PARALLEL`'s default, divided among pytest-xdist's
  workers.
- **tolk:** `Setting.parallel`'s default, `Setting.cpu_count`, and
  `tolk_cores` in `lib/helpers_stubs.c`, which reads `rig_pool_cores` of
  `rig.pool`.
- **Differs:** `PARALLEL` defaults to the count rig's host pool sizes its
  jobs by: the CPUs of the affinity mask, at most the smallest CPU quota of
  the process's cgroup v2 and its ancestors rounded up, then at most the
  domains OCaml's runtime recommends; on macOS rig counts the physical
  cores, where tinygrad counts the logical CPUs. tinygrad reads the quota of
  the mount's root cgroup alone and rounds it down. tolk has no pytest-xdist
  to divide by.
- **Reason:** (c): compiles and rig's host launches share the host's cores,
  so one count, kept in rig's pool, bounds both. A process in a cgroup below
  the root, as each systemd scope or slice is, keeps its quota there, which
  the root's file does not state; a quota of 1.5 CPUs rounded down leaves a
  third of it idle.
- **Pinned by:** the Setting suite's `parallel is between one and the
  domains the runtime recommends`, which bounds the default alone, and rig's
  pool suite, which reads the count from trees of cgroup files (`a cgroup
  without cpu.max` and its neighbours). No suite sets a quota or an affinity
  on its own process, so none pins the default to the count.

## D350. A call compiled on its own writes through storage or is refused

- **tinygrad:** `schedule/indexing.py` (`ALWAYS_CONTIGUOUS`: the
  defines, `AFTER`, `CONST`, `MSELECT`, `MSTACK`, `LOAD` and `CALL`) and
  `realize_custom_kernel_srcs` (which stores whole each other argument of
  a kernel's call after its reshapes, a written one included, and matches
  no lowered precompiled body); `uop/ops.py`
  (`UOp.custom_kernel`, whose results are each argument ordered after the
  call); `uop/ops.py` (`call_with_outputs`, which makes a precompiled
  call's arguments contiguous).
- **tolk:** `Prepare.prepare_rangeify` (`call_storage`, among its earliest
  rewrites); `Indexing.realize_custom_kernel_srcs`, for a precompiled
  call's lowered body too; `Spec.tensor`, which leaves a call's arguments
  to it.
- **Differs:** for a call compiled on its own, a kernel given as code or a
  precompiled call, preparing a tensor graph passes an argument that is a
  run of storage's elements, or of a state of it, from the first, of the
  storage's type, as the storage or the state. Indexing then passes
  storage, a state of it, a call's result, a load, a scalar of a
  precompiled call and a shrink of storage that loops move along, whose
  elements on each trip are a run, as they are, and stores any other
  argument whole. It
  raises `Invalid_argument` where the call writes such an argument
  (`Shape.written_params`, every argument where the body does not show its
  writes), and where it reads a constant or another value that loops move
  along. An argument passed as it is must start where its parameter
  assumes (D54, `Shape.storage_phase`): one that does not is stored whole,
  storage the call only reads outside loops by preparing the graph
  (`copy_off_start`), and it raises, naming both starts, where the call
  writes it, where loops move it, and where the parameter assumes a start
  off the 16 bytes new storage starts on. The tensor graph's specification
  checks no call's arguments against its parameters, so these copies and
  refusals are reached; the kernel graph's still checks them. tinygrad
  stores any such argument whole and the kernel writes the
  copy: at the pinned commit, a kernel copying four floats into `a[2:6]`,
  `a[::2]`, the first four of `a`'s transpose or a broadcast of `a[0:1]`
  leaves `a` and the call's result all zeros. It passes a constant as it
  is, where the kernel expects storage. tinygrad's precompiled functions
  copy every argument first (`call_with_outputs`), so they write a copy;
  tolk's `Ops.call ~precompile:true` takes its arguments as a kernel's
  call does, writing through storage, a run from the first element and a
  loop's row, which rune's loops update in place. Its lowered body matched
  no rule before, and a view at a constant offset was passed as its
  storage: a call reading `a[4:8]` read `a[0:4]`. A precompiled call's
  rows were passed whatever their start, its parameter's assumption
  included.
- **Reason:** (e): the write landed in a copy, or at other elements, with
  no error, and a constant reached a kernel as no storage; with the
  specification checks off, a row off its parameter's boundary reached
  the call, whose vector accesses assume the boundary. A row a loop
  moves along is what rune's loops pass, and its kernels write new
  storage or a state of it, so no raven library makes a refused call.
- **Pinned by:** the Prepare suite's `prepare_rangeify › calls compiled on
  their own` (a run from the first element passed as the storage, a state
  of one as the state, a run at another element as it is); the
  Tolk_engine suite's `calls of views` (reads of a run at an offset, every
  other element and a transpose by a kernel and a precompiled call, and of
  a state of a view; their writes into those, a broadcast, a computed
  value and a constant refused, at either argument; a read of a constant
  refused; writes through a run from the first element, through a second
  kernel's state of one, and through a loop's row on each trip, and a
  loop's column refused; a kernel's read of a value of a loop's row
  refused; a read of a run off 16 bytes and of storage 4 bytes past a
  boundary, which are copied, their writes refused; reads and writes of a
  loop's column and of rows off 16 bytes refused, by both calls, with the
  specification checks on; a view stored whole for a parameter 4 bytes past
  a boundary refused); the Spec suite's `tensor › arguments against their
  parameters`.

## D356. A run's binding of a name holds over the value a call binds it to

- **tinygrad:** `engine/realize.py` (`get_call_kernel_args`, which passes a
  variable its call binds as a constant of its value) and `exec_kernel` (whose
  arguments and launch sizes read a call's bound values over the run's
  `var_vals`); `engine/jit.py` `jit_lower`, where a capture unbinds its input
  variables so that each run binds them.
- **tolk:** `Realize.get_call_kernel_args` and `Realize.call_value`;
  `Tolk_engine.launch`, whose launch sizes and estimates read the variable
  as its arguments do (`Realize.call_values`);
  `Tolk_engine.var_word` and `Tolk_engine.reader`.
- **Differs:** a call that binds a variable a program reads by name, such as
  a custom kernel's (`Shape.custom_kernel`), passes the bound variable, and
  a run reads it, in the program's arguments, its launch sizes and its
  estimates, as the run's binding of its name, or else as its bound value.
  tinygrad passes the bound value, whatever the run binds. A schedule
  that was not captured keeps such a binding, and lists the name among the
  variables a run binds (`Schedule.create_linear_with_vars`): a run of
  `Tolk_engine.run ~vars` that bound `k` to 7 computed the schedule's
  kernels that read `k` with 7, and a custom kernel binding `k` to 3 with 3.
- **Reason:** (c): a linked schedule runs many times
  (`Tolk_engine.link`, `Tolk_engine.run`), each run binding the
  schedule's variables, where tinygrad realizes a schedule once with the
  values it was built with.
- **Pinned by:** the Tolk_engine suite: `link and run › custom kernels with
  variables › a custom kernel reads the run's binding of its variable` and
  `› a custom kernel sized by its variable runs the run's binding`, and the
  same on each GPU; the Realize suite: `get_call_kernel_args › is the
  variable a call binds to each name, in the program's order`, `› is the
  variable itself, at no slot, where the call leaves it free` and `› is the
  call's argument in the slot of a scalar parameter`.

## D355. Withdrawn

A division by zero was defined; merged into D342, which holds its rule, its
reason and its pins.

## D357. A group's threads read consecutive addresses

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  grouping: the first of the first three reduce axes that splits by 16 into
  local threads from the top, `(16, AxisType.LOCAL, True)`);
  `codegen/opt/postrange.py` (`shift_to`, whose split from the top makes the new
  range the axis's outer part); `codegen/gpudims.py` `group_gpudims` (the
  workgroup's thread ranges in the order they were made, the first `lidx0`).
- **tolk:** `Heuristic.group` and `Heuristic.group_threads`.
- **Differs:** the group's 16 threads take consecutive indices of the axis
  they split, its inner end, and of the first three reduce axes, one that a
  load reads at unit stride first. Where no load reads the axis the group
  takes at unit stride but one reads an output axis so, 16 threads first take
  consecutive indices of the first such axis whose split leaves the group
  one, and, made first, are the workgroup's fastest. tinygrad's group takes
  the outer end of the first axis, so that each thread sums a run of it and
  consecutive threads read a run apart: the threads of a sum of rows of 8192
  read 512 floats apart, and the sum of a readout over a batch and a horizon
  (`r_3_256_16_64_64`) read its second operand 1048576 floats apart from one
  `lid.x` to the next. The CUDA C++ Best Practices Guide, §10.2.1 ("Coalesced
  Access to Global Memory"): "Global memory loads and stores by threads of a
  warp are coalesced by the device into as few as possible transactions",
  and, §10.2.1.4, "non-unit-stride global memory accesses should be avoided
  whenever possible". Apple's "Creating threads and threadgroups" (SIMD
  groups): "The threads in a threadgroup are further organized into
  single-instruction, multiple-data (SIMD) groups, also known as warps or
  wavefronts, that execute concurrently." The rule is the renderers' alike,
  as the SIMD group is every GPU target's unit of a load. Kernels that
  stream, take a tensor core or no group keep tinygrad's choices; a sum's
  order changes with the threads' parts.
- **Reason:** (d): rune's compiled sums of few outputs on fudge's RTX 5000
  Ada, rune bench rows `Cuda/sum-rows-float32-2048x8192` (71.7-72.7 us
  against 23.7-23.8 us) and `Cuda/sum-columns-float32-16384x2048` (872-988
  against 297-298 us), and `Cuda/set-row-of-1024x1024-summed` (11.2-14.2
  against 7.8-7.9 us), three interleaved runs each under the timing lock;
  no other `Cuda/` row moved past its range, and the old Lorenz training
  step's loss and gradient at a batch of 1024 kept its 57 ms. Apple timing
  is pending: fudge has no Metal device, and the Metal source is checked
  instead.
- **Pinned by:** the Codegen suite's `consecutive lid.x threads of the
  readout read one address or consecutive ones` (Metal, CUDA and HIP);
  `Tolk.Heuristic`'s `a group's fastest threads read at unit stride where a
  load reads a reduce or output axis so` (every recorded GPU case that
  groups and has a load at unit stride along one of the first three reduce
  axes or an output axis that 16 divides) and its `readout` cases; the goldens generated from tinygrad with
  this rule; the rune bench rows above on fudge, their baselines pending the
  scheduled re-record.

## D403. A power's exponent is a number, read at no width

- **tinygrad:** `uop/symbolic.py` `symbolic_simple` (a casted constant and a
  bare one commit at the promotion), `uop/weak.py` (`commit_srcs_at`),
  `mixin/elementwise.py` `pow` (`_broadcasted`),
  `codegen/decomp/transcendental.py` (`xpow_int`'s `sb.eq(-1)`).
- **tolk:** `Ops.is_number`, which `Shape.fold_const_alu`, the fold of a casted
  constant and a bare one, `Uop_weak.commit_srcs_at` and `Uop_weak`'s cast rule
  read; `Ops.pow`; `Ops.exec_alu`, `Ops.exact_power` and `Shape.simplify_pow`;
  `Transcendental.xpow_int`.
- **Differs:** a weak integer exponent of an integer base keeps its value: it
  is not committed at or read through its base's type, and `Ops.pow` does not
  promote it; a float base reads its exponent in the float as tinygrad does. tinygrad
  commits it there, so `0 ** 65536` at uint16 is 1, where it is 0: the exponent
  wraps to 0. With the exponent whole, a negative one on an unsigned base is
  reachable, and `xpow_int` gives its greatest value no special meaning of -1:
  an unsigned base has none. A power at no width whose exact value would not
  fit in memory has no value: `Ops.exec_alu` refuses it and the fold leaves it,
  where tinygrad's Python integer would exhaust memory; a base of 0, 1 or -1
  folds at any exponent. A weak exponent expression of a committed integer
  base, whose known bounds no 64-bit integer type holds (`Ops.fits_no_int`), is
  refused at lowering
  (`Transcendental.xpow_int`): its
  value cannot reach the kernel, where tinygrad computes it wrapped and takes a
  power by the wrong number.
- **Reason:** (e): at the pinned commit, `Cast(uint16, 0) ** 65536` folds to 1,
  where it is 0, and a uint8 to a weak exponent of 256 is 1.
- **Pinned by:** the Tolk_engine suite's `a power by an exponent of any size
  computes the power`, compiled and through the interpreter, for each integer
  type, literal and weak exponents around the width, -1, 2^8, 2^16 and 2^32 and
  exponents of the base's type, against the power written in the test; the
  Symbolic suite's `a power of constants folds by its exponent, of any size`,
  `a float power of a constant reads its exponent in the float`, `a power no
  width truncates stays a power past what it can hold` and `a power no width
  truncates folds for a base of 0, 1 or -1 at any size`; and the Uop_weak
  suite's `a weak exponent keeps its value past its base's type` and `a cast of
  a weak power computes it and leaves its exponent weak`; the Transcendental
  suite's `xpow_int › an exponent's bounds`.

## D413. An integer literal holds its operation's value from construction

- **tinygrad:** `mixin/elementwise.py` (`_broadcasted`, which keeps a weak
  constant weak under a committed operation), `uop/weak.py` (`commit_srcs_at`,
  which commits it late), and so every rule between the two reads the literal as
  written: `uop/symbolic.py` `symbolic_simple` (two stage folding),
  `codegen/decomp/op.py` (`floordiv_to_idiv`, `floormod_to_mod`,
  `powers_of_two`, the `fast_idiv` rules).
- **tolk:** `Ops.v`, which every constructor and `Ops.replace` goes through; its
  readers were deleted (`Shape.fold_const_alu`'s read of a weak operand,
  `Shape.wrap_folded`, the two stage maximum's reading, `Decomp_op.at_width`,
  `Decomp_op.divisor`, and the negation rule's reading of its constant).
- **Differs:** a bare integer literal under an operation on a committed integer,
  or a cast to one, holds the value that type gives it, wrapped to its width,
  and stays a bare weak literal, so patterns still match it. A comparison reads
  it at the operands' meet, another binary or ternary operation at its own type,
  a cast at its target. A shift's count and a power's exponent are numbers
  (D346, D403), and floats are untouched (D128). So `-1` under a `uint32`
  operation is `4294967295` in the graph, `2^64` under a `uint64` division is
  `0`, and a golden whose literal was written past its width differs from
  tinygrad's in that spelling.
- **Reason:** (e): each rule that read a literal as written computed a wrong
  value or crashed: on int8, `x // 200` gave 0 where -1 is due; on uint8,
  `max(max(w, 1), -3)` became `max(w, 1)` where the machine computes
  `max(w, 253)`; on int32, `(x * 65536) * 65536` made a shift by 32, which C
  leaves undefined; `x:i32 * 2^32` failed to compile. Six readers repaired
  these one rule at a time, and a new rule had to know to read so; the literal
  now has the value before any rule sees it.
- **Pinned by:** the Ops suite's `integer literals`, at every integer type, at
  -1, at the bounds, past the width and at the width's power; the Codegen
  suite's `constants past the width` (`a literal computes as its value at the
  operation's type`, every operation and type, `a product or conjunction by a
  literal that is 0 at its type is 0`, `a square of a product whose constant
  squares to 0 is 0`, `a weak divisor past the width divides as its value
  there`, `a shift by a weak expression count keeps the count whole`); the
  Symbolic suite's `committed constants hold their type's value` and `integers
  wrap › a square of a product keeps its value, its constants folded at an
  unsigned width`; the Decomp_op suite's `late_patterns › a product by a power of
  two shifts by less than its width`, `a quotient by a power of two reads it in
  its operand's type` and `a quotient by a weak divisor past its width reads it
  in its type`.

## D358. An object defines the half conversions Clang calls

- **tinygrad:** `runtime/support/compiler_cpu.py`
  (`ClangCompiler.compile`, which compiles the source alone).
- **tolk:** `Compiler_cpu.clang` (`half_conversions`, `converts_halves`).
- **Differs:** for an x86_64 processor without F16C, each object defines
  `__extendhfsf2` and `__truncsfhf2`, the conversions between halves and
  floats Clang calls there, rounding to nearest with ties to even. The
  rendered source is unchanged; the conversions follow it at compile, and a
  source's length class counts without them.
- **Reason:** (c). rig takes a symbol the object does not define from the
  process (`Rig.section-host_symbols`). The C math library's functions Clang
  calls on such a processor, such as `truncf` without SSE4.1 and `fmaf`
  without FMA (D178), are there, but compiler-rt's conversions are in no
  library a raven process loads, so every program with a half cast failed to
  link on such a processor: on another machine's x86_64 host, whose baseline
  `x86-64` has no F16C (D95). A widened NaN is made quiet, as F16C's and
  arm64's conversions make it.
- **Pinned by:** `Rune.jit on another machine's host › half casts give
  eager's bits` (every half widened to float32 and float64, float32 and
  float64 edges and ties narrowed, and any float32 narrowed).

## D359. comgr compiles with LLVM's IR verifier on

- **tinygrad:** `runtime/support/compiler_amd.py` (`compile_hip`, the
  options of the compile and of code generation, which a release comgr's
  Clang driver turns into compiler commands with `-disable-llvm-verifier`).
- **tolk:** `Compiler_amd.verified`, given from comgr 2.8
  (`Compiler_amd.verified_from`), and `Compiler_amd.hip`'s `table`, which
  digests both.
- **Differs:** from comgr 2.8 (ROCm 6.2, LLVM 18), both steps run with
  `-fverify-intermediate-code`: LLVM's verifier checks the IR Clang makes of
  the source, and again before instruction selection, and invalid IR ends
  the compile's process with the verifier's finding, which
  `Renderer.Compiler.Compile_error` carries. comgr 2.7 and earlier, whose
  driver rejects the option, compile without it. comgr 3.0 (ROCm 7.0.0) on
  nonnormal's i9-9900K, 192 kernels of gpt-oss-20b's first call and of a
  beam search over a float32 product, each compiled 3 times both ways in
  one process: the verifier costs 1.9% and 2.5% of the compile time in two
  runs (0.1-4.7% per kernel), and every code object is unchanged.
- **Reason:** (e), as D275: an optimization that writes invalid IR goes
  unseen by a release comgr, and compiles into wrong values or crashes
  instruction selection.
- **Pinned by:** `Tolk.Compiler_amd › LLVM's verifier › invalid IR raises
  Compile_error with the verifier's finding`, a call of an intrinsic with an
  argument it does not take, which comgr 3.0 compiles without the
  verifier.

## D416. The constant folder divides floats

- **tinygrad:** `uop/ops.py` (`exec_alu`, `python_alu`): no rule for `FDIV`, so
  a folded float division raises `KeyError`.
- **tolk:** `Ops.exec_alu`.
- **Differs:** `exec_alu` folds an `FDIV` as IEEE does, then rounds to the
  operation's type, as it multiplies: a zero divisor gives an infinity of the
  quotient's sign and `0. / 0.` is NaN.
- **Reason:** (e). A quotient the folder could not compute would differ from
  the one compiled code computes for the same constants, or stop the compile
  with a host-language exception.
- **Pinned by:** `Tolk.Ops › exec_alu › exec_alu_values.golden`, whose `.tolk`
  holds each `op=Ops.FDIV` row's quotient.

## D417. A round's stop compares two kernels timed again in turn

- **tinygrad:** `codegen/opt/search.py` (`beam_search`: the beam
  starts as `[(s, inf)]`, and a round ends when
  `beam[0][1]-opts[0][1] < min_progress`, the fastest's time of this round
  against the beam's first's from the round that kept it).
- **tolk:** `Search.beam_search` (`time_pair`).
- **Differs:** from the second round, once a round's candidates are timed,
  the beam's first kernel and the round's fastest are timed again in turn,
  three times each, and the stop and whether the fastest is kept alone
  compare their least times. The early stop (three times the beam's first's
  time), the ranking of the round and the next beam are tinygrad's. A time
  taken again that raises `Failure` is infinite, and two infinite times end
  the search, where tinygrad's `inf - inf < min_progress` goes on. The
  round's fastest keeps its linked program through the round, and the beam's
  first through the next. Each round from the second times six more runs.
  `a search chooses what tinygrad's chooses` counts the first timings against
  tinygrad's measurements.
- **Reason:** (d), to re-measure on fudge. A device's clock follows its
  load, and a GPU idles while a round compiles, so times taken rounds apart
  ran at different clocks. On nonnormal's R9700 at the firmware's clocks,
  searches of a 1024 by 1024 float32 product (`Rune.jit' ~beam:2
  ~parallel:8`, 8 interleaved processes) stopped at round 5 in 4 of 4 under
  tinygrad's rule, and reached round 7 in 4 of 4 under this one. The kernel
  chosen took 156-172 us of GPU time per call against 136.8-137.7 us (medians
  of 9 profiles of 200 calls), and the first call 0.97-1.06 s against
  1.41-2.06 s, for the two more rounds. A row sum of 8192 by 2048 and a
  softmax of 4096 by 1024 chose kernels of equal time under both rules
  (37.7-40.5 and 68.1-74.1 us).
  On fudge's RTX 5000 Ada (CUDA, and NV through the nvidia driver), the same
  product's beam's first timed again agreed with its earlier samples within
  3.5% from the third round, and 6.6% below them in the second, in 12
  searches, which chose as tinygrad's rule did. Timing both in turn keeps a
  near tie going on about half the time, as tinygrad's rule does between two
  minima of a round, where timing the beam's first alone sets one time
  against the least of a round's: on fudge's host, a first call of
  `Rune.jit ~beam:2` of a 64 by 64 float32 product and its tanh took
  1.36 s (0.57-2.29), against 1.25 s (1.03-1.99) with tinygrad's rule and
  3.28 s (2.21-3.35) timing the beam's first alone, over 3, 3 and 4 rounds:
  medians of 5 interleaved cold processes at a load of 7 to 9, so the 9% is
  indicative. No row on fudge holds the choice: a near tie at round 4
  decides between kernels of 161 and 122 us per call under either rule.
- **Pinned by:** `Tolk.Search › a search goes on past a round whose clock is
  half the round before's` and `› a search chooses as at a steady clock when
  its device's clock changes between rounds`, which fail under tinygrad's
  rule; `› after a round, the beam's first and its fastest are timed again in
  turn; …`; `› a time taken again that raises ›` (`Failure` counts as
  infinite, for one or both, and `Exit` is raised).

## D404. Calls that interfere with no other run as one launch of bodies

- **tinygrad:** `schedule/rangeify.py` (`split_store`, one kernel per store or
  end), `schedule/__init__.py` (`create_schedule`, one call per kernel of the
  kernel graph); `sched_batches` (`runtime/support/hcq2.py` `_build_queues`,
  `_finalize_batch`) only batches consecutive calls of a device into a queue
  program, D146's form. Kernels of several outputs were deleted upstream in
  `1fce864a6`.
- **tolk:** `Schedule.launches` (`launch_linear`), after the schedule cache and
  `pm_copy_from_store`, over the finished schedule; `Shape.launch`,
  `Shape.launch_members` and `Shape.is_launch`; `Codegen.lower_member`,
  `Codegen.member_args`, `Codegen.launcher`; `Cstyle.outlined`,
  `Cstyle.render`'s qualifiers and the CUDA renderer's choices read from a
  program's bodies; `Shape.call_accesses` in `Shape.program_info_of_sink`;
  `Realize.split_launch` and `Renderer.t.launch_members`,
  `Renderer.t.launch_args` and `Renderer.t.launch_locals`. The goldens record no
  program whose schedule holds a launch.
- **Differs:** calls of one device that run in either order run as one launch
  whose members are their kernels, each lowered and compiled as a kernel of its
  own and laid out with its workgroups end to end, gated by the launch's
  index (D305). Two calls interfere when one writes a byte the other reads or
  writes: a view's bytes are those of its run of storage, any other view its
  whole storage, a loop or a copy writes its storage whole, and on the state of
  one write marked disjoint (D405) two of its kernels do not interfere. A call
  joins the earliest launch of its device that starts after the last position
  it interferes with. It moves up only past calls it does not interfere with,
  and only where no storage's life grows: past the first use of each storage
  it takes, unless the storage is given to the schedule, holds one element, or
  the call is in a loop's body, whose storage lives for the whole loop. At the
  top level a call of many elements joins only a launch writing storage it
  writes: the top level's order is chosen for what storage calls make, and the
  stores of one state are what a launch of many elements is for. A member is a
  call of a kernel whose ranges have constant sizes and whose storage lies on
  one device that is no disk. The arguments of a launch are its members', each
  once, storage first; a variable a member's call binds in its slot is read
  from that slot, as a launch binds each name once, so members may bind one
  name to several values.

  What a target can run is the target's fact: `Realize.split_launch` splits a
  launch before it is compiled, into launches a target can run. A member that
  fixes its threads or reduces, which may share values between a workgroup's
  threads through workgroup memory and barriers, joins only members of its
  kernel, so that every thread reaches each barrier of the member it runs
  (D285); Metal declares workgroup memory only in a kernel function, so there
  such a member runs alone (`Renderer.t.launch_locals`). A launch holds at
  most `Renderer.t.launch_members` members (CUDA's 64, NVRTC's time a member
  rising past it) and `Renderer.t.launch_args` arguments (CUDA's 512, the 4 KiB
  of parameters a kernel takes), and with `MAX_KERNEL_BUFFERS` at `n` above 0
  at most `n` buffers. Every subset of a launch is a launch, so splitting one
  is always legal.

  A launch's members have their own kernels, heuristics and gates: a member
  reading shared storage reads it as a parameter passed once; no arithmetic is
  shared across members, as fused kernels did, since `Schedule.schedule` keeps
  each kernel's own compute shared by D137 before kernels exist.

  A schedule ordered by its storage (`by_storage`, a program over several
  devices) and a program made per run are launches of what they make: storage
  is live only from a member's first use, so a piece's store waits for the
  value it reads and no launch of them holds every piece's value at once.

  Each member is its own body; alike members are one body, lowered once, the
  dispatch lowers about 27 instructions a member, and Clang compiles 256 alike
  members in 0.27 s; NVRTC's time a member falls to its least at 64 members and
  grows past them, and past its member limit a CUDA launch runs as several,
  alike chunks one program: 256 alike pieces compile cold in 0.11 s on CUDA and
  run as 4 launches in 18.6-18.9 µs against 347-352 apart. On the host a launch
  compiles cold in 0.46 s against 0.09 s apart and costs nothing at run time.
- **Reason:** (d). On CUDA a launch costs about 1.3-2 µs, so a kernel's calls
  are its cost. fudge's RTX 5000 Ada, against the prior two passes (`contract`
  and fused `merge_linear`), four interleaved runs each, GPU shared in some:
  NUTS on eight schools, 4 chains, 0.95-1.06 ms a step against 1.20-2.64;
  `launch/cuda` rows and rune's `Cuda/stack-*` and `Cuda/cat-*` equal within
  their spread (minima within 1-5%). Launches against calls apart: 16 pieces
  concatenated along axis 1 under `Rune.jit` run as one launch, a median 7.2
  µs a call over 38 interleaved runs, as the two prior passes (a fifth of the
  runs of either take 8-10), against 29-33 with rune's mark removed (8 runs);
  16 products of `[64; 1024]`
  8.6-8.8 µs against 36.6, 64 pieces of 64 elements 9.6 against 96.6, all with
  the same bits. Pinned by the rows `launch/host/*`, `launch/cuda/*`,
  `launch/lower-256`, rune's `Cuda/cat-16-columns-of-exp-64x64`, and norn's
  `nuts/schools/4` and
  `nuts/schools-density-cuda/4`.
- **Pinned by:** `Tolk.Launch_law` (`test/schedule/schedule`,
  `test_launch_law.ml`): `launches › a schedule's launches write what its graph
  states, in either order of their members` and `› a write marked disjoint runs
  its columns as one launch`; the `Tolk.Launch` suite
  (`test/schedule/schedule/test_launch.ml`): `launches › on the host, …` and
  `› on CUDA, the stores of one state run as one launch and write what their
  kernels apart write`, `cases › … alike members compile one body`, `› on the
  host, a launch's bodies are out of line`, `› on CUDA, a launch's bodies are
  out of line`, `› on CUDA, sixteen products on tensor cores run as one launch`,
  `› on the host, a launch's blocks split its members`, `› on CUDA, members of
  fewer threads run gated`, `› on CUDA, a state past a launch's members runs as
  several`, `› a launch never makes storage live longer` and `› a launch is
  split by what its target can run`; the `Tolk.Schedule` suite's
  `create_linear_with_vars › launches` (independent calls in a loop and of
  one element at the top level run as one, calls of many elements at the top
  level run apart, a reader of another's output and an overwriter of what
  another reads run apart, disjoint runs of one buffer run as one launch, a
  reduction joins, two devices run apart, a call moves up only past calls it
  does not conflict with, the launches' linear work, the lifetime of storage
  of many elements, the laws that outputs run as one write what their tensors
  compute and that calls of a loop run as one, reading and overwriting one
  another's slots and the trip, write what their tensors compute);
  `Tolk_engine › a scan's merged calls carry in place as its calls apart do`
  and the custom-kernel cases of two values of one name, on each GPU present;
  `Jit › variables › merged custom kernels of one value read each run's` and
  `› of two values keep each its own`; rune's `Jit › fully sharded › *`.

## D405. A write marked disjoint is exempt on its state

- **tinygrad:** none. `schedule/__init__.py` makes no launches.
- **tolk:** `Ops.after_disjoint`, `Ops.is_disjoint`, `Schedule.tagged`,
  `Ops.call_info`'s `write`, read by `Schedule.launches`; rune's
  `Compile.assembled`.
- **Differs:** a maker that has proved the stores of one write write disjoint
  elements states it with `Ops.after_disjoint`; `Schedule` numbers each such
  write and tags each of its kernels with the number and the position of the
  state among its arguments (`call_info.write`, outside the program's and the
  call's key). In `Schedule.launches` two kernels of one such write do not
  interfere on that state. An unmarked `After` gets the byte relation (D404),
  so tolk's own API cannot make a race: overlapping stores of one unmarked
  write land in an unspecified order, and two stores of one state that may
  overlap are never given a launch on the mark's say-so; stores that break a
  mark's promise give undefined values, as `Ops.after_disjoint` states. The
  mark keeps through `Rangeify.bufferize_to_store`, every other rewrite
  keeps it as a node's argument, and the disk cache's codec holds both the
  mark and the tag.
- **Reason:** (d) of D404 and rune's own proof. A concatenation along a later
  axis writes columns, strided regions which no byte range separates; a
  variable start is unreadable too, and a stepped assembly's stores overlap as
  ranges while its regions never meet. rune's `states` (`meet`, step-aware)
  already proves what tolk would have to re-derive per kernel. Without the
  exemption the row of concatenating 16 pieces along axis 1 costs 29-33 µs on
  CUDA against a median 7.2, and NUTS 337 launches a step against 272 (programs 149
  against 132). A box test of store destinations in the tensor graph, the
  other way to keep them, is correct on 21,000 programs and three times the
  lines, with fewer regions it can read; the program shape where its maker
  already says so needs no such test.
- **Pinned by:** rune's `Jit › assemblies give eager's bits`'s test of the launch
  count (`a concatenation along a later axis runs as one launch`, the mark kept
  through rangeify and the disk cache), `Tolk.Launch_law › a write marked
  disjoint runs its columns as one launch`, `› a schedule read back keeps a
  marked write's launch`, `› marked writes of two calls run in turn` and `› a
  marked write of no stores marks nothing`, and the `create_linear_with_vars ›
  writes › stores crossed into two buffers write both` pin of D330.

## D418. A slot given to storage is taken

- **tinygrad:** `uop/ops.py` `UOp.new_buffer`, `UOp.alloc` and
  `UOp.placeholder`: a given number leaves `UOp.unique_num` where it was.
- **tolk:** `Ops.new_buffer`, `Shape.alloc` and `Shape.placeholder` take a given
  slot (`Ops.take_unique`).
- **Differs:** a number given to one of them is never returned by a later
  `Ops.unique_num`.
- **Reason:** (e). Storage made later without a slot could draw the given number
  and, at the same size and type, be the same node: two storages as one, so a
  write to one reaches the other.
- **Pinned by:** `Tolk.Ops › storage › a buffer given a slot is no buffer made
  without one`, `› an alloc given a slot is no alloc made without one` and `› a
  placeholder given a slot is no placeholder made without one`.

## D419. Metal hides a bfloat16 product's operands from its compiler

- **tinygrad:** `renderer/cstyle.py` `MetalRenderer.render_kernel`: each
  product's matrices are of its own input type, `simdgroup_bfloat8x8` for a
  bfloat16 product, in a helper the compiler may inline.
- **tolk:** `Cstyle.metal_wmma`.
- **Differs:** a product of bfloat16 matrices into float32 converts each
  lane's elements of `a` and `b` to float and multiplies
  `simdgroup_float8x8` matrices. bfloat16 has float32's exponents, so the
  product of two bfloat16 values is exact in float32 while it stays in
  float32's normal range, and the sums are float32's, as the bfloat16
  product's were: while every product stays in that range, the result is
  the same. A product into bfloat16 keeps its bfloat16
  matrices, since widened its sums would round where float32's do, not
  where its own do; its helper is `__attribute__((noinline))` instead.
- **Reason:** (b): Apple7's compiler service (an M1 Max) crashes, with
  `XPC_ERROR_CONNECTION_INTERRUPTED`, building a kernel whose bfloat16
  product has an operand it knows, such as literal zeros, into float32 or
  into bfloat16, and Metal makes no pipeline. rune's attention derivative
  on Metal (`Lower_attention.core_dq` and `core_dkdv`) gives its products
  zeros where a tile holds fewer rows or keys than its steps, or where a
  score program does not read the score. The cost: the float32 product
  gives up the bfloat16 matrix type, and the bfloat16 product a call per
  product. Neither's speed is measured, since the Mac does not time.
- **Pinned by:** `Tolk.Cstyle › tensor cores on Metal › a bfloat16 product
  with an operand of literal zeros holds the product`, into each type, which
  fails to build its pipeline on the M1 Max without it;
  `metal_tc_bf16_f32_8_8_8.tolk` and `metal_tc_bf16_bf16_8_8_8.tolk`.
