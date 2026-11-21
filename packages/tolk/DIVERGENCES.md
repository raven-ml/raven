# Divergences from tinygrad

tolk follows the tinygrad commit `TINYGRAD` in `test/gen/generate.py` pins. This
file lists every place where it does something else. An entry is admitted
for one of five reasons only:

- **(a) an OCaml constraint:** acyclic modules, static types, the GC, domains;
- **(b) a named consumer:** a raven call site that fails without it;
- **(c) the executor's contract:** rig's devices, submissions and programs,
  which `tolk.engine` drives with the compiler's output;
- **(d) measured speed:** of a named raven workload, pinned by a bench row on
  fudge whose baseline would catch the regression;
- **(e) correctness:** a divergence that keeps correctness where tinygrad
  computes a wrong value or does not terminate, pinned by a test, stays
  without a consumer (`docs/dev/design.md`: correctness is never traded).

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

An entry states each part as briefly as it can be stated exactly. Differs
states the rule with every exception, condition and scope limit, and what
tinygrad does instead. Reason gives the letter and a sentence or two that
admit it: the constraint or contract, the consumer, the bench row and the one
number that justifies it, or the failing case. A withdrawn entry says in a
line or two what holds its rule, or that tolk follows tinygrad again.
Measurements beyond that number, run histories and quotes from references
live in `_plans/` and in commit messages.

## D1. Withdrawn

A device's timeline was its signal word alone, whose values a batch's host
program took as parameters. rig numbers each device's work and orders it by the
memory it uses, and tolk encodes no queue (D146).

## D2. Withdrawn

Compiling a pattern to closures (`Ops.Upat.compile`,
`Ops.Pattern_matcher.rule_ctx`) without the source generation and
`UPAT_COMPILE` of `uop/upat.py` `upat_compile` and `uop/ops.py`
`upat_deferred_compile` is scope (README Exclusions). A pattern that
`upat_compile` refuses, a repeat over a pattern with several namings or too
many alternatives, has the namings `uop/ops.py` `UPat.match` gives, which
`Upat.match_` promises for every pattern. A repeat that names nodes matches no
sources as `UPat.match` does, its names unbound, where `upat_compile` requires
a source (`uop/upat.py` `_get_clause`).

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk:** `Device` (the compiler half of `device.py`);
  `Tolk_engine.target` and `Tolk_engine.link`; `Ops.param_arg` (the type and
  the function) and `Ops.new_buffer`.
- **Differs:** tolk holds the compiler half: `Compiler` and the renderer and
  compiler selection of `Compiled`, returning data. The lazy `Buffer` and
  running a schedule are `tolk.engine`'s, which has no registry of devices by
  name: its caller maps each name to a device. The IR half of each `ops_*.py`
  (queues, `pm_encode`, program data), allocators, programs, drivers and
  profile events are rig's (D146). A node of the compiler holds no runtime
  state: `ParamArg` has no `buffer`, a `BUFFER` is named by its slot for the
  engine to bind, and `UOp.buffer`, `realized`, `is_realized`, `_buffer_view`,
  `_base_buffer_is_realized`, `from_buffer` and `_frompy` (`uop/ops.py`) are
  the engine's. Bound storage is its own operation, `BUFFER` or `ALLOC`, so
  the checks of `ParamArg.buffer` in `uop/spec.py` `spec_tensor` and
  `spec_kernel_graph` hold by construction.
- **Reason:** (c).
- **Pinned by:** for `Ops`: `Tolk.Ops › storage › new_buffer takes the
  next slot without one` and `Tolk.Ops › arguments › reprs.golden`; for the
  engine, the Engine suite (`test/engine/tolk_engine`): `refusals › link
  refuses a device the map does not hold` and `refusals › run refuses a
  parameter it binds no buffer`.

## D4. Import cycles are broken

- **tinygrad:** the imports inside functions that Python uses to hide a cycle:
  `uop/ops.py` calls `symbolic`, `render`, `spec`, `upat`, `schedule.prepare`,
  `mixin.rand` and `renderer`; `uop/spec.py` against `device`;
  `renderer/__init__.py` imports `device.Compiler`, while `device.Compiled`
  holds renderers; `codegen/opt/postrange.py` `apply_opts` against `search`
  and `heuristic`, with `search` importing `engine.realize`;
  `schedule/__init__.py` against `engine.realize`, and `tensor.py` against
  `engine.jit` and `engine.realize`; `renderer/cstyle.py` imports the
  compilers and `ops_metal`.
- **tolk:** for `uop/ops.py`, `Shape.simplify_rules` and the line after
  them, which sets them; `Ops.Make_elementwise`, `Ops.repr`,
  `Ops.bufferize_opts`; `Render.render`, `Render.srender`;
  `Renderer.Compiler`; `Prepare.contiguous_view`;
  `Schedule.pm_flatten_linear`; `Codegen.apply_opts`;
  `Realize.lower_and_compile`; and `Compiler_metal`.
- **Differs:**
  - the `UOp` methods that call a later module are functions of that module:
    `contiguous_view` of `Prepare`, `to_elf` of `Device`, `render` and
    `srender` of `Render`;
  - `engine/realize.py`'s `pm_flatten_linear`, which `schedule/__init__.py`
    imports, is `Schedule.pm_flatten_linear`, since `Realize` follows
    `Schedule`;
  - `simplify` is `Shape`'s, with the functions that call it (reshaping,
    `resolve`, shapes) and the rules it rewrites with, which read shapes:
    `symbolic.py`'s `symbolic_simple`, `commutative` and `symbolic`,
    `divandmod.py`'s `div_and_mod_symbolic`, `movement.py`'s `mop_cleanup` and
    `weak.py`'s `pm_uncast_const`. `simplify` precedes the rules and reads them
    through one reference that the line after them sets, so a program that links
    `Shape` has them. `Symbolic` keeps `sym` and the rules on conditions, which
    call `simplify`. The other `UOp` methods that read shapes, of storage,
    variables, calls and programs, are `Shape`'s too;
  - `render.py`'s `pretty_print` is `Ops.pp`, since arguments print the nodes
    they hold (`argstr`) and the two recurse;
  - the kept methods of `mixin/*.py` are functions of `Ops`, or of `Shape`
    where they read shapes, since `ops.py` calls them and a module cannot
    call a later one; the elementwise ones, which patterns share with
    nodes, are one functor applied to both;
  - the small types that `ops` and `spec` name from later files
    (`Estimates`, `BufferizeOpts`), and `device.py`'s `is_disk_device`, are
    defined in the earliest module that needs them; `Opt`, whose file
    depends on nothing, stays in its own module;
  - `spec.py`'s imports of `codegen.opt`, `schedule.rangeify` and
    `renderer` serve only `pyrender_globals`, which is not ported (see the
    README), so they need no break;
  - `device.py`'s `Compiler` and `CompileError` are `Renderer.Compiler`,
    since a renderer holds its compiler and `Device` follows `Renderer`;
  - `apply_opts` takes the optimiser as an argument. `Search` follows
    `Postrange` and takes how to link and time a program as its `link` and
    `time` arguments, where `beam_search` takes `rawbufs` and `var_vals`: its
    caller makes the buffers `args_from_ast` describes (`Tolk_engine.slots`),
    since `tolk` cannot name rig's buffers. `get_kernel_actions` returns each
    candidate with its action, without `include_0` and its positions in the
    action table. `Codegen.full_rewrite_to_sink` and `Codegen.to_program` take
    the beam search as their `beam` argument, a function of the width the
    kernel asks for, and raise when a kernel asks for one and none is given;
  - the engine has one order (schedule, realize, jit), and each late binding
    is a function argument, never a global reference:
    `Realize.compile_linear` takes the target of each device from its caller;
  - the compiler modules precede `Cstyle`; `ops_metal.py`'s `MetalCompiler`,
    which `cstyle.py` imports inside `MetalRenderer`, is `Compiler_metal`;
    the rest of `ops_metal.py` is rig's (D146).
- **Reason:** (a). Each layer's review checks that its breaks are the
  smallest possible.
- **Pinned by:** for `Codegen`: `Tolk.Codegen › beam search › a kernel that
  asks for a beam of width w is optimised by beam w` and `› raises
  Invalid_argument when a kernel asks for a beam and none is given`; for
  `Ops`: `Tolk.Ops › resolve › simplify leaves a constant, and a sink of
  constants and stacks of constants, alone`, and the `Tolk.Ops › resolve`
  tests that simplify with `Symbolic`'s rules; `Tolk.Ops › printing ›
  pretty.golden`; `Tolk.Ops › elementwise patterns › the pattern operators
  are the named pattern operations`; for `Renderer`: `Tolk.Renderer ›
  Compiler › compile raises what the toolchain rejects`, its
  `Compile_error`; for `Prepare`: `Tolk.Prepare › contiguous_view › a
  reshape keeps the order`; for `Schedule`: `Tolk.Schedule ›
  pm_flatten_linear › Linears nested in Linears are inlined in order`; for
  the engine's order: `Tolk.Realize › lower_and_compile with a beam search ›
  raises Invalid_argument for a kernel that asks for one when none is
  given`, and `Jit JITBEAM › JITBEAM unset › a kernel is searched with
  BEAM's width`, through `Realize.compile_linear`; for `Compiler_metal`:
  `Tolk.Compiler_metal › MTLCompiler › compiles a kernel to a Metal
  library`.

## D5. Compilation workers are domains

- **tinygrad:** `engine/worker.py` (`multiprocessing` spawn workers);
  `helpers.py` (`Context` and `ContextVar`, one value per process);
  `codegen/__init__.py` (`to_program_context`, the settings a worker
  process is started with, and `to_program_cache`, which the parent fills);
  `schedule/indexing.py` `apply_movement_op` (`functools.cache`).
- **tolk:** `Worker.spawned` and `Worker.map`; `Setting.t` and
  `Setting.context`; `Codegen.to_program_cache` (`to_program`'s cache);
  `Indexing.memo` and `Prepare.pm_mops_in`, and the memo
  `Codegen.full_rewrite_to_sink` makes for its devectorizing rewrite;
  `Compiler_worker.call`, and the C files
  `lib/runtime/support/compiler_worker.h`, `compiler_amd_worker.c`,
  `compiler_cuda_worker.c` and `compiler_metal_worker.c`.
- **Differs:**
  - compilation runs on domains, which `Worker.map` spawns for the call and
    joins before it returns, where tinygrad keeps a pool. The process
    machinery goes with the pool: the hidden `__main__`, the copied
    environment, the recycling of a worker after 16 tasks,
    `terminate_worker_pool`. The domains block every signal but a fault's,
    where a tinygrad worker ignores SIGINT, which a domain cannot, since a
    handler is the process's. The workers' context (`ALLOW_DEVICE_USAGE`,
    `VIZ`, `TRACK_MATCH_STATS`) goes: the compiler opens no device and has
    neither of the other settings;
  - a setting holds one value per domain, where tinygrad holds one per
    process: a domain starts with the values of the domain that spawns it,
    and a `context` override is seen by its own domain only. A spawned
    domain starts with its spawner's settings, so `to_program_context` goes
    too;
  - `to_program` keeps its programs, and `lower_sink_to_linear` its
    schedules, in a table every domain reads and fills under a lock
    (`Helpers.Memo`), and a domain that asks for one another is making waits
    for it, so each program is compiled once, as tinygrad's parent compiles
    each key once, and each body is scheduled once;
  - the matchers a tinygrad worker keeps per process (`codegen/decomp/op.py`
    `get_simplifying_rewrite_patterns`, `get_late_rewrite_patterns` and
    `codegen/decomp/transcendental.py` `get_transcendental_patterns`,
    `functools.cache`) are made once and shared by every domain
    (`Decomp_op.simplifying_patterns`, `Decomp_op.late_patterns`,
    `Transcendental.patterns`), and a rule compiles on the domain that tries
    it first, or on each that tries it at once;
  - the results of `apply_movement_op`, which tinygrad keeps in each worker
    for its 16 tasks and in the parent, which schedules and renders a lone
    kernel, for the process, are kept for one kernel's devectorizing rewrite,
    in a memo its context holds; scheduling and the other callers compute
    each movement again;
  - MTLCompiler, comgr and NVRTC compile in processes of their own, spawned
    by the compiling domains, where a tinygrad worker process loads the
    library once and compiles in it. tinygrad compiles a lone kernel in the
    parent; tolk sends it to a process too, at the cost of one process start
    in a process's first compile. Making the compiler finds the library and
    those processes load it, so a file found that is no library raises
    `Compile_error` at each compile, where it fails making tinygrad's
    compiler. Each process is a small C program per library that tolk
    carries, run from memory on Linux and elsewhere from a file in
    `Helpers.cache_dir`, or in the temporary directory where the cache cannot
    be written or its file cannot run: macOS, Metal's only system, has no
    memory file to run. Such a file runs only if it is this user's regular
    file, reached through no link, that no other user can write and macOS has
    not quarantined; another is written again, since a shared directory lets
    another user plant it. A process serves compiles in turn over a socket
    (two pipes on Windows), idle between them in a list every domain takes
    from; a compile that finds none idle starts another. A compile whose
    process ends without a reply raises `Compile_error` naming the signal or
    exit status, where it would end tinygrad's worker, and that process
    serves no other;
  - the NVRTC process sets `CUDA_CACHE_DISABLE=1`, and the comgr process
    `AMD_COMGR_CACHE=0`, a cache on by default from ROCm 7.0 under
    `~/.cache/comgr`, which tinygrad leaves on: tolk's table is the one cache
    it keeps.
- **Reason:** (a). An idle pooled domain takes part in every minor
  collection. A tinygrad worker process has its own settings, so a `Context`
  entered while it compiles (the construction check's `CHECK_OOB=0`) leaves
  the others alone; process-wide settings shared by compiling domains would
  let one domain's override and restore clobber another's. The calling domain
  renders too and lives as long as the program, which nothing recycles, so a
  table of movements per domain would keep its nodes for the program's life;
  one rewrite is a scope that ends. MTLCompiler
  re-parses its options into LLVM's global registry, which is not
  thread-safe, comgr serialises a process's compiles on its own mutex, and
  NVRTC 13.3 and 13.4's assembler crashes on some valid sources, hence the
  processes; a lone kernel goes to one too so that a crash has one outcome,
  and a serving one saves the library's setup, about 100 ms for NVRTC.
  NVRTC's cache key misses the math options (`--use_fast_math`,
  `--prec-sqrt`, `--prec-div`, `--ftz`), `-I`, `-D` and `--std`, so it could
  serve tolk a binary built under other options or against other headers;
  comgr's cache served no measurable time over a cold table.
- **Pinned by:** `Tolk.Setting › context › is not seen by the other domains` and
  `Tolk.Setting › context › binds for the domains spawned while it runs`;
  `Tolk.Worker` (every test); `Tolk.Codegen › programs are kept › calls from
  several domains at once make one program, compiled once`; `Tolk.Decomp_dtype,
  compiled in parallel › targets lowered on several domains at once get their
  own rules`; `Tolk.Compiler_metal › compiles in processes of their own` (every
  test), `› a library that does not load` (every test), `› a library without
  MTLCompiler's functions`, `› a library that makes no service`, `› a process
  that cannot start`, `› a cache that cannot be written`, `› a cache whose
  program cannot run` and `› cache › a compile the table holds starts no
  worker`; `Tolk.Compiler_metal (fork) › a forked process` (every test);
  `Tolk.Compiler_amd › compiles in processes of their own` (every test) and `› a
  process that cannot start`; `Tolk.Compiler_cuda › compiles in processes of
  their own` (every test), `› a library that does not load` (every test), and `›
  a process that cannot start`; `Tolk.Compiler_amd › a library that does not
  load` (every test); `Tolk.Compiler_cuda (fork) › a forked process` (every
  test); `Tolk.Indexing › apply_movement_op › laws › one memo keeps the index
  of each movement it holds`.

## D6. Devices are named, never parsed

- **tinygrad:** `device.py` `_Device._canonicalize`, `_Device.get_class`,
  `Compiled.__init__`, `Compiled._select_iface` (device strings split at `:`).
- **tolk:** `Device.renderers`, `Device.renderer`.
- **Differs:** the caller gives a target, whose device is a kind and whose
  `arch` is the device's; tolk picks the renderer and compiler from the
  target and never parses the name, but for one exception: a name starting
  with `DISK` is a disk, as tinygrad reserves it and rig names its disk
  device (`Ops.on_disk`, `is_disk_device`, `copy_to_device`, `clone`).
- **Reason:** (c).
- **Pinned by:** for targets, `Tolk.Device › renderer takes a device's
  name as it is` and `Tolk.Device › renderer › picks a device's renderer as
  tinygrad does`; for the disk, `Tolk.Ops › several devices › on_disk holds
  for a disk device`, `Tolk.Ops › several devices › copy_to_device rejects
  a disk and a weak type` and `Tolk.Ops › storage › clone rejects a disk`.

## D7. Withdrawn

A device's stamp slots followed nx.device's `Submission.record`, so a batch's
kernels were spans of a profile. rig times each submission itself, and tolk
stamps nothing (D146).

## D8. The disk cache maps strings to strings

- **tinygrad:** `helpers.py` `diskcache_get`, `diskcache_put` (an SQLite
  database of pickled values, keyed by strings, integers or dictionaries of
  columns), `diskcache_clear`.
- **tolk:** `Helpers.Diskcache`; `Diskcache.generation`, `Diskcache.renew`
  and `Diskcache.clear`.
- **Differs:** keys and values are strings, which callers encode. Each entry
  is a file under `CACHEDB`, a directory, written aside and renamed into
  place, holding the digest of its key and value: an entry damaged on disk
  raises instead of reading as another value, and a graph's reader makes it
  anew. Tables are versioned by tolk's own version. Entries belong to a
  generation, an empty file named by a count and a unique name; a clear
  makes one past the newest, which empties every table at one step, then
  removes the old files. A tolk
  from before generations reads an entry of one as a write in progress: its
  clear and trim remove it only once an hour old, and its trim does not
  count it.
- **Reason:** (a). OCaml has no pickle: a cache returning any type would be
  `Marshal`, which crashes when a rebuilt program reads a value of a changed
  type. Without an SQLite transaction, writing aside and renaming keeps an
  entry whole when a writer crashes or races another, and the digest keeps a
  changed byte from running as another kernel. Without SQLite's drop, a clear
  that removes files one by one orders against no call: tinygrad's
  (`diskcache_clear`) drops the tables one by one, each drop committed on its
  own, so a reader between two drops sees one table cleared and another not.
  A generation per table would cost what one costs while promising less.
- **Pinned by:** `Tolk.Helpers › Diskcache › get reads back any key and
  value put`, `Tolk.Helpers › Diskcache › behaves as a table of entries per
  table`, `› a put and a get on another domain come before or after a
  clear`, `› a put and a get in another process come before or after a
  clear` and `Tolk.Helpers › Diskcache on two domains › behaves as a table
  of entries, whatever the order of the calls`, which run with `CACHEDB`
  set, as dune runs them, and `Graph › a damaged entry
  reads back as its graph, or misses`.

## D9. Narrow float conversions are IEEE conversions

- **tinygrad:**
  - `dtype.py` (`float_to_bf16` rounds to float32 with
    `truncate[dtypes.float]`, then to bfloat16; `const` converts an integer
    to a float through a double, `float(val)`);
  - `codegen/decomp/dtype.py` `l2i`, `f2f`, `pm_float_decomp`: `l2i` converts
    a long to a float32 word by word; an emulated cast narrows its source to
    float32 first; `f2f` flushes subnormals to zero both ways, but for those
    within half a subnormal ulp of the least normal, which round up to it,
    and narrows every NaN to its format's one quiet NaN of its sign;
  - `renderer/cstyle.py` `ClangRenderer.extra_matcher`: Clang casts a
    double to half through a float32, rounded to nearest.
- **tolk:** `Dtype.encode_format` (one rounding for every narrow float) and
  `Dtype.float_of_integer`; `Decomp_dtype.long_to_float`,
  `Decomp_dtype.narrow` and `Decomp_dtype.f2f`; Clang's double to half cast in
  `Cstyle.clang`'s matcher, through `Decomp_dtype.narrow`.
- **Differs:** converting to a narrow float rounds once, to nearest with ties
  to even, from any source, and treats every value as IEEE does, folded or
  emulated:
  - a double rounds to bfloat16 once, and an integer converts once, from its
    exact value. An emulated cast narrows a double, or an integer more
    precise than a float32, to float32 by rounding to odd, so that the cast's
    rounding to the narrow float is the one rounding; an emulated 64-bit
    integer converts to a float32 once. Clang's double to half cast narrows
    to float32 the same way;
  - emulation keeps subnormals, both ways.

  A NaN converts as the NaN rule allows, which no golden distinguishes:
  `Dtype.truncate`, the folder, quiets it and, in a 16-bit or wider float, keeps
  its sign and the top of its payload (an 8-bit float's is its format's NaN, D10
  and D20); the bfloat16 cast of Clang and HIP (`Cstyle.pm_manual_bf16_cast`)
  shifts a float32's bits, so a signalling NaN may stay signalling; `f2f`
  narrows every NaN to its format's quiet NaN of its sign, as tinygrad's does.
  Each gives a NaN, signalling only from a signalling one, so a folded NaN and
  the same NaN computed in a kernel may differ in bits. A copy or a selection,
  which converts nothing, keeps a signalling one (D62).
- **Reason:** (b). rune folds constants with `Dtype`, and nx converts eagerly
  with one rounding; an emulated kernel exists only because its target lacks
  the type, so it must give the values a native one gives. The rounding to
  odd of `Decomp_dtype.narrow`: (b). rune's `Lower_arith.store_real` narrows
  a double or an integer to float32 with it before it encodes a narrow float,
  so that the narrow float rounds once. The emulated cast's own rounding to
  odd: (e). rune hands an emulated cast only a float32, but without it the
  cast rounds twice: on the CPU with `EMULATED_DTYPES=bfloat16`, the float64
  `1 + 2**-8 + 2**-40` cast to bfloat16 gives `0x3f80` (1.0) where
  `Dtype.truncate`'s one rounding gives
  `0x3f81`. The 8-bit facets: (e). rune hands tolk no 8-bit float, storing
  each as `Uint8` (rune's `Lower_arith.dtype`), but without them, under
  `EMULATED_DTYPES=fp8e4m3,fp8e5m2`, an e4m3 cast of `2**-9` gives `0x00`
  where tinygrad's folder (`dtype.py` `float_to_fp8`) gives `0x01`, and the
  float64 `1 + 2**-4 + 2**-40` becomes e4m3 `0x38` where the folder gives
  `0x39`. Clang's double to half cast: (e), as the rounding to odd; rune
  hands tolk no double to cast to half, but without it `65520 - 2^-20` is
  infinity where one rounding gives `65504`.
- **Pinned by:** `Tolk.Dtype › truncate › truncation.golden` (the near-tie
  rows), `Tolk.Dtype › truncate › bfloat16 rounds once, to the nearest, ties
  to even` and `Tolk.Dtype › truncate › an integer rounds to a narrower float
  once, from its value`; for emulation, in `Tolk.Decomp_dtype › narrow float
  conversions are IEEE conversions`: `an emulated narrow float keeps its
  subnormals, both ways`, `a double, or an integer more precise than a
  float32, rounds once`, `an emulated 64-bit integer converts to a float32
  once`, `an fnuz format stores an underflow to negative zero as positive
  zero`; `Tolk.Decomp_dtype › NaNs › f2f narrows a quiet NaN to a quiet NaN`
  and `› an emulated cast of a quiet float32 or double NaN is a quiet NaN`;
  `Tolk.Decomp_dtype › goldens`; in rune, `rune jit › jit › an integer cast
  to a float gives eager's bits` and `Rune.jit on a GPU › integers › an
  integer cast to a float gives eager's bits`, whose ties with a bit below
  float32's precision a cast that rounds twice gets wrong. The 8-bit facets:
  the e4m3, e5m2 and fnuz cases of `Tolk.Decomp_dtype › narrow float
  conversions are IEEE
  conversions › an emulated narrow float keeps its subnormals, both ways`,
  and `Tolk.Decomp_dtype › emulated narrow floats › an emulated cast of
  doubles near every <format> tie rounds once` for each 8-bit format.
  Clang's double to half cast: `Tolk.Codegen › a double rounds once to half
  on the host` and `Tolk.Cstyle › extra_matcher`'s `rewritten.tolk`.

## D10. Withdrawn

e4m3's `0xff` decoded to a negative NaN. Under the NaN rule a decoded NaN's
sign is unspecified, so `Dtype.decode_format` decodes every e4m3 NaN as
tinygrad's positive `math.nan` (`dtype.py` `fp8_to_float`).

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
- **Differs:** the digest is BLAKE2b-256, of a writing of the operation, type
  and argument that tells every two arguments apart, field by field, where
  tinygrad's `str` writes every NaN `nan` and tolk's repr prints an `Argument`
  stage as a `Whole` one (D27, D465): a float enters by its bits, a node an
  argument holds by its key, a call's write marks not at all.
- **Reason:** (a) for the digest: keys name compiled programs in caches that
  tolk alone writes, and OCaml's standard library has MD5 and BLAKE2 but not
  SHA-256. (e) for the writing: a key that two different nodes share serves
  one's schedule or program to the other, and tinygrad's `str` writes alike
  NaNs of different bits, as tolk's repr writes an `Argument` stage and a
  `Whole` one alike, and a program's target whose fields split one text two
  ways.
- **Pinned by:** `Tolk.Ops › key › tells arguments apart`, `› two nodes of one
  key are one node, but for a call's marks`, `› tells apart arguments a repr
  prints alike`; `Graph › a graph of any argument reads back as itself`.

## D13. Folding reads and writes committed constants at their width

- **tinygrad:** `uop/symbolic.py` (`fold_const_alu`), which writes a
  folded integer result at its type's width; `symbolic_simple` (the collapse
  of committed const conversions); `uop/ops.py` (`_min_max`), which bounds an
  operand by its written value; `uop/weak.py` (`uncast_const`), which leaves
  the literal bare. tinygrad's own `TestModularWraparound` expects the
  wrapped results and is marked `xfail_broken_const_wraparound`: its cases
  fold to them at this commit, but for the signed one of `test_div`, whose
  `//` floors where the test truncates.
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
  which applies to signed `x` only, from a wrong answer where the offset wraps;
  D24 keeps that rule to offsets that do not wrap.
- **Reason:** (b): every integer expression's compiled value is eager nx's
  modular one, signed included (RFC 0012's Law 1). rune's `rune jit ›
  classes › integer constants keep their bits and wrap` and `› a square of a
  product whose constant squares to 0`, at every signed and unsigned width,
  compile constants whose arithmetic wraps against eager nx.
- **Pinned by:** `Tolk.Symbolic › symbolic_simple › constants › an operation
  reads committed constants at their width`, `Tolk.Symbolic ›
  symbolic_simple › constants › a comparison reads a committed constant at
  its width`, `Tolk.Symbolic › committed constants hold their type's value`,
  the law `Tolk.Symbolic › laws › sym keeps the value of an integer
  expression at a committed width, wrapping included`, and `Tolk.Symbolic ›
  tinygrad › tests.golden › TestModularWraparound.<test>`; the offset's
  interaction with D24: `Tolk.Symbolic › integers wrap › an offset crosses a
  comparison only where neither side wraps` (the committed case).

## D14. Withdrawn

Folded comparisons of NaN constants follow IEEE: `uop/symbolic.py`
`fold_const_alu` reads each operand through `const_arg`, so `nan != nan` folds
to `True` there as in `Shape.fold_const_alu`.

## D15. Withdrawn

A compiler loaded its library at the first compile. As tinygrad's
`runtime/support/compiler_cuda.py` `NVRTCCompiler.__init__` (`nvrtcVersion`),
`runtime/support/compiler_amd.py` (`HIPCompiler` asserts comgr is loaded) and
`runtime/ops_metal.py` (`MetalCompiler`) do when the compiler is made,
`Compiler_cuda.nvrtc`, `Compiler_amd.hip` and `Compiler_metal.compiler` find
their library, raising `Failure` without it, so `Cstyle.cuda`, `Cstyle.hip`,
`Cstyle.metal` and `Device.renderer` raise it too; the processes that compile
load it (D5).

## D16. Withdrawn

A CUDA conversion to an 8-bit float kept an infinity special. rune hands tolk
no 8-bit float, so `Cstyle.cuda` saturates, as `renderer/cstyle.py`
`base_rewrite` does: an infinity becomes the greatest finite value of its
sign. This departs from nx's e5m2 and from `Dtype.encode_format`, which keeps
an infinity in e5m2 and makes it a NaN in e4m3, as `dtype.py` `float_to_fp8`
does.

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
  in `int`, and Clang one on an `__fp16` in `float`, and only an assignment
  narrows the result back. The source is tinygrad's for the kernel with a
  cast to its own type after each inlined scalar operation of one of these
  types that its user does not store. The unsigned types take tinygrad's
  rule, which avoids the signed overflow of `int`. Vectors are not promoted,
  and CUDA's `__half` and `nv_bfloat16`, HIP's `_Float16` and Metal's `half`
  and `bfloat` compute natively, so they keep tinygrad's source; so does
  every kernel without such an inlined operation.
- **Reason:** (b). An inlined operation would neither wrap nor round until the
  kernel stores: `(char)100 + 100`, halved, gives `100` where each operation
  wrapping gives `-28`. Eager nx and compiled code agree: nx computes each
  operation in its own type, as the folder (`Ops.exec_alu`) and tinygrad's
  interpreter do.
- **Pinned by:** `Tolk.Cstyle › narrowing` and `Tolk.Cstyle › sources`;
  `Tolk.Cstyle › execution on the host › rounds each operation on halves to
  a half`; and the slow `Tolk.Cstyle › execution on the host › a kernel over
  a narrow type wraps and rounds as the interpreter`, over each type's whole
  range. The unsigned types' rule: `› wraps each operation on unsigned
  chars`. Clang's, HIP's and Metal's rules were checked with their
  compilers; CUDA's rests on the operators `cuda_fp16.hpp` and
  `cuda_bf16.hpp` declare, and is on the hardware checks of
  `test/README.md`.

## D18. Withdrawn

Metal computed a bfloat16 `TRUNC` in float32, where tinygrad truncated the
bfloat, which Metal has no `trunc` of. `renderer/cstyle.py`
`MetalRenderer.extra_matcher` now computes `TRUNC` in float32 with `SQRT`,
`EXP2`, `LOG2` and `SIN`, as `Cstyle.metal_extra_matcher` does.

## D19. Withdrawn

A hierarchical allreduce to one device landed there. Withdrawn: multi-device
is out of scope, see README.

## D20. Withdrawn

Storage kept an e5m2 NaN's payload bits. rune hands tolk no 8-bit float, so no
consumer folds such a bitcast: `Dtype.encode_format` stores every e5m2 NaN as
`0x7f` of its sign and `Dtype.decode_format` reads it as the NaN of its sign,
as `dtype.py` `float_to_fp8` and `fp8_to_float` do.

## D21. Withdrawn

A reshape of a value sharded on two axes divided each by its own shard count.
Withdrawn: multi-device is out of scope, see README.

## D22. An emulated long converts no float past its words' range

- **tinygrad:** `codegen/decomp/dtype.py` (`l2i` makes a long's low
  word by casting the float to an int32, and its high word by casting the
  float over 2^32).
- **tolk:** `Decomp_dtype.l2i` (the cast of a float to a long).
- **Differs:** the words are the quotient and remainder of the truncated
  float's magnitude by 2^32, each converted from a float the word holds, and
  negated as a long for a negative float. tinygrad converts a float past
  2^31 to an int32 word, which C leaves undefined.
- **Reason:** (b): rune's kernels on a target without 64-bit integers cast
  floats to longs through this emulation, and must give the long nx gives.
- **Pinned by:** `Tolk.Decomp_dtype › emulated 64-bit integers › an
  emulated cast of a float32 to a 64-bit integer converts no float to a word
  that cannot hold it`.

## D23. Withdrawn

A shard selected through a movement read its own shard. Withdrawn:
multi-device is out of scope, see README.

## D24. Rewrites keep IEEE and modular values

- **tinygrad:** the rules and bounds each facet names below.
- **tolk:** each facet's rules below.
- **Differs:** tinygrad's rewrites assume real arithmetic. A rewrite here
  keeps the value compiled code computes, bit for bit: an integer of a
  committed type wraps at its width, and the lowering computes signed
  arithmetic on the unsigned bit pattern; a float follows IEEE. A rewrite
  that holds only in real arithmetic applies to integers and booleans, and to
  a committed integer only where no value it computes wraps (`Ops.exact`).
  Index arithmetic is weak (`Weak_int`), never wraps, and keeps every fold
  tinygrad makes, so the restrictions touch float and committed-integer
  values only. The one rounding that stays is that of float powers, a
  constant one computed by products and square roots and any other as
  `exp2 (y * log2 x)` (`Transcendental.xpow`).
  - **Integer bounds wrap.** tinygrad: `uop/ops.py` (`wrapped`, which `_min_max`
    applies to an unsigned sum, difference, product, left shift and complement),
    `UOp._min_max` (a cast, a constant table). tolk: `Ops.at_width`,
    `Ops.min_max`, `Ops.cast_bounds`, and the constant table's case of
    `Ops.compute_min_max`. The bounds of a committed integer that leave its type
    are one value wrapped, or the type's, signed or not, whatever the operation,
    where tinygrad wraps those of the unsigned operations alone, and to the
    type's, one value included; an integer cast to an integer keeps its
    interval, which then wraps, where tinygrad gives a cast that leaves its type
    the type's bounds, one value included; a constant table holding a NaN has
    its type's bounds, as a NaN constant has. tinygrad folds int8 `(y + 100) <
    0` for `y` in `[0, 50]` to `false`, which is `true` at 40.
  - **Wrapping rules.** tinygrad: `uop/symbolic.py` `symbolic` (`(x // c1) //
    c2`, `c0 + x < c1`, `-x < -y`, `-x < c`, the maximum from a selection),
    `reduce_mul_chain` (a factor out of a maximum), `fits`;
    `codegen/simplify.py` `pm_reduce_collapse` (`x + y < c`, `x * y < c`)
    and `pm_reduce_load_collapse` (`x + y = c` under a cast). tolk:
    `Shape.symbolic_simple`, `Shape.symbolic`, `Simplify.solves_sum`,
    `Simplify.pm_reduce_collapse`, `Simplify.pm_reduce_load_collapse`,
    `Symbolic.reduce_mul_chain`. tinygrad keeps `c0 + x < c1`, the
    comparisons of `Simplify` but the last, and a factor out of a maximum
    from unsigned integers, and applies them to signed ones, taking signed
    overflow to be undefined and checking only that the constants it makes
    fit the type. tolk applies each to a signed integer, and `x + y = c` to
    any committed integer, only where every value it computes fits the type,
    and `-x < -y` and `-x < c` to a signed one only where no negation wraps.
    The comparisons of `Simplify` apply to integers only, since moving a
    float term rounds, and `x + y = c` only under a cast that does not
    narrow, where tinygrad takes any cast whose operand's bounds fit its type.
    `where (a < b) c (a + (c - b))` is a maximum for a signed `a` only where
    `a + (c - b)` does not wrap.
  - **Float folds.** tinygrad: `uop/symbolic.py` `symbolic_simple` (`x + 0`,
    `x / x`, `x * 0`). tolk: `Shape.symbolic_simple`. A float `x + 0` is `x`
    only for `-0.`; `x * 0` is `0` for integers and booleans only; `x / x`
    is gone, since true division is always a float, NaN at 0.
  - **Signed zeros.** tinygrad: `uop/symbolic.py` `symbolic` (complementary
    selections). tolk: `Shape.symbolic`. `where c t 0 + where c 0 f` becomes
    `where c t f` for integers and booleans only: at `t = -0.` the sum is
    `+0.`.
  - **Reassociation.** tinygrad: `uop/symbolic.py` `symbolic_simple` (two
    constants of an associative operation), `symbolic` (like terms, a sum of
    two selections, constants to the end), `reduce_mul_chain`, `no_overflow`.
    tolk: `Shape.fold_where_closure`, `Shape.symbolic_simple`,
    `Shape.symbolic`, `Symbolic.reduce_mul_chain`. Sums, products and maxima
    regroup for integers and booleans only, and a factor leaves a float
    reduction nowhere: a float sum or product rounds at each step, and a
    maximum keeps a NaN only as its first operand, so `max (NaN, max (x,
    0.))` is NaN, where `max (x, max (0., NaN))` is `x`.
    `x + x` is still `x * 2`, which is exact, and `(x * -1) * y` is
    `(x * y) * -1`, since a negation commutes with rounding. tinygrad
    restricts these rules to integers too, but not the two stage fold of a
    product or a maximum, nor `x + x * -1`, which it makes `x * 0`, `-0.` at
    `x = -1.` where the sum is `+0.`. Its check that an intermediate does not
    overflow (`no_overflow`) holds wherever tolk would check it, since the
    bounds of a committed integer wrap (`Ops.at_width`), so tolk makes none.
  - **Maxima.** tinygrad: `uop/symbolic.py` `symbolic`. tolk:
    `Shape.symbolic`. A maximum by bounds applies to integers only, since a
    float's bounds leave out NaN and the order of zeros. A selection that
    computes a maximum becomes one for integers only: rune builds a float
    maximum that follows IEEE from selections, which the fold would turn
    back into the target's own maximum, taking `-0.` and `+0.` for one
    constant. A float ReLU stays a selection.
  - **Sigmoid forms.** tinygrad: `uop/symbolic.py` `sym`. tolk: the rules
    are left out of `sym`: `x * (1 / (1 + x))` at `1e-8` is `1e-8`, where
    `1 - 1 / (1 + x)` is `0.`.
  - **Pow.** tinygrad: `uop/symbolic.py` (`simplify_pow`), `symbolic_simple`
    (`c ** x`). tolk: `Shape.simplify_pow` and the `c ** x` rule of
    `Shape.symbolic_simple`. The reciprocal of the base is taken for
    exponents of magnitude at least 1 only, where the power overflows
    whenever the reciprocal does. A float half-integer power selects `+0.`
    at `-0.` and `+inf` at `-inf`, where `sqrt` gives `-0.` and NaN.
    `c ** x` goes to `xpow` for `c = inf`, since `exp2 (0 * log2 inf)` is
    NaN. `1 ** x` is `1` of the power's type at every `x`, NaN included, as
    IEEE's `pow (1, x)` is; tinygrad does not fold it, and leaves a float `x`
    to `exp2 (x * 0)`, NaN at an infinity.
  - **Simplify's reductions.** tinygrad: `codegen/simplify.py`
    `pm_reduce_collapse` (`sum_between`, a parameter lifted out of a sum, a
    product by a boolean cast). tolk: `Simplify.sum_between`,
    `Simplify.pm_reduce_collapse`. A float sum counted in closed form is `+0.`
    over an empty part of its range, where `0 * inf` is NaN; a product by a
    boolean cast is a selection for integers only; a parameter conjunct is
    lifted out of a sum on a selection as a selection: tinygrad lifts it as a
    product by its cast, which its product rule turns back into a selection, and
    tolk's product rule does so for integers only. A sum over a range its value
    does not use, `x * n`, and a sum of a sum split in two stay: each is the sum
    of the same terms in another association, which RFC 0012's rounded sum class
    allows; a sum of `-0.` terms computed as `-0. * n` is `+0.` once the
    lowering adds `+0.` to every float sum's result, which the `x + 0` fold
    keeps.
  - **Transcendental polynomials.** tinygrad: `helpers.py` (`polyN`, from
    `0.0`). tolk: `Transcendental.poly_n`. A polynomial starts from its
    first coefficient, so the decompositions do not rely on the float folds
    of `0. * x` and `x + 0` to be what tinygrad renders.
  - **Tensor-core accumulators.** tinygrad: `codegen/__init__.py`
    (`pm_wmma_add`, which adds the running sum to a WMMA's accumulator
    operand). tolk: `Codegen.wmma_accumulate`, which the fold at the reduce
    calls (D260). The running sum replaces a zero accumulator: the sum's
    identity is the accumulator's initial value, set once before the loop,
    and the lowering adds `+0.` to the result, so only a partial sum's zero
    sign, which the rounded sum class leaves open, differs. Adding it, `+0. +
    acc`, would add `+0.` to each accumulator element on every iteration,
    since the `x + 0` fold is kept to `-0.`. A core of another accumulator,
    which only a core built by hand has, keeps it, and the sum adds its value
    as it adds any term, where tinygrad's rule regrouped it as `(c + acc) +
    P`.
- **Reason:** (b). Every constructor's compiled result, alone or fused,
  meets its class against eager nx, whose integer arithmetic is modular and
  whose floats follow IEEE (RFC 0012's Law 1). rune's `rune jit › classes ›
  jit gives eager's bits for exact functions` and `› … at every dtype`,
  `Lower_real › exact`, `Lower_real › transcendental › pow within 2 ulps at
  float32, 2 at float64` and `rune jit › jit › an integer maximum of a
  running sum wraps` compile drawn compositions, special values and a
  wrapping maximum against eager nx.
- **Pinned by:** each facet by value tests, which evaluate a graph with the
  reference interpreter before and after the rewrite:
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
  - transcendental polynomials: the `Tolk.Transcendental` goldens;
  - tensor-core accumulators: the sources of every tensor core,
    `Tolk.Cstyle › sources › by default › metal_tc_*`, `hip_tc_*` and
    `metal_matmul`; `Tolk.Codegen › tensor-core accumulators › the running
    sum replaces a zero accumulator`, on each target's tensor cores, one or
    two steps of K a trip; and `Tolk.Codegen › tensor cores on CUDA › a
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
  `CStyleLanguage.code_for_op` (no C-style language writes `MULACC`),
  `renderer/ptx.py` `asm_for_op` (PTX writes a float `MULACC` as `fma.rn` at
  its own type, so a half's rounds once); `uop/ops.py` `python_alu`.
- **tolk:** `Compiler_cpu` (`-ffp-contract=off`), `Compiler_amd`
  (`-ffp-contract=off`), `Compiler_cuda` (`--fmad=false`), `Compiler_metal`
  (`#pragma METAL fp contract(off)` before the source);
  `Helpers.Diskcache.version` 2, since a cached binary compiled with
  contraction answers the same key; `Codegen.fuses`,
  `Codegen.lower_reduce`, and the
  operations `Codegen.full_rewrite_to_sink` decomposes with, without `Mulacc`;
  `Cstyle.fma` (`fma` in C-style and Metal, `__builtin_fmaf` and
  `__builtin_fma` in Clang and HIP, `__fmaf_rn` and `__fma_rn` in CUDA, and
  for halves `__hfma` in CUDA, of bfloat16s too, and `__builtin_fmaf16` in
  HIP); `Ops.mulacc`, and its arm in `Ops.exec_alu`; `Shape.pm_data_invalid`'s
  `Mulacc` rule; `Renderer.t.fma`, and `Codegen.narrow_mulacc`, which
  `Codegen.full_rewrite_to_sink` applies after the products fuse.
- **Differs:**
  - no compiler contracts a product and a sum that a rendered expression
    holds together, `a*b + c`, as Clang at `-O2`, HIP, NVRTC and Metal would.
    Metal's `-ffp-contract=off` does not reach the code where its pragma
    does: on an Apple GPU a kernel gives `2^-24` without the pragma and 0
    with it, and still `2^-24` with `-ffp-contract=off` alone;
  - a sum of products of float32 or float64, on a renderer that writes a
    multiply-add and has the type natively, adds each product into its
    running sum as one multiply-add (`Op.Mulacc`), rounded once, in the order
    it adds them unfused: an accumulator's `acc + a*b` is `fma(a, b, acc)`,
    and a horizontal reduce's `a0*b0 + a1*b1 + ...` is `fma(a1, b1, a0*b0)`
    and so on, which the accumulator then adds. A horizontal reduce of
    unrolled lanes beside upcast ones sums a permuted view of the products;
    its products fuse too (`fuses` looks under the source's movements, as
    the generator's `fuses` looks under `x.base`);
  - `Decomp_op`'s `a * b + c` rule never applies: decomposition takes the
    renderer's operations without `Mulacc`. Every other product and sum
    keeps the two roundings the graph states;
  - a float `Mulacc` that a graph states, as rune's lowering of `Nx.fma` does,
    folds and evaluates rounded once, where `python_alu` rounds twice: at
    float64 by `Float.fma`, and below by a sum rounded to odd at double
    precision, which the dtype's truncation rounds once. A `Mulacc` moves inside
    the Invalid gate of each operand, and an Invalid operand makes it Invalid,
    as `pm_data_invalid` does for binary operations; no tinygrad tensor graph
    holds `MULACC`, so it has no such rule;
  - a `Mulacc` of a half, a bfloat16 or an 8-bit float rounds once too. A
    target writes it natively where its specification states one rounding
    (`Renderer.t.fma`): CUDA's `__hfma` of halves from capability 53 and of
    bfloat16s from 80, HIP's `__builtin_fmaf16`, Metal's `fma` of halves from
    Apple family 4 (fast math off), and Clang's `__builtin_fmaf16` and
    `__builtin_elementwise_fma` of halves on arm64 with `fp16` (FEAT_FP16),
    LLVM's fma of a legal half type, one `fmadd` or `fmla`.
    Elsewhere its operands widen to a float32, where their product is exact,
    a bfloat16's within guards at the ends of its range; the sum is rounded
    to odd there, and the cast back rounds once (Boldo and Melquiond, IEEE
    Trans. Computers 57(4), 2008, Theorem 3). An e4m3 or e4m3fnuz `Mulacc`
    keeps the float32 multiply-add, which rounds once on every finite
    triple. A bfloat16 `Mulacc` compiles on Metal. For a target that flushes
    subnormals, Metal's, a lane whose operand, product, sum or result is
    subnormal in its type is not promised one rounding; every other lane is:
    the float32 route keeps an addend below `2^-125` whole.
- **Reason:** (b). A reduction's result is in the rounded-sum class (RFC 0012's
  Law 1), whose error bound a fused multiply-add stays within; every other
  compiled result meets eager nx, which rounds a product and a sum apart, and
  rune's two-part products and sums are exact only where each operation rounds
  as written. The fusion is the IR's, so that only a sum's products fuse, where
  a compiler's contraction fuses any `a*b + c` of an expression. `Nx.fma` (RFC
  0015's D3) states a `Mulacc` whose compiled bits must be eager's, so its fold
  rounds once too, and `Nx.ewma`, built on it, compiles. Fusing a sum's products
  is (d): on fudge rune's `Jit/matmul-64` takes 7.3 us fused against 49 us, and
  the tolk bench's `lorenz/r_128_16_3_4_4_3_3_400_400/heuristic` 0.79 ms against
  1.47 ms. A narrow float's `Mulacc` is (e): the bits of one graph differed
  between hosts and between targets.
- **Pinned by:** `Tolk.Compiler_cpu › execution on the host › a product and a
  sum round twice, never fused`; `Tolk.Compiler_metal › MTLCompiler › a product
  and a sum compile under the no-contraction pragma`. NVRTC and HIP are README's
  hardware checks. `Tolk.Codegen › multiply-adds › a sum of products adds each
  into its running sum rounded once, of 2` and `› of 64` and `› a sum of
  products unrolled beside upcast lanes adds each into its running sum rounded
  once` and `› a product and a sum outside a reduction are not fused`; the
  Codegen and C-style goldens, with tolk's output naming D25 in `.tolk` files
  beside them. `Tolk.Ops › exec_alu › a float multiply-add folds rounded once`
  (its float32 case past a double's rounding ties to the wrong float32 when the
  sum is not rounded to odd); `Tolk.Symbolic › invalid values › a multiply-add
  moves inside the gate of each operand` and `› a multiply-add of invalid is
  invalid`. In rune, `Lower_arith › exact binary kinds, comparisons, where and
  fma give eager's bits` and `Lower_real › exact › fma`, and `rune jit › classes
  › jit gives eager's bits for exact functions`. On rune: missing for a sum's
  products and for `Nx.ewma`. A narrow float's `Mulacc`: `Tolk.Codegen › a
  multiply-add of narrow floats rounds once › of halves`, `› of bfloat16s` and
  `› of float8 <format>s` for e5m2, e5m2fnuz, e4m3 and e4m3fnuz, each on the
  host and each GPU, in 1, 4 and 8 lanes, against the exact sum rounded once; on
  the host also at `arm64,generic` and `arm64,generic,fp16`, the second written
  by `fmla` and `fmadd`, which `Tolk.Compiler_cpu › a half multiply-add on
  arm64` checks in the object.

## D26. A minus never meets a minus in C-style source

- **tinygrad:** `renderer/cstyle.py` `CStyleLanguage.code_for_op` (`Ops.NEG` is
  `-{x}`, `Ops.SUB` is `({a}-{b})`).
- **tolk:** `Cstyle.joined`, `Cstyle.infix`, and `Neg` in
  `Cstyle.code_for_op`.
- **Differs:** a negation or a subtraction whose operand starts with a minus
  sign, itself a negation or a negative constant, is written with a space
  between the two signs: `(a- -b)`, `- -b`. tinygrad writes `(a--b)`, which
  C, Metal, CUDA and HIP read as the decrement operator and reject. Every
  other source is written as tinygrad writes it.
- **Reason:** (b). rune lowers `Nx.sub` to `Ops.sub` (rune's
  `Lower_real.binary`) and its FFT's twiddles with `Ops.neg` (rune's
  `Lower_fft`), so a difference of a negative constant, as `Nx.sub_s x (-1.)`
  states, or a negation of a negation can reach the renderer, and a rejected
  source is a failed program. No rune test is shown to reach it.
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
  `Ops.key`; `Shape.fold_bitcast`.
- **Differs:** a float constant is its bits. `Dtype.const` keeps a NaN's sign
  and the payload its type holds, a signalling NaN staying one, in a float of
  16 bits or more (an 8-bit float's NaN is the one its storage holds, D10 and
  D20). Constants compare and hash by their bits, so NaNs of different bits
  are different nodes with different keys: a schedule cached under one NaN's
  key is never the schedule of another's. A parameter's bounds and value key
  by their bits too. No golden reads a key; tinygrad's writes every NaN
  `nan`. `fold_bitcast` reads a float constant as its
  type stores it, where tinygrad's conversion quiets a signalling NaN and
  makes an 8-bit NaN canonical. A NaN an operation makes from other values
  is `Dtype.nan`, tinygrad's `math.nan`, whatever the host's FPU gives, so
  that folding does not depend on the host: `exec_alu` makes the NaN of an
  invalid operation (`inf - inf`, `0 * inf`, `0 / 0`, a square root or
  logarithm of a negative) canonical, where x86 gives a negative NaN. The
  NaN rule leaves that NaN's bits free, so this part follows tinygrad.
- **Reason:** (b): under the NaN rule a bitcast keeps a NaN's bits, as a
  kernel's and nx's do, so a bitcast that rune folds must keep them, which a
  constant can only if it holds them; a rewrite must keep a graph's value.
- **Pinned by:** `Tolk.Ops › identity › NaN constants of different bits
  are different nodes`; `Tolk.Ops › key › tells apart NaN constants of
  different bits`; `Tolk.Ops › exec_alu › an invalid operation's
  NaN is the canonical positive quiet NaN, whatever the host gives`;
  `Tolk.Symbolic › symbolic_simple › casts › a bitcast round trip of every
  8- and 16-bit word folds to its value`; `Tolk.Dtype › const › const keeps
  the bits of every 8- and 16-bit float word but an 8-bit float's NaN
  codes`.

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
  unsigned integers, where tinygrad's widening and narrowing makes every NaN
  its format's quiet one. Arithmetic still converts, and gives a NaN as the
  NaN rule allows (D9). A move keeps a merged access's lanes together, so it
  loads and stores vectors of the storage, which CUDA does not name as the
  renderer writes them (`unsigned_short4`, `unsigned_char4`): the CUDA
  renderer declares a vector of `unsigned char` or `unsigned short` as it
  declares one of `half`, but for 2 and 4 bytes, which it writes as one
  integer word (D482).
- **Reason:** (b). Under the NaN rule a move keeps a NaN's bits, and nx's
  eager copy, movement or `where` keeps a signalling NaN, so a compiled
  function computes what it computes eagerly only if the emulated moves keep
  the bits too. rune stores the 8-bit floats and `float4_e2m1fn` as `uint8`
  (`Lower_arith.of_byte`, `Lower_arith.to_byte`), so tolk no longer emulates
  an 8-bit float for it, and `float16` and `bfloat16` in their own types
  (`Lower_arith.dtype`), so the move that remains is a compiled copy of
  `bfloat16` or `float16` storage on a target that emulates it: `bfloat16` on
  a riscv64 host, on an Apple GPU before family 6 and on CUDA before sm_80.
- **Pinned by:** the `Decomp_dtype` suite (`test/codegen/decomp/decomp_dtype`):
  `NaNs › an emulated copy keeps every NaN code, a signalling one's included`
  and `› an emulated selection keeps every NaN code, a signalling one's
  included`, on every emulated float, and the goldens of the `where`, `flip`,
  `gather` and `pad` kernels; the Codegen suite's `emulated floats' vectors on
  CUDA › a copy of every code, a thread's lanes moved together, keeps it`
  (`bfloat16` on sm_75, `fp8e4m3` on sm_80 and `fp8e4m3fnuz` on sm_89) and
  `› writes an sm_75 copy of bfloat16 lanes with its vector declared`. On
  rune: missing.

### A NaN constant is written from its bits (D156)

- **tinygrad:** `renderer/cstyle.py` `base_rewrite` (every NaN constant is
  the language's `nan` cast to its type), `CStyleLanguage.nan` (`NAN`),
  `ClangRenderer.nan` (`__builtin_nanf("")`), `CUDARenderer.render_kernel`
  (`#define NAN (__int_as_float(0x7fffffff))`) and `HIPRenderer.__init__`
  (HIP's 8-bit floats, `f32_to_fp8(NAN, …)`).
- **tolk:** `Cstyle.nan_const`, the `bitcast` of each language
  (`Cstyle.cstyle`, `Cstyle.metal_lang`, `Cstyle.cuda_lang`) and
  `Cstyle.cdna_rewrite`.
- **Differs:** a NaN constant (D27) is written as the language's bitcast of
  its bits, an unsigned literal of its width:
  `__builtin_bit_cast(float, (unsigned int)(0xffc00005u))` in C and HIP,
  `as_type<float>(…)` in Metal, `tg_bitcast<float>(…)` in CUDA. A program
  stores the constant's sign and payload, a signalling NaN staying one,
  where tinygrad's stores the conversion of one NaN of the language. An
  infinity's writing is unchanged.
- **Reason:** (b). Under the NaN rule a move keeps a NaN's bits. rune lowers
  every float constant from the bits nx stores for it (rune's `Lower.padded`
  fills a padded load with its fill's bits through `Shape.pad`), and padding
  moves those bits into the result, so a compiled kernel must store the bits
  eager nx stores, as it must store the bits of a constant folded from a bitcast
  (D27).
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `a NaN constant
  is stored with its sign and payload › *`; and the `*_inf_nan_*` goldens of
  the `Cstyle` suite, for every renderer.

### A folded selection or negation keeps a NaN's bits (D176)

- **tinygrad:** `uop/ops.py` (`exec_alu`), which truncates every
  result to its type through `dtype.py` (`truncate`), whose
  conversions to `float`, `half` and `bfloat16` quiet a signalling NaN.
- **tolk:** `Ops.exec_alu`, whose `Where` and `Neg` give a NaN as
  `Dtype.const` holds it, and `Shape.fold_const_alu`, which reads a float
  constant operand as `Dtype.const` holds it, its bits (D27).
- **Differs:** a folded selection is the NaN it picks, and a folded negation
  that NaN with its sign flipped, a signalling NaN staying one, in a float of
  16 bits or more. Every other operation still truncates its result. tolk
  has no absolute value or sign copy among its operations: rune computes
  both on a float's bits (`Lower_bits`).
- **Reason:** (b). Under the NaN rule a selection and a negation keep a
  NaN's bits, as LLVM's LangRef states ("Behavior of Floating-Point NaN
  values"). nx's `where` keeps a signalling NaN (D62), and rune lowers it to
  `Ops.where` (`Lower_arith`), so a selection tolk folds must give the bits
  the kernel's selection gives. The `Neg` half has no rune call site (rune
  negates on a float's bits, `Lower_bits.negated`; `Op.Neg` reaches the
  folder only from `Decomp_op`'s rewrite of `x * -1`) and rests on D27's
  rule that a rewrite keeps a graph's value: a `Neg` the folder leaves
  renders as C's `-x`, which flips the sign bit alone.
- **Pinned by:** `Tolk.Ops › exec_alu › a selection keeps a NaN's bits and a
  negation flips only its sign, a signalling NaN's included`; and
  `Tolk.Symbolic › symbolic_simple › constants › a folded selection keeps a
  signalling NaN's bits, and a negation flips only its sign`. On rune:
  `Lower_arith › a folded selection keeps a NaN's bits, and a negation or an
  absolute value sets only its sign`.

### Clang selects a 16-bit float by its bits (D325)

- **tinygrad:** `renderer/cstyle.py` `CStyleLanguage.code_for_op`'s
  `Ops.WHERE` (`(c?a:b)`) and `base_rewrite`'s gated `Ops.LOAD`
  (`(gate?*p:alt)`), under `ClangRenderer.type_map`, which renders a half as
  `__fp16`, and a bfloat16 as a `__bf16` off x86.
- **tolk:** `Cstyle.bit_where`, Clang's `code_for_op` of `Op.Where`, and
  `Cstyle.bit_loads`, in its rewrite, for `Float16`, and for `Bfloat16` where
  it is a `__bf16`.
- **Differs:** a scalar selection of such a value, and a gated load's merge
  with its alternative, choose between the arms' bits:
  `__builtin_bit_cast(__fp16, (unsigned short)((c?A:B)))`, each arm read as
  an `unsigned short`. Clang computes `c?a:b` of two `__fp16`s in float and
  narrows the result back, which on x86 (`vcvtph2ps`, `vcvtps2ph`) quiets a
  signalling NaN wherever LLVM's folds leave the round trip. A `__bf16` is
  not promoted, but riscv64's backend, in clang 19, moves a selected one
  through float32 and narrows it with `__truncsfbf2`; arm64's keeps its bits,
  and takes the rule for one form across targets. Vector selections are not
  promoted and are unchanged.
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
- **Reason:** (c). rig's host calls linked code with one ABI,
  `void f(const void *params, int64_t first, int64_t end)`, places each
  buffer's address among the parameters, and runs a launch's blocks on
  several cores at once, each call with its own `first` and `end`. `NAME` is
  `Codegen.symbol` (D485), and on x86_64 Windows the entry is declared
  System V's (D178). The copy is inline because Clang compiles a copy of
  many bytes as a call of the process's `memcpy`, which a remote Windows
  host's object, whose target names no system (D178), would call by System V.
- **Pinned by:** the `Codegen` suite, `programs › a host program's binary
  holds rig's host entry after the kernel` and `› a host program runs as a
  launch of rig's host, with ten scalars past the argument registers` and
  `› a split host program calls no function it does not define`; the
  Tolk_engine suite, `host programs in blocks › a program writes the same
  bits in any blocks` and `link and run › custom kernels with
  variables` (four tests).

## D29. A float32 tensor core takes operands widened from its narrow input

- **tinygrad:** `codegen/opt/postrange.py` `_apply_tc_opt` (a core applies
  only when both operands of the multiply have its input dtype) and
  `Scheduler._apply_tc_opt`.
- **tolk:** `Postrange`'s `tc_operand`, `try_core` and `use_wmma`.
- **Differs:** a core whose input is a narrow float (`half`, `bfloat16`, an
  8-bit float) and whose output is `float32` also takes a `float32` operand
  that is a cast of its input dtype, and multiplies the narrow value. The core
  and the widened products may differ in bits within `Nx.contract`'s bound,
  `γ(K + 1, 2u) (|init| + Σ|a||b|)` for `float32`'s `u`; a core shown to
  exceed it is disabled. A core with a narrow output takes no widened operand,
  since its sum rounds to the narrow type. A `float32` operand that is not
  such a cast takes no core, and tolk never narrows one by a cast that rounds
  (an integer the narrow type holds exactly enters at it, D451): a `bfloat16`
  core would round a value the program did not round, outside
  `Nx.contract`'s bound. So attention's values, `float32` weights by
  `bfloat16` values, run without a core. A program that wants the core rounds
  its operand itself, and this rule then applies. rune never gives tolk an
  8-bit float (rune's `Lower_arith.dtype` stores each as `Uint8`), so no
  contraction of rune's reaches an 8-bit core. Cores are tried in the
  renderer's order, so Metal keeps its `float32` core, which comes first.
  tinygrad's `dot` never builds the widened form, so its goldens do not
  change.
- **Reason:** (b). rune's contraction lowering (`Lower_reduce.entered`) casts
  each operand to the accumulator's type, `MUL(CAST f32 a, CAST f32 b)` summed
  at `float32`; without this rule it takes no tensor core on CUDA or HIP,
  where the `float32` core is TF32 or absent, and gpt-oss's `float16` and
  `bfloat16` products on CUDA lose theirs.
- **Pinned by:** the Postrange suite, `apply_opts optimises a kernel as tinygrad
  does › tc_{cuda,amd}_widened_{half,bfloat}.golden` (CUDA sm_89 and HIP
  gfx1100) and `› tc_{cuda,amd}_widened_{half,bfloat}_shaped.golden`, with
  `optimising keeps a kernel's writes › small kernels › tc_*_widened_*_shaped`.
  On rune, which widens by tolk's cast (`Lower_arith.decode`): `test_jit_gpu`'s
  `products › a float16 or bfloat16 matmul is within its bound of the exact
  product`, on the machine's first GPU, at extents the core takes and refuses,
  with `› sixteen terms under float32's unit hold the bound` and `a contraction
  on matrix units holds its bound › *`; these pin `Nx.contract`'s bound,
  `γ(K + 1, 2u) Σ|a||b|`, only, since whether a core was taken is not
  observable through rune. The Codegen suite's `tensor cores on CUDA › a
  float32 operand by a widened bfloat16 takes no core, and is the exact
  product` pins the refusal.

## D30. Withdrawn

A range around calls on devices with queues was a loop of their batch, its
commands repeated per trip. A range is a loop of rig's program, one submission
per trip of each submission under it (D146).

## D31. Payne-Hanek reduces exactly, to the nearest quadrant

- **tinygrad:** `codegen/decomp/transcendental.py` `payne_hanek_reduction`
  and `sin_poly_large`.
- **tolk:** `Transcendental.one_over_two_pi`, `Transcendental.turn_fraction`,
  `Transcendental.radians` and `Transcendental.payne_hanek`.
- **Differs:** tinygrad rounds the quotient on `f < 0.5`, `f` frexp's mantissa,
  which never holds, so it returns the fraction less a quarter turn and the next
  quadrant: `|r|` reaches `pi/2` and a tiny remainder loses its low bits. It
  multiplies 32 bits of the mantissa by 96 bits of `2/pi` from a table of 190
  bits, so a float64 is reduced wrong from about 30 up, and beyond the table's
  bits for large exponents. Here the quotient rounds to the nearest quadrant on
  the fraction's top bit, so `|r| <= pi/4`; the whole mantissa, 24 bits or 53 in
  two words, is multiplied by the bits of `1/(2pi)` at its exponent, from a
  table of 1312 bits that covers a float64's greatest exponent, to 128 bits of
  the fraction, carried word by word; the table's words are selected by one
  selection per bit of their index, where tinygrad's `_take` compares the index
  with each position; and the remainder is the signed fraction, summed exactly
  in two floats from parts that convert exactly and multiplied by `2pi/2^64` in
  two floats (D126). The remainder is within an ulp of the exact one in float32
  and float64, `6381956970095103 * 2^797`, the float64 nearest a multiple of
  `pi/2`, included.
- **Reason:** (b): rune's `cos` and `tan` reduce by `pi/2` with this
  reduction from 30 in magnitude, in float32 and float64, and with
  Cody-Waite's below (`Lower_real.reduce`, `trig_switch`), and its `sin` is
  tolk's `Sin`, whose `xsin` uses it beyond its switch-over, 30 by default;
  both must give nx's values. Selecting the words by bits: (d), compile time,
  since every compiled sine or cosine of an unbounded angle builds the
  reduction: rune's compile bench row `trig/sin` makes its first call in
  0.14 s of CPU against 0.17 s (fudge).
- **Pinned by:** `Tolk.Transcendental › reductions ›
  payne_hanek_near_multiples.golden`, float32 and float64 angles near
  multiples of `pi/2`; `› payne_hanek_reduction removes quarter turns`; the
  `payne_hanek_*`, `xsin_*` and `values` goldens; rune's compile bench row
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

- **tinygrad:** `schedule/memory.py` `memory_plan_rewrite` (each entry's
  buffers from its sources after the first, through `_collect_bufs`, which
  passes buffers, `MSELECT` and `MSTACK` only; `buf_hold`, a copy's buffer
  living `last - first + 1` calls past its last); `schedule/__init__.py`
  `create_linear_with_vars` (holds the buffers among its call's sources).
- **tolk:** `Memory.viewed`, `Memory.uses` and
  `Memory.memory_plan_rewrite`'s lifetimes (`call_bufs`, the events) and
  lanes (`key`); `Schedule.create_linear_with_vars`'s `held_bufs`.
- **Differs:**
  - An `END` of ranges around calls takes the buffers of each call it runs,
    each call at its place in one trip (D476), and a call takes the buffers its
    arguments view (`SHRINK`, `BITCAST`, `AFTER`) as well as those they are.
    tinygrad takes an `END`'s sources after the first, its ranges, so a buffer
    used inside a range lives only where it is used outside it; it leaves a
    buffer only views reach unplanned, and one reached both directly and through
    a view lives only where it is reached directly, so the plan can place it
    over bytes another buffer still needs. A schedule's call holds the buffers
    its sources are or view, where tinygrad's holds those they are. No
    recorded schedule plans a buffer a view reaches.
  - A copy's buffer is free from the call after its last, as a kernel's is,
    where tinygrad holds it as many calls again.
  - Copies and kernels of a device share one lane in every schedule, where
    tinygrad keeps copies in a lane of their own. Across devices rig orders
    by allocation, which a lane avoids only in part. Where a host-run device's
    work reuses bytes a GPU copy touched, the call waits for the GPU. A copy
    lane avoids that case but not host work that writes the lane after a GPU
    copy used it; removing the wait is rig's to do.
- **Reason:** (b) for ranges and views: `Rune.scan`'s staged loop schedules
  calls inside a range, whose arguments are views that move with it, of buffers
  the calls after it read whole; rune's programs over one device store into runs
  of a buffer through views of it (D184), which left out of the plan would hold
  their memory for the whole schedule. (c) for the hold: an arena is one rig
  allocation, and rig orders work that writes an allocation after every use of
  it by any device (`Rig.section-order`, rule 3, Memory), its device's work by
  device order (rule 1) and the parts of a submission that share memory (rule
  2), and `tolk.engine` passes no unordered ref, so a later call that reuses a
  copy's bytes follows the copy, and the hold only costs memory: tinygrad's
  `test_deferred_copy_frees_chain` schedule needs 512 bytes with it and 256
  without. (b) for one lane: the same order makes a copy lane guard nothing on
  one device, and rune's fully sharded steps hold within RFC 0005's bound for a
  schedule linked per run (D173) only with one.
- **Pinned by:** the Engine suite (`test/engine/tolk_engine`): `link and run ›
  a planned buffer a range writes is not placed over one it leaves` and `› a
  buffer a range writes through a view is not placed over another`, each
  failing without its half; the Memory suite (`test/schedule/memory`):
  `memory_plan_rewrite › the buffers a call takes › a buffer a call views is
  planned` and `› a buffer viewed after another's life is not placed over it`,
  `› rules › a copy's buffer is free from the call after its last` and `› a
  copy's buffer and a call's of one device share bytes once their lives are
  apart`, the recorded `copy_chain`, `copies_share`, `random_*` plans, and the
  law over drawn schedules; the Jit suite's `jit_lower › arenas › a device's
  buffers share one arena, a copy's among them` and its recorded
  `weights_copy_lowered`, `held_constant_lowered`,
  `split_multidev_xfer_lowered` and `split_multidev_copy_lowered`; the
  Schedule suite's recorded `copy_view` schedule, whose copy reads a view of a
  buffer its call is given.

## D42. Withdrawn

A batch's command buffers, and its placeholders where a device had no window,
were pinned memory. tolk makes no command buffer, and a program's memory is
rig's (D146).

## D43. Withdrawn

A queue bufferized its own commands, and a region kept its alignment. tolk
encodes no queue (D146).

## D44. An integer cast of a weak expression computes in integers

- **tinygrad:** `uop/weak.py` `cast_weak_srcs`; `dtype.py` `promo_lattice`,
  `least_upper_dtype`.
- **tolk:** `Uop_weak.cast_weak_srcs`.
- **Differs:** a cast to a committed type over a weak expression commits the
  expression at the least upper type of the cast's type and the committed
  types of the expression and its weak sources. Where that type is a float and
  the cast's type an integer, the expression commits at `Int64` here, or at
  `Uint64` where one of the committed types is, and the 64-bit result converts
  to the cast's type as C converts an integer. tinygrad computes such an
  expression in float32, since a 64-bit unsigned integer and a signed one meet
  only at a float: `(b * 3 + 1).cast(dtypes.uint64)` rounds past 2^24. Every
  other cast commits as tinygrad's does.
- **Reason:** (b): rune's scatter kernel (`Lower_index.walk`) casts a run's
  first position, `g * len` over a weak range `g`, to `Uint64`; in float, one
  past 2^24 rounds and updates land in the wrong run.
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

- **tinygrad:** `renderer/cstyle.py` `CStyleLanguage.code_for_op` (no
  `FDIV`) and `ClangRenderer.code_for_op` (Clang's adds it);
  `codegen/decomp/op.py` `get_late_rewrite_patterns` (a target that lists
  `FDIV` gets its reciprocals as divisions, and `a * (1/b)` as `a/b`).
- **tolk:** the `FDIV` rule of `Cstyle.base_rewrite`; the `FDIV` rules of
  `Decomp_op.late_patterns`, which turn a reciprocal into `1/x` and nothing
  else.
- **Differs:** Metal, CUDA and HIP write an `FDIV` as `(a/b)`, as Clang does,
  where tinygrad's renderers fail on it. Their tables still leave `FDIV` out,
  so code generation keeps their reciprocals, and every kernel built from
  tinygrad's operations keeps tinygrad's source. On a target that lists
  `FDIV`, a product by a reciprocal stays a product: `a/b` rounds once where
  `a * (1/b)` rounds twice.
- **Reason:** (b). rune lowers nx's float division, and the power and the
  arc tangent built on it, to `FDIV`: eager nx divides as IEEE does, rounding
  once, as at `a = 0x0.000000016db99p-1022`, `b = 0x1.5d24f36473bb3p-998`,
  where a product by the reciprocal differs.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `sources › by
  default › <target>_fdiv_<type>`, for Clang, Metal, CUDA and HIP in each
  float type they have; `division › the operands tell a quotient from a
  product by the reciprocal` and the slow `› Metal divides as IEEE does,
  rounding once`. The `decomp_op` golden `late_division_fdiv`; rune's
  `Lower_real › exact › fdiv` and `› recip`, and `rune jit › classes › jit
  gives eager's bits for exact functions`. On rune: missing for a product by a
  reciprocal at the value above. CUDA's and HIP's `/` are correctly rounded by
  their compilers' defaults, which is on the hardware checks of
  `test/README.md`.

## D51. Withdrawn

An NV launch read its local memory size from a word of the device. rig's NV
driver sizes local memory as it loads a function (D146).

## D52. A -0. fill or alternative is not +0.

A selection or a pad by `-0.` keeps `-0.`: tinygrad's tests for 0 take
`-0. == 0` as true.

### A selection of -0. stays off a gated load

- **tinygrad:** `uop/symbolic.py` `pm_move_where_on_load`, whose pattern
  constant `0` matches `-0.`.
- **tolk:** `Symbolic.pm_move_where_on_load`.
- **Differs:** a selection between a load and `-0.` stays a selection.
  tinygrad moves its condition into the load's gate, and a gated load reads
  `+0.` where its gate fails, so a pad of `-0.`, lowered as a selection of
  `-0.` off the padded load, computes `+0.` on the padding.
- **Reason:** (b). `Nx.pad` fills with the value given, `-0.` included, and
  rune's compiled pad must compute eager nx's bits.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `signed zeros ›
  a pad with -0. fill renders and computes -0.` and `› a pad and a selection
  of signed zeros keep the interpreter's bits`; in rune, `Lower › padded loads
  give eager's bits`, whose examples fill with `-0.`.

### A pad by -0. fills its padding with -0. (D155)

- **tinygrad:** `mixin/op.py` `_pad_constant` and `pad_to`, which return the
  plain pad when `value == 0`, true of `-0.`.
- **tolk:** `Shape.is_zero`, used by `Shape.pad` and `Shape.pad_to`.
- **Differs:** a fill is left to the pad's own zeros only when its bits are
  all zero. A `-0.` fill takes the selection off the padded mask, so the
  padding holds `-0.` where tinygrad's holds `+0.`.
- **Reason:** (b). nx's `pad` fills with the value given, `-0.` included, and
  rune lowers a padded load (rune's `Lower.padded`) through `Shape.pad` with
  that fill, so its compiled pad must compute eager nx's bits.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `signed zeros ›
  Shape.pad by -0. computes -0. on the padding`.

## D54. Storage says what it knows of where it starts within 16 bytes

- **tinygrad:** `uop/ops.py` `ParamArg` (no such field);
  `codegen/late/coalesce.py` `memory_coalescing` (merges a run of `l`
  elements where `l` divides its element offset, taking every buffer to start
  on a 16-byte boundary); `schedule/__init__.py` `contiguous_mops_to_view`
  (stages a contiguous slice without a copy, which later kernels read with
  vector loads that take it to start aligned).
- **tolk:** the fields `phase` and `align` of the type `Ops.param_arg`, and
  the function `Ops.param_arg`, which checks them; `Shape.view_start` and
  `Shape.storage_phase`, which `Shape.param_like` gives a parameter;
  `Rangeify.debuf`, which gives them a kernel's parameter;
  `Coalesce.memory_coalescing` (the merge).
- **Differs:** a parameter or buffer carries a congruence for its start: its
  first element lies `phase` bytes past a multiple of `align`, a power of two
  up to 16, and `phase` is a multiple of its element's size or of `align`,
  whichever is less. They default to `0` and `16`, tinygrad's assumption, so
  no graph of tinygrad's changes.
  - Off the CPU device, a run of `l` elements merges where `l` divides the
    element's count from that boundary and the run is no wider than `align`,
    so every vector access is aligned to its width; an access through a
    bitcast to elements of a size the phase is not a multiple of merges
    nothing. On the CPU device, whose vector types are aligned to their
    element (`renderer/cstyle.py` `ClangRenderer.render_vector_prefix`), the
    merge takes every buffer to start on a boundary of its widest access, as
    tinygrad's takes it on 16 bytes (the width, D187).
  - A parameter made from storage keeps the storage's congruence, moved by the
    bytes a shrink of the storage seen whole skips. A shrink by a symbolic
    start is known only modulo the largest power of two its moving bytes are a
    multiple of. A symbolic start into a view that reorders or pads the
    storage is known only to its element's size. Any other view of a view that
    reorders the storage keeps its storage's, and storage on a disk keeps the
    default.
  - A stage of a view of a buffer through movements and bitcasts is that view
    when the schedule finds it contiguous (`Schedule.contiguous_mops_to_view`),
    and storage the schedule allocates otherwise: it is known at phase 0 modulo
    the largest power of two up to the buffer's alignment that the byte of the
    view's first element is a multiple of, which holds of both, and modulo 1
    byte when a size is symbolic. A view whose first or last element is padding,
    or whose ends lie farther apart or closer than a run of its size, is never
    made the view, and its stage keeps the boundary.
  - rune's staged loops give each body parameter the alignment and phase of
    what the call passes it. The congruence is part of the graph, so of a
    program's cache key.
- **Reason:** (b). rune's jit runs a program over the storage it is given, each
  buffer at its phase (rune's `Compile.storage`), and mapped weights put a
  tensor at any byte offset of its file; a vector access from an address not a
  multiple of its width faults on CUDA, and on Metal reads the wrong elements
  once adjacent vector loads at constant offsets merge. On the host, in Clang's
  vector types aligned to their size, a batched range (D30, since withdrawn)
  faulted on x86's `movaps` for windows 4 bytes off, and norn's NUTS step on
  data read from its sixth double failed verification through a stage made the
  view (`Loop.cut`); on the CPU device those types are now aligned to their
  element.
- **Pinned by:** the `Coalesce` suite (`test/codegen/late/coalesce`): `phase ›
  one float past a boundary, eight loads are of one, two, four and one`, `›
  three floats past a boundary, a load of four starts at element 1`, `› stores
  start where loads would`, `› two halves past a boundary, a load of four
  starts at element 2`, `› floats known to start on 4 bytes alone are not
  merged`, `› floats known to start on 8 bytes merge in twos`, `› halves three
  past 8 bytes merge no wider than 8 bytes, from element 1` and `› on the CPU
  device, floats past a boundary merge as from one`, and the laws `every
  vector access is aligned to its width and within the alignment, for every
  phase and alignment` and `coalescing preserves the kernel's writes, for
  every phase and alignment`; the `Ops` suite: `param_arg takes a phase that
  is a multiple of the element size below 16`, `param_arg takes an alignment
  that is a power of two up to 16, and a phase below it`, `pp_arg writes a
  parameter's phase that is not 0 and alignment that is not 16`, `equal_arg
  tells apart parameters that differ in phase` and `param_like keeps its
  storage's phase, moved by a shrink, and knows it modulo what a symbolic
  start moves by`; the graph format's round trip of `a buffer past a 16-byte
  boundary`; the `Schedule` suite's `disk_view_to_linear.golden`, a view 8
  bytes into a disk file whose parameter has no phase. In rune, `rune jit ›
  jit › an argument off 16 bytes`, 0, 4, 8 and 12 bytes into a window; the
  `Schedule` suite's `contiguous_mops_to_view › a stage of doubles one past a
  boundary is known to start on 8 bytes`, `› a stage of rows padded apart,
  starting one double past a boundary, is storage of its own on one` and `› a
  stage's alignment and phase hold of the storage it gets, a view or its own`;
  jera's `Split`, `Ode` and `Sde` suites' compiled marches, whose rows are
  padded apart. On rune: missing for a scan and on Metal.

## D55. Metal names a vector after its element's one-word name

- **tinygrad:** `renderer/cstyle.py` `_render_dtype` (a vector after its
  element's `type_map` name, spaces made underscores) and
  `MetalRenderer.type_map` (renames only `uint` and `bfloat` of
  `CStyleLanguage.type_map`'s C names).
- **tolk:** `Cstyle.lang`'s field `vector_names`, which
  `Cstyle.render_dtype` reads, and Metal's in `Cstyle.metal_lang`.
- **Differs:** a Metal vector of `signed char`, `unsigned char`, `unsigned
  short` or `unsigned long` is a `char`, `uchar`, `ushort` or `ulong` vector;
  tinygrad writes `signed_char4`, `unsigned_char4`, `unsigned_short4` and
  `unsigned_long4`, which Metal does not have, so such a kernel does not
  compile. Scalars keep tinygrad's names. Metal's vectors have 2, 3 or 4 lanes.
- **Reason:** (b). rune's jit on Metal computes `Int8` and `Uint8` arrays
  (rune's `Lower_arith.dtype`, which also stores booleans and narrow floats as
  `Uint8`), whose upcast kernels hold `char4` and `uchar4` values.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `sources › by
  default › metal_vector_<element>_<lanes>`, for each element and 2, 3 and 4
  lanes, and `metal_vector_cast_char`; the slow `every GPU kernel compiles
  with its target's toolchain › metal_vector_*`, with MTLCompiler. On rune:
  missing.

## D56. The logarithm of a negative subnormal is NaN

- **tinygrad:** `codegen/decomp/transcendental.py` `xlog2` (selects NaN where
  `d < -0.0`, then `-inf` where the reciprocal of `d` is `-inf`, meant for
  `-0.0`).
- **tolk:** `Transcendental.xlog2`.
- **Differs:** the `-inf` select comes first and the NaN select after it, so
  `-0.0` gives `-inf` and every negative number NaN. In tinygrad's order a
  negative number whose reciprocal overflows to `-inf`, of magnitude below
  `2^-128` in float32 (`2^-1024` in float64, about `2^-16` in float16), gives
  `-inf`.
- **Reason:** (e). The logarithm of a negative number is NaN, where
  tinygrad's order gives `-inf`. rune computes `Nx.log2` with nx's sequence
  (rune's `Lower_real.log2`) and reads tolk's `Log2` only for attention's
  `lse`, the logarithm of a positive sum (rune's `Lower_attention.lse_of`),
  so no raven call site reaches the rule.
- **Pinned by:** the `Transcendental` suite
  (`test/codegen/decomp/transcendental`): `special values › xlog2 of a
  negative number whose reciprocal overflows is NaN, and of -0. is -inf`, and
  the negative float16 rows of `values.tolk` of magnitude below `2^-16`.

## D58. Withdrawn

`Spec.program` rejected an elementwise operation on a vector, which no
consumer needed; it is `spec_program` (`uop/spec.py` `spec_program`) on every
renderer.

## D60. A loop of a call stays in the schedule

- **tinygrad:** `schedule/rangeify.py` `pm_no_views` (strips every view of
  storage), `split_store` (makes a kernel of an end of a call) and
  `pm_renumber_range` (renumbers that kernel's ranges);
  `schedule/__init__.py` `create_schedule` (schedules the call of an end and
  drops the end); `uop/spec.py` `spec_kernel_graph` (device ranges only, no
  end).
- **tolk:** `Prepare.pm_loop_calls`, `Prepare.check_loop`,
  `Rangeify.loop_range`, `Rangeify.pm_no_views` (which keeps a view that moves
  with a loop's range, and a run of storage, D350) and
  `Rangeify.split_store`; `Schedule.create_schedule` (`loops`, `argument` and
  the kept end); `Spec.loop`, `Spec.loop_bound` and `Spec.kernel_graph`.
- **Differs:** an end of a precompiled call over loop ranges
  (`Axis_type.Loop`), `AFTER(buf, END(CALL(fn, args), r))`, is an effect of
  the kernel graph and no kernel.
  - `Prepare.pm_loop_calls` makes every loop of a call one: an inline call an
    end or a back edge runs is compiled on its own, where tinygrad's
    `resolve_function` inlines it into stores its `split_store` makes one
    kernel of, each run once over all trips; and an end of an end of a call is
    one end over the outer ranges, then the inner, where tinygrad's
    `create_schedule` asserts an end's source is a call.
  - Any other loop outside a kernel is refused when `Prepare.sequence` meets it,
    naming what it runs: an end of a call over a range of another kind, an end
    around a back edge, a group of effects or a linear of calls, and a back edge
    around anything but a call; tinygrad asserts in `create_schedule`, or fails
    its spec, or drops the loop.
  - The calls a loop runs, or an inner loop, are the body of a call it runs. The
    arguments that move with `r` stay views, `r` keeps its number, since
    `split_store` renumbers only the ranges of the kernel it makes, and
    `create_schedule` keeps the end around the call, ordered by the call's reads
    and writes. Once the body is scheduled and resolved, the schedule is `LINEAR
    [END(LINEAR [CALL k1; CALL k2; ...], r)]`, which the memory plan of a range
    (D41) and `Tolk_engine.run`, a loop of rig's program (D146), run.
  - The kernel graph spec admits loop ranges, ends of calls over them, and
    views that move with them, whose bounds are constants, loop ranges, and
    weak integer sums and products of them, which a call may also pass as a
    scalar argument (D140); a range of any other kind outside a kernel is
    still refused.
- **Reason:** (b). rune's `Rune.scan` compiles its body once and runs it once
  per trip, in place over its carry and over rows of its inputs and outputs, as
  such a loop. The inline call and the end of an end keep that loop's meaning
  for every call a loop runs, however it is built: inlined as tinygrad would,
  the stores run once over all trips and a trip no longer reads what the trips
  before it wrote.
- **Pinned by:** the `Rangeify` suite (`test/schedule/rangeify`):
  `get_kernel_graph › loops of calls › a loop of a precompiled call is no
  kernel` and `› a loop's call reads views that move with its range`; the
  `Schedule` suite (`test/schedule/schedule`): `create_schedule › rules › an
  end of a call over a loop schedules the call in its loop`, `› a loop runs
  after the call that writes what it reads`, `› a loop runs before the call
  that overwrites what it reads`, and `create_linear_with_vars › loops of
  calls › a loop of a precompiled call is a loop of its body's calls`, `› a
  loop inside a loop's body keeps its own end`, and `› an end of a back edge
  is refused, naming what it runs` with the seven refusals beside it, each at
  the top level and in a precompiled and an inlined body; the `Tolk_engine`
  suite (`test/engine/tolk_engine`): `link and run › a scan runs its body once
  per trip, carrying in place` and `› a scan's merged calls carry in place as
  its calls apart do` (D404), on the host and each GPU present; `calls of
  views › nested loops of a call run the outer range slowest`, `› a loop reads
  each run as its trip starts`, `› a loop in a body reads its parameters as
  each trip starts` and `› an After state's fixed run under a back edge`, with
  the loop's call inlined, the nested loops and the loop in a body also against
  `Tensors`. The `Spec` suite (`test/uop/spec`): `kernel_graph › loops of
  calls › accepts a loop of a call over a loop range`, `› refuses a loop of a
  call over a range of any other kind` and `› refuses a weak sum of a weak
  integer variable`; its `verdicts.tolk`.

## D61. Withdrawn

A stack took the per-shard sub-view of a whole source beside sharded ones.
Withdrawn: multi-device is out of scope, see README.

## D62. Withdrawn

A move of an emulated float kept its bits; merged into D27, which holds its
rule, its reason and its pins.

## D63. Programs and schedules are kept on disk

- **tinygrad:** `codegen/__init__.py` `to_program_config`, `to_program_key`
  and `to_program_cache` (a dictionary of the process); `schedule/__init__.py`
  `lower_sink_to_linear`'s schedule cache (on disk only from `SCACHE=2`);
  `helpers.py` `SCACHE` (`1` by default); `codegen/opt/postrange.py`
  `Scheduler.get_optimized_ast` (a kernel's name, coloured).
- **tolk:** `Codegen.program_key`, `Codegen.kept` and
  `Codegen.made_program`; `Schedule.in_order`, `Schedule.schedule_key` and
  `Schedule.lower_sink_to_linear`; `Graph.cached`;
  `Postrange.Scheduler.get_optimized_ast`'s name; `Setting.scache` (`2` by
  default), `Setting.shaping` and `Setting.cc`; and `lib/dune`'s rule for
  `source_digest.ml`, written by `tools/source_digest.ml`.
- **Differs:**
  - The program `to_program` makes of a kernel, up to its source, and by
    default the schedule `lower_sink_to_linear` makes of a function, are put
    in the disk cache (tables `to_program` and `schedule_cache`) as their
    graphs' text (`Graph`), and a later process reads them back. A program
    read back takes its binary from its compiler's table, under its source.
    `SCACHE=1` keeps schedules in memory only, as tinygrad's default does.
  - Each key is the kernel or function, a function with its ranges and its
    storage replaced by placeholders numbered in order (the schedule is kept
    with the same placeholders and given back each function's own); for
    programs, the renderer and its target; the value of every setting and
    environment variable that shapes what compilation makes; and the digest of
    the library's sources, which dune computes when it builds the library.
  - Every variable is a setting declared with its reach (D114), among them
    those tinygrad reads with `getenv` where it compiles (`BEAM_ESTIMATE`,
    `BEAM_LOCAL_MAX`, `BEAM_MIN_PROGRESS`, `BEAM_PADTO`, `BEAM_UOPS_MAX`,
    `BEAM_UPCAST_MAX`, `DMC`, `EXPAND_SSA`, `HCQ_NUM_SDMA`, `JITBEAM`, and the
    `MV` and `REDUCEOP_SPLIT` variables), and the keys take every one that
    reaches output (`Setting.shaping`), one declared later by tolk or its
    caller included; a setting of the process alone, such as `DEBUG`,
    `PARALLEL` or `CC`, is not. One key serves memory and disk, so a setting
    changed by `context` makes the result again, where tinygrad's in-memory
    keys leave out `TUPLE_ORDER` and, for schedules, every setting. The compile
    cache's table names the compiler `CC` runs, where tinygrad's names Clang
    alone.
  - An entry that does not read as a program or a schedule is a miss, and is
    replaced. The program of a kernel that asks for a beam search is not kept.
    A kernel's name holds no colour, where tinygrad colours it unless
    `NO_COLOR` is set. A program read back shows its source and instructions
    at `DEBUG` 4 and 7, as a compiled one does.
  - `lower_and_compile` reads the programs the disk keeps on the calling
    domain (`Codegen.read`) and hands only the others to `Worker.map`, where a
    tinygrad worker reads its compile cache. A compiled schedule is kept whole
    too (D460).
- **Reason:** (d): a later process's first compiled call of a model. On fudge
  at `PARALLEL=1`, rune's compile bench row `warm/model/schools-start`, a fresh
  process's first call over a disk cache an earlier one filled, takes 101 ms
  against 263 ms with binaries alone kept. Its other fudge rows: rune's
  `warm/linalg/eigh-8x8` and `WarmStart/chain-100`, and the tolk bench's
  `*/warm` rows. Reading on the calling domain: a read took 0.4 to 0.5 ms on
  fudge, less than a domain's start.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `programs are
  kept › a program is made again under another value of ›` each setting,
  `programs are kept on disk › a program made by one process is read back by
  the next`, `› read reads back a program and its binary, and is none
  without`, `› a program made under one setting is not read back under
  another ›` each setting, `› a damaged entry is made anew, and replaced ›
  truncated` and `› holding no program`, `› an entry of another build of the
  library is not read back`, and `› processes making one program at once all
  get it`; the `Schedule` suite (`test/schedule/schedule`):
  `create_linear_with_vars › cache › a body scheduled under one setting misses
  under another ›` each setting, a setting its caller declares among them,
  `create_linear_with_vars › schedules are kept on disk ›` the same as
  programs' for schedules, `› with SCACHE at 1, nothing is kept on disk`,
  `create_linear_with_vars › loops of calls › a loop whose range has another
  number is the same body` and `› a body that hits under other numbers keeps
  its own loops`, and `› cache › storage numbered in another counter state is
  the same body`; the `Helpers` suite's `shaping ›` tests; the `Compiler_cpu`
  suite's `without Clang › an object cached by the default compiler is not
  served under CC`; the `Codegen` suite's `programs are kept › read waits for
  no program being made`, and the `Realize` suite's `lower_and_compile in
  parallel › reads the programs the disk keeps without spawning a domain`.

## D64. A cast to a narrow float through a float32 rounds once

- **tinygrad:** `renderer/cstyle.py` `create_non_native_float_pats` (casts a
  source of any type but float32 to a float32, then to the narrow float).
- **tolk:** the cast rule of `Cstyle.create_non_native_float_pats`, with
  `Decomp_dtype.narrow` (D9's).
- **Differs:** a renderer without arithmetic on a narrow float, Clang's on
  bfloat16 and HIP's on bfloat16 and the 8-bit floats, casts to it through a
  float32. A source more precise than a float32 (a float64, or an integer of
  32 or 64 bits) reaches the float32 rounded to odd, as an emulated cast's
  does (D9), so the cast rounds once, from the exact value, as the folder
  (`Dtype.truncate`) does. tinygrad rounds to the float32 to nearest, then
  again. No golden holds such a cast.
- **Reason:** (e). A cast is IEEE's conversion, one rounding: on the host,
  tinygrad's bfloat16 of the double `1 + 2^-8 + 2^-52` is `1.0` where it is
  `1.0078125`. rune hands tolk only a float32 to cast to a narrow float,
  rounding a more precise source to odd first with `Decomp_dtype.narrow`
  (`Lower_arith.store_real`), so no raven call site reaches the rule.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `casts to
  bfloat16 › an integer or a double rounds once on the host`, for int64,
  int32, uint32, uint64 and float64 sources, among them doubles just past a
  tie, one of them subnormal. On rune: `Lower_arith › casts give eager's
  values`, whose float64 operands include the doubles one unit each side of
  the bfloat16 tie `1 + 2^-8`.

## D65. Withdrawn

An operation on an emulated narrow float rounded its result in the kernel. rune
computes in float32 and encodes itself (`Lower_arith.narrowed`), and
`Decomp_dtype.pm_float_decomp` follows `codegen/decomp/dtype.py`
`pm_float_decomp`.

## D66. Withdrawn

A counted run's log entry was its kernel descriptor's address and its times,
for the counters an AMD batch read. tolk encodes no counter yet (D146).

## D67. Withdrawn

A long range ran as chunks of its trips, so that a batch held a bounded number
of commands. A range is a loop of rig's program, which holds one trip's work
(D146).

## D68. Withdrawn

The scheduler inlined a one-hot sum, rune's old gather, as a load, which left
buffers with two definitions. An explicit gather (D73) replaces it.

## D69. Withdrawn

A store through a pad wrote only within the pad's source. rune writes dropped
rows through a gather whose index is Invalid, which tinygrad schedules as a
gated store, and nothing stores through a pad.

## D70. Withdrawn

A batch reached host memory through nx.device's staging. rig's borrows and
staged copies replace it (D146).

## D71. Withdrawn

A launch stated a constant bank's size in 16-byte units. rig's NV driver writes
its launch descriptors (D146).

## D72. Withdrawn

Only a view of storage had a contiguous view. `Prepare.contiguous_view` finds a
view of any value, as `uop/ops.py` `UOp.contiguous_view` does.

## D73. A gather is an INDEX of a tensor by a value with axes

- **tinygrad:** `uop/ops.py` `UOp._shape` (INDEX's shape: its index sources'
  shapes, then the rest of the source's), which no tinygrad pass puts in a
  tensor graph; `schedule/indexing.py` `pm_generate_realize_map` and its
  rules, `data_srcs`, `pm_apply_rangeify` and `run_rangeify`'s consumer
  ranges; `schedule/prepare.py` `_mop_index`, `pm_mops` and
  `prepare_rangeify`.
- **tolk:** `Indexing.storage`, `Indexing.is_gather`,
  `Indexing.realize_gathered`, `Indexing.data_srcs`, `Indexing.convert_gather`
  and its rule in `Indexing.pm_apply_rangeify`, and
  `Indexing.assign_ranges`'s consumer ranges of a gather;
  `Prepare.move_index`, `Prepare.mops` and `Prepare.pm_tensor_mops`;
  `Codegen.index_lane`.
- **Differs:** an `INDEX` whose one index source has axes, `INDEX(x, L)`,
  reads `x` at the row each element of `L` holds, and its shape is `L`'s
  followed by the rest of `x`'s. The scheduler takes it as a load:
  - `x` is stored whole unless it is storage, so a view of storage is read
    through its movements;
  - `L` takes the gather's leading ranges, one per axis of `L`, by its
    position among the gather's sources, and the gather reads `x` at `L`'s
    element followed by its trailing ranges;
  - an index that loads from a storage state moves through a movement with
    each load held as a parameter of its bounds, so the index's
    simplification does not rebuild the stores the state is ordered after;
  - in the tensor graph, prepare's movement rules leave the gather whole;
    `pm_mops`, which kernels use with vector indices, is tinygrad's;
  - refused: a gather by several indices.
  The index is in range: rune clamps it and selects `0` where it was not.
- **Reason:** (b): rune's gather (`Lower_bits.take`, which `Lower_index.gather`
  calls) emits this INDEX for `Nx.take` and `Nx.take_along_axis`; D68 is
  withdrawn. The sharded clauses are withdrawn: multi-device is out of scope,
  see README.
- **Pinned by:** the Rangeify suite's `gathers` programs (`index_rows`,
  `index_rows_computed`, `index_read_twice`, `index_under_reduce`,
  `index_broadcast`, `index_view_source`, `index_zip`, `index_of_index`,
  `index_zero_fill`, `index_assign_self`) and their kernel counts,
  `gather_kernel_counts`; the Prepare suite's `gather_of_self`,
  `gather_by_self` and `gather_of_reshape`, and `pm_mops › rules › an index
  that loads from a storage state moves with the state as it is`;
  `Tolk.Codegen › index_lane › a window's lane is the movement rules' index,
  with its state` (both cases); rune's
  `Lower_index` suite: `gathers and scatters › gathers give eager's bits`,
  compiled on the host.

## D74. Float arithmetic has bounds, and a bounded sine takes the short reduction

- **tinygrad:** `uop/ops.py` `UOp._min_max` (bounds binary operations on
  integers only, has no case for `TRUNC` or `NEG`, bounds a `WHERE` by both
  its branches whatever selects them, and gives a value or cast it cannot
  bound its type's `min` and `max`, finite in the 8-bit formats without
  infinities, `dtype.py` `DType.max`); `codegen/decomp/transcendental.py`
  `xsin` (builds the Payne-Hanek reduction unless its caller passes `fast`,
  choosing at run time with `x_abs < switch_over`, which `symbolic_simple`
  does not fold).
- **tolk:** `Ops.unbounded`, `Ops.compute_min_max` (its `Where`, `Trunc`
  and `Neg` cases), `Ops.selected`, `Ops.float_bounds` and
  `Ops.cast_bounds`; `Transcendental.xsin`.
- **Differs:**
  - A float sum, difference or product of operands with finite bounds has the
    bounds of its corners, computed in double and widened by a relative
    `2^-m` and the smallest normal of its type, `m` its mantissa's bits, so
    they hold the result rounded at its type, or flushed to zero; an operand's
    end within the subnormals counts as 0, for a target that flushes it. A
    result that may overflow its
    type, and one of a weak float, has the type's bounds.
  - A truncation's bounds are its operand's, truncated, and a float
    negation's are its operand's, negated and swapped.
  - A float selection by `a < b` narrows the branch it selects where the
    comparison holds, so where neither operand is NaN: `a` to below `b`'s
    greatest value, and `b` to above `a`'s least.
  - Finite float bounds state that a value is not NaN: a float that nothing
    bounds has infinite bounds, in `float8_e4m3` and the `fnuz` formats too,
    and a cast into a float keeps its source's rounded bounds only where they
    are values of its type, so a value cast from an unbounded float stays
    unbounded. A cast into a weak float rounds its bounds to doubles, as
    tinygrad does since d94ec221b, whose cast of an infinite bound into an
    8-bit format without infinities is NaN and falls back to the type's finite
    bounds, where tolk's stays unbounded. Neither bounds a float comparison to
    `true`.
  - `xsin` of an angle whose bounds lie strictly within `switch_over` is its
    `fast` form, the Cody-Waite reduction alone.
- **Reason:** (d): the sine or cosine of a bounded value on the CPU, such as a
  Gaussian draw (`Nx.Rng.normal`), whose angle is `2 pi u` for a uniform `u` in
  `[0, 1)` fused into the draw; rune's reduction (`Lower_real.reduce`) reads
  tolk's bounds (`Ops.vmin`, `Ops.vmax`), builds Cody-Waite's alone below
  `Lower_real.trig_switch` and holds the remainder to `[-1, 1]`
  (`Lower_real.reduced`). The host's renderer has no sine, so `xsin` reduces
  what remains. On fudge rune's `Jit/normal-float32-1M` takes 2.8 ms against 4.3
  ms without the entry; the compile bench rows `rng/normal-f32` and
  `rng/normal-f64` time the first calls of a draw of 1024. The 8-bit half (a
  float without infinities is unbounded) has no rune consumer, since rune stores
  the 8-bit floats as `uint8` (D62); with the type's finite greatest value, a
  compiled sine, cosine or tangent of an 8-bit NaN gave -1, 0 and -452.
- **Pinned by:** the Ops suite (`test/uop/ops`): `bounds › bounds hold every
  value a float operation rounds to`, sums, differences, products and
  selections, `› a float operation of bounded operands is bounded`, `› a float
  operation of an operand within the subnormals bounds a flushed operand too`,
  `› a float operation of an unbounded operand, or that can overflow, has its
  type's bounds`, `› a float selection by a comparison narrows what it selects`,
  `› a float truncation and negation map their operand's bounds`, `› a value of
  a float type without infinities, which may be NaN, has no finite bounds` and
  `› bounds hold every value a cast takes`; the Transcendental suite: `graphs ›
  a sine of an angle bounded below the switch-over is its fast form`; rune's
  `Lower_real` suite: `bounded arguments › cos of an argument bounded below the
  switch-over` and `› tan of an argument bounded below the switch-over`. On
  rune: missing for the count of long reductions a normal draw builds, and the
  sine and cosine of an angle read from a buffer build, and for the 8-bit NaN's
  sine, cosine and tangent.

## D75. A constant condition assumes nothing in its branches

- **tinygrad:** `uop/symbolic.py` `fold_where_closure` (substitutes `True`
  for a selection's condition in its true branch and `False` in its false
  branch, whatever the condition is).
- **tolk:** `Shape.fold_where_closure`.
- **Differs:** `fold_where_closure` leaves a condition that is a constant,
  broadcast, padded or not, in the branches: a broadcast constant is one node
  for every use of that constant, so substituting it is no assumption about
  the selection. A selection by a `CONST` still folds, as in tinygrad.
- **Reason:** (e): `where F (where F x (where F y x)) y`, with `F` a broadcast
  `False`, is valid by construction, and tinygrad's rewrite raises "infinite
  loop in graph_rewrite" on it, and tolk's without the rule "graph_rewrite does
  not terminate". The fold of a selection whose condition is a constant through
  movements, this entry's other part, is withdrawn.
- **Pinned by:** the Symbolic suite: `symbolic › selections › a broadcast
  constant condition is not folded in the branches`, which raises without
  the rule, and `› a padded constant condition is not folded in the
  branches`.

## D76. Withdrawn

A stack of `Invalid` lanes stayed a stack. tinygrad folds it to an `Invalid`
expanded to its shape (`uop/symbolic.py` `Shape.pm_data_invalid`), which keeps
the width, and devectorizes no expanded scalar into a stack; with it
`Tolk.Codegen › lanes all Invalid` and `lanes of an unrolled reduce` sum to 0.

## D77. A broadcast value that runs a transcendental stays stored

- **tinygrad:** `schedule/indexing.py` `run_rangeify` (stores an elementwise
  value or reduction whose ranges a broadcast ends),
  `BufferizeOpts.removable` and `create_bufferize_and_index_srcs` (its value);
  `schedule/rangeify.py` `cleanup_dead_axes` and `remove_bufferize` (inlines
  that store again when its value reads at most three buffers and no
  reduction reads a buffer).
- **tolk:** `Indexing.assign_ranges` (which records the values a broadcast
  stores, and a gather's stored source that its gathers would compute more
  of inlined) and `Indexing.bufferize_and_index` (which marks
  their stages `Broadcast`); `Ops.keep`, in place of `removable`;
  `Rangeify.remove_bufferize`, which keeps such a stage, with `Rangeify.work`,
  `Rangeify.transcendental_work` and `Rangeify.read_selected`.
- **Differs:** where a broadcast ends ranges below a value, its store is
  marked, and the cost check keeps it if computing it runs `EXP2`, `LOG2`,
  `SIN` or `POW`, or at least the work of tolk's float32 exponential
  (`Transcendental.xexp2`, simplified, a multiply-add counting as two) in
  elementwise operations inside the value's ranges, as rune's restatement of
  nx's exponential, sine and logarithm spells them, except at a read in a sum
  that selects one trip, where D163 counts its copies. The cost check sees
  the value with its producers already stored or inlined, so a value that
  reads a stored exponential computes none, and is inlined. A marked store still
  loses the axes its value does not vary along. A value read once per
  element, as a rotated query reads its rows of a stored sine table, is not
  marked. A gather's source (D73), stored before
  ranges are assigned, is marked when its consumers broadcast it, or when its
  gathers, inlined, would compute more elements than it holds: each element a
  gather reads, once for each trip of the ranges that end above it, that work
  at its least and the elements held at their most. `Indexing.assign_ranges`
  carries the ranges that end above each node, as tinygrad's `ending_ranges`
  does. tinygrad's cost check counts
  buffers and reductions only, so its consumer computes the value again for
  every element of the ranges it does not vary along.
- **Reason:** (d): gpt-oss-20b (`apps/gpt-oss`), where inlining
  computes a SwiGLU's sigmoid once per output column of the down projection and
  a RoPE's sines and cosines once per head. Since rune lowers `Nx.attention` to
  kernels of its own, gpt-oss's attention no longer meets this rule; the rest of
  its layers still do. On fudge's RTX 5000 Ada `GptOss/prefill-512` takes 132 ms
  against 265 ms without the rule. The prefill rows' fudge wall-time baselines
  are pending the scheduled quiet re-record. The cost is memory: a kept value
  holds a buffer while the call runs. Once rune compiled its exponentials to
  nx's sequence, no rune graph ran the four operations, and gpt-oss's decode
  computed its MoE activation in each of the down projection's 720
  workgroups, 140 us a layer against a 33 us byte bound; its step at cache
  1024 on real weights took 10.84 ms against 8.76 ms at 726e08c6cb (one
  round each; with the work test, pending a quiet GPU). The work test is
  the cost of one exponential, the cost this rule already keeps a value for,
  and like the operation test it ignores the broadcast's extent. A read in a
  selecting sum, as of `selected_scaled`'s 160 nodes, is left to D163,
  which counts its copies at the selected index and inlines it as tinygrad
  does, unless an enclosing sum reads it again at outputs (D106). A
  broadcast value of that much plain arithmetic read elsewhere is
  stored too, which no bench row runs. A table its gathers read again is
  the same case: on fudge's host a table of 64 rows read at 4096 takes
  0.8-1.2 ms stored against 2.8-3.2 ms inlined.
- **Pinned by:** the Rangeify suite's `swiglu_down`, `attention`,
  `exp_dead_axis` (a kept exponential of a row expanded to a matrix is stored
  as the row), `exp_cheap_consumer` (a value that reads a kept exponential is
  inlined) and `rope_decode` (a decode step's rotated query reads its gathered
  rows of a cosine table and is not stored) rows of `kernel_counts.golden` and
  their kernel goldens, `selected_scaled` (a heavy value a selecting sum
  reads, which this rule leaves to D163; D106 stores it) of
  `kernel_counts.tolk`, and its `get_kernel_graph › gathers read where
  broadcast`, `› gathers that read more than their table
  holds`, `› broadcast exponentials` and `› a spelled exponential selected
  and broadcast`; rune's `jit ›
  an exponential every output of a product reads is stored` and `› a routed
  product's sigmoid is stored` (`Nx.exp`,
  `Nx.sigmoid`, `Nx.sin` and `Nx.log` as rune compiles them, at every float
  dtype); kaun's `GptOss/prefill-512` and `GptOss/prefill-512-Q8_0`.

## D79. A chain of one operation keeps its right operand's grouping

- **tinygrad:** `renderer/cstyle.py` `base_rewrite`'s ALU rule (drops the
  parentheses of any operand that is the same associative operation, `ADD`,
  `MUL`, `XOR`, `OR`, `AND`, on either side, but the right operand of a signed
  integer operation since 845977979).
- **tolk:** `Cstyle.base_rewrite`'s ALU rule.
- **Differs:** only the first operand drops them, of floats and unsigned
  integers too. C groups `a*b*c` as `(a*b)*c`, so `a*(b*c)` written without
  parentheses is regrouped, and a float sum or product rounds and overflows by
  its grouping: `4 * (2^126 * 0.25)` written `4*2^126*0.25` overflows.
- **Reason:** (b): nx.quant's product, `Nx_quant.apply`, decodes each weight
  in the kernel that multiplies it, `x * (value * scale)`, and gave infinity
  where the eager product's infinities of both signs give NaN.
- **Pinned by:** the Cstyle suite: `grouping › Clang computes a product of a
  product as it is grouped`; and the rendered sources of the `codegen`,
  `renderer/cstyle` and `engine/jit` goldens. On rune: missing.

## D78. Withdrawn

The CUDA device's kernels compiled to a cubin, which measured no gain over PTX.
As tinygrad's `renderer/cstyle.py` `CUDARenderer.__init__`, `Cstyle.cuda`
compiles PTX for the CUDA device and a cubin for the NV device. D320 compiles a
cubin again, measured with the driver's cache missing.

## D80. A host program runs its output loop in blocks on the host's cores

- **tinygrad:** `runtime/ops_cpu.py` `CPUProgram.__call__` (calls a kernel
  once per launch on the calling thread); `codegen/__init__.py`
  `full_rewrite_to_sink` (the pipeline after `pm_cast_const`),
  `do_number_param` (and its pass) and `do_to_program`'s estimates;
  `uop/ops.py` `KernelInfo` and `ProgramInfo.from_sink`; `engine/realize.py`
  `get_call_kernel_args` and `exec_kernel`.
- **tolk:** `Codegen.split_blocks`, its place in
  `Codegen.full_rewrite_to_sink`, which a search's candidates take (D489),
  `Postrange.misses`,
  `Codegen.pm_number_params` and `Codegen.whole_loop`; `Ops.kernel_info`'s
  `split`; `Shape.program_info_of_sink`; `Realize.get_call_kernel_args`;
  `Tolk_engine.compute_work`, `Tolk_engine.memory_work`,
  `Tolk_engine.memory_blocks`, `Tolk_engine.blocks` and
  `Tolk_engine.split_grid`.
- **Differs:**
  - For a CPU target, once the kernel is final (after `pm_cast_const`), the
    largest range of axis type `WEAK` whose end reads no range, of at least
    two iterations, and on which every store to global memory, by its
    address, and every call and piece of custom code, by its sources, depend,
    runs `block_lo + r` for `r` below `block_hi - block_lo`.
  - `block_lo` and `block_hi` are new variables of the range's type, bounded
    by `[0, n]` for a loop of `n` iterations, in the first two slots past
    those the kernel's parameters took, a loop's trip among them (D140), and
    named afresh where the kernel has a variable of either name. The variables
    numbered after them take the slots past every slot a parameter takes, the
    bounds' included, one slot per name.
  - The kernel's `KernelInfo` records `split = (n, lo, hi)`, the iterations
    and the two slots, by which each launch finds the bounds; `n` is its
    program's first global size, and its estimates count the whole loop,
    exactly when they read no other variable.
  - A host launch cuts `[0, n)` into `min(n, ops / 2^16)` blocks where
    `ops > mem`, and `min(n, max(ops / 2^18, min(8, mem / 2^18)))` otherwise,
    at least one, `ops` and `mem` the estimates' operations and bytes of
    memory. Each block is a group of rig's host of as many threads as
    iterations, which rig's pool runs on the performance cores, a thread for
    each block up to their count; the block count does not read the host's
    cores. The entry
    (D28) runs the iterations its threads number. tinygrad's own launch
    (`whole_loop` in `engine/realize.py`) runs one block, the whole loop.
- **Reason:** (d): sofo's and symo's training steps on the CPU (outside the
  repository), which ran each kernel on one core. On fudge the tolk bench's
  `lorenz/r_128_16_3_4_4_3_3_400_400/heuristic` takes 0.79 ms against 5.85 ms in
  one block. A block costs its threads' start and join, 1.3-1.5 us on all 14 of
  fudge's cores (rig's `launch/empty-all-cores`); 2^16 operations a block is the
  smallest at which no row of the rune or tolk bench lost. Bytes keep 2^18,
  since a block for each 2^17 or 2^16 bytes made a row slower, and ask at most 8
  blocks, since a memory-bound kernel gains to about 8 and on a loaded host each
  further thread can be held up. tinygrad deleted its CPU threading
  (`OptOps.THREAD`) at 4197f7423, before the pin.
- **Pinned by:** the Codegen suite: `host programs in blocks › a host program
  splits the loop each store's address reads`, `› a split loop's estimates
  count its whole loop past 2^31`, `› stores of separate loops split none`, `›
  a store of no loop splits none`, `› a reduction's loop is never split`, `› a
  serial loop is never split`, `› a loop shrunk by its guard splits at its
  shrunk end, unguarded`, `› a split loop of 2^25 iterations keeps 32-bit
  indices` and `› a split's bounds and a variable take slots of their own past
  buffers that skip one`; `calls and custom code beside host blocks › a loop
  beside one that does not depend on it does not split`, `› one beside a loop
  runs once`, `› a loop that one depends on splits, beside no store` and `› a
  loop that code depends on splits`; the Tolk_engine suite: `host programs in
  blocks › a launch of more operations than bytes runs a block per 2^16
  operations, another per 2^18 operations or bytes, bytes up to 8`, `› a
  program writes the same bits in any blocks`, `› a kernel that moves many
  bytes for few operations writes its values` and `› a run of a large kernel
  computes its sums`, `a variable named as a block's bounds is a variable like
  any other`, and `a kernel whose estimates fit no int › splits and runs`.

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
constant. No consumer depended on it; `Spec.program` accepts it, as
`shared_spec` (`uop/spec.py` `spec_shared`) does.

## D87. Withdrawn

A store through a gather of a sharded value stored each shard's rows.
Withdrawn: multi-device is out of scope, see README.

## D88. A float zero a host comparison reads reaches C as a value the backend cannot see

- **tinygrad:** `renderer/cstyle.py` (`ClangRenderer`, whose
  `string_rewrite` writes a float zero as a literal, `0.0f`).
- **tolk:** `Cstyle.float_zero`, `Cstyle.opaque`, `Cstyle.opaque_zeros`,
  `Cstyle.pm_converted_selects` and `Cstyle.clang` (which adds them for
  each host target with its types).
- **Differs:** a float zero constant that a comparison reads renders in the
  comparison as an empty `asm` statement over a register holding it,
  `({float z = 0.0f; __asm__("" : "+w"(z)); z;})`, in a float for a narrow
  type and cast back to it: on arm64 at float16, bfloat16, float32 and
  float64, on riscv64 at float16, float32 and float64, and on x86_64 at
  float16. On x86_64 a scalar selection between a zero and a conversion of
  its compared value selects at that value's type and converts after, and
  one between a zero and the float16 or float32 its compared value widens
  compares that float16 or float32. The register class is the target's for
  a float: `w` on arm64, `x` on x86_64, `f` on riscv64. Every other use of
  the zero, and every other zero, keeps the literal.
- **Reason:** (b): backends lower a select of a value and a zero, by a
  comparison of the two, to a minimum or maximum that may keep the value's
  zero (`fminnm`/`fmaxnm` on AArch64, `fmin`/`fmax` on RISC-V, `fminf`/`fmaxf`
  at x86 float16); at float32 and float64 the x86 backend's `minss` and
  `maxss` pick as the select does. InstCombine rebuilds a select whose arm
  converts its compared value to pick the zero at a tie. Selects of the
  compared value itself, and of vectors, keep their order. Under `Rune.jit`,
  `Nx.where (Nx.less x z) x z` gave `-0.` on arm64 where eager gives `0.`,
  and on x86_64 `Nx.where (Nx.less z x) z x` gave `0.` at a float16 `-0.`.
  Converting after the selection, or comparing before the widening, keeps
  every value. Hidden in every comparison at x86_64 float32, the zero cost a
  maximum before a 64 × 64 matmul 1.10 times its time (D166); hidden
  everywhere, it kept a routed MXFP4 product's select from becoming a mask,
  1.17 times the time without the select against 1.05 with the literal.
- **Open:** the GPU renderers keep the literal, and NVRTC 13.4 lowers `(v <
  0.0f) ? v : 0.0f` to `min.f32` for sm_89: under `Rune.jit` on the RTX 5000,
  the selection gives `-0.` at `-0.`. NVRTC has no statement expressions, so
  CUDA needs a helper function around the `asm`; Metal has no inline `asm`.
  HIP and Metal are untested, as is a GPU's selection of a float16 against
  a zero, which InstCombine may make pick `+0.` at `-0.`.
- **Pinned by:** the Cstyle suite (`test/renderer/cstyle`): `zeros a
  comparison reads › a float zero a comparison reads is opaque to a backend
  that lowers it to an extreme, and a literal elsewhere › *`, `› a select
  of a value and a zero picks the zero at -0., on the host › *` and `› a
  select of a value and a zero, compared and selected at two types, picks as
  the select does, on the host › *`; rune's `rune jit` suite: `classes › a
  selection of a minimum against its constant is exact`, a selection on
  `minimum c x < c` at `c = 0.` over inputs holding `-0.`, in float16,
  float32 and float64, and `classes › a
  selection of a value and a zero is exact › *`.

## D89. A contraction of little reuse streams its operand of least reuse

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  matrix-vector case, which applies only when the first reduce's product has
  two loads as operands, the vector's index has the first weak reduce range
  as a term of its sum and its ranges are among the matrix's: that axis
  split into `MV_THREADS_PER_ROW` local threads, 8, the first global axis
  that 16 divides into `MV_BLOCKSIZE` local threads, 4, then
  `MV_ROWS_PER_THREAD` upcast lanes, 4, whatever the matrix's layout, size
  and reuse),
  `hand_coded_optimizations` (the tensor cores, tried first),
  `hand_coded_optimizations` (the upcasts of output axes), and
  `codegen/opt/postrange.py` (`apply_opt`, which refuses a local split of a
  reduce axis whose reduction runs inside a weak range that a reduction
  closes).
- **tolk:** `Heuristic.lanes` (`Renderer.t.simd_threads`), `Heuristic.rows`,
  `Heuristic.columns`, `Heuristic.sector_lanes`,
  `Renderer.t.local_threads_max`, `Renderer.t.stream_threads`,
  `Heuristic.in_flight`, `Heuristic.in_flight_held`, `Heuristic.stream_first`,
  `Postrange.Scheduler.upcast_max`, `Heuristic.summed`,
  `Renderer.t.per_byte_first`,
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
    row. A reduction that reads a buffer is stored (`remove_bufferize`);
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
    where none applies. Past `R = 16` the operand is read in `ceil (R / 16)`
    passes, and a core of two stages over a known count of workgroups
    (`Stage.tile`) takes the kernel first, whatever its threads: sm_89 was
    measured, the other CUDA parts of two stages, from sm_80, take the rule
    unmeasured, and AMD and Metal, of one stage, keep streaming. Past
    `R = 32` a core takes it where one applies, and the other rules where
    none does. A decoded operand takes a core past `R = 1` where one
    applies, whatever its multiply-adds a byte, and streams at any reuse
    where none does (D230);
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
    make `stream_threads` threads, halved until the workgroup's shared
    memory holds a slot for each output of each thread, which their sum
    passes through; with more outputs than `stream_threads` the kernel takes
    the other rules. An axis those threads and lanes do not divide is padded
    to a multiple of them within 6/5 of its size, the ratio D491 bounds a
    core's padded work by, its padded columns loading nothing and stored
    nowhere, where tinygrad's case takes only an axis 16 divides. `MV_BLOCKSIZE`, `MV_THREADS_PER_ROW`
    and `MV_ROWS_PER_THREAD` are gone;
  - few outputs (D110's part): when the outputs are too few for the columns
    layout to run `stream_threads` threads, a workgroup of the renderer's
    `local_threads_max` threads (D484), 1024 on CUDA, takes fewer threads
    across them, down to 8, and more along the reduce, so that more
    workgroups share the outputs; 8 threads of 2 bfloat16 columns read a
    32-byte sector. `stream_threads` is 32768 on sm_89's parts, as D175's
    floor is: at 16384 the decode step's key and value projection
    (`stream/decode-kv`) is 1.16x slower. Metal takes 16384. rune's scatter
    plan, which `stream_threads` also sizes, is untimed;
  - a group inside a reduce: a reduce axis whose reduction runs inside a
    serial reduce, a weak range that a reduction closes, splits into local
    threads, each iteration of the outer reduce storing and reading the shared
    buffer between barriers. One inside an unrolled reduce, an upcast range
    that a reduction closes, is refused, as tinygrad refuses it; the lanes of
    a tensor core's K, which the core contracts and no reduction closes, take
    a group in both;
  - the constants: `lanes`, `in_flight`, `in_flight_held`,
    `stream_threads`, `stream_first`, `summed`, `per_byte_first`, D175's
    `core_threads` and D479's `group_least` each move the order of a sum,
    `stream_threads` also through `Heuristic.group` (D479). They are fixed
    per target, never read from a device, since the program key names the
    target alone (D170). `stream_threads`, D175's `local_workgroups`,
    `core_threads` and `per_byte_first` are fields of the renderer, since a
    second reader (rune's attention lowering) or a second target's fit
    states another value. `lanes` reads the renderer's `simd_threads`: 64 on
    a CDNA (gfx9) wave, unmeasured, and 32 elsewhere. Every target keeps
    `per_byte_first` at 64 until its own crossover is placed: gfx1201's lies
    above 149, its rows pending on nonnormal. Routing does not order the two
    paths: across eight routings no kernel moves by more than 5% on AMD or
    10% on CUDA, and a path chosen by data would make a row's bits depend on
    other rows' routes. A
    gather fused into the streamed product, its rows gathered again in each
    workgroup, costs 1.3x on CUDA, which the order does not weigh: an open
    cost.
- **Reason:** (d): gpt-oss-20b (`apps/gpt-oss`) on CUDA, and
  nx.quant's products (`Nx_quant.apply`, `Nx.map_segments` over it)
  compiled by rune, whose streamed operand tinygrad's layout reads from
  memory more than it multiplies. On fudge's RTX 5000 Ada, tolk's bench
  `stream/q8_0-1` goes from 1230-1240 to 291 us. 64 multiply-adds a byte is
  the measured crossover against the cores (between 60 and 66); blocks
  sharing their matrices at 40-58 a byte stay streamed, and that band,
  1.1-1.55x slower on fudge, is the cost of one program per target. The
  constant is not measured on AMD's and Metal's cores: nonnormal's R9700
  rows (`GptOss/experts-at-prefill-512`, `GptOss/prefill-512`) are pending,
  and no machine a bench may run on holds a Metal device. Past `R = 16` the stream reads its operand in two passes or
  more, and on sm_89 a core of two stages, its plan spread over the units,
  beat it on every shape measured, short of threads too; the rule reads no
  unit count, which would move a sum's order with one. Past `R = 32` float32
  streamed loses to the tiles from `R = 64`; narrower operands take the
  other rules past 32 until a tolk row confirms them streamed. Metal's
  `stream_threads` of 16384: Reason: none. It was fitted on an M1 Max
  (Apple7), other Apple parts unmeasured, but no machine a bench may run on
  holds a Metal device, so no row can pin it. The columns' pad: on fudge,
  TESSERA's distillation step's weight gradient of its 10-band input layer
  over 23040 tokens, 440 columns that 16 does not divide, took 1.43 ms a
  launch on the other rules (330 threads of 16 outputs, 14 GB/s) and 0.14
  streamed on 448 columns; its sums of 440-column products over the tokens,
  three of their five programs now streamed, 38.7 ms a call and 31.4; the
  step's kernels 1077.6 and 1058.3 ms a call (DEBUG=2), and
  `distil-step/medium` 1.069-1.070 s and 1.052-1.053 (thumper's medians,
  three alternated rounds under TIMING). A float32 row by `[4096; 1000]`
  (`vecmat_1000`) took 54.3 us and takes 38.9, 10 rows 231 and 85, 16 rows
  by `[2880; 440]` 172 and 58. The cost is the layout's own at few columns,
  whose workgroups are a 16th of them: bfloat16 products of 8 and 16 rows
  by `[2880; 440]` took 34-50 us on the other rules and take 48-63, the
  layout's time at 448 columns.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are tinygrad's ›
  <environment> › the optimised kernel`, cases `matvec_*`, `vecmat_*`,
  `gpt_oss_*`, `experts_down_*`, `routed_blocks_*`, `routed_own_*`,
  `routed_shared_*`, `routed_at_bound_*` and `routed_past_bound_*`,
  `shared_keys_*`, `rows_m{4,12,32}_*`, `q8_0_blocks_*`, `gguf_mxfp4_*`,
  `mxfp4_m12_*`, `rows_short_*` and `routed_by_argmax_*`; `streaming › the
  streamed operand's reuse outside a thread is its reuse over its tile`, `› a
  SIMD group's load of an operand streamed along the reduce reads whole
  sectors`, `› the unroll is the largest divisor of its loop up to 8, 4 with
  reuse`, `› a contraction streams up to a reuse of 32, before a tensor core`,
  `› a contraction of several passes streams unless a two-stage core takes
  it` and `› a recorded contraction streams before a tensor core up to 64
  multiply-adds per byte of its streamed operand`, `› each renderer streams
  before a core up to 64 multiply-adds a byte` and `› a contraction streams
  before a core up to the renderer's bound a byte`, `› a product's workgroups
  hold at most a block's threads on every GPU`, `› a product streamed by
  its rows reduces over a SIMD group's threads`, `› a product streamed by
  its columns splits its reduce into threads` and `› a product streamed by
  its columns pads what its threads do not divide`, and the cases
  `vecmat_1000_{cuda,metal,amd}`; the Cstyle suite's `a
  renderer states its SIMD group's threads and a workgroup's most`; `the
  hand-coded optimisations keep a kernel's writes › small kernels ›
  experts_down_metal` (a group inside another reduce); the Postrange suite's
  cases `double_sum_group`, `double_sum_group_twice` and
  `double_sum_unrolled_group`; on fudge, tolk's bench `stream/*`
  (`stream/mxfp4-12` with its decode
  stored as float32, D163) and `gpt_oss/*` rows and kaun's `Quant/cuda/*` rows.

## D90. A linked schedule resolves its launches at link

- **tinygrad:** `engine/realize.py` (`resolve_params`) and `exec_kernel`, which
  find a call's buffers, launch dimensions and variable values on every run, and
  `uop/ops.py` (`UOp._sym_fxn`, cached on the node, which `UOp.sym_infer`
  calls on each evaluation).
- **tolk:** `Tolk_engine.affine`, `Tolk_engine.computation`,
  `Tolk_engine.value`, `Tolk_engine.locate` and `Tolk_engine.launch`;
  `Shape.sym_compile`, and
  `Shape.sym_infer`, which is `Shape.sym_compile` applied once.
- **Differs:** linking a schedule turns each call into a launch per lane of
  rig's program (D146), whose buffers, offsets, scalars and grid are words of
  the run's ints or constants: each variable of the run is a word, a value
  computed from them a word the run computes once (`Shape.sym_compile`, which
  simplifies once and computes on `int`s, exactly where a value does not fit),
  and a value that moves with a loop's trips (an offset, a scalar, a grid or a
  loop's trips) a value of rig's program: the sum of trip words times constant
  factors.
  - A cast to an integer type that does not hold its operand's bounds keeps
    the operand's low bits wherever it sits in a value, as tinygrad's
    conversion of an argument to its parameter's type does for the whole
    value: `Tolk_engine.computation` reads it as a variable of the cast's type, which the run computes
    truncated. A link refuses such a cast where its operand moves with a
    loop's trips.
  - A cast to `uint64` keeps an `int`'s bits (`Tolk_engine.reading`): a
    scalar of 8 bytes and another cast read them as they are; where its
    number is read, under another operation or as an offset, a loop's trips,
    a grid or a split loop's count, a link refuses a constant past an `int`
    and the run a value it computes.
  - A scalar of 1, 2 or 4 bytes takes only a value its bytes hold, signed or
    unsigned: a link refuses a constant they do not hold, and the run a value
    of 4 bytes, where that conversion keeps the low bits.
  - A run refuses a binding outside its variable's bounds, which decide
    whether a cast narrows; tinygrad checks only where a variable is bound
    (`UOp.bind`, `uop/ops.py`).
  - `Shape.sym_compile` ports `_sym_fxn` and `Shape.sym_infer` ports
    `sym_infer` (`uop/ops.py` `sym_infer`); a caller that evaluates an
    integer many times keeps the function `Shape.sym_compile` returns.
  - A `DEBUG` report of a call lists the run's variables that the schedule
    reads, where it listed every variable of the run.
- **Reason:** (c): rig's program takes words and values of them (D146), which
  a link resolves once, and refuses a scalar its width does not hold. Speed is
  not a reason: no fudge row can time tinygrad's per-run resolution through
  rig's program.
- **Pinned by:** the Tolk_engine suite (`test/engine/tolk_engine`): `link and
  run › a linked scan carries its storage across runs on each run's
  parameters`, `› a scan runs its body once per trip, carrying in place`, `› a
  schedule runs with each binding of its variables`, `› a narrowing cast's low
  bits reach the kernel as its scalar`, `› a cast to an unsigned type keeps its
  operand's low bits`, `› a narrowing cast under a quotient keeps its low
  bits`, `› a narrowing cast of a constant keeps its low bits`, `› a cast to
  uint64 past an int under another operation is refused at the run`, `› a
  cast to uint64 past an int as a loop's trips or a split loop's count is
  refused at the run`, `› a cast to uint64 past an int as an offset is
  refused at the run, which writes nothing`, `› a constant cast to uint64 past
  an int whose number is read is refused at link`, `› a cast to uint64 past
  an int as a 4-byte scalar is refused`, `› a narrowing cast in a split
  loop's count keeps its low bits`, `› a 4-byte scalar that 32 bits
  hold reaches the kernel`, `› a 4-byte scalar that 32 bits do not hold is
  refused`, `› a binding outside its variable's bounds is refused at the
  run`, `› a narrowing of a value that moves with a loop's trips is refused`,
  `› a loop's trips move with an outer loop's trip`,
  `GPUs › <kind> › a call's grid moves with its loop's trips` and `runs › runs
  of one schedule from two domains each compute their own`; the Ops suite
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
  conversions and iteration). The axis's lanes then share one decode of each
  value. An operand is decoded when its value converts integer codes into
  floats, by a cast or a bitcast, outside any access's address: integers
  that a load of integer storage gives, through integer arithmetic alone. A
  float's bits read as an integer, as ReLU of a bfloat16 selects its sign,
  are no codes, and such an operand takes no decoded rule here, in D230's
  stream or in D348's panel. A host renderer that does not know its
  registers takes at most 4. On a GPU a streamed contraction takes the same
  reuse as a term of the operand's reuse (D89), whatever the operand's type;
  one that does not stream takes no run.
- **Reason:** (d): on the host, where D163 inlines the decode into the
  product, tinygrad's 4 rows decode a value 16 times at 64 rows. On fudge's
  AVX2, one P-core, the kernels of kaun's
  `Quant/host/mxfp4-product-of-1024-{10,64}-bfloat16-tokens` take 2.90 and
  15.1 ms with 8 rows against 4.29 and 18.6 ms with 4; past the budget, 16
  rows by 4 columns took 13.7 and 54-56 ms at 64 rows and 512 tokens. Float32
  rows took 2.27 and 12.05 ms against 2.6 and 8.5 ms for the decode stored
  and the 4-row product, a gap D348's panel closes. Files of 32 registers (NEON, AVX-512) take 16 rows
  unmeasured.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › in the default environment › the optimised kernel`, cases
  `routed_*_cpu` and `routed_*_avx2`; `the host's decodes`: the
  decodes of products of 1 to 512 rows on four register files and on a host
  without registers, blocks of 16 to 128 rows on four files, and a larger
  file decoding no more times; `streaming › a product of ReLU of bfloat16
  values is not streamed`; the Stage suite's `decoded panels › a decoded
  operand converts integer codes it loads`; `the hand-coded optimisations
  keep a kernel's writes › large kernels › routed_blocks_*` (slow); on fudge,
  kaun's `Quant/host/mxfp4-product-of-1024-*` and
  `Quant/host/routed-product-64-tokens`, and rune's
  `Jit/experts-mxfp4-512x4of32-256`.

## D93. The where-closure rule leaves loads alone, and its slice queries walk no more than they need

- **tinygrad:** `uop/symbolic.py` (`fold_where_closure`, which asks
  whether the condition is in `t.bool_slice` or `f.bool_slice`, and whether
  the condition or a branch reaches an `INDEX`); `uop/ops.py`
  (`_bool_slice`, a `recursive_property`) and `backward_slice` (a
  `cached_property`, and `op_in_backward_slice_with_self`, which walks it);
  `codegen/late/coalesce.py` (`memory_coalescing`, which addresses a
  load of several elements by a `SHRINK` of the buffer).
- **tolk:** `Shape.fold_where_closure`, `Ops.reaches`, `Ops.ops_reached` and
  `Ops.op_in_backward_slice_with_self`.
- **Differs:**
  - the load gate: a selection whose condition or branch reaches a `LOAD` is
    left alone, as one that reaches an `INDEX` is. This covers a load whose
    address memory coalescing made a SHRINK, valid included, and a register's
    read, the load of a reduction's accumulator. The goldens do not change;
  - the condition: the rule asks whether `t` or `f` reaches the condition, a
    scalar boolean, with `Ops.reaches`: a walk from the branch that enters no
    node built before the condition. `Ops.bool_slice` is gone. The search
    runs last, after the INDEX and load gate, where tinygrad's lookup runs
    first: a branch that reaches an INDEX
    or a LOAD is rejected without a walk;
  - the gate's sets: each node keeps, as a property filled from its sources,
    the set of the operations of itself and of the nodes it reaches outside
    call bodies; a node whose set is a source's shares it.
    `op_in_backward_slice_with_self` reads that set and builds no
    `backward_slice`.

  But for loads, the rule folds the same selections, and the answers are
  the same.
- **Reason:** (d): compile time. Without the load gate the rule assumed a
  condition read from memory down through every link of a jitted chain of
  `Nx.betainc`, copying the links: 4 links compiled cold in 172 s, and
  compile in 2.9 s with the gate (rune's compile bench row
  `special/betainc-chain-f32`). With the gate kept, tinygrad's boolean sets
  and whole slices make that row 8.05 and 7.75 s against 2.99 and 2.70 s,
  and they grow with the square of a chain of `Nx.sin`. A condition built
  before a long chain of selections still makes each search walk the chain
  below it, which the gate spares when the branches read memory.
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
  `DEV.target`); `device.py` (`_select_renderer`); `runtime/ops_cpu.py`
  (`CPUDevice`, whose arch, remote or not, is this machine's
  `platform.machine()` and `native`).
- **tolk:** `Tolk_engine.target`; `Device.renderer`; `Helpers.Target`.
- **Differs:** there is no `DEV` setting. `Tolk_engine.target` is a function
  of the device: its vendor's kind, which its driver declares
  (`Rig.capability`), and its `arch`, or the host's CPU target.
  `Device.renderer` takes the target and renders for it with the renderer
  of its kind. `Helpers.Target` keeps the record and its syntax, which tests
  and the goldens' generator write. A host compiles for its `Rig.arch`: on
  x86-64 its psABI level, such as `x86_64,x86-64-v3`, and on arm64
  `generic` with the features its arch names, as in `arm64,generic,fp16`;
  this machine's host also tunes for its processor, as in
  `x86_64,x86-64-v3,tune=native`.
- **Reason:** (c). rig places values on devices, and a target read from the
  environment applies to every device of the process; a malformed `DEV`
  meant for another program failed every program linking tolk. A program
  records the arch it was compiled for, which `Rig_program.load` checks
  against the device's, and `native` would use instructions beyond the level
  it records; tuning adds none.
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
- **Reason:** (d): `Vega.Loss_scale.step`, whose `finite`, the conjunction
  of `Nx.all (Nx.isfinite g)` over every gradient, selects each update. On
  fudge's RTX 5000 Ada, rune's `Cuda/finite-selects-148-float16` takes
  295-302 us against 5.0 ms with each consumer reducing again.
- **Pinned by:** the Rangeify suite's `sum_all_broadcast` and `finite_checks`
  rows of `kernel_counts.golden` and their kernel goldens, and its
  `reduce_expand_child` and `preserve_multistage_reduce` rows; the Jit suite's
  `nonzero` and `masked_select` goldens. On rune: missing (the old rune's
  `lower_linalg` goldens `qr_q` and `qr_r` went with its rewrite).

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
  buffer fits in shared memory. The shared memory checked is the sum of the
  grouped reductions' buffers. A reduction that reads another stays out,
  since the final reductions of shared buffers that one thread range closes
  run as one loop.
- **Reason:** (d): the kernel of D100's `finite`, whose 16 threads each ran
  the other 147 reductions whole. On fudge's RTX 5000 Ada, rune's
  `Cuda/finite-of-148-float16` takes 165-174 us against 1.45 ms with the
  group on the first reduction alone.
- **Pinned by:** the Postrange suite's cases `sibling_sums_group` (a
  reduction the amount does not divide stays out), `ten_sums_group` (eight
  of ten buffers fit) and `single_kernel_softmax_group` (the sum that reads
  a maximum stays out); the Codegen suite's `two_grouped_stores_local`
  goldens and its claim that a barrier follows each local store.

## D102. Withdrawn

An empty argument of a precompiled call reached the kernel graph as its
scalar, for rune's staged scan, which now replaces it by zeros in the step
(`Loop`). `Rangeify.no_indexing_calls` follows `schedule/rangeify.py`
`no_indexing_calls`, and such a call fails verification on the expand, as there.

## D103. An integer constant is written as the value its type holds

- **tinygrad:** `renderer/cstyle.py` `base_rewrite` (the constant rules,
  which write a signed or 8- or 16-bit constant unwrapped).
- **tolk:** `Cstyle.int_const`.
- **Differs:** a constant of an integer type is written as the value the type
  holds: the integer part of a float, wrapped to the type's width, with the
  suffix or cast of its type. The least `int64` is written
  `(-9223372036854775807l-1)`. An infinity or a NaN, which has no integer
  value, is a conversion of the float as the program runs. tinygrad writes
  the least `int64` as an unsigned literal negated.
- **Reason:** (b): rune compiles integer constants at every width's
  extremes, the least `int64` among them; a float constant converted to an
  integer was written as a literal past 64 bits, which C refuses.
- **Pinned by:** the Cstyle suite: `execution on the host › stores each integer
  constant as its type holds it` and `› converts an infinity or a NaN to an
  integer type`; `clang_transcendental_f16.golden`; rune's `rune jit` suite:
  `classes › integer constants keep their bits and wrap`. On rune, a float
  converted to an integer past 64 bits: missing (the old rune's
  `test_jit_programs` went with its rewrite).

## D104. A value read widened from a narrower type is stored narrow

- **tinygrad:** `schedule/rangeify.py` (`bufferize_to_store`, which
  stores a stage's value at its committed dtype).
- **tolk:** `Ops.widens`, `Rangeify.bufferize_to_store` and
  `Rangeify.pm_add_buffers`' rule for the index of a narrow store through its
  widening. rune's `Loop.cut` keeps the same widening in a loop's body
  (`Ops.widens`), off the host, so the loop stores its source narrow, and
  rune's `Compile` stores a draw that two operations read narrow where it
  widens, off the host and unless a construct or the output takes it whole.
- **Differs:** a stage that is not stored whole, whose value is a cast to a
  wider type that holds every value of its source (`Dtype.can_lossless_cast`,
  `bool` and the weak types excepted), stores the source at its own type, and
  each index of the buffer widens the value it reads. Where the stage is
  placed and how it is laid out do not change.
- **Reason:** (d): rune widens a product's `bfloat16` operands to `float32`
  before multiplying them (D29), so gpt-oss-20b stored `float32` values of
  `bfloat16`, twice the bytes, which no CUDA tensor core loads. On fudge,
  kaun's `GptOss/prefill-512` takes 132 ms against 367-368 ms with each
  stage stored at its committed type, and `GptOss/prefill-512-Q8_0` 141 ms
  against 1.35-1.36 s. The rows' fudge wall-time baselines are pending the
  scheduled quiet re-record.
- **Pinned by:** the Heuristic suite's `projection_of_stored` kernel, whose
  golden reads the stored attention output as `bfloat16`, and its cases
  `projection_of_stored_{metal,cuda,amd}` (the tensor cores apply); kaun's
  `GptOss/prefill-512` and `GptOss/prefill-512-Q8_0`. rune's loop rule: the
  `Rune.Loop` suite's `a widening is passed narrow, but on the host and to a
  call`, and its GPU suite's `products › a scan whose step narrows its
  weights computes, unoptimised, its steps' bits` and `› … is within the
  bound of its products`. rune's draw rule: the `Rune.jit` suite's `jit › on
  the host, a draw two operations read is stored whole` and its GPU suite's
  `draws › a draw two operations read is stored as the bytes it widens`.

## D105. A tensor core takes rows that share the other operand's tile

- **tinygrad:** `codegen/opt/postrange.py` (`_apply_tc_opt`: M runs along a
  range of `in0` that `in1` does not read, N along one of `in1` that `in0` does
  not read), `Scheduler._apply_tc_opt` (the core multiplies the multiply's own
  operands), and `codegen/opt/heuristic.py` (`hand_coded_optimizations` tries
  the cores on a kernel of one reduce axis only, below `TC_OPT=1`).
- **tolk:** `Postrange.Scheduler.in_tiles`, `Postrange.Scheduler.try_core`'s
  `roles` and `Postrange.Scheduler.use_wmma`'s `own`.
- **Differs:** a range both operands read is the core's M (or N) range of one
  of them when the other reads it only as `r / d`, `d` a multiple of the
  core's M (or N): that operand reads one value over each tile of the core,
  and the core reads its bits of the range at 0. Either operand may be the
  core's A: the choices with `in0` as A come first, as tinygrad orders them,
  then those with `in1`. D450 tries the cores on any number of reduce axes.
  Blocks of up to 32 rows stream before the cores unless they do more than
  64 multiply-adds per byte of their matrices (D89).
- **Reason:** (d): `Nx.map_segments` over `Nx_quant.apply` on a prompt
  multiplies blocks of up to 16 rows by their expert's matrix,
  `W[owner[row / 16]]`, where tinygrad finds no M range, and the decoded
  bytes split the reduce into bytes and a byte's two values (D89), which
  tinygrad's heuristic leaves to the hand-coded path; a block of 8 fits
  CUDA's N of 8 and not its M of 16, so its rows are the core's B. On fudge,
  kaun's `GptOss/experts-at-prefill-512` runs 4.66-4.68 ms on the cores
  against 18.5-18.7 streamed.
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

## D106. A value a reduction reads again at its outputs stays stored

- **tinygrad:** `schedule/rangeify.py` (`remove_bufferize`, which
  inlines a stored value that reads at most three buffers and no reduction
  reading a buffer, whoever reads it) and `schedule/indexing.py`
  (`run_rangeify`, which stores a value its consumers read at different
  indices without marking it broadcast).
- **tolk:** `Rangeify.remove_bufferize`'s `read_by_every_output` and
  `reread`, with `Rangeify.reduced`, `Rangeify.mark_across` (through
  `reread_across` and `data_reads`), `Rangeify.read_bytes` and
  `Rangeify.held_bytes`, and `Indexing.assign_ranges`, which marks such a
  value `Broadcast` when its consumers broadcast it.
- **Differs:** a stored value that is broadcast and reads more than one
  buffer stays stored when a reduction reads it on its data path (not as
  a load's position or a selection's condition) along a range it sums,
  not one whose sum selects one trip, and the reduction's outputs run
  along a range the read does not: each output along it, as a product's
  output columns, would compute the value again. A read inside a sum that
  selects one trip counts where an enclosing sum reads it so. It must do
  work and hold no more bytes than it reads. A read in the value of such a
  stage, one neither kept whole nor running a reduction of storage, counts
  too where its index runs over at least as many points as the stage it
  reads holds: inlined, that stage is read again with it. A value whose
  elements times the work of each are within 360 operations, the budget
  that bounds an inlining's copies in nodes, is inlined, since storing costs a launch the schedule does not
  price and computing it again adds no more to each output than an
  inlining may. A value read only through the reduce ranges of its consumer
  stays stored without these gates, as a matrix-vector product reads its
  vector. A value stored for consumers that read it at different indices,
  as the query, key and value projections read one normalised activation,
  is marked broadcast when they broadcast it, as D77 marks a value that a
  broadcast ends. So is a gather's source that its gathers read at more
  positions than it holds (D77): a sum of such reads through its reduced
  ranges alone keeps a source of several buffers stored.
- **Reason:** (d): gpt-oss-20b on CUDA, whose projections computed the
  normalised activation again at each output, and whose experts' gathered
  rows, given the activation's buffers, were stored in its place. On
  fudge, `GptOss/decode-step-cache-1024` took 10.2 ms against 11.2-11.3 ms
  without the rule for a value every output reads.
- **Pinned by:** the Rangeify suite's `normed_vecmat` (a normalised vector
  read by a product), `normed_vecmats` (by two products, stored once),
  `normed_matmul` (a normalised matrix, stored), `selected_scaled`
  (stored), `rope_decode`, `gather_rotary` and `prefix_sum_broadcast`
  (values within the budget, inlined) rows of `kernel_counts.golden` and
  their kernel goldens, `variable_read_twice`
  (a value two full reductions read, inlined), its `get_kernel_graph ›
  gathers that read more than their table holds` and `get_kernel_graph ›
  values a reduction reads again`, and rune's `rune norms › compiled, one
  pass over x before its map`.

## D107. Withdrawn

A batch submitted a streamed queue after the queues it waited for. rig orders
submissions across queues (D146).

## D108. Withdrawn

On the host, a load a reduce reads filled the lanes by 2; merged into D186,
where the fill is a candidate within the register budget.

## D109. Withdrawn

On the host, a kernel of several reduces unrolled within 32 lanes; merged
into D186, whose budget adds accumulators only for an upcast of the outputs.

## D110. Withdrawn

A matrix-vector product with few outputs took fewer threads across them;
merged into D89.

## D111. A hung AM device is lost, never recovered in the process

- **tinygrad:** `runtime/ops_amd.py` (`_collect_interrupts`, whose `reset`
  resets the compute engines through `AMDev.recover`, and `on_device_hang`)
  and `AMDDevice.__init__` (`can_recover`, true under AM off a virtual
  function); `runtime/support/am/amdev.py` (`recover`).
- **tolk:** none: rig's AMD driver owns the GPU, and loses it on a hang,
  driver-less after its hang limit and under the kernel's driver when the
  driver reports one.
- **Differs:** a hang or fault loses the device under AM as under KFD. Every
  later operation on memory it can reach raises `Lost`, and the process does
  not reset the GPU. An open of the GPU then gives a fresh device under KFD,
  with new queues, and refuses it under AM; the next process that opens it
  resets it.
- **Reason:** (c): after a reset nothing the device held can be trusted, and
  nx's values are immutable, so a value must fail loudly rather than outlive
  its storage: after a fault every operation on the device's values raises,
  naming the device and the fault.
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
  (`Setting.beam_padto` to `Setting.beam_estimate`), `Search.beam_search`'s
  key and `Tolk_engine.part`.
- **Differs:** a table also names everything besides the source that
  determines a binary:
  - for Clang, the digest of what `clang -###` states it runs for a source,
    its version and installation first, of the command of its compiler
    proper that compiles a short and a long source, with the IR verifier on
    (D275: the processor and features `native` resolves to, every option),
    and of the length between them (D167), with `-ffile-compilation-dir=.`
    keeping the working directory out. Clang runs once per process to state
    its command: the options depend on the LLVM release it states (D167), and
    a process asks first for those of the release the last process found, a
    hint kept in the disk cache's table `clang_llvm` while `CCACHE` holds,
    and again when Clang states another. The hint picks options only, and
    the table digests what Clang states for them, so a stale hint costs a
    second run and never an object of another compiler. A key on the
    binary's path, size and modification time alone would miss the
    processor `native` resolves to, Clang's configuration files and the
    environment its driver reads;
  - for MTLCompiler, NVRTC and comgr, the build of macOS or the library's
    file, size and modification time, and the options, PTX or cubin
    included.

  The table is the compiler's identity whatever `CCACHE` holds: the keys of
  programs and searches take it, and `CCACHE`, read when a binary or a
  program is compiled, decides only whether the disk keeps it. A program is
  kept on disk under its compiler's table. The beam search's key adds the
  library's sources, the renderer, its compiler's table, the part of the
  device it times on (its arch, last-level cache and an AMD GPU's compute
  units, as rig states them: `Tolk_engine.part`), and the settings that
  shape compilation (`Setting.shaping`), among which those that pick
  the candidates (`TC` and `BEAM_PADTO`, D114) and those the search
  declares, `BEAM_UOPS_MAX`, `BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`,
  `BEAM_MIN_PROGRESS` and `BEAM_ESTIMATE`. `BEAM_STRICT_MODE` is not keyed:
  a strict search that completes finds what another finds.
- **Reason:** (b): raven's test suites and rune share the default cache, so a
  hit must answer the compilation it stands for. tinygrad's key answers a
  changed flag, an upgraded toolchain, another compiler of the same renderer
  name or a search under other settings with a binary or a search made for
  something else; with `CCACHE=0` two compilers that differ in their options
  alone share programs. Its search's key names the device kind alone, so a
  search timed on one part answers every part and arch of that kind with what
  was fastest on the first.
- **Pinned by:** `Tolk.Compiler_cpu › cache`, `Tolk.Compiler_metal › cache`,
  `Tolk.Compiler_cuda › cache`, `Tolk.Compiler_amd › cache`,
  `Tolk.Renderer › Compiler › the table is asked for once, when first
  needed`, `› ccache is read when compiling` and `› compile_cached compiles
  every time with ccache off`, `Tolk.C › identity`, `Tolk.Codegen › programs
  are kept › a program is not returned for a compiler of other options, with
  CCACHE=0`, `Tolk.Codegen › programs are kept on disk › a program is not
  read back for a compiler of another table`,
  `Tolk.Search › a search kept under one setting measures again under
  another`, `› a search kept for one part measures again on another`, `› a
  search kept under one value of a setting declared by its caller measures
  again under another` and `› the search's settings shape compilation`, and
  `Tolk_engine › target › a software device is the host's part`.

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
  otherwise. The caches key on every `Output` setting (`Setting.shaping`,
  D63).
  - `BEAM` and `JITBEAM` reach output: `compile_linear` writes the width
    into the kernels it searches. `JITBEAM` is a setting whose `None` stands
    for `BEAM`'s current value. A function compiled with an explicit width
    keys on that width in `BEAM`'s entry, without `JITBEAM`'s, since the
    width overrides both.
  - `CC`, `CUDA_PATH` and `ROCM_PATH` reach the process: they are read when
    a compiler is made, and its table names the tools and options they pick
    (D113).
  - `TC_OPT` is the level of hand-coded optimizations alone, default `0`;
    the search's candidates take the level `2`, a constant
    (`Search.tc_opt_all`), which admits every kernel the lower levels admit.
    `TC_OPT` in the environment does not reach the search.
  - `TC` and `BEAM_PADTO` are read when the search makes its candidates
    (`actions` is a function), so a `context` binding of `TC` reaches the
    search, where tinygrad's search sees the environment alone.
  - `compile_linear` reads neither `DEBUG` nor `PROFILE`, since it makes no
    batch to stamp (D146): rig's profiles record each submission's span, and
    the engine makes each kernel a submission of its own from `DEBUG` 2,
    when it prints each kernel's time. There is no `PROFILE`.
  - A variable tinygrad reads on each use, such as `DBGTV`, is read when the
    program starts.
  - A library that compiles with tolk declares its own variables the same
    way, so rune's jit keys on its `Output` settings too, and beyond them
    only on what is not a setting of tolk.
  - `ALLOW_TF32`, default `false` as in tinygrad, is an explicit choice of
    precision: a TF32 core rounds each `float32` operand to 10 mantissa
    bits, so a contraction compiled under it may err past `Nx.contract`'s
    bound.
- **Reason:** (b): rune's jit memo and tolk's program, schedule and search
  caches key on `Setting.shaping`, and a variable read outside a declaration
  is missing from every key: the search's raw `TC_OPT` kept a search made
  with `TC_OPT` unset for a process with `TC_OPT=0`. A setting read by
  compilation but classed `Process` must be keyed again by each caller that
  keeps what it compiles, as rune's jit did for `BEAM`, `JITBEAM` and
  `DEBUG`; `DEBUG` read by
  `compile_linear` would reach output and key every cache on verbosity.
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
  compiled (`Codegen.to_program` completes the linearized program). The
  search keeps and drops the same candidates.
- **Reason:** (d): a beam search of lorenz_simple's step (`Rune.jit ~beam`),
  whose candidates past the cap are the largest unrolls, the slowest to
  compile. On fudge, the tolk bench's
  `search/beam/r_128_16_3_4_4_3_3_400_400` takes 0.47 to 0.59 s against 0.71
  to 1.58 s when each candidate compiles before the cap drops it.
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
  (`Renderer.Compiler.compile_cached`); its program enters neither of
  `to_program`'s tables. A search keeps its choice, and the kernel compiled
  with it keeps its own program and binary, compiled apart from the
  candidate it was, since a candidate is named `"test"`. `to_program` takes
  only a kernel's sink, and `compile` only what `linearize` returns:
  `to_program ast ren` is `compile (linearize ast ren) ren`, kept. Every
  candidate compiles, and a binary timed before is dropped after its
  compilation, as in `codegen/opt/search.py` `beam_search`. tinygrad's
  `do_to_program`, and `lower_and_compile` with it, also resume a program
  from any stage.
- **Reason:** (a): tinygrad's worker processes (`get_worker_pool`) free the
  programs they kept when they exit after `BEAM_MAX_TASKS_PER_CHILD` tasks,
  where tolk compiles on domains of the calling process (D5), whose `to_program`
  table would hold every candidate of every search for the life of the process.
  A stage resumed by `to_program` is one reached by `Codegen.linearize` or
  `Codegen.compile`, so no caller loses one.
- **Pinned by:** `Tolk.Search › a search keeps each candidate's binary in its
  compiler's table, as a kernel compiled with it keeps its own`; an absence:
  `Search.try_compile` calls `Codegen.linearize` and `Codegen.compile`, which
  keep nothing; `Tolk.Codegen › linearize and compile` (every test).

## D117. Withdrawn

A search timed its kernel first and progressed only past `BEAM_MIN_PROGRESS`
on every sample, which gained nothing on fudge. `Search.beam_search` follows
`codegen/opt/search.py` `Search.beam_search`.

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

Storage a stage or copy fills carried its value's bounds into the parameter
that reads it. With D211 withdrawn they changed nothing on fudge, and a gain
held only on another machine is no reason.

## D123. The engine runs a loop of calls while a flag holds

- **tinygrad:** `uop/spec.py` `spec_shared` (a `BACKEDGE` closes a bound-less
  range, inside a kernel), `engine/realize.py` `run_linear` (every call once,
  in order).
- **tolk:** `Spec.shared` and `Spec.kernel_graph` (a `Backedge` around
  calls, in the tensor and kernel-graph specs), `Schedule.is_loop` and
  `Schedule.linearize` (`create_schedule` keeps a back edge of a call),
  `Memory.uses` (the plan counts its calls and flag), and
  `Tolk_engine.loop`.
- **Differs:** an `Op.Backedge` stands around a call of a schedule, or a
  linear of calls, with a loop range of at most as many trips and a flag, one
  boolean of storage the calls write, which the program makes or a caller
  binds. The schedule keeps it as an entry, and the engine makes it a loop of
  rig's program (D146) that reads the flag before each trip and runs the calls
  while it holds, its trip in a word of the run's ints, which a call's scalar
  argument reads too (D140). A range of one trip around a call folds to a
  constant, which `Shape.symbolic` drops from the end, so the call runs once;
  tinygrad deleted that rule (07efd3877), since its ends close only a kernel's
  ranges (`codegen/simplify.py` `flatten_range`).
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

- **tinygrad:** `codegen/opt/search.py` `timeout_handler`, and the alarm of
  `BEAM_TIMEOUT_SEC` that `_try_compile` sets and clears.
- **tolk:** `Search.try_compile`.
- **Differs:** a candidate's lowering and compilation run to their end.
  tinygrad drops a candidate when an alarm of `BEAM_TIMEOUT_SEC` (10 s) rings
  in its worker process; the alarm stops Python code and a Clang child, but
  comgr, NVRTC and MTLCompiler run in the worker through ctypes, so such a
  candidate is dropped only after its compile ends, and a compile that hangs
  hangs its search. tolk bounds a candidate by its size instead:
  one of `BEAM_UOPS_MAX` instructions or more is dropped before it is compiled
  (D115), and lowering raises past `REWRITE_STACK_LIMIT`.
- **Reason:** (a): tinygrad bounds a candidate as one unit in a worker
  process. A tolk candidate is lowered on a domain (D5), which no
  signal stops, and a process with domains cannot fork; a limit would reach
  only the toolchain processes and promise what lowering does not keep.
- **Pinned by:** an absence; the bound tolk keeps is pinned by D115's
  `Tolk.Search › BEAM_PADTO=1 ... › a candidate of BEAM_UOPS_MAX
  instructions or more is not compiled`.

## D125. Withdrawn

A launch descriptor's field held all of its value. rig's NV driver writes its
launch descriptors (D146).

## D126. The transcendental functions are within one ulp

- **tinygrad:** `codegen/decomp/transcendental.py` `TRANSCENDENTAL_DTYPES`
  (float16 included), `cody_waite_reduction` (by half turns, whose float32
  parts are the first three of Sleef's split for FMA and the fourth of its
  other split), `sin_poly`, `sin_poly_small` and `sin_poly_large`, `xsin`, and `xlog2`.
- **tolk:** `Transcendental.transcendental_dtypes`, `Transcendental.radians`,
  `Transcendental.cody_waite`, `Transcendental.sin_kernel`,
  `Transcendental.cos_kernel`, `Transcendental.xsin` and
  `Transcendental.xlog2`.
- **Differs:** `xexp2`, `xlog2` and `xsin` are defined in float32 and float64;
  float16 computes at float32 and rounds once, as bfloat16 and the 8-bit
  floats did. `xexp2` is unchanged. Each is within 1 ulp of the correctly
  rounded result.
  - Both reductions remove quarter turns and return the remainder in two
    floats, the rounded remainder and the rest. Cody-Waite subtracts `q pi/2`
    in four parts whose products by a quotient below `2^15` are exact, and
    carries the rounding of the third difference to the last, where
    tinygrad's fourth term belongs to another split of pi and leaves
    `1.2e-10 q` behind; Payne-Hanek's
    remainder is summed and scaled in two floats (D31).
  - The sine and cosine of the remainder are fdlibm's kernels, which take the
    rest to first order, where tinygrad's is one polynomial on half turns
    evaluated as `d p(d^2)`. `xsin` returns a zero of either sign unchanged.
  - `xlog2` is fdlibm's `log2` (`e_log2.c`): `m` in `[1/sqrt 2, sqrt 2)`,
    `log (1 + f) = f - h + s (h + R)`, `f - h` split so that its product by
    the leading part of `1/ln 2` is exact, the exponent added last with its
    rounding carried, where tinygrad's is a polynomial in `(m - 1)/(m + 1)`
    whose rounding scales into the result.
- **Reason:** rune computes its kinds with nx's sequences (rune's
  `Lower_real.unary`), reading no tolk `sin`; it reads tolk's `exp2`, which
  is tinygrad's, and its `log2`. `xsin`: (e), since tinygrad's float32
  `xsin` near a multiple of pi below 30 has no correct bit. `xlog2`: (b),
  unconfirmed: rune's attention takes each row's `lse` from it (rune's
  `Lower_attention.lse_of`) within the bound rune states, where tinygrad's
  `xlog2` is 5 units off; no test shows the bound failing at 5 units.
- **Pinned by:** the Transcendental suite
  (`test/codegen/decomp/transcendental`): `accuracy › accuracy.golden`
  (mpmath's correctly rounded values,
  `test/gen/codegen/decomp/transcendental.py`), and `xsin ~fast` below 30,
  within an ulp of them; `special values › sin <type> -0x0p+0 is -0x0p+0` and
  the subnormal rows; `types › <function> refuses a node of another type ›
  half`; the graph goldens and `values.tolk`; and rune's `rune attention ›
  jit within the bounds of eager`, on the host.

## D127. Withdrawn

A selection moved into a converted gated load only when the load's type held
its other value. `Gater.move_where_load` moves it only into a load of the
selection's own type, as `codegen/late/gater.py` `move_where_load` does.

## D128. A committed float operation reads a weak float operand rounded

- **tinygrad:** `uop/ops.py` `UOp._min_max`, which bounds a weak float operand
  of a committed float operation by its value as written.
- **tolk:** `Ops.rounded` and `Ops.operand_bounds`.
- **Differs:** a committed float operation bounds a weak float operand rounded
  to its own type, as compiled code computes it. A bound that rounds to NaN,
  past a type without infinities, leaves the type's bounds.
- **Reason:** (b). rune's `Nx.pad ~value:5e-324` of a float32 tensor compiled
  to `where c x 5e-324`, bounded away from 0, so `== 0` folded to false where
  eager compares 0 with 0.
- **Pinned by:** the Ops suite (`test/uop/ops`): `bounds › a committed float
  selection holds a weak constant at its type, where it may round to zero`.
  On rune: missing (the old rune's `a compiled program › computes eager's
  bits`, whose examples held the pad, went with its rewrite).

## D129. Withdrawn

A search kept nothing of its candidates; merged into D116, which holds what
a search keeps.

## D130. Withdrawn

A round's candidates were timed in their own order. `Search.beam_search` times
each as its compilation ends (`Worker.iter`), on every device, as
`codegen/opt/search.py` `beam_search`'s `pool.imap_unordered` does; on one
domain, in their order.

## D132. A contiguous view's rewrite stops at the effects its storage waits on

- **tinygrad:** `uop/ops.py` `UOp.contiguous_view`, whose rewrite with
  `pm_mops` and `symbolic` visits every node the index reaches, the effects
  an `AFTER` orders its storage after included, and returns `out.src[0]`.
- **tolk:** `Prepare.pm_stop_at_effects` and `Prepare.contiguous_view`.
- **Differs:** the rewrite stops at void nodes, the effects storage is ordered
  after, and leaves them and what they compute as they are, so the storage it
  returns is the node of the view's graph. The effects decide nothing of where
  the view's elements lie. The graphs they compute hold rune's gathers (D73),
  which `pm_mops` moves through a reshape into an index whose shape repeats
  the gather's, and weak-integer arithmetic, on which `Divandmod`'s
  `divide_by_gcd` raises. tinygrad's tensor graphs hold neither.
- **Reason:** (b): the kernel split calls `Prepare.contiguous_view` through
  `Schedule.view_of` (from `Schedule.pm_callify_ctx_collect` and
  `Schedule.contiguous_mops_to_view`); rune calls neither. norn's NUTS warmup and sampling over
  sixteen chains, compiled as one function, raised in `divide_by_gcd` and on a
  reshape of a gather's index of four axes.
- **Pinned by:** the `Prepare` suite: `contiguous_view › a view of storage
  after effects is a view of that storage, whatever the effects compute`.

## D133. A greatest common divisor's coefficient is a scalar

- **tinygrad:** `uop/ops.py` `UOp.gcd` (`uops[0].const_like(...)`) and
  `uop/divandmod.py` `fold_divmod_general` (`divide_by_gcd`). Over terms with
  a shape a divisor of 1 is a broadcast 1, which divides no term:
  `UOp.stack(a, b) // 3` raises in `unwrap`.
- **tolk:** `Shape.gcd`.
- **Differs:** the coefficient is a scalar constant of the first term's type,
  whatever the terms' shape, so a divisor of 1 is the `CONST` 1, and a larger
  one divides each term as `divides` does.
- **Reason:** (b): the norn compile D132 names reached `divide_by_gcd`, before
  D132, on a weak-integer tensor of rune's gather positions, and raised there. The rule applies to any
  vector of index arithmetic.
- **Pinned by:** the `Divandmod` suite: `values › each rewrite of a random
  division of vectors keeps each lane's value`.

## D134. Withdrawn

A widening cast of an unsigned `x land y`, or `x lsr k` with `k` within the
operand's width, widened the operands; kaun's decode rows on fudge showed no
gain. The cast stays where it is, as tinygrad's `uop/symbolic.py`
`Shape.symbolic` keeps it.

## D135. Withdrawn

A Metal command buffer signalled through the device's signaler. rig's Metal
driver signals its own (D146).

## D137. Consumers of one domain share their loops

- **tinygrad:** `schedule/indexing.py` `run_rangeify` (new ranges for every
  stored value; a node its consumers index apart is stored, and each
  reduction takes new ranges for the axes it reduces); `schedule/rangeify.py`
  `remove_bufferize` (inlines such a store into each consumer) and
  `split_store` (one kernel per store); `codegen/__init__.py`
  `merge_reduce_ends`. Upstream calls several outputs per kernel pending:
  `test/runtime/test_assign.py` `TestAssign.test_simple_assignment_multioutput`,
  `TestAssign.test_permuted_reduceop_multioutput_dual_use` and
  `TestAssign.test_permuted_reduceop_multioutput_dual_use_possible`, and
  `test/runtime/test_jit.py` `TestMultioutputJit`, skip; codegen already
  compiles a kernel of one end over a group of stores
  (`test/runtime/test_custom_kernel.py` `flip_contract_kernel`).
- **tolk:** for stores, `Indexing.region`, `Indexing.views`,
  `Indexing.waits_over`, `Indexing.share_loops` (its `crossed`),
  `Indexing.loops_of` and `Indexing.end_shared` (which
  `Indexing.bufferize_and_index` returns for each store of a group), with
  `Indexing.data_srcs`, `Indexing.create_bufferize_and_index_based_on_ranges`
  and `Rangeify.handle_after`, which take a group (`Ops.is_group`) as its
  stores and no state of a buffer, and `Rangeify.is_noop_after_dep`, which
  takes a group emptied of its stores as no effect; for reductions,
  `Indexing.reaches`, `Indexing.reduce_loops` and its use in
  `Indexing.assign_ranges`; `Codegen.merge_reduce_ends`.
- **Differs:** consumers of one domain take one set of ranges, so a node they
  share is indexed alike and computed once.

  Stores: before ranges are assigned, the stores of the tensor graph that
  have a loop are grouped. Two join when:
  - their values have one shape and one device, and they write different
    storage;
  - neither reads the other or the storage the other writes, through
    anything;
  - neither waits for a write over storage that the other reads in a state
    before it;
  - the computed nodes each reaches before storage or another stored value
    share one;
  - no computed node both read through views (movements) is read through
    different views, so two stored slices of one reduction stay two kernels;
  - neither reaches a reduction the other does not, since a kernel of two
    reductions can lose the optimisations each would take.

  Two groups join only when neither reads the other through the stores of a
  third. A group's stores take one set of loops and end as one end over a
  group of the stores, which the kernel split makes one kernel. A group whose
  every store is dropped, as a store of a value into itself is, leaves the
  empty stack, which waits on nothing and makes no kernel. Outputs that share
  only loads stay apart; finished calls may still share a launch (D404).

  Reductions: a reduction whose output is indexed by the same ranges as an
  earlier one's in the walk, and that reduces axes of the same sizes, takes
  that reduction's reduce ranges when no reduction of the group reaches it,
  through anything. A group of a loop that several reductions close takes a
  buffer of shared memory for each (D101). Two ranges that reductions close
  merge in every reduction that closes them or not at all, as tinygrad merges
  ranges since df93a6cb8. A sum over a range its source does not read drops it
  (`pm_reduce_unparented`), so it can stop closing its group's ranges.

  Matrix products that a tensor core takes never share a kernel, a group of
  reductions or not (`Indexing.split_products`, before the stores are
  grouped). A matrix product is an addition of products, through movements
  and at most one cast, whose factors each read an axis of the sum's output
  that the other does not read; a core takes it where one of its device's
  cores (`Heuristic.cores`, which the engine passes to
  `Schedule.create_linear_with_vars ~cores`) multiplies both factors into
  the sum's type (`Tc.takes`, which `Postrange`'s `try_core` asks too) and
  fits its rows, outputs and block as the heuristic asks before it takes a
  core (`Heuristic.core_fits_shape`, which `Heuristic.core_fits` reads): a
  sum of 32 rows or fewer, which the heuristic streams, stays grouped. Each
  factor is asked as decoded where it converts integer codes into floats,
  as a Q8_0 matrix's, so a product whose streamed factor is decoded takes a
  kernel of its own where a core holds its rows (D230). Of
  two such products a stored value's kernel computes, neither reading the
  other, the later in the walk is stored: as the first narrowing cast its
  one consumer reads it through, past casts and the addition of an initial
  value that computes no reduction, or as itself where there is none. Sums
  of different lengths are covered, a product another reads is not, and
  there is no threshold beyond the core's own shape. The stored value is
  the value the kernel computed: bits equal the fused kernel's at the same
  optimisations; against main, within the contraction's bound, since a core
  sums each product in its own order. A device with no core for the
  products' types, as the host, keeps the grouping above.

  Both merge every end, a store's as a reduction's, and give fresh ranges to
  every end that closes a range an earlier end closes, so each range has one
  end; tinygrad does so only for an end at a different depth. No recorded
  kernel has such ends.
- **Reason:** for stores, (d): `Nx_wide`'s high and low parts and rune's
  gradients of several arguments share one chain; without the grouping,
  rune's `compile/special/betainccinv-grad-f32` row ran 593 kernels in place
  of one, with a first call of 29.7 s against 6.1 s on fudge. Through views,
  (b): rune's fully sharded training (RFC 0005's memory bound), where grouped
  slices were all live at once. The check costs kernel graphs up to 2% more
  words (tolk's `*/kernel_graph` rows). A group emptied of its stores, (b):
  `Rune.jit` of `Rune.jvp` of a scan whose step gives back its carry made a
  kernel of no storage, and on a set of GPUs raised. For reductions, (d):
  rune's reverse mode sums one reduction per broadcasting operation; on
  fudge, rune's `Jit/grad-broadcast-radius-float64-104x104` takes 215 us as
  one loop against 383 us as 28. For products a core takes, (d): `Postrange`'s
  `use_wmma` takes a core for one reduction over its K, so a kernel of
  several products runs each without one; tessera's distillation step on an
  RTX 5000 Ada spent 596 ms of 1.55 s in the in-projection's input
  gradient, three products of one cotangent in one SIMT kernel. Splitting
  them is owed its GPU measurement.
- **Pinned by:** for stores, the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › outputs that share a computation` (every test, `slices
  of one product through two views run as two kernels`, `slices that also
  share a computation run as two kernels`, `outputs that read a product
  through one view run as one kernel` and `outputs that store their own
  storage back are no kernel` among them); its recorded
  `read_around_assign_kernels`, `crossed_pairs_kernels` and `kernel_counts`,
  and the Postrange suite's `where_max_multioutput`, `padto_where_multioutput`
  and `axes` cases. On rune: the `compile/special/betainccinv-grad-f32` bench
  row, and `rune jit › fully sharded` for the views and `rune scan ›
  transformations › derivatives and maps over views are the fold's` for an
  emptied group (the old rune's `Rune nx.wide › kernels` went with its
  rewrite). For reductions, the Rangeify suite: `get_kernel_graph ›
  reductions that share a domain` (every test), and its law over generated
  programs, whose `sum k + max k` and `x - sum k` steps cover reductions that
  share a loop; its recorded `multireduce_*` and `ugly_reduceop_pairing`
  kernel graphs and `kernel_counts`; the Indexing suite's recorded
  `two_consumers` graph; the Postrange suite's `sum_and_max_grouped`,
  `ten_shared_sums_group` and `ten_shared_sums_group_unfit` cases, and the
  Codegen suite's `two_grouped_stores_local`. The Codegen suite's
  `reductions that share a range › a range two reductions close at two depths
  runs as two loops`. On rune: missing (went with the old rune's rewrite):
  its Packed suite's bit example of `a compiled call computes eager's values`;
  its Jit suite's `transformations › a compiled gradient of a parameter
  broadcast over a grid is its gradient`; and its bench row
  `reverse/jit-of-grad-clipped-disc-float64-104x104-host`. For products a core
  takes, rune's Lower_reduce suite: `sums of products` (every test), and the
  Rangeify suite's `get_kernel_graph › products a core takes` (every test,
  `products of 24 rows by codes of 64 run a kernel each` and `products of
  codes of 24 by codes of 64 run a kernel each` among them).

## D138. A load's index is simplified without the loads in it assuming its gate

- **tinygrad:** `codegen/late/coalesce.py` `simplify_valid_load`, which
  simplifies a gated index with `uop/symbolic.py` `uop_given_valid` down into
  the loads the index reads; `uop/symbolic.py` `gated_given_valid`, which
  refuses an index that reaches an `INDEX`.
- **tolk:** `Coalesce.simplify_valid_load`, `Symbolic.gated_given_valid`.
- **Differs:** each load in the index stands as a variable of its bounds while
  the gate simplifies the arithmetic around it, and is put back after, so a
  load keeps its own index and gate. Substituting into it would drop a gate
  the outer gate implies, such as a pad's, and the load, which runs whatever
  the outer gate, would read outside its buffer. `gated_given_valid` refuses
  an index that reaches a `LOAD`, where tinygrad's refuses only one that
  reaches an `INDEX`: memory coalescing addresses a load of several elements
  by a `SHRINK` of its buffer, with no `INDEX`. A staged core orders the slot
  its reads take after its copies of the next trip's tiles, so the reads'
  gate, assumed in the slot, reached the copies' loads and dropped the gate
  that stops them at the last trip.
- **Reason:** (b): `Nx.concatenate` of a buffer and an `Nx.take` under
  `Rune.jit`, as nested sampling's evidence does, read before the start of
  the indices and faulted; `Nx.matmul` of float32 matrices of 16384 by 440
  and 440 by 1320 cast to bfloat16 under `Rune.jit`, tessera's widths, read
  28 rows past its second operand on CUDA and faulted on NV.
- **Pinned by:** the Coalesce suite (`test/codegen/late/coalesce`):
  `indexing_simplify › a gather through a pad reads its indices only inside
  the pad`; the `codegen` suite's `stages › every access of global memory is
  gated or within its buffer`.

## D139. Construction checks no specification

- **tinygrad:** `uop/ops.py` `UOpMetaClass.__call__` (checks each node against
  `spec_full` when `SPEC` is above 1, computes its shape above 2), and
  `schedule/rangeify.py`'s `Context(SPEC=min(SPEC.value, 2))` around
  `pm_apply_rangeify`.
- **tolk:** `Ops.v`, `Setting.spec` and `Indexing.run_rangeify`.
- **Differs:** `Ops.v` checks nothing and computes no shape, whatever `SPEC`.
  `SPEC=0` checks nothing, and any other value checks the graphs passed
  between stages, as `SPEC=1` does in tinygrad. A malformed node is reported
  at the end of its stage; `Spec.type_verify ~calls:Enter Spec.full` checks a
  graph after a suspect rewrite.
- **Reason:** (a). The full specification reads shapes, which read
  `simplify`, which builds nodes, so the check is defined above construction,
  and calling it from there needs a hook that holds only while every module is
  linked.
- **Pinned by:** the `Spec` suite: `construction › builds a node that breaks
  the full specification, whatever SPEC`.

## D140. A call of a loop takes its trip as a scalar argument

- **tinygrad:** `schedule/__init__.py` `resolve_linear_call` (binds a nameless
  scalar parameter only to a variable), `engine/realize.py`
  `get_call_arg_uops` (leaves out every variable) and `get_call_kernel_args`
  (reads each scalar parameter by name); `codegen/opt/postrange.py`
  `args_from_ast`.
- **tolk:** `Schedule.scalar_argument`, `Schedule.pass_values` and
  `Schedule.resolve_linear_call`; `Realize.get_call_arg_uops` and
  `Realize.get_call_kernel_args`; `Tolk_engine.affine`.
- **Differs:** a precompiled call inside a loop may pass a scalar parameter of
  its body a weak integer expression of the loop's ranges, such as `r`,
  `2r + 1` or `n - 1 - r`, which the schedule keeps as the call's argument,
  no state. Each kernel of the body that reads the parameter keeps it,
  unnamed, in a slot past its call's arguments, bounded by its type, and its
  call passes the expression there: `get_call_kernel_args` reads a nameless
  scalar parameter from its call's slot, and `get_call_arg_uops` leaves out
  every scalar argument, an expression as well as a variable. The value
  reaches the program as any variable's does, converted to the parameter's
  type at the ABI (D28): a launch inside a loop of rig's program reads it from
  a scalar of its parameters, the sum of the loops' trip words times constant
  factors (D90). The kernel's program depends on neither the range nor its
  trips, so that loops of any range and length compile one. Beam search samples positional scalars by slot and named variables
  by name; its `link` callback takes a complete call, with the scalar samples
  in their slots and the launch sizes evaluated from the same samples, and
  `Tolk_engine.link_call` links it; the candidate's compiled body keeps its
  parameters, where tinygrad's `args_from_ast` reads every scalar's `expr` and
  constructs only buffer arguments.
- **Reason:** (b). rune's `Loop.repeat` and staged scans give each step its
  trip's index, which a kernel or a buffer of indices would otherwise compute.
- **Pinned by:** the `Schedule` suite: `create_linear_with_vars › loops of
  calls › a loop's call passes its kernel the trip as a scalar argument`; the
  `Spec` suite: `kernel_graph › loops of calls › accepts a loop's call whose
  scalar argument is an expression of its range`; the `Realize` suite:
  `get_call_arg_uops › is a call's storage arguments in order, without its
  scalar ones` and `get_call_kernel_args › is the call's argument in the slot
  of a scalar parameter`; the `Tolk_engine` suite: `link and run › a loop's
  call reads its trip as a scalar argument on each trip`, `› a back edge's
  call reads its trip on each trip, at most its trips`, `› a loop of one trip
  passes its call the trip as a scalar argument`, `› loops of any range and
  trips around one call compile one program`, and the same two on each GPU;
  `Tolk.Search › a search binds a positional size in its call and launch`;
  `Tolk.Search on the host › a search runs candidates with named and
  positional scalars`. On rune: missing (the old rune's `Rune.jit › keys › a
  scan searched with beam 2 draws as eager does` went with its rewrite;
  `rune jit › jit › a function searched at a width` searches no loop).

## D141. The host computes upcast lanes as vectors

- **tinygrad:** `codegen/__init__.py` `do_devectorize` (splits every
  elementwise operation, load and store of several lanes into one per lane, on
  every target) and `lower_reduce` (a reduce's unrolled lanes indexed out of
  its source); `codegen/late/coalesce.py` (register memory never merges);
  `renderer/cstyle.py` `base_rewrite` (every conversion a C cast, for a vector
  a reinterpretation of its bits).
- **tolk:** `Renderer.t`'s `vector_lanes`; `Codegen.lanes_of`,
  `Codegen.in_order`, `Codegen.vector_operation`, `Codegen.lanes_at`,
  `Codegen.split_widths`, `Codegen.split_lanes`, `Codegen.distinct_lanes`,
  `Codegen.vector_dtypes`, `Codegen.do_devectorize`,
  `Codegen.address_lanes`, `Codegen.pm_vector_constants`,
  `Codegen.product_operands`, `Codegen.first_lane`, `Codegen.slicer`,
  `Codegen.run_widest`, `Codegen.lower_reduce` and `Codegen.transposes`; `Coalesce.memory_coalescing`; `Cstyle.mask_widths`, `Cstyle.clang_lanes`, `Cstyle.clang_vectors` and
  `Cstyle.clang_kernel`; `Decomp_dtype.emulates`; `Uop_weak.pm_commit_weak`'s
  cast over a stack.
- **Differs:** a renderer may compute on vectors (`Renderer.vector_lanes` past
  1; Clang's is 64). There an elementwise operation of several lanes on a type
  the target computes natively stays one operation on a vector of its lanes,
  on one axis, or past `vector_lanes` lanes vectors of that many and a last of
  the rest, the last `vector_lanes + 1` as two vectors where one lane would be
  left (D241), when:
  - each source holds them in order: a vector operation, or a view (reshapes,
    broadcasts, permutations, stacks) of constants (a literal), of loads (an
    accumulator's in register memory among them), of lanes of vectors and of
    vector operations, a load or a lane in as many lanes as read it (a
    shuffle), or a scalar that every lane reads;
  - no two of its lanes read the same sources: a lane of a vector operation is
    its own, a lane of a stack its element.

  Otherwise, as when a source holds lanes computed apart, or lanes repeat,
  the operation computes each lane apart. Loads, stores, tensor-core products,
  calls, and operations on a weak type (an address) or an emulated one still
  compute each lane apart; an address is computed lane by lane through the
  vector operations it reads, so accesses still merge, register memory's too
  (where tinygrad merges memory's alone). A sum of products sums each run of
  its widest lane axis, the last of them on a tie (`Codegen.run_widest`:
  the sum is rebuilt over its source with that axis last, its running sums
  held in that order and permuted back once summed), with the axes of one
  lane after it, as one vector: each unrolled lane's products of a run are
  one vector of the run's lanes, in their order, the operation on its
  operands' slices, through elementwise operations and, in a run of at
  least 4 lanes, bitcasts that keep the lane count (as an MXFP4 code's
  decode), unless the run is narrower than a vector register
  (`Codegen.register_lanes`: 8 float32 lanes on AVX2, 4 on SSE or NEON)
  and the bitcast's lanes fill whole registers of its type
  (`Codegen.in_runs`), when it is computed whole and each run takes its
  lanes of it; in a run
  of at least 4 lanes any other node's slice is then the stack of its
  lanes, and each
  multiply-add adds one into the run's running
  sums, which an accumulator holds as vectors; each lane keeps its terms,
  their order and their roundings. A sum that reads lanes of a vector product
  apart fuses each lane into its running sum, and computes its first lane,
  which it adds to nothing, as a product of scalars, so no vector product is
  left; a selection by constant conditions picks each lane. Any number of lanes up to 64 is one
  vector, which LLVM computes in the next power of two. Stacks that each take
  lane k of the same rows, c distinct vectors of u lanes (c and u powers of
  two, c > 2), two or more of them, are the columns of the matrix the rows
  hold and are computed together by an in-register transpose
  (`Codegen.transposes`): in pieces of 16 bytes (p lanes, c >= p), each
  piece's rows concatenated by blocks, then log2 p stages of zips within
  blocks of p lanes, every step a stack of lanes of two vectors.

  Clang writes the vectors as `ext_vector_type` values: a conversion is
  `__builtin_convertvector`, square roots, truncations and multiply-adds are
  `__builtin_elementwise_*`, a scalar operand of a builtin or a selection is
  splat, a stack of lanes of one or two vectors of one type is
  `__builtin_shufflevector` cast to its type (put together lane by lane, LLVM
  folds a shuffle of shuffles back into a gather from every vector they read),
  and a vector of booleans is a mask of -1 and 0 lanes as wide as the
  values it compares, converted to the width of the masks or values it meets;
  a lane of a mask is read as a `_Bool`, and a mask of one-byte lanes as
  signed chars. A cast of a stack of weak constants commits them at the
  cast's type, as each lane would be alone (`uop/weak.py` `pm_commit_weak`).
  Which lanes a host kernel takes is D188's.
- **Reason:** (d): RFC 0025's special functions on the host. Lanes as scalars
  made Clang compile each lane's copy of the body, in a time that grows faster
  than the copies (float64 `log_betainc`'s gradient in `a` took 174 s), and
  kept its loop vectorizer off; as vectors each lane computes the scalar
  kernel's bits, and on fudge every lane apart makes rune's
  `Jit/sin-float32-64K` 806 us against 82 us and the compile bench's
  `trig/sin-nested-8` 1.98 s against 0.74 s. A load filling each lane that
  reads it: nx.quant's MXFP4 decode stays a vector without branches, and on
  fudge kaun's `Quant/host/mxfp4-product-1-token` takes 0.91 ms against 5.03
  ms with a load filling one lane; lanes that other sources fill alike stay
  apart, since Q6_K's scale converted in each of its 16 lanes made the dense
  x86 kernel 33% slower on kimchi. A matmul's upcast products
  (`clang_matmul_upcasted*`) compute their first lane as scalars: as one
  vector, a conv kernel's 9 lanes made the conv rows' linearize stage 19%
  slower for the same instructions. Up to 64 lanes as one vector: jera's
  compiled Milstein march over 3 elements compiled in 57 s lane by lane, 14 s
  as one vector. A broadcast or permutation as a shuffle: rune's map of 3
  lanes whose lanes stop apart, with each float8 operation computed in float32
  (`maps › compiled, lanes stop apart at every dtype`), went lane by lane; the
  compile bench's `lanes/held-fp8-36` takes 3.2-7.2 s against 0.49-1.04 s.
  Committing a stack of weak constants, (e): committed apart, it took the
  narrowest type that holds them, and a complex128 signalling NaN became a
  conversion of floats, which quiets it. The cost: GGUF's q8_0, q4_k and q6_k
  host products become vectors read back lane by lane, with no range apart on
  fudge (kaun's `Quant/host/q8_0-product-1-token` 2.6-3.9 ms against 1.8-3.1
  ms unpinned), against the 2-7 s of Clang per kernel at each first call that
  lane by lane cost. A sum of products' lanes as vectors: on fudge's Clang
  19.1.7, which compiles without the straight-line vectorizer (D167), a
  3328 x 440 x 1792 float32 product ran scalar multiply-adds, 11.8 and 17.5
  GFLOP/s on one P-core with its matrix [in; out] and [out; in]; as vectors
  97.2 and 62.0, the instructions that vectorizer gives (locked, min of 5).
  The transpose: written lane by lane, Clang builds each column of the
  matrix [out; in] with a broadcast and a blend per lane, and folds a
  network written as vector literals back into that gather; the network of
  `__builtin_shufflevector`s takes its 8 x 8 x 4 tile (D188) from 88.0 to
  136.4 GFLOP/s (locked, min of 5), and in 16-byte pieces a float64 one from
  22.3 to 16.2 cycles a step (llvm-mca).
- **Rejected:** a power of two of lanes per vector, made for Clang 18.1.3's
  crash on rune's uint4 sort (D167), which was not from its vector and gains
  no speed; an odd number as vectors of powers of two and its last lane apart,
  which made eight_schools' kernel `E_8_5_2` take Clang 8.9 s against 0.2 s;
  `#pragma clang loop vectorize(disable)`, which changes nothing on
  vector kernels; `-fno-vectorize` on scalar kernels and 16-byte registers,
  which trade replay for compile time (D167 takes `-fno-vectorize` for a
  source of more than 400 000 bytes).
- **Pinned by:** the Cstyle suite's `sources › by default › clang_vector_*`,
  and `execution on the host › every kernel the interpreter runs writes what
  it computes`; the Codegen suite's `vectors in programs › a host program
  computes on vectors of its lanes` (every host case), `› a host program of 6
  lanes computes one vector`, `› a host program of 48 lanes computes one
  vector`, `› a host program of 24 lanes computes one vector`, `› a host
  program of n lanes computes one vector`, `› a host program of 128 lanes
  computes vectors of 64`, `› a host program of 80 lanes computes vectors of
  64 and 16`, `› a host program of 65 lanes computes vectors of 33 and 32`, `›
  a host operation of any lanes computes vectors of at most 64 and writes what
  its kernel writes` (every case), `› a host selection by a broadcast of
  computed lanes computes no lane apart` and `› a host sum of a product's
  unrolled lanes multiplies vectors of its output lanes` (every case), `› a
  host sum whose last lane axis holds one lane multiplies vectors of the axis
  before it`, `› a host sum whose widest lane axis is not its last
  multiplies vectors of the widest`, rune's `Lower_reduce`'s `GGUF's MXFP4
  blocks › by one row, in runs of 4 rows on an AVX2 host, decode their
  codes whole`, `› a host product of operands stored along its sum gathers its
  vectors by a transpose of two-vector shuffles` (every case), `› a host selection of
  8, 16 or 32 lanes by a comparison of integers takes each lane's arm` and `›
  a host selection of 3, 6, 12, 24 or 48 lanes by a comparison of integers
  takes each lane's arm` (every case), `› a weak constant stored into four
  lanes of half on the host converts them` and `multiply-adds` (every test);
  `Tolk.Uop_weak › pm_commit_weak › a cast of a stack of weak constants
  commits them at its type, signalling NaN bits kept`; rune's `Lower › padded
  loads give eager's bits` (a complex128 signalling NaN fill); rune's `rune
  jit › classes › a uint4 sort along the last axis of every other row and
  column, flipped, gives eager's codes` and `› nx.quant's decodes compile`;
  rune's compile bench: `shaped-4k-3-lanes` and `lanes/held-fp8-36`. On rune,
  missing (went with the old rune's rewrite): its Rune.jit suite's `lanes`,
  `stated values › a window set into bits at an odd offset keeps the bits
  around it`, and lower_index's `quantised products › a product decodes a
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

Padding divided an integer division by `where(valid, d, 1)`. D355 defines it,
and the Tolk_engine suite's `a pad of integer divisions writes what its tensors
compute` and `a pad of a quotient by a constant table writes what it computes`
(on the host and each GPU) and the Codegen suite's `padto_idiv_upcast` keep the
values.

## D146. The runtime is rig: no command queues, batches or host batches

- **tinygrad:** `runtime/support/hcq2.py` (`hcq_compile`, `sched_batches`,
  the queue encoders it calls through `Compiled.pm_encode`, the host
  programs that submit them, the staging copies and `hcq_link`);
  `runtime/ops_*.py` (`pm_encode` and the queues); `engine/realize.py`
  `compile_linear` (which ends in `hcq_compile`) and `run_linear`;
  `runtime/ops_cuda.py` `CUDADevice.function`.
- **tolk:** `Realize.compile_linear`; `Tolk_engine.link`, `Tolk_engine.run`
  and `Tolk_engine.time`; `test/gen/engine/jit.py`, which stops tinygrad's
  `compile_linear` before `hcq_compile`.
- **Differs:** `compile_linear` ends once the kernels are compiled: it batches
  no call, encodes no queue, makes no host program, splits no call into lanes
  and stages no copy. `Tolk_engine.link` turns the compiled calls into a
  program of rig (`Rig_program`):
  - a kernel is a launch on each lane's device, a host program's on its host,
    its parameters laid out as `hcq2.py`'s `layout_args` lays them out;
  - a copy is a copy part of a device that copies between its two sides, or
    else a move, a queued copy of rig's (`Rig.Buffer.submit_copy`);
  - a range or a back edge is a loop;
  - consecutive calls of one device are one submission.

  rig's drivers encode each vendor's queues, order the work across queues and
  devices by the memory it uses, stage what no device copies directly, and
  time the submissions where the driver can. A host program's entry is rig's
  (D28). Storage the schedule makes is the program's, or an input each run
  makes where it is linked per run (D173), and storage a caller binds is an
  input of each run. Each linked program loads its own images
  (`Rig.Image.load`), each call's binary again, so each image's module
  variables are its own and a program shares an image among its own launches
  alone; tinygrad loads a binary once per device and process.
- **Reason:** (c). rig owns each GPU's queues: two runtimes cannot share a
  GPU, and memory belongs to the runtime that made it. Sharing a binary without
  module variables across programs would save 1-1.5 ms of a 70 ms gpt-oss-20b
  first call on CUDA, 2.5 ms of 60 ms on NV, and 3.3-4.2 ms (1-4%) of its
  prefill and first step on an R9700, against a driver function and an image
  table (about 80 lines).
- **Pinned by:** the Tolk_engine suite: `link and run › recorded`, `GPUs`
  (each recorded program, loop and refusal on Metal, CUDA, NV and AMD where
  the machine has one), `lanes and words` and `runs`; the jit goldens
  (`test/engine/jit/*_lowered.golden`), which `gen/engine/jit.py` makes
  without batches.

## D147. The x86 host holds bfloat16 as its bits

- **tinygrad:** `renderer/cstyle.py` `ClangRenderer.type_map`, which keeps
  `CStyleLanguage.type_map`'s `__bf16` on every machine.
- **tolk:** `Cstyle.clang` (on x86, bfloat16 as `unsigned short` and its
  constants as bits with `pm_bf16_ushort_const`).
- **Differs:** on x86, a bfloat16 value, pointer or vector is written as
  `unsigned short`, and a bfloat16 constant as its bits, as OpenCL writes them
  (`renderer/cstyle.py` `OpenCLRenderer.type_map`,
  `OpenCLRenderer.string_rewrite`). The kernel's graph is tinygrad's; only its
  C types change. Every bfloat16 operation already computes in float32
  through `create_non_native_float_pats` and `pm_manual_bf16_cast`, so a
  bfloat16 value is only loaded, stored, selected and bitcast, which act on
  its bits.
- **Reason:** (b). LLVM's x86 backend without AVX512-BF16 aborts on a
  selection between bfloat16 vectors ("Cannot select: v8bf16 =
  X86ISD::BLENDV") and truncates a gated load's merge, or a bitcast of stacked
  lanes, with compiler-rt's `__truncsfbf2`, which rig's host linker finds in no
  library of the process. rune compiles bfloat16 selections and loads on the
  host in `rune jit › classes ›` `a selection of a minimum against its
  constant is exact`, `a selection of a value and a zero is exact`, `a
  segmented contraction gives eager's values` and `experts applied give
  eager's values` (`Nx_quant.apply_experts`), each with bfloat16 among its
  dtypes.
- **Pinned by:** the `Codegen` suite's `a selection between bfloat16 vectors
  takes each lane's arm` and `a gated bfloat16 load takes its alternative
  where its gate is false`, which run on the host and fail on x86 without
  AVX512-BF16 under `__bf16`. The x86 Clang goldens of
  bfloat16 kernels (`bf16_clang`, `clang_*_bf16`). On rune: the four tests
  above, on an x86 host without AVX512-BF16; that each fails without the rule
  is unconfirmed.

## D148. A selection keeps its guard of a conversion

- **tinygrad:** `uop/symbolic.py` `fold_where_closure` (assumes a selection's
  condition true in its true branch and false in the other, unless an `INDEX`
  is involved).
- **tolk:** `Shape.guards_partial` and `Shape.fold_where_closure`.
- **Differs:** the condition is not assumed in the branches where it reaches
  the operand of a conversion of a float to an integer in one of them:
  `where c (cast (where c q 0)) f` stays. A guard of an integer divisor,
  `where c (x // where c d 1) 0`, folds as tinygrad folds it: D355 defines the
  division by any divisor whose bounds reach 0.
- **Reason:** (b). C leaves a conversion of an out-of-range float undefined,
  and Clang's vector selections keep the undefined lane, which LLVM folds into
  any value. rune guards the quadrant of a sine's reduction this way; without
  the guard the compiled sine of `[0; inf]` on x86 read `0x1.c9e8dep+97` for
  `sin 0`.
- **Pinned by:** the `Symbolic` suite's `selections › a guard of a conversion
  to an integer is kept in the branches`. On rune: missing (the old rune's `a
  compiled program › computes eager's values within the stated rounding`,
  whose third example was that sine, went with its rewrite).

## D149. Withdrawn

The Clang renderer wrote no calling convention on Windows. tinygrad no longer
supports Windows: `Cstyle.clang_lang` keeps the empty `abi` and plain
`kernel_typedef` of `CStyleLanguage` (`CStyleLanguage.abi`,
`CStyleLanguage.kernel_typedef`) on every system, as `renderer/cstyle.py`
`ClangRenderer` does; D178 holds Windows' convention.

## D150. The folder's sine, logarithm and powers are correctly rounded

- **tinygrad:** `uop/ops.py` `python_alu`'s `SIN` (`math.sin`), `LOG2`
  (`math.log2`), `EXP2` (`safe_exp2`'s `2 ** x`) and `POW` (`safe_pow`'s
  `pow`): the C library's; `uop/symbolic.py` `symbolic_simple` (`c ** x` as
  `exp2 (x * math.log2 c)`).
- **tolk:** `Ops.sine`, `Ops.py_log2`, `Ops.py_exp2` and `Ops.py_pow`, which
  `python_alu` calls, and the `c ** x` rule of `Shape.symbolic_simple`, which
  takes `log2 c` from `Ops.exec_alu`; `test/gen/generate.py`, whose child
  computes these folds correctly rounded with mpmath.
- **Differs:** `exec_alu` computes each on integers and rounds once, correctly
  rounded on every host. A sine's argument is reduced exactly by pi/2, held to
  1200 bits, and the sine or cosine of the remainder is summed in fixed point
  to 192 bits. A logarithm, a power of two and a power are summed in fixed
  point (`atanh` and `exp` series) with a bound on their error, at 64 bits,
  then 128, 256, ..., until both ends of the bound round to one double; a
  power of two `2^(e y)` and a power that a double or a midpoint between two
  can equal are computed exactly. A NaN operand of the sine gives `Dtype.nan`.
- **Reason:** for the sine, (b). rune's `sin` and `cos` are `SIN` nodes
  (rune's `Lower_real.sin_`), which fold through `exec_alu`, and a fold must
  not depend on the host: on Windows `sin 1e300` folded to `0.979`, of the
  wrong sign. For the logarithm and the powers, (e): a C library's value one
  unit off is the wrong value, and differs between hosts (glibc 2.41's `log2
  0.86`, `2^-0.338`, `5.4 ** 0.8`).
- **Pinned by:** `Tolk.Ops › exec_alu › exec_alu_values.golden`, compared
  exactly: its `SIN` rows (`1e300`, `3.4e38`, pi, floats near multiples of pi
  and floats glibc misrounds) and its float64 `LOG2`, `EXP2` and `POW` rows
  (operands glibc misrounds, logarithms near 1, of subnormals and past the
  greatest double, powers of two that round to a subnormal or to 0, exponents
  near `2^60`, exact powers, one halfway between two doubles:
  `68718952449 ** 1.5`, `262143^3`); `Tolk.Symbolic › tinygrad › tests.golden
  › TestWeakConstFolding.test_float_unaries`, which folds `sin 4.0`; and
  `Tolk.Symbolic › symbolic_simple › powers › c ** x multiplies x by log2 c
  correctly rounded`, at `c = 0.86`.

## D151. An x86 kernel declares that it uses 512-bit vectors

- **tinygrad:** `renderer/cstyle.py` `kernel_typedef` (`void`) and
  `ClangRenderer.__init__`.
- **tolk:** `Cstyle.clang` (on x86, `kernel_typedef` is
  `__attribute__((min_vector_width(512))) void`).
- **Differs:** on x86, every kernel is marked as a function that uses 512-bit
  vectors, so on a processor with AVX-512 512-bit types are legal even where
  it prefers 256-bit vectors. A kernel whose vectors are no wider than 256
  bits compiles to the same instructions, and without AVX-512 the mark
  changes nothing.
- **Reason:** (b). The renderer writes vectors wider than 256 bits (D141), and
  Clang 18 segfaults splitting a negation, absolute value or copysign of more
  than 16 halves for an AVX-512 processor that prefers 256-bit vectors.
  `Rune.jit` writes float16 `Nx.neg` and `Nx.abs` on the bits, the sign bit
  flipped or cleared (`Lower_bits.negated`, `Lower_bits.magnitude`), and LLVM
  folds both back into a negation and an absolute value of halves: on fudge
  the kernel `E_2_16_2` that `Rune.jit` compiles for either over 64 halves,
  vectors of 32, crashes Ubuntu 24.04's Clang 18.1.3 and LLVM's 18.1.8 in X86
  instruction selection for `skylake-avx512`, `icelake-server` and
  `sapphirerapids` without the mark, and compiles with it. Clang 17 and
  Clang 19 and later compile them without the mark. LLVM folds every other
  rendering of the negation back into the same one. tinygrad computes each
  lane apart and writes no such vector.
- **Pinned by:** `Tolk.Compiler_cpu › features › a kernel over more than 16
  halves compiles for AVX-512`, which renders `half_neg`, `half_abs`,
  `half_copysign` and `half_min` (32 lanes each), and `› a sign bit of more
  than 16 halves compiles for AVX-512`, which renders 32 halves' sign bit
  flipped and cleared as rune writes them, both for `skylake-avx512`,
  `icelake-server` and `sapphirerapids`, and failing under `CC` naming Clang
  18 without the mark; the x86 Clang goldens, whose kernels carry it. On rune:
  missing (the old rune's `Rune.jit programs › computes eager's values within
  the stated rounding`, whose examples held the program found, went with its
  rewrite).

## D152. The disk cache holds at most CACHESIZE bytes

- **tinygrad:** `helpers.py` `diskcache_clear`, `diskcache_get`,
  `diskcache_put`: nothing removes a row but `diskcache_clear`.
- **tolk:** `Helpers.Diskcache.mark`, `Helpers.Diskcache.size_file`,
  `Helpers.Diskcache.count`, `Helpers.Diskcache.trim`,
  `Helpers.Diskcache.make_room`, `Helpers.Diskcache.put` and
  `Helpers.Diskcache.clear`; `Setting.cachesize` (5 GiB by default, `0` for
  no bound).
- **Differs:**
  - A put that would take the cache's files past `CACHESIZE` bytes first
    removes entries of every table and version, least recently used by file
    time first, until they would hold four fifths of it with its entry; it
    removes them before it writes, so that on a full disk the write finds
    their space. A put of an entry larger than the bound
    writes nothing. It also removes the writes killed processes left, last written an hour
    or more before, the age at which ccache removes its temporary files, so
    another process's write in progress stays.
  - A hit sets its entry's time through a descriptor shared for deletion
    (`lib/helpers_stubs.c`, `futimens` or `SetFileTime`), and on Windows a put
    renames with POSIX semantics (`tolk_rename`), so a put over an entry a
    read or a mark holds still wins.
  - Processes count each other's puts in the length of one file, to which
    each put appends a byte per KiB.
  - `clear` keeps a write in progress, an entry or size it cannot remove (on
    Windows, one another program holds open without sharing deletion), and
    the tables' directories.
- **Reason:** (b). Entries are keyed by the digest of tolk's sources (D63), so
  each change leaves the previous build's unread, about 0.6 GB for the tests
  of tolk and rune; the shared cache reached 96 GB on one machine, and a full
  disk fails the put of a compiled binary (`Renderer.Compiler.compile_cached`)
  and so the compile.
- **Pinned by:** `Tolk.Helpers › Diskcache › the files of the cache never
  hold more than cachesize`, `› a put past cachesize removes entries used
  least recently first`, `› a put past cachesize removes the writes a killed
  process left`, `› a put past cachesize keeps another process's write in
  progress`, `› a put that cannot write removes entries past cachesize
  first`, `› a put that cannot count the cache's size stores its entry`, `›
  clear leaves the entries it cannot remove`, `› clear keeps the directories
  of the tables`, `› a put replaces an entry that a read holds open`, `› clear
  keeps another process's write in progress`, `› processes putting in turn
  keep the cache within cachesize`, `› a cachesize of 0 keeps every entry` and
  `› a negative cachesize is refused`; `Tolk.Helpers › Diskcache on two
  domains › behaves as a table of entries, whatever the order of the calls`.

## D154. A maximum over a range whose size may be 0 is its identity

- **tinygrad:** `uop/symbolic.py` `reduce_mul_chain` (an integer maximum of a
  product times the factor that ignores its ranges).
- **tolk:** `Symbolic.reduce_mul_chain`.
- **Differs:** an integer maximum of a product by a factor that ignores its
  ranges, over a range whose size may be 0, is its identity there, where
  tinygrad's identity times the factor wraps.
- **Reason:** (c). `Tolk_engine.run` binds a variable to any value of its
  range, its lower bound included, so a size of 0 is a binding the schedule
  must compute; with concrete shapes `Prepare` folds an empty value first.
- **Pinned by:** `Tolk.Codegen › reductions over a variable size`, which runs
  sums and maxima over sizes of at most 1 and 3 on the host at every size, an
  int32 maximum of `2 k` among them.

## D155. Withdrawn

A pad by `-0.` filled its padding with `-0.`; merged into D52, which holds its
rule, its reason and its pins.

## D156. Withdrawn

A NaN constant was written from its bits; merged into D27, which holds its
rule, its reason and its pins.

## D157. A C-style value read once is written into its reader only while its text is short and shallow

- **tinygrad:** `renderer/cstyle.py` `_render` (a value read once is written
  into its reader, `r[u] = l`).
- **tolk:** `Cstyle.max_nesting`, `Cstyle.nesting`, `Cstyle.max_inlined`, and
  `inlined` in `Cstyle.render_uops`, with its call there.
- **Differs:** a value read once is declared as a local variable when its
  expression nests more than 64 brackets, counting `(`, `[` and `{`, or is
  longer than 4096 characters. Two exceptions stay written at any length, as
  in tinygrad: a call of code written into its reader (`Ops.call ~inline`), at
  any depth too, and an address (an index, a shrink, a buffer passed by
  address, a pointer cast), whose brackets the nesting limit bounds. No golden
  changes.
- **Reason:** nesting, (b). Clang and Metal reject a statement past 256
  bracket levels (Clang's `BracketDepth`; HIP's comgr limit is unverified on
  hardware), and rune's float64 `fmod` (rune's `Lower_real.fmod`) nests about
  374: rune's `Lower_arith › exact binary kinds, comparisons, where and fma
  give eager's bits` (float64) and `Lower_real › exact › mod` failed on the
  host. Length, (d): compile time,
  since each reader copies and scans its operand's text; on fudge a host sum
  of 25,000 loads rendered in 15.5 s unbounded against 0.34 s bounded. The
  bound costs against holding long texts aside and writing them out once,
  with 120 more lines of code: `chains/render-sum`'s 8,000 loads allocate 48%
  more words (21.5 M against 14.5 M), and a cold first call of rune's
  `special/betaincinv-grad-f64` 196.7 M words against 195.8 M; unbounded, with
  D468, it allocated 244.9 M. No raven consumer builds a chain of inline
  calls.
- **Pinned by:** nesting, the Cstyle suite: `parentheses › a chain of any
  length nests a bounded number of brackets`, which renders chains of 1 to 300
  steps with each renderer, and `execution on the host › computes a chain too
  deep for one expression`. Length, tolk's bench `chains/render-sum`, on
  fudge, and the Cstyle suite's `parentheses › a sum of any length writes
  statements of bounded length` and `execution on the host › adds a sum too
  long for one expression in order`.

## D158. A dead read of a staged value is 0

- **tinygrad:** `schedule/rangeify.py` `remove_bufferize` (inlines a stage's
  value at a read and leaves each range whose index is `Invalid` in place).
- **tolk:** `Rangeify.remove_bufferize`.
- **Differs:** a read of a stage that would be inlined, at indices of which
  one is `Invalid`, is 0, as a load at an `Invalid` index is. tinygrad inlines
  the stage's value with its own ranges, which no kernel ends, and its
  specification refuses the free range unless a gate folds it away.
- **Reason:** (b). rune's compiled Fold (rune's `Lower_index.fold`) of a
  complex128 `[2; 3; 1]` by `lo [0; 1; 2]`, `interior [0; 1; 0]` and a window
  of size 2 and step 2 on its last axis, whose every term the crop drops,
  raised: rune's `a fold wraps integers and holds its float bound` failed.
- **Pinned by:** the `Rangeify` suite (`test/schedule/rangeify`):
  `get_kernel_graph › dead reads › a staged value read only in padding reads
  0 there`; rune's `Lower_index › folds and assemblies › a fold wraps
  integers and holds its float bound`.

## D159. A captured schedule tells the settings it compiles under

- **tinygrad:** `engine/jit.py` `jit_lower` (reads `JITBEAM`, or else `BEAM`,
  and passes the width to `compile_linear`, binding no setting);
  `codegen/opt/search.py` `beam_search`, whose cache keys on no setting.
- **tolk:** `Jit.width`, `Jit.bindings`, `Jit.shaping`, and `Jit.jit_lower`,
  which compiles under `bindings`; `Setting.shaping_under`.
- **Differs:** `jit_lower ~beam` compiles with `BEAM` at its width and
  `JITBEAM`, read by then, unset, so the caches of what it compiles, which key
  on `Setting.shaping`, key on the width alone. `Jit.shaping ~beam` is
  `Setting.shaping` under those bindings, read without binding them
  (`Setting.shaping_under`): two lowerings whose `shaping` is equal compile
  alike, whatever `BEAM` and `JITBEAM` were. tinygrad has no such value.
- **Reason:** (b). rune's `Rune.jit ~beam` (rune's `Compile.shaping`) keys a
  compiled function's variants on the settings that shape its programs; with
  an explicit width, neither `BEAM` nor `JITBEAM` shapes them, and rune read
  the lowering's rule by repeating it.
- **Pinned by:** the `Jit JITBEAM` suite (`test/engine/jit`): `JITBEAM unset ›
  a width given and BEAM's alike compile alike` and `JITBEAM=3 › a width given
  and JITBEAM's alike compile alike`; and the `Tolk.Setting` suite: `shaping ›
  under bindings is shaping in their context, and binds nothing`.

## D160. A read no order keeps reads a copy of its state

- **tinygrad:** `schedule/__init__.py` `create_schedule` (a write-after-read
  edge to every other kernel writing over a state it reads; raises "cycle
  detected in assign graph", and "sibling AFTERs detected in assign graph"
  for two AFTERs over the state each names first); `schedule/rangeify.py`
  `check_buf_states` (raises "cycle detected while indexing" where a kernel
  reads a storage in two states); `test/runtime/test_assign.py::
  TestAssignOrdering::test_war_reader_already_depends_on_write` states the
  raise. A call given one buffer at two arguments,
  writing one, is ordered against no other kernel and races with itself.
- **tolk:** `Schedule.schedule`, `Schedule.cuts`, `Schedule.copy_call`,
  `Schedule.copy_before_overwrites` and `Schedule.aliases`;
  `Prepare.sequence`; `Rangeify.stand_in`; `Shape.written_params` and
  `Shape.custom_kernel`.
- **Differs:** one relation orders every kernel of the graph: a call of a
  kernel, of a copy, of a precompiled body or a custom kernel, and a loop of a
  call.
  - A kernel reads the states its arguments name, a join being the state
    under it (D202), and writes the states the Afters that list it are over,
    but for a kernel already on an earlier state of that storage, which orders
    the state and writes nothing over it, as tinygrad's (3bdafaedf).
  - It runs after the kernels that write the states it reads, and after the
    kernels of the write that made the state it writes over, so a reader of a
    state runs after every write under it, as tinygrad orders a state after
    the state under it (413ecebe6). It runs before every
    other kernel of each write over a state it reads (D330), whichever
    elements each touches.
  - Where these orders form a cycle, one read on it reads a copy of its whole
    state, which a kernel of its own makes into call-local storage before the
    overwrite, and the schedule is made again with the copy. Kahn's order
    finds the cycle where no kernel is ready, walking back through the kernels
    not run; the read cut is one whose state is copied already, else the one
    of fewest bytes, of storage that can be copied first. The walk is greedy,
    since fewest copies is NP-hard. A cycle in the graph made again raises as
    an internal error. A kernel graph whose kernels form no cycle gets no copy;
    grouping stores into one kernel can make one, so the promise is of the
    kernel graph `create_schedule` is given. Swaps, rotations and crossed
    reads compute, with a copy for each cycle. A kernel reading one storage in
    two states takes a parameter for each, and lies on a cycle, so no uncut
    kernel binds one buffer at two parameters.
  - A call given one state at an argument its body writes and at one it does
    not reads, as `memmove` reads its source, a copy made before it at the
    argument it only reads (`Shape.written_params`, at any depth through the
    calls the body makes); in a loop, the copy is made as each trip starts. A
    custom kernel orders after it only the arguments its body writes.
  - A write's prior state is the state before it on its storage's chain
    (`Prepare.sequence`, D473): beside `a = x.after(x.store(1))`,
    `x.after(a.store(2))` writes over `a` (D474). Two writes over one state,
    which no storage holds, and a call whose body does not show its writes given one state twice,
    raise `Invalid_argument` (D474).
  - A partial overwrite of elements a kernel does not read copies too, at a
    kernel and the state's size in storage.
- **Reason:** (e). tinygrad raises on valid graphs (a swap, a stale read
  beside its overwrite) and computes wrong values where it orders silently (a
  call given one buffer twice, the join cases of D202); rune's jit promises
  eager's values, and a program refused on a set without kernels raises. No
  raven suite schedules a cycle; jera's compiled gradient
  of an adaptive solve makes the one copy they reach (`Ode › solve
  derivatives › compiled grad equals eager grad`). The relation
  costs allocation in every schedule: on fudge `matmul/schedule` allocates
  738 words against 613 without it, and `gpt_oss_decode/schedule` about 13.1
  thousand against 11.6.
- **Pinned by:** the `Tolk.Schedule` suite: `create_schedule › laws › a
  schedule copies a state iff its kernels form a cycle` and
  `create_linear_with_vars › laws › a schedule writes what its tensors
  compute`; `create_linear_with_vars › orders and copies`, `› writes` and `›
  calls given one state twice`; `create_schedule › rules › kernels that
  overwrite each other's inputs read one copy`, `› a kernel that waits for the
  overwrite of a state it reads reads it` and the rules beside it, for a copy
  two kernels share, a kernel with two copies, a copy of a state a rebuilt
  kernel writes and the copy's phase; the programs `read_overwritten_through`,
  `read_beside_overwrite`, `swap_through_values`, `call_given_one_state_twice`,
  `after_older_write`, `readers_after_older_write` and
  `after_older_and_new_write`, whose recorded schedules and linears and whose
  values (`create_linear_with_vars › values`); `write_over_written`'s linear
  and values; the kernel graph of `after_part_of_older_write`, whose schedule
  and linear differ by D184; and the `Tolk.Rangeify` suite's `get_kernel_graph
  › buffer states`.

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

- **tinygrad:** `schedule/rangeify.py` `remove_bufferize` (inlines a stored
  value that reads at most three buffers and no reduction reading a buffer, at
  every position and trip it is read at) and `get_kernel_graph`;
  `schedule/indexing.py` `run_rangeify` (stores a value its consumers index
  differently, or one a reduction or an elementwise operation reads past a
  broadcast: the stores `remove_bufferize` weighs).
- **tolk:** `Rangeify.extent`, `Rangeify.read_ranges` and `Rangeify.read_copies`
  (the copies a read makes of a stage's value), with `Rangeify.digit_of` (a
  range read through floor divisions and remainders); `Rangeify.collapses`,
  `Rangeify.selected` and `Rangeify.collapsed` (a sum that selects one trip,
  asked only where the answer changes a stage's decision), and
  `Rangeify.storage` (a stage of no storage read in such a sum), through
  `Simplify.load_collapse`, which `Simplify.pm_load_collapse` applies;
  `Rangeify.reads`, `Rangeify.stage_reads`, `Rangeify.reduced`,
  `Rangeify.stage_copies` and `Rangeify.stage_bounds` (the copies of each
  stage); `Rangeify.inline_budget` and `Rangeify.copied`; `Rangeify.inlined`;
  `copies` in `Rangeify.remove_bufferize` and `Rangeify.does_work`;
  `Rangeify.destination`, which `Rangeify.pm_remove_bufferize` applies to a
  store's destination; `Rangeify.pm_cleanups` and `Rangeify.get_kernel_graph`,
  which counts the reads before the cleanups; across kernels,
  `Rangeify.store_shared`, `Rangeify.candidates`, `Rangeify.kept`,
  `Rangeify.computed_nodes`, `Rangeify.union` and `Rangeify.is_kernel`.
- **Differs:**
  - **The budget.** A stage whose reads make `c` copies of its value stays
    stored when `n * (c - 1) > 360`, where `n` counts the nodes of its value,
    its producers already inlined, that its ranges reach: those an index copies,
    since the substitution leaves the others shared. `c` sums over the stage's
    reads, so several windowed reads of one stage add up. A read makes one copy
    at each distinct index; in a reduction it makes one per trip `w` of the
    reduction ranges its index runs in, those a reduction of the graph closes,
    at most its trips per element of the stage, `t / e`, since a trip that
    computes no element again computes nothing that storing saves. A running
    sum's windows of `w` count `w`; a sum that reads each element once counts 1;
    a read whose index runs in no reduction range, as a broadcast through an
    expand, a reshape or padding, counts 1. Trips in padding count: inlined
    there, the value's loads are gated and its arithmetic runs. `e` and `t`
    leave out the axes the value does not vary along, and take a symbolic size
    at its largest. Inlined copies compound (`k` values each read at two
    positions are `2^k` copies of the first), and tinygrad has the same
    exponential behaviour and computes a value read in a window again at every
    trip. Since every read counts at least once, the code inlining adds is at
    most the work, 360 nodes.
  - **What is weighed.** A view of storage, a read whose index computes only
    positions, is inlined at any count, as tinygrad inlines it: stored, it would
    copy storage its readers load in place. Positions are weak integers and
    their comparisons, on no device. An elementwise operation, a reduction or a
    call on a device, a position loaded from storage, or a value of another type
    computes; so does any value that is not a read, a mask of positions
    included, and each is weighed. A stage a store writes through is inlined
    whatever it costs, past the budget and past tinygrad's own checks too (more
    than three buffers, a reduction reading one). A user's materialisation is
    kept, and written as asked.
  - **Dropped digits.** An output range of constant size that each component of
    the index reading it reads through one chain of floor divisions and
    remainders by positive constants, each dividing the size of what it is
    applied to, in components that read no other range, repeats the value across
    the outputs that differ only in the digits the chains drop: those outputs
    read it at one index. That reuse, the extent of such ranges over the extent
    of the stage axes their components index, divides `t / e`; the codegen
    splits a range at a remainder by a constant dividing it
    (`Simplify.pm_split_ranges`) and shares a quotient's repeats on a tensor
    core's `M` or a streamed operand's reuse, so a tile of those outputs shares
    the value as it shares a broadcast through a range the index does not name,
    which D77 and D106 weigh. A range that another component also reads, or that
    a sum selecting one trip runs in, stays counted; so do a window (`r // 16 +
    k`), a gather (a component reading a buffer), a framed read `(r // L) * h +
    r % L` and a digit sum `(r % m) + (r // m)`, which read a range twice in one
    component, a constant that does not divide its operand (`r // 16` of 50
    rows), and a range of symbolic size, which is never split. Reduction ranges
    keep their count: the MXFP4 scales' `k // 32` counts 32. Rangeify counts
    without the target, and a target that shares fewer lanes computes the value
    again more often than counted. On a GPU the codegen owes that sharing, and
    states it: D230's bound of `ceil(R / 16)` decodes a value. On the host it
    shares a decode over as many rows as its registers hold (D92).
  - **Selections.** A sum over one range that selects one trip, where an index
    equals the range, counts that range as the ranges its selecting conditions
    read besides it: the kernel computes the value at that index alone
    (`Simplify.load_collapse`), once for each point of those, so a routed row
    decodes only its own expert's weights. A factor over another range inside
    the sum, `where (ids (j) = e) (v (e, d) * y (m)) 0` summed over `e`,
    broadcasts the selected value along `m`, which counts `m` neither; a sum
    with no such condition counts every range it runs in. The test is the
    collapse's own, run on the sum as its kernel holds it: reads of stages whose
    values read no storage are seen as those values, simplified as the cleanups
    simplify them. Counting the work after the substitution follows Mullapudi et
    al. 2016, §4.3.
  - **Positions in a selecting sum.** A read of a stage whose value reads no
    storage, in a sum the test finds selecting one trip, counts that sum's own
    trips, as a read in a sum that selects none: stored, it would leave the sum
    comparing a load with its range, which no kernel computes at one trip. Its
    reads elsewhere count as before, so such a value past the budget in other
    reads stays stored, and the sum then runs every trip.
  - **Across kernels.** A value of no range, which `run_rangeify` never stages,
    is computed by each kernel that reads it. Once the cleanups have run (no
    later pass removes a stage), a computed value of no range that `k` kernels
    compute is staged in one element and read by each when `(k - 1) n > 360`,
    `n` its computed nodes. One walk, kernels first, gives each node the kernels
    its consumers are computed in, a staged value being a kernel of its own; a
    value is weighed where some consumer is computed in fewer kernels than it,
    so the values below it move with it. A weak value is never staged, since it
    takes its readers' widths and has no storage of its own; below it, as below
    a load several kernels share or a value with ranges, the consumers' fewest
    kernels pass down, and the value under it is weighed. Its `n` is counted
    again down to the values kept below it, consumers first, and a value within
    the budget is computed in its kernels after all, the values below it
    counting those kernels too: one decision for all its consumers, as Mullapudi
    et al. 2016, §4.3. The walk is linear: a node's kernels are counted exactly
    to 362, and each weighing stops at the `360 / (k - 1) + 1` computed nodes
    that decide it. A value with ranges is computed in several kernels only
    through a range they share; it is left as it is, since storing it would hold
    its extent alive between its kernels (RFC 0005's bound).
  - **The bound.** A program of `N` nodes makes kernels of at most `N` nodes
    plus 360 for each stage inlined and for each value of no range left in
    several kernels, the reads the cleanups make aside, and nodes with ranges
    that several kernels share, which no rule bounds, aside too. D77 and D106
    weigh the other recomputation inlining makes, across ranges a read's index
    does not run in, so they stay separate rules.
  - **Counting.** The reads are counted once, in a walk of the graph the
    cleanups start from, at each axis of a stage's ranges; a stage's count is
    the least over its axes, since stages over one range share that axis.
    Indexing gives each stage a later pass may remove ranges new to it, except
    on an axis whose size is a range, which that range indexes: a stage with any
    other range has one of its own, where its reads are counted alone. A stage
    over such axes alone, as one value of each of a device's lanes, shares them
    and pools its reads. A read the cleanups make is not counted, and a read
    through a stack is not one: every stage has a device by then, so no cleanup
    turns it into a read of the stage.
- **Reason:** (e), `Rangeify.destination`: kept, a stage a store writes through
  takes the stores, and the storage holds what it held; 47 stores of one state
  into the rows of a `[2; 47; 9]` buffer read its view at 47 indices, 368 nodes
  in all, and wrote a copy of it. (d) the rest. rune's compiled
  `Nx.associative_scan`, whose prefix circuit reads each level at two or three
  positions: jera's `scan/tridiagonal-100` compiles cold in 1.25 CPU-s and 34.4
  M words against 2.08 s and 100.3 M with no budget. norn's nested sampling, a
  running sum over 100 × 1213 negated logs of uniform draws, took 0.87 s a call
  inlined against 0.18 s stored; running sums of `2x` in windows of 400, before
  D194, took 0.59 ms stored against 1.24 ms inlined, and in windows of 16 `2x`
  is inlined. 360 is placed by `GptOss/decode-step-Q8_0-cache-1024`, whose MXFP4
  block scale, 341 nodes past the first copy, runs 1.011 times as long stored as
  inlined, apart (so the bound is at least 341), while jera's
  `root-newton-kepler-1k/compiled`, 371 nodes, overlaps in one round of high
  load, so it does not decide, and its median asks for a bound under 371;
  wall-clock baselines of these rows on fudge are pending the scheduled
  re-record. Dropped digits: counted as copies, they stored every decoded weight
  a quantised product of many rows reads, and `GptOss/prefill-512` took
  1.18-1.19 s against 0.44-0.45 s. The count is the same on CUDA and AMD;
  inlined bfloat16 rows run slower on nonnormal's R9700 under the heuristic's
  kernels (64 rows 8.2-8.8 ms against 5.0-5.1 stored), where a search over
  tolk's optimisations finds the inlined kernels faster on both, so the count
  stands and the codegen owes its sharing (D230); float32 rows, which no core
  takes on CUDA, streamed (D230) take 4.98-6.68 ms at 64 rows, under the stored
  decode's 9.6; D89's order by multiply-adds per byte takes the bfloat16
  products of an MoE layer at 512 tokens to the tensor cores (4.66-4.68 ms a
  layer). Positions in a selecting sum: a one-hot gather's `arange`, counted at
  the sum's outer ranges, was stored past `k = 182`, and
  `Nx_quant.apply_experts` of gpt-oss-20b's gate and up projection at 512 tokens
  on fudge took 1.49 s, against 26.4-27.6 ms now. Across kernels, (d) too:
  ymir's `aperture-104/compiled-grad` computes a value of no range of about
  3,000 nodes in 18 kernels; stored where its copies pass the budget, its cold
  first call takes 22.7-25.9 CPU-s against 31.8-36.9 s, with the same bits; a
  chain of scalars each read by a kernel of its own, as rune's chain of outputs
  reads it, was quadratic in its length and is linear. The row's wall-clock
  baseline is pending the scheduled re-record.
- **Pinned by:** the Rangeify suite's rows of `kernel_counts.tolk` and their
  kernel goldens: across kernels, `scalar_read_by_two` (computed in each),
  `long_scalar_read_by_two` (stored) and `scalar_chain` (one stored);
  `read_at_offsets` (stored), `windowed_chain` (stored), `windowed_pair`
  (stored), `windowed_scaled` (inlined), `windowed_scaled_long` (inlined into
  its windows of 16), `read_by_two` (inlined), `long_read_by_two` (stored),
  `routed_product_2` (inlined), `routed_product_64` (stored) and
  `routed_product_wide` (the gather's positions inlined), `selected_scaled`
  (stored by D106, which an enclosing sum's outputs decide),
  `blocked_product_16` (read at `r // 16`, inlined) and
  `dense_decoded_64` (read at `r % 4`, inlined), `tiled_chain` (inlined) and
  `padded_pair` (inlined); `cost › levels each read twice make kernels that stop
  growing with depth`; the `copies` group: `a value of 1200 nodes read at one
  index is inlined`, `› a value read twice whose second copy is the budget is
  inlined`, `› a value read twice whose second copy passes the budget is
  stored`, `› a value of 600 nodes outside its ranges read twice is inlined` and
  `› reads of a stage sharing a range are not the other's`; the `recomputation`
  group: `a value of 1200 nodes reduced once an element is inlined`, `› a cheap
  value reduced in windows is inlined`, `› a value reduced in windows whose work
  is under the budget is inlined`, `› a value reduced in windows whose work
  passes the budget is stored`, `› a value of no storage reduced in windows past
  the budget is stored`, `› two windowed reads of a stage add up`, `› a value of
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
  group: `a dropped digit's reuse is no copy` (every case), `› a read of merged
  outputs decides as one of them split` and `› a digit a selection runs in stays
  counted`; `selections › a read a sum selects decides as one at the selected
  index`; `own ranges › a removable stage has a range no other stage has`;
  rune's `Lower_reduce › work › a Rows contraction's work is one slice's over
  256 contracted` and `› a scaled float4 Rows contraction's work is one slice's
  over 256`; rune's `rune jit › classes › an associative scan over a tuple gives
  eager's bits` and `› a value stored or computed again in its readers gives
  eager's bits`; on fudge, the `scan/tridiagonal-100` row of rune's compile
  bench, the `Jit/cumsum-neg-log-uniform-8x1024`,
  `Jit/experts-mxfp4-16x4of32-256` and `Cuda/sin-read-by-two-float64-1000` rows
  of its bench, jera's `root-newton-kepler-1k/compiled` and
  `root-bracket-kepler-1k/compiled`, gpt-oss's `GptOss/prefill-512` and
  `GptOss/experts-at-prefill-512`, and kaun's
  `Quant/cuda/mxfp4-product-of-16384-*-tokens` and
  `Quant/host/mxfp4-product-of-1024-*-tokens`. Across kernels, the Rangeify
  suite's `get_kernel_graph › values that several kernels compute` (every test),
  and ymir's `aperture-104/compiled-grad` row.

## D164. Withdrawn

The read of an accumulator after its loop waited on the initial store too, since
`sym` folded a store of a loaded accumulator to a bare `NOOP`. tinygrad's
`uop/symbolic.py` `sym` now keeps the accumulator's index (`NOOP.after(index)`,
`NOOP.after(*x.src)`), and `codegen/__init__.py` `lower_reduce` orders the read
after the end alone, as `Codegen.lower_reduce` does. The `Codegen` suite's `an
accumulator no lane adds to › is stored before it is read` and `› sums to zero`,
and rune's `Lower_index › folds and assemblies › a fold wraps integers and holds
its float bound`, hold.

## D165. A weak maximum's identity is its accumulator's least value

- **tinygrad:** `uop/ops.py` `identity_element`, whose `Ops.MAX` identity is
  `dt.min`: for `weakint`, of bitsize 800 (`dtype.py` `DTypes.weakint`),
  `-2**799`.
- **tolk:** `Ops.identity_element`.
- **Differs:** a maximum over a weak type has as identity the least value of
  `Dtype.strong` of the type, the width its accumulator is allocated at. A weak
  maximum arises where a pad's bare fill is all that remains of a committed
  maximum's operand. In tinygrad such a maximum, gated by an Invalid, takes
  `-2**799`, which `lift_reduce_gate` (`uop/symbolic.py` `lift_reduce_gate`)
  selects where the gate fails, and committing the selection's width
  (`cast_weak_srcs`, `commit_int`) raises "does not fit any int".
- **Reason:** (b). rune's compiled Fold (rune's `Lower_index.fold`) of a float64
  `[2; 2; 2]` by `lo [2; 2; 0]`, `hi [0; -1; -1]`, `interior [1; 0; 0]` and
  three windows maximises the uint8 marks of each element's terms; where no term
  reaches an element the mark is the pad's weak `0`, and compiling raised:
  rune's `a fold wraps integers and holds its float bound` failed.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `a maximum of
  weak integers › is the greatest value it reduces`; `Tolk.Ops › data types ›
  identities.golden`; rune's `Lower_index › folds and assemblies › a fold wraps
  integers and holds its float bound`.

## D166. A minimum of a minimum against one value is one selection

- **tinygrad:** `codegen/decomp/op.py` `get_late_rewrite_patterns`, which has no
  rule for a nest of selections.
- **tolk:** `Decomp_op.extreme_of`, `Decomp_op.nested_extreme`,
  `Decomp_op.unwrapped`, `Decomp_op.holds`, `Decomp_op.keeps` and
  `Decomp_op.late_patterns`.
- **Differs:** a selection of a value and `c` by their less-than comparison, a
  minimum or a maximum against `c`, whose value is such a selection of the same
  kind against the same `c`, is rewritten to one selection: the inner one where
  it picks `c` where its comparison is false, at a tie or a NaN, or where the
  two pick alike, and otherwise `where (v < c) v c` or `where (c < v) v c` of
  the inner one's other operand `v`, cast to the outer one's type. Both are what
  the nest computes at every `v`, NaN and both zeros included. Values are
  matched through the casts that keep them (`Decomp_op.keeps`): a widening to a
  float type that holds the operand's, and a narrowing of a value its type holds
  (`Decomp_op.holds`), a constant, a selection of such values at any depth, or
  such a value widened. LLVM's InstCombine narrows a `float32` selection of
  widened `float16` values to a `float16` selection, which keeps every value,
  then folds the plain nest, and rune computes a `float16` minimum as a
  `float32` selection of widened operands, narrowed back, so a nest of them
  reaches that fold.
- **Reason:** (b): LLVM's InstCombine (`foldSPFofSPF`) folds a minimum of a
  minimum, or a maximum of a maximum, against one constant into the inner
  selection, though the two pick apart at a tie or a NaN: with `m = (0.0 < v) ?
  0.0 : v`, `(m < 0.0) ? m : 0.0` becomes `m`, `-0.0` at `v = -0.0` where the
  selection gives `0.0`, and with `1.0` for `0.0` NaN at a NaN where it gives
  `1.0`. It runs on every target from `-O1` in clang 19, at `-O3` for amdgcn,
  which HIP's comgr runs, and in NVRTC 13.4, and LLVM's main of 2026-10 keeps
  it; tinygrad on x86_64 computes these `-0.0` and NaN. Under `Rune.jit`,
  `Nx.where (Nx.less m c) m c` of `m = Nx.minimum c x` gave `-0.` where eager
  gives `0.` and NaN where eager gives `c`, and a reciprocal of it `-inf` for
  `inf`: rune's `a step over donated leaves keeps the values` failed. One
  selection keeps its lowering to one instruction: hiding the constant from LLVM
  in every comparison instead cost a maximum before a 64 × 64 matmul 1.10 times
  its time on fudge. LLVM can still form such a nest itself, as when it
  vectorises scalar minima of a vector minimum's lanes; no tolk kernel is known
  to.
- **Pinned by:** the `Decomp_op` suite (`test/codegen/decomp/decomp_op`):
  `late_patterns › late_nests_cmplt.golden`, `› a nest of extremes against one
  value keeps its values › *` and `› a nest of 16-bit extremes through float32
  keeps its values › *`, which also holds the rule back where a narrowing
  rounds; rune's `rune jit › classes › a selection of a minimum against its
  constant is exact`, at `float16` and `bfloat16` too.

## D167. Clang's arguments: LLVM 18, 19 or 20 and Apple's Clang 17 for x86_64 without its straight-line vectorizer, a long source without its loop vectorizer, arm64 without the interleaved load combine, every kernel without loop idiom recognition

- **tinygrad:** `runtime/support/compiler_cpu.py` `ClangCompiler.compile`
  (Clang's arguments at `-O2`, which runs LLVM's SLP and loop vectorizers).
- **tolk:** `Compiler_cpu.stated_release` (the release a Clang's `-###`
  statement names, LLVM's or Apple's), `Compiler_cpu.long_source`,
  `Compiler_cpu.compile_args`, which adds `-fno-slp-vectorize` when it is LLVM
  18, 19 or 20, or Apple's 17 for x86_64, `-mllvm
  -disable-interleaved-load-combine` for arm64 and `-mllvm
  -disable-loop-idiom-all` for every Clang,
  `Compiler_cpu.without_loop_vectorizer`, and `Compiler_cpu.clang` (its `table`,
  which names both commands and the length); `Compiler_amd.hip`, whose code
  generation's options end with `-mllvm -disable-loop-idiom-all`.
- **Differs:** a Clang that states LLVM 18, 19 or 20 compiles every kernel with
  `-fno-slp-vectorize`, after `-O2`, which would turn the vectorizer back on,
  and so does Apple's Clang 17 for x86_64. Every Clang compiles a source of more
  than 400 000 bytes without the loop vectorizer: its compiler proper runs
  without `-vectorize-loops`, as the driver runs it under `-fno-vectorize`.
  Every Clang compiles for arm64 without LLVM's interleaved load combine. Every
  Clang compiles without LLVM's loop idiom recognition (`-mllvm
  -disable-loop-idiom-all`, which every LLVM release from 12 defines), and so
  does comgr's code generation for AMD GPUs. Otherwise every Clang, Apple's
  included, compiles with the arguments tinygrad passes. The table of objects
  names the command for either length, and the length (D113).
- **Reason:**
  - (b), the straight-line vectorizer: rune's `rune jit › classes › a uint4 sort
    along the last axis of every other row and column, flipped, gives eager's
    codes` failed on CI runners with AVX-512 under Clang 18.1.3 ("SmallVector
    unable to grow"): for `x86-64-v4`, `skylake-avx512` or `znver4` the
    vectorizer packs the clamped indices of kernel `r_3_2` and writes `sext <4 x
    i32> ... to <4 x i16>`, invalid IR, on which X86 instruction selection
    crashes. Clang 19 has the same defect without the crash: for `x86-64-v3`,
    `x86-64-v4`, `arrowlake` or `znver4` (not `x86-64-v2` or `apple-m1`) it
    writes `trunc <4 x i8> ... to <4 x i32>` in the binary search `E_16_2_4` and
    compiles garbage rows, so rune's `Lower_reduce` test `rows in blocks give
    eager's integers and hold their bound` failed on fudge. Apple's Clangs state
    their own release, not LLVM's: Apple clang 17 writes the same for `E_16_2_4`
    for x86_64, on an Intel Mac or for another machine's x86-64 host, and
    compiles both kernels for arm64, where the vectorizer stays on: without it
    Apple Clang 17 adds 9.5% instructions over 466 kernels. LLVM 20, (e): its
    straight-line vectorizer narrows a signed lane that it converts to float as
    unsigned to too few bits, which computes wrong values, so tinygrad's
    arguments miscompile such a source: llvm/llvm-project#129244, a regression
    of LLVM 19, fixed on main by e1e20c07e4 (2025-02-28), which 21.1.0 holds
    and release/20.x never took, so every LLVM 20 through 20.1.8 has it. On
    fudge the issue's reproducer returns 3 under Clang 19.1.7 at `-O2` and 0
    with `-fno-slp-vectorize`, and under Clang 18.1.3 and LLVM 23.1.5. Its case
    here was an attention kernel under ALiBi that converted each int64's 32-bit
    halves, which for `x86-64-v3` LLVM 19 and 20 narrowed to `uitofp <2 x
    i8>`. rune now casts a 64-bit integer to float32 with one `Ops.cast`
    (rune's `Lower_arith.to_float32`), and rune's attention suite passes under
    Clang 19.1.7 with the vectorizer on, so no raven kernel is known to meet
    the defect; the gate holds a cited miscompile of a release Clang that
    distributions still ship. Rewriting the source only moves what the vectorizer
    meets; disabling it for every Clang would change 44% of a sample of
    objects, so it stays on wherever it is correct. With the IR verifier on
    (D275), invalid IR from it is a compile error naming the kernel.
  - (d), the loop vectorizer: rune's test_special timed out at 60 s on Ubuntu CI
    compiling float64 `betainccinv`'s gradients, each one scalar loop kernel,
    `E_1021`, of 1.3 MB of C, whose loop vectorizer time grows faster than the
    body (about n^2.4): on fudge Clang 19.1.7 took 19.7 s on it with the
    vectorizer and 5.7 s without, at the cost of replay, 1.5-5.5 ms with it
    against 4.5-22 ms without, as D141 found 204 ms against 475 ms for
    `log_betainc`'s gradient. Of the kernels tolk's bench, rune's compile bench
    and rune's suites but test_special compile, the longest is 136 KB, and
    test_special's gradients past 400 KB took Clang 4.1-29.4 s, against 0.7-1.4
    s for those of 158-279 KB; the one other past 400 000 bytes,
    `special/betainc-chain-f32` (530 KB), has no loop, and its object is the
    same byte for byte. A `#pragma clang loop vectorize(disable)
    interleave(disable)` saves the same, but the renderer would write it before
    each loop of a source tinygrad writes without it, and the goldens with it,
    while the flag leaves every source as it is and the table keys the objects.
  - (b), the interleaved load combine: rune's `Lower_reduce › work › a
    Contracting contraction of 8 groups works in blocks` timed out at 60 s on
    macOS CI (Apple Clang 17, LLVM 19) after D240's upcasts: its kernel
    `r_23_4_16_4_16`, vectors of 1024 lanes put together by shuffles of 32
    loaded rows, took Clang 19.1.7 293 s for `apple-m1`, 274 s of them in the
    AArch64 `InterleavedLoadCombinePass`, and 8 s without it. Of the 1241
    distinct kernels the rest of that suite compiles for `apple-m1`, the objects
    are the same byte for byte without the pass.
  - (b), loop idiom recognition: rune's `Nx.eigh` under `Rune.jit` of a float64
    matrix of 24 rows took 8.5 s to compile under Clang 23, and of 32 rows
    (SOFO's subspace solve) did not finish in minutes, against 0.3 s under Clang
    19. From LLVM 22 the pass's CRC recogniser (`HashRecognize`) walks the
    operands of the running selection of `Lower_linalg.exponent`'s loop, for an
    innermost loop of one block whose constant trip count is a multiple of 8,
    without marking those it has seen, 2^31 paths. LLVM fixed the walk on main
    after the 23 branch (llvm/llvm-project#211711, #211887); `release/22.x` and
    `release/23.x` do not have it. The pass has nothing to do in a kernel: under
    `-ffreestanding` Clang marks its functions `"no-builtins"`, so LLVM offers
    it no `memset` or `memcpy`, its other idioms need a loop that exits on its
    data, which no kernel producer writes today, and no lowering renders a CRC;
    the objects of rune's linear algebra and suites are the same byte for byte
    without it under Clang 18, 19 and LLVM 23 for x86_64, and Clang 19 for arm64
    and riscv64. Every Clang compiles without it, Apple's included, since
    Apple's release is not LLVM's (Apple clang 17 is based mainly on LLVM 19 and
    Apple clang 21 on LLVM 21, and Swift's `main` builds on swiftlang's
    `stable/23.x`, which has the walk), and the CRC recogniser's own option is
    an error before LLVM 22. A gate to LLVM 22 and 23 would let a future
    superlinear addition to the pass show silently, as a compile that never
    ends, where a rename of the blanket option fails every host compile with
    Clang's message. The item is revisited when a kernel loop exits on its data
    (`Cstyle` renders a void range as `for (;;)`), and goes when tolk supports
    no Clang of LLVM 22 or 23 and Apple's bases are past 23.
  - comgr runs the pass too, under the same reasoning. ROCm 7.0.0's comgr (AMD's
    LLVM 20.0.0git) has no CRC recogniser; its pass turns a loop counting a
    word's bits into `s_bcnt1`, which no kernel producer writes, and the 1384
    code objects rune's GPU suite and `Rune.jit` of `Nx.eigh` compile for
    gfx1201 are the same byte for byte without it. A ROCm built on LLVM 22 or 23
    would carry the walk, so comgr compiles without the pass, from every
    release.
- **Pinned by:** `Tolk.Compiler_cpu › features › four clamped gathers compile
  for AVX-512 on` (`x86_64,x86-64-v4`, `x86_64,skylake-avx512`,
  `x86_64,znver4`), over `r_3_2`'s indices, which fails under Clang 18 without
  the flag; `Tolk.Compiler_cpu › features › a binary search of four lanes
  compiles for AVX2 and AVX-512 on` (`x86_64,x86-64-v3`, `x86_64,x86-64-v4`,
  `x86_64,znver4`), over `E_16_2_4` reduced, which fails under Clang 19 and
  Apple's Clang 17 without the flag; rune's `rune jit › classes › a uint4 sort
  along the last axis of every other row and column, flipped, gives eager's
  codes`, under Clang 18 on an x86 host with AVX-512; rune's `Lower_reduce ›
  cases › rows of one group in a block at each batch index give eager's values`,
  under Clang 19 on an x86 host with AVX2. LLVM 20: missing (no raven kernel
  meets the defect, and fudge has no Clang 20). The loop vectorizer:
  `Tolk.Compiler_cpu › features › a loop over a count
  it takes is packed into vectors iff its source is at most 400000 bytes long:`
  (rows 400000 and 400001 among them); rune's compile bench
  `special/betainccinv-grad-f32` (914 KB), whose first call takes 3.88 s on
  fudge against 6.57 s with the loop vectorizer. The interleaved load combine:
  `Tolk.Compiler_cpu › compile time › an arm64 kernel of 1024 lanes gathered
  from 32 rows compiles within a minute`, which times out at 60 s on fudge with
  the pass; rune's `Lower_reduce` test above, on macOS CI. Loop idiom
  recognition: `Tolk.Compiler_cpu › features › a loop that exits on its data
  stays a loop: idioms are off`, which Clang 19 turns into `popcnt` with the
  pass; `Tolk.Compiler_cpu › compile time › a host maximum of 32 float64 lanes
  in a loop of 32 trips compiles within a minute for` (`x86_64,x86-64-v3`,
  `arm64,apple-m1`), over `r_32_32`'s source; rune's `Lower_linalg › eigh ›
  compiled for 32 rows of float64 gives eager's eigenvalues`; rune's compile
  bench `linalg/eigh-f64-32x32`. These three fail without the fix only under a
  Clang of LLVM 22 or 23. comgr: `Tolk.Compiler_amd › comgr › a loop that exits
  on its data stays a loop: idioms are off`, which ROCm 7.0.0's comgr turns into
  `s_bcnt1` without the option.

## D168. Withdrawn

The renderer listed a program's callees by a scan of its instructions, in place
of tinygrad's toposort of the program. On fudge the gain was 2% of render words
and 0.4% of compile instructions, which did not pay for the scan.

## D169. A scheduler reads its kernel's axes once, and keeps a flat kernel

- **tinygrad:** `codegen/opt/postrange.py` `Scheduler.__init__` (rewrites its
  kernel by `pm_flatten_range`), `Scheduler.rngs` (and `full_shape`, properties
  computed at each access), `copy` (makes a scheduler of the kernel, rewriting
  it again), `axes_of` (computes `reduce_ranges` and `rngs`, itself computing
  `reduce_ranges`) and `reduceops`.
- **tolk:** `Postrange.Scheduler.view` (`view_of`), which `slice`, `rngs`,
  `full_shape`, `reduce_ranges` and `reduceops` read; `Postrange.Scheduler.flat`
  and `Postrange.Scheduler.v`.
- **Differs:**
  - the kernel's slice, its reductions, the ranges they close, its axes and
    their sizes are computed when first asked after the kernel changes, and kept
    until it changes. tinygrad computes them at each query;
  - a kernel whose reductions and ends each close distinct ranges, which the
    rewrite would leave unchanged, is kept without it; any other is rewritten.
    tinygrad rewrites every kernel, a copy's included.

  The kernels and the answers are the same.
- **Reason:** (d): a beam search's rounds and the hand-coded optimisations of
  every kernel. With a range's reduce role read from the reductions, each axis
  query listed the kernel's slice up to five times, and each copy of a scheduler
  rewrote its kernel; with both changes the tolk bench's `search/actions/*` rows
  run 25 to 27% faster than before the role, and `matmul/codegen` within 1%,
  where it had taken 12% longer.
- **Pinned by:** `Tolk.Postrange › a scheduler's axes answer as a fresh copy's,
  whatever came before`; `› the scheduler's axes after optimising are as
  recorded`; `› Scheduler ›` `v's kernel has the ranges its reductions and ends
  close flattened`, `v's kernel of a recorded kernel is it flattened` and `a
  copy's kernel after optimising is the kernel flattened`; the tolk bench's
  `search/actions/*` rows on fudge, and its `*/codegen` rows.

## D170. A kernel's parameters name no device

- **tinygrad:** `schedule/rangeify.py` `debuf` gives each kernel parameter its
  buffer's device, and `codegen/__init__.py` `to_program_key` keys a program by
  `ast.key`, so each device makes its own program of one kernel. Copy detection
  and the check that a kernel spans one device read the kernel's parameters
  (`schedule/__init__.py` `assert_all_same_devices`, `copy_kernel_to_store`,
  `simplify_copy_kernel`), the renderer is picked from the call's device, which
  a call takes from its body (`engine/realize.py` `_get_call_to_compile`), and
  the search names its parameters' device itself (`codegen/opt/search.py`
  `_try_compile`).
- **tolk:** `Rangeify.debuf`; `Schedule.is_copy`, `copy_runs`,
  `assert_all_same_devices` and `copy_call`; `Codegen.program_key`
  (`Codegen.params`); `Realize.get_call_to_compile`; `Search.timed_call`;
  `Spec.program`.
- **Differs:** a kernel's parameters name no device: the device is its call's
  arguments', where a kernel launches (`engine/realize.py` `exec_kernel`
  launches there too). Kernels of one expression on several devices of one
  target are one node, one key and one program, in memory and on disk. Copy
  detection and the one-device check read the call's arguments, the renderer is
  the target's of the call's first storage argument, where the engine launches
  the kernel, and `Spec.program` refuses a parameter that names a device. A
  search's timed call names its target's device, and its caller places it on a
  device of the target (rune's `search`). `test/gen/graph.py` `unplaced_kernels`
  writes a kernel body's parameters without a device, so the goldens carry this
  whatever tinygrad computes.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.make`, `Pieces.split`) lower each device's operations on that
  device, so with the devices in the key a data-parallel step's first call over
  16 software devices compiled each kernel 16 times; with one program per target
  it compiles each once. A device kept in the kernel and erased for the key
  would hold one fact in two places.
- **Pinned by:** `Tolk.Schedule › create_linear_with_vars › rules › one
  expression on two devices of a target is one kernel`, `… › a kernel on buffers
  of two devices is refused` and `copies into runs`; `Tolk.Search › a search
  kept for one device serves another of its target`; `Tolk.Spec › program ›
  parameters`; rune's `Rune.jit over several devices`, `searches` among them.

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

- **tinygrad:** `engine/jit.py` `CapturedJit`: the linked linear of a captured
  schedule holds its intermediates' buffers from its first call until
  `free_intermediates` drops it, so they outlive every run, and `engine/jit.py`
  `jit_lower` places them in arenas (`memory_plan_rewrite`).
- **tolk:** `Tolk_engine.link`'s `storage`, `Tolk_engine.per_run_storage`,
  `Tolk_engine.run_with`.
- **Differs:** a schedule linked `Per_run` is planned into arenas (D41), in the
  order by storage (D183, D185), and makes each storage node no buffer is bound
  to before its run's first call and frees it after its last, its memory coming
  back once the work on it is done. A run is one program of rig's, which loads
  each image once per device. A run's submissions are those of the same schedule
  linked `Linked`: the order by storage interleaves devices, and under rig's one
  point per submission they are what cross-device overlap costs, since grouping
  a device's work into fewer submissions lengthens a GPU-faithful critical path.
  Linked `Linked`, the default, its storage is the program's, as tinygrad's. A
  run of either keeps none of the buffers it was given once it returns. In a set
  of a GPU and a host-run device, where the plan places host work over bytes a
  GPU copy touched, the run waits for the GPU's earlier work before it returns
  (D41): the plan keeps the bound and gives up that call's asynchrony.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.variant`, for a split program) promise that a device holds between
  calls only the values its caller keeps, and RFC 0005 bounds what a fully
  sharded step holds within a call. Linked `Linked`, fully sharded training over
  two software devices held 4.75 layers on each device beside its share after a
  step (`rune jit › fully sharded`, d = 128). Each buffer made at its first call
  and freed after its last fails the bound on a GPU, which reuses freed memory
  only once its work is done while a run returns before it is: on fudge's RTX
  5000, eight chained products of 64 MiB, two live at once, held 640 MiB at the
  call's return, where planned into an arena they hold 320 MiB with the
  arguments and the result. That design also made one program of rig's per
  stretch between makes and frees, 1168 for a data-parallel step over 16
  software devices. Planned per run, rune's five `fully sharded` cases meet
  their bounds; copies in a lane of their own put each over its bound, since two
  arenas made for the whole run hold both lanes' peaks at once.
- **Pinned by:** `Tolk_engine › runs › a schedule linked per run holds no
  storage between runs`, `› a run holds its storage whole, no less than its
  buffers live at once`, `› a schedule linked per run does the work it does
  linked` and `› a run keeps none of the buffers it was given`; `Tolk.Memory ›
  memory_plan_rewrite › rules › a copy's buffer and a call's of one device share
  bytes once their lives are apart` and the plan law; rune's `fully sharded`
  tests, whose shares are exact after a step, and `Rune.jit on a GPU › memory ›
  a split program holds at its return at most its result and its plan`.

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

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations`, without
  `TC_MIN_GLOBALS`: after the core, M and N are upcast and N split into local
  threads, by the first size that divides them, whatever the workgroups left;
  the same at `6b7b018da`.
- **tolk:** `Renderer.t.core_threads`, `Renderer.t.local_workgroups` and
  `Heuristic.tensor_cores`.
- **Differs:** each upcast applies only while the kernel keeps at least
  `core_threads / Tc.threads` workgroups after it, the product of its global
  axes, or when that product is not known: 16384 threads, 512 workgroups of a
  CUDA or RDNA core of 32 threads and 256 of a CDNA core of 64, and on Metal
  32768, 1024 workgroups of its core of 32. An upcast leaves fewer threads to
  keep memory and the L2 busy (concurrency is latency times throughput, Volkov,
  UCB/EECS-2016-143 §3.1). The local split keeps the threads and leaves fewer
  workgroups to share among the compute units, so threads do not govern it: it
  takes the largest of 4 and 2 that divides N and leaves at least
  `local_workgroups` = 256 workgroups, or an unknown count, where tinygrad takes
  the first that divides. 16384 is a bound fitted to sm_89's parts, which every
  target takes until it fits its own; Metal takes 32768.
  Under `TC_MIN_GLOBALS` the splits are tinygrad's. A core whose operands are
  staged through shared memory takes neither these splits nor their floors but a
  tile of its own (D262), with the floor `core_threads / stages`. The floors are
  fixed per target: one read from the part would give two parts of one target
  different bits, since the floor moves the order of a sum (D89's constants).
- **Reason:** (d): gpt-oss-20b on fudge's RTX 5000 Ada, its products at bfloat16
  widened by a cast (D29). Since staging (D262), `stream/key-projection` and
  `stream/dense-bf16-64` no longer reach the floors, and the key projection's
  numbers below are those that set them; the row that still gains is
  `stream/decode-keys`, 7.4 against 7.8 us without the floors, 1.05x for about
  15 lines; gpt-oss-20b's routed experts take the unstaged path and reach the
  floors, which refuse nothing there. Past 16384 threads the splits lose reuse
  (at 65536 `matmul_half_1024` is 1.97x slower), and the decode step's keys took
  5.37-5.86 us at a floor of 100 workgroups (4096 threads) and 4.15 us at 16384,
  so fudge's window is (4096, 16384]. The local split as tinygrad's, unfloored,
  made the key projection 1.34x slower, and a decode step's attention values, 64
  by 64 outputs over 1024 slots, take 32 workgroups and 23.6 us where tinygrad's
  splits leave one and 68.9 us. A split is harmless where `ceil(W / (L U)) L <=
  ceil(W / U)`, for `W` one-warp workgroups, a split `L` and `U` units, which
  needs `U`, a fact of the part. 256 workgroups with the fall back to 2 is the
  floor fudge's rows admit, between 1.44 and 2.56 times its 100 SMs: against 512
  with D211 it takes a split by 2 that the key projection gains by (88-115 to
  78-96 us). On parts of 24 to 142 SMs it is a fit, not the test: unmeasured. On
  parts the size of an RTX 4060 (24 SMs, 272 GB/s) the model puts the window at
  about (1000, 3900] threads, where 16384 may cost 1.4-2x: unmeasured, and a row
  on a small part is owed. AMD takes the same value; its own is to measure on
  nonnormal's R9700. Metal's 32768: Reason: none. On an M1 Max (Apple7, 32 GPU
  cores), in GPU time, a float32 product of 1024 squared ran in 268 us against
  316 at 16384; other Apple parts are unmeasured, and the heuristic's
  `matmul_1000` and `scores`, whose splits 32768 changes, are untimed. No
  machine a bench may run on holds a Metal device, so no row can pin it.
- **Pinned by:** `Tolk.Heuristic`'s cases, on CUDA: `matmul_half_1024`,
  `matmul_half_512`, `matmul_half_256`, `attention_values` and
  `matmul_half_variable` (a variable counts the workgroups), with the layouts in
  the suite's `.tolk` files; its `the tensor core's floor › the splits after a
  tensor core are those that leave its floor of workgroups`, over every recorded
  core kernel, CDNA's included, with the local split's floor of 256 workgroups;
  the `routed_run` cases of D200 on CUDA, whose local split by 2 the floor of
  256 admits; its `Metal's renderer floors its cores at 32768 threads`, where
  `matmul_1000` and `scores` split otherwise at 16384; the tolk bench's
  `stream/decode-keys` row on fudge.

## D176. Withdrawn

A folded selection or negation kept a NaN's bits; merged into D27, which holds
its rule, its reason and its pins.

## D177. A name bound to several values is passed by each call that reads it

- **tinygrad:** `schedule/__init__.py` `create_linear_with_vars` (raises "bind
  mismatch" when a schedule's kernels read one name bound to two values) and
  `resolve_linear_call` (substitutes a bound variable into its kernels
  unbound, read by name from the run's `var_vals`); `function.py`
  `is_implicit_storage`, `pm_ctx` (a traced function captures each variable
  its body reads as an argument); `schedule/__init__.py` `pm_replace_buf`,
  which passes the schedule's own the same way.
- **tolk:** `Schedule.create_linear_with_vars` (`bound_several`, and its
  `used_vars`, which reads the kernels of loops),
  `Schedule.canonicalize_call_body` and `Schedule.pm_pass_bound`,
  `Schedule.resolve_linear_call` and `Schedule.pass_values`;
  `Tolk_engine.var_word`, and `Realize.call_value`, through which the grid
  and split size of `Tolk_engine.launch` read a nameless scalar from its
  call's slot.
- **Differs:**
  - Where the graph binds variables of one name to several values, a kernel
    that reads one takes it as a nameless parameter of its range, in a slot
    past its arguments, and its call passes the bound variable there, as a
    loop's call passes its trip (D140). The name is not among the schedule's
    variables, and the engine reads each bound variable apart, so each launch
    computes with its call's value, unless a run binds the name. A kernel
    takes such parameters in the order its body reaches them, so kernels that
    differ in their values alone, or in the order the schedule met those
    values, are one program. A name bound to one value is read by name, as
    tinygrad reads it.
  - A call passes the variables its body binds by the same rule
    (`pm_pass_bound`), where the schedule gives each call's call-local
    storage its slots: each becomes a nameless parameter of the call in a slot
    past its arguments, one per value, and the call passes the bound variable
    there. tinygrad's function does this as it
    traces a body into a call; tolk takes a call's body as built (`Ops.call`),
    so the schedule does it, once, for every builder. No body the schedule
    splits into kernels then binds a name: stores of one kernel (D137) and
    calls run as one launch (D404) read each value from a slot of their own. A
    kernel the body calls, such as a custom kernel whose code binds a
    variable, reads it from the body's slots as the kernels split from the
    body do (`Schedule.pm_pass_kernel_bound`), which resolving the scope fills
    (`Schedule.resolve_linear_call`).
  - A name of one value that a loop's kernels read is among the schedule's
    variables, where tinygrad, without loops in a schedule, reads from its
    top-level kernels alone.
- **Reason:** (b): rune's compiled programs over several devices
  (`Pieces.split`, `Compile.moving`) bind each device's window start, of a
  move or of a loop's coordinates, to a variable named by its bound, so that
  every device's kernel is one program (D170) and the data-parallel step
  takes as many programs over 2 to 16 software devices. A body's variables,
  (b): rune's loops (`Loop.cut`) keep the variables their steps bind in the
  loop's body, such as the start of each piece of an assembly
  (`Compile.region`), which binds one name to each piece's start; named in
  the body, two stores or calls
  of a trip that one kernel joined passed the name one value.
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
  value`, and the same on each GPU; `Tolk_engine › link and run › one program
  launched on several devices computes each device's window at its start`
  and `› kernels whose sizes are one name of two values run each its own`,
  and the same on each GPU; rune's `rune jit › jit › two assemblies of alike
  pieces compile`, `Rune.jit over several devices › maps over a cut axis ›
  per-device code's programs do not grow over 2 to 16`, `› programs › a
  data-parallel step's programs do not grow over 2 to 16`, `› loops › a step
  that reads its coordinates gives eager's bits`.

## D178. A Windows host's code is compiled for Windows' calling convention

- **tinygrad:** `runtime/ops_cpu.py` `CPUDevice.__init__` (the CPU's arch, its
  machine and `native`); `runtime/support/compiler_cpu.py`
  `ClangCompiler.compile` (compiles for `<machine>-none-unknown-elf` on every
  system); `renderer/cstyle.py` `base_rewrite` (an `int64` literal ends in
  `l`), `CStyleLanguage.abi` (`""`, which `ClangRenderer` keeps) and
  `CStyleLanguage.type_map` (an `int64` is a `long`).
- **tolk:** `Tolk_engine.target`, whose host arch on this machine under Windows
  names the system after its machine (`x86_64-windows,x86-64-v3,tune=native`,
  `arm64-windows,generic,tune=native`); `Compiler_cpu.clang` and
  `Compiler_cpu.win64`; `Cstyle.clang`, whose language under `x86_64-windows`
  names a 64-bit integer `long long`, writes its literals with `ll`, and
  declares the kernel `sysv_abi`; `Codegen.host_entry`, which declares the
  entry `sysv_abi` there.
- **Differs:** a host target names Windows after its machine. Under
  `x86_64-windows` the object is compiled for `x86_64-w64-windows-gnu-elf`
  with `-mno-stack-arg-probe` and `-mno-red-zone`, so that it calls every
  function by Windows' convention, and its kernel and entry are declared
  System V's. Windows is LLP64, so a 64-bit integer is a `long long` and its
  literals end in `ll`. tinygrad compiles for no system everywhere, and its
  CPU arch names none. Every other target renders and compiles as before. The
  system comes from the device (D95): this machine's host runs this process's
  system. A remote Windows host's target names none, so its programs call by
  System V, until rig states a machine's system. A program's key holds its
  target (D113), so it holds the system.
- **Reason:** (c). rig calls linked code by System V and takes each symbol the
  object does not define from the process, whose functions follow the
  platform's convention (`Rig.section-host_abi`, `Rig.section-host_symbols`).
  Compiled for no system, the object called them by System V: `frexp` read
  its pointer from `rdx` where the kernel put it in `rdi`, and wrote the 32
  bytes above its return address that Windows lets a callee use, so `CG ›
  calls in a kernel › an external function writes through its pointer
  argument` crashed on windows-2025. Clang also calls functions the source
  does not name (`fma` and `trunc` without FMA or SSE4.1, `fminf` and `fmaxf`
  for a half selected against a constant, half conversions without F16C,
  `memcpy` for a large copy), which compiled for Windows follow Windows'
  convention, as `Rig.section-host_abi` states such code. rig commits
  a thread's stack before it runs linked code, so the code probes none, and
  Windows may write below the stack pointer.
- **Pinned by:** `Tolk.Cstyle › x86_64 Windows`: every x86_64 Clang case
  renders with no LP64 `long` and as its golden but for its 64-bit types,
  literals and the kernel's `sysv_abi`; `› the kernel is declared System
  V's`; `› a 64-bit integer is a long long and its literals end in ll`; `›
  an object is compiled for Windows' triple and calls no stack probe`; and,
  on an x86_64 host, every Clang case compiled for Windows loads and every
  interpreted one that calls nothing of the process runs through rig's host
  and writes what the interpreter computes. `Tolk_engine › target › the host
  compiles C for Clang at its arch, tuned for its own processor, naming its
  system under Windows, whose functions follow a convention of their own`. On
  windows-2025 CI, `CG › calls in a kernel`.

## D179. An index of a reshape's added leading axes is its source

- **tinygrad:** `schedule/prepare.py` `_mop_index`, whose shortcut for a
  reshape whose added leading axes are indexed whole tinygrad 8b4d8c026
  deleted as dead code.
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

- **tinygrad:** `schedule/indexing.py` `pm_fix_deviceless` and
  `run_rangeify`, which place every stage of no device on the sink's device,
  the first device among its sources.
- **tolk:** `Indexing.fix_deviceless`.
- **Differs:** a stage of no device, an arange or a table computed from
  constants that rangeify stores, takes the device of the nearest node above
  it that has one, the device of the kernel that reads it. Read by kernels on
  several devices, it is one stage per device, each computed there. A stage
  that no node with a device reads, or that only a store to a disk reads,
  lives on the sink's device. On a sink of one device the two rules agree.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.lower` of `Pieces.split`) schedule one sink whose kernels each
  compute on one device. A constant table they share (the shifts that unpack
  a sub-byte bitcast, `Compile.unpack`, `Lower_arith`'s bitcast, or an
  `exp2` of an arange) was stored on the first device and read by the others'
  kernels, which `Schedule` refuses, so the call evaluated.
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
  own); `schedule/__init__.py` `pm_copy_kernel` (lowers a copy kernel that
  reads and writes both buffers at its range alone).
- **tolk:** `Prepare.storage_run` and its use in `Prepare.earliest_rewrites`;
  `Schedule.run_start` and `Schedule.copy_runs` in
  `Schedule.pm_copy_from_store`.
- **Differs:** a copy stored into a run of consecutive elements of storage on
  its device that shrinks and reshapes view (`Prepare.contiguous_view`) is the
  store, so the copy lands in the run. Its kernel reads its source at its
  range and writes the destination at its range plus the run's first element,
  and lowers to a copy between views of the call's arguments, each the run it
  copies. tinygrad copies into storage of its own and stores that into the run
  with a kernel, so both are live at once. A copy stored into a run of a
  bitcast of storage is a copy into a view of it (D184): the copy moves bytes,
  so it lands in place too. A copy of one element indexes each side at its
  element and lowers the same way. No recorded program stores a copy into a
  part of storage.
- **Reason:** (b): rune's split assembly of a gathered value (RFC 0005), which
  stores each device's piece into its run of the whole. With kernels ordered
  as D183 says, fully sharded training's backward call of one layer at d = 512
  holds 1.812, 1.531 and 1.391 layers beside its arguments over 2, 4 and 8
  software devices, against RFC 0005's bound of 1.875, 1.562 and 1.406, where
  each piece copied apart held 2.062, 1.594 and 1.422 (`rune jit › fully
  sharded`).
- **Pinned by:** the Schedule suite (`test/schedule/schedule`): `copies into
  runs › a copy stored into rows of storage on its device copies into them`,
  `› a copy stored into a first row copies into it`, `› a copy stored
  into columns lands in storage of its own`, `› a copy stored into a run of
  a bitcast copies into a view of it` and `› a copy of one element stored
  into filled storage copies into it`; the Engine suite's `link and run › a
  copy stored into rows of a buffer writes them alone`.

### A store into a run of a buffer stores into a view of it (D184)

- **tinygrad:** `schedule/__init__.py` `pm_callify_ctx_collect`:
  `contiguous_mops_to_view` makes a view of a buffer for a copy or a stage of
  movements of it, and for a store only into a bitcast of it. A store into a
  shrink of a buffer indexes the buffer at the shrink's offset in its kernel,
  so each offset is a program of its own.
- **tolk:** `Schedule.view_of`, through `Schedule.pm_callify_ctx_collect`.
- **Differs:**
  - A store into movements or a bitcast of a buffer that select a contiguous
    run of its elements stores into a view of that run: the kernel takes the
    run as its buffer, and a copy stored there is a copy into it (`Prepare`'s
    rule for a copy stored into storage on its device). Stores of one shape
    into runs of one buffer at other offsets are one program.
  - A store into a run of a state of the buffer stores into a view of the run
    too: the states order the store (D473), and it writes the buffer as it
    runs.
  - A store whose value reads the buffer in its own kernel, short of the
    stored states of other storage, keeps indexing it, which the kernel split
    stages by the buffer's identity.
  - A copy or a stage of a run of a buffer that a store of the call writes
    keeps reading the run itself, which the kernel split stages: a view, a
    parameter of its own that nothing writes, would be read in whatever state
    the buffer holds when its reader runs.
  - A run may start at bound variables, whose values its call knows
    (`Prepare.contiguous_view` checks the run at those values): a store into
    it stores into a view at that start, whose kernel knows the start modulo
    what the variables move it by (D54), so stores of one shape at starts of
    other alignments are one program too. A copy or a stage of a view still
    needs a constant start.
  - The program `after_part_of_older_write`, two stores into one element each
    of one buffer, is recorded as a kernel graph alone: its linear here is one
    launch (D404) over two one-element views, where tinygrad's indexes the
    whole buffer in a kernel per store.
- **Reason:** (b): rune's compiled programs over several devices
  (`Compile.assembled`) hold an assembly as a buffer filled by a store per
  piece, each at its device's offset. Indexed at the offset, a data-parallel
  step over 2, 4, 8 and 16 software devices made 19, 31, 55 and 103 programs,
  against rune's promise that they do not grow with the devices (D170);
  through views, 11 each, and 11, 13, 13 and 13 once rune reads a closing's
  own slot from storage of its own. A view at a constant start keeps that
  start's alignment, and a closing's parts lie at starts of several: a sum
  over an axis cut across 2, 4, 8 and 16 software devices made 4, 7, 9 and 9
  programs, and 4, 7, 7 and 7 with each device's own part stored at a bound
  start; indexing the whole buffer at that start held 3.02 MB per device in
  fully sharded training over two devices, against the 2.29 MB its bound
  allows and views keep. The refusal of a view of a buffer the call writes is (e):
  `x[0:2].contiguous()` read after `x.assign(...)` in one realize read the
  assigned `x` (tinygrad's `test/runtime/test_assign.py::test_swap_slices`,
  an expected failure), at the cost of a stage kernel per such view.
- **Pinned by:** `Tolk.Schedule › contiguous_mops_to_view › stores into runs of
  a buffer at a bound start store into views of one alignment, whatever the
  start`, `› stores of one shape into two runs of a buffer share a kernel`, `› a
  store into contiguous rows of a buffer stores into a view of them`, `› a store
  into rows of a buffer after a store stores into a run of that state`, `› a
  store repeated over a later state stores again`, `› a read of the run a store
  stores into reads the state`, `› a store into rows of a buffer of a value that
  reads it has no view`, `› a store into rows of a buffer of a value that reads
  it through other storage stores into a view` and `› a store into columns of
  storage has no view`; `Tolk_engine › link and run › a store into rows of a
  buffer of a value read through other storage writes what it computes`;
  `Tolk.Schedule › create_linear_with_vars › merged calls › the work on a
  schedule of many outputs is linear in their number` and `› views of written
  buffers › a stage of a run of a buffer the call overwrites reads it as it was`
  and `› halves of a buffer swapped through stages swap`, and the programs
  `stage_of_overwritten` and `copy_of_overwritten`; rune's `Rune.jit over
  several devices › programs` and `› maps over a cut axis › partials give
  eager's bits, their programs do not grow`, and `rune jit › fully sharded`.

## D183. Ready kernels run by their number, those that free storage first

`Schedule.storage_order` keeps both parts.

### Ready kernels run by their number, and one that frees storage first

- **tinygrad:** `schedule/__init__.py` `create_schedule`'s linearize (Kahn's
  algorithm over a FIFO `deque`, so ready kernels run in the order they became
  ready).
- **tolk:** `Schedule.storage_order`, the one order of `create_schedule`
  and `create_linear_with_vars`, and the storage `Schedule.schedule` sees
  written; `Memory.taken`, the storage an entry takes as the memory plan
  counts it; `Tolk_engine.footprint`, which counts what a link of the
  order makes. `Ops.call_info`'s `loop`, which tinygrad's `CallInfo` has
  not and `Ops.pp_arg` leaves out: set on a call that an end over loop
  ranges holds before the schedule enters its body, and read, with
  `schedule_key`, when the body is scheduled.
- **Differs:** a ready kernel runs by its number, its place in the order
  `Ops.toposort` of the kernel graph meets it (each kernel after what it
  reads), so where the walk reaches a value's readers right after it, the
  value is made just before them. A ready kernel that frees storage, being the
  last to take storage an earlier kernel made, and that on no device makes
  more than it frees, runs before the others, bytes counted unrounded. The
  storage a kernel writes lives from the first kernel that takes it to the
  last; storage only kernels read lives throughout, and so do the
  parameters of a body an end over loop ranges holds, storage the body is
  given, whatever the loop's number of trips, one included, where the loop
  folds and the call inlines into its caller. Where the walk reaches each
  of two independent chains whole, as from a sum of their results, each
  runs to its end before the other begins; where the sink lists their parts interleaved, the numbers
  interleave them too. The order is a postorder that takes a node's sources
  in their order, and no promise of the least storage
  (on a tree the least is the postorder by decreasing peak, Liu 1987, which
  tolk does not sort for). tinygrad runs every kernel whose inputs are ready
  breadth first: a closing slices its whole partial into every piece before
  any is copied or summed, and every independent chain starts before any ends.
  Picking costs one pass over each ready kernel's storage, so a schedule of
  `k` kernels costs `O(k²)` at worst. Every schedule takes this order, linked
  once or made per run: one order is what `Rune.footprint` plans and what a
  run allocates.
- **Reason:** (b): rune's fully sharded training (RFC 0005's memory bound),
  whose split programs' storage is planned by each buffer's life (D173). With
  copies landing in place (D182), the backward call of one layer at d = 512 holds 1.812, 1.531 and 1.391 layers
  beside its arguments over 2, 4 and 8 software devices, against the bound of
  1.875, 1.562 and 1.406, where tinygrad's order held 2.625, 2.312 and 2.156
  (`rune jit › fully sharded`). Ordering ties by the kernel that became ready
  last instead makes every piece of a gathered value before any is read, and
  holds 2,759,680 bytes against the bound of 2,293,760 over two devices. And
  rune's compiled gradients on one device, with the placement of D477: on the
  host, k independent chains of 4 remat'd blocks held 1,671,280 bytes for k = 4 in
  tinygrad's order and hold 557,168. A loop body's parameters given: on the
  host a scan summing in its carry the gradients of 8 remat'd blocks read as
  rows of two stacks held 1,638,532, 1,294,484, 1,204,372 and 893,076 bytes
  over 1, 2, 4 and 8 chunks, its first store into the carry making the whole
  stack and waiting behind every remat call, and holds 1,441,924,
  1,032,340, 811,156 and 667,796, below the 1,443,588, 1,034,004, 812,820
  and 669,460 of separate arrays. The 1 chunk row comes from the remats'
  loops of one trip, which fold and inline: their callers' storage is no
  more made by a store in the inlined body. The order interleaves devices, so a
  data-parallel step over 16 software devices makes 1018 `Submit` steps;
  preferring, among each kind of ready kernel, one on the last kernel's
  device raised four of rune's five `fully sharded` cases over their bounds.
  Grouping a device's work at link instead keeps the bounds but lengthens a
  GPU-faithful critical path under rig's one point per submission. On a GPU, the launches of one device's work the order groups are
  unmeasured on fudge.
- **Pinned by:** the Schedule suite (`test/schedule/schedule`):
  `create_schedule › order › a kernel that frees storage runs before one
  that only makes it`, `› pieces of one value are each read before the
  next is made` and `› chains the walk reaches whole run one after the
  other`; `create_linear_with_vars › loops of calls › a loop's body makes
  and frees its own storage`, `› a loop's body and the body called once are
  scheduled apart`, `› a carry's read keeps its region's store after it` and
  `› a carry's read through storage keeps its region's store after it`;
  rune's `fully sharded` tests,
  `rune remat › memory › independent chains are computed one at a time`
  and, for a loop body's parameters, `› stacks read by rows hold what
  separate arrays hold`. The recorded schedules whose order differs carry
  this entry in their `.tolk`.

### A ready kernel that only frees runs before one that also makes storage (D185)

- **tinygrad:** `schedule/__init__.py` `create_schedule`'s linearize (Kahn's
  algorithm over a FIFO `deque`).
- **tolk:** `Schedule.storage_order`'s `pick`, for a schedule made per run
  (D183).
- **Differs:** of the ready kernels that free storage (D183), one that makes
  no storage runs first, by its number; only where none does, one that on no
  device makes more than it frees. A kernel holds what it makes beside what
  it reads, so one that frees much but makes some raises what a schedule
  holds at once where one that only frees does not. tinygrad has no such
  order.
- **Reason:** (b): rune's fully sharded training (RFC 0005's memory bound),
  whose split programs' storage is planned by each buffer's life (D173). At
  the end of a closing, the copy of a device's own slot of the last piece is
  the last reader of the partial and makes the slot's storage, where the sum of
  the piece before it only frees. Run first, the copy held three slots beside
  the partial: the backward call of one layer at d = 512 held 1.938, 1.594 and
  1.422 layers over 2, 4 and 8 software devices,
  against this order's 1.812, 1.531 and 1.391 and the bound of 1.875, 1.562
  and 1.406.
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

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (upcasts
  of output axes: by 3 or 4, of an axis that some access does not index while
  it indexes every upcast axis, while 1024 elements remain and the kernel has
  fewer than 32 lanes; the unroll of the last reduce axis, whole when it is at
  most 32, by 4 otherwise, while the kernel has fewer than 64 lanes).
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
  consecutive addresses (Clang packs a register only as wide as the stores it
  reads, D187). Every reduce's accumulators, one for each output lane, summed
  over the kernel with no notion of which loop is hottest, take at most half
  the registers, judged before each split: a later split that nests a stored
  term, `(a * 4 + b) * 3 + c`, reads as lanes apart afterwards, so
  `held_registers` can then exceed the half (`r_128_16_3_4_4_3_3_400_400` on
  AVX2, 16 of 16); reading through the nest admits more unrolls, and one rune
  kernel ran 1.7 times slower. The upcasts after the masked and shared ones
  (D92) and the unroll take the budget in place of tinygrad's lane counts:
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
  at most 4. The register file is the compile target's, which the compiler's
  cache key holds, so a program follows the instruction set it is compiled
  for and no fact of the part it runs on.
- **Reason:** (d): lorenz_simple's step (sofo-raven) and the host's
  contractions. Each accumulator carried around a loop is a dependency chain,
  and Intel's optimization manual asks for latency × pipes independent chains
  (8 vector accumulators on Golden Cove), with loop-carried values in
  registers before invariants. Timed on fudge, one P-core (`taskset -c 4`,
  the variants interleaved under the timing lock, loaded, three runs), a
  float32 matmul of 128³ (`host/matmul/heuristic`) takes 30.0-30.5 µs against
  69.8-71.9 on origin/main, under D91's 32 lanes; the fill takes lorenz's
  four-sum kernel (`lorenz/r_128_16_3_4_4_3_3_400_400/heuristic`) from
  8.01-8.40 ms to 6.23-6.68. A model of which reduces share the hottest loop
  changed 3 of 864 recorded kernels, none on a GPU; against it, lorenz's
  `r_128_16_3_4_4_3_3_400_400` runs 1.27-1.37 times slower and its
  `r_32_256_50_4_4_2_4_4` about 1.2 times faster.
  D91's 32 lanes were this budget at a render that filled half of each float32
  register.
- **Pinned by:** the Heuristic suite: `the host's budget › on AVX2 a
  kernel's accumulators hold` (every case), `the host's budget › the register
  budget › a kernel's accumulators take at most half the registers` and
  `› the kernels the budget is checked on have each of its shapes`, `the
  host's budget › on the host a larger register file never holds fewer output
  lanes` and `› on the host a reduce of doubles over a stored axis of a
  multiple of 32 takes half the lanes of floats, in the registers floats
  fill`, over every recorded kernel and lorenz_simple's, on files of 16 × 16,
  16 × 32, 32 × 16 and 32 × 64; `the optimisations chosen are tinygrad's › …
  › the optimised kernel`, the `cpu` (16 × 16) and `avx2` (16 × 32) cases; the
  Compiler_cpu suite's `the vector registers, their number and bytes, of`
  (every case) and `without Clang › an x86_64 processor has SSE's 16
  registers of 16 bytes`; the tolk bench's `host/*/heuristic` and
  `lorenz/*/heuristic` rows on fudge.

## D187. The host's stores take a full vector register

- **tinygrad:** `codegen/late/coalesce.py` `memory_coalescing`'s lengths (4 or
  2 elements of float32, float16, int32, uint32 and the 8-bit floats, 8 of
  float16 under `ALLOW_HALF8`, none of the others, bytes included; D261 makes
  that 8 the renderer's, for copies into local memory).
- **tolk:** `Coalesce.memory_coalescing`, through `Renderer.registers`.
- **Differs:** where the renderer knows its vector registers, loads and stores
  of float64, float32, float16, int32, uint32, the 8-bit floats, int8 and
  uint8 merge into accesses of a power of two of elements up to one register:
  8 floats, 4 doubles or 32 bytes in AVX2's 32-byte registers, 4 floats, 2
  doubles or 16 bytes on SSE and NEON. A store of doubles or bytes merges so
  only when each value is a load, as an accumulator's read is; doubles and
  bytes computed lane by lane are stored one by one, as tinygrad stores them.
  A group of byte loads any of which a computation reads alone is not merged,
  as tinygrad merges no bytes: LLVM's load combine folds an or of bytes in
  descending addresses, from vector loads, into a byte-swapped load at the
  address of the first byte's vector, the wrong one where that byte is not its
  vector's first, so a compiled running sum of a widening bitcast of flipped
  bytes read a byte off; Clang merges bytes that feed one lane each itself. On
  the CPU device a buffer is taken to start on a boundary of that width (D54).
  Other renderers merge as tinygrad does, but where the renderer writes bytes
  as words (D482).
- **Reason:** (d): the host's contractions and lorenz_simple's step. Clang's
  straight-line vectorizer packs values into registers as wide as the stores
  it reads them from: a float32 matmul's inner loop with its 16 stores as
  `float4` takes 65 instructions and spills 11 values, as `float8` 31 and
  none, and a load as wide as a register feeds it whole: `tangent_products`
  replays in 51.0-55.6 µs against 190.5-193.5 µs with loads of 16 bytes. Cost:
  a kernel whose loads of a register are read lane by lane into a reduce along
  them takes longer (`host/max_rows/heuristic` 2.8 µs to 3.4-3.5,
  `lorenz/r_128_16_3_4_4_3_3_400_400/searched` 6.53-6.60 ms to 7.22-7.28),
  kept because the gains are larger. Doubles computed lane by lane, merged,
  made Clang pack the computations behind them: rune's float64 normal sampler
  (`host/normal_f64/heuristic`, rune's compile bench `rng/normal-f64`) replayed
  in 26.8-28.2 µs against 13.8-14.1 stored one by one. Floats computed lane by
  lane keep the merge: lorenz_simple's two-sum kernel stores them, and runs
  2.99-3.15 ms with it against 3.50-3.62 without. Bytes: an MXFP4 code's
  byte feeds two lanes, so its loads reach Clang as a vector built with one
  `vpinsrb` a lane; merged, kaun's `Quant/host/mxfp4-product-1-token` takes
  6.48-6.81 ms against 9.89-10.45. Only the loads were measured, so bytes
  computed lane by lane are stored one by one, as doubles are. Timed as in
  D186.
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
  upcasts of output axes by 3 or 4; the unroll of the last reduce axis, and of
  a second one when both are at most 3; the upcast by 4 of a kernel that has
  none).
- **tolk:** `Heuristic_host.host_vector_bytes`,
  `Heuristic_host.parallel_operations`, `Heuristic_host.upcast_vectors`,
  `Heuristic_host.stored_contiguous`, `Heuristic_host.pad_waste`,
  `Heuristic_host.upcast_contiguous` and `Heuristic_host.unroll_vector`.
- **Differs:** on the host a group of lanes computed as one vector spans 64
  bytes of its widest type, two registers of AVX2. A kernel with no reduce
  takes, after its masked axes, one upcast of its last output axis that 2
  divides, by the largest power of two, at most 16, that keeps its lanes
  within 64 bytes of the widest type it reads, writes or computes, and again
  while they have room: 16 float32 lanes, 8 float64, 32 float16. It takes
  none when its stored values hold 5 arithmetic operations or more for each
  of their longest chain (a sine is one operation when the heuristic runs),
  or when it computes a type the target emulates. A reduce kernel's first
  upcast takes its stored axis to 64 bytes of its accumulators (D186). In a
  kernel that sums products, an access may read that axis apart when it reads
  the last reduce axis it reads at unit stride, as a matrix stored [out; in]
  is read by its product, or a row of MXFP4 codes by its bytes: its lanes
  along the axis are gathered by a transpose (D141), where every reduce axis
  after that one is short enough to unroll whole; or when it reads the sum
  only in blocks of at least 16 terms, every reduce axis it reads as such a
  quotient, as a row's scales are read once a block of 32 codes.
  Such a gathered axis takes one register of lanes, and the output axis the
  gathering accesses do not read, or read only as a quotient by d, as a block
  of d routed rows reads its expert's matrix, is then upcast by the largest
  power of two that divides each d and fits the budget: a product of float32
  operands both stored along its sum takes 8 rows by 8 columns, BLIS's
  micro-tile with its B gathered, and so do a prompt's blocks of 64 routed
  rows against their decoded experts. In
  a kernel that sums products, a stored axis whose largest power-of-two
  divisor holds fewer than 16 bytes of lanes is first padded to a multiple of
  16 bytes, where that adds at most a third of its extent: a head of 110
  floats to 112, 9 to 12; 36, runs of 4, is left. When
  the last reduce axis is at most 3 and wholly unrolled and the next one is
  larger than 3, the next is unrolled by the largest power of two that
  divides it and keeps the unrolled lanes within 64 bytes of the reduce's
  type, if the budget holds: 8 beside a pair of float32 lanes, 4 beside 3.
  The budget holds a conversion to 64 lanes, counted, where a float sum of
  products computes it a run at a time (D141), one that differs along every
  upcast axis, at one output axis's lanes: gpt-oss's routed MXFP4 product
  then takes 4 experts by 16 rows, 8 of them gathered, unrolling 4 bytes of
  codes beside their pairs.
  tinygrad upcasts output axes by 3 or 4 while 1024 elements remain and fewer
  than 32 lanes, by 4 a kernel that has none, and unrolls the next reduce axis
  only when it is at most 3 too. On 16-byte registers, SSE's and NEON's, a
  vector is four of them, unmeasured. Whether memory-bound kernels take lanes
  at all is open.
- **Reason:** (d). A padded axis: x [3328; 440] by 110, 50, 22 and 6 outputs
  ran 1.6 to 2.7 times as fast padded to 4 floats, by 9 and 25 (odd) 1.2 to
  1.5 times, at wastes up to a third; padding runs of 4 (36, 44, 100) to 8
  ran 0.64 to 1.0 times as fast (fudge, locked, P-cores 2 and 3, max of 3).
  A gathered axis: x [3328; 440] by a matrix [1792; 440] took
  4 rows by 4 columns, whose vectors span half a register and whose
  transposes no rows share; 8 by 8 runs at 136.4 GFLOP/s on one P-core
  against 62.7 (fudge, locked, min of 5), the same bits, and a 4 by 16 tile,
  two registers a column, at 82.2. Elementwise kernels (RFC 0025's special functions, D141):
  lanes give the core independent chains while each waits on its latency, and
  computed as vectors (D141) they compile once; eight nested float64 sines
  over 1024 elements replay 1.25 times as fast at 64 bytes as at 32 and as
  with none (fudge, one P-core); 128 bytes replay a further 1.05 times at 1.6
  times Clang's time. A kernel whose operations already run 5 at a
  time gains nothing from lanes and pays for them in compile time (`lgamma`,
  `log_betainc` and its gradients; 5 was measured on an M1 Max's cores, to
  re-measure on fudge). A reduce
  kernel's stored axis at 64 bytes runs a float64 matmul in 70.8-73.7 µs
  against 96.9-97.3 µs at one register. From D143, nx.quant's MXFP4 product
  reduces over a byte axis and the pair of codes each byte holds; unrolled to
  a vector, the dense product of a checkpoint's codes ran 2.6-3.0 ms against
  6.7-9.8 ms on an M1 Max. kaun's `Quant/host/*` rows hold the MXFP4 products;
  the tolk bench's `host/sin_nested_8/heuristic` and
  `host/select_zero/heuristic` the elementwise kernels, and rune's compile
  bench `trig/sin-nested-8` their compile.
- **Pinned by:** the Heuristic suite's `the optimisations chosen are tinygrad's
  › … › the optimised kernel`: the host cases `exp_float`, `exp_double`,
  `add_half`, `add_uchar`, `cast_to_double`, `compare`, `add_24`, `add_17`,
  `wide_tree`, `long_chain` and every host case without a reduce, and
  `vecmat_decoded`, `gpt_oss_gate_up`, `routed_blocks`, `routed_tiles` and
  `experts_down`, and the gathered axes of `scores`, `scores_biased`,
  `scores_biased_first`, `matmul_256_kt`, `matmul_256_kt_128`, `matvec`,
  `matvec_tall`, `rows_m4`, `rows_m12`, `rows_m32`, `rows_short` and
  `routed_by_argmax`, the padded axes of `heads_projection` and
  `heads_projection_biased`, on `cpu` and `avx2`, with `the host's budget ›
  the register budget › r_128 on AVX2 reads 8 registers once its stored axis
  is padded` and `the host's lanes › on AVX2 a
  recorded host kernel of few outputs takes its lanes` (`matvec_tall`); rune's
  `Jit/linear-float32-3328x440-to-1792`; the tolk bench's `host/*/heuristic` rows
  and rune's compile bench `trig/sin-nested-8` on fudge. On rune: missing for
  a quantised product decoding a checkpoint's codes 16 at once on the host.

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

- **tinygrad:** `runtime/support/compiler_cpu.py` `ClangCompiler`'s x86_64
  arguments (`-march=<cpu>` and its features); `runtime/ops_cpu.py`
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
  FP16. Every other Clang, and every other machine, takes tinygrad's
  arguments.
- **Reason:** (b): rune's `Lower_arith › casts give eager's values` at
  `(float16, 11)` and `rune vmap › compositions › under jit` at float16
  failed on GitHub's ubuntu-latest under Ubuntu 24.04's Clang 18.1.3 ("clang
  frontend command failed with exit code 139") on runners whose processor
  states AVX10.1. Clang 18.1.3 and 18.1.8 crash in machine loop-invariant
  code motion on the cast kernel `E_21_230` for `sapphirerapids`,
  `graniterapids`, `x86-64-v3` with `-mavx10.1-256` and `x86-64-v4` with
  `-mavx512fp16`, and compile it for `x86-64-v4`, `znver4` and each of these
  with avx512fp16 off. AVX10.1 implies avx512fp16,
  and under LLVM 18 disabling avx512fp16 or avx10.1-256 alone leaves it on:
  avx10.1-512, then avx10.1-256, then avx512fp16 must go. Clang 17.0.6 and
  19.1.7 compile the kernel for each of these processors, so every other
  Clang keeps the features.
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
compile's words or instructions. `Ops.key` writes every argument field by
field (D12), and `of_program` and `Tiny_elf.t` went with it.

## D194. A running operation scans chunks of 16 at every level

- **tinygrad:** `mixin/op.py` `_split_cumalu` (past 512 elements, chunks of
  256, the first padded before, scanned by windows, then the chunks' last
  elements scanned by one window each).
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
  makes `+0`; rune computes extremes over integer keys. Padded after at the
  first stage, a chunk's loads would be gated each by its own element, which
  the host kernel loads one at a time. Each element reduces a window of at
  most 16 elements and combines once per stage, a tail's elements once more
  in the first, so the work is linear in the axis: estimated
  (`Realize.estimate_uop`) at 6 operations an element over 65536 float32
  elements against tinygrad's 285, whose window is 256 elements, or the whole
  axis below 513. An axis of `n` elements takes about `log₁₆ n` stages.
  Values are a reassociation of the same terms: integers, booleans and
  extremes are unchanged, and a float sum or product of `n` terms keeps
  rune's bound of `γ(n - 1)`.
- **Reason:** (e), the chunks aligned from element 0: a running operation's
  prefix is the prefix of the axis's, bit for bit, as `Nx.cumsum` promises;
  padded before, an element's chunk moves with the axis's length, and a float
  sum of a prefix rounds otherwise. (d), the chunks of 16 at every level:
  compiled running sums, `Nx.cumsum` under `Rune.jit`, as norn's nested
  sampling's error estimate takes, and rune's lowering of an extreme's
  position and of segments' starts (`Lower_reduce`). On fudge, chunks of 16
  took float32 over 100 × 400 in 55-66 µs against tinygrad's 2.75-5.00 ms
  (both padded before), and aligned chunks take it in 44-59 µs against
  44-47 µs for chunks padded before with one window up to 32 elements; 16 was chosen among 8,
  16, 32, 64 and 256, as fast as 8 and a stage fewer.
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

- **tinygrad:** `renderer/cstyle.py` `base_rewrite`'s `BITCAST`
  (`__builtin_bit_cast(T, (S)(x))` for every value).
- **tolk:** `Cstyle.clang_vectors`'s bitcast rule and `Cstyle.byte_mask`.
- **Differs:** on Clang, a bitcast between vectors, the host's lanes of
  D141, is the C cast `((T)((S)(x)))`, which reinterprets the bits of a
  vector of one size as another's, and a mask of one-byte lanes is read as
  signed chars by that cast. A bitcast with a scalar side stays
  `__builtin_bit_cast`. Each lane holds the same bits either way. Of the 339
  kernels of rune's compile bench, the 61 with a vector bitcast compile to
  the same object but two, whose vectors of 3 lanes Clang pads to 4 in
  memory: `__builtin_bit_cast`'s temporary copies the padding lane, and
  their instructions differ in order and by up to 7 of 4520.
- **Reason:** (d): rune's compile bench row `special/betaincinv-grad-f64`.
  Clang writes `__builtin_bit_cast` of a vector as a store and a load of a
  temporary; the row's one kernel, with 2495 vector bitcasts, spent 5.2 of
  Clang's 6.2 s of CPU removing them, and 0.08 s with casts.
- **Pinned by:** the Cstyle suite's `sources › by default ›
  clang_vector_bitcast` and `› clang_vector_byte_mask`, and every host golden
  with a vector bitcast; rune's compile bench row `special/betaincinv-grad-f64`,
  its fudge baseline pending the scheduled re-record.

## D196. A schedule's host kernels compile in one run of Clang per domain

- **tinygrad:** `engine/realize.py` `lower_and_compile` (each kernel's
  program, `to_program`, made apart on its pool's workers);
  `codegen/__init__.py` `do_compile` and `device.py`
  `Compiler.compile_cached` (one source at a time);
  `runtime/support/compiler_cpu.py` `ClangCompiler.compile` (a `clang`
  process per source, from standard input).
- **tolk:** `Codegen.to_programs`, which `Realize.lower_and_compile` calls
  for every kernel; `Renderer.Compiler.batch` and
  `Renderer.Compiler.compile_all`; `Compiler_cpu.clang`'s `compile`, with
  `Compiler_cpu.cc1` and `Compiler_cpu.compile_files`.
- **Differs:** up to `PARALLEL` workers on `Worker.map` each take the
  kernels a schedule lacks, those `Codegen.read` does not find, one at a
  time, to lower and render until none is left, then compile their sources,
  each compiler's in one call (`Renderer.Compiler.compile_all`). A text
  two workers render, from kernels of two keys, is compiled by the first to
  reach its compile, and the other waits for its binary, compiling the text
  itself if that compile raised; tinygrad's workers each compile it unless
  the disk cache holds it when they look. Clang serves it with one run of its compiler proper (`clang -cc1`, the command
  Clang states for the options with `-###`, its IR verifier on, D275), each
  source the file `-` of a directory of its own, after a line marker that
  names it `<stdin>`, the files named in a response file. Each object is the
  one Clang's driver makes of the source alone from standard input, so the
  cache keys, tables and entries are unchanged. When a kernel asks for a
  beam search, each kernel compiles alone, in order. Only the compiler
  proper's own command saves the starts: Clang's driver starts a process for
  each of several sources (`clang -c a.c b.c` costs as many starts as
  `clang -c` per source).
- **Reason:** (d): rune's compile bench rows `scan/checks-48`,
  `scan/tridiagonal-100`, `devices/partials-16` and `model/schools-start`,
  which compile 111, 52, 60 and 39 host kernels, most under 8 KB, where
  Clang's start dominates. On fudge checks-48's 111 kernels took 2.4-3.5 s of
  CPU as 111 runs and 0.54-0.65 s as one run of the compiler proper. A text
  compiled once across workers: (d), its time to measure on fudge, with no
  bench row yet: tessera's cold training step compiled 5 of its 165 sources
  twice at 2 workers, each a compile that makes a binary already being
  made.
- **Pinned by:** the Compiler_cpu suite's `several sources › each object is
  the one its source compiles to alone` and `› a rejected source among
  others raises its own diagnostics`; the Renderer suite's `Compiler ›
  compile_all compiles the sources the table lacks in one call, each once`,
  `› compile_all calls nothing when the table holds every source`, `›
  compile_all raises for the first source the toolchain rejects` and `›
  compile_cached behaves as a table of binaries`; the Codegen suite's
  `programs are kept › to_programs makes each kernel's program, compiled in
  at most a list per domain`, `› to_programs on one domain compiles every
  source in one call`, `› to_programs compiles once a text two domains
  render` and `› to_programs raises for a text two domains render that is
  rejected`; rune's compile bench rows above.

## D197. A program key orders commutative index operands by the kernel's ranges

- **tinygrad:** `uop/symbolic.py` `commutative` (orders the two operands of
  a commutative index operation by `tuplize`, a range's number among it);
  `schedule/rangeify.py` `pm_renumber_range` in `split_store` (renumbers a
  kernel's ranges from 0 after the symbolic rules ran over the whole graph);
  `codegen/__init__.py` `to_program_key` (keys a program by its kernel's
  `ast.key`).
- **tolk:** `Codegen.params`, which `Codegen.program_key`,
  `Codegen.to_program` and `Codegen.read` read.
- **Differs:** codegen orders the operands of each commutative index
  operation of a kernel by the kernel's own range numbers, as the symbolic
  rules order them, then keys that kernel and makes its program from it.
  Two kernels that differ only in the order the whole graph numbered their
  ranges are one kernel: one key, one source, one program, which computes
  either. The key cannot map two sources to one binary: it reads the very
  kernel the source is rendered from, and the disk cache keeps a binary by
  its source. The schedule's kernel itself is tinygrad's.
- **Reason:** (b): rune's compiled programs over several devices
  (`Pieces.split`) put every device's kernels in one graph, where one
  device's scan of its tile numbered its ranges in the other order: a
  cumulative sum over an axis cut across 2, 4, 8 and 16 software devices made
  5, 7, 8 and 7 programs, against rune's promise that they do not grow with
  the devices (D170).
- **Pinned by:** the Codegen suite: `programs are kept › kernels share a
  key where they render one source, as a sum of ranges in either order
  does`; rune's `Rune.jit over several devices › maps over a cut axis ›
  partials give eager's bits, their programs do not grow`.

## D198. A store that reads its destination is staged only where it reads it elsewhere

- **tinygrad:** `schedule/indexing.py` `realize_store_after_src` (stores the
  value of every store whose value reaches its destination's buffer, for
  hazards like `TestAssign.test_assign_double_diamond_reduce`).
- **tolk:** no `realize_store_after_src`; `Rangeify.fix_store_hazard`
  decides within a kernel, and `Schedule.create_schedule` across kernels
  (D160).
- **Differs:** a store whose value reads its destination's storage is not
  staged for that alone. Within its kernel, whose iterations run in
  parallel, the value is staged where it reads the destination at another
  index than it writes (`fix_store_hazard`, as tinygrad); a store into a
  broadcast, which writes an element on several trips, is refused (D345).
  A value that reads its destination only at the element it writes, as an
  optimizer step `p := p - lr * g` or a loop's carry does, loads each
  element before it stores it, a data dependency of the store: one kernel.
  Across kernels, another kernel reading the destination's old state runs
  before the store, or reads a copy where no order does (D160): the double
  diamond, each store reading through a sum what the other overwrites, is a
  cycle that one copy breaks.
- **Reason:** (d). Kernels per step drop with bit-equal values: the norn
  bench's `nuts/schools/4`, NUTS on eight schools, whose doubling cascade's
  trip ran 17 kernels and runs 10 (fudge, counted); in tinygrad an Adam
  step's 13 kernels drop to 7. The optimizer steps are measured on fudge in
  tinygrad only until vega's Adam row, rune's MLP step row and tolk's
  `adam_mlp` rows are re-recorded; the wall-clock baselines are pending the
  scheduled re-record.
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

D199 fused independent calls of one domain into one kernel. A launch of bodies
(D404) runs the same grouping and more without one set of optimisations for the
union, as fast on CUDA, on the host and in compile time.

## D200. A tensor core's row upcast keeps a warp within a run of rows

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations`
  (a tensor core upcasts the core's M range by the first of 5, 4, 3 and 2
  that divides it).
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
  matrix. rune's routed product (`Lower_reduce`) takes blocks of 64 rows, so
  the rule fires where the block count is a multiple of 5: gpt-oss-20b's
  experts at a prefill of 536 tokens (65 blocks), kaun's
  `GptOss/experts-at-prefill-536`, take 3.02 ms a layer on fudge against
  6.96 ms without it (upcast 4 against 5), measured with D211 and D122
  withdrawn. At blocks of 16 rows no candidate
  keeps a warp of 16-row tiles within a block, so tinygrad's candidate
  stands; an upcast of 5 over blocks of 16 is not measured.
- **Pinned by:** the Heuristic suite: the `routed_run_16`, `routed_run_32`,
  `routed_run_64` and `routed_run_128` cases on each renderer, and `a core's
  warp holds rows of one run where an upcast of the core's rows keeps it in
  one`; on fudge, kaun's `GptOss/experts-at-prefill-536` row, its baseline
  pending the scheduled re-record.

## D201. A search's samples start from a clean, cold cache

- **tinygrad:** `codegen/opt/search.py` `_time_program` (asked to clear the
  L2 only where the device has `invalidate_caches`, NV's) and
  `engine/realize.py` `time_call` (invalidates NV's caches before each
  sample, and elsewhere realizes a tensor of 4 MB, which the search never
  asks for): CUDA, AMD, Metal and the host time warm.
- **tolk:** `Tolk_engine.scrub` (`device_scrub`, `scrub_kernel`,
  `cache_bytes`) and `Tolk_engine.time` (`read_scrub`); rune's
  `Compile.search`; `Rig.cache_bytes`.
- **Differs:** a search makes, beside the slots it times on, a scrub: for
  each device that runs the kernel and whose driver states its last-level
  cache's bytes, a buffer of twice those bytes, written once, and a kernel
  that reads it. Before each sample, outside its time, the kernel reads it
  and the time waits for the read. Where the scrub does not fit in memory,
  rune's search times warm. A read leaves clean lines, so the timed kernel
  finds none of its buffers cached and no line to write back. The drivers
  that state the size are CUDA, NV, AMD through amdgpu and the host; a Metal
  device and an AMD GPU over PCI are timed warm.
- **Reason:** (c): `Search.beam_search` promises each sample from cold
  caches where the device can, and rig gives no flush on CUDA, its default
  path on fudge, while AMD's L2 is invalidated on every submission and its
  memory-side cache is not; a read reaches every driver. A warm search times
  a state a model's run rarely gives a kernel whose operands other kernels
  evicted: on fudge's RTX 5000 Ada, before this change's floor, cold searches
  picked programs 1.09-1.85x faster from memory on `matmul_half_1024`. A write would leave the cache
  dirty, and a search timed after one picked a program 1.48-1.51x slower from
  a clean cache; a fill of the cache's own 64 MiB left 4 of 19 rows warm, by
  where their buffers lay, and 128 and 256 MiB evicted every row. The read costs about 0.25 ms a sample.
- **Pinned by:** the Tolk_engine suite's `timing › a time with a scrub
  reads it before the run` (the host, a software device, and CUDA, NV and AMD
  where present); rig's tests of `Rig.cache_bytes`. Whether the read leaves
  the cache cold no test times, since on a GPU other processes share a warm
  sample is not reliably warm; a bench row of a searched program would flap
  with the search's pick, so none holds it.

## D220. A loop's bound may be loaded, and no axis it reads runs as lanes or launches

- **tinygrad:** `uop/spec.py` `spec_shared` (an AFTER of a scalar or a
  computed value, a LOAD's among them, after any dependencies, which
  `spec_program` includes), which `codegen/__init__.py`
  `full_rewrite_to_sink` checks under `SPEC`, 1 by default;
  `codegen/late/linearizer.py` `pm_add_control_flow` (wraps a nested loop's
  bound in an AFTER of the range or loop end it follows);
  `codegen/opt/postrange.py` `convert_loop_to_global`, `upcastable_dims` and
  `apply_opt`.
- **tolk:** `Spec.program`; `Postrange.Scheduler.read_by_bounds`,
  `upcastable_dims`, `apply_opt` and `convert_loop_to_global`;
  `Ops.known_at_launch`, which `Codegen.split_blocks` (D80) and
  `Renderer.Estimates.of_uops` (D191) read; `Heuristic.streams`.
- **Differs:**
  - a loop bounded by a load, as `j < x[i]`, compiles: a program's Spec
    accepts an AFTER of a scalar LOAD whose dependencies are ranges and loop
    ends, the linearizer's edges, or loads, as a lane selected from loaded
    lanes is; a graph's Spec refuses one, as tinygrad's does. An AFTER of a
    load and a store is refused, where tinygrad's Spec accepts an AFTER of
    any scalar value: the linearizer orders the AFTER, not the load it holds,
    which would read before the store;
  - an axis that a range's bound reads, through a load or not (`j < i + 1`),
    is not upcastable, and an upcast of it is refused, where tinygrad makes
    the bound a vector that C compilers reject: a loop has one count of trips
    per thread;
  - a group and a tensor core are refused on a kernel where a range's bound
    reads an axis, since a group's barriers need every thread of the
    workgroup to run the same trips, and the heuristic streams no
    contraction there;
  - a weak range whose bound reads a range or memory, through a load or a
    bare index, as a scheduled kernel reads it, is neither made global nor
    split into host blocks, where tinygrad makes it global: a launch computes
    its counts before the kernel reads memory. An effect the bound is ordered
    after is no part of it. The estimates (D191) take the same test of a
    bound.
- **Reason:** (c): tolk's own interface builds such a loop, since
  `Ops.range` takes any bound and `Shape.shrink` any size, a loaded one
  included, and a graph that interface accepts must compute; without this,
  one with a loaded or row-dependent bound fails at Spec, at the C compiler
  or at launch.
- **Pinned by:** the Tolk_engine suite: `loops bounded by a loaded value`,
  on the host and each GPU present (CUDA and NV on fudge), every test. The
  Ops suite: `ranges › a launch knows a bound of constants and variables,
  ...`. The Postrange suite: `loops bounded by an axis`, and `axes.tolk`'s
  `padto_reduction_bound`. The Spec suite: `verdicts.tolk`'s `an int32 load
  after a range`, `a range bounded by a load after a range` and `a load after
  a store`. The Codegen suite: `hand-written kernels on the host › a lane of
  stored lanes keeps every lane's store` and `a loop whose bound reads memory
  stays a loop of a kernel on threads`.

## D230. A GPU kernel decodes a value R rows read at most ceil(R / 16) times

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations`
  (tensor cores at `TC_OPT` 0, upcasts of output axes by 3 or 4) and
  `codegen/opt/postrange.py` `_apply_tc_opt` (a core pads only at level 2);
  it streams nothing (D89) and states no bound on how often a kernel decodes
  a value its outputs share.
- **tolk:** `Heuristic.stream`'s upcast of the reuse ranges,
  `Heuristic.hand_coded_with`'s `core_first` and late stream,
  `Heuristic.padded_within`, `Heuristic.core_fits` and `core_rows` under
  `~decoded`, and `Indexing.core_product`.
- **Differs:** on a GPU, a decoded operand (D163) that a stream would read,
  the operand of least reuse, whose `R` outputs along one axis read it at
  one index:
  - past one row, takes a tensor core of its types before a stream, whatever
    its multiply-adds a byte (D89's gate). Past `upcast_max` (16) rows the
    core's pad of the axis the operand does not read is not counted against
    D491's 6/5, only PADTO's bound; up to 16 it is, so 12 rows, which pad
    CUDA's 16 by 4/3, stream, and 14-16 take the core on its M. 7-8 rows
    fit only CUDA's N tile of 8, which the core's three candidates reach
    over one reduce range, as Q8_0's partial form, or over two where K
    takes the last; tinygrad's MXFP4 graph, whose last range is a code
    byte's 2 halves, streams them where `core_fits` names a core. On
    sm_80 and sm_89 the staged tile's pad of N falls on a range of one
    trip, which staging refuses, so such a core runs unstaged. Pads of its
    own axes and of K count at any reuse;
  - with no such core, streams up to 16 rows in one pass, past them in
    `ceil(R / 16)` passes, the axis padded where they do not divide it.

  `core_fits` and `core_rows` take whether each factor is decoded, in the
  product's order, and decide by the heuristic's own rules
  (`Heuristic.first_streams`, `core_from` and `pad_shared`): a core past 1
  where the factor that streams is decoded, the first at a tie, and the pad
  of an extent past 16 that no decoded factor reads uncounted, so that rune
  writes such rows of a Q8_0 matrix in the partial form a core takes, and
  `Indexing` gives a product of a decoded factor a kernel of its own where
  the core takes it (D137).

  A core decodes a value once a tile of its rows where its floor (D175)
  refuses their upcast, and once in each lane its fragment holds it (two on
  RDNA3). The host shares its decodes as its registers allow (D92).
- **Reason:** (d): each pass decodes and reads the operand again, and a
  padded core decodes it once; on fudge, MXFP4 products of 17 and 32
  bfloat16 rows run on the core at under half the stream's time, which
  tinygrad's heuristic exceeds 60-fold. Up to 16 rows a stream decodes once
  too, but multiplies each row on its lanes where a core does it in one
  instruction: 16 rows of MXFP4 and Q8_0 run 0.53 and 0.88 of the stream's
  time on the core.
- **Pinned by:** the Heuristic suite's `a GPU kernel decodes a value R rows
  read at one index at most ceil(R / 16) times its core's copies` (1 to 512
  rows of bfloat16, half and float32 on the CUDA, RDNA3, RDNA4, CDNA and
  Metal renderers), `a decoded product's optimisations keep its writes`
  (17 and 19 rows on a core, 64 float32 rows streamed), `core_fits names
  the decoded products the heuristic takes a core of` and `core_fits names
  the products of stored and decoded factors the heuristic takes a core of`
  (both factor orders, ties, one and two decoded factors); the Rangeify
  suite's `get_kernel_graph › products a core takes`, `products of 24 rows
  by codes of 64 run a kernel each` to `products of codes of 17 by codes of
  64 share one kernel`; the Heuristic suite's `a decoded operand of up to
  16 rows takes a core that holds them` (12 and 16 rows on five
  renderers), `core_fits names the decoded products the heuristic takes a
  core of` (1 to 512 rows, 7-8 on CUDA streamed where it names a core) and `a contraction streams before a core up to the
  renderer's bound a byte` (decoded blocks of 16 on a core at bounds of
  63 and 72); rune's GPU suite's `a
  Q8_0 product's rows take a tensor core where core_fits holds`; kaun's
  `Quant/cuda/mxfp4-product-of-16384-{7,8,12,14,16,17,32}-bfloat16-tokens`,
  `-64-float32-tokens` and
  `q8_0-product-of-16384-{7,8,12,14,16,17,32}-bfloat16-tokens`.

## D235. A tensor core's lanes are a warp of its threads

- **tinygrad:** `codegen/gpudims.py` `group_gpudims` (lays a kernel's warp
  and local ranges on the thread axes whatever its tensor cores need), and
  `codegen/simplify.py` `mark_range_mod` and `pm_split_ranges` (split a
  local range, never a warp, by its divisions, the most significant part
  taking the first thread axis).
- **tolk:** `Gpudims.group_gpudims`, through `check_lanes`.
- **Differs:** a kernel holding a tensor core of more than one thread raises
  `Invalid_argument` unless its first thread axis is a warp range of that many
  threads. A product's fragments are laid out by the hardware's lane, the
  thread's index within its warp, so the kernel's own lane index must be it,
  which only a warp range on the first axis gives. tinygrad's optimiser makes
  that range; a kernel written by hand may number its lanes with a local
  range instead, and tinygrad then computes another product: a local range
  of 128 threads read as `t // 32` and `t % 32` splits into those two parts,
  the warp's index takes `threadIdx.x`, and each lane reads another lane's
  elements.
- **Reason:** (e).
- **Pinned by:** `Tolk.Gpudims › group_gpudims › a tensor core's lanes › are
  refused from other ranges` and `› are a warp of its threads on the first
  thread axis`; on CUDA, `Tolk.Codegen › tensor cores written by hand › a
  warp per product, lanes a warp range, gives each warp's exact product`.

## D236. A bitcast that changes a vector's lane count stays whole

- **tinygrad:** `codegen/__init__.py` `do_stack_wmma` (replaces a tensor
  core's operand by the stack of its lanes) and `uop/movement.py`
  `mop_cleanup` (folds that stack back to the operand), so that a bitcast
  operand that changes the lane count, such as four words read as eight
  bfloat16, rewrites forever; `renderer/cstyle.py`
  `CUDARenderer.string_rewrite` (renders a bitcast with its scalar types, so
  that a bitcast of a vector does not compile).
- **tolk:** `Codegen.do_stack_wmma`, which keeps such a bitcast whole;
  `Cstyle`'s base rewrite renders a bitcast with each side's vector type on
  CUDA, which has no rule of its own.
- **Differs:** a tensor core takes such a bitcast as the vector it is, and
  CUDA renders it `tg_bitcast<nv_bfloat168>((uint4)(...))`. A scalar bitcast
  renders unchanged. tinygrad does not terminate on these kernels; tolk
  computes the right product.
- **Reason:** (e).
- **Pinned by:** `Tolk.Codegen › tensor cores written by hand › a tensor
  core's operand bitcast from words lowers for CUDA`, which needs no GPU; on
  CUDA, `› a warp per product, lanes a warp range, gives each warp's exact
  product`, whose rows with words read each fragment as words bitcast to
  bfloat16, and `› a bitcast of words to bfloat16 lanes takes each word's low
  half first, on CUDA`, and on the host.

## D237. Ends of one range that reach each other run apart, each a loop

- **tinygrad:** `codegen/__init__.py` `merge_reduce_ends` (merges the ends
  that close the same ranges in one context, and leaves every end of those
  ranges unmerged when one reaches another: a copy of a tile reads, through
  the scores, what the copy before it staged, and merged would contain
  itself); unmerged, the ends share their ranges, and over a loop range the
  linearizer refuses the kernel (`codegen/late/linearizer.py`
  `CFGContext.__init__`).
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
  them, which tinygrad does not (its linearizer then refuses the kernel); over threads, `lanes and threads › an end reading the stores of an
  end of its threads runs after it on CUDA`.

## D238. A tensor core says whether its accumulator chains

- **tinygrad:** `renderer/tc.py` `TensorCore`, whose fragments hold the
  fact and whose comment states that the k bits may be permuted, identically
  in A and B, with no function that answers it.
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

- **tinygrad:** `uop/symbolic.py` `symbolic`'s ordering rule: an `AFTER`
  keeps among its dependencies only effects, ranges, ends, back edges,
  linears and stages, and replaces any other by its sources, an `AFTER`
  included. `x.after(y')`, where `y'` is `y` once assigned, becomes `x` after
  `y`'s kernel, which `schedule/__init__.py` `create_schedule` reads as a
  state of `x` that `y`'s kernel writes.
- **tolk:** `Shape.symbolic`'s ordering rule; `Ops.is_join`;
  `Prepare.sequence`; `Schedule.value_state`, `Schedule.swap` and the waits
  of `Schedule.schedule`; `Rangeify.unjoin` and `Rangeify.handle_after`.
- **Differs:** an `AFTER` of global storage stays among an `AFTER`'s
  dependencies. An `AFTER` whose dependencies are all `AFTER`s is a join:
  it names no state of its own and holds the value of the state under it.
  `Prepare.sequence` replaces a join by that state (D473): its reader reads
  the state and waits for the state's writers, and the effects the join's
  dependencies reach still run, where the states they name order them, and
  an effect that reads the join runs after them, so it is no second write
  over a state they made (D474). A write over a join is a write over
  that state. In a kernel graph `Schedule.create_schedule` is given, a
  source read through joins reads the state under them, and its reader
  waits for that state's writers and for what each join waits for; the copy
  of a read through a join goes under the join, which keeps its waits; a
  kernel's parameter for storage read through a join is the state's, its
  call's argument waiting for the join's dependencies. Afters of registers
  and local memory still flatten, so no kernel's source changes: the
  goldens of every module regenerate unchanged.
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
  `read_join_of_overwritten`, their recorded schedules, linears and values
  (`create_linear_with_vars › values`).

## D250. Two stores of one value into one storage are two states

- **tinygrad:** `schedule/prepare.py` `earliest_rewrites` (removes two
  STOREs that store the same thing to the same place, for
  `TestSchedule.test_dedup_assign`), which rewrites `a1.after(a1.store(v))`,
  where `a1` is `buf.after(buf.store(v))`, to `a1`: the program's second state
  becomes its first. A loop of a call that writes the second state and reads
  the first then takes one state at both arguments, and its schedule passes
  the call the storage at both, so the loop would read what each trip left
  (tinygrad renders no loop of a call, and runs none). Without the rule, the loop runs after the second
  store and before the overwrite of the first, a cycle, which tinygrad
  refuses (`schedule/__init__.py` `create_schedule`).
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

## D255. A kernel whose loop's bound reads memory is not searched

- **tinygrad:** `codegen/opt/postrange.py` `apply_opts` (searches every
  kernel under BEAM, timing it on `args_from_ast`'s new buffers, their
  contents unspecified, and each variable at the middle of its bounds).
- **tolk:** `Postrange.apply_opts` (`bound_by_memory`), `Ops.bound_loads`.
- **Differs:** a kernel with a loop bounded by memory, read through a load
  or a bare index, as `j < x[i]` (D220), takes the heuristic's optimisations when a search is asked for, as
  without one. tinygrad refuses such loops.
- **Reason:** (e), following D220: the trips such a loop runs are what
  memory holds, which no sample the search makes knows. On unspecified bytes
  a search timed up to 2^31 - 1 trips a thread for an `int32` bound, each
  candidate for minutes or more, measured whatever the memory's last user
  left, and kept its choice in the disk cache keyed by the kernel alone. A
  call's own values are not the search's: it runs while the program
  compiles, and its choice serves every later call. Zero bytes usually give
  no trips, so the search would rank the kernel without its loop. No stated
  count fits every bound: the middle of an `int64` bound clamped at 0 is
  2^62.
- **Pinned by:** `Tolk.Search on the host`: `a kernel whose loop's bound
  reads memory is not searched, and takes the heuristic's optimisations`,
  through a load and an index.

## D210. A GPU merges bfloat16 accesses as float16's

- **tinygrad:** `codegen/late/coalesce.py` `memory_coalescing` (a GPU merges
  adjacent accesses of `float`, `half`, `int`, `uint` and the 8-bit floats
  into vectors of 4 and 2, and leaves `bfloat16` scalar; the same at
  `6b7b018da`); `renderer/cstyle.py` `CUDARenderer.render_kernel` (its
  2-lane `bfloat16` vector, never merged there, has no constructor: it
  renders `make_nv_bfloat162`, which `cuda_bf16.h` names `make_bfloat162`).
- **tolk:** `Coalesce.memory_coalescing`; `Cstyle.cuda_kernel`.
- **Differs:** on a renderer other than the CPU device's, adjacent `bfloat16`
  accesses merge into vectors of 4 and 2 as `float16`'s do, and the CUDA
  renderer defines `make_nv_bfloat162` as `make_bfloat162` when a kernel holds
  a 2-lane `bfloat16` vector. The host is unchanged: its `bfloat16` accesses
  stay scalar. tinygrad states no reason for leaving `bfloat16` out, and its
  CUDA kernels could not build its 2-lane vector, a defect of its own.
- **Reason:** (d): a tensor core's `bfloat16` operands, each loaded as a
  2-byte scalar, made every bfloat16 product on fudge's RTX 5000 Ada about 1.5
  times slower than its float16 product of the same optimisations: 2048
  cubed took 480-482 us, and takes 310-325. Metal (`bfloat2`) and AMD (an
  `ext_vector_type` of `__bf16`) render the same merges, unmeasured, as fudge
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
of M and N. Staging (D262) took the contraction rows it was measured on, and on
gpt-oss-20b's routed experts, what it still touched, it cost (3.54 ms a layer
against 2.77).

## D212. A contraction on a GPU's registers holds 8 by 8 outputs a thread

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (the
  upcast loop takes each axis once, by 3 or 4, 3 first where both divide it,
  while fewer than 32 lanes are upcast; the unroll needs fewer than 64; the
  locals stop at 128 threads; the same at `6b7b018da`).
- **tolk:** `Heuristic.four_first`, `Heuristic.block`, `Heuristic.upcast_more`,
  `Heuristic.along_reduce`, `Heuristic.split_reduce`,
  `Heuristic.sector_bytes`, `Heuristic.split_trips`,
  `Heuristic.blocked_threads`, `Heuristic.locals` and `Heuristic.general`.
- **Differs:**
  - on a renderer with local indices, a contraction summed in 4 bytes or
    fewer whose upcast loop took two axes, each by 4 and leaving part of its
    axis, each unread by some operand of the product (a load after the
    reduce, as a bias, is none), where 2 divides both and what is left
    of them holds a SIMD group of threads in powers of two, unrolls its last
    unrollable axis by 4 where 4 divides it, and by nothing otherwise;
  - it takes a second upcast of each axis by 2, so a thread holds 8 by 8
    outputs, where those threads number at least two workgroups of 128 on
    each of the target's `units`, or are not known. Its workgroup takes
    tinygrad's bound of 128 threads, in D486's order, the later axis's split
    first as `lidx0`: the local loop's split where it lays a power of two of
    at least 2 threads along the later axis, else 16 or 8 there, whichever pads it less, and the rest
    along the other, each axis padded (PADTO) where they do not divide it,
    unless a pad is refused;
  - where one of the two upcasts took its axis whole, or left an odd part
    of it, the other output axis that the same operands of the product read
    takes its second upcast, where 2 divides it; an odd part with no such
    axis is padded (PADTO) to even and takes it, as whisper's 1500 frames
    leave 375 runs of 4;
  - where these rules block no contraction without padding, they run
    again with each output axis padded (PADTO) within `core_waste` (6/5)
    of its size, and block where they block so: a part 4 does not divide
    counts as padded to 4 for the 4 before 3 below, and the upcast loop
    pads it to 4 before taking 3; what is left of each axis holds its
    threads padded to them, which the workgroup's split pads, taking the
    local loop's split only where it fills a SIMD group;
  - where both operands of the product read the unrolled axis at unit
    stride, as a product by the transpose of a matrix stored `[outputs;
    inputs]` does, and so every float32 input gradient `dY Wᵀ` whatever
    `W`'s layout, attention's scores `q kᵀ` as a product, and on sm_75 a
    bfloat16 product no core takes, on a renderer that copies nothing
    asynchronously (on one that does, D497's staged tile takes it), what the
    unroll leaves of it splits, blocked or not, into the first local threads
    (`lidx0`), as many as read in their runs of 4 a 32-byte sector of the
    widest operand's row, 2 for float32 and 4 for bfloat16, halved while
    they do not divide it or leave each fewer than 16 trips; their partials
    meet through shared memory, as a group's do, and the reduce stays whole
    where they would not fit it: a slot for each output lane of each sum
    the split closes, for each of the workgroup's 128 threads; The later axis then takes
    the largest power of two of threads whose square the SIMD group's other
    lanes hold, 4 on CUDA, and the other the rest of the workgroup's 128,
    each padded as above, where tinygrad's 16 lanes across the columns each
    load 16 bytes of a row of their own;
  - a kernel upcast past those two axes' 16 lanes, as the masked upcasts
    take a materialised mask's axes whole, is not blocked, nor a sum in 8
    bytes;
  - only for such a contraction, upcast by nothing before the loop, whose
    blocked threads, a 64th of its outputs, number at least the renderer's
    `core_threads`, does the upcast loop take an axis by 4 before 3; of an
    output axis split into parts that the same operands read, as a
    projection's heads and their width, the innermost part counts there,
    which the loop upcasts and `block` doubles.
- **Reason:** (d): at equal registers, 8 by 8 outputs a thread in workgroups
  of 128 beat tinygrad's 4 by 4 on fudge's RTX 5000 Ada wherever each SM
  holds two of them, which at the 255 registers a thread may take fill its
  64K. The unroll by 4 on both sides of that floor keeps each output's sum
  in one order whatever `units` a target states, which moves speed alone,
  and so does the split of a reduce both operands lie along. That split
  serves TESSERA's students loaded from PyTorch's `[outputs; inputs]`
  weights, whose products ran at 8.3-8.7 TFLOP/s against 12.8-13.9 with the
  same weights in C order; its own gain is to measure on fudge (rune's
  `Cuda/matmul-float32-transposed-16384x440x1320` against
  `Cuda/matmul-float32-16384x440x1320`). A sum of three float32 products
  along their sum, blocked, holds 96 KiB of partials, past every target's
  workgroup: split, it failed the compile on sm_75, gfx1201 and gfx942,
  where no staged tile takes it first. A second upcast through the other
  part of an axis serves TESSERA's second embedding, whose day encoding
  splits its 440 columns into 220 pairs: blocked and staged (D497) it takes
  2.9 ms a call of 163840 rows on fudge's RTX 5000 Ada, against 17.1 at 4
  by 4 in 64 threads (rune's
  `Cuda/matmul-float32-transposed-pairs-16384x440x440`). The pad of an odd
  part, and the 4 before 3 of a projection into heads, serve whisper's
  encoder, whose products over 1500 frames ran at 4 by 3 from the L1, at
  4.1-4.5 TFLOP/s: staged, they run at 21.7-27.3, and large-v3-turbo's
  products take 85 ms a call against 446 (DEBUG=2). An encoder call takes
  114-116 ms against 478-487 for large-v3-turbo, 19.8-21.3 against
  64.1-65.6 for small and 4.7 against 5.7-5.8 for tiny, three alternated
  runs on fudge's RTX 5000 Ada (whisper's `bench_whisper.exe --recording`),
  where openai-whisper's PyTorch takes 111 ms for large-v3-turbo. The
  padded pass serves AION-1's projections into heads and their merges,
  which keep a batch's objects and their tokens apart, two axes that merge
  in no index: 858 tokens, which 4 does not divide, ran 4 by 3 at
  2.8-3.0 TFLOP/s; 580, whose 145 runs of 4 the batch doubles, 4 by 4 at
  2.5; 383 and 581 at 5.1-6.8. Blocked and staged they run at 22-28, 16
  objects of 858 tokens into 12 heads of 64 over 768 taking 0.58 ms
  against 5.49 (DEBUG=2, two alternated rounds), and an embed of 16
  objects with aion-base takes 172 ms a call against 370 (DEBUG=2), where
  AION-1's PyTorch takes 257. A layout that blocks without pads keeps its
  kernel: with the pads first, TESSERA medium's projections into 4 heads
  of 110 unstaged ran 7-10% slower padded to 112 than lending the heads'
  second upcast to the width.
- **Pinned by:** `Tolk.Heuristic`'s `a contraction in registers` cases
  (`matmul`, `matmul_256`, `scores`, `matmul_768`, `batched_768`,
  `two_batches_768`, `matmul_440`, padded, `heads_projection` and
  `heads_projection_biased`, and with D495 `batched_768_biased`,
  `scores_biased` and `scores_biased_first`, blocked on CUDA and AMD,
  which state no units; on sm_89's renderer `heads_projection` blocked and
  `matmul_768`, `batched_768` and `two_batches_768` held at 4 by 4;
  `two_batches_768_biased` and `batched_768_biased`, whose bias the
  batches do not read, optimised as without it; `matmul_1000` blocked
  padded to 1024; `matmul_48` and `scores_masked` not blocked; `a reduce
  split into threads keeps its partials within shared memory`, three
  products on sm_75 and gfx942),
  `a batch's projection into heads` (16 objects of 858, 580 or 383 tokens
  and 8 of 581, staged on sm_89 and 8 by 8 a thread on sm_75, and a
  product of 88 by 88 that holds no SIMD group even padded), rune's
  `a batch's projections into heads, and their merge, its tokens padded,
  plus a bias, are within their bound`, `a product laid out along its reduce`'s
  `16384 by 220 pairs over 440`, `1500 by 1280 over 1280` and `1500 by 20
  heads of 64 over 1280`, rune's
  `Cuda/matmul-float32-transposed-pairs-16384x440x440` and
  `Cuda/matmul-float32-transposed-1500x1280x5120` on fudge, and
  `a contraction laid out along its reduce` (`matmul_256_kt_128`
  with its reduce split on CUDA, AMD and sm_75; `matmul_256_kt` and
  `scores`, under 16 trips, and `matmul_256` and `matmul_440` without),
  with the layouts in the suite's `.tolk` files, and `a kernel's sums keep
  their order whatever units its renderer states`; rune's `test_jit_gpu`
  `a blocked product padded to its threads, plus a bias, is within its
  bound` and `a projection into heads, blocked over their width, is within
  its bound` and `a projection of 1500 frames into heads, its frames padded,
  plus a bias, is within its bound`, blocked on sm_89; the tolk bench's
  `contract/f32-2048` and
  `contract/qk-f32-2048` rows and rune's
  `Cuda/matmul-float32-16384x440x1320`,
  `Cuda/matmul-float32-transposed-16384x440x1320`,
  `Cuda/matmul-float32-transposed-16384x64x1320`,
  `Cuda/matmul-float32-input-gradient-16384x1320x440` and
  `Cuda/matmul-float32-heads-16384x440x4x110` on fudge, their baselines
  pending the scheduled re-record.

## D221. A part of a pattern that names nothing gives one naming

- **tinygrad:** `uop/ops.py` `UPat.match` (extends its result with a store
  for each ordering of each source pattern, so a store comes back once for
  each way a part that names nothing matches).
- **tolk:** `Ops.Upat.first`, `Ops.Upat.alternatives` and so
  `Ops.Upat.match_` and `Ops.Pattern_matcher.rewrite`.
- **Differs:** ways of matching that differ only within a part of the
  pattern that names nothing give one naming, where tinygrad gives the same
  store once for each. A rule's function is called once for that naming,
  where tinygrad calls it again after it declines. A rule's result is the
  same when its function's result depends only on the naming.
- **Reason:** (d): constant folding's rules, repeated over every source of
  an operation whose sources may each be a constant or a stack of them,
  enumerate orderings that name nothing. On fudge, a naming for each
  ordering makes rune's first compile of float64 `betaincinv`'s gradient
  (the compile bench's `special/betaincinv-grad-f64`) allocate 216.3 M words
  where one naming allocates 190.2 M.
- **Pinned by:** the Ops suite: `Pattern_matcher › a part of a pattern that
  names nothing gives one naming`; the tolk bench's `sort/kernel_graph-fresh`,
  `gpt_oss_prefill/kernel_graph-fresh` and `gpt_oss_prefill/codegen-fresh`
  rows.

## D240. The host's output upcasts take no floor of elements

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations`
  (upcasts output axes while the upcastable axes hold 1024 elements).
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
  launch's blocks (D80) would each repeat the load. Against the 1024 floor,
  on one of fudge's P-cores, `gpt_oss_kv` took 1508-2293 µs, and takes 89-98.
  A floor counted in the split loop's iterations, at least 16 or the blocks
  the work fills, was slower than none on every kernel it bound. Measured
  under load only: whether a memory-bound product of few outputs loses its
  blocks on a quiet host is for the scheduled quiet re-record. The cost: the
  compiler's own passes allocate 1.3-2.9 times the words on the tolk bench's
  `linearize`, `render` and `warm` rows of its recorded programs.
- **Pinned by:** the Heuristic suite's `the host's lanes › on AVX2 a
  recorded host kernel of few outputs takes its lanes` and `the
  optimisations chosen are tinygrad's › … › the optimised kernel`, the
  `cpu` and `avx2` cases, among them `matvec_tall`, `vecmat_long` and
  `column_sums`; the tolk bench's `host/gpt_oss_kv/heuristic`,
  `host/vecmat_1000/heuristic`, `host/matvec_tall/heuristic` and
  `lorenz/r_32_3_2_4_3_400/heuristic`, their fudge baselines pending the
  scheduled re-record.

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
  64 for the unroll. Clang computes at most 64 lanes of a vector operation
  at once (D141's `Renderer.vector_lanes`): an operation of more lanes is
  vectors of 64 and a last of the rest, and the last 65 lanes are vectors of
  33 and 32, since a rest of one lane would be computed apart, which made
  Clang's loop strength reduction take seconds (D141).
- **Reason:** (d), one P-core of fudge's AVX2:
  - the conversions' bound: without it `gpt_oss_kv` takes an unroll of its
    sum and 408-429 µs against 218-228, on gpt-oss's decode path; it costs
    `conv_half`, a kernel of the Heuristic suite, 1.27 times. The lanes stand
    for the unroll's cost, whose cause is not measured;
  - the iteration's bound: Clang's time grows much faster than the lanes
    where they put together vectors from many loaded rows. rune's
    contraction of 8 groups (`r_23_4_16_4_16`, 16 by 4 output lanes by its
    16 rows unrolled whole) took Clang 5.3-5.6 s, and at 512 lanes takes 1.1
    s, runs as fast and keeps its rows' loads out of its output loop, which
    an unroll by 4 or none would load four times, past the byte bound of
    rune's `Lower_reduce › work › a Contracting contraction of 8 groups works
    in blocks` (73,600 estimated bytes against 153,088). It costs the Heuristic suite's `conv`, whose window
    unrolls by 3 alone, 1.12 times. 256 lanes would take the routed MXFP4
    decodes (`decoded_rows_*`, gpt-oss's experts) from their unroll by 8 to
    4; 1024 took three of rune's products from 16 output lanes unrolled whole
    to 32 lanes and no unroll, 2724-2726 µs against 2301-2310. With the vectors of 64 the bound still stays: without it a layer's
    float32 matmul goes to 64 by 32 lanes and rune's
    `Jit/grad-of-jit-mlp-float32` takes 215-226 µs against 201-204;
  - the vectors of 64: rune's compile bench row `products/segments-grad`
    takes 0.53-0.56 s cold against 1.45-1.46 s with run times held. Vectors
    of 16 made the compile bench's `lanes/held-fp8-36` 1.30-1.36 times as
    long.
- **Pinned by:** the Heuristic suite's `the host's lanes › on AVX2 a
  recorded host kernel's conversions take their lanes` and `› on the host a
  conversion computes at most 64 lanes`, over every recorded kernel and
  lorenz_simple's on four register files, and the host cases of `the
  optimisations chosen are tinygrad's`; rune's compile bench row
  `products/vecmat-bf16` and the tolk bench's `host/gpt_oss_kv/heuristic`,
  their fudge baselines pending the scheduled re-record. The iteration: the
  Heuristic suite's `the host's lanes › on the host an iteration computes at
  most 512 lanes`, over the same kernels and rune's contraction of 8 groups
  (`contracting_groups.graph`), `› on AVX2 an output upcast leaves room for
  the whole unroll of an axis of at most 32` and `› an output upcast leaves 4
  lanes for the unroll of an axis past 32`; rune's `Lower_reduce › work › a
  Contracting contraction of 8 groups works in blocks`; rune's compile bench
  row `products/segments-grad` and the tolk bench's `host/conv/heuristic`,
  their baselines pending the scheduled re-record. The vectors: the Heuristic
  suite's `the host's lanes › on AVX2 a recorded host kernel's program
  declares vectors of at most 64 lanes`, and the Codegen suite's `vectors in
  programs › a host operation of any lanes computes vectors of at most 64 and
  writes what its kernel writes`.

## D260. A tensor core accumulates its running sum at the reduce

- **tinygrad:** `codegen/__init__.py` `pm_wmma_add` (an add of a WMMA, of a
  permute of one, or of a permute of a reshape of one, folds the addend into
  the core's accumulator) and its use in `pm_expand_broadcast` and
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
  opts' unroll or with shared-memory staging. On Ada's cores, which
  truncate, the fold's error is 0.497 of `Nx.contract`'s bound on rune's
  adversarial rows (D29), within the bound. A core accumulating two steps a
  trip beat the same kernel unfolded by 1.06 to 1.14 on bfloat16 products of
  512 by 512 by 2880, 512 by 4096 by 2880 and 2048 cubed on fudge.
- **Pinned by:** the Codegen suite, `tensor cores on CUDA › a product summed
  in float32 has the same bits whether the core takes one or two steps of K
  a trip` (fails unfolded), `tensor-core accumulators › the running sum
  replaces a zero accumulator` (fails unfolded, on every target's cores),
  and `stages › … › matmul_{half,half_big,half_tile,bf16_tile,tc}_{cuda,hip}`
  and `matmul_tc_unroll_metal`; the cstyle and renderer goldens of each core.

## D261. 16-bit floats and bytes copied into shared memory merge to 16 bytes

- **tinygrad:** `codegen/late/coalesce.py` `memory_coalescing` (a GPU merges
  `half` accesses into vectors of 8 under `ALLOW_HALF8`, an environment
  setting off by default, and of 4 and 2 otherwise; it merges no GPU bytes).
- **tolk:** `Coalesce.memory_coalescing`; `Renderer.t.half8`, set by
  `Cstyle.cuda` and `Cstyle.hip`; `Renderer.t.byte_words` and `Cstyle`'s
  `word`, CUDA's. `Setting.allow_half8` is withdrawn.
- **Differs:** on a renderer that writes vectors of eight 16-bit floats (CUDA
  and HIP; Metal's vectors hold at most four), adjacent `float16` or
  `bfloat16` accesses of a group every access of which copies from global
  into local memory (loads of global memory whose only use is the value of a
  store into local memory, or those stores) merge into vectors of 8, 4 and 2,
  16 bytes at most; every other access of them merges into 4 and 2, as
  float16's do in tinygrad without `ALLOW_HALF8` (bfloat16 by D210). No
  setting changes it. On a renderer that writes bytes as integer words
  (CUDA), `uint8` accesses of such a group merge into 16, 4 and 2, and 16
  bytes print as four 32-bit words (`uint4`), read and built by shifts, as
  2 and 4 bytes print as one word.
- **Reason:** (d): a tensor core's operands staged through shared memory
  (D262) are copied 16 bytes a thread, which PTX's `cp.async` copies in one
  instruction (D263); 8 bytes at a time they take twice the copies and no
  asynchronous one. A decoded operand's codes and held scales (D262) are
  copied the same way, one `cp.async` each, where 4 bytes take four copies
  and no asynchronous one. Elsewhere 16 bytes gained as often as they lost:
  merged everywhere, they made gpt-oss-20b's attention product of
  probabilities by values 785-791 us against 562-564 on fudge. On RDNA4 the
  rule leaves unstaged operands alone: a lane's operand of the
  16-by-16-by-16 core is two runs of four 16-bit values, eight elements
  apart, so 8 bytes is the widest contiguous load. On nonnormal's R9700
  (gfx1201), float16 and bfloat16 products of 2048 cubed, weights in either
  layout, staged and unstaged, compile to the same HIP source and the same
  comgr instructions with every access merged to 16 bytes. RDNA3 and CDNA are unmeasured.
- **Pinned by:** the Stage suite's `values › a staged dense MXFP4 product
  …` (its codes copied as `uint4`); the Coalesce suite's `vectors of eight
  16-bit floats` group (`the runs of eight 16-bit floats' loads and stores`,
  of `float16` and `bfloat16`; `half_cuda`, `half8_cuda` and
  `accesses_cuda`, generated from tinygrad with this rule); the Stage
  suite's `sources` (`cp.async`); kaun's `GptOss/prefill-512`.

## D262. A tensor core's operands are staged through shared memory

- **tinygrad:** none: no pass stages an operand. Its `OptOps.LDS`, one opt a
  buffer, was reverted (#9669, #9691), and its hand-written kernels stage by a
  scheduler `STAGE`, which skips the heuristic. `uop/spec.py`
  (`spec_shared`: an AFTER of a scalar or a computed value only).
- **tolk:** `Stage.operands`, which `Codegen.full_rewrite_to_sink` runs after
  the opts, a search's candidates' included (D489); `Stage.depth`,
  `Stage.layout`, `Stage.tile` and `Stage.steps`;
  `Heuristic.staged_tile`; `Chains`, the blocks of a sum's running sums
  that codegen and the tile both read; `Renderer.t.async_copy`, `units`
  and `in_flight`; `Setting.stage`.
- **Differs:**
  - **Tiles.** After the opts, each core operand that reads global memory
    through an index whose terms reading the workgroup's threads and lanes
    form a dense box of strided dimensions is copied by every thread of the
    workgroup into shared memory, 16 bytes a copy where its innermost
    dimension is contiguous, in rows whose 16-byte chunks are permuted by an
    XOR with the row (`line` 128 bytes of 32 banks, `copy` 16 bytes,
    generalized to rows of any even count of chunks); the core reads its
    fragments back through the same layout.
  - **Computed operands.** An operand the kernel computes elementwise from
    loads and does not decode, as an attention output times the share of
    each head's row its sink leaves, has its loads copied where each through
    which it reads the threads and lanes has such a tile: those whose tiles
    step through the loops of its tile of most elements, where those are the
    loops of each operand that is a load; the others, as the per-head share
    while the core's loop runs within a head, stay loads, and the operand
    computes on the copies. A tile's loops are its serial ranges of more
    than one trip; a range of one trip, as K unrolled whole leaves, is no
    tile's loop.
  - **Stages.** Two stages where the renderer copies asynchronously and they
    fit `shared_max`, the next trip's copy issued before this trip's reads
    and ordered so by the slot's offset, an AFTER of the trip at two trips,
    and the first trip's copy pinned, by an AFTER of their ranges, inside
    every serial loop the outermost loop runs in, so that each run of the
    loop copies it; one stage otherwise. A tile whose index steps through a
    nest of loops copies the next trip of the nest in two stages where one
    node closes the loops, their extents are constant and it closes no
    other loop of more than one trip inside the outermost: its trips are
    counted outermost first, the next trip's loops being its digits in their
    extents, the outermost without a remainder, so that the trip past the
    last fails the copy's gate; a tile whose loops are no such nest copies
    in one stage. The barriers are those the implicit rules place. A gated
    operand (PADTO's validity, or a core's selection of 0) copies 0 where
    its gate fails, its thread ranges rebuilt from the tile's coordinates.
  - **The staged tile.** A staged core takes its own tile: 2 by 2 warps of
    64 by 64, the warp tile halved, M then N, to 32 by 32 while the kernel
    keeps fewer than `core_threads / stages` threads, its stages planned as
    `Stage.operands` copies them, at the types its loads copy and beside the
    kernel's earlier cores' stages. Past that plan, where the renderer
    states `units` and `in_flight`, the core has two stages and the kernel's
    workgroups are known, it goes down to the core's own tile among plans
    whose accumulators a thread blocks (at most `Chains.held`, 320 bytes):
    the plan whose busiest compute unit copies the fewest bytes a value of
    K, ⌈workgroups / units⌉ workgroups at the bytes staging copies. K is
    unrolled to the least depth from 32 at which the busiest unit's copies
    of a trip reach `in_flight`, among the depths that keep the stages and
    the blocks of the sum's running sums where K is the sum's innermost
    range (`Chains`, whose walk codegen's blocks are), or the deepest that
    keeps them; K is unrolled to chunks of 32 elsewhere, and never split
    among a workgroup's warps. `in_flight` is Volkov's estimate of the bytes
    in flight that sustain nine tenths of a peak: 1.3 times a copy's least
    trip from memory on a small grid, 0.32 us, times one SM's peak fill from
    one workgroup's copies, 44 GB/s, 18304 bytes; `units` is the RTX 5000
    Ada's 100 SMs. Both are fixed for sm_89 and 0 elsewhere, which keeps
    the plans above. No sum's order follows the units: every plan the rule
    takes blocks its sums as the plan above does, and every depth keeps the
    blocks, so each output sums the same terms in the same order, and a part
    of another count runs the same programs, at another speed. Where no
    plan's upcast of N divides it, N is first padded to the full warp tile's
    upcast (D491). Its rows stay within a run of rows that read one value of
    an operand, a warp's tiles of rows falling by one until they fit it, so
    that a run of three tiles takes one warp, and the heuristic takes it only
    where `Stage.admits` the kernel as the core leaves it, so a core staging
    refuses keeps the unstaged splits.
  - **Gathered rows.** One dimension of a tile may be rows gathered through
    a row map: a term whose only read of the box's ranges is one load of
    global memory, whose own index's terms form one dimension of the box,
    as a routed product's blocks read their rows (`rowmap[block, r] * k +
    c`), or a weight read at each block's expert (`owner[block] * k * n +
    c`). Each copy loads its row's entry and copies from that row, 0 where
    the gate on the entry fails (a pad row), and the core reads the row back
    at its place in the map. Such a tile copies in one stage, so it is
    staged only where several warps read it, where a local range does not
    move its index. A core reading gathered rows keeps the unstaged splits.
  - **Decoded operands.** A decoded operand's loads (codes cast to floats
    through integer arithmetic) are copied where the renderer copies
    asynchronously and every copy of each is one 16-byte `cp.async`: its
    innermost dimension contiguous in whole copies, every other term of its
    address and its storage's start multiples of 16 bytes, its index split
    after `Symbolic.sym`, which reads MXFP4's scale index `(32 L + i) / 32`
    over lanes `i < 32` as the loop `L`; they copy in the stage of the
    core's other operand. A load of such an operand whose trip's copies are
    not so, whose one loop moves it by at most a line (128 bytes) over all
    its trips, and whose tile over the threads, lanes and that loop is one
    aligned run of 16-byte copies, as MXFP4's e8m0 scales, a byte a row a
    trip, is held: copied once before the loop into shared memory, in the
    room every core's stages leave, and read there each trip. The first
    trip's reads wait for the held copies at the stages' first barrier. A
    tile is held only where the operand has copied loads and each is
    staged. A core that keeps the unstaged splits has its `K` upcast by
    `Stage.steps`, so that each row staging copies a trip, where `K` steps
    the row along its innermost dimension, covers a 32-byte sector, the unit
    in which global memory is read: MXFP4's codes at 64 values a trip, where
    at 32 each sector is read at two trips. Q8_0's blocks of 34 bytes and
    GGUF's of 17 fail the copy's test and stay unstaged; a target without
    `async_copy` (HIP, Metal) stages no decoded operand.
  - **Unstaged.** Other operands whose terms mix the box's ranges with
    others, a computed operand whose largest tile steps through another loop
    than an operand that is a load, a core whose operands step through
    different loops, a kernel with other shared memory, and `Setting.stage`
    off, as the goldens compare with tinygrad, stay unstaged; a tile whose
    address gathers copies in one stage, so that no gather reads past the
    loop's bound.
- **Reason:** (d), GPU time on fudge's RTX 5000 Ada unless marked:
  - staging: kaun's `GptOss/prefill-512` 199.7-201.3 ms to 174.3-175.6 ms;
  - the warp tile's halving is a fit at two stages on fudge: the window
    lies between 5760 and 11520 threads, so a part of 142 SMs may leave SMs
    idle and one of 24 halve tiles it could keep;
  - gathered rows: MXFP4 experts' gate and up at 512 tokens 1271 to 931 us;
    in two stages they ran slower, 1025-1034 against 935-958 us;
  - a core reading gathered rows keeps the unstaged splits: on the staged
    tile, its registers halving the warps an SM holds, bfloat16 experts'
    gate and up at 512 tokens ran 2276-2283 us against 2205-2212 unstaged,
    and the last wave that decides the sign follows the card's count of
    SMs, which no target fact states;
  - rows each warp reads alone save no load and add a wait and two barriers
    a trip: 2300-2310 us staged against 2199-2202 unstaged;
  - a run of three tiles in one warp: experts' down product in blocks of 48
    rows 1069-1081 us against 1103-1116 for three warps of one tile;
  - a computed operand: the tolk bench's `stream/output-projection-64`
    105.4-106.0 to 58.1-58.2 us;
  - a decoded operand only as 16-byte `cp.async`: copied through registers,
    Q8_0's output projection ran 1.34-1.43 times slower at 64 tokens;
  - held scales: with the codes copied by `cp.async.cg` and the scales
    held, at 32 values a trip, on real gpt-oss-20b weights and routing, gate
    and up at 512 tokens 775-798 to 737-758 us (to re-measure on fudge). A
    tile is held only where the operand has copied loads: its scales held
    with the codes left as loads ran the down projection at 64 tokens 19-20%
    slower;
  - `Stage.steps`, 64 values a trip against 32, scales held in both: the
    tolk bench's `stream/mxfp4-64` 74.2-74.9 to 63.0-67.2 us (to re-measure
    on fudge);
  - the nest, the spread and the depth: `stream/key-projection-64`
    39.7-39.8 to 15.7 us and `stream/output-projection-64` 57.8-58.0 to
    28.1-28.3.
    An `in_flight` one trip shallower ran L2-warm `stream/key-projection`
    faster, but the compiler cannot see which operand is resident, so the
    cold case, a model's weights read once a step, decides.

  AMD gfx1201 and Metal stage in one stage of register copies with CUDA's
  bank constants, gathered rows included, and state no `units` or
  `in_flight`; correct, unmeasured.
- **Pinned by:** the Stage suite: `values › a staged kernel computes the bits
  of the same optimisations unstaged`, `values › a staged product of
  gathered rows …`, `values › a staged dense MXFP4 product …`, `a dense
  MXFP4 product's staged rows cover a sector a trip`, `values › a staged
  computed operand …`, `values › a staged core's plans at two counts of
  units compute the same bits`, `values › a staged core's plans under
  another sum at any units …`, `kernels the race checker runs › a product
  over a nest of loops …`, `a core under another sum whose tile takes its K
  whole …`, `a core reading gathered rows keeps the unstaged splits`,
  `gathered rows each warp reads alone stay unstaged`, `serial ranges › …`,
  `barriers › …` (on CUDA and on HIP's RDNA3 and RDNA4), `races`
  (`compute-sanitizer`'s race checker in hazard mode on a kernel of each
  depth, one of gathered rows, one of a computed operand, one over a nest of
  loops, and a dense and a routed MXFP4 product whose scales are held, each
  also exact in `kernels the race checker runs`), `stages`
  (`Stage.depth`), `a staged core's workgroups, threads and trips`, `a
  staged warp holds a block's rows of one matrix`, `the layout` and
  `sources`; the tolk bench's `contract/bf16-*` rows, `bf16-512x4096x2880`
  and `bf16-512x512x2880` among them, `stream/output-projection-64` and
  `stream/key-projection-64`, `stream/mxfp4-64` and `stream/mxfp4-512`,
  rune's `Cuda/experts-bfloat16-64x4of32-2880x5760` and
  `-512x4of32-2880x5760` and `Cuda/experts-mxfp4-prompt-64x4of32-2880x5760`
  and kaun's `GptOss/experts-at-prefill-512`, their baselines pending the
  scheduled re-record.

## D263. CUDA copies 16 bytes from global to shared memory asynchronously

- **tinygrad:** `renderer/cstyle.py` (`CUDARenderer.barrier`,
  `__syncthreads()`), `base_rewrite` (a store is an assignment).
- **tolk:** `Cstyle.cuda` (`copies_async`, `cuda_async_lang`, `cp_async`);
  `Renderer.t.async_copy`, from sm_80.
- **Differs:** a store into shared memory of a 16-byte load from global
  memory prints as `cp.async.cg.shared.global`, a gated load with
  `ignore-src` where its gate fails, which copies zeros, and every barrier
  of a kernel holding one prints as `cp.async.wait_all; __syncthreads()`,
  since a thread waits only for its own copies. A kernel without one renders
  as tinygrad's. With at most one stage in flight, waiting for all copies
  waits for none too many, so no group is counted.
- **Reason:** (d): the second stage of D262 overlaps the next trip's copy
  with this trip's cores only where the copy does not wait; its rows are
  D262's.
- **Pinned by:** the Stage suite's `barriers` (its second clause), `races`
  and `sources`.

## D264. Withdrawn

A read of 16-bit elements of an operand staged in two stages was a load marked
`ldmatrix` or `ldmatrix.trans`; it now reads each element narrow. It gained
on staged bfloat16 contractions and nothing end to end, and the loss on those
rows (1.04-1.18x on `contract/bf16-*` on fudge) is the stated cost.

## D265. No axis a call reads runs as lanes

- **tinygrad:** `codegen/opt/postrange.py` `upcastable_dims` and
  `apply_opt`, which upcast an axis that a call's argument reads, of a body
  or of code in the target's language (a `CUSTOM_FUNCTION` with `fmt`);
  `codegen/opt/heuristic.py` `hand_coded_optimizations`, whose
  matrix-vector layout upcasts output axes; `test/runtime/test_custom_kernel.py`
  (`test_call_loop_mini_opts`, `test_call_loop_split`, expected failures).
- **tolk:** `Postrange.Scheduler.read_by_opaque`, `upcastable_dims` and
  `apply_opt`; `Heuristic.streams`.
- **Differs:** an axis that a source of a call reads, as `i` in `f(&x[i])`,
  `f(i)`, the statement `{0}[0] += 1;` on `&C[i]` or the expression
  `__builtin_popcount({0})` on `x[i]`, is not upcastable, and an upcast of
  it is refused, as is a tensor core in a kernel with a call; the heuristic
  streams no contraction there. tinygrad upcasts such an axis and renders
  the call with a vector where it takes one pointer or value, which C
  compilers reject. tinygrad gates a pad's effects (18a86504f) and makes
  code in the target's language a call, an effect (b63ad7ead), so a pad of
  an axis a call runs along is refused there too. An expression is a call
  that its reader writes inline, an effect like any call, so D271 applies to
  it.
- **Reason:** (e): under the heuristic, `test_call_loop_mini`'s kernel and
  that code's over 16 `i` fail at the C compiler, as does
  `out[i] = __builtin_popcount(x[i])` as an expression over 16 `i` upcast by
  4 (Clang: "passing 'int4' to parameter of incompatible type 'unsigned
  int'"). An expression's text states no lanes: the target's builtins take
  one value. One call per lane, gated on its point, would keep the upcast,
  for a kernel no raven library builds yet.
- **Pinned by:** the Codegen suite: `calls in a kernel › a call in an
  optimised kernel runs once per element`, `effects run as often as written
  › code in the target's language in an optimised kernel runs at each
  step`. The Postrange suite: `axes a call or custom code reads`, each case
  a call of a body, a statement and an expression. The Heuristic suite:
  `streaming › a kernel whose call reads its output axes is not streamed`.

## D270. A range of a typed integer splits and pads as a weak one

- **tinygrad:** `uop/ops.py` `divides` (divides a `CONST` and no `CAST` of
  one, though `UOp.val` reads a cast constant as one constant);
  `codegen/opt/postrange.py` `Scheduler.apply_opt` (pads only a range whose
  bound is a `CONST`); `codegen/opt/heuristic.py` `hand_coded_optimizations`
  (gives threads only to such ranges and sums only a `CONST` stride);
  `codegen/simplify.py` `mark_range_mod` (splits only such ranges) and
  `mark_gated` (shrinks a range only to a `CONST` guard);
  `codegen/late/coalesce.py` `memory_coalescing` (reads only a `CONST`
  index or term as an access's offset).
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
  offsets merge into vectors, as with weak ones. tinygrad refuses every
  split of it, and sums no typed stride.
- **Reason:** (e): tinygrad's heuristic raises on a kernel whose ranges are
  of `Int32`, `out[16i + j] = a[16i + j] + 1` over 64 `i` and 16 `j`: its
  checks read the bound through the cast and pass a split that `apply_opt`
  then refuses ("16 can't divide" the bound 16 on the host, "4 can't
  divide" it on CUDA), so a valid kernel does not compile, which no setting
  avoids but disabling the heuristic. The strides and tiles follow from the
  same reading: 119 of the heuristic's 431 recorded cases, indexed in
  `Int32`, took other optimisations, as a matmul upcasting its rows where it
  upcasts its columns, a routed product refusing its tensor core. A typed
  guard left its range whole, and loads at typed constant indices loaded one
  element each.
- **Pinned by:** the Heuristic suite: `a range of a typed integer is split
  as one of the weak integer` (`Int32`, `Int64` and `Uint32`, on the host
  and on CUDA), `a kernel indexed in a typed integer is optimised as in the
  weak one` (each recorded case). The Postrange suite: `apply_opt pads a
  range of a typed integer as a weak one`, and the golden case
  `split_int32_axis`, which applies. The Simplify suite: `pm_split_ranges ›
  a range of a typed integer splits as a weak one` and `pm_simplify_ranges ›
  shrinking › a range of a typed integer shrinks to its guard as a weak
  one`. The Coalesce suite: `memory_coalescing › lengths › loads at typed
  offsets merge as at weak ones, for`.

## D271. No axis a call does not read takes threads

- **tinygrad:** `codegen/opt/postrange.py` `convert_loop_to_global`, which
  makes global every output range that each `STAGE` reads, and
  `Scheduler.apply_opt`, which splits any axis into threads and takes a
  tensor core in a kernel with a `CALL`; `codegen/opt/heuristic.py`
  `hand_coded_optimizations`, which ranks such axes for threads;
  `codegen/gpudims.py` `group_gpudims`, which masks only a global store, to
  the threads whose local indices it misses are 0.
- **tolk:** `Postrange.Scheduler.opaque`, `unread_by_opaque`,
  `convert_loop_to_global` and `apply_opt`; `Heuristic.streams` and the
  heuristic's local splits.
- **Differs:** an axis that the sources of some call, of a body or of code
  in the target's language, do not read stays a loop: it is not made
  global, a local split of it (a group, for a reduce axis) is refused, and
  the heuristic gives it no threads. A tensor core, whose warp is threads
  along axes no call reads, is refused in a kernel with a call, and the
  heuristic streams no contraction there. tinygrad gives such an axis
  threads, and every thread runs the call. An expression in the target's
  language is a call too (b63ad7ead), whose reader writes it inline.
- **Reason:** (e): `n[0] += 1` by a call or by code, beside `out[i] = a[i] +
  1` over 64 `i` on CUDA, runs in each of the 16 threads the heuristic
  gives `i`, and beside a sum over 4096 `r` in each of the 16 threads of the
  group it makes: `n` counts 16 for 1. A call can write anything, so masking it
  to one thread of each such axis, as a global store is, would keep the
  threads only where no other thread reads what it writes, and drop what it
  writes to a thread's own registers in the others.
- **Pinned by:** the Codegen suite: `calls, custom code and stores on CUDA ›
  one beside stores across threads runs once`, `› one beside a sum runs
  once` (they skip without a GPU). The Postrange suite, each case a call of
  a body, a statement and an expression: `axes a call or custom code reads ›
  unread_by_opaque is the axes its sources do not read`, `›
  convert_loop_to_global gives workgroups only to an axis its sources read`,
  `› apply_opt refuses a local split of an axis its sources do not read`,
  `› a group is refused over a reduce axis its sources do not read`, `› a
  tensor core is refused in a kernel with one`. The Heuristic suite: `a
  kernel whose call reads no axis takes no threads`, on CUDA's renderer.

## D275. Clang compiles with LLVM's IR verifier on

- **tinygrad:** `runtime/support/compiler_cpu.py` `ClangCompiler.compile`
  (Clang's driver with its defaults, which in a release build pass
  `-disable-llvm-verifier` to the compiler proper).
- **tolk:** `Compiler_cpu.cc1`, which drops `-disable-llvm-verifier` from
  the command Clang states, and `Compiler_cpu.clang`'s `table`, which
  digests the command run (D113).
- **Differs:** the compiler proper runs with the verifier on: it checks the
  IR Clang makes of the source before the optimizations, and again before
  instruction selection, and invalid IR stops the compile with a diagnostic
  that names the function, which `Renderer.Compiler.Compile_error` carries.
  On fudge it costs 0.5-2.2% of a kernel's compile (cachegrind's
  instruction counts). The verifier after every pass
  (`-llvm-verify-each`), which cost 6.6-36% and names the pass, not the
  function, is not used: both cases below survive to instruction selection,
  where the verifier above stops them.
- **Reason:** (e): an optimization that writes invalid IR goes unseen by a
  release Clang. LLVM 19's straight-line vectorizer wrote `trunc <4 x i8>
  ... to <4 x i32>` for rune's routed rows (D167), which compiled into
  garbage row numbers; LLVM 18's wrote `sext <4 x i32> to <4 x i16>` for
  `r_3_2`, which crashed instruction selection. D167 turns that vectorizer
  off for those releases and LLVM 20, whose wrong values were valid IR;
  other Clangs and Apple's are unchecked, and with the verifier on, a
  defect of their passes of that kind is a compile error instead of wrong
  values. With the verifier, Clang 18 rejects `r_3_2` naming it, and Clang
  19 `E_16_2_4`. tinygrad's own LLVM compiler verifies its input and runs the
  verifier after every pass (`runtime/support/compiler_llvm.py`
  `LLVMCompiler.__init__`, `LLVMCompiler.compile`).
- **Pinned by:** `Tolk.Compiler_cpu › invalid IR › a source that LLVM 19's
  straight-line vectorizer makes invalid raises Compile_error naming its
  function`, under a Clang of LLVM 19 with that vectorizer turned back on.

## D280. An effect keeps its ranges when its index folds

- **tinygrad:** `codegen/__init__.py` `full_rewrite_to_sink` (simplifies a
  kernel's indices and ranges with nothing holding an effect's ranges),
  `uop/symbolic.py` `symbolic` (a value whose bounds meet folds to a
  constant) and `uop/ops.py` `UOp._ranges`, `UOp.ranges` (a node depends on
  the ranges its sources depend on).
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
  that runs no trip. An `After` of ranges alone that a builder wrote goes
  too where the effect depends on them without it, which places nothing
  differently. Not covered: an effect with no index, and a rewrite after the
  expander. The expander drops an `After`'s upcast ranges, so lanes of a
  range an effect reads only through the `After` would run it once: such a
  range, if weak, breaks its promise, which a store whose cells show it is
  refused for before any optimisation (D333), and a serial one takes no
  lanes.
- **Reason:** (e): an effect that runs once per step of its loops as built
  runs once, or runs in a loop of no trip.
- **Pinned by:** the Codegen suite's `effects at an index that folds run as
  often as written`; `reg_placeholder_then_reduce`'s goldens, and the
  kernels the Heuristic, Postrange, Search and Simplify suites record where
  their passes receive them, which carry the `After`s.

## D285. A loop holding a barrier or a matrix instruction has a uniform bound

- **tinygrad:** `codegen/__init__.py` `full_rewrite_to_sink`, which places
  the implicit barriers and lowers the ranges to hardware indices with no
  check of the loops a barrier runs in; `codegen/opt/postrange.py`
  `apply_opts`, which runs the heuristic, or BEAM, on any kernel without a
  `STAGE`.
- **tolk:** `Uniform.varying`, `Uniform.check` and its call in
  `Codegen.full_rewrite_to_sink` after `pm_implicit_barriers`;
  `Postrange.apply_opts` (`threads_fixed`).
- **Differs:** a kernel whose barrier, matrix multiply-accumulate, or call
  of a body holding one, runs in a serial loop whose trips may differ
  between a workgroup's threads is refused when lowered, naming the kernel,
  the operation and the loop, under `SPEC=0` too. Neither the heuristic nor
  a search optimises a kernel that holds, or calls a body that holds,
  workgroup or register storage, a barrier, a matrix multiply-accumulate or
  code in the target's language; the optimisations such a kernel lists
  still apply.
- **Reason:** (e): a barrier some threads skip, or reach fewer times, hangs
  the workgroup or reads workgroup memory not yet written, and tinygrad
  compiles a barrier in a loop bounded by a thread index without a word.
  Its heuristic splits a hand-made kernel's axes into threads and lanes
  while the kernel's workgroup and register buffers stay sized for one
  workgroup and one thread: on CUDA, a kernel reading its neighbour's staged
  value computed 224 of 256 values wrong, and one whose staging loop got its
  bound from `threadIdx.y`, around a barrier, once hung the GPU.
- **Pinned by:** `Tolk.Uniform` (`varying`, `check`); `Tolk.Postrange`'s
  `apply_opts › neither hand-optimises nor searches a kernel whose builder
  fixed its threads with` and `› applies the optimisations a kernel whose
  builder fixed its threads asks for`; `Tolk.Codegen`'s `kernels whose
  builder fixed their threads`, on CUDA; and the tolk, rune, jera and kaun
  suites, which lower every kernel through the check.

## D300. A streamed thread loads a pair of float32 values at once

- **tinygrad:** streams nothing (D89); its `codegen/opt/heuristic.py`
  `hand_coded_optimizations` unrolls the last reduce range whole when it is
  at most 32 and few lanes are upcast, which the stream's own unroll leaves
  no room for.
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
- **Reason:** (d): kaun's `Quant/cuda/routed-product-512-tokens`,
  gpt-oss-20b's gate and up projection on float32 activations, takes
  24.4-25.9 to 12.6-12.9 ms on fudge. Unrolled for bfloat16 too, 12 and 17
  rows ran 1.1-1.15 times faster and 16 and 32 rows 1.1-1.2 times slower,
  so bfloat16 keeps the loop until a rule separates them. AMD and Metal take
  the same rule, unmeasured.
- **Pinned by:** the Heuristic suite's `streaming › no loop is left over
  the other operand's values of one load`, over the recorded streamed
  kernels, with the layouts in the suite's `.tolk` files
  (`gpt_oss_gate_up`, `gpt_oss_down`, `mxfp4_m12`, `routed_blocks`,
  `vecmat_decoded`); kaun's
  `Quant/cuda/mxfp4-product-of-16384-64-float32-tokens` and
  `Quant/cuda/routed-product-512-tokens`, whose fudge baselines (the first
  recorded before D300, the second not yet recorded) wait for the
  scheduled re-record.

## D320. The CUDA device's kernels compile to a cubin

- **tinygrad:** `renderer/cstyle.py` `CUDARenderer.__init__` (gives
  `NVRTCCompiler` `ptx=True` for the device `CUDA`);
  `runtime/support/compiler_cuda.py` `NVRTCCompiler.compile` (then returns
  PTX); `runtime/ops_cuda.py` `CUDADevice.function` (whose
  `cuModuleLoadData` translates it for the GPU).
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
  from the disk cache and the driver's cache missing, on fudge: rune's
  `cuda/sin-nested-8` kernel loads in 40.7 ms as PTX and 0.19 ms as a
  cubin, and a two-layer gpt-oss-20b's 12-token prefill takes 710-1050 ms
  with PTX against 58-131 ms with cubins. A cold compile is no slower with
  cubins: the slowdown D78 found, in whole calls timed under load, did not
  hold interleaved.
- **Pinned by:** the Cstyle suite's `CUDA's binaries › the CUDA device's
  kernels compile to a cubin`; rune's compile bench rows
  `cuda/sin-nested-8` and `warm/cuda/sin-nested-8`, whose processes run
  with the driver's cache off; their wall-clock baselines are missing until
  fudge's scheduled recording of every suite.

## D305. A reduce under a uniform gate runs its loop only where the gate holds

- **tinygrad:** `codegen/__init__.py` `full_rewrite_to_sink`, which runs the
  post-optimisation symbolic pass and then the expander: a reduce whose
  result a gate selects, or whose store a gate guards, runs every trip of
  its loops in every workgroup, where the gate fails too.
- **tolk:** `Codegen.lift_uniform_gates`, after the post-optimisation
  symbolic pass, through `Uniform.varying`; `Shape.fold_where_closure`,
  whose assumption leaves each range whole (`Ops.assuming`);
  `Codegen.gated_by_memory` and its use in `Codegen.apply_opts`.
- **Differs:** a value computed by casts and arithmetic from one reduce,
  which `where (g, v, e)` selects where `g` holds (or where it fails) or a
  store through `where (g, i, Invalid)` writes, gets the reduce's outermost
  serial loop bounded by `where (c, n, 0)`: `c` the conjunction of the
  clauses of `g` that are the same for every thread of a workgroup and read
  only ranges the reduce runs inside. A workgroup whose clauses fail skips
  the loop and its loads; the reduce's identity it then holds is never
  read. Each arm of a selection is bounded alone. The rewrite declines when
  no clause is uniform, when the reduce has no bounded serial loop, or when
  a node of the reduce that computes from the loop is read outside it, as by
  a second reduce over the loop. A selection's condition, assumed within its
  arms, stops at a range: the lifted bound holds the condition, and
  rewriting the bound inside an arm would fork a second loop of the range's
  identity. A kernel that the rewrite, taking every value as uniform before
  the opts, would give a loop bounded by a value read from memory takes the
  heuristic's optimisations when a search is asked for, as a kernel whose
  loop's bound reads memory does (D255): the search times on storage of unspecified
  contents, where such a gate switches the blocks off. tinygrad searches
  every kernel. Routing data chosen for the search would make a kernel's
  program depend on one model's data.
- **Reason:** (d): rune's routed product (`Lower_reduce`, blocks of 64 rows
  of one group), whose blocks past the groups' hold only padding rows, which
  it states as `Invalid` and the gated store carries. kaun's
  `GptOss/experts-at-prefill-512` takes 3.29 ms without the rewrite and 2.80
  ms with it (minima of four interleaved rounds); the rows' wall-clock baselines wait for the scheduled
  re-record. Searched, the routed product at prefill 512 timed 6.84 us
  against 8810 us in the model, 94% of the step's loss under BEAM. Leaving a
  range whole in `fold_where_closure` is part of that
  rewrite: tinygrad makes no bound that holds a condition, and without it a
  lifted kernel with a load gated by the loop's own clause does not compile.
- **Pinned by:** `Tolk.Codegen`'s `reduces under a uniform gate` (on the
  host and CUDA); `Tolk.Symbolic`'s selections (`the condition is not
  assumed in the bound of a loop in a branch`); the goldens
  `uniform_gate_reduce`, `uniform_gate_store` and `varying_gate_reduce`;
  rune's `rows in blocks give eager's integers and hold their bound`;
  kaun's `GptOss/experts-at-prefill-512` and `GptOss/prefill-512`, and the
  tolk bench's `gated/bf16-2048-5-of-16-blocks-off` and
  `gated/bf16-2048-all-on`; `Tolk.Search on the host`'s `a gate read from
  memory › a kernel whose sum a gate read from memory selects is not
  searched, and takes the heuristic's optimisations` and `› a kernel whose
  sum a gate of its indices selects is searched`.

## D315. A global store is masked by the workgroup indices it does not use

- **tinygrad:** `codegen/gpudims.py` `group_gpudims`, which masks a global
  store to the threads whose local indices its index misses are 0, and runs
  it in every workgroup.
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

## D330. The kernels of one marked write are unordered whichever Afters list them

- **tinygrad:** `schedule/__init__.py` `create_schedule`'s write-after-read
  edges, which order a kernel reading a state before the other kernels of
  every `AFTER` over that state but the `AFTER` the read was listed under
  (`if a is u: continue`), and raise "cycle detected in assign graph" where
  those edges form a cycle. One kernel per store (`schedule/rangeify.py`
  `split_store`) lists each kernel under one `AFTER`, so its stores never
  meet the case.
- **tolk:** `Schedule.schedule`.
- **Differs:** a kernel reading a state runs before every other kernel of
  each write over it. A kernel of a write marked disjoint (D405) that two
  `AFTER`s list, as one whose stores share their loops (D137) into `p` and
  `q`, reads `q` as the store it makes there, beside the other kernels of
  `q`'s write, and is ordered before none of them, as it would be listed
  under that write alone. tinygrad orders it, through its read listed under
  `p`'s write, before them. A kernel of an unmarked write has no such
  exemption: through an argument it does not write it reads the state
  before the write's other kernels, where tinygrad skips every read listed
  under the write. A launch of the kernels of one write holds them apart
  unless the write is marked disjoint (D405).
- **Reason:** (b): rune's split program `[a; b] - [b; a]`, two assemblies of
  the same computed pieces stored into their regions of two buffers, makes
  two kernels, each storing a column of both. Each was ordered before the
  other, and no copy cuts the cycle, since each read is the kernel's own
  store; the program was declined, and over a set without kernels it
  raises.
- **Pinned by:** the `Tolk.Schedule` suite: `create_schedule › laws › a
  schedule copies a state iff its kernels form a cycle`, and
  `create_linear_with_vars › writes › stores crossed into two buffers write
  both`; rune's `Jit_devices › assemblies give eager's bits › crossed
  assemblies of computed pieces`.

## D331. A reduce of one trip or none, and a range of no trip, compile

- **tinygrad:** `codegen/simplify.py` `reduce_unparented` (asserts that
  every source of a reduce past its value is a range, where
  `uop/symbolic.py` `symbolic` folds a range of one trip to a constant) and
  `merge_ranges` (merges a range of size 0 into a range of the size it has,
  the same node, and substitutes its quotient for it);
  `codegen/opt/heuristic.py` `hand_coded_optimizations` (optimises a kernel
  that runs no trip of a range as any other).
- **tolk:** `Simplify.pm_reduce_unparented`, `Simplify.pm_simplify_ranges`
  and `Heuristic.hand_coded_optimizations`.
- **Differs:** a constant source of a reduce, a range of one trip folded,
  binds nothing, and the reduce drops it, a typed integer's constant (a
  cast of one) too: a sum over one trip is its one value. Two ranges merge
  only into a range other than the first, so a first range of no trip stays
  whole. A kernel holding a range of no trip takes no hand-coded
  optimisation, since it does no work along that range to share among
  lanes or threads: an unoptimised one writes the same values, a sum over no
  trip its identity.
- **Reason:** (e): tinygrad raises on kernels built by hand that compute a
  value. `out[i, j] = Σ_k a[i, k] b[k, j]` over one `k` raises "some
  reduce srcs aren't ranges", with or without optimisations; over no row
  `i`, `graph_rewrite` stalls, its substitution reaching itself; over no
  `k`, the heuristic's upcast of `j` by 4 merges loads of `b`, which holds
  no element, into a vector past its end ("invalid shrink"). tolk raised
  the same, and on CUDA its streamed product found no contracted range, and
  on the host over no column an upcast of rows put two lanes' stores at one
  address ("attempting multiple stores").
- **Pinned by:** the Codegen suite's `kernels of few trips › on the host,
  write what they compute node by node` and `› on CUDA, ...`, against the
  interpreter, with and without the heuristic, their indices in the weak
  integer or `Int32`; the Simplify suite's
  `pm_reduce_simplify › a sum over a range of size 1 is its value where the
  range is 0` and `› a sum over a typed range of size 1 is its value where
  it is 0`.

## D332. A call's arguments lose their indices once ranges are applied

- **tinygrad:** `schedule/rangeify.py` `no_indexing_calls`, applied at
  `get_kernel_graph`, after `pm_limit_bufs` has staged values: `stage` and
  `fix_store_hazard` index a stage by `s.ranges`.
- **tolk:** `Rangeify.get_kernel_graph`, which applies
  `pm_no_indexing_calls` right after `Indexing.run_rangeify` too.
- **Differs:** a movement of storage that a call takes and another kernel
  reads is one node, which applying ranges indexes by that kernel's ranges,
  for the call too. Until the call drops the index, those ranges are among
  the ranges of every value that reads the call's results, so a store that
  reads its destination at another index stages its value over ranges no
  kernel ends, and tinygrad's kernel-graph check refuses the graph. tolk
  drops the index before any pass reads the ranges.
- **Reason:** (b): norn's `evidence › a conjugate normal`, a jit of a vmap
  of SMC, which stores a value gathered from its destination beside a
  loop's call whose argument another kernel reads, failed the check and
  raised.
- **Pinned by:** `Tolk.Rangeify › get_kernel_graph › recorded ›
  assign_flipped_beside_call_1_kernels.golden` and the kernel-graph laws
  over `assign_flipped_beside_call_1`; norn's `Norn.Smc › evidence › a
  conjugate normal`.

## D333. A weak range is its builder's promise, refused where a store breaks it

- **tinygrad:** `uop/ops.py` `AxisType` (whose `WEAK` states nothing of
  its trips' independence); `codegen/opt/postrange.py` `apply_opts` (which
  optimises a kernel whatever its stores) and `shift_to` (which splits a
  range whatever its bound reads, by any amount that divides it).
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
  counted over every value of the variables. Only a range whose bound is a
  constant or a variable is counted, its points at the bound's greatest
  value, which some launch reaches for every such bound at once. A range
  whose bound reads memory or another range is not known at launch, and no
  split makes it lanes or threads. tinygrad optimises such kernels as any
  other.
  - Silent, as broken promises whose result is unspecified: a store whose
    validity or gate reads the ranges; two stores whose cells meet across
    points (`A[r] = 1; A[r + 1] = 2`); a read at a cell another point writes
    (`C[r + 1] = C[r] + 1`); cells that memory names; the effects of calls
    and custom code; a store whose cells are few only at some value of the
    variables (`O[r // m]`).
  - A range whose points another range's bound restricts can be refused
    falsely (`O[max(i, 1) + 4 j]` over `j < i`).
- **Reason:** (e): at the pinned commit, `C[r // 8] += 1` over a weak
  `r < 16` upcast by 4 counts 2 a cell where 8 is wanted, and on a GPU its
  threads race; `C[i] += 1` over `j < 4 (i + 1)`, `j` upcast by 4, counts
  `[1; 2]` for `[4; 8]`.
- **Pinned by:** the Codegen suite's `weak ranges` group: the refusals of
  `C[r // 8] += 1` over a weak `r < 16` on lanes, in host blocks and on
  threads, and over `r < 8`, of `C[r + s] += 1`, `C[0] += 1` after `r`,
  `O[0] = r`, `O[r // 8] = 5`, `C[16 (r // 8)] += 1`, `O[2i + 2j] = 1` and
  workgroup memory staged under a weak range; the counts of the same stores
  over serial ranges; the kernels it does not refuse (a packed triangle, a
  store gated or valid at one point, loaded and variable bounds, bounds that
  share a variable, a range of no trip, a scatter to cells memory names);
  and the refused upcast of `j < 4 (i + 1)`.

## D335. A long arange is laid out in rows, typed by its last element

- **tinygrad:** `mixin/op.py` `arange`: its type is checked and chosen from
  `start` and `stop - step`, and its elements are the running sum
  (`_cumalu`, `OpMixin._cumalu`) of `n` steps, whose pool
  (`mixin/movement.py` `_pool`) repeats the input to `(n + 1)(2n - 1)`
  elements before it cuts the windows.
- **tolk:** `Shape.arange` and its `run`.
- **Differs:** up to 1518500249 elements, the most whose pool's views fit
  an OCaml `int`, an arange is tinygrad's running sum. A longer one of `n`
  elements lays them out in rows of `b`, the power of two with
  `b² <= n < 4b²`: an element is its row's first element, an arange by
  `b * step`, plus its offset along the row, an arange by `step`, each
  built the same way. A float arange that long is laid out in `Int64` and
  cast, so each element is its integer rounded once. An arange of more than
  `max_int - 2^30` elements raises. The type of a nonempty arange is
  checked and chosen from `start` and its last element,
  `start + (n - 1) * step`, where tinygrad reads `stop - step`, and its
  length and bounds are computed exactly.
- **Reason:** rows: (a): a shape is an OCaml `int`, 63 bits, where
  tinygrad's is a Python integer, and the pool's views past `max_int`
  elements raised. (b): rune's `Unique`, which numbers its rows by an
  arange, lowers up to `2^31` rows and raised past 1518500249. A row and a
  column of about `sqrt n` elements keep both aranges within an int's views,
  and their kernel is elementwise. Rows summed
  in a float type would round each part: `Float32` elements were off by
  hundreds near `10^10`. Bounds: (e): when `stop - start` is not a multiple
  of `step`, the last element passes `stop - step`, and `arange ~dtype:Int8
  ~step:2 129` wraps its last element, 128, to -128 in tinygrad.
- **Pinned by:** the Ops suite's `aranges › an arange's element i is start +
  i * step at any length`, over integer and float types; `movement ›
  arange rejects more than max_int - 2^30 elements`, `› arange counts its
  elements past an int's range` and `› arange's type holds its last
  element, past [stop - step]`; the Tolk_engine suite's `aranges › an
  arange past 2^31 elements holds start + i * step`, on the host and on
  each GPU; rune's `Lower_index › numbering › a Unique of 2^31 rows is
  lowered`.

## D338. Withdrawn

Replaced by D404, one pass over the finished schedule, and D405, the mark an
assembly's maker states.

## D340. Lanes keep registers apart and threads keep the code around them

- **tinygrad:** `codegen/opt/postrange.py` `convert_loop_to_global` (makes
  a weak output range global whatever the code around it), `shift_to`
  (splits a range into lanes whatever its register stores read) and
  `upcastable_dims`.
- **tolk:** `Postrange.Scheduler.threadless`, which
  `convert_loop_to_global`, `shift_to` (and the splits of `apply_opt`) and
  `Heuristic`'s local splits read; `shift_to` and `upcastable_dims` for
  lanes; `Shape.written_params`.
- **Differs:**
  - Lanes: the lanes of a range share one thread's registers, so a range
    takes no lanes where a register store's cells, value or gate, or the
    condition a backedge repeats on, depends on it in data, a register load
    by its offsets and gate alone, an `After` by its first source, a
    reduction by its value less the ranges it closes. Under the weak
    promise a step reads a register cell only after writing it, or one no
    step writes, so a register store that depends on no lane writes what
    each step would.
  - Threads: threads run unordered with the code outside their range, and
    an effect that does not depend on a thread range runs in one of its
    threads, so a weak range takes no threads (it stays weak, and a local
    split of it is refused) where an effect depending on it and one not
    depending on it access one storage of global or workgroup memory, one
    of them writing it, or the second reads a register the first writes.
  - Effects are stores, calls, which write the arguments their body writes,
    and code, which writes every pointer it is passed; a body lowered to
    lines inside a kernel writes what its stores write.

  tinygrad splits and makes global such ranges as any other.
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
  scratch over a range made global; the splits kept (a counter to 3, a
  register written from another, a register ordered after a store, a
  register beside or holding a reduction, its range unrolled); the ranges
  given no threads on CUDA under the heuristic and their values (reads
  after, a register read after, a write before, writes by a call and by
  code), the local splits refused, the range a read-only call keeps global,
  and a lowered body's writes.

## D341. A range states its axis type

- **tinygrad:** `uop/ops.py` `UOp.range` (whose `axis_type` defaults to
  `AxisType.WEAK`).
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

- **tinygrad:** `renderer/cstyle.py` `code_for_op` (writes `+`, `-`, `*`, a
  unary `-` and `<<` whatever the integer type) and `pm_narrow_uint` (for an
  unsigned char or short alone); `runtime/support/compiler_cpu.py`
  `ClangCompiler.compile` (Clang, without `-fwrapv`); `codegen/decomp/op.py`
  `get_late_rewrite_patterns` (a product by a power of two becomes a left
  shift, of a signed operand too).
- **tolk:** `Cstyle.pm_wrap_signed`, in the extra matchers of
  `Cstyle.clang`, `Cstyle.cuda`, `Cstyle.hip` and `Cstyle.metal`.
- **Differs:** C computes an operation on a signed integer in `int`, or in
  `long long` for a 64-bit one, and leaves its overflow undefined, as it
  does a left shift of a negative value; CUDA, HIP and Metal follow C++,
  which does the same, and a compiler takes either for impossible. tolk's
  integers wrap at their width (`Ops.exec_alu`). An `Add`, `Sub`, `Mul`,
  `Neg`, `Shl` or `Mulacc` of a signed type computes in the unsigned type of
  `int`'s or `long long`'s width, its operands converted, and converts
  back, unless its operands' bounds (`Ops.operand_bounds`) keep its exact
  result in that type and, for a shift, the shifted value at or above 0 and
  the count below the width. An index, whose bounds chose its width, keeps
  tinygrad's source; a value loaded, or decomposed from a float's bits, does
  not. The conversion back of a value past the signed type keeps its low
  bits on every compiler tolk targets, as D17's narrowing does. `-fwrapv`
  would cover Clang alone: NVRTC's library holds no such option. On fudge,
  the kernels of the bench rows it changes (`host/sin_nested_8`,
  `host/normal_f64`, `gpt_oss/gpt_oss_decode_values`, `stream/decode-keys`,
  `stream/routed-argmax-16-bf16`) compile to the same machine code, host
  and SASS, as before, and the stages that linearize and render such operations allocate
  up to 14% more words (`embedding/render`), for the casts.
- **Reason:** (e): at the pinned commit, a CUDA kernel computes
  `x + y < x` of int32 as `y < 0`, true where `x + y` wraps below `x`.
- **Pinned by:** the Tolk_engine suite's `signed arithmetic wraps as the
  interpreter`, on the host (`kernels on the host`) and on each GPU
  (`GPUs ›`, run on fudge's CUDA and NV); rune's `jit › integer arithmetic
  at its type's edges gives eager's values`, shifts included, and its GPU
  suite's `integers ›` the same law.

### A division that may overflow and a shift past the width are defined (D346)

- **tinygrad:** `renderer/cstyle.py` `code_for_op` (writes `/`, `%`, `<<`
  and `>>` whatever their operands).
- **tolk:** `Cstyle.pm_defined_div_shift`, in the extra matchers of
  `Cstyle.clang`, `Cstyle.cuda`, `Cstyle.hip` and `Cstyle.metal`.
- **Differs:** C leaves undefined a signed division of the least value by
  -1, its quotient's and its remainder's, and a shift by a count at or past
  the width of the promoted operand; CUDA, HIP and Metal follow C++, which
  does the same, and a vector lane of a narrow type, which C does not
  promote, is undefined from its own width. x86 traps on that division,
  scalar or in a lane. tolk's integers wrap at their width
  (`Ops.exec_alu`), as nx states its values: the quotient is the least
  value, the remainder 0, a left shift and an unsigned right shift past the
  width 0, and a signed right shift 0 or -1.
  - A `Truncdiv` or `Truncmod` of a signed type whose operands' bounds
    (`Ops.operand_bounds`) reach the least value and -1 divides the
    magnitudes in the unsigned type of its width and takes the quotient's
    sign from the operands' or the remainder's from the dividend's.
  - A `Shl` or `Shr` whose count's bounds, in the count's own type (a
    `Uint32` count of a narrow value included), leave 0 to the width less
    one shifts by the count's low bits and selects 0 past the width, or for
    a signed right shift shifts by the count capped at the width less one.
    D342's shift reads its count so too.
  - An index's division by a positive constant and a shift by a constant
    count keep tinygrad's source. A zero divisor is D355's.
  - A shift's count is a number of bits, read at no width.
    `Ops.operand_bounds` keeps a count's own bounds, and `Ops.exec_alu`
    takes a count of any size: a left shift's value is 0 at or past the
    width of a committed type, a right shift the floor at any size. A left
    shift nothing truncates, by more than `Ops.max_exact_shift` bits, has no
    value, where tinygrad's Python integer would exhaust memory. A count at
    or past the width keeps the `(x & mask) >> k` rule, which tinygrad
    applies, from dropping a mask that clears the sign. tinygrad's fold of a
    constant and a bare count (`uop/symbolic.py`) commits the count at the
    operand's type, which wraps it; `Symbolic`'s fold reads it as it is.
  - tinygrad's `uop/weak.py` `commit_srcs_at` commits a weak count at the
    operand's type too: a uint8 shifted left by 256 is itself. tolk's
    `Uop_weak.commit_srcs_at` saturates a literal count at or past the
    width at the width, which every integer type holds, as it does a weak
    expression count always past it, and a weak expression count the
    operand's type does not hold at its own committed type, then converts
    it to the operand's; one whose known bounds no 64-bit integer type holds
    (`Ops.fits_no_int`), and that is not always past the width, is refused,
    since wrapped it would shift by the wrong number. A shift of a weak
    operand has no such type and commits as any weak expression does. A
    weak count does not widen the type its shift computes at.
  - tinygrad's `codegen/decomp/dtype.py` `l2i` emulates a 64-bit shift from
    the count's low word: a count of 2^32 shifts by 0, a count of 64 or more
    by its low 6 bits, and a `Uint32` count fails. tolk's reads both words
    of the count, applies this rule past the width (0, or the sign fill for
    a signed right shift), and takes the low word of a `Uint32` count.
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
  host`) and on each GPU (`GPUs ›`, run on fudge's CUDA and NV), against
  `Ops.exec_alu`; `a narrow type shifted by a uint32 count computes as the
  interpreter`; `a shift by a literal or weak count of any size computes
  the shift`, compiled and through the interpreter, on the host and each
  GPU; `an emulated 64-bit shift reads the whole of its count`, compiled
  with 64-bit integers emulated, by counts of the operand's type, of
  `Uint32`, weak literals and weak expressions; and the Symbolic suite's `a
  shift of constants folds by its count, of any size` and `a masked right
  shift past the width keeps its mask's sign`.

### A division by zero is defined (D355)

- **tinygrad:** `renderer/cstyle.py` `code_for_op` (writes `/` and `%`
  whatever the divisor); `uop/symbolic.py` `symbolic_simple` (`x//x -> 1`);
  `codegen/decomp/op.py` `floordiv_to_idiv` (corrects a negative dividend's
  quotient whatever the divisor); `codegen/decomp/dtype.py` `l2i` (an
  emulated 64-bit quotient, each bit of which a divisor of 0 sets);
  `helpers.py` `cdiv` and `floordiv` (whose quotient by 0 is 0, which
  `uop/ops.py` `python_alu` computes); `uop/ops.py` `_min_max` (bounds a
  load of a constant table by the table's values, whatever its gate).
- **tolk:** `Cstyle.pm_defined_div_shift`, in the extra matchers of
  `Cstyle.clang`, `Cstyle.cuda`, `Cstyle.hip` and `Cstyle.metal`;
  `Shape.symbolic_simple` (`x // x`); `Decomp_op.floordiv_to_idiv`;
  `Decomp_dtype`'s `long_division`; `Ops.vmin` and `Ops.vmax` of a gated
  load of a constant table.
- **Differs:** C leaves an integer division by zero undefined, its
  quotient's and its remainder's; CUDA, HIP and Metal follow C++, which does
  the same. x86 traps on it, scalar or in a lane. tolk's integer quotient by
  0 is 0 and its remainder the dividend (`Ops.exec_alu`), as nx states its
  values and as tinygrad's interpreter computes them.
  - A `Truncdiv` or `Truncmod` of an integer type whose divisor's bounds
    (`Ops.operand_bounds`) reach 0 divides by `where (d <> 0) d 1` and
    selects 0, or the dividend, where the divisor is 0; where D346 divides
    magnitudes, it divides by the divisor's magnitude made so. The rule
    reads such a divisor as nonzero, which its bounds cannot show.
  - `x // x` folds to 1 only where `x`'s bounds hold no 0; a floored
    division's correction of a negative dividend skips a divisor of 0; an
    emulated 64-bit quotient selects 0 where both words of the divisor are
    0. A division whose divisor's bounds hold no 0, as an index's by a
    positive constant, keeps tinygrad's source. rune divides as it lowers
    any divisor (`Lower_arith.int_div`).
  - A gated load of a constant table is bounded by its alternative too,
    which it reads where its gate fails: a pad's padding reads 0 there,
    which the table's values alone left out, so a division by such a load
    went unguarded.
- **Reason:** (e): at the pinned commit, the host's kernel of an integer
  division by a loaded 0 dies with SIGFPE, and CUDA's uint8 quotient of 0
  by 0 is 255, where tinygrad's interpreter gives 0.
- **Pinned by:** the Tolk_engine suite's `divisions and shifts at the
  extremes compute as the interpreter`, on the host (`kernels on the host`)
  and on each GPU (`GPUs ›`, run on fudge's CUDA and NV), with 0 among the
  divisors, `a // a` and a constant 0, against `Ops.exec_alu`; the Symbolic
  suite's `identities › x // x is 1 where x's bounds hold no 0`; the
  Decomp_dtype suite's `emulated 64-bit integers › i64 quotient by 0, which
  is 0` and its neighbours; rune's `jit › integer arithmetic at its type's
  edges gives eager's values`, its GPU suite's `integers ›` the same law,
  and `Lower_arith › divisions by constants give eager's bits`; the
  Symbolic suite's `bounds › a gated load of a constant table may read its
  alternative`; the Tolk_engine suite's `a pad of integer divisions writes
  what its tensors compute`, `a pad of a quotient by a constant table writes
  what it computes` on the host and on each GPU, and the Codegen suite's
  `padto_idiv_upcast`.

## D348. On the host, a decoded operand is decoded once for every row tile

- **tinygrad:** none: no pass stages an operand on the host, and
  `codegen/opt/heuristic.py` `hand_coded_optimizations` shares a decoded value
  only between the rows a tile upcasts; `codegen/late/linearizer.py` nests a
  kernel's output loops in the order of their axes.
- **tolk:** `Stage.decodes`, `Stage.panel`, `Stage.panel_bytes` and
  `Stage.decoded` (moved from `Heuristic`); `Codegen.full_rewrite_to_sink`,
  which runs it after `Stage.operands`, on a search's candidates too (D489);
  `Setting.stage`.
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
- **Reason:** (d): D92's tile shares a decode between the rows it upcasts, 8 on
  AVX2, so 64 rows decoded each value 8 times, and the decode was the whole gap
  to a stored decode: kaun's float32 rows by an MXFP4 matrix of 1024 by 1024
  took 12.05 ms against 8.5 ms stored. On fudge's P-core, kaun's bfloat16 rows
  (`Quant/host/mxfp4-product-of-1024-{10,64}-bfloat16-tokens`) go from
  3.01-3.03 to 2.15-2.17 ms and from 16.3-16.5 to 10.5-10.6 ms, and float32
  rows of 64 from 13.2-13.3 to 9.8-10.4 ms, beside 10.2-10.3 ms for the
  stored decode. BLIS likewise packs and casts a panel
  reused across the other matrix's rows (Low et al., TOMS, §4.3.2; Van Zee and
  van de Geijn, TOMS, §5.1.3). A panel past the L1 still gains (64 float32 rows
  by 8192, a 128 KiB panel past fudge's 48 KiB L1: 749 ms decoded per tile
  against 532 ms staged), so the bound is the stack the panel lies on: rig's
  pool workers have 8 MiB, the calling thread its own, 512 KiB for a secondary
  thread on macOS. The scheduler stays target-free (D92): the panel is the
  host's codegen. One row and rune's `Jit/experts-mxfp4-512x4of32-256`, whose
  operand reads the rows' blocks, keep their kernels. Files of 32 registers
  (NEON, AVX-512) take 16 rows a tile (D92), so 64 rows take 4 tiles and share
  one panel: unmeasured, fudge has neither.
- **Pinned by:** the Stage suite's `decoded panels`: `on the host a staged
  decoded product computes the bits of the same optimisations unstaged`
  (panels staged, refused at one row tile and past 64 KiB), `on the host a
  decoded operand read by several row tiles is decoded once` (on files of 16
  and 32 registers) and `staging off decodes once a row tile`; on fudge,
  kaun's `Quant/host/mxfp4-product-of-1024-*-bfloat16-tokens`.

## D345. A store through a broadcast is refused

- **tinygrad:** `uop/ops.py` `UOp.store` (stores through any view);
  `tensor.py` `Tensor.assign` (stores into the view it is called on, an
  expand included); `codegen/opt/postrange.py` `convert_loop_to_global`
  (makes the store's rows threads).
- **tolk:** `Prepare.prepare_rangeify` (`refuse_broadcast_store`, among its
  earliest rewrites, after inline calls are inlined); no
  `fix_repeated_store` in `Rangeify`.
- **Differs:** preparing a tensor graph raises `Invalid_argument` where a
  store's destination is an `Op.Expand` that stretches an axis of one
  element to more, or a movement, index, bitcast, `Op.After`, `Op.Detach`
  or `Op.Contiguous_backward` of one, whether the builder wrote it or an
  inline call bound a parameter its body stores into to such an argument,
  since several of the destination's elements are one cell of its storage,
  which the store would write once per element, in no order. A view of the
  broadcast that reaches each cell once, such as one row of it, is refused
  too; an expand that only adds axes of one, or empties one, is no
  broadcast. tinygrad stores through it: the kernel of
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

- **tinygrad:** `helpers.py` `_get_cpu_count` (the CPUs of the process's
  affinity, at most the quota of `/sys/fs/cgroup/cpu.max` rounded down) and
  `PARALLEL`'s default, divided among pytest-xdist's workers.
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

- **tinygrad:** `schedule/indexing.py` (`ALWAYS_CONTIGUOUS`: the defines,
  `AFTER`, `CONST`, `MSELECT`, `MSTACK`, `LOAD` and `CALL`) and
  `realize_custom_kernel_srcs` (which stores whole each other argument of a
  kernel's call after its reshapes, a written one included, and matches no
  lowered precompiled body); `uop/ops.py` (`UOp.custom_kernel`, whose results
  are each argument ordered after the call); `uop/ops.py`
  (`call_with_outputs`, which makes a precompiled call's arguments
  contiguous); `schedule/rangeify.py` `pm_no_views` (strips every view of
  storage); `uop/spec.py` `spec_kernel_graph` (no shrink).
- **tolk:** `Prepare.prepare_rangeify` (`call_storage`, among its earliest
  rewrites, which makes a run through any movements one shrink);
  `Indexing.realize_custom_kernel_srcs`, for a precompiled call's lowered
  body too; `Shape.run_of_storage`; `Rangeify.pm_no_views` and
  `Schedule.scheduled`, which keep a run as the call's argument;
  `Spec.tensor`, which leaves a call's arguments to it, and
  `Spec.kernel_graph`, which accepts the run, its bounds constants or
  variables outside loops.
- **Differs:** for a call compiled on its own, a kernel given as code or a
  precompiled call, indexing passes storage, a state of it, a call's
  result, a load, a scalar of a precompiled call and a run of storage (a
  shrink of storage, or of a state of it, whose elements are a run, on each
  trip where loops move it along), wherever the run starts, as they are,
  and stores any other argument whole. The run stays the call's argument
  in the kernel graph and the schedule, and the engine places it at its
  start. It raises `Invalid_argument` where the call
  writes such an argument (`Shape.written_params`, every argument where the
  body does not show its writes), and where it reads a constant or another
  value that loops move along. An argument passed as it is must start where
  its parameter assumes (D54, `Shape.storage_phase`): one that does not is
  stored whole, storage the call only reads outside loops by preparing the
  graph (`copy_off_start`), and it raises, naming both starts, where the
  call writes it, where loops move it, and where the parameter assumes a
  start off the 16 bytes new storage starts on. The tensor graph's
  specification checks no call's arguments against its parameters, so these
  copies and refusals are reached; the kernel graph's still checks them.
  tinygrad stores any such argument whole and the kernel writes the copy: a
  kernel copying four floats into `a[2:6]`, `a[::2]`, the first four of
  `a`'s transpose or a broadcast of `a[0:1]` leaves `a` and the call's
  result all zeros, and a kernel reading `a[4:8]` reads a copy of it, one
  more kernel (`custom_kernel_of_views`: 2 kernels, against tinygrad's 3).
  It passes a constant as it is, where the kernel expects storage.
  tinygrad's precompiled functions copy every argument first
  (`call_with_outputs`), so they write a copy; tolk's `Ops.call
  ~precompile:true` takes its arguments as a kernel's call does, writing
  through storage and a run of it, a loop's row among them, which rune's
  loops update in place.
- **Reason:** (e): the write landed in a copy, or at other elements, with
  no error, and a constant reached a kernel as no storage; with the
  specification checks off, a row off its parameter's boundary reached
  the call, whose vector accesses assume the boundary. A row a loop
  moves along is what rune's loops pass, and a run of an assembly's
  storage what its assemblies give the kernels of their pieces, each on
  16 bytes; its kernels write new storage, a state of it or such a run,
  so no raven library makes a refused call.
- **Pinned by:** the Prepare suite's `prepare_rangeify › calls compiled on
  their own`; the Tolk_engine suite's `calls of views` (by a kernel and a
  precompiled call; `› a call writes through a run of storage wherever it
  starts`; the refusals of a loop's column and of rows off 16 bytes with the
  specification checks on); the Spec suite's `tensor › arguments against
  their parameters` and its `verdicts` recording, a shrink of storage
  in the kernel graph; the Rangeify suite's `custom_kernel_of_views`
  recordings.

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
  variables a run binds (`Schedule.create_linear_with_vars`).
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
  `codegen/opt/postrange.py` (`shift_to`, whose split from the top makes the
  new range the axis's outer part); `codegen/gpudims.py` `group_gpudims` (the
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
  read 512 floats apart. The rule is the renderers' alike, as the SIMD group
  is every GPU target's unit of a load (CUDA C++ Best Practices Guide,
  §10.2.1; Apple's "Creating threads and threadgroups"). Kernels that
  stream, take a tensor core or no group keep tinygrad's choices; a sum's
  order changes with the threads' parts.
- **Reason:** (d): rune's compiled sums of few outputs on fudge's RTX 5000 Ada,
  as `Cuda/sum-rows-float32-2048x8192` (71.7-72.7 us against 23.7-23.8 us); no
  other `Cuda/` row moved past its range. Apple is untimed, since no machine a
  bench may run on holds a Metal device, and the Metal source is checked
  instead.
- **Pinned by:** the Codegen suite's `consecutive lid.x threads of the
  readout read one address or consecutive ones` (Metal, CUDA and HIP);
  `Tolk.Heuristic`'s `a group's fastest threads read at unit stride where a
  load reads a reduce or output axis so` and its `readout` cases; the goldens
  generated from tinygrad with this rule; on fudge the rune bench rows
  `Cuda/sum-rows-float32-2048x8192`, `Cuda/sum-columns-float32-16384x2048`
  and `Cuda/set-row-of-1024x1024-summed`, their baselines pending the
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
  promote it; a float base reads its exponent in the float as tinygrad does.
  tinygrad commits it there, so the exponent wraps. With the exponent whole, a
  negative one on an unsigned base is reachable, and `xpow_int` gives its
  greatest value no special meaning of -1: an unsigned base has none. A power
  at no width whose exact value would not fit in memory has no value:
  `Ops.exec_alu` refuses it and the fold leaves it, where tinygrad's Python
  integer would exhaust memory; a base of 0, 1 or -1 folds at any exponent. A
  weak exponent expression of a committed integer base, whose known bounds no
  64-bit integer type holds (`Ops.fits_no_int`), is refused at lowering
  (`Transcendental.xpow_int`): its value cannot reach the kernel, where
  tinygrad computes it wrapped and takes a power by the wrong number.
- **Reason:** (e): at the pinned commit, `Cast(uint16, 0) ** 65536` folds to 1,
  where it is 0, and a uint8 to a weak exponent of 256 is 1.
- **Pinned by:** the Tolk_engine suite's `a power by an exponent of any size
  computes the power`, compiled and through the interpreter, for each integer
  type; the Symbolic suite's `a power of constants folds by its exponent, of
  any size`, `a float power of a constant reads its exponent in the float`, `a
  power no width truncates stays a power past what it can hold` and `a power
  no width truncates folds for a base of 0, 1 or -1 at any size`; and the
  Uop_weak suite's `a weak exponent keeps its value past its base's type` and
  `a cast of a weak power computes it and leaves its exponent weak`; the
  Transcendental suite's `xpow_int › an exponent's bounds`.

## D413. An integer literal holds its operation's value from construction

- **tinygrad:** `mixin/elementwise.py` (`_broadcasted`, which keeps a weak
  constant weak under a committed operation), `uop/weak.py` (`commit_srcs_at`,
  which commits it late), and so every rule between the two reads the literal
  as written: `uop/symbolic.py` `symbolic_simple` (two stage folding),
  `codegen/decomp/op.py` (`floordiv_to_idiv`, `floormod_to_mod`,
  `powers_of_two`, the `fast_idiv` rules).
- **tolk:** `Ops.v`, which every constructor and `Ops.replace` goes through.
- **Differs:** a bare integer literal under an operation on a committed integer,
  or a cast to one, holds the value that type gives it, wrapped to its width,
  and stays a bare weak literal, so patterns still match it. A comparison reads
  it at the operands' meet, another binary or ternary operation at its own type,
  a cast at its target. A shift's count and a power's exponent are numbers
  (D346, D403), and floats are untouched (D128). So `-1` under a `uint32`
  operation is `4294967295` in the graph, and a golden whose literal was
  written past its width differs from tinygrad's in that spelling.
- **Reason:** (e): each rule that read a literal as written computed a wrong
  value or crashed: on int8, `x // 200` gave 0 where -1 is due; on uint8,
  `max(max(w, 1), -3)` became `max(w, 1)` where the machine computes
  `max(w, 253)`; on int32, `(x * 65536) * 65536` made a shift by 32, which C
  leaves undefined; `x:i32 * 2^32` failed to compile. The literal has the
  value before any rule sees it, so no rule must know to read it so.
- **Pinned by:** the Ops suite's `integer literals`, at every integer type, at
  -1, at the bounds, past the width and at the width's power; the Codegen
  suite's `constants past the width` (`a literal computes as its value at the
  operation's type`, every operation and type, `a product or conjunction by a
  literal that is 0 at its type is 0`, `a square of a product whose constant
  squares to 0 is 0`, `a weak divisor past the width divides as its value
  there`, `a shift by a weak expression count keeps the count whole`); the
  Symbolic suite's `committed constants hold their type's value` and `integers
  wrap › a square of a product keeps its value, its constants folded at an
  unsigned width`; the Decomp_op suite's `late_patterns › a product by a power
  of two shifts by less than its width`, `a quotient by a power of two reads it
  in its operand's type` and `a quotient by a weak divisor past its width reads
  it in its type`.

## D358. An object defines the half conversions Clang calls

- **tinygrad:** `runtime/support/compiler_cpu.py`
  (`ClangCompiler.compile`, which compiles the source alone).
- **tolk:** `Compiler_cpu.clang` (`half_conversions`, `converts_halves`).
- **Differs:** for an x86_64 processor without F16C, each object defines
  `__extendhfsf2` and `__truncsfhf2`, the conversions between halves and
  floats Clang calls there, rounding to nearest with ties to even. The
  rendered source is unchanged; the conversions follow it at compile, and a
  source's length class counts without them. A widened NaN is made quiet, as
  F16C's and arm64's conversions make it.
- **Reason:** (c). rig takes a symbol the object does not define from the
  process (`Rig.section-host_symbols`). The C math library's functions Clang
  calls on such a processor, such as `truncf` without SSE4.1 and `fmaf`
  without FMA (D178), are there, but compiler-rt's conversions are in no
  library a raven process loads, so every program with a half cast failed to
  link on such a processor: on another machine's x86_64 host, whose baseline
  `x86-64` has no F16C (D95).
- **Pinned by:** `Rune.jit on another machine's host › half casts give
  eager's bits`.

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
  driver rejects the option, compile without it.
- **Reason:** (e), as D275: an optimization that writes invalid IR goes
  unseen by a release comgr, and compiles into wrong values or crashes
  instruction selection. The verifier costs 1.9% and 2.5% of the compile
  time in two runs of gpt-oss-20b's first call and a beam search (comgr 3.0,
  nonnormal), and every code object is unchanged.
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
- **Reason:** (d), to re-measure on fudge. A device's clock follows its load,
  and a GPU idles while a round compiles, so times taken rounds apart ran at
  different clocks. On nonnormal's R9700 at the firmware's clocks, searches of a
  1024 by 1024 float32 product (`Rune.jit' ~beam:2 ~parallel:8`) stopped at
  round 5 under tinygrad's rule and reached round 7 under this one, choosing a
  kernel of 136.8-137.7 us of GPU time per call against 156-172 us. Timing both
  in turn keeps a near tie going on about half the time, as tinygrad's rule does
  between two minima of a round, where timing the beam's first alone sets one
  time against the least of a round's. No row on fudge holds the choice: a near
  tie at round 4 decides between kernels of 161 and 122 us per call under either
  rule.
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
- **Differs:**
  - Calls of one device that run in either order run as one launch whose
    members are their kernels, each lowered and compiled as a kernel of its
    own and laid out with its workgroups end to end, gated by the launch's
    index (D305). Two calls interfere when one writes a byte the other reads
    or writes: a view's bytes are those of its run of storage, any other view
    its whole storage, a loop or a copy writes its storage whole, and on the
    state of one write marked disjoint (D405) two of its kernels do not
    interfere. A call joins only a launch of its device that starts after the
    last position it interferes with, so it moves up only past calls it does
    not interfere with.
  - At the top level it moves up only where no storage's life grows: past the
    first use of each storage it takes, unless the storage is given to the
    schedule or holds one element; and a call of many elements joins only a
    launch writing storage it writes. The top level is tinygrad's schedule,
    one call a kernel, but for these launches of one state's stores and of
    calls of one element: that is what the recorded linears hold.
  - A loop's body (D60) has no tinygrad schedule, and runs its calls every
    trip, its own storage planned within a trip (D476): a call there joins
    the earliest launch at whose start no step up to it would hold more of
    the body's own storage (`Memory.bytes`, each buffer from its first use to
    its last) than the body's peak before any call joined, storage that the
    schedule is given, that entries outside the body take or that holds one
    element not counting, as at the top level. The
    check counts what the call makes and leaves out what it frees by moving
    up. A capturer passes the storage it holds (`create_linear_with_vars
    ~held`); without it the call's arguments are held, which for
    `Tolk_engine.compile` would leave rune's loop rows and carries out of what
    a trip holds.
  - A member is a call of a kernel whose ranges have constant sizes and whose
    storage lies on one device that is no disk. The arguments of a launch are
    its members', each once, storage first; a variable a member's call binds
    in its slot is read from that slot, as a launch binds each name once, so
    members may bind one name to several values.
  - What a target can run is the target's fact: `Realize.split_launch` splits
    a launch before it is compiled, into launches a target can run. A member
    that fixes its threads or reduces, which may share values between a
    workgroup's threads through workgroup memory and barriers, joins only
    members of its kernel, so that every thread reaches each barrier of the
    member it runs (D285); Metal declares workgroup memory only in a kernel
    function, so there such a member runs alone (`Renderer.t.launch_locals`).
    A launch holds at most `Renderer.t.launch_members` members (CUDA's 64,
    where a launch runs fastest and past which NVRTC's time a member rises
    steeply, and Metal's 40, past which its compiler's rises) and
    `Renderer.t.launch_args` arguments (CUDA's 512, the 4 KiB of parameters a
    kernel takes), and with `MAX_KERNEL_BUFFERS` at `n` above 0 at most `n`
    buffers. Every subset of a launch is a launch, so splitting one is always
    legal; past its member limit a CUDA launch runs as several, alike chunks
    one program.
  - A launch's members have their own kernels, heuristics and gates: a member
    reading shared storage reads it as a parameter passed once; no arithmetic
    is shared across members, as fused kernels did, since `Schedule.schedule`
    keeps each kernel's own compute shared by D137 before kernels exist.
  - Under a beam width (`Realize.compile_linear`), each member asks for the
    search, as `pm_beam` asks of each called kernel, and is searched as a
    kernel of its own, timed outside its launch (`Codegen.launcher`). The
    launch's own sink asks for none; its program is not kept on disk while a
    member asks for one (`Codegen.to_program`).
  - A schedule, ordered by its storage (D183), and a program made per run are
    launches of what they make: storage is live only from a member's first
    use, so a piece's store waits for the value it reads and no launch of them
    holds every piece's value at once.
  - Each member is its own body; alike members are one body, lowered once. A
    member of a power-of-two count of workgroups, laid at a multiple of it, is
    selected by an equality on the workgroup index, which Metal's front end
    threads into a switch after each call; for members whose count of
    workgroups is no power of two the cap costs launches and saves no compile
    time. On the host a launch compiles cold in 0.46 s against 0.09 s apart
    and costs nothing at run time.
- **Reason:** (d). On CUDA a launch costs about 1.3-2 µs, so a kernel's calls
  are its cost: on fudge's RTX 5000 Ada 16 pieces concatenated along axis 1
  under `Rune.jit` run as one launch, a median 7.2 µs a call, against 29-33
  with rune's mark removed, with the same bits. The member caps: on fudge,
  256 alike pieces of 64 by 64 run fastest at CUDA's cap of 64, 25-26 µs a
  call, and NVRTC's time a member grows past it, 3.3 ms at 128 against 2.0;
  `launch/cuda/compile-256` and `launch/cuda/alike-64` pin it, their
  baselines at the scheduled re-record. Metal's cap of 40: Reason: none.
  On an M1 Max Metal's time a member of one or two workgroups was least at
  40 members and grew past them, and 878 such members did not build in 4
  minutes, but no machine a bench may run on holds a Metal device, so no
  row can pin it. In a
  loop's body, (b) and (d): rune's remat'd scan steps and NUTS. A remat's
  recompute reads only its step's arguments, so a call that joins any earlier
  launch makes its values at the trip's first launch, where the forward
  pass's live, and the remat saves nothing in a loop of two trips or more.
  A scan over chunks of 2048 rows whose carry sums the loss and the
  weights' gradients of four remat'd residual layers, 256 to 1024 to 256,
  holds 39,845,908 bytes at 2 and 4 trips on the host, against 58,721,300
  without remats (`rune remat › memory › a scan of a remat'd chain holds a
  trip's storage at any number of trips`). The first use as the bound in
  loops too ran NUTS's loop bodies as 105 dispatches a transition against
  31, counted out of the schedule's launches before the order of D183's
  loop clause, 1.6 ms a step against 1.03 on fudge's RTX 5000 Ada.
  The cost: the loop rule allows no byte above the trip's peak. A buffer of
  one element does not count, as at the top level (two chains of scalar
  additions in a loop's body run a call of each chain a launch). Counted on
  the host after `Realize.split_launch` with CUDA's caps, jera's loop bodies
  dispatch, against counting one element as a block of 256 bytes: test_ode
  774 against 832 (614 against 669 before the split), test_root 53 and
  test_sde 330 against the same, NUTS's transition 35 against the same (25
  out of the schedule's launches, before the split); the top levels and
  every byte of the remat probes and of rune's per-trip bounds are
  unchanged. Joining the device's latest group alone, past the conflict,
  dispatches more on test_ode (951), test_sde (354) and NUTS (40), one
  fewer on test_root (52), and holds more: test_remat's `memory › stacks
  read by rows hold what separate arrays hold`, two chunks without remats,
  1,998,996 bytes stacked against 1,853,204 apart. Small storage of
  several elements still runs apart; on the host a launch costs nothing at
  run time, so the rest matters for jera on a GPU only, which no row
  measures.
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
  `create_linear_with_vars › launches`; `Tolk_engine › a scan's merged calls
  carry in place as its calls apart do` and the custom-kernel cases of two
  values of one name, on each GPU present; `Jit › variables › merged custom
  kernels of one value read each run's` and `› of two values keep each its
  own`; rune's `Jit › fully sharded › *`. The members' search: `Jit JITBEAM
  › JITBEAM unset › every kernel is searched, in launches and loops too`, its
  `JITBEAM=3` case, `Realize › lower_and_compile with a beam search ›
  searches each launch's members with the widths they ask for, in order, one
  at a time` and rune's `search.t`. Metal's cap: `test_launch`'s case
  of 512 one-element pieces on Metal and rune's Set of rows at 878 written
  positions (`Row_sets.examples`). The rows `launch/host/*`, `launch/cuda/*`,
  `launch/lower-256`, rune's `Cuda/cat-16-columns-of-exp-64x64`, and norn's
  `nuts/schools/4` and `nuts/schools-density-cuda/4`. The loop body's rule:
  the Schedule suite's `create_linear_with_vars › launches › a call in a
  loop's body joins an earlier launch only where no step holds more than the
  trip's peak` and `› a loop's independent calls run as few launches`, the
  first failing with no bound and with the first use as the bound, the second
  with the first use, `› a loop's calls of one element run as one launch a
  step` and `› a loop's calls of one element run after the calls whose
  element they read or overwrite`; rune's `rune remat › memory › a step that
  sums a gradient in its carry plans each trip` and `› a scan of a remat'd
  chain holds a trip's storage at any number of trips`.

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
  write land in listed order (D427), and two stores of one state that may
  overlap are never given a launch on the mark's say-so; stores that break a
  mark's promise give undefined values, as `Ops.after_disjoint` states. The
  mark keeps through `Rangeify.bufferize_to_store`, every other rewrite
  keeps it as a node's argument, and the disk cache's codec holds both the
  mark and the tag.
- **Reason:** (d) of D404 and rune's own proof. A concatenation along a later
  axis writes columns, strided regions which no byte range separates; a
  variable start is unreadable too, and a stepped assembly's stores overlap as
  ranges while its regions never meet. rune's `states` (`meet`, step-aware)
  already proves it. Without the exemption the row of concatenating 16 pieces
  along axis 1 costs 29-33 µs on CUDA against a median 7.2. A box test of
  store destinations in the tensor graph, the other way, is three times the
  lines and reads fewer regions.
- **Pinned by:** rune's `Jit › assemblies give eager's bits`'s test of the
  launch count (`a concatenation along a later axis runs as one launch`, the
  mark kept through rangeify and the disk cache), `Tolk.Launch_law › a write
  marked disjoint runs its columns as one launch`, `› a schedule read back
  keeps a marked write's launch`, `› marked writes of two calls run in turn`
  and `› a marked write of no stores marks nothing`, and the
  `create_linear_with_vars › writes › stores crossed into two buffers write
  both` pin of D330.

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
  the same. A product into bfloat16 keeps its bfloat16 matrices, since
  widened its sums would round where float32's do; its helper is
  `__attribute__((noinline))` instead.
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

## D426. A view of storage names its storage

- **tinygrad:** `schedule/__init__.py` (`pm_replace_buf`, `create_schedule`):
  a view of a buffer lifted into a parameter of its own, and a state keyed by
  the buffer alone, so a kernel reading a run and a kernel storing through it
  are not related.
- **tolk:** `Ops.whole`, `Schedule.view_param`, `Schedule.state_of`,
  `Schedule.value_state`, the view filter of `Schedule.schedule`'s reads,
  `Spec.shared` and `Spec.kernel_graph`.
- **Differs:** a run of a buffer a store writes keeps a parameter of its own,
  which names its storage's parameter as a second source (`Ops.whole`). The
  link enters the schedule key by structure; a view's offset stays out, but a
  loop's read of a run is a value of its wrapper body (D437), whose offset is
  part of that body. A kernel reading through a view reads its storage's
  state, and a view it stores through names the storage and no state, so a
  load through it reads the state the write is over. A kernel of a write
  follows the writers of the state it writes over, whatever it is given
  (D160). An `After` over a view of a state is a write over that state, so two
  such `After`s are two writes over it, refused as tinygrad refuses them unless
  one reads what the other left (D474). A cut reads a copy of the view's own
  run; the per-trip copy of a loop is for a state the loop itself writes.
  Regions stay in the launch scan; ordering is over whole states.
  `Indexing.joinable` keeps its base check: a kernel's slice now reaches the
  storage through the link, so a window store and a reader of a later state
  that share a value split into one more kernel, values equal.
- **Reason:** (e): `x[1:6] = 7` beside `y[:5] = x[1:6]` read 7 in `y`: the
  reader of a run and the store through it were not ordered. A schedule key
  holds a run's start only modulo its alignment, so a relation over regions is
  unsound on a cache hit; the relation is over whole states. tinygrad makes the
  same lift without the link. Cost: `*/schedule` words +1.7% on tolk's bench;
  rune's suite makes the same kernels and copies as without the link.
- **Pinned by:** `Tolk.Launch › states read as they were`, among them `a reader
  through the view a write stores into reads it as it was`, `a chain of stores
  through views of one buffer, its first run read`, `a loop reading an older
  run of its storage reads a copy of the run`; `Tolk.Schedule › rules`.

## D427. An unmarked After lands its stores in the order listed

- **tinygrad:** `schedule/__init__.py` (`create_schedule`): the stores of one
  `AFTER` are kernels in no order, and stores that overlap land in whichever
  order the kernels run.
- **tolk:** `Prepare.sequence`, `Schedule.schedule` and `Schedule.cuts`,
  `Indexing.share_loops`.
- **Differs:** an unmarked `AFTER` lists its effects in the order they land,
  the first over the state the `AFTER` names, an effect listed twice at its
  first place, each reading the states the graph names, older states as they
  were, so a later store wins where stores overlap. `Prepare.sequence` makes
  each listed effect a state of its own on its storage's chain (D473), so
  each kernel runs after the one listed before it; a store past the first
  that reads a state an earlier one overwrites reads it through a copy, made
  where those orders form a cycle (D160). Stores that share their loops
  (`Indexing.share_loops`) keep the listing. A call or loop loading through
  an argument it stores through loads what the kernels listed before it left
  there: load and store are one parameter, which no copy can part.
- **Reason:** (e): overlapping stores of one `AFTER` landed in an unspecified
  order, so `after x [store (x 0..6) f(x); store (x 2..8) g(x)]` could read
  `g(f(x))` at 2..6. rune and jera emit no unmarked `AFTER` of several stores,
  so only programs that list overlapping stores pay a copy or a kernel; the
  pass that orders them is D473's cost.
- **Pinned by:** `Tolk.Launch › an After lists its stores in the order they
  land`, among them `› a store listed twice lands at its first place`;
  `Tolk.Launch_law › a schedule's launches write what its graph states`.

## D428. A loop's call reads a fixed run of storage it writes each trip

- **tinygrad:** tinygrad's precompiled functions copy every argument before
  the call (`uop/ops.py`, `call_with_outputs`) and render no loop of a call;
  `schedule/__init__.py` (`pm_replace_buf`) lifts a view for a copy or a store
  only.
- **tolk:** `Prepare.trip_reads`, `Schedule.copy_start`,
  `Prepare.copy_off_start`, `Indexing.realize_custom_kernel_srcs`,
  `Shape.reads_storage_written`, `Spec.args_fit`.
- **Differs:** a contiguous run at a constant offset that a loop's call only
  reads, of the state the call writes through another argument, is a value
  the call computes each trip (D437), in every scope: the call's wrapper takes
  the state whole, which is copied each trip beside the call that writes it,
  as a loop's whole-state read is (D160's per-trip copy). An argument off its
  parameter's start (D54) of storage the call writes takes no copy before the
  loop: scheduling makes the copy, per trip in a loop that writes the state, at
  the start new storage takes (`Shape.allocated_start`) where the state's own
  start does not fit the parameter, cycle or none
  (`Schedule.copy_off_start_reads`). A read of an older state, or of storage
  written before the loop, stays stored once, as the graph names it. A fixed
  run of storage the loop does not write is stored once.
- **Reason:** (e): the loop read the run as it was before its first trip: a
  call storing row 1 of a buffer, plus one, into row `r` left `5,6,7,8` in row
  2 where the trips compute `6,7,8,9`. A stage made before the loop holds no
  trip's writes. rune and jera emit no read of this shape: their suites'
  schedules are the same with and without the rule.
- **Pinned by:** the `Tolk_engine` suite's `calls of views › a call's fixed read
  sees what its loop wrote`, `› a loop's fixed read off its parameter's start is
  read each trip`, `› a back edge's fixed read sees what its loop wrote`,
  `› a loop reads a flip of its written storage each trip`, `› a loop reads
  earlier states as they were`, `› a call reads storage it writes off its
  parameter's start, no after listing the storage`, `› a loop reads each run
  as its trip starts`, `› a loop's fixed read of an After state it writes is
  read each trip`, `› an After state's fixed run off its parameter's start`,
  `› an After state's fixed run under a back edge`, `› a fixed run of the
  state two stores deep`, `› a loop reads each run of an After state as its
  trip starts`, `› an After state's fixed runs at other offsets`, `› two fixed
  runs of an After state`, `› an After state's fixed run through a join`, `› a
  loop named by an older state, a run of the later`, `› a bitcast run of an
  After state read each trip`, `› older and sibling states' runs stay as they
  were`, `› two loops in sequence read their runs`, `› a loop's older run of
  storage leaves the state before it`.

## D429. Withdrawn

A store listed over a later state of its storage than its destination names
wrote that later state (`Schedule.name_written_states`), and an `After` listing
only stores into other storage was a join after them
(`Schedule.join_foreign_stores`). Merged into D473, where an effect runs after
the state of each `After` listing it whose storage it writes, and an `After`
listing no write of its storage names the latest state of it its effects
read or run after; D473 keeps this entry's reason and pins.

## D430. Conversions of bfloat16 on their bits, on HIP and with another formula

- **tinygrad:** `renderer/cstyle.py` `HIPRenderer.extra_matcher` holds
  `create_non_native_float_pats((dtypes.bfloat16, *dtypes.fp8s))` and not
  `pm_manual_bf16_cast`, so Clang converts between float32 and bfloat16.
  `renderer/llvmir.py` `LLVMRenderer` holds `pm_manual_bf16_cast` except on
  gfx950. `cast_float_to_bf16`, which `ClangRenderer`, `MetalRenderer` and
  `OpenCLRenderer` also use, tests the exponent and the low 16 bits, and folds
  the zero-class and rounding cases together.
- **tolk:** `Cstyle.hip_extra_matcher` holds `Cstyle.pm_manual_bf16_cast` on
  every HIP target, gfx950 included, and `Cstyle.cast_float_to_bf16` is one
  formula for every renderer: `(x != x) ? bits | 0x400000 : bits + ((bits >>
  16) & 1) + 0x7fff`, then the top 16 bits.
- **Differs:** a conversion between float32 and bfloat16 is written on the
  float32's bits on HIP, and the narrowing's formula is another one on every
  target. The value is the same as tinygrad's where tinygrad's is defined: the
  low 16 bits round to nearest with ties to even, and a NaN stays a NaN with
  its sign, its quiet bit set.
- **Reason:** (b): LLVM's InstCombine folds `fptrunc (uitofp x to float) to
  bfloat` into `uitofp x to bfloat` where the first conversion is exact, through
  a negation, an absolute value, a truncation, a selection or a phi, and narrows
  a `zext` of an `i1`, and an `or` or `and` of such `zext`s, to the `i1`. AMD's
  backend selects no conversion to bfloat16 from an `i1`, and comgr aborts with
  "Cannot select ... bf16 = uint_to_fp". rune's `Lower_arith.store_real` casts a
  boolean array to bfloat16 through float32, so `Nx.cast Nx.bfloat16` of one
  fails to compile on gfx1201 (`test_jit_gpu`'s integer casts). clang 19
  reproduces the abort on gfx1100, gfx1201 and gfx942, and only for an `i1`: an
  8- or 16-bit source compiles. The `i1` comes from LLVM's own analysis, so
  `(int)a | (int)b` of two comparisons aborts though `Ops.vmin` and `Ops.vmax`
  do not bound it, and no rule that looks only at a conversion's source or at
  the float operations next to it covers every case; a rule that walked a whole
  kernel's slice would, at the price of the walk. gfx950's conversion
  instructions cover `fptrunc` only, and the abort is keyed on the integer
  source (`SITargetLowering`), so it most likely remains there; gfx950 is
  unverified, clang 19 having no such target. The formula is LLVM's own
  expansion of a narrowing to bfloat16: on gfx1201, in clang 19, tinygrad's
  takes 8 instructions a lane more than `fptrunc` and this one 3 more. The speed
  on a workload is to be measured on the R9700, and on fudge's host at the next
  re-record.
- **Pinned by:** `Tolk.Cstyle › bfloat16 on AMD › leave LLVM no conversion to
  fold`, `› an integer rounds once`, `› a bool is exact`, `› a float32 narrows
  to its nearest code`, `› a bfloat16 widens to its value`; the `.tolk` files
  that name D430.

## D431. Note: a cast of an integer to a float on Metal rounds to nearest

- **tinygrad:** `renderer/cstyle.py` `MetalRenderer` writes a cast of an
  integer to a float as `(float)(x)`, `(half)(x)` or `(bfloat)(x)`.
- **tolk:** `Cstyle.metal` writes it the same way, and
  `Compiler_metal.compiler` passes no rounding option.
- **Differs:** nothing. This entry records why tolk needs no rounding of its
  own there.
- **Reason:** none needed. The Metal Shading Language Specification 4.1
  allows a conversion to a float to round toward zero (Section 8.6), but a
  cast in the source is a `static_cast`, which Section 2.23 rounds to nearest;
  Section 1.6.3 makes rounding toward zero opt-in, by
  `-fmetal-rtz-fp-conversion`, for float-to-float conversions only.
- **Pinned by:** `Rune.jit on a GPU › integers › an integer cast to a float
  gives eager's bits`, run on Metal, which holds the ties of every exponent.

## D432. Withdrawn

A view `Param` named the state a loop's call read a run of, so that a run of
a written state was read each trip with no offset in the schedule's key. That
read is now a value the loop's call computes each trip in every scope (D437):
a view names its storage again (D426).

## D433. A store through a bitcast of storage stores into the run its bytes cover

- **tinygrad:** `schedule/prepare.py` (`earliest_rewrites`): a store into a
  `BITCAST` of a target stores the value bitcast to the target's type, which
  strips the outermost bitcast only; `expand_bitcast` then rewrites any other
  bitcast between types of different sizes as shifts. `UOp.contiguous_view`
  reads a run through a bitcast at a constant offset only.
- **tolk:** `Prepare.store_into_run`, `Prepare.store_into_cut`,
  `Prepare.contiguous_view`.
- **Differs:** a store whose target is movements and bitcasts of storage, or of
  a state of it, whose elements are a run of whole elements of the storage
  (`Prepare.contiguous_view`), stores the value, flattened and bitcast to the
  storage's type, into that run: `x[1:4] as uint8 bytes 4..16 read as
  float32 <- v` writes `v`'s bits into elements 1 to 3, into the storage or
  into a state after other stores, from a constant or a bound start, the
  target 1-D or in rows, of float32 or of the type the bytes are viewed as, its
  value reading the storage or not. A run through a bitcast at bound variables
  is a run of the source where it starts and ends on whole elements at every
  value of them; in a call's body, a scalar parameter counts as such a
  variable. A store into a bitcast of storage itself takes tinygrad's
  rule. A run from a constant start that begins or ends inside an element of
  the storage stores the whole elements it touches, their bytes the run
  covers from the value and the others loaded from the target. A target whose
  only resizing bitcast is the outermost takes tinygrad's rule too, the
  bitcast moved to the value. Any other target through a bitcast to a type of
  another size is refused with a message: strided or reversed bytes such as
  columns of the storage's bytes, a reshape over a bitcast of strided
  elements, or a run whose end a bound variable places that may cut an
  element.
- **Reason:** (e), as D473: `Ops.store` through a view of storage and
  `Ops.bitcast` give the graph a meaning, and tinygrad raises an internal
  error on it ("UOp verification failed ... STORE [CAST u8, CAST u8]") once
  `expand_bitcast` has made the target a value: at the pin, `x.bitcast(
  dtypes.uint8)[4:16].bitcast(dtypes.float32).assign(v)` raises it on a
  realized buffer and on a state. No raven consumer emits such a store.
- **Pinned by:** `Tolk_engine › a store through whole elements of bytes viewed
  as another type writes the value's bits into them` and `› a store through a
  run of bytes that cuts elements writes the value's bytes into it, the others
  kept`, against `Tensors`, on the host and in each GPU group;
  `Tolk_engine › a store through a bitcast of a view that is no run writes
  the value's bits through the view` and `› a store through bytes that are
  no run, or that may cut elements where a bound variable places an end, is
  refused`; `Tolk.Launch_law › compiled` and `› GPUs`, whose programs store
  through bytes of states.

## D460. A compiled schedule is kept on disk whole

- **tinygrad:** `engine/jit.py` keeps a captured schedule only when a user
  pickles it; `schedule/__init__.py` keeps a function's schedule and
  `codegen/__init__.py` each kernel's program, in a process and, for schedules,
  on disk from `SCACHE=2`. A later process schedules and reads its programs
  again.
- **tolk:** `Tolk_engine.compile`, which rune's `Compile.link` calls;
  `Schedule.numbered`, `engine/dune`'s `engine_digest.ml` rule and
  `Setting.scache`.
- **Differs:** a later process that compiles an equal graph reads what
  `Jit.jit_lower` made from one disk entry (table `compiled`): it schedules
  nothing and compiles or reads no kernel. The key is the graph up to its
  numbering of ranges and storage (`Schedule.numbered`), the positions of the
  inputs and the held buffers in that numbering, each device's name, renderer,
  target and compiler key, the digest of the library's and the engine's sources
  and the settings that shape compilation (`Jit.shaping`). The entry holds the
  capture's variables, the schedule's text with each program reduced to what
  linking reads (its `Program` argument, a `Sink` holding its `Kernel` argument
  and no body, a `Linear` of its `Param`s, an empty source and a binary) and
  the binaries as bytes beside it, where quoting would grow them. The engine
  links it live, so its reads of devices and bound buffers are not in the key.
  A call that asks for a beam search, under `DEBUG` or `CHECK_OOB`, or a
  compiler that names no toolchain, keeps nothing, and neither does one
  collected by `Realize.kernels`, since a hit makes no program to report.
- **Reason:** (d) gpt-oss's first call of 12 tokens in a later process
  (`apps/gpt-oss`'s bench, `GptOss/first-call-12`). The schedule
  and program tables serve the kernels but still run the passes around them
  on every process start. On fudge, over the 24-block graph, the process's
  CPU time for the first call falls from 84.5 ms to 45.7 ms, and its
  allocated words from 12.40 M to 7.93 M.
- **Pinned by:** rune's `test_first_call` (`Rune.jit across processes`): `a
  later process's first call › compiles nothing, computes the same:
  arithmetic`, `a reduction` and `a scan`, `tells apart arguments in another
  order`, `reads this process's captured constants`, `a damaged entry is made
  anew, and replaced ›` truncated, holding no graph and of another value, `an
  entry of other sources is made anew, and replaced ›` the engine's digest is
  its own, of another engine and of another tolk, and `nothing is kept ›` with
  `DEBUG` on, without the binary cache and with `SCACHE` at 1.

## D465. A stage a call reads is no destination of a store

- **tinygrad:** `schedule/rangeify.py` (`pm_remove_bufferize`): a store through
  a view of a stage of `removable=False` writes the stage, a stage a user
  materialises and one a custom kernel's argument makes alike.
- **tolk:** `Ops.keep` (`Argument`), `Indexing.bufferize_and_index`,
  `Rangeify.destination`.
- **Differs:** a stage made because a custom kernel reads a view is
  `Argument`: kept whole, as `Whole`, but a store through the same node of the
  view writes the storage, the stage's own index inlined. A stage the user
  materialises stays `Whole` and is written as asked. `Ops.repr` prints both
  as `removable=False`, as tinygrad does; `Ops.key` (D12) and the serialized
  graph, which writes `argument=True` after tinygrad's fields, tell them apart,
  so a graph a user builds with an `Argument` stage keys apart from one with a
  `Whole` stage and reads back as itself.
- **Reason:** (e): `b1 = after b [store v (v + 2)]` beside a call reading the
  same node `v` as its argument lost the store: the store's target was the
  stage that holds the read, and `b` kept its old elements, a silent wrong
  value. The call still reads `v` as it was, through the stage.
- **Pinned by:** `Tolk_engine › calls of views › a store through a run a call
  reads lands`, `› a store through a reshape of a run a call reads lands`, `› a
  loop reading a run of the state under a store through it keeps the store`;
  `Graph › a graph of any argument reads back as itself`; `Tolk.Ops › key ›
  tells apart arguments a repr prints alike`; `Tolk.Indexing › run_rangeify ›
  recorded graphs › custom_kernel_rangeified.golden`, whose `.tolk` writes
  the stage `argument=True`.

## D434. A scalar value has no axis of its consumers, and stores join over what they write

- **tinygrad:** `schedule/indexing.py` (`run_rangeify`): the ranges of a value's
  consumers are its ranges, also for a store or a cast of a scalar, which
  broadcasts it: consumers at other ranges realize the scalar, and the stage of
  a shape `()` value is indexed by them, an index of a constant no pass folds.
  Stores are not joined over shared values there (D-numbers of tolk's
  `Indexing.share_loops`).
- **tolk:** `Indexing.assign_ranges` (a scalar's consumer ranges),
  `Indexing.joinable`.
- **Differs:** a scalar value takes no ranges from a consumer that is no
  broadcast of it. Two stores of one value join only over storage of one
  shape, the shape stored into; a scalar value stored into two sizes stays two
  kernels.
- **Reason:** (e): one constant stored through bitcasts at two states of a
  buffer reached the renderer as an index of the constant, `(1092616192)[i]`,
  a Clang error on a valid graph. Keeping the value unstaged then joined its
  stores across the two sizes, one store writing a fraction of its storage.
- **Pinned by:** `Tolk_engine › calls of views › one constant stored at two
  states through bitcasts`, `› one value stored whole into storages of other
  sizes`.

## D435. Withdrawn

A reader of a state over the whole of its storage ran before the stores of a
sibling `After` over a view of the state under it (`Schedule.sibling_edges`).
Two writes over one state are refused now, as tinygrad refuses them, unless
one reads what the other left, and then they form a chain (D474).

## D436. A state of a bitcast is a bitcast of a state

- **tinygrad:** `schedule/rangeify.py` (`pm_mops`): only movements move outside
  an `AFTER`, and a bitcast of storage is not one.
- **tolk:** `Prepare.sequence`, `Prepare.same_size_bitcast`,
  `Schedule.view_param`.
- **Differs:** an `After` whose first source is a bitcast of storage or of a
  state, or a movement over one, becomes that bitcast, and those movements,
  of a state of the storage on its chain, when the graph's effects are
  sequenced (D473): a state's first source is storage or a view of it,
  whatever type its readers take. A store through a movement of a same-size
  bitcast stores the value bitcast into the movement of the state, a weak
  value first taken at the view's type, a copy keeping its bitcast; one
  through a bitcast that resizes elements follows D433. A view of a loop's
  read of a state names that state in the form the movement rewrite leaves
  it, so the link stays a state through it.
- **Reason:** (e): `Ops.after` gives such a program a meaning, and it raised
  internal errors (a failed verification, a cycle detected, a Clang error) or
  the refusal of a loop's fixed read of the state. Every read then is a read of
  a state of the storage: the runs read each trip, and the older states read
  as they were, need no case of their own.
- **Pinned by:** `Tolk_engine › calls of views › a loop over a bitcast state
  writes an older name of its storage`, `› a call reads states of a bitcast and
  of its storage`, `› a later state over a bitcast is read each trip`, `› a loop
  reads a run of a state of a bitcast each trip`, `› one constant stored at two
  states through bitcasts`, `› a store through a run of a bitcast of a
  state`, `› a weak value through a run of a bitcast`.

## D437. A loop's read of a value of its state moves into the call's body

- **tinygrad:** tinygrad renders no loop of a call; a precompiled function
  copies its arguments once (`uop/ops.py`, `call_with_outputs`).
- **tolk:** `Prepare.trip_reads`, `Indexing.realize_custom_kernel_srcs`.
- **Differs:** an argument of a loop's call the call only reads, reading the
  state its write is over, that the call cannot take as it is (a run, a flip, a
  strided or reshaped view, a value computed from one or more reads) is
  computed inside a wrapper body each trip, in every scope: the schedule's and
  each body's, precompiled or inlined, before any body is scheduled. The
  wrapper takes that state whole, a row the loop moves as it is, and every
  other read as an argument decided by itself: an older state, or one older
  than the loop's listing, stored once. A plain body inlines into the wrapper,
  so the value fuses with its kernel; a kernel body is called with a stored
  value each trip. A run's offset is part of the wrapper, so runs at other
  offsets are other schedules. A value that
  moves with the loop and is no row, and a stage of a moving value, stay
  refused.
- **Reason:** (e): `Ops.after` gives these programs a meaning and they raised
  "reads storage the call writes".
- **Pinned by:** `Tolk_engine › calls of views › a loop's value of a state it
  writes is computed each trip`, `› a loop reads a value of several states each
  trip`, `› a loop's value read twice in the body, or sharing a leaf`, `› a loop's
  value through a bound variable`, `› a state over a view that is no run is read
  each trip`, `› a kernel body takes a value computed each trip`, `› a back edge's
  value is computed each trip`, `› a contiguous value of a written read is read
  each trip`, `› a value moving with a loop in a call is refused`, `› a loop
  reading a state older than its listing reads it as it was`, `› a loop reads a
  join of its own states each trip`, `› a loop in a body reads its parameter's
  run each trip`, `› a loop in a body reads its parameters as each trip
  starts`.

## D466. A long host kernel is written as several functions

- **tinygrad:** `renderer/cstyle.py` (`CStyleLanguage.render`) writes a
  program's body as one function, compiled by `runtime/support/compiler_cpu.py`
  (`ClangCompiler.compile`).
- **tolk:** `Cstyle.cut_kernel`, `Cstyle.render_host` and `Cstyle.clang`, on
  the CPU device only.
- **Differs:** a Clang kernel for the CPU device of more than 20 000
  operations, those written into their reader counted, is written as a kernel
  that calls functions of at most 2 000 operations in order; a statement of
  more is a function of its own. Statements keep their order, but for the
  buffers the kernel declares, which move to its start. A block too long for a
  function stays in the kernel and is cut inside. A value read outside the
  function that declares it is held in a frame on the kernel's stack, one
  array per C type: it is written where it is declared and read where a
  function first uses it, and its slot is free once the last reader of its
  value has run; a value declared outside a loop and read inside lives to the
  loop's end, since its next trip reads it again. A kernel of 20 000
  operations or fewer renders unchanged, so no golden changes.
- **Reason:** (d). An elementwise program written as an OCaml loop is one
  kernel, and Clang's time on one function grows faster than the function. On
  fudge (Clang 19.1.7, user seconds, machine under load) a fixed-step pendulum
  of 400 steps over 64 float64 under `Rune.jit'` compiles in 308 s whole and
  14.0 s cut, and runs 1.2% slower cut, with the same bits. A value read at the
  top of its function instead of at its first use was live across the function:
  a sum of 1000 `exp (sin x)` ran 20% slower so. Clang 23.1.5, whose
  straight-line vectorizer tolk leaves on (D167), compiles the cut sum in 14.0 s
  against 148 s whole, and its vectorizer then speeds the cut sum by 3.4% at no
  compile cost on the pendulum; Clang 18.1.3 gains as 19 does. LLVM 20 to 22, Apple's Clang, arm64 and Windows are
  unmeasured. rune's gradients of float64 `betaincinv` and float32
  `betainccinv`, the longest kernels its suites build, stay whole.
- **Pinned by:** rune's `compile` bench `long/pendulum-200`, on fudge;
  `Tolk.Cstyle › execution on the host › writes a long kernel in pieces`,
  `› leaves a GPU's kernels whole`, `› computes a long chain of values read
  once`, `› keeps two loops whose indices render as one name` and `› cut into
  pieces, writes what it writes whole`, whose cases include `keeps a value
  read early in a loop for its next trip`.

## D450. A tensor core is tried on any number of reduce axes

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`: the
  cores only on a kernel of one reduce axis, below `TC_OPT=1`; its docstring
  gives no reason).
- **tolk:** `Heuristic.tensor_cores`.
- **Differs:** `TC` positive tries the cores, whatever the kernel's reduce axes:
  the core takes one and the others stay loops around it (`use_wmma`'s extra
  reduces), except those D471 unrolls. `TC_OPT` keeps only level 2's padding.
  A convolution of tinygrad's shape moves onto a core: the `conv_half` and
  `conv` rows; rune's own `Nx.conv` of float16, measured at two sizes on
  fudge's CUDA, compiles the same kernels at `TC_OPT` 0 and 1.
- **Reason:** (d): attention's output with sinks, scaled per head and row, is
  read apart along head and head dimension, so the out projection's two reduce
  axes cannot merge; only this gate kept it off the cores. On an RTX 5000 Ada a
  512-token prefill of 24 blocks, on the bench's random weights, takes
  96.4-101.8 ms against 124.0-136.1, decode steps within noise. Kernels of a
  tinygrad-shaped convolution are unmeasured.
- **Pinned by:** the Heuristic suite's `projection_two_axes_{metal,cuda,amd}`,
  `q8_0_partial_{metal,cuda,amd}` and `conv_half_{metal,cuda,amd}`,
  `conv_metal`; kaun's `GptOss/prefill-512`; the Stage suite's `a core under
  another sum whose tile does not read it is the exact product` (D262); the
  Codegen suite's `tensor cores on CUDA › a convolution of three reduce axes
  takes a core and is the exact result`.

## D451. A float32 cast of an integer enters a narrow core

- **tinygrad:** `codegen/opt/postrange.py` (`_apply_tc_opt`: an operand is its
  own value, of the core's input type, or tolk's float32 cast of it, D105).
- **tolk:** `Postrange.Scheduler.tc_operand`.
- **Differs:** a float32 cast of an integer that the core's narrow input holds
  exactly (`Dtype.can_lossless_cast`, so `fp8` is refused) enters the core at
  that input. The cast through the narrow float is what a fold removed:
  `Cast(float32, Cast(bfloat16, int8))` is `Cast(float32, int8)`.
- **Reason:** (d): a Q8_0 product's codes (rune's L9). An `int8` times a
  `bfloat16` is exact in float32 barring overflow, as the float32 product of
  the program is. On fudge's RTX 5000 Ada, kaun's `GptOss/prefill-512-Q8_0`
  takes 141 ms against 2.11 s with the codes cast to float32 and the
  products off the cores. The decode step's Q8_0 products take no core and
  are unchanged. The row's fudge wall-time baseline is pending the scheduled
  quiet re-record.
- **Pinned by:** the Heuristic suite's `rows_codes_{metal,cuda,amd}`,
  `q8_0_partial_{metal,cuda,amd}`; rune's `Jit_gpu › quantised products`;
  kaun's `GptOss/prefill-512-Q8_0`.

## D467. A group pads a long reduce axis that 16 does not divide

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  grouping: the first of the first three reduce axes that splits by 16; none
  that 16 divides, no group); `codegen/opt/postrange.py` (`PADTO`, which only
  `BEAM` and the tensor cores apply).
- **tolk:** `Heuristic.group` and `Heuristic.unroll_whole`.
- **Differs:** of the first three reduce axes, those a load reads at unit
  stride first (D357), then the others, the group takes the first that splits
  by its threads (16, or D487's width), else the longest longer than 32 padded
  to a multiple of them by `PADTO`,
  its reduction's identity in the padded steps and its loads gated. So a
  unit-stride axis is padded rather than a strided one that 16 divides
  grouped. Axes of 32 or less are left to `unroll`, which unrolls them whole.
  The pad is tried on a copy and kept only where the group applies. An extreme
  or its position takes the same bits; a sum's order changes with the threads'
  parts, as any group's does. As with any group, no other split follows, so a
  second reduce axis of 32 or less that `unroll` would have unrolled stays a
  loop; no such kernel is measured. tinygrad leaves such a reduction to few
  threads that loop over the whole axis: `Prepare.split_reduceop` splits
  gpt-oss's 201088 logits, 2^7 x 1571 with 1571 prime, into 128 runs of 1571,
  one workgroup of 32 threads.
- **Reason:** (d): gpt-oss-20b's greedy decode step (`Layer_loop.greedy`), whose
  argmax over the vocabulary is two such reductions. rune's row
  `Cuda/argmax-bfloat16-201088` took 272 us and takes 29.5 us, one interleaved
  pair under load.
  Vocabularies of 151936 (2^7 x 1187) and 128256 (2^8 x 501) split the same way.
  AMD and Metal take the rule untimed. The longest of the axes that pad, not
  the first, serves a kernel whose reduces close axes of their own: whisper's
  timestamp rule sums 1501 exponentials beside the maximum of a split's 35
  partials, and with the 35 grouped each thread added all 1501: 79 us a
  decode step of large-v3-turbo on fudge's RTX 5000 Ada, and 9.4 us with the
  1501 grouped (DEBUG=2).
- **Pinned by:** the Heuristic suite's `max_vocab_{metal,cuda,amd}`,
  `sum_prime_{metal,cuda,amd}`, `sum_cols_prime_{metal,cuda,amd}` (16 threads
  first across the columns), `sum_33_{metal,cuda,amd}` (the shortest axis
  padded) and `sum_strided_prime_{metal,cuda,amd}` (a unit-stride axis padded
  before a strided one that 16 divides), and `a group's fastest threads read
  at unit stride where a load reads a reduce or output axis so` over them;
  `sum_17` and `matmul_half_ragged`, whose axes of 32 or less stay unrolled;
  `a group pads the longest of the reduces' axes`;
  rune's `Jit_gpu › an argmax or argmin over a length 16 does not divide is
  nx's position` and `› a sum and a product over a length 16 does not divide
  are exact`; rune's bench row `Cuda/argmax-bfloat16-201088` (baseline to
  record) and kaun's `GptOss/decode-step-cache-1024` and
  `GptOss/decode-step-Q8_0-cache-1024`, on fudge.

## D468. Ties of the linear order compare by numbers

- **tinygrad:** `codegen/late/linearizer.py` (`linearize`) sorts the nodes by
  their priority and, under `TUPLE_ORDER`, by `UOp.tuplize` (`uop/ops.py`), a
  tuple of the operation, the argument's repr, the type and the sources'
  tuples, whose comparison walks both graphs.
- **tolk:** `Linearizer.linearize` and `Ops.structure_ranks`.
- **Differs:** the order is the same, ties included. Under `TUPLE_ORDER` the
  kernel's nodes are first numbered in that order, once, by inserting them,
  sources first, into a treap over an order-maintenance list, where a node's
  sources compare by their numbers; ties then compare numbers. tinygrad walks
  the two graphs at each comparison, so a chain of operations each read once
  by the next costs the square of its length.
- **Reason:** (d): compile time. On fudge a chain of products and sums read
  once, ordered alone, takes over 120 s against 0.17 s at 100,000
  operations, in the same order; rune's compile row `long/pendulum-200`,
  cold, runs 20.48 G instructions against 16.58 G (Clang excluded). A kernel
  without long chains pays the numbering: tolk's linearize stage rows take
  30% to 55% longer and allocate about 30% more words.
- **Pinned by:** tolk's bench `chains/linearize`, on fudge;
  `Tolk.Ops › compare_structure › structure_ranks numbers a graph's nodes in
  its order` and `› structure_ranks numbers NaNs of different bits alike`.

## D469. Withdrawn

A long expression written into its reader was held aside behind a NUL and an
index, written out once in the statement that read it. An expression longer
than 4096 characters is now declared instead (D157), and a NUL in a caller's
text is no longer refused. A chain of calls of code written into their
readers, which D157 leaves unbounded as tinygrad does, renders in quadratic
time again. The frame slots of a host kernel cut into functions belong to
D466.

## D438. A gated load in a loop whose address a workgroup shares reads ungated on HIP for gfx12

- **tinygrad:** `renderer/cstyle.py` `CStyleLanguage.string_rewrite` renders a
  gated `LOAD` as `(gate?*(ptr):alt)`, on HIP as elsewhere.
- **tolk:** `Cstyle.pm_select_gated_loads`, in `Cstyle.hip_extra_matcher`
  for an architecture of the gfx12 family, rewrites a gated scalar load into
  an ungated load at index `clamp i 0 (n - 1)` and `where gate value alt`.
  The load must be from global or workgroup memory that is not volatile and
  holds a known n >= 1 elements, and its index `i` must satisfy two
  conditions:
  - `Uniform.varying` calls it the same for every thread of the workgroup;
  - it reads a serial loop's variable (`Loop` or `Weak`).
  `Uniform.varying` counts as the same in every thread: workgroup ids, loop
  variables of uniform bounds, kernel variables, constants, and loads of
  memory no thread of the launch writes, at such indices. The clamp is
  dropped where the index's bounds already lie in the buffer.
- **Differs:** where the gate fails, the thread reads the clamped element
  and the selection drops it; tinygrad reads nothing. That element is
  readable: a kernel's parameters and its workgroup memory are allocated
  whole, and a buffer of symbolic size is left as tinygrad renders it.
  - The clamp keeps the address the workgroup's under a gate each thread
    holds, as rune's SIMT attention's row gate; `where gate i 0` would make
    it each thread's.
  - Excluded loads, each left as tinygrad renders it:
    - A load at an index of a thread's id, or loaded at one: ungated, each
      holds its address and value in vector registers until used, and
      gpt-oss's expert matmul at prefill on gfx1201
      (`r_63_18_4_32_2_2_2_4_5_90_2`) took 10.7 ms ungated against 8.7.
    - A load whose index reads no loop variable: read ungated, each issues at
      the kernel's start and its value stays live until its use (float32
      attention under gpt-oss's mask on gfx1201, 66.6 µs against 50.8).
      Within a loop, such a load could be hoisted out with the same cost, and
      it has no trip's loads to issue with.
    - A vector load: a clamped index can start a vector the buffer does not
      hold whole, or off the alignment of its vectors.
    - A volatile buffer, which the host may change during a launch, so it is
      read only where the kernel reads it. tolk has no atomic load.
    - Every load on an architecture before gfx12: on gfx1100 the float32
      SIMT attention kernel spilled more scalar registers to vector lanes,
      and no RDNA3 or CDNA device has timed the rule.
  - Another thread may write the clamped element during the read. C leaves
    that race undefined; HIP compiles through LLVM, where a racing load yields
    an undefined value and has no other effect, and the selection drops it.
  - The values stored are tinygrad's.
- **Reason:** (d). C may not read `*(ptr)` where the gate fails, so Clang puts
  each gated load in a branch, AMD's backend puts the load's first use in that
  branch, and the loads go to memory one after another. rune's SIMT attention,
  which runs float32 attention on AMD, reads each key element of a tile through
  such a load; with this rule comgr (ROCm 7.0) issues a trip's 33 scalar loads
  under one wait on gfx1201. Whether LLVM loads them with scalar loads depends
  on its knowing the buffer is not written in the kernel. On the R9700,
  rune.thumper's `Amd/attention/jit-f32-decode-8x4096x64` took 174 µs where it
  took 468. `Amd/attention/jit-f32-8x2048x64-full` is unchanged, as are
  gpt-oss's decode attention, decode step and expert products, where no kernel
  takes the rule.
- **Pinned by:** `Tolk.Codegen › on AMD, a gated load in a loop is read
  ungated on gfx12 where a workgroup shares its address`; `Tolk.Codegen › a
  gated load takes its alternative where its gate fails` on every GPU and the
  host.

## D470. Variables of one name are one parameter

- **tinygrad:** `codegen/__init__.py` (`do_number_param`, `pm_number_params`)
  gives variables of one name one slot, each its own parameter, and
  `renderer/cstyle.py` (`CStyleLanguage._render`) names a parameter by its
  name and shape. A kernel that reads variables of one name stating different
  ranges declares the name twice: `Variable("s", 0, 2)` and
  `Variable("s", 0, 6)` read in one kernel render `const int s_, const int
  s_` for CUDA, which NVRTC refuses.
- **tolk:** `Codegen.pm_number_params`.
- **Differs:** variables of one name, whatever range or type each states, are
  the first one met, cast to each one's type: one parameter in the name's
  slot. A run binds the name to one value, which lies in the range of each,
  so the cast is exact.
- **Reason:** (e): tinygrad renders a kernel that the compiler of every
  target naming its parameters (CUDA, HIP) refuses, for a graph whose value
  is defined; the host and Metal read both from the name's slot.
- **Pinned by:** `Tolk.Codegen › programs › variables of one name are one
  parameter, past the parameters`; `Tolk_engine › link and run › a kernel
  reads one name of two ranges as one value` and `› a kernel reads one name
  of two types as one value`, and the same on each GPU; the law
  `Tolk.Launch_law › GPUs › on NV, a schedule's launches write what its graph
  states`, whose launches join members that read one name at two ranges.

## D471. A tensor core's trip reads a reduce axis inside its own K unrolled

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`
  returns after the core's upcasts and local split; its unroll of a small
  last reduce axis runs only on the path without a core, under
  `upcast_size() < 64`, which a core's upcasts exceed). `codegen/simplify.py`
  (`merge_ranges`) keeps two ranges apart when merging leaves a quotient or a
  remainder, as a 4-bit operand's bytes and a byte's two codes do.
- **tolk:** `Heuristic.unroll_beside`, after an unstaged core's splits (both
  branches of `Heuristic.tensor_cores`).
- **Differs:** on a kernel whose core takes one of several reduce axes
  (D450), each other reduce axis of `n` values that an operand reads at unit
  stride, and the core's own range at stride `n`, is upcast whole where `n`
  of the operand's values fit 16 bytes. A decoded MXFP4 operand splits the
  reduce into bytes and a byte's two codes, which the activations read at
  `2b + c`: the core takes the bytes, and its trip takes both codes, so a
  thread reads the four adjacent activations its fragment holds in one load
  where two trips read them two apart. A kernel of one reduce axis,
  tinygrad's, is unchanged; of the cases' kernels it reaches only the routed
  MXFP4 products (`routed_*`), on every GPU.
- **Reason:** (d): kaun's `GptOss/prefill-512` on fudge's RTX 5000 Ada, CUDA
  (128-byte L2 fetches), 85.00 ms to 78.52 (medians of 4 interleaved
  rounds). Without 128-byte L2 fetches, which rig.nv does not request, the
  unrolled gate and up product of gpt-oss's experts ran 7% slower in a
  standalone harness (2 workgroups an SM at 174 registers, against 3). AMD
  and Metal take the same unroll unmeasured.
- **Pinned by:** the Heuristic suite's `routed_{tiles,shared,past_bound}`
  and `routed_run_{16,32,64,128}` on `metal`, `cuda` and `amd` (the recorded
  opts and kernels), and `the tensor core's floor`, whose model states the
  unroll and covers it; the Codegen suite's `tensor cores on each GPU › CUDA ›
  a product by 4-bit codes, the core on their bytes and a byte's two codes
  unrolled, is the exact result`; kaun's `GptOss/prefill-512`.

## D472. A large reduction another reduces over is stored

- **tinygrad:** `schedule/indexing.py` (`run_rangeify`: a reduction is
  stored where a broadcast ends its ranges, "fusion decision: REDUCE before
  the broadcast", or where its readers index it differently; otherwise it
  takes its single reader's ranges, those of a reduction that reads it
  included).
- **tolk:** `Indexing.assign_ranges`, `Indexing.nested` and
  `Indexing.long`.
- **Differs:** a reduction of at least `2^21` terms (`nested_terms`, its
  operand's elements, a symbolic size at its largest) and of more than 32
  terms an element (`inline_terms`), whose readers index it by a range
  that a reduction reduces, is stored with ranges of its own, where tinygrad
  runs its terms inside the reading reduction's loop. The host is not told
  apart, since that would take the target into rangeify, which names devices
  without kinds.
- **Reason:** (d): the inlined reduction runs a loop inside one that no
  thread split, tile or tensor core takes. On fudge's RTX 5000 Ada, rune's
  `Cuda/sum-of-contraction-4096x512x4096` takes 46.8 ms inlined and 1.0
  stored. Below `2^21` terms the inlined loop takes about a launch's 10 us.
  A short reduction costs little inlined: Q8_0's sums of blocks of 32 codes,
  stored, took kaun's `GptOss/prefill-512-Q8_0` from 86 ms to 250. On the
  host storing gained 4x on the row sums of 2048 x 2048 and lost 1.4x on a
  sum over 64 of squared sums.
- **Pinned by:** rune's `Rune.jit on a GPU › a reduction another reads ›
  read by a reduction adds one kernel, its own` (a contraction and a sum,
  each under six readers), `› of 128 x 128 x 64 terms stays inside the one
  that reads it`, `› of 32 terms an element stays inside the one that reads
  it`, `› holds its bound`, `› over a broadcast stores no broadcast`; the
  rows `Cuda/sum-of-contraction-4096x512x4096`,
  `Cuda/attention/jit-jvp-bf16-8x2048x64-full` and
  `Cuda/sum-of-row-sums-4096x4096`, and kaun's `GptOss/prefill-512-Q8_0`.

## D473. Effects are ordered by the states they name

- **tinygrad:** `schedule/prepare.py` (`prepare_rangeify`), which keeps the
  program's `AFTER`s as written, but for `pm_inline_calls`'s rule
  (`uop/ops.py` `resolve_returned_after`) that makes an `AFTER` of a call's
  output over the call's stores that output's stored value;
  `schedule/__init__.py` (`create_schedule`), which orders a kernel after the
  states its arguments name, and an `AFTER` after its prior state, its joins
  and its kernels, a kernel already on an earlier state of the storage being
  no write of a later one.
- **tolk:** `Prepare.sequence`, run in `Prepare.prepare_rangeify` after the
  calls are inlined and in `Schedule.create_linear_with_vars` before
  `Schedule.transform_to_call`; `Prepare.pm_inline_calls`, which has no
  `resolve_returned_after` rule; `Schedule.view_of`; `Schedule.schedule`;
  `Ops.after` and `Ops.sink`.
- **Differs:** a graph's stores, calls and loops of calls are its effects,
  each run once, ordered by the states they name alone: an effect runs after
  the state of each `After` listing it whose storage it writes, and after the
  effects whose states it reads. The order of a sink's items orders nothing.
  `Prepare.sequence` states that order in the graph. Each storage's states
  become one chain, each state listing the one effect, or the stores of one
  write marked disjoint, that makes it over the state before; every read
  names the state of the chain it reads; a destination, and an argument a
  call writes, names the state its write is over. A join, and a value ordered
  after effects, becomes the value under it (D202, D475), the effects it
  lists still run; an effect listed again over a state that already holds it
  lands at its first place (D427); an `After` listing no write of its storage
  names the latest state of it its effects read or run after (`Ops.after`),
  and orders nothing. The sink lists each chain's
  last state, the effects that write no storage, and its other items. Effects
  that no state orders keep the order a walk of the graph reaches them, which
  writes what any other order would, since two writes over one state are
  refused (D474): values and refusals do not depend on the order of a sink's
  items. Which kernels share a launch (D404) and which read a cut's copy
  (D160) follow that walk, so a sink listed in another order can group the
  same kernels into other launches. A store's destination is its storage as the store runs
  (`Schedule.view_of` drops its states), so a store into a run of a state
  stores into a view (D182); `create_schedule` orders the kernels along the
  chains (D160). `after x1 [store (x 2..4) (x1[0:2] + 1)]` stores over `x1`,
  and a call given `x` to write and `x1` to read, listed over `x1`, reads `x1`
  as `memmove` reads its source; `after x [store y v]` is `x` as it was.
  tinygrad's recorded `*_prepared` graphs differ wherever `sequence` changes
  a graph: chains where an `AFTER` lists several effects, an inlined call's
  stores as a chain where `resolve_returned_after` takes the stored value,
  and the sink listing the chains' ends. A program whose every effect is a store
  listed alone by an `After` over its storage, the only write of it, is left
  as it is, which `sequence` checks in one pass over the graph's nodes
  (`Prepare.chained`); any other pays the walk that sequences it.
- **Reason:** (b) and (e). rune's loops (`Loop`) pass a carry's first state
  to the call a loop runs and list the loop over the carry's storage, which
  needs an order taken from the states the program names (D474). tolk's own
  rules ordered a kernel listed over a later state than its arguments name by
  where a walk of the sink first met it, so a kernel writing `x` and `y`, listed by
  `after y [k]` and `after x1 [k]`, left `0 2 4 6` or `1 3 5 7` in `x` by the
  order of the sink's two items, and `after x1 [store (x 2..4) …]` and
  `after x [store y v]`, which `Ops.after` gives a meaning, raised internal
  errors.
- **Pinned by:** the `Tolk.Schedule` suite: `create_linear_with_vars › laws ›
  the order of a sink's items changes no result` and `› a schedule writes
  what its tensors compute`; `create_linear_with_vars › two writes over one
  state › an effect listed after another lands after it, whichever the sink
  reaches first`; `create_linear_with_vars › joins › a store
  naming an older state, listed over a later one`, `› a store through a run
  of a join`, `› an after listing only a store into other storage is that
  state` and `› stores into one other storage land as listed`; the
  `Tolk_engine` suite's `calls of views › a call given an older state to
  write and a later to read reads it before the call`, `› a kernel listed
  over two storages names the later state in either listing` and `› runs of
  one other storage land as listed in an after`; `Tolk.Launch › an After
  lists its stores in the order they land › a store listed twice lands at
  its first place`; the `Tolk.Prepare` suite's recorded `*_prepared` graphs
  that `sequence` changes, whose `.tolk` files name this entry.

## D474. Two writes over one state are refused unless one runs after the other

- **tinygrad:** `schedule/__init__.py` (`create_schedule`), which raises
  "sibling AFTERs detected in assign graph" at a second `AFTER` over one
  state, the state `_unwrap_src` reaches through movements, so that `AFTER`s
  over views of one state count (c00c9ada9);
  `test/runtime/test_after.py::TestAfterCounterexamples::test_sibling_afters_rejected`.
- **tolk:** `Prepare.sequence`; `Schedule.schedule`; `Ops.after`.
- **Differs:** tolk follows tinygrad's refusal: two writes over one state, an
  `After` over a view of a state being a write over that state, raise
  `Invalid_argument` when the graph is prepared or scheduled, naming the
  storage's slot, both writes and the two ways to order them ("list one over
  the After of the other, or list both in one After"), whichever order the
  sink lists them in. `create_schedule` refuses a kernel graph whose two
  `After`s of one state list different kernels. A write no `After` of its
  storage lists is over the state its destination names, so two such stores
  into one storage are two writes over one state, refused in either order. In
  a program, a write is no second write over a state where it reads what the
  write that made the state left, or a value ordered after that write, where
  its destination or an argument it writes names that state, where an effect
  listed before it in its `After` made it, or where that state holds it
  already: `sequence` puts it over that state, a chain. tinygrad refuses
  these: its test's `x.after(a.store(2))` beside `a = x.after(x.store(1))`
  raises, where tolk stores `2` over `a`.
- **Reason:** (b): each carry of a rune loop (`Loop`) is stored first as
  `first = s.after(s.store init)`, passed as the argument the loop's call
  writes, and named after the loop as `s.after(loop)`: two `AFTER`s over `s`,
  which tinygrad refuses, in every program with a loop.
- **Pinned by:** the `Tolk.Schedule` suite: `create_linear_with_vars › two
  writes over one state` (overlapping runs refused in either order, a read
  beside another write, the message, the same writes listed in one `After`, a
  write over a state beside one it reads the result of, writes no `After`
  lists refused, a write no `After` lists over the state its destination
  names, a kernel graph refused), and `› laws › the order of a sink's items changes no result`,
  which covers refused and computed programs; the program
  `write_over_written`, tinygrad's test, whose linear and values tolk
  computes; `Tolk.Launch › an After lists its stores in the order they land ›
  an After of a state and one of a view of it raise, whole first` and `› …
  view first`; the `Tolk_engine` suite's `calls of views › a whole state
  holds the sibling store its own store reads`, and the refusals beside it:
  `› a read of a whole state beside a sibling After is refused`, `› a value
  reading a join of sibling writes is refused`, `› a loop reading a join of
  sibling writes is refused`, `› a loop reading a sibling state's run is
  refused` and `› Afters over runs of a bitcast of one state are
  refused`; rune's suites, whose loops write over their carries' first
  states.

## D475. A value ordered after effects is the value

- **tinygrad:** `schedule/rangeify.py` (`handle_after`), which raises
  "kernel boundary requires storage" at an `AFTER` of a value held by no
  storage; `test/runtime/test_schedule.py::TestSchedule::test_value_order_dependency_requires_storage`.
- **tolk:** `Prepare.sequence`, which replaces such an `After` by its value
  before `Rangeify.handle_after`, whose refusal a sequenced graph does not
  reach; `Ops.after`.
- **Differs:** an `After` whose first source is no storage nor a state of
  it names no state: it holds that value, and the effects it lists run where
  the states they name order them (D473), and an effect that reads the
  value runs after them, so it is no second write over a state they made
  (D474). `UOp.const(1).after(c[0].store(5))` is `1`, and the store runs.
- **Reason:** (e), as D473: `Ops.after` gives the program a meaning, and
  tinygrad refuses it. No raven consumer emits it.
- **Pinned by:** the `Tolk.Schedule` suite's `create_linear_with_vars ›
  boundaries › a constant ordered after a store is the constant, and the
  store runs`; the `Tolk_engine` suite's `calls of views › a value ordered
  after effects is the value` and `› a state over a view a loop moves,
  ordered after effects, is read each trip`.

## D476. A buffer only one loop's calls take lives within a trip

- **tinygrad:** `schedule/memory.py` (`memory_plan_rewrite` takes each
  entry's buffers, an `END`'s from its sources after the first, so a
  buffer used only inside a range is not planned there).
- **tolk:** `Memory.steps` and `Memory.life`, which give
  `Memory.memory_plan_rewrite`'s lifetimes.
- **Differs:** the calls of a loop around calls take steps of their own,
  in the order of one trip. A buffer that only the calls of one loop take,
  as call-local storage of the loop's call is, lives from the first of them
  that takes it to the last, so buffers of one trip whose lives do not
  overlap share bytes, and the loop holds its trip's peak for its whole
  run. A buffer that calls inside and outside a loop take, and a back
  edge's flag, live for the whole loop. The plan relies on no trip reading
  such a buffer that an earlier trip left, a precondition of
  `Memory.memory_plan_rewrite` it does not check: rune passes a loop's
  carries, rows and results as the call's arguments, stored before the
  loop or read after it. tinygrad's planner sees no buffer used inside a
  range at all (D41).
- **Reason:** (b): rune's compiled scans whose step takes a gradient, as
  gradient accumulation over chunks of a batch does, and remats inside
  such steps. The gradient of 8 remat'd blocks per chunk, scanned over 2
  chunks of a batch of 256 (width 32, hidden 256, on the host), held
  2,441,344 bytes, the sum of a step's buffers, and holds 360,576.
- **Pinned by:** the Memory suite (`test/schedule/memory`):
  `memory_plan_rewrite › loops › buffers only one loop's calls take share
  bytes within a trip`, `› a buffer a loop's calls and a later call take
  lives the whole loop`, `› a buffer of a nested loop shares bytes within
  the outer trip` and `› a back edge's flag lives the whole loop`, and the
  law over drawn schedules, whose calls run in loops nested two deep; the
  Engine suite's `runs › a run holds its storage whole, no less than its
  buffers live at once`; rune's `rune remat › memory › a step that takes a
  gradient plans each trip`, and, for the precondition, rune's scan law
  `compiled › carries through nested loops, branches and scatters`, which
  compiles drawn loops against eager.

## D477. Buffers are placed largest first, each in the smallest gap that holds it

- **tinygrad:** `schedule/memory.py` (`memory_plan_rewrite`: a
  `TLSFAllocator` per lane allocates each buffer as its life starts and
  frees it as its life ends); `runtime/support/memory.py`
  (`TLSFAllocator`).
- **tolk:** `Memory.memory_plan_rewrite`; tolk has no TLSF allocator.
- **Differs:** a lane's buffers are placed largest first, ties in the
  order they first appear, each at the start of the smallest gap that
  holds it between the buffers already placed whose lives meet its, else
  past the last byte of those (Pisarchyk and Lee, 2020, §5.2, Greedy by
  Size for Offset Calculation). The lifetimes, rounding, lanes and arenas
  are tinygrad's (D41, D476). Placing each buffer as its life starts can
  leave a gap below a long-lived buffer too small for a later, larger one,
  and the arena then exceeds the bytes live at once. Placing costs `O(b²)`
  for `b` buffers of a lane.
- **Reason:** (b): rune's compiled gradients of remat'd blocks, with the
  order of D183. On the host, the gradient of a chain of 8 remat'd blocks
  takes 557,056 bytes live at once; the TLSF placed them in an arena of
  786,432, and this placement in 557,056 (589,936 bytes allocated against
  819,312, over the 655,360 that `rune remat › memory › a chain's backward
  pass computes one block at a time` allows).
  Over the 31 recorded tinygrad schedules no arena is larger than
  tinygrad's. On random schedules it is no promise: of 1,000, greedy was
  larger than TLSF on 15, by up to 20.6%.
- **Pinned by:** the Memory suite (`test/schedule/memory`):
  `memory_plan_rewrite › rules › buffers are placed largest first`; rune's
  `rune remat › memory › a chain's backward pass computes one block at a
  time`; the laws (`› laws`), that buffers whose lives meet share no byte
  and that each recorded plan's arena is at most tinygrad's; the recorded
  plans whose places differ carry this entry in their `.tolk`.

## D478. An output that stores a stage whole holds it

- **tinygrad:** `schedule/rangeify.py` (`bufferize_to_store`, which gives
  every stage storage of its own, `UOp.alloc`); `schedule/prepare.py`
  (`forward_call_outputs`, which makes an output the storage of a `STAGE`
  stored into it, and of no other value). tinygrad's own outputs never meet
  this, since `Tensor.linear_with_vars` makes every reader of a realized
  tensor read its storage; an assign of a value another reader stores apart
  copies it: `out.assign(c)` of a contraction `c` realized with `c.sum()` is
  4 kernels, `c` realized with its sum 3.
- **tolk:** `Rangeify.buffers`, `Rangeify.bufferize_to_store`, the
  store rule of `Rangeify.pm_add_buffers` and `Rangeify.name_held`.
- **Differs:** a stage that an item of the graph, `AFTER(b, STORE(b, s))`,
  stores whole into `b`, in the stage's order, at its type and device, is
  stored into `b` where `b` is storage that nothing else in the graph reads
  or writes, its state after the store included, that the stage's value
  does not read, and where another item reads the stage; the store, then a
  copy of `b` into itself, is dropped, and the item names the state the
  stage's store leaves in `b`. A stage the store alone reads, as a
  maximum of a broadcast sum is once the maximum folds to the sum, keeps its
  own storage: dropped with its only reader, its store would be lost. A
  stage stored narrower than its readers read it (`Ops.widens`) keeps
  its storage. Where tinygrad stores the stage apart and copies it into
  `b`, tolk writes `b` once. tinygrad's own fix, every reader of a returned
  value reading its output, belongs to rune's lowering, where it would make
  a reader of a cheap value read the whole output and wait for the output's
  kernel; the rule acts only where rangeify stores the value anyway.
- **Reason:** (d): rune hands tolk each output as such a store into fresh
  storage, and its readers read the value, so a returned value that another
  reader needs stored was copied whole into its output. On fudge's RTX 5000
  Ada, rune's `Cuda/logits-and-loss-2048x1024x8192` takes 2.06 ms copied and
  1.83 stored once.
- **Pinned by:** the `Tolk.Schedule` suite's `create_linear_with_vars ›
  outputs that hold a stored value`; the Rangeify suite's `get_kernel_graph ›
  laws › a scheduled function writes what its tensors compute`, whose
  programs return values along the way, and its `, for` rows `[2], then
  sum 0, expand, max 0` and `[4; 4], then sum 1, return, x - sum 0`; rune's
  `Rune.jit on a GPU › a reduction another reads › returned too is stored
  once`; the row above.

## D479. A reduce starved of threads groups past 2048 outputs

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  grouping: at most 2048 outputs, 240 on QCOM, whatever the reduce).
- **tolk:** `Heuristic.group`, `Heuristic.group_least` and
  `Renderer.t.stream_threads`.
- **Differs:** a GPU kernel that neither streams nor takes a core also
  groups its reduce past 2048 outputs where its outputs are fewer than
  `stream_threads` (32768 on sm_89 and the targets that take its constants,
  16384 on Metal), each of its reduces sums at least `group_least` values an
  output, and it is not a contraction (`Heuristic.contraction`, D89's). One
  thread an output then leaves the part short of threads, and each runs its
  reduce as one serial chain at a load's latency (Larsen and Henriksen, FHPC
  2017, §2 and §3.1). The group's 16 threads each sum a 16th, then combine 16
  partials, so `group_least` is `16 * 16`, 256: the measured crossover lay
  between 128 and 512 values at 16384 outputs, and 256 took no measured
  loss. A contraction keeps tinygrad's rule, since a group drops the upcasts
  that reuse its operands: a float32 product of 128 by 4096 by 128 took 0.40
  ms and 0.89 grouped. rune casts each operand before the product (D29,
  `Lower_reduce.sum`), so its products are seen; a product summed through a
  cast, `REDUCE(CAST(MUL))`, is not, and groups. The predicate also takes a
  product without reuse, a sum of squares or a row's dot product, which
  keeps the rule where the stream's layout refuses it. An extreme or an
  integer reduction keeps its bits; a float sum's order changes with the
  group's partials, fixed per target as D89's constants are.
- **Reason:** (d): the reductions of a prefill of 2049 to 32767 tokens on a
  GPU. On fudge's RTX 5000 Ada (nsys kernel medians of a call), the row's
  maximum that starts `Nx.logsumexp`, `Nx.softmax` and `Nx.rms_norm`: at
  4096 rows of 2880 a logsumexp call took 171.5 us and 29.0. rune's election
  of a `Set` scatter of rows, which kaun's `Cache_index.put` runs for each
  layer's keys and values, is one reduce of `k` comparisons a row, which
  replaces rune's two kernels of about `√k` parts: at 512 rows 3.13 to 1.98
  us (grouped by tinygrad's rule), 2048 11.61 to 7.30, 4096 29.8 to 21.8,
  8192 even, and a call with its stores 1.24-1.33x faster. Sums of 8192 rows of 32
  and of 16384 rows of 256 do not move. The rule costs where the group gives
  up the reuse rune's parts kept by an upcast: elections of 16384 and 24576
  rows take 1.40 and 1.88 times as long a call, and an election at `p > 1`
  indices whose `p k` reaches `stream_threads`, no longer grouped, up to 3.5
  times (`p = 2`, `k = 16384`); no library caller elects at `p > 1`. A split
  decided where the renderer is known would close both. AMD and Metal take
  the rule untimed.
- **Pinned by:** the Heuristic suite's `past 2048 outputs` cases:
  `sum_rows_long_{metal,cuda,amd}`, `sum_rows_16384_{cuda,amd}` and
  `sum_rows_eight_{metal,cuda,amd}` (whose ranges together count past a
  machine integer) group; `sum_rows_short_{metal,cuda,amd}`,
  `sum_rows_240_{metal,cuda,amd}`, `sum_rows_pair_{metal,cuda,amd}`,
  `sum_rows_16384_metal` and `matmul_long_{cuda,amd}` do not; rune's
  `Rune.jit on a GPU › scatters › a Set scatter of rows compiles to eager's
  bits` (an election of 4096 rows); rune's bench rows
  `Cuda/scatter-4096-rows-into-donated-bf16-8192x512`,
  `Cuda/scatter-16384-rows-into-donated-bf16-32768x512` and
  `Cuda/logsumexp-float32-4096x2880` (baselines to record).

## D480. A float sum adds at most 256 terms into one running sum

- **tinygrad:** `codegen/__init__.py` (`lower_reduce`), which gives each
  reduce one register accumulator over all of its ranges, however many
  terms they take.
- **tolk:** `Codegen.nest_chains`, which `Codegen.full_rewrite_to_sink`
  applies after the expander, before `pm_reduce_local`; `Codegen.chain`,
  `Codegen.held`.
- **Differs:** a float `Add` reduce whose ranges, its unrolled lanes and a
  tensor core's products counted, take more than 256 terms a thread
  becomes an inner reduce over its innermost ranges, at most 256 terms from
  `+0`, inside an outer one over the rest, which adds the inner sums in
  order. A range that does not fit splits at its greatest divisor that
  fills at least half the room left, else at the room, its last block's
  trips past the range gated off, every index over the range, read or
  written, in any memory, as well as their terms, so that none reaches
  past its buffer (`4096` trips of 4 lanes are 64 blocks of 64; `16381`
  trips are 64 blocks of 256, the last of 253). Every reduce
  over one range splits it alike, at the least room any of them leaves, so
  siblings keep one loop. A reduce whose lanes leave no room, past 128,
  nests with no range inside: each trip's lanes sum from `+0`, in blocks of
  256, and the outer reduce adds them. A range a uniform gate bounds,
  `where (c, n, 0)` (D305), splits at its `n` trips, its blocks' loop
  bounded by `where (c, ⌈n / d⌉, 0)`, and a sum over it ends that loop and
  its block's, not the ranges the gate reads. Two kinds of sum keep one
  running sum, and their bounds are a single chain's: a sum over a range
  whose trip count no constant or gate's selection of constants states, as
  a count read from memory, and, on a renderer with local threads, any sum
  whose accumulators pass 320 bytes a thread, float32 past 80 and float64
  past 40, a staged tensor core's tile among them, since a second set would
  spill the thread's registers (on CUDA the staged tile's 128 float32
  accumulators take 228 registers alone, and nested 255 with 560 bytes of
  spills). The kernel's name, which its optimisations give, does not
  change.
- **Reason:** (e) and (b): nx's kernel contract bounds a float sum of `n`
  terms by `γ(β(n)) Σ|x|`, `β(n) = ⌈5 min(n, 256) / 4⌉ + ⌈n / 128⌉`, on
  every library (`Nx_kernel`'s error bounds), and one running sum exceeds
  it past 324 terms. The (b) consumer is outside this repository: tessera's
  `validate --training` derives its margins from that bound, and its float32
  weight gradients over 16384 rows sat at 0.975 of the old one. A bfloat16
  product of 1760 × 440 × 16384 on CUDA
  goes from a relative error of 2.37e-5 against float64 to 2.86e-7.
- **Pinned by:** rune's `rune jit › classes › a long sum holds the blocked
  bound` and `› a weight gradient holds the blocked bound`, which fail with
  one running sum (terms that lose half an ulp each in it), and `Rune.jit on
  a GPU › a long sum and a weight gradient hold the blocked bound`; the
  `codegen` suite's `stages › every long float sum is in blocks of 256
  terms`, over every kernel it lowers, and its `› a padded block's trips
  read nothing past the buffer`, `› a padded trip reads nothing past a held
  tile of scales` (shared memory, on sm_89) and `› a padded trip reads
  nothing past a host's decoded panel` (registers); its `reduces under a
  uniform gate › a sum of 2880 terms a row runs in blocks, the gate bounding
  their loop`, `› a padded trip of a sum of 2900 terms reads nothing past
  the buffer` and `› on the host` and `› on CUDA, equal the reference ...
  k = 2900` (blocks padded under the gate); the Stage suite's `a
  staged core's padded trips read no global memory`, over a staged core's
  copies; the `codegen` goldens `readout_*`, `sum_all_clang`, `matvec_clang`
  and `linearizer_fail_1_clang`, whose `.tolk` hold the blocked loops.

## D481. A stage the buffer limit kept is inlined into readers stored as they are

- **tinygrad:** `schedule/rangeify.py` (`remove_bufferize`, which keeps
  a stored value whose value reads more than three buffers, whoever reads
  it), and `get_kernel_graph`, which decides each read of a stage once its
  producers are decided and before its consumers are.
- **tolk:** `Rangeify.remove_bufferize`'s `limited`, with
  `Rangeify.stored_readers` (whether every read of a stage is computed in a
  value stored as it is), and the `cover` of
  `Rangeify.read_copies`' `share`, summed per axis in `Rangeify.stage_reads`.
- **Differs:** a `Removable` stage whose value reads more than three
  buffers is inlined, the other tests of `remove_bufferize` applying, when
  every read of it is computed in a value stored as it is (a store, the
  sink, a stage kept `Whole` or `Argument`) and its reads compute each of
  its elements at most once in all. A read's cover is the unfloored count
  `read_copies` computes, the elements its index runs over by the stage's,
  `0.5` for a read of half of them, and infinite where a reduction runs the
  read. Inlined there, the value loads no buffer more often than storing it
  does, and no decision is left that its buffers could change: a reader
  that is a stage still to decide could be pushed past the limit by them
  and stored in its place, once for each reader, as `Nx_wide.sum`'s chunks
  would be. A `Broadcast` stage stays kept, since it is computed again along
  the ranges its index does not count, and the limit is the only bound on
  that (a 4-buffer value broadcast along 4096 rows took 11.3 ms on the host
  inlined, 0.64 stored); so does a value read in a reduction, which would
  move its work into the reduction (a sum of two halves of a 4-buffer value
  over 4096 rows of 64 took 20.4 us on CUDA inlined, 13.4 stored).
  The decision stays in tinygrad's one pass and its order.
- **Reason:** (d), to re-measure on fudge. gpt-oss-20b's prefill MoE block
  on CUDA stored gate_up's bias add, `Y[slot[r]] + bias[ids[r]]`, as a
  scatter of its own; inlined into its store, the block takes one kernel
  fewer and its bits are equal. `GptOss/experts-at-prefill-512` went from
  1.80 to 1.71 ms, its allocated words unchanged; its fudge baseline holds words only, so the entry is to
  re-measure on fudge until the scheduled re-record gives it a wall
  baseline there. AMD and Metal take the rule unmeasured.
- **Pinned by:** the Rangeify suite's `get_kernel_graph › buffer limit`
  (a value of four storages a store reads in halves is one kernel, and one a
  materialisation reads in halves is computed there; one that stages read,
  one a reduction reads in halves and one read where it is broadcast stay
  stored), and the bench row `GptOss/experts-at-prefill-512`.

## D482. CUDA merges byte loads and stores into words

- **tinygrad:** `codegen/late/coalesce.py` (`memory_coalescing`, which on a
  GPU merges `float`, `half`, `int`, `uint` and the 8-bit floats into
  vectors of 4 and 2 and leaves bytes scalar); `renderer/cstyle.py`
  (`CUDARenderer`, which writes a vector of bytes as a struct of `unsigned
  char` members).
- **tolk:** `Coalesce.memory_coalescing`, through `Renderer.t.byte_words`;
  CUDA's language in `Cstyle` (`word`, `render_dtype`, `render_index`, the
  `Stack` rule, `cuda_kernel`'s vector definitions).
- **Differs:** where the renderer writes bytes as words, CUDA's and so
  NV's, adjacent `uint8` loads and stores merge into accesses of 4 and 2
  under the alignment rule of the other merges. A vector of 2 or 4 `uint8`
  is an `unsigned short` or a `uint`: lane `i` is read as `(unsigned
  char)(w >> 8i)`, and a vector is built as an or of its lanes shifted by
  `8i`. An emulated 8-bit float's vectors of 2 and 4 bytes are words too.
  Loads of `int8` stay apart; the host keeps D187, and Metal and HIP merge
  no bytes.
- **Reason:** (d). A merged vector of bytes, in CUDA's struct form, needs a
  guard against LLVM's load combine, which folds an or of bytes in
  descending addresses, read from vector loads, into one load at the
  address of the first byte's vector, wrong where that byte is not its
  vector's first: upstream NVPTX and AMDGPU fold the struct form so, and
  NVRTC 13.4 does not, a compiler's behaviour. A word leaves no vector of
  bytes to fold, so it needs no guard. With rune's MXFP4 decode in
  `bfloat16` (D483), an MXFP4 core's lane reads its codes two bytes at a
  time; loaded byte by byte, gpt-oss-20b's prefill at 512 tokens on fudge's
  RTX 5000 Ada took 49.9 ms against 48.3. TESSERA's distillation step
  draws its dropout masks as bytes, four a thread over rows of 440: stored
  byte by byte its two draws of eight masks took 15.3 and 10.1 ms a step,
  as words 4.9 and 3.2. Metal and HIP keep bytes apart, unmeasured.
- **Pinned by:** the Codegen suite's `byte assemblies on CUDA › an int32
  assembled from bytes in descending addresses is exact` (on the machine's
  first CUDA GPU) and `› writes a thread's four bytes as a word read by
  shifts`; its `words of bytes on CUDA ›` `bytes copied from any start,
  stride and phase keep their values`, `› bytes stored at any phase keep
  their values`, `› a gated word of bytes is stored where the gate holds`,
  `› words of bytes read in descending addresses keep their values`, `› a
  gated word reads its bytes where the gate holds, zeros elsewhere`, `› an
  emulated float's selected lanes, stored as a word, keep each code` and the
  sources they write; the Coalesce suite's `lengths › on a GPU that writes
  bytes as words, eight consecutive` (byte loads, byte stores) and `› on a
  GPU that writes bytes as words, int8 loads stay apart`; kaun's
  `GptOss/prefill-512`, `prefill-12` and `experts-at-prefill-512` rows, and
  kaun's decode bench rows
  `Quant/cuda/mxfp4-product-of-16384-1-bfloat16-tokens` and `-512-`,
  their fudge baselines pending the scheduled re-record.

## D483. A renderer states whether its multiplies keep subnormals at full rate

- **tinygrad:** `renderer/__init__.py` (`Renderer`, which states no fact
  of subnormals).
- **tolk:** `Renderer.t.full_rate_subnormals`, true in `Cstyle.cuda`'s
  renderer (CUDA and NV), false elsewhere.
- **Differs:** the renderer states whether a float multiply takes a
  subnormal operand unflushed and as fast as a normal one. tolk reads it
  nowhere; rune's lowering builds an MXFP4 code's `bfloat16` through one
  (`Lower_reduce.e2m1_bfloat16`) only where it holds.
- **Reason:** (d), for rune's `Lower_reduce.entered`. On fudge's RTX 5000
  Ada the decode through the subnormal took gpt-oss-20b's MXFP4 gate and up
  product at 512 tokens from 919 to 831 us, where a decode free of
  subnormal operands took 888: two thirds of the gain lost, for one formula
  on every target. The model's prefill took 49.9 ms at 512 tokens against
  54.1 and 26.6 at 12 against 29.6. It is false for the host, where x86
  multiplies a subnormal through a microcode assist: the same decode made a
  routed product of 64 bfloat16 tokens on 8 experts of 512 by 512 take 5.0
  ms against 3.0; for Metal,
  whose specification lets arithmetic flush subnormal operands to zero (MSL
  4.1 §8.1, read, not run); and for AMD, unmeasured.
- **Pinned by:** rune's `Jit_gpu › products › an MXFP4 product of bfloat16
  rows decodes each code at each scale exactly` and its host twin in
  `Jit › classes`, for the values; for the speed, kaun's `GptOss/prefill-512`
  and `prefill-12` rows on CUDA and its decode bench row
  `Quant/host/routed-product-of-8-experts-of-512-64-bfloat16-tokens` on the
  host, their fudge baselines pending the scheduled re-record.

## D484. A renderer states the most threads a workgroup holds

- **tinygrad:** `renderer/__init__.py` (`Renderer.local_max`, a bound on
  each axis, and no bound on a workgroup's threads over all of them).
- **tolk:** `Renderer.t.local_threads_max`: 1024 in `Cstyle.cuda`'s
  renderer (CUDA and NV) and `Cstyle.hip`'s, 1024 in `Cstyle.metal`'s but
  512 on Apple2 and Apple3, 1 in `Cstyle.clang`'s; read by
  `Heuristic`'s columns layout (D89) and by
  `Postrange.Scheduler.apply_opt` (D488).
- **Differs:** the renderer states a workgroup's thread bound, which CUDA
  ("Maximum number of threads per block"), HIP (its hardware features) and
  Metal (its feature set tables, "Maximum threads per threadgroup") state
  apart from each axis's. The columns layout takes it in place of the 1024
  it held as the square of a SIMD group's 32 threads; `apply_opt` refuses
  a local split or pad past it, so no search candidate passes it. On CUDA,
  RDNA, Mac2 and Apple's GPUs from family 4 it is the 1024 both held
  before, so their kernels are unchanged; on Apple2 and Apple3 the columns
  layout's workgroup is 512 where it was 1024.
- **Reason:** (b), for rune's `Lower_index.workgroup`, which sizes a
  scatter's workgroup by the renderer: reading the x axis's `local_max`,
  unbounded on HIP and Metal, it built workgroups of more than 1024 threads
  there from 2^19 updates, past what either launches. The columns
  layout reads it as D89 states, where a CDNA wave of 64 threads, squared,
  would make workgroups of 4096.
- **Pinned by:** the Heuristic suite's `streaming › a product's workgroups
  hold at most a block's threads on every GPU`, gfx942 included; the Cstyle
  suite's `a renderer states its SIMD group's threads and a workgroup's
  most`; the Search suite's `a candidate's workgroup holds at most its
  renderer's most threads`; rune's `Lower_index › workgroups › a scatter's
  workgroups hold at most a block's threads on every GPU`. AMD and Metal are
  unrun.

## D485. Kernels whose sources differ in their names and comments alone take one binary

- **tinygrad:** `renderer/cstyle.py` (`CStyleLanguage._render`, which names
  the kernel's function after the sink's `function_name` and each parameter
  after its shape, `data0_4096`, and `base_rewrite`'s `SPECIAL` rule, which
  writes a hardware index's size in a comment, `/* 64 */`); `device.py` (`Compiler.compile_cached`, a binary kept
  under its source); `runtime/ops_cuda.py` and `engine/realize.py`
  (`get_runtime`), which find the kernel in its binary by `function_name`.
- **tolk:** `Codegen.compiled_text`, `Codegen.trades` and
  `Codegen.without_display`, `Codegen.symbol`, `Codegen.host_entry`,
  `Cstyle.param_name`; `Tolk_engine.launch`, which runs
  `Codegen.symbol` and labels the launch with the kernel's name
  (`Rig_program.Launch`'s `label`, `Rig.Submission.launch ~label`).
- **Differs:** the text compiled is the program's source, read as C's and
  C++'s tokens (comments, string, character and raw string literals,
  numbers with digit separators), without its comments and with the
  kernel's name and `Codegen.symbol` traded wherever either is an
  identifier, so every binary names its kernel `Codegen.symbol`. A source
  holding a line splice or `??`, which no renderer writes, keeps its
  comments and shares no binary. Each parameter's name and `tolk_argN`, `N`
  its slot (`data0_4096` and `tolk_arg0`), trade places the same way,
  unless the kernel's name is one of these names, when the parameters keep
  theirs. Two
  kernels whose sources differ in those alone, as an operation's kernels
  over two extents of a split host loop do (`E_4096_4`
  and `E_12288_4` on the host), or GPU kernels whose opts agree over two
  extents (the launch sizes in comments) or over buffers of two sizes (the
  parameters' names), compile to one binary, kept once in the compiler's
  table, and a link loads it once; each launch passes its own sizes and
  bounds, as before. The program keeps the renderer's source, with the
  kernel's and parameters' names and the comments, and profiles name a
  launch's work after its kernel.
- **Reason:** (d), to re-measure on a quiet fudge: tessera's distillation
  step (nano, 64 pixels, `--device cpu-jit`, cold) makes 222 programs whose
  binaries fall from 217 to 160, its process 24.6 s to 22.5 s, its loss
  unchanged. The parameters' trade reaches GPU kernels alone, a host
  kernel taking its arguments as bytes, and is unmeasured.
- **Pinned by:** the `Codegen` suite's `kernels of one source take one
  binary` (every test); the bench row is owed.

## D486. A workgroup's first thread index runs along its innermost local axis

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`'
  local splits: outside its QCOM branch, the ranked splits applied in
  ascending axis order, `sorted(to_local[:3])`); `codegen/opt/postrange.py`
  (`shift_to`, a new range numbered after every existing one);
  `codegen/gpudims.py` `group_gpudims` (local ranges sorted by number, the
  first `lidx0`).
- **tolk:** `Heuristic.locals`.
- **Differs:** the splits that `locals`
  ranks are applied from the innermost axis out, so the split of the latest
  axis is numbered first and becomes `lidx0`, the first hardware thread
  index, as tinygrad's QCOM branch already does. Which axes take threads,
  and how many, is tinygrad's. A kernel's innermost axis is its output's
  unit stride, and a SIMD group holds consecutive values of the first
  thread index (CUDA C++ Programming Guide §2.3.2 and §3.2.2.2; HIP's
  "Hierarchical thread model": that x is the fastest index is read, which
  threads a wavefront holds is unread; Metal's order unread), so its lanes store and load consecutive elements,
  where in ascending order a two-axis map gave `lidx0` to the rows:
  `Nx.rms_norm`'s scaled map over 512 rows of 2880 bfloat16
  (`E_16_60_8_16_3_4`) spread a warp's 2-byte loads over 8 rows. Each
  split finds its axis by its range, since an earlier split may delete an
  axis or place a local before a weak one, where tinygrad corrects a planned
  index for deleted axes alone (`deleted_shape`). Kernels that group, stream
  or take a tensor core keep their threads (D357, D89); a contraction
  takes the order as a map does, blocked or not: blocked in workgroups of
  128 threads (D212), rows on `lidx0` ran 2048 cubed and 4096 by 512 by
  4096 1.7 to 1.8 times as long.
- **Reason:** (d): gpt-oss-20b's norms and any map over rows of a wide
  output on CUDA. On fudge's RTX 5000 Ada, with bits equal, rune's
  `Cuda/rms-norm-bfloat16-512x2880` takes 28.6 to 18.7 us.
- **Pinned by:** the Codegen suite's `consecutive lid.x threads of a map
  over two local axes store one run after another` (every recorded GPU map
  whose workgroup spans two axes, on Metal, CUDA and HIP); the heuristic and
  stage goldens of tinygrad's GPU renderers whose kernels take two or more
  local splits, as `.tolk` files headed `D486` or, blocked, `D212`; rune's `Cuda/rms-norm-bfloat16-512x2880` and
  `Cuda/attention/jit-grad-bf16-2048x40x1024-full` rows and kaun's
  `GptOss/prefill-512` row on CUDA, their baselines pending the scheduled
  re-record. AMD and Metal are untimed.

## D487. A group takes a SIMD group's threads along a long reduce

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (a
  group of 16).
- **tolk:** `Heuristic.group`.
- **Differs:** a group takes 16 threads along its reduce, doubled up to
  `Renderer.t.simd_threads` while the mean length of the kernel's reduces,
  each over the axes it closes, exceeds `2t²`; beside an output axis split
  across it keeps 16. Of each class of axes, unit-stride first (D357), the
  widest width that splits or pads one (D467) is taken. tinygrad's group is
  16 on every target.
- **Reason:** (d): a thread adds `n/t` terms and then `t` partials, and
  within a SIMD group a wider group costs no issue slot, so `2t` pays past
  `n = 2t²`. BEAM reaches the width's loads and cost in two local splits,
  `(16, LOCAL)` then `(2, LOCAL)` or `(4, LOCAL)`, but BEAM is not tolk's
  default, as it is not tinygrad's. On fudge's RTX 5000 Ada, rune's
  `Cuda/top-k-64-of-1x4096` takes 320-321 us to 218.
- **Pinned by:** the heuristic goldens `sum`, `sum_prime`,
  `sum_strided_prime`, `sum_rows_eight`, `sum_rows_long`, `max_vocab`,
  `vecmat_1000` and `vecmat_*_mv_0` (CUDA, AMD, Metal), and
  `sum_cols_prime`, which keeps 16 beside an across split; stages'
  `sum_all_{cuda,hip,metal}`; the Heuristic suite's `a lone group takes the
  widest power of two threads from 16 up to its SIMD group whose square is
  less than twice its reduces' mean length`; rune's
  `Cuda/top-k-64-of-1x4096` and
  `Cuda/scatter-16384-rows-into-donated-bf16-32768x512`. AMD and Metal are
  unrun.

## D488. A local split or pad keeps a workgroup's threads and a sum's partials

- **tinygrad:** `codegen/opt/postrange.py` `Scheduler.apply_opt` (a split
  checks shared memory only, a pad nothing of a local axis);
  `codegen/opt/search.py` `get_kernel_actions` (`BEAM_LOCAL_MAX`).
- **tolk:** `Postrange.Scheduler.apply_opt`; `Search.get_kernel_actions`.
- **Differs:** a local split, a pad of a local axis or a tensor core is
  refused where the workgroup would pass `Renderer.t.local_threads_max`
  threads, and a local split or a pad of a local axis where a group would
  add more than `Chains.chain` threads' partials of a float sum, its
  siblings' included; a pad of a local axis checks the workgroup's shared
  memory as a split does. A hand-given opt raises at `Postrange.apply_opts`.
  tinygrad applies them all: its runtimes refuse a launch past the threads,
  and a group past 256 partials sums them in one running sum.
  `get_kernel_actions` bounds a candidate's threads by
  `Setting.beam_local_max` alone, as tinygrad does.
- **Reason:** (b) and (e): `Search.get_kernel_actions`, whose candidates
  must launch on the target (D484), so on Apple2 and Apple3 at most 512
  threads; and `Nx_kernel`'s error bound, which a running sum past 256
  terms breaks.
- **Pinned by:** postrange's `ten_sums_group`,
  `ten_shared_sums_group_unfit` and
  `local_whole_axis_beyond_host_integers` (`cases.tolk`, refused); the
  Postrange suite's `workgroups › apply_opt keeps a workgroup's threads
  within its target's and a group's float sum within 256 partials, a pad's
  included` on sm_89, gfx90a and Apple2, a tensor core beside 32 and 64
  local threads on sm_89 and gfx90a; the Search suite's `candidates that
  pad a local axis`.

## D489. A search lowers each candidate as the compile lowers its chosen kernel

- **tinygrad:** `codegen/opt/search.py` `_try_compile` (compiles
  `get_optimized_ast`'s kernel through `get_program`, whose
  `full_rewrite_to_sink` runs with `optimize=False`, which skips only the
  passes before `apply_opts` and `apply_opts` itself) and `_time_program`
  (`get_test_global_size` on every program's global size).
- **tolk:** `Search.try_compile` (`Codegen.linearize ~optimize:true`);
  `Codegen.blocked_sums`, where a kernel that is `optimized` skips the
  simplification before the opts and `Postrange.apply_opts`; `Search.test_size`.
- **Differs:** a candidate is lowered as the kernel the compile optimises is
  lowered after `Postrange.apply_opts`, so D262's staging, D348's panels and
  D80's blocks apply to it, and the binary timed for a kernel's
  optimisations is the binary that runs with them. A host program in blocks
  is timed over its whole loop: its blocks are no workgroups, and its launch
  runs every iteration whatever its first global size.
- **Reason:** (d), with D80, D262 and D348, whose passes change a program
  after the opts: a search that timed its candidates without them chose among
  other programs than those it ships. Searched unstaged, the stream kernel
  `key-projection-64` ran 32.7 us against the heuristic's 13.3 us. No bench
  row runs a search; the law below pins the rule.
- **Pinned by:** `Tolk.Search`'s `a search times the program each candidate
  compiles to and returns it` (CUDA's staged tensor cores, the host's
  blocks) and `measuring on fewer workgroups › launches a host program's
  blocks over its whole loop`; `searches.golden`'s host rows, which count a
  host program's workgroups without its blocks.

## D490. A padded sum of products multiplies its factors gated to zero

- **tinygrad:** `codegen/opt/postrange.py` (`PADTO`: a reduce over the padded
  axis sums `valid.where(src, identity)`), `Scheduler._apply_tc_opt` (a core
  whose reduce's source is such a `WHERE` gates each of its operands by it).
- **tolk:** `Postrange.Scheduler.apply_opt`'s `pad_reduce`, `use_wmma`.
- **Differs:** a sum whose source is a product, or a cast of one, padded
  along its reduce axis, multiplies its factors each gated to `0` by the
  padded range's validity, below the lossless widenings it reads them
  through (`Ops.widens`), where tinygrad selects the product's identity:
  `where v a 0 * where v b 0` for `where v (a * b) 0`. Any other reduce's
  padded steps add its identity, as tinygrad's do. A padded step adds
  `+0 * +0`, which is `+0` whatever the factors' producers give at the
  padding (`log 0`, a reciprocal's infinity, a bias), so the sum keeps its
  multiply-add, rounded once (D25), where a selected product rounds alone and
  adds; the sum then computes the bits of the same sum over its factors with
  zeros appended. A tensor core reads the gated factors as its operands, so
  `use_wmma` no longer gates them itself: one gate serves both paths, and the
  staged copy folds it into its zero fill as before (D262).
- **Reason:** (b): D25's sum of products, each product added into its
  running sum rounded once, holds through a pad only with the gate on the
  factors, and the tensor cores D491 pads by default would otherwise sum
  their padded and unpadded twins of one kernel (a padded K, and its
  operands with zeros appended) to different bits wherever a padded kernel
  falls to the SIMT path. tinygrad's form computes a value within
  `Nx.contract`'s bound, rounding each padded sum's products twice.
- **Pinned by:** the Codegen suite's `multiply-adds › a sum of products
  padded computes the bits of its factors with zeros appended` (a
  reciprocal's product padded from 37 terms to 48 on the host, against the
  same sum over 48 with zeros appended), the Postrange suite's
  `apply_opts optimises a kernel as recorded › padto_matmul_2`,
  `padto_matmul_4`, `padto_upcasted_5`, `padto_group_full_unroll_sum` and
  `tc_{metal,cuda,amd}_shape_padded`.

## D491. A tensor core pads its extents by default, within a bound on the work

- **tinygrad:** `helpers.py` (`TC_OPT`, default `0`), `codegen/opt/heuristic.py`
  (`hand_coded_optimizations`: the core at `TC_OPT`'s level, `2` only in a
  search), `codegen/opt/postrange.py` (`_apply_tc_opt`: an axis the core's
  dims do not divide is padded only at level 2).
- **tolk:** `Setting.tc_opt`, `Heuristic.tensor_cores`, `Heuristic.core_waste`,
  `Heuristic.core_takes`, `Heuristic.core_fits`, `Heuristic.core_rows`,
  `Heuristic.staged_tile`, `Stage.tile`.
- **Differs:** `TC_OPT` defaults to `2`. The heuristic tries every
  candidate axis's core at level 0 before any at the setting's level, and
  records the level that applied, so a kernel some axis's core divides takes
  the core tinygrad's default takes, on the same axis. A padded core is taken
  only where its padded extents hold at most 6/5 of the work of the kernel's
  own (`core_waste`), and `core_fits` and `core_rows`, which rune asks for
  the form of a contraction, read the same bound, leaving out the pad that
  D230 leaves out of a decoded factor's, in place of divisibility
  (at `TC_OPT` 0 or 1 the bound is 1, divisibility). An extent the kernel
  holds as a range of one trip is no axis, and a core that would pad it is
  refused, where tinygrad's lookup of the axis raises. At level 2, a staged
  core pads N to a multiple of its whole warp tile's upcast where neither
  that upcast nor its half divides N (`Stage.tile`, D262), on a padded or an
  unpadded core, beside the bound and within PADTO's 4x; a pad on no axis
  of the kernel leaves the core unstaged. At `TC_OPT=0` no core and no
  staged tile pads, so the kernels are those of tinygrad's default and the
  staging of D262. Padding computes the padded steps' products of zeros
  (D490): values stay within `Nx.contract`'s bound, and a kernel's bits
  move only where it pads.
- **Reason:** (d). On fudge's RTX 5000 Ada, against main on PROJ-2's staged
  plans, three interleaved rounds, each case on an empty GPU, minimum over
  rounds of the minimum of 10 calls: tessera's distillation step, bfloat16
  products from float32 parameters over 68608 rows a scan trip, ms, the 440
  by 440 weight gradient 25.57 to 2.26, the 1792 by 440 one 16.71 to 3.65,
  the 440-wide input gradient 3.29 to 1.07, the feed-forward products 10.68
  to 4.61 and 4.83 to 3.04. gpt-oss-20b's projections (L2 evicted, sums of
  kernel medians over 8 calls, us): q at 127 tokens 2070 to 56, at 100 1145
  to 115, at 68 480 to 103, at 40, 56, 72, 88, 104 and 120 tokens 0.27-0.58
  of main, k at 72 and 120 41.5 to 15.6 and 49.7 to 16.0, the attention
  block at 127 tokens 8236 to 290; at 128 tokens, which no core pads, the
  same kernels and bits. The default's backward of attention at a head of
  129, bfloat16, 0.61-0.89 of main's time. The bound is chosen, not
  fitted: products of 65536 rows by K by 512, bound by their float32
  output, took 0.32 and 0.88 of main's time at K = 440 and 136 and
  0.97-0.98 at K = 72 and 40 (1.11 and 1.2 times the work), while q at 100
  and 68 tokens, padded 1.12 and 1.18 times, gain 10x and 4.7x and a head
  of 129 (1.12) 1.1-1.6x; a padded core can lose by its
  own shape, K = 72 padded with an exp epilogue 1.23-1.29 times main's SIMT
  kernel, as main's core runs at K = 80 (TODO). The staged pad of N:
  products of 65536 rows by 512 by N took 0.28-0.83 of main's at N = 440,
  136, 72, 24 and 20 (tiles of 448, 192, 128 and 64 columns), and 0.99 at
  40; the 440 by 440 weight gradient, a split sum, takes about 2.27 ms
  staged against 1.78 with staging off (1.28x), both about 11 times
  faster than main. AMD and Metal take the default unmeasured.
- **Pinned by:** the Heuristic suite's `streaming › a core pads what its tile
  does not divide, within a bound`, `core_fits names the contractions a core
  takes`, `core_fits reads the core settings`, `core_rows names the fewest
  rows a core takes`, the cases `heads_projection_metal`,
  `heads_projection_biased_metal` and
  `matmul_half_ragged_{metal,cuda,amd}_tc_opt_2` (refused past the bound),
  the Heuristic suite's `a core of one tile of N is optimised without a pad
  of N` and `a product's workgroups hold at most a block's threads on every
  GPU` (a row of one trip), the Codegen suite's `conv_big_metal`, the
  Setting suite's defaults, the Stage suite's `sources › a float16 product
  whose core pads M and K`, its value law (`an axis the core pads`, `N
  padded to the staged tile`) and its barriers law's HIP case of N one
  tile, and rune's `Jit_gpu › products › a sum a tensor core pads computes
  the bits of its operands with zeros appended` and `› a product whose N is
  one tile of the core gives eager's bits`.

## D492. A kernel declares the type of every parameter it names

- **tinygrad:** `renderer/cstyle.py` (`uops_to_dtypes`, which lists the
  types of a kernel's values and leaves out its buffers' pointers, and
  `CUDARenderer.render_kernel` and `HIPRenderer.render_kernel`, which
  include `cuda_fp16.h`, `cuda_bf16.h` and `cuda_fp8.h`, or define `half`
  and the 8-bit floats, from that list).
- **tolk:** `Cstyle.uops_to_dtypes`, which `cuda_kernel` and `hip_kernel`
  read.
- **Differs:** the types a CUDA or HIP kernel declares are its values' and
  each parameter's and buffer's element type, as one lane, whether or not
  the kernel reads or writes it. A store whose index no thread reaches is
  dropped and its parameter stays in the signature, where tinygrad then
  writes a type it never declared. A kernel whose parameters' types are
  among its values' writes tinygrad's source.
- **Reason:** (b). rune's compiled gradient of an `Nx.attention` over values
  of no column on CUDA runs `attention_combine` with its `o` of no column:
  its store to `o` is dropped and its `nv_bfloat16*` parameter stays, which
  NVRTC rejects (`attribute "__global__" does not apply here`) without
  `cuda_bf16.h`.
- **Pinned by:** the Cstyle suite's `parameters' types › a parameter the
  kernel never reads or writes has its type declared` (CUDA's 16-bit and
  8-bit floats, HIP's `half`) and `› a kernel with a parameter it never
  reads or writes compiles` (NVRTC, where the machine has it).

## D493. A partial form's trip takes its block whole

- **tinygrad:** `codegen/opt/heuristic.py` (`hand_coded_optimizations`: a
  core only for one reduce axis at `TC_OPT` 0; on the path without a core,
  a last reduce of at most 32 values unrolled whole and the loop around it
  kept).
- **tolk:** `Heuristic.tensor_cores` (`partial_trip`, `unroll_whole`).
- **Rule:** an unstaged core whose operand is decoded and whose sum runs
  another reduce axis beside its `K`, as a Q8_0 matrix's blocks, unrolls
  `K` whole where it holds at most 32 values, as a last unrollable axis.
- **Reason:** (d). Each trip of the blocks waits on its own loads: on
  fudge's RTX 5000 Ada, gpt-oss-20b's Q8_0 prefill takes 18.5 ms a step at
  64 tokens where it took 20.3, and 46.3 at 512 where it took 49.8.
- **Pinned by:** the Heuristic suite's `streaming › a partial form's trip
  takes its block whole` (sm_89, sm_80, gfx1100, Apple9), the cases
  `q8_0_partial_{cuda,amd,metal}` (`q8_0_partial_*.tolk`, `cases.tolk`),
  the tensor core floor's stated splits, tolk's bench row
  `stream/q8_0-core-64` and kaun's gpt-oss row `prefill 512, Q8_0`.

## D494. A 64-bit integer cast to float32 on the host rounds once

- **tinygrad:** `renderer/cstyle.py` `ClangRenderer.extra_matcher`, which
  leaves the cast to C: `(float)x`.
- **tolk:** `Cstyle.clang_extra_matcher`, through `Decomp_dtype.narrow` at
  `Float64`.
- **Differs:** a cast of an `Int64` or `Uint64` whose bounds pass `2^53` to
  `Float32` converts to a double rounded to odd, then to float32. Within
  `2^53` the cast stays C's.
- **Reason:** (e). LLVM IR's `sitofp` and `uitofp` round once: "If the
  value cannot be exactly represented, it is rounded using the default
  rounding mode" (LLVM Language Reference, `sitofp .. to` and `uitofp ..
  to` Instructions, Semantics), and the default environment assumes "the
  round-to-nearest rounding mode" (Floating-Point Environment). LLVM's
  vectorizers pack the scalar casts of an upcast kernel, and the AArch64
  backends of Apple clang 17 and of LLVM 18 and 19 miscompile the packed
  conversion: doubles rounded to nearest, then narrowed, two roundings.
  `2^53 + 2^29 + 1` becomes `2^53`, where one rounding gives `2^53 + 2^30`.
  LLVM 21 to 23 convert each lane once. The rewrite holds whatever the
  compiler, since whether a loop is packed is the compiler's choice, and
  the kernel's source does not depend on which Clang compiles it. Within
  `2^53`, as every 32-bit integer is, the double is exact and even the
  miscompiled lowering rounds once, so those casts stay C's. Cost: about
  twenty operations a lane in place of one, unmeasured; rune's bench row
  `Jit › cast-int64-float32-1M` times it.
- **Pinned by:** `Tolk.Codegen › a 64-bit integer rounds once to float32 on
  the host`, in 1, 2, 4 and 8 lanes; rune's `Lower_arith › casts of bounded
  64-bit integers give eager's values` and `rune jit › jit › an integer
  cast to a float gives eager's bits`, under Apple clang 17.

## D495. A GPU's upcasts count the loads a reduce repeats

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (the
  upcast loop takes an axis where some buffer of the kernel, `k.bufs`,
  every load and store, has stride 0 on it; the same at `6b7b018da`).
- **tolk:** `Heuristic.upcast_more`.
- **Differs:** on a renderer with local indices, in a kernel that reduces,
  only a store or a load that reads a reduce range offers an axis it does
  not read. A load read once an output, as a bias or an additive mask
  after a product, offers none: the batches or heads it does not read,
  which both operands of the product read, are not upcast for it.
- **Reason:** (d): upcasting an axis that a load does not read shares the
  load among the axis's lanes. A load whose index reads a reduce range is
  repeated at every trip, so sharing it saves a load each trip; one whose
  index reads none, as a bias after the product or a value the reduce
  holds fixed, is loaded once an output, so sharing it saves almost
  nothing. Counting it took a biased product's batches by 4, which left
  D212 nothing to block. A store stays counted: one whose index is a bare
  range offers it, as tinygrad's count does, which keeps every kernel
  without such a load as it was. On fudge's RTX 5000 Ada, the tolk bench's
  `contract/scores-masked-f32` (attention's float32 scores of 32 heads at
  1024 positions over 64, plus an additive mask) takes 306 us against 535
  under tinygrad's count. The host keeps tinygrad's count: its gain is
  unmeasured.
- **Pinned by:** `Tolk.Heuristic`'s `a contraction in registers › a bias
  the batches do not read leaves a batched product's optimisations as
  without it` (`two_batches_768` and `batched_768` on CUDA, AMD and sm_89)
  and `on a GPU, both upcast axes reused, a thread holds 8 by 8 outputs in
  a workgroup of 128 threads` (`batched_768_biased`, `scores_biased` and
  `scores_biased_first` on CUDA and AMD), with the layouts in the suite's
  `.tolk` files headed D495; the tolk bench's `contract/scores-masked-f32`
  and `contract/batched-biased-f32` rows on fudge, their baselines pending
  the scheduled re-record. AMD is untimed.

## D496. A where rebuilt on an Invalid branch keeps its committed type

- **tinygrad:** `uop/ops.py` `RewriteContext.unified_rewrite` (a node whose
  sources were rewritten is rebuilt as `UOp(op, new_src, arg)`, its type
  derived again), with `uop/symbolic.py` `pm_data_invalid` (a where of an
  Invalid condition is Invalid) and `schedule/indexing.py`
  `convert_pad_to_where_to_keep_behavior_local` (a pad's zero is a weak
  constant).
- **tolk:** `Ops.graph_rewrite`, through `kept_where_type`.
- **Differs:** a where whose type is committed and whose rebuild on its
  rewritten sources derives a weak type commits its weak constant branches
  at the type it had. tinygrad rebuilds it weak: a pad's weak zero beside a
  branch that folded to Invalid, which states no width.
- **Reason:** (e). A value read on the outer padding of a value padded twice
  is `where(False, where(Invalid, x, 0), 0)`; once the inner where is
  Invalid the outer one, and the zero it folds to, are weak. A bitcast of it
  then reads no width: at the pin,
  `Tensor([1.0], dtype=float64).pad((0, 1)).pad((1, 0))[0:1].bitcast(int64)`
  commits the zero at float32 and clang refuses the kernel (`source size
  does not equal destination size (4 vs 8)`), where tolk raised in
  `Prepare.expand_bitcast` (`no unsigned integer of 100 bytes`). Committing
  the pad's zero where the pad becomes a where instead fails 20 of the
  schedule suites' recorded tests.
- **Pinned by:** the Ops suite's `graph_rewrite › a where rebuilt on an
  Invalid branch keeps its committed type` (both passes, both orders) and
  the engine suite's `link and run › a bitcast of a float64's outer padding
  reads its bits` (padding of zero and of one).

## D497. A product in registers whose operands lie along its sum is staged through shared memory

- **tinygrad:** none: no pass stages an operand (D262), and
  `codegen/opt/heuristic.py` `hand_coded_optimizations` upcasts a
  contraction's output axes by 3 or 4 and unrolls its reduce, each thread
  reading its operands from global memory. Its hand-written kernels stage
  through local memory by a scheduler `STAGE` (`extra/gemm/amd_uop_matmul.py`
  `hand_spec_kernel3`: 128 by 128 by 8, 4 by 4 outputs twice a thread, `A`
  transposed on its store into local memory).
- **tolk:** `Heuristic.staged_product`, `Heuristic.general`;
  `Stage.product_tile`, `Stage.stages_product`, `Stage.products`,
  `Stage.staged_products`, `Stage.deepen`, `Stage.nest` and
  `Stage.trip_steps`, beside D262's `Stage.operands`;
  `Coalesce.memory_coalescing` (an index ordered after effects);
  `Cstyle.async_bytes` and `Cstyle.cp_async`.
- **Differs:**
  - **The tile.** On a renderer that copies asynchronously, two stages
    fitting, a contraction that D212 blocks, blocked or short of the units,
    whose operands both read its unrolled reduce at unit stride, as `x · wᵀ`
    with `w` stored `[outputs; inputs]` and every input gradient `dY · Wᵀ`,
    takes 16 threads along M and 8 along N, the threads splitting each axis
    first so that adjacent threads hold adjacent rows, 8 by 8 outputs a
    thread (4 by 4 short of the units), each axis padded to the tile, and its
    reduce unrolled by a copy of 16 bytes, 4 float32 values, or where 16
    bytes do not divide the sum and it holds at least 16 copies of them, of
    8 bytes, where D212 unrolls by 4 and splits the reduce into threads.
  - **Staging.** A sum of products in registers (a `Mul` a `Reduce` of `Add`
    sums) is staged as D262 stages a core's operands, where each staged
    tile's innermost dimension is the sum's lanes and every other dimension
    moves first with a thread: a quarter of a SIMD group reads 8 adjacent
    rows, 16 bytes along the sum, which `Stage.layout` places on 8 slots of
    a line. The loop of the sum takes trips of several steps
    (`Stage.trip_steps`): the deepest of at most 32 values, leaving two
    trips, keeping the running sums' blocks whole (D480) and two stages
    within `shared_max`; the steps are a loop inside the trips, part of the
    tile, so each output adds the same terms in the same order as the
    unstaged loop. A sum over a nest of loops, as a projection's heads and
    their dimensions, takes the steps in its innermost loop, and its two
    stages step through the nest. A copy of 8 bytes is PTX's
    `cp.async.ca`, the only form of that size (`.cg` takes 16). D212's
    layout, the C-order `[K; N]` operand and every kernel with other shared
    memory stay unstaged.
  - **Reads merge.** A staged tile's reads, ordered after the next trip's
    copies, merge into vector loads as unordered reads do: 16 bytes of
    float32 a step, and pairs of 16-bit values of a core's fragments (D262).
  - tinygrad's tile is the heuristic's; its hand-written kernel's store
    through registers transposes `A`, as CUTLASS's SIMT products of operands
    laid along K do (`TransposePitchLinearThreadMapSimt`, one element a
    load).
- **Reason:** (d): TESSERA's students, whose linear layers store their
  weights `[outputs; inputs]`, run their products at 15-17.5 TFLOP/s from
  the L1 where cuBLAS's SIMT kernels run 26-30 (perf_500, RTX 5000 Ada).
  Staged, TESSERA's products of 16384 rows run at 23.9-30.1 TFLOP/s against
  15.4-18.5 split, and perf_500's model time falls from 43.0 to 32.5 s for
  the medium student and from 75.6 to 54.1 s for the large, below
  PyTorch's 35.4 and 56.6. The tolk bench's `contract/qk-f32-2048` takes
  283 us against 336-341 and `stream/q8_0-64` 2.56 ms against 3.00.
  Copies of 8 bytes and a nest's trips stage the medium student's output
  projection, a sum over 4 heads of 110: perf_500's model time falls from
  29.6-30.1 to 28.4-28.5 s.
- **Pinned by:** `Tolk.Heuristic`'s `a product laid out along its reduce`
  (`is staged where the renderer copies asynchronously`, `is not staged
  laid out along its columns, or on sm_75`) and `a contraction laid out
  along its reduce` on sm_75; the Stage suite's `kernels the race checker
  runs › products in registers in trips of several steps …` (over 110 in
  copies of 8 bytes too) and `a product summed over heads of 110 in copies
  of 8 bytes …`, and their race checks, and the Stage suite's `sources`
  (`bf16_2048.cu`, `f16_padded.cu`,
  merged reads); the tolk bench's `contract/qk-f32-2048` and
  `stream/q8_0-64` rows and rune's
  `Cuda/matmul-float32-transposed-16384x440x1320`,
  `Cuda/matmul-float32-input-gradient-16384x1320x440` and
  `Cuda/matmul-float32-merged-heads-16384x4x110x440` on fudge.

## D498. A GPU's contraction pads an axis 4 does not divide, to upcast it

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (the
  upcast loop takes an axis only where 3 or 4 divides it, and stops when
  none is left; the same at `6b7b018da`).
- **tolk:** `Heuristic.padded_to_four`, `Heuristic.upcast_more`.
- **Differs:** on a renderer with local indices, a contraction summed in 4
  bytes or fewer whose upcast loop finds no axis that 3 or 4 divides takes
  one that 4 does not divide, where a load the loop counts does not read
  it, padded (PADTO) to a multiple of 4 within `core_waste` (6/5) of its
  size, where tinygrad's loop stops. D212 then blocks it.
- **Reason:** (d): the sums of attention's derivative over a head's keys
  or rows at 110 dimensions, 2048 of them over 8 heads, stopped at one
  upcast by 4 and ran 6.7 ms (the sum over the keys, its rows read 16
  bytes a row) and 1.07 ms; padded to 112 and blocked, 1.00 and 0.60 ms on
  fudge's RTX 5000 Ada. A float32 product of 16384 rows by 2048 into 110
  columns took 5.30 ms so, 4 outputs a thread, and takes 0.66 padded.
- **Pinned by:** `Tolk.Heuristic`'s `a contraction pads an axis 4 does not
  divide to upcast it`; rune's `Cuda/matmul-float32-16384x2048x110` on
  fudge.

## D499. An integer within 2^24 converts to bfloat16 through float32 on CUDA below sm_90

- **tinygrad:** `renderer/cstyle.py` `CUDARenderer.extra_matcher` (no rule
  for the cast) and `CStyleLanguage.render_cast`: `(nv_bfloat16)(x)`.
- **tolk:** `Cstyle.pm_int_to_bf16`, in `Cstyle.cuda_extra_matcher` below
  sm_90.
- **Differs:** below sm_90, a cast to `Bfloat16` of an integer or a bool
  that is not a constant, whose bounds (`Ops.vmin`, `Ops.vmax`) lie within
  `2^24`, is a cast to `Float32` then to `Bfloat16`:
  `(nv_bfloat16)((float)(x))`. A wider integer, and every integer from
  sm_90, stays CUDA's cast. Below sm_80 `Bfloat16` is emulated and no such
  cast reaches the renderer.
- **Reason:** (d). Below sm_90, `cuda_bf16.hpp`'s `__int2bfloat16_rn`, which
  `nv_bfloat16`'s constructor calls for an `int`, a `char` and a `bool`,
  converts three times, rounded up, down and towards zero, then compares
  and selects; sm_90 has `cvt.rn.bf16.s32`. An integer within `2^24` is
  exact in float32, so one conversion there rounds once, as CUDA's does. A
  Q8_0 product decodes its codes to bfloat16 for a tensor core: on sm_89 a
  kernel of int8 casts runs 4 conversions a thread where it ran 12, and no
  compare or select. On fudge's RTX 5000 Ada, gpt-oss-20b's Q8_0 query,
  key and value projection at 512 tokens took 160.8 us against 192.6, its
  output projection 136-141 against 183.3 (probe, cold); the bench row's
  baseline is pending. A wider integer would need its float32 rounded to
  odd (`Decomp_dtype.narrow`), twice the instructions of CUDA's own
  rounding to odd, so it is left.
- **Pinned by:** `Tolk.Cstyle › bfloat16 on CUDA` (through a float32 on
  sm_80 and sm_89, CUDA's cast for a wider integer and on sm_90, and each
  such integer rounded once on sm_89); rune's `Rune.jit on a GPU ›
  integers › every integer of 16 bits and the ties of wider ones cast to
  bfloat16 give eager's bits`; tolk's bench rows `stream/q8_0-core-64` and
  `stream/q8_0-512`, the latter's baseline pending the scheduled
  re-record, and kaun's gpt-oss row `prefill 512, Q8_0`.

## D500. A large reduction splits along a padded axis where no divisor divides it

- **tinygrad:** `schedule/prepare.py` (`split_reduceop`: a reduced axis's
  divisors from `min(256, 2**22 // prod(reduce.shape))` down to 8; none, no
  split; the same at `6b7b018da`).
- **tolk:** `Prepare.split_reduceop`.
- **Differs:** where none divides, the longest reduced axis of at least 8
  times the largest split is padded with the reduction's identity to a
  multiple of it and split so: at most an eighth more work.
- **Reason:** (d): whisper's vocabulary of 51866 is twice a prime, so each of
  a decode step's seven reductions over its logits ran in one workgroup of
  32 threads looping 1621 times, 111 to 244 us. Split, a minute of
  large-v3-turbo's decoding spends 8.8 ms in them against 205 (DEBUG=2,
  fudge's RTX 5000 Ada).
- **Pinned by:** the Prepare suite's `split_prime` (`.tolk`) and `a split pads
  with its identity an axis no divisor from 256 to 8 divides`; rune's
  `Jit_gpu`'s argmax, argmin, sum and product over 51866 and 65537, and its
  bench row `Cuda/log-softmax-float32-51866` on fudge (358 us to 27.5).

## D501. A map's innermost threads store 16 bytes together

- **tinygrad:** `codegen/opt/heuristic.py` `hand_coded_optimizations` (each
  ranked axis takes the first of `[32] * (axis == 0) + [16,8,4,3,2]` that
  divides it within 128 threads; the same at `6b7b018da`).
- **tolk:** `Heuristic.run_bytes`, `Heuristic.store_lanes` and
  `Heuristic.locals`.
- **Differs:** in a map on a GPU, where every store reads the innermost axis
  at the stride of a thread's run of consecutive elements (its upcast along
  the stores' unit stride, or one element), and that axis's candidate gives
  it fewer threads than the runs of the narrowest store's elements that
  span 16 bytes, the axis takes the least divisor of its size that reaches
  them, within the workgroup's 128 threads. Those threads are `lidx0`'s
  (D486), consecutive in a SIMD group. Elsewhere, where no divisor reaches
  them, and in a kernel that reduces, the candidate stands.
- **Reason:** (d): the concurrent accesses of a warp coalesce into as many
  transactions as the 32-byte sectors they touch (CUDA C++ Best Practices
  Guide §10.2.1). A head of 110 channels, TESSERA's, takes 2 threads under
  tinygrad's candidates, so a bfloat16 transpose `[r; h; t; d] → [r; t; h;
  d]` spread each warp's accesses over 16 runs of 4 bytes: on fudge's RTX
  5000 Ada 391 us a launch (235 GB/s), runs of 8 bytes 221 us (416 GB/s),
  and from 10 bytes on, 10 threads included, 177 to 184 us (499 to
  519 GB/s), a contiguous add's rate. AMD and Metal are untimed.
- **Pinned by:** the Codegen suite's `a head transpose's lid.x threads store
  16 bytes together` and its `phi_arange_255` and `phi_arange_float` stage
  goldens of CUDA, HIP and Metal; the heuristic suite's `add_odd` goldens of
  Metal, CUDA and AMD and its table (`.tolk`, D501); tessera's
  `distil-step/medium` row on CUDA.
