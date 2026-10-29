# Divergences from tinygrad

tolk follows tinygrad `79af1ca70e7021f504919c4ff5631245acc33ed6`. This
ledger lists every place where it does something else. An entry is admitted
for one of three reasons only:

- **(a) an OCaml constraint:** acyclic modules, static types, the GC, domains;
- **(b) a named consumer:** a raven call site that fails without it;
- **(c) the executor's contract:** rig's devices, submissions and programs,
  which `tolk.engine` drives with the compiler's output.

Taste, speed without a measurement, and "the old tolk did it" are not reasons.
A difference in numerics also needs a failing rune test, since nx semantics
belong in rune's lowering.

Each entry gives the tinygrad file and line, the tolk file and line, what
differs, its reason, and the test that pins it. An entry whose test does not
exist yet names the layer that brings it; the entry is rejected at that
layer's review if the test is still missing. An entry goes when its reason
goes. Entries are in the order of their numbers, and a number is never
reused, since entries and documents cite them. Keeping only part of a file is
scope, recorded under Exclusions in `README.md`, not a divergence.

## D1. Withdrawn

A device's timeline was its signal word alone, whose values a batch's host
program took as parameters. rig numbers each device's work and orders it by the
memory it uses, and tolk encodes no queue (D146).

## D2. Withdrawn

tinygrad matches patterns without generating code when `UPAT_COMPILE` is 0
(`uop/ops.py:1545`, `upat_interpret`), and `UPat`, `PatternMatcher` and
`graph_rewrite` live in `uop/ops.py`. tolk ports that path in
`Ops`, so leaving out `uop/upat.py`, the pattern compiler, is scope: see
the Exclusions of `README.md`.

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk:** `lib/device.ml` (the compiler half of `device.py`);
  `engine/tolk_engine.ml` (`target`, `link`); `lib/uop/ops.ml:229` (the type
  `param_arg`), `:3225` (`param_arg`), `:3253` (`new_buffer`).
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
  (`uop/ops.py:856-993`) are the engine's. Whether storage is bound is the
  operation itself, `BUFFER` or `ALLOC`, so the specification's checks of
  `ParamArg.buffer` (`uop/spec.py:149,153,266,268`) hold by construction.
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
  - `uop/ops.py:15,204,270,517,935,1167-1190,1572` calls `symbolic`, `render`,
    `spec`, `upat`, `schedule.prepare`, `mixin.rand` and `renderer`;
  - `uop/spec.py:7,287-289` against `device`, `codegen.opt`,
    `schedule.rangeify` and `renderer`;
  - `renderer/__init__.py` imports `device.Compiler`, while `device.Compiled`
    holds renderers;
  - `codegen/opt/postrange.py:267,273` against `search` and `heuristic`, with
    `search` importing `engine.realize`;
  - `schedule/__init__.py` against `engine.realize`, and `tensor.py` against
    `engine.jit` and `engine.realize`;
  - `renderer/cstyle.py` imports the compilers and `ops_metal`.
- **tolk:** for `uop/ops.py`, `lib/uop/shape.ml:87` (`simplify_rules`) and
  `:2861` (which sets them); `lib/uop/ops.ml:1190` (`Make_elementwise`), `:816`
  (`repr`), `:244` (`bufferize_opts`), `:262` (`Calls`);
  `lib/uop/render.ml:202` (`render`), `:212` (`srender`);
  `lib/renderer/renderer.ml:119` (`Compiler`); `lib/schedule/prepare.ml:709`
  (`contiguous_view`); `lib/schedule/schedule.ml:164` (`pm_flatten_linear`);
  `lib/codegen/codegen.ml:621` (`apply_opts`); `lib/engine/realize.ml:117`
  (`lower_and_compile`); and `lib/runtime/support/compiler_metal.ml`
  (`Compiler_metal`).
- **Differs:**
  - the `UOp` methods that call a later module become functions of that
    module: `contiguous_view` and its matcher go to `Prepare`,
    `to_elf` to `Device`, and `render` and `srender` to `Render`;
  - `engine/realize.py`'s `pm_flatten_linear`, which `schedule/__init__.py`
    imports, is `Schedule.pm_flatten_linear`, since `Realize` follows
    `Schedule`;
  - `simplify` is `Shape`'s, with the functions that call it (reshaping,
    `resolve`, shapes) and the rules it rewrites with, which read shapes:
    `symbolic.py`'s `symbolic_simple`, `commutative` and `symbolic`,
    `divandmod.py`'s `div_and_mod_symbolic`, `movement.py`'s `mop_cleanup`
    and `weak.py`'s `pm_uncast_const`. `simplify` precedes the rules and reads
    them through one reference that the module's last line sets, so a
    program that links `Shape` has them. `Symbolic` keeps `sym` and the rules
    on conditions, which call `simplify`;
  - `CallInfo.aux`, `hcq2.py`'s `HCQInfo`, is the record `hcq_info` of
    `Ops`, since call arguments hold it, though no call of tolk's carries
    one (D146);
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
  pattern operations`; `Tolk.Ops › queue calls › pp_hcq_info formats
  every field, as the record's repr`; for `Renderer`: `Tolk.Renderer ›
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
  `helpers.py:169-186` (`Context` and `ContextVar`, one value per process);
  `codegen/__init__.py:495-505` (`to_program_context`, the settings a worker
  process is started with, and `to_program_cache`, which the parent fills).
- **tolk:** `lib/engine/worker.ml:10` (`spawned`) and `:21` (`map`);
  `lib/setting.ml:151` (`t`) and `:201` (`context`);
  `lib/codegen/codegen.ml:1008` (`to_program`'s cache);
  `lib/runtime/support/compiler_metal.ml:38` (`build`);
  `lib/runtime/support/compiler_worker.ml` (`run`), `compiler_worker.h`,
  `compiler_amd_worker.c` and `compiler_cuda_worker.c`.
- **Differs:** compilation runs on domains, not processes. `Worker.map`
  spawns its domains for the call and joins them before it returns, where
  tinygrad keeps a pool: an idle domain still takes part in every minor
  collection. So the process machinery goes with the pool: the hidden
  `__main__`, the copied environment, the ignored SIGINT, the recycling of
  a worker after 16 tasks, and `terminate_worker_pool`. The workers' context
  (`ALLOW_DEVICE_USAGE`, `VIZ`, `TRACK_MATCH_STATS`) goes too: the compiler
  opens no device and has neither of the other settings. A setting holds one
  value per domain, where tinygrad holds one per process: a domain starts with
  the values of the domain that spawns it, and a `context` override is seen by
  its own domain only. A tinygrad worker process has its own settings, so a
  `Context` entered while it compiles (the construction check's `CHECK_OOB=0`)
  leaves the others alone; process-wide settings shared by compiling domains
  would let one domain's override and restore clobber another's. A spawned
  domain starts with its spawner's settings, so `to_program_context` goes
  too. `to_program` keeps its programs in one table that every domain reads
  and fills under a lock, and a domain that asks for a program another is
  making waits for it, so each program is compiled once, as tinygrad's
  parent compiles each key once. MTLCompiler, which a tinygrad worker loads
  in its own process, is loaded once for every domain, and it re-parses its
  options into LLVM's global option registry on every build, which is not
  thread-safe: its builds run one at a time. comgr serialises the compiles
  of a process on a mutex of its own, and NVRTC 13.3 and 13.4's assembler
  crashes on some valid sources, so comgr and NVRTC compile in processes
  of their own, spawned by the compiling domains, where a tinygrad worker
  process loads the library once and compiles in it: a small C program per
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
  `› a library that does not load › a compile in turn is served by an idle
  worker` and `› a compile after its idle workers are killed raises the same
  error`, and `› a process that cannot start`; `Tolk.Compiler_cuda (fork) ›
  a forked process` (every test).

## D6. Devices are named, never parsed

- **tinygrad:** `device.py:26,36,395,491` (device strings split at `:`).
- **tolk:** `lib/device.ml:121` (`renderers`), `:144` (`renderer`).
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

- **tinygrad:** `helpers.py:398-447` (an SQLite database of pickled values,
  keyed by strings, integers or dictionaries of columns).
- **tolk:** `lib/helpers.ml:406-691` (`Diskcache`).
- **Differs:** keys and values are strings, which callers encode. Each entry
  is a file under `CACHEDB`, a directory, written aside and renamed into
  place; tables are versioned by tolk's own version.
- **Reason:** (a). OCaml has no pickle, no type-safe serialization of
  arbitrary values: a cache returning any type would be `Marshal`, which
  crashes when a rebuilt program reads a value of a changed type. Without an
  SQLite transaction, writing aside and renaming is what keeps an entry whole
  when a writer crashes or races another.
- **Pinned by:** `Tolk.Helpers › Diskcache › get reads back any key and
  value put` and `Tolk.Helpers › Diskcache › behaves as a table of
  entries per table`, which run with `CACHEDB` set, as dune runs them.

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
- **tolk:** `lib/dtype.ml:470-491` (`encode_format`, one rounding for
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
    that the cast's rounding to the narrow float (D65) is the one rounding; an
    emulated 64-bit integer
    converts to a float32 once, where tinygrad's word arithmetic rounds each
    word and their sum;
  - emulation keeps subnormals, both ways;
  - a finite value becomes an infinity in a 16-bit float from the greatest
    finite value plus half an ulp, the tie rounding to even; an 8-bit float
    saturates to its greatest finite value;
  - an infinity stays one in e5m2 and becomes the NaN of e4m3 and the `fnuz`
    formats, of its sign where the format has one (D10);
  - a conversion quiets a signalling NaN; a copy or a selection, which
    converts nothing, keeps it (D62).
- **Reason:** (b). rune folds constants with `Dtype`, and nx converts eagerly
  with one rounding. An emulated kernel exists only because its target lacks
  the type, so it must give the bits a native one gives.
- **Pinned by:** `Tolk.Dtype › truncate › truncation.golden` (the
  near-tie rows, stated in code), `Tolk.Dtype › truncate › bfloat16 rounds
  once, to the nearest, ties to even` and `Tolk.Dtype › truncate › an
  integer rounds to a narrower float once, from its value`; for emulation, one
  test per facet in `Tolk.Decomp_dtype › D9`: `an emulated narrow float
  keeps its subnormals, both ways`, `a 16-bit float overflows from its greatest
  value plus half an ulp, the tie to infinity`, `an infinity stays one in e5m2
  and is the NaN of e4m3 and the fnuz formats, and a finite overflow
  saturates`, `a double, or an integer more precise than a float32, rounds
  once`, `an emulated 64-bit integer converts to a float32 once`, `an fnuz
  format stores an underflow to negative zero as positive zero`;
  `Tolk.Decomp_dtype › goldens`, which are tinygrad's graphs with
  `f2f` and `f2f_clamp` replaced by these conversions.

## D10. A float8 NaN keeps its sign when decoded

- **tinygrad:** `dtype.py:279` (`fp8_to_float` returns `math.nan` for
  e4m3's NaN codes, whatever their sign bit).
- **tolk:** `lib/dtype.ml:431-448` (`decode_format`).
- **Differs:** e4m3's `0xff` decodes to a negative NaN, so it encodes back to
  `0xff`, where tinygrad gives `0x7f`. e5m2 already kept the sign.
- **Reason:** (b). nx's decoder keeps the sign, so a value read eagerly and a
  folded constant agree, and every NaN code keeps its sign through a round
  trip.
- **Pinned by:** `Tolk.Dtype › storage › a NaN decodes with the sign of
  its bits`.

## D11. Kernel optimisations are typed

- **tinygrad:** `codegen/opt/__init__.py:9-15` (`Opt(op, axis, arg)`, with an
  `arg` of any type), checked by `codegen/opt/postrange.py:109-164`.
- **tolk:** `lib/codegen/opt/opt.ml` (`t`, `target`).
- **Differs:** each kind of optimisation is a constructor with its own
  fields, which stand for `OptOps` too, and a split's target is one of the
  three axis types a split can make (`split_targets`, `postrange.py:14`). The
  malformed arguments that `postrange.py` refuses at run time cannot be
  built.
- **Reason:** (a): static types.
- **Pinned by:** the type itself, whose printing `Tolk.Ops › arguments ›
  reprs.golden` checks against tinygrad's `Opt` repr (the row "kernel with
  opts"); `Tolk.Postrange` drops tinygrad's malformed-argument cases with
  this entry as the reason.

## D12. A node's key is a BLAKE2 digest

- **tinygrad:** `uop/ops.py:266-268` (`key`, the SHA-256 of
  `str((op, dtype, arg))` followed by the keys of the sources).
- **tolk:** `lib/uop/ops.ml:1063` (`key`).
- **Differs:** the same text is digested with BLAKE2b-256.
- **Reason:** (a). No key is ever compared with a key tinygrad computed: keys
  name compiled programs in caches that tolk alone writes, so the digest
  is free to differ, and tolk takes the one OCaml's standard library
  has, which has MD5 and BLAKE2 but not SHA-256.
- **Pinned by:** `Tolk.Ops › key › ignores tags` and `Tolk.Ops ›
  key › tells arguments apart`.

## D13. Folding reads and writes committed constants at their width

- **tinygrad:** `uop/symbolic.py:29` (`fold_const_alu`, whose result keeps
  the unwrapped value, `truncate_output=False`), `:154-155` (the collapse
  of committed const conversions) and `:279-280` (two stage ALU folding),
  which read a constant with `UOp.val`
  (`uop/ops.py:259-263`), unwrapped; `uop/ops.py:1104-1163` (`_min_max`),
  which bounds a weak operand of a committed operation by its unwrapped value;
  `uop/weak.py:82-87` (`uncast_const`), which leaves the literal bare.
  tinygrad's own `TestModularWraparound` expects the wrapped results and is
  marked `xfail_broken_const_wraparound`.
- **tolk:** `lib/uop/shape.ml:1853` (`fold_const_alu`), `:2219` and `:2647`
  (the two stage fold of a maximum);
  `lib/uop/ops.ml:1470` (`at_width`) and `:1492` (`operand_bounds`);
  `lib/uop/shape.ml:1351` (`uncast_const`).
- **Differs:** a committed integer constant holds its type's value. A fold
  reads a committed constant, a cast of a literal to a type of known width,
  wrapped to that width, and so does a cast of it that collapses; it reads a
  weak integer operand of an operation on a committed integer at that
  operation's width, as compiled code commits it; and it writes a committed
  integer result wrapped. The bounds read operands, and bound constants, the
  same way. So `uncast_const` only ever drops a cast whose literal fits. This
  reverses tinygrad's clause that a folded integer is kept mathematical, for
  emission to wrap: the emission never saw the mathematical value, since every
  later fold and bound read it first. tinygrad reads the unwrapped value, so a
  fold gives what no machine computes: `(uint32 0xFFFFFFFF + 1) >> 1` folds to
  `2147483648` where the machine gives `0`, `threefry2x32(5, 10)` folds to
  another key than the unfolded graph computes, the int64 cast of the int32
  constant `2^31` folds to `2^31` where the machine gives `-2^31`, and
  `x < uint8 300` compares against `300` where the machine compares against
  `44`. On uint8, `max(-3, a)` becomes `a` where the machine computes
  `max(253, a)`; on uint32, `max(1, b) // -2` folds to `-1`, which is
  `4294967295`, where the machine divides by `4294967294` and gives `0`; and
  `max(-max(b % 2, -c), b)` at `c = 2` becomes `b` where the machine gives
  `2`, since `-c` folded to `-254`. A sum, a product and a bitwise operation
  fold the same before or after wrapping; a maximum orders the wrapped values,
  so the two stage fold reads a maximum's weak constants at its width: on
  uint8, `max(max(w, 1), -3)` would become `max(w, 1)` where the machine
  computes `max(w, 253)`. A maximum by bounds that keeps a weak
  operand commits it to the maximum's type, so the operations that read it do
  not lose their width. Reading at the width also keeps
  `c0 + x < c1 → x < c1 - c0` from a wrong answer where the offset wraps: on
  uint8, `u + uint8 -1 < 255` would become `u < 0`; D24 keeps that rule to
  offsets that do not wrap.
- **Reason:** (b): rune's `Nx.Rng` (Threefry), jitted with constant keys,
  must draw the numbers eager nx draws, and RFC 0012's Law 1: every integer
  expression's compiled value is eager nx's modular one.
- **Pinned by:** `Tolk.Symbolic › symbolic_simple › constants › an
  operation reads committed constants at their width` (a fold, a cast of a
  committed constant, a uint8 remainder), `Tolk.Symbolic ›
  symbolic_simple › constants › a comparison reads a committed constant at its
  width` (the uncast), `Tolk.Symbolic › committed constants hold their
  type's value` (each case above, evaluated before and after), the law
  `Tolk.Symbolic › laws › sym keeps the value of an integer expression at
  a committed width, wrapping included`, and `Tolk.Symbolic › tinygrad ›
  tests.golden › TestModularWraparound.<test>` and
  `› TestThreefryConstFolding.test_threefry`, whose goldens, generated with
  the same change in `test/gen/tinygrad.patch`, hold the machine values;
  `Tolk.Ops › bounds › a typed integer constant outside its type is
  bounded by its wrapped value, a non-finite one by the type`; the
  offset's interaction with D24: `Tolk.Symbolic › integers wrap ›
  an offset crosses a comparison only where neither side wraps` (the
  committed case).

## D14. Folded comparisons treat NaN as IEEE does

- **tinygrad:** `uop/symbolic.py:29` (`fold_const_alu`), whose operands come
  from `UOp.val` (`uop/ops.py:259-263`) as `ConstFloat`s (`dtype.py:8-22`).
  `ConstFloat` makes NaN equal to NaN on purpose, so that NaN constants
  intern as one node, and `exec_alu` compares with it: `nan != nan` and
  `nan < nan` fold to `False`, where `exec_alu` on floats gives `True` and
  `False`.
- **tolk:** `lib/uop/shape.ml:1853` (`fold_const_alu`); constants are
  interned by `Dtype.equal_const`, and `exec_alu` compares floats.
- **Differs:** a folded comparison of NaN constants follows IEEE: `nan <> nan`
  is `true`.
- **Reason:** (b): `Nx.not_equal x x` on a NaN is `true` in eager nx, and
  rune's jitted graph, whose constants fold here, must agree.
- **Pinned by:** `Tolk.Symbolic › symbolic_simple › constants › NaN is
  unequal to itself when constants fold, as IEEE says`.

## D15. Compilers load their library at the first compile

- **tinygrad:** `runtime/support/compiler_cuda.py:62` (`NVRTCCompiler`
  calls `nvrtcVersion`), `runtime/support/compiler_amd.py:80` (`HIPCompiler`
  asserts comgr is loaded) and `runtime/ops_metal.py:39` (`MetalCompiler`
  creates its code generation service), each when the compiler is made, so
  making the renderer that holds it fails without the library.
- **tolk:** `lib/runtime/support/compiler_cuda.ml` (`nvrtc`),
  `lib/runtime/support/compiler_amd.ml` (`hip`),
  `lib/runtime/support/compiler_metal.ml` (`compiler`).
- **Differs:** making a compiler loads nothing. The library is loaded at the
  first compile, once per process (comgr: by each compile's own process;
  NVRTC: by each process that serves compiles, D5), and a compile without it
  raises `Compile_error` with the reason.
- **Reason:** (b): `Cstyle`'s CUDA, HIP and Metal renderers are made, and
  render, on machines without NVRTC, comgr or MTLCompiler: the source goldens
  of the `Cstyle` suite and rune's rendering of a kernel for inspection.
- **Pinned by:** `Tolk.Compiler_cuda › a library that does not load`,
  `Tolk.Compiler_amd › a library that does not load` and
  `Tolk.Compiler_metal › a library that does not load`, run where the library's
  variable names a file that is no library: making the compiler succeeds, each
  compile raises `Compile_error`, and a cached binary is served without a load.
  Where the library is absent, `› without NVRTC on the machine`,
  `› without comgr on the machine` and `› without MTLCompiler on the machine`
  pin the error that names the library and its variable.

## D16. CUDA keeps a float8 infinity special

- **tinygrad:** `renderer/cstyle.py:33,42` (a cast to `__nv_fp8_e4m3` or
  `__nv_fp8_e5m2` is the constructor, which converts with
  `__NV_SATFINITE`), `:25-26` (an infinite constant is cast the same way).
- **tolk:** `lib/renderer/cstyle.ml:965-978` (`fp8_infinity`,
  `cuda_fp8_guard`, `is_fp8_guarded`), `:1018-1036` (the two rules of
  `cuda_lang`) and `:1144` (the helper in the prefix).
- **Differs:** the saturating conversion turns ±inf into ±max. tolk
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
- **Pinned by:** `Tolk.Cstyle › float8 infinities on CUDA`, which
  checks where the guard is declared, the byte each infinity writes and the
  bits of infinite e5m2 constants; `Tolk.Cstyle › sources › by default ›`
  `cuda_dtype_float8_e4m3`, `cuda_dtype_float8_e5m2`, `cuda_inf_nan_float8_e4m3`
  and `cuda_inf_nan_float8_e5m2`, which compare with tinygrad's source once the
  guard is written back as tinygrad writes it; and `Tolk.Cstyle › every
  GPU kernel compiles with its target's toolchain › cuda_*` (slow, skipped
  without NVRTC).

## D17. Each operation on a narrow scalar is narrowed in the source

- **tinygrad:** `renderer/cstyle.py:245-250` (an ALU used once is inlined into
  its user, whatever its type) with `:66-67` (the operator text).
- **tolk:** `lib/renderer/cstyle.ml:86,565,771` (`promoted`), `:622-632`
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
- **Pinned by:** `Tolk.Cstyle › narrowing`, which checks for every
  kernel that the kernel whose tinygrad source is its source is the kernel
  with those casts, and `Tolk.Cstyle › sources`, which compares the 21
  sources it changes with tinygrad's for that kernel; `Tolk.Cstyle ›
  execution on the host › wraps each operation on unsigned chars` and
  `› rounds each operation on halves to a half`; and the slow
  `Tolk.Cstyle › execution on the host › a kernel over a narrow type wraps
  and rounds as the interpreter`, over each type's whole range.

## D18. Metal computes a bfloat trunc in float

- **tinygrad:** `renderer/cstyle.py:368-372` (`MetalRenderer.extra_matcher`
  computes `SQRT`, `EXP2`, `LOG2` and `SIN` of a bfloat16 in float32).
- **tolk:** `lib/renderer/cstyle.ml:887` (`metal_extra_matcher`).
- **Differs:** `TRUNC` of a bfloat16 is computed in float32 as well. Metal
  has no `trunc` of a `bfloat`: `trunc(x)` converts `x` to `float` and
  returns a `float`, which does not convert to a `bfloat` implicitly, so a
  kernel that stores or adds it to a bfloat does not compile. The graph
  handed to the renderer has a cast to float32 and back around each such
  `TRUNC`.
- **Reason:** (b). rune runs bfloat16 models on Metal (gpt-oss), where
  `Nx.trunc`, `floor`, `ceil` and `round` of a bfloat16 lower to `TRUNC`.
- **Pinned by:** `Tolk.Cstyle › bfloat16 truncation on Metal ›
  truncates a bfloat16 in float32` and `› leaves it to CUDA, which truncates a
  bfloat16 with htrunc`, and the slow `› compiles a kernel that truncates a
  bfloat16`; `Tolk.Cstyle › every GPU kernel compiles with its target's
  toolchain › metal_transcendental_bf16`, tinygrad's graph, is an expected
  failure.

## D19. A hierarchical allreduce to one device lands there

- **tinygrad:** `schedule/allreduce.py:28-34`, the hierarchical branch of
  `handle_allreduce`, which returns before it reads the target device
  (`:11`), so its result is on every device of the source even when the
  allreduce targets one. `create_allreduce_function` (`:68-75`) then stores
  that value into storage on the one target device. Repro: with
  `ALLREDUCE_NODE_NDEVS=2`,
  `UOp.new_buffer(("CPU:0","CPU:1","CPU:2","CPU:3"), 16, dtypes.float).allreduce(Ops.ADD, "CPU:0")`
  handled by `handle_allreduce` gives an `MSTACK` on the four devices.
- **tolk:** `lib/schedule/allreduce.ml:104` (`handle_allreduce`).
- **Differs:** when the target is one device, the hierarchical branch copies
  each reduced chunk there from the device of its rank in the first node, as
  the ring and all-to-all branches copy their reduced chunks, and the result
  is on that device.
- **Reason:** (b): rune's multi-device reductions. A sharded sum reduced to
  one device under `ALLREDUCE_NODE_NDEVS`, the multi-node setting, would
  otherwise compute a value on every device and store it into storage on
  one.
- **Pinned by:** `Tolk.Allreduce › handle_allreduce › recorded ›
  nodes_to_one_device_handled.golden, landed on its device`, which
  states tinygrad's golden in code with each gather of a chunk replaced by its
  copy to the target, and `Tolk.Allreduce › rules › a hierarchical
  allreduce to one device lands there`, which evaluates the value and
  the function on two devices and finds the reduction on the target alone.

## D20. Storage keeps an e5m2 NaN's payload

- **tinygrad:** `dtype.py:251,278` (`float_to_fp8` stores every e5m2 NaN as
  `0x7f` of its sign, and `fp8_to_float` decodes every NaN code as `math.nan`
  of its sign).
- **tolk:** `lib/dtype.ml:412-448` (`nan_of_payload`, `nan_payload` and
  `decode_format`) and `:470-491` (`encode_format`).
- **Differs:** storage moves an e5m2 NaN's two payload bits through the double,
  as it moves the payload of the 16-bit formats, so `bitcast` keeps every e5m2
  code: `0x7d` and `0x7e` come back as themselves, where tinygrad gives `0x7f`,
  and the canonical NaN stores as `0x7e`, its quiet code. A conversion still
  gives `0x7f` of its sign, as nx's encoder does.
- **Reason:** (b). A kernel's bitcast and nx's are byte reinterpretations, so
  a bitcast that rune folds must keep the bits as they do.
- **Pinned by:** `Tolk.Dtype › storage › a bitcast through a float
  gives back every 8- and 16-bit word`, on every e5m2 code, and the NaN rows
  of `Tolk.Dtype › storage › reencode.golden` and `Tolk.Dtype ›
  truncate › truncation.golden`, stated in code.

## D21. A reshape of a value sharded on two axes divides each by its own count

- **tinygrad:** `schedule/multi.py:139` (`reshape_multi`), which divides every
  sharded axis of the new shape by the shard count of the loop variable `rng`
  left over from `:130`: the count of the last sharded axis. On a value
  sharded on two axes with different counts, the shard's reshape has the
  wrong size, and scheduling fails. Repro, a 2×4 mesh on eight devices:
  `rng = UOp.range(8, -1, AxisType.DEVICE); r0, r1 = rng // 4, rng % 4`,
  `t = Tensor(Tensor.arange(48).float().reshape(4, 12).contiguous().realize().uop.copy_to_device(devs)._shard(0, r0)._shard(1, r1).unshard((0, 1), (r0, r1)))`,
  then `(t.reshape(4, 12, 1) * 2).to("CPU").schedule_linear()` raises
  `ValueError: size mismatch, can't reshape ((2, 3)) -> ((1, 3, 1))`.
- **tolk:** `lib/schedule/multi.ml:366` (`reshape_multi`).
- **Differs:** each sharded axis of the new shape is divided by the shard
  count of its own range, so the shard of the repro reshapes to `(2, 3, 1)`.
  When the sharded axes share one count, the two agree.
- **Reason:** (b): rune's multi-axis placement (RFC 0005's meshes, data by
  tensor parallelism), where a mesh's two axes have different shard counts.
- **Pinned by:** `Tolk.Multi › multi_pm › two sharded axes › a reshape
  divides each sharded axis by its own count` and `› a reshape of a mesh
  of 2 by 4 devices keeps each tile`, and the law `Tolk.Multi ›
  multi_pm › laws › a rewritten value holds the value computed whole` on grids
  of 2 by 2 and 2 by 4 devices.

## D22. An emulated long converts no float past its words' range

- **tinygrad:** `codegen/decomp/dtype.py:33-34` (`l2i` makes a long's low
  word by casting the float to an int32, and its high word by casting the
  float over 2^32).
- **tolk:** `lib/codegen/decomp/decomp_dtype.ml` (`l2i`, the cast of a
  float to a long).
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

## D23. A shard selected through a movement reads its own shard

- **tinygrad:** `schedule/multi.py:40-41` (`replace_allreduce`), which moves
  an `MSELECT` before a movement and keeps the movement's arguments as they
  are. When they hold the device range, as a sharded shrink's start
  `drange * n` does, the selected value, now on one device, still reads the
  range, and so reads the shard of whichever device runs it. Repro:
  `x = Tensor([10.,20.]).shard(("CPU:0","CPU:1"), 0).realize(); w = Tensor([1.,2.]).to(("CPU:0","CPU:1")).realize(); (x + w)[0:1].tolist()`
  raises `RuntimeError: unbound Variable '_device_num'`.
- **tolk:** `lib/schedule/multi.ml:133` (the rule), with `:54`
  (`at_device`).
- **Differs:** the movement's arguments take the selected shard's position
  for the device range, as a shrink moved before an `MSTACK` already does
  (`_apply_shrink`), and are simplified.
- **Reason:** (b): rune's sharded programs, where a shard is selected from a
  value computed from a sharded and a replicated one.
- **Pinned by:** `Tolk.Multi › multi_pm › shard selections › a
  selection of a movement by the device range takes the selected device's
  position`, and the law `Tolk.Multi › multi_pm › laws › a
  rewritten value holds the value computed whole`.

## D24. Rewrites keep IEEE and modular values

- **tinygrad:** the rules and bounds each facet names below.
- **tolk:** each facet's lines below; `test/gen/tinygrad.patch`, the same
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
  that stays is that of powers, a constant one computed by products and square
  roots and any other as `exp2 (y * log2 x)` (`Transcendental.xpow`).
  - **Integer bounds wrap.** tinygrad: `uop/ops.py:1104-1163` (`_min_max`),
    `:1154-1164` (a cast), `:1147-1149` (a constant table). tolk:
    `lib/uop/ops.ml:1484` (`min_max`), `:1557` (`cast_bounds`), `:1525`. The
    bounds of a committed integer that leave its type are the type's; an
    integer cast to an integer keeps its interval, which then wraps, where
    tinygrad clamps a signed target to the overlap; a constant table holding a
    NaN has its type's bounds, as a NaN constant has. tinygrad folds uint8
    `(u + 1) < 1` to `false`, which is `true` at 255, int8 `(y + 100) < 0`
    for `y` in `[0, 50]` to `false`, which is `true` at 40, and
    `cast (cast (y + 100) uint8) int32` to one cast, `-116` where the machine
    gives `140`.
  - **Wrapping rules.** tinygrad: `uop/symbolic.py:282` (`(x // c1) // c2`),
    `:285` (`c0 + x < c1`), `uop/divandmod.py:101` (`(x // c + a) // d`),
    `codegen/simplify.py:100-103` (`x + y < c`, `x * y < c`) and `:123`
    (`x + y <> c` under a cast), `uop/symbolic.py:390-398`
    (`reduce_mul_chain`, a factor out of a maximum). tolk:
    `lib/uop/shape.ml:2614,2623`, `lib/uop/shape.ml:1630`,
    `lib/codegen/simplify.ml:252,268,276,344`, `lib/uop/symbolic.ml:167`.
    Each applies to a committed integer only where every value it computes
    fits the type; the comparisons of `Simplify` apply to integers only, since
    moving a float term rounds, and `x + y <> c` only under a cast that does
    not narrow. tinygrad folds uint8 `u - 1 < 255` for every `u`, int32
    `(w // 2 + 2^30) // 2` at `2^30` to `-268435456` where the machine gives
    `805306368`, `(w // 65536) // 65536` to a division by `2^32`, which
    wraps to `0`, the uint8 maximum of `r * u` over `r < 4` to `3 * u`,
    `253` at `255` where the machine gives `255`, and the int8 maximum of
    `r * p * q` over `r < 2` to `p * q`, `-112` at `12, 12` where the
    machine gives `0`.
  - **Float folds.** tinygrad: `uop/symbolic.py:117` (`x + 0`), `:170-176`
    (`x / x`, `(x * y) / y`, `x * 0`), `:247` (`(x / y) / z`). tolk:
    `lib/uop/shape.ml:2100,2273`. A float `x + 0` is `x` only for `-0.`
    (`-0. + +0.` is `+0.`); `x * 0` is `0` for integers and booleans only (a
    float product by zero is NaN at an infinity or a NaN and `-0.` at a
    negative `x`); `x / x`, `(x * y) / y` and `(x / y) / z` are gone, since
    true division is always a float: `x / x` is NaN at 0, `(1e10 * 1e30) /
    1e30` is `inf`, and `(1e20 / 1e20) / 1e20` is `1e-20` where
    `1e20 / 1e40` is `0.`.
  - **Signed zeros.** tinygrad: `uop/symbolic.py:248` (`-(x + c)`), `:267`
    (complementary selections), `:472` (`-(x + y)`). tolk:
    `lib/uop/shape.ml:2460,2528`, `lib/uop/symbolic.ml:247`. For integers and
    booleans only: `-(x + 3)` at `x = -3` is `-0.`, where `-x + -3` is `+0.`, and
    `where c t 0 + where c 0 f` at `t = -0.` is `+0.`, where `where c t f` is
    `-0.`.
  - **Reassociation.** tinygrad: `uop/symbolic.py:240-246` (like terms),
    `:264-265` (a sum of two selections), `:279-280` (two constants of an
    associative operation), `:293-294` (constants to the end), `:390-398,470`
    (`reduce_mul_chain`). tolk: `lib/uop/shape.ml:2435,2516,2597,2653`,
    `lib/uop/symbolic.ml:87`.
    Sums, products and maxima regroup for integers and booleans only, and a
    factor leaves a float reduction nowhere: `(x + 1e8) + -1e8` at `x = 1` is
    `0.`, where `x + 0.` is `1.`; `(y + x) + x` at `y = 1`, `x = 2^-24` is
    `1.`, where `y + x * 2` is `1.0000001`; `x * 1.1 + x * 2.2` at `1.` is
    `3.3000002`, where `x * 3.3` is `3.2999999`; a sum over `r < 3` of
    `(r * 2e38) * 0.25` is `inf`, where `0.25 * 2e38` times the sum of `r` is
    finite; and a maximum keeps a NaN only as its first operand, so
    `max (NaN, max (x, 0.))` is NaN, where `max (x, max (0., NaN))` is `x`.
    `x + x` is still `x * 2`, which is exact.
  - **Maxima.** tinygrad: `uop/symbolic.py:273-275`. tolk:
    `lib/uop/shape.ml:2560,2580`. A maximum by bounds applies to integers
    only, since a float's bounds leave out NaN and the order of zeros:
    `max (x, inf)` at NaN is NaN, where the fold gives `inf`. A selection that
    computes a maximum becomes one for integers only: rune builds a float
    maximum that follows IEEE from selections, which the fold would turn back
    into a maximum that renders as the target's own, and it took `-0.` and
    `+0.` for one constant: `where (x < -0.) 0. x` at `-1e-45` is `+0.`,
    where `max (x, -0.)` is `-0.`. A float ReLU stays a selection.
  - **Reciprocal and sigmoid forms.** tinygrad: `uop/symbolic.py:463-468`.
    tolk: the rules are left out of `sym`. `1 / (x * x)` at `1e20` is
    `0.`, where `(1 / x) * (1 / x)` is `1e-40`; `x * (1 / (1 + x))` at `1e-8`
    is `1e-8`, where `1 - 1 / (1 + x)` is `0.`, and at `inf` is NaN, where it
    is `1.`.
  - **Pow.** tinygrad: `uop/symbolic.py:16-21` (`simplify_pow`), `:190`
    (`c ** x`). tolk: `lib/uop/shape.ml:1813,2328`. The reciprocal of the
    base is taken for exponents of magnitude at least 1 only, where the power
    overflows whenever the reciprocal does: `1e-40 ** -0.8` is `1e32`, where
    `(1 / 1e-40) ** 0.8` is `inf`. A float half-integer power selects `+0.`
    at `-0.` and `+inf` at `-inf`, where `sqrt` gives `-0.` and NaN. `c ** x`
    goes to `xpow` for `c = inf`: `inf ** 0` is `1.`, where
    `exp2 (0 * log2 inf)` is NaN.
  - **Simplify's reductions.** tinygrad: `codegen/simplify.py:104-111`
    (`sum_between`), `:115-116` (a parameter lifted out of a sum) and `:118`
    (a product by a boolean cast). tolk: `lib/codegen/simplify.ml:281,368,379`.
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
  - **Transcendental polynomials.** tinygrad: `helpers.py:153` (`polyN`, from
    `0.0`). tolk: `lib/codegen/decomp/transcendental.ml:388` (`poly_n`). A
    polynomial starts from its first coefficient, so the decompositions no
    longer rely on the float folds of `0. * x` and `x + 0` to be what tinygrad
    renders.
  - **Tensor-core accumulators.** tinygrad: `codegen/__init__.py:102-104`
    (`pm_wmma_add`, which adds the running sum to a WMMA's accumulator
    operand). tolk: `lib/codegen/codegen.ml:167`
    (`wmma_accumulate`). A WMMA built for a sum starts from a zero accumulator, and the
    running sum used to be added to it, `+0. + acc`, which only the float
    `x + 0` fold removed: with that fold kept to `-0.`, every tensor-core
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
  modular and whose floats follow IEEE. rune's `check_wrapping_comparisons`,
  `check_float_identities`, `check_signed_zeros` and `check_pow` run compiled
  graphs of each kind against eager nx; rune2's `jit › an integer maximum of a
  running sum wraps` checks a compiled maximum of a wrapping sum against the
  value nx states.
- **Pinned by:** each facet by value tests, which evaluate a graph with the
  reference interpreter before and after the rewrite at bindings where the
  restricted rewrite changes the value:
  - integer bounds: `Tolk.Ops › bounds › a committed integer that can
    leave its type has its bounds`, `› an integer cast to a signed type
    it leaves wraps`, `› a constant table holding a NaN has its type's
    bounds` and the law `› bounds hold the value a committed integer
    wraps to`;
  - wrapping rules: `Tolk.Symbolic › integers wrap` (every test);
    `Tolk.Divandmod › nested divisions › a committed division stays
    where x + a * c wraps`; `Tolk.Simplify › keeping values`,
    its comparison tests;
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
  `codegen/__init__.py:197-209` (`reduce_ranges_to_acc`,
  `expand_horizontal_reduce`: a sum adds its products unfused),
  `codegen/decomp/op.py:118-121` (`a*b + c` becomes `MULACC` for a renderer
  that writes it), `renderer/cstyle.py:139-150` (no C-style language writes
  `MULACC`).
- **tolk:** `lib/runtime/support/compiler_cpu.ml` (`-ffp-contract=off`),
  `lib/runtime/support/compiler_amd.ml` (`-ffp-contract=off`),
  `lib/runtime/support/compiler_cuda.ml` (`--fmad=false`),
  `lib/runtime/support/compiler_metal.ml` (`#pragma METAL fp contract(off)`
  before the source); `lib/helpers.ml` (`Diskcache.version` 2, since a cached binary
  compiled with contraction answers the same key). `lib/codegen/codegen.ml`
  (`fuses`, `reduce_ranges_to_acc`, `expand_horizontal_reduce`, and the
  operations `full_rewrite_to_sink` decomposes with, without `Mulacc`);
  `lib/renderer/cstyle.ml` (`fma`: `fma` in C-style and Metal,
  `__builtin_fmaf` and `__builtin_fma` in Clang and HIP, `__fmaf_rn` and
  `__fma_rn` in CUDA). `lib/uop/ops.ml` (`mulacc`, and its arm in
  `exec_alu`); `lib/uop/shape.ml` (`pm_data_invalid`'s `Mulacc` rule).
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
- **Reason:** (b). RFC 0012's Law 1, as the maintainer amended it: a
  reduction's result is in the rounded-sum class, whose error bound a fused
  multiply-add stays within; every other compiled result meets eager nx, which
  rounds a product and a sum apart, and rune's accurate compositions
  (two-part products and sums) are exact only where each operation rounds as
  written. The fusion is the IR's, so that only a sum's products fuse, where a
  compiler's contraction fuses any `a*b + c` of an expression. `Nx.fma`
  (RFC 0015's D3) states a `Mulacc` whose compiled bits must be eager's, so its
  fold rounds once too, and `Nx.ewma`, built on it, compiles. The sofo
  sketch's kernel `r_128_128_3_4_25_4` runs 1.74 ms with its sums fused,
  against 3.19 ms unfused and 1.79 ms under Clang's contraction (one E-core of
  an Intel Core Ultra 5 235, the same harness).
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
  invalid is invalid`. In rune, `Test_lower_arith › multiply-adds › a
  product that the sum cancels is kept whole` and `Test_compiled_rules ›
  compositions › a compiled ewma is its eager value, bit for bit, and so is
  its gradient`.

## D26. A minus never meets a minus in C-style source

- **tinygrad:** `renderer/cstyle.py:140` (`Ops.NEG` is `-{x}`), `:144`
  (`Ops.SUB` is `({a}-{b})`).
- **tolk:** `lib/renderer/cstyle.ml` (`joined`, `infix`, `Neg`).
- **Differs:** a negation or a subtraction whose operand starts with a minus
  sign, itself a negation or a negative constant, is written with a space
  between the two signs: `(a- -b)`, `- -b`. tinygrad writes `(a--b)`, which C,
  Metal, CUDA and HIP read as the decrement operator and reject. Every other
  source is written as tinygrad writes it.
- **Reason:** (b). rune's lowering subtracts negated values (`acos`'s
  `1 - (-x)` before it was written `1 + x`), and the graph may reach the
  renderer so; a source the compiler rejects is a failed program.
- **Pinned by:** `Tolk.Cstyle › negation › a minus before a minus is
  apart › *` (every renderer), `› Clang compiles and runs a difference with a
  negated operand` and `› … with a negative constant`.

## D27. Constants keep a NaN's bits

- **tinygrad:** `dtype.py:79-82` (`DType.const` makes every NaN `math.nan`),
  `dtype.py:16-19` (`ConstFloat` compares every NaN equal to every other), and
  `uop/symbolic.py:23-26` (`fold_bitcast` converts the constant with
  `truncate` before reading its bits).
- **tolk:** `lib/dtype.ml:17` (`equal_const`), `:27` (`hash_const`) and
  `:647` (`const`); `lib/uop/ops.ml` (`arg_key`, which `key` reads);
  `lib/uop/shape.ml:1835` (`fold_bitcast`);
  `test/gen/tinygrad.patch`, which applies the same rules to tinygrad before
  the goldens are generated.
- **Differs:** a float constant is its bits. `Dtype.const` keeps a NaN's sign
  and the payload its type holds, a signalling NaN staying one, and constants
  compare and hash by their bits, so NaNs of different bits are different
  nodes, with different keys: a schedule cached under one NaN's key is never
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
  negative NaN, and so does the patched tinygrad's.
- **Reason:** (b), as D20's: a kernel's bitcast and nx's are byte
  reinterpretations, so a bitcast that rune folds must keep the bits as they
  do, and a rewrite must keep a graph's value.
- **Pinned by:** `Tolk.Ops › identity › NaN constants of different bits
  are different nodes`; `Tolk.Ops › key › tells apart NaN constants of
  different bits`; `Tolk.Ops › exec_alu › an invalid operation's
  NaN is the canonical positive quiet NaN, whatever the host gives`,
  which compares bits and fails on an x86 host without the canonical NaN; `Tolk.Symbolic › symbolic_simple › casts › a
  bitcast round trip of every 8- and 16-bit word folds to its value`, which evaluates each graph
  with the reference interpreter before and after `simplify`;
  `Tolk.Dtype › const › const keeps the bits of every 8- and 16-bit float
  word`.

## D28. A host program has rig's entry

- **tinygrad:** `codegen/__init__.py:449-451` (`do_compile`: the binary is the
  source compiled), `runtime/ops_cpu.py:58-70` (`CPUProgram.__call__` calls
  the kernel's own signature through ctypes, every argument a `c_uint64`).
- **tolk:** `lib/codegen/codegen.ml` (`host_entry`, `do_compile`).
- **Differs:** a program for a CPU target is compiled from its source with
  the kernel renamed `NAME_` and an entry `NAME` appended:
  `void NAME(const struct NAME_args *p, long long first, long long end)`. The
  structure holds a pointer to each buffer, in the order of the program's
  globals, then each variable but a split's bounds, in order, each aligned to
  its size, and for a split program (D80) the iterations of its loop. The
  entry passes them on to the kernel, as C converts them to its parameters'
  types, and `first` and `end`, at most the iterations, as the split's
  bounds. The program's source is the renderer's; only its binary differs.
- **Reason:** (c). rig's host calls a function of linked code with one ABI,
  `void f(const void *params, int64_t first, int64_t end)`, places each
  buffer's address among the parameters, and names a submission's span after
  the entry. The entry keeps the kernel's name in profiles, and passes each
  scalar at its own type: an Apple arm64 stack argument of 32 bits takes 4
  bytes, where ctypes' `c_uint64` would take 8.
- **Pinned by:** the `Codegen` suite, `programs › a host program's binary
  holds rig's host entry after the kernel` and `› a host program runs as a
  launch of rig's host, with ten scalars past the argument registers`; the
  Tolk_engine suite, `host programs in blocks › a program writes the same
  bits in 1, 3 and 32 blocks`.

## D29. A float32 tensor core takes operands widened from its narrow input

- **tinygrad:** `codegen/opt/postrange.py:185` (`_apply_tc_opt`: a core
  applies only when both operands of the multiply have its input dtype),
  `:222` (the core multiplies the multiply's own operands).
- **tolk:** `lib/codegen/opt/postrange.ml` (`tc_operand`, `try_core`,
  `use_wmma`); `test/gen/tinygrad.patch`, which gives tinygrad's matcher the
  same rule before the goldens are generated.
- **Differs:** a core whose input is a narrow float (`half`, `bfloat16`, an
  8-bit float) and whose output is `float32` also takes a `float32` operand
  that is a cast of its input dtype, and multiplies the narrow value. The
  product of two narrow floats is exact in `float32`, and the core computes
  it exactly, so the widened product and the core agree bit for bit; a core
  with a narrow output still takes no widened operand, since its products
  round. Cores are tried in the renderer's order, so Metal, whose `float32`
  core comes first, keeps it. tinygrad never builds the widened form, which
  its `dot` does not make, so no golden it made before changes: `generate.py
  --check` matches every one with the patch applied.
- **Reason:** (b). rune's `Matmul` lowering computes nx's products of narrow
  floats, which are exact, as `MUL(CAST f32 a, CAST f32 b)` summed at
  `float32` (RFC 0012, the `Matmul` row); without this rule it takes no
  tensor core on CUDA or HIP, where the `float32` core is TF32 or absent, and
  gpt-oss's `float16` and `bfloat16` products on CUDA lose theirs.
- **Pinned by:** the Postrange suite, `apply_opts optimises a kernel as
  tinygrad does › tc_{cuda,amd}_widened_{half,bfloat}.golden` (the core
  applies on CUDA sm_89 and HIP gfx1100) and
  `› tc_{cuda,amd}_widened_{half,bfloat}_shaped.golden`, with `optimising
  keeps a kernel's writes › small kernels › tc_*_widened_*_shaped` (shaped for
  the core, the kernel keeps its values); rune's
  `lower_linalg › tensor cores › cuda › *`. The core's exact product
  is README's hardware check.

## D30. Withdrawn

A range around calls on devices with queues was a loop of their batch, its
commands repeated per trip. A range is a loop of rig's program, one submission
per trip of each submission under it (D146).

## D31. Payne-Hanek reduces exactly, to the nearest quadrant

- **tinygrad:** `codegen/decomp/transcendental.py:66-113`
  (`payne_hanek_reduction`) and `:164-166` (`sin_poly_large`).
- **tolk:** `lib/codegen/decomp/transcendental.ml:124`
  (`one_over_two_pi`), `:173` (`turn_fraction`), `:285` (`radians`) and `:326`
  (`payne_hanek`); `test/gen/tinygrad.patch`, which gives tinygrad the same
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
  switch-over; both must give nx's sine there. Every compiled sine or cosine
  of an unbounded angle builds the reduction, and a kernel's compile time
  grows with its nodes: selected as tinygrad's `_take` does, over this
  40-word table, the float64 words would take 345 selections and 43
  comparisons of 728 nodes; by bits they take 111 and 6 of 428.
- **Pinned by:** `Tolk.Transcendental › reductions ›
  payne_hanek_near_multiples.golden`, remainders and quadrants of float32 and
  float64 angles near multiples of `pi/2`, from pi to 1400 bits, which the
  old reduction fails; `› payne_hanek_reduction removes quarter turns`; and
  the `payne_hanek_*`, `xsin_*` and `values` goldens, from the equally
  patched tinygrad.

## D34. Withdrawn

A profiled Metal command buffer waited in its stamps retained. rig's Metal
driver commits and times its own command buffers (D146).

## D35. A selection of a value on no device is that value

- **tinygrad:** `schedule/multi.py:24-27` (`lower_broadcast_copy` simplifies
  the copy's source and reads its device), `uop/ops.py:895` (the device of an
  `MSELECT` asserts that its source is on several devices).
- **tolk:** `lib/schedule/multi.ml:75-87` (`pm_unselect_deviceless`,
  applied in `lower_broadcast_copy`).
- **Differs:** after simplifying the source of a copy to several devices,
  every shard selection of a value on no device is replaced by that value, so
  the copy stacks it on each device. tinygrad raises instead. Simplifying folds
  an integer product with zero on a sharded value to a constant, which has no
  device, and the selection above it then has no shards to select: a shard of
  such a product that a cast, a sum or a max follows raises "mselect must be on
  tuple device, getting None". The fold is right, since a constant is on no
  device, so the fix is in the selection, not in the fold.
- **Reason:** (b). rune lowers sharded integer code, where multiplying by zero
  and then reducing or casting is ordinary, onto this rewrite; the reference
  refuses such programs.
- **Pinned by:** `Tolk.Multi › multi_pm › laws ›` "a shard of a product
  with zero cast to a float is zero (D35)", "a shard of a sum of a product with
  zero is zero (D35)", and the law "a rewritten value holds the value computed
  whole" over every drawn program.

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

## D41. The memory plan sees a range's calls, and leaves buffers reached through views

- **tinygrad:** `schedule/memory.py:28-33` (`memory_plan_rewrite` takes each
  entry's buffers from its sources after the first, through `_collect_bufs`,
  which passes buffers, `MSELECT` and `MSTACK` only).
- **tolk:** `lib/schedule/memory.ml:18` (`viewed`), `:27` (`calls`) and
  `memory_plan_rewrite`'s lifetimes (`through_views`).
- **Differs:** an `END` of ranges around calls counts as one entry that takes
  the buffers of each call it runs, and a buffer that some call reaches
  through a view (`SHRINK`, `BITCAST`, `AFTER`) is not planned. tinygrad
  takes an `END`'s sources after the first, its ranges, so a buffer used
  inside a range lives only where it is used outside it; and a buffer reached
  both directly and through a view lives only where it is reached directly.
  Either way the plan can place it over bytes another buffer still needs. A
  schedule that reaches a buffer only through views, as tinygrad's do, plans
  as tinygrad's. The rule gives up the reuse of a buffer reached through a
  view: a scan's stacked rows, its double-buffered carries and any other
  viewed intermediate keep their own memory for the whole schedule, where
  planning them through their views could share it once they are dead.
- **Reason:** (b). `Rune.scan`'s staged loop (RFC 0012) schedules calls
  inside a range, whose arguments are views that move with it, of buffers
  the calls after it read whole; tinygrad's scheduler hands the planner
  neither.
- **Pinned by:** the Engine suite (`test/engine/tolk_engine`): `link and
  run › a planned buffer a range writes is not placed over one it leaves` and
  `› a buffer a range writes through a view is not placed over another`, each
  of which fails without its half.

## D42. Withdrawn

A batch's command buffers, and its placeholders where a device had no window,
were pinned memory. tolk makes no command buffer, and a program's memory is
rig's (D146).

## D43. Withdrawn

A queue bufferized its own commands, and a region kept its alignment. tolk
encodes no queue (D146).

## D44. An integer cast of a weak expression computes in integers

- **tinygrad:** `uop/weak.py:27-33` (`cast_weak_srcs`), `dtype.py:180-194`
  (`promo_lattice`, `least_upper_dtype`).
- **tolk:** `lib/uop/uop_weak.ml:56` (`cast_weak_srcs`).
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
- **Reason:** (b). The host programs of staged scans (D30) compute each trip's
  addresses and signal positions as 64-bit unsigned words cast from a range's
  weak integers, and computed them in float: an offset past 2^24 bytes read
  another trip's window.
- **Pinned by:** the Uop_weak suite (`test/uop/uop_weak`): `pm_commit_weak ›
  a 64-bit unsigned cast of a weak expression keeps its value past a float's
  precision` and `laws › pm_commit_weak computes an integer cast in
  integers`.

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

- **tinygrad:** `renderer/cstyle.py:139-147` (`CStyleLanguage.code_for_op`
  has no `FDIV`) and `:277-280` (Clang's adds it); `codegen/decomp/op.py:122-125`
  (a target that lists `FDIV` gets its reciprocals as divisions, and a product
  by a reciprocal, `a * (1/b)`, as the quotient `a/b`).
- **tolk:** `lib/renderer/cstyle.ml:370-374` (the `FDIV` rule of
  `base_rewrite`); `lib/codegen/decomp/decomp_op.ml:300-308` (the late rules of
  `FDIV`, which turn a reciprocal into `1/x` and nothing else).
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
  stay products; rune's `Rune.jit › division`, where a compiled product
  by a reciprocal and a compiled quotient each equal nx's eager bits at the
  value above, and its generated programs, whose products take reciprocals
  as factors (`test_jit_programs`).
  CUDA's and HIP's `/` are correctly rounded by their compilers' defaults,
  which is on the hardware checks of `test/README.md`.

## D51. Withdrawn

An NV launch read its local memory size from a word of the device. rig's NV
driver sizes local memory as it loads a function (D146).

## D52. A selection of -0. stays off a gated load

- **tinygrad:** `uop/symbolic.py:421-425` (`pm_move_where_on_load`), whose
  pattern constant `0` matches `-0.` since `-0. == 0`.
- **tolk:** `lib/uop/symbolic.ml:229` (`pm_move_where_on_load`).
- **Differs:** a selection between a load and `-0.` stays a selection.
  tinygrad moves its condition into the load's gate, and a gated load reads
  `+0.` where its gate fails, so a pad of `-0.`, lowered as a selection of
  `-0.` off the padded load, computes `+0.` on the padding.
- **Reason:** (b). nx's `pad` fills with the value given, `-0.` included
  (rune's lowering ledger, I1), and rune's compiled pad must compute eager
  nx's bits.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `signed zeros › a pad with -0. fill renders and computes -0.` and `› a pad and a
  selection of signed zeros keep the interpreter's bits`.

## D54. Storage says what it knows of where it starts within 16 bytes

- **tinygrad:** `uop/ops.py:23-37` (`ParamArg`, which has no such field) and
  `codegen/late/coalesce.py:152` (`memory_coalescing` merges a run of `l`
  elements where `l` divides its element offset, taking every buffer to start
  on a 16-byte boundary). tinygrad folds a view's offset into a kernel's
  index, except for a contiguous slice that it stages without a copy
  (`schedule/__init__.py:199`, `contiguous_mops_to_view`): a later kernel
  reads that view with vector loads that take it to start aligned, as
  `x[1:].contiguous() + 1` loads `float4`s from 4 bytes past a boundary on
  Metal.
- **tolk:** `lib/uop/ops.ml:241` (`phase` and `align`), `:3244`
  (`param_arg`, which checks them); `lib/uop/call.ml:403` (`view_start`)
  and `:482` (`storage_phase`, which `param_like` gives a parameter);
  `lib/schedule/rangeify.ml:532` (`debuf`, which gives them a kernel's
  parameter); `lib/codegen/late/coalesce.ml:130` (the merge).
- **Differs:** a parameter or buffer carries a congruence for its start: its
  first element lies `phase` bytes past a multiple of `align`, a power of two
  up to 16, the width of the widest vector access, and `phase` is a multiple
  of its element's size or of `align`, whichever is less. They default to `0`
  and `16`, tinygrad's assumption, so no graph of tinygrad's changes. A run of
  `l` elements merges where `l` divides the element's count from that
  boundary and the run is no wider than `align`, so every vector access is
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
  byte when a size is symbolic or the buffer sharded. A view whose first or
  last element is padding, or whose ends lie farther apart or closer than a
  run of its size, is never made the view, and its stage keeps the
  boundary. rune's staged loops give each body parameter the alignment and
  phase of what the call passes it. The congruence is part of the graph, so
  of a program's cache key.
- **Reason:** (b). rune's `Compiled` runs an operation over the storage it is
  given, and mapped weights put a tensor at any byte offset of its file: a
  vector access from an address that is not a multiple of its width is
  undefined in Clang's `aligned(16)` vector types, faults on CUDA, and is
  undefined in Metal, which requires device pointers aligned to their type:
  there it reads the wrong elements once Metal merges adjacent vector loads
  at constant offsets into one wider load, which drops the low bits of the
  address. Single loads at computed indices are not merged, and read
  correctly. A batched range (D30) binds each trip's window at its own
  address, so a kernel vectorised for a window that starts aligned faulted on
  x86, where Clang emits `movaps`, on the trips that moved it by 4 bytes. A
  rune loop (`Loop.cut`) passes its body a stage of what it reads that does
  not change, and the schedule makes a stage of a contiguous view the view:
  norn's NUTS step on a posterior whose data the model reads from its sixth
  double failed verification, its body compiled for storage on a boundary
  and passed the data 8 bytes past one.
- **Pinned by:** the `Coalesce` suite (`test/codegen/late/coalesce`): `phase › one float past a boundary, eight loads are of one, two, four and
  one`, `› three floats past a boundary, a load of four starts at element 1`,
  `› stores start where loads would`, `› two halves past a boundary, a load of
  four starts at element 2`, `› floats known to start on 4 bytes alone are
  not merged`, `› floats known to start on 8 bytes merge in twos` and `›
  halves three past 8 bytes merge no wider than 8 bytes, from element 1`, and
  the laws `every vector access is aligned to its width and within the
  alignment, for every phase and alignment` and `coalescing preserves
  the kernel's writes, for every phase and alignment`; the `Ops` suite: `param_arg takes a
  phase that is a multiple of the element size below 16`, `param_arg
  takes an alignment that is a power of two up to 16, and a phase below it`,
  `pp_param_arg writes a phase that is not 0 and an alignment that is
  not 16`, `equal_arg tells apart parameters that differ in phase` and
  `param_like keeps its storage's phase, moved by a shrink, and
  knows it modulo what a symbolic start moves by`;  the graph format's round trip of `a buffer past
  a 16-byte boundary`; the `Schedule` suite's `disk_view_to_linear.golden`,
  a view 8 bytes into a disk file whose parameter has no phase. In rune,
  the `Lower` suite's `a parameter of storage 4 bytes past a 16-byte boundary
  has phase 4`, `a capture of storage 4 bytes past a 16-byte boundary has
  phase 4` and `a program over an argument 4 bytes past a 16-byte boundary
  computes nx's values`, and the `Compiled` suite's `edges › an operand whose
  buffer starts 2 bytes into its memory is read where it is`, on the
  host and on Metal, whose sweeps draw buffers that start at any byte;  the `Schedule`
  suite's `contiguous_mops_to_view › a stage of doubles one past a boundary is
  known to start on 8 bytes`, `› a stage of rows padded apart, starting
  one double past a boundary, is storage of its own on one` and `› a
  stage's alignment and phase hold of the storage it gets, a view or its own`;
  rune's `Scan` suite's `compiled › a scan under jit reads a copy of a
  slice that starts off a 16-byte boundary`, `› a scan under jit reads the
  bits of a slice that starts off a 16-byte boundary` and `› a scan under jit
  reads rows of a slice that starts off a 16-byte boundary`; jera's `Split`,
  `Ode` and `Sde` suites' compiled marches, whose rows are padded apart.

## D55. Metal names a vector after its element's one-word name

- **tinygrad:** `renderer/cstyle.py:186` (`_render_dtype` names a vector
  after its element's name with spaces made underscores) and `:362`
  (`MetalRenderer.type_map` renames only `uint` and `bfloat`).
- **tolk:** `lib/renderer/cstyle.ml:87,119` (`vector_names`) and
  `:874-879` (Metal's).
- **Differs:** a Metal vector of `signed char`, `unsigned char`,
  `unsigned short` or `unsigned long` is a `char`, `uchar`, `ushort` or
  `ulong` vector, as Metal names them; tinygrad writes `signed_char4`,
  `unsigned_char4`, `unsigned_short4` and `unsigned_long4`, which Metal does
  not have, so such a kernel does not compile. Scalars keep tinygrad's names.
  Metal's vectors have 2, 3 or 4 lanes; none has 8.
- **Reason:** (b). rune's compiled backend folds and unfolds int8 arrays on
  Metal, whose kernels hold `char4` values (the Compiled suite's `fold of
  integers`).
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `sources › by
  default › metal_vector_<element>_<lanes>`, tinygrad's source with the D55
  names, for each element and 2, 3 and 4 lanes, and `metal_vector_cast_char`;
  and the slow `every GPU kernel compiles with its target's toolchain ›
  metal_vector_*`, which compiles each with MTLCompiler.

## D56. The logarithm of a negative subnormal is NaN

- **tinygrad:** `codegen/decomp/transcendental.py:250-255` (`xlog2` selects
  NaN where `d < -0.0`, then `-inf` where the reciprocal of `d` is `-inf`,
  which it means for `-0.0`).
- **tolk:** `lib/codegen/decomp/transcendental.ml:569-577` (`xlog2`),
  and `test/gen/tinygrad.patch`, which reorders tinygrad's selects the same
  way.
- **Differs:** the reciprocal of a negative number of magnitude below
  `2^-128` in float32 (`2^-1024` in float64, about `2^-16` in float16)
  overflows to `-inf` too, so
  tinygrad's last select turned the NaN of such a number's logarithm into
  `-inf`. The `-inf` select now comes first and the NaN select after it:
  `-0.0` still gives `-inf`, and every negative number gives NaN.
- **Reason:** (b). The planned AMD ISA renderer and the eager GPU kernels
  compute nx's logarithms with `xlog2`, and nx gives NaN for the logarithm of
  every negative number, subnormals included.
- **Pinned by:** the `Transcendental` suite
  (`test/codegen/decomp/transcendental`): `special values › xlog2 of a
  negative number whose reciprocal overflows is NaN, and of -0. is -inf`, and
  the negative float16 rows of `values.golden` of magnitude below
  `2^-16`, from the patched tinygrad.

## D58. A program applies no elementwise operation to a vector

- **tinygrad:** `uop/spec.py:205` (`spec_program`, which accepts an
  elementwise operation of any shape).
- **tolk:** `lib/uop/spec.ml:421` (the first rule of `program_rules`).
- **Differs:** `Spec.program` rejects an operation of `Op.Set.elementwise`
  on values, casts and bitcasts included, whose shape has an axis; a bitcast
  of memory, which views it, stays allowed. `SPEC` defaults to 1
  and code generation checks every lowered kernel against `Spec.program`, so
  a kernel that still holds vector arithmetic fails at lowering, naming the
  operation, on every target whose renderer computes lane by lane.
  Devectorize leaves none there while every source keeps its width, which D76
  keeps when every lane of a gated load's index is `Invalid`. A renderer that
  computes on vectors (D141) is checked against `Spec.vector_program`, which
  takes an operation on one axis of a power of two of lanes.
- **Reason:** (b). CUDA's vectors are structs without arithmetic, casts or
  selects, so a vector operation left after devectorize is a kernel that does
  not compile there, and one Metal renders without complaint; rune compiles
  its kernels for both.
- **Pinned by:** the `Spec` suite (`test/uop/spec`): `vectors in programs › a program has no elementwise operation on a vector` (add, cast and
  where on two lanes, and the same on one); the `Codegen` suite's `vectors in
  programs › a weak constant stored into four lanes of half is refused` and,
  on every case but the host's, `› no program applies an elementwise
  operation to a vector`; the `Cstyle` suite's `every GPU kernel compiles with its target's
  toolchain › cuda_vector_cast`, tinygrad's source, which NVRTC rejects
  (slow, an expected failure).

## D60. A loop of a call stays in the schedule

- **tinygrad:** `schedule/rangeify.py:165-168` (`pm_no_views`, which strips
  every view of storage), `:335-337` (`split_store`, which makes a kernel of
  an end of a call), `schedule/__init__.py:72-75` (`create_schedule`, which
  schedules the call of an end and drops the end), and `uop/spec.py:254-280`
  (`spec_kernel_graph`, which admits device ranges only, and no end).
- **tolk:** `lib/schedule/rangeify.ml:236` (`loop_range`), `:241`
  (`pm_no_views`, which keeps a view that moves with a loop's range and
  untags the range), `:618` (`split_store`), `lib/schedule/schedule.ml:141`
  (`create_schedule`: `loops`, `argument` and the kept end),
  `lib/uop/spec.ml:489` (`loop`, `loop_bound` and `kernel_graph`), and
  `test/gen/tinygrad.patch`, which gives tinygrad's `spec_kernel_graph` the
  same rules.
- **Differs:** an end of a precompiled call over loop ranges
  (`Axis_type.Loop`), `AFTER(buf, END(CALL(fn, args), r))`, is an effect of
  the kernel graph and no kernel. The arguments that move with `r` stay
  views, the range loses the tag of kernel ranges, and `create_schedule`
  keeps the end around the call, ordered by the call's reads and writes.
  Once the call's body is scheduled and resolved, the schedule is `LINEAR
  [END(LINEAR [CALL k1; CALL k2; ...], r)]`, the form that the memory plan of a range (D41) and `Tolk_engine.run`, a
  loop of rig's program (D146), run. An end over device ranges is still the call, bound at launch.
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
  an end of a call over a loop schedules the call in its loop`, `› an
  end of a call over device ranges schedules the call`, `› a loop runs
  after the call that writes what it reads`, `› a loop runs before
  the call that overwrites what it reads`, and
  `create_linear_with_vars › loops of calls › a loop of a precompiled
  call is a loop of its body's calls` and `› a loop inside a loop's body
  keeps its own end`; and the `Tolk_engine` suite
  (`test/engine/tolk_engine`): `link and run › a scan runs its body
  once per trip, carrying in place`, against the unrolled loop on the
  host. The `Spec` suite (`test/uop/spec`): `kernel_graph › loops of calls › accepts a loop of a call over a loop range`, `› refuses a loop of
  a call over a range of any other kind`, `› accepts an open device range`
  and `› refuses a weak sum of a weak integer variable`; its
  `verdicts.golden` records the patched spec on ends and shrinks.

## D61. A stack takes the per-shard sub-view of a whole source

- **tinygrad:** `schedule/multi.py:188-200` (`stack_multi`, which stacks a
  source that is not sharded as it is beside the shards of the others).
- **tolk:** `lib/schedule/multi.ml:500` (`stack_multi`), and
  `test/gen/tinygrad.patch`, which gives tinygrad the same rule.
- **Differs:** in a stack whose sharded sources are sharded alike, a whole
  source, one value of the full shape on every device, takes its per-shard
  sub-view (`shard_subview`), as an elementwise operation's does
  (`alu_multi`). tinygrad stacks it whole beside the shards: the stack's
  shards then mix a shard with a whole value, and
  `Tensor.stack(a.shard(devices, axis=0), b.shard(devices))` gives wrong
  values, or fails at a later operation on the shard sub-view's shape check
  when the whole source comes first.
- **Reason:** (b). rune's compiled `Nx.stack` of a value sharded on an
  axis and a replicated one gives other values than eagerly. The same stack
  is how rune lowers a concatenation, and a scatter-add, of such values.
- **Pinned by:** the `Multi` suite (`test/schedule/multi`): `multi_pm ›
  recorded › programs › stack_whole_multi.golden` and
  `› stack_whole_first_multi.golden`, and `› values › stack_whole writes
  what it wrote before` and `› stack_whole_first …`, from the patched
  tinygrad, whose values for both stacks equal numpy's.

## D62. A move of an emulated float keeps its bits

- **tinygrad:** `codegen/decomp/dtype.py:181-206` (`pm_float_decomp`, which
  widens every load of an emulated float to the emulating float and narrows
  every store back).
- **tolk:** `lib/codegen/decomp/decomp_dtype.ml:677` (`moved`) and
  `:816` (the store rule of `pm_float_decomp` that stores it), and
  `test/gen/tinygrad.patch`, which gives tinygrad the same rule.
- **Differs:** a store of a value that moves stored bits without arithmetic
  (a load, a constant the float holds exactly, a selection between such
  values, and stacks and lanes of them) stores those bits, as the storage's
  unsigned integers. tinygrad widens and narrows each such value, which keeps
  its value but quiets a signalling NaN: a copy of e5m2's `0x7d`, half's
  `0x7c01` or bfloat16's `0x7f81` stores `0x7f`, `0x7e01` or `0x7fc1`.
  Arithmetic still converts, and quiets a signalling NaN (D9).
- **Reason:** (b). nx's moves preserve bits: an eager copy, movement or
  `where` keeps a signalling NaN, and rune compiles the 8-bit floats on
  the host and on Metal, where tolk emulates them, so a compiled function
  computes what it computes eagerly only if the emulated moves keep the bits
  too.
- **Pinned by:** rune's `Compiled` suite: `host › 8-bit floats › a copy
  of every 8-bit float code keeps its bits` and `› a selection of
  every 8-bit float code keeps its bits`, and the same under `metal ›`
  (slow), against eager on all 256 codes of e4m3 and e5m2; the
  `Decomp_dtype` suite (`test/codegen/decomp/decomp_dtype`): `NaNs › an
  emulated copy keeps every NaN code, a signalling one's included`
  and `› an emulated selection keeps every NaN code, a signalling one's
  included`, on every emulated float, and the goldens of the
  `where`, `flip`, `gather` and `pad` kernels, from the patched tinygrad.


## D63. Programs and schedules are kept on disk

- **tinygrad:** `codegen/__init__.py:497-505` (`to_program_config`,
  `to_program_key` and `to_program_cache`, a dictionary of the process),
  `schedule/__init__.py:127-137` (`lower_sink_to_linear`'s schedule cache,
  on disk only from `SCACHE=2`), `helpers.py:286` (`SCACHE`, `1` by
  default), and `codegen/opt/postrange.py:45-46` (a kernel's name, coloured).
- **tolk:** `lib/codegen/codegen.ml:1540` (`program_key`), `:1556`
  (`kept`) and `:1565` (`made_program`); `lib/schedule/schedule.ml:356`
  (`in_order`), `:388` (`schedule_key`) and `:393`
  (`lower_sink_to_linear`); `lib/uop/graph.ml:957`
  (`cached`); `lib/codegen/opt/postrange.ml:663` (`get_optimized_ast`'s
  name); `lib/setting.ml:266` (`scache`, `2` by default) and `:141`
  (`shaping`); `lib/setting.ml:294` (`cc`); and
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
  `getenv` where it compiles (`ALIGNED`, `ALLOW_HALF8`, `BEAM_ESTIMATE`,
  `BEAM_LOCAL_MAX`, `BEAM_MIN_PROGRESS`, `BEAM_PADTO`, `BEAM_UOPS_MAX`,
  `BEAM_UPCAST_MAX`, `DMC`, `EXPAND_SSA`, `HCQ_NUM_SDMA`, `JITBEAM`,
  `LATE_ALLREDUCE`, the `MV` and `REDUCEOP_SPLIT` variables,
  `RING_ALLREDUCE_THRESHOLD`, `SUM_DTYPE` and `WAVES_PER_SH`), and the keys
  take every one that reaches output (`Setting.shaping`), so one declared
  later, by tolk or by its caller, is keyed without being listed; a setting
  of the process alone, such as `DEBUG`, `PARALLEL` or `CC`, is not. One
  key serves memory
  and disk, so a setting changed within the process by `context` makes the
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
- **Reason:** (b). rune's first compiled call of a model is gated at 10% of
  the old rune's, which kept compiled schedules on disk. With binaries alone
  on disk, a warm process schedules and lowers every kernel again: on
  gpt-oss's tiny random checkpoint on Metal, 0.64 s of scheduling and 2.08 s
  of lowering of a 3.16 s first call, against 0.58 s for the old rune.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `programs are
  kept › a program is made again under another value of ›` each setting,
  `programs are kept on disk › a program made by one process is read back by
  the next`, `› a program made under one setting is not read back under another
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
  object cached by the default compiler is not served under CC`.

## D64. A cast to a narrow float through a float32 rounds once

- **tinygrad:** `renderer/cstyle.py:89-90` (`create_non_native_float_pats`,
  which casts a source of any type but float32 to a float32, then to the
  narrow float).
- **tolk:** `lib/renderer/cstyle.ml:434` (the rule's cast), with
  `lib/codegen/decomp/decomp_dtype.ml:531` (`narrow`, D9's).
- **Differs:** a renderer without arithmetic on a narrow float, Clang's on
  bfloat16 and HIP's on bfloat16 and the 8-bit floats, casts to it through a
  float32. A source more precise than a float32 (a float64, or an integer of
  32 or 64 bits) reaches the float32 rounded to odd, as an emulated cast's
  does (D9), so that the cast rounds once, from the exact value. tinygrad
  rounds to the float32 to nearest, then again: on the host, the bfloat16 of
  the int64 `2^40 + 2^32 + 1` is `2^40` there, where its correctly rounded
  value, which Metal's native conversion gives, is `2^40 + 2^33`. No golden
  holds such a cast.
- **Reason:** (b). nx casts with one rounding, and rune compiles its
  casts to bfloat16 for the host: an int64 arange of bfloat16 from 2^40
  computes a cast of each integer.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `casts to
  bfloat16 › an integer or a double rounds once on the host`, for
  int64, int32, uint32, uint64 and float64 sources; rune's `Jit` suite:
  `values › a bfloat16 arange from 2^40 inside a compiled call equals
  eager's`.


## D65. An emulated value is a value of its float

- **tinygrad:** `codegen/decomp/dtype.py:198-201` (`pm_float_decomp`: a cast
  to the emulated float only clamps its operand, `f2f_clamp`, and an
  operation on it computes in the emulating float; only a store rounds).
- **tolk:** `lib/codegen/decomp/decomp_dtype.ml:709` (`rounded`), `:791`
  (the cast rule) and `:815` (the operation rule), and
  `test/gen/tinygrad.patch`, which gives tinygrad the same rules.
- **Differs:** a cast to an emulated narrow float, and an operation on it,
  round their result to it in the kernel: its bits encoded as the narrow
  float's and decoded back. Every emulated node then holds a value of its
  float, and the store's conversion is exact. tinygrad rounds only at the
  store, so a value that a kernel casts to the narrow float and uses before
  storing keeps the emulating float's precision: the e5m2 of 4113 cast back
  to a float32 is 4113 there, and 4096 here, and an operation's result
  feeds the next unrounded. A lane, a stack and a selection move values
  already rounded, and stay as they are.
- **Reason:** (b). nx rounds each operation's result to its dtype, and
  rune compiles chains of operations on the 8-bit floats into one
  kernel on the host and on Metal, which emulate them: `(y + y) * y` of an
  e5m2 `y` computed eagerly differs from the kernel that rounds only once.
- **Pinned by:** the `Decomp_dtype` suite
  (`test/codegen/decomp/decomp_dtype`): `rounding in the kernel › a
  cast to an emulated float and back is the value rounded once` and `› an
  operation on an emulated float rounds its result to it`, on every emulated
  float, and the goldens of every kernel that casts to or computes in an
  emulated float, from the patched tinygrad; rune's `Jit` suite: `one
  device › a chain of 8-bit float operations rounds after each, as eager
  does`, and the same on Metal.

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

## D72. Only a view of storage has a contiguous view

- **tinygrad:** `uop/ops.py:935-948` (`UOp.contiguous_view`), which rewrites
  the index of any value with `pm_mops`, `symbolic` and
  `pm_contiguous_view_offset` (`:1905-1911`). On a constant the rewrite does
  not terminate: the offset rules mark the constant the index reaches, and
  `symbolic` rebuilds the constant without its mark, so each undoes the
  other. `UOp.const(1, dtypes.long).contiguous_view()` raises "infinite loop
  in graph_rewrite", in tinygrad and in the patched tinygrad of
  `test/gen/tinygrad.patch`. tinygrad's own caller asks only about views of
  buffers (`schedule/__init__.py:210`).
- **tolk:** `lib/schedule/prepare.ml:709` (`contiguous_view`).
- **Differs:** a value whose storage base (`Ops.storage_base`) is not storage
  (`Op.Buffer`, `Op.Alloc` or `Op.Param`) has no contiguous view, without a
  rewrite: a constant, and a computed value, which tinygrad would mark and
  return.
- **Reason:** (b): rune's jit asks whether each result of a compiled call is a
  view of a buffer it makes (`packages/rune/lib/jit.ml`, `take` and
  `whole`), and a result can be a constant or any computed value. kaun's
  decoder test `gradients › a fully padded row, compiled` has a constant
  result, and did not compile.
- **Pinned by:** the `Prepare` suite: `contiguous_view › a constant is no view`, `› a constant of one element is no view` and `› a computed
  value is no view`.

## D73. A gather is an INDEX of a tensor by a value with axes

- **tinygrad:** `uop/ops.py:353-356` (INDEX's shape: its index sources'
  shapes, then the rest of the source's), which no tinygrad pass puts in a
  tensor graph; `uop/spec.py:179-180` (an allreduce's operations);
  `schedule/indexing.py:30-45` (`realize_srcs`, `pm_generate_realize_map`),
  `:70` (the sources an INDEX's ranges reach), `:97-98` and `:136`
  (`pm_apply_rangeify`) and `:230` (`run_rangeify`'s consumer ranges);
  `schedule/prepare.py:65` (`_mop_index`), `:76` (`pm_mops`), `:94-101`
  (`fix_store_hazard`), `:114` (`split_reduceop`) and `:275-277`
  (`prepare_rangeify`); `schedule/multi.py:202` (`index_multi`) and `:291`.
- **tolk:** `lib/uop/spec.ml:363`; `lib/schedule/indexing.ml:46`
  (`storage`), `:53` (`is_gather`), `:55` (`realize_gathered`), `:125`
  (`data_srcs`), `:180` (`convert_gather`), `:257` and `:440`
  (`run_rangeify`'s consumer ranges); `lib/schedule/prepare.ml:90`
  (`move_index`), `:133` (`mops`), `:159` (`pm_tensor_mops`) and `:195`
  (`fix_store_hazard`'s `reorders`); `lib/schedule/multi.ml:558`
  (`same_devices`), `:569` (`gather_shards`), `:592` (`gather_multi`) and
  `:790`; the same rules in `test/gen/tinygrad.patch`, which the goldens are
  recorded with.
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
  - a store whose value gathers from its destination's storage materialises
    the value first, as for a permutation; the index, and an index by
    scalars, read the destination in place;
  - sharded along axes after the gathered one, each shard gathers its own.
    Sharded along the gathered axis, each shard reads the rows it holds and
    `0` elsewhere, and the shards' bits are joined by an allreduce with a
    bitwise or, which keeps `-0.` and NaN payloads a sum would change; the
    spec admits Or as an allreduce's operation for it. Every device then
    holds the whole result, so the gathered rows cross the devices: a model
    that shards weights by rows routes its activations instead;
  - a sharded index gathers each part of itself from a whole value. From a
    sharded value it is first joined whole on each device, in `int64`, by the
    copy's sum of its padded shards, and gathers as a whole index does, so
    every device holds the result: a gather of experts split over devices by
    routes split over them runs today. The join moves the index, which is
    small beside the rows;
  - refused: a gather by several indices (rune emits one), by an index on
    other devices than its value, of a value sharded on the gathered axis and
    another, and across devices of elements whose width has no unsigned
    integer type.
  The index is in range: rune clamps it and selects `0` where it was not.
- **Reason:** (b): rune lowers `Nx.take` and `Nx.take_along_axis` to a
  one-hot sum over the whole axis, which costs a reduction per element and
  made gpt-oss's decode step schedule embedding and cache reads as their own
  kernels; D68, which recognised the sum in the scheduler, was withdrawn.
  rune's gather lowering emits this INDEX instead.
- **Pinned by:** the Rangeify suite's `gathers` programs (`index_rows`,
  `index_rows_computed`, `index_read_twice`, `index_under_reduce`,
  `index_broadcast`, `index_view_source`, `index_zip`, `index_of_index`,
  `index_zero_fill`, `index_assign_self`) and their kernel counts,
  `gather_kernel_counts`; the Prepare suite's `gather_of_self`,
  `gather_by_self` and `gather_of_reshape`, and `pm_mops › rules › an index
  that loads from a storage state moves with the state as it is`; the Multi
  suite's `gathers` programs and `multi_pm › gathers`; the Schedule suite's
  `shard_gather_rows` goldens; the Spec suite's verdict `an allreduce by a
  bitwise or`; rune's Jit suite (`gathers` and `gathers across devices`).

## D74. Float arithmetic has bounds, and a bounded sine takes the short reduction

- **tinygrad:** `uop/ops.py:1104-1163` (`UOp._min_max`), which bounds binary
  operations on integers only (`:1105`), has no case for `TRUNC` or `NEG`, and
  bounds a `WHERE` by both its branches whatever selects them (`:1142`),
  and gives a value it cannot bound, or a cast it cannot, its type's
  `min` and `max` (`:1163`), finite in the 8-bit formats without infinities
  (`dtype.py:74-78`);
  `codegen/decomp/transcendental.py:170-191` (`xsin`), which builds the
  Payne-Hanek reduction unless its caller passes `fast`, and chooses between
  the reductions with `x_abs < switch_over` (`:187`), a comparison that the
  decompositions' rewrite (`symbolic_simple`) does not fold.
- **tolk:** `lib/uop/ops.ml:1485` (`unbounded`), `:1519`
  (`compute_min_max`), `:1536` (`Where`), `:1568` (`Trunc`), `:1572` (`Neg`),
  `:1585` (`selected`), `:1600` (`float_bounds`) and `:1644` (`cast_bounds`);
  `lib/codegen/decomp/transcendental.ml:444` (`xsin`); `test/gen/tinygrad.patch`,
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
  of its type, so a value cast from an unbounded float stays unbounded. With
  the type's finite greatest value instead, `a < inf` of an 8-bit NaN folded
  true and a compiled sine, cosine or tangent of it gave -1, 0 and -452.
  `xsin` of an angle whose
  bounds lie strictly within `switch_over` is its `fast` form, the Cody-Waite
  reduction alone.
- **Reason:** (b): the sine or cosine of a bounded value on the CPU, such as
  sofo's and symo's Gaussian draws (`Nx.Rng.normal`), whose angle is `2 pi u`
  for a uniform `u` in `[0, 1)` fused into the draw. rune's `sin` and `cos`
  (`packages/rune/lib/lower_arith.ml`, `by_quadrant`) skip their own long
  reduction for an angle bounded below their limit, and clamp their remainder,
  at most about a quarter turn, to `[-pi/2, pi/2]` by two such selections
  (`within_quarter_turn`), since the cancellation in `a - q pi/2` hides that
  bound from interval arithmetic. The host's renderer has no sine, so `xsin`
  reduces what remains. Without this entry every
  compiled normal draw built the Payne-Hanek reduction for angles below
  `2 pi`. On kimchi's x86 E-cores (`taskset -c 6-13`), drawing `2^20` samples
  with `Rune.jit` (median of 31 calls, the key's uniforms included) took 29.8
  ms in float32 and 113.9 ms in float64, and takes 12.8 ms and 18.9 ms with
  this entry, rune's `by_quadrant` and the draw's fused uniforms; the first
  call of a draw of 1000, compiling included, took about 200 ms and 420 ms,
  and takes 115 ms and 85 ms. The sine of `2^20` values read from a buffer
  took 62 ms in float32 and 202 ms in float64, its first call about 370 ms
  and 1200 ms, and takes 22.7 ms and 79 ms, its first call 180 ms and 510 ms,
  with `within_quarter_turn`; the cosine likewise.
- **Pinned by:** the Ops suite (`test/uop/ops`): `bounds › bounds hold every
  value a float operation rounds to`, sums, differences, products and
  selections, `› a float operation of bounded operands is bounded`,
  `› a float operation of an operand within the subnormals bounds a flushed
  operand too`, `› a float operation of an unbounded operand, or that can
  overflow, has its type's bounds`, `› a float selection by a
  comparison narrows what it selects`, `› a float truncation and
  negation map their operand's bounds`, `› a value of a float type
  without infinities, which may be NaN, has no finite bounds` and the float
  rows of `binary_bounds.golden`, from the
  equally patched tinygrad; the Transcendental suite: `graphs › a sine of an
  angle bounded below the switch-over is its fast form`; and
  rune's Compiled suite: `› 8-bit floats › the sine, cosine and tangent of
  an 8-bit float NaN are NaN`, on the host and on Metal (slow); and rune's
  `lower_arith` suite: `long reductions › a normal draw takes
  none`, `› the sine of an angle read from a buffer takes its own only` and
  `› the cosine of an angle read from a buffer takes its own only`.

## D75. A selection by a constant condition

- **tinygrad:** `uop/symbolic.py:198`, which folds a `WHERE` whose condition is
  a `CONST` (`UPat.cvar("gate")`), and `:228-233` (`fold_where_closure`),
  which substitutes `True` for a selection's condition in its true branch and
  `False` in its false branch, whatever the condition is.
- **tolk:** `lib/uop/shape.ml:2101` (`broadcast_const`), `:2362` and `:2433`
  (`fold_where_closure`); `test/gen/tinygrad.patch`, which gives tinygrad the
  same.
- **Differs:** a selection whose condition is a constant through movements
  that keep each element's value (reshape, expand, permute, shrink, flip) is
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
  cosine (`by_quadrant`) to broadcast constants in the tensor graph. Two
  symbolic passes over such graphs did not terminate:
  `Prepare.contiguous_view`'s, on rune's staged scan reading a draw on Metal
  (`Rune_internals.Jit › metal › staged scans › stage a scan whose step reads
  a draw and another scan's result made before it`), and
  `Multi.lower_broadcast_copy`'s, on a sharded model's rope (kaun's
  `test_decode_devices`). Folding the selection at its constant also drops
  the branch never taken, such as the long reduction of a bounded angle.
- **Pinned by:** the Symbolic suite: `symbolic_simple › selections › a
  selection by a broadcast constant is the branch it picks` and `› a selection
  by a padded constant is no constant's`, and `symbolic › selections › a
  padded constant condition is not folded in the branches`; and those two
  tests.

## D76. A stack of Invalid lanes keeps its width

- **tinygrad:** `uop/symbolic.py:80` (`pm_data_invalid`'s first rule,
  `invalid_pat.broadcast()`), which folds a stack of `Invalid` lanes to one
  `Invalid`.
- **tolk:** `lib/uop/shape.ml:1990` (`pm_data_invalid`, without that rule);
  `test/gen/tinygrad.patch`, which removes it from tinygrad too.
- **Differs:** a stack of `Invalid` lanes stays a stack. When every upcast or
  unrolled lane of a gated index is `Invalid`, devectorize splits the index
  and its load into lanes, each lane's load folds to its own `0`, and the
  value keeps its width. One `Invalid` in the stack's place drops the width:
  the reshape and permute that arranged the lanes no longer type (tinygrad
  raises `bad reshape: () -> (1, 2)`), a select mixes a scalar with lanes,
  which `Spec.program` refuses (D58), and a reduce's lanes read components
  of a scalar, which the C compiler refuses.
- **Reason:** (b): rune's compiled tangent of a Cholesky factor, once a gather
  lowers to an INDEX: the diagonal's gather fuses with `tril`'s gated loads
  into the triangular solve's kernel, whose upcast lanes all read `Invalid`,
  and compiling it raised `cannot reshape () to (2, 1)`.
- **Pinned by:** the `Symbolic` suite: `invalid values › a stack of invalid
  keeps its width`; the `Codegen` suite: the `invalid_lanes` and
  `invalid_lanes_int8` goldens, recorded from tinygrad with the rule removed;
  `lanes of an unrolled reduce › a reduce whose lanes all read an Invalid
  index sums to zero` and `› a sum of int8 lanes that all read an Invalid
  index is zero`; `lanes all Invalid › a kernel whose upcast lanes all read
  an Invalid index sums to zero`; `vectors in programs › a select of lanes
  whose loads all fold is devectorized`.

## D77. A broadcast value that runs a transcendental stays stored

- **tinygrad:** `schedule/indexing.py:270-276` (`run_rangeify`, which
  stores an elementwise value or reduction whose ranges a broadcast ends),
  `:60` (`BufferizeOpts.removable`) and `:90` (its value), and
  `schedule/rangeify.py:25` (`cleanup_dead_axes`) and `:50-95`
  (`remove_bufferize`, which inlines that store again when its value reads
  at most three buffers and no reduction reads a buffer).
- **tolk:** `lib/schedule/indexing.ml:412` (`assign_ranges`, which records
  the values a broadcast stores) and `:139` (`bufferize_and_index`, which
  marks their stages `Broadcast`); `lib/uop/ops.mli:262` (`keep`, in place of
  `removable`); `lib/schedule/rangeify.ml:114` (`remove_bufferize`, which
  keeps such a stage); `test/gen/tinygrad.patch`, which gives tinygrad the
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
- **Reason:** (b). On the host, where the transcendental functions are
  polynomials, recomputing them dominates. On an M-series Mac under load
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
  at 2048 tokens 1.10 GB and 2.17 GB were allocated.
- **Pinned by:** the Rangeify suite's `swiglu_down`, `attention`,
  `exp_dead_axis` (a kept exponential of a row expanded to a matrix is
  stored as the row), `exp_cheap_consumer` (a value that reads a kept
  exponential is inlined) and `rope_decode` (a decode step's rotated query
  reads its gathered rows of a cosine table and is not stored) rows of
  `kernel_counts.golden` and their kernel goldens, from the equally patched
  tinygrad.

## D79. A chain of one operation keeps its right operand's grouping

- **tinygrad:** `renderer/cstyle.py:67` (`base_rewrite`'s ALU rule), which
  drops the parentheses of any operand that is the same associative operation
  (`ADD`, `MUL`, `XOR`, `OR`, `AND`), on either side.
- **tolk:** `lib/renderer/cstyle.ml:346` (`base_rewrite`'s ALU rule);
  `test/gen/tinygrad.patch`, which gives tinygrad the same.
- **Differs:** only the first operand drops them. C groups `a*b*c` as
  `(a*b)*c`, so writing `a*(b*c)` without parentheses regroups it, and a float
  sum or product rounds and overflows by its grouping. An upcast reduction
  `acc + (v0+v1+v2+v3)` was written `acc+v0+v1+v2+v3` and added in another
  order than the kernel's; `4 * (2^126 * 0.25)` was written `4*2^126*0.25`,
  which overflows.
- **Reason:** (b): nx.quant's product, `Nx_quant.apply`, decodes each weight
  in the kernel that multiplies it, `x * (value * scale)`, and gave infinity
  where the eager product's infinities of both signs give NaN (rune's quant
  suite, `values › compiled, a product is eager's › the largest scales`).
- **Pinned by:** the Cstyle suite: `grouping › Clang computes a product of a
  product as it is grouped`; and the rendered sources of the `codegen`,
  `renderer/cstyle` and `engine/jit` goldens, from the
  equally patched tinygrad.

## D78. The CUDA device's kernels compile to a cubin

- **tinygrad:** `renderer/cstyle.py:409` (`CUDARenderer`'s compiler, which
  makes PTX when the device is `CUDA` and a cubin otherwise).
- **tolk:** `lib/renderer/cstyle.ml:1214` (`cuda`'s compiler).
- **Differs:** the CUDA device's kernels compile to a cubin, as the NV
  device's do. NVRTC is given the GPU's architecture either way, so PTX
  bought no portability: a GPU newer than NVRTC is refused at compile time
  in both forms. The cubin's code is that of NVRTC's assembler rather than
  the driver's. PTX that a disk cache kept from before still loads, since
  the driver takes either form.
- **Reason:** (b). The driver translates PTX at every load, and only its
  own cache, bounded in size and per user, saves that work between
  processes. On an RTX 5000 Ada (driver 615, CUDA 13.4), loading 200
  kernels of 48 unrolled sines each took 137 ms a kernel with the driver's
  cache cold, 0.23 ms warm, and 0.09 ms as cubins, whose compile costs what
  the PTX's does (about 350 ms a kernel) and whose bytes are half the PTX's
  (98 KB against 209 KB).
- **Pinned by:** the Cstyle suite: `CUDA's binaries › the CUDA device's
  kernels compile to a cubin` (slow, skipped without NVRTC).

## D80. A host program runs its output loop in blocks on the host's cores

- **tinygrad:** `runtime/ops_cpu.py:58-72` (`CPUProgram.__call__`), which
  calls a kernel once per launch on the calling thread;
  `codegen/__init__.py:371` (the pipeline after `pm_cast_const`) and `:435`
  (`do_estimates`); `uop/ops.py:1330` (`KernelInfo`) and `:1359`
  (`ProgramInfo.from_sink`); `engine/realize.py:16` (`get_call_var_uops`) and
  `:160` (`exec_kernel`).
- **tolk:** `lib/codegen/codegen.ml:708` (`split_blocks`), `:924` (its place
  in the pipeline) and `:1030` (`whole_loop`); `lib/uop/ops.ml:340`
  (`kernel_info.split`); `lib/uop/call.ml:719` (`program_info_of_sink`);
  `lib/engine/realize.ml:27` (`get_call_var_uops`);
  `engine/tolk_engine.ml` (`block_ops`, `blocks` and `split_grid`);
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
  after them take the slots past every slot a parameter takes, where tinygrad
  numbers them from the count of the numbered parameters
  (`codegen/__init__.py:380`): a kernel whose buffers skip a slot, as a beam
  search's timer links them, would give a variable a bound's slot, and a
  launch would read the bound for it. The kernel's `KernelInfo`
  records `split = (n, lo, hi)`, the iterations and the two slots, by which
  each launch finds the bounds; `n` is its program's first global size, and
  its estimates count the whole loop, exactly when they read no other
  variable: counted in the loop's 32-bit type, a count past 2^31 would wrap. A host launch cuts `[0, n)` into `min(n, ops / 2^18)` blocks, at least
  one, each a group of rig's host of as many threads as iterations, which
  its pool runs on the performance cores, and the entry (D28) runs the
  iterations its threads number; tinygrad's own launch (`whole_loop` in
  `engine/realize.py`) runs one block, the whole loop.
- **Reason:** (b): sofo's and symo's training steps on the CPU, which ran each
  kernel on one core. Their hottest kernels, split by hand into 32 blocks on
  kimchi's 8 E-cores (`taskset -c 6-13`, best of 30), took 6.17 ms and 0.77 ms
  (r_256_8_4_4_32_4), 3.38 and 0.42 (r_32_32_4_4_128_16), 0.28 and 0.04
  (r_256_3_3_100_4) and 17.8 and 2.83 (r_128_64_3_4_400_100_4). tinygrad
  deleted its CPU threading at 4197f7423, after the pin.
- **Pinned by:** the Codegen suite: `host programs in blocks › a host program
  splits the loop each store's address reads`, `› a split loop's estimates
  count its whole loop past 2^31`, `› stores of separate loops
  split none`, `› a store of no loop splits none`, `› a reduction's loop is
  never split`, `› a serial loop is never split`, `› a loop shrunk by its
  guard splits at its shrunk end, unguarded`, `› a split loop of 2^25 iterations keeps 32-bit indices` and `› a split's bounds and a variable take slots of their own past buffers that skip one`; the
  Tolk_engine suite: `host programs in blocks › a program writes the same bits in 1, 3 and 32
  blocks` and `› a run of a large kernel computes its sums`, and `a variable
  named as a block's bounds is a variable like any other`.

## D81. The matrix-vector layout reads its operands through conversions and decoding

- **tinygrad:** `codegen/opt/heuristic.py:61-79` (`hand_coded_optimizations`'
  matrix-vector case), which applies only when the reduced product's two
  operands are loads (`mulop.src[0].op is Ops.INDEX and mulop.src[1].op is
  Ops.INDEX`) and the vector's index has the first reduce range as a term of
  its sum.
- **tolk:** `lib/codegen/opt/heuristic.ml:61` (`term_of`) and `:209`
  (`operands`), which D89 widens; `test/gen/tinygrad.patch`, which gives
  tinygrad the same before the goldens are recorded.
- **Differs:** the vector is a load read through dtype conversions (`CAST`,
  `BITCAST`), and the matrix any computation of loads with no reduce, whose
  ranges then stand for the matrix load's index's. The first reduce range may
  be a term of the vector's index alone or times a constant. A product of two
  loads is chosen as before.
- **Reason:** (b): rune's quantised products (`Nx_quant.apply`), whose matrix
  is MXFP4 codes decoded by bit operations and scaled by their group, and
  whose vector is a bfloat16 activation widened to float32.
  A decoded byte splits the reduce into a range over bytes and one over a
  byte's two values, so the vector reads the byte range times 2. tinygrad
  declines the case, as it does a vector read through a cast alone, and the
  kernel then ran each row's whole reduce in one thread: gpt-oss-20b's expert
  products on CUDA launched 960 threads, the down product taking 1.98 ms per
  layer and gate-up 2.01 ms. On Metal (M1 Max), the down product of four
  experts at gpt-oss's widths takes 0.65 ms where it took 2.2 ms with the
  codes decoded by bit operations, and 0.64 to 0.77 ms where it took 2.8 to
  8.2 ms with the table.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `vecmat_of_cast_*` and `vecmat_decoded_*`,
  recorded from the equally patched tinygrad.

## D82. A sharded value is not stored into a replicated destination

- **tinygrad:** `schedule/multi.py:256` (`store_value_multi`), which stores
  each shard of a sharded value into its own part of an unsharded
  destination.
- **tolk:** `lib/schedule/multi.ml:697` (`store_value_multi`);
  `test/gen/tinygrad.patch`.
- **Differs:** a store of a sharded value into a destination that is whole
  and lives on several devices raises `Invalid_argument`. Each device would
  write only its own part into its own copy, so every copy of a replicated
  value would be partial. tinygrad's Tensor programs never build such a
  store; a destination on one device, or without one, as a kernel's output
  tile is, still takes each shard into its part.
- **Reason:** (b): rune, where nx places a result replicated and the
  lowered value came out sharded, as a gather of split rows by a split index
  did: half of each copy read back as zeros. rune now converts such a value
  before the store; the refusal makes a missed conversion fail where it is
  built.
- **Pinned by:** the `Multi` suite: `multi_pm › stores and calls › a store
  of a sharded value into one replicated on its devices is refused`.

## D83. A sharded value joined on several devices keeps its bits

- **tinygrad:** `schedule/multi.py:228-252` (`copy_multi`), which places each
  shard at its offset in zeros and sums the devices' values.
- **tolk:** `lib/schedule/multi.ml:182` (`joined`) and `:302`, its use in
  `copy_multi`; `:150`, shard selections moved before casts and bitcasts;
  `test/gen/tinygrad.patch`, which gives tinygrad the same.
- **Differs:** the devices' values are joined by an allreduce with a bitwise
  or over each element's bits, as unsigned integers of its width, `uint8`
  for a boolean. Each element is nonzero on at most one device, so the or is
  its exact bits, where the sum turns `-0.` into `+0.` and quiets a
  signalling NaN. A joined index type, which has no width, is refused. A
  shard selection moves before a cast or bitcast as it moves before
  arithmetic, where tinygrad's stops at arithmetic, so the bitcast back is
  read where the selected shard is read: `mesh_to_one` schedules 11
  kernels, as with the sum.
- **Reason:** (b): rune reads a value sharded over devices whole on each, as
  nx places a gather of split rows by a split index, and expects eager's
  bits, `-0.` included: `Jit › gathers across devices`.
- **Pinned by:** the `Multi` suite: `multi_pm › copies › a copy of a sharded
  value to several devices keeps its bits`, and the recorded programs that
  copy a sharded value to several devices.

## D84. Withdrawn

A dispatch's thread-trace marker numbered its queue's dispatches. tolk encodes
no thread trace yet (D146).

## D85. Withdrawn

A GFX11 trace set `TTRACE_EXEC` by name. tolk encodes no thread trace yet
(D146).

## D86. A program reads a lane of a vector at a constant

- **tinygrad:** `uop/spec.py:83` (`shared_spec`, which accepts an index of
  any integer) and `:205` (`spec_program`); `renderer/cstyle.py:168`
  (`render_index`, which writes a lane picked by a value as `(v)[i]`).
- **tolk:** `lib/uop/spec.ml:412` (the second rule of `program`).
- **Differs:** `Spec.program` rejects an index of a vector value
  (`AddrSpace.ALU`) whose lane is not a constant. No pass of tinygrad or
  tolk makes one; a hand-built kernel that does fails at lowering, naming
  the index.
- **Reason:** (b), as D58's. CUDA's vectors are structs whose lanes are
  members, so NVRTC rejects `(v)[i]` ("no operator [] matches these
  operands"); rune compiles its kernels for CUDA.
- **Pinned by:** the Spec suite (`test/uop/spec`): `vectors in programs ›
  a program reads a lane of a vector at a constant`; the Cstyle suite's
  `every GPU kernel compiles with its target's toolchain › cuda_dynamic_lane`,
  tinygrad's source, which NVRTC rejects (slow, an expected failure).

## D87. A store through a gather of a sharded value stores each shard's rows

- **tinygrad:** `schedule/multi.py:291` (`index_multi`, for every `INDEX` of
  a sharded value); a store reads its destination's `INDEX` as any other, so
  a store through a gather of a sharded value has no rule of its own.
- **tolk:** `lib/schedule/multi.ml:617` (`scatter_dests`, run as the `bpm` of
  the rewrite in `lib/schedule/prepare.ml:631`), `:637` (`whole_index`),
  `:650` (`scatter_shards`), `:683` (`store_scattered`) and `:778`.
- **Differs:** a gather that a store writes through, of a sharded value, is
  marked before the rewrite reaches it and becomes a scatter: its index is
  whole on each device, a sharded one joined with its validity beside it. Of
  a value sharded on trailing axes, each shard stores its part; of a value
  sharded on its rows, each shard stores the rows it holds at their index
  within it, and every other row's index is Invalid, which drops its store.
  The stored value is whole on each device. Read as a gather instead, each
  device stored into a joined copy of the rows.
- **Reason:** (b): rune writes a lent value at loaded indices with one indexed
  store (`Lower_index.scatter`'s region), a key-value cache's rows among them,
  and a cache split over devices along its rows: `Jit › a lent write of rows ›
  a pool split along the written axis is written whole`.
- **Pinned by:** the `Multi` suite: `multi_pm › scatters › *`.

## D88. On arm64 a float zero a comparison reads reaches C as a value the backend cannot see

- **tinygrad:** `renderer/cstyle.py:264` (`ClangRenderer`, whose
  `string_rewrite` writes a float zero as a literal, `0.0f`).
- **tolk:** `lib/renderer/cstyle.ml:843` (`float_zero`), `:863`
  (`opaque_zero`) and `:912` (`clang`, which adds it for an arm64 target);
  `test/gen/tinygrad.patch`, which gives tinygrad's `ClangRenderer` the same
  rule.
- **Differs:** for an arm64 target, a float zero constant, at float16,
  bfloat16, float32 or float64, that a comparison reads renders in the
  comparison as an empty `asm` statement over a register holding it,
  `({float z = 0.0f; __asm__("" : "+w"(z)); z;})`, cast to its dtype. Every
  other use of the zero, and every zero on x86_64, keeps the literal.
- **Reason:** (b): clang's AArch64 backend lowers a select of a value and a
  zero literal, by a comparison of the two, to `fminnm` or `fmaxnm`, which
  keep the value's zero where the select returns the literal's: `(v < 0.0f) ?
  v : 0.0f` is `-0.0` at `v = -0.0` from `-O1` up, in Homebrew clang 22.1.7
  and Apple clang 17. A maximum against a zero, rendered as such a select,
  meets it too, and an add of `+0.0` after it is then dropped. Under
  `Rune.jit`, `Nx.where (Nx.less x z) x z` gave `-0.` where eager gives `0.`,
  and a sum of a minimum against a folded zero gave `-0.`, which a float sum
  never is. The lowering needs the select's operands to be the comparison's,
  so a zero hidden in the comparison alone avoids it. Hidden everywhere, the
  zero also kept the backend from making `c ? v : 0.0f`, a select by any
  other condition, a mask: a routed MXFP4 product on the host, whose rows are
  selected by their validity before the product, ran 1.17 times as long as
  the same product without the select, and runs 1.05 times as long with the
  literal there (single-threaded kernel, paired). `-ffp-exception-behavior=maytrap`
  or `-frounding-math` also avoid the lowering, at 2.7× to 4.7× the time of a
  matmul or an element-wise kernel.
- **Pinned by:** the Cstyle suite (`test/renderer/cstyle`): `zeros on arm64 ›
  a float zero a comparison reads is opaque to the backend on arm64, and a
  literal on x86_64`, `› a float zero that no comparison reads is a literal
  on arm64` and `› a select of a value and a zero picks the zero at -0., on
  the host`; rune's Jit programs suite: `a compiled program › selects a zero
  as eagerly, whatever the other value's sign` and the rounded law's example.

## D89. A matrix-vector product is laid out by its matrix

- **tinygrad:** `codegen/opt/heuristic.py:61-79` (`hand_coded_optimizations`'
  matrix-vector case: the first reduce axis split into `MV_THREADS_PER_ROW`
  local threads, 8, the first global axis that 16 divides into
  `MV_BLOCKSIZE` local threads, 4, then `MV_ROWS_PER_THREAD` upcast lanes,
  4, whatever the matrix's layout and size), and `codegen/opt/postrange.py:132-134`
  (`apply_opt`, which refuses a local split of a reduce axis inside another
  reduce).
- **tolk:** `lib/codegen/opt/heuristic.ml:167-184` (`lanes`, `rows`,
  `columns`, `busy`, `in_flight`), `:209` (`operands`), `:224` (`units`) and
  `:232` (`matvec`); `lib/codegen/opt/postrange.ml:307-316`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** the vector is any computation of accesses with no reduce,
  either operand of the product, in fewer ranges than the matrix. The range of
  unit stride in an access of the matrix that reads along the reduce chooses
  the layout. Along the reduce (`W[n, k]`): the reduce splits into up to 32
  local threads, adjacent threads reading adjacent elements, 4 rows a
  workgroup, and what is left of the reduce is unrolled by its largest divisor
  up to 8. Along an output (`W[k, n]`): that axis splits into 32 local threads
  and 2 upcast lanes, and the reduce into the threads, up to 32, that make
  32768 threads; with more than 32768 outputs the kernel takes the other
  rules. `MV_BLOCKSIZE`, `MV_THREADS_PER_ROW` and `MV_ROWS_PER_THREAD` are
  gone; `MV=0` stays. A reduce axis inside another reduce splits into local
  threads, each iteration of the outer reduce storing and reading the shared
  buffer between barriers; inside an unrolled reduce it is still refused.
- **Reason:** (b): gpt-oss-20b's decode on CUDA (RTX 5000 Ada, 576 GB/s).
  tinygrad's layout puts 8 threads on a row whatever the matrix: rows along
  the reduce read one byte a thread, columns along an output read 8 rows apart,
  and few outputs leave the GPU short of threads. The query projection
  multiplies a normalised activation, which tinygrad declines, and ran 1024
  threads; the experts' down product sums four experts, a reduce around the
  matrix-vector one, which could not split, and ran 720 threads reading
  11520 bytes each. Per decode step, from 45.5 ms to 11.1 ms: the down
  products from 15.72 ms to 0.91 ms (5% to 81% of the bandwidth), gate-up
  from 15.38 ms to 2.60 ms (10% to 57%), the attention projections from
  8.56 ms to 2.17 ms, the output projection from 1.57 ms to 1.12 ms (63% to
  88%); the 201088 outputs of the head keep their layout.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `matvec_*`, `vecmat_*`, `experts_down_*`
  and `gpt_oss_*`, recorded from the equally patched tinygrad, and `the
  hand-coded optimisations keep a kernel's writes › small kernels ›
  experts_down_metal` (a group inside another reduce); the Postrange suite's
  cases `double_sum_group` and `double_sum_group_twice`; the tolk bench's
  `cpu/` and `cuda/` rows.

## D90. A linked schedule resolves its launches at link

- **tinygrad:** `engine/realize.py:132` (`resolve_params`) and `:160`
  (`exec_kernel`), which find a call's buffers, launch dimensions and variable
  values on every run, and `uop/ops.py:1177` (`UOp.sym_infer`), which
  substitutes a symbolic integer's variables and simplifies it on each
  evaluation.
- **tolk:** `engine/tolk_engine.ml` (`affine`, `value`, `locate` and
  `launch`); `lib/uop/shape.ml:1230` (`sym_compile`).
- **Differs:** linking a schedule turns each call into a launch per lane of
  rig's program (D146), whose buffers, offsets, scalars and grid are words of
  the run's ints or constants: each variable of the run is a word, a value
  computed from them a word the run computes once
  (`Shape.sym_compile`, which simplifies once and computes on `int`s, exactly
  where a value does not fit), and a value that moves with a loop's trips a
  hole of rig's program, the sum of trip words times constant factors. A
  `DEBUG` report of a call lists the run's variables that the schedule reads,
  where it listed every variable of the run.
- **Reason:** (b): symo's tutorial step on the CPU runs 2,273 host launches,
  mostly inside the ranges of its staged scans, and sofo's steps do likewise.
  Resolving a launch on each run cost about 5 us on an M1 Max, for kernels
  of 0.3 to 40 us: a jitted scan of 256 small steps (1,024 launches) took
  5.13 ms and took 0.27 ms resolved at link.
- **Pinned by:** the Tolk_engine suite (`test/engine/tolk_engine`): `link and
  run › a linked scan carries its storage across runs on each run's
  parameters`, `› a scan runs its body once per trip, carrying in place`, `› a
  schedule runs with each binding of its variables` and `runs › runs of one
  schedule from two domains each compute their own`; the Ops suite
  (`test/uop/ops`): `sym_compile › computes what sym_infer does, variables
  within 50`, `› within 2147483648`, `› at the int extremes`, `› computes
  exactly where int arithmetic overflows` and `› raises as sym_infer does
  where a value fits no int`.

## D91. The host's upcast lanes stay within 32

- **tinygrad:** `codegen/opt/heuristic.py:115-120` (`hand_coded_optimizations`:
  more upcasts while `k.upcast_size() < 32`, each by 3 or 4, so the last one
  can take a kernel to 64 lanes or more).
- **tolk:** `lib/codegen/opt/heuristic.ml:431-435` (`host_lanes`,
  `beyond_host_lanes`), `:463` (`upcast_more`).
- **Differs:** on the host (`target.device = "CPU"`), the heuristic does not
  take an upcast that would make the kernel's upcast and unrolled lanes more
  than 32. Other devices upcast as tinygrad does.
- **Reason:** (b), measured. Each lane holds a value across the kernel's
  loops, and past 32 they spill out of the host's registers. lorenz_simple's
  (sofo-raven) tangent kernel `r_32_256_400_4_2_4_2` takes 2.12 ms as
  tinygrad's heuristic optimises it (64 lanes) and 1.51 ms within 32 (six
  P-cores of an Intel Core Ultra 5 235, the tolk bench's `lorenz` cases).
- **Pinned by:** `Tolk.Heuristic`'s `paired_products` cases (on `cpu` two
  upcasts by 4 and D108's by 2, on `metal`, `cuda` and `amd` three by 4); the
  goldens, from tinygrad with D91 applied by its generator.

## D92. A decoded operand shared by a run of an output axis is decoded once a run

- **tinygrad:** `codegen/opt/heuristic.py:112-138` (`hand_coded_optimizations`'
  upcasts of output axes, which find reuse only on an axis that some access
  does not read, take 3 or 4 values of it, and come after the matrix-vector
  layout of `:61-79` has returned).
- **tolk:** `lib/codegen/opt/heuristic.ml:322` (`run_cap`), `:326` (`run`),
  `:75` (`decoded`), `:350` (`upcast_shared`) and `:608` (its place, before
  `matvec`); `test/gen/tinygrad.patch`, which gives tinygrad the same before
  the goldens are recorded.
- **Differs:** before the matrix-vector layout, an output axis that an operand
  of a summed product reads only as `r / d` is upcast by the largest amount up
  to 4 that divides both `d` and the axis, when that operand is decoded: its
  value converts integers it reads into floats, by a cast or a bitcast,
  outside any access's address. The axis's lanes then share the operand's
  value. With several such axes, each is upcast.
- **Reason:** (b): `Nx.map_segments` over `Nx_quant.apply` on a prompt, which
  sorts positions by expert into blocks of 2 rows that read their block's
  matrix, `owner[row / 2]`. Every row decoded the matrix's MXFP4 codes and
  scales again: the row axis reaches the matrix through the gather, so tinygrad
  sees no reuse. Upcast by 2, a block's rows decode once. The 512-token gate and
  up product's kernel on an RTX 5000 Ada, from 42.5 ms to 26.2 ms
  (`r_520_5760_32_4_9_2_5` to `r_260_5760_32_4_2_9_2_5`), against 34.4 ms for
  one matrix per position; on the Mac's Metal, from 275.9 ms to 225.5 ms,
  against 258.5 ms. gpt-oss-20b's 512-token prefill on CUDA, from 1.49 s to
  1.11 s. An operand of floats is left alone: upcasting the query heads that
  share a key and value head (`h / 8`), whose cache a select joins to the new
  position's, made gpt-oss-20b's decode on CUDA 3.4% slower, as the lanes'
  shared load is one the cache serves and the upcast costs threads.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `routed_blocks_*` and `shared_keys_*`
  (keys of floats, left alone), recorded from the equally patched
  tinygrad, and `the hand-coded optimisations keep a kernel's writes › large
  kernels › routed_blocks_*` (slow).

## D93. The where-closure rule searches for its condition

- **tinygrad:** `uop/symbolic.py:228-230` (`fold_where_closure`, which asks
  whether the condition is in `t.bool_slice` or `f.bool_slice`) and
  `uop/ops.py:289-293` (`_bool_slice`, a `recursive_property`: each node
  keeps the set of the boolean nodes it reaches).
- **tolk:** `lib/uop/shape.ml:2433` (`fold_where_closure`) and
  `lib/uop/ops.ml:1082` (`reaches`).
- **Differs:** the rule asks whether `t` or `f` reaches the condition, a
  scalar boolean, with `Ops.reaches`: a walk from the branch that enters no
  node built before the condition, since a node is built after its sources.
  `Ops.bool_slice` is gone. The search runs last, after the INDEX gate,
  which reads a property of each node (D94), where tinygrad's lookup runs
  first: a branch that reaches an INDEX is rejected without a walk. The rule
  folds the same selections.
- **Reason:** (b): rune lowers `Nx.sin` to a reduction full of comparisons
  and selections before tolk sees it, so in a jitted chain of `Nx.sin` each
  node's boolean set holds every comparison of the links before it, and the
  sets grow with the square of the chain. Compiling a chain of 80 `Nx.sin`
  cold on an M1 Max took 63.0 s and a heap of 618M words with the property,
  and takes 16.6 s and 76M words with the search, whose walks stay inside
  the link that built the condition. A condition built before a long chain
  of selections still makes each search walk the chain below it, which the
  INDEX gate spares when the branches read memory: the jitted training step
  of a model of 2031 kernels spent 3 s of its 10 s schedule in the search.
- **Pinned by:** the Symbolic suite (`test/uop/symbolic`): `cost › sym's work
  on a chain of selections is linear in its length`, `› sym's work on a chain
  under one far condition is linear in its length` and tinygrad's
  `test_where_closure_folding*` goldens; the Ops suite (`test/uop/ops`):
  `graphs › reaches is membership in the node's toposort` and `› reaches
  ~calls:Enter enters call bodies`; the rune bench's `Jit/jit-run-chain`.

## D94. A node keeps the operations it reaches

- **tinygrad:** `uop/ops.py:277-287` (`backward_slice`, a `cached_property`
  holding the whole slice of each node asked, and
  `op_in_backward_slice_with_self`, which walks it).
- **tolk:** `lib/uop/ops.ml:1065` (`ops_reached`) and `:1076`
  (`op_in_backward_slice_with_self`).
- **Differs:** each node keeps, as a property filled from its sources, the
  set of the operations of itself and of the nodes it reaches outside call
  bodies; a node whose set is a source's shares it.
  `op_in_backward_slice_with_self` reads that set and no longer builds the
  node's `backward_slice`. Answers are the same.
- **Reason:** (b): the where-closure rule asks whether the condition and both
  branches reach an `Index`, which builds and keeps the slice of every node it
  asks about. In a jitted chain of `Nx.sin` every link's selections are
  asked, each slice holds the whole chain below, and the work and memory grow
  with the square of the chain. Compiling a chain of 80 `Nx.sin` cold on an M1
  Max took 28.0 s and a heap of 218M words, and takes 16.4 s and 73M words.
- **Pinned by:** the Symbolic suite: `cost › sym's work on a chain of
  selections is linear in its length`; the Ops suite: `graphs ›
  op_in_backward_slice_with_self is an operation of the slice` and `›
  op_in_backward_slice_with_self enters call bodies only under Enter`; the
  rune bench's
  `Jit/jit-run-chain`.

## D95. A device's target follows from the device alone

- **tinygrad:** `helpers.py:221-236` (`_DEV`, the `DEV` setting, and
  `DEV.target`); `device.py:483-486` (`_select_renderer`, which renders for
  `DEV.target` of the device's kind).
- **tolk:** `engine/tolk_engine.ml` (`target`); `lib/device.ml:144`
  (`renderer`); `lib/helpers.ml:25` (`Target`).
- **Differs:** there is no `DEV` setting. `Tolk_engine.target` is a function
  of the device: its vendor's kind, which its driver declares
  (`Rig.capability`), and its `arch`, or the host's CPU target.
  `Device.renderer` takes the target and renders for it, the first renderer
  of its kind that suits it when it names none. `Helpers.Target` keeps the
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

## D96. A matrix-vector workgroup's rows are the matrix's own

- **tinygrad:** `codegen/opt/heuristic.py:61-79` (`hand_coded_optimizations`'
  matrix-vector case, which splits `MV_BLOCKSIZE` local threads from the
  first global axis that divides, whichever operand reads it).
- **tolk:** `lib/codegen/opt/heuristic.ml:209` (`operands`) and `:244`
  (`rows_layout`); `test/gen/tinygrad.patch`, which gives tinygrad the same
  before the goldens are recorded.
- **Differs:** D89's rows layout splits its 4 SIMD groups from the first
  global axis that 4 divides among those the vector does not read, and from
  one the vector reads only when none of the others divides.
- **Reason:** (b): `Nx.map_segments` over `Nx_quant.apply` on a prompt, whose
  blocks of rows each multiply their expert's matrix. The vector, a block's
  rows, reads the block axis, so the 4 groups of a workgroup took 4 blocks, each
  reading its own rows for every output. In blocks of 4, D92's upcast, a
  workgroup reads 16 rows of 2880 floats, 184 KB, more than an SM's L1 holds on
  an RTX 5000 Ada; in blocks of 2, half that. On the matrix's rows the 4 groups
  read one block's rows. gpt-oss-20b's 512-token gate and up product on that
  GPU, in blocks of 4, from 56.6 ms to 21.9 ms, and in blocks of 2 from 26.0 ms
  to 25.3 ms, with the same registers (63 and 61, no spills); on an M1 Max's
  Metal from 224 ms to 162 ms and from 225 ms to 167 ms.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `routed_blocks_metal`,
  `routed_blocks_cuda` and `routed_blocks_amd` (the local split on the
  matrix's row axis), recorded from the equally patched tinygrad.

## D100. A reduction of no axes that a broadcast ends is stored

- **tinygrad:** `schedule/indexing.py:271-276` (`run_rangeify`, which stores
  an elementwise value or reduction whose ranges a broadcast ends by
  realizing each of its axes, so a value of no axes is not stored).
- **tolk:** `lib/schedule/indexing.ml:478` (`assign_ranges`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the
  goldens are recorded.
- **Differs:** a value of no axes whose consumers broadcast it, and whose
  computation holds a reduction, is stored as one element, as a value with
  axes is. The cost check (`remove_bufferize`) then inlines it again where
  no reduction of it reads a buffer and it reads at most three. A value of
  no axes without a reduction, such as a shard's offset, stays inlined.
  tinygrad computes such a reduction inside its consumer, once for each
  element: `x - x.sum()` sums the whole of `x` for each element.
- **Reason:** (b): `Vega.Loss_scale.step`, whose `finite`, the conjunction
  of `Nx.all (Nx.isfinite g)` over every gradient, selects each update.
  In GPT-2 124M's float16 step on an M1 Max's Metal, five kernels each
  reduced all 148 gradients again: the scale and its counters, and the
  embedding updates, about 14 ms a step. Stored, one kernel reduces them.
- **Pinned by:** the Rangeify suite's `sum_all_broadcast` and
  `finite_checks` rows of `kernel_counts.golden` and their kernel goldens,
  and its `reduce_expand_child` and `preserve_multistage_reduce` rows; the
  Jit suite's `nonzero` and `masked_select` goldens; rune's `lower_linalg`
  goldens `qr_q` and `qr_r`, whose reflections' norms are stored; all from
  the equally patched tinygrad.

## D101. A group shares its threads with the kernel's independent reductions

- **tinygrad:** `codegen/opt/postrange.py:123-135` (`apply_opt`: a local
  split of a reduce axis splits that axis alone, and sizes the shared
  memory by the first reduction's type).
- **tolk:** `lib/codegen/opt/postrange.ml:203` (`siblings`), `:171`
  (`smem`) and `:387` (its use in `apply`); `test/gen/tinygrad.patch`,
  which gives tinygrad the same before the goldens are recorded.
- **Differs:** a local split of an axis of reductions that no reduce axis
  encloses also splits, by the same threads, one axis of each other such
  reduction: its first reduce axis that the amount divides, when only such
  reductions close it, when it neither reads a grouped reduction nor is
  read by one, and while every grouped reduction's buffer fits in shared
  memory. The shared memory checked is the sum of the grouped reductions'
  buffers. A reduction that reads another stays out, since the final
  reductions of shared buffers that one thread range closes run as one
  loop.
- **Reason:** (b): the kernel of D100's `finite`. Grouped on its first
  reduction, each of its 16 threads ran the other 147 whole. In GPT-2
  124M's float16 step on an M1 Max's Metal the kernel took 3.1 ms and takes
  0.77 ms; 200 checks of float16 vectors, from 3.6 ms to 0.90 ms.
- **Pinned by:** the Postrange suite's cases `sibling_sums_group` (a
  reduction the amount does not divide stays out), `ten_sums_group` (eight
  of ten buffers fit) and `single_kernel_softmax_group` (the sum that reads
  a maximum stays out); the Codegen suite's `two_grouped_stores_local`
  goldens and its claim that a barrier follows each local store; all from
  the equally patched tinygrad. The rune bench's `finite/` and `metal/`
  rows.

## D102. An empty argument of a precompiled call is its constant

- **tinygrad:** `schedule/prepare.py:264` (the size-0 rule, which makes every
  value with no element a constant expanded to its shape), `schedule/rangeify.py:143`
  (`no_indexing_calls`), `uop/spec.py:254-280` (`spec_kernel_graph`, which
  admits no expand) and `schedule/__init__.py:19-23,72-75` (`_states` and the
  call's arguments in `create_schedule`, which take buffer states only).
- **tolk:** `lib/schedule/rangeify.ml:227` (`no_indexing_calls`) and
  `lib/schedule/schedule.ml:55` (`empty_argument`) and `:74`
  (`create_schedule`).
- **Differs:** an argument with no element of a precompiled call, which the
  size-0 rule made an expanded constant, reaches the kernel graph as the
  scalar constant, without its expand. `create_schedule` reads no state of it
  and passes it as the call's argument. The body, scheduled on its own, makes
  the empty parameter a constant too, so none of its kernels binds the
  argument. A kernel's argument is still a buffer state. tinygrad fails the
  kernel graph's verification on the expand.
- **Reason:** (b): `Rune.vmap (Rune.grad (Rune.jit scan))` over no lane. The
  staged scan's backward step is a precompiled call in a loop whose carry has
  no element and whose rows have some.
- **Pinned by:** the `Tolk_engine` suite: `link and run › a scan whose carry
  has no element runs on its rows`; rune's `test_jit_transformations`: `jit
  is the identity under a transformation › a scan`, whose mapped shapes hold
  no lane.

## D103. An integer constant is written as the value its type holds

- **tinygrad:** `renderer/cstyle.py:26-35` (the constant rules: an `int64` is
  its value with `l`, a `uint32` or a `uint64` its truncated value with `u` or
  `ul`, an 8- or 16-bit integer a cast of its value, any other its value).
- **tolk:** `lib/renderer/cstyle.ml:197` (`int_const`); `test/gen/tinygrad.patch`,
  which gives tinygrad the same before the goldens are recorded.
- **Differs:** a constant of an integer type is written as the value the type
  holds: the integer part of a float, wrapped to the type's width, with the
  suffix or cast of its type. The least `int64` is written
  `(-9223372036854775807l-1)`. An infinity or a NaN, which has no integer
  value, is a conversion of the float as the program runs. tinygrad writes a
  signed or 8- or 16-bit constant unwrapped, so a value past 64 bits is no C
  literal, and the least `int64` is an unsigned literal negated.
- **Reason:** (b): rune's `test_jit_programs` property, where a float padding
  constant converted to an integer was written as a literal past 64 bits, which
  C refuses.
- **Pinned by:** the Cstyle suite: `execution on the host › stores each
  integer constant as its type holds it` and `› converts an infinity or a NaN
  to an integer type`; `clang_transcendental_half.golden`, recorded from the
  equally patched tinygrad.

## D104. A value read widened from a narrower type is stored narrow

- **tinygrad:** `schedule/rangeify.py:208-210` (`bufferize_to_store`, which
  stores a stage's value at its committed dtype).
- **tolk:** `lib/schedule/rangeify.ml:348` (`widens`), `:358`
  (`bufferize_to_store`) and `:469` (the index of a narrow store through its
  widening); `test/gen/tinygrad.patch`, which gives tinygrad the same before
  the goldens are recorded.
- **Differs:** a stage that is not stored whole, whose value is a cast to a
  wider type that holds every value of its source (`Dtype.can_lossless_cast`,
  `bool` and the weak types excepted), stores the source at its own type, and
  each index of the buffer widens the value it reads. Where the stage is
  placed and how it is laid out do not change.
- **Reason:** (b): rune widens a product's `bfloat16` operands to `float32`
  before multiplying them (D29), and a stage sits where its consumers end
  their ranges, after the widening. gpt-oss-20b's attention probabilities, its
  attention output and the rows of its experts were stored as `float32`
  values of `bfloat16`: twice the bytes, and a `float32` load, which no CUDA
  tensor core multiplies. Stored narrow, the output projection of a 512-token
  prefill on an RTX 5000 Ada runs on the tensor cores in 0.30 ms where it took
  2.34 ms, and the scores in 0.29 ms where they took 0.87 ms.
- **Pinned by:** the Heuristic suite's `projection_of_stored` kernel, whose
  golden reads the stored attention output as `bfloat16`, and its cases
  `projection_of_stored_{metal,cuda,amd}` (the tensor cores apply), recorded
  from the equally patched tinygrad.

## D105. A tensor core takes rows that share the other operand's tile

- **tinygrad:** `codegen/opt/postrange.py:187-197` (`_apply_tc_opt`: M runs
  along a range of `in0` that `in1` does not read, N along one of `in1` that
  `in0` does not read), `:222` (the core multiplies the multiply's own
  operands), and `codegen/opt/heuristic.py:28` (`hand_coded_optimizations`
  tries the cores on a kernel of one reduce axis only, below `TC_OPT=1`).
- **tolk:** `lib/codegen/opt/postrange.ml:351` (`in_tiles`), `:559` (`roles`)
  and `:670` (`own`); `lib/codegen/opt/heuristic.ml:86` (`decoded_product`)
  and `:98` (`tensor_cores`); `test/gen/tinygrad.patch`, which gives tinygrad
  the same before the goldens are recorded.
- **Differs:** a range both operands read is the core's M (or N) range of one
  of them when the other reads it only as `r / d`, `d` a multiple of the
  core's M (or N): that operand reads one value over each tile of the core,
  and the core reads its bits of the range at 0. Either operand may be the
  core's A: the choices with `in0` as A come first, as tinygrad orders them,
  then those with `in1`. A summed product with a decoded operand (D92's
  `decoded`) tries the cores whatever its number of reduce axes.
- **Reason:** (b): `Nx.map_segments` over `Nx_quant.apply` on a prompt, which
  sorts positions by expert into blocks of up to 16 rows, each multiplying its
  expert's matrix, `W[owner[row / 16]]`. The rows read the row axis and the
  matrix reads it by blocks, so tinygrad finds no M range; the decoded bytes
  split the reduce into bytes and a byte's two values (D81), which tinygrad's
  heuristic leaves to the hand-coded path. A block of 8, which fewer positions
  take, fits CUDA's N of 8 and not its M of 16, so its rows are the core's B.
  gpt-oss-20b's 512-token gate and up product on an RTX 5000 Ada takes 2.9 ms on
  the tensor cores, from 13.3 ms, and the down product 1.5 ms, from 6.9 ms; the
  prefill takes 0.22 s, from 0.71 s.
- **Pinned by:** the Heuristic suite's cases `routed_tiles_{metal,cuda,amd}`
  (the cores apply to a product in blocks of 16 rows), recorded from the
  equally patched tinygrad; the Postrange suite's `tc_operands_swapped`
  (the second choice of a matmul makes its `in1` the core's A); rune's
  `Rune.quant › gpus › CUDA › on CUDA, a prompt's routes in blocks are
  eager's`, blocks of 16 and of 8 (slow, on the GPU).

## D106. A value every output of a reduction reads stays stored

- **tinygrad:** `schedule/rangeify.py:50-102` (`remove_bufferize`, which
  inlines a stored value that reads at most three buffers and no reduction
  reading a buffer, whoever reads it) and `schedule/indexing.py:267`
  (`run_rangeify`, which stores a value its consumers read at different
  indices without marking it broadcast).
- **tolk:** `lib/schedule/rangeify.ml:119` (`read_by_every_output`, in
  `remove_bufferize`) and `lib/schedule/indexing.ml:463` (`assign_ranges`,
  which marks such a value `Broadcast` when its consumers broadcast it);
  `test/gen/tinygrad.patch`, which gives tinygrad the same rule.
- **Differs:** a stored value that is broadcast, read only through the
  reduce ranges of its consumer, and reads more than one buffer stays
  stored. Its consumer is a reduction that every output of reads it, as a
  matrix-vector product reads its vector, so inlined it would be computed,
  and each of its buffers loaded, once per output. A value read through an
  output range, as a matrix's rows are, is inlined as before: a tiled
  product computes it once for the outputs a thread holds. A value stored
  for consumers that read it at different indices, as the query, key and
  value projections read one normalised activation, is marked broadcast
  when they broadcast it, as D77 marks a value that a broadcast ends.
- **Reason:** (b): gpt-oss-20b's decode on CUDA (RTX 5000 Ada). The
  normalised activation, the residual divided by its root mean square and
  scaled by its gain, reads three buffers and divides, and every projection
  of a decode step computed it again for each of its outputs: the router's
  32, the key and value projections' 512, the experts' 23040 gate and up
  rows. Stored, one 2880-element kernel per normalisation (1.2 us), the
  router went from 37.1 us to 6.8 us, the key and value projections from
  22.1 us to 9.4 us, the gate and up product from 113.9 us to 73.1 us (54%
  to 84% of the bandwidth), and the step's kernels from 11.94 ms to 9.73 ms.
  The 512-token prefill keeps its kernels and its 0.71 s.
- **Pinned by:** the Rangeify suite's `normed_vecmat` (a normalised vector
  read by a product), `normed_vecmats` (by two products, stored once) and
  `normed_matmul` (a normalised matrix, inlined) rows of
  `kernel_counts.golden` and their kernel goldens, and `variable_read_twice`
  (a value two full reductions read, inlined), from the equally patched
  tinygrad.

## D107. Withdrawn

A batch submitted a streamed queue after the queues it waited for. rig orders
submissions across queues (D146).

## D108. On the host, a load a reduce reads fills the lanes by 2

- **tinygrad:** `codegen/opt/heuristic.py:115-138` (`hand_coded_optimizations`'
  upcasts of output axes: by 3 or 4, of an axis that some access does not
  read while it reads every upcast axis, until the kernel has 32 lanes).
- **tolk:** `lib/codegen/opt/heuristic.ml:59` (`reads`), `:444` (`choice
  ~fill`) and `:499` (the fill); `test/gen/tinygrad.patch`, which gives
  tinygrad the same before the goldens are recorded.
- **Differs:** on the host (`target.device = "CPU"`), when no upcast by 3 or 4
  is left, an output axis not yet upcast is upcast by 2 if the kernel's lanes
  stay within D91's 32 and some access that a reduce reads does not read the
  axis. A range counts as read by an index that is the range itself. Among
  such axes the order is tinygrad's: fewest accesses reading the axis, then
  the least sum of their strides.
- **Reason:** (b): lorenz_simple's step (sofo-raven), measured on kimchi's
  P-cores (cores 0-5 of an Intel Core Ultra 5 235). Its kernel of four sums
  into a `[128 tangents, 256, 3]` output stopped at 12 lanes: once its axis of
  3 is upcast whole, no access reads every upcast axis and not another, so
  tinygrad finds no axis to upcast next. Upcast by 2 along the
  tangents, whose lanes then share each load of the hidden activations and
  the weights in the 400-long sums, it takes 1.36 ms instead of 1.84 ms
  (`r_128_64_3_4_3_3_400_100_4` to `r_64_64_3_4_2_3_3_400_400`); a beam search
  of width 2 finds 1.36 ms with 48 lanes. The step's other tangent kernel
  fills its 32 lanes the same way (`r_32_128_100_4_4_2_4_4`, D109). Of the
  recorded kernels, `conv` on the host goes from 35 us to 27 us and
  `paired_products` from 21 us to 19 us.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `tangent_products_cpu`, `conv_cpu`,
  `conv_half_cpu` and `paired_products_cpu`, recorded from the equally
  patched tinygrad; the tolk bench's `lorenz` rows.

## D109. On the host, a kernel of several reduces unrolls within 32 lanes

- **tinygrad:** `codegen/opt/heuristic.py:140-154` (the unroll of the last
  reduce axis: whole when it is at most 32, by 4 otherwise, while the kernel
  has fewer than 64 lanes).
- **tolk:** `lib/codegen/opt/heuristic.ml:517` (`unroll`, its `fits`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** on the host, a kernel with more than one reduce does not take
  an unroll that would make its upcast and unrolled lanes more than D91's 32.
  A kernel of one reduce unrolls as tinygrad's does.
- **Reason:** (b): lorenz_simple's step, measured as in D108. Its hidden
  layer's tangent, two sums of 4 into a `[128, 256, 400]` output, had 16
  upcast lanes and tinygrad's whole unroll of one of the sums took it to 64:
  1.42 ms (`r_32_256_100_4_4_4_4`). The unrolled sum folds into the store's
  code for each lane, which clang leaves scalar. Within 32 lanes it takes
  0.79 ms with no unroll, and 0.85 ms with D108's upcast by 2 that the
  heuristic now takes (`r_32_128_100_4_4_2_4_4`); a beam search finds 0.72
  ms. The four-sum kernel of D108, unrolled by 4 at its 24 lanes, takes 1.52
  ms instead of 1.36 ms. Kernels of one reduce keep tinygrad's unroll past 32
  lanes, which they need: `max_rows` on the host takes 2.1 us unrolled to 64
  lanes and 6.2 us within 32, `matmul` 14.8 us and 19.1 us, and lorenz_simple's
  `r_64_100_4_4_4` 2.0 ms and 3.0 ms a step. With D108, the step goes from
  117 ms to 83 ms, against 80 ms with the beam search, which takes 65 s cold.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, case `tangent_products_cpu` (no unroll; on
  `metal`, `cuda` and `amd` the sum is unrolled), recorded from the equally
  patched tinygrad; the tolk bench's `lorenz` rows.

## D110. A matrix-vector product with few outputs takes fewer threads across them

- **tinygrad:** `codegen/opt/heuristic.py:61-79` (`hand_coded_optimizations`'
  matrix-vector case, `MV_BLOCKSIZE` local threads on the output, whatever
  its size); D89 replaces it.
- **tolk:** `lib/codegen/opt/heuristic.ml:177` (`sector_lanes`) and `:274`
  (`columns_layout`'s `across`); `test/gen/tinygrad.patch`, which gives
  tinygrad the same before the goldens are recorded.
- **Differs:** D89's columns layout puts 32 threads across the outputs and
  up to 32 along the reduce. When the outputs are too few for that to run
  32768 threads, a workgroup of 1024 threads takes fewer across, down to 8,
  and more along the reduce, as many as the reduce's power-of-two divisor
  allows, so that more workgroups share the outputs. 8 threads of 2
  bfloat16 columns read a 32-byte sector. A product of 32 outputs, which
  took D89's 64 at least, now takes this layout too.
- **Reason:** (b): gpt-oss-20b's decode on CUDA (RTX 5000 Ada, 100
  multiprocessors). The key and value projections, 512 outputs, ran on 8
  workgroups and took 9.05 us each; on 32 they take 7.8 us. The router, 32
  outputs, fell back to a group of 16 threads per output and took 6.7 us; it
  takes 3.7 us.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `gpt_oss_kv_*`, `gpt_oss_router_*`,
  `vecmat_*` and `shared_keys_*`, recorded from the equally patched
  tinygrad; rune's compiled bench's `cuda/decode-kv-2880x512-bfloat16` and
  `cuda/decode-router-2880x32-bfloat16`.
## D111. A hung AM device is lost, never recovered in the process

- **tinygrad:** `runtime/ops_amd.py:782-801` (`_collect_interrupts`, whose
  `reset` resets the compute engines through `AMDev.recover`, rewinds the
  compute queue and moves the timeline past the hung value, and
  `on_device_hang`) and `:856` (`can_recover`, true under AM off a virtual
  function); `runtime/support/am/amdev.py:283-290` (`recover`).
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

- **tinygrad:** `device.py:335-341` (`Compiler.__init__`, which drops
  `cachekey` when `CCACHE` is 0, and `compile_cached`: a binary is kept under
  its source in the table `cachekey`);
  `runtime/support/compiler_cpu.py:16`, `runtime/ops_metal.py:40`,
  `runtime/support/compiler_cuda.py:54` and
  `runtime/support/compiler_amd.py:84` (the tables, named by the compiler,
  the architecture and a key); `codegen/opt/search.py:107` (the beam search's
  key: the kernel, the width, `allow_test_size`, the device and the suffix).
- **tolk:** `lib/renderer/renderer.ml:143-165` (`Compiler.v`'s `cachekey`, a
  function asked once, and `compile_cached`);
  `lib/runtime/support/compiler_cpu.ml:38` (`statement`)
  and `:110`, `compiler_metal.ml:98`, `compiler_cuda.ml:94` and
  `compiler_amd.ml:86` (the tables); `lib/runtime/support/c.ml:108`
  (`C.identity`); `lib/codegen/codegen.ml:1540-1559` (`program_key`, `kept`);
  `lib/setting.ml:285-290` (the search's settings) and
  `lib/codegen/opt/search.ml:300` (the beam search's key).
- **Differs:** a table also names everything besides the source that
  determines a binary: for Clang, the digest of what `clang -###` states it
  runs (its version and installation, the processor and features `native`
  resolves to, every option), with `-ffile-compilation-dir=.` keeping the
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

- **tinygrad:** `helpers.py:157-162` (`getenv`, read where it is called,
  with that call's default) and `:179` (`ContextVar`); `:241` (`TC`, default
  `1`, and `TC_OPT`, default `0`); `codegen/opt/search.py:18-22` (the
  actions, reading `getenv("TC", 1)` and `getenv("TC_OPT", 2)`);
  `codegen/opt/heuristic.py:21` (`TC_OPT`'s default, `2` during a search and
  `0` otherwise); `engine/jit.py:36` (`getenv("JITBEAM", BEAM.value)`);
  `engine/realize.py:275` (`compile_linear`'s `profile`, `PROFILE or
  DEBUG >= 2` unless given, for `hcq_compile`); `codegen/__init__.py:386`
  (`os.environ.get("DBGTV")`, read on each failure).
- **tolk:** `lib/setting.ml:87` (`reach`), `:141` (`shaping`), `:151`
  (`t`), `:215` (`jitbeam`) and `:220` (`tc_opt`);
  `lib/codegen/opt/search.ml:17` (`tc_opt_all`) and `:19` (`actions`);
  `lib/codegen/opt/heuristic.ml:101`; `lib/setting.ml:300` (`dbgtv`); `lib/engine/realize.ml`
  (`compile_linear`); rune's `lib/jit.ml:105` (`settings`).
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
  overrides both.
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
  setting its caller declares`; `Rune.jit › keys › a setting that shapes
  compilation ›` each setting, `› a call under DEBUG=2, which reports
  kernel times, retraces once`, `› a call under DEBUG=1 replays the
  program` and `› a call compiled with ~beam replays its program under
  another BEAM or JITBEAM`.

## D115. A search linearizes a candidate before it compiles it

- **tinygrad:** `codegen/opt/search.py:64-70` (`_try_compile`: `to_program`,
  then the count of `prg.src[1].src` against `BEAM_UOPS_MAX`).
- **tolk:** `lib/codegen/opt/search.ml:102` (`try_compile`) and
  `lib/codegen/codegen.ml:1167` (`linearize`).
- **Differs:** a candidate is linearized (`Codegen.linearize`), dropped if it
  has `BEAM_UOPS_MAX` instructions or more, and only then rendered and
  compiled (`Codegen.to_program` completes the linearized program). tinygrad
  renders and compiles every candidate, then drops the one past the cap. The
  search keeps and drops the same candidates.
- **Reason:** speed, measured by PR #235 on lorenz_simple's step on AMD: the
  candidates past the cap are the largest unrolls, the slowest to compile,
  and compiling them took 154 s of a search, 30 s once they were dropped
  before compiling. Admitting a speed rule is the maintainer's call: reason
  (b) speaks of a call site that fails.
- **Pinned by:** `Tolk.Search › BEAM_PADTO=1 ... › a candidate of
  BEAM_UOPS_MAX instructions or more is not compiled`; `Tolk.Codegen ›
  linearize and compile › compile completes linearize into the program
  to_program makes` and `› linearize compiles nothing`.

## D116. A search compiles each kernel and each source once

- **tinygrad:** `codegen/opt/search.py:132-142` (each candidate compiled,
  then dropped if its binary is in `seen_libs`).
- **tolk:** `lib/codegen/opt/search.ml:335` (`compiled`), `:337` (the
  memoized compiler) and `:344` (`compile`); `:138` (`memo`).
- **Differs:** a search keeps each kernel's compilation, by its scheduler's
  kernel (`Postrange.Scheduler.ast`), and a candidate whose kernel it met
  before, in this round or an earlier one, takes that result uncompiled. The
  memo is filled before a round's compilations are spread over the domains.
  tinygrad compiles both: two sequences of actions, such as an upcast then a
  swap and the swap then the upcast of the swapped axis, reach equal kernels
  whose kernel information records different optimisations, so
  `to_program`'s cache misses, and `seen_libs` drops the second binary only
  after compiling it. A program depends only on its kernel and its name,
  `"test"`, so the search keeps and drops the same candidates. Distinct
  kernels can also render one source: the search compiles through a memo of
  its compiler, so a source is compiled or rejected once, and a domain asking
  for one another is compiling waits for it. A binary or a rejection depends
  only on its source, so `seen_libs` sees the same binaries. Candidates are
  compiled without the disk cache of the compiler's binaries.
- **Reason:** speed, measured by PR #235 on GNODE and lorenz_simple on AMD,
  where 12 to 16% of a search's candidates were such duplicates, and in the
  host searches of a lorenz_simple step on AMD, where 10 of 12 candidates of
  a round compiled to binaries already timed. Admitting a speed rule is the
  maintainer's call, as for D115.
- **Pinned by:** `Tolk.Search › a search compiles each kernel once`
  (`matmul_small`, whose upcast and swap commute; it counts renders) and
  `› a search compiles each source once` (both tests).

## D117. A search starts from its kernel's samples and progresses beyond their spread

- **tinygrad:** `codegen/opt/search.py:113` (the beam starts as
  `[(s, inf)]`, and `s` is never timed) and `:160-163` (a round ends the
  search if nothing was timed, if the fastest took less than
  `BEAM_MIN_PROGRESS`, or if it beat the beam's first by less than that;
  the search then keeps the fastest alone if it beat the beam's first).
- **tolk:** `lib/codegen/opt/search.ml:388` (`progresses`) and `:462`
  (`start`); the golden generator, `test/gen/codegen/opt/search.py`
  (`beam_search`).
- **Differs:** the search compiles and times its kernel before its first
  round, with no early stop, and the beam starts as the kernel with its
  samples. A round progresses only when the greatest sample of its fastest
  candidate plus `BEAM_MIN_PROGRESS` is less than the least sample of the
  beam's first kernel. Otherwise the search answers the beam's first kernel:
  the exit on a fastest time under `BEAM_MIN_PROGRESS` and the fastest kept
  alone are gone. One comparison decides each round. The first round's early
  stop is three times the kernel's least sample, where tinygrad's is
  infinite.
- **Reason:** a search must never answer a kernel slower than the one it
  started from. tinygrad's rule compares the least of each program's noisy
  timings, and the least over hundreds of candidates is biased low: a round
  continues, and a candidate is kept alone, on a gain within the noise of
  the samples. PR #235 measured searches on GNODE and lorenz_simple on AMD
  whose trailing rounds ran on gains of 2 to 4% (25.96 to 25.42 us), and
  first rounds that searched kernels already at their floor, with nothing
  to compare against. The maintainer admitted this departure.
- **Pinned by:** `Tolk.Search › a round goes on iff each sample of its
  fastest beats each of the incumbent's by more than BEAM_MIN_PROGRESS`
  (law); `› the kernel is sampled three times before any candidate`; `› a
  search whose kernel does not compile progresses on any candidate`; `›
  rounds` (4 tests); `searches.golden` and `searches_environment.golden`,
  whose measurements count the kernel's three samples.

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

- **tinygrad:** `schedule/rangeify.py:230` (`bufferize_to_store`'s `ALLOC`),
  `:283` (`debuf`) and `schedule/prepare.py:164,172` (`copy_to_anon_store`,
  `stage_to_anon_store`), whose storage carries no bounds, so a read of it
  has its type's; `uop/ops.py:1163` (`_min_max`, which bounds a `BITCAST` by
  its type).
- **tolk:** `lib/uop/ops.ml:1826` (`stored_bounds`) and `:1630`
  (`bitcast_bounds`); `lib/schedule/rangeify.ml:423` (a stage's storage) and
  `:554` (`debuf`); `lib/schedule/prepare.ml:434,448`;
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the
  goldens are recorded.
- **Differs:** storage that a stage or a copy fills carries the bounds of the
  value stored when they are narrower than its type's, and the parameter a
  kernel reads it through keeps them, so each read has them. Bounds of one
  value are not kept: a parameter whose bounds are one value folds to that
  constant, in place of the storage. A bitcast between integer types of one
  width keeps the bounds of a value that both types hold, from 0 to the
  lesser of their greatest values.
- **Reason:** (b): rune lowers `Nx.take` to a gather at the clamped index
  and a check of the index, one unsigned comparison, `bitcast i < n`. A
  gather by indices that other kernels compute, clamp and store read them
  at their type's bounds, so the check stayed, and `pm_move_where_on_load`
  moved it onto each load the gather feeds. With the stored range known, the
  gather keeps no per-load check. A routed MXFP4 product whose experts are
  sorted and clamped in kernels of their own masked its 40 code and scale
  byte loads: gpt-oss-20b's gate and up product of 512 tokens on an M1
  Max's Metal took 42.6 ms and takes 36.4 ms, as fast as a gather of the
  decoded matrices. Within one kernel, the bitcast hid the clamp's bounds
  from the check.
- **Pinned by:** the Rangeify suite's `index_stored_zero_fill` gather, whose
  `_kernels` golden reads the stored indices with bounds `(0, 7)` and keeps
  no check; the Ops suite's `bitcast_bounds.golden` and `› bounds › storage
  keeps the bounds of its values where they are narrower than its type's
  and hold more than one value`; the memory goldens `convs` and
  `matmul_chain`, whose stored activations after a ReLU carry
  `(0.0, inf)`, recorded from the equally patched tinygrad.

## D123. The engine runs a loop of calls while a flag holds

- **tinygrad:** `uop/spec.py:88` (a `BACKEDGE` closes a bound-less range,
  inside a kernel), `engine/realize.py:281-286` (`run_linear` runs every
  call of a schedule once, in order).
- **tolk:** `lib/uop/spec.ml:223` and `:575` (a `Backedge` around calls, in
  the tensor and kernel-graph specs), `lib/schedule/schedule.ml:60` and
  `:184` (`create_schedule` keeps a back edge of a call),
  `lib/schedule/memory.ml:29` (`uses`, the plan counts its calls and flag),
  and `engine/tolk_engine.ml` (`loop`).
- **Differs:** an `Op.Backedge` stands around a call of a schedule, or a
  linear of calls, with a loop range of at most as many trips and a flag, one
  boolean of storage the calls write. The schedule keeps it as an entry, and the engine makes it a loop of rig's
  program (D146) that reads the flag before each trip and runs the calls
  while it holds, its trip in a word of the run's ints, which a call's scalar
  argument reads too (D140). The flag is storage the program makes or a
  buffer a caller binds. tinygrad's schedules run each call once, and stop on
  no value.
- **Reason:** (b). rune's `Rune.iterate` under `Rune.jit` stages a loop that
  stops on a condition of its carry as one body compiled once, which no
  schedule tinygrad makes can run.
- **Pinned by:** the Engine suite: `link and run › a loop runs its call while
  its flag holds, at most its trips`, `› a loop's flag that views wider
  storage is refused`, `› a planned buffer a loop writes is not placed over
  one it leaves`, `› a loop's planned flag is not placed under a buffer before
  it`, `› a back edge whose flag is bound storage reads that storage` and
  `GPUs › Metal › a loop runs its call while its flag holds, at most its
  trips`; rune's `Jit` and `Iterate` suites.

## D124. A search times no compilation

- **tinygrad:** `codegen/opt/search.py:50-60,77` (`timeout_handler`, and the
  alarm of `BEAM_TIMEOUT_SEC` that `_try_compile` sets and clears).
- **tolk:** `lib/codegen/opt/search.ml:93` (`try_compile`).
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
- **tolk:** `lib/codegen/decomp/transcendental.ml:21`
  (`transcendental_dtypes`), `:272` (`radians`), `:335` (`cody_waite`),
  `:380` (`sin_kernel`), `:404` (`cos_kernel`), `:431` (`xsin`) and `:513`
  (`xlog2`); `test/gen/tinygrad.patch`, which gives tinygrad the same
  functions before the goldens are generated.
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
- **Reason:** (b). rune's `exp`, `sin` and `cos` compute with `exp2` and `sin`
  (`packages/rune/lib/lower_arith.ml`), and the planned AMD ISA renderer and
  the eager GPU kernels compute every transcendental kind with these
  functions, under a bound of one unit for the narrow floats and of the
  vendors' documented maxima elsewhere.
  A target's function one unit off leaves rune's compositions within their
  bound of two; tinygrad's `xsin` left the sine of a float32 near a multiple
  of pi below 30 with no correct bit, and its `xlog2` put rune's compiled
  `log` two units off on the host.
- **Pinned by:** the Transcendental suite (`test/codegen/decomp/transcendental`):
  `accuracy › <function> of <type> is within an ulp` for `exp2`, `log2` and
  `sin` in float16, bfloat16, float32 and float64, against references computed
  in integers (`exact.ml`), and `› xsin ~fast of <type> is within an ulp below
  30`; `special values › sin <type> -0x0p+0 is -0x0p+0` and the subnormal rows;
  `types › <function> refuses a node of another type › half`; the graph
  goldens and `values.golden`, from the patched tinygrad; and rune's
  `lower_arith` suite: `transcendental functions on the host › log`, `› exp`,
  `› sin`, `› cos`.

## D127. A converted load reads only an alternative its conversion keeps

- **tinygrad:** `codegen/late/gater.py:5-7` (`move_where_load`), which makes
  the other value of `where gate (cast (load ...)) a` the load's alternative,
  converted to the load's type: `a.cast(l.dtype)`, or the constant's value at
  that type.
- **tolk:** `lib/codegen/late/gater.ml:17` (`holds`) and `:37`
  (`move_where_load`); `test/gen/tinygrad.patch`, which gives tinygrad the same
  rule before the goldens are generated.
- **Differs:** the selection moves into the load only when the load's type
  holds its other value: when the load is not converted, when the value is a
  conversion of a value of the load's type, or when it is a constant that
  converts to the load's type and back to its bits, sign included. A NaN or
  an infinity converts to no integer type, so it never moves into an integer
  load, and neither does an integer that a float load rounds to an infinity.
  Otherwise the selection stays. tinygrad converted the value to the load's
  type and back, so `where gate (float32 (load float16)) 0x1p-149` read 0
  where the gate failed, a non-integer or a `-0.` read through an integer
  load lost its fraction or its sign, and a float32 above 65504 read through
  a float16 load read infinity.
- **Reason:** (b). rune's pad of a widened float16 with a float32 pad value
  (`Nx.pad` of `Nx.cast Nx.float32 x`), read through a slice, compiled to such
  a selection: a pad of `0x1p-149` read 0, where eager reads `0x1p-149`.
- **Pinned by:** the Gater suite (`test/codegen/late/gater`):
  `pm_move_gates_from_index › a selection of a converted gated load keeps its
  value` and `laws › a moved selection of a converted gated load keeps its
  value`, and the cases `where_of_cast` and `where_of_half` of
  `moves_output.golden`, from the patched tinygrad; rune's Rune.jit programs
  suite: `a compiled program › computes eager's bits`, whose examples hold the
  pad.

## D128. A committed float operation reads a weak float operand rounded

- **tinygrad:** `uop/ops.py:1104` (`UOp._min_max`), which bounds a weak float
  operand of a committed float operation by its value as written, at no
  width.
- **tolk:** `lib/uop/ops.ml:1568` (`rounded`) and `:1589`
  (`operand_bounds`); `test/gen/tinygrad.patch`, which gives tinygrad the
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
  selection holds a weak constant at its type, where it may round to zero`;
  rune's Rune.jit programs suite: `a compiled program › computes eager's
  bits`, whose examples hold the pad.

## D129. A search keeps nothing of its candidates

- **tinygrad:** `codegen/opt/search.py:64` (`_try_compile` calls
  `to_program`), `codegen/__init__.py:465-491` (`do_to_program` takes a
  sink or a program at any stage), `:502-504` (`to_program_cache`) and
  `:451` (`do_compile` calls `compile_cached`); `engine/realize.py:231`
  (`_get_call_to_compile` takes a program not yet compiled).
- **tolk:** `lib/codegen/opt/search.ml:93` (`try_compile`),
  `lib/codegen/codegen.ml:1200` (`compile`) and `:1255` (`to_program`), and
  `lib/engine/realize.ml:126` (`get_call_to_compile`).
- **Differs:** a candidate is compiled by `Codegen.linearize` then
  `Codegen.compile`, which keep nothing: the compiler neither reads nor fills
  its disk cache table, and the program enters neither of `to_program`'s
  tables. tinygrad compiles a candidate through `to_program`, which keeps the
  program in memory for the life of the process, and through
  `compile_cached`, which keeps its binary on disk. A search keeps its
  choice, and the kernel compiled with it keeps its own binary, as in
  tinygrad; as in tinygrad, that kernel is compiled apart from the candidate
  it was, since a candidate is named `"test"`. `to_program` takes only a
  kernel's sink, and `compile` only what `linearize` returns:
  `to_program ast ren` is `compile (linearize ast ren) ren`, kept. tinygrad's
  `do_to_program`, and `lower_and_compile` with it, also resume a program
  from any stage. Under `ASSERT_COMPILE` a search not kept compiles its
  candidates, and the kernel compiled with it raises at its own compilation,
  where tinygrad raises at the first candidate.
- **Reason:** disk and memory, measured: a search makes hundreds of
  candidates, and kept, they grew tolk's disk cache without bound, to 8.9 GB
  on an M1 Max, and its memory table for the life of a process. A later
  search of the same kernel reads its choice back and compiles no candidate,
  so the kept binaries served only a search run again under
  `IGNORE_BEAM_CACHE`. A program resumed from a later stage had no caller
  but tests, and one keeping and one transient pipeline leave each fact in
  one place. Admitting a resource rule is the maintainer's call, as for
  D115, and the maintainer asked for this one.
- **Pinned by:** `Tolk.Search › a kernel compiled with a search keeps the
  search's choice and its own binary alone`; `Tolk.Codegen › linearize and
  compile` (every test).

## D130. A search times its candidates in order, compiling meanwhile on a device's clock

- **tinygrad:** `codegen/opt/search.py:132` (`pool.imap_unordered`: a
  round's candidates are timed in the order their compilations end, on every
  device).
- **tolk:** `lib/codegen/opt/search.ml:352` (`compile`);
  `lib/engine/worker.ml:77` (`iter`); `engine/tolk_engine.ml` (`clock`).
- **Differs:** a round's candidates are timed in their own order, each once
  its compilation has ended, while other domains compile the later ones
  (`Worker.iter`). Where the host's clock times the runs (`Search.Host`: a
  program the host calls, or a search whose kernel was not timed), a round compiles all its candidates before it times
  any. The search reads the clock from its kernel's linked program.
- **Reason:** the compute filter's running minimum, `seen_libs` and ties
  between equal samples read the order: timed as their compilations end, two
  searches of equal measurements choose apart, and `searches.golden` could
  not pin tinygrad's choices. A device stamps its runs on its own clock, but
  the host times its runs on its own cores, which compilations on the other
  cores slow. Admitting this is the maintainer's call, as for D115.
- **Pinned by:** `Tolk.Search › a search on a device's clock times its
  candidates as one on the host's, whatever order their compilations end
  in` and `› a search on the host's clock compiles nothing while it times`;
  `Tolk.Worker › streaming` (every test); `Tolk_engine › timing › the clock is a GPU's on a GPU, the host's
  otherwise` and `› a
  profile taken around a time changes neither its clock nor its value`.

## D132. A contiguous view's rewrite stops at the effects its storage waits on

- **tinygrad:** `uop/ops.py:934-948` (`UOp.contiguous_view`), whose rewrite
  of the index with `pm_mops` and `symbolic` visits every node the index
  reaches, the effects an `AFTER` orders its storage after and the graphs they
  compute included, and returns the storage node it reaches as rewritten
  (`b.rtag(None)`).
- **tolk:** `lib/schedule/prepare.ml:720` (`pm_stop_at_effects`) and `:730`
  (`contiguous_view`).
- **Differs:** the rewrite stops at void nodes, the effects storage is ordered
  after, and leaves them and what they compute as they are, so that the
  storage it returns is the node of the view's graph. The effects decide
  nothing of where the view's elements lie. The graphs they compute hold
  rune's gathers (D73), which `pm_mops`, a rule for a view's index, moves
  through a reshape into an index whose shape repeats the gather's, and
  arithmetic on weak-integer tensors, on which `Divandmod`'s `divide_by_gcd`
  raises. tinygrad's tensor graphs hold neither.
- **Reason:** (b): rune's jit asks whether each result of a compiled call is
  a view of a buffer it makes (`take` in `packages/rune/lib/jit.ml`). norn's
  NUTS warmup and sampling of a posterior over sixteen chains, compiled as one
  function, has results ordered after loops whose steps gather by
  weak-integer positions, and did not compile: it raised in `Divandmod`'s
  `divide_by_gcd`, and, past it, on a reshape of a gather's index of four
  axes.
- **Pinned by:** the `Prepare` suite: `contiguous_view › a view of storage
  after effects is a view of that storage, whatever the effects compute`.

## D133. A greatest common divisor's coefficient is a scalar

- **tinygrad:** `uop/ops.py:1083-1088` (`UOp.gcd`, whose coefficient is
  `uops[0].const_like(...)`, a constant broadcast to the shape of the first
  term) and `uop/divandmod.py:77-81` (`divide_by_gcd`, which divides every
  term by a divisor that is not the `CONST` 1). Over terms with a shape, a
  vector of index expressions or a weak-integer tensor, a divisor of 1 is a
  broadcast 1, which is not the `CONST` 1 and divides no term:
  `UOp.stack(a, b) // 3` raises in `unwrap`.
- **tolk:** `lib/uop/shape.ml:1098` (`gcd`).
- **Differs:** the coefficient is a scalar constant of the first term's type,
  whatever the terms' shape, so a divisor of 1 is the `CONST` 1, and a larger
  one divides each term as `divides` does.
- **Reason:** (b): the norn compile D132 names reached `divide_by_gcd`, before
  D132, on a weak-integer tensor of rune's gather positions, and raised there.
  The rule applies to any vector of index arithmetic.
- **Pinned by:** the `Divandmod` suite: `values › each rewrite of a random
  division of vectors keeps each lane's value`.

## D134. A widening cast of an unsigned mask or shift widens its operands

- **tinygrad:** `uop/symbolic.py:304-308` (cast/long folding), which keeps a
  cast of `x & y` or `x >> k` as it is.
- **tolk:** `lib/uop/shape.ml:2757` (`symbolic`'s cast rules);
  `test/gen/tinygrad.patch`, which gives tinygrad the same rule before the
  goldens are generated.
- **Differs:** a cast of an unsigned `x land y`, or `x lsr k` with `k` within
  `[0, width of x)` by its bounds, to a wider unsigned type is the operation on
  the widened operands, `cast x land cast y` or `cast x lsr cast k`. A zero
  extension commutes with both, so every value is kept. A signed operand, a
  shift that may reach the operand's width, and a cast to a narrower or a float
  type keep the cast where it is.
- **Reason:** (b). rune computes a uint4 at a byte and unpacks it where it
  reads it, a shift and a mask of the byte, so nx.quant's compiled MXFP4
  product, whose codes are uint4, masked each nibble as a byte and widened it
  to decode it at uint32. Clang vectorises that worse on arm64: the host's
  4096 x 4096 one-token product kernel took 40 ms where the same kernel
  unpacking at uint32 takes 21.6 ms. kaun's decode bench times the products a
  user calls (`Quant/host/*`).
- **Cost:** on x86, for codes of one value. The routed one-token product's
  kernel, the only one the rule changes in its program, is faster with it on
  random codes, as real weights hold, on arm64 and on x86 (kimchi, clang 19,
  2.1% on six cores), and the routed 64-token product runs at 0.978 of its
  time before the rule. On codes of one value the kernel takes 4.4% longer
  with the rule on six x86 cores and 2-6% longer on one, where the M1 still
  runs it faster. kaun's decode bench times random codes.
- **Pinned by:** the Symbolic suite (`test/uop/symbolic`): `symbolic › casts ›
  a widening cast of an unsigned mask masks the widened operand` and
  the four tests after it; rune's lower_index suite: `quantised products › a
  product unpacks its codes at the width it decodes them in`.

## D135. Withdrawn

A Metal command buffer signalled through the device's signaler. rig's Metal
driver signals its own (D146).

## D137. Stored values that share a computation share their loops

- **tinygrad:** `schedule/indexing.py:233` (new ranges for every stored
  value) and `:253-276` (a node its consumers index apart is stored);
  `schedule/rangeify.py:50` (`remove_bufferize`, which inlines such a store
  back into each consumer) and `:335-348` (`split_store`, one kernel per
  store). Upstream calls scheduling several outputs into one kernel pending:
  `test/runtime/test_assign.py:355`, `:384` and `:397` skip "multi output
  not supported anymore", and `test/runtime/test_jit.py:485` skips "Pending
  multioutput implementation #3607". Codegen already compiles a kernel of
  one end over a group of stores (`test/runtime/test_custom_kernel.py:139`).
- **tolk:** `lib/schedule/indexing.ml:503` (`region`), `:521`
  (`waits_over`), `:548` (`share_loops`), `:602` (`loops_of`) and `:156`
  (`end_shared`, which `bufferize_and_index` returns for each store of a
  group); `test/gen/tinygrad.patch`, which gives tinygrad the same before the
  goldens are recorded.
- **Differs:** before ranges are assigned, the stores of the tensor graph that
  have a loop are grouped. Two join when their values have one shape and one
  device, they write different storage, neither reads the other or the
  storage the other writes, through anything, neither waits for a write over
  storage that the other reads in a state before it, which one kernel of both
  could run neither before nor after, and the computed nodes each reaches
  before storage or another stored value share one. Neither may reach
  a reduction the other does not: a kernel of two reductions can lose the
  optimisations each would take. A group's stores take one set of loops, so
  their shared nodes are indexed alike and computed once, and they end as one
  end over a group of the stores, which the kernel split makes one kernel
  writing them all. Outputs that share only loads stay apart.
- **Reason:** (b): `Nx_wide`, whose operations return a double word's high
  and low parts, the two ends of one chain. tinygrad schedules each part as a
  kernel that computes the whole chain: a compiled double-word add ran six
  kernels and runs one, and `compile/wide/sum-1e6` compiled 17 kernels.
- **Pinned by:** the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › outputs that share a computation` (every test); its
  recorded `shard_of_computed_kernels`, `read_around_assign_kernels` and
  `kernel_counts`, and the Postrange suite's `where_max_multioutput` cases,
  recorded from the equally patched tinygrad; rune's `Rune nx.wide ›
  kernels`.

## D138. A load's index is simplified without the loads in it assuming its gate

- **tinygrad:** `codegen/late/coalesce.py:41-43` (`simplify_valid_load`),
  which simplifies a gated index with `uop/symbolic.py:342`
  (`uop_given_valid`) down into the loads the index reads.
- **tolk:** `lib/codegen/late/coalesce.ml:16` (`simplify_valid_load`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** each load in the index stands as a variable of its bounds while
  the gate simplifies the arithmetic around it, and is put back after, so a
  load keeps its own index and gate. tinygrad substitutes the gate's bounds
  into the loads too. A load's index whose gate implies the outer gate, such
  as a pad's, loses that gate, and the load, which runs whatever the outer
  gate, reads outside its buffer: the index of a gather through a pad is read
  before the pad's first element. tinygrad's own `gated_given_valid`
  (`uop/symbolic.py:427`) refuses an index that loads for the same reason.
- **Reason:** (b): `Nx.concatenate` of a buffer and an `Nx.take` under
  `Rune.jit`, as nested sampling's evidence concatenates its dead points and
  its live points in likelihood order: the compiled kernel read up to the
  buffer's length before the start of the indices and faulted.
- **Pinned by:** the Coalesce suite (`test/codegen/late/coalesce`):
  `indexing_simplify › a gather through a pad reads its indices only inside
  the pad`.

## D139. Construction checks no specification

- **tinygrad:** `uop/ops.py:209` (`UOpMetaClass.__call__`, which checks each
  node it creates against `spec_full` when `SPEC` is above 1, and computes its
  shape when `SPEC` is above 2), and `schedule/rangeify.py`'s
  `Context(SPEC=min(SPEC.value, 2))` around `pm_apply_rangeify`.
- **tolk:** `lib/uop/ops.ml` (`v`), `lib/setting.ml` (`spec`) and
  `lib/schedule/indexing.ml` (`run_rangeify`).
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

- **tinygrad:** `schedule/__init__.py:103-115` (`resolve_linear_call`, which
  binds a scalar parameter only to a variable, and substitutes it into each
  kernel that reads it), `engine/realize.py:15-17` (`get_call_arg_uops`, which
  leaves out bound variables only, and `get_call_var_uops`, which reads each
  variable by its name).
- **tolk:** `lib/schedule/schedule.ml:59` (`scalar_argument`), `:259`
  (`pass_values`) and `:283` (`resolve_linear_call`);
  `lib/engine/realize.ml:14` (`get_call_arg_uops`) and `:17`
  (`get_call_var_uops`);
  `engine/tolk_engine.ml` (`affine`).
- **Differs:** a precompiled call inside a loop may pass a scalar parameter of
  its body a weak integer expression of the loop's ranges, such as `r`,
  `2r + 1` or `n - 1 - r`. The schedule keeps it as the call's argument, no
  state. Each kernel of the body that reads the parameter keeps it, unnamed,
  in a slot past the kernel's buffers, bounded by its type, and its call passes
  the expression in that slot: `get_call_var_uops` reads an unnamed variable of
  a program from its call's slot, and `get_call_arg_uops` leaves out every
  scalar argument, a free variable included. The value reaches the program
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
  links that call. tinygrad's `codegen/opt/postrange.py:255` (`args_from_ast`)
  reads every scalar's `expr` and constructs only buffer arguments.
- **Reason:** (b). rune's `Loop.repeat` and staged scans give each step its
  trip's index, which a kernel or a buffer of indices would otherwise compute.
- **Pinned by:** the `Schedule` suite: `create_linear_with_vars › loops of
  calls › a loop's call passes its kernel the trip as a scalar argument`; the
  `Spec` suite: `kernel_graph › loops of calls › accepts a loop's call whose
  scalar argument is an expression of its range`; the `Realize` suite:
  `get_call_arg_uops › is a call's storage arguments in order, without its
  scalar ones` and `get_call_var_uops › is the call's argument in the slot of a
  scalar parameter`; the `Tolk_engine` suite: `link and run › a loop's call
  reads its trip as a scalar argument on each trip`, `› a back edge's call
  reads its trip on each trip, at most its trips`, `› a loop of one trip passes
  its call the trip as a scalar argument`, `› loops of any range and trips
  around one call compile one program`, and the same two on each GPU; `Tolk.Search › a search binds a positional size in its call and
  launch`; `Tolk.Search on the host › a search runs candidates with named and
  positional scalars`; `Rune.jit › keys › a scan searched with beam 2 draws as
  eager does`.

## D141. The host computes upcast lanes as vectors of 64 bytes

- **tinygrad:** `codegen/__init__.py:117` (`do_devectorize`, which splits
  every elementwise operation, load and store of several lanes into one per
  lane, on every target) and `:137` (`devectorizer2`);
  `renderer/cstyle.py:40-42`, which writes a conversion of a vector that is
  not in registers as a C cast, a reinterpretation of its bits;
  `codegen/opt/heuristic.py:112-138` (the upcasts of output axes by 3 or 4)
  and `:156-159` (the upcast by 4 of a kernel that has none).
- **tolk:** `lib/renderer/renderer.mli:138` (`vector_alu`);
  `lib/codegen/codegen.ml:235` (`constant_lanes`), `:270` (`in_order`),
  `:291` (`segment`), `:304` (`vector_operation`), `:337`
  (`distinct_lanes`), `:358` (`vector_dtypes`), `:369` (`vector_lanes`),
  `:375` (`do_devectorize`), `:414` (`address_lanes`), `:472`
  (`pm_vector_constants`) and `:861` (`product_operands`);
  `lib/renderer/cstyle.ml:546` (`mask_widths`), `:909` (`clang_vectors`) and
  `:1144` (`clang_kernel`); `lib/uop/spec.ml:504` (`vector_program`); `lib/codegen/decomp/decomp_dtype.ml:880` (`emulates`);
  `lib/codegen/opt/heuristic.ml:446` (`host_vector_bytes`), `:463`
  (`parallel_operations`) and `:494` (`upcast_vectors`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded, and makes its compiled patterns match a repeated source
  lazily, as its interpreted ones do: a stack whose first lane is a constant
  and whose others are lanes of a vector raised in a bind of the first lane's
  source that the match never needed.
- **Differs:** a renderer may compute on vectors (`Renderer.vector_alu`;
  Clang does). There an elementwise operation of an even number of lanes on
  a type the target computes natively stays an operation on vectors of its
  lanes, on one axis, when each source holds them in order: a vector
  operation, constants in any view (a literal), a stack of loads of memory, a
  load in as many lanes as read it, and of lanes of vectors, each in one
  lane, or a scalar that every lane reads, and when no two of its lanes read
  the same sources: a lane of a vector operation is its own, a lane of a
  stack its element. Its lanes are one vector when they are a power of two,
  and m vectors of 2^k consecutive lanes when they are m·2^k, m odd: 48 lanes
  are 3 vectors of 16, 6 lanes 3 of 2. A source that holds computed lanes in
  another order, a broadcast or a permutation across lanes, or apart in
  registers, as an accumulator's lanes, costs a vector shuffles that a scalar
  reads for free, and lanes that repeat are computed once each apart: such an
  operation computes each lane apart. An operation of an odd number of lanes,
  3 or 9, holds no vector of a power of two of lanes beyond 1, and computes
  each lane apart too. Loads, stores, tensor-core products, calls, and
  operations on a weak type (an address) or an emulated one still compute each
  lane apart; an address is computed lane by lane through the vector
  operations it reads, so that accesses still merge. A sum fuses each lane of
  a vector product into its running sum, a selection by constant conditions
  picks each lane, and a lane of a vector whose sources fold to scalars, as a
  selection by a comparison that holds in every lane, is the scalar. A program
  for such a renderer is checked against
  `Spec.vector_program` (D58). Clang writes the vectors as `ext_vector_type`
  values: a conversion is `__builtin_convertvector`, square roots,
  truncations and multiply-adds are `__builtin_elementwise_*`, a scalar
  operand of a builtin or a selection is splat, and a vector of booleans is a
  mask of -1 and 0 lanes as wide as the values it compares, converted to the
  width of the masks or values it meets; a lane of a mask is read as a
  `_Bool`, and a mask of one-byte lanes, which C types as plain chars, as
  signed chars.

  On the host, a kernel with no reduce takes, after its masked axes, one
  upcast of its last output axis that 2 divides, by the largest power of two,
  at most 16, that keeps its lanes within 64 bytes of the widest type it
  reads, writes or computes, and again while they have room: 16 float32
  lanes, 8 float64, 32 float16. It takes none when its stored values hold 5
  arithmetic operations or more for each of their longest chain, or when it
  computes a type the target emulates. tinygrad upcasts output axes by 3 or 4
  while 1024 elements remain and fewer than 32 lanes, and by 4 a kernel that
  has none.
- **Reason:** (b): RFC 0025's special functions on the host. Lanes copied as
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
  A kernel whose operations already run 5 at a time gains nothing from lanes
  and pays for them in compile time, which grows with lanes times operations:
  `lgamma`, float64 `sin` and `log_betainc` take none, and `log_betainc` and
  its gradients compile as fast as without the upcast (2.7 s and 4.9-8.9 s
  cold). The 64 bytes and the 5 were measured on the M1 alone.

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
  lanes stay apart, where each is computed once. With D143, the
  host's products of a checkpoint's codes run 1.7-3.6 times as fast for one
  token's routed product and 2.4-3.3 times for a dense one, on an M1 Max
  (alternated under load). A matmul's upcast products now build vectors of
  their shared loads that each lane's multiply-add reads back
  (`clang_matmul_upcasted*`); Clang compiles them to the same instructions.

  LLVM widens a vector whose lanes are not a power of two to the next power
  of two. Ubuntu 24.04's Clang 18.1.3 crashed on x86 ("SmallVector unable to
  grow") compiling rune's sort of a uint4 [2; 3] view along its last axis:
  its kernel `r_3_2`, a masked axis of 3 upcast whole by the 2 nibbles of a
  byte, selected 6 bytes by a comparison of 6 longs. Clang 19 compiles it.
  Split into vectors of a power of two of lanes, which LLVM never widens, the
  kernel selects 3 pairs of bytes by 3 comparisons of pairs of longs, and the
  host keeps its vectors where an axis of 3 is upcast whole: the 48 and 24
  lanes of lorenz_simple's kernels `r_128_16_3_4_4_3_3_400_400` and
  `r_32_3_2_4_3_400` are 3 vectors of 16 and 3 of 8.
- **Rejected:** compile time is paid once and replay on every call, so
  neither trade below was taken. `#pragma clang loop vectorize(disable)`
  changes nothing on vector kernels (Clang 0.24 s with and without it on
  float64 `erfinv`, replay 0.99-1.00 times on 9 kernels); `-fno-vectorize`
  on scalar kernels cuts the gradient of `log_betainc` from 7.68 s to 1.36 s
  of Clang but slows its replay from 204 ms to 475 ms. One 16-byte register
  of lanes at 5 operations or more saves at most 2.1 times of Clang (that
  gradient, 6.35 s to 3.05 s) and replays 1.17-1.79 times slower (the
  gradient 1.17, float64 `sin` 1.28, float32 `lgamma` 1.79).
- **Pinned by:** the Cstyle suite's `sources › by default › clang_vector_*`,
  from the patched tinygrad, and `execution on the host › every kernel the
  interpreter runs writes what it computes`; the Codegen suite's `vectors in
  programs › a host program computes on vectors of its lanes` and `› a host
  program computes on vectors of a power of two of lanes` (every host case),
  `› a host program of 6 lanes computes 3 vectors of 2`, `› a host program
  of 48 lanes computes 3 vectors of 16`, `› a host program of 24 lanes
  computes 3 vectors of 8`, `› a host selection of 8, 16 or 32 lanes by a
  comparison of integers takes each lane's arm` and `› a host selection of 3,
  6, 12, 24 or 48 lanes by a comparison of integers takes each lane's arm`
  (every case, under Clang 18 on x86), `› a weak constant stored into four lanes of half on the host
  converts them` and `multiply-adds` (every test); the Spec suite's `vectors
  in programs › a program of a renderer that computes on vectors …`;
  rune's Rune.jit suite: `lanes` (every test); rune's packed suite: `a
  compiled call computes eager's values › a uint4 sort along the last axis
  of every other row and column, flipped`, under Clang 18 on x86, and
  `stated values › a window set into bits at an odd offset keeps the bits
  around it`, whose last vector of 8 lanes selects 0 in every lane; rune's
  Rune.quant suite:
  `decoding` (every law); rune's lower_index suite: `quantised products › a
  product decodes a checkpoint's codes 16 at once on the host` (with D143)
  and `› a product over GGUF's Q6_K blocks converts each scale once`; the
  Codegen suite's `invalid_lanes_clang` and `invalid_lanes_int8_clang`, whose
  lanes of one constant compute once;
  the Heuristic suite's `the
  optimisations chosen are tinygrad's › applied_opts`, the host cases
  `exp_float_cpu`, `exp_double_cpu`, `add_half_cpu`, `add_uchar_cpu`,
  `cast_to_double_cpu`, `compare_cpu`, `add_24_cpu`, `add_17_cpu`,
  `wide_tree_cpu` and `long_chain_cpu`, and every other host case without a
  reduce, recorded from the equally patched tinygrad.

## D142. Withdrawn

The host ran a run of its calls as one host program, which called each through
nx.device's entry. Each call is a launch of rig's host, consecutive ones one
submission, and a range a loop of rig's program (D146).

## D143. The host unrolls a reduce beside a small unrolled axis to a vector

- **tinygrad:** `codegen/opt/heuristic.py:140-153` (the unroll of the last
  reduce axis, and of a second one when both are at most 3).
- **tolk:** `lib/codegen/opt/heuristic.ml:611` (`unroll_vector`) and `:625`
  (`unroll`); `test/gen/tinygrad.patch`, which gives tinygrad the same before
  the goldens are recorded.
- **Differs:** on the host, when the last reduce axis is at most 3 and wholly
  unrolled and the next one is larger than 3, the next is unrolled by the
  largest power of two that divides it and keeps the unrolled lanes within 64
  bytes of the reduce's type (`host_vector_bytes`, D141): 8 beside a pair of
  float32 lanes, 4 beside 3. It reaches every host reduce of that shape, as
  `experts_down`'s product summed over its two experts. tinygrad unrolls the
  next axis only when it is at most 3 too.
- **Reason:** (b). nx.quant's MXFP4 product of a checkpoint's codes reduces
  over a byte axis and the pair of codes each byte holds: the pair alone left
  each iteration two decoded codes per row, against 16 for GGUF's blocks,
  whose 16 code bytes are the last axis, and an iteration's loop, its scale
  decode and its loads outweighed them. With D141's vectors, kaun's decode
  bench on an M1 Max (alternated under load 22-24) times the dense product of
  a checkpoint's codes at 2.6-3.0 ms against 2.6-3.3 ms for the blocks, from
  6.7-9.8 ms; one token's routed product at 11-21 ms from 37-55 ms; 64 tokens'
  at 270-326 ms from 340-563 ms. Without D141's vectors the same unroll is
  slower: its lanes decode one at a time (64 tokens' product 780-985 ms). On
  one core, the dense kernel unrolled by 4, 8 and 16 bytes ran in 8.5, 7.7
  and 8.9 ms, the blocks' in 7.3 ms; the routed one's in 39, 34 and 32 ms,
  compiled in 0.11, 0.15 and 0.26 s.
- **Pinned by:** the Heuristic suite's `the optimisations chosen are
  tinygrad's › applied_opts`, the host cases `vecmat_decoded_cpu`,
  `gpt_oss_gate_up_cpu`, `routed_blocks_cpu`, `routed_tiles_cpu` and
  `experts_down_cpu`, recorded from the equally patched tinygrad; rune's
  lower_index suite: `quantised products › a product decodes a checkpoint's
  codes 16 at once on the host`.

## D144. Reductions of one domain that read none of each other share their loop

- **tinygrad:** `schedule/indexing.py:299-301` (`run_rangeify`: each
  reduction takes new ranges for the axes it reduces), `:253-276` (a node its
  consumers index apart is stored) and `schedule/rangeify.py:50`
  (`remove_bufferize`, which inlines such a store back into each consumer);
  `codegen/__init__.py:177` (`merge_reduce_ends`, which already makes one
  loop of the reductions that close one range); `codegen/simplify.py:21`
  (`simplify_merge_adjacent`, which merges a reduction's ranges in that
  reduction alone).
- **tolk:** `lib/schedule/indexing.ml:572` (`reaches`), `:583`
  (`reduce_loops`) and `:709` (its use in `assign_ranges`);
  `lib/codegen/simplify.ml:106` (`merge_reductions`);
  `lib/codegen/codegen.ml:613` (`merge_reduce_ends`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** a reduction whose output is indexed by the same ranges as an
  earlier one's in the walk, and that reduces axes of the same sizes, takes
  that reduction's reduce ranges when no reduction of the group reaches it,
  through anything. The walk meets a reduction before those it reads, so only
  a reduction already in the group can read it. The sources are then indexed
  alike: a node they share is one node, computed once per element of the
  loop, where tinygrad indexes it by each reduction's ranges, stores it, and
  inlines the store back into each loop. A group of a loop that several
  reductions close takes a buffer of shared memory for each (D101). Two
  ranges that reductions close merge, at the kernel's sink, in every
  reduction that closes them, when that leaves all of them no more floor
  divisions and remainders: merged in one alone, they would leave its
  siblings' loops apart from its own, a range that runs again after the loop
  that ends it, which the linearizer refuses. A reduction can stop closing
  its group's ranges: a sum over a range its source does not read drops it
  (`pm_reduce_unparented`), so a sum over `i` and `k` of a source that reads
  `k` alone closes `k` alone, where a maximum beside it closes `k` within
  `i`'s loop. tinygrad gives a reduction's end fresh ranges only when it
  closes the same ranges as another at a different depth; tolk gives them to
  every end that closes a range an earlier end closes, so each range has one
  end. No recorded kernel has such ends, so the patch leaves tinygrad's
  `merge_reduce_ends` as it is.
- **Reason:** (b): rune's reverse mode. The cotangent of a value that
  operations broadcast sums one reduction per operation, each over the same
  cotangent computation: the gradient of an aperture's weights over a 104 x
  104 grid in its radius, which 28 operations broadcast, was one kernel of 28
  loops, each computing the whole cell, and took 647-672 ms compiled against
  93-171 ms eagerly on an M1 Max under load 11-15; one loop takes 30-31 ms.
- **Pinned by:** the Rangeify suite (`test/schedule/rangeify`):
  `get_kernel_graph › reductions that share a domain` (every test), and its
  law over generated programs, whose `sum k + max k` and `x - sum k` steps
  cover reductions that share a loop; its recorded `multireduce_*` and
  `ugly_reduceop_pairing` kernel graphs and `kernel_counts`; the Indexing
  suite's recorded `two_consumers` graph; the Postrange suite's
  `sum_and_max_grouped`, `ten_shared_sums_group` and
  `ten_shared_sums_group_unfit` cases, and the Codegen suite's
  `two_grouped_stores_local`, whose upstream axes count one loop for the two
  maximums; all from the equally patched tinygrad. The Codegen suite's
  `reductions that share a range › a range two reductions close at two
  depths runs as two loops`. The Simplify suite's
  `pm_simplify_ranges › merging › reductions that share ranges merge them in
  all of them` and `… keep them when one would divide`. Rune's Packed suite:
  the bit example of `a compiled call computes eager's values`, two views a
  bit apart whose sums share their loops. Rune's Jit suite:
  `transformations › a compiled gradient of a parameter broadcast over a grid
  is its gradient`; the rune bench's
  `reverse/jit-of-grad-clipped-disc-float64-104x104-host` row.

## D145. An integer division on padding divides by 1 there

- **tinygrad:** `schedule/indexing.py:101-105`
  (`convert_pad_to_where_to_keep_behavior_local`) and `:177`
  (`apply_movement_op` on a `PAD`, which makes the source's ranges invalid on
  the padding); `codegen/opt/postrange.py:146-161` (`PADTO`, which gates the
  loads of the padded axis). Either way the source is computed on every lane,
  and its loads read zeros on the padding.
- **tolk:** `lib/schedule/indexing.ml:236` (`constant`), `:249`
  (`guard_division_on_padding`) and `:330` (its rule in `pm_apply_rangeify`);
  `lib/codegen/opt/postrange.ml:490` (`pad_division`, in `Padto`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** a `CDIV`, `CMOD`, `FLOORDIV` or `FLOORMOD` of integers divides by
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
  tinygrad; norn's `reproducibility › a compiled transition is the eager one`
  on x86.

## D146. The runtime is rig: no command queues, batches or host batches

- **tinygrad:** `runtime/support/hcq2.py` (`hcq_compile`, `sched_batches`,
  the queue encoders it calls through `Compiled.pm_encode`, the host
  programs that submit them, the staging copies and `hcq_link`);
  `runtime/ops_*.py` (`pm_encode` and the queues);
  `engine/realize.py:271` (`compile_linear`, which ends in `hcq_compile`)
  and `:281` (`run_linear`).
- **tolk:** `lib/engine/realize.ml` (`compile_linear`);
  `engine/tolk_engine.ml` (`link`, `run`, `time`); `test/gen/engine/jit.py`,
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
  program's, and storage a caller binds is an input of each run.
- **Reason:** (c). rig owns each GPU's queues: two runtimes cannot share a
  GPU, and memory belongs to the runtime that made it.
- **Pinned by:** the Tolk_engine suite: `link and run › recorded`, `GPUs`
  (each recorded program, loop and refusal on Metal, CUDA, NV and AMD where
  the machine has one), `lanes and words` and `runs`; the jit goldens
  (`test/engine/jit/*_lowered.golden`), which `gen/engine/jit.py` makes
  without batches.

## D147. The x86 host holds bfloat16 as its bits

- **tinygrad:** `renderer/cstyle.py:276` (`ClangRenderer.type_map`, which
  writes a bfloat16 as Clang's `__bf16` on every machine).
- **tolk:** `lib/renderer/cstyle.ml:1180` (the Clang renderer's language,
  which on x86 maps bfloat16 to `unsigned short` and renders its constants as
  their bits with `pm_bf16_ushort_const`).
- **Differs:** on x86, a bfloat16 value, pointer or vector is written as
  `unsigned short`, and a bfloat16 constant as its bits, as HIP writes them
  on targets without bfloat16. The kernel's graph is tinygrad's; only its C
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
  kernels (`bf16_clang`, `clang_*_bf16`). The rune cases above on x86.

## D148. A selection keeps its guard of a conversion or a division

- **tinygrad:** `uop/symbolic.py:228-235` (`fold_where_closure`, which
  assumes a selection's condition true in its true branch and false in the
  other, unless an `INDEX` is involved).
- **tolk:** `lib/uop/shape.ml:2447` (`guards_partial`) and `:2457`
  (`fold_where_closure`).
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
  is kept in the branches`; rune's `a compiled program › computes eager's
  values within the stated rounding`, whose third example is that sine.

## D149. Host code calls by System V on Windows too

- **tinygrad:** `renderer/cstyle.py:286-288` (`ClangRenderer.abi` and
  `kernel_typedef`, `__attribute__((ms_abi))` on Windows) and `:71` (the
  `CALL` rule, which casts a function pointer with `abi`).
- **tolk:** `lib/renderer/cstyle.ml:376` (`base_rewrite`'s call rule) and
  `:1034` (`clang_lang`).
- **Differs:** the Clang renderer writes no calling convention, on every
  system: a kernel and the function pointer a call casts to are both plain C,
  which Clang's ELF target compiles to System V on x86_64.
- **Reason:** (c). tinygrad calls a kernel through ctypes, which uses the
  system's convention, and a callback it passes is a ctypes function. rig's
  host calls linked code by System V on Windows too, and tolk's kernel is
  called only by the entry compiled with it (D28). A function the kernel
  calls through a pointer is another image's, which an entry ref places in
  the launch's parameters (`Rig.Submission.entry`), so it is linked code,
  System V as well.
- **Pinned by:** the Clang goldens, which hold no convention and are checked
  on Windows: the cstyle suite's `sources`, among them `clang_call_out`,
  `clang_call_ret` and `clang_call_stack`; the codegen, realize and jit
  goldens of CPU kernels.

## D150. The folder's sine is correctly rounded

- **tinygrad:** `uop/ops.py:1410` (`python_alu`'s `SIN`, `math.sin`, the C
  library's).
- **tolk:** `lib/uop/ops.ml:2483` (`sine`), which `python_alu` calls.
- **Differs:** `exec_alu` computes a sine on integers: the argument is
  reduced exactly by pi/2, held to 1200 bits, and the sine or cosine of the
  remainder is summed in fixed point to 192 bits, then rounded once. The
  result is correctly rounded on every host, where tinygrad's is the C
  library's: on an M1 Mac about 3% of drawn float64 sines are one unit off,
  and mingw-w64's reduces by x87's 66-bit pi. A NaN operand gives `Dtype.nan`.
- **Reason:** (b). rune's `sin` and `cos` are `SIN` nodes
  (`packages/rune/lib/lower_arith.ml`), which fold through `exec_alu`, and a
  fold must not depend on the host: on Windows `sin 1e300` folded to
  `0.979`, of the wrong sign.
- **Pinned by:** `Tolk.Ops › exec_alu › exec_alu_values.golden`, whose `SIN`
  rows include `1e300`, `3.4e38`, pi and floats near multiples of pi, and
  which fail on Windows with the C library's sine.

## D151. An x86 kernel declares that it uses 512-bit vectors

- **tinygrad:** `renderer/cstyle.py:122` (`kernel_typedef`, `void`) and
  `:306` (`ClangRenderer.__init__`).
- **tolk:** `lib/renderer/cstyle.ml:1196` (the Clang renderer's language on
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
  carry it; rune's `Rune.jit programs › computes eager's values within the
  stated rounding`, whose examples hold the program found, under Clang 18 on
  an x86 host with AVX-512.

## D152. The disk cache holds at most CACHESIZE bytes

- **tinygrad:** `helpers.py:416-447` (`diskcache_clear`, `diskcache_get`,
  `diskcache_put`): a put inserts or replaces a row and nothing removes one
  but `diskcache_clear`.
- **tolk:** `lib/helpers.ml:478` (`mark`), `:511` (`size_file`), `:526`
  (`count`), `:592` (`trim`), `:624` (`make_room`), `:663` (`put`) and
  `:684` (`clear`); `lib/setting.ml:247` (`cachesize`, 5 GiB by default, `0`
  for no bound).
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
  `vmax + 1` over a node's ranges), `codegen/simplify.py:81-91`
  (`reduce_unparented`, a sum times the size, a product to its power, a
  maximum the value) and `uop/symbolic.py:390-398` (`reduce_mul_chain`, an
  integer maximum of a product times the factor that ignores its ranges).
- **tolk:** `lib/codegen/late/linearizer.ml:16` (`runs`),
  `lib/codegen/simplify.ml:242` (`reduce_unparented`) and
  `lib/uop/symbolic.ml:167` (`reduce_mul_chain`); `gen/tinygrad.patch` makes
  the three changes in tinygrad, so the goldens state tolk's order.
- **Differs:** the linearizer counts a range as at least two runs, so a
  loop's body always sorts after the nodes around the loop. tinygrad counts a
  range over a variable of size 0 or 1 as one run, its body ties with what
  precedes the loop, and a sum's first store of its accumulator was placed
  inside the loop: at size 0 the sum read its accumulator unwritten. A
  reduction over a range its value ignores, whose size may be 0, is its
  identity where the size is 0: tinygrad's sum of an infinity over it is NaN
  there, and its maximum the value. Where the size's bounds make each use of
  the range fold, as `r // 2` for a size of at most 1, the value ignores the
  range and the reduction took that form. An integer maximum of a product by
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
  of variable sizes; `Tolk.Simplify › pm_reduce_unparented › a reduction
  keeps its value over a range that may run no time`; the linearizer's
  `orders_output` goldens.

## D155. A pad by -0. fills its padding with -0.

- **tinygrad:** `mixin/op.py:289` (`_pad_constant`) and `:295` (`pad_to`),
  which return the plain pad when `value == 0`, true of `-0.`.
- **tolk:** `lib/uop/shape.ml:654` (`is_zero`), used by `:668` (`pad`) and
  `:694` (`pad_to`).
- **Differs:** a fill is left to the pad's own zeros only when its bits are
  all zero. A `-0.` fill takes the selection off the padded mask, so the
  padding holds `-0.` where tinygrad's holds `+0.`.
- **Reason:** (b). nx's `pad` fills with the value given, `-0.` included, and
  rune lowers a padded load (`dev/rune2/lib/lower.ml`, `padded`) through
  `Shape.pad` with that fill, so its compiled pad must compute eager nx's
  bits.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `signed zeros ›
  Shape.pad by -0. computes -0. on the padding`.

## D156. A NaN constant is written from its bits

- **tinygrad:** `renderer/cstyle.py:25-26` (`base_rewrite` writes every NaN
  constant as the language's `nan` cast to its type), `:138` (`NAN`), `:271`
  (Clang's `__builtin_nanf("")`), `:445` (CUDA's
  `#define NAN (__int_as_float(0x7fffffff))`) and `:504-506` (HIP's 8-bit
  floats, `f32_to_fp8(NAN, …)`).
- **tolk:** `lib/renderer/cstyle.ml:197` (`nan_const`), the `bitcast` of each
  language (`:674`, `:1251`, `:1417`) and `cdna_rewrite` (`:1611`);
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
- **Reason:** (b). nx's values are their bits, and rune lowers every float
  constant from those bits (`dev/rune2/lib/lower.ml`, `padded`, fills a
  padded load with its fill's bits through `Shape.pad`), so a compiled kernel
  must store the bits eager nx stores.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `a NaN constant
  is stored with its sign and payload › *`, which runs a store of NaNs of both
  signs, quiet and signalling, of half, float and double on the host; and the
  `*_inf_nan_*` goldens of the `Cstyle` suite, for every renderer.

## D157. A C-style expression nests at most 64 brackets

- **tinygrad:** `renderer/cstyle.py:244-249` (`_render`'s rule that writes a
  value read once into its reader).
- **tolk:** `lib/renderer/cstyle.ml:720` (`max_nesting`), `:723` (`nesting`),
  `:783-796` (`inlined`) and its call at `:847`.
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
- **Reason:** (b). rune's float64 `fmod` (`dev/rune2/lib/lower_real.ml`,
  `fmod`) computes `r := (r << left) % my` 187 times, which written as one
  expression nests about 374 levels, so its kernel did not compile on the
  host: rune's `binary kinds, comparisons, where and fma give eager's bits`
  (float64) and `binary › mod` failed.
- **Pinned by:** the Cstyle suite: `parentheses › a chain of any length nests
  a bounded number of brackets`, which renders chains of 1 to 300 steps with
  each renderer, and `execution on the host › computes a chain too deep for
  one expression`, which runs 300 steps on the host against a fold in OCaml.

## D158. A dead read of a staged value is 0

- **tinygrad:** `schedule/rangeify.py:100-102` (`remove_bufferize`), which
  inlines a stage's value at a read and leaves each range whose index is
  `Invalid` in place.
- **tolk:** `lib/schedule/rangeify.ml:141-151` (`remove_bufferize`);
  `test/gen/tinygrad.patch`, which makes tinygrad's rule the same before
  the goldens are generated.
- **Differs:** a read of a stage that would be inlined, at indices of which
  one is `Invalid`, is 0, as a load at an `Invalid` index is. tinygrad inlines
  the stage's value there with the stage's own ranges, which no kernel ends,
  so a value that a gate around the read does not fold away leaves a free
  range in the kernel graph, which its specification refuses: "UOp
  verification failed … RANGE … AxisType.WEAK".
- **Reason:** (b). rune's compiled Fold (`dev/rune2/lib/lower_index.ml`,
  `fold`) of a complex128 `[2; 3; 1]` by `lo [0; 1; 2]`, `interior [0; 1;
  0]` and a window of size 2 and step 2 on its last axis, whose every term
  the crop drops, stages its operand to read its two parts and raised
  there: rune2's `a fold wraps integers and holds its float bound` failed.
- **Pinned by:** the `Rangeify` suite (`test/schedule/rangeify`):
  `get_kernel_graph › dead reads › a staged value read only in padding reads
  0 there`.

## D159. A captured schedule tells the settings it compiles under

- **tinygrad:** `engine/jit.py:29-37` (`jit_lower`), which reads `JITBEAM`,
  or else `BEAM`, at `:35` and passes the width to `compile_linear` as its
  `beam`, binding no setting; its search cache keys on no setting
  (`codegen/opt/search.py:107`).
- **tolk:** `lib/engine/jit.ml:8-23` (`width`, `bindings`, `shaping`), and
  `jit_lower`, which compiles under `bindings`; `lib/setting.ml`
  (`shaping_under`).
- **Differs:** `jit_lower ~beam` compiles with `BEAM` at its width and
  `JITBEAM`, read by then, unset, so the caches of what it compiles key on
  the width alone. `Jit.shaping ~beam` is `Setting.shaping` under those
  bindings, read without binding them (`Setting.shaping_under`): two
  lowerings whose `shaping` is equal compile alike, whatever `BEAM` and
  `JITBEAM` were. tinygrad has no such value. tolk's caches key on every
  setting that shapes compilation (`Setting.shaping`), so they keyed on
  `JITBEAM` after the lowering had read it; unset, they key on the width
  alone.
- **Reason:** (b). rune2's `Rune.jit ~beam` (`dev/rune2/lib/compile.ml`,
  `shaping`) keys a compiled function's variants on the settings that shape
  its programs, and with an explicit width neither `BEAM` nor `JITBEAM`
  does; it read the lowering's rule by repeating it.
- **Pinned by:** the `Jit JITBEAM` suite (`test/engine/jit/test_jitbeam.ml`):
  `JITBEAM unset › a width given and BEAM's alike compile alike` and
  `JITBEAM=3 › a width given and JITBEAM's alike compile alike`; and the
  `Tolk.Setting` suite: `shaping › under bindings is shaping in their
  context, and binds nothing`.

## D160. A kernel that waits for the overwrite of a state it reads reads a copy

- **tinygrad:** `schedule/__init__.py:59-65` (`create_schedule`), which adds
  no write-after-read edge from a kernel to an overwrite of a state it reads
  when the overwrite is already among the kernel's dependencies. The kernel
  then runs after the overwrite and reads the new bytes. Repro: `a =
  Tensor([1., 2., 3.]).contiguous().realize(); b = a * 2; a.assign(a + 1);
  (b + (a * 3).contiguous()).tolist()` gives `[10, 15, 20]` for `[8, 13,
  18]`; `test/runtime/test_assign.py::TestAssignOrdering::
  test_war_reader_already_depends_on_write` accepts the wrong value.
- **tolk:** `lib/schedule/schedule.ml:81-314` (`copy_state`,
  `copy_before_overwrites`, `schedule`); `test/gen/tinygrad.patch`, which
  makes tinygrad copy the same way before the goldens are generated.
- **Differs:** no order runs such a kernel both before and after the
  overwrite, so it reads instead a copy of the state it read: a kernel of its
  own copies the whole state into call-local storage before the overwrite,
  and the schedule is made again from the graph with the copy. This holds for
  any overwrite of the storage, whichever elements each kernel touches, so a
  partial overwrite of elements the kernel does not read copies too, at a
  kernel and the state's size in storage. A state on several devices is
  refused with `Invalid_argument`.
- **Reason:** none of (a) to (c): no raven call site is known to reach it,
  since rune2 orders its own lent stores (`assign`) and gives up a lending
  where they have no order. The lead ruled the entry admitted on principle:
  the skipped edge gives a wrong answer with no error, and a refusal would
  make rune2's `Rune.jit` (`dev/rune2/lib/compile.ml`, `declined`) evaluate
  the program, which raises on a set of GPUs, which has no eager kernels;
  copying, as XLA does for an aliasing hazard, keeps the program compiled.
- **Pinned by:** the `Tolk.Schedule` suite: `create_schedule › rules › a
  kernel that waits for the overwrite of a state it reads reads it`, on the
  kernel graph `read_overwritten_through_kernels.golden`, `› a kernel reading
  storage before and after a write reads both` and `› reads it before`, and
  the rules after them, for a copy two kernels share, a kernel with two
  copies, a copy of a state a rebuilt kernel writes, the copy's phase and a
  state on several devices; and
  the program `read_overwritten_through`, whose recorded schedule and linear
  and whose values (`create_linear_with_vars › values`) are tinygrad's,
  patched.

## D161. A folded weak constant wraps, and a power of two is read at its width

- **tinygrad:** `uop/symbolic.py:278-280` (two stage ALU folding), which
  folds `c1 op c2` as weak constants, unbounded, under an operation on a
  committed integer; `codegen/decomp/op.py:64-96` (`powers_of_two`), which
  turns a product, quotient or remainder by `2^i`, `i < 64`, into a shift or
  a mask by the constant's exact value, where the operation reads a weak
  constant at its operand's width. On int32, `(y * 65536) * (y * 65536)`
  becomes `(y * y) << 32`, a shift by the full width, which C leaves
  undefined: clang gives poison. Repro: `y = Tensor.arange(8,
  dtype=dtypes.int32).contiguous().realize(); p = y * c; (p * p).tolist()`
  renders `<< 32` and `<< 36` on the CPU for `c = 65536`, and for `c =
  -2**31` renders `<< 62` and the literal `73786976294838206464`, which clang
  refuses.
- **tolk:** `lib/uop/shape.ml:1891-1907` (`wrap_folded`) and `2709-2710`;
  `lib/codegen/decomp/decomp_op.ml:101-117` and `149-162` (`at_width`,
  `power_of_two`, `shift_of`); `test/gen/tinygrad.patch`, which makes
  tinygrad's rules the same before the goldens are generated.
- **Differs:** a weak result of folding two weak constants under an
  operation on a committed integer, past its width, is wrapped to it, as the
  operation reads it: `(x * 65536) * 65536` on int32 is `x * 0`. A product by
  a power of two reads the constant modulo `2^bits`, and a quotient or
  remainder reads it in the operand's type, before it becomes a shift or a
  mask: no shift reaches the width, and on int32 a division by `2^31`, the
  least int32, is no shift.
- **Reason:** (b). rune2's compiled `Nx.square (Nx.mul c y)` at int32 and
  uint32, with `c` 65536 or the least int32, gave values that varied by run
  where eager gives 0: rune2's `jit gives eager's bits for exact functions at
  every dtype` left such functions out.
- **Pinned by:** the `Symbolic` suite: `integers wrap › a square of a
  product folds its constants at their width`; the `Decomp_op` suite:
  `late_patterns › a product by a power of two shifts by less than its
  width` and `› a quotient by a power of two reads it in its operand's
  type`; the `Codegen` suite: `constants past the width › a square of a
  product whose constant squares to 0 is 0`.

## D162. A divisor past its operand's width is read in its operand's type

- **tinygrad:** `codegen/decomp/op.py:54-62` (`floordiv_to_idiv`,
  `floormod_to_mod`) and `97-103` (the `fast_idiv` rules), which read a weak
  constant divisor's exact value where the division reads it in its operand's
  type: the sign test skips the correction and `fast_idiv` divides by the
  wrong number. Repro on the CPU: `x = Tensor([0, 1, 7, 100, 127],
  dtype=dtypes.int8).contiguous().realize(); (x // 128).tolist()` gives
  `[0, 0, 0, 0, 0]` for `[0, -1, -1, -1, -1]`, and on uint8 `x // 259` gives
  `[0, 0, 0, 0, 0]` where `x % 259` gives `[0, 1, 1, 1, 1]`.
- **tolk:** `lib/codegen/decomp/decomp_op.ml:118-128` (`divisor`) and its
  uses in `floordiv_to_idiv`, `floormod_to_mod` and the `fast_idiv` rules;
  `test/gen/tinygrad.patch`, which makes tinygrad's rules the same before the
  goldens are generated.
- **Differs:** a weak integer divisor past a committed integer operand's
  width is wrapped to the operand's type before a division or remainder
  lowers: on int8, 128 is -128 and 200 is -56, and on uint8 259 is 3.
- **Reason:** none of (a) to (c): no raven call site is known to reach it.
  It needs the maintainer's ruling, as D160 had: the gap gives a wrong value
  with no error, and the change is a guard in the rules that read the
  divisor, in tinygrad's place.
- **Pinned by:** the `Decomp_op` suite: `late_patterns › a quotient by a weak
  divisor past its width reads it in its type`; the `Codegen` suite:
  `constants past the width › a weak divisor past the width divides as its
  value there`.

## D163. A value read at several positions inlines at most 1024 nodes

- **tinygrad:** `schedule/rangeify.py:50-102` (`remove_bufferize`, which
  inlines a stored value that reads at most three buffers and no reduction
  reading a buffer, at every position it is read at) and `:351-358`
  (`get_kernel_graph`).
- **tolk:** `lib/schedule/rangeify.ml:75-123` (`reads`, `stage_reads` and
  `reads_of`: the distinct indices each stage is read at), `:126-137`
  (`inline_budget`, `copied`), `:205` (`duplicates`, in `remove_bufferize`)
  and `:791-808` (`pm_cleanups`, `get_kernel_graph`, which counts the reads
  before the cleanups); `test/gen/tinygrad.patch`, which gives tinygrad the
  same rule.
- **Differs:** a stage read at `r > 1` distinct indices stays stored when `n
  * r > 1024`, where `n` counts the nodes of its value, its producers
  already inlined, that its ranges reach: those an index copies, since the
  substitution leaves the others shared. Inlined, a value is copied to each
  index it is read at, and the copies compound: a value whose readers are
  inlined at several positions in turn is copied again with them, so `k`
  values each read at two positions are `2^k` copies of the first, in one
  kernel. tinygrad has the same exponential behaviour. With the bound an
  inlining copies at most 1024 nodes into its readers, and a program of `N`
  nodes and `S` stages makes kernels of at most `N + 1024 S` nodes, the
  reads the cleanups make aside. The reads are counted once, in a walk of
  the graph the cleanups start from, at each axis of a stage's ranges; a
  stage's count is the least over its axes, since stages over one device
  range share that axis, so stages with no other range pool their reads and
  may stay stored when read once. A read the cleanups make is not counted,
  and a read through a stack is not one: every stage has a device by then,
  so no cleanup turns it into a read of the stage. A value read at one
  position, or whose copies stay under the bound, is inlined as before:
  elementwise and stencil fusion, whose values are tens of nodes read at a
  few positions, keep their kernels, and no other golden changes. The bound
  is where a copy costs as much as a kernel: on fudge a kernel's compile
  costs about 30 ms, mostly Clang's, and a kernel of a thousand nodes about
  50 ms in all, growing faster than its nodes. It is also past where storing
  the value is cheaper to run: a thousand operations an element against one
  store and `r` loads. XLA's fusion bounds the same duplication by the
  copies alone, at 15
  (`FusionNodeIndexingEvaluation::kAllowedCodeDuplication`).
- **Reason:** (b). rune2's compiled `Nx.associative_scan`, whose prefix
  circuit reads each level's pairs and results at two or three positions:
  jera's tridiagonal solve of 100 equations, three scans one of them over
  4-tuples, compiled cold in 62 s and compiles in 2.4 s (fudge, under load),
  as a scan of 1000 4-tuples does; a scan of 4 is 0.05 s either way.
  tinygrad compiles the same graph as slowly, 3.4 s at 32 elements. The
  kernels left are about 30 ms each to compile, so the solve stays slower
  than storing every level by hand (1.3 s).
- **Pinned by:** the Rangeify suite's `read_at_offsets` (a value of some
  forty nodes read at 32 offsets, stored) rows of `kernel_counts.golden` and
  its kernel golden, from the equally patched tinygrad; `cost › levels each
  read twice make kernels that stop growing with depth`; the `copies` group:
  `a value of 1200 nodes read at one index is inlined`, `› a value read
  twice whose copies are 1024 nodes is inlined`, `› a value read twice whose
  copies are 1028 nodes is stored`, `› a value of 600 nodes outside its
  ranges read twice is inlined` and `› reads of a stage sharing a range are
  not the other's`; rune2's `classes › an associative scan over a tuple
  gives eager's bits` and the `scan/tridiagonal-100` row of its compile
  bench.

## D164. An accumulator's read waits on its initial store

- **tinygrad:** `codegen/__init__.py:197-204` (`reduce_ranges_to_acc`), whose
  read of the accumulator after the loop is ordered after the loop's END
  alone, with `uop/symbolic.py:454` (`sym`'s first load/store folding rule, a
  store of a load of its own index to `NOOP`).
- **tolk:** `lib/codegen/codegen.ml:730-748` (`reduce_ranges_to_acc`);
  `test/gen/tinygrad.patch`, which makes tinygrad's the same before the
  goldens are generated.
- **Differs:** the read after the loop is ordered after the initial store as
  well as the END. In tinygrad, when every lane of an accumulation folds to
  adding 0, which devectorizing shows only lane by lane after the accumulator
  is built, `sym` folds each store of the loop to `NOOP`: the END holds no
  store, nothing orders the read after the initial store, which is then
  removed, and the program reads registers it never wrote. tinygrad's later
  `ef6ec5f15` (#18858) keeps a folded store's sources as an AFTER on its
  `NOOP` instead, which its specification at the pinned commit rejects; the
  entry goes when the pin passes it.
- **Reason:** (b). rune's compiled Fold (`dev/rune2/lib/lower_index.ml`,
  `fold`) of an int16 `[3; 3]` by `lo [0; -1]`, `interior [0; 1]` and two
  windows of step 2 sums no term into some elements, and it read uninitialized
  registers there, where nx2's Fold contract gives 0: rune2's `a fold wraps
  integers and holds its float bound` failed.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `an
  accumulator no lane adds to › is stored before it is read` and `› sums to
  zero`.

## D165. A weak maximum's identity is its accumulator's least value

- **tinygrad:** `uop/ops.py:61` (`identity_element`), whose `Ops.MAX` identity
  is `dt.min`: for `weakint`, of bitsize 800 (`dtype.py:119`), `-2**799`.
- **tolk:** `lib/uop/ops.ml:1212-1219` (`identity_element`);
  `test/gen/tinygrad.patch`, which makes tinygrad's the same before the
  goldens are generated.
- **Differs:** a maximum over a weak type has as identity the least value of
  `Dtype.strong` of the type, the width its accumulator is allocated at. In
  tinygrad a weak integer's maximum, gated by an Invalid, takes `-2**799`:
  `pm_reduce_identity` selects it where the gate fails, and committing the
  selection's width (`cast_weak_srcs`, `commit_int`) raises "does not fit any
  int". A weak maximum arises where a pad's bare fill is all that remains of a
  committed maximum's operand.
- **Reason:** (b). rune's compiled Fold (`dev/rune2/lib/lower_index.ml`,
  `fold`) of a float64 `[2; 2; 2]` by `lo [2; 2; 0]`, `hi [0; -1; -1]`,
  `interior [1; 0; 0]` and three windows maximises the uint8 marks of each
  element's terms; where no term reaches an element, the mark is the pad's
  fill, a weak `0`, and compiling raised: rune2's `a fold wraps integers and
  holds its float bound` failed.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `a maximum of
  weak integers › is the greatest value it reduces`; `Tolk.Ops › data types ›
  identities.golden`.
