# Divergences from tinygrad

tolk follows tinygrad `79af1ca70e7021f504919c4ff5631245acc33ed6`. This
ledger lists every place where it does something else. An entry is admitted
for one of three reasons only:

- **(a) an OCaml constraint:** acyclic modules, static types, the GC, domains;
- **(b) a named consumer:** a raven call site that fails without it;
- **(c) the executor's contract:** nx.device's submission protocol, which
  `tolk.engine` drives with the compiler's output.

Taste, speed without a measurement, and "the old tolk did it" are not reasons.
A difference in numerics also needs a failing rune test, since nx semantics
belong in rune's lowering.

Each entry gives the tinygrad file and line, the tolk file and line, what
differs, its reason, and the test that pins it. An entry whose test does not
exist yet names the layer that brings it; the entry is rejected at that
layer's review if the test is still missing. An entry goes when its reason
goes. Entries are in the order of their numbers, and a number is never
reused, since tests cite them. Keeping only part of a file is scope, recorded
under Exclusions in `README.md`, not a divergence.

## D1. Timeline values are parameters

- **tinygrad:** `runtime/support/hcq2.py:65-66` (`timeline`, `timeline_value`),
  `:231-239` (the batch slots, with the slot of the timeline), `:256`, `:276-278`
  and `:415-434` (`hcq_fence`).
- **tolk:** `lib/runtime/support/hcq2.ml:212` (`signal_word`), `:224`
  (`submitted`), `:225` (`value`), `:638` (`make_ctx`), `:748` (`start_ins`),
  `:808` (the bump) and `:1115` (`hcq_fence`).
- **Differs:** a device's timeline is its signal word alone, one word. The value
  of the work submitted before a batch and the value the batch signals are
  variables of its host program, `submitted d` and `value d`, which the engine
  binds on each run (`Nx_device.submitted`, `Submission.value`), where tinygrad
  loads the submitted value from the timeline's second word and writes it back
  bumped. The host program neither reads nor writes a submitted value: its queues
  wait for `submitted d` and signal `value d`. The fence keeps only its re-arming
  of the queue signals, ordered after the run's `submitted d`, as tinygrad's is
  after the timeline's loads, so that it runs on every run; its wait for the
  batch's previous run, and its record of the value that run signals, are the
  engine's (`Submission.wait`), before it writes the address table and calls
  the host program, so the slots lose the timeline's slot. Runs of one linked
  batch stay serialized, as RFC 0012 requires. With no load of a timeline in
  a host program, Metal's `pm_lower` (`runtime/ops_metal.py:192-194`), which
  reads a timeline's first word from the device's event (`mtl_poll`), has
  nothing to lower and is not ported.
- **Reason:** (c). nx.device's runtime writes the submitted value and never
  publishes it (RFC 0011, Amendment 2), and `Submission.wait` has the timeout,
  `Lost` and Metal's completion that a spin in the host program lacks.
- **Pinned by:** the Hcq2 suite (`test/runtime/support/hcq2`): `timeline
  values (D1)`, whose recorded batches and host programs are tinygrad's with
  D1 applied by their generator (`gen/runtime/support/hcq2.py`), and `linking
  and running › a run waits for its batch's previous run before it rewrites
  the batch's memory`; and the Engine suite (`test/engine/tolk_engine`):
  `batches › each run of a batch signals its device's next value once`.

## D2. Withdrawn

tinygrad matches patterns without generating code when `UPAT_COMPILE` is 0
(`uop/ops.py:1545`, `upat_interpret`), and `UPat`, `PatternMatcher` and
`graph_rewrite` live in `uop/ops.py`. tolk ports that path in
`Ops`, so leaving out `uop/upat.py`, the pattern compiler, is scope: see
the Exclusions of `README.md`.

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk:** `lib/device.ml` (the compiler half of `device.py`);
  `lib/runtime/ops_metal.ml`, `ops_cuda.ml`, `ops_amd.ml` and `ops_nv.ml` (the
  IR half of each `ops_*.py`); `engine/tolk_engine.ml:56` (`device`) and
  the vendors `engine/metal.macos.ml`, `cuda.ml`, `amd.ml` and `nv.ml`;
  `lib/uop/ops.ml:229` (the type `param_arg`), `:3225` (`param_arg`), `:3253`
  (`new_buffer`).
- **Differs:** tolk holds the compiler half: `Compiler`, the renderer
  and compiler selection of `Compiled`, and the IR half of each `ops_*.py`
  (queues, `pm_encode`, program data), all returning data. The lazy `Buffer`
  and running a schedule are `tolk.engine`'s, which has no registry of
  devices by name: its caller maps each name to a device; allocators,
  programs, drivers and profile
  events are nx.device's. So a node of the compiler holds no runtime state:
  `ParamArg` has no `buffer`, a `BUFFER` is named by its slot for the engine
  to bind, and `UOp.buffer`, `realized`, `is_realized`, `_buffer_view`,
  `_base_buffer_is_realized`, `from_buffer` and `_frompy`
  (`uop/ops.py:856-993`) are the engine's. Whether storage is bound is the
  operation itself, `BUFFER` or `ALLOC`, so the specification's checks of
  `ParamArg.buffer` (`uop/spec.py:149,153,266,268`) hold by construction.
- **Reason:** (c).
- **Pinned by:** for `Ops`: `Tolk.Ops › storage › new_buffer takes the
  next slot without one` and `Tolk.Ops › arguments › reprs.golden`, where
  a `BUFFER` prints and interns by its slot alone; for the queues: the
  `recorded cases` of `Tolk.Ops_metal`, `Tolk.Ops_cuda`,
  `Tolk.Ops_amd` and `Tolk.Ops_nv`, which encode each vendor's
  queues as nodes on a machine without its device; for the engine: the Engine
  suite (`test/engine/tolk_engine`): `device › a name the map does not
  hold is refused` and `refusals › link refuses a device the map does not
  hold`, which a registry would resolve, and `refusals › run refuses a
  parameter it binds no buffers`, which a `BUFFER` holding its buffer would
  run.

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
- **tolk:** for `uop/ops.py`, `lib/uop/ops.ml:1685` (`simplify_hook`),
  `:4614` (`Private`),
  `:1190` (`Make_elementwise`), `:816` (`repr`), `:244` (`bufferize_opts`),
  `:262` (`Calls`); `lib/uop/render.ml:202` (`render`), `:212` (`srender`);
  `lib/renderer/renderer.ml:119` (`Compiler`); `lib/schedule/prepare.ml:709`
  (`contiguous_view`); `lib/schedule/schedule.ml:164` (`pm_flatten_linear`);
  `lib/codegen/codegen.ml:621` (`apply_opts`); `lib/engine/realize.ml:117`
  (`lower_and_compile`); `lib/runtime/support/hcq2.ml:416` (`device`),
  `:1875` (`pm_beam`), `:1892` (`compile_linear`); and
  `lib/runtime/support/compiler_metal.ml` (`Compiler_metal`).
- **Differs:**
  - the `UOp` methods that call a later module become functions of that
    module: `contiguous_view` and its matcher go to `Prepare`,
    `to_elf` to `Device`, and `render` and `srender` to `Render`;
  - `engine/realize.py`'s `pm_flatten_linear`, which `schedule/__init__.py`
    imports, is `Schedule.pm_flatten_linear`, since `Realize` follows
    `Schedule`;
  - `simplify` stays in `Ops`, since reshaping, `resolve` and shapes call
    it: it rewrites with the `symbolic` matcher that `Symbolic` installs when
    the library is initialised. It is set once; `lib/dune` links the library
    whole (`-linkall`), so it is set before any program runs, and `simplify`
    raises if its rules are missing. A sink of constants and stacks
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
  - `apply_opts` takes the optimiser as an argument, and `Search` follows
    `Postrange` and takes how to link and time a program as its `link`
    and `time` arguments, where `beam_search` takes `rawbufs` and
    `var_vals`: its caller makes the buffers `args_from_ast` describes
    (`Tolk_engine.slots`), since `tolk` cannot name nx.device's buffers,
    and `get_kernel_actions` returns each candidate with its action, without
    tinygrad's `include_0` and its positions in the action table;
    `Codegen.full_rewrite_to_sink` and `Codegen.to_program` take
    the beam search as their `beam` argument, a function of the width the
    kernel asks for, and raise when a kernel asks for one and none is given;
  - the engine has one order (schedule, realize, hcq2, jit), and each late
    binding is passed as a function argument, never a global reference;
    `Realize` precedes `Hcq2`, as `hcq2.py` imports `realize.py` at its top;
    `realize.py`'s `compile_linear` (`:271`), which reaches `hcq2.py` through
    the import at its bottom, is `Hcq2.compile_linear`, with `pm_beam`, its
    one reader; the device's queue encoders, which `hcq2.py` finds in the
    device registry (`Device[d].pm_encode`, `has_copy_queue`, `host`), are
    given by the caller in its description of each device (`Hcq2.device`);
  - the compiler modules precede `Cstyle`, and `ops_metal.py`'s
    `MetalCompiler`, which `cstyle.py` imports inside `MetalRenderer`, is
    `Compiler_metal`, one of them, since the rest of `ops_metal.py`,
    `Ops_metal`, follows `Hcq2`.
- **Reason:** (a). Each layer's review checks that its breaks are the
  smallest possible.
- **Pinned by:** for `Codegen`:
  `Tolk.Codegen › beam search (D4) › a kernel that asks for a beam of
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
  given`, `Tolk.Hcq2 › compile_linear › makes each kernel ask for a beam
  of the width BEAM sets`, which passes the search, and `› copies through the
  halves of a staging buffer of the host where the queues cannot reach`, whose
  caller describes the devices and their queues; for `Compiler_metal`:
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
  `lib/runtime/support/compiler_amd.ml` (`run`) and
  `compiler_amd_worker.c`.
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
  of a process on a mutex of its own, so each comgr compile runs in a
  process of its own, spawned by the compiling domain, where a tinygrad
  worker process loads comgr once and compiles in it: a small C program that
  tolk carries, which runs from memory on Linux and from a file in
  `Helpers.cache_dir` elsewhere. It reads its request and writes its reply
  through files, and a compile whose process ends without a reply raises
  `Compile_error`, where it would end tinygrad's worker.
- **Reason:** (a).
- **Pinned by:** `Tolk.Setting › context › is not seen by the other
  domains` and `Tolk.Setting › context › binds for the domains spawned
  while it runs`; `Tolk.Worker` (every test); `Tolk.Codegen ›
  programs are kept › calls from several domains at once make one program,
  compiled once (D5)`; `Tolk.Compiler_metal › MTLCompiler › compiles
  from several domains at once`; `Tolk.Compiler_amd › a compile in a process
  of its own` (every test) and `› a process that cannot start`.

## D6. Devices are named, never parsed

- **tinygrad:** `device.py:26,36,395,491` (device strings split at `:`).
- **tolk:** `lib/device.ml:121` (`renderers`), `:144` (`renderer`).
- **Differs:** the caller gives a target, whose device is a kind and whose
  `arch` is the device's; tolk picks the renderer and compiler from the
  target and never parses the name, with one exception: a name starting with `DISK` is a disk, as
  tinygrad reserves it and nx.device names its disk devices
  (`Ops.on_disk`, `is_disk_device`, `copy_to_device`, `clone`).
- **Reason:** (c).
- **Pinned by:** for targets, `Tolk.Device › renderer takes a device's
  name as it is (D6)` (a name with an index, in lower case, or a disk's, is no
  device, and a name gives no renderer or architecture) and `Tolk.Device ›
  renderer › picks a device's renderer as tinygrad does` (the architecture
  comes from the target); for the disk, `Tolk.Ops › several
  devices › on_disk holds for one disk device`, `Tolk.Ops › several
  devices › copy_to_device rejects a disk and a weak type` and `Tolk.Ops ›
  storage › clone rejects a disk`.

## D7. Stamp slots follow `Submission.record`

- **tinygrad:** `runtime/support/hcq2.py:231,240,301`.
- **tolk:** `lib/runtime/support/hcq2.ml:713` (`stamps`), `:638` (the
  slots); `lib/runtime/ops_metal.ml` (`submit`), which writes a command's
  command buffer over words 3 and 5 of the device's slots, where tinygrad's
  are 5 and 7 (`runtime/ops_metal.py:159-160`).
- **Differs:** a device's slots are its queue signals, then two slots per call
  when profiling, with no slot of the timeline between them (D1). A call's two
  slots are adjacent, so its start and end stamps are words 1 and 3 of one
  32-byte record, which the engine hands to `Submission.record`.
- **Reason:** (c). `Submission.record` takes that layout (RFC 0011,
  Amendment 2, GAP-3).
- **Pinned by:** the Hcq2 suite (`test/runtime/support/hcq2`): `stamp slots
  (D7)` and `linking and running › a profile records a span of each kernel on
  its device, in order`; and the Engine suite (`test/engine/tolk_engine`):
  `batches › a kernel is a span of its compute lane and a copy of its copy
  lane`.

## D8. The disk cache maps strings to strings

- **tinygrad:** `helpers.py:398-447` (an SQLite database of pickled values,
  keyed by strings, integers or dictionaries of columns).
- **tolk:** `lib/helpers.ml:393-504` (`Diskcache`).
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
  key › tells arguments apart`; for the profile keys of batched kernels, the
  Hcq2 suite (`test/runtime/support/hcq2`): `profile keys (D12)`, whose
  recorded graphs are compared without them.

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
- **tolk:** `lib/uop/symbolic.ml:100` (`fold_const_alu`), `:467` and `:895`
  (the two stage fold of a maximum);
  `lib/uop/ops.ml:1470` (`at_width`) and `:1492` (`operand_bounds`);
  `lib/uop/uop_weak.ml:203` (`uncast_const`).
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
  type's value (D13)` (each case above, evaluated before and after), the law
  `Tolk.Symbolic › laws › sym keeps the value of an integer expression at
  a committed width, wrapping included`, and `Tolk.Symbolic › tinygrad ›
  tests.golden › TestModularWraparound.<test>` and
  `› TestThreefryConstFolding.test_threefry`, whose goldens, generated with
  the same change in `test/gen/tinygrad.patch`, hold the machine values;
  `Tolk.Ops › bounds › a typed integer constant outside its type is
  bounded by its wrapped value, a non-finite one by the type (D13)`; the
  offset's interaction with D24: `Tolk.Symbolic › integers wrap (D24) ›
  an offset crosses a comparison only where neither side wraps` (the
  committed case).

## D14. Folded comparisons treat NaN as IEEE does

- **tinygrad:** `uop/symbolic.py:29` (`fold_const_alu`), whose operands come
  from `UOp.val` (`uop/ops.py:259-263`) as `ConstFloat`s (`dtype.py:8-22`).
  `ConstFloat` makes NaN equal to NaN on purpose, so that NaN constants
  intern as one node, and `exec_alu` compares with it: `nan != nan` and
  `nan < nan` fold to `False`, where `exec_alu` on floats gives `True` and
  `False`.
- **tolk:** `lib/uop/symbolic.ml:100` (`fold_const_alu`); constants are
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
  first compile, once per process (comgr: by each compile's own process, D5),
  and a compile without it raises `Compile_error` with the reason.
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
- **Pinned by:** `Tolk.Cstyle › float8 infinities on CUDA (D16)`, which
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
- **Pinned by:** `Tolk.Cstyle › narrowing (D17)`, which checks for every
  kernel that the kernel whose tinygrad source is its source is the kernel
  with those casts, and `Tolk.Cstyle › sources`, which compares the 21
  sources it changes with tinygrad's for that kernel; `Tolk.Cstyle ›
  execution on the host › wraps each operation on unsigned chars (D17)` and
  `› rounds each operation on halves to a half (D17)`; and the slow
  `Tolk.Cstyle › execution on the host › a kernel over a narrow type wraps
  and rounds as the interpreter (D17)`, over each type's whole range.

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
- **Pinned by:** `Tolk.Cstyle › bfloat16 truncation on Metal (D18) ›
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
  nodes_to_one_device_handled.golden, landed on its device (D19)`, which
  states tinygrad's golden in code with each gather of a chunk replaced by its
  copy to the target, and `Tolk.Allreduce › rules › a hierarchical
  allreduce to one device lands there (D19)`, which evaluates the value and
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
  divides each sharded axis by its own count (D21)` and `› a reshape of a mesh
  of 2 by 4 devices keeps each tile (D21)`, and the law `Tolk.Multi ›
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
  position (D23)`, and the law `Tolk.Multi › multi_pm › laws › a
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
    (`x + y <> c` under a cast). tolk: `lib/uop/symbolic.ml:862,871`,
    `lib/uop/divandmod.ml:252`, `lib/codegen/simplify.ml:252,268,276,344`.
    Each applies to a committed integer only where every value it computes
    fits the type; the comparisons of `Simplify` apply to integers only, since
    moving a float term rounds, and `x + y <> c` only under a cast that does
    not narrow. tinygrad folds uint8 `u - 1 < 255` for every `u`, int32
    `(w // 2 + 2^30) // 2` at `2^30` to `-268435456` where the machine gives
    `805306368`, and `(w // 65536) // 65536` to a division by `2^32`, which
    wraps to `0`.
  - **Float folds.** tinygrad: `uop/symbolic.py:117` (`x + 0`), `:170-176`
    (`x / x`, `(x * y) / y`, `x * 0`), `:247` (`(x / y) / z`). tolk:
    `lib/uop/symbolic.ml:348,521`. A float `x + 0` is `x` only for `-0.`
    (`-0. + +0.` is `+0.`); `x * 0` is `0` for integers and booleans only (a
    float product by zero is NaN at an infinity or a NaN and `-0.` at a
    negative `x`); `x / x`, `(x * y) / y` and `(x / y) / z` are gone, since
    true division is always a float: `x / x` is NaN at 0, `(1e10 * 1e30) /
    1e30` is `inf`, and `(1e20 / 1e20) / 1e20` is `1e-20` where
    `1e20 / 1e40` is `0.`.
  - **Signed zeros.** tinygrad: `uop/symbolic.py:248` (`-(x + c)`), `:267`
    (complementary selections), `:472` (`-(x + y)`). tolk:
    `lib/uop/symbolic.ml:708,776,1316`. For integers and booleans only:
    `-(x + 3)` at `x = -3` is `-0.`, where `-x + -3` is `+0.`, and
    `where c t 0 + where c 0 f` at `t = -0.` is `+0.`, where `where c t f` is
    `-0.`.
  - **Reassociation.** tinygrad: `uop/symbolic.py:240-246` (like terms),
    `:264-265` (a sum of two selections), `:279-280` (two constants of an
    associative operation), `:293-294` (constants to the end), `:390-398,470`
    (`reduce_mul_chain`). tolk: `lib/uop/symbolic.ml:683,764,845,901,1156`.
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
    `lib/uop/symbolic.ml:808,828`. A maximum by bounds applies to integers
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
    (`c ** x`). tolk: `lib/uop/symbolic.ml:61,576`. The reciprocal of the
    base is taken for exponents of magnitude at least 1 only, where the power
    overflows whenever the reciprocal does: `1e-40 ** -0.8` is `1e32`, where
    `(1 / 1e-40) ** 0.8` is `inf`. A float half-integer power selects `+0.`
    at `-0.` and `+inf` at `-inf`, where `sqrt` gives `-0.` and NaN. `c ** x`
    goes to `xpow` for `c = inf`: `inf ** 0` is `1.`, where
    `exp2 (0 * log2 inf)` is NaN.
  - **Simplify's reductions.** tinygrad: `codegen/simplify.py:104-111`
    (`sum_between`) and `:118` (a product by a boolean cast). tolk:
    `lib/codegen/simplify.ml:231,326`. A float sum counted in closed form is
    `+0.` over an empty part of its range, where `0 * inf` is NaN; a product by
    a boolean cast is a selection for integers only (`-1. * 0.` is `-0.` and
    `inf * 0.` NaN, where the selection gives `+0.`). A sum over a range its
    value does not use, `x * n`, and a sum of a sum split in two stay: each is
    the sum of the same terms in another association, which RFC 0012's rounded
    sum class allows; a sum of `-0.` terms computed as `-0. * n` is `+0.` once
    the lowering adds `+0.` to every float sum's result, which the `x + 0` fold
    now keeps.
  - **Transcendental polynomials.** tinygrad: `helpers.py:153` (`polyN`, from
    `0.0`). tolk: `lib/codegen/decomp/transcendental.ml:375` (`poly_n`). A
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
  graphs of each kind against eager nx.
- **Pinned by:** each facet by value tests, which evaluate a graph with the
  reference interpreter before and after the rewrite at bindings where the
  restricted rewrite changes the value:
  - integer bounds: `Tolk.Ops › bounds › a committed integer that can
    leave its type has its bounds (D24)`, `› an integer cast to a signed type
    it leaves wraps (D24)`, `› a constant table holding a NaN has its type's
    bounds (D24)` and the law `› bounds hold the value a committed integer
    wraps to (D24)`;
  - wrapping rules: `Tolk.Symbolic › integers wrap (D24)` (every test);
    `Tolk.Divandmod › nested divisions › a committed division stays
    where x + a * c wraps (D24)`; `Tolk.Simplify › keeping values (D24)`,
    its comparison tests;
  - float folds, signed zeros, reassociation, maxima, reciprocal and sigmoid
    forms, pow: `Tolk.Symbolic › floats keep IEEE values (D24)`, one test
    per facet, with `› a Dekker split and product keep their low parts` and
    `› a float selection that computes a maximum stays`, the structural tests
    marked D24 in `symbolic_simple`, `symbolic` and `sym`, and the law
    `Tolk.Symbolic › laws › sym keeps a float expression's value bit for
    bit at special values`;
  - Simplify's reductions: `Tolk.Simplify › keeping values (D24) › a
    float sum over an empty part of a range is +0., whatever the value` and
    `› a float product by a boolean mask keeps its value`, and
    `Tolk.Simplify › pm_reduce_collapse › a float product by a comparison
    cast from a boolean stays (D24)`;
  - transcendental polynomials: the `Tolk.Transcendental` goldens,
    generated from the patched tinygrad, whose value tables are unchanged;
  - tensor-core accumulators: the sources of every tensor core,
    `Tolk.Cstyle › sources › by default › metal_tc_*`, `hip_tc_*` and
    `metal_matmul`, whose loops add nothing to the accumulator, as tinygrad's;
    `Tolk.Codegen › tensor-core accumulators (D24) › the running sum
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
  `exec_alu`); `lib/uop/symbolic.ml` (`pm_data_invalid`'s `Mulacc` rule).
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
  MTLCompiler › a product and a sum compile under the no-contraction pragma
  (D25)`. On an Apple GPU, the same kernel compiled by MTLCompiler gives
  `2^-24` without the pragma and 0 with it, and still `2^-24` with
  `-ffp-contract=off` alone. NVRTC and HIP are README's hardware checks.
  `Tolk.Codegen › multiply-adds (D25) › a sum of products adds each into
  its running sum rounded once, of 2` and `› of 64` (the sum of `-(1 +
  2^-11)` and `(1 + 2^-12)^2` is `2^-24`, with a multiply-add in the source)
  and `› a sum of products unrolled beside upcast lanes adds each into its
  running sum rounded once` (four running sums of a loop unrolled by 4, three
  multiply-adds each, and the same `2^-24`) and `› a product and a sum outside a reduction are not
  fused`; the Codegen and C-style goldens, from tinygrad with D25 applied by
  its generator.
  `Tolk.Ops › exec_alu › a float multiply-add folds rounded once (D25)` (its
  float32 case past a double's rounding ties to the wrong float32 when the sum
  is not rounded to odd); `Tolk.Symbolic › invalid values › a multiply-add
  moves inside the gate of each operand (D25)` and `› a multiply-add of
  invalid is invalid (D25)`. In rune, `Test_lower_arith › multiply-adds › a
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
  `:647` (`const`); `lib/uop/symbolic.ml:82` (`fold_bitcast`);
  `test/gen/tinygrad.patch`, which applies the same rules to tinygrad before
  the goldens are generated.
- **Differs:** a float constant is its bits. `Dtype.const` keeps a NaN's sign
  and the payload its type holds, a signalling NaN staying one, and constants
  compare and hash by their bits, so NaNs of different bits are different
  nodes. `fold_bitcast` reads a float constant as its type stores it, where
  tinygrad's conversion quiets a signalling NaN and makes an 8-bit NaN
  canonical. So folding `bitcast(bitcast(0x7d01, half), uint16)` gives
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
  are different nodes`; `Tolk.Ops › exec_alu › an invalid operation's
  NaN is the canonical positive quiet NaN, whatever the host gives (D27)`,
  which compares bits and fails on an x86 host without the canonical NaN; `Tolk.Symbolic › symbolic_simple › casts › a
  bitcast round trip of every 8- and 16-bit word folds to its value`, which evaluates each graph
  with the reference interpreter before and after `simplify`;
  `Tolk.Dtype › const › const keeps the bits of every 8- and 16-bit float
  word`.

## D28. A host program has nx.device's entry

- **tinygrad:** `codegen/__init__.py:449-451` (`do_compile`: the binary is the
  source compiled), `runtime/ops_cpu.py:58-70` (`CPUProgram.__call__` calls
  the kernel's own signature through ctypes, every argument a `c_uint64`).
- **tolk:** `lib/codegen/codegen.ml` (`host_entry`, `do_compile`).
- **Differs:** a program for a CPU target is compiled from its source with
  the kernel renamed `NAME_` and an entry `NAME` appended:
  `void NAME(void **b, const long long *v)`, which passes the buffers, in the
  order of the program's globals, and the variables, in order, on to the
  kernel, as C converts them to its parameters' types. The program's source is
  the renderer's; only its binary differs.
- **Reason:** (c). nx.device calls a host program with one ABI,
  `Nx_device.Program.call`'s, and names its profile span after the entry.
  The entry keeps the kernel's name in profiles, and passes each scalar at its
  own type: an Apple arm64 stack argument of 32 bits takes 4 bytes, where
  ctypes' `c_uint64` would take 8.
- **Pinned by:** the `Codegen` suite, `programs › a host program's binary
  holds nx.device's entry after the kernel` and `› a host program runs
  through Program.call, with ten scalars past the argument registers`.

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

## D30. A range around calls is a loop in its batch

- **tinygrad:** `schedule/__init__.py:72` (linearization drops a call's
  `END`); `runtime/support/hcq2.py:40-48` (`get_enqueue_devs` enqueues calls
  only), `:357,379-385` (`HWQueue.loop` repeats the command bytes and their
  words), `:467-476` (`bufferize_cmdbuf` merges each nested linear once) and
  `:50-54` (`unwrap_view` reads constant offsets), `:248-249` (`_wait_ins`:
  an NV compute queue that waits also waits for its previous launch);
  `runtime/ops_metal.py:103`
  (`MetalQueue` inherits `HWQueue.loop`, whose bytes it does not use);
  `uop/weak.py:27-33` (`cast_weak_srcs`).
- **tolk:** `lib/runtime/support/hcq2.ml`: `range_placement`, which
  `stages` reads, and `item` in `sched_batches`, the positions and the two
  visits and the loop edges of `make_ctx`, `Queue.loop`, the copies per trip in
  `bufferize_cmdbuf`, the ranges of each group in `patch`, the moving offsets
  of `lower_call`, the halves of `sched_batches`, and the kernels not
  compiled yet of `get_enqueue_devs`; `lib/runtime/ops_metal.ml`,
  `lib/runtime/ops_cuda.ml`, `lib/runtime/ops_amd.ml` and
  `lib/runtime/ops_nv.ml` (`loop`).
- **Differs:** an `END` of ranges around calls that are all enqueued, on
  devices of one kind, belongs to their batch. Each queue its calls run on
  loops over its commands of one trip, as `HWQueue.loop` does, and a nested
  linear whose words read the ranges, directly or through the address of a
  nested linear that reads them, such as a kernel's arguments or a launch
  descriptor that addresses them, has a copy for each trip, which the trip's
  words address. A call's position counts
  every run of the calls before it, and a call waits for the calls on other
  queues it depends on in the current trip and, after it, in the trip before
  (for the value `0` in the first trip). Each group of patched words loops
  over ranges of its own, since a program ends a range once. Metal repeats a
  loop's indirect commands and their arguments, CUDA's host program loops
  over its launches, each trip's reading its trip's extra words, and AMD's
  AQL queue repeats a loop's packets, each trip's running its trip's bytes of
  the command buffer (tinygrad's `AMDComputeAQLQueue` keeps its packets apart
  from the bytes `HWQueue.loop` repeats). NV's compute queue ends its chain of
  launches at a loop's edges, each trip's chaining onto its own descriptors,
  and its channel runs the chains it schedules at once: so a call on it past
  a loop's edge also waits for the queue's previous call, as tinygrad's
  `_wait_ins` has a call do where a wait ends its chain.
  - An address that moves with a range (its storage's plus the view's offset)
    and a position a signal stores are 64-bit words cast from the range's weak
    integers, which D44 computes in integers. tinygrad's loop builds no such
    word: its offsets only index the command buffer.
  - A range whose calls none is enqueued stays a range around them, reading
    `range_value` variables that the engine binds on each trip; one that
    mixes the two, or two kinds of device, is refused. `get_enqueue_devs`
    takes a kernel not compiled yet as its program, so that `stages` answers
    on a schedule before it compiles.
  - A batch whose submission a queue cannot hold, such as AMD's AQL ring's
    share, whose vendor raises `Over_capacity`, runs as two batches, one after
    the other: its calls halved, or a range's first trips and the rest, each a
    range of its own, until each part fits. A staged scan then takes one
    submission for each share of the queue that holds it; `stages` does not
    decline for size.
  - `n` trips of `k` calls take `n·k` commands and `n·k` copies of their
    arguments, made at link: a command and its arguments take 160 bytes on
    the NULL queues and 264 on Metal. A range of more calls than a chunk runs
    as chunks of its trips (D67), so its batches hold at most two chunks'.
- **Reason:** (b). `Rune.scan`'s staged loop (RFC 0012) schedules calls
  inside a range and runs them as one submission, whatever the trip count
  (Law 6); tinygrad's scheduler never hands `hcq2.py` a range.
- **Pinned by:** the Hcq2 suite (`test/runtime/support/hcq2`): `ranges
  (D30)`, among them `› a range its queue cannot hold in one submission runs as
  several`, `› a call its queue cannot hold in one submission is refused`, `› a range with a host program, a host copy or two kinds
  of device does not stage`, `› a run of a batched range is one submission,
  whatever its trips`, `› a trip's copy waits for the kernel of the trip
  before, on another queue`, `› on NV a compute queue's call past a loop's
  edge waits for its previous call`, `› a ranged batch's addresses are integers,
  profiled or not` and `› a trip reads its window past what a float offset
  holds`, and `Deps › a write that does not trim keeps the accesses to the
  bytes it writes`, `› forgotten accesses are no longer followed`; the
  Ops_metal suite (`test/runtime/ops_metal`): `loops (D30) › a range's
  addresses are integers, profiled or not`, and on macOS `execution › each
  trip of a range runs its kernel on its own window`, `› a range of 20000
  trips runs from one indirect command buffer` and `› a profiled range
  records a span of each trip's kernel` (slow); the Ops_cuda suite
  (`test/runtime/ops_cuda`): `loops (D30) › a range is a loop of the host
  program around its launches` and `› a range's addresses are integers,
  profiled or not`; the Ops_amd suite (`test/runtime/ops_amd`): `loops
  (D30)`; the Ops_nv suite (`test/runtime/ops_nv`): `loops (D30) › each
  trip's two launches chain on descriptors of the trip's own, and the channel
  schedules each trip's chain`.

## D31. Payne-Hanek reduces exactly, to the nearest quadrant

- **tinygrad:** `codegen/decomp/transcendental.py:66-113`
  (`payne_hanek_reduction`) and `:164-166` (`sin_poly_large`).
- **tolk:** `lib/codegen/decomp/transcendental.ml:124`
  (`one_over_two_pi`), `:173` (`turn_fraction`), `:272` (`radians`) and `:313`
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
  word; and the remainder is the signed fraction, summed exactly in two floats
  from parts that convert exactly and multiplied by `2pi/2^64` in two floats
  (D126). The remainder is within an ulp of the exact one in float32 and
  float64, `6381956970095103 * 2^797`, the float64 nearest a multiple of
  `pi/2`, included.
- **Reason:** (b): rune's `sin` and `cos` reduce by `pi/2` with this
  reduction beyond `2^12` in float32 and `2^22` in float64, where their exact
  Cody-Waite parts stop (RFC 0012), and tolk's own `xsin` uses it beyond its
  switch-over; both must give nx's sine there.
- **Pinned by:** `Tolk.Transcendental › reductions ›
  payne_hanek_near_multiples.golden`, remainders and quadrants of float32 and
  float64 angles near multiples of `pi/2`, from pi to 1400 bits, which the
  old reduction fails; `› payne_hanek_reduction removes quarter turns`; and
  the `payne_hanek_*`, `xsin_*` and `values` goldens, from the equally
  patched tinygrad.

## D34. A profiled Metal command buffer waits in its stamps retained

- **tinygrad:** `runtime/ops_metal.py:155-160` (`MetalQueue.submit` writes
  the command buffer over a command's start stamp and `0` over its end
  stamp), `:275-285` (`synchronize` reads their times, then drains the
  autorelease pool that holds the command buffer).
- **tolk:** `lib/runtime/ops_metal.ml` (`selectors`, and the stamps in
  `submit`).
- **Differs:** the host program sends the command buffer `retain` and writes
  what it returns over the start stamp. Before that it sends `release` to the
  start stamp's command buffer if the end stamp is still `0`: a run of the
  batch whose times were never read left it there. A stamp whose times were
  read, or that was never written, is sent to `nil`, which does nothing. The
  selectors `retain` and `release` follow tinygrad's in the `mtl_sel` words.
- **Reason:** (c). nx.device's Metal library reads a command buffer's times at
  the device's next synchronization, which may run on another domain than the
  run that committed it, and releases it then (RFC 0011, Amendment 2): a
  command buffer that only an autorelease pool holds may be freed, and its
  address reused, before that.
- **Pinned by:** the Ops_metal suite (`test/runtime/ops_metal`):
  `recorded cases › chain_profile`, `› one_profile`, whose host programs are
  tinygrad's with D34 applied by their generator
  (`gen/runtime/ops_metal.py`); on macOS, `execution › a profile records a
  span of each kernel on the device, in order` and `› a profiled batch run
  twice keeps the second run's spans`, which releases the first run's command
  buffer (slow).

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

## D36. A CUDA host program reads a function's address from a word

- **tinygrad:** `runtime/ops_cuda.py:23` (`extern`: a `Buffer` of the host
  placed at an address), `:38` (`CUDAQueue.extern`: the address of such a
  buffer), `:51` (a kernel's function) and `:64` (the host function that
  stamps a slot).
- **tolk:** `lib/runtime/ops_cuda.ml` (`extern`).
- **Differs:** a kernel's `CUfunction` and the stamping host function reach
  the host program as words it loads, from placeholders of the device tagged
  `("function", lib, name)` and `"stamp"`, which the engine fills when it
  links the batch, in memory the host reads: a function is loaded on each
  device. tinygrad takes them as the address of a buffer placed at the
  function, which the batch loads from its address table.
- **Reason:** (c). nx.device's buffers are memory: none lies at a function's
  address, and a placeholder's storage is a buffer. A word holding the
  address is how a C function already reaches a host program
  (`Hcq2.ccall`).
- **Pinned by:** the Ops_cuda suite (`test/runtime/ops_cuda`): `recorded
  cases`, whose host programs are tinygrad's with D36 applied by their
  generator (`gen/runtime/ops_cuda.py`), and `function words (D36) › a batch
  over two devices reads a kernel's function from a word of each`; on an
  NVIDIA GPU, `execution` (slow), whose every kernel reads its function from
  such a word.

## D37. AMD's queues wait on a signal word for equality and write it whole

- **tinygrad:** `runtime/ops_amd.py:402` (`AMDComputeQueue.wait`), `:409-412`
  (`AMDComputeQueue.signal`), `:482-486` (`AMDSDMAQueue.wait`) and `:496-498`
  (`AMDSDMAQueue.signal`).
- **tolk:** `lib/runtime/ops_amd.ml:74` (`is_signal_word`), `:379`
  (the compute queue's `wait`), `:395` (its `signal_mem`), `:677` (the copy
  queue's `wait`) and `:709` (its `signal`).
- **Differs:** tinygrad's queues compare the low 32 bits of a 64-bit word,
  and write only them: the compute queue waits until they are at least the
  value's (`WAIT_REG_MEM`, `>=`) and signals with `RELEASE_MEM` of the low 32
  bits (`send_32_bit_low`); the copy queue polls for `>=` and fences the low 32
  bits. A device's values pass 2^32, and then the word's high half is never
  written, so a 64-bit reader never sees the value, and a wait for a value just
  past the wrap passes on a word just before it (`0xfffffffd >= 5`). In
  tolk a wait on a device's signal word, which is for the value its work
  before the batch signals and which only the batch's own work passes, waits
  until the low 32 bits equal the value's; a signal of a device's value writes
  all 64 bits: in one `RELEASE_MEM` of 64-bit data on the compute queue, and on
  a copy queue with a fence of the low 32 bits then, when they are `0`, a fence
  of the high 32 bits, four NOPs otherwise, chosen by the host program since the
  value is its variable. A high half is then written only when it changes, so
  one that lands after a later value's write holds that value's high half, and
  a wait on the low half never passes before its value is written. Waits and
  signals of the queues' signals in the batch's slots, whose values are small,
  keep tinygrad's encoding.
- **Reason:** (c). nx.device's AMD library states this rule for every work on
  an AMD device (its low-level section): its own copies wait for equality of
  the low 32 bits and write the high half only when the low one is `0`, and a
  batch's work shares their signal word.
- **Pinned by:** the Ops_amd suite (`test/runtime/ops_amd`):
  `signal_words.golden`, the words of the waits and signals at 2^32 - 1, 2^32,
  2^32 + 1, 2^33 - 1 and 2^33 from tinygrad with D37 applied by its generator
  (`gen/runtime/ops_amd.py`); `carry law (D37)`, which decodes them into
  memory writes and checks, in every interleaving of three works' writes and
  a wait, that the word never reads above the latest value whose work
  completed and never goes below a value once its writes landed, and that
  each wait passes only once its value is written; and every `recorded
  cases` golden.

## D38. A program's code object is loaded by the engine

- **tinygrad:** `runtime/ops_amd.py:533-541` (`amd_build_program` makes a
  placeholder tagged `program` and stores the image into it at link), `:1032`
  (`AMDDevice.program_buffer` allocates it), `:547-548` (`_amd_program_image`
  refuses a kernel whose LDS exceeds the GPU's); `runtime/ops_nv.py:314-320`
  (`nv_build_program` makes a placeholder tagged `program` and stores the
  image into it at link, with a row for each relocation of the program's own
  address).
- **tolk:** `lib/runtime/ops_amd.ml:153` (`amd_build_program`),
  `lib/runtime/ops_nv.ml:340` (`build_program`); the engine's AMD and NV
  modules.
- **Differs:** the program's placeholder is tagged
  `("program", binary, name)`, and no link patch writes an image into it: the
  engine loads the code object on the device (`Nx_device.Program.load`) and
  binds the placeholder to the image nx.device uploaded and relocated
  (`Nx_amd_device.kernel`'s `code`, `Nx_nv_device.kernel`'s `image`), whose
  layout is the one the compiler reads its descriptor from (`Nx_device_elf`,
  sections aligned to 128 bytes on NV). The LDS check is the AMD load's; NV's
  relocations are its load's.
- **Reason:** (c). Programs are nx.device's (D3): its AMD and NV libraries
  upload a code object once per device into memory the GPU fetches from,
  relocate it, check the kernel against the GPU and record the load in
  profiles.
- **Pinned by:** the Ops_amd and Ops_nv suites: every `recorded cases`
  golden, from tinygrad with D38 applied by its generator, whose offsets into
  the image tinygrad's ELF loader computes and the encoders read from
  `Nx_device_elf`; the Ops_nv suite's `storage › a compute queue names its
  program's cubin, ...`.

## D39. A submission writes at most half of each AMD ring

- **tinygrad:** `runtime/ops_amd.py:420-428` (`AMDComputeQueue.push`, which
  the AQL queue's `submit` calls too) and `:500-520` (`AMDSDMAQueue.submit`):
  the host program writes a submission's packets from the ring's put
  position without looking at how far the engine has read, and the SDMA
  queue refuses only a command buffer larger than its ring.
- **tolk:** `lib/runtime/ops_amd.ml:599` (the AQL queue's `submit`) and
  `:733` (the copy queue's).
- **Differs:** a submission writes at most half of each ring. A copy
  queue's command buffer over a quarter of its ring, since zeroing the tail
  when it does not fit before the ring's end can double what it takes, and
  AQL packets over half of theirs, are over the queue's capacity
  (`Hcq2.Over_capacity`): the batch runs as several submissions, its calls or
  a range's trips split between them, and only a call too large for its
  queue on its own is refused. nx.device waits, before the host program runs,
  until each ring of the device is at most half full, so no host program
  overwrites packets its engine has not read. The copy queue keeps streaming
  its command buffer into its ring, as tinygrad's does: an SDMA indirect
  buffer on a user queue is used by nothing this port can check against.
- **Reason:** (c). nx.device's queue writers wait until the engine leaves room
  (the AMD library's low-level section), and a host program cannot wait with
  the device's timeout nor lose it; `Nx_device.submit` does, through the
  driver's `room`.
- **Pinned by:** the Ops_amd suite: `room (D39)` and `splits (D39)`, a range
  of 10,000 trips on AQL, whose chunks (D67) split into batches within their
  ring's share, and one of 1,000 copies, split likewise; nx.device's suite: `timeline ›
  a submission runs once its device's queues have room` and `› a device whose
  queues stay full through its timeout is lost ...`.
## D40. The copy engine signals the high word of a value too

- **tinygrad:** `runtime/ops_nv.py:197-201` (`NVCopyQueue.semaphore` and
  `.signal`, a one-word copy-engine release of the value's low 32 bits).
- **tolk:** `lib/runtime/ops_nv.ml` (`copy_signal`).
- **Differs:** the copy queue signals a value in two one-word releases: its low
  word into the signal word, then its high word into the next four bytes of
  the signal word when the low word is `0`, and into a word of its own, a
  volatile placeholder of the device tagged `nv_sink`, otherwise. The value
  is known when the host program runs, so the second release is always
  encoded, and its address chosen then.
- **Reason:** (c). Timeline values are 64 bits, and nx.nv.device writes them
  as this rule says (its low-level section). A low word alone loses the high
  word at 2^32: the signal word falls from 2^32 - 1 to 0 and stays there. A
  high word rewritten on every release takes the value back when another
  channel's release of the next value lands between the two words.
- **Pinned by:** the Ops_nv suite (`test/runtime/ops_nv`): `command words ›
  the copy engine releases the low word, then the high word where the value
  lives (D40)` for the values 2^32 - 1, 2^32, 2^32 + 1, 2^33 - 1 and 2^33;
  `the 32-bit carry law`, over every order in which the writes of three works
  on the compute channel, this copy queue and nx.nv.device's copies can land,
  and its counterexamples, tinygrad's release and a high word rewritten late;
  the recorded cases, whose host programs are tinygrad's with D40 applied by
  their generator (`gen/runtime/ops_nv.py`).

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

## D42. A batch's command buffers, and its placeholders where a device has no window, are pinned memory

- **tinygrad:** `runtime/support/hcq2.py:588` (`bufferize_buf` allocates a
  volatile placeholder in uncached host memory, a command buffer in device
  memory the host maps and the GPU reads uncached, `BufferSpec(uncached=True,
  cpu_access=True)`, and the others in device memory the host maps,
  `BufferSpec(cpu_access=True)`); `runtime/ops_nv.py:447-485` (NV maps such
  memory through BAR1); `runtime/ops_amd.py:633` (AMD raises without a large
  BAR).
- **tolk:** `engine/tolk_engine.ml` (`placeholder`: a volatile
  placeholder and a command buffer, `Hcq2.is_cmdbuf`, with
  `Nx_device.Buffer.create ~memory:Pinned`, the others with `~memory:Mapped`).
- **Differs:**
  - A command buffer is pinned memory, system memory the command processor
    fetches across the bus, where tinygrad uses device memory the GPU reads
    uncached.
  - Where a device has no window onto its memory that the host writes through,
    or the window is full, its mapped memory is its pinned memory, where
    tinygrad's AMD device raises and its NV device needs the window. A device
    whose vendor library describes no window (CUDA) has its launch descriptors,
    constant buffers and kernel arguments in pinned memory.
- **Reason:** (c). nx.device has no device memory that the device reads
  uncached. Pinned memory is the kind that keeps the command processor's fetch
  coherent with the host's writes with no invalidation. nx.device's mapped
  memory falls back to pinned memory rather than failing, so that a batch runs
  on every machine.
- **Pinned by:** the Engine suite (`test/engine/tolk_engine`): `batches ›
  a batch's kernel arguments are mapped memory, and its command buffers and
  volatile words pinned memory`; the Ops_amd and Ops_nv execution suites on a
  GPU.

## D43. A queue bufferizes its own commands, and a region keeps its alignment

- **tinygrad:** `runtime/support/hcq2.py:471` (`bufferize_cmdbuf` lays each
  nested `LINEAR` out on 128 bytes), `:479-481` (`encode_submit` bufferizes
  the queue's commands, then hands the buffer to `HWQueue.submit`).
- **tolk:** `lib/uop/ops.ml` (the `Region` argument of `Op.Linear`),
  `lib/runtime/support/hcq2.ml:1374` (`bufferize_cmdbuf`), `:1470`
  (`encode_submit`).
- **Differs:** a nested `LINEAR`'s argument is a region, a name and an
  alignment that the vendor creating it states, 128 by default, and printed
  as the name alone then; `bufferize_cmdbuf` starts each region at its own
  alignment. A region's words may address another region: every region the
  command words address, directly or through the words of regions, is laid
  out once, in the buffer of its name, where tinygrad lays a region addressed
  through another out again in a buffer of its own, of the same name.
  `commands.submit` takes no command buffer: `encode_submit` calls it once
  every command is encoded, and the vendor finishes its queue, then calls
  `bufferize_cmdbuf` itself, as AMD's AQL queue already bufferizes its
  packets on the host. A vendor's `submit` raises `Over_capacity` for a
  submission its queue cannot take, and the batch is split (D30): NV's when
  its command buffer has more words than a ring entry's 21-bit length field
  holds (`lib/runtime/ops_nv.ml:404`, `submit_cmdbuf`), where tinygrad's
  entry would carry the length's high bits into its other fields.
- **Reason:** (b). NV's launch descriptors and constant buffers are regions
  (Ops_nv), so that a range's trips each get their own copy of them, as the
  staged scan batches a scan body in one submission (RFC 0012). A launch
  descriptor starts on 256 bytes, since the channel takes its address shifted
  by 8, and it is final only once the commands after it are encoded: the next
  launch chains onto it and the next signals are its releases. tinygrad
  patches a descriptor buffer whose address is known at launch; a region's
  address is known only once its bytes are. A descriptor addresses its
  constant buffer and the next descriptor of its chain, both regions, and
  each must have one address.
- **Pinned by:** `Tolk.Hcq2 › patch and bufferize_cmdbuf › each region
  starts at its own alignment in the buffer of its name` and `› a region
  addressed through another region is laid out once, in the buffer of its
  name`; the Hcq2, NULL, Metal, CUDA and AMD goldens, unchanged; the Ops_nv
  recorded cases, whose generator applies D43 to tinygrad.

## D44. An integer cast of a weak expression computes in integers

- **tinygrad:** `uop/weak.py:27-33` (`cast_weak_srcs`), `dtype.py:180-194`
  (`promo_lattice`, `least_upper_dtype`).
- **tolk:** `lib/uop/uop_weak.ml:55` (`cast_weak_srcs`).
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
  precision (D44)` and `laws › pm_commit_weak computes an integer cast in
  integers (D44)`; the Hcq2, Ops_cuda and Ops_metal suites' `a range's
  addresses are integers, profiled or not` and the Hcq2 suite's `ranges (D30)
  › a trip reads its window past what a float offset holds`.

## D45. Staging memory is a placeholder of the host

- **tinygrad:** `runtime/support/hcq2.py:131-134` (`_staging`: a host
  `Buffer` of 128 MiB, allocated once per device and kept), `:155-156`
  (`stage_copy` names it by `UOp.from_buffer`).
- **tolk:** `lib/runtime/support/hcq2.ml:507` (`staging_size`), `:522`
  (the placeholder in `stage_copy`); `engine/tolk_engine.ml:512`
  (`staging`).
- **Differs:** a copy between memory the queues cannot reach goes through the
  two halves of a placeholder of the host tagged `"staging"`, of 128 MiB. The
  engine gives it, at link, the host's staging memory, nx.device's
  (`Nx_device.staging`): one per host, which every linked schedule and
  nx.device's own staged copies share, and which is kept for the life of the
  process, as tinygrad's is. Runs that stage through one host take turns with
  it, whatever their devices, since each touches it and so takes the host;
  nx.device's copies take the host too, and wait for those runs' work before
  they fill a slot. One area per device would let runs overlap for 128 MiB a
  device, and waits for a measured bottleneck.
- **Reason:** (c). The compiler opens no device and allocates nothing
  (plan §1a): storage it names is a placeholder, and the engine's link
  allocates it.
- **Pinned by:** the Hcq2 suite (`test/runtime/support/hcq2`): `compile_linear
  › copies through the halves of a staging buffer of the host where the
  queues cannot reach`, which finds one placeholder tagged `"staging"` of
  128 MiB on the host, and six copies; the Engine suite
  (`test/engine/tolk_engine`): `batches › linked schedules that stage
  share the host's staging memory`, `› staged runs of two programs on other
  devices take turns` and `› staged runs of two programs from two domains each
  copy their own`.

## D46. Whether a device's queues reach memory is described, not tried

- **tinygrad:** `runtime/support/hcq2.py:151-155` (`stage_copy` maps each
  buffer on the device with `get_buf` and stages the copy when that raises).
- **tolk:** `lib/runtime/support/hcq2.ml:467` (`reached`, from
  `queues.reaches`); `engine/cuda.ml` and `engine/amd.ml` (`reaches`).
- **Differs:** the caller's description of a device says which other devices'
  memory its queues address (`Hcq2.queues.reaches`), and a copy is staged when
  either side's memory is not reached. tinygrad tries to map the buffers and
  stages the copy on failure. A buffer that `reaches` admits but the device
  cannot map is staged by nx.device instead (D70).
- **Reason:** (c). The compiler opens no device, so it cannot try a mapping;
  the engine describes each device's reach from nx.device's
  (`Nx_device.reaches`), which the devices' drivers describe from their
  machine's topology.
- **Pinned by:** the Hcq2 suite: `compile_linear › copies through the halves
  of a staging buffer of the host where the queues cannot reach`, whose device
  description reaches every device but CPU:2.

## D48. A kernel's dispatch packet in its arguments is words

- **tinygrad:** `runtime/ops_amd.py:61-67` (`dispatch_packet`), `:361-364`
  (`kernargs`).
- **tolk:** `lib/runtime/ops_amd.ml:162` (`dispatch_packet`).
- **Differs:** tinygrad gives a known grid size as a Python integer, which the
  AQL queue turns into a 32-bit constant but the kernel arguments of the PM4
  queue put as they are among the sources of a `LINEAR`: encoding a kernel
  that reads its dispatch packet on a PM4 queue raises. tolk's grid sizes
  are 32-bit constants in both.
- **Reason:** (a). A node's sources are nodes: OCaml's types admit no integer
  among them.
- **Pinned by:** the Ops_amd suite: `recorded cases › scratch`, a kernel that
  reads its dispatch packet and scratch memory, on a PM4 queue.

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
  they have; `division (D50) › the operands tell a quotient from a product by
  the reciprocal` and the slow `› Metal divides as IEEE does, rounding once`.
  The `decomp_op` golden `late_division_fdiv`, whose products by a reciprocal
  stay products; rune's `Rune.jit › division (D50)`, where a compiled product
  by a reciprocal and a compiled quotient each equal nx's eager bits at the
  value above, and its generated programs, whose products take reciprocals
  as factors (`test_jit_programs`).
  CUDA's and HIP's `/` are correctly rounded by their compilers' defaults,
  which is on the hardware checks of `test/README.md`.

## D51. A launch reads its local memory size from a word of the device

- **tinygrad:** `runtime/ops_nv.py:283-291` (`NVProgramData` writes the
  device's `slm_per_thread` into the launch template's
  `shader_local_memory_high_size`), `:688-697`
  (`NVDevice._ensure_has_local_memory` grows the device's local memory when
  it builds a program).
- **tolk:** `lib/runtime/ops_nv.ml:330` (`local_word`), `:352`
  (`build_program`); `engine/nv.ml` (`local`).
- **Differs:** a launch template's local memory size reads a 32-bit word,
  a placeholder of the device tagged `("nv_local", bytes)` for kernels that
  need `bytes` per thread, which the host program loads when it runs. Every
  launch on the devices that needs `bytes` reads one such placeholder, since
  placeholders of one tag in a batch become views of one buffer. The engine
  binds it to one word of the device,
  grows the device's local memory to `bytes` when it links the batch
  (`Nx_nv_device.local_memory`) and writes the bytes per thread the memory
  then provides into the word. A batch linked before a later one grew the
  memory launches with the grown size.
- **Reason:** (c). The compiler opens no device, so it cannot read or grow
  the device's local memory; nx.nv.device owns it, and grows it as work on the
  device's timeline.
- **Pinned by:** the Ops_nv suite: every `recorded cases` golden, from
  tinygrad with D51 applied by its generator, and `storage › a compute queue
  names its program's cubin, its channel's words, and the local memory its
  launches need` and `› the launches of two programs in one batch read one
  local memory word, whatever the engine binds to it (D51)`; the Ops_nv
  execution suite on an NVIDIA GPU.

## D52. A selection of -0. stays off a gated load

- **tinygrad:** `uop/symbolic.py:421-425` (`pm_move_where_on_load`), whose
  pattern constant `0` matches `-0.` since `-0. == 0`.
- **tolk:** `lib/uop/symbolic.ml:1222` (`pm_move_where_on_load`).
- **Differs:** a selection between a load and `-0.` stays a selection.
  tinygrad moves its condition into the load's gate, and a gated load reads
  `+0.` where its gate fails, so a pad of `-0.`, lowered as a selection of
  `-0.` off the padded load, computes `+0.` on the padding.
- **Reason:** (b). nx's `pad` fills with the value given, `-0.` included
  (rune's lowering ledger, I1), and rune's compiled pad must compute eager
  nx's bits.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `signed zeros
  (D52) › a pad with -0. fill renders and computes -0.` and `› a pad and a
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
  (`param_arg`, which checks them), `:3741` (`view_start`) and `:3806`
  (`storage_phase`, which `param_like` gives a parameter);
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
- **Pinned by:** the `Coalesce` suite (`test/codegen/late/coalesce`): `phase
  (D54) › one float past a boundary, eight loads are of one, two, four and
  one`, `› three floats past a boundary, a load of four starts at element 1`,
  `› stores start where loads would`, `› two halves past a boundary, a load of
  four starts at element 2`, `› floats known to start on 4 bytes alone are
  not merged`, `› floats known to start on 8 bytes merge in twos` and `›
  halves three past 8 bytes merge no wider than 8 bytes, from element 1`, and
  the laws `every vector access is aligned to its width and within the
  alignment, for every phase and alignment (D54)` and `coalescing preserves
  the kernel's writes, for every phase and alignment (D54)`; the `Ops` suite: `param_arg takes a
  phase that is a multiple of the element size below 16 (D54)`, `param_arg
  takes an alignment that is a power of two up to 16, and a phase below it
  (D54)`, `pp_param_arg writes a phase that is not 0 and an alignment that is
  not 16 (D54)`, `equal_arg tells apart parameters that differ in phase
  (D54)` and `param_like keeps its storage's phase, moved by a shrink, and
  knows it modulo what a symbolic start moves by (D54)`; the `Hcq2` suite's
  `ranges (D30) › a kernel reads windows an odd number of floats apart a
  float at a time, and windows four floats apart four at a time` and `› a
  trip reads its window past what a float offset holds`, on x86; the graph format's round trip of `a buffer past
  a 16-byte boundary`; the `Schedule` suite's `disk_view_to_linear.golden`,
  a view 8 bytes into a disk file whose parameter has no phase. In rune,
  the `Lower` suite's `a parameter of storage 4 bytes past a 16-byte boundary
  has phase 4`, `a capture of storage 4 bytes past a 16-byte boundary has
  phase 4` and `a program over an argument 4 bytes past a 16-byte boundary
  computes nx's values`, and the `Compiled` suite's `edges › an operand whose
  buffer starts 2 bytes into its memory is read where it is`, on the
  host and on Metal, whose sweeps draw buffers that start at any byte; the
  slow `Ops_metal (execution)` suite's `phase (D54) › a float16 buffer 2 or 6
  bytes into its memory is read where it lies with its phase`; the `Schedule`
  suite's `contiguous_mops_to_view › a stage of doubles one past a boundary is
  known to start on 8 bytes (D54)`, `› a stage of rows padded apart, starting
  one double past a boundary, is storage of its own on one (D54)` and `› a
  stage's alignment and phase hold of the storage it gets, a view or its own
  (D54)`; rune's `Scan` suite's `compiled › a scan under jit reads a copy of a
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
- **tolk:** `lib/codegen/decomp/transcendental.ml:556-564` (`xlog2`),
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
  negative number whose reciprocal overflows is NaN, and of -0. is -inf
  (D56)`, and the negative float16 rows of `values.golden` of magnitude below
  `2^-16`, from the patched tinygrad.

## D58. A program applies no elementwise operation to a vector

- **tinygrad:** `uop/spec.py:205` (`spec_program`, which accepts an
  elementwise operation of any shape).
- **tolk:** `lib/uop/spec.ml:390` (the first rule of `program`).
- **Differs:** `Spec.program` rejects an operation of `Op.Set.elementwise`
  on values, casts and bitcasts included, whose shape has an axis; a bitcast
  of memory, which views it, stays allowed. `SPEC` defaults to 1
  and code generation checks every lowered kernel against `Spec.program`, so
  a kernel that still holds vector arithmetic fails at lowering, naming the
  operation, on every target. Devectorize leaves none while every source
  keeps its width, which D76 keeps when every lane of a gated load's index
  is `Invalid`.
- **Reason:** (b). CUDA's vectors are structs without arithmetic, casts or
  selects, so a vector operation left after devectorize is a kernel that does
  not compile there, and one Metal renders without complaint; rune compiles
  its kernels for both.
- **Pinned by:** the `Spec` suite (`test/uop/spec`): `vectors in programs
  (D58) › a program has no elementwise operation on a vector` (add, cast and
  where on two lanes, and the same on one); the `Codegen` suite's `vectors in
  programs › a weak constant stored into four lanes of half is refused` and,
  on every case, `› no program applies an elementwise operation to a
  vector`; the `Cstyle` suite's `every GPU kernel compiles with its target's
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
  [END(LINEAR [CALL k1; CALL k2; ...], r)]`, the form that ranges of calls
  on HCQ devices (D30), the memory plan of a range (D41) and `Engine.run`
  already run. An end over device ranges is still the call, bound at launch.
  The kernel graph spec admits loop ranges, ends of calls over them, and
  views that move with them, whose bounds are constants, loop ranges, and
  weak integer sums and products of them; a range of any other kind outside
  a kernel is still refused, so a range rangeify leaks never runs as a loop.
  tinygrad has no such loop: its `split_store` wraps the end in a kernel of
  its own, and its `create_schedule` drops the end.
- **Reason:** (b). rune's `Rune.scan` compiles its body once and runs it
  once per trip, in place over its carry and over rows of its inputs and
  outputs, as such a loop.
- **Pinned by:** the `Rangeify` suite (`test/schedule/rangeify`):
  `get_kernel_graph › loops of calls (D60) › a loop of a precompiled call is
  no kernel` and `› a loop's call reads views that move with its range`; the
  `Schedule` suite (`test/schedule/schedule`): `create_schedule › rules ›
  an end of a call over a loop schedules the call in its loop (D60)`, `› an
  end of a call over device ranges schedules the call`, `› a loop runs
  after the call that writes what it reads (D60)`, `› a loop runs before
  the call that overwrites what it reads (D60)`, and
  `create_linear_with_vars › loops of calls (D60) › a loop of a precompiled
  call is a loop of its body's calls` and `› a loop inside a loop's body
  keeps its own end`; and the `Tolk_engine` suite
  (`test/engine/tolk_engine`): `link and run › a scan runs its body
  once per trip, carrying in place (D60)`, against the unrolled loop on the
  host. The `Spec` suite (`test/uop/spec`): `kernel_graph › loops of calls
  (D60) › accepts a loop of a call over a loop range`, `› refuses a loop of
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
  of every 8-bit float code keeps its bits (D62)` and `› a selection of
  every 8-bit float code keeps its bits (D62)`, and the same under `metal ›`
  (slow), against eager on all 256 codes of e4m3 and e5m2; the
  `Decomp_dtype` suite (`test/codegen/decomp/decomp_dtype`): `NaNs › an
  emulated copy keeps every NaN code, a signalling one's included (D62)`
  and `› an emulated selection keeps every NaN code, a signalling one's
  included (D62)`, on every emulated float, and the goldens of the
  `where`, `flip`, `gather` and `pad` kernels, from the patched tinygrad.


## D63. Programs and schedules are kept on disk

- **tinygrad:** `codegen/__init__.py:497-505` (`to_program_config`,
  `to_program_key` and `to_program_cache`, a dictionary of the process),
  `schedule/__init__.py:127-137` (`lower_sink_to_linear`'s schedule cache,
  on disk only from `SCACHE=2`), `helpers.py:286` (`SCACHE`, `1` by
  default), and `codegen/opt/postrange.py:45-46` (a kernel's name, coloured).
- **tolk:** `lib/codegen/codegen.ml:1157` (`program_key`), `:1167`
  (`kept`) and `:1182` (`made_program`); `lib/schedule/schedule.ml:309`
  (`schedule_key`) and `:314` (`lower_sink_to_linear`); `lib/uop/graph.ml:957`
  (`cached`); `lib/codegen/opt/postrange.ml:663` (`get_optimized_ast`'s
  name); `lib/setting.ml:266` (`scache`, `2` by default) and `:141`
  (`shaping`); `lib/setting.ml:294` (`cc`); and
  `lib/dune`'s rule for `source_digest.ml`, written by
  `tools/source_digest.ml`.
- **Differs:** the program `to_program` makes of a kernel, and by default
  the schedule `lower_sink_to_linear` makes of a function, are also put in
  the disk cache (tables `to_program` and `schedule_cache`) as their graphs'
  text (`Graph`), and a later process reads them back instead of making
  them. tinygrad keeps programs for its process only, and schedules on disk
  only when asked. Each key is the kernel or function, a function with its
  ranges numbered in order (whoever makes a loop numbers its range from a
  counter whose value depends on what the process did before), for programs
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
  loops of calls › a loop whose range has another number is the same body`;
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
  bfloat16 (D64) › an integer or a double rounds once on the host`, for
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
  (`test/codegen/decomp/decomp_dtype`): `rounding in the kernel (D65) › a
  cast to an emulated float and back is the value rounded once` and `› an
  operation on an emulated float rounds its result to it`, on every emulated
  float, and the goldens of every kernel that casts to or computes in an
  emulated float, from the patched tinygrad; rune's `Jit` suite: `one
  device › a chain of 8-bit float operations rounds after each, as eager
  does`, and the same on Metal.

## D66. A counted run's log entry is its kernel descriptor's address and its times

- **tinygrad:** `runtime/ops_amd.py:157-165` (`prof_start` writes the program
  placeholder's slot, a constant, into the run's entry of `prof_log`, one word
  a run), `:1056-1068` (`collect_prof` hands that slot to its profile event,
  which `viz/serve.py:353` maps to the program, and pairs the event with the
  program's runs in order).
- **tolk:** `lib/runtime/ops_amd.ml:547` (`start_run` writes the address of
  the kernel descriptor, `getaddr lib + desc_offset`, and the GPU's clock
  before the kernel; `stop_run` the clock after it).
- **Differs:** a run's entry is three words: the address of its kernel
  descriptor, which the host program computes from the code object's address
  at link, where tinygrad writes a constant that names the program's
  placeholder in its compiler; then the GPU's clock before and after the run,
  which the compute queue writes with two more packets a counted run.
- **Reason:** (c). nx.amd.device reads the log at each synchronization and
  reports each run's counters as an event of the run
  (`Nx_device.Profile.Counters`): named after its kernel, which it knows by
  its descriptor address, the program's handle, and nothing of a compiler's
  placeholders; and timed by the run itself, so that counters belong to their
  run whatever runs between them go uncounted or are lost, where pairing them
  with spans in order would give a run another run's counts.
- **Pinned by:** the Ops_amd suite: `recorded cases › counters`,
  `counters_gfx1201` and `counters_gfx942`, from the generator patched as
  `test/gen/runtime/ops_amd.py` says.

## D67. A long range runs as chunks of its trips

- **tinygrad:** `runtime/support/hcq2.py:379-385` (`HWQueue.loop`, which
  repeats a trip's command bytes and words once per trip:
  `self.blob += self.blob[start:] * int(r.vmax)`).
- **tolk:** `lib/runtime/support/hcq2.ml:1215` (`chunk_calls`), `:1219`
  (`parts`), `:1230` (`chunked`) and `:1184` (`trips_from`), used by
  `sched_batches`; `engine/tolk_engine.ml:906` (`run_call`'s `Range`, which
  binds the chunk's variable).
- **Differs:** a range of a batch (D30) of more than `chunk_calls` calls,
  `n` trips of `k` calls, runs as chunks of `c = max 1 (chunk_calls / k)`
  trips. One batch holds a chunk: its loop reads the range as
  `range_value r' * c` plus its own range, for a range `r'` of `n / c` trips
  that stays in the schedule as a range of the engine around the batch, so
  the engine runs the batch once per chunk with the chunk as a variable. A
  second batch holds the `n mod c` trips left, when there are any. The
  batches hold at most `2·c·k` commands and their arguments, whatever `n`,
  and a run takes `n / c` submissions and one more for the trips left,
  where tinygrad's loop holds `n·k` commands in one submission.
  `chunk_calls` is 1,024: on Metal a kernel's launch takes about 3 µs and a
  submission about 24 µs, so a chunk's submission costs under 1% of its
  launches, and its commands and arguments take about 270 KB.
- **Reason:** (b). `Rune.scan`'s staged loop runs a body of a few calls per
  step for as many steps as the scan has: unchunked, a scan of a million
  steps held about 800 MB of commands and arguments on Metal (808 bytes a
  step), and its program grew with its trip count, which staging exists to
  avoid.
- **Pinned by:** the Hcq2 suite (`test/runtime/support/hcq2`): `ranges
  (D30) › a range of more calls than a chunk runs as a batch of a chunk, once
  per chunk` (a batch of 1,024 trips around which the engine loops twice, a
  batch of the 5 trips left, three submissions, and every trip's window
  written); the Ops_amd suite's `splits (D39) › a range of 10,000 trips on
  AQL …` (a chunk over half its ring halves, the trips left fit); rune's
  Jit suite on Metal, `staged scans › stage a thousand steps` and `› stage
  more steps than a batch holds, in chunks and the rest` (3,001 steps), each
  against the eager scan.

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

## D70. A batch reaches host memory through nx.device's staging

- **tinygrad:** `device.py:303-308` (`HostAllocator._alloc`: every host buffer
  is its own `mmap`, so it starts on a page and every device maps it), and
  `runtime/support/hcq2.py:151-155` (`stage_copy`, which maps each buffer).
- **tolk:** `engine/tolk_engine.ml:382` (`reach`), `:604` (link's
  `reach_word`) and `:721` (`run_batch`); `lib/runtime/support/hcq2.ml:222`
  (`call_writes`, the batch's `writes`).
- **Differs:** nx.device starts a host buffer on a page only from 64 KiB, so a
  CUDA, AMD or NV device, which maps whole pages, cannot borrow a small one.
  A batch's devices reach the storage its words and inputs address through
  `Nx_device.Buffer.reach`, which stages such memory in pinned memory of the
  device and copies it in and back around each submission. The access each
  memory is reached for is `Read_write` when the batch writes it through any
  word or input, from the batch's queue data's `writes`, which tinygrad's
  `HCQInfo` has not: the storage under each output of each of its calls, a
  copy's destination and a kernel's outputs, in-place ones included, and under
  every argument of a call whose outputs are not known. A placeholder's
  storage is borrowed, never staged.
- **Reason:** (c). nx.device keeps small tensors off a page each, and staging
  is its to order. tinygrad's `written_bufs` leaves out the storage a call also
  reads and parameters, so it cannot say which staged memory a run writes.
- **Pinned by:** the Engine suite (`test/engine/tolk_engine`): `batches ›
  a host buffer the device cannot borrow is staged, as storage` and `› as an
  input`, of 12, 256 and 1,280 bytes, on CPU:4, a test device that maps whole
  pages: each copies the host buffer in and a device buffer back over it, twice,
  the host rewriting it between runs; and `› a run waits for its batch only
  when it writes a staged buffer`, whose read returns while CPU:4's late queue
  has not signaled.

## D71. A launch states a constant bank's size in 16-byte units

- **tinygrad:** `runtime/ops_nv.py:302` (`NVProgramData` writes a constant
  bank's bytes into the launch template's `constant_buffer_size_shifted4`).
- **tolk:** `lib/runtime/ops_nv.ml:301` (`program_data`).
- **Differs:** a launch template states each constant bank's size rounded up
  to 16 bytes and shifted right by 4, as the field's name says, where
  tinygrad writes the bytes. The field holds 13 bits, so tinygrad refuses a
  bank of 8 KiB or more, a parameter bank of about a thousand arguments, and
  states a smaller bank 16 times its size.
- **Reason:** (b). tinygrad writes a byte count into a field that counts
  16-byte units. rune's staged scan `write out four hundred steps, each
  carry stored` launches a kernel whose parameter bank holds 13,192 bytes,
  which NV refused (`constant_buffer_size_shifted4_0=0x3388 does not fit`)
  and CUDA runs.
- **Pinned by:** the Ops_nv suite: every `recorded cases` golden, from
  tinygrad with D71 applied by its generator; rune's Jit suite on an
  NVIDIA GPU: `nv › staged scans › write out four hundred steps, each carry
  stored`.

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
- **Pinned by:** the `Prepare` suite: `contiguous_view › a constant is no view
  (D72)`, `› a constant of one element is no view (D72)` and `› a computed
  value is no view (D72)`.

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
  `lib/codegen/decomp/transcendental.ml:431` (`xsin`); `test/gen/tinygrad.patch`,
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
  value a float operation rounds to (D74)`, sums, differences, products and
  selections, `› a float operation of bounded operands is bounded (D74)`,
  `› a float operation of an operand within the subnormals bounds a flushed
  operand too`, `› a float operation of an unbounded operand, or that can
  overflow, has its type's bounds (D74)`, `› a float selection by a
  comparison narrows what it selects (D74)`, `› a float truncation and
  negation map their operand's bounds (D74)`, `› a value of a float type
  without infinities, which may be NaN, has no finite bounds` and the float
  rows of `binary_bounds.golden`, from the
  equally patched tinygrad; the Transcendental suite: `graphs › a sine of an
  angle bounded below the switch-over is its fast form (D74)`; and
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
- **tolk:** `lib/uop/symbolic.ml:334` (`broadcast_const`), `:610` and `:666`
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
- **tolk:** `lib/uop/symbolic.ml:237` (`pm_data_invalid`, without that rule);
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
  `renderer/cstyle`, `runtime/support/hcq2` and `engine/jit` goldens, from the
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
  (`kernel_info.split`) and `:4729` (`program_info_of_sink`);
  `lib/engine/realize.ml:27` (`get_call_var_uops`);
  `engine/tolk_engine.ml:202` (`block_ops`) and `:230` (`Program.split`);
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
  first two slots past those the kernel's parameters took, and named afresh
  where the kernel has a variable of either name. The variables numbered
  after them take the slots past every slot a parameter takes, where tinygrad
  numbers them from the count of the numbered parameters
  (`codegen/__init__.py:380`): a kernel whose buffers skip a slot, as a beam
  search's timer links them, would give a variable a bound's slot, and a
  launch would read the bound for it. The kernel's `KernelInfo`
  records `split = (n, lo, hi)`, the iterations and the two slots, by which
  each launch finds the bounds; `n` is its program's first global size, and
  its estimates count the whole loop, exactly when they read no other
  variable: counted in the loop's 32-bit type, a count past 2^31 would wrap. A host launch cuts
  `[0, n)` into `min(n, 4 × workers, ops / 2^18)` blocks, at least one, that
  nx.device's thread pool runs (`Nx_device.Program.call ~split`); a queue's
  launch (`get_call_var_uops`) and tinygrad's own (`whole_loop` in
  `engine/realize.py`) run one block, the whole loop.
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
  Tolk_engine suite: `host programs in blocks › a program writes the same bits
  in 1, 3 and 32 blocks`, `› a run of much work splits into blocks`, `› a run
  of little work runs as one block` and `› a run of a large kernel computes
  its sums`, and `a variable named as a block's bounds is a variable like any
  other`; nx.device's host suite: `a split call runs each iteration once,
  in its block` and `a split call runs while another domain's holds the
  threads`.

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

## D84. A dispatch's thread-trace marker numbers its queue's dispatches

- **tinygrad:** `runtime/ops_amd.py:248` (`sqtt_setup_exec` takes the
  marker's command id from `self.dev.sqtt_next_cmd_id`, a counter of the
  device for the life of the process), `:902` (the counter).
- **tolk:** `lib/runtime/ops_amd.ml:581` (`commands`, a counter of the queue
  being encoded) and `:584` (`trace_markers`).
- **Differs:** the command id of a dispatch's RGP event marker counts the
  dispatches of its queue from 0, where tinygrad counts every dispatch the
  device's queues encoded in the process.
- **Reason:** (c). A compiled batch is a value the engine links and caches:
  with a device's counter, its packets would depend on what compiled before
  it. The id only ties a trace's markers together, and each run's trace is
  its own.
- **Pinned by:** the Ops_amd suite: `recorded cases › traces`,
  `traces_gfx1201`, `traces_gfx942` and `counters_traces`, from the generator
  patched as `test/gen/runtime/ops_amd.py` says.

## D85. A GFX11 trace sets TTRACE_EXEC by name

- **tinygrad:** `runtime/support/amd.py:10` (`AMDReg.encode` shifts each value
  to its field's lowest bit without cutting it), `runtime/ops_amd.py:310`
  (`sqtt_start` gives `token_exclude` the bit 11 of
  `SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT`, past the field's 11 bits on GFX11, which
  so sets `TTRACE_EXEC`).
- **tolk:** `lib/runtime/ops_amd.ml:412` (`encode` cuts each value to its
  field, as `bits` does), `:698` (a GFX11 trace sets `ttrace_exec`).
- **Differs:** a value wider than its field loses its high bits, and a GFX11
  trace sets `TTRACE_EXEC` by name where tinygrad sets it through the spilled
  bit. `SQ_THREAD_TRACE_TOKEN_MASK` takes the same value; no other value of
  the recorded cases spills.
- **Reason:** (c). A value spilling into the next field changes what the
  hardware does without anyone asking for it; Mesa's register macros cut each
  value to its field. Setting the bit by name keeps the traces configured as
  those nx.amd.device's decoder is tested on.
- **Pinned by:** the Ops_amd suite: `recorded cases › traces` and
  `counters_traces`, from tinygrad's own encoding, which spills.

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
  a program reads a lane of a vector at a constant (D86)`; the Cstyle suite's
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

## D90. A linked schedule resolves its host programs' launches at link

- **tinygrad:** `engine/realize.py:132` (`resolve_params`) and `:160`
  (`exec_kernel`), which find a call's buffers, launch dimensions and variable
  values on every run, and `uop/ops.py:1177` (`UOp.sym_infer`), which
  substitutes a symbolic integer's variables and simplifies it on each
  evaluation.
- **tolk:** `engine/tolk_engine.ml:283` (`env`, the run's variable cells),
  `:481` (`operand`), `:507` (`lane_operand`), `:528` (`launch`), `:1171`
  (`settle`), `:1180` (`run_launch`) and `:1232` (`run_call`'s `Range`);
  `lib/uop/ops.ml:3964` (`sym_compile`).
- **Differs:** linking a schedule turns each call of a host program into a
  launch per lane: a view of linked storage at a constant offset is made
  once, the offset of a view that moves with a range or a variable is an
  integer function of the run's variables (`Ops.sym_compile`, which simplifies
  once and computes on `int`s, exactly where a value does not fit), and each
  variable, of the run or of a range, has a cell that the run and the range's
  trips write. The devices of a host program's buffers, which nx.device's
  ordering has the run synchronize (c), are synchronized once until the run
  queues work again, with a batch: a copy returns once its bytes have landed.
  Results are those of a resolution on each run. A `DEBUG` report of a call
  lists the run's variables that the schedule reads, where it listed every
  variable of the run.
- **Reason:** (b): symo's tutorial step on the CPU runs 2,273 host launches,
  mostly inside the ranges of its staged scans, and sofo's steps do likewise.
  Resolving a launch on each run cost about 5 us on an M1 Max, for kernels
  of 0.3 to 40 us: a jitted scan of 256 small steps (1,024 launches) took
  5.13 ms and takes 0.27 ms.
- **Pinned by:** the Tolk_engine suite (`test/engine/tolk_engine`): `link and
  run › a linked scan carries its storage across runs on each run's
  parameters`, `› a scan runs its body once per trip, carrying in place`, `› a
  schedule runs with each binding of its variables`, `runs › runs of one
  schedule from two domains each compute their own` and `batches › a host
  kernel runs once the copy that feeds it landed, after another synchronized
  the host`; the Ops suite (`test/uop/ops`): `sym_compile › computes what
  sym_infer does, variables within 50`, `› within 2147483648`, `› at the int
  extremes`, `› computes exactly where int arithmetic overflows` and `› raises
  as sym_infer does where a value fits no int`.

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
- **tolk:** `lib/uop/symbolic.ml:680` (`fold_where_closure`) and
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
- **tolk:** `engine/tolk_engine.ml:24` (`target`); `lib/device.ml:144`
  (`renderer`); `lib/helpers.ml:25` (`Target`).
- **Differs:** there is no `DEV` setting. `Tolk_engine.target` is a function
  of the device: its vendor's kind and its `arch`, or the host's CPU target.
  `Device.renderer` takes the target and renders for it, the first renderer
  of its kind that suits it when it names none. `Helpers.Target` keeps the
  record and its syntax, which tests and the goldens' generator write.
- **Reason:** (c). nx.device places values on devices, and the compiler
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

## D107. A batch submits a streamed queue after the queues it waits for

- **tinygrad:** `runtime/support/hcq2.py:285-293` (`_finalize_batch`, which
  submits the queues in their first use) and `runtime/ops_cuda.py:28-69`
  (`CUDAQueue`, whose commands are calls of CUDA's driver).
- **tolk:** `lib/runtime/support/hcq2.ml:427` (the type `submission`), `:1051`
  (`streamed`), `:1066` (`streamed_order`) and `:1093` (`finalize_batch`);
  `lib/runtime/ops_cuda.ml:173`.
- **Differs:** a vendor's queues say how its host program hands them their
  commands: all at once at their submission (`Buffered`: Metal, AMD, NV), or
  each as the host program makes it, the host waiting while a queue is full
  (`Streamed`: CUDA, whose stream calls enqueue at once). A batch on streamed
  queues submits each queue after the queues its calls wait for, the first in
  use first among those that can come next; a batch whose streamed queues
  wait for each other runs as two, as a batch its queue cannot hold does. A
  device's closing waits, a few commands at the end of a queue, may still name
  a queue submitted after it. Batches on buffered queues are unchanged.
- **Reason:** (b): symo's tutorial (`tutorial/rnn.ml`) on CUDA. Its compiled
  optimizer step's batch runs about 35 kernels on the compute stream, waits
  for a copy into the GPU, then runs 300 more; tinygrad's order submits the
  compute stream first, which stops at the wait, fills, and blocks the host in
  `cuLaunchKernel` before it submits the copy: the step hung. A copy queued
  first runs the step.
- **Pinned by:** the Hcq2 suite (`test/runtime/support/hcq2`): `streamed queues
  › a queue is submitted after the queue it waits for, which it follows in the
  batch`, `› a copy of a kernel's output runs after it`, `› queues that wait
  for each other run as several batches` and `› a range whose queues wait for
  each other runs to the end`, on a model of streamed queues that hold two
  commands (`Batches.run ~capacity`); `Tolk.Ops_cuda`'s `execution › kernels
  after a copy into the GPU that they read all run` on a GPU.

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
- **tolk:** none: nx.amd.device owns the GPU, and loses it on a hang: under
  AM after 30 s without progress (`packages/nx/lib/amd/device/nx_amd_device.ml`,
  `hang_ms`, "hang detected"), under KFD when the driver reports one.
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
- **Pinned by:** nx.amd.device's hardware suite
  (`packages/nx/test/amd/device/test_amd_hw.ml`): `failures › work that
  never signals loses the device`, on a GPU.

## D112. An AMD memory barrier leaves the host data path to the host

- **tinygrad:** `runtime/ops_amd.py:142-146` (`memory_barrier`: a
  `WAIT_REG_MEM` that writes every bit of the bus interface's
  `GPU_HDP_FLUSH_REQ` and waits until `GPU_HDP_FLUSH_DONE` holds them all,
  then `acquire_mem`).
- **tolk:** `lib/runtime/ops_amd.ml:388` (`memory_barrier`, `acquire_mem`
  alone) and `:318` (`wait_reg_mem`, which no longer writes a register
  before its wait); `engine/amd.ml:144` (`queues`, whose submission hook
  flushes the host data path from the host before each host program).
- **Differs:** a barrier invalidates the GPU's caches and does not flush the
  host data path (HDP), whose flush registers the queue encoder no longer
  names. The host flushes it through the driver's remapped
  `HDP_MEM_FLUSH_CNTL` (`Nx_amd_device.flush_hdp`) before each submission,
  after every write through the BAR: memory a host program writes on each run
  is pinned host memory, which the HDP does not carry.
- **Reason:** (c). On a Radeon AI PRO R9700 (GFX12, nbif 6.3.1) under the
  amdgpu driver, a compute queue running batches back to back while the host
  allocates and frees device memory hangs within a few hundred batches, and
  the driver resets the GPU for every process on it. At the hang no wave runs
  and the MEC waits on a register read (`CP_CPC_STALLED_STAT1`
  `MEC1_WAIT_ON_RCIU_READ`) inside the batch, the barrier's wait on
  `GPU_HDP_FLUSH_DONE`, the only register the batch reads. The request is
  a handshake shared by the GPU's engines, one bit each: on nbif 6.3.1 bits
  0-9 are the command processors' (`CP0`-`CP9`), 10-11 the copy engines',
  12-31 reserved. amdgpu's rings request and wait on their own bit only,
  derived from the ring's ME and pipe (`ref_and_mask_cp2 << pipe` on MEC1).
  tinygrad requests all 32 bits and waits until all are done, so its wait
  also depends on every other engine's handshake, the driver's copy rings'
  among them. No narrower mask is correct for a user queue: the scheduler
  (MES or HWS) maps it to a pipe it chooses, and may remap it, so the queue
  cannot name its own bit, and the pipes' bits are the driver's rings' too.
  ROCr never flushes from a user queue: it writes the remapped
  `HDP_MEM_FLUSH_CNTL` from the host, as nx does. On other generations the
  change relies on the same host flush, which `Nx_amd_device.flush_hdp` does
  identically through the KFD remap on GFX9 and GFX11 and through the
  register under AM, and on nx giving no mapped memory when the driver
  remaps no flush register, so that the host never writes through the BAR
  without one.
- **Pinned by:** the Ops_amd suite: every recorded case, from the generator
  patched as `test/gen/runtime/ops_amd.py` says; and on a GPU,
  `test_ops_amd_exec`'s `execution › a thousand runs back to back while the
  host allocates`.

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
  (`C.identity`); `lib/codegen/codegen.ml:1181-1198` (`program_key`, `kept`);
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
  DEBUG >= 2` unless given); `codegen/__init__.py:386`
  (`os.environ.get("DBGTV")`, read on each failure).
- **tolk:** `lib/setting.ml:87` (`reach`), `:141` (`shaping`), `:151`
  (`t`), `:215` (`jitbeam`) and `:220` (`tc_opt`);
  `lib/codegen/opt/search.ml:17` (`tc_opt_all`) and `:19` (`actions`);
  `lib/codegen/opt/heuristic.ml:101`; `lib/setting.ml:300`
  (`dbgtv`); `lib/runtime/support/hcq2.ml:2051` (`compile_linear`);
  `engine/tolk_engine.ml:1083` (`profile`); rune's `lib/jit.ml:105`
  (`settings`).
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
  whose `None` stands for `BEAM`'s current value. `compile_linear` and
  `jit_lower` take a required `profile`, `Stamped` or `Unstamped`, and read
  no setting for it: their caller asks the engine (`Tolk_engine.profile`),
  which is `Stamped` from `DEBUG` 2, when it prints each kernel's time, and
  there is no `PROFILE`: nx.device's profiles record the spans of what was
  compiled to stamp them. A variable tinygrad reads on each use, such as
  `DBGTV`, is read when the program starts. A library that compiles with
  tolk declares its own variables the same way, so rune's jit keys on its
  `Output` settings too, and beyond them only on what is not a setting of
  tolk: `Tolk_engine.profile ()` and the counters and traces of the
  profile being taken. A function compiled with an explicit width keys on
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

## D118. A program's launch reads its own kernel of its cubin

- **tinygrad:** `runtime/ops_nv.py:249-258` (`NVProgramData` takes the
  registers and stack of the last `EIATTR_REGCOUNT` and
  `EIATTR_MIN_STACK_SIZE` of `.nv.info` whatever function they name, every
  `.nv.constantN*` section as a bank, and the whole image as code from
  offset 0 when the cubin has no `.text.<name>`) and `:315`
  (`nv_build_program` caches a program by its binary alone).
- **tolk:** `lib/runtime/ops_nv.ml:39` (`program_data`) and `:117`
  (`build_program`), through `Nx_nv_cubin.kernel`.
- **Differs:** a launch reads the attributes of its kernel's function, by the
  symbol each names, and the banks `.nv.constantN` and
  `.nv.constantN.<name>` only. A program whose cubin has kernels but not its
  own is refused (`Invalid_argument`, naming the cubin's kernels). Programs
  are cached by binary and kernel name.
- **Reason:** (c). nx.nv.device loads a program's function by name and
  refuses a cubin without it, and its loader and tolk read cubins through one
  library, `nx.nv.cubin`. tinygrad's reading launches the wrong code from
  offset 0 when a name is missing, and gives every kernel of a module of
  several the same registers.
- **Pinned by:** the Ops_nv suite: `refusals › a program whose cubin lacks
  its kernel is refused`, and every `recorded cases` golden, whose generator
  (`gen/runtime/ops_nv.py`) names each program's kernel in its cubin and
  points the crafted cubin's attributes at its kernel's symbol; nx's
  `nx.nv.cubin › kernels › each kernel reads its own registers, stack and
  banks`.

## D119. AMD's compute queue stamps the GPU's clock as it reaches the packet

- **tinygrad:** `runtime/ops_amd.py:404-407` (`AMDComputeQueue.timestamp`, a
  `RELEASE_MEM` of the GPU's clock on the end-of-pipe event
  `CACHE_FLUSH_AND_INV_TS_EVENT`).
- **tolk:** `lib/runtime/ops_amd.ml:638` (`clock_into`, through
  `Nx_amd_packet.Pm4.copy_data`), used by `:718` (the compute queue's
  `timestamp`) and by the counted runs' times of D66 (`:655`, `:669`).
- **Differs:** the compute queue writes the GPU's clock with a `COPY_DATA`
  from the clock counter, 64 bits with a confirmed write, which the queue
  executes as it reaches the packet. An end-of-pipe write waits for the pipe
  to drain, and on GFX12 the drain can include the dispatch queued behind the
  stamp: on an R9700 (gfx1201, KFD), a stamp before a kernel of 0.6 s was
  written 11 us before the stamp after it in most runs, so the kernel's span
  was 11 us. A `BOTTOM_OF_PIPE_TS` event did the same. The queue reaches a
  stamp after the work before it is complete, since each dispatch ends with a
  CS partial flush and each AQL packet has its barrier bit, so a stamp after a
  kernel still follows its waves.
- **Reason:** (b). `Tolk_engine.time` times a search's candidates by their
  spans. In the lorenz training step on AMD, most of one kernel's candidates
  timed at 10 to 13 us; the search picked one, which took 0.8 s of each 1.08 s
  step, where NV's search picked a candidate of 2.9 ms.
- **Pinned by:** the Ops_amd execution suite
  (`test/runtime/ops_amd/test_ops_amd_exec.ml`): `a profile's span of a long
  kernel covers its run`; the Ops_amd suite: `recorded cases › profile`,
  `counters` and `traces`, from the generator patched as
  `test/gen/runtime/ops_amd.py` says.

## D120. An AMD program's descriptor is its code object's one kernel

- **tinygrad:** `runtime/ops_amd.py:545-555` (`amd_build_program`: the
  descriptor at the start of `.rodata`, relocated by its own loop).
- **tolk:** `lib/runtime/ops_amd.ml:163` (`program_data`).
- **Differs:** tolk reads the code object through `nx.amd.code_object`,
  which nx's AMD loader reads too: the descriptor is the symbol `name.kd` of
  the code object's one kernel, and a code object of several kernels is
  refused. A code object comgr links for one kernel places that kernel's
  descriptor at the start of `.rodata`, so both read the same bytes.
- **Reason:** (b), `nx.amd.device`'s loader, whose kernel descriptor parsing
  is the same layout, read once.
- **Pinned by:** the `recorded cases` of `Tolk.Ops_amd`, whose fixtures hold
  one kernel each; `nx.amd.code_object`'s suite for the layout.

## D121. AMD's packet layouts are nx.amd.packet's

- **tinygrad:** `runtime/ops_amd.py:60-136` (`dispatch_packet`, `pkt3`,
  `wreg`, `pred_exec`, `wait_reg_mem`, `acquire_mem`, `release_mem`),
  `:366-399` (`exec`'s dispatch), `:430-443` (the AQL indirect buffer) and
  `:470-500` (`AMDSDMAQueue`'s packets); the PM4, SDMA, HSA and register
  constants of `runtime/autogen/am` and `runtime/autogen/hsa.py`.
- **tolk:** `lib/runtime/ops_amd.ml:71` (`term`, `lower`), `:256`
  (`compute_queue`) and `:944` (`copy_queue`).
- **Differs:** every packet's layout, the GC registers it writes and their
  addresses, the scratch ring's `COMPUTE_TMPRING_SIZE`, and the constants they
  read come from `nx.amd.packet`, which nx's AMD runtime encodes its copies
  and its KIQ's packets with. Its packets are polymorphic in the values they
  place, and a layout's arithmetic on a value is a term of additions, right
  shifts and ors, with no step of 0: the pieces of an SDMA copy at their
  offsets, the program's and the scratch's addresses from their bit 8, and
  the scratch descriptor's bit 63. tolk lowers them to the nodes tinygrad's
  call sites build: a constant word is a `uint32` constant, an addition adds
  a `uint64` constant, a shift is by a weak literal, and an or is of a
  constant. A PM4 dispatch's register writes and user SGPRs are one packet
  (`Pm4.dispatch`), built from the code object's kernel, which nx's launches
  build too; its scratch descriptor's base is the scratch address the die's
  part adds 0 to, where tinygrad's is the address itself, a node no golden
  holds since no recorded kernel reads a scratch descriptor. Arithmetic
  tinygrad spells otherwise stays at tolk's call site: the program's address
  (`+` of a weak literal), the die's scratch (`+` of a weak literal, kept at
  0), the dispatch packet's address (`+` of a weak literal), the AQL
  dispatch's grid (a product) and kernel object, the AQL indirect buffers'
  addresses, the timestamp slots, the SDMA signal's high word (`lsr` of a
  `uint64` constant), and the profiling copies' addresses.
  GFX9's wait on a UCONFIG register addresses it from UCONFIG's start in the
  packet, where tinygrad subtracts at the call. An SDMA fence takes a memory
  type on SDMA from version 5, where tinygrad asks whether the graphics
  target's major is not 9: the same engines on every GPU tolk supports. A PM4
  predicated block's count is the packet encoded again with it, where tinygrad
  patches the count's bits in place.
- **Reason:** (b), `nx.amd.device`'s SDMA queue and KIQ, whose packets are
  the same layouts, defined once.
- **Pinned by:** the `recorded cases` of `Tolk.Ops_amd`; `Tolk.Ops_amd ›
  packets`, a law per packet that tolk's encoding, evaluated, is
  `Nx_amd_packet.dwords`'s; `nx.amd.packet`'s suite for the words.

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
  `lib/runtime/support/hcq2.ml:1222` (`range_placement`), `:1246` (`stops`)
  and `:1356` (`on_host` in `sched_batches`), and `engine/tolk_engine.ml:1036`
  (`Loop` in `link`) and `:1290` (in `run_call`).
- **Differs:** an `Op.Backedge` stands around a call of a schedule, or a
  linear of calls, with a loop range of at most as many trips and a flag, one
  boolean of storage the calls write. The schedule keeps it as an entry; hcq2
  batches its calls alone, trip by trip, with the range read as a variable,
  and runs a range around one from the engine; the engine reads the flag
  before each trip and runs the calls while it holds. tinygrad's schedules run
  each call once, and stop on no value.
- **Reason:** (b). rune's `Rune.iterate` under `Rune.jit` stages a loop that
  stops on a condition of its carry as one body compiled once, which no
  schedule tinygrad makes can run.
- **Pinned by:** the Engine suite: `link and run › a loop runs its call while
  its flag holds, at most its trips`, `› a loop's flag that views wider
  storage is refused`, `› a planned buffer a loop writes is not placed over
  one it leaves`, `› a loop's planned flag is not placed under a buffer before
  it` and `Metal › a loop runs its batch while its flag holds, at most its
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
  stops, and a process with domains cannot fork. NVRTC and MTLCompiler
  compile in this process and cannot be stopped either. A limit would reach
  only Clang and comgr, the toolchains that run as processes, and give the
  search a promise that depends on the backend.
- **Pinned by:** an absence; the bound tolk keeps is pinned by D115's
  `Tolk.Search › BEAM_PADTO=1 ... › a candidate of BEAM_UOPS_MAX
  instructions or more is not compiled`.

## D125. A launch descriptor's field holds all of its value

- **tinygrad:** `runtime/ops_nv.py:73` (`QMD.write` patches a value as the
  widest unsigned word within its field).
- **tolk:** `lib/runtime/nv_packet.ml:53` (`structure`), over
  `Nx_nv_packet.hole`, whose holes are their fields' bit ranges; the golden
  generator, `test/gen/runtime/ops_nv.py` (`write`, `holes`).
- **Differs:** a field a value fills is a hole of the narrowest unsigned word
  that covers it. When the field is narrower than its word, the word is the
  value masked to the field's bits, ored with the word's other bits as the
  descriptor holds them.
- **Reason:** (c). The descriptor's layout is nx.nv.packet's, which tolk's
  queues and nx.nv.device share, and a word within the field drops the
  value's high bits. Version 3's
  24-bit `shader_local_memory_high_size` took a 2-byte word, so a kernel with
  64 KiB or more of local memory per thread ran with that size modulo
  64 KiB, and the 17- to 25-bit high words of addresses (program, prefetch,
  constant banks, releases) dropped bit 48 and above. The law that nodes and
  integers encode alike cannot see it, since both truncated alike.
- **Pinned by:** nx's `nx.nv.packet › launch descriptors › a local memory of
  64 KiB per thread fills its field`, `› an address keeps bit 48`, `› a hole
  keeps the fields it shares bytes with` and `› holes cover their fields, in
  order`; the Ops_nv suite's recorded cases, whose generator applies the same
  holes to tinygrad.

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
- **tolk:** `lib/codegen/late/gater.ml:14` (`holds`) and `:25`
  (`move_where_load`); `test/gen/tinygrad.patch`, which gives tinygrad the same
  rule before the goldens are generated.
- **Differs:** the selection moves into the load only when the load's type
  holds its other value: when the load is not converted, when the value is a
  conversion of a value of the load's type, or when it is a constant that
  converts to the load's type and back to its bits, sign included. Otherwise
  the selection stays. tinygrad converted the value to the load's type and
  back, so `where gate (float32 (load float16)) 0x1p-149` read 0 where the gate
  failed, a non-integer or a `-0.` read through an integer load lost its
  fraction or its sign, and a float32 above 65504 read through a float16 load
  read infinity.
- **Reason:** (b). rune's pad of a widened float16 with a float32 pad value
  (`Nx.pad` of `Nx.cast Nx.float32 x`), read through a slice, compiled to such
  a selection: a pad of `0x1p-149` read 0, where eager reads `0x1p-149`.
- **Pinned by:** the Gater suite (`test/codegen/late/gater`):
  `pm_move_gates_from_index › a selection of a converted gated load keeps its
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
  `lib/engine/worker.ml:77` (`iter`); `engine/tolk_engine.ml:1369`
  (`clock`).
- **Differs:** a round's candidates are timed in their own order, each once
  its compilation has ended, while other domains compile the later ones
  (`Worker.iter`). Where the host's clock times the runs (`Search.Host`: a
  program the host calls, any program while a profile is taken, or a search
  whose kernel was not timed), a round compiles all its candidates before it
  times any. The search reads the clock from its kernel's linked program.
- **Reason:** the compute filter's running minimum, `seen_libs` and ties
  between equal samples read the order: timed as their compilations end, two
  searches of equal measurements choose apart, and `searches.golden` could
  not pin tinygrad's choices. A device stamps its runs on its own clock, but
  the host times its runs on its own cores, which compilations on the other
  cores slow. Admitting this is the maintainer's call, as for D115.
- **Pinned by:** `Tolk.Search › a search on a device's clock times its
  candidates as one on the host's, whatever order their compilations end
  in` and `› a search on the host's clock compiles nothing while it times`;
  `Tolk.Worker › streaming` (every test); `Tolk_engine › timing › the clock
  is the device's where it has queues, the host's otherwise` and `› the
  clock is the host's under a profile`.

## D131. On the host, a kernel without a reduce whose accesses merge is upcast only along its masked axes

- **tinygrad:** `codegen/opt/heuristic.py:112-138` (the upcasts of output
  axes by 3 or 4) and `:156-159` (the upcast by 4 of a kernel that has none).
- **tolk:** `lib/codegen/opt/heuristic.ml:444` (`host_merged`) and `:630`
  (`hand_coded_optimizations`); `lib/codegen/late/coalesce.ml:53` (`merges`);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** on the host (`target.device = "CPU"`), a kernel with no reduce
  and an access whose elements merge into vectors (`Coalesce.merges`:
  float32, float16, int32, uint32, the 8-bit floats) takes neither upcast.
  Its masked axes are still upcast whole. A kernel with a reduce, or whose
  accesses do not merge (float64, 64-bit and 8-bit integers), is upcast as
  tinygrad's is. Upcast lanes of a merging type read one vector and compute
  each element in scalars, which keeps Clang's loop vectorizer off the loop it
  vectorizes unasked. Lanes of a type that does not merge stay scalar, and
  Clang vectorizes the four lanes two at a time, eight chains at once, which a
  chain bound by latency gains from.
- **Reason:** (b): RFC 0025's special functions on the host. Their kernels,
  compiled by Clang -O2 and timed directly on one core of an M1 Max (10^5
  elements, median of 31 alternated runs), ran without the upcast in 0.23 to
  0.71 of their upcast time in float32 (`exp`, a sigmoid, `erf`, `erfinv`,
  `sin`; a sum at 1.00). In float64 the upcast won on chains bound by latency
  (`erfc` 1.62 times as long without it, `erfinv` 1.46, `ndtr` 1.39, `ndtri`
  1.36, `lgamma` 1.28) and lost on wide ones (`sin` 0.77, `tanh` 0.92), so
  float64 keeps it.
- **Pinned by:** the Heuristic suite: `the optimisations chosen are
  tinygrad's › applied_opts`, cases `add_cpu`, `add_broadcast_cpu`,
  `add_large_cpu`, `add_small_cpu`, `outer_add_cpu`, `transpose_cpu`,
  `stack_8_cpu` and `softmax_cpu` (float32, no upcast; `stack_cpu` and
  `pad_7x7_cpu` keep their masked upcasts); the Codegen suite's `long_clang`,
  `uint64_clang` and `int8_clang`, upcast; all recorded from the equally
  patched tinygrad.

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
  after effects is a view of that storage, whatever the effects compute
  (D132)`.

## D133. A greatest common divisor's coefficient is a scalar

- **tinygrad:** `uop/ops.py:1083-1088` (`UOp.gcd`, whose coefficient is
  `uops[0].const_like(...)`, a constant broadcast to the shape of the first
  term) and `uop/divandmod.py:77-81` (`divide_by_gcd`, which divides every
  term by a divisor that is not the `CONST` 1). Over terms with a shape, a
  vector of index expressions or a weak-integer tensor, a divisor of 1 is a
  broadcast 1, which is not the `CONST` 1 and divides no term:
  `UOp.stack(a, b) // 3` raises in `unwrap`.
- **tolk:** `lib/uop/ops.ml:3904` (`gcd`).
- **Differs:** the coefficient is a scalar constant of the first term's type,
  whatever the terms' shape, so a divisor of 1 is the `CONST` 1, and a larger
  one divides each term as `divides` does.
- **Reason:** (b): the norn compile D132 names reached `divide_by_gcd`, before
  D132, on a weak-integer tensor of rune's gather positions, and raised there.
  The rule applies to any vector of index arithmetic.
- **Pinned by:** the `Divandmod` suite: `values › each rewrite of a random
  division of vectors keeps each lane's value (D133)`.

## D134. A widening cast of an unsigned mask or shift widens its operands

- **tinygrad:** `uop/symbolic.py:304-308` (cast/long folding), which keeps a
  cast of `x & y` or `x >> k` as it is.
- **tolk:** `lib/uop/symbolic.ml:1005` (`symbolic`'s cast rules);
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
- **Pinned by:** the Symbolic suite (`test/uop/symbolic`): `symbolic › casts ›
  a widening cast of an unsigned mask masks the widened operand (D134)` and
  the four tests after it; rune's lower_index suite: `quantised products › a
  product unpacks its codes at the width it decodes them in`.

## D135. A Metal command buffer signals through the device's signaler

- **tinygrad:** `runtime/ops_metal.py:85-86` (`HANDLES`, `SELECTORS`),
  `:162` (`MetalQueue.submit`: the last command buffer encodes
  `encodeSignalEvent:value:` on the device's event).
- **tolk:** `lib/runtime/ops_metal.ml` (`handles`, `selectors`, and the
  signal in `submit`); `engine/metal.macos.ml` (`sels`).
- **Differs:** the host program sends every command buffer to the device's
  signaler (`Nx_metal_device.signaler`), `[signaler signal:cb value:v]`, before
  it commits it: the last with the batch's value, the others with `0`. No
  command buffer encodes a signal on the event, and the `mtl_sel` words hold
  no `event` handle, since the signaler owns the event: the handle `signaler`
  follows tinygrad's others, and the selector `signal:value:` follows `retain`
  and `release`.
- **Reason:** (c). A command buffer that Metal fails still runs the signal
  encoded in it, so a GPU-side signal reports failed work as complete.
  nx.device's Metal library signals a value from the command buffer's
  completed handler once it completed, and loses the device with Metal's
  reason otherwise; the runtime loses a device only on its driver's report.
- **Pinned by:** the Ops_metal suite (`test/runtime/ops_metal`): every
  `recorded cases` golden, whose host programs are tinygrad's with D135
  applied by their generator (`gen/runtime/ops_metal.py`); on macOS, every
  `execution` test, which waits for its batches through the signaler; and
  `nx.metal.device › failures › a command buffer Metal fails loses the device
  with Metal's reason`.

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
- **tolk:** `lib/schedule/indexing.ml:451` (`region`), `:481`
  (`share_loops`), `:533` (`loops_of`) and `:145` (`end_shared`, which
  `bufferize_and_index` returns for each store of a group);
  `test/gen/tinygrad.patch`, which gives tinygrad the same before the goldens
  are recorded.
- **Differs:** before ranges are assigned, the stores of the tensor graph that
  have a loop are grouped. Two join when their values have one shape and one
  device, they write different storage, neither reads the other or the
  storage the other writes, through anything, and the computed nodes each
  reaches before storage or another stored value share one. Neither may reach
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
  recorded `shard_of_computed_kernels` and `kernel_counts`, and the
  Postrange suite's `where_max_multioutput` cases, recorded from the equally
  patched tinygrad; rune's `Rune nx.wide › kernels`.

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
  the full specification, whatever SPEC (D139)`.
