# Divergences from tinygrad

tolk.next follows tinygrad `79af1ca70e7021f504919c4ff5631245acc33ed6`. This
ledger lists every place where it does something else. An entry is admitted
for one of three reasons only:

- **(a) an OCaml constraint:** acyclic modules, static types, the GC, domains;
- **(b) a named consumer:** a raven call site that fails without it;
- **(c) the executor's contract:** nx.device's submission protocol, which
  `tolk.next.engine` drives with the compiler's output.

Taste, speed without a measurement, and "the old tolk did it" are not reasons.
A difference in numerics also needs a failing rune test, since nx semantics
belong in rune's lowering.

Each entry gives the tinygrad file and line, the tolk.next file and line, what
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
- **tolk.next:** `lib/runtime/support/hcq2.ml:212` (`signal_word`), `:224`
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
  the batch's memory`; and the Engine suite (`test/engine/tolk_next_engine`):
  `batches › each run of a batch signals its device's next value once`.

## D2. Withdrawn

tinygrad matches patterns without generating code when `UPAT_COMPILE` is 0
(`uop/ops.py:1545`, `upat_interpret`), and `UPat`, `PatternMatcher` and
`graph_rewrite` live in `uop/ops.py`. tolk.next ports that path in
`Ops`, so leaving out `uop/upat.py`, the pattern compiler, is scope: see
the Exclusions of `README.md`.

## D3. device.py and the ops_*.py files are split

- **tinygrad:** `device.py`, `runtime/ops_*.py`.
- **tolk.next:** `lib/device.ml` (the compiler half of `device.py`);
  `lib/runtime/ops_metal.ml`, `ops_cuda.ml`, `ops_amd.ml` and `ops_nv.ml` (the
  IR half of each `ops_*.py`); `engine/tolk_next_engine.ml:56` (`device`) and
  the vendors `engine/metal.macos.ml`, `cuda.ml`, `amd.ml` and `nv.ml`;
  `lib/uop/ops.ml:229` (the type `param_arg`), `:3225` (`param_arg`), `:3253`
  (`new_buffer`).
- **Differs:** tolk.next holds the compiler half: `Compiler`, the renderer
  and compiler selection of `Compiled`, and the IR half of each `ops_*.py`
  (queues, `pm_encode`, program data), all returning data. The lazy `Buffer`
  and running a schedule are `tolk.next.engine`'s, which has no registry of
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
- **Pinned by:** for `Ops`: `Tolk_next.Ops › storage › new_buffer takes the
  next slot without one` and `Tolk_next.Ops › arguments › reprs.golden`, where
  a `BUFFER` prints and interns by its slot alone; for the queues: the
  `recorded cases` of `Tolk_next.Ops_metal`, `Tolk_next.Ops_cuda`,
  `Tolk_next.Ops_amd` and `Tolk_next.Ops_nv`, which encode each vendor's
  queues as nodes on a machine without its device; for the engine: the Engine
  suite (`test/engine/tolk_next_engine`): `device › a name the map does not
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
- **tolk.next:** for `uop/ops.py`, `lib/uop/ops.ml:572`
  (`construction_check`), `:1685` (`simplify_hook`), `:4614` (`Private`),
  `:1190` (`Make_elementwise`), `:816` (`repr`), `:244` (`bufferize_opts`),
  `:262` (`Calls`); `lib/uop/render.ml:202` (`render`), `:212` (`srender`);
  `lib/renderer/renderer.ml:119` (`Compiler`); `lib/schedule/prepare.ml:665`
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
  - `apply_opts` takes the optimiser as an argument, and `Search` follows
    `Postrange` and takes the timing of a kernel as its `measure` argument;
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
  `Tolk_next.Codegen › beam search (D4) › a kernel that asks for a beam of
  width w is optimised by beam w` and `› raises Invalid_argument when a kernel
  asks for a beam and none is given`; for `Ops`: `Tolk_next.Ops › resolve ›
  simplify leaves a constant, and a sink of constants and stacks of
  constants, alone`, and the `Tolk_next.Ops › resolve` tests that simplify with
  `Symbolic`'s rules; `Tolk_next.Ops › printing › pretty.golden`;
  `Tolk_next.Ops › elementwise patterns › the pattern operators are the named
  pattern operations`; `Tolk_next.Ops › queue calls › pp_hcq_info formats
  every field, as the record's repr`; for `Renderer`: `Tolk_next.Renderer ›
  Compiler › compile raises what the toolchain rejects`, its
  `Compile_error`; for `Prepare`: `Tolk_next.Prepare › contiguous_view › a
  reshape keeps the order`; for `Schedule`: `Tolk_next.Schedule ›
  pm_flatten_linear › Linears nested in Linears are inlined in order`; for
  the engine's order: `Tolk_next.Realize › lower_and_compile with a beam
  search › raises Invalid_argument for a kernel that asks for one when none is
  given`, `Tolk_next.Hcq2 › compile_linear › makes each kernel ask for a beam
  of the width BEAM sets`, which passes the search, and `› copies through the
  halves of a staging buffer of the host where the queues cannot reach`, whose
  caller describes the devices and their queues; for `Compiler_metal`:
  `Tolk_next.Compiler_metal › MTLCompiler › compiles a kernel to a Metal
  library`. Each calls the function where its break puts it, or passes the
  late binding as an argument, so none compiles without the break.

## D5. Compilation workers are domains

- **tinygrad:** `engine/worker.py:1-2` (`multiprocessing` spawn workers);
  `helpers.py:169-186` (`Context` and `ContextVar`, one value per process);
  `codegen/__init__.py:495-505` (`to_program_context`, the settings a worker
  process is started with, and `to_program_cache`, which the parent fills).
- **tolk.next:** `lib/engine/worker.ml:10` (`spawned`) and `:21` (`map`);
  `lib/helpers.ml:106` (`Context_var`) and `:133` (`context`);
  `lib/codegen/codegen.ml:1008` (`to_program`'s cache);
  `lib/runtime/support/compiler_metal.ml:38` (`build`).
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
  thread-safe: its builds run one at a time.
- **Reason:** (a).
- **Pinned by:** `Tolk_next.Helpers › context › is not seen by the other
  domains` and `Tolk_next.Helpers › context › binds for the domains spawned
  while it runs`; `Tolk_next.Worker` (every test); `Tolk_next.Codegen ›
  programs are kept › calls from several domains at once make one program,
  compiled once (D5)`; `Tolk_next.Compiler_metal › MTLCompiler › compiles
  from several domains at once`.

## D6. Devices are named, never parsed

- **tinygrad:** `device.py:26,36,395,491` (device strings split at `:`).
- **tolk.next:** `lib/device.ml:122` (`renderers`), `:145` (`renderer`).
- **Differs:** the caller gives a target, a device name and its `arch`;
  tolk.next picks the renderer and compiler from the `arch` and never parses
  the name, with one exception: a name starting with `DISK` is a disk, as
  tinygrad reserves it and nx.device names its disk devices
  (`Ops.on_disk`, `is_disk_device`, `copy_to_device`, `clone`).
- **Reason:** (c).
- **Pinned by:** for targets, `Tolk_next.Device › renderer takes a device's
  name as it is (D6)` (a name with an index, in lower case, or a disk's, is no
  device, and a name gives no renderer or architecture) and `Tolk_next.Device ›
  renderer › picks a device's renderer as tinygrad does` (the architecture
  comes from DEV or the caller); for the disk, `Tolk_next.Ops › several
  devices › on_disk holds for one disk device`, `Tolk_next.Ops › several
  devices › copy_to_device rejects a disk and a weak type` and `Tolk_next.Ops ›
  storage › clone rejects a disk`.

## D7. Stamp slots follow `Submission.record`

- **tinygrad:** `runtime/support/hcq2.py:231,240,301`.
- **tolk.next:** `lib/runtime/support/hcq2.ml:713` (`stamps`), `:638` (the
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
  its device, in order`; and the Engine suite (`test/engine/tolk_next_engine`):
  `batches › a kernel is a span of its compute lane and a copy of its copy
  lane`.

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
- **Pinned by:** `Tolk_next.Helpers › Diskcache › get reads back any key and
  value put` and `Tolk_next.Helpers › Diskcache › behaves as a table of
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
- **tolk.next:** `lib/dtype.ml:470-491` (`encode_format`, one rounding for
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
- **Pinned by:** `Tolk_next.Dtype › truncate › truncation.golden` (the
  near-tie rows, stated in code), `Tolk_next.Dtype › truncate › bfloat16 rounds
  once, to the nearest, ties to even` and `Tolk_next.Dtype › truncate › an
  integer rounds to a narrower float once, from its value`; for emulation, one
  test per facet in `Tolk_next.Decomp_dtype › D9`: `an emulated narrow float
  keeps its subnormals, both ways`, `a 16-bit float overflows from its greatest
  value plus half an ulp, the tie to infinity`, `an infinity stays one in e5m2
  and is the NaN of e4m3 and the fnuz formats, and a finite overflow
  saturates`, `a double, or an integer more precise than a float32, rounds
  once`, `an emulated 64-bit integer converts to a float32 once`, `an fnuz
  format stores an underflow to negative zero as positive zero`;
  `Tolk_next.Decomp_dtype › goldens`, which are tinygrad's graphs with
  `f2f` and `f2f_clamp` replaced by these conversions.

## D10. A float8 NaN keeps its sign when decoded

- **tinygrad:** `dtype.py:279` (`fp8_to_float` returns `math.nan` for
  e4m3's NaN codes, whatever their sign bit).
- **tolk.next:** `lib/dtype.ml:431-448` (`decode_format`).
- **Differs:** e4m3's `0xff` decodes to a negative NaN, so it encodes back to
  `0xff`, where tinygrad gives `0x7f`. e5m2 already kept the sign.
- **Reason:** (b). nx's decoder keeps the sign, so a value read eagerly and a
  folded constant agree, and every NaN code keeps its sign through a round
  trip.
- **Pinned by:** `Tolk_next.Dtype › storage › a NaN decodes with the sign of
  its bits`.

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
- **Pinned by:** the type itself, whose printing `Tolk_next.Ops › arguments ›
  reprs.golden` checks against tinygrad's `Opt` repr (the row "kernel with
  opts"); `Tolk_next.Postrange` drops tinygrad's malformed-argument cases with
  this entry as the reason.

## D12. A node's key is a BLAKE2 digest

- **tinygrad:** `uop/ops.py:266-268` (`key`, the SHA-256 of
  `str((op, dtype, arg))` followed by the keys of the sources).
- **tolk.next:** `lib/uop/ops.ml:1063` (`key`).
- **Differs:** the same text is digested with BLAKE2b-256.
- **Reason:** (a). No key is ever compared with a key tinygrad computed: keys
  name compiled programs in caches that tolk.next alone writes, so the digest
  is free to differ, and tolk.next takes the one OCaml's standard library
  has, which has MD5 and BLAKE2 but not SHA-256.
- **Pinned by:** `Tolk_next.Ops › key › ignores tags` and `Tolk_next.Ops ›
  key › tells arguments apart`; for the profile keys of batched kernels, the
  Hcq2 suite (`test/runtime/support/hcq2`): `profile keys (D12)`, whose
  recorded graphs are compared without them.

## D13. Folding reads and writes committed constants at their width

- **tinygrad:** `uop/symbolic.py:29` (`fold_const_alu`, whose result keeps
  the unwrapped value, `truncate_output=False`) and `:154-155` (the collapse
  of committed const conversions), which read a constant with `UOp.val`
  (`uop/ops.py:259-263`), unwrapped; `uop/ops.py:1104-1163` (`_min_max`),
  which bounds a weak operand of a committed operation by its unwrapped value;
  `uop/weak.py:82-87` (`uncast_const`), which leaves the literal bare.
  tinygrad's own `TestModularWraparound` expects the wrapped results and is
  marked `xfail_broken_const_wraparound`.
- **tolk.next:** `lib/uop/symbolic.ml:100` (`fold_const_alu`) and `:467`;
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
  `2`, since `-c` folded to `-254`. A maximum by bounds that keeps a weak
  operand commits it to the maximum's type, so the operations that read it do
  not lose their width. Reading at the width also keeps
  `c0 + x < c1 → x < c1 - c0` from a wrong answer where the offset wraps: on
  uint8, `u + uint8 -1 < 255` would become `u < 0`; D24 keeps that rule to
  offsets that do not wrap.
- **Reason:** (b): rune's `Nx.Rng` (Threefry), jitted with constant keys,
  must draw the numbers eager nx draws, and RFC 0012's Law 1: every integer
  expression's compiled value is eager nx's modular one.
- **Pinned by:** `Tolk_next.Symbolic › symbolic_simple › constants › an
  operation reads committed constants at their width` (a fold, a cast of a
  committed constant, a uint8 remainder), `Tolk_next.Symbolic ›
  symbolic_simple › constants › a comparison reads a committed constant at its
  width` (the uncast), `Tolk_next.Symbolic › committed constants hold their
  type's value (D13)` (each case above, evaluated before and after), the law
  `Tolk_next.Symbolic › laws › sym keeps the value of an integer expression at
  a committed width, wrapping included`, and `Tolk_next.Symbolic › tinygrad ›
  tests.golden › TestModularWraparound.<test>` and
  `› TestThreefryConstFolding.test_threefry`, whose goldens, generated with
  the same change in `test/gen/tinygrad.patch`, hold the machine values;
  `Tolk_next.Ops › bounds › a typed integer constant outside its type is
  bounded by its wrapped value, a non-finite one by the type (D13)`; the
  offset's interaction with D24: `Tolk_next.Symbolic › integers wrap (D24) ›
  an offset crosses a comparison only where neither side wraps` (the
  committed case).

## D14. Folded comparisons treat NaN as IEEE does

- **tinygrad:** `uop/symbolic.py:29` (`fold_const_alu`), whose operands come
  from `UOp.val` (`uop/ops.py:259-263`) as `ConstFloat`s (`dtype.py:8-22`).
  `ConstFloat` makes NaN equal to NaN on purpose, so that NaN constants
  intern as one node, and `exec_alu` compares with it: `nan != nan` and
  `nan < nan` fold to `False`, where `exec_alu` on floats gives `True` and
  `False`.
- **tolk.next:** `lib/uop/symbolic.ml:100` (`fold_const_alu`); constants are
  interned by `Dtype.equal_const`, and `exec_alu` compares floats.
- **Differs:** a folded comparison of NaN constants follows IEEE: `nan <> nan`
  is `true`.
- **Reason:** (b): `Nx.not_equal x x` on a NaN is `true` in eager nx, and
  rune's jitted graph, whose constants fold here, must agree.
- **Pinned by:** `Tolk_next.Symbolic › symbolic_simple › constants › NaN is
  unequal to itself when constants fold, as IEEE says`.

## D15. Compilers load their library at the first compile

- **tinygrad:** `runtime/support/compiler_cuda.py:62` (`NVRTCCompiler`
  calls `nvrtcVersion`), `runtime/support/compiler_amd.py:80` (`HIPCompiler`
  asserts comgr is loaded) and `runtime/ops_metal.py:39` (`MetalCompiler`
  creates its code generation service), each when the compiler is made, so
  making the renderer that holds it fails without the library.
- **tolk.next:** `lib/runtime/support/compiler_cuda.ml` (`nvrtc`),
  `lib/runtime/support/compiler_amd.ml` (`hip`),
  `lib/runtime/support/compiler_metal.ml` (`compiler`).
- **Differs:** making a compiler loads nothing. The library is loaded at the
  first compile, once per process, and a compile without it raises
  `Compile_error` with the reason.
- **Reason:** (b): `Cstyle`'s CUDA, HIP and Metal renderers are made, and
  render, on machines without NVRTC, comgr or MTLCompiler: the source goldens
  of the `Cstyle` suite and rune's rendering of a kernel for inspection.
- **Pinned by:** `Tolk_next.Compiler_cuda › a library that does not load`,
  `Tolk_next.Compiler_amd › a library that does not load` and
  `Tolk_next.Compiler_metal › a library that does not load`, run where the library's
  variable names a file that is no library: making the compiler succeeds, each
  compile raises `Compile_error`, and a cached binary is served without a load.
  Where the library is absent, `› without NVRTC on the machine`,
  `› without comgr on the machine` and `› without MTLCompiler on the machine`
  pin the error that names the library and its variable.

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
- **Pinned by:** `Tolk_next.Cstyle › float8 infinities on CUDA (D16)`, which
  checks where the guard is declared, the byte each infinity writes and the
  bits of infinite e5m2 constants; `Tolk_next.Cstyle › sources › by default ›`
  `cuda_dtype_float8_e4m3`, `cuda_dtype_float8_e5m2`, `cuda_inf_nan_float8_e4m3`
  and `cuda_inf_nan_float8_e5m2`, which compare with tinygrad's source once the
  guard is written back as tinygrad writes it; and `Tolk_next.Cstyle › every
  GPU kernel compiles with its target's toolchain › cuda_*` (slow, skipped
  without NVRTC).

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
- **Pinned by:** `Tolk_next.Cstyle › narrowing (D17)`, which checks for every
  kernel that the kernel whose tinygrad source is its source is the kernel
  with those casts, and `Tolk_next.Cstyle › sources`, which compares the 21
  sources it changes with tinygrad's for that kernel; `Tolk_next.Cstyle ›
  execution on the host › wraps each operation on unsigned chars (D17)` and
  `› rounds each operation on halves to a half (D17)`; and the slow
  `Tolk_next.Cstyle › execution on the host › a kernel over a narrow type wraps
  and rounds as the interpreter (D17)`, over each type's whole range.

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
- **Pinned by:** `Tolk_next.Cstyle › bfloat16 truncation on Metal (D18) ›
  truncates a bfloat16 in float32` and `› leaves it to CUDA, which truncates a
  bfloat16 with htrunc`, and the slow `› compiles a kernel that truncates a
  bfloat16`; `Tolk_next.Cstyle › every GPU kernel compiles with its target's
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
- **tolk.next:** `lib/schedule/allreduce.ml:104` (`handle_allreduce`).
- **Differs:** when the target is one device, the hierarchical branch copies
  each reduced chunk there from the device of its rank in the first node, as
  the ring and all-to-all branches copy their reduced chunks, and the result
  is on that device.
- **Reason:** (b): rune's multi-device reductions. A sharded sum reduced to
  one device under `ALLREDUCE_NODE_NDEVS`, the multi-node setting, would
  otherwise compute a value on every device and store it into storage on
  one.
- **Pinned by:** `Tolk_next.Allreduce › handle_allreduce › recorded ›
  nodes_to_one_device_handled.golden, landed on its device (D19)`, which
  states tinygrad's golden in code with each gather of a chunk replaced by its
  copy to the target, and `Tolk_next.Allreduce › rules › a hierarchical
  allreduce to one device lands there (D19)`, which evaluates the value and
  the function on two devices and finds the reduction on the target alone.

## D20. Storage keeps an e5m2 NaN's payload

- **tinygrad:** `dtype.py:251,278` (`float_to_fp8` stores every e5m2 NaN as
  `0x7f` of its sign, and `fp8_to_float` decodes every NaN code as `math.nan`
  of its sign).
- **tolk.next:** `lib/dtype.ml:412-448` (`nan_of_payload`, `nan_payload` and
  `decode_format`) and `:470-491` (`encode_format`).
- **Differs:** storage moves an e5m2 NaN's two payload bits through the double,
  as it moves the payload of the 16-bit formats, so `bitcast` keeps every e5m2
  code: `0x7d` and `0x7e` come back as themselves, where tinygrad gives `0x7f`,
  and the canonical NaN stores as `0x7e`, its quiet code. A conversion still
  gives `0x7f` of its sign, as nx's encoder does.
- **Reason:** (b). A kernel's bitcast and nx's are byte reinterpretations, so
  a bitcast that rune folds must keep the bits as they do.
- **Pinned by:** `Tolk_next.Dtype › storage › a bitcast through a float
  gives back every 8- and 16-bit word`, on every e5m2 code, and the NaN rows
  of `Tolk_next.Dtype › storage › reencode.golden` and `Tolk_next.Dtype ›
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
- **tolk.next:** `lib/schedule/multi.ml:362` (`reshape_multi`).
- **Differs:** each sharded axis of the new shape is divided by the shard
  count of its own range, so the shard of the repro reshapes to `(2, 3, 1)`.
  When the sharded axes share one count, the two agree.
- **Reason:** (b): rune's multi-axis placement (RFC 0005's meshes, data by
  tensor parallelism), where a mesh's two axes have different shard counts.
- **Pinned by:** `Tolk_next.Multi › multi_pm › two sharded axes › a reshape
  divides each sharded axis by its own count (D21)` and `› a reshape of a mesh
  of 2 by 4 devices keeps each tile (D21)`, and the law `Tolk_next.Multi ›
  multi_pm › laws › a rewritten value holds the value computed whole` on grids
  of 2 by 2 and 2 by 4 devices.

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
- **Pinned by:** `Tolk_next.Decomp_dtype › emulated 64-bit integers › an
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
- **tolk.next:** `lib/schedule/multi.ml:131` (the rule), with `:54`
  (`at_device`).
- **Differs:** the movement's arguments take the selected shard's position
  for the device range, as a shrink moved before an `MSTACK` already does
  (`_apply_shrink`), and are simplified.
- **Reason:** (b): rune's sharded programs, where a shard is selected from a
  value computed from a sharded and a replicated one.
- **Pinned by:** `Tolk_next.Multi › multi_pm › shard selections › a
  selection of a movement by the device range takes the selected device's
  position (D23)`, and the law `Tolk_next.Multi › multi_pm › laws › a
  rewritten value holds the value computed whole`.

## D24. Rewrites keep IEEE and modular values

- **tinygrad:** the rules and bounds each facet names below.
- **tolk.next:** each facet's lines below; `test/gen/tinygrad.patch`, the same
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
    `:1154-1164` (a cast), `:1147-1149` (a constant table). tolk.next:
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
    (`x + y <> c` under a cast). tolk.next: `lib/uop/symbolic.ml:862,871`,
    `lib/uop/divandmod.ml:252`, `lib/codegen/simplify.ml:252,268,276,344`.
    Each applies to a committed integer only where every value it computes
    fits the type; the comparisons of `Simplify` apply to integers only, since
    moving a float term rounds, and `x + y <> c` only under a cast that does
    not narrow. tinygrad folds uint8 `u - 1 < 255` for every `u`, int32
    `(w // 2 + 2^30) // 2` at `2^30` to `-268435456` where the machine gives
    `805306368`, and `(w // 65536) // 65536` to a division by `2^32`, which
    wraps to `0`.
  - **Float folds.** tinygrad: `uop/symbolic.py:117` (`x + 0`), `:170-176`
    (`x / x`, `(x * y) / y`, `x * 0`), `:247` (`(x / y) / z`). tolk.next:
    `lib/uop/symbolic.ml:348,521`. A float `x + 0` is `x` only for `-0.`
    (`-0. + +0.` is `+0.`); `x * 0` is `0` for integers and booleans only (a
    float product by zero is NaN at an infinity or a NaN and `-0.` at a
    negative `x`); `x / x`, `(x * y) / y` and `(x / y) / z` are gone, since
    true division is always a float: `x / x` is NaN at 0, `(1e10 * 1e30) /
    1e30` is `inf`, and `(1e20 / 1e20) / 1e20` is `1e-20` where
    `1e20 / 1e40` is `0.`.
  - **Signed zeros.** tinygrad: `uop/symbolic.py:248` (`-(x + c)`), `:267`
    (complementary selections), `:472` (`-(x + y)`). tolk.next:
    `lib/uop/symbolic.ml:708,776,1316`. For integers and booleans only:
    `-(x + 3)` at `x = -3` is `-0.`, where `-x + -3` is `+0.`, and
    `where c t 0 + where c 0 f` at `t = -0.` is `+0.`, where `where c t f` is
    `-0.`.
  - **Reassociation.** tinygrad: `uop/symbolic.py:240-246` (like terms),
    `:264-265` (a sum of two selections), `:279-280` (two constants of an
    associative operation), `:293-294` (constants to the end), `:390-398,470`
    (`reduce_mul_chain`). tolk.next: `lib/uop/symbolic.ml:683,764,845,901,1156`.
    Sums, products and maxima regroup for integers and booleans only, and a
    factor leaves a float reduction nowhere: `(x + 1e8) + -1e8` at `x = 1` is
    `0.`, where `x + 0.` is `1.`; `(y + x) + x` at `y = 1`, `x = 2^-24` is
    `1.`, where `y + x * 2` is `1.0000001`; `x * 1.1 + x * 2.2` at `1.` is
    `3.3000002`, where `x * 3.3` is `3.2999999`; a sum over `r < 3` of
    `(r * 2e38) * 0.25` is `inf`, where `0.25 * 2e38` times the sum of `r` is
    finite; and a maximum keeps a NaN only as its first operand, so
    `max (NaN, max (x, 0.))` is NaN, where `max (x, max (0., NaN))` is `x`.
    `x + x` is still `x * 2`, which is exact.
  - **Maxima.** tinygrad: `uop/symbolic.py:273-275`. tolk.next:
    `lib/uop/symbolic.ml:808,828`. A maximum by bounds applies to integers
    only, since a float's bounds leave out NaN and the order of zeros:
    `max (x, inf)` at NaN is NaN, where the fold gives `inf`. A selection that
    computes a maximum becomes one for integers only: rune builds a float
    maximum that follows IEEE from selections, which the fold would turn back
    into a maximum that renders as the target's own, and it took `-0.` and
    `+0.` for one constant: `where (x < -0.) 0. x` at `-1e-45` is `+0.`,
    where `max (x, -0.)` is `-0.`. A float ReLU stays a selection.
  - **Reciprocal and sigmoid forms.** tinygrad: `uop/symbolic.py:463-468`.
    tolk.next: the rules are left out of `sym`. `1 / (x * x)` at `1e20` is
    `0.`, where `(1 / x) * (1 / x)` is `1e-40`; `x * (1 / (1 + x))` at `1e-8`
    is `1e-8`, where `1 - 1 / (1 + x)` is `0.`, and at `inf` is NaN, where it
    is `1.`.
  - **Pow.** tinygrad: `uop/symbolic.py:16-21` (`simplify_pow`), `:190`
    (`c ** x`). tolk.next: `lib/uop/symbolic.ml:61,576`. The reciprocal of the
    base is taken for exponents of magnitude at least 1 only, where the power
    overflows whenever the reciprocal does: `1e-40 ** -0.8` is `1e32`, where
    `(1 / 1e-40) ** 0.8` is `inf`. A float half-integer power selects `+0.`
    at `-0.` and `+inf` at `-inf`, where `sqrt` gives `-0.` and NaN. `c ** x`
    goes to `xpow` for `c = inf`: `inf ** 0` is `1.`, where
    `exp2 (0 * log2 inf)` is NaN.
  - **Simplify's reductions.** tinygrad: `codegen/simplify.py:104-111`
    (`sum_between`) and `:118` (a product by a boolean cast). tolk.next:
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
    `0.0`) and `codegen/decomp/transcendental.py:244` (`xlog2` adds an integer
    `0` outside float32). tolk.next: `lib/codegen/decomp/transcendental.ml:195,343`.
    A polynomial starts from its first coefficient and `xlog2` adds its low
    term only in float32, so the decompositions no longer rely on the float
    folds of `0. * x` and `x + 0` to be what tinygrad renders.
  - **Tensor-core accumulators.** tinygrad: `codegen/__init__.py:102-104`
    (`pm_wmma_add`, which adds the running sum to a WMMA's accumulator
    operand). tolk.next: `lib/codegen/codegen.ml:167`
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
  - integer bounds: `Tolk_next.Ops › bounds › a committed integer that can
    leave its type has its bounds (D24)`, `› an integer cast to a signed type
    it leaves wraps (D24)`, `› a constant table holding a NaN has its type's
    bounds (D24)` and the law `› bounds hold the value a committed integer
    wraps to (D24)`;
  - wrapping rules: `Tolk_next.Symbolic › integers wrap (D24)` (every test);
    `Tolk_next.Divandmod › nested divisions › a committed division stays
    where x + a * c wraps (D24)`; `Tolk_next.Simplify › keeping values (D24)`,
    its comparison tests;
  - float folds, signed zeros, reassociation, maxima, reciprocal and sigmoid
    forms, pow: `Tolk_next.Symbolic › floats keep IEEE values (D24)`, one test
    per facet, with `› a Dekker split and product keep their low parts` and
    `› a float selection that computes a maximum stays`, the structural tests
    marked D24 in `symbolic_simple`, `symbolic` and `sym`, and the law
    `Tolk_next.Symbolic › laws › sym keeps a float expression's value bit for
    bit at special values`;
  - Simplify's reductions: `Tolk_next.Simplify › keeping values (D24) › a
    float sum over an empty part of a range is +0., whatever the value` and
    `› a float product by a boolean mask keeps its value`, and
    `Tolk_next.Simplify › pm_reduce_collapse › a float product by a comparison
    cast from a boolean stays (D24)`;
  - transcendental polynomials: the `Tolk_next.Transcendental` goldens,
    generated from the patched tinygrad, whose value tables are unchanged;
  - tensor-core accumulators: the sources of every tensor core,
    `Tolk_next.Cstyle › sources › by default › metal_tc_*`, `hip_tc_*` and
    `metal_matmul`, whose loops add nothing to the accumulator, as tinygrad's;
    `Tolk_next.Codegen › tensor-core accumulators (D24) › the running sum
    replaces a zero accumulator` and `› the lowering keeps a tensor core's value, apart from a
    zero's sign`, which draws accumulators at `-0.`;
  - the goldens, generated from the patched tinygrad.

## D25. Compilers keep each product and sum its own rounding

- **tinygrad:** `runtime/support/compiler_cpu.py` (Clang's arguments),
  `runtime/support/compiler_amd.py` (HIP's options), `runtime/support/
  compiler_cuda.py` (NVRTC's options), `runtime/ops_metal.py:60` (Metal's
  parameters): none turns floating-point contraction off.
- **tolk.next:** `lib/runtime/support/compiler_cpu.ml` (`-ffp-contract=off`),
  `lib/runtime/support/compiler_amd.ml` (`-ffp-contract=off`),
  `lib/runtime/support/compiler_cuda.ml` (`--fmad=false`),
  `lib/runtime/support/compiler_metal.ml` (`#pragma METAL fp contract(off)`
  before the source); `lib/helpers.ml` (`Diskcache.version` 2, since a cached binary
  compiled with contraction answers the same key).
- **Differs:** each compiler would fuse a product and a sum that a rendered
  expression holds together, `a*b + c`, into one multiply-add with one
  rounding: Clang at `-O2` (`-ffp-contract=on`), HIP (`fast`), NVRTC
  (`--fmad=true`) and Metal, whose `-ffp-contract=off` does not reach the code
  where its pragma does. The graph states two roundings, and they stay two.
  The IR's own multiply-add (`Op.Mulacc`) comes only from `Decomp_op`'s
  `a * b + c` rule, for a renderer that renders it; no renderer of tolk.next
  does, so no kernel holds one.
- **Reason:** (b). RFC 0012's Law 1: every constructor's compiled result,
  alone or fused, meets its class against eager nx, which rounds a product and
  a sum apart; and rune's accurate compositions (two-part products and sums)
  are exact only where each operation rounds as written.
- **Pinned by:** `Tolk_next.Compiler_cpu › execution on the host › a product
  and a sum round twice, never fused` (`(1 + 2^-12)^2 - (1 + 2^-11)` is 0,
  where a fused multiply-add gives `2^-24`); `Tolk_next.Compiler_metal ›
  MTLCompiler › a product and a sum compile under the no-contraction pragma
  (D25)`. On an Apple GPU, the same kernel compiled by MTLCompiler gives
  `2^-24` without the pragma and 0 with it, and still `2^-24` with
  `-ffp-contract=off` alone. NVRTC and HIP are README's hardware checks.

## D26. A minus never meets a minus in C-style source

- **tinygrad:** `renderer/cstyle.py:140` (`Ops.NEG` is `-{x}`), `:144`
  (`Ops.SUB` is `({a}-{b})`).
- **tolk.next:** `lib/renderer/cstyle.ml` (`joined`, `infix`, `Neg`).
- **Differs:** a negation or a subtraction whose operand starts with a minus
  sign, itself a negation or a negative constant, is written with a space
  between the two signs: `(a- -b)`, `- -b`. tinygrad writes `(a--b)`, which C,
  Metal, CUDA and HIP read as the decrement operator and reject. Every other
  source is written as tinygrad writes it.
- **Reason:** (b). rune's lowering subtracts negated values (`acos`'s
  `1 - (-x)` before it was written `1 + x`), and the graph may reach the
  renderer so; a source the compiler rejects is a failed program.
- **Pinned by:** `Tolk_next.Cstyle › negation › a minus before a minus is
  apart › *` (every renderer), `› Clang compiles and runs a difference with a
  negated operand` and `› … with a negative constant`.

## D27. Constants keep a NaN's bits

- **tinygrad:** `dtype.py:79-82` (`DType.const` makes every NaN `math.nan`),
  `dtype.py:16-19` (`ConstFloat` compares every NaN equal to every other), and
  `uop/symbolic.py:23-26` (`fold_bitcast` converts the constant with
  `truncate` before reading its bits).
- **tolk.next:** `lib/dtype.ml:17` (`equal_const`), `:27` (`hash_const`) and
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
- **Pinned by:** `Tolk_next.Ops › identity › NaN constants of different bits
  are different nodes`; `Tolk_next.Ops › exec_alu › an invalid operation's
  NaN is the canonical positive quiet NaN, whatever the host gives (D27)`,
  which compares bits and fails on an x86 host without the canonical NaN; `Tolk_next.Symbolic › symbolic_simple › casts › a
  bitcast round trip of every 8- and 16-bit word folds to its value`, which evaluates each graph
  with the reference interpreter before and after `simplify`;
  `Tolk_next.Dtype › const › const keeps the bits of every 8- and 16-bit float
  word`.

## D28. A host program has nx.device's entry

- **tinygrad:** `codegen/__init__.py:449-451` (`do_compile`: the binary is the
  source compiled), `runtime/ops_cpu.py:58-70` (`CPUProgram.__call__` calls
  the kernel's own signature through ctypes, every argument a `c_uint64`).
- **tolk.next:** `lib/codegen/codegen.ml` (`host_entry`, `do_compile`).
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
- **tolk.next:** `lib/codegen/opt/postrange.ml` (`tc_operand`, `try_core`,
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
  `Rune_next.Lower_linalg › tensor cores › cuda › *`. The core's exact product
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
- **tolk.next:** `lib/runtime/support/hcq2.ml`: `range_placement`, which
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
- **tolk.next:** `lib/codegen/decomp/transcendental.ml:110`
  (`one_over_two_pi`), `:156` (`payne_hanek_reduction`) and `:315`
  (`sin_poly_large`); `test/gen/tinygrad.patch`, which gives tinygrad the same
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
  word; and the remainder is the signed fraction's two 64-bit halves, each
  converted and scaled. The remainder is within an ulp or two of the exact
  one in float32 and float64, `6381956970095103 * 2^797`, the float64 nearest
  a multiple of `pi/2`, included. `sin_poly_large` takes the sine of an odd
  quadrant as `sin (pi/2 - |r|)`, which stays within the polynomial's range
  for the nearest remainder.
- **Reason:** (b): rune's `sin` and `cos` reduce by `pi/2` with this
  reduction beyond `2^12` in float32 and `2^22` in float64, where their exact
  Cody-Waite parts stop (RFC 0012), and tolk's own `xsin` uses it beyond its
  switch-over; both must give nx's sine there.
- **Pinned by:** `Tolk_next.Transcendental › reductions ›
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
- **tolk.next:** `lib/runtime/ops_metal.ml` (`selectors`, and the stamps in
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
- **tolk.next:** `lib/schedule/multi.ml:75-87` (`pm_unselect_deviceless`,
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
- **Pinned by:** `Tolk_next.Multi › multi_pm › laws ›` "a shard of a product
  with zero cast to a float is zero (D35)", "a shard of a sum of a product with
  zero is zero (D35)", and the law "a rewritten value holds the value computed
  whole" over every drawn program.

## D36. A CUDA host program reads a function's address from a word

- **tinygrad:** `runtime/ops_cuda.py:23` (`extern`: a `Buffer` of the host
  placed at an address), `:38` (`CUDAQueue.extern`: the address of such a
  buffer), `:51` (a kernel's function) and `:64` (the host function that
  stamps a slot).
- **tolk.next:** `lib/runtime/ops_cuda.ml` (`extern`).
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
- **tolk.next:** `lib/runtime/ops_amd.ml:74` (`is_signal_word`), `:379`
  (the compute queue's `wait`), `:395` (its `signal_mem`), `:677` (the copy
  queue's `wait`) and `:709` (its `signal`).
- **Differs:** tinygrad's queues compare the low 32 bits of a 64-bit word,
  and write only them: the compute queue waits until they are at least the
  value's (`WAIT_REG_MEM`, `>=`) and signals with `RELEASE_MEM` of the low 32
  bits (`send_32_bit_low`); the copy queue polls for `>=` and fences the low 32
  bits. A device's values pass 2^32, and then the word's high half is never
  written, so a 64-bit reader never sees the value, and a wait for a value just
  past the wrap passes on a word just before it (`0xfffffffd >= 5`). In
  tolk.next a wait on a device's signal word, which is for the value its work
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
- **tolk.next:** `lib/runtime/ops_amd.ml:153` (`amd_build_program`),
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
- **tolk.next:** `lib/runtime/ops_amd.ml:599` (the AQL queue's `submit`) and
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
- **tolk.next:** `lib/runtime/ops_nv.ml` (`copy_signal`).
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
- **tolk.next:** `lib/schedule/memory.ml:18` (`viewed`), `:27` (`calls`) and
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
- **Pinned by:** the Engine suite (`test/engine/tolk_next_engine`): `link and
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
- **tolk.next:** `engine/tolk_next_engine.ml` (`placeholder`: a volatile
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
- **Pinned by:** the Engine suite (`test/engine/tolk_next_engine`): `batches ›
  a batch's kernel arguments are mapped memory, and its command buffers and
  volatile words pinned memory`; the Ops_amd and Ops_nv execution suites on a
  GPU.

## D43. A queue bufferizes its own commands, and a region keeps its alignment

- **tinygrad:** `runtime/support/hcq2.py:471` (`bufferize_cmdbuf` lays each
  nested `LINEAR` out on 128 bytes), `:479-481` (`encode_submit` bufferizes
  the queue's commands, then hands the buffer to `HWQueue.submit`).
- **tolk.next:** `lib/uop/ops.ml` (the `Region` argument of `Op.Linear`),
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
- **Pinned by:** `Tolk_next.Hcq2 › patch and bufferize_cmdbuf › each region
  starts at its own alignment in the buffer of its name` and `› a region
  addressed through another region is laid out once, in the buffer of its
  name`; the Hcq2, NULL, Metal, CUDA and AMD goldens, unchanged; the Ops_nv
  recorded cases, whose generator applies D43 to tinygrad.

## D44. An integer cast of a weak expression computes in integers

- **tinygrad:** `uop/weak.py:27-33` (`cast_weak_srcs`), `dtype.py:180-194`
  (`promo_lattice`, `least_upper_dtype`).
- **tolk.next:** `lib/uop/uop_weak.ml:55` (`cast_weak_srcs`).
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
- **tolk.next:** `lib/runtime/support/hcq2.ml:457` (`staging_size`), `:472`
  (the placeholder in `stage_copy`); `engine/tolk_next_engine.ml:360`
  (`staging`).
- **Differs:** a copy between memory the queues cannot reach goes through the
  two halves of a placeholder of the host tagged `"staging"`, of 128 MiB. The
  engine gives it, at link, the host's staging memory: one pinned buffer per
  host, which every linked schedule shares and which is kept for the life of
  the process, as tinygrad's is. Runs that stage through one host take turns
  with it, whatever their devices, since each touches it; one area per device
  would let them overlap for 128 MiB a device, and waits for a measured
  bottleneck.
- **Reason:** (c). The compiler opens no device and allocates nothing
  (plan §1a): storage it names is a placeholder, and the engine's link
  allocates it.
- **Pinned by:** the Hcq2 suite (`test/runtime/support/hcq2`): `compile_linear
  › copies through the halves of a staging buffer of the host where the
  queues cannot reach`, which finds one placeholder tagged `"staging"` of
  128 MiB on the host, and six copies; the Engine suite
  (`test/engine/tolk_next_engine`): `batches › linked schedules that stage
  share the host's staging memory`, `› staged runs of two programs on other
  devices take turns` and `› staged runs of two programs from two domains each
  copy their own`.

## D46. Whether a device's queues reach memory is described, not tried

- **tinygrad:** `runtime/support/hcq2.py:151-155` (`stage_copy` maps each
  buffer on the device with `get_buf` and stages the copy when that raises).
- **tolk.next:** `lib/runtime/support/hcq2.ml:467` (`reached`, from
  `queues.reaches`); `engine/cuda.ml` and `engine/amd.ml` (`reaches`).
- **Differs:** the caller's description of a device says which other devices'
  memory its queues address (`Hcq2.queues.reaches`), and a copy is staged when
  either side's memory is not reached. tinygrad tries to map the buffers and
  stages the copy on failure. A buffer that `reaches` admits but the device
  cannot map is staged by the engine instead, per batch (D70).
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
- **tolk.next:** `lib/runtime/ops_amd.ml:162` (`dispatch_packet`).
- **Differs:** tinygrad gives a known grid size as a Python integer, which the
  AQL queue turns into a 32-bit constant but the kernel arguments of the PM4
  queue put as they are among the sources of a `LINEAR`: encoding a kernel
  that reads its dispatch packet on a PM4 queue raises. tolk.next's grid sizes
  are 32-bit constants in both.
- **Reason:** (a). A node's sources are nodes: OCaml's types admit no integer
  among them.
- **Pinned by:** the Ops_amd suite: `recorded cases › scratch`, a kernel that
  reads its dispatch packet and scratch memory, on a PM4 queue.

## D50. Every C-style renderer writes a division

- **tinygrad:** `renderer/cstyle.py:139-147` (`CStyleLanguage.code_for_op`
  has no `FDIV`) and `:277-280` (Clang's adds it); `codegen/decomp/op.py:122-125`
  (a target that lists `FDIV` gets its reciprocals as divisions).
- **tolk.next:** `lib/renderer/cstyle.ml:355-359` (the `FDIV` rule of
  `base_rewrite`).
- **Differs:** Metal, CUDA and HIP write an `FDIV` as `(a/b)`, as Clang does,
  where tinygrad's renderers fail on it. Their tables still leave `FDIV` out,
  so code generation keeps their reciprocals, and every kernel built from
  tinygrad's operations keeps tinygrad's source.
- **Reason:** (b). rune lowers nx's float division, and the power and the
  arc tangent built on it, to `FDIV` (RFC 0012): eager nx divides as IEEE
  does, rounding once, which a product by the reciprocal does not.
- **Pinned by:** the `Cstyle` suite (`test/renderer/cstyle`): `sources › by
  default › <target>_fdiv_<type>`, tinygrad's source once its renderer lists
  `FDIV` as Clang does, for Clang, Metal, CUDA and HIP in each float type
  they have; `division (D50) › the operands tell a quotient from a product by
  the reciprocal` and the slow `› Metal divides as IEEE does, rounding once`.
  CUDA's and HIP's `/` are correctly rounded by their compilers' defaults,
  which is on the hardware checks of `test/README.md`.

## D51. A launch reads its local memory size from a word of the device

- **tinygrad:** `runtime/ops_nv.py:283-291` (`NVProgramData` writes the
  device's `slm_per_thread` into the launch template's
  `shader_local_memory_high_size`), `:688-697`
  (`NVDevice._ensure_has_local_memory` grows the device's local memory when
  it builds a program).
- **tolk.next:** `lib/runtime/ops_nv.ml:330` (`local_word`), `:352`
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
- **tolk.next:** `lib/uop/symbolic.ml:1222` (`pm_move_where_on_load`).
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
- **tolk.next:** `lib/uop/ops.ml:241` (`phase` and `align`), `:3244`
  (`param_arg`, which checks them) and `:3489` (`storage_phase`, which
  `param_like` gives a parameter); `lib/schedule/rangeify.ml:502` (`debuf`,
  which gives them a kernel's parameter); `lib/codegen/late/coalesce.ml:130`
  (the merge).
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
  a disk, which no vector access reads, keeps the default. The congruence is part of the graph, so of a program's cache
  key.
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
  x86, where Clang emits `movaps`, on the trips that moved it by 4 bytes.
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
  a view 8 bytes into a disk file whose parameter has no phase. In rune.next,
  the `Lower` suite's `a parameter of storage 4 bytes past a 16-byte boundary
  has phase 4`, `a capture of storage 4 bytes past a 16-byte boundary has
  phase 4` and `a program over an argument 4 bytes past a 16-byte boundary
  computes nx's values`, and the `Compiled` suite's `edges › an operand whose
  buffer starts 2 bytes into its memory is read where it is`, on the
  host and on Metal, whose sweeps draw buffers that start at any byte; the
  slow `Ops_metal (execution)` suite's `phase (D54) › a float16 buffer 2 or 6
  bytes into its memory is read where it lies with its phase`.

## D55. Metal names a vector after its element's one-word name

- **tinygrad:** `renderer/cstyle.py:186` (`_render_dtype` names a vector
  after its element's name with spaces made underscores) and `:362`
  (`MetalRenderer.type_map` renames only `uint` and `bfloat`).
- **tolk.next:** `lib/renderer/cstyle.ml:87,119` (`vector_names`) and
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
- **tolk.next:** `lib/codegen/decomp/transcendental.ml:433-439` (`xlog2`),
  and `test/gen/tinygrad.patch`, which reorders tinygrad's selects the same
  way.
- **Differs:** the reciprocal of a negative number of magnitude below
  `2^-128` in float32 (`2^-1024` in float64, about `2^-16` in float16)
  overflows to `-inf` too, so
  tinygrad's last select turned the NaN of such a number's logarithm into
  `-inf`. The `-inf` select now comes first and the NaN select after it:
  `-0.0` still gives `-inf`, and every negative number gives NaN.
- **Reason:** (b). rune's lowering computes nx's logarithms with `xlog2`,
  and nx gives NaN for the logarithm of every negative number, subnormals
  included.
- **Pinned by:** the `Transcendental` suite
  (`test/codegen/decomp/transcendental`): `special values › xlog2 of a
  negative number whose reciprocal overflows is NaN, and of -0. is -inf
  (D56)`, and the negative float16 rows of `values.golden` of magnitude below
  `2^-16`, from the patched tinygrad.

## D58. A program applies no elementwise operation to a vector

- **tinygrad:** `uop/spec.py:205` (`spec_program`, which accepts an
  elementwise operation of any shape).
- **tolk.next:** `lib/uop/spec.ml:390` (the first rule of `program`).
- **Differs:** `Spec.program` rejects an operation of `Op.Set.elementwise`
  on values, casts and bitcasts included, whose shape has an axis; a bitcast
  of memory, which views it, stays allowed. `SPEC` defaults to 1
  and code generation checks every lowered kernel against `Spec.program`, so
  a kernel that still holds vector arithmetic fails at lowering, naming the
  operation, on every target. Devectorize leaves none while every source
  keeps its width. A fold there can drop one: when every lane of a gated
  load's index is `Invalid`, the load folds to a scalar `0`, and the select
  around it stays a vector select, as in tinygrad, which the weak lowering
  makes a vector cast of a stack of constants, and the rule refuses.
  rune builds no such kernel, since it lowers a fold or an unfold whose
  windows along an axis read only padding to zeros.
- **Reason:** (b). CUDA's vectors are structs without arithmetic, casts or
  selects, so a vector operation left after devectorize is a kernel that does
  not compile there, and one Metal renders without complaint; rune compiles
  its kernels for both.
- **Pinned by:** the `Spec` suite (`test/uop/spec`): `vectors in programs
  (D58) › a program has no elementwise operation on a vector` (add, cast and
  where on two lanes, and the same on one); the `Codegen` suite's `vectors in
  programs (D58) › a cast left on two lanes after devectorize is refused`,
  `› a weak constant stored into four lanes of half is refused` and, on every
  case, `› no program applies an elementwise operation to a vector`.

## D60. A loop of a call stays in the schedule

- **tinygrad:** `schedule/rangeify.py:165-168` (`pm_no_views`, which strips
  every view of storage), `:335-337` (`split_store`, which makes a kernel of
  an end of a call), `schedule/__init__.py:72-75` (`create_schedule`, which
  schedules the call of an end and drops the end), and `uop/spec.py:254-280`
  (`spec_kernel_graph`, which admits device ranges only, and no end).
- **tolk.next:** `lib/schedule/rangeify.ml:236` (`loop_range`), `:241`
  (`pm_no_views`, which keeps a view that moves with a loop's range and
  untags the range), `:588` (`split_store`), `lib/schedule/schedule.ml:141`
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
  keeps its own end`; and the `Tolk_next_engine` suite
  (`test/engine/tolk_next_engine`): `link and run › a scan runs its body
  once per trip, carrying in place (D60)`, against the unrolled loop on the
  host. The `Spec` suite (`test/uop/spec`): `kernel_graph › loops of calls
  (D60) › accepts a loop of a call over a loop range`, `› refuses a loop of
  a call over a range of any other kind`, `› accepts an open device range`
  and `› refuses a weak sum of a weak integer variable`; its
  `verdicts.golden` records the patched spec on ends and shrinks.

## D61. A stack takes the per-shard sub-view of a whole source

- **tinygrad:** `schedule/multi.py:188-200` (`stack_multi`, which stacks a
  source that is not sharded as it is beside the shards of the others).
- **tolk.next:** `lib/schedule/multi.ml:474` (`stack_multi`), and
  `test/gen/tinygrad.patch`, which gives tinygrad the same rule.
- **Differs:** in a stack whose sharded sources are sharded alike, a whole
  source, one value of the full shape on every device, takes its per-shard
  sub-view (`shard_subview`), as an elementwise operation's does
  (`alu_multi`). tinygrad stacks it whole beside the shards: the stack's
  shards then mix a shard with a whole value, and
  `Tensor.stack(a.shard(devices, axis=0), b.shard(devices))` gives wrong
  values, or fails at a later operation on the shard sub-view's shape check
  when the whole source comes first.
- **Reason:** (b). rune.next's compiled `Nx.stack` of a value sharded on an
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
- **tolk.next:** `lib/codegen/decomp/decomp_dtype.ml:677` (`moved`) and
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
  `where` keeps a signalling NaN, and rune.next compiles the 8-bit floats on
  the host and on Metal, where tolk emulates them, so a compiled function
  computes what it computes eagerly only if the emulated moves keep the bits
  too.
- **Pinned by:** rune.next's `Compiled` suite: `host › 8-bit floats › a copy
  of every 8-bit float code keeps its bits (D62)` and `› a selection of
  every 8-bit float code keeps its bits (D62)`, and the same under `metal ›`
  (slow), against eager on all 256 codes of e4m3 and e5m2; the
  `Decomp_dtype` suite (`test/codegen/decomp/decomp_dtype`): `NaNs › an
  emulated copy keeps every NaN code, a signalling one's included (D62)`
  and `› an emulated selection keeps every NaN code, a signalling one's
  included (D62)`, on every emulated float, and the goldens of the
  `where`, `flip`, `gather` and `pad` kernels, from the patched tinygrad.

## D64. A cast to a narrow float through a float32 rounds once

- **tinygrad:** `renderer/cstyle.py:89-90` (`create_non_native_float_pats`,
  which casts a source of any type but float32 to a float32, then to the
  narrow float).
- **tolk.next:** `lib/renderer/cstyle.ml:430` (the rule's cast), with
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
- **Reason:** (b). nx casts with one rounding, and rune.next compiles its
  casts to bfloat16 for the host: an int64 arange of bfloat16 from 2^40
  computes a cast of each integer.
- **Pinned by:** the `Codegen` suite (`test/codegen/codegen`): `casts to
  bfloat16 (D64) › an integer or a double rounds once on the host`, for
  int64, int32, uint32, uint64 and float64 sources; rune.next's `Jit` suite:
  `values › a bfloat16 arange from 2^40 inside a compiled call equals
  eager's`.

## D65. An emulated value is a value of its float

- **tinygrad:** `codegen/decomp/dtype.py:198-201` (`pm_float_decomp`: a cast
  to the emulated float only clamps its operand, `f2f_clamp`, and an
  operation on it computes in the emulating float; only a store rounds).
- **tolk.next:** `lib/codegen/decomp/decomp_dtype.ml:709` (`rounded`), `:791`
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
  rune.next compiles chains of operations on the 8-bit floats into one
  kernel on the host and on Metal, which emulate them: `(y + y) * y` of an
  e5m2 `y` computed eagerly differs from the kernel that rounds only once.
- **Pinned by:** the `Decomp_dtype` suite
  (`test/codegen/decomp/decomp_dtype`): `rounding in the kernel (D65) › a
  cast to an emulated float and back is the value rounded once` and `› an
  operation on an emulated float rounds its result to it`, on every emulated
  float, and the goldens of every kernel that casts to or computes in an
  emulated float, from the patched tinygrad; rune.next's `Jit` suite: `one
  device › a chain of 8-bit float operations rounds after each, as eager
  does`, and the same on Metal.

## D67. A long range runs as chunks of its trips

- **tinygrad:** `runtime/support/hcq2.py:379-385` (`HWQueue.loop`, which
  repeats a trip's command bytes and words once per trip:
  `self.blob += self.blob[start:] * int(r.vmax)`).
- **tolk.next:** `lib/runtime/support/hcq2.ml:1215` (`chunk_calls`), `:1219`
  (`parts`), `:1230` (`chunked`) and `:1184` (`trips_from`), used by
  `sched_batches`; `engine/tolk_next_engine.ml:906` (`run_call`'s `Range`, which
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
  AQL …` (a chunk over half its ring halves, the trips left fit); rune.next's
  Jit suite on Metal, `staged scans › stage a thousand steps` and `› stage
  more steps than a batch holds, in chunks and the rest` (3,001 steps), each
  against the eager scan.

## D68. Withdrawn

The scheduler recognised a one-hot sum, rune's lowering of a gather, and
inlined it as a load where it was read. Inlining values that read state
written elsewhere in the graph left buffers with two definitions, and it
was withdrawn; an explicit gather, which rune emits as an indexed load and
tolk lowers as one, replaces it.

## D69. A store through a padded view writes only within the pad's source

- **tinygrad:** `schedule/indexing.py:101-105`
  (`convert_pad_to_where_to_keep_behavior_local`), which turns every pad it
  ranges into a selection, a store's destination included: the store then
  targets a `WHERE`, which `uop/spec.py` refuses ("UOp verification failed …
  Ops.STORE … Ops.WHERE").
- **tolk.next:** `lib/schedule/indexing.ml:86` (`own_destination`, run
  first by `run_rangeify`), `:59` (`mark_stored_pads`) and `:209`
  (`convert_pad_to_where_to_keep_behavior_local`).
- **Differs:** a store whose destination moves through a pad writes the
  elements whose index falls within the pad's source and drops the ones in
  the padding. The store's destination is first made its own, its movements
  tagged, so that no read shares them; on that path the pad's validity goes on
  the index into the storage, an `INDEX` whose index carries it, which codegen
  renders as a guarded store, and the pad is removed as any movement is. A
  read through a pad, the same pad as the store's included, stays a
  selection, as in tinygrad: a fill other than zero, a selection off the pad,
  reads its fill. tinygrad refuses every graph this changes, so every graph
  it accepts schedules as before. It is the write-side dual of a read through
  a pad, which is a gated load.
- **Reason:** (b). Rune.next's compiled call stores a lent indexed write row
  by row (`Lower_index.scatter_rows`), and a row whose index lies outside its
  target is dropped, as nx's scatter drops it. Through a pad, the dropped row
  stores into the padding and reads nothing, so each row's store fuses with
  the kernel that computes the row: a decode step's cache write with its
  projection. As a selection of the target's own row, the store reads its
  destination, and the kernel computing the row is stored apart.
- **Pinned by:** the engine suite (`test/engine/tolk_next_engine`): `a store
  through a padded view (D69) › writes the row within the source` (rows 0, 3
  and 7) and `› writes nothing outside the source` (-1, 8 and 9), each one
  kernel, and `› a read of the same padded node reads its fill in the
  padding` (one pad node stored through and read with a fill of 7, beside a
  read of the storage without the pad), and `› a destination that already
  carries a tag keeps the read's fill`, on the host; `Metal › a store through
  a padded view writes the row within the source, and nothing outside it
  (D69)` and `› a read of a padded node stored through reads its fill in the
  padding (D69)` (slow); rune.next's Jit suite, `a lent write of rows › *`.

## D70. A batch stages host memory its device cannot map

- **tinygrad:** `device.py:303-308` (`HostAllocator._alloc`: every host buffer
  is its own `mmap`, so it starts on a page and every device maps it), and
  `runtime/support/hcq2.py:151-155` (`stage_copy`, which maps each buffer).
- **tolk.next:** `engine/tolk_next_engine.ml:321` (`stage_on`), `:354`
  (`address`, which stages on a refused borrow), `:572` (link's stages)
  and `:669` (`run_batch`: an input's stage, the copies in after the
  waits and out after completion); `lib/runtime/support/hcq2.ml:219`
  (`call_writes`, the batch's `writes`).
- **Differs:** a host buffer of less than 64 KiB does not start on a page in
  nx.device, so a CUDA, AMD or NV device, which maps whole pages, cannot
  borrow it, and memory `of_bigarray` wraps may not start on one either.
  Where a batch's device addresses such memory of this machine's host, the
  engine stages it: the device addresses pinned memory of its own of the same
  size, one per memory for the life of the linked batch (per input, made again
  when the input's size changes), which each run fills from the host memory
  inside its submission, after its waits. The batch's queue data carries
  `writes`, which tinygrad's `HCQInfo` has not: the storage under each output
  of each of its calls, a copy's destination and a kernel's outputs, in-place
  ones included, and under every argument of a call whose outputs are not
  known. A run copies back the stages of that storage once its devices
  synchronized, and so completes before `run` returns; a run whose stages are
  only read does not wait, and the next run's waits for it come before its
  copies in. A placeholder's storage is never staged.
- **Reason:** (c). nx.device starts a host buffer on a page from 64 KiB, so
  that small tensors do not take a page each; its contract stages copies of
  smaller ones (`Nx_device.Buffer.borrow`), and the engine honours it for the
  addresses a batch's commands hold. tinygrad's `written_bufs` leaves out the
  storage a call also reads and parameters, so it cannot say which stages a
  run writes.
- **Pinned by:** the Engine suite (`test/engine/tolk_next_engine`): `batches ›
  a host buffer the device cannot borrow is staged, as storage` and `› as an
  input`, of 12, 256 and 1,280 bytes, on CPU:4, a test device that maps whole
  pages: each copies the host buffer in and a device buffer back over it, twice,
  the host rewriting it between runs; and `› a run waits for its batch only
  when it writes a staged buffer`, whose read returns while CPU:4's late queue
  has not signaled.

## D71. A launch states a constant bank's size in 16-byte units

- **tinygrad:** `runtime/ops_nv.py:302` (`NVProgramData` writes a constant
  bank's bytes into the launch template's `constant_buffer_size_shifted4`).
- **tolk.next:** `lib/runtime/ops_nv.ml:301` (`program_data`).
- **Differs:** a launch template states each constant bank's size rounded up
  to 16 bytes and shifted right by 4, as the field's name says, where
  tinygrad writes the bytes. The field holds 13 bits, so tinygrad refuses a
  bank of 8 KiB or more, a parameter bank of about a thousand arguments, and
  states a smaller bank 16 times its size.
- **Reason:** (b). tinygrad writes a byte count into a field that counts
  16-byte units. rune.next's staged scan `write out four hundred steps, each
  carry stored` launches a kernel whose parameter bank holds 13,192 bytes,
  which NV refused (`constant_buffer_size_shifted4_0=0x3388 does not fit`)
  and CUDA runs.
- **Pinned by:** the Ops_nv suite: every `recorded cases` golden, from
  tinygrad with D71 applied by its generator; rune.next's Jit suite on an
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
- **tolk.next:** `lib/schedule/prepare.ml:668` (`contiguous_view`).
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

## D74. Float arithmetic has bounds, and a bounded sine takes the short reduction

- **tinygrad:** `uop/ops.py:1104-1163` (`UOp._min_max`), which bounds binary
  operations on integers only (`:1105`), has no case for `TRUNC` or `NEG`, and
  bounds a `WHERE` by both its branches whatever selects them (`:1142`);
  `codegen/decomp/transcendental.py:170-191` (`xsin`), which builds the
  Payne-Hanek reduction unless its caller passes `fast`, and chooses between
  the reductions with `x_abs < switch_over` (`:187`), a comparison that the
  decompositions' rewrite (`symbolic_simple`) does not fold.
- **tolk.next:** `lib/uop/ops.ml:1507` (`compute_min_max`), `:1524` (`Where`),
  `:1556` (`Trunc`), `:1560` (`Neg`), `:1573` (`selected`) and `:1586`
  (`float_bounds`);
  `lib/codegen/decomp/transcendental.ml:322` (`xsin`); `test/gen/tinygrad.patch`,
  which gives tinygrad the same bounds and the same `xsin` before the goldens
  are generated.
- **Differs:** a float sum, difference or product of operands with finite
  bounds has the bounds of its corners, computed in double and widened by a
  relative `2^-m` and the smallest normal of its type, `m` its mantissa's bits,
  so that they hold the result rounded at its type, or flushed to zero. A
  result that may overflow its type, and one of a weak float, has the type's
  bounds. A truncation's bounds are its operand's, truncated, and a float
  negation's are its operand's, negated and swapped. A float selection by a
  comparison `a < b` narrows the branch it selects where the comparison
  holds, so where neither operand is NaN: `a` to below `b`'s greatest value,
  and `b` to above `a`'s least. `xsin` of an angle whose
  bounds lie strictly within `switch_over` is its `fast` form, the Cody-Waite
  reduction alone.
- **Reason:** (b): the sine or cosine of a bounded value on the CPU, such as
  sofo's and symo's Gaussian draws (`Nx.Rng.normal`), whose angle is `2 pi u`
  for a uniform `u` in `[0, 1)` fused into the draw. rune's `sin` and `cos`
  (`packages/rune/next/lib/lower_arith.ml`, `by_quadrant`) skip their own long
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
  `› a float operation of an unbounded operand, or that can overflow, has its
  type's bounds (D74)`, `› a float selection by a comparison narrows what it
  selects (D74)`, `› a float truncation and negation map their operand's
  bounds (D74)` and the float rows of `binary_bounds.golden`, from the
  equally patched tinygrad; the Transcendental suite: `graphs › a sine of an
  angle bounded below the switch-over is its fast form (D74)`; and
  rune.next's `lower_arith` suite: `long reductions › a normal draw takes
  none`, `› the sine of an angle read from a buffer takes its own only` and
  `› the cosine of an angle read from a buffer takes its own only`.
