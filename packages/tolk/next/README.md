# tolk.next

tolk.next is tolk rebuilt on tinygrad at commit
`79af1ca70e7021f504919c4ff5631245acc33ed6`. It turns tensor graphs into
compiled programs, as data: source, binaries, memory plans and the queue
programs that launch them. It is a compiler and depends on no runtime; rune
runs its output on `nx.device`, which owns every driver. It is internal to
raven until it replaces the `tolk` library.

## Layout

The library `tolk.next` (module `Tolk_next`) has one module per tinygrad file.
Each file sits at its tinygrad path, and `lib/dune` uses
`(include_subdirs unqualified)`, so module names are flat: `uop/ops.py` is
`lib/uop/ops.ml`, the module `Tolk_next.Ops`. Whoever knows a file in one tree
finds it in the other by its path.

The namespace is flat because dependencies must be per file, as Python's
imports are. With directories as modules (`qualified`), naming `Codegen.X`
depends on every module under `codegen/`, and tinygrad's directories import
each other both ways: `uop/symbolic.py` uses
`codegen/decomp/transcendental.py`, which uses `uop/ops.py`, and dune reports
a dependency cycle between them although the files form none.

A module is named after its file, with two rules:

- **A package `__init__.py`** lives in the file named after its directory:
  `codegen/__init__.py` is `lib/codegen/codegen.ml`, `Codegen`. The exception
  is `uop/__init__.py`, which is `lib/uop/op.ml`, `Op`, since `Ops` is
  `uop/ops.py`; `Op.t` is one operation.
- **A file below the top whose name is taken**, by another in-scope file or
  by a Stdlib module, takes its directory's name as a prefix:
  `codegen/decomp/dtype.py` is `lib/codegen/decomp/decomp_dtype.ml`,
  `Decomp_dtype`, since `dtype.py` is `Dtype`. Without the prefix a module
  would hide the other one from the whole library. Of two such files, the one
  tolk.next keeps only part of takes the prefix.

The modules not named after their file:

| tinygrad | tolk.next file | module |
|---|---|---|
| `uop/__init__.py` (the `Ops` enum, `GroupOp`) | `lib/uop/op.ml` | `Op` |
| `uop/weak.py` | `lib/uop/uop_weak.ml` | `Uop_weak` (Stdlib has `Weak`) |
| `codegen/__init__.py` | `lib/codegen/codegen.ml` | `Codegen` |
| `codegen/decomp/dtype.py` | `lib/codegen/decomp/decomp_dtype.ml` | `Decomp_dtype` (`dtype.py`) |
| `codegen/decomp/op.py` | `lib/codegen/decomp/decomp_op.ml` | `Decomp_op` (`uop/__init__.py`'s `Op`) |
| `codegen/opt/__init__.py` | `lib/codegen/opt/opt.ml` | `Opt` |
| `renderer/__init__.py` | `lib/renderer/renderer.ml` | `Renderer` |
| `schedule/__init__.py` | `lib/schedule/schedule.ml` | `Schedule` |
| `runtime/support/memory.py` (its `TLSFAllocator`) | `lib/runtime/support/support_memory.ml` | `Support_memory` (`schedule/memory.py`) |

The mixins have no files: the methods `UOp` keeps from `mixin/*.py` are
folded into `Ops`. No other in-scope file shares its name with
another or with a Stdlib module. An empty `__init__.py` has no module.

## Exclusions

What tolk.next does not port from tinygrad, and why. Keeping part of a file
is scope, not a divergence: the part left out is listed here, and
`DIVERGENCES.md` lists only what tolk.next does differently.

| Not ported | Reason |
|---|---|
| `tensor.py`, `function.py`, `nn/*`, and the `Tensor` surface of `mixin/*`: dtype shorthands, creation, reductions, randomness, the composite ops of `mixin/op.py` | nx and kaun are raven's frontend, and tinygrad's decompositions there are rune's lowering. The mixin methods `UOp` itself uses stay in the IR. |
| `TinyJit`, `_TinyJit` and `_prepare_jit_inputs` in `engine/jit.py` | the `Tensor` surface of the jit; rune walks the parameters with `Ptree`. |
| `CapturedJit`, the device registry, `BufferSpec` and the lazy `Buffer` of `device.py`, and running a schedule | execution is rune's; tolk.next returns what to run (see D3). |
| `device.py` but `TinyELF` and `Compiled`'s choice of renderer: `ALL_DEVICES`, `HCQ_RUNTIME_DEV`, `canonicalize_device`, `MultiBuffer`, `BufferStorage`, the allocators, `Program`, the rest of `Compiled` (its timeline, runtime buffers, signal waits, synchronization, interfaces and `pm_bufferize`) and `enumerate_devices_str`; the renderers that `Compiled` lists but tolk.next does not port (LLVM, LVP, X86, PTX, NVCC, NAK, HIPCC) | the runtime is rune's and nx.device's (see D3), and those renderers are excluded above. |
| Pickling a captured jit (`CapturedJit.__reduce__`) | raven has no persistent jit cache. |
| `mixin/gradient.py` and the `compute_gradient` path | rune owns differentiation. |
| `llm/*` | models are examples or a package of their own, never part of the compiler. |
| `viz/*` and the viz hooks in `helpers.py` | a Python web UI. |
| `tqdm`, `fetch` and `fetch_fw` in `helpers.py` | progress bars and downloads belong to the programs and packages that need them. |
| The profile events of `helpers.py` and `device.py` | nx.device's `Profile` records them. |
| The runtime: drivers, allocators, `Program`, memory, ELF loading, the driver half of each `runtime/ops_*.py` | nx.device owns it, and rune drives it (see D3). |
| `runtime/support/memory.py` but its `TLSFAllocator`: `MMIOInterface`, `BumpAllocator`, `AddrSpace`, `VirtMapping`, `PageTableTraverseContext`, `MemoryManager` | they map and allocate device memory, which is nx.device's (see D3). The TLSF stays, as `Support_memory.Tlsf_allocator`, since the memory planner places buffers with it. |
| `renderer/{ptx,llvmir,nir,wgsl}.py` | no raven target renders with them by default. |
| `pm_validate_wmma_rdna3`, `pm_validate_wmma_rdna4` and `pm_validate_wmma_cdna` in `renderer/tc.py` | only `renderer/llvmir.py`, excluded above, applies them. |
| A node as the shift count of `shl` and `shr` in `codegen/decomp/transcendental.py` | every caller shifts by a number. |
| `renderer/isa/*`, `renderer/amd/*`, `codegen/late/regalloc.py`, the ISA branches of `codegen/__init__.py`, and `Renderer.asm` in `renderer/__init__.py` | they serve hand-written instruction kernels and x86 host code; host programs are compiled with Clang. |
| The OpenCL, Intel, QCOM and WGSL languages of `renderer/cstyle.py`; the NVCC, HIPCC, PTX and X86 compilers; `compiler_{llvm,mesa,qcom}.py`; the refusal of `schedule/memory.py`'s `_can_plan` to plan buffers on `CL` and `WEBGPU` devices, which cannot view a buffer; and the refusal of `UOp.contiguous_view` (`uop/ops.py:943`) to find a view on them | not raven targets, or they need a full toolchain. |
| The `output` argument of `create_allreduce_function` in `schedule/allreduce.py` | no caller passes it: the function always allocates its output. |
| `runtime/support/compileserver.py`, with `Compiler.server` and `Compiler.compile_server` in `device.py`, which start it and talk to it | compilation workers are domains (see D5). |
| `runtime/support/c.py` but `DLL.findlib`, and in `findlib` the macOS shortcut for `libc` and `m` (`:94`), since no caller loads either: the ctypes structures, pointers and bindings, and `runtime/autogen/*` | raven has no ctypes: the compilers' C stubs declare the few NVRTC and comgr functions they call, and load the library that `C.findlib` finds. The second name under which tinygrad loads comgr 3 (`comgr_3`, with its override `COMGR_3_PATH`) is scope: both names search the same paths, and `lib<p>.so[.0-9]*` finds ROCm 6's `libamd_comgr.so.2` and ROCm 7's `libamd_comgr.so.3` alike (ROCm 7's name is from its release layout, unverified on an install), so tolk.next loads one library, found by `findlib`, and picks the constants of its version. |
| `runtime/ops_metal.py` but `MetalCompiler`: the device, allocator, programs and queues; and in `MetalCompiler`, the import of the LLVM library before MTLCompiler (`:34`), which only keeps tinygrad's own LLVM from sharing MTLCompiler's symbols, `__reduce__`, and `disassemble`, which runs a checkout of the applegpu disassembler under tinygrad's `extra/` | the runtime half is rune's and nx.device's (see D3); tolk.next loads no LLVM, has no pickling, and has no `extra/`. One code generation service serves the process, where tinygrad makes one per compiler. |
| `pretty_ptx` in `runtime/support/compiler_cuda.py` | it colours PTX for the `DEBUG>=5` print of `runtime/ops_cuda.py`, which loads programs and is rune's (see D3). |
| `jitlink_check` and `osx_docker_cmd` in `runtime/support/compiler_cuda.py`, with the macOS branches of `NVRTCCompiler` | they serve the PTX compiler and the compile server, both excluded above. |
| The `cachekey` argument of `ClangCompiler` in `runtime/support/compiler_cpu.py` | no caller passes it. |
| `ImageDType` and image paths | only OpenCL and QCOM use them. |
| The `IMAGE` branch of `gated_given_valid` in `uop/symbolic.py` (`:431`) | an image path, excluded above. |
| `_drop_valid_stmts`, `simplify_valid_image_load`, `image_valid_dims`, `transform_to_image`, `store_image` and `pm_simplify_add_image` in `codegen/late/coalesce.py`, and the image lengths of its `memory_coalescing` (`:141`) | image paths, excluded above. |
| The DSP renderer of `runtime/ops_dsp.py`, the DSP lengths of `memory_coalescing` in `codegen/late/coalesce.py` (`:138`), the DSP upcast limit of `Scheduler.apply_opt` in `codegen/opt/postrange.py` (`:122`) and the DSP upcasts of `hand_coded_optimizations` in `codegen/opt/heuristic.py` (`:113-118`) | Qualcomm's Hexagon DSP is not a raven target. |
| The QCOM grouping limit and workgroups of `hand_coded_optimizations` in `codegen/opt/heuristic.py` (`:82`, `:164-176`) | QCOM's OpenCL renderer is excluded above. |
| The `IMAGE` branches of `hand_coded_optimizations` in `codegen/opt/heuristic.py` (`:49-59`, `:103-107`) | image paths, excluded above. |
| `args_from_ast` in `codegen/opt/postrange.py` (`:256-258`), which allocates device buffers for beam search | allocating buffers to measure a kernel is execution, which is rune's (see D3); a search takes its measurement as a function. |
| The numpy and torch interop of `dtype.py` (`_to_np_dtype`, `_from_np_dtype`, `_to_torch_dtype`, `_from_torch_dtype`) | raven's arrays are nx's, and rune maps nx's types onto `Dtype`. |
| `dtypes.int8s`, `int16s`, `int32s` and `int64s` in `dtype.py` | only the x86 ISA renderer, excluded above, reads them. |
| `uop/validate.py`, whole: every function in it builds z3 terms, and its one entry point, `validate_index_with_z3`, is the optional z3 check of `uop/spec.py` | raven has no SMT solver among its dependencies. With `CHECK_OOB`, an access whose index bounds do not prove it in range fails the check, and says that the bound could not be proven without a solver. |
| `pyrender`, `pm_pyrender`, `pm_pyrender_extra`, `sugar`, `srcs`, `_render_with_splits` and `renderer_infer` in `uop/render.py`, with `UOp.pyrender` and the `pm` argument of `UOp.render`; `test_pyrender`, `eval_pyrender` and `pyrender_globals` in `uop/spec.py`, `uop/upat.py` and the setting `UPAT_COMPILE` | they generate Python source: `renderer_infer` writes the expressions that `UOp._sym_fxn` executes, and is the only other matcher `UOp.render` takes. Their other readers are diagnostics: the `DEBUG>=5` prints of `codegen/__init__.py:277` and `codegen/opt/search.py:121`, the `SPEC>1` round trip of `uop/spec.py:37,299` and `uop/ops.py:209`, and the leaking-ranges message of `uop/ops.py:1265`. tolk.next matches patterns as tinygrad does when `UPAT_COMPILE` is 0. |
| The code generation of `UOp._sym_fxn` in `uop/ops.py` | it generates Python source; `sym_infer` evaluates the expression directly, with the same arithmetic. |
| Match tracking in `uop/ops.py`: `TRACK_MATCH_STATS`, `PRINT_MATCH_STATS`, `match_stats`, `TrackedPatternMatcher`, `rewrite_group`, `TrackedGraphRewrite`, `RewriteTrace`, `UOp.trace_num`, `uop_fields`, `launch_viz`, process-replay capture, `get_location`, `UPat.location`, and the `name` of `graph_rewrite` and `substitute` | they serve viz and profiling; without them `rewrite_group` only calls its function, so passes are called directly. |
| Pickling in `uop/ops.py` and `renderer/__init__.py`: `Renderer.__reduce__`, `UOp.__reduce__`, `CallInfo.__reduce__`, `UPat.__reduce__`, `PatternMatcher.__reduce__`, `deconstruct_function`, `TEST_PICKLE` | raven has no persistent jit cache. |
| `CallInfo.grad_fxn`, `CallInfo.precompile_backward`, and the `grad_fxn` of `UOp.call` and `call_with_outputs` | rune owns differentiation. |
| `UOp.metadata` and `all_metadata` | they carry `Metadata`, excluded above. |
| `UOp.__getitem__` | Python's subscript syntax; its buffer path is `shrink`, `permute` and `index`, which callers compose. |
| `consumer_map_from_toposort` and `UOp.on_creation_device` | their readers, `pyrender` and the `NPY` and `PYTHON` devices, are not ported. |
| Nodes as tags (`uop/ops.py:546`) | only `function.py`'s callify, excluded above, tags nodes with nodes; tags are literals. |
| `Ops.PYLITERAL` and `Ops.REWRITE_ERROR` in `uop/__init__.py`, with the code that handles them (`uop/ops.py:129,332,377`, `uop/spec.py:116,239`) | produced only by the pattern compiler's code generation and viz, both excluded. `Tolk_next.Op › Op › has no counterpart for exactly REWRITE_ERROR and PYLITERAL` pins it. |
| RDMA splitting and encoding in `runtime/support/hcq2.py` (`:136-147,507`) | multi-node placement is not designed yet. |
| MOCK interfaces and the PYTHON device | raven has no mock drivers; `runtime/ops_python.py` survives only as the tests' reference interpreter. |
| SQTT, PMC and PMA profiling; `NVEncDecQueue`; USB | no consumer. |
| The parts of `helpers.py` that serve Python: `argfix`, `get_shape`, `fully_flatten`, `is_numpy_ndarray`, `make_tuple`, `to_tuple`, `get_child`, `fromimport`, `panic`, `suppress_finalizing`, `disable_gc`, the ctypes helpers (`from_mv`, `to_mv`, `mv_address`, `to_char_p_p`, `flat_mv`), code pickling, `Profiling`, and `lines`, `printable` and `get_stacktrace` | they work around Python's typing, interpreter, GC or FFI; OCaml types state shapes and arities, and raven has no ctypes. |
| The parts of `helpers.py` OCaml's standard library provides: `flatten`, `partition`, `unwrap`, `merge_dicts`, `count` | `List.concat`, `List.partition`, `Option.get`, `Map.union` and a counter reference. |
| The parts of `helpers.py` whose readers are not ported: `is_image_shape`; `polyN`, `strides_for_shape`, `canonicalize_strides` and `all_int` on symbolic integers (they live with the symbolic integer type in `Ops`); `round_down`, `next_power2`, `to_be32`, `to_be64`, `getbits`, `i2u`, `word_wrap`, `pad_bytes`, `colorize_float`, `temp`, `stderr_log`, `Timing`, `BASEDIR`, `WIN`, the `diskcache` decorator; `capstone_flatdump` and `wait_cond`; `Metadata` | no tolk.next module uses them: the frontend, viz, pickling and the runtime that read them are excluded above. |
| CPython's leniency in `int()` and `float()`, which `getenv` inherits: Unicode white space and decimal digits, and integers beyond OCaml's `int` | environment values are ASCII in practice, matching it needs Unicode tables with no consumer, and no setting needs integers beyond `int`. |
| CPython's refusal to convert an integer whose magnitude rounds to 2^1024 or more (at least 2^1024 - 2^970) to a float (`float(x)` raises `OverflowError`), which `dtype.py`'s `const`, `truncate` and `bitcast` inherit, and so does bounds arithmetic mixing such an integer with a float | an accident of the host language: IEEE conversion rounds such an integer to the infinity of its sign in a double, and once, as the finite value it is, in a narrower float: to the infinity of its sign in a 16-bit or 32-bit float, and to the greatest finite value of its sign in an 8-bit float. |
| CPython's `True == 1` in the node cache, which makes `UOp(op, tag=True)` and `UOp(op, tag=1)` one node | an accident of the host language: `Tag.Bool true` and `Tag.Int 1` are different tags. |
| CPython's `TypeError` when `exec_alu`'s WHERE picks an `Invalid` branch, and `Invalid`'s truth as a condition | an accident of the host language: the chosen branch is the result, `Invalid` included, and an `Invalid` condition is refused. |
| The integer `0` that `helpers.py`'s `cdiv` and `floordiv` return on a zero divisor, whatever the operands, which `exec_alu` keeps for the weak float type | an accident of the host language: a float division by zero gives a float zero. |
| CPython's `TypeError` when a rule of `uop/spec.py` measures or compares an argument of the wrong kind, as for an `MSELECT` or an `UNSHARD` whose argument is `None` | an accident of the host language: the rule rejects the node. |
| CPython's treatment of constants other than integers in `codegen/decomp/op.py`: its power-of-two rules find `True` and `2.0` among the integer keys of `powers_of_two`, `fast_idiv` divides by `True` as by `1` and by a float until `d & -d` raises `TypeError`, and `fast_idiv` and the `c1 < x & x < c2` rule raise `TypeError` on `Invalid` | an accident of the host language: these rules decline a constant that is not an integer. `Tolk_next.Decomp_op › constants other than integers are declined` pins it. |
| CPython's failures in `codegen/gpudims.py` when `_split_dims` must split a symbolic size (`bool` or `math.sqrt` of a node) or has fewer than three bounds (`IndexError`) | an accident of the host language: both raise the `Invalid_argument` "cannot limit dim" that sizes which cannot fit raise. `Tolk_next.Gpudims › grouped_dims › grouped_dims.golden` and `Tolk_next.Gpudims › grouped_dims › symbolic sizes › grouped_symbolic_failures.golden` pin it. |
| `add_gpudims` in `codegen/gpudims.py` putting a warp's size, a node if symbolic, among the integers of `local_max` | an accident of the host language: bounds are integers, so a warp's bound is its size's upper bound, as `_group_dims` bounds a symbolic size. `Tolk_next.Gpudims › add_gpudims › add_gpudims_symbolic_warp.golden` pins it. |
| CPython's `ValueError` when `UOp._min_max` in `uop/ops.py` (`:1113-1114`) shifts the bounds of a shift by a constant negative count, when `exec_alu` computes such a shift, and when `uop/symbolic.py` folds one (`:29`) or computes `1 << k` for a mask (`:145`) | an accident of the host language: a shift by a negative count is undefined, so its bounds are its type's, `exec_alu` refuses it with `Invalid_argument`, and folding such a shift declines. `Tolk_next.Ops › bounds › a shift by a negative constant has its type's bounds`, `Tolk_next.Ops › exec_alu › a shift by a negative count has no value` and `Tolk_next.Symbolic › symbolic_simple › constants › a shift of constants by a negative count stays` pin it. |
| CPython's arithmetic on a constant other than an integer in `mark_range_mod` of `codegen/simplify.py` (`:69`), which asks whether a range's size divides a `True` or a float remainder | an accident of the host language: the rule declines a constant that is not an integer. `Tolk_next.Simplify › pm_split_ranges › a remainder by a constant that is not an integer leaves the range` pins it. |
| CPython's `OverflowError` for a float power that overflows, which `uop/ops.py`'s `safe_pow` lets through `exec_alu` | an accident of the host language: `exec_alu` gives the infinity IEEE gives, as `safe_exp2` already does for `exp2`. |
| CPython's `CalledProcessError` from `ClangCompiler.compile` in `runtime/support/compiler_cpu.py`, which leaves Clang's diagnostics on the terminal where the other compilers raise `CompileError`; and the comgr log that `compile_hip` in `runtime/support/compiler_amd.py` prints on standard output before raising `compile failed` | an accident of the host language: `subprocess` raises its own exception, and printing is how a Python script reports. A toolchain's rejection is a `Compile_error` that carries its message: Clang's diagnostics, NVRTC's log, comgr's log. |
| The order in which `pathlib.iterdir()` lists a directory, which decides the library that `DLL.findlib` in `runtime/support/c.py` (`:110`) finds when several files there match `lib<p>.so[.0-9]*` | an accident of the host language: the file system chooses that order. `C.findlib` takes the first match in the order of names. `Tolk_next.C › one directory › the first ELF file in the order of names is found` pins it. |
| CPython's `struct` packing every float16 NaN as the canonical `0x7e00`, of its sign, and its conversions between float32 and double quieting a signalling NaN, which `dtype.py`'s `float_to_fp16`, storage conversions and `bitcast` inherit | an accident of the host language: storage and `bitcast` keep a NaN's bits, as a kernel's bitcast and nx's do, and a conversion quiets a signalling NaN and keeps its payload, as nx's encoders do. |
| CPython's `TypeError` when `codegen/opt/__init__.py` compares two `Opt`s that split one axis by one amount into different targets, since `AxisType` has no order | an accident of the host language: `Opt.compare` is a total order, with `Upcast` before `Unroll` before `Local`. `Tolk_next.Opt › order › compare is a total order` pins it. |
| CPython's iteration order of a set of integers, which orders the candidate divisors of `nest_by_factor` in `uop/divandmod.py` (`:60`), and so decides which of two equally small results it keeps | an accident of the host language: `Divandmod` tries the divisors in increasing order. `Tolk_next.Divandmod › tinygrad's tests › test_div_by_factor_tie_break.golden` pins it. |
| CPython's tuple comparison of `tuplize` keys in `codegen/late/linearizer.py` (`:36`), which stops at the first element that is not the same object (`uop/ops.py:228`): two nodes whose sources differ only by a tag tie there, whatever their later sources | an accident of the host language: `Linearizer.linearize` orders nodes by `Ops.compare_structure`, which ignores tags and goes on to the later sources. `Tolk_next.Linearizer › linearize › nodes whose sources differ only by a tag are ordered by their later sources` pins it. |
| CPython's `TypeError` when `do_split_ends` in `codegen/late/linearizer.py` (`:90`) sorts ranges by argument and two share their identity but differ in axis type, or one identity is a prefix of the other, since `AxisType` has no order | an accident of the host language: `pm_split_ends` orders identities lexicographically, a prefix before its extensions, then axis types in declaration order, the greatest innermost. `Tolk_next.Linearizer › pm_split_ends › ranges of one identity and different axis types nest by axis type` pins it. |
| CPython's `TypeError` when a rule of `uop/symbolic.py` or `uop/divandmod.py` (`:103-105`) orders or computes with the value of `Invalid`, which is no number, its `OverflowError` or `ValueError` when `simplify_pow` takes the integer part of an infinite or NaN exponent (`:19-20`), and when a cast of an infinite or NaN constant to an integer type folds (`:152-155`) | an accident of the host language: such a rule does not apply. The power is left for `sym` to compute with `xpow`, and the cast stays, since the conversion has no value. `Tolk_next.Symbolic` and `Tolk_next.Divandmod` pin each: `invalid values › a rule that computes with the value of invalid does not apply` in both, `symbolic_simple › powers › x ** infinity and x ** NaN are left to xpow` and `symbolic_simple › constants › a cast of an infinity or a NaN to an integer stays a cast`. |
| CPython's `len` of a device name in `shard_srcs` of `schedule/multi.py` (`:60`): resharding sources that are on one device gives the sharding range as many values as the device's name has characters | an accident of the host language: sources on one device reshard over the range of their unshard, as sources on no device do. |
| openpilot's `pm_fold_moved_after` pass and its `found_after` rule in `schedule/prepare.py` (`:45-60,276`) | a workaround for openpilot's models, which raven does not run. |
| The settings `OPENPILOT_HACKS` and `FLOAT16` of `helpers.py` | they gate openpilot's `pm_fold_moved_after` pass (`schedule/prepare.py:45-60,276`), excluded with it. |
| The setting `CAPTURING` of `helpers.py` | it gates jit capture (`schedule/__init__.py:296`), which is rune's (see D3). |
| `GlobalCounters` and the settings `MAX_BUFFER_SIZE` and `VALIDATE_WITH_CPU` of `helpers.py` | they are read where kernels run and buffers are allocated (`VALIDATE_WITH_CPU` at `engine/realize.py:283`), which is rune's (see D3); tolk.next keeps `compile_linear`'s `validate` argument and the `pm_validate` rewrite. |
| The setting `ALLOW_DEVICE_USAGE` of `helpers.py`, and the contexts that set it (`codegen/__init__.py:464`, `codegen/opt/postrange.py:270`, `engine/worker.py:9`, `function.py:61`) | the guard is structural: tolk.next cannot open a device. |
| The other settings of `helpers.py` whose readers are not ported: `IMAGE`, `JIT`, `WINO`, `TRACEMETA`, `TRAINING`, `LRU`, `HCQ2`, `FUSE_OPTIM`, `USE_ATOMICS`, `CAPTURE_PROCESS_REPLAY`, `NULL_ALLOW_COPYOUT`, `VIZ`, `PROFILE`; the `PYTEST_XDIST_WORKER_COUNT` share of `PARALLEL`'s default; the `{DEV}_CC` migration check, and the `{DEV}_{RENDERER}` one of `Compiled._select_renderer` in `device.py` (`:484`) | image paths, the `Tensor` frontend and `TinyJit`, nx.device's allocators, the legacy AMD queue path, `nn`, process replay, the NULL device, viz and profiling are excluded above; raven never read `{DEV}_CC` or `{DEV}_{RENDERER}`. |

## Tests and ledgers

`test/README.md` describes the suites, the slow tests and the goldens
recorded from tinygrad. `DIVERGENCES.md` lists every place where tolk.next
differs from tinygrad and why, and `test/REGRESSIONS.md` maps each test of
the old tolk and of tinygrad to the test that replaces it.
