# Changelog

All notable changes to this project will be documented in this file.

- Only document user-facing changes (features, bug fixes, performance improvements, API changes, etc.)
- Add new entries at the top of the appropriate section (most recent first)

## [1.0.0~beta1] - Unreleased

### General

- Add compatibility with OCaml 5.5.

### Ppx_ptree (new)

- Add the `ppx_ptree` deriver, which generates the `map`, `map2`, and `iter`
  operations required by `Nx.Ptree.S` for records, products, containers, and
  recursive parameter types, with generated code and diagnostics located at the
  originating source forms.
- Add a runnable linear-regression example using a derived parameter module
  directly with `Rune.grad` and `Rune.jit2`.

### Munin (new)

Local experiment tracking for Raven. Evolves `kaun-board` into a full
experiment tracker — the Raven equivalent of W&B or MLFlow, without a server.
Log metrics and artifacts from your training script, monitor runs live in the
terminal with `munin watch`, then compare results with `munin compare`. Data
is plain JSON on disk, so `jq` and shell scripts work out of the box. Git
commit, command line, and system info are captured automatically. The
`munin.sys` sub-library adds opt-in CPU and memory monitoring in a background
thread.

- Add `x-kaun-mnist-jit`, an example tracking a `Rune.jit2`-compiled CNN
  training run (forward, backward, and SGD update in one compiled program,
  Metal by default) with live `munin watch` monitoring.

### Norn (new)

- New package: Markov chain Monte Carlo sampling with automatic gradients via
  Rune. Provides HMC and NUTS samplers with Stan-style window adaptation (dual
  averaging for step size, Welford estimation for mass matrix). Includes
  symplectic integrators (leapfrog, mclachlan, yoshida), mass matrix metrics
  (unit, diagonal, dense), and convergence diagnostics (ESS, split R-hat).
  Equivalent to BlackJAX/PyMC in Python.

### Tolk (new)

- Normal `dune build` no longer runs the debug golden-test generator; its
  `DEBUG=6` AST diagnostics and `.actual` fixtures are confined to `runtest`.
- Codegen distributes the negation of a sum as a multiply by `-1` over its
  terms, so a later-negated constant-scaled subexpression folds its sign into
  the constant factor (`c*x` negated becomes `(-c)*x`) rather than re-negating
  the scaled product. Deep element-wise chains that reuse a scaled difference
  (e.g. an Euler Lorenz step) now lower to a single canonical form instead of
  keeping a redundant negate.
- Fix a compile-time blowup in schedule creation: `Rangeify.get_kernel_graph`
  re-derived tensor shapes without memoisation and rescanned the whole graph
  history once per kernel, so compile time grew super-linearly in graph size
  and deep element-wise folds (e.g. a 100-step Lorenz integration) stalled
  indefinitely. Shape derivation is now cached and kernel splitting stays within
  a single kernel, so compilation scales linearly.
- Constant-folding an integer `Floordiv`/`Floormod` by zero no longer raises;
  it folds to `0` / the dividend, matching the existing `Cdiv`/`Cmod` guards.
- Fix jitted random-number generation: the call transformation sized staging
  buffers for broadcast expands one rank short, so `Rand` under a captured
  jit failed with `buffer_like: unknown shape`.
- Fix reading realized contiguous views: a slice read back through `Run`
  returned its source's data at offset zero; views now alias the source
  allocation at the correct byte offset and share storage instead of
  copying.
- Full parity refresh against the reference compiler: dtypes are now a flat
  scalar enum (`Dtype.Val`/`Ptr`/`Image` and vector widths are gone — vector
  width comes from a value's shape, pointer provenance from its address
  space), reduces carry an op and leading-axes count, expands prepend leading
  dims with `Uop.broadcast_to` as the same-rank broadcast, and `dtypes.index`
  types shapes and loop bounds. Every generated kernel is verified
  byte-identical to the reference across the parity, codegen, renderer,
  kernel-graph, and debug golden suites.
- Fix kernel search on CPU: waited calls now report elapsed wall-clock time,
  so BEAM can rank CPU kernels — previously every candidate tied at infinity
  and selection was arbitrary.
- Fix kernel cost estimates: vector lanes count toward op/load/store volumes
  again, repeated reads of one buffer cap at its footprint, and tensor-core
  flops count the full matmul volume — beam decisions were skewed on all
  three.
- Fix multi-device scheduling: sharded elementwise graphs no longer hang
  kernel lowering, and ring allreduce no longer crashes on gated chunk
  reassembly.
- Scalar operands adopt their paired tensor's dtype: `float16_t + s` stays
  float16 instead of silently upcasting to float32, matching the reference's
  mixed-precision behavior.
- `Buffer.copy_from` is the canonical buffer copy, routed through the engine;
  `copyin`/`copyout`/`transfer` remain as the low-level allocator bridge.
- `State.safe_load` reads fp8 tensors (`F8_E4M3`, `F8_E5M2`), and
  `load_state_dict` only reconciles the exact scalar-to-one-vector shape
  pair, so stray unit-dimension mismatches fail loudly.
- Clang kernels declare half-precision buffers as `__fp16*`, matching the
  reference's C dialect.
- The renderer drops its unused pre-matcher hook, and `Renderer.make`'s
  emulated-floats option takes flat dtype pairs.
- CPU jit: bfloat16 kernels no longer fail with `Compiler.Compile_error` on
  hosts whose clang predates `__bf16` support (clang < 15 on x86-64). The
  CPU device now probes the compiler once and falls back to the float32
  storage-emulation path already used on riscv64; `Cstyle.clang`/
  `clang_no_abi` gained an optional `?native_bf16` flag.
- `Tolk_nn.Linear.create` and `Embedding.create` now randomly initialise
  their parameters (uniform `±1/sqrt in_features`, Glorot uniform) instead
  of zeros.
- The gpt2 example gains `--temperature` and `--seed`; at non-zero
  temperature it samples from the temperature-scaled softmax and reproduces
  the reference token stream at the same seed.
- Add `Rand`, a counter-based (Threefry) random number frontend:
  `manual_seed`, `rand`, `randn`, `randint`, `uniform`, `normal`,
  `scaled_uniform`/`glorot_uniform`/`kaiming_uniform`/`kaiming_normal`,
  `randperm`, `multinomial`, and `dropout`. Values are deterministic per
  seed, identical across devices, and keep advancing inside `Jit` captures.
  `rand` is limited to 32-bit floats for now. Also adds
  `Elementwise.threefry`, the Threefry-2x32 mixing primitive.
- Fix `Elementwise.neg` (and therefore `sub`) on unsigned integer tensors:
  negation now wraps at the operand's width instead of promoting to a wider
  signed type.
- Add `Compiler.cachekey`, exposing a compiler's disk-cache table name as a
  compiler/architecture fingerprint for callers keying their own caches.
- `Uop.export`/`Uop.import` serialize hash-consed graphs across processes;
  import re-interns every node so structurally-equal live nodes are reused
  physically. Export raises on graphs carrying gradient functions; import
  raises `Failure` on malformed input. `Uop.intern` is removed (it was a
  no-op on foreign graphs); `Schedule.fresh_internal_buffer_slot` is exposed
  for renumbering imported internal buffer slots.
- `Diskcache.put` writes atomically via rename; concurrent writers can no
  longer tear a cache entry.
- The gpt2 example supports `HALF=1`, storing weights and attention
  activations in float16; generated text matches the reference and every
  compiled kernel is byte-identical to it.
- Fix `layernorm` computing the epsilon add in float32 for reduced-precision
  inputs: the constant now follows the operand's dtype, so `float16`/
  `bfloat16` layer norms keep their variance and rsqrt in half precision
  instead of silently widening.
- Fix multi-device scheduling of keepdims-style reductions: the multi
  rewrite minted shape dimensions as `int32` constants while the rest of the
  compiler uses weak integer constants, so broadcasting a realized reduce
  buffer back against a sharded operand failed shape validation and jit
  raised `Failure "buffer_like: unknown shape"`.
- Memoize `Uop.axis` like `Uop.shape`: the unmemoized walk was exponential
  in residual depth, making multi-device compilation of deep networks
  appear to hang.
- `Realize.Buffers.seed_multi` binds a buffer node to a caller-provided
  multi-device buffer, mirroring `seed`.
- Fix multi-device scheduling of computed sharded outputs: buffer allocation
  sized MULTI-wrapped values per shard twice, and the store-revert rule
  cycled on multi-device store targets.
- Fix a rewrite cycle in rangeify when shard axes are realigned across
  devices (e.g. `x @ transpose x` on a sharded value): a symbolic variable
  parameter in a shard offset was mistaken for a buffer and indexed.
- Add a device registry: `Device.register` installs a backend opener per
  name prefix and `Device.get` opens and caches devices by canonical name;
  `Run` registers the CPU/CUDA/METAL backends there.
- The engine now executes multi-device schedules: buffers placed on a
  device tuple allocate one shard per device, kernels launch once per device
  with the `_device_num` variable bound, and copies transfer per shard pair
  (natively within a backend, via a host bounce across backends). Previously
  `Realize.resolve` silently read only the first shard of
  `MSELECT`/`MSTACK`.
- The memory planner suballocates multi-device internal buffers into
  per-placement arenas; single-device planning is unchanged.
- Fix scheduled `COPY`/`SLICE` calls dropping their destination buffer, and
  scalar (rank-0) values losing their flat index through staging and kernel
  splitting.
- Multi-device scheduling now matches the reference: `RING` defaults to 1
  (ring allreduce with >2 devices above the 256k-element threshold), and the
  `LATE_ALLREDUCE` toggle is supported (default 1 wraps allreduce into a
  precompiled function; 0 expands it inline during the multi rewrite).
- Fix several multi-shard rewrite bugs: `SHRINK`-before-`MSTACK` computed
  wrong sizes, pads/shrinks of sharded tensors were rejected or mis-shaped,
  sharded `PARAM`s were never resolved, and unsharding padded to the
  per-shard instead of the full size.
- Each allreduce output now gets a fresh buffer; identical allreduces no
  longer collapse onto one output allocation.
- Graph replay now repatches buffer arguments whose binding was reseeded
  between runs, not just input `PARAM` slots; previously such replays used
  stale device addresses. Changed arguments are diff-patched, so stable
  bindings cost one lookup. New `Jit.batch_graphs` exposes the JIT's graph
  batching to callers driving `Realize` directly.
- JIT replay now batches consecutive CUDA kernels and device copies into
  CUDA execution graphs, replaying each batch as a single `cuGraphLaunch`
  instead of per-kernel launches. Symbolic variable values, launch
  dimensions, and rebound input buffers are patched into the instantiated
  graph on every call. `JIT_BATCH_SIZE` controls the initial batch size;
  `JIT=2` disables batching.
- `Device.make` accepts a `?graph` batched-dispatch capability
  (`Device.Graph`) and `Device.prog` carries the backend kernel `handle`;
  the CUDA device provides both.
- CUDA device-to-host copies (`Device.Buffer.copyout`) now stage through a
  pinned host buffer, improving transfer bandwidth and releasing the OCaml
  runtime lock during the copy.
- Schedule-time bufferize removal now mirrors the reference cost rule:
  staged reads re-index through the producer regardless of range/index
  sizes, so `arange` embedding gathers fold completely and constant-index
  views of computed/assigned tensors (q/k/v selectors, KV-cache reads) no
  longer emit whole-buffer copy kernels. GPT-2 no longer copies the full KV
  cache per layer per decode step (~16% faster CUDA decode).
- Index arithmetic builds `a - b` as `a + b * (-1)` in schedule indexing
  and reduce-collapse rules, and collapsed range-sum clamps use the exact
  reference `min`/`max` structure, so gather offsets cancel symbolically.
- Weakint comparisons now lower with the index-dtype pass, keeping
  valid-bounded gather indices in 32-bit arithmetic instead of widening to
  int64.
- Fix C-style renderers duplicating upcast lane-0 load/store address
  expressions: `render_index` no longer re-orders `ADD` index chains at
  render time, so lane 0 reuses the shared named subexpression (`aluN`)
  instead of re-deriving it with a different term order.
- `examples/gpt2` now decodes through a per-layer key-value cache with
  symbolic positions and a captured JIT: one kernel set serves every decode
  step, taking greedy generation from ~2 tok/s to ~19 tok/s on CUDA
  (`--validate` reproduces the reference texts on CUDA and CPU).
- `Creation.full`/`zeros`/`ones` (and `_like` variants) now materialize a
  fresh buffer by default so in-place `Op.assign` has storage to write to;
  pass `~buffer:false` for the previous fold-into-consumers broadcast
  constant.
- `Run.realize` no longer fails on tensors whose graph folds to a constant
  expression (e.g. a realized `arange`); they stay lazy.
- Fix a shared-memory sizing miscompile in grouped reductions (the
  `GROUP`+`LOCAL`+`UPCAST` matvec path raised "invalid RESHAPE"); local
  staging buffers are now materialized by codegen.
- Kernel-source fidelity fixes: float `x + y*-1` renders as `x - y`; folded
  float constants keep full precision; kernels with two symbolic variables
  no longer emit invalid `make_void()`; dimensionless kernels are named
  `E`/`r` without a trailing underscore.
- Add `tolk.nn`: `Embedding`, `Linear`, and `Layer_norm` layers plus
  `State.safe_load`/`State.load_state_dict` for loading safetensors
  checkpoints into layer parameters.
- Add a GPT-2 (124M) text-generation example
  (`packages/tolk/examples/gpt2`) that reproduces reference greedy
  generations on CPU and CUDA.
- Add `Run.of_bytes` to create tensors from raw little-endian bytes and
  `Run.device_name` to inspect the realization device.
- Fix exponential graph walks in `Uop.ranges`, `Uop.addrspace`,
  `Uop.semantic_key`, symbolic lane counting, and the C-style renderer by
  memoizing them; realizing deep transformer graphs and reducing over
  prime-sized axes now takes milliseconds instead of minutes.
- Add `Jit`, a capture-and-replay JIT for tensor functions: the first call
  runs eagerly, the second records and compiles every kernel the function
  realizes, and later calls replay the compiled program on fresh inputs and
  symbolic variable values without rebuilding, rescheduling, or recompiling.
  Buffers backing live tensors (weights, KV caches, outputs) keep their
  storage across replays; other intermediates are folded into arena memory.
- `Run` now exposes the execution device and buffer storage registry
  (`device`, `buffer_of_node`, `buffer_nodes`), and `Tensor.live_tensors`
  lists the tensors currently reachable by the program.
- Kernels with symbolic sizes now render with reference-parity kernel names
  (e.g. `r_28start_pos2B129`), symbolic loop bounds, and symbolic GPU launch
  dimensions; new `Render.expr_to_string` renders scalar expressions
  compactly.
- Fix reductions over symbolically-sized axes silently collapsing: range
  creation now consults expression-level shapes.
- Support symbolic shapes in the tensor frontend: new `Movement.symbolic_shrink`,
  `symbolic_reshape`, and `symbolic_broadcast_to` entry points, and
  symbolic-shape handling through broadcasting, `dot`, `softmax`, reductions,
  and `Op.assign` — enough for a KV-cache transformer decode step where the
  sequence position is a bound variable. Operations that need concrete shapes
  (`pool`, `split`, strided indexing, ...) keep raising `Invalid_argument`.
- Add symbolic-integer helpers to `Uop`: `resolve`, `smax`, `smin`, `sprod`,
  a checked `broadcast_shape`, and `unbind` for splitting a `Bind` into its
  variable and value. `Uop.bind` now validates the value against the
  variable's range.
- Fix the engine so one schedule serves every bound value of a symbolic
  variable: callified `Bind` inputs keep their variable name and range,
  kernel graphs recover the canonical variable, and variable values are
  passed to kernels at launch instead of being resolved as buffers.
- Fix post-schedule parameter substitution to index call arguments including
  `Bind`s; previously a buffer argument following a `Bind` was bound to the
  wrong slot.
- JIT replay is now parameter-substitution based: capture substitutes input
  buffer nodes with slotted `Param`s, memory-plans the combined schedule
  once, and replays with per-call `input_uops` and `var_vals`, so replays
  with different variable bindings get correct kernel `vals` and launch
  dimensions. `Jit.call` takes input buffer nodes, and its `held_buffers`
  argument is honored: buffers that outlive the jitted computation (e.g.
  in-place caches) keep their allocation instead of being folded into
  arenas.
- The schedule capture hook moved to `Realize.capturing`;
  `Schedule.create_linear_with_vars` hands captured schedules over unplanned
  and its `?memory_plan` flag is removed (`Schedule.memory_plan_rewrite` is
  now exported). Capture can be disabled with `CAPTURING=0`.
- Add a CUDA runtime: tensor programs now compile through NVRTC and execute
  on NVIDIA GPUs. The driver and NVRTC libraries are loaded dynamically at
  run time, so builds do not require a CUDA toolkit and fail cleanly at
  device creation when no GPU is present.
- The default device is now chosen by scanning available backends in
  priority order (`METAL`, `CUDA`, `CPU`); set the `DEV` environment
  variable (e.g. `DEV=CUDA`, `DEV=cpu`) to force one.
- Fix gated vectorized loads: the masked fallback rendered a scalar zero
  for a vector access, which the CUDA compiler rejects; the zero is now
  stacked to the access width.
- Add in-place assignment: `Op.assign` records a buffer write in the graph,
  including writes through sliced views (e.g. a transformer kv-cache update),
  and repoints every live tensor aliasing the buffer so later reads observe
  the write. `Run.realize` now also rebinds assigned and previously realized
  nodes onto their computed buffers, so an assignment executes once and reads
  of a realized tensor reuse its buffer instead of recomputing.
- Add `Op.scaled_dot_product_attention` (optional additive or boolean
  `attn_mask`, or `is_causal` masking) and `Op.layernorm`.
- Fix a compilation- and schedule-cache collision: the cache key hashed node
  payloads with the polymorphic hash, which cannot distinguish constants such
  as `0` and `-1` (OCaml folds the halves of an `int64` with xor), so two
  kernels differing only in such a constant could silently share one compiled
  program. Constant payloads are now rendered exactly into the key.
- Fix buffer identity reuse across realizations: allocation slots are now
  drawn from one process-wide counter (`Uop.fresh_buffer_slot`), so two
  distinct allocations can no longer hash-cons onto the same node and
  read back each other's storage.
- CUDA source generation gains an `SM90` (Hopper) target tier in
  `Gpu_target.cuda`: `sm_90` uses the `sm_89` tensor-core table and fp8
  dtype support, and `CUDA_ARCH`/`CUDA_SM` values of 90+ resolve to `SM90`.
- Tensor-core (WMMA) kernels now lower end to end: fixed the WMMA output
  shape rule, upcast-axis deduplication, warp-lane decomposition, and WMMA
  devectorization, so tensor-core matmuls render byte-identical to the
  reference (covered by an fp8 `mma.sync` parity golden).
- New package: a minimal ML compiler for tensor computation. A tensor program
  is a graph of micro-operations; Tolk schedules it into kernels, lowers and
  optimizes them, renders C-style source, compiles, and executes the result —
  entirely from OCaml. Equivalent to tinygrad in Python.
- The `Tolk_frontend` library builds these graphs through a NumPy-style tensor
  surface: broadcasting and dtype promotion, movement ops, reductions,
  `matmul`, `conv2d` and pooling, cumulative scans, `sort`/`argsort`/`topk`,
  `gather`/`scatter`, `masked_select`/`nonzero`, `softmax` and friends, and
  NumPy-style advanced indexing — and executes them end to end
  (`Run.realize`) with host-data round-tripping.
- A capture/replay JIT (`Tolk.Jit`) records a traced program once and
  re-executes it against new inputs. Compiled kernels are first-class graph
  nodes carrying their rendered source and compiled binary.
- Runtimes: CPU (via Clang) and Metal. Renderers additionally target CUDA,
  AMD/HIP, and OpenCL for GPU code generation.

### Vega (new)

- Add `Loss_scale` for float16 training: static and dynamic loss scales
  with `scale`/`unscale`/`grads_finite`/`adjust`; all state is scalar
  tensors updated by `Nx.where` arithmetic, so it threads through
  `Rune.jit`/`pmap` steps and adapts across compiled calls.
- `sgd_step` with `momentum = 0.` (the default) no longer reads or updates
  the velocity state; under `Rune.jit` this stops parameter-sized zero
  velocities from being captured and transferred every step.
- New package: gradient-based optimizers and learning-rate schedules. Built
  on Nx with no autodiff dependency. Equivalent to Optax in JAX.
- The primary surface is structural: `sgd_init`/`sgd_step`,
  `adam_init`/`adam_step`, `adamw_init`/`adamw_step`, `global_norm`,
  `clip_by_global_norm`, and `clip_by_value` step whole parameter
  structures — any type implementing `Nx.Ptree.S` — with optimizer state
  shaped like the parameters themselves.
- Below it, the per-tensor tier composes Optax-style gradient
  transformations on single tensors via `Vega.chain` (RMSprop, Adagrad,
  Lion, LAMB, RAdam, LARS, Adan, Adafactor, ...).
- Schedules are unified across both tiers: a schedule is a plain
  `int -> float` function; structural loops evaluate it at the step counter
  and pass `~lr`, while per-tensor chains consume it via
  `scale_by_learning_rate`/`scale_by_schedule`.
- **Breaking** (relative to earlier unreleased revisions): the per-tensor
  `clip_by_value : float -> t` transformation is renamed to `clip`; the
  `clip_by_value` name now belongs to the structural gradient
  transformation.

### Nx

- The default C backend and `nx.io` codecs now build cleanly with strict GCC
  warnings and single-pass ELF linkers. Empty DEFLATE streams also avoid
  allocating the encoder's match tables.
- Add float32- and float64-preserving `dct`, `idct`, `dst`, and `idst`
  transforms of types I–IV, including N-D variants and forward, backward, and
  orthonormal scaling modes.
- `Nx.concatenate` on the OxCaml backend now uses SIMD and unrolled contiguous
  block copies, with stride-aware paths for offset, transposed, flipped, and
  broadcast views.
- `truncated_normal` now rejects integer dtype witnesses at compile time,
  matching the other normal samplers.
- `rand` and `randn` now reject integer dtype witnesses at compile time instead
  of accepting them and raising `Invalid_argument` at runtime.
- Fix elementwise arithmetic on non-contiguous views: `mul_s`, `div_s`, and
  tensor `div` now honor the view's offset and strides instead of reading
  out-of-view values from the underlying buffer.
- Replace the vendored camlzip and stb image libraries with owned ISC codecs in
  `nx.io`. NPZ no longer creates a temporary NPY file for each entry, image and
  archive decoding writes directly into Nx buffers, and Nx no longer needs zlib
  or pkg-config.
- Speed up `load_npy`, stored `load_npz`, compressible `save_npz`, and `gunzip`
  by removing redundant checksum passes, processing stored data in larger
  batches, and bounding DEFLATE match searches.
- Add `Nx_io.gunzip` with checksum validation and atomic destination replacement.
  **Breaking:** `save_image` now supports only modern PNG and JPEG output; BMP
  and TGA output and the public `nx.zip`/`nx.io.stb_image*` libraries are removed.
- Replace the default `nx.c` backend with the self-contained C implementation.
  FFT and dense linear algebra no longer require PocketFFT, OpenBLAS, LAPACKE,
  libomp, or platform depext configuration; macOS automatically uses Accelerate
  for eligible matmuls and every platform retains the owned GEMM fallback.
  **Breaking:** the public `nx.pocketfft` vendored library is removed.
- Preserve Nx semantics across the backend cutover for detailed `matmul` shape
  errors, empty `all`/`any`, vector right-hand sides, explicit-size real FFTs,
  and complex pseudoinverses.
- Prevent parallel `nx.c` operations from hanging in a forked child by rebuilding
  the backend worker pool after `fork`.
- Correct the backend interface docs: `reshape` never copies — it raises
  `Invalid_argument` when the existing strides cannot express the new shape —
  and `triangular_solve`'s `transpose` solves with the conjugate transpose
  (`Aᴴ`) for complex dtypes.
- Unify random number generation on one splittable Threefry generator, reached
  through `Nx.Rng`. The explicit samplers `Nx.Rng.uniform`/`normal`/`randint`/
  `bernoulli` are pure, order-independent functions of a key; the implicit scope
  `Nx.Rng.run`/`with_key` still drives the keyless `Nx.rand`/`randn`/… with no
  key argument, and now shares the explicit stream by construction (`Nx.rand` is
  `Nx.Rng.uniform` on a subkey). A key is now a transparent `[|2|]` int32 tensor
  (previously an opaque host int), so it flows as a parameter-tree leaf, a jit
  input and a `vmap`/`pmap` axis; `Nx.Rng.fold_in_axis` derives a per-lane key
  under a transform. `Nx.Rng.to_int` is removed — read a key with `Nx.to_array`.
  Breaking: the same seed now yields different `Nx.rand`/`randn`/… values.
  Migration: re-bless any exact-value goldens; keyless call sites are unchanged.
- Split the backend contract's `eig`/`eigh` (each a `vectors:bool -> ... option`
  returning an optional vectors component) into four total functions matching
  the public API: `eigvals`/`eigvalsh` return values only, `eig`/`eigh` return a
  non-optional `(values, vectors)` pair. The values-only variants drive the
  cheaper LAPACK no-vectors path. Removes a representable invalid state (a
  runtime flag steering an option) from the contract. `Nx.eig`/`eigh`/`eigvals`/
  `eigvalsh` are unchanged.
- Add `Nx.Linalg_error`, a typed exception for numeric linear-algebra failures,
  carrying the failing operation and a `kind`
  (`` `Not_positive_definite ``, `` `Singular ``, `` `No_convergence ``). A
  non-positive-definite `cholesky` now raises `Linalg_error` (previously an
  untyped `Invalid_argument "cholesky: not positive-definite"`) and a failed
  `qr` raises it with `` `No_convergence ``. Precondition violations (non-square
  input, wrong dtype) still raise `Invalid_argument`.
- Make the backend contract's `scatter` take required `~mode` and
  `~unique_indices` labels instead of optionals, adopting the rule that
  backend-contract operations carry no optional arguments (user-facing defaults
  live on the frontend). `Nx.scatter` keeps its `?mode`/`?unique_indices`
  defaults.
- Collapse the backend contract's four reductions (`reduce_sum`, `reduce_prod`,
  `reduce_max`, `reduce_min`) into a single `reduce ~op ~axes`, matching
  `associative_scan`. The op always returns the result with the reduced axes
  removed; the frontend reinserts size-1 axes for `~keepdims:true`, so backends
  no longer implement `keepdims`. `Nx.sum`/`max`/`min`/`prod` are unchanged.
- Split the backend contract's `div` into `fdiv` (IEEE 754 float/complex
  division) and `idiv` (truncated integer division), matching the effect
  layer's existing dtype dispatch. Backend implementors now provide two
  domain-specific primitives instead of one that branches on dtype; the
  frontend selects between them. `Nx.div`'s behavior is unchanged.
- Support boolean-mask indexing in `slice` and `set_slice`: an `M mask` spec
  selects (or writes at) the positions where the rank-1 boolean `mask` is true
  along the axis it addresses. The mask length must equal that axis.
- Require the labeled argument `~indices` in `take` and `take_along_axis`, for
  consistency with `put`, `scatter`, and the other indexing functions.
- Require `~axis` in `concatenate`. The old axis-less form silently flattened
  every input; ravel the inputs first to recover it.
- Remove `squeeze_axis` and `unsqueeze_axis`; use `squeeze ~axes:[ i ]` and
  `unsqueeze ~axes:[ i ]`.
- Remove the per-element tensor-form `map`, `iter`, and `fold` (each scalar
  presented as a scalar tensor); use the faster `map_item`, `iter_item`, and
  `fold_item`, which pass raw scalars.
- Remove the formatting helpers `format_to_string`, `print_with_formatter`,
  `print_data`, `dtype_to_string`, `shape_to_string`, and `pp_shape`. `pp`,
  `pp_data`, `print`, `to_string`, `data_to_string`, and `pp_dtype` remain.
- Remove the numpy stack shorthands `vstack`, `hstack`, and `dstack`, along
  with the `Nx.Infix` concatenation operators `( @= )` and `( @|| )`. Use
  `concatenate`/`stack` directly, reshaping 1-D inputs as needed.
- Remove the commutative reverse-scalar aliases `radd_s`, `rmul_s`,
  `rmaximum_s`, and `rminimum_s`; use `add_s`, `mul_s`, `maximum_s`, and
  `minimum_s` (the operands commute). The non-commutative `rsub_s`, `rdiv_s`,
  `rpow_s`, and `rmod_s` remain.
- Remove redundant property and conversion aliases: `size` (use `numel`),
  `dims` (use `shape`), `astype` (use `cast`), `clip` (use `clamp`), `invert`
  (use `bitwise_not`), `expand_dims` (use `unsqueeze ~axes`), `identity` (use
  `eye`), `stride i t` (use `(strides t).(i)`), and `lerp_scalar_weight` (use
  `lerp` with a `scalar_like` weight).
- Remove the duplicate `cmp*` comparison family (`cmplt`, `cmpne`, `cmpeq`,
  `cmpgt`, `cmple`, `cmpge`). Use the named spellings `less`, `not_equal`,
  `equal`, `greater`, `less_equal`, `greater_equal` instead.
- Declare the licenses of the vendored components in `nx.opam`: the package
  now advertises `ISC AND LGPL-2.1-or-later WITH OCaml-LGPL-linking-exception
  AND BSD-3-Clause AND (MIT OR Unlicense)` covering camlzip, pocketfft, and
  stb_image, instead of claiming plain ISC.
- Remove the `?out` parameter from the backend `fft`/`ifft`/`rfft`/`irfft`
  operations. It was the only destination-passing parameter in the backend
  interface and the frontend never passed it; the FFT ops now allocate their
  result like every other compute operation.
- `einsum` failures now raise `Invalid_argument` with an `einsum:`-prefixed
  message like every other frontend error, instead of bare `Failure`.
- Remove the unused scalar-arithmetic surface from `Nx_core.Dtype`: `add`,
  `sub`, `mul`, `div`, and `bits`. Element arithmetic is performed by the
  backend kernels; these host-side helpers had no callers.
- Remove the unused validity-mask machinery from `Nx_core.View`: the `?mask`
  argument of `View.create` and the `mask`, `is_valid`, `linear_index`,
  `pad`, `strides_opt`, `can_get_strides`, and `is_materializable` functions.
  No view ever carried a mask (eager `pad` copies into a fresh buffer), so
  every view now has well-defined strides and `View.strides` is total.
  `View.create` validates the length of explicit `?strides` eagerly.
- Fix `float8_e4m3` conversions: the top binade was broken (256–448 saturated
  to 448 on write and decoded as 240 or NaN on read) and values below `2^-6`
  underflowed to zero instead of using the format's subnormals down to
  `2^-9`. Out-of-range values and infinities now convert to NaN instead of
  saturating to ±448, matching `ml_dtypes` and PyTorch `float8_e4m3fn` casts;
  clamp before casting if saturation is wanted. `float8_e5m2` subnormal
  rounding now keeps the sticky bits, so round-to-nearest-even resolves ties
  correctly. Both conversions apply to buffer element access and every C
  kernel operating on float8 tensors.
- The js_of_ocaml stubs for extended dtypes now compute the same values as
  the C implementation. Previously on JavaScript, `Nx_buffer.kind` returned
  the wrong dtype for every buffer, creating a bfloat16/float8/bool buffer
  raised, bfloat16 stores truncated instead of rounding, int4 stores raised
  on out-of-range values instead of clamping, uint64 element access threw,
  and the bytes blits read garbage.
- Fix int4/uint4 offset arithmetic in `Nx_buffer.blit_from_bytes` and
  `blit_to_bytes`: source and destination offsets disagreed about nibble
  packing (one side counted a byte per element, the other rounded the byte
  offset up), silently corrupting any copy with a nonzero offset. Offsets are
  element offsets mapping to byte `off / 2` on both sides; odd offsets now
  raise `Invalid_argument`, as does an odd length that does not reach the end
  of the destination buffer.
- `Nx_buffer.to_bigarray1` now raises `Invalid_argument` for extended kinds
  (bfloat16, float8, int4, uint32/64, bool) instead of returning a bigarray
  that standard operations silently misread — `Bigarray.Array1.get` decoded
  bfloat16 bits as float16, and int4 buffers read out of bounds.
  `of_bigarray1` and `of_genarray` likewise reject `Char`, `Int` and
  `Nativeint` bigarrays, which buffers never supported. Marshalling buffers is
  now documented as unsupported (it silently dropped the extended kind).
- Merge `Nx_buffer.kind` and `Dtype.t` into a single GADT: a dtype now *is*
  the buffer kind (`('a, 'b) Dtype.t = ('a, 'b) Nx_buffer.kind`).
  `Dtype.of_buffer_kind` and `Dtype.to_buffer_kind` are gone — pass the dtype
  directly. `Nx_buffer` constructors and values now use the dtype spellings
  (`Int8`/`int8` instead of `Int8_signed`/`int8_signed`, `Complex64` for the
  8-byte complex, `Complex128` for the 16-byte one), and the extended element
  types are renamed accordingly (`int4_elt`, `uint4_elt`, `int8_elt`,
  `uint8_elt`, `int16_elt`, `uint16_elt`). New `Nx_buffer.kind_name` names a
  kind; `Dtype.to_string` is now an alias for it.
- Reductions along non-innermost axes stream rows instead of striding a cache
  line per element: `sum ~axes:[0]` on 512×512 is ~9.6× faster (and `mean`
  with it), with bit-for-bit identical results.
- Vectorize `sum` along the contiguous axis (`sum ~axes:[1]` on a C-contiguous
  matrix): ~12.5× on 512×512.
- Contiguous elementwise ops stay serial-SIMD below 16M elements instead of
  parallelizing at 32768: a single vectorized core saturates memory bandwidth,
  so `add`/`mul` on 1M floats are ~7× faster on Apple Silicon.
- `copy`, `contiguous`, and axis-aligned `concatenate` collapse to a single
  `memcpy` when source and destination regions are contiguous (~31× on a
  512×512 `concatenate`); results are bit-identical.
- Speed up full-array `sum`: the reduction is now vectorized instead of
  parallelized (fork/join overhead dominated the bandwidth-bound sum) — up to
  125× at 128×128 and 20× at 1M elements on Apple Silicon.
- Benchmark suites across the workspace now run under a dedicated `bench`
  alias with a shared lock instead of `runtest` (`nx` and its
  `matmul`/`conv2d`/`einsum` suites, `norn`, `talon`, `vega`, `fehu`,
  `nx-oxcaml`, `brot`, `sowilo` — matching `rune` and `kaun`, which already
  did this). `dune runtest` no longer runs perf regression checks (which could
  fail an ordinary test run on measurement noise); run them with
  `dune build @bench`.
- Expand `bench_nx` to ~30 cases across `binary`, `unary`, `reduce`, and
  `structural` groups — adding `sub`/`div`, unary elementwise, axis-wise
  reductions, broadcasting, non-contiguous inputs (transposed-view operands,
  strided-axis reductions, transpose materialization), and
  `cast`/`copy`/`concatenate`. A `lab` tag marks a fast representative subset
  (select with `--tag lab`).
- Fix unary `-` from `Nx.Infix` to negate tensors with `neg`. It previously
  performed `logical_not`, unexpectedly turning zero into one and nonzero
  values into zero.
- **Breaking.** Remove the `^` logical-XOR operator; use `logical_xor`
  directly. Its concatenation-level precedence misgrouped comparisons.
- **Breaking.** Remove the `<.>` dot-product operator; use `dot` directly.
  Its comparison-level precedence grouped mixed arithmetic unexpectedly.
- **Breaking.** Rename the infix matrix-multiplication operator from `@@` to
  `*@`, giving it multiplication precedence in mixed arithmetic expressions.
- Fix `cast` and all float16 compute: the float32-to-float16 conversion
  corrupted any value with an odd biased exponent that needed mantissa
  rounding (e.g. casting `0.274` to float16 returned `0.5`), converted `inf`
  to `nan`, and flushed subnormals to zero. Conversion is now IEEE
  round-to-nearest-even with subnormal support, matching numpy. Casting a
  signaling NaN to `bfloat16` no longer returns `inf`.
- Add deferred host tensors to `nx.effect`: `Nx_effect.deferred` creates a
  tensor whose bytes arrive on first data access. Metadata reads (`shape`,
  `dtype`) answer without transfer; the first read runs a fill thunk once
  and memoizes the result. Rune uses them to keep jit outputs
  device-resident.
- Add `scatter`, the pure counterpart of `put_along_axis`: returns a new
  tensor with `values` placed along `axis`, with `` `Set``/`` `Add`` modes
  and a `unique_indices` hint. Works under `Rune.jit` and differentiates
  with respect to both inputs.
- Fix `` `Add``-mode scatter on the C backend to accumulate updates into the
  template's values instead of a zeroed buffer, matching the jit lowering
  and the autodiff rule.
- Add `Nx_buffer.unsafe_data_ptr`: the address of a buffer's first element,
  for wrapping tensor memory in external systems without copying. The caller
  must keep the buffer reachable while the pointer is in use.
- Fix `rfft` and `irfft` bypassing the effect-based backend dispatch: they
  called the C backend directly, making them invisible to every effect
  handler (autodiff, vmap, jit). They now perform `E_rfft`/`E_irfft` like
  the other FFT operations, with the target `dtype` carried in the effect.
- Add `Nx.Ptree`: parameter trees. The `Ptree.S` module type is the traversal
  interface shared across the ecosystem — autodiff transformations (Rune),
  structural optimizers (Vega), and checkpointing (Kaun) all operate on any
  user structure implementing its three traversals (`map`, `map2`, `iter`).
  A stock dynamic tree (`Ptree.t` with tensor, list, and dict nodes) covers
  structures only known at runtime.
- Require OCaml >= 5.5.0 (module-dependent functions are used by the
  `Ptree.S`-based APIs downstream).
- Fix `flatten` raising on rank-0 tensors; it now reshapes them to `[|1|]`.
- Extend safetensors I/O: cover the remaining dtypes with SafeTensors
  equivalents (float64, int64, int8, uint8, bool, ...) and support rank-0
  tensors. Dtypes with no SafeTensors equivalent (complex, int4) fail with a
  clear error.
- Route `to_host` through a new `E_to_host` effect so effect handlers observe
  value reads. Transformations can now materialize or reject concretization
  deliberately (a JIT tracer needs this; reading a batched tensor inside a
  vectorizing map is now detectable instead of silently exposing the physical
  buffer).
- Fix the scatter effect dropping its mode: `scatter ~mode:\`Add` was silently
  executed as a `Set`-mode scatter whenever an effect handler (autodiff, vmap)
  intercepted the operation. `E_scatter` now carries `mode` and
  `unique_indices`.
- Remove `~out` parameter from all backend compute operations. Operations now
  allocate and return their result instead of writing to a caller-provided
  buffer. This simplifies the effect system, fixes vmap, and prepares the
  architecture for JIT compilation.
- Add `Shape.reduce_output_shape` for computing output shapes after axis
  reduction.
- Add machine learning examples: PCA, K-Means, DBSCAN, and t-SNE implemented
  from Nx primitives.
- Fix incorrect results for views and slices in binary, unary, ternary, cast,
  and shape C stubs. The `iterate_inner_dims` helpers did not account for the
  ndarray offset, producing wrong results when the data starts at a non-zero
  offset in the underlying buffer.

### Rune

- `jacfwd'` and `jacrev'` support float32 and float64 inputs without an
  implicit float64 specialization. Forward-mode Jacobians keep the output
  dtype, reverse-mode Jacobians keep the input dtype, and both evaluate the
  differentiated function only once.
- `pmap` now decorrelates per-device randomness: under `Rune.pmap`/`pmap2`,
  `Nx.Rng.fold_in_axis key` folds each device's own index into the key, so a
  replicated key yields an independent draw per device (device `i` draws
  `Nx.Rng.fold_in key i`). Data-parallel dropout masks now differ across
  devices instead of replicating. Previously every device drew the identical
  values from a replicated key.
- The `rune` bench suite now covers `jit`: a `Jit` group times compiled
  execution of the MLP forward pass and the deep elementwise chain against
  their eager equivalents, with compilation hoisted out of the measured region.
- Random number generation lives in `Nx.Rng`: `rune` no longer declares a `Rng`
  module or a `type key`. The unified splittable keys and samplers are reached
  as `Nx.Rng.*`; rune's transforms only answer the generator's effects (jit
  lowers threefry, `vmap` batches per-lane keys with `Nx.Rng.fold_in_axis`),
  adding no RNG vocabulary of their own. Migration: rename `Rune.Rng.*` to
  `Nx.Rng.*`.
- `jit` now compiles random number generation: threefry lowers to the
  compiler's primitive with bit-exact parity to eager execution. A key that
  does not depend on the jitted function's inputs (`Nx.rand` and friends, or
  a captured key) raises `Jit_error` at trace time instead of silently
  replaying one frozen draw per call — thread an `Nx.Rng` key through the
  inputs.
- `jit` compilations now persist across processes: compiled kernels are
  stored on disk (`$XDG_CACHE_HOME/tolk/rune_jit`) keyed on the traced
  computation and compile environment, so a warm process skips scheduling,
  lowering, and kernel compilation (gpt2 train first step ~17 s -> ~4.4 s,
  results bit-identical). Set `JITCACHE=0` to disable; `pmap` compilations
  are not persisted.
- `pmap` now differentiates through `~keepdims:true` reductions
  (`max`/`sum`/`mean`), unblocking softmax, layer norm, attention, and the
  stock losses in data-parallel training.
- `jit`, `jit2`, `jit'`, `pmap`, and `pmap2` gain `?donate` (default
  `false`): the call consumes device-resident input handles, releasing
  their buffers to the allocator once it completes, so a state-to-state
  loop holds ~2 generations of device memory instead of one per call
  awaiting GC (9x lower peak on a 512 MB synthetic state loop). Reading a
  donated handle raises `Invalid_argument`; host tensors and handles
  already read are unaffected.
- Add `pmap` and `pmap2`: compile a function to run in parallel across a
  device tuple. `in_axes` shards or replicates each input leaf; the function
  observes global shapes and reductions over a sharded axis become
  cross-device allreduces automatically, so differentiating a mean loss
  inside `pmap` yields data-parallel gradients. Outputs stay resident per
  device and feed back into matching placements with no transfer.
- Fix `jit`/`jit2` raising `Jit_error` ("not scheduled to a buffer") when a
  function returned an input leaf of rank 2 or higher unchanged.
- `jit` compiled programs now replay through device execution graphs (CUDA
  graphs): consecutive kernels batch into single graph launches, honoring
  `JIT` (>= 2 disables) and `JIT_BATCH_SIZE`. GPT-2 training drops ~116 to
  ~110 ms/step and decode rises ~320 to ~380 tok/s on an H100.
- **Breaking.** `jit` closure captures are now compile-time constants on
  every device: they are bound once when the trace compiles and never
  refreshed, and a jitted function that assigns to a capture (`assign`,
  `blit`) raises `Jit_error` at trace time instead of writing state back.
  Thread mutable state through the input structure instead; assigning to an
  input leaf still writes back on every call. Mutating a captured tensor
  between calls now has unspecified visibility (the CPU device may observe
  it through zero-copy aliasing; other devices never do).
- Tensors captured by a jitted closure are uploaded to the device once per
  closure and shared by all compiled signatures, instead of once per
  signature — halving resident weight memory for prefill+decode closures.
- `jit` outputs on CUDA and Metal now stay resident on the device until
  read: metadata reads never transfer, and an unread output fed back as an
  input of a jit call on the same device seeds the compiled program directly
  with its device buffer. Iterated jitted calls (training steps, decode
  loops) no longer round-trip state through the host: GPT-2 decoding goes
  from ~142 to ~320 tok/s at 100 tokens and training steps from ~380 to
  ~117 ms on an H100, with bit-identical results. Device memory is reclaimed
  when outputs are read or collected (budget via
  `RUNE_JIT_RESIDENT_BUDGET`).
- Add `Rune.jit_stats`/`Rune.reset_jit_stats` transfer counters, a
  `RUNE_JIT_DEBUG=1` per-call transfer log, and `RUNE_JIT_FORCE_COPY=1` to
  exercise the device copy path on the CPU device.
- Inside `jit`, `Nx.full`/`Nx.zeros`/`Nx.ones` (and `*_like`) now trace as
  broadcast scalar constants instead of captured host tensors re-uploaded on
  every call, scalar constants fold into kernels as immediates, and replay
  reuses its transfer staging buffers — a jitted GPT-2 124M train step on
  CUDA drops from ~2.1 s to ~0.35 s.
- `jit` uploads closure-captured tensors to non-CPU devices once per
  compilation instead of on every call (captures the function assigns to are
  still re-read each call, so in-place state carries across calls). Jitted
  functions capturing large weights no longer pay a full re-upload per call.
  CPU behavior is unchanged.
- `jit` accepts `~device:"CUDA"`: jitted programs compile through NVRTC and
  run on NVIDIA GPUs.
- `jit` takes a `?device` argument selecting where kernels compile and run:
  `"CPU"` (the default) or `"METAL"` on macOS.
- On the CPU device, jitted programs now run on the tensors' own memory:
  contiguous inputs and captured tensors are read in place and outputs are
  computed directly into the returned tensors' storage, removing the byte
  copies previously made on every call. Non-contiguous tensors and other
  devices still go through copies.
- Add just-in-time compilation: `jit`, `jit2`, and `jit'` trace a function
  once per input signature (leaf dtypes and shapes), compile the trace into
  fused native kernels through the Tolk compiler, and replay the compiled
  program on subsequent calls. Differentiating inside a jitted function
  compiles the forward and backward passes together — a whole training step
  (forward, backward, parameter update) compiles into one program; under an
  enclosing transformation (`grad`, `vmap`, `with_debug`) the wrapped
  function runs eagerly so results never change. Sliding-window operations
  (`extract_patches`/`combine_patches`, and convolution built on them)
  compile too. Reading a traced tensor's value or using an operation the
  compiler cannot express (FFT, linear algebra, RNG, complex dtypes) raises
  `Jit_error` at trace time instead of compiling a wrong program.
- **Breaking.** Ground-up rewrite. Transformations now operate over typed
  parameter structures: `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`,
  `hvp`, and friends take a first-class module implementing `Nx.Ptree.S`
  and return gradients with the same structure and leaf dtypes as the
  parameters — mixed-dtype parameters differentiate in a single forward
  and backward pass. Functions of a single tensor use the primed variants
  (`grad'`, `value_and_grad'`, `vjp'`, `jvp'`, `vmap'`, `jacfwd'`,
  `jacrev'`, `hessian'`, `hvp'`).
- New transformation surface: structured-output `vjp2`/`jvp2`/`vmap2`,
  reusable pullbacks (`vjp_fun`), gradient checkpointing (`remat`),
  Hessian-vector products (`hvp`), custom differentiation rules
  (`custom_vjp`, `custom_jvp`), directional gradient checking
  (`check_grads`), staging-ready control flow (`scan`, `cond`,
  `while_loop`), and operation logging (`with_debug`, replacing `debug`).
- Removed: the list-based variants (`grads`, `value_and_grads`, `vjps`,
  `jvps`, ...) — use a `Ptree.S` structure instead; the finite-difference
  `check_gradient` API — use `check_grads`; and `jit`/`trace_graph` —
  JIT compilation via Tolk will return as a transformation in a later
  release.

### Kaun

- The MNIST CNN example now saves and restores model parameters with both
  AdamW moment trees and the step counter, demonstrating how to resume
  momentum-based optimization without resetting its history.
- The `kaun` bench suite is broader: alongside the MLP Adam train step and
  forward pass it now covers an SGD train step, a small CNN train step
  (conv + max-pool blocks with `Conv`/`Pool`), and a single `Linear` layer
  forward and forward+backward in isolation.
- `Dropout.apply` takes an optional `?key:Rune.Rng.key`: the mask becomes a
  pure function of the key and the input's shape, so dropout composes with
  `Rune.jit` (pass the key as an input leaf; keyless dropout under jit
  raises `Jit_error`) and with `vmap` via per-lane keys.
- The gpt2 example trains stochastically: `Gpt2.logits` takes
  `?dropout:(rate, key)` enabling the canonical dropout sites, and
  `train.exe` gains `--dropout` and `--seed`, deriving per-step mask keys
  with `Rune.Rng.fold_in` for seed-reproducible runs.
- The gpt2 example is dtype-generic: `main.exe --dtype float16|bfloat16`
  for half-precision generation (float16 greedy tokens match float32 at
  half the weight memory), `train.exe --compute-dtype bfloat16|float16` for
  mixed-precision training with float32 master weights (bfloat16 engages
  tensor cores).
- Add `astype` to every layer (`Linear`, `Embedding`, `Conv`, `Layer_norm`,
  `Attention` and its `Cache`, `Batch_norm` and its `Stats`): cast parameter
  trees to another float dtype; gradients flow back at each leaf's original
  dtype, so casting float32 parameters inside a loss yields float32
  gradients.
- Half-precision inputs now compute attention scores/softmax and
  layer/batch-norm statistics in float32 islands; float32 and float64
  graphs are unchanged.
- `Batch_norm` is now dtype-generic (`'b params`, `'b Stats.stats`) like the
  other layers; `Batch_norm.t` and `Stats.t` remain the float32 aliases.
- The GPT-2 training example gains `--devices` for data-parallel training
  through `Rune.pmap2` — a CPU device count (`--devices 4`) or an explicit
  tuple (`--devices CUDA:0,CUDA:1`). Parameters replicate, the batch shards
  on axis 0, gradients allreduce automatically; per-step losses match the
  single-device step within fp32 reduction order.
- `Attention.apply_cached` updates the cache with a gather instead of a
  one-hot matmul, cutting the per-step update from O(len*seq*head_dim) to
  O(len*head_dim).
- **Breaking.** The attention KV cache moved into an `Attention.Cache`
  submodule: `Attention.cache`/`map_cache`/`map2_cache`/`iter_cache` are now
  `Attention.Cache.make`/`map`/`map2`/`iter` on `'b Attention.Cache.t`.
- New GPT-2 training example (`examples/04-gpt2/train.ml`): jitted
  forward+backward+SGD via `Rune.jit2` and `Vega.sgd_step` with the tied
  `wte` LM head, exporting per-step metrics and final weights as
  safetensors.
- Add key-value cache decoding to `Attention`: `cache`, `apply_cached`, and
  `map_cache`/`map2_cache`/`iter_cache`. The cache is functional and
  addresses slots with tensor arithmetic on the position, so a single-token
  decode step compiles once under `Rune.jit`; the GPT-2 example decodes
  through it (~85x faster jitted CUDA decode).
- The GPT-2 example loads a local safetensors checkpoint when one is cached
  (`Gpt2.from_file`, `Gpt2.of_checkpoint`), tokenizes via `tokenizer.json`,
  and can compile its forward pass with `Rune.jit` on CPU or CUDA
  (`--jit DEVICE`).
- **Breaking.** Ground-up rewrite on typed parameter structures. There is
  no `Layer.t` and no `Train` driver anymore: a layer is a plain record of
  tensors with a pure `apply` function (`Linear`, `Conv`, `Embedding`,
  `Attention`, `Layer_norm`, `Batch_norm`, `Dropout`, ...), a model is a
  record of layers with a hand-written `Nx.Ptree.S` traversal, and a
  training step is code you own: `Rune.value_and_grad` composed with a
  structural Vega optimizer update. Losses, initializers, activations
  (`Fn`), data batching, metrics, checkpoints, and HuggingFace Hub
  integration (`kaun.hf`, `kaun.datasets`) are provided as plain functions
  over these records.

### Talon

- Add `to_html` and `pp_display` for rich table rendering in Quill notebooks.
  Tables display as styled HTML in the web UI and published books, and as inline
  HTML in markdown output files.
- Add `Talon.take` for selecting rows by an array of indices. Indices may repeat
  and need not be sorted.
- Fix CSV auto-detection defaulting numeric columns to float32. Parsed values go
  through `float_of_string` which produces 64-bit floats; defaulting to float32
  silently truncated precision. Now defaults to float64.

### Hugin

- Fix contour rendering. The marching squares implementation produced disconnected
  2-point line segments instead of joined polylines. Contour lines now render as
  smooth connected curves, and filled contours (`~filled:true`) produce correct
  closed polygons instead of degenerate 2-point fills.

### Quill

- Allow `quill file.md` without requiring `quill -- file.md` or `quill run file.md`.
  The CLI now detects file arguments and routes them to the default TUI command.
- Fix image Display outputs showing raw base64 text in markdown files. Images now
  render as inline `<img>` tags with data URIs, visible in any markdown viewer.
- Add `--figures-dir` flag to `quill run` for writing images to disk and
  referencing them by path instead of inlining base64 data.
- Add rich table display for Talon dataframes in liveview and published books.
- Improve table styling in the web notebook and book build with clean borders,
  monospace font, and proper header treatment.
- Resolve relative notebook paths to absolute and change into the notebook
  directory before execution, so that relative file references in code cells
  work correctly.
- Add `vega` to the default Raven packages loaded in Quill kernels.
- Remove `Quill_top.install_printer_fn`. It was unused and relied on
  `Toploop.install_printer`, which was removed in OCaml 5.5. Use
  `Quill_top.install_printer` instead.

## [1.0.0~alpha3] - 2026-03-14

This release reshapes raven's foundations. Every package received API
improvements, several were rewritten, and two new packages — nx-oxcaml and
kaun-board — were built as part of our Outreachy internships.

### Highlights

- **Unified tensor type** — `Nx.t` and `Rune.t` are now the same type.
  Downstream packages no longer need to choose between them or convert at
  boundaries. Rune is now a pure transformation library (grad, vjp, vmap)
  over standard Nx tensors.
- **nx-oxcaml** (new, Outreachy) — Pure-OCaml tensor backend using OxCaml's
  unboxed types and SIMD intrinsics. Performance approaches the C backend —
  in pure OCaml.
- **kaun-board** (new, Outreachy) — TUI dashboard for monitoring training
  runs in the terminal. Live metrics, loss curves, and system stats.
- **quill** — Rewritten from the ground up with two interfaces: a terminal UI
  with syntax highlighting and code completion, and a web frontend via
  `quill serve` with a CodeMirror 6 editor, WebSocket-based execution,
  autocompletion, and diagnostics.
- **brot** — The tokenization library formerly known as saga. Complete rewrite
  with a cleaner API. [1.3-6x faster than HuggingFace Tokenizers](packages/brot/bench/)
  on most benchmarks.
- **nx** — Redesigned backend interface, RNG with effect-based scoping.
  Einsum **8-20x** faster, matmul dispatch at BLAS parity with NumPy.

### Breaking changes

- **nx**: Redesigned backend interface with new `Nx_buffer` type. Removed
  `nx.datasets` library. Moved NN functions to Kaun (use `Kaun.Fn`). Renamed
  `im2col`/`col2im` to `extract_patches`/`combine_patches`. RNG uses
  effect-based implicit scoping instead of explicit key threading. Removed
  in-place mutation operations (`ifill`, `iadd`, `isub`, `imul`, `idiv`,
  `ipow`, `imod`, `imaximum`, `iminimum` and `_s` variants). Removed
  `Symbolic_shape` module; shapes are concrete `int array` throughout.
  Removed `Instrumentation` module.
- **rune**: `Rune.t` no longer exists — use `Nx.t` everywhere. `Rune` no
  longer re-exports tensor operations; use `open Nx` for tensor ops and
  `Rune.grad`, `Rune.vjp`, etc. for autodiff. Remove any `Rune.to_nx` /
  `Rune.of_nx` calls. Removed `enable_debug`, `disable_debug`, `with_debug`;
  use `Rune.debug f x` instead.
- **rune**: Removed JIT/LLVM backend. This will come back in a future
  release with a proper ML compiler.
- **kaun**: Rewritten core modules API, datasets, and HuggingFace integration.
  Removed `kaun-models`.
- **brot**: Renamed from saga. Rewritten API focused on tokenization.

### Nx

- Unify `Nx.t` and `Rune.t` into a single tensor type. A new `nx.effect` library (`Nx_effect`) implements the backend interface with OCaml 5 effects: each operation raises an effect that autodiff/vmap/debug handlers can intercept, falling back to the C backend when unhandled. `Nx.t` is now `Nx_effect.t` everywhere — no more type conversions between Nx and Rune.
- Make transcendental, trigonometric, and hyperbolic operations (`exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, `sigmoid`) polymorphic over all numeric types including complex, matching the backend and effect definitions.
- Make `isinf`, `isfinite`, `ceil`, `floor`, `round` polymorphic (non-float dtypes return all-false/all-true or no-op as appropriate).
- Redesign backend interface with more granular operations (e.g. dedicated unary and binary kernels). This improves performance by letting backends optimize individual ops directly, and prepares for the JIT pipeline which will decompose composite operations at the compiler level instead of the frontend.
- Rewrite `Nx_buffer` module with new interface. The backend now returns `Nx_buffer.t` instead of raw bigarrays.
- Add new C kernels for unary, binary, and sort operations, and route new backend ops to C kernels.
- Add scipy-style `correlate`, `convolve`, and sliding window filters.
- Generalize `unfold`/`fold` to arbitrary leading dimensions.
- Remove neural-network functions from Nx (softmax, log_softmax, relu, gelu, silu, sigmoid, tanh). These now live in `Kaun.Fn`.
- Rename `im2col`/`col2im` to `extract_patches`/`combine_patches`.
- Remove `nx.datasets` module. Datasets are now in `kaun.datasets`.
- Simplify `Nx_io` interface. Inline vendor libraries (safetensors, and npy) directly into nx_io.
- Move the `Rng` module from Rune into Nx with effect-based implicit scoping. Random number generation uses `Nx.Rng.run` to scope RNG state instead of explicit key threading.
- Reduce matmul dispatch overhead to reach BLAS parity with NumPy.
- Fix Threefry2x32 to match the Random123 standard.
- Fix `save_image` crash on multi-dimensional genarray.
- Pre-reduce independent axes in einsum to avoid OOM on large contractions.
- Make Nx backends pluggable via Dune virtual libraries. The new `nx.backend` virtual library defines the backend interface, with the C backend (`nx.c`) as the default implementation. Alternative backends (e.g., `nx-oxcaml`) can be swapped in at link time. The `Nx_c` module is renamed to `Nx_backend`.
- Fix `.top` libraries failing to load in utop with "Reference to undefined compilation unit `Parse`".
- Fix OpenMP flag filtering in `discover.ml`: strip `-Xpreprocessor -fopenmp` as a pair on macOS to prevent dangling `-Xpreprocessor` from consuming subsequent flags and causing linker failures. (@Alizter)
- Add missing bool→low-precision cast support (f16/bf16/fp8) in the C backend.
- Add UInt32/UInt64 dtypes, rename complex dtypes to Complex64/Complex128, and drop Complex16/QInt8/QUInt8/Int/NativeInt as tensor element dtypes.
- Remove in-place mutation operations (`ifill`, `iadd`, `isub`, `imul`, `idiv`, `ipow`, `imod`, `imaximum`, `iminimum` and `_s` variants). Use functional operations instead.
- Remove `Symbolic_shape` module; shapes are now concrete `int array` throughout.
- Remove `Instrumentation` module. Nx no longer wraps operations in tracing spans. Debugging tensor operations is handled by Rune's effect-based debug handler.
- Fix critical correctness issue in fancy slicing (`L`) where permutations were ignored if the number of indices matched the dimension size (e.g., `slice [L [1; 0]] x` returned `x` unmodified).
- Rewrite `slice` implementation to use `as_strided` for contiguous operations, reducing overhead to **O(1)** for view-based slices and separating gather operations for better performance.
- Optimize `set_slice` by replacing scalar-loop index calculations with vectorized coordinate arithmetic, significantly improving performance for fancy index assignments.
- Improve `einsum` performance **8–20×** with greedy contraction path optimizer (e.g., MatMul 100×100 f32 207.83 µs → 10.76 µs, **19×**; BatchMatMul 200×200 f32 8.78 ms → 435.39 µs, **20×**)
- Rewrite `diagonal` using flatten + gather approach instead of O(N²) eye matrix masking, reducing memory from O(N²) to O(N)
- Improve error messages for shape operations (`broadcast`, `reshape`, `blit`) with per-dimension detail and element counts.

### nx-oxcaml (new)

New pure-OCaml tensor backend that can be swapped in at link time via Dune virtual libraries. Uses OxCaml's unboxed types for zero-cost tensor element access, SIMD intrinsics for vectorized kernels, and parallel matmul. Performance approaches the native C backend — in pure OCaml. Supports the full Nx operation set: elementwise, reductions, matmul, gather/scatter, sort/argsort, argmax/argmin, unfold/fold, pad, cat, associative scan, and threefry RNG. (@nirnayroy, @tmattio)

### Rune

- Unify tensor types: `Rune.t` is now `Nx.t`. Rune no longer re-exports the Nx frontend — it is a pure transformation library exporting only `grad`, `grads`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `no_grad`, `detach`, and debugging/gradcheck utilities. All tensor creation and manipulation uses `Nx` directly.
- Remove `Tensor` module and `Nx_rune` backend. Effect definitions moved to the new `nx.effect` library shared with Nx.
- Remove `Rune.to_nx` / `Rune.of_nx` (no longer needed — types are identical).
- Remove `Rune.enable_debug`, `Rune.disable_debug`, `Rune.with_debug`. Use `Rune.debug f x` to run a computation with debug logging enabled.
- Remove JIT compilation support from Rune. The `Rune.Jit` module and LLVM/Metal backends have been removed and will be re-introduced later as a standalone package.
- Update to new `Nx_buffer.t` type.
- Propagate new backend operations through effects and autodiff.
- Rewrite `Autodiff` module to fix critical JVP correctness issues, enable higher-order derivatives (nested gradients), and introduce `vjp` as a first-class primitive.
- Fix pointer-based hashing in autodiff, correcting nested JVP handler behavior.
- Add autodiff support for `as_strided`, enabling gradients through slicing and indexing operations
- Add autodiff support for `cummax` and `cummin` cumulative operations
- Add autodiff support for FFT operations
- Add autodiff support for some linear algebra operations: QR decomposition (`qr`), Cholesky decomposition (`cholesky`), and triangular solve (`triangular_solve`).

### Kaun

- Simplify and redesign the core API for better discoverability and composability. Layers, optimizers, and training utilities now follow consistent patterns and compose more naturally.
- Add `Fn` module with `conv1d`, `conv2d`, `max_pool`, `avg_pool` — neural network operations that were previously in Nx now live here with a cleaner, more focused API.
- Redesign datasets and HuggingFace integration with simpler, more composable APIs.
- Remove `kaun-models` library. Pre-built models now live in examples.
- Reinitialize dataset each epoch to avoid iterator exhaustion (#147, @Shocker444, @tmattio)

### kaun-board (new)

TUI dashboard for monitoring training runs in the terminal. Displays live metrics, loss curves, and system stats. Extracted from kaun's console module into a standalone package. (#166, #167, #170, @Arsalaan-Alam)

### Brot

- Rename the library from saga to brot.
- Simplify brot to a tokenization-only library. Remove the sampler, n-gram models, and I/O utilities. The sampler is rewritten with nx tensors and moved to `dev/mimir` as the seed of an experimental inference engine.
- Merge `brot.tokenizers` sub-library into `brot`.
- Remove dependency on Nx.
- Use `Buffer.add_substring` instead of char-by-char loop in whitespace pre-tokenizer.
- Compact BPE symbols in-place after merges, avoiding an intermediate array allocation.
- Replace list cons + reverse with forward `List.init` in BPE `word_to_tokens`.
- Use pre-allocated arrays with `Array.blit` instead of `Array.append` in encoding merge and padding, halving per-field allocations.
- Avoid allocating an unused `words` array in post-processor encoding conversion.
- Reduce WordPiece substring allocations from O(n²) to O(n) per word by building the prefixed candidate string once per position.
- Add `encode_ids` fast path that bypasses `Encoding.t` construction entirely when only token IDs are needed.
- Add ASCII property table for O(1) character classification in pre-tokenizers, replacing O(log n) binary search for `is_alphabetic` (600 ranges), `is_numeric` (230 ranges), and `is_whitespace` (10 ranges). Yields 12-27% speedup on encode benchmarks with ~30% allocation reduction.
- Add inline ASCII fast paths in all pre-tokenizer loops, skipping UTF-8 decoding and using `Buffer.add_char` instead of `String.sub` for single-byte characters. Combined with the property table, yields 20-30% total speedup and 36-55% allocation reduction vs baseline.
- Parallelize batch encoding with OCaml 5 domains.
- Optimize BPE merge loop with open-addressing hash, flat arrays, and shift-based heap.
- Add trie-based WordPiece lookup and normalizer fast path.
- Remove dependency on `str` library.
- Generate unicode data offline, removing runtime dependency on `uucp`.
- Remove unused `Grapheme` module. Grapheme cluster segmentation is not needed for tokenization.
- Remove `uutf` dependency in favour of OCaml `Stdlib` unicode support.

### Fehu

- Simplify and redesign the core API. Environments and training utilities now follow consistent functional patterns that are easier to use and compose.
- Remove `fehu.algorithms` — fehu now only depends on rune, and users bring their own algorithms. Examples provided for well-known RL algorithms like DQN and REINFORCE.

### Sowilo

- Cleaner public API — internal implementation split into focused submodules while the public surface stays small.
- Faster grayscale conversion, edge detection, and gaussian blur.

### Quill

Rewritten from the ground up. Terminal UI with syntax highlighting, code completion, and a compact single-line footer. Web frontend via `quill serve` with a CodeMirror 6 editor, WebSocket-based execution, autocompletion, and diagnostics. Markdown notebook format shared across both interfaces.

Interactive REPL: `quill` with no file argument launches a toplevel with syntax highlighting, tab completion, persistent history, smart phrase-aware submission, and piped mode.

### Hugin

Rewritten from the ground up with a declarative, composable API. Plots are
built by combining inert mark descriptions (`line`, `point`, `bar`, `hist`,
`heatmap`, `contour`, `errorbar`, etc.) with `layers`, decorating them
(`title`, `xlabel`, `legend`, etc.), and laying them out (`grid`, `hstack`,
`vstack`). A compilation pass resolves data to a Scene IR that separate
backends render.

- New declarative specification API replacing the imperative figure/axes/artist
  architecture. Marks compose with `layers`, decorations chain functionally,
  and grid layouts nest arbitrarily.
- **ucairo** — Minimal Cairo FFI bindings (36 C stubs) replacing the `cairo2`
  opam dependency.
- Dual-backend rendering: Cairo (PNG, PDF, interactive SDL window) and SVG from
  a shared Scene IR.
- OKLCH perceptual color space with `Color.oklch`, `Color.hex`, named CSS
  colors, and alpha support.
- Curated colormaps (`Cmap.viridis`, `plasma`, `inferno`, `magma`, `cividis`,
  `turbo`, `coolwarm`, `spectral`).
- Theme system with `light`, `dark`, and `minimal` presets.
- Linear, log, and symlog axis scaling with automatic tick generation.
- Legend placement with configurable location and multi-column layout.
- Interactive `show` with SDL window resizing, Escape/Q to close.
- Rewritten examples and documentation.

### Talon

- Remove `jsont`, `bytesrw`, and `csv` dependencies from Talon. CSV support is now built-in via the `talon.csv` sub-library with a minimal RFC 4180 parser.
- Remove `talon.json` sub-library.

## [1.0.0~alpha2] - 2025-11-03

We're excited to announce the release of Raven 1.0.0~alpha2! Less than a month after alpha1, this release notably includes contributions from Outreachy applicants in preparation for the upcoming _two_ internships.

Some highlights from this release include:

- NumPy-compatible text I/O with `Nx_io.{save,load}_text`
- Lots of new functions in Nx/Rune, including neural-net ones `dropout`, `log_softmax`, `batch_norm`, `layer_norm`, and activation functions like `celu` and `celu`, and generic ones like `conjugate`, `index_put`, and more.
- Addition of `.top` libraries for `nx`, `rune`, and `hugin` that auto-install pretty-printers in the OCaml toplevel. You can run e.g. `#require "nx.top"`.
- Addition of a visualization API in Fehu via the new `fehu.visualize` library, supporting video recording.
- Redesign of Kaun core datastructure and checkpointing subsystem for complete snapshotting.
- Many, many bug fixes and correctness improvements.

We've also made numerous performance improvements across the board:

- Nx elementwise ops: 5–50× faster (e.g., Add 50×50 f32 88.81 µs → 1.83 µs, **48×**; Mul 100×100 f32 78.51 µs → 2.41 µs, **33×**).
- Nx conv2d: **4–5×** faster on common shapes; up to **115×** on heavy f64 batched cases (e.g., B16 C64→128 16×16 K3 f64 1.61 s → 13.96 ms).
- Rune autodiff: **1.2–3.7×** faster on core grads (e.g., MatMulGrad Medium 34.04 ms → 11.91 ms, **2.86×**; Large 190.19 ms → 50.97 ms, **3.73×**).
- Talon dataframes: big wins in joins and group-bys (Join 805.35 ms → 26.10 ms, **31×**; Group-by 170.80 ms → 19.03 ms, **9×**; Filter 9.93 ms → 3.39 ms, **3×**).
- Brot tokenizers: realistic workloads **4–17%** faster (e.g., WordPiece encode single 136.05 µs → 115.92 µs, **1.17×**; BPE batch_32 24.52 ms → 22.27 ms, **1.10×**)

We're closing 8 user-reported issues or feature requests and are totalling 30 community contributions from 8 unique contributors.

### Nx

- Fix einsum output axis ordering for free axes (e.g., `i,jk->jki`, `ij,klj->kli`) by correcting final transpose permutation and intermediate left-axis reordering.
- Add `Nx_io.Cache_dir` module with consolidated cache directory utilities respecting `RAVEN_CACHE_ROOT`, `XDG_CACHE_HOME`, and `HOME` fallback, replacing project-specific cache logic across the whole raven ecosystem (#134, @Arsalaan-Alam)
- Add `Nx_io.save_txt` / `Nx_io.load_txt` with NumPy-compatible formatting, comments, and dtype support (#120, @six-shot)
- Optimize `multi_dot` for matrix chains, reducing intermediate allocations and improving performance
- Add public `index_put` function for indexed updates
- Clarify `reshape` documentation to match its view-only semantics
- Provide `nx.top`, `rune.top`, and `hugin.top` libraries that auto-install pretty printers in the OCaml toplevel and update Quill to load them
- Add `ifill` for explicit in-place fills and make `fill` return a copied tensor
- Speed up contiguous elementwise ops via vectorized loops
- Fast-path contiguous single-axis reductions to avoid iterator fallback
- Speed up float reductions with contiguous multi-axis fast paths
- Fast-path padding-free `unfold` to lower conv2d overhead
- Move neural-network operations (softmax, log_softmax, relu, gelu, silu, sigmoid, tanh) from Kaun to Nx
- Add public `conjugate` function for complex number conjugation (#125, @Arsalaan-Alam)
- Fix complex vdot to conjugate first tensor before multiplication, ensuring correct mathematical behavior (#123, @Arsalaan-Alam)
- Update comparison and conditional operations to use boolean tensors (#115, @nirnayroy)
- Add support for rcond parameter and underdetermined systems to `lstsq` (#102, @Shocker444)
- Fix `matrix_rank`/`pinv` Hermitian fast paths to use eigen-decomposition and match NumPy for complex inputs (#96, @six-shot, @tmattio)
- Optimize matmul BLAS dispatch for strided tensors, improving matrix multiplication performance
- Fix slow builds reported since alpha1 (#88, @tmattio)
- Fix macOS ARM crash when loading extended bigarray kinds
- Add float16 and bfloat16 support to safetensors I/O, including precise conversions that preserve denormals/NaNs (#84, @six-shot, @tmattio)
- Refined `View` internals for leaner contiguity checks and stride handling, cutting redundant materialization on hot paths
- Merge `Lazy_view` into the core `View` API so movement ops operate on a single composed view
- Documented the reworked `View` interface
- Documented the `Symbolic_shape` interface
- Added Accelerate framework flag when compiling on macOS, fixing issues in some environments (#129, @nirnayroy)

### Hugin

- Fix random `SIGBUS`/bus errors on macOS when closing `Hugin.show` windows by
  destroying SDL windows with the correct pointer in the finalizer.
- Let `Hugin.show` windows close cleanly via the window button or `Esc`/`q`, avoiding frozen macOS REPL sessions

### Rune

- Add `Rune.no_grad` and `Rune.detach` to mirror JAX stop-gradient semantics
- Improve gradient performance slightly by replace the reverse-mode tape's linear PhysicalTbl with an identity hash table
- Fix `Rune.Rng.shuffle` flattening outputs for multi-dimensional tensors; the
  shuffle now gathers along axis 0 and keeps shapes intact
- Replace `Rune.Rng.truncated_normal` clipping with rejection sampling so
  samples stay inside the requested interval without boundary spikes
- Add support for categorical sampling with `Rune.Rng.categorical` (#89, @nirnayroy)
- Allow plain `llvm-config` in discovery, fixing build in some platforms (#71, @stepbrobd)

### Kaun

- Added Similarity and Polysemy analysis to the BERT example (#137, @nirnayroy)
- Support attention masks via the new `Kaun.Attention` module
- Support loading sharded Hugging Face safetensors
- Fix BERT and GPT‑2 model loading
- API simplification: removed type parameters from public types; `Ptree` now supports mixed‑dtype trees via packed tensors with typed getters.
- Checkpointing overhaul: versioned `Train_state` with schema tagging, explicit `Checkpoint.{Snapshot,Artifact,Manifest,Repository}` (retention, tags, metadata), and simple save/load helpers for snapshots and params.
- Overhaul dataset combinators: derive tensor specs from Rune dtype, fix sampling/window bugs, validate weighted sampling, and respect `drop_remainder`
- Make dataset `prefetch` truly asynchronous with background domains and allow reusing an external Domainslib pool via `parallel_map ~pool`
- Use `Dataset.iter` for epoch batches to reduce overhead
- Update BERT and GPT-2 tokenizer cache to use `Nx.Cache` for consistent cache directory resolution (#134, @Arsalaan-Alam)
- Honor text dataset encodings via incremental Uutf decoding (#122, @Satarupa22-SD).
- Preserve empty sequential modules when unflattening so indices stay aligned for checkpoint round-tripping
- Prevent `Training.fit`/`evaluate` from consuming entire datasets eagerly and fail fast when a dataset yields no batches, avoiding hangs and division-by-zero crashes
- Allow metric history to tolerate metrics that appear or disappear between epochs so dynamic metric sets no longer raise during training
- Make `Optimizer.clip_by_global_norm` robust to zero gradients and empty parameter trees to avoid NaNs during training
- Split CSV loader into `from_csv` and `from_csv_with_labels` to retain labels when requested (#114, @Satarupa22-SD)
- Implement AUC-ROC and AUC-PR in Kaun metrics and simplify their signatures (#124, #131, @Shocker444)
- Add mean absolute percentage error, explained variance, R² (with optional adjustment), KL-divergence, and top-k accuracy to Kaun metrics
- Add NDCG, MAP, and MRR ranking metrics to Kaun metrics
- Add BLEU, ROUGE, and METEOR metrics to Kaun for pre-tokenized sequences, removing tokenizer dependencies
- Add SSIM, IoU, and Dice metrics for vision workloads in Kaun

### Talon

- Remove automatic sentinel-based null detection for numeric columns; explicit masks (via [_opt] constructors) now define missing data semantics
- Replace join nested loops with hashed join indices, cutting lookup from O(n·m) to near O(n)
- Reuse a shared Nx-based column reindexer so filter/sample paths avoid repeated array copies
- Fix `fillna` to honor column null masks and replacements, restoring expected nullable semantics
- Preserve null masks when reindexing during joins so sentinel values remain valid data
- Handle numeric index columns in `pivot`, preventing distinct keys from collapsing into a single bucket
- Respect null masks when serializing numeric columns to JSON, emitting JSON `null` instead of sentinel values
- Detect big integers as int64 in Talon CSV loader (#121, @Arsalaan-Alam)
- Allow forcing column types in Talon JSON loader (#104, @nirnayroy)
- Add documentation to compare Talon and Pandas (#154, Satarupa22-SD)

### Saga

- Remove legacy `Normalizers.nmt` and `Normalizers.precompiled` constructors (and their JSON serializers) so the public surface only advertises supported normalizers
- Tighten template processor JSON parsing: require integer type ids, drop the legacy special-token list format, and ensure multi-id special tokens round-trip with the new record fields
- Make tokenizer JSON loading tolerant of HuggingFace quirks (missing `model.type`, string-encoded merges), restoring compatibility with upstream `tokenizer.json` files
- Cache byte-level encode/decode lookup tables to avoid rebuilding them during tokenization, trimming avoidable allocations
- Skip BPE dropout sampling when dropout is disabled, removing redundant RNG work on common hot paths
- Fix Unigram tokenization so longest matches are emitted without aborting the sequence when a vocab hit occurs
- Recompute pad token ids when the pad special string changes, preventing padding with stale ids
- Fix Unigram `token_to_id`/`id_to_token` vocabulary lookups (#117, @RidwanAdebosin)
- Optimize `Pre_tokenizers.whitespace` to reduce allocations and improve tokenization performance
- Simplify tokenizers interface

### Sowilo

- Add `resize` (nearest & bilinear) that works for 2D, batched, and NHWC tensors
- Update grayscale conversion and RGB/BGR channel swaps to run entirely on Rune ops, keeping batched inputs compatible with JIT backends
- Make `median_blur` compute the true median so salt-and-pepper noise is removed as expected
- Fix `erode`/`dilate` so custom structuring elements (e.g. cross vs. square) and batched tensors produce the correct morphology result

### Fehu

- Added snapshot-based save/load for DQN and REINFORCE agents (#127, @RidwanAdebosin, @tmattio)
- Added typed `Render` payloads with enforced `render_mode` selection in `Env.create`, auto human-mode rendering, and vectorized `Env.render` accessors so environments consistently expose frames for downstream tooling
- Introduced the `Fehu_visualize` library with ffmpeg/gif/W&B sinks, overlay combinators, rollout/evaluation recorders, and video wrappers for single and vectorized environments, providing a cohesive visualization stack for Fehu
- Added a `Fehu.Policy` helper module (random/deterministic/greedy) and sink `with_*` guards so visualization sinks handle directory creation and cleanup automatically
- Added `Buffer.Replay.sample_tensors` to streamline batched training loops and exploration handling
- Reworked `Fehu_algorithms.Dqn` around `init`/`step`/`train` primitives with functional state, warmup control, and snapshotting helpers
- Rebuilt `Fehu_algorithms.Reinforce` on the same `init`/`step`/`train` interface with optional baselines, tensor-based rollouts, snapshot save/load, and updated tests/examples/docs using the new workflow
- Upgraded the GridWorld environment to return ANSI and RGB-array frames using the new render types, and updated the DQN example to optionally record pre- and post-training rollouts via `FEHU_DQN_RECORD_DIR` using `Fehu_visualize` sinks
- Reworked space sampling to return `(value, next_rng)` and split keys internally, fixing correlated draws in Box/Multi-discrete/Tuple/Dict/Sequence/Text samplers while adding `Space.boundary_values` for deterministic compatibility checks
- Extended vectorized environments to reuse space boundary probes and now store structured `final_observation` payloads in `Info`, improving downstream consumption
- Added `Buffer.Replay.add_many` and `Buffer.Replay.sample_arrays`, preserved backing storage on `clear`, and exposed struct-of-arrays batches for vectorised learners
- Tightened `Env.create` diagnostics with contextual error messages and an optional `~validate_transition` hook for custom invariants
- Enriched `Wrapper` utilities with `map_info`, Box `clip_action`/`clip_observation`, and time-limit info reporting elapsed steps
- Upgraded `Info` values to carry int/float/bool arrays with stable JSON round-tripping (handling NaN/∞) and sorted metadata serialization for deterministic diffs
- Improved training helpers: Welford-based normalization with optional unbiased variance, documented `done = terminated || truncated`, and returned `nan` when explained variance is undefined
- Treat time-limit truncations as terminals when computing rollout advantages and expose the `truncated` flag in buffer steps
- Require callers of `Training.compute_gae` to pass final bootstrapping values and ensure `Training.evaluate` feeds the current observation to policies
- Allow `Space.Sequence.create` to omit `max_length`, keeping sequences unbounded above while preserving validation and sampling semantics
- Validate vectorized environments by round-tripping sample actions/observations across every instance, preventing incompatible spaces from slipping through
- Finish clipped value loss support in Fehu.Training (#119, @nirnayroy)

### Nx-datasets

- Migrate to `Nx.Cache` for cache directory resolution, enabling consistent behavior. (#133, @Arsalaan-Alam)
- Fix cache directory resolution to respect `RAVEN_CACHE_ROOT` (or fall back to `XDG_CACHE_HOME`/`HOME`), allowing custom cache locations. (#128, @Arsalaan-Alam)
- Switch CIFAR-10 loader to the binary archive so parsing succeeds again
- Add a CIFAR-10 example
- Standardize dataset examples on `Logs`
- Use `Logs` for dataset loader logging (#95, @Satarupa22-SD)

## [1.0.0~alpha1] - 2025-10-02

This release expands the Raven ecosystem with three new libraries (Talon, Saga, Fehu) and significant enhancements to existing ones. `alpha1` focuses on breadth—adding foundational capabilities across data processing, NLP, and reinforcement learning—while continuing to iterate on core infrastructure.

### New Libraries

#### Talon - DataFrame Processing
We've added Talon, a new DataFrame library inspired by pandas and polars:
- Columnar data structures that support mixed types (integers, floats, strings, etc.) within a single table (aka heterogeneous datasets)
- Operations: filter rows, group by columns, join tables, compute aggregates
- Load and save data in CSV and JSON formats
- Seamless conversion to/from Nx arrays for numerical operations

#### Saga - NLP & Text Processing
Saga is a new text processing library for building language models. It provides:
- Tokenizers: Byte-pair encoding (BPE), WordPiece subword tokenization, and character-level splitting
- Text generation: Control output with temperature scaling, top-k filtering, nucleus (top-p) sampling, and custom sampling strategies
- Language models: Train and generate text with statistical n-gram models (bigrams, trigrams, etc.)
- I/O: Read large text files line-by-line and batch-process corpora

#### Fehu - Reinforcement Learning
Fehu brings reinforcement learning to Raven, with an API inspired by Gymnasium and Stable-Baselines3:
- Standard RL environment interface (reset, step, render) with example environments like Random Walk and CartPole
- Environment wrappers to modify observations, rewards, or episode termination conditions
- Vectorized environments to collect experience from multiple parallel rollouts
- Training utilities: Generalized advantage estimation (GAE), trajectory collection and management
- RL algorithms: Policy gradient method (REINFORCE), deep Q-learning (DQN) with replay buffer
- Use Kaun neural networks as function approximators for policies and value functions

### Major Enhancements

#### Nx - Array Computing
We've significantly expanded Nx's following early user feedback from alpha0:
- Complete linear algebra suite: LAPACK-backed operations matching NumPy including singular value decomposition (SVD), QR factorization, Cholesky decomposition, eigenvalue/eigenvector computation, matrix inverse, and solving linear systems
- FFT operations: Fast Fourier transforms (FFT/IFFT) for frequency domain analysis and signal processing
- Advanced operations: Einstein summation notation (`einsum`) for complex tensor operations, extract/construct diagonal matrices (`diag`), cumulative sums and products along axes
- Extended dtypes: Machine learning-focused types including bfloat16 (brain floating point), complex16, and float8 for reduced-precision training
- Symbolic shapes: Internal infrastructure for symbolic shape inference to enable dynamic shapes in future releases (not yet exposed in public API)
- Lazy views: Array views only copy and reorder memory when stride patterns require it, avoiding unnecessary allocations

#### Rune - Autodiff & JIT
We've continued iterating on Rune's autodiff capabilities, and made progress on upcoming features:
- Forward-mode AD: Compute Jacobian-vector products (`jvp`) for forward-mode automatic differentiation, complementing existing reverse-mode
- JIT: Ongoing development of LLVM-based just-in-time compilation for Rune computations (currently in prototype stage)
- vmap: Experimental support for vectorized mapping to automatically batch operations (work-in-progress, not yet stable)
- LLVM backend: Added compilation backend with support for LLVM versions 19, 20, and 21
- Metal backend: Continued work on GPU acceleration for macOS using Metal compute shaders

#### Kaun - Deep Learning
We've expanded Kaun with high-level APIs for deep learning. These APIs are inspired by popular Python frameworks like TensorFlow, PyTorch, and Flax, and should feel familiar to users building models in Python:
- High-level training: Keras-style `fit()` function to train models with automatic batching, gradient computation, and parameter updates
- Training state: Encapsulated training state (TrainState) holding parameters, optimizer state, and step count; automatic history tracking of loss and metrics
- Checkpoints: Save and load model weights to disk for model persistence and transfer learning
- Metrics: Automatic metric computation during training including accuracy, precision, recall, F1 score, mean absolute error (MAE), and mean squared error (MSE)
- Data pipeline: Composable dataset operations (map, filter, batch, shuffle, cache) inspired by TensorFlow's `tf.data` for building input pipelines
- Model zoo: Reference implementations of classic and modern architectures (LeNet5 for basic CNNs, BERT for masked language modeling, GPT2 for autoregressive generation) including reusable transformer components
- Ecosystem integration: Load HuggingFace model architectures (`kaun.huggingface`), access common datasets like MNIST and CIFAR-10 (`kaun.datasets`), and use standardized model definitions (`kaun.models`)

### Contributors

Thanks to everyone who contributed to this release:

- @adamchol (Adam Cholewi) - Implemented the initial `associative_scan` native backend operation for cumulative operations
- @akshay-gulab (Akshay Gulabrao)
- @dhruvmakwana (Dhruv Makwana) - Implemented `einsum` for Einstein summation notation
- @gabyfle (Gabriel Santamaria) - Built PocketFFT bindings that replaced our custom FFT kernels
- @lukstafi (Lukasz Stafiniak) - Major contributions to Fehu and FunOCaml workshop on training Sokoban agents
- @nickbetteridge
- @sidkshatriya (Sidharth Kshatriya)

## [1.0.0~alpha0] - 2025-07-05

### Initial Alpha Release

We're excited to release the zeroth alpha of Raven, an OCaml machine learning ecosystem bringing modern scientific computing to OCaml.

### Added

#### Core Libraries

- **Nx** - N-dimensional array library with NumPy-like API
  - Multi-dimensional tensors with support for several data types.
  - Zero-copy operations: slicing, reshaping, broadcasting
  - Element-wise and linear algebra operations
  - Swappable backends: Native OCaml, C, Metal
  - I/O support for images (PNG, JPEG) and NumPy files (.npy, .npz)

- **Hugin** - Publication-quality plotting library
  - 2D plots: line, scatter, bar, histogram, step, error bars, fill-between
  - 3D plots: line3d, scatter3d
  - Image visualization: imshow, matshow
  - Contour plots with customizable levels
  - Text annotations and legends

- **Quill** - Interactive notebook environment
  - Markdown-based notebooks with live formatting
  - OCaml code execution with persistent session state
  - Integrated data visualization via Hugin
  - Web server mode for browser-based editing

#### ML/AI Components

- **Rune** - Automatic differentiation and JIT compilation framework
  - Reverse-mode automatic differentiation
  - Functional API for pure computations
  - Basic JIT infrastructure (in development)

- **Kaun** - Deep learning framework (experimental)
  - Flax-inspired functional API
  - Basic neural network components
  - Example implementations for XOR and MNIST

- **Sowilo** - Computer vision library
  - Image manipulation: flip, crop, color conversions
  - Filtering: gaussian_blur, median_blur
  - Morphological operations and edge detection

#### Supporting Libraries

- **Nx-datasets** - Common ML datasets (MNIST, Iris, California Housing)
- **Nx-text** - Text processing and tokenization utilities

### Known Issues

This is an alpha release with several limitations:
- Quill editor has UI bugs being addressed
- APIs may change significantly before stable release

### Contributors

Initial development by the Raven team. Special thanks to all early testers and contributors.

@axrwl
@gabyfle
@hesterjeng
@ghennequin
@blueavee

And to our early sponsors:

@daemonfire300
@gabyfle
@sabine

[1.0.0~alpha0]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha0
[1.0.0~alpha1]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha1
[1.0.0~alpha2]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha2
