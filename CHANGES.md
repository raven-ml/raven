# Changelog

All notable changes to this project will be documented in this file.

- Only document user-facing changes (features, bug fixes, performance improvements, API changes, etc.)
- Add new entries at the top of the appropriate section (most recent first)

## [1.0.0~beta1] - Unreleased

### General

- `fehu`, `sowilo`, `norn`, and `nx-oxcaml` move to `contrib/`. Each is its own
  dune project with its own version, builds against `main`, and sits outside
  the 1.0 API commitment. Changes to `fehu`, `sowilo`, and `norn` are now
  recorded in `contrib/<package>/CHANGES.md`. The `raven` package no longer
  installs `fehu` and `sowilo`; install them by name.
- Add compatibility with OCaml 5.5.

### Hugin

- Marks and `Hugin_vg.Picture.image` read a placed value to the host once, as
  they are made. A placed image raised in the raster and PDF backends.
- Hugin no longer depends on cairo or SDL2. Plots render through
  `hugin.vg` with the bundled Inter font, so `opam install hugin` needs no
  system libraries and a specification produces the same bytes on every
  machine. PDF output keeps text as text with the font embedded, and SVG
  output embeds the font too.
- Remove `Hugin.show`. There is no interactive window any more; view plots
  in Quill or render them to a file.
- `Theme.font` loses its `family` field: the bundled font is used, and
  `weight` picks its regular or bold face.
- Fix y-axis labels, which read top to bottom; they now read bottom to top.
- Add `hugin.vg`, a 2D vector picture model with no system dependencies,
  and one library per renderer. `Hugin_vg.Path` builds paths from lines,
  cubic Béziers and bulk polylines, `Affine` composes transforms, and
  `Stroke` and `Color` describe how paths are drawn. `Font` bundles two
  faces of Inter and reads any TrueType font with `Font.of_string`; it
  measures text with kerning (`advance`, `bounds`, `ascent`, `descent`) and
  turns it into paths (`outline`, `glyph_path`), so text looks the same on
  every machine. `Picture` is the display list: fills with nonzero or
  even-odd rules, strokes with caps, joins and dashes, text, Nx images,
  groups, path clips, affine transforms, and `stamp` for drawing one
  picture at many points. `Picture.bounds`, `Path.bounds` and
  `Font.bounds` give the `Box` a drawing occupies, so aligning any picture
  is one `transform`, `Path.flatten` folds over a path with its curves
  replaced by chords, and `Picture.pp` prints a picture for tests and
  debugging. `hugin.vg.raster` draws a picture into an
  `[|height; width; 4|]` RGBA tensor with analytic antialiasing;
  `hugin.vg.svg` and `hugin.vg.pdf` write it as a self-contained SVG or
  single-page PDF document, with text kept as text, the fonts it uses
  embedded, images stored losslessly and every PDF stream deflated.

### Vega

- Add L-BFGS to the structural tier for deterministic objectives: full-batch
  fits, MAP estimates, calibration, the second stage of training a PINN.
  `Vega.minimize (module P) f params` runs it from `params` until a gradient
  or value tolerance is met and returns the final state with a `status`;
  `lbfgs_init` and `lbfgs_step` are the step it loops, `Vega.Lbfgs_state (P)
  (V)` the state's `Nx.Ptree.S`. The objective returns its value and gradient
  at once, the type `Rune.value_and_grad (module P) loss` has, so an analytic
  gradient serves as well. Without `~lr` a step chooses its length by a
  strong-Wolfe line search (eager); with `~lr` it preconditions a fixed rate
  and traces under `Rune.jit`. Every scalar the method keeps is at the
  objective's dtype, so a `float64` objective drives a `float64` search.
- Add `Vega.global_dot (module P) dt a b`, the inner product of two parameter
  trees over all their float leaves as a scalar tensor accumulated at `dt`,
  in tensor arithmetic so it traces under `Rune.jit`.
- Optimizer state now compiles. The structural states are parameter trees —
  `Vega.Sgd_state (P)` and `Vega.Adam_state (P)` are the `Nx.Ptree.S` for the
  state over a parameter tree `P` — so the state is one field of a
  `Rune.jit2`/`pmap2` step's input and output records (whose traversals
  delegate to it, by hand or via `ppx_ptree`), threaded across compiled
  calls as ordinary leaves. The leaf order (payload leaves, then the counter) is part
  of a compiled step's leaf signature and is fixed.
- **Breaking:** `sgd_state` is `{ velocity; step }` — every structural state
  now carries its step counter, a scalar `int32` tensor, so a schedule
  applies to `st.step` whichever optimizer is stepping; without it an SGD
  loop under `jit` had to thread a counter of its own.
- **Breaking:** `adam_state` is `{ mu; nu; step }` — `step` is a scalar
  `int32` tensor, the number of completed steps. A host `int` counter burned
  the Adam bias corrections into the trace at compile time and replayed them
  stale on every later call; the tensor counter tracks correctly under `jit`,
  and the corrections `1 - b^t` are derived from it inside each step, per
  leaf at the leaf's dtype (`float64` parameters keep their exact analytic
  corrections). The state carries nothing the counter does not determine, so
  checkpoints hold the moments plus one scalar.
- **Breaking:** the structural step functions take the learning rate as a
  scalar tensor: `~lr:(float, 'b) Nx.t`, cast to each leaf's dtype. `Vega.lr
  v` is the constant-rate helper (`Nx.scalar Nx.float32 v`; any float dtype
  is accepted, so `float64` loops can pass a `float64` rate); a scheduled
  rate is the schedule applied to the state's `step` leaf. The per-element
  arithmetic is otherwise unchanged.
- **Breaking:** `Schedule.t` is now a function from a scalar `int32` step
  tensor to a scalar `float32` rate tensor — pure tensor arithmetic, so one
  schedule family serves eager loops and compiled steps alike, every
  schedule included (`exponential_decay`, `polynomial_decay` and
  `cosine_decay_restarts` too). `Schedule.eval` reads a schedule at a host
  `int` for logging and eager loops that keep their own count; the
  per-tensor tier (`scale_by_schedule`, `scale_by_learning_rate`,
  `add_decayed_weights`, the aliases) is unchanged at its call sites.
  `cosine_decay_restarts` now validates `t_mul >= 1` and `m_mul > 0`, and
  `exponential_decay` validates `decay_rate > 0`.
- `clip_by_global_norm` computes its scale factor in `float32` tensor
  arithmetic and selects it with `Nx.where` — no host read — so it traces
  under `jit` on any device and can sit between a jitted backward pass and a
  jitted optimizer step. `global_norm` remains the `float64` host read for
  reporting.

### Rune

- Compiled `Nx_quant.apply` runs tolk's kernel, which decodes MXFP4 weights in
  registers, when a matrix meets few rows (32 on Metal and 64 on the CPU, at
  bfloat16). On an M1 Max, gpt-oss-20b decodes a token in 48-54 ms against
  122-126 ms, 8 sequences in 100 ms against 446 ms, and a 512-token prompt in
  3.5 s against 8.8 s.
- `Nx_quant.apply` and `Nx_quant.dequant` compile under `Rune.jit`, and `grad`,
  `jvp`, `vmap` and `with_debug` take them: gradients flow to `x` only, and a
  quantised weight whose part is differentiated raises.
- Fix compiled `int64` and `uint64` constants beyond 2^62: `Rune.jit` read
  them through OCaml's 63-bit `int`, so `Int64.min_int` became 0 and
  `0x4000000000000000L` became `-2^62`. They now keep every bit.
- Fix reading a strided view of a placed value (`Nx.to_array`, `Nx.copy`, or
  an eager operation on `Nx.transpose` of it): the elements were copied
  through OCaml values, which quieted signalling NaNs. They are now copied as
  bits.
- Compiled `Nx.sort` and `Nx.argsort` put NaN after every number in either
  direction, as eager ones do. A NaN dropped out of the sorted values, with
  another value repeated in its place, and the positions around it were wrong.
- Compiled `Nx.cummax` and `Nx.cummin` are NaN from the first NaN on, as
  eager ones are. They kept the running maximum or minimum past a NaN.
- **Breaking:** Remove `Rune.to_device`. Place values with
  `Nx.place (Nx.Placement.device (Rune.device "METAL")) x`; on the host,
  `Nx.place` returns its argument where `to_device` made it contiguous.
- A placed leaf or capture that views part of its storage (a slice, a
  transpose, a flip, a broadcast) is read in place by a compiled function on its
  device, where it was copied through the host on every call.
- A compiled function runs where its placed inputs and captures live, else on
  `Rune.default_device`. A leaf or capture elsewhere, or a `pmap` output, raises
  instead of going through the host; so does `float64` on Metal.
- Add `Rune.device`, `Rune.devices` and `Rune.default_device`: one
  `Nx.Device.t` per name (`"METAL"`, `"CUDA:3"`). `"CPU"` is `Nx.Device.host`,
  so placing on the host inside a host program is the identity; `"CPU:1"`,
  `"CPU:2"`... have storage of their own.
- **Breaking:** Remove `RUNE_JIT_FORCE_COPY`; compile for `"CPU:1"` to run the
  device path without a GPU. `Rune.pmap` over `"CPU"` raises.
- Reading a compiled function's output no longer moves it to the host:
  `Nx.item` on resident logits copies one element and the logits stay
  resident, and eager operations and `grad` over resident values keep their
  results on the device. `Nx.place Nx.Placement.host` makes a host copy.
- `Rune.to_device` is no longer the identity inside transformations: under
  `grad` and `jvp` it is linear and a gradient returns to its input's
  placement, under `vmap` it places the batched value, and inside `jit` it
  raises `Jit_error` unless the device is the program's.
- A `Rune.pmap` output stays on its devices when read, and an nx operation on
  it reads it and returns a host value, as before.
- Placing a value on a device that cannot hold its dtype raises
  `Invalid_argument`, at `Rune.to_device` and at an eager operation whose
  result has that dtype: `float64` on Metal, and complex and 4-bit integers on
  every rune device, so `Nx.rfft` of a placed value raises; place it on the
  host first.
- `RUNE_JIT_RESIDENT_BUDGET` counts every device allocation since the last
  major collection, eager results and uploads included, and a device that
  still cannot allocate after a collection raises `Nx.Device.Out_of_memory`.
- `Rune.jit_step` raises when a state leaf is a view of part of its storage,
  which cannot be donated; pass a copy of it.
- A compiled function whose output has no elements returns an empty tensor of
  that output's dtype and shape instead of raising "an output of the traced
  function was not scheduled to a buffer".
- `Nx.cumsum` under `Rune.jit` keeps int8 and int16 results in their dtype. The
  compiled scan left them in int32, so values came back wrong (int8
  `[100; 100; 100; 1]` gave `[100; 0; 0; 0]`) and the process could crash.
- Compiled functions on a device share the memory their intermediates need,
  sized to the largest, instead of each holding its own: gpt-oss-20b run one
  compiled layer kind at a time peaks at 17.4 GB instead of 19.4 GB.
- A `Rune.scan` staged under `Rune.jit` updates a carry in place when its step
  writes it with `Nx.set` or reads it only where it writes: a step writing one
  row of a stacked cache no longer copies the whole cache.
- `Rune.scan (module C) (module X) (module Y)` folds over structures, `scan'`
  over single tensors, and `Nx.Ptree.leaf` fills a single-tensor role. Put
  per-step data such as stacked layer weights in `xs`: `jit` reads it in place.
- A `Rune.scan` staged under `Rune.jit` replays its body as batched device
  graphs instead of launching each kernel: a 64-step scan on Metal goes from
  18.5 ms to 6.4 ms.
- A `Rune.scan` staged under `Rune.jit` no longer waits for the device on
  every iteration: the body writes what the loop used to copy. A 64-step scan
  on Metal goes from 60 ms to 18 ms.
- A compiled call on a GPU returns without waiting for its kernels; reads wait.
  Twenty-six chained small calls on Metal take 2.9 ms instead of 10.8 ms.
- Add `Rune.jit_step`: it reads its first argument and consumes and returns its
  state. `?donate` leaves `jit`, `jit2` and `jit'`: `jit2 ~donate:true` becomes
  `jit_step (module Nx.Ptree) (module S) f` applied to `Nx.Ptree.list []`.
- Tracing a function under `Rune.jit` no longer allocates a buffer for every
  traced value. Each placeholder was an uninitialised tensor of the result's
  full size; the pages were never touched, but OCaml counted the bytes and ran
  major collections throughout the trace. The first call of a gpt-oss-20b step
  goes from 47 s to 11 s. Placeholders are now `Nx_effect.Symbolic` tensors:
  dtype and shape, no bytes.
- A compiled function plans the memory of its intermediates: buffers whose
  lifetimes do not overlap share one arena per device instead of each owning an
  allocation for the life of the function. A single-token step of gpt-oss-20b
  on Metal held 11 GB of intermediates beside its 13.8 GB of weights and now
  holds 0.2 GB; no longer under memory pressure, it drops from 6.7 s to 0.42 s.
  `NO_MEMORY_PLANNER=1` turns it off.
- An upload reads a tensor over a mapped file from the file, not through the
  mapping. Copying mapped pages into device buffers is bound by the page-fault
  path once the file no longer fits in the cache beside the buffers: placing a
  13.76 GB checkpoint on Metal took 28.6 s and now takes 7.3 s. A transposed
  weight is read as the run of the file it permutes. If the path no longer
  names the file that was mapped, the mapping is read as before.
- `RUNE_JIT_RESIDENT_BUDGET` counts the outputs of compiled calls only. Placed
  weights stay resident by design, and counting them ran a major collection
  before every output allocation once a model larger than the budget was
  placed. `jit_stats ().resident_bytes` still counts them.
- A compiled function that captures a value placed with `Rune.to_device` on
  its own device binds the value's buffer as its constant: nothing is uploaded,
  and every compiled function over the same weights shares one device copy,
  where each used to upload its own. A bound value keeps its buffer for as long
  as it is reachable: a host read copies it out and leaves the buffer in place,
  and a `~donate:true` call that takes it as an input does not consume it.
- Add `Rune.to_device ?device x`, `x` with its bytes held by a device. The
  result has `x`'s type and value and is resident like an unread output of a
  compiled call: feeding it to a compiled function on that device moves no
  bytes, `~donate:true` consumes it, and a host read brings it back. Its buffer
  bypasses the allocator's cache. On the CPU device it is `Nx.contiguous x`,
  and inside `jit`, `grad`, `jvp` and `vmap` it is `x`.
- Copies between host and device move 64 MiB at a time. `Rune.jit` staged each
  upload and read-back in a host buffer the size of the tensor and kept one per
  distinct size for the life of the compiled function, 1.1 GB for a 1B
  parameter model, and made a strided tensor contiguous whole before staging
  it. A strided tensor is now copied piece by piece, and a long run of copies
  synchronizes the device every 256 MiB.
- `Rune.jit` on the CPU device compiles kernels that read host memory at any
  address. The kernels declared their vector types aligned to their size while
  tensors were read in place wherever their data sat, so on x86-64 a slice
  starting inside a buffer, a tensor over a mapped file, or a four-wide float64
  kernel over ordinary allocated memory could kill the process.
- Compiled programs on Metal replay as batched GPU submissions: the kernels of
  a `Rune.jit` trace are recorded once and each call submits them in a few
  command buffers instead of one per kernel. The per-kernel launch cost drops
  from about 27 to 3 microseconds, level with the CPU device: a trace of 256
  small kernels takes 0.9 ms per call where it took 7.7 ms, and a GPT-2 124M
  decode step 6.6 ms where it took 9.2 ms. `JIT=2` restores one submission
  per kernel.
- Reverse mode no longer copies every cotangent. A cotangent is held as the
  lazy view its pull produced (a transpose, a broadcast) and materialized where
  a reshape needs it and where a gradient leaves `Rune.grad`. Compiled
  gradients run far fewer kernels, 95 against 210 for a two-block decoder, and
  transposes that cancel are free in the backward pass as they were in the
  forward one.
- `Nx.set` with a run-time window start (`Nx.D`) under `Rune.jit` costs the
  window instead of the destination: it compiles to a store at the window's
  flat positions, in place on a donated tensor. A one-row write into a
  1048576x64 cache takes 0.26 ms on Metal, the same as into 4096 rows, where
  it took 3.4 ms.
- Fix `Rune.grad` through `Nx.scatter` in `` `Set `` mode with repeated indices:
  every update aimed at a position received the cotangent, where only the
  last one reaches the output. Shadowed updates now get zero.
- `Rune.jit ~donate:true` writes a scatter over the donated destination's
  storage, so `Nx.scatter` into a donated tensor costs its updates alone: a
  64-row write into a 131072x8x64 pool takes 0.35 ms on Metal, the same as
  into 4096 rows. Compiled with donation, the program never copies a
  destination that is an input: the write lands in the output's buffer, which
  takes the donated storage or, on a call that cannot donate, is given the
  input's value by one device copy. `reused_bytes` counts the pool.
- `Nx.scatter` under `Rune.jit` costs the number of updates plus one copy of
  the destination, instead of destination size times update count. The
  gradient of `Nx.take` and `Nx.take_along_axis` is such a scatter: an
  embedding gradient for 1024 tokens over a 131072x256 table went from 2.1 s
  to 20 ms on CPU. `unique_indices` now reaches the compiled kernel, and an
  index outside the axis still writes nothing.
- `Rune.grad` through `Nx.matmul` now sums the cotangent of an operand over
  its batch axes of extent one that broadcast against the other operand. The
  gradient used to come back at the broadcast shape and a later pullback
  raised on the element count, so grouped-query attention (several query
  groups against one key head) could not be differentiated.
- `Rune.jit`'s compile cache now keys on every setting that changes a compiled
  trace: tolk's program configuration (`NOLOCALS`, `TC`, `IMAGE`,
  `TRANSCENDENTAL`, ...) and the scheduling variables (`SPLIT_REDUCEOP`,
  `REDUCEOP_SPLIT_THRESHOLD`, `PCONTIG`, `RING`, ...). Changing one after a
  first compile kept serving the entry built under the old value.
- `Rune.jit` no longer lets a fresh output share a buffer with an input the
  program does not read: the two buffer-slot counters (rune's and tolk's)
  could hand out the same slot, so a resident input fed to a later call had
  its bytes overwritten by that call's output. Rune now draws every slot from
  tolk's process-wide counter.
- `Rune.jit ~donate:true` writes an output over the donated input it derives
  from when every path between them stays at the same element and no later
  kernel reads the input, so a jitted training step or decode step holds one
  generation of state on the device instead of two. `jit_stats` counts the
  bytes reused in `reused_bytes`, and `RUNE_JIT_DEBUG=1` reports per donated
  leaf whether its storage was reused or copied.
- `Nx.set` with a `D` window follows every transform: `grad` differentiates
  both operands, a mapped start under `vmap` writes each example at its own
  clamped position, and a window on `pmap`'s mapped axis is written shard by
  shard.
- **Breaking**: with tensors as values (RFC 0001) there is no in-place
  update to replay. `Rune.jit` no longer writes an assigned input leaf back
  to the host on every call, and `grad`, `jvp` and `vmap` have nothing to
  refuse; carry state by returning it, as the `jit2` example shows.
- `jit_stats` retires the outputs that were dropped unread and collected
  before it reports, so `resident_bytes` counts only reachable handles. Their
  buffers used to wait for the next compiled call, which made the counter
  depend on when the GC ran.
- A parameter structure is a positional sequence of leaves: a tensor behind
  two leaves is two parameters. `jit` bound two such leaves to a single input
  at trace time, so a later call passing distinct tensors read one of them for
  both positions; every leaf visit is now its own input. `grad`, `vjp`, `jvp`
  and their variants gave both leaves the summed gradient and, for `jvp`, one
  leaf's tangent; each leaf now gets its own. Tie weights by structure, not by
  aliasing.

### Ppx_ptree (new)

- `[@@deriving ptree]` recognises a field typed `Nx.Rng.key` as a tensor leaf.
  A key belongs in a parameter structure whenever a compiled step draws from it,
  but the deriver only knew qualified types whose name ends in `t`, so the field
  had to be spelled `(int32, int32_elt) Nx.t`.

- Add `~mirror` to the deriver: `[@@deriving ptree ~mirror]` on a concrete
  record also generates its payload-generic `module Uniform`, `to_uniform`, and
  — when every leaf dtype is statically known — a dtype-checked `of_uniform`.
- Extend `[@@deriving ptree]` to payload-generic types: a type with one
  parameter occurring outside tensor leaves derives `map`, `map2`, `iter`,
  `fold` and `fold2` over dot-joined leaf paths, plus `names : 'a t -> string t`.
- Add the `ppx_ptree` deriver, which generates the `map`, `map2`, and `iter`
  operations required by `Nx.Ptree.S` for records, products, containers, and
  recursive parameter types, with generated code and diagnostics located at the
  originating source forms.
- Add a runnable linear-regression example using a derived parameter module
  directly with `Rune.grad` and `Rune.jit2`.

### Munin (new)

- Artifact digests are BLAKE2b-256 from the standard library instead of
  SHA-256, so `munin` no longer depends on the `sha` package. Blobs live
  under `blobs/blake2b` in the store.
- The system statistics stubs build on Windows. CPU times come from
  `GetSystemTimes`, memory from `GlobalMemoryStatusEx`, disk space and page
  size from the Win32 calls; the load average, which Windows does not keep,
  reads as zero.

Local experiment tracking for Raven. Evolves `kaun-board` into a full
experiment tracker — the Raven equivalent of W&B or MLFlow, without a server.
Declare a metric once with `Session.metric` and log samples through the handle
it returns, so a key is never spelled twice and a typo cannot silently create a
second channel. Save artifacts from your training script, monitor runs live in
the terminal with `munin watch`, then compare results with `munin compare`.
Data is plain JSON on disk, so `jq` and shell scripts work out of the box. Git
commit, command line, and system info are captured automatically. The
`munin.sys` sub-library adds opt-in CPU and memory monitoring in a background
thread.

- Add `x-kaun-mnist-jit`, an example tracking a `Rune.jit2`-compiled CNN
  training run (forward, backward, and SGD update in one compiled program,
  Metal by default) with live `munin watch` monitoring.

### Tolk (new)

- Add `Op.quant_matmul`, a product with MXFP4 weights that decodes them in
  registers and reads each packed byte once per tile of rows, and
  `Op.quant_row_bound`, the most rows a matrix should meet before decoding it
  costs less. One gpt-oss expert (5760 by 2880) takes 50 us at one row on an
  M1 Max, 180 GB/s over its packed bytes.
- `Op.scatter_indexed ~unique:true`, and so `Nx.scatter ~unique_indices:true`
  under `Rune.jit`, gets launch sizes from the optimizer instead of one thread
  per workgroup: 64 rows of 32768 on Metal take 0.17 ms instead of 2.9 ms. A
  scatter without the promise keeps its duplicates in index order.
- A store whose length is a variable of at most 1 writes nothing when the
  length is 0. It wrote one element, which in a decode step overwrote row 0.

- A loop whose size is a variable of at most 1 stays a loop. The symbolic rules
  replaced it with its single index, so it ran once when the size was 0.

- Add `Decomp_dtype.is_dtype_supported`: whether a renderer's programs can use
  a dtype, natively or by emulation. `float64` on Metal is neither.

- Add `Device.Buffer.as_buffer`, a buffer's bytes as host memory without a
  copy on devices the host addresses (Metal, the CPU device), as tinygrad's
  `_as_buffer`.

- `Op.cumsum`, `Op.cumprod` and `Op.cummax` values scan axes over 512 in chunks
  of 256 (262,144 elements on Metal: 209 ms to 0.44 ms; `cummax` indices stay
  quadratic). An empty int8 `Op.cumsum` returns int32, like a non-empty one.

- `Device.Lru_allocator` no longer leaks a buffer that the GC frees while an
  allocation searches the cache. The allocation stored the cache it had read,
  dropping the new entry, so that buffer was never reused or freed.

- Dropping a Metal graph no longer risks a crash at a later synchronize. Its
  finaliser pruned the in-flight command list and could run while another
  graph was pruning it, leaving a released command buffer to be awaited.

- Long Metal runs no longer exit silently with status 2. Releasing a buffer
  view from a GC finaliser could relock the buffer table inside an allocation
  that already held it, which killed decode loops after a few dozen steps.

- Emulated `int64` and `uint64` casts to `float64` preserve double precision
  and the correct high word, fixing rounded results and zero output for `2^63`.

- Split ranges retain their complete axis identities through expansion,
  scheduling and tensor-core contraction. `Uop.axis_id` exposes that identity;
  `wmma_info.tc_upcast_axes` now carries full IDs instead of root axis numbers.

- `Device.compile_program` preserves the selected renderer's alignment and
  each call's device and optimization metadata. It uses the source-based
  compiler cache, avoiding unsafe reuse of aligned kernels for unaligned inputs.

- `DEV` selects lazy renderer factories and exact compiler architectures,
  including CPU tuning and feature flags. Program caches distinguish targets;
  CUDA compilation uses the exact GPU architecture instead of a rendering tier.

- `Device.Renderer_set.make` accepts named target-aware factories. Superseded
  renderer controls and GPU architecture environment readers are removed.

- `DEV` accepts tinygrad target strings and per-backend configurations.
  AMD/NV interface selection and visible-device lists use this shared setting;
  legacy `*_IFACE` and `HCQ_VISIBLE_DEVICES` settings report replacements.

- `Op.getitem` preserves symbolic dimensions outside advanced-index axes and
  accepts symbolically sized index tensors. Both combined and separate index
  axes retain their logical shapes through gathering and masking.

- `Op.getitem` supports symbolic axis lengths and negative slice bounds.
  `Movement.parsed` and `parse_view_index` now carry symbolic sizes and bounds;
  symbolic slices require a unit step and a provably non-negative length.

- Advanced integer-tensor indexing rejects indices on a different device,
  matching tinygrad instead of constructing an invalid mixed-device graph.

- `Run.data` rejects symbolic logical shapes, matching tinygrad and the typed
  host readers. It no longer exposes the allocation's maximum-size bytes as
  though they were the tensor's concrete shape.

- `Creation.clone` supports symbolic shapes, allocating at dimension bounds
  while preserving the logical view. Assignment can now initialize symbolic
  pending tensors, and clones retain independent storage through later writes.

- `Op.assign` propagates writes through bitcast aliases and initializes pending
  values without evaluating the computation being overwritten. Partial writes
  into pending contiguous storage preserve initialization and update effects.

- `Rand.rand` supports concrete float widths beyond float32, including
  float16, bfloat16 and float64 on capable devices. Packed draws and counter
  advancement match tinygrad, including odd-sized and empty half-width draws.

- Kernels whose writes simplify away accept the resulting empty effect group,
  as tinygrad does. Identity assignments and reading the random counter after
  an empty draw no longer fail during linearization.

- `Dtype_ops.bitcast` supports different element widths by rescaling the last
  axis. Packing preserves byte order through non-contiguous views and
  unaligned subword slices.

- `Rand.rand` draws and dropout masks own fresh storage, matching tinygrad.
  Random draws can receive indexed updates before their first realization.

- Reading an assigned tensor or view now executes its pending write before
  returning bytes. `Run.data` no longer returns stale values through the
  contiguous-view shortcut, and repeated reads do not repeat the assignment.

- `Op.assign` accepts weak scalar values that promote to the destination dtype
  and commits weak destinations to concrete storage. Assignment and
  `Op.scatter_indexed` reject weak-float updates to integer storage.

- Conditional simplification folds known branch conditions before merging
  guards, avoiding redundant predicates. Constant guards stay outside index
  validity rewrites instead of repeatedly adding masks to the same access.

- Staged weak arithmetic preserves wide integer values when it needs an
  intermediate buffer. `Run.of_bytes` rejects weak dtypes, which have no storage
  representation.

- `Creation.clone`, `Creation.full`, and scalar reads select a concrete storage
  dtype without narrowing large weak integers. Scan, scatter, sort, and pooling
  padding use extrema of that concrete dtype; `Creation.empty` rejects weak dtypes.

- `Creation.full ~buffer:false` leaves an inferred numeric dtype weak, so its
  consumers choose the precision. `Op.scatter_indexed` commits weak update values
  to the destination dtype, matching assignment.

- `Reduce.sum`, `Reduce.prod`, `Reduce.max`, and `Op.mean` preserve weak integer
  inputs beyond `int32` by choosing a storage dtype from their bounds. Exact values
  beyond all integer storage types are rejected before accumulation.

- `Uop.max_numel` rejects overflowing element counts and handles zero-sized
  shapes with large dimensions exactly. Large shape products no longer cause
  gated buffer indices to narrow incorrectly to `int32`.

- Weak integer conversions preserve truncation through nested casts and
  arithmetic. A narrow result cast no longer narrows wide division operands
  before the computation.

- Preserve floating-point rounding when simplifying comparisons with added
  constants. The rewrite `(c0 + x) < c1` to `x < c1 - c0` now applies only
  to integers.

- Fix symbolic simplification changing integer bit masks into multiplication.
  Validity-predicate rewrites now apply only to booleans, preserving expressions
  such as `(x & 12) & x`.

- `Uop.vmin`, `Uop.vmax` and parameter bounds retain exact integers and
  floating-point limits. Weak scalars can commit to `uint64`; values beyond
  all supported integer widths are rejected instead of silently narrowing.
- Remove the obsolete `Dtype.Uint128` and `Dtype.Uint256` storage helpers,
  matching the target scalar dtype set. Existing serialized caches are invalidated.
- Missing runtime variables report their name and program or replay call,
  including variables needed to compute launch dimensions.
- `CHECK_OOB` rejects unproved scalar accesses containing bitcasts or stacks
  and accesses without a known buffer extent. Out-of-range signed casts no
  longer produce spurious empty bounds.
- Beam search uses the 0.01µs progress threshold and reconsiders candidates
  rejected by a previous round's compute filter while reusing compiled code.
- `State.safe_load` decodes escaped Unicode names and validates metadata,
  shapes and byte offsets before uploading tensors. Empty host inputs no
  longer attempt zero-byte native allocations.
- Fix `asinh` for large negative inputs, log-space operations at infinity,
  activation saturation, and integer `Op.var`.
- Extrema, scans, sort and padded integer `Op.max_pool2d` preserve full-width
  integer identities. `Op.arange` accumulates small floats in float32 before
  narrowing, and `Op.pad_to` supports a nonzero fill value.
- Release unreachable tensor buffers and cached graph properties. Live views
  and JIT captures retain their storage; serialized program cache keys no
  longer depend on process-local node identities.
- `Uop.exec_alu` preserves full-width integer values and exact weak-integer
  intermediates. Literal matching avoids float rounding; weak promotion
  preserves padded values. `Const.Int` now carries a `Z.t` mathematical value.
- `Dtype.min` and `Dtype.max` return exact `Z.t` integer bounds, including
  weak integers, and finite limits for FP8 formats without infinities.
- AMD and NV devices automatically fall back to PCI when kernel-driver
  interface initialization fails. Explicit `AMD_IFACE`/`NV_IFACE` selections
  continue to restrict the device to the requested interface.
- Metal kernels cast between `bfloat16` and `float32` by bit manipulation, as
  the reference does to avoid a Metal compiler bug with
  `as_type<half>((bfloat)(const))`. tolk rendered native casts. Values are
  unchanged: both round half to even.
- `DEBUG=2` prints one line for every executed kernel, view, copy and batched
  graph: device, call count, name, memory in use, time, and GFLOPS and GB/s
  from the kernel's estimates. `Helpers.Global_counters` holds the running
  totals. Batching hides kernels inside a graph call, so `JIT=2` gives the
  per-kernel profile of a compiled function.
- A compiled function that is dropped now releases the device memory of its
  batched graphs. Recorded graphs were kept in a table that was never emptied,
  and each one holds the buffers of its intermediates: a process that compiled
  many functions, or one function at many shapes, grew without bound on Metal
  and CUDA (108 MB per dropped function in a three-matmul probe at 3072x3072).
  `Tolk.Realize.graph_runners` reports how many recorded graphs are live.
- `Cstyle.clang` and `Tolk_cpu.create` take `?aligned`. `~aligned:false`
  declares vector types aligned to one byte, for a device that binds memory it
  did not allocate. Absent, the `ALIGNED` environment variable decides, as
  before.
- The Metal device carries a `Device.Graph` capability: a batched call sequence
  is encoded once into an indirect command buffer and replayed as a single
  command buffer, with rebound buffers, variable values and launch dimensions
  patched in between. `Device.Graph.t` gains `max_buffer_offset`, which keeps
  a call whose buffer view starts past 4 GiB out of Metal graphs, and
  `FIX_METAL_ICB` overrides the pre-M3 pipeline workaround.
- `Creation.clone` takes `?device` and copies a source that lives on another
  device across. `Op.scatter_indexed` places its buffers on the device of its
  operands.
- Add `Op.scatter_indexed`: a scatter along one axis whose kernel ranges over
  the updates instead of the destination, and writes the destination's
  storage in place, as `Op.assign` does. Duplicate updates land in index
  order (the last `` `Set `` wins, `` `Add `` accumulates), an index outside the
  axis writes nothing, and `~unique:true` lets the updates run in parallel:
  a position aimed at twice then holds an unspecified one of its updates, and
  every other position stays exact.
  `Op.scatter` and `Op.scatter_reduce` keep the reference lowering.
  `Creation.clone` is now exposed.
- Fix `contiguous` over a window that narrows a trailing axis, such as the
  first two columns of a 2x3 buffer. The view was taken for one range of the
  flat buffer and read the wrong elements; only a window whose earlier axes
  all have extent one is such a range.
- Add `Uop.custom_kernel` and `Tensor.custom_kernel`: a kernel written in uops
  runs from the tensor graph over realized sources, and each source is read
  back after the kernel. `Uop.placeholder` and `Uop.placeholder_like` build
  the storage a kernel body addresses, and `Creation.empty` allocates storage
  for a kernel to fill, on `?device` when given.
- A gather over 32768 rows or more compiles to one kernel with one gated load
  per output element. The reduce split used to fire before the gather collapse
  and the kernel read the whole table into an intermediate buffer, so
  `Op.gather`, tensor-index `Op.getitem` and rune's compiled `Nx.take` cost a
  pass over the table.
- The in-memory program cache keys on `Realize.program_config`: `NOOPT`,
  `NOLOCALS`, `TC`, `IMAGE`, `DISABLE_FAST_IDIV`, `TRANSCENDENTAL`,
  `ALLOW_TF32` and the default dtypes. A kernel compiled under one setting was
  served under another when the setting changed through `with_context`.
- Every context variable is declared once in `Helpers`, and a second
  declaration of a key raises. `PCONTIG` had two independent copies, so an
  override reached only one reader. `IMAGE`, `FLOAT16`, `TC`, `TC_SELECT`,
  `TC_OPT`, `NOOPT`, `TRANSCENDENTAL`, `DISABLE_FAST_IDIV` and `ALLOW_TF32` are
  now context variables read once at startup; override them with
  `Context_var.with_context`. `Heuristic.nolocals_var` is `Helpers.nolocals`.
- Float literals in rendered kernels are laid out by tolk rather than the C
  runtime, so they are identical on every platform; Windows printed their
  exponents with three digits.
- A copy between devices is scheduled as a kernel storing the source's
  flat view into a buffer on the target device, then turned back into a
  transfer once the schedule is linear. Multi-device graphs therefore
  emit one kernel per per-shard copy, as the reference does, instead of
  fusing copies into their consumers; a kernel that mixes devices without
  being a copy is rejected with `all buffers must be on the same device`.
- Multi-device sharding slices along a device range instead of a
  `_device_num` variable. The range is not a program axis: codegen lowers
  it to `_device_num` and keeps it out of range splitting and the
  optimizer's axes, and kernels number it first, so a sharded kernel's loop
  axes now start at 1 (`Lidx1`), as in the reference.
- Fix the symbolic division rules, which never fired: `x/x`, `(x*y)/y`,
  `0/0`, `(x*0)/0` and `(x/y)/z` were matched on `Fdiv`, an operation that
  only exists after the late decompositions, while a division in the graph
  is `x * recip y`. Chained divisions now fold to one division, so a kernel
  computing `(a/b)/c` renders `a/(b*c)`.
- CPU kernels run on Windows. Their entry now carries the Microsoft calling
  convention there, as the reference does: the object is compiled for a
  generic ELF target whose x86-64 convention is System V, so the host read
  the kernel's arguments from the wrong registers and larger kernels crashed.
- The CPU thread pool is sized from the runtime's recommended domain count
  instead of a `getconf` subprocess, which does not exist on Windows.
- CPU kernel timings come from a monotonic high-resolution clock. The wall
  clock moves in millisecond steps on Windows, which tied every fast kernel at
  zero and left beam search nothing to rank.
- The disk cache creates its temporary file exclusively under a random name
  instead of one derived from the process id. On Windows `Unix.getpid` is a
  handle value that sibling processes routinely share, so concurrent writers
  wrote into one temporary and tore the entry. A temporary whose final rename
  loses is also removed, so none accumulate next to the entries.
- The CUDA, NVRTC, comgr, and driver runtimes build on Windows. The vendor
  libraries load through `LoadLibrary`; the hcq layer maps anonymous memory
  through `VirtualAlloc`, so the queue builders run; the system layer's file
  primitives work, while the Linux kernel interfaces report themselves
  unsupported.
- New `Tolk_frontend.Linalg`: `qr`, `solve_triangular`, and `cholesky` unroll
  at graph-construction time into ordinary Tolk compositions, so they compile
  for every Tolk device. Large systems solve block-by-block as GEMMs, with
  the diagonal blocks inverted by one batched substitution.

- Driver-less NVIDIA (`NV_IFACE=PCI`) hardening: opening now waits for the
  GPU's boot firmware to report ready before sizing VRAM (a device opened
  mid-boot, or straight after the armed-region recovery reset, could read
  garbage), the GSP client is registered before the first object call,
  shutdown finalizes the boot layers in reverse bring-up order, and stalled
  waits back off to draining GSP events after 200ms instead of on every
  poll.

- `NV_DEBUG=4` traces every register write on the driver-less NVIDIA path
  (`wreg: 0x<addr> = 0x<value>`); `DEBUG>=2` logs the armed-region recovery
  reset and `DEBUG>=3` decodes incoming GSP RPCs by name.

- NVIDIA GPUs can now be driven over PCI with no kernel driver: setting
  `NV_IFACE=PCI` boots the GPU's GSP firmware directly — the falcon and
  chain-of-trust bring-up, the shared-memory RPC queues, the golden-image
  channel and context setup, and the object, control and memory calls as GSP
  remote procedures (`Tolk_nv.Pci_iface`), reading firmware from
  `NV_FW_PATH`. Opt-in and unvalidated on hardware so far; the kernel driver
  remains the default.

- A stalled AMD device wait now fails with the whole story — the timeout
  (expected and observed timeline values) folded with the driver's fault
  report — where it could previously raise `Failure("")` when the driver had
  no fault to report. Stalled waits also back off to the driver's event
  sleep after 200ms instead of 2s, surfacing faults sooner.

- NVIDIA GPUs are now a hardware-queue runtime target: `DEV=NV` (or
  `Tolk_nv.create`) drives the kernel driver's channels directly, with
  kernels compiled straight to cubin by nvrtc, covering the Ampere, Ada,
  and Blackwell generations. The userspace CUDA backend (`DEV=CUDA`)
  remains available unchanged.

- Driver-less AMD devices can now sleep on interrupts instead of spinning:
  with `VFIO=1` (and the `vfio-pci` kernel module), the device's MSI vector
  is routed to an eventfd that stalled waits block on. Without it, waits
  poll as before.

- AMD GPUs can now be driven over PCI with no kernel driver: setting
  `AMD_IFACE=PCI` boots the GPU directly — firmware loading, memory hubs,
  security processor, engines (`Tolk_amd.Pci_iface`) — covering the
  RDNA3/RDNA4 consumer parts. Opt-in and unvalidated on real hardware so
  far; the kernel driver remains the default.

- AMD GPUs are now a runtime target: `DEV=AMD` (or `Tolk_amd.create`) drives
  the GPU through the Linux kernel driver's hardware queues, with kernels
  compiled by the ROCm comgr library. Supports gfx942, gfx950, and the
  gfx11/gfx12 generations (single-die), with DMA-engine host transfers and
  device-side execution timing.

- Fix the AMD renderer's `__ockl_get_local_id`/`__ockl_get_group_id`/
  `__ockl_get_local_size` declarations: the return and argument types were
  swapped (`unsigned int f(size_t)` instead of `size_t f(unsigned int)`), so
  kernels using launch indices declared the OCKL intrinsics with the wrong
  signatures.

- `Elf` now loads shared-object (`ET_DYN`) images in addition to
  relocatable objects, and builds the image from program sections rather
  than all allocatable sections, so GPU code objects load with the same
  layout their producers intended.

- The AMD/HIP renderer (`Cstyle.amd`) is now covered by the tinygrad parity
  and golden corpora on gfx1100, including tensor-core matmul cases across
  RDNA3/RDNA4 (WMMA f16/bf16), CDNA3 (MFMA bf16), and CDNA4 (scaled MFMA
  fp8).

- `Search.beam_parallel` is a `Helpers.Context_var`, so a caller can scope
  the `BEAM_PARALLEL` worker count to one compilation with
  `Context_var.with_context` instead of setting it process-wide.

- `Realize.pm_compile` takes `?beam`, stamping kernels that do not already
  carry a beam width, so a caller can enable beam-search autotuning for one
  compilation instead of process-wide through the `BEAM` environment
  variable.

- Free `BEAM`-search timing buffers as soon as each kernel's search finishes:
  they were reclaimed only when the GC ran, and then into the unbounded LRU
  allocator cache (which matches by exact size), so a model with many
  distinct kernel shapes accumulated their GPU memory until a failed
  allocation flushed the cache — OOM on large graphs with `BEAM>=1` where
  `BEAM=0` fits. Timing buffers now bypass the LRU cache and are released
  deterministically per kernel.

- Compile `BEAM`-search candidates in parallel domains (`BEAM_PARALLEL=N`,
  default off): the CPU-side compile of a step's candidates — optimize, lower,
  render, nvrtc — now overlaps across domains, while the GPU timing phase
  still runs one candidate at a time so timings never contend. Shared state
  is safe under domains: the hash-cons table, kernel naming, program cache,
  and disk cache take locks, and the per-node memo caches are domain-local.
  On the RNN grad repro (CUDA,
  `BEAM=2`, cleared tolk and driver JIT caches): 27s -> 6s with
  `BEAM_PARALLEL=8`; with warm caches ~11s -> ~7s.

- Stop `BEAM` search when progress falls below timer noise: the default
  `BEAM_MIN_PROGRESS` is now 5µs (env still overrides). The old 0.01µs
  default sat below the ~0.5µs device timer resolution, so search only
  stopped when a step brought no improvement at all — kernels kept searching
  on measurement noise, ~2.6× more candidate compiles on a CUDA `BEAM=2`
  pass.

- Test same-call WAR dependencies by node identity instead of structural
  equality in `fix_war_deps`. `Uop.t` nodes are hash-consed, so the two checks
  give the same verdict, but polymorphic equality on the `call_of` options
  descends into the full shared payload DAGs (exponentially on graphs with
  heavy sharing) where `==` is O(1).

- Reduce cold `BEAM`-search compile time by removing two sources of redundant
  CPU work: the device dispatch handle (driver module load, PTX → SASS JIT) is
  loaded once per compiled binary and reused across timed candidates, and
  candidates whose AST was already compiled are deduplicated up front via the
  hash-consed tag instead of only after `nvrtc`.

- Fix CUDA tensor-core kernels failing to compile. Every `__WMMA_*` primitive
  was emitted with scalar parameters and empty asm operand lists —
  `float f(half a, half b, float c)` — while the kernel body correctly built
  vector operands for it. The renderer rebuilt the three operand widths from
  the tensor-core upcast axes, which the expander clears once it has applied
  them, so the one-lane fallback always fired; the widths now come from the
  operands themselves. Metal was unaffected.
- Fix a gated image load falling back to a single zero instead of a full
  vector. The value a gated load reads when its gate is false was sized by
  re-deriving the access width from the index expression, which had no case for
  an image coordinate and defaulted to one lane; an image access reads four
  floats, so three lanes were left unwritten. The width now comes from the
  load's own lane count, exposed as `Uop.max_numel`.
- Fix multi-device programs failing to compile with
  `Invalid_argument "buffer copy: size or dtype mismatch"`. An operand that
  broadcasts along the shard axis — one of lower rank than the result, or with
  size one there — was split across devices as if it held a distinct slice per
  device, so a collective was sized from a fraction of its real shape. Such an
  operand now stays whole on every device. Separately, the scheduler refused to
  read a shape off a `Pad` or `Shrink` whose offset was symbolic, even though
  the shape is the size argument alone; the resulting unknown propagated into
  an elementwise node as a too-small shape and mis-sized the allreduce buffers.
- `Op.cat` joins operands with equal extents on the concatenated axis through a
  single stack node instead of padding each operand to the full width and
  summing them, so a concatenation of `n` inputs now selects on one loop range
  rather than emitting `n` pads and `n - 1` adds. It also rejects operands whose
  shapes differ off the concatenated axis, which used to broadcast silently.
- Move `stack` from `Op` to `Movement`, where it now builds a stack node
  directly rather than unsqueezing every operand and concatenating. Call sites
  change from `Op.stack` to `Movement.stack`.
- Fix an unsound simplification of `((hi << 32) | lo) >> 32`, which returned
  `hi` even when `hi` had bits above 32 that the shift had discarded. It now
  requires `hi` to come from a 32-bit source, matching what packing two `uint32`
  halves into a `uint64` actually produces.

- Fix a scheduler miscompile that made mixed-precision training steps fail to
  compile. Rangeify records per-node loop ranges while walking the tensor
  graph, and was carrying those records over to the nodes it rebuilt. Because
  rebuilding removes movement ops, two values that differed only in a
  broadcast collapse into one node — which then inherited the wrong rank, and
  its reader was indexed with fewer indices than the staged buffer had axes.
  The result was one store per element to a single address, rejected as
  `Coalesce: multiple stores to the same offset`, or silently wrong values
  where it slipped through. A shape shared between consumers of different rank
  is now mapped into each source's own axes, and a mismatched index raises
  instead of being tolerated.

- Rewrite the graph rewriter's traversal so nodes are visited in dependency
  order rather than plain depth-first order. Passes that number things as they
  go — accumulator registers, local buffer slots — now number them the same way
  the reference does, so generated kernel source matches it exactly instead of
  differing in register and buffer names. A call or function body is now left
  alone on every path that reaches it, not only on the edge through its call.
  This costs compile time: roughly 20% more wall time and 15% more allocation,
  concentrated in the codegen, schedule, and rangeify stages. The new traversal
  keeps a scheduled set and revisits each rebuilt node, which is inherently
  more bookkeeping than the depth-first walk it replaces.

- Tighten index arithmetic in tensor-core kernels. A scaled remainder and its
  quotient partner now recombine even when the quotient's divisor has absorbed
  an inner division, so two adjacent single-bit extracts of a thread id
  collapse into one multi-bit extract instead of being emitted separately.
- Fix a symbolic rewrite that dropped valid indices from a bounds test. A
  comparison of the form `x // d < c` with a non-positive `c` was rewritten to
  a bound one step too tight, so `x // 2 < 0` became `x < -1` and excluded
  `x = -1`, which satisfies the original. Any gate, mask, or loop bound reduced
  through that shape could exclude a valid element.
- Fix silently wrong results from any kernel holding both a group reduce and
  an ordinary loop reduce — the shape of most backward passes on a GPU. The two
  reduces were given the same accumulator register and the second's zero-init
  was dropped, so the closing add read one accumulator twice and the kernel
  computed `2(a+b)` where the answer is `a+b`. Affects every target with local
  memory (CUDA, Metal, OpenCL); CPU was never affected.
- Restore multi-threaded CPU kernels for large fused reductions. The
  ops-per-thread heuristic formed the iteration-space product in a machine
  integer, which wrapped on kernels fusing enough reduce axes (an unrolled
  recurrence's weight gradients reach 2^110), so the thread split was silently
  skipped and those kernels ran single-threaded.
- Multi-device programs with two outputs that both need a cross-device
  reduction — a data-parallel step returning both the loss and the gradient of
  a replicated parameter — no longer miscompile. Flattening a bufferize folded
  the source's own shape into the flattened extent, which overflowed to a
  negative size whenever that shape was unresolved, leaving the gradient with a
  one-element buffer that every lane wrote to at index zero. `Rune.pmap` steps
  that hit this failed to compile rather than returning wrong numbers.
- Tensor-core kernels now declare each `WMMA` result at the accumulator width
  the instruction actually returns, so tensor cores are usable again on Metal
  and CUDA. The renderer declared them scalar, emitting programs the target
  compilers reject (`cannot initialize a variable of type 'float' with an
  rvalue of type 'float2'` on Metal). Any tensor-core candidate therefore
  failed to compile, and `BEAM` search silently discarded all of them.
- Reading a tensor whose graph folds to a pure constant (`Run.data`,
  `Run.to_float_array`, `Run.item_float`, ...) now materializes it into a fresh
  buffer instead of raising: such a graph is placed on no device and owns no
  storage, so `Run.buffer_of` had nothing to return. Affected every
  constant-folded result, for instance `Rand.dropout ~p:1.`.
- A cast to an unsigned dtype now keeps the source's exact symbolic bounds when
  that source already fits the destination window; only a source that can wrap
  (negative, or wider than the destination) falls back to the full dtype range.
  `Uop.vmin`/`Uop.vmax` previously gave up on every unsigned cast.
- Reject the tensor-core optimisation when one of its X or Y axes is a reduce
  axis. Those axes index the WMMA accumulator tile, so a kernel that reduces
  over a matmul's output dimensions (`(a @ b)` summed over `M`, as a recurrent
  loss does) compiled to a kernel that computed the wrong values. `BEAM` search
  ranks candidates by runtime alone, so it could select it silently.
- Invalidate the on-disk cache, so a beam result cached before the tensor-core
  fix above is re-searched rather than replayed into an optimisation that is
  no longer legal.
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

- The structural optimizers (`sgd_step`, `adam_step`, `adamw_step`) pass
  non-float leaves through unchanged instead of updating them, so a structure
  carrying an `Nx.Rng.key` alongside its parameters survives a step. Adam is
  where it showed: its direction runs each leaf through a square root and a
  division.
- The structural functions take their parameter-tree module as
  `(module Nx.Ptree.S with type t = 'p)`, so a first-class module value —
  e.g. the result of `Nx.Ptree.instantiate` — can be bound once and passed
  to the optimizer steps and gradient transformations.
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

- Add `nx.quant`: `Nx_quant.mxfp4` builds a weight from a checkpoint's MXFP4
  codes and scales without a copy, and `Nx_quant.apply ?ids` multiplies by it
  with `Nx.matmul`'s shapes, decoding a bounded chunk at a time eagerly.
- `Nx.top_k` with `k` above 8 no longer takes one entry per pass, and over an
  axis longer than 2048 no longer sorts it: a radix select on the bits of each
  entry finds the `k`th greatest and only the `k` entries kept are ordered.
  Compiled on Metal, 40 of 50257 float32 take 2 ms instead of 890 ms and 512
  of 32768 0.9 ms instead of 3.7 ms; eagerly it costs about 2.5 times the old
  sort at sampling sizes, and 64 rows of 131072 peak at about 480 MB instead
  of 200 MB. Compiled, NaN now comes last as documented.
- Add `Nx.bitcast`, which reads each element's bits as another dtype of the
  same width without converting it, NaN payloads and subnormals included. It
  compiles under `Rune.jit`, except to or from float8, which the compiler
  emulates and `Rune.jit` refuses; it maps under `vmap` and has zero
  derivative.
- `Nx_buffer.create` documents what it does: the contents of a new buffer are
  unspecified. It claimed a zero fill, which only some element kinds got, so a
  new `float32` buffer could hold NaN; call `Nx_buffer.fill` for zeros.
- Add `Nx.Device`, `Nx.Placement`, `Nx.place` and `Nx.placement`: a value
  can live on a device a runtime opens, and where it lives is a value. An
  operation on placed operands returns a placed result, host operands join
  them, and operands on two devices raise `Invalid_argument`.
- A read of a placed value (`item`, `to_array`, `to_buffer`, `pp`) copies the
  elements it reads and leaves the value where it is. `Nx.data` of a placed
  value raises: it has no host storage.

- Fix `Nx_io.load_safetensors` and `save_safetensors` corrupting Unicode and
  control characters in tensor names. Decode JSON Unicode escapes and surrogate
  pairs, emit valid JSON escapes, and reject malformed string escapes.

- Add `Nx.Ptree.leaf`, the structure that is one tensor, for every dtype.
  Pass it for the single-tensor roles of a transformation that takes several
  structures, instead of writing a one-leaf module per tensor type.
- Add `Nx_buffer.register_file` and `Nx_buffer.file_range`. A buffer whose
  memory lies inside a recorded file mapping, views and reinterpretations
  included, answers with the file and the byte offset of its first element, so
  that an upload can read the bytes from the file instead of faulting them in
  through the mapping. `Nx_io.load_safetensors` records its mappings.
- `Nx_io.load_safetensors` maps the file instead of reading it: loading reads
  the header only, and each tensor is a view of the file whose pages are read
  when first used. It used to hold the file twice in memory and copy every
  tensor out element by element; a 2.5 GB checkpoint that took 3 s to load
  takes 0.05 s. Entries at an address their dtype cannot be read from are
  copied. A loaded file must not be modified in place while its tensors are
  alive; `Nx.copy` detaches a tensor from its file.
- `Nx_io.load_safetensors` loads `F8_E8M0`, `F4`, `F6_E2M3` and `F6_E3M2`
  entries as their `uint8` bytes instead of dropping them with a warning,
  loads 16-bit entries at odd offsets instead of raising, and rejects a file
  whose length disagrees with its header, a header that names a tensor twice,
  and anything that is not a regular file. Its errors name the file.
- Add `Nx_buffer.reinterpret kind buf`, `buf`'s memory read as elements of
  `kind` without a copy. It is the only way to view existing memory, such as a
  mapped file, as `bfloat16`, `float8`, `bool`, `uint32` or `uint64`, whose
  kinds only allocation could set before.
- `Nx_io.save_safetensors` no longer truncates its destination in place: it
  writes a temporary file beside it, syncs it and renames it, so a crash or a
  failed save leaves the previous file whole. If the rename is refused the
  written file is kept and the error names it. Saved files now have mode
  `0o640`, as the other `Nx_io` writers give theirs.
- `Nx.cast` at the tensor's own dtype is the tensor itself and no longer a
  copy: only a change of dtype allocates. Use `Nx.copy` for fresh storage.
- Fix `einsum` with a repeated index that does not sit at the end of its
  operand (for example `abnb->an`): the surviving label stayed where the index
  first appeared instead of moving with the diagonal to the end of the
  operand, so the result took the wrong shape and values.
- Add `Nx.top_k ~k ?axis`, the `k` greatest entries along an axis and their
  positions, as `(values, indices)`: the first `k` of a descending `sort`, ties
  lowest position first, NaN last. Up to 16 entries it costs `k` passes and no
  sort, which is what a mixture-of-experts router needs under `Rune.jit`, where
  `argsort` is quadratic in the axis. `values` differentiates.
- `sort`'s documentation said NaN sorts first in descending order. It sorts
  last in either direction, as the backend contract states.
- `scatter` states what a broken `unique_indices` promise leaves: a position
  selected more than once holds an unspecified one of its updates under
  `` `Set `` and an unspecified value under `` `Add ``, and every other position
  is exact. The whole result used to be undefined, which ruled out aiming the
  updates one does not want at a scratch row. Eager and `Rune.jit` both keep
  the narrower promise.
- Add `sliding_window`, a zero-copy view framing a tensor into windows of a
  given length along an axis. Framing without a copy was reachable only through
  `stft`, which bundles a taper and a transform with it; a reduction, a filter
  or an overlap-save convolution wanting the frames themselves had to gather
  them with `extract_patches`.
- **Breaking**: tensors are values (RFC 0001, `doc/rfc/0001-tensors-as-values.md`).
  The in-place writers `blit`, `set` (int-list form), `set_slice`, `set_item`,
  `put`, `index_put`, `put_along_axis` and the `.%{}<-` / `.${}<-` operators
  are gone, with `empty` and `empty_like` (a value has no uninitialized form
  to fill in; use `zeros`) and the `?mode` of `take`. One
  functional `set specs v t` returns `t` with `v` at the selected positions,
  and a new index form `D (start, len)` selects a run from a run-time start,
  so a KV-cache write traces once for every position. Views, broadcasts and
  overlapping windows are ordinary tensors; `data` lends storage read-only and
  `of_bigarray` and `of_buffer` take ownership. Tensor-valued indices
  (`take`, `scatter`) must lie in range: the C backend no longer wraps
  negatives and raises `Invalid_argument` instead of `Failure`. Build tensors
  element by element with `create`, `init`, `stack` or a filled bigarray.
- The backend contract gains `update`, the pure window write `set` lowers to
  for single indices, unit-step ranges and run-time runs; a compiler can
  perform it in place.
- Add `Nx.erfinv`, the inverse error function, beside `erf`. At float64 it
  carries double precision, a seven-digit polynomial refined by Newton steps
  against a series for `erf` that does not depend on the backend's own. It
  was an internal helper of `Nx.Rng.truncated_normal`, whose float64 draws now
  carry double precision as well instead of seven digits.
- `Nx.Rng.poisson` runs at its rate's compute dtype instead of always at
  float64, so a float32 rate compiles on Metal and every other device without
  double precision. The rejection test now evaluates the log pmf in a form
  that does not cancel terms of size `rate log rate`, which is what had forced
  float64. Draws for a float64 rate are unchanged; a float32 rate gives a
  different stream than before.
- Add `Nx.Rng.bits k shape`, the generator's raw uniformly random 32-bit
  words, so a distribution the module does not provide can be built on the
  same generator with the same purity and transform guarantees. `uniform` at
  float32 is the low 24 bits of these words scaled into `[0, 1)`.
- Add `Nx_io.encode_png`, the PNG bytes of a `uint8` image tensor as a
  string, for embedding images in documents or sending them over a socket
  without going through a file.
- Add `Nx_io.deflate` and `Nx_io.inflate`, zlib compression of strings,
  the format PDF and PNG streams use.
- The C backend compiles with mingw-w64 on Windows. It no longer relies on the
  C11 `CMPLX` constructors or `aligned_alloc`, which mingw's headers lack, and
  registers no fork handlers there. The I/O layer writes through Windows file
  handles, so `save_npy`, `save_npz`, `save_txt`, the image encoders, and gzip
  output work there.
- `save_txt` writes exactly the requested `newline`. The channel was in text
  mode, which turned every line ending into CRLF on Windows.
- `save_txt` formats floats itself, correctly rounded to numpy's 19
  significant digits on every platform. A C runtime only has to round 17,
  and Windows' stops there and prints three-digit exponents.
- **Breaking:** the samplers take their distribution parameters as tensors,
  elementwise, and the draw has the parameters' shape and dtype:
  `Nx.Rng.bernoulli k p`, `poisson k rate`, `gamma k concentration`,
  `beta k a b`, `truncated_normal k lower upper` and `dirichlet k
  concentration` (components on the last axis), with the keyless `bernoulli`
  and `truncated_normal` following. A tensor of rates now gives one count per
  rate in a single draw, and a parameter that is a jit input or a vmap axis
  traces or batches the draw with it. A scalar parameter is spelled
  `Nx.broadcast_to shape (Nx.scalar dtype v)`. Parameter values are no longer
  checked, since under a transform they are not known when the program is
  built; each docstring states its domain. `Nx.Rng.uniform` loses its unused
  `?low`/`?high`, and `categorical` loses `?shape`: its result is the shape of
  the logits with the axis removed, so broadcast the logits for more draws.
  `randint` keeps its host-int bounds, as every size in Nx is a host int.
  `truncated_normal` differentiates through both bounds; `gamma`, `beta` and
  `dirichlet` differentiate through their concentrations with the bias of
  the accepted proposal. Draws from `gamma`, `beta`, `dirichlet`,
  `truncated_normal` and `categorical` differ from before for a given key.
- `Nx.Rng.poisson` samples at any rate; the cap at 100 is gone. Below rate
  10 the count is read off the cumulative distribution with a single uniform,
  from 10 up it comes from a transformed rejection sampler with a fixed round
  count, so the work per element no longer grows with the rate. Draws for a
  given key differ from before.
- The inverse error function behind `Nx.Rng.truncated_normal` now has a finite
  derivative at zero. Its tail branch took `sqrt` of a quantity that is zero
  at that point, and although the branch is never selected there, its infinite
  derivative turned the gradient into NaN under `Rune.grad`. Values are
  unchanged.
- Every keyed draw (`Nx.Rng.uniform`, `normal`, `permutation`, `split`, and
  the samplers built on them) no longer copies its key once per Threefry
  block before hashing. The copy was as large as the draw itself; the kernel
  reads the key through a stride-0 view instead. Values are unchanged.
- Add `Nx.solve_triangular ?upper ?transpose ?unit_diag a b`, a first-class
  triangular solver named after its scipy analog. It skips the factorization
  cost of `solve` for a pre-triangularized `a`; `b` is a vector or a stack of
  right-hand sides, batched like `a`.
- `Nx.solve`, `Nx.inv`, and `Nx.matrix_power` raise `Linalg_error` with kind
  `` `Singular `` for a singular matrix, as `solve_triangular` does, instead
  of `Invalid_argument`. The check now lives in the graph rather than reading
  the factor back to the host, which is what lets `solve` and `inv` compile
  under `Rune.jit`.
- **Breaking:** the backend operation `triangular_solve` and its effect
  `E_triangular_solve` are renamed `solve_triangular` and `E_solve_triangular`.
  Out-of-tree backends and effect handlers must follow.
- `Nx.diag` no longer reads its operand back to the host, so it traces under
  `Rune.jit` and construction differentiates through `scatter`. It now raises
  for inputs of rank above 2, as documented; the undocumented reading of the
  row-major flattening as a matrix is gone. The packed `int4` and `uint4`
  dtypes, which the graph operations reject, are no longer accepted either.

- Speed up batched `fft`, `rfft` and `irfft` in the default C backend: the
  worker count was picked as though a transform line cost one pass over its
  samples, so a stack of a few dozen medium-length lines ran on a single core.
  Lines are now weighted by the `n log n` work a transform actually does; short
  stacks still run serially.
- Speed up the C backend's real FFTs: even-length `rfft` and `irfft` pack into
  a half-size complex transform, and `irfft` no longer stages a serial copy of
  its input. Even-length `rfft` now returns exactly real DC and Nyquist bins.
- Pad Bluestein lengths (a prime factor above 13) to the nearest 7-smooth size
  instead of the next power of two, so `fft` at 4099 runs on 8232 points, not
  16384. Results there change in the last bits (error within ~1.1x of before).
- `irfft` at an odd Bluestein length now discards `Im X[0]` like every other
  length, so a non-Hermitian input no longer leaks ~1e-16 of it into the output.
- `irfftn` and `irfft2` now honour `s` along every transformed axis: the
  leading, complex axes are cropped or zero-padded to the requested lengths,
  as in `ifftn`. Previously only the last axis was resized while the
  normalization still divided by the full product of `s`, so a leading-axis
  size change returned the wrong shape and scale.
- **Breaking**: `Rng.run ~seed f` is replaced by `Rng.with_key (Rng.key seed) f`,
  and `Rng.split_off` is renamed `Rng.next_key`. The two entry points were one
  handler differing only in what it rooted at, and `~seed` was the one that
  silently rooted at a constant — the case `Rune.jit` must refuse. A scope now
  visibly inherits its root key's properties: root it at a jitted function's
  input leaf and the keyless samplers inside compile.
- Add `Rng.beta` and `Rng.dirichlet`, built from `Rng.gamma` and inheriting its
  approximation. `dirichlet` puts its components on a new trailing axis, so a
  draw of `shape` gives `shape @ [| n |]` with every row on the simplex.
- Add `Rng.gamma` and `Rng.poisson`. `gamma` is the keystone
  for statistical work — beta is `g1 /. (g1 +. g2)`, a Dirichlet is a vector of
  gammas over its own sum, chi-square and Student's t follow in turn — and its
  docstring records those derivations. It is the one sampler in `Rng` that is
  not exact: every gamma algorithm rejects and a rejection loop cannot be
  traced, so eight attempts are drawn and the first acceptance taken, leaving
  about one element in `1e14` to fall back to the mean. `poisson` is exact,
  expressing Knuth's count as a cumulative product rather than a loop that stops
  when the draw says so; its cost is `O(rate)` per element, so `rate` is capped
  at 100.
- Add `Rng.gumbel` and `Rng.exponential`. `gumbel` is the noise
  behind `categorical`, which now builds on it; adding it to log-probabilities
  and taking a softmax instead of an argmax gives the relaxed, differentiable
  form. `exponential` is `-log (1 - u)` — built from `1 - u` because a draw can
  be exactly `0`, where `-log u` would diverge.
- float64 draws carry 53 random bits instead of 24. `uniform` built every draw
  in float32 and widened it, and `normal` ran the whole Box-Muller transform in
  float32, so a float64 sample was a double holding float32 noise — 2^24
  distinct values instead of 2^53. This matters most to `Norn`, whose HMC and
  NUTS samplers draw momenta with `randn f64` and accept with `rand f64`.
  float64 streams therefore change; narrower dtypes are unaffected.
- Add `Rng.fold_in_tensor`, deriving a subkey from a value known only at run
  time — a step counter carried through a compiled loop, a device index.
  `Rng.fold_in` takes a host `int`, so under `Rune.jit` it freezes whatever the
  counter held at trace time; only the mapped-axis specialisation
  (`Rng.fold_in_axis`) was reachable before.
- `permutation` and `shuffle` order 64-bit random sort keys instead of a
  24-bit uniform draw. A uniform carries at most 24 significant bits, so at
  60,000 elements — one MNIST epoch through `Kaun.Data` — about 107 pairs
  collided and `argsort` resolved every one of them towards the input order.
- Unscoped draws (`Nx.rand` and friends outside `Rng.run`/`Rng.with_key`) are
  seeded from system entropy and so differ from run to run, as the docs always
  claimed. They came from OCaml's default `Random` state, which is seeded
  deterministically, so they repeated exactly on every run — and would have
  started varying if any linked library called `Random.self_init`. Open a scope
  for reproducibility.
- `Rng.truncated_normal`, `Rng.categorical`, `Rng.permutation` and
  `Rng.shuffle` complete the keyed sampler set: every distribution now has a
  pure form that composes with `Rune.jit`, `vmap` and `pmap`, and each keyless
  sampler is that form applied to a subkey of the ambient scope. The keyless set
  at the top level of `Nx` is closed at the reflexive draws (`rand`, `randn`,
  `randint`, `bernoulli`, `truncated_normal`, `categorical`, `permutation`,
  `shuffle`); the distributions added since take a key, which also leaves
  `gamma` and `beta` free for the special functions of those names.
- `truncated_normal` draws by inverting the conditioned distribution instead of
  rejecting out-of-range samples. The rejection loop read its stopping
  condition back to the host, so it could not be traced or compiled at all, and
  it gave up after 1000 rounds on a narrow interval; inverting costs one draw
  per element whatever the bounds. `Kaun.Init.glorot_normal`, `he_normal` and
  `lecun_normal` go through it, so they are now compilable.
- **Breaking**: `truncated_normal` takes `~lower` and `~upper` before its dtype,
  matching the keyed form.
- **Breaking**: `randint` and `Rng.randint` take their bounds as `?low` and
  `~high` and always return `int32`, replacing the trailing positional `low`,
  the arbitrary `?high` default of `10`, and the dtype argument. The dtype
  argument accepted float dtypes and raised at run time, and the draw was
  computed in `int32` before being widened, so ranges beyond `int32` were
  silently wrong; both are now impossible. Cast the result for another integer
  width. Bounds outside `int32` raise `Invalid_argument` (#196).
- Sliding-window extraction is 10-77x faster on the C backend. The kernel used
  to divide ten times per output element; it now walks contiguous runs, so
  convolution (`correlate`, `convolve`) and the pooling filters
  (`maximum_filter`, `uniform_filter`) move at memory bandwidth instead of
  around 0.4 GB/s.
- `relu`, `sigmoid`, `clamp`, `hypot`, `tril`/`triu`, `logical_not`, and the
  boolean reductions build their constant operand as a scalar instead of a
  full-size tensor, making them 1.4-3.1x faster. Values are unchanged.
- Element-wise ops against a scalar operand are 1.4-9.5x faster. A broadcast
  0-d operand (every `_s` variant, and any operand broadcast along the innermost
  axis) used to fall onto a scalar code path; the C map, comparison, and `where`
  kernels now hoist it out of the loop and stay vectorized. Results are
  unchanged bit for bit.
- Speed up `Nx.Rng.uniform` about 2x and `Nx.Rng.normal` about 4x (4M float32
  draws: 550ms to 283ms and 1153ms to 301ms). A Threefry row yields two words
  and `uniform` discarded one of them; `normal` then drew two uniforms per
  sample and threw away the sine half of Box-Muller. Both halves are now kept.
  Values for a fixed key change.
- Fix `Nx.Rng.uniform` and `Nx.Rng.randint` returning `high`, which both
  document as excluded: the draw is now built from the random bits so it is
  half-open at every float dtype, rather than by float32 rounding that reached
  `1.0` about once in 2^24 draws (once in 4000 at float16). `Nx.rand` also
  samples at its result dtype instead of narrowing a float32 draw.
- Fix `Nx.Rng.randint` skewing towards zero for a negative `low`: it truncated
  the shifted float, so `low` was never drawn and `0` was drawn twice as often.
  Values for a fixed key change.
- **Breaking:** the stock dynamic tree is payload-generic: `Nx.Ptree.Tree` is
  a `Uniform` structure (constructors `Leaf`/`List`/`Dict`, paths from list
  positions and dict keys) and `Nx.Ptree.t` is `tensor Tree.t`. The `Tensor`
  constructor is now `Tree.Leaf`; the `tensor`/`list`/`dict` constructors and
  the rank-2 traversals are unchanged.
- Add `Nx.Ptree.instantiate`, which fills a `Ptree.Uniform` structure's payload
  hole at one tensor type as a first-class `Ptree.S` module — `let mlp =
  Ptree.instantiate (module Mlp)` — so a payload-generic tree passes to
  `Rune.grad` and the Vega optimizers with typed leaves and no packing.
- Add the payload-generic module types `Nx.Ptree.Traverse` (the traversal
  core: `map`, `map2`, `iter`) and `Nx.Ptree.Uniform` (the core plus
  `fold`/`fold2` over dot-joined leaf paths and `names`, the tree of those
  paths), and `Nx.Ptree.Make`, which turns a `Traverse` into an `Nx.Ptree.S`
  with packed tensor leaves — so one `'a params` declaration serves both the
  model and parameter-shaped data. `Nx.Ptree.unpack` recovers a typed tensor
  from a packed leaf, naming the position in the error on a dtype mismatch.
- **Breaking:** the real FFT family (`rfft`, `irfft`, `hfft`, `ihfft` and their
  2-D/N-D variants) and `fftfreq`/`rfftfreq` now take the output dtype first,
  like the constructors. It selects storage precision independent of the
  input's, so a float32 signal can keep a `complex64` spectrum.
- Add complex accessors `Nx.real`, `Nx.imag`, `Nx.magnitude`, `Nx.angle` and
  the `Nx.complex ~re ~im` constructor, taking the result dtype first so a
  `complex64` spectrum can yield a `float32` magnitude. `Nx.conjugate` now runs
  the same element-wise kernels instead of reading every element back to the
  host one at a time; it returns NaN components where the input has a
  non-finite one, which the boxed version handled exactly.
- Add `Nx.stft`, `Nx.istft`, and `Nx.hann` for short-time Fourier analysis.
  `stft` frames through a view rather than materializing, returns time-major
  `[frames; bins]`, and tapers with a periodic Hann by default; `istft`
  overlap-adds and divides by the windows' own envelope, so any
  `step <= window` inverts.
- **Breaking:** `Nx_core.Backend_intf.S` gains `sliding_window`, a pure-view
  movement producing overlapping windows. Out-of-tree engines implementing the
  `nx.backend` virtual library must add it.
- **Breaking:** consolidate tensor formatting around the compact `Nx.pp`,
  `Nx.to_string`, and `Nx.print`. Remove `pp_data`, `data_to_string`,
  `print_data`, `format_to_string`, `print_with_formatter`, `dtype_to_string`,
  and `shape_to_string`; add `Nx.pp_shape` alongside `Nx.pp_dtype`.
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

- `Rune.vmap` now maps `Nx.matmul` correctly when the other operand carries
  leading batch dimensions of its own: the map's axis was aligned against the
  operand's first batch dimension, which raised a shape mismatch or silently
  paired the wrong matrices.
- `grad` through `Nx.solve_triangular ~unit_diag:true` no longer assigns a
  gradient to the diagonal, which the solve never reads; it disagreed with
  both the function and `jvp` there.
- `grad` and `jvp` through a batch of matrices are now correct for
  `Nx.cholesky`, `Nx.qr`, and `Nx.solve_triangular`: the rules reversed every
  axis and extracted, rather than built, their diagonal terms, so a stack of
  matrices gave wrong gradients or a shape error.
- `Rune.jit` compiles `Nx.qr`, `Nx.solve_triangular`, `Nx.cholesky`,
  `Nx.solve`, and `Nx.inv`: the factorizations unroll at trace time into the
  fixed number of steps their shapes imply (see `Tolk_frontend.Linalg`), and
  `grad` through them compiles as well. A singular or non-positive-definite
  input yields infinities or nans in the compiled program rather than an error.

- `Rune.jit` compiles `Nx.qr` and `triangular_solve` — Householder QR and
  forward substitution unrolled at trace time into the fixed number of steps
  their shapes imply, so the whole factorization lowers to ordinary Tolk
  compositions and compiles for every Tolk device. Compiled results match the
  eager kernels, including the LAPACK reflector sign and the zero-tail
  no-reflector convention. A linear solve inside jit is the same composition
  written out by hand (`Nx.solve` itself still refuses to trace, because its
  singularity check reads a traced value); a singular system yields infinities
  rather than an error. Wide right-hand sides (nrhs ≥ 32, n > 64) solve
  block-by-block — 32-row blocks, diagonal blocks inverted once, one GEMM per
  block against the rows solved so far — instead of unrolling one thin matmul
  per row, which cuts the compiled solve's O(n²·nrhs) concatenation copying to
  O(n²·nrhs/32): replay at 256×256 drops ~80× (58 ms to 0.7 ms) and the
  compiled solve now beats the eager C kernel 3-5× from 128×128 up. Compile
  time grows linearly in the matrix dimension, and `grad` inside a jitted
  function now also differentiates through `qr` and `cholesky`: the tape
  pullbacks' `diag` use read host bytes and refused to trace, so the QR and
  Cholesky pullbacks (and the triangular-solve JVP rule) form the diagonal
  terms from the identity instead.

- **Breaking:** when `~device` is omitted, `Rune.jit`, `jit2`, and `jit'` now
  run on the best available backend — the `DEV` environment variable selects
  one by name, otherwise METAL, AMD, NV, CUDA are probed in order with CPU as
  the fallback — instead of always CPU. Pass `~device:"CPU"` or set `DEV=CPU`
  to keep the old behavior.

- `Rune.jit` and `Rune.pmap` accept `~device:"NV"` — NVIDIA GPUs driven on
  the kernel driver's hardware queues (Linux), with kernels compiled straight
  to cubin. `"CUDA"` keeps selecting the userspace CUDA driver API backend.

- `Rune.jit` and `Rune.pmap` accept `~device:"AMD"` (`"AMD:n"` for a specific
  GPU), running compiled programs on AMD GPUs on Linux. Previously the device
  factory rejected the name with `Invalid_argument: unknown device AMD:0`
  even though the runtime had landed; an AMD device that cannot open now
  reports `device AMD:0 unavailable` with the reason, like CUDA.

- `Rune.jit` compiles `Rune.scan` as a loop in the compiled program — the fold
  step compiles once and runs per slice — instead of unrolling every step into
  the trace, and `grad` through a jitted scan compiles a reversed loop over the
  step's pullback. Compile time for a scanned recurrence (an RNN, a sampling
  loop) no longer grows with its sequence length. A carry that changes shape
  across steps, and a scan reached through `vmap` or `pmap`, unroll into the
  trace as before.

- `grad`, `vjp`, and `jvp` now differentiate `Nx.rfft` and `Nx.irfft` — both
  are linear, so each rule is an exact transpose — and `vmap` batches all four
  FFT transforms (`fft`, `ifft`, `rfft`, `irfft`), so spectral losses built on
  real FFTs train end to end.
- `jit`, `jit2`, `jit'`, `pmap`, and `pmap2` take `?beam_parallel`: the
  number of domains compiling a beam-search round's candidates, scoping the
  `BEAM_PARALLEL` setting to one compiled function. It only changes compile
  time, never the compiled code.

- `jit`, `jit2`, `jit'`, `pmap`, and `pmap2` take `?beam`: beam-search
  autotuning of the compiled function's kernels, equivalent to compiling under
  `BEAM=n` but scoped to that one function. The width is part of the
  persistent compile-cache key, so tuned and untuned compilations of the same
  trace do not collide.
- A parameter structure may now hold leaves that are not parameters — an
  `Nx.Rng.key` threaded through a compiled step, a step counter, a batch of
  indices. `grad`, `value_and_grad` and `vjp` carry them instead of raising:
  they are not tracked, and their slot in the gradient structure holds zeros.
  One structure can therefore serve both `grad` and `jit`, where before a key
  had to be captured in a closure and the parameters kept in a second structure
  — the workaround the GPT-2 example spells out across four hand-written ptree
  modules. The single-tensor `grad'`/`vjp'` still reject an integer argument,
  having nowhere to carry one.

- Gradient rules that combine with a constant (`asin`, `atan`, `tanh`, `sqrt`,
  `pow`, `max`/`min`, `where`, `cholesky`, `cumprod`) no longer materialize a
  full-size tensor of ones or zeros to do it.
- Fix the gradient of `Nx.fft` and `Nx.ifft`: reverse mode pulled cotangents
  back through the opposite transform, which reversed the frequency index. Each
  transform is its own transpose and now pulls back through itself.
- Fix the derivative of `abs` on complex tensors for cotangents that are not
  real: the modulus is real-valued, so the rule now keeps only the real part of
  the cotangent and pushes forward to a real tangent.
- The transformations take their parameter-tree module as
  `(module Ptree.S with type t = 'p)` instead of a dependent module argument,
  so a first-class module value — e.g. the result of `Nx.Ptree.instantiate` —
  can be bound once and passed to `grad`, `jit`, `vjp`, and friends.
- Fix the derivative of `abs` on complex tensors, in both forward and reverse
  mode: it pulled back through `sign z` instead of its conjugate, which negated
  the imaginary part's contribution. Gradients of a real-valued function that
  passes through a complex magnitude were wrong; real dtypes are unaffected.
- Reverse mode differentiates the sliding-window movement through `fold`
  rather than a materialized scatter, so the backward pass is a single
  overlap-add instead of allocating a `window`-fold copy of the cotangent.
- `vmap` now batches `Nx.extract_patches` and `Nx.combine_patches` instead of
  raising; both preserve leading dimensions, so the batch axis passes through.
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

- The GPT-2, Llama and gpt-oss examples' importers and cache builders take
  `?placement : role -> axis:int -> Nx.Placement.t` and place each leaf and
  cache pool with it as they build it, naming each leaf's tensor-parallel cut.
- `Metric` functions read placed predictions, labels and scores to the host
  once and compute there. They ran on the device and placed every
  intermediate, and raised on Metal, which cannot hold the float64 they sum.
- Add `Cache_index.every m index` for layers that keep one entry per block of
  `m` positions, such as a compressed key: `make ~every` and `rows ~every` give
  the index a table of blocks, and a block is stored by its last token and seen
  from that token on.
- Add `Cache_index.select columns index`: each token reads only the columns it
  chose, so `extend` returns `[batch; seq; k; ...]` rows and a sparse or
  windowed layer's read costs `k` rows per token whatever the context.
- `Kaun_hf.load_checkpoint` no longer asks the Hub for a shard index when the
  repository is already cached as a single `model.safetensors`: a cached model
  used to make one network request on every start, and to stall without a
  connection. `Kaun_hf.download_file` now runs `curl` directly instead of
  through a shell: the previous detection was a POSIX shell command, which
  `cmd.exe` cannot run, so downloads failed on Windows with "curl not found".
- The gpt-oss example takes text: `--prompt` (with `--system`, `--reasoning`
  and `--show-analysis`) renders a harmony conversation with the checkpoint's
  tokenizer, streams the model's final answer as it decodes and stops when the
  model closes its turn. Its `Harmony` module renders and parses the format,
  checked against `openai-harmony` by `validate_text.exe`.
- Remove `Kaun_hf.rename`, `transpose` and `split`. They existed because a
  template found entries by its own paths; an importer now asks for each entry
  by the file's name, so a rename is the name at the field, a transpose is
  `Nx.matrix_transpose` and a fused tensor is `Nx.split ~axis n`, all views.
- Add `Checkpoint.to_tensor ~shape dtype name` and `Checkpoint.to_float ~shape
  dtype name`, which read one entry by name and check its shape. A model's
  importer is now an ordinary function that builds the parameter record from
  them, so no template is allocated, leaves may have different dtypes, and a
  wrong configuration fails at import with the entry's name. `to_tensor` is
  strict and returns the entry as stored, a view of the file; `to_float` casts
  between `float16`, `bfloat16`, `float32` and `float64` and refuses anything
  else.
- `Checkpoint.to_params` and `to_packed` lose `?cast` and raise on any dtype
  mismatch, so a restart that names the wrong dtype fails instead of narrowing
  its state. To convert, read the entry with `Checkpoint.to_float`.
- `Checkpoint.load` and `Kaun_hf.load_checkpoint` map their files, as
  `Nx_io.load_safetensors` now does: loading reads headers only, entries are
  views of the file, and entries whose dtype nx lacks arrive as `uint8` bytes
  instead of being skipped. A loaded file must not be modified in place while
  its entries are alive; `Checkpoint.save` replaces its destination atomically.
- `Kaun_hf.download_file` downloads to a uniquely named temporary file beside
  the cache path and renames it once complete. An interrupted download used to
  leave a partial file at the cache path, which later runs served as cached,
  and two processes fetching one file wrote over each other.
  `Kaun_hf.clear_cache` runs a major collection and retries once when a file
  cannot be removed.
- The pieces `Attention.apply` and `Attention.cached` are made of are public:
  `Attention.split` projects and splits into heads, `Attention.attend` is
  grouped-query attention with `?mask`, `?scale` and `?sinks`,
  `Attention.merge` concatenates heads through the output projection, and
  `Attention.Cache.extend` stores a call's keys and values and returns what its
  tokens attend over. A model with its own attention variant composes a layer
  in a dozen lines. `apply` and `cached` compute what they did, bit for bit.
- `Attention.scaled_dot_product_attention` takes `?scale`, which replaces
  `1 / sqrt d`, and `?sinks`, attention-sink logits that join each query's
  softmax as one more key of value zero, as gpt-oss needs. With sinks a query
  that sees no key yields zero. Without the options the computation is
  unchanged.
- Add `Rope.of_frequencies`, a schedule from one head's inverse frequencies,
  for schedules the module does not name, and `Rope.yarn`, the YaRN
  long-context frequencies with an untruncated correction range, as gpt-oss
  uses them. `Rope.apply` keeps norms: YaRN's attention temperature is a
  number the model passes to the attention core, not part of the schedule.
- **Breaking**: the sampling masks `Fn.top_k` and `Fn.top_p` are now
  `Fn.keep_top_k` and `Fn.keep_top_p`. They return the logits with everything
  outside the kept set at negative infinity, where `Nx.top_k` returns the `k`
  greatest entries: one name, one meaning.
- Add `Cache_index.pool ~slots dtype shape`, an empty pool of `slots` slots of
  shape `shape`: the one place that knows the scratch row. A layer with its
  own cache record builds its leaves with it, as `Attention.Cache.make` does.
- **Breaking**: a sliding window is part of the cache index.
  `Cache_index.window w index` is `index` seeing the last `w` positions, and
  `Cache_index.extend` and `Cache_index.mask` lose `?window` and read the
  index's, so a layer can no longer zero with one window and mask with
  another. `Attention.cached` loses `?window` too: a layer with a window is
  `cached p cache (Cache_index.window w index) x`.
- `Attention.apply` and `Attention.cached` no longer copy the keys and values
  to group them under their query heads. The decode step of
  `kaun/bench/decode` runs 24 fewer kernels and allocates 9% fewer host words;
  on Metal it takes 8.0 ms against 8.5 ms at a cache of 256, and the same 8.9
  ms at 1024.
- **Breaking**: a decoder has one forward pass. `Attention.cached` takes a
  `Kaun.Cache_index.t`; over `Cache_index.whole`, which
  reads and keeps nothing, it is plain causal attention and returns its cache
  untouched. A model's `hidden` is
  `fst (cached ... (Cache_index.whole ~batch ~seq ()) ids)`, the second fold
  over `Attention.apply` is gone from the `04-gpt2` and `05-llama` examples,
  and its compiled gradient costs what that fold did.
- **Breaking**: `Kaun.Cache_index` replaces `Attention.Span`, the `route` type
  and `Attention.route ~slots`; a model resolves nothing. A cache index is
  opaque: `Cache_index.make ?row ~pos ~table ()` takes the tokens' positions
  and the slots holding each sequence, and `?row` names the sequence of each
  lane. A token stores at the slot its table names at its position.
  `Cache_index.rows`, `advance` and `positions` keep the meaning they had on
  `Span`. A layer calls `Cache_index.extend index values pool`, which stores
  the call's values and returns what its tokens attend over, and attends under
  `Cache_index.mask`.
- **Breaking**: `Attention.Cache.make ~slots` allocates `slots + 1` rows. `-1`
  addresses nothing everywhere, and the last row is a scratch row that
  receives such writes and is never observed, so the cache write is one
  `Nx.scatter ~unique_indices:true` over the call's tokens with no pass over
  the pool. Under `Rune.jit ~donate:true` on Metal a two-layer decode step at
  a context of 256 takes 3.6 ms over 4096 slots and 3.8 ms over 131072, where
  it took 3.8 ms and 24.6 ms; the GPT-2 124M shaped step of
  `kaun/bench/decode` takes 8.4 ms and 8.7 ms at caches of 256 and 1024
  against 9.1 ms and 10.3 ms.
- Attention is total. A query whose mask hides every key yields zero from
  `Attention.scaled_dot_product_attention`, `Attention.apply` and
  `Attention.cached`, with zero gradients, where it yielded `nan`.
- **Breaking**: `Attention.causal_mask ~valid` no longer keeps the diagonal of
  a padded query. The kept key existed to avoid that `nan`; a padded query's
  output is now the projection of zero.
- The `05-llama` example's `validate` checks the residual stream after every
  block, a ragged batch through the key-value caches, half precision
  (`--dtype`) and compiled runs (`--jit`), and ships a second fixture,
  TinyLlama 1.1B, which covers an untied head and the standard rotary
  schedule on real weights.
- New example `05-llama`: Llama 3.2 1B written on the decode contract
  (grouped-query attention, rotary positions with the Llama 3 schedule,
  RMS norm, SwiGLU), loaded from an ungated mirror whose weights are
  byte-identical to Meta's, with sampled generation through key-value caches
  and a `validate` program that checks the import against the reference
  implementation's float32 logits, block by block.
- `Kaun.Fn.keep_top_k` and `Kaun.Fn.keep_top_p` mask next-token logits for
  sampling: entries outside the kept set become negative infinity and the
  shape is unchanged, so they compose with a temperature division and
  `Nx.Rng.categorical` in any order and compile. `k` and `p` are tensors, a
  scalar or one entry per row, so a batch can mix requests.
- `Loss.softmax_cross_entropy` and `softmax_cross_entropy_sparse` compute
  their log-probabilities and reduction in a float32 island for half and
  quarter precision logits and cast the result back: a bfloat16 log-sum-exp
  over a large vocabulary biases the gradient.
- **Breaking**: cached decoding is addressed by positions and slots.
  `Attention.cached ~head_dim ?rope p cache route x` replaces `apply_cached`
  and its single scalar position. A cache is a flat pool of slots with no
  batch axis, so `Attention.Cache.make ~slots ~kv_heads ~head_dim` replaces
  `?batch ~num_heads ~head_dim ~len`, and `Attention.Cache.List` traverses a
  model's per-block caches. An `Attention.Span.t` (`make`, `rows`, `advance`,
  `positions`) carries each token's position and the slot holding each
  position of each row's sequence, and `Attention.route ~slots span` resolves
  it once per call for every block. Rows at different positions share a
  batch; paging, shared prefixes and beams are values of the slot map rather
  than cache types; an address outside its range addresses nothing (`-1` is
  padding); a prompt fed whole or in chunks gives the same outputs; and
  admitting a sequence changes values, never the compiled program. See RFC
  0002.
- **Breaking**: `Attention.apply` requires `~head_dim` and takes `?mask` in
  place of `?num_heads` and `?causal:bool`, so causality and padding
  intersect; `Attention.causal_mask ~seq ?valid ()` builds the mask and keeps
  the diagonal, so a padded query never yields `nan`. `?rope` rotates queries
  and keys, and `Attention.make ?q_dim ?kv_dim` sizes the projections for
  grouped-query attention, where the keys broadcast over their group.
- `Kaun.Rope` adds rotary position embeddings: a schedule is the inverse
  frequencies of one head (`Rope.make`, and `Rope.llama3` for the Llama 3.1
  long-context bands) and `Rope.apply` rotates queries or keys at per-token
  positions, forming the angles at float32 whatever the activation dtype.
- `Kaun.Rms_norm` adds root mean square normalization, the norm of
  Llama-class models, with the float32 island `Layer_norm` has for half and
  quarter precision inputs.
- Seeded `glorot_normal`, `he_normal` and `lecun_normal` initialisers produce
  different values for a given key: they draw through
  `Nx.truncated_normal`, whose bounds are now tensors and whose draw changed
  with them. `Dropout` masks are unchanged.
- `Loss.huber`, `Loss.sigmoid_bce`, and `Fn.leaky_relu` compare against a
  scalar rather than a materialized constant tensor.
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

### Brot

- OLMo 2 and Phi-4 take the cl100k scanner too: they ask for the pattern's
  matches as `Removed` with `invert`, which is now recognised as the same
  pieces (Phi-4 24 → 45 MB/s single-threaded; not through the fused kernel).
- Llama 3, GPT-4 (cl100k), Qwen2 and Qwen3.5 tokenizers encode through the
  fused C kernel that GPT-2 uses, with a walker for their pattern: Llama 3
  43 → 79 MB/s and Qwen2.5 40 → 90 MB/s single-threaded on OpenWebText
  (GPT-2 is unchanged at 155 MB/s).
- Llama 3, OLMo, GPT-4 (cl100k), Qwen2 and Qwen3.5 tokenizers pre-tokenize
  about twice as fast: their `Split` patterns are recognised and run by a
  scanner checked against the regular expression (Llama 3 27 → 43 MB/s
  single-threaded). Such a pipeline can also be cut across domains by
  `encode_batch_ids`. `Pre_tokenizer.pp` shows `walker=cl100k(...)` for them.
- Tokenizer files whose `Split` pre-tokenizer carries a regular expression now
  load and tokenize as HuggingFace does: Llama 3, Qwen2.5, DeepSeek-V3 and
  gpt-oss (o200k) where `Brot.from_file` used to fail with "regular expression
  'pattern' is not supported". `Pre_tokenizer.split_regex` builds one directly.
- Regular expressions from tokenizer files now accept the case-insensitive
  option (`(?i)`, `(?i:..)`) and a lookahead (`(?=..)`, `(?!..)`) that ends the
  pattern or one of its alternatives, so `Normalizer.replace_regex` and
  `Replace` normalizers using them load instead of being rejected.
- **Breaking:** the stage modules' internal plumbing is no longer exported:
  `Brot` is now published from a single signature, so
  `Pre_tokenizer.plan`/`fill`/`lead_class`,
  `Encoding.token`/`of_run`/`with_overflowing` and `Post_processor.affixes`
  are gone from the public API. The documented API is unchanged.
- `encode_batch_ids` is ~3–5% faster: the fused C kernels now write each
  chunk's ids straight into its int32 result buffer instead of filling an
  int buffer that was copied per document (GPT-2 batch 179 → 186 MB/s
  single-threaded, 810 → 845 MB/s across domains; Mistral 58 → 60 MB/s).
- SentencePiece-style BPE tokenizers (Llama 1/2, Mistral, Gemma 2) encode
  another ~1.2–1.35× faster on native code (llama 33 → 40 MB/s, mistral
  44 → 58 single-threaded): the `▁`/punctuation unit walk, the pretoken-cache
  probe and the short merge run fused in the C kernel, as the byte-level path
  already does. Bytecode and js_of_ocaml keep the OCaml path.
- SentencePiece word units also end before the eight frequent punctuation
  bytes no vocabulary piece or merge reaches across, shrinking the distinct
  units the pretoken cache holds: another ~1.3× on Llama and Mistral. The
  same split applies to any non-byte-level BPE vocabulary that passes the
  safety scan, `▁`-free ones included, on pretokens longer than a cache key.
- SentencePiece-style BPE tokenizers (Llama 1/2, Mistral, Gemma 2) encode
  4–5× faster: when a creation-time vocabulary scan proves no piece or merge
  can cross a `▁`-opened word boundary, `encode`/`encode_ids`/
  `encode_batch_ids` cut a whole-document span into `▁`-run word units that
  take the pretoken cache and the linear merge instead of one whole-document
  heap merge. Ids, offsets and tokens are unchanged; a vocabulary that fails
  the scan (Gemma 3) keeps the previous path.
- Honour a `stride` when truncation overflows: `truncation` gains a `stride`
  field and `?stride` argument, successive overflow windows overlap by it,
  and windows now match HuggingFace exactly — a pair's windows are the same
  combinations `tokenizers` produces (they were dropped before), and windows
  cover the tokenized pretokens rather than the whole excess, since
  HuggingFace stops tokenizing once truncation's `max_length` is reached.
- `Encoding.truncate`'s `~stride` and `~direction` are now optional,
  defaulting to `0` and `` `Right``; a stride at or past `max_length` raises
  `Invalid_argument` when the encoding is actually truncated.
- **Breaking:** Tidy the public API: `encode_pairs_batch` is now
  `encode_batch_pairs`, `from_json` is `of_json` and `train_wordlevel` is
  `train_word_level`; `Pre_tokenizer.whitespace`, `whitespace_split`, `bert`,
  `unicode_scripts` and `Decoder.byte_level`, `byte_fallback`, `fuse` are
  plain values; `Post_processor.bert` drops its `unit`; `Encoding.concat`
  takes a list, replacing `concat_list`; `save_model_files` is
  `?prefix -> t -> folder:string -> string list`, dropping its `unit`.
- **Breaking:** Remove the trainer options that did nothing: `?show_progress`
  on all four trainers, and `?shrinking_factor`, `?max_piece_length` and
  `?n_sub_iterations` on `train_unigram`.
- `Encoding.create` now validates that its seven arrays share one length and
  raises `Invalid_argument`, instead of crashing later on mismatched arrays.
- Add `Encoding.pp`, formatting an encoding as a table of tokens, ids,
  offsets, word ids and masks.
- The BPE pretoken cache gained a small resident front table (128 KB,
  direct-mapped, filled by promoting the main table's hits) probed before the
  8 MB main table, cutting its memory traffic on large corpora.
  `cache_capacity 0` disables both tables.
- Faster GPT-2 byte-level encoding: the C kernel now classifies text in
  64-byte batches (NEON on arm64, SWAR elsewhere) and derives pretoken
  boundaries with bitmask algebra, walking only non-ASCII neighbourhoods and
  batch edges byte by byte. `encode_ids` on wiki-64k drops ~12.7 to
  ~10.3 ns/pretoken; single-domain OpenWebText throughput rises ~10%.
- Faster byte-level batch encoding on real text: the C kernel's
  pretoken-cache probe is software-pipelined, fetching each span's cache line
  while the next span is walked — ~12% faster single-domain and ~7%
  multi-domain `encode_batch_ids` (GPT-2/RoBERTa over OpenWebText), at a
  small cost on single small documents.
- Tokenizers with added tokens configured encode faster: the scan for
  added-token occurrences now seeks candidate first bytes a word at a time
  instead of probing a table per byte, lifting single-domain GPT-2
  `encode_batch_ids` from ~99 to ~122 MB/s, with similar gains for RoBERTa and
  BERT.
- Tokenizers with added tokens configured encode faster: the scan for
  added-token occurrences now seeks candidate first bytes a word at a time
  instead of probing a table per byte, lifting single-domain GPT-2
  `encode_batch_ids` from ~99 to ~122 MB/s, with similar gains for RoBERTa and
  BERT.
- On native code, byte-level BPE (GPT-2, RoBERTa) now encodes through a fused
  C kernel: the byte-level pattern walk, the pretoken-cache probe and the
  merging of pretokens up to 15 bytes run in one pass over the text, roughly
  halving `encode_ids`/`encode_batch_ids` time on English text. Results are
  identical to the pure-OCaml path, which bytecode and js_of_ocaml keep using.
- BPE's pretoken cache is now two-way set-associative in the same 8 MB: a
  colliding pair of pretokens no longer evict each other, cutting the miss
  rate from 7.9% to 4.8% on real text (`encode_batch_ids` on OpenWebText runs
  ~9% faster). `cache_capacity` keeps its meaning — entries, rounded up to a
  power of two, `0` disables.
- BERT-style normalization (`Normalizer.bert`) runs as one pass with an ASCII
  fast lane: 24× faster on English text (18 → 440 MB/s), 2–3× on non-Latin
  scripts, identical output and offsets. `Brot.encode` with offsets on
  bert-base goes from 14.7 to 3.4 ms per 64 KB document, `encode_ids` from
  5.0 to 1.6.
- Unicode normalization (`nfc`, `nfd`, `nfkc`, `nfkd`) is streamed with a fast
  lane for ASCII and for characters already in normal form: 6–7× faster on
  mostly-ASCII text and 10–40% faster on Cyrillic, Greek, Vietnamese, Korean,
  Arabic and CJK; `lowercase`, `strip_accents`, `nmt` and `strip` are 3–5×
  faster; `apply_aligned` costs 1.3–1.5× `apply` instead of 2–7×, so offsets
  on normalized text are 3–35× cheaper, and `prepend`/`strip` track
  alignments for free.
- Fixed `apply_aligned` under `nfc`/`nfkc` composing an LV Hangul syllable
  with the following U+11C3 (a plain starter, not a trailing jamo); `apply`
  and HuggingFace never did.
- Offsets are exact through pre-tokenizer sequences that rewrite after
  splitting (`Sequence [WhitespaceSplit; Metaspace]` as in T5/ALBERT/XLNet,
  `Sequence [Split; ByteLevel]`): each token now reports its own bytes instead
  of the whole word's, and `Pre_tokenizer.pre_tokenize` places the pieces of
  later members exactly. Such pipelines also encode faster (T5 `encode_ids`
  ~1.6×) and take part in cut-document parallel batches.
- `Pre_tokenizer.metaspace ~prepend_scheme:`First` is honoured: the marker is
  prepended only to the piece that opens the document — not after white
  space, an added token, or bytes a normalizer removed — as HuggingFace does
  (`Always` was used before).
- Offsets of unknown tokens and byte-fallback runs match HuggingFace: a fused
  unknown run and every byte token of a fallback run stand for the run's
  bytes, so the tokens after them are no longer shifted (`Encoding.offsets`
  on Unigram/BPE models).
- A Unigram, WordPiece or WordLevel model behind a byte-level pre-tokenizer
  (`Pre_tokenizer.byte_level`, alone or in a sequence) is now handed
  byte-level-encoded pieces and matches its vocabulary in that form, as
  HuggingFace does; before, its ids were wrong.
- Add `Brot.encode_batch_ids`, the throughput path: the ids of a whole batch
  in one `int32` Bigarray (`Brot.ids`) plus per-row lengths, straight into
  `Nx` via `Nx.of_bigarray`; no `Encoding.t` and nothing allocated per token,
  and a long text is spread over domains when its pipeline allows. 7–8× over
  one domain on 8+ cores.
- `Brot.encode_batch` and `Brot.encode_pairs_batch` take `?domains` and split
  work by bytes rather than by document count, so one long text no longer
  leaves the other domains idle and small batches no longer pay for spawning
  domains (a 32-document batch went from 3.7 ms to 0.6 ms).
- Unigram tokenization now finds the segmentation whose scores add up to the
  most, as SentencePiece and HuggingFace do, instead of the longest match at
  each position, which mis-cut e.g. `traces` as `▁trace`+`s`; a run of
  characters the vocabulary does not hold is one unknown token, or its bytes
  under `byte_fallback`, and white space inside a pretoken is no longer
  dropped. `Brot.unigram` gains `?unk_id` and `?byte_fallback` (`unk_token`
  names the unknown entry when the vocabulary holds it; without one an
  uncovered character raises `Failure`); T5-style `tokenizer.json` (`unk_id`,
  `byte_fallback`) loads and saves faithfully.
- Unigram encoding is at least as fast as the greedy encoder it replaces
  (about 1.2×) and model loading about twice as fast, through a double-array
  trie.
- `Decoder.byte_level` now decodes as HuggingFace does: a token whose
  characters are not all in the byte-level alphabet stands for its own bytes as
  a whole rather than character by character, and the bytes of every token are
  read as one text. A character spelled across two tokens now decodes, and
  every maximal ill-formed byte sequence becomes one `U+FFFD` instead of being
  returned as invalid UTF-8.
- `Pre_tokenizer.pre_tokenize` now reports offsets into the text it was given
  for every pre-tokenizer. A `metaspace`, alone or in a `sequence`, reported
  offsets into the marked text, which could run past the end of the input —
  `Sequence [WhitespaceSplit; Metaspace]` placed the second piece of
  `"Hello world"` at `(6, 14)` instead of `(6, 11)`.
- `Pre_tokenizer.metaspace ~split:false` now gives every token exact byte
  offsets. Its pre-tokenizer joined the walking path, where before it fell
  back to whole pieces and every token of a document reported the document's
  span.
- `Encoding.offsets` reports byte spans of the text as it was passed in rather
  than of the normalized text: a token of `"café"` under an accent-stripping
  normalizer spans the accented bytes. They were in normalized coordinates
  before, which for BERT- and LLaMA-style pipelines pointed at the wrong bytes.
  Reading `offsets` on a pipeline with a normalizer costs a second
  normalization pass; reading only `ids` costs none.
- `Encoding.word_ids` is `Some` for every content token, numbering the
  pretokens of a sequence from `0`, an added token counting as one. It was
  `None` throughout.
- `Encoding.tokens`, `Encoding.offsets` and `Encoding.word_ids` are worked out
  when first read, so `encode` costs no more than `encode_ids` for a caller
  that only wants the ids.
- `Brot.encode` truncates before the post-processor runs, on a budget of
  `max_length` minus the special tokens it will add, and a pair gives up
  tokens the way HuggingFace's `LongestFirst` does. Truncation ran on the
  finished encoding before, so special tokens pushed content past
  `max_length`.
- `Brot.encode` truncates from the left by keeping the last `max_length`
  tokens, matching HuggingFace; it previously kept the first and left the rest
  in `overflowing`.
- `Brot.encode_ids` is ~5.9× faster on a 64 KB GPT-2 document and allocates
  about a thousandth of what it did (330 → 56 ns per pretoken, 778k → 0.9k
  minor words); `Encoding.with_type_id`, `Encoding.with_overflowing` and
  `Post_processor.affixes` are new.
- **Breaking:** `Normalizer.replace ~pattern ~replacement` now replaces a
  literal string with a plain scan (2.3× faster and 2.2× less allocation than
  through the regex engine). Regular expressions move to the new
  `Normalizer.replace_regex`, which reads the Unicode-aware dialect of
  tokenizer files (`\s`, `\d`, `\w`, `\p{..}` by short or long general
  category name; `.` and negated classes match characters, not bytes) and
  rejects unsupported constructs (`(?i)`, lookaround, backreferences, `\b`,
  ...) with a message saying which.
- **Breaking:** `Normalizer.byte_level` is a plain value: it dropped
  `add_prefix_space`, which HuggingFace's `ByteLevel` normalizer has no field
  for, so the JSON is canonical and the behaviour identical.
- Add `Normalizer.nmt`, the `Nmt` control-character cleanup, matching
  HuggingFace character for character.
- Normalizer JSON now round-trips `Replace` patterns written as `{"String":..}`
  or `{"Regex":..}`, `{"type":"ByteLevel"}` and `{"type":"Nmt"}`, so
  CLIP-style tokenizer files load; an unsupported regex is reported as
  `invalid regular expression "...": <why>`.
- Fix regex `Replace` splitting a multibyte character when stepping over an
  empty match: empty matches now advance by whole characters and one right
  after a match is skipped, as HuggingFace does.
- `Pre_tokenizer.punctuation ~behavior:`Merged_with_previous`` and
  `Pre_tokenizer.split ~behavior:`Contiguous`` returned the whole text as one
  piece instead of splitting it. They now match HuggingFace: the first keeps
  each delimiter with the text before it, the second keeps neighbours that are
  both delimiters, or both not, as one piece.
- `Pre_tokenizer.split ~invert:true` treated every byte outside the pattern as
  a delimiter of its own. It now inverts whole segments as HuggingFace does, so
  `~pattern:","` inverted splits `"a,,b"` into the two commas rather than into
  single bytes.
- `Pre_tokenizer.split ~pattern:""` returned the whole text. An empty pattern
  now makes every character a piece, and no pieces at all with `~invert:true
  ~behavior:`Removed``.
- `Post_processor.process` keeps both sequences of a pair when
  `~add_special_tokens:false`. It used to return only the first, silently
  dropping the second sentence. The second sequence gets type ID `1`, except
  under `roberta`, which has a single segment and puts every type ID at `0`
  with or without special tokens.
- A `template` post-processor applies its template when
  `~add_special_tokens:false`, dropping only the special pieces, so the order
  and type IDs a pair template assigns to `$A` and `$B` are honoured. Building
  the processed encoding is 8× faster.
- `train_bpe`, `train_wordpiece`, `train_wordlevel` and `train_unigram` count
  the pre-tokens their own pipeline produces: every text goes through
  `?normalizer` then `?pre`, and each piece is one word, as in HuggingFace.
  Training used to split on spaces whatever the pipeline was, so a byte-level
  model learned merges over words it would never meet. With no `?pre` a whole
  text is one word — pass `~pre:(Pre_tokenizer.whitespace_split ())` for the
  old behaviour. Training from `` `Files`` keeps each line's newline, as
  HuggingFace does, so a byte-level model trained from a file learns a token
  for it and blank lines count as a `"\n"` word.
- `initial_alphabet` entries are code points, not bytes:
  `train_bpe ~initial_alphabet:["é"]` now puts `é` in the vocabulary instead
  of the raw byte `"\xc3"`. Each string contributes the code point it starts
  with; empty or invalid entries are dropped.
- `train_bpe` and `train_wordpiece` no longer cap the alphabet at 1000
  characters when `limit_alphabet` is omitted, matching HuggingFace and the
  documented default of keeping every character.
- `train_wordlevel` numbers words after the special tokens instead of reusing
  ids `0..n-1` for both, which produced a vocabulary with two tokens per id;
  special tokens now count against `vocab_size`, as in HuggingFace.
- The `train_*` documentation now matches the code: `init` carries added and
  special tokens but never the model, `show_progress` displays nothing,
  `max_token_length` counts characters and holds a merge back once the joined
  run reaches it, and `train_unigram` states that its EM training is not
  implemented.
- `save_pretrained` writes a post-processor HuggingFace can read. The
  `ByteLevel` post-processor was missing `add_prefix_space`, which HuggingFace
  requires alongside `trim_offsets`, so a saved GPT-2 tokenizer failed to load
  at all; `TemplateProcessing` wrote `"pair": null`, which HuggingFace also
  rejects, and now writes the pair template.
- `Post_processor.roberta` honours `trim_offsets` and `add_prefix_space`, which
  it stored and ignored, and `Post_processor.byte_level` takes
  `?add_prefix_space`. Trimming now matches HuggingFace: it counts the space
  marker and whitespace in the encoded token, so a byte-level encoded tab or
  newline keeps its offsets, and a token that is only whitespace loses both
  ends.
- `Post_processor.template` without `~pair` uses HuggingFace's default
  `$A:0 $B:1` instead of raising when a pair is processed, `to_json` writes it,
  and `added_tokens ~is_pair:true` counts that pair rather than the single
  template's special tokens.
- Pre-tokenizers walk byte spans instead of building intermediate pieces:
  `Pre_tokenizer.pre_tokenize` is 1.2–2.1× faster and the byte-level (GPT-2)
  split runs at ~245 MB/s allocating nothing.
- `Pre_tokenizer.metaspace`'s `?replacement` is a `string` defaulting to `"▁"`
  (U+2581), which a `char` could not hold, and must be exactly one character.
  The marker is prepended only when the marked text does not already start
  with one, and `~split:false` reports the offsets of the text as given — both
  matching HuggingFace, which brot diverged from on text already containing the
  marker.
- `Pre_tokenizer.char_delimiter` takes a `string` of one character, so a
  multi-byte delimiter such as `"▁"` now works; HuggingFace's
  `CharDelimiterSplit` allows one.
- `Pre_tokenizer.pre_tokenize` returns no piece for an empty text, as
  HuggingFace does; `Byte_level ~use_regex:false`, `metaspace ~split:false` and
  `split ~pattern:""` used to return one empty piece.
- `Pre_tokenizer.pre_tokenize` no longer raises or reads past the input on
  malformed UTF-8: a truncated sequence, a byte that cannot lead one, and a
  surrogate encoding are each one byte of no category. It used to raise
  `Invalid_argument` on WTF-8 input.
- `Pre_tokenizer.split ~behavior:`Merged_with_previous`` reports the offsets of
  a delimiter that follows another delimiter instead of repeating the previous
  piece's; this matches HuggingFace.
- `Pre_tokenizer.to_json` writes a `Split` pattern as `{"String": …}` and a
  `Metaspace` `prepend_scheme` in lower case, the shapes HuggingFace requires;
  it used to write a bare string and `"Always"`, which HuggingFace refused to
  load. `of_json` reads both, defaults `prepend_scheme`, `split` and
  `Punctuation`'s `behavior` when absent, and reports a clear error for a
  `{"Regex": …}` pattern, which has no equivalent in brot.
- `Normalizer.to_json` writes `BertNormalizer`, the type name HuggingFace uses,
  so a saved tokenizer round-trips unchanged; it previously wrote `Bert`, which
  HuggingFace accepts but rewrites. Reading a `Strip` normalizer with a missing
  `strip_left` or `strip_right` now defaults it to `true`, matching
  `Normalizer.strip`, instead of stripping only on the right.
- `Normalizer.apply_aligned` returns the normalized text together with an
  `alignment` mapping its bytes back to the input, and
  `Normalizer.original_span` reads a span off it. Inserted characters take the
  span of the character they were placed next to and removed ones take none,
  matching what HuggingFace reports, so token offsets can be given in the
  coordinates of the original text.
- BPE tokenization now answers a repeated pretoken from a direct-mapped cache
  seeded with the whole vocabulary. `bpe`'s `cache_capacity` is the slot count
  of that cache (32 bytes a slot, one table per domain), default `262144`
  instead of `10000`; short words merge by a linear rank scan rather than a
  binary heap, which is 2x faster on a miss.
- Fixed a heap overflow in `bpe` with `byte_fallback` and
  `continuing_subword_prefix` or `end_of_word_suffix`: byte fallback spells a
  character out with its affixes, so one source byte becomes several tokens
  and the merge buffers were sized for the bytes. Words of more than a few
  characters corrupted the heap.
- Fixed the placement of `unk_token` around byte fallbacks in `bpe`. A
  fallback no longer lets a pending unknown token out first, so `"za"` with
  `<0x7A>` absent gives the fallback tokens of `a` followed by `<unk>`, as
  HuggingFace does, rather than the reverse.
- Fixed `bpe` reading past the end of a pretoken whose last UTF-8 sequence is
  cut short; with `continuing_subword_prefix` or `end_of_word_suffix` this
  raised `Invalid_argument`. The bytes that remain are now taken as one unit
  and fall through to the byte fallback or the unknown token.
- Loading and saving a tokenizer now carry the BPE model's `byte_fallback`,
  `fuse_unk`, `ignore_merges` and `dropout`. LLaMA and other SentencePiece
  models were dropping `byte_fallback` on load, so every character outside the
  vocabulary became `<unk>` instead of its `<0xNN>` byte tokens, and
  `save_pretrained` wrote the flags back as `false`.
- Decoders rewrite each token instead of joining the list first.
  `Decoder.replace`, `Decoder.strip`, `Decoder.wordpiece` and `Decoder.ctc`
  were collapsing, which stopped a later `Decoder.byte_fallback` in a
  `Decoder.sequence` from ever seeing a byte token — LLaMA decoded `<0x0A>`
  literally.
- `Decoder.ctc` cuts `pad_token` out of a token wherever it occurs rather than
  only dropping tokens equal to it, and drops the tokens left empty:
  `["x<pad>y"]` decodes to `"xy"`.
- `Decoder.strip` takes `~content:string ~start:int ~stop:int`, the counts
  HuggingFace serializes, instead of `~left ~right` booleans; `~content` was a
  `char` and could not hold a marker like `▁`.
- `Decoder.metaspace` takes `~replacement:string` and `~prepend_scheme` instead
  of a `char` and `~add_prefix_space`, and drops every marker in the first
  token rather than one leading space.
- `Decoder.bpe` turns every occurrence of its suffix into the space that
  follows the word, and `~suffix` now defaults to `"</w>"`. It no longer
  inserts a space after a token that has no suffix.
- `Decoder.byte_fallback` decodes a run of byte tokens that is not valid UTF-8
  as one U+FFFD per byte, instead of returning invalid bytes.
- `Decoder.to_json` writes the type names and the `Replace` pattern shape
  HuggingFace reads; the files it produced before were rejected by
  `tokenizers` for `byte_level`, `byte_fallback` and `wordpiece` decoders.
- `Decoder.of_json` reads only HuggingFace's spellings; the brot-only
  `"Byte_level"`, `"Byte_fallback"`, `"Word_piece"` and bare-string `Replace`
  pattern are gone. A tokenizer saved by an earlier brot carries them and must
  be re-saved to load again — those files were never readable by `tokenizers`
  either.
- `Brot.id_to_token` and `Brot.decode` give an added token matched against
  normalized text its normalized form, as HuggingFace does: `id_to_token t 2`
  on LLaMA is `"▁</s>"` while `token_to_id` still takes `"</s>"`, and a `<s>`
  written literally in the input round-trips through encode and decode.
- `train_bpe` now applies `end_of_word_suffix` and `continuing_subword_prefix`
  while learning: the vocabulary gains the affixed characters (`w</w>`) and the
  merges are written over them (`lo w</w>`). A model trained with a suffix
  previously held no suffixed entry at all, so every word-final character
  missed at encode time.
- `train_bpe` now drops characters excluded by `limit_alphabet` from the words
  instead of merging them, counts `max_token_length` in characters rather than
  bytes, settles equally frequent pairs by vocabulary id, and can merge a pair
  a second time when it reappears, recording it once at its later rank. Trained
  vocabularies and merges match HuggingFace's `BpeTrainer` exactly wherever
  that trainer is deterministic.
- `train_bpe` learns merges from an incremental pair index instead of
  recounting every pair each round: a 1 MB corpus trains in 0.03 s instead of
  3.0 s.
- A merge pair listed twice in a `merges.txt` or `tokenizer.json` now takes the
  rank of its last occurrence, as HuggingFace does; the first occurrence used
  to win.
- `Brot.special` is now `Brot.added_token` and the record it builds is
  `added_token`, its `token` field renamed to `content`; `?specials` is
  `?added_tokens` on every constructor and `specials` is `added_tokens`. The
  type covers HuggingFace's added tokens, of which special ones are a subset,
  so the old names described only half of what it holds.
- `add_tokens` takes an `added_token list` and works for every model, not just
  the word-level one: it registers added tokens exactly as passing them at
  construction would, and no longer raises `Invalid_argument`. Added tokens no
  longer enter the model's own vocabulary — they are numbered from the end of
  it, as HuggingFace does — so registering the same token twice no longer
  drifts the identifiers it hands out. To build a model vocabulary, pass
  `vocab` to the constructor.
- `Pre_tokenizer.unicode_scripts` now matches HuggingFace on whitespace and
  unknown scripts. A leading run of spaces used to be emitted as a piece of its
  own; it is now dropped, since the first piece opens at the first script
  change. Only U+0020 and characters of no known script join the surrounding
  run — the other whitespace characters (`\t`, `\n`, U+00A0, U+3000, …) carry
  their own script and split.
- `encode` now splits added and special tokens out of the input ahead of the
  pre-tokenizer and the model, matching HuggingFace: `"a<|endoftext|>b"` with
  GPT-2 gives `[64; 50256; 65]` instead of tokenizing the marker as text. At a
  given position the longest token wins; `~single_word`, `~lstrip`, `~rstrip`
  and `~normalized` on `Brot.added_token` all take effect.
- `Brot.added_token` takes `?special` (default `true`); `?normalized` now
  defaults to `not special`, so `added_token c` matches HuggingFace's
  `add_special_tokens([c])` and `added_token ~special:false c` matches
  `add_tokens([c])`. `decode ~skip_special_tokens:true` drops only the tokens
  with `special` set, so a plain added token survives decoding.
- A `bos_token`, `eos_token` or `pad_token` is now a special token in its own
  right: matched atomically in the input, numbered from the end of the
  vocabulary when the model does not hold it, and skipped when decoding. This
  makes `padding` work with a pad token that is not in the model vocabulary.
  `unk_token` is unaffected — it configures the model's unknown handling and is
  never matched in the input.
- `token_to_id`, `id_to_token`, `vocab` and `vocab_size` now cover added tokens
  the model does not hold; those are numbered from the end of the model
  vocabulary, as HuggingFace does. `added_tokens` reports the same set that
  `to_json` writes, with real ids and `special` flags.
- BPE `end_of_word_suffix` is now appended to the last character of a word
  instead of the first, and a one-character word takes it too;
  `continuing_subword_prefix` goes on every character but the first. Models
  such as CLIP that set a suffix previously produced wrong tokens for every
  word. Byte fallback now covers the affixed character, prefix and suffix bytes
  included.
- `Decoder.wordpiece ~cleanup:true` now applies HuggingFace's detokenization
  cleanup: the space before `.`, `?`, `!`, `,` and the English contractions is
  taken back and `" do not"` becomes `" don't"`, so `["hello"; ","; "world"]`
  decodes to `"hello, world"`. It no longer trims or collapses whitespace.
- `Decoder.ctc ~cleanup:true` now applies the same cleanup to each token before
  replacing the word delimiter, as HuggingFace does. Previously only the
  delimiter was replaced.
- Fix BERT tokenization to match HuggingFace: the `Bert` and `Punctuation`
  pre-tokenizers now treat all 32 printable ASCII non-alphanumerics as
  punctuation. `$ + < = > ^ ` | ~` were missing, so `==` tokenized as one word
  instead of two.
- `Normalizer.bert` now strips only nonspacing marks after NFD, keeping spacing
  and enclosing marks. Stripping every mark dropped the vowel signs of abugidas,
  so `नमस्ते हिन्दी` lost two characters.
- `Normalizer.lowercase` and `Normalizer.bert ~lowercase:true` now apply the
  Unicode lowercase mapping instead of case folding: `ß` and `ﬁ` lowercase to
  themselves rather than expanding to `ss` and `fi`.
- `Normalizer.strip_accents` no longer decomposes to NFD on its own, matching
  HuggingFace's `StripAccents`. Compose it after `Normalizer.nfd` to strip the
  accents of precomposed characters.
- `Normalizer.bert ~clean_text:true` keeps unassigned codepoints instead of
  discarding them as control characters, so they reach the model and become its
  unknown token.
- Fix `encode` returning stale tokens from a previously encoded word, or
  crashing, when a word is made only of characters with no id and the model
  has no `unk_token` or `byte_fallback`. Such a word now yields no tokens,
  matching HuggingFace.
- Fix `encode_batch` returning wrong tokens in rare cases: domains racing on
  the BPE word cache could pair one word's key with another word's tokens, so a
  cache hit returned the wrong ids. Merge scratch buffers are now held per
  domain, so parallel encoding no longer allocates a fresh word and merge queue
  per token.
- Fix GPT-2 (`ByteLevel`) pre-tokenization of whitespace: a run followed by
  text keeps all but its last character, so `"\n\nNot"` splits as `"\n"`,
  `"\n"`, `"Not"` and `"x  y"` as `"x"`, `" "`, `" y"`, as HuggingFace has it.
- Fix the optional leading character of a letter, number or symbol run in that
  same pattern: it is a space, not any whitespace, so `"\tab"` splits as `"\t"`
  then `"ab"`. `add_prefix_space` likewise only skips a leading space.
- Fix letters in that pattern being the Alphabetic property instead of the
  Unicode Letter category, which wrongly joined a combining mark to the letter
  it follows.
- Fix the `Whitespace` pre-tokenizer's word class, which now holds for
  combining marks and connector punctuation and no longer for numbers such as
  `"½"`, matching `\w`.
- Fix `Bpe` emitting any pre-token found in the vocabulary as a single token.
  That shortcut is what `ignore_merges` selects, and it never applies under
  `dropout`; without it the merges decide, and a vocabulary entry no merge can
  build comes out as its decomposition.
- Fix `Bpe` reusing cached merges under `dropout`, which replayed the first
  result for every later occurrence of a word instead of drawing again.
- `Bpe.create` now reads `""` for `continuing_subword_prefix` and
  `end_of_word_suffix` as no affix, which is how tokenizer files spell it: the
  accessors return `None` for those models, and GPT-2 no longer takes the
  allocating affix path when encoding.

### Talon

- `Col.of_tensor` and `of_nx` read a value placed on a device to the host once,
  so a dataframe's columns are host values. A placed column was read one
  element at a time, and eager operations on it ran through the device.
- Fix `cast_column` and mixed-dtype `concat` leaving a source dtype's null
  sentinel in the cast tensor. Casting a nullable integer column to a float
  dtype made `to_array` read its nulls as `Int64.min_int` converted to float
  instead of `nan`; null positions now hold the target dtype's sentinel.
  `cast_column` also raises `Not_found` for a missing column, as documented.
- `to_nx` now takes the target dtype and an optional `?columns` selection,
  `to_nx ?columns dtype df`, instead of always producing a float32 tensor of
  every numeric column. Nulls become `nan` for float dtypes and raise for
  integer dtypes; before, an integer column's nulls silently came through as
  the sentinel value `Int64.min_int` cast to float.
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

- Building quill no longer needs a node toolchain. The bundling rule for the
  server frontend was a target, so a directory build such as
  `dune build packages/quill` ran esbuild and failed without `node_modules`;
  the rule now lives under the `assets` alias only, and
  `dune build @assets --auto-promote` still refreshes the committed `dist/`.
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
