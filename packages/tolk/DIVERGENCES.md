# Divergences

Intentional divergence from the tinygrad reference.
Standing rulings; work items live in TODO.md. Anchors reference the tinygrad
clone pin named there.

## Declined ports

Do not re-open without a consumer.

- **Void-RANGE loop header** (`b764599d8`, `39924387b`, its `spec.py` rules).
  Serves HCQ wait loops, a runtime tolk does not port; nothing in tolk
  constructs a void range, so every branch would be unreachable and no test
  could catch a bad port. `Axis_type.Loop` names the concept, deliberately
  without a constructor. Port together with a producer and its test.

- **`Renderer.abi` and CALL-in-C** (`d1f215d37`, `6e979b879`). Both serve the
  CPU uop worker runtime. `abi` only feeds `kernel_typedef`, which tolk spells
  statically; CALL-in-C renders an address-valued CALL nothing constructs.
  Port together with a host-callable runtime.

- **`ParamArg.volatile`** (`a6fda6b10`). Producers are `ops_cpu.py`,
  `ops_qcom.py`, `hcq2.py`; tolk would carry an always-false field. Add with
  the first runtime that needs uncached parameter memory.

- **`ProgramInfo.target`; tolk's `aux` deleted rather than renamed**
  (`9fdaa4bff`). `target` exists so `UOp.to_elf()` can build a `TinyELF` — a
  runtime restructure tolk does not port; `Program_spec.t` already carries
  `device`. `aux` had no reader. (`call_info.aux` is unrelated and stays.)

- **Host-scalar `bitcast` stays in `symbolic.ml`**, upstream moved it to
  `dtype.ml` (`67dc02d7e`). Forced by layering: `Const` depends on `Dtype`.
  The only would-be consumer is host-side rand arithmetic, and
  `lib/frontend/rand.ml` builds `_bits_to_rand` from UOp `Bitcast` nodes.

## Beam search

Behavior changed relative to the reference, each reviewed with the beam
campaign.

- **`BEAM_MIN_PROGRESS` defaults to 5µs; the reference defaults to 0.01µs.**
  0.01µs sits below device timer resolution (~0.5µs), so the progress exits
  never fire on real gains and searches run to exhaustion. The reference's
  own production configs use 5–10µs. Env still overrides.

- **`BEAM_PARALLEL=N` compiles candidates across N domains, default off; the
  reference uses a `PARALLEL` process pool, default on for GPU devices.**
  Domains share the process, so: global state is lock-guarded and the
  per-node memo caches are domain-local; workers never touch the device; the
  SIGALRM compile timeout is skipped in parallel mode (it is process-global);
  and each step compiles all candidates before timing any, instead of
  streaming, so timing never contends with compile load.

- **Dispatch handles for timed candidates are cached for the process; the
  reference loads one per candidate and unloads it after timing.** tolk never
  unloaded them anyway (`Device.prog.free` has no caller), so caching
  strictly reduces loads. Deterministic unload remains open.

- **Candidates dedup by optimized-AST tag before compiling; the reference
  dedups by binary after.** Same AST means same binary, so verdicts are
  unchanged and duplicates skip nvrtc. Side effect: a candidate rejected by
  the 1000× compute-ops filter is never reconsidered at a later step.

- **Timing buffers bypass the LRU cache (`nolru`) and are freed per kernel;
  the reference allocates them normally.** It can: refcounting frees
  promptly. Under a lazy GC the exact-size LRU cache hoards every searched
  shape, and driver module loads OOM without triggering the allocator's
  failure flush.

## AMD runtime

- **The driver-less PCI interface is opt-in (`AMD_IFACE=PCI`); the reference
  selects it automatically when the kernel driver cannot open the device.**
  The path boots the GPU by writing firmware and engine registers directly
  and has never run on hardware — a wrong sequence can wedge the device
  until a bus-level reset — so the kernel driver stays the only automatic
  choice. Promotion criterion: automatic KFD→PCI fallback once the path has
  been validated on hardware.

- **The PCI iface base maps neither CPU-backend buffers nor remote devices,
  and the multi-die `p2p_paddrs` override is not carried.** Buffers here
  carry the PCI metadata type, so a CPU-backend buffer cannot be passed;
  the remote-device backend is a declined port; and the multi-die override
  only diverges from the base on fabrics the consumer PCI-id allowlist
  keeps out. Revisit with the first cross-backend, remote, or multi-die
  consumer.

## NV runtime

- **The driver-less PCI interface is opt-in (`NV_IFACE=PCI`); the reference
  auto-falls-back NVK→PCI (`_select_iface`/`select_first_inited`) when the
  kernel driver cannot open the device.** The path boots the GSP firmware by
  writing engine and falcon registers directly and has never run on hardware,
  so the kernel driver stays the only automatic choice. Promotion criterion:
  restore the automatic NVK→PCI fallback once the path is validated on
  hardware.

- **The runtime carries the kernel-driver metadata type on every buffer, so
  the PCI interface adapts the memory-manager buffers to it, keyed by virtual
  address.** The `Nv_iface.t` seam fixes buffer metadata to
  `{h_memory; owner_id}` (the reference parameterizes the whole runtime by
  buffer metadata instead). The driver-less interface keeps its base
  allocations in a side table so `free` and `map` reach the memory manager;
  peer mapping across driver-less devices is out (single-device only). The
  shared open path allocates the channel ring without `force_devmem`, a
  divergence to revisit at hardware validation.

## CPU runtime

- **The CPU device runs on tolk's own worker queue; the reference's CPU device
  is HCQ.** A free waits for the queue, as `HCQAllocator._free` does, except a
  free that a GC finaliser runs inside the queue's lock: it is deferred to the
  next synchronize (`runtime/cpu/tolk_cpu.ml` `after_queue`), so that memory
  can return later than in the reference.

## Tolk extensions

Code tolk carries that the reference does not. Every site has a comment
containing the phrase "tinygrad counterpart" — `grep -rn "tinygrad
counterpart" lib` lists them. An extension needs a consumer; without one,
delete it rather than registering it.

- **`CALL(CUSTOM_FUNCTION "loop")` — the staged-scan loop**
  (`engine/realize.ml` `exec_loop`; `schedule/rangeify.ml` `find_bufs`
  walking `enter_calls:false` and `split_store` passing a precompiled CALL
  through as its own kernel; builder with its consumer in rune's `jit.ml`).
  Rune stages `Rune.scan` as one compiled body replayed per slice (the
  design is in `rune/lib/scan.ml` and `stage_scan` in `rune/lib/jit.ml`);
  the reference's answer to a recurrence is unrolling plus TinyJit, so there
  is nothing to port. The loop launches its body and nothing else: the
  schedule writes the buffers it starts from, and the body writes every
  result. A row of a stacked argument (a layer's weights, a step's input) is
  bound as a view at byte offset `i * stride` for iteration `i`, never
  copied; rows are padded to 16 bytes so every view is aligned. The body is
  batched into graph calls like any compiled linear; each iteration replays
  them through the upstream graph runner with its rebound slot buffers
  patched in, cycling through three recordings of each graph
  (`exec_loop_graph`) because patching a graph waits for its previous
  replay. The named-payload mechanism is upstream's own ("graph", "encdec",
  "hcq"); "loop" is a tolk-local name in it, and no tinygrad-shaped graph
  can reach the new branches. Pin moves must keep the two rangeify branches
  — a re-sync of `rangeify.py` will not find them upstream. The tolk corpus
  cannot build a loop call; rune's `test_jit.ml` scan groups are this
  extension's parity suite.

- **`split_reduceop` leaves a one-hot sum whole** (`schedule/rangeify.ml`
  `is_one_hot_sum`). The reference splits any reduce whose input is 32768
  times its output, in the tensor graph, and collapses a gather's one-hot
  reduce to a gated load later, per kernel. At 32768 rows and up the split
  wins: the sum becomes 256 chunk sums behind a `contiguous`, each collapsing
  to a load gated on its chunk, and a second kernel adds the 256 slots, so an
  element costs 256 gated loads, an intermediate 256 times the output and two
  kernels where one load would do. tolk
  declines the split for a sum over `where(c, x, 0)` when `c` compares
  integers and one side varies only along reduced axes while the other varies
  along none of them. The reference has the same behaviour and no test above
  the threshold (`test_arange.py` indexes 2048 rows, `test_llama_embedding` a
  vocabulary of 10). Parity case `gather_split_threshold` holds tolk's kernel
  to the reference's with `SPLIT_REDUCEOP=0`. Consumer: every `Op.gather` and
  tensor-index `getitem` over a large axis, so rune's `take` over a
  vocabulary. Drop the guard if the reference orders the two rewrites itself.

- **Index-ranged scatter** (`frontend/op.ml` `scatter_indexed`). The
  reference lowers every scatter through a one-hot mask over the destination
  with a trailing axis over the indices, so writing `k` rows into `n` costs
  `n * k`; its one kernel that stores at a loaded index is the embedding
  gradient behind `USE_ATOMICS`, on CPU and AMD only. tolk generalises that
  kernel to the along-axis scatter through the ported `custom_kernel`, and like
  it writes the storage it is given in place: the coordinates off the
  scattered axis are parallel lanes, the index range is a serial `Reduce`
  range inside each lane, so duplicates land in index order on every device
  without atomics (the reference clips a bad index onto an edge row; tolk
  gates the store off, which is what `Nx.scatter` documents). The reference
  `scatter` and `scatter_reduce` stay as ported and are
  what the reference's own tests cover; parity cases `indexed_store_set`,
  `indexed_store_add` and `indexed_store_unique` hold the kernel shapes to the
  reference's codegen. Consumers: rune's `E_scatter`, hence `Nx.scatter`, the
  gradient of `Nx.take` and kaun's embedding gradient; and rune's `E_update`
  at a traced corner, over the flattened destination.

- **`?aligned` on the Clang renderer** (`renderer/cstyle.ml`
  `clang_vector_prefix`, passed down from `Tolk_cpu.create`). The reference
  selects unaligned vector types through the `ALIGNED` environment variable
  alone. tolk also takes the choice as an argument, so a caller can select it
  for one device without touching the process environment; absent, the
  variable decides as in the reference, and the rendered source is the
  reference's either way. Consumer: rune's CPU device, which binds host memory
  it did not allocate (slices, mapped files) and passes `~aligned:false`.
