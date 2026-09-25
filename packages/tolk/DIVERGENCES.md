# Divergences

Intentional divergence from the tinygrad reference.
Retained rulings from the September 2026 audit; unresolved gaps live in
[TODO.md](TODO.md).

## OCaml representation and lifetime

- **Rune owns custom differentiation before Tolk lowering.** The target's
  custom-kernel gradients run in its tensor autodiff layer. Tolk retains call
  metadata but has no separate autodiff consumer: Rune's effect handlers run
  custom VJP/JVP rules and trace their resulting operations into Tolk. Adding
  another gradient engine would duplicate Rune's ownership. Coverage:
  `test_custom` compiled reverse/forward rules and duplicate-index backward
  scatter; `test_jit_metal` backward scatter with fresh bindings and GC between
  replays. Reconsider if a standalone Tolk consumer needs differentiation of
  custom kernels without Rune.

- **Queue replay validates writable aliases before submission.** The target
  derives dependencies from graph identities without checking whether later
  runtime bindings add overlaps. Tolk rejects new writable overlaps between
  unordered calls before patching the address table. FIFO order and transitive
  queue waits still permit donation; disjoint views and read-only aliases also
  remain valid. Shared graph roots establish dependencies for writable aliases. Consumers: retained Rune
  and Tolk submissions, particularly external pointer imports. Coverage:
  `test_hcq2` duplicate writable inputs, overlapping external wrappers, unchanged
  timelines on rejection, aliases across regrouped batches, and accepted
  read-only/disjoint bindings. Reconsider
  when dependencies can be recompiled automatically for a changed alias layout.

- **Peer regrouping respects cross-group byte dependencies.** The frozen
  target groups a whole queue-compatible run by peer group. Tolk combines
  interleaved groups only when doing so preserves preceding reads and writes;
  ordinary calls remain boundaries. This matters for device groups sharing
  host memory. Coverage: `test_hcq2` independent regrouping, read-after-write
  chains and write-after-read barriers. Reconsider when upstream orders shared
  memory accesses between incompatible peer groups.

- **Metal queues leave kernels with more than 15 arguments to direct
  dispatch.** The target encodes every kernel into an indirect command buffer.
  On an M1 Max (Apple7) a kernel with 16 to 29 arguments (buffers plus scalar
  variables) and a body with many distinct constants computes wrong values
  from an indirect command buffer and correct ones from a direct dispatch of
  the same pipeline and buffers. A standalone Objective-C program with no
  tolk or tinygrad code reproduces it, with the arguments bound as buffers or
  as fields of one argument buffer; GPU shader validation hides it, and the
  target's pre-M3 empty dispatch does not help. The target returns wrong
  values for a `cat` of 16 tensors. Tolk's queue capability
  `max_kernel_bindings` is `Some 15` on Metal, and `Hcq2.enqueue` leaves a
  kernel above it to an ordinary dispatch between batches. At the time of the
  ruling, no kernel of the rune Metal tests (apart from this coverage) or of
  the tiny gpt-oss validation exceeded 10 arguments. Coverage: Rune's
  `test_jit_metal` concatenation of 16 sorts between queued kernels, and the
  Metal runtime's argument structures of 15 buffers (queued), 16 and 33
  (direct), which rebind between calls. Reconsider when Apple fixes the
  driver, and measure Apple8 and later, which were not available.

- **Multi-axis reshapes use each axis's own shard count.** The frozen target
  reuses the last range's count while constructing all local dimensions, so a
  `[2; 4]` tile sharded 2-by-3 cannot reshape its `[4; 12]` logical shape to
  `[4; 3; 4]`. Tolk keeps the owning range with each axis and produces a local
  `[2; 1; 4]` view. Consumer: per-thread fragments and multi-axis device tiles.
  Coverage: `test_multi` unequal-count reshape and six-device tiled gather.
  Reconsider when upstream fixes the range lookup.

- **The schedule cache key includes the settings scheduling reads.** The
  frozen target keys its schedule cache on the graph alone, but scheduling
  reads context settings after the lookup: `RING`, `ALL2ALL`,
  `ALLREDUCE_NODE_NDEVS`, `RING_ALLREDUCE_THRESHOLD` and `ALLREDUCE_CAST` choose
  how an allreduce expands, and `FLOAT16`, `SPLIT_REDUCEOP`,
  `REDUCEOP_SPLIT_THRESHOLD`, `REDUCEOP_SPLIT_SIZE`, `OPENPILOT_HACKS`,
  `PCONTIG` and `MAX_KERNEL_BUFFERS` shape the kernels. A graph scheduled again
  under other settings replayed the first schedule. Tolk appends
  `Schedule.config ()`, which renders those values and the startup values of
  `LATE_ALLREDUCE` and `NO_MEMORY_PLANNER`, to the key. Consumer: rune's
  compile cache, which stores the schedule this cache returns under a key that
  records the same settings. Coverage: `test_engine_schedule` schedules one sum
  with and without `SPLIT_REDUCEOP`; `test_multi` schedules one reduction under
  four forced strategies and gets four schedules, and the forced-strategies
  value test runs each of them. Reconsider when upstream keys its cache on
  these settings.

- **Hierarchical allreduce folds boxes in one order on every device.** The
  frozen target sums a chunk on each box's device starting from its own
  partial, then the other boxes' in index order, and each device takes the
  chunk from its own box. With three or more boxes the devices fold in
  different orders and their replicas differ in the last bits. Tolk folds the
  partials in box order on every device and stores each partial first. The
  stored partial works around the C renderer, which prints a nested sum as one
  flat expression and so reassociates a partial fused into the fold. Traffic
  is unchanged. Consumer: RFC 0005's replicated values, which readers take
  from any one device. Coverage: `test_multi` hierarchical replicas on 6 and 8
  devices with 1 and 2 devices per box. Reconsider when upstream fixes the
  fold order; the `contiguous` can go when the renderer keeps the graph's
  association.

- **Compiled AMD submission bounds polling and reserves ring space before
  writing commands.** The frozen target can spin forever or overwrite unread
  packets when producers outrun a queue. Native helpers latch failures while
  OCaml is released, suppress ring writes/publication after failure, and leave
  one slot unused to distinguish full from empty with ring-relative pointers.
  SDMA publishes wrap padding before reserving space for the next stream.
  Consumers: retained replay and large asynchronous copy batches. Coverage:
  host-executed AMD wrap, full-ring and stalled-replay tests; hardware validation
  remains open. Reconsider when upstream provides equivalent bounds and
  backpressure.

- **AMD compute recovery requires idle SDMA queues.** The reference resets
  compute processors and advances their abandoned timeline without cancelling
  queued DMA commands. Those commands can still wait on signals from discarded
  compute work and later access reused storage. Tolk retains the fault when a
  copy ring has outstanding work. Recovery of an idle-copy device clears both
  its timeline error and its native submission latch, while reporting the lost
  work to the caller. Coverage: `test_amd_amdev` full guarded recovery, healthy
  peer isolation and refusal to release active copy storage. Reconsider when
  DMA cancellation and retirement have hardware validation.

- **NV channels track completion without a hardware consumer pointer.**
  Ampere exposes GPPut but no GPGet. Tolk appends an engine release carrying
  a channel sequence; direct and compiled submissions share that sequence
  and leave a FIFO slot empty. Direct staging waits before overlapping live
  storage, retained replay waits before patching its old command tail, and
  synchronization drains both channels. The frozen target does not bound
  independent submissions this way. Native waits latch timeout and suppress
  publication. Coverage: `test_runtime_nv` FIFO saturation and resumption,
  independent retained batches, staging wrap across channels, delayed command
  tails and 32-bit completion rollover. Hardware validation remains open.
  Reconsider when upstream provides equivalent occupancy and retirement rules.

- **AMD/NV submissions read the mapped producer position and retain timeline
  addresses across rollover.** Direct `Device.prog` launches coexist with
  compiled submissions, so a second host counter can overwrite unread ring
  entries. Before the low timeline dword reaches its comparison limit, Tolk
  drains work and advances the high-word epoch at the same address. Retained
  host fences remain monotonic while GPU dword waits restart safely. AQL scratch
  updates touch only scratch fields, preserving live dispatch counters. Coverage:
  mapped producer tests, two epoch transitions and compiled AMD host replay
  across rollover. Hardware acceptance remains in TODO. Reconsider when direct
  dispatch is removed or upstream provides an equivalent rollover protocol.

- **Direct AMD/NV launches fence kernel-argument arena wrap.** Tolk retains
  direct `Device.prog` launches for autotuning and standalone runtime callers;
  the frozen target's queue compiler owns argument storage per linked batch.
  The direct arena waits for the preceding timeline value before wrapping,
  without waiting for ordinary allocations. Coverage: `test_runtime_amd` and
  `test_runtime_nv` preserve arguments, QMDs and producer positions after a
  failed wrap wait, then retry after completion. Reconsider when direct
  dispatch uses the compiled queue's per-batch storage ownership.

- **Host access, direct calls and native transfers wait for existing importers.**
  Tolk still exposes asynchronous `Device.prog` dispatch outside compiled queue
  dependencies. `Device.runtime` waits for foreign owners and importers before
  direct dispatch; host reads/writes and native buffer transfers also wait for
  importing devices. Address binding itself does not wait. Compiled host
  submissions use `Device.queue_runtime` and their encoded fences, so replay
  does not accidentally drain every device. They wait for recorded foreign
  accesses outside their own timelines, including separate groups sharing host
  memory. Coverage: storage mapping/transfer
  lifetime tests, CPU direct-binding waits, host queue replay with zero implicit
  owner waits, uncovered host-writer retirement before submission, and real
  Metal host-view dispatch. Reconsider when all direct
  dispatch participates in the same byte-interval dependency protocol.

- **Staging slots belong to a prepared schedule.** The frozen target caches
  one host staging allocation per host device. Independently retained Tolk
  batches fence their own command storage, so sharing those slots would let
  one batch overwrite transfers belonging to another. Each staged schedule
  owns two 64 MiB slots, reused with byte-interval queue dependencies. Coverage:
  `test_hcq2` checks chunk tails, read-before-reuse waits, independent storage,
  replay rebinding and input lifetime. Reconsider when a shared staging pool
  has reservations spanning independent submissions.

- **PCI peers can import system memory owned by small-BAR devices.** Host
  timelines and queue signals use system physical pages, not the owner's BAR.
  The frozen target rejects them together with inaccessible device memory,
  preventing the very cross-queue signals needed for staged peer copies.
  Device-memory imports still request staging. Consumer: staged PCI peer
  transfers; hardware acceptance remains explicit in TODO. Reconsider when
  upstream distinguishes system and device memory in its small-BAR check.

- **Failed buffer setup unwinds acquired resources.** The frozen target does
  not consistently roll back allocation and mapping failures. Tolk returns
  PCI virtual/physical reservations, CPU mappings and new page tables, releases KFD/NVK
  handles acquired by failed setup, and closes NVK temporary mapping file
  descriptors. A failed page-table rollback retains backing storage and
  virtual reservations because the device may still address them.
  Bootstrap failures release descriptors and per-device KFD
  events. KFD's process-wide event page remains cached; an ambiguous registration
  failure is latched because the kernel may already retain that page. Failed
  KFD/NVK device construction retires queues before releasing their buffers;
  a failed queue stop retains storage and suppresses late finalizers. NVK
  unregisters channels with the installed driver's layout, then unwinds UVM
  registrations, control mappings and the per-device RM object tree.
  Borrowed host mappings remain owned by the caller. Consumers:
  allocation retries and long-lived accelerator sessions. Coverage:
  `test_memory` allocation, zeroing, entry-write and flush failures, including
  adjacent mappings and precreated tables; `test_amd_system` covers reverse
  rollback order, acquisition failures, cleanup failures, descriptor release
  and buffer finalizers after successful or failed queue retirement;
  `test_nv_tables` covers 570/580/610 unregister layouts.
  Driver fault injection and hardware
  recovery remain open in TODO. Reconsider when upstream provides equivalent
  failure ownership rules.

- **PCI cleanup releases owned system-memory virtual ranges and the CPU
  view's address.** The frozen reference frees only device-memory ranges and
  assumes the GPU address is the CPU mapping address. BAR mappings can differ;
  retaining either allocation leaks page tables or host mappings. Consumers:
  AMD/NV pinned staging and CPU-visible queue storage. Coverage: repeated
  system/peer mapping release in `test_memory` and hardware-gated AMD CPU-map
  release. Remove this ruling when upstream releases both resources correctly.

- **Profiles export immutable events as Chrome trace JSON.** The target
  serializes Python objects for its visualizer; Tolk's OCaml consumers can
  inspect `Device.profile` events or use `Profile.output` without adding a
  serialization dependency. Pending repeated slots coalesce until synchronization
  as in the target. Coverage: collector lifetime/failure tests, escaped trace
  output, host queue replay and real Metal asynchronous replay. Calibrated
  device timestamps share the host wall clock and a single trace origin;
  AMD/NV/CUDA hardware alignment remains an acceptance requirement in TODO.
  Reconsider the format if Raven gains a shared profiling
  artifact protocol needed by these consumers.

- **CPU kernels expose an entry taking buffer and scalar arrays.** The Clang
  wrapper casts each scalar to its declared type, so OCaml can call arbitrary
  kernel arities through one native stub without an FFI dependency. `Tiny_elf`
  retains the typed signature and dispatch checks both array lengths. Coverage:
  CPU execution tests for mixed-width scalars, sparse slots, rejected arities
  and typed host calls. Reconsider when adding a CPU renderer that cannot emit
  this entry convention.

- **CUDA submission calls ordinary C helpers instead of Python-callable driver
  objects.** Generated host code runs with the OCaml runtime released, so
  helpers retain the first submission failure for synchronization to report.
  Event handoffs also order the autotuner’s direct `Device.prog` launches and
  allocator uploads with nonblocking queues. Consumers: compiled submission,
  JIT replay and search timing. Coverage: driver-independent submission
  compilation and CUDA runtime replay/transfer cases (hardware acceptance is
  still open in TODO). Reconsider the handoffs if direct dispatch and uploads
  move entirely into queue compilation.

- **Symbolic index expressions use `Movement.symbolic_shrink`**, composed with
  `Movement.squeeze` for a scalar selection. `Movement.index` keeps integer
  bounds; `Op.getitem` supports symbolic axis lengths and index-tensor shapes.
  This keeps dimension expressions in the existing typed bounds API. Consumers:
  GPT-2 token lookup, KV-cache updates and position selection in
  `examples/gpt2/main.ml`. Coverage: the frontend symbolic indexing/cache tests
  and GPT-2 execution suite. Reconsider if a consumer needs mixed symbolic and
  advanced indexing that cannot be composed from these operations.

- **Host-scalar `bitcast` stays in `symbolic.ml`**, upstream moved it to
  `dtype.ml` (`67dc02d7e`). Forced by layering: `Const` depends on `Dtype`.
  The only would-be consumer is host-side rand arithmetic, and
  `lib/frontend/rand.ml` builds `_bits_to_rand` from UOp `Bitcast` nodes.

- **Compilation workers use OCaml domains rather than Python processes.**
  This is an execution mechanism, not a reason for different search policy.
  Shared context, bounded worker ownership and cancellation must be validated
  before parallel compilation becomes the default; those tasks are in TODO.
  Reconsider if domain isolation cannot preserve compilation semantics.

- **Timing buffers bypass the LRU cache (`nolru`) and are freed per kernel;
  the reference allocates them normally.** It can: refcounting frees
  promptly. Under a lazy GC the exact-size LRU cache hoards every searched
  shape, and driver module loads OOM without triggering the allocator's
  failure flush.

## Validation dependencies

- **Scalar out-of-bounds validation has no SMT solver fallback.** Keep Tolk's
  dependency footprint small: interval and symbolic proofs must establish
  safety, and unproved accesses are rejected under `CHECK_OOB`. Some safe
  relationally constrained indices accepted by tinygrad may therefore be
  rejected. Coverage: the scalar/spec OOB tests. Reconsider if a concrete
  consumer requires proofs that cannot be expressed by the existing rules.

## Rendering

- **Pointer casts preserve volatile qualifiers.** The frozen reference qualifies
  parameters but its `CStyleLanguage.render_ptr` drops the qualifier when casting
  a vector access. Tolk keeps it so explicit vector accesses to polling or shared
  runtime buffers retain their memory semantics. Coverage: the renderer's
  volatile vector-pointer regression. Remove this ruling when upstream preserves
  the qualifier in access casts too.

## Tolk extensions

Code tolk carries that the reference does not. Every site has a comment
containing the phrase "tinygrad counterpart" — `grep -rn "tinygrad
counterpart" lib` lists them. An extension needs a consumer; without one,
delete it rather than registering it.

- **`CALL(CUSTOM_FUNCTION "loop")` — the staged-scan loop**
  (`engine/realize.ml` `exec_loop`; `schedule/rangeify.ml` `find_bufs`
  walking `enter_calls:false`; builder with its consumer in rune's `jit.ml`).
  Rune stages `Rune.scan` as one compiled body replayed per slice (the
  design is in `rune/lib/scan.ml` and `stage_scan` in `rune/lib/jit.ml`);
  the reference's answer to a recurrence is unrolling plus TinyJit, so there
  is nothing to port. The loop launches its body and nothing else: the
  schedule writes the buffers it starts from, and the body writes every
  result. Outputs depend directly on the call effect, through the shared
  allocation/call protocol. A row of a stacked argument (a layer's weights,
  a step's input) is bound as a view at byte offset `i * stride` for iteration `i`, never
  copied; rows are padded to 16 bytes so every view is aligned. Loop bodies
  use the same compile/link/run protocol as other schedules; each iteration
  rebinds materialized slot buffers. The named
  `CUSTOM_FUNCTION` payload is the extension seam. Keep the two rangeify
  branches when updating the reference. Rune's `test_jit.ml` scan groups and
  `test_jit_metal.ml` exercise this extension. Reconsider it if upstream gains
  a cross-kernel recurrence that supports these consumers.

- **Ordered argument accesses on compiled queue calls** (`Uop.queue_info.accesses`).
  Rune's donation and staged-carry analysis must see whether a later kernel
  reads an input after an earlier kernel writes a candidate output. A host
  submission's own pointer accesses do not describe that order. Keep the
  original dispatches' argument slots as metadata. Rune's donation and Metal
  replay suites cover the consumer. Remove this metadata if reuse analysis moves before queue
  compilation or upstream exposes equivalent access information.

- **`split_reduceop` leaves a one-hot sum whole** (`schedule/prepare.ml`
  `is_one_hot_sum`). The reference splits any reduce whose input is 32768
  times its output, in the tensor graph, and collapses a gather's one-hot
  reduce to a gated load later, per kernel. For the 65536-row regression, the
  sum becomes 256 chunk sums behind a `contiguous`, each collapsing
  to a load gated on its chunk, and a second kernel adds the 256 slots, so an
  element costs 256 gated loads, an intermediate 256 times the output and two
  kernels where one load would do. tolk
  declines the split for a sum over `where(c, x, 0)` when `c` compares
  integers and one side varies only along reduced axes while the other varies
  along none of them. Parity case `gather_split_threshold` holds tolk's kernel
  to the reference's with `SPLIT_REDUCEOP=0`. The frozen target still schedules
  one gather kernel at 16384 rows and two at 32768/65536 rows with splitting
  enabled; Tolk's frontend regression requires one at all three sizes.
  Consumer: every `Op.gather` and
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

  When the caller states that no two updates of a lane share an index
  (`~unique:true`, `Nx.scatter ~unique_indices:true`), the index range is a
  parallel range and the kernel goes through the hand-coded optimizations,
  where the reference builds its embedding gradient with
  `opts_to_apply=()` (`tinygrad/nn/__init__.py:363` at baa614806) and lays
  out its own local axis and atomics by hand. tolk's kernel has neither:
  built as is, it runs one thread per workgroup, and a 64 x 32768 scatter
  took 2.9 ms on Metal instead of 0.17 ms. tolk never infers uniqueness. A
  scatter without the flag keeps the serial range and its kernel as built,
  so duplicates still land in index order. A caller that breaks the promise
  gets, at a repeated position, an unspecified one of its updates under
  `Set` and an unspecified value under `Add`; every other position stays
  exact. Consumer: `Nx.top_k`, whose compaction is a permutation.

- **A gate clause on a loaded value survives a reshape** (`uop/symbolic.ml`
  `pm_drop_and_clauses`). The reference keeps, on each axis of a reshape, only
  the clauses over that axis's ranges; its index gates bound addresses, and
  meaning lives in values. tolk's indexed scatter gates its store on the
  loaded index, and the quantised and block products gate loads on a loaded
  id, so the gate carries meaning. No axis's ranges imply a clause on a loaded
  value, and dropping it stored an out-of-range update at the index modulo the
  axis (one update, or a destination axis of size 1). tolk keeps a clause that
  reads memory on every axis; every golden and parity output is unchanged.
  Covered by `test_run`, the block and quant gate tests and rune's
  out-of-range scatter checks, on CPU and Metal.

- **Quantised matrix product** (`frontend/op.ml` `quant_matmul`). The
  reference's only fused quantised products are hand-written AMD kernels in
  `extra/`; a product with MXFP4 weights written as tensor operations decodes
  every weight before it multiplies, and the matrix-vector options do not
  apply to a reduction whose source is a decode. tolk writes the product as a
  custom kernel through the ported `custom_kernel`, with its options pinned per
  device (`quant_options`: measured on Metal and the CPU, none elsewhere).
  Each 32-value group's code bytes and scale byte are read once per tile of
  rows and decoded in registers. Codes and 16-bit inputs are read through a
  float32 placeholder over their storage, since the reference folds only float
  loads into vectors. With ids, a position whose id selects no matrix bounds
  the outer reduce loop at zero on a GPU and gates its loads on the CPU.
  Consumer: rune's lowering of `Nx_quant.apply` (RFC 0004), which takes it
  within `quant_row_bound`'s rows. Coverage:
  `test/unit/frontend/test_quant_matmul.ml` (a host reference on the default
  device, and per renderer that every float multiply lies in the id-bounded
  loop on a GPU and that no loop reads memory on the CPU), the opt-correctness
  sweeps on CPU and Metal, and rune's Law 2 battery.

- **A loop bounded by a loaded value counts at its bound in estimates**
  (`program_spec.ml` `estimate_of_size`). The reference multiplies a loop's
  trip count into the estimates symbolically. A trip count that reads memory,
  such as the quantised product's id-bounded loop, is not known before the
  kernel runs, and `sym_infer` cannot evaluate it, so tolk counts it at its
  upper bound. The reference has no loop of this kind.

- **Block matrix product** (`frontend/op.ml` `block_matmul`). The reference
  multiplies blocks of rows by matrices chosen per block only through `matmul`
  over the gathered matrices, which copies one matrix per block, and a block
  whose id selects nothing still multiplies. tolk's block kernel, built through
  the ported `custom_kernel`, reads each block's matrix in place. Its
  contraction is two loops: an outer loop over tiles of 8 whose bound, on a
  renderer with work groups, is zero when the block's id is outside the stack,
  around a constant loop of 8 that the tensor-core option splits. Such a block
  reads its id, runs no multiply-adds and stores zeros. The bound is one value
  per work group only while the block axis is a whole global dimension, so the
  builder applies its options to the kernel and raises if the block axis does
  not survive them whole. The CPU runs work groups as a loop, where a bound that
  reads that loop's index miscompiles, so there the bound is constant, the
  weight's load is gated and a select zeroes the store, as for a contraction of
  one input anywhere, whose loop of one trip folds away. The options are pinned
  per renderer and shape: on Metal the tensor cores, rows upcast by up to 8
  tiles, columns by 3, and a local split of 4 on the columns, measured at 6.6 to
  7.6 TFLOPS at bfloat16 on an M1 Max for 46 filled blocks of 64 rows at
  gpt-oss's shapes, where the heuristic's `matmul` reaches 8.2 on a dense
  product; on the CPU at every dtype, rows upcast by up to 8, columns by up to
  16 within 64 accumulators, and the loop over tiles unrolled by 4, timed on
  an M1 Max at gpt-oss's shapes 9 to 37 times faster than no options at
  float32 and 1.4 to 2.1 times at bfloat16; elsewhere none yet. Coverage:
  `test/unit/frontend/test_block_matmul.ml` (values on the default device, and
  each renderer's loop bounds) and the opt-correctness workloads `block_matmul`
  and `block_matmul_t`, under every action that leaves the block axis whole.
  Consumer: rune's lowering of `Nx_quant` products over expert ids, grouped or
  one block per position.

- **`?aligned` on the Clang renderer** (`renderer/cstyle.ml`
  `clang_vector_prefix`, passed down from `Tolk_cpu.create`). The reference
  selects unaligned vector types through the `ALIGNED` environment variable
  alone. tolk also takes the choice as an argument, so a caller can select it
  for one device without touching the process environment; absent, the
  variable decides as in the reference, and the rendered source is the
  reference's either way. Consumer: rune's CPU device, which binds host memory
  it did not allocate (slices, mapped files) and passes `~aligned:false`.

- **A gather is an all-gather of pure copies** (`schedule/multi.ml`
  `allgather`, `schedule/prepare.ml` `storage_window`, `engine/schedule.ml`
  `copy_kernel_params`). The reference lowers a copy of a split value to
  several devices as an allreduce of zero-padded shards, and to one device as
  a sum of padded shards. Tolk lowers both to one precompiled call named
  `allgather` over (dst, src): each target gets one buffer, and each shard is
  written once into its window of it, by a transfer from another device or a
  store on its own. Each device receives (n-1)/n of the value, where the
  reference's ring and naive allreduces move 2(n-1)/n and n-1 full buffers,
  and no padded shard or sum is materialized. The windows need two
  extensions: prepare drops a copy into a contiguous window of a parameter or
  buffer with a concrete shape (the reference only into a whole buffer, so a
  window gets a staging buffer), and a copy kernel storing
  `dst[i + a] <- src[i + b]` becomes a transfer between byte views. The
  reference forbids offset copies because SDMA cannot do them; tolk's copy
  paths take them: Realize resolves the SHRINK to a `Device.Buffer.view`,
  `Deps_tracker.uop` keys byte intervals by view offset, and the AMD SDMA
  `COPY_LINEAR` path (`tolk_amd.ml`) takes byte addresses with no alignment
  requirement. Unverified on SDMA, CUDA and NV queues until the node runs.
  Symbolic windows keep the staging buffer. Inner-axis windows are not
  contiguous and stage every foreign shard. Outputs are forwarded again after
  `multi_pm` (`schedule/prepare.ml` `prepare_rangeify`; the reference
  forwards only before it), so a realized gather writes the result's storage
  instead of a fresh allocation it then copies. Consumer: every copy of a
  split value (`Creation.clone`, `U.copy` to a device list, resharding in
  `multi_pm`).
  Coverage: `test/unit/engine/test_collectives.ml` "all-gather" and
  `test/unit/engine/test_multi.ml`.

- **A collective call takes its whole output allocation**
  (`schedule/allreduce.ml` `collective`). The reference's
  `create_allreduce_function` passes the output's view (a SHRINK of a RESHAPE
  of the allocation) as the call's first argument. `realize_custom_kernel_srcs`
  realizes any argument that is not buffer-like into a copy of its values, so
  when that view is not a plain reshape (a symbolic slice of an inner axis),
  the call writes the copy and the result reads an allocation nothing wrote:
  such a realized allreduce returned zeros. Tolk passes the allocation and
  views it inside the call's body, which is also the raw storage a library
  collective needs. Coverage: `test/unit/engine/test_collectives.ml` "a
  realized allreduce of a symbolic slice keeps its values".

- **Every call's arguments are realized under one rule**
  (`schedule/indexing.ml` `realize_call_args`, `engine/schedule.ml`
  `call_arg_node`). A call's argument is storage: a buffer, or a contiguous
  window of one, which reaches the call as a byte view. Any other argument
  the call only reads is realized into a copy; one it stores into raises
  `Invalid_argument` naming the call. The rule holds for every call,
  whatever its body. The reference realizes the non-buffer arguments of
  bodies still to be lowered only, written or not, so a write through a view
  was lost; and its `create_schedule` rebuilds every call's arguments as
  their base buffers, relying on those copies, so a window reached a lowered
  body as the base buffer from offset 0. Tolk's `create_schedule` keeps a
  window over part of its buffer as the argument. The collectives pass their
  whole allocation. Coverage: `test/unit/engine/test_schedule.ml` "call
  arguments".

- **`Creation.shard` splits a replicated value where it lives**
  (`frontend/creation.ml` `shard`). The reference raises on any multi-device
  source. A value replicated on exactly the target devices is split without a
  copy: each device keeps its own shard of its replica, as the graph
  `Unshard(Shrink by the DEVICE range)`. That graph is the reshard from
  replicated to split which a fully sharded gradient needs and which a
  reduce-scatter lowering matches, and this is its one constructor. Any other
  multi-device source still raises. Consumer: `test/unit/engine/test_collectives.ml`
  (the fully sharded step's gradient and the reshard traffic test). Coverage:
  `test/unit/frontend/test_run.ml` "a replicated tensor splits where it lives".

- **Composed QR, Cholesky and triangular solves** (`frontend/linalg.ml`).
  Rune's `E_qr`, `E_cholesky` and `E_solve_triangular` handlers use these
  shape-unrolled graphs to compile linear algebra and its gradients. Coverage:
  `test/unit/frontend/test_linalg.ml` and Rune's JIT factorization/gradient
  cases. Retain while these consumers need compiled factorizations; reconsider
  if equivalent operations gain a shared upstream implementation.

- **Queue configs** (`device.mli` `queue.config`, `engine/realize.ml`
  `queue_config`). The reference keeps compiled queues per process and has no
  persistent compiled-schedule cache. Tolk's backends render the state their
  queue encoders read: on AMD the ring kind (AQL or PM4), the interface (AM or
  KFD), `WAVES_PER_SH` and the ring sizes; on NV the ring entries, and also the
  channels' work-submission tokens and the per-thread local memory its QMD
  templates embed, which are per process, so a stored NV queue is served only
  to a process whose values match; nothing on Metal and CUDA. `queue_config`
  adds profiling, `ALL2ALL` and `HCQ_NUM_SDMA`. Consumers: tolk's queue
  template cache and rune's jit cache key. Coverage: `test_hcq2` (another
  config compiles anew) and `test_jit_cache` (PROFILE). Reconsider when NV
  reads its tokens and local memory through link-time patches, and if rune's
  cache stops storing compiled queues.

- **`program_config` keys the plain environment settings that change a
  program** (`engine/realize.ml` `program_config`). Besides the context
  variables the reference keys, it keys `MV`, `MV_BLOCKSIZE`,
  `MV_THREADS_PER_ROW`, `MV_ROWS_PER_THREAD`, `OCCUPANCY_FLOOR`, `DMC`,
  `EXPAND_SSA` and `ALIGNED` on their raw text, and `ALLOW_HALF8` as the
  value tolk read at startup. The reference reads them once per process
  through a cached getenv and keys only context variables; tolk reads them per
  call, and rune's disk cache serves programs across processes. Consumer:
  rune's jit cache key. Coverage: `test_jit_cache` "program settings".
  Reconsider if tolk's getenv becomes process-constant and rune's cache keys
  the environment itself.
