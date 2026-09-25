# Divergences

Intentional divergence from the tinygrad reference.
Retained rulings from the September 2026 audit; unresolved gaps live in
[TODO.md](TODO.md).

## OCaml representation and lifetime

- **Multi-axis reshapes use each axis's own shard count.** The frozen target
  reuses the last range's count while constructing all local dimensions, so a
  `[2; 4]` tile sharded 2-by-3 cannot reshape its `[4; 12]` logical shape to
  `[4; 3; 4]`. Tolk keeps the owning range with each axis and produces a local
  `[2; 1; 4]` view. Consumer: per-thread fragments and multi-axis device tiles.
  Coverage: `test_multi` unequal-count reshape and six-device tiled gather.
  Reconsider when upstream fixes the range lookup.

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

- **NV host waits are bounded and preserve in-flight command storage after
  timeout.** The generated host program shares AMD’s latched failure state and
  masks dependent writes, preventing a failed replay fence from corrupting the
  QMD or its arguments. Coverage: NV host-executed stalled replay checks the
  timeline, signal slots, descriptor and command bytes. Hardware recovery remains
  open. Reconsider when upstream provides equivalent timeout ownership rules.

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

- **Host access and direct cross-device binding wait for existing importers.**
  Tolk still exposes asynchronous `Device.prog` dispatch outside compiled queue
  dependencies. Waiting for the importing devices prevents CPU reads or writes
  from racing a kernel that holds mapped storage. Coverage: the storage mapping
  lifetime test and hardware-gated host-view dispatch tests. Reconsider when all
  direct dispatch participates in the same byte-interval dependency protocol.

- **PCI cleanup releases owned system-memory virtual ranges and the CPU
  view's address.** The frozen reference frees only device-memory ranges and
  assumes the GPU address is the CPU mapping address. BAR mappings can differ;
  retaining either allocation leaks page tables or host mappings. Consumers:
  AMD/NV pinned staging and CPU-visible queue storage. Coverage: repeated
  system/peer mapping release in `test_memory` and hardware-gated AMD CPU-map
  release. Remove this ruling when upstream releases both resources correctly.

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

- **The refusal of a zero-byte allocation sits in `Device.Buffer.allocate`;
  the reference asserts it in `Allocator.alloc`.** Tolk's allocators are
  records with no shared entry point like tinygrad's base class, so the check
  covers every buffer allocation, external pointers and LRU-wrapped allocators
  included, but not a runtime calling its allocator directly: the CUDA, AMD and
  NV staging copies, all of positive size. Reconsider if allocators gain a
  shared entry point.

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
  original dispatches' argument slots as metadata, without retaining another
  executable graph. Rune's donation and Metal replay suites cover the
  consumer. Remove this metadata if reuse analysis moves before queue
  compilation or upstream exposes equivalent access information.

- **`split_reduceop` leaves a one-hot sum whole** (`schedule/rangeify.ml`
  `is_one_hot_sum`). The reference splits any reduce whose input is 32768
  times its output, in the tensor graph, and collapses a gather's one-hot
  reduce to a gated load later, per kernel. For the 65536-row regression, the
  sum becomes 256 chunk sums behind a `contiguous`, each collapsing
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
  product; elsewhere none yet. Coverage:
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

- **Composed QR, Cholesky and triangular solves** (`frontend/linalg.ml`).
  Rune's `E_qr`, `E_cholesky` and `E_solve_triangular` handlers use these
  shape-unrolled graphs to compile linear algebra and its gradients. Coverage:
  `test/unit/frontend/test_linalg.ml` and Rune's JIT factorization/gradient
  cases. Retain while these consumers need compiled factorizations; reconsider
  if equivalent operations gain a shared upstream implementation.
