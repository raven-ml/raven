# Divergences

Intentional divergence from the tinygrad reference.
Retained rulings from the September 2026 audit; unresolved gaps live in
[TODO.md](TODO.md).

## OCaml representation and lifetime

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
