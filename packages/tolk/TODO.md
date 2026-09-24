# TODO

Open work for the tinygrad migration from `baa6148066f1a29f56bb870f6f139c15bb3f495f`
to `471a3aeb6924257d5e9bf321f5ff0a519163f18e`. Intentional differences belong in
[DIVERGENCES.md](DIVERGENCES.md). Remove an item when its acceptance tests pass.

## Reference and coverage

- Migrate every Python reference driver to the target API and generate the
  complete corpus separately. Attribute every changed expectation to its owner;
  require exact source parity for supported renderers.
- Add reference cases for image loads/stores, `multi_stack`, 128³ Metal WMMA,
  weak-integer overflow with movements, sliced aliases and symbolic copies.
- Minimize the CUDA-only `Coalesce: multiple stores to the same offset` report
  and verify it against the target coalescer.
- Port missing custom-kernel cases: sharding, duplicate arguments, anonymous
  allocations, source kernels, reshape/flip/slice, overlapping assignments,
  invalid stores and schedule order. Exercise gradient variants through Rune.

## Scalar semantics and UOp contracts

- Make a `Bitcast` to or from an emulated float8 act on the stored byte. Today
  the float decomposition decodes the element through float16 and back, which
  flushes subnormals and clamps infinities, so rune refuses a compiled float8
  `Nx.bitcast`.

- Adopt derived dtypes, weak CONST plus typed CAST, final ParamArg/CallInfo/
  ProgramInfo metadata, ALLOC, effectful CALL, scalar binding effects and
  void RANGE/BACKEDGE. Remove deleted operations and stale enum/cache formats.
- Adopt the final axis kinds. Port sorted-axis
  UNSHARD and the corresponding COPY/CALL/WMMA spec rules with their producers.
- Port final weak commitment/lowering and remaining symbolic rules.
- Centralize shape, numel, range and backward-slice properties. Remove parallel
  reconstruction and silent guesses in broadcast, callify, indexing, stage
  buffer sizes, renderer widths, range metadata and OS detection.
- Measure rewrite performance and long-lived memory use with weak node caches.

## Preparation and scheduling

- Introduce the final preparation owner and move early rewrites out of
  rangeify/callify. Port disk/view rules.
- Normalize explicit allocations and calls, lexical scalar formals and the
  separate becomes map. Handle precompiled calls without preallocation.
- Track RAW/WAR dependencies for sliced/overlapping assignments, self-copy,
  nested calls and shared aliases; reject cycles.
- Replace SLICE memory-plan views with SHRINK/BITCAST byte offsets. Unify
  contiguous-view folding and test leading-dimension and symbolic views.
- Port final sharding/indexing/allreduce ownership and hierarchical allreduce;
  cover symbolic maximum sizes and call argument slots. Complete the narrow
  frontend sharding surface.
- Adapt Rune staged-scan and indexed-scatter extensions to the new call/storage
  protocol. Revalidate the large one-hot gather guard at the split threshold
  and place it with preparation.

## Optimizer and rendering

- Replace optimizer actions with TC/SPLIT/PADTO/SWAP and port split-target,
  reduction-range, shared-memory, padding and rollback rules.
- Move tensor-core descriptions to the renderer and adopt fragment-derived
  layouts. Converge large WMMA accumulator ordering at the optimizer/expander;
  port CUDA MMA, AMD MFMA/FP8 variants and Metal BF16 support.
- Port final gpudims, slot allocation, range merge, gating and WAR barriers.
  Preserve WARP dimensions and symbolic extents.
- Canonicalize image coordinates across producer, gater, coalescer and renderer.
  Preserve access flags and volatile metadata during coalescing.
- Port spec-safe compact-float load/store emulation and long weak-constant
  splitting. Use float32 arithmetic for FP8 emulation and port the remaining
  division/modulo and transcendental fixes under live spec verification; remove superseded casts at the owning pass.
- Render host-call ABI, volatile parameters, void loops and final constants.
  Coordinate Metal's argument-struct ABI with its runtime binding.

## Storage, execution and devices

- Adopt BufferStorage, HostAllocator, per-device mappings and typed dispatch/
  transfer identities. Remove the nativeint-only transfer seam and Metal token
  workaround; preserve view lifetimes and 64-bit offsets. Represent zero-byte
  storage explicitly so empty inputs can participate in JIT capture.
- Port HCQ2 queue construction, byte-interval dependency tracking, compile/link/
  run phases and retained JIT execution. Replace old graph APIs instead of
  implementing the deleted upstream graph architecture.
- Distinguish ordinary runtime caches from transient timing links and modules;
  release transient resources deterministically after queued work completes.
- Migrate CPU host calls and threadless tensor kernels; port ELF/TinyELF and
  x86 out-of-range relocation trampolines. Rebaseline CPU matmul performance.
- Migrate Metal/CUDA queues and argument bindings; implement CUDA peer enablement
  and synchronized cross-device transfer with unsupported-peer fallback.
- Migrate AMD queue descriptors, AQL/multi-XCC, race/recovery fixes and consumed
  firmware/register tables. Port NV channel/descriptor, semaphore, GSP and
  compute submission fixes, including compute hunks in video-labelled commits.
- Make AMD/NV frees safe inside GC finalisers. A `Device.Buffer` finaliser runs
  at any allocation and reaches the raw `free` for buffers that bypass the LRU
  cache (`nolru`, LRU=0), so it can land inside any HCQ device operation.
  Between `Timeline.next_timeline` and `submit` its synchronize waits for a
  value not yet submitted, times out and latches `error_state`. On driver-less
  AMD and NV it also re-enters `Memory.vfree`/`valloc`, corrupting the TLSF
  allocators and page tables. Apply the CPU runtime's `after_queue` rule over
  a window that covers every HCQ device operation (alloc, free, call, copy,
  signal and kernarg paths): a free made inside it runs at the next
  synchronize, after its wait. Waiting only for the last submitted value is
  unsound: the owning buffer can be finalised while the packet being built
  still targets its memory. Needs hardware validation.
- Fix PCI host/multi-die mappings and NV per-interface storage metadata, peer
  mapping and ring placement.
- Generate real hsaco/cubin fixtures with compiler provenance. Validate kernel
  dispatch, copy/compute dependencies, replay, peer transfers and recovery on
  the relevant hardware; skipped tests are not hardware evidence.

## Frontend and consumers

- Review whether Tolk needs its `nn` layer. Design shared safetensors ownership
  with Nx and model/state ownership with Kaun, avoiding duplicate codecs and
  additional JSON dependencies.

- Revalidate scan, scatter and external-buffer contracts under the new call
  and storage protocol.
- Review the audited Rune residency, uploads, memory planning and symbolic
  placeholders against the final storage/call protocol. Carry applicable
  correctness and lifetime regressions into consumer validation.
- Review upstream gradient, Conv2d, optimizer, GPT-OSS, GGUF/quantization and AMD
  custom-kernel changes against actual Rune/Kaun consumers. Port applicable
  correctness fixes and measure accelerator candidates on supported hardware.

## Beam, concurrency and acceptance

- Add a deterministic beam regression for reconsidering candidates rejected
  by the per-step compute filter; measure search cost and selected kernels
  at the upstream stopping threshold.
- Share a bounded compilation worker facility across beam and ordinary lowering;
  snapshot context and handle cancellation/timeouts and errors safely. Respect
  affinity/container limits and audit nested contexts and concurrent caches.
- Port final beam actions, heuristics, explicit variable values, device-aware
  compilation and dynamic cache policy. Measure search cost, chosen-kernel
  latency, JIT replay, allocations and handle counts on consumer workloads.
- Remove closed divergence rulings and record retained ones with current
  consumer, test and reconsideration criterion. Adopt the frozen reference only
  when drivers, expectations and implementation agree.
- Run source parity, scalar/spec/OOB tests, lifetime stress, backend execution
  and relevant Rune/Kaun suites. Keep every unresolved difference attributable.

## Separately scoped capabilities

- Decide whether a concrete consumer warrants additional renderers/backends
  (ISA/x86, LLVMIR, PTX/NVCC, NIR/NAK, WGSL, OpenCL, DSP, HIP runtime, QCOM,
  NumPy runtime, RDMA/BNXT, USB/remote), low-level SQTT/PMC/PMA profiling, NV
  video or additional frontend APIs. Port shared compute fixes independently
  of these capability decisions.
