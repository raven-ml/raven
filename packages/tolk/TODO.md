# TODO

Migrate the reference from `baa6148066f1a29f56bb870f6f139c15bb3f495f` to the
frozen target `471a3aeb6924257d5e9bf321f5ff0a519163f18e`. Intentional differences
belong in [DIVERGENCES.md](DIVERGENCES.md). Remove work when its acceptance
criteria pass. The remaining milestones below define migration completion;
separately scoped work does not block it.

Work in milestone order. Defer individual renderer discrepancies, optimizer
policy and accelerator tuning until storage and execution use the shared protocol, unless
one blocks that path. Use focused tests during implementation and the broader
consumer suites at milestone boundaries. Commit coherent architectural changes
with their rationale and validation; commit count is not an acceptance metric.

## 2. Migrate storage, execution and existing consumers

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

- Complete AMD/NV fault-reporting and recovery handoffs for retained
  submissions. Validate NV channel retirement across independently linked
  batches and direct dispatches on hardware, including kernel-argument arena
  reuse during long asynchronous batches.
- Calibrate GPU and CPU profiling clocks for cross-device trace alignment.
  Validate staged peer transfers and shared host signals on small-BAR devices.
  Validate AMD AQL/multi-XCC
  dispatch and direct ring/staging reuse under long asynchronous batches.
  Complete AMD race/recovery fixes and consumed firmware/register
  tables.
  Reconcile scratch growth across retained and multi-device links. Port NV channel/descriptor, semaphore, GSP and compute submission
  fixes, including compute hunks in video-labelled commits.
- Complete PCI multi-die mappings, including harvested AID discovery and
  exclusion of dead memory hubs and doorbell routes, and NV ring placement. Finish rollback of
  failed KFD/NVK device initialization;
  inject driver mapping failures to validate allocation unwinding. Validate host/device placement,
  host/peer mappings and CPU mapping cleanup on large-BAR and small-BAR hardware.
  Preserve tinygrad's default PCI selection.

Acceptance: every existing backend builds against the shared protocol and no
consumer depends on the old graph or storage interfaces. CPU and Metal execute
kernels, copies and replay with correct dependencies and lifetimes; Rune and
Kaun consumer suites pass. Generate real hsaco/cubin fixtures with compiler
provenance. Validate AMD/NV/CUDA dispatch, replay, peer transfers and recovery
on their respective hardware. Record unavailable hardware as an explicit open
acceptance requirement; skipped tests are not execution evidence.

## 3. Close parity and adopt the reference

- Make a `Bitcast` to or from an emulated float8 act on the stored byte. Today
  the float decomposition decodes the element through float16 and back, which
  flushes subnormals and clamps infinities, so rune refuses a compiled float8
  `Nx.bitcast`.

- Migrate every Python driver to the target API and generate the complete
  corpus separately, including AMD/NV queue drivers. Attribute every changed
  expectation; require exact source parity for supported renderers.
- Reconcile remaining intermediate IR/GROUP and source differences in FP8
  `sm_80`, `rangeify`, `moe_gather_block`, `softmax_sink`, `swiglu_clamped`,
  `topk_rounds` and `multi_output`. Reconcile CPU/Metal `lorenz_fold` ordering
  and Metal `vectorize_index` after the constant representation is migrated.
  Reconcile CUDA kernel ordering in `multi_allreduce_ring` and render
  SPECIAL launch-size comments from final constants.
  Remove the Llama driver's manual staging shortcuts: its attention-score
  kernel is named `E_2`, whereas the target tensor graph produces
  `r_2_2_2_2_2`.
- Add reference cases for image loads/stores, `multi_stack`, 128³ Metal WMMA,
  weak-integer overflow with movements, sliced aliases and symbolic copies.
  Minimize the CUDA-only `Coalesce: multiple stores to the same offset` report
  and compare it with the target coalescer.
- Validate large WMMA accumulator ordering at optimizer/expander boundaries,
  including BF16/FNUZ across supported renderers. Port remaining gpudims, slot
  allocation, range merge, gating and WAR barriers while preserving symbolic
  extents. Canonicalize image coordinates across all producers and consumers,
  and render final constants.
- Remove remaining parallel property reconstruction and silent guesses in
  renderer widths, view offsets, stage buffer sizes and range metadata. Port
  remaining symbolic rules and measure rewrite performance and long-lived
  memory use with weak node caches.
- Add deterministic beam coverage for reconsidering candidates rejected by
  the per-step compute filter. Measure search cost and selected kernels at
  the upstream stopping threshold. Port remaining heuristics, device-aware
  compilation and dynamic cache policy. Share bounded compilation workers
  with lowering, with context snapshots, cancellation/timeouts, errors,
  affinity/container limits, nested-context audits and concurrent-cache
  coverage. Synchronize device opening and runtime caches; concurrent callers
  currently mutate the shared registry and execution Hashtbls without locking.
  Isolate schedule capture hooks and Rune's shared upload scratch across
  concurrent callers as part of that audit.
- Review upstream gradient, Conv2d, optimizer, GPT-OSS, GGUF/quantization and
  AMD custom-kernel changes against current Rune/Kaun consumers. Port applicable
  correctness fixes; measure accelerator candidates on supported hardware.
- Update the TSan workspace's OCaml 5.4 pin for the current dependency set and
  run runtime/lifetime stress tests. Measure search cost, selected-kernel
  latency, JIT replay, allocations and handle counts on consumer workloads.
- Remove closed divergence rulings. Give retained differences a current
  consumer, test and reconsideration criterion. Update the reference pin and
  expectations only when drivers and implementation agree.

Acceptance: the complete reference corpus is regenerated reproducibly and
every difference is either closed or an explicitly justified divergence.
Source parity, scalar/spec/OOB tests, lifetime stress, backend execution and
Rune/Kaun suites pass. Review performance against the frozen target, the
hardware evidence and commit history, then rebase and push the completed
migration to main.

## Separately scoped work

- Review whether Tolk needs its `nn` layer. Design shared safetensors ownership
  with Nx and model/state ownership with Kaun, avoiding duplicate codecs and
  additional JSON dependencies.
- Decide whether a concrete consumer warrants additional renderers/backends
  (ISA/x86, LLVMIR, PTX/NVCC, NIR/NAK, WGSL, OpenCL, DSP, HIP runtime, QCOM,
  NumPy runtime, RDMA/BNXT, USB/remote), low-level SQTT/PMC/PMA profiling, NV
  video or additional frontend APIs. Port shared compute fixes independently
  of these capability decisions.
