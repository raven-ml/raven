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

- Move AMD/NV allocator transfers onto shared execution. Delete duplicate
  command builders and
  handoff bookkeeping once their callers migrate, preserving transfer ownership
  and ordering.
  Move host-mapped initialization/readback to direct host access before routing
  unmapped byte transfers through owned host buffers and shared STORE calls;
  queue image/command initialization currently calls allocator copyin, so a
  blanket shared-submission replacement would recursively link queues. Remove
  AMD's eager 32-by-2-MiB staging pool with its obsolete transfer path. Keep
  OCaml byte storage alive through completion without borrowing moving pointers.
  Preserve bounded waits and retirement for independently linked submissions;
  these are separate safety requirements.
- Port the target's distinction between one-shot linker ring allocations and
  retained command storage. Share NV code images within a compiled schedule
  and use one QMD/argument arena per run. Measure allocation counts and
  test independent retained links, descriptor alignment and replay patching.
- Validate deferred buffer finalization on AMD/NV hardware with `nolru` and
  `LRU=0`, forcing GC during allocation, mapped-buffer teardown, direct dispatch,
  compiled submission, signal reservation and kernarg reuse. Verify that waits
  target submitted work and PCI allocator/page-table operations cannot re-enter.
- Complete AMD/NV fault-reporting and recovery handoffs for retained
  submissions, including failures during direct publication and timed dispatch.
  Validate NV channel retirement across independently linked
  batches and direct dispatches on hardware, including kernel-argument arena
  reuse during long asynchronous batches.
- Validate AMD/NV/CUDA profiling clock alignment on hardware.
  Validate staged peer transfers and shared host signals on small-BAR devices.
  Validate AMD AQL/multi-XCC
  dispatch and direct ring/staging reuse under long asynchronous batches.
  Complete AMD race/recovery fixes and consumed firmware/register
  tables. Audit VF mailbox leases, gated register access and PF-only boot
  operations against the existing AMD device scope; justify any retained gap.
  Validate compiled PM4 scratch separation on multi-die hardware and reconcile
  scratch growth across retained and multi-device links.
  Validate NV channel/descriptor, semaphore and GSP compute submission behavior
  on hardware, including compute changes from video-labelled commits.
- Regenerate consumed AMD register tables and firmware hashes from the frozen
  target instead of old-pin inputs plus selected handwritten additions. Port
  the consumed SMU/PSP 13.0.15 branches coherently. Restore target
  firmware retrieval/cache behavior when installed blobs are missing or have
  different hashes, using existing fetch facilities without new dependencies.
  Replace test-driven busy-spin delays with actual production sleeps and a
  narrow test clock boundary.
- Complete PCI multi-die mappings: export AMD hive memory through XGMI peer
  addresses instead of its PCI BAR, and select BAR or fabric addresses according
  to the receiving device. Establish raw-PCI fabric identity and reachability;
  region index/count alone cannot prove a shared hive, and driver sysfs state
  cannot be assumed after takeover. Check mixed-vendor/cross-hive mappings and
  reject imports outside the receiver's virtual-address aperture.
  Complete rollback of firmware
  bootstrap failures before a booted interface exists. Validate post-boot
  AMD/NV queue/runtime rollback with injected hardware failures, including
  faulted queue retirement and doorbell mappings; inject driver mapping
  failures to validate allocation unwinding. Validate host/device placement,
  host/peer mappings and CPU mapping cleanup on large-BAR and small-BAR hardware.
  Preserve tinygrad's default PCI selection.

Acceptance: every existing backend builds against the shared protocol and no
consumer depends on the old graph or storage interfaces. CPU and Metal execute
kernels, copies and replay with correct dependencies and lifetimes; Rune and
Kaun consumer suites pass. Validate AMD/NV/CUDA dispatch, replay, peer transfers
and recovery on their respective hardware. Record unavailable hardware as an explicit open
acceptance requirement; skipped tests are not execution evidence.

## 3. Close parity and adopt the reference

- Audit the remaining rule ports for promotion: `symbolic.ml`, `divandmod.ml`,
  `postrange.ml` and the other codegen rule bodies build with `U.O`, which
  promotes nothing, where the reference's source uses its promoting operators.
  A ported body uses `U.Promoting` there, as the reduce-collapse rules do.
- Audit the shared ELF loader's allocatable `NOBITS` sections against the
  target's `PROGBITS`-only image. Linked HIP objects include a zero-filled
  `__hip_cuid` symbol; Tolk reserves its address and the linker padding, while
  the target omits them. Establish the actual CPU/custom-kernel consumer
  requirement, then justify the difference or remove it. Compare whole images
  in addition to the code/descriptor checks in the real AMD fixture.

- Make a `Bitcast` to or from an emulated float8 act on the stored byte. Today
  the float decomposition decodes the element through float16 and back, which
  flushes subnormals and clamps infinities, so rune refuses a compiled float8
  `Nx.bitcast`.

- Migrate every Python driver to the target API and generate the complete
  corpus separately, including AMD/NV queue drivers. Update the C-style fixture
  builder to emit committed constants (`cconst`); its bare boolean CONST
  currently renders an empty condition. Attribute every changed
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
  renderer widths, view offsets, stage buffer sizes and range metadata. Remove
  the C renderer's independent multidimensional stride reconstruction in
  favor of canonical flat indexes. Unify ordered range and range
  membership traversal; use checked central numel in placeholders, frontend
  tensors and rangeify. Remove ignored `dtype` arguments from source-derived
  `Uop.load`, `reduce`, `wmma` and `noop`, updating callers and contracts. Port
  remaining symbolic rules and measure rewrite performance and long-lived
  memory use with weak node caches.
- Replace `Uop.contiguous_view`'s separate movement interpreter with the
  target's flattened-index proof using shared movement and symbolic rewrites.
  Cover cancelling transposes, leading dimensions, symbolic bounds and byte
  offsets across bitcasts. Preserve pending effects when extracting views.
- Compare symbolic `STAGE` extent end to end: Tolk uses the active symbolic
  size, while the target reserves its maximum extent. Establish allocation and
  indexing requirements with a paired execution case before changing it.
  Test optimizer resource bounds with symbolic dimensions rather than silently
  treating unknown extents as one; distinguish a proven mismatch from a
  conservative optimization difference.
- Remove wrapping host `Int64` arithmetic from late comparison rewrites.
  The interval proof in `decomp_op` can turn an empty signed-64 interval into
  equality at `INT64_MIN`. Cover the local proof and its normal-pipeline
  reachability, plus adjacent negated-comparison rules, against target exact
  arithmetic. Audit PADTO and shared-memory products at the same boundaries.
- Use the target's sanitized `signed_char` name for RDNA3 int8 tensor-core
  fragments; distinguish source naming parity from fragment layout correctness.
- Add deterministic beam coverage for reconsidering candidates rejected by
  the per-step compute filter. Measure search cost and selected kernels at
  the upstream stopping threshold. Port remaining heuristics, including
  `TC_MIN_GLOBALS` global-occupancy preservation and its compilation cache key,
  plus device-aware compilation and dynamic cache policy. Share bounded workers
  with lowering, with context snapshots, cancellation/timeouts, errors,
  affinity/container limits, nested-context audits and concurrent-cache
  coverage. Join every started compilation domain before propagating a worker
  or spawn failure, so restored contexts cannot race with abandoned workers.
  Synchronize device opening and runtime caches; concurrent callers
  currently mutate the shared registry and execution Hashtbls without locking.
  Isolate schedule capture hooks and Rune's shared upload scratch across
  concurrent callers as part of that audit. Cover finalizers registered on
  another domain and systhreads sharing a domain: operation scopes currently
  prevent only same-domain GC re-entry and do not serialize device callers.
  Run the concurrent runtime and cache coverage under TSan.
- Review upstream gradient, Conv2d, optimizer, GPT-OSS, GGUF/quantization and
  AMD custom-kernel changes against current Rune/Kaun consumers. Port applicable
  correctness fixes; measure accelerator candidates on supported hardware.
- Measure search cost, selected-kernel latency, JIT replay, allocations and
  handle counts on consumer workloads.
- At the reference pin move, `find_bufs`' read/write cycle check keys on the
  pointer node (upstream `read_from.setdefault(buf, state:=idx.src[0]) is not
  state`), where the pin and tolk key on its op. Rune's remat barrier reads a
  materialised argument as `AFTER(AFTER(buf, STORE), cotangents)`, and a
  second-order backward can read it as `AFTER(buf, STORE)` in the same kernel:
  one op today, two states under the new check. `test_grad`'s remat group and
  `test_remat_memory` would then raise "cycle detected while indexing buffer";
  the barrier needs another form before the pin moves.
- Port the reference's `CallifyCtx.views` into `Callify.transform_to_call`:
  the reference records the byte views its copy fold creates, and
  `replace_input` replaces only those. Tolk's `replace_input` tries
  `contiguous_view` on every concrete SHRINK or BITCAST of the body, which
  for one over a split buffer now runs a `multi_pm` rewrite that returns
  `None`: correct, but work the reference skips.
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

- Unrolling the block kernel's constant inner loop of 8 on the CPU
  (`Op.block_matmul`, a `Split` of kind `Unroll` on the last axis) fails to
  lower with "invalid axis" in `full_rewrite_to_sink`, although Postrange
  applies it; the pinned CPU options unroll the outer loop over tiles instead.
  Acceptance: `Split` of the inner axis by 8 lowers and runs in
  `test_block_matmul`'s CPU values.
- Make stored NV queues process-independent. The encoder embeds the channels'
  work-submission tokens (`tolk_nv.ml` `tolk_hcq_gpfifo` call) and the
  device's per-thread local memory in each QMD template (`template_dev`); both
  come from this process. Read them through link-time patches, as AMD reads
  scratch through its `amd_scratch` placeholder, then drop `COMPUTE_TOKEN`,
  `COPY_TOKEN` and `SLM` from NV's queue config. Acceptance: an NV host test
  encodes one program for two devices that differ only in tokens and local
  memory and gets equal linears.
- Make emulated float8 conversions in compiled kernels agree with the
  constant codec. `decomp_dtype.ml`'s `f2f` ports the reference's
  decomposition, which flushes subnormals to zero both ways, `f2f_clamp`
  saturates overflow and infinities, and the `CAST` rule clamps without
  rounding. On an emulating device (CPU, Metal), against eager nx on
  float32 inputs:

  | Path | Input | Eager | Compiled |
  |---|---|---|---|
  | store `cast e4m3 x` | 2^-7, 2^-8, 2^-9, 1.5·2^-10 | same, 2^-9 for the last | 0 |
  | store | 500, inf | NaN | 448 |
  | fused `cast f32 (cast e4m3 x)` | 336, 1.0625, 1.5·2^-10 | 320, 1, 2^-9 | unchanged |
  | load `cast f32 x`, x : e4m3 | 2^-7, 2^-8, 2^-9 | same | 0 |
  | load, x : e5m2 | 2^-15, 2^-16 | same | 0 |

  Every e4m3 code from 0x01 to 0x07 reads as zero, so a compiled float8
  dequantisation on CPU or Metal loses them, while CUDA's native float8
  keeps them. Make `f2f` denormalise instead of flushing in both
  directions, round the `CAST` rule through `f2f` both ways, and decide
  overflow after rounding with no pre-clamp, as the codec does; the native
  CUDA and HIP constructors need a compare and select to stop saturating.
  Settle at the same time that tolk emulates float8 through `float32` where
  the reference uses `half` on renderers with native half. Record the result
  in DIVERGENCES.md. Land it before any compiled float8 path is claimed to
  match eager, and before RFC 0004's float8 weights run compiled on CPU or
  Metal. rune's `test_jit.ml` float8 sort and bitcast cases know part of
  this.

- Review whether Tolk needs its `nn` layer. Design shared safetensors ownership
  with Nx and model/state ownership with Kaun, avoiding duplicate codecs and
  additional JSON dependencies.
- Decide whether a concrete consumer warrants additional renderers/backends
  (ISA/x86, LLVMIR, PTX/NVCC, NIR/NAK, WGSL, OpenCL, DSP, HIP runtime, QCOM,
  NumPy runtime, RDMA/BNXT, USB/remote), low-level SQTT/PMC/PMA profiling, NV
  video or additional frontend APIs. Port shared compute fixes independently
  of these capability decisions.
