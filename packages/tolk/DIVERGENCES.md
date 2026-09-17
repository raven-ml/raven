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

## Scheduling defaults

Behavior changed relative to the reference; each keeps its env override.

- **`REDUCEOP_SPLIT_THRESHOLD` defaults to 16384; the reference defaults to
  32768.** The threshold is the reduce depth per output element below which a
  reduce stays fused into its consumer's kernel. Fusing a deep reduce into a
  kernel whose global range is set by a small output is catastrophic on a
  GPU: the kernel's few threads each re-run the whole reduction as a serial,
  latency-bound loop, and the reference's own heuristic (GROUPTOP amount 16)
  cannot group the result unless the reduced axis happens to divide by 16.
  With a [32, 25480] x [25480, n] dot-product reduce (n in 1..31, i.e. just
  below the reference threshold) inside a gradient graph, replay ran ~395x
  slower than a padded variant of the same math and could crash the GPU (Xid
  79); see the `tolk_shape_slowdown` reproducer in the workspace. 16384 keeps
  every reduce the reference splits split, and additionally bounds the
  residual serial depth at depth/256 for depths in [16384, 32768). Env still
  overrides.

- **The split's second-stage reduce materializes when its output is small
  (at most 4096 elements); the reference leaves it free to fuse.** Left
  free, the sum over the first stage's partials fuses into whatever consumes
  it, and a consumer whose global range exceeds the stage's output — a
  broadcast consumer, or an elementwise kernel over a much wider tensor —
  re-derives the whole divisor-deep partial sum in every thread. On the
  reproducer above that meant ~102,000 threads each re-reading a [32, 32,
  245] partials buffer (~100 GB of L2 traffic per launch, 189 ms replay
  where 18 ms was available). Large outputs keep fusing: a consumer indexed
  1:1 over them reads each partial once either way.

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

## Tolk extensions

Code tolk carries that the reference does not. Every site has a comment
containing the phrase "tinygrad counterpart" — `grep -rn "tinygrad
counterpart" lib` lists them. An extension needs a consumer; without one,
delete it rather than registering it.

- **`CALL(CUSTOM_FUNCTION "loop")` — the staged-scan loop**
  (`engine/realize.ml` `exec_loop`; `schedule/rangeify.ml` `find_bufs`
  walking `enter_calls:false` and `split_store` passing a precompiled CALL
  through as its own kernel; builder with its consumer in rune's `jit.ml`).
  Rune stages `Rune.scan` as one compiled body replayed per slice
  (`rune/doc/05-staged-scan.md`); the reference's answer to a recurrence is
  unrolling plus TinyJit, so there is nothing to port. The named-payload
  mechanism is upstream's own ("graph", "encdec", "hcq"); "loop" is a
  tolk-local name in it, and no tinygrad-shaped graph can reach the new
  branches. Pin moves must keep the two rangeify branches — a re-sync of
  `rangeify.py` will not find them upstream. The tolk corpus cannot build a
  loop call; rune's `test_jit.ml` scan groups are this extension's parity
  suite.
