# Divergences

Intentional divergence from the tinygrad reference.
Retained rulings from the September 2026 audit; unresolved gaps live in
[TODO.md](TODO.md).

## OCaml representation and lifetime

- **Automatic buffer teardown defers GC re-entry during device operations.**
  OCaml may run a buffer finalizer during another allocator or queue operation;
  releasing that buffer can re-enter the same lock or alter state mid-update.
  `Storage.with_operation` defers such automatic releases to the outer operation
  boundary. Explicit release still reports errors. Coverage: `test_device`
  finalization during device calls and failed teardown, allocator tests, and
  Metal lifetime checks. This scope is not cross-domain or systhread locking;
  concurrency remains open in TODO.
  Reconsider if native ownership makes automatic teardown non-reentrant.

- **Failed automatic teardown retains its owner until process exit.** An
  allocator may release only part of a mapping or native object before raising;
  retrying can double-free it, while dropping its owner can release backing still
  reachable by the device. A later successful synchronization proves completion,
  not which teardown effects happened. The allocator contract has no general
  recovery or device-destruction acknowledgement, so uncertain owners remain
  retained and the original exception is propagated. This trades storage after
  a teardown failure for explicit lifetime safety, without another cleanup
  protocol. Coverage: `test_device` reports the failure once, keeps the owner
  alive through collection, and never retries. Reconsider when a concrete
  backend recovery consumer can prove complete destruction of those resources.

- **Metal publishes completion after checking command status and timestamps.**
  The target polls a GPU event and collects command buffers separately. Tolk's
  native queue interpreter uses completion to authorize host access and storage
  retirement, so it collects commands in submission order before advancing its
  timeline. Failed commands latch an error and retain uncertain backing.
  Coverage: native completion ordering, timestamp visibility, failure and
  retirement tests, plus real Metal queue replay. The target's fused 1024-row
  matrix-product reduction was also observed returning partial values after
  an `Impacting Interactivity` command failure; successful event waits do not
  establish successful execution. Reconsider when event
  signaling alone provides the same failure and lifetime guarantees.

- **AMD queue retirement propagates HQD dequeue timeouts.** The target can
  suppress that timeout, but Tolk uses successful retirement to decide whether
  GPU-reachable backing may be freed during rollback. Reporting success without
  quiescence would invalidate that ownership decision. Coverage: scripted
  `test_amd_amdev` teardown failures; hardware retirement remains in TODO.
  Reconsider when equivalent quiescence can be established after the timeout.

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

- **Provably overlapping copies use the bounded host fallback.** Native DMA
  and parallel byte-copy kernels need not preserve overlapping input. Tolk
  retains its existing directional-copy behavior for views sharing a root or
  overlapping addresses in the same canonical device address space. Queue
  preflight checks original STORE calls before publication and falls back in
  order. Coverage: bounded forward/backward copies and shared-submission tests
  using both views and independent external-pointer wrappers. Reconsider when
  native overlap-safe copies are available; unrelated device address spaces
  are not assumed to alias.

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
  values for a `cat` of 16 tensors. Tolk's shared Metal submission encoder
  dispatches kernels above 15 arguments directly between indirect ranges,
  with buffer barriers and the same resource ownership and completion fence.
  The private threshold participates in the queue configuration. At the time of the
  ruling, no kernel of the rune Metal tests (apart from this coverage) or of
  the tiny gpt-oss validation exceeded 10 arguments. Coverage: Rune's
  `test_jit_metal` concatenation of 16 sorts between queued kernels, and the
  Metal runtime's argument structures of 15 buffers (indirect), 16, 29 and 33
  (direct), which rebind between calls with and without profiling. Reconsider when Apple fixes the
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
  partials in box order on every device and stores each partial once: the
  other boxes copy it and its own device reads it back rather than computing
  it again inside the fold. Traffic is unchanged. Consumer: RFC 0005's
  replicated values, which readers take from any one device. Coverage:
  `test_multi` hierarchical replicas on 6 and 8 devices with 1 and 2 devices
  per box. Reconsider when upstream fixes the fold order.

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
  a channel sequence and leaves a FIFO slot empty. Retained replay waits before
  patching its old command tail, and synchronization drains both channels.
  The frozen target does not bound independent submissions this way. Native
  waits latch timeout and suppress publication. Coverage: `test_runtime_nv`
  FIFO saturation and resumption, independent retained batches, delayed command
  tails and 32-bit completion rollover. Hardware validation remains open.
  Reconsider when upstream provides equivalent occupancy and retirement rules.

- **AMD/NV submissions retain timeline addresses across rollover.** Before
  the low timeline dword reaches its comparison limit, Tolk drains work and
  advances the high-word epoch at the same address. Independently retained
  host fences remain monotonic while GPU dword waits restart safely. AQL scratch
  updates touch only scratch fields, preserving live dispatch counters. Coverage:
  mapped producer tests, two epoch transitions and compiled AMD host replay
  across rollover. Hardware acceptance remains in TODO. Reconsider when
  upstream provides an equivalent rollover protocol for retained submissions.

- **Host access waits for existing importers.** Host reads/writes wait for
  importing devices. CPU runtime calls enforce this boundary too; address
  binding itself does not wait. Compiled host submissions use
  `Device.queue_runtime` and their encoded fences, so replay does not
  accidentally drain every device. They wait for recorded foreign accesses
  outside their own timelines, including separate groups sharing host memory.
  Coverage: storage mapping lifetime tests, CPU binding waits, host queue replay
  with zero implicit owner waits, uncovered host-writer retirement before
  submission, and real Metal host-view dispatch. Reconsider when all CPU
  execution participates in the same byte-interval dependency protocol.

- **OCaml byte transfers use synchronous owned staging.** Unlike Python's
  stable memoryviews, OCaml bytes cannot be borrowed by asynchronous device
  work. `Storage.copyin` and `copyout` use direct synchronized host access when
  available, otherwise one owned host allocation and ordinary STORE calls.
  Offset-capable storage bounds staging to 64 MiB and waits before reuse or
  host reads; command initialization remains directly host mapped. Coverage:
  `test_storage_copy` chunk boundaries and failed-wait ownership, CPU byte
  access without the engine, and Metal copies after asynchronous kernels.
  Reconsider if a nonmoving public host-buffer API replaces these conveniences.

- **Staging slots belong to a prepared schedule.** The frozen target caches
  one host staging allocation per host device. Independently retained Tolk
  batches fence their own command storage, so sharing those slots would let
  one batch overwrite transfers belonging to another. Each staged schedule
  owns two 64 MiB slots, reused with byte-interval queue dependencies. Coverage:
  `test_hcq2` checks chunk tails, read-before-reuse waits, independent storage,
  replay rebinding and input lifetime. Reconsider when a shared staging pool
  has reservations spanning independent submissions.

- **Multi-die PM4 dispatch partitions resident-wave scratch.** The frozen
  target offsets each die by `private_segment_size / xccs`, although that field
  is bytes per thread. Tolk uses the requested resident-wave allocation divided
  by the die count, matching its direct dispatch layout. Every die needs room
  for its resident CUs, waves and lanes, including the 128-byte thread minimum.
  Consumer: multi-XCC PM4 dispatch with `AMD_AQL=0`. Coverage: the compiled
  queue regression checks all eight predicated addresses and their disjoint
  resident-wave regions. Hardware validation remains in TODO. Reconsider when
  upstream computes the offset from resident-wave storage.

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
  virtual reservations because the device may still address them. Failed import
  rollback also retains the source, blocks explicit deallocation and propagates
  the original error; neither successful unrelated work nor GC authorizes retry.
  Failed PCI claims release their lock and acquired descriptors so they can
  be retried. Bootstrap failures release descriptors and per-device KFD
  events. KFD's process-wide event page remains cached; an ambiguous registration
  failure is latched because the kernel may already retain that page. Failed
  KFD/NVK and booted PCI device construction retire queues before releasing
  their buffers; a failed queue stop retains storage and suppresses late
  finalizers. PCI teardown tracks partially programmed SDMA rings, removes
  failed AMD interrupt registrations, releases NV doorbell mappings and
  disables failed runtime shutdown hooks. Faulted PCI devices retain storage
  unless queue retirement can be established. NVK
  unregisters channels with the installed driver's layout, then unwinds UVM
  registrations, control mappings and the per-device RM object tree.
  Borrowed host mappings remain owned by the caller. Consumers:
  allocation retries and long-lived accelerator sessions. Coverage:
  `test_memory` allocation, zeroing, entry-write and flush failures, including
  adjacent mappings, precreated tables and source retention after failed import
  rollback; `test_amd_system` covers reverse
  rollback order, acquisition failures, cleanup failures, descriptor release,
  PCI claim retries and buffer finalizers after successful or failed queue retirement;
  `test_nv_tables` covers 570/580/610 unregister layouts; `test_amd_amdev`
  injects a register failure after enabling an SDMA ring and checks teardown.
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
  Consumers: compiled submission and JIT replay. Allocator-upload handoffs
  remain transitional work in TODO. Coverage: driver-independent submission
  compilation and CUDA runtime replay/transfer cases (hardware acceptance is
  still open in TODO). Reconsider the helper boundary if a native driver
  binding can provide the same calling convention and failure lifetime.

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

- **Cache eviction uses the device of the timed candidate.** The target's
  fallback materializes 1024×1024 float32 ones on the frontend's default
  device. `Realize.time_call` has an explicit device and no frontend dependency;
  its equivalent fill runs there so a non-default candidate evicts the correct
  cache. Coverage: `test_runtime_cpu` verifies the fill, scoped BEAM suppression,
  and absence of capture/statistics side effects. Reconsider if the engine gains
  a shared default-device policy that guarantees the same device selection.

- **Timing buffers bypass the LRU cache (`nolru`) and are freed per kernel;
  the reference allocates them normally.** It can: refcounting frees
  promptly. Under a lazy GC the exact-size LRU cache hoards every searched
  shape, and driver module loads OOM without triggering the allocator's
  failure flush.

## Numerics

- **Float constants fold with nx's codec.** The reference saturates a finite
  float8 value past the largest finite one and rounds a bfloat16 constant
  through binary32 (`struct.pack('f')`), rounding twice. tolk folds with
  `Nx_dtype.Scalar.encode`, the encoder nx's eager stores use, which rounds
  once from binary64, and a result past the largest finite value becomes the
  format's infinity, or NaN in the formats without one. RFC 0005's Law 1 lets
  a compiled result differ from the eager one only by the engine's rounding,
  and a constant is not rounded by the engine. Consumers: `Const` and `Bound`
  folding in every compiled graph, through `Dtype.truncate_float`. Coverage:
  `test_dtype`'s float8 and bfloat16 cases; nx's `test_float_codecs`, which
  checks the codec against an exact reference over every float8 code and
  float32 and float64 sweeps. Reconsider only if rune stops running eager code
  on nx.

- **Integer bounds account for wrap-around.** The reference's interval
  arithmetic is exact: Tolk bounded `uint8` `x - 1` over [0, 255] by
  [-1, 254], so `x - 1 < 255` folded to true though x = 0 gives 255, and the
  rule `c0 + x < c1 -> x < c1 - c0` turned `uint32` `x - 1 < 5` into `x < 6`,
  true at x = 0 where the program compares 4294967295 with 5. Tolk widens an
  integer interval that reaches past its dtype's range to the whole range,
  and applies that rule at a fixed width only when neither `x + c0` nor
  `c1 - c0` can wrap. Index arithmetic is `weakint`, whose range no kernel
  reaches, so it keeps every fold. Coverage: rune `test_jit` and
  `test_jit_metal` "integer comparisons read wrapped values". Remove this
  ruling when upstream reasons modularly.

- **Symbolic keeps float sums and products as written.** The reference folds
  `(x + c1) + c2` into `x + (c1 + c2)`, moves constants to the tail of a
  chain and factors `x*c0 + x*c1` into `x*(c0 + c1)` at every dtype; in
  float that changes the rounding the program asked for: `(x + 1e8) - 1e8`
  compiled to `x` (1 at x = 1 where eager gives 0) and `(x * 1e30) * 1e-30`
  to `x`. Tolk applies these rules at integer and boolean dtypes only. The
  arithmetic tolk's own decompositions introduce is theirs to arrange, so
  they state their constants combined (`tanh`'s exponent is one multiply by
  `-2 / ln 2`, `erf`, `sinh`, `cosh` and `atanh` likewise). A program's own
  constant chains now run as written: on the CPU a tanh-form gelu costs 14%
  more (4M float32, 14.85 against 16.9 ms, from its `1 + tanh` across
  tolk's `2 s - 1`), a Lorenz step half as much (9.07 against 4.45 ms) and
  swiglu 7% less; softmax and layer norm are unchanged. Coverage: rune
  `test_jit` and `test_jit_metal` "float constants keep their grouping".
  Remove this ruling when upstream keeps float association.

- **The listed float identities apply only where IEEE keeps them.** The reference
  folds `x / x` to 1, `x * 0` to 0, `(x * y) / y` to `x`, `x + 0` and
  `x - 0` to `x`, splits `1 / (x * x)` and `1 / (x * c)` into products of
  reciprocals, merges `(x / y) / z` into `x / (y * z)`, and rewrites
  `x * (1 / (1 + x))` as `1 - 1 / (1 + x)`, at every dtype. In float these
  change the program: `x / x` is NaN at 0, inf and NaN, `x * 0` is NaN at
  inf and -0 at negative x, `(x * y) / y` is NaN where `x * y` overflows,
  `-0 + 0` is +0, and `1 - 1 / (1 + x)` cancels every digit of `x / (1 + x)`
  for small x (1e-8 gave 0). Tolk drops the rules on reciprocals, which only
  float arithmetic reaches (at integers x * (1/x) is not 1 either), applies
  `x * 0` at integer and boolean dtypes (`exact_algebra`), and at float keeps
  `x * 1`, `x + -0`, `x - +0`, `x + x -> 2x`, and `x + 0` or `x - 0` where
  bounds exclude zero. No CPU kernel of sigmoid, silu, swiglu, softmax, gelu,
  layer norm or Lorenz changes time. Coverage: rune `test_jit` and
  `test_jit_metal` "float identities hold only where IEEE keeps them". Remove
  this ruling when upstream restricts these rules.

- **An ordered comparison is false at NaN.** The reference writes `a >= b` as
  `not (a < b)` and `a <= b` as `not (b < a)`, true when either operand is
  NaN: compiled `x >= 5`, `x <= 5` and `x >= x` were true at NaN where eager
  is false, and `where (x >= 0) x 0` kept a NaN eager drops. At float Tolk
  writes them `(a > b) | (a = b)` and `(a < b) | (a = b)`; integers keep the
  negation, which is exact there. Coverage: rune `test_jit` and
  `test_jit_metal` "ordered comparisons are false at NaN". Remove this
  ruling when upstream keeps NaN out of `>=` and `<=`.

- **Max propagates NaN and keeps its second operand on a tie.** The
  reference lowers `max(x, y)` to `where(x < y, y, x)`, which drops a NaN `y`
  and keeps `x` on a tie, and folds a max by bounds that exclude NaN:
  compiled `max(|x| + 1, sin(x * inf))` was 3 where eager is NaN, and
  `max(-0, +0)` was -0 where eager is +0. At float Tolk lowers it to
  `where(y < x, x, where(x != x, x, y))`, which constant folding reduces to
  one comparison for a constant first operand, or to `where(x < y, y, x)`
  when the second operand is a nonzero constant other than NaN, and folds by
  bounds only when the dropped operand is a constant other than NaN, by a
  strict bound when it is the second. Clip costs what it did; relu 3% more
  on the CPU. Coverage: rune `test_jit` and `test_jit_metal` "max propagates
  NaN". Remove this ruling when upstream's max propagates NaN.

- **A float zero keeps its sign through negation, selects and sums.** The
  reference distributes `-(x + y)` into `(-x) + (-y)`, merges
  `c ? t : 0 + c ? 0 : f` into `c ? t : f`, and folds a sum's `+0` start
  away when the reduction is unrolled or over a size-one axis. Each loses a
  zero's sign: `-(x + 3)` at x = -3 is -0 (compiled +0), `t + 0` at t = -0
  is +0 (compiled -0), and a sum of -0s is +0 (compiled -0). Tolk
  distributes the negation at integer and boolean dtypes only, merges the
  selects at float only with -0 zeros, and starts a float sum from +0 where
  no accumulator does: a reduce left without a loop folds its lanes onto +0
  (a partial sum nested inside an accumulating loop carries it too, one add
  per iteration), and a sum with nothing left to reduce adds +0. Coverage:
  rune `test_jit` and `test_jit_metal` "zeros keep their sign". Remove this
  ruling when upstream keeps these signs.

- **A sum reduction's association is unspecified; its hoists stay.** The
  preceding rulings keep a program's own float arithmetic as written. An ADD
  reduction is the exception, as it is in the reference: it denotes the sum
  of its terms in an unspecified association, so symbolic may reorder it and
  move loop-invariant factors out of it (`reduce(x * c) -> reduce(x) * c`
  and the loop-invariant MUL terms of an ADD reduce), visible only in rounding
  and at overflow. Restricting them to integers cost a scaled contraction
  21% on the CPU (256^3, 0.75 against 0.91 ms), and attention scores up to
  50% in the review's measurement. MAX's hoist of a non-negative factor is
  exact and stays. Rune states the contract in `Rune.jit`'s documentation
  and the compilation guide.

- **A fixed-width integer constant holds its dtype's value.** The reference
  keeps integer constants exact until emission: folding runs with
  `truncate_output=False`, and a cast of a weak literal is stripped whatever
  it does to the value. So `uint8` `full 200 + 100` stayed 300 and
  `x < 300` folded to true, where the program compares with 44; the same held
  for `255 * 255`, `int8` `127 + 1`, `uint16` `65535 + 1` and `int32`
  `max + 1`. Tolk wraps every constant it builds at a fixed width
  (`Const.integer`), as it already rounds float constants to their precision,
  and keeps weak integers exact; `exec_alu` loses its `truncate_output` knob.
  A committed cast of a literal goes only where the literal keeps its value,
  and `(x // c1) // c2 -> x // (c1 * c2)` needs `c1 * c2` in range. Index
  arithmetic is weakint and unchanged. Coverage: rune `test_jit` and
  `test_jit_metal` "folded integer constants wrap"; tolk `test_weak`
  "uncasting keeps a wrapping cast", `test_symbolic` "a cast constant keeps
  its wrapped value". Remove this ruling when upstream wraps its constants.

- **A power decomposes with the transcendentals, at float32 or wider.** The
  reference rewrites every `POW` into `xpow` (`exp2 (e * log2 |x|)` with sign
  and zero fixups) inside `sym`, importing the decomposition into its
  symbolic module. Tolk rewrites it in the late decompositions beside
  `SQRT -> xpow`, where no renderer spells `POW`, so the rule lives with the
  decomposition it uses. A float16, bfloat16 or float8 power is computed at
  float32 and rounded once: at float16, `e * log2 x` loses enough bits that
  compiled powers were off by up to 100% (6282 float16 powers on the CPU,
  exponents in [-8, 8]: mean relative error 7.4e-2 computed in float16,
  4.1e-7 at float32). The reference's xpow reads parity from an `int32`
  cast of the exponent, undefined past 2^31 and for non-finite values (ARM
  saturates, Metal gives NaN, and folding a constant inf raised), and its
  sign from `base < 0`, which misses -0: `(-0) ** -1` was +inf and
  `(-2) ** 1e10` was -inf. Tolk reads parity in float (`trunc e != e`,
  `trunc (e * 0.5) * 2 != e`, exact for every float) and the sign from the
  sign bit, and adds C's `1 ** y = 1` and `(-1) ** ±inf = 1`. A cast of a
  non-finite float constant to an integer stays a cast (`Const.converts`)
  instead of raising in the fold. Coverage: rune `test_jit` and
  `test_jit_metal` "pow of a tensor base matches eager"; tolk `test_symbolic`
  "a non-finite cast to an integer stays a cast".

- **Constant exponents keep pow's special values.** The reference takes the
  reciprocal of the base for every negative constant exponent and expands
  half-integers as `x^k * sqrt x`. The reciprocal overflows for a subnormal
  base whose power does not (`1e-40 ** -0.8` gave inf, not 1e32), and
  `sqrt` turns -0 into -0 and -inf into NaN where pow gives +0 and +inf
  (`(-inf) ** 2.5`). Tolk takes the reciprocal first only for exponents of
  magnitude at least 1, where the power overflows whenever the reciprocal
  does, writes `x ** -0.5` as `1 / sqrt x`, leaves other negative exponents to
  `xpow`, and selects +0 and +inf for a half-integer power of a zero or -inf.
  Coverage: rune `test_jit` "pow of a subnormal base" and both suites' "pow
  of a tensor base matches eager".

## Validation dependencies

- **Scalar out-of-bounds validation has no SMT solver fallback.** Keep Tolk's
  dependency footprint small: interval and symbolic proofs must establish
  safety, and unproved accesses are rejected under `CHECK_OOB`. Some safe
  relationally constrained indices accepted by tinygrad may therefore be
  rejected. Coverage: the scalar/spec OOB tests. Reconsider if a concrete
  consumer requires proofs that cannot be expressed by the existing rules.

## Rendering

- **A float sum or product keeps the graph's grouping.** The reference's
  C-style renderer strips the parentheses of every same-operator operand of
  ADD and MUL, so `c + (a + b)` prints as `c+a+b`, which C evaluates as
  `(c + a) + b`: 0 instead of 1 for a = 1e8, b = -1e8, c = 1 in float32.
  Tolk strips them only from a left operand, which C groups the same way, or
  at a dtype where the operator is associative (integers and booleans). An
  unrolled reduction now adds its lanes' sum to the accumulator
  (`acc+(v0+v1+v2+v3)`) as the graph says, which also shortens the CPU loop's
  dependency chain. Coverage: rune `test_jit` and `test_jit_metal` "float sums
  and products keep their grouping". Remove this ruling when upstream keeps
  the association.

- **A narrow integer result is cast back to its dtype.** C computes on
  `char` and `short` in `int`, and the reference leaves the result
  unnarrowed: `uint8` `x - 1 < x` renders as
  `((val0+((unsigned char)(-1u)))<val0)`, which at x = 1 compares 256 with 1
  where the program compares 0 with 1. Tolk casts a scalar sum, difference,
  product, negation, left shift or quotient of an 8- or 16-bit integer dtype
  back to that dtype, the operations whose `int` result can leave its range;
  vector types do not promote and keep their spelling. Coverage: rune
  `test_jit` and `test_jit_metal` "integer comparisons read wrapped values".
  Remove this ruling when upstream narrows its results.

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

  Over a destination split across devices each device writes its own slice
  (the reference builds its custom kernels over one device's placeholders).
  Split along the scattered axis, every device reads every update and keeps
  those in its rows: the loaded index, widened to the index type, is offset by
  the device's first row, taken from the device range. Coverage: test_multi's
  "Kernels over split storage".

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
  the outer reduce loop at zero on a GPU; on the CPU, and for a single group
  anywhere, it reads matrix 0 and a select zeroes its store, since gating each
  load on the id made clang spill the narrow-input unpacking.
  Consumer: rune's lowering of `Nx_quant.apply` (RFC 0004), which takes it
  within `quant_row_bound`'s rows. Over operands split across devices each
  device multiplies its own slices; whole instances over matrices split along
  their first axis, or over split inputs, leave each device a float32 partial,
  its ids offset by its first matrix, which a sum across the devices (an
  allreduce) completes. Coverage:
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
  one input anywhere, whose loop of one trip folds away. Where the reference's
  `matmul` at bfloat16 and float16 rounds each product to the operands' dtype
  outside tensor cores, the kernel casts both operands to float32 before
  multiplying, so every device and path computes exact products; on Metal the
  tensor-core option then takes the float32 tensor cores, which on an M1 Max
  multiply 16 filled blocks of 64 rows at bfloat16 as fast as the bfloat16 ones
  (4.34 against 4.33 ms at 5760 outputs, 2.22 ms at 2880). On CUDA and AMD the
  matcher would give this form the narrow-in, float32-out tensor cores (see
  widened tensor-core operands below), but no block options are pinned there
  yet. The options are pinned per renderer and shape: on Metal the tensor cores,
  rows upcast by up to 8 tiles, columns by 3, and a local split of 4 on the
  columns, measured at 6.6 to 7.6 TFLOPS at bfloat16 on an M1 Max for 46 filled
  blocks of 64 rows at gpt-oss's shapes on the bfloat16 tensor cores (the
  float32 ones time equal), where the heuristic's `matmul` reaches 8.2 on a
  dense product; on the CPU at every dtype, rows upcast by up to 8, columns by
  up to 16 within 64 accumulators, and the loop over tiles unrolled by 4, timed
  on an M1 Max at gpt-oss's shapes 9 to 37 times faster than no options at
  float32 and 8 to 18 times at bfloat16; elsewhere none yet. Coverage:
  `test/unit/frontend/test_block_matmul.ml` (values on the default device, and
  each renderer's loop bounds) and the opt-correctness workloads `block_matmul`
  and `block_matmul_t`, under every action that leaves the block axis whole.
  Consumer: rune's lowering of `Nx_quant` products over expert ids, grouped or
  one block per position. Over operands split across devices, as for the
  quantised product: each device multiplies its own slices, and whole blocks
  over matrices split along their first axis, or over split inputs, leave a
  float32 partial per device that an allreduce sums (test_multi's "Kernels
  over split storage").

- **A narrow-in, float32-out tensor core takes widened operands**
  (`codegen/opt/postrange.ml` `tc_operand`). The reference's matcher takes a
  tensor core only when the multiply's operands have the core's input dtype,
  so a product of narrow floats widened to float32 first,
  `MUL(CAST f32 a, CAST f32 b)`, takes no narrow tensor core, and on CUDA none
  at all without `ALLOW_TF32`. That product is exact unless it leaves
  float32's normal range, which only a bfloat16 product can, and a narrow-in,
  float32-out tensor core computes exactly it, so tolk accepts a float32
  operand that is a cast of the core's narrower float input dtype and gives the
  core the narrow value. A core with a narrow output takes no widened operand:
  its products round. Cores are tried in the renderer's order, as in the
  reference, so on Metal, whose float32 core comes first, such a product keeps
  the float32 core, which on an M1 Max multiplies gpt-oss's 512-row bfloat16
  products as fast as the bfloat16 core (2.54 ms at best for 512 x 2880 x
  4096 either way). Consumers: rune's `Nx.matmul` at bfloat16, float16 and
  float8, which widens its operands to compute eager's exact products, and
  `Op.block_matmul`'s product once options take tensor cores on CUDA or AMD.
  Coverage: `test/unit/codegen/test_tc.ml` (widened operands) and rune's
  `test_jit` narrow matrix products, exact against eager on CPU and Metal.

- **`?aligned` on the Clang renderer** (`renderer/cstyle.ml`
  `clang_vector_prefix`, passed down from `Tolk_cpu.create`). The reference
  selects unaligned vector types through the `ALIGNED` environment variable
  alone. tolk also takes the choice as an argument, so a caller can select it
  for one device without touching the process environment; absent, the
  variable decides as in the reference, and the rendered source is the
  reference's either way. Consumer: rune's CPU device, which binds host memory
  it did not allocate (slices, mapped files) and passes `~aligned:false`.

- **Copies read and write contiguous windows of storage in place**
  (`schedule/prepare.ml` `storage_view`, `engine/schedule.ml`
  `copy_kernel_params`). The reference's copies take whole buffers: a copy
  from a window stages the window first (`materialize_cross_device_src`), and
  a copy into a window lands in a staging buffer that a kernel copies again.
  Tolk keeps a copy's source or destination when it is storage (a buffer, or
  a STAGE, which becomes one) or a contiguous window of storage, and a copy
  kernel storing `dst[i + a] <- src[i + b]` becomes a transfer between byte
  views. The reference forbids offset copies because SDMA cannot do them;
  tolk's copy paths take them: Realize resolves the SHRINK to a
  `Device.Buffer.view`, `Deps_tracker.uop` keys byte intervals by view
  offset, and the AMD SDMA `COPY_LINEAR` path (`tolk_amd.ml`) takes byte
  addresses with no alignment requirement. Unverified on SDMA, CUDA and NV
  queues until the node runs. Non-contiguous windows are still staged.
  A stage counts as storage, so a value is staged once; a stage of a
  symbolic value with a symbolic inner axis is not a window, and no
  cross-device store of one arises (see `storage_view`). Coverage:
  `test/unit/engine/test_collectives.ml` "copies".

- **A gather is an all-gather of pure copies** (`schedule/multi.ml`
  `allgather`). The reference lowers a copy of a split value to several
  devices as an allreduce of zero-padded shards, and to one device as a sum
  of padded shards. Tolk lowers both to one precompiled call implementing
  `Allgather axes` over (dst, src): each target gets one buffer, and each
  shard is written once into its window of it, by a transfer from another
  device or a store on its own (see the window copies above). Each device
  receives (n-1)/n of the value, where the reference's ring and naive
  allreduces move 2(n-1)/n and n-1 full buffers, and no padded shard or sum
  is materialized.
  Inner-axis windows are not contiguous and stage every foreign shard. A
  consumer does not fuse into a gather: a gather to one device followed by a
  reduction holds the gathered value, where the reference's sum of padded
  shards held the n-1 received pieces; write it as a reduce-scatter.
  Outputs are forwarded after `multi_pm` (`schedule/prepare.ml`
  `prepare_rangeify`; the reference forwards before it), and allreduces
  become calls just before that, where the reference makes them among the
  earliest rewrites, so a realized gather or allreduce writes the result's
  storage instead of a fresh allocation it then copies. Consumer: every
  copy of a split value (`Creation.clone`, `U.copy` to a device list,
  resharding in `multi_pm`). Coverage:
  `test/unit/engine/test_collectives.ml` "all-gather" and
  `test/unit/engine/test_multi.ml`.

- **A reshard of an allreduce is a reduce-scatter** (`schedule/multi.ml`
  `lower_allreduces`, `schedule/prepare.ml` `forward_call_outputs`). The
  reference allreduces the whole value and each device keeps its rows:
  2(n-1)/n of the value sent per device under ring, n-1 whole partials under
  naive, and a whole replica held. After `multi_pm`, once every consumer of an
  allreduce is known, tolk lowers one whose only consumer is a shrink keeping
  each device's own block along one axis by the device range (through the
  casts ALLREDUCE_CAST adds) to one precompiled call implementing
  `Reducescatter (op, axis)` over (dst, src): device k receives block k of
  every other device's partial, read in place as a window, and folds the
  partials in device order. That is the naive allreduce's order, so the blocks
  equal its rows bit for bit: the reduce-scatter is direct whatever RING and
  ALL2ALL say, sending (n-1)/n of the partial per device and holding (n-1)/n
  of it in received blocks (ALLREDUCE_NODE_NDEVS makes it hierarchical, see
  below). A ring variant (2/n held) is not built. An allreduce with any other consumer
  stays one allreduce, and its reshard slices the replica. With
  LATE_ALLREDUCE=0 `multi_pm` expands every allreduce before its consumers are
  seen, so no reduce-scatter is built and a reshard moves the allreduce's
  bytes. Outputs are forwarded through a split output's UNSHARD, so blocks
  assigned into a split buffer, as a gradient is, are written there directly.
  Coverage: `test/unit/engine/test_collectives.ml` "reduce-scatter" and "fully
  sharded step".

- **Gathers and reduce-scatters are hierarchical under ALLREDUCE_NODE_NDEVS**
  (`schedule/multi.ml` `allgather` and `reducescatter`,
  `schedule/allreduce.ml` `box_size`). The reference reads
  ALLREDUCE_NODE_NDEVS for its allreduce only. Tolk reads it as the box size
  of every collective of concrete shape whose targets are its sources, when it
  splits them into several boxes of several devices: device i sits in box i/h
  at rail i mod h, and only copies between devices of one rail cross a box.
  Concrete shapes only, as the hierarchical allreduce requires, so a
  reduce-scatter keeps the fold order of the allreduce it replaces. A
  reduce-scatter folds each block within each box, in device order, on the
  box's device at the block's rail, which for the block's own box is its
  destination; the block's device then folds the other boxes' stored partials
  into it in place, in box order: the hierarchical allreduce's order, so its
  blocks equal that allreduce's rows bit for bit. An all-gather copies each
  shard along its rail to every box, then within each box from the device that
  received it, reading it back from that device's destination. Both are two
  phases of one call: a collective's body is a list of phases, each reading
  and writing the state the previous one left. Each device still sends
  (reduce-scatter) or receives (all-gather) (n-1)/n of the value, of which
  (b-1)/n crosses a box for b boxes, against (n-h)/n flat. The partials a
  device holds for the other boxes add (b-1)/n of the partial to a
  reduce-scatter's peak until they are copied; an all-gather split on its
  outer axis adds nothing (an inner-axis gather already stages its windows). A
  fully sharded step under boxes therefore stays over its two-layer bound by
  (b-1)/n of a layer less its slack (2.141, 2.320 and 2.031 layers against
  2.125, 2.062 and 2.023 in boxes of 2); reusing the dead in-box blocks inside
  a collective's body would recover it, which is memory planning inside call
  bodies, not yet done. Until device identity names boxes,
  ALLREDUCE_NODE_NDEVS does. Coverage: `test/unit/engine/test_collectives.ml`
  "a gather to its own devices crosses boxes along rails", "a reduce-scatter
  over its own devices crosses boxes along rails", "each device holds its
  partial and (n-1)/n + (b-1)/n of it more", the symbolic-slice cases, and the
  equality cases under the hierarchical strategies.

- **An allreduce lays its reduced chunks back together by selection**
  (`schedule/allreduce.ml` `assemble`). The reference reassembles the ring,
  all-to-all and hierarchical results by summing the chunks zero-padded into
  place, a concatenation built as a sum: every -0 in the result came back +0
  once three or more chunks meet (two fold into one select). tolk takes each
  chunk where its padded footprint is true. Kernel launches are unchanged.
  Coverage: `test_collectives` "replicas keep a sum's -0", every strategy on
  2, 3, 4 and 6 devices.

- **A collective call takes whole storage for both arguments**
  (`schedule/allreduce.ml` `collective`). The reference's
  `create_allreduce_function` passes the output's view (a SHRINK of a RESHAPE
  of the allocation) as the call's first argument and its source made
  contiguous as the second. The realize map turns an argument that is not a
  buffer into a copy of its values, so when the output view is not a plain
  reshape (a symbolic slice of an inner axis) the call wrote the copy and the
  result read an allocation nothing wrote: such a realized allreduce returned
  zeros. Tolk passes the allocation, and for a source that is a view of
  storage (a buffer, or a STAGE) the storage, and views both inside the
  call's body: the output is written in place, the source is read without a
  staged copy of the view, and each is the raw storage a library collective
  needs. Coverage: `test/unit/engine/test_collectives.ml` "a realized
  allreduce of a symbolic slice keeps its values" and "a gather of a slice of
  a split buffer stages nothing".

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

- **Simplifying under a valid leaves loads opaque** (`uop/symbolic.ml`
  `uop_given_valid`). The reference simplifies an address under its access's
  gate right through the loads inside it. A gather's index load sits in the
  address of the gathered load, and a padded or concatenated piece gates both
  by the same clauses: given them, the index load's own gate folds away, and
  the kernel loads the index unconditionally, before the gated access that
  uses it, out of bounds wherever the gate is false. The reference at
  baa614806 and tolk both render `val0 = *(data2_50+(Lidx1+-30))` for a gather
  between two pieces of a concatenation. tolk replaces each load by a variable
  of its bounds, in the valid and the expression alike, and restores it
  afterwards: a clause bounding a loaded value still applies, and the kernel
  reads the index once, gated. Every golden and parity output is unchanged.
  Coverage: `test_symbolic` "uop_given_valid", and `test_run` "a padded
  gather loads its indices under the pad's gate" and "a gather between other
  pieces loads its indices under their gate", which check every index load
  of the lowered kernel.

- **A select becomes a gated load's alternative only when its value survives
  the load's dtype** (`codegen/late/gater.ml` `alt_in_load_dtype`). The
  reference folds `where(gate, cast(load), a)` into the load and casts `a` to
  the load's dtype. A load narrower than the select rounds `a`: with a
  bfloat16 load widened to float64, an index 257 on the other branch came
  back 256. tolk folds when the alternative is Invalid, has the load's dtype,
  is a cast from the load's dtype, or is a constant that comes back bit for
  bit through the load's dtype; otherwise the select stays. Every golden and
  parity output is unchanged. Coverage: `test_lower` "gater folds a select
  into a load only when its value survives".

- **Concatenation selects its pieces** (`frontend/op.ml` `cat`). The
  reference joins pieces of unequal extent by summing them zero-padded
  (`bitwise_or` for booleans). A sum returns its operand only where
  `x + 0 = x`, which IEEE arithmetic breaks: -0 + 0 is +0, an add quiets a
  signalling NaN, and Metal flushes a subnormal operand of a float add to
  zero. Rune's sort tests join 12 pieces of unequal extent, 804 float32
  elements, and the compiled join returned none of their 24 -0 as -0. tolk
  pads each piece into place and takes it with a `where` on its padded
  footprint, a padded `true`, so every bit pattern survives and booleans need
  no special case. Equal extents already stack. Kernels are unchanged in
  number. A loaded piece's select becomes the alternative of its gated load,
  so loaded pieces cost fewer operations than the sum; a join of two
  computed pieces costs 6 selects per 4 elements where the reference folds
  its sum into 3. Every golden and parity output is unchanged, since their
  cats have equal extents. Consumers: every unequal-extent `cat`, among them
  rune's `E_cat` (`Nx.concatenate`).
  Coverage: `test_run` "unequal extents keep every bit", rune's `test_jit`
  and `test_jit_metal` "concatenation keeps every bit", on CPU and Metal.

- **Several tensor indices read through one linear gather**
  (`frontend/op.ml` `getitem`). The reference selects `x[i, :, j]` by
  summing `x` under the product of one mask per indexed axis. That product
  does not reduce to a load. `x[2, :, 0]` of a `[256; 2; 256]` float32 whose
  selected elements are -0, holding a NaN at `[3; 0; 0]`, came back
  `[NaN; +0]` on CPU and Metal: the sum turned -0 into +0, and, split into
  chunks, it multiplied the unread NaN by zero. tolk flattens the indexed
  axes, which the reference's order puts first when they are not
  consecutive, and gathers once at the linear index. `x[7, :, 200]` of a
  `[256; 4; 256]` float32 compiles to one kernel of gated loads, where the
  mask product took two with a 1024-element partial sum between them. Every
  golden and parity output is unchanged. Coverage: `test_run`
  "non-consecutive tensor indices select their elements", the shape tests in
  `test_frontend`.
