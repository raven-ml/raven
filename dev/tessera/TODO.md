# tessera: what raven lacks

Friction met while building tessera, and defects its design reviews found,
by the library that owns the fix. Each is fixed by a change that simplifies
where it can, with a design pass when it touches an interface, and a review.
An item leaves when its library fixes it, or when a design decides it stays.

## thumper

- A case cannot state its work, so reports show times only: `bench/`
  re-derives TFLOP/s and GB/s over `Thumper.measure`, and the rates never
  reach a baseline or `check`. Wanted: work (FLOPs, bytes) on a case,
  reported as rates.
- The `accept:`/`rerun:` hints drop a suite's own options, so a device
  flag would bless or rerun on the default device; `bench/` takes its device
  from `TESSERA_DEVICE` instead. Wanted: suite options the hints repeat.

## rig

- rig cannot report a device's total or free memory, so a training step
  cannot check `Rune.footprint` against what the GPU holds; dev/tessera
  takes `--memory` meanwhile. Found by the M4 training design.

- Every suite and example maps "cuda", "nv", "amd", "metal" to
  `Rig.open_ (Rig_x.gpu 0)` and forks a child to ask whether a GPU opens
  (its driver must not start before thumper forks): kaun's GPT-2, rune's
  bench, gpt-oss's bench, `bench/` and `bin/main.exe`. Wanted: opening a
  device by name, and asking whether it opens, in rig.

- rig's NV launch sets a kernel's shared-memory carveout to at least 32 KiB
  (`rig/lib/nv/abi/launch.ml:24` offers only 32, 64 and 100 KiB), so a
  kernel using no shared memory still gives up 32 KiB of L1, where CUDA picks
  the smallest carveout: NV's well-factored f32 GEMMs run at ~10 TFLOP/s
  against CUDA's ~17 (M0). An 8/16 KiB configuration is unmeasured. Found by
  the M2 GEMM review.

## nx

- A random draw on a GPU set needs a compiled program (one per shape and
  dtype), or a draw on the host placed over, which doubles host memory
  (8 GB of q, k, v at large's T = 256). Wanted: draws at a kernel-less
  placement, as creations already work.
- nx.cpu's bound on how long a kernel holds a domain's lock (`nx_cpu.mli`)
  is broken by memory-mapped operands: a small gather holds it through page
  faults of ~170 µs each. Found by the streaming-data skeptic; decided with
  part 2 of that design (where a batch's read depth lives), which needs the
  same fact that an operand is file pages.
- Floats narrower than float32 compute in float32 on the CPU, and nothing at
  the call says so: bf16 rows on the CPU time f32 kernels plus conversions.
- `Nx.standardize`'s error bound (`nx.mli`) names `F`, `L` and `D` as
  "nx.cpu's for its moments", but `nx_cpu.mli` (Moments) defines `L` and
  `D` alone; tessera's batch-normalisation law reads `F` as the variance's
  bound there, `(L + D + 5) u V` to first order. Found by the M4b review's
  tolerance derivation.
- `Nx.split` returns a list, so taking a GRU's three gates needs a
  non-exhaustive match or slices.

- `Nx_io.save_tiff` compresses by Deflate only (lz decodes LZW and does not
  encode it), so `Embedding.save_geotiff` writes Deflate with the floating
  point predictor where GeoTessera's GeoTIFFs default to LZW, a choice its
  `--compress` leaves to the user. GDAL reads both; a 1116 × 228 tile of 128
  bands takes 56.5 MB in Deflate against 161 MB in GeoTessera's LZW. An LZW
  writer is wanted only if a reader needs it.
- `Nx.round` rounds halves away from zero; GeoTessera's int8 embedding codes
  round ties to even (numpy's `np.round`), and 23 of the oracle fixtures'
  15.7M codes differ. A rounding mode is a vocabulary question for nx
  (found by the tessera-stack audit). Until nx decides, `Embedding.quantize`
  rounds a half as twice the rounding of its half, exact in binary.
- Reading a host value's elements in a host loop, as packing a batch does,
  takes `Nx.Repr.read`, then `Nx_array.read` with a contiguity check, and
  handing a filled bigarray back takes `Nx_array.of_bigarray` and
  `Nx.Repr.unsafe_of_array`, since `Nx.to_bigarray` and `Nx.of_bigarray`
  copy. These are public interfaces, and `Nx.of_bigarray`'s doc gives the
  composition. tessera holds it in a private `Flat`; the item records the
  friction.

## vega

- LARS's trust ratio is not eq. 6 of You, Gitman & Ginsburg, "Large Batch
  Training of Convolutional Networks" (2017), found by the per-leaf decay
  skeptic: a pass of its own decides whether vega's form is a stated
  deviation or a defect.

- nx's ragged suite has a flaky cover ("a text of 9 bytes"): windtrap draws
  a random root seed each run and some seeds never reach the case. A cover
  some seeds miss is a flaky test (AGENTS.md); found by the streaming-data
  review, independent of that change.

## rune

- Running one body eagerly on the host and compiled on a GPU means writing
  `if jit then Rune.jit spec f else f` with an explicit signature at every
  call, even for `Nx.matmul`, and threading data the body uses (an
  attention bias) through the signature. tessera writes it three times for
  `Model.apply` (the executable, its validation of blocks, the bench), and
  `Embedding.encode` takes the forward function for that reason.
- Under `Rune.jit`, a remat inside a scan step whose body contains a `grad`
  saves nothing (identical bytes with and without it), and a loop's storage
  is never reused within a trip (4 chunks hold 1.24 MB against 0.59 MB for
  the whole batch). Being decided by the remat design campaign.
- `Nx.Rng.permute` compiles in about 37 ms per round under `Rune.jit` (about
  2 s at 60 rounds, 9 s at 240), once per n: about 7 s once per TESSERA run.
  Acceptable for now; the cause is in tolk (found by the streaming-data
  review). It compiles once per n, not once per round count: `n` is a 0-d
  operand, but under a trace the compiler folds it into the kernel as a
  constant and specialises on it. One compile per round count needs either
  a traced integer argument or tolk not specialising on folded constants
  (rune/tolk).
- No f32 attention derivative kernel, and none above head dim 128: those
  cases fall back to a dense backward (packed rows 2.3× per-pixel buckets in
  f32 training; TESSERA's large student, head dim 160, and the teacher,
  1024, in bf16). Found by the stack's design skeptic.
- A query tile's key loop visits every key tile and votes on each; limiting
  it to the tile's visible key range would make flat segments (no rows) a
  simpler layout than packed rows (today 1.9-98× slower).
- Attention-probability dropout composes from existing ops, which is the
  dense path; a fused form is a speed item, not a missing capability.
- tolk's loop-body launch rule (main 1ab249a3d3, which fixed remat inside a
  scan of two or more trips: 87.0 → 41.9 MB) allows no byte above the
  trip's peak and counts a one-element buffer as a 256-byte block, so loop
  bodies of small storage lose launches (1000 chained scalar calls stay at
  1000 dispatches where the old rule ran 500; jera's test_ode 619 → 686,
  test_root 41 → 48). Measure jera's loops on a GPU, then decide whether
  small-storage bodies need the top level's one-element exception (tolk).
- A stacked parameter (`Nx.get` from `[layers; …]`) still costs a second
  gradient in a scan's carry: its gradient is assembled into its own buffer
  before the add (1,171,476 B against 780,052 B for separate arrays at k = 4
  under remat). TESSERA uses separate per-layer arrays (rune).
- Each new pixel count or sequence length compiles again: 191 small cases
  took 100 s cold on the host (Clang, one process per kernel), and the
  port's length buckets will meet the same cost.

## tolk

- Compiling a distillation step's gradient for CUDA raises
  `Invalid_argument "a AxisType.UPCAST axis comes from a AxisType.GLOBAL,
  AxisType.LOCAL, AxisType.WEAK axis, not a AxisType.UPCAST one"` in
  `Heuristic.block` (`codegen/opt/heuristic.ml:970`, from
  `Postrange.apply_opts`; the 8-by-8 blocking of d35db2e8d0), wherever the
  step takes the dense attention path (dropout above 0): in float32 at
  `gen/grads.py init`'s student (d 64), in bfloat16 at nano's 4096 pixels
  and medium's 2048 a group. The same student in bfloat16 compiles, and
  so do nano's and medium's steps on the fused path (dropout 0 at the
  layers). Reproduce:
  `main.exe validate --training DIR --tile DIR --device cuda`. It blocks
  training with v2's dropout on CUDA. Found by the M4 distillation track.
- A distillation step end to end (medium, 4096 pixels in 4 groups, bf16,
  fused attention) takes 8.04 s on fudge's RTX 5000 against PyTorch's
  1.744 s (m0-baselines.md §2.3, which drops out at 0.1); unprofiled. The
  bf16 training products (3.6-17× PyTorch's) and the missing attention
  derivative kernels (rune) are the suspects. Found by the M4 distillation
  track.
- Compiled on the host (`cpu-jit`), that float32 gradient lies further from
  float64 than PyTorch's float32 does: the queries' and keys' weights at a
  relative 1.9-2.3e-6 against PyTorch's 5.0-6.0e-7 (up to 2.2 times the
  validation's bound of twice PyTorch's), the heads' at 4.6-6.0e-7 against
  2.3-2.4e-7; eager nx.cpu meets the bound (0.95 of it at worst). The cause
  (the compiled exp, the sums' order) is unexamined. A pretraining group's
  gradient (Barlow Twins with mixup on a small teacher, `main.exe validate
  --barlow DIR --device cpu-jit`) shows the same: 17 of 150 tensors past the
  bound, at most 1.38 times it (queries', keys' and QK norms' parameters,
  layer norms, the pooling query), where eager meets it at 0.78 of it.

- On fudge's RTX 5000, float32 products whose output is 440 wide, the
  medium student's width, run at 4.6-6.6 TFLOP/s (bench op rows
  `medium/{embed2,out_proj,ffn2}`), while medium's `ffn1`, 440 wide on its
  input, runs at 17.3-17.6, and large's products of width 640 at 17-18.
  End to end medium embeds 2,993 px/s against large's 3,172, though large
  computes twice its FLOPs (bench `e2e/*/encode`; PyTorch's reference
  5,039 and 3,472). The cause is tolk's to find, beside f32 GEMM parity
  with cuBLAS.
- LayerNorm is ~2× PyTorch's on CUDA and 2-3.5× on CPU (M0); see the next
  item for the compiled reads.
- tolk's `Heuristic.block` trusts the upcast indices `upcast_more` recorded;
  when an upcast consumes a whole axis (TESSERA's dense attention scores at
  [64; 4; 256; 256]: the 4 heads), the index shifts onto an UPCAST range and
  compiling raises "UPCAST axis comes from GLOBAL/LOCAL/WEAK", or silently
  blocks the wrong axis. It blocks training with dropout and f32 gradients
  on CUDA. Diagnosed; being fixed within the M2 GEMM track, which edits the
  same heuristic.
- tolk's `Heuristic.block` ignores upcasts made before it by
  `upcast_masked`: a materialised `[b; h; i; j]` mask gave 256 accumulators a
  thread. Values were correct; the cost is registers and occupancy (found by
  the second M2 GEMM review, pre-existing on main).
- tolk compiles reductions that read the same operand over the same axes
  as separate kernels: compiled `Nx.standardize` reads its input five times
  (Max for the overflow rescue, Σx, Σd, Σd², then the map) where nx.cpu reads
  it three times, and `rms_norm` three. Fusing such reductions would cut
  standardize to three reads and let a fixed-scale accumulation (Blue's, as
  LAPACK's dnrm2.f90 after Anderson, ACM TOMS 2017) drop the Max pass from
  standardize, rms_norm, var and std together. GPU wall time unmeasured
  (found by the layer-norm review).
- CUDA f32 GEMM shape coverage (M0, m0-baselines.md): products whose N is
  divisible by 3 (every fused QKV projection: N = 432, 768, 1320, 1920) and
  those at TESSERA's d = 144 and 440 run at ~5 TFLOP/s, against 16-20 for
  well-factored N, 3.8-4.4× cuBLAS; well-factored N are 1.3-1.45×. Together
  they are most of the students' 1.8-2.8× forward gap on CUDA. NV's
  well-factored GEMMs run at ~10 TFLOP/s against CUDA's ~17. bf16 training
  products at medium's shapes are 3.6-17× PyTorch's.
- Compiling the teacher's attention (head dim 1024) takes about 10 minutes of
  NVRTC on CUDA and NV, and runs 7-11× PyTorch's; head dims 36, 64 and 160
  are 1.6-3× PyTorch's memory-efficient kernel (M0).
- TF32 is reached from user code as a compiler setting
  (`Tolk.Setting.context [B (allow_tf32, true)]`), NVIDIA only, and silently
  ignored elsewhere: a `tf32` row on AMD or the CPU measures f32. Wanted: a
  precision stated on the contraction, refused where no unit has it.

## kaun

- `Kaun.Linear.apply` rounds twice at bfloat16: the product, then the bias
  added to it, where a product rounding its sum plus bias once is what
  PyTorch's autocast linear computes (`Nx.contract ~init` can). The fix
  changes float32 bits slightly too. tessera rounds once in its private
  `Product` meanwhile, its one copy. Found by the M4 training design.

- No GRU in kaun. TESSERA's v1.0 and v1.1 checkpoints hold tessera-util's
  `CustomGRUCell` (Chung et al.'s form, reset gate applied before the
  candidate's product), not PyTorch's `nn.GRU` layout (no `weight_ih_l0`
  keys), so a kaun GRU with PyTorch's layout would not load them. v1.0 is an
  oracle only in the stack's design, so this waits for v1.1 coming into
  scope or another consumer.
- `Layer_norm.apply` takes `?eps`, so it must be eta-expanded to pass to
  `Rune.jit`.

## sowilo

- Its docs show a `Kaun.Data.prepare ~shuffle` that does not exist (found by
  the streaming-data review).

## Repository

- OCaml's arm64 backend contracts `a +. b *. c` (and the subtract forms)
  into one fused multiply-add (`asmcomp/arm64/selection.ml:183-198` in OCaml
  5.5.2); x86-64 does not. So OCaml float code can give different bits on
  the M1 Max than on fudge, below any promise an `.mli` states in bits
  across machines. jord states where it wants fusion with `Float.fma` and
  keeps the rest unfused; raven's other OCaml float code (nx's host-side
  arithmetic, rune's, jera's, ymir's) has not been checked. Found by the
  jord review.

- `bench/` cannot meet the 2-minute rule for a package's suite: its shapes
  are the point. Undecided: an exemption for `dev/`, a short default run
  with the full run behind an explicit command, or an alias of its own.
- nx's CUDA suites hold only dune's workspace lock, not the machine's GPU
  lock (`AGENTS.local.md`'s `~/wt/nx-cuda-GPU.lock`), so `dune build
  @packages/nx/runtest` can use the GPU while another run holds it. A
  machine rule that a suite cannot follow is a gap between the two: either
  the GPU suites take the lock themselves, or the rule says how a runtest
  that reaches the GPU is wrapped (found by the remat track).
- A dune rule cannot pass the device choice through the passive build
  server, so a run on a chosen device runs the executable by hand.
