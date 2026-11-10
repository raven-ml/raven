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

- Every suite and example maps "cuda", "nv", "amd", "metal" to
  `Rig.open_ (Rig_x.gpu 0)` and forks a child to ask whether a GPU opens
  (its driver must not start before thumper forks): kaun's GPT-2, rune's
  bench, gpt-oss's bench and `bench/`. Wanted: opening a device by name,
  and asking whether it opens, in rig.

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
- `Nx.split` returns a list, so taking a GRU's three gates needs a
  non-exhaustive match or slices.

- `Nx.round` rounds halves away from zero; GeoTessera's int8 embedding codes
  round ties to even (numpy's `np.round`), and 23 of the oracle fixtures'
  15.7M codes differ. A rounding mode is a vocabulary question for nx
  (found by the tessera-stack audit).

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
  attention bias) through the signature.
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
- Each new pixel count or sequence length compiles again: 191 small cases
  took 100 s cold on the host (Clang, one process per kernel), and the
  port's length buckets will meet the same cost.

## tolk

- tolk compiles reductions that read the same operand over the same axes
  as separate kernels: compiled `Nx.standardize` reads its input five times
  (Max for the overflow rescue, Σx, Σd, Σd², then the map) where nx.cpu reads
  it three times, and `rms_norm` three. Fusing such reductions would cut
  standardize to three reads and let a fixed-scale accumulation (Blue's, as
  LAPACK's dnrm2.f90 after Anderson, ACM TOMS 2017) drop the Max pass from
  standardize, rms_norm, var and std together. GPU wall time unmeasured
  (found by the layer-norm review).
- TF32 is reached from user code as a compiler setting
  (`Tolk.Setting.context [B (allow_tf32, true)]`), NVIDIA only, and silently
  ignored elsewhere: a `tf32` row on AMD or the CPU measures f32. Wanted: a
  precision stated on the contraction, refused where no unit has it.

## kaun

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
