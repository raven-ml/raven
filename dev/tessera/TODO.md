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
- Floats narrower than float32 compute in float32 on the CPU, and nothing at
  the call says so: bf16 rows on the CPU time f32 kernels plus conversions.
- `Nx.split` returns a list, so taking a GRU's three gates needs a
  non-exhaustive match or slices.

- `nx.mli` (the `Ptree.Walk.under` example) calls a layer whose positions
  are fixed arrays "frozen", but vega steps fixed arrays and rune
  differentiates them. Either the wording is wrong or fixed arrays should
  mean what it says: a design pass decides which is the simpler rule.

## vega

- `doc/03-schedules.md` teaches a scheduled weight decay applied after the
  step, which does nothing to bfloat16 parameters (0 of 1000 weights moved
  over 1000 steps where the exact factor is 0.990): the step's single
  stochastic rounding has already happened. It exists because
  `~weight_decay` is a float while `~lr` is a schedule. Being decided with
  the per-leaf decay design.
- `vega.mli` says AdamW at `weight_decay = 0.` is exactly Adam; for an
  infinite parameter it gives NaN (`0 × ∞`). Being decided with the per-leaf
  decay design (omitting the term makes it true by construction).

- LARS's trust ratio is not eq. 6 of You, Gitman & Ginsburg, "Large Batch
  Training of Convolutional Networks" (2017), found by the per-leaf decay
  skeptic: a pass of its own decides whether vega's form is a stated
  deviation or a defect.

## rune

- Running one body eagerly on the host and compiled on a GPU means writing
  `if jit then Rune.jit spec f else f` with an explicit signature at every
  call, even for `Nx.matmul`, and threading data the body uses (an
  attention bias) through the signature.
- A scan over a strided stack of weights (layers stacked on one axis,
  scanned one layer a step) copies the whole stack, forward pass included:
  +524,288 B in the remat audit's 8-layer probe, so a layer-scanned model
  holds its weights twice. Found by the remat design audit; to fix with
  that design or on its own, after a check of where the copy is made.
- Under `Rune.jit`, a remat inside a scan step whose body contains a `grad`
  saves nothing (identical bytes with and without it), and a loop's storage
  is never reused within a trip (4 chunks hold 1.24 MB against 0.59 MB for
  the whole batch). Being decided by the remat design campaign.
- Each new pixel count or sequence length compiles again: 191 small cases
  took 100 s cold on the host (Clang, one process per kernel), and the
  port's length buckets will meet the same cost.

## tolk

- TF32 is reached from user code as a compiler setting
  (`Tolk.Setting.context [B (allow_tf32, true)]`), NVIDIA only, and silently
  ignored elsewhere: a `tf32` row on AMD or the CPU measures f32. Wanted: a
  precision stated on the contraction, refused where no unit has it.

## kaun

- No GRU: v1.0's and v1.1's pooling is written by hand with `Rune.scan'`
  (about 25 lines with its walk). Wanted: a GRU cell and its scan, with
  PyTorch's gate layout so checkpoints map directly.
- `Layer_norm.apply` takes `?eps`, so it must be eta-expanded to pass to
  `Rune.jit`.

## Repository

- `bench/` cannot meet the 2-minute rule for a package's suite: its shapes
  are the point. Undecided: an exemption for `dev/`, a short default run
  with the full run behind an explicit command, or an alias of its own.
- A dune rule cannot pass the device choice through the passive build
  server, so a run on a chosen device runs the executable by hand.
