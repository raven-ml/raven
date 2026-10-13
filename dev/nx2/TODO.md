# Unfinished nx2 work

Implementation stays paused until rig moves to `packages/rig`. Remove each
item when its gate is met. Recovery refs below preserve source for adaptation
to the current API; they do not establish readiness to land.

## Correctness and contracts

- Complete the active `kernel-plain-args` migration in pool 10: replace every
  Spec consumer with the settled shape and argument contracts. Validate all
  backends and retain shape laws and warm allocation checks.
- Reject a map program's `Coord` past its input rank before execution. The
  partial patch is `refs/audit/20261010-deep/worktrees/wt-16`; complete
  admission, documentation and the regression together.
- Bound sort-key packing for axes longer than `2^31`, and compute Bluestein's
  square modulo `2*n` without overflowing int64. Recover the three-file patch
  from `refs/audit/20261010-deep/worktrees/wt-14`; reproduce both failures and
  validate the corrections on current code.
- Direct backend entries must refuse malformed contraction extents and
  padded shapes beyond `max_numel` before writing or allocating scratch.
  Port the regressions from `refs/audit/20261010-deep/worktrees/wt-26` to the
  settled kernel interface; keep one rule for each shape invariant.
- Normalize noncanonical Bool bytes in AMD and CUDA contractions. AMD's
  residual is in `refs/audit/20261010-deep/branches/round3`; CUDA's decoder,
  no-pack shortcut and skinny integer path are in
  `refs/audit/20261010-deep/worktrees/wt-10`. Rebuild kernels and run the
  noncanonical-byte laws on their GPUs.
- Preserve `exp2` NaNs in GPU code, and fix the GPU reference's infinite
  scaled sums and NaN error ratios. Recover the targeted regressions from
  `refs/audit/20261010-deep/worktrees/wt-10` and validate special values.
- Register Metal in generic backend laws while keeping CPU/CUDA-specific
  reduction promises with those libraries. Adapt
  `refs/audit/20261010-deep/branches/metal-laws` and run all shared laws.
- Resolve the outstanding external engine review findings: donation through
  `Eval.received`, the shared `Interp.running` state across fibers, and
  `Nx_array.to_device` copying a strided view's enclosing span. Recover the
  probes from `refs/audit/20261010-deep/worktrees/wt-18`; turn confirmed
  contract failures into public regressions before fixing them.
- Validate order/counting operations on current next, including their
  contracts, public laws and benchmark baselines.

## Performance and incomplete implementations

- Evaluate MR-aligned GEMM row subdivision from
  `refs/audit/20261010-deep/branches/few-units`. Require public matmul A/B
  measurements and final baselines on the Mac and kimchi before landing.
- Reassess the engine dispatch costs after the kernel migration. The staged
  patches and missing `dispatch/real-1` and `dispatch/sum-1` rows are preserved
  in `refs/audit/20261010-deep/worktrees/wt-13`. Require per-stage A/Bs,
  kimchi correctness and baselines; keep compiler internals private.
- Decide the CUDA elementwise and reduce drafts in
  `refs/audit/20261010-deep/worktrees/wt-10` against the intended backend
  contract. Any accepted implementation needs current argument/access
  consumers, regenerated cubins, kernel and public laws, and recorded rows.
- Resolve the CUDA lane-order experiment's decode regression before adopting
  it. `refs/audit/20261010-deep/worktrees/wt-11` preserves the working source
  and its unmerged index is in the audit archive. Reconcile current APIs,
  validate exact order and tails, and run interleaved decode A/Bs. Its new
  float8 pack-output path needs explicit coverage if adopted.
- Reduce across split axes with partial results and a reduction-specific
  combine. An `[8192;4096]` sum over eight devices should transfer partials,
  instead of gathering the whole input (112 MiB versus 112 KiB per device).
- Implement `nx.io` on nx2, then Rune over nx2 with tolk lowering, before the
  GPT-2 and gpt-oss vLLM comparisons.
