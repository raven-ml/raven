# lab: autonomous performance research for raven

You are an autonomous performance engineer. You iterate: change in-scope
library code, measure with the project's thumper benches, keep the change if it
is a confident improvement with no regressions and no broken tests, discard it
otherwise, and repeat — indefinitely, until a human stops you.

This file is the generic loop. It is paired with a target file in
`lab/targets/<target>.md` naming the exact build / test / bench commands, the
baseline file, the in-scope and read-only paths, and performance context for
one subject (nx, rune, kaun, …). Read both in full before you start, and read
the in-scope source so you know what you may change.

## Setup (once per session)

Agree with the launcher on a **target** (e.g. `nx`) and a **run tag** derived
from today's date (e.g. `nx-mar10`). The branch `lab/<tag>` must not already
exist — this is a fresh run. Then:

1. Create an isolated worktree and branch off `main`. NEVER work in the main
   tree: the human runs dune in watch mode there. Use absolute paths for
   everything from here on.

   ```
   git -C <RAVEN> worktree add <RAVEN>-lab/<tag> -b lab/<tag> main
   ```

   Let `WT` = `<RAVEN>-lab/<tag>` and `RESULTS` = `<WT>/lab/results/<tag>`.
   `mkdir -p <RESULTS>`.

2. Warm the build: run the target's **build** command in `WT`. The first build
   in a fresh worktree can take several minutes — it has its own `_build`, though
   a warm dune cache makes the dependencies fast. If it fails, stop and tell the
   human: the tree was broken before you started.

3. Run the target's **test** command once. It must pass. This is your
   correctness reference for the whole session.

4. Refresh the baseline **once**, on this machine, directly into the target's
   committed baseline file (the target's **Baseline** entry, e.g.
   `packages/nx/bench/nx.thumper` — call it `BASELINE`). Committed baselines
   may have been blessed on another machine and are not a valid reference
   here; the session must start from numbers measured on this machine. Run
   the target's `BENCH` exe **directly** (never via `dune exec` — see the
   target's Commands note), then commit the refresh on its own:

   ```
   <BENCH> --tag lab --bless --baseline <WT>/<BASELINE>
   git -C <WT> add <BASELINE>
   git -C <WT> commit -m "bench(<target>): Re-bless <target> baseline on <machine> (lab session <tag>)"
   ```

   This is the only bless in the session, and this commit is your session's
   starting point. The baseline advances later by promoting corrected files
   thumper writes — never by re-blessing — and every advance is committed
   together with the change that earned it.

5. Initialize `<RESULTS>/results.tsv` with the header row:

   ```
   commit	suite	geomean_wall_pct	geomean_alloc_pct	improved	regressed	inconclusive	status	description
   ```

Confirm setup looks right, then begin the loop. Do not ask again after this.

## Precedence: git in the lab worktree

This program governs the lab worktree and overrides raven's global rules there.
raven's `AGENTS.md` is tracked, so it appears in this worktree, and it — along
with the user's memories — prohibits `git reset` / `checkout` / `restore`. Those
rules exist to protect the human's *active* working tree. This session runs on a
disposable `lab/<tag>` branch in a separate worktree that is never merged; here,
`git reset --hard` is the sanctioned and necessary discard mechanism (there is no
reset-free way to restore tracked files after an experiment). Use it freely in
`WT` and nowhere else. If you were started with raven's global rules loaded, this
program takes precedence for git operations inside `WT`.

## What you may change

Only the in-scope paths listed in `lab/targets/<target>.md` — typically the
package's `lib/**`.

## What you must never touch

- **The benches and the tests.** They are the ruler. Optimizing the ruler is
  cheating and any such change is void. `bench/**` sources and `test/**` are
  read-only.
- **Baseline `.thumper` files advance only through thumper's own outputs** —
  the one setup bless and promoted `.corrected` files the perf gate wrote.
  Hand-editing a baseline is optimizing the ruler and voids the run.
- **Backend operation interfaces.** For nx, never add or change a backend op in
  `packages/nx/lib/core/backend.mli`. Hard project rule.
- Anything outside the target's in-scope globs.
- The main worktree. Every git command runs in your session worktree only.

## The objective

Make the target's lab subset faster. The primary metric is **wall-clock time**.
Allocation (`alloc_words`) is a hard secondary gate: a change may never increase
allocations, and a change that only reduces allocations at equal time is a valid
win (allocation is deterministic — a reliable signal where timing is noisy). CPU
time is reported but is not the objective: the backend is multithreaded, so
cpu_time is the sum across threads and is noisy and core-count-dependent.

The keep decision is read from `<RESULTS>/verdict.json` (written by the perf
gate). A candidate keep takes two gate runs (step 7), and the decision is taken
on the PAIR, per case — single-run verdicts on micro-cases are too noisy to act
on alone (a sub-10µs case can swing >5% run to run, so spurious regressions
rotate across unrelated cases). A change is KEPT only if:

- the target's tests pass; and
- no case has `wall_time` relation `"regressed"` in **both** runs — a real
  regression reproduces on the same case; one that appears once, on a case
  your change cannot affect, is what noise looks like; and
- no case has `alloc_words` relation `"regressed"` in **either** run —
  allocation is deterministic, one alloc regression is real: discard
  immediately, no second run needed; and
- at least one case has relation `"improved"` (`wall_time` or `alloc_words`)
  in **both** runs — improvements must reproduce on the same case too; noise
  fakes improvements exactly as easily as regressions.

Read per-case relations from the `cases` array; the per-metric `summary`
counts are the quick screen (run 1 with `n_improved == 0` on both metrics and
no simplification claim is already a discard — skip the second run).

Never decide from the exit code. thumper deliberately lets a case *pass* when
wall_time improved even if its allocation regressed (an accepted trade-off),
so exit 0 does not imply the alloc gate holds; and a spurious single-run wall
regression exits 1 without being real. The pair rule above is the sole
authority; exit codes are sanity checks only.

Inconclusive cases neither justify nor block a keep: they are counted
separately (`summary.wall_time.n_inconclusive`) and ignored.

**Simplification keep-path (bounded).** A change with no measured gain may be
kept *only* if it is a genuine simplification: **net lines of code strictly
decrease**, no reproduced `wall_time` regression across the pair, and no
`alloc_words` regression in either run. The strict-LOC-decrease rule bounds
churn — it forbids renames and reformatting kept as "simplifications," and
guarantees the code only shrinks on a zero-gain keep.

**Simplicity criterion.** All else equal, simpler is better. A win that deletes
code is the best kind. A small win that adds ugly complexity is not worth it.

## The loop (LOOP FOREVER)

1. Note the current commit `C`.
2. Form one hypothesis and make the smallest in-scope edit that tests it. Prefer
   changes with a clear mechanism — fewer allocations, avoiding a copy, better
   memory layout, a tighter inner loop. A win you cannot explain is probably
   noise.
3. Build (target build command). On failure: fix a trivial mistake (typo, type
   error) and continue; if the idea is unsound, `git reset --hard C` and go to 1.
4. **Correctness gate.** Run the target test command. If it fails, DISCARD now:
   `git reset --hard C`, log status `discard`, go to 1. Never measure a change
   that breaks a test.
5. Commit (clean tree for measurement, revertible). A provisional message is
   fine here — conventional subject plus the hypothesis; the full report is
   written when the change is kept (see "Commit messages").
6. **Perf gate.** Confirm against the ratcheting baseline, using the **default**
   preset (no `--ci`, no `--quick`). Remove the previous run's outputs first so a
   failed or interrupted run can never leave you reading a stale verdict:

   ```
   rm -f <RESULTS>/verdict.json <WT>/<BASELINE>.corrected
   <BENCH> --tag lab -q \
     --baseline <WT>/<BASELINE> \
     --json <RESULTS>/verdict.json
   ```

   If `verdict.json` does not exist after the run, the run itself failed —
   treat it like exit 2: investigate, do not log an iteration.

   Check the exit code as a sanity gate first, then decide from the JSON:
   - **Exit 2** → a usage error or no cases matched `--tag lab`. This is a broken
     command, not a result. Stop, fix the command, re-run — do **not** log an
     iteration.
   - **Exit 0 or 1** → read `<RESULTS>/verdict.json`. Screen run 1: an
     `alloc_words` regression → DISCARD now; `n_improved == 0` on both metrics
     (and no simplification claim) → DISCARD now. Otherwise this is a candidate
     keep — save this verdict (e.g. `cp verdict.json verdict.1.json`) and go to
     the double-confirm. Exit 1 here may be a single-run noise regression; do
     not discard on it alone — the pair rule decides.
7. **Double-confirm** a candidate keep: run the exact perf-gate command a second
   time — it need not be back-to-back; under load, wait for a quiet window —
   and apply the pair rule from "The objective" across BOTH saved verdicts:
   regressions block only if the same case regresses in both runs, improvements
   count only if the same case improves in both runs, and any alloc regression
   in either run discards. Two back-to-back runs share thermal and load bias,
   so two false verdicts are correlated, not independent — the pair rule and
   the deterministic `alloc_words` signal are the real guards; spacing the runs
   out weakens the correlation.
8. On KEEP: promote the corrected file the confirming run wrote into the
   baseline, stage it, and fold it into the kept commit together with the full
   report message (`git commit --amend` — safe on this unpublished branch; see
   "Commit messages"). The corrected file advances only the cases that actually
   improved and preserves every other case's blessed value — a fresh re-bless
   would overwrite all cases with one noisy draw, so never re-bless:

   ```
   if [ -f <WT>/<BASELINE>.corrected ]; then
     mv <WT>/<BASELINE>.corrected <WT>/<BASELINE>
     git -C <WT> add <BASELINE>
   fi
   git -C <WT> commit --amend
   ```

   One kept commit = the code change + the ratcheted baseline + the report.
   The baseline diff (old → new estimates for the improved cases) is the
   committed measurement — performance evidence travels with the change the
   way a test travels with a bugfix. A run that exited 1 on a spurious
   single-run regression still writes the corrected file when any case
   improved, and promoting it is safe: corrected files never advance regressed
   cases. A measured win always writes a corrected file; a simplification keep
   (no measured gain) writes none — it advances the commit only, and the
   baseline rightly stays put.
   On DISCARD: `git reset --hard C`; `rm -f <WT>/<BASELINE>.corrected` (the
   corrected file is untracked; a check run never modifies the tracked
   baseline).
9. Log one row to `<RESULTS>/results.tsv` (see below).
10. Go to 1. Never stop to ask whether to continue.

To develop an idea before the confirm, iterate fast on the single case you are
changing with `<BENCH> -f <case> --quick --explore`. `--quick` is for the inner
loop only — never for a keep decision.

## Commit messages

A kept commit is the permanent record of an experiment — write it to the
standard of a Linux-kernel perf patch. Step 5's provisional message (subject +
hypothesis) is enough to measure against; when a change is KEPT, amend so one
commit carries the code change, the ratcheted baseline, and a message a reader
can judge without running anything:

- **Subject**: `perf(<package>): <what changed>` — imperative, capitalized.
- **Mechanism**: WHY it is faster — the causal explanation, and when the fast
  path applies vs falls through to the old path.
- **Measurements**, from the confirming run's `verdict.json`: every case with
  an `improved` relation, with its delta and CI bounds; the geomean wall/alloc
  deltas over the lab subset; and an explicit "no case regressed on wall or
  alloc".
- **Methodology**, one line: preset, subset, double-confirm, machine.
- **Correctness**, one line: the package test suite passed before measurement.

Numbers over adjectives; no hedging, no filler. Example shape:

```
perf(nx): Memcpy fast path for contiguous copy and concat

When a C-contiguous source is copied into a contiguous destination region,
the per-element strided copy collapses to a single memcpy. Transposed and
padded sources fall through to the general path unchanged.

Measured (thumper default preset, --tag lab, 15 cases, double-confirmed,
Apple M3 Max):

  structural/copy 1M                        wall -54.1% [-56.9%, -51.2%]
  structural/concatenate axis0 two 512x512  wall -31.7% [-34.0%, -29.3%]
  geomean over lab subset: wall -6.8%, alloc -0.0%; 0 regressions

nx test suite passes; outputs are bit-identical.
```

## Reading the deltas for the log

From `<RESULTS>/verdict.json`, per-metric `summary`:
`summary.wall_time.geomean_delta` and `summary.alloc_words.geomean_delta` are the
geomean deltas (fractions; ×100 for the percent columns, negative = better), and
`summary.wall_time.{n_improved,n_regressed,n_inconclusive}` are the case counts
for the log. This is the same file the perf gate read — one measurement, one read.

## Noise discipline

Measurement on a dev laptop is noisy; the design fights this at every step:
- The keep decision always uses the **default** preset (2% target CI, 10s/case
  cap), never `--quick` (5% CI). Under the default preset inconclusive is neutral,
  which the keep rule relies on.
- The baseline is blessed once, locally, into the committed baseline file with
  the same default preset — so it matches this machine's OCaml version and
  profile (no environment check trips) and shares the check's measurement
  settings.
- `alloc_words` is deterministic: an allocation regression is a hard, reliable
  stop (read it from the JSON, not the exit code); an allocation win is a real
  win even when timing is in the noise.
- Every keep is decided on a pair of runs, per case (see "The objective") —
  single-run verdicts on micro-cases (≲20µs) are not actionable either way.
- Measure on an otherwise-idle machine, and **never while any build is running —
  including your own**. Finish building and testing (loop steps 3–4), let it
  settle, then run the perf gate (step 6). Another agent or the human's watch
  build inflates variance.
- Check load before every gate run: compare `uptime`'s 1-minute load average
  to the core count (`sysctl -n hw.ncpu`); while load exceeds about half the
  cores, wait and retry. If a quiet window never comes, say so in the log
  rather than measuring — a measurement taken on a loaded machine is worse
  than none.

## Logging results

Append one tab-separated row per iteration to `results.tsv`:

```
commit	suite	geomean_wall_pct	geomean_alloc_pct	improved	regressed	inconclusive	status	description
```

- `geomean_wall_pct`, `geomean_alloc_pct`: geomean deltas, negative = better;
  blank on crash.
- `improved` / `regressed` / `inconclusive`: wall_time case counts.
- `status`: `keep`, `discard`, or `crash`.
- `description`: the hypothesis in a few words. No tabs.

Do not commit `results.tsv`; it lives under the gitignored `lab/results/`.

## NEVER STOP

Once the loop begins, do not pause to ask the human anything. They may be
asleep. If you run out of ideas, think harder: re-read the in-scope code for hot
paths, re-read the target's perf notes, revisit near-misses and combine them,
try a more radical rewrite of one kernel. The loop runs until you are manually
stopped, period.
