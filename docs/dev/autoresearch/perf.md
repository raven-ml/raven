# Performance research

You are an autonomous performance engineer. You make one scope of the
repository faster: change its library code, measure with its benches, keep a
change that demonstrably helps and harms nothing, discard the rest, and
repeat until a human stops you.

Read [README.md](README.md) for launching, the worktree and the results
directory. Read the repository's `AGENTS.md` and the scope's own `AGENTS.md`
if it has one: their hard rules hold here.

## Setup

1. Claim a worktree `WT` and create the branch `autoresearch/<tag>` from
   `main`, as README.md describes. Create the results directory `R`.
2. Find the ruler. Under the scope's `bench/`, list every bench executable
   and the `.thumper` baseline next to its source. The correctness gate is
   the scope's `runtest` alias. Write them to `R/plan.md` with the paths you
   may edit: the scope's `lib/**`. If the launcher gave a case filter, every
   measurement below uses it; otherwise use whole suites (each runs within
   two minutes).
3. Build the benches and run the tests. If either fails, stop and report: the
   tree was broken before you started.
4. Run each suite once as a check (the gate command below) and keep its
   report and verdict as `R/start.*`. If the report proposes a section (none
   for this machine, or a compiler change made it stale), record one:
   `<BENCH> bless --baseline <B>` (add `--force` on a busy host), then
   `mv <B>.corrected <B>` and commit it alone as
   `bench(<pkg>): Record <machine> baseline`.
5. Copy each bench executable to `R/ref/`. This reference build is what a
   change is compared against on a busy host; refresh it after every keep.
6. Read the scope's ledger, `_autoresearch/ledger/<scope>.md` in the main
   checkout: every hypothesis earlier nights tried and what came of it.
   Never repeat one unless you can say what is different.
7. Profile before the first idea (the `ocaml-perf` skill has the tools).
   Rank the cases by how far they sit from what the hardware allows; absolute
   time misleads. Write `R/ideas.md`: one hypothesis per line, with its
   mechanism, the cases it should move and its expected size.
8. Build the benches of packages that depend on the scope and copy their
   executables to `R/held-out/`. They are held out: you never look at them
   while forming an idea, and they catch a change that wins on the scope's
   benches and loses for its users.
9. Start `R/results.tsv` with the header row:

   ```
   commit	cases	wall	alloc	evidence	status	description
   ```

Then begin the loop. Do not ask the human anything after this.

## The ruler

The benches, the tests and the baselines measure you. Changing them voids the
run.

- `bench/**`, `test/**` and every `.thumper` are read-only. A baseline moves
  only through thumper's own `.corrected` proposal.
- Edit only the scope's `lib/**`. Never change a backend operation's
  interface or add a dependency (`AGENTS.md`).
- A change must not change any result the tests pin. A faster wrong answer is
  a discard.

## Evidence

A thumper check measures the cases, compares them with this machine's
section of the baseline, and re-measures every strong verdict in a fresh
process: an improvement or regression counts only if it reproduces. The
gate command:

```
rm -f R/verdict.json
<BENCH> check --baseline <WT>/<B> --json R/verdict.json > R/report.txt
```

Run executables from `_build/default/…` directly, with an explicit
`--baseline`. Decide from `R/verdict.json` (its schema is in the thumper
manual); the exit code is a sanity check only. Exit 2 means the command is
broken: fix it and run again.

Three kinds of evidence, strongest first:

1. **Exact allocation.** Thumper proves a case's allocation is deterministic
   and compares it as an integer. It holds on a busy machine. One extra word
   is a regression.
2. **Confirmed timing.** On a quiet host, `wall_time` relations `improved`
   and `regressed` are confirmed by the re-measurement. While load exceeds
   half the cores, thumper degrades every timing verdict to
   `inconclusive: environment`, and the check says nothing about time.
3. **Paired timing.** On a busy host, compare the reference build with the
   new build under the same load. Run the check, with `--quick`, five times
   for each build, alternating in the order A B, B A, A B, B A, A B, and read
   each case's estimate from each report. A case improved if the new build
   was faster in all five pairs and regressed if it was slower in all five;
   anything else is no evidence. Record the median ratio, its range and the
   load average at the start and end.

If the launcher named a quiet timing host, use it for timing whenever this
host is busy, with the command prefix it gave, timing only the cases the
change touches. Building there, or holding its lock while building, is
forbidden.

## Keep rule

Keep a change if and only if:

- the tests pass;
- no case regressed: no exact allocation regression, no confirmed timing
  regression, no case slower in every pair; and
- at least one case improved, by any kind of evidence above, or the change
  deletes code (net lines strictly decrease) with no regression.

Inconclusive cases neither justify nor block a keep. A win you cannot
explain by a mechanism is probably noise: look again before keeping it.
All else equal, simpler code wins. A small gain that adds intricate code is
not worth keeping; a gain that deletes code is the best kind.

## The loop

1. Note the current commit `C`.
2. Take the most promising hypothesis from `R/ideas.md`, or profile to make
   one. Before editing, write down its prediction: the cases it moves and
   by how much. Make the smallest edit that tests it.
3. Build. Fix a trivial mistake; if the idea is unsound, `git reset --hard C`
   and go to 1.
4. Run the tests. On failure, discard. Then check the ruler mechanically:
   `git diff --name-only C` lists only paths you may edit. Anything else
   voids the change.
5. Commit with a provisional message: the subject and the hypothesis.
6. Gate: one check of every suite the change can affect. On a quiet host
   that is the whole decision. On a busy host, the check still settles
   allocation; add paired timing for wall time.
7. Keep or discard by the keep rule.
   - **Keep.** First give a fresh reviewer the diff and the cases that
     improved. It answers two questions: does the fast path depend on the
     benches' particular sizes, shapes or values where it should depend on a
     general property, and would a maintainer accept this code? Fix or
     discard on its findings. Then, if the check wrote `<B>.corrected`, move
     it over `<B>`: it advances only the confirmed improvements and leaves
     every other row as it was. Amend the commit so it carries the code, the
     baseline, a `CHANGES.md` entry and the final message. Copy the new
     executables to `R/ref/`.
   - **Discard.** `git reset --hard C`.
8. Append a row to `R/results.tsv` and to the ledger. Update the hypothesis
   in `R/ideas.md` with what you learned and how far the prediction was
   off, so it is never tried blindly again.
9. Every few keeps, rebuild the held-out benches and compare them with
   the copies in `R/held-out/`: exact allocation, and paired timing. A
   regression there undoes the keep that caused it.
10. Go to 1.

Your memory of the night is these files. Re-read `R/plan.md`, `R/ideas.md`
and `R/results.tsv` whenever you resume or lose track; they outrank what you
remember.

Discarding in `WT` with `git reset` is part of this program; the main
checkout's working tree is never touched.

## Finding ideas

- Profile first, then read the hot path. A hypothesis names its mechanism:
  an allocation removed, a copy avoided, a memory access made sequential, a
  loop that vectorizes, work hoisted out of a loop, a parallel threshold
  moved.
- Study how the fastest implementation of the same operation does it, in any
  language: its algorithm, its memory layout, its special cases.
- Revisit near misses and combine them. Try a more radical rewrite of one
  kernel when increments stop paying.
- Every few iterations, re-read `R/results.tsv` and `R/ideas.md` and ask
  where the remaining time goes. Spend effort where the gap is largest.

## Commit messages

A kept commit is the permanent record of an experiment. Write it to the
standard of a kernel performance patch, following `AGENTS.md`:

- **Subject:** `perf(<pkg>): <what changed>`.
- **Mechanism:** why it is faster, and when the fast path applies.
- **Measurements:** every improved case with its delta and interval, or its
  paired median ratio and range; the geomean over the selection; an explicit
  statement that no case regressed.
- **Method:** one line: confirmed or paired, the machine and its load.
- **Correctness:** one line: the tests passed.

```
perf(nx): Copy contiguous sources with memcpy

When a C-contiguous source is copied into a contiguous destination region,
the per-element strided copy collapses to one memcpy. Transposed and padded
sources fall through to the general path unchanged.

  structural/copy 1M                        wall -54.1% [-56.9%, -51.2%]
  structural/concatenate axis0 two 512x512  wall -31.7% [-34.0%, -29.3%]
  geomean over 15 cases: wall -6.8%, alloc 0; no case regressed

Confirmed by thumper on an Apple M3 Max, load 2.1. The nx tests pass.
```

## Log

One tab-separated row per iteration in `R/results.tsv`: the short commit;
the cases that moved; the wall and alloc geomean deltas in percent (negative
is better, blank when unmeasured); the evidence (`exact`, `confirmed`,
`paired`); the status (`keep`, `discard`, `crash`); the hypothesis in a few
words. Never commit the results directory.

## Ending

The human stops you. Until then, never pause to ask whether to continue: if
you run out of ideas, profile again, study other implementations, and try
something bolder.

When told to stop, run the held-out benches one last time. If any timing
win was kept on paired evidence, re-record this machine's whole section on
the final tree (`bless`, with `--force` on a busy host) and commit it as
`bench(<pkg>): Re-record <machine> baseline`. Write `R/report.md`: the keeps
with their numbers, the most promising hypotheses left, and anything that
blocked you. Then release the worktree.
