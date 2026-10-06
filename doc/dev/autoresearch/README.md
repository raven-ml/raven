# Autoresearch

Four programs an agent runs unattended, typically overnight, on one scope of
the repository:

- [perf.md](perf.md) makes the scope faster. Its ruler is the scope's
  benches and tests: a change is kept when the measurements show it helps
  and harms nothing.
- [simplify.md](simplify.md) returns the scope to a pristine state after
  fixes and performance work have piled debt onto its design: one
  simplification at a time, each keeping every fix and every performance
  gain. This is the program for making a library simpler.
- [design.md](design.md) makes the scope's design better: fewer concepts,
  call sites that read as prose, more done by composition, less code. There
  is no number to optimize, so the program builds its ruler from user tasks
  and blind comparisons. It explores large alternatives and is best at
  producing design insights and proposals.
- [tests.md](tests.md) makes the scope's tests and benches better: each test
  of the right kind, checking a promise the interface makes, and suites
  that read as prose and run fast.

All four run experiments in an isolated worktree, record every attempt in a
results directory, and leave a report for the maintainer to read in the
morning.

## Launching

Start a fresh agent session at the root of the main checkout and give it the
program, a scope and a tag:

```
Run doc/dev/autoresearch/design.md on packages/talon, tag design-talon-oct06.
```

The scope is a package (`packages/talon`), a sub-library
(`packages/nx/lib/io`), a module, or a boundary between libraries ("how
kaun's training loop composes with vega"). The tag is
`<program>-<scope>-<month><day>`. The performance program also accepts a
filter on bench cases and a quiet machine to time on, with the command
prefix for timing there.

## The worktree

Claim a worktree from the pool as `AGENTS.md` describes, with the lock
reason `<session> autoresearch <tag>`, and run one passive dune watch server
in it. Every experiment happens in this worktree, where the program may `git
reset` to discard one. In the main checkout, the program writes only its
results directory.

The performance, simplify and test programs keep their changes on one branch,
`autoresearch/<tag>`, each kept change on top of the last. The design
program builds each candidate on its own branch, `autoresearch/<tag>-<id>`,
from the starting commit or from a design kept earlier that night.

## Results

The results directory `R` is `_autoresearch/<tag>/` in the main checkout.
Git ignores it, and it outlives the worktree. It holds the program's plan,
its log of every attempt, its working notes, and `report.md`, which the
program keeps current so it can be read whenever the session stops.

`_autoresearch/ledger/<scope>.md` records, across nights and programs, every
attempt made on a scope and its outcome, so no night repeats another's
work. After reading a report, write your verdicts there: on each kept and
contested design, on deletions a test night made. The programs read these
verdicts and weigh them above their own judges.

## Ending and landing

Stop a session by telling it to stop. It finishes its report, releases the
worktree, and leaves its branches. Each commit on them builds, passes the
tests, updates every consumer and carries its `CHANGES.md` entry when the
change is visible to users, so any subset can be cherry-picked onto `main`.
