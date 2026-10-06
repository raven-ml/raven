# Simplify

You return one scope of the repository to a pristine state. Work since its
design was last thought through (bug fixes, performance work, features
added under pressure) has accumulated debt: special cases, knobs, exported
internals, names that drifted, rules with exceptions, duplicated paths.
Your job is to remove that debt while keeping every fix and every
performance gain: one simplification at a time, each proven to change no
behaviour, keep every measurement, and leave the scope simpler. You keep
going until a human stops you.

The target is the best design for what the scope does today, knowing
everything the fixes and the performance work taught. Often that is the
original design with its debt removed, and most steps remove something.
But debt is also evidence: when fixes kept patching one mechanism, or every
caller carries the same special case, the design did not fit what the code
had to do, and the right simplification is a better design, thought
through from first principles (see Reshapes).

Read [README.md](README.md) for launching, the worktree and the results
directory, and the repository's `AGENTS.md`, whose hard rules and design
bar hold here. You orchestrate; fresh subagents audit, implement and review,
each with a short brief that points at one section of this file.

## What simpler means

The pristine state gives the user:

- a mental model that reads in one pass: a few central ideas, each stated
  in a sentence, from which every function follows;
- one rule where there are several, and rules without exceptions;
- call sites that read as prose;
- the smallest surface that serves every use, with conveniences documented
  as their one-line expansion into primitives;
- familiar ground: where the domain has a vocabulary its users know (arrays
  have NumPy's), the API speaks it;
- less code, in the library and in its callers.

Most of this can be counted. Before and after each change, count: public
values, types and modules; optional arguments; function families that
differ by one parameter; sentences of caveats in the interface (those
containing "except", "unless", "only if", "note that", or a constant stated
as contract); lines of library, consumers and tests. A change is simpler
when these go down and no call site reads worse. When a count goes up, the
change must say what the extra buys.

## What the pristine state looks like

Pointers, distilled from libraries that stayed well designed for decades.
They help you see; they forbid nothing.

- **A central type with a meaning,** defined in one sentence; every
  operation an equation on that meaning.
- **A small core, and conveniences as expansions.** Once a general
  combinator exists, the special cases go.
- **Parameters as values.** A family of functions that differ by one
  parameter becomes one function taking a value.
- **Deliberate user code.** What users write clearly in a line stays theirs.
- **Invariants by construction.** The one exceptional case is named.
- **Errors by cause.** Misuse raises `Invalid_argument`; failures from the
  world are values.
- **Effects at the edge.** The core is data and total functions.
- **One name, one concept,** across modules and packages.
- **The interface is the specification,** short enough to read, honest
  about what is undecided.
- **Libraries keep their weight to themselves.** A change never makes a
  library depend on a heavier one that its other users do not need.

## What accretion looks like

The audit sorts what it finds into these kinds. Each has a typical remedy.

| Kind | Looks like | Remedy |
|---|---|---|
| Redundant | a function a fix added that another now expresses | delete it, or keep the general one and document the expansion |
| Knob | an optional argument or flag added for one caller | move the choice to that caller, or find the value that removes it |
| Special case | a rule stated with exceptions; an `if` on one dtype, device or shape | find the one rule that covers the cases |
| Leak | an internal concept exported for a test or one library | make it private; test through the public interface |
| Drift | a name or meaning that diverged from its siblings or from the design | restore the one name and meaning |
| Duplicate path | two implementations of one operation, a fast path that grew its own semantics | one path, with the fast path an implementation detail |
| Dead | a value no one calls outside its own tests | delete it with its tests |
| Patch stack | several fixes to one mechanism | the mechanism is in the wrong place: a structural change (see Reshapes) |
| Caveat pile | interface docs that list rules the code enforces in many places | one rule in the code, one sentence in the docs |

## The preserved list

Simplifying must lose nothing that work since the pristine point gained.
The preserved list makes that checkable, and it is the ruler of the night:

- every fix since the pristine point, with the test that guards it (from
  `git log` and the commit's diff); a fix without a test gets one first;
- every performance gain since then, with the bench rows that show it, and
  a reference build of each bench executable;
- every capability added since then that users rely on (a caller outside
  the scope uses it).

A change that breaks a preserved test, regresses a preserved bench row, or
removes a relied-on capability is discarded, however much simpler it is.

## Setup

1. Claim a worktree `WT` and create the branch `autoresearch/<tag>` from
   `main`, as README.md describes. Every kept change lands on this one
   branch, on top of the last. Create the results directory `R`.
2. Fix the pristine point: the commit where the scope's design was last
   thought through, usually where its RFC (in `doc/rfc/`) landed. The
   launcher may name it; otherwise find it from the RFC and the history,
   and write it in `R/plan.md` with the reason.
3. Build everything and run the scope's tests. Build each bench executable
   that covers the scope and copy it to `R/ref/`. If anything fails, stop
   and report.
4. **Audit** (see Auditor): one fresh agent writes `R/accretion.md`, every
   piece of accretion with its kind, its evidence and what must be
   preserved, and `R/preserved.md`, the preserved list.
5. **Cold read** (see Cold reader): one fresh agent reads the scope's
   interfaces as a new user and lists every place it had to stop and think.
   Add those to `R/accretion.md`.
6. Write `R/opportunities.md`: the simplifications to make, ranked by what
   they remove per unit of risk, each with the accretion items it resolves,
   the preserved entries it must keep, and its counts before. Start
   `R/results.tsv` with the header row:

   ```
   commit	item	kind	values	options	caveats	lines	review	status	description
   ```

Then begin the loop. Do not ask the human anything after this.

## The loop

1. Take the top opportunity. Trivial items of one kind (dead values, a name
   restored across a module) can go together in one commit.
2. **Implement** (see Simplifier) on the branch: the change, every consumer
   in the same sweep, no compatibility shims, the interface docs, and a
   `CHANGES.md` entry when users see it. For a change to a public
   interface, write the before and after call sites first; if the after
   does not read better, stop.
3. **Gates**, all mechanical:
   - the build is green, and the tests of every package whose files
     changed pass;
   - every preserved test passes;
   - every preserved bench row, and every bench row the change can affect,
     shows no regression against `R/ref/`: no allocation increase, and no
     timing regression by the evidence rules of [perf.md](perf.md);
   - `git diff --name-only` touches no test that guards a preserved entry
     except to follow a changed interface;
   - no new dependency between libraries.
4. **Review** (see Reviewer): one fresh reviewer. When it keeps the change
   with high confidence, keep it. When its confidence is lower, settle its
   crux with evidence (a count, a search, a test, a measurement) and ask a
   second reviewer; keep only when both keep it.
5. **Keep:** commit, with a message that says what had accreted and why,
   why it can go now, and what is preserved. **Discard:** `git reset --hard`
   to the previous commit, and record why in `R/opportunities.md`; a
   discarded idea is never retried blindly.
6. Append a row to `R/results.tsv` and update `R/report.md`.

Every five keeps, ask for a new cold read of the interfaces as they now
stand, and re-rank `R/opportunities.md`: each simplification exposes others.

## Reshapes

When several accretion items share a cause (a patch stack, the same caveat
in five places, a special case repeated in every caller, a rule that keeps
needing exceptions), the design itself is in question. Step back and ask
what the best design for this part of the scope is, from first principles:
what it is for, and what each operation means on its values, given
everything the preserved list says it must do. Today's design is one
candidate among others; consider at least one genuinely different shape
(another central type, another split between modules, another line
between library and user code), and keep the one in which the accretion
disappears.

Write `R/reshapes/<name>.md`, at most two pages: the problem and the
accretion that shows it, the design today, the alternatives considered and
why each loses, the design chosen and its central ideas, every preserved
entry and how the new design keeps it, the call sites before and after,
and the counts before and the prediction after. Two fresh reviewers read it
before anything is built, with the labels swapped; build it only when both
agree it is simpler and keeps everything. A reshape lands as a series of
commits, each passing the gates.

A reshape that needs a decision only the maintainer can take (a new
dependency, a backend operation, a change across many packages, a
departure from the familiar vocabulary) goes to the report as a proposal.

## Briefs

Each subagent starts fresh, with a brief that names its role, points at its
section below and at the files it reads, and says where it writes.

### Auditor

Read the scope's RFC, its interfaces at the pristine point and today, and
`git log -p` of those interfaces and of the implementation since the
pristine point, with each commit's subject and message. Read the callers
outside the scope. For every addition and change to the public surface, and
for the larger internal ones, write in `R/accretion.md`: what it is, the
commit that introduced it and why (fix, performance, feature), its kind
(see What accretion looks like), the evidence (`file:line`), what of it must
be preserved, and the simplification you see, if any. Write `R/preserved.md`
from the same history: each fix with its test, each performance gain with
its bench rows, each relied-on capability with a caller. Facts only, with
locations; mark a guess as a guess.

### Cold reader

Read only the scope's public interfaces, top to bottom, as a user who knows
the domain and has never seen the library. Write down every place you had
to stop: a concept you could not define in a sentence, a rule with an
exception, two names for one thing, an argument whose effect you could not
predict, a function whose reason to exist you could not see. Quote each
place. Do not propose fixes.

### Simplifier

Make one simplification, or one batch of trivial ones of the same kind, as
`R/opportunities.md` describes it, on the branch in `WT`. Keep every
preserved entry it names. Update every consumer, the interface docs, the
tests (a test of a removed promise goes, with the promise named in the
commit), and `CHANGES.md` when users see the change. Run the gates. Report
the counts before and after, the gate results, and anything the change
could not keep. If the simplification turns out wrong once it meets the
code, stop and write why; that finding is worth more than a forced change.
Stop after about an hour per opportunity.

### Reviewer

You receive one change: its diff, the interface before and after, the call
sites before and after, the counts before and after, and the preserved
entries it must keep. Answer, quoting the code:

1. What concept, rule, value, argument or caveat does this remove, or what
   reads better? Name it.
2. Does any call site read worse, or any use become harder?
3. Is anything from the preserved entries weakened?
4. Is this the shape a careful author would write today, or a half-step
   that leaves something worse in between?

Then keep, revise (say exactly what), or drop, with your confidence (high,
medium, low) and your crux: the one question whose answer would most change
your verdict, phrased so a count, a search, a test or a measurement could
settle it.

## Morning report

Keep `R/report.md` current after every keep; the session can stop at any
moment. In order:

1. **The counts** for the scope, at the pristine point, at the start, and
   now: public values, types, optional arguments, interface caveats, lines
   of library, consumers and tests.
2. **The simplifications kept,** one line each with its commit, grouped by
   kind, and the three that matter most with their call sites before and
   after.
3. **Reshapes** built and proposed, and **decisions for the maintainer.**
4. **Drift from the RFC:** where the code and the RFC now disagree, and
   which should change.
5. **Discarded simplifications,** one line each with the reason.

## Ending

The human stops you. Until then, never pause to ask whether to continue;
when the opportunities run out, ask for a cold read and audit again from
what the branch now is. When told to stop, finish the report and release
the worktree. The branch keeps every kept change, each one green on its
own, so the maintainer can take any prefix of it.
