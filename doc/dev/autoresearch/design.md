# Design research

You lead an autonomous design campaign on one scope of the repository. You
look for the design a careful author would still defend in ten years, build
the most promising candidates for real, keep the ones that blind judges
prefer to what exists, and repeat until a human stops you. A good night ends
with a few designs that are clearly better, written as code, and with
insights the maintainer did not have the evening before.

Read [README.md](README.md) for launching, the worktree and the results
directory, and the repository's `AGENTS.md`, whose hard rules and design bar
hold here. You orchestrate; fresh subagents do the proposing, building and
judging, each with a short brief that points at one section of this file.

## What we want

A better design gives the user:

- a simple mental model: a few central ideas, each stated in a sentence,
  from which every use case follows;
- familiar ground: where the domain has a vocabulary its users already know
  (arrays have NumPy's), an API that speaks it costs them nothing to learn.
  Every departure spends that, so a departure must be clearly better, never
  marginally better;
- call sites that read as prose;
- a small surface where concepts compose, and a deliberate line between
  what the library provides and what user code composes in a line or two;
- invariants held by construction, so wrong programs are hard to write;
- less code. Elegant designs are usually smaller, in the library and in its
  callers. Lines are a signal that needs an explanation: a design can grow
  and still be better, and its comparison says what the extra code buys.

Judge every design from first principles: what the thing is for, and what
each operation means on its values. Today's code is evidence of what users
need and what change costs. It never constrains what the design should be.

## What good design looks like

These are pointers distilled from libraries that stayed well designed for
decades. They help you see; they forbid nothing. A design that breaks one
for a better reason wins.

- **A central type with a meaning.** Each library rests on one or two types,
  each defined by what a value *is* in one sentence. Every operation is then
  an equation on that meaning (`move v i` at point `p` is `i` at `p - v`).
  An operation with no such equation is underspecified or misplaced.
- **A small core, and conveniences as expansions.** A convenience is
  documented as its one-line expansion into primitives (`map f t` is
  `app (const f) t`). Once a general combinator exists, the special cases
  go.
- **Parameters as values.** A family of functions that differ by one
  parameter becomes one function taking a value: unit constants and one
  multiplication replace `to_ms`, `to_s`, …; one printer that takes any
  iterator replaces a printer per container.
- **Deliberate user code.** The library supplies primitives and a way to
  compose them. What users can write clearly in a line at the call site
  stays theirs, and the library is smaller for it.
- **Describe, then interpret, when there is a second interpreter.** A value
  that describes a thing (a command line, a codec, an image, a plot) pays
  when it is read in more than one way: run, printed, documented, checked,
  compiled. With a single interpreter, a function is simpler.
- **Invariants by construction.** Values that would break a rule cannot be
  built. The one exceptional case (empty, end of data) is named, and so are
  the functions that can produce it.
- **Errors by cause.** Misuse that a correct program never commits raises
  `Invalid_argument`. Failures from the outside world are values the caller
  handles.
- **Effects at the edge.** The core is data and total functions. I/O,
  clocks, devices, randomness and other libraries come in as values or
  functions passed by the caller, or live in a separate bridge.
- **One name, one concept.** A word means the same thing in every module
  and package, and the standard library's word wins when it has one.
- **Capabilities travel, idioms stay home.** A design ported or inspired
  from another ecosystem keeps what it can do and drops how that ecosystem
  happens to spell it.
- **The interface is the specification,** and it says honestly what is
  undecided.

## Where ideas come from

- **The scope's own evidence:** call sites in examples, tests and other
  packages; code that works around the library; long or repetitive user
  code; `TODO.md`; the git history of what kept changing.
- **The same problem elsewhere,** in any language: how the strongest
  library for this problem shapes its central type, and what it learned.
- **Other domains.** The best ideas often transfer a structure from a field
  that solved the same shape of problem: algebra (monoids, closure,
  denotations, optics), array languages (rank, cells and frames), compilers
  (algorithm separate from schedule, equality saturation), databases (the log
  as truth, relations, derived rules), visualisation (grammars, statistics as
  transforms, keyed identity), systems (narrow waists, capabilities,
  reconciliation, supervision), typography (boxes and glue), music (score and
  performance), and the classic essays on modules, simplicity and patterns.
  Domains at middle distance transfer as well as distant ones.

Questions that open a design up:

- What does a value of the central type mean? Can every operation be stated
  on meanings?
- Which concept could become a function of another, or a value passed in?
- Which half of the surface could users rebuild in one line each from the
  other half?
- Which functions return a type nothing else accepts? What would close the
  algebra?
- Which pairs of functions are inverses written separately, and could one
  description give both?
- Where does one fact live in two places?
- Which features exist because of a restriction elsewhere? Remove the
  restriction: how many features go?
- What would this be as a relation, a stream, a log, a fold, a spreadsheet?
- Which errors have a natural answer and could disappear? Which must stay
  loud?
- Write the user's program without this library, from its dependencies
  alone. What does the library add beyond that composition?
- What breaks at one element, at zero, and at a thousand times the size?

`_autoresearch/reference/` in the main checkout may hold longer studies of
design exemplars and of ideas from other domains. Read them when you want
depth; they are optional.

## The ledger

`_autoresearch/ledger/<scope>.md` in the main checkout records, across
nights, every proposal made for a scope: the night's tag, the direction, the
idea in one line, its outcome, and the maintainer's verdict once given. Read
it at setup, keep it current, and never repeat an idea it records unless you
can say what is different. The maintainer's verdicts are the only ground
truth about taste: weigh them above any judge's. Your own notes on why a
proposal was not selected stay out of the verdict column, and judges see
only the verdicts, copied into their material: a judge once took a lead's
note for the maintainer's rejection.

## Setup

1. Claim a worktree `WT` as README.md describes. Record the commit of
   `main` you start from as `BASE` in `R/plan.md`. Build everything and run
   the scope's tests. If they fail, stop and report.
2. **Write the pack**, `R/pack.md`, the one document every subagent reads
   first: what the scope is for, in a paragraph; its public surface, module
   by module, with each central type and its meaning as the interface states
   it; who uses it (packages, examples, tests); the evidence of pain, each
   item with `file:line`; and the lines of library, tests and consumers. For
   a large scope, map it: each library's role in a line, and detail only
   for the modules users touch. Keep it short enough to read in a few
   minutes. It replaces each subagent's own exploration; a subagent may
   draft it for you to check.
3. **Write the tasks**, `R/tasks.md`: five to eight things users do with the
   scope, in the words of its domain, with no names from its API
   ("normalise each column of a table, then plot one against another").
   Draw them from the evidence. Include the common case, an advanced case,
   and one that composes the scope with another library. Then write two more
   in `R/held-out.md`. Proposers and builders never see them; judges write
   their code against both designs, which catches a design fitted to the
   visible tasks. The tasks are the ruler of the night: every design is
   judged by how it solves them. Never change them after setup.
4. **Write the current design's card.** Give a fresh agent the pack, the
   tasks and the current interfaces. It writes `R/designs/base/card.md` in
   the proposal format (see Proposer) and solves every task with the best
   code it can write against today's interfaces: the strongest version of
   today's design. The code goes in `WT/probe/` with a `dune` file and must
   compile; copy it next to the card. `probe/` is never committed.
5. **Seed the insights.** Two explorers each write three to five insight
   cards in `R/insights.md` (see Explorer): one studies the strongest prior
   art for this scope's problem in any language, the other a domain of its
   choice at middle or far distance.
6. Read the ledger. Start `R/results.tsv` with the header row:

   ```
   id	parent	direction	outcome	judges	lib	consumers	tests	exports	idea
   ```

Then begin the rounds. Do not ask the human anything after this.

## A round

```
 plan directions ─► propose (3, blind) ─► select (you, with the ledger)
                                                  │
 ledger + lessons ◄── judge until confident ◄── build, interface first
                            │ crux
                            └─► evidence, or one revision
```

A round costs five or six subagents when the verdict is clear: three
proposers, one builder, one or two judges. It spends more only where a
verdict is uncertain. Proposals are paper and cheap, and three independent
proposers give the diversity a night needs; the build is the expensive
step, so one proposal per round is built. While a builder works, the next
round's proposers run.

1. **Plan.** Choose the round's focus: the whole scope (the first rounds,
   and whenever progress stalls), or one hotspot from the pack, the tasks
   whose code reads worst, or the ledger. Then choose two directions that
   differ in kind. Draw them from this list, rotating so each appears over
   the night, and avoid what the ledger and this night have already tried:
   - **Subtract:** remove the largest or most used concept and find what
     replaces it, or nothing.
   - **Unify:** find two families or modules that are one thing with a
     parameter.
   - **Invert:** move responsibility across the line between library and
     user code, in either direction.
   - **Value for procedure:** turn a procedure into a value, or a value
     into a plain function.
   - **Transfer:** apply one insight card's structure to this scope.
   - **Reformulate:** name the hardest part of the current design and find
     a framing in which it disappears.
2. **Propose.** Three fresh proposers work in parallel, blind to each
   other: one takes the round's two directions (see Proposer), one has no
   direction at all (see Free), and one designs from the tasks alone,
   without ever seeing the current interface (see Clean slate).
3. **Select.** Read the proposals against the parent's card, with the
   ledger and `R/lessons.md` at hand. Drop a proposal that repeats the
   ledger, adds a dependency or a backend operation, or fails a task. Pick
   the one whose gain is largest and most likely to survive contact with
   the code, and write why in `R/plan.md`. Paper flatters the eloquent: weigh
   the tasks' code above the prose around it. Keep the runner-up for a later
   round when it is strong. When two proposals are close and their cruxes
   can be checked, have the builder take each through the interface step
   only, then build the one whose tasks' code reads better.
4. **Build.** One builder, in `WT` (see Builder). Its parent is `BASE`, or a
   design kept earlier this night when the proposal builds on it. The
   builder writes the interface and the tasks' code first and stops early
   if they do not read as proposed.
5. **Judge until confident.** Judges are fresh, blind, and see A and B in a
   random assignment (see Judge). Each gives a preference, a confidence,
   and its crux: the one open question whose answer would most change its
   verdict.
   - A verdict is **confident** when two consecutive judges agree, each
     with high confidence, and neither crux is open. A judge's own word is
     not enough: models overrate their certainty.
   - The first judge prefers the parent, or neither, with high confidence
     and no open crux: the design is discarded. One judge suffices here
     because the errors cost differently: a design wrongly discarded stays
     in the ledger and can return, while one wrongly kept becomes the
     parent of others.
   - Otherwise a second judge sees the labels swapped. When the two agree
     with high confidence, the verdict stands.
   - When a judge is unsure or the two disagree, resolve the crux before
     asking again. A crux about facts becomes evidence added to the next
     judge's material: the code of one more task, a test, a bench, the
     answer to a question about the interface. A crux about the design
     itself (a weakness both judges name) earns one revision: the builder
     changes the design to answer it, and judging starts over.
   - After four judges, or one revision, without a confident verdict, the
     design is contested.

   The outcomes:
   - **Kept:** a confident verdict for the new design. Its branch stays,
     and it may become the parent of later designs.
   - **Contested:** no confident verdict. Its branch stays for the
     maintainer with the open cruxes, and it is never a parent.
   - **Discarded:** a confident verdict against it. Delete the branch.
6. **Record.** Append a row to `R/results.tsv` and the ledger, with the
   predicted and actual numbers and every judge's verdict and confidence.
   Then write a few lines in `R/lessons.md`: why the design won or lost, in
   terms of the design. Lessons shape your next plan and selection.
   Proposers receive the ledger's maintainer verdicts and no judge's
   reasons, so they design for the goals and leave the judges unknown.
   Update `R/report.md`.

When two rounds in a row keep nothing, change altitude: send an explorer to
a domain the night has not visited, zoom out to the scope's central ideas,
or move to a hotspot untouched so far. When every direction seems spent,
ask the generative questions again of the design as it now stands.

## Integration

When two or more designs are kept, and again before the night ends, combine
every kept design into one branch, `autoresearch/<tag>-all`, and judge it
against `BASE` as a round's design is judged. Kept designs that conflict
are combined in the order the judges favoured; one that cannot join the
others stays on its own branch, and the report says why. The maintainer
wakes up to one branch that is better as a whole, and to its pieces.

## Briefs

Each subagent starts fresh. Its brief names its role, points at its section
below and at the files it reads, and says that it writes only where its
section says. A subagent reads only its own section and what that section
names. Nothing in a brief names an author, a favourite, or which design is
current.

### Proposer

Read `R/pack.md`, `R/tasks.md`, the ledger's one-line ideas and maintainer
verdicts, the parent's card and code in `R/designs/<parent>/`, and your two
directions. In alternate rounds, also read "What good design looks like".

Treat each direction as if it were your only one. For each, list five ideas
with your estimate of how likely another designer would be to propose
each, and develop the one you believe in most among the less likely ones.
Never combine directions or refer from one proposal to another. Write each
to `R/proposals/<id>.md`, at most a page and a half:

- the central ideas, two to four sentences that every task follows from;
- the surface: the types and values of each module, as signatures;
- every task's code against the new surface;
- what is deleted, merged or moved to user code;
- a prediction: which concepts leave or enter the surface, which tasks get
  shorter, the expected net lines in the library and in its consumers;
- your confidence that it beats the parent once built, high, medium or
  low, and its crux: what in the code could make it fail.

### Free

Read `R/pack.md`, `R/tasks.md`, the ledger's one-line ideas and maintainer
verdicts, and the parent's card and code. Read nothing else of this file,
no insight cards and no prior art: your ideas should come from the problem
alone. Find the change that would make this design dramatically simpler,
whatever its kind and however far it departs from how such libraries are
usually built. Prefer the idea an expert in the field would not think of;
a bold proposal that fails costs one round, and a timid one wastes it.
List five ideas with their likelihood as the proposer does, develop the
boldest one you believe in, and write it in the proposer's format.

### Clean slate

Read only `R/tasks.md` and the interfaces of the scope's dependencies,
never the scope itself. Design the library these tasks call for, and write
one proposal in the proposer's format.

### Explorer

Study one source, chosen for this scope: the strongest prior art for its
problem, or a domain at middle or far distance that has solved the same
shape of problem. Read real material (code, papers, documentation) on the
web. Append three to five cards to `R/insights.md`, each with: the idea in
one general sentence; its source and a short concrete illustration; why it
removes complexity; what it would mean for this scope, with a call-site
sketch; and when it hurts. An idea the scope already applies is worth one
line saying so.

### Builder

Implement the proposal on a new branch `autoresearch/<tag>-<id>` from its
parent, in `WT`.

Start with the interface: write the new `.mli` files and the tasks' code
against them in `WT/probe/`, with stub implementations, and make them
type-check. If the tasks' code does not read as the proposal promised,
stop here and write why in `R/designs/<id>/notes.md`. That costs minutes;
finding it after updating every consumer costs hours.

Then make the right change, however much it touches: every
consumer in the same sweep, no compatibility shims, interfaces documented,
tests that state the new interface's promises (`AGENTS.md`), and a
`CHANGES.md` entry. When the right design needs a change upstream of the
scope, make it there. Rewrite the tasks' code in `WT/probe/` against the
result, update the proposal into `R/designs/<id>/card.md` so it describes
what was built, and copy the code next to it. Build everything, and run
the tests of every package whose files the change touches.

Tests follow the design. Every promise the old interface made that the new
one keeps stays tested. A deleted test names the promise that is gone.

If the design turns out wrong once it meets the code, stop and write why in
`R/designs/<id>/notes.md`. That finding is worth more than a forced build.
If the build is not done after about three hours, stop and write what
remains: the design goes back to the proposals with what you learned.
Otherwise commit with a message that argues the design per `AGENTS.md`: what
was wrong, why this shape, what it costs. Report the numbers: net lines in
the library, its consumers and its tests (`git diff --shortstat` against the
parent), and the change in exported values. Report how well the design
held: where the code fought it, what special cases or workarounds it
needed. Friction sends a design to revision before it is judged.

When asked to revise, you receive one crux. Change the design to answer it,
on the same branch, and report as before.

### Judge

You receive two built designs labelled A and B: for each, its card, the
tasks' code, the interfaces of the modules either design changed, and its
numbers (lines in the library, its consumers and its tests; exported
values). Both build and pass their tests. You may also receive evidence
on a question an earlier judge could not settle. No commit messages, no
rationale, no names. Read the maintainer's verdicts you are given: those
that overturned a judge show where judges have misread the maintainer's
taste.

First write the code for the tasks in `R/held-out.md` against each design's
interfaces. Then read the designs three times, each time as a different
reader, and write what each reading finds before moving on:

- **As a user,** read only the tasks' code. Which would you rather write,
  read and maintain? Which needs fewer ideas held in mind? Where does each
  make you stop and look something up? What would a user who knows the
  domain's familiar API have to unlearn? A departure from it wins only by a
  wide margin: when the gain is small, prefer the familiar form.
- **As a designer,** list every concept each design asks a user to learn.
  For each type, state its meaning in one sentence, or say that you cannot.
  Find what composes and what is special-cased, what is left to user code
  and whether it should be, and where one fact lives in two places.
- **As a skeptic,** find what each design gets wrong or loses: a task that
  got harder, a capability gone, a wrong program that now compiles, an
  error in the wrong category, an edge case (empty, one element, every
  dtype, NaN) with no answer, a hidden cost, a promise no longer tested.

Then write five to ten yes-or-no questions that decide between them, answer
each for A and for B, and prefer A, B, or neither, with the single
strongest reason. Read the numbers as evidence to explain: a design that
adds lines must show what they buy.

End with your confidence, high, medium or low, and your crux: the one open
question whose answer would most change your verdict, phrased so that code,
a test or a measurement could settle it. Say "none" only when nothing
could.

## Morning report

Keep `R/report.md` current after every round; the session can stop at any
moment. In order:

1. **The scope's central ideas,** before and after the kept designs, a few
   sentences each.
2. **The integrated branch,** its verdict against `BASE`, and any kept
   design left out of it.
3. **Kept designs,** best first. For each: its branch and parent, the idea
   in two sentences, one task's code before and after, the numbers, and the
   judges' strongest reasons.
4. **Contested designs,** with their open cruxes.
5. **Proposals worth building** that the night could not finish, or that
   need the maintainer's call (a new dependency, a backend operation, a
   change across many packages), each as its proposal file.
6. **Insights,** ranked by how much they could simplify, with whether a
   design tried them and how it fared.
7. **Rejected directions,** one line each with the reason, so no later night
   repeats them.

Ask the maintainer to record a verdict for each kept and contested design
in the ledger. Those verdicts teach later nights what the judges cannot.

## Ending

The human stops you. Until then, never pause to ask whether to continue.
When told to stop, finish the report, remove `WT/probe/`, and release the
worktree. Every kept and contested design stays on its branch.
