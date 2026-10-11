# Documentation

raven's API documentation is the doc comments in each `.mli` and the guide
pages under a library's `doc/`. This page sets its standard. rig
(`dev/rig`) is the first library documented to it, and the excerpts below
come from its `rig.mli`.

## Two readers

An `.mli` has two readers. One reads it top to bottom and should come away
with the module's model, as from a chapter of a book. The other, a person
or an agent, looks up one value and needs its whole contract without
reading the rest.

Reference documentation that serves both readers, such as Go's packages,
Plan 9's manual and cmdliner, does it with four levels of text. No text
appears at two levels.

| Level | Job | Size |
|---|---|---|
| Guide pages | Narrative, worked examples, stories across modules | a few thousand words for a library |
| Module preamble | The module's model, the life cycle of its main object, its defaults | 100 to 600 words, up to about 700 for a library's main module |
| Section preamble | What the values of a section share | a few sentences, rarely over 120 words |
| Value doc | One value's contract | median 20 to 60 words, rarely over 150 |

The chapter reader reads the module preamble and the section preambles in
order, and the guide pages for what crosses modules. The lookup reader
reads one value's doc and the section preamble above it. odoc renders an
`.mli` in source order, and agents read the raw file, so both readers meet
the text in the order it is written. The model therefore comes before the
first `val`.

The references show what happens without this shape. Python's asyncio
documented each function well and had no overview until 2025; a CPython
issue said its docs "lack a cohesive overview of the systems design and
architecture". Haskell's `MVar` states its wakeup order in the module
header and again in `takeMVar`, and the two copies disagree on whether
FIFO order is promised.

## Where things live

- **The `.mli` files own every definition and every rule,** including the
  rules that span modules. Each lives in exactly one `.mli`. An `.mli`
  states each rule in full and links only to the API: modules, types,
  values and sections. No `.mli` links to a guide page.
- **A rule that spans modules lives in the lowest module whose values it
  governs and that every reader of the rule depends on.** In rig, ordering
  and loss live in the core module, `rig.mli`. Driver obligations live in
  the driver contract, `rig_edge.mli`, and the rules of jobs across
  machines in the remote module, `rig_remote.mli`.
- **A summary module** that lists its submodules' rules either names the
  owner in each item and says no more than the owner does, or states the
  rule itself while the owner links to it.
- **Guide pages under `doc/` add examples and narrative and state no rule
  of their own.** `index.md` is the quick start, and it explains every
  argument of the first program. The other pages hold narrative, worked
  examples and stories that cross modules, such as one call traced through
  a driver to the hardware and back. Every rule a page states is stated in
  an `.mli`, and the page links to it by relative path and names its
  section, as in ``the Order section of [`rig.mli`](../lib/rig.mli)``. When
  a page and an `.mli` disagree, the `.mli` is right and the page is fixed.
- **Mechanism goes in `.ml` comments.** What a caller cannot observe, such
  as how a copy is staged or how the collector is paced, is commented
  beside the code that does it.
- **An `.mli` never holds** measurements, test history, host names, commit
  hashes, dates, or references to plans, RFCs or reviews. Citations of
  kernel or driver source, and lists of their header files, go to `.ml`
  comments.
- **An `.mli` may cite a published specification** that defines a format
  or ABI the library encodes, such as the x86-64 psABI. It may also state
  a version pin that is part of the contract, such as the firmware release
  whose images an open accepts.
- **A status** such as "written, never run on hardware" or "untested" is
  stated once, in the module concerned. Other modules and guide pages link
  to it.

## The shape of an `.mli`

### Module preamble

The preamble has these parts, in order:

1. **A first sentence that stands alone and names the thing,** as Elixir's
   GenServer opens: "A behaviour module for implementing the server of a
   client-server relation."
2. **The model in continuous prose,** as a Plan 9 manual page or a Go
   package doc gives it. It says what the module's values are, how they
   relate and why they exist. Each term is defined, marked `{e term}`,
   before its first use, and the preamble defines every term the sections
   below it use.
3. **The life cycle of the main object,** where the module has one, as
   SQLite's prepared statement page lists it: "The life-cycle of a
   prepared statement object usually goes like this: 1. Create ... 5.
   Destroy ...".
4. **Defaults, stated once,** so that silence in a value doc means
   something. Go's `io` says "unless otherwise informed clients should not
   assume they are safe for parallel execution", and each type that is
   safe says so in its own doc.
5. **One short example,** when the module has a typical use, with a
   sentence before it that says what it does and why it works.

The preamble may name the sections in a sentence of prose where their
order matters. It never lists them as questions, since odoc renders a
table of contents.

`rig.mli` opens:

```
(** Devices such as GPUs, and the order of the work they run.
```

It defines each term, marked `{e term}`, before its first use:

```
    Work on a GPU runs asynchronously. The call that submits it returns
    before the work has run. So that the host can tell what has run, rig
    numbers the work submitted to a device 1, 2, 3, … in the order the device
    accepts it. These numbers are the {e values} of the device's
    {e timeline}. A value [v] is {e reached} once the work of every value up
    to [v] has completed. A {e point} is a device together with one of its
    values ({!Point}). Work or host access {e follows} other work when it
    starts after that work has completed, and so sees its writes. It follows
    a point when it starts after the point is reached.
```

It states its defaults once, in a list that begins "Unless a function's doc
says otherwise".

### Sections

Sections come in the order the model needs, and a policy section usually
follows the types it governs. When the module's model is a policy that
governs the rest of it, the sections that state the policy may come first
with their few values, and the sections for the module's types and values
follow.

A section preamble holds what the section's values share, stated once.
cmdliner's section on optional arguments says "The information of an
optional argument must have at least one name or [Invalid_argument] is
raised.", and no value below repeats it.

A family of operations is defined once in its section preamble. Go's
`sync/atomic` says "The swap operation, implemented by the SwapT functions,
is the atomic equivalent of" `old = *addr; *addr = new; return old`. Each
member then needs one sentence: "SwapInt32 atomically stores new into
*addr and returns the previous *addr value."

Where a module has a section on failure, failure rules live there.

The first two sentences of `rig.mli`'s section on opening state what every
open shares:

```
(** {2:opening Opening and closing}

    An open returns [Error why] when the device cannot be opened, and raises
    [Invalid_argument] when it is misused. Every open is [Error] after the
    process has failed ({!fail}), while the name's device is lost and its
    driver's stop ({!Rig_edge.Driver.val-stop}) has not returned, when the
    machine's host ({!section-machines}) is lost or closed, and once the
    process has opened 65,535 devices.
```

### Value docs

- **The first sentence stands alone and opens with the value applied to
  its arguments,** `[f x] is ...` for a value and `[f x] does ...` for an
  effect. bytesrw writes "[read r] reads the next slice from [r]."
- **The parts come in one fixed order** across the library: what the
  value is or does, when it blocks, what it requires, and `Raises ...`
  last. A reader who has seen one contract knows where every other keeps
  its errors, as Rust's standard library puts errors and panics under the
  fixed headings `# Errors` and `# Panics`.
- **Keep to the sizes in the table above.** A longer contract is a family
  to define in the section preamble, or a sign that the value does too
  much.
- **A contract links its terms and never restates the model.** It repeats
  a rule only where breaking it fails silently, in one sentence, as Go's
  `sync` repeats "must not be copied" on each type.
- **Obligations use "must",** as GenServer's "A GenServer must never be
  used for code organization purposes." Use it for every obligation, and
  always for one whose violation fails silently.
- **No `@param` lists.** They restate the signature, and in Java's
  `ConcurrentHashMap` their median is 57 words, against 26 for the prose.
  A value doc has an example only when the call is not obvious.
- **Document the current API.** When a contract cannot be written
  truthfully or briefly without changing the API, write it for the API as
  it is and open an issue for the change it needs.

`rig.mli`'s `Point.wait` follows the order:

```
  val wait : t -> unit
  (** [wait p] returns once [p]'s device has reached [p]'s value. It commits
      the device's work up to that value, then blocks however long that work
      runs, unless the device has a hang bound ({!section-hangs}). While it
      blocks it releases the domain lock ({!section-domains}).

      Raises {!Lost} if the device is lost or is lost during the wait. *)
```

### Names and references

- **One name per thing, and one meaning per word.** In rig, "follows" is
  about order only, and a use "depends on" a point when it is valid only
  if that point's work has completed. A value that a lost device reports
  without completing its work is "dropped", and never "reached".
- **Name a thing by its identifier,** as in "the driver's `submit`", and
  coin no other word for it. A C function returns its result.
- **A `{!...}` reference goes only to the file's own library or to one it
  depends on.** A reference up the dependency graph, such as a wire format
  naming the library built on it, is written as code: `[Rig_remote]`.
- **Lines in an `.mli` stay under 80 columns,** with odoc markup as the
  surrounding files use it.

### Private modules

A private module's `.mli` is its contract with its siblings. Its preamble
names what it holds, then states its invariants, concurrency, errors and,
for C, ownership, where they exist. A value gets a doc only for what its
name, type and the public docs leave unsaid: preconditions, raises,
blocking, ownership, units and effects. A module that implements a public
one points to it and documents only what differs. The representation of
an abstract type is commented at the type in the `.ml`. The rules on
sentences below apply as for a public module.

## Guide pages

Guide pages read like the Go Memory Model and Racket's Evaluation Model.
They are precise and use the terms the `.mli` files define. Each example
is a complete small program with a sentence that says why it is correct.
They explain behaviour from first principles. Comparisons with other
projects belong to benchmarks.

## Sentences

Write each sentence after a sentence in the references that does the same
job. Start from these:

- Go `sync`, a contract: "Lock locks m. If the lock is already in use, the
  calling goroutine blocks until the mutex is available."
- Plan 9 `lock(2)`, contracts that read as one paragraph: "Lock blocks
  until the lock has been obtained. Canlock is non–blocking."
- Rust `std::pin`, a term made precise: "in the sense that its address in
  memory cannot change".
- Rust `std::sync::atomic`, a term defined where it is first needed:
  "Here, accesses are *conflicting* if they affect overlapping regions of
  memory and at least one of them is a write."
- bytesrw, a value stated as what it is: "[pos] is the position reported
  for the error".
- OpenBSD `intro(2)`, a definition: "A process is a collection of one or
  more threads, plus the resources shared by those threads such as process
  ID, address space, ..."
- Java's `CountDownLatch`, a rule in the model's vocabulary under a fixed
  label: "Memory consistency effects: Until the count reaches zero, actions
  in a thread prior to calling countDown() happen-before actions following
  a successful return from a corresponding await() in another thread."

These constructions are never used:

| Construction | Write instead |
|---|---|
| Contrastive negation, "X, not Y" or "is no X: only Y" | What it is |
| Triples built for rhythm | The items there are |
| Colon reveals, "runs asynchronously: the call returns before ..." | Two sentences, or "in the sense that" |
| Punchy closers, "These calls order themselves." | The fact, or nothing |
| Interpretive "-ing" clauses, ", making the order explicit" | A sentence that states the effect |
| Chains of abstract nouns | Verbs |
| Em-dashes | Commas, parentheses or two sentences |
| "[why] being what [Lost] carries" | "where [why] is what [Lost] carries" |
| "is the caller's" | Who does what: "The program orders the two." |
| Simple past in "once" clauses, "once [d] reached [v]" | "once [d] has reached [v]" |
| Definitions by negation, "is no X", "makes no X", "of no X" | A positive statement; a plain "returns no value" is fine |
| "at once" for "in parallel" | "in parallel", "concurrently" or "immediately" |

Uniform rhythm, stacked qualifications and hedges also read as written by a
machine.

## Checks

Documentation is done when it passes four checks.

### Correctness review

A reviewer checks every claim against the `.ml` or `.c` that implements it
and gives each a verdict: holds, wrong or unverifiable. Raises and errors,
blocking and ordering, lifetimes, safety facts and statuses come first. A
claim nobody can check is removed.

The reviewer then compares the new text with the old. A rule the old text
stated and the new one dropped is a defect, unless it was mechanism a
caller cannot observe, which moves to an `.ml` comment. Each example in a
doc must compile against the current signatures.

### Style review

A reviewer who knows the references reads as a demanding editor:

- quotes each sentence that reads as machine-written, with a rewrite after
  a named reference sentence;
- measures each preamble and the median value doc, and says where the
  extra text of each value doc over 100 words belongs;
- reads the preamble and section preambles alone, top to bottom, writes
  the model in five sentences, and lists each place it lost the model;
- picks five values and checks that each can be understood from its doc
  and the section preamble above it.

### Fresh-reader test

A reader who has never seen the code or an earlier draft receives the
`.mli` files alone, the main module first, and the guide pages after. The
reader explains the model back, such as by tracing one call end to end.
The explanation is checked against the code. Every disagreement is a
defect, and so is every term the reader met before its definition. The fix
goes in the docs, or in the code where the code is wrong. The docs pass
when another fresh reader's explanation holds against the code.

### Mechanical checks

- The package's `@check` alias builds its `.mli` files with warnings as
  errors.
- The package's doc `runtest` alias runs the guide pages under mdx, so each
  code block compiles and runs. A block that needs hardware is preceded by
  `<!-- $MDX skip -->`.
- Every `{!...}` must resolve to a module, type, value or heading, and
  every Markdown link to a file and heading that exist.
- A commit that changes only documentation touches only comments. Its
  `.ml` and `.c` diff holds comment hunks alone, and its `.mli` diff
  changes no signature, type or value. Moving a `val` within its `.mli` is
  allowed.

## References

- Go: [`sync`](https://pkg.go.dev/sync),
  [`sync/atomic`](https://pkg.go.dev/sync/atomic),
  [`io`](https://pkg.go.dev/io) and
  [the Go Memory Model](https://go.dev/ref/mem).
- Plan 9: [`intro(2)`](https://9p.io/magic/man2html/2/intro) and
  [`lock(2)`](https://9p.io/magic/man2html/2/lock).
- OpenBSD: [`intro(2)`](https://man.openbsd.org/intro.2).
- Java:
  [`java.util.concurrent`](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/package-summary.html).
- SQLite: [`sqlite3_stmt`](https://www.sqlite.org/c3ref/stmt.html).
- Elixir: [`GenServer`](https://hexdocs.pm/elixir/GenServer.html).
- Racket:
  [the Evaluation Model](https://docs.racket-lang.org/reference/eval-model.html).
- Rust: [`std::pin`](https://doc.rust-lang.org/std/pin/) and
  [`std::sync::atomic`](https://doc.rust-lang.org/std/sync/atomic/).
- OCaml: [cmdliner](https://erratique.ch/software/cmdliner/doc/) and
  [bytesrw](https://erratique.ch/software/bytesrw/doc/).
