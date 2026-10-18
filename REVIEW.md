# REVIEW.md

How to review a change to raven. A package's own `REVIEW.md` adds questions
for changes under its path.

A review has two layers, in this order. The design review comes first: when
the design is wrong, the lines that implement it don't need reviewing yet.

## 1. Design

Read `doc/dev/design.md`. Step back from the diff, read what the code does,
case by case, then ask what the change should have been.

- **What the change is for.** State, in one sentence a user would care
  about, the behaviour the change gives them. A change without one adds
  code and nothing a user gets.
- **Whether it needs fixing at all.** Should raven promise the behaviour
  the change repairs? A case that only a misuse, an impossible input or an
  event that ends the program reaches may need a stated limit or an error
  at the call, or nothing. Whether a library stays consistent after an
  exception raised at any allocation, such as Ctrl-C's, is a question of
  that kind. Every promise
  binds every later change, so the alternative to such a fix is dropping
  or narrowing the promise, and the finding names the code that goes with
  it.
- **Symptom or cause.** Does the change handle a symptom whose cause could be
  removed? A guard, a special case, a retry, a flag, a second path, a pass
  that repairs what an earlier one produced: each is a sign. Name the cause,
  the change that removes it, and the code that removal deletes.
- **A design that wouldn't need this.** Propose at least one shape the change
  did not consider, and say what it removes. Judge the change against it,
  never only against the alternatives its author weighed.
- **The layer.** Is the change in the layer that owns the question? A
  function passed down so a lower layer can call back up means each layer
  answers the other's question. A workaround in a consumer for a gap in a
  library below belongs in that library.
- **The second fix.** Run `git log` on the mechanism. Earlier fixes of the
  same class mean the mechanism is in the wrong place or has the wrong
  shape, and the finding is its redesign.
- **Costs as arguments.** Speed, allocation and diff size justify nothing
  without a measurement on the real path. Correctness decides first.
- **Concepts.** Does the change add a module type, functor, mode, optional
  argument, cache or mutable field that plain data, an existing noun or
  composition would cover? Is its size in proportion to its job?
- **Public `.mli`.** Is this the shape a careful library author would defend
  in ten years? Does it remove a case that used to work?
- **Hard constraints** of `doc/dev/design.md`.

A design finding gives the cause or the better shape, what it removes, and
what it costs. Its verdict is "redesign before landing" or "lands as is";
a redesign is never filed for later.

## 2. Semantics

Does each promise hold, and is it the right promise?

- **List the promises** the change makes or relies on: values, bounds,
  rounding, ordering, failures, state across calls.
- **Find the reference that fixes each one's meaning**: the one its `.mli`
  cites, else a specification (IEEE 754, a file format, a vendor's
  interface) or the paper that states the algorithm. Read it; never recall
  it. Give its title, section and a quote of at most two sentences. An
  implementation without a written design (tinygrad, NumPy, PyTorch,
  another library's code) is a datapoint.
- **Compare** the `.mli` with the reference, then the code with the `.mli`,
  case by case and at the edges: zero-size and one-element shapes, every
  dtype, strided and broadcast views, NaN, `-0.`, infinities, int extremes.
  Where no reference exists, argue from first principles and say so.
- **Tests** take expected values from the reference. A value learned by
  running the code is a baseline, written as `expect`. A test states a
  promise through the public interface and fails without the change.

Bug classes that recur in raven's fixes:

- **Floats across systems.** Printing, parsing and rounding go through
  raven's own correctly rounded code. libc's `printf`, `strtod` and libm
  differ on Windows and between C libraries. On arm64, ocamlopt fuses
  `a +. b *. c` into one `fmadd`, so an OCaml reference that must match C
  bit for bit wraps the product in `Sys.opaque_identity`.
- **Files.** A file is created exclusively, written through the descriptor
  that made it, and synced before its rename.
- **Hot paths.** A path the `.mli` promises allocates nothing takes no
  optional argument its callers supply: under the dev profile (`-opaque`)
  each supplied `?x` allocates a `Some`.
- **C.** It builds with GCC `-Werror` on Linux as well as clang on macOS, and
  reads no byte past a buffer's end.

## What Important means

A finding is Important when, once landed, a user would get a wrong number
with no error, a result that differs between platforms or between eager and
compiled runs beyond what the `.mli` allows, a crash, hang, leak or lost
data, or work done behind their back. A design finding is Important when
the change repairs a behaviour raven should not promise, treats a symptom
whose cause can be removed, or sits in the wrong layer.

Tag each finding Important, Nit or Pre-existing. A problem the change did
not introduce is Pre-existing, whatever its severity.

## Evidence

- Every finding gives `file:line` and either a concrete failure (inputs and
  the wrong output or the crash), the sentence of the reference or `.mli` it
  breaks, or for design, the alternative with what it removes.
- Read the code. A claim from a name, a comment or a commit message is a
  guess. Check every claim about other code against the tree.
- When a bug follows a pattern, search the tree for it and list every
  instance in one finding.

## Do not report

- Formatting that ocamlformat decides.
- Generated files: goldens, `*.expected`, `*.thumper` baselines, generator
  output. Check only that the generator made them.
- `_ref/`, `_tinygrad*/`, `_plans/`.
- A value with no caller as dead code. Find why it exists first.
- Removing a capability, or sharing code between rig's drivers.
- A smaller diff as a reason.

## Nits

At most three, and only for a rule written in an `AGENTS.md` or
`doc/dev/documentation.md`.

## Summary

Start with the counts per layer: "Design: 1 finding (Important). Semantics:
2 findings (1 Important, 1 Nit)." With none, write "No findings" and stop.
Then the findings, design first, most severe first within each layer. Don't
restate what the change does.

## Rounds

A re-review reads only what changed since the last round and never repeats a
finding already posted. After three rounds on one change, return to the
design layer.
