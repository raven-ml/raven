# RFCs

An RFC records a design decision that is expensive to unwind: a change to a
public API or a backend contract, a new subsystem, or a choice that several
packages must agree on. Its job is to make the decision explicit and to kill
the alternatives in public, so the question is not reopened every time
someone reads the code. Reversible or invisible changes need no RFC; a
one-page proposal in a pull request is enough for most things.

## Lifecycle

`ideation` (a number is reserved, the answer is not known yet),
`discussion` (a draft is under review), `published` (the design is
accepted; nothing is implied about priority or who implements it),
`committed` (the document describes how the system works, not an
intention), `abandoned`. Once published, prefer a new RFC over rewriting an
old one; amendments are for corrections. An RFC whose design has not shipped
in a release may be replaced in place: the header gains a `Revision` item,
and the Rationale records what the earlier revision chose and why it lost.

## Structure

Each RFC is `NNNN-short-name.md`, numbered in order of reservation, with
this header and these sections. Write it so a reader who knows the packages
but did not watch the design finishes it in one sitting, and make it
self-contained: state a constraint where it is used rather than pointing at
a document outside the repository.

- **Header**: number and title, then `Status`, `Date`, `Packages`, once it
  lands `Implementation` (the pull request), and `Revision` when the text
  replaces an earlier one.
- **Summary**: one paragraph.
- **Motivation**: the problem, with evidence from the tree, and what happens
  if nothing is done.
- **Guide**: the design explained as if it had shipped, through the code a
  user writes. Written before the mechanism; a surface that needs a paragraph
  of explanation per example is wrong.
- **Reference**: the mechanism, its contracts and corner cases, by example.
- **Laws**: numbered invariants, each naming the failure it prevents.
- **Drawbacks**: what this costs, with numbers where they exist.
- **Rationale and alternatives**: every alternative considered and why it
  loses, the strongest one first. Precedent in other systems is context, not
  motivation.
- **Non-goals**: things that could reasonably have been goals and were
  declined.
- **Unresolved questions**: tiered by when each resolves; empty or answered
  once the RFC is committed.
- **Future possibilities**: with the rule that nothing listed here is a
  reason to accept this or a later RFC.

[0001](0001-tensors-as-values.md) is the reference example.

## Index

- [0001](0001-tensors-as-values.md) Tensors as values (committed)
- [0002](0002-decode-contract.md) The decode contract (committed)
- [0003](0003-loading-weights.md) Loading weights (committed)
- [0004](0004-quantised-weights.md) Quantised weights (published)
- [0005](0005-placement.md) Placement (discussion)
- [0006](0006-structures.md) Structures and compiled signatures (discussion)
