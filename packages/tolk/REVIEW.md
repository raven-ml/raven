# Reviewing tolk

## Design

tolk is a port, so the design questions are tinygrad's:

1. Is this the same owner as in tinygrad?
2. Is it implemented in the same place?
3. Is it as simple as tinygrad, or simpler for an OCaml reason?

And:

- A fix in the renderer for what a pass produced wrong. When source
  differs, the owner is found in this order: `opt/` or a late pass, the
  linearizer, the generic renderer.
- A special case that makes one kernel's source match.
- A new IR, pass or layer: can the current one not hold the semantics?
- A structural departure from tinygrad: its `DIVERGENCES.md` entry, reason
  and test.

## Semantics

References: nx's documented meaning and IEEE 754-2019 for values; the
target's specification for what rendered code means (the C standard, the
Metal Shading Language, PTX, the AMD ISA). tinygrad at the pinned commit is
a datapoint for values.

Failure here is a silent miscompile: a kernel that computes the wrong values
with no error. Then a source diff from the golden, and a stale program
served from a cache whose key misses an input.

- A rewrite rule's guards: the gates of loads, a NaN or infinity reaching an
  integer, a division over padding, a conversion's guard kept through a
  selection.
- A bug found in one rule: grep for its pattern. Eleven miscompiles once
  came from one pattern.
- The dtype of every constant a rule makes.
- A kernel parameter's base address: Metal merges adjacent vector loads and
  drops address bits below their alignment, so a misaligned base reads
  wrong only when loads merge.
- Paths, environment and processes on Windows.

## Do not report

A golden diff as a test to update. Goldens are tinygrad's output.

## Skip

`test/golden/`, `test/gen/` output, `bench/*.graph`, `bench/tolk.thumper`,
`LICENSE-tinygrad`.
