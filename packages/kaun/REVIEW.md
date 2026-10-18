# Reviewing kaun

## Design

- A layer that holds state, calls rune, or exists for one model.
- A model-specific helper: does it compose from layers in the example
  instead?

## Semantics

References: the paper each `.mli` cites for its layer, initializer or loss.
Check the formula, its default constants and where its normalisation or
scaling applies.

- Narrow dtypes: which steps compute at `float32`.
- Masks and padding in attention, positions in rotary embeddings, a decode
  cache's index at its first and last row.
