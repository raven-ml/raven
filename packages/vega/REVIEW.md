# Reviewing vega

## Design

- State shaped for one optimizer instead of following the parameters'
  structure.
- An optimizer that could be a composition of existing transformations.

## Semantics

References: the paper each `.mli` cites for its optimizer or schedule.
Check the update against the paper's algorithm statement: bias correction,
where weight decay applies, epsilon's place, default constants.

- Schedules at step 0, 1 and the last step.
- Leaves that are neither floats nor complex pass through untouched; each
  part of a complex leaf steps as a real leaf of the complex leaf's shape,
  with its own state, and a leaf's norms sum both parts.
