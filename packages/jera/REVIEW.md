# Reviewing jera

## Design

- A tolerance loosened, an iteration cap raised or a case special-cased so a
  problem passes: is the method's stopping test or step control wrong?
- A fix for one method that its siblings need too (Newton in roots, systems
  and minimisation; event location across ODE methods).

## Semantics

References: the paper or book the `.mli` cites for each method. The mpmath
and scipy goldens are values to compare with; the method's statement
decides what jera computes.

Failure here is a status that reports convergence when the tolerance is not
met, or failure when it is. Most of jera's fixes are at the arithmetic's
limit: a Newton step the floats cannot resolve, a line search with no
decrease, a contraction measured over steps that round to nothing.

- The stopping test against the stated tolerance, at a tolerance near
  machine epsilon.
- One lane's failure leaving the other lanes' results and statuses alone.
- Events: a crossing located to the time's resolution, a sign carried
  across zero steps.
- Delays: history called at the right time per lag.

## Skip

`test/golden/`, `lib/sobol.ml`, `bench/jera.thumper`.
