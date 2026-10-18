# jera

Numerical methods: linear and nonlinear systems, minimisation, quadrature,
interpolation, and ordinary, stiff, differential-algebraic, delay and
stochastic differential equations. Every solve meets a stated tolerance and
reports a status per lane.

- jera depends on rune to state derivatives and to stage its loops. It never
  calls `jit`.
- Goldens come from mpmath and scipy:
  `uv run packages/jera/test/gen/generate.py`. It also generates
  `lib/sobol.ml`.
