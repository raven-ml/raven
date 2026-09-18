# Changelog

All notable changes to Norn are documented in this file.

## Unreleased

- Norn is a contrib package: it has its own version and changelog, and
  `opam install raven` does not install it.
- New package: Markov chain Monte Carlo sampling with automatic gradients via
  Rune. Provides HMC and NUTS samplers with Stan-style window adaptation (dual
  averaging for step size, Welford estimation for mass matrix). Includes
  symplectic integrators (leapfrog, mclachlan, yoshida), mass matrix metrics
  (unit, diagonal, dense), and convergence diagnostics (ESS, split R-hat).
  Equivalent to BlackJAX/PyMC in Python.
