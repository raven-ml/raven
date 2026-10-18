# Reviewing fehu

## Design

- A wrapper or buffer for one environment or one algorithm.

## Semantics

References: Schulman et al., "High-Dimensional Continuous Control Using
Generalized Advantage Estimation" (2016) for GAE; Gymnasium's documented
`reset`/`step` contract as a datapoint.

Failure here is a learning signal silently wrong: termination treated as
truncation or the reverse, a value bootstrapped across an episode's end,
vectorised environments resetting out of step.

- Termination and truncation through GAE, returns and buffers.
- An episode of one step; a batch where every environment ends at once.
- Seeds: the same seed gives the same trajectory.
