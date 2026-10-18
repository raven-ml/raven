# fehu

Reinforcement-learning environments over nx: environments with `reset` and
`step` over typed spaces, wrappers, trajectory collection, replay buffers,
advantage estimation and vectorised environments.

- fehu depends on nx only. Users apply rune to their policies.
- Changes go in `contrib/fehu/CHANGES.md`.
- An episode ends terminated or truncated, and the two differ: a truncated
  episode bootstraps from its last value.
