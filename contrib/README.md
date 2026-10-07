# Contrib

Packages built on Raven's core libraries that sit outside the 1.0 commitment.
Each one is its own dune project with its own version and opam metadata.
`fehu`, `sowilo` and `ymir` keep their own `CHANGES.md`.

| Package                   | What it does                                          |
| ------------------------- | ----------------------------------------------------- |
| [**fehu**](fehu/)         | Reinforcement learning environments                   |
| [**sowilo**](sowilo/)     | Differentiable computer vision                        |
| [**ymir**](ymir/)         | Astronomy: units, frames, FITS, photometry, cosmology |

## Policy

- Contrib packages build and test against `main` in CI. A change to a core
  library that breaks a contrib package fixes it in the same commit.
- They carry no API stability guarantee and release on their own schedule.
  `opam install raven` does not install them.
- They depend only on the public libraries of core packages, with a lower
  bound on the oldest core release they build against.
- Each package names its maintainers in its `dune-project`.
