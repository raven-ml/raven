# Contrib

Packages built on Raven's core libraries that sit outside the 1.0 commitment.
Each one is its own dune project with its own version and opam metadata.
`norn`, `fehu`, and `sowilo` keep their own `CHANGES.md`.

| Package                   | What it does                                    |
| ------------------------- | ----------------------------------------------- |
| [**norn**](norn/)         | MCMC sampling with automatic gradients          |
| [**fehu**](fehu/)         | Reinforcement learning environments             |
| [**sowilo**](sowilo/)     | Differentiable computer vision                  |
| [**nx-oxcaml**](nx-oxcaml/) | Experimental Nx backend on OxCaml unboxed types |

## Policy

- Contrib packages build and test against `main` in CI. A change to a core
  library that breaks a contrib package fixes it in the same commit.
- They carry no API stability guarantee and release on their own schedule.
  `opam install raven` does not install them.
- They depend only on the public libraries of core packages, with a lower
  bound on the oldest core release they build against.
- Each package names its maintainers in its `dune-project`.

`nx-oxcaml` needs an OxCaml switch, so the main workspace treats it as data
and it builds from its own directory. See its README.
