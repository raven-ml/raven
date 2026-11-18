# Contrib

Packages built on Raven's core libraries that sit outside the 1.0 commitment.
Each one is its own dune project with its own version and opam metadata.
`fehu`, `sowilo`, `ymir` and `jord` keep their own `CHANGES.md`.

| Package                   | What it does                                          |
| ------------------------- | ----------------------------------------------------- |
| [**fehu**](fehu/)         | Reinforcement learning environments                   |
| [**sowilo**](sowilo/)     | Differentiable computer vision                        |
| [**ymir**](ymir/)         | Astronomy: units, frames, FITS, photometry, cosmology |
| [**jord**](jord/)         | Earth observation: systems, grids, warps, products    |

## Policy

- Contrib packages build and test against `main` in CI. A change to a core
  library that breaks a contrib package fixes it in the same commit.
- They carry no API stability guarantee and release on their own schedule.
  `opam install raven` does not install them.
- They depend on the public libraries of core packages, with a lower bound
  on the oldest core release they build against, and on other OCaml
  libraries only as `doc/dev/design.md` allows: where a library's type is
  in the package's interface (jord's `jsont`, whose `Jsont.json` its
  readers take), or by the maintainer's decision (jord's `xmlm`, which
  `jord.safe` reads Sentinel-2's XML with).
- Each package names its maintainers in its `dune-project`.
