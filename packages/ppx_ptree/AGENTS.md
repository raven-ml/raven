# ppx_ptree

Derives a structure's `walk` (`Nx.Ptree.S`) from its type: the function a
hand-written structure would have.

- It is for users. No raven library depends on it, not even for its tests.
- A derived type's parameters are read by position, payload then set, as
  the hand-written structures of nx, kaun and vega are. It generates walks
  only: a description is at one set, so users make it where they use it.
- Walks are named by convention with no role check. Each named walk is
  constrained to its part's type, so OCaml reports a mismatch at the part.
- `test/diagnostics.t` holds the errors a user sees; each points at the
  user's own type.
