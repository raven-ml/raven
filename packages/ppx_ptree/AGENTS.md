# ppx_ptree

Derives a structure's `walk` (`Nx.Ptree.S`) from its type: the function a
hand-written structure would have.

- It is for users. No raven library depends on it, not even for its tests.
- `test/diagnostics.t` holds the errors a user sees; each points at the
  user's own type.
