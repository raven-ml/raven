# Regressions

Each module's section maps every behaviour its tests must keep to the test
that keeps it, or to the reason it is dropped. The sources are:

- every behaviour an old tolk test pins (`packages/tolk/test`);
- every relevant case of tinygrad's tests for the module's file.

A row names its source as `old: <file>:<line> <test name>` or
`tinygrad: <file>::<class>::<test>`, and its outcome as the new test's path
(`<suite> › <group> › <test>`) or as `dropped: <reason>`. A module's section
starts when its test pass does.

## Helpers

| Source | Behaviour | Outcome |
|---|---|---|

## Dtype

| Source | Behaviour | Outcome |
|---|---|---|

## Uop.Op

An old test's line is the line of the call its assertion checks.

| Source | Behaviour | Outcome |
|---|---|---|
| old: `unit/uop/test_ops.ml:123` op order matches vendored tinygrad | the operations and their order are tinygrad's | `Tolk_next.Uop.Op › Op › declares tinygrad's operations in tinygrad's order`, `Tolk_next.Uop.Op › Set.to_list › of all is the declared operations`; the golden replaces running Python at test time |
| old: `unit/uop/test_ops.ml:64` op order matches vendored tinygrad | BACKEDGE spliced in from a second checkout ahead of the pin move | dropped: tinygrad HEAD declares BACKEDGE itself, and the golden records it |
| old: `unit/uop/test_ops.ml:146` GroupOp memberships match vendored tinygrad | each group's members are tinygrad's | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members` (one case per tinygrad operation, checking every group column) |
| old: `unit/uop/test_ops.ml:154` predicates match public groups | `is_unary` and the other predicates agree with the group lists | dropped: the interface has no per-group predicates; `Set.mem` is the membership test, pinned by the `Tolk_next.Uop.Op › Set.mem` group |
| old: `unit/uop/test_uop.ml:98` Ops and dtype access | `Add` is an ALU operation | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ADD` |
| old: `unit/uop/test_uop.ml:184` Ops tinygrad order | the full hand-written order, with WAIT, REWRITE_ERROR and PYLITERAL | `Tolk_next.Uop.Op › Op › declares tinygrad's operations in tinygrad's order`; WAIT is gone from tinygrad HEAD; REWRITE_ERROR and PYLITERAL: `Tolk_next.Uop.Op › Op › has no counterpart for exactly REWRITE_ERROR and PYLITERAL` |
| old: `unit/uop/test_uop.ml:192-193` Ops.Group algebra | `mem` finds `Add` in binary and rejects `Range` | `Tolk_next.Uop.Op › Set.mem › of of_list is membership of the list`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ADD`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.RANGE` |
| old: `unit/uop/test_uop.ml:194` Ops.Group algebra | `union` merges two lists without duplicates | `Tolk_next.Uop.Op › Set.mem › of union is membership of either`, `Tolk_next.Uop.Op › Set.to_list › of of_list is the listed operations in declaration order, once each`; the order changes from first occurrence to declaration order, since a group is a set |
| old: `unit/uop/test_uop.ml:198` Ops.Group algebra | `without binary comparison` removes the comparisons | `Tolk_next.Uop.Op › Set.mem › of diff is membership of the first and not the second`, `Tolk_next.Uop.Op › Set.diff › removes Threefry from alu and keeps the rest` |
| old: `unit/uop/test_uop.ml:219` Ops.Group algebra | reduce is Add, Mul and Max | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ADD`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.MUL`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.MAX` |
| old: `unit/uop/test_uop.ml:221` Ops.Group algebra | defines holds Buffer, Alloc and Param | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.BUFFER`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ALLOC`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.PARAM` |
| old: `unit/uop/test_uop.ml:223` Ops.Group algebra | irreducible holds Param and Getaddr | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.PARAM`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.GETADDR` |
| old: `unit/uop/test_uop.ml:225` Ops.Group algebra | broadcastable leaves out Group | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.GROUP` |
| old: `unit/uop/test_uop.ml:227,229` Ops.Group algebra | Cdiv, Cmod, Floordiv and Floormod are binary | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.CDIV`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.CMOD`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.FLOORDIV`, `Tolk_next.Uop.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.FLOORMOD` |
| old: `unit/uop/test_uop.ml:230` Ops.Group algebra | predicates agree with the group lists | dropped: no predicates, as for `test_ops.ml:154` |
| tinygrad: `uop/__init__.py` (no test file targets it) | `str` and `repr` of an operation are `Ops.<NAME>` | `Tolk_next.Uop.Op › Op › pp is Ops. followed by the name` and the golden's `op` column |
| tinygrad: `uop/__init__.py` (no test file targets it) | the enum's integer values order the toposort | `Tolk_next.Uop.Op › Op order › compare agrees with tinygrad's integer order` |
| tinygrad: `uop/__init__.py` (no test file targets it) | `auto()` values are unique across every `FastEnum` subclass | dropped: OCaml variants of distinct types never compare equal; `X86Ops` belongs to the renderer's section |
| tinygrad: `null/test_upat_compile.py::TestUPatCompile::test_const_folding` | `GroupOp.ALU - {Ops.THREEFRY}` | `Tolk_next.Uop.Op › Set.diff › removes Threefry from alu and keeps the rest`; the pattern compilation itself belongs to `Uop.Upat` |
| tinygrad: `null/test_linearizer.py`, `runtime/test_linearizer.py`, `null/test_schedule.py`, `null/test_const_folding.py`, `null/test_dtype_weak.py`, `null/test_pattern_matcher.py` | `u.op in GroupOp.ALU` counts arithmetic | `Tolk_next.Uop.Op › Named sets › hold tinygrad's members` (the ALU column); each test belongs to its own module's section |
| tinygrad: `null/test_graph_rewrite.py`, `null/test_uop_graph.py` | `UPat(GroupOp.All)` matches every operation | `Tolk_next.Uop.Op › Set.mem › of all always holds`; each test belongs to its own module's section |
| tinygrad: `null/test_viz.py::TestViz::test_colored_label`, `test_colored_label_multiline`, `test_inf_loop`, `TestVizGC::test_gc_uop_in_arg` | PYLITERAL and REWRITE_ERROR nodes in the graph viewer | dropped: viz is excluded; `Tolk_next.Uop.Op › Op › has no counterpart for exactly REWRITE_ERROR and PYLITERAL` |
