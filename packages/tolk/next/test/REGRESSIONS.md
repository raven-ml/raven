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
| tinygrad: null/test_helpers.py::TestContextVars::test_initial_value_is_set | a setting starts from its default | Helpers › Context_var › an int setting starts from its default when its variable is unset |
| tinygrad: null/test_helpers.py::TestContextVars::test_cannot_recreate | a name is declared once | Helpers › Context_var › a name is declared once, whatever the setting's type |
| tinygrad: null/test_helpers.py::TestContextVars::test_new_var_inside_context | a declaration inside a context outlives it | Helpers › Context_var › a setting declared inside a context stays declared after it |
| tinygrad: null/test_helpers.py::TestContextVars::test_value_across_modules | the registry is one per process, whatever module declares | Helpers › Context_var › a library setting's name cannot be declared again |
| tinygrad: null/test_helpers.py::TestContextVars::test_assignment_across_modules | assigning `.value` directly | dropped: a setting has no assignment; only `context` changes its value |
| tinygrad: null/test_helpers.py::TestContextVars::test_context_assignment | a context binds for its extent | Helpers › context › binds a setting for the extent of its function |
| tinygrad: null/test_helpers.py::TestContextVars::test_unknown_param_to_context | `Context` refuses an unknown name | dropped: a binding holds the setting itself, so an unknown setting cannot be named |
| tinygrad: null/test_helpers.py::TestContextVars::test_nested_context | nested contexts restore in order | Helpers › context › restores nested bindings in order; Helpers › context › binds within its extent and restores after it, returned or raised |
| tinygrad: null/test_helpers.py::TestContextVars::test_decorator | `Context` as a decorator | Helpers › context › binds a setting for the extent of its function (`context` takes the function) |
| tinygrad: null/test_helpers.py::TestContextVars::test_decorator_recursive | recursive contexts restore | Helpers › context › restores recursive bindings |
| tinygrad: null/test_helpers.py::TestContextVars::test_context_exit_reverts_updated_values | exit restores the value before entry, not the initial one | Helpers › context › restores nested bindings in order |
| tinygrad: null/test_helpers.py::TestAllSame (4 tests) | `all_same` on empty, one, equal, unequal | Helpers › integers › all_same is true iff every element equals the first |
| tinygrad: null/test_helpers.py::TestMergeDicts::test_merge_dicts | `merge_dicts` | dropped: not ported, `Map.union` (README exclusions) |
| tinygrad: null/test_helpers.py::TestStripParens (7 tests) | `strip_parens` on nested, casted, unmatched and one-sided parentheses | Helpers › terminal text › strip_parens strips as tinygrad does (`text.golden`, every input of the class) |
| tinygrad: null/test_helpers.py::TestProd::test_empty | `prod ()` is 1 | Helpers › integers › prod of nothing is 1 |
| tinygrad: null/test_helpers.py::TestProd::test_ints | `prod` multiplies | Helpers › integers › prod multiplies; Helpers › integers › prod maps concatenation to multiplication |
| tinygrad: null/test_helpers.py::TestProd::test_variable, test_variable_order | `prod` over symbolic integers | dropped: `prod` is on `int`; the product of symbolic integers belongs to `Uop.Ops` (README exclusions) |
| tinygrad: null/test_helpers.py::TestRoundUp::test_round_up | `round_up` on negative and positive numbers | Helpers › integers › round_up keeps a multiple and rounds up a negative number; Helpers › integers › divide as tinygrad does |
| tinygrad: null/test_helpers.py::TestCeilDiv::test_int | `ceildiv` on integers | Helpers › integers › divide as tinygrad does (`division.golden`) |
| tinygrad: null/test_helpers.py::TestCeilDiv::test_symbolic, test_symbolic_negative_offset | `ceildiv` on UOps | dropped: `ceildiv` is on `int`; its UOp form belongs to `Uop.Ops`' suite |
| tinygrad: null/test_helpers.py::TestCount (2 tests) | `count` and its pickling | dropped: not ported, a counter reference (README exclusions) |
| tinygrad: null/test_helpers.py::TestFetch (9 tests) | `fetch` | dropped: url fetch is not ported (plan, L0 scope) |
| tinygrad: null/test_helpers.py::TestFullyFlatten (2 tests) | `fully_flatten` | dropped: not ported, it serves Python's typing (README exclusions) |
| tinygrad: null/test_helpers.py::TestMemoryview (3 tests) | `from_mv`, `to_mv`, `mv_address` | dropped: the ctypes helpers are not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestGetShape (2 tests) | `get_shape` | dropped: not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestPolyN (2 tests) | `polyN` | dropped: it lives with the symbolic integer type in `Uop.Ops` (README exclusions) |
| tinygrad: null/test_helpers.py::TestTimeToStr (7 tests) | `time_to_str` units, boundaries and width | Helpers › terminal text › time_to_str writes a duration as tinygrad does (`durations.golden`, every input of the class) |
| tinygrad: null/test_helpers.py::TestCStyleDivMod (4 tests) | `cdiv` and `cmod` by positive and negative divisors | Helpers › integers › divide as tinygrad does; Helpers › integers › cdiv and cmod are OCaml's truncated division |
| tinygrad: null/test_helpers.py::TestGetBits (5 tests) | `getbits` | dropped: not ported, no reader (README exclusions) |
| tinygrad: null/test_helpers.py::TestArgFix (4 tests) | `argfix` | dropped: not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestWordWrap (4 tests) | `word_wrap` | dropped: not ported, no reader (README exclusions) |
| tinygrad: null/test_helpers.py::TestIsNumpyNdarray (4 tests) | `is_numpy_ndarray` | dropped: not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestDisableGC::test_recursive_decorator | `disable_gc` | dropped: not ported (README exclusions) |
| tinygrad: null/test_disk_cache.py::DiskCache::test_putget | put, get, put again | Helpers › Diskcache › put replaces the value of a key; Helpers › Diskcache › behaves as a table of entries per table |
| tinygrad: null/test_disk_cache.py::DiskCache::test_putcomplex | a structured value reads back | Helpers › Diskcache › get reads back any key and value put (values are strings, D8) |
| tinygrad: null/test_disk_cache.py::DiskCache::test_getotherprocess | another process reads an entry | Helpers › Diskcache › another process reads an entry put back |
| tinygrad: null/test_disk_cache.py::DiskCache::test_putotherprocess | an entry put by another process | Helpers › Diskcache › an entry put by another process reads back |
| tinygrad: null/test_disk_cache.py::DiskCache::test_no_table | a table never written | Helpers › Diskcache › get is None for a table never written |
| tinygrad: null/test_disk_cache.py::DiskCache::test_ret | `diskcache_put` returns its value | dropped: `put` returns `unit`; the returned value served the `diskcache` decorator, which is not ported |
| tinygrad: null/test_disk_cache.py::DiskCache::test_non_str_key | an integer key equals its string | dropped: keys are strings (D8) |
| tinygrad: null/test_disk_cache.py::DiskCache::test_decorator | the `diskcache` decorator | dropped: not ported (README exclusions) |
| tinygrad: null/test_disk_cache.py::DiskCache::test_dict_key | keys of several columns | dropped: keys are strings, which callers encode (D8) |
| tinygrad: null/test_disk_cache.py::DiskCache::test_table_name | a table name with `:` and `-` | Helpers › Diskcache › behaves as a table of entries per table (table `test_gfx1010:xnack-`) |
| tinygrad: null/test_disk_cache.py::DiskCache::test_clear_cache | clear empties every table, and runs again | Helpers › Diskcache › clear removes the entries of every table |
| tinygrad: null/test_device.py::TestDevVar::test_parse | DEV parses targets and prints them back | Helpers › Target › parse reads a target as tinygrad does (`targets.golden`, every input of the test); Helpers › startup › reads DEV as targets separated by semicolons |
| tinygrad: null/test_device.py::TestDevVar::test_target | the target of a device under DEV | Helpers › Target › target picks a device's target as tinygrad does (`device_targets.golden`, every DEV of the test) |
| tinygrad: null/test_device.py::TestDevVar::test_dev_arch_override | an arch in DEV reaches the renderer | dropped: the renderer belongs to `Device`'s suite; `target`'s arch is pinned by `device_targets.golden` |
| tinygrad: null/test_device.py::TestDevice::test_nonexistent_renderer | "did you mean: 'CLANG'" | Helpers › selection › select_by_name selects as tinygrad does (`candidates=CLANG,LLVM query=CLANGJIT`); the renderer lookup belongs to `Device`'s suite |
| tinygrad: null/test_device.py::TestCompiler (3 tests) | the compiler cache follows CCACHE | dropped: `Compiler` belongs to `Device`'s suite |
| tinygrad: null/test_hashing.py::TestKeccak::test_shape_keeping | `Tensor.keccak` | dropped: the `Tensor` surface; the frontend is nx |
| tinygrad: null/test_tqdm.py (all) | `tqdm` | dropped: not ported (plan, L0 scope) |
| old: unit/test_helpers.ml:19 time_to_str picks the unit above ten of the next | units and widths | Helpers › terminal text › time_to_str writes a duration as tinygrad does (`durations.golden` holds each input) |
| old: unit/test_helpers.ml:26 size_to_str | bytes, KB, MB, GB | Helpers › terminal text › size_to_str writes a size as tinygrad does |
| old: unit/test_helpers.ml:33 colored | foreground, bright, background | Helpers › terminal text › colored paints as tinygrad does; the `None` color is dropped: a color is a variant |
| old: unit/test_helpers.ml:40 ansilen ignores escape sequences | `ansilen` of colored text and `ESC [K` | Helpers › terminal text › ansilen counts as tinygrad does; Helpers › terminal text › ansipad does not count escape sequences |
| old: unit/test_helpers.ml:49 concurrent allocations preserve live byte counts | memory counters under concurrent domains | dropped: `GlobalCounters` are read where kernels run and buffers are allocated, which is rune's (README Exclusions, D3) |
| old: unit/test_helpers.ml:68 mem_used follows allocation and release | memory used rises and falls | dropped: `GlobalCounters` are read where kernels run and buffers are allocated, which is rune's (README Exclusions, D3) |
| old: unit/test_helpers.ml:78 reset leaves mem_used alone | reset keeps the memory used | dropped: `GlobalCounters` are read where kernels run and buffers are allocated, which is rune's (README Exclusions, D3) |
| old: unit/test_helpers.ml:189 nested duplicate overrides restore after an exception | duplicate bindings, raise, restore | Helpers › context › gives a setting bound twice its later binding, and restores the first value; Helpers › context › restores a setting when its function raises |
| old: unit/test_helpers.ml:197 overlapping domains retain their own contexts | domain-local overrides | dropped: overrides are process-global, as in tinygrad; the new contract is Helpers › context › binds for every domain while it runs |
| old: unit/test_helpers.ml:199 overlapping systhreads retain their own contexts | thread-local overrides | dropped: as the row above |
| old: unit/test_helpers.ml:201 snapshots are immutable and replace a worker's current context | context snapshots for workers | dropped: global overrides reach workers without a snapshot |
| old: unit/test_helpers.ml:202 exited scopes release their values | no leak of bound values | Helpers › context › keeps no bound value once it returns |
| old: unit/test_helpers.ml:206 unlimited and malformed quota retain available CPUs | cgroup quota parsing | dropped: the quota parser is private to `parallel`'s default and reads a Linux file; Helpers › settings › parallel is between one and the domains the runtime recommends |
| old: unit/test_helpers.ml:211 quota bounds workers by whole available CPUs | cgroup quota bound | dropped: as the row above |
| old: unit/test_helpers.ml:222 target strings preserve architecture and interface spelling | case kept in arch and interface | Helpers › Target › parse reads a target as tinygrad does (`input=remote:host:2+nv:cuda:sm_89`) |
| old: unit/test_helpers.ml:230 target strings normalize empty fields without inventing defaults | empty fields print as nothing | Helpers › Target › parse reads a target as tinygrad does (the six inputs of the test); Helpers › Target › parse reads back what to_string writes |
| old: unit/test_helpers.ml:236 target strings reject excess separators | two `+`, three `:` | Helpers › Target › parse reads a target as tinygrad does (`input=PCI+NV+CUDA`, `input=CPU:CLANG:arm64:extra`) |
| old: unit/test_helpers.ml:240 per-backend targets and defaults do not leak between contexts | per-device targets, arch default, restore | Helpers › Target › target picks a device's target as tinygrad does; Helpers › Target › target keeps a target's interface and indices; Helpers › context › restores a setting when its function raises. The device name with an index (`NV:1`) is dropped: `target` takes a device name (D6) |
| old: unit/test_helpers.ml:250 the first matching wildcard supplies the target | first wildcard wins | Helpers › Target › target picks a device's target as tinygrad does (`dev=PCI+;NV:CUDA device=NV`) |
| old: unit/test_helpers.ml:253 DEV selects one interface without trying alternatives | interface selection | dropped: interface selection belongs to the device layer; `select_by_name` is pinned by `selection.golden` |
| old: unit/test_helpers.ml:259 an unknown interface fails before initialization | unknown interface | dropped: as the row above |
| old: unit/test_helpers.ml:263 mock interfaces require explicit selection | MOCK interfaces | dropped: raven has no mock drivers (README exclusions) |
| old: unit/test_helpers.ml:267 legacy target settings point to DEV | `NV_IFACE`, `NV_CC` | dropped: the `{DEV}_CC` migration check is not ported (README exclusions) |
| old: unit/test_helpers.ml:275 prefers the kernel driver without initializing PCI | first success stops | Helpers › selection › select_first_inited takes the first candidate and tries no other |
| old: unit/test_helpers.ml:280 falls back after driver initialization fails | fallback | Helpers › selection › select_first_inited tries the next candidate after a failure |
| old: unit/test_helpers.ml:286 preserves the error from an explicitly selected interface | a single candidate's error | Helpers › selection › select_first_inited reports the only candidate's error as it is |
| old: unit/test_helpers.ml:289 reports failures from every attempted interface | every error, one per line | Helpers › selection › select_first_inited reports every candidate's error, one per line |
| old: unit/test_helpers.ml:297 does not retry after the selected runtime fails | a later failure of the selected value | Helpers › selection › select_first_inited takes the first candidate and tries no other |
| old: unit/test_helpers.ml:304 does not treat cancellation as an unavailable driver | exceptions escape | Helpers › selection › select_first_inited lets a candidate's exception escape, trying no other |
| old: unit/test_diskcache.ml:319 compiler cache policy follows nested contexts and worker snapshots | the compiler cache under CCACHE and CACHELEVEL | dropped: `Compiler` belongs to `Device`'s suite; the CACHELEVEL half is Helpers › Diskcache › a disabled cache neither reads nor writes |
| old: unit/test_diskcache.ml:322 assert compile permits hits and blocks every cache miss | ASSERT_COMPILE | dropped: `Compiler` belongs to `Device`'s suite |
| old: unit/test_diskcache.ml:325 platform detection does not require shell tools | host detection without PATH | dropped: tolk.next reads the platform from the build configuration, not at run time |
| old: unit/test_diskcache.ml:327 cache location follows the platform, not directory contents | cache directory per platform | Helpers › startup › puts the cache in the platform's cache directory without XDG_CACHE_HOME |
| old: unit/test_diskcache.ml:329 round-trips a value across processes | cross-process persistence | Helpers › Diskcache › an entry put by another process reads back; Helpers › Diskcache › another process reads an entry put back |
| old: unit/test_diskcache.ml:331 missing key is a miss | missing key | Helpers › Diskcache › get is None for a table never written; Helpers › Diskcache › behaves as a table of entries per table |
| old: unit/test_diskcache.ml:332 corrupt or truncated entry is a miss | damaged entries | Helpers › Diskcache › get fails on a truncated entry; Helpers › Diskcache › get fails on an entry that is no entry. The contract changed: a malformed entry raises `Failure`, as tinygrad's unpickling raises |
| old: unit/test_diskcache.ml:334 concurrent writers never tear an entry | atomic writes | Helpers › Diskcache › writers in several processes never tear an entry; Helpers › Diskcache › writers on several domains never tear an entry |

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
