# Regressions

Each module's section maps every behaviour its tests must keep to the test
that keeps it, or to the reason it is dropped. The sources are:

- every behaviour an old tolk test pins (`packages/tolk/test`);
- every relevant case of tinygrad's tests for the module's file.

A row names its source as `old: <file>:<line> <test name>`, `old: <file> <function>` (anchored by name, which survives edits to the file) or
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
| tinygrad: null/test_helpers.py::TestProd::test_variable, test_variable_order | `prod` over symbolic integers | dropped: `prod` is on `int`; the product of symbolic integers belongs to `Ops` (README exclusions) |
| tinygrad: null/test_helpers.py::TestRoundUp::test_round_up | `round_up` on negative and positive numbers | Helpers › integers › round_up keeps a multiple and rounds up a negative number; Helpers › integers › divide as tinygrad does |
| tinygrad: null/test_helpers.py::TestCeilDiv::test_int | `ceildiv` on integers | Helpers › integers › divide as tinygrad does (`division.golden`) |
| tinygrad: null/test_helpers.py::TestCeilDiv::test_symbolic, test_symbolic_negative_offset | `ceildiv` on UOps | dropped: `ceildiv` is on `int`; its UOp form belongs to `Ops`' suite |
| tinygrad: null/test_helpers.py::TestCount (2 tests) | `count` and its pickling | dropped: not ported, a counter reference (README exclusions) |
| tinygrad: null/test_helpers.py::TestFetch (9 tests) | `fetch` | dropped: url fetch is not ported (plan, L0 scope) |
| tinygrad: null/test_helpers.py::TestFullyFlatten (2 tests) | `fully_flatten` | dropped: not ported, it serves Python's typing (README exclusions) |
| tinygrad: null/test_helpers.py::TestMemoryview (3 tests) | `from_mv`, `to_mv`, `mv_address` | dropped: the ctypes helpers are not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestGetShape (2 tests) | `get_shape` | dropped: not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestPolyN (2 tests) | `polyN` | dropped: it lives with the symbolic integer type in `Ops` (README exclusions) |
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

The suite is `tolk.next.dtype` (`dtype/test_dtype.ml`), written `D` below. A
golden check is named after its golden, and each of its rows is a test keyed by
its input cells, such as `D › truncate › truncation.golden › dtype=dtypes.half value=65520.0`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_dtype.py::TestEqStrDType::test_strs | a data type prints as `dtypes.<alias>` | `D › data types › properties.golden` (column `dtype`) |
| tinygrad: null/test_dtype.py::TestToDtype::test_dtype_to_dtype | `to_dtype` returns a DType as is | dropped: `of_string` takes a string, and a `Dtype.t` needs no conversion |
| tinygrad: null/test_dtype.py::TestToDtype::test_str_to_dtype | a name reads as its data type | `D › names › names.golden`; `D › names › of_string reads what pp prints` |
| tinygrad: null/test_dtype.py::TestCastConvenienceMethod::test_method | `Tensor.half()` and friends | dropped: Tensor surface, rune's lowering |
| tinygrad: null/test_dtype.py::TestDtypeTolist::test_bfloat16 | bfloat16 rounds -60000, 1.5, 3.1, 60000 | `D › truncate › truncation.golden` (dtype=dtypes.bfloat16) |
| tinygrad: null/test_dtype.py::TestDtypeTolist::test_fp8 | 8-bit floats saturate ±30000 and round 3.1 | `D › truncate › truncation.golden` (dtype=dtypes.fp8e4m3, dtypes.fp8e5m2) |
| tinygrad: null/test_dtype.py::TestCanLosslessCast::test_can_lossless_cast | signed to unsigned is lossy, int8 to half lossless, int8 to bfloat16 lossy | `D › lossless casts › lossless_cast.golden` |
| tinygrad: null/test_dtype.py::TestInvalidSingleton::test_singleton, test_pickle | `Invalid` is one object, also through pickle | dropped: `` `Invalid `` is a constant constructor, one value by construction, and nothing is pickled |
| tinygrad: null/test_dtype.py::TestBitCast::test_shape_change_bitcast_exceptions | a Tensor bitcast needs a shape its item sizes divide | dropped: Tensor shapes. The scalar rule is `D › bitcast › a bitcast keeps the item size` |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_is_int, test_is_unsigned_uints, test_is_unsigned_signed_ints, test_is_float, test_bf16_is_float, test_fp8s_are_float | the predicates | `D › data types › properties.golden` (columns `is_*`) |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_from_py | the data type of a literal and of a list of literals | `D › literals › of_const.golden`, `D › literals › of_consts.golden` |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_from_py (None, {}, set()) | a value that is no literal raises | dropped: a `const` holds only literals, so the type checker rejects them |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_dtype_range, test_dtype_range_vec | `min` and `max` of every data type | `D › data types › properties.golden` (columns `min`, `max`) |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_float_to_fp16 | half rounds and overflows at 65520 | `D › truncate › truncation.golden` (dtype=dtypes.half) |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_float_to_bf16 | bfloat16 rounding of torch's cases, overflow to ±inf | `D › truncate › truncation.golden` (dtype=dtypes.bfloat16) |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_float_to_bf16_nan | NaN of any payload stays NaN | `D › truncate › truncation.golden` (value=nan, -nan); `D › truncate › bfloat16 rounding is odd` (draws NaN payloads) |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_float_to_bf16_round | ties to even, through float32 | `D › truncate › truncation.golden` (values 0x3F807000, 0x3F80C000, 0x3F808000, 0x3F818000, 0x41238000, 0xC1468000 as floats); `D › truncate › bfloat16 rounds once, to the nearest, ties to even`. By ruling, a float rounds once to bfloat16: the golden rows where the float32 step lands on a tie (1.0039062500000002, 1+2^-8+2^-40) are stated in code |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_float_to_bf16_boundary | rounding at the largest finite bfloat16 | `D › truncate › truncation.golden` (values 0x7F7F7FFF, 0x7F7F8000, 0x7F7FC000) |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_truncate_fp8e4m3, test_truncate_fp8e5m2 | 8-bit rounding matches torch, saturates, NaN on infinities | `D › truncate › truncation.golden`; `D › truncate › truncation is idempotent`; `D › truncate › a finite float stays finite in an 8-bit float`; `D › truncate › rounding to a float is monotone` |
| tinygrad: null/test_dtype_spec.py::TestHelpers::test_finfo | `finfo` of the floats | `D › data types › finfo.golden`; `D › data types › finfo rejects every data type but the floats of known width` |
| tinygrad: null/test_dtype_spec.py::TestTypePromotion::test_self_promo_to_self | a data type bounds itself | `D › promotion › a data type is its own least upper bound` |
| tinygrad: null/test_dtype_spec.py::TestTypePromotion::test_promo_resulted_higher_than_inputs | the bound is no lower than its inputs | `D › promotion › least_upper is at least each of its inputs` |
| tinygrad: null/test_dtype_spec.py::TestTypePromotion::test_dtype_promo, test_weakint_promo, test_weakfloat_promo | the promotion table | `D › promotion › least_upper.golden`; weakfloat outside `floats`: `D › data types › groups.golden` |
| tinygrad: null/test_dtype_spec.py::TestTypeSpec::test_set_dtype_default | DEFAULT_INT and DEFAULT_FLOAT select the defaults | `D › defaults › default_int is the integer DEFAULT_INT names`; `D › defaults › default_float is the float DEFAULT_FLOAT names` |
| tinygrad: null/test_dtype_spec.py::TestTypeSpec::test_bool_ops, test_functions_return_index, test_tensor_indexing_returns_same_dtype, test_gather_returns_same_dtype, test_attention_returns_same_dtype | result dtypes of Tensor operations | dropped: Tensor surface, rune's lowering |
| tinygrad: null/test_dtype_spec.py::TestAutoCastType::test_least_upper_float_input_is_float | a float keeps its type whatever the default | `D › promotion › least_upper_float keeps a float whatever the default` |
| tinygrad: null/test_dtype_spec.py::TestAutoCastType::test_least_upper_float_input_is_int | an integer goes to the default float | `D › promotion › least_upper_float takes an integer to the default float` |
| tinygrad: null/test_dtype_spec.py::TestAutoCastType::test_sum | the accumulator of each data type | `D › defaults › projections.golden` (column `sum_acc`) |
| tinygrad: null/test_dtype_spec.py::TestAutoCastType::test_broadcast_scalar, test_pad_scalar, test_sort, test_int_div_int, test_mean, test_cumsum, test_cumsum_empty, test_matmul, test_linear, test_where_no_scalar, test_where_one_scalar, test_where_two_scalars, test_where_non_bool_cond_raises, test_maximum, test_maximum_const, test_div, test_div_const | result dtypes of Tensor operations | dropped: Tensor surface, rune's lowering. The table they rely on is `D › promotion › least_upper.golden` |
| tinygrad: null/test_dtype_spec.py::TestUnitTypeSpec::test_default_dtype_context | a context overrides the defaults and restores them | `D › defaults › default_float is the float DEFAULT_FLOAT names`; the restoring is `Helpers.context`'s, tested in the Helpers suite |
| tinygrad: null/test_dtype_spec.py::TestUnitTypeSpec::test_env_set_default_float (skipped upstream) | DEFAULT_FLOAT=HALF selects half; INT32 and TYPO are rejected | `D › defaults › DEFAULT_FLOAT is read in any case`; `D › defaults › default_float rejects what is not a float of known width` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_float_unary_on_weakint_stays_weak | `least_upper_float weakint` is weakfloat | `D › defaults › projections.golden` (dtype=dtypes.weakint) |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion (every other test), TestWeakStorageBoundary, TestNoRedundantWide | how weak Tensors and UOps promote and commit | dropped: Tensor and UOp graphs, in the Uop_weak suite (L3) and rune's lowering |
| tinygrad: runtime/test_dtype_weak.py::TestWeakStorageBoundary::test_literal_beyond_any_int_raises | a literal no 64-bit integer holds raises | `D › literals › of_consts.golden` (consts=[18446744073709551616], [-9223372036854775809]); `D › literals › commit.golden` |
| tinygrad: runtime/test_dtype_weak.py (every other test) | weak values on a device | dropped: execution on a device, in the Uop_weak suite (L3) and rune's lowering |
| tinygrad: runtime/test_dtype.py::TestFp8sConversions::test_min_max_representable | the extremes of an 8-bit float round-trip | `D › data types › properties.golden` (columns `min`, `max`); `D › storage › storage round-trips every value` |
| tinygrad: runtime/test_dtype.py::TestFp8sConversions::test_float_to_fp8e4m3, test_float_to_fp8e5m2, test_float_to_fp8e4m3fnuz, test_float_to_fp8e5m2fnuz and their _extreme_values | encoding matches torch's bytes, overflow and NaN | `D › storage › truncation.golden` (column `storage`) |
| tinygrad: runtime/test_dtype.py::TestFp8sConversions::test_fp8e4m3_to_float, test_fp8e5m2_to_float, test_fp8e4m3fnuz_to_float, test_fp8e5m2fnuz_to_float, test_smallest_normals | decoding every byte | `D › storage › decode.golden`; `D › storage › every bit pattern decodes and encodes back, a NaN to a NaN` |
| tinygrad: runtime/test_dtype.py::TestFp8sConversions::test_cast_of_zero | ±0.0 encode as 0 and 0x80, fnuz has no negative zero | `D › storage › truncation.golden` (value=0.0, -0.0) |
| tinygrad: runtime/test_dtype.py::TestBFloat16DType::test_bf16 | 10000 rounds to 9984 | `D › truncate › truncation.golden` (dtype=dtypes.bfloat16 value=10000.0, -10000.0) |
| tinygrad: runtime/test_dtype.py::TestBitCast::test_bitcast_float_to_int32, test_bitcast_bf16_from_cast | float bits as an integer; bfloat16 1.0 read as half is 1.875 | `D › bitcast › bitcasts.golden` (from=dtypes.float to=dtypes.int value=1.0; from=dtypes.bfloat16 to=dtypes.half value=1.0) |
| tinygrad: runtime/test_dtype.py::TestIntegerCast::test_narrow_then_widen, TestInt8DType::test_int8_to_uint8_negative, test_int8_to_uint16_negative, TestUint8DType::test_uint8_to_int8_overflow, TestUint16DType::test_uint16_to_int8_overflow, TestInt64DType::test_int64_to_uint32_to_int64 | integer casts wrap | `D › truncate › truncation.golden`; `D › truncate › an integer wraps modulo two to its width` |
| tinygrad: runtime/test_dtype.py::TestDoubleDType::test_float64_to_float32_cast_inf | float32 overflows to inf | `D › truncate › truncation.golden` (dtype=dtypes.float value=1e+39) |
| tinygrad: runtime/test_dtype.py::TestDType and its per-dtype subclasses, TestBitCast (subword and shape tests), TestInt8DType::test_bitcast_alt, TestUint64DType, TestBFloat16DTypeCast, TestFloatDType, TestDoubleDType (other tests), TestImplicitFunctionTypeChange, TestTensorMethod, TestDtypeUsage, TestOpsBFloat16 | casts, bitcasts and ALU on a device | dropped: execution on a device, in the codegen and runtime layers (L4-L7). The scalar rules they check against are the truncation, decode and bitcasts goldens |
| tinygrad: runtime/test_dtype_alu.py (every test) | device ALU results against numpy, rounded with `truncate` | dropped: execution on a device (L5-L7). The rounding reference it uses is `D › truncate › truncation.golden` |
| tinygrad: runtime/test_dtype_spec.py::TestTypeSpec::test_dtype_str_arg | "nonexistdtype" and "" name no data type | `D › names › the error names what is not a data type`; `D › names › the empty string is not a data type` |
| tinygrad: runtime/test_dtype_spec.py (every other test) | dtypes of Tensor creation and reductions | dropped: Tensor surface, rune's lowering |
| old: unit/uop/test_dtype.ml predicates | priority, bitsize and predicates of the weak types | `D › data types › properties.golden` |
| old: unit/uop/test_dtype.ml repr_surface | `dtypes.<alias>` printing | `D › data types › properties.golden` (column `dtype`) |
| old: unit/uop/test_dtype.ml address_space | address spaces print as `global`, … | `D › address spaces › an address space prints as its enum member`: they now print as tinygrad's repr, `AddrSpace.GLOBAL`; `D › address spaces › addr_space_of_string reads the name pp prints after AddrSpace.` |
| old: unit/uop/test_dtype.ml promotion_matrix, promotion_edges | the promotion table | `D › promotion › least_upper.golden` |
| old: unit/uop/test_dtype.ml promotion_errors | empty list and void raise; a singleton bounds itself | `D › promotion › least_upper rejects the empty list`; `D › promotion › least_upper rejects void`; `D › promotion › a data type is its own least upper bound` |
| old: unit/uop/test_dtype.ml least_upper_float_cases, strong_and_weak | `least_upper_float`, `strong`, `weak` | `D › defaults › projections.golden` |
| old: unit/uop/test_dtype.ml lossless_matrix, lossless_to_weak | the lossless table, casts into weakint | `D › lossless casts › lossless_cast.golden` |
| old: unit/uop/test_dtype.ml sum_acc | `sum_acc` per data type | `D › defaults › projections.golden` (column `sum_acc`) |
| old: unit/uop/test_dtype.ml fp16_conversion | half rounding, the 2^-25 tie, overflow | `D › truncate › truncation.golden` (dtype=dtypes.half) |
| old: unit/uop/test_dtype.ml bf16_conversion | 1234 to 1232, rounding above a tie, 1e39 to inf | `D › truncate › truncation.golden` (dtype=dtypes.bfloat16); `D › truncate › bfloat16 rounds once, to the nearest, ties to even` |
| old: unit/uop/test_dtype.ml fp8_conversion | 8-bit encodes, saturation, NaN for infinities without one | `D › storage › truncation.golden`; `D › truncate › an infinity is NaN in an 8-bit float without infinities` |
| old: unit/uop/test_dtype.ml storage_format_mapping | each data type maps to an Nx_dtype scalar | dropped: tolk.next is a compiler with no nx dependency, so its data types map to no storage format of nx's |
| old: unit/uop/test_dtype.ml integer_truncation | integer wrapping; floats rejected | `D › truncate › truncation.golden` (rows `raises TypeError`); `D › truncate › an integer wraps modulo two to its width` |
| old: unit/uop/test_dtype.ml storage_formats | `storage_fmt` | `D › defaults › projections.golden` (column `storage_fmt`) |
| old: unit/uop/test_dtype.ml truncation_surface | bool of NaN is true, wrapping, saturation, bf16 | `D › truncate › truncation.golden` |
| old: unit/uop/test_dtype.ml storage_roundtrips | storage words of bf16 and fp8, negative zero | `D › storage › truncation.golden`; `D › storage › decode.golden`; `D › storage › storage round-trips every value` |
| old: unit/uop/test_dtype.ml bounds | bounds of every data type; void has none | `D › data types › properties.golden`. Void's bounds are now tinygrad's, `False` and `True` |
| old: unit/uop/test_dtype.ml float_info | `finfo`, weakfloat and ints rejected | `D › data types › finfo.golden`; `D › data types › finfo rejects every data type but the floats of known width` |
| old: unit/uop/test_dtype.ml defaults | defaults under an unset environment | `D › defaults › projections.golden` |
| old: unit/uop/test_dtype.ml env_dtype_parsing | DEFAULT_FLOAT and SUM_DTYPE names, aliases rejected | `D › defaults` (DEFAULT_FLOAT through `Helpers.context`, not a child process); the cram test `dtype/sum_dtype.t`, in a fresh process per SUM_DTYPE, which is read once |
| old: unit/uop/test_dtype.ml const_float_identity | one NaN, two zeros | `D › constants › every NaN is the same constant`; `D › constants › zero and negative zero are different constants`; `D › constants › every NaN hashes alike` |
| old: unit/uop/test_dtype.ml const_uop_float_identity | hash-consing of constant UOps | dropped: UOps, in the Ops suite (L1) |
| old: unit/uop/test_dtype.ml properties › "promotion commutative", properties › "promotion idempotent" | lattice laws | `D › promotion › least_upper is commutative`; `D › promotion › a data type is its own least upper bound` |
| old: unit/uop/test_dtype.ml properties › "lossless reflexive" | a cast to itself is lossless | `D › lossless casts › lossless_cast.golden` (its diagonal) |
| old: unit/uop/test_dtype.ml properties › "sum_acc idempotent" | the accumulator accumulates in itself | `D › defaults › projections.golden` covers every data type, so the law follows from the table |
| old: unit/uop/test_dtype.ml properties › "fp16 idempotent", properties › "bf16 idempotent", properties › "fp8 idempotent", properties › "truncate_int idempotent" | truncation is idempotent | `D › truncate › truncation is idempotent` |
| old: unit/uop/test_uop.ml const_scalar_payload_constructors | int payload coerced to float, NaN canonical, zeros distinct, Invalid kept | `D › const › const.golden`; `D › constants` |
| old: unit/uop/test_uop.ml scalar_float_to_weak_integer | a weakint beyond 64 bits prints in full | `D › constants › const_repr.golden` (const=18446744073709551616, -2^799) |
| old: unit/uop/test_weak.ml (every test) | weak commits in graphs | dropped: UOp graphs, in the Uop_weak suite (L3) |

## Op

An old test's line is the line of the call its assertion checks.

| Source | Behaviour | Outcome |
|---|---|---|
| old: `unit/uop/test_ops.ml:123` op order matches vendored tinygrad | the operations and their order are tinygrad's | `Tolk_next.Op › Op › declares tinygrad's operations in tinygrad's order`, `Tolk_next.Op › Set.to_list › of all is the declared operations`; the golden replaces running Python at test time |
| old: `unit/uop/test_ops.ml:64` op order matches vendored tinygrad | BACKEDGE spliced in from a second checkout ahead of the pin move | dropped: tinygrad HEAD declares BACKEDGE itself, and the golden records it |
| old: `unit/uop/test_ops.ml:146` GroupOp memberships match vendored tinygrad | each group's members are tinygrad's | `Tolk_next.Op › Named sets › hold tinygrad's members` (one case per tinygrad operation, checking every group column) |
| old: `unit/uop/test_ops.ml:154` predicates match public groups | `is_unary` and the other predicates agree with the group lists | dropped: the interface has no per-group predicates; `Set.mem` is the membership test, pinned by the `Tolk_next.Op › Set.mem` group |
| old: `unit/uop/test_uop.ml:98` Ops and dtype access | `Add` is an ALU operation | `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ADD` |
| old: `unit/uop/test_uop.ml:184` Ops tinygrad order | the full hand-written order, with WAIT, REWRITE_ERROR and PYLITERAL | `Tolk_next.Op › Op › declares tinygrad's operations in tinygrad's order`; WAIT is gone from tinygrad HEAD; REWRITE_ERROR and PYLITERAL: `Tolk_next.Op › Op › has no counterpart for exactly REWRITE_ERROR and PYLITERAL` |
| old: `unit/uop/test_uop.ml:192-193` Ops.Group algebra | `mem` finds `Add` in binary and rejects `Range` | `Tolk_next.Op › Set.mem › of of_list is membership of the list`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ADD`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.RANGE` |
| old: `unit/uop/test_uop.ml:194` Ops.Group algebra | `union` merges two lists without duplicates | `Tolk_next.Op › Set.mem › of union is membership of either`, `Tolk_next.Op › Set.to_list › of of_list is the listed operations in declaration order, once each`; the order changes from first occurrence to declaration order, since a group is a set |
| old: `unit/uop/test_uop.ml:198` Ops.Group algebra | `without binary comparison` removes the comparisons | `Tolk_next.Op › Set.mem › of diff is membership of the first and not the second`, `Tolk_next.Op › Set.diff › removes Threefry from alu and keeps the rest` |
| old: `unit/uop/test_uop.ml:219` Ops.Group algebra | reduce is Add, Mul and Max | `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ADD`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.MUL`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.MAX` |
| old: `unit/uop/test_uop.ml:221` Ops.Group algebra | defines holds Buffer, Alloc and Param | `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.BUFFER`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.ALLOC`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.PARAM` |
| old: `unit/uop/test_uop.ml:223` Ops.Group algebra | irreducible holds Param and Getaddr | `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.PARAM`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.GETADDR` |
| old: `unit/uop/test_uop.ml:225` Ops.Group algebra | broadcastable leaves out Group | `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.GROUP` |
| old: `unit/uop/test_uop.ml:227,229` Ops.Group algebra | Cdiv, Cmod, Floordiv and Floormod are binary | `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.CDIV`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.CMOD`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.FLOORDIV`, `Tolk_next.Op › Named sets › hold tinygrad's members › ops.golden › op=Ops.FLOORMOD` |
| old: `unit/uop/test_uop.ml:230` Ops.Group algebra | predicates agree with the group lists | dropped: no predicates, as for `test_ops.ml:154` |
| tinygrad: `uop/__init__.py` (no test file targets it) | `str` and `repr` of an operation are `Ops.<NAME>` | `Tolk_next.Op › Op › pp is Ops. followed by the name` and the golden's `op` column |
| tinygrad: `uop/__init__.py` (no test file targets it) | the enum's integer values order the toposort | `Tolk_next.Op › Op order › compare agrees with tinygrad's integer order` |
| tinygrad: `uop/__init__.py` (no test file targets it) | `auto()` values are unique across every `FastEnum` subclass | dropped: OCaml variants of distinct types never compare equal; `X86Ops` belongs to the renderer's section |
| tinygrad: `null/test_upat_compile.py::TestUPatCompile::test_const_folding` | `GroupOp.ALU - {Ops.THREEFRY}` | `Tolk_next.Op › Set.diff › removes Threefry from alu and keeps the rest`; the pattern compilation itself belongs to `Upat` |
| tinygrad: `null/test_linearizer.py`, `runtime/test_linearizer.py`, `null/test_schedule.py`, `null/test_const_folding.py`, `null/test_dtype_weak.py`, `null/test_pattern_matcher.py` | `u.op in GroupOp.ALU` counts arithmetic | `Tolk_next.Op › Named sets › hold tinygrad's members` (the ALU column); each test belongs to its own module's section |
| tinygrad: `null/test_graph_rewrite.py`, `null/test_uop_graph.py` | `UPat(GroupOp.All)` matches every operation | `Tolk_next.Op › Set.mem › of all always holds`; each test belongs to its own module's section |
| tinygrad: `null/test_viz.py::TestViz::test_colored_label`, `test_colored_label_multiline`, `test_inf_loop`, `TestVizGC::test_gc_uop_in_arg` | PYLITERAL and REWRITE_ERROR nodes in the graph viewer | dropped: viz is excluded; `Tolk_next.Op › Op › has no counterpart for exactly REWRITE_ERROR and PYLITERAL` |

## Ops

The suite is `Tolk_next.Ops` (`uop/ops/`), written `O` below. A golden
check is named after its golden, and a table golden's rows are tests keyed by
their input cells. Graph goldens are the `O › graphs` group, one test per
golden. `L2` marks the mixin tests folded into this module (plan §9, L2).

Tests of tinygrad that run a later module's rewrites (`sym`, `symbolic`,
`full_rewrite`, `to_uops_list`, `pm_mops`, the renderers, `Estimates`) belong
to that module's section; each is listed here once, with its owner.

### tinygrad: null/test_uop_graph.py

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_uop_graph.py::TestTuplize::test_equality_is_identity | structural order is zero exactly on equal nodes, and ignores tags | `O › compare_structure › is zero exactly on equal graphs, when untagged`; `O › compare_structure › ignores tags` |
| tinygrad: null/test_uop_graph.py::TestTuplize::test_deep_shared_subgraphs | deep graphs differing at the bottom order both ways, quickly | `O › compare_structure › tells apart deep graphs that differ only at the bottom` |
| tinygrad: null/test_uop_graph.py::TestTuplize::test_does_not_retain_uops | the cached tuple does not keep nodes alive | `O › identity › a node nothing references is collected`; `compare_structure` caches nothing |
| tinygrad: null/test_uop_graph.py::TestGraphRewriteConst (2 tests) | `sym` folds an index of a constant stack and a sum of stacks | Symbolic's section (`sym`); the index fold at construction is `O › kernel nodes › an index of a stack by a constant is the element` |
| tinygrad: null/test_uop_graph.py::TestModularWraparound (6 tests, xfail) | `simplify` folds constants modulo the width | Symbolic's section (`simplify` with `symbolic`); the wrapping itself is `O › exec_alu › exec_alu_values.golden` |
| tinygrad: null/test_uop_graph.py::TestGraphRewrite::test_dedup | a rewrite keeps equal nodes shared | `O › graph_rewrite › keeps shared nodes shared` |
| tinygrad: null/test_uop_graph.py::TestGraphRewrite::test_no_dedup_args (xfail) | a node inside a variable's bounds | dropped: a variable's bounds are values (`Dtype.value`), so a node cannot be one |
| tinygrad: null/test_uop_graph.py::TestGraphRewrite::test_simple, test_depth_2_late, test_double, test_triple, test_diamond, test_magic_4 | `simple_pm` folds to a fixed point | `O › graph_rewrite › folds to a fixed point` |
| tinygrad: null/test_uop_graph.py::TestGraphRewrite::test_depth_2_fold | a rule's result is rewritten in turn | `O › graph_rewrite › rewrites a node's result in turn` |
| tinygrad: null/test_uop_graph.py::TestGraphRewrite::test_commutative_work, test_consts_go_last_right_away, test_consts_go_last | `simplify` orders operands | Symbolic's section |
| tinygrad: null/test_uop_graph.py::TestUOpGraph::test_where_same_fold, test_where_const_fold, test_depth_2_const_fold | `simplify` folds selections and sums | Symbolic's section |
| tinygrad: null/test_uop_graph.py::TestUOpGraph::test_const_cast, test_cast_alu_fold, test_double_cast_fold, test_bitcast_to_same_dtype_fold, test_sub_with_cast_folds, test_where_on_gated_load_fold, test_where_on_gated_load_folds_swapped_branches, test_where_on_gated_load_with_cast, test_where_on_casted_gated_load_extra_cond, test_where_on_casted_gated_load_extra_cond_swapped, test_where_in_store_becomes_gate, test_load_idx_becomes_int, test_load_idx_no_math_on_loaded, test_fold_gated_load, test_fold_gated_load_local, test_fold_gated_store | the codegen pipeline folds kernels | Codegen's section (`full_rewrite`, `to_uops_list`) |
| tinygrad: null/test_uop_graph.py::TestUOpGraph::test_devectorize_derives_lane_dtype, test_devectorize_zero_sized_scalar_expand | devectorizing | Codegen's section |
| tinygrad: null/test_uop_graph.py::TestUOpGraph::test_gep_vec_const_fold | an index of a stack by a constant is its element | `O › kernel nodes › an index of a stack by a constant is the element`; `O › graphs › kernel_nodes.golden` |
| tinygrad: null/test_uop_graph.py::TestUOpGraph::test_after_end | an end closes its range, and an after on it too | `O › ranges › an end closes its ranges, and an after ordered on it too` |
| tinygrad: null/test_uop_graph.py::TestUOpGraph::test_external_call_preserves_ranges | a call to a function keeps its arguments' ranges | `O › ranges › a call to an external function keeps its arguments' ranges` |
| tinygrad: null/test_uop_graph.py::TestUOpGraph::test_backedge_preserves_outer_range | a backedge closes its loop only | `O › ranges › a backedge closes its loop and keeps the condition's other ranges` |
| tinygrad: null/test_uop_graph.py::TestReduceCollapse (2 tests) | `pm_reduce_collapse`, `full_rewrite` | Codegen's section |
| tinygrad: null/test_uop_graph.py::TestMovementOps (2 tests) | `pm_mops` folds reshapes into indices | Schedule.Prepare's section |
| tinygrad: null/test_uop_graph.py::TestConstBufferize (2 tests) | `pm_const_buffer_folding` | Schedule.Rangeify's section; `bufferize` itself is `O › arguments › reprs.golden` (name=bufferize) and `O › shapes › a stage puts its ranges' sizes in front` |
| tinygrad: null/test_uop_graph.py::TestUOpTags::test_inc_by_one | a tag makes a rewrite apply once, and removing tags reopens it | `O › graph_rewrite › tags let a rewrite apply once` (it checks the graphs, since folding the sums is `simplify`'s) |
| tinygrad: null/test_uop_graph.py::TestUOpGetItem (18 tests) | `UOp.__getitem__` | dropped: `__getitem__` is not ported (README exclusions); its shrink, permute and index are `O › graphs › movement.golden` and `O › graphs › kernel_nodes.golden` |
| tinygrad: null/test_uop_graph.py::TestUOpBroadcast::test_broadcast_row, test_broadcast_col, test_broadcast_lower_dim, test_broadcast_scalar, test_broadcast_symbolic_same_shape | elementwise operations broadcast shapes | `O › shapes › an elementwise operation broadcasts its sources' shapes`; `O › shapes › an elementwise operation keeps a symbolic shape` |
| tinygrad: null/test_uop_graph.py::TestUOpBroadcast::test_broadcast_axes | `broadcast_axes`, symbolic sizes, rejection | `O › shapes › broadcast_axes is the axes broadcasting adds or expands`; `O › shapes › broadcast_axes compares symbolic sizes` |

### tinygrad: null/test_pattern_matcher.py, test_rewrite_bottom_up_gate.py, test_graph_rewrite.py

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_simple_match | a constant pattern with a type | `O › Upat › matches a constant of a type` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_upat_any | `UPat.any` tries each alternative | `O › Upat › any matches through any of its alternatives` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_minimum_len | `allow_any_len` takes more sources, not fewer | `O › Upat › allow_any_len takes more sources, never fewer` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_match_sz_0 (skipped) | a rule with a closure | `O › Pattern_matcher › rule_ctx reads the context`: OCaml rules are closures, so tinygrad's skip reason is gone |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_match_sz_0_ctx | a rule reads a context, and an empty source list matches exactly | `O › Pattern_matcher › rule_ctx reads the context` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_uop | one operation | `O › Upat › matches an operation, and no other` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_uop_set | a set of operations | `O › Upat › matches any operation of a set` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_arg | arguments compare as numbers | `O › Upat › matches an argument as a number: 0, 0.0 and false are equal` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_filter_arg | a rule filters on its sources | `O › Upat › the sources of a rule filter it` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_dup_name | a name used twice | `O › Upat › a name used twice matches one node` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_dtype, test_dtype_set | one type, several types | `O › Upat › matches a type among several` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_src_one | ordered sources, exact count | `O › Upat › src matches the sources in order, and exactly as many` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_src_permutations | sources in any order | `O › Upat › perm matches the sources in any order` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_src_repeat | a repeated source pattern | `O › Upat › each matches every source` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_allow_len | `allow_any_len` | `O › Upat › allow_any_len takes more sources, never fewer` |
| tinygrad: null/test_pattern_matcher.py::TestPatternMatcher::test_deep_src_permutations | nested permutations | `O › Upat › permutations nest` |
| tinygrad: null/test_rewrite_bottom_up_gate.py::TestBottomUpGate (2 tests) | a gate keeps the node and skips its sources | `O › graph_rewrite › a gate keeps a bottom-up node and leaves its sources unvisited` |
| tinygrad: null/test_graph_rewrite.py::TestModuloAndDivisionFolding::test_graph_rewrite_div_folding_bug | `full_rewrite` of a stack comparison | Codegen's section |
| tinygrad: null/test_graph_rewrite.py::TestEdgeCasesAndSpecialOperations::test_full_graph_rewrite_transcendental_edge_cases | log2 of -1 is NaN, 1/0 is +inf, folded by `full_rewrite` | Codegen's section; the values are `O › exec_alu › exec_alu_values.golden` (op=Ops.LOG2, op=Ops.RECIPROCAL) |
| tinygrad: null/test_graph_rewrite.py::TestGEPAndVectorizeRewrite (3 tests) | index and stack folding by `full_rewrite` | Codegen's section; the construction-time fold is `O › kernel nodes › an index of a stack by a constant is the element` |
| tinygrad: null/test_graph_rewrite.py::TestBottomUpRewrite::test_const_folding | bottom-up and top-down reach the same fold with `symbolic_simple` | Symbolic's section; the law on this module's own rules is `O › fixed points › bottom-up reaches the same fixed point on these rules` |
| tinygrad: null/test_graph_rewrite.py::TestSubstitute::test_simple, test_double, test_diamond | substitution everywhere | `O › substitute › replaces a node wherever it is` |
| tinygrad: null/test_graph_rewrite.py::TestSubstitute::test_sin, test_sin_to_sqrt, test_double_sin_to_sqrt | the node nearest the root is replaced first | `O › substitute › replaces the node nearest the root first` |
| tinygrad: null/test_graph_rewrite.py::TestSubstitute::test_tagged_replace | a rebuilt node keeps its tag | `O › substitute › keeps a rebuilt node's tag` |
| tinygrad: null/test_graph_rewrite.py::TestRecurse::test_no_inf_loop, test_no_inf_loop_bottom_up | a rule that returns its node declines | `O › graph_rewrite › a rule that returns its node declines, whatever the direction`; the `TrackedPatternMatcher` half is dropped: match tracking is not ported (README exclusions) |
| tinygrad: null/test_graph_rewrite.py::TestRecurse::test_inf_loop, test_inf_loop_bottom_up | a rule pair that bounces is rejected | `O › graph_rewrite › rejects rules that never settle, whatever the direction` |
| tinygrad: null/test_graph_rewrite.py::TestRecurse::test_self_referential_replacement, test_indirect_rewrite_dependency_cycle | a replacement that depends on itself is rejected | `O › graph_rewrite › rejects a replacement that depends on the node it replaces` (the message's `SIN@`/`SQRT@` labels are Python ids, not ported) |
| tinygrad: null/test_graph_rewrite.py::TestRecurse::test_self_referential_call_argument | a call whose argument is the node it replaces | `O › graph_rewrite › rejects a call whose argument depends on the call` |
| tinygrad: null/test_graph_rewrite.py::TestCallRewrite::test_wrap_node_in_call | a node becomes a call holding it, in every mode | `O › calls › a node can become a call that holds it` |
| tinygrad: null/test_graph_rewrite.py::TestCallRewrite::test_body_shared_with_argument | a body is entered only with `enter_calls` | `O › calls › a body is rewritten only with enter_calls, its arguments always` |
| tinygrad: null/test_graph_rewrite.py::TestCallRewrite::test_body_shared_with_sibling | a sibling of the body is rewritten, the body is not | `O › calls › a body shared with a sibling is left alone, the sibling rewritten` |
| tinygrad: null/test_graph_rewrite.py::TestBidirectional::test_simple | `bpm` visits before the sources, `pm` after | `O › graph_rewrite › bpm rewrites before the sources and the matcher after them` |
| tinygrad: null/test_graph_rewrite.py::TestStopEarly::test_stop_early | `extra_pm` never enters a replacement | `O › substitute › rewrites with extra_pm too, never inside a replacement` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_topdown_simple_substitute, test_walk_topdown_rewrites_children, test_walk_topdown_diamond | a top-down walk substitutes | `O › walk › top-down, substitutes once` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_topdown_does_not_traverse_into_replacement | top-down walk against the greedy rewrite | `O › walk › top-down, does not enter a replacement` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_topdown_no_fixed_point | a bouncing rule applies once | `O › walk › top-down, applies a bouncing rule once`; `O › graph_rewrite › rejects rules that never settle, whatever the direction` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_topdown_children_rewritten_before_parent | sources first | `O › walk › top-down, rewrites the sources before the rebuilt node` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_topdown_self_referential_replacement | a replacement holding its node | `O › walk › top-down, accepts a replacement that holds the replaced node` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_topdown_visit_order | post-order | `O › walk › top-down, visits after the sources` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_bottomup_simple_substitute, test_walk_bottomup_does_not_traverse_into_replacement, test_walk_bottomup_unmatched_falls_through_to_children | a bottom-up walk | `O › walk › bottom-up, substitutes once and never enters a replacement` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_bottomup_parent_match_skips_children | a matched node's sources are skipped | `O › walk › bottom-up, a matched node's sources are never visited` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_bottomup_no_fixed_point | a bouncing rule applies once | `O › walk › bottom-up, applies a bouncing rule once` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_bottomup_visit_order | pre-order | `O › walk › bottom-up, visits before the sources` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_bidirectional_visit_order | `bpm` pre-order, `pm` post-order | `O › walk › both ways, bpm visits before the sources and the matcher after` |
| tinygrad: null/test_graph_rewrite.py::TestWalkRewrite::test_walk_bidirectional_bpm_short_circuits | a `bpm` match skips the node's `pm` | `O › walk › both ways, a bpm match skips the node's sources and its matcher` |

### tinygrad: null/test_uop_repr.py, test_uop_resolve.py, test_uop_vmin_vmax.py

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_uop_repr.py::TestUOpRepr (4 tests) | `repr` of a node, with `x0:=` sharing | `O › printing › pretty.golden` (every node of the tests, and tags, kernels, ranges and typed constants) |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_simple_int, test_int_add, test_rfloordiv, and the integer half of test_weak_const | `int()` of a typed, weak or summed integer | `O › resolve › to_int, to_float and to_bool read a literal`; `O › resolve › to_int reads a typed constant and an integer sum of constants` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_weak_const (float half) | `float()` of a weak float | `O › resolve › to_int, to_float and to_bool read a literal` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_rtruediv, test_float_direct, test_ssimplify | float and remainder folding | Symbolic's section: without `symbolic` a float sum has no single-valued bounds |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_lt, test_leq, test_ne, test_ne_f, test_ngt | comparisons of constants | `O › resolve › to_bool decides comparisons of constants` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_ambiguous_less_than | `resolve` falls back to its default | `O › resolve › resolve takes the default when the comparison is undecided` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_var_cmp_t, test_var_cmp_t2, test_var_cmp_f, test_var_cmp_f2, test_max | bounds decide a comparison | `O › resolve › to_bool decides a comparison the bounds decide` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_or_true, test_and_false | an absorbing boolean decides | `O › resolve › to_bool decides a disjunction with true and a conjunction with false` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_or_false, test_and_true, test_x_lt_xp1, test_var_cmp_range, test_var_cmp_assert | an undecided condition raises | `O › resolve › to_bool rejects a condition with two possible values` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_x_lt_x, test_plus_ordering_lt | `simplify` folds `x < x` and `i+j < j+i` | Symbolic's section |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_constant | a constant | `O › bounds › a constant is its own bounds` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_copy | a copy and a contiguous pass bounds through | `O › bounds › a copy and a contiguous keep their source's bounds` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_cmpne | `!=` of constants | `O › bounds › a comparison of constants is decided` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_addition_with_variable, test_vmin_vmax_subtraction_with_variable, test_vmin_vmax_multiplication_with_variable, test_vmin_vmax_with_negative_multiplication, test_vmin_vmax_with_negative_multiplication2 | interval arithmetic | `O › bounds › a variable offset or scaled moves its bounds`; `O › bounds › binary_bounds.golden` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_and_with_variable, test_vmin_vmax_and_with_negative_variable | masks | `O › bounds › a mask bounds a variable by the mask`; `O › bounds › binary_bounds.golden` (op=Ops.AND) |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_variable_inside_special | a hardware index | `O › bounds › a hardware index counts from 0 to below its end` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_multiplication_0_inf | `0.0 * load` is unbounded, not NaN | `O › bounds › a product with an unbounded float is unbounded, never NaN` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_nested_min_max | max then min | `O › bounds › maximum then minimum clamps` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_where | selections | `O › bounds › a selection spans both branches` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_shl, test_vmin_vmax_shr | shifts by a constant | `O › bounds › a shift by a constant shifts the bounds`; `O › bounds › binary_bounds.golden` (op=Ops.SHL, Ops.SHR) |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_cast_unsigned, test_vmin_vmax_cast, test_vmin_vmax_cast_float_to_int, test_vmin_vmax_cast_int_to_float_grid | casts keep the part that fits, rounded | `O › bounds › cast_bounds.golden`; `O › bounds › a variable cast to float, bool or unsigned`; `O › bounds › a typed constant outside its type has the type's bounds`; `O › bounds › a NaN constant has its type's bounds`; the `simplify` of `x != int(x)` is Symbolic's |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_xor_neg1 | `x ^ -1` | `O › bounds › binary_bounds.golden` (op=Ops.XOR b_lo=-1) |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_invalid | Invalid has no single value | `O › bounds › Invalid has no single value` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxProperties::test_vmin_vmax_invalid_vconst | a stack with Invalid lanes | `O › bounds › a stack's bounds span its values, Invalid left out` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxDivMod (7 tests) | division and remainder, constant and symbolic divisors, empty numerators | `O › bounds › binary_bounds.golden` (op=Ops.FLOORDIV, Ops.FLOORMOD, Ops.CDIV, Ops.CMOD); `O › bounds › an empty range divides to 0` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxVConst (5 tests) | constant stacks of integers, floats, booleans | `O › bounds › a stack's bounds span its values, Invalid left out` |
| tinygrad: null/test_uop_vmin_vmax.py::TestVminVmaxVConst::test_vmin_vmax_vector_with_gep | a load of an int buffer divided by 32 | `O › bounds › a load of an integer buffer has its type's bounds` |
| tinygrad: null/test_uop_vmin_vmax.py::TestConstFactor (7 tests; 1 skipped) | `const_factor` | `O › divisibility › const_factor is a known divisor`; the skipped `(x*3)*5` case is dropped: tinygrad skips it as broken |
| tinygrad: null/test_uop_vmin_vmax.py::TestDivides (7 tests; 1 skipped) | `divides` | `O › divisibility › divides divides a known multiple`; the skipped `(x*6)/6` case is dropped: tinygrad skips it as broken |

### tinygrad: null/test_uops.py, test_uops_stats.py, runtime/test_uops.py

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_broadcastable_promotion, test_same_dtype_fast_path | broadcastable operations promote | `O › data types › dtypes_of.golden`; `O › data types › promo_dtype is the shared type, or the least upper one` |
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_where_promotion | a selection promotes its branches | `O › data types › dtypes_of.golden` (op=Ops.WHERE); `O › graphs › validity.golden` |
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_const_dtype_from_value | a constant's type comes from its value | `O › data types › a constant's data type is its literal's`; the tuple argument is dropped: `arg` has no tuple constant |
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_const_default_dtype_is_derived | the same under `SPEC=2` | `O › data types › a constant's data type is its literal's`; the `SPEC=2` check is Spec's section |
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_invalid_dtype_and_consumers | Invalid is boolean, ignores the type, and stays last | `O › constants › Invalid ignores the type`; `O › graphs › typed_constants.golden`; `O › graphs › stacks.golden`; `Tensor.invalids` is dropped (Tensor surface); `type_verify` is Spec's; `pm_remove_invalid` is Symbolic's |
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_remove_invalid_stack_lanes | `pm_remove_invalid` | Symbolic's section |
| tinygrad: null/test_uops.py::TestMemoryCoalescing, TestLowerIndexDtype (3 tests) | coalescing, `pm_lower_weak` | Codegen's section and Uop_weak's section |
| tinygrad: null/test_uops.py::TestSafeCast (3 tests) | `simplify` removes casts | Symbolic's section |
| tinygrad: null/test_uops.py::TestConstFloatEq::test_nan_eq_ne_agree, test_invalid_eq_defers_to_reflected | Python's `==`/`!=` protocol on constants | dropped: constants compare with `Dtype.equal_const`, which Dtype's section tests |
| tinygrad: null/test_uops.py::TestConstFloatEq::test_matchers_agree_on_nan | a NaN argument matches a NaN constant | `O › Upat › a NaN argument matches a NaN constant`; the compiled matcher is dropped (the pattern compiler is not ported) |
| tinygrad: null/test_uops.py::TestExecALU::test_sqrt | sqrt of 0 | `O › exec_alu › sqrt of zero is zero` |
| tinygrad: null/test_uops.py::TestExecALU::test_trunc_nonfinite, test_invalid_poison, test_div, test_floordiv, test_floormod, test_recip, test_bool_cmplt, test_bool_cmpne, test_bool_where, test_overflow | `exec_alu` values, truncation, Invalid | `O › exec_alu › exec_alu_values.golden` (every operand of these tests is a row); `O › exec_alu › Invalid poisons every binary operation` |
| tinygrad: null/test_uops.py::TestGatedStoreRewrite (3 tests), TestFastIdiv (13 tests) | gated stores, fast division | Codegen's section |
| tinygrad: null/test_uops.py::TestUOpMethod::test_compare_alu_same_src_different_arg (skipped) | nodes are ordered by `<` | dropped: tinygrad skips it; `O › identity › compare is a total order that agrees with equal` |
| tinygrad: null/test_uops.py::TestUOpMethod::test_uop_variables | a `Tensor` program's variables | dropped: Tensor surface; `O › variables › variables are sorted by name, with a device range's device number` |
| tinygrad: null/test_uops.py::TestUOpMethod::test_const_factor | `const_factor` of a hardware index | `O › divisibility › const_factor is a known divisor` |
| tinygrad: null/test_uops.py::TestUOpMethod::test_cmp_self_folding_multidim | `simplify` folds `x < x` | Symbolic's section |
| tinygrad: null/test_uops.py::TestUOpMethod::test_replace | `replace` changes a field, and rejects an unknown one | `O › identity › replace is the node itself when nothing changes`; the unknown field is dropped: labelled arguments make it a type error |
| tinygrad: null/test_uops.py::TestUOpMethod::test_const_zero_neg_zero_different | 0.0 and -0.0 are different nodes | `O › identity › zero and negative zero are different nodes` |
| tinygrad: null/test_uops.py::TestUOpMethod::test_const_nan_same | NaN constants are one node | `O › identity › every NaN is the same node` |
| tinygrad: null/test_uops.py::TestUOpStr (3 tests) | `str` is compact and `eval` reads it back | Render's section; `eval` is dropped (Python source) |
| tinygrad: null/test_uops.py::TestUPatHelpers::test_location | a pattern records its source location | dropped: `UPat.location` serves match statistics, not ported (README exclusions) |
| tinygrad: null/test_uops.py::TestUopsObject::test_timing | building 10k constants | dropped: a timing print |
| tinygrad: null/test_uops.py::TestUopsObject::test_nested | the device of a 10k-deep graph | `O › several devices › the device of a deep graph needs no deep recursion` |
| tinygrad: null/test_uops.py::TestUOpRender (7 tests) | `render` | Render's section |
| tinygrad: null/test_uops.py::TestContiguousViewOffset (7 tests) | `contiguous_view_offset` | Schedule.Prepare's section: `contiguous_view` is one of its functions (D4) |
| tinygrad: null/test_uops.py::TestBitcastBufferView, TestLocalAccess, TestAssembly | renderer output | the renderers' sections |
| tinygrad: null/test_uops_stats.py (all) | kernel cost estimates | Renderer's section (`Estimates.from_uops`) and the engine's (`estimate_uop`) |
| tinygrad: runtime/test_uops.py::TestFloatUOps, TestNonFloatUOps, TestBoolUOps | a device computes each operation as Python does | the Python side is `O › exec_alu › exec_alu_values.golden`; the device side is an Exec test of the renderer layers |
| tinygrad: runtime/test_uops.py::TestBitcastBufferView, TestLocalAccess, TestZeroRange, TestUOpPrograms | kernels run on a device | Exec tests of the renderer layers |

### L2: null/test_tensor_uop_mixin.py, test_tensor_uop_representation.py

A `Tensor` and a UOp built by the same calls must be the same node. tolk.next
has no `Tensor`, so each kept method's UOp graph is a golden instead.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpBinop::test_mul_float_int, test_mul_bool_int | mixed types promote with a cast | `O › elementwise › a binary operation promotes its operands to their least upper type`; `UOp.arange` is dropped (not kept) |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpBinop::test_add_scalar_float_on_int | a float literal on an int | `O › graphs › weak_promotion.golden` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpBinop::test_div_tensor_by_tensor, test_div_int_by_int, test_div_broadcast_tensor_by_tensor | true division | `O › graphs › division.golden`; `O › graphs › constant_division.golden` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpBinop::test_div_sum_by_sum, test_isclose | `sum`, `isclose` | dropped: not kept (plan §3, L2) |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpBinop::test_floordiv_int, test_floordiv_float, test_rfloordiv_int, test_mod_int, test_mod_float, test_div_trunc_int, test_div_trunc_float, test_fmod_int, test_fmod_float, test_floordiv_bool, test_mod_bool, test_fmod_bool | rounding divisions by type | `O › graphs › constant_division.golden` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpClone::test_clone | `clone` | `O › graphs › clones.golden` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpClone::test_clone_deviceless_const | a clone of a constant goes to the default device | dropped: there is no default device (RFC 0010 Law 9); `O › storage › a clone of a weak value commits its type` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpGradient | `gradient` | dropped: rune owns differentiation |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpGetitem (all) | `__getitem__` | dropped: not ported (README exclusions) |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpCumalu, TestTensorUOpCumMinMax, TestTensorUOpArgMinMax, TestTensorUOpSequential, TestTensorUOpOneHot, TestTensorUOpSort, TestTensorUOpAllclose, TestTensorUOpRand, TestTensorUOpGather, TestTensorUOpInterpolate, TestTensorUOpLoss, TestTensorUOpScatter, TestTensorUOpScatterReduce, TestTensorUOpMaskedSelect, TestTensorUOpNonzero, TestTensorUOpPool, TestTensorUOpConv2d, TestTensorUOpHashing, TestTensorUOpEinsum, TestTensorUOpSoftmax, TestTensorUOpQR, TestTensorUOpSVD | the other `Tensor` methods | dropped: not kept (plan §3, L2); rune's lowering |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpCast::test_cast_str_dtype, TestTensorUOpBitcast::test_bitcast_str_dtype | a type named by a string | dropped: types are `Dtype.t` values; reading a name is `Dtype.of_string` (Dtype's section) |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpBitcast::test_bitcast_same_dtype | a bitcast to the same type is the node | `O › elementwise › a cast or bitcast to the node's own type is the node` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpBitcast::test_bitcast_same_and_diff_size | bitcasts that keep, widen or narrow the element | `O › graphs › bitcasts.golden`; `O › shapes › a bitcast rescales the last axis, and rejects a size that does not divide` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpCat (4 tests) | `cat` along each axis, three inputs, a negative axis | `O › graphs › concatenation.golden` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpPad::test_pad_flat, test_pad_flat_negative, test_pad_grouped_none | constant padding, negative, `None` axes | `O › graphs › padding.golden` (the flat spelling is `Tensor` syntax; the interface takes one pair per axis) |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpPad::test_pad_circular, test_pad_circular_zero_after, test_pad_reflect, test_pad_reflect_negative, test_pad_replicate, test_pad_replicate_negative | other padding modes | dropped: only constant padding is kept (plan §3, L2) |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpStack::test_stack_dim0, test_stack_dim1, test_stack_3tensors, test_stack_new_last, test_stack_mixed_dtype | `stack` | `O › graphs › stacks.golden` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpStack::test_stack_index_dtype | a stack of weak constants | `O › graphs › bitcasts.golden` (its last node) |
| tinygrad: null/test_tensor_uop_mixin.py::TestUOpEmpty::test_empty_dtype_string | a type named by a string | dropped: types are values |
| tinygrad: null/test_tensor_uop_mixin.py::TestUOpEmpty::test_empty_like_dtype_override | `empty_like` with a type is storage | `O › graphs › clones.golden`; `O › several devices › empty_like on one device takes a sharded value's whole shape` |
| tinygrad: null/test_tensor_uop_mixin.py::TestUOpEmpty::test_empty_like_sharded_to_single_device | a sharded value's `empty_like` on one device | `O › several devices › empty_like on one device takes a sharded value's whole shape`; the singleton-tuple spelling is dropped: canonicalizing devices is Device's (L6) |
| tinygrad: null/test_tensor_uop_mixin.py::TestUOpEmpty::test_empty_direct_singleton_tuple_device | a one-device tuple canonicalizes | dropped: canonicalizing devices is Device's (L6) |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpCreation::test_empty, test_empty_like | `empty`, `empty_like` | `O › graphs › clones.golden` |
| tinygrad: null/test_tensor_uop_mixin.py::TestTensorUOpCreation::test_full, test_full_kwargs, test_full_symbolic_fill, test_zeros, test_ones, test_invalids, test_arange, test_arange_empty, test_arange_step, test_linspace, test_linspace_one_step, test_eye, test_eye_rect, test_triu, test_triu_diagonal, test_tril, test_tril_diagonal | other creation methods | dropped: not kept (plan §3, L2) |
| tinygrad: null/test_tensor_uop_representation.py (5 tests) | a realized `Tensor` is a BUFFER | dropped: realization is rune's (D3) |

### old tolk: unit/uop/test_uop.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/uop/test_uop.ml full_width_scalar_bindings | int64 extremes bind, pass through `vals` and infer | `O › bounds › a constant at its type's edge has exact bounds`; `O › variables › bind binds a value within the range`; dropped for `vals` and `sym_infer`: variable values are OCaml ints, as the executor passes them |
| old: unit/uop/test_uop.ml constants_preserve_operand_shape | `const_like` keeps shape and type | `O › constants › const_like expands to the node's shape, in its type`; `O › graphs › constants_like.golden` |
| old: unit/uop/test_uop.ml compiled_signature_preserves_slots_and_types, binary_argument_layout, incomplete_program_has_no_binary | `to_elf` and its argument layout | Device's section: `to_elf` becomes one of its functions (D4) |
| old: unit/uop/test_uop.ml commutative_axes_use_lexical_argument_order | `simplify` orders ranges | Symbolic's section |
| old: unit/uop/test_uop.ml ops_access, ops_tinygrad_order, group_algebra | operations and their groups | Op's section |
| old: unit/uop/test_uop.ml hashcons_identity | equal constructions are one node | `O › identity › building a graph twice gives the same node` |
| old: unit/uop/test_uop.ml hashcons_churn_compacts_dead_buckets, hashcons_resize_discards_retired_bucket_accounting | the old hash table's buckets | dropped: they test the old table's internals; the contract is `O › identity › a node nothing references is collected` |
| old: unit/uop/test_uop.ml concurrent_buffer_slots | slots from several domains are distinct | `O › storage › unique_num never returns a number twice, from any domain`; `reserve_buffer_slots` is dropped (not in tinygrad) |
| old: unit/uop/test_uop.ml add_has_two_srcs, infix_builds_mul | construction and operators | `O › graphs › subtraction.golden`; `O › elementwise › the operators are the named operations` |
| old: unit/uop/test_uop.ml validity_accessors_accept_invalid_and_generic_values | `get_idx`/`get_valid` of Invalid and plain values | `O › graphs › validity.golden` |
| old: unit/uop/test_uop.ml bool_folds_are_logical, mixed_folds_promote | `usum`/`uprod` on booleans, with promotion | `O › elementwise › usum and uprod fold booleans with or and and`; `O › elementwise › a weak constant takes the other operand's kind and stays weak`; `O › graphs › sums_and_products.golden` |
| old: unit/uop/test_uop.ml param_arg_symbolic_constructor | a variable is a scalar parameter with bounds | `O › variables › a variable is a named scalar with a range`; `O › arguments › reprs.golden` (name=variable) |
| old: unit/uop/test_uop.ml arithmetic_helpers_tinygrad_parity | the division operators, `const_factor`, `divides`, `divide_exact`, `gcd` | `O › graphs › division.golden`; `O › divisibility` (every test); `O › divisibility › a typed constant is a cast, whose divisors are not known` records tinygrad's answer for a typed constant, where the old suite divided it |
| old: unit/uop/test_uop.ml bind_requires_concrete_value, bind_validates_range, unbind_splits_bound_variables | `bind`, `unbind` | `O › variables` (every test); the `as_bind` view is dropped: a bound variable is its argument's value |
| old: unit/uop/test_uop.ml integer_bounds_parity | bounds of empty ranges, hardware indices, floor division and remainder, int64 and uint64 edges | `O › bounds › a hardware index counts from 0 to below its end`; `O › bounds › an empty range divides to 0`; `O › bounds › binary_bounds.golden`; `O › bounds › a constant at its type's edge has exact bounds` |
| old: unit/uop/test_uop.ml wrapping_integer_bounds, unsigned_arithmetic_bounds_cover_emission | bounds widen to the type when arithmetic may wrap | dropped: tinygrad's bounds do not wrap; overflow is undefined. The new contract is `O › bounds › bounds hold every value the expression takes` and the exact rows of `O › bounds › binary_bounds.golden` (dtype=dtypes.char, dtypes.uchar, dtypes.uint) |
| old: unit/uop/test_uop.ml cast_bounds | casts keep what fits | `O › bounds › cast_bounds.golden` |
| old: unit/uop/test_uop.ml flat_storage_parameters | a parameter is flat storage viewed as its shape | `O › graphs › storage.golden`; `O › graphs › symbolic_storage`; `O › shapes › max_shape takes a symbolic size's greatest value`; the image parameter is dropped (images are not ported) |
| old: unit/uop/test_uop.ml backward_slice_tracks_shared_dependencies | `backward_slice` | `O › graphs › backward_slice is the reached nodes without the root or call bodies` |
| old: unit/uop/test_uop.ml allocations_preserve_shape_and_address_space | `alloc` keeps shape, type and address space | `O › graphs › storage.golden`; `O › graphs › storage_like.golden`; the rejection of a local `alloc` with a device is dropped: tinygrad checks it in `placeholder` only (`O › storage › placeholder rejects a device for local storage`) |
| old: unit/uop/test_uop.ml placeholder_checks_shape_product, max_numel_checks_host_range, max_numel_handles_zero_after_large_dimensions | sizes past the largest int, and empty axes | `O › shapes › placeholder rejects a size past the largest int`; `O › shapes › max_numel is 0 when an axis is empty` |
| old: unit/uop/test_uop.ml movement_dimensions_do_not_wrap | a shrink past the largest int is rejected | `O › shapes › a shrink past the largest int is rejected, not wrapped` |
| old: unit/uop/test_uop.ml bitcast_dimensions_remain_exact_until_host_conversion | a byte count past the largest int | dropped: sizes are OCaml ints; see `O › shapes › placeholder rejects a size past the largest int` |
| old: unit/uop/test_uop.ml exact_symbolic_bounds | bounds are exact at any size; NaN bounds; a fractional cast | `O › bounds › bounds are exact integers, whatever their size`; `O › bounds › a NaN constant has its type's bounds`; `O › bounds › a typed constant outside its type has the type's bounds`; `parse_valid` is Symbolic's section |
| old: unit/uop/test_uop.ml stack_stage_slice_constructors, stack_promotes_all_operands, stack_prepends_leading_dim | stacks and stages | `O › graphs › stacks.golden`; `O › shapes › a stack prepends its length`; `O › shapes › a stage puts its ranges' sizes in front` |
| old: unit/uop/test_uop.ml uop_constructor_parity_shortcuts | shortcuts that return the node, an index of a stack | `O › elementwise › a cast or bitcast to the node's own type is the node`; `O › kernel nodes › an index of a stack by a constant is the element`; `O › kernel nodes › an index of a stack by a negative constant counts from the end, as a Python tuple`; `O › kernel nodes › an end of no ranges, and an after of nothing, are the node`; `O › elementwise › contiguous stages a placed value, and is the node otherwise`; the rendered strings are Render's |
| old: unit/uop/test_uop.ml const_scalar_payload_constructors | typed constants, NaN, -0.0, Invalid | `O › graphs › typed_constants.golden`; `O › identity › every NaN is the same node`; `O › identity › zero and negative zero are different nodes`; `O › constants › Invalid ignores the type` |
| old: unit/uop/test_uop.ml call_constructor_parity | `call` and `call_with_outputs` | `O › calls › call rejects a body that computes a value`; `O › calls › call rejects a range leaking out of its body, but a device range`; `O › graphs › outputs.golden` |
| old: unit/uop/test_uop.ml deviceless_partition_selection | an MSTACK of unplaced lanes | dropped: a placement names its devices (`device` has no absent lane), as tinygrad's `device` returns a tuple of strings |
| old: unit/uop/test_uop.ml property_helpers_parity | sharding axis, shard shapes, bounds, movement arguments, stages, storage views, call output shapes | `O › several devices` (axis, sharding, shard shapes and bounds tests); `O › movement › marg reads each movement's argument and rejects other nodes`; `O › storage › a stage is its own base and storage, without buffer identity`; `O › storage › buf_uop is the storage a node accesses`; `O › graphs › symbolic_outputs` (a call output's shape takes the call's argument); `O › graphs › constants_like.golden` |
| old: unit/uop/test_uop.ml reduce_layouts | tensor and kernel reductions | `O › graphs › reductions.golden`; `O › graphs › kernel_nodes.golden` |
| old: unit/uop/test_uop.ml binary_and_getaddr_dtypes | BINARY is bytes, GETADDR is uint64 | `O › shapes › binary code is a vector of its bytes`; `O › graphs › getaddrs.golden` |
| old: unit/uop/test_uop.ml void_and_value_op_shapes | effects have no shape, typed instructions are scalars | `O › shapes › shape_opt is None for effects and program structure`; `O › shapes › a typed instruction is a scalar, and a custom node broadcasts its sources` |
| old: unit/uop/test_uop.ml prepend_expand | an expand prepends | `O › shapes › an expand prepends its sizes` |
| old: unit/uop/test_uop.ml bitcast_size_change | a bitcast rescales the last axis | `O › shapes › a bitcast rescales the last axis, and rejects a size that does not divide` |
| old: unit/uop/test_uop.ml child_ops_reports_child_op_set | the set of a node's source operations | dropped: it is the matcher's private early-reject memo; its behaviour is `O › Pattern_matcher › early_reject skips a rule unless the sources hold its operations` |
| old: unit/uop/test_uop.ml property_caches_release_nodes | memoized properties do not keep nodes alive | `O › identity › a node nothing references is collected` |
| old: unit/uop/test_uop.ml exec_alu_folds_and_absorbs_invalids, exec_alu_exact_scalars, scalar_width_boundaries, exec_alu_weak_intermediates, exec_alu_trunc_keeps_nonfinite | `exec_alu` | `O › exec_alu › exec_alu_values.golden`; `O › exec_alu › truncating is Dtype.truncate of the exact result`; `O › exec_alu › Invalid poisons every binary operation` |
| old: unit/uop/test_uop.ml exec_alu_float_division | FDIV folds with IEEE values | dropped: tinygrad's `exec_alu` has no FDIV (`O › exec_alu › exec_alu_values.golden` row op=Ops.FDIV raises); division folds through RECIPROCAL, whose rows hold the IEEE values |
| old: unit/uop/test_uop.ml scalar_float_to_weak_integer | a float becomes a weak integer exactly | Dtype's section (`Dtype.const`) |
| old: unit/uop/test_uop.ml alu_unary_promotes_transcendentals | transcendentals widen to a float | `O › data types › dtypes_of.golden` (op=Ops.SQRT, Ops.EXP2, Ops.RECIPROCAL); the rejection of a binary operation is dropped: `alu` applies an operation as given |
| old: unit/uop/test_uop.ml division_promotes_integer_operands | FDIV of integers is a float | `O › data types › dtypes_of.golden` (op=Ops.FDIV); its folding is Symbolic's |
| old: unit/uop/test_uop.ml runtime_realization_state_parity | realized buffers | dropped: realization is rune's (D3) |
| old: unit/uop/test_uop.ml semantic_tag_and_side_metadata | a tag is identity, not in the key; metadata aside | `O › identity › a tag is part of the node`; `O › key › ignores tags`; metadata is dropped (not ported) |
| old: unit/uop/test_uop.ml info_function_names_follow_tinygrad | kernel function names | `O › arguments › function_name is the name as an identifier`; the program's name is Renderer's |
| old: unit/uop/test_uop.ml cache_info_semantic_key_parity | the key tells beams apart and ignores aux | `O › key › tells arguments apart`; `O › key › tells apart an operation, a type, an argument and a source`; `O › key › ignores a call's auxiliary data`; serialization is dropped (not ported) |
| old: unit/uop/test_uop.ml remove_all_tags_parity | `remove_all_tags` | `O › module matchers › remove_all_tags removes every tag, and leaves an untagged graph alone`; metadata is dropped (not ported) |
| old: unit/uop/test_uop.ml program_constructor_prefix_layouts | PROGRAM source layouts | Spec's section: the interface builds programs with `v` |
| old: unit/uop/test_uop.ml program_info_from_sink_parity, program_launch_dims_floor_divmod | `ProgramInfo.from_sink`, launch sizes | `O › programs` (every test) |
| old: unit/uop/test_uop.ml sym_infer_host_scalars | `sym_infer` on casts, bound variables, huge values | `O › sym_infer` (every test); values past the largest int are dropped: `sym_infer` returns an OCaml int |
| old: unit/uop/test_uop.ml debug_prints_toposort_like_tinygrad, debug_prints_ranges_and_supplied_list_sources, debug_prints_tinygrad_dtype_reprs, debug_prints_float_and_special_args_like_tinygrad, debug_prints_direct_string_args_like_tinygrad, debug_prints_ranges_in_tinygrad_arg_order, debug_prints_reduce_arg_tuple, debug_listing_omits_tags | the listing `print_uops` writes | Render's section; the argument texts are `O › arguments › reprs.golden` |
| old: unit/uop/test_uop.ml debug_prints_rich_args_dataclass_style | argument reprs | `O › arguments › reprs.golden`; `O › arguments › each payload formats as its argument does`; the `Opt` repr is in the Opt section |
| old: unit/uop/test_uop.ml debug_print_ignores_side_metadata | metadata is not printed | dropped: metadata is not ported |
| old: unit/uop/test_uop.ml upat_matches_add, upat_captures_operands | a pattern matches and names | `O › Upat › matches an operation, and no other`; `O › Upat › match_ names the nodes of each way a pattern matches` |
| old: unit/uop/test_uop.ml pattern_matcher_rewrites | identity rules | `O › fixed points` (the rule `x + 0`) |
| old: unit/uop/test_uop.ml upat_operator_surface_matches_tinygrad | `/`, `//`, `%`, `cdiv`, `cmod` patterns | `O › elementwise patterns › floor division and remainder operators match their operations`; `O › elementwise patterns › a binary pattern matches the node its operation builds` |
| old: unit/uop/test_uop.ml pattern_matcher_context_rewrites | a rule reads a context | `O › Pattern_matcher › rule_ctx reads the context`; `O › Pattern_matcher › with_ctx joins a matcher without context to one with` |
| old: unit/uop/test_uop.ml upat_matches_node_tags | tags narrow a pattern | `O › Upat › a tag narrows the nodes a pattern matches` |
| old: unit/uop/test_uop.ml upat_numeric_literals | arguments compare exactly as numbers | `O › Upat › matches integers and floats exactly`; `O › Upat › matches an argument as a number: 0, 0.0 and false are equal` |
| old: unit/uop/test_uop.ml pattern_matcher_rejects_opless_rules, context_matcher_rejects_opless_rules | a rule needs an operation | `O › Pattern_matcher › v rejects a rule whose pattern has no operation` |
| old: unit/uop/test_uop.ml custom_early_reject_skips_callback | `early_reject` | `O › Pattern_matcher › early_reject skips a rule unless the sources hold its operations` |
| old: unit/uop/test_uop.ml upat_dtype_matches_scalar_of_vector | a scalar type matches a vector | dropped: tinygrad HEAD has no vector types; `O › Upat › matches a type among several` |
| old: unit/uop/test_uop.ml upat_explicit_source_patterns | fixed, permuted, repeated and alternative sources | `O › Upat › src matches the sources in order, and exactly as many`; `O › Upat › perm matches the sources in any order`; `O › Upat › each matches every source`; `O › Upat › allow_any_len takes more sources, never fewer`; `O › Upat › any matches through any of its alternatives` |
| old: unit/uop/test_uop.ml upat_matches_reduce_arg | a reduce pattern's operation | `O › Upat › a reduce pattern names its operation in the argument` |
| old: unit/uop/test_uop.ml upat_rejects_reserved_ctx_capture | the name `ctx` | `O › Upat › v rejects two ways of matching sources` (the name `ctx` is allowed, as tinygrad's interpreter allows it) |
| old: unit/uop/test_uop.ml upat_permutation_matches_are_deduplicated | one naming per distinct match | dropped: tinygrad lists a naming per permutation of distinct source patterns; `O › Upat › match_ names the nodes of each way a pattern matches` |
| old: unit/uop/test_uop.ml pattern_matcher_ignores_self_replacement | a rule returning its node declines | `O › Pattern_matcher › the first rule that matches and does not decline wins` |
| old: unit/uop/test_uop.ml graph_rewrite_walk_does_not_enter_replacements | walk | `O › walk › top-down, does not enter a replacement` |
| old: unit/uop/test_uop.ml graph_rewrite_bpm_runs_before_post_order | `bpm` then `pm` | `O › graph_rewrite › bpm rewrites before the sources and the matcher after them` |
| old: unit/uop/test_uop.ml graph_rewrite_walk_bpm_short_circuits_replacement | a walk's `bpm` match skips the subtree | `O › walk › both ways, a bpm match skips the node's sources and its matcher` |
| old: unit/uop/test_uop.ml graph_rewrite_bottom_up_gate_skips_post_and_children | a gate | `O › graph_rewrite › a gate keeps a bottom-up node and leaves its sources unvisited` |
| old: unit/uop/test_uop.ml graph_rewrite_walk_bottom_up_gate_skips_post_and_children | a gate in a walk | dropped: tinygrad's walk does not catch the gate (question sent to the implementer) |
| old: unit/uop/test_uop.ml graph_rewrite_enters_native_callee | a callee's dependencies are rewritten with the caller | dropped: tinygrad leaves call bodies alone without `enter_calls` (`O › calls › a body is rewritten only with enter_calls, its arguments always`) |
| old: unit/uop/test_uop.ml graph_rewrite_skips_call_body_by_default | the body is left, the arguments rewritten | `O › calls › a body is rewritten only with enter_calls, its arguments always` |
| old: unit/uop/test_uop.ml graph_rewrite_pins_call_body_on_every_path | a body reached another way is left alone | dropped: tinygrad rewrites a node reached outside the call; `O › calls › a body shared with a sibling is left alone, the sibling rewritten` |
| old: unit/uop/test_uop.ml graph_rewrite_detects_bottom_up_cycles, graph_rewrite_detects_top_down_cycles | cycles are rejected | `O › graph_rewrite › rejects rules that never settle, whatever the direction` |
| old: unit/uop/test_uop.ml after_closes_ranges_from_dependencies | an after closes what its dependencies end | `O › ranges › an end closes its ranges, and an after ordered on it too` |
| old: unit/uop/test_uop.ml shared_ending_dependencies_have_bounded_allocations | ranges of shared barriers allocate little | dropped: an allocation measure of the old range cache; ranges are computed over a gated toposort (`O › shapes › the shape of a deep graph needs no deep recursion` shows the shape) |
| old: unit/uop/test_uop.ml nested_ending_dependencies_preserve_live_range_order, range_order_and_membership_share_scope | ranges through nested ends, in order | `O › ranges › an after of a barrier over an end closes the ended range`; the order and `ranges_subset` are dropped: the interface leaves the order open, and `ranges_subset` is not in tinygrad |
| old: unit/uop/test_uop.ml linear_closes_ranges | LINEAR closes its ranges | `O › ranges › a linear program closes the ranges it lays out` |
| old: unit/uop/test_uop.ml resolve_decides_comparisons_from_bounds | `resolve` | `O › resolve › resolve takes the default when the comparison is undecided`; `O › resolve › resolve rejects a node that is not boolean` |
| old: unit/uop/test_uop.ml smax_smin_fold_when_bounds_decide | `smax`/`smin` | `O › resolve › smax and smin of integers are integers`; `O › resolve › smax and smin of a symbolic size bound it as max and min do`; folding to a node is Symbolic's |
| old: unit/uop/test_uop.ml sprod_simplifies | `sprod` | `O › Sint › prod multiplies from 1`; folding is Symbolic's |
| old: unit/uop/test_uop.ml inferred_broadcast_shapes_are_checked, broadcast_shape_symbolic_and_raising | broadcasting | `O › shapes › broadcast_shape aligns right and keeps the size that is not 1`; `O › shapes › broadcast_shape rejects two sizes other than 1`; `O › shapes › an elementwise operation rejects sources that do not broadcast` |

### old tolk: unit/uop/test_spec.ml, unit/uop/test_serialize.ml, unit/frontend/test_frontend.ml, unit/test_contiguous_view.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/uop/test_spec.ml "Where non-bool cond rejected", "Cmplt returns bool", "Stack derives child dtype", "Empty Stack void accepted" | the types construction derives | `O › data types › dtypes_of.golden`; `O › shapes › a stack prepends its length` |
| old: unit/uop/test_spec.ml "Param without Param_arg rejected", "Buffer without Param_arg rejected" | storage needs a ParamArg | `O › data types › an operation whose type is its argument's needs that argument` |
| old: unit/uop/test_spec.ml "Reduce op arg required", "Reduce rejects old Op arg" | a reduction's argument | dropped: `arg`'s `Reduce` constructor makes both unrepresentable |
| old: unit/uop/test_spec.ml "call outputs reject invalid positions" | `output_pos` | `O › calls › call_with_outputs rejects output positions that do not ascend within the arguments` |
| old: unit/uop/test_spec.ml "bind accepts a variable and constant", "bind accepts a 64-bit variable", "variable supplies the bound literal dtype" | `bind` | `O › variables › bind binds a value within the range`; `O › bounds › a constant at its type's edge has exact bounds` |
| old: unit/uop/test_spec.ml "bind rejects a nonconstant value", "bind rejects a call parameter" | `bind` refuses a node or a parameter | dropped for the node: `bind` takes a value; `O › variables › bind rejects a bound variable, a value out of range, and a value off its multiple` rejects a parameter |
| old: unit/uop/test_spec.ml "Movement validates shape contracts" | movements check their shapes | `O › shapes › a movement checks its argument against its source's shape` |
| old: unit/uop/test_spec.ml (every other test) | the specification's verdicts | Spec's section |
| old: unit/uop/test_serialize.ml (10 tests) | exporting and importing graphs | dropped: pickling is not ported (README exclusions); the graph format lives in test support |
| old: unit/frontend/test_frontend.ml movement group (reshape, -1, same shape, size mismatch, expand, -1 axis, permute, negative axes, flip, pad, large padding, shrink, squeeze, flatten, unflatten) | movements on UOps | `O › graphs › movement.golden`; `O › graphs › expansion.golden`; `O › graphs › squeezes.golden`; `O › movement` (every test); unsqueeze, transpose, repeat, unfold and split are dropped (not kept, plan §3 L2) |
| old: unit/frontend/test_frontend.ml broadcast group | `broadcast_shape`, broadcasting in arithmetic, a stretch is one expand permuted back | `O › shapes › broadcast_shape aligns right and keeps the size that is not 1`; `O › shapes › an elementwise operation broadcasts its sources' shapes`; `O › graphs › expansion.golden` |
| old: unit/frontend/test_frontend.ml elementwise group (sub, div, neg, comparisons, eq, floordiv, mod, float div, int by float divisor, minimum, sqrt, where, promotion, const_like) | elementwise construction | `O › elementwise` (every test); `O › graphs › subtraction.golden`, `division.golden`, `constant_division.golden`, `comparisons.golden`, `extrema.golden`, `unary.golden`, `selection.golden`, `constants_like.golden`; relu is dropped (not kept) |
| old: unit/frontend/test_frontend.ml dtype group (cast, same dtype, bitcast needs concrete types, element_size, weak contiguous) | casts and their checks | `O › elementwise › a cast or bitcast to the node's own type is the node`; `O › elementwise › a bitcast rejects a weak type on either side`; `O › elementwise › element_size is the type's size, and rejects a weak type`; `O › elementwise › contiguous stages a placed value, and is the node otherwise`; `is_floating_point` is dropped (not kept) |
| old: unit/frontend/test_frontend.ml creation group ("a weak clone has a concrete dtype and fresh storage", "empty storage rejects weak dtypes", "numel checks concrete products...", "scalar const has empty shape") | clones, empty storage, numel | `O › storage › a clone of a weak value commits its type`; `O › storage › storage rejects a weak type`; `O › shapes › ndim, numel, max_shape and max_numel read the shape`; the rest of the group is dropped (Tensor registration, zeros, ones, full: not kept) |
| old: unit/frontend/test_frontend.ml op group (cat, stack), pool group (shrink_to, pad_to), scan group (pad_constant) | kept methods | `O › graphs › concatenation.golden`; `O › graphs › stacks.golden`; `O › graphs › movement.golden`; `O › graphs › padding.golden`; the other tests of these groups are dropped (not kept) |
| old: unit/frontend/test_frontend.ml elementwise2 group (pow, cdiv, fmod, lshift, rshift) | kept operations | `O › graphs › powers.golden`; `O › graphs › division.golden`; `O › graphs › bitwise.golden`; round, clamp, copysign, logaddexp, lerp, isnan and the other functions are dropped (not kept) |
| old: unit/frontend/test_frontend.ml scalar operand dtype group ("narrow int keeps its width through an int-scalar op", "a literal does not widen the tensor it meets", "scalar constructors are weak") | weak literals | `O › graphs › weak_promotion.golden`; `O › elementwise › a weak constant takes the other operand's kind and stays weak`; the activation tests are dropped (not kept) |
| old: unit/frontend/test_frontend.ml shape_memo group | a deep diamond's shape is cheap | `O › shapes › the shape of a deep graph needs no deep recursion` |
| old: unit/frontend/test_frontend.ml reduce, index, logspace, creation2, pad_modes, scatter, select, sort, conv groups, and the rest of op, scan, pool | `Tensor` methods | dropped: not kept (plan §3, L2); rune's lowering |
| old: unit/frontend/test_frontend.ml assignment sharding group | assigning to sharded tensors | dropped: the `Tensor` surface is rune's |
| old: unit/test_contiguous_view.ml (17 tests) | contiguous views of storage | Schedule.Prepare's section: `contiguous_view` is one of its functions (D4) |

## Spec

The suite is `Tolk_next.Spec` (`uop/spec/`), written `S` below.
`verdicts.golden` gives, for 273 nodes, the verdict of each of the six
specifications (`True`, `False`, or `None` when no rule decides); its test is
`S › verdicts › verdicts.golden › <case>`. `bounds.golden` gives the shared
verdict of 41 accesses with `CHECK_OOB` off and on, and `failures.golden`
the message of `type_verify` under `tensor`, `program`, and `tensor` without
entering calls. Each table's nodes are the sources of the sink of its
`_nodes.golden`, in row order.

tinygrad hands the accesses its bounds do not prove to z3, which tolk.next
does not have (README exclusions); the generator stands in a z3 that cannot
decide, so each `checked` verdict is tinygrad's under the ruling that an
unproved access fails.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `uop/spec.py` spec_shared, spec_tensor, spec_program, spec_hcq, spec_full, spec_kernel_graph (no test file targets the rules) | each rule's verdict on nodes it accepts, rejects and declines | `S › verdicts › verdicts.golden` (every row) |
| tinygrad: `uop/spec.py` type_verify | the message `UOp verification failed at {i} on {op} {dtype} {n} {srcs} {arg}`, the position in toposort order, `enter_calls` | `S › type_verify › failures.golden`; `S › type_verify › fails at the first node, sources first, that the specification does not accept` |
| tinygrad: `uop/spec.py` type_verify (`DEBUG >= 3`) | the graph's listing prints before the failure | `S › type_verify › prints the graph when DEBUG is 3 or more, before failing › debug_listing.golden`; `S › type_verify › prints nothing when DEBUG is below 3` |
| tinygrad: `uop/spec.py` type_verify (`SPEC > 1`: test_pyrender) | a checked graph round-trips through pyrender | dropped: pyrender is not ported (README exclusions) |
| tinygrad: `uop/spec.py` validate_index | with `CHECK_OOB`, vmin and vmax prove an access; multi-index accesses and the invalid index skip the check | `S › bounds › bounds.golden`; `S › bounds › an access passes iff the bounds of its index lie within the storage`; `S › bounds › without CHECK_OOB, every access passes` |
| tinygrad: `uop/spec.py` validate_index (z3 on the gate) | a gate or a mask proves an access | dropped: z3 is excluded (README); pinned the other way by `S › bounds › a gate never changes whether an access passes` |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC > 1`) | a created node is checked against spec_full, with `CHECK_OOB` off | `S › construction › checks each node it builds against the full specification when SPEC is 2`; `… › rejects a new node iff the full specification does not accept it`; `… › never checks bounds, whatever CHECK_OOB`; `… › accepts the forms only the full specification holds` |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC <= 1`, cache hit) | no check below 2; a cached node returns before the check | `S › construction › checks nothing when SPEC is 1`; `… › checks nothing when SPEC is 0`; `… › checks a node only when it creates it` (kills the `Ops` mutant at `ops.ml:672`) |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC > 2`) | the shape of each created node is computed | `S › construction › computes the shape of each node it builds when SPEC is 3` |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC > 3`) | pyrender round trip | dropped: pyrender is not ported (README exclusions) |
| tinygrad: null/test_dtype_weak.py::TestWeakDtypes::test_weak_int_binop (the `type_verify([bad], spec_shared)` lines) | bitwise operations on float32 and weak floats fail the shared spec | `S › verdicts › verdicts.golden` (an and of float32, an and of weak floats); `type_verify`'s list form is dropped: every tinygrad caller passes a graph, and a one-node list is the node's graph with sources that pass |
| tinygrad: null/test_dtype_weak.py::TestWeakDtypes::test_weak_int_binop (the dtype lines) | shift and bitwise data types | Ops' section (`dtype_of`) |
| tinygrad: null/test_uops.py::TestUOpMethod::test_invalid_dtype_and_consumers (the `type_verify(u, spec_shared)` loop) | Invalid matches any type in a stack, a sum, a where, both sides of a comparison, an index | `S › verdicts › verdicts.golden` (a stack of float32 and the invalid constant, a sum of float32 and the invalid constant, a where over float32 and the invalid constant, a cmplt of the invalid constant and float32, a cmplt of float32 and the invalid constant, an index by the invalid constant) |
| tinygrad: null/test_uops.py::TestUOpMethod::test_invalid_dtype_and_consumers, test_remove_invalid_stack_lanes (the `spec_program` lines) | `pm_remove_invalid`'s output is a program | dropped here: `pm_remove_invalid` is codegen (L4), whose suite checks its output with `Spec.program` |
| tinygrad: null/test_uops.py::TestUOpMethod::test_const_default_dtype_is_derived (`SPEC=2`) | constants build under the construction check | `S › construction › rejects a new node iff the full specification does not accept it` (constants are leaves of every graph); the data types are Ops' section |
| tinygrad: null/test_uops.py::TestUPatHelpers::test_location | spec_shared's first pattern is located in spec.py | dropped: pattern locations are match tracking (README exclusions) |
| tinygrad: null/test_uop_symbolic.py::TestMoveWhereOnLoad::test_bool_index_preserves_dtype | the rewrite's output passes spec_shared | `S › tinygrad › tests.golden › TestMoveWhereOnLoad.test_bool_index_preserves_dtype` |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_const_index | constant indexes 0, 15, 16, 42 into 16 elements | `S › bounds › bounds.golden` (a load at the first element, … at the last element, … one past the last element, … far past the last element) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_variable_index | variables over 0..15, 0..20, -5..10 | `S › bounds › bounds.golden` (a load over the elements, … past the last element, … before the first element) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_range_with_mask, test_variable_with_mask, test_gated_store, test_or_in_mask, test_xor_in_mask, test_float_cast_in_mask, test_bool_cast_in_mask, test_load_as_index, test_load_bool_as_mask, test_gated_local | a mask proves an access in bounds | `S › bounds › bounds.golden` (a load over a guarded range, … a guarded variable, a store over a guarded variable, a load from a local buffer within its end: all fail, as z3 cannot decide); the rest dropped: z3 is excluded (README) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_floordiv, test_mod, test_shr, test_and, test_max | vmin and vmax of floor division, modulo, shifts, masks and maximum prove an access | `S › bounds › bounds.golden` (a load over half a longer range, … modulo the length, … modulo more than the length, … masked to the length, … shifted into the buffer, … shifted past the buffer, … a clamped variable, … a variable clamped too little) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_shl, test_or, test_or_negative, test_float_cast_in_index, test_bitcast_in_index, test_load_from_shrink_as_index | bounds that z3 proves beyond vmin and vmax | dropped: z3 is excluded (README); the bounds these nodes have are Ops' section |
| tinygrad: null/test_validate_oob.py::TestShiftBounds (7 tests) | `validate_index_with_z3` | dropped: z3 is excluded (README) |
| tinygrad: runtime/test_tensor_variable.py::TestTensorVariable (`CHECK_OOB`) | a shrink by a variable past its dimension fails at realize | dropped: needs the schedule and the runtime (L6, L9) |
| tinygrad: null/test_const_folding.py, null/test_graph_rewrite.py, null/test_simplify_valid_idx.py, runtime/test_arange.py, runtime/test_linalg.py, runtime/test_tensor_cores.py, runtime/test_wait_loop.py, runtime/test_uops.py | set `SPEC` or `CHECK_OOB` for their own tests | dropped: they test other modules under a setting |
| old: `unit/uop/test_spec.ml` sink_void, empty_stack_void, stack_sources_match, const_matching_dtype | sinks, empty stacks, stacks, constants pass | `S › verdicts › verdicts.golden` (an empty sink, an empty stack, a stack of int32, an int32 constant) |
| old: `unit/uop/test_spec.ml` param_with_param_arg, buffer_with_param_arg, param_rejects_empty_arg, buffer_rejects_empty_arg | storage takes a `ParamArg` | `S › verdicts › verdicts.golden` (a parameter, a variable, a local buffer); a storage node without one cannot be built (`Ops.dtype_of`) |
| old: `unit/uop/test_spec.ml` shared_rejects_global_buffer, tensor_accepts_global_buffer, buffer_rejects_alu_addrspace, program_buffer_rules | global buffers are tensor-only; ALU buffers fail; local and register buffers pass programs | `S › verdicts › verdicts.golden` (a global buffer, an alu buffer, a local buffer, a register buffer) |
| old: `unit/uop/test_spec.ml` stack_derives_child_dtype, stack_rejects_mixed_dtype, stack_rejects_mixed_child_counts | stack sources share the type, a weak source passes, sources share a shape | `S › verdicts › verdicts.golden` (a stack of int32 and float32, a stack of int32 and a weak integer, a stack of a scalar and a vector) |
| old: `unit/uop/test_spec.ml` where_bool_cond, where_rejects_non_bool_cond | a where on a boolean passes; a non-boolean condition cannot be built | `S › verdicts › verdicts.golden` (a where over float32); construction is Ops' section |
| old: `unit/uop/test_spec.ml` cmplt_is_bool, alu_operand_scalars_match, cdiv_rejects_float, shift_count_dtypes, bitwise_rejects_float_operands | comparisons, sums, float divisions, shift counts (uint32, same type, weak; not narrower), bitwise floats | `S › verdicts › verdicts.golden` (a cmplt of int32, a sum of int32, a cdiv of float32, a shl of int8 by uint32, a shl of int32 by int32, a shl of int8 by a weak integer, a shl of uint64 by uint16, an and of float32, …); float shifts cannot be built (Ops' section) |
| old: `unit/uop/test_spec.ml` index_accepts_integer_offsets, index_rejects_gate_source | integer indexes pass; a boolean source declines | `S › verdicts › verdicts.golden` (an index by int32, an index by a boolean) |
| old: `unit/uop/test_spec.ml` special_accepts_raw_name, special_dtype_by_stage | weak launch dimensions in tensor graphs, int32 in programs | `S › verdicts › verdicts.golden` (a special over a weak integer, a special over int32) |
| old: `unit/uop/test_spec.ml` group_rejects_value_source, group_after_bad_layouts | a group of values declines | `S › verdicts › verdicts.golden` (a group of a value, a group of stores); an after without sources cannot be built (Ops' section) |
| old: `unit/uop/test_spec.ml` after_rejects_value_first_source, program_rejects_loose_after_layout | an after on a value declines, and full accepts it | `S › verdicts › verdicts.golden` (a constant after a store, a load after a store) |
| old: `unit/uop/test_spec.ml` end_rejects_non_range_tail, program_end_range_boundaries, end_requires_an_effect | an end closes ranges around an effect | `S › verdicts › verdicts.golden` (an end of a store over a constant, … over a loop header, an end of a value over a range, an end of a store over an int32 range) |
| old: `unit/uop/test_spec.ml` range_rejects_bad_layouts | a range needs its axis | `S › verdicts › verdicts.golden` (a range, a range with a nested axis); a range without an argument crashes tinygrad's kernel_graph rule, so it has no golden row |
| old: `unit/uop/test_spec.ml` barrier_boundaries | a barrier passes | `S › verdicts › verdicts.golden` (a barrier) |
| old: `unit/uop/test_spec.ml` conditional_loop_contract | a backedge on a scalar boolean passes; not on the invalid constant | `S › verdicts › verdicts.golden` (a backedge on a boolean, … on the invalid constant, … on a vector of booleans, a backedge of a bounded range); shapes are Ops' section |
| old: `unit/uop/test_spec.ml` copy_matching_dtype, copy_rejects_device_source_layout, copy_rejects_lowered_range_sources, copy_rejects_bad_device_or_dtype | copies to a device, not to a disk; a multi-device copy carries its device range | `S › verdicts › verdicts.golden` (a copy, a copy to two devices, … without its range, a copy to a disk); positional and empty device selectors do not exist in tolk.next's `device` |
| old: `unit/uop/test_spec.ml` call_reject_bad_layouts, call_source_contracts | calls of opaque bodies pass; value bodies decline | `S › verdicts › verdicts.golden` (a call of a sink, a call of a store, a call of a custom function, a call of a constant, a call of a load, a call without call info) |
| old: `unit/uop/test_spec.ml` call_outputs_reject_invalid_positions, bind_rejects_nonconstant_value, bind_rejects_parameter, bind_accepts_64_bit_variable, bind_variable_supplies_dtype, typed_host_call_contract | constructor preconditions of calls and binds | Ops' section; `S › verdicts › verdicts.golden` (a bound variable) |
| old: `unit/uop/test_spec.ml` bind_accepts_variable_const | a bound variable passes the tensor spec | `S › verdicts › verdicts.golden` (a bound variable) |
| old: `unit/uop/test_spec.ml` reduce_arg_required, tensor_reduce_accepts_lowered_integer_tail, reduce_rejects_old_op_arg | reduces by a reducing operation over integer ranges | `S › verdicts › verdicts.golden` (a reduce, a reduce over ranges, … over an int32 range, … over a uint32 range, a reduce by sine, a reduce without an argument) |
| old: `unit/uop/test_spec.ml` allreduce_layouts, allreduce_rejects_bad_device_or_dtype | allreduces by a reducing operation | `S › verdicts › verdicts.golden` (an allreduce, an allreduce by sine, an allreduce without an argument) |
| old: `unit/uop/test_spec.ml` multi_device_selection_layouts, multi_device_stack_layouts | a selection within the shards of a multi-device value; stacks of single-device values | `S › verdicts › verdicts.golden` (a selection of a shard, a selection past the shards, a selection on one device, a stack of shards, a stack of sharded values, a stack of one value without a device, a stack of values without a device) |
| old: `unit/uop/test_spec.ml` multi_device_multi_layouts | an unshard's axes against its source's rank and placement | `S › verdicts › verdicts.golden` (an unshard, an unshard over a derived range, an unshard missing a range, an unshard over an int32 range); the axis-range and placement checks are dropped: tinygrad's rule checks the count and types of the sharding ranges only |
| old: `unit/uop/test_spec.ml` stage_rejects_bad_layouts | stages pass with any ranges | `S › verdicts › verdicts.golden` (a staged value, a staged value over a range); the integer-range and placement checks are dropped: tinygrad's rule accepts any stage |
| old: `unit/uop/test_spec.ml` movement_validates_shape_contracts | movements pass the tensor spec; a pad's ends have one shape | `S › verdicts › verdicts.golden` (a reshape, an expand, a pad, a pad by a shorter end, a shrink, a permute, a flip); element counts and bounds are shapes, `S › construction › computes the shape of each node it builds when SPEC is 3` and Ops' section |
| old: `unit/uop/test_spec.ml` tensor_rejects_if_endif, program_control_flow_boundaries, program_accepts_if_with_shrink_index, program_rejects_bad_if_layouts, program_rejects_if_dedup_source_matrix, program_rejects_bad_endif_layouts | ifs and endifs are program-only, over an index, a cast or a shrink | `S › verdicts › verdicts.golden` (an if on an index, … on a cast, … on a shrink, … on a load, … on an int32 gate, an endif, an endif of a store) |
| old: `unit/uop/test_spec.ml` program_rejects_invalid_const, program_rejects_weakint, program_range_forms | programs state every width | `S › verdicts › verdicts.golden` (the invalid constant, a sum of weak integers, a sum under a constant, a range, a global int32 range) |
| old: `unit/uop/test_spec.ml` program_rejects_tensor_only_ops | copies, allreduces, reduces, stacks and selections of shards are not program nodes | `S › verdicts › verdicts.golden` (their rows, column `program`) |
| old: `unit/uop/test_spec.ml` program_accepts_plain_load, program_accepts_cast_index_load, program_accepts_load_gate_on_load, program_rejects_load_gate_on_index, program_accepts_plain_store, program_accepts_store_gate_on_store, program_rejects_store_gate_on_index, program_rejects_nested_casted_index_source | loads and stores through an index, a cast of one, with gates on the load or store | `S › verdicts › verdicts.golden` (a load of an index, a load of a cast index, a gated load, a store to an index, a gated store, a load of a twice cast index, a store to a load, …) |
| old: `unit/uop/test_spec.ml` program_bitcast_index_same_dtype_is_plain_index, program_rejects_real_bitcast_index_load | a load through a bitcast index declines | `S › verdicts › verdicts.golden` (a load of a bitcast index); the same-type bitcast fold is Ops' section |
| old: `unit/uop/test_spec.ml` program_oob_disabled_accepts_out_of_bounds_load, program_oob_enabled_rejects_out_of_bounds_load, program_oob_enabled_accepts_minmax_in_bounds_load, program_oob_uses_explicit_buffer_shape, program_oob_symbolic_store_remains_rejected, program_oob_rejects_unproved_indices | `CHECK_OOB` off passes all; on, vmin and vmax against max_numel | `S › bounds › bounds.golden`; `S › bounds › an access passes iff the bounds of its index lie within the storage`; `S › bounds › without CHECK_OOB, every access passes` |
| old: `unit/uop/test_spec.ml` program_oob_image_pointer_bypasses_bounds | images skip the check | dropped: images are excluded (README) |
| old: `unit/uop/test_spec.ml` program_oob_false_gate_accepts_out_of_bounds_load, program_oob_symbolic_false_gate_accepts_out_of_bounds_load, program_oob_masked_symbolic_bounds_are_accepted | a gate proves an access | dropped: z3 is excluded, and an unproved access fails whatever its gate (ruling); `S › bounds › a gate never changes whether an access passes` |
| old: `unit/uop/test_spec.ml` program_oob_masked_symbolic_lower_bound_only_rejected, program_oob_loaded_component_bounds, program_oob_offset_component_bounds, program_oob_shift_component_bounds, program_oob_distributed_scan_bounds, program_oob_distributed_metal_bounds, program_oob_affine_proof_preserves_narrowing, program_oob_affine_proof_preserves_unsigned_overflow, program_oob_affine_proof_checks_committed_constants, program_oob_unsigned_add_can_wrap (uint32, uint64), program_oob_small_unsigned_add_wraps, program_oob_committed_guard_constants, program_oob_proof_variables_are_fresh, program_oob_component_bounds_preserve_casts | the old affine prover of masked accesses | dropped: the old tolk's prover stood in for z3, which is excluded; tolk.next proves with vmin and vmax only (ruling) |
| old: `unit/uop/test_spec.ml` verify_list_validates_flat_program | the list form of verification | dropped: `type_verify` takes a graph; no tinygrad caller passes a list |
| old: `unit/uop/test_spec.ml` full_spec_has_no_catch_all | full rejects a node no rule decides | `S › verdicts › verdicts.golden` (a cast of two sources, an index by a float, … column `full`) |
| old: `unit/uop/test_spec.ml` full_spec_accepts_intermediate_forms | full accepts loose afters, loads and stores and ends over launch dimensions | `S › verdicts › verdicts.golden` (a constant after a store, a load of a parameter, a store to a constant, an end of a store over a launch dimension, an index of a stack) |

## Transcendental

Outcomes are tests of the suite `Tolk_next.Transcendental`. The graph goldens
pin every operation, constant and data type of each function for Float16,
Float32 and Float64, and `values.golden` and `pow_values.golden` pin the value
of tinygrad's graphs, evaluated node by node, at about 260 inputs per function
and type: special values, the reductions' boundaries and values drawn from
their bits.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_payne_hanek_reduction` | `(r, q)` of `12π + 0.1`, `12π`, `12π - 0.1` | reductions › payne_hanek_reduction removes quarter turns (the three inputs); graphs › `payne_hanek_*.golden` |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_cody_waite_reduction` | `(r, q)` of `12π + 0.1` | reductions › cody_waite_reduction removes half turns; graphs › `cody_waite_*.golden` |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_frexp` | mantissa and exponent of ±1, ±2, 5, 1000 in Float64, the sign dropped | reductions › frexp splits a double into a mantissa in [0.5, 1) and an exponent; reductions › frexp keeps the sign of a float's mantissa; graphs › `frexp_*.golden` |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_rintk` | rounding of 0, ±5, ±5.5, ±5.999 | bits › rintk rounds halves away from zero; bits › rintk gives the signed integer of its float's width |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_pow2if` | `2^q` for q in ±{0, 1, 2, 10, 63} | bits › pow2if is two to an integer; bits › pow2if gives the float of its integer's width; graphs › `pow2if_*.golden` |
| tinygrad: `null/test_transcendental.py::TestTranscendentalSchedule` (3 tests) | sin, log2 and exp2 of sums fuse into one kernel under `TRANSCENDENTAL=2` | dropped: fusion is the scheduler's, its suite (L6); the rewrite the setting forces is patterns › `patterns_all_forced_float.golden` |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalMath::test_float64`, `test_float32`, `test_float16` | exp, log and sin agree with numpy within atol/rtol 3e-2/1e-5, 2e-5/1e-5 and 1e-2/5e-3 (sin of Float64 below 1e8) | accuracy › xexp2, xlog2, xsin and xsin ~fast of each type are libm's within tinygrad's tolerance (on exp2 and log2: `Tensor.exp` and `Tensor.log` scale them in the frontend, nx); values › `values.golden` |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalMath::test_exp_near_inf` | exp just below overflow is finite | values › `values.golden` (xexp2 at 1023.9, 127.9, 15.9, 22.9 and the thresholds); special values › xexp2 of float overflows at 128 and underflows below -149 |
| tinygrad: `runtime/test_transcendental.py::TestFromFuzzer::test_sin` | sin at ±25, ±35, 30, 0, π/2 within 1 ulp of 1.0, 2π within 1.5 | fuzzer cases › xsin of `<type>` `<x>` |
| tinygrad: `runtime/test_transcendental.py::TestFromFuzzer::test_log2` | log2 of ±tiny × {1, 1e10, 1e20, 1e30}, 0 and 9e-7 within 1 ulp of 1.0 | fuzzer cases › xlog2 of `<type>` `<x>` |
| tinygrad: `runtime/test_transcendental.py::TestFloat16Log2::test_float16_log2_basic` | Float16 log2 of 1 to 1000 | values › `values.golden` (xlog2 half rows); accuracy › xlog2 of half |
| tinygrad: `runtime/test_transcendental.py::TestFloat16Log2::test_float16_log2_special` | Float16 log2 of inf, 0, -1, NaN | special values › xlog2 half inf/0/-1/nan rows |
| tinygrad: `runtime/test_transcendental.py::TestFloat16Log2::test_float16_log2_denormal` | Float16 log2 of 1e-4, 6e-5, 1e-5 | values › `values.golden` (the three inputs are special inputs of every type) |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalVectorized::test_exp2_vectorized`, `test_log2_vectorized`, `test_sin_vectorized` | the functions on vectors of widths 1 to 128 | dropped: vector widths are the devectorizer's and the executor's (L4, L7); the scalar values are `values.golden` and the accuracy properties |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalVectorized::test_pow_vectorized` | pow of (0.001, 200) to (-10, 10) | xpow › its four tests; values › `pow_values.golden`; vectors dropped as above |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalVectorized::test_sqrt_vectorized` | sqrt of (0, 100) | patterns › a rewritten operation computes it, rounded to its type; vectors dropped as above |
| tinygrad: `runtime/test_dtype_alu.py::TestDTypeALU` (the unary tests of bfloat16 and the 8-bit floats) | exp2, log2, sin and sqrt of the narrow floats on a device | patterns › a rewritten operation computes it, rounded to its type; patterns › `patterns_none_{bfloat16,fp8e4m3,fp8e5m2fnuz}.golden`; the device run is the executor's (L7) |
| tinygrad: `null/test_graph_rewrite.py::TestEdgeCasesAndSpecialOperations::test_full_graph_rewrite_transcendental_edge_cases` | `log2(-1)` folds to NaN | dropped: constant folding is Symbolic's, its suite; the decomposition's value is special values › xlog2 … -1 is nan |
| tinygrad: `null/test_dtype_weak.py::TestWeakPromotion::test_weak_transcendentals` | `Tensor.exp` of a Python number is weak | dropped: the frontend's promotion (nx), and `Uop.Ops`' data types |
| tinygrad: `transcendental.py` `get_transcendental_patterns` (no test) | which operations are rewritten, the Float32 detour of the narrow floats, SQRT to `xpow` | patterns › `patterns_*.golden` (none, all, all forced, exp2+log2, sqrt of bfloat16); patterns › a target with every operation keeps the graph; patterns › a target without the operations is left none of them |
| tinygrad: `transcendental.py` asserts `d.dtype in TRANSCENDENTAL_DTYPES`, dictionary lookups (no test) | a function of the wrong type fails | types › `<function>` refuses a node of another type; types › pow2if refuses an integer of another width; types › xpow takes the other floats |
| tinygrad: `transcendental.py` `exponent_bias` (no test) | the bias of each float, fnuz one more | bits › `exponent_biases.golden` |
| tinygrad: `transcendental.py` `shl`, `shr` (no test) | multiplication and floor division by `2^n` | bits › shl multiplies and shr floors a division by a power of two; bits › shl and shr refuse a negative count; graphs › `shifts.golden` |
| old: `unit/codegen/test_decompositions.ml` "exponent arithmetic uses promoting operations" | exp2's residual and split use promoting operations | graphs › `xexp2_*.golden` (every operation and data type) |
| old: `unit/codegen/test_decompositions.ml` "exponent masks remain weak until commitment" | log2's exponent mask is a weak constant | graphs › `xlog2_*.golden` |
| old: `unit/codegen/test_decompositions.ml` "sqrt decomposition builds Where" | SQRT becomes `xpow`, a selection at its root | patterns › `patterns_none_*.golden` |
| old: `unit/codegen/test_decompositions.ml` "xpow refuses a width without an integer" | `xpow` of a weak float fails | dropped: the old tolk's refusal; tinygrad's `xpow` has no type check. types › xpow takes the other floats |
| old: `unit/codegen/test_decompositions.ml` "POW promotes weak exponents before parity arithmetic" | POW's lowering to `xpow` casts a weak exponent | dropped: POW to `xpow` is Symbolic's rule (`uop/symbolic.py:452`), its suite; `xpow`'s graph is graphs › `xpow_*.golden` |
| old: `unit/codegen/test_decompositions.ml` "log2 denormal scale uses float power" | Float32 log2 scales subnormals by `2.0 ** 64` | graphs › `xlog2_float.golden`; special values › xlog2 of the least subnormal float is -149 |
| old: `unit/codegen/test_decompositions.ml` "sin f16 Cody-Waite casts quadrant to f32" | Float16's Cody-Waite reduction runs in Float32 | graphs › `cody_waite_half.golden`, `xsin_half.golden` |
| old: `unit/test_runtime_cpu.ml` "software sine handles large arguments and word boundaries" | compiled software sine of large arguments, across the 32-bit words of 2/π | values › `values.golden` (xsin to 1e20, 39800 and inputs drawn from every exponent); accuracy › xsin of each type; the compiled run is the executor's (L7) |
| old: `unit/test_cstyle.ml` "transcendentals" | AMD renders native sqrt and sin | dropped: the renderer's native functions, Renderer.Cstyle's suite (L5) |

## Tc

Outcomes are tests of the suite `Tolk_next.Tc`. `tensor_cores.golden` holds a
row per core of every target list, with its `repr`, types, fragments, `dims`,
`threads`, `axis_coords`, `base_upcast_axes` and both relabellings, and
`frag_coords.golden` the coordinates of each operand. tinygrad's tests of
tensor cores all apply the TC optimisation, render or run a kernel; the tables
they read are these goldens.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/test_tensor_cores.py::TestTensorCores` (23 tests: `test_tensor_cores`, `_nan`, `_emulated_half`, `_partial_sum_in_accumulator`, `_extra_locals`, `_upcast_shared_axis`, `_padto_warp`, `_group_reduce`, `_failed_padto`, `_nested_reduce`, `_contracted_m`, `_codegen`, `_padded`, `_padded_uops`, `_padto_unroll`, `_padto_masked_operand`, `_multi_reduce`, `_unroll_phi`, `_unroll_casted_phi`, `_unroll_casted_phi_with_children`, `test_tensor_core_opts`, `test_tc_shape_padded`, `test_tc_padto_full_upcast`) | the TC optimisation, its padding and its WMMA on each renderer's cores | dropped: `Codegen.Opt.Postrange` (L4), the renderers (L5) and the executor (L7); the `dims`, types and fragments they read are tensor cores › `<column>` › `tensor_cores.golden` |
| tinygrad: `null/test_custom_kernel.py::TestCustomKernel::test_loop_acc_gemm_tc_refused` | a serial loop refuses the TC optimisation | dropped: `Codegen.Opt.Postrange` (L4) |
| tinygrad: `renderer/tc.py` `pm_validate_wmma_rdna3`, `pm_validate_wmma_rdna4`, `pm_validate_wmma_cdna` (no test) | bit reinterpretations of WMMA operands for the LLVM AMD renderer | dropped: excluded with their only reader, `renderer/llvmir.py` (README exclusions) |
| tinygrad: `renderer/tc.py` `TensorCore.__post_init__` (no test) | the fragments a core refuses | v › `refused.golden` (17 cases, accepted and refused) |
| tinygrad: `renderer/tc.py` `get_cuda`, `get_amd` (no test) | the cores of an architecture | targets › `cuda.golden`, `amd.golden` |
| tinygrad: `renderer/tc.py` dataclass `repr` | a core prints as its fields | tensor cores › repr |
| old: `unit/codegen/test_tc.ml` "`<target>` tables are constructed" (8 tests) | every target list builds | targets › a target lists tinygrad's cores, in order |
| old: `unit/codegen/test_tc.ml` "rejects malformed coordinates" | a bit name without a dimension and index | dropped: a bit is a variant, so a malformed name cannot be built; a negative index is v › `refused.golden` › A names a negative bit |
| old: `unit/codegen/test_tc.ml` "rejects unequal lane counts" | lane counts differ | v › `refused.golden` › A has a lane fewer than C; B has a lane more than C |
| old: `unit/codegen/test_tc.ml` "rejects missing own coordinates" | a fragment lacks one of its bits | v › `refused.golden` › A lacks an element; A broadcasts an N bit that C lacks; B trades an N bit for an M bit |
| old: `unit/codegen/test_tc.ml` "rejects duplicate coordinates" | a fragment names a bit twice | v › `refused.golden` › A holds a bit twice; A holds a bit twice besides every bit of its tile; A holds a bit as a lane and as an element |
| old: `unit/codegen/test_tc.ml` "rejects foreign element bits" | an element of another dimension | v › `refused.golden` › A's element is an N bit; B's element is an M bit; C's element is a K bit; C's lane is a K bit |
| old: `unit/codegen/test_tc.ml` "rejects different input contraction permutations" | A and B order K differently | v › `refused.golden` › A and B order K differently |
| old: `unit/codegen/test_tc.ml` "to_string" (6 tests), "tinygrad table names" (6 tests) | `WMMA_<dims>_<in>_<out>` names | dropped: tinygrad's cores have no name; the WMMA function's name is the renderer's (`renderer/cstyle.py:114`), Renderer.Cstyle's suite. The dims and types it spelled are tensor cores › dims, dtype_in, dtype_out |
| old: `unit/codegen/test_tc.ml` "CUDA tile bits and element slots match the reference", "Metal tile bits map to SIMD fragment slots", "CDNA K128 places the high contraction bit in an element slot" | `axis_coords`, `base_upcast_axes`, `relabel` of three cores | tensor cores › axis_coords, base_upcast_axes, relabel_a, relabel_b (every core, those three included) |
| old: `unit/codegen/test_tc.ml` "table composition" (6 tests) | the length and composition of each list | targets › a target lists tinygrad's cores, in order; targets › cuda_sm80 holds every core of cuda_sm75; targets › cuda_sm89 is cuda_sm80 then two cores of 8-bit floats |
| old: `unit/codegen/test_tc.ml` "apply_tc_opt validation", "apply_tc_opt triggering", "apply_tc_opt widened operands", "apply_tc_opt padding", "apply_tc_opt WMMA construction", "apply_tc_opt with other opts" (27 tests) | the TC optimisation | dropped: `Codegen.Opt.Postrange` (L4), its suite |
| old: `unit/codegen/test_postrange.ml` "TC basic apply creates WMMA" and the five TC tests after it | the TC optimisation over `Tc.metal` | dropped: `Codegen.Opt.Postrange` (L4), its suite |
| old: `unit/test_cstyle.ml` "metal tensor cores follow Apple GPU family" | Metal's cores by GPU family; sm90 takes sm89's | targets › `cuda.golden` › arch="sm_90"; the Metal family is the renderer's choice (`renderer/cstyle.py:349`), Renderer.Cstyle's suite (L5) |
| old: `unit/test_runtime_metal.ml` "tensor cores retain warp lanes across four local dimensions" and the six tensor-core tests after it | Metal runs tensor-core kernels | dropped: the executor (L7), on Metal hardware |
| old: `parity/tc_matmul_*` (9 cases), `parity/tc_symbolic_extent` | tensor-core kernels through the pipeline, per renderer | dropped: pipeline parity cases, retired into the end-to-end suite (plan §4, kind 4) |

## Opt

The suite is `Tolk_next.Opt` (`codegen/opt/opt/`), written `P` below.

tinygrad orders two optimisations as the tuples `(op, axis, arg)`, and cannot
compare two splits of one axis and amount into different targets, a host
accident recorded in the README's CPython rows: `comparisons.golden` leaves
those pairs out, and `P › order › compare is a total order` covers them. A split that is not from
the top is `(amount, target)` in tinygrad whether or not it spells the
`False`; the heuristic never spells it, and tolk.next prints it the same way.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | `Opt.__repr__` | `P › printing › pp is tinygrad's repr` (every row); `P › printing › a split that is not from the top prints without its flag` |
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | the axis an optimisation acts on | `P › printing › axis is the axis tinygrad prints` (every row) |
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | `dataclass(order=True)` and `OptOps.__lt__` | `P › order › compare agrees with tinygrad where tinygrad orders` (every row); `P › order › compare is a total order`; `P › order › compare is 0 exactly on equal optimisations`; `P › order › splits into different targets are different` |
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | `check` and `KernelOptError` | `P › check › is unit when its condition holds`; `P › check › raises its message when its condition fails` |
| tinygrad: `runtime/test_kernel_opts.py`, `runtime/test_linearizer.py`, `null/test_linearizer.py`, `runtime/test_opt_gemm.py`, `runtime/test_tensor_cores.py`, `null/test_uops.py`, `null/test_uops_stats.py`, `null/test_gen_float4.py`, `null/test_linearizer_rewrite.py`, `runtime/test_custom_kernel.py`, `null/test_custom_kernel.py` | optimisations applied to kernels, and the `KernelOptError` of those that do not apply | Postrange's section: `apply_opt` makes the checks, `Opt` only names them |
| old: `unit/uop/test_uop.ml` debug_prints_rich_args_dataclass_style | "Opt repr" of a split into an upcast | `P › printing › pp is tinygrad's repr › reprs.golden › opt=Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))` |
| old: `unit/codegen/test_postrange.ml`, `unit/opt_fuzz/tolk_opt_fuzz.ml` | applying optimisations | Postrange's section |

## Uop_weak

The suite is `Tolk_next.Uop_weak` (`uop/uop_weak/`), written `W` below. A golden
check is named after its golden, which holds a graph and its rewrite by one
pass.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_expression_anchors_at_strong_lub | a cast commits a mixed expression at the cast's width, whatever the default float | `W › pm_commit_weak › cast_anchors_a_mixed_expression_at_the_cast.golden`; the `Tensor` half is dropped: `Tensor` surface, the frontend is nx |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_cast_weak_expression_commits_at_cast_floor | a cast below the default float never narrows | `W › pm_commit_weak › cast_never_narrows_below_the_default_float.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_store_weak_value_uses_destination_dtype | a store commits a weak value at its destination's type | `W › pm_commit_weak › store_commits_its_value_at_the_destination.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_srcs_commit_only_at_a_concrete_lub | weak sources stay weak without a committed peer; a where's weak arm stays bare | `W › pm_commit_weak › weak_sources_stay_weak_without_a_committed_peer.golden`; `W › pm_commit_weak › where_keeps_a_weak_arm_bare.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_derivable_const_rounds_at_the_derived_width | a derivable literal is rounded to its peer's width, in place | `W › pm_commit_weak › peer_rounds_a_derivable_literal.golden`; the folds `x * 1` and `x * -1` that follow are `S › tinygrad › tests.golden › TestWeakPromotion.test_derivable_const_rounds_at_the_derived_width` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_shift_lhs_commits_the_node | a shift commits its weak operand, and so the node | `W › pm_commit_weak › shift_commits_its_weak_operand.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_committed_const_conversion_folds | `symbolic_simple` folds a cast of a committed constant | `S › tinygrad › tests.golden › TestWeakPromotion.test_committed_const_conversion_folds` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_uop_scalar_const_lifts_kind, test_weak_int_binop, test_float_unary_on_weakint_stays_weak | how arithmetic on nodes promotes weak constants, and the specification of weak bitwise operations | dropped: `Ops`' and `Dtype`'s sections (construction and promotion), and Spec's (L3) |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_rand_requires_concrete, test_reduce_strips_weakness, test_broadcasted_keeps_const_weak, test_div_sub_operand_kept_weak, test_changed_rows, test_unchanged_rows, test_dot_defers_weak, test_weak_transcendentals | the data types of `Tensor` operations | dropped: `Tensor` surface, the frontend is nx; rune's lowering maps nx's types |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_null_lowering, test_computed_float_index_lowers | a realized program stores no weak type | dropped: realizing a `Tensor` on a device, the engine's (L7) |
| tinygrad: null/test_dtype_weak.py::TestWeakStorageBoundary::test_weak_has_no_storage | a `Tensor` of a weak type has no storage | dropped: `Tensor` surface; `Ops.param` and `Ops.new_buffer` refuse weak types in `Ops`' section |
| tinygrad: null/test_dtype_weak.py::TestNoRedundantWide (2 tests) | a kernel computes in 64 bits only where its bounds need it | dropped: the whole codegen pipeline (`full_rewrite`), Codegen's section (L4) |
| tinygrad: runtime/test_dtype_weak.py (every test) | weak `Tensor` values realized on a device: defaults, stacked and truncating weak casts, bounds through movements, storage | dropped: realizing `Tensor`s on a device, the engine's (L7) and rune's lowering; the conversions they check are `W › pm_lower_weak › lower_stacked_weak_casts_as_two_conversions.golden`, `W › pm_lower_weak › lower_a_weak_int_cast_of_a_float_as_a_conversion.golden` and `W › laws › pm_lower_weak keeps the values of what it lowers` |
| tinygrad: null/test_uops.py::TestLowerIndexDtype::test_gated_shrink_lowers_to_selected_width | a gated shrink lowers to the width its bounds need, and no weak width is left but a literal's | `W › pm_lower_weak › lower_a_gated_shrink_to_the_width_its_bounds_need.golden`; `W › laws › pm_lower_weak leaves no weak width but a literal's` |
| tinygrad: null/test_uops.py::TestLowerIndexDtype::test_reg_buffer_size_lowers | a register buffer's size lowers | `W › pm_lower_weak › lower_a_register_buffer_size.golden` |
| tinygrad: null/test_simplify_valid_idx.py::TestImageSimplification::test_drop_gate_committed_in_the_index_pass | committing inside the index pass leaves the gate's copy of an expression shared | dropped: runs with `indexing_simplify` on image loads, Codegen's section (L4) |
| old: unit/uop/test_weak.ml unconstrained_int_const_commits_at_int32 | an unconstrained integer lowers to int32 | `W › pm_lower_weak › lower_an_int_to_int32.golden` |
| old: unit/uop/test_weak.ml unconstrained_overflowing_const_commits_at_int64 | an integer beyond int32 lowers to int64 | `W › pm_lower_weak › lower_an_int_beyond_int32_to_int64.golden` |
| old: unit/uop/test_weak.ml unconstrained_float_const_commits_at_default_float | a weak float lowers to the default float | `W › pm_lower_weak › lower_a_float_to_the_default_float.golden` |
| old: unit/uop/test_weak.ml uint64_width_and_unrepresentable_const | an integer beyond int64 lowers to uint64; one no 64-bit integer holds is refused | `W › pm_lower_weak › lower_an_int_beyond_int64_to_uint64.golden`; `W › pm_lower_weak › an int no 64-bit integer holds cannot be lowered` |
| old: unit/uop/test_weak.ml peer_commits_weak_const_at_its_own_width | a derivable literal stays bare beside a committed peer | `W › pm_commit_weak › peer_keeps_a_derivable_literal_bare.golden` |
| old: unit/uop/test_weak.ml peer_commits_weak_alu_by_cast | a weak expression commits at its peer's width | `W › pm_commit_weak › peer_commits_a_weak_expression.golden` |
| old: unit/uop/test_weak.ml all_weak_sources_stay_weak | nothing commits without a committed peer | `W › pm_commit_weak › weak_sources_stay_weak_without_a_committed_peer.golden` |
| old: unit/uop/test_weak.ml store_commits_value_at_destination_dtype | a store commits its value | `W › pm_commit_weak › store_commits_its_value_at_the_destination.golden` |
| old: unit/uop/test_weak.ml consumer_cast_widens | a cast widens a weak expression | `W › pm_commit_weak › cast_widens_a_weak_expression.golden` |
| old: unit/uop/test_weak.ml consumer_cast_never_narrows | a cast never narrows below the bounds | `W › pm_commit_weak › cast_never_narrows_below_the_bounds.golden` |
| old: unit/uop/test_weak.ml cast_preserves_operand_widths | a cast commits a division's operands at their own bounds | `W › pm_commit_weak › cast_commits_operands_at_their_own_bounds.golden` |
| old: unit/uop/test_weak.ml consecutive_weak_casts_preserve_integer_conversion | two stacked weak casts are two conversions | `W › pm_lower_weak › lower_stacked_weak_casts_as_two_conversions.golden`; `W › laws › pm_lower_weak keeps the values of what it lowers`; the folded value is `S › symbolic_simple › constants › a cast of a constant is the constant of the cast's type` |
| old: unit/uop/test_weak.ml weak_integer_cast_is_a_value_conversion | a weak integer cast of a float converts its value | `W › pm_lower_weak › lower_a_weak_int_cast_of_a_float_as_a_conversion.golden`; `W › pm_lower_weak › lower_a_weak_int_cast_of_a_bool_as_a_conversion.golden` |
| old: unit/uop/test_weak.ml range_arithmetic_lowers_to_concrete_int | range arithmetic lowers to int32 | `W › pm_lower_weak › lower_range_arithmetic.golden`; `W › laws › pm_lower_weak leaves no weak width but a literal's` |
| old: unit/uop/test_weak.ml comparison_unifies_operand_widths | a comparison's operands lower to one width | `W › pm_lower_weak › lower_a_comparison.golden` |
| old: unit/uop/test_weak.ml gated_long_index_narrows_for_small_buffers | a gated long index into a small buffer narrows | `W › pm_lower_weak › lower_a_gated_long_index_into_a_small_buffer_to_int32.golden` |
| old: unit/uop/test_weak.ml gated_long_index_keeps_wide_storage | a gated long index into a huge buffer keeps int64 | `W › pm_lower_weak › lower_a_gated_long_index_into_a_huge_buffer_keeps_int64.golden` |
| old: unit/uop/test_weak.ml uncast_preserves_operand_and_result_types | a committed literal loses its cast only where the node derives the same types | `W › pm_uncast_const › uncast_a_committed_literal.golden`; `W › pm_uncast_const › uncast_keeps_literals_with_no_committed_peer.golden`; `W › pm_uncast_const › uncast_keeps_a_shifted_literal.golden` |
| old: unit/uop/test_weak.ml uncast_keeps_a_wrapping_cast | a cast that changes the literal's value stays | `Tolk_next.Symbolic › symbolic_simple › constants › a comparison reads a committed constant at its width` (uncast, the literal is read at its width: D13); `W › pm_uncast_const › uncast_drops_a_cast_that_fits.golden` |
| old: unit/uop/test_weak.ml final_constants_state_width_on_each_edge | every constant gets its width at each consumer, booleans included, and the result is stable | `W › pm_cast_const › cast_consts_state_each_edge_width.golden`; `W › laws › pm_cast_const states the width of every constant`; `W › laws › pm_cast_const is idempotent` |
| old: unit/uop/test_weak.ml late_simplification_preserves_committed_literals | `symbolic_simple` keeps an emulated operand's width, `symbolic` exposes literals | `S › tinygrad › tests.golden › TestWeakPromotion.test_committed_const_conversion_folds`; `S › symbolic_simple › constants › a cast of a constant is the constant of the cast's type` |
| old: parity/weak_movement_width | a weak product beyond int32, reshaped and permuted, compiles for CPU and CUDA | dropped: an end-to-end parity case of the codegen stages, Codegen's section (L4) |

## Movement

The suite is `Tolk_next.Movement` (`uop/movement/`), written `M` below. A
golden check is named after its golden, which holds a graph and its cleanup. A
test of offsets that are nodes simplifies them with Symbolic's rules.

tinygrad tests `mop_cleanup` only through the passes that include it
(`symbolic_simple`, `earliest_rewrites`, the codegen pipeline and the jit's
input capture); those tests belong to their passes' sections.

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/uop/test_symbolic.ml "adjacent SHRINKs compose every axis offset" | two shrinks merge, their starts summed on each axis | `M › shrinks › merge_two_shrinks.golden`; `M › shrinks › merge_three_shrinks.golden`; `M › laws › mop_cleanup keeps the elements a chain denotes` |
| old: unit/uop/test_symbolic.ml "adjacent SHRINKs retain symbolic offsets and sizes" | shrinks with node starts and sizes merge | `M › shrinks › merge_shrinks_of_symbolic_starts.golden` |
| old: unit/uop/test_symbolic.ml "adjacent SHRINKs promote mixed offset widths" | starts of two committed widths are summed at the wider | `M › shrinks › merge_shrinks_of_symbolic_starts.golden`; the sum of two starts is `Ops`' arithmetic, whose promotion `Ops`' section pins |
| old: unit/uop/test_symbolic.ml "adjacent scalar SHRINKs return the scalar" | two shrinks of a scalar are the scalar | dropped: a shrink of a scalar has no bounds, and tinygrad builds none (`_mop` returns the scalar); a bare one fails in `marg` |
| old: unit/uop/test_symbolic.ml "INDEX on INDEX chains scalar coordinates" | an index of an index by scalars is one index | `M › indexing › index_an_index_by_scalars.golden` |
| old: unit/uop/test_symbolic.ml index_stack_const_folds | a stack indexed by a constant is its source | `M › indexing › index_a_stack_by_a_constant.golden` |

## Divandmod

The suite is `Tolk_next.Divandmod` (`uop/divandmod/`), written `DM` below.
Each golden holds divisions and remainders, each followed by its rewrite by
`div_and_mod_symbolic` applied once, or by itself where no rule applies. A
golden named after a claim holds one case, built in the suite. A golden
`test_<name>.golden` holds every division and remainder of the tinygrad test of
that name, which `gen/uop/divandmod.py` records while the test runs.

tinygrad's symbolic tests assert what the whole rule set makes of an
expression (`sym`, `simplify`): those claims are Symbolic's section. Here each
division and remainder they build is rewritten by this matcher alone.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_uop_symbolic.py (every test that builds a division or a remainder, one golden each: TestSymbolic, TestSymbolicNumeric, TestSymInfer, TestSymbolicRealWorld, TestFuzzFailure, TestInvalidIndex, TestGatedUopGivenValid, TestRangeSplitting) | the rewrite of each division and remainder the test builds | `DM › tinygrad's tests › test_<name>.golden`; `DM › values › each rewrite of the goldens' divisions keeps the division's value` |
| tinygrad: null/test_uop_symbolic.py::TestSymbolic::test_div_mod_zero | a division or remainder by 0 raises | `DM › zero divisors` (4 tests) |
| tinygrad: null/test_uop_symbolic.py::TestSymbolic::test_variable_divmod | a variable bounded by another | dropped: a variable's bounds are values (`Ops.variable`); tinygrad's bounds arithmetic refuses a bound that is a node too |
| tinygrad: null/test_uop_symbolic.py (the tests that build no division or remainder) | the other symbolic rules | `S › tinygrad › tests.golden › <Class>.<test>` (Symbolic's section) |
| tinygrad: null/test_symbolic_failures.py::TestFuzzFailure (11 tests) | simplifying an expression keeps its value | `DM › tinygrad's tests › test_fuzz_failure<n>.golden` (test_fuzz_failure1 is test_uop_symbolic.py's too); `DM › values › each rewrite of the goldens' divisions keeps the division's value`; the claim on the whole rule set is `S › tinygrad › tests.golden › TestFuzzFailure.test_fuzz_failure<n>` |
| tinygrad: null/test_simplify_valid_idx.py (44 tests) | valid and index simplification | `S › tinygrad › tests.golden › TestValidIdxSimplification.<test>` (Symbolic's section); the image tests and `indexing_simplify` are dropped there |
| old: unit/codegen/test_divandmod.ml positive_floor_div_does_not_rewrite_without_structure | a plain division stays | `DM › constant divisors › keep_a_plain_division.golden`; `DM › constant divisors › keep_a_plain_remainder.golden` |
| old: unit/codegen/test_divandmod.ml nested_div_fires | `(x // c + a) // d` merges | `DM › nested divisions › merge_nested_divisions.golden` |
| old: unit/codegen/test_divandmod.ml nested_div_accepts_negative_inner_divisor | it merges for a negative inner divisor | `DM › nested divisions › merge_nested_divisions_by_a_negative_inner_divisor.golden` |
| old: unit/codegen/test_divandmod.ml add_const_div_fires_for_negative_constant | the constant split holds for a negative constant | `DM › constant terms › split_a_negative_constant_out_of_a_division.golden` |
| old: unit/codegen/test_divandmod.ml add_const_mod_splits_the_constant | `(x + c) % d` splits its constant | `DM › constant terms › split_the_constant_out_of_a_remainder.golden` |
| old: unit/codegen/test_divandmod.ml add_const_div_fires_for_negative_divisor | the constant split holds for a negative divisor | `DM › constant terms › split_the_constant_out_of_a_division_by_a_negative_divisor.golden`; `DM › nested divisions › split_nested_divisions_by_a_negative_divisor.golden` |
| old: unit/codegen/test_divandmod.ml remove_nested_floormod_fires | a nested remainder in a sum drops | `DM › constant divisors › drop_a_nested_remainder_from_a_sum.golden` |
| old: unit/codegen/test_divandmod.ml crossing_denominator_does_not_fold_zero_singleton | 0 divided by a divisor that can be 0 stays | `DM › other divisors › divide_zero_by_a_divisor_that_can_be_zero.golden` |
| old: unit/codegen/test_divandmod.ml zero_denominator_raises_before_sentinel_bailout | a division by 0 raises whatever the numerator's bounds | `DM › zero divisors › a division of any integer by 0 raises Division_by_zero` |
| old: unit/codegen/test_divandmod.ml singleton_quotient_floordiv_folds, singleton_quotient_floormod_folds | one possible quotient folds | `DM › one quotient › fold_a_division_with_one_quotient.golden`; `DM › one quotient › fold_a_remainder_with_one_quotient.golden`; `DM › values` |
| old: unit/codegen/test_divandmod.ml cancel_one_sided_bounded_divisor_div, cancel_one_sided_bounded_divisor_mod | one quotient by a variable folds | `DM › one quotient › fold_a_division_by_a_variable_with_one_quotient.golden`; `DM › one quotient › fold_a_remainder_by_a_variable_with_one_quotient.golden` |
| old: unit/codegen/test_divandmod.ml nested_single_term_mod_folds | `(a % 12) % 3` drops the inner remainder | `DM › constant divisors › drop_a_nested_remainder.golden` |
| old: unit/codegen/test_divandmod.ml nested_single_term_div_folds | `(a % 12) // 3` is `(a // 3) % 4` | `DM › constant divisors › nest_the_division_of_a_remainder.golden` |
| old: unit/codegen/test_divandmod.ml symbolic_gcd_divides_variable_denominator_div, symbolic_gcd_divides_variable_denominator_mod | a common node factor divides out | `DM › tinygrad's tests › test_symbolic_gcd_div.golden`; `S › tinygrad › tests.golden › TestSymbolic.test_symbolic_gcd_div` |
| old: unit/codegen/test_divandmod.ml symbolic_gcd_divides_mixed_constant_factor | a common constant factor of a node divisor divides out | `DM › other divisors › divide_a_common_divisor_out_of_a_division_by_a_variable.golden`; `DM › other divisors › divide_a_common_divisor_out_of_a_remainder_by_a_variable.golden` |
| old: unit/codegen/test_divandmod.ml factor_remainder_rejects_negative_denominator_range_for_div, factor_remainder_rejects_negative_denominator_range_for_mod | a divisor that can be negative keeps its multiples | `DM › other divisors › keep_a_division_by_a_variable_that_can_be_negative.golden`; `DM › other divisors › keep_a_remainder_by_a_variable_that_can_be_negative.golden` |
| old: unit/codegen/test_divandmod.ml factor_remainder_still_accepts_positive_denominator_range | the multiples of a positive divisor leave | `DM › other divisors › take_the_multiples_of_a_variable_divisor_out_of_a_division.golden`; `DM › other divisors › take_the_multiples_of_a_variable_divisor_out_of_a_remainder.golden` |
| old: unit/codegen/test_divandmod.ml factor_remainder_floormod_splits_constant_factor_without_exact_quotient, factor_remainder_preserves_remainder_order | a constant factor splits out of a remainder, its terms in order | `DM › constant divisors › split_a_constant_factor_out_of_a_remainder.golden` |
| old: unit/codegen/test_divandmod.ml factor_remainder_floormod_splits_multiple_constant_factors | several constant factors split out | `DM › constant divisors › split_constant_factors_out_of_a_remainder.golden` |
| old: unit/codegen/test_divandmod.ml large_constant_residue_double_does_not_overflow_rewrite | huge coefficients stay exact | `DM › constant divisors › keep_a_division_of_huge_coefficients_exact.golden` |
| old: unit/codegen/test_divandmod.ml nest_by_factor_divides_the_common_factor | `(2a + 3) // 4` divides the common factor out | `DM › constant divisors › divide_a_common_factor_out_of_a_division.golden`; `DM › constant divisors › nest_a_division_by_a_factor_of_a_term.golden` |
| old: unit/codegen/test_divandmod.ml property_folds_are_numerically_correct | every fold keeps the value | `DM › values › each rewrite of a random division keeps its value`; `DM › values › each rewrite of the goldens' divisions keeps the division's value` |
| old: unit/codegen/test_divandmod.ml param_multiple_of_folds_mod_and_leaves_div | a declared multiple's remainder is 0 and its division stays | `DM › declared multiples` (4 goldens) |
| old: unit/codegen/test_divandmod.ml param_without_multiple_of_does_not_fold | an undeclared multiple's remainder stays | `DM › constant divisors › keep_a_plain_remainder.golden` |
| old: unit/codegen/test_divandmod.ml simplify_preserves_index_values, adjacent_bit_extracts_recombine, quotient_partner_recombines_through_a_merged_divisor, shifted_quotient_partner_recombines | the whole rule set keeps values and recombines quotients with remainders | `S › laws › sym keeps the value of an integer expression where nothing wraps`; `S › symbolic_simple › recombination` (every test) |
| old: unit/uop/test_symbolic.ml divandmod_tests (2 tests) | `Cdiv` and `Cmod` of a range by its size | dropped: tinygrad HEAD's rules fold floor division only (Symbolic's section) |
| old: unit/uop/test_symbolic.ml "nested division commits newly built weak arithmetic" | merging nested divisions of committed integers casts the new constants | `DM › nested divisions › merge_nested_divisions_of_committed_integers.golden` |
| old: unit/codegen/test_decompositions.ml early_floordiv_by_zero_raises, early_floormod_by_zero_raises | a division or remainder by 0 raises in the early rewrites | `DM › zero divisors › a division by the constant 0 raises Division_by_zero`; `DM › zero divisors › a remainder by the constant 0 raises Division_by_zero`; the early rewrites are Codegen's (L4) |

## Render

A test that renders after simplifying does so with Symbolic's rules.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_uops.py::TestUOpRender::test_render_ssimplified_marg_outside_toposort` | a movement's size, simplified to a node the graph never held, is still written, and a range after it is its name | `Tolk_next.Render › render after simplifying › symbolic_rendered.golden › case=shrink_of_simplified_offset`, `case=range_after_shrink` |
| tinygrad: `null/test_uops.py::TestUOpRender::test_render_vectorize_empty`, `test_render_vectorize_same`, `test_render_vectorize_different` | a stack is its sources between braces, `{}` when empty | `Tolk_next.Render › render › expressions_rendered.golden › case=stack_empty`, `case=stack_same`, `case=stack_different` |
| tinygrad: `null/test_uops.py::TestUOpRender::test_render_vectorize_empty_simplified`, `test_render_vectorize_same_simplified`, `test_render_vectorize_different_simplified` | the same after simplifying | `Tolk_next.Render › render after simplifying › expressions_rendered.golden` (the `simplified` column) |
| tinygrad: `null/test_helpers.py::TestProd::test_variable`, `test_variable_order` | the product of a variable and integers writes as `(a*12)`, whatever the order | `Tolk_next.Render › srender › writes a node as render does`; the product itself belongs to `Ops` |
| tinygrad: `null/test_helpers.py::TestCeilDiv::test_symbolic`, `test_symbolic_negative_offset` | `ceildiv` of a node writes as `((v+5)//6)` | dropped here: the claim is `ceildiv`'s simplification, which `Ops`' suite owns |
| tinygrad: `null/test_uop_symbolic.py` (`helper_test_variable` and the eight `.render()` asserts), `null/test_simplify_valid_idx.py` (the `.render()` asserts) | a rewrite gives the expected expression | dropped here: they test the symbolic rules, with `render` as the printer: `S › tinygrad › tests.golden › <Class>.<test>` |
| tinygrad: `null/test_uop_graph.py` (`print(sink.render())`) | none, a debug print | dropped: no claim |
| tinygrad: `null/test_viz.py` (`ret.render()` in `rewrite_group` names) | none about `render` | dropped: viz is excluded |
| tinygrad: `null/test_uops.py`, `null/test_encodings.py`, `null/test_renderer_failures.py` (`renderer.render(uops)`) | a device renderer's source | dropped here: those are `Renderer.render`, not `uop/render.py`; each belongs to its renderer's section |
| tinygrad: `uop/render.py` `renderer` (no test file covers every rule) | every rule: storage, ranges, loops, constants, casts of constants, casts, the unary and ternary forms, movements, the twelve infix operations and their precedence, indices, stages, loads, stacks, and the repr of any other node | `Tolk_next.Render › render › expressions_rendered.golden` (145 cases), `unrendered_rendered.golden`; `an expression reads back as the tree it writes` (the parenthesis law); `tags are not written` |
| tinygrad: `uop/render.py` `print_uops` (no test file) | the listing's columns, the sources as positions, quoted constants or `--`, the argument as `str` prints it, colour, and columns that overflow | `Tolk_next.Render › pp_uops › program_listing.golden`, `program_listing_colored.golden`, `partial_listing.golden`, `wide_listing.golden`, `wide_listing_colored.golden`, `an empty list prints nothing`, `prints a line per node, numbered from 0, the last not ended` |
| tinygrad: `uop/render.py` `pretty_print` | a node's repr | dropped here: `Ops.pp`, in `Ops`' suite |
| tinygrad: `uop/render.py` `renderer_infer`, `pyrender` | Python source | dropped: excluded (README Exclusions) |
| old: `test/unit/uop/test_uop.ml` "committed constants render their value" | a cast of a constant is its value | `Tolk_next.Render › render › expressions_rendered.golden › case=typed_int` and the other `typed_*` and `forced_cast_*` cases |
| old: `test/unit/uop/test_uop.ml` "nonconstant casts retain their width" | `(int)(x)` | `Tolk_next.Render › render › expressions_rendered.golden › case=cast_to_int` and the other `cast_to_*` cases |
| old: `test/unit/uop/test_uop.ml` `debug_prints_toposort_like_tinygrad` | the listing of a toposort, with constants as quoted sources | `Tolk_next.Render › pp_uops › program_listing.golden` |
| old: `test/unit/uop/test_uop.ml` `debug_prints_ranges_and_supplied_list_sources` | a list that leaves sources out prints `--` | `Tolk_next.Render › pp_uops › partial_listing.golden` |
| old: `test/unit/uop/test_uop.ml` `debug_prints_tinygrad_dtype_reprs` | the type column prints `dtypes.weakint`, `dtypes.float`, `dtypes.long` | `Tolk_next.Render › pp_uops › program_listing.golden`, `partial_listing.golden` |
| old: `test/unit/uop/test_uop.ml` `debug_prints_float_and_special_args_like_tinygrad` | a float constant's argument keeps `.0`; a special's name is not quoted | `Tolk_next.Render › pp_uops › program_listing.golden` (`1.5`, `0.0`, `gidx0`), `wide_listing.golden` (`1e+16`, `-0.0`, `nan`) |
| old: `test/unit/uop/test_uop.ml` `debug_prints_direct_string_args_like_tinygrad` | source text and a copy's device print without quotes | `Tolk_next.Render › pp_uops › wide_listing.golden` |
| old: `test/unit/uop/test_uop.ml` `debug_prints_ranges_in_tinygrad_arg_order` | the ranges column sorts by argument | `Tolk_next.Render › pp_uops › wide_listing.golden` (`m1,0_1,10,11,12,13`) |
| old: `test/unit/uop/test_uop.ml` `debug_prints_rich_args_dataclass_style` | `ParamArg`, `KernelInfo`, `Opt`, `Estimates`, `ProgramInfo`, `BufferizeOpts`, `CallInfo` reprs in a listing | `Tolk_next.Render › pp_uops › program_listing.golden`, `wide_listing.golden` (`ParamArg`, `KernelInfo`); the reprs themselves are `Ops.pp_arg`'s, in `Ops`' suite |
| old: `test/unit/uop/test_uop.ml` `debug_prints_reduce_arg_tuple` | a reduction's argument prints `(Ops.ADD, 0)` | `Tolk_next.Render › pp_uops › wide_listing.golden`; `(Ops.ADD, 1)` is `Ops.pp_arg`'s, in `Ops`' suite |
| old: `test/unit/uop/test_uop.ml` `debug_print_ignores_side_metadata` | metadata is not printed | dropped: `Ops` has no metadata table |
| old: `test/unit/uop/test_uop.ml` `debug_listing_omits_tags` | tags are not printed | `Tolk_next.Render › pp_uops › wide_listing.golden` (the constant tagged `hidden`); `Tolk_next.Render › render › tags are not written` |
| old: `lib/uop/render.mli` `uops_to_string ?label` | a `=== label ===` header | dropped: not in tinygrad, and no reader |
| old: `lib/uop/render.mli` `python_float_string`, `compare_uops` | CPython's float repr; the structural order | dropped here: `Dtype.pp_const` and `Ops.compare_structure`, in their modules' suites |
| old: `test/parity/helpers.ml` (`uops_to_string` listings) | parity cases compared as listings | dropped: graph goldens use the graph format, which keeps what a listing drops (test/README.md) |
| old: `test/unit/codegen/test_linearizer.ml` (`Render.pp_uops` as a failure printer) | none | dropped: no claim |

## Renderer

The suite is `Tolk_next.Renderer` (`renderer/renderer/`), written `R` below.
`kernels.golden` holds the kernels of `null/test_uops_stats.py`, linearized as
`to_program` linearizes them, and `estimates.golden` what `Estimates.from_uops`
gives for each, with and without `ignore_indexing`; `symbolic_kernels.golden`
and `symbolic_estimates.golden` hold kernels over a variable and their
estimates at three of its values.

A kernel's trip counts are typed constants, casts of literals, which the
symbolic rules fold. The tests tagged `assert-compile` run in
a second process with `ASSERT_COMPILE=1`, since a process reads the variable
once; the default process runs with `ASSERT_COMPILE=0`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_uops_stats.py::TestMemoryCount::test_add`, `test_add_const`, `test_expanded`, `test_self_add`, `test_self_add_transposed`, `test_self_add_assign` | `mem` counts each buffer once, a broadcast read at its own size | `R › estimates of recorded kernels › estimates.golden › case=add_uint8`, `add_const_uint8`, `add_expanded_uint8`, `self_add_uint8`, `self_add_transposed_uint8`, `self_add_assign_uint8` |
| tinygrad: `null/test_uops_stats.py::TestMemoryCount::test_add_slice`, `test_both_expanded` | none | dropped: skipped in tinygrad ("depends on subbuffer working", "no longer supported") |
| tinygrad: `null/test_uops_stats.py::TestMemoryCount::test_copyout` | a copy's estimate | dropped: `estimate_uop` of a copy is `engine/realize.py`'s (L7), not `from_uops` |
| tinygrad: `null/test_uops_stats.py::TestUOpsStatsMatmulHalf` (3 tests) | `GlobalCounters.global_ops` of emulated tensor-core matmuls | dropped: counters of a run on the PYTHON device, the executor's (L7); the count of a tensor core product is `R › estimates of hand-written kernels › counts a tensor core product as 2NMK shared among its threads` and `estimates.golden › case=gemm_tc_half` |
| tinygrad: `null/test_uops_stats.py::TestUOpsStats::test_isa_store_estimate` | a store of one int32 counts 4 bytes in `lds` and `mem` | `R › estimates of hand-written kernels › counts the bytes a store writes` (the X86 renderer is excluded; its estimate is `from_uops` before instruction selection) |
| tinygrad: `null/test_uops_stats.py::TestUOpsStats::test_simple_add`, `test_simple_add_sq`, `test_cat_equal_pieces`, `test_simple_matmul`, `test_simple_matmul_8192` | `ops` and `mem` of elementwise, concatenation and matmul kernels | `R › estimates of recorded kernels › estimates.golden › case=simple_add`, `simple_add_sq`, `cat_equal_pieces`, `cat_unequal_pieces`, `simple_matmul`, `simple_matmul_8192` |
| tinygrad: `null/test_uops_stats.py::TestUOpsStats::test_mulacc` | a `MULACC` has the stats of a `MUL` and an `ADD` | `R › estimates of hand-written kernels › counts a multiply-add as a multiply and an add` |
| tinygrad: `null/test_uops_stats.py::TestStatsOptimized::test_gemm`, `test_gemm_one_upcasted`, `test_gemm_upcasted`, `test_gemm_upcasted_locals`, `test_gemm_group`, `test_reduce` | `ops`, `mem` and `lds` of a 64x64 gemm under optimisations, and of a sum | `R › estimates of recorded kernels › estimates.golden › case=gemm`, `gemm_one_upcasted`, `gemm_upcasted`, `gemm_upcasted_locals`, `gemm_group`, `reduce` (the kernels with locals on CUDA) |
| tinygrad: `null/test_uops_stats.py::TestStatsOptimized::test_gemm_tc_unroll`, `test_gemm_tc_unroll_half` | a tensor-core gemm | `R › estimates of recorded kernels › estimates.golden › case=gemm_tc_half`; the float gemm refuses the TC optimisation on sm_80 without TF32 and tinygrad skips it, and the half one is skipped in tinygrad |
| tinygrad: `null/test_device.py::TestCompiler::test_compile_cached` | a miss compiles and fills the disk cache | `R › Compiler › compile_cached compiles a source once and keeps its binary` |
| tinygrad: `null/test_device.py::TestCompiler::test_compile_cached_disabled` | with `CCACHE=0` nothing is cached | `R › Compiler › compile_cached compiles every time with ccache off when made` |
| tinygrad: `null/test_device.py::TestCompiler::test_device_compile` | a device compiles with `CCACHE=0` | dropped: realizes on a device, the executor's (L7) |
| tinygrad: `null/test_method_cache.py` (`compiler.compile_cached = None`) | the method cache avoids compiling again | dropped: `engine/realize.py`'s cache (L7) |
| tinygrad: `device/metal/test_metal.py`, `device/amd/test_llvm.py` (`CompileError`) | a toolchain rejects bad source | dropped here: the Metal and AMD compilers' sections (L5, L7); `R › Compiler › compile raises what the toolchain rejects` pins the propagation |
| tinygrad: `device/cl/test_ocl.py::TestCLProgram::test_compile_cached` | a hit does not compile | dropped: OpenCL is excluded; `R › Compiler › compile_cached returns the binary the table holds` pins the hit |
| tinygrad: `null/*`, `runtime/*` (28 files: `supported_dtypes()` in skip conditions) | none | dropped: skip conditions, no claim |
| tinygrad: `renderer/__init__.py` `Renderer.supported_dtypes` (no test) | every data type, without double when long is emulated | `R › supported_dtypes › supported_dtypes.golden` (7 settings: none, `long`, `int64`, `double`, `half,long`, `ulong`, `,long,`); `keeps only native data types, in order`; `keeps double when the target lacks long natively`; `rejects an emulated name that is no data type` |
| tinygrad: `renderer/__init__.py` `with_storage` (no test; used by `ptx.py`, `nir.py`) | an access restated at another type retypes its storage | `R › with_storage › restated.golden` (8 accesses: of a parameter, gated, loaded, after a store, of a buffer, of local and register storage, at its own type); laws: restates the storage and keeps every other source, identity at the storage's type, round trip, idempotent; `rejects a node whose first sources reach no storage` |
| tinygrad: `renderer/__init__.py` `Estimates.__add__`, `simplify`, and `from_uops` branches with no test (loops, `SPECIAL`, registers, the `END` gate of `ignore_indexing`) | the arithmetic of estimates, and each rule of the count | `R › add and zero` (laws: associative, commutative, neutral zero); `R › simplify` (value kept, idempotent); `R › estimates of hand-written kernels` (one test per rule) |
| tinygrad: `renderer/__init__.py` `Renderer` class attributes, `render`, `Compiler()` | the base renderer's defaults | `R › v › defaults describe a target that renders nothing`, `render raises by default`, `the default compiler returns its source and caches nothing`, `native holds for every data type by default`, `keeps the fields it is given` |
| tinygrad: `device.py` `Compiler.compile_cached` `ASSERT_COMPILE` (no test) | a miss is refused under `ASSERT_COMPILE` | `R › ASSERT_COMPILE` (5 tests: refused naming the source, refused without a cachekey, a held binary returned, refused while the disk cache is disabled, `compile` unaffected); `R › Compiler › compile_cached compiles while ASSERT_COMPILE holds 0` |
| tinygrad: `renderer/__init__.py` `Renderer.asm`, `__reduce__`; `device.py` `Compiler.server`, `compile_server` | assembly, pickling, a compile server | dropped: the ISA path is excluded, OCaml does not pickle, and compilation workers are domains (D5) |
| old: `unit/test_program_spec.ml` "Estimates.of_program" › "counts basic ALU ops" | two ALU ops count two | `R › estimates of hand-written kernels › counts each arithmetic operation once` |
| old: `unit/test_program_spec.ml` "mulacc counts as 2 FLOPs" | | `R › estimates of hand-written kernels › counts a multiply-add as a multiply and an add` |
| old: `unit/test_program_spec.ml` "wmma counts 2*M*N*K per warp, divided across threads", "wmma thread count divides the FLOP factor" | | `R › estimates of hand-written kernels › counts a tensor core product as 2NMK shared among its threads` (32 and 64 threads) |
| old: `unit/test_program_spec.ml` "wmma FLOPs scale with the loop multiplier", "loop multiplier stacks" | | `R › estimates of hand-written kernels › counts the operations of a range once per iteration`, `multiplies the trip counts of nested ranges` |
| old: `unit/test_program_spec.ml` "an unbounded loop contributes no multiplier" | | `R › estimates of hand-written kernels › counts a loop without trip count as one iteration` |
| old: `unit/test_program_spec.ml` "a loop bounded by a loaded value counts at its bound" | the count takes the bound of a loaded trip count | dropped: tinygrad keeps such a trip count symbolic (`mults *= u.src[0].ssimplify()`), and the old count was a divergence without an entry |
| old: `unit/test_program_spec.ml` "special multiplier stacks" | | `R › estimates of hand-written kernels › multiplies everything after a hardware index by its size` |
| old: `unit/test_program_spec.ml` "load/store tracks lds and memory bytes" | | `R › estimates of hand-written kernels › counts the loads and the stores of a buffer apart in mem`, `counts the bytes a store writes` |
| old: `unit/test_program_spec.ml` "index arithmetic excluded from FLOPs" | | `R › estimates of hand-written kernels › ignore_indexing leaves out the operations of indices` (the old walk always ignored indexing; `from_uops` does so under `ignore_indexing`) |
| old: `unit/test_program_spec.ml` "repeated reads cap memory at buffer size" | | `R › estimates of hand-written kernels › counts every read in lds and a buffer read again once in mem` |
| old: `unit/test_program_spec.ml` "Symbolic estimates" › "every operation contributes its symbolic loop count" | | `R › estimates of symbolic kernels › a symbolic trip count counts at the value of its variable` |
| old: `unit/test_program_spec.ml` "adding estimates preserves equal symbolic contributions" | | `R › add and zero › add sums symbolic counts` |
| old: `unit/test_program_spec.ml` "final FLOPs simplify a cancelling symbolic loop bound" | | `R › estimates of symbolic kernels › a trip count that simplifies to an integer counts as one` |
| old: `unit/test_program_spec.ml` "Exact estimates" › "concrete sums retain values beyond a host integer", "nested loop multiplicities retain their exact product", "memory footprints and loop traffic retain exact byte counts", "WMMA division follows the exact numerator product", "loaded trip bounds retain the full scalar width" | counts past `max_int` stay exact | dropped: counts are `Ops.sint`, whose arithmetic raises past `int` (Ops' suite); `R › add and zero › add past max_int raises rather than wrapping` pins that nothing wraps |
| old: `unit/test_program_spec.ml` "symbolic traffic is capped at the buffer footprint" | | `R › estimates of symbolic kernels › reads are capped at the buffer for each value of a variable` |
| old: `unit/test_program_spec.ml` "exact estimates can be forwarded", "symbolic estimates require caller handling" | `Program_spec.Estimates.of_uop` | dropped: `Program_spec` has no counterpart; estimates are `Ops.estimates` |
| old: `unit/test_diskcache.ml` "compiler cache policy follows nested contexts and worker snapshots" | `CCACHE` read when a compiler is made, `CACHELEVEL=0` compiles without caching | `R › Compiler › ccache is read when the compiler is made`, `compile_cached compiles every time with ccache off when made`, `compile_cached compiles every time with the disk cache disabled`; worker snapshots are gone with Helpers' contexts (Helpers' section) |
| old: `unit/test_diskcache.ml` "assert compile permits hits and blocks every cache miss" | | `R › ASSERT_COMPILE` (the whole group); the old refusal raised `Compile_error`, tinygrad asserts, and tolk.next raises `Invalid_argument` |
| old: `unit/runtime/cpu/test_compiler.ml` (5 tests) | Clang's output and errors | dropped here: `Compiler_cpu`'s section (L5) |
| old: `unit/test_cstyle.ml` (`Renderer.supports_dtype Cstyle.qcom`) | a target's data types | dropped here: `Renderer.Cstyle`'s section (L5); QCOM is excluded |
| old: `unit/test_runtime_cpu.ml` `test_emulated_long_buffer_arithmetic`, `test_emulated_long_division`, `test_emulated_compact_float_storage` (`~supports_dtype`) | emulated types in generated code | dropped here: the decompositions' sections (L4) and CPU execution (L5) |
| old: `lib/renderer.mli` `supported_ops`, `all_supported_ops`, `emulated_float_dtypes`, `image_pitch_alignment`, `with_target`, `with_compiler` | | dropped: not in tinygrad's `Renderer`; images are excluded |

## Decomp_op

The suite is `Tolk_next.Decomp_op` (`codegen/decomp/decomp_op/`), written `DO`
below. A graph golden holds the sink of its inputs and that sink rewritten to a
fixed point by one matcher alone. `fast_idiv_grid.golden` holds tinygrad's
`fast_idiv` of every integer type over 13 bounds and 28 divisors (the uint64
edges, 2^64 and 2^70 + 1 included), with and without a wider type, and the
laws check every quotient against `Z.div` in the interpreter.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_uops.py::TestFastIdiv::test_division_power_of_two` | CDIV of int32 and uint32 by 2 becomes SHR | `DO › late_patterns › late_cdiv_shr.golden` (uint by 8, int by 8 of a dividend of each sign); `DO › late_patterns › truncating divisions and remainders keep their values` |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_floormod_power_of_two` | FLOORMOD by 8 becomes AND, no CMOD | `DO › simplifying_patterns › simplifying_floormod_shr_and.golden` (`m % 4`, `k % 8`, `w % 2^63`) |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_max_keeps_bound_for_idiv` | MAX survives the simplifying patterns, so `(max(x,0)+1)//3` needs no sign correction | `DO › simplifying_patterns › simplifying_floordiv_*.golden` (the last division); MAX's late lowering is `DO › late_patterns › late_max_cmplt.golden` |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_floordiv_power_of_two` | FLOORDIV by 2 of each 32 and 64-bit type becomes SHR, no CDIV or CMOD | `DO › simplifying_patterns › simplifying_floordiv_shr_and.golden` (`m // 8`, `w // 2^63`); `DO › simplifying_patterns › floor divisions and remainders keep their values` |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_unsigned_floordiv_is_cdiv` | unsigned FLOORDIV and FLOORMOD have no sign correction | `DO › simplifying_patterns › simplifying_floordiv_*.golden`, `simplifying_floormod_*.golden` (`w // 2^63`, `w % 2^63` by a typed constant) |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_and_mod` | uint32 CDIV and CMOD by 3 become SHR | `DO › late_patterns › late_cdiv_shr_fast_idiv.golden` (`x` in [0, 2^32 - 1] by 7); `DO › fast_idiv › fast_idiv_grid.golden` |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_nonpositive_divisor` | no SHR for a divisor of -3 or 0 | `DO › fast_idiv › fast_idiv declines a divisor that is not positive`; `late_cdiv_shr_fast_idiv.golden` (`p` by -3, `p` mod 0) |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_cmod_kept_when_idiv_declines` | CMOD by a variable, and uint64 CMOD by 3, stay CMOD | `DO › late_patterns › late_cdiv_shr_fast_idiv.golden` (`p` mod `d`, `w` mod 3) |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_bounded_numerator_zero` | `x` in [0, 1] divided by 3 is 0 | `DO › fast_idiv › fast_idiv_folds_a_small_dividend.golden`; `DO › fast_idiv › fast_idiv of a dividend below the divisor is zero of its type` |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_remove_powers_of_two` | `r // 448` for `r` below 2^20 shifts out 64 first and stays in int32 | `DO › fast_idiv › fast_idiv_shifts_out_powers_of_two.golden` |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_overflow` | (expected failure) uint32 by 7 over the whole range | `DO › fast_idiv › fast_idiv_grid.golden` (uint 4294967295 by 7 is `None` without a wider type, and widens to ulong with one) |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_disable_fast_idiv` | with `DISABLE_FAST_IDIV`, CDIV by 3 stays | `DO › late_patterns › late_cdiv_shr.golden` (`~disable_fast_idiv:true`) |
| tinygrad: `null/test_uops.py::TestUOpGraph::test_mulacc_shl` | `a*4096 + b` becomes MULACC after SHL | `DO › late_patterns › late_mulacc_mulacc_shl.golden` |
| tinygrad: `null/test_uops.py::TestUOpGraph::test_use_cmpeq` | `(x != 7) != True` becomes CMPEQ | `DO › late_patterns › late_comparison_cmpeq.golden` |
| tinygrad: `external/fuzz_fast_idiv.py` | z3 proves the rewrite equals truncating division for random types, bounds and divisors | `DO › fast_idiv › fast_idiv x d is x / d wherever it applies` (500 cases); `DO › fast_idiv › fast_idiv is division on the grid's dividends` |
| tinygrad: `null/test_const_folding.py::TestThreefryConstFolding::test_threefry` | THREEFRY of constants, decomposed, folds to a constant | `S › tinygrad › tests.golden › TestThreefryConstFolding.test_threefry` (the fold of the decomposed hash to the hash's value, D13); `DO › threefry2x32 › the hash of constants simplifies to a constant`; `DO › threefry2x32 › the hash of constants folds to its value` |
| tinygrad: `runtime/test_randomness.py::TestRandomness::test_threefry_against_reference` | JAX's `threefry_2x32` under the key (0, 1337) | `DO › threefry2x32 › JAX's values under the key (0, 1337)` (ten counter pairs) |
| tinygrad: `runtime/test_randomness.py::TestRandomness::test_threefry_against_reference_full`, `test_threefry_tensors_cnt`, `test_threefry_same_kernels` and the other `Tensor.rand` tests | seeds, counters and floats of `Tensor.rand` | dropped: the frontend's random numbers are nx's `Rng`; the hash itself is `DO › threefry2x32` |
| tinygrad: `null/test_randomness.py::TestRandomness::test_threefry_doesnt_use_long` | a program with THREEFRY has no 64-bit values on a target without them | dropped here: 64-bit emulation is `Decomp_dtype`'s, and the whole program is `Codegen`'s (L4) |
| tinygrad: `null/test_tensor_uop_mixin.py::test_threefry`, `test_threefry_random_bits` | `Tensor.threefry` is `UOp.threefry` | dropped: the `Tensor` surface (README exclusions) |
| tinygrad: `codegen/decomp/op.py` `threefry2x32` (no test) | the graph of the hash | `DO › threefry2x32 › threefry.golden`; `DO › threefry2x32 › Random123's known answers` (Random123's three `threefry2x32_20` vectors) |
| tinygrad: `codegen/decomp/op.py` `get_simplifying_rewrite_patterns` (no test) | each rule with and without SHR, AND and THREEFRY | `DO › simplifying_patterns › simplifying_{floordiv,floormod,threefry}_{none,shr_and,shr_and_threefry}.golden`; `DO › simplifying_patterns › a target with Threefry keeps it` |
| tinygrad: `codegen/decomp/op.py` `get_late_rewrite_patterns` (no test) | each rule with and without its operations | `DO › late_patterns › late_<family>_<ops>.golden` (max, logical, mul, cdiv, negation, comparison, extremes, mulacc, division); `DO › late_patterns › the late rules keep the values of integer expressions` |
| tinygrad: `codegen/decomp/op.py` power-of-two rules, `fast_idiv` and the one-integer-between rule on `True`, a float or `Invalid` (no test) | CPython's `TypeError`s and `True == 1` (README, CPython rows) | `DO › constants other than integers are declined` (four tests) |
| old: `unit/codegen/test_decompositions.ml` "magicgu is correct", "magicgu supports wide bounds" | `x // d = (x * m) >> s` up to `max_int` | `DO › fast_idiv › fast_idiv_grid.golden`, `fast_idiv is division on the grid's dividends` (bounds to 2^64 - 1); `magicgu` is private |
| old: `unit/codegen/test_decompositions.ml` "threefry2x32 is uint64", "Threefry derives uint64" | the hash is a Uint64, and THREEFRY is rewritten when the target lacks it | `DO › threefry2x32 › the hash of constants is a Uint64`; `DO › simplifying_patterns › simplifying_threefry_*.golden` |
| old: `unit/codegen/test_decompositions.ml` "Floordiv same sign lowers to Cdiv", "same-sign divisor bounds may include zero" | no sign correction when the signs agree, bounds touching 0 included | `DO › simplifying_patterns › simplifying_floordiv_*.golden`, `simplifying_floormod_*.golden` (`p // 3`, `n // q`, `a // b` in [0, 3], `c // e` in [-3, 0]) |
| old: `unit/codegen/test_decompositions.ml` "Floordiv mixed sign lowers with correction" | a corrected CDIV | `DO › simplifying_patterns › simplifying_floordiv_none.golden` (`m // 3`, `m // d`); `DO › simplifying_patterns › floor divisions and remainders keep their values` |
| old: `unit/codegen/test_decompositions.ml` "Floormod power of two lowers to And for negative input", "floor powers of two lower before truncation" | AND and SHR for any sign; unsigned 2^63 keeps 64 bits | `DO › simplifying_patterns › simplifying_floor{div,mod}_shr_and.golden` (`m // 8`, `m % 4`, `w // 2^63`, `w % 2^63`, and 2^64, which is no shift) |
| old: `unit/codegen/test_decompositions.ml` "late Cmod power of two rejects negative signed input", "late Cmod power of two uses generic rule without And" | CMOD by 4 of a signed dividend stays CMOD | `DO › late_patterns › late_cdiv_shr_fast_idiv.golden` (`m` mod 4); `DO › late_patterns › truncating divisions and remainders keep their values` |
| old: `unit/codegen/test_decompositions.ml` "fast idiv small range folds to zero", "fast idiv accepts wide divisor" | 0 for a small dividend; a divisor of `Int64.max_int` | `DO › fast_idiv › fast_idiv_folds_a_small_dividend.golden`; `DO › fast_idiv › fast_idiv divides by a divisor beyond every integer type`; `late_cdiv_shr_fast_idiv.golden` (`l` by 2^63 - 1) |
| old: `unit/codegen/test_decompositions.ml` "fast idiv recursion uses shifts" | powers of two shifted out before widening | `DO › fast_idiv › fast_idiv_shifts_out_powers_of_two.golden` |
| old: `unit/codegen/test_decompositions.ml` "fast idiv is enabled for Metal" | Metal's renderer lowers by fast_idiv | dropped here: which renderer disables it is `Renderer.Cstyle`'s (L5); the lowering is `late_cdiv_shr_fast_idiv.golden` |
| old: `unit/codegen/test_decompositions.ml` "fast idiv promotion checks dtype support", "late Cmod preserves unoptimizable division" | no widening to an unsupported type; CMOD by a variable stays | `DO › late_patterns › late_cdiv_shr_fast_idiv_narrow.golden`; `DO › fast_idiv › fast_idiv_grid.golden` (`renderer=none` rows) |
| old: `unit/codegen/test_decompositions.ml` "signed Cdiv pow2 uses constant condition" | a non-negative dividend needs no `x < 0` | `DO › late_patterns › late_cdiv_shr.golden` (`p` by 8) |
| old: `unit/codegen/test_decompositions.ml` "late Mul by one is not shifted", "late Cdiv by one is not shifted" | no shift by 0 | `DO › late_patterns › late_mul_shl.golden` (`x * 1`); `late_cdiv_shr.golden` (`u` and `m` by 1) |
| old: `unit/codegen/test_decompositions.ml` "Max bounds survive early lowering" | MAX is lowered only late | `DO › simplifying_patterns › simplifying_floordiv_*.golden` (the MAX is kept); `DO › late_patterns › late_max_cmplt.golden` |
| old: `unit/codegen/test_decompositions.ml` "early Floordiv by zero raises before trunc lowering", "early Floormod by zero raises before trunc lowering" | a division by the constant 0 raises in the symbolic rules | dropped here: constant folding is Symbolic's, its suite |
| old: `unit/codegen/test_decompositions.ml` "not (x < c) canonicalizes", "not (c < x) canonicalizes", "not-ne uses CMPNE true shape", "add-neg rewrite is commutative", "mul-recip rewrite is commutative", "-x < y*c canonicalizes", "-x < c canonicalizes", "bounded CMPLT collapses to equality" | the late comparison, negation and division rules | `DO › late_patterns › late_comparison_cmplt.golden`, `late_comparison_cmpeq.golden`, `late_negation_neg_sub.golden` (`x + -y`, `-y + x`), `late_division_fdiv.golden` (both orders) |
| old: `unit/codegen/test_decompositions.ml` "bounded CMPLT preserves a midpoint above the lower bound's width", "... below the upper bound's width" | the midpoint of mixed-width bounds | `DO › late_patterns › late_comparison_cmplt.golden` (weak and int32 bounds); `the late rules keep the values of integer expressions` |
| old: `unit/codegen/test_decompositions.ml` "late comparisons use exact integer proofs" | Int64's extremes, an empty interval and 2^80 | `DO › late_patterns › late_extremes_cmplt.golden` |
| old: `unit/codegen/test_decompositions.ml` "comparison extrema simplify before late codegen" | the whole pipeline folds those comparisons | dropped here: the pipeline is `Codegen`'s (L4) |
| old: `unit/codegen/test_decompositions.ml` "MAX promotes comparison and selection operands" | MAX of int16 and int32 compares and selects in int32 | `DO › late_patterns › late_max_cmplt.golden` (`max(s, y)`) |
| old: `unit/codegen/test_decompositions.ml` "floor correction promotes its boolean before negation", "floor mask retains the generated weak integer", "floor shift retains the generated weak count" | the correction's cast, the mask's and the shift's weak constants | `DO › simplifying_patterns › simplifying_floor{div,mod}_*.golden` (every node's type) |
| old: `unit/codegen/test_decompositions.ml` long, float and transcendental groups | 64-bit and narrow-float emulation, transcendental functions | dropped here: `Decomp_dtype`'s and `Transcendental`'s sections |
| old: `unit/frontend/test_run.ml` `constant_integer_division` | compiled CDIV, CMOD, FLOORDIV and FLOORMOD of int32 and uint32 by 2 to 2^31 - 1 | `DO › late_patterns › lowered divisions compute truncating and floor quotients and remainders` (the same values, lowered by both matchers and evaluated); the compiled run is the executor's (L7) |
| old: `unit/uop/test_symbolic.ml` "constant THREEFRY is not UOp-folded" (2 tests) | THREEFRY of constants is not folded | dropped here: Symbolic's section |
| old: `unit/test_cstyle.ml` "renderer op capabilities match cstyle render surface" | no renderer claims MAX, MULACC or THREEFRY | dropped here: `Renderer.Cstyle`'s (L5) |

## Symbolic

The suite is `Tolk_next.Symbolic` (`uop/symbolic/`), written `S` below. Its
group `S › tinygrad › tests.golden` replays tinygrad's tests:
`gen/uop/symbolic.py` runs each test and records every simplification it asks
for (a rewrite by one of `symbolic.py`'s matchers, `simplify`,
`simplify_valid`), with its result and the bounds tinygrad gives the result.
The replay of a test, `<Class>.<test>`, checks that Symbolic makes the same
rewrite, that the bounds are the same, and that an integer graph keeps its
value at bindings of its leaves wherever nothing wraps, and a graph of
constants the value the machine computes: the claim tinygrad proves with z3.
The rendered strings of `helper_test_variable` are compared as nodes: tinygrad
evaluates each string into the node it checks, and the golden holds that node.
`S › tinygrad › sym simplifies random integer expressions as tinygrad does`
replays 200 random expressions the same way, and `S › laws` state the laws on
expressions generated here, weak and at committed widths.

tinygrad folds committed constants without wrapping them to their width, which
its own `TestModularWraparound` marks as expected failures; tolk.next reads
them at their width (D13). The replays whose folds differ from tinygrad's for
it check that each result is the machine's value instead, and name D13.

### tinygrad: null/test_uop_symbolic.py, test_symbolic_failures.py, test_simplify_valid_idx.py

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_uop_symbolic.py::TestSymbolic (every test but the four below, 184 tests) | `sym` of each expression and its bounds; `commutative` of each expected expression; `simplify`, `ssimplify`, `gcd` and `divide_exact` results | `S › tinygrad › tests.golden › TestSymbolic.<test>` |
| tinygrad: null/test_uop_symbolic.py::TestSymbolic::test_equality | equal expressions are one node, and operand order counts | `S › tinygrad's other tests › equal expressions are the same node, and operand order counts` |
| tinygrad: null/test_uop_symbolic.py::TestSymbolic::test_divide_exact_not | `divide_exact` gives up | `S › tinygrad's other tests › divide_exact gives up on what does not divide` |
| tinygrad: null/test_uop_symbolic.py::TestSymbolic::test_div_mod_zero | a division or remainder by 0 raises | `DM › zero divisors` (Divandmod's section) |
| tinygrad: null/test_uop_symbolic.py::TestSymbolic::test_variable_divmod | a variable bounded by another | dropped: a variable's bounds are numbers (plan, L3 ruling) |
| tinygrad: null/test_uop_symbolic.py::TestSymbolicPickle (2 tests) | a variable survives pickling | dropped: pickling is Python's; a graph's text form round-trips in the graph format (test support's `Graph` suite) |
| tinygrad: null/test_uop_symbolic.py::TestSymbolicNumeric (9 tests) | a rewrite of a constant is its value, and bounds hold every value | `S › tinygrad › tests.golden › TestSymbolicNumeric.<test>`; `S › laws › sym keeps the value of an integer expression where nothing wraps` |
| tinygrad: null/test_uop_symbolic.py::TestSymbolicVariables::test_simple, test_compound, test_dedup | `variables` lists each variable once, sorted | `S › tinygrad's other tests › variables lists each variable once, sorted by name` |
| tinygrad: null/test_uop_symbolic.py::TestSymbolicVariables::test_variable_min_eq_max_bind_folds | a bound variable of one value folds | `S › tinygrad › tests.golden › TestSymbolicVariables.test_variable_min_eq_max_bind_folds` |
| tinygrad: null/test_uop_symbolic.py::TestSymInfer::test_sym_infer, test_sym_infer_floordiv_floormod, test_sym_infer_with_cast | `sym_infer` of sums, products, floor division and casts | `O › sym_infer › an integer is itself`; `O › sym_infer › a node takes its variables' values`; `O › sym_infer › divisions round as their operations say`; `O › sym_infer › a cast converts without truncating to a width` (Ops' section) |
| tinygrad: null/test_uop_symbolic.py::TestSymInfer::test_sym_infer_with_bitcast | `sym_infer` through bit reinterpretations | `S › tinygrad's other tests › sym_infer reads bits through bit reinterpretations` |
| tinygrad: null/test_uop_symbolic.py::TestSymInfer::test_sym_infer_deeply_nested | `sym_infer` of an expression 200 deep | `S › tinygrad's other tests › sym_infer evaluates an expression nested 200 deep` |
| tinygrad: null/test_uop_symbolic.py::TestSymbolicSymbolicOps | none | dropped: a string literal in tinygrad, not a test |
| tinygrad: null/test_uop_symbolic.py::TestInvalidIndex (7 tests), TestStoreLoadFolding, TestGatedUopGivenValid (2 tests), TestSymbolicRealWorld | invalid gates, store and load folding, gated index simplification, a real index | `S › tinygrad › tests.golden › <Class>.<test>`; the rendered text of test_resnet_half is Render's |
| tinygrad: null/test_uop_symbolic.py::TestMoveWhereOnLoad::test_bool_index_preserves_dtype | the rewrite keeps a boolean index's type, which `type_verify` checks | `S › tinygrad › tests.golden › TestMoveWhereOnLoad.test_bool_index_preserves_dtype` (reading the golden checks every node's derived type) |
| tinygrad: null/test_uop_symbolic.py::TestRangeSplitting::test_backedge_preserves_constant_condition | a constant backedge condition stays | `S › tinygrad › tests.golden › TestRangeSplitting.test_backedge_preserves_constant_condition`; `S › symbolic › ordering › a backedge keeps a constant condition` |
| tinygrad: null/test_uop_symbolic.py::TestRangeSplitting::test_range_split_on_mod | `pm_split_ranges` | dropped here: `codegen/simplify.py`'s matchers, Simplify's section (L3) |
| tinygrad: null/test_uop_symbolic.py::TestBounds::test_unrolled_arange | bounds of an arange index | `S › tinygrad's other tests › the bounds of an unrolled arange's index` |
| tinygrad: null/test_uop_symbolic.py::TestBounds::test_where_float_consts | bounds of float selections and their casts | `S › tinygrad's other tests › the bounds of selections between float constants` |
| tinygrad: null/test_uop_symbolic.py::TestFuzzFailure::test_fuzz_failure1, null/test_symbolic_failures.py::TestFuzzFailure (11 tests) | simplifying keeps the value at a binding | `S › tinygrad › tests.golden › TestFuzzFailure.test_fuzz_failure<n>` (the value law at the extremes and random bindings of every record) |
| tinygrad: null/test_simplify_valid_idx.py::TestValidIdxSimplification (every test but the one below, 13 tests) | `sym` with `pm_move_where_on_load` on gated loads; `simplify_valid` | `S › tinygrad › tests.golden › TestValidIdxSimplification.<test>` |
| tinygrad: null/test_simplify_valid_idx.py::TestValidIdxSimplification::test_valid_becomes_const1_z3 | z3 proves the gated index of test_valid_becomes_const1 is `r0*1568`, and not a wrong one | `S › tinygrad's other tests › an index simplified under its gate keeps its value where it holds`; that z3 refutes a wrong index tests the prover, dropped |
| tinygrad: null/test_simplify_valid_idx.py::TestDropTrueGate::test_const_gate_clause_is_not_moved_to_load | a constant clause stays in the selection | `S › tinygrad › tests.golden › TestDropTrueGate.test_const_gate_clause_is_not_moved_to_load`; `S › conditions › pm_move_where_on_load › a constant clause stays` |
| tinygrad: null/test_simplify_valid_idx.py::TestDropTrueGate::test_drop_true_gate_on_index | `indexing_simplify` drops a true gate | dropped here: `codegen/late/coalesce.py`, Codegen's section (L4) |
| tinygrad: null/test_simplify_valid_idx.py::TestImageSimplification (18 tests), TestImageStore | image indexing | dropped: images are not ported (README) |
| tinygrad: null/test_simplify_valid_idx.py::TestRangeShrink (8 tests) | the code generation pipeline shrinks guarded ranges | dropped here: `full_rewrite`, Codegen's section (L4) |

### tinygrad: null/test_const_folding.py, runtime/test_const_folding.py

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_const_folding.py::TestWeakConstFolding (4 tests), TestBitcastConstFolding::test_out_of_range_source_value, test_scalar_bitcast | weak constants fold exactly; a bitcast of a constant folds | `S › tinygrad › tests.golden › <Class>.<test>` |
| tinygrad: null/test_const_folding.py::TestThreefryConstFolding::test_threefry | a threefry of constants folds once decomposed | `S › tinygrad › tests.golden › TestThreefryConstFolding.test_threefry` (the machine's value, D13) |
| tinygrad: null/test_const_folding.py::TestBitcastConstFolding::test_vec_bitcast | a bitcast of a stack of constants | dropped here: the lanes fold after devectorizing, `full_rewrite`, Codegen's section (L4) |
| tinygrad: null/test_const_folding.py::TestMovedConstFolding (5 tests), TestReduceOpsConstFolding::test_sum_output_dtype | constant folding in `Tensor` programs | dropped: `Tensor` surface, the frontend is nx |
| tinygrad: runtime/test_const_folding.py::TestMovedConstFolding (2 tests), TestReduceOpsConstFolding (9 tests), TestMultiConstFolding (2 tests) | constant folding in `Tensor` programs run on a device | dropped: `Tensor` surface and execution |
| tinygrad: runtime/test_const_folding.py::TestTautologicalCompare (5 tests) | `x < x`, `x == x` and `x != x` on tensors | dropped: `Tensor` surface; the folds are `S › symbolic_simple › zeros › x < x is false`, `S › symbolic_simple › zeros › x <> x is false for integers and booleans`, `S › symbolic_simple › zeros › x <> x stays for floats, which may be NaN` |

### tinygrad: the tests of other files that other sections leave to Symbolic's

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_uop_graph.py::TestGraphRewriteConst (2 tests), TestGraphRewrite::test_commutative_work, test_consts_go_last_right_away, test_consts_go_last, TestUOpGraph::test_where_same_fold, test_where_const_fold, test_depth_2_const_fold | `sym` and `simplify` fold stacks, order operands, move constants last, fold selections | `S › tinygrad › tests.golden › <Class>.<test>` |
| tinygrad: null/test_uop_graph.py::TestModularWraparound (6 tests, xfail in tinygrad) | `simplify` folds constants modulo their width | `S › tinygrad › tests.golden › TestModularWraparound.<test>` (test_div, test_neg and test_payne_hanek_reduction_bug: the machine's value, D13, which tinygrad's own tests expect) |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_rtruediv, test_float_direct, test_ssimplify, test_x_lt_x, test_plus_ordering_lt | `simplify` under `float`, `bool` and `ssimplify` | `S › tinygrad › tests.golden › TestUOpResolve.<test>` |
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_remove_invalid_stack_lanes, and the `pm_remove_invalid` half of test_invalid_dtype_and_consumers | `pm_remove_invalid` zeroes invalid lanes and gates | `S › tinygrad › tests.golden › TestDTypeFromUOp.test_remove_invalid_stack_lanes`; `S › pm_remove_invalid › a gate's invalid is 0 of the gate's type`; `S › pm_remove_invalid › a float gate's invalid is 0.0 of its type` |
| tinygrad: null/test_uops.py::TestSafeCast (3 tests), TestUOpMethod::test_cmp_self_folding_multidim | `simplify` removes casts, folds `x < x` on shaped nodes | `S › tinygrad › tests.golden › <Class>.<test>` |
| tinygrad: null/test_graph_rewrite.py::TestBottomUpRewrite::test_const_folding | bottom-up and top-down `symbolic_simple` reach one fold | `S › tinygrad › tests.golden › TestBottomUpRewrite.test_const_folding` (both directions are recorded) |
| tinygrad: null/test_graph_rewrite.py::TestEdgeCasesAndSpecialOperations::test_full_graph_rewrite_transcendental_edge_cases | `log2(-1)` folds to NaN and `1/0` to infinity | `S › tinygrad's other tests › log2 of -1 folds to NaN and the reciprocal of 0 to infinity` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_committed_const_conversion_folds, test_derivable_const_rounds_at_the_derived_width | a cast of a committed constant folds; `x * 1` and `x * -1` fold after a literal is rounded | `S › tinygrad › tests.golden › TestWeakPromotion.<test>` (`symbolic_simple` alone and with `pm_commit_weak`) |
| tinygrad: `null/test_uop_symbolic.py` (the eight `.render()` asserts) | the rewritten expression renders as written | `S › tinygrad › tests.golden › <Class>.<test>` for the rewrite; the text is Render's |

### old tolk: unit/uop/test_symbolic.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/uop/test_symbolic.ml "x + 0 -> x", "x * 1 -> x", identity_fold (9 tests) | identities | `S › symbolic_simple › identities › x + 0, x lxor 0 and x lor 0 are x`; `S › symbolic_simple › identities › x * 1 and x // 1 are x`; `S › symbolic_simple › zeros › x % x, x lxor x and x land 0 are 0`; `cdiv(x, 1)` is dropped, below |
| old: unit/uop/test_symbolic.ml "int neutral chain -> x", "associative combine", associative_fold, combine_terms | `((x + 1) * 1) + -1`, `(x + 3) + 5`, `x + x` | `S › symbolic › constants › two associative operations on constants fold them`; `S › symbolic › terms › like terms combine` |
| old: unit/uop/test_symbolic.ml "x // x -> 1", "x % x -> 0", "x < x -> false", self_fold "x floordiv x", "x ^ x", "x < x" | self folding | `S › symbolic_simple › identities › x // x is 1`; `S › symbolic_simple › zeros › x < x is false`; `S › symbolic_simple › zeros › x % x, x lxor x and x land 0 are 0` |
| old: unit/uop/test_symbolic.ml self_fold "cdiv(x, x)", "cdiv(x, -1) -> -x", identity_fold "cdiv(x, 1)", bool_cast_fold "cmod(x, x)", divandmod_tests (2 tests), lt_fold "lt cdiv fold", divmod_reconstitute "cdiv/cmod recombine" (2 tests) | truncating division folds like floor division | dropped: tinygrad HEAD's rules fold floor division only; truncating division appears when floor division is decomposed (Codegen, L4) |
| old: unit/uop/test_symbolic.ml "cast const -> const", "cast(const(3), float32)", "cast to same dtype" | casts of constants and to their own type | `S › symbolic_simple › constants › a cast of a constant is the constant of the cast's type`; `S › symbolic_simple › casts › a cast or bitcast to its operand's type is its operand` |
| old: unit/uop/test_symbolic.ml "x \| !x -> true", "!!x -> x", "!c.where(t, f) -> c.where(f, t)" | boolean identities, either side of the negation | `S › symbolic › terms › x lor not x is true`; `S › symbolic_simple › identities › a double negation is x`; `S › symbolic › selections › a selection by a negation swaps its branches` |
| old: unit/uop/test_symbolic.ml "an offset crosses a comparison only without wrapping" | `c0 + x < c1` over uint8 keeps a wrapping offset | dropped: tinygrad HEAD moves the offset under the rules' contract that nothing wraps, which `S › laws › sym keeps the value of an integer expression where nothing wraps` states |
| old: unit/uop/test_symbolic.ml "a cast constant keeps its wrapped value" | `x < uint8 300` is `x < 44` | `S › symbolic_simple › constants › a comparison reads a committed constant at its width` (D13) |
| old: unit/uop/test_symbolic.ml "a non-finite cast to an integer stays a cast" | a cast of an infinity or NaN to an integer stays | `S › symbolic_simple › constants › a cast of an infinity or a NaN to an integer stays a cast` |
| old: unit/uop/test_symbolic.ml "constant cdiv/cmod use truncating semantics", const_fold (every test) | constants fold, truncating and floor division round as they say, a zero divisor gives 0 or the dividend, stacks fold lane by lane | `S › symbolic_simple › constants › a truncating division of constants rounds toward zero`; `S › symbolic_simple › constants › a division of constants by zero is 0, a remainder the dividend`; `S › symbolic_simple › constants › an operation on committed constants computes at their width`; `S › symbolic_simple › constants › an operation on stacks of constants folds lane by lane`; `S › symbolic_simple › constants › a where, a comparison and a negation of stacks fold lane by lane`; the values of floor division are `O › exec_alu` |
| old: unit/uop/test_symbolic.ml "invalid gate survives zero multiply", "non-weak invalid comparison gates bool result", "direct invalid comparison keeps bool dtype", "invalid gate cast stays gated", invalid_where (2 tests) | operations move inside an invalid gate | `S › invalid values › a binary operation moves inside the gate of its first operand` (a product and a comparison); `S › invalid values › a cast moves inside the gate`; `S › invalid values › a comparison of invalid is left`; `S › invalid values › a selection by invalid is invalid`; `S › invalid values › a gate on a condition moves out of the selection`; `S › tinygrad › tests.golden › TestInvalidIndex.test_invalid_times_0` |
| old: unit/uop/test_symbolic.ml "where closure folds a nested condition", "where closure keeps unrelated conditions", "where closure precedes gate merging", "a nonzero where becomes a guard", where_fold (every test but the last) | selection folding | `S › symbolic › selections` (every test); `S › tinygrad › tests.golden › TestSymbolic.test_where_closure_folding_before_gate_merge` |
| old: unit/uop/test_symbolic.ml where_fold "where eq one zero flips to ne zero one" | `CMPEQ` | dropped: tinygrad HEAD has no `CMPEQ`; equality is `logical_not` of `<>` |
| old: unit/uop/test_symbolic.ml "constant guards stay out of index validity" | a constant clause stays in the selection | `S › conditions › pm_move_where_on_load › a constant clause stays` |
| old: unit/uop/test_symbolic.ml "cast(bool) != const folds", bool_cast_fold's three `cast(bool -> int)` tests | a boolean cast to an integer compared to a constant | `S › symbolic_simple › identities › a boolean cast to an integer and compared to 0 is the boolean`; `... compared to 1 is its negation`; `... differs from any other integer` |
| old: unit/uop/test_symbolic.ml "constant BITCAST folds", "double BITCAST collapses", "bitcast const float32 to int32 folds" | bitcasts of constants and chains | `S › symbolic_simple › casts › a bitcast of a constant has the same bits`; `S › symbolic_simple › casts › two bitcasts are one` |
| old: unit/uop/test_symbolic.ml "STACK const bitcast folds", bool_cast_fold "cast STACK const folds lane-wise", "bitcast STACK const folds lane-wise" | a cast or bitcast of a stack of constants folds lane by lane | dropped here: tinygrad HEAD folds those lanes after devectorizing (Codegen, L4) |
| old: unit/uop/test_symbolic.ml "constant THREEFRY is not UOp-folded", bool_cast_fold "constant Threefry is not folded" | threefry of constants stays | `S › symbolic_simple › constants › a threefry of constants stays` |
| old: unit/uop/test_symbolic.ml "INDEX(STACK const) folds", index lane pushing (3 tests) | an index of a stack by a constant is its element | `M › indexing › index_a_stack_by_a_constant.golden` (Movement's section); the fold at construction is Ops' |
| old: unit/uop/test_symbolic.ml spec "full_spec accepts value INDEX lane selection" | the specification | dropped here: Spec's section |
| old: unit/uop/test_symbolic.ml "NaN cmpeq folds to false" | NaN is not equal to itself | `S › symbolic_simple › constants › NaN is unequal to itself when constants fold, as IEEE says` |
| old: unit/uop/test_symbolic.ml divmod_reconstitute "floor div/mod recombine", "scaled nested floor div/mod recombine" | a remainder and a quotient recombine | `S › symbolic_simple › recombination` (every test); `S › tinygrad › tests.golden › TestSymbolic.test_mod_recombine_with_outer_mul` |
| old: unit/uop/test_symbolic.ml divmod_reconstitute "nested floor div/mod recombine with a positive symbolic radix", "symbolic range coordinates recover the flattened index" | recombination by a symbolic radix | dropped: tinygrad HEAD recombines constant radices only (`_quotient_base` needs a constant divisor) |
| old: unit/uop/test_symbolic.ml range_fold (2 tests) | a range of one value is 0; a range of a symbolic end is not | `S › symbolic › bounds › a range of a constant end with one value is 0`; `S › symbolic › bounds › a range of a symbolic end is not folded` |
| old: unit/uop/test_symbolic.ml bool_cast_fold "bool MUL → AND", "pow constant exponent rewrites by squaring", "nested where" | boolean products; powers of constants; nested selections | `S › symbolic_simple › booleans › a boolean product is a conjunction`; `S › symbolic_simple › powers › a power of constants is its value`; `S › symbolic › selections › nested selections sharing a false branch merge by conjunction` |
| old: unit/uop/test_symbolic.ml lt_fold (every test but "lt cdiv fold") | comparisons fold; a float comparison keeps its rounding; `(x / y) / z` | `S › symbolic › comparisons` (every test, `c0 + x < c1 stays for floats, where moving c0 rounds` among them); `S › symbolic › terms › (x / y) / z is x / (y * z)` |
| old: unit/uop/test_symbolic.ml where_fold "cast stays outside a conditional", "Boolean selection stays inside its integer cast" | a cast of a selection stays | `S › symbolic › selections › a cast of a selection stays outside it`; `S › tinygrad › tests.golden › TestSymbolic.test_where_cast` |
| old: unit/uop/test_symbolic.ml reduce "mul-term hoist floats non-range factors out of a lowered reduce" | factors move out of a kernel reduction | `S › sym › factors independent of a sum's ranges move out of it`; `S › sym › only non-negative factors move out of a maximum` |
| old: unit/uop/test_symbolic.ml reduce "add tensor reduce floats const and preserves axes" | a tensor-level reduction keeps the factors that vary along its axes | dropped: `sym` runs on kernels, whose reductions name their ranges; tinygrad HEAD's `reduce_mul_chain` reads a reduction's ranges and does not see a tensor-level one's axes |
| old: unit/uop/test_symbolic.ml load_store (3 tests) | loads and stores of invalid and gated indices | `S › invalid values › a load from an invalid index is its alternative, or 0`; `S › sym › storing a selection of the loaded value stores where it differs` |
| old: unit/uop/test_symbolic.ml sigmoid (3 tests) | `x * (1 / (1 + x))` stays at float | dropped: tinygrad HEAD rewrites it, `S › sym › x * (1 / (1 + x)) is 1 - 1 / (1 + x)`; the old precision divergence had no admitted reason |
| old: unit/uop/test_symbolic.ml simplify_valid (3 tests) | a bitwise and is no valid; a clause others read comes first; the rewrite fires on the raw predicate | `S › tinygrad › tests.golden › TestValidIdxSimplification.test_bitwise_and_is_not_a_valid`; `S › conditions › simplify_valid › a clause on an expression others read is applied first`; `S › conditions › pm_simplify_valid › a conjunction is simplified` |
| old: unit/uop/test_symbolic.ml uop_given_valid "a load keeps its own gate" | `uop_given_valid` leaves a load's gate | dropped: tinygrad HEAD substitutes into the load's gate; `pm_simplify_valid` leaves a gated value that reads an index alone (`S › tinygrad › tests.golden › TestInvalidIndex.test_gated_load_keeps_index_valid`) |
| old: unit/uop/test_symbolic.ml uop_given_valid "a clause on a loaded value still applies" | a clause bounds a loaded value | `S › conditions › uop_given_valid › a clause on a loaded value bounds it` |
| old: unit/uop/test_symbolic.ml masked_div | `(x & -4) // 4` | `S › symbolic_simple › zeros › a mask of the bits a division by a power of two drops is removed` |
| old: unit/uop/test_symbolic.ml unpack_u64 (3 tests) | a packed 64-bit integer unpacks, a wide high half does not | `S › symbolic_simple › powers › a 64-bit integer packed from two halves unpacks to the half read` |
| old: unit/uop/test_symbolic.ml mop_cleanup (5 tests) | movement cleanups | Movement's section |
| old: unit/uop/test_symbolic.ml remove_invalid (3 tests) | invalid gates and lanes become zeros of their type | `S › pm_remove_invalid` (every test) |
| old: unit/uop/test_symbolic.ml "END preserves effects" | an end of folded ranges is its store; a live range stays; a constant backedge condition stays | `S › symbolic › ordering › an end of constant ranges only is its store`; `S › symbolic › ordering › an end drops the ranges that became constants`; `S › symbolic › ordering › a backedge keeps a constant condition` |
| old: unit/uop/test_symbolic.ml "distributed negation keeps scaled terms shared" | a negated sum distributes and folds its coefficients; a float one stays | `S › sym › a negated sum with a scaled term folds each term's coefficient`; the float half is dropped: tinygrad HEAD distributes a negation over a float sum too, `S › sym › -(x + y) is -x + -y` |
| old: unit/uop/test_symbolic.ml rule body promotion (4 tests) | a rule builds weak constants and promotes an integer exponent | `S › symbolic_simple › powers › c ** x computes in float for an integer exponent`; `S › symbolic › terms › a term's new coefficient is a weak constant`; the nested division is Divandmod's section |
| old: unit/uop/test_symbolic.ml integer width folding (4 tests) | 64-bit arithmetic that fits computes in 32 bits; bounded cast chains collapse | `S › symbolic › casts › 64-bit arithmetic that can overflow 32 bits stays`; `S › symbolic › casts › a cast chain of a bounded integer is one cast, of an unbounded one two` |
| old: unit/engine/test_symbolic.ml (every test) | symbolic sizes run on a device | dropped here: execution, Engine's section (L7) |
| old: unit/codegen/test_simplify.ml (every group but node_vmin / node_vmax) | `codegen/simplify.py`'s matchers | dropped here: Simplify's section (L3) |
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax (17 tests) | bounds of nodes | Ops' section (`O › bounds`) |

### Other sections' rows left to Symbolic's

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/uop/test_uop.ml commutative_axes_use_lexical_argument_order | `simplify` orders commutative operands | `S › commutative › two sums of the same weak integer terms are the same node`; `S › laws › commutative orders the operands of a weak integer sum` |
| old: unit/uop/test_uop.ml division_promotes_integer_operands (its folding) | a float division of integers folds | `S › tinygrad › tests.golden › TestUOpResolve.test_rtruediv` |
| old: unit/uop/test_uop.ml smax_smin_fold_when_bounds_decide, sprod_simplifies (their folding) | maxima and products fold | `S › symbolic › bounds › a maximum of operands whose bounds do not overlap is the greater`; `S › symbolic › constants › two associative operations on constants fold them` |
| old: unit/uop/test_uop.ml exact_symbolic_bounds (`parse_valid`) | a clause reads as a bound | `S › conditions › uop_given_valid › a clause bounds an expression`; `S › conditions › uop_given_valid › a negated clause bounds an expression from below`; `S › conditions › uop_given_valid › a clause that is not a bound is ignored` |
| old: unit/uop/test_weak.ml late_simplification_preserves_committed_literals, consecutive_weak_casts_preserve_integer_conversion (the folded value) | casts of committed constants fold | `S › symbolic_simple › constants › a cast of a constant is the constant of the cast's type`; `S › tinygrad › tests.golden › TestWeakPromotion.test_committed_const_conversion_folds` |
| old: unit/codegen/test_decompositions.ml "POW promotes weak exponents before parity arithmetic" | a power is computed by `xpow` | `S › sym › a power is computed from exp2 and log2` |
| old: unit/codegen/test_divandmod.ml simplify_preserves_index_values, adjacent_bit_extracts_recombine, quotient_partner_recombines_through_a_merged_divisor, shifted_quotient_partner_recombines | the whole rule set keeps values and recombines | `S › laws › sym keeps the value of an integer expression where nothing wraps`; `S › symbolic_simple › recombination` (every test); `S › tinygrad › tests.golden › TestSymbolic.test_div_mod_recombine_merged_quotient`, `TestSymbolic.test_div_mod_recombine_shifted_quotient` |

## Gpudims

The suite is `Tolk_next.Gpudims` (`codegen/gpudims/`), written `GD` below.
`grouped_dims.golden` holds every case of tinygrad's test, the old tolk's
three, and 392 sizes drawn from a fixed seed against the bounds of real
targets: each row gives the hardware indices and their sizes, and each loop's
index as an expression, or the exception tinygrad raises. tinygrad proves each
case one-to-one with z3; here `GD › grouped_dims › grouped_dims numbers each
iteration of the golden's cases once` enumerates every launch of each case up to
2^12 iterations and checks that each iteration is numbered exactly once,
and that each hardware index is within its bound, the first axis excepted
when the last one's divisors move onto it.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_grouped_dims` | sizes of 20 cases, reversal, splits, merges, and three `RuntimeError`s | `GD › grouped_dims › grouped_dims.golden` (the first rows); the z3 proof is `GD › grouped_dims › grouped_dims numbers each iteration of the golden's cases once` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_grouped_direct_dims_are_special` | loops 2 and 3 of (2,3,4,5) are hardware indices | `GD › grouped_dims › a loop that keeps its own axis is that hardware index` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_grouped_dims_high_rank` | 4 to 6 loops onto 2 or 3 axes; no bounds leaves every loop its own index | `GD › grouped_dims › grouped_dims.golden` (the four rows); `GD › grouped_dims › without bounds, every loop is its own hardware index` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_symbolic_dims_cross_launch_limit` | `(1, n)` with `n` in [1, 4] under (4, 3) | `GD › grouped_dims › symbolic sizes › grouped_symbolic_crosses_a_limit_by_merging.golden` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_global_prod_max` | `global_prod_max` bounds workgroups by threads | `GD › add_gpudims › add_gpudims_bounds_globals_by_threads.golden`, `add_gpudims_bounds_globals_by_merged_threads.golden`, `add_gpudims_bounds_globals_by_threads_alone.golden` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_max_sizes_none` | no bounds | `GD › grouped_dims › grouped_dims.golden` (`None` rows) |
| tinygrad: `codegen/gpudims.py` `add_gpudims`, `pm_device_to_var` (no test) | globals, locals, warps, masks, device ranges | `GD › add_gpudims › add_gpudims_*.golden` (17 kernels); `GD › add_gpudims › add_gpudims_declines.golden`, `add_gpudims_failures.golden` |
| old: `unit/codegen/test_gpudims.ml` "single dim fits", "two dims fit", "reverse two dims", "three dims not reversed", the six "splitting same-length" cases, "(512,4,2) / (8192,2,2)", the five "expansion" cases, the four "contraction" cases | sizes and one-to-one numbering | `GD › grouped_dims › grouped_dims.golden` (the same rows as tinygrad's); `GD › grouped_dims › grouped_dims numbers each iteration of the golden's cases once` |
| old: `unit/codegen/test_gpudims.ml` "reverse maps returned expressions to original axes" | reversed indices map back to their loops | `GD › grouped_dims › grouped_dims.golden` (`(2, 3)` reversed: `[gidx1, gidx0]`) |
| old: `unit/codegen/test_gpudims.ml` "split redistribution retains exact intermediate products" | sizes near 2^61 do not overflow | `GD › grouped_dims › grouped_dims.golden` (the `2305843009213693952` row) |
| old: `unit/codegen/test_gpudims.ml` "split decomposition uses integer floor division", "unmerged dims decompose through integer arithmetic" | `//` and `%`, never FDIV | `GD › grouped_dims › grouped_dims.golden` (`(7, 7)` under `(49, 1, 1)`, and every expression) |
| old: `unit/codegen/test_gpudims.ml` "symbolic dimensions can cross a limit by grouping", "symbolic contraction keeps grouped SPECIAL size symbolic", "symbolic fitting dimensions keep their physical extent", "symbolic passthrough keeps SPECIAL size symbolic" | symbolic sizes merge into symbolic hardware sizes | `GD › grouped_dims › symbolic sizes › grouped_symbolic_*.golden` |
| old: `unit/codegen/test_gpudims.ml` "symbolic dimensions cannot be split at their maximum" | a symbolic size cannot be split | `GD › grouped_dims › symbolic sizes › grouped_symbolic_failures.golden` |
| old: `unit/codegen/test_gpudims.ml` "grouping feasibility does not overflow host integers" | 2^32 × 2^32 under one bound fails cleanly | `GD › grouped_dims › grouped_dims.golden` (the `4294967296` row) |
| old: `unit/codegen/test_gpudims.ml` "prime dim 23 unfactorable", "unfactorable (128,3,4) / (16,2,2)", "too many dims (2,3,4,5,6)" | `cannot limit dim` | `GD › grouped_dims › grouped_dims.golden` (the `RuntimeError` rows) |
| old: `unit/codegen/test_gpudims.ml` "coordinate promotion: grouping widens dimensions before multiplying" | int16 by int32 sizes merge in int32 | `GD › grouped_dims › symbolic sizes › grouped_symbolic_merges_committed_sizes.golden` |
| old: `unit/codegen/test_gpudims.ml` "device axes become scalar parameters and leave END ranges" | DEVICE ranges become `_device_num` | `GD › add_gpudims › add_gpudims_device_range.golden`, `add_gpudims_device_range_without_kernel.golden`; `add_gpudims_keeps_an_end_of_a_variable.golden` |
| old: `unit/codegen/test_gpudims.ml` "keeps the warp dimension separate while folding local axes" | a warp keeps its own thread axis | `GD › add_gpudims › add_gpudims_keeps_the_warp_apart.golden` |
| old: `unit/codegen/test_gpudims.ml` "replaces global ranges with SPECIAL", "replaces global+local ranges" | ranges become hardware indices | `GD › add_gpudims › add_gpudims_globals.golden`, `add_gpudims_globals_by_axis_order.golden`, `add_gpudims_globals_and_locals.golden`, `add_gpudims_merges_four_globals.golden`, `add_gpudims_splits_a_global.golden`, `add_gpudims_keeps_a_reduce_range.golden` |
| old: `unit/codegen/test_gpudims.ml` "no-op when no GPU ranges", "no-op when SPECIAL already present" | `None` | `GD › add_gpudims › add_gpudims_declines.golden` (and without kernel information) |
| old: `unit/codegen/test_gpudims.ml` "global_prod_max caps global size by local hardware size" | workgroups bounded by threads | `GD › add_gpudims › add_gpudims_bounds_globals_by_threads.golden` |
| old: `unit/codegen/test_gpudims.ml` "missing local range gets gated with Invalid", "two missing local ranges gate on a bool AND of equalities" | the store's index is valid only on thread 0 of each missing index | `GD › add_gpudims › add_gpudims_masks_a_store_by_its_missing_local.golden`, `..._missing_locals.golden`; a local store is not masked: `add_gpudims_leaves_a_local_store_unmasked.golden` |
| old: `unit/codegen/test_gpudims.ml` "missing local range rejects multi-index global store" | a two-index store fails | `GD › add_gpudims › add_gpudims_failures.golden` |
| tinygrad: `codegen/gpudims.py` `_split_dims` on a symbolic size or fewer than three bounds (`ValueError`, `AssertionError`, `IndexError`) | README, CPython rows: "cannot limit dim" | `GD › grouped_dims › grouped_dims.golden` (the `IndexError` rows); `GD › grouped_dims › symbolic sizes › grouped_symbolic_failures.golden` |
| tinygrad: `codegen/gpudims.py` `add_gpudims` with a symbolic warp (`ValueError`) | README, CPython rows: the warp's bound is its size's upper bound | `GD › add_gpudims › add_gpudims_symbolic_warp.golden` |

## Gater

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `codegen/late/gater.py` (no test file covers every rule) | the two-index rules, the first-index rules on `INDEX` and `SHRINK` for loads and stores, the alternative of a gated load, the four alternatives of the selection fold and its negated form, and the accesses no rule matches | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden` (32 cases, `*_kept` for the accesses left alone) |
| tinygrad: `codegen/__init__.py` `full_rewrite_to_sink` (the "move gates from index" pass) | the pass on compiled kernels: padded loads on the CPU and on CUDA, a padded convolution, a gated store into workgroup memory | `Tolk_next.Gater › pm_move_gates_from_index › pad_moved.golden`, `pad_value_moved.golden`, `conv_moved.golden`, `pad_cuda_moved.golden`, `sum_group_moved.golden` |
| tinygrad: `null/test_uops.py::TestGatedStoreRewrite::test_tiny_gate_store`, `test_gate_some_stores`, `test_merge_ifs_alt` | a store through a gated index is gated, then becomes `IF`/`STORE`/`ENDIF` | the gate: `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=index_store`; the `IF` and `ENDIF`: dropped here, `codegen/__init__.py`'s `pm_linearize_cleanups`, in `Codegen`'s suite |
| old: `test/unit/codegen/test_lower.ml` "gater leaves already-gated invalid-index load unchanged" | a load that already has a gate is left alone | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=gated_load_kept` |
| old: `test/unit/codegen/test_lower.ml` "gater strips both image indexes with same invalid gate", "gater strips image load coordinates with same invalid gate" | two indices under one gate are stripped, the gate on the load | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=two_indices_load` |
| old: `test/unit/codegen/test_lower.ml` "gater strips image store coordinates with same invalid gate" | the same for a store, its value kept | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=two_indices_store` |
| old: `test/unit/codegen/test_lower.ml` "gater strips only the first variadic invalid index" | with three indices, only the first is stripped | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=three_indices`, `case=two_indices_two_gates` |
| old: `test/unit/codegen/test_lower.ml` "gater zeroes every lane of a gated image load" | the alternative is as wide as the load | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=shrink_load` (a four-wide `SHRINK`; images are excluded) |
| old: `test/unit/codegen/test_lower.ml` "gater folds a select into a load only when its value survives" | the fold applies only when the alternative survives the load's type | dropped: tinygrad HEAD folds without the guard (`gater.py:5-6`), and the lowerings checked never give the where the load's own gate, so no raven path rounds; a divergence waits for a failing rune test (reason b). `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=where_of_half`, `case=where_of_cast`, `case=where_of_cast_constant` pin the unguarded fold |
| old: `test/unit/codegen/test_lower.ml` "invalid index gate moves onto store" | through the lowering pipeline, a gated index gates its store | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=index_store`; the pipeline itself belongs to `Codegen`'s suite |
| old: `test/unit/codegen/test_lower.ml` "range comparison invalid value becomes gated store" | a stored `where g v invalid` gates the store | `Tolk_next.Gater › pm_move_gates_from_index › moves.golden › case=gated_value_kept` pins that the gater leaves it; the gating is `pm_remove_invalid`'s, in its module's suite |
| law | no load or store through a first index gated by `Invalid` remains; the pass is idempotent | `Tolk_next.Gater › laws › no access through a first index gated by invalid remains`, `moving the gates is idempotent` |

## Linearizer

The kernel goldens are compiled by tinygrad (`gen/codegen/late/linearizer.py`
lists them) and recorded at the pass each test applies: `<kernel>.golden` at
`linearize`, `<kernel>_unsplit.golden` at the final rewrite (for
`pm_split_ends`), `<kernel>_unchained.golden` at `pm_add_control_flow`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `codegen/late/linearizer.py` `linearize` (no test file pins the order) | the order of a compiled kernel: run counts, the kind priorities, parameters by slot, the structural tie-break | `Tolk_next.Linearizer › linearize › <kernel>_linear.golden` for 18 kernels (matmul with and without opts, softmax, a padded convolution, two reductions, a transpose, a symbolic size, a dependent loop bound, a CUDA reduction through workgroup memory with a barrier, a CUDA matmul with locals, four runtime kernels, four do-while loops) |
| tinygrad: `linearize` priorities | each rule of the ideal order on a graph built for it | `Tolk_next.Linearizer › linearize › orders.golden` (16 cases) |
| tinygrad: `helpers.py` `TUPLE_ORDER=0` | without the structural tie-break, equal priorities keep their topological order | `Tolk_next.Linearizer › linearize › matmul_linear_toposort.golden`, `conv_linear_toposort.golden`, `orders_output_toposort.golden` |
| tinygrad: `linearize` `DEBUG_LINEARIZE` | each node printed with its position, operation, ranges and priority | `Tolk_next.Linearizer › DEBUG_LINEARIZE › debug_linearize.golden` (a run of its own, with the variable set); `linearize › without DEBUG_LINEARIZE, nothing is printed` |
| tinygrad: `do_split_ends`, CPython's `TypeError` sorting range arguments that tie up to an axis type or where one identity is a prefix of the other (README CPython row) | the ranges nest by identity, a prefix first, then by axis type, the greatest innermost | `Tolk_next.Linearizer › pm_split_ends › ranges of one identity and different axis types nest by axis type` |
| tinygrad: `linearize`, CPython's `tuplize` comparison (README CPython row) | nodes whose sources differ only by a tag | `Tolk_next.Linearizer › linearize › nodes whose sources differ only by a tag are ordered by their later sources` |
| tinygrad: `CFGContext`, `pm_add_control_flow` | siblings chained by dependency then position, a loop's first child after its range, the sink's first child first | `Tolk_next.Linearizer › pm_add_control_flow › chains.golden` (10 cases), `Tolk_next.Linearizer › pm_add_control_flow › <kernel>_chained.golden` for 11 kernels |
| tinygrad: `CFGContext` (`assert y.src[1] not in x.backward_slice_with_self`) | a range that would run after a loop depending on it is refused | `Tolk_next.Linearizer › pm_add_control_flow › a range that would run after a loop depending on it is rejected` |
| tinygrad: `pm_split_ends` | an end split into one end per range, the greatest innermost; ranges of other sources; repeated ranges once; an end of no range | `Tolk_next.Linearizer › pm_split_ends › splits.golden` (12 cases), `Tolk_next.Linearizer › pm_split_ends › <kernel>_split.golden` for 8 kernels |
| tinygrad: `null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_dependent_loop_bound` | a loop bounded by a load: ranges, stores and ends in order, the ends closing the ranges in reverse | `Tolk_next.Linearizer › linearize › dependent_loop_bound_linear.golden`, `Tolk_next.Linearizer › pm_add_control_flow › dependent_loop_bound_chained.golden` |
| tinygrad: `null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_reduction`, `test_arange`, `test_kernel_info` | a program compiles; its name and opts | dropped here: `to_program` end to end, `Codegen`'s suite |
| tinygrad: `null/test_linearizer.py::TestLinearizer` (7 tests), `TestLinearizerRenderers` (2 tests) | load dedup, zero folding, accumulator types, reduce upcasts, locals with upcasts, all through `to_program` | dropped here: the expander, devectorizer and opts, in `Codegen`'s and `Postrange`'s suites |
| tinygrad: `null/test_linearizer_failures.py::TestLinearizerFailures::test_fail_1`, `runtime/test_linearizer_dumb.py::TestLinearizerFailure::test_failure_beam_mnist` | a kernel compiles through `to_program` | dropped here: `Codegen`'s suite |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_late_bias_load` | a bias is loaded after the reduction's loop ends | `Tolk_next.Linearizer › linearize › late_bias_load_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_two_nested_range_alt_indexing` | an ALU and a load sit between the two ranges | `Tolk_next.Linearizer › linearize › two_nested_range_alt_indexing_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_range_outer_op_before_phi` | one load is placed before the range | `Tolk_next.Linearizer › linearize › range_outer_op_before_phi_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_simple_unroll_no_between_phi_dependencies` | register stores fed by ALUs sit inside the loop | `Tolk_next.Linearizer › linearize › simple_unroll_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer` (the 16 other tests) | buffer dedup, range collapse, casts, launch dimensions, folds, grouped stores, on a device | dropped here: scheduling, the expander, `Gpudims` and execution, in their modules' suites |
| tinygrad: `runtime/test_wait_loop.py::TestWaitLoop::test_wait_loop`, `test_nested_loop_in_range`, `test_two_sequential_loops`, `test_loop_in_loop`, `test_wait_loop_spec` | do-while loops (`BACKEDGE`), alone, in a range, in sequence and nested | `Tolk_next.Linearizer › linearize › wait_loop_linear.golden`, `nested_loop_linear.golden`, `two_loops_linear.golden`, `loop_in_loop_linear.golden`, and their `_chained` goldens under `pm_add_control_flow`; their execution belongs to the execution tests |
| tinygrad: `runtime/test_wait_loop.py::TestWaitLoop::test_loop_carried_registers`, `test_register_pressure_loop`, `TestVolatileLoops::test_async_wait_ext` | register pressure and a volatile host signal | dropped here: register allocation (ISA, excluded) and execution |
| law | `linearize` is a topological order of the sink's nodes, the sink last, with and without the tie-break; after `pm_split_ends` and `pm_add_control_flow`, loops nest and a range's body runs inside its loop; `pm_split_ends` leaves one range per end, closes the same ranges, and is idempotent | `Tolk_next.Linearizer › laws` (7 properties over generated loop trees, the kernel goldens as examples) |
| old: `test/unit/codegen/test_linearizer.ml` `conditional_loop_nesting` | a do-while loop inside a range closes before the range | `Tolk_next.Linearizer › linearize › nested_loop_linear.golden`; `laws › after splitting and chaining, loops nest` |
| old: `test/unit/codegen/test_linearizer.ml` "multi-range End lowers to nested End_range pairs" | an end of two ranges becomes two nested ends | `Tolk_next.Linearizer › pm_split_ends › splits.golden › case=ranges_in_order`; `laws › pm_split_ends leaves each end closing one range` |
| old: `test/unit/codegen/test_linearizer.ml` "outer-range loads are scheduled before entering inner ranges", "nested range increases run_count" | fewer runs come first | `Tolk_next.Linearizer › linearize › orders.golden › case=run_count`, `case=hoisted_load` |
| old: `test/unit/codegen/test_linearizer.ml` "After nodes stay in Program ownership after linearize", "effect-only After nodes preserve store ordering" | `AFTER` nodes are placed after what they wait on | `Tolk_next.Linearizer › linearize › orders.golden › case=after_value`, `case=after_store` |
| old: `test/unit/codegen/test_linearizer.ml` "nested alt-index loads stay between the two ranges" | a gated load between two ranges | `Tolk_next.Linearizer › linearize › two_nested_range_alt_indexing_linear.golden` |
| old: `test/unit/codegen/test_linearizer.ml` "gated stores become IF/STORE/ENDIF", "single casted gated stores become IF/STORE/ENDIF", "bitcasted gated stores are not linearize-cleanup matches", "nested-cast gated stores are not linearize-cleanup matches" | gated stores become `IF`/`STORE`/`ENDIF` | dropped here: `codegen/__init__.py`'s `pm_linearize_cleanups`, in `Codegen`'s suite |
| old: `test/unit/codegen/test_linearizer.ml` "equal-priority nodes use structural tie-breaks" | `1+2` before `2-1` | `Tolk_next.Linearizer › linearize › orders.golden › case=structure_breaks_ties` |
| old: `test/unit/codegen/test_linearizer.ml` "late bias loads are scheduled after reduce end" | a bias load after the reduction's end | `Tolk_next.Linearizer › linearize › late_bias_load_linear.golden` |
| old: `test/unit/codegen/test_linearizer.ml` "outer ops are placed before loop phis" | a load used inside and after a loop is placed before it | `Tolk_next.Linearizer › linearize › range_outer_op_before_phi_linear.golden` |
| old: `test/unit/codegen/test_linearizer.ml` "loop-carried reg stores stay inside the range" | accumulator stores inside the loop | `Tolk_next.Linearizer › linearize › simple_unroll_linear.golden`; `laws › after splitting and chaining, a range's body runs inside its loop` |
| old: `test/unit/codegen/test_linearizer.ml` "gated loads without alts are rejected", "unlowered Reduce nodes are rejected", "graph IF nodes are rejected", "graph ENDIF nodes are rejected" | the old linearizer checked its input | dropped here: `linearize` orders any sink; the program specification (`Spec`) and `pm_linearize_cleanups` (`Codegen`) refuse these |
| old: `test/unit/codegen/test_linearizer.ml` "sibling ends under sink are ordered", "three sibling ends are chain-ordered" | sibling loops run one after another | `Tolk_next.Linearizer › pm_add_control_flow › chains.golden › case=siblings`, `case=three_siblings` |
| old: `test/unit/codegen/test_linearizer.ml` "three-range end exercises cfg nesting" | three ranges of one end nest | `Tolk_next.Linearizer › pm_split_ends › splits.golden › case=ranges_in_order`; `laws › after splitting and chaining, loops nest` |
| old: `test/unit/codegen/test_linearizer.ml` "two independent reduces are sequenced" | two reductions under the sink run one after the other | `Tolk_next.Linearizer › pm_add_control_flow › two_sums_chained.golden`, `Tolk_next.Linearizer › linearize › two_sums_linear.golden` |
| old: `test/unit/codegen/test_linearizer.ml` "cyclic control-flow edge is rejected" | a cycle is refused | `Tolk_next.Linearizer › pm_add_control_flow › a range that would run after a loop depending on it is rejected` |
| old: `test/unit/codegen/test_linearizer.ml` "empty Group is a valid no-op effect" | an empty group is placed | `Tolk_next.Linearizer › linearize › orders.golden › case=empty_group` |
| old: `test/unit/codegen/test_linearizer.ml` "load offsets follow lexical argument order" | constants tie-break by their text: 10, 100, 2 | `Tolk_next.Linearizer › linearize › orders.golden › case=constants_by_their_text` |
| old: `test/unit/codegen/test_linearizer.ml` "params ordered by index", "define_var ordered by name" | parameters by slot, variables by name | `Tolk_next.Linearizer › linearize › orders.golden › case=params_by_slot`, `case=variables_by_name` |
| old: `test/unit/codegen/test_linearizer.ml` "define_reg before define_local" | register storage before workgroup storage | `Tolk_next.Linearizer › linearize › orders.golden › case=buffers_before_local_buffers` |
| old: `test/unit/codegen/test_linearizer.ml` "three ranges with mixed kinds are sorted", "same-axis ranges are split by full range argument" | ranges nest by argument, identities of several parts included | `Tolk_next.Linearizer › pm_split_ends › splits.golden › case=axis_types`, `case=identities_of_parts` |
| old: `test/unit/codegen/test_linearizer.ml` "end with zero ranges passes through" | an end of no range is its value | `Tolk_next.Linearizer › pm_split_ends › splits.golden › case=no_range`, `case=source_without_ranges` |
| old: `test/unit/codegen/test_linearizer.ml` "barrier emission", "special emission", "cast and bitcast emission", "vectorize emission", "value index emission", "custom and custom_inline emission", "after on ptr stays in program", "group forwards first source" | each node kind becomes a `Program` instruction | dropped: the old linearizer built a `Program`; `linearize` returns the nodes themselves, and `laws › linearize is a topological order of the sink's nodes, the sink last` covers that every node is placed |

## Simplify

The suite is `Tolk_next.Simplify` (`codegen/simplify/`), written `SI` below.
`SI › tinygrad's kernels` holds kernels tinygrad compiles, recorded where a
pass running one of the matchers receives them, with what the matcher alone
makes of each: `pm_load_collapse`, `pm_split_ranges` and `pm_simplify_ranges`
at their codegen passes on the CPU renderer, and `pm_reduce_simplify` where
the scheduler collapses reductions (`get_kernel_graph`). `SI › tinygrad's
rewrites` holds hand-built cases of each matcher, each rewritten alone, from a
fresh context for the two that read one. `SI › laws` checks that every
reduction rewrite keeps the value of what it rewrites, at bindings of its
variables, ranges, parameters and storage, on those cases and on generated
sums; and that every range rewrite keeps what a kernel writes
(`Interpreter.writes`), on the cases, the recorded kernels and generated
kernels. `pm_simplify_ranges`' law starts from the symbolic pass the pipeline
runs before it: a guard that holds everywhere would grow a range to it
(`SI › tinygrad's rewrites › simplify_ranges.golden › case=guard_beyond_size`).

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_uop_symbolic.py::TestRangeSplitting::test_range_split_on_mod | a range taken modulo 2 splits, and the enclosing end closes the parts | `SI › tinygrad's rewrites › split_ranges.golden › case=nested_sink`; `SI › laws › split_ranges keeps the writes › split_ranges.golden › case=nested_sink` |
| tinygrad: null/test_uop_graph.py::TestReduceCollapse::test_multi_range_reduce_add | a sum of a sum over two ranges is the sum of two sums | `SI › tinygrad's rewrites › reduce_collapse.golden › case=sum_of_a_sum_over_two_ranges`; `SI › pm_reduce_collapse › a sum of a sum is the sum of the sums` |
| tinygrad: null/test_uop_graph.py::TestReduceCollapse::test_reduce_shapeless_const_unroll | a sum of a constant over an unroll range is the constant times its size | `SI › tinygrad's rewrites › reduce_unparented.golden › case=sum_over_an_unroll_range`; that no reduction survives `full_rewrite` is Codegen's section (L4) |
| tinygrad: null/test_simplify_valid_idx.py::TestRangeShrink (8 tests) | guarded ranges shrink to their greatest guard, unless read unguarded or reduced | `SI › TestRangeShrink › shrink_<case>_simplified.golden` (each case recorded where it reaches `simplify ranges`); the ranges left after `full_rewrite` are Codegen's section (L4) |
| tinygrad: null/test_arange.py::TestArange::test_cat_complexity, test_tri_complexity | an arange or a mask compiles to few operations | the collapse: `SI › tinygrad's kernels › arange_collapsed.golden`, `triu_collapsed.golden`; the estimates after `compile_linear` are Codegen's and Renderer's sections (L4) |
| tinygrad: null/test_schedule.py::TestSchedule::test_arange_sum, test_arange_sum_alt, test_permute_arange, test_arange_transposed, test_arange_index | aranges fuse and collapse | the collapse: `SI › tinygrad's kernels › arange_sum_collapsed.golden`, `arange_transposed_collapsed.golden`, `arange_index_collapsed.golden`; the kernel counts are the scheduler's section (L6) |
| tinygrad: null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_arange | an arange kernel's code | dropped here: rendered code, Codegen's section (L4) |

### old tolk: unit/codegen/test_simplify.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/codegen/test_simplify.ml "toposorts range children of End" | an end's ranges reordered so that a range comes after those its size reads | dropped: tinygrad's `flatten_range` keeps the order ranges are listed in; `SI › pm_flatten_range › ranges keep the order they are listed in`, `SI › tinygrad's rewrites › flatten_range.golden › case=end_of_a_dependent_range_first` |
| old: unit/codegen/test_simplify.ml "noop when ranges already sorted" | an end of ranges is left | `SI › tinygrad's rewrites › flatten_range.golden › case=end_of_ranges` |
| old: unit/codegen/test_simplify.ml "does not rewrite gated store gate as ranges" | a store's gate is left | `SI › pm_simplify_ranges › shrinking › a store's gate is no guard`; `SI › tinygrad's rewrites › simplify_ranges.golden › case=store_gate` |
| old: unit/codegen/test_simplify.ml "split and simplify do not crash on Copy" | a graph without ranges is left | `SI › tinygrad's rewrites › split_ranges.golden › case=mod_3_of_7` and the other fixed points; a copy never reaches these passes in tinygrad (the copy kernels of `schedule/__init__.py` hold stores) |
| old: unit/codegen/test_simplify.ml "nested sinks keep enclosing split binders consistent" | the end around a nested sink closes both parts | `SI › tinygrad's rewrites › split_ranges.golden › case=nested_sink` |
| old: unit/codegen/test_simplify.ml "splits Range(8) used with mod 2", "split produces correct sizes", "splits Range(12) used with floormod 4" | a range splits into its quotient and remainder | `SI › pm_split_ranges › a range taken modulo a divisor of its size splits in two`; `… › the outer part is numbered 0 and the inner 1 under the range's identity`; `SI › tinygrad's rewrites › split_ranges.golden › case=mod_2_of_8`, `case=mod_4_of_12` |
| old: unit/codegen/test_simplify.ml "no split when size does not divide constant" | | `SI › pm_split_ranges › a range taken modulo a number that does not divide its size stays`; `case=mod_3_of_7`, `case=mod_3_of_8`, `case=mod_16_of_8` |
| old: unit/codegen/test_simplify.ml "does not split Range(12) used with cmod 4" | | `SI › pm_split_ranges › a truncating remainder is no modulo`; `case=cmod_4_of_12` |
| old: unit/codegen/test_simplify.ml "merging three loops preserves their iteration count" | | `SI › pm_simplify_ranges › merging › ranges a kernel does not read merge`; `case=adjacent_three`, `case=adjacent_unused`; the writes law |
| old: unit/codegen/test_simplify.ml "merges adjacent ranges in End with same kind" | | `SI › pm_simplify_ranges › merging › adjacent ranges indexed contiguously merge into their product`; `… › the merged range keeps the first range's identity` |
| old: unit/codegen/test_simplify.ml "no merge when different kind" | | `SI › pm_simplify_ranges › merging › ranges of different axis types stay`; `case=adjacent_different_types`, `case=loop_and_reduce` |
| old: unit/codegen/test_simplify.ml "does not merge when floor div would increase divmod count" | | `SI › pm_simplify_ranges › merging › ranges whose merge adds a division stay`; `case=adjacent_one_used`, `case=adjacent_transposed` |
| old: unit/codegen/test_simplify.ml "nested sinks keep enclosing shrunk binders consistent" | | `SI › tinygrad's rewrites › simplify_ranges.golden › case=nested_sink`; the writes law |
| old: unit/codegen/test_simplify.ml "shrinks range with single guard", "picks max guard across multiple loads", "no shrink when unguarded elsewhere", "no shrink for reduce ranges", "shrink to single iteration" | | `SI › pm_simplify_ranges › shrinking` (the matching tests); `SI › TestRangeShrink`; `case=single_guard`, `two_guards`, `guarded_and_unguarded`, `guard_of_a_reduce_range`, `guard_of_one` |
| old: unit/codegen/test_simplify.ml "does not shrink stacked gated indexes", "does not shrink from later index coordinates" | | `SI › tinygrad's rewrites › simplify_ranges.golden › case=guards_in_a_stack`, `case=guard_on_a_later_index` |
| old: unit/codegen/test_simplify.ml "no shrink when guard >= range size" | | `SI › TestRangeShrink › shrink_guard_ge_max_simplified.golden`; alone, the pass grows the range to the guard: `case=guard_beyond_size` |
| old: unit/codegen/test_simplify.ml "shrink with store where invalid", "shrink with store where invalid flipped" | | `SI › TestRangeShrink › shrink_store_where_invalid_simplified.golden`, `shrink_store_where_invalid_flipped_simplified.golden`; alone, the pass leaves a selection of an invalid value (`case=store_where_invalid`), which the pipeline's symbolic pass moves onto the index first (`case=store_through_a_gated_index`) |
| old: unit/codegen/test_simplify.ml "separate store gate is preserved" | | `SI › pm_simplify_ranges › shrinking › a store's gate is no guard` |
| old: unit/codegen/test_simplify.ml "removes unparented range from ADD reduce", "removes unparented range from MUL reduce", "MAX reduce ignores unparented ranges", "noop when all ranges parented" | | `SI › pm_reduce_unparented` (the matching tests); `SI › tinygrad's rewrites › reduce_unparented.golden` |
| old: unit/codegen/test_simplify.ml "distributes add over reduce" | | `SI › pm_reduce_collapse › a sum of a sum is the sum of the sums`; `reduce_collapse.golden › case=sum_of_a_sum` |
| old: unit/codegen/test_simplify.ml "bound from above", "bound from below", "bound from two sides" | a masked sum is its count times its value | `SI › pm_reduce_collapse › a sum of a value below a bound is the bound times the value`, `… above a bound counts the rest`, `… between two bounds counts what lies between`; `reduce_simplify.golden › case=sum_below_a_bound`, `sum_above_a_bound`, `sum_between_bounds` |
| old: unit/codegen/test_simplify.ml "unparented range removed from ADD reduce" | | `SI › pm_reduce_simplify › an unparented range is removed` |
| old: unit/codegen/test_simplify.ml "mul casted bool becomes where" | | `SI › pm_reduce_collapse › a product by a comparison cast from a boolean is a selection`; `case=sum_of_a_product_by_a_cast_comparison` |
| old: unit/codegen/test_simplify.ml "multi-range reduce collapse" | | `SI › pm_reduce_simplify › a sum over two ranges collapses each in turn` |
| old: unit/codegen/test_simplify.ml "lift x*y out of reduce", "lift x+y out of reduce on lt" | | `SI › pm_reduce_collapse › a comparison of a product is solved by a rounded-up division`, `… of a sum is solved for its range`; `case=sum_below_a_scaled_range`, `sum_below_a_shifted_range` |
| old: unit/codegen/test_simplify.ml "lt lift matches a bare sum, not a cast of one" | | `SI › tinygrad's rewrites › reduce_simplify.golden › case=sum_below_a_cast_shifted_range` |
| old: unit/codegen/test_simplify.ml "reduce-fold counts clamp only at zero" | | `SI › pm_reduce_collapse › a count by variable bounds is clamped at zero`; `case=sum_below_a_variable`, `sum_above_a_variable`, `sum_between_variables` |
| old: unit/codegen/test_simplify.ml "collapses a range-bounded conditional sum" | | `case=integer_sum_below_a_bound` |
| old: unit/codegen/test_simplify.ml "AND on WHERE with define_var" | | `SI › pm_reduce_collapse › a parameter guarding a sum is lifted out of it`; `case=sum_gated_by_a_parameter` (a variable is a parameter) |
| old: unit/codegen/test_simplify.ml "collapses an arange row gather with independent output ranges" | | `SI › tinygrad's kernels › gather_collapsed.golden`, `embedding_collapsed.golden`, `arange_index_collapsed.golden` |
| old: unit/codegen/test_simplify.ml "collapses reduce over gated load", "reduce on gated load with casted range" | | `SI › pm_load_collapse › a sum of the value an index selects is the value at that index`, `… an index outside the range selects zero`; `load_collapse.golden › case=sum_selected_by_a_constant`, `sum_selected_by_a_cast_range` |
| old: unit/codegen/test_simplify.ml "collapses one-hot equality multiply" | | `load_collapse.golden › case=sum_of_a_one_hot_product` |
| old: unit/codegen/test_simplify.ml "lift x+y out of reduce on ne" | | `load_collapse.golden › case=sum_selected_by_a_shifted_range`, `sum_selected_by_a_shifted_cast_range`, `sum_selected_by_a_shifted_load` |
| old: unit/codegen/test_simplify.ml "undo rule: no math on loaded index", "undo rule ignores concrete loaded index" | | `SI › pm_load_collapse › a comparison of a shifted index read from memory is solved for it`, `… of a committed type is left` |
| old: unit/codegen/test_simplify.ml group "node_vmin / node_vmax" (17 tests) | bounds of nodes | dropped here: `vmin` and `vmax` are `uop/ops.py`'s, Ops' and Symbolic's sections |
| old: unit/codegen/test_simplify.ml group "promoting rule bodies" (4 tests) | a weak operand meets a committed one only promoted | the dtype of every node of every output is the golden's, which reading a graph derives and checks: `reduce_collapse.golden › case=sum_between_variables`, `reduce_unparented.golden › case=sum_over_a_symbolic_range`, `product_over_a_symbolic_range`, `load_collapse.golden › case=comparison_of_a_shifted_load_by_variables` |
