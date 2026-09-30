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

A test marked `(L3)` needs the symbolic rules, which Symbolic installs at L3:
it is tagged `L3` and left out of the default run until then. Its pair,
`O › resolve › simplify rejects a graph other than constants while the symbolic rules are not installed`,
is tagged `pre-L3` and pins the time before.

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
| tinygrad: null/test_uop_graph.py::TestUOpBroadcast::test_broadcast_row, test_broadcast_col, test_broadcast_lower_dim, test_broadcast_scalar, test_broadcast_symbolic_same_shape | elementwise operations broadcast shapes | `O › shapes › an elementwise operation broadcasts its sources' shapes`; `O › shapes › an elementwise operation keeps a symbolic shape` (L3) |
| tinygrad: null/test_uop_graph.py::TestUOpBroadcast::test_broadcast_axes | `broadcast_axes`, symbolic sizes, rejection | `O › shapes › broadcast_axes is the axes broadcasting adds or expands`; `O › shapes › broadcast_axes compares symbolic sizes` (L3) |

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
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_simple_int, test_int_add, test_rfloordiv, and the integer half of test_weak_const | `int()` of a typed, weak or summed integer | `O › resolve › to_int, to_float and to_bool read a literal`; `O › resolve › to_int reads a typed constant and an integer sum of constants` (L3) |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_weak_const (float half) | `float()` of a weak float | `O › resolve › to_int, to_float and to_bool read a literal` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_rtruediv, test_float_direct, test_ssimplify | float and remainder folding | Symbolic's section: without `symbolic` a float sum has no single-valued bounds |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_lt, test_leq, test_ne, test_ne_f, test_ngt | comparisons of constants | `O › resolve › to_bool decides comparisons of constants` (L3) |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_ambiguous_less_than | `resolve` falls back to its default | `O › resolve › resolve takes the default when the comparison is undecided` (L3) |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_var_cmp_t, test_var_cmp_t2, test_var_cmp_f, test_var_cmp_f2, test_max | bounds decide a comparison | `O › resolve › to_bool decides a comparison the bounds decide` (L3) |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_or_true, test_and_false | an absorbing boolean decides | `O › resolve › to_bool decides a disjunction with true and a conjunction with false` (L3) |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_or_false, test_and_true, test_x_lt_xp1, test_var_cmp_range, test_var_cmp_assert | an undecided condition raises | `O › resolve › to_bool rejects a condition with two possible values` (L3) |
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
| old: unit/uop/test_uop.ml flat_storage_parameters | a parameter is flat storage viewed as its shape | `O › graphs › storage.golden`; `O › graphs › symbolic_storage` (L3); `O › shapes › max_shape takes a symbolic size's greatest value` (L3); the image parameter is dropped (images are not ported) |
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
| old: unit/uop/test_uop.ml property_helpers_parity | sharding axis, shard shapes, bounds, movement arguments, stages, storage views, call output shapes | `O › several devices` (axis, sharding, shard shapes and bounds tests); `O › movement › marg reads each movement's argument and rejects other nodes`; `O › storage › a stage is its own base and storage, without buffer identity`; `O › storage › buf_uop is the storage a node accesses`; `O › graphs › symbolic_outputs` (L3) (a call output's shape takes the call's argument); `O › graphs › constants_like.golden` |
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
| old: unit/uop/test_uop.ml resolve_decides_comparisons_from_bounds | `resolve` | `O › resolve › resolve takes the default when the comparison is undecided` (L3); `O › resolve › resolve rejects a node that is not boolean` |
| old: unit/uop/test_uop.ml smax_smin_fold_when_bounds_decide | `smax`/`smin` | `O › resolve › smax and smin of integers are integers`; `O › resolve › smax and smin of a symbolic size bound it as max and min do` (L3); folding to a node is Symbolic's |
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
compare two splits of one axis and amount into different targets, since
`AxisType` has no order: `comparisons.golden` leaves those pairs out, and
`P › order › compare is a total order` covers them. A split that is not from
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
