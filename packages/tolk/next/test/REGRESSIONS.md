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
