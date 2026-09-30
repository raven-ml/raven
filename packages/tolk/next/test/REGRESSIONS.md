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

The suite is `Tolk_next.Helpers` (`helpers/test_helpers.ml`), written `Helpers`
below.

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
| tinygrad: null/test_helpers.py::TestCeilDiv::test_symbolic, test_symbolic_negative_offset | `ceildiv` on UOps | dropped: `ceildiv` on nodes has no reader among the ported files; `Helpers.ceildiv` is on integers |
| tinygrad: null/test_helpers.py::TestCount (2 tests) | `count` and its pickling | dropped: not ported, a counter reference (README exclusions) |
| tinygrad: null/test_helpers.py::TestFetch (9 tests) | `fetch` | dropped: url fetch is not ported (plan, L0 scope) |
| tinygrad: null/test_helpers.py::TestFullyFlatten (2 tests) | `fully_flatten` | dropped: not ported, it serves Python's typing (README exclusions) |
| tinygrad: null/test_helpers.py::TestMemoryview (3 tests) | `from_mv`, `to_mv`, `mv_address` | dropped: the ctypes helpers are not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestGetShape (2 tests) | `get_shape` | dropped: not ported (README exclusions) |
| tinygrad: null/test_helpers.py::TestPolyN (2 tests) | `polyN` | dropped: it lives with the symbolic integer type in `Ops` (README exclusions) |
| tinygrad: null/test_helpers.py::TestTimeToStr (7 tests) | `time_to_str` units, boundaries and width | Helpers › terminal text › time_to_str writes a duration as tinygrad does (`durations.golden`, every input of the class) |
| tinygrad: null/test_helpers.py::TestCStyleDivMod (4 tests) | `cdiv` and `cmod` by positive and negative divisors | dropped: `cdiv` and `cmod` are not ported (README); `Ops` divides constants as C does on `Dtype.value`, `O › exec_alu › exec_alu_values.golden` (the `Ops.CDIV` and `Ops.CMOD` rows) |
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
| tinygrad: null/test_device.py::TestDevVar::test_parse | DEV parses targets and prints them back | Helpers › Target › of_string reads a target as tinygrad does (`targets.golden`, every input of the test); Helpers › startup › reads DEV as targets separated by semicolons |
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
| old: unit/test_helpers.ml:49 concurrent allocations preserve live byte counts | memory counters under concurrent domains | dropped: `GlobalCounters` are read where kernels run and buffers are allocated, which is `tolk.next.engine`'s (README Exclusions, D3) |
| old: unit/test_helpers.ml:68 mem_used follows allocation and release | memory used rises and falls | dropped: `GlobalCounters` are read where kernels run and buffers are allocated, which is `tolk.next.engine`'s (README Exclusions, D3) |
| old: unit/test_helpers.ml:78 reset leaves mem_used alone | reset keeps the memory used | dropped: `GlobalCounters` are read where kernels run and buffers are allocated, which is `tolk.next.engine`'s (README Exclusions, D3) |
| old: unit/test_helpers.ml:189 nested duplicate overrides restore after an exception | duplicate bindings, raise, restore | Helpers › context › gives a setting bound twice its later binding, and restores the first value; Helpers › context › restores a setting when its function raises |
| old: unit/test_helpers.ml:197 overlapping domains retain their own contexts | domain-local overrides | Helpers › context › is not seen by the other domains; `… › binds for the domains spawned while it runs` |
| old: unit/test_helpers.ml:199 overlapping systhreads retain their own contexts | thread-local overrides | dropped: the threads of a domain share its settings, as the threads of a tinygrad process share its `Context` |
| old: unit/test_helpers.ml:201 snapshots are immutable and replace a worker's current context | context snapshots for workers | dropped: a domain starts with the values of the domain that spawns it, so workers need no snapshot; Helpers › context › binds for the domains spawned while it runs |
| old: unit/test_helpers.ml:202 exited scopes release their values | no leak of bound values | Helpers › context › keeps no bound value once it returns |
| old: unit/test_helpers.ml:206 unlimited and malformed quota retain available CPUs | cgroup quota parsing | dropped: the quota parser is private to `parallel`'s default and reads a Linux file; Helpers › settings › parallel is between one and the domains the runtime recommends |
| old: unit/test_helpers.ml:211 quota bounds workers by whole available CPUs | cgroup quota bound | dropped: as the row above |
| old: unit/test_helpers.ml:222 target strings preserve architecture and interface spelling | case kept in arch and interface | Helpers › Target › of_string reads a target as tinygrad does (`input=remote:host:2+nv:cuda:sm_89`) |
| old: unit/test_helpers.ml:230 target strings normalize empty fields without inventing defaults | empty fields print as nothing | Helpers › Target › of_string reads a target as tinygrad does (the six inputs of the test); Helpers › Target › of_string reads back what pp writes |
| old: unit/test_helpers.ml:236 target strings reject excess separators | two `+`, three `:` | Helpers › Target › of_string reads a target as tinygrad does (`input=PCI+NV+CUDA`, `input=CPU:CLANG:arm64:extra`) |
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

The suite is `Tolk_next.Dtype` (`dtype/test_dtype.ml`), written `D` below. A
golden check is named after its golden, and each of its rows is a test keyed by
its input cells, such as `D › truncate › truncation.golden › dtype=dtypes.half value=65520.0`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_dtype.py::TestEqStrDType::test_strs | a data type prints as `dtypes.<alias>` | `D › data types › properties.golden` (column `dtype`) |
| tinygrad: null/test_dtype.py::TestToDtype::test_dtype_to_dtype | `to_dtype` returns a DType as is | dropped: `of_string` takes a string, and a `Dtype.t` needs no conversion |
| tinygrad: null/test_dtype.py::TestToDtype::test_str_to_dtype | a name reads as its data type | `D › names › names.golden`; `D › names › of_string reads the name pp prints after dtypes.` |
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
| tinygrad: runtime/test_dtype.py::TestFp8sConversions::test_fp8e4m3_to_float, test_fp8e5m2_to_float, test_fp8e4m3fnuz_to_float, test_fp8e5m2fnuz_to_float, test_smallest_normals | decoding every byte | `D › storage › decode.golden`; `D › storage › a bitcast through a float gives back every 8- and 16-bit word` |
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
| old: unit/uop/test_dtype.ml const_float_identity | one NaN, two zeros | `D › constants › zero and negative zero are different constants`; the single NaN is dropped: a constant keeps a NaN's bits (D27), `D › constants › NaNs of different bits are different constants` |
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
| tinygrad: `null/test_upat_compile.py::TestUPatCompile::test_const_folding` | `GroupOp.ALU - {Ops.THREEFRY}` | `Tolk_next.Op › Set.diff › removes Threefry from alu and keeps the rest`; its compilation of the pattern is dropped with the other seven below |
| tinygrad: `null/test_upat_compile.py::TestUPatCompile` (7 other tests: `test_double`, `test_single`, `test_xpx`, `test_xp0`, `test_bool`, `test_single_c`, `test_range_named`) | the pattern compiler generates code for these patterns | dropped: the pattern compiler is excluded (README); patterns match directly, which the `O › patterns` group tests |
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
| tinygrad: null/test_uop_graph.py::TestMovementOps (2 tests) | `pm_mops` folds reshapes into indices | `Tolk_next.Prepare › pm_mops › rules ›` "an index of a reshape's added leading axis is its source" (`test_pm_mops_partial_reshape_index_removes_reshape`), "an index of a reshape whose trailing axes change is left as it is" (`…_suffix_mismatch_does_nothing`) |
| tinygrad: null/test_uop_graph.py::TestConstBufferize (2 tests) | `pm_const_buffer_folding` | `Tolk_next.Rangeify › get_kernel_graph › recorded ›` `setitem_column_kernels.golden` and `setitem_cube_kernels.golden` (Rangeify's rows left by other sections); `bufferize` itself is `O › arguments › reprs.golden` (name=bufferize) and `O › shapes › a stage puts its ranges' sizes in front` |
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
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_simple_int, test_int_add, test_rfloordiv, and the integer half of test_weak_const | `int()` of a typed, weak or summed integer | `O › resolve › to_z, to_float and to_bool read a literal`; `O › resolve › to_z reads a typed constant and an integer sum of constants` |
| tinygrad: null/test_uop_resolve.py::TestUOpResolve::test_weak_const (float half) | `float()` of a weak float | `O › resolve › to_z, to_float and to_bool read a literal` |
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
| tinygrad: null/test_uops.py::TestUOpMethod::test_const_nan_same | two `float('nan')` constants are one node | `O › identity › NaN constants of different bits are different nodes`, whose last claim is this; NaNs of different bits are not (D27) |
| tinygrad: null/test_uops.py::TestUOpStr (3 tests) | `str` is compact and `eval` reads it back | Render's section; `eval` is dropped (Python source) |
| tinygrad: null/test_uops.py::TestUPatHelpers::test_location | a pattern records its source location | dropped: `UPat.location` serves match statistics, not ported (README exclusions) |
| tinygrad: null/test_uops.py::TestUopsObject::test_timing | building 10k constants | dropped: a timing print |
| tinygrad: null/test_uops.py::TestUopsObject::test_nested | the device of a 10k-deep graph | `O › several devices › the device of a deep graph needs no deep recursion` |
| tinygrad: null/test_uops.py::TestUOpRender (7 tests) | `render` | Render's section |
| tinygrad: null/test_uops.py::TestContiguousViewOffset (7 tests) | `contiguous_view_offset` | `Tolk_next.Prepare › contiguous_view ›` "storage is its own view from 0" (`test_simple`), "a shrink of leading rows starts at their first element" (`test_shrink`), "an element is a view at its offset" (`test_shrink_to_one`), "an expand repeats" (`test_expand_is_none`), "a pad adds elements" (`test_shrink_invalid`), "a shrink of inner columns skips elements" (`test_strided`); `test_2d` by the law `contiguous_view › laws`; `contiguous_view` is Prepare's (D4) |
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
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_uop_scalar_const_lifts_kind | a scalar constant takes the kind of the other operand's type and stays weak, its value converted to that kind; a bare weak constant node and the scalar build one node | `O › elementwise › a weak constant takes the other operand's kind and stays weak`; `O › graphs › weak_promotion.golden` (`a + 1`, `a + 1.5`, `x + 2`); `O › constants › a constant without a type is its weak literal` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_int_binop (the dtype lines) | shifts and bitwise operations of weak and committed integers keep the left operand's type, and floats are refused | `O › data types › dtypes_of.golden` (op=Ops.SHL, Ops.SHR, Ops.AND, Ops.OR, Ops.XOR) |
| tinygrad: null/test_tensor_uop_representation.py (5 tests) | a realized `Tensor` is a BUFFER | the engine's section (L7): realization is `tolk.next.engine`'s (D3) |

### old tolk: unit/uop/test_uop.ml, and the bounds of unit/codegen/test_simplify.ml

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
| old: unit/uop/test_uop.ml wrapping_integer_bounds, unsigned_arithmetic_bounds_cover_emission | bounds widen to the type when arithmetic may wrap | `O › bounds › a committed integer that can leave its type has its bounds (D24)`; `O › bounds › an integer cast to a signed type it leaves wraps (D24)`; `O › bounds › bounds hold the value a committed integer wraps to (D24)`; the rows of `O › bounds › binary_bounds.golden` with dtype=dtypes.char, dtypes.uchar and dtypes.uint |
| old: unit/uop/test_uop.ml cast_bounds | casts keep what fits | `O › bounds › cast_bounds.golden` |
| old: unit/uop/test_uop.ml flat_storage_parameters | a parameter is flat storage viewed as its shape | `O › graphs › storage.golden`; `O › graphs › symbolic_storage.golden`; `O › shapes › max_shape takes a symbolic size's greatest value`; the image parameter is dropped (images are not ported) |
| old: unit/uop/test_uop.ml backward_slice_tracks_shared_dependencies | `backward_slice` | `O › graphs › backward_slice is the reached nodes without the root or call bodies` |
| old: unit/uop/test_uop.ml allocations_preserve_shape_and_address_space | `alloc` keeps shape, type and address space | `O › graphs › storage.golden`; `O › graphs › storage_like.golden`; the rejection of a local `alloc` with a device is dropped: tinygrad checks it in `placeholder` only (`O › storage › placeholder rejects a device for local storage`) |
| old: unit/uop/test_uop.ml placeholder_checks_shape_product, max_numel_checks_host_range, max_numel_handles_zero_after_large_dimensions | sizes past the largest int, and empty axes | `O › shapes › placeholder rejects a size past the largest int`; `O › shapes › max_numel is 0 when an axis is empty` |
| old: unit/uop/test_uop.ml movement_dimensions_do_not_wrap | a shrink past the largest int is rejected | `O › shapes › a shrink past the largest int is rejected, not wrapped` |
| old: unit/uop/test_uop.ml bitcast_dimensions_remain_exact_until_host_conversion | a byte count past the largest int | dropped: sizes are OCaml ints; see `O › shapes › placeholder rejects a size past the largest int` |
| old: unit/uop/test_uop.ml exact_symbolic_bounds | bounds are exact at any size; NaN bounds; a fractional cast | `O › bounds › bounds are exact integers, whatever their size`; `O › bounds › a NaN constant has its type's bounds`; `O › bounds › a typed constant outside its type has the type's bounds`; `parse_valid` is Symbolic's section |
| old: unit/uop/test_uop.ml stack_stage_slice_constructors, stack_promotes_all_operands, stack_prepends_leading_dim | stacks and stages | `O › graphs › stacks.golden`; `O › shapes › a stack prepends its length`; `O › shapes › a stage puts its ranges' sizes in front` |
| old: unit/uop/test_uop.ml uop_constructor_parity_shortcuts | shortcuts that return the node, an index of a stack | `O › elementwise › a cast or bitcast to the node's own type is the node`; `O › kernel nodes › an index of a stack by a constant is the element`; `O › kernel nodes › an index of a stack by a negative constant counts from the end, as a Python tuple`; `O › kernel nodes › an end of no ranges, and an after of nothing, are the node`; `O › elementwise › contiguous stages a placed value, and is the node otherwise`; the rendered strings are Render's |
| old: unit/uop/test_uop.ml const_scalar_payload_constructors | typed constants, NaN, -0.0, Invalid | `O › graphs › typed_constants.golden`; `O › identity › NaN constants of different bits are different nodes`; `O › identity › zero and negative zero are different nodes`; `O › constants › Invalid ignores the type` |
| old: unit/uop/test_uop.ml call_constructor_parity | `call` and `call_with_outputs` | `O › calls › call rejects a body that computes a value`; `O › calls › call rejects a range leaking out of its body, but a device range`; `O › graphs › outputs.golden` |
| old: unit/uop/test_uop.ml deviceless_partition_selection | an MSTACK of unplaced lanes | dropped: a placement names its devices (`device` has no absent lane), as tinygrad's `device` returns a tuple of strings |
| old: unit/uop/test_uop.ml property_helpers_parity | sharding axis, shard shapes, bounds, movement arguments, stages, storage views, call output shapes | `O › several devices` (axis, sharding, shard shapes and bounds tests); `O › movement › marg reads each movement's argument and rejects other nodes`; `O › storage › a stage is its own base and storage, without buffer identity`; `O › storage › buf_uop is the storage a node accesses`; `O › graphs › symbolic_outputs.golden` (a call output's shape takes the call's argument); `O › graphs › constants_like.golden` |
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
| old: unit/uop/test_uop.ml runtime_realization_state_parity | realized buffers | the engine's section (L7): realization is `tolk.next.engine`'s (D3) |
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
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax "const int", "const bool", "define_var", "vectorize bounds" | bounds of constants, variables and stacks | `O › bounds › a constant is its own bounds`; `O › bounds › a comparison of constants is decided`; `O › bounds › a variable offset or scaled moves its bounds`; `O › bounds › a stack's bounds span its values, Invalid left out` |
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax "range" | a range counts from 0 below its end | `O › bounds › a hardware index counts from 0 to below its end`; `O › bounds › an empty range divides to 0` |
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax "add", "sub", "neg", "mul with negative" | sums, differences and products of intervals; a negation that may leave its type | `O › bounds › a variable offset or scaled moves its bounds`; `O › bounds › binary_bounds.golden` (op=Ops.ADD, Ops.SUB, Ops.MUL); `O › bounds › a committed integer that can leave its type has its bounds (D24)`; `O › bounds › bounds hold the value a committed integer wraps to (D24)`. The old `neg` case gave `-r` for `r` in `[0, 3]` the whole int32 range; the negation fits, and its bounds are `[-3, 0]` |
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax "idiv positive", "mod constant", "cmplt known true", "cmplt unknown" | division, remainder and comparison | `O › bounds › binary_bounds.golden` (op=Ops.FLOORDIV, Ops.FLOORMOD, Ops.CMPLT) |
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax "max", "where int", "and mask", "shl constant", "shr constant" | maxima, selections, masks and shifts | `O › bounds › maximum then minimum clamps`; `O › bounds › a selection spans both branches`; `O › bounds › a mask bounds a variable by the mask`; `O › bounds › a shift by a constant shifts the bounds` |
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax "float binary falls back to dtype" | a float operation has its type's bounds | `O › bounds › a product with an unbounded float is unbounded, never NaN` |

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
| old: unit/uop/test_serialize.ml "node tags round-trip" | a tag reads back | `Graph › a graph reads back as itself ›` "a tuple tag", "a bytes tag", "a string tag" |
| old: unit/uop/test_serialize.ml "split range identities and WMMA axes round-trip" | tensor core axes and split ranges read back | `Graph › a graph reads back as itself ›` "a tensor core product", "a range" |
| old: unit/uop/test_serialize.ml "compiled target survives serialization and separates program keys", "compiled program round-trips physically", "semantic_key is preserved" | a program and its target read back as the same node, so with the same key | `Graph › a graph reads back as itself › a program` |
| old: unit/uop/test_serialize.ml "import reuses live structurally-equal nodes" | reading gives the live node | `O › identity › building a graph twice gives the same node` (hash-consing) |
| old: unit/uop/test_serialize.ml "import rejects malformed input" | malformed text is refused | `Graph › reading ›` (every test) |
| old: unit/uop/test_serialize.ml "deep chains round-trip without stack overflow" | a deep graph reads back | `Graph › a graph reads back as itself › a chain of 100000 nodes` |
| old: unit/uop/test_serialize.ml "export rejects gradient functions" | a gradient function is not written | dropped: rune owns differentiation, and the format writes no function (test/README.md) |
| old: unit/uop/test_serialize.ml "imported internal buffer slots can collide" | slots of read buffers | dropped: slots are the caller's (D3) |
| old: unit/uop/test_serialize.ml "cross-process export/import lands on this universe" | reading in another process | dropped: the graph format has no process universe |
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
| old: unit/test_contiguous_view.ml (17 tests) | contiguous views of storage | Prepare's section, test by test: `contiguous_view` is one of its functions (D4) |

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
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_int_binop (the `type_verify([bad], spec_shared)` lines) | bitwise operations on float32 and weak floats fail the shared spec | `S › verdicts › verdicts.golden` (an and of float32, an and of weak floats); `type_verify`'s list form is dropped: every tinygrad caller passes a graph, and a one-node list is the node's graph with sources that pass |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_int_binop (the dtype lines) | shift and bitwise data types | Ops' section (`dtype_of`) |
| tinygrad: null/test_uops.py::TestUOpMethod::test_invalid_dtype_and_consumers (the `type_verify(u, spec_shared)` loop) | Invalid matches any type in a stack, a sum, a where, both sides of a comparison, an index | `S › verdicts › verdicts.golden` (a stack of float32 and the invalid constant, a sum of float32 and the invalid constant, a where over float32 and the invalid constant, a cmplt of the invalid constant and float32, a cmplt of float32 and the invalid constant, an index by the invalid constant) |
| tinygrad: null/test_uops.py::TestUOpMethod::test_invalid_dtype_and_consumers, test_remove_invalid_stack_lanes (the `spec_program` lines) | `pm_remove_invalid`'s output is a program | dropped here: `pm_remove_invalid` is codegen (L4), whose suite checks its output with `Spec.program` |
| tinygrad: null/test_uops.py::TestUOpMethod::test_const_default_dtype_is_derived (`SPEC=2`) | constants build under the construction check | `S › construction › rejects a new node iff the full specification does not accept it` (constants are leaves of every graph); the data types are Ops' section |
| tinygrad: null/test_uops.py::TestUPatHelpers::test_location | spec_shared's first pattern is located in spec.py | dropped: pattern locations are match tracking (README exclusions) |
| tinygrad: null/test_uop_symbolic.py::TestMoveWhereOnLoad::test_bool_index_preserves_dtype | the rewrite's output passes spec_shared | `Tolk_next.Symbolic › tinygrad › tests.golden › TestMoveWhereOnLoad.test_bool_index_preserves_dtype` |
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
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | `check` and `KernelOptError` | Postrange's section: a refusal is `Scheduler.apply_opt`'s `Error`, and `apply_opts` raises it as `Invalid_argument`, so `Opt` holds the types alone |
| tinygrad: `runtime/test_kernel_opts.py`, `runtime/test_linearizer.py`, `null/test_linearizer.py`, `runtime/test_opt_gemm.py`, `runtime/test_tensor_cores.py`, `null/test_uops.py`, `null/test_uops_stats.py`, `null/test_gen_float4.py`, `null/test_linearizer_rewrite.py`, `runtime/test_custom_kernel.py`, `null/test_custom_kernel.py` | optimisations applied to kernels, and the `KernelOptError` of those that do not apply | Postrange's section: `apply_opt` makes the checks and returns a refusal as an `Error` |
| old: `unit/uop/test_uop.ml` debug_prints_rich_args_dataclass_style | "Opt repr" of a split into an upcast | `P › printing › pp is tinygrad's repr › reprs.golden › opt=Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))` |
| old: `unit/codegen/test_postrange.ml`, `unit/opt_fuzz/tolk_opt_fuzz.ml` | applying optimisations | Postrange's section |

## Uop_weak

The suite is `Tolk_next.Uop_weak` (`uop/uop_weak/`), written `W` below. A golden
check is named after its golden, which holds a graph and its rewrite by one
pass.

| Source | Behaviour | Outcome |
|---|---|---|
| DIVERGENCES D44 | an integer cast of a weak expression computes in integers, a 64-bit unsigned one included | `W › pm_commit_weak › a 64-bit unsigned cast of a weak expression keeps its value past a float's precision (D44)`, `W › laws › pm_commit_weak computes an integer cast in integers (D44)` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_expression_anchors_at_strong_lub | a cast commits a mixed expression at the cast's width, whatever the default float | `W › pm_commit_weak › cast_anchors_a_mixed_expression_at_the_cast.golden`; the `Tensor` half is dropped: `Tensor` surface, the frontend is nx |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_cast_weak_expression_commits_at_cast_floor | a cast below the default float never narrows | `W › pm_commit_weak › cast_never_narrows_below_the_default_float.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_store_weak_value_uses_destination_dtype | a store commits a weak value at its destination's type | `W › pm_commit_weak › store_commits_its_value_at_the_destination.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_srcs_commit_only_at_a_concrete_lub | weak sources stay weak without a committed peer; a where's weak arm stays bare | `W › pm_commit_weak › weak_sources_stay_weak_without_a_committed_peer.golden`; `W › pm_commit_weak › where_keeps_a_weak_arm_bare.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_derivable_const_rounds_at_the_derived_width | a derivable literal is rounded to its peer's width, in place | `W › pm_commit_weak › peer_rounds_a_derivable_literal.golden`; the folds `x * 1` and `x * -1` that follow are `S › tinygrad › tests.golden › TestWeakPromotion.test_derivable_const_rounds_at_the_derived_width` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_shift_lhs_commits_the_node | a shift commits its weak operand, and so the node | `W › pm_commit_weak › shift_commits_its_weak_operand.golden` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_committed_const_conversion_folds | `symbolic_simple` folds a cast of a committed constant | `S › tinygrad › tests.golden › TestWeakPromotion.test_committed_const_conversion_folds` |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_uop_scalar_const_lifts_kind, test_weak_int_binop, test_float_unary_on_weakint_stays_weak | how arithmetic on nodes promotes weak constants, and the specification of weak bitwise operations | dropped here: Ops' section (`test_uop_scalar_const_lifts_kind` and the dtype lines of `test_weak_int_binop`), Spec's (L1, its `type_verify` lines) and Dtype's (`test_float_unary_on_weakint_stays_weak`) |
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
| tinygrad: `null/test_helpers.py::TestCeilDiv::test_symbolic`, `test_symbolic_negative_offset` | `ceildiv` of a node writes as `((v+5)//6)` | dropped: `ceildiv` on nodes has no reader among the ported files; `Helpers.ceildiv` is on integers |
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

## Decomp_dtype

The suite is `Tolk_next.Decomp_dtype` (`codegen/decomp/decomp_dtype/`),
written `DD` below. A golden `<type>_<kernel>.golden` holds the kernel that
tinygrad's CPU pipeline hands to its "decomp dtypes" pass for a `Tensor`
program, with `<type>` emulated, and what the pass makes of it. For the narrow
floats, tinygrad's `f2f` and `f2f_clamp` are held as placeholders, which the
suite replaces with `Decomp_dtype`'s conversions (D9). A cast to a narrow float
from a type more precise than a float32, a cast of an emulated 64-bit integer
to a float32 or narrower (both D9's) and a cast of a float to an emulated 64-bit
integer (D22's) have no golden: their value laws pin them. Each golden is
checked on a target that lacks the type. The check on a target told to emulate
it, and the run of a kernel that computes outside a narrow float, emulated
against native, cover one kernel of each type by default and every kernel under
the slow tag. The value laws compare emulated kernels with `Dtype.truncate` and
the codecs, on a sample of the codes by default and on every code under the
slow tag.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/test_dtype.py::TestEmulatedHalf`, `TestEmulatedBFloat16Type`, `TestEmulatedFp8e4m3`, `TestEmulatedFp8e5m2` (the `TestDType` cases, emulated) | casts to and from each type, its bit reinterpretations and its arithmetic, on data without subnormals, within a tolerance that allows flushing | `DD › emulated narrow floats › *` for every narrow float: casts from float32, doubles, 8 to 64-bit integers and the other narrow floats; casts to float32; bit reinterpretations; each exact, subnormals included (D9); `DD › goldens › <type>_*.golden` and `... computes what it computes natively` |
| tinygrad: `runtime/test_dtype.py::TestEmulatedInt64DType`, `TestEmulatedUInt64DType`, with `test_int64_to_uint32_to_int64`, `test_uint64_load`, `test_uint64_cast_double` | 64-bit casts, loads and bit reinterpretations through 32-bit words | `DD › emulated 64-bit integers › {long,ulong} from int/uint/char/bool`, `{int,char,ushort} from {long,ulong}`, `... bitcast to ...`, `... to float32, the nearest`, `... to double, the nearest`; `DD › D9 › an emulated 64-bit integer converts to a float32 once`; `DD › goldens › {long,ulong}_*.golden` |
| tinygrad: `runtime/test_dtype_alu.py::TestDTypeALU::test_emulated_float16`, `test_emulated_bfloat16`, `test_emulated_fp8e4m3`, `test_emulated_fp8e5m2`, `test_emulated_fp8e4m3fnuz`, `test_emulated_fp8e5m2fnuz` | binary operations of the emulated type against numpy, within a tolerance for flushed subnormals | `DD › emulated narrow floats › emulated <type> arithmetic computes in float32 and rounds once, at the store` (add, sub, mul, max, exact); `... an emulated comparison of <type>s compares their values`; `... an emulated selection of <type>s keeps their codes` |
| tinygrad: `runtime/test_dtype_alu.py::TestDTypeALU::test_emulated_*_unary` (float types) | exp2, log2, sqrt, sin and the others on an emulated type | `DD › goldens › <type>_exp2.golden`, `<type>_sqrt.golden` (the operand widened, the result narrowed); the functions themselves are `Transcendental`'s |
| tinygrad: `runtime/test_dtype_alu.py::TestDTypeALU::test_emulated_int64`, `test_emulated_uint64`, `test_emulated_int64_unary`, `test_emulated_uint64_unary` | integer binary and unary operations of emulated 64-bit integers | `DD › emulated 64-bit integers › {long,ulong} {add,sub,mul,and,or,xor,max,truncating division,truncating remainder,cmplt,cmpeq,cmpne,negation,selection}` (edges of the type included) |
| tinygrad: `runtime/test_dtype_alu.py::TestDTypeALU::test_emulated_shl`, `test_emulated_shr` | shifts of int64 and uint64 by 0, 5, 31, 32 and 62 or 63 | `DD › emulated 64-bit integers › {long,ulong} shl`, `... shr` (tinygrad's values as examples, then drawn counts below 64) |
| tinygrad: `runtime/test_tensor_variable.py::TestTensorVariable::test_long_variable_emulated_raises` | an emulated 64-bit variable raises | `DD › pm_dtype_decomps › a 64-bit integer variable cannot be emulated` |
| tinygrad: `runtime/test_dtype_spec.py` `_assert_eq` | an emulated float compares with a tolerance, "denormals are zero" | replaced by exact comparisons (D9): `DD › D9 › an emulated narrow float keeps its subnormals, both ways` |
| tinygrad: `runtime/test_tensor_cores.py::TestTensorCores::test_tensor_cores_emulated_half` | tensor cores with emulated halves on the PYTHON device | dropped: the tensor core rewrite is `Tc`'s, the run is the executor's; `DD › goldens › half_*.golden` covers the emulation of halves |
| tinygrad: `null/test_randomness.py::TestRandomness::test_threefry_doesnt_use_long` | a program with THREEFRY has no 64-bit values on a target without them | dropped here: the whole pipeline is `Codegen`'s (L4); the emulation it relies on is `DD › emulated 64-bit integers` |
| tinygrad: `null/test_dtype_weak.py` (an emulated float constant), `null/test_uop_symbolic.py::test_mul_by_zero_casted_to_emulated_dtype` and the constant bit reinterpretation of an emulated float | folding of constants of emulated types | dropped here: `Uop_weak`'s and `Symbolic`'s sections |
| tinygrad: `codegen/decomp/dtype.py` `pm_float_decomp` (no test) | storage retyped to the unsigned integer of the width; vector accesses split per lane; loads widened, stores narrowed; bit reinterpretations of loads, from and to float32; casts clamped; arithmetic, stacks and lanes computed in float32 | `DD › goldens › <type>_{add,mulsub,maximum,where,exp2,sqrt,sum,lt,from_float,to_float,from_char,to_char,flip,gather,pad,bitcast_to_storage,bitcast_from_storage,bitcast_sum}.golden` for each narrow float |
| tinygrad: `codegen/decomp/dtype.py` `pm_long_decomp` (no test) | 64-bit storage, indices, loads, stores, constants, comparisons, casts, shifts, selections, the other ALU operations and bit reinterpretations split into words | `DD › goldens › {long,ulong}_*.golden` (22 kernels each, the division as the signed quotient and the unsigned remainder); `DD › pm_dtype_decomps › 64-bit storage holds two 32-bit words per element` |
| tinygrad: `codegen/decomp/dtype.py` `do_dtype_decomps`, `pm_dtype_decomps` (no test) | the types found are emulated in promotion order; an unsigned 64-bit integer goes with the signed one | `DD › goldens › half_fp8e4m3_cast.golden`, `bfloat16_half_cast.golden`, `ulong_named_alone.golden`; `DD › pm_dtype_decomps › unsigned 64-bit integers are emulated when the signed ones are`, `a target with every data type keeps its kernels`, `a kernel without narrow floats or 64-bit integers is kept`, `a setting that names no data type is refused` |
| tinygrad: `codegen/decomp/dtype.py` `f2f`, `f2f_clamp` (no test) | the conversions, and their only-narrow-float-and-float32 contract | `DD › f2f › *` (every 8-bit code, sampled 16-bit codes, exhaustive under slow), `DD › f2f_clamp › *`, `DD › NaNs › *`, `DD › f2f › f2f converts only between a narrow float and a float32` |
| tinygrad: `codegen/decomp/dtype.py` `f2f`, `f2f_clamp` and the CAST rule, D9's facets | flushed subnormals, overflow above the greatest value, saturated infinities, two roundings from wide sources and from emulated 64-bit integers, negative zero in fnuz | `DD › D9 › *`, one test per facet |
| old: `unit/codegen/test_decompositions.ml` "long storage doubles flat size", "unbounded param keeps unbounded size" | a 64-bit parameter holds twice as many 32-bit words, or no size | `DD › pm_dtype_decomps › 64-bit storage holds two 32-bit words per element`, `64-bit storage of no known size keeps none` |
| old: `unit/codegen/test_decompositions.ml` "scalar variables fail instead of narrowing their binding" | a 64-bit variable is refused | `DD › pm_dtype_decomps › a 64-bit integer variable cannot be emulated` |
| old: `unit/codegen/test_decompositions.ml` "MUL lowers", "IDIV lowers", "MOD lowers" | multiplication, division and remainder of words | `DD › goldens › long_mul.golden`, `long_div.golden`, `ulong_mod.golden`; `DD › emulated 64-bit integers › * mul`, `* truncating division`, `* truncating remainder` |
| old: `unit/codegen/test_decompositions.ml` "long comparisons split before arithmetic" | comparisons read split words | `DD › goldens › {long,ulong}_{lt,eq}.golden`; `DD › emulated 64-bit integers › * cmplt`, `* cmpeq`, `* cmpne` |
| old: `unit/codegen/test_decompositions.ml` "CAST float->long lowers", "CAST float->long high half uses reciprocal", "CAST long->int lowers", "BITCAST long->long lowers" | casts between floats, int32 and 64-bit integers, and between the 64-bit integers | `DD › goldens › {long,ulong}_{to_char,bitcast}.golden`; `DD › emulated 64-bit integers › * from float32, rounded towards zero` (with floats past 2^31 as examples), `an emulated cast of a float32 to a 64-bit integer converts no float to a word that cannot hold it` (D22), `* to float32, the nearest`, `int from *`, `* bitcast to *`; the word arithmetic of a cast to a float32 is D9's `long_to_float` (`DD › D9 › an emulated 64-bit integer converts to a float32 once`) |
| old: `unit/codegen/test_decompositions.ml` "CONST halves truncate to int32", "untagged CONST is low half" | a constant splits into its two words | `DD › goldens › {long,ulong}_add_const.golden` (2^40 + 5) |
| old: `unit/codegen/test_decompositions.ml` "untagged INDEX is not rewritten", "tagged INDEX narrows storage before reindexing", "tagged INDEX preserves multi-index tail" | an index into 64-bit storage becomes an index of the word, keeping its gate | `DD › goldens › {long,ulong}_*.golden` (every index), `{long,ulong}_pad.golden` (a masked index); the tags that carry a word are the pass's own |
| old: `unit/codegen/test_decompositions.ml` "variable SHL uses native narrow shift", "variable SHR uses native narrow shift" | a shift by a variable uses 32-bit shifts | `DD › goldens › {long,ulong}_shl_by.golden`; `DD › emulated 64-bit integers › * shl`, `* shr` |
| old: `unit/codegen/test_decompositions.ml` "bf16 load promotes to f32", "f32 store demotes to bf16 bits", "bf16 vector load reindexes SHRINK", "compact-float accesses narrow storage before reconstruction" | loads widen, stores narrow, vector accesses split per lane, storage retyped first | `DD › goldens › bfloat16_{to_float,from_float,gather}.golden`, `<fp8 or half>_add.golden` (vector loads through SHRINK); `DD › emulated narrow floats › an emulated cast of a <type> to a float32 is exact`, `... of float32s to <type> rounds as Dtype folds` |
| old: `unit/codegen/test_decompositions.ml` "compact-float arithmetic stores retain numeric conversion" | an arithmetic result is narrowed at its store | `DD › goldens › <type>_{add,mulsub}.golden`; `DD › emulated narrow floats › emulated <type> arithmetic computes in float32 and rounds once, at the store` |
| old: `unit/codegen/test_decompositions.ml` "compact-float masked loads encode numeric fallbacks", "compact-float conditional loads encode weak numeric fallbacks" | a masked load's fill value is a number of the type | `DD › goldens › <type>_pad.golden` (a fill of 1.5), and its run against the native kernel |
| old: `unit/codegen/test_decompositions.ml` "compact-float gathers preserve raw storage", "... conditional selection preserves raw storage", "... value indexing preserves raw storage", "... copies preserve raw storage", "... copies preserve vector windows", "... copies preserve gates and raw alternatives", "... copies preserve address validity masks" | a copy, gather or selection keeps the stored code | tinygrad converts through float32, and D9 makes that exact: `DD › emulated narrow floats › an emulated copy of <type>s keeps every code`, `an emulated selection of <type>s keeps their codes`, `DD › NaNs › an emulated copy keeps every NaN code, quieting a signalling one`, `an emulated selection keeps NaN codes, quieting a signalling one`; `DD › goldens › <type>_{flip,gather,pad,where}.golden` |
| old: `unit/codegen/test_decompositions.ml` "compact-float bitcast stores preserve their gate", "compact-float bitcast stores retain destination effects", "gated f32 store is not decomposed" | a gated store, or one ordered after other effects | dropped: the old tolk's store gate and effect sources; a gate rides on the index (`DD › goldens › <type>_pad.golden`), and a bit reinterpretation stored is `<type>_bitcast_from_storage.golden` |
| old: `unit/uop/test_weak.ml` "gated long index narrows", "gated long index keeps its width for huge buffers" | the width of an index into 64-bit storage | dropped here: `Uop_weak`'s section |
| old: `unit/frontend/test_run.ml` "long cumprod of 8-bit floats" | a compiled scan of emulated 8-bit floats | dropped here: compiled runs are the executor's (L7) and rune's (L9); the kernels' emulation is `DD › goldens › fp8*_sum.golden` |

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
| old: unit/uop/test_symbolic.ml "an offset crosses a comparison only without wrapping" | `c0 + x < c1` over uint8 keeps a wrapping offset | `S › integers wrap (D24) › an offset crosses a comparison only where neither side wraps` (uint8 `u - 1 < 255`, weak and committed) |
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
| old: unit/uop/test_symbolic.ml lt_fold (every test but "lt cdiv fold") | comparisons fold; a float comparison keeps its rounding; `(x / y) / z` | `S › symbolic › comparisons` (every test, `c0 + x < c1 stays for floats, where moving c0 rounds` among them); `S › symbolic › terms › (x / y) / z stays (D24)` |
| old: unit/uop/test_symbolic.ml where_fold "cast stays outside a conditional", "Boolean selection stays inside its integer cast" | a cast of a selection stays | `S › symbolic › selections › a cast of a selection stays outside it`; `S › tinygrad › tests.golden › TestSymbolic.test_where_cast` |
| old: unit/uop/test_symbolic.ml reduce "mul-term hoist floats non-range factors out of a lowered reduce" | factors move out of a kernel reduction | `S › sym › factors independent of a sum's ranges move out of it`; `S › sym › only non-negative factors move out of a maximum` |
| old: unit/uop/test_symbolic.ml reduce "add tensor reduce floats const and preserves axes" | a tensor-level reduction keeps the factors that vary along its axes | dropped: `sym` runs on kernels, whose reductions name their ranges; tinygrad HEAD's `reduce_mul_chain` reads a reduction's ranges and does not see a tensor-level one's axes |
| old: unit/uop/test_symbolic.ml load_store (3 tests) | loads and stores of invalid and gated indices | `S › invalid values › a load from an invalid index is its alternative, or 0`; `S › sym › storing a selection of the loaded value stores where it differs` |
| old: unit/uop/test_symbolic.ml sigmoid (3 tests) | `x * (1 / (1 + x))` stays at float | `S › sym › reciprocals of products stay (D24)`; `S › floats keep IEEE values (D24) › reciprocal and sigmoid forms stay` |
| old: unit/uop/test_symbolic.ml simplify_valid (3 tests) | a bitwise and is no valid; a clause others read comes first; the rewrite fires on the raw predicate | `S › tinygrad › tests.golden › TestValidIdxSimplification.test_bitwise_and_is_not_a_valid`; `S › conditions › simplify_valid › a clause on an expression others read is applied first`; `S › conditions › pm_simplify_valid › a conjunction is simplified` |
| old: unit/uop/test_symbolic.ml uop_given_valid "a load keeps its own gate" | `uop_given_valid` leaves a load's gate | dropped: tinygrad HEAD substitutes into the load's gate; `pm_simplify_valid` leaves a gated value that reads an index alone (`S › tinygrad › tests.golden › TestInvalidIndex.test_gated_load_keeps_index_valid`) |
| old: unit/uop/test_symbolic.ml uop_given_valid "a clause on a loaded value still applies" | a clause bounds a loaded value | `S › conditions › uop_given_valid › a clause on a loaded value bounds it` |
| old: unit/uop/test_symbolic.ml masked_div | `(x & -4) // 4` | `S › symbolic_simple › zeros › a mask of the bits a division by a power of two drops is removed` |
| old: unit/uop/test_symbolic.ml unpack_u64 (3 tests) | a packed 64-bit integer unpacks, a wide high half does not | `S › symbolic_simple › powers › a 64-bit integer packed from two halves unpacks to the half read` |
| old: unit/uop/test_symbolic.ml mop_cleanup (5 tests) | movement cleanups | Movement's section |
| old: unit/uop/test_symbolic.ml remove_invalid (3 tests) | invalid gates and lanes become zeros of their type | `S › pm_remove_invalid` (every test) |
| old: unit/uop/test_symbolic.ml "END preserves effects" | an end of folded ranges is its store; a live range stays; a constant backedge condition stays | `S › symbolic › ordering › an end of constant ranges only is its store`; `S › symbolic › ordering › an end drops the ranges that became constants`; `S › symbolic › ordering › a backedge keeps a constant condition` |
| old: unit/uop/test_symbolic.ml "distributed negation keeps scaled terms shared" | a negated sum distributes and folds its coefficients; a float one stays | `S › sym › a negated sum with a scaled term folds each term's coefficient`; `S › sym › -(x + y) is -x + -y for integers, and stays for floats (D24)`; `S › floats keep IEEE values (D24) › signed zeros: a negated sum and complementary selections` |
| old: unit/uop/test_symbolic.ml rule body promotion (4 tests) | a rule builds weak constants and promotes an integer exponent | `S › symbolic_simple › powers › c ** x computes in float for an integer exponent`; `S › symbolic › terms › a term's new coefficient is a weak constant`; the nested division is Divandmod's section |
| old: unit/uop/test_symbolic.ml integer width folding (4 tests) | 64-bit arithmetic that fits computes in 32 bits; bounded cast chains collapse | `S › symbolic › casts › 64-bit arithmetic that can overflow 32 bits stays`; `S › symbolic › casts › a cast chain of a bounded integer is one cast, of an unbounded one two` |
| old: unit/engine/test_symbolic.ml (every test) | symbolic sizes run on a device | dropped here: execution, Engine's section (L7) |
| old: unit/codegen/test_simplify.ml (every group but node_vmin / node_vmax) | `codegen/simplify.py`'s matchers | dropped here: Simplify's section (L3) |
| old: unit/codegen/test_simplify.ml node_vmin / node_vmax (19 tests) | bounds of nodes | dropped here: Ops' section, one row per group of tests |

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
| old: unit/codegen/test_simplify.ml "mul casted bool becomes where" | | `SI › pm_reduce_collapse › an integer product by a comparison cast from a boolean is a selection`; `SI › pm_reduce_collapse › a float product by a comparison cast from a boolean stays (D24)`; `case=sum_of_a_product_by_a_cast_comparison` |
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
| old: unit/codegen/test_simplify.ml group "node_vmin / node_vmax" (19 tests) | bounds of nodes | dropped here: `vmin` and `vmax` are `uop/ops.py`'s, Ops' section |
| old: unit/codegen/test_simplify.ml group "promoting rule bodies" (4 tests) | a weak operand meets a committed one only promoted | the dtype of every node of every output is the golden's, which reading a graph derives and checks: `reduce_collapse.golden › case=sum_between_variables`, `reduce_unparented.golden › case=sum_over_a_symbolic_range`, `product_over_a_symbolic_range`, `load_collapse.golden › case=comparison_of_a_shifted_load_by_variables` |

## Coalesce

The suite is `Tolk_next.Coalesce` (`codegen/late/coalesce/`), written `C`
below. `<kernel>.golden` is the sink that `memory_coalescing` receives when
tinygrad compiles a kernel (for Clang, and for CUDA where the kernel needs
workgroups), and `_coalesced`, `_scalar` and `_allow_half8` its result for a
renderer with vector accesses, one without, and with `ALLOW_HALF8=1`. The
hand-built cases are `accesses.golden` for `memory_coalescing` and
`indices.golden` for `indexing_simplify`. `DMC` and `ALLOW_HALF8` are read
once per process, so the groups tagged `dmc` and `allow_half8` run in
processes of their own with the variable set.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_basic` | an upcast elementwise add loads and stores four floats at once | `C › memory_coalescing › test_gen_float4.py › basic has (2, 1) loads and stores of four floats`; `C › memory_coalescing › kernels › basic` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_multidim` | two upcasts | `C › memory_coalescing › test_gen_float4.py › multidim has (4, 2) loads and stores of four floats` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_unaligned_load` | loads shifted by one element are not merged, the store is | `C › memory_coalescing › test_gen_float4.py › unaligned_load has (0, 1) loads and stores of four floats` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_multidim_unaligned_load` | the same over two upcasts | `C › memory_coalescing › test_gen_float4.py › multidim_unaligned_load has (0, 2) loads and stores of four floats` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_sometimes_unaligned` | a convolution whose loads are aligned only sometimes | `C › memory_coalescing › test_gen_float4.py › sometimes_unaligned has (0, 0) loads and stores of four floats` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_multidim_sometimes_unaligned` | the same with the output upcast too | `C › memory_coalescing › test_gen_float4.py › multidim_sometimes_unaligned: one vector store, a vector load or not`; the golden pins tinygrad's `(1, 1)` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_expand` | an expanded operand is not contiguous | `C › memory_coalescing › test_gen_float4.py › expand has (0, 1) loads and stores of four floats` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_heterogeneous` | one operand aligned, the other not | `C › memory_coalescing › test_gen_float4.py › heterogeneous has (1, 1) loads and stores of four floats` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_aligned_variable` | an offset by a variable that is a multiple of 4 | `C › memory_coalescing › test_gen_float4.py › aligned_variable has (2, 1) loads and stores of four floats` |
| tinygrad: `null/test_gen_float4.py::TestFloat4::test_float4_unaligned_variable` | an offset by a variable that is a multiple of 2 | `C › memory_coalescing › test_gen_float4.py › unaligned_variable has (1, 1) loads and stores of four floats` |
| tinygrad: `null/test_uops.py::TestMemoryCoalescing::test_volatile_view_not_coalesced` | a volatile parameter viewed at another type is never merged | `C › memory_coalescing › left alone › a volatile parameter viewed at another type`; `accesses.golden › case=volatile_view` |
| tinygrad: `null/test_uops.py::TestLowerIndexDtype::test_gated_shrink_lowers_to_selected_width` | lowering the index width of a gated `SHRINK` | dropped here: `pm_lower_weak` is `Uop_weak`'s |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_load_dedup` | overlapping reads load each element once | `C › memory_coalescing › test_linearizer.py › load_dedup: one to four loads of three overlapping elements`; `kernels › load_dedup` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_grouped_store_locals_and_globals` | with locals, both the local and the global stores of a matmul are vectors | `C › memory_coalescing › test_linearizer.py › grouped_store: every store to local or global memory is a vector` (on CUDA); `kernels › grouped_store`. The barrier and `IF` counts are the linearizer's |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_reduce_upcast`, `runtime/test_linearizer.py` `test_simple_unroll_no_between_phi_dependencies`, `test_grouped_store_phis`, `test_grouped_store_values`, `test_grouped_store_local_only`, `runtime/test_kernel_opts.py` upcast cases | registers, phis and opts of whole programs | dropped here: they pin the expander, the linearizer and the opts (skipped in tinygrad where marked) |
| tinygrad: `null/test_simplify_valid_idx.py::TestValidIdxSimplification::test_cumsum`, `test_simplify_within_valid1`, `test_valid_becomes_const1`, `test_valid_becomes_const2`, `test_valid_with_non_const_rhs` | an index simplified under its gate | `C › indexing_simplify › indices.golden › case=cumsum`, `within_valid`, `becomes_constant`, `becomes_constant_nested`, `non_constant_bound`: the same gated loads through `indexing_simplify` alone (tinygrad runs them through `sym` and `pm_move_where_on_load`, Symbolic's) |
| tinygrad: `null/test_simplify_valid_idx.py::TestValidIdxSimplification::test_simplify_within_valid2`, `test_bitwise_and_is_not_a_valid`, `test_valid_order_matters1`, `test_valid_order_matters2`, `test_valid_stronger_bound_first`, `test_simplify_valid_from_div`, `test_load_in_valid`, `test_from_merge_views` | `simplify_valid` and `sym` | dropped here: Symbolic's section |
| tinygrad: `null/test_simplify_valid_idx.py::TestValidIdxSimplification::test_valid_becomes_const1_z3` | the simplification checked by z3 | dropped: z3 is not a dependency (README); the value of the same load is `case=becomes_constant` |
| tinygrad: `null/test_simplify_valid_idx.py::TestImageSimplification` (16 tests), `test_drop_non_monotonic_window`, `test_drop_gate_committed_in_the_index_pass`, `TestImageStore::test_half_store_converts_lane_by_lane` | image indices, image valid dimensions and image stores | dropped: image paths are excluded (README); `C › README › an image access, through two indices, is not simplified` pins the exclusion |
| tinygrad: `null/test_simplify_valid_idx.py::TestDropTrueGate::test_drop_true_gate_on_index` | an index gated by `True` loses its gate | `C › indexing_simplify › indices.golden › case=true_gate`: `indexing_simplify` alone keeps the gate; `sym`, which tinygrad composes with it, drops it (Symbolic's section) |
| tinygrad: `null/test_simplify_valid_idx.py::TestDropTrueGate::test_const_gate_clause_is_not_moved_to_load`, `TestRangeShrink` (8 tests) | `pm_move_where_on_load`, range shrinking | dropped here: Symbolic's and Simplify's sections |
| tinygrad: `codegen/late/coalesce.py` `memory_coalescing`, lines 111-167 | the grouping key (op, buffer, base, gate, argument), the lengths per element type, `must_divide`, `DMC`, `ALLOW_HALF8` | `C › memory_coalescing › accesses` (54 cases, each against `accesses_coalesced`, `accesses_scalar` and, in its process, `accesses_allow_half8`); `C › memory_coalescing › lengths`; `C › ALLOW_HALF8`; `C › DMC` |
| tinygrad: `codegen/late/coalesce.py` `memory_coalescing`, the DSP lengths (`:138`) | 128 down to 4 elements, unaligned | `C › README › a DSP renderer merges as any other: aligned runs of four at most`, `C › README › a DSP renderer leaves 8-bit integers alone` (README: DSP excluded) |
| old: `unit/codegen/test_images.ml` "adjacent float loads become coalesced load with lane indexes" | | `C › memory_coalescing › lengths › four consecutive loads from offset 0 are one load of four`; `lengths › a merged load's elements are read back by index` |
| old: `unit/codegen/test_images.ml` "adjacent float stores become vector store with vector value" | | `C › memory_coalescing › lengths › four consecutive stores are one store of four`; `lengths › a merged store stores the stack of the former values` |
| old: `unit/codegen/test_images.ml` "volatile accesses through bitcast views never coalesce" | loads and stores | `C › memory_coalescing › left alone › a volatile parameter viewed at another type`, `left alone › stores to a volatile parameter` |
| old: `unit/codegen/test_images.ml` "access flags separate adjacent loads and stores" | `nontemporal` splits a run, and the merged access keeps it | `C › memory_coalescing › accesses › accesses_coalesced.golden › accesses.golden › case=nontemporal`, `case=half_nontemporal` |
| old: `unit/codegen/test_images.ml` "CPU and CUDA agree on repeated stores" | a store stored twice is one store; two stores of one element are refused | `C › memory_coalescing › left alone › a store stored twice is one store`; `C › memory_coalescing › errors › two stores to one element` (only `supports_float4` of a renderer is read, so one renderer stands for both; the old message is tinygrad's assertion, and tolk.next raises `Invalid_argument`) |
| old: `unit/codegen/test_images.ml` "gated memory ops are rejected" | | `C › memory_coalescing › errors › a gated load`, `errors › a gated store` |
| old: `unit/codegen/test_images.ml` "non-index memory ops are rejected" | | `C › memory_coalescing › errors › a load of a parameter`, `errors › a load through a shrink` |
| old: `unit/codegen/test_images.ml` groups "image valid dimensions", "coalesce image selection", "image stores", "load-store indexing strips coordinate casts" | | dropped: image paths are excluded (README) |
| old: `unit/codegen/test_images.ml` "remove invalid zeroes a sentinel at its consumer's dtype", "lower index dtype concretizes index binary math", "move where on value index keeps loads late", "lower pipeline applies renderer extra matcher" | | dropped here: Symbolic's, Uop_weak's and the pipeline's sections |
| old: `unit/test_cstyle.ml` vector pointer casts and `__builtin_nontemporal_load` | rendering vector accesses | dropped here: `Renderer.Cstyle`'s section (L5) |
| old: `parity/*/stage7_*.expected` | vector accesses in whole compiled kernels | dropped here: end-to-end parity (plan §4, slow); the kernel goldens compare this pass alone |

## Postrange

The suite is `Tolk_next.Postrange` (`codegen/opt/postrange/`), written `PR`
below. A kernel golden is the sink that compiling a real kernel hands to
`apply_opts`. `cases.golden` lists 316 cases, each a kernel, a renderer
(`renderers.golden`: CPU, Metal, CUDA sm_89 and AMD gfx1100, the last three
with their tensor cores), the optimisations asked for, the settings, and
tinygrad's outcome. An accepted case's graph golden is what `apply_opts`
returns (`PR › apply_opts optimises a kernel as tinygrad does`). A refused
case also checks that the refused optimisation leaves the scheduler as it
was (`PR › a refused optimisation leaves the scheduler as it was`), and
`axes.golden` is the scheduler's view after the optimisations: every axis
query, the coloured shape, and the axes the last one made (`PR › the
scheduler's axes after optimising are tinygrad's`). tinygrad's runtime tests
compare the outputs of a kernel run with and without the optimisations. Here
the interpreter compares the writes, before and after, of every accepted
case up to 2^18 iterations that it can evaluate (`PR › optimising keeps a
kernel's writes`, slow above 2^12). The slow fuzzer applies 400 random
sequences to ten small kernels (`PR › each random optimisation is refused as
a whole or keeps the kernel's writes`).

Cases are named in the table below by their `cases.golden` row. tinygrad's
`Tensor.rand(...).realize()` inputs become `Tensor.empty`, which schedules
the same kernels.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/test_kernel_opts.py::test_opt_without_axis`, `test_swap_invalid_arg` (`None`, `True`), `test_padto_arg` (`True`) | an axis or argument that is no integer | dropped: `Opt.t`'s fields are integers and its split targets a closed type (static types, Opt's section); the integer arguments are `swap_invalid_arg_0`, `swap_invalid_arg_1`, `padto_arg_*` |
| tinygrad: `runtime/test_kernel_opts.py::test_local_and_grouped_reduce` | locals, groups and upcasts on `sqrt + sum.exp` | `local_and_grouped_reduce_0` to `_10` |
| tinygrad: `runtime/test_kernel_opts.py::test_grouped_reduce_with_local_upcast_padto` | groups with pads and full splits | `sum_and_max_grouped`, `flip_pad_sum_grouped`, `strided_conv_grouped` |
| tinygrad: `runtime/test_kernel_opts.py::test_unrolled_padded_cumsum` | | `cumsum_unrolled_padded` |
| tinygrad: `runtime/test_kernel_opts.py::test_upcasts`, `test_full_upcast` | | `elementwise_upcast_2`, `_4`, `_8`, `elementwise_full_upcast` |
| tinygrad: `runtime/test_kernel_opts.py::test_matmul`, `test_matmul_upcast_group` | | `matmul_0` to `_8`, `matmul_upcast_group` |
| tinygrad: `runtime/test_kernel_opts.py::test_double_reduce` | | `double_reduce_0` to `_13` |
| tinygrad: `runtime/test_kernel_opts.py::test_padto_matmul`, `test_padto_upcasted_not_ok` | pads, and a pad of an upcast axis refused | `padto_matmul_*`, `padto_upcasted_*` (`_6` to `_8` refused) |
| tinygrad: `runtime/test_kernel_opts.py::test_padto_sum_ok`, `test_padto_sum`, `test_padto_max`, `test_padto_where`, `test_padto_where_multioutput` | pads under sums, maxima, casts, comparisons and two outputs | `padto_shrunk_*`, `padto_exp_sum*`, `padto_compare_sum*`, `padto_max_*`, `padto_where*` |
| tinygrad: `runtime/test_kernel_opts.py::test_padto_group_full_unroll_sum`, `test_padto_unrolled_sum`, `test_padto_unrolled_max`, `test_padto_unrolled_upcast`, `test_padto_unrolled_prod` | | `padto_group_full_unroll_sum`, `padto_unrolled_*` |
| tinygrad: `runtime/test_kernel_opts.py::test_padto_nested_reduce` | the pad's gate and the inner reduction's identity | `padto_nested_*`; the values `wanna_output` checks are the writes law's |
| tinygrad: `runtime/test_kernel_opts.py::test_padto_unindexed_reduce`, `test_padto_reduce_identity`, `test_padto_masked_reduce` | | `padto_unindexed_*`, `padto_twice`, `padto_all`, `padto_masked_*`, `padto_prefix_masked_sum` |
| tinygrad: `runtime/test_kernel_opts.py::test_padto_arg` | a multiple of at most 1 | `padto_arg_0` to `_2` |
| tinygrad: `runtime/test_kernel_opts.py::test_color_shapes_with_local` | the colour of each axis | `color_shapes_*`; `PR › the shape and name of a kernel are coloured by the roles of its axes › colors.golden` |
| tinygrad: `runtime/test_kernel_opts.py::test_arange_opts`, `test_top_split_non_reduce_axis`, `test_double_sum_group` | | `arange_local*`, `top_split_upcast`, `double_sum_*` (refused: a group inside another reduction) |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores`, `test_tensor_cores_codegen` | each tensor core of each renderer, on its own dimensions | `tc_<renderer>_<n>_basic` for every tensor core; the rendered instructions are the renderers' (L5) |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_padded_uops`, `test_tensor_cores_padded` | a pad only from `tc_opt=2`, never a pad of more than four times the work | `tc_<renderer>_0_padded_*`, `_small_n`, `_small_m`, `_small_k`, on each renderer's first tensor core |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_extra_locals`, `test_tensor_cores_upcast_shared_axis`, `test_tensor_core_opts` | optimisations after a tensor core | `tc_<renderer>_tiled_extra_locals`, `tc_<renderer>_batched_upcast`, `tc_<renderer>_half_128_*` |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_padto_warp`, `test_tensor_cores_group_reduce`, `test_tensor_cores_failed_padto`, `test_tensor_cores_nested_reduce`, `test_tensor_cores_contracted_m` | refusals after or in a tensor core; a failed pad leaves the kernel | `tc_<renderer>_padto_warp`, `_group_*`, `_failed_padto`, `_nested_reduce`, `_contracted_m`, each checked to leave the scheduler as it was |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_padto_unroll`, `test_tensor_cores_padto_masked_operand`, `test_tc_shape_padded`, `test_tc_padto_full_upcast` | | `tc_<renderer>_padto_unroll`, `_padto_shifted_operand`, `_padto_masked_operand`, `_shape_padded`, `_padto_full_upcast` |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_multi_reduce` | the nine choices of axes of a convolution | `tc_metal_conv_0` to `_8` |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_unroll_phi`, `test_tensor_cores_unroll_casted_phi`, `test_tensor_cores_unroll_casted_phi_with_children` | an unroll after a tensor core | `tc_<renderer>_unroll`, `tc_<renderer>_unroll_relu`; where the accumulator lives is the expander's (L4, Codegen) |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_nan`, `test_tensor_cores_emulated_half`, `test_tensor_cores_partial_sum_in_accumulator` | values computed on a device | dropped here: execution (plan §4, slow, L5) |
| tinygrad: `ALLOW_TF32` on CUDA (`postrange.py:170`) | a float tensor core only with TF32 allowed | `tc_cuda_5_basic` (allowed), `tc_cuda_float_without_tf32` (refused) |
| tinygrad: `null/test_custom_kernel.py::test_gemm_group_refused`, `test_gemm_unroll_refused`, `test_loop_acc_gemm_tc_refused` | an axis with no reduction cannot be grouped or unrolled; a serial loop takes no tensor core | `custom_gemm_group`, `custom_gemm_unroll`, `tc_amd_loop_acc_tc` |
| tinygrad: `runtime/test_custom_kernel.py::test_local_reduce` | reduced local and warp axes | `local_sum_axes`, `warp_sum_axes` (`axes.golden`) |
| tinygrad: `runtime/test_custom_kernel.py::test_split_range_id_free_of_loop` | a split's new axis takes an identity after a loop's | `loop_counter_upcast` (`made`: `2:2:UPCAST`) |
| tinygrad: `runtime/test_custom_kernel.py::test_stage_then_reduce` | weak axes outside a buffered value stay weak | `stage_then_reduce_globals`, `stage_outside_an_output_globals` |
| tinygrad: `runtime/test_custom_kernel.py` `test_group_reduce_split_range`, `test_nested_group_reduce`, `test_reg_stage_then_reduce`, `test_reg_placeholder_then_reduce`, the gated and unshard tests | lowering of kernels that ask for no optimisation | dropped here: the expander, local buffers and control flow (L4, Codegen) |
| tinygrad: `null/test_linearizer_rewrite.py::test_kernel_info` | no optimisation when none is asked; a kernel keeps its name | `PR › apply_opts › applies nothing to a kernel that asks for no optimisation`; `› keeps a kernel's name other than test` |
| tinygrad: `runtime/test_linearizer.py`, `null/test_linearizer.py`, `runtime/test_opt_gemm.py`, `null/test_uops.py::test_mulacc_unrolled`, `null/test_gen_float4.py`, `null/test_uops_stats.py` (the gemm estimates) | kernels optimised to test later passes | dropped here: those passes' sections (Coalesce, Linearizer, Codegen, Renderer's estimates) |
| tinygrad: `codegen/opt/postrange.py` `apply_opt` checks (no test) | each check's boundary | `split_axis_negative`, `split_axis_past_end`, `split_last_axis`, `padto_axis_past_end`, `swap_axis_past_end`, `split_amount_one`, `split_amount_negative`, `local_without_locals`, `unroll_over_32`/`unroll_32`, `upcast_over_16`/`upcast_16`, `upcast_reduce`, `unroll_global`, `split_not_dividing`, `group_over_shared_memory`/`group_at_shared_memory`, `local_after_group_over_shared_memory`, `unroll_without_reduce`, `padto_upcast_axis`, `padto_quadruple`/`padto_under_quadruple`, `padto_symbolic_axis`, `split_symbolic_axis`, `padto_warp_axis`, `swap_not_global`, `swap_on_cpu`, `tc_on_cpu`, `tc_not_first`, `tc_negative_axis`, `tc_select_*`, `tc_opt_out_of_range`, `tc_use_tc_*`, `tc_without_reduce`, `tc_on_max`, `tc_axis_out_of_choices` |
| tinygrad: `codegen/opt/postrange.py` `split_targets`, `Scheduler.copy`, `shift_to`, `get_optimized_ast`, `apply_opts` (no test) | | `PR › split_targets`, `PR › Scheduler`, `PR › shift_to`, `PR › apply_opts`; a split by 1 stalls tinygrad's rewrite, and raises up front here: `PR › shift_to › raises Invalid_argument on an amount of 1` |
| tinygrad: `codegen/opt/postrange.py` `args_from_ast`, BEAM search | device buffers for a search | dropped: execution is `tolk.next.engine`'s (README); `apply_opts` takes the search as `~beam`, `PR › apply_opts › searches with beam instead, when given` |
| tinygrad: `codegen/opt/postrange.py:112` the DSP upcast cap | | dropped: DSP is excluded (README) |
| tinygrad: `postrange.py:114` a shared-memory check whose symbolic size cannot be decided raises `ValueError` | | `group_maybe_beyond_symbolic_shared_memory`, refused with `Invalid_argument`; `group_within_symbolic_shared_memory` fits |
| old: `unit/codegen/test_postrange.ml` "splits range evenly", "input_new_rng is used as provided node", "rejects non-divisible amount" | | `PR › shift_to › splits an axis into its quotient and a new axis of the amount`, `› makes the new axis of the range it is given`, `› raises Invalid_argument on an amount that does not divide the axis` |
| old: `unit/codegen/test_postrange.ml` "top=true reverses expression order", "full amount creates size-1 replaced range", "replaced range drops old parents like tinygrad replace" | | `PR › shift_to › keeps the kernel's writes, once flattened` (from the top); `elementwise_full_upcast` (the size-1 axis leaves `axes.golden`); every split's graph golden |
| old: `unit/codegen/test_postrange.ml` "SPLIT uses absolute reduction axis indices", "LOCAL accepts a contracted reduction axis", "GROUPTOP on reduce creates a local reduction range", "UNROLL after GROUPTOP", "combined LOCAL + GROUPTOP + UNROLL + UPCAST", "double GROUPTOP on reduce", "LOCAL splits global into local tile", "UPCAST on global range", "UPCAST with amount=0 uses full range size", "UPCAST with amount=0 uses vmax extent" | | `matmul_*`, `double_reduce_*`, `local_and_grouped_reduce_*` and their `axes.golden` rows |
| old: `unit/codegen/test_postrange.ml` "SPLIT rejects reduction kinds without a REDUCE owner", "SPLIT validates amount and target before changing state", "UPCAST rejects amount > 16", "UNROLL rejects amount > 32", "UPCAST rejects reduce axis", "UNROLL rejects non-reduce axis", "LOCAL without renderer locals rejected", "shared memory budget exceeded" | | the boundary cases above; a split to a reduce, warp or global role cannot be written (`Opt.target`) |
| old: `unit/codegen/test_postrange.ml` "shared memory products cannot overflow the host integer" | a 2^60 local axis | `group_beyond_host_integers` |
| old: `unit/codegen/test_postrange.ml` "shared memory budgets prove symbolic local extents" | fits at 8, refused at 16 | `group_within_symbolic_shared_memory`; at 16 tinygrad cannot decide and raises (the row above) |
| old: `unit/codegen/test_postrange.ml` "shift_to validates source kinds and preserves index dtype" | | `PR › shift_to › raises Invalid_argument on a target its axis's role cannot split to`; an int32 axis is refused by tinygrad, whose size is then a cast (`split_int32_axis`) |
| old: `unit/codegen/test_postrange.ml` "local and warp reductions retain independent output threads" | | `local_sum_axes`, `warp_sum_axes` |
| old: `unit/codegen/test_postrange.ml` "PADTO preserves extents beyond host integers" | | `padto_beyond_host_integers`; a whole-axis split of it, `local_whole_axis_beyond_host_integers` |
| old: `unit/codegen/test_postrange.ml` "PADTO pads axis to next multiple", "PADTO keeps store target as Index", "PADTO keeps load sources as guarded Index nodes", "PADTO preserves existing index validity", "PADTO guards unsafe pad ops in reduce backward slice", "PADTO guards max reduce on reduce axis" | | `padto_matmul_*`, `padto_masked_*`, `padto_exp_sum*`, `padto_max_*` graph goldens |
| old: `unit/codegen/test_postrange.ml` "PADTO rejects upcast axis", "PADTO rejects warp axes", "PADTO rejects a multiple of one" | | `padto_upcast_axis`, `padto_warp_axis`, `padto_arg_2` |
| old: `unit/codegen/test_postrange.ml` "SWAP exchanges equal-sized axes without erasing unrelated tags", "SWAP exchanges two global axes", "SWAP exchanges full range identity arguments", "SWAP rejects non-global axes" | | `swap_keeps_tags`, `swap_globals`, `swap_then_split`, `swap_not_global` |
| old: `unit/codegen/test_postrange.ml` "upcast products remain exact beyond host integers" | 2^64 | `upcasts_beyond_host_integers` (`upcast_size`) |
| old: `unit/codegen/test_postrange.ml` "upcast products retain symbolic extents" | | `group_within_symbolic_shared_memory` (`full_shape` holds `n`); no Tensor program upcasts a symbolic axis, since `upcastable_dims` leaves it out |
| old: `unit/codegen/test_postrange.ml` "rngs sorted by axis_to_pos then axis", "rngs filters out size-1 ranges", "upcastable_dims and unrollable_dims" | | every `axes.golden` row |
| old: `unit/codegen/test_postrange.ml` "conditional loops are scopes rather than numeric optimization axes" | | `loop_counter_upcast` |
| old: `unit/codegen/test_postrange.ml` "copy preserves independent optimization state" | | `PR › Scheduler › an optimisation applied to a copy leaves the original as it was`, `› ... to the original leaves its copy as it was` |
| old: `unit/codegen/test_postrange.ml` "loop-to-global ignores ranges closed by nested END tails" | | `nested_end_globals` |
| old: `unit/codegen/test_postrange.ml` "postrange flatten preserves closed range dependencies", "postrange flatten does not merge through extra floor div" | | `flatten_keeps_a_closed_extent`; every graph golden flattens |
| old: `unit/codegen/test_postrange.ml` "filters symbolic params and sorts by slot" (`bufs_from_ast`) | | dropped: `args_from_ast` is `tolk.next.engine`'s (README) |
| old: `unit/codegen/test_postrange.ml` "get_optimized_ast produces valid kernel_info", "get_optimized_ast name generation", "kernel identity does not depend on earlier compilations" | | `PR › Scheduler › get_optimized_ast names the kernel name_override`; `PR › apply_opts › names a kernel named test after its reduction and its axes`; every golden's name, all optimised in one process |
| old: `unit/codegen/test_postrange.ml` "apply_opts respects opts_to_apply", "opts_to_apply applied in order", "beam_search closure is called", "hand_coded closure is called", "already-optimized kernel returns unchanged" | | every accepted case; `PR › apply_opts` |
| old: `unit/codegen/test_postrange.ml` "LOOP ranges become GLOBAL on GPU", "LOOP ranges stay LOOP on CPU", "reduce ranges stay REDUCE after conversion" | | `PR › apply_opts › hand-optimises a kernel that asks for nothing, after making its weak outputs global`; `PR › Scheduler › convert_loop_to_global leaves a kernel for a renderer without locals` |
| old: `unit/codegen/test_postrange.ml` "TC basic apply creates WMMA", "TC with padding (tc_opt=2)", "TC rejects non-reduce kernel", "TC rejects invalid tc_select", "TC must be first opt", "TC use_tc=2 skips WMMA construction" | | `tc_*_basic`, `tc_*_padded_2`, `tc_without_reduce`, `tc_select_*`, `tc_not_first`, `tc_*_tiled_use_tc_2` |
| old: `unit/opt_fuzz/tolk_opt_fuzz.ml` | random optimisations against the unoptimised kernel, run on a device | `PR › each random optimisation is refused as a whole or keeps the kernel's writes` (slow), on the interpreter; running compiled kernels is the executor's (L5) |

## Heuristic

The suite is `Tolk_next.Heuristic` (`codegen/opt/heuristic/`), written `H`
below. `cases.golden` holds 173 cases: 37 real kernels on the CPU, Metal,
CUDA sm_89 and AMD gfx1100 renderers, the matrix multiplications under `TC=0`,
`TC=2`, `TC_MIN_GLOBALS`, `TC_OPT` and `TC_SELECT`, and the matrix-vector
layout under `MV` and `MV_*`, with the optimisations that
`hand_coded_optimizations` chose (`H › the optimisations chosen are
tinygrad's › applied_opts`). The graph golden named after a case is what
`apply_opts` returns (`› the optimised kernel`). The writes law is
Postrange's.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `codegen/opt/heuristic.py` (no test targets it) | tensor cores, then matrix-vector, grouping, masked upcasts, upcasts, unrolls, the default upcast and locals | `H › the optimisations chosen are tinygrad's` (every row); the branches by kernel: tensor cores `matmul_half_*`, `batched_matmul_half_*`, `conv_half_*_tc_opt_1`, `matmul_half_ragged_*_tc_opt_2`; matrix-vector `vecmat_*`, and no match for an operand that is no load (`vecmat_of_exp_*`) or a global axis that 16 does not divide (`vecmat_1000_*`); grouping `sum_*`, `sum_rows_*`, `max_rows_*`, `cumsum_*`, and the first reduce axis refused, the next taken, `sum_two_axes_*`; masked upcasts `stack_*`, `pad_*`, at their bounds of 7 (`stack_7_*`, `stack_8_*`) and 49 (`pad_7x7_*`, `pad_7x8_*`); upcasts `matmul_*`, `conv_*`; unrolls `sum_17_*`, `sum_100_*`, `sum_3x3_*`, `conv_*`, and no second one after 3 when the next is 5 (`sum_5_by_3_*`); locals `add_broadcast_*`, `transpose_*`, `outer_add_*` (an axis made wholly local shifts the next) |
| tinygrad: `USE_TC`, `TC_OPT`, `TC_SELECT`, `TC_MIN_GLOBALS`, `ALLOW_TF32` | | `matmul_half_<renderer>_no_tc`, `_tc_shape`, `_tc_min_globals` (N upcast skipped), `_tc_min_globals_1` (taken), `_tc_select_2`, `conv_half_<renderer>_tc_opt_1`, `matmul_half_ragged_<renderer>_tc_opt_2`, `matmul_cuda_tf32` |
| tinygrad: `hand_coded_optimizations` makes a copy | the scheduler it is given is left as it was | `H › hand_coded_optimizations leaves the scheduler it is given as it was` (every kernel) |
| tinygrad: `MV`, `MV_BLOCKSIZE`, `MV_THREADS_PER_ROW`, `MV_ROWS_PER_THREAD` | the layout off, sizes of 1 skipped, all sizes 1 | their defaults, `vecmat_*`; read once from the environment, the others run in processes of their own (the suite's dune): `vecmat_<renderer>_mv_0`, `_mv_block_rows_1`, `_mv_sizes_1` |
| tinygrad: the `IMAGE`, `QCOM` and `DSP` branches | | dropped: images, QCOM and DSP are excluded (README) |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_partial_sum_in_accumulator` | the heuristic's tiling after a tensor core | `matmul_half_*` (the tiling); where the partial sums accumulate is the expander's (L4, Codegen) |
| old: `unit/codegen/test_heuristic.ml` "applies GROUPTOP when upcastable prod small", "early return after grouping", "skips grouping when upcastable prod large" | | `sum_rows_*`, `sum_rows_wide_*` |
| old: `unit/codegen/test_heuristic.ml` "full unrolls small reduce", "split unrolls large reduce by 4", "double unrolls tiny reduces" | | `sum_17_*`, `sum_100_*`, `conv_*` (two unrolls of 3) |
| old: `unit/codegen/test_heuristic.ml` "applies 4x upcast when nothing upcasted" | | `add_*`, `add_small_*` |
| old: `unit/codegen/test_heuristic.ml` "broadcast upcast prefers lower stride axis", "upcast size bounded by 32" | | `add_broadcast_*`, `matmul_cpu`, `conv_*` |
| old: `unit/codegen/test_heuristic.ml` "detects matvec and applies GROUP LOCAL UPCAST", "matvec early return prevents further opts", "rejects LOAD(INDEX) matvec shape", "matvec skipped on CPU" | | `vecmat_metal`, `_cuda`, `_amd`; `matvec_*` (no match: grouped); `vecmat_cpu` |
| old: `unit/codegen/test_heuristic.ml` "upcasts small WHERE-guarded dim" | | `stack_*`, `pad_*` |
| old: `unit/codegen/test_heuristic.ml` "applies locals on GPU", "at most 3 locals", "local budget respected", "expand axis gets larger LOCAL from budget", "deleted_shape adjusts axis indices" | | `add_*`, `conv_*`, `add_broadcast_*`, `transpose_*`, `outer_add_*` on Metal, CUDA and AMD |
| old: `unit/codegen/test_heuristic.ml` "elementwise on GPU", "reduce on GPU with grouping", "reduce on GPU without grouping", "matmul on GPU", "elementwise on CPU", "large kernel on CPU" | | `add_*`, `sum_*`, `sum_cols_*`, `matmul_*`, `add_large_*` |
| old: `unit/codegen/test_heuristic.ml` "tensor-core upcasts preserve requested global occupancy" | | `matmul_half_*_tc_min_globals` |
| old: `unit/codegen/test_heuristic.ml` the `IMAGE` group, "IMAGE default occupancy floor skips small global grid", "IMAGE occupancy accounts for cumulative global upcasts", "QCOM uses a smaller grouping threshold" | | dropped: images and QCOM are excluded (README) |

## Search

The suite is `Tolk_next.Search` (`codegen/opt/search/`), written `S` below,
with `Tolk_next.Search on the host` (`test_search_exec.ml`, slow), written
`SH`. `actions.golden` is the table of actions and `actions_padto.golden` the
table under `BEAM_PADTO=1 TC=2 TC_OPT=0`. `candidates.golden` holds the
positions `get_kernel_actions` returns for 14 kernels on the Clang, Metal,
CUDA sm_89 and HIP gfx1100 renderers (`S › candidates.golden`, 52 rows).
`searches.golden` holds what `beam_search` chooses, and how many times it
measures, for 14 searches under a measurement that computes each program's time
from its optimisations and its launch, compiled with no compiler (`S › a search
chooses what tinygrad's chooses`; the GPU rows are slow). The environment
variables the search reads once run in a process of their own (`S ›
BEAM_PADTO=1 TC=2 ...`, the suite's dune).

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/test_search.py::TestSearch::test_beam_symbolic_kernel` | a beam search of a symbolic kernel on the CPU applies optimisations | `SH › a searched kernel computes what its unoptimised kernel computes › symbolic` (through `Codegen.to_program ~beam`, timed by `Tolk_next_engine.measure`); the same kernel's candidates, `candidates.golden` `symbolic` |
| tinygrad: `codegen/opt/search.py` `actions` | the table, in order; `BEAM_PADTO`, `TC`, `TC_OPT` | `S › the actions are tinygrad's, in order`; `S › BEAM_PADTO=1 ... › the actions are tinygrad's, pads included` |
| tinygrad: `codegen/opt/search.py` `get_kernel_actions` | actions out of range, a whole-axis split by its size, `BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`, a tensor core's lanes, `max_up`, `include_0` | `candidates.golden` (every row, `max_up` rows `add clang 4`, `matmul_half metal 16`); `S › get_kernel_actions` (4 laws) |
| tinygrad: `codegen/opt/search.py` `beam_search` | rounds, the beam of width `amt`, `BEAM_MIN_PROGRESS`, the fastest kept at the end, binaries timed once, measurements that fail | `searches.golden` (every row); `S › a search chooses the fastest program it measured` (law, slow); `S › rounds` (3 tests); `S › a failing measurement` |
| tinygrad: `codegen/opt/search.py` `_time_program`, `get_test_global_size` | three measurements, stopping early above three times the best; launches of more than 65536 workgroups halved from the last size above 16, times scaled | `S › rounds › a candidate is measured three times ...`; `S › measuring on fewer workgroups` (5 tests; `add_3d` pins the last size above 16, `add_large` a launch of exactly 65536); the `measurements` column of `searches.golden` |
| tinygrad: `codegen/opt/search.py` `_try_compile` | the kernel named `test`, storage on the renderer's device, `BEAM_UOPS_MAX`, failures dropped, `BEAM_STRICT_MODE`, `DEBUG>=4` | `S › a candidate's storage is placed on its renderer's device`; `S › a candidate that does not compile is dropped` (2 tests); `S › BEAM_PADTO=1 ... › a candidate of BEAM_UOPS_MAX instructions or more is dropped`, `› a compilation that raises is raised under BEAM_STRICT_MODE`; `S › a compilation's failure is printed under DEBUG=4` |
| tinygrad: `codegen/opt/search.py` `BEAM_TIMEOUT_SEC`, `BEAM_DEV_TIMEOUT` | a compilation interrupted by an alarm, a run's time limit | dropped: excluded (README): a domain cannot be interrupted, and a run's limit is the measurement's |
| tinygrad: `codegen/opt/search.py` `diskcache_get`, `diskcache_put`, `IGNORE_BEAM_CACHE`, `CACHELEVEL` | a kept search measures nothing and reapplies what it kept | `S › a search kept in the cache measures nothing, unless it is ignored`; `S › the cache keeps tensor cores and swaps` (slow) |
| tinygrad: `codegen/opt/search.py` `DEBUG>=2`, `BEAM_DEBUG`, `BEAM_LOG_SURPASS_MAX` | progress, the kernel, failures and the choice; kernels dropped for their lanes and instructions | `S › a search prints its progress under DEBUG=2`; `S › a search prints nothing by default`; `S › BEAM_PADTO=1 ... › a search prints the kernel, failures and its choice under BEAM_DEBUG`, `› kernels of too many lanes are reported ...`, the uops test (its output); the kernel prints as a graph, since `pyrender` is excluded (README) |
| tinygrad: `codegen/opt/search.py` `get_worker_pool`, `imap_unordered` | candidates compiled in parallel | `S › a search measures and chooses alike on one domain and on several`: the order is the candidates', tinygrad's order without a pool (D5) |
| tinygrad: `codegen/opt/search.py` the compute filter (1000 times the fewest operations) | a candidate whose estimated operations exceed a thousand times the fewest of its round so far is not timed | `S › BEAM_PADTO=1 ... › searches_environment.golden` (`transpose_33` on Metal: the copies have no arithmetic, and each pad adds a selection per element), `› a candidate of more than 1000 times the fewest operations is not measured` |
| old: `unit/test_opt_correctness.ml` "beam actions preserve semantics (CPU)", `unit/test_opt_correctness_metal.ml` "... (Metal)", `unit/opt_fuzz/tolk_opt_fuzz.ml` (every sequence of two actions against the unoptimised kernel) | every kernel the search can choose computes what the kernel computes | `S › every kernel a search can choose writes what its kernel writes` (law over the four renderers, up to three actions, interpreted); `S › every kernel two actions make writes what its kernel writes` (slow, exhaustive, Clang and Metal); `SH` (compiled and run on the host). The interpreter evaluates no tensor core product: tensor cores' values are Postrange's and Codegen's (`tc_*` goldens) |
| old: `unit/test_runtime_search.ml` "selected kernel compilation produces correct output", "completes on 1D elementwise kernel", "completes on 2D elementwise kernel", "optimized kernel produces correct output", "completes on variable-sized kernel" | | `SH` (`add_small`, `sum_rows`, `variable_rows`, `pad_7x7`, `symbolic`) |
| old: `unit/test_runtime_search.ml` "accepts compact raw buffers for sparse parameter slots", "uses explicit max shape for beam buffers", "beam_search does not corrupt input buffers", "search timing on CPU" | buffers and timings on a device | dropped: the search takes no buffers; allocating and timing are `Tolk_next_engine.measure`'s (README, `args_from_ast`) |
| old: `unit/test_runtime_search.ml` "uses the supplied symbolic value during timing", "codegen rounds negative timing midpoints down without cache eviction" | each variable at the middle of its bounds, rounded down | `S › a search asks for cold runs with each variable at its bounds' middle` (`n` from 1 to 16 is 8) |
| old: `unit/test_runtime_search.ml` "disable_cache bypasses cache" | | `S › a search kept in the cache measures nothing, unless it is ignored` |
| old: `unit/test_runtime_search.ml` "transient program lifetimes" (3 tests), "codegen retires retained timing buffers ..." (2 tests), "beam invokes the available cache hook for each timing sample" | programs and buffers released, caches cleared | dropped: loading and releasing are the engine's; every measurement is asked cold (`S › a search asks for cold runs ...`) |
| old: `unit/test_runtime_search.ml` "beam codegen requires an explicit runtime" | | `CG › beam search (D4)`: a kernel that asks for a search without one is refused |
| old: `unit/test_runtime_search.ml` completed_compile_budget (2 tests) | | dropped: the compile timeout is excluded (README) |
| old: `unit/test_runtime_search.ml` "parallel compilation joins workers before propagating failure", "... interruption", "sequential compilation propagates interruption" | | `Worker`'s section: `Worker.map` joins its domains and raises the first element's exception |
| old: `unit/test_runtime_search.ml` "beam reconsiders compute-filtered candidates in later rounds" | | not pinned: no recorded search times in a later round a binary it filtered in an earlier one; the port marks a binary timed only past the filter, as tinygrad does |
| old: `unit/test_runtime_search.ml` "beam rejects overflowing resource products" | lanes and threads multiplied past 64 bits | `candidates.golden` `huge_upcasts`, `huge_locals` (products in integers of any size) |
| old: `unit/test_runtime_search.ml` "beam retains candidate PROGRAM metadata and scales only its launch", "beam scales symbolic launch products beyond host integer bounds" | | `S › measuring on fewer workgroups`; launch products are taken in integers of any size |

## Codegen

The suite (`test/codegen/codegen`, `CG` below) compiles the kernels of
`kernels.golden` for Clang, Metal, CUDA and HIP, each as a row of
`cases.golden` named `<case>_<target>`, and compares each stage with tinygrad's
(patched as D24 says): `stages › full_rewrite_to_sink lowers each kernel as tinygrad does`, `stages ›
to_program makes each program as tinygrad does` (the lowered sink with its
estimates, the instructions, the program information), `stages › each program
is rendered as tinygrad renders it` and `stages › an optimisation that does not
apply raises tinygrad's error`. A row `CG › stages › <case>` below stands for
those tests on every target of the case. The default run compiles each case for
one target, in turn; the slow run compiles every row.

| Source | Behaviour | Outcome |
|---|---|---|
| old: `golden/codegen` `elementwise_add`, `sum_reduce`, `max_reduce`, `dot_product`, `reduce_rows`, `gated_store`, `elementwise_where`, `elementwise_cast_f16`, `elementwise_sqrt`, `parallel_reduce`, `elementwise_int32`, `lorenz_fold`, `no_optimize` (`clang_`, `cuda_`, `metal_`, `amd_`: 52 goldens) | the source each target writes for hand-built kernels | `CG › stages › <case>`, the kernel rebuilt at tinygrad HEAD with the scheduler's `LOOP` ranges where the old one had `GLOBAL` ranges (and none on the CPU); `no_optimize` is tagged, so `to_program` does not optimise it; `multi_output` is `two_outputs` |
| old: `golden/codegen` `matmul_small`, `elementwise_2d` (`cuda_`, `metal_`, `amd_`: 6 goldens) | the same, on GPUs only | `CG › stages › matmul_small`, `elementwise_2d` (Metal, CUDA, HIP) |
| old: `golden/codegen` `llama_embedding`, `llama_rmsnorm`, `llama_ffn_gate`, `llama_vector_scale`, `llama_output_projection` (20 goldens) | the kernels of tinygrad's LLaMA | `CG › stages › llama_*`, found by the name tinygrad HEAD gives each once optimised for Clang |
| old: `golden/codegen` `opencl_*` (26 goldens) | OpenCL sources | dropped: OpenCL is excluded (plan §3) |
| old: `golden/codegen` `*.expected.diff` (float grouping, a leading `+0` of a reduction, `MAX` of NaN and ties) | the old tolk's IEEE differences from tinygrad's source | dropped: tolk.next writes tinygrad's source; its differences are D16 and D17, compared in `CG › stages › each program is rendered as tinygrad renders it` |
| old: `unit/codegen/test_lower.ml` "bounded shared loop finishes reads before the next write" | a loop that reads the local memory it writes ends with a barrier | `CG › tinygrad's claims on lowered kernels › shared_loop_barrier: ...`; `CG › stages › shared_loop_barrier` |
| old: `unit/codegen/test_lower.ml` "conditional shared loop finishes reads before the backedge" | the same for an unbounded loop, which keeps its condition | `CG › tinygrad's claims on lowered kernels › shared_backedge_barrier: ...` |
| old: `unit/codegen/test_lower.ml` "conditional shared loop keeps independent buffers separate" | no barrier when another buffer is read | `CG › tinygrad's claims on lowered kernels › shared_backedge_two_buffers: ...` |
| old: `unit/codegen/test_lower.ml` "conditional shared loop retains its existing barrier" | a barrier already there is not wrapped again | `CG › tinygrad's claims on lowered kernels › shared_backedge_barrier_kept: ...` |
| old: `unit/codegen/test_lower.ml` "accumulators and staged locals share slots after explicit storage" | new buffers take the slots after the kernel's | `CG › tinygrad's claims on lowered kernels › explicit_local_slots: ...` (slots 17, 18, 19) |
| old: `unit/codegen/test_lower.ml` "anonymous local storage survives lowering until linearization" | an `ALLOC` stays in the lowered sink and is a `BUFFER` of its slot in the program | `CG › tinygrad's claims on lowered kernels › anonymous_local: ...` |
| old: `unit/codegen/test_lower.ml` "WMMA contracts complete split-axis identities" | the expander contracts a tensor core's upcast axes | `CG › stages › matmul_tc`, `matmul_half`, `matmul_tc_unroll` (the lowered tensor cores); a hand-built `WMMA` has no tinygrad counterpart |
| old: `unit/codegen/test_lower.ml` "final rewrite concretizes leftover index dtypes" | no weak type is left but a constant's | `CG › tinygrad's claims on lowered kernels › weak_index_store: leaves a weak type only to constants`; the program specification is checked by every lowering (`SPEC=1`, the default) |
| old: `unit/codegen/test_lower.ml` "memory operands feeding ALU become explicit loads" | arithmetic reads loads, a store's destination is no load | `CG › tinygrad's claims on lowered kernels › negated_index: computes on loads, never on addresses` |
| old: `unit/codegen/test_lower.ml` "PARAM slot -1 is numbered from existing param count", "BUFFER variables become numbered scalar formals" | variables are numbered after the buffers | `CG › tinygrad's claims on lowered kernels › numbered_variables: numbers variables after the buffers, in order` |
| old: `unit/codegen/test_lower.ml` "lowering flattens sink-like children" | a sink of a sink, a stack and a no-op | dropped: tinygrad HEAD's kernels are sinks of stores; a sink's cleanup is `Symbolic`'s `pm_clean_up_group_sink` |
| old: `unit/codegen/test_lower.ml` "SPEC verifies final program spec" | a lowered graph that breaks the program specification raises | `CG › errors › a lowered graph that breaks the specification raises Invalid_argument` |
| old: `unit/codegen/test_lower.ml` "invalid index gate moves onto store", the gater tests | | Gater's section; the pipeline: `CG › gated stores`, `CG › stages › gated_store`, `gated_loop` |
| old: `unit/codegen/test_linearizer.ml` "gated stores become IF/STORE/ENDIF", "single casted gated stores become IF/STORE/ENDIF" | | `CG › pm_linearize_cleanups › a gated store becomes a store inside an if on its gate`, `so does a gated store through a cast of its index` |
| old: `unit/codegen/test_linearizer.ml` "bitcasted gated stores are not linearize-cleanup matches", "nested-cast gated stores are not linearize-cleanup matches" | | `CG › pm_linearize_cleanups › leaves alone a store it does not match › through a bitcast of its index`, `through a cast of a cast of its index` |
| old: `unit/codegen/test_linearizer.ml` "graph IF nodes are rejected", "graph ENDIF nodes are rejected" | | `CG › pm_linearize_cleanups › raises Invalid_argument on an if already in the list › if`, `endif` |
| old: `unit/codegen/test_decompositions.ml` "comparison extrema simplify before late codegen" | comparisons of a long with its type's bounds fold | `CG › tinygrad's claims on lowered kernels › comparison_extrema: folds comparisons with the bounds of a long` |
| old: `unit/test_program_spec.ml` "reads and writes are deduplicated", "buffer tracing passes through cast and after", "buffer args are treated as globals", "local and register storage are not runtime globals", "program_info mirrors extracted metadata" | a program's buffers, outputs and inputs | `CG › stages › to_program makes each program as tinygrad does` (each program's `ProgramInfo`); `CG › programs › the argument of a program is its program information for the target`; `program_info_of_sink` itself is Ops' (`O › programs`) |
| old: `unit/test_program_spec.ml` "thread-group launch expressions are preserved", "global idx uses flat thread launch", "program_info preserves symbolic local dimensions" | launch sizes | `CG › stages › *_metal`, `*_cuda`, `*_hip` (`global_size`, `local_size`), `CG › stages › symbolic`, `symbolic_sum` |
| old: `unit/test_program_spec.ml` "launch variables are resolved by name", "missing launch variables identify the program", "launch dimensions retain exact intermediate products", "launch dimensions preserve Python signed shifts", "launch casts convert values without storage narrowing", "launch bitcasts retain the source representation", "launch floor div and mod use Python semantics" | evaluating launch sizes | dropped here: `Ops.launch_dims`, `Ops.vals` and `sym_infer` (`O › programs`, `O › sym_infer`) |
| old: `unit/test_program_spec.ml` "core_id is an ordinary scalar", "duplicate launch axis is rejected", "mixed launch models are rejected" | the old program record's checks | dropped: `ProgramInfo` has no core index and no launch model; tinygrad checks neither |
| old: `parity/weak_movement_width` | a weak product beyond int32, reshaped and permuted, compiles for CPU and CUDA | `CG › stages › weak_product_permuted` |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_load_dedup` | an upcast loads each element once | `CG › tinygrad's claims on programs › load_dedup_upcast: loads each element at most once` |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_upcast_cse` | | dropped: skipped in tinygrad ("handled at higher level now") |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_reduce_upcast` | an upcast and unrolled convolution keeps no accumulator | `CG › tinygrad's claims on programs › reduce_upcast_unroll: keeps no accumulator and stores once` |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_zero_fold` | stacking two values computes nothing | `CG › tinygrad's claims on programs › zero_fold_upcast: stacks two values without arithmetic` |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_sum_acc_dtype` | the accumulator of a sum of bools, shorts, halves and bfloat16s | `CG › tinygrad's claims on programs › sum_acc_bool`, `sum_acc_short`, `sum_acc_half`, `sum_acc_bf16` |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_arg_acc_dtype` | the accumulator a sum, a matmul, an einsum and a convolution ask for | `CG › tinygrad's claims on programs › matmul_acc_half: accumulates in the half it asks for`; the sums of halves and bfloat16s above; the einsum and the convolution reduce as the matmul does (`Tensor` surface, the frontend is nx) |
| tinygrad: `null/test_linearizer.py::TestLinearizer::test_sum_collapse` | a sum of a broadcast collapses | dropped here: the scheduled kernel is Rangeify's section |
| tinygrad: `null/test_linearizer.py::TestLinearizerRenderers::test_upcast_with_locals_cpu` | | dropped: an expected failure in tinygrad |
| tinygrad: `null/test_linearizer.py::TestLinearizerRenderers::test_upcast_with_locals` | a local store of four lanes, then a global store of one float | `CG › tinygrad's claims on programs › upcast_with_locals_opts: ...` on Metal, CUDA and HIP (tinygrad's is AMD's LLVM renderer, which is excluded) |
| tinygrad: `null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_reduction`, `test_arange` | an optimised program compiles | `CG › stages › rewrite_reduction_opts`, `arange_upcast` |
| tinygrad: `null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_kernel_info` | an empty list applies no optimisation; a kernel keeps its name | `CG › tinygrad's claims on programs › a kernel that lists no optimisation is compiled without any`, `a kernel named by its argument keeps its name` |
| tinygrad: `null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_dependent_loop_bound` | a loop bounded by a load closes after its stores | `CG › tinygrad's claims on programs › dependent_loop_bound: closes each loop after its stores, inner first` |
| tinygrad: `null/test_linearizer_failures.py::TestLinearizerFailures::test_fail_1` | a kernel that once failed compiles | `CG › stages › linearizer_fail_1` |
| tinygrad: `runtime/test_linearizer_dumb.py::TestLinearizerFailure::test_failure_beam_mnist` | | `CG › stages › failure_beam_mnist` (Metal, CUDA, HIP) |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_arg_dedup`, `test_load_removed`, `test_assign_fold` | realized buffers and values | dropped: realizing and running (the engine, L7) |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_cast_there_and_back`, `test_cast_back_and_there`, `test_grouped_store_phis` | | dropped: skipped or an expected failure in tinygrad |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_two_nested_range`, `test_three_nested_range`, `test_range_outer_op_before_phi_nested_range` | a broadcast sum collapses to one loop | `CG › tinygrad's claims on programs › two_nested_range`, `three_nested_range`, `range_outer_op_before_phi_nested_range` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_default_global_reversed` | the last axis launches first | `CG › tinygrad's claims on programs › default_global_reversed: launches the last axis first` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_where_fold` | the select of an assignment folds away | `CG › tinygrad's claims on programs › where_fold: ...`; its values: `CG › values › on the host, a program writes what its kernel writes › where_fold` (slow) |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_phi_simplification` | an arange keeps no accumulator | `CG › tinygrad's claims on programs › phi_arange_float`, `phi_arange_negative`, `phi_arange_255` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_two_grouped_stores_local` | two grouped reductions, a barrier each | `CG › tinygrad's claims on programs › two_grouped_stores_local: puts a barrier after each local store` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_late_bias_load`, `test_two_nested_range_alt_indexing`, `test_range_outer_op_before_phi`, `test_simple_unroll_no_between_phi_dependencies`, `test_grouped_store_values`, `test_grouped_store_locals_and_globals`, `test_grouped_store_local_only` | | Linearizer's and Coalesce's sections |
| tinygrad: `runtime/test_kernel_opts.py`, `runtime/test_tensor_cores.py` | optimisations and tensor cores | Postrange's section; the lowered programs: `CG › stages › matmul_tc`, `matmul_tc_unroll` (Metal), `matmul_half` (the heuristic's tensor cores), `matmul_half_tc_shaped` (`TC=2`), `matmul_half_no_tc` (`TC=0`), `matmul_local`, `sum_group`, `max_group` |
| tinygrad: `runtime/test_tensor_cores.py::test_tensor_cores_unroll_phi`, `test_tensor_cores_unroll_casted_phi`, `test_tensor_cores_unroll_casted_phi_with_children`, `test_tensor_cores_partial_sum_in_accumulator` | where the accumulator of an unrolled tensor core lives | `CG › stages › matmul_tc_unroll` (Metal), `matmul_half` |
| tinygrad: `null/test_gen_float4.py` | | Coalesce's section |
| tinygrad: `null/test_renderer_failures.py` | | Cstyle's section |
| tinygrad: `runtime/test_custom_kernel.py::TestCustomKernel::test_gemm`, `test_arange`, `test_eye`, `test_sum`, `test_flip_contract`, `test_slice_sum`, `test_simple_qkv`, `test_simple_sharded` | custom kernels compile | `CG › stages › custom_gemm`, `custom_arange`, `custom_eye`, `custom_sum`, `flip_contract`, `slice_sum`, `simple_qkv`, `sharded_custom_add` |
| tinygrad: `runtime/test_custom_kernel.py::TestCustomKernel::test_group_reduce_split_range`, `test_nested_group_reduce`, `test_local_reduce`, `test_stage_then_reduce`, `test_reg_stage_then_reduce`, `test_reg_placeholder_then_reduce`, `test_split_range_id_free_of_loop` | the expander, local buffers and control flow of kernels that ask for no optimisation | `CG › stages › group_reduce_split_range`, `nested_group_reduce`, `local_reduce`, `warp_reduce`, `stage_then_reduce`, `reg_stage_then_reduce`, `reg_placeholder_then_reduce`, `split_range_id_free_of_loop` |
| tinygrad: `runtime/test_custom_kernel.py` (the other tests) | calling, sharding and differentiating custom kernels | dropped: `Tensor.custom_kernel` and execution (Schedule, the engine); the frontend is nx |
| tinygrad: `null/test_uop_graph.py::TestUOpGraph::test_devectorize_derives_lane_dtype`, `test_devectorize_zero_sized_scalar_expand` | `do_devectorize` and `devectorizer2` on hand-built nodes | dropped: the devectorizer is internal to `Codegen`; every lowering runs it (`CG › stages`) |
| tinygrad: `null/test_uop_graph.py::TestReduceCollapse::test_reduce_shapeless_const_unroll` | no reduction survives lowering; `3 × 4` is `12` | `CG › tinygrad's claims on programs › reduce_shapeless_const_unroll: folds the sum of a constant over an unroll` |
| tinygrad: `null/test_uop_graph.py::TestReduceCollapse::test_multi_range_reduce_add` | | Simplify's section (`pm_reduce_collapse`) |
| tinygrad: `null/test_graph_rewrite.py::TestEdgeCasesAndSpecialOperations::test_full_graph_rewrite_transcendental_edge_cases` | | `CG › full_rewrite_to_sink folds whole graphs › log2 of -1 is NaN and the reciprocal of 0 is infinity` |
| tinygrad: `null/test_graph_rewrite.py::TestGEPAndVectorizeRewrite` (3 tests) | lanes of a vector of constants | `CG › full_rewrite_to_sink folds whole graphs › a lane of a vector is the lane's value`, `a vector of lanes is the vector of their values`, `a vector of every lane of a vector is the vector` |
| tinygrad: `null/test_graph_rewrite.py::TestModuloAndDivisionFolding::test_graph_rewrite_div_folding_bug` | | `CG › full_rewrite_to_sink folds whole graphs › a comparison of lanes keeps the lanes apart` |
| tinygrad: `null/test_const_folding.py::TestBitcastConstFolding::test_vec_bitcast`; old: `unit/uop/test_symbolic.ml` "STACK const bitcast folds", "cast STACK const folds lane-wise", "bitcast STACK const folds lane-wise" | | `CG › full_rewrite_to_sink folds whole graphs › a bitcast of constants folds lane by lane`, `a cast of constants folds lane by lane` |
| tinygrad: `null/test_uops.py::TestGatedStoreRewrite::test_tiny_gate_store`, `test_gate_some_stores` | a gated store is the one store inside an if | `CG › gated stores › a store gated by its index is the one store inside an if`, `an ungated store beside it stays out of the if` |
| tinygrad: `null/test_uops.py::TestGatedStoreRewrite::test_merge_ifs_alt` | | dropped: skipped in tinygrad ("we don't merge ifs anymore") |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_division_power_of_two`, `test_floormod_power_of_two`, `test_max_keeps_bound_for_idiv`, `test_floordiv_power_of_two`, `test_unsigned_floordiv_is_cdiv`, `test_fast_idiv_and_mod`, `test_fast_idiv_nonpositive_divisor`, `test_fast_idiv_cmod_kept_when_idiv_declines`, `test_fast_idiv_bounded_numerator_zero`, `test_fast_idiv_remove_powers_of_two`, `test_disable_fast_idiv` | | `CG › divisions` (one test each, in order); `test_max_keeps_bound_for_idiv`'s renderer is Clang's operations without `MAX`, as `CStyleLanguage`'s lack it, and its dividend `max x 0 + 1` wraps at the greatest int (D24), so `a maximum proves its dividend positive before it is decomposed` divides `max x 0`, and `a dividend that can wrap keeps the correction of a floor division (D24)` pins the difference |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_overflow` | | dropped: an expected failure in tinygrad |
| tinygrad: `null/test_uops.py::TestMemoryCoalescing`, `TestLowerIndexDtype` | | Coalesce's and Uop_weak's sections |
| tinygrad: `null/test_simplify_valid_idx.py::TestRangeShrink` (8 tests) | the ranges left after lowering | `CG › range shrinking` (one test each); the pass alone is Simplify's section |
| tinygrad: `null/test_simplify_valid_idx.py::TestImageSimplification::test_drop_gate_committed_in_the_index_pass` | | dropped: images are excluded |
| tinygrad: `null/test_dtype_weak.py::TestNoRedundantWide` (2 tests) | 64 bits only where the bounds need them | `CG › tinygrad's claims on programs › long: computes a long in 64 bits`, `fancy_index: indexes without 64-bit arithmetic` |
| tinygrad: `null/test_randomness.py::TestRandomness::test_threefry_doesnt_use_long` | | `CG › tinygrad's claims on programs › threefry: computes random bits without 64-bit values` (Clang) |
| tinygrad: `null/test_arange.py::TestArange::test_cat_complexity`, `test_tri_complexity` | an arange and a mask cost few operations | `CG › tinygrad's claims on programs › cat: concatenates in 20 operations an element`, `triu_noopt: masks in 4 operations an element` |
| tinygrad: `codegen/__init__.py` `pm_wmma_add` (D24) | the running sum replaces a zero tensor-core accumulator; a sum keeps its value, apart from a zero's sign | `CG › tensor-core accumulators (D24)` (every test); the lowered tensor cores: `CG › stages › matmul_tc`, `matmul_half`, `attention` |
| tinygrad: `codegen/__init__.py` `beam` (D4) | a kernel asking for a beam search of width `w` is searched with `beam w` | `CG › beam search (D4)` (every test) |
| tinygrad: `codegen/__init__.py` `to_program_cache`, `engine/worker.py` (D5) | programs are kept per kernel, renderer and settings; workers share them | `CG › programs are kept` (every test) |
| tinygrad: `codegen/__init__.py` `do_to_program`, `pm_to_program` | a program resumes from a sink or a sink and its instructions | `CG › programs › to_program resumes a program from its lowered sink` |
| tinygrad: `codegen/__init__.py` `do_estimates` | estimates leave index arithmetic out | `CG › programs › the estimates count the instructions, leaving out index arithmetic` |
| tinygrad: `codegen/__init__.py` `do_linearize`, `do_render`, `do_compile` (`DEBUG`) | diagnostics | `CG › diagnostics` (every test) |
| tinygrad: `codegen/__init__.py` `line_rewrite` | | `CG › line_rewrite` (every test) |
| tinygrad: `codegen/__init__.py` (emulated narrow floats, D9) | an emulated float8 program computes IEEE conversions | `CG › stages › an emulated float8 program launches as tinygrad's (D9)`; `CG › values › on the host, a program writes what its kernel writes › fp8` |

## Cstyle

The suite is `Tolk_next.Cstyle` (`renderer/cstyle/`), written `CS` below.
`kernels.golden` holds 255 linearized kernels, each an `Ops.LINEAR` named after
its case: real `Tensor` programs linearized as `to_program` linearizes them for
each target (Clang `x86_64,x86-64`, Metal `Apple9`, CUDA `sm_89`, HIP
`gfx1100`, and `gfx1201`, `gfx942` and `gfx950` for their float8, bfloat16 and
tensor cores), and hand-built kernels rendered as they are. `cases.golden`
gives each case's target and setting, and `<case>.golden` is its source, byte
for byte. `EXPAND_SSA` and `ALIGNED` are read once per process, so the cases
rendered under them run in processes of their own. `rewrite_inputs.golden`,
`rewrites.golden` and `rewritten.golden` hold each target's `extra_matcher`
applied to 25 graphs; `declarations.golden` what each renderer declares for 24
targets; `written.golden` how each writes its native operations. D16's CUDA
sources are compared once D16's guard is written back as tinygrad writes it
(`CS › sources`, `tinygrad_of_d16`). A kernel with an operation D17 narrows has a second kernel
recorded, the same with a cast after each such operation, and its golden is
tinygrad's source for that kernel (`CS › narrowing (D17)`). Execution runs on the host through the
engine (`support/run.ml`), which compiles each kernel with nx.device's entry
(D28).

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `renderer/cstyle.py` `CStyleLanguage`, `ClangRenderer`, `MetalRenderer`, `CUDARenderer`, `HIPRenderer` (no test of the source) | the source of each kind of kernel: every data type each language has, vectors, tensor cores, gated loads and stores, hardware indices, barriers and shared memory, the transcendental decompositions, infinities and NaNs, constants, casts and bitcasts, variables, custom code, calls, constant tables, loops without a trip count, nontemporal loads, volatile parameters | `CS › sources › by default` (250 goldens) |
| tinygrad: `renderer/cstyle.py` `getenv("EXPAND_SSA")`, `getenv("ALIGNED", 1)` | every value a local variable; Clang's vectors aligned to one byte | `CS › sources › with EXPAND_SSA=1` (`clang_add`, `clang_matmul_upcasted`, `cuda_matmul_locals`), `with ALIGNED=0` (`clang_add`, `clang_dtype_bool`) |
| tinygrad: `renderer/cstyle.py` each `extra_matcher` | bfloat16 and float8 arithmetic through float, manual bfloat16 conversions (not on gfx950), double to half through float on Clang, Metal's bfloat16 transcendentals, CUDA's float8 to float8 conversions, HIP's float8 tensor-core operands as ulongs | `CS › extra_matcher › rewrites.golden` (25 inputs, each for the 7 targets) |
| tinygrad: `renderer/cstyle.py` class attributes and `supported_dtypes` | launch limits, shared memory, tensor cores, native operations, data types, compiler cache key, per architecture | `CS › declarations › <field> › declarations.golden` (24 targets, 11 fields) |
| tinygrad: `renderer/cstyle.py` each `code_for_op` | how each native operation is written, per data type | `CS › code_for_op › written.golden` (405 rows); law `CS › rendering › writes each native operation as code_for_op does` |
| tinygrad: `renderer/cstyle.py:349`, `:410`, `:473`, `:497`, `ClangCompiler.__init__` | the architecture chooses tensor cores and data types, and a malformed one is refused | `CS › declarations`; `CS › architectures` (5 tests) |
| tinygrad: `renderer/cstyle.py:75` `x.arg[0].format(...)` | custom code is a `str.format` string | `CS › rendering › formats custom code as str.format does` (5 cases); `refuses custom code str.format refuses, naming it` (5 cases: an unmatched `{`, a single `}`, an index or a name that names no operand) |
| tinygrad: `renderer/cstyle.py:242` `assert l is not None`, `:463`, `:563` (the `KeyError` and `IndexError` of a node the target cannot write) | an operation the target lacks, a tensor core product of a type without a core, a float8 conversion on a GPU without float8, are refused | `CS › rendering › raises Invalid_argument on an operation the target lacks` (4 renderers); `raises Invalid_argument on a kernel of another target it cannot write` (2 cases) |
| tinygrad: `renderer/cstyle.py:205-258` `_render` | rendering is a function of the kernel alone | `CS › rendering › keeps no state from one kernel to the next`, `keeps no reference to the kernel it rendered`, `writes the same CUDA source for the CUDA and NV devices` |
| tinygrad: `null/test_renderer_failures.py::TestCStyleFailures::test_repeat_add`, `test_repeat_mul`, `test_repeat_xor`, `test_repeat_or`, `test_repeat_and`, `test_repeat_sub` | an associative chain is written without its operands' parentheses, a subtraction with them | `CS › parentheses › an associative chain is written flat › add`, `mul`, `xor`, `or`, `and`, `sub`; `sources › clang_chain_*` |
| tinygrad: `null/test_renderer_failures.py::TestWGSLFailures::test_folded_packed_store` | a packed char access | the WGSL renderer is excluded; its kernel is `CS › sources › *_packed_cast`, `*_packed_bitcast`, and `CS › execution on the host › reads four chars as a uint through a cast of their address`, `... through a bitcast of their address` |
| tinygrad: `runtime/test_renderer_failures.py::TestCStyleFailures::test_inline_const_alu` | `MAX` of a load and the int above the least | `CS › execution on the host › takes the maximum of a load and a literal`; `sources › *_inline_const_alu` |
| tinygrad: `runtime/test_renderer_failures.py::TestRendererFailures` (2 tests), `device/nv/test_renderer_failures.py` (2 tests) | gated stores on PTX and the Python emulator | dropped: PTX and the Python renderer are excluded; the gated stores of C are `CS › sources › *_gated_store_on_threads`, `*_gated_store_in_loop` and `CS › execution on the host › stores only where its gate holds` |
| tinygrad: `null/test_compile_failures.py::TestCompileFailures::test_interpolate_atari`, `test_add_max_uchar` | kernels that once failed to compile | `CS › sources › clang_interpolate_atari_0`, `_1`, `clang_add_max_uchar`; slow: `CS › execution on the host › every kernel compiles and loads` |
| tinygrad: `null/test_compile_failures.py::TestDisassembly::test_float16_alu` | Clang on arm64 adds halves without converting | dropped here: disassembly is `Compiler_cpu`'s; that each `__fp16` operation rounds to a half is `CS › execution on the host › rounds each operation on halves to a half (D17)` |
| tinygrad: `device/cpu/test_cpu.py::TestCPU::test_arch_feats` | the architecture's features reach the compiled code (`vmov` with `avx`) | dropped here: the compiler's flags are `Compiler_cpu`'s; `CS › declarations › cachekey` pins that Clang's cache table is named after the architecture |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_compile_success`, `test_compile_error` | Metal compiles a kernel, and refuses bad source | slow: `CS › every GPU kernel compiles with its target's toolchain › metal_*` (49 kernels; `metal_transcendental_bf16` is xfail: tinygrad's graph truncates a bfloat without the float cast, D18); the refusal is `Compiler_metal`'s |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_alloc_oom`, `test_failed_newLibraryWithData`, `test_free` | Metal memory and pipelines | dropped: the device's, nx.device and `tolk.next.engine` (plan §1a) |
| tinygrad: `renderer/cstyle.py` CUDA and HIP sources through NVRTC and comgr (no test) | the sources compile | slow: `CS › every GPU kernel compiles with its target's toolchain › cuda_*`, `hip_*`, skipped where the library is absent |
| tinygrad: `device/cpu/test_call.py::TestExternalCall::test_call_out_param`, `test_call_ret` | a call of a function pointer, with an out parameter or a result | `CS › sources › clang_call_out`, `clang_call_ret`; their execution needs a host callback, which the executor does not provide |
| tinygrad: `null/test_call.py::TestCallCodegen::test_call_stack_pointer` | a call whose argument is the address of a register | `CS › sources › clang_call_stack` (holds `(unsigned int*)((buf0+0))`) |
| tinygrad: `null/test_call.py::TestCallCodegen::test_compiled_scalar_slots_are_not_call_slots` | the slots of a compiled program's variables | dropped: `ProgramInfo` and `resolve_linear_call`, the engine's (L7) |
| tinygrad: `runtime/test_custom_kernel.py::TestCustomKernel::test_binary` | a constant byte table | `CS › sources › clang_table`; `CS › execution on the host › reads a constant table` |
| tinygrad: `device/cpu/test_custom_kernel.py::TestCustomKernel::test_simple_from_source`, `test_simple_from_source_alt` | a program from hand-written C | dropped: `Tensor.custom_kernel` of a `SOURCE`, the frontend's and the engine's |
| tinygrad: `null/test_randomness.py::TestRandomness::test_threefry_doesnt_use_long` | Clang's random kernels use no 64-bit values | dropped here: the decomposition of `THREEFRY` is `Decomp_op`'s; its source is `CS › sources › clang_rand` |
| tinygrad: `runtime/test_uops.py::TestNonFloatUOps` `test_shr_int32`, `test_shl_int32` (C-style skips), `TestBitcastBufferView`, `null/test_uops.py::TestBitcastBufferView::test_render` | shifts and a bitcast view of a buffer, compiled | `CS › rendering › writes each native operation as code_for_op does` (shifts on every renderer); `CS › sources › *_packed_bitcast`; the value tests run on a device, the engine's (L7) |
| tinygrad: `null/test_uops.py::TestUOpsGraph::test_max_keeps_bound_for_idiv` (`CStyleLanguage(Target())`) | the base language has no `MAX` | `CS › declarations › code_for_op › declarations.golden` (no row lists `MAX`) |
| tinygrad: `null/test_linearizer.py::TestLinearizerRenderers::test_upcast_with_locals_cpu` | Clang renders a kernel with a local split | dropped: an expected failure in tinygrad |
| tinygrad: `runtime/test_arange.py` embedding backward tests, `runtime/test_linearizer.py::TestLinearizer::test_where_fold`, `null/test_tensor.py` index dtype tests, `null/test_device.py::TestDevice::test_compiler_autodetect_fallback` | renderer checks in skip conditions and whole programs | dropped: skip conditions and realizations, the engine's (L7) |
| tinygrad: `renderer/cstyle.py` `ClangRenderer` (no test) | kernels run as their graph says | `CS › execution on the host` (18 tests on known values); slow law `every kernel the interpreter runs writes what it computes` (36 kernels, 10 random inputs each; the others excluded in the suite with their reasons) |
| D17: `renderer/cstyle.py:245-250` C promotion of narrow operands | char, short and `__fp16` operations compute in int or float, and only the stored value narrows; tolk.next casts each inlined one back | `CS › narrowing (D17) › cases.golden` (every kernel's written kernel is `with_d17_casts` of it; 21 sources are tinygrad's for the kernel with those casts); `CS › execution on the host › rounds each operation on halves to a half (D17)`, `wraps each operation on unsigned chars (D17)`; slow law `a kernel over a narrow type wraps and rounds as the interpreter (D17)` (8 kernels, inputs over each type's whole range) |
| D18: `renderer/cstyle.py:368-372` Metal's `extra_matcher` computes `SQRT`, `EXP2`, `LOG2` and `SIN` of a bfloat16 in float32, not `TRUNC` | Metal truncates only floats, so a bfloat16 `TRUNC` is computed in float32 too | `CS › bfloat16 truncation on Metal (D18) › truncates a bfloat16 in float32`, `leaves it to CUDA, which truncates a bfloat16 with htrunc`; slow: `compiles a kernel that truncates a bfloat16` (skipped without MTLCompiler) |
| D16: `renderer/cstyle.py:33,42` a float8 conversion saturates infinities on CUDA | an infinity stays one in e5m2 and becomes a NaN of its sign in e4m3; finite values saturate | `CS › float8 infinities on CUDA (D16)` (6 tests: the guard exactly where a value converts to a float8, the byte of each infinity, infinite constants as their bits); `CS › sources › cuda_dtype_float8_*`, `cuda_inf_nan_float8_*` through `tinygrad_of_d16`; NVRTC: slow, `CS › every GPU kernel compiles with its target's toolchain › cuda_*` |
| old: `unit/test_cstyle.ml` "Volatile parameters" › "buffer qualifiers survive rendering and serialization", "vector access casts retain the volatile qualifier" | `volatile` in the signature | `CS › sources › *_volatile`; `CS › execution on the host › accesses volatile buffers`; serialization is `Graph`'s (`ParamArg.volatile`) |
| old: `unit/test_cstyle.ml` "Constants" (9 tests) | int, float, double, bool, NaN and infinity, `l`, `u`, `ul` suffixes, a ulong of -1 | `CS › sources › *_constants`, `*_inf_nan_*`; `CS › execution on the host › stores each constant as its type holds it` |
| old: `unit/test_cstyle.ml` "ALU Operations" › "Binary", "Unary", "Ternary" › "where", "Backend-specific" (CUDA half intrinsics, Metal precise sin, Clang builtins) | each operation's spelling | `CS › code_for_op › written.golden`; law `CS › rendering › writes each native operation as code_for_op does` |
| old: `unit/test_cstyle.ml` "raw Fdiv is Clang-only", "max", "Ternary" › "mulacc" | an operation the target lacks is refused | `CS › rendering › raises Invalid_argument on an operation the target lacks`; `declarations › code_for_op` |
| old: `unit/test_cstyle.ml` "integer division", "comparison operators", "reciprocal" | | `CS › code_for_op › written.golden` (`CDIV`, `CMPLT`, `CMPNE`, `CMPEQ`, `RECIPROCAL`) |
| old: `unit/test_cstyle.ml` "paren stripping" | | `CS › parentheses` |
| old: `unit/test_cstyle.ml` "Control Flow" › "for loop", "nested loops", "conditional" | | `CS › sources › *_sum`, `clang_matmul`, `clang_interpolate_atari_1`, `*_gated_store_in_loop` |
| old: `unit/test_cstyle.ml` "Memory" › "simple load/store", "gated load", "gated store", "vector load/store casts access pointer", "dtype-changing load casts access pointer", "shrink renders like index" | | `CS › sources › *_add`, `*_padded`, `*_gated_store_*`, `*_matmul_upcasted`, `*_packed_cast`, `*_vector_cast` |
| old: `unit/test_cstyle.ml` "clang vector types are aligned unless asked otherwise" | | `CS › sources › with ALIGNED=0 › clang_add_unaligned`, `clang_dtype_bool_unaligned`; `by default › clang_add` |
| old: `unit/test_cstyle.ml` "opencl image load/store", "non-opencl image rejected", "OpenCL image indexes keep separate coordinates", "OpenCL fp16 pragma", "Intel" › "kernel attribute" | | dropped: OpenCL, Intel and images are excluded |
| old: `unit/test_cstyle.ml` "Cast and Bitcast" (3 tests), "GPU pointer bitcasts use backend bitcast syntax", "CUDA bitcast template" | | `CS › sources › *_bitcast`, `*_packed_bitcast`, `*_dtype_*` |
| old: `unit/test_cstyle.ml` "Special Dimensions" › "Group_id", "Local_id" | | `CS › sources › *_matmul_locals`, `*_group_reduce`, `*_gated_store_on_threads` |
| old: `unit/test_cstyle.ml` "Special Dimensions" › "Global_idx", "Clang fails" | a global index, a hardware index on Clang | dropped: HEAD's renderers have group and local indices only, and code generation gives Clang none (`declarations › has_local`) |
| old: `unit/test_cstyle.ml` "Shared Memory and Barrier" (2 tests) | | `CS › sources › metal_group_reduce`, `cuda_group_reduce`, `hip_group_reduce`, `*_group_reduce_upcasted` |
| old: `unit/test_cstyle.ml` "ordinary final indexes require one flat coordinate" | an index of two coordinates is refused | dropped: HEAD indexes by one coordinate or two (images, excluded); the renderer does not check |
| old: `unit/test_cstyle.ml` "flat indexes preserve a symbolic stride", "vectorize", "index" | | `CS › sources › *_shrunk`, `*_matmul_upcasted`, `*_idiv` |
| old: `unit/test_cstyle.ml` "Clang renders a dynamic value lane index" | | `CS › sources › *_dynamic_lane`; `CS › execution on the host › picks a lane of a vector by a variable` |
| old: `unit/test_cstyle.ml` "vector cast uses the resulting shape", "size-changing bitcast uses the resulting shape" | | `CS › sources › *_vector_cast`, `clang_register_cast`; `CS › execution on the host › reads two uint registers as a ulong through a cast of their address` |
| old: `unit/test_cstyle.ml` "Kernel Signature" › "function prefix", "kernel name", "parameter qualifiers", "scalar parameter", "64-bit scalar parameter" | | `CS › sources` (every golden); `*_scalar_params`; `CS › execution on the host › passes variables of 32 and 64 bits` |
| old: `unit/test_cstyle.ml` "explicit names survive numbered buffer and scalar parameters" | | `CS › sources › *_named_params`; `CS › execution on the host › passes named parameters` |
| old: `unit/test_cstyle.ml` "buffer parameter and body both use type_map" | | `CS › sources › metal_dtype_bf16`, `*_constants` |
| old: `unit/test_cstyle.ml` "renderer op capabilities match cstyle render surface", "a dtype is supported natively or by emulation", "renderer dtype capabilities are backend-specific", "metal tensor cores follow Apple GPU family" | | `CS › declarations › code_for_op`, `supported`, `tensor_cores`; `CS › architectures › Metal has no tensor cores on a family that is not Apple's`; emulation is `Decomp_dtype`'s; QCOM and OpenCL are excluded |
| old: `unit/test_cstyle.ml` "Preamble" › "CUDA fp16 include", "CUDA WMMA helper follows tinygrad asm preamble", "CUDA WMMA helper is emitted once per signature", "CUDA WMMA declares the accumulator width, not the operand width", "Metal stdlib", "Metal WMMA helper follows tinygrad simdgroup preamble" | | `CS › sources › cuda_dtype_half`, `cuda_tc_*` (8 cores), `metal_*`, `metal_tc_*` (5 cores) |
| old: `unit/test_cstyle.ml` "Non-native Rewrites" (6 tests) | emulated floats through float, manual bfloat16 conversions not on CDNA4, fp8 tensor-core operands packed | `CS › extra_matcher › rewrites.golden` (`bf16_*`, `*_to_bf16`, `e4m3_product`, `e5m2fnuz_product`, `half_product` on each target) |
| old: `unit/test_cstyle.ml` "Clang ABI" › "fixed ABI wrapper", "fixed ABI wraps inner kernel" | a C entry the host calls | dropped: tinygrad writes no entry (`ClangRenderer._render_entry` is empty); the test executor `Host` adds one, and rune's executor will (L9) |
| old: `unit/test_cstyle.ml` "CUDA Launch Bounds", "CUDA Device Name", "uint spelling" (2 tests) | | `CS › sources › cuda_*`; `CS › rendering › writes the same CUDA source for the CUDA and NV devices`; `CS › declarations › cachekey` (`compile_nv_sm_89`) |
| old: `unit/test_cstyle.ml` "Variable Naming" (2 tests) | | `CS › sources` (`Lidx0`, `Ridx0`, `gidx0`, `lidx0` throughout) |
| old: `unit/test_cstyle.ml` "Unbounded loops" (3 tests) | `for (;;)`, the exit test at the bottom, indented inside the loop | `CS › sources › *_unbounded_loop`; `CS › execution on the host › loops until its bottom test fails` |
| old: `unit/test_cstyle.ml` "Register loads" (2 tests) | a register load read once is inlined, one read more is named | `CS › sources › *_sum`, `*_matmul` |
| old: `unit/test_cstyle.ml` "Custom" › "custom_inline" | | `CS › sources › *_custom`; `CS › rendering › formats custom code as str.format does`; `CS › execution on the host › runs custom code` |
| old: `unit/test_cstyle.ml` "AMD/HIP" › "nontemporal loads keep their scalar or vector pointer type" | | `CS › sources › hip_nontemporal` (a scalar load; HEAD merges no nontemporal vectors) |
| old: `unit/test_cstyle.ml` "AMD/HIP" › "special dims", "transcendentals", "barrier", "kernel attribute", "bf16 target paths" | | `CS › sources › hip_*`, `hip_transcendental_*`, `hip_group_reduce`, `hip_dtype_bf16`, `hip_cdna4_dtype_bf16` |
| old: `unit/test_cstyle.ml` "AMD/HIP" › "CDNA FP8 formats follow the hardware encoding", "CDNA3 FNUZ conversion uses its exponent bias and range", "CDNA3 FNUZ MFMA uses packed operands", "cdna fp8 constants use f32_to_fp8 helper", "cdna fp8 casts use f32_to_fp8 helper", "cdna WMMA emits MFMA macro and extra call arguments" | | `CS › declarations › supported`; `CS › sources › hip_cdna3_*`, `hip_cdna4_*` (dtypes, constants, infinities, tensor cores); `CS › extra_matcher › rewrites.golden › case=hip_cdna3_e5m2fnuz_product` |
| old: `unit/test_cstyle.ml` "AMD/HIP" › "rdna4 WMMA emits gfx12 builtin macro", "rdna3 int8 WMMA uses sanitized signed-char names", "rdna3 half output WMMA emits wrapper" | | `CS › sources › hip_rdna4_tc_*`, `hip_tc_signed_char_int_16_16_16`, `hip_tc_half_half_16_16_16` |
| old: `unit/test_cstyle.ml` "Properties" › "rendering does not retain the expression graph", "deterministic" | | `CS › rendering › keeps no reference to the kernel it rendered`, `keeps no state from one kernel to the next` |
| old: `unit/test_cstyle.ml` "Properties" › "non-empty output", "contains kernel name", "balanced braces" | | `CS › sources` pins every source whole |
| old: `golden/cstyle/<lang>_<case>.expected` for Clang, Metal, CUDA and AMD (64 files) | `simple_add_f32`, `simple_mul_i32`, `bitcast_f32_to_i32`, `cast_f16_to_f32`, `conditional`, `const_inf_nan`, `gated_load`, `loop`, `nested_loops`, `multi_param`, `shared_memory`, `special_dims`, `unary_sqrt_f16`, `unary_sqrt_f32`, `vectorize_index`, `where_select` | `CS › sources › <renderer>_add`, `_dtype_int`, `_bitcast`, `_dtype_half`, `_gated_store_in_loop`, `_inf_nan_*`, `_padded`, `_sum`, `_matmul`, `_scalar_params`, `_group_reduce`, `_matmul_locals`, `_transcendental_half`, `_transcendental_float`, `_matmul_upcasted`, `_where_max` |
| old: `golden/cstyle/opencl_*.expected` (16 files) | | dropped: OpenCL is excluded |
| old: `runtime/cpu/test_compiler.ml`, `test_cuda_abi.ml` | | dropped here: `Compiler_cpu`'s and `Compiler_cuda`'s sections |

### Rows other sections left to Cstyle's

| Source | Behaviour | Outcome |
|---|---|---|
| Transcendental: old `unit/test_cstyle.ml` "transcendentals" | AMD writes native sqrt and sin | `CS › code_for_op › written.golden › renderer=HIP op=Ops.SQRT`, `Ops.SIN`; `sources › hip_transcendental_*` |
| Tc: old `unit/codegen/test_tc.ml` "to_string", "tinygrad table names" | `WMMA_<dims>_<in>_<out>` names | `CS › sources › *_tc_*` (the helpers and calls named `__WMMA_8_16_16_half_float`, `__WMMA_16_16_16_signed_char_int`, ...) |
| Tc: old `unit/test_cstyle.ml` "metal tensor cores follow Apple GPU family" | | `CS › declarations › tensor_cores › declarations.golden` (Apple5 to Apple9, Mac2) |
| Renderer: old `unit/test_cstyle.ml` `Renderer.supports_dtype Cstyle.qcom` | | dropped: QCOM is excluded; the other targets' types are `CS › declarations › supported` |
| Decomp_op: old `unit/codegen/test_decompositions.ml` "fast idiv is enabled for Metal" | | dropped: no renderer of `cstyle.py` turns fast division off at HEAD; `CS › sources › metal_idiv` pins Metal's division |
| Decomp_op: old `unit/test_cstyle.ml` "renderer op capabilities match cstyle render surface" | | `CS › declarations › code_for_op` (no row lists `MAX`, `MULACC` or `THREEFRY`) |
| Coalesce: old `unit/test_cstyle.ml` vector pointer casts and `__builtin_nontemporal_load` | | `CS › sources › *_matmul_upcasted`, `*_group_reduce_upcasted`, `hip_nontemporal` |

## C

The suite is `Tolk_next.C` (`runtime/support/c/`), written `C` below.
`findlib_trees.golden` holds 41 searches of `DLL.findlib`, each a tree of
files, an environment and the paths searched, with the file tinygrad finds:
25 run as Linux searches and 16 as macOS's, whatever the platform that
generated them, and each row runs on its own platform. No tinygrad test and no
old tolk test covers `findlib`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | `NAME_PATH` naming a file wins, an empty or missing one is passed over, a directory it names is searched first | `C › as tinygrad › findlib_trees.golden › platform=* case=name_path_*` |
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | `LD_LIBRARY_PATH` then the system's directories then `extra_paths`, in order, empty entries ignored, missing directories skipped | `C › as tinygrad › findlib_trees.golden › platform=* case=ld_library_path_*`, `extra_paths_order`, `missing_dirs_skipped`; `C › search order › searching two lists of directories finds the first of each search` (the law) |
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | Linux: `lib<p>.so[.0-9]*` that starts as an ELF file; linker scripts, directories, dangling links and other names passed over | `C › as tinygrad › findlib_trees.golden › platform=linux case=version_*`, `linker_script_*`, `other_names_rejected`, `directory_named_as_library`, `symbolic_link_to_elf`, `dangling_symbolic_link` |
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | Linux: several ELF candidates in one directory | `C › one directory › the first ELF file in the order of names is found`, `a linker script first in the order of names is passed over`. tinygrad takes the first that `iterdir` lists, in an order the file system chooses |
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | macOS: `lib<p>.dylib`, `<p>.dylib`, `<p>`, in that order per directory; a dangling link inside a framework | `C › as tinygrad › findlib_trees.golden › platform=darwin case=*`; `C › the system's directories › MTLCompiler is found among macOS's private frameworks` |
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | an absolute path is taken if it is a file, else the next path is searched; each path is searched through every directory before the next | `C › as tinygrad › findlib_trees.golden › platform=* case=absolute_*`, `paths_before_directories`, `name_path_dir_each_path` |
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | not found | `C › absence › a name nowhere is not found`, `no paths find nothing, even with a directory that holds the name`; `C › as tinygrad › findlib_trees.golden › platform=* case=not_found`, `no_paths` |
| tinygrad: `runtime/support/c.py` `DLL.findlib` (no test) | the system's directories on Linux, `/lib/<MULTIARCH>` among them | `C › the system's directories › the math library is found among Linux's library directories`; `C › multiarch › names a Linux machine` |
| tinygrad: `runtime/support/c.py` `DLL.findlib` `libc` and `m` on macOS | `/usr/lib/lib<nm>.dylib` returned without a search | dropped: no caller of `C.findlib` loads the C or math library |

## Compiler_cpu

The suite is `Tolk_next.Compiler_cpu` (`runtime/support/compiler_cpu/`),
written `CPU` below. `kernels.golden` holds the kernels it renders with
`Cstyle.clang`: `add` and `half_add`, the `Tensor` programs of tinygrad's tests,
and `sqrt`. CC is read once per process, so the tests tagged `no-clang` run in
a second process, where CC names a program that does not exist. A test that
compiles for a machine the installed Clang has no backend for, or disassembles
an object the installed objdump cannot read, skips.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `device/cpu/test_cpu.py::TestCPU::test_arch_feats` | `x86_64,x86-64,avx` puts `vmov` in the add kernel, `-avx` does not | `CPU › as tinygrad › the add kernel moves vectors with vmov iff AVX is on: › x86_64,x86-64,avx`, `› x86_64,x86-64,-avx` |
| tinygrad: `null/test_compile_failures.py::TestDisassembly::test_float16_alu` | on Apple's processors a half addition has no `fcvt` | `CPU › as tinygrad › a half addition on an Apple processor converts nothing` |
| tinygrad: `null/test_elf.py::TestElfLoader::test_clang_jit_compiler_external_raise`, `test_load_clang_jit_strtab`, `test_link` | the loader refuses an unresolved symbol, reads `.rela.text`, links libm | dropped: loading is nx.device's (`Nx_device.Program`); that the compiled object needs nothing of a library is `CPU › execution on the host › a compiled square root runs without a library` |
| tinygrad: `runtime/support/compiler_cpu.py` `-fno-math-errno` (no test) | a square root is an instruction, not a call | `CPU › objects › a square root is one instruction on › x86_64,x86-64`, `› arm64,generic`; `CPU › execution on the host › a compiled square root runs without a library` |
| tinygrad: `runtime/support/compiler_cpu.py` `-ffixed-x18` (no test) | x18 is left alone on arm64 | `CPU › features › arm64 leaves x18 alone` |
| tinygrad: `runtime/support/compiler_cpu.py` features (no test) | arm64 `f` is `+f` and `-f` is `+nof`; riscv64 joins features as extensions, `native` is `rv64g` | `CPU › features › a half addition on arm64 converts iff fp16 is off:` (2), `arm64 vectorizes with its SIMD registers`, `-simd on arm64 disables them`, `a feature on riscv64 is an extension the processor gains`; `CPU › objects › native on riscv64 is rv64g` |
| tinygrad: `runtime/support/compiler_cpu.py` the arch assertion and `unsupported arch` | a malformed arch or another machine is refused | `CPU › architectures › an architecture of fewer than two fields is refused, named:` (3), `another machine is refused, named:` (3) |
| tinygrad: `runtime/support/compiler_cpu.py` `Compiler.__init__` cache key | `compile_clang_obj_` and the arch's fields joined by `_` | `CPU › cache › objects are cached in the table of the architecture` (3), `objects are not cached without ccache`; `CPU › without Clang › a cached object is served without running Clang` |
| tinygrad: `runtime/support/compiler_cpu.py` `disassemble` | `objdump -d` of the object | `CPU › disassembly › prints what objdump prints of the object` |
| old: `unit/runtime/cpu/test_compiler.ml` "outputs relocatable ELF for normalized host" | ELF magic, class, byte order, relocatable type, the host's machine | `CPU › objects › native is the host's processor` |
| old: `unit/runtime/cpu/test_compiler.ml` "explicit architectures" | the ELF machine of `x86_64`, `arm64` and `riscv64` | `CPU › objects › compiles to a relocatable ELF object for › x86_64,x86-64`, `› arm64,generic`, `› riscv64,rv64g` |
| old: `unit/runtime/cpu/test_compiler.ml` "architecture requires a CPU field" | `arm64` alone is refused | `CPU › architectures › an architecture of fewer than two fields is refused, named: › "arm64"`; the old refusal was a `Compile_error` at compile, tinygrad's and tolk.next's is at construction |
| old: `unit/runtime/cpu/test_compiler.ml` "output is parseable by ELF support" | the object has `.text` and the symbol | dropped: ELF parsing is nx.device's; `CPU › execution on the host › a compiled kernel adds two buffers` loads and runs the object |
| old: `unit/runtime/cpu/test_compiler.ml` "invalid C raises Compile_error" | Clang's diagnostics in the error | `CPU › errors › a rejected source raises Compile_error with Clang's diagnostics`; `CPU › without Clang › a compile raises Compile_error naming the program CC names` |
| old: `unit/test_elf.ml` external-call relocations of Clang objects | PLT32 relocations of an object | dropped: ELF relocation is nx.device's |
| none | determinism | `CPU › objects › one source compiles to the same bytes` |

## Compiler_cuda

The suite is `Tolk_next.Compiler_cuda` (`runtime/support/compiler_cuda/`),
written `CUDA` below. NVRTC is loaded once per process, so the tests tagged
`no-library` run in a second process, where `NVRTC_PATH` names a file that is
no library. The tests of NVRTC itself are slow and skip without it; they pass
against NVRTC 12.8 on Linux arm64.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/support/compiler_cuda.py` `NVRTCCompiler.compile` (no test) | PTX and cubins | slow: `CUDA › NVRTC › compiles a kernel to PTX`, `compiles a kernel to a cubin`, `compiles from several domains at once` |
| tinygrad: `runtime/support/compiler_cuda.py` `nvrtc_check` | `Nvrtc Error`, NVRTC's error string and log | slow: `CUDA › NVRTC › a rejected source raises Compile_error with NVRTC's log` |
| tinygrad: `runtime/support/compiler_cuda.py` `NVRTCCompiler.__init__` cache key | `compile_<cache_key>_<arch>` | `CUDA › cache` (4 tests); `CUDA › a library that does not load › a cached binary is served without loading NVRTC` |
| tinygrad: `runtime/support/compiler_cuda.py` `cuda_disassemble` | `ptxas` then `nvdisasm`, or why they failed | `CUDA › disassembly › a PTX that ptxas cannot assemble prints why`, `a cubin that nvdisasm cannot read prints why` |
| D15 | making the compiler loads nothing; the first compile raises | `CUDA › without NVRTC on the machine › a compile raises Compile_error naming nvrtc and NVRTC_PATH`; `CUDA › a library that does not load` (4 tests) |
| tinygrad: `device/nv/test_renderer_failures.py`, `runtime/test_renderer_failures.py` | kernels that once failed to render | dropped here: rendering is `Cstyle`'s section; slow: `Tolk_next.Cstyle › every GPU kernel compiles with its target's toolchain` compiles every CUDA kernel |
| old: `unit/test_cuda_abi.ml` "concurrent initialization publishes a complete driver table" | a load from several domains at once | `CUDA › a library that does not load › compiles from several domains at once raise it`; slow: `CUDA › NVRTC › compiles from several domains at once`. The driver table itself is the executor's (nx.device) |
| old: `unit/test_cuda_abi.ml` "missing driver symbols stay failed across callers" | a failed load stays failed | `CUDA › a library that does not load › every compile raises the same Compile_error` |
| old: `unit/test_cuda_abi.ml` the timestamp, submission, copy, host registration and shutdown tests (5) | the CUDA driver's queues and memory | dropped: the runtime is nx.device's and `tolk.next.engine`'s (plan §1a) |
| old: `unit/test_runtime_nv.ml` "the recorded nvrtc kernel parses to its recorded fields" | a recorded kernel | dropped: the NV runtime is nx.device's |

## Compiler_amd

The suite is `Tolk_next.Compiler_amd` (`runtime/support/compiler_amd/`),
written `AMD` below. The tests tagged `no-library` run in a second process,
where `COMGR_PATH` names a file that is no library. The tests of comgr itself
are slow and skip without it; no machine here has ROCm, so they have not run.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/support/compiler_amd.py` `compile_hip` (no test) | HIP to a code object; `.text` sources assembled | slow: `AMD › comgr › compiles a kernel to a code object`, `assembles a source whose first line is .text`, `compiles from several domains at once` |
| tinygrad: `runtime/support/compiler_amd.py` `HIPCompiler.compile` | comgr's failure is a `CompileError` | slow: `AMD › comgr › a rejected source raises Compile_error with comgr's log` |
| tinygrad: `runtime/support/compiler_amd.py` `HIPCompiler.__init__` cache key | `compile_hip_<arch>` | `AMD › cache` (2 tests); `AMD › a library that does not load › a cached code object is served without loading comgr` |
| tinygrad: `runtime/support/compiler_amd.py` `disassemble` | `llvm-objdump` of the code object | slow: `AMD › comgr › disassembles a code object` |
| D15 | making the compiler loads nothing; the first compile raises | `AMD › without comgr on the machine › a compile raises Compile_error naming comgr and COMGR_PATH`; `AMD › a library that does not load` (4 tests) |
| tinygrad: `external/external_test_hip_compile.py`, `external/external_benchmark_hip_compile.py` | compile time against a reference | dropped: benchmarks |
| tinygrad: `amd/hw/*` (`HIPCompiler` through `amd/hw/helpers.py`) | instructions on AMD hardware | dropped: they run on the device, the executor's |
| tinygrad: `device/amd/test_llvm.py` | `AMDLLVMCompiler` | dropped: the LLVM compilers are excluded (README) |
| old: `unit/test_runtime_amd.ml` "missing comgr degrades to Failure" | a missing comgr fails at compile | `AMD › without comgr on the machine › a compile raises Compile_error naming comgr and COMGR_PATH`; tinygrad's and tolk.next's failure is a `Compile_error` |
| old: `unit/test_runtime_amd.ml` "load failure is retried, not latched" | a failed load is tried again | dropped: tinygrad loads comgr once, at import; `AMD › a library that does not load › every compile raises the same Compile_error` pins the latch |
| old: `unit/test_runtime_amd.ml` "compiles a trivial HIP kernel", "broken source raises Compile_error" | | slow: `AMD › comgr › compiles a kernel to a code object`, `a rejected source raises Compile_error with comgr's log` |

## Compiler_metal

The suite is `Tolk_next.Compiler_metal` (`runtime/support/compiler_metal/`),
written `M` below. It covers `MetalCompiler`, the part of `ops_metal.py` that
precedes `Cstyle` (DIVERGENCES D4). MTLCompiler is
loaded once per process, so the tests tagged `no-library` run in a second
process, where `MTLCOMPILER_PATH` names a file that is no library. The tests
of MTLCompiler itself skip elsewhere than on macOS.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `device/metal/test_metal.py::TestMetal::test_compile_success` | a kernel compiles to a library | `M › MTLCompiler › compiles a kernel to a Metal library` |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_compile_error` | `CompileError` on bad source | `M › MTLCompiler › a rejected source raises Compile_error with the compiler's message` |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_alloc_oom`, `test_failed_newLibraryWithData`, `test_free` | the device's memory and pipelines | dropped: the device is nx.device's and `tolk.next.engine`'s (plan §1a) |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.compile` (no test) | the reply's data starts after the header and the warnings | `M › MTLCompiler › a source that draws warnings compiles to a Metal library` |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.compile` (no test) | the Metal version by macOS, `-fno-fast-math` | `M › MTLCompiler › the language is the latest the running macOS compiles kernels of`, `fast math is off` |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.__reduce__` | one compiler per forked process | dropped: no pickling; `M › MTLCompiler › compiles from several domains at once` pins the concurrent use of one service |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.disassemble` | the applegpu disassembler | `M › MTLCompiler › disassembly prints nothing` (README) |
| tinygrad: `runtime/ops_metal.py` cache key | `compile_metal_direct` | `M › cache` (2 tests); `M › a library that does not load › a cached library is served without loading MTLCompiler` |
| D15 | making the compiler loads nothing; the first compile raises | `M › without MTLCompiler on the machine › a compile raises Compile_error naming MTLCompiler and MTLCOMPILER_PATH`; `M › a library that does not load` (4 tests) |
| tinygrad: `external/external_metal_compile_fail.py` | a kernel that crashed Metal's compiler | dropped: a crash reproducer of a driver bug, outside tinygrad's suite |
| old: `unit/test_runtime_metal.ml` "compile and run one kernel" | | dropped: running is the device's; compiling is `M › MTLCompiler › compiles a kernel to a Metal library` |

## Indexing

The suite is `Tolk_next.Indexing` (`schedule/indexing/`), written `IX` below.
`IX › apply_movement_op` holds tinygrad's index of each movement on hand-built
cases (`movement_<case>.golden`), and `IX › apply_movement_op › laws` states,
per kind of movement over generated shapes, that the index reads the element
the movement places there (`Tensors`, the `Interpreter`) and that the indices
of two movements compose. `IX › run_rangeify › recorded graphs` holds, for
each `Tensor` program, the graph tinygrad hands `run_rangeify` when it
schedules the program on the CPU and what `run_rangeify` returns
(`<program>_rangeified.golden`), and a few hand-built graphs; `IX ›
run_rangeify › writes` states that ranges keep what a program writes, where
the result is one kernel; `IX › run_rangeify › debug` holds the printout; the
other groups state the interface's rules one at a time. One mutant survives,
equivalent: `i < data_src_count` as `<=` in `indexing.ml`'s indexing of
sources, since only a movement has storage past its data sources, and
movements are removed afterwards.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: runtime/test_rangeify.py::TestDoubleMatmul::test_double_matmul | two matmuls in a row | `IX › run_rangeify › recorded graphs › double_matmul_rangeified.golden`; the numbers are the executor's (`tolk.next.engine`) |
| tinygrad: runtime/test_rangeify.py::TestRangeifyAssign::test_assign_permuted | an assign through a permute | `IX › run_rangeify › recorded graphs › assign_permuted_rangeified.golden`; `IX › run_rangeify › writes › assign_permuted writes what its tensors write` |
| tinygrad: runtime/test_rangeify.py::TestRangeifyEdgeCase::test_variable_stack_data | a stack of variables used as data gets ranges | `IX › run_rangeify › recorded graphs › variable_stack_rangeified.golden` |
| tinygrad: runtime/test_rangeify.py::TestRangeifyEdgeCase::test_variable_data_and_shape | a variable read as data and as a size | `IX › run_rangeify › recorded graphs › variable_data_and_shape_rangeified.golden` |
| tinygrad: runtime/test_rangeify.py::TestRangeifyEdgeCase::test_matmul_relu_cat | a matmul concatenated to a buffer | `IX › run_rangeify › recorded graphs › matmul_relu_cat_rangeified.golden` |
| tinygrad: runtime/test_rangeify.py::TestRangeifyEdgeCase::test_multi_gather | two gathers of one table, a stage placed on a device | `IX › run_rangeify › recorded graphs › two_gathers_rangeified.golden`; `IX › run_rangeify › rewrites › a stage of a value placed nowhere lives on the sink's device` |
| tinygrad: runtime/test_rangeify.py::TestRangeifyPM (7 tests) | `pm_rangeify` | dropped: skipped upstream, the matcher no longer exists |
| tinygrad: runtime/test_assign.py::TestAssign::test_assign_double_diamond_reduce | a value stored into storage it reads is stored first | `IX › run_rangeify › recorded graphs › assign_double_diamond_rangeified.golden`; `IX › run_rangeify › stores › a value stored into storage it reads is stored whole first` |
| tinygrad: runtime/test_custom_kernel.py (the kernels' sources) | a source of a kernel given as code is stored, and stays | `IX › run_rangeify › recorded graphs › custom_kernel_rangeified.golden` (its stage is not removable) |
| tinygrad: null/test_schedule.py::TestSchedule::test_basic_binop_fusion, test_basic_binop_fusion_deep, test_mulacc_fusion, test_binop_reshape_fusion, test_binop_permute_fusion, test_reduce_reshape_binop_fusion, test_reduce_permute_binop_fusion, test_diamond_folded, test_fold_double_unary, test_push_permute_through_reshape, test_children_dont_push, test_shrink_fuse, test_reduce_permute_nofuse, test_multistage_reduce, test_two_sum, test_contiguous_add, test_reduce_shrink | what these programs store whole | `IX › run_rangeify › recorded graphs ›` `elementwise_three`, `mulacc`, `binop_reshape`, `binop_permute`, `reduce_reshape_binop`, `reduce_permute_binop`, `shared_sum`, `reduce_unary`, `permute_through_reshape`, `children_dont_push`, `shrink_fuse`, `reduce_permute_nofuse`, `multistage_reduce`, `two_consumers`, `contiguous_add`, `reduce_shrink` (`_rangeified.golden`) and the writes law on each that is one kernel; the kernel counts are Rangeify's section (L6), which splits kernels |
| tinygrad: null/test_schedule.py::TestSchedule::test_pad_reduce_safe, test_layernorm_onelayer, test_argmax, test_argmax_one_kernel, test_conv2d, test_resnet_block | pads under reductions, norms, argmax, convolutions | `IX › run_rangeify › recorded graphs › pad_reduce`, `layernorm`, `standardize`, `rmsnorm`, `argmax`, `conv`, `conv_bn_relu` (`_rangeified.golden`); the kernel counts are Rangeify's section |
| tinygrad: null/test_schedule.py (the other kernel-count tests of TestSchedule, TestContiguous, TestSimpleSchedule, TestFusionOp, TestBufferView, TestLimitBufs, TestCopyFolding) | kernel counts of the whole scheduler | dropped here: Rangeify's section (L6); every behaviour of `run_rangeify` they reach is one of the rules and recorded graphs above |
| tinygrad: null/test_arange.py, runtime/test_arange.py | aranges, gathers and embeddings compile to few operations | `IX › run_rangeify › recorded graphs › arange`, `embedding`, `gather`, `two_gathers`, `cumsum`, `triu` (`_rangeified.golden`); the operation counts are Simplify's and Codegen's sections |
| tinygrad: external/external_test_schedule_scaling.py | scheduling time grows linearly | dropped: a timing benchmark |
| tinygrad: external/external_uop_gc.py (`apply_movement_op.cache_clear`) | the cache of indices releases its graphs | dropped: tinygrad's `functools.cache`; the port memoizes nothing |

### old tolk: unit/test_schedule_rangeify.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/test_schedule_rangeify.ml group "is_always_contiguous" (13 tests) | which nodes are storage | the set is private; storage sources of a gather across devices are not stored (`IX › run_rangeify › recorded graphs › shard_sum_rangeified.golden`), computed ones are (`IX › run_rangeify › rewrites › a source of a gather across devices is stored on its device`), and so are a custom kernel's (`custom_kernel_rangeified.golden`). `CONTIGUOUS` and `COPY` are no storage in tinygrad (a copy is a store before rangeify) |
| old: unit/test_schedule_rangeify.ml "size 1 gives const 0", "symbolic size resolving to 1 gives const 0" | a unit axis gets no range | `IX › run_rangeify › new ranges › a stored node gets a range per axis, and a unit axis none`; a size that folds to 1 is `Int 1` before rangeify |
| old: unit/test_schedule_rangeify.ml "size 0 gives Range (resolve(s!=1) is true)" | | `IX › run_rangeify › new ranges › an empty axis gets a range` |
| old: unit/test_schedule_rangeify.ml "size > 1 gives Range", "axis increments", "kind propagates" | | `IX › run_rangeify › new ranges › the axes a reduction reduces get reduce ranges, numbered after` |
| old: unit/test_schedule_rangeify.ml "range size returns existing range" | an axis whose size is a range reuses it, and takes no number | `IX › run_rangeify › rewrites › a stored axis whose size is a range is indexed by that range`; `expand_by_range_stored_rangeified.golden` (new ranges 0 to 2 around range 7) |
| old: unit/test_schedule_rangeify.ml group "range helpers" (2 tests) | `get_idx` and `get_valid` | dropped here: Ops' section |
| old: unit/test_schedule_rangeify.ml groups "apply_movement_op › shrink", "› flip" (4 tests) | | `IX › apply_movement_op › movement_shrink.golden`, `movement_flip.golden`; the shrink and flip laws |
| old: unit/test_schedule_rangeify.ml "swap [1;0]" | | `movement_permute.golden`; the permute laws |
| old: unit/test_schedule_rangeify.ml "identity elides at construction" | | dropped here: `Ops.mop`'s, Ops' section |
| old: unit/test_schedule_rangeify.ml group "apply_movement_op › expand" (3 tests) | | `movement_expand.golden`, `movement_expand_symbolic.golden`; the expand laws |
| old: unit/test_schedule_rangeify.ml group "apply_movement_op › pad" (4 tests) | an unpadded axis passes, a padded one is valid within its source, at either end, of symbolic size | `movement_pad.golden`, `movement_pad_end.golden`, `movement_pad_symbolic.golden`; the pad laws |
| old: unit/test_schedule_rangeify.ml group "apply_movement_op › reshape" (4 tests) | | `movement_reshape_flatten.golden`, `movement_reshape_unflatten.golden`, `movement_reshape_symbolic.golden`; the reshape laws (a reshape to the same shape among them) |
| old: unit/test_schedule_rangeify.ml "realized node creates Realized", "realized node has range_map entry", "2D realized node has all axes" | | the map is private: `IX › run_rangeify › new ranges › a stored node gets a range per axis, and a unit axis none`; `IX › run_rangeify › stores › a stored store is closed by an end over its ranges` |
| old: unit/test_schedule_rangeify.ml "the apply pass adds no map entries" | nodes that differ only in movements keep their own ranks | `IX › run_rangeify › recorded graphs › scalar_and_wide_uses_rangeified.golden` (the same graph) |
| old: unit/test_schedule_rangeify.ml "elementwise inherits consumer ranges" | | `IX › run_rangeify › new ranges › consumers that index a node alike share its ranges` |
| old: unit/test_schedule_rangeify.ml "reduce creates reduce-kind ranges" | | `IX › run_rangeify › new ranges › the axes a reduction reduces get reduce ranges, numbered after` |
| old: unit/test_schedule_rangeify.ml "movement op has different in and out ranges" | | `IX › run_rangeify › debug › *_debug.golden` (a movement prints its source's ranges, then its own) |
| old: unit/test_schedule_rangeify.ml "symbolic param shape creates symbolic range size" | | `IX › run_rangeify › recorded graphs › variable_shrink_rangeified.golden`, `variable_offset_rangeified.golden` |
| old: unit/test_schedule_rangeify.ml "reduce indexes direct source before lowering" | | `IX › run_rangeify › rewrites › a reduction of leading axes becomes a reduction over ranges`; `sum_rangeified.golden` |
| old: unit/test_schedule_rangeify.ml "pad where uses indexed child" | | `IX › run_rangeify › rewrites › a pad becomes a selection of its source and of zero`; `pad_rangeified.golden` |
| old: unit/test_schedule_rangeify.ml "staged elementwise indexes raw params" | a stage's source reads storage through its ranges | `IX › run_rangeify › recorded graphs › two_consumers_permuted_rangeified.golden`, `shared_view_rangeified.golden` |
| old: unit/test_schedule_rangeify.ml "partial reshape index maps to source prefix" | `apply_movement_op` on a prefix of a shape | the reshape laws (the prefix is a shape of its own); the call is Prepare's section |
| old: unit/test_schedule_rangeify.ml group "get_kernel_graph" (13 pipeline tests), group "reshape merge" | kernel counts of fusions | the programs are recorded here (see the tinygrad row of test_schedule.py; "reshape chain" is `reshape_chain_rangeified.golden`); the counts and "rejects distinct written states" are Rangeify's section |
| old: unit/test_schedule_rangeify.ml group "stack selection" (5 tests) | a selection of 1, 8, 9, 17 and 1024 sources picks each and nests in logarithmic depth | `IX › run_rangeify › stacks › a stack of <n> constants writes each` (1, 8, 9, 17, 100); `IX › run_rangeify › stacks › a stack of <n> constants selects the last at a negative index, in logarithmic depth` (2, 8, 9, 17, 1024); `stack_eight_rangeified.golden`, `stack_twelve_rangeified.golden`. An index past the last source is dropped: the interface states only a negative one, and ranges never leave the stack |
| old: unit/test_schedule_rangeify.ml groups "split_reduce", "symbolic variables", "stage capacity", "symbolic empty shapes", "moved materializations", "symbolic storage views", "packed argument buffer limits", "kernel splitting preserves independent symbolic ranges", "Shape queries release graphs" | | dropped here: Rangeify's, Prepare's and Ops' sections |

### old tolk: golden/rangeify

| Source | Behaviour | Outcome |
|---|---|---|
| old: golden/rangeify `binop_permute`, `binop_reshape`, `contiguous_add`, `diamond`, `elementwise_3way`, `elementwise_add`, `expand_permute`, `mulacc`, `multistage_reduce`, `permute_through_reshape`, `reduce_permute_binop`, `reduce_reshape_binop`, `reduce_shrink`, `reduce_unary`, `reshape_chain`, `shrink_fuse`, `two_sum` (each on 5 renderers) | the kernels of each program | the ranges: `IX › run_rangeify › recorded graphs ›` `binop_permute`, `binop_reshape`, `contiguous_add`, `shared_sum`, `elementwise_three`, `add`, `children_dont_push`, `mulacc`, `multistage_reduce`, `permute_through_reshape`, `reduce_permute_binop`, `reduce_reshape_binop`, `reduce_shrink`, `reduce_unary`, `reshape_chain`, `shrink_fuse`, `two_consumers`; the rendered sources are the renderers' and the end-to-end suites (L4, L5) |
| old: golden/rangeify `llama_*` (5 cases), `test_llama.ml` | a small Llama's kernels | dropped here: the end-to-end suite; its operations are recorded here as `rmsnorm`, `attention`, `softmax`, `matmul`, `embedding` |

## Allreduce

The suite is `Tolk_next.Allreduce` (`schedule/allreduce/`), written `AR`
below. `AR › handle_allreduce › recorded` holds tinygrad's expansion of
allreduces on 2 to 8 CPU devices under each algorithm's settings
(`<case>_handled.golden`), and `AR › create_allreduce_function › recorded`
the function of some (`<case>_function.golden`, its new storage numbered as
tinygrad's); `AR › handle_allreduce › values` and `› laws` state that an
expansion and a function hold, on each device, the elementwise reduction of the
shards (`Tensors`), on the recorded cases and on generated ones;
`AR › handle_allreduce › algorithms` states which algorithm applies and how
each moves values between devices. One mutant survives, equivalent:
`(i + step) mod ndev` as `i - step` in the ring's source index, which is read
only at step 0.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_allreduce.py::TestRingAllReduce::test_schedule_ring | a ring on 4 devices makes 2N(N-1) copies along a ring | `AR › handle_allreduce › algorithms › a ring moves values only to the next device` |
| tinygrad: null/test_allreduce.py::TestRingAllReduce::test_schedule_naive | a naive allreduce makes N(N-1) copies between distinct devices | `AR › handle_allreduce › algorithms › a naive allreduce copies each shard to every device`; its two kernels are the scheduler's section |
| tinygrad: schedule/allreduce.py, the hierarchical branch of `handle_allreduce` (no upstream test) | a hierarchical allreduce to one device | D19: `AR › handle_allreduce › recorded › nodes_to_one_device_handled.golden, landed on its device (D19)` states the corrected graph from tinygrad's; `AR › rules › a hierarchical allreduce to one device lands there (D19)` checks its value and placement, and its function's. A known upstream defect: tinygrad leaves the value on every device, and `create_allreduce_function` then stores it into storage on the one target device |
| tinygrad: null/test_allreduce.py::TestAllreduceCast (3 tests) | `ALLREDUCE_CAST` keeps 16-bit copies | dropped here: the cast is `schedule/multi.py`'s, Multi's section |
| tinygrad: null/test_multitensor.py::TestMultiRamUsage::test_multi_layer_allreduce, test_allreduce_cast_dtype_memory | memory of allreduces | dropped here: Memory's and Multi's sections |
| tinygrad: null/test_multitensor.py (the other tests) | sharding, multi-device ALU, batch norm | dropped here: Multi's section |

### old tolk: unit/engine/test_collectives.ml, unit/engine/test_multi.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_collectives.ml "replicas agree bit for bit" (each strategy, 2 to 8 devices) | every device holds the same sums | `AR › handle_allreduce › values › <case> reduces its shards`; `AR › handle_allreduce › laws › an expansion reduces its shards` (each device's elements equal the reduction's exactly) |
| old: unit/engine/test_collectives.ml "replicas keep a sum's -0" | an allreduce of -0 is -0 | dropped: the reduction identity; RFC 0012's lowering decides. Every compiled sum starts from +0, so a sum of -0 is +0 on one device as on several |
| old: unit/engine/test_collectives.ml "a realized allreduce holds no more than a consumed one" | memory | dropped here: Memory's section |
| old: unit/engine/test_collectives.ml "a realized allreduce of a symbolic slice keeps its values", "blocks of a symbolic slice equal the allreduce's rows under boxes" | a symbolic allreduce | `AR › handle_allreduce › recorded › naive_symbolic_handled.golden`, `naive_symbolic_function.golden`; `AR › handle_allreduce › algorithms › a symbolic shape is kept, and a function stores its greatest` (a symbolic shape takes the naive algorithm, whatever the settings); the values need the executor |
| old: unit/engine/test_collectives.ml groups "copies", "all-gather", "reduce-scatter", "an allreduce also used whole is reduced once", "an allreduce in a call body raises", "float16 partials reduce-scatter in float16 under ALLREDUCE_CAST", the training harness | old tolk's collective calls and resharding | dropped here: tinygrad has no all-gather or reduce-scatter call; resharding and the cast are Multi's section, the harness the end-to-end suite |
| old: unit/engine/test_multi.ml "forced ring handles aligned empty chunks on four devices", "hierarchical scalar handles empty chunks" | chunks of no element | `AR › handle_allreduce › laws › an expansion reduces its shards` (covers "a chunk is empty" and "devices form nodes") |
| old: unit/engine/test_multi.ml "forced strategies reduce uneven chunks on four devices" | | `AR › handle_allreduce › recorded › ring_uneven_chunks_handled.golden`; `AR › handle_allreduce › values › ring_uneven_chunks reduces its shards`; the laws |
| old: unit/engine/test_multi.ml "each forced strategy schedules its own collective" | | `AR › handle_allreduce › recorded` (`naive_two_devices`, `ring_two_devices`, `all2all`, `nodes_of_two` differ); `AR › handle_allreduce › algorithms › the algorithm is the first that applies` |
| old: unit/engine/test_multi.ml "hierarchical maximum handles negative values" | | `AR › handle_allreduce › recorded › nodes_of_three_handled.golden` (a maximum); the laws draw negative elements and maxima |
| old: unit/engine/test_multi.ml "symbolic allreduce retains logical sizes under forced ring" | | `AR › handle_allreduce › algorithms › a symbolic shape is kept, and a function stores its greatest` |
| old: unit/engine/test_multi.ml groups "Ownership", "Resolution", "Execution", "Kernels over split storage", "Cuda", "partial multi-axis allreduce is rejected" | | dropped here: Multi's section, and the executor's |

## Support_memory

The suite is `Tolk_next.Support_memory` (`runtime/support/support_memory/`),
written `SM` below. `SM › traces.golden` replays 40 random traces recorded from
tinygrad's `TLSFAllocator`, one allocator each, and checks every address an
allocation returns. `SM › Tlsf_allocator › laws` holds a stateful test against
a model of the live blocks: each block lies in the range, aligned from the
base, and overlaps no live block, and an allocation is `None` exactly when no
free stretch holds the smallest subdivision that fits it. A property states
that freeing every block restores a fresh allocator. Four mutants survive, all
equivalent: the two updates of the per-level counts, which only let a search
skip empty levels; `block_size <= 0` as `< 0`, since a zero block size has
fewer bits than any level count and is refused by the next check; and
`size > 0` as `>= 0`, which inserts a block of no address that no allocation
can take.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: external/external_test_tlsf.py::TestTLSFAllocator::test_basic_alloc_free | blocks in address order, a freed one handed out again | `SM › Tlsf_allocator › blocks are handed out in address order, and a freed one again` |
| tinygrad: external/external_test_tlsf.py::TestTLSFAllocator::test_merge_blocks | freed neighbours merge | `SM › Tlsf_allocator › freed neighbours merge` |
| tinygrad: external/external_test_tlsf.py::TestTLSFAllocator::test_split_blocks | a freed block splits | `SM › Tlsf_allocator › a freed block splits` |
| tinygrad: external/external_test_tlsf.py::TestTLSFAllocator::test_out_of_memory | `MemoryError` past the range | `SM › Tlsf_allocator › a block larger than the range is None` (`None` for tinygrad's `MemoryError`) |
| tinygrad: external/external_test_tlsf.py::TestTLSFAllocator::test_fragmentation_handling | freeing alternate blocks | the stateful law and `traces.golden` |
| tinygrad: external/external_test_tlsf.py::TestTLSFAllocator::test_block_size_alignment | blocks of 20 and 35 addresses start at multiples of 16 | `SM › Tlsf_allocator › a block is at least the block size`. The upstream test fails at HEAD: a block is `max block_size n` addresses, so the second starts at 20 |
| tinygrad: external/external_test_tlsf.py::TestTLSFAllocator::test_custom_start_address, test_block_tracking | a base address, the block table | `SM › Tlsf_allocator › addresses start at the base`, `a block past the base is freed by its address`. Both upstream tests error at HEAD: they use `start_addr`, which `TLSFAllocator` no longer has, and the block table is private |
| tinygrad: external/external_fuzz_tlsf.py | random allocations and frees never corrupt each other's bytes | `SM › Tlsf_allocator › laws › blocks fit, align and never overlap, and None means no room` (blocks that never overlap cannot corrupt each other); `traces.golden` |

### old tolk: unit/runtime/support/test_memory.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/runtime/support/test_memory.ml group "shared virtual addresses" (2 tests) | domains allocate from one allocator under a lock | dropped: tinygrad's allocator has no concurrency contract, and tolk.next carries none it lacks |
| old: unit/runtime/support/test_memory.ml groups "Map_range", "Unmap_range", "Page_tables", "Alloc_vaddr", "Palloc", "Valloc", "Vfree", "Identity_map", "Six_level_dual" | the memory manager and its page tables | dropped here: `Support_memory` ports the TLSF allocator; the memory manager of the same tinygrad file is the runtime's |
| old: unit/test_amd_amdev.ml (TLSF as a virtual address allocator) | | dropped here: the AMD device's section |

## Memory

The suite is `Tolk_next.Memory` (`schedule/memory/`), written `ME` below.
`ME › memory_plan_rewrite › recorded` holds tinygrad's plans of schedules and
their held buffers (`<case>_planned.golden`, its arenas numbered as
tinygrad's): the schedules of tinygrad's `test_memory_planner.py`, random
ones, and those `linear_with_vars` plans for `Tensor` programs on CPU devices.
`ME › memory_plan_rewrite › laws` states, over generated schedules, that a
planned buffer is the bytes of an arena at a block, viewed as its type; that
buffers alive at once share no byte; that copies and other calls use arenas of
their own; and that held buffers stay.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_memory_planner.py::TestMemoryPlanner::test_simple_buffer, test_simple_pinned, test_all_pinned | live buffers never overlap; held buffers stay | `ME › memory_plan_rewrite › recorded › simple_planned.golden`, `some_held_planned.golden`, `all_held_planned.golden`; `ME › memory_plan_rewrite › rules › a schedule of held buffers only is itself`; the laws |
| tinygrad: null/test_memory_planner.py::TestMemoryPlanner::test_simple_buffer_offset, test_buffer_offset, test_buffer_offset2, test_all_offsets_of_one | buffers passed again by later calls (their `base=` aliases the same buffer) | `ME › memory_plan_rewrite › recorded › reused_planned.golden`; the laws (generated calls pass a buffer again) |
| tinygrad: null/test_memory_planner.py::TestMemoryPlanner::test_very_small_buffers | buffers under a block | `ME › memory_plan_rewrite › recorded › very_small_planned.golden` |
| tinygrad: null/test_memory_planner.py::TestMemoryPlanner::test_very_big_buffers | buffers of 2^64 and 2^128 bytes | `ME › memory_plan_rewrite › recorded › big_planned.golden` (up to 2^50 elements); sizes past OCaml's `int` are dropped: storage sizes are `int` |
| tinygrad: null/test_memory_planner.py::TestMemoryPlanner::test_copy_bufs_separate_from_compute, test_copy_bufs_reuse_among_copies, test_compute_bufs_reuse_among_compute, test_copy_and_compute_no_cross_reuse | copies and other calls use arenas of their own, each shared within its kind | `ME › memory_plan_rewrite › recorded › copy_apart_from_compute_planned.golden`, `copies_share_planned.golden`, `computes_share_planned.golden`; the laws |
| tinygrad: null/test_memory_planner.py::TestMemoryPlanner::test_multiple_copy_bufs_with_offsets, test_copy_bufs_pinned_mixed | | `ME › memory_plan_rewrite › recorded › copies_held_mixed_planned.golden`; the laws |
| tinygrad: null/test_memory_planner.py::TestMemoryPlanner::test_deferred_copy_frees_chain | a copy's buffer lives on after its last call | `ME › memory_plan_rewrite › recorded › copy_chain_planned.golden`; the laws (a copy's buffers live for as many calls again) |
| tinygrad: schedule/memory.py `_can_plan` on CL and WEBGPU devices | no plan on devices without views | dropped: those devices are excluded (README Exclusions); a disk is `ME › memory_plan_rewrite › recorded › disk_planned.golden` and `ME › memory_plan_rewrite › rules › a schedule on a disk is itself` |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/test_engine_schedule.ml "concurrent memory plans keep arenas distinct" | plans on several domains take distinct arena slots | `ME › memory_plan_rewrite › rules › each plan's arenas take new slots`; that the counter is safe across domains is `Ops.unique_num`'s, Ops' section |
| old: unit/engine/test_schedule.ml "memory plans internal buffers when not capturing" | the executor plans unless it captures | dropped here: when to plan is the executor's (`tolk.next.engine`) |

## Device

The suite is `Tolk_next.Device` (`device/`), written `DV` below.
`renderers.golden` holds `Compiled._select_renderer` for 36 DEV settings, each
device and each architecture the device reports (its own and none): the target
it renders for, and the renderer it picks, the error of a target that names a
renderer the device lacks, or the failure of the renderer itself, whose text
is the renderer's own. A device lists the renderers of its `Compiled` that
tolk.next ports. `<case>_program.golden` holds programs that `to_program`
compiles for Clang and Metal, with the empty compiler so that no toolchain
shapes them, and `elfs.golden`, `signatures.golden` and `layouts.golden` what
`to_elf` and `TinyELF.iter_sig` give for each; `signatures.golden` and
`reprs.golden` also hold tinygrad's repr, which the printers match. DEV is bound with
`Helpers.context`, so every setting runs in one process. The test that two
domains share a renderer cannot run under mutation testing, which forks.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_device.py::TestDevice::test_nonexistent_renderer | `CPU:TYPO` has no renderer; `CPU:CLANGJIT` suggests `CLANG` | `DV › renderer › picks a device's renderer as tinygrad does › renderers.golden` (`dev=CPU:TYPO`, `dev=CPU:CLANGJIT`, and a misspelling for each device); `DV › renderer › names the renderer a target misspells` |
| tinygrad: null/test_device.py::TestDevVar::test_dev_arch_override | an arch in DEV reaches the renderer | `renderers.golden` (`dev=::gfx942`, `dev=CUDA::sm_75`, ...); `DV › renderer › renders for the setting's target of the device` (the NULL device of the test is excluded) |
| tinygrad: null/test_device.py::TestDevice::test_env_online | the renderer follows DEV within a context, and is remembered | `renderers.golden` under `Helpers.context`; `DV › renderer's memory › returns the same renderer to a second call`, `returns one renderer per target, whatever the order of the calls` |
| tinygrad: null/test_device.py::TestDevice::test_env_overwrite_default_compiler | `DEV=CPU:LLVM`, `AMD:LLVM` pick another compiler | dropped: the LLVM renderers are excluded (README); `renderers.golden` pins that `CPU:LLVM` has no renderer and `CPU:CLANG` and `AMD:HIP` pick theirs |
| tinygrad: null/test_device.py::TestDevice::test_compiler_autodetect_fallback | a renderer that fails to make gives way to the next | dropped here: each device has one ported renderer, so the fallback never runs; `Helpers.select_first_inited` is `Helpers › selection`'s. A failing renderer's own message is `DV › renderer › fails with the renderer's own message on an architecture it refuses` |
| tinygrad: null/test_device.py::TestDevice::test_old_renderer_env_raises | `CPU_LLVM=1` is refused | dropped: the `{DEV}_{RENDERER}` migration check is not ported (README exclusions) |
| tinygrad: null/test_device.py::TestDevice::test_canonicalize, test_lowercase_canonicalizes | `cpu`, `CL:0` and `disk:...` canonicalize | `DV › renderer takes a device's name as it is (D6)`: tolk.next never parses a name, and refuses `cpu`, `CPU:0` and `DISK:...` |
| tinygrad: null/test_device.py::TestDevice::test_getitem_not_exist | `Device["TYPO"]` fails | `DV › renderer › raises on a name that is no device` |
| tinygrad: null/test_device.py::TestDevice::test_nonexistent_iface, test_dev_id_out_of_range, test_old_device_env_raises, test_set_device_default_raises, test_dev_contextvar | interfaces, device indices, `Device.DEFAULT` | dropped: opening devices is `tolk.next.engine`'s and nx.device's, and there is no default device (D3) |
| tinygrad: null/test_device.py::TestCompiler (3 tests) | the compiler cache | Renderer's section (`R › Compiler`) |
| tinygrad: null/test_device.py::TestRunAsModule::test_module_runs | `enumerate_devices_str` | dropped: not ported (README exclusions) |
| tinygrad: uop/ops.py `to_elf` (read by ops_cpu, ops_cuda, ops_metal, ops_nv, ops_qcom, realize) | buffers compacted in globals order, then variables | `DV › Tiny_elf.of_program › lays out the signature as tinygrad's to_elf does › signatures.golden`; `numbers buffers by their place among the globals, then variables` |
| tinygrad: device.py `TinyELF.iter_sig` (read by ops_cpu, ops_hip, hcq2) | values packed from an offset, each aligned to its size | `DV › Tiny_elf.iter_sig › packs a signature as tinygrad's iter_sig does › layouts.golden` (offsets 0, 3 and 8); `packs each value at the next offset aligned to its size` |
| tinygrad: test/helpers.py:146, test/amd/hw/*.py (`dev.runtime(prg.to_elf())`, hand-made `TinyELF`) | loading a program | dropped: loading is nx.device's (D3) |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/test_device.ml "compilation canonicalizes interleaved kernel arguments" | buffers in slots 7 and 2 and a variable take signature slots 0, 1, 2 | `DV › Tiny_elf.of_program › numbers buffers by their place among the globals, then variables` (its first example is this kernel); `signatures.golden` (`program=sparse`). The rendered prototype is Cstyle's |
| old: unit/uop/test_uop.ml compiled_signature_preserves_slots_and_types | name, lib, target, profile key, slots, shapes, names and types of `to_elf` | `DV › Tiny_elf.of_program › compiles as tinygrad's to_elf does`, `takes the program's binary as its lib`, `keys the program's profile with its key`, `takes the target of the program, whatever rendered it`, `signatures.golden` (named buffers and variables) |
| old: unit/uop/test_uop.ml binary_argument_layout | a packed layout of a signature, refused for void and weak types | `DV › Tiny_elf.iter_sig › packs each value at the next offset aligned to its size`; the refusal is dropped: tinygrad's `iter_sig` refuses no type |
| old: unit/uop/test_uop.ml incomplete_program_has_no_binary | `to_elf` of a sink or an uncompiled program | `DV › Tiny_elf.of_program › refuses a node that is no program, and a program not compiled` |
| old: unit/test_program_spec.ml "bounded scalar metadata retains the complete ABI", "named scalar formals preserve binding order and deduplication" | variables follow the buffers, in order | `DV › Tiny_elf.of_program › numbers buffers by their place among the globals, then variables` |
| old: unit/test_program_spec.ml "incomplete scalar metadata is rejected before ABI construction", "conflicting scalar declarations are rejected before dispatch", "buffer and scalar declarations share collision checks" | the old program record refused variables without names or bounds, and names that render alike | dropped: `Program_spec` has no counterpart, and tinygrad's `to_elf` checks neither; a variable's name and bounds are `Ops.variable`'s |
| old: unit/test_device.ml "Buffer.copy_from delegation" (3 tests), unit/test_device_no_engine.ml (2 tests) | buffer copies, and their failure before the engine links | the engine's section (L7): buffers are `tolk.next.engine`'s (D3) |
| old: unit/test_device.ml "device bootstrap registration rolls back failed initialization", "concurrent device lookup runs one opener", "incomplete devices are private to the initializing thread", "failed device initialization wakes waiting callers", "failed openers cannot publish provisional devices" | the device registry | dropped: the engine has no registry of devices by name (README) |
| old: unit/test_device.ml "independent views share root ownership across domains", "program storage belongs to the device across links", "failed buffer finalizers are reported without retrying teardown", "buffer finalizers wait for device operations", "buffer finalizers wait for overlapping systhreads", "suspended device operations reject unrelated fibers", "foreign retirement waits for the active native owner", "buffer finalizers preserve queued retirement error timing", the three "teardown shares device ownership" tests, "foreign access completion is captured, coalesced and retried", "same-name foreign completions retain separate owners" | buffer and device lifetimes across domains and threads | dropped: storage and its lifetime are `tolk.next.engine`'s and nx.device's (D3); tinygrad has none of these contracts |
| old: unit/test_device.ml "host storage owns zeroed pages suitable for GPU registration", "per-device mappings share base ownership and release before storage", "opaque storage access requires a type identity", "BUFFER owns storage across execution contexts", the three "serialization" tests, "buffer byte ranges reject overflow", "empty storage never calls an allocator", "failed view allocation preserves ownership", "stale views refresh on every storage access", "external views refresh without freeing their owner" | allocators, views and storage | dropped: allocators are nx.device's and the lazy `Buffer` `tolk.next.engine`'s (D3) |

### Rows other sections left to Device's

| Source | Behaviour | Outcome |
|---|---|---|
| Helpers: tinygrad null/test_device.py::TestDevVar::test_dev_arch_override, TestDevice::test_nonexistent_renderer | | this section's rows for the same tests |
| Helpers: tinygrad null/test_device.py::TestCompiler (3 tests); old unit/test_diskcache.ml:319, :322 | the compiler cache | Renderer's section: `Compiler` is `Renderer.Compiler` |
| Ops: tinygrad null/test_tensor_uop_mixin.py::TestUOpEmpty::test_empty_like_sharded_to_single_device, test_empty_direct_singleton_tuple_device | a one-device tuple canonicalizes | dropped: devices are named, never canonicalized (D6); rune names them |
| Ops: old unit/uop/test_uop.ml compiled_signature_preserves_slots_and_types, binary_argument_layout, incomplete_program_has_no_binary | | this section's old tolk rows |

## Worker

The suite is `Tolk_next.Worker` (`engine/worker/`), written `WK` below.
`WK › law` states `map f l = List.map f l` over generated functions and
lists under PARALLEL -1, 0, 1, 2, 3 and 8. The other groups pin the budget,
nesting, failure and settings clauses of the interface. Tests make
applications meet, so that they run at once on distinct domains, and count
the applications running at once. `WK › compiling` compiles the Clang kernels
of `Compiler_cpu`'s `kernels.golden` on domains and serially. Every test but
two spawns domains, and OCaml refuses `Unix.fork` in a process that has, so
mutation testing arms each of `worker.ml`'s mutants in a process of its own
(`--arm`). Every group bounds its tests at 30 s, so the four mutants whose
calls never return fail. One mutant survives, equivalent: `j < i` as `<=` in
`fail`, since each index fails at most once.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: engine/worker.py `get_worker_pool` (no test) | `PARALLEL=0` gives no pool, so the caller compiles | `WK › domains › applies every element on the calling domain under › PARALLEL=-1`, `PARALLEL=0`, `PARALLEL=1` |
| tinygrad: engine/worker.py `get_worker_pool` (no test) | a daemon worker makes no pool of its own | `WK › nesting › a call from f while the outer call holds every domain stays on f's`, `calls from f compute List.map and share the budget` |
| tinygrad: engine/worker.py `Pool(PARALLEL.value, ...)` (no test) | at most PARALLEL workers | `WK › domains › runs at most PARALLEL applications at once under`, `shares PARALLEL - 1 domains between concurrent calls`, `works on the calling domain alone when no domain is free`, `holds at most one domain fewer than its elements`, `gives its domains back when it returns or raises`, `computes List.map with PARALLEL above the runtime's domain limit` |
| tinygrad: engine/worker.py `_init_worker`, `_without_main`, `_spawnv_passfds`, `BEAM_MAX_TASKS_PER_CHILD`, `terminate_worker_pool` (no test) | worker processes' context, SIGINT, recycling and teardown | dropped: compilation workers are domains that live for one call (D5) |
| tinygrad: engine/realize.py:245-255, codegen/opt/search.py:116-168 (`pool.imap_unordered` over indexed tasks) | results placed by task index | `WK › law › map f l is List.map f l`, `keeps the order of l when later elements finish first`, `applies f once to each element under`; `WK › compiling › compiles Clang kernels to the binaries of a serial compilation` |
| tinygrad: a worker process's `Context` (helpers.py:169-186) | a worker's overrides stay in its process | `WK › settings › a setting bound in an application is seen by it alone`, `applications building nodes keep their own CHECK_OOB` (D5) |
| tinygrad: null/test_schedule.py:2076, null/test_viz.py:46 (`PARALLEL=0`) | their suites compile in process | dropped: they pin nothing of the pool |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/test_runtime_search.ml parallel_failure_joins_workers (`Stack_overflow`) | started workers finish before the failure propagates; workers see the caller's settings | `WK › failure › raises only after the applications it started have ended`, `raises the exception of the lowest failing element`; `WK › settings › every application sees the caller's settings` |
| old: unit/test_runtime_search.ml parallel_failure_joins_workers (`Sys.Break`), sequential_compile_interrupt | an interruption propagates, and stops compilation | `WK › failure › stops applying f once an element raised, under › PARALLEL=0`, `PARALLEL=4`; an interruption is an exception like any other |
| old: unit/test_runtime_search.ml completed_compile_budget (2 tests) | over-budget compilations are discarded before timing | dropped here: the beam search's time budget is the search's, not the pool's |
| old: unit/engine/test_realize.ml "positive scopes reuse shared admission and zero compiles inline" | the caller's settings reach workers; concurrent batches share the budget; `PARALLEL=0` compiles on the caller | `WK › settings › every application sees the caller's settings`, `concurrent callers each pass their own settings`; `WK › domains › shares PARALLEL - 1 domains between concurrent calls`, `applies every element on the calling domain under`. That the first batch fixes the limit for later ones is dropped: each call reads its caller's PARALLEL |
| old: unit/engine/test_realize.ml "parallel lowering retains call order, deduplicates, and permits nested batches" | order kept; nested batches make progress | `WK › law › keeps the order of l when later elements finish first`; `WK › nesting › calls from f compute List.map and share the budget`. Deduplication is the lowering's |
| old: unit/engine/test_realize.ml "beam lowering stays in the caller" | no pool under BEAM | dropped here: realize.py chooses not to use the pool, which is the lowering's |
| old: lib/engine/worker.mli (no test) | "The first parallel batch fixes the shared admission limit from PARALLEL" | dropped: each call reads its caller's PARALLEL |

## Multi

The suite is `Tolk_next.Multi` (`schedule/multi/`), written `MU` below.
`MU › multi_pm › recorded › programs` holds what tinygrad's `multi_pm` makes of
the graph that scheduling a `Tensor` program on 2, 4 or 8 CPU devices hands it
(`<program>_multi.golden`), and `› kernels` what it makes of kernels sharded
across a workgroup's threads. `MU › multi_pm › values` states, on each
recorded program `Tensors` can run, that the rewrite keeps what the program
writes, and `MU › multi_pm › laws` that a generated sharded value, after a few
operations, holds on each device the value computed whole; its values draw
grids of 2 by 2 and 2 by 4 devices. The other groups state one rule each.
`test_multi_early` runs under `LATE_ALLREDUCE=0`, which the library reads once,
in a process of its own, so that each branch of that setting is tested by the
process that takes it. Every mutant of `multi.ml` the suites reach is killed;
the setting's own check runs at initialisation, outside every test, and each
of its branches fails one process's tests.

D21 is pinned by `MU › multi_pm › two sharded axes ›` "a reshape divides each
sharded axis by its own count (D21)" and "a reshape of a mesh of 2 by 4 devices
keeps each tile (D21)"; D23 by `MU › multi_pm › shard selections › a selection
of a movement by the device range takes the selected device's position (D23)`
and by the law's first example.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_allreduce.py::TestAllreduceCast::test_allreduce_cast_half, test_allreduce_cast_bf16 | a sum of a half or bfloat16 cast up crosses the devices in 16 bits, and in 32 without `ALLREDUCE_CAST` | `MU › multi_pm › recorded › programs › allreduce_cast_multi.golden`, `allreduce_no_cast_multi.golden`, `allreduce_cast_bfloat16_multi.golden`; `MU › multi_pm › reductions ›` "a value cast up from a half crosses the devices as a half", "… from a bfloat16 …", "without allreduce_cast, it crosses in the type it is reduced in" |
| tinygrad: null/test_allreduce.py::TestAllreduceCast::test_allreduce_cast_float32_noop | | `MU › … › allreduce_cast_float_multi.golden` |
| tinygrad: schedule/multi.py:24 `lower_broadcast_copy` on a selected shard of a product with zero (no upstream test) | tinygrad HEAD raises "mselect must be on tuple device, getting None" when a product with zero, then a cast or a reduction, is selected, since simplifying the copy of the shard folds it to a constant | D35: the shard is the constant; `MU › multi_pm › laws ›` "a shard of a product with zero cast to a float is zero (D35)", "a shard of a sum of a product with zero is zero (D35)", and the law over every drawn program |
| tinygrad: null/test_multitensor.py::TestMultiRamUsage (13 tests) | bytes each device holds | dropped here: memory is Memory's section and the executor's |
| tinygrad: null/test_multitensor.py::TestMultiScalarALU::test_multi_times_replicated_scalar, test_multi_add_replicated_scalar | a sharded value times a scalar on every device stays sharded | `MU › … › add_replicated_scalar_multi.golden`, its value; `MU › multi_pm › arithmetic › a scalar source is kept as it is` |
| tinygrad: null/test_multitensor.py::TestMultiScalarALU::test_multi_times_call_scalar | a per-device scalar from a call | `MU › multi_pm › stores and calls › a call of a compiled function passes its arguments' shards`; the value is the executor's |
| tinygrad: null/test_multitensor.py::TestMultiAxis::test_reshape_shard_invalid, TestMultiTensor::test_shard_reshape_cross_boundary | a reshape that moves elements between shards raises | `MU › multi_pm › movements › a reshape that moves elements between shards is refused` |
| tinygrad: null/test_multitensor.py::TestMultiAxis::test_reshape_shard_valid | | `MU › multi_pm › movements › a reshape keeps a sharded axis whole`; `reshape_split_multi.golden`, `reshape_inner_multi.golden` |
| tinygrad: null/test_multitensor.py::TestMultiAxis::test_uop_shard_axis_none, test_empty_like_sharded, test_symbolic_reshape_shard_axis; TestMultiTensor::test_shard_like, test_shard_not_multiple | `UOp.axis`, `shard`, `empty_like` | dropped here: `Ops`' constructors and properties, Ops' section |
| tinygrad: null/test_multitensor.py::TestMultiTensor::test_bn_ast_on_devices; TestBatchNorm::test_synced_vs_unsynced_bn | a batch norm on 4 devices | `MU › … › batchnorm_stats_multi.golden`, `batchnorm_stats_1_multi.golden`, their values; one kernel per device is Rangeify's |
| tinygrad: null/test_multitensor.py::TestMultiTensor::test_init_rand_with_multiple_devices_fail, test_rand_like_* (5 tests) | | dropped: `Tensor` surface, the frontend is nx |
| tinygrad: null/test_multitensor.py::TestBackendMultiTensor::test_shard_invalids_contiguous | | `MU › … › shard_invalids_multi.golden`; its kernel count is Rangeify's |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_to, test_shard, test_shard_empty, test_shard_same_device, test_numpy, test_tensor_from_multi | | `MU › … › shard_multi.golden`, `replicate_multi.golden`, `gather_multi.golden`, their values; `MU › multi_pm › copies`; the rest is the `Tensor` surface |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_four_add, test_elementwise_dtype, test_shard_elementwise, test_simple_add, test_simple_add_X, test_simple_add_W, test_simple_add_XW, test_alu_deviceless_const, test_add_rank_expand_shard | | `MU › … › add_multi.golden`, `add_four_multi.golden`, `add_eight_multi.golden`, `add_whole_multi.golden`, `add_broadcast_multi.golden`, `cast_half_multi.golden`, `arange_multi.golden`, `expand_multi.golden`, their values; `MU › multi_pm › arithmetic`; `MU › multi_pm › laws` |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_shrink_on_shard_axis, test_const_like_shrink_on_shard_axis, test_arange_shrink | | `MU › … › shrink_one_shard_multi.golden`, its value; `MU › multi_pm › movements › a shrink to one device's shard places that shard on every device`; the laws' `select shard` step |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_simple_reduce, test_shard_reduce, test_shard_plus_one_sum, test_shard_plus_one_sum_d0, test_allreduce_shard_ring_sum | | `MU › … › sum_sharded_axis_multi.golden`, `sum_other_axis_multi.golden`, `sum_all_multi.golden`, `max_sharded_axis_multi.golden`, their values; `MU › multi_pm › reductions`; the ring is Allreduce's section |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_stack | | `MU › … › stack_multi.golden`; `MU › multi_pm › arithmetic › a stack of values sharded alike stacks their shards` |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_allreduce_naive, test_allreduce_ring, test_allreduce_all2all, test_fuzz_allreduce | an allreduce of a sharded value | `MU › … › explicit_allreduce_multi.golden`; `MU › multi_pm › reductions › an allreduce of a sharded value reduces its shards`; the algorithms are Allreduce's section |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_allreduce_cast_half, test_allreduce_cast_half_assign | | `MU › … › allreduce_cast_multi.golden`; the kernel counts are Rangeify's |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_multiple_to_single_device, test_to_single_device_gather_memory | | `MU › … › gather_multi.golden`, `gather_four_multi.golden`, `select_first_multi.golden`, their values; `MU › multi_pm › copies`; memory is Memory's section |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_matmul_shard_none, _X_0, _X_1, _W_0, _W_1, _0_0, _0_1, _1_0, _1_1 (9 tests), test_double_matmul_shard_* (4 tests) | | `MU › … › matmul_rows_multi.golden` (X 0), `matmul_columns_multi.golden` (W 1), `matmul_contracted_multi.golden` (1, 0), `matmul_resharded_multi.golden` (0, 0), `double_matmul_multi.golden`, their values; the laws |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_conv_data_shard, test_conv_bias_data_shard | | `MU › … › conv_multi.golden`, its value |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_embedding, test_rmsnorm, test_sdpa_causal_shard_batch | | `MU › … › embedding_multi.golden`, `rmsnorm_multi.golden`, `attention_multi.golden`, their values |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_flip, test_reshape_on_axis, test_shard_reshape | | `MU › … › flip_multi.golden`, `reshape_split_multi.golden`, `reshape_inner_multi.golden`, their values; `MU › multi_pm › movements` |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_gradient, test_backprop_conv, test_backprop_conv_wino, test_backward_sum, test_embedding_backward, test_embedding_backward_shard_weight, test_lr_scheduler_OneCycleLR | | dropped: gradients, tolk.next does not differentiate |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_shard_no_recompile, test_shard_beam, test_copy_jit, test_allreduce_*_jit, test_multitensor_jit_*, test_multi_tensor_jit_*, test_data_parallel_* | | dropped here: the captured jit (L8) and the end-to-end suite |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_assign_kv_cache_multi, test_mlb_assign_change_axis, test_clone | | `MU › … › assign_multi.golden`, `assign_shard_multi.golden`, `variable_shrink_multi.golden`, their values; the jit is L8's |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_symbolic_broadcast_copy, test_symbolic_broadcast_consumed | | `MU › multi_pm › copies › a copy of a value on one device to several is a copy to each`; `variable_shrink_multi.golden` |
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_rand_on_multiple_devices*, test_rand_like_on_shard*, test_full_like_on_shard*, test_dropout_on_shard*, test_shard_memory, test_multi_const_folding; TestHandleData; TestMultiFromUnrenderable | | dropped: `Tensor` surface and the executor |
| tinygrad: runtime/test_multitensor.py::TestMultiBufferView::test_shrink_2d, test_reshape_then_shrink, test_chained_shrink, test_4_devices | a view of a sharded buffer | `MU › … › shrink_rows_multi.golden`, `reshape_then_shrink_multi.golden`, `shrink_chained_multi.golden`, `shrink_element_multi.golden`, their values; that the view needs no kernel is the scheduler's |
| tinygrad: runtime/test_multitensor.py::Test2DShard (6 tests) | two axes sharded as a grid of devices | `MU › … › grid_add_multi.golden`, `grid_sum_all_multi.golden`, `grid_sum_other_axis_multi.golden`, `grid_matmul_multi.golden`, `grid_to_one_multi.golden`, their values; the laws draw grids |
| tinygrad: runtime/test_multitensor.py::TestShrinkMultiTensorShardedAxis::test_shrink_bad_args | | `MU › multi_pm › movements › a shrink of part of a sharded axis is refused`, `a shrink of another axis shrinks each shard` |
| tinygrad: runtime/test_multitensor.py::TestShrinkMultiTensorShardedAxis::test_ops, test_add_two_partitions, test_add_different_tensors | operations on one device's shard | `MU › … › add_two_partitions_multi.golden`, its value; the laws' `select shard` step followed by operations |
| tinygrad: runtime/test_multitensor.py::TestBatchNorm (4 tests) | | `batchnorm_stats`; the backward passes are dropped, as above |
| tinygrad: runtime/test_multitensor.py::TestMultiSetitem::test_setitem_scalar_axis0, test_setitem_slice_cross_shard, test_setitem_full_slice, test_setitem_stride, test_setitem_single_shard, test_setitem_tensor_value_replicated, test_setitem_tensor_value_sharded_aligned | | `MU › … › setitem_row_multi.golden`, `setitem_rows_multi.golden`, `setitem_columns_multi.golden`, `setitem_stride_multi.golden`, `setitem_replicated_value_multi.golden`, `setitem_sharded_value_multi.golden`, their values |
| tinygrad: runtime/test_multitensor.py::TestMultiSetitem::test_setitem_scalar_axis_none | a replicated value holds no shard | dropped: no rule of `multi_pm` applies |
| tinygrad: runtime/test_multitensor.py::TestTensorOps::test_interpolate, test_bitcast | | `MU › … › interpolate_multi.golden`, `bitcast_multi.golden`, their values |
| tinygrad: runtime/test_multitensor.py::TestMultiTransformer | | dropped here: the end-to-end suite |
| tinygrad: runtime/test_custom_kernel.py::TestUnshardIndex::test_contiguous_fragment_index, test_strided_fragment_index | an index into a thread's rows | `MU › multi_pm › recorded › kernels › fragment_blocks_multi.golden`, `fragment_strided_multi.golden`; `MU › multi_pm › fragments` |
| tinygrad: runtime/test_custom_kernel.py::TestUnshardIndex::test_fragment_index_cannot_shard | | `MU › multi_pm › fragments › an index into rows another thread holds is refused` |
| tinygrad: runtime/test_custom_kernel.py::TestUnshardAlu (2 tests) | | `MU › … › kernels › alu_scalar_multi.golden`, `alu_whole_multi.golden` |
| tinygrad: runtime/test_custom_kernel.py::TestUnshardStore::test_store_unshard_value, test_store_unshard_value_2axis, test_store_load_reg_fragment | | `MU › … › kernels › store_value_multi.golden`, `store_value_two_axes_multi.golden`, `store_load_multi.golden`; `MU › multi_pm › stores and calls` |
| tinygrad: runtime/test_custom_kernel.py::TestUnshardStore::test_store_load_local_fragment | an expected failure at run time | dropped: what fails is the kernel's execution; its rewrite is `store_load`'s |
| tinygrad: runtime/test_custom_kernel.py::test_simple_sharded, test_sharded_add_one; runtime/test_function.py multi-device custom kernels | a custom kernel on shards | `MU › … › custom_kernel_multi.golden`, `inline_function_multi.golden`; `MU › multi_pm › stores and calls › a call of a compiled function passes its arguments' shards` |

### old tolk: unit/engine/test_multi.ml, unit/engine/test_collectives.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_multi.ml "flat sharded params allocate only their local maximum", "axes sort with ranges and close their scope" | `max_shape`, `sharding` and `ranges` of an Unshard | dropped here: `Ops`' properties, Ops' section |
| old: unit/engine/test_multi.ml "reshape divides each axis by its own range count" | | `MU › multi_pm › two sharded axes › a reshape divides each sharded axis by its own count (D21)` |
| old: unit/engine/test_multi.ml "multi-axis ALU slices whole tiles locally" | | `MU › multi_pm › two sharded axes › an operation with a whole value takes its tile of it` |
| old: unit/engine/test_multi.ml "an operand of lower rank keeps its axis where it broadcasts" | `axis` of a broadcast operation | dropped here: `Ops.axis`, Ops' section |
| old: unit/engine/test_multi.ml "permutation keeps the owning range with its axis" | | `MU › multi_pm › two sharded axes › a permute keeps each range with its axis` |
| old: unit/engine/test_multi.ml "own-shard shrink resolves one axis at a time" | | `MU › multi_pm › two sharded axes › a shrink to one axis's own shard keeps the other's sharding`; `MU › multi_pm › fragments › a shrink to a thread's own shard removes its sharding` |
| old: unit/engine/test_multi.ml "thread indices resolve only their owned shard" | | `MU › multi_pm › fragments` (3 tests) |
| old: unit/engine/test_multi.ml "unsharded stores select each fragment's destination" | | `MU › multi_pm › stores and calls › a store of a sharded value into a whole one stores into its part`; `store_value_multi.golden` |
| old: unit/engine/test_multi.ml "two-axis device gather preserves every tile" | | `MU › … › grid_to_one_multi.golden`, its value; the laws draw grids |
| old: unit/engine/test_multi.ml "partial multi-axis allreduce is rejected" | | `MU › multi_pm › reductions › a reduction of some sharded axes but not all is refused` |
| old: unit/engine/test_multi.ml group "Resolution" (2 tests) | resolving Mstack and Mselect to device buffers | dropped: rune's executor |
| old: unit/engine/test_multi.ml group "Execution" (7 tests) | shard, gather, elementwise, broadcast and reductions on 2 and 4 devices | `MU › multi_pm › values` (`shard`, `gather`, `add`, `add_broadcast`, `sum_sharded_axis`, `max_sharded_axis`, `sum_other_axis`); `MU › multi_pm › laws` |
| old: unit/engine/test_multi.ml group "Kernels over split storage" (7 tests) | old tolk's `scatter_indexed` and `block_matmul` on split storage | dropped: old tolk's frontend operations; an index that crosses shards is `MU › multi_pm › fragments › an index into rows another thread holds is refused` |
| old: unit/engine/test_multi.ml group "Cuda" | | dropped: hardware and the executor |
| old: unit/engine/test_collectives.ml group "copies" | copies of rows of a staged value | `MU › … › shrink_one_shard_multi.golden`, `select_first_multi.golden`; the bytes moved are Memory's and the executor's |
| old: unit/engine/test_collectives.ml group "all-gather" | gathers of a sharded value, on one axis and on a grid | `MU › … › gather_multi.golden`, `gather_four_multi.golden`, `grid_to_one_multi.golden`, their values; that a gather is one call and the bytes each device holds are dropped: tinygrad has no all-gather call |
| old: unit/engine/test_collectives.ml group "reduce-scatter", "an allreduce also used whole is reduced once" | resharding | `MU › … › add_resharded_multi.golden`, `matmul_resharded_multi.golden`, `reshard_devices_multi.golden`, their values; `MU › multi_pm › arithmetic › sources sharded differently are resharded on the result's axis`; tinygrad has no reduce-scatter call |
| old: unit/engine/test_collectives.ml "float16 partials reduce-scatter in float16 under ALLREDUCE_CAST" | | `MU › multi_pm › reductions › a value cast up from a half crosses the devices as a half` |
| old: unit/engine/test_collectives.ml "an allreduce in a call body raises" | | dropped: tinygrad expands an allreduce in a call body; `MU › … › inline_function_multi.golden` rewrites a body |

## Prepare

The suite is `Tolk_next.Prepare` (`schedule/prepare/`), written `PR` below.
`PR › prepare_rangeify › recorded` holds what tinygrad's `prepare_rangeify`
makes of the graph that scheduling a `Tensor` program on the CPU hands it, and
of hand-built graphs for the rules no program reaches
(`<name>_prepared.golden`), compared up to the numbers of the storage it makes
(`Uops.numbered_like`). `PR › prepare_rangeify › values` states that each
evaluable graph writes the same into its parameters and buffers before and
after (`Tensors`); call-local storage is scratch. `PR › pm_mops › laws` states
that an index of a chain of movements, rewritten, reads the element they place
there, and `PR › contiguous_view › laws` that a view found is exactly a run of
its storage and that reshapes, shrinks and permutes are found exactly when they
are one. The other groups state one rule each.

Three mutants of `prepare.ml` survive, all equivalent: `op item = After && op s
= Store` as `||` (an item that is not an after of a store is kept whichever
source is taken), `s > 1` as `>=` in the split's axis ranges (an axis of one
element has no divisor from 8 to 256, so whether it counts as broadcast never
matters), and `ns > os` as `>=` in the bitcast expansion (equal sizes return
before). `PR › pm_mops › pads` states `Indexing.apply_movement_op`'s
contract that the pm_mops law leaves out: the validity a pad gives an index
lasts only while the index carries it, and a caller masks padded values.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: runtime/test_assign.py::TestAssign::test_post_permuted_assignment, test_post_flipped_assignment, test_post_flipped_assignment_axis1, test_post_reshape_assignment | a value that reads its destination through a permute or flip is materialised first; through a reshape it is not | `PR › prepare_rangeify › recorded ›` `assign_permuted_self_prepared.golden`, `assign_flipped_self_prepared.golden`, `assign_reshaped_self_prepared.golden`, their values; `hazard_behind_other_after_prepared.golden`, `flip_of_other_prepared.golden` |
| tinygrad: runtime/test_assign.py::TestAssign::test_overlapping_shrink_assignment_forward, test_overlapping_shrink_assignment_reverse, test_nonoverlapping_shrink_assignment | a shrunk destination read through another shrink is materialised | `assign_shifted_self_prepared.golden`, `assign_disjoint_self_prepared.golden`, `assign_shrunk_self_prepared.golden`, `assign_shrunk_in_place_prepared.golden` (the destination's own shrink is safe), their values |
| tinygrad: runtime/test_assign.py::TestAssign::test_assign_bitcast, test_assign_bitcast_unrealized, test_assign_double_bitcast, test_assign_shrink_then_bitcast, test_assign_bitcast_different_size | a store into a bitcast stores the value bitcast | `assign_bitcast_prepared.golden`, `assign_double_bitcast_prepared.golden`, `assign_shrink_then_bitcast_prepared.golden`, `assign_bitcast_wider_prepared.golden`, their values; `PR › prepare_rangeify › earliest rewrites › a store into a bitcast of storage stores the value bitcast` |
| tinygrad: runtime/test_assign.py::TestAssign::test_assign_cross_device, test_assign_temporary_copy_reshape | a store across devices is the copy | `assign_cross_device_prepared.golden`, `copy_into_storage_prepared.golden`, their values; `PR › … › a copy to another device than its destination's is stored first` |
| tinygrad: runtime/test_assign.py::TestAssign::test_assign_shape_broadcast, test_assign_shape_broadcast_2d | | `assign_broadcast_prepared.golden`, its value |
| tinygrad: runtime/test_assign.py::TestAssign::test_disk_assignment | | `assign_to_disk_1_prepared.golden` |
| tinygrad: runtime/test_assign.py::TestAssign::test_assign_deviceless_const | | `assign_deviceless_const_prepared.golden`, its value |
| tinygrad: runtime/test_assign.py::TestAssign::test_nested_after_contiguous_store, test_nested_after_contiguous_store_no_init | a store of a storage's own contents into itself | `assign_own_contents_prepared.golden`; `PR › … › the second of two equal stores into one storage is dropped` |
| tinygrad: runtime/test_assign.py::TestAssign::test_assign_to_function_output, test_nested_function_assign | | `assign_to_function_output_prepared.golden`, `inline_function_prepared.golden`, their values |
| tinygrad: runtime/test_assign.py, the other tests of TestAssign, and TestAssignOrdering, TestAssignToUnrealizedView, TestPartialAssignToSharedBuffer, TestAfterCachePatterns | the order of stores and reads, and kernel counts | dropped here: the order is the schedule's, the counts Rangeify's section; `assign_twice_prepared.golden`, `store_ordered_before_prepared.golden`, `placed_after_read_prepared.golden` hold the preparation of such graphs |
| tinygrad: runtime/test_assign.py::TestMultiAssign (12 tests) | stores into sharded values | dropped here: Multi's section |
| tinygrad: null/test_assign.py (4 tests), null/test_setitem_schedule.py (3 tests) | kernel counts | dropped here: Rangeify's section; `setitem_prepared.golden`, `setitem_tensor_prepared.golden` hold the preparation |
| tinygrad: null/test_schedule.py::TestSchedule::test_zero_size, test_zero_size_alt, test_zero_size_assign, test_zero_size_children | empty values | `sum_of_nothing_prepared.golden`, `max_of_nothing_prepared.golden`, `add_nothing_prepared.golden`; `PR › … › a value with an empty axis is zero` |
| tinygrad: null/test_schedule.py::TestSchedule::test_detach_assign, test_contiguous_backward_assign | | `detach_prepared.golden`, `contiguous_backward_prepared.golden`; `PR › … › a detach and a gradient marker are their source` |
| tinygrad: null/test_schedule.py::TestSchedule::test_dedup_assign | | `PR › … › the second of two equal stores into one storage is dropped` |
| tinygrad: null/test_schedule.py::TestSchedule::test_reduce_doesnt_split and the split tests | large reductions over few outputs split in two | `split_sum_prepared.golden`, `split_rows_prepared.golden`, `split_max_prepared.golden`, `split_expanded_prepared.golden` (a broadcast axis is not split), `split_unit_axis_prepared.golden`, `split_prime_prepared.golden` (no divisor), `split_at_threshold_prepared.golden`, `below_threshold_prepared.golden`, `no_split_prepared.golden`; `PR › … ›` "a split reduction keeps its first reduction's output within 2^22 elements", "a split takes the largest divisor from 256 down", "a split is announced at debug level 3", "a reduction of a symbolic shape is not split" |
| tinygrad: null/test_schedule.py::TestSchedule::test_contiguous_buffer, test_double_contiguous_realizes_once, test_bitcast_fuses; TestCopyFolding (3 tests) | | `contiguous_of_storage_prepared.golden`, `bitcast_*_prepared.golden`, `copy_*_prepared.golden`, `clone_prepared.golden`; `PR › … › a materialisation of storage is the storage`; their kernel counts are Rangeify's |
| tinygrad: schedule/prepare.py `pm_disk_copy` (no upstream test) | a disk copy reads its view without materialising it | `copy_from_disk_prepared.golden`, `copy_permuted_from_disk_prepared.golden`, `copy_staged_view_from_disk_prepared.golden`, `disk_staged_view_prepared.golden`, `bitcast_on_disk_prepared.golden`; `PR › … › a bitcast on a disk keeps its size` |
| tinygrad: runtime/test_custom_kernel.py, runtime/test_function.py (inline and precompiled functions) | calls inlined or compiled on their own | `custom_kernel_prepared.golden`, `inline_function_prepared.golden`, `inline_sharded_prepared.golden`, `inline_symbolic_prepared.golden`, `precompiled_function_prepared.golden`; `PR › prepare_rangeify › inline calls` (7 tests) |
| tinygrad: schedule/prepare.py `pm_mops` on a padded view (no upstream test) | an index through a pad, then a reshape to an axis of one element, an expand, or a shrink onto an axis of one element, loses the pad's gate | `PR › pm_mops › pads` (3 tests), as tinygrad HEAD for an upstream report: `graph_rewrite(p._mop(Ops.RESHAPE,(1,)).pad(((0,1),)).index(r), pm_mops)` with `p` a 1-element param is `INDEX(p, 0)`; a 6-element param reshaped (3,2), padded `((0,3),(2,6))` and shrunk `((0,1),(0,3))`, indexed by two ranges, is `INDEX(p, 0)`; a 2-element param expanded by `(2,)` and padded `((1,3),(0,2))` is `INDEX(p, r1)` |
| tinygrad: schedule/prepare.py `pm_fold_moved_after`, `OPENPILOT_HACKS`, `FLOAT16` | | dropped: excluded (README) |
| tinygrad: uop/ops.py `contiguous_view` on CL and WEBGPU | no view | dropped: excluded (README) |
| tinygrad: uop/ops.py:934 `contiguous_view` on flips that cancel across a reshape (no upstream test) | a run it does not find: tinygrad HEAD gives `None` for a `(2, 1, 16)` view flipped on axis 0, reshaped to `(2, 16)` and flipped on axis 0 again, which is the identity | `PR › contiguous_view › flips that cancel across a reshape are not found` |
| tinygrad: uop/ops.py:934 `contiguous_view` on a read within one copy of an expanded value (no upstream test) | a run it does not find: tinygrad HEAD gives `None` for a `(1, 2, 4, 2)` view expanded to `(2, 2, 4, 2)`, reshaped to `(32,)` and shrunk to `(6, 9)`, elements 6 to 8 of the storage | `PR › contiguous_view › a read within one copy of an expanded value is not found` |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/test_contiguous_view.ml "cancelling movement chains prove a contiguous view" | | `PR › contiguous_view › two permutes that cancel are a view`; the law |
| old: unit/test_contiguous_view.ml "prepend EXPAND respects contiguous storage" | | `PR › contiguous_view ›` "an expand of an axis of one element is a view", "an expand repeats" |
| old: unit/test_contiguous_view.ml "view anchors retain pending effects" | | `PR › contiguous_view › a view of storage after its stores is a view of the ordered storage` |
| old: unit/test_contiguous_view.ml "subword byte offsets retain typed anchors", "one-element bitcast views preserve byte extent" | | `PR › contiguous_view ›` "bytes within an element are a view of the bytes", "bytes on whole elements start at their element", "a bitcast of a view with an axis of one element is a view" |
| old: unit/test_contiguous_view.ml "empty views preserve offsets and storage anchors" | an empty view | `PR › contiguous_view › an empty view is no view`, as tinygrad |
| old: unit/test_contiguous_view.ml "symbolic leading views compose their flattened index" | a symbolic view at an offset | `PR › contiguous_view › a view of a symbolic size is no view`, as tinygrad |
| old: unit/test_contiguous_view.ml "existing movement views preserve byte offsets", "view offsets use exact arithmetic before host narrowing", "caller tags are preserved without certifying strided views", "constant folding preserves tensor shape during view proofs" | | the laws `PR › contiguous_view › laws`; offsets count elements of the storage's type, as tinygrad's |
| old: unit/test_contiguous_view.ml "unsupported backends reject typed views" | | dropped: CL and WebGPU are excluded (README) |
| old: unit/test_contiguous_view.ml "storage windows retain allocation boundaries", "storage windows retain typed effects", "partition storage requires owned lanes", "partial reshape compares symbolic suffix dimensions", "partial reshape uses symbolic prefix dimensions" | old tolk's `Indexing.storage_window` and partial reshapes | the partial reshape is `PR › pm_mops › rules ›` "an index of a reshape's leading axes indexes its source when the trailing axes are kept", "an index of a reshape whose trailing axes change is left as it is"; storage windows were old tolk's, tinygrad has none |
| old: unit/test_schedule_rangeify.ml group "early_movement_pass" | | `PR › pm_mops › rules` (7 tests), `PR › pm_mops › laws` |
| old: unit/test_schedule_rangeify.ml group "split_reduce" | | the split rows above |
| old: unit/test_schedule_rangeify.ml group "symbolic empty shapes" (3 tests) | an empty reduction keeps its identity | `sum_of_nothing_prepared.golden`, `max_of_nothing_prepared.golden`; symbolic empty shapes are dropped: tinygrad's rule reads a concrete zero |
| old: unit/test_schedule_rangeify.ml group "moved materializations" (3 tests) | | dropped: openpilot's `pm_fold_moved_after`, excluded (README) |
| old: unit/test_schedule_rangeify.ml group "symbolic storage views" (2 tests) | | `variable_shrink_prepared.golden`, `inline_symbolic_prepared.golden`; the removal of stages is Rangeify's section |

## Rangeify

The suite is `Tolk_next.Rangeify` (`schedule/rangeify/`), written `RA` below.
`RA › get_kernel_graph › recorded` holds the kernel graph tinygrad's
`get_kernel_graph` makes of the graph that scheduling a `Tensor` program on the
CPU hands it (`<program>_kernels.golden`), and `kernel_counts.golden` its
number of kernels per graph, each a test of its own. `RA › get_kernel_graph ›
values` states, on each single-device graph Tensors can run, that running the
kernels (`Kernel_graphs`, each kernel reading its arguments in the states they
name) writes into the function's parameters what the tensor graph does, and
`RA › get_kernel_graph › laws` the same end to end, through Prepare, on
generated functions. `RA › get_kernel_graph › kernel graphs` states over every
recorded graph that each kernel's storage and ranges are numbered from 0, that
calls pass storage, that new storage has a committed type, and that storage a
kernel writes is read after.

Coverage of `rangeify.ml` is 85%. The code no test reaches is reached by no
tinygrad program either: the stage of storage that effects write
(`bufferize_to_store`'s `After` branch; Indexing always materialises an after,
so none is ever staged), call arguments that are indices or shrinks, the
index of a deviceless gather, and the index through a weak cast. A spy on
tinygrad's `bufferize_to_store` over `test/null`'s `test_schedule`,
`test_assign`, `test_custom_kernel`, `test_setitem_schedule` and
`test_multitensor`, and `test/runtime`'s `test_assign` and
`test_custom_kernel`, never saw a staged after. The code stays as the port of
the file.

Of the mutants the suite reaches, 46 are killed and 15 survive, none of them
observable in a kernel graph of any program or hand-built graph tried:

- `:620`, `1 + max` as `1 - max`, and `:292`, a device range renumbered by the
  buffer limit: the fresh ranges must not collide with live ones, since equal
  ranges are one node, but where they collide (`many_matrices_limited`, whose
  greatest range is 1) the stage they make is indexed by its own ranges and
  the kernel graph is the same. The other mutants of the buffer limit are
  killed by `many_matrices_limited` and `many_cubes_limited`;
- `:15`, `:49`, `:380-381`, `:492`: tinygrad's guards on symbolic ranges and
  its shrinks to symbolic sizes, reached by `variable_*` and `symbolic_*`, whose
  shrinks simplify away;
- `:41`, `:47`, `:50`: the dead-axis cleanup of a stage: run_rangeify already
  stages a value over the ranges it varies along, as `expand_kept`'s buffer of
  4 elements shows with or without it;
- `:121`, the exclusion of constant and invalid indices when a stage is
  inlined; `:143`, the size check of an all-invalid read; `:641`, the spec
  check, which only fails on a graph the pass would build wrongly.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_schedule.py::TestSchedule (the kernel-count tests) | how many kernels a program schedules | `RA › kernel_counts.golden` and `RA › get_kernel_graph › recorded`, on ports: `arange_sum`, `permute_arange`, `expand_before_cast`, `push_pads_elementwise`, `allow_push_permutes`, `div_collapse`, `reduce_same_size`, `reduce_multiple_paths`, `reduce_ext_reduce_child`, `reduce_expand_child`, `reduce_broadcast_not_recomputed`, `ugly_reduceop_pairing`, `reduce_expand_reduce`, `multireduce_parallel`, `std`, `multireduce_diffops_parallel`, `multimatmul`, `multireduce_push_shrink_chase`, `multireduce_midreduce_nochase`, `partial_fuse`, `pad_reduce_safe`, `pad_reduce_unsafe`, `shrink_pad_unsafe`, `base_change_expand_pad`, `base_change_pad_expand`, `zero_size_children`, `preserve_multistage_reduce`, `clone`; their values |
| tinygrad: null/test_schedule.py::TestSchedule, the other tests | | dropped here: the realized-buffer, gradient and jit tests are the `Tensor` surface and L8; the Tensor programs above stand for their kernel counts |
| tinygrad: null/test_schedule.py::TestInvalidTensor::test_full_invalid_is_zero_kernels | | `RA › … › full_invalid_kernels.golden` (no kernel); `RA › get_kernel_graph › rules › a store of invalid values runs no kernel`; `invalids_read`, `partially_invalid`, `invalids_sharded` |
| tinygrad: null/test_schedule.py::TestLimitBufs | the buffers of one kernel | `many_inputs_kernels.golden`, `many_inputs_limited_kernels.golden`, `many_matrices_limited_kernels.golden`, `many_cubes_limited_kernels.golden`, `many_sums_limited_kernels.golden`, `many_sharded_limited_kernels.golden`; `RA › … ›` "a kernel accesses at most max_kernel_buffers storages", "without a limit, one kernel reads every storage"; the scaling test is dropped: a timing |
| tinygrad: null/test_schedule.py::TestCopyFolding | | `copy_kernels.golden`, `clone_kernels.golden`; the copy kernels' lowering is the schedule's |
| tinygrad: null/test_assign.py (4 tests), null/test_setitem_schedule.py (3 tests) | kernel counts of assigns and setitems | `assign`, `assign_permuted`, `assign_double_diamond`, `assigned_read_twice`, `assigned_contiguous`, `stage_of_assigned`, `setitem`, `setitem_tensor`: their kernel graphs, counts and values |
| tinygrad: null/test_multitensor.py::TestBackendMultiTensor::test_shard_invalids_contiguous | one kernel | `invalids_sharded` |
| tinygrad: runtime/test_rangeify.py::test_double_matmul, test_assign_permuted, test_variable_stack_data, test_variable_data_and_shape, test_matmul_relu_cat, test_multi_gather | | `double_matmul`, `assign_permuted`, `variable_*`, `shard_gather`: graphs and counts; the rest of those programs are Indexing's section |
| tinygrad: runtime/test_rangeify.py, the `*_match` tests (7) | kernel structure after codegen | dropped here: the codegen pipeline's (L4) |
| tinygrad: runtime/test_custom_kernel.py, runtime/test_function.py | custom kernels and functions | `custom_kernel`, `custom_kernel_permuted`, `custom_kernel_in_place`, `custom_kernel_of_views`, `inline_function`, `precompiled_function`, `precompiled_function_1`: graphs and counts |
| tinygrad: schedule/rangeify.py `DEBUG_RANGEIFY` | the ranges printed | `RA › get_kernel_graph › debug` (2 tests) |
| tinygrad: schedule/rangeify.py `SPEC` | | `RA › get_kernel_graph › spec › with spec checks on, every recorded kernel graph passes them` |
| tinygrad: schedule/rangeify.py `check_buf_states` | a kernel reading one storage in two states | `RA › get_kernel_graph › rules › a kernel reading one storage in two states is refused` |
| tinygrad: schedule/rangeify.py `bufferize_to_store`'s size assertion | a symbolic stage | `RA › get_kernel_graph › rules › a materialisation of a symbolic size takes its greatest size`: as in tinygrad, the stage's size is its greatest, and nothing raises |
| tinygrad: schedule/rangeify.py `DEVICE_MAX_BUFS` | WebGPU's limit | dropped: excluded (README) |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/test_schedule_rangeify.ml group "get_kernel_graph" `pipeline_test`s (elementwise, mulacc, binop reshape and permute, diamond, double unary, reduce reshape and permute binop, explicit contiguous, push permute through reshape, multistage reduce, children dont push, reduce permute nofuse) | the number of kernels | `RA › kernel_counts.golden ›` `elementwise_three`, `mulacc`, `binop_reshape`, `binop_permute`, `shared_sum`, `reduce_unary`, `reduce_reshape_binop`, `reduce_permute_binop`, `contiguous_add`, `permute_through_reshape`, `multistage_reduce`, `children_dont_push`, `reduce_permute_nofuse`; their graphs and values |
| old: unit/test_schedule_rangeify.ml "rejects distinct written states of one buffer in a kernel" | | `RA › get_kernel_graph › rules › a kernel reading one storage in two states is refused` |
| old: unit/test_schedule_rangeify.ml group "stage capacity" (8 tests) | old tolk's stage forwarding | dropped: tinygrad has no such pass; the stages of recorded programs are the goldens' |
| old: unit/test_schedule_rangeify.ml group "stack selection", "reshape merge" | | Indexing's section |
| old: unit/test_schedule_rangeify.ml group "packed argument buffer limits" (4 tests) | a kernel's buffers on CPU and Metal | `many_inputs`, `many_inputs_limited` and the two rules above; Metal's limit is its renderer's |
| old: unit/test_schedule_rangeify.ml "kernel splitting preserves independent symbolic ranges" | | `variable_offset`, `variable_staged`, `variable_read_twice`, `variable_reduce`, `variable_same`, `variable_two`, `symbolic_kept`, `symbolic_contiguous`: graphs and counts |
| old: unit/test_schedule_scaling.ml group "get_kernel_graph scaling" | time over graph size | dropped: a timing, the benchmarks' |
| old: golden/rangeify (138 cases: 17 programs on 5 renderers, and 5 Llama cases) | rendered kernels | the programs' kernel graphs are `RA › … recorded` (`elementwise_three`, `mulacc`, `binop_reshape`, `binop_permute`, `shared_sum`, `reduce_unary`, `reduce_reshape_binop`, `reduce_permute_binop`, `reduce_shrink`, `shrink_fuse`, `multistage_reduce`, `permute_through_reshape`, `reshape_chain`, `contiguous_add`, `children_dont_push`); the rendering is the renderers' and the end-to-end suite's (L4, L5); Llama is the end-to-end suite's |

### Rows other sections left to Rangeify's

| Source | Behaviour | Outcome |
|---|---|---|
| Ops: tinygrad `null/test_uop_graph.py::TestConstBufferize` (2 tests) | `pm_const_buffer_folding`: a constant staged over one range or several folds to the constant | `RA › get_kernel_graph › recorded ›` `setitem_column_kernels.golden` (the mask of a setitem into a column, staged over one range) and `setitem_cube_kernels.golden` (into a cube, staged over a range and two constant indices); a constant staged over several live ranges, which tinygrad's test builds by hand, never comes out of scheduling a program, so the rule is pinned where tinygrad's own flow reaches it; the dead-axis cleanup runs first there, as in tinygrad's matcher |

## Schedule

The suite is `Tolk_next.Schedule` (`schedule/schedule/`), written `SC` below.
`SC › create_linear_with_vars › recorded` holds, for each `Tensor` program
realized on the CPU, the sink tinygrad hands `create_linear_with_vars`
(`<program>.golden`) and the planned schedule it returns
(`<program>_linear.golden`), compared up to the numbering of the buffers made;
`var_vals.golden` holds the variables' values. `SC › create_schedule ›
recorded` holds each kernel graph tinygrad hands `create_schedule` and the
schedule it returns, and `arguments.golden` holds each graph's scalar
arguments. `SC › create_schedule › values` states that running a schedule's
calls in order (`Kernel_graphs.linear_writes`) writes what the kernel graph
does. `SC › create_linear_with_vars › values` states the same end to end: the
unplanned schedule of a program writes into its buffers what its tensors
compute (Tensors). `SC › create_linear_with_vars › capturing` states that the
schedule left unplanned under `capturing`, once planned with the program's
buffers held, is the default one. `SC › create_linear_with_vars › cache`
covers hits, misses, keys, the `DEBUG` line and domains; the cache tests
schedule programs whose constant no other run used, so they hold in any order
and on any rerun.

The values laws leave out custom kernels (the Interpreter does not compile
them), graphs on several devices, and graphs with calls that are not kernels.
They also leave out `assign_bitcast`: its call passes a buffer and a bitcast of
it as two arguments, one memory that storage keyed by slot cannot alias. The
end-to-end law leaves out programs with variables, whose movements Tensors
does not run.

Coverage of `schedule.ml` is 97.9%. The code no test reaches:

- the defensive refusals of a queued kernel that is not a call and of storage
  without its argument;
- a buffer bound twice in one invocation, and call-local storage without a
  device: rangeify gives every allocation a device, and each is rewritten once;
- a free variable (slot -1) outside a kernel body;
- a call of an already compiled `Program`, which the jit makes;
- a nested call's scalar argument that is not a variable.

The domain tests spawn domains, and OCaml refuses `fork` after that, so mutation
runs with `-e domains`. Of the 72 mutants reached, 69 are killed and 3
survive:

- `:228`, `op x = Param && addrspace x = Some Alu` as `||`: it also binds a
  nested call's buffer arguments by position. That is observable only when a
  nested call inherits an enclosing scalar at a slot where it passes a buffer,
  which no program tried makes (`precompiled_scalar` passes its scalar).
- `:291` and `:552`, the `SPEC` checks turned on by default and off under
  `SPEC=1`: every graph the pipeline builds passes `Spec.tensor`
  (`SC › create_linear_with_vars › spec`), so the checks are inert, as
  Rangeify's `:641` is.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_schedule_cache.py::TestScheduleCache::test_bound_variable_var_vals | a bound variable's value | `SC › var_vals.golden` (`variable_shrink`, `variable_reduce`, `variable_two`, `variable_same`, `variable_offset`, `variable_unused`, `precompiled_scalar`) |
| tinygrad: null/test_schedule_cache.py::TestScheduleCache::test_disable_schedule_cache | `SCACHE=0` neither writes nor reads the cache | `SC › create_linear_with_vars › cache ›` "with SCACHE=0 every schedule misses", "a body scheduled again is a hit under the key it missed on"; the profile-event count is dropped, being the profiler's |
| tinygrad: runtime/test_schedule_cache.py::TestScheduleCache::test_bound_variable_reuses_cache | | `SC › … › cache › a variable bound to another value is the same body` |
| tinygrad: runtime/test_schedule_cache.py::TestScheduleCache::test_simple | a repeated schedule does not grow the cache | `SC › … › cache ›` "a body scheduled again is a hit under the key it missed on", "a hit is the schedule the miss made" |
| tinygrad: runtime/test_schedule_cache.py::TestScheduleCache::test_chained_functions_with_local_allocations_reuse_cache | one body scheduled for three calls | `SC › … › cache › a function called three times is scheduled once` (tinygrad's `DEBUG=3` lines), `chained_functions`; `SC › … › cache › call-local storage in other slots is the same body` |
| tinygrad: runtime/test_schedule_cache.py::TestScheduleCache::test_custom_kernel, test_same_custom_function_reuses_cache | | `custom_kernel`: its schedule and linear; the cache key is structural, as the cache tests state |
| tinygrad: runtime/test_schedule_cache.py::TestScheduleCache::test_simple_precompile | | `precompiled_function`; the backward pass is dropped, being the gradient's (L8) |
| tinygrad: null/test_schedule.py, runtime/test_schedule.py (TestSchedule, TestLimitBufs, TestSwizzle, TestView) | kernel counts and realized values | Rangeify's section for the counts; the realized values are the `Tensor` surface (L8) |
| tinygrad: runtime/test_buffer.py, runtime/test_subbuffer.py | device buffers and their views | dropped here: the device runtime's; the schedule's views are `SC › contiguous_mops_to_view` and `copy_view`, `disk_view_to`, `assign_bitcast` |
| tinygrad: schedule/__init__.py `create_schedule`'s assertions | a cycle, an effect that is not a call, end, store or After, an end of something else, an argument that is not storage | `SC › create_schedule › rules` |
| tinygrad: schedule/__init__.py `assert_all_same_devices` | | `SC › create_linear_with_vars › rules › a kernel on buffers of two devices is refused` |
| tinygrad: schedule/__init__.py `create_linear_with_vars`'s bind mismatch | | `SC › create_linear_with_vars › rules ›` "two variables of one name bound to two values are refused", "… to one value are one" |
| tinygrad: schedule/__init__.py `pm_copy_from_store` | copy kernels become stores | `copy`, `copy_one`, `copy_view`, `disk_to`, `disk_view_to`, `disk_store`: their linears |
| tinygrad: schedule/__init__.py `lower_sink_to_linear`'s `DEBUG` print | | `SC › … › cache ›` "DEBUG=1 prints only schedules of several kernels", "DEBUG=0 prints nothing", "the time printed is the time scheduling took"; the caller's frame and the node count are the README's row on the `DEBUG` print |
| tinygrad: schedule/__init__.py `SCACHE=2` | the disk cache | dropped: the README's row on the disk half of `SCACHE=2` |
| tinygrad: schedule/__init__.py `SPEC` | | `SC › create_linear_with_vars › spec` |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_schedule.ml "cache callbacks can lower another schedule" | lowering a body while another is lowered | `precompiled_function`, `chained_functions`, `precompiled_scalar`: nested bodies scheduled inside their callers' |
| old: unit/engine/test_schedule.ml "partitions AFTER dependencies like tinygrad", "orders a reader before a superseding writer (WAR)", "rejects cycles instead of returning empty or partial schedules" | | `SC › create_schedule › rules` (9 tests) and `SC › create_schedule › values`, with `read_then_overwrite` and `assign_double_diamond` |
| old: unit/engine/test_schedule.ml "nested scalar arguments shadow and inherit lexical bindings" | | shadowing: `precompiled_scalar`; inheriting is the `:228` survivor above |
| old: unit/engine/test_schedule.ml "resolves allocations per invocation while preserving owners", "nested calls own separate anonymous allocations" | | `chained_functions_linear.golden`, `precompiled_function_linear.golden` |
| old: unit/engine/test_schedule.ml "PARAM slots count BIND arguments" | | `SC › arguments.golden` through `SC › create_schedule › values` (`precompiled_scalar_kernels`, `variable_*_kernels`) |
| old: unit/engine/test_schedule.ml "returns only binds used by scheduled kernels" | | `SC › var_vals.golden › program=variable_unused` |
| old: unit/engine/test_schedule.ml "memory-plans internal buffers when not capturing", "hands the unplanned schedule to an active capturer" | | `SC › create_linear_with_vars › capturing` (46 tests); the capturer itself is the jit's (L7) |
| old: unit/engine/test_schedule.ml group "call arguments" (5 tests) | views as call arguments | `SC › contiguous_mops_to_view` (11 tests), `copy_view`, `disk_view_to`, `assign_bitcast`; the refusal of a written view is old tolk's `copy_call`, which tinygrad has no counterpart of |
| old: unit/test_engine_schedule.ml "disk views move after explicit bulk transfers", "ordinary slices keep the base call input", "explicit contiguous views are normalized before their bases" | | `disk_view_to`, `copy_view`, `SC › contiguous_mops_to_view` |
| old: unit/test_engine_schedule.ml "volatile inputs survive scheduling" | | dropped: tinygrad's `param_like` makes a buffer's parameter without its volatile mark (`uop/ops.py:1247`), so a schedule keeps none |
| old: unit/test_engine_schedule.ml "AFTER partition ignores STORE and keeps kernel order", "AFTER dependencies use all producer kernels", "CALL args are resolved through AFTER to buffer uops" | | `SC › create_schedule › rules ›` "a kernel runs after the one it reads and before its overwriter", "a kernel overwriting a state runs after the kernel that made it", "a kernel reading storage before and after a write runs after it" |
| old: unit/test_engine_schedule.ml "schedule cache uses semantic key", "schedule cache key is identical across bind values" | | `SC › … › cache ›` "two bodies miss under two keys", "call-local storage in other slots is the same body", "a variable bound to another value is the same body" |
| old: unit/test_engine_schedule.ml "schedule cache keys on the settings scheduling reads" | | dropped: tinygrad keys a body on its structure alone (`function.key`) |
| old: unit/test_engine_schedule.ml "create_linear_with_vars keeps only used binds", "transform_to_call keeps variable identity on bound PARAM", "create_linear_with_vars extracts the binding from CALL args" | | `SC › var_vals.golden`, `SC › create_linear_with_vars › recorded` (`variable_*`) |
| old: unit/test_engine_schedule.ml "fresh internal buffer slots keep buffers distinct", "concurrent internal slots keep imported buffers distinct", "concurrent memory plans keep arenas distinct" | | `SC › … › cache ›` "domains scheduling one body at once schedule it alike" (each domain makes buffers of its own), "domains scheduling bodies at once schedule each as alone" |

## Realize

`R` is the `Realize` suite (`test/engine/realize`). Its goldens are schedules of
`Tensor` programs realized on the CPU, the schedules `lower_and_compile`
returns, and `calls.golden`, each call's reading by `get_call_*`,
`get_call_name` and `estimate_uop`.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: engine/realize.py `get_call_arg_uops`, `get_call_var_uops`, `get_call_outs_ins`, `get_call_written_bufs`, `get_call_name`, `estimate_uop` | what a call does | `R › calls of recorded schedules › calls.golden` (21 calls of 14 programs), and `R › get_call_arg_uops`, `› get_call_var_uops`, `› get_call_outs_ins`, `› get_call_written_bufs`, `› get_call_name`, `› estimate_uop` (25 tests) |
| tinygrad: engine/realize.py `lower_and_compile` | kernels compiled once, in parallel unless one asks for a beam | `R › lower_and_compile of recorded schedules` (14 goldens), `R › lower_and_compile` (8 tests), `R › lower_and_compile with a beam search`, `R › lower_and_compile in parallel` |
| tinygrad: engine/realize.py `pm_beam`, `compile_linear`'s `BEAM` | a kernel asking for no beam asks for `BEAM`'s | `Hcq2 › compile_linear › makes each kernel ask for a beam of the width BEAM sets` |
| tinygrad: engine/realize.py `get_call_kernels`, `track_stats`, `ExecContext`, `exec_*`, `runtime_cache`, `run_linear`, `time_call`, `link_linear`, `capturing` | running a linear | dropped here: the run half is `tolk.engine`'s (plan §1a); its execution is `Hcq2 › linking and running` |
| tinygrad: engine/realize.py `pm_validate`, `exec_validate`, `VALIDATE_WITH_CPU` | | dropped: excluded (ruling) |
| tinygrad: engine/realize.py `get_call_name`'s `encdec` and `get_call_outs_ins`'s | | dropped: the video decoder queue is excluded |
| tinygrad: null/test_call.py::TestCallCodegen::test_compiled_scalar_slots_are_not_call_slots | a compiled program keeps its own variable slots | `R › get_call_var_uops › is the constant a call binds each variable to, in the program's order` (a program's variables are read by name, whatever the call's slots) |
| tinygrad: null/test_call.py::TestCallCodegen::test_call_stack_pointer | a custom-function call's arguments in C | dropped here: Cstyle's rendering of `CALL` |
| tinygrad: null/test_call.py (TestCall, TestCallShape, TestCallSchedule, TestArgOrder) | call construction and scheduling | dropped here: `Ops`' `call`/`call_with_outputs` and Schedule's sections |
| tinygrad: null/test_method_cache.py (3 tests) | a program compiled once is not compiled again | `R › lower_and_compile › compiles each kernel once`; the cache is `Codegen.to_program`'s (Codegen's section, D5) |
| tinygrad: runtime/test_realize_is_realize.py (13 tests) | `Tensor.realize` marks buffers realized | dropped: the `Tensor` surface; nx is the frontend |
| tinygrad: null/test_custom_kernel.py, runtime/test_custom_kernel.py | custom kernels scheduled and run | `custom_kernel.golden` and its compiled schedule; the kernels' codegen and values are Codegen's, Rangeify's and Cstyle's sections; the realized values are the `Tensor` surface |
| tinygrad: runtime/test_call.py (TestCall, TestCallSchedule, TestArgOrder, TestCallMultiSharded) | precompiled calls realized | `precompiled_scalar.golden`; scheduling is Schedule's section, the values the `Tensor` surface |
| tinygrad: runtime/test_after.py (12 tests) | ordering of assigns and their gradients | dropped here: Schedule's ordering (its section) and gradients (L8) |
| tinygrad: runtime/test_wait_loop.py::TestWaitLoop (7 tests) | loops without bounds, rendered and run | dropped here: Codegen's and Cstyle's (`Ops.loop`, `backedge`) |
| tinygrad: runtime/test_wait_loop.py::TestVolatileLoops::test_async_wait_ext | a kernel spinning on a word another thread writes | dropped: the device runtime's; D1 removed the host program's spin |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_realize.ml group "Renderer selection" (4 tests) | renderer selection and compile caches | dropped here: `Device.renderer` (Device's section) and `Codegen.to_program`'s cache; `R › lower_and_compile › raises Invalid_argument when no renderer serves a call's target` |
| old: unit/engine/test_realize.ml group "Owner cache lifetime" (5 tests) | program owners, runtime graphs, cache retry | dropped: old tolk's owner caches, which tinygrad has no counterpart of (plan, the 100 divergences) |
| old: unit/engine/test_realize.ml "positive scopes reuse shared admission and zero compiles inline" | | dropped: old tolk's worker admission; Worker's section (D5) |
| old: unit/engine/test_realize.ml "parallel lowering retains call order, deduplicates, and permits nested batches" | | `R › lower_and_compile in parallel › compiles each kernel into the program it compiles into alone`, `R › lower_and_compile › compiles each kernel once` |
| old: unit/engine/test_realize.ml "beam lowering stays in the caller" | | `R › lower_and_compile with a beam search › searches each kernel with the width it asks for, in order, one at a time` |
| old: unit/engine/test_realize.ml group "PROGRAM completion" (6 tests) | a program partly compiled is completed, a complete one kept | `R › lower_and_compile ›` "compiles a program not yet compiled", "leaves a compiled program alone"; the rest is `Codegen.to_program`'s (its section) |
| old: unit/engine/test_realize.ml group "Scoped capture" (5 tests) | | dropped: the captured jit (L8) |
| old: unit/engine/test_realize.ml "multi-owner templates retire…", "concurrent submissions retain their own address tables", "independent links serialize reservations…", "submission addresses are resolved after preparation", "submission scope permits reentry…", "failed preparation leaves the address table unchanged…", "retained queue replay rejects replaced device owners…" | submissions of a linked batch | `Hcq2 › linking and running ›` "a run waits for its batch's previous run before it rewrites the batch's memory", "a link serves any buffers bound to its inputs"; the owners and their failures are old tolk's contracts, dropped |
| old: unit/engine/test_realize.ml "concurrent execution records every cost and timing", "execution counters retain exact large costs" | | dropped: `GlobalCounters` belong to the run half (`tolk.engine`) |
| old: unit/engine/test_realize.ml "compilation resolves beam context once and respects explicit zero" | | `Hcq2 › compile_linear › makes each kernel ask for a beam of the width BEAM sets` |
| old: unit/engine/test_realize.ml "timing samples…", "failed timing dispatch…", "failed timing drain…" | | dropped: `time_call` is the engine's `measure` |
| old: unit/engine/test_realize.ml "compiled launch uses fixed workgroups", "passes every scalar from program metadata", "requires scalar variables" | | `Hcq2 › linking and running › a schedule's variables reach its kernels`; launch sizes are `Ops.launch_dims` (Ops' section) |
| old: unit/engine/test_realize.ml group "Program cache" (5 tests) | cache keys of programs | dropped here: `Codegen.to_program`'s cache (its section, D5) |
| old: unit/engine/test_realize.ml "rewrites CALL(SINK) to CALL(PROGRAM) with source and binary" | | `R › lower_and_compile › makes each call of a kernel a call of its program`, `R › lower_and_compile of recorded schedules` |
| old: unit/engine/test_realize.ml group "Buffer copy" (5 tests) | copies between devices and staging | `Hcq2 › linking and running ›` "a copy between devices and the kernel it feeds agree…", "copies across three devices agree…", `Hcq2 › compile_linear › copies through the halves of a staging buffer…`; size checks are nx.device's `Buffer.copy` |
| old: unit/engine/test_realize.ml group "Owned buffer resolution" (5 tests), group "Linear execution" (6 tests) | resolving arguments and running kernels | `Hcq2 › linking and running` (the engine's link and run); the owners are old tolk's, dropped |

## Hcq2

`H` is the `Hcq2` suite (`test/runtime/support/hcq2`). Its goldens come from
tinygrad's NULL queue on the CPU devices CPU:1 to CPU:3, with D1 applied in the
generator: for each of ten cases, the batches `sched_batches` makes, the
schedule `compile_linear` returns and its host programs' source. It runs batches
through `tolk.engine` on the NULL device of test support, which runs the queues
the host programs submit on a domain of its own. `test_hcq2_sdma` runs apart,
with `HCQ_NUM_SDMA` set.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_hcq2.py::TestHCQ2Deps (6 tests) | byte-range dependencies | `H › Deps` (6 tests, one each) and `H › Deps law › Deps agrees with a byte-by-byte model` |
| tinygrad: null/test_hcq2.py `run`, `check`, `orders` | the symbolic executor of a batch's queues | `Batches` (test support) and `H › sched_batches`'s `well_formed` |
| tinygrad: null/test_hcq2.py::TestHCQ2Schedule::test_kernels_run_in_order | | `H › sched_batches › kernels on one queue run in order`; `chain_batched.golden` |
| tinygrad: null/test_hcq2.py::TestHCQ2Schedule::test_a_peer_kernel_runs_after_the_copy_that_feeds_it | | `H › sched_batches › a kernel runs after the copy that feeds it`; `peer_copy_batched.golden` |
| tinygrad: null/test_hcq2.py::TestHCQ2Schedule::test_lanes_of_a_sharded_kernel_do_not_wait_for_each_other | | `H › sched_batches › kernels of different devices do not wait for each other`; `sharded_batched.golden` |
| tinygrad: null/test_hcq2.py::TestHCQ2Schedule::test_a_device_without_a_copy_queue_copies_with_a_kernel | | `H › compile_linear › a copy on a device without copy queues is a kernel`, `H › linking and running › a copy without copy queues runs as a kernel, and copies`; `peer_copy_kernel_*.golden` |
| tinygrad: null/test_hcq2.py::TestHCQ2Schedule::test_a_host_kernel_splits_the_batch | | `H › sched_batches › a call on a device without queues splits the batch`; `host_split_*.golden` |
| tinygrad: null/test_hcq2.py::TestHCQ2Schedule::test_batches_of_real_workloads_are_well_formed | | `H › sched_batches › copies between devices of three kinds are well formed`; `sharded_sum_*.golden`, `copies_*.golden` |
| tinygrad: null/test_hcq2.py::TestHCQ2Profile::test_profiling_reports_a_range_per_kernel | | `H › linking and running › a profile records a span of each kernel on its device, in order`; `H › stamp slots (D7)` |
| tinygrad: null/test_hcq2.py::TestHCQ2Profile::test_slots_addressed_by_the_device | slots read through the device's own addresses (`pm_lower`) | dropped: `pm_lower` is excluded (ruling) |
| tinygrad: null/test_hcq2.py::TestHCQ2Link::test_links_serve_any_input | | `H › linking and running › a link serves any buffers bound to its inputs` |
| tinygrad: null/test_hcq2.py::TestHCQ2Link::test_eager_templates_compile_once | | dropped: the eager-template caches are excluded (ruling) |
| tinygrad: null/test_hcq2.py::TestHCQ2Link::test_repeated_word_loops | | `H › patch and bufferize_cmdbuf › a word written at several offsets is written by one loop` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_compile_and_link_are_idempotent | | `H › compile_linear › returns a linear holding a lowered batch as it is`; `H › lower_call › raises Invalid_argument for a batch lowered already`; linking twice is the engine's |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_new_inputs_each_call, test_jit_symbolic | | `H › linking and running ›` "a link serves any buffers bound to its inputs", "a schedule's variables reach its kernels"; the jit is L8 |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_repeated_copy | | `H › linking and running › a batch split by a host kernel agrees with running them one by one` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_map_cpu_buffer_preserves_contents, test_caches_hold_no_buffers, test_jit_has_no_rt_buffers | | dropped: nx.device's borrows, the excluded template caches and ring buffers |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Fence::test_a_schedule_waits_for_its_previous_run | | D1: the wait is the engine's `Submission.wait`; `H › linking and running › a run waits for its batch's previous run before it rewrites the batch's memory`; the re-arm is `H › timeline values (D1) ›` "the fence is of every queue's signal", "the fence lowers to stores of zero into the slots, and nothing else" |
| tinygrad: runtime/test_hcq2.py::TestHCQ2FFI::test_ffi_ccall, test_ffi_cstruct, test_nested_cstruct_patches | | `H › ccall, cstruct and cfield` (4 tests) |
| tinygrad: runtime/support/hcq2.py `layout_args`, `pack_args` | | `H › layout_args and pack_args` (4 tests) |
| tinygrad: runtime/support/hcq2.py `patch`, `HWQueue.q` | | `H › patch and bufferize_cmdbuf ›` "patch writes its blob, then its rows…", "a row known at link is written at link…"; `H › Queue` (4 tests) |
| tinygrad: runtime/support/hcq2.py `sched_batches`'s queue choice, `HCQ_NUM_SDMA`, `ALL2ALL` | | `H › sched_batches ›` "a program runs on its compute queue, a copy on its source's copy queue", "AMD copies between peers take one queue, and one per peer with ALL2ALL"; `test_hcq2_sdma` |
| tinygrad: runtime/support/hcq2.py `_wait_ins`'s NV chain | | `H › sched_batches › on NV a compute queue that waits for another queue waits for its previous call too` |
| tinygrad: runtime/support/hcq2.py `get_enqueue_devs`'s Metal copies | | `H › sched_batches › Metal's copies stay outside batches, since the host copies its memory` |
| tinygrad: runtime/support/hcq2.py `stage_copy` | | `H › compile_linear › copies through the halves of a staging buffer of the host where the queues cannot reach` |
| tinygrad: runtime/support/hcq2.py `pm_unwrap_multi` | | `H › compile_linear › runs a call on sharded buffers once per device, each on its shard`, `H › linking and running › a sharded kernel computes each shard on its device` |
| tinygrad: runtime/support/hcq2.py `lower_call` | | `H › lower_call` (6 tests), `*_compiled.golden` |
| tinygrad: runtime/support/hcq2.py `hcq_link`, `pm_link`, `LinkCtx` | | dropped here: the engine's link; run through `H › linking and running` |
| tinygrad: runtime/support/hcq2.py `split_rdma`, `pm_rdma_encode` | | dropped: RDMA is excluded |
| — | D1 | `H › timeline values (D1)` (9 tests); the goldens |
| — | D7 | `H › stamp slots (D7)` (4 tests) |
| — | D12 | `H › profile keys (D12)` (2 tests) |
| — | D30 | `H › ranges (D30)` (13 tests); `H › Deps › a write that does not trim keeps the accesses to the bytes it writes`, `› forgotten accesses are no longer followed` |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_hcq2.ml all_to_all_copy_queues | SDMA queue counts | `H › sched_batches › AMD copies between peers…`, `test_hcq2_sdma` |
| old: unit/engine/test_hcq2.ml staged_peer_dependencies | staging alternates two slots, each reused after its reads | `H › compile_linear › copies through the halves of a staging buffer…` (the copies' count); the dependencies are `H › Deps` |
| old: unit/engine/test_hcq2.ml peer_group_batches | one batch per kind, in order | `H › sched_batches › consecutive calls of two kinds make a batch of each, in the order they appear` |
| old: unit/engine/test_hcq2.ml sharded_batches | a sharded kernel and the next copies in one batch | `sharded_batched.golden`, `sharded_sum_batched.golden` |
| old: unit/engine/test_hcq2.ml byte_dependencies | | `H › Deps law › Deps agrees with a byte-by-byte model` |
| old: unit/engine/test_hcq2.ml region_identity | views of one storage and lanes | `H › unwrap_view and unwrap_lane` (7 tests) |
| old: unit/engine/test_hcq2.ml overlap_waits, parameter_views | only overlapping accesses wait | `H › Deps ›` "a write of other bytes keeps the dependencies", "views of one storage depend on the bytes they share" |
| old: unit/engine/test_hcq2.ml nv_chain | | `H › sched_batches › on NV a compute queue…` |
| old: unit/engine/test_hcq2.ml alias_ordering | | `H › Deps › an access never waits for itself through an alias`, `H › Deps › a write waits for every read since the last write` |
| old: unit/engine/test_hcq2.ml peers_and_timestamps | epilogues of peers, profile slots | `H › timeline values (D1) › a queue touching a peer's memory waits for the peer's submitted work too`, `H › stamp slots (D7)` |
| old: unit/engine/test_hcq2.ml compiled_host_submission | the host program patches and replays | `H › linking and running` |
| old: unit/engine/test_link.ml replacement_ownership, obsolete_link_collection, concurrent_link_publication | | dropped: old tolk's owners and link caches (the templates are excluded) |
| old: unit/engine/test_link.ml allocation_specs | command buffers uncached, volatile placeholders on the host | dropped here: the engine's allocation (`tolk.engine`'s link) |
| old: unit/engine/test_link.ml initialization, cast_patches, addresses | blobs, words and addresses written at link | `H › patch and bufferize_cmdbuf`, `H › ccall, cstruct and cfield › a C structure holds each field…` |
| old: unit/engine/test_link.ml input_links, preserve_runtime, host_call_replay | | `H › linking and running › a link serves any buffers bound to its inputs`, `H › patch and bufferize_cmdbuf › a word written at several offsets…` (two runs) |

## Engine

The suite is `Tolk_next_engine` (`engine/tolk_next_engine/`), written `EN`
below. `EN › batches` runs batches on the NULL devices of test support
(`Null_device`), whose queues run behind the host on a domain of their own:
the engine's half of D1 (each run signals its device's next value once), D7
(each kernel's span on its compute lane, each copy's on its copy lane), runs
from two domains, and the waits of RFC 0011's Amendment 2, on the device and
on the host. The batches' encoding and the rest of their execution, the fence
under latency included, are the Hcq2 suite's (`H`). `EN › Metal` (slow, on
macOS) runs the recorded copies on the Metal device, from the host: what they
write, the buffers each run binds, one signal per run through Metal's shared
event, runs from two domains, no allocation or load during a run, and
`measure`'s stamps; the rest of Metal's execution is the Ops_metal suite's. `EN › link and run › recorded` runs the Schedule suite's recorded
programs end to end: each is scheduled, compiled for the devices it names
(the host and test devices of the host's memory, `Run.devices`), linked with
its storage bound to buffers of small integers, and run, and its storage then
holds what its tensors compute (Tensors), with each variable at the value it
runs with. One program of each kind of call runs by default, the rest are slow.
`EN › link and run › a schedule runs on the buffers each run binds to its
parameters` makes a program's buffers parameters of the compiled schedule and
runs it on two sets of buffers.

Kernels compile on the Worker's domains and the NULL devices run their queues
on one, and OCaml refuses `Unix.fork` in a process that has spawned a domain,
so each of `tolk_next_engine.ml`'s 75 mutants is armed in a process of its own
(`--arm`) over the default run: 62 fail it. With the run lock removed by hand,
the two-domain test fails on each of three runs. `||` as `&&` in the storage
a batch's patches reach drops that storage from the buffers a run touches,
which alone tell `Nx_device.submit` about link-folded storage of a device
outside the batch: `EN › batches › a run waits for a device without queues of
storage` fails, since the work that fills its source touches nothing else the
run touches. The survivors:
- dismissed in the source as equivalent (five), and, equivalent too, any
  parameter or any tagged node counting as a placeholder in a batch's
  patches, which hold only placeholders and storage, and `>` for `>=` in
  either bound of `measure`'s repetitions, off by one run of 1000 or one
  nanosecond of 10 000;
- the kernel count of a `DEBUG=2` line, one less: the count is the process's,
  so a test sees only that each line counts one more than the line before;
- `sp.device == d && sp.name = name` as `||` in a batch's `DEBUG=2` lines: a
  batch's kernels take their devices' spans in order, so the first span of the
  device is the kernel's own;
- `sp.device == d && sp.name = name` as `||` in `measure`: a host profile of
  one call holds one span;
- `cold` as `not cold`: only NV invalidates its caches;
- `info.table < 0` as `<= 0`: a batch whose host program reads its address
  table first takes it as its first argument, and no batch here does. The
  NULL devices' batches read their submission's word first, as tinygrad's
  NULL batches read their doorbell, and Metal's read `objc_msgSend`'s word
  first, or their slots under a profile.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/support/hcq2.py` `_staging` (no test) | one staging buffer of 128 MiB per host, which every schedule's staged copies share | `EN › batches › linked schedules that stage share the host's staging memory`; their order: `› staged runs of two programs on other devices take turns`, `› staged runs of two programs from two domains each copy their own`, which the NULL devices also order through the host memory every batch touches (D45) |
| tinygrad: null/test_hcq2.py::TestHCQ2Link::test_links_serve_any_input | a linked schedule serves any input, whose address it does not keep | `EN › link and run › a schedule runs on the buffers each run binds to its parameters` (contiguous, copy_view, shard_add) |
| tinygrad: null/test_hcq2.py::TestHCQ2Link::test_eager_templates_compile_once | compiling a schedule again returns the compiled one | dropped here: compile_linear's result is Hcq2's; the engine keeps no cache of links, and rune keeps its programs by key |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_repeated_copy | a copy out, a copy in and a copy out run in order | `EN › link and run › copies run in the order of their schedule` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_caches_hold_no_buffers | freeing the values frees the device memory | `EN › runs › a linked schedule holds its storage while it is reachable` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_has_no_rt_buffers | a jit's link owns its buffers, none from the one-shot ring | dropped: the engine has no ring; link allocates everything a schedule names (`EN › runs › a run of kernels allocates and loads nothing`) |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_new_inputs_each_call, test_jit_symbolic | a captured jit's new inputs and bindings | the engine's half: `EN › link and run › a schedule runs on the buffers each run binds to its parameters`, `a schedule runs with each binding of its variables`; the jit is L8's |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_map_cpu_buffer_preserves_contents | mapping host memory for a device keeps its bytes | dropped: mapping is nx.device's (`Buffer.borrow`) |
| tinygrad: runtime/test_wait_loop.py::TestWaitLoop::test_wait_loop, test_nested_loop_in_range, test_two_sequential_loops, test_loop_in_loop | do-while loops count to 10, 12, 25 and 12 | `EN › Program › loops ›` "wait_loop runs its loops to 10", "nested_loop …", "two_loops …", "loop_in_loop …", from the Linearizer's goldens |
| tinygrad: runtime/test_wait_loop.py::TestWaitLoop::test_wait_loop_spec | the loop under `SPEC=2` | dropped: the specification is Spec's; the loop runs in `EN › Program › loops` |
| tinygrad: runtime/test_wait_loop.py::TestVolatileLoops::test_async_wait_ext | a kernel spins on a host word another thread sets | dropped: every wait of a run is `Submission.wait` (D1), so no program the engine runs spins on the host |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_kernel_run, test_profile_kernel_run_wait | a kernel run is one profile range named after the kernel | `EN › runs › each kernel of a run is a span of the host under a profile`; on a device with queues, `EN › batches › a kernel is a span of its compute lane and a copy of its copy lane` |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_copyin, test_profile_multiops, test_profile_multidev, TestSimpleProfiler::test_profiler, TestProfiler::test_cpu_profile | copies and host ranges are profile events | dropped: nx.device records its copies and host spans (`Nx_device.Profile`) |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_multidev_transfer, test_profile_graph, test_dev_jitter_matrix | device-to-device transfers, graph events and clock jitter | dropped: hardware with two devices, and nx.device's calibration |
| tinygrad: runtime/test_search.py::TestSearch::test_beam_symbolic_kernel | beam search applies optimisations to a symbolic kernel | dropped here: the search is `Codegen.Opt.Search`'s; the measurement it takes is `EN › measure` |
| tinygrad: runtime/test_realize_is_realize.py (13 tests) | `Tensor.realize` of lists, ones, disk, multi-device and variables | dropped: the `Tensor` surface; the frontend is nx, and realization rune's (L9) |
| tinygrad: runtime/test_after.py (12 tests) | ordered and disjoint stores, and the gradients through `after` | dropped: the `Tensor` surface and autodiff (rune, L9); the order of stores is Schedule's |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_symbolic.ml "symbolic shrink+sum runs for several bind values on CPU" | one compiled shrink and sum, run for several values of its variable | `EN › link and run › a schedule runs with each binding of its variables` (variable_reduce at 1, 10 and 5) |
| old: unit/engine/test_symbolic.ml "symbolic launch dims run on CPU" | a variable sizes a kernel's loop | `EN › link and run › recorded › variable_offset writes what its tensors compute` and the slow `variable_*` programs |
| old: unit/engine/test_symbolic.ml "symbolic shrink+sum runs for several bind values on CUDA", "symbolic launch dims run on CUDA" | | dropped: CUDA runs batches, whose encoder the engine does not have yet |
| old: unit/engine/test_link.ml "executes linked host calls with rebound buffers and scalars" | a linked schedule runs on buffers and variables bound at each run | `EN › link and run › a schedule runs on the buffers each run binds to its parameters`, `a schedule runs with each binding of its variables` |
| old: unit/engine/test_link.ml "does not cache link-time inputs" | | `EN › link and run › a schedule runs on the buffers each run binds to its parameters` |
| old: unit/engine/test_link.ml "concurrent first links publish one retained graph", "secondary owner replacement invalidates cached links without invalidating retained links", "obsolete owner storage retires even while the original graph remains live" | | dropped: the engine keeps no cache of links and no owners; a linked schedule holds its storage while reachable (`EN › runs › a linked schedule holds its storage while it is reachable`) |
| old: unit/engine/test_realize.ml "passes every scalar from program metadata" | each variable of a program reaches its call, by name | `EN › Program › a kernel runs on its buffers, in the order of its globals`, `a variable left out of vars takes its bound value` |
| old: unit/engine/test_realize.ml "requires scalar variables" | | `EN › Program › run refuses an unbound variable`; `EN › refusals › run refuses a variable it binds no value`; `EN › measure › an unbound variable is refused` |
| old: unit/engine/test_realize.ml "copies bytes between host-backed devices", "copies bytes across backend prefixes" | | `EN › link and run › recorded ›` "copy …", `copies run in the order of their schedule` |
| old: unit/engine/test_realize.ml "preserves overlapping views across staging chunks in both directions", "preserves overlapping external allocations while streaming", "rejects size or dtype mismatches before copy" | | dropped: copies are nx.device's (`Buffer.copy`) |
| old: unit/engine/test_realize.ml "resolves the owner retained by each BUFFER node", "unplaced buffers require owned storage" | storage bound at link, allocated otherwise | `EN › link and run › recorded` (bound), `EN › runs › a linked schedule holds its storage while it is reachable` (allocated); `EN › refusals › link refuses a bound node that is no storage` |
| old: unit/engine/test_realize.ml "resolves PARAM through input_uops", "resolves PARAM kernel args from input_uops", "runs a kernel call with resolved buffers" | | `EN › link and run › a schedule runs on the buffers each run binds to its parameters` |
| old: unit/engine/test_realize.ml "resolves byte view as an offset view", "resolves an offset byte view structurally" | an argument that is a view at an offset | `EN › link and run › a schedule runs on the buffers each run binds to its parameters › copy_view`, `recorded › copy_view writes what its tensors compute` (slow) |
| old: unit/engine/test_realize.ml "rejects an unbound PARAM" | | `EN › refusals › run refuses a parameter it binds no buffers`, `run refuses slots that stop before the last parameter` |
| old: unit/engine/test_realize.ml "replays keep one runtime under changed compile settings" | a replay loads nothing | `EN › runs › a run of kernels allocates and loads nothing` |
| old: unit/engine/test_realize.ml "replays resolve storage views without building nodes" | | dropped: the nodes a run builds are not observable; that it allocates no device memory is `EN › runs › a run of kernels allocates and loads nothing` |
| old: unit/engine/test_realize.ml "execution counters retain exact large costs" | | dropped: the engine keeps no counters; a call's cost is `Realize.estimate_uop` |
| old: unit/engine/test_realize.ml "keys cached programs by exact device name", "same-named devices use their own runtime loader" | a program is loaded per device | `EN › Program › a binary loaded twice is loaded once`; loading per device of a call is `EN › link and run › recorded › shard_add writes what its tensors compute` |
| old: unit/engine/test_realize.ml "concurrent submissions retain their own address tables", "independent links serialize reservations on their shared device timeline", "submission addresses are resolved after preparation" | runs of one batch rewrite its address table only once the previous run is done | `EN › batches › runs of one batched schedule from two domains each compute their own`; `H › linking and running › a run waits for its batch's previous run before it rewrites the batch's memory` |
| old: unit/engine/test_realize.ml "timing samples forward timeout and release transient runtimes", "failed timing dispatch drains before runtime release", "failed timing drain preserves its runtime" | timing a program on a device | `EN › batches › a program's run on a device with queues takes a positive time`, `EN › measure`; the runtimes they release are old tolk's |
| old: unit/engine/test_link.ml allocation_specs, in Hcq2's section | command buffers uncached, volatile placeholders in memory the host sees | dropped: link keeps its placeholders, so which memory holds them is not observable; on the NULL devices it is one allocator. A vendor's placeholders are its `placeholder`'s, which link binds (`EN › batches › link refuses a C function of a library it does not know`) |

### Deferred to the engine by other sections

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_symbolic.ml (every test), in Symbolic's section | | rows above |
| tinygrad: runtime/test_rangeify.py::TestDoubleMatmul::test_double_matmul, in Indexing's section | the numbers of two matmuls | `EN › link and run › recorded › double_matmul writes what its tensors compute` (slow) |
| tinygrad: null/test_call.py::TestCallCodegen::test_compiled_scalar_slots_are_not_call_slots, in Cstyle's section | a compiled program's variables are not its call's slots | `EN › link and run › recorded › precompiled_scalar writes what its tensors compute` |
| tinygrad: runtime/test_linearizer.py::TestLinearizer::test_arg_dedup, test_load_removed, test_assign_fold, in Codegen's section | realized values | `EN › link and run › recorded` (assign, setitem, read_then_overwrite; slow) |
| old: unit/test_device.ml "Buffer.copy_from delegation" (3 tests), unit/test_device_no_engine.ml (2 tests), in Device's section | | `EN › link and run › recorded ›` "copy …", `copies run in the order of their schedule`; the copy itself is nx.device's |
| tinygrad: null/test_tensor_uop_representation.py (5 tests), old: unit/uop/test_uop.ml runtime_realization_state_parity, in Ops' section | a realized value is a BUFFER | dropped: the engine binds storage to a BUFFER at link and keeps no realized state (D3); a realized value is rune's (L9) |

## Ops_metal

`O` is the `Ops_metal` suite (`test/runtime/ops_metal`). Its goldens come from
tinygrad's `MetalQueue` on a METAL device described without a GPU, with D1, D7
and D34 applied in the generator: for each of eight cases (Apple9, Apple7 and
Mac2 families, no residency set, a profiled chain and a profiled single
command, a launch size that reads a variable, and copies to the host between
batches), the schedule `sched_batches` receives, the schedule `compile_linear`
returns and its host programs' source. It needs no GPU and no MTLCompiler.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/ops_metal.py` `MetalQueue.exec`, `.submit` (no test) | the argument layout, the indirect command buffer's placeholder and the messages of the host program | `O › recorded cases › *_compiled.golden`, `*_host.golden` (8 cases) |
| tinygrad: `runtime/ops_metal.py` `MetalQueue.submit`, `residency.value is None` | an encoder declares the buffers resident without a residency set | `O › recorded cases › chain_no_residency_set`: the engine's table of resources runs only where Metal has no residency sets (before macOS 15), and no such Mac runs `OX`, so the recorded host program is its only check |
| tinygrad: `runtime/ops_metal.py` `MetalQueue.submit`, `int(arch[5:]) < 9` | before Apple9, the encoder sets each pipeline | `O › recorded cases › chain_apple7`, `chain_mac2`; `OX` runs it on GPUs before Apple9, such as the M1's |
| tinygrad: `runtime/ops_metal.py` `MetalQueue.submit`, the stamps | a profiled command runs in a command buffer of its own, which its stamps hold | `O › recorded cases › chain_profile` (a range over the commands but the last), `one_profile`; D7 and D34 |
| tinygrad: `runtime/ops_metal.py` `MetalQueue.exec`, symbolic sizes | a launch size that reads a variable is set on the command | `O › recorded cases › variable` |
| tinygrad: `runtime/ops_metal.py` `get_enqueue_devs` on METAL | Metal's copies are the host's, between batches | `O › recorded cases › host_split` |
| tinygrad: `runtime/ops_metal.py` `MetalDevice.pm_lower` (`mtl_poll`) | a host program reads the timeline from the event | dropped: no host program reads a timeline (D1) |
| tinygrad: runtime/test_wait_loop.py | host loops that wait on a signal | the `Hcq2` suite's: no Metal host program waits (D1) |
| tinygrad: device/metal/test_metal.py::TestMetal::test_alloc_oom, test_failed_newLibraryWithData, test_free | the device's memory and pipelines | nx.metal.device's suite |

### Execution

`OX` runs batches through `tolk.engine` on the Mac's GPU (`test_ops_metal_exec`,
the `slow` alias, built on macOS only).

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/ops_metal.py` `MetalDevice.pm_bufferize`, `sels`, `new_icb`, `new_slots` (no test) | the engine's words of a batch | `OX › execution` (9 tests) |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_repeated_copy | copies out, in and out between the GPU and the host | `OX › copies out, in and out again leave the host the bytes copied in` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_new_inputs_each_call | a linked batch serves new inputs on each run | `OX › a run waits for its batch's previous run before it rewrites the batch's arguments` (eight inputs, run without synchronizing) |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_symbolic | a symbolic size on each run | `OX › a launch size that reads a variable is set on each run` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_map_cpu_buffer_preserves_contents | | dropped: tinygrad skips it on METAL, which maps nothing |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_compile_and_link_are_idempotent, test_caches_hold_no_buffers, test_jit_has_no_rt_buffers | the jit's caches and runtime ring | dropped here: the captured jit is L8's; the engine has no runtime ring |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Fence, TestHCQ2FFI | the fence and C calls on the CPU | the `Hcq2` and `Engine` suites' |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_kernel_run, test_profile_multiops | a kernel's span on its device | `OX › a profile records a span of each kernel on the device, in order`; `OX › a profiled batch run twice keeps the second run's spans` (D34) |
| DIVERGENCES D30 | a range's addresses, integers whatever the element type | `O › loops (D30) › a range's addresses are integers, profiled or not` |
| tinygrad: `runtime/ops_metal.py` `MetalQueue.exec`, symbolic sizes (`d.cast(dtypes.uint64)`) | a launch size of an expression, cast to a 64-bit word, stays an integer: the kernel's compilation commits it | `O › launch sizes › a launch size of an expression is computed in integers` |
| DIVERGENCES D30 | a range around calls, a loop of indirect commands in one submission | `OX › each trip of a range runs its kernel on its own window`, `› a range of 20000 trips runs from one indirect command buffer`, `› a profiled range records a span of each trip's kernel` |
| old: `unit/test_metal_completion.ml` (7 tests) | the old runtime's command ownership, completion order and retirement | dropped: completion and command buffers are nx.device's (`Submission`, `resolve`) |
| old: `unit/test_runtime_metal.ml` "an argument structure of 15/16/29/33 buffers ..." | many buffers dispatch and rebind | `OX › a kernel of 33 buffers runs from its arguments' buffer`: one argument buffer, so no direct dispatch |
| old: `unit/test_runtime_metal.ml` "replays symbolic local workgroup dimensions" | | `OX › a launch size that reads a variable is set on each run` |
| old: `unit/test_runtime_metal.ml` "replays a multi-kernel chain in order", "compile and run one kernel", "exec is ordered" | | `OX › a chain of kernels computes what the interpreter says` |
| old: `unit/test_runtime_metal.ml` "collects GPU timestamps across profiled replay", "wait returns gpu time" | | `OX › a profile records a span ...`, `› a profiled batch run twice ...` |
| old: `unit/test_runtime_metal.ml` "relaunches without an intervening synchronize" | | `OX › a run waits for its batch's previous run ...` |
| old: `unit/test_runtime_metal.ml` "shared copies respect buffer view offsets", "buffer views copy at byte offsets", "as_buffer aliases ...", "nested buffer views ...", "LRU-reused buffers ...", "collects dropped views ..." | | dropped: buffers and copies are nx.device's |
| old: `unit/test_runtime_metal.ml` tensor-core, reduction and typed-argument tests | kernels' values | dropped here: kernels are the codegen and renderer suites'; their execution on Metal is L9's graph parity |
| old: `unit/test_runtime_metal.ml` "beam timing replays compiled Metal queues" | | the engine's `measure` and L7's `Search` |
| old: `unit/test_runtime_metal.ml` "multi-device calls ...", "a sharded kernel on two Metal devices ..." | | dropped: a Mac has one Metal device |
| old: `unit/test_runtime_metal.ml` "CPU kernels map Metal storage ...", "Metal kernels map borrowed host memory ..." | | dropped here: borrows are nx.device's, and the engine's `link` suite |

## Ops_cuda

`C` is the `Ops_cuda` suite (`test/runtime/ops_cuda`). Its goldens come from
tinygrad's `CUDAQueue` on a CUDA device described without a GPU (sm_89, which
reaches the host's memory), with D1 and D36 applied in the generator: for each
of six cases (a chain, a profiled chain, a launch size that reads a variable,
a copy in from the host and its profiled form, and copies to the host between
batches), the schedule `sched_batches` receives, the schedule `compile_linear`
returns and its host programs' source. It needs no GPU and no driver.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/ops_cuda.py` `CUDAQueue.launch`, `.exec` (no test) | the arguments after their size, the launch's extra words on the queue, `cuLaunchKernel` on the compute stream | `C › recorded cases › chain`, `variable` (a launch size that reads a variable) |
| tinygrad: `runtime/ops_cuda.py` `CUDAQueue.copy`, `.wait`, `.signal` (no test) | copies on the copy stream, waits and writes of 64-bit words between the streams | `C › recorded cases › copy_in`, `host_split` |
| tinygrad: `runtime/ops_cuda.py` `CUDAQueue.timestamp` (no test) | a host function stamps a slot | `C › recorded cases › chain_profile`, `copy_in_profile` |
| tinygrad: `runtime/ops_cuda.py` `CUDAQueue.submit` (no test) | the status of the last call is stored | `C › recorded cases › *_host.golden` |
| tinygrad: `runtime/ops_cuda.py` `CUDADevice.pm_bufferize`, `handles`, `stamp`, `function` | the engine's words of a batch | `CX` (below) |
| tinygrad: `runtime/ops_cuda.py` `CUDAAllocator`, `CUDADevice._wait_signal`, `count` | memory, peer maps and waits | nx.cuda.device's suite |
| tinygrad: `runtime/ops_cuda.py` the `MOCK` interface (`test/mockgpu/cuda`) | a CUDA driver in Python | dropped: no mock drivers (plan §10) |
| old: `unit/test_cuda_queue.ml` "compiles mixed-width arguments and symbolic launch dimensions" | | `C › recorded cases › variable`; argument layout is `Hcq2.layout_args`'s |
| old: `unit/test_cuda_queue.ml` "profiles compute and copy calls with native host callbacks" | | `C › recorded cases › copy_in_profile` |
| old: `unit/test_cuda_queue.ml` "compiles dependencies crossing compute and copy queues" | | `C › recorded cases › copy_in`, `host_split` |
| old: `unit/test_cuda_queue.ml` "host copies retain an ordinary execution fallback" | | `C › recorded cases › host_split`: a copy the queues reach runs on the copy stream; staging is `Hcq2`'s |
| old: `unit/test_cuda_queue.ml` "compatible peers share a submission with cross-device dependencies", "independent groups regroup without crossing ordinary calls", "peer timelines use each device's own context" | batching across devices | the `Hcq2` suite's batching, which no vendor changes |
| DIVERGENCES D30 | a range around launches, a loop of the host program, its addresses integers | `C › loops (D30) › a range is a loop of the host program around its launches`, `› a range's addresses are integers, profiled or not` |

### Execution

`CX` runs batches through `tolk.engine` on NVIDIA GPUs (`test_ops_cuda_exec`,
the `slow` alias), and skips each test on a machine without the GPUs it needs.
No CI machine has one: it runs on hardware by hand.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_repeated_copy | copies out, in and out between the GPU and the host | `CX › copies out, in and out again leave the host the bytes copied in` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_new_inputs_each_call | a linked batch serves new inputs on each run | `CX › a run waits for its batch's previous run before it rewrites the batch's arguments` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_symbolic | a symbolic size on each run | `CX › a launch size that reads a variable is set on each run` |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_kernel_run, test_profile_multiops | a kernel's span on its device | `CX › a profile records a span of each kernel on the device, in order`, `› a profiled batch run twice keeps the second run's spans` |
| tinygrad: `runtime/ops_cuda.py` `CUDAAllocator._map`, `hcq2.py` `stage_copy` | GPU memory is reached through the host | `CX › a copy between two GPUs goes through the host's staging memory` (two GPUs) |
| DIVERGENCES D30 | a range around calls, a loop of launches in one submission | `CX › each trip of a range runs its kernel on its own window` |
| old: `unit/test_runtime_cuda.ml` "compile and run one kernel", "exec is ordered", "replays a multi-kernel chain" | | `CX › a chain of kernels computes what the interpreter says` |
| old: `unit/test_runtime_cuda.ml` "passes scalar variables", "patches scalar values between launches", "queue call replays with updated variables" | | `CX › a launch size that reads a variable is set on each run` |
| old: `unit/test_runtime_cuda.ml` "rebinds buffers through repeated asynchronous launches", "queue call replays with rebound inputs", "... rebound input and output slots" | | `CX › a run waits for its batch's previous run ...` |
| old: `unit/test_runtime_cuda.ml` "copies feed dependent kernels and later copies", "queues mapped host copies and falls back for unaligned imports" | | `CX › a copy to the host and back runs on the host, between batches`, `› copies out, in and out again ...` |
| old: `unit/test_runtime_cuda.ml` "cross-device copy preserves views with peer or host fallback", "queued peer copies preserve views with mapping fallback" | | `CX › a copy between two GPUs goes through the host's staging memory` |
| old: `unit/test_runtime_cuda.ml` "typed arguments preserve scalar widths in dispatch and replay" | | `CX › a kernel of 33 buffers runs from its arguments' buffer`; the layout is `Hcq2.layout_args`'s |
| old: `unit/test_runtime_cuda.ml` "wait returns gpu time" | | `CX › a profile records a span ...` |
| old: `unit/test_runtime_cuda.ml` buffer views, LRU reuse, pinned storage, "cached functions survive independent link collection", "concurrent first links share one timeline and context descriptor" | | dropped: buffers, programs and timelines are nx.device's |
| old: `unit/test_runtime_cuda.ml` "f16 tensor-core matmul" | | dropped here: kernels are the codegen suites'; their execution is L9's graph parity |

## Jit

`J` is the `Jit` suite (`test/engine/jit`), `JB` its JITBEAM suite
(`test_jitbeam`), which runs once per setting of `JITBEAM`, since the
environment is read once per process. `J`'s goldens are 66 captures,
generated by `gen/engine/jit.py`: each function that tinygrad's four jit test
files compile, captured by `TinyJit` on its second call over realized buffers,
on the host and on CPU:1 to CPU:3 with the NULL device's queues and a copy
queue. A case's goldens are its captured schedule, the buffers the capture
holds, its inputs in slot order, and what `jit_lower` makes of them; each case
records its variables with their bound values (`captures.golden`).

For each case, `J › jit_lower › recorded` compares the lowering with
tinygrad's, profile keys left out (D12), and checks that each input the
schedule reaches is the parameter of its slot, of its type, device and size,
and that no other parameter is bound; that the storage the lowering reaches is
the held storage of the capture that is no input, plus arenas the capture does
not reach; and that lowering twice gives the same schedule up to the numbers
of the storage and placeholders it makes. `J › replay › recorded` links the
lowering on the host and the NULL devices and replays it as a jitted function
is called: one to three runs, the first from contents drawn from a seed for
each held buffer and input, each later one keeping what the runs before it
left, with new inputs; each variable takes the values tinygrad's tests replay
it with. After the last run, the held buffers and inputs hold what the captured
schedule leaves when it runs the same runs unplanned, compiled as it is with
its held buffers and inputs bound. `J › jit_lower › drawn` states the same
laws over drawn captures: calls of compiled kernels and copies over buffers of
four or seventy floats on the host and CPU:1, whose inputs and constants no
call writes, holding the last buffer written and those the draw keeps. Six
cases run by default (a kernel on the host, an input a kernel writes, held
buffers without inputs, copies to a device with queues, a held constant of
Python's data, a variable); the other 60 are slow.

The parts of `engine/jit.py` that are not ported:

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `engine/jit.py` `prune_linear` | with `prune=True`, a capture runs once the calls that none of its inputs feeds, and replays the rest | dropped: rune never runs a capture's kernels once; the values they compute are captures, which the compiled function holds (RFC 0012) |
| tinygrad: `engine/jit.py` `_copy_input`, `CapturedJit._written_uops` | a replay copies each input its program writes into fresh storage first | dropped: rune copies consumed storage that cannot lend before the run, and a program writes no other input (RFC 0012, Replaying) |
| tinygrad: `engine/jit.py` `JitError` | the jit's exception | dropped: its raisers are `_copy_input` and the `Tensor` surface (arguments that differ from the capture's, nothing captured, a result that is no `Tensor`); rune's compiled call raises its own `Jit_error` (RFC 0012) |
| tinygrad: `engine/jit.py` `CapturedJit.linear` | a captured schedule is linked once, outside the link cache | `Tolk_next_engine.link`: the engine keeps no link cache; `J › replay › recorded` links each lowering once and replays it |
| tinygrad: `engine/jit.py` `CapturedJit.__call__` | a replay runs the linked schedule on the call's inputs and variables, and at `DEBUG=1` says how many calls it runs | `Tolk_next_engine.run`; `J › replay › recorded` (66 cases); `EN › runs › at DEBUG=1, a run of ten calls says how many it runs`, `… of nine calls says nothing` |
| tinygrad: `engine/jit.py` `CapturedJit._symbolic_ret` | results of symbolic shape are rebound to the call's variables | dropped: `Tensor` surface; rune wraps each result (RFC 0012) |
| tinygrad: `engine/jit.py` `CapturedJit.free_intermediates` | the planned buffers are freed on demand | dropped: a linked schedule holds its arenas while it is reachable (`EN › runs › a linked schedule holds its storage while it is reachable`) |
| tinygrad: `engine/jit.py` `CapturedJit.__reduce__`, `_TinyJit.__reduce__` | a captured jit pickles | dropped: no persistent cache (RFC 0012) |
| tinygrad: `engine/jit.py` `_prepare_jit_inputs`, `_TinyJit`, `TinyJit` | the warm-up count, capture, argument checks and `Tensor` rebinding | dropped: `Tensor` surface; rune's compiled call traces directly and checks its keys (RFC 0012). The generator drives `TinyJit`'s capture to record the goldens |

### tinygrad

A case name `c` stands for its four tests in `J › jit_lower › recorded › c`
and its replay in `J › replay › recorded › c`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `engine/jit.py` `jit_lower` | inputs become parameters of their slots, held buffers stay out of the plan, the schedule compiles with `JITBEAM`'s width | every case; `J › jit_lower › drawn` (four laws); `JB` (six tests) |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_input_view | an input read through a view | `input_view` |
| tinygrad: runtime/test_jit.py::TestJit::test_global_counters_jit | three chained kernels, and the counters a replay adds to | `chain_of_three`; the counters dropped: `GlobalCounters` is not ported, a run's work is its `DEBUG=2` lines (`EN`) |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_assign, test_jit_assign_int8; runtime/test_jit_footguns.py::TestJitCorrectBehavior::test_input_mutation_consistent | a kernel adds one to its input on every call | `assign`, `assign_int8` (each replay's later runs accumulate) |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_copyin | a constant of Python's data copied in on every call | `copyin` |
| tinygrad: runtime/test_jit.py::TestJit::test_jitted_clone | a clone | `clone` |
| tinygrad: runtime/test_jit.py::TestJit::test_jitted_transfers | two inputs copied to another device | `transfers` |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_several_devs | copies to a device, then kernels there | `several_devs` |
| tinygrad: runtime/test_jit.py::TestJit::test_jitted_view | a reduction bitcast and copied to another device | `view_bitcast` |
| tinygrad: runtime/test_jit.py::TestJit::test_jitbeam_triggers_beam | `JITBEAM` makes the lowering search kernels | `JB › JITBEAM=3 › a kernel is searched with JITBEAM's width, not BEAM's`, `… with BEAM at 0`, `… BEAM keeps its value once the lowering returns`; `JB › JITBEAM=0 › no kernel is searched whatever BEAM is`; `JB › JITBEAM unset › a kernel is searched with BEAM's width`, `… BEAM at 0 searches no kernel` |
| tinygrad: runtime/test_jit.py::TestJit::test_simple_jit_reset, test_simple_jit_norealize, test_simple_jit_norealize_list, test_simple_jit_norealize_dict, test_kwargs_jit, test_array_jit, test_method_jit | calling conventions of `TinyJit` and `reset` | dropped: `Tensor` surface (`_TinyJit`'s counter and rebinding); their capture is `add` |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_multiple_outputs; TestMultioutputJit (3 tests) | three outputs | `multiple_outputs` |
| tinygrad: runtime/test_jit.py::TestJit::test_nothing_jitted, test_jit_zero_does_not_jit, test_jit_not_capturing, test_jit_shape_mismatch, test_jit_shape_views_mismatch, test_jit_duplicate_fail, test_jit_output_non_tensor_fail, test_jit_init_empty, test_jit_const_input, test_jit_deviceless_compute_input; runtime/test_symbolic_jit.py::TestSymbolicJit::test_jit_symbolic_shape_mismatch | `JitError` for arguments, results and captures the `Tensor` surface refuses, `JIT=0`, `CAPTURING=0` | dropped: `Tensor` surface; rune's compiled call checks its keys and raises `Jit_error` (RFC 0012) |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_size1_input | an input of one element | `assign`, `explicit`, `two_kernels` |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_random_regen, test_jit_v_nojit_random_regen, test_jit_multiple_random_regen, test_jit_random_after_unrealized_random; TestJitRandom::test_jit_rangeify; runtime/test_jit_footguns.py::TestJitCorrectBehavior::test_random_regenerates | random tensors regenerate on each call, reproducibly from the seed | dropped: `Tensor`'s random state; nx's generator and its keys are rune's (RFC 0012) |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_realization_and_sampling | a weight the function closes over | `weight` |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_output_clone | a result cloned out of the jit's storage | dropped: `Tensor` surface; the capture is `explicit` |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_init_empty_alt | an input assigned from another | `assign_input` |
| tinygrad: runtime/test_jit.py::TestJit::test_jit_lazy_grad_after_replay | gradients created during capture are held, not planned | `lazy_grad` (`… places no held buffer or input in an arena`) |
| tinygrad: runtime/test_jit.py::TestCopyInsideJit::test_copy_inside_jit | a copy into the jit's device | `copy_inside` |
| tinygrad: runtime/test_jit.py::TestJitPrune::test_prune_w_copy_correct, test_prune_w_independent_copy_correct, test_simple_prune | pruned and unpruned captures of a function of a weight | the unpruned captures: `weights_copy`, `weights_independent_copy`, `weights_kernel`; pruning dropped (`prune_linear`, above) |
| tinygrad: runtime/test_jit.py::TestJitFree::test_free_intermediates | intermediates freed and reallocated | `held_constant`; freeing dropped (`free_intermediates`, above) |
| tinygrad: runtime/test_jit.py::TestJitFree::test_updated_not_freed | a held buffer the function updates stays | `accumulator` (each replay's later runs accumulate) |
| tinygrad: runtime/test_jit.py::TestJitGraphSplit::test_jit_split_simple, test_jit_cpu_simple, test_jit_cpu_several, test_jit_multidev, test_jit_multidev_xfer, test_jit_multidev_copy | kernels and copies across the host and devices with queues | `split_simple`, `split_cpu`, `split_cpu_several`, `split_multidev`, `split_multidev_xfer`, `split_multidev_copy` |
| tinygrad: runtime/test_jit.py::TestJitInsideJit::test_jit_jit_error | a jit inside a jit | dropped: `Tensor` surface; rune's nested calls are `Op.intercepted` (RFC 0010) |
| tinygrad: runtime/test_jit_cases.py::TestJitCases::test_explicit, test_implicit_input, test_implicit_output, test_implicit_io | explicit and implicit inputs and outputs | `explicit`, `implicit_input`, `implicit_output`, `implicit_io` |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_output_buffer_reuse | a replay rewrites its output's storage | `sum` (the output is held; each replay's later runs rewrite it) |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_output_buffer_workaround | a result cloned out | dropped: `Tensor` surface |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_input_output_aliasing | two kernels, whose output is fed back as the next input | `two_kernels`; feeding a result back is rune's per-call copy (RFC 0012) |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_multiple_outputs_same_intermediate | two outputs of one window | `cat_window` |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_intra_kernel_output_input_aliasing | a window shifted into its own storage | `shift_window` (unpruned, 16 elements); the copy of the aliased input dropped (`_copy_input`, above) |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_slice_assign_works_without_realize | a held cache written at a variable position | `slice_assign` (`pos` over 0 to 3) |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_implicit_inputs_need_realize | a closed-over input | `implicit_input` |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_shape_change_after_capture_fails, test_python_constants_frozen, test_unrealized_const_input_error, test_conditional_branches_frozen, test_positional_kwargs_cannot_mix, test_class_method_shared_across_instances, test_side_effects_only_during_capture, test_item_creates_unrealized_return, test_item_bakes_in_values, test_tolist_bakes_in_values; TestJitCorrectBehavior::test_kwargs_order_doesnt_matter | tracing and argument semantics of `TinyJit` | dropped: `Tensor` surface; rune's compiled call traces and keys its calls (RFC 0012) |
| tinygrad: runtime/test_jit_footguns.py::TestJitFootguns::test_masked_select_static_size_jittable, test_nonzero_static_size_jittable | data-dependent selections of static size | `masked_select`, `nonzero` |
| tinygrad: runtime/test_symbolic_jit.py::TestSymbolicJit (22 tests but `test_jit_symbolic_shape_mismatch`) | captures over symbolic views, replayed over the variables' values | `plus1`, `inner_bound_var_view`, `plus1_pad_view`, `plus1_pad`, `symbolic_add`, `symbolic_matmul`, `mixed_with_no_symbol_kernel`, `symbolic_attention`, `cat_dim0`, `cat_dim1`, `cat_dim0_two_vars`, `cat_dim1_two_vars`, `two_vars_plus1_ij`, `two_vars_plus1_ji`, `symbolic_shrink`, `symbolic_slice`, `slice_var_shape`, `ones_sum`, `mean` (`mean`, `mean0`, `mean1`), `mean_2d` (three), `var` (three), `var_2d` (three); `i` over 1 to 4, `j` over 2 to 4 |

### old tolk

| Source | Behaviour | Outcome |
|---|---|---|
| old: unit/engine/test_jit.ml TinyJit › "create requires a function or a captured schedule", "reset requires a function-backed jit", "empty capture raises and clears the capture registry"; Capture and replay › "replay validates input size dtype and device" | the `TinyJit` surface | dropped: `Tensor` surface (RFC 0012) |
| old: unit/engine/test_jit.ml Capture and replay › "warmup, capture, and replay run the kernel" | each replay runs the captured kernels | `J › replay › recorded` (every case) |
| old: unit/engine/test_jit.ml Capture and replay › "replay passes per-call var_vals to the runtime" | each replay binds its call's variables | `J › replay › recorded › plus1` and the other symbolic cases, each run with its own values |
| old: unit/engine/test_jit.ml Capture and replay › "capture and replay retain both signed int64 endpoints" | a variable bound to `Int64.min_int` and `max_int` | dropped here: `Tolk_next_engine.run` binds a variable to an `int`, and binding is `EN`'s |
| old: unit/engine/test_jit_capture.ml Capture and replay (numeric) › "elementwise double: capture computes, replay recomputes" | a replay recomputes on new inputs | `explicit`; each replay's later runs take new inputs |
| old: unit/engine/test_jit_capture.ml Capture and replay (numeric) › "running sum (cumsum) reduces a triangular window" | a cumulative sum replays | `masked_select`, whose selection is a cumulative sum; the kernel's values are Codegen's |
| old: unit/engine/test_jit_capture.ml Capture and replay (numeric) › "sum to scalar hits the shape () output path" | a result of shape () | `sum`, `ones_sum`, `mean` |
| old: unit/engine/test_jit_capture.ml Multi-kernel program › "two chained kernels: double then add-ten through a planned intermediate" | an intermediate planned into an arena | `two_kernels`, `chain_of_three`, `held_constant`; `J › jit_lower › drawn › replays what its capture leaves, run unplanned` (covers an arena holding several buffers) |
| old: unit/frontend/test_jit.ml (414 lines) | the frontend's jit | dropped: frontend (plan L8); rune's compiled call |
| old: unit/gpt2/test_gpt2.ml (233 lines) | GPT-2 end to end | dropped: frontend (plan L8); returns as a rune test in L9 |

### Measures

The default run of `J` takes 1.0 s on an idle machine, `JB` 0.1 s; the slow
cases take 8 s. Coverage of `jit.ml` is 100% (12 of 12 points) from the
default run. ppx_windtrap finds no mutation site in `jit.ml`, so its
mutants were made by hand, each built and run against `J` and `JB`: the
inputs left as buffers (killed by the compiler: `param` unused), the slots
shifted by one, the inputs in reverse order, the held buffers ignored, no
plan, a parameter of one element, a parameter without a device, `JITBEAM`
ignored, and `JITBEAM` defaulting to 0 are killed. Two survive, equivalent:
`~walk:false` in the substitution of the inputs, whose parameters hold no
input to rewrite again, and planning with the inputs held, which the
substitution has already replaced.
