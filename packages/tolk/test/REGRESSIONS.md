# Regressions

Each module's section maps every relevant case of tinygrad's tests for the
module's file to the test that keeps its behaviour, or to the reason it is
dropped. A row names its source as `tinygrad: <file>::<class>::<test>`, and its
outcome as the test's path (`<suite> › <group> › <test>`) or as
`dropped: <reason>`. A module's section starts when its test pass does.

## Helpers

The suite is `Tolk.Helpers` (`helpers/test_helpers.ml`), written `Helpers`
below, and that of the settings, which tinygrad keeps in `helpers.py`, is
`Tolk.Setting` (`setting/test_setting.ml`), written `Setting`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_helpers.py::TestContextVars::test_initial_value_is_set | a setting starts from its default | Setting › declarations › an int setting starts from its default when its variable is unset |
| tinygrad: null/test_helpers.py::TestContextVars::test_cannot_recreate | a name is declared once | Setting › declarations › a name is declared once, whatever the setting's type |
| tinygrad: null/test_helpers.py::TestContextVars::test_new_var_inside_context | a declaration inside a context outlives it | Setting › declarations › a setting declared inside a context stays declared after it |
| tinygrad: null/test_helpers.py::TestContextVars::test_value_across_modules | the registry is one per process, whatever module declares | Setting › declarations › a library setting's name cannot be declared again |
| tinygrad: null/test_helpers.py::TestContextVars::test_assignment_across_modules | assigning `.value` directly | dropped: a setting has no assignment; only `context` changes its value |
| tinygrad: null/test_helpers.py::TestContextVars::test_context_assignment | a context binds for its extent | Setting › context › binds a setting for the extent of its function |
| tinygrad: null/test_helpers.py::TestContextVars::test_unknown_param_to_context | `Context` refuses an unknown name | dropped: a binding holds the setting itself, so an unknown setting cannot be named |
| tinygrad: null/test_helpers.py::TestContextVars::test_nested_context | nested contexts restore in order | Setting › context › restores nested bindings in order; Setting › context › binds within its extent and restores after it, returned or raised |
| tinygrad: null/test_helpers.py::TestContextVars::test_decorator | `Context` as a decorator | Setting › context › binds a setting for the extent of its function (`context` takes the function) |
| tinygrad: null/test_helpers.py::TestContextVars::test_decorator_recursive | recursive contexts restore | Setting › context › restores recursive bindings |
| tinygrad: null/test_helpers.py::TestContextVars::test_context_exit_reverts_updated_values | exit restores the value before entry, not the initial one | Setting › context › restores nested bindings in order |
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
| tinygrad: null/test_device.py::TestDevVar::test_parse | DEV parses targets and prints them back | Helpers › Target › of_string reads a target as tinygrad does (`targets.golden`, every input of the test); the DEV setting is dropped (D95): Setting › startup › starts whatever DEV holds |
| tinygrad: null/test_device.py::TestDevVar::test_target | the target of a device under DEV | dropped: a device's target follows from the device alone (D95) |
| tinygrad: null/test_device.py::TestDevVar::test_dev_arch_override | an arch in DEV reaches the renderer | dropped: the renderer belongs to `Device`'s suite, and there is no DEV (D95) |
| tinygrad: null/test_device.py::TestDevice::test_nonexistent_renderer | "did you mean: 'CLANG'" | dropped: `select_by_name` and its suggestion are not ported (README exclusions); the renderer lookup belongs to `Device`'s suite |
| tinygrad: null/test_device.py::TestCompiler (3 tests) | the compiler cache follows CCACHE | dropped: `Compiler` belongs to `Device`'s suite |
| tinygrad: null/test_hashing.py::TestKeccak::test_shape_keeping | `Tensor.keccak` | dropped: the `Tensor` surface; the frontend is nx |
| tinygrad: null/test_tqdm.py (all) | `tqdm` | dropped: not ported (plan, L0 scope) |

## Dtype

The suite is `Tolk.Dtype` (`dtype/test_dtype.ml`), written `D` below. A
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
| tinygrad: null/test_dtype_spec.py::TestAutoCastType::test_sum | the accumulator of each data type | dropped: `sum_acc_dtype` is not ported (README exclusions); rune's lowering picks a sum's accumulator |
| tinygrad: null/test_dtype_spec.py::TestAutoCastType::test_broadcast_scalar, test_pad_scalar, test_sort, test_int_div_int, test_mean, test_cumsum, test_cumsum_empty, test_matmul, test_linear, test_where_no_scalar, test_where_one_scalar, test_where_two_scalars, test_where_non_bool_cond_raises, test_maximum, test_maximum_const, test_div, test_div_const | result dtypes of Tensor operations | dropped: Tensor surface, rune's lowering. The table they rely on is `D › promotion › least_upper.golden` |
| tinygrad: null/test_dtype_spec.py::TestUnitTypeSpec::test_default_dtype_context | a context overrides the defaults and restores them | `D › defaults › default_float is the float DEFAULT_FLOAT names`; the restoring is `Setting.context`'s, tested in the Helpers suite |
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

## Op

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `uop/__init__.py` (no test file targets it) | `str` and `repr` of an operation are `Ops.<NAME>` | `Tolk.Op › Op › pp is Ops. followed by the name` and the golden's `op` column |
| tinygrad: `uop/__init__.py` (no test file targets it) | the enum's integer values order the toposort | `Tolk.Op › Op order › compare agrees with tinygrad's integer order` |
| tinygrad: `uop/__init__.py` (no test file targets it) | `auto()` values are unique across every `FastEnum` subclass | dropped: OCaml variants of distinct types never compare equal; `X86Ops` belongs to the renderer's section |
| tinygrad: `null/test_upat_compile.py::TestUPatCompile::test_const_folding` | `GroupOp.ALU - {Ops.THREEFRY}` | `Tolk.Op › Set.diff › removes Threefry from alu and keeps the rest`; its compilation of the pattern is dropped with the other seven below |
| tinygrad: `null/test_upat_compile.py::TestUPatCompile` (7 other tests: `test_double`, `test_single`, `test_xpx`, `test_xp0`, `test_bool`, `test_single_c`, `test_range_named`) | the pattern compiler generates code for these patterns | dropped: the pattern compiler is excluded (README); patterns match directly, which the `O › patterns` group tests |
| tinygrad: `null/test_linearizer.py`, `runtime/test_linearizer.py`, `null/test_schedule.py`, `null/test_const_folding.py`, `null/test_dtype_weak.py`, `null/test_pattern_matcher.py` | `u.op in GroupOp.ALU` counts arithmetic | `Tolk.Op › Named sets › hold tinygrad's members` (the ALU column); each test belongs to its own module's section |
| tinygrad: `null/test_graph_rewrite.py`, `null/test_uop_graph.py` | `UPat(GroupOp.All)` matches every operation | `Tolk.Op › Set.mem › of all always holds`; each test belongs to its own module's section |
| tinygrad: `null/test_viz.py::TestViz::test_colored_label`, `test_colored_label_multiline`, `test_inf_loop`, `TestVizGC::test_gc_uop_in_arg` | PYLITERAL and REWRITE_ERROR nodes in the graph viewer | dropped: viz is excluded; `Tolk.Op › Op › has no counterpart for exactly REWRITE_ERROR and PYLITERAL` |

## Ops

The suite is `Tolk.Ops` (`uop/ops/`), written `O` below. A golden
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
| tinygrad: null/test_uop_graph.py::TestMovementOps (2 tests) | `pm_mops` folds reshapes into indices | `Tolk.Prepare › pm_mops › rules ›` "an index of a reshape's added leading axis is its source" (`test_pm_mops_partial_reshape_index_removes_reshape`), "an index of a reshape whose trailing axes change is left as it is" (`…_suffix_mismatch_does_nothing`) |
| tinygrad: null/test_uop_graph.py::TestConstBufferize (2 tests) | `pm_const_buffer_folding` | `Tolk.Rangeify › get_kernel_graph › recorded ›` `setitem_column_kernels.golden` and `setitem_cube_kernels.golden` (Rangeify's rows left by other sections); `bufferize` itself is `O › arguments › reprs.golden` (name=bufferize) and `O › shapes › a stage puts its ranges' sizes in front` |
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
| tinygrad: null/test_graph_rewrite.py::TestCallRewrite::test_body_shared_with_argument | a body is entered only with `enter_calls` | `O › calls › a body is rewritten only under Enter, its arguments always` |
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
| tinygrad: null/test_uops.py::TestDTypeFromUOp::test_const_default_dtype_is_derived | the same under `SPEC=2` | `O › data types › a constant's data type is its literal's`; the `SPEC=2` check is dropped: construction checks no specification (D139) |
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
| tinygrad: null/test_uops.py::TestContiguousViewOffset (7 tests) | `contiguous_view_offset` | `Tolk.Prepare › contiguous_view ›` "storage is its own view from 0" (`test_simple`), "a shrink of leading rows starts at their first element" (`test_shrink`), "an element is a view at its offset" (`test_shrink_to_one`), "an expand repeats" (`test_expand_is_none`), "a pad adds elements" (`test_shrink_invalid`), "a shrink of inner columns skips elements" (`test_strided`); `test_2d` by the law `contiguous_view › laws`; `contiguous_view` is Prepare's (D4) |
| tinygrad: null/test_uops.py::TestBitcastBufferView, TestLocalAccess, TestAssembly | renderer output | the renderers' sections |
| tinygrad: null/test_uops_stats.py (all) | kernel cost estimates | Renderer's section (`Estimates.from_uops`) and the engine's (`estimate_uop`) |
| tinygrad: runtime/test_uops.py::TestFloatUOps, TestNonFloatUOps, TestBoolUOps | a device computes each operation as Python does | the Python side is `O › exec_alu › exec_alu_values.golden`; the device side is an Exec test of the renderer layers |
| tinygrad: runtime/test_uops.py::TestBitcastBufferView, TestLocalAccess, TestZeroRange, TestUOpPrograms | kernels run on a device | Exec tests of the renderer layers |

### L2: null/test_tensor_uop_mixin.py, test_tensor_uop_representation.py

A `Tensor` and a UOp built by the same calls must be the same node. tolk
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
| tinygrad: null/test_tensor_uop_representation.py (5 tests) | a realized `Tensor` is a BUFFER | the engine's section (L7): realization is `tolk.engine`'s |

## Spec

The suite is `Tolk.Spec` (`uop/spec/`), written `S` below.
`verdicts.golden` gives, for 273 nodes, the verdict of each of the six
specifications (`True`, `False`, or `None` when no rule decides); its test is
`S › verdicts › verdicts.golden › <case>`. `bounds.golden` gives the shared
verdict of 41 accesses with `CHECK_OOB` off and on, and `failures.golden`
the message of `type_verify` under `tensor`, `program`, and `tensor` without
entering calls. Each table's nodes are the sources of the sink of its
`_nodes.golden`, in row order.

tinygrad hands the accesses its bounds do not prove to z3, which tolk
does not have (README exclusions); the generator stands in a z3 that cannot
decide, so each `checked` verdict is tinygrad's under the ruling that an
unproved access fails.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `uop/spec.py` spec_shared, spec_tensor, spec_program, spec_full (without spec_hcq, see the README), spec_kernel_graph (no test file targets the rules) | each rule's verdict on nodes it accepts, rejects and declines | `S › verdicts › verdicts.golden` (every row) |
| tinygrad: `uop/spec.py` type_verify | the message `UOp verification failed at {i} on {op} {dtype} {n} {srcs} {arg}`, the position in toposort order, `enter_calls` | `S › type_verify › failures.golden`; `S › type_verify › fails at the first node, sources first, that the specification does not accept` |
| tinygrad: `uop/spec.py` type_verify (`DEBUG >= 3`) | the graph's listing prints before the failure | `S › type_verify › prints the graph when DEBUG is 3 or more, before failing › debug_listing.golden`; `S › type_verify › prints nothing when DEBUG is below 3` |
| tinygrad: `uop/spec.py` type_verify (`SPEC > 1`: test_pyrender) | a checked graph round-trips through pyrender | dropped: pyrender is not ported (README exclusions) |
| tinygrad: `uop/spec.py` validate_index | with `CHECK_OOB`, vmin and vmax prove an access; multi-index accesses and the invalid index skip the check | `S › bounds › bounds.golden`; `S › bounds › an access passes iff the bounds of its index lie within the storage`; `S › bounds › without CHECK_OOB, every access passes` |
| tinygrad: `uop/spec.py` validate_index (z3 on the gate) | a gate or a mask proves an access | dropped: z3 is excluded (README); pinned the other way by `S › bounds › a gate never changes whether an access passes` |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC > 1`) | a created node is checked against spec_full, with `CHECK_OOB` off | `S › construction › builds a node that breaks the full specification, whatever SPEC`; `S › verdicts` pins `Spec.full` |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC <= 1`, cache hit) | no check below 2; a cached node returns before the check | `S › construction › builds a node that breaks the full specification, whatever SPEC` |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC > 2`) | the shape of each created node is computed | dropped: construction computes no shape (D139) |
| tinygrad: `uop/ops.py` UOpMetaClass.__call__ (`SPEC > 3`) | pyrender round trip | dropped: pyrender is not ported (README exclusions) |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_int_binop (the `type_verify([bad], spec_shared)` lines) | bitwise operations on float32 and weak floats fail the shared spec | `S › verdicts › verdicts.golden` (an and of float32, an and of weak floats); `type_verify`'s list form is dropped: every tinygrad caller passes a graph, and a one-node list is the node's graph with sources that pass |
| tinygrad: null/test_dtype_weak.py::TestWeakPromotion::test_weak_int_binop (the dtype lines) | shift and bitwise data types | Ops' section (`dtype_of`) |
| tinygrad: null/test_uops.py::TestUOpMethod::test_invalid_dtype_and_consumers (the `type_verify(u, spec_shared)` loop) | Invalid matches any type in a stack, a sum, a where, both sides of a comparison, an index | `S › verdicts › verdicts.golden` (a stack of float32 and the invalid constant, a sum of float32 and the invalid constant, a where over float32 and the invalid constant, a cmplt of the invalid constant and float32, a cmplt of float32 and the invalid constant, an index by the invalid constant) |
| tinygrad: null/test_uops.py::TestUOpMethod::test_invalid_dtype_and_consumers, test_remove_invalid_stack_lanes (the `spec_program` lines) | `pm_remove_invalid`'s output is a program | dropped here: `pm_remove_invalid` is codegen (L4), whose suite checks its output with `Spec.program` |
| tinygrad: null/test_uops.py::TestUOpMethod::test_const_default_dtype_is_derived (`SPEC=2`) | constants build under the construction check | dropped: construction checks no specification (D139); the data types are Ops' section |
| tinygrad: null/test_uops.py::TestUPatHelpers::test_location | spec_shared's first pattern is located in spec.py | dropped: pattern locations are match tracking (README exclusions) |
| tinygrad: null/test_uop_symbolic.py::TestMoveWhereOnLoad::test_bool_index_preserves_dtype | the rewrite's output passes spec_shared | `Tolk.Symbolic › tinygrad › tests.golden › TestMoveWhereOnLoad.test_bool_index_preserves_dtype` |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_const_index | constant indexes 0, 15, 16, 42 into 16 elements | `S › bounds › bounds.golden` (a load at the first element, … at the last element, … one past the last element, … far past the last element) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_variable_index | variables over 0..15, 0..20, -5..10 | `S › bounds › bounds.golden` (a load over the elements, … past the last element, … before the first element) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_range_with_mask, test_variable_with_mask, test_gated_store, test_or_in_mask, test_xor_in_mask, test_float_cast_in_mask, test_bool_cast_in_mask, test_load_as_index, test_load_bool_as_mask, test_gated_local | a mask proves an access in bounds | `S › bounds › bounds.golden` (a load over a guarded range, … a guarded variable, a store over a guarded variable, a load from a local buffer within its end: all fail, as z3 cannot decide); the rest dropped: z3 is excluded (README) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_floordiv, test_mod, test_shr, test_and, test_max | vmin and vmax of floor division, modulo, shifts, masks and maximum prove an access | `S › bounds › bounds.golden` (a load over half a longer range, … modulo the length, … modulo more than the length, … masked to the length, … shifted into the buffer, … shifted past the buffer, … a clamped variable, … a variable clamped too little) |
| tinygrad: null/test_validate_oob.py::TestValidateOOB::test_shl, test_or, test_or_negative, test_float_cast_in_index, test_bitcast_in_index, test_load_from_shrink_as_index | bounds that z3 proves beyond vmin and vmax | dropped: z3 is excluded (README); the bounds these nodes have are Ops' section |
| tinygrad: null/test_validate_oob.py::TestShiftBounds (7 tests) | `validate_index_with_z3` | dropped: z3 is excluded (README) |
| tinygrad: runtime/test_tensor_variable.py::TestTensorVariable (`CHECK_OOB`) | a shrink by a variable past its dimension fails at realize | dropped: needs the schedule and the runtime (L6, L9) |
| tinygrad: null/test_const_folding.py, null/test_graph_rewrite.py, null/test_simplify_valid_idx.py, runtime/test_arange.py, runtime/test_linalg.py, runtime/test_tensor_cores.py, runtime/test_wait_loop.py, runtime/test_uops.py | set `SPEC` or `CHECK_OOB` for their own tests | dropped: they test other modules under a setting |

## Transcendental

Outcomes are tests of the suite `Tolk.Transcendental`. The graph goldens
pin every operation, constant and data type of each function for Float32 and
Float64, and of the reductions and `xpow` for Float16 too, and `values.golden`
and `pow_values.golden` pin the value of tinygrad's graphs, evaluated node by
node, at about 260 inputs per function and type: special values, the
reductions' boundaries and values drawn from their bits. The accuracy
properties hold each function within one ulp of the correctly rounded result
in every float from Float16 up.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_payne_hanek_reduction` | `(r, q)` of `12π + 0.1`, `12π`, `12π - 0.1` | reductions › payne_hanek_reduction removes quarter turns (the three inputs); graphs › `payne_hanek_*.golden` |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_cody_waite_reduction` | `(r, q)` of `12π + 0.1` | reductions › cody_waite_reduction removes quarter turns; graphs › `cody_waite_*.golden` |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_frexp` | mantissa and exponent of ±1, ±2, 5, 1000 in Float64, the sign dropped | reductions › frexp splits a double into a mantissa in [0.5, 1) and an exponent; reductions › frexp keeps the sign of a float's mantissa; graphs › `frexp_*.golden` |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_rintk` | rounding of 0, ±5, ±5.5, ±5.999 | bits › rintk rounds halves away from zero; bits › rintk gives the signed integer of its float's width |
| tinygrad: `null/test_transcendental_helpers.py::TestTranscendentalFunctions::test_pow2if` | `2^q` for q in ±{0, 1, 2, 10, 63} | bits › pow2if is two to an integer; bits › pow2if gives the float of its integer's width; graphs › `pow2if_*.golden` |
| tinygrad: `null/test_transcendental.py::TestTranscendentalSchedule` (3 tests) | sin, log2 and exp2 of sums fuse into one kernel under `TRANSCENDENTAL=2` | dropped: fusion is the scheduler's, its suite (L6); the rewrite the setting forces is patterns › `patterns_all_forced_float.golden` |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalMath::test_float64`, `test_float32`, `test_float16` | exp, log and sin agree with numpy within atol/rtol 3e-2/1e-5, 2e-5/1e-5 and 1e-2/5e-3 (sin of Float64 below 1e8) | accuracy › exp2, log2 and sin of each type are within an ulp, xsin ~fast below 30 (on exp2 and log2: `Tensor.exp` and `Tensor.log` scale them in the frontend, nx); values › `values.golden` |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalMath::test_exp_near_inf` | exp just below overflow is finite | values › `values.golden` (xexp2 at 1023.9, 127.9, 15.9, 22.9 and the thresholds); special values › xexp2 of float overflows at 128 and underflows below -149 |
| tinygrad: `runtime/test_transcendental.py::TestFromFuzzer::test_sin` | sin at ±25, ±35, 30, 0, π/2 within 1 ulp of 1.0, 2π within 1.5 | accuracy › sin of `<type>` is within an ulp, whose examples they are |
| tinygrad: `runtime/test_transcendental.py::TestFromFuzzer::test_log2` | log2 of ±tiny × {1, 1e10, 1e20, 1e30}, 0 and 9e-7 within 1 ulp of 1.0 | accuracy › log2 of `<type>` is within an ulp, whose examples they are |
| tinygrad: `runtime/test_transcendental.py::TestFloat16Log2::test_float16_log2_basic` | Float16 log2 of 1 to 1000 | accuracy › log2 of half is within an ulp |
| tinygrad: `runtime/test_transcendental.py::TestFloat16Log2::test_float16_log2_special` | Float16 log2 of inf, 0, -1, NaN | special values › log2 half inf/0/-1/nan rows |
| tinygrad: `runtime/test_transcendental.py::TestFloat16Log2::test_float16_log2_denormal` | Float16 log2 of 1e-4, 6e-5, 1e-5 | accuracy › log2 of half is within an ulp, which draws its subnormals |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalVectorized::test_exp2_vectorized`, `test_log2_vectorized`, `test_sin_vectorized` | the functions on vectors of widths 1 to 128 | dropped: vector widths are the devectorizer's and the executor's (L4, L7); the scalar values are `values.golden` and the accuracy properties |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalVectorized::test_pow_vectorized` | pow of (0.001, 200) to (-10, 10) | xpow › its four tests; values › `pow_values.golden`; vectors dropped as above |
| tinygrad: `runtime/test_transcendental.py::TestTranscendentalVectorized::test_sqrt_vectorized` | sqrt of (0, 100) | patterns › a rewritten operation computes it, rounded to its type; vectors dropped as above |
| tinygrad: `runtime/test_dtype_alu.py::TestDTypeALU` (the unary tests of bfloat16 and the 8-bit floats) | exp2, log2, sin and sqrt of the narrow floats on a device | patterns › a rewritten operation computes it, rounded to its type; patterns › `patterns_none_{bfloat16,fp8e4m3,fp8e5m2fnuz}.golden`; the device run is the executor's (L7) |
| tinygrad: `null/test_graph_rewrite.py::TestEdgeCasesAndSpecialOperations::test_full_graph_rewrite_transcendental_edge_cases` | `log2(-1)` folds to NaN | dropped: constant folding is Symbolic's, its suite; the decomposition's value is special values › xlog2 … -1 is nan |
| tinygrad: `null/test_dtype_weak.py::TestWeakPromotion::test_weak_transcendentals` | `Tensor.exp` of a Python number is weak | dropped: the frontend's promotion (nx), and `Uop.Ops`' data types |
| tinygrad: `transcendental.py` `get_transcendental_patterns` (no test) | which operations are rewritten, the Float32 detour of the narrow floats, SQRT to `xpow` | patterns › `patterns_*.golden` (none, all, all forced, exp2+log2, sqrt of bfloat16); patterns › a target with every operation keeps the graph; patterns › a target without the operations is left none of them |
| tinygrad: `transcendental.py` asserts `d.dtype in TRANSCENDENTAL_DTYPES`, dictionary lookups (no test) | a function of the wrong type fails | types › `<function>` refuses a node of another type, Float16 included for the functions; types › pow2if refuses an integer of another width; types › xpow takes the other floats |
| tinygrad: `transcendental.py` `exponent_bias` (no test) | the bias of each float, fnuz one more | bits › `exponent_biases.golden` |
| tinygrad: `transcendental.py` `shl`, `shr` (no test) | multiplication and floor division by `2^n` | bits › shl multiplies and shr floors a division by a power of two; bits › shl and shr refuse a negative count; graphs › `shifts.golden` |

## Tc

Outcomes are tests of the suite `Tolk.Tc`. `tensor_cores.golden` holds a
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

## Opt

The suite is `Tolk.Opt` (`codegen/opt/opt/`), written `P` below.

tinygrad orders two optimisations as the tuples `(op, axis, arg)`, and cannot
compare two splits of one axis and amount into different targets, a host
accident recorded in the README's CPython rows: `comparisons.golden` leaves
those pairs out, and `P › order › compare is a total order` covers them. A split that is not from
the top is `(amount, target)` in tinygrad whether or not it spells the
`False`; the heuristic never spells it, and tolk prints it the same way.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | `Opt.__repr__` | `P › printing › pp is tinygrad's repr` (every row); `P › printing › a split that is not from the top prints without its flag` |
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | the axis an optimisation acts on | `P › printing › axis is the axis tinygrad prints` (every row) |
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | `dataclass(order=True)` and `OptOps.__lt__` | `P › order › compare agrees with tinygrad where tinygrad orders` (every row); `P › order › compare is a total order`; `P › order › compare is 0 exactly on equal optimisations`; `P › order › splits into different targets are different` |
| tinygrad: `codegen/opt/__init__.py` (no test file targets it) | `check` and `KernelOptError` | Postrange's section: a refusal is `Scheduler.apply_opt`'s `Error`, and `apply_opts` raises it as `Invalid_argument`, so `Opt` holds the types alone |
| tinygrad: `runtime/test_kernel_opts.py`, `runtime/test_linearizer.py`, `null/test_linearizer.py`, `runtime/test_opt_gemm.py`, `runtime/test_tensor_cores.py`, `null/test_uops.py`, `null/test_uops_stats.py`, `null/test_gen_float4.py`, `null/test_linearizer_rewrite.py`, `runtime/test_custom_kernel.py`, `null/test_custom_kernel.py` | optimisations applied to kernels, and the `KernelOptError` of those that do not apply | Postrange's section: `apply_opt` makes the checks and returns a refusal as an `Error` |

## Uop_weak

The suite is `Tolk.Uop_weak` (`uop/uop_weak/`), written `W` below. A golden
check is named after its golden, which holds a graph and its rewrite by one
pass.

| Source | Behaviour | Outcome |
|---|---|---|
| DIVERGENCES D44 | an integer cast of a weak expression computes in integers, a 64-bit unsigned one included | `W › pm_commit_weak › a 64-bit unsigned cast of a weak expression keeps its value past a float's precision`, `W › laws › pm_commit_weak computes an integer cast in integers` |
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
| tinygrad: null/test_dtype_weak.py::TestWeakStorageBoundary::test_weak_has_no_storage | a `Tensor` of a weak type has no storage | dropped: `Tensor` surface; `Call.param` and `Ops.new_buffer` refuse weak types in `Ops`' section |
| tinygrad: null/test_dtype_weak.py::TestNoRedundantWide (2 tests) | a kernel computes in 64 bits only where its bounds need it | dropped: the whole codegen pipeline (`full_rewrite`), Codegen's section (L4) |
| tinygrad: runtime/test_dtype_weak.py (every test) | weak `Tensor` values realized on a device: defaults, stacked and truncating weak casts, bounds through movements, storage | dropped: realizing `Tensor`s on a device, the engine's (L7) and rune's lowering; the conversions they check are `W › pm_lower_weak › lower_stacked_weak_casts_as_two_conversions.golden`, `W › pm_lower_weak › lower_a_weak_int_cast_of_a_float_as_a_conversion.golden` and `W › laws › pm_lower_weak keeps the values of what it lowers` |
| tinygrad: null/test_uops.py::TestLowerIndexDtype::test_gated_shrink_lowers_to_selected_width | a gated shrink lowers to the width its bounds need, and no weak width is left but a literal's | `W › pm_lower_weak › lower_a_gated_shrink_to_the_width_its_bounds_need.golden`; `W › laws › pm_lower_weak leaves no weak width but a literal's` |
| tinygrad: null/test_uops.py::TestLowerIndexDtype::test_reg_buffer_size_lowers | a register buffer's size lowers | `W › pm_lower_weak › lower_a_register_buffer_size.golden` |
| tinygrad: null/test_simplify_valid_idx.py::TestImageSimplification::test_drop_gate_committed_in_the_index_pass | committing inside the index pass leaves the gate's copy of an expression shared | dropped: runs with `indexing_simplify` on image loads, Codegen's section (L4) |

## Movement

The suite is `Tolk.Shape.mop_cleanup` (`uop/movement/`, after tinygrad's
`uop/movement.py`), written `M` below. A
golden check is named after its golden, which holds a graph and its cleanup. A
test of offsets that are nodes simplifies them with Symbolic's rules.

tinygrad tests `mop_cleanup` only through the passes that include it
(`symbolic_simple`, `earliest_rewrites`, the codegen pipeline and the jit's
input capture); those tests belong to their passes' sections.

## Divandmod

The suite is `Tolk.Shape.div_and_mod_symbolic` (`uop/divandmod/`, after
tinygrad's `uop/divandmod.py`), written `DM` below.
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

## Render

A test that renders after simplifying does so with Symbolic's rules.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_uops.py::TestUOpRender::test_render_ssimplified_marg_outside_toposort` | a movement's size, simplified to a node the graph never held, is still written, and a range after it is its name | `Tolk.Render › render after simplifying › symbolic_rendered.golden › case=shrink_of_simplified_offset`, `case=range_after_shrink` |
| tinygrad: `null/test_uops.py::TestUOpRender::test_render_vectorize_empty`, `test_render_vectorize_same`, `test_render_vectorize_different` | a stack is its sources between braces, `{}` when empty | `Tolk.Render › render › expressions_rendered.golden › case=stack_empty`, `case=stack_same`, `case=stack_different` |
| tinygrad: `null/test_uops.py::TestUOpRender::test_render_vectorize_empty_simplified`, `test_render_vectorize_same_simplified`, `test_render_vectorize_different_simplified` | the same after simplifying | `Tolk.Render › render after simplifying › expressions_rendered.golden` (the `simplified` column) |
| tinygrad: `null/test_helpers.py::TestProd::test_variable`, `test_variable_order` | the product of a variable and integers writes as `(a*12)`, whatever the order | `Tolk.Render › srender › writes a node as render does`; the product itself belongs to `Ops` |
| tinygrad: `null/test_helpers.py::TestCeilDiv::test_symbolic`, `test_symbolic_negative_offset` | `ceildiv` of a node writes as `((v+5)//6)` | dropped: `ceildiv` on nodes has no reader among the ported files; `Helpers.ceildiv` is on integers |
| tinygrad: `null/test_uop_symbolic.py` (`helper_test_variable` and the eight `.render()` asserts), `null/test_simplify_valid_idx.py` (the `.render()` asserts) | a rewrite gives the expected expression | dropped here: they test the symbolic rules, with `render` as the printer: `S › tinygrad › tests.golden › <Class>.<test>` |
| tinygrad: `null/test_uop_graph.py` (`print(sink.render())`) | none, a debug print | dropped: no claim |
| tinygrad: `null/test_viz.py` (`ret.render()` in `rewrite_group` names) | none about `render` | dropped: viz is excluded |
| tinygrad: `null/test_uops.py`, `null/test_encodings.py`, `null/test_renderer_failures.py` (`renderer.render(uops)`) | a device renderer's source | dropped here: those are `Renderer.render`, not `uop/render.py`; each belongs to its renderer's section |
| tinygrad: `uop/render.py` `renderer` (no test file covers every rule) | every rule: storage, ranges, loops, constants, casts of constants, casts, the unary and ternary forms, movements, the twelve infix operations and their precedence, indices, stages, loads, stacks, and the repr of any other node | `Tolk.Render › render › expressions_rendered.golden` (145 cases), `unrendered_rendered.golden`; `an expression reads back as the tree it writes` (the parenthesis law); `tags are not written` |
| tinygrad: `uop/render.py` `print_uops` (no test file) | the listing's columns, the sources as positions, quoted constants or `--`, the argument as `str` prints it, colour, and columns that overflow | `Tolk.Render › pp_uops › program_listing.golden`, `program_listing_colored.golden`, `partial_listing.golden`, `wide_listing.golden`, `wide_listing_colored.golden`, `an empty list prints nothing`, `prints a line per node, numbered from 0, the last not ended` |
| tinygrad: `uop/render.py` `pretty_print` | a node's repr | dropped here: `Ops.pp`, in `Ops`' suite |
| tinygrad: `uop/render.py` `renderer_infer`, `pyrender` | Python source | dropped: excluded (README Exclusions) |

## Renderer

The suite is `Tolk.Renderer` (`renderer/renderer/`), written `R` below.
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
| tinygrad: `null/test_device.py::TestCompiler::test_compile_cached_disabled` | with `CCACHE=0` nothing is cached | `R › Compiler › compile_cached compiles every time with ccache off` |
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

## Decomp_op

The suite is `Tolk.Decomp_op` (`codegen/decomp/decomp_op/`), written `DO`
below. A graph golden holds the sink of its inputs and that sink rewritten to a
fixed point by one matcher alone. `fast_idiv_grid.golden` holds tinygrad's
`fast_idiv` of every integer type over 13 bounds and 28 divisors (the uint64
edges, 2^64 and 2^70 + 1 included), with and without a wider type, and the
laws check every quotient against `Bigint.div` in the interpreter.

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

## Decomp_dtype

The suite is `Tolk.Decomp_dtype` (`codegen/decomp/decomp_dtype/`),
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

## Symbolic

The suite is `Tolk.Symbolic` (`uop/symbolic/`), written `S` below. Its
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
its own `TestModularWraparound` marks as expected failures; tolk reads
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

## Gpudims

The suite is `Tolk.Gpudims` (`codegen/gpudims/`), written `GD` below.
`grouped_dims.golden` holds every case of tinygrad's test, 392 sizes drawn
from a fixed seed against the bounds of real targets, and three edge cases: two
sizes merged to fill one bound exactly, sizes of 2^61 split across bounds of
2^60 and 2^61, and two sizes of 2^32 that one bound of 2^31 - 1 cannot hold.
Each row gives the hardware indices and their sizes, and each loop's index as
an expression, or the exception tinygrad raises. tinygrad proves each case
one-to-one with z3; here `GD › grouped_dims › grouped_dims numbers each
iteration of the golden's cases once` enumerates every launch of each case up
to 2^12 iterations and checks that each iteration is numbered exactly once, and
that each hardware index is within its bound, the first axis excepted when the
last one's divisors move onto it.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_grouped_dims` | sizes of 20 cases, reversal, splits, merges, and three `RuntimeError`s | `GD › grouped_dims › grouped_dims.golden` (the first rows); the z3 proof is `GD › grouped_dims › grouped_dims numbers each iteration of the golden's cases once` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_grouped_direct_dims_are_special` | loops 2 and 3 of (2,3,4,5) are hardware indices | `GD › grouped_dims › a loop that keeps its own axis is that hardware index` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_grouped_dims_high_rank` | 4 to 6 loops onto 2 or 3 axes; no bounds leaves every loop its own index | `GD › grouped_dims › grouped_dims.golden` (the four rows); `GD › grouped_dims › without bounds, every loop is its own hardware index` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_symbolic_dims_cross_launch_limit` | `(1, n)` with `n` in [1, 4] under (4, 3) | `GD › grouped_dims › symbolic sizes › grouped_symbolic_crosses_a_limit_by_merging.golden` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_global_prod_max` | `global_prod_max` bounds workgroups by threads | `GD › add_gpudims › add_gpudims_bounds_globals_by_threads.golden`, `add_gpudims_bounds_globals_by_merged_threads.golden`, `add_gpudims_bounds_globals_by_threads_alone.golden` |
| tinygrad: `null/test_gpudims.py::TestGroupedDims::test_max_sizes_none` | no bounds | `GD › grouped_dims › grouped_dims.golden` (`None` rows) |
| tinygrad: `codegen/gpudims.py` `add_gpudims`, `pm_device_to_var` (no test) | globals, locals, warps, masks, device ranges | `GD › add_gpudims › add_gpudims_*.golden` (17 kernels); `GD › add_gpudims › add_gpudims_declines.golden`, `add_gpudims_failures.golden` |
| tinygrad: `codegen/gpudims.py` `_split_dims` on a symbolic size or fewer than three bounds (`ValueError`, `AssertionError`, `IndexError`) | README, CPython rows: "cannot limit dim" | `GD › grouped_dims › grouped_dims.golden` (the `IndexError` rows); `GD › grouped_dims › symbolic sizes › grouped_symbolic_failures.golden` |
| tinygrad: `codegen/gpudims.py` `add_gpudims` with a symbolic warp (`ValueError`) | README, CPython rows: the warp's bound is its size's upper bound | `GD › add_gpudims › add_gpudims_symbolic_warp.golden` |

## Gater

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `codegen/late/gater.py` (no test file covers every rule) | the two-index rules, the first-index rules on `INDEX` and `SHRINK` for loads and stores, the alternative of a gated load, the four alternatives of the selection fold and its negated form, and the accesses no rule matches | `Tolk.Gater › pm_move_gates_from_index › moves.golden` (32 cases, `*_kept` for the accesses left alone) |
| tinygrad: `codegen/__init__.py` `full_rewrite_to_sink` (the "move gates from index" pass) | the pass on compiled kernels: padded loads on the CPU and on CUDA, a padded convolution, a gated store into workgroup memory | `Tolk.Gater › pm_move_gates_from_index › pad_moved.golden`, `pad_value_moved.golden`, `conv_moved.golden`, `pad_cuda_moved.golden`, `sum_group_moved.golden` |
| tinygrad: `null/test_uops.py::TestGatedStoreRewrite::test_tiny_gate_store`, `test_gate_some_stores`, `test_merge_ifs_alt` | a store through a gated index is gated, then becomes `IF`/`STORE`/`ENDIF` | the gate: `Tolk.Gater › pm_move_gates_from_index › moves.golden › case=index_store`; the `IF` and `ENDIF`: dropped here, `codegen/__init__.py`'s `pm_linearize_cleanups`, in `Codegen`'s suite |
| law | no load or store through a first index gated by `Invalid` remains; the pass is idempotent | `Tolk.Gater › laws › no access through a first index gated by invalid remains`, `moving the gates is idempotent` |

## Linearizer

The kernel goldens are compiled by tinygrad (`gen/codegen/late/linearizer.py`
lists them) and recorded at the pass each test applies: `<kernel>.golden` at
`linearize`, `<kernel>_unsplit.golden` at the final rewrite (for
`pm_split_ends`), `<kernel>_unchained.golden` at `pm_add_control_flow`.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `codegen/late/linearizer.py` `linearize` (no test file pins the order) | the order of a compiled kernel: run counts, the kind priorities, parameters by slot, the structural tie-break | `Tolk.Linearizer › linearize › <kernel>_linear.golden` for 18 kernels (matmul with and without opts, softmax, a padded convolution, two reductions, a transpose, a symbolic size, a dependent loop bound, a CUDA reduction through workgroup memory with a barrier, a CUDA matmul with locals, four runtime kernels, four do-while loops) |
| tinygrad: `linearize` priorities | each rule of the ideal order on a graph built for it | `Tolk.Linearizer › linearize › orders.golden` (16 cases) |
| tinygrad: `helpers.py` `TUPLE_ORDER=0` | without the structural tie-break, equal priorities keep their topological order | `Tolk.Linearizer › linearize › matmul_linear_toposort.golden`, `conv_linear_toposort.golden`, `orders_output_toposort.golden` |
| tinygrad: `linearize` `DEBUG_LINEARIZE` | each node printed with its position, operation, ranges and priority | `Tolk.Linearizer › DEBUG_LINEARIZE › debug_linearize.golden` (a run of its own, with the variable set); `linearize › without DEBUG_LINEARIZE, nothing is printed` |
| tinygrad: `do_split_ends`, CPython's `TypeError` sorting range arguments that tie up to an axis type or where one identity is a prefix of the other (README CPython row) | the ranges nest by identity, a prefix first, then by axis type, the greatest innermost | `Tolk.Linearizer › pm_split_ends › ranges of one identity and different axis types nest by axis type` |
| tinygrad: `linearize`, CPython's `tuplize` comparison (README CPython row) | nodes whose sources differ only by a tag | `Tolk.Linearizer › linearize › nodes whose sources differ only by a tag are ordered by their later sources` |
| tinygrad: `CFGContext`, `pm_add_control_flow` | siblings chained by dependency then position, a loop's first child after its range, the sink's first child first | `Tolk.Linearizer › pm_add_control_flow › chains.golden` (10 cases), `Tolk.Linearizer › pm_add_control_flow › <kernel>_chained.golden` for 11 kernels |
| tinygrad: `CFGContext` (`assert y.src[1] not in x.backward_slice_with_self`) | a range that would run after a loop depending on it is refused | `Tolk.Linearizer › pm_add_control_flow › a range that would run after a loop depending on it is rejected` |
| tinygrad: `pm_split_ends` | an end split into one end per range, the greatest innermost; ranges of other sources; repeated ranges once; an end of no range | `Tolk.Linearizer › pm_split_ends › splits.golden` (12 cases), `Tolk.Linearizer › pm_split_ends › <kernel>_split.golden` for 8 kernels |
| tinygrad: `null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_dependent_loop_bound` | a loop bounded by a load: ranges, stores and ends in order, the ends closing the ranges in reverse | `Tolk.Linearizer › linearize › dependent_loop_bound_linear.golden`, `Tolk.Linearizer › pm_add_control_flow › dependent_loop_bound_chained.golden` |
| tinygrad: `null/test_linearizer_rewrite.py::TestLinearizerRewrite::test_reduction`, `test_arange`, `test_kernel_info` | a program compiles; its name and opts | dropped here: `to_program` end to end, `Codegen`'s suite |
| tinygrad: `null/test_linearizer.py::TestLinearizer` (7 tests), `TestLinearizerRenderers` (2 tests) | load dedup, zero folding, accumulator types, reduce upcasts, locals with upcasts, all through `to_program` | dropped here: the expander, devectorizer and opts, in `Codegen`'s and `Postrange`'s suites |
| tinygrad: `null/test_linearizer_failures.py::TestLinearizerFailures::test_fail_1`, `runtime/test_linearizer_dumb.py::TestLinearizerFailure::test_failure_beam_mnist` | a kernel compiles through `to_program` | dropped here: `Codegen`'s suite |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_late_bias_load` | a bias is loaded after the reduction's loop ends | `Tolk.Linearizer › linearize › late_bias_load_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_two_nested_range_alt_indexing` | an ALU and a load sit between the two ranges | `Tolk.Linearizer › linearize › two_nested_range_alt_indexing_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_range_outer_op_before_phi` | one load is placed before the range | `Tolk.Linearizer › linearize › range_outer_op_before_phi_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer::test_simple_unroll_no_between_phi_dependencies` | register stores fed by ALUs sit inside the loop | `Tolk.Linearizer › linearize › simple_unroll_linear.golden` |
| tinygrad: `runtime/test_linearizer.py::TestLinearizer` (the 16 other tests) | buffer dedup, range collapse, casts, launch dimensions, folds, grouped stores, on a device | dropped here: scheduling, the expander, `Gpudims` and execution, in their modules' suites |
| tinygrad: `runtime/test_wait_loop.py::TestWaitLoop::test_wait_loop`, `test_nested_loop_in_range`, `test_two_sequential_loops`, `test_loop_in_loop`, `test_wait_loop_spec` | do-while loops (`BACKEDGE`), alone, in a range, in sequence and nested | `Tolk.Linearizer › linearize › wait_loop_linear.golden`, `nested_loop_linear.golden`, `two_loops_linear.golden`, `loop_in_loop_linear.golden`, and their `_chained` goldens under `pm_add_control_flow`; their execution belongs to the execution tests |
| tinygrad: `runtime/test_wait_loop.py::TestWaitLoop::test_loop_carried_registers`, `test_register_pressure_loop`, `TestVolatileLoops::test_async_wait_ext` | register pressure and a volatile host signal | dropped here: register allocation (ISA, excluded) and execution |
| law | `linearize` is a topological order of the sink's nodes, the sink last, with and without the tie-break; after `pm_split_ends` and `pm_add_control_flow`, loops nest and a range's body runs inside its loop; `pm_split_ends` leaves one range per end, closes the same ranges, and is idempotent | `Tolk.Linearizer › laws` (7 properties over generated loop trees, the kernel goldens as examples) |

## Simplify

The suite is `Tolk.Simplify` (`codegen/simplify/`), written `SI` below.
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

## Coalesce

The suite is `Tolk.Coalesce` (`codegen/late/coalesce/`), written `C`
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

## Postrange

The suite is `Tolk.Postrange` (`codegen/opt/postrange/`), written `PR`
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
| tinygrad: `codegen/opt/postrange.py` `args_from_ast`, BEAM search | device buffers for a search | ported as `Tolk_engine.slots`, of a linked schedule: `EN › timing › run takes the slots of a linked schedule`, `› a program whose buffers skip a slot links and runs on its slots`, `› a time allocates and loads nothing`; `apply_opts` takes the search as `~beam`, `PR › apply_opts › searches with beam instead, when given` |
| tinygrad: `codegen/opt/postrange.py:112` the DSP upcast cap | | dropped: DSP is excluded (README) |
| tinygrad: `postrange.py:114` a shared-memory check whose symbolic size cannot be decided raises `ValueError` | | `group_maybe_beyond_symbolic_shared_memory`, refused with `Invalid_argument`; `group_within_symbolic_shared_memory` fits |

## Heuristic

The suite is `Tolk.Heuristic` (`codegen/opt/heuristic/`), written `H`
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

## Search

The suite is `Tolk.Search` (`codegen/opt/search/`), written `S` below,
with `Tolk.Search on the host` (`test_search_exec.ml`, slow), written
`SH`. `actions.golden` is the table of actions and `actions_padto.golden` the
table under `BEAM_PADTO=1 TC=2`. `candidates.golden` holds the
positions of the actions `get_kernel_actions` applies for 14 kernels on the Clang, Metal,
CUDA sm_89 and HIP gfx1100 renderers (`S › candidates.golden`, 52 rows).
`searches.golden` holds what `beam_search` chooses, and how many times it
measures, for 14 searches under a measurement that computes each program's time
from its optimisations and its launch, compiled with no compiler, by tinygrad's
search as D117 departs from it (`S › a search
chooses what tinygrad's chooses`; the GPU rows are slow). The environment
variables the search reads once run in a process of their own (`S ›
BEAM_PADTO=1 TC=2 ...`, the suite's dune).

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/test_search.py::TestSearch::test_beam_symbolic_kernel` | a beam search of a symbolic kernel on the CPU applies optimisations | `SH › a searched kernel computes what its unoptimised kernel computes › symbolic` (through `Codegen.to_program ~beam`, linked and timed by `Tolk_engine.link_call` and `Tolk_engine.time` with optimised programs' samples scaled down: under D117 a search of a kernel this small answers the kernel when no candidate beats its samples); the same kernel's candidates, `candidates.golden` `symbolic` |
| tinygrad: `codegen/opt/search.py` `actions` | the table, in order; `BEAM_PADTO`, `TC`; `TC_OPT` at its search default, `2` | `S › the actions are tinygrad's, in order`; `S › BEAM_PADTO=1 ... › the actions are tinygrad's, pads included` |
| tinygrad: `codegen/opt/search.py` `get_kernel_actions` | actions out of range, a whole-axis split by its size, `BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`, a tensor core's lanes, `max_up`, `include_0` | `candidates.golden` (every row, `max_up` rows `add clang 4`, `matmul_half metal 16`; the suite numbers each candidate's action by its position from 1, after the kernel's 0); `S › get_kernel_actions` (3 laws) |
| tinygrad: `codegen/opt/search.py` `beam_search` | rounds, the beam of width `amt`, `BEAM_MIN_PROGRESS`, binaries timed once, measurements that fail; the kernel timed first and progress beyond the samples' spread (D117) | `searches.golden` (every row); `S › a search chooses the fastest program it measured` (law, slow); `S › rounds` (4 tests); `S › a round goes on iff ...` (law); `S › the kernel is sampled three times before any candidate`; `S › a search whose kernel does not compile progresses on any candidate`; `S › a timing that raises` (4 tests); `S › a search links each program it times once, before its samples` (law); `S › a search compiles each kernel once` |
| tinygrad: `codegen/opt/search.py` `_time_program`, `get_test_global_size` | three measurements, stopping early above three times the best; launches of more than 65536 workgroups halved from the last size above 16, times scaled | `S › rounds › a candidate is measured three times unless slower than three times the incumbent`; `S › each program is linked once and sampled until ...` (law); `S › measuring on fewer workgroups` (5 tests; `add_3d` pins the last size above 16, `add_large` a launch of exactly 65536); the `measurements` column of `searches.golden` |
| tinygrad: `codegen/opt/search.py` `_try_compile` | the kernel named `test`, storage on the renderer's device, `BEAM_UOPS_MAX`, failures dropped, `BEAM_STRICT_MODE`, `DEBUG>=4` | `S › a candidate's storage is placed on its renderer's device`; `S › a candidate that does not compile is dropped` (2 tests); `S › BEAM_PADTO=1 ... › a candidate of BEAM_UOPS_MAX instructions or more is dropped`, `› ... is not compiled`, `› a compilation that raises is raised under BEAM_STRICT_MODE`; `S › a compilation's failure is printed under DEBUG=4` |
| tinygrad: `codegen/opt/search.py` `BEAM_TIMEOUT_SEC`, `BEAM_DEV_TIMEOUT` | a compilation interrupted by an alarm, a run's time limit | dropped: a search times no compilation (D124), and a run's limit is the timing's, excluded (README) |
| tinygrad: `codegen/opt/search.py` `diskcache_get`, `diskcache_put`, `IGNORE_BEAM_CACHE`, `CACHELEVEL` | a kept search measures nothing and reapplies what it kept | `S › a search kept in the cache links and times nothing, unless it is ignored`; `S › the cache keeps tensor cores and swaps` (slow) |
| tinygrad: `codegen/opt/search.py` `DEBUG>=2`, `BEAM_DEBUG`, `BEAM_LOG_SURPASS_MAX` | progress, the kernel, failures and the choice; kernels dropped for their lanes and instructions | `S › a search prints its progress under DEBUG=2`; `S › a search prints nothing by default`; `S › BEAM_PADTO=1 ... › a search prints the kernel, failures and its choice under BEAM_DEBUG`, `› kernels of too many lanes are reported ...`, the uops test (its output); the kernel prints as a graph, since `pyrender` is excluded (README) |
| tinygrad: `codegen/opt/search.py` `get_worker_pool`, `imap_unordered` | candidates compiled in parallel | `S › a search measures and chooses alike on one domain and on several`: the order is the candidates', tinygrad's order without a pool (D5) |
| tinygrad: `codegen/opt/search.py` the compute filter (1000 times the fewest operations) | a candidate whose estimated operations exceed a thousand times the fewest of its round so far is not timed | `S › BEAM_PADTO=1 ... › searches_environment.golden` (`transpose_33` on Metal: the copies have no arithmetic, and each pad adds a selection per element), `› a candidate of more than 1000 times the fewest operations is not measured` |

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
| tinygrad: `null/test_const_folding.py::TestBitcastConstFolding::test_vec_bitcast` | | `CG › full_rewrite_to_sink folds whole graphs › a bitcast of constants folds lane by lane`, `a cast of constants folds lane by lane` |
| tinygrad: `null/test_uops.py::TestGatedStoreRewrite::test_tiny_gate_store`, `test_gate_some_stores` | a gated store is the one store inside an if | `CG › gated stores › a store gated by its index is the one store inside an if`, `an ungated store beside it stays out of the if` |
| tinygrad: `null/test_uops.py::TestGatedStoreRewrite::test_merge_ifs_alt` | | dropped: skipped in tinygrad ("we don't merge ifs anymore") |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_division_power_of_two`, `test_floormod_power_of_two`, `test_max_keeps_bound_for_idiv`, `test_floordiv_power_of_two`, `test_unsigned_floordiv_is_cdiv`, `test_fast_idiv_and_mod`, `test_fast_idiv_nonpositive_divisor`, `test_fast_idiv_cmod_kept_when_idiv_declines`, `test_fast_idiv_bounded_numerator_zero`, `test_fast_idiv_remove_powers_of_two`, `test_disable_fast_idiv` | | `CG › divisions` (one test each, in order); `test_max_keeps_bound_for_idiv`'s renderer is Clang's operations without `MAX`, as `CStyleLanguage`'s lack it, and its dividend `max x 0 + 1` wraps at the greatest int (D24), so `a maximum proves its dividend positive before it is decomposed` divides `max x 0`, and `a dividend that can wrap keeps the correction of a floor division` pins the difference |
| tinygrad: `null/test_uops.py::TestFastIdiv::test_fast_idiv_overflow` | | dropped: an expected failure in tinygrad |
| tinygrad: `null/test_uops.py::TestMemoryCoalescing`, `TestLowerIndexDtype` | | Coalesce's and Uop_weak's sections |
| tinygrad: `null/test_simplify_valid_idx.py::TestRangeShrink` (8 tests) | the ranges left after lowering | `CG › range shrinking` (one test each); the pass alone is Simplify's section |
| tinygrad: `null/test_simplify_valid_idx.py::TestImageSimplification::test_drop_gate_committed_in_the_index_pass` | | dropped: images are excluded |
| tinygrad: `null/test_dtype_weak.py::TestNoRedundantWide` (2 tests) | 64 bits only where the bounds need them | `CG › tinygrad's claims on programs › long: computes a long in 64 bits`, `fancy_index: indexes without 64-bit arithmetic` |
| tinygrad: `null/test_randomness.py::TestRandomness::test_threefry_doesnt_use_long` | | `CG › tinygrad's claims on programs › threefry: computes random bits without 64-bit values` (Clang) |
| tinygrad: `null/test_arange.py::TestArange::test_cat_complexity`, `test_tri_complexity` | an arange and a mask cost few operations | `CG › tinygrad's claims on programs › cat: concatenates in 20 operations an element`, `triu_noopt: masks in 4 operations an element` |
| tinygrad: `codegen/__init__.py` `pm_wmma_add` | the running sum replaces a zero tensor-core accumulator; a sum keeps its value, apart from a zero's sign | `CG › tensor-core accumulators` (every test); the lowered tensor cores: `CG › stages › matmul_tc`, `matmul_half`, `attention` |
| tinygrad: `codegen/__init__.py` `beam` | a kernel asking for a beam search of width `w` is searched with `beam w` | `CG › beam search` (every test) |
| tinygrad: `codegen/__init__.py` `to_program_cache`, `engine/worker.py` | programs are kept per kernel, renderer and settings; workers share them | `CG › programs are kept` (every test) |
| tinygrad: `codegen/__init__.py` `do_to_program`, `pm_to_program` | a program resumes from a sink or a sink and its instructions | narrowed (D129): `CG › linearize and compile › compile completes linearize into the program to_program makes`, `› compile refuses what linearize does not make`, `› to_program refuses a program` |
| tinygrad: `codegen/__init__.py` `do_estimates` | estimates leave index arithmetic out | `CG › programs › the estimates count the instructions, leaving out index arithmetic` |
| tinygrad: `codegen/__init__.py` `do_linearize`, `do_render`, `do_compile` (`DEBUG`) | diagnostics | `CG › diagnostics` (every test) |
| tinygrad: `codegen/__init__.py` `line_rewrite` | | `CG › line_rewrite` (every test) |
| tinygrad: `codegen/__init__.py` (emulated narrow floats, D9) | an emulated float8 program computes IEEE conversions | `CG › stages › an emulated float8 program launches as tinygrad's`; `CG › values › on the host, a program writes what its kernel writes › fp8` |

## Cstyle

The suite is `Tolk.Cstyle` (`renderer/cstyle/`), written `CS` below.
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
tinygrad's source for that kernel (`CS › narrowing`). Execution runs on the host through the
engine (`support/run.ml`), which compiles each kernel with rig's host entry
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
| tinygrad: `null/test_compile_failures.py::TestDisassembly::test_float16_alu` | Clang on arm64 adds halves without converting | dropped here: disassembly is `Compiler_cpu`'s; that each `__fp16` operation rounds to a half is `CS › execution on the host › rounds each operation on halves to a half` |
| tinygrad: `device/cpu/test_cpu.py::TestCPU::test_arch_feats` | the architecture's features reach the compiled code (`vmov` with `avx`) | dropped here: the compiler's flags are `Compiler_cpu`'s; `CS › declarations › cachekey` pins that Clang's cache table is named after the architecture |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_compile_success`, `test_compile_error` | Metal compiles a kernel, and refuses bad source | slow: `CS › every GPU kernel compiles with its target's toolchain › metal_*` (49 kernels; `metal_transcendental_bf16` is xfail: tinygrad's graph truncates a bfloat without the float cast, D18); the refusal is `Compiler_metal`'s |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_alloc_oom`, `test_failed_newLibraryWithData`, `test_free` | Metal memory and pipelines | dropped: the device's, rig and `tolk.engine` (plan §1a) |
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
| D17: `renderer/cstyle.py:245-250` C promotion of narrow operands | char, short and `__fp16` operations compute in int or float, and only the stored value narrows; tolk casts each inlined one back | `CS › narrowing › cases.golden` (every kernel's written kernel is `with_d17_casts` of it; 21 sources are tinygrad's for the kernel with those casts); `CS › execution on the host › rounds each operation on halves to a half`, `wraps each operation on unsigned chars`; slow law `a kernel over a narrow type wraps and rounds as the interpreter` (8 kernels, inputs over each type's whole range) |
| D18: `renderer/cstyle.py:368-372` Metal's `extra_matcher` computes `SQRT`, `EXP2`, `LOG2` and `SIN` of a bfloat16 in float32, not `TRUNC` | Metal truncates only floats, so a bfloat16 `TRUNC` is computed in float32 too | `CS › bfloat16 truncation on Metal › truncates a bfloat16 in float32`, `leaves it to CUDA, which truncates a bfloat16 with htrunc`; slow: `compiles a kernel that truncates a bfloat16` (skipped without MTLCompiler) |
| D16: `renderer/cstyle.py:33,42` a float8 conversion saturates infinities on CUDA | an infinity stays one in e5m2 and becomes a NaN of its sign in e4m3; finite values saturate | `CS › float8 infinities on CUDA` (6 tests: the guard exactly where a value converts to a float8, the byte of each infinity, infinite constants as their bits); `CS › sources › cuda_dtype_float8_*`, `cuda_inf_nan_float8_*` through `tinygrad_of_d16`; NVRTC: slow, `CS › every GPU kernel compiles with its target's toolchain › cuda_*` |

## C

The suite is `Tolk.C` (`runtime/support/c/`), written `C` below.
`findlib_trees.golden` holds 41 searches of `DLL.findlib`, each a tree of
files, an environment and the paths searched, with the file tinygrad finds:
25 run as Linux searches and 16 as macOS's, whatever the platform that
generated them, and each row runs on its own platform. No tinygrad test covers
`findlib`.

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

The suite is `Tolk.Compiler_cpu` (`runtime/support/compiler_cpu/`),
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
| tinygrad: `null/test_elf.py::TestElfLoader::test_clang_jit_compiler_external_raise`, `test_load_clang_jit_strtab`, `test_link` | the loader refuses an unresolved symbol, reads `.rela.text`, links libm | dropped: loading is rig's (`Rig.Image`); that the compiled object needs nothing of a library is `CPU › execution on the host › a compiled square root runs without a library` |
| tinygrad: `runtime/support/compiler_cpu.py` `-fno-math-errno` (no test) | a square root is an instruction, not a call | `CPU › objects › a square root is one instruction on › x86_64,x86-64`, `› arm64,generic`; `CPU › execution on the host › a compiled square root runs without a library` |
| tinygrad: `runtime/support/compiler_cpu.py` `-ffixed-x18` (no test) | x18 is left alone on arm64 | `CPU › features › arm64 leaves x18 alone` |
| tinygrad: `runtime/support/compiler_cpu.py` features (no test) | arm64 `f` is `+f` and `-f` is `+nof`; riscv64 joins features as extensions, `native` is `rv64g` | `CPU › features › a half addition on arm64 converts iff fp16 is off:` (2), `arm64 vectorizes with its SIMD registers`, `-simd on arm64 disables them`, `a feature on riscv64 is an extension the processor gains`; `CPU › objects › native on riscv64 is rv64g` |
| tinygrad: `runtime/support/compiler_cpu.py` the arch assertion and `unsupported arch` | a malformed arch or another machine is refused | `CPU › architectures › an architecture of fewer than two fields is refused, named:` (3), `another machine is refused, named:` (3) |
| tinygrad: `runtime/support/compiler_cpu.py` `Compiler.__init__` cache key | `compile_clang_obj_` and the arch's fields joined by `_` | `CPU › cache › objects are cached in the table of the architecture` (3), `the table is named with ccache off`; `CPU › without Clang › a cached object is served without running Clang` |
| tinygrad: `runtime/support/compiler_cpu.py` `disassemble` | `objdump -d` of the object | `CPU › disassembly › prints what objdump prints of the object` |
| none | determinism | `CPU › objects › one source compiles to the same bytes` |

## Compiler_cuda

The suite is `Tolk.Compiler_cuda` (`runtime/support/compiler_cuda/`),
written `CUDA` below. Settings are read when the program starts, so the tests
tagged `no-library` run in a second process, where `NVRTC_PATH` names a file
that is no library, and those tagged `no-spawn` in a third; the fork tests,
`Tolk.Compiler_cuda (fork)`, run in a program that spawns no domain. The tests of NVRTC
itself are slow and skip without it; they pass against NVRTC 12.8 on Linux
arm64 and 13.4 on Linux x86_64.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/support/compiler_cuda.py` `NVRTCCompiler.compile` (no test) | PTX and cubins | slow: `CUDA › NVRTC › compiles a kernel to PTX`, `compiles a kernel to a cubin`, `compiles from several domains at once` |
| tinygrad: `runtime/support/compiler_cuda.py` `nvrtc_check` | `Nvrtc Error`, NVRTC's error string and log | slow: `CUDA › NVRTC › a rejected source raises Compile_error with NVRTC's log` |
| tinygrad: `runtime/support/compiler_cuda.py` `NVRTCCompiler.__init__` cache key | `compile_<cache_key>_<arch>` | `CUDA › cache` (4 tests); `CUDA › a library that does not load › a cached binary is served without loading NVRTC` |
| tinygrad: `runtime/support/compiler_cuda.py` `cuda_disassemble` | `ptxas` then `nvdisasm`, or why they failed | `CUDA › disassembly › a PTX that ptxas cannot assemble prints why`, `a cubin that nvdisasm cannot read prints why` |
| D15 | making the compiler loads nothing; the first compile raises | `CUDA › without NVRTC on the machine › a compile raises Compile_error naming nvrtc and NVRTC_PATH`; `CUDA › a library that does not load` (4 tests) |
| D5 | compiles in processes of their own, which serve in turn: a crash is a `Compile_error` naming the signal and fails that compile only, a killed idle process is replaced, NVRTC's cache of binaries stays empty, a process that cannot start is `Failure` | slow: `CUDA › compiles in processes of their own` (6 tests); `CUDA › a library that does not load › a compile in turn is served by an idle worker`, `a compile after its idle workers are killed raises the same error`; `CUDA › a process that cannot start`; `Tolk.Compiler_cuda (fork) › a forked process` (2 tests) |
| tinygrad: `device/nv/test_renderer_failures.py`, `runtime/test_renderer_failures.py` | kernels that once failed to render | dropped here: rendering is `Cstyle`'s section; slow: `Tolk.Cstyle › every GPU kernel compiles with its target's toolchain` compiles every CUDA kernel |

## Compiler_amd

The suite is `Tolk.Compiler_amd` (`runtime/support/compiler_amd/`),
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

## Compiler_metal

The suite is `Tolk.Compiler_metal` (`runtime/support/compiler_metal/`),
written `M` below. It covers `MetalCompiler`, the part of `ops_metal.py` that
precedes `Cstyle` (DIVERGENCES D4). MTLCompiler is
loaded once per process, so the tests tagged `no-library` run in a second
process, where `MTLCOMPILER_PATH` names a file that is no library. The tests
of MTLCompiler itself skip elsewhere than on macOS.

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `device/metal/test_metal.py::TestMetal::test_compile_success` | a kernel compiles to a library | `M › MTLCompiler › compiles a kernel to a Metal library` |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_compile_error` | `CompileError` on bad source | `M › MTLCompiler › a rejected source raises Compile_error with the compiler's message` |
| tinygrad: `device/metal/test_metal.py::TestMetal::test_alloc_oom`, `test_failed_newLibraryWithData`, `test_free` | the device's memory and pipelines | dropped: the device is rig's and `tolk.engine`'s (plan §1a) |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.compile` (no test) | the reply's data starts after the header and the warnings | `M › MTLCompiler › a source that draws warnings compiles to a Metal library` |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.compile` (no test) | the Metal version by macOS, `-fno-fast-math` | `M › MTLCompiler › the language is the latest the running macOS compiles kernels of`, `fast math is off` |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.__reduce__` | one compiler per forked process | dropped: no pickling; `M › MTLCompiler › compiles from several domains at once` pins the concurrent use of one service |
| tinygrad: `runtime/ops_metal.py` `MetalCompiler.disassemble` | the applegpu disassembler | `M › MTLCompiler › disassembly prints nothing` (README) |
| tinygrad: `runtime/ops_metal.py` cache key | `compile_metal_direct` | `M › cache` (2 tests); `M › a library that does not load › a cached library is served without loading MTLCompiler` |
| D15 | making the compiler loads nothing; the first compile raises | `M › without MTLCompiler on the machine › a compile raises Compile_error naming MTLCompiler and MTLCOMPILER_PATH`; `M › a library that does not load` (4 tests) |
| tinygrad: `external/external_metal_compile_fail.py` | a kernel that crashed Metal's compiler | dropped: a crash reproducer of a driver bug, outside tinygrad's suite |

## Indexing

The suite is `Tolk.Indexing` (`schedule/indexing/`), written `IX` below.
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
| tinygrad: runtime/test_rangeify.py::TestDoubleMatmul::test_double_matmul | two matmuls in a row | `IX › run_rangeify › recorded graphs › double_matmul_rangeified.golden`; the numbers are the executor's (`tolk.engine`) |
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

## Allreduce

The suite is `Tolk.Allreduce` (`schedule/allreduce/`), written `AR`
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
| tinygrad: schedule/allreduce.py, the hierarchical branch of `handle_allreduce` (no upstream test) | a hierarchical allreduce to one device | D19: `AR › handle_allreduce › recorded › nodes_to_one_device_handled.golden, landed on its device (D19)` states the corrected graph from tinygrad's; `AR › rules › a hierarchical allreduce to one device lands there` checks its value and placement, and its function's. A known upstream defect: tinygrad leaves the value on every device, and `create_allreduce_function` then stores it into storage on the one target device |
| tinygrad: null/test_allreduce.py::TestAllreduceCast (3 tests) | `ALLREDUCE_CAST` keeps 16-bit copies | dropped here: the cast is `schedule/multi.py`'s, Multi's section |
| tinygrad: null/test_multitensor.py::TestMultiRamUsage::test_multi_layer_allreduce, test_allreduce_cast_dtype_memory | memory of allreduces | dropped here: Memory's and Multi's sections |
| tinygrad: null/test_multitensor.py (the other tests) | sharding, multi-device ALU, batch norm | dropped here: Multi's section |

## Support_memory

The suite is `Tolk.Support_memory` (`runtime/support/support_memory/`),
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

## Memory

The suite is `Tolk.Memory` (`schedule/memory/`), written `ME` below.
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

## Device

The suite is `Tolk.Device` (`device/`), written `DV` below.
`renderers.golden` holds `Compiled._select_renderer` for 36 DEV settings, each
device and each architecture the device reports (its own and none): the target
it renders for, which the suite gives `Device.renderer`, and the renderer it
picks, the error of a target that names a renderer the device lacks (which
tolk gives without tinygrad's suggestion of a misspelling), or the failure of the renderer itself, whose text
is the renderer's own. A device lists the renderers of its `Compiled` that
tolk ports. `<case>_program.golden` holds programs that `to_program`
compiles for Clang and Metal, with the empty compiler so that no toolchain
shapes them, and `elfs.golden`, `signatures.golden` and `layouts.golden` what
`to_elf` and `TinyELF.iter_sig` give for each; `reprs.golden` also holds
tinygrad's repr, which `Tiny_elf.pp` matches. The test that two
domains share a renderer cannot run under mutation testing, which forks.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: null/test_device.py::TestDevice::test_nonexistent_renderer | `CPU:TYPO` has no renderer; `CPU:CLANGJIT` suggests `CLANG` | `DV › renderer › picks a device's renderer as tinygrad does › renderers.golden` (`dev=CPU:TYPO`, `dev=CPU:CLANGJIT`, and a misspelling for each device); `DV › renderer › refuses a renderer the device lacks`. The suggestion is dropped (README exclusions): the suite compares tinygrad's error without it |
| tinygrad: null/test_device.py::TestDevVar::test_dev_arch_override | an arch in DEV reaches the renderer | `renderers.golden` (`dev=::gfx942`, `dev=CUDA::sm_75`, ...), each row's target given to `Device.renderer`; `DV › renderer › renders for the target it is given` (the NULL device of the test is excluded) |
| tinygrad: null/test_device.py::TestDevice::test_env_online | the renderer follows DEV within a context, and is remembered | dropped: there is no DEV (D95); `DV › renderer's memory › returns the same renderer to a second call`, `returns one renderer per target, whatever the order of the calls` |
| tinygrad: null/test_device.py::TestDevice::test_env_overwrite_default_compiler | `DEV=CPU:LLVM`, `AMD:LLVM` pick another compiler | dropped: the LLVM renderers are excluded (README); `renderers.golden` pins that `CPU:LLVM` has no renderer and `CPU:CLANG` and `AMD:HIP` pick theirs |
| tinygrad: null/test_device.py::TestDevice::test_compiler_autodetect_fallback | a renderer that fails to make gives way to the next | dropped here: each device has one ported renderer, so there is no fallback, and `select_first_inited` is not ported (README exclusions). A failing renderer's own message is `DV › renderer › fails with the renderer's own message on an architecture it refuses` |
| tinygrad: null/test_device.py::TestDevice::test_old_renderer_env_raises | `CPU_LLVM=1` is refused | dropped: the `{DEV}_{RENDERER}` migration check is not ported (README exclusions) |
| tinygrad: null/test_device.py::TestDevice::test_canonicalize, test_lowercase_canonicalizes | `cpu`, `CL:0` and `disk:...` canonicalize | `DV › renderer takes a device's name as it is`: tolk never parses a name, and refuses `cpu`, `CPU:0` and `DISK:...` |
| tinygrad: null/test_device.py::TestDevice::test_getitem_not_exist | `Device["TYPO"]` fails | `DV › renderer › raises on a name that is no device` |
| tinygrad: null/test_device.py::TestDevice::test_nonexistent_iface, test_dev_id_out_of_range, test_old_device_env_raises, test_set_device_default_raises, test_dev_contextvar | interfaces, device indices, `Device.DEFAULT` | dropped: opening devices is rig's, and there is no default device (D3) |
| tinygrad: null/test_device.py::TestCompiler (3 tests) | the compiler cache | Renderer's section (`R › Compiler`) |
| tinygrad: null/test_device.py::TestRunAsModule::test_module_runs | `enumerate_devices_str` | dropped: not ported (README exclusions) |
| tinygrad: uop/ops.py `to_elf` (read by ops_cpu, ops_cuda, ops_metal, ops_nv, ops_qcom, realize) | buffers compacted in globals order, then variables | `DV › Tiny_elf.of_program › lays out the signature as tinygrad's to_elf does › signatures.golden`; `numbers buffers by their place among the globals, then variables` |
| tinygrad: device.py `TinyELF.iter_sig` (read by ops_cpu, ops_hip, hcq2) | values packed from an offset, each aligned to its size | `DV › Tiny_elf.iter_sig › packs a signature as tinygrad's iter_sig does › layouts.golden` (offsets 0, 3 and 8); `packs each value at the next offset aligned to its size` |
| tinygrad: test/helpers.py:146, test/amd/hw/*.py (`dev.runtime(prg.to_elf())`, hand-made `TinyELF`) | loading a program | dropped: loading is rig's (D3) |

### Rows other sections left to Device's

| Source | Behaviour | Outcome |
|---|---|---|
| Helpers: tinygrad null/test_device.py::TestDevVar::test_dev_arch_override, TestDevice::test_nonexistent_renderer | | this section's rows for the same tests |
| Helpers: tinygrad null/test_device.py::TestCompiler (3 tests) | the compiler cache | Renderer's section: `Compiler` is `Renderer.Compiler` |
| Ops: tinygrad null/test_tensor_uop_mixin.py::TestUOpEmpty::test_empty_like_sharded_to_single_device, test_empty_direct_singleton_tuple_device | a one-device tuple canonicalizes | dropped: devices are named, never canonicalized (D6); rune names them |

## Worker

The suite is `Tolk.Worker` (`engine/worker/`), written `WK` below.
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
| tinygrad: a worker process's `Context` (helpers.py:169-186) | a worker's overrides stay in its process | `WK › settings › a setting bound in an application is seen by it alone`, `applications building nodes keep their own CHECK_OOB` |
| tinygrad: null/test_schedule.py:2076, null/test_viz.py:46 (`PARALLEL=0`) | their suites compile in process | dropped: they pin nothing of the pool |

## Multi

The suite is `Tolk.Multi` (`schedule/multi/`), written `MU` below.
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
| tinygrad: runtime/test_multitensor.py::TestMultiTensor::test_gradient, test_backprop_conv, test_backprop_conv_wino, test_backward_sum, test_embedding_backward, test_embedding_backward_shard_weight, test_lr_scheduler_OneCycleLR | | dropped: gradients, tolk does not differentiate |
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

## Prepare

The suite is `Tolk.Prepare` (`schedule/prepare/`), written `PR` below.
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

## Rangeify

The suite is `Tolk.Rangeify` (`schedule/rangeify/`), written `RA` below.
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

### Rows other sections left to Rangeify's

| Source | Behaviour | Outcome |
|---|---|---|
| Ops: tinygrad `null/test_uop_graph.py::TestConstBufferize` (2 tests) | `pm_const_buffer_folding`: a constant staged over one range or several folds to the constant | `RA › get_kernel_graph › recorded ›` `setitem_column_kernels.golden` (the mask of a setitem into a column, staged over one range) and `setitem_cube_kernels.golden` (into a cube, staged over a range and two constant indices); a constant staged over several live ranges, which tinygrad's test builds by hand, never comes out of scheduling a program, so the rule is pinned where tinygrad's own flow reaches it; the dead-axis cleanup runs first there, as in tinygrad's matcher |

## Schedule

The suite is `Tolk.Schedule` (`schedule/schedule/`), written `SC` below.
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
| tinygrad: engine/realize.py `pm_beam`, `compile_linear`'s `BEAM` | a kernel asking for no beam asks for `BEAM`'s | `JB › JITBEAM unset › a kernel is searched with BEAM's width` |
| tinygrad: engine/realize.py `get_call_kernels`, `track_stats`, `ExecContext`, `exec_*`, `runtime_cache`, `run_linear`, `time_call`, `link_linear`, `capturing` | running a linear | dropped here: the run half is `tolk.engine`'s (plan §1a); its execution is `EN › link and run` |
| tinygrad: engine/realize.py `pm_validate`, `exec_validate`, `VALIDATE_WITH_CPU` | | dropped: excluded (ruling) |
| tinygrad: engine/realize.py `get_call_name`'s `encdec` and `get_call_outs_ins`'s | | dropped: the video decoder queue is excluded |
| tinygrad: engine/realize.py the `HCQInfo` branches of `get_call_outs_ins`, `get_call_written_bufs` and `estimate_uop` | | dropped: a call that submits queues is excluded (README) |
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

## Engine

The suite is `Tolk_engine` (`engine/tolk_engine/`), written `EN`
below. `EN › link and run › recorded` runs the Schedule suite's recorded
programs end to end: each is scheduled, compiled for the devices it names
(the host and memory devices, `Run.devices`), linked with its storage bound
to buffers of small integers, and run, and its storage then holds what its
tensors compute (Tensors), with each variable at the value it runs with. One
program of each kind of call runs by default, the rest are slow. `EN › link
and run › a schedule runs on the buffers each run binds to its parameters`
makes a program's buffers parameters of the compiled schedule and runs it on
two sets of buffers. `EN › GPUs` runs recorded programs, loops, a gather and
the `DEBUG` reports on Metal, CUDA, NV and AMD through rig, each skipping
without its GPU. `EN › timing` times a program on the host, on a memory
device, whose kernels the host runs, and on Metal.

### tinygrad

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: `runtime/support/hcq2.py` `_staging` (no test) | one staging buffer of 128 MiB per host, which every schedule's staged copies share | dropped: staging is rig's (`Rig.Buffer.copy`, D146) |
| tinygrad: null/test_hcq2.py::TestHCQ2Link::test_links_serve_any_input | a linked schedule serves any input, whose address it does not keep | `EN › link and run › a schedule runs on the buffers each run binds to its parameters` (contiguous, copy_view, shard_add) |
| tinygrad: null/test_hcq2.py::TestHCQ2Link::test_eager_templates_compile_once | compiling a schedule again returns the compiled one | dropped here: the engine keeps no cache of links, and rune keeps its programs by key |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_repeated_copy | a copy out, a copy in and a copy out run in order | `EN › link and run › copies run in the order of their schedule` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_caches_hold_no_buffers | freeing the values frees the device memory | `EN › runs › a linked schedule holds its storage while it is reachable` |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_has_no_rt_buffers | a jit's link owns its buffers, none from the one-shot ring | dropped: the engine has no ring; link allocates everything a schedule names (`EN › runs › a run of kernels allocates and loads nothing`) |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_jit_new_inputs_each_call, test_jit_symbolic | a captured jit's new inputs and bindings | the engine's half: `EN › link and run › a schedule runs on the buffers each run binds to its parameters`, `a schedule runs with each binding of its variables`; the jit is L8's |
| tinygrad: runtime/test_hcq2.py::TestHCQ2Schedule::test_map_cpu_buffer_preserves_contents | mapping host memory for a device keeps its bytes | dropped: mapping is rig's (`Rig.Buffer.borrow`) |
| tinygrad: runtime/test_wait_loop.py::TestWaitLoop::test_wait_loop, test_nested_loop_in_range, test_two_sequential_loops, test_loop_in_loop | do-while loops count to 10, 12, 25 and 12 | `EN › kernels on the host › loops ›` "wait_loop runs its loops to 10", "nested_loop …", "two_loops …", "loop_in_loop …", from the Linearizer's goldens |
| tinygrad: runtime/test_wait_loop.py::TestWaitLoop::test_wait_loop_spec | the loop under `SPEC=2` | dropped: the specification is Spec's; the loop runs in `EN › kernels on the host › loops` |
| tinygrad: runtime/test_wait_loop.py::TestVolatileLoops::test_async_wait_ext | a kernel spins on a host word another thread sets | dropped: a run's waits are rig's (D146), so no program the engine runs spins on the host |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_kernel_run, test_profile_kernel_run_wait | a kernel run is one profile range named after the kernel | `EN › runs › each kernel of a run is in a span of the host under a profile`; on a GPU, `EN › GPUs › Metal › at DEBUG=2, a run prints a line per kernel` |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_copyin, test_profile_multiops, test_profile_multidev, TestSimpleProfiler::test_profiler, TestProfiler::test_cpu_profile | copies and host ranges are profile events | dropped: rig records its copies and host spans (`Nx_device.Profile`) |
| tinygrad: runtime/test_profiler.py::TestProfiler::test_profile_multidev_transfer, test_profile_graph, test_dev_jitter_matrix | device-to-device transfers, graph events and clock jitter | dropped: hardware with two devices, and rig's calibration |
| tinygrad: runtime/test_search.py::TestSearch::test_beam_symbolic_kernel | beam search applies optimisations to a symbolic kernel | dropped here: the search is `Codegen.Opt.Search`'s; the timing it takes is `EN › timing` |
| tinygrad: runtime/test_realize_is_realize.py (13 tests) | `Tensor.realize` of lists, ones, disk, multi-device and variables | dropped: the `Tensor` surface; the frontend is nx, and realization rune's (L9) |
| tinygrad: runtime/test_after.py (12 tests) | ordered and disjoint stores, and the gradients through `after` | dropped: the `Tensor` surface and autodiff (rune, L9); the order of stores is Schedule's |

### Deferred to the engine by other sections

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: runtime/test_rangeify.py::TestDoubleMatmul::test_double_matmul, in Indexing's section | the numbers of two matmuls | `EN › link and run › recorded › double_matmul writes what its tensors compute` (slow) |
| tinygrad: null/test_call.py::TestCallCodegen::test_compiled_scalar_slots_are_not_call_slots, in Cstyle's section | a compiled program's variables are not its call's slots | `EN › link and run › recorded › precompiled_scalar writes what its tensors compute` |
| tinygrad: runtime/test_linearizer.py::TestLinearizer::test_arg_dedup, test_load_removed, test_assign_fold, in Codegen's section | realized values | `EN › link and run › recorded` (assign, setitem, read_then_overwrite; slow) |
| tinygrad: null/test_tensor_uop_representation.py (5 tests) | a realized value is a BUFFER | dropped: the engine binds storage to a BUFFER at link and keeps no realized state (D3); a realized value is rune's (L9) |

## Jit

`J` is the `Jit` suite (`test/engine/jit`), `JB` its JITBEAM suite
(`test_jitbeam`), which runs once per setting of `JITBEAM`, since the
environment is read once per process. `J`'s goldens are 66 captures,
generated by `gen/engine/jit.py`: each function that tinygrad's four jit test
files compile, captured by `TinyJit` on its second call over realized buffers,
on the host and on CPU:1 to CPU:3. A case's goldens are its captured schedule, the buffers the capture
holds, its inputs in slot order, and what `jit_lower` makes of them; each case
records its variables with their bound values (`captures.golden`).

For each case, `J › jit_lower › recorded` compares the lowering with
tinygrad's, and checks that each input the schedule reaches is the parameter
of its slot, of its type, device and size, and that no other parameter is
bound; that the storage the lowering reaches is
the held storage of the capture that is no input, plus arenas the capture does
not reach; and that lowering twice gives the same schedule up to the numbers
of the storage and placeholders it makes. `J › replay › recorded` links the
lowering on the host and memory devices and replays it as a jitted function
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
| tinygrad: `engine/jit.py` `CapturedJit.linear` | a captured schedule is linked once, outside the link cache | `Tolk_engine.link`: the engine keeps no link cache; `J › replay › recorded` links each lowering once and replays it |
| tinygrad: `engine/jit.py` `CapturedJit.__call__` | a replay runs the linked schedule on the call's inputs and variables, and at `DEBUG=1` says how many calls it runs | `Tolk_engine.run`; `J › replay › recorded` (66 cases); `EN › runs › at DEBUG=1, a run of ten calls says how many it runs`, `… of nine calls says nothing` |
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

### Measures

The default run of `J` takes 1.0 s on an idle machine, `JB` 0.1 s; the slow
cases take 8 s. Coverage of `jit.ml` is 100% (12 of 12 points) from the
default run. ppx_windtrap finds no mutation site in `jit.ml`, so its
mutants were made by hand, each built and run against `J` and `JB`: the
inputs left as buffers (killed by the compiler: `param` unused), the slots
shifted by one, the inputs in reverse order, the held buffers ignored, no
plan, a parameter of one element, a parameter without a device, `JITBEAM`
ignored, and `JITBEAM` defaulting to 0 are killed. Two survive, equivalent:
`~pass:Fixed_point` in the substitution of the inputs, whose parameters hold no
input to rewrite again, and planning with the inputs held, which the
substitution has already replaced.

