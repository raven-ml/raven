# Regressions

Each module's section maps every behaviour its tests must keep to the test
that keeps it, or to the reason it is dropped. The sources are:

- every behaviour an old rune test pins (`packages/rune/test`) that the module
  now owns;
- every relevant case of tinygrad's tests for the module's operations.

A row names its source as `old: <file> <test name>` or
`tinygrad: <file>::<class>::<test>`, and its outcome as the new test's path
(`<suite> › <group> › <test>`) or as `dropped: <reason>`. A module's section
starts when its test pass does.

## Compiled

The suite is `Rune_next.Compiled` (`next/test/compiled/test_compiled.ml`),
written `Compiled` below. Its laws run each kernel of `Nx_backend.S` on the
same operands in nx.cpu and in the compiled backend, over every layout (axes
permuted, reversed, broadcast, gapped, offset, in a buffer that starts inside
another), empty and scalar shapes, and the dtypes the device computes; a
law's path is given once for the host (`Compiled › host › ...`), and runs
swept on the host and on Metal (`› host, swept › ...`, `› metal › ...`,
slow). The tinygrad file is `test/runtime/test_ops.py` at `79af1ca70`.
Movements, creations and nx's compositions (activations, losses, pooling,
convolutions, normalisations) reach the backend as the kernels they are made
of, so their rows name those kernels' laws or drop with that reason.

### Old rune tests

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit_metal.ml element-wise chain matches eager | elementwise kernels on Metal give eager's values | Compiled › host › elementwise › exact unary; › exact binary (and their Metal runs) |
| old: test_jit_metal.ml duplicate scatter updates land in order | the last duplicate wins under `Set` | Compiled › host › indexed › scatter exactly |
| old: test_jit_metal.ml gathers keep -0 on the GPU | a gather keeps bits | Compiled › host › indexed › gather |
| old: test_jit_metal.ml concatenation keeps every bit on the GPU | cat keeps NaN payloads and -0 | Compiled › host › edges › a concatenation of 17 pieces, a kernel of 18 arguments, keeps every bit; › indexed › cat |
| old: test_jit_metal.ml an index outside the axis beside unit axes on the GPU | an out-of-range index reads zero | Compiled › host › indexed › gather (indices drawn from -2 to n+1 and the int32 extremes, unit axes drawn) |
| old: test_jit_metal.ml sorted values are the input's elements on the GPU; sort matches eager | a sort permutes its elements | Compiled › host › reductions › sort; › argsort |
| old: test_jit_metal.ml sort keeps subnormals | a sort moves subnormals unflushed on Metal | Compiled › metal › reductions › sort (operands of kernels that move values are not flushed) |
| old: test_jit_metal.ml scans keep subnormals | cummax and cummin keep subnormals on Metal | Compiled › metal › reductions › scan exactly |
| old: test_jit_metal.ml top_k over a row of 2^20 entries; top_k selects what it selects eagerly | top_k | dropped: `Nx.top_k` is nx's composition of `sort` and `argsort` |
| old: test_jit_metal.ml empty values have no storage | empty outputs, inputs and sums over empty slices | Compiled › host › * (every law draws dims of 0; an empty destination runs nothing) |
| old: test_jit_metal.ml float sums and products keep their grouping; float constants keep their grouping; float identities hold only where IEEE keeps them | tolk keeps IEEE float rewrites | Compiled › host › reductions › reduce floats; › scan floats (a zero result has eager's sign); the rewrites across operations belong to the compiled call's suite |
| old: test_jit_metal.ml ordered comparisons are false at NaN | comparisons at NaN | Compiled › host › elementwise › comparisons |
| old: test_jit_metal.ml max propagates NaN | maximum and max of NaN | Compiled › host › elementwise › exact binary; › reductions › reduce exactly |
| old: test_jit_metal.ml zeros keep their sign | signed zeros through arithmetic | Compiled › host › elementwise › exact unary; › exact binary; › edges |
| old: test_jit_metal.ml integer comparisons read wrapped values; folded integer constants wrap | integers wrap | Compiled › host › elementwise › exact binary; › comparisons |
| old: test_jit_metal.ml pow of a tensor base matches eager | pow | Compiled › host, swept › elementwise › transcendental binary; › exact binary (integer power) |
| old: test_jit_metal.ml a 17-argument kernel between queued work matches eager | a kernel of many arguments | Compiled › host › edges › a concatenation of 17 pieces, a kernel of 18 arguments, keeps every bit |
| old: test_jit_metal.ml a read after a call waits for it | a read waits for queued work | Compiled › host › domains › runs of one key queued with no read between them each give their own result |
| old: test_jit_metal.ml placed views bind at any offset | operands at any offset | Compiled › host › * (layouts drawn at offsets 0 to 7 in buffers that start 0 to 3 elements into their memory); › edges › an operand whose buffer starts 2 bytes into its memory is read where it is |
| old: test_jit_metal.ml a dtype Metal cannot hold raises at placement; a dtype Metal cannot hold raises before a compiled call | float64 on Metal | Compiled › metal › refusals of Metal › float64 is refused |
| old: test_jit_metal.ml Metal has one device | the device count | dropped: nx.metal.device's suite |
| old: test_jit_metal.ml bitcast reads on the GPU the bits eager reads; bitcast outputs retain their own dtype; float8 bitcasts preserve raw bytes through movements | bitcast | dropped: a bitcast is a view in nx, no kernel |
| old: test_jit_metal.ml grad inside jit, rematerialized second derivatives, custom backward, multi-kernel traces, staged scan body, command storage, arenas, two programs on one consumed state (8 tests) | the compiled call | dropped: `Rune.jit`'s suite, not the eager backend |
| old: test_jit_metal.ml placed weights, reads and moves (16 tests) | placement, binding, consumption, moves | dropped: placement and the compiled call's suites |
| old: test_jit_cuda.ml element-wise chain matches eager; float16 and bfloat16 matmul equal eager | kernels on CUDA | dropped until a CUDA device joins the suite's devices: the laws take any device, and the Metal runs cover a queued GPU |
| old: test_jit_cuda.ml (24 other tests) | the compiled call, placement, randomness and consumption on CUDA | dropped: `Rune.jit`'s and placement's suites |
| old: test_jit_alignment.ml an input at an address that is 4 modulo 16; the CPU device declares no vector alignment | a kernel reads an operand at any address | Compiled › host › * (buffers starting 0 to 3 elements into their memory); › edges › an operand whose buffer starts 2 bytes into its memory is read where it is |
| old: test_jit_alignment.ml a capture at an address that is 4 modulo 16 | captures | dropped: the compiled call's captures |
| old: test_jit_cache.ml (7 tests) | the persistent compile cache | dropped: the process-wide table is `Compiled › host › cache › *`; the disk cache is the compiled call's |
| old: test_half.ml eager vs jit (4 tests) | half softmax and layernorm | dropped: compositions; their kernels run at float16 and bfloat16 in every law |

### tinygrad test_ops.py

| Source | Behaviour | Outcome |
|---|---|---|
| tinygrad: test_ops.py::TestOps::test_trunc, test_floor, test_ceil, test_round | roundings | Compiled › host › elementwise › exact unary |
| tinygrad: test_ops.py::TestOps::test_neg, test_abs, test_abs_exact, test_sign, test_sign_exact, test_sqrt | neg, abs, sign, sqrt | Compiled › host › elementwise › exact unary |
| tinygrad: test_ops.py::TestOps::test_exp, test_log, test_sin, test_cos, test_tan, test_asin, test_acos, test_atan, test_sinh, test_cosh, test_tanh, test_tanh_extreme, test_erf | transcendentals | Compiled › host, swept › elementwise › transcendental unary |
| tinygrad: test_ops.py::TestOps::test_add, test_add3, test_broadcasted_add, test_broadcasted_add_2, test_sub, test_mul, test_div, test_div_int, test_mod, test_fmod, test_maximum, test_minimum, test_mul_naninf, test_div_naninf, test_idiv_shift_rewrite_negative | binary arithmetic | Compiled › host › elementwise › exact binary |
| tinygrad: test_ops.py::TestOps::test_xor, test_and, test_or | bitwise | Compiled › host › elementwise › exact binary |
| tinygrad: test_ops.py::TestOps::test_pow_full, test_pow, test_pow_const, test_pow_const_direct, test_pow_neg_inf_frac_exponent, test_pow_zero_exponent, test_pow_zero_tensor, test_pow_zero_const, test_int_pow_const_int, test_pow_int, test_pow_int_base_float_exponent | pow | Compiled › host, swept › elementwise › transcendental binary; › host › elementwise › exact binary (integer power) |
| tinygrad: test_ops.py::TestOps::test_cmp_eq, test_cmp_gt, test_cmp_ge, test_cmp_lt, test_cmp_le | comparisons | Compiled › host › elementwise › comparisons |
| tinygrad: test_ops.py::TestOps::test_where, test_where_permute, test_where_nan_cond, test_inf_where | where | Compiled › host › elementwise › where |
| tinygrad: test_ops.py::TestOps::test_cast; TestOpsUint8::test_cast | cast | Compiled › host › elementwise › cast |
| tinygrad: test_ops.py::TestOps::test_small_cumsum, test_simple_cumsum, test_cumsum, test_cumsum_zero_axis, test_small_cumprod, test_simple_cumprod, test_cumprod, test_cumprod_zero_axis | running sums and products | Compiled › host › reductions › scan floats; › scan exactly |
| tinygrad: test_ops.py::TestOps::test_small_cummax, test_simple_cummax, test_cummax, test_cummax_zero_axis, test_small_cummin, test_simple_cummin, test_cummin, test_cummin_zero_axis | running extremes | Compiled › host › reductions › scan exactly |
| tinygrad: test_ops.py::TestOps::test_argmax, test_argmin | arg-reductions | Compiled › host › reductions › argmax and argmin |
| tinygrad: test_ops.py::TestOps::test_sort, test_sort_independent_outputs, test_argsort | sorts | Compiled › host › reductions › sort; › argsort |
| tinygrad: test_ops.py::TestOps::test_sum_simple, test_sum_full, test_sum_tiny, test_sum, test_sum_with_zeros_shape, test_prod, test_min, test_max, test_const_reduce, test_sum_fake, test_sum_twice | reductions | Compiled › host › reductions › reduce exactly; › reduce floats |
| tinygrad: test_ops.py::TestOps::test_sum_collapse, test_sum_collapse_neg, test_sum_pad_collapse, test_sum_cat_collapse, test_max_dont_collapse, test_sum_relu | rewrites across fused operations | dropped: one kernel is one operation; fusion is the compiled call's |
| tinygrad: test_ops.py::TestOps::test_dot_1d, test_dot, test_matmul_simple, test_matmul, test_matmul_batched, test_matmul_batched_vector, test_small_gemm, test_9_gemm, test_small_gemm_padded, test_small_gemm_range, test_small_gemm_eye, test_gemm_fp16, test_gemm, test_gemm_with_zeros_shape, test_broadcastdot, test_mulacc_with_zero_strides, test_matvec | products | Compiled › host › windows and products › float products; › integer products |
| tinygrad: test_ops.py::TestOps::test_big_gemm | a large product | dropped: size is the compiler's concern; the laws draw every layout at small sizes |
| tinygrad: test_ops.py::TestOps::test_pad | constant padding | Compiled › host › indexed › pad; › cache › a pad with a fill of -0. after one of 0. keeps its fill's sign |
| tinygrad: test_ops.py::TestOps::test_cat, test_multicat, test_stack, test_stack_max | concatenation | Compiled › host › indexed › cat; › edges › a concatenation of 17 pieces, a kernel of 18 arguments, keeps every bit |
| tinygrad: test_ops.py::TestOps::test_gather, test_gather_bool_index | gather | Compiled › host › indexed › gather |
| tinygrad: test_ops.py::TestOps::test_scatter, test_scatter_add | scatter | Compiled › host › indexed › scatter exactly; › scatter add of floats |
| tinygrad: test_ops.py::TestOps::test_scatter_mul, test_scatter_reduce, test_scatter_reduce_prod_zeros | scatter with products and extremes | dropped: nx's scatter has `Set` and `Add` only |
| tinygrad: test_ops.py::TestOps::test_unfold | windows | Compiled › host › windows and products › unfold |
| tinygrad: test_ops.py::TestOps convolutions (every test_*conv* and test_padding_add) | convolutions | dropped: nx's composition of `unfold` and `matmul` (and `fold` for the transposes): Compiled › host › windows and products › unfold; › fold of floats; › float products |
| tinygrad: test_ops.py::TestOps pooling (test_max_pool*, test_avg_pool*, test_global_avg_pool2d, test_max_unpool2d*) | pooling | dropped: nx's composition of `unfold` and a reduction |
| tinygrad: test_ops.py::TestOps activations and compositions (relu, leaky_relu, celu, selu, silu, swish, log10, log2, exp2, rsqrt, copysign, logaddexp, softsign, sigmoid, logsigmoid, hardsigmoid, softplus, gelu, quick_gelu, elu, relu6, hardswish, mish, hardtanh, asinh, acosh, atanh, lerp, clip, isinf, isnan, isfinite, isclose, logical_not, bitwise_not, softmax, log_softmax, softmin, logsumexp, logcumsumexp, mean, var, std, std_mean, normalize, any, all, einsum, multidot, matvecmat, masked_fill, one_hot, tril, triu, and their extreme and exact variants) | compositions | dropped: nx's compositions of the kernels above |
| tinygrad: test_ops.py::TestOps losses and attention (19 tests: cross entropy, nll, binary cross entropy, scaled dot product attention) | compositions | dropped: nx's compositions |
| tinygrad: test_ops.py::TestOps::test_lshift, test_rshift, test_lshift_signed, test_rshift_signed | shifts | dropped: no nx kernel shifts |
| tinygrad: test_ops.py::TestOps::test_pad_reflect_mode, test_pad_replicate_mode, test_pad_circular_mode | padding modes | dropped: nx's compositions of movements and `cat` |
| tinygrad: test_ops.py::TestOps movements (slices, fancy indexing, transpose, permute, reshape, view, flip, squeeze, unsqueeze, flatten, unflatten, expand, broadcast, roll, diag, diagonal, repeat, repeat_interleave, split, chunk, nested_shrink, pad_reshape, pad_slice, stack_slice, flip_eye_crash) | movements | dropped: a movement is a view, no kernel; every law draws its operands' views |
| tinygrad: test_ops.py::TestOps creations (full, zeros, ones, eye, arange, linspace, meshgrid, empty and their like variants) | creations | dropped: nx creates values; no kernel |
| tinygrad: test_ops.py::TestOps::test_interpolate_* | interpolation | dropped: nx's composition |
| tinygrad: test_ops.py::TestOps::test_masked_select, test_masked_select_size, test_nonzero, test_nonzero_size | data-dependent shapes | dropped: nx computes them from host reads |
| tinygrad: test_ops.py::TestOps::test_topk, test_topk_independent_outputs | top_k | dropped: nx's composition of `sort` and `argsort` |
| tinygrad: test_ops.py::TestOps::test_detach, test_topo_sort, test_round_quantization_gradient, test_cmp_ne_backwards, test_cmp_lt_backwards, test_div_rounding_mode, test_exp2_log2_zero_times_negative | autograd and tinygrad's own API | dropped: no kernel of the backend |
| tinygrad: test_ops.py::TestOps::test_bitcast; TestOpsUint8::test_int_or | bitcast, uint8 or | dropped: a bitcast is a view in nx; Compiled › host › elementwise › exact binary (or over uint8) |
| tinygrad: test/null/test_ops.py (13 tests) | creation argument errors | dropped: nx's frontend validates shapes |
