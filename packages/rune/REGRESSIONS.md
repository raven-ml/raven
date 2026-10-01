# Regressions

Each module's section maps every behaviour its tests must keep to the test
that keeps it, or to the reason it is dropped. The sources are:

- every behaviour an old rune test pins (`packages/rune/test`) that the module
  now owns;
- every relevant case of tinygrad's tests for the module's operations.

A row names its source as `old: <file> <path>; <path>; ...` or
`tinygrad: <file>::<class>::<test>`, and its outcome as the new test's path
(`<suite> › <group> › <test>`) or as `dropped: <reason>`. An old test's path
is the one its suite's `-l` prints; a path ending in ` › *` covers a group,
and `*` the whole file. A row may name a test that was since deleted, to
record what became of it. `next/test/regressions` checks that a row maps every
old test (`dune build @packages/rune/next/test/regressions/regressions`). A
module's section starts when its test pass does.

## Owners

| Old test file | Section | Suites' designer |
|---|---|---|
| test_grad, test_engine, test_jacobian, test_control, test_composition, test_custom, test_total; the structural groups of test_jvp and test_complex | Transformations core | rune-core-tests |
| test_ops, test_fft, test_rng; the rule groups of test_jvp and test_complex; test_vmap | Rule tables | rune-rules-tests |
| test_jit, test_jit_metal, test_jit_cuda, test_jit_alignment, test_jit_cache, test_jit_scratch, test_device_lists, test_remat_memory, test_half, test_tensor_parallel, test_quant | Compiled, and the compiled call's | rune-compiled-test |
| test_read_lifetime | Transformations core (dropped: nx's read path) | rune-core-tests |
| exhaustive.t | the constructs' exhaustiveness | rune-core |

## Compiled

The suite is `Rune_next.Compiled` (`next/test/test_compiled.ml`),
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

Rows here map the old tests that the eager compiled kernels own. The rest of
these files belong to the compiled call, in its own section.

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit_metal.ml metal device › element-wise chain matches eager | elementwise kernels on Metal give eager's values | Compiled › metal › elementwise › exact unary; › exact binary |
| old: test_jit_metal.ml metal device › duplicate scatter updates land in order | the last duplicate wins under `Set` | Compiled › metal › indexed › scatter exactly |
| old: test_jit_metal.ml metal device › gathers keep -0 on the GPU | a gather keeps bits | Compiled › metal › indexed › gather |
| old: test_jit_metal.ml metal device › concatenation keeps every bit on the GPU | cat keeps NaN payloads and -0 | Compiled › metal › edges › a concatenation of 17 pieces, a kernel of 18 arguments, keeps every bit; › indexed › cat |
| old: test_jit_metal.ml metal device › an index outside the axis beside unit axes on the GPU | an out-of-range index reads zero | Compiled › metal › indexed › gather (indices from -2 to n+1 and the int32 extremes, unit axes drawn) |
| old: test_jit_metal.ml metal device › sorted values are the input's elements on the GPU; metal device › sort matches eager | a sort permutes its elements | Compiled › metal › reductions › sort; › argsort |
| old: test_jit_metal.ml metal device › sort keeps subnormals | a sort moves subnormals unflushed on Metal | Compiled › metal › reductions › sort (operands of kernels that move values are not flushed) |
| old: test_jit_metal.ml metal device › scans keep subnormals | cummax and cummin keep subnormals on Metal | Compiled › metal › reductions › scan exactly |
| old: test_jit_metal.ml metal device › top_k over a row of 2^20 entries on the GPU; metal device › top_k selects on the GPU what it selects eagerly | top_k | dropped: `Nx.top_k` is nx's composition of `sort` and `argsort`, whose kernels Compiled › metal › reductions › sort; › argsort check |
| old: test_jit_metal.ml metal device › empty values have no storage | empty outputs, inputs and sums over empty slices | Compiled › metal › reductions › reduce floats (every law draws dims of 0; an empty destination runs nothing) |
| old: test_jit_metal.ml metal device › float sums and products keep their grouping; metal device › float constants keep their grouping; metal device › float identities hold only where IEEE keeps them | IEEE float arithmetic in a kernel | Compiled › metal › reductions › reduce floats; › scan floats; › elementwise › exact binary |
| old: test_jit_metal.ml metal device › ordered comparisons are false at NaN | comparisons at NaN | Compiled › metal › elementwise › comparisons |
| old: test_jit_metal.ml metal device › max propagates NaN | maximum and max of NaN | Compiled › metal › elementwise › exact binary; › reductions › reduce exactly |
| old: test_jit_metal.ml metal device › zeros keep their sign | signed zeros through arithmetic | Compiled › metal › elementwise › exact unary; › exact binary |
| old: test_jit_metal.ml metal device › integer comparisons read wrapped values; metal device › folded integer constants wrap | integers wrap | Compiled › metal › elementwise › exact binary; › comparisons |
| old: test_jit_metal.ml metal device › pow of a tensor base matches eager | pow | Compiled › metal › elementwise › transcendental binary; › exact binary |
| old: test_jit_metal.ml metal device › a 17-argument kernel between queued work matches eager | a kernel of many arguments | Compiled › metal › edges › a concatenation of 17 pieces, a kernel of 18 arguments, keeps every bit |
| old: test_jit_metal.ml placed weights › a dtype Metal cannot hold raises at placement; placed weights › a dtype Metal cannot hold raises before a compiled call | float64 on Metal | Compiled › metal › Metal › float64 is refused |
| old: test_jit_metal.ml metal device › Metal has one device | the device count | dropped: nx.metal.device's suite counts its devices |
| old: test_jit_metal.ml metal device › bitcast reads on the GPU the bits eager reads; metal device › bitcast outputs retain their own dtype; metal device › float8 bitcasts preserve raw bytes through movements | bitcast | dropped: a bitcast is a view in nx and reaches no kernel |
| old: test_jit_cuda.ml cuda device › element-wise chain matches eager; cuda half › float16 matmul equals eager; cuda half › bfloat16 matmul equals eager | kernels on CUDA | dropped: no CUDA device in the suite's devices yet; the laws take any device, and Compiled › metal › * runs them on a queued GPU |
| old: test_jit_alignment.ml host memory in place › an input at an address that is 4 modulo 16; host memory in place › the CPU device declares no vector alignment | a kernel reads an operand at any address | Compiled › host › edges › an operand whose buffer starts 2 bytes into its memory is read where it is |
| old: test_jit_cache.ml persistent compile cache › * | the persistent compile cache | dropped: the cache on disk is tolk's (its Diskcache suite); the compiled call keys on the settings it documents |
| old: test_jit.ml reductions › half-precision sums accumulate wide | a float16 or bfloat16 sum runs at float32 and rounds once | Compiled › host › narrow floats accumulate at float32 › a float16 sum of 4096 ones is 4096; › a bfloat16 sum of 512 ones is 512; › a float16 running sum of 4096 ones rounds each count once; › a float16 contraction of 4096 ones is 4096 |
| old: test_jit.ml reductions › half-precision products multiply wide | a float16 product runs at float32 and rounds once | Compiled › host › narrow floats accumulate at float32 › a float16 product past the float16 range and back is exact; › a float16 running product overflows only where its value does |
| old: test_half.ml eager vs jit › * | half softmax and layernorm | dropped: compositions of kernels that every Compiled law runs at float16 and bfloat16 |

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

## Transformations core

The suites are in `next/test/`, each through the public `Rune` alone: `Rune
derivatives` (`test_derivatives.ml`), `Rune.scan` (`test_scan.ml`), `Rune
structures` (`test_structure.ml`), `Rune nesting` (`test_nesting.ml`), `Rune
compositions` (`test_composition.ml`), `Rune constructs` (`test_constructs.ml`),
`Rune totals` (`test_total.ml`) and `Rune custom rules` (`test_custom.ml`).
Tests of an operation's rule in these files (ties and zeros of reductions,
`set`, bitcast, pad, sort, an operation with no rule, a power at a zero base,
the per-operation transposes) are mapped in Rule tables.


### test_grad.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_grad.ml grad over records › aliased leaves are separate parameters | one tensor at two leaves | Rune derivatives › grad › a tensor behind two leaves is two parameters; Rune derivatives › jvp › a tensor behind two leaves has two tangents |
| old: test_grad.ml grad over records › a capture of the argument is a constant | a capture is a constant | Rune derivatives › grad › a capture that is also the argument is a constant |
| old: test_grad.ml grad over records › matches the analytic gradient | a record's gradient | Rune derivatives › grad › the gradient of a record is its analytic gradient |
| old: test_grad.ml grad over records › unused leaf has zero gradient | an unused leaf | Rune derivatives › grad › a leaf the objective does not use has a gradient of +0. |
| old: test_grad.ml grad over records › preserves structure and shapes | the gradient's structure | Rune derivatives › grad › the gradient of a record is its analytic gradient; Rune structures › preconditions › integer, bool and key leaves beside a float one are carried |
| old: test_grad.ml grad over records › value_and_grad returns the value | the value | Rune derivatives › value › the value is the objective's, bit for bit |
| old: test_grad.ml grad over records › value_and_grad_aux returns auxiliary data | auxiliary results | Rune derivatives › value › an auxiliary result leaves through its structure as values; Rune derivatives › value › an auxiliary result does not contribute to the gradient |
| old: test_grad.ml grad over records › mixed dtypes differentiate in one pass | two dtypes | Rune derivatives › grad › leaves of two dtypes differentiate in one pass |
| old: test_grad.ml grad over records › gradient descent converges | descent | Rune derivatives › grad › gradient descent on a square shrinks it by the step each time |
| old: test_grad.ml grad over records › rejects an integer single-tensor argument | no float leaf | Rune structures › preconditions › a structure with no real or complex tensor is refused › grad' of an integer tensor |
| old: test_grad.ml grad over records › carries a non-differentiable leaf | carried leaves | Rune structures › preconditions › integer, bool and key leaves beside a float one are carried |
| old: test_grad.ml vjp › scales by the cotangent; vjp › accepts non-scalar outputs | pullbacks | Rune derivatives › vjp › the pullback scales by the cotangent |
| old: test_grad.ml vjp › pulls back structured cotangents; vjp › vjp_fun pulls back a structured result | structured results | Rune derivatives › vjp › a structured result's pullback is the gradient of its pairing |
| old: test_grad.ml vjp › rejects a cotangent shape mismatch | cotangent shapes | Rune derivatives › vjp › a cotangent of another shape is refused at the pullback |
| old: test_grad.ml vjp › rejects cotangents of another structure | cotangent structures | Rune derivatives › vjp › cotangents of another structure are refused at the pullback |
| old: test_grad.ml remat › gradients are unchanged; remat › values are unchanged | remat is its function | Rune constructs › with no transformation › remat with no transformation runs its function once; Rune compositions › pairs › grad ∘ grad › remat; Rune constructs › a remat and what it captures › a captured weight's gradient |
| old: test_grad.ml remat › takes a signature | signatures | Rune structures › signatures › a function of k arguments is itself through a signature |
| old: test_grad.ml remat › a returned argument is not counted twice | a result that is an argument | Rune constructs › rules whose result is an argument, rules under a map › a remat of the identity adds its cotangent once |
| old: test_grad.ml remat › rejects a consumed argument | consumed arguments | Rune structures › signatures › remat refuses a consumed argument when given its signature |
| old: test_grad.ml remat › is its function under jvp | remat under jvp | Rune compositions › pairs › jvp ∘ jvp › remat; Rune constructs › a remat and what it captures › a captured weight's tangent |
| old: test_grad.ml remat › composes with vmap | remat under vmap | Rune compositions › pairs › grad ∘ vmap › remat; Rune compositions › pairs › vmap ∘ grad › remat |
| old: test_grad.ml remat › gradients are unchanged under jit | remat under jit | Rune compositions › pairs › jit ∘ grad › remat |
| old: test_grad.ml remat › second derivatives are unchanged under jit | second derivatives under jit | Rune compositions › triples › jit ∘ jvp ∘ grad › remat |
| old: test_grad.ml remat › second derivatives with respect to weights under jit | second derivatives in weights | Rune constructs › a remat and what it captures › second derivatives in a weight passed to the remat; Rune constructs › a remat and what it captures › second derivatives in a weight the remat captures |
| old: test_grad.ml remat › differentiates a captured tensor | captures | Rune constructs › a remat and what it captures › a captured weight's gradient |
| old: test_grad.ml remat › pushes forward a captured tensor's tangent | captures | Rune constructs › a remat and what it captures › a captured weight's tangent |
| old: test_grad.ml remat › differentiates a tensor both captured and passed | captures | Rune constructs › a remat and what it captures › a weight both captured and passed gets both shares |
| old: test_grad.ml remat › maps a batched capture | captures under a map | Rune constructs › a remat and what it captures › a remat inside a map captures the lane |
| old: test_grad.ml remat › differentiates an argument it also captures | captures | Rune constructs › a remat and what it captures › an argument the function also captures gets both shares |
| old: test_grad.ml single-tensor variants › grad' matches the analytic gradient | grad' | Rune derivatives › shorthands › grad' is grad at one tensor |
| old: test_grad.ml single-tensor variants › vjp' pulls back the cotangent | vjp' | Rune derivatives › shorthands › vjp' is vjp at one tensor |
| old: test_grad.ml no_grad scopes are independent across domains; no_grad scopes are independent across systhreads | no_grad per domain | dropped: `no_grad` is gone; transformations on two domains are independent: Rune nesting › independence › differentiations on two domains at once |

### test_engine.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_engine.ml higher order › second derivative composes | nested gradients | Rune nesting › perturbation › the second derivative of a cube |
| old: test_engine.ml higher order › third derivative composes | nested gradients | Rune nesting › perturbation › the third derivative of a fourth power |
| old: test_engine.ml gradient flow › detach stops the gradient | detach | Rune derivatives › detach › under grad a detached value has no derivative |
| old: test_engine.ml gradient flow › no_grad region is constant | no_grad | dropped: `no_grad` is gone; Rune derivatives › detach › under grad a detached value has no derivative |
| old: test_engine.ml error contracts › grad requires a scalar objective | scalar objectives | Rune structures › preconditions › a non-scalar objective is refused › grad' |
| old: test_engine.ml statefulness › grad is repeatable | no state | Rune derivatives › grad › two differentiations of one function give one gradient |
| old: test_engine.ml statefulness › value reads are transparent | reads | Rune derivatives › grad › a value read inside the objective is its primal |
| old: test_engine.ml backward pass › cotangents stay lazy views until a reshape or the result | no copies | Rune derivatives › operations › a pair of cancelling transposes adds no copy and no arithmetic to a gradient |
| old: test_engine.ml debugging › with_debug logs ops and preserves results | with_debug | dropped: `with_debug` is gone; an interpreter installed with `Nx.Op.intercept` replaces it, nx's suite |

### test_jacobian.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jacobian.ml pullbacks › pullback is reusable across cotangents | reuse | Rune derivatives › vjp › a pullback applied twice equals two fresh pullbacks |
| old: test_jacobian.ml pullbacks › pullback rejects a cotangent shape mismatch | checks | Rune derivatives › vjp › a cotangent of another shape is refused at the pullback |
| old: test_jacobian.ml gradient checking › accepts correct gradients | check_grads | Rune derivatives › check_grads › a correct gradient is accepted |
| old: test_jacobian.ml gradient checking › catches a wrong custom rule | check_grads | Rune derivatives › check_grads › a pullback twice the true one is caught |
| old: test_jacobian.ml jacobians › jacobians preserve float64; jacobians › jacobians preserve float32; jacobians › mixed-dtype jacobians follow tangent space dtypes | dtypes | Rune derivatives › jacobians › jacfwd' has the result's dtype and jacrev' the argument's; Rune derivatives › laws › jacfwd' equals jacrev' |
| old: test_jacobian.ml jacobians › jacobian matches the analytic matrix | values | Rune derivatives › jacobians › the Jacobian is its analytic matrix |
| old: test_jacobian.ml jacobians › jacobians restore input and output shapes | shapes | Rune derivatives › jacobians › the Jacobian's shape is the result's then the argument's |
| old: test_jacobian.ml jacobians › jacobians evaluate the function once | one run | Rune derivatives › jacobians › each runs the function once |
| old: test_jacobian.ml jacobians › hessian matches the analytic matrix | Hessians | Rune derivatives › jacobians › a Hessian is jacfwd' of grad' |
| old: test_jacobian.ml jacobians › hvp agrees with the materialized hessian; jacobians › structured hvp matches analytic | Hessian-vector products | Rune derivatives › jacobians › a Hessian-vector product is jvp of grad; Rune nesting › perturbation › a Hessian-vector product, forward over reverse |

### test_control.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_control.ml scan › running-sum scan is cumsum | the fold | Rune.scan › fold › a running sum is a cumulative sum |
| old: test_control.ml scan › returns the final carry | the carry | Rune.scan › fold › the result's carry is the last step's |
| old: test_control.ml scan › differentiates like the primitive | grad | Rune.scan › transformed › grad of a scan is grad of its primitive |
| old: test_control.ml scan › vectorizes over the batch | vmap | Rune.scan › transformed › vmap of a scan is the cumulative sum of each row |
| old: test_control.ml scan › rejects a scalar input | refusal | Rune.scan › refusals › a scalar row tensor is refused |
| old: test_control.ml scan › folds structures | structures | Rune.scan › fold › a structured carry, rows and outputs; Rune.scan › fold › a fold with nothing to emit returns unit |
| old: test_control.ml scan › rejects a changed carry | refusal | Rune.scan › refusals › messages › a carry of another length |
| old: test_control.ml scan › rejects a changed carry under jit | refusal under jit | Rune.scan › compiled › a changed carry is refused under jit |
| old: test_control.ml scan › a staged scan under vmap and jvp rejects a changed carry | refusal under transformations | Rune.scan › refusals › under transformations › a changed carry is refused under jvp; Rune.scan › refusals › under transformations › a changed carry is refused under vmap |
| old: test_control.ml scan › rejects changed outputs | refusal | Rune.scan › refusals › messages › outputs that differ from the first step's |
| old: test_control.ml scan › rejects a carry of another dtype | refusal | Rune.scan › refusals › messages › a carry of another dtype |
| old: test_control.ml scan › per-sample gradients (vmap of grad) | vmap of grad | Rune.scan › transformed › per-row gradients of a scan |
| old: test_control.ml scan › second-order gradients (grad of grad) | grad of grad | Rune.scan › transformed › second derivatives of a scan |
| old: test_control.ml scan › hessian-vector product (jvp of grad) | jvp of grad | Rune.scan › transformed › Hessian-vector products of a scan |
| old: test_control.ml branches › differentiates the taken branch | branches | Rune derivatives › grad › a branch on a value differentiates the branch taken |
| old: test_control.ml branches › differentiates the taken iterations | recursion | Rune derivatives › grad › a recursion on a value differentiates the iterations taken |
| old: test_control.ml cond › selects the branch by predicate; cond › differentiates the taken branch | cond | dropped: `cond` is gone; Rune derivatives › grad › a branch on a value differentiates the branch taken |
| old: test_control.ml while_loop › iterates until the predicate fails; while_loop › differentiates the taken iterations | while_loop | dropped: `while_loop` is gone; Rune derivatives › grad › a recursion on a value differentiates the iterations taken |
| old: test_control.ml exceptions › eager › * | exceptions reach the call | Rune constructs › exceptions › under no transformation |
| old: test_control.ml exceptions › jit › * | a refused operation | Rune constructs › exceptions › an operation a compiled function refuses raises at its call |
| old: test_control.ml exceptions › grad › * | exceptions under grad | Rune constructs › exceptions › under grad |
| old: test_control.ml exceptions › grad of rerun code › * | exceptions in rerun code | Rune constructs › exceptions › under a remat's function |
| old: test_control.ml exceptions › jvp › * | exceptions under jvp | Rune constructs › exceptions › under jvp |
| old: test_control.ml exceptions › vmap › * | exceptions under vmap | Rune constructs › exceptions › under vmap |
| old: test_control.ml exceptions › jvp of a map › * | exceptions under jvp of a map | Rune constructs › exceptions › under jvp of a map |
| old: test_control.ml exceptions › a total's scope › * | exceptions in a scope | Rune constructs › exceptions › under a total's scope |
| old: test_control.ml exceptions › an operation left unhandled inside a call | unhandled effects | Rune constructs › effects › an effect a construct's code leaves unhandled is that code's; Rune.scan › body › an effect the body leaves unhandled is the body's |

### test_composition.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_composition.ml grad (vmap f) › * | grad of vmap | Rune compositions › pairs › grad ∘ vmap |
| old: test_composition.ml vmap (grad f) › * | vmap of grad | Rune compositions › pairs › vmap ∘ grad |
| old: test_composition.ml jit (grad f) › * | jit of grad | Rune compositions › pairs › jit ∘ grad |
| old: test_composition.ml jit (vmap f) › * | jit of vmap | Rune compositions › pairs › jit ∘ vmap |
| old: test_composition.ml jvp (vmap f) › * | jvp of vmap | Rune compositions › pairs › jvp ∘ vmap |
| old: test_composition.ml vmap (jvp f) › * | vmap of jvp | Rune compositions › pairs › vmap ∘ jvp |
| old: test_composition.ml jit (grad (vmap f)) › * | jit of grad of vmap | Rune compositions › triples › jit ∘ grad ∘ vmap |

### test_jvp.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jvp.ml jvp over records › matches the analytic tangent | a record's tangent | Rune derivatives › jvp › the tangent of a record is its analytic tangent |
| old: test_jvp.ml jvp over records › mixed dtypes propagate in one pass | two dtypes | Rune derivatives › jvp › leaves of two dtypes push their tangents forward in one pass |
| old: test_jvp.ml jvp over records › constant function has zero tangent | constants | Rune derivatives › jvp › a function of nothing it differentiates has a zero tangent |
| old: test_jvp.ml jvp over records › rejects tangent shape mismatch; gates and errors › rejects a leaf tangent shape mismatch | tangent shapes | Rune derivatives › jvp › a tangent of another shape is refused |
| old: test_jvp.ml jvp over records › agrees with grad on scalar objectives | one derivative | Rune derivatives › laws › jvp along v is the gradient paired with v |
| old: test_jvp.ml jvp over records › gives per-leaf output tangents | structured results | Rune derivatives › jvp › each result leaf has its own tangent |
| old: test_jvp.ml jvp over records › jvp_aux returns auxiliary data | jvp_aux | dropped: `jvp_aux` is gone; the auxiliary value is part of `jvp`'s result structure |
| old: test_jvp.ml operands without a tangent › a product by a constant has a finite tangent at an infinite operand | no term for a constant | Rune derivatives › edges › a constant operand adds no term at an infinite argument |
| old: test_jvp.ml composition › hessian-vector product (forward over reverse) | forward over reverse | Rune nesting › perturbation › a Hessian-vector product, forward over reverse |
| old: test_jvp.ml composition › grad of jvp (reverse over forward) | reverse over forward | Rune nesting › perturbation › a gradient of a tangent, reverse over forward |
| old: test_jvp.ml composition › nested jvp | forward over forward | Rune nesting › perturbation › a tangent of a tangent |
| old: test_jvp.ml gates and errors › detach stops tangents | detach | Rune derivatives › detach › under jvp a detached value has no tangent |
| old: test_jvp.ml gates and errors › no_grad stops tangents | no_grad | dropped: `no_grad` is gone; Rune derivatives › detach › under jvp a detached value has no tangent |
| old: test_jvp.ml gates and errors › rejects tangents of another structure | tangent structures | Rune derivatives › jvp › tangents of another structure are refused |

### test_complex.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_complex.ml convention › the gradient of \|z - c\|^2 is 2 (z - c) | convention | Rune derivatives › complex › the gradient of \|z - c\|² is 2 (z - c) |
| old: test_complex.ml convention › the gradient of Re (c * z) is conj c | convention | Rune derivatives › complex › the gradient of Re (c z) is conj c |
| old: test_complex.ml convention › a complex-valued objective is differentiated through its real part | convention | Rune derivatives › complex › a complex objective is differentiated through its real part |
| old: test_complex.ml convention › a step against the gradient descends | convention | Rune derivatives › complex › a step against the gradient descends by lr \|g\|² to first order |
| old: test_complex.ml convention › vjp is the adjoint of jvp | convention | Rune derivatives › complex › vjp's pullback is the adjoint of jvp on complex tensors |
| old: test_complex.ml convention › custom_vjp's bwd takes and returns gradients | convention | Rune derivatives › complex › a complex custom_vjp equals its function's pullback |
| old: test_complex.ml convention › jacrev' is jacfwd' on a complex-differentiable function | convention | Rune derivatives › complex › jacrev' is jacfwd' on a complex-differentiable function |
| old: test_complex.ml convention › hvp is the derivative of the gradient | convention | Rune derivatives › complex › the Hessian-vector product of \|z\|² is 2v |
| old: test_complex.ml convention › check_grads accepts a complex parameter | convention | Rune derivatives › complex › check_grads accepts a complex parameter |

### test_custom.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_custom.ml custom_vjp › the rule replaces autodiff | the rule is used | Rune custom rules › custom_vjp › the rule replaces the derivative |
| old: test_custom.ml custom_vjp › a correct rule matches autodiff | a true rule | Rune custom rules › custom_vjp › a true rule matches the function's derivative |
| old: test_custom.ml custom_vjp › the rule composes inside a graph | composition | Rune custom rules › custom_vjp › the rule composes inside a function |
| old: test_custom.ml custom_vjp › undifferentiated calls run fwd | no differentiation | Rune custom rules › custom_vjp › with no differentiation the call is its rule's value |
| old: test_custom.ml custom_vjp › constants pass through | constants | Rune custom rules › custom_vjp › a call on a constant inside grad contributes nothing |
| old: test_custom.ml custom_vjp › multi-leaf structures get per-leaf gradients | structures | Rune custom rules › custom_vjp › a rule over two arguments gives each its gradient |
| old: test_custom.ml custom_vjp › rejects forward mode | no forward derivative | Rune custom rules › custom_vjp › jvp of a custom_vjp is refused |
| old: test_custom.ml custom_vjp › checks bwd's gradients | checks | Rune custom rules › custom_vjp › a pullback of another structure than the arguments is refused |
| old: test_custom.ml custom_vjp › a structured result | structured results | Rune custom rules › custom_vjp › a structured result |
| old: test_custom.ml custom_vjp › a result that is its parameter | a result that is an argument | Rune constructs › rules whose result is an argument, rules under a map › a custom_vjp whose result is its argument adds its cotangent once; Rune constructs › rules whose result is an argument, rules under a map › a remat of the identity adds its cotangent once |
| old: test_custom.ml custom_vjp › a unit result runs fwd under forward mode | no tensor result | Rune custom rules › custom_vjp › a custom_vjp with no tensor result runs under forward mode |
| old: test_custom.ml custom_jvp › the rule replaces autodiff | the rule is used | Rune custom rules › custom_jvp › the tangent map replaces the derivative |
| old: test_custom.ml custom_jvp › a correct rule matches autodiff | a true rule | Rune custom rules › custom_jvp › a true tangent map matches the function's derivative |
| old: test_custom.ml custom_jvp › rejects reverse mode | reverse mode | dropped: a custom_jvp now serves reverse mode by its transposed tangent map: Rune custom rules › custom_jvp › under reverse mode the tangent map is transposed |
| old: test_custom.ml custom_jvp › a structured result | structured results | Rune custom rules › custom_jvp › a structured result |
| old: test_custom.ml custom_jvp › checks jvp's tangents | checks | Rune custom rules › custom_jvp › a tangent of another shape than the result is refused |
| old: test_custom.ml custom_jvp › undifferentiated calls run f | no differentiation | Rune custom rules › custom_jvp › with no differentiation the call is its rule's value |
| old: test_custom.ml custom_jvp › a result that is its parameter | a result that is an argument | Rune constructs › rules whose result is an argument, rules under a map › a custom_jvp whose result is its argument keeps the argument's tangent |
| old: test_custom.ml custom_jvp › a unit result runs f under reverse mode | no tensor result | Rune custom rules › custom_jvp › a custom_jvp with no tensor result runs once under reverse mode |
| old: test_custom.ml composition › per-sample gradients through a custom rule | vmap of grad | Rune custom rules › under a map › per-example gradients through a custom_vjp |
| old: test_custom.ml composition › plain vmap batches the forward function | vmap | Rune custom rules › under a map › a map of a custom_vjp is the map of its value |
| old: test_custom.ml composition › grad of vmap differentiates the batched forward computation; composition › grad of vmap applies the custom vjp rule | grad of vmap | Rune custom rules › under a map › grad of a map applies a custom_vjp's pullback to the lanes |
| old: test_custom.ml composition › jvp of vmap keeps the mapped tangent shape; composition › jvp of vmap applies the custom jvp rule | jvp of vmap | Rune custom rules › under a map › jvp of a map applies a custom_jvp's tangent map to the lanes; Rune custom rules › under a map › a tangent map inside a map sees each lane's tangent |
| old: test_custom.ml composition › grad of vmap of a custom jvp raises | reverse mode | dropped: a custom_jvp now serves reverse mode: Rune custom rules › under a map › grad of a map applies a custom_jvp's transposed tangent map |
| old: test_custom.ml composition › vmap passes on a custom jvp that captures its lanes | lanes | Rune custom rules › under a map › a map passes on a custom_jvp that reads its lanes |
| old: test_custom.ml composition › jvp of vmap of a custom vjp raises | no forward derivative | Rune custom rules › under a map › jvp of a map of a custom_vjp is refused |
| old: test_custom.ml composition › vmap passes on a custom vjp that captures its lanes | lanes | Rune custom rules › under a map › a map passes on a custom_vjp that reads its lanes |
| old: test_custom.ml compiled rules › custom reverse rule survives compilation and replay | under jit | Rune custom rules › under a compiled function › a custom_vjp's pullback under jit, replayed |
| old: test_custom.ml compiled rules › custom forward rule survives compilation and replay | under jit | Rune custom rules › under a compiled function › a custom_jvp's tangent map under jit, replayed |
| old: test_custom.ml compiled rules › custom backward uses indexed scatter | under jit | Rune custom rules › under a compiled function › a custom_vjp whose pullback scatters, under jit, replayed |

### test_total.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_total.ml scopes › no scope is inert | no scope | Rune totals › scopes › with no scope an addition does nothing |
| old: test_total.ml scopes › the innermost scope of a total collects | innermost scope | Rune totals › scopes › the innermost scope of a total collects |
| old: test_total.ml scopes › a shape mismatch raises at the addition | shapes | Rune totals › scopes › an addition of another shape is refused where it is made |
| old: test_total.ml scopes › an exception leaves the scope | exceptions | Rune totals › scopes › an exception leaves the scope |
| old: test_total.ml scopes › a caught exception keeps its additions | exceptions | Rune totals › scopes › a caught exception keeps its additions |
| old: test_total.ml scopes › a jit inside a scope runs eagerly | jit in a scope | Rune totals › scopes › a jit inside a scope runs eagerly and its additions count |
| old: test_total.ml scans and remats › an eager scan counts each step | scans | Rune totals › scans and remats › a scan counts each step once |
| old: test_total.ml scans and remats › a staged scan counts each step, replayed | staged scans | Rune totals › scans and remats › a staged scan counts each step, replayed |
| old: test_total.ml scans and remats › a remat | remats | Rune totals › scans and remats › a remat counts its addition once |
| old: test_total.ml scans and remats › a key scope inside a scope keeps its draws | key scopes | Rune totals › scans and remats › a key scope inside a scope keeps a scan's draws |
| old: test_total.ml scans and remats › restarted traces discard their additions | restarts | Rune totals › scans and remats › restarted traces discard their additions |
| old: test_total.ml scans and remats › a scope with no additions costs a staged scan nothing | cost | dropped: the cost of a staged loop belongs to the compiled call's suite |
| old: test_total.ml scans and remats › an exception reaches the performer | exceptions | Rune totals › scans and remats › an exception of a scan or a remat reaches its call inside the scope |
| old: test_total.ml scans and remats › placement restarts discard their additions | placement | dropped: jit's `?devices` and its placement restarts are gone; a trace places values as it runs |
| old: test_total.ml maps › an addition crossing a map is the loop's | maps | Rune totals › maps › an addition crossing a map is the loop's; Rune totals › maps › an addition crossing a map over a scan is the loop's |
| old: test_total.ml maps › a scope inside a map collects per lane | maps | Rune totals › maps › a scope inside a map collects per lane |
| old: test_total.ml differentiation › a total is differentiated | differentiation | Rune totals › differentiation › a collected total is differentiated as a value; Rune totals › differentiation › a collected total is differentiated as a value, compiled |
| old: test_total.ml sketch › a marked loss's Gauss-Newton sketch | sketches | Rune totals › sketches › a marked loss's Gauss-Newton sketch; Rune totals › sketches › a marked loss's sketch, compiled |
| old: test_total.ml sketch › a marked model trains under grad | sketches | Rune totals › sketches › a marked model trains under grad |
| old: test_total.ml sketch › a mark inside the model's own map | sketches | Rune totals › sketches › a mark inside the model's own map |
| old: test_total.ml reverse mode › a scope outside grad counts once | reruns | Rune totals › code that runs again › a scope outside grad counts a scan's additions once; Rune totals › code that runs again › a scope outside grad counts a remat's addition once; Rune totals › code that runs again › a scope outside grad counts once, compiled |
| old: test_total.ml reverse mode › rerun code inside rerun code | reruns | Rune totals › code that runs again › a remat in a remat counts once; Rune totals › code that runs again › a scan in a remat counts once |
| old: test_total.ml reverse mode › a custom call in rerun code | reruns | Rune totals › code that runs again › a custom_vjp rule in a remat counts once; Rune totals › code that runs again › a custom_jvp rule with no tensor result in a remat counts once |
| old: test_total.ml reverse mode › higher order | reruns | Rune totals › code that runs again › forward over reverse counts once; Rune totals › code that runs again › reverse over reverse counts once; Rune totals › code that runs again › a pullback applied twice counts once; Rune totals › code that runs again › jacrev' counts once; Rune totals › code that runs again › jacfwd', a map over the columns, counts once per column |
| old: test_total.ml reverse mode › no_grad in rerun code | no_grad | dropped: `no_grad` is gone; additions in rerun code count once: Rune compositions › code that runs again adds to a total once |

### test_quant.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_quant.ml debug › debug | with_debug of nx.quant's operations | dropped: `with_debug` is gone, and nx.quant's effect reaches no rune.next interpreter before quantised activations land in nx |

### test_read_lifetime.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_read_lifetime.ml * | a read of a placed temporary under collection | dropped: nx's read path, not rune's: nx's placement suite |

### exhaustive.t

| Source | Behaviour | Outcome |
|---|---|---|
| old: exhaustive.t * | an interpreter that forgets a construct does not compile | next/test/exhaustive.t |

## Rule tables

The forward-mode suite is `next/test/test_jvp_rules.ml`, written
`J` below. Its rows are the rows of nx's operations, one per constructor of
`Nx.Op.t` and kind, each applied through `Nx.Op.eval` to operands drawn inside
its domain, with its static arguments drawn over their range (axes, shapes,
pads, windows, flags, which operands carry a tangent). Every row runs the same
laws: its primal and its tangent's metadata at each dtype it takes, its tangent
against a central difference on float64 and complex128, against the closed
form for elementwise rows, a linear row's tangent against the row applied to
the tangent bit for bit, and the second order against a central difference of
the tangent; an integer or boolean row has no tangent. A path
`J › <row> › <law>` names such a law; `J › edges › <row> › …` and
`J › cumulative › …` and `J › factorisations › …` are named cases. An old
test of a fixed shape or fixture maps to the row whose generator draws it. The
suite checks forward mode only; the pullbacks are the reverse-mode suite's.

The reverse-mode suite is `next/test/test_transposes.ml`,
written `T` below. Over the same rows and draws, ties and zeros included, it
checks each row's pullback against its tangent by the adjoint identity
`Re ⟨w, J v⟩ = Re ⟨J* w, v⟩` to rounding (with no finite difference), a
linear row's pullback against the row itself, the pullback of a remat of the
row and of a custom_vjp whose pullback is the row's, and a custom_jvp whose
tangent map applies the row: transposed when the map is linear, refused,
naming the row, when it is not. `T › compositions › …` holds nx's functions
made of several rows (their old fixtures, both laws, and closed forms), and
`T › edges › …` the named cases.

The batching suite is `next/test/test_batching.ml`, written `B`
below. Over the same rows and draws it checks that a map of the row is its
loop over the rows of its batched operands, whichever of them are batched,
bit for bit for the rows whose elements come from their own inputs and to
rounding for those that accumulate; through a moved batch axis, inside
another map, and over no row; and that the row's `jacfwd'` and `jacrev'`
are the loops of its tangent and its pullback. `B › edges › …` holds the
operands a row's draws never batch (conditions, indices, starts, keys),
reads inside a map, and nx's functions made of several rows.

The compiled rules are `next/test/test_compiled_rules.ml`, written `Jc`
below: every row's tangent and pullback under `Rune.jit`, on the host and
on Metal (slow), against eager, and `Jit_error` for a row or dtype the
compiled call cannot compute.

`Rune.vmap` as a whole is `next/test/test_vmap.ml`, written `V`
below: the structures it maps, its captures, its refusals that name a
leaf's path, the randomness of its lanes, and `lanes`.

### Old rune tests

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_ops.ml unary rules › neg; unary rules › exp; unary rules › log; unary rules › sqrt; unary rules › recip; unary rules › sin; unary rules › cos; unary rules › tan; unary rules › asin; unary rules › acos; unary rules › atan; unary rules › sinh; unary rules › cosh; unary rules › tanh; unary rules › abs; unary rules › erf | each unary pullback | T › unary <kind> › the pullback is the adjoint of the tangent map; J › unary <kind> › the tangent agrees with a central difference |
| old: test_ops.ml unary rules › sigmoid, across its two sides | sigmoid | T › compositions › real › sigmoid, across its two sides |
| old: test_ops.ml binary rules › add; binary rules › sub; binary rules › mul; binary rules › div; binary rules › pow; binary rules › maximum; binary rules › minimum; binary rules › atan2 | each binary pullback | T › binary <kind> › the pullback is the adjoint of the tangent map (div is fdiv; ties drawn) |
| old: test_ops.ml broadcasting › * | broadcast binaries | T › compositions › real › add broadcasts a row; mul broadcasts a column; sub broadcasts a scalar |
| old: test_ops.ml reduction rules › sum over all axes; reduction rules › sum over one axis; reduction rules › prod over one axis; reduction rules › max over one axis; reduction rules › min over one axis | reduction pullbacks | T › reduce <kind> › the pullback is the adjoint of the tangent map (axes, ties and zeros drawn) |
| old: test_ops.ml reduction rules › sum keepdims; reduction rules › max keepdims | reductions keeping their axes | T › compositions › real › sum keeping its axes; T › reduce max › the pullback is the adjoint of the tangent map (keepdims is a reshape) |
| old: test_ops.ml reduction rules › mean over one axis | mean | T › compositions › real › mean over one axis |
| old: test_ops.ml movement rules › reshape; movement rules › transpose; movement rules › broadcast_to; movement rules › shrink; movement rules › flip | movement pullbacks | T › move reshape, move permute, move expand, move shrink, move flip › a linear operation's pullback is its adjoint |
| old: test_ops.ml movement rules › pad | pad's pullback | T › pad › a linear operation's pullback is its adjoint; T › edges › indexed › a pad's fill receives nothing |
| old: test_ops.ml movement rules › sliding window; movement rules › sliding window strided; movement rules › sliding window with gaps; movement rules › sliding window on a leading axis | window pullbacks | T › move window › a linear operation's pullback is its adjoint; T › edges › indexed › an element no window reads receives zero |
| old: test_ops.ml movement rules › concatenate | cat's pullback | T › cat › a linear operation's pullback is its adjoint |
| old: test_ops.ml movement rules › slice; movement rules › tril | slice, tril | T › compositions › real › slice; tril |
| old: test_ops.ml selection rules › where; selection rules › take_along_axis; selection rules › sort | selection pullbacks | T › where, gather, sort › the pullback is the adjoint of the tangent map |
| old: test_ops.ml selection rules › scatter (set); selection rules › scatter (set) with a repeated index; selection rules › scatter (add) | scatter's pullback | T › scatter › a linear operation's pullback is its adjoint; T › edges › indexed › under Set a shadowed update receives nothing; under Set the overwritten target receives nothing; under Add every duplicate update receives the cotangent |
| old: test_ops.ml scan rules › cumsum; scan rules › cumprod; scan rules › cummax; scan rules › cummin | scan pullbacks | T › scan <kind> › the pullback is the adjoint of the tangent map (ties and zeros drawn) |
| old: test_ops.ml scan rules › cummax gives each running maximum's cotangent to its element; scan rules › cummin gives each running minimum's cotangent to its element; scan rules › cumprod is exact at zeros; scan rules › cumprod's gradient differentiates exactly at zeros; scan rules › cummax has the gradient of its definition; scan rules › cummin has the gradient of its definition; scan rules › cumprod has the gradient of its definition | the cumulative gradients at ties and zeros, at every order | J › cumulative › running extrema › cummax: the gradient of the sum is its definition's; cummin: the gradient of the sum is its definition's; of equal elements a running extremum takes the convention's; J › cumulative › running products › cumprod is exact at zeros; the Hessian of cumprod at [2; 3; 0], summed over its rows, is [4; 3; 5]; cumprod: the gradient of the sum is its definition's |
| old: test_ops.ml matmul rules › * | matmul's pullback | T › matmul › the pullback is the adjoint of the tangent map; › a linear operation's pullback is its adjoint (leading axes, broadcasts on either side or both, ranks that differ drawn) |
| old: test_ops.ml linalg rules › cholesky; linalg rules › cholesky (batched); linalg rules › qr (reduced); linalg rules › qr (reduced, batched); linalg rules › lu (square, batched) | factorisation pullbacks | T › cholesky, qr, lu › the pullback is the adjoint of the tangent map (batches drawn) |
| old: test_ops.ml linalg rules › det is det(a) times the inverse transpose; linalg rules › slogdet; linalg rules › solve and inv | det, slogdet, solve, inv | T › compositions › real › det; slogdet; solve; inv; the gradient of det is det times the inverse transpose |
| old: test_ops.ml linalg rules › solve_triangular (batched vector rhs) | the triangular solve's pullback | T › solve_triangular › the pullback is the adjoint of the tangent map; T › compositions › real › a batch of triangular solves against one right-hand side |
| old: test_ops.ml linalg rules › solve_triangular (unit diagonal) | the unread diagonal | T › edges › indexed › under unit_diag the diagonal receives nothing |
| old: test_ops.ml complex accessors › * | complex accessors | T › compositions › complex › magnitude of an assembled complex tensor; real and imag; angle; conjugate of an assembled complex tensor; the gradient of |z| is z / |z| |
| old: test_ops.ml composites › * | composites | T › compositions › real › softmax cross-entropy shaped loss; layer-norm shaped function; windowed energy |
| old: test_jvp.ml unary rules › * | each unary tangent | J › unary <kind> › the tangent agrees with a central difference; › the tangent is the derivative's closed form |
| old: test_jvp.ml binary rules › * | each binary tangent | J › binary <kind> › the tangent agrees with a central difference (div is fdiv) |
| old: test_jvp.ml operands without a tangent › a power with a constant exponent has tangent 0 at a zero base | no NaN from a constant exponent | J › edges › binary pow › at a zero base › x ** 2 has tangent 0 |
| old: test_jvp.ml operands without a tangent › a product by a constant has a finite tangent at an infinite operand | no term for a constant operand | J › edges › binary mul › a constant's infinite coefficient never meets a zero |
| old: test_jvp.ml operands without a tangent › x ** 2: jvp is the transpose of vjp; operands without a tangent › x ** 3: jvp is the transpose of vjp; operands without a tangent › 2 x: jvp is the transpose of vjp; operands without a tangent › x / 2: jvp is the transpose of vjp; operands without a tangent › maximum x 0: jvp is the transpose of vjp; operands without a tangent › minimum 0 x: jvp is the transpose of vjp; operands without a tangent › atan2 x 1: jvp is the transpose of vjp; operands without a tangent › atan2 1 x: jvp is the transpose of vjp | the adjoint identity with a constant operand, at zeros and ties | T › binary pow, mul, fdiv, maximum, minimum, atan2 › the pullback is the adjoint of the tangent map |
| old: test_jvp.ml broadcasting › * | broadcast binaries | T › compositions › real › add broadcasts a row; mul broadcasts a column |
| old: test_jvp.ml reduction rules › sum over one axis; reduction rules › sum keepdims; reduction rules › prod over one axis; reduction rules › max over one axis; reduction rules › min over one axis | reduction tangents | J › reduce <kind> › the tangent agrees with a central difference |
| old: test_jvp.ml reduction rules › mean over one axis; movement rules › slice | mean, slice | T › compositions › real › mean over one axis; slice |
| old: test_jvp.ml movement rules › reshape; movement rules › transpose; movement rules › pad; movement rules › shrink; movement rules › flip; movement rules › sliding window; movement rules › sliding window tangent is the windowed tangent; movement rules › concatenate | movement tangents | J › move reshape, move permute, pad, move shrink, move flip, move window, cat › a linear operation's tangent is the operation on the tangent |
| old: test_jvp.ml selection rules › where; selection rules › take_along_axis | selection tangents | J › where, gather › a linear operation's tangent is the operation on the tangent |
| old: test_jvp.ml selection rules › sort | a sort's tangent | J › edges › sort › the tangent is the tangent gathered by the primal's argsort, bit for bit |
| old: test_jvp.ml scan rules › cumsum; scan rules › cumprod; scan rules › cummax; scan rules › cummin | scan tangents | J › scan <kind> › the tangent agrees with a central difference |
| old: test_jvp.ml scan rules › cummax carries the tangent of each running maximum's element | the running extremum's element's tangent | J › cumulative › running extrema › cummax: the tangent is each running extremum's element's, bit for bit; of equal elements a running extremum takes the convention's |
| old: test_jvp.ml scan rules › cumprod is exact at zeros | no division in the running product | J › cumulative › running products › cumprod is exact at zeros |
| old: test_jvp.ml scan rules › cumsum: jvp is the transpose of vjp; scan rules › cumprod: jvp is the transpose of vjp; scan rules › cummax: jvp is the transpose of vjp; scan rules › cummin: jvp is the transpose of vjp; scan rules › cumprod along a first axis: jvp is the transpose of vjp; scan rules › cummax along a first axis: jvp is the transpose of vjp | the scans' adjoint identity | T › scan <kind> › the pullback is the adjoint of the tangent map (axes drawn) |
| old: test_jvp.ml matmul rules › * | matmul tangents | J › matmul › the tangent agrees with a central difference |
| old: test_jvp.ml linalg rules › cholesky; linalg rules › cholesky (batched) | cholesky's tangent | J › cholesky › the tangent agrees with a central difference |
| old: test_jvp.ml linalg rules › cholesky reads the lower triangle | the unread triangle | J › edges › cholesky › a tangent above the diagonal has no effect, in both modes |
| old: test_jvp.ml linalg rules › lu (square, batched) | lu's tangent | J › lu › the tangent agrees with a central difference |
| old: test_jvp.ml linalg rules › det, solve and inv | det, solve, inv | T › compositions › real › det; solve; inv |
| old: test_jvp.ml linalg rules › solve_triangular (batched vector rhs) | the triangular solve's tangent | J › solve_triangular › the tangent agrees with a central difference |
| old: test_jvp.ml composites › * | composites | T › compositions › real › softmax cross-entropy shaped loss; complex › magnitude of an assembled complex tensor |
| old: test_jvp.ml gates and errors › unsupported op raises when input is active | a tangent with no rule | dropped: every operation has a tangent rule; the tangents with no definition raise in J › factorisations › undefined › a complete SVD of a non-square matrix has no tangent and J › edges › qr › a complete factorisation of a tall matrix has no tangent |
| old: test_complex.ml holomorphic rules › * | complex tangents and pullbacks | J › <row> › on complex values the tangent agrees with a central difference; T › <row> › on complex values the pullback is the adjoint of the tangent map |
| old: test_complex.ml modulus › abs (reverse); modulus › abs (forward) | the modulus | J, T › unary abs › on complex values … |
| old: test_complex.ml modulus › magnitude (reverse); modulus › magnitude (forward); modulus › abs of a product (reverse); modulus › abs of a product (forward) | the modulus composed | T › compositions › complex › magnitude; abs of a product |
| old: test_complex.ml sign › sign (reverse); sign › sign (forward) | the complex sign | J, T › unary sign › on complex values … |
| old: test_complex.ml sign › sign of a product (reverse); sign › sign of a product (forward); sign › abs, second order (reverse); sign › abs, second order (forward) | the complex sign composed | T › compositions › complex › sign of a product; the modulus's gradient, differentiated again |
| old: test_complex.ml transforms › fft (reverse); transforms › fft (forward); transforms › ifft (reverse); transforms › ifft (forward); transforms › irfft (reverse); transforms › irfft (forward); transforms › irfft, odd length (reverse); transforms › irfft, odd length (forward); transforms › irfft, truncated (reverse); transforms › irfft, truncated (forward) | transform tangents and pullbacks | J, T › fft, irfft › on complex values … (inverse and output sizes drawn) |
| old: test_complex.ml transforms › ifft of fft (reverse); transforms › ifft of fft (forward); transforms › rfft of a real part (reverse); transforms › rfft of a real part (forward); transforms › complex-filtered round trip (reverse); transforms › complex-filtered round trip (forward) | spectral compositions | T › compositions › spectral › ifft of fft; rfft of a real part; complex-filtered round trip |
| old: test_complex.ml component access › * | component access | T › compositions › complex › real; imag; angle of a complex tensor; conjugate; reassembled |
| old: test_complex.ml linear and movement rules › neg (reverse); linear and movement rules › neg (forward); linear and movement rules › sum (reverse); linear and movement rules › sum (forward); linear and movement rules › cumsum (reverse); linear and movement rules › cumsum (forward); linear and movement rules › cumprod (reverse); linear and movement rules › cumprod (forward); linear and movement rules › cat (reverse); linear and movement rules › cat (forward); linear and movement rules › gather (reverse); linear and movement rules › gather (forward); linear and movement rules › flip (reverse); linear and movement rules › flip (forward); linear and movement rules › where (reverse); linear and movement rules › where (forward) | complex linear rows | J, T › <row> › on complex values … |
| old: test_complex.ml linear and movement rules › broadcast and reduce (reverse); linear and movement rules › broadcast and reduce (forward) | broadcast and reduce | T › compositions › complex › broadcast and reduce |
| old: test_complex.ml arithmetic › add (reverse); arithmetic › add (forward); arithmetic › sub (reverse); arithmetic › sub (forward); arithmetic › matmul, batched (reverse); arithmetic › matmul, batched (forward); arithmetic › matmul, vector (reverse); arithmetic › matmul, vector (forward) | complex arithmetic rows | J, T › binary add, binary sub, matmul › on complex values … |
| old: test_complex.ml arithmetic › square (reverse); arithmetic › square (forward); arithmetic › log2 (reverse); arithmetic › log2 (forward); arithmetic › exp2 (reverse); arithmetic › exp2 (forward); arithmetic › rsqrt (reverse); arithmetic › rsqrt (forward); arithmetic › mean (reverse); arithmetic › mean (forward); arithmetic › trace (reverse); arithmetic › trace (forward); arithmetic › vdot (reverse); arithmetic › vdot (forward) | complex arithmetic compositions | T › compositions › complex › square; log2; exp2; rsqrt; mean; trace; vdot |
| old: test_complex.ml movements › reshape (reverse); movements › reshape (forward); movements › transpose (reverse); movements › transpose (forward); movements › pad (reverse); movements › pad (forward); movements › shrink (reverse); movements › shrink (forward); movements › sliding window (reverse); movements › sliding window (forward); movements › scatter, set (reverse); movements › scatter, set (forward); movements › scatter, add (reverse); movements › scatter, add (forward) | complex movement rows | J, T › move reshape, move permute, pad, move shrink, move window, scatter › on complex values … |
| old: test_complex.ml movements › slice, strided (reverse); movements › slice, strided (forward); movements › slice, dynamic (reverse); movements › slice, dynamic (forward); movements › set, dynamic (reverse); movements › set, dynamic (forward); movements › tile (reverse); movements › tile (forward); movements › roll (reverse); movements › roll (forward); movements › set (reverse); movements › set (forward); movements › diagonal (reverse); movements › diagonal (forward); movements › correlate (reverse); movements › correlate (forward) | complex movement compositions | T › compositions › complex › slice, strided; slice, dynamic; set, dynamic; tile; roll; set; diagonal; correlate |
| old: test_complex.ml triangular solves › lower (reverse); triangular solves › lower (forward); triangular solves › upper (reverse); triangular solves › upper (forward); triangular solves › lower, transposed (reverse); triangular solves › lower, transposed (forward); triangular solves › upper, transposed (reverse); triangular solves › upper, transposed (forward); triangular solves › unit diagonal, transposed (reverse); triangular solves › unit diagonal, transposed (forward); triangular solves › vector, transposed (reverse); triangular solves › vector, transposed (forward) | the conjugate transpose in the solve | T › edges › a solve's pullback in b is the solve by the conjugate transpose; J, T › solve_triangular › on complex values … (all 8 flag combinations drawn) |
| old: test_complex.ml triangular solves › solve (reverse); triangular solves › solve (forward); triangular solves › inv (reverse); triangular solves › inv (forward) | complex solve, inv | T › compositions › complex › solve, complex; inv, complex |
| old: test_complex.ml cholesky › lower (reverse); cholesky › lower (forward); cholesky › upper (reverse); cholesky › upper (forward) | the Hermitian factor | J, T › cholesky › on complex values …; J › edges › cholesky › an imaginary tangent on the diagonal has no effect |
| old: test_complex.ml cholesky › of a Gram matrix (reverse); cholesky › of a Gram matrix (forward) | a Gram matrix's factor | T › compositions › complex › cholesky of a Gram matrix |
| old: test_complex.ml qr › * | complex qr | J, T › qr › on complex values … (tall, square and wide drawn) |
| old: test_complex.ml factorisations › det (reverse); factorisations › det (forward) | complex det | T › compositions › complex › det, complex |
| old: test_complex.ml factorisations › lu (reverse); factorisations › lu (forward) | complex lu | J, T › lu › on complex values … |
| old: test_grad.ml reduction derivatives preserve zeros and ties | products at zeros, shared ties, eager and compiled | J › edges › reduce prod › one zero leaves the product of the others, two leave zero; J › edges › reduce max, reduce min › tied elements share the derivative; Jc › reductions › a compiled product keeps the multiplicity of its zeros; compiled tied extrema share the derivative |
| old: test_grad.ml half reduction derivatives count ties without overflow | a float16 tie count, eager and compiled | J › edges › reduce max, reduce min › a float16 tie count above 65,504 does not overflow; Jc › reductions › a compiled float16 tie count above 65,504 does not overflow |
| old: test_grad.ml set › differentiates both operands | set's pullback in its target and its value | T › compositions › real › set differentiates both operands |
| old: test_grad.ml single-tensor variants › a bitcast has zero derivative | a bitcast carries no tangent | J › bitcast › it has no tangent and passes no cotangent |
| old: test_engine.ml regressions › pad keeps its fill value under grad | pad's fill under grad | T › edges › indexed › a pad's fill receives nothing; J › edges › pad › the tangent's fill is zero |
| old: test_engine.ml regressions › sort routes gradient through the permutation | sort's gradient through its permutation | J › edges › sort › the tangent is the tangent gathered by the primal's argsort, bit for bit; T › sort › the pullback is the adjoint of the tangent map |
| old: test_engine.ml gradient flow › constants pass through unsupported ops; error contracts › unsupported op raises when its input is tracked | operations with no rule | dropped: every operation has a tangent rule; the tangents with no definition raise in J › factorisations › undefined › a complete SVD of a non-square matrix has no tangent and J › edges › qr › a complete factorisation of a tall matrix has no tangent |
| old: test_fft.ml gradients › *; one-way losses › * | spectral round trips and one-way losses | T › compositions › spectral › (the same names; the c2c pass is "a complex pass in the chain") |
| old: test_fft.ml pulls against the DFT transpose › * | the transforms' pullbacks against their definition | T › compositions › spectral pullbacks › rfft against its definition; irfft against its definition (lengths 4 and 5) |
| old: test_fft.ml forward mode › round trip, even length; forward mode › round trip, odd length; forward mode › forward and reverse pairings agree | spectral tangents and the adjoint identity | T › compositions › spectral › round trip, even length; round trip, odd length; filtered spectral energy (both laws) |
| old: test_fft.ml forward mode › rfft tangent is rfft of the tangent; forward mode › irfft tangent is irfft of the tangent | linear transform tangents | J › rfft, irfft › a linear operation's tangent is the operation on the tangent |
| old: test_fft.ml vmap › fft; vmap › ifft; vmap › rfft; vmap › irfft; vmap › fft along a non-last axis; vmap › rfft along a non-last axis; vmap › non-leading batch axis | the transforms against the loop | B › fft, rfft, irfft › a map is its loop (axes drawn); › a map through a moved axis is its loop |
| old: test_fft.ml vmap › vmap of grad | per-sample spectral gradients | B › edges › compositions › per-sample gradients of a spectral round trip |
| old: test_fft.ml jit › rfft is refused under jit | a transform under jit | Jc › host › rfft (the compiled call raises Jit_error for a transform) |
| old: test_vmap.ml loop oracle › elementwise chain; loop oracle › closure constants broadcast; loop oracle › scalar closure constant; loop oracle › constant output broadcasts; loop oracle › softmax composite; loop oracle › centering uses the unbatched mean | compositions against the loop | B › edges › compositions › an elementwise chain; captured constants broadcast; a scalar captured constant; a constant result is broadcast; softmax; centering uses each lane's mean |
| old: test_vmap.ml loop oracle › bitcast; loop oracle › full reduction; loop oracle › axis reduction on matrix elements; loop oracle › max reduction; loop oracle › vector-matrix multiply; loop oracle › matrix-matrix multiply; loop oracle › reshape and transpose; loop oracle › where selects per element; loop oracle › sort; loop oracle › cumsum; loop oracle › concatenate with itself; loop oracle › pad; loop oracle › sliding windows; loop oracle › sliding windows on a leading axis; loop oracle › extract_patches; loop oracle › combine_patches; loop oracle › take_along_axis with constant indices | each row against the loop | B › bitcast, reduce <kind>, matmul, move <kind>, where, sort, scan sum, cat, pad, move window, unfold, fold, gather › a map is its loop |
| old: test_vmap.ml loop oracle › matmul against a constant with its own batch dimensions | a captured operand's own batch axes | B › edges › compositions › a product with a captured constant of batch axes of its own |
| old: test_vmap.ml loop oracle › slice | slice against the loop | B › move shrink › a map is its loop |
| old: test_vmap.ml axes and structure › maps a moved axis | a batch axis not leading in memory | B › <row> › a map through a moved axis is its loop |
| old: test_vmap.ml axes and structure › maps all leaves of a structure; axes and structure › maps leaves of different leading ranks | structures | V › structures › every leaf of a structure is mapped; every argument of a curried function is mapped; leaves of different ranks are mapped along their own axis 0 |
| old: test_vmap.ml axes and structure › a captured value is a constant; axes and structure › a capture of the argument is a constant | captures | V › captures › a captured value is a constant of the map; a capture that is also the argument is a constant |
| old: test_vmap.ml axes and structure › rejects arguments with no leaf | no tensor to map | St › preconditions › vmap of arguments with no tensor is refused |
| old: test_vmap.ml axes and structure › rejects a consumed argument | a consumed argument | St › signatures › vmap refuses a consumed argument when given its signature |
| old: test_vmap.ml axes and structure › rejects mismatched batch sizes; axes and structure › rejects scalar leaves | refusals naming a leaf's path | V › refusals › leaves of two leading lengths are refused, naming both; a scalar leaf is refused, naming it |
| old: test_vmap.ml axes and structure › raises without a batching rule | a row with no batching rule | dropped: every row has a batching rule (B › <row> › a map is its loop) |
| old: test_vmap.ml axes and structure › reading a batched value raises; axes and structure › reading a constant value is fine | reads inside a map | B › edges › reads › reading a lane raises; reading a constant inside a map computes |
| old: test_vmap.ml set › * | update under a map | B › update › a map is its loop; B › edges › integer operands › batched starts alone write each lane's window; batched starts and values write each lane's window |
| old: test_vmap.ml nesting › vmap of vmap | a map inside a map | B › <row> › a map inside another map is its loop |
| old: test_vmap.ml lanes › the named map answers; lanes › a shared value is broadcast; lanes › another map keeps its lanes | lanes | V › lanes › the named map answers with every lane's value; a value every lane shares is gathered as its copies; a map between the gather and its named map keeps its own lanes |
| old: test_vmap.ml lanes › one lane without the map | lanes with no named map (its compiled assertion: step 4) | V › lanes › with no map of its name around it a gather is one lane |
| old: test_vmap.ml lanes › a named map passes the lane index on | lane_index through a named map | V › randomness › a named map passes the lane index of the anonymous map around it on |
| old: test_vmap.ml lanes › linear under jvp; lanes › grad outside the named map | lanes under differentiation | V › lanes › a gather's tangent is the gather of its tangent; outside its named map, reverse mode differentiates through the gather |
| old: test_vmap.ml lanes › raises under grad inside the named map | reverse mode over lanes inside their named map | T › edges › lanes › a lane's gradient is its row of every lane's cotangent; a lane's gradient through shared weights is the lane count times its row (the refusal became a value) |
| old: test_vmap.ml randomness › implicit RNG draws are identical per lane | randomness a lane captures | V › randomness › an implicit draw is a constant of the map: every lane draws the same |
| old: test_vmap.ml structured outputs › batches every output leaf | structured results | V › structures › every leaf of a structured result gains the batch axis |
| old: test_vmap.ml composition › vmap of grad: per-sample gradients; composition › grad of vmap; composition › jvp of vmap | maps composed with differentiation | M › pairs (grad, vmap), (vmap, grad), (jvp, vmap) |
| old: test_vmap.ml composition › per-sample gradients of a gather; composition › per-sample gradients of a sliding window | per-sample gradients | B › edges › compositions › per-sample gradients of a gather; per-sample gradients of a sliding window |
| old: test_rng.ml samplers › same key, same values; samplers › different keys differ | a draw is a function of its key | dropped: nx's Rng suite, keys › the same values from the same key, held in any layout › * and keys › equal seeds give equal keys, and distinct seeds distinct keys |
| old: test_rng.ml samplers › uniform range and mean; samplers › normal moments; samplers › randint range; samplers › bernoulli probability | the samplers' supports and laws | dropped: nx's Rng suite, supports › * and distributions › * |
| old: test_rng.ml samplers › argument validation | the samplers' refusals | dropped: nx's Rng suite, errors › * |
| old: test_rng.ml keys › * | key derivation | dropped: nx's Rng suite, keys › split makes n subkeys, …; keys › fold_in gives distinct counters distinct keys; distributions › uniforms from split's two subkeys, multiplied follows its law |
| old: test_rng.ml front-ends › the implicit scope draws the explicit samples | the implicit scope | dropped: nx's Rng suite, scope › a keyless sampler is its keyed twin on next_key (), in a scope that replays its draws |
| old: test_rng.ml jit › * | randomness under the compiled call | dropped: the compiled call's suite pins compiled randomness (step 4) |
| old: test_rng.ml transformations › samples are constants of the tape | a draw is a constant of a differentiation | T › edges › draws › a drawn mask is a constant of the differentiation |
| old: test_rng.ml transformations › truncated_normal differentiates through its bounds | a reparameterised draw | T › compositions › real › a truncated normal draw through its lower bound |
| old: test_rng.ml transformations › vmap over per-lane keys decorrelates lanes | a map over keys | V › randomness › a map over a batch of keys draws each key's values |
| old: test_rng.ml transformations › lane_index is lane 0 outside a transform | lane_index with no map | V › randomness › outside a map the lane index is 0 |
| old: test_rng.ml transformations › vmap over per-lane key scopes decorrelates lanes | a scope rooted at a mapped key | V › randomness › a scope rooted at a mapped key draws each key's values |
| old: test_rng.ml transformations › vmap lane_index decorrelates lanes | lane_index folded into a key | V › randomness › a key folded with the lane index draws per lane |
| old: test_rng.ml transformations › lane_index decorrelates lanes over devices | lane_index over devices | dropped: a map over a split axis under the compiled call (step 4, with the compiled call's suite) |
| old: test_jit.ml linear algebra › the gradient of a Cholesky-using loss compiles; linear algebra › the gradient of det compiles; linear algebra › the gradient of a QR-using loss compiles | factorisations' gradients compiled | Jc › compositions › the gradient of a Cholesky-using loss compiles; the gradient of det compiles; the gradient of a QR-using loss compiles |
| old: test_jit.ml indexed access › gradients through indices outside the axis; indexed access › gradient of take with repeated tokens; indexed access › gradient of top_k | indexed gradients compiled | Jc › compositions › gradients through indices outside the axis; the gradient of take with repeated tokens; the gradient of top_k lands on the chosen entries |
| old: test_jit.ml indexed access › scatter under vmap | a compiled map of scatter | Jc › compositions › a compiled map of scatter is its eager map |

## Compiled call

The suite is `Rune_next.Jit` (`next/test/test_jit.ml`), written `J`
below, over the host and test devices that share the host's memory; its
`swept › …` laws and the device and backend keys are slow. Kernel numerics
under a compiled call are the kernels' own: those rows name the `Compiled`
suite's laws, which run each kernel over every layout, beside `J › values ›
one operation per family equals eager › …`, which runs them through one
traced program. Rows the Metal suite and the staged scan will cover come with
them.

### test_jit.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit.ml jit basics › 64-bit constants keep every bit | 64-bit constants in a program | J › values › 64-bit integer constants keep every bit |
| old: test_jit.ml jit basics › narrow constants wrap; jit basics › folded integer constants wrap | integer constants wrap at their width | J › values › integer constants wrap at the operand's width |
| old: test_jit.ml jit basics › integer comparisons read wrapped values; jit basics › ordered comparisons are false at NaN | comparisons | Compiled › host › elementwise › comparisons |
| old: test_jit.ml jit basics › pow of a tensor base matches eager; jit basics › pow of a subnormal base | pow | Compiled › host, swept › elementwise › transcendental binary; › exact binary |
| old: test_jit.ml jit basics › float sums and products keep their grouping; jit basics › float constants keep their grouping; jit basics › float identities hold only where IEEE keeps them | IEEE arithmetic in a traced program | J › values › float identities hold only where IEEE keeps them; Compiled › host › reductions › reduce floats |
| old: test_jit.ml jit basics › max propagates NaN; jit basics › zeros keep their sign | extremes and signed zeros | J › values › one operation per family equals eager › neg, abs, max; Compiled › host › elementwise › exact binary |
| old: test_jit.ml jit basics › element-wise chain matches eager | a chain of elementwise operations | J › values › one operation per family equals eager › neg, abs, max |
| old: test_jit.ml jit basics › bitcast matches eager; jit basics › bitcast outputs retain their own dtype; jit basics › float8 bitcasts preserve raw bytes through movements | bitcast | dropped: a bitcast is a view in nx and reaches no kernel; a traced bitcast is a movement of its leaf, which J › keys › strides out of C order retrace once and J › values › one operation per family equals eager › a flip and a pad cover |
| old: test_jit.ml jit basics › replay reads fresh input data | a replay reads its new arguments | J › values › a replay reads its new arguments, and an earlier call's again |
| old: test_jit.ml jit basics › a new shape retraces | a new shape retraces | J › keys › another extent retraces once; › another rank retraces once |
| old: test_jit.ml jit basics › zero-size outputs are empty tensors | zero-size results | J › values › a zero-size result is an empty tensor |
| old: test_jit.ml jit basics › closure-captured weights (matmul) | a captured weight | J › captures › a captured tensor is a constant of the program |
| old: test_jit.ml jit basics › a structured result | structured results | J › values › a structured result equals eager's leaf by leaf |
| old: test_jit.ml jit basics › aliased input leaves are separate inputs | one value at two read leaves | J › results › one value passed at two read leaves is read at both |
| old: test_jit.ml keys › reports key programs; keys › a leafless element keys programs; keys › cases key programs | reports in the key | J › keys › another reported integer retraces once; › another case retraces once; › another list length retraces once; › an option's presence retraces once |
| old: test_jit.ml composition › grad inside jit matches eager grad; composition › jit under grad runs eagerly; composition › jit under vmap runs eagerly | the compiled call under and around the transformations | J › transformations › under a transformation a compiled function runs its function; › under grad a compiled function consumes nothing (the cells of every order are the composition suite's) |
| old: test_jit.ml composition › scan matches eager; composition › a scan over structured rows | a scan inside a trace | J › scans › a scan folds inside the trace and equals eager |
| old: test_jit.ml composition › grad through a scan matches eager; composition › grad through a scan, stacked outputs only; composition › grad through a scan, final carry only; composition › grad through a scan with a multi-leaf carry; composition › grad through a scan with an asymmetric pair carry; composition › grad through nested scans; composition › grad through a scan with a captured weight; composition › grad through a scan with a vector carry; composition › grad through a scan with an external input; composition › grad through a scan with external matrices; composition › grad through a scan with a matrix carry | gradients through a scan inside a trace | J › scans › a gradient through a scan equals eager's |
| old: test_jit.ml composition › shape-unstable carry unrolls instead of staging | a carry that changes shape | J › scans › a carry that changes its shape across steps is written out |
| old: test_jit.ml composition › a scan rejects ragged or scalar rows | the scan's refusals inside a trace | J › scans › an empty scan axis raises Rune.scan's message |
| old: test_jit.ml sliding windows › unfold matches eager; sliding windows › fold of unfold matches eager; sliding windows › sliding window matches eager; sliding windows › correlate matches eager | windows | Compiled › host › windows and products › unfold; › fold of floats; J › keys › overlapping windows retrace once |
| old: test_jit.ml reductions › half-precision sums accumulate wide; reductions › half-precision products multiply wide | narrow floats accumulate at float32 | Compiled › host › narrow floats accumulate at float32 › a float16 sum of 4096 ones is 4096; › a float16 product past the float16 range and back is exact |
| old: test_jit.ml reductions › narrow matrix products multiply exactly; reductions › vector products are matrix products | products | Compiled › host › narrow floats accumulate at float32 › a float16 contraction of 4096 ones is 4096; › windows and products › float products |
| old: test_jit.ml reductions › extremes | extremes | Compiled › host › reductions › reduce exactly |
| old: test_jit.ml cumulative reductions › small integer scans keep their dtype; cumulative reductions › long scans match eager; cumulative reductions › scans propagate NaN; cumulative reductions › scans order -0 below +0 | running reductions | Compiled › host › reductions › scan exactly; › scan floats |
| old: test_jit.ml cumulative reductions › 8-bit float scans along a long axis | 8-bit float scans | dropped: the host's renderer has no 8-bit float, Compiled › host › refusals of the host › an 8-bit float is refused on the host › float8_e4m3 |
| old: test_jit.ml indexed access › scatter matches eager; indexed access › scatter orders duplicate updates; indexed access › scatter orders thousands of duplicate updates; indexed access › scatter along a middle axis; indexed access › scatter with unique indices; indexed access › scatter with unique indices broken at one row; indexed access › scatter drops an update outside the axis; indexed access › scatter carries int and bfloat16 payloads | scatter | Compiled › host › indexed › scatter exactly; › scatter add of floats |
| old: test_jit.ml indexed access › gather of a narrowed comparison; indexed access › gathers read zero outside the axis; indexed access › take over a large table matches eager; indexed access › gathers keep -0; indexed access › an index outside the axis beside unit axes | gather | Compiled › host › indexed › gather |
| old: test_jit.ml indexed access › concatenation keeps every bit | concatenation | Compiled › host › edges › a concatenation of 17 pieces, a kernel of 18 arguments, keeps every bit |
| old: test_jit.ml indexed access › sorted values are the input's elements; indexed access › sort matches eager; indexed access › sort of every dtype matches eager | sorts | Compiled › host, swept › reductions › sort; › argsort |
| old: test_jit.ml indexed access › top_k matches eager; indexed access › top_k radix select matches eager; indexed access › top_k over a row of 2^20 entries; indexed access › compiled argsort is not quadratic | top_k and the cost of argsort | dropped: `Nx.top_k` is nx's composition of `sort` and `argsort` (Compiled › host, swept › reductions › argsort), and an argsort's cost is the lowering's |
| old: test_jit.ml indexed access › diag matches eager | diag | dropped: nx's composition of movements and a gather, Compiled › host › indexed › gather |
| old: test_jit.ml linear algebra › reduced QR matches eager; linear algebra › a zero-tail column takes no reflector | QR | Compiled › host, swept › linear algebra › qr |
| old: test_jit.ml linear algebra › cholesky matches eager in both triangles | Cholesky | Compiled › host, swept › linear algebra › cholesky |
| old: test_jit.ml linear algebra › triangular solve matches eager for every flag combination; linear algebra › triangular solve takes a vector right-hand side; linear algebra › triangular solve is batched | triangular solves | Compiled › host, swept › linear algebra › solve_triangular |
| old: test_jit.ml linear algebra › LU matches eager; linear algebra › solve and inv match eager | LU | Compiled › host, swept › linear algebra › lu |
| old: test_jit.ml errors › reading a traced value raises | reading a traced value | J › errors › reading a traced value raises Jit_error |
| old: test_jit.ml errors › traced values have no storage; errors › a leaked traced value raises | a traced value outside its call | J › errors › a traced value kept after the call raises on read |
| old: test_jit.ml errors › unsupported operations raise | an operation no target computes | J › errors › an operation no target computes raises Jit_error |
| old: test_jit.ml state › non-contiguous inputs fall back to copies | non-contiguous arguments | dropped: arguments are read in place, J › placement › a placed view is read where it lies; nx makes views only through movements, which a program expresses |
| old: test_jit.ml state › offset views read the right span | offset views | J › keys › an argument starting 4 bytes further within 16 bytes of memory retraces once; › an argument starting 16 bytes further shares the program |
| old: test_jit.ml state › outputs have their own storage | fresh results | J › results › every result leaf has storage of its own |
| old: test_jit.ml values › set with a traced window start replays the position; values › set at a traced corner over two axes; values › slice with a traced window start replays the position | a window at a position read when the call runs | J › lending › a window written at a position read when the call runs reuses the cache; Compiled › host › indexed › update |
| old: test_jit.ml values › set with static specs matches eager | set at static positions | J › lending › an indexed write takes the leaf it writes before any other result |
| old: test_jit.ml training › jitted training follows the eager trajectory | a training loop | J › lending › two programs alternating on one consumed state keep its storage |
| old: test_jit.ml placement › a placed value equals its argument; placement › a compiled function runs where its inputs live | a call runs where its arguments lie | J › placement › a call runs where its arguments lie, and leaves its results there |
| old: test_jit.ml placement › strided and offset values; placement › placed views bind without a copy; placement › windows bind from aligned offsets; placement › captured views bind | placed views read in place | J › placement › a placed view is read where it lies |
| old: test_jit.ml placement › a placed value feeds an input with no transfer | no transfer for a placed argument | J › placement › a placed argument feeds a call with no transfer |
| old: test_jit.ml placement › a resident value is returned as it is | a returned argument | J › results › a result that returns a read argument is a copy |
| old: test_jit.ml placement › on the host device; placement › a program on the host is on the host | the host | J › values › one operation per family equals eager › neg, abs, max |
| old: test_jit.ml placement › placement under grad, jvp, vmap and jit | placement through the transformations | J › transformations › under a transformation a compiled function runs its function |
| old: test_jit.ml placement › an unbound placed value is consumed | a consumed placed value | J › placement › a consumed split state is lent on every device |
| old: test_jit.ml placement › item reads one element; placement › a move to the host keeps its source; placement › mixed placements raise | reads, moves and mixed placements | dropped: nx's placement suite (`nx placement`) |
| old: test_jit.ml placement › an input on another device raises | operands on two devices | J › errors › operands on two devices raise nx's message |
| old: test_jit.ml placement › one device per name | one device per name | J › errors › a name met with two devices raises |
| old: test_jit.ml placement › a capture decides the device; placement › a capture's device is remembered | the device of a capture | J › captures › a capture decides the device of a call of host arguments |
| old: test_jit.ml placement › placed views share programs; placement › strides key programs | views in the key | J › keys › strides out of C order retrace once; › an argument starting 16 bytes further shares the program |
| old: test_jit.ml placement › views of inputs as outputs | a returned view | J › results › a result that returns a read argument is a copy |
| old: test_jit.ml placement › overlapping views are copied | overlapping windows | J › keys › overlapping windows retrace once |
| old: test_jit.ml placement › a window's view is released | a view released after a call | dropped: release is nx.device's (its suite); a program holds only its captures, J › captures › a capture placed where the call computes is bound, not uploaded |
| old: test_jit.ml device lists › * | eager placement over device lists | dropped: nx's placement suite (`nx placement`) |
| old: test_jit.ml compiled over device lists › a split input; compiled over device lists › views of split values are read in place | a split argument | J › placement › a split argument computes on each device, and stays split |
| old: test_jit.ml compiled over device lists › a consumed carry keeps its placement | a consumed split state | J › placement › a consumed split state is lent on every device |
| old: test_jit.ml compiled over device lists › leaves on other devices raise; compiled over device lists › operands that cannot meet raise | operands that cannot meet | J › errors › operands on two devices raise nx's message |
| old: test_jit.ml bound captures › binding a placed capture moves no bytes | a bound capture | J › captures › a capture placed where the call computes is bound, not uploaded |
| old: test_jit.ml bound captures › two compiled functions share one buffer | a buffer two programs bind | J › captures › two compiled functions share one captured buffer |
| old: test_jit.ml bound captures › a bound capture returned unchanged is a copy | a returned capture | J › results › a result that returns a capture is a copy |
| old: test_jit.ml bound captures › consuming a bound storage; bound captures › a consumed bound storage goes with its owners | consuming a captured storage | J › captures › a host capture another call consumes makes the program raise, naming its path |
| old: test_jit.ml bound captures › the collection budget counts every allocation; bound captures › a device that cannot allocate raises Out_of_memory | allocation budgets | dropped: nx.device's allocator and its budget (`nx runtime devices`) |
| old: test_jit.ml chunked transfers › * | transfers in chunks | dropped: `Nx_device.Buffer.copy` (`nx runtime devices`) |
| old: test_jit.ml residency › feedback chain moves no bytes | a result fed back moves no bytes | J › placement › a placed argument feeds a call with no transfer |
| old: test_jit.ml residency › forced handles feed current bytes; residency › handles feed other jitted closures; residency › handles feed new signatures without forcing; residency › grad over jit forces deferred arguments; residency › vmap over jit forces deferred arguments; residency › signature dispatch never forces; residency › dropped handles are reclaimed | deferred handles | dropped: a compiled call's results are values; nothing is deferred |
| old: test_jit.ml residency › the same handle can seed two leaves | one value at two leaves | J › results › one value passed at two read leaves is read at both |
| old: test_jit.ml residency › duplicate output leaves are two values | one value at two results | J › results › a value at two result leaves comes back as two values |
| old: test_jit.ml residency › empty values are consumed and returned fresh | empty values | J › values › a zero-size result is an empty tensor |
| old: test_jit.ml residency › pass-through outputs survive later calls | a returned argument survives | J › results › a result that returns a read argument is a copy |
| old: test_jit.ml residency › captures upload once across signatures | a host capture uploaded once | J › captures › a host capture of a call on a device is placed there once |
| old: test_jit.ml residency › a read after a call waits for it | a read waits | J › domains › two domains replay one program, each reading its own arguments |
| old: test_jit.ml residency › programs own their arenas; residency › dropping a program releases its arena; residency › a buffer freed under a running kernel is not reused | arenas | dropped: tolk.engine's linked storage (its suite) |
| old: test_jit.ml consumption › lending follows derivation; consumption › a consumed input hands its storage to the output | lending | J › lending › a result takes the consumed leaf it derives from at its own index; › a consumed host argument lends its storage to the result |
| old: test_jit.ml consumption › a consumed argument between read ones; consumption › a step reads its first argument; consumption › a step reuses only its state's storage | a consumed argument among read ones | J › lending › a window written at a position read when the call runs reuses the cache |
| old: test_jit.ml consumption › consumption bounds resident memory at two generations | two generations | J › lending › a loop consuming its state holds two generations of it |
| old: test_jit.ml consumption › a movement path refuses reuse and stays correct; consumption › a later reader refuses reuse and stays correct | a result read through a movement | J › lending › a result that reads its consumed leaf at other indices takes fresh storage; a result read through a flip of a leaf takes a leaf it does not read, and the leaf goes to a result derived at its own index |
| old: test_jit.ml consumption › a consumed pass-through moves its storage | an unchanged consumed leaf | J › lending › a consumed leaf returned unchanged is lent with no store |
| old: test_jit.ml consumption › a run-time window write reuses the cache; consumption › a pool read after its write still reuses storage | a window write | J › lending › a window written at a position read when the call runs reuses the cache |
| old: test_jit.ml consumption › two programs alternate on one consumed state | alternating programs | J › lending › two programs alternating on one consumed state keep its storage |
| old: test_jit.ml consumption › consumption reuses a pool written by scatter; consumption › scatter of values read from the consumed pool; consumption › scatter of the consumed pool into itself; consumption › scatter refuses a later reader of the consumed pool; consumption › scatter beside a reader of the old value | indexed writes into a consumed pool | J › lending › an indexed write takes the leaf it writes before any other result |
| old: test_jit.ml consumption › scatter without consumption keeps its input; consumption › an indexed write into a read leaf keeps it; consumption › an updated input returned unchanged stays readable; consumption › outputs never write into an input's buffer; consumption › jit never consumes its inputs | read arguments are never written | J › consumption › read arguments stay readable after any number of calls |
| old: test_jit.ml consumption › a partial view of a storage is not consumed | a consumed slice | J › consumption › a consumed slice raises before any work, consuming nothing |
| old: test_jit.ml consumption › a storage both arguments reach raises; consumption › a storage two leaves reach raises; consumption › a handle in both arguments raises | a storage two leaves reach | J › consumption › a consumed leaf that another leaf reaches raises before any work, naming both paths; › two consumed leaves over one storage raise before any work |
| old: test_jit.ml consumption › every derived leaf is reused | every derived leaf lends | J › lending › every leaf of a consumed state derived at its own index is lent |
| old: test_jit.ml consumption › a consumed handle raises on read; consumption › re-feeding a consumed handle raises; consumption › a value read before the call is still consumed | a consumed value raises | J › consumption › a consumed argument raises on read, naming its path; › a consumed argument raises as an operand and as an argument |
| old: test_jit.ml consumption › captures that reach consumed storage raise; consumption › a copied capture of consumed storage raises | a captured storage consumed | J › consumption › a consumed leaf whose storage the function captures raises |
| old: test_jit.ml consumption › a host input is consumed; consumption › a consumed host input holds its storage alone | a consumed host argument | J › lending › a consumed host argument lends its storage to the result |

### test_jit_alignment.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit_alignment.ml host memory in place › a capture at an address that is 4 modulo 16 | a capture at any address | J › keys › an argument starting 4 bytes further within 16 bytes of memory retraces once |

### test_jit_cuda.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit_cuda.ml cuda device › grad inside jit matches eager; cuda device › multi-kernel traces replay through compiled queues; cuda residency › *; cuda half › float16 softmax matches eager; cuda half › bfloat16 softmax matches eager; cuda half › float16 sandwich grad is fp32; cuda half › bfloat16 sandwich grad is fp32; cuda device lists › *; cuda rng › *; cuda consumption › * | the compiled call on CUDA | dropped: no CUDA device in the suite's devices yet; J's laws take any device, and every row has its host counterpart in J |

### test_jit_scratch.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit_scratch.ml partial consumption upgrade releases prior cells; consumption excludes alias reads; consumption excludes same-placement alias moves; shared reader excludes consumption | a consuming call excludes every other view | J › consumption › a consumed leaf that another leaf reaches raises before any work, naming both paths; › a view of consumed storage taken before the call raises on read |
| old: test_jit_scratch.ml capture pins storage during tracing; failed trace releases its capture pin | capture pins | J › captures › a capture placed where the call computes is bound, not uploaded; J › consumption › a call that raises while tracing consumes nothing |
| old: test_jit_scratch.ml jit releases its owner after trace failure; jit over devices releases its owner after trace failure | a failed trace leaves nothing behind | J › errors › a call that raised traces again at the next call |
| old: test_jit_scratch.ml independent replays keep their intermediates | replays from several domains | J › domains › two domains replay one program, each reading its own arguments |
| old: test_jit_scratch.ml concurrent transfers preserve accounting; independent uploads keep their bytes; concurrent device lookups keep one identity | transfers and device identity across domains | dropped: nx.device's (`nx runtime devices`) |

### test_device_lists.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_device_lists.ml numerics › a new shape retraces | a new shape retraces | J › keys › another extent retraces once |
| old: test_device_lists.ml residency › a split output in eager code; residency › operations over a batch split; residency › rows of a split table; residency › a roll by one slice | eager placement over a split | dropped: nx's placement suite (`nx placement`) |
| old: test_device_lists.ml errors › * | errors of placements | dropped: nx's placement suite (`nx placement`) |
| old: test_device_lists.ml consumption › consumed storage is lent on every device | lending on every device | J › placement › a consumed split state is lent on every device |

### test_jit_metal.ml

The kernel rows of this file are the Compiled section's.

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit_metal.ml metal device › grad inside jit matches eager | a gradient inside a compiled call | J › transformations › under a transformation a compiled function runs its function (the Metal cells are the composition suite's) |
| old: test_jit_metal.ml metal device › rematerialized second derivatives replay correctly; metal device › custom backward replays indexed scatter | rules under a compiled call | dropped: the compiled-rules suite's (`next/test/rules/jit`) |
| old: test_jit_metal.ml metal device › multi-kernel traces replay as compiled queues; metal device › a read after a call waits for it | a call queued on Metal, and its results read | J › metal › metal › a call runs where its arguments lie, and leaves its results there |
| old: test_jit_metal.ml metal device › command storage is released with its function; metal device › programs own their arenas | a program's storage | dropped: tolk.engine's linked storage (its suite) |
| old: test_jit_metal.ml metal device › two programs alternate on one consumed state | alternating programs | J › lending › two programs alternating on one consumed state keep its storage |
| old: test_jit_metal.ml placed weights › a placed view is read bit for bit | a placed view on Metal | J › metal › metal › a placed view is read where it lies |
| old: test_jit_metal.ml placed weights › placed weights bind and replay as compiled queues | bound weights on Metal | J › metal › metal › a capture placed where the call computes is bound, not uploaded |
| old: test_jit_metal.ml placed weights › consuming a bound storage | consuming a captured storage | J › captures › a host capture another call consumes makes the program raise, naming its path |
| old: test_jit_metal.ml placed weights › a step reads its weights and consumes its state | a step on Metal | J › metal › metal › a consumed placed argument lends its storage |
| old: test_jit_metal.ml placed weights › a capture resident on another device raises | a capture on another device | J › errors › operands on two devices raise nx's message |
| old: test_jit_metal.ml placed weights › placed views bind at any offset | views at any offset on Metal | J › metal › metal › a float16 argument starting 2 bytes further retraces once, and is read where it lies |
| old: test_jit_metal.ml reads, moves and loops › item on resident logits reads one element; reads, moves and loops › a move to the host keeps its source; reads, moves and loops › mixed placements raise; reads, moves and loops › a value moves between backends | reads and moves of placed values | dropped: nx's placement suite (`nx placement`) |
| old: test_jit_metal.ml reads, moves and loops › a loop whose state starts on the host compiles once | a state that moves to the device | J › keys › another device retraces once |

### test_jit_scratch.ml, continued

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jit_scratch.ml jit rejects overlapping cold calls; jit rejects reentrant replay; jit over devices rejects overlapping cold calls; jit over devices rejects reentrant replay | overlapping and reentrant calls | dropped: a compiled function now runs from several domains at once and traces through inside its own function, J › domains › two domains meeting one new key trace it once; J › transformations › a compiled function called inside another one's trace traces through |
| old: test_jit_scratch.ml destructive failure consumes old aliases | a call failing after its first kernel | dropped: no failure after a call's first kernel is observable on the suite's devices |
| old: test_jit_scratch.ml reads keep their resident owner alive; failed release preserves all owners | storage lifetime across reads | dropped: nx's storage claims (`nx placement`) |

### test_half.ml

The eager-against-compiled rows of this file are the Compiled section's.

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_half.ml astype sandwich › * | gradients through casts to and from narrow floats | dropped: the cast rows of the rule suites, and their compiled forms in the compiled-rules suite (`next/test/rules/jit`) |
| old: test_half.ml two devices › * | narrow floats over two devices | J › placement › a split argument computes on each device, and stays split |
| old: test_half.ml vmap › * | narrow floats under a map | dropped: the batching suite's rows at every dtype |

### test_remat_memory.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_remat_memory.ml jit (grad) under remat keeps under half the activations | a compiled gradient through remats keeps few activations | J › one device › a compiled gradient through remats keeps under half the activations |
