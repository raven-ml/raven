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

## Transformations core

The suites are in `next/test/`, one per directory, each through the public
`Rune` alone: `Rune derivatives` (`derivatives/`, written `D` below),
`Rune.scan` (`scan/`, `Sc`), `Rune structures` (`structure/`, `St`),
`Rune nesting` (`nesting/`, `N`), `Rune compositions` (`composition/`, `M`)
and `Rune constructs` (`constructs/`, `C`). Rows owned by the suites of the
operations' rules, of `vmap`, of the custom rules and of totals are listed in
those sections.

### test_grad.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_grad.ml reduction derivatives preserve zeros and ties; half reduction derivatives count ties without overflow | rules of `Reduce` | the rules' suite |
| old: test_grad.ml no_grad scopes are independent across domains; …across systhreads | `no_grad` per domain | dropped: `no_grad` is gone; what it guarded is N › independence › differentiations on two domains at once, and D › pullbacks › a pullback may run on two domains at once |
| old: test_grad.ml grad over records › aliased leaves are separate parameters | one tensor at two leaves | D › grad › a tensor behind two leaves is two parameters; D › jvp › a tensor behind two leaves has two tangents |
| old: test_grad.ml grad over records › a capture of the argument is a constant | captures are constants | D › grad › a capture that is also the argument is a constant |
| old: test_grad.ml grad over records › matches the analytic gradient | a record's gradient | D › grad › the gradient of a record is its analytic gradient |
| old: test_grad.ml grad over records › unused leaf has zero gradient | unused leaves | D › grad › a leaf the objective does not use has a gradient of +0. |
| old: test_grad.ml grad over records › preserves structure and shapes | the result's structure | D › grad › the gradient of a record is its analytic gradient; St › preconditions › integer, bool and key leaves beside a float one are carried |
| old: test_grad.ml grad over records › value_and_grad returns the value | the value | D › value › the value is the objective's, bit for bit |
| old: test_grad.ml grad over records › value_and_grad_aux returns auxiliary data | auxiliary results | D › value › an auxiliary result leaves through its structure as values; › an auxiliary result does not contribute to the gradient |
| old: test_grad.ml grad over records › mixed dtypes differentiate in one pass | float32 and float64 leaves | D › jacobians › jacfwd' has the result's dtype and jacrev' the argument's (a cast between them) |
| old: test_grad.ml grad over records › gradient descent converges | descent | D › grad › gradient descent on a square shrinks it by the step each time |
| old: test_grad.ml grad over records › rejects an integer single-tensor argument | no float leaf | St › preconditions › a structure with no real or complex tensor is refused |
| old: test_grad.ml grad over records › carries a non-differentiable leaf | carried leaves | St › preconditions › integer, bool and key leaves beside a float one are carried |
| old: test_grad.ml vjp › scales by the cotangent; accepts non-scalar outputs | pullbacks | D › vjp › the pullback scales by the cotangent |
| old: test_grad.ml vjp › pulls back structured cotangents | structured results | D › vjp › a structured result's pullback is the gradient of its pairing |
| old: test_grad.ml vjp › rejects a cotangent shape mismatch; rejects cotangents of another structure | cotangent checks | D › vjp › cotangents of another structure are refused at the pullback; › a cotangent of another shape is refused at the pullback; › a cotangent of another dtype is refused at the pullback |
| old: test_grad.ml vjp › vjp_fun pulls back a structured result | `vjp_fun` | D › vjp › a structured result's pullback is the gradient of its pairing (`vjp` returns the pullback) |
| old: test_grad.ml remat › 15 tests | `remat` under each transformation, with captures | M › pairs › every cell of the remat column; C › with no transformation › remat with no transformation runs its function once; the capture cases are the custom rules' and remat's suite |
| old: test_grad.ml remat › rejects a consumed argument | refusal | St › signatures › remat refuses a consumed argument when given its signature |
| old: test_grad.ml set › differentiates both operands | rule of `Update` | the rules' suite |
| old: test_grad.ml single-tensor variants › grad' matches the analytic gradient; vjp' pulls back the cotangent | `'` forms | D › shorthands › grad' is grad at one tensor; › vjp' is vjp at one tensor |
| old: test_grad.ml single-tensor variants › a bitcast has zero derivative | rule of `Bitcast` | the rules' suite |

### test_engine.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_engine.ml higher order › second derivative composes; third derivative composes | nested gradients | N › perturbation › the second derivative of a cube; › the third derivative of a fourth power |
| old: test_engine.ml gradient flow › detach stops the gradient | detach | D › detach › under grad a detached value has no derivative |
| old: test_engine.ml gradient flow › no_grad region is constant | `no_grad` | dropped: `no_grad` is gone; D › detach |
| old: test_engine.ml gradient flow › constants pass through unsupported ops | an operation with no rule on constants | the rules' suite (an operation with no rule, until its rule lands) |
| old: test_engine.ml error contracts › unsupported op raises when its input is tracked | an operation with no rule | the rules' suite |
| old: test_engine.ml error contracts › grad requires a scalar objective | scalar objective | St › preconditions › a non-scalar objective is refused; › a scalar objective may have any shape of one element |
| old: test_engine.ml statefulness › grad is repeatable | no state between calls | D › grad › two differentiations of one function give one gradient |
| old: test_engine.ml statefulness › value reads are transparent | reads inside the objective | D › grad › a value read inside the objective is its primal |
| old: test_engine.ml regressions › pad keeps its fill value under grad; sort routes gradient through the permutation | rules of `Pad` and `Sort` | the rules' suite |
| old: test_engine.ml backward pass › cotangents stay lazy views until a reshape or the result | no copies in a gradient | D › operations › a pair of cancelling transposes adds no copy and no arithmetic to a gradient |
| old: test_engine.ml debugging › with_debug logs ops and preserves results | `with_debug` | dropped: `with_debug` is gone; its replacement is an interpreter installed with `Nx.Op.intercept`, which D › operations uses |

### test_jacobian.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jacobian.ml pullbacks › pullback is reusable across cotangents | reuse | D › vjp › a pullback applied twice equals two fresh pullbacks |
| old: test_jacobian.ml pullbacks › pullback rejects a cotangent shape mismatch | checks | D › vjp › a cotangent of another shape is refused at the pullback |
| old: test_jacobian.ml gradient checking › accepts correct gradients; catches a wrong custom rule | `check_grads` | D › check_grads › a correct gradient is accepted; › a pullback twice the true one is caught |
| old: test_jacobian.ml jacobians › jacobians preserve float64; preserve float32; mixed-dtype jacobians follow tangent space dtypes | dtypes | D › jacobians › jacfwd' has the result's dtype and jacrev' the argument's; D › laws › jacfwd' equals jacrev' |
| old: test_jacobian.ml jacobians › jacobian matches the analytic matrix | values | D › jacobians › the Jacobian is its analytic matrix |
| old: test_jacobian.ml jacobians › jacobians restore input and output shapes | shapes | D › jacobians › the Jacobian's shape is the result's then the argument's |
| old: test_jacobian.ml jacobians › jacobians evaluate the function once | one run | D › jacobians › each runs the function once |
| old: test_jacobian.ml jacobians › hessian matches the analytic matrix | `hessian'` | D › jacobians › a Hessian is jacfwd' of grad' (`hessian'` is gone) |
| old: test_jacobian.ml jacobians › hvp agrees with the materialized hessian; structured hvp matches analytic | `hvp`, `hvp'` | D › jacobians › a Hessian-vector product is jvp of grad; N › perturbation › a Hessian-vector product, forward over reverse (`hvp` and `hvp'` are gone) |

### test_control.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_control.ml scan › running-sum scan is cumsum; returns the final carry; folds structures | the fold | Sc › fold › a running sum is a cumulative sum; › the result's carry is the last step's; › a structured carry, rows and outputs; › a fold with nothing to emit returns unit |
| old: test_control.ml scan › differentiates like the primitive; vectorizes over the batch | transformed scans | Sc › transformed › grad of a scan is grad of its primitive; › vmap of a scan is the cumulative sum of each row |
| old: test_control.ml scan › rejects a scalar input | refusal | Sc › refusals › a scalar row tensor is refused |
| old: test_control.ml scan › rejects a changed carry; rejects changed outputs; rejects a carry of another dtype | refusals | Sc › refusals › messages |
| old: test_control.ml scan › rejects a changed carry under jit; a staged scan under vmap and jvp rejects a changed carry | refusals under transformations | Sc › refusals › under transformations; Sc › compiled › a changed carry is refused under jit |
| old: test_control.ml scan › per-sample gradients; second-order gradients; hessian-vector product | nested transformations of a scan | Sc › transformed › per-row gradients of a scan; › second derivatives of a scan; › Hessian-vector products of a scan |
| old: test_control.ml cond › selects the branch by predicate; differentiates the taken branch | `cond` | dropped: `cond` is gone; D › grad › a branch on a value differentiates the branch taken |
| old: test_control.ml while_loop › iterates until the predicate fails; differentiates the taken iterations | `while_loop` | dropped: `while_loop` is gone; D › grad › a recursion on a value differentiates the iterations taken |
| old: test_control.ml exceptions › each transformation × each call, eager and compiled | an exception reaches its call | C › exceptions › under no transformation, grad, jvp, vmap, a total's scope, a remat's function (each over remat, custom_jvp, custom_vjp, scan); Sc › body; the compiled cells are the compiled call's suite. "grad of rerun code" is C › exceptions › under a remat's function, without `no_grad` |
| old: test_control.ml exceptions › an operation left unhandled inside a call | an unhandled effect is the code's | C › effects › an effect a construct's code leaves unhandled is that code's; Sc › body › an effect the body leaves unhandled is the body's |

### test_composition.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_composition.ml 7 compositions × 4 constructs | the matrix | M › pairs (every ordered pair of grad, jvp, vmap and jit over eight columns, against closed forms) and M › triples; the cells the old matrix left out (a custom rule in the mode it lacked) are cells that raise or, for `custom_jvp` under reverse mode, that compute |

### test_jvp.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jvp.ml jvp over records › matches the analytic tangent | a record's tangent | D › jvp › the tangent of a record is its analytic tangent |
| old: test_jvp.ml jvp over records › mixed dtypes propagate in one pass | dtypes | D › jacobians › jacfwd' has the result's dtype and jacrev' the argument's |
| old: test_jvp.ml jvp over records › constant function has zero tangent | constants | D › jvp › a function of nothing it differentiates has a zero tangent |
| old: test_jvp.ml jvp over records › jvp_aux returns auxiliary data | `jvp_aux` | dropped: `jvp_aux` is gone; the value is part of `jvp`'s result structure |
| old: test_jvp.ml jvp over records › rejects tangent shape mismatch | checks | D › jvp › a tangent of another shape is refused |
| old: test_jvp.ml jvp over records › agrees with grad on scalar objectives | one derivative | D › laws › jvp along v is the gradient paired with v |
| old: test_jvp.ml jvp over records › gives per-leaf output tangents | structured results | D › jvp › each result leaf has its own tangent |
| old: test_jvp.ml composition › hessian-vector product; grad of jvp; nested jvp | nesting | N › perturbation › a Hessian-vector product, forward over reverse; › a gradient of a tangent, reverse over forward; › a tangent of a tangent |
| old: test_jvp.ml gates and errors › no_grad stops tangents | `no_grad` | dropped: `no_grad` is gone |
| old: test_jvp.ml gates and errors › detach stops tangents | detach | D › detach › under jvp a detached value has no tangent |
| old: test_jvp.ml gates and errors › rejects a leaf tangent shape mismatch; rejects tangents of another structure | checks | D › jvp › tangents of another structure are refused; St › mismatches |
| old: test_jvp.ml gates and errors › unsupported op raises when input is active | an operation with no rule | the rules' suite |
| old: test_jvp.ml operands without a tangent › a product by a constant has a finite tangent at an infinite operand | no term for a constant | D › edges › a constant operand adds no term at an infinite argument |
| old: test_jvp.ml operands without a tangent › a power with a constant exponent has tangent 0 at a zero base | rule of `Pow` | the rules' suite |
| old: test_jvp.ml the rule groups | each rule | the rules' suite |

### test_complex.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_complex.ml convention › 9 tests | the complex convention | D › complex (one test each, same claims) |
| old: test_complex.ml the rule groups | each rule on complex operands | the rules' suite |

### test_read_lifetime.ml

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_read_lifetime.ml a read of a placed temporary under collection | nx's read path | dropped: no transformation is involved; nx's placement suite |

## Rule tables

The forward-mode suite is `next/test/rules/jvp/test_jvp_rules.ml`, written
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


### Old rune tests

| Source | Behaviour | Outcome |
|---|---|---|
| old: test_jvp.ml unary rules › neg, exp, log, sqrt, recip, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, abs, erf | each unary tangent | J › unary <kind> › the tangent agrees with a central difference; › the tangent is the derivative's closed form |
| old: test_jvp.ml binary rules › add, sub, mul, div, pow, maximum, minimum, atan2 | each binary tangent | J › binary <kind> › the tangent agrees with a central difference (div is fdiv), for each operand carrying a tangent |
| old: test_jvp.ml operands without a tangent › a power with a constant exponent has tangent 0 at a zero base | no NaN from the exponent's coefficient | J › edges › binary pow › at a zero base › x ** 2 has tangent 0 |
| old: test_jvp.ml operands without a tangent › a product by a constant has a finite tangent at an infinite operand | no term for a constant operand | J › edges › binary mul › a constant's infinite coefficient never meets a zero |
| old: test_jvp.ml reduction rules › sum over one axis; sum keepdims; prod over one axis; max over one axis; min over one axis | reduction tangents | J › reduce <kind> › the tangent agrees with a central difference (axes drawn; keepdims is a reshape of the result) |
| old: test_jvp.ml movement rules › reshape; transpose; shrink; flip; sliding window | movement tangents | J › move reshape, move permute, move shrink, move flip, move window › the tangent agrees with a central difference; › a linear operation's tangent is the operation on the tangent |
| old: test_jvp.ml movement rules › sliding window tangent is the windowed tangent | a window's tangent is the windowed tangent | J › move window › a linear operation's tangent is the operation on the tangent |
| old: test_jvp.ml movement rules › pad; concatenate | pad and cat tangents | J › pad; J › cat (the same laws); J › edges › pad › the tangent's fill is zero; J › edges › cat › a piece with no tangent contributes zeros |
| old: test_jvp.ml selection rules › where; take_along_axis | selection tangents | J › where; J › gather (the same laws) |
| old: test_jvp.ml selection rules › sort | a sort's tangent | J › sort (the same laws); J › edges › sort › the tangent is the tangent gathered by the primal's argsort, bit for bit |
| old: test_jvp.ml scan rules › cumsum; cumprod; cummax; cummin | scan tangents | J › scan <kind> › the tangent agrees with a central difference |
| old: test_jvp.ml scan rules › cummax carries the tangent of each running maximum's element | the running extremum's element's tangent | J › cumulative › running extrema › cummax: the tangent is each running extremum's element's, bit for bit; › of equal elements a running extremum takes the convention's |
| old: test_jvp.ml scan rules › cumprod is exact at zeros | no division in the running product's rule | J › cumulative › running products › cumprod is exact at zeros |
| old: test_jvp.ml matmul rules › 2d x 2d; batched x batched; 2d x batched; batched x 2d | matmul tangents | J › matmul › the tangent agrees with a central difference (leading axes drawn, extent-1 axes broadcast, ranks that differ) |
| old: test_jvp.ml linalg rules › cholesky; cholesky (batched) | cholesky's tangent | J › cholesky › the tangent agrees with a central difference |
| old: test_jvp.ml linalg rules › cholesky reads the lower triangle | the tangent of the unread triangle | J › edges › cholesky › a tangent above the diagonal has no effect, in both modes |
| old: test_jvp.ml linalg rules › lu (square, batched) | lu's tangent | J › lu › the tangent agrees with a central difference (pivots drawn with a margin) |
| old: test_jvp.ml linalg rules › solve_triangular (batched vector rhs) | the triangular solve's tangent | J › solve_triangular › the tangent agrees with a central difference (flags, vector and matrix right-hand sides, batches drawn); J › edges › solve_triangular › a tangent in the triangle the solve does not read has no effect |
| old: test_jvp.ml gates and errors › unsupported op raises when input is active | a tangent with no rule raises | dropped: every operation has a tangent rule; the tangents with no definition raise in J › factorisations › undefined › a complete SVD of a non-square matrix has no tangent, and J › edges › qr › a complete factorisation of a tall matrix has no tangent |
| old: test_complex.ml holomorphic rules › recip, sqrt, exp, log, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, mul, fdiv, pow, matmul, reduce_prod (forward) | complex tangents | J › <row> › on complex values the tangent agrees with a central difference |
| old: test_complex.ml modulus › abs (forward) | the modulus's real-linear tangent | J › unary abs › on complex values the tangent agrees with a central difference |
| old: test_complex.ml sign › sign (forward) | the complex sign's tangent | J › unary sign › on complex values the tangent agrees with a central difference |
| old: test_complex.ml transforms › fft; ifft; irfft, even, odd and truncated (forward) | transform tangents | J › fft (inverse drawn); J › irfft (output sizes drawn) › on complex values the tangent agrees with a central difference |
| old: test_complex.ml linear and movement rules › neg, sum, cumsum, cumprod, cat, gather, flip, where (forward) | complex linear tangents | J › <row> › on complex values the tangent agrees with a central difference |
| old: test_complex.ml arithmetic › add, sub, matmul batched, matmul vector (forward) | complex arithmetic tangents | J › binary add, binary sub, matmul › on complex values the tangent agrees with a central difference |
| old: test_complex.ml movements › reshape, transpose, pad, shrink, sliding window, scatter set, scatter add (forward) | complex movement tangents | J › <row> › on complex values the tangent agrees with a central difference |
| old: test_complex.ml triangular solves › lower, upper, lower transposed, upper transposed, unit diagonal transposed, vector transposed (forward) | the conjugate transpose in the solve's tangent | J › solve_triangular › on complex values the tangent agrees with a central difference (all 8 flag combinations drawn) |
| old: test_complex.ml cholesky › lower; upper (forward) | the Hermitian factor's tangent | J › cholesky › on complex values the tangent agrees with a central difference; J › edges › cholesky › an imaginary tangent on the diagonal has no effect |
| old: test_complex.ml factorisations › lu (forward) | complex lu | J › lu › on complex values the tangent agrees with a central difference |
| old: test_grad.ml reduction derivatives preserve zeros and ties (forward and eager reverse) | products at zeros, shared ties | J › edges › reduce prod › one zero leaves the product of the others, two leave zero; J › edges › reduce max, reduce min › tied elements share the derivative |
| old: test_grad.ml half reduction derivatives count ties without overflow (eager) | a float16 tie count | J › edges › reduce max, reduce min › a float16 tie count above 65,504 does not overflow |
| old: test_grad.ml single-tensor variants › a bitcast has zero derivative | a bitcast carries no tangent | J › bitcast › it has no tangent and passes no cotangent |
| old: test_fft.ml forward mode › rfft tangent is rfft of the tangent; irfft tangent is irfft of the tangent | linear transform tangents | J › rfft, irfft › a linear operation's tangent is the operation on the tangent |
