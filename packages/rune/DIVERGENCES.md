# Divergences

rune follows JAX for what its transformations mean and tinygrad, through
tolk, for how it lowers nx's operations. This file lists where it departs
from either: each entry names the code (file and function), what differs
and why, and the test that pins it. An entry goes when its reason goes,
and its number is not reused.

A test is named by its module's suite (`packages/rune/test/test_<module>.ml`)
and its path in it.

## Transformations

The reference is JAX's `jax.lax` and `jax.vmap`.

### T1. A switch index out of range raises

- **Reference:** `lax.switch` clamps its index into `0 .. n-1`.
- **Raven:** `program.ml` (`default`, its `Switch` case); `compile.ml`
  (`switch`); `vmap.ml` (`switch_apart`).
- **Differs:** an index outside `0 .. n-1` raises `Invalid_argument` at the
  call, under a `jit` as the call returns, naming the lane under a map.
- **Reason:** a clamped index takes the first or last branch with no error.
- **Pinned by:** `Switch › an index out of range raises`.

### T2. An iterate is bounded and differentiates in reverse

- **Reference:** `lax.while_loop` runs any number of steps and has no reverse
  derivative; a loop differentiated in reverse is a `scan` of fixed length.
- **Raven:** `rune.ml` (`iterate_named`); `trace.ml` (`iterate_parts`);
  `program.ml` (`iterate`); `compile.ml` (`loop`).
- **Differs:** an iterate takes a `max` and raises when `until` still fails
  after `max` steps. Reverse mode splits it into a known iterate, which stacks
  each step's residuals, and a linear scan over those rows. Where the loop
  runs at the call and reads no other transformation's value, the stacks hold
  the steps taken; inside a traced function, a map or forward mode, they hold
  `max` rows and the scan runs the steps taken, so a capture receives a zero
  cotangent from the rows not taken. `linear_transpose`
  transposes an iterate whose stop reads nothing computed from the argument.
- **Reason:** a solver's loop gets its gradient without a fixed trip count.
  Inside a traced function, a map or forward mode, every shape is fixed
  before the steps are known, so the stack needs a bound, and `max` is that
  bound.
- **Pinned by:** `Iterate › derivatives › iterate's derivatives are its
  loop's`, `› the steps taken`, `› linear_transpose over an iterate`,
  `Iterate › memory › reverse mode keeps the steps taken`.

### T3. Lanes that part are held

- **Reference:** under `vmap`, a `lax.switch` whose index has lanes runs every
  branch over every lane and selects. A `lax.while_loop` whose test has lanes
  runs the body over every lane until the last stops, selecting each stopped
  lane's carry: its documentation is silent there, and its batching rule
  (`jax/_src/lax/control_flow/loops.py`, `_while_loop_batching_rule`) does
  this.
- **Raven:** `vmap.ml` (`donors`, `lend`, `switch_apart`, `iterate`,
  `masked_scan`).
- **Differs:** a branch no lane takes runs nothing. A lane that did not take a
  branch, or has stopped, is held: it computes at its donor, the first lane
  that runs, on the donor's values detached, its own discarded. Its checks
  pass, its additions to a total are dropped, and its `lane_index` is its
  donor's.
- **Reason:** `switch` promises that a branch not taken computes nothing: no
  code runs at a point no lane reached, so a held lane's checks, totals and
  values never see its untaken inputs. A shared capture still receives a held
  lane's zero cotangent times its donor's partial derivative, which is NaN
  where that partial is infinite.
- **Pinned by:** `Switch › the untaken branch computes nothing`, `› a lane
  computes only its branch`, `Iterate › maps › a stopped lane computes its
  donor's step`, `› a stopped lane keeps a non-finite carry`.

## Lowering

rune lowers each nx kernel operation to tolk's UOps (`lower*.ml`). Where
tinygrad's `Tensor` builds the same operation, the lowering builds tinygrad's
decomposition if it computes what nx states. This section lists every place
where it builds something else, or builds what tinygrad has no source for,
except these, not yet recorded: the complex kinds (`lower_complex.ml`), the
transforms (`lower_fft.ml`), `eigh`, logsumexp and moments, `expm1` and
`log1p`, and scaled and segmented contractions.

An entry is admitted for one of three reasons only:

- **(a) an OCaml constraint;**
- **(b) nx's stated meaning:** the operation's contract in `Nx_kernel`
  (`packages/nx/lib/kernel/nx_kernel.mli`, `prog.mli`), which nx's eager
  kernels compute;
- **(c) the engine's contract:** rig's submission protocol.

Each entry also gives the reference (tinygrad `79af1ca70`, or none), nx's
meaning, the agreement class (exact, a NaN result being any NaN; an ulp
bound; a rounded sum; a stated bound) and its reason. An ulp bound is the one
`lower_real.mli` states, which `Lower_real`'s suite holds on the host; a
rounded sum's is `lower_reduce.mli`'s; a matrix routine's is
`lower_linalg.mli`'s.

### A1. Narrow floats compute at float32

- **Reference:** `mixin/elementwise.py`, each operation at the operands' dtype.
- **Raven:** `lower_arith.ml` (`narrow`, `decode`, `encode`, `narrowed`).
- **Differs:** an operation or comparison on `float16`, `bfloat16`, an 8-bit
  float or `float4_e2m1fn` decodes its operands exactly to `float32` by integer
  formulas on their bits, computes, and encodes the result once, to nearest
  with ties to even. Neither a renderer's half-precision intrinsics nor a
  target's own conversions apply, so no target decides a tie, a subnormal or a
  NaN's bits.
- **nx:** `prog.mli`, Results: floats narrower than `float32` compute in
  `float32` and round once to their dtype.
- **Class:** as the operation's; a transcendental kind within one rank of the
  correctly rounded result.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › exact unary kinds give eager's bits`, `› exact
  binary kinds, comparisons, where and fma give eager's bits`, `› narrow
  floats' transcendental kinds are within two ranks of eager's`.

### A2. Integer arithmetic wraps

- **Reference:** `mixin/elementwise.py:74,84,103,125` (`neg`, `add`, `sub`,
  `mul`) and `pow`.
- **Raven:** `lower_arith.ml` (`lift`, `store`, `int_unary`, `int_pow`,
  `int_binary`).
- **Differs:** signed `Neg`, `Abs`, `Add`, `Sub`, `Mul`, `Fma` and integer
  `Pow` compute on the unsigned bit pattern: `uint32` for widths below 32 bits
  (C, Metal and CUDA promote them to `int`, whose products overflow), the
  unsigned type of the width otherwise. tinygrad computes in the signed type,
  whose overflow C leaves undefined. Integer `Pow` squares over the exponent's
  bits; tinygrad refuses an exponent that is not a constant.
- **nx:** `prog.mli`, `binary`: integers wrap; `Pow` of a negative exponent is
  0, but 1 for a base of 1 and `±1` for a base of -1.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › exact unary kinds give eager's bits`, `› exact
  binary kinds, comparisons, where and fma give eager's bits`.

### A3. Integer reciprocal

- **Reference:** `mixin/elementwise.py:460` (`reciprocal`).
- **Raven:** `lower_arith.ml` (`int_unary`, its `Recip` case).
- **Differs:** on integers, `x` at 1 and -1, 0 elsewhere and at 0. tinygrad's
  reciprocal is a float operation.
- **nx:** `prog.mli`, `Recip`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › exact unary kinds give eager's bits`.

### A4. Absolute value

- **Reference:** `mixin/elementwise.py:911` (`x * sign x`).
- **Raven:** `lower_bits.ml` (`magnitude`), through `lower_real.ml`
  (`unary`); `lower_arith.ml` (`int_unary`).
- **Differs:** a float's sign bit is cleared, so `abs (-0.)` is `0.` and a
  NaN keeps its payload; tinygrad's product gives `-0.` at `-0.`. A signed
  integer is negated modularly (A2).
- **nx:** `prog.mli`, Results: `Abs` sets a NaN's sign bit as a number's, its
  payload kept.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › abs`, `Lower_arith › exact unary kinds
  give eager's bits`.

### A5. Sign of NaN

- **Reference:** `mixin/elementwise.py:901` (`sign`).
- **Raven:** `lower_real.ml` (`sign`).
- **Differs:** `sign nan` is NaN; tinygrad gives 1.
- **nx:** `prog.mli`, `Sign`: NaN for a NaN.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › sign`.

### A6. Rounding half away from zero

- **Reference:** `mixin/elementwise.py:890` (`round`, half to even).
- **Raven:** `lower_real.ml` (`round`).
- **Differs:** a half rounds away from zero, decided on the exact rest
  `x - trunc x`.
- **nx:** `prog.mli`, `Round`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › round`.

### A7. Exponential in two parts

- **Reference:** `mixin/elementwise.py:511` (`exp2 (x * 1/ln 2)`).
- **Raven:** `lower_real.ml` (`exp_parts`, `exp2_split`, `exp`).
- **Differs:** rounding the product `x log2 e` costs up to about `|x|` ulps.
  The product is kept in two parts, its error exact from an fma, and
  `2^(hi + lo)` is `2^hi (1 + lo ln 2)`. A positive `hi` takes `2^(hi - 1)`,
  doubled after the product, since `hi` may round up to where `2^hi`
  overflows while the result does not.
- **nx:** `prog.mli`, Results, `Exp`.
- **Class:** ulp bound: 9 at `float32` and `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › exp`.

### A8. Cosine reduced by a quarter turn

- **Reference:** `mixin/elementwise.py:500` (`sin (pi/2 - x)`).
- **Raven:** `lower_real.ml` (`reduce`, `sin_cos`, `cos`).
- **Differs:** `pi/2 - x` rounds first, so tinygrad's cosine is 0 at
  `f32(pi/2)`, where it is `-4.4e-8`. The lowering reduces `|x|` to
  `q pi/2 + r`, `|r| <= pi/4`, by tolk's Cody-Waite reduction below 30 and its
  Payne-Hanek reduction from there, and picks `sin r` or
  `cos r = 1 - 2 sin^2 (r/2)` by the quadrant: the target's sine sees only `r`.
  The sine itself is tolk's `Sin`, as tinygrad's.
- **nx:** `prog.mli`, Results, `Cos`.
- **Class:** ulp bound: 9.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › cos`.

### A9. Tangent of the reduced argument

- **Reference:** `mixin/elementwise.py:921` (`sin / cos`).
- **Raven:** `lower_real.ml` (`tan`).
- **Differs:** A8's reduction, then `sin r / cos r` or `-cos r / sin r` by the
  quadrant's parity, with `x`'s sign.
- **nx:** `prog.mli`, Results, `Tan`.
- **Class:** ulp bound: 19.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › tan`.

### A10. Arcsine and arccosine by the arctangent

- **Reference:** `mixin/elementwise.py:931,944` (Abramowitz-Stegun 4.4.46,
  `acos = pi/2 - asin`).
- **Raven:** `lower_real.ml` (`asin`, `acos`).
- **Differs:** tinygrad's polynomial is float32-grade, has no relative accuracy
  near 0, and `pi/2 - asin` cancels near 1. The lowering takes
  `asin x = atan (x / sqrt (1 - x^2))`, `1 - x^2` by an fma, and
  `acos x = 2 atan (sqrt ((1 - x) / (1 + x)))`, through A11.
- **nx:** `prog.mli`, Results, `Asin`, `Acos`.
- **Class:** ulp bound: 5 at `float32`, 9 at `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › asin`, `› acos`.

### A11. Arctangent

- **Reference:** `mixin/elementwise.py:954` (`asin (x / sqrt (1 + x^2))`).
- **Raven:** `lower_real.ml` (`atan_unit`, `angle`, `atan`).
- **Differs:** `x^2` overflows past `1.8e19` in `float32`, where tinygrad's
  arctangent is 0. The lowering takes the arctangent of `min (|x|, 1/|x|)`,
  in `[0, 1]`, by a polynomial (at `float64` after the half-angle step), and
  `pi/2` in two parts less it past 1, with `x`'s sign.
- **nx:** `prog.mli`, Results, `Atan`.
- **Class:** ulp bound: 2 at `float32`, 7 at `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › atan`.

### A12. Two-argument arctangent

- **Reference:** none.
- **Raven:** `lower_real.ml` (`atan2`).
- **Differs:** a raven composition: A11's angle of `|y|` over `|x|`, `pi` in two
  parts less it where `x`'s sign bit is set, `y`'s sign, and C's values for
  both zeros and both infinities.
- **nx:** `prog.mli`, `Atan2`.
- **Class:** ulp bound: 6 at `float32`, 8 at `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › binary › atan2`.

### A13. Hyperbolic functions

- **Reference:** `mixin/elementwise.py:1034,1046` (`(e^x -+ e^-x) / 2`), `:757`
  (`2 sigmoid (2x) - 1`).
- **Raven:** `lower_real.ml` (`half_exp`, `cosh`, `sinh`, `tanh`).
- **Differs:** `sinh` and `tanh` cancel near 0 (`tanh` is 0 below about
  `6e-8`), and `sinh` and `cosh` overflow for `x` in `(88.72, 89.42]` in
  `float32`. Below `|x| = 20` the lowering takes `cosh` as
  `(e^|x| + 1/e^|x|) / 2`, `sinh` as `(t + t / (t + 1)) / 2` and `tanh` as
  `t / (t + 2)`, `t` the `expm1` of `|x|` or `2|x|`, which do not cancel. From
  20 `cosh` and `|sinh|` are `2^(|x| log2 e - 1)`, which overflows only with
  the result, and `|tanh|` is 1.
- **nx:** `prog.mli`, Results, `Sinh`, `Cosh`, `Tanh`.
- **Class:** ulp bound: `sinh` 21, `cosh` 10, `tanh` 18.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › sinh`, `› cosh`, `› tanh`.

### A14. Error function

- **Reference:** `mixin/elementwise.py:1058` (Abramowitz-Stegun 7.1.26).
- **Raven:** `lower_real.ml` (`erf32`, `erf_near64`, `erf_far64`, `erf`).
- **Differs:** tinygrad's approximation has an absolute error of `1.5e-7` and
  no relative accuracy near 0. The lowering takes a polynomial per range: at
  `float32`, one in `x^2` below 1 and one in `|x| - 2.5` to 4; at `float64`,
  two up to 2.5, then `1 - e^(G(a) - a^2)`, `a` held at 5.95, where `erf`
  rounds to 1.
- **nx:** `prog.mli`, Results, `Erf`.
- **Class:** ulp bound: 2 at `float32`, 3 at `float64`, measured.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › erf`.

### A15. True division

- **Reference:** `mixin/elementwise.py:229` (`a * (1/b)`).
- **Raven:** `lower_real.ml` (`( /: )`, through `binary`).
- **Differs:** one rounding: `Ops.FDIV`, which tolk renders as `a / b`.
  tinygrad multiplies by the reciprocal, two roundings.
- **nx:** `prog.mli`, `Fdiv`: the IEEE 754 quotient.
- **Class:** exact where the target's division is correctly rounded.
- **Reason:** (b).
- **Pinned by:** `Lower_real › binary › fdiv`.

### A16. Integer quotient and remainder by 0 and -1

- **Reference:** `mixin/elementwise.py:229,216` (`Ops.CDIV`, `Ops.CMOD`).
- **Raven:** `lower_arith.ml` (`int_div`).
- **Differs:** a quotient by 0 is 0 and a remainder by 0 the dividend; a
  quotient by -1 is the modular negation, so the least integer's is itself,
  and a remainder by -1 is 0. The divisor is made 1 in those lanes before
  dividing, since a kernel computes both sides of a selection and C traps or
  leaves them undefined.
- **nx:** `prog.mli`, `Idiv`, `Mod`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › exact binary kinds, comparisons, where and fma
  give eager's bits`.

### A17. Exact float remainder

- **Reference:** `mixin/elementwise.py:216` (`x - trunc (x * (1/y)) * y`).
- **Raven:** `lower_real.ml` (`fmod`).
- **Differs:** tinygrad's composition rounds three times and fails for
  `|x/y| >= 2^24`. The lowering computes C's exact `fmod` on the significands
  as integers: `(mx 2^d) mod my`, `d` the exponent difference, reduced 40 bits
  of `d` at a time in `float32` (7 steps) and 11 in `float64` (186 steps), each
  shifted remainder in 64 bits, then scaled by `y`'s exponent in two exact
  steps. The steps are unrolled.
- **nx:** `prog.mli`, `Mod`: `fmod` on floats.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › binary › mod`.

### A18. Power with its product in two parts

- **Reference:** `mixin/elementwise.py:548` (`Ops.POW`, `xpow` =
  `exp2 (y log2 |x|)`).
- **Raven:** `lower_real.ml` (`pow`).
- **Differs:** rounding `y log2 |x|` costs up to about its magnitude in ulps.
  The product's error, exact from an fma, is kept, and the two parts are
  exponentiated as A7's. A negative base takes the sign of an odd integral
  exponent and is NaN for a fractional one, and the special values are C99
  F.9.4.4's.
- **nx:** `prog.mli`, `Pow` and Results.
- **Class:** ulp bound: `9 + 6 |y log2 |x||`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › binary › pow`.

### A19. Maximum and minimum

- **Reference:** `mixin/elementwise.py:378,393` (`Ops.MAX`; `minimum` as
  `-max(-x, -y)`).
- **Raven:** `lower_real.ml` (`maximum`, `minimum`).
- **Differs:** NaN where an operand is NaN; otherwise the greater (lesser)
  operand, and of equal operands the AND (OR) of their bits, so that `-0.`
  orders below `0.`. tinygrad's `MAX` keeps the larger operand by comparison,
  which a NaN never is, and leaves a zero's sign to the target.
- **nx:** `prog.mli`, `Maximum`, `Minimum`: NaN propagates and `-0` orders
  below `+0`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › binary › maximum`, `› minimum`.

### A20. Less than or equal

- **Reference:** `mixin/elementwise.py:330` (`not (x > y)`).
- **Raven:** `lower_real.ml` (`le`, through `compare`).
- **Differs:** `x < y or x = y`, false on NaN; tinygrad's is true on NaN.
- **nx:** `prog.mli`, Domains: every comparison with a NaN is false but
  `Not_equal`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › compare › less_equal`.

### A21. Saturating conversion to integers

- **Reference:** `mixin/dtype.py:19` (`cast`).
- **Raven:** `lower_arith.ml` (`truncate`).
- **Differs:** a float converted to an integer is held at the integer's range
  and NaN is 0; the conversion itself sees only values in range. C leaves both
  undefined.
- **nx:** `dtype.mli`, Stores: integers truncate toward zero and saturate to
  their range.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › casts give eager's bits`.

### A23. A double narrows through float32 rounded to odd

- **Reference:** `mixin/dtype.py:19` (`cast`).
- **Raven:** `lower_arith.ml` (`float_odd`, `odd64`, `store_real`).
- **Differs:** a `float64` converted to a narrow float is first narrowed to
  `float32` rounded to odd (truncated, its last bit set when a discarded bit
  was), so that the conversion rounds once; a 64-bit integer goes through its
  `float64` rounded to odd, from its two exact halves. tinygrad's host
  conversion goes through `float32` rounded to nearest and rounds twice near a
  tie: `bfloat16 (1 + 2^-8 + 2^-30)` comes out 1.
- **nx:** `prog.mli`, `Cast`: an integer's exact value, or a float, rounded
  once.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › casts give eager's bits`, whose `float64`
  operands lie one ulp around each narrow format's ties.

### A24. Logarithm by fdlibm, NaN below zero

- **Reference:** `mixin/elementwise.py:830` (`log2 (x) * ln 2`), whose `log2`
  is `codegen/decomp/transcendental.py:219` (`xlog2`) on a target without it.
- **Raven:** `lower_real.ml` (`log`).
- **Differs:** the product by `ln 2` adds its rounding to `log2`'s error, two
  ulps near 1. The lowering builds fdlibm's `log` of the exact kinds, reading
  no `log2`. `xlog2` takes `x` for `-0.` where its reciprocal is `-inf`, as for
  a negative subnormal whose reciprocal overflows, and a target that flushes
  subnormals takes every negative subnormal for `-0.`: the lowering selects
  NaN where the bits of `x` have the sign bit and a magnitude.
- **nx:** `prog.mli`, Results, `Log`: NaN below zero, `-inf` at either zero.
- **Class:** ulp bound: 1, on every target.
- **Reason:** (b).
- **Pinned by:** `Lower_real › unary › log`, `› log within one ulp near 1`.

### R1. A float sum adds +0.

- **Reference:** `mixin/reduce.py:20` (`sum`), `mixin/op.py:758`
  (`_split_cumalu`).
- **Raven:** `lower_reduce.ml` (`arith`, `negative_zeros`).
- **Differs:** the lowering adds `+0.` to each float sum it computes, once per
  output, for reductions and scans. A kernel starts a loop's accumulator from
  `+0.`, but sums the terms alone when no loop is left (an axis of one
  element, or one it unrolls whole), so tinygrad's sum of `[-0.]` is `-0.`. A
  running sum from a carry is `-0.` where every term so far and the carry are.
- **nx:** `lower_reduce.mli`: a sum whose terms are all `-0` is `+0`, and the
  sum of no term `+0`.
- **Class:** rounded sum.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › reductions › float sums hold their bound`,
  `› scans › float running sums and products hold their bound`.

### R2. Sums and products accumulate in the carrier, integers unsigned

- **Reference:** `mixin/reduce.py:20` (`sum`: the `sum_acc_dtype`
  accumulator, converted back for the narrow floats only), `:47` (`prod`: at
  the operand's dtype).
- **Raven:** `lower_reduce.ml` (`folded`, `lift`, `store`, `arith`, `reduce`,
  `scan`).
- **Differs:** a sum and a product both accumulate in `Nx_kernel.carrier`'s
  dtype, integers on the unsigned pattern of 32 or 64 bits, and convert once
  to the destination's dtype. tinygrad accumulates signed integers in a signed
  type, whose overflow C, Metal and CUDA leave undefined; returns an integer
  sum in its accumulator's dtype; and multiplies narrow floats and narrow
  integers at their own width.
- **nx:** `nx_kernel.mli`, `carrier` and Reductions: each result rounds once
  into its destination; integers wrap.
- **Class:** exact on integers, rounded sum on floats.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › reductions › integer sums and products wrap
  as eager's`, `› float sums hold their bound`, `› scans › integer running
  sums and products wrap as eager's`, `› cases › a uint16 product wraps past
  int`, `› a uint16 running product wraps past int`.

### R3. Extremes of reductions and scans order keys

- **Reference:** `mixin/reduce.py:73` (`max`, `Ops.MAX`), `mixin/op.py:473`
  (`min`, `-max(-x)`), `:798`, `:816` (`cummax`, `cummin`).
- **Raven:** `lower_reduce.ml` (`order`, `extreme`).
- **Differs:** a float extreme is the `Max` of order keys: the signed integer
  of the bits with a negative float's magnitude flipped, complemented for a
  minimum, and the greatest key for a NaN term, which reads back as a NaN.
  `-0.` orders below `0.`. tinygrad's `MAX` of the values keeps the larger
  operand by comparison, which a NaN never is, and leaves a zero's sign to the
  target. Integers take tinygrad's `MAX`.
- **nx:** `nx_kernel.mli`, Reductions; `prog.mli`, `Maximum`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › reductions › extremes give eager's values`,
  `› scans › extreme scans give eager's values`.

### R4. Arg-reductions match bits

- **Reference:** `mixin/op.py:863` (`argmax`), `:890` (`argmin`).
- **Raven:** `lower_reduce.ml` (`same`, `position`, `changes`).
- **Differs:** the extreme is R3's; its position is that of the first term
  with its bits, or with a NaN where it is a NaN. tinygrad's equality with the
  maximum never holds for a NaN and holds for both zeros. A running arg, which
  tinygrad has no source for, is at each position the last one up to it where
  the running extreme changes its bits.
- **nx:** `nx_kernel.mli`, `Arg`: the first NaN, else the first extreme term.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › logsumexp, moments and arg › an arg gives
  eager's values and positions`, `› a running arg gives eager's values and
  positions`.

### R5. Stable sorts of packed keys

- **Reference:** `mixin/op.py:913` (`sort`: a bitonic network of the values,
  each position recovered by matching equal values and their counts), `:965`
  (`argsort`).
- **Raven:** `lower_index.ml` (`bitonic`, `ranked`, `key`, `keys`, `sort`).
- **Differs:** tinygrad's network sorts order keys, each the unsigned integer
  of its width with R3's ordering, every NaN the greatest, complemented for a
  descending sort, packed in an `int64` above its position. Packed integers
  are distinct, so the network gives the stable order, and the positions are
  their low bits. A key of more than 32 bits sorts in passes of 32, the least
  significant first. The sorted values are the operand's elements at those
  positions, read by I3's gather. tinygrad's recovery never matches a NaN, and
  its network compares floats.
- **nx:** `nx_kernel.mli`, `sort`: stable, `-0` below `+0` and every NaN above
  `+inf`, reversed when descending, bits unchanged.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_index › sorts › sorts give eager's values and
  positions`.

### I2. Pieces are selected as their bits

- **Reference:** `mixin/op.py:750` (`cat`: each piece zero-padded to the whole,
  the pieces summed).
- **Raven:** `lower_index.ml` (`assemble`, `placed`, `joined`).
- **Differs:** an assembly places each piece at its region and selects it with
  `where` over the pieces before it, floats as their bits; empty pieces are
  dropped first. The reference's sum turns a `-0.` into `+0.` and quiets a
  signalling NaN. An assembly whose selections exceed 16 times eager's work is
  refused, and the program runs uncompiled.
- **nx:** `nx_kernel.mli`, `assemble`: the last piece holding each index, or
  the fill, bits copied.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_index › folds and assemblies › an assembly gives
  eager's bits`, `Lower_index › work bounds › a scatter and an assembly past
  16 times eager's work are refused`.

### I3. Gather by an index

- **Reference:** `mixin/op.py:1041` (`gather`: a one-hot selection summed).
- **Raven:** `lower_index.ml` (`take`, `gather`).
- **Differs:** a gather is one tolk INDEX of the operand, flattened, at each
  element's offset. A position outside the axis, found by an unsigned
  comparison, is read at 0 and selected as zero. The read is a load, so each
  element keeps its bits; the reference sums the values, which turns a
  gathered `-0.` into `+0.`, and costs a reduction over the axis per element.
- **nx:** `nx_kernel.mli`, `gather`: a position outside the axis reads the
  element of zero bits.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_index › gathers and scatters › gathers give eager's
  bits`.

### I4. Scatter keeps unreached positions and sets bits

- **Reference:** `mixin/op.py:1077` (`_pre_scatter`), `:1127`
  (`scatter_reduce` sum: the masked updates summed, then `x` added), `:1168`
  (`scatter` through `_masked_merge`, one `where` per update along the axis).
- **Raven:** `lower_index.ml` (`scatter`, `summed`).
- **Differs:** each element meets each update along the axis in a mask.
  - A position no update reaches is `x`'s element, selected on the mask of
    reached positions; the reference adds `x` to a sum of zeros there, which
    turns a `-0.` into `+0.`.
  - `Add`: `x` and its updates summed in the carrier and rounded once, `-0.`
    where every term is; integers wrap on the unsigned pattern (R2).
  - `Set`: the bits of the update of highest position along the axis that
    reaches a position, picked among zeros by an unsigned `Max` in one
    reduction. The reference's chain of one `where` per update grows the graph
    with the updates.
  - `Max`, `Min`: R3's extreme of `x` and the updates that reach it.
  - A scatter whose mask exceeds 16 times eager's work is refused, and the
    program runs uncompiled.
- **nx:** `nx_kernel.mli`, `scatter`: `into` where no update lands, bits
  unchanged.
- **Class:** exact; rounded sum for float `Add`.
- **Reason:** (b).
- **Pinned by:** `Lower_index › gathers and scatters › a scatter by Set, Max
  or Min gives eager's bits`, `› a scatter by Add wraps integers and holds its
  float bound`; `Lower_index › work bounds › a scatter and an assembly past 16
  times eager's work are refused`.

### I6. Fold

- **Reference:** none.
- **Raven:** `lower_index.ml` (`fold`, `unwindow`, `unends`, `uninterior`).
- **No source:** the adjoint of a padded load: each of its movements undone in
  reverse by its adjoint, the windows' copies summed where they overlap, in
  the carrier and rounded once. Integers wrap on the unsigned pattern of 32 or
  64 bits. The same movements over a mark per term, folded by their maximum,
  find where every term is `-0.`, which the sum from `+0.` would make `+0.`.
  A boolean fold is refused.
- **nx:** `nx_kernel.mli`, `fold`: the terms summed with no `+0` before the
  first, `+0` where there is none.
- **Class:** rounded sum.
- **Reason:** (b).
- **Pinned by:** `Lower_index › folds and assemblies › a fold wraps integers
  and holds its float bound`.

### I7. A window longer than its axis gives none

- **Reference:** `mixin/op.py:1323` (`pool`).
- **Raven:** `lower.ml` (`window`).
- **Differs:** a window whose span exceeds its axis gives an axis of no
  windows; tinygrad's pool refuses it.
- **nx:** `nx_kernel.mli`, `Shape.padded`: a window longer than its axis
  gives it no window.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower › movements give eager's bits`, `› padded loads give
  eager's bits`, which draw window spans of up to 5 over axes of up to 4; no
  `cover` holds the case.

### L1. Products widen before they multiply

- **Reference:** `mixin/op.py:367` (`dot`: `(x * w).sum(-1)`, the products at
  the operands' dtype, summed in `sum_acc_dtype`'s).
- **Raven:** `lower_reduce.ml` (`entered`, `sum`, `contract`).
- **Differs:** each operand is stored into the accumulator's dtype, times its
  block's scale where it has one, before it is multiplied; the products and
  the init are summed there, integers on the unsigned pattern (R2), and
  rounded once to the output's dtype. A contraction of no products is its
  init, a `-0.` included. tinygrad rounds each product of `float16`,
  `bfloat16` and the 8-bit floats to their dtype before the `float32` sum, and
  multiplies and sums signed integers in a signed type.
- **nx:** `nx_kernel.mli`, `contract`.
- **Class:** rounded sum on floats, exact on integers.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › contractions › integer contractions wrap as
  eager's`, `› float contractions hold their bound`, `Lower_reduce › cases ›
  an int16 contraction wraps at the extremes`, `› a float16 contraction of no
  products in float32 is its init`.

### L2. QR by Householder reflections, R triangular

- **Reference:** `mixin/op.py:1799` (`qr`).
- **Raven:** `lower_linalg.ml` (`householder`, `triu`, `qr`).
- **Differs:** tinygrad's reflections, with LAPACK's choice of columns: a
  column already zero below the diagonal takes no reflection and keeps its
  diagonal element's sign, and leaves `q` and `r` as they are, so that a NaN
  or an infinity in it does not spread; tinygrad reflects every column that is
  not all zero. Each reflector is built from its column divided by its largest
  magnitude, so that no square, sum or quotient overflows or underflows. `r`'s
  elements below the diagonal are selected as `+0.`, where tinygrad leaves the
  rounding error of the zeros the reflections make. Each quotient is
  `Ops.FDIV`, rounded once, where tinygrad multiplies by the reciprocal. The
  columns run as one step of a tolk loop (`Loop.repeat`), where tinygrad
  unrolls them.
- **nx:** `nx_kernel.mli`, `qr`: `q` orthonormal, `r` upper triangular with
  `+0` below its diagonal.
- **Class:** stated bound: `Qr`'s.
- **Reason:** (b).
- **Pinned by:** `Lower_linalg › qr › q r reconstructs a`.

### L3. SVD sweeps to the roundoff and completes its vectors

- **Reference:** `mixin/op.py:1817` (`svd`: `4 num` rounds of one-sided Jacobi
  rotations over a round-robin pairing, the singular values sorted by
  `sort`, `U`'s columns divided by them).
- **Raven:** `lower_linalg.ml` (`sweeps`, `rounds`, `converged`, `svd`).
- **Differs:**
  - the rotations act on the `r` of the operand's QR (L2), scaled by the
    power of two of its largest magnitude, which the values take back;
  - they run `⌈p (⌈log2 num⌉ + 1) / 10⌉` sweeps of `num - 1` rounds (`num` for
    an odd `num`), `p` the dtype's precision; tinygrad's `4 num` rounds are
    about four sweeps, and the sweeps needed grow with `num`;
  - a pair whose inner product is within the roundoff of the product of its
    norms takes no rotation;
  - `u`'s columns are the reflections that triangularize the sorted rotated
    columns, each with the sign of its diagonal element. tinygrad divides each
    column by its singular value, which leaves a column of zeros for a zero
    singular value and an inaccurate one for a small one;
  - each quotient is `Ops.FDIV`, and the rounds run as one step of a tolk
    loop, as in L2;
  - a matrix holding NaN or an infinity, or whose rotated columns are not
    orthogonal within `16 num u` after the sweeps, is NaN in every result.
    tinygrad returns what its rounds reach.
- **nx:** `nx_kernel.mli`, `svd`: `s` descending and not negative, a zero one
  `+0`; NaN for a matrix that fails.
- **Class:** stated bound: `Svd`'s.
- **Reason:** (b).
- **Pinned by:** `Lower_linalg › svd › u diag s vh reconstructs a`, `› a
  matrix with an infinity is NaN`, `› graded and repeated spectra converge`,
  `› squares past the dtype's range`, `› a zero singular value is +0`.

### L4. Cholesky

- **Reference:** none.
- **Raven:** `lower_linalg.ml` (`cholesky`).
- **No source:** a right-looking composition, one column per step of a tolk
  loop: the column's diagonal element's square root heads it, the rest is
  divided by that root, and the trailing rows lose the column's product with
  itself. Only the lower triangle is read, and `Upper`'s factor is the
  transpose. A matrix with a pivot that is not positive, NaN included, has a
  factor of NaN in every element: one flag per matrix, from the diagonal,
  selects NaN.
- **nx:** `nx_kernel.mli`, `cholesky`.
- **Class:** stated bound: `Cholesky`'s; exact, all NaN, where the matrix is
  not positive definite.
- **Reason:** (b).
- **Pinned by:** `Lower_linalg › cholesky › L Lᵀ reconstructs a`, `› a matrix
  that is not positive definite is NaN`, `› a zero pivot of either sign is
  NaN`.

### L5. Triangular solve

- **Reference:** none.
- **Raven:** `lower_linalg.ml` (`solve_triangular`).
- **No source:** the system is made lower triangular, transposed under
  `adjoint` and reversed along both axes when the triangle read is the upper
  one, and solved by substitution, one row a step of a tolk loop; the other
  triangle, and the diagonal under `unit_diagonal`, are never read. A zero on
  the diagonal read makes the solution NaN in every element: one flag per
  matrix selects NaN.
- **nx:** `nx_kernel.mli`, `solve_triangular`.
- **Class:** stated bound: `Solve_triangular`'s; exact, all NaN, where the
  matrix is singular.
- **Reason:** (b).
- **Pinned by:** `Lower_linalg › solve_triangular › x solves op(a) x = b`,
  `› a zero diagonal is NaN`.

### L6. LU with partial pivoting

- **Reference:** none.
- **Raven:** `lower_linalg.ml` (`lu`, `first_max`).
- **No source:** one column a step of a tolk loop. The pivot is the first
  element of largest magnitude on or below the diagonal, over magnitudes in
  which a NaN on the diagonal is the greatest and one below it the least; the
  rows are exchanged by an indexed read, bit for bit. The column below a
  nonzero pivot is divided by it, and the trailing rows take the rank-one
  update; a zero pivot leaves its column unscaled.
- **nx:** `nx_kernel.mli`, `lu`.
- **Class:** stated bound: `Lu`'s, the pivots exact.
- **Reason:** (b).
- **Pinned by:** `Lower_linalg › lu › L U reconstructs P a`, `› pivots follow
  the rule`.
