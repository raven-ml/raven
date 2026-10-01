# Divergences

## Lowering

rune lowers each nx operation to tolk's UOps (`packages/rune/next/lib/`). Where
tinygrad's `Tensor` builds the same operation, the lowering builds tinygrad's
decomposition if it computes what nx documents. This section lists every place
where it builds something else, or builds what tinygrad has no source for.

An entry is admitted for one of three reasons only:

- **(a) an OCaml constraint;**
- **(b) nx's documented meaning:** the operation's contract in `nx.mli` or
  `nx_backend.mli`, which nx.cpu computes;
- **(c) the engine's contract:** nx.device's submission protocol.

Each entry gives the reference (tinygrad `79af1ca70`, or none), the raven
lines, what differs, nx's meaning, the agreement class (RFC 0012: exact,
rounded sum, ulp per target, measured bound), the reason, and the test that
pins it. An entry without its test is rejected at review. An entry goes when
its reason goes.

A test is named by its module's suite (`packages/rune/next/test/<module>/`)
and its path in it. An ulp row is measured against correctly rounded results
over its sweep, and the measured maxima per target are recorded as each
target's run lands.

### A1. Narrow floats compute at float32

- **Reference:** `mixin/elementwise.py`, each operation at the operands' dtype.
- **Raven:** `lower_arith.ml:27` (`float1`, `float2`).
- **Differs:** an operation on `float16`, `bfloat16` or an 8-bit float is a
  cast to `float32`, the operation, and one cast back. A renderer's
  half-precision intrinsics never apply.
- **nx:** `nx.mli`, Arithmetic: narrow floats compute at `float32` and round
  once.
- **Class:** as the operation's; the ulp rows hold to 1 ulp.
- **Reason:** (b).
- **Pinned by:** `exact unary floats › * › float16`, `› bfloat16`;
  `transcendental functions › * › float16`, `› bfloat16`.

### A2. Integer arithmetic wraps

- **Reference:** `mixin/elementwise.py:74,84,103,125` (`neg`, `add`, `sub`,
  `mul`) and `pow`.
- **Raven:** `lower_arith.ml:52` (`lift`, `wrapping1`, `wrapping2`), `:75`
  (`abs`), `:641` (`pow_int`).
- **Differs:** signed `Neg`, `Abs`, `Add`, `Sub`, `Mul` and integer `Pow`
  compute on the unsigned bit pattern: `uint32` for widths below 32 bits (C,
  Metal and CUDA promote them to `int`, whose products overflow), the unsigned
  type of the width otherwise. tinygrad computes in the signed type, whose
  overflow C leaves undefined. Integer `Pow` squares over the exponent's bits;
  tinygrad refuses an exponent that is not a constant.
- **nx:** modular arithmetic (`nx_backend.mli`; nx.cpu computes in the unsigned
  width); a negative integer exponent gives the integer quotient of 1.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact binary integers › {add,sub,mul,pow} › *`,
  `exact unary integers › {neg,abs} › *`,
  `exact operations on the host › integers › *` (slow).

### A3. Integer reciprocal

- **Reference:** `mixin/elementwise.py:460` (`reciprocal`).
- **Raven:** `lower_arith.ml:64` (`recip`).
- **Differs:** on integers, `1 / x` truncated: 1 at 1, -1 at -1, 0 elsewhere
  and at 0. tinygrad's reciprocal is a float operation.
- **nx:** integer division by zero is zero (`nx.mli`, `div`).
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary integers › recip › *`.

### A4. Absolute value

- **Reference:** `mixin/elementwise.py:911` (`x * sign x`).
- **Raven:** `lower_arith.ml:75` (`abs`).
- **Differs:** a float's sign bit is cleared, so `abs (-0.)` is `0.` and
  `abs nan` a NaN; tinygrad's product gives `-0.` at `-0.`. A signed integer is
  negated modularly (A2).
- **nx:** `abs`, C's `fabs` and the modular negation.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary floats › abs › *`, `exact unary integers › abs`.

### A5. Sign of NaN

- **Reference:** `mixin/elementwise.py:901` (`sign`).
- **Raven:** `lower_arith.ml:85` (`sign`).
- **Differs:** `sign nan` is NaN; tinygrad gives 1.
- **nx:** `nx_backend.mli`, `Sign`: NaN for a NaN.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary floats › sign › *`.

### A6. Rounding half away from zero

- **Reference:** `mixin/elementwise.py:890` (`round`, half to even).
- **Raven:** `lower_arith.ml:102` (`round`).
- **Differs:** a half rounds away from zero, decided on the exact rest
  `x - trunc x`.
- **nx:** `nx_backend.mli`, `Round`: C's `round`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary floats › round › *`.

### A7. Exponential in two parts

- **Reference:** `mixin/elementwise.py:511` (`exp2 (x * 1/ln 2)`).
- **Raven:** `lower_arith.ml:187` (`exp_parts`, `exp`).
- **Differs:** rounding the product `x log2 e` costs up to about `|x|` ulps.
  The product is kept in two parts (Dekker), `2^(t+e)` is `2^t (1 + e ln 2)`,
  and `2^t` is taken of `t` moved by a power of two multiplied in last, so that
  results near the dtype's extremes neither overflow early nor round twice in
  the subnormals.
- **nx:** `exp`, libm's `expf` and `exp`.
- **Class:** ulp, per target; budget 4.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › exp › *`.

### A8. Sine and cosine reduced by a quarter turn

- **Reference:** `mixin/elementwise.py:490` (`Ops.SIN`), `:500`
  (`sin (pi/2 - x)`).
- **Raven:** `lower_arith.ml:259` (`quarter_turns`, `by_quadrant`, `sin`, `cos`).
- **Differs:** `pi/2 - x` rounds first, so tinygrad's cosine is 0 at
  `f32(pi/2)`, where it is `-4.4e-8`; and the sine of a large argument is only
  as good as the target's own reduction, which on Clang loses the remainder
  near multiples of `pi/2` (336 ulps at `1445.1` in `float32`). The lowering
  reduces `|x|` to `q pi/2 + r`, `|r|` about `pi/4`: by `pi/2` in exact parts
  (12 bits below `2^12` in `float32`, 30 bits below `2^22` in `float64`),
  and by the bits of `2/pi` beyond. The quadrant picks `sin r`, or
  `cos r = 1 - 2 sin^2 (r/2)`, and the target's sine sees only `r`.
- **nx:** `sin`, `cos`, libm's.
- **Class:** ulp, per target; budget 4, up to the greatest double.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › {sin,cos} › *` and
  `› {sin,cos} of large arguments › *`, and the same on the host (slow).

### A9. Tangent of the accurate sine and cosine

- **Reference:** `mixin/elementwise.py:921` (`sin / cos`).
- **Raven:** `lower_arith.ml:291` (`tan`).
- **Differs:** through the sine and cosine (A8) only.
- **nx:** `tan`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › tan › *`.

### A10. Arcsine and arccosine

- **Reference:** `mixin/elementwise.py:931,944` (Abramowitz-Stegun 4.4.46,
  `acos = pi/2 - asin`).
- **Raven:** `lower_arith.ml:313` (`asin_small`, `half`), `:331` (`asin`),
  `:340` (`acos`).
- **Differs:** tinygrad's polynomial is float32-grade, has no relative accuracy
  near 0, and `pi/2 - asin` cancels near 1. The lowering refines the
  polynomial with Newton's steps on `sin y = t` (one in `float32`, two in
  `float64`) on `|t| <= 0.71`, and reduces the rest by the half-angle forms:
  `asin a = pi/2 - 2 asin (sqrt ((1 - a)/2))`, `acos` as twice that or `pi`
  less it.
- **nx:** `asin`, `acos`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › {asin,acos} › *`.

### A11. Arctangent

- **Reference:** `mixin/elementwise.py:954` (`asin (x / sqrt (1 + x^2))`).
- **Raven:** `lower_arith.ml:351` (`atan_positive`, `atan`).
- **Differs:** `x^2` overflows past `1.8e19` in `float32`, where tinygrad's
  arctangent is 0. The lowering takes the arcsine form of `min (|x|, 1/|x|)`
  only, and `pi/2` less it past 1.
- **nx:** `atan`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › atan › *`.

### A12. Two-argument arctangent

- **Reference:** none.
- **Raven:** `lower_arith.ml:359` (`atan2`).
- **Differs:** a raven composition: the arctangent of `|y| / |x|`, `pi` less it
  for a negative `x` (its sign bit), the sign of `y`, and C's values for both
  zeros and both infinities.
- **nx:** `atan2`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › atan2 › *`.

### A13. Hyperbolic functions

- **Reference:** `mixin/elementwise.py:1034,1046` (`(e^x -+ e^-x) / 2`), `:757`
  (`2 sigmoid (2x) - 1`).
- **Raven:** `lower_arith.ml:386` (`sinh`), `:395` (`cosh`), `:399` (`tanh`).
- **Differs:** `sinh` and `tanh` cancel near 0 (`tanh` is 0 below about
  `6e-8`), and `sinh` and `cosh` overflow for `x` in `(88.72, 89.42]` in
  `float32`. The lowering computes `e^|x| / 2 +- e^-|x| / 2` with the halving in
  the exponential's last scaling (A7), `sinh` below 1 as its Taylor series, and
  `tanh` as `sinh / cosh`, `+-1` past 22.
- **nx:** `sinh`, `cosh`, `tanh`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › {sinh,cosh,tanh} › *`.

### A14. Error function

- **Reference:** `mixin/elementwise.py:1058` (Abramowitz-Stegun 7.1.26).
- **Raven:** `lower_arith.ml:520` (`erf`).
- **Differs:** tinygrad's approximation has an absolute error of `1.5e-7` and
  no relative accuracy near 0. The lowering builds fdlibm's rational
  approximations on four intervals, the coefficients of its double precision
  `erf`, rounded to the dtype.
- **nx:** `erf`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › erf › *`.

### A15. True division

- **Reference:** `mixin/elementwise.py:229` (`a * (1/b)`).
- **Raven:** `lower_arith.ml:134` (`( /: )`).
- **Differs:** one rounding: `Ops.FDIV`, which tolk renders as `a / b`
  (tolk's ledger). tinygrad multiplies by the reciprocal, two roundings.
- **nx:** `nx_backend.mli`, `Fdiv`: the IEEE 754 quotient.
- **Class:** exact where the target's division is correctly rounded.
- **Reason:** (b).
- **Pinned by:** `exact binary floats › div › *`,
  `exact operations on the host › floats › div` (slow).

### A16. Integer quotient and remainder by 0 and -1

- **Reference:** `mixin/elementwise.py:229,216` (`Ops.CDIV`, `Ops.CMOD`).
- **Raven:** `lower_arith.ml:741` (`idiv`), `:748` (`rem`).
- **Differs:** a quotient or remainder by 0 is 0, a remainder by -1 is 0, and
  the quotient of the least integer by -1 wraps. The divisor is made 1 in those
  lanes before dividing, since a kernel computes both sides of a selection and
  C traps or leaves them undefined.
- **nx:** `nx.mli`, `div` and `mod_`: by zero is zero.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact binary integers › {div,mod} › *`,
  `exact operations on the host › integers › {div,mod}` (slow).

### A17. Exact float remainder

- **Reference:** `mixin/elementwise.py:216` (`x - trunc (x * (1/y)) * y`).
- **Raven:** `lower_arith.ml:676` (`fmod`).
- **Differs:** tinygrad's composition rounds three times and fails for
  `|x/y| >= 2^24`. The lowering computes C's exact `fmod` on the significands
  as integers: `(mx 2^d) mod my`, `d` the exponent difference, reduced 40 bits
  of `d` at a time in `float32` (7 steps) and 11 in `float64` (187 steps), each
  shifted remainder in 64 bits, then scaled by `y`'s exponent in two exact
  steps. The steps are unrolled, not a loop (RFC 0012, Unresolved 3).
- **nx:** `nx.mli`, `mod_`: C's `fmod` on floats.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact binary floats › mod › *`,
  `exact operations on the host › floats › mod` (slow).

### A18. Power in two parts

- **Reference:** `mixin/elementwise.py:548` (`Ops.POW`, `xpow` =
  `exp2 (y log2 |x|)`).
- **Raven:** `lower_arith.ml:569` (`log2_parts`), `:600` (`pow_float`).
- **Differs:** `y log2 |x|` carries `y` times the logarithm's error. The
  lowering takes `log2 |x|` to twice the precision (`|x| = m 2^e`,
  `ln m = 2 atanh s` with its leading term in two parts), keeps the product in
  two parts, and exponentiates as A7. The special values are C's `pow`'s.
- **nx:** `pow`, libm's.
- **Class:** ulp, per target; budget 16.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › pow › *`,
  `transcendental functions › pow of negative bases and integral exponents › *`.

### A19. Maximum and minimum

- **Reference:** `mixin/elementwise.py:378,393` (`Ops.MAX`; `minimum` as
  `-max(-x, -y)`).
- **Raven:** `lower_arith.ml:759` (`extreme`).
- **Differs:** IEEE 754-2019's maximum and minimum: NaN when an operand is NaN,
  and `-0.` below `0.`, read on the sign bit. tinygrad's `MAX` keeps the larger
  operand by comparison, which a NaN never is.
- **nx:** `nx.mli`, `maximum`, `minimum` (RFC 0012, Extremes).
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `extremes of NaN`, `extremes`.

### A20. Less than or equal

- **Reference:** `mixin/elementwise.py:330` (`not (x > y)`).
- **Raven:** `lower_arith.ml:827` (`compare`, `at_most`).
- **Differs:** `x < y or x = y`, false on NaN; tinygrad's is true on NaN.
- **nx:** `nx.mli`, `less_equal`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `comparisons › less_equal`,
  `exact operations on the host › floats › less_equal` (slow).

### A21. Saturating conversion to integers

- **Reference:** `mixin/dtype.py:19` (`cast`).
- **Raven:** `lower_arith.ml:840` (`saturate`).
- **Differs:** a float converted to an integer is held at the integer's range
  and NaN is 0; the conversion itself sees only values in range. C leaves both
  undefined.
- **nx:** `nx.mli`, `cast`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `conversions › a float saturates at an integer's range, NaN
  at 0 › *`, `exact operations on the host › conversions › *` (slow).

### A22. Threefry on packed words

- **Reference:** `mixin/rand.py:14` (the key packed as `uint64`),
  `mixin/elementwise.py:457` (`Ops.THREEFRY`).
- **Raven:** `lower_arith.ml:881` (`threefry`).
- **Differs:** nx's words are `int32` pairs along the last axis, the low word
  first; the lowering packs each pair into a `uint64`, hashes, and unpacks.
- **nx:** `nx_backend.mli`, `threefry`: Threefry-2x32-20, bit-identical under
  every lowering.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `random bits › a traced key draws eager's words, compiled for
  the host`.

### A23. A double narrows through float32 rounded to odd

- **Reference:** `mixin/dtype.py:19` (`cast`).
- **Raven:** `lower_arith.ml:860` (`to_float32_odd`, `cast`).
- **Differs:** a `float64` converted to `float16`, `bfloat16` or an 8-bit float
  is first narrowed to `float32` rounded to odd (truncated, its last bit set
  when a discarded bit was), so that the conversion rounds once. tinygrad's
  host conversion goes through `float32` rounded to nearest and rounds twice
  near a tie: `bfloat16 (1 + 2^-8 + 2^-30)` comes out 1.
- **nx:** `nx.mli`, `cast`: one rounding (nx.cpu's `double_to_float_odd`).
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `conversions › a double rounds to bfloat16 once`,
  `exact operations on the host › conversions › a double near a bfloat16 tie
  rounds once` (slow).

### A24. Logarithm below zero

- **Reference:** `mixin/elementwise.py:830` (`log2 (x) * ln 2`), whose `log2`
  is `codegen/decomp/transcendental.py:219` (`xlog2`) on a target without it.
- **Raven:** `lower_arith.ml:223` (`log`).
- **Differs:** `xlog2` takes `x` for `-0.` where its reciprocal is `-inf`,
  which it is for a negative subnormal whose reciprocal overflows, so the
  logarithm is `-inf`. A target that flushes subnormals takes every negative
  subnormal for `-0.`. The lowering selects NaN where the bits of `x` have the
  sign bit and a magnitude, before any target reads the value.
- **nx:** `log`, libm's `logf` and `log`: NaN below zero, `-inf` at either
  zero.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `logarithms › log and log2 are NaN below zero and -inf at
  -0.`, `logarithms › * › * has libm's classes`, and the same on the host
  (slow).

### R1. A float sum adds +0.

- **Reference:** `mixin/reduce.py:20` (`sum`), `mixin/op.py:758`
  (`_split_cumalu`).
- **Raven:** `lower_reduce.ml:63` (`accumulated`).
- **Differs:** the lowering adds `+0.` to each float sum it computes, once per
  output, for `Reduce` and `Scan` with `Sum`. A kernel starts a loop's
  accumulator from `+0.`, but sums the terms alone when no loop is left (an
  axis of one element, or one it unrolls whole), so tinygrad's sum of `[-0.]`
  is `-0.`.
- **nx:** `nx.mli`, `sum` and `cumsum`: a float sum is `0.` plus its terms, so
  a sum that is exactly zero is `0.`.
- **Class:** rounded sum.
- **Reason:** (b).
- **Pinned by:** `sums and products › a zero sum is +0. › *`,
  `scans › a running sum starts from +0.`, and through tolk's rewrites
  `compiled for the host › a zero sum is +0. › *` (slow).

### R2. Sums and products accumulate wide and unsigned, and name their dtype

- **Reference:** `mixin/reduce.py:20` (`sum`: the `sum_acc_dtype`
  accumulator, converted back for the narrow floats only), `:47` (`prod`: at
  the operand's dtype).
- **Raven:** `lower_reduce.ml:63` (`accumulated`).
- **Differs:** a sum and a product both accumulate in `Dtype.sum_acc`'s type,
  unsigned for the signed integers, and convert once to the operand's dtype.
  tinygrad accumulates signed integers in a signed type, whose overflow C,
  Metal and CUDA leave undefined; returns an integer sum in its accumulator's
  dtype; and multiplies narrow floats and narrow integers at their own width.
- **nx:** `nx_backend.mli`, `reduce` and `scan`: the result has the operand's
  dtype; integers wrap; `nx.mli`, Arithmetic: narrow floats compute at
  `float32` and round once.
- **Class:** exact on integers, rounded sum on floats.
- **Reason:** (b).
- **Pinned by:** `sums and products › integer sums wrap › *`,
  `› integer products wrap › *`, `› narrow floats round once`,
  `scans › integer scans are exact › *`,
  `compiled for the host › integer sums and products wrap` (slow).

### R3. Extremes of reductions and scans order keys

- **Reference:** `mixin/reduce.py:73` (`max`, `Ops.MAX`), `mixin/op.py:473`
  (`min`, `-max(-x)`), `:798`, `:816` (`cummax`, `cummin`).
- **Raven:** `lower_reduce.ml:37` (`keys`, `values`), `:73` (`extreme`).
- **Differs:** a float maximum is `Ops.MAX` over integer keys, the bits with a
  negative float's magnitude flipped and every NaN at the greatest key, mapped
  back to floats; a minimum takes the maximum of the keys' complements, every
  NaN at the least key. tinygrad's `MAX` keeps the larger operand by
  comparison, which a NaN never is, and leaves the order of `-0.` and `+0.` to
  the association; its minimum negates. Integers agree (`graph parity ›
  max_int`, `min_int`, `cummax_int`, `cummin_int`).
- **nx:** `nx.mli`, `max`, `min`, `cummax`, `cummin`: IEEE 754-2019 maximum and
  minimum, NaN propagating and `-0.` below `0.`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `extremes › *`, `scans › float running extremes are exact ›
  *`, `scans › a running extreme is NaN from the first NaN on`,
  `compiled for the host › extremes of NaN and both zeros` (slow).

### R4. Arg-reductions order keys

- **Reference:** `mixin/op.py:863` (`argmax`), `:890` (`argmin`).
- **Raven:** `lower_reduce.ml:95` (`argmax`), `:107` (`arg_reduce`).
- **Differs:** tinygrad's decomposition runs over R3's keys: the first element
  equal to the maximum of the keys, the first NaN if there is one, and `-0.`
  below `0.`. Over floats, tinygrad's equality with the maximum never holds for
  a NaN. Integers agree (`graph parity › argmax_int`, `argmin_int`).
- **nx:** `nx.mli`, `argmax`, `argmin`: the first index holding the element
  `max` and `min` return, the first NaN if there is one.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `arg-reductions › *`.

### R5. Stable sorts of packed keys

- **Reference:** `mixin/op.py:913` (`sort`: a bitonic network of the values,
  each position recovered by matching equal values and their counts), `:965`
  (`argsort`).
- **Raven:** `lower_reduce.ml:129` (`bitonic`), `:191` (`positions`), `:202`
  (`take`), `:225` (`argsort`), `:247` (`sort`).
- **Differs:** tinygrad's network sorts R3's keys, NaN at the greatest key
  ascending and the least descending, each read as the unsigned integer of its
  width and packed in an `int64` above its position, complemented for a
  descending sort. Packed integers are distinct, so the network gives the
  stable order, and the positions are their low bits. A 64-bit key sorts in two
  such passes, its low half first. The sorted values are the operand's
  elements at those positions, a one-hot sum over their bits. tinygrad's
  recovery never matches a NaN, and its network compares floats.
- **nx:** `nx_backend.mli`, `sort`, `argsort`: stable, NaN last in either
  direction, `-0.` before `0.` ascending, and `sort` is the operand taken
  along `argsort`, bit for bit.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `sorts › *`; the network's kernels by `graph parity ›
  argsort_int`, `argsort_int_descending`.

### I1. A pad of `-0.`

- **Reference:** `mixin/op.py:289` (`_pad_constant`: a fill equal to 0 is the
  movement's zeros).
- **Raven:** `lower_index.ml:49` (`pad`).
- **Differs:** a fill of `-0.` is selected on the padding with `where`, as the
  reference fills any other value; the reference takes `-0.` for 0 and pads
  `+0.`.
- **nx:** `nx_backend.mli`, `pad`: the padding holds the value given.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `assembly › a pad of -0. keeps its sign`.

### I2. Pieces of different lengths are selected

- **Reference:** `mixin/op.py:750` (`cat`: each piece zero-padded to the whole,
  the pieces summed).
- **Raven:** `lower_index.ml:63` (`cat`).
- **Differs:** each piece is selected with `where` on the stretch it fills; the
  reference's sum turns a `-0.` into `+0.` and quiets a signalling NaN. Empty
  pieces are dropped first, and pieces of one length are stacked as the
  reference stacks them.
- **nx:** `nx_backend.mli`, `cat`: the arrays' elements, one after the other.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `assembly › pieces of different lengths keep -0.`,
  `assembly › cat › *`.

### I3. Gather over bit patterns

- **Reference:** `mixin/op.py:1041` (`gather`: a one-hot selection summed).
- **Raven:** `lower_index.ml:91` (`gather`), `:32` (`bits`), `:41` (`pick`).
- **Differs:** the one-hot selection is summed over the elements' bit patterns
  as unsigned integers of their width (booleans as `uint8`) and read back; the
  reference sums the values, which turns a gathered `-0.` into `+0.`. An index
  out of range selects nothing and reads the bits 0, `+0.`.
- **nx:** `nx_backend.mli`, `gather`: the element at the index; an index
  outside the axis reads zero.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `indexed access › a gathered -0. keeps its sign`,
  `› an index out of range reads +0.`, `› gather › *`.

### I4. Scatter keeps unreached positions and sets bits

- **Reference:** `mixin/op.py:1077` (`_pre_scatter`), `:1127`
  (`scatter_reduce` sum: the masked updates summed, then `x` added), `:1168`
  (`scatter` through `_masked_merge`, one `where` per update along the axis).
- **Raven:** `lower_index.ml:99` (`scatter`);
  `lower_reduce.ml:87` (`reduce`).
- **Differs:**
  - `Add`: a position no update reaches is `x`'s element, selected on the
    mask of reached positions; the reference adds `x` to a sum of zeros there,
    which turns a `-0.` into `+0.`. A reached position is `x` and its updates
    summed from `+0.`, at `float32` or wider and rounded once; integers wrap
    on the unsigned bit pattern (A2).
  - `Set`: the bits of the last update that reaches a position, the one of
    highest index along the axis (every reaching update with `unique`), are
    selected by a one-hot sum over bit patterns, as I3, in one reduction. The
    meaning agrees with the reference's `_masked_merge`, which also keeps the
    last update; its construction, a chain of one `where` per update along the
    axis, grows the graph with the number of updates.
- **nx:** `nx_backend.mli`, `scatter`: `x` where no update lands, the last
  duplicate wins under `Set`, every update adds under `Add`.
- **Class:** exact; rounded sum for float `Add`.
- **Reason:** (b); for `Set`'s construction, the consumer is `Nx.set`, whose
  flat scatter carries one update per selected position, thousands of them,
  where a graph that grows with the updates is unusable.
- **Pinned by:** `indexed access › a position no update reaches keeps -0.`,
  `› the last of duplicate positions is set`, `› scatter set › *`,
  `› scatter set unique › *`, `› scatter add › *`.

### I5. A window at a corner read at run time

- **Reference:** `tensor.py:502` (`__setitem__`), `mixin/op.py:146` (advanced
  setitem through `_masked_merge`). The reference's windows start at constants;
  a corner computed by the program is only reachable as advanced indexing, a
  mask of every window position against every position of `x`, merged by one
  `where` per window position.
- **Raven:** `lower_index.ml:139` (`update`).
- **Differs:** along each axis `v` does not fill, `v` is moved to its start by a
  one-hot selection over bit patterns (I3), and the moved `v` is selected on
  the window's mask; along an axis `v` fills, the start is 0.
- **nx:** `nx_backend.mli`, `update`: `starts` is read when the kernel runs,
  already clamped so that the window fits.
- **Class:** exact.
- **Reason:** (b); the consumer is kaun's decode, whose `Cache_index` (RFC
  0002) writes a window at a position known only at run time, every step.
- **Pinned by:** `indexed access › update › *`,
  `› a window at constant starts`.

### I6. Fold

- **Reference:** none.
- **Raven:** `lower_index.ml:223` (`fold`), `:206` (`cut`);
  `lower_reduce.ml:87` (`reduce`).
- **No source:** the transpose of the unfold: each movement of `Ops.pool` undone
  in reverse order, a shrink by a pad of zeros, and the copies of the input
  summed, which sums the windows where they overlap, from `+0.`, at `float32`
  or wider and rounded once. Integers wrap; booleans are whether any window
  holds. A fold without a window along some axis, or whose windows along it
  read only padding, is zeros, a constant: no element of its input lands in
  the output, and the kernel that sums nothing is one whose every lane reads
  an `Invalid` index, which tolk cannot lower.
- **nx:** `nx_backend.mli`, `fold`: the windows put back, summed where they
  overlap.
- **Class:** rounded sum.
- **Reason:** (b).
- **Pinned by:** `windows › fold › *`,
  `windows › overlapping windows of -0. fold to +0.`,
  `› a single window of -0. folds to +0.`, `› a fold whose windows along an
  axis read only padding is zeros`, `› a fold with no window is zeros`;
  `Compiled › edges › a fold whose windows along an axis read only padding is
  zeros`, `› a fold with no window is zeros`.

### L1. Products widen before they multiply

- **Reference:** `mixin/op.py:367` (`dot`: `(x * w).sum(-1)`, the products at
  the operands' dtype, summed in `sum_acc_dtype`'s).
- **Raven:** `lower_linalg.ml:59` (`matmul`), `:55` (`dot`);
  `lower_reduce.ml:59` (`accumulator`).
- **Differs:** the operands are converted to `Lower_reduce.accumulator`'s type
  before they are multiplied, and the products summed by `Lower_reduce.reduce`
  (R1, R2) and converted once to the operands' dtype. tinygrad rounds each
  product of `float16`, `bfloat16` and the 8-bit floats to their dtype before
  the `float32` sum, and multiplies and sums signed integers in a signed type.
  The widened product is exact, and CUDA and HIP apply their narrow-in,
  `float32`-out tensor cores to it through tolk's D29; Metal takes its
  `float32` core, which comes first. A `float32` product agrees (`graph parity
  › matmul`, `› matmul_batched`).
- **nx:** `nx.mli`, Arithmetic: narrow floats compute at `float32` and round
  once; nx.cpu's GEMM converts narrow floats to `float32` as it packs them,
  accumulates integers in 64 bits, and wraps on the store.
- **Class:** rounded sum on floats, exact on integers.
- **Reason:** (b).
- **Pinned by:** `products › *`, `products › narrow products are exact before
  they are summed`, `products › a zero product is +0. › *`, `tensor cores ›
  metal › *`, `tensor cores › cuda › *`, `compiled for the host › *` (slow).

### L2. QR by Householder reflections, R triangular

- **Reference:** `mixin/op.py:1799` (`qr`).
- **Raven:** `lower_linalg.ml:75` (`householder`), `:102` (`triu`), `:106`
  (`qr`).
- **Differs:** tinygrad's reflections, with `r`'s elements below the diagonal
  selected as `+0.`: tinygrad leaves there the rounding error of the zeros the
  reflections make. Each quotient is `Ops.FDIV`, rounded once, where tinygrad
  multiplies by the reciprocal and rounds twice: IEEE division is the rule of
  every rune composition, as of tolk's arithmetic (D9, D24), and it takes the
  measured maxima from 6.9 and 13.5 to 2.4 and 3.7. `float16` computes at `float32`, and the reduced factors
  are the leading columns of `q` and rows of `r`. The signs agree in meaning
  only: a column with no element below the diagonal is still reflected, where
  nx.cpu, as LAPACK, takes no reflection, so a diagonal element of `r` may
  have eager's opposite sign. A norm is the square root of a sum of squares,
  unscaled, so elements whose squares overflow or leave the normal range lose
  accuracy. Compiled code never raises `No_convergence`.
- **nx:** `nx_backend.mli`, `qr`: `q` orthonormal, `r` upper triangular; nx
  pins no factor's signs.
- **Class:** measured bound: within `16 max(m, n) u` of the largest element of
  eager's factors, up to the signs of the diagonal of `r`, for well-conditioned
  matrices of up to 5 x 5; measured maxima over 300 such matrices, in units of
  `max(m, n) u`: 2.4 (`float32`), 3.7 (`float64`), 0.5 (`float16`, `u` its
  own).
- **Reason:** (b).
- **Pinned by:** `qr › matrices › *`, `qr › a zero column takes no
  reflection`, `› one element`, `› no column: q is the identity`, `› batch
  axes`; the construction, its single-rounding quotients included, by `graph
  parity › qr_q`, `› qr_r`, whose generator builds tinygrad's reflections with
  `Ops.FDIV`.

### L3. SVD sweeps to the roundoff and completes its vectors

- **Reference:** `mixin/op.py:1817` (`svd`: `4 num` rounds of one-sided Jacobi
  rotations over a round-robin pairing, the singular values sorted by
  `sort`, `U`'s columns divided by them).
- **Raven:** `lower_linalg.ml:127` (`pairs`), `:134` (`next_pairs`), `:146`
  (`rounds`), `:150` (`rotate`), `:198` (`svd`).
- **Differs:**
  - the rotations run `ceil (log2 num) + 3` sweeps of `num - 1` rounds (`num`
    for an odd `num`). tinygrad's `4 num` rounds are about four sweeps, which
    leave `float64` values of random 9 x 9 matrices `3.4e5 num u` from
    eager's and of 16 x 16 ones `5.6e10 num u`; the sweeps needed grow with
    `num` (8 at 48 in `float64`), and these reach `1.8 num u` to 48;
  - `u`'s columns are the reflections that triangularize the sorted rotated
    columns, each with the sign of its diagonal element. tinygrad divides each
    column by its singular value, which leaves a column of zeros for a zero
    singular value and an inaccurate one for a small one. The columns are
    sorted by `Lower_reduce.argsort`, which is stable;
  - each quotient of the rotations is `Ops.FDIV`, rounded once, as in L2;
  - the values are `float64`, refused (`Jit_error`) on a target without it,
    such as Metal. Compiled code never raises `No_convergence`: it runs its
    fixed sweeps, and NaN in `a` gives NaN values, as eager does.
- **nx:** `nx.mli`, `svd`: `a = U diag(S) Vh`, `S` descending, non-negative,
  a zero one `+0`; `nx_backend.mli`: `u` and `vt` orthonormal.
- **Class:** measured bound: within `16 max(m, n) u` of the largest singular
  value for the values, and of one for the orthonormality of the vectors and
  the reconstruction, for well-conditioned matrices of up to 4 x 4; measured
  maxima, in units of `max(m, n) u`: 3.1 (`float32`), 4.0 (`float64`), 0.5
  (`float16`, `u` its own).
- **Reason:** (b).
- **Pinned by:** `svd › matrices › *`, `svd › float64 values of a 12 x 12
  matrix reach its roundoff` (which fails under `4 num` rounds), `› a
  rank-deficient matrix has orthonormal vectors and +0. values`, `› NaN gives
  NaN values`, `› one element`, `› no element: the full factors are
  identities`, `› a target without float64 refuses it`, `› batch axes`.

### L4. Cholesky

- **Reference:** none.
- **Raven:** `lower_linalg.ml:324` (`cholesky`).
- **No source:** a right-looking composition, one column per step: the
  column's diagonal element's square root heads it, the rest is divided by
  that root, and the working matrix loses the column's product with itself.
  Only the lower triangle is read, and `upper` is the transpose. A pivot that
  is not positive is NaN, so the column it heads and every later one are NaN
  on and below the diagonal, where eager raises `Linalg_error`
  `Not_positive_definite`; a last pivot of zero, which would give finite
  values, is NaN too.
- **nx:** `nx.mli`, `cholesky`.
- **Class:** measured bound: within `4 n u` of the largest element of eager's
  factor for well-conditioned positive-definite matrices of up to 5 x 5;
  measured maxima, in units of `n u`: 0.4 (`float32`), 0.6 (`float64`), 0
  (`float16`). Pinned values where eager raises.
- **Reason:** (b).
- **Pinned by:** `cholesky › positive-definite matrices › *`, `cholesky › a
  pivot that is not positive is NaN, and every column after it`, `› a zero
  pivot is NaN`, `› one element`, `› no element`.

### L5. Triangular solve

- **Reference:** none.
- **Raven:** `lower_linalg.ml:359` (`solve_triangular`).
- **No source:** the system is made lower triangular, transposed under
  `transpose` and reversed along both axes when the triangle read is the upper
  one, and solved by substitution, one row a step, from the strictly lower
  triangle and the diagonal; the other triangle, and the diagonal under
  `unit_diag`, are never read. A zero on the diagonal, where eager raises
  `Linalg_error` `Singular`, makes that row and every later one in the order
  of substitution non-finite: the quotient by zero, then its products.
- **nx:** `nx.mli`, `solve_triangular`.
- **Class:** measured bound: within `4 n u` of the largest element of eager's
  solution for diagonally dominant matrices of up to 5 x 5; measured maxima,
  in units of `n u`: 0.9 (`float32`, `float64`), 0 (`float16`). Pinned values
  where eager raises.
- **Reason:** (b).
- **Pinned by:** `triangular solves › dominant matrices › *`, `triangular
  solves › a zero pivot makes its row and those after it non-finite`, `› no
  element`.

### L6. LU with partial pivoting

- **Reference:** none.
- **Raven:** `lower_linalg.ml:261` (`lu`).
- **No source:** one column a step. The pivot is the first element of largest
  magnitude on or below the diagonal, found by `Lower_reduce.arg_reduce` over
  magnitudes in which a NaN on the diagonal is the greatest and one below it
  the least; the rows are exchanged with `Lower_index.gather`, bit for bit.
  The column below a nonzero pivot is divided by it, and the trailing rows take
  the rank-one update. It raises in neither eager nor compiled code: a zero
  pivot leaves its column unscaled, as eager does.
- **nx:** `nx_backend.mli`, `lu`.
- **Class:** measured bound, with eager's pivots and row order: within
  `4 max(m, n) u` of the largest element of eager's factors for
  well-conditioned matrices of up to 5 x 5 with shuffled rows; measured maxima,
  in units of `max(m, n) u`: 0.7 (`float32`), 0.2 (`float64`), 0
  (`float16`). The kernel rounds the update's product and difference apart
  (tolk's D25), where nx.cpu's C, built with Clang's default contraction, may
  fuse them.
- **Reason:** (b).
- **Pinned by:** `lu › matrices › *`, `lu › the first of equal magnitudes is
  the pivot`, `› a zero column leaves its column unscaled`, `› a NaN below the
  diagonal is never the pivot`, `› a NaN on the diagonal is the pivot`, `› one
  element`, `› no element`.


## Targets

A target computes what its hardware computes where that differs from nx's
meaning and no lowering recovers it. Each entry gives the targets, what
differs, nx's meaning, and the test that pins it.

### T1. Subnormals flush to zeros of their sign

- **Targets:** Metal, for `float32` and for `bfloat16`, which computes at
  `float32`.
- **Differs:** arithmetic reads a subnormal operand as a zero of its sign and
  writes a subnormal result as a zero of its sign: `recip (-0x1.fffffep127)`
  is `-0.`, where nx.cpu gives `-0x1p-128`, and `atan2 (-4) 0x1.fffffep127`
  is `-0.`. A kernel that moves or orders values without computing on them,
  a copy, a gather or a sort, keeps every bit.
- **nx:** `nx_backend.mli`: IEEE 754 binary arithmetic, with gradual
  underflow.
- **Pinned by:** `Compiled › metal › elementwise › exact unary`, `› exact
  binary`, `› transcendental unary`, `› transcendental binary` and `› cast`
  (slow), which draw operands without subnormals on a flushing target and
  compare with eager's result flushed the same way; `› reductions › sort`
  keeps subnormals.
