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
- **Raven:** `lower_arith.ml:53` (`lift`, `wrapping1`, `wrapping2`), `:78`
  (`abs`), `:644` (`pow_int`).
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
- **Raven:** `lower_arith.ml:65` (`recip`).
- **Differs:** on integers, `1 / x` truncated: 1 at 1, -1 at -1, 0 elsewhere
  and at 0. tinygrad's reciprocal is a float operation.
- **nx:** integer division by zero is zero (`nx.mli`, `div`).
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary integers › recip › *`.

### A4. Absolute value

- **Reference:** `mixin/elementwise.py:911` (`x * sign x`).
- **Raven:** `lower_arith.ml:78` (`abs`).
- **Differs:** a float's sign bit is cleared, so `abs (-0.)` is `0.` and
  `abs nan` a NaN; tinygrad's product gives `-0.` at `-0.`. A signed integer is
  negated modularly (A2).
- **nx:** `abs`, C's `fabs` and the modular negation.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary floats › abs › *`, `exact unary integers › abs`.

### A5. Sign of NaN

- **Reference:** `mixin/elementwise.py:901` (`sign`).
- **Raven:** `lower_arith.ml:88` (`sign`).
- **Differs:** `sign nan` is NaN; tinygrad gives 1.
- **nx:** `nx_backend.mli`, `Sign`: NaN for a NaN.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary floats › sign › *`.

### A6. Rounding half away from zero

- **Reference:** `mixin/elementwise.py:890` (`round`, half to even).
- **Raven:** `lower_arith.ml:105` (`round`).
- **Differs:** a half rounds away from zero, decided on the exact rest
  `x - trunc x`.
- **nx:** `nx_backend.mli`, `Round`: C's `round`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `exact unary floats › round › *`.

### A7. Exponential in two parts

- **Reference:** `mixin/elementwise.py:511` (`exp2 (x * 1/ln 2)`).
- **Raven:** `lower_arith.ml:190` (`exp_parts`, `exp`).
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
- **Raven:** `lower_arith.ml:251` (`quarter_turns`, `by_quadrant`, `sin`, `cos`).
- **Differs:** `pi/2 - x` rounds first, so tinygrad's cosine is 0 at
  `f32(pi/2)`, where it is `-4.4e-8`; and the sine of a large argument is only
  as good as the target's own reduction, which on Clang loses the remainder
  near multiples of `pi/2` (336 ulps at `1445.1` in `float32`). The lowering
  reduces `|x|` to `q pi/2 + r`, `|r|` about `pi/4`: by `pi/2` in exact parts
  (12 bits below `2^12` in `float32`, 30 bits below `2^22` in `float64`),
  and by the bits of `2/pi` beyond. The quadrant picks `sin r`, or
  `cos r = 1 - 2 sin^2 (r/2)`, and the target's sine sees only `r`.
- **nx:** `sin`, `cos`, libm's.
- **Class:** ulp, per target; budget 4. Beyond the exact parts' limit the
  class waits for the bits of `2/pi` to be rounded on the right bit (tolk).
- **Reason:** (b).
- **Pinned by:** `transcendental functions › {sin,cos} › *`,
  `transcendental functions on the host › {sin,cos} › *` (slow).

### A9. Tangent of the accurate sine and cosine

- **Reference:** `mixin/elementwise.py:921` (`sin / cos`).
- **Raven:** `lower_arith.ml:283` (`tan`).
- **Differs:** through the sine and cosine (A8) only.
- **nx:** `tan`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › tan › *`.

### A10. Arcsine and arccosine

- **Reference:** `mixin/elementwise.py:931,944` (Abramowitz-Stegun 4.4.46,
  `acos = pi/2 - asin`).
- **Raven:** `lower_arith.ml:305` (`asin_small`, `half`), `:323` (`asin`),
  `:324` (`acos`).
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
- **Raven:** `lower_arith.ml:343` (`atan_positive`, `atan`).
- **Differs:** `x^2` overflows past `1.8e19` in `float32`, where tinygrad's
  arctangent is 0. The lowering takes the arcsine form of `min (|x|, 1/|x|)`
  only, and `pi/2` less it past 1.
- **nx:** `atan`, libm's.
- **Class:** ulp, per target; budget 8.
- **Reason:** (b).
- **Pinned by:** `transcendental functions › atan › *`.

### A12. Two-argument arctangent

- **Reference:** none.
- **Raven:** `lower_arith.ml:351` (`atan2`).
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
- **Raven:** `lower_arith.ml:378` (`sinh`), `:387` (`cosh`), `:391` (`tanh`).
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
- **Raven:** `lower_arith.ml:512` (`erf`).
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
- **Raven:** `lower_arith.ml:137` (`( /: )`).
- **Differs:** one rounding: `Ops.FDIV`, which tolk renders as `a / b`
  (tolk's ledger). tinygrad multiplies by the reciprocal, two roundings.
- **nx:** `nx_backend.mli`, `Fdiv`: the IEEE 754 quotient.
- **Class:** exact where the target's division is correctly rounded.
- **Reason:** (b).
- **Pinned by:** `exact binary floats › div › *`,
  `exact operations on the host › floats › div` (slow).

### A16. Integer quotient and remainder by 0 and -1

- **Reference:** `mixin/elementwise.py:229,216` (`Ops.CDIV`, `Ops.CMOD`).
- **Raven:** `lower_arith.ml:759` (`idiv`), `:766` (`rem`).
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
- **Raven:** `lower_arith.ml:679` (`fmod`).
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
- **Raven:** `lower_arith.ml:561` (`log2_parts`), `:603` (`pow_float`).
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
- **Raven:** `lower_arith.ml:777` (`extreme`).
- **Differs:** IEEE 754-2019's maximum and minimum: NaN when an operand is NaN,
  and `-0.` below `0.`, read on the sign bit. tinygrad's `MAX` keeps the larger
  operand by comparison, which a NaN never is.
- **nx:** `nx.mli`, `maximum`, `minimum` (RFC 0012, Extremes).
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `extremes of NaN`, `extremes`.

### A20. Less than or equal

- **Reference:** `mixin/elementwise.py:330` (`not (x > y)`).
- **Raven:** `lower_arith.ml:845` (`compare`, `at_most`).
- **Differs:** `x < y or x = y`, false on NaN; tinygrad's is true on NaN.
- **nx:** `nx.mli`, `less_equal`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `comparisons › less_equal`,
  `exact operations on the host › floats › less_equal` (slow).

### A21. Saturating conversion to integers

- **Reference:** `mixin/dtype.py:19` (`cast`).
- **Raven:** `lower_arith.ml:858` (`saturate`).
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
- **Raven:** `lower_arith.ml:899` (`threefry`).
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
- **Raven:** `lower_arith.ml:878` (`to_float32_odd`, `cast`).
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
