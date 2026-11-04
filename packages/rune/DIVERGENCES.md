# Divergences

rune follows JAX for what its transformations mean and tinygrad, through
tolk, for how it lowers nx's operations. This file lists where it departs
from either: each entry names the code (file and function), what differs
and why, and the test that pins it. A reference is cited by file and line at
the version its section pins; raven code by file and function, never by line.
An entry goes when its reason goes, and its number is not reused.

A test is named by its module's suite (`packages/rune/test/test_<module>.ml`)
and its path in it.

## Transformations

The reference is JAX's `jax.lax` and `jax.vmap` at `jax-v0.12.0.1`
(`dc4af684446e`), its files under `jax/_src/`.

### T1. A switch index out of range raises

- **Reference:** `lax/control_flow/conditionals.py:70,132` (`switch`: the
  index clamped into `0 .. n-1`).
- **Raven:** `program.ml` (`default`, its `Switch` case); `compile.ml`
  (`switch`); `vmap.ml` (`switch_apart`).
- **Differs:** an index outside `0 .. n-1` raises `Invalid_argument` at the
  call, under a `jit` as the call's check (at the call on the host, at the
  next wait on its set elsewhere), naming the lane under a map.
- **Reason:** a clamped index takes the first or last branch with no error.
- **Pinned by:** `Switch › an index out of range raises`.

### T2. An iterate is bounded and differentiates in reverse

- **Reference:** `lax/control_flow/loops.py:1663` (`while_loop`, any number of
  steps), `:2131` (`_while_transpose_error`: no reverse derivative), `:2671`
  (`fori_loop`: a loop differentiated in reverse is a `scan` of fixed
  length).
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
  branch over every lane and selects (`lax/control_flow/conditionals.py:461`,
  `_cond_batching_rule`). A `lax.while_loop` whose test has lanes runs the
  body over every lane until the last stops, selecting each stopped lane's
  carry: its documentation is silent there, and its batching rule
  (`lax/control_flow/loops.py:1860`, `_while_loop_batching_rule`) does
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

### T4. A compiled call keys on the maps its trace asked about

- **Reference:** `jax.jit`'s tracing cache keys on the whole axis
  environment, the name and size of every enclosing named map
  (`core.py:1427`, `axis_env_state`, `include_in_jit_key=True`), and
  `lax.axis_size` raises for a name no map binds (`lax/parallel.py:844`).
- **Raven:** `jit.ml` (`find`, `current`); `trace.ml` (`trace_in`, its `Bound`
  answer; `claims`); `rune.ml` (`lane_count`).
- **Differs:** a key holds, for each name the trace passed to `lanes`,
  `lane_index` or `lane_count`, the lanes of the map of that name around the
  call, or that none is. A map the trace never asked about does not retrace
  it. `lane_count` answers `None` where no map of the name is around.
- **Reason:** a function under a named map it never reads compiles once, and
  code that acts only inside a map, such as a custom rule's tangent map that
  gathers the lanes, asks without raising.
- **Pinned by:** `Jit › maps around › a call retraces iff its shape and its
  trace's answers are new`, `› a caught raise outside the map is keyed`,
  `Vmap › names › lane counts`.

## Lowering

rune lowers each nx kernel operation to tolk's UOps (`lower*.ml`). Where
tinygrad's `Tensor` builds the same operation, the lowering builds tinygrad's
decomposition if it computes what nx states. This section lists every place
where it builds something else, or builds what tinygrad has no source for.

An entry is admitted for one of four reasons only:

- **(a) an OCaml constraint;**
- **(b) nx's stated meaning:** the operation's contract in `Nx_kernel`
  (`packages/nx/lib/kernel/nx_kernel.mli`, `prog.mli`), which nx's eager
  kernels compute;
- **(c) the engine's contract:** rig's submission protocol;
- **(d) measured speed:** of a named raven workload, pinned by a bench row on
  fudge whose baseline would catch the regression.

A part of an entry whose reason is missing says "Reason: none" until it is
measured, withdrawn or redesigned.

Each entry also gives the reference (tinygrad `f1d8f0f0def4`, the commit
tolk follows, its files under `tinygrad/`; or none), nx's meaning, the
agreement class (exact, bit for bit as `rune.mli` defines it; an ulp bound; a
rounded sum; a stated bound) and its reason. An ulp bound is the
one `lower_real.mli` states, which `Lower_real`'s suite holds on the host; a
rounded sum's is `lower_reduce.mli`'s; a matrix routine's is
`lower_linalg.mli`'s.

### A1. Narrow floats compute at float32

- **Reference:** `mixin/elementwise.py`, each operation at the operands' dtype.
- **Raven:** `lower_arith.ml` (`narrow`, `decode`, `encode`, `narrowed`).
- **Differs:** an operation or comparison on `float16`, `bfloat16`, an 8-bit
  float or `float4_e2m1fn` widens its operands exactly to `float32`, computes,
  and narrows the result once, to nearest with ties to even. A renderer's
  half-precision intrinsics never apply. `float16` and `bfloat16` convert by
  tolk's casts, as tinygrad's do; a NaN they give is a NaN of unspecified sign
  and payload, quiet where its operand is, as nx's NaN rule allows. The rule
  departs from IEEE 754-2019 §6.2.1 as LLVM's semantics of NaN values
  (LangRef, "Behavior of Floating-Point NaN values") and Rust's (RFC 3514) do:
  no operation reads an exception flag, so a signalling NaN computes as a
  quiet one. tolk holds
  the 8- and 4-bit floats as bytes, which rune decodes and encodes by integer
  formulas on their bits.
- **nx:** `prog.mli`, Results: floats narrower than `float32` compute in
  `float32` and round once to their dtype. `dtype.mli`, Stores: the NaN rule.
- **Class:** as the operation's; a transcendental kind within one rank of the
  correctly rounded result.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › exact unary kinds give eager's bits`, `› exact
  binary kinds, comparisons, where and fma give eager's bits`, `› narrow
  floats' transcendental kinds are within one ulp of the correctly rounded
  result`, `› 16-bit casts › *` on the host and `Jit_gpu › 16-bit casts › *`
  on a GPU (every code and every tie, the targets' casts against eager's
  bits).

### A2. Integer arithmetic wraps

- **Reference:** `mixin/elementwise.py:73,83,102,124,547` (`neg`, `add`,
  `sub`, `mul`, `pow`).
- **Raven:** `lower_arith.ml` (`lift`, `store`, `int_unary`, `int_pow`,
  `int_binary`).
- **Differs:** signed `Neg`, `Abs`, `Add`, `Sub`, `Mul`, `Fma` and integer
  `Pow` compute on the unsigned bit pattern: `uint32` for widths below 32
  bits, the unsigned type of the width otherwise. tinygrad computes in the
  signed type, whose overflow C leaves undefined, and so does its `Pow` where
  it multiplies out a constant exponent (`uop/symbolic.py:16`,
  `simplify_pow`). An integer `Pow` is tolk's `Pow` of the unsigned patterns,
  which squares over the exponent's bits
  (`codegen/decomp/transcendental.py:267`, `xpow_int`), and a negative
  exponent's 0, 1 or -1 is selected on the signed values; tinygrad's `pow`
  refuses a negative constant exponent of an integer
  (`mixin/elementwise.py:563`).
- **nx:** `prog.mli`, `binary`: integers wrap; `Pow` of a negative exponent is
  0, but 1 for a base of 1 and `±1` for a base of -1.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › exact unary kinds give eager's bits`, `› exact
  binary kinds, comparisons, where and fma give eager's bits`, `› powers by
  constants give eager's bits`.

### A3. Integer reciprocal

- **Reference:** `mixin/elementwise.py:459` (`reciprocal`).
- **Raven:** `lower_arith.ml` (`int_unary`, its `Recip` case).
- **Differs:** on integers, `x` at 1 and -1, 0 elsewhere and at 0. tinygrad's
  reciprocal is a float operation (`uop/ops.py:148`).
- **nx:** `prog.mli`, `Recip`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_arith › exact unary kinds give eager's bits`.

### A4. Absolute value

- **Reference:** `mixin/elementwise.py:909` (`x * sign x`).
- **Raven:** `lower_bits.ml` (`magnitude`), through `lower_real.ml`
  (`unary`); `lower_arith.ml` (`int_unary`).
- **Differs:** a float's sign bit is cleared, so `abs (-0.)` is `0.` and a
  NaN keeps its payload; tinygrad's product gives `-0.` at `-0.`. A signed
  integer is negated modularly (A2).
- **nx:** `prog.mli`, Results: `Abs` sets a NaN's sign bit as a number's, its
  payload kept.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › exact › abs`, `Lower_arith › exact unary kinds
  give eager's bits`.

### A5. Sign of NaN

- **Reference:** `mixin/elementwise.py:899` (`sign`).
- **Raven:** `lower_real.ml` (`sign`).
- **Differs:** `sign nan` is NaN; tinygrad gives 1.
- **nx:** `prog.mli`, `Sign`: NaN for a NaN.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › exact › sign`.

### A6. Rounding half away from zero

- **Reference:** `mixin/elementwise.py:888` (`round`, half to even).
- **Raven:** `lower_real.ml` (`round`).
- **Differs:** a half rounds away from zero, decided on the exact rest
  `x - trunc x`.
- **nx:** `prog.mli`, `Round`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › exact › round`.

### A7. Exponential in two parts

- **Reference:** `mixin/elementwise.py:510` (`exp2 (x * 1/ln 2)`).
- **Raven:** `lower_real.ml` (`exp_parts`, `exp2_split`, `exp`).
- **Differs:** rounding the product `x log2 e` costs up to about `|x|` ulps.
  The product is kept in two parts, its error exact from an fma, and
  `2^(hi + lo)` is `2^hi (1 + lo ln 2)`. A positive `hi` takes `2^(hi - 1)`,
  doubled after the product, since `hi` may round up to where `2^hi`
  overflows while the result does not.
- **nx:** `prog.mli`, Results, `Exp`.
- **Class:** ulp bound: 9 at `float32` and `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › exp within 2 ulps at float32, 9
  at float64`.

### A8. Cosine reduced by a quarter turn

- **Reference:** `mixin/elementwise.py:499` (`sin (pi/2 - x)`).
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
- **Pinned by:** `Lower_real › transcendental › cos within 2 ulps at float32, 9
  at float64`, `Lower_real › bounded arguments › cos of an argument bounded
  below the switch-over`.

### A9. Tangent of the reduced argument

- **Reference:** `mixin/elementwise.py:919` (`sin / cos`).
- **Raven:** `lower_real.ml` (`tan`).
- **Differs:** A8's reduction, then `sin r / cos r` or `-cos r / sin r` by the
  quadrant's parity, with `x`'s sign.
- **nx:** `prog.mli`, Results, `Tan`.
- **Class:** ulp bound: 19.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › tan within 19 ulps at float32,
  19 at float64`, `Lower_real › bounded arguments › tan of an argument bounded
  below the switch-over`.

### A10. Arcsine and arccosine by the arctangent

- **Reference:** `mixin/elementwise.py:929,942` (Abramowitz-Stegun 4.4.46,
  `acos = pi/2 - asin`).
- **Raven:** `lower_real.ml` (`asin`, `acos`).
- **Differs:** tinygrad's polynomial is float32-grade, has no relative accuracy
  near 0, and `pi/2 - asin` cancels near 1. The lowering takes
  `asin x = atan (x / sqrt (1 - x^2))`, `1 - x^2` by an fma, and
  `acos x = 2 atan (sqrt ((1 - x) / (1 + x)))`, through A11.
- **nx:** `prog.mli`, Results, `Asin`, `Acos`.
- **Class:** ulp bound: 5 at `float32`, 9 at `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › asin within 5 ulps at float32,
  9 at float64`, `› acos within 5 ulps at float32, 9 at float64`.

### A11. Arctangent

- **Reference:** `mixin/elementwise.py:952` (`asin (x / sqrt (1 + x^2))`).
- **Raven:** `lower_real.ml` (`atan_unit`, `angle`, `atan`).
- **Differs:** `x^2` overflows past `1.8e19` in `float32`, where tinygrad's
  arctangent is 0. The lowering takes the arctangent of `min (|x|, 1/|x|)`,
  in `[0, 1]`, by a polynomial (at `float64` after the half-angle step), and
  `pi/2` in two parts less it past 1, with `x`'s sign.
- **nx:** `prog.mli`, Results, `Atan`.
- **Class:** ulp bound: 2 at `float32`, 7 at `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › atan within 2 ulps at float32,
  7 at float64`.

### A12. Two-argument arctangent

- **Reference:** none.
- **Raven:** `lower_real.ml` (`atan2`).
- **Differs:** a raven composition: A11's angle of `|y|` over `|x|`, `pi` in two
  parts less it where `x`'s sign bit is set, `y`'s sign, and C's values for
  both zeros and both infinities.
- **nx:** `prog.mli`, `Atan2`.
- **Class:** ulp bound: 6 at `float32`, 8 at `float64`.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › atan2 within 6 ulps at float32,
  8 at float64`.

### A13. Hyperbolic functions

- **Reference:** `mixin/elementwise.py:1032,1044` (`(e^x -+ e^-x) / 2`), `:755`
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
- **Class:** ulp bound: `sinh` 23, `cosh` 10, `tanh` 18.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › sinh within 23 ulps at float32,
  23 at float64`, `› cosh within 10 ulps at float32, 10 at float64`, `› tanh
  within 18 ulps at float32, 18 at float64`.

### A14. Error function

- **Reference:** `mixin/elementwise.py:1056` (Abramowitz-Stegun 7.1.26).
- **Raven:** `lower_real.ml` (`erf32`, `erf_near64`, `erf_far64`, `erf`).
- **Differs:** tinygrad's approximation has an absolute error of `1.5e-7` and
  no relative accuracy near 0. The lowering takes a polynomial per range: at
  `float32`, one in `x^2` below 1 and one in `|x| - 2.5` to 4; at `float64`,
  two up to 2.5, then `1 - e^(G(a) - a^2)`, `a` held at 5.95, where `erf`
  rounds to 1.
- **nx:** `prog.mli`, Results, `Erf`.
- **Class:** ulp bound: 2 at `float32`, 3 at `float64`, measured.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › erf within 2 ulps at float32, 3
  at float64`.

### A15. True division

- **Reference:** `mixin/elementwise.py:228` (`a * (1/b)`).
- **Raven:** `lower_real.ml` (`( /: )`, through `binary`).
- **Differs:** one rounding: `Ops.FDIV`, which tolk renders as `a / b`.
  tinygrad multiplies by the reciprocal, two roundings.
- **nx:** `prog.mli`, `Fdiv`: the IEEE 754 quotient.
- **Class:** exact where the target's division is correctly rounded.
- **Reason:** (b).
- **Pinned by:** `Lower_real › exact › fdiv`.

### A16. Integer quotient and remainder by 0 and -1

- **Reference:** `mixin/elementwise.py:228,215` (`Ops.TRUNCDIV`,
  `Ops.TRUNCMOD`).
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

- **Reference:** `mixin/elementwise.py:215` (`x - trunc (x * (1/y)) * y`).
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
- **Pinned by:** `Lower_real › exact › mod`.

### A18. Power with its product in two parts

- **Reference:** `mixin/elementwise.py:547` (`Ops.POW`, `xpow` =
  `exp2 (y log2 |x|)`).
- **Raven:** `lower_real.ml` (`pow`).
- **Differs:** rounding `y log2 |x|` costs up to about its magnitude in ulps.
  The product's error, exact from an fma, is kept, and the two parts are
  exponentiated as A7's. A negative base takes the sign of an odd integral
  exponent and is NaN for a fractional one, and the special values are C99
  F.9.4.4's: `pow 1 nan` is 1, where tinygrad's `xpow`
  (`codegen/decomp/transcendental.py:257`) gives NaN.
- **nx:** `prog.mli`, `Pow` and Results.
- **Class:** ulp bound: 9.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › pow within 2 ulps at float32, 2
  at float64`.

### A19. Maximum and minimum

- **Reference:** `mixin/elementwise.py:377,392` (`Ops.MAX`; `minimum` as
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
- **Pinned by:** `Lower_real › exact › maximum`, `› minimum`.

### A20. Less than or equal

- **Reference:** `mixin/elementwise.py:329` (`not (x > y)`).
- **Raven:** `lower_real.ml` (`le`, through `compare`).
- **Differs:** `x < y or x = y`, false on NaN; tinygrad's is true on NaN.
- **nx:** `prog.mli`, Domains: every comparison with a NaN is false but
  `Not_equal`.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_real › exact › less_equal`.

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
- **Pinned by:** `Lower_arith › casts give eager's values`.

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
- **Pinned by:** `Lower_arith › casts give eager's values`, whose `float64`
  operands lie one ulp around each narrow format's ties.

### A24. Logarithm by fdlibm, NaN below zero

- **Reference:** `mixin/elementwise.py:828` (`log2 (x) * ln 2`), whose `log2`
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
- **Pinned by:** `Lower_real › transcendental › log within 1 ulps at float32, 1
  at float64`, whose golden points crowd around 1.

### A25. `expm1` and `log1p`

- **Reference:** none.
- **Raven:** `lower_real.ml` (`expm1`, `log1p`).
- **No source:** `expm1` is Kahan's: with `w = e^x` rounded, `(w - 1) x / log
  w`, so that `exp`'s error passes through, halved near 0. It is `x` where
  `w` is 1, -1 where `w - 1` is, and `w` where `w` is infinite. `log1p` is
  Goldberg's: with `w = 1 + x` rounded, `log w x / (w - 1)`, `w - 1` exact,
  and `x` where `w` is 1 or `x` is `+inf`. The `exp` and `log` are A7's and
  A24's.
- **nx:** `prog.mli`, `Expm1`, `Log1p`: accurate near 0; Results.
- **Class:** ulp bound: `expm1` 23, `log1p` 5.
- **Reason:** (b).
- **Pinned by:** `Lower_real › transcendental › expm1 within 23 ulps at
  float32, 23 at float64`, `› log1p within 5 ulps at float32, 5 at float64`,
  whose points include Annex F's specials.

### A26. Complex kinds

- **Reference:** none: tinygrad has no complex dtype (`dtype.py:183`).
- **Raven:** `lower_complex.ml` (`unary`, `binary`, `compare`, `mul`, `fdiv`,
  `smith`, `sign_raw`, `sqrt_raw`, `exp_raw`, `exp2_raw`, `log_abs`,
  `pow_raw`, `sinh_raw`, `cosh_raw`, `tanh_raw`, `asin_raw`, `acos_raw`,
  `atanh_raw`); `lower_arith.ml` (`parts`, `of_parts`, `cast_parts`, `raw`).
- **No source:** a number is a pair of float nodes, its real part first, each
  kind a composition of `Lower_real`'s kinds.
  - `Neg`, `Add` and `Sub` act on each part. `Maximum`, `Minimum` and the
    comparisons order by real part, then imaginary part, and treat a number
    with a NaN part as a NaN. The comparisons compare each part as IEEE 754
    does, `-0.` equal to `0.`; `Maximum` and `Minimum` take `-0.` below
    `0.`.
  - `Mul` is the textbook product, each part one fma over the other product.
    `Fdiv` is Smith's algorithm, a dividend whose parts are below the least
    normal over `u` scaled by `2^25` or `2^54` first and the quotient back.
    `Recip` is `(1 + 0i) / z`.
  - `Exp` and `Exp2` multiply `cis y` by `2^(hi + lo)` split at the nearest
    integer `k`, `2^k` applied last, so that a part overflows only with the
    result. `Log` is `log |z| + i atan2 (y, x)`, `log |z|` by `log1p` of
    exact products near the unit circle. `Sqrt` is Kahan's. `Pow` is
    `e^(w log z)`; a real `w` multiplies `log z`'s parts alone, and `z^0` is
    1. `Sign` is `z / |z|` of `z` scaled by a power of two.
  - The hyperbolic kinds take their real parts' `sinh` and `cosh` and, past
    `hyp_far`, the polar form; `Tanh` is Kahan's. `Asin` and `Acos` are
    Kahan's from `sqrt (1 - z)` and `sqrt (1 + z)`, with their leading terms
    at infinity past `2^13` (`float32`) or `2^28` (`float64`). `Atanh` is
    Kahan's. The circular kinds turn the hyperbolic ones by `i`, and `Asinh`
    and `Acosh` turn `Asin` and `Acos`.
  - A cast from a complex number reads its real part; one into it sets the
    imaginary part to `+0.`. A `complex64`'s bits are its real part's in the
    low word.
- **nx:** `prog.mli`, Domains and Results, Complex numbers.
- **Class:** exact for `Neg`, `Add`, `Sub`, `Maximum`, `Minimum` and the
  comparisons; stated bound: `lower_complex.mli`'s normwise table for the
  others.
- **Reason:** (b).
- **Pinned by:** `Lower_complex › special values › *`, `Lower_complex ›
  bounds › *`, each over every kind at `complex64` and `complex128`;
  `Lower_arith › exact unary kinds give eager's bits`, `› exact binary kinds,
  comparisons, where and fma give eager's bits`, `› casts into complex128 give
  eager's values in both parts`.

### R1. A float sum adds +0.

- **Reference:** `mixin/reduce.py:20` (`sum`), `mixin/op.py:781`
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

- **Reference:** `mixin/reduce.py:73` (`max`, `Ops.MAX`), `mixin/op.py:496`
  (`min`, `-max(-x)`), `:821`, `:839` (`cummax`, `cummin`).
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

- **Reference:** `mixin/op.py:886` (`argmax`), `:913` (`argmin`), `:821`
  (`cummax`), `:839` (`cummin`).
- **Raven:** `lower_reduce.ml` (`same`, `position`, `changes`).
- **Differs:** the extreme is R3's; its position is that of the first term
  with its bits, or with a NaN where it is a NaN. tinygrad's equality with the
  maximum never holds for a NaN and holds for both zeros. A running arg is at
  each position the last one up to it where the running extreme changes its
  bits, the first term with them; tinygrad's `cummax` gives the last term up
  to it equal to the running maximum, by the same equality.
- **nx:** `nx_kernel.mli`, `Arg`: the first NaN, else the first extreme term.
- **Class:** exact.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › logsumexp, moments and arg › an arg gives
  eager's values and positions`, `› a running arg gives eager's values and
  positions`.

### R5. Stable sorts of packed keys

- **Reference:** `mixin/op.py:936` (`sort`: a bitonic network of the values,
  each position recovered by matching equal values and their counts), `:988`
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

### R6. Complex reductions, scans and contractions by parts

- **Reference:** none: tinygrad has no complex dtype (`dtype.py:183`).
- **Raven:** `lower_reduce.ml` (`refuse`, `extreme`, `fold`, `position`,
  `products`), `lower_index.ml` (`keys`).
- **No source:**
  - A sum adds each part as a float sum, as R1 and R2.
  - An extreme orders by real part, then imaginary part, on R3's keys. With
    `float32` parts, each key is biased to unsigned and the real key packed
    above the imaginary one in a `uint64`, whose `Max` is the extreme. With
    `float64` parts, a `Max` of the real keys, then a `Max` of the imaginary
    keys of the terms whose real key is it. A term with a NaN part takes the
    greatest key, and the extreme then has two NaN parts. An arg is R4's
    position, a term matching where each part has the extreme's bits, or
    where both have a NaN part.
  - A sort orders R5's keys of the real part, then the imaginary part, a
    number with a NaN part the greatest.
  - A contraction sums, per product, `ar br - ai bi` and `ar bi + ai br`,
    each rounded before the sum takes it; otherwise as L1.
  - A `Prod` reduction or scan, and a `complex128` `Max`, `Min` or `Arg`
    scan, whose second `Max` a scan cannot take, are refused, and the
    program runs uncompiled.
- **nx:** `nx_kernel.mli`, Reductions: a complex `Sum` adds each part as a
  float sum, and to every other reduction a number with a NaN part is a NaN;
  Contraction: a complex product's terms; `prog.mli`, Domains: the order.
- **Class:** exact for extremes, args and sorts; rounded sum for sums;
  `lower_reduce.mli`'s complex bound for contractions.
- **Reason:** (b). The refusals: none.
- **Pinned by:** `Lower_reduce › reductions › extremes give eager's values`,
  `› complex sums add their parts`, `Lower_reduce › scans › extreme scans
  give eager's values`, `› complex running sums add their parts`,
  `Lower_reduce › logsumexp, moments and arg › an arg gives eager's values and
  positions`, `› a running arg gives eager's values and positions`,
  `Lower_reduce › contractions › complex contractions hold their bound`,
  `Lower_reduce › cases › a complex product is refused`, `› a complex running
  product is refused`, `› a complex128 running maximum is refused`,
  `Lower_index › sorts › sorts give eager's values and positions`.

### R7. Logsumexp in the carrier, its scan by the network

- **Reference:** `mixin/op.py:653` (`logsumexp`: `c` the maximum where finite
  and 0 elsewhere, `log Σ exp (x - c) + c`), `:853` (`logcumsumexp`).
- **Raven:** `lower_reduce.ml` (`logsumexp`, `shift`, `lse_terms`,
  `lse_combine`, `network`, `reduce`, `scan`).
- **Differs:**
  - A reduction builds tinygrad's composition, with R3's maximum, A7's `exp`,
    A24's `log` and R1's and R2's sum. It computes in the carrier and rounds
    once into the destination. tinygrad shifts, exponentiates and takes the
    `log` of a narrow float at its own dtype, around a `float32` sum.
  - A scan combines states of a maximum and a sum of `exp (x - c)` by the
    odd/even network, each combination rescaling both sums to the combined
    maximum: `O(n)` work in `2 log₂ n` kernels. tinygrad sums, at each
    position, the `exp`s of the whole axis against the running maximum,
    masked past the position by the dtype's least value, `-∞` where it has
    one: `n²` work, and a sum over the axis whose association tolk chooses
    from the axis's extent, so that a prefix's results can differ from the
    whole axis's.
- **nx:** `nx_kernel.mli`, `Logsumexp`: `-∞` for no term or terms all `-∞`,
  `+∞` with a `+∞` term and no NaN; `carrier`; Scans: a result is a function
  of the terms up to it, bit for bit, and the odd/even network.
- **Class:** stated bound: `lower_reduce.mli`'s `Logsumexp` rows.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › logsumexp, moments and arg › a logsumexp
  holds its bound`, `› a running logsumexp holds its bound`, which cover
  slices of `-∞` alone, NaN and infinite terms and narrow destinations. No
  test holds the prefix property.

### R8. Moments by corrected two passes, their scan by the network

- **Reference:** `mixin/op.py:519` (`mean`), `:546` (`var` with
  `correction=0`, the population variance: `Σ (x - mean)² / n`).
- **Raven:** `lower_reduce.ml` (`moments`, `moments_combine`, `network`,
  `reduce`, `scan`).
- **Differs:**
  - A reduction's variance is `(Σ d² - Σ d (Σ d / n)) / n`, `d = x - mean`
    (Chan, Golub and LeVeque's corrected two-pass algorithm), `+0.` where
    rounding makes it negative and `Σ d²` where that is `+∞`. tinygrad
    divides `Σ d²` by `n`. Its quotients are A15's. It computes in the
    carrier and rounds once into each destination; tinygrad squares a narrow
    float's deviations at its own dtype.
  - A scan has no source: states of a mean, a sum of squared deviations and
    a count, of the terms less the slice's first term (0 where not finite),
    combined by Chan, Golub and LeVeque's (1.5b), the mean weighted, over the
    odd/even network. The variance is the sum over the count, NaN where the
    mean is not finite.
- **nx:** `nx_kernel.mli`, `Moments`: the population mean and variance, NaN
  for no term or a NaN term, the variance NaN with an infinite term and
  never negative; `carrier`; Scans.
- **Class:** stated bound: `lower_reduce.mli`'s `Moments` rows.
- **Reason:** (b) for the carrier and the scan. The correction term: none.
  nx leaves the algorithm to each library; `nx_cpu.mli` (Logsumexp and
  Moments) states the same one for the eager kernel.
- **Pinned by:** `Lower_reduce › logsumexp, moments and arg › moments hold
  their bounds`, `› running moments hold their bounds`, `Lower_reduce › cases
  › a running variance that overflows is +∞`.

### I2. Pieces are selected as their bits

- **Reference:** `mixin/op.py:773` (`cat`: each piece zero-padded to the whole,
  the pieces summed).
- **Raven:** `lower_index.ml` (`assemble`, `placed`, `joined`, `crowded`);
  `compile.ml` (`issue`, `assembled`).
- **Differs:** an assembly places each piece at its region and selects it with
  `where` over the pieces before it, floats as their bits; empty pieces are
  dropped first. The reference's sum turns a `-0.` into `+0.` and quiets a
  signalling NaN. In a program on one device that runs on the host, an
  assembly whose selections exceed 16 times eager's work (`p N > 16 (N +
  m)`, `p` pieces of `m` elements in all over `N`) is declined, and the
  program runs uncompiled. On another device, as a GPU, which has no eager
  kernels to run it, the selections compile whatever the work, as the
  reference sums any number of pieces. In a program over several devices an
  assembly is storage of its own (`compile.ml`, `assembled`): pieces that hold each element once
  are each stored into their region; otherwise the fill is stored over it,
  then each piece in order, a region by steps as its box with the piece
  selected over what the box held. Each store reads one piece, where a
  selection reads all of them in one kernel.
- **nx:** `nx_kernel.mli`, `assemble`: the last piece holding each index, or
  the fill, bits copied.
- **Class:** exact.
- **Reason:** (b).
  - The decline on the host: none until a baseline of
    `Jit/stack-64-sines-of-64` in `rune.thumper`, 64 rows of 64 sines
    stacked, is recorded; then (d) by it. On fudge, three interleaved runs
    each, the row takes 52 to 87 µs evaluated against 0.83 to 4.2 ms
    compiled, since the selections compute every row's sines at each
    element.
  - The bound of 16 itself: none. It counts selections alone, and timing
    places the crossing elsewhere. In a probe on fudge (medians of 11, two
    interleaved runs), a compiled stack of copied rows of 64, past the bound,
    took 10.7 µs at 33 rows against 17.6 evaluated and 306 µs at 256 against
    301 to 379 in the quieter run, and a stack of 32 rows of sines, below the
    bound, 334 to 706 µs compiled against 44 to 68 µs evaluated. Placing the
    decline needs a cost model of the selected work, a design not yet made.
  - The stores over several devices: none. They keep a fully sharded step
    within the bound `rune.mli` states (Fully sharded training): the most a
    step holds above its state, over 2, 4 and 8 devices at `d = 512`, is
    2,231,040, 1,774,080 and 1,544,320 bytes, against 3,082,240, 2,590,720
    and 2,346,880 with one selecting kernel per device and bounds of
    2,293,760, 1,802,240 and 1,556,480. No admitted reason covers memory,
    and they cost speed: `Jit/devices/dp-step-16` takes 4.9 to 6.2 ms
    against 1.19 to 1.22 ms with the selection, and
    `Jit/devices/fsdp-backward-8` 20.9 to 24.9 ms against 14.3 to 25.7 ms
    (three interleaved runs each). Trading the memory against that time
    needs a design.
- **Pinned by:** `Lower_index › folds and assemblies › an assembly gives
  eager's bits`, `Lower_index › work bounds › a scatter or an assembly past
  16 times eager's work gives eager's bits`, `Jit › jit › a stack of 33 rows
  on the host is evaluated`, `Jit_gpu › assemblies › an assembly past 16
  times eager's work compiles to eager's bits`,
  `Jit_devices › assemblies give eager's bits › *`, `Jit › fully sharded ›
  *`.

### I3. Gather by an index

- **Reference:** `mixin/op.py:1064` (`gather`: a one-hot selection summed).
- **Raven:** `lower_index.ml` (`gather`), `lower_bits.ml` (`take`).
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

- **Reference:** `mixin/op.py:1100` (`_pre_scatter`), `:1150`
  (`scatter_reduce` sum: the masked updates summed, then `x` added), `:1191`
  (`scatter` through `_masked_merge`, one `where` per update along the axis).
- **Raven:** `lower_index.ml` (`scatter`, `summed`, `looped`, `walk`).
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
  - Past 16 times eager's work (`n k > 16 (n + k)`, `n` positions and `k`
    updates along the axis), or where the destination's storage holds `x`
    already, as a loop's carry or a donated argument does, the scatter
    updates that storage in place in a kernel of its own (`looped`, `walk`):
    each thread takes a run of positions and walks the updates in C order,
    combining each that lands in its run, for `runs k + n` work. The
    reference has no such kernel.
- **nx:** `nx_kernel.mli`, `scatter`: `into` where no update lands, bits
  unchanged.
- **Class:** exact; rounded sum for float `Add`.
- **Reason:** (b).
  - The kernel that walks the updates past the bound: (d), by
    `Jit/take-grad-1024-of-4096x64` (an embedding's gradient) and
    `Jit/scatter-1024-rows-into-donated-4096x64` (a donated table) in
    `rune.thumper`.
  - Below the bound, into a carry's or a donated argument's storage: none
    until wall-time baselines of `Jit/scatter-16-rows-into-donated-4096x64`
    and `JitLoops/iterate-64-rows-into-4096x64-replay` are recorded with the
    kernel, the first's being the comparison's and the second's missing;
    then (d) by them. On fudge, three interleaved runs each, they take 1.6
    to 6.9 µs and 158 to 276 µs, against 766 to 1040 µs and 1.02 to 1.89 ms
    with the comparison of every element.
  - The bound of 16 itself: none, and timing contradicts it. In a probe on
    fudge's host (medians of 11, two interleaved runs), a fresh scatter of
    `k` updates into 1024 elements takes 2.3 to 2.5 µs looped from `k = 2`
    against 3.2 to 8.0 µs compared up to `k = 16`; `k` rows of 64 into 4096
    take 66 to 94 µs looped against 212 to 573 µs compared from `k = 8` to
    16. At `k = 1`, and at 4 rows, the runs overlap. On the RTX 5000 Ada
    both forms take 10 to 28 µs up to 17 updates, within each other's
    noise. Lowering the bound, or one looped form for every scatter, needs
    a design.
- **Pinned by:** `Lower_index › gathers and scatters › a scatter by Set, Max
  or Min gives eager's bits`, `› a scatter by Add wraps integers and holds its
  float bound`, `› a scatter by Set, Max or Min past 16 times eager's work
  gives eager's bits`, `› a scatter by Add past 16 times eager's work holds
  its bound`; `Lower_index › work bounds › a scatter or an assembly past 16
  times eager's work gives eager's bits`.

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

- **Reference:** `mixin/movement.py:614,619` (`_pool`).
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

- **Reference:** `mixin/op.py:390` (`dot`: `(x * w).sum(-1)`, the products at
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

- **Reference:** `mixin/op.py:1838` (`qr`).
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

- **Reference:** `mixin/op.py:1856` (`svd`: `4 num` rounds of one-sided Jacobi
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

### L7. Segmented contractions read each group's indices by position

- **Reference:** none: tinygrad has no segmented contraction. Its gather,
  `mixin/op.py:1064`, is a one-hot `where` summed over the axis, which
  `codegen/simplify.py:122` (`pm_reduce_load_collapse`, under
  `pm_load_collapse`, `:157`) rewrites into a load at the index.
- **Raven:** `lower_reduce.ml` (`route`, `rows`, `blocked`,
  `block_size`), `lower_bits.ml` (`take`).
- **Differs:** every read is a load at its index by `take`, never a one-hot
  sum. A `Rows` contraction of `n` rows over `g` groups below `4 g` rows
  reads `b` at each row's id, clamped into `[0, g - 1]`. From `4 g` rows
  it routes the rows by counting (a one-hot of the ids, its running sum,
  each group's count and the scan of its blocks, each slot's row by a binary
  search of its group's running count) into blocks of rows of one group, 64
  from 16 rows a group, fewer below with their padding under half the rows:
  a block reads its group's slice, its rows at their slots, a slot of none
  reading `0`, and each row reads its slot's stored output. A row reads its
  own group's slice alone, its bits exact, a NaN's payload and `-0.`
  included. A `Contracting` pair is routed alike and cuts each group into
  blocks of `c` indices from its first, each position a load at its index,
  a position past its group's last reading `0`; each block sums into a
  partial result stored apart, and each group sums its own blocks' over the
  `⌈n / c⌉` places a group can take, its `i`th block at place `i`, `+0.`
  past its last. Where blocks take no fewer products and sums than `g n`,
  `c` is `n`, one block a group. Either way a group's sum is split by its
  own count, never by the other groups'.
- **nx:** `nx_kernel.mli`, `segments` and `contract`: each group's products
  alone; an output without products is its `init`.
- **Class:** exact for the `Rows` read and on integers; a rounded sum on
  floats, its association one of those the bound allows and a function of
  the shapes and the group's own count, a zero result's sign open.
- **Reason:** (b).
- **Pinned by:** `Lower_reduce › contractions › segmented contractions give
  eager's integers and hold their bound`, `› contracting pairs in blocks give
  eager's integers and hold their bound`, `› rows in blocks give eager's
  integers and hold their bound`, `Lower_reduce › work › a Rows
  contraction's work is one slice's at 128 and 256 groups`, `› a scaled
  float4 Rows contraction's work is one slice's`, `› a Rows contraction of
  8 rows a group works as 1.5 times its rows'`, `› a Rows contraction in
  blocks works as its slots' at 8 groups`, `› a Contracting contraction of 8
  groups works in blocks`, `Jit › classes › a segmented contraction gives
  eager's values`, `› a segmented contraction's gradients give eager's
  values`, `› a group's weight gradient keeps its bits whatever the other
  groups' counts`, `› experts applied give eager's values`.

### L8. Eigenvalues by two-sided Jacobi rotations

- **Reference:** none.
- **Raven:** `lower_linalg.ml` (`eigh`, `tangents`, `rotations`, `rotate`,
  `rounds`, `norm_roundoff`, `diagonal_within`, `order`).
- **No source:** the symmetric matrix the lower triangle names, the upper one
  never read, divided by the power of two of its largest magnitude, which
  the eigenvalues take back. A round rotates the pairs of a round-robin
  pairing, rows and columns, each by the angle that zeroes its off-diagonal
  element; an element within `u ‖a‖_F` takes none. It runs `⌈p ⌈log₂ n⌉ /
  10⌉` sweeps of `n - 1` rounds (`n` for an odd `n`), `p` the dtype's
  precision, one round a step of a tolk loop, and `v` accumulates the
  rotations. The eigenvalues are the diagonal in a stable ascending order,
  `v`'s columns in the same order. A matrix whose read triangle holds NaN or
  an infinity, or whose off-diagonal norm after the sweeps exceeds
  `16 n u ‖a‖_F`, is NaN in every result. A complex matrix is refused, and
  the program runs uncompiled.
- **nx:** `nx_kernel.mli`, `eigh`: ascending eigenvalues, orthonormal `v`,
  the lower triangle and the diagonal's real parts alone read, NaN as for
  `svd`.
- **Class:** stated bound: `Eigh`'s.
- **Reason:** (b). The refusal: none.
- **Pinned by:** `Lower_linalg › eigh › v diag w vᵀ reconstructs a`, whose
  upper triangle is garbage, `› NaN in the lower triangle is NaN`, `› norms
  past the dtype's range, and subnormals`; `Lower_linalg › Nx's values alone
  compiled are eager's`.

### L9. Block scales multiply each element in the accumulator

- **Reference:** none in `Tensor`. `llm/gguf.py:29` (`ggml_data_to_tensor`)
  decodes a block-scaled weight into a `float32` tensor, each element times
  its block's scale (MXFP4 `:171`, Q8_0 `:63`), which `mixin/op.py:390`
  (`dot`) then multiplies.
- **Raven:** `lower_reduce.ml` (`entered`, `blocks`, `power`,
  `bfloat16_values`, `contract`).
- **Differs:**
  - Each element is stored into the accumulator and multiplied there by its
    block's scale, read into the accumulator, rounded once; `float64`
    included. tinygrad's decode computes in `float32`.
  - Code 255 makes its block NaN. tinygrad's MXFP4 decode reads 255 as
    `2^127` times twice the element, infinite unless the element is `±0` or
    `±0.5`.
  - A `float4_e2m1fn` element with a code scale in `float32`, against a
    `bfloat16` operand, enters as the `float32` of its `bfloat16`, the same
    value, so that both operands take tolk's `bfloat16` tensor cores.
  - A scale with an accumulator other than `float32` or `float64` is
    refused, and the program runs uncompiled.
- **nx:** `nx_kernel.mli`, `scale`: code `e` scales by `2^(e - 127)`, 255
  makes it NaN; Contraction: the library states which a scale multiplies,
  each element rounded once into the accumulator among them.
- **Class:** rounded sum: `Nx.contract`'s bound with `K + 1 + s` for `s`
  scaled operands.
- **Reason:** (b). The `bfloat16` entry: none; it would be (d) by
  `GptOss/experts-at-prefill-512` (`gpt_oss.thumper`), whose fudge baseline
  holds allocated words alone. The refusal: none.
- **Pinned by:** `Lower_reduce › contractions › scaled float contractions
  hold their bound`, which covers a NaN code; `Lower_reduce › work › a scaled
  float4 Rows contraction's work is one slice's`; `Lower_reduce › cases › a
  contraction scaled in int32 is not lowered`; `Jit › classes › an MXFP4
  product is within its bound of the exact one` on the host and `Jit_gpu ›
  products › an MXFP4 product is within its bound of the exact one` on a GPU,
  each over `bfloat16` operands and scale bytes up to 255.

### L10. Fourier transforms by prime factors

- **Reference:** none.
- **Raven:** `lower_fft.ml` (`twiddle`, `table`, `line`, `butterfly`,
  `dense`, `bluestein`, `c2c`, `r2c`, `c2r`).
- **No source:** a line of `n` points is `p`-point transforms of `n / p`
  interleaved lines, `p` its least prime factor, then twiddles and the
  shorter lines (Cooley and Tukey). A 2-point stage is a butterfly, a `p` up
  to 250 a sum over the `p × p` twiddle table, and a larger `p` Bluestein's
  convolution over the least power of two at least `2p - 1`. Each stage is
  stored apart: one kernel per prime factor. A twiddle's parts are sines of
  arguments at most `π/2`, from `4k = q n + s`, exact at whole quarter turns.
  `r2c` transforms its points as complex numbers along the last axis first
  and keeps bins 0 to `n/2`. `c2r` completes the bins by `conj X[n - k]`,
  bin 0's and an even `n`'s bin `n/2`'s imaginary parts set to `+0.`, and
  keeps the real parts. An infinite point makes NaN results where it enters.
- **nx:** `nx_kernel.mli`, `direction`, `c2c`, `r2c`, `c2r`.
- **Class:** stated bound: `lower_fft.mli`'s.
- **Reason:** (b).
- **Pinned by:** `Lower_fft › transforms › complex transforms hold their
  bound`, `› real transforms hold their bound`, `› real inverse transforms
  hold their bound`, which cover transforms of no point, odd and mixed
  factors and a prime factor above 250.

### L11. Attention folds its keys in kernels of its own

- **Reference:** `mixin/rand.py:298-330` (`scaled_dot_product_attention`:
  the scores' product in float32, cast to the query's dtype, `softmax`,
  then the product with the values, each a kernel and the scores stored).
- **Raven:** `lower_attention.ml` (`lower`, `simt`, `core`, `plan`,
  `classes`, `skipping`, `run_tiles`, `combine`).
- **Differs:** one kernel folds each query row over its keys in tiles of
  32 or 64 with a running maximum, sum and accumulator in base 2 (online
  softmax), never storing a score; the rows of a group of query heads
  share each tile of keys. On a target whose tensor core of a narrow
  float's products chains its accumulator into its own operand, float16
  and bfloat16 attention computes both products on the core and rounds
  each weight once to the values' dtype. A tile the mask's ranges over the
  tile's pairs hide is skipped; one they leave unknown runs where a
  workgroup's vote finds a visible pair. Few workgroups split each row's
  keys into runs, folded in order by a second kernel.
- **nx:** `nx_kernel.mli` (attention: the pair program, visibility,
  [o] and [lse]), `nx.mli` (`attention`: the bound, the weights rounded
  once to [v]'s dtype for float16 and bfloat16, and a skipped hidden key's
  non-finite value).
- **Class:** stated bound: `nx.mli`'s, with `rune.mli`'s compiled
  `Nx.logsumexp` for [ε].
- **Reason:** (b): nx states attention as one operation over pairs whose
  kernels store nothing of the pair loop's shape; tinygrad's composition
  stores the scores and computes the weights in the query's dtype, which
  nx's bound does not allow.
- **Pinned by:** `rune attention › jit within nx's bound of a float64
  reference › every dtype`; `Rune.jit on a GPU › attention › jit holds nx's
  bound against a float64 reference`.
