(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for host memory.

    [computes_on d] is {!Rig.shares_host_memory}[ d]. Each kernel computes from
    operands of [dst]'s shape and any layouts, on memory the host addresses. It
    claims every operand through the door of [nx_array.h] for the extent of its
    call. It answers [Done] once its work is done, or, before any write, a
    refusal: the door's, or [Shape_mismatch] if the operands' shapes differ.

    [apply0] to [apply3] compute every kind at every dtype of its domain. A
    [Copy] whose operand's dtype is not [dst]'s is refused with
    [Wrong_dtype]. [map] computes every program. A padded load is first
    copied padded into host memory, which the call holds until it returns;
    a pad that fits no operand of its load's shape is refused with
    [Shape_mismatch].

    [contract] computes every contraction in [float32], [float64],
    [complex64], [complex128] or an integer of 4 to 64 bits, of [a], [b]
    and [init] of any dtype, into a [dst] of any dtype, at any layout. Each
    element of [a], [b] and [init] enters [acc] as {!apply1}'s [Cast] casts
    it. That is exact where [acc] holds its value: [float32] holds [bool],
    [bit], the integers of at most 16 bits and the floats of at most 32;
    [float64] also [int32], [uint32] and [float64]; a complex dtype what its
    parts' dtype holds and the complex numbers of at most its width; an
    integer [acc] of [w] bits every integer modulo [2^w]. Otherwise it
    rounds once as [Cast] rounds it: an integer or a float to the nearest
    value of a float [acc], a complex number by its real part into a real
    one, a float toward zero and saturated into an integer one. An integer
    sum is exact modulo [2^w] whatever the order. A [dst] of another dtype
    than [acc] takes the sum in [acc] rounded once, as [Cast] rounds it.

    A float or complex contraction adds in the order it is given
    ({!Nx_kernel.section-orders}), at any shape.
    [contract_order ~rows ~columns] is [Chain] where [rows] and [columns]
    are both above 4 and multiply to at least 64, and [Lane] otherwise.
    With no products, an output is [init] as [acc] holds it, rounded into
    [dst], or [+0]: [init]'s bits where the three share a dtype. Otherwise
    a NaN result's sign and payload are unspecified.

    A complex output of [K] products is within
    [γ(2K + 2) (|init| + Σ|a||b|)] of the exact sum of [a], [b] and [init]
    as [acc] holds them, where [γ(n) = n u / (1 - n u)] and [u] is its
    parts' unit roundoff, barring overflow and underflow. A rounding errs
    by at most [u] times the modulus of the partial sum it rounds, both
    parts at once, and a partial sum of any of the terms has a modulus of
    at most [|init| + Σ|a||b|]: chain order rounds [2K] times. Lane order
    holds a product's two terms in neighbouring lanes until lane 0 takes
    lane 1: before, a term passes at most [⌈2K/16⌉ + 3] roundings and a
    product's two weigh at most [√2 |a||b|]; after, at most [log₂] of the
    blocks [+ 2]. Both stay within [2K + 2] roundings.

    [reduce] and [scan] compute one [Sum], [Prod], [Max] or [Min] of a
    program's one operand into its own dtype, read plain or padded, at
    [float32], [float64] and the 8- to 64-bit integers, [Max] and [Min] at
    [bool], and [Prod], [Max] and [Min] at [complex64] and [complex128];
    [reduce] also computes one [Arg] so at the dtypes [Max] computes but
    [bool], and one [Logsumexp] or [Moments] at [float32] and [float64];
    [scan] also computes [Moments] so at [float32] and [float64]. They
    answer [Wrong_dtype] where {!Nx_kernel.accepts} is [false] of a
    reduction and its output's dtype, and [Declined] for the other cases. A
    padded operand is first copied padded into host memory, which the call
    holds until it returns. They refuse with [Shape_mismatch] what
    {!Nx_kernel.Shape.padded} and {!Nx_kernel.Shape.reduce} or
    {!Nx_kernel.Shape.scan} answer [Error] for, and destinations whose
    shapes are not the ones they give.

    A float sum or product adds each output's terms in lane order
    ({!Nx_kernel.section-orders}), and a complex product multiplies them so
    by {!Nx_kernel.Prog.Mul}. Each complex multiplication errs by at most
    twice the parts' unit roundoff [u], so a product of [n] terms is within
    a normwise relative [γ(2n - 2)] of the exact one, barring overflow and
    underflow. A scan cuts each slice along its axis into chunks of 4096
    from its start. A chunk's total is its terms reduced in that order. The
    carry into the first chunk is [+0] for a sum, and a product's first
    chunk has none: its results start from the slice's first term. The
    carry into the next chunk is the carry into this one combined with this
    one's total; each result of a chunk is its carry combined with the
    chunk's terms up to it, left to right.

    [Logsumexp] and [Moments] reduce an output's terms in two passes, each
    summing in lane order.
    - [Logsumexp]: the terms' maximum [m], and [c], [m] where finite, else
      [0]; then [c + log s], [s] the sum of [exp (x - c)]. An [exp] of less
      than [-87] ([float32]) or [-708] ([float64]), the least integers
      whose [exp] is normal, is taken as [0].
    - [Moments]: the mean [c], the terms' sum over their count, then the
      variance [(Σ d² - Σ d (Σ d / n)) / n], [d = x - c], [+0] where that is
      negative, or [Σ d²] where that is [+∞]: the bits of Nx's own
      expansion of [Moments].

    For [n] terms, let [L = ⌈min (n, 1024) / 16⌉ + 3], the most roundings
    of a block's sum a term takes, [D = ⌈log₂ ⌈n / 1024⌉⌉], the depth of the
    blocks' tree, and [u] the dtype's unit roundoff. Barring overflow and
    underflow, to first order in [u], a [Logsumexp] [r] is within
    [(L + D + 5 + 6 ln n) u + u|r|] of the exact one. A [Moments] mean is
    within [E = (L + D) u M + u|μ|] of the exact [μ], [M] the mean of the
    terms' magnitudes, and its variance within
    [(L + D + 5) u V + 2 (L + D + 1) u E √V + (3L + 3D + 7) u E²] of the
    exact [V].

    [gather] and [scatter] compute every case, at every dtype, and refuse with
    [Shape_mismatch] operands and destinations whose shapes do not fit. A
    scatter's [Add] adds a target's updates left to right.

    [sort] and [assemble] compute every dtype, and [fold] every dtype
    {!Nx_kernel.Prog.Add} takes. The three refuse with [Shape_mismatch]
    operands and destinations whose shapes do not fit.

    [c2c], [r2c], [c2r] and the routines of matrices answer [Wrong_dtype]
    for dtypes {!Nx_kernel.S} does not take, and compute the others.

    The transforms refuse with [Shape_mismatch] what {!Nx_kernel.Shape}
    answers [Error] for and results of other shapes. A transform's result
    is a function of its operand's values alone, whatever its layout, its
    batch or the threads, but for a NaN result's sign and payload, which are
    unspecified. Barring underflow and overflow, a transform [ŷ] of [N]
    points, the product of the points along each of [axes], is within
    [‖ŷ - y‖₂ <= ε u √N ‖x‖₂] of the exact [y], [u] the unit roundoff of
    [x]'s parts and [‖x‖₂] its norm, a [c2r]'s that of its completed bins,
    at most [√2] times its operand's. [ε] is the sum over [axes] of [e(n)],
    [n] the points along the axis:
    - [e(n) = 8 Σ p{^3/2} + 8] over [n]'s prime factors [p], with
      multiplicity: a line computes by passes of radix 2, 3, 4, 5 and
      [n]'s other primes, or by Bailey's four steps.
    - Where a prime factor of [n] exceeds 250, [e(n) = 6 e(m) + 26], [m]
      the least power of two at least [2n - 1]: the line computes by
      Bluestein's convolution of [m] points, unless the transform of the
      conjugate chirp [e{^πi j²/n}], wrapped at [m], exceeds
      [2 √(2n - 1)] in magnitude, when it computes by passes within the
      first [e(n)].
    - The last axis of a real transform of even [n >= 512] counts
      [e(n/2) + 8] in place of [e(n)]: its lines compute through complex
      ones of [n/2] points.

    [cholesky], [lu], [qr], [solve_triangular], [eigh], [svd] and [eig]
    refuse with [Shape_mismatch] what
    {!Nx_kernel.Shape} answers [Error] for and results of other shapes. A
    matrix's results are a function of its values alone, whatever its
    batch, its layout or the threads. With [γ(k) = k u / (1 - k u)] and
    [ε(k) = k η], [u] the dtype's unit roundoff and [η] its least positive
    subnormal (2{^-149} or 2{^-1074}), [k] doubled on complex dtypes, and
    [|·|] taken elementwise, the bounds below hold. Their [ε] terms bound
    what rounding below the least normal number adds:
    - [cholesky]: [L Lᴴ = a + Δ] with
      [|Δ_ij| <= γ(n + 1) (|L| |Lᴴ|)_ij + ε(n + 1) (1 + L_jj)]. The
      [Upper] factor is the [Lower] one's conjugate transpose, bit for bit.
    - [lu]: [L U = P a + Δ] with
      [|Δ_ij| <= γ(2k) (|L| |U|)_ij + ε(2k) (1 + |U_pp|)], [p = min(i, j)]
      and [P] the rows' permutation.
    - [qr]: with [g = γ(k (2m + 10))] and [h = ε(k (2m + 10))], each column
      [a_j] has [‖a_j - q r_j‖₂ <= g ‖a_j‖₂ + √m h], and each element of
      [qᴴq - I] is at most [g] in magnitude. [q] is the product of [k]
      Householder reflectors [I - τ v vᴴ], its first [k] columns when
      reduced, whole when complete, and [r]'s diagonal is real, of either
      sign.
    - [solve_triangular]: [(op a + Δ) x = b + δ] with [|Δ| <= γ(2n) |op a|]
      and [|δ_i| <= ε(2n) (1 + |(op a)_ii|)] for each column of [b], [op a]
      the triangle read, conjugate-transposed under [adjoint], ones on its
      diagonal under [unit_diagonal].
    - [eigh]: with [g = γ(16n)] and [h = ε(16n)],
      [‖a - v diag w vᴴ‖_F <= g ‖a‖_F + h] and [‖vᴴv - I‖_F <= g]; each
      eigenvalue within [g ‖a‖₂ + h] of the exact one, with [v] or without.
    - [svd]: with [g = γ(16 max(m, n) + 100)] and
      [h = ε(16 max(m, n) + 100)], [‖a - u diag s vh‖_F <= g ‖a‖_F + h]
      over [u]'s first [k] columns and [vh]'s first [k] rows,
      [‖uᴴu - I‖_F <= g] and [‖vh vhᴴ - I‖_F <= g], reduced or complete;
      each singular value within [g ‖a‖₂ + h] of the exact one, with
      [vectors] or without. The 100 is the iteration's test of convergence
      on the bidiagonal, which drops a superdiagonal element below
      [max(10, min(100, u{^-1/8})) u] times its neighbours. Values alone
      may differ from the vectors' call's in their last bits.
    - [eig]: [a] is first balanced, [b = D⁻¹ Pᵀ a P D] with [P] a
      permutation and [D] diagonal, of powers of 2 that bring each row's
      norm near its column's; [D = I] where they are equal already, as on
      a normal matrix or one with [|a_ij| = |a_ji|]. With [g = γ(16n)]
      and [h = ε(16n)], [w] is the spectrum of [a + E] with
      [‖D⁻¹ Pᵀ E P D‖_F <= g ‖b‖_F + h], with [v] or without, and each
      column [v_j] has [|‖v_j‖₂ - 1| <= g] and
      [‖D⁻¹ Pᵀ (a v_j - w_j v_j)‖₂ <= (g ‖b‖_F + h) ‖D⁻¹ Pᵀ v_j‖₂]. On
      real [a], [w]'s values that are not real come in adjacent conjugate
      pairs, the one of positive imaginary part first, with conjugate
      columns of [v]; a real value has a [+0] imaginary part and its
      column of [v] [+0] imaginary parts. [w] is in the order the Schur
      form's diagonal leaves it, and each column of [v] has an element of
      largest magnitude real, positive on complex [a]. Values alone may
      differ from the vectors' call's. *)

include Nx_kernel.S

(**/**)

val reduce_on :
  threads:int ->
  Nx_kernel.Prog.t ->
  axes:int array ->
  (Nx_kernel.reduction * int) array ->
  dsts:Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer
(** [reduce_on ~threads] is {!reduce} on one thread where [threads] is [1],
    on as many as its job gives where it is [0]. The bits are the same. *)

val scan_on :
  threads:int ->
  Nx_kernel.Prog.t ->
  axis:int ->
  Nx_kernel.reduction * int ->
  dsts:Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer
(** [scan_on ~threads] is {!scan}, threaded as {!reduce_on}. *)

val c2c_on :
  threads:int ->
  Nx_kernel.direction ->
  axes:int array ->
  dst:(Complex.t, 's) Nx_array.t ->
  (Complex.t, 's) Nx_array.t ->
  Nx_array.answer
(** [c2c_on ~threads] is {!c2c}, threaded as {!reduce_on}. *)

val r2c_on :
  threads:int ->
  axes:int array ->
  dst:(Complex.t, 'c) Nx_array.t ->
  (float, 'r) Nx_array.t ->
  Nx_array.answer
(** [r2c_on ~threads] is {!r2c}, threaded as {!reduce_on}. *)

val c2r_on :
  threads:int ->
  axes:int array ->
  dst:(float, 'r) Nx_array.t ->
  (Complex.t, 'c) Nx_array.t ->
  Nx_array.answer
(** [c2r_on ~threads] is {!c2r}, threaded as {!reduce_on}. *)

(**/**)
