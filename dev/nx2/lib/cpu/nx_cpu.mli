(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for host memory.

    [computes_on d] is {!Rig.reaches}[ Rig.host d]: the host addresses [d]'s
    memory, which the kernels compute on. Each kernel computes from
    operands of [dst]'s shape and any layouts, on memory the host addresses. It
    claims every operand through the door of [nx_array.h] for the extent of its
    call. It answers [Done] once its work is done, or, before any write, a
    refusal: the door's, or [Shape_mismatch] if the operands' shapes differ.
    It never answers [Declined]: nx.cpu computes every case its contract
    offers, and a kernel that would decline one, a bug in nx.cpu, raises
    [Failure] naming itself and its operands' dtypes and shapes.

    A call keeps other domains waiting to collect for no longer than about
    one unloaded core's copy of 1 MiB. It holds the domain lock while it
    reads its operands, then runs its work with the lock released where
    nx.cpu estimates that work to take longer than that copy; the estimate
    is an upper bound of the work's time on one core of the machines it was
    measured on, an Apple M1 Max and an Intel Core Ultra 5 235.

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
    ({!Nx_kernel.section-orders}), at any shape, so its parts compose across
    windows of the contracting axes as both orders state there.
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
    program's one operand, read plain or padded, at every dtype, into a
    destination of any dtype. Terms of a float narrower than [float32]
    reduce in [float32], of [int4] and [uint4] in [int8] and [uint8], and
    of [bit] in [bool], their carrier, each block of them converted into it
    as a cast does; others in their own dtype. Each result rounds once
    into its destination's dtype, as a cast does, which keeps a NaN's sign
    and payload: into the operand's own dtype, a NaN result where a term is
    NaN is the first such term quieted. They also compute one [Arg] so, its
    positions into [int64] and its extreme into the operand's own dtype the
    term itself, quieted where a NaN, and one [Logsumexp] so at every
    float dtype; [reduce] one [Moments] so, and [scan] one [Moments] so, its
    operand converted whole into its carrier first where that is not its
    dtype, or a program's tile by tile. Both take any program [map]
    computes, over its loads or, with none, over the loop's shape alone, a
    reduction's [shape] or a scan's destinations': its terms are the
    program's output there as [map] computes it, so a reduction or a scan
    of an output has the bits of the output mapped into an array, then
    reduced or scanned. [reduce] computes several such reductions of one
    loop, each one's bits alone, every destination checked before any is
    written, any number of them. They answer [Wrong_dtype] where
    {!Nx_kernel.accepts} is [false] of a reduction and its output's dtype
    or an [Arg]'s positions are not [int64], and [Bad_arity] for
    destinations other than a reduction's one or two. A padded operand is
    first copied padded into host memory, which the call holds until it
    returns. A destination identical to the operand, or to a program's
    load, is accepted: that operand or load is then read from a copy the
    call holds, and the results are those of fresh destinations. They
    refuse with [Shape_mismatch] what {!Nx_kernel.Shape.padded} and
    {!Nx_kernel.Shape.reduce} or {!Nx_kernel.Shape.scan} answer [Error]
    for, a reduction's loads whose shape once loaded is not its [shape],
    and destinations whose shapes are not the ones they give.

    A float sum or product adds each output's terms in lane order
    ({!Nx_kernel.section-orders}), a complex sum each part's as a float
    sum, and a complex product multiplies them so by
    {!Nx_kernel.Prog.Mul}. Each complex multiplication errs by at most
    twice the parts' unit roundoff [u], so a product of [n] terms is within
    a normwise relative [γ(2n - 2)] of the exact one, barring overflow and
    underflow. Every monoid's scan cuts its chunks and carries as
    {!Nx_kernel.type-reduction} states a float sum's, from a carry where
    one is given. A scan by [Arg] or [Logsumexp] cuts the same chunks of
    4096 and carries a state: a chunk's total is the state
    of its terms taken left to right from none, the carry into the next
    chunk the carry merged with it, and each result its carry's state taken
    on through the chunk's terms up to it. [Logsumexp]'s state is the
    terms' maximum [m] and [s], the sum of [exp (x - m)]: a term [x] makes
    the maximum [n], and [s] becomes [s exp (m - n) + exp (x - n)], [s]
    itself where [n] is [m]; a merge rescales each side's [s] so. The
    result is [n + log s], or [n] where [n] is not finite, an [exp] taken
    as [0] as the reduction takes it.

    [Logsumexp] and [Moments] reduce an output's terms in the two passes
    {!Nx_kernel.reduction} states, each summing in lane order: the bits of
    Nx's own expansion of each.

    For [n] terms, let [L = ⌈min (n, 1024) / 16⌉ + 3], the most roundings
    of a block's sum a term takes, [D = ⌈log₂ ⌈n / 1024⌉⌉], the depth of the
    blocks' tree, and [u] the carrier's unit roundoff; the rounding into a
    narrower destination adds its own. Barring overflow and underflow, to
    first order in [u], a [Logsumexp] [r] is within
    [(L + D + 5 + 6 ln n) u + u|r|] of the exact one. A [Logsumexp] scan's
    result [r] after [N] terms in [C] chunks, where [K] counts the terms,
    other than a chunk's first, greater than every earlier term of their
    chunk, is within
    [(min (N, 8192) + C + 6 (min (K, 8192) + C) + 5 + 6 ln N) u + u|r|] of
    the exact one.
    A [Moments] mean is within [E = (L + D) u M + u|μ|] of the exact [μ],
    [M] the mean of the terms' magnitudes, and its variance within
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
    unspecified. Where its result cannot hold the work, as for an axis after
    the first, lines whose points lie apart and Bluestein's convolutions, a
    transform works in host memory the call holds until it returns. Barring
    underflow and overflow, a transform [ŷ] of [N] points, the product of the
    points along each of [axes], is within [‖ŷ - y‖₂ <= ε u √N ‖x‖₂] of the
    exact [y], [u] the unit roundoff of [x]'s parts and [‖x‖₂] its norm, a
    [c2r]'s that of its completed bins, at most [√2] times its operand's.
    [ε] is the sum over [axes] of [e(n)], [n] the points along the axis:
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
    batch, its layout or the threads. A destination of [solve_triangular]
    identical to [a] or [b] is accepted: they are then read from copies the
    call holds, and the results are those of a fresh destination. With [γ(k) = k u / (1 - k u)] and
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
  shape:int array ->
  axes:int array ->
  (Nx_kernel.reduction * int) array ->
  dsts:Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer
(** [reduce_on ~threads] is {!reduce} on one thread where [threads] is [1],
    on as many as its job gives where it is [0]. The bits are the same. *)

val scan_on :
  threads:int ->
  ?carry:Nx_array.any ->
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
