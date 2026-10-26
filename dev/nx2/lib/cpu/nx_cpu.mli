(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for host memory.

    [computes_on d] is {!Rig.reaches}[ Rig.host d], true where the host
    addresses [d]'s memory, which the kernels compute on. Each kernel
    computes from operands of any layouts, on memory the host addresses. It
    claims every operand through the door of [nx_array.h] for the extent of
    its call. It answers [Done] once its work is done, or a refusal before
    any write. It answers [Declined] only where its doc says, and computes
    every other case its contract offers.

    A call keeps other domains waiting to collect for no longer than about
    one unloaded core's copy of 1 MiB at a time, however many operands,
    reductions or program nodes it has
    ({!Nx_array.section-kernels}). It runs its work with the domain lock
    released where nx.cpu estimates that work to take longer than that copy;
    the estimate is an upper bound of the work's time on one core of the
    machines it was measured on, an Apple M1 Max and an Intel Core Ultra 5
    235. Other threads of the calling domain may run OCaml code during a
    call.

    A kernel's result is a function of its operands' values alone, in the
    sense that neither layouts nor threads change a bit of it. An element's
    association, its algorithm and whether the kernel declines depend on
    the extents the element reads whole (a reduction's, a scan's, a sort's
    or a transform's axes, the contracted extent, a routine's matrices, a
    gather's or a scatter's axis, the axes a padded load or a fold pads or
    windows), the dtypes, the attributes and the host's architecture, never
    on the result's other extents; a contraction's association follows its
    rows, columns and depth, as stated below. Each kind computes through
    the function of its name in [nx_kinds.h], with that function's bits,
    NaNs aside, on every target. So [Exp2] is exact at integers whose power
    is in the dtype, and [Log2] at powers of two. The kinds keep subnormal
    operands and results.

    {e Lane order} puts the terms of a sum or a product, numbered as
    {!Nx_kernel.section-reductions} and {!Nx_kernel.section-contraction}
    number them, into blocks of 1024 consecutive numbers and, within a
    block, into 16 lanes by number modulo 16, each lane adding its terms in
    increasing number from [+0], or multiplying them from its first. Lane
    [i] takes lane [i + 8] for [i] below 8, then lane [i + 4] for [i] below
    4, then [i + 2] for [i] below 2, then lane 0 takes lane 1, a lane
    holding no term taking no part. The blocks are combined by a binary
    tree whose left part holds the largest power of two of blocks below
    their count, each part combined the same way.

    [map] computes every program. A padded load is first copied padded
    into host memory, which the call holds until it returns, once along an
    axis its operand broadcasts and the pad leaves alone; a pad that fits
    no operand of its load's shape is refused with [Shape_mismatch].

    [contract] computes every contraction in [float32], [float64],
    [complex64], [complex128] or an integer of 4 to 64 bits, of [a], [b]
    and [init] of any dtype, into a [dst] of any dtype, at any layout, with
    or without segments, scaled in [float32] and [float64] alone, and
    declines the others. Each
    element of [a], [b] and [init] enters [acc] as [Cast] casts
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
    An element of a scaled operand enters [acc] as its [Cast] times its
    scale block's scale, the scale read into [acc] ([Cast] of a [Value],
    [2^(e - 127)] of a code [e], NaN of [255]), the product rounded once. So
    a scale multiplies each element, rounded once into [acc], at any contracted
    extent. Each scaled operand adds one unit roundoff of [acc] per term
    to the bound, barring overflow of an element times its scale, which
    gives an infinite term where the exact one is finite.

    A float or complex contraction adds its products, numbered as
    {!Nx_kernel.section-contraction} numbers them, in {e chain order}, one
    chain in increasing number from [init] or [+0], where its rows, [a]'s
    free extents multiplied, and its columns, [b]'s, are both above 4 and
    multiply to at least 64, and its depth, the contracted extent, is below
    128 times their product. It adds them in lane order, then [init],
    otherwise, so that the
    threads share the depth of a contraction of few outputs. A segmented
    contraction chooses one order for the whole call, from its rows,
    columns and depth as above, [Rows]' columns those of one index along
    [b_axis], [Contracting]'s depth the pair's whole extent; each output
    adds its group's products in that order. A segmented contraction
    reads the offsets whole before any write, so they may be [dst]
    itself.

    A complex output of [K] products is within
    [γ(2K + 2) (|init| + Σ|a||b|)] of the exact sum of [a], [b] and [init]
    as [acc] holds them, with [γ] as {!Nx_kernel.section-reductions}
    defines it and [u] its parts' unit roundoff, barring overflow and
    underflow. A rounding errs by at most [u] times the modulus of the
    partial sum it rounds, both parts at once, and a partial sum of any of
    the terms has a modulus of at most [|init| + Σ|a||b|]. Chain order
    rounds [2K] times. Lane order holds a product's two terms in
    neighbouring lanes until lane 0 takes lane 1. Until then, a term passes
    at most [⌈2K/16⌉ + 3] roundings and a product's two weigh at most
    [√2 |a||b|]; after, at most [log₂] of the blocks [+ 2]. Both stay
    within [2K + 2] roundings.

    [reduce] and [scan] compute one [Sum], [Prod], [Max] or [Min] of a
    program's one operand, read plain or padded, at every dtype, into a
    destination of any dtype. Terms reduce in their carrier
    ({!Nx_kernel.carrier}), each block of them converted into it as a cast
    does where it is not their dtype. Each result rounds once into its
    destination's dtype, as a cast does. They also compute one [Arg] so,
    its positions into [int64] and its extreme into the operand's own dtype
    the term itself, and one [Logsumexp] and one [Moments]
    so at every float dtype, the operand of a scan by either converted
    whole into its carrier first where that is not its dtype, or a
    program's tile by tile. Both take any program [map] computes, over its
    loads or, with none, over the loop's shape alone, a reduction's [shape]
    or a scan's destinations': its terms are the
    program's output there as [map] computes it, so a reduction or a scan
    of an output has the bits of the output mapped into an array, then
    reduced or scanned. [reduce] computes several such reductions of one
    loop, each one's bits alone, every destination checked before any is
    written, any number of them. They answer [Wrong_dtype] where
    {!Nx_kernel.accepts} is [false] of a reduction and its output's dtype
    or an [Arg]'s positions are not [int64], and [Shape_mismatch] for
    destinations other than a reduction's one or two. A padded operand is
    first copied padded into host memory, which the call holds until it
    returns. A destination identical to the operand, or to a program's
    load, is accepted: that operand or load is then read from a copy the
    call holds, and the results are those of fresh destinations. They
    refuse with [Shape_mismatch] what {!Nx_kernel.Shape.padded} and
    {!Nx_kernel.Shape.reduce} or {!Nx_kernel.Shape.scan} answer [Error]
    for, a reduction's loads whose shape once loaded is not its [shape],
    and destinations whose shapes are not the ones they give.

    A float or complex [Sum] or [Prod] reduction adds or multiplies each
    output's terms in lane order, a complex [Sum] part by part and a complex
    [Prod] by {!Nx_kernel.Prog.Mul}. A float or complex [Sum] or [Prod] scan
    cuts each slice along its axis into {e chunks} of 4096 terms from its
    start. A chunk's total is its terms reduced in lane order, as a reduction
    of exactly them. The carry into chunk 0 is the slice's element of a given
    carry, else the identity, but a product's chunk 0 given none has none, and
    its first result is its first term. The carry into chunk [c + 1] is the
    carry into chunk [c] combined with [c]'s total, and each result of a chunk
    is its carry combined with the chunk's terms up to it, left to right. So
    the scan of a slice's terms from a chunk's start, with the carry into that
    chunk, has the bits of the scan of the whole slice from that chunk on.

    A [Logsumexp] or a [Moments] reduction runs two passes, each summing in
    lane order on nx.cpu's own [Max], [Sum] and kinds. [Logsumexp]'s first
    finds the terms' maximum [m]. Its second computes [c + log s], [c]
    being [m] where finite and [0] elsewhere and [s] the sum of
    [exp (x - c)], an [exp] of less than [-87] ([float32]) or [-708]
    ([float64]), the least integers whose [exp] is normal, taken as [0].
    These thresholds are the {e cut}. The result is a NaN where [m] is.
    [Moments]' first pass computes the mean [c], the terms' sum over their
    count. Its second computes the variance [(Σ d² - Σ d (Σ d / n)) / n],
    [d = x - c], [+0] where that is negative or [Σ d²] where that is [+∞].
    Both are a NaN where [c] is.

    For a [Logsumexp] or [Moments] reduction of [n] terms, let
    [L = ⌈min (n, 1024) / 16⌉ + 3], the most roundings of a block's sum a
    term takes, [D = ⌈log₂ ⌈n / 1024⌉⌉], the depth of the blocks' tree,
    and [u] the carrier's unit roundoff; the rounding into a narrower
    destination adds its own. Barring overflow and
    underflow, to first order in [u], a [Logsumexp] [r] is within
    [(L + D + 5 + 6 ln n) u + u|r|] of the exact one. A [Moments] mean is
    within [E = (L + D) u M + u|μ|] of the exact [μ], [M] the mean of the
    terms' magnitudes, and its variance within
    [(L + D + 5) u V + 2 (L + D + 1) u E √V + (3L + 3D + 7) u E²] of the
    exact [V].

    A [Logsumexp] or a [Moments] scan combines its states on each slice by
    the odd/even network ({!Nx_kernel.section-scans}), each operation in
    the output's carrier as {!Nx_kernel.Prog} computes it, with these
    states and bounds, [D] the network's:
    - [Logsumexp]: a state is a maximum [m] and a sum [s], and [e z] is
      [exp z], or [0] for [z] below the cut. A term [x] is the state
      [(x, e (x - c))], [c] being [x] where finite and [0] elsewhere, and a
      state [a] then a state [b] combine as [m = Maximum (ma, mb)] and
      [s = sa e (ma - c) + sb e (mb - c)], [c] being [m] where finite and
      [0] elsewhere. Result [i] is [m + log s], or [m] where [m] is not
      finite. Where the terms up to [i] have a finite maximum, the result
      is within [(7 D + 6 ln (i + 1)) u + u|r|] of the exact [r],
      {!Nx_kernel.Prog.Exp} and {!Nx_kernel.Prog.Log} being within 2 ulps.
    - [Moments]: a term [t] is the state [(t - c, +0, 1)] of a mean [μ], a
      sum [S] of squared deviations and a count [n], [c] the slice's first
      term, or [0] where that term is not finite. A state [a] then a state
      [b] combine as [n = na + nb], [δ = μb - μa], [w = nb / n],
      [μ = μa + δ w] and [S = (Sa + Sb) + (δ δ) (na w)]. Where [μa] or [μb]
      is not finite, [μ] is [μa + μb]. Result [i] is the mean [c + μ] and
      the variance [S / (i + 1)], or [μ - μ] where [μ] is not finite. For
      [i + 1] exact in the carrier, [R] the range of the terms up to [i]
      and [V] their variance, the mean [μ] is within [(4 D + 1) u R + u|μ|]
      of the exact one and the variance within
      [(7 D + 1) u V + (4 D² - 2 D) u R √V].

    [gather] and [scatter] compute every case, at every dtype, and refuse with
    [Shape_mismatch] operands and destinations whose shapes do not fit. A
    scatter's [Add] adds a target's updates left to right,
    [(… ((+0 + x) + u1) + …) + un]. A narrow float's keeps its float32 sums
    and the positions it sorts by target in host memory the call holds
    until it returns. A {e slice} is the updates at one index off the
    scattered axis, with their positions and targets; a {e row of slices}
    is the slices at successive indices of the last axis off it, joined
    with the axes before it where every operand steps through them as
    one. Over [m] updates along the scattered axis, the Add takes 12 KiB
    and [32 m] bytes on each of its threads; in place of the [32 m],
    [40 m + 8] bytes once where its positions step 0 along each of fewer
    rows of slices than threads, and otherwise, where its slices are fewer
    than its threads and its dtype is [float4_e2m1fn], 32 bytes for each
    update in all. Where its slices are fewer than its threads, each the
    only slice of its row or one whose positions differ along its row, and
    its dtype of [w] bytes, [w ≥ 1], it sorts a slice's updates by
    {e bin}, a run of [2{^k}] successive targets along the axis, at most
    1024 bins in all, a block of updates at a time: all [m] where the axis
    has [m] targets or more, else at most 2{^22}. It takes instead
    [8 + w] bytes for each update of a block, and on each thread
    [16 (8 + w) + 8] bytes for each bin and a bin's sums, a byte of flag
    and a float32 sum for each of its targets, the flags padded to 64
    bytes: at most half a core's L2 of sums, or those of a 512th of the
    axis's targets. Along an axis of more than 2{^19} updates and more
    updates than targets, every bin keeps its own sums, about five bytes
    for each target of its bins, the last counted whole.

    [sort] and [assemble] compute every dtype, and [fold] every dtype
    {!Nx_kernel.Prog.Add} takes. The three refuse with [Shape_mismatch]
    operands and destinations whose shapes do not fit. A fold adds an
    index's elements to [+0] from left to right, in C order of their taps.

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
    call holds, and the results are those of a fresh destination. With [γ]
    as {!Nx_kernel.section-reductions} defines it, [ε(k) = k η], [u] the
    dtype's unit roundoff and [η] its least positive subnormal (2{^-149} or
    2{^-1074}), [k] doubled on complex dtypes, and [|·|] taken elementwise,
    the bounds below hold. Their [ε] terms bound what rounding below the
    least normal number adds.
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

val one_thread : (unit -> 'a) -> 'a
(** [one_thread f] is [f ()], every call of the calling thread running its
    work on that thread alone while [f] runs, nested calls included. The
    bits are those of any threads. For tests. *)

(**/**)
