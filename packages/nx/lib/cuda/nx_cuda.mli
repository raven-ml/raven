(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels of {!Nx_kernel.S} for NVIDIA GPUs.

    [computes_on d] is [true] iff this library has a cubin for [d]'s
    architecture ({!Rig.arch}) that loads on [d]. It has one for [sm_89].
    The first call for [d], of [computes_on] or of a kernel, loads it, and
    a load that fails answers [false] from [computes_on] and [Declined]
    from a kernel, as every later call does. [computes_on d] raises
    {!Rig.Lost} if [d] is lost. Kernels claim through {!Nx_array.door}
    until their work is submitted, and answer [Done] once it is on [d]'s
    timeline; on a lost [d] they raise {!Rig.Lost}, as the door does.
    Several domains may call its kernels concurrently.

    A kernel's work on [d] is a sequence of {e launches}, each of one GPU
    function of the cubin. A kernel allocates {e words} on the OCaml heap,
    a number its operands' extents do not change, and keeps none once it
    returns but the caches it fills for later calls on [d]: for each GPU
    function, the submissions ({!Rig.Submission}) of at most 64 calls that
    launch it first, and the transforms' plans
    ({!section-transforms_memory}). The
    exceptions, whose words grow with their extents, are a contraction
    computed window by window ({!section-windows}), a matrix routine on
    large matrices ({!section-matrices_memory}), some moves
    ({!section-moves_memory}), a numbering ({!section-sorts_memory}) and
    the transforms ({!section-transforms_memory}). Beside its arrays, a
    kernel takes {e workspace}, the memory on [d] its section states, and
    keeps at most 192 MiB of it for later calls on [d]; the rest returns
    to [d] once the call's work is done. Kept memory that a later call
    replaces, by a larger buffer or another transform's plan, returns once
    collected ({!Rig.section-reclaim}), since another domain's call may
    still use it. A kernel raises {!Rig.Out_of_memory} where [d] lacks that
    memory.

    Where a kernel computes a kind of {!Nx_kernel.Prog}, it computes it
    through the C function of its name in [nx_kinds.h], with that
    function's bits but for a NaN's sign and payload, and keeps subnormal
    operands and results. The {e matrix units} are the circuits of an
    [sm_89] GPU that multiply small matrices of [bfloat16], [float16] or
    [int8] elements, through its [mma] instructions. Contractions and
    attention sum on them where their sections say.

    Each section below states, for a family of kernels, what it computes,
    how its float results associate and what bounds them, and the memory
    it takes. *)

(** {1:loop_kernels Loops}

    [map], [reduce] and [scan] compute every case {!Nx_kernel.S} offers, at
    every dtype, over plain and padded loads, scans from a carry included.
    At a narrow float, a kind computes at [float32] and rounds once to the
    narrow float, and [Fma] rounds the exact value once. A scan's
    destination may be its carry. A reduction or a scan of a program's
    output has the bits of the output mapped into an array, then reduced
    or scanned.

    A call too wide for one launch gives the same results. It takes
    workspace of at most a temporary for each node of its program, an
    element of the node's dtype at each index of the loop's axes along
    which the node's leaves vary, a load's of more than one element and a
    nonzero stride, a padded load's and a coordinate's; and an array of the
    loop's shape for each padded load it copies, each destination
    identical to a load and a scan's terms; each of these rounded up to a
    multiple of 16 bytes. *)

(** {2:reductions Sums and products}

    A float or complex [Sum] or [Prod] reduction combines each output's
    terms in {e lane order}. Their numbers
    ({!Nx_kernel.section-reductions}) fall in {e tiles} of 1024 consecutive
    numbers, and a tile's in 16 lanes by number modulo 16. Each lane adds
    its terms in increasing number from [+0], or multiplies them from its
    first. Then, for [h] = 8, 4, 2 and 1, lane [i] takes lane [i + h] for
    [i] below [h]. A lane that holds no term is left out. The tiles combine
    by a binary tree whose left subtree holds the largest power of two of
    tiles below their count, each subtree combined the same way. *)

(** {2:states Logsumexp and Moments}

    A [Logsumexp] or a [Moments] reduction of [n] terms runs two passes,
    each summing in lane order. Let [R = ⌈min (n, 1024) / 16⌉ + 3], the
    most roundings of a tile's sum a term takes, [T = ⌈log₂ ⌈n / 1024⌉⌉],
    the depth of the tiles' tree, and [u] the carrier's unit roundoff. The
    bounds below hold barring overflow and underflow, to first order in
    [u] and before the rounding into a narrower destination. *)

(** {3:logsumexp Logsumexp}

    The first pass finds the terms' maximum [m]. The second computes
    [c + log s], [c] being [m] where finite and [0] elsewhere and [s] the
    sum of [exp (x - c)], an [exp] of less than [-87] ([float32]) or [-708]
    ([float64]) taken as [0]. These thresholds are the {e cut}. The result
    is a NaN where [m] is. A result [r] is within
    [(R + T + 5 + 6 ln n) u + u|r|] of the exact one. *)

(** {3:moments Moments}

    The first pass computes the mean [c], the terms' sum over their count.
    The second computes the variance [(Σ d² - Σ d (Σ d / n)) / n],
    [d = x - c], [+0] where that is negative or [Σ d²] where that is [+∞].
    Both results are a NaN where [c] is. The mean is within
    [E = (R + T) u M + u|μ|] of the exact [μ], [M] the mean of the terms'
    magnitudes, and the variance within
    [(R + T + 5) u V + 2 (R + T + 1) u E √V + (3R + 3T + 7) u E²] of the
    exact [V]. *)

(** {2:scans Scans}

    A {e slice} is the terms along the scanned axis at one index of the
    other axes. A float or complex [Sum] or [Prod] scan not in nx.cuda's
    own order ({!section-own_order}) cuts each slice into {e chunks} of
    4096 terms from its start. A chunk's total is its terms combined in
    lane order, as a reduction of exactly them. The carry into chunk 0 is
    the slice's element of a given carry, else the identity, but a
    product's chunk 0 given none has none. The carry into chunk [c + 1] is
    the carry into chunk [c] combined with chunk [c]'s total. Each result
    of a chunk is its carry combined with the chunk's terms up to it, left
    to right. *)

(** {3:own_order nx.cuda's own order}

    A [Sum] scan whose carrier is [float32] or [float64], into a
    destination of that dtype, along the destination's innermost axis of
    more than one element, adds in nx.cuda's {e own order}. A slice falls
    in chunks of 4096 terms from its start, a chunk in 256 {e spans} of 16
    terms, and the spans in 8 {e strips} of 32. Each span adds its terms in
    order. A strip scans its spans' sums in five steps. In step [d], for
    [d] = 1, 2, 4, 8 and 16, span [r ≥ d] adds span [r - d]'s value from
    the step before ahead of its own. A strip's total is its last span's
    value, and a chunk's total its strips' totals added in order. A chunk's
    carry is the slice's carry, then the earlier chunks' totals, added in
    order. A result adds, left to right, its chunk's carry, the totals of
    the earlier strips of its chunk, the value of the previous span of its
    strip, and its span's terms up to it. *)

(** {3:logsumexp_scans Logsumexp scans}

    A [Logsumexp] scan combines states on each slice by the odd/even
    network ({!Nx_kernel.section-network}), each operation in the output's
    carrier as {!Nx_kernel.Prog} computes it. A state is a maximum [m] and
    a sum [s], and [e z] is [exp z], or [0] for [z] below the cut
    ({!section-logsumexp}). A term [x] is the state [(x, e (x - c))], [c]
    being [x] where finite and [0] elsewhere. A state [a] then a state [b]
    combine as [m = Maximum (ma, mb)] and
    [s = sa e (ma - c) + sb e (mb - c)], [c] being [m] where finite and [0]
    elsewhere. Result [i] is [m + log s], or [m] where [m] is not finite.
    With [D] as the network defines it and [u] the carrier's unit
    roundoff, where the terms up to [i] have a finite maximum, the result
    is within [(7 D + 6 ln (i + 1)) u + u|r|] of the exact [r]. *)

(** {3:moments_scans Moments scans}

    A [Moments] scan combines states as a [Logsumexp] scan does
    ({!section-logsumexp_scans}). A term [t] is the state [(t - c, +0, 1)]
    of a mean [μ], a sum [S] of squared deviations and a count [n], [c] the
    slice's first term, or [0] where that term is not finite. A state [a]
    then a state [b] combine as [n = na + nb], [δ = μb - μa], [w = nb / n],
    [μ = μa + δ w] and [S = (Sa + Sb) + (δ δ) (na w)]. Where [μa] or [μb]
    is not finite, [μ] is [μa + μb]. Result [i] is the mean [c + μ] and the
    variance [S / (i + 1)], or [μ - μ] where [μ] is not finite. For
    [i + 1] exact in the carrier, [Q] the range of the terms up to [i], [V]
    their variance, and [D] and [u] as for [Logsumexp], the mean is within
    [(4 D + 1) u Q + u|μ|] of the exact one and the variance within
    [(7 D + 1) u V + (4 D² - 2 D) u Q √V]. *)

(** {1:moves Gathers, scatters, assemblies and folds}

    [gather], [gather_ranges], [scatter], [assemble] and [fold] compute
    every case {!Nx_kernel.S} offers but some tiled adds. A {e tiled add}
    is a float [Add] without [~unique] whose axis has more than 1024
    updates, or 256 where the dtype is complex or [idx] steps 0 along the
    last axis. It declines where [⌈log₂ t⌉ + ⌈log₂ b⌉ > 64], [t] the
    elements of [into], a complex one counted twice, and [b] the tiles of
    1024 or 256 positions along the axis.

    A fold adds an index's elements from left to right, in C order of
    their taps, with no [+0] before the first, and stores [+0] at an index
    no tap reads. A narrow float's or a sub-byte dtype's fold rounds each
    addition to its dtype, as {!Nx_kernel.Prog.Add} does. *)

(** {2:move_sums Scatter sums}

    A scatter's float [Add] at a target [x] whose updates lie at positions
    [j1 < … < jn] along the axis is [x + S], rounded once to the dtype, a
    narrow float's from [float32] and a complex number's real and imaginary
    parts apart. [S] sums the updates by the complete binary tree over the
    axis's positions. Each subtree, an aligned power of two of positions,
    adds its two halves, the left first, and a half without an update is
    left out, so no [+0] enters. Each term passes through at most
    [h = min n (⌈log₂ m⌉ + 1)] additions, [m] the updates' extent along the
    axis, so the sum is within [γ(h) (|x| + Σ|ui|)] of the exact one, [γ]
    as {!Nx_kernel.section-bounds} defines it. *)

(** {2:moves_memory Memory}

    While it runs, a range gather takes 24 bytes on [d] a run and 8 for
    each 4096 runs of a row; a [Set] without [~unique] 4 bytes a
    target, 8 where the axis has [2{^32}] updates or more; a [Max] or [Min]
    of complex numbers 4 bytes a target; and a tiled add 4 bytes a target
    of [into], at most [83 + 4 w] bytes an update, [w] the bytes of an
    element, 4 for a narrow float, and 5 KiB.

    A range gather allocates more words past 4096 runs a row, and a tiled
    add's words grow with [⌈log₂ t⌉ + ⌈log₂ b⌉]. *)

(** {1:contractions Contraction}

    [contract] computes every contraction {!Nx_kernel.S} offers, of
    operands, [init] and results of any dtype through any layout, scaled or
    segmented. It declines a scale with an integer or complex accumulator.
    An element enters the accumulator, and a sum enters [dst], as
    {!Nx_kernel.Prog.Cast} stores it. A scale multiplies each element,
    rounded once into the accumulator, its value as a cast stores it, where
    an [Exponent] code [e] stands for [2{^e - 127}]. An integer sum wraps in
    the accumulator, so every association gives it the same bits, and
    [int8] operands into [int32] sum exactly. *)

(** {2:contraction_sums Float sums}

    A float contraction's association depends on the dtypes, whether it is
    scaled or segmented and the extents alone, never on a layout or a
    segmented call's ids. A box of a result computed alone may round
    otherwise. Its products are numbered as
    {!Nx_kernel.section-contraction} numbers them, and a complex sum is the
    float sums of its real and imaginary parts, both in one association.
    Its error factor ({!Nx_kernel.section-bounds}) is the one of its kind
    below, [K] an output's products, [u] the accumulator's unit roundoff.
    An output without products is as {!Nx_kernel.section-contraction}
    states, in every kind. *)

(** {2:matrix_units Matrix units}

    A call sums on the matrix units where it is unscaled and unsegmented,
    of a contracted extent [k] of at most [2{^30}], into [float32], from
    operands both [float16] or each [bfloat16] or float8. Let [m] and [n]
    be the products of [a]'s and [b]'s free extents. A call with a float8
    operand sums on them only where [m] and [n] each exceed 8. A matrix
    unit adds 8 exact products and a
    running sum by aligning them to the largest exponent among them,
    keeping 25 bits of each and truncating their sum to [float32]. *)

(** {3:matrix_ranges Ranges}

    An output's [r = ⌈k / 128⌉] spans of 128 terms go in [S] ranges, range
    [s] taking the spans from [⌊s r / S⌋] to below [⌊(s + 1) r / S⌋]:
    {ul
    {- [S = max 1 (min 16 ⌊r / 16⌋)] where [m] or [n] is at most 4, or the
       outputs, the batch elements times [m n], number at most [2{^18}];}
    {- [S = max 1 (min 16 ⌊r / 256⌋)] otherwise.}}
    The first range's sum starts from [init], the others' from [+0]. A
    range adds its products in increasing k, 8 at a time on a matrix unit.
    The ranges' sums are added in increasing k, and the result rounds once
    into [dst]. The error factor is [γ(K + 2⌈K/8⌉)]. *)

(** {2:fused Fused multiply-adds}

    Every other float sum cuts each output's [k] into at most 64 ranges of
    whole spans of 32 terms. How many follows from the call's dtypes and
    extents by a model of its speed, and may differ between the outputs of
    one call. Each range is summed from [+0] in increasing k, one fused
    multiply-add a term. The ranges' sums are added in increasing k, then
    [init], and the result rounds once into [dst]. The error factor is
    [γ(K + 1 + σ)] for a real accumulator, [σ] its scaled operands, and
    [γ(2K + 1)] for a complex one. *)

(** {2:windows Windows}

    A call computes its results window by window past [2{^30}] rows or
    columns, on the matrix units past 65,535 batch elements, and segmented
    past [2{^24}] batch elements times rows (Rows) or times the contracted
    extent (Contracting, one batch element at least). A window's
    association follows the dtypes, the call's extents and its own; on the
    matrix units its ranges are the call's. The call's words grow with the
    number of windows. A segmented window takes a few bytes of workspace
    and, for each of its batch elements, in a Rows call at most 36 bytes a
    row and 12 a group, and in a Contracting call 4 bytes a contracted
    index (8 past a contracted extent of [2{^31} - 1]), 16 a group, and at
    most 8 a group and 4096 contracted indices. The indices of no group
    count as one group more. *)

(** {2:contraction_memory Memory}

    A call copies into workspace each operand, [init] or result its kernels
    do not read or write in place. A copy holds its array's elements in the
    dtype its kernel reads, an [init]'s or a result's in the accumulator's,
    integers of fewer than 32 bits in 32, elements of fewer than 8 bits one
    a byte, its contracted extent padded to 16 bytes. It holds an operand's
    or [init]'s axes of one kind of stride 0 once, and contracted ones of
    stride 0 past 128 elements as 128 copies. A complex call's [b] is
    copied as each element's 2 x 2 real matrix [[re, -im], [im, re]], twice
    its bytes. A sum cut into ranges of k holds each range's sums of a tile
    of outputs. *)

(** {1:matrices Matrices}

    The routines compute every case {!Nx_kernel.S} offers. A matrix's
    results are a function of its values alone, whatever its batch or its
    layout. *)

(** {2:matrices_order Order of operations}

    Each element of [cholesky]'s, [lu]'s and [solve_triangular]'s results
    takes the operations of the unblocked right-looking algorithm, whatever
    the extents. Its updates come in step order (from the last row for an
    upper [op a]), each one fused multiply-add into its own value, a
    complex one two for each of its real and imaginary parts, its terms in
    the order {!Nx_kernel.section-contraction} numbers them, and its
    division last. A division by a complex number whose imaginary part is
    [0], as Cholesky's pivots are, divides the dividend's real and
    imaginary parts by the divisor's real part, and any other is [Fdiv]'s
    ({!Nx_kernel.Prog}). A Cholesky pivot is the square root of its
    element's real part. *)

(** {2:matrices_bounds Bounds}

    With [γ] as {!Nx_kernel.section-bounds} defines it, [ν(k) = k η], [u]
    the dtype's unit roundoff and [η] its least positive subnormal
    (2{^-149} or 2{^-1074}), [k] doubled on complex dtypes, and [|·|] taken
    elementwise, the bounds below hold. Their [ν] terms bound what rounding
    below the least normal number adds.
    {ul
    {- [cholesky]: [L Lᴴ = a + Δ] with
       [|Δ_ij| <= γ(n + 1) (|L| |Lᴴ|)_ij + ν(n + 1) (1 + L_jj)]. The
       [Upper] factor is the [Lower] one's conjugate transpose, bit for
       bit.}
    {- [lu]: [L U = P a + Δ] with
       [|Δ_ij| <= γ(k + 2) (|L| |U|)_ij + ν(k + 2) (1 + |U_pp|)],
       [p = min(i, j)] and [P] the rows' permutation.}
    {- [qr]: with [g = γ(k (2m + 10))] and [h = ν(k (2m + 10))], each
       column [a_j] has [‖a_j - q r_j‖₂ <= g ‖a_j‖₂ + √m h], and each
       element of [qᴴq - I] is at most [g] in magnitude. [q] is the product
       of [k] Householder reflectors [I - τ v vᴴ], its first [k] columns
       when reduced, whole when complete, and [r]'s diagonal is real, of
       either sign.}
    {- [solve_triangular]: [(op a + Δ) x = b + δ] with
       [|Δ| <= γ(n + 2) |op a|] and [|δ_i| <= ν(n + 2) (1 + |(op a)_ii|)]
       for each column of [b], [op a] the triangle read,
       conjugate-transposed under [adjoint], ones on its diagonal under
       [unit_diagonal].}} *)

(** {3:spectral_bounds Bounds of [eigh] and [svd]}

    [eigh], [svd] and [eig] iterate. A matrix on which an iteration does
    not converge within its limit is NaN in every result. Otherwise, with
    [θ(k) = k t], [t] the dtype's least normal number (2{^-126} or
    2{^-1022}) and [k] doubled on complex dtypes:
    {ul
    {- [eigh]: with [g = γ(16n)] and [h = ν(16n)],
       [‖a - v diag w vᴴ‖_F <= g ‖a‖_F + h] and [‖vᴴv - I‖_F <= g]; each
       eigenvalue within [g ‖a‖₂ + h] of the exact one with [v], and within
       [g ‖a‖₂ + θ(16n)] without.}
    {- [svd]: with [g = γ(16 max(m, n) + 100)] and
       [h = ν(16 max(m, n) + 100)], [‖a - u diag s vh‖_F <= g ‖a‖_F + h]
       over [u]'s first [k] columns and [vh]'s first [k] rows,
       [‖uᴴu - I‖_F <= g] and [‖vh vhᴴ - I‖_F <= g], reduced or complete;
       each singular value within [g ‖a‖₂ + h] of the exact one with
       [vectors], and within [g ‖a‖₂ + θ(16 max(m, n) + 100)] without.}} *)

(** {3:eig_bounds Bounds of [eig]}

    [eig] first balances [a] into [b = D⁻¹ Pᵀ a P D], [P] a permutation and
    [D] diagonal, of powers of 2 that bring each row's norm near its
    column's; [D = I] where they are equal already. On convergence, with
    [g = γ(16n)] and [h = ν(16n)], [w] is the spectrum of [a + E] with
    [‖D⁻¹ Pᵀ E P D‖_F <= g ‖b‖_F + h], with [v] or without, and each column
    [v_j] has [|‖v_j‖₂ - 1| <= g] and
    [‖D⁻¹ Pᵀ (a v_j - w_j v_j)‖₂ <= (g ‖b‖_F + h) ‖D⁻¹ Pᵀ v_j‖₂].

    [w] is in the order the Schur form's diagonal leaves it, and each
    column of [v] has an element of largest magnitude real and positive. On
    real [a], [w]'s values that are not real come in adjacent conjugate
    pairs, the one of positive imaginary part first, with conjugate columns
    of [v]; a real value has a [+0] imaginary part and its column of [v]
    [+0] imaginary parts. *)

(** {2:matrices_memory Memory}

    [solve_triangular] copies an [a] whose batch axes do not merge into one
    strided axis into workspace, and [qr] an [a] it does not read in place.
    With [c] [q]'s columns and [w] the bytes of an element, a [qr] of more
    than 32 columns (16 at [complex128]) or 1024 rows, or for which
    [(64 + n + n² + n (n - 1) / 2 + n c) w] exceeds 48 KiB, also takes
    [m b + 2 b max(n, c) + b² ⌈k / b⌉] elements a matrix, [b = min k 128]
    (64 at [complex128]). Every call takes workspace of a few elements a
    matrix and a row, and some calls a tile of each matrix. A call on large
    matrices allocates words that grow with their extents. *)

(** {3:spectral_memory Memory of [eigh], [svd] and [eig]}

    [eigh], [svd] and [eig] copy an [a] that is not C-contiguous into
    workspace. [eigh] past order 32, [svd] past 32 rows or columns, and
    [eig] take workspace of elements of [a]'s dtype, reals of its
    precision and bytes, for a matrix of order [n], or of [m] x [n] with
    [M = max(m, n)] and [c] [u]'s columns ([vh]'s rows where [m < n], [0]
    without [vectors]):
    {ul
    {- [eigh]: [n² + 65 n + 2048] elements, [3 n] reals and 16 bytes, and
       with [v] [n² + 32 n] elements, [3 n² + 75 n - 2 l (32 - l)] reals
       and [24 n + 16] bytes more, [l = n - 32 (⌈n / 32⌉ - 1)];}
    {- [svd]: [M (k + c + 33) + k² + 64 max(c, k) + 2 k + 2048] elements,
       [2 k² + 263 k + 260] reals and [4 (192 k + 203)] bytes;}
    {- [eig]: [4 n² + n] elements, [2 n² + 4 n] reals and 16 bytes.}}
    Alignment adds at most 512 bytes a matrix and 2 KiB a call. *)

(** {1:sorting Sorts and numbering}

    [sort] and [unique] compute every dtype, layout, direction and number of
    kept elements, and keys of any number of columns, within the bounds
    below. Both use only integer arithmetic, so their results are the
    contract's exact ones.

    Each declines a call that needs a launch larger than [sm_89] takes,
    which a sort reaches only past [2{^39}] elements of its operand and a
    numbering only past [2{^41}] rows. *)

(** {2:sorts_memory Memory}

    A sort takes workspace of at most 70 bytes an element of its operand,
    and a numbering at most 75 bytes a row of its keys (100 past [2{^32}]
    rows), each plus a few kilobytes. A numbering's words grow with its
    keys' columns. *)

(** {1:transforms Fourier transforms}

    [c2c], [r2c] and [c2r] compute every case {!Nx_kernel.S} offers. A
    line of [n] points is computed by a mixed-radix FFT of radices 2 to 16
    where [n] is 7-smooth, its only prime factors 2, 3, 5 and 7, and
    otherwise by Rader's convolution or by Bluestein's, of a 7-smooth
    length from [2n - 1] to [4n - 2]. A real line of even [n] is computed
    as a complex line of [n / 2] points. Each product by a twiddle or
    another constant fuses into the addition after it. A line's arithmetic
    depends on its points and its precision alone, so a result is a
    function of its operand's values, whatever its layout or batch. *)

(** {2:transforms_bounds Bounds}

    Barring underflow and overflow, and to first order in [u], a transform
    [ŷ] of [N] points, the product of the points along [axes], is within
    [‖ŷ - y‖₂ <= E u √N ‖x‖₂] of the exact [y], [u] the unit roundoff of
    [x]'s real and imaginary parts, [‖x‖₂] its norm, a [c2r]'s √2 times its
    operand's. [E] sums [e(n)] over [axes], [n] the points along the axis:
    {ul
    {- [e(n) = 5.35 log₂ n] for a 7-smooth [n];}
    {- [e(n) = 91 log₂ n + 236] for any other [n];}
    {- along the last axis of an [r2c] of even [n], [e(n/2) + 11], and of
       a [c2r] of even [n], [e(n/2) + 20].}} *)

(** {2:transforms_memory Memory}

    The plans a device keeps for later transforms take at most 64 MiB of
    the 192 MiB it keeps. A transform takes workspace of at most [3m]
    complex points a line of the axis that needs most, [m = n] for a
    7-smooth [n] and [m = 4n] otherwise, and none for lines of at most 512
    points; and a [c2r] along several axes a copy of its operand beside
    it. A transform raises [Stdlib.Out_of_memory] where the host has no
    memory for a plan. Its words grow with its axes and its lines' points.
    *)

(** {1:attention Attention}

    [attention] computes every case {!Nx_kernel.S} offers. A
    {e lead index} is an index of [q]'s [lead…] axes. A query head holds
    [tq] rows, and [G = hq / hk] query heads read each key head. Each query
    row reads its keys in {e blocks} of 64 from key 0, up to its lead
    index's length, and keeps a maximum [m], a sum [l] and an accumulator
    [a] of [dv] elements, from [-∞], [+0] and [+0].

    A call takes one of two {e paths}, by its dtypes and extents. The
    matrix units take [float16] or [bfloat16] queries into [float32] with
    [dk ≤ 128] and more than 16 rows a key head, [tq G]. Fused
    multiply-adds take every other call. *)

(** {2:attention_blocks A block}

    A pair's [x] is its output 1 on its score
    ({!Nx_kernel.section-attention}). Let [m'] be the greatest of [m] and
    the [x] of a row's visible pairs in a block, a NaN where one is. A
    block leaves the row's [m], [l] and [a] as they were where the row has
    no visible pair in it or [m'] is [-∞]. Otherwise each weight is
    [exp (x - m')], [0] at a hidden pair. [l] becomes [α l + s],
    [α = exp (m - m')] and [s] the block's weights summed term [j] into
    lane [j mod 8], then lane [i] taking lane [i + h] for [i] below [h],
    for [h] = 4, 2 and 1. [a] becomes [α a], unchanged where [α] is [1],
    then adds each weight times its key's value over the block's keys, a
    hidden pair's weight [0] included, or over the visible pairs alone
    where a hidden value of the block is a NaN or an infinity. *)

(** {2:attention_results Results}

    Once the row's parts merge ({!section-attention_merges}),
    [o = a / l + 0] in the accumulator, rounded once to [o]'s dtype, the
    [+0] turning into [+0] a [-0] that [α a] leaves. [lse = m + log l],
    [+∞] where [m] is; a row of [l = 0] has [o = +0] and [lse = -∞]. [exp]
    and [log] are {!Nx_kernel.Prog.Exp}'s and {!Nx_kernel.Prog.Log}'s. So
    a value at a pair hidden from a row, finite or not, never reaches its
    [o]. *)

(** {2:attention_sums Sums of each path}

    On fused multiply-adds, [q·k] adds the [dk] products of the row's and
    the key's elements, each widened to the accumulator, in increasing
    dimension from [+0], and [a] adds the weights times the values in
    increasing key. Each product fuses into its sum.

    On the matrix units, [q·k] adds its products from [+0] in increasing
    dimension, 8 at a time, as a matrix unit sums them
    ({!section-matrix_units}). Each weight rounds once to nearest into the
    values' dtype ([c = 1], {!Nx_kernel.section-attention}), and [a] adds
    the value products in increasing key on the matrix units as [q·k] does,
    but a block whose values hold a NaN or an infinity adds them as fused
    multiply-adds of the unrounded weights, over each row's visible pairs
    alone. *)

(** {2:attention_parts Parts}

    Let [R] be 128, or 64 with a [float64] accumulator. A {e query tile}
    holds, at one lead index, [c = min G (max 1 ⌊R / min tq R⌋)] of the
    query heads that read one key head and [min tq ⌊R / c⌋] rows of each.
    Let [g] be the query tiles times the pieces of [R] columns of [o], [b]
    the blocks of [tk] keys, and [P' = min (min ⌈256 / g⌉ ⌊b / 4⌋) 65535].
    Where [g < 256], [b ≥ 8] and [P' ≥ 2], a call's keys go in
    [P = ⌈b / ⌈b / P'⌉⌉] {e parts} of [⌈b / P'⌉] blocks, and otherwise in
    one. Each part keeps its own [m], [l] and [a]. *)

(** {3:attention_merges Merges}

    A row's parts merge in order. With [m'] the greater of [m] and the
    part's [mₚ], [α = exp (m - m')] and [β = exp (mₚ - m')], [l] becomes
    [α l + β lₚ] and [a] becomes [β aₚ + α a], one fused multiply-add,
    [α a] unchanged where [α] is [1]; where [m'] is [-∞], [a] becomes
    [a + aₚ]. *)

(** {2:attention_program The pair program}

    The pair program's outputs have the bits [map] gives them. A call reads
    no key or value of a block whose pairs output 0 hides from each row of
    a query tile, where {!Nx_kernel.Prog.ranges} shows it from the loads'
    least and greatest elements over the block. *)

(** {3:attention_nodes Nodes}

    The kernel computes a program's nodes of these kinds: in [float32] and
    [float64], [Neg], [Add], [Sub], [Mul], [Fdiv], [Maximum], [Minimum],
    [Exp], [Log], [Tanh] and comparisons; in [int64], [Neg], [Abs],
    [Sign], [Add], [Sub], [Mul], [Idiv], [Mod], [Maximum], [Minimum],
    [And], [Or], [Xor] and comparisons; in [uint32], [bool] and [bit],
    [And], [Or], [Xor], [Maximum], [Minimum] and comparisons; comparisons
    of [uint16], [uint8] and [uint4]; [Where] of a dtype of 64 bits or
    fewer; casts among [float32], [float64], [int64] and [bool]; and the
    nodes that compute nothing, as [Copy]. It holds the program's values,
    beside the path's tiles, in at most 99 KiB of [sm_89]'s shared memory,
    and its instructions and the call's arrays in a launch's 4 KiB of
    parameters. *)

(** {3:attention_maps Split programs}

    A program with a node of another kind, or too large for that memory,
    is split. [map] first computes some of its nodes into temporaries on
    [d], which the call then reads as loads: the first of these that
    leaves a program the kernel holds. They are output 0; the nodes of
    output 1 that do not read the score but feed one that does; both; and
    output 1 itself, from the scores that a first launch of the call's path
    stores, with output 0 as it is or as a temporary. A temporary holds its
    node's values exactly, so the results are those of the whole program.
    *)

(** {2:attention_bounds Bounds}

    Let [p(k) = ⌈k / 256⌉ - 1], which bounds the merges of a row's parts
    over [k] keys, as a part holds at least 256 keys, and [u] the
    accumulator's unit roundoff. The error factor
    ({!Nx_kernel.section-bounds}) of [q·k] and of [o]'s sum is
    [γ(k + 1 + p(k))] for [float32] and [float64] queries and
    [γ(k + 2⌈k / 8⌉ + p(k))] for [float16] and [bfloat16] ones.

    For a row that reads [t] keys in [n = ⌈t / 64⌉] blocks, the [ε] of
    {!Nx_kernel.section-attention} is
    [e{^λu} (13 (n + 2p(t)) + 9 + ln t) u' + (e{^λu} - 1) ln t + u' |lse|],
    [u' = u / (1 - u)] and [λ] being 3 in [float32] and 5 in [float64]. *)

(** {2:attention_memory Memory}

    A call whose keys go in [P > 1] parts takes [P r (dv + 2)] accumulator
    elements on [d], [r] the rows of [q]. A split program
    ({!section-attention_maps}) takes a temporary for each node [map]
    computes, an element of the node's dtype at each index of the loop's
    axes along which the node's leaves vary, a load's of more than one
    element and a stride and a coordinate's, and what [map] takes to
    compute them; and output 1 from the scores takes an accumulator element
    for each pair, [r tk]. *)

include Nx_kernel.S

(**/**)

val contract_wide_k : int -> unit
(** [contract_wide_k n] makes a Contracting call hold its contracted
    indices in 8 bytes past a contracted extent of [n], at least 0 and at
    most [2{^31} - 1], so that a test reaches them with few indices. *)

val numbering_words : int -> unit
(** [numbering_words n] makes a numbering's submission sort at most [n]
    columns of its keys, at least one and at most 8, so that a test
    reaches several submissions with few columns. *)
