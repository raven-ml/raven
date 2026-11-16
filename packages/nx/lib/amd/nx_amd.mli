(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels of {!Nx_kernel.S} for AMD GPUs of RDNA4.

    [computes_on d] is [true] iff this library has a code object for [d]'s
    processor ({!Rig.arch}) that loads on [d], with each of its kernels.
    It has one for [gfx12-generic], which loads on [gfx1200] and
    [gfx1201]; its kernels have run on [gfx1201] alone.
    The first call for [d], of [computes_on] or of a kernel, loads it, and
    a load that fails answers [false] from [computes_on] and [Declined]
    from a kernel, as every later call does. [computes_on d] raises
    {!Rig.Lost} if [d] is lost. Kernels claim through {!Nx_array.door}
    until their work is submitted, and answer [Done] once it is on [d]'s
    timeline; on a lost [d] they raise {!Rig.Lost}, as the door does.
    Several domains may call its kernels concurrently.

    A kernel's work on [d] is one submission ({!Rig.Submission}) of
    {e launches}, each of one GPU function of the code object, which run
    in order. No function uses scratch memory, so the library grows no
    device's scratch. A kernel allocates {e words} on the OCaml heap, a
    number its operands' extents do not change unless its section states
    that they grow with them, and keeps none once it returns but the
    transforms' line plans ({!section-transforms_memory}) and the
    submissions it keeps for later calls on [d], at most 64 MiB of them as
    it counts them: 1 KiB a call, and [32 (9 + ⌈w / 2⌉ + r)] bytes for each
    launch of a GPU function of [w] words, at most 528, that names [r]
    buffers, so at most 8,736 bytes and 32 a buffer. Past 64 MiB, and past
    64 calls that launch one GPU function first, [d] drops the submissions
    least recently used. A call of more than 64 MiB takes a submission of
    its own, which no later call reuses. A submission also holds, left
    out of the count and for as long as [d] keeps it, [32 (1 + ⌈w / 2⌉)]
    bytes of each launch for each call beyond the first that ever wrote
    it at once, so that the submissions [d] keeps hold at most [n] times
    64 MiB, [n] the most calls ever on [d] at once. A section states where
    a call's launches, and so its words, grow with its extents. Beside its
    arrays, a kernel takes {e workspace}, memory on [d] for its
    temporaries, and {e tickets}, zero 32-bit words its groups count on,
    of the sizes its section states. It
    keeps at most 64 MiB of workspace and 4 MiB of tickets for later calls
    on [d], and a call that needs more takes buffers of its own, which
    return to [d] once its work is done. A kernel whose section names
    {e constants}, tables that depend only on a call's dtypes and extents,
    takes them from [d]'s memory: [d] keeps the 16 most recently made, at
    most 64 MiB of them together, for later calls, and a call that finds a
    constant missing uploads it once nothing refuses the call and waits
    for the upload. A constant of more than 64 MiB is uploaded for its
    call alone and returns to [d] once the call's work is done.
    Kept memory that a later call replaces, by a larger buffer or by its
    own constants, returns once collected ({!Rig.section-reclaim}), since
    another domain's call may still use it. A kernel raises
    {!Rig.Out_of_memory} where [d] lacks that memory.

    Each section below states, for a family of kernels, what it computes,
    how its float results associate and what bounds them, and the memory
    it takes. [attention], of no section, declines every call. *)

(** {1:loop_kernels Loops}

    [map], [reduce] and [scan] compute every case {!Nx_kernel.S} offers, at
    every dtype, over plain and padded loads, scans from a carry included.
    Where they compute a kind of {!Nx_kernel.Prog}, they compute it through
    the C function of its name in [nx_kinds.h], with that function's bits
    but for a NaN's sign and payload, and keep subnormal operands and
    results. At a narrow float, a kind computes at [float32] and rounds once
    to the narrow float, and [Fma] rounds the exact value once. A scan's
    destination may be its carry. A reduction or a scan of a program's
    output has the bits of the output mapped into an array, then reduced
    or scanned.

    A call too wide for one launch gives the same results. It takes
    workspace of at most a temporary for each node of its program, an
    element of the node's dtype at each index of the loop's axes along
    which the node's leaves vary, a load's of more than one element and a
    nonzero stride, a padded load's and a coordinate's; and an array of the
    loop's shape for each padded load it copies and each destination
    identical to a load; each of these rounded up to a multiple of 16
    bytes and placed on 256. *)

(** {2:reductions Sums and products}

    A float or complex [Sum] or [Prod] reduction combines each output's
    terms in {e lane order}. Their numbers
    ({!Nx_kernel.section-reductions}) fall in {e tiles} of 1024 consecutive
    numbers, and a tile's in 16 lanes by number modulo 16. Each lane adds
    its terms in increasing number from [+0], or multiplies them from its
    first. Then, for [h] = 8, 4, 2 and 1, lane [i] takes lane [i + h] for
    [i] below [h]. A lane that holds no term is left out. The tiles combine
    by a binary tree whose left subtree holds the largest power of two of
    tiles below their count, each subtree combined the same way. These are
    nx.cpu's bits. *)

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
    other axes. A float or complex [Sum] or [Prod] scan cuts each slice
    into {e chunks} of 4096 terms from its start. A chunk's total is its
    terms combined in lane order, as a reduction of exactly them. The carry
    into chunk 0 is the slice's element of a given carry, else the
    identity, but a product's chunk 0 given none has none. The carry into
    chunk [c + 1] is the carry into chunk [c] combined with chunk [c]'s
    total. Each result of a chunk is its carry combined with the chunk's
    terms up to it, left to right. These are nx.cpu's bits. *)

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

(** {2:loops_memory Memory}

    Beside a too wide call's temporaries, a reduction of [t] terms an
    output takes, where [m = ⌈t / 16384⌉] exceeds 1 or its outputs hold
    fewer than 256 tiles in all and an output more than one, at most the
    larger of [32 m] and [8 ⌈t / 1024⌉] bytes of workspace for each
    output, and a ticket for each output; a [Logsumexp] or a [Moments]
    reduction 8 bytes an output more. A reduction of one output of 2 to
    16 tiles may thus take workspace and a ticket. A [Logsumexp] or a
    [Moments] scan takes no workspace; another scan of slices of more than
    4096 terms at most 32 bytes a chunk. *)

(** {1:gathers Gathers, scatters, assemblies and folds}

    [gather], [gather_ranges], [scatter], [assemble] and [fold] compute
    every case {!Nx_kernel.S} offers.

    A fold adds an index's elements from left to right, in C order of
    their taps, with no [+0] before the first, and stores [+0] at an index
    no tap reads. A narrow float's or a sub-byte dtype's fold rounds each
    addition to its dtype, as {!Nx_kernel.Prog.Add} does.

    An assembly's launches grow with its pieces: one fills [dst] where its
    pieces may not cover it, and one copies each [min 16 ⌊448 / (7 + 5 r)⌋]
    pieces with an element, [r] the rank of [dst]. *)

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
    as {!Nx_kernel.section-bounds} defines it.

    A float [Add] with [~unique] adds each update to its target in one
    launch. Without it, a {e tile} is 1024 positions along the axis, or
    256 where the dtype is complex or where the axis is not the last and
    [idx] steps 0 along the last axis, which has more than one update. An
    axis of at most a tile sums each target in one launch. A longer one
    sorts its entries by target, then sums each target's run of entries by
    the tree, a launch for each 8 bits of the entries' positions: the
    tiles' sums of each of their targets, each at its tile's index along
    the axis, where the axis has at most a tile's targets, so that few
    targets take few entries, and otherwise the updates, each at its
    position. *)

(** {2:moves_memory Memory}

    While it runs, a range gather takes 24 bytes of workspace a run and 8
    for each 4096 runs of a row; a [Set] without [~unique] 8 bytes a
    target for an element of 1, 2 or 4 bytes and 4 for others, 8 where the
    axis has [2{^32}] updates or more; a [Max] or [Min]
    of complex numbers 8 bytes a target for complex64, 16 for complex128;
    and a float [Add] past one tile at most [64 + 4 w] bytes an update,
    [w] the bytes of an element, 4 for a narrow float, 3 KiB and a ticket.
    No other takes a ticket.

    A range gather allocates more words past 4096 runs a row, and a float
    [Add] past one tile's words grow with [⌈log₂ t⌉ + ⌈log₂ m⌉], [t] the
    elements of [into] and [m] the axis's updates. *)

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

    The transforms keep, on the OCaml heap, the plans of the 64 lines the
    process last planned, by points, precision and whether a line is a
    halved real one, for every device; a plan holds no table.

    A transform's constants are its lines' tables of twiddles and of
    Rader's or Bluestein's kernel, one constant for each length and
    precision along its axes, of at most [9n] complex points and 3 KiB for
    a line of [n] points. A constant is made on the host, Bluestein's
    kernel by an FFT of [m] points in double-double arithmetic, in time
    that grows as [m log m]: seconds for [m] of millions. A constant of
    more than 64 MiB, which no device keeps, is made again for each call.
    A transform takes workspace of at most [3m] complex points a line of
    the axis that needs most, [m = n] for a 7-smooth [n] and [m = 4n]
    otherwise, and none for lines of at most 512 points; a [c2r] along
    several axes, a copy of its operand beside it; and no ticket. It raises
    [Stdlib.Out_of_memory] where the host has no memory for a table. Its
    words grow with its axes and its lines' points. *)

(** {1:contractions Contraction}

    [contract] computes every contraction {!Nx_kernel.S} offers, of
    operands, [init] and results of any dtype through any layout, scaled or
    segmented. It declines a scale with an integer or complex accumulator.
    An element enters the accumulator, and a sum enters [dst], as
    {!Nx_kernel.Prog.Cast} stores it. A scale multiplies each element,
    rounded once into the accumulator, its value as a cast stores it, where
    an [Exponent] code [e] stands for [2{^e - 127}]. An integer sum wraps in
    the accumulator, so every association gives it the same bits. *)

(** {2:contraction_sums Float sums}

    A float contraction's association depends on the dtypes, whether it is
    scaled or segmented and the extents alone, never on a layout or a
    segmented call's ids. A box of a result computed alone may round
    otherwise. Its products are numbered as
    {!Nx_kernel.section-contraction} numbers them, and a complex sum is the
    float sums of its real and imaginary parts, both in one association.
    Each range of an output's products below adds into one running sum,
    which may take more than 256 of them, so [β(n)] is [2n + 1]
    ({!Nx_kernel.section-bounds}). An output's products sum from [+0], and
    its [init] is added after. Its error factor is the one of its kind
    below, [K] an output's products, [u] the accumulator's unit roundoff.
    An output without products is as {!Nx_kernel.section-contraction}
    states, in every kind. *)

(** {2:matrix_units Matrix units}

    A call sums on the matrix units where it is unscaled and unsegmented,
    of a contracted extent [k] from 1 to [2{^30}], into [float32] from
    operands both [float16], both [bfloat16], each float8, or a float8 one
    beside a [float16] or [bfloat16] one, which the units read in its
    partner's dtype, exactly; and into [int32] or [uint32] from [int8],
    [uint8], [int4] or [uint4] operands, a 4-bit one read as 8 bits.

    A unit's {e step} adds exact products [p] to a running sum [c]: 4
    products of 16-bit operands, in the pairs of terms [k] and [k + 1], or
    8 of float8 ones, in the groups of terms [k] to [k + 3], [k] the
    step's first or the one after its pair or group. A nonzero product of
    factors of exponents [ea] and [eb], a subnormal's that of its dtype's
    least normal, lies on the grid [2{^ea + eb - 24}]; a zero product
    lies on none. The step truncates each pair's or group's products
    toward [-∞] to the coarsest grid among them, then the pairs' or groups'
    sums and [c] to the coarser of the products' coarsest grid and
    [2{^h - 26}], [h] the exponent of [c]'s leading bit or [-126] if
    lower; a negative [c] below that grid, where it is above [2{^-126}],
    truncates to [0]. It rounds the total once to nearest even, and a total
    that rounds to zero, a negative one included, gives [+0]. A step of [q]
    nonzero products of normal factors errs by at most
    [(q + 1) u (|c| + Σ|p|)].

    A subnormal factor, in the dtype the units read, counts as underflow
    and voids the error factor below. A product whose factors are at least
    [2{^-δa}] and [2{^-δb}] times their dtype's least normal lies on a grid
    up to [2{^δa + δb}] times coarser against itself, [δa] and [δb] at most
    10 of [float16], 7 of [bfloat16], 3 of e4m3 and 2 of e5m2. With [δ] the
    largest [δa + δb] among a step's products, the step errs by at most
    [(1 + 2{^δ} q) u (|c| + Σ|p|)]: [4097 u] for four [float16] products,
    one with a factor of [2{^-24}]. *)

(** {3:matrix_ranges Ranges}

    The units compute tiles of outputs: of 16 x 64 where [m] or [n] is at
    most 16, [m] and [n] the products of [a]'s and [b]'s free extents, and
    otherwise of 128 x 128 or 16 x 64, whichever has the least
    [⌈t S / 64⌉ w], [t] the tiles a call takes, [S] its ranges below, [w]
    [16384] for 128 x 128 and [3072] for 16 x 64, 128 x 128 on a tie. A
    tile of 16 x 64 takes 16 rows, or 16 columns where [n <= 16 < m].
    An output's [r = ⌈k / 128⌉] spans of 128 terms go in
    [S = max 1 (min (min 64 ⌊R / t⌋) ⌊r / l⌋)] ranges, [R] 64 for tiles of
    128 x 128 and 256 for 16 x 64, range [s] taking the spans from
    [⌊s r / S⌋] to below [⌊(s + 1) r / S⌋]: [l = 16] where [m] or [n] is
    at most 4, or the outputs, the batch elements times [m n], number at
    most [2{^18}], and [l = 256] otherwise.
    Each range's sum starts from [+0]. A range adds its products in
    increasing k, one step at a time, the last step's missing products
    zero. The ranges' sums are added in increasing k, rounding each
    addition, then [init], and the result rounds once into [dst]. A sum
    that rounds to zero is [+0], and so is that sum plus an [init] of
    [-0]. With [s] the products of a step, the error factor
    is [γ(K + ⌈K / s⌉ + S - 1)], at most [γ(2K + 1)]. *)

(** {2:fused Fused multiply-adds}

    Every other float sum cuts each output's [k] into at most 64 ranges of
    whole spans of 32 terms. How many follows from the call's dtypes and
    extents by a model of its speed, and may differ between the outputs of
    one call. Each range is summed from [+0] in increasing k, one fused
    multiply-add a term. The ranges' sums are added in increasing k, then
    [init], and the result rounds once into [dst]. The error factor is
    [γ(K + σ)] for a real accumulator, [σ] its scaled operands, and
    [γ(2K + 1)] for a complex one. *)

(** {2:windows Windows}

    A call computes its results window by window past [2{^30}] rows or
    columns, on the matrix units past 65,535 batch elements, and segmented
    past [2{^24}] batch elements times rows (Rows) or times the contracted
    extent (Contracting, one batch element at least). A window's
    association follows the dtypes, the call's extents and its own; on the
    matrix units its ranges are the call's. The call's words grow with
    the number of windows. A segmented window takes a few bytes of
    workspace and, for each of its batch elements, in a Rows call at most
    36 bytes a row and 12 a group, and in a Contracting call 4 bytes a
    contracted index (8 past a contracted extent of [2{^31} - 1]), 16 a
    group, and at most 8 a group and 4096 contracted indices. The indices
    of no group count as one group more. *)

(** {2:contraction_memory Memory}

    A call copies into workspace each operand, [init] or result its kernels
    do not read or write in place. A copy holds its array's elements in the
    dtype its kernel reads, an [init]'s or a result's in the accumulator's,
    integers of fewer than 32 bits in 32 but for the matrix units, elements
    of fewer than 8 bits one a byte, its contracted extent padded to 16
    bytes. It holds an operand's or [init]'s axes of one kind of stride 0
    once, and contracted ones of stride 0 past 16 elements as 16 copies, 64
    for the matrix units. A complex call's [b] is copied
    as each element's 2 x 2 real matrix [[re, -im], [im, re]], twice its
    bytes. A sum cut into ranges of k holds each range's sums of a tile of
    outputs, and a ticket for each tile: on the matrix units at most 4 MiB
    of sums. *)

(** {1:sorting Sorts and numbering}

    [sort] and [unique] compute every dtype, layout, direction and number
    of kept elements, and keys of any number of columns. Both use only
    integer arithmetic, so their results are the contract's exact ones.

    An element's key has [k] bytes: the element's, or one where it has
    fewer than 8 bits. A sort's launches depend on its slices' length
    [n]:
    {ul
    {- A slice of at most 4096 elements, 2048 of [complex128], sorts in a
       group's LDS: one launch.}
    {- A longer slice that keeps at most a quarter of that is selected: a
       launch zeroes the selects, a round of radix select takes a launch
       for each of the [k] key bytes and of the [p] bytes of [n - 1], a
       launch gathers the elements kept and one sorts them in LDS:
       [k + p + 3] launches.}
    {- Any other slice sorts by radix: a launch zeroes the digit counts,
       one counts the digits of every byte of the keys, then each byte
       takes a launch that scatters, after one that counts where a slice
       has several parts: [2 + k] or [2 + 2k] launches. A byte that every
       key of a slice shares is skipped within its launches.}}

    A numbering of [r > 0] rows of [w] columns zeroes its counts in one
    launch, sorts the rows a column at a time from the last by passes as
    a sort of 8-byte keys, in [1 + 8 q] launches a column with [q] 1 or
    2, then numbers them in four: [5 + w (1 + 8 q)] launches. A numbering
    of no row takes one. *)

(** {2:sorts_memory Memory}

    A sort takes workspace of at most 66 bytes an element of its operand
    and 512 KiB, and a ticket a slice. A numbering takes workspace of at
    most 65 bytes a row of its keys, 89 past [2{^32}] rows, 16 KiB a
    column and 544 KiB, and one ticket. A numbering's words grow with its
    keys' columns. *)

(** {1:matrices Matrices}

    [cholesky], [lu], [qr] and [solve_triangular] compute every case
    {!Nx_kernel.S} offers. A matrix's results are a function of its values
    alone, whatever its batch or its layout. *)

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
    element's real part.

    A [qr] past its one launch ({!section-matrices_memory}) applies each
    block's reflectors by products. A product of [r] x [c] outputs of [k]
    terms each cuts an output's [K] real terms ([2k] at a complex dtype)
    into [R = ⌈K / 32⌉] spans of 32, and range [s] of [S] takes the spans
    from [⌊s R / S⌋] to below [⌊(s + 1) R / S⌋], with
    [S = min (⌈k / 512⌉, ⌈128 / t⌉)] where [t = ⌈r / 64⌉ ⌈c / 64⌉] is below
    64 and [k] at least 2048, and [S = 1] otherwise. Each range is summed
    in increasing k, one fused multiply-add a term, from [+0], or from the
    output's own value for the first range of a product that updates it;
    the ranges' sums are added in increasing k.

    A [qr]'s panels, of at most 32 columns (16 at [complex128]), take
    Householder steps in column order. Step [j]'s sums over the rows below
    its diagonal, of [|x_i|²], of the largest real or imaginary part of
    [x_i] and of the later columns' elements, and of [conj(x_i) a_ic] for
    each later column [c], run over the matrix's rows in blocks of 256 from
    row 0. A block's rows go to 16 lanes by row modulo 16, each lane's sum
    from [+0] in increasing row, one fused multiply-add a term, two for
    [|x_i|²] and each part of a complex product; lane [l] then adds lane
    [l + 8]'s, [l + 4]'s, [l + 2]'s and [l + 1]'s, in turn. The blocks from
    the panel's first go 32 to a group, combined by the same tree over 32
    lanes, [l + 16] first, and the groups' sums are added in increasing
    order. Where [x]'s largest part [g] is below [2{^-30}] or above
    [2{^30}] ([2{^-250}] and [2{^250}] at a 64-bit part), or the exponents
    of [g] and of the later columns' largest part add to below [-100] or
    above [100] ([-900] and [900]), the sums are of [x_i 2{^-e}],
    [e] [g]'s exponent. The reflector is xLARFG's: [β = -sign(Re α)
    ‖(α, x)‖₂], taken again of [x] and [α] multiplied by [2{^102}]
    ([2{^969}]) while [|β|] is below [2{^-102}] ([2{^-969}]), at most 20
    times, [τ = (β - α) / β], [σ = 1 / (α - β)] and [v = (1, σ x)]. A later
    column takes [w_c = conj(τ) (a_jc + conj(σ) s_c)], [s_c] its sum, and
    [a_ic - w_c v_i], one fused multiply-add. T is xLARFT's from the sums
    of [conj(v_ia) v_ib], [T_ri = -τ_i Σ T_rc g_ci] summed from [+0] in
    increasing [c]. The one launch's [q] is [I - V M], [M = T V1ᴴ], each
    element of [M] summed from [+0] and each of [q] from its identity
    element, in increasing step. *)

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

(** {2:matrices_memory Memory}

    With [e] the bytes of an element, [w = min(n, 64)] and
    [b = min(k, 128)] (32 and 64 at [complex128]), and [B = max(n, c)],
    [c] [q]'s columns, each routine takes at most these bytes of workspace
    a matrix:
    {ul
    {- [cholesky]: [4 + 3 (n - w) (w e + 16)] past order 64 (32 at
       [complex128]).}
    {- [solve_triangular]: [n² e] where [a]'s batch axes do not merge into
       one strided axis, and past order 64 (32 at [complex128])
       [4 + (n + 2 r) (n e + 16)] more, [r] [b]'s columns.}
    {- [lu]: [m ((w + 1) e + 16) + p (3 w e + e + 117)], [p = ⌈m / 128⌉]
       in a batch of fewer than 32 matrices of more than 512 rows, [1]
       otherwise; past 64 columns (32 at [complex128]) [(m + 2n) (n e + 16)]
       where that is more.}
    {- [qr]: [e (k + m n + (m + 800) (w' + 2) + w'²)],
       [w' = min(n, 32)] (16 at [complex128]). Past 32 columns (16 at
       [complex128]) or 1024 rows, or where
       [(64 + n + n² + n (n - 1) / 2 + n c) e] exceeds 32 KiB,
       [e (m b + ⌈k / b⌉ b² + 2 b B) + (b + 2 B) m e + 16 (m + 2 B)] more,
       and where [m] is at least 2048 [S ⌈b / h⌉ h ⌈B' / v⌉ v e'] more and
       [4 ⌈b / 32⌉ ⌈B / 16⌉] bytes of tickets: [S = min (⌈m / 512⌉, 128)],
       the most ranges of its products ({!section-matrices_order}),
       [B' = B] and [e' = e] at a real dtype, [2 B] and [e / 2] at a
       complex one, and [h] x [v] the tile its products take, one of
       128 x 128, 64 x 64, 32 x 32 and 8 x 256 at [float32] and
       [complex64], of 64 x 64, 32 x 32 and 8 x 128 at [float64] and
       [complex128]. Its [m n]
       elements copy [a], which a [qr] that takes no more reads in place
       where it is C-contiguous.}}
    Alignment adds at most 3 KiB a call. Of these routines, only [qr]
    takes a ticket.

    A [cholesky] or [solve_triangular] past order 64, an [lu] past 64
    columns (32 at [complex128] for each) or of more than 512 rows in a
    batch of fewer than 32 matrices, and a [qr] that takes the second
    workspace above launch a number of functions that grows with their
    extents, so their words grow with them. *)

(** {1:spectral Spectral routines}

    [eigh], [svd] and [eig] compute every case {!Nx_kernel.S} offers. A
    matrix's results are a function of its values alone, whatever its
    batch or its layout.

    [eigh] of order at most 32 and [svd] of both extents at most 32 rotate
    each matrix in LDS by cyclic Jacobi. Past them, [eigh] reduces a matrix
    to a tridiagonal one by Householder reflectors in panels of 32
    columns, then finds its eigenvalues by bisection, or with [v] by
    divide and conquer on leaves of 32, and applies the reflectors to the
    eigenvectors; [svd] reduces a matrix to a bidiagonal one, then finds
    its singular values by bisection on its Golub-Kahan matrix, or with
    [vectors] by implicit QR sweeps, and applies the reflectors to the
    vectors. [eig] balances each matrix, reduces it to Hessenberg form and
    to Schur form by shifted QR in one group, then finds the eigenvectors
    by back substitution and multiplies them by the Schur vectors.

    The products between these steps sum as {!section-fused} states, a
    matrix they update as their [init], each in ranges of k that its
    matrices' extents alone set. *)

(** {2:spectral_bounds Bounds}

    [eigh], [svd] and [eig] iterate: Jacobi to 60 sweeps, a secular
    equation to 30 steps, the first of them its initial guess, and a
    step's equation over three poles, where it takes one, to 40
    iterations, the bidiagonal QR sweeps to [6 k{^2}] rotations, [k] the
    smaller extent, and [eig] to [30 max(10, h)] QR
    iterations an eigenvalue, [h] the order balancing leaves to iterate
    on. A matrix on which an iteration does not converge within its limit,
    or that holds a NaN or an infinity where it is read, is NaN in every
    result. Otherwise, with [γ] as {!Nx_kernel.section-bounds} defines it,
    [u] the dtype's unit roundoff, [ν(k) = k η] and [θ(k) = k t], [η] the
    dtype's least positive subnormal (2{^-149} or 2{^-1074}) and [t] its
    least normal number (2{^-126} or 2{^-1022}), [k] doubled on complex
    dtypes:
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
       [vectors], and within [g ‖a‖₂ + θ(16 max(m, n) + 100)] without.}
    {- [eig]: it balances [a] into [b = D⁻¹ Pᵀ a P D], [P] a permutation
       and [D] diagonal, of powers of 2 that bring each row's norm near its
       column's; [D = I] where they are equal already. With [g = γ(16n)]
       and [h = ν(16n)], [w] is the spectrum of [a + E] with
       [‖D⁻¹ Pᵀ E P D‖_F <= g ‖b‖_F + h], with [v] or without, and each
       column [v_j] has [|‖v_j‖₂ - 1| <= g] and
       [‖D⁻¹ Pᵀ (a v_j - w_j v_j)‖₂ <= (g ‖b‖_F + h) ‖D⁻¹ Pᵀ v_j‖₂].}}

    [eigh]'s [w] ascends. [svd]'s [s] descends. [eig]'s [w] is in the
    order the Schur form's diagonal leaves it, and each column of [v] has
    an element of largest magnitude real and positive. On real [a], [w]'s
    values that are not real come in adjacent conjugate pairs, the one of
    positive imaginary part first, with conjugate columns of [v]; a real
    value has a [+0] imaginary part and its column of [v] [+0] imaginary
    parts. *)

(** {2:spectral_memory Memory}

    [eigh], [svd] and [eig] copy an [a] that is not C-contiguous into
    workspace, its elements in C order. [eigh] past order 32, [svd] past
    32 rows or columns, and [eig] take workspace of elements of [a]'s
    dtype, reals of its precision and bytes, for a matrix of order [n], or
    of [m] x [n] with [M = max(m, n)] and [c] [u]'s columns ([vh]'s rows
    where [m < n], [0] without [vectors]):
    {ul
    {- [eigh]: [n² + 65 n + 2048] elements, [3 n] reals and 16 bytes, and
       with [v] [n² + 32 n] elements, [3 n² + 75 n - 2 l (32 - l)] reals
       and [24 n + 16] bytes more, [l = n - 32 (⌈n / 32⌉ - 1)];}
    {- [svd]: [M (k + c + 33) + k² + 64 max(c, k) + 2 k + 2048] elements,
       [2 k² + 263 k + 260] reals and [4 (192 k + 203)] bytes;}
    {- [eig]: [4 n² + n] elements, [2 n² + 4 n] reals and 16 bytes.}}
    Alignment adds at most 512 bytes a matrix and 2 KiB a call. Each
    product takes, while it runs, the workspace and tickets
    {!section-contraction_memory} states for a contraction of its
    matrices; no other step takes a ticket.

    Past order 32, [eigh]'s launches, and so its words, grow with [n], and
    past 32 rows or columns [svd]'s with [k]. *)

include Nx_kernel.S

(**/**)

val bound_fresh : workspace:int -> tickets:int -> unit
(** [bound_fresh ~workspace ~tickets] makes a call take buffers of its own
    past [workspace] bytes of workspace and [tickets] bytes of tickets, in
    place of 64 MiB and 4 MiB, so that tests reach them with small arrays.
    It applies to calls planned after it returns.

    Raises [Invalid_argument] if a bound is negative or above its kept
    size. *)

val contract_wide_k : int -> unit
(** [contract_wide_k n] makes a Contracting call hold its contracted
    indices in 8 bytes past a contracted extent of [n], at least 0 and at
    most [2{^31} - 1], so that a test reaches them with few indices. It
    applies to calls planned after it returns.

    Raises [Invalid_argument] if [n] is out of range. *)

val contract_plan :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  segments:
    (int64, Nx_array.Dtype.int64_elt) Nx_array.t Nx_kernel.segments option ->
  acc:Nx_array.Dtype.any ->
  dst:('v, 's) Nx_array.t ->
  init:('v, 's) Nx_array.t option ->
  Nx_kernel.operand ->
  Nx_kernel.operand ->
  (Plan.t option, Nx_array.answer) result
(** [contract_plan ~batch ~contracting ~segments ~acc ~dst ~init a b] is the
    plan {!contract} submits for these arguments, [None] for a call of no
    launch, or the answer it gives without one. It runs nothing, so that
    tests name the kernels and ranges their draws reach. *)

val contract_unit_tile : int option -> unit
(** [contract_unit_tile rows] makes calls on the matrix units take the tile
    of [rows] rows, 16 or 128, the ranges of {!section-matrix_ranges}
    counting that tile's, so that tests reach each tile with small arrays;
    [None] restores the stated tile. It applies to calls planned after it
    returns.

    Raises [Invalid_argument] if [rows] is another number. *)

val bound_kept : int -> unit
(** [bound_kept n] makes a device keep at most [n] bytes of submissions,
    in place of 64 MiB, and a call of more bytes take a submission of its
    own, so that tests reach both with small arrays;
    {!Plan.kept_submissions} restores the library's bound. It applies to
    calls planned after it returns, and a device drops the submissions
    past it at its next keep.

    Raises [Invalid_argument] if [n] is negative or above 64 MiB. *)

val bound_scatters :
  parts:int -> groups:int -> whole:int -> wide:bool -> unit
(** [bound_scatters ~parts ~groups ~whole ~wide] makes a float [Add]'s sort
    by target cut its entries into at most [parts] parts, in place of 256,
    and its launches take at most [groups] groups, in place of
    {!Plan.stride_groups}; its runs of at most [whole] entries, in place of
    64, summed whole, none where [0]; and with [wide] its items wide,
    whatever their bits, so that tests reach parts of several tiles, groups
    of several parts, runs summed by levels and wide items with small
    arrays. It applies to calls planned after it returns.

    Raises [Invalid_argument] if [parts] is not from 1 to 256, [groups]
    not from 1 to {!Plan.stride_groups} or [whole] not from 0 to 64. *)

val bound_iterations :
  sweeps:int -> steps:int -> qr:int -> rotations:int -> unit
(** [bound_iterations ~sweeps ~steps ~qr ~rotations] sets the spectral
    routines' limits ({!section-spectral_bounds}) to [sweeps] Jacobi
    sweeps, [steps] steps of a secular equation, the first of them its
    initial guess, which [~steps:1] leaves uncorrected, [qr max(10, h)] QR
    iterations of [eig] an eigenvalue and [max(rotations k{^2}, k)]
    rotations of the bidiagonal QR sweeps, so that tests reach each
    failure with small matrices. [~sweeps:60 ~steps:30 ~qr:30
    ~rotations:6], the initial limits, restores them. It applies to calls
    planned after it returns.

    Raises [Invalid_argument] if [sweeps] or [steps] is below [1], or if
    [qr] or [rotations] is negative. *)

val kept_calls : Rig.t -> int
(** [kept_calls d] is the number of submissions kept for later calls on
    [d], [0] before [d]'s first call. *)

val kept_bytes : Rig.t -> int
(** [kept_bytes d] is the bytes, as the preamble counts them, of the
    submissions kept for later calls on [d], [0] before [d]'s first
    call. *)

val drop_kept : Rig.t -> unit
(** [drop_kept d] drops every submission kept for later calls on [d], so
    that a test counts the ones its calls keep. *)

val bound_constants : int -> unit
(** [bound_constants n] makes a device keep at most [n] bytes of
    constants, in place of 64 MiB, so that tests reach a call's own
    constants with small ones. It applies to calls submitted after it
    returns.

    Raises [Invalid_argument] if [n] is negative or above 64 MiB. *)

val kept_constants : Rig.t -> int
(** [kept_constants d] is the number of constants kept for later calls on
    [d], [0] before [d]'s first call. *)

val call :
  Rig.t ->
  written:Nx_array.any iarray ->
  read:Nx_array.any iarray ->
  Plan.t ->
  Nx_array.answer
(** [call d ~written ~read c] is the answer of a kernel on [d] that planned
    [c], claiming [written] and [read] through {!Nx_array.door}. *)

module Kernels = Kernels
(** The code object's kernels and their words. For tests. *)

module Kept = Kept
(** Entries by key within a byte budget, as a device keeps its
    submissions. For tests. *)

module Plan = Plan
(** Calls of the kernels, as a family plans them. For tests. *)

module Sort = Sort
(** The plans of sorts and numberings, and their bounds. For tests. *)

val product :
  ?init_first:bool ->
  ?neg:bool ->
  ?conj_a:bool ->
  ?conj_b:bool ->
  ?tri:[ `Lower | `Upper ] ->
  ?ranges:int ->
  ?workspace:bool ->
  acc:Nx_array.Dtype.any ->
  init:bool ->
  y:Nx_array.any ->
  Nx_array.any ->
  Nx_array.any ->
  Nx_array.answer
(** [product ~acc ~init ~y a m] is the matrix routines' product
    [y <- [y +] op(a) op(m)] of [a] [batch; rows; k] and [m]
    [batch; k; cols] into [y] [batch; rows; cols], as one call, with the
    routines' options: [init_first] starts the first range of k from [y];
    [neg] negates [a]; [conj_a] and [conj_b] conjugate [a] and [m]; [tri]
    computes only the tiles that hold an element of that triangle, the
    diagonal included; [ranges] [s] cuts each output's [k'] terms,
    [k' = 2k] for a complex [acc], into [S = min s ⌈k' / 32⌉] ranges, range
    [i] taking the spans of 32 terms from [⌊i r / S⌋] to below
    [⌊(i + 1) r / S⌋], [r = ⌈k' / 32⌉]. With [workspace], each matrix is copied
    into a piece of the workspace laid out as it is in its buffer, the
    product computed there and [y] copied back. [y] shares no byte with [a]
    or [m]. For tests. *)

(**/**)
