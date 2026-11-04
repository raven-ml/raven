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

    On x86-64 nx.cpu requires x86-64-v3 (AVX2, FMA, F16C and BMI2): on a
    host without it, initialising the library raises [Failure], before any
    kernel can run. Other architectures have none.

    A call keeps other domains waiting to collect for no longer than about
    one unloaded core's copy of 1 MiB at a time, however many operands,
    reductions or program nodes it has
    ({!Nx_array.section-kernels}). It runs its work with the domain lock
    released where nx.cpu estimates that work to take longer than that copy;
    the estimate is an upper bound of the work's time on one core of the
    machines it was measured on. Other threads of the calling domain may
    run OCaml code during a call. Several domains may call concurrently. The
    pool runs one job on several threads at a time, so a released call
    whose work takes several threads waits for any other such job to end.

    A kernel's result is a function of its operands' values alone, in the
    sense that neither layouts nor threads change a bit of it, but for a
    NaN's sign and payload where {!Nx_kernel.Prog} leaves them unspecified:
    such a NaN result stays a NaN, and its bits may change with the
    element's position, the layouts, the operands' addresses or the
    threads. An element's association, its algorithm and whether the
    kernel declines depend on the extents the element reads whole (a
    reduction's, a scan's, a sort's or a transform's axes, a numbering's
    keys, the contracted extent, a routine's matrices, a gather's or a
    scatter's axis, the axes a padded load or a fold pads or windows), the
    dtypes, the attributes and the host's architecture, never on the
    result's other extents; a contraction's association follows its rows,
    columns and depth, as stated below. Each kind computes through the
    function of its name in [nx_kinds.h], with that function's bits, NaNs
    aside, on every target. So [Exp2] is exact at integers whose power is
    in the dtype, and [Log2] at powers of two. The kinds keep subnormal
    operands and results. A transform's bits too are the same on every
    target, NaNs aside. *)

(** {1:lane_order Lane order}

    {e Lane order} puts the terms of a sum or a product, numbered as
    {!Nx_kernel.section-reductions} and {!Nx_kernel.section-contraction}
    number them, into blocks of 1024 consecutive numbers and, within a
    block, into 16 lanes by number modulo 16, each lane adding its terms in
    increasing number from [+0], or multiplying them from its first. Lane
    [i] takes lane [i + 8] for [i] below 8, then lane [i + 4] for [i] below
    4, then [i + 2] for [i] below 2, then lane 0 takes lane 1, a lane
    holding no term taking no part. The blocks are combined by a binary
    tree whose left part holds the largest power of two of blocks below
    their count, each part combined the same way. *)

(** {1:maps Maps}

    [map] computes every program, whatever the number of its loads and
    outputs. Its loop holds the outputs, the loads and, where a node reads
    a coordinate, one array more; a loop of more than
    {!Nx_array.max_operands} holds its tables of them on the host heap for
    the call. *)

(** {1:padded Padded loads}

    A padded load of [map], [reduce] or [scan] is read from one copy in
    host memory that the call holds until it returns: its operand padded
    before the windows, which are a view of the copy, an axis the operand
    broadcasts and the pad leaves alone held at one element. A pad that
    fits no operand of its load's shape, or a padded shape other than the
    loop's, is refused with [Shape_mismatch] before the copy is made. *)

(** {1:contractions Contractions}

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
    gives an infinite term where the exact one is finite. *)

(** {2:contraction_order Order}

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
    reads the ids whole before any write, so they may be [dst] itself. It
    groups the indices by counting and copies what the ids cut into
    scratch, group after group: [Rows]' rows of [a] and [init], whose
    results it copies back, or both operands along [Contracting]'s pair.
    Each group computes as an unsegmented call of its indices alone. *)

(** {2:contraction_bound Bound}

    A complex output of [K] products is within
    [γ(2K + 2) (|init| + Σ|a||b|)] of the exact sum of [a], [b] and [init]
    as [acc] holds them, with [γ] as {!Nx_kernel.section-bounds}
    defines it and [u] its parts' unit roundoff, barring overflow and
    underflow. A rounding errs by at most [u] times the modulus of the
    partial sum it rounds, both parts at once, and a partial sum of any of
    the terms has a modulus of at most [|init| + Σ|a||b|]. Chain order
    rounds [2K] times. Lane order holds a product's two terms in
    neighbouring lanes until lane 0 takes lane 1. Until then, a term passes
    at most [⌈2K/16⌉ + 3] roundings and a product's two weigh at most
    [√2 |a||b|]; after, at most [log₂] of the blocks [+ 2]. Both stay
    within [2K + 2] roundings. *)

(** {1:reductions Reductions and scans}

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
    program's tile by tile. Both take any program [map] computes, of any
    number of loads. They read it over its loads or, with none, over the
    loop's shape alone, a reduction's [shape] or a scan's destinations':
    its terms are the program's output there as [map] computes it, so a
    reduction or a scan of an output has the bits of the output mapped into
    an array, then reduced or scanned. [reduce] computes several such
    reductions of one loop, each one's bits alone, every destination checked
    before any is written, any number of them. They answer [Wrong_dtype]
    where {!Nx_kernel.accepts} is [false] of a reduction and its output's dtype
    or an [Arg]'s positions are not [int64], and [Shape_mismatch] for
    destinations other than a reduction's one or two. A destination identical to the operand, or to a program's
    load, is accepted: that operand or load is then read from a copy the
    call holds, and the results are those of fresh destinations. They
    refuse with [Shape_mismatch] what {!Nx_kernel.Shape.padded} and
    {!Nx_kernel.Shape.reduce} or {!Nx_kernel.Shape.scan} answer [Error]
    for, a reduction's loads whose shape once loaded is not its [shape],
    and destinations whose shapes are not the ones they give. *)

(** {2:sums Sums and products}

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
    chunk, has the bits of the scan of the whole slice from that chunk on. *)

(** {2:states Logsumexp and Moments}

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
    exact [V]. *)

(** {2:state_scans Logsumexp and Moments scans}

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
      [(7 D + 1) u V + (4 D² - 2 D) u R √V]. *)

(** {1:scatters Gathers and scatters}

    [gather], [gather_ranges] and [scatter] compute every case, at every
    dtype, and refuse with [Shape_mismatch] operands and destinations whose
    shapes do not fit. [gather_ranges] walks each lead index's ends once in
    order, and copies each run's rows inside the source's axis as one
    [memcpy] where the source's rows lie one after the other, else as a
    copy of strided elements; it splits the destination among threads at
    whole cache lines, so that no two write one byte. A
    scatter's [Add] adds a target's updates left to right,
    [(… ((x + u1) + u2) + …) + un]. A {e slice} is the updates at one index
    off the scattered axis, with their positions and targets; a {e row of
    slices} is the slices at successive indices of the last axis off it,
    joined with the axes before it where every operand steps through them
    as one. Over [m] updates and [t] targets along the scattered axis, the
    [Add] of a float of a byte or more or of a complex number, of [s]
    slices fewer than its threads, each the only slice of its row or, for
    a narrow float, one whose positions differ along its row, with
    [m ≥ 2{^16}] but for a narrow float, sums each slice on one thread
    where [s t] elements of its dtype, of [float32] for a narrow float, fit
    in the host's largest cache, and otherwise sorts its updates by target
    on every thread.

    A narrow float's [Add] keeps its float32 sums and the positions it
    sorts by target in host memory the call holds until it returns. It
    takes 12 KiB and [32 m] bytes on each of its threads; in place of the
    [32 m], [40 m + 8] bytes once where its positions step 0 along each of
    fewer rows of slices than threads, and otherwise, where its slices are
    fewer than its threads and its dtype is [float4_e2m1fn], 32 bytes for
    each update in all. Where it sums each slice on one thread, it takes
    instead [5 t + 64] bytes on each thread that sums one. Where it sorts
    its updates on every thread, at a dtype of [w] bytes, it takes instead
    [(8 + w) m] bytes in all where [t ≥ m], else [(8 + w) min (m, 2{^22})],
    and on each thread [16 w + 136] KiB and five eighths of a core's L2
    cache or [5 t / 512] bytes, whichever is more; where [m] exceeds both
    [t] and 2{^19}, about [5 t] bytes in all in place of each thread's L2
    or [5 t / 512]. *)

(** {1:sorts_folds Sorts, assemblies and folds}

    [sort] and [assemble] compute every dtype, and [fold] every dtype
    {!Nx_kernel.Prog.Add} takes. The three refuse with [Shape_mismatch]
    operands and destinations whose shapes do not fit. A fold adds an
    index's elements from left to right, in C order of their taps, with no
    [+0] before the first; it stores [+0] at an index no tap reads. A
    narrow float's or a sub-byte dtype's fold rounds each addition to its
    dtype, as {!Nx_kernel.Prog.Add} does. *)

(** {1:numbering Numbering}

    [unique] refuses with [Shape_mismatch] operands and destinations whose
    shapes do not fit, and declines keys of [2{^40} - 1] rows or more. Its
    [k] keys of [w] words each find their number in a hash table of linear
    probing, by a hash that folds each word in by the high and low halves of
    a 128-bit product xored, wyhash's, keyed by secrets. Keys of at
    most 4 words whose columns' spans, from each column's least word to its
    greatest, multiply to at most [2{^20}] values, and to at most twice a
    thread's rows and 1024 more, find it at their place among those values
    in a direct table, from one more read of the keys that finds the spans.
    A thread takes up to [2 (w + 3) k] words for its keys, their words,
    hashes, first rows and counts, and a table of 1024 slots or 2 to 4
    times its keys, a word each, or a direct table of 4 bytes a value.

    On one thread keys are numbered as their first rows are met. On several,
    each thread numbers the keys of a block of consecutive rows. The blocks'
    keys then merge into tables by their hashes, which take as much again
    as the blocks' tables, and each key is numbered by its first row's rank
    among the keys' first rows, from a bit a row. A last pass maps each
    row's number in its block to its key's, one more read and write of 8
    bytes a row.

    The hash is keyed by a seed drawn from the system's random source when
    the library initialises, so that keys chosen in advance cannot share
    their hashes and lengthen the probes. The results follow first
    appearance alone: the same bits under any seed, on every run. *)

(** {1:transforms Fourier transforms}

    [c2c], [r2c] and [c2r] answer [Wrong_dtype] for dtypes {!Nx_kernel.S}
    does not take, and compute the others. They refuse with
    [Shape_mismatch] what {!Nx_kernel.Shape} answers [Error] for and
    results of other shapes. A transform's result
    is a function of its operand's values alone, whatever its layout, its
    batch or the threads, but for a NaN result's sign and payload, which are
    unspecified. A transform works in host memory the call holds until it
    returns. On Linux, a line by Bailey's four steps of at least 32 MiB and
    past the largest cache, from an operand apart from its result and into
    points that lie side by side there, keeps its points between the steps
    in its result in place of that memory. Barring
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
      ones of [n/2] points. *)

(** {1:attention Attention}

    [attention] computes every case of [q], [k] and [v] of float16,
    bfloat16, float32 or float64 in an accumulator, [lse]'s dtype, of
    float32 or float64 and not narrower. It refuses with [Shape_mismatch] what
    {!Nx_kernel.Shape.attention} answers [Error] for and destinations of
    other shapes. Each query row reads its keys in {e blocks} of 64 from
    key 0, up to its lead index's length, and keeps a maximum [m], a sum
    [l] and an accumulator [a] of [dv] elements, from [-∞], [+0] and [+0].
    For each block in which the row has a visible pair:
    - each pair's score is the sum of the [dk] products of the row's and
      the key's elements in the accumulator, added in increasing
      dimension from [+0], each fused into its sum, times the scale
      rounded to the accumulator, rewritten by the program's output 1;
    - [m'] is the greatest of [m] and the block's visible scores, a NaN
      where one is. Where it is [-∞], the block's weights are [0] and the
      state is left as it was; otherwise each weight is
      [exp (x - m')], [0] at a hidden pair, [l] becomes [α l + s] with
      [α = exp (m - m')] and [s] the block's weights summed term [j] into
      lane [j mod 8], lanes [i] and [i + 4], then [i + 2], then [i + 1]
      added, and [a] becomes [α a], unchanged where [α] is [1];
    - [a] then adds each weight times its key's value, in increasing key,
      each product fused into its sum, over the block's keys, a hidden
      pair's weight [0] included; where a value of the block hidden from
      the row is a NaN or an infinity, over its visible pairs alone.
    A block in which the row has no visible pair leaves its state as it
    was. At the end, [o = a / l + 0] in the accumulator, rounded once to
    [o]'s dtype: the [+0] makes [+0] a [-0] that [α a] leaves of zero
    products, as [α = 0] does of a negative [a]. [lse = m + log l], [+∞]
    where [m] is; a row of [l = 0] has [o = +0] and [lse = -∞]. [exp] and
    [log] are {!Nx_kernel.Prog.Exp}'s and {!Nx_kernel.Prog.Log}'s. So a
    value at a pair hidden from a row, finite or not, never reaches its
    [o], and a row's bits depend on its own pairs, [dk], [dv] and the
    dtypes alone.

    The program runs, as [map] computes it, for the pairs of the blocks the
    kernel reads: its output 0 for a block before its keys are read, its
    output 1 on the block's scores. A program whose output 0 is the
    constant [true], or whose output 1 is its operand 0, runs no node for
    it. The rows of a lead index's key head go in {e units}: [h]
    consecutive query heads of its group, [h] the lesser of the group's
    size and 64, and [⌊64 / h⌋] consecutive rows of each. Before a unit
    reads its blocks, it runs the ranges of output 0
    ({!Nx_kernel.Prog.ranges}) over each block's box: the unit's lead
    index, heads and rows, the block's 64 keys, and each load's least and
    greatest element over them, a load of fewer than 8 bits taking its
    dtype's ends. A block whose ranges hide every pair is skipped, and one
    whose ranges show every pair is read without running output 0. Where
    output 0 reads neither the score, nor a coordinate of the lead axes or
    the heads, nor a load that steps along one, and there are more units
    than the rows of one query head fill, as with several lead indices or
    key heads, or a group of more than 64 heads, the classes are taken
    once a call, at lead index 0 and head 0: the ranges first, then output
    0 at every pair of a block they leave mixed, which is read, or skipped,
    without running it again where it shows or hides all its pairs.

    Keys and values of the accumulator's dtype whose dimension steps one
    element are read in place, keys only in float32 and for fewer than 12
    rows of a key head; others are copied into the accumulator by blocks.
    Each thread holds at most [w (128 dk + 192 dv + 8192) + 12288] bytes of
    host memory for the call, [w] the accumulator's bytes, and
    [2 ⌈tk / 64⌉] more where output 0 is not constant; the call at most
    [tq ⌈tk / 64⌉] bytes where output 0 is shared so. The call then holds
    the ranges' bounds: for each load they read, two elements per lead
    index, unit and block along the axes the load steps on, and 16 bytes
    per index of each lead axis, per unit's heads, per unit's rows and per
    block.

    For a row that reads [t] keys in [n = ⌈t / 64⌉] blocks, barring
    overflow and underflow, to first order in the accumulator's unit
    roundoff [u], [lse] is within [(7 n + 9 + (λ + 1) ln t) u + u |lse|] of
    the exact log-sum-exp of the row's computed scores, and each weight [o]
    applies within a relative [(13 n + 9 + 2b + ln t) u] of its exact one,
    [b] bounding the visible scores' magnitudes, {!Nx_kernel.Prog.Exp}
    being within 2 ulps and {!Nx_kernel.Prog.Log} within 1 in float32 and
    2 in float64, [λ] being [3] and [5]: attention.c derives both.
    The bound {!Nx.attention} reads as its [ε], which covers both, is
    [(13 n + 9 + (λ + 2) ln t) u + u |lse|]. The value product takes the
    weights in the accumulator. *)

(** {1:matrices Matrices}

    [cholesky], [lu], [qr], [solve_triangular], [eigh], [svd] and [eig]
    answer [Wrong_dtype] for dtypes {!Nx_kernel.S} does not take, and
    compute the others. They refuse with [Shape_mismatch] what
    {!Nx_kernel.Shape} answers [Error] for and results of other shapes. A
    matrix's results are a function of its values alone, whatever its
    batch, its layout or the threads. A destination of [solve_triangular]
    identical to [a] or [b] is accepted: they are then read from copies the
    call holds, and the results are those of a fresh destination. With [γ]
    as {!Nx_kernel.section-bounds} defines it, [ε(k) = k η], [u] the
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

(** {1:kernels Kernels} *)

include Nx_kernel.S

(**/**)

val one_thread : (unit -> 'a) -> 'a
(** [one_thread f] is [f ()], every call of the calling thread running its
    work on that thread alone while [f] runs, nested calls included. The
    bits are those of any threads. For tests. *)

val targets : string list
(** [targets] is the names of the target tables the host runs, each one's
    instructions a superset of the one's before: ["base"] first, and last
    the best, which nx.cpu runs from the start. Only the contractions' and
    linear algebra's kernels differ by table. For tests. *)

val with_target : string -> (unit -> 'a) -> 'a
(** [with_target t f] is [f ()], every call running target table [t]'s
    kernels while [f] runs, on every thread. Raises [Invalid_argument] where
    [t] is not in {!targets}. For tests, while no other domain calls
    nx.cpu. *)

val with_cache : int -> (unit -> 'a) -> 'a
(** [with_cache n f] is [f ()], every call choosing its algorithms while
    [f] runs as if the host's largest cache held [n] bytes. The bits are
    the same. For tests, while no other domain calls nx.cpu. *)

val unique_seed : int64 -> unit
(** [unique_seed s] keys [unique]'s hash by [s] for the calls that start
    after it. For tests. *)

val unique_state : int64 array -> int64
(** [unique_state ws] is the state of [unique]'s hash after the words [ws]
    under the current seed. Rows whose words leave one state hash alike: a
    row [[|a; c lxor unique_state [|a|]|]] has the hash of every other such
    row of the same [c]. For tests. *)

(**/**)
