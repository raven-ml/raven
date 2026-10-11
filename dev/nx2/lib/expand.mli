(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Expansions: an operation's optional case as core operations, applied through
    [apply].

    A map of several nodes expands into one-node maps, one per node in order,
    each over the values of the nodes it reads; a constant or a coordinate is a
    creation of the map's shape. A reduction or a scan that is not plain, one
    monoid of its operand itself accumulated and rounded in the operand's dtype,
    expands into a map of each output it reduces into its accumulator ([float32]
    for the floats narrower than 32 bits, the byte-wide dtype for the sub-byte
    ones, the output's dtype otherwise), the plain loop, and a cast to the
    result's dtype. A sum of complex numbers sums their parts, read as floats
    along a last axis of two.

    [Logsumexp], [Moments] and [Arg] reduce through core loops, in the
    accumulator for the first two and in the output's dtype for [Arg]. A
    [Logsumexp] is a [Max] [m], [m'] its finite elements and [0] elsewhere, a
    [Sum] of [exp (x - m')], then [m' + log s], or [m] where it is a NaN; with
    no term, [log 0]. [Moments] are two sums: the mean [Σ x / n], then the
    variance [Σ (x - mean)² / n], each the mean where it is a NaN. An [Arg] is
    the [Max] or [Min] [m], then a [Min] of the positions, numbered in C order
    of the reduced indices, whose term has [m]'s bits or, where [m] is a NaN, is
    a NaN. Where a float sum of [n] terms is within [γ(n - 1) Σ|x|] of the exact
    sum ({!Nx_kernel.reduction}), and barring overflow and underflow, a
    [Logsumexp] [r] of [n] terms whose maximum is finite is within
    [γ(2n) + u|r| + ε] of the exact one, [ε] the error [Exp] and [Log] add; the
    mean within [γ(n) Σ|x| / n], and the variance [V] within
    [γ(n + 3) V + (1 + γ(n + 3)) γ(n)² (Σ|x| / n)²], for [n] exact in the
    accumulator. A scan of [Logsumexp], [Moments] or [Arg], the first two in
    the accumulator, doubles: along an axis of [n] it is [K = ⌈log₂ n⌉] maps,
    at least one, pass [k] combining each index's state with the state [2^k]
    before it, the earlier one first: O(n log n).
    [Logsumexp] combines [a] and [b] into [M + log1p (exp (m - M))], [M] the
    two's [Maximum] and [m] their [Minimum], or [M] where [M] is not finite,
    [a] taken as [-∞] where none is before; a result of [n] terms of largest
    finite magnitude [X] is within [K (u (X + ln n + 1) + ε)] of the exact
    one, [ε] the error [Exp] and [Log1p] add. [Moments] combines the means
    [μ] and sums of squared deviations [S] of the terms less the slice's
    first term [c], [c] taken as [0] where it is not finite: [na] and [nb]
    terms into [μa + δ nb / n] and [Sa + Sb + δ² na nb / n], [δ = μb - μa]
    and [n = na + nb], or the mean [μa + μb] where either is not finite. At
    [n] terms the mean is [c + μ] and the variance [S / n], or [μ - μ] where
    [μ] is not finite. With [R] the terms' range, the mean [μ] is within
    [(4K + 1) u R + u|μ|] of the exact one and the variance [V] within
    [(8K + 1) u V + (6K² + 2) u R √V]. These bounds hold to first order,
    barring overflow, for [n] exact in the accumulator. [Arg] combines pairs
    of an extreme and its position, keeping the earlier pair unless the
    later extreme is past it in [Max]'s or [Min]'s order: exact.

    A one-node map at a dtype other than a base one ({!base}), but a cast, a
    bitcast or a copy, computes at its dtypes' accumulators, which hold their
    values exactly, and rounds once to its dtype. A selection, a constant and a
    copy move bits: they compute at the dtypes {!kept} gives. An assembly
    expands into a fill of its flat result, then per piece in order a scatter of
    the piece's elements at their flat positions, a map of coordinates: O(n) per
    piece. A gather, a scatter or a sort of a sub-byte dtype casts its values to
    their accumulator, computes there and casts the result back once; an
    integer's [Add] wraps there to the same bits as at its own dtype. Any other
    scatter whose targets may repeat is one unique scatter per position along
    its axis, in order, each of the target so far: O(m |into|) for m positions.
    A sum associates left to right; a narrow float's sums in its accumulator,
    rounded once, and a [Set] of [true] at the same targets keeps the bits of
    each target no update reaches: twice the scatters.

    A contraction whose accumulator is a float other than its output's dtype
    first runs as a contraction into the accumulator, then casts to the output.
    Any other expands into a map of each pair of elements multiplied in the
    accumulator, a sum over the contracted axes, the [init] added, and a cast to
    the output's dtype. A sort of another dtype sorts positions by its order
    keys, int64 values below [2^32], one for an integer or a float of at most 32
    bits, two for one of 64 and per part for a complex number, the least
    significant first: each pass a bitonic network over the key packed above its
    position, along the axis padded to a power of two [N], then the elements
    gathered once: O(K N log² N) for K keys. A map with a padded load reads it
    as an assembly of its operand padded, each element it keeps at its padded
    position and the fill elsewhere, viewed through its windows: O(padded
    elements) per load; a reduction or a scan with a padded load expands into
    such a map. A fold is maps, each adding a run of at most 14 taps in their C
    order to the running sum, the first from [+0]: a tap is the fold's operand
    at its window indices, read by the padded load that is the adjoint of the
    fold's along each axis, so each element sums as {!Nx_kernel.S.fold}
    states: O(taps / 14) passes over the result. A transform computes on float
    pairs, a complex value's real parts then its imaginary parts, one axis after
    another: along an axis of a power of two [n], Stockham's autosort, [log₂ n]
    maps; along another, Bluestein's chirp, a convolution by three such
    transforms at the power of two [M >= 2n - 1]. [R2c] gives bins [0] to [n/2]
    of its operand with [+0] imaginary parts, and [C2r] completes its last
    axis's bins from their conjugates and keeps the real parts. With [Cos] and
    [Sin] within two units in the last place, a transform [y] of [N] points of
    [x] is within [‖ŷ - y‖₂ <= 20 u P √N ‖x‖₂], [P] the sum over its axes of
    [log₂ n] for a power of two and [3 log₂ M + 2] otherwise, [u] the parts'
    unit roundoff, and a [C2r]'s [x] its completed bins. A routine has no
    expansion yet. Every other operation is core. *)

val base : Nx_array.Dtype.any -> bool
(** [base dt] is [true] for the dtypes every library's kernels compute: float32,
    float64, the 8- to 64-bit integers and bool. *)

val kept : Nx_array.Dtype.any -> Nx_array.Dtype.any
(** [kept d] is the dtype a selection, a constant or a copy of [d] computes in
    where a library declines [d]: [d] itself for a base dtype; for a float of 8
    or 16 bits, the unsigned integer of its width, over the same bits; for a
    sub-byte dtype, its byte-wide accumulator, which holds each of its codes and
    gives it back; [d] itself otherwise. *)

val single : Nx_kernel.Prog.t -> Nx_kernel.Prog.node option
(** [single p] is [Some n] where [p] is the one node [n] over its operands in
    order, which a kernel computes alone: [p]'s one output is its last node
    [n], every node before it is [In i] at index [i], and [n] is a [Const], a
    [Coord], or a kind of [m] operands reading nodes [0] to [m - 1] in order,
    [m] nodes before it. [None] otherwise: a map of [p] expands. *)

val run :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  'r Value.prim ->
  'r option
(** [run apply ~by op] is [Some r], [r] [op]'s expansion applied through
    [apply], or [None] for a core case. *)
