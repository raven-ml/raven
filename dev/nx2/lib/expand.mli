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
    expands into a map of each output it reduces into its accumulator, the
    output's carrier ({!Nx_kernel.carrier}), the plain loop, and a cast to the
    result's dtype. A sum of complex numbers sums their parts, read as floats
    along a last axis of two.

    [Logsumexp], [Moments] and [Arg] reduce through core loops, in the
    accumulator for the first two and in the output's dtype for [Arg]. A
    [Logsumexp] is a [Max] [m], [m'] its finite elements and [0] elsewhere, a
    [Sum] of [exp (x - m')], an [exp] of less than [-87] (float32) or [-708]
    (float64) taken as [0], then [m' + log s], or [m] where it is a NaN; with
    no term, [log 0]: nx.cpu's two passes ({!Nx_cpu}), and their bits. A
    float narrower than its accumulator rounds into its own dtype by a cast,
    which keeps a NaN's sign and payload. [Moments] are the mean [Σ x / n],
    then two sums of
    [d = x - mean] and the variance [(Σ d² - Σ d (Σ d / n)) / n], [+0] where
    that is negative or [Σ d²] where that is [+∞], each the mean where it is
    a NaN. An [Arg] is
    the [Max] or [Min] [m], then a [Min] of the positions, numbered in C order
    of the reduced indices, whose term has [m]'s bits or, where [m] is a NaN, is
    a NaN. Where a float sum of [n] terms is within [γ(n - 1) Σ|x|] of the exact
    sum ({!Nx_kernel.reduction}), and barring overflow and underflow, a
    [Logsumexp] [r] of [n] terms whose maximum is finite is within
    [γ(2n) + u|r| + ε] of the exact one, [ε] the error [Exp] and [Log] add; the
    mean within [γ(n) M], [M = Σ|x| / n], and the variance [V] within
    [(γ(n + 4) + 2γ(n)²) V + 2γ(n + 6) γ(n) M √V + 4γ(n + 4) γ(n)² M²], for
    [n] exact in the accumulator. A scan of [Logsumexp], [Moments] or [Arg],
    the first two in the accumulator, combines its terms' states by {!Pairwise.scan}: output [i]
    through at most [D_i = max(0, 2⌈log₂(i+1)⌉ - 1)] combinations, the
    earlier state first, O(n) along an axis of [n]. [Logsumexp] combines [a]
    and [b] into [M + log1p (exp (m - M))], [M] the two's [Maximum] and [m]
    their [Minimum], or [M] where [M] is not finite; index [0] is its term
    plus [+0]. Its result at [i] is within [D_i (u (X + ln(i+1) + 1) + ε)]
    of the exact one, [X] the largest finite magnitude among the terms up to
    [i] and [ε] the error [Exp] and [Log1p] add, to first order, barring
    overflow, for [i + 1] exact in the accumulator. [Moments] scans as
    {!Nx_kernel.reduction} states, in the accumulator. [Arg] combines pairs
    of an extreme and its position, keeping the earlier pair unless the
    later extreme is past it in [Max]'s or [Min]'s order: exact. A NaN
    result of these scans, carried through the combinations with its bits,
    is then quieted, as the kernels give it.

    A one-node map at a dtype other than a base one ({!Nx_kernel.base}), but a
    cast, a bitcast or a copy, computes at its dtypes' accumulators, which hold
    their values exactly, and rounds once to its dtype. A selection, a constant
    and a copy move bits: they compute at the dtypes {!kept} gives; a [Copy] is
    the map of one copy. An assembly expands into a fill of its flat result,
    then per piece in order a scatter of the piece's elements at their flat
    positions, a map of coordinates: O(n) per piece. A gather, a scatter or a
    sort of a sub-byte dtype casts its values to their accumulator, computes
    there and casts the result back once; an integer's [Add] wraps there to the
    same bits as at its own dtype. Any other scatter whose targets may repeat is
    one unique scatter per position along its axis, in order, each of the target
    so far: O(m |into|) for m positions. A sum associates left to right; a
    narrow float's sums in its accumulator, rounded once, and a [Set] of [true]
    at the same targets keeps the bits of each target no update reaches: twice
    the scatters.

    A contraction whose accumulator is not its output's dtype first runs as a
    contraction into the accumulator, then casts to the output. Any other
    expands into a map of each pair of elements multiplied in the accumulator,
    a sum over the contracted axes, and the [init] added.
    A sort of another dtype sorts positions by its order
    keys, int64 values of [b = 63 - ⌈log₂ n⌉] bits along an axis of [n]
    elements, [⌈w / b⌉] for an integer or a float of [w] bits and per part for
    a complex number, the least significant first: each pass a bitonic network
    over the key packed above its position in an int64, along the axis padded
    to a power of two [N], then the elements gathered once: O(K N log² N) for
    K keys. A map with a padded load reads it
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

val kept : Nx_array.Dtype.any -> Nx_array.Dtype.any
(** [kept d] is the dtype a selection, a constant or a copy of [d] computes in
    where a library declines [d]: [d] itself for a base dtype
    ({!Nx_kernel.base}); for a float of 8 or 16 bits, the unsigned integer of
    its width, over the same bits; for a sub-byte dtype, its byte-wide
    accumulator, which holds each of its codes and gives it back; [d] itself
    otherwise. *)

val single : Nx_kernel.Prog.t -> Nx_kernel.Prog.node option
(** [single p] is [Some n] where [p] is the one node [n] over its operands in
    order, which a kernel computes alone: [p]'s one output is its last node
    [n], every node before it is [In i] at index [i], and [n] is a [Const], a
    [Coord], or a kind of [m] operands reading nodes [0] to [m - 1] in order,
    [m] nodes before it. [None] otherwise: a map of [p] expands. *)

val quieted :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t
(** [quieted apply ~by x] is [x] with each NaN of a float, or NaN part of a
    complex number, quieted, as the kernels return a NaN term: its quiet bit,
    the top bit of its fraction, set, its sign and the rest of its payload
    kept. Other dtypes are [x]. One map; a complex [x] is read as its parts,
    so it needs an axis below the greatest rank. *)

val sort :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  bits:int ->
  int ->
  bool ->
  int option ->
  ('v, 's, 'd) Value.t ->
  (('v, 's, 'd) Value.t * 'd Value.index) option
(** [sort apply ~by ~bits axis descending k x] is the expansion of
    [Sort { axis; descending; k; x }] by order keys of [bits] bits, for [bits]
    in \[[1], [63 - ⌈log₂ n⌉]\] along an axis of [n] elements, so that a key
    packed above its position fits an int64. {!run} sorts at the width [b]
    stated above. *)

val square_mod :
  int ->
  (int64, Nx_array.Dtype.int64_elt) Nx_kernel.Prog.E.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_kernel.Prog.E.t
(** [square_mod m t] is [t² mod m] at the int64 [t], exact for [t] in
    \[[0], [2{^63}]) and [m] in \[[1], [2{^54}]\]: the moduli [2n] of
    Bluestein's chirp along an axis of [n] elements. For [m] up to [2{^31}] it
    is [(t mod m)² mod m]; past it, a product by Horner's rule over digits of
    [t mod m] small enough that no step leaves int64. *)

val run :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  'r Value.prim ->
  'r option
(** [run apply ~by op] is [Some r], [r] [op]'s expansion applied through
    [apply], or [None] for a core case. No result of [r] is one of [op]'s
    operands: one that would be is its copy. *)

