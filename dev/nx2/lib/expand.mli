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
    sum ({!Nx_kernel.Spec.reduction}), and barring overflow and underflow, a
    [Logsumexp] [r] of [n] terms whose maximum is finite is within
    [γ(2n) + u|r| + ε] of the exact one, [ε] the error [Exp] and [Log] add; the
    mean within [γ(n) Σ|x| / n], and the variance [V] within
    [γ(n + 3) V + (1 + γ(n + 3)) γ(n)² (Σ|x| / n)²], for [n] exact in the
    accumulator. A scan of [Logsumexp] or [Arg] has no expansion yet.

    A one-node map at a dtype other than a base one ({!base}), but a cast, a
    bitcast or a copy, computes at its dtypes' accumulators, which hold their
    values exactly, and rounds once to its dtype. A selection, a constant and a
    copy move bits: they compute at the dtypes {!kept} gives. An assembly
    expands into a fill of its flat result, then per piece in order a scatter of
    the piece's elements at their flat positions, a map of coordinates: O(n) per
    piece. A gather or a scatter of a sub-byte dtype casts its values to their
    accumulator, computes there and casts the result back once; an integer's
    [Add] wraps there to the same bits as at its own dtype. Any other scatter
    whose targets may repeat is one unique scatter per position along its axis,
    in order, each of the target so far: O(m |into|) for m positions. A sum
    associates left to right.

    A contraction whose accumulator is a float other than its output's dtype
    first runs as a contraction into the accumulator, then casts to the output.
    Any other expands into a map of each pair of elements multiplied in the
    accumulator, a sum over the contracted axes, the [init] added, and a cast to
    the output's dtype. Every other operation is core. *)

val base : Nx_array.Dtype.any -> bool
(** [base dt] is [true] for the dtypes every library's kernels compute: float32,
    float64, the 8- to 64-bit integers and bool. *)

val kept : Nx_array.Dtype.any -> Nx_array.Dtype.any
(** [kept d] is the dtype a selection, a constant or a copy of [d] computes in
    where a library declines [d]: [d] itself for a base dtype; for a float of 8
    or 16 bits, the unsigned integer of its width, over the same bits; for a
    sub-byte dtype, its byte-wide accumulator, which holds each of its codes and
    gives it back; [d] itself otherwise. *)

val run :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  'r Value.prim ->
  'r option
(** [run apply ~by op] is [Some r], [r] [op]'s expansion applied through
    [apply], or [None] for a core case. *)
