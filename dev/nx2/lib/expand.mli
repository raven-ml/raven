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
    along a last axis of two. [Moments] and [Arg] have no expansion yet. A
    one-node map at a dtype other than a base one ({!base}), but a cast, a
    bitcast or a copy, computes at its dtypes' accumulators, which hold their
    values exactly, and rounds once to its dtype. A selection, a constant and a
    copy move bits: they compute at the dtypes {!kept} gives. An assembly
    expands into a fill of its flat result, then per piece in order a scatter of
    the piece's elements at their flat positions, a map of coordinates: O(n) per
    piece. A gather or a scatter of a sub-byte dtype casts its values to their
    accumulator, computes there and casts the result back once; an integer's
    [Add] wraps there to the same bits as at its own dtype.

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
