(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scans by an associative combination, as core operations applied through
    [apply]: the expansions' running reductions and Nx's recurrences, which
    no kernel offers, share it. *)

val scan :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  int array ->
  int ->
  ?first:Nx_kernel.Prog.E.any array ->
  Nx_kernel.Prog.E.any array ->
  'd Value.any array ->
  'd Value.any array
(** [scan apply ~by shape axis ?first combine states] scans [states],
    the [n] components of each term's state, values of [shape], along [axis]
    by the odd/even recursion, through [apply]. [combine] is the state of an
    earlier run of terms followed by a later one, over operands [0] to
    [n - 1], the earlier state's components, and [n] to [2n - 1], the
    later's; [first] is index [0]'s result from its state, operands [0] to
    [n - 1], the state itself if [first] is absent.

    Of [m] states [s], pairs [p_j = combine s_2j s_2j+1] for [j < ⌊m/2⌋]
    scan recursively into [q]; then [y_2j+1 = q_j], [y_0 = s_0] and
    [y_2j = combine q_j-1 s_2j]. One state is itself. So output [i] is a
    tree of at most [D_i = max(0, 2⌈log₂(i+1)⌉ - 1)] combinations over the
    states up to [i], whose shape depends on [i] alone: the scan of a prefix
    of the axis is the prefix of the scan, bit for bit. No combination lacks
    a state, so none needs an identity. A level is a map of the pairs, a
    map of the even indices past [0] and an assembly of the three parts:
    fewer than [2n] combinations and [4n] elements written along an axis of
    [n], in about [3⌈log₂ n⌉] steps. *)
