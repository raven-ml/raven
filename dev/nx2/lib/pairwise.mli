(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scans by an associative combination, as simpler operations applied
    through the [apply] {!Make} takes: the expansions' running reductions and
    Nx's recurrences, which no kernel offers, share it. *)

module Make : Value.Apply -> sig
  val circuit :
    by:string ->
    int array ->
    ('d Value.any array -> 'd Value.any array -> 'd Value.any array) ->
    'd Value.any array ->
    'd Value.any array
  (** [circuit ~by axes combine states] scans [states], the components
      of each term's state, along [axes.(k)] for component [k], by the
      odd/even recursion, through [apply]. Every component has the same
      extent [n] along its axis; shapes may differ otherwise. [combine e l] is
      the state of an earlier run of terms [e] followed by a later one [l],
      computed index by index, each of its components of [e]'s shape.

      Of [m] states [s], pairs [p_j = combine s_2j s_2j+1] for [j < ⌊m/2⌋]
      scan recursively into [q]; then [y_2j+1 = q_j], [y_0 = s_0] and
      [y_2j = combine q_j-1 s_2j]. One state is itself. So output [i] is a
      tree of at most [D_i = max(0, 2⌈log₂(i+1)⌉ - 1)] combinations over the
      states up to [i], whose shape depends on [i] alone: the scan of a prefix
      of the axis is the prefix of the scan, bit for bit. No combination lacks
      a state, so none needs an identity. A level calls [combine] twice, on the
      pairs and on the even indices past [0], and assembles the three parts:
      at most [2⌈log₂ n⌉] calls, combining fewer than [2n] pairs of terms. *)

  val scan :
    by:string ->
    int array ->
    int ->
    Nx_kernel.Prog.E.any array ->
    'd Value.any array ->
    'd Value.any array
  (** [scan ~by shape axis combine states] is {!circuit} over [states]
      of [shape] along [axis], each combination one map of the program
      [combine] over operands [0] to [n - 1], the earlier state's [n]
      components, and [n] to [2n - 1], the later's. A level writes fewer than
      [4n] elements along an axis of [n], in about [3⌈log₂ n⌉] steps. *)
end
