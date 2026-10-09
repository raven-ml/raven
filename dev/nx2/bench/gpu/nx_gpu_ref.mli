(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host reference of the GPU suites and benches. *)

type view = {
  bytes : string;  (** The operand's memory, from its first element. *)
  dtype : int;  (** Its dtype's code. *)
  strides : int array;  (** Its three strides, in elements. *)
}
(** The type for operands of three axes: element [(z, r, c)] at
    [z * strides.(0) + r * strides.(1) + c * strides.(2)]. *)

type result = {
  worst : float;
      (** For a float accumulator, the largest ratio of an output's error to its
          allowance: within the bound iff at most [1]. *)
  wrong : int;
      (** The outputs that differ from an exact answer: an integer accumulator's
          sum, or the NaN or infinity of a float sum with a NaN or an infinity
          among its terms; and the NaNs of finite float sums. *)
  at : int;
      (** The first wrong output, else the worst one, as [z·m·n + i·n + j], or
          [-1] if every output is exact. *)
}
(** The type for a check's outcome. *)

val contract :
  a:view ->
  b:view ->
  ?init:view ->
  y:view ->
  batch:int ->
  m:int ->
  n:int ->
  k:int ->
  acc:int ->
  ?flush:bool ->
  samples:int ->
  unit ->
  result
(** [contract ~a ~b ~init ~y ~batch ~m ~n ~k ~acc ~flush ~samples ()] checks
    [samples] outputs [(z, i, j)] (every output if there are fewer) of
    [y = init + a·b], [a] [batch × m × k], [b] [batch × n × k], against the
    exact [s = init(z, i, j) + Σ_q a(z, i, q) · b(z, j, q)].

    For a float [acc], an output's allowance is the bound
    [γ(k + 1, 2u) (|init| + Σ|a||b|)], [u] [acc]'s unit roundoff, plus half a
    unit in the last place of [y]'s dtype; with [flush] (defaults to [false]),
    for a device whose arithmetic reads and writes subnormals as zero,
    [2^-126 (1 + Σ(1 + |a| + |b|))] more. A NaN or an infinity among the terms
    asks for IEEE's answer (any NaN for NaN), and an [s] that rounds past [y]'s
    range its infinity. An integer or bool [y] of a float [acc] lies between the
    casts of the ends of the sums the bound allows. For an integer [acc], [y]
    holds [s] wrapped to [acc], then cast from [acc] to [y]'s dtype: widened by
    [acc]'s sign and wrapped, rounded once to a float, or nonzero for bool.

    [ref.h]'s [nx_ref_contract] is the same check for C callers. *)
