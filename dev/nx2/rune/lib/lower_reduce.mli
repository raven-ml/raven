(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reductions, scans and contractions as tolk graphs.

    Each function takes nodes as {!Lower_arith} holds elements and computes what
    {!Nx_kernel.reduction} and {!Nx_kernel.S.contract} state, to the class each
    states:

    - {b Extremes} give nx's bits: [-0] below [+0], any and all on booleans,
      complex numbers by their real parts, then their imaginary parts.
    - {b Integer sums and products} wrap, exactly.
    - {b Float sums and products} add or multiply their terms in the accumulator
      in an association tolk chooses: a sum of [n] terms is within
      [γ(n - 1) Σ|x|] of the exact one, a product within a relative [γ(n - 1)],
      barring overflow and underflow, [γ(k) = k u / (1 - k u)] for the
      accumulator's unit roundoff [u]. A sum whose terms are all [-0] is [+0];
      the sum of no term is [+0] and the product [1]. A complex sum adds each
      part as a float sum.
    - {b NaN results} give nx's bits in every fold, on every target, as
      {!Nx_kernel.reduction} states: where a term is a NaN, the first such term
      in C order of the reduced indices, quieted; with none, the quiet NaN of a
      clear sign bit. To an extreme, a complex number with a NaN part is a NaN,
      quieted part by part, and a scan's carry is a term before the slice's
      first. To a sum or a product, a NaN carry counts as made from numbers. *)

open Tolk

(** The type for the monoids a graph folds with directly. nx's others,
    [Logsumexp], [Moments] and [Arg], lower through their expansions. *)
type monoid = Sum | Prod | Max | Min

val reduce :
  monoid ->
  int list ->
  src:Nx_array.Dtype.any ->
  Nx_array.Dtype.any ->
  Ops.t ->
  Ops.t
(** [reduce m axes ~src dt x] is the node [x], a value of [src], folded by [m]
    along [axes] in [src]'s carrier ({!Nx_kernel.carrier}) and rounded once to
    [dt]. The result drops [axes].

    Raises [Invalid_argument] for a complex [Prod]. *)

val scan :
  monoid ->
  int ->
  carry:Ops.t option ->
  src:Nx_array.Dtype.any ->
  Nx_array.Dtype.any ->
  Ops.t ->
  Ops.t
(** [scan m axis ~carry ~src dt x] is the inclusive running fold by [m] of the
    node [x], a value of [src], along [axis], in [src]'s carrier and rounded
    once to [dt]. With [Some c], each slice starts from its element of [c], a
    value of the carrier of extent [1] along [axis].

    Raises [Invalid_argument] for a complex [Prod], and a [complex128] [Max] or
    [Min]. *)

val contract :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  acc:Nx_array.Dtype.any ->
  out:Nx_array.Dtype.any ->
  Nx_array.Dtype.any * Ops.t ->
  Nx_array.Dtype.any * Ops.t ->
  Ops.t option ->
  Ops.t
(** [contract ~batch ~contracting ~acc ~out (da, a) (db, b) init] is the
    contraction of [a], a value of [da], with [b], a value of [db], by the pairs
    [batch] and [contracting], as {!Nx_kernel.S.contract} states: each operand
    stored into [acc], each product in [acc], their sum along the contracted
    axes and [init], a value of [out] where present, added in [acc], then
    rounded once to [out]. An integer [acc] wraps exactly. A float [acc]'s
    result is within [γ(K + 1) (|init| + Σ|a·b|)] of the exact one for [K]
    contracted terms, barring overflow and underflow: tolk may fuse a product
    into the sum that takes it. A complex [acc]'s real part is within
    [γ(K + 2) (|init| + Σ(|ar br| + |ai bi|))] of the exact one, its imaginary
    part within [γ(K + 2) (|init| + Σ(|ar bi| + |ai br|))], [|init|] that
    part's: each takes two products and a difference or a sum per term. *)
