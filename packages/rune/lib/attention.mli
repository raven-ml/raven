(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {!Nx.attention}'s derivative, as its construct's default computes it.

    At a call's operands and its [o] and [lse], the derivative is linear in
    the tangents of the query, keys, values and the pair program's float
    loads, and its transpose maps cotangents of [o] and [lse] back to them.
    A pair's weight is [p = exp (x - lse)], [0] where hidden; [ẋ] is the
    pair program's tangent. Along tangents, [l̇se = Σ p ẋ] and
    [ȯ = Σ p ẋ v - l̇se o + Σ p v̇]. Transposed, [δ = ō·o - l̄se] and each
    pair's score receives [p (ō·v - δ)], which the program's transpose
    reads back to the scaled score and each load. A row of no visible pair
    has zero weights and zero cotangents. *)

type spec = {
  prog : Nx_kernel.Prog.t;  (** The call's pair program. *)
  scale : float;
  paged : bool;  (** Whether the keys and values are pools read by a table. *)
  lengths : bool;  (** Whether the call has lengths. *)
  tangents : bool array;
      (** For each load, whether it has a tangent: a float load. *)
}
(** The type for what a derivative knows of its call beside its values. *)

val linear :
  spec -> transposed:bool -> 'd Nx.Prim.any array -> 'd Nx.Prim.any array ->
  'd Nx.Prim.any array
(** [linear spec ~transposed data args] is the derivative at [data] — the
    query, keys and values, the table and the lengths where [spec] has
    them, the loads, then [o] and [lse] — applied to [args]: the tangents
    of the query, keys, values and each load [spec.tangents] marks, giving
    [ȯ] and [l̇se]; where [transposed], cotangents of [o] and [lse], giving
    the query's, keys', values' and each marked load's. Keys' and values'
    are the pools' where paged. It computes as contractions and maps of the
    whole pair loop, by {!Nx.Prim.eval}.

    Raises [Invalid_argument] if [data] or [args] are not of these dtypes
    and devices. *)
