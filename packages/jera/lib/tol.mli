(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tolerances: the implementation of {!Jera.Tol}, with the scale and the ratio
    that the solves' acceptance tests compute from one. *)

type t

val v : rel:float -> abs:float -> t
val rel : float -> t
val abs : float -> t
val ulps : float -> t
val pp : Format.formatter -> t -> unit

val scale : t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [scale t y] is the scale [s] of each element of [y]: [abs + rel * |y|], or
    [k * eps * |y|] for [ulps k] with [eps] that of [y]'s dtype. *)

val ratio :
  t -> e:(float, 'b, 'd) Nx.t -> y:(float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [ratio t ~e ~y] is [|e / scale t y|] elementwise: [0] where [e = 0], and
    infinite where the scale is [0] and [e] is not. *)

val zero_hint : t -> string
(** [zero_hint t] is the sentence a failure report appends to say that an answer
    that may be zero needs [abs], or [""] if [t] has a positive [abs]. *)
