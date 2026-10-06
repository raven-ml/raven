(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tolerances and their acceptance test, which {!Tol} exports and the solves
    apply. *)

type t
(** The type for tolerances. *)

val v : rel:float -> abs:float -> t
val rel : float -> t
val abs : float -> t
val ulps : float -> t

val pp : Format.formatter -> t -> unit
(** As {!Tol}'s. *)

val scale : t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [scale t y] is [s = abs + rel * |y|] in [y]'s dtype. *)

val ratio : t -> e:(float, 'b) Nx.t -> y:(float, 'b) Nx.t -> (float, 'b) Nx.t
(** [ratio t ~e ~y] is [e / s] elementwise, [0] where [e = 0] and infinite where
    [s = 0] and [e <> 0]. *)

val zero_hint : t -> string
(** [zero_hint t] is the advice for a solve that stopped short under [t] without
    [abs]: an answer that may be zero needs one; [""] when [t] has [abs]. *)
