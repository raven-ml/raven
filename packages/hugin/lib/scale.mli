(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Data-to-unit mapping functions.

    {b Internal module.} Maps data-space values to the unit interval [[0, 1]]
    for linear, logarithmic, square-root, inverse-hyperbolic-sine, and
    symmetric-log scales. When [~invert] is [true], the mapping is reversed
    ([lo] maps to [1] and [hi] to [0]). *)

type t = {
  to_unit : float -> float;  (** [to_unit v] maps data value [v] to [[0, 1]]. *)
  from_unit : float -> float;
      (** [from_unit u] maps unit value [u] back to data space. *)
  lo : float;  (** Lower bound in data space. *)
  hi : float;  (** Upper bound in data space. *)
}
(** The type for scales. *)

val linear : ?invert:bool -> lo:float -> hi:float -> unit -> t
(** [linear ~lo ~hi ()] is a linear scale over [[lo, hi]]. *)

val log : ?invert:bool -> lo:float -> hi:float -> unit -> t
(** [log ~lo ~hi ()] is a base-10 logarithmic scale over [[lo, hi]]. *)

val sqrt : ?invert:bool -> lo:float -> hi:float -> unit -> t
(** [sqrt ~lo ~hi ()] is a square-root scale over [[lo, hi]]. Values below zero
    are clamped. *)

val asinh : ?invert:bool -> lo:float -> hi:float -> unit -> t
(** [asinh ~lo ~hi ()] is an inverse-hyperbolic-sine scale over [[lo, hi]].
    Transitions smoothly from linear near zero to logarithmic at large absolute
    values. Handles negative values. *)

val symlog :
  ?invert:bool -> linthresh:float -> lo:float -> hi:float -> unit -> t
(** [symlog ~linthresh ~lo ~hi ()] is a symmetric logarithmic scale. Linear
    within \[[-linthresh];[linthresh]\], logarithmic outside. *)

val make :
  ?invert:bool ->
  [ `Linear | `Log | `Sqrt | `Asinh | `Symlog of float ] ->
  lo:float ->
  hi:float ->
  unit ->
  t
(** [make kind ~lo ~hi ()] is a scale of the given [kind]. *)
