(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tolerances.

    A solve accepts a lane when its error estimate [e] at its value [y] meets
    the tolerance: the root mean square, over the lane's float components, of
    [e_i / s_i] is at most [1], with [s_i = abs + rel * |y_i|]. A component with
    [e_i = 0] counts [0]; one with [s_i = 0] and [e_i <> 0] is never accepted,
    so an answer that may be zero needs [abs]. Each solve says what its [e] and
    [y] are.

    The components add in a fixed order, pairwise over the flattened components,
    so a decision depends only on the values of the user's function, eagerly and
    compiled. Tolerances are OCaml floats: constants of a compiled program. *)

type t
(** The type for tolerances. *)

val v : rel:float -> abs:float -> t
(** [v ~rel ~abs] is the tolerance [abs + rel * |y|].

    Raises [Invalid_argument] if either is negative or not finite, or if both
    are [0]. *)

val rel : float -> t
(** [rel r] is [v ~rel:r ~abs:0.]. *)

val abs : float -> t
(** [abs a] is [v ~rel:0. ~abs:a]. *)

val ulps : float -> t
(** [ulps k] is [rel (k *. eps)] with [eps] the distance from [1] to the next
    float of the dtype the tolerance meets. It is met only where the problem
    determines its answer to [k] units: a zero of [f] to about [f]'s rounding
    error over [|f'|], a minimum to about the square root of [f]'s rounding over
    its curvature.

    Raises [Invalid_argument] if [k] is not finite and positive. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] formats [t] as ["rel 1e-06 abs 1e-10"] or ["ulps 4"]. *)

(**/**)

val scale : t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [scale t y] is [s = abs + rel * |y|] in [y]'s dtype. *)

val ratio : t -> e:(float, 'b) Nx.t -> y:(float, 'b) Nx.t -> (float, 'b) Nx.t
(** [ratio t ~e ~y] is [e / s] elementwise, [0] where [e = 0] and infinite where
    [s = 0] and [e <> 0]. *)

(**/**)
