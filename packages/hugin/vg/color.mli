(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Colors.

    A color is plain sRGB data with straight (non-premultiplied) alpha.
    Renderers composite colors with source-over and clamp components to
    \[[0];[1]\]. *)

type t = { r : float; g : float; b : float; a : float }
(** The type for colors. [r], [g] and [b] are sRGB components and [a] the
    opacity, each in \[[0];[1]\]. *)

val v : ?a:float -> float -> float -> float -> t
(** [v ~a r g b] is the color with the given components. [a] defaults to [1.],
    fully opaque. *)

val black : t
(** [black] is [v 0. 0. 0.]. *)

val white : t
(** [white] is [v 1. 1. 1.]. *)

val transparent : t
(** [transparent] is [v ~a:0. 0. 0. 0.]. Drawing with it has no effect. *)
