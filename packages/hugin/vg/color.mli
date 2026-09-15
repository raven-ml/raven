(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Colors.

    Colors are plain sRGB data with straight (non-premultiplied) alpha. *)

type t = { r : float; g : float; b : float; a : float }
(** The type for colors. Components are in \[0;1\]. *)

val v : ?a:float -> float -> float -> float -> t
(** [v ~a r g b] is the color with the given components. [a] defaults to [1.].
*)

val black : t
val white : t
val transparent : t
