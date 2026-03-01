(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Colormaps.

    A colormap is a continuous mapping from \[[0];[1]\] to {!Color.t}.
    Internally stored as a 256-entry lookup table with OKLCH interpolation, so
    {!eval} is a single array access. *)

(** {1:types Types} *)

type t
(** The type for colormaps. *)

(** {1:eval Evaluation} *)

val eval : t -> float -> Color.t
(** [eval cmap v] is the color at position [v], clamped to \[[0];[1]\]. *)

(** {1:constructors Constructors} *)

val of_colors : Color.t array -> t
(** [of_colors stops] is a colormap interpolating linearly through [stops] in
    OKLCH space. The stops are evenly spaced from [0] to [1].

    Raises [Invalid_argument] if [stops] has fewer than 2 elements. *)

(** {1:predefined Predefined colormaps}

    Perceptually uniform sequential colormaps from the
    {{:https://bids.github.io/colormap/}viridis family}, plus a diverging
    colormap. *)

val viridis : t
val plasma : t
val inferno : t
val magma : t
val cividis : t
val coolwarm : t

val gray : t
(** Linear grayscale (black to white). *)

val gray_r : t
(** Reversed grayscale (white to black). The standard default for astronomical
    image display. *)

val hot : t
(** Black-red-yellow-white. Common in X-ray astronomy. *)
