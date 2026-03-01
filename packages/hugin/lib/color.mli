(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Perceptually uniform colors.

    Colors are represented internally in the
    {{:https://bottosson.github.io/posts/oklab/}OKLCH} color space. All
    operations ({!lighten}, {!darken}, {!mix}) produce perceptually uniform
    results: equal numerical steps yield equal perceived differences.

    Constructors accept common input formats (sRGB, hex) and convert to OKLCH on
    creation. The reverse conversion {!to_rgba} is called only at render time.
*)

(** {1:types Types} *)

type t
(** The type for colors in OKLCH space. Components are lightness \[0, 1\],
    chroma \[0, ~0.4\], hue \[0, 360), and alpha \[0, 1\]. *)

(** {1:constructors Constructors} *)

val oklch : l:float -> c:float -> h:float -> unit -> t
(** [oklch ~l ~c ~h ()] is the fully opaque OKLCH color with lightness [l],
    chroma [c], and hue [h] (in degrees). *)

val oklcha : l:float -> c:float -> h:float -> a:float -> unit -> t
(** [oklcha ~l ~c ~h ~a ()] is like {!oklch} with alpha [a]. *)

val rgb : r:float -> g:float -> b:float -> unit -> t
(** [rgb ~r ~g ~b ()] is the fully opaque color with sRGB components [r], [g],
    [b] in \[0, 1\], converted to OKLCH. *)

val rgba : r:float -> g:float -> b:float -> a:float -> unit -> t
(** [rgba ~r ~g ~b ~a ()] is like {!rgb} with alpha [a]. *)

val hex : string -> t
(** [hex s] is the color parsed from the hex string [s]. Accepts ["#RRGGBB"] and
    ["#RRGGBBAA"] formats.

    Raises [Invalid_argument] if [s] is not a valid hex color. *)

(** {1:accessors Accessors} *)

val lightness : t -> float
(** [lightness c] is the OKLCH lightness of [c] in \[0, 1\]. *)

val chroma : t -> float
(** [chroma c] is the OKLCH chroma of [c] in \[0, ~0.4\]. *)

val hue : t -> float
(** [hue c] is the OKLCH hue of [c] in degrees \[0, 360). *)

val alpha : t -> float
(** [alpha c] is the alpha of [c] in \[0, 1\]. *)

(** {1:converting Converting} *)

val to_rgba : t -> float * float * float * float
(** [to_rgba c] is [(r, g, b, a)] with sRGB components in \[0, 1\]. Values are
    clamped to the sRGB gamut. *)

(** {1:operations Operations} *)

val with_alpha : float -> t -> t
(** [with_alpha a c] is [c] with alpha set to [a]. *)

val lighten : float -> t -> t
(** [lighten amount c] is [c] with lightness increased by [amount], clamped to
    \[0, 1\]. *)

val darken : float -> t -> t
(** [darken amount c] is [c] with lightness decreased by [amount], clamped to
    \[0, 1\]. *)

val mix : float -> t -> t -> t
(** [mix ratio a b] is the perceptual blend of [a] and [b]. [ratio] is the
    interpolation factor: [0.0] gives [a], [1.0] gives [b]. Hue is interpolated
    along the shortest arc. *)

(** {1:named Named colors}

    The default named colors follow the
    {{:https://jfly.uni-koeln.de/color/}Okabe-Ito} palette, designed to be
    distinguishable under all forms of color-vision deficiency. *)

val orange : t
val sky_blue : t
val green : t
val yellow : t
val blue : t
val vermillion : t
val purple : t
val black : t
val white : t
val gray : t

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats the color as [oklch(L C H / A)]. *)
