(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Colours.

    A colour is a point of the sRGB gamut with an opacity, stored as the numbers
    every output format consumes: three {e encoded} sRGB components, the values
    of a CSS hex colour or of an 8-bit PNG pixel scaled to \[[0];[1]\], and a
    straight (not premultiplied) alpha. Renderers composite in encoded sRGB with
    source-over, as browsers and PDF viewers do, so a colour looks the same in
    raster, SVG and PDF output.

    Perceptual work happens in Oklab, where Euclidean distance approximates
    perceived difference and [0.02] is about the smallest visible step. {!mix}
    interpolates there, and {!of_oklab} and {!of_oklch} bring coordinates that
    lie outside the gamut into it by {{!section-gamut}gamut mapping}. Oklch is
    Oklab in polar form: lightness, chroma and hue.

    References:
    - Björn Ottosson.
      {{:https://bottosson.github.io/posts/oklab/}A perceptual color space for
       image processing}. 2020.
    - W3C. {{:https://www.w3.org/TR/css-color-4/}CSS Color Module Level 4}. *)

(** {1:types Types} *)

type t
(** The type for colours. Invariant: the encoded components and the alpha are in
    \[[0];[1]\]. *)

(** {1:constructors Constructors and constants} *)

val v : ?alpha:float -> float -> float -> float -> t
(** [v ~alpha r g b] is the colour with encoded sRGB components [r], [g] and [b]
    and opacity [alpha], which defaults to [1.].

    Raises [Invalid_argument] if an argument is not in \[[0];[1]\], [nan]
    included. *)

val gray : ?alpha:float -> float -> t
(** [gray ~alpha l] is [v ~alpha l l l]. *)

val black : t
(** [black] is [v 0. 0. 0.]. *)

val white : t
(** [white] is [v 1. 1. 1.]. *)

val red : t
(** [red] is [v 1. 0. 0.], the sRGB red primary. *)

val green : t
(** [green] is [v 0. 1. 0.], the sRGB green primary, which CSS names [lime]. *)

val blue : t
(** [blue] is [v 0. 0. 1.], the sRGB blue primary. *)

val transparent : t
(** [transparent] is [v ~alpha:0. 0. 0. 0.]. Drawing with it has no effect. *)

(** {1:accessors Accessors} *)

val r : t -> float
(** [r c] is the encoded red component of [c]. *)

val g : t -> float
(** [g c] is the encoded green component of [c]. *)

val b : t -> float
(** [b c] is the encoded blue component of [c]. *)

val alpha : t -> float
(** [alpha c] is the opacity of [c], from [0.] (transparent) to [1.] (opaque).
*)

val with_alpha : float -> t -> t
(** [with_alpha a c] is [c] with opacity [a].

    Raises [Invalid_argument] if [a] is not in \[[0];[1]\]. *)

(** {1:oklab Oklab and Oklch}

    Conversions to Oklab and Oklch drop the alpha; conversions from them take it
    as an optional [alpha], which defaults to [1.] and must lie in \[[0];[1]\].

    {2:gamut Gamut mapping}

    Oklab coordinates outside the sRGB gamut are mapped into it as CSS Color 4
    specifies. A lightness at or above [1.] gives white and one at or below [0.]
    black. Otherwise the chroma is reduced, at constant lightness and hue, until
    clipping each component to \[[0];[1]\] moves the colour by less than [0.02]
    in Oklab, and the clipped colour is the result. Unlike clipping alone, this
    keeps the hue. *)

val to_oklab : t -> float * float * float
(** [to_oklab c] is [(l, a, b)], the Oklab coordinates of [c]: the lightness [l]
    from [0.] for black to [1.] for white, and [a] and [b], which run from green
    to red and from blue to yellow, within about \[[-0.4];[0.4]\] for sRGB
    colours. *)

val of_oklab : ?alpha:float -> float -> float -> float -> t
(** [of_oklab ~alpha l a b] is the colour at Oklab coordinates [(l, a, b)],
    {{!section-gamut}gamut mapped}.

    Raises [Invalid_argument] if [l], [a] or [b] is not finite or [alpha] is not
    in \[[0];[1]\]. *)

val to_oklch : t -> float * float * float
(** [to_oklch c] is [(l, c, h)] for the Oklab coordinates [(l, a, b)] of [c]:
    the lightness [l], the chroma [c = Float.hypot a b] and the hue
    [h = Float.atan2 b a] in radians, in \[[0];[2π]\[. The hue is [nan] iff the
    chroma is below [4e-6]: such a colour is grey up to rounding, and its hue is
    {e powerless}, carrying no information. *)

val of_oklch : ?alpha:float -> float -> float -> float -> t
(** [of_oklch ~alpha l c h] is [of_oklab ~alpha l (c *. cos h) (c *. sin h)],
    where a [nan] hue, a powerless one, counts as [0.].

    Raises [Invalid_argument] if [l] or [c] is not finite, [c] is negative, [h]
    is infinite, or [alpha] is not in \[[0];[1]\]. *)

(** {1:mixing Mixing and contrast} *)

val mix : float -> t -> t -> t
(** [mix t c c'] is the colour a fraction [t] of the way from [c] to [c'] in
    Oklab, interpolated with premultiplied alpha as CSS Color 4 does: the
    opacity is [(1 - t) αc + t αc'], and each Oklab coordinate is the average of
    those of [c] and [c'] weighted by [(1 - t) αc] and [t αc'], so that a
    transparent end adds opacity and no colour. A zero opacity gives
    {!transparent}; any other result is {{!section-gamut}gamut mapped}. Since
    Oklab is Cartesian, a mix of white and blue passes through light blues and
    no other hue. [mix] does not transform its last argument: [c |> mix t c'] is
    a fraction [t] of the way from [c'] to [c]. Results are exact only up to the
    round trip through Oklab: [mix 0. c c'] may differ from [c] in the last bits
    of its components.

    Raises [Invalid_argument] if [t] is not in \[[0];[1]\]. *)

val contrast : t -> t
(** [contrast c] is {!black} or {!white}, whichever has the higher WCAG 2
    contrast ratio against [c]: {!black} iff the relative luminance of [c],
    [0.2126 R + 0.7152 G + 0.0722 B] for its linear components [R], [G] and [B],
    is at least [sqrt 0.0525 - 0.05], about [0.179], where both ratios are
    equal. The alpha of [c] is ignored. *)

(** {1:hex Hexadecimal notation} *)

val of_hex : string -> (t, string) result
(** [of_hex s] is [Ok c] if [s] is a CSS hex colour, [#] then 3, 4, 6 or 8
    hexadecimal digits of either case ([#rgb], [#rgba], [#rrggbb] or
    [#rrggbbaa]), and [Error msg] otherwise, [msg] saying why. In the short
    forms a digit [d] stands for [dd]; a pair of digits of value [n] is the
    component [n /. 255.]. Without an alpha pair the colour is opaque. *)

val to_hex : t -> string
(** [to_hex c] is [c] as [#rrggbbaa] in lowercase, each pair the component times
    [255.] rounded to the nearest integer, halves away from zero, with the alpha
    pair omitted when it is [ff]. [of_hex (to_hex c)] is [Ok c] iff every
    component of [c] is [n /. 255.] for an integer [n]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal c c'] is [true] iff [c] and [c'] have equal components and alpha. *)

val compare : t -> t -> int
(** [compare c c'] orders colours lexicographically by red, green, blue and
    alpha. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf c] formats [to_hex c]. Unlike most printers of this library its
    output is stable, and unequal colours may print the same. *)
