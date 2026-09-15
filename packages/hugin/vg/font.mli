(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fonts.

    A font is a TrueType face with glyph outlines; the size is chosen at each
    use, as the em height in the picture's units. Two faces of
    {{:https://rsms.me/inter/}Inter} are bundled, {!regular} and {!bold}, so
    text renders identically on every machine without system fonts. Other
    TrueType files load with {!of_string}.

    Text is laid out left to right, one glyph per Unicode character, with pair
    kerning applied. Characters the font lacks use its fallback glyph. Measure
    text with {!advance} and {!bounds}, draw it with {!Picture.text}, and reach
    the glyph level with {!glyphs} and {!glyph_path} when a renderer or a caller
    needs outlines. *)

(** {1:types Types} *)

type t
(** The type for fonts. *)

val regular : t
(** [regular] is the bundled sans-serif face, Inter at weight 400. *)

val bold : t
(** [bold] is the bundled bold sans-serif face, Inter at weight 700. *)

(** {1:loading Loading} *)

(** The type for loading errors. The string says which table or feature is at
    fault. *)
type error =
  | Unsupported of string  (** A table or outline format that is not handled. *)
  | Malformed of string  (** Bytes that are not a TrueType font. *)

val of_string : string -> (t, error) result
(** [of_string s] is [Ok f] if [s] is a TrueType file with [glyf] outlines and a
    Unicode character map, and [Error e] otherwise. OpenType fonts with CFF
    outlines are [Unsupported]. Kerning is read from the [GPOS] pair positioning
    of the [kern] feature; a font without it lays out with plain advances. *)

val pp_error : Format.formatter -> error -> unit
(** [pp_error fmt e] formats [e] for users. *)

(** {1:identity Identity}

    Renderers use these to name and embed a font in a document. *)

val family : t -> string
(** [family f] is [f]'s family name from its [name] table, or [""] if it has
    none. *)

val weight : t -> int
(** [weight f] is [f]'s CSS weight from its [OS/2] table, [400] for regular and
    [700] for bold. *)

val bytes : t -> string
(** [bytes f] is the TrueType file [f] was loaded from. *)

(** {1:metrics Metrics}

    Metrics are in the picture's units for a given [size], the em height. They
    are exact for the glyph outlines, without hinting. *)

val ascent : t -> size:float -> float
(** [ascent f ~size] is the distance from the baseline up to the top of the em
    box. *)

val descent : t -> size:float -> float
(** [descent f ~size] is the distance from the baseline down to the bottom of
    the em box, as a positive number. *)

val advance : t -> size:float -> string -> float
(** [advance f ~size s] is the horizontal distance the pen moves when drawing
    [s], kerning included. It is [0.] for the empty string. *)

val bounds : t -> size:float -> string -> Box.t option
(** [bounds f ~size s] is the box enclosing the ink of [s] drawn with its origin
    at [(0, 0)] on the baseline, y down, so ink above the baseline has negative
    [y]. It is [None] when [s] has no ink, such as the empty string or
    whitespace. *)

(** {1:glyphs Glyphs}

    Glyph ids are indices into [f]'s glyph table; [0] is the fallback glyph. *)

type glyph = { id : int; x : float; advance : float }
(** The type for laid out glyphs: the glyph [id], the pen [x] offset at which it
    is drawn and its own [advance]. The kerning with the next glyph is that
    glyph's [x] minus [x +. advance]. *)

val glyphs : t -> size:float -> string -> glyph list
(** [glyphs f ~size s] are the glyphs of [s] in drawing order, positioned from
    an origin at [x = 0.]. *)

val glyph_path : t -> size:float -> int -> Path.t
(** [glyph_path f ~size g] is the outline of glyph [g] at [size] with its origin
    at [(0, 0)] on the baseline, y down. Glyphs without ink give {!Path.empty}.
    Fill it under the nonzero rule. *)
