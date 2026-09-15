(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fonts.

    A font is a TrueType face with glyph outlines; the size is chosen at each
    use. Two faces of {{:https://rsms.me/inter/}Inter} are bundled so that text
    renders identically on every machine. *)

type t
(** The type for fonts. *)

val regular : t
(** [regular] is the bundled sans-serif face. *)

val bold : t
(** [bold] is the bundled bold sans-serif face. *)

(** {1:loading Loading} *)

type error =
  | Unsupported of string  (** A table or outline format not handled. *)
  | Malformed of string  (** The bytes are not a TrueType font. *)

val of_string : string -> (t, error) result
(** [of_string s] parses the TrueType font in [s]. Fonts with CFF outlines are
    {!Unsupported}. *)

val pp_error : Format.formatter -> error -> unit

(** {1:identity Identity} *)

val family : t -> string
(** [family f] is the font's family name, or [""] if it has none. *)

val weight : t -> int
(** [weight f] is the font's CSS weight, [400] for regular and [700] for bold.
*)

val bytes : t -> string
(** [bytes f] is the TrueType file [f] was read from, for embedding in
    documents. *)

(** {1:metrics Metrics}

    All metrics are in pixels for the given [size], the em height in pixels.
    Text is laid out left to right with kerning; unmapped characters use the
    font's fallback glyph. *)

val ascent : t -> size:float -> float
(** [ascent f ~size] is the distance from the baseline to the top of the em box.
*)

val descent : t -> size:float -> float
(** [descent f ~size] is the distance from the baseline to the bottom of the em
    box, as a positive number. *)

val advance : t -> size:float -> string -> float
(** [advance f ~size s] is the horizontal distance the pen moves when drawing
    [s]. *)

val bounds : t -> size:float -> string -> Box.t option
(** [bounds f ~size s] is the box enclosing the ink of [s] drawn with its origin
    at [(0, 0)] on the baseline, y down, or [None] for text without ink such as
    whitespace. *)

(** {1:glyphs Glyphs} *)

type glyph = { id : int; x : float; advance : float }
(** The type for laid out glyphs: a glyph id, the pen x offset at which it is
    drawn and its own advance, so the gap to the next glyph is the kerning. *)

val glyphs : t -> size:float -> string -> glyph list
(** [glyphs f ~size s] are the glyphs of [s] in order. *)

val glyph_path : t -> size:float -> int -> Path.t
(** [glyph_path f ~size g] is the outline of glyph [g] at [size], with its
    origin at [(0, 0)] on the baseline, y down. *)
