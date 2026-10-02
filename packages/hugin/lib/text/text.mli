(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rich text.

    A text is a sequence of Unicode characters, each with a {e style}: bold or
    not, italic or not, a size factor, a baseline shift and an optional colour.
    {!Layout} sets a text in a list of faces at a size as the glyph runs
    ({!Hugin_font.Run}) that draw it, and measures them.

    Texts and layouts are immutable, every function is pure, and values may be
    shared between domains. *)

open Hugin_gg
open Hugin_font

(** {1:texts Texts} *)

type t
(** The type for texts. *)

val v : string -> t
(** [v s] is the characters of the UTF-8 string [s], neither bold nor italic,
    with size factor [1.], shift [0.] and no colour. Each maximal invalid
    subpart of [s], as {!String.get_utf_8_uchar} decodes it, becomes U+FFFD.
    Newlines end lines ({!Layout.section-lines}). *)

val concat : t list -> t
(** [concat ts] is the characters of [ts] in order, each with its style.
    [concat []] is [v ""]. *)

(** {1:styles Styles}

    A character of size factor [k] and shift [d] in a text set at size [s] is
    set at size [k *. s] with its baseline [d *. s] below the line's, above it
    if [d] is negative.

    Each combinator restyles every character of its argument and keeps what an
    inner combinator set: in [color c (concat [ a; color c' b ])] the characters
    of [b] keep [c']. *)

val bold : t -> t
(** [bold t] is [t] with every character bold. *)

val italic : t -> t
(** [italic t] is [t] with every character italic. *)

val scale : float -> t -> t
(** [scale s t] is [t] with every size factor and shift multiplied by [s].

    Raises [Invalid_argument] if [s] is not finite and positive. *)

val color : Color.t -> t -> t
(** [color c t] is [t] with the colour [c] on every character that has none. *)

val sup : t -> t
(** [sup t] is [t] as a superscript: every size factor [k] and shift [d] become
    [0.7 *. k] and [0.7 *. d -. 0.4]. *)

val sub : t -> t
(** [sub t] is [t] as a subscript: every size factor [k] and shift [d] become
    [0.7 *. k] and [0.7 *. d +. 0.2]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal t t'] is [true] iff [t] and [t'] have the same characters with the
    same styles, size factors and shifts compared by [Float.equal] and colours
    by {!Hugin_gg.Color.equal}. *)

val compare : t -> t -> int
(** [compare t t'] is a total order on texts compatible with {!equal}. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] formats [t] for debugging, in spans of one style. *)

(** {1:layout Layout} *)

module Layout : sig
  (** A layout is a text set in a list of faces at a size: lines of glyph runs
      placed about an origin, the point its alignment names. Lengths are in the
      unit of that size.

      {1:faces Faces}

      The first face [p] of the list is the {e plain} face. A bold character
      asks for weight [700] and an italic one for the slant [`Italic]; the
      others ask for the weight and slant of [p]. For each request the faces are
      ranked as CSS Fonts Level 4 ranks the faces of one family, the list being
      the family: by slant, then by weight, then by position.
      - Slant: the slant asked for, then for [`Italic] the [`Oblique] then the
        [`Normal] faces, for [`Oblique] the [`Italic] then the [`Normal] faces,
        and for [`Normal] the [`Oblique] then the [`Italic] faces.
      - Weight [w]: if [400 <= w <= 500], the weights from [w] to [500]
        increasing, then those below [w] decreasing, then those above [500]
        increasing; if [w < 400], the weights up to [w] decreasing, then the
        others increasing; if [w > 500], the weights from [w] up increasing,
        then the others decreasing.

      A character is drawn with the glyph of the best-ranked face whose
      character map has it ({!Hugin_font.Font.glyph} is not [0]). A
      character no face has is drawn with glyph [0] of the best-ranked face and
      listed by {!missing}. A default ignorable character (the Unicode 16.0
      [Default_Ignorable_Code_Point] property) is ignored: it is drawn with no
      glyph and the text is set as if it were absent.

      Faces are not synthesized and characters are not shaped: each is drawn
      with the one glyph its face maps it to, left to right in text order, a
      combining mark as a character of its own.

      {1:lines Lines}

      A newline ends a line: U+000A to U+000D, U+0085, U+2028 and U+2029, with
      U+000D U+000A counting as one. Newlines are drawn with no glyph, so a text
      with [n] newlines has [n + 1] lines, and the empty text one empty line.

      A {e break} is the point between a space (U+0020) and a following
      character that is neither a space nor a newline, unless only spaces
      precede that space on its line. The end points of a line are the breaks
      after its start, then the next newline or the end of the text. Each line
      starts where the previous one ended and ends at its first end point if it
      is wider than [width] there, otherwise at the last end point before the
      first at which it is wider than [width], or at its last end point if there
      is none, so a word wider than [width] overflows a line of its own. The
      spaces that end a line are drawn with no glyph and add nothing to its
      width.

      {1:placing Placing glyphs}

      A character of size factor [k] and shift [d] in a layout of size [size] is
      set at size [s = k *. size] with its baseline [e = d *. size] below the
      line's. A line's first glyph has its origin at the line's start, and each
      next one is advanced by the previous glyph's
      {!Hugin_font.Font.advance} times its size, plus, between glyphs of
      one face, size and shift, their {!Hugin_font.Font.kerning} times that
      size. A line's width is the advance of its glyphs. Consecutive glyphs of
      one face, size, shift and colour form one run, whose origin is its first
      glyph's.

      A glyph of face [f] reaches [Font.ascent f *. s -. e] above the line's
      baseline and [Font.descent f *. s +. e] below it. A line's {e ascent} and
      {e descent} are the greatest of these over its glyphs and of
      [Font.ascent p *. size] and [Font.descent p *. size]. Its {e cap height}
      is [Font.cap_height p *. s] for the greatest size [s] of its glyphs with
      [e = 0.], or [Font.cap_height p *. size] if there is none. Each next
      line's baseline lies below the previous one by the previous descent,
      [Font.line_gap p *. size] and its own ascent.

      {1:alignment Alignment}

      The horizontal alignment starts each line at [x = 0.] for [`Left], centres
      it on [x = 0.] for [`Center] and ends it at [x = 0.] for [`Right]. The
      vertical alignment puts at [y = 0.] the first line's top (its baseline
      less its ascent) for [`Top], the first line's cap height above its
      baseline for [`Cap], the last line's baseline for [`Baseline], the point
      midway between those of [`Cap] and [`Baseline] for [`Middle], and the last
      line's bottom (its baseline plus its descent) for [`Bottom]. *)

  type text := t

  type t
  (** The type for layouts. *)

  type halign = [ `Left | `Center | `Right ]
  (** The type for horizontal alignments ({!section-alignment}). *)

  type valign = [ `Top | `Cap | `Middle | `Baseline | `Bottom ]
  (** The type for vertical alignments ({!section-alignment}). *)

  val v :
    ?width:float ->
    ?halign:halign ->
    ?valign:valign ->
    fonts:Font.t list ->
    size:float ->
    text ->
    t
  (** [v ~width ~halign ~valign ~fonts ~size t] is [t] set in [fonts]
      ({!section-faces}) at size [size] ({!section-placing}), broken into lines
      at most [width] wide ({!section-lines}) and aligned by [halign] and
      [valign] ({!section-alignment}). [width] defaults to [infinity], [halign]
      to [`Left] and [valign] to [`Baseline].

      Raises [Invalid_argument] if [fonts] is empty, if [size] or the size of a
      character is not finite and positive, if [width] is negative or [nan], or
      if a coordinate of a glyph's origin, of {!box} or of {!ink} is not finite.
  *)

  val box : t -> Box2.t
  (** [box l] is the smallest box holding each line's box, which spans the
      line's width and its ascent above and descent below its baseline. An empty
      line's box has width [0.] at [x = 0.]. *)

  val ink : t -> Box2.t option
  (** [ink l] is the union of the {!Hugin_font.Run.bounds} of the runs of
      [l] at their origins, or [None] if no glyph has ink. *)

  val missing : t -> Uchar.t list
  (** [missing l] is the characters of [l]'s text that no face has, each once,
      in order of first occurrence ({!section-faces}). Newlines and default
      ignorable characters are never missing. *)

  val fold : ('a -> Color.t option -> P2.t -> Run.t -> 'a) -> 'a -> t -> 'a
  (** [fold f acc l] folds [f] over the runs of [l] in text order:
      [f acc c at r] for the run [r] with origin [at] and colour [c], [None] if
      its characters have none. Painting each run at its origin in [c], or in
      the caller's colour for [None], as [Hugin_vg.Picture.glyphs] does,
      draws [l]. *)

  val equal : t -> t -> bool
  (** [equal l l'] is [true] iff [l] and [l'] have equal boxes and {!fold} equal
      runs ({!Hugin_font.Run.equal}) at equal origins with equal colours,
      in the same order. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf l] formats [l] for debugging: its box and its runs. *)
end
