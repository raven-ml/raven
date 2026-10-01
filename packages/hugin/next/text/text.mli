(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rich text.

    A text is a sequence of Unicode characters, each with a {e style}: whether
    it is bold, whether it is italic, a size factor, a baseline shift, and a
    colour or none. Labels, titles and legend entries are texts. {!v} makes a
    plain text from a string, {!concat} joins texts, and the
    {{!section-styles}style combinators} restyle every character of a text, so
    that [Text.(concat [ v "step size η"; sub (v "0") ])] is an axis title with
    a subscript.

    {!Layout} sets a text in a list of faces at a size: it picks a face for each
    character, breaks lines, and places the glyph runs ({!Hugin_next_font.Run})
    that pictures draw, measuring them without drawing.

    Texts and layouts are immutable values, every function of this module is
    pure, and values may be shared between domains. *)

open Hugin_next_gg
open Hugin_next_font

(** {1:texts Texts} *)

type t
(** The type for texts. *)

val v : string -> t
(** [v s] is the characters of the UTF-8 string [s], neither bold nor italic,
    with size factor [1.], no shift and no colour. Each maximal invalid subpart
    of [s], as {!String.get_utf_8_uchar} decodes it, becomes U+FFFD REPLACEMENT
    CHARACTER. Newlines in [s] end lines ({!Layout.section-lines}). *)

val concat : t list -> t
(** [concat ts] is the characters of the texts of [ts] in order, each with its
    style. [concat []] is [v ""]. *)

(** {1:styles Styles}

    A character's size factor [k] and shift [d] are relative to the size [s] its
    text is set at ({!Layout.v}): the character is set at size [k *. s] with its
    baseline [d *. s] below the line's, above it when [d] is negative. {!v}
    gives [k = 1.] and [d = 0.].

    Each combinator below restyles every character of its argument, so
    combinators nest: [bold (concat [ a; italic b ])] makes [a] bold and [b]
    bold and italic, and in [color c (concat [ a; color c' b ])] the characters
    of [b] keep [c']. *)

val bold : t -> t
(** [bold t] is [t] with every character bold. *)

val italic : t -> t
(** [italic t] is [t] with every character italic. *)

val scale : float -> t -> t
(** [scale s t] is [t] at [s] times its size: every character's size factor and
    shift are multiplied by [s].

    Raises [Invalid_argument] if [s] is not finite and positive. *)

val color : Color.t -> t -> t
(** [color c t] is [t] with the colour [c] on every character that has none. *)

val sup : t -> t
(** [sup t] is [t] as a superscript: at [0.7] times its size, with its baseline
    raised by [0.4] times the size of the text around it. Every character's size
    factor [k] and shift [d] become [0.7 *. k] and [0.7 *. d -. 0.4]. *)

val sub : t -> t
(** [sub t] is [t] as a subscript: at [0.7] times its size, with its baseline
    lowered by [0.2] times the size of the text around it. Every character's
    size factor [k] and shift [d] become [0.7 *. k] and [0.7 *. d +. 0.2]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal t t'] is [true] iff [t] and [t'] have the same characters with the
    same styles, size factors and shifts compared by [Float.equal] and colours
    by {!Hugin_next_gg.Color.equal}. So [concat [ v "a"; v "b" ]] equals
    [v "ab"], and [bold (bold t)] equals [bold t]. *)

val compare : t -> t -> int
(** [compare t t'] is a total order on texts compatible with {!equal}. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] formats [t] for debugging: its characters in spans of one style.
*)

(** {1:layout Layout} *)

module Layout : sig
  (** A layout is a text set in a list of faces at a size: lines of glyph runs
      placed about an origin, which is the anchor its alignment names. {!box}
      and {!ink} measure it without drawing, and {!fold} gives its runs to
      whatever draws them. Renderers put each glyph where its run says, so what
      a layout measures is what is drawn.

      Lengths are in the unit of the size given to {!v}, points in Hugin.

      {1:faces Faces}

      The first face of the list a text is set in is its {e plain} face [p]:
      plain characters are set in it. A bold character asks for weight [700] and
      any other for the weight of [p] ({!Hugin_next_font.Font.weight}); an
      italic character asks for the slant [`Italic] and any other for the slant
      of [p] ({!Hugin_next_font.Font.slant}). For each such request the faces
      are ranked as CSS Fonts Level 4 ranks the faces of one family, the whole
      list being one family: by slant, then by weight, then by position in the
      list.
      - By slant: the faces of the slant asked for, then, for [`Italic], the
        [`Oblique] faces, then the [`Normal] ones; for [`Oblique], the [`Italic]
        faces, then the [`Normal] ones; for [`Normal], the [`Oblique] faces,
        then the [`Italic] ones.
      - By weight, for a weight [w] asked for: if [400 <= w <= 500], the weights
        from [w] to [500] increasing, then those below [w] decreasing, then
        those above [500] increasing; if [w < 400], the weights up to [w]
        decreasing, then the others increasing; if [w > 500], the weights from
        [w] up increasing, then the others decreasing.

      A character is drawn with the glyph that the best-ranked face whose
      character map has it ({!Hugin_next_font.Font.glyph} is not [0]) gives it,
      so the faces after the first are fallbacks for the characters it lacks. A
      character that no face has is drawn with glyph [0], [.notdef], of the
      best-ranked face and listed by {!missing}.

      A default ignorable character (the [Default_Ignorable_Code_Point] property
      of Unicode 16.0, such as U+00AD SOFT HYPHEN, U+200D ZERO WIDTH JOINER or a
      variation selector) is drawn with no glyph, whether or not a face maps it,
      and is otherwise ignored: a text is set as the same text without its
      default ignorable characters.

      No face is synthesized: a bold character is set in the best-ranked face
      that has it, bold or not. Each character is drawn with the one glyph its
      face's character map gives it, without ligatures, contextual forms or
      complex-script shaping, and a combining mark is set as a character of its
      own, neither composed with nor positioned on its base. Glyphs are placed
      left to right in the order of their characters, with no bidirectional
      reordering, so right-to-left text such as Hebrew is drawn in reverse
      reading order.

      {1:lines Lines}

      A newline ends a line: U+000A to U+000D, U+0085, U+2028 and U+2029, with
      U+000D followed by U+000A counting as one. A newline is drawn with no
      glyph, so a text with [n] newlines has [n + 1] lines before breaking, and
      the empty text has one empty line.

      A {e break} is the point between a space (U+0020) and a following
      character that is neither a space nor a newline, unless only spaces
      precede that space on its line; no other character offers one. The end
      points of a line are the breaks after its start and before the next
      newline, and that newline or the end of the text. Lines break greedily:
      each line starts where the previous one ended and ends at its first end
      point if it is wider than [width] there, and otherwise at the last end
      point before the first one at which it is wider than [width], or at its
      last end point if there is no such one. So a word wider than [width]
      overflows a line of its own.

      A line's width is the distance its glyphs advance ({!section-placing}).
      The spaces that end a line hang: they are drawn with no glyph and add
      nothing to its width.

      {1:placing Placing glyphs}

      A character of size factor [k] and shift [d]
      ({!Hugin_next_text.Text.section-styles}) in a text set at size [size] is
      set at size [s = k *. size], with its baseline [d *. size] below the
      line's. A line's first glyph has its origin at the line's start, and each
      other glyph's origin follows the previous one's by the previous glyph's
      advance ({!Hugin_next_font.Font.advance}) times that glyph's size, plus,
      when both glyphs are from one face with one size and shift, their kerning
      ({!Hugin_next_font.Font.kerning}) times that size. Consecutive glyphs of a
      line from one face with one size, shift and colour form one glyph run,
      whose origin is its first glyph's and whose text is the characters its
      glyphs draw.

      A glyph of face [f] set at size [s] with its baseline [e] below the line's
      reaches [Font.ascent f *. s -. e] above the line's baseline and
      [Font.descent f *. s +. e] below it. For the plain face [p]:
      - A line's {e ascent} is the greatest of these heights of its glyphs and
        [Font.ascent p *. size].
      - Its {e descent} is the greatest of these depths of its glyphs and
        [Font.descent p *. size]. A superscript or subscript can thus make its
        line taller.
      - Its {e cap height} is [Font.cap_height p *. s] for the greatest size [s]
        of its glyphs set on its baseline ([e = 0.]), or
        [Font.cap_height p *. size] if it has none, so that lines of one size
        anchor alike whichever faces draw them.
      - The baseline of each line after the first lies below the previous one by
        the previous line's descent, plus [Font.line_gap p *. size], plus its
        own ascent.

      {1:alignment Alignment}

      The alignment places the lines about the layout's origin, so drawing a
      layout at a point puts that point where the alignment says, and rotating
      it about that point rotates its runs about its origin.
      - The horizontal alignment places each line: [`Left] starts it at
        [x = 0.], [`Center] centres it on [x = 0.], and [`Right] ends it at
        [x = 0.].
      - The vertical alignment puts at [y = 0.]: for [`Top], the top of the
        first line, its ascent above its baseline; for [`Cap], the top of the
        capitals of the first line, its cap height above its baseline; for
        [`Baseline], the baseline of the last line; for [`Middle], the point
        midway between the points of [`Cap] and [`Baseline], the middle of the
        capitals and figures of a one-line text; and for [`Bottom], the bottom
        of the last line, its descent below its baseline. *)

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
  (** [v ~width ~halign ~valign ~fonts ~size t] is [t] set in the faces [fonts]
      ({!section-faces}) with plain characters at em size [size], in lines at
      most [width] wide where they can break ({!section-lines}), aligned by
      [halign] and [valign] ({!section-alignment}), where:
      - [width] defaults to [infinity], so that only newlines end lines;
      - [halign] defaults to [`Left];
      - [valign] defaults to [`Baseline], so that the origin of a one-line text
        is the start of its baseline, and the lines of a longer one rise from
        it.

      Raises [Invalid_argument] if [fonts] is empty, if [size] or the size
      [k *. size] of a character ({!section-placing}) is not finite and
      positive, if [width] is negative or [nan], or if a coordinate of a glyph's
      origin, of {!box} or of {!ink} would not be finite, which takes sizes near
      [max_float]. *)

  val box : t -> Box2.t
  (** [box l] is the smallest box holding the box of every line of [l], which
      spans the line's width horizontally and its ascent above and descent below
      its baseline vertically. An empty line's box has width [0.] at [x = 0.].
      The box of [l] rotated by [a] about the origin lies within
      [Box2.transform (Affine.rotate a) (box l)]. *)

  val ink : t -> Box2.t option
  (** [ink l] is the smallest box containing the ink of the glyphs of [l]: the
      union of the {!Hugin_next_font.Run.bounds} of its runs, each moved to its
      origin. It is [None] if no glyph of [l] has ink. *)

  val missing : t -> Uchar.t list
  (** [missing l] is the characters of the text of [l] that no face of its list
      has, each once, in the order they first occur. [l] draws them with
      [.notdef] glyphs ({!section-faces}). Newlines and default ignorable
      characters are never missing. *)

  val fold : ('a -> Color.t option -> P2.t -> Run.t -> 'a) -> 'a -> t -> 'a
  (** [fold f acc l] folds [f] over the glyph runs of [l] in the order of their
      characters in the text, calling [f acc c at r] for the run [r] with its
      origin at [at] and [c] the colour of its characters, [None] if they have
      none. Painting each run's glyphs at [at] in [c], or the caller's colour
      for [None], as [Hugin_next_vg.Picture.glyphs c at r] does, draws [l]. *)

  val equal : t -> t -> bool
  (** [equal l l'] is [true] iff [l] and [l'] have equal boxes and {!fold} the
      same runs ({!Hugin_next_font.Run.equal}) at equal origins with equal
      colours in the same order. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf l] formats [l] for debugging: its box, and its runs with their
      origins and colours. *)
end
