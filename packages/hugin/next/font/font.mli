(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** OpenType fonts.

    A font is one face of an OpenType file with TrueType outlines, decoded once:
    its character map, glyph advances, pair kerning, vertical metrics, glyph
    outlines and their ink boxes. Two faces of {{:https://rsms.me/inter/}Inter}
    are bundled, {!regular} and {!bold}, so output is the same on every machine
    without system fonts; other faces load from a file's bytes with {!of_string}
    or {!of_file}.

    Fonts are immutable values. Every function of this module is pure, none
    fails on account of a font's data once {!of_string} returned it, and fonts
    may be shared between domains.

    {1:units Units}

    Metrics and outlines are in {e em}, the font's design size: text set at size
    [s] scales them by [s]. Vertical metrics are distances from the baseline,
    positive away from it. Outlines and boxes are in the y-down plane of the
    {{!Hugin_next_gg.section-conventions}geometry conventions} with the glyph's
    origin at [(0, 0)] on the baseline, so ink above the baseline has negative
    y. *)

open Hugin_next_gg

(** {1:types Types} *)

type t
(** The type for fonts. *)

type glyph = int
(** The type for glyph ids. They index the font's glyph table, from [0] to
    [glyph_count f - 1]. Glyph [0] is [.notdef], which fonts draw for characters
    they lack, often as a box. *)

type slant = [ `Normal | `Italic | `Oblique ]
(** The type for slants: upright, italic, or upright letterforms slanted. *)

(** {1:bundled Bundled faces}

    Inter is distributed under the SIL Open Font License 1.1. The bundled faces
    are subsets covering Latin and Greek letters, digits, punctuation, arrows
    and common mathematical symbols, without Cyrillic or CJK. Their digits are
    tabular: the ten share one advance, so numbers set in a column align. *)

val regular : t
(** [regular] is Inter Regular, weight 400. *)

val bold : t
(** [bold] is Inter Bold, weight 700. *)

(** {1:loading Loading} *)

(** The type for loading errors. Each message says what is at fault and where,
    for users. *)
type error =
  | Io of string  (** The file could not be read: the system's message. *)
  | Malformed of string  (** The bytes are not a valid font. *)
  | Unsupported of string
      (** The font is valid but uses a feature this decoder does not handle. *)

val of_string : string -> (t, error) result
(** [of_string s] is [Ok f] if [s] is an OpenType font file with TrueType
    ([glyf]) outlines, and [Error e] otherwise. Decoding reads the tables
    [head], [hhea], [maxp], [hmtx], [cmap], [loca] and [glyf], and [OS/2],
    [name], [post], [GPOS] and [kern] when present, and checks all it reads,
    every glyph outline included, so that later queries cannot fail.

    The error is [Unsupported _] for a valid font that uses what this decoder
    does not handle, such as CFF or CFF2 outlines, a font collection, or a
    character map without a Unicode subtable (platform 0, or platform 3 with
    encoding 1 or 10) of format 4 or 12, and [Malformed _] for anything else,
    which includes a composite glyph whose components, expanded down to simple
    glyphs, hold more than 65535 points or more than 65535 components. A
    variable font decodes as its default instance. *)

val of_file : string -> (t, error) result
(** [of_file path] is [of_string s] for the contents [s] of the file [path], or
    [Error (Io msg)] if the file cannot be read. *)

val pp_error : Format.formatter -> error -> unit
(** [pp_error ppf e] formats [e] for users. *)

(** {1:identity Identity}

    A font is identified by its {!bytes}: renderers embed a font once however
    many runs use it, and pictures compare fonts by {!equal}. *)

val bytes : t -> string
(** [bytes f] is an OpenType file holding exactly the face of [f]. For a font
    from {!of_string} it is the string decoded. PDF and SVG output embed it. *)

val equal : t -> t -> bool
(** [equal f f'] is [String.equal (bytes f) (bytes f')]. *)

val compare : t -> t -> int
(** [compare f f'] is a total order on fonts compatible with {!equal}. *)

val family : t -> string
(** [family f] is the typographic family name of [f] (name ID 16) if it has one
    and its family name (name ID 1) otherwise, or [""] if [f] names no family. A
    name is read from the US English Windows Unicode record (platform 3,
    encoding 1 or 10, language 0x409) if there is one, else from the first
    Windows Unicode record in table order, else from the first Macintosh Roman
    one (platform 1, encoding 0). *)

val postscript_name : t -> string
(** [postscript_name f] is the PostScript name of [f] (name ID 6), the name PDF
    output gives the font, read as {!family} reads names, or [""] if [f] has
    none. *)

val weight : t -> int
(** [weight f] is the weight class of [f] from its [OS/2] table, clamped to
    \[[1];[1000]\]: [400] is regular and [700] bold. It is [400] without an
    [OS/2] table. *)

val slant : t -> slant
(** [slant f] is [`Italic] if the [OS/2] table of [f] marks it italic, or its
    [head] table does when it has no [OS/2] table; [`Oblique] if [OS/2] marks it
    oblique; and [`Normal] otherwise. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf f] formats the family, weight and slant of [f] for debugging, the
    family as {!Run.pp} formats text. *)

(** {1:subsetting Subsetting} *)

val subset : t -> glyph list -> string
(** [subset f gs] is an OpenType file holding the face of [f] cut down to the
    glyphs it keeps: glyph [0], the glyphs [gs], and the components of every
    composite glyph it keeps. PDF output embeds it in place of {!bytes}, so that
    a document carries only the glyphs it shows.

    Glyphs keep their ids, so a run of glyphs of [gs] draws the same from the
    subset as from [f]. Decoded by {!of_string}, the subset is a font [f'] such
    that:
    - [glyph_count f'] is one more than the largest glyph it keeps.
    - [outline f' g] is [outline f g] and [advance f' g] is [advance f g] for a
      glyph [g] it keeps. Any other glyph has an empty outline and a zero
      advance.
    - [glyph f' u] is [glyph f u] if that glyph is kept, and [0] otherwise.
    - Its {!family}, {!postscript_name}, {!weight}, {!slant}, {!ascent},
      {!descent}, {!line_gap}, {!italic_angle} and {!bounds} are those of [f],
      and so are its {!cap_height} and {!x_height} where the [OS/2] table of [f]
      records them. Face-wide maxima of the [head], [hhea] and [maxp] tables are
      those of [f], which bound the glyphs it keeps.
    - It has no kerning: runs place its glyphs, so it holds no [GPOS], [kern] or
      other layout table.

    The subset holds the tables [head], [hhea], [maxp], [hmtx], [cmap], [loca],
    [glyf] and [post], without glyph names, and the [OS/2] and [name] tables of
    [f] and its hinting tables [cvt ], [fpgm], [prep] and [gasp] when it has
    them. It depends only on [f] and the set of glyphs [gs]: their order and
    repetitions do not change a byte.

    Raises [Invalid_argument] if a glyph of [gs] is not in
    \[[0];[glyph_count f - 1]\]. *)

(** {1:metrics Metrics} *)

val ascent : t -> float
(** [ascent f] is the distance above the baseline that lines of [f] reserve: the
    typographic ascender of the [OS/2] table if [OS/2] sets [USE_TYPO_METRICS],
    and the ascender of the [hhea] table otherwise. {!descent} and {!line_gap}
    read the same table. *)

val descent : t -> float
(** [descent f] is the distance below the baseline that lines of [f] reserve,
    from the table {!ascent} reads. *)

val line_gap : t -> float
(** [line_gap f] is the extra space [f] puts between the descent of a line and
    the ascent of the next, from the table {!ascent} reads. *)

val cap_height : t -> float
(** [cap_height f] is the height of flat capital letters above the baseline: the
    [OS/2] value if [f] records a positive one, else the top of the ink of [f]'s
    glyph for [H] if it has ink, else {!ascent}, as CSS assumes. *)

val x_height : t -> float
(** [x_height f] is the height of flat lowercase letters above the baseline: the
    [OS/2] value if [f] records a positive one, else the top of the ink of [f]'s
    glyph for [x] if it has ink, else [0.5], as CSS assumes. *)

val italic_angle : t -> float
(** [italic_angle f] is the angle of [f]'s upright strokes from the vertical, in
    radians, negative when they lean right (the sign of OpenType and PDF,
    opposite to the plane's convention for angles), from the [post] table. It is
    [0.] for an upright face or without a [post] table. *)

val bounds : t -> Box2.t
(** [bounds f] is the box enclosing every glyph of [f] as recorded in its [head]
    table. *)

(** {1:glyphs Glyphs}

    Functions that take a glyph raise [Invalid_argument] if it is not in
    \[[0];[glyph_count f - 1]\]. *)

val glyph_count : t -> int
(** [glyph_count f] is the number of glyphs of [f], at least [1]. *)

val glyph : t -> Uchar.t -> glyph
(** [glyph f u] is the glyph [f]'s character map gives [u], and [0] if it gives
    none or gives a glyph [f] does not have. *)

val advance : t -> glyph -> float
(** [advance f g] is the horizontal distance the pen moves after drawing [g]
    ([hmtx]). *)

val kerning : t -> glyph -> glyph -> float
(** [kerning f g g'] is the change to [advance f g] when [g'] follows [g],
    usually negative. It is the sum, over the set of lookups that any [kern]
    feature of the [GPOS] table refers to, each counted once whatever the number
    of features or scripts referencing it, of the x advance of the first pair
    adjustment of that lookup covering the pair. Other adjustments are ignored.
    A font whose [GPOS] table has no [kern] feature, or that has no [GPOS]
    table, uses its [kern] table instead: the sum of the values that its
    horizontal format 0 subtables give the pair, minimum and cross-stream
    subtables excluded. A font with neither has no kerning. *)

val outline : t -> glyph -> Path.t
(** [outline f g] is the outline of [g], unhinted, to be filled under the
    nonzero rule. Its quadratic curves are exact, composite glyphs are the union
    of their transformed components, and a glyph without ink, such as a space,
    is {!Path.empty}. Its ink box is {!ink}. *)

val ink : t -> glyph -> Box2.t option
(** [ink f g] is [Path.bounds (outline f g)], the smallest box containing the
    ink of [g], or [None] if [g] has no ink. It is recorded when [f] is decoded,
    so [ink] builds no outline. *)
