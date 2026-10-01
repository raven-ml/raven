(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Glyph runs.

    A run is text set in one font at one size: glyphs, each placed relative to
    the run's origin on its baseline, and the text they render. Text layout
    makes runs and pictures draw them. Every renderer puts each glyph where the
    run says, so raster, SVG and PDF output agree with the layout that measured
    them, and the text keeps PDF and SVG output searchable and selectable.

    Positions and sizes are in the units of the picture that draws the run, in
    the y-down plane of the
    {{!Hugin_next_gg.section-conventions}geometry conventions}. *)

open Hugin_next_gg

(** {1:types Types} *)

type t
(** The type for glyph runs. *)

(** {1:constructors Constructors} *)

val v :
  ?ys:float array ->
  ?clusters:int array ->
  font:Font.t ->
  size:float ->
  text:string ->
  glyphs:Font.glyph array ->
  xs:float array ->
  unit ->
  t
(** [v ~ys ~clusters ~font ~size ~text ~glyphs ~xs ()] is the run of the
    [glyphs] of [font] at em size [size] rendering [text], glyph [i] with its
    origin at [(xs.(i), ys.(i))], where:
    - [ys] defaults to zeros, every glyph on the baseline.
    - [clusters] maps glyphs to [text]: [clusters.(i)] is the byte index in
      [text] of the first Unicode character glyph [i] renders. Glyphs with equal
      indices form a cluster, which renders the bytes from its index to the next
      larger index or to the end of [text]. Clusters describe ligatures, one
      glyph for several characters, and decompositions, several glyphs for one
      character. Defaults to one glyph per character: [clusters.(i)] is the
      index of the [i]th Unicode character of [text].

    The arrays are copied.

    Raises [Invalid_argument] if:
    - [glyphs], [xs], [ys] or [clusters] differ in length;
    - a glyph is not in \[[0];[Font.glyph_count font - 1]\];
    - [size] is negative, or [size] or a position is not finite;
    - [text] is not valid UTF-8;
    - [glyphs] is empty and [text] is not;
    - [clusters] does not start at [0], decreases, or holds an index that does
      not start a Unicode character of [text] ([0] for an empty [text]);
    - [clusters] is absent and [text] does not have one Unicode character per
      glyph. *)

(** {1:accessors Accessors} *)

val font : t -> Font.t
(** [font r] is the font of [r]. *)

val size : t -> float
(** [size r] is the em size of [r]. *)

val text : t -> string
(** [text r] is the text [r] renders. *)

val length : t -> int
(** [length r] is the number of glyphs of [r]. *)

(** {2:glyphs Glyphs}

    Functions that take a glyph index raise [Invalid_argument] if it is not in
    \[[0];[length r - 1]\]. *)

val glyph : t -> int -> Font.glyph
(** [glyph r i] is the glyph at index [i] of [r]. *)

val x : t -> int -> float
(** [x r i] is the x coordinate of the origin of glyph [i] of [r]. *)

val y : t -> int -> float
(** [y r i] is the y coordinate of the origin of glyph [i] of [r]. *)

val cluster : t -> int -> int
(** [cluster r i] is the byte index in [text r] of the first Unicode character
    that glyph [i] of [r] renders. *)

(** {1:bounds Bounds} *)

val bounds : t -> Box2.t option
(** [bounds r] is the smallest box containing the ink of the glyphs of [r], each
    outline scaled by [size r] and moved to its origin, or [None] if no glyph
    has ink.

    Raises [Invalid_argument] if a corner of that box is not finite, which takes
    a size or a position near [max_float]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal r r'] is [true] iff [r] and [r'] have {!Font.equal} fonts, and equal
    sizes, texts, glyphs, positions and clusters. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf r] formats [r] for debugging. *)
