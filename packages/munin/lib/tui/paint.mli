(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The styles of the screens and how they draw text into the grid.

   Two weights rank what a screen shows: normal for what the user reads, dim for
   what frames it. Bold marks a state word, a run's name and the number to read.
   Colour repeats a glyph or a word and never carries a fact alone, so
   [NO_COLOR], which keeps attributes and drops colours, keeps every fact. Every
   colour is an entry of the terminal's palette, except the cursor row's tint,
   used at 256 colours and more; below, the cursor row is reverse video. *)

type depth = Mono | Ansi16 | Ansi256 | Truecolor

val depth : unit -> depth
(** [depth ()] is the colour depth the environment gave at start ([NO_COLOR],
    [COLORTERM], [TERM]), or the one {!set_depth} set last. *)

val set_depth : depth -> unit
(** [set_depth d] makes [d] the colour depth, for the tests. *)

val tinted : unit -> bool
(** [tinted ()] is [true] iff the cursor row is a tint, at 256 colours or
    more. *)

val plain : Ansi.Style.t
val dim : Ansi.Style.t
val bold : Ansi.Style.t
val reverse : Ansi.Style.t
val ok : Ansi.Style.t
val bad : Ansi.Style.t
val warn : Ansi.Style.t
val accent : Ansi.Style.t
val alert : Ansi.Style.t

val mark_color : char -> Ansi.Color.t
(** [mark_color c] is the colour of the marked run lettered [c]. *)

val mark : char -> Ansi.Style.t
(** [mark c] is the bold style of the letter [c]. *)

val with_ : Ansi.Style.t -> Ansi.Style.t -> Ansi.Style.t
(** [with_ s o] is [s] with the properties [o] sets. *)

val selected : light:bool -> Ansi.Style.t
(** [selected ~light] is the cursor row's style. *)

val state : string -> string * Ansi.Style.t
(** [state w] is the glyph and style of the state word [w]. *)

val word : string -> (Ansi.Style.t * string) list
(** [word w] is the state word [w] after its glyph, in its style. *)

(* Drawing *)

type spans = (Ansi.Style.t * string) list
(** Text in parts, each in its style. *)

val put :
  ?style:Ansi.Style.t ->
  Matrix_grid.t ->
  x:int ->
  y:int ->
  w:int ->
  string ->
  int
(** [put g ~x ~y ~w s] draws [s] cut to [w] cells, nothing outside the grid, and
    is the cells drawn. *)

val text :
  ?style:Ansi.Style.t ->
  Matrix_grid.t ->
  x:int ->
  y:int ->
  w:int ->
  string ->
  unit
(** [text] is {!put} without its result. *)

val spans :
  ?base:Ansi.Style.t ->
  Matrix_grid.t ->
  x:int ->
  y:int ->
  w:int ->
  spans ->
  unit
(** [spans g ~x ~y ~w l] draws [l] from [x], cut at [x + w], each part's style
    over [base]. *)

val width : spans -> int

val right :
  ?base:Ansi.Style.t ->
  Matrix_grid.t ->
  x:int ->
  y:int ->
  w:int ->
  spans ->
  unit
(** [right g ~x ~y ~w l] draws [l] ending at column [x + w]. *)

val fill : Matrix_grid.t -> x:int -> y:int -> w:int -> Ansi.Style.t -> unit

val sep : Ansi.Style.t * string
(** [sep] is the dim [ · ] between parts. *)

val join : spans list -> spans
(** [join parts] is the parts that are not empty, between {!sep}s. *)

val fit : ?drop:int list -> w:int -> spans -> spans list -> spans
(** [fit ~w head parts] is [head] and as many of [parts] as fit in [w] cells,
    joined: parts leave whole, those at the indices [drop] first, in order, then
    from the end. *)

val hints :
  Matrix_grid.t -> x:int -> y:int -> w:int -> (string * string) list -> unit
(** [hints g ~x ~y ~w keys] draws the key line: each key and its action, keys
    plain and actions dim, those that do not fit left out whole from the end,
    and [? keys] last. *)

val key : string -> string * string
(** [key k] is the metric key [k]'s prefix, up to and with its last [/], and the
    rest. *)

val key_spans : ?style:Ansi.Style.t -> string -> spans
(** [key_spans k] is [k] with its prefix dim. *)

val path : int -> string -> string
(** [path w p] is the path [p] in at most [w] cells, its end kept. *)

val centre : Matrix_grid.t -> y:int -> w:int -> spans -> unit
(** [centre g ~y ~w l] draws [l] centred in the first [w] columns. *)
