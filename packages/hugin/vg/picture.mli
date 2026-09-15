(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pictures.

    A picture is an immutable description of a drawing. Renderers fold over it,
    so every output format sees the same drawing. *)

type rule = [ `Nonzero | `Evenodd ]
(** The type for fill rules. *)

type t = private
  | Empty
  | Fill of { rule : rule; color : Color.t; path : Path.t }
  | Stroke of { stroke : Stroke.t; color : Color.t; path : Path.t }
  | Text of {
      font : Font.t;
      size : float;
      color : Color.t;
      x : float;
      y : float;
      text : string;
    }
  | Image of { x : float; y : float; w : float; h : float; data : Nx.uint8_t }
  | Group of t list
  | Clip of { path : Path.t; picture : t }
  | Transform of { m : Affine.t; picture : t }
  | Stamp of { picture : t; xs : float array; ys : float array }
      (** The type for pictures. Later elements of a {!Group} are drawn over
          earlier ones and colors composite with source-over. The representation
          is visible for renderers; build pictures with the functions below. *)

val empty : t

val fill : ?rule:rule -> Color.t -> Path.t -> t
(** [fill ~rule c p] fills [p] with [c]. Open subpaths are closed for filling.
    [rule] defaults to [`Nonzero]. *)

val stroke : Stroke.t -> Color.t -> Path.t -> t
(** [stroke s c p] strokes [p] with style [s] and color [c]. *)

val text : Font.t -> size:float -> Color.t -> x:float -> y:float -> string -> t
(** [text f ~size c ~x ~y s] draws [s] in [f] at em height [size], with the
    origin of its first glyph at [(x, y)] on the baseline. Alignment is the
    caller's arithmetic from {!Font.advance} and {!Font.bounds}. *)

val image : x:float -> y:float -> w:float -> h:float -> Nx.uint8_t -> t
(** [image ~x ~y ~w ~h img] draws [img], of shape [[|rows; cols; c|]] with [c]
    in [1], [3] or [4] channels, or [[|rows; cols|]], scaled to fill the
    rectangle with corner [(x, y)].

    Raises [Invalid_argument] on any other shape. *)

val group : t list -> t
(** [group ps] draws the pictures of [ps] in order. *)

val clip : Path.t -> t -> t
(** [clip p pic] restricts [pic] to the inside of [p] under the nonzero rule. *)

val transform : Affine.t -> t -> t
(** [transform m pic] draws [pic] with its coordinates mapped through [m]. *)

val stamp : t -> float array -> float array -> t
(** [stamp pic xs ys] draws [pic] translated to each [(xs.(i), ys.(i))].
    Equivalent to a group of translated copies; renderers may draw [pic] once
    and reuse it.

    Raises [Invalid_argument] if the arrays differ in length. *)

(** {1:bounds Bounds} *)

val bounds : t -> Box.t option
(** [bounds pic] is the box enclosing [pic], or [None] if it draws nothing. The
    box is tight for fills, text and images; a stroke is enclosed by its pen
    width, and by its miter limit at miter joins. *)
