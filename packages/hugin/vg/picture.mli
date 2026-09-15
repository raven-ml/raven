(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pictures.

    A picture is an immutable description of a drawing. Make leaves with
    {!fill}, {!stroke}, {!text} and {!image}, combine them with {!group} and
    place them with {!transform}, {!clip} and {!stamp}. Renderers fold over the
    result, so every output format sees the same drawing.

    Drawing order is the order of a {!group}: later pictures are composited over
    earlier ones with source-over. Coordinates are in a y-down plane whose unit
    the renderer decides, a pixel or a point. *)

(** {1:types Types} *)

type rule = [ `Nonzero | `Evenodd ]
(** The type for fill rules: whether a point is inside a path when its winding
    number is non-zero or when it is odd. *)

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
      (** The type for pictures. The cases are exposed so that renderers can
          match on them; each is built and validated by the function of the same
          name below. *)

(** {1:leaves Leaves} *)

val empty : t
(** [empty] draws nothing. *)

val fill : ?rule:rule -> Color.t -> Path.t -> t
(** [fill ~rule c p] fills the inside of [p] with [c] under [rule], which
    defaults to [`Nonzero]. Open subpaths are closed for filling. It is {!empty}
    if [p] is. *)

val stroke : Stroke.t -> Color.t -> Path.t -> t
(** [stroke s c p] draws the pen of style [s] along [p] in color [c]. Open
    subpaths get caps, closed ones a join at their start. It is {!empty} if [p]
    is. *)

val text : Font.t -> size:float -> Color.t -> x:float -> y:float -> string -> t
(** [text f ~size c ~x ~y s] draws [s] in font [f] at em height [size] in color
    [c], with the origin of its first glyph at [(x, y)] on the baseline. To
    align, offset [(x, y)] by the box of {!Font.bounds}, or {!transform} the
    picture by its own {!bounds}. It is {!empty} if [s] is. *)

val image : x:float -> y:float -> w:float -> h:float -> Nx.uint8_t -> t
(** [image ~x ~y ~w ~h img] draws [img] scaled to fill the rectangle with corner
    [(x, y)], width [w] and height [h], its first row at the top. [img] has
    shape [[|rows; cols|]] for grey or [[|rows; cols; c|]] with [c] channels:
    [1] grey, [3] RGB or [4] RGBA with straight alpha. Renderers keep the pixels
    crisp; they do not interpolate.

    Raises [Invalid_argument] on any other shape. *)

(** {1:composing Composing} *)

val group : t list -> t
(** [group ps] draws the pictures of [ps] in order, each over the previous ones.
*)

val clip : Path.t -> t -> t
(** [clip p pic] draws only the part of [pic] inside [p] under the nonzero rule.
    Clips nest by intersection. *)

val transform : Affine.t -> t -> t
(** [transform m pic] draws [pic] with its coordinates mapped through [m].
    Stroke widths, dashes and text sizes scale with [m]. *)

val stamp : t -> float array -> float array -> t
(** [stamp pic xs ys] draws [pic] translated to each point [(xs.(i), ys.(i))],
    in order. It means the same as a {!group} of translated copies, but
    renderers may draw [pic] once and reuse it, so it is the way to draw the
    same marker at many points. Points with a non-finite coordinate are skipped.

    Raises [Invalid_argument] if [xs] and [ys] differ in length. *)

(** {1:bounds Bounds} *)

val bounds : t -> Box.t option
(** [bounds pic] is the box enclosing what [pic] draws, or [None] if it draws
    nothing. The box is tight for fills, text and images. A stroke is enclosed
    by its pen width, enlarged by the miter limit when its joins are mitered, so
    it may exceed the ink. Clips do not shrink the box. *)

(** {1:formatting Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp fmt pic] formats [pic] for tests and debugging as nested s-expressions,
    one per case, with paths as SVG path data, colors as [rgb] or [rgba] and
    images by their shape. The output is not stable across releases. *)
