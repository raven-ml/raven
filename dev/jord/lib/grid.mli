(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Grids: the cells of a raster, placed in a coordinate reference system.

    A {e grid} is [rows × cols] cells and an affine map from pixel points to a
    {!Crs.t}'s plane. An array {e on} a grid has the grid's rows and columns as
    its first two axes, as an image's [[h; w; c]].

    A {e pixel point} is [(row, col)], in the array's axis order. Cell [(i, j)]
    is the square of pixel points \[i, i + 1) × \[j, j + 1): its corner [(i, j)]
    is pixel point [(i, j)] and its centre is [(i + 0.5, j + 0.5)]. This is
    GeoTIFF's PixelIsArea convention, whose raster space has "(0,0) denoting the
    upper-left corner of the image" (OGC 19-008r4, Annex B.2.2), with its (I, J)
    axes swapped into array order. The pixel point [(r, c)] lies at

    {[
     origin + r · row_step + c · col_step
    ]}

    in the system's plane, each term a pair (x, y). A north-up grid of
    resolution [d] has [row_step = (0, -d)] and [col_step = (d, 0)]; any other
    invertible map, rotated or sheared as GeoTIFF's ModelTransformation allows,
    is a grid too. *)

type t = private {
  crs : Crs.t;
  rows : int;
  cols : int;
  origin : float * float;  (** Where pixel point [(0, 0)] lies. *)
  row_step : float * float;  (** From one row to the next. *)
  col_step : float * float;  (** From one column to the next. *)
}
(** The type for grids. *)

val v :
  Crs.t ->
  rows:int ->
  cols:int ->
  origin:float * float ->
  row_step:float * float ->
  col_step:float * float ->
  t
(** [v crs ~rows ~cols ~origin ~row_step ~col_step] is the grid of those fields.

    Raises [Invalid_argument] if [rows] or [cols] is negative, a number is not
    finite, or the steps' determinant is 0. *)

val world : t -> 'd Nx.float64_t -> 'd Nx.float64_t
(** [world g p] is the pixel points [p], [[...; 2]], as points of [g.crs],
    computed where [p] lies.

    Raises [Invalid_argument] if [p]'s last axis is not 2. *)

val pixel : t -> 'd Nx.float64_t -> 'd Nx.float64_t
(** [pixel g x] is the points [x] of [g.crs], [[...; 2]], as pixel points, by
    the inverse of {!world}'s map. A point lies in cell [(⌊row⌋, ⌊col⌋)]. For a
    grid whose steps are axis-aligned, each coordinate is one subtraction and
    one division, rounded once each.

    Raises [Invalid_argument] if [x]'s last axis is not 2. *)

val centres : at:'d Nx.Placement.t -> t -> 'd Nx.float64_t
(** [centres ~at g] is each cell's centre in [g.crs], [[rows; cols; 2]], made at
    [at]. *)

val window : row:int -> col:int -> rows:int -> cols:int -> t -> t
(** [window ~row ~col ~rows ~cols g] is the block of [rows × cols] cells of [g]
    from cell [(row, col)]. Its origin is [world g] of pixel point [(row, col)],
    so its cells lie where [g]'s do up to that rounding. It may reach beyond
    [g], as a padded tile does.

    Raises [Invalid_argument] if [rows] or [cols] is negative. *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same system, size and
    numbers, compared exactly. *)
