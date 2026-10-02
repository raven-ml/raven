(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images of cells: converting pixels, and gathering the cells that raster
    output samples. *)

module Box2 := Hugin_next_gg.Box2
module Picture := Hugin_next_vg.Picture

val rgba : ('a, 'b) Nx.t -> Nx.uint8_t
(** [rgba px] is the image [px], of shape [[|h; w|]] or [[|h; w; c|]], as
    {!Picture.image} takes it, on the device of [px]: [uint8] values as they
    are, and floating-point values clamped into \[[0];[1]\] and scaled to
    \[[0];[255]\], a pixel with a NaN component transparent. *)

(** {1:gathering Gathering} *)

type plan = {
  window : Box2.t;  (** Aligned with the device pixels. *)
  rows : int array;  (** The source row of each row, [-1] outside the box. *)
  cols : int array;  (** Likewise for columns. *)
}
(** A gather: the image whose pixel [(i, j)] is the source cell
    [(rows.(i), cols.(j))], painted over [window], draws at the density what the
    source image painted over the box draws. *)

val plan : density:float -> Box2.t -> rows:int -> cols:int -> plan option
(** [plan ~density box ~rows ~cols] is the gather of an image of [rows] by
    [cols] cells over [box], if it has more than 4 cells per device pixel along
    an axis, and [None] otherwise. *)

val gather : plan -> Nx.uint8_t -> Nx.uint8_t
(** [gather p px] is the gathered image of the [[|h; w; 4|]] image [px], on its
    device. *)
