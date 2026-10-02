(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gathering the cells of images that raster output samples. *)

module Box2 := Hugin_gg.Box2
module Picture := Hugin_vg.Picture

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
(** [gather p px] is the gathered image of the [[|h; w; c|]] image [px], on its
    device, as RGBA: the samples outside the source box are transparent. *)

val gathered : density:float -> Picture.t -> Picture.t
(** [gathered ~density p] is [p] with each image outside a transform or a stamp
    that has more than 4 cells per device pixel along an axis replaced by its
    gather ({!plan}). It draws at the density what [p] draws. *)
