(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pixel surfaces and clips, and compositing onto them.

    A surface holds premultiplied RGBA in levels, \[[0];[255]\] per
    component, row by row, as single-precision floats. No pixel is rounded
    to a level until {!to_straight}, so that translucent drawing over many
    blends reaches the colour it composites to. The functions here check no
    bound, so indices and windows must lie within the surface. *)

(* Surfaces *)

type pixels =
  (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for pixel storage. The four components of pixel [(x, y)] of a
    surface of width [w] start at index [((y * w) + x) * 4]. *)

type t = { w : int; h : int; px : pixels }
(** The type for surfaces of [w] by [h] pixels. *)

val create : int -> int -> t
(** [create w h] is a transparent surface of [w] by [h] pixels. *)

val clear : t -> unit
(** [clear s] makes every pixel of [s] transparent. *)

(* Clips *)

type clip = {
  x0 : int;
  y0 : int;
  x1 : int;  (** Exclusive. *)
  y1 : int;  (** Exclusive. *)
  mask : float array option;
      (** The fraction of each pixel of the window that the clip lets
          through, row by row over the window, or [None] for all of each. *)
  box : Hugin_gg.Box2.t option;
      (** The device box of the clip paths the clip comes from, or [None] for
          none. *)
}
(** The type for clips: a window of pixels and what it lets through. *)

val whole : t -> clip
(** [whole s] is the clip that lets all of [s] through. *)

val shut : clip
(** [shut] is a clip that lets nothing through. *)

val is_shut : clip -> bool
(** [is_shut c] is [true] iff [c]'s window is empty. *)

val mask_at : clip -> int -> int -> float
(** [mask_at c x y] is the fraction of pixel [(x, y)] that [c] lets through.
    [(x, y)] must lie in [c]'s window if [c] has a mask. *)

(* Compositing *)

val blend :
  pixels -> int -> float -> float -> float -> float -> float -> unit
(** [blend px i sr sg sb sa k] composites the premultiplied colour
    [(sr, sg, sb, sa)], components in \[[0];[255]\], with coverage [k] over
    the pixel whose components start at index [i] of [px]. *)

val composite : t -> clip -> t -> int -> int -> float -> unit
(** [composite dst clip src ox oy alpha] composites [src], its top left
    pixel at [(ox, oy)] of [dst], faded by [alpha], through [clip]. *)

val to_straight : t -> (int, Bigarray.int8_unsigned_elt) Nx.t
(** [to_straight s] is the pixels of [s] as a [[|h; w; 4|]] tensor of levels
    with straight alpha, each component the level nearest its exact value.
    A pixel whose alpha rounds to [0] is [(0, 0, 0, 0)]. *)
