(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Geometric transforms, the implementation of {!Sowilo}'s transform
    functions.

    Each function raises [Invalid_argument] for an image whose rank is
    neither 3 nor 4, except {!rotate90} with [k] a multiple of 4, which
    returns the image. *)

type interpolation = Nearest | Bilinear

val resize :
  ?interpolation:interpolation ->
  height:int ->
  width:int ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t

val crop :
  y:int -> x:int -> height:int -> width:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t

val center_crop : height:int -> width:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
val hflip : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
val vflip : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
val rotate90 : ?k:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t

val pad :
  ?value:float -> int * int * int * int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
