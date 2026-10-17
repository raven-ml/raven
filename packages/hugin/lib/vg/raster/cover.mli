(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Exact area coverage of polygons.

    An accumulator takes the edges of one primitive at a time and yields the
    fraction of each pixel that the primitive's area covers under a fill
    rule. Edges are cut to a window, the pixel rectangle of a clip, and what
    lies left or right of it winds from the window's sides, so a closed
    polygon may extend past the window.

    One accumulator serves every surface of a render: {!start} sets the
    window, {!line} deposits edges, and {!paint}, {!take} or {!clear} reads
    or discards what they touched and leaves the accumulator empty. The
    edges deposited between {!start} and that read must form closed
    polygons. *)

type t
(** The type for accumulators. *)

val create : unit -> t
(** [create ()] is an empty accumulator. *)

val start : t -> w:int -> h:int -> Surface.clip -> unit
(** [start c ~w ~h clip] prepares [c] for the edges of a primitive on a
    surface of [w] by [h] pixels, in the window of [clip]. [c] must be
    empty. *)

val line : t -> float -> float -> float -> float -> unit
(** [line c x0 y0 x1 y1] deposits the edge from [(x0, y0)] to [(x1, y1)], in
    device pixels. Coordinates are clamped to [±1e15], and NaN is taken as
    [1e15]. *)

val paint :
  t ->
  Surface.t ->
  Surface.clip ->
  Hugin_vg.Picture.rule ->
  float ->
  float ->
  float ->
  float ->
  unit
(** [paint c s clip rule sr sg sb sa] composites the premultiplied colour
    [(sr, sg, sb, sa)] over the pixels of [s] that the deposited edges cover
    under [rule], through [clip], and empties [c]. A coverage of at most
    a two-thousandth is none. *)

val take :
  t ->
  Surface.clip ->
  Hugin_vg.Picture.rule ->
  float array ->
  ox:int ->
  oy:int ->
  dw:int ->
  unit
(** [take c clip rule dst ~ox ~oy ~dw] writes into [dst], a window of rows of
    [dw] pixels from [(ox, oy)], the coverage under [rule] of each pixel of
    {!touched}'s rectangle times what [clip] lets through, and empties [c].
    The other pixels of [dst] are left as they are. *)

val clear : t -> unit
(** [clear c] empties [c] without reading it. *)

val touched : t -> int * int * int * int
(** [touched c] is the rectangle [(x0, y0, x1, y1)], [x1] and [y1]
    exclusive, of the window's pixels that the deposited edges touch, and
    [(0, 0, 0, 0)] when nothing was deposited. *)
