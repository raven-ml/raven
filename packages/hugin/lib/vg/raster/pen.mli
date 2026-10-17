(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Strokes as coverage.

    A subpath is stroked in its own coordinates, where the pen is round, as
    the polygon of its outline, along its segments, joins and caps. Every
    outline winds the same way, so where two overlap their windings add,
    and the [`Nonzero] rule covers their union. Vertices are mapped to the
    device as they are made and deposited in a {!Cover.t}. *)

val stroke :
  Cover.t ->
  Surface.clip ->
  Hugin_gg.Affine.t ->
  pen:float ->
  Hugin_gg.Stroke.t ->
  Hugin_gg.Path.t ->
  unit
(** [stroke cover clip m ~pen s path] deposits in [cover] the outline of
    [path] stroked with [s], its lengths multiplied by [pen], in the
    coordinates that [m] maps to the device. A pen that [m] maps beyond the
    range of floats deposits nothing. [cover] must have been started on
    [clip] ({!Cover.start}). *)
