(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Specification to scene resolution.

    {b Internal module.} Walks a {!Spec.t} tree, computes data bounds and
    layout, and emits a {!Scene.t} with all coordinates in device pixels. *)

type text_measurer = font:Theme.font -> string -> float * float
(** The type for text measurers. Returns [(width, height)] for a string rendered
    in the given font. *)

val resolve_prepared :
  text_measurer:text_measurer ->
  theme:Theme.t ->
  width:float ->
  height:float ->
  Prepared.t ->
  Scene.t
(** [resolve_prepared ~text_measurer ~theme ~width ~height prepared] is the
    resolved scene for [prepared] at the given dimensions. Layout-only work
    (pixel coordinates, text measurement) is done here; data work is already
    done in {!Prepared.compile}. *)

val resolve :
  text_measurer:text_measurer ->
  theme:Theme.t ->
  width:float ->
  height:float ->
  Spec.t ->
  Scene.t
(** [resolve ~text_measurer ~theme ~width ~height spec] is the resolved scene
    for [spec] at the given dimensions. Convenience wrapper that calls
    {!Prepared.compile} then {!resolve_prepared}. *)
