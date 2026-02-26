(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integration of Cairo with Tsdl (SDL2 bindings). *)

type t
(** Abstract type representing an SDL window with a Cairo drawing context. *)

val create : width:int -> height:int -> title:string -> t
(** [create ~width ~height ~title] initializes an SDL window and associates it
    with a Cairo context drawing onto an internal SDL surface. Handles HighDPI
    scaling. *)

val context : t -> Cairo.context
(** [context t] returns the Cairo context associated with [t]. *)

val width : t -> int
(** [width t] returns the current drawable width of the rendering surface in
    pixels. *)

val height : t -> int
(** [height t] returns the current drawable height of the rendering surface in
    pixels. *)

val redraw : t -> unit
(** [redraw t] updates the SDL window by rendering the current state of the
    Cairo surface to an SDL texture and presenting it. *)

val resize : t -> unit
(** [resize t] checks the underlying window/renderer for size changes and
    recreates the SDL and Cairo surfaces if necessary to match the new drawable
    dimensions. This should be called when a resize event occurs. *)

val cleanup : t -> unit
(** [cleanup t] frees resources associated with [t] and shuts down SDL. *)
