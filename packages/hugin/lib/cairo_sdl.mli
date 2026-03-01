(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cairo-SDL integration.

    {b Internal module.} Manages a shared ARGB8888 surface between Cairo and SDL
    for interactive rendering. *)

type t
(** The type for Cairo-SDL contexts. *)

val create : width:int -> height:int -> title:string -> t
(** [create ~width ~height ~title] initializes SDL, creates a resizable window,
    and sets up a shared Cairo surface.

    Raises [Failure] if SDL initialization fails. *)

val context : t -> Ucairo.t
(** [context t] is the current Cairo drawing context. Valid until the next
    {!present} or {!resize}. *)

val width : t -> int
(** [width t] is the current surface width in pixels. *)

val height : t -> int
(** [height t] is the current surface height in pixels. *)

val present : t -> unit
(** [present t] flushes the Cairo surface to the SDL window and prepares a fresh
    Cairo context for the next frame. *)

val resize : t -> unit
(** [resize t] updates the surface dimensions to match the renderer output size.
    No-op if the size has not changed. *)

val destroy : t -> unit
(** [destroy t] frees all SDL and Cairo resources and calls {!Usdl.quit}. *)
