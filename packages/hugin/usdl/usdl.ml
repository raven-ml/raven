(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Init / quit *)

external init : unit -> unit = "caml_usdl_init"
external quit : unit -> unit = "caml_usdl_quit"

(* Handle types *)

type renderer
type surface
type texture

(* Window *)

module Window = struct
  type t

  external create : title:string -> w:int -> h:int -> t
    = "caml_usdl_window_create"

  external destroy : t -> unit = "caml_usdl_window_destroy"
end

(* Renderer *)

module Renderer = struct
  type t = renderer

  external create : Window.t -> t = "caml_usdl_renderer_create"
  external output_size : t -> int * int = "caml_usdl_renderer_output_size"
  external clear : t -> unit = "caml_usdl_renderer_clear"
  external copy : t -> texture -> unit = "caml_usdl_renderer_copy"
  external present : t -> unit = "caml_usdl_renderer_present"
  external destroy : t -> unit = "caml_usdl_renderer_destroy"
end

(* Surface *)

module Surface = struct
  type t = surface

  external create_argb8888 : w:int -> h:int -> t
    = "caml_usdl_surface_create_argb8888"

  external pitch : t -> int = "caml_usdl_surface_pitch"

  external pixels :
    t -> (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    = "caml_usdl_surface_pixels"

  external destroy : t -> unit = "caml_usdl_surface_destroy"
end

(* Texture *)

module Texture = struct
  type t = texture

  external of_surface : Renderer.t -> Surface.t -> t
    = "caml_usdl_texture_of_surface"

  external destroy : t -> unit = "caml_usdl_texture_destroy"
end

(* Event *)

module Event = struct
  type t
  type event_type = [ `Quit | `Window_event | `Key_down | `Unknown of int ]

  type window_event =
    [ `Resized | `Size_changed | `Exposed | `Close | `Unknown of int ]

  external create : unit -> t = "caml_usdl_event_create"
  external wait : t -> bool = "caml_usdl_event_wait"
  external raw_type : t -> int = "caml_usdl_event_type" [@@noalloc]
  external raw_window_id : t -> int = "caml_usdl_event_window_id" [@@noalloc]
  external keycode : t -> int = "caml_usdl_event_keycode" [@@noalloc]

  let typ t =
    match raw_type t with
    | 0x100 -> `Quit
    | 0x200 -> `Window_event
    | 0x300 -> `Key_down
    | n -> `Unknown n

  let window_event_id t =
    match raw_window_id t with
    | 5 -> `Resized
    | 6 -> `Size_changed
    | 2 -> `Exposed
    | 14 -> `Close
    | n -> `Unknown n
end

(* Keycodes *)

module Keycode = struct
  let escape = 27
  let q = Char.code 'q'
end
