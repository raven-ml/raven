(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Window = struct
  type t
end

module Renderer = struct
  type t
end

module Surface = struct
  type t
end

module Texture = struct
  type t
end

module Event = struct
  type t
end

type 'a result = Ok of 'a | Error of string

module Init = struct
  let video = 0x00000020 (* SDL_INIT_VIDEO *)
end

module Hint = struct
  let render_scale_quality = "SDL_RENDER_SCALE_QUALITY"
end

module Window_flags = struct
  let windowed = 0x00000004 (* SDL_WINDOW_SHOWN *)
  let resizable = 0x00000020 (* SDL_WINDOW_RESIZABLE *)
  let allow_highdpi = 0x00002000 (* SDL_WINDOW_ALLOW_HIGHDPI *)
end

module Renderer_flags = struct
  let accelerated = 0x00000002 (* SDL_RENDERER_ACCELERATED *)
  let presentvsync = 0x00000004 (* SDL_RENDERER_PRESENTVSYNC *)
end

module Pixel = struct
  let format_argb8888 = Int32.of_int 0x16362004 (* SDL_PIXELFORMAT_ARGB8888 *)
end

(* External C function declarations *)
external ml_sdl_init : int -> int = "caml_sdl_init"
external ml_sdl_quit : unit -> unit = "caml_sdl_quit"
external ml_sdl_get_error : unit -> string = "caml_sdl_get_error"
external ml_sdl_set_hint : string -> string -> bool = "caml_sdl_set_hint"

external ml_sdl_create_window : string -> int -> int -> int -> Window.t option
  = "caml_sdl_create_window"

external ml_sdl_destroy_window : Window.t -> unit = "caml_sdl_destroy_window"

external ml_sdl_create_renderer : Window.t -> int -> Renderer.t option
  = "caml_sdl_create_renderer"

external ml_sdl_destroy_renderer : Renderer.t -> unit
  = "caml_sdl_destroy_renderer"

external ml_sdl_get_renderer_output_size : Renderer.t -> (int * int) option
  = "caml_sdl_get_renderer_output_size"

external ml_sdl_render_clear : Renderer.t -> int = "caml_sdl_render_clear"

external ml_sdl_render_copy : Renderer.t -> Texture.t -> int
  = "caml_sdl_render_copy"

external ml_sdl_render_present : Renderer.t -> unit = "caml_sdl_render_present"

external ml_sdl_create_rgb_surface_with_format :
  int -> int -> int -> int32 -> Surface.t option
  = "caml_sdl_create_rgb_surface_with_format"

external ml_sdl_free_surface : Surface.t -> unit = "caml_sdl_free_surface"

external ml_sdl_get_surface_pitch : Surface.t -> int
  = "caml_sdl_get_surface_pitch"

external ml_sdl_get_surface_pixels :
  Surface.t ->
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_sdl_get_surface_pixels"

external ml_sdl_create_texture_from_surface :
  Renderer.t -> Surface.t -> Texture.t option
  = "caml_sdl_create_texture_from_surface"

external ml_sdl_destroy_texture : Texture.t -> unit = "caml_sdl_destroy_texture"

external ml_sdl_create_event : unit -> Event.t
  = "caml_sdl_alloc_event_storage" (* Renamed stub *)

external ml_sdl_wait_event : Event.t -> int
  = "caml_sdl_wait_event" (* Returns 1 on event, 0 on quit, -1 on error *)

external ml_sdl_get_event_type : Event.t -> int = "caml_sdl_get_event_type"

external ml_sdl_get_window_event_id : Event.t -> int
  = "caml_sdl_get_window_event_id"

external ml_sdl_get_event_keycode : Event.t -> int
  = "caml_sdl_get_event_keycode"

(* Wrappers for error handling and flag combining *)
let get_error () = ml_sdl_get_error ()
let init flags = if ml_sdl_init flags = 0 then Ok () else Error (get_error ())
let quit () = ml_sdl_quit ()
let set_hint name value = ml_sdl_set_hint name value

let create_window ~title ~w ~h flags =
  match ml_sdl_create_window title w h flags with
  | Some w -> Ok w
  | None -> Error (get_error ())

let destroy_window win = ml_sdl_destroy_window win

let create_renderer ~flags win =
  match ml_sdl_create_renderer win flags with
  | Some r -> Ok r
  | None -> Error (get_error ())

let destroy_renderer ren = ml_sdl_destroy_renderer ren

let get_renderer_output_size ren =
  match ml_sdl_get_renderer_output_size ren with
  | Some (w, h) -> Ok (w, h)
  | None -> Error (get_error ())

let render_clear ren =
  if ml_sdl_render_clear ren = 0 then Ok () else Error (get_error ())

let render_copy ren tex =
  if ml_sdl_render_copy ren tex = 0 then Ok () else Error (get_error ())

let render_present ren = ml_sdl_render_present ren

let create_rgb_surface_with_format ~w ~h ~depth fmt =
  match ml_sdl_create_rgb_surface_with_format w h depth fmt with
  | Some s -> Ok s
  | None -> Error (get_error ())

let free_surface surf = ml_sdl_free_surface surf
let get_surface_pitch surf = ml_sdl_get_surface_pitch surf
let get_surface_pixels surf = ml_sdl_get_surface_pixels surf

let create_texture_from_surface ren surf =
  match ml_sdl_create_texture_from_surface ren surf with
  | Some t -> Ok t
  | None -> Error (get_error ())

let destroy_texture tex = ml_sdl_destroy_texture tex
let create_event () = ml_sdl_create_event ()

let wait_event event_opt =
  match event_opt with
  | None ->
      Error "wait_event requires an allocated event structure"
      (* SDL_WaitEvent(NULL) is valid but we need storage *)
  | Some event -> (
      match ml_sdl_wait_event event with
      | 1 -> Ok true (* Event received *)
      | 0 -> Ok false (* SDL_QUIT received by the C stub *)
      | _ -> Error (get_error ()))

module Event_type = struct
  type t = [ `Quit | `Window_event | `Key_down | `Unknown of int ]

  let from_int = function
    | 0x100 -> `Quit (* SDL_QUIT *)
    | 0x200 -> `Window_event (* SDL_WINDOWEVENT *)
    | 0x300 -> `Key_down (* SDL_KEYDOWN *)
    | other -> `Unknown other
end

module Window_event_id = struct
  type t = [ `Resized | `Size_changed | `Exposed | `Close | `Unknown of int ]

  let from_int = function
    | 5 -> `Resized (* SDL_WINDOWEVENT_RESIZED *)
    | 6 -> `Size_changed (* SDL_WINDOWEVENT_SIZE_CHANGED *)
    | 2 -> `Exposed (* SDL_WINDOWEVENT_EXPOSED *)
    | 14 -> `Close (* SDL_WINDOWEVENT_CLOSE *)
    | other -> `Unknown other
end

let get_event_type event = Event_type.from_int (ml_sdl_get_event_type event)

let get_window_event_id event =
  Window_event_id.from_int (ml_sdl_get_window_event_id event)

let get_event_keycode event = ml_sdl_get_event_keycode event

module Keycode = struct
  let escape = 27
  let q = int_of_char 'q'
end
