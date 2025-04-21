(** Minimal SDL2 Bindings *)

(** Opaque types for SDL objects *)
module Window : sig
  type t
end

module Renderer : sig
  type t
end

module Surface : sig
  type t
end

module Texture : sig
  type t
end

module Event : sig
  type t
end

(** Result type for functions that can fail *)
type 'a result = Ok of 'a | Error of string

(** Initialization and Shutdown *)
module Init : sig
  val video : int
end

val init : int -> unit result
val quit : unit -> unit
val get_error : unit -> string

(** Hints *)
module Hint : sig
  val render_scale_quality : string (* SDL hints are strings *)
end

val set_hint : string -> string -> bool

(** Window Management *)
module Window_flags : sig
  val windowed : int
  val resizable : int
  val allow_highdpi : int
end

val create_window : title:string -> w:int -> h:int -> int -> Window.t result
val destroy_window : Window.t -> unit

(** Renderer Management *)
module Renderer_flags : sig
  val accelerated : int
  val presentvsync : int
end

val create_renderer : flags:int -> Window.t -> Renderer.t result
val destroy_renderer : Renderer.t -> unit
val get_renderer_output_size : Renderer.t -> (int * int) result
val render_clear : Renderer.t -> unit result
val render_copy : Renderer.t -> Texture.t -> unit result
val render_present : Renderer.t -> unit

(** Surface Management *)
module Pixel : sig
  (* We only need ARGB8888 which SDL maps internally. Pass the
     SDL_PIXELFORMAT_ARGB8888 enum value directly. We'll define it as an int32
     constant in the .ml file. *)
  val format_argb8888 : int32
end

val create_rgb_surface_with_format :
  w:int -> h:int -> depth:int -> int32 -> Surface.t result

val free_surface : Surface.t -> unit
val get_surface_pitch : Surface.t -> int

val get_surface_pixels :
  Surface.t ->
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val create_texture_from_surface : Renderer.t -> Surface.t -> Texture.t result
(** Texture Management *)

val destroy_texture : Texture.t -> unit

(** Event Handling *)
module Event_type : sig
  type t = [ `Quit | `Window_event | `Unknown of int ]
end

module Window_event_id : sig
  type t = [ `Resized | `Size_changed | `Exposed | `Unknown of int ]
end

val create_event : unit -> Event.t (* Allocates storage for one event *)
val wait_event : Event.t option -> bool result
(* Fills the provided event, returns Ok true on success, Ok false on SDL_QUIT,
   Error on error *)

val get_event_type : Event.t -> Event_type.t
val get_window_event_id : Event.t -> Window_event_id.t
