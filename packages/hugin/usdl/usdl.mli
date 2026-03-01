(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minimal SDL2 bindings.

    Thin bindings covering window creation, renderer management, surface pixel
    access, and event polling. Designed for the Cairo-SDL integration layer; not
    a general-purpose SDL binding.

    All functions raise [Failure] on SDL errors. *)

(** {1:init Initialization} *)

val init : unit -> unit
(** [init ()] initializes SDL video and sets the render scale quality hint.

    Raises [Failure] if SDL initialization fails. *)

val quit : unit -> unit
(** [quit ()] shuts down SDL. *)

(** {1:handles Handle types} *)

type renderer
(** The type for SDL renderers. *)

type surface
(** The type for SDL surfaces. *)

type texture
(** The type for SDL textures. *)

(** {1:window Window} *)

module Window : sig
  type t
  (** The type for SDL windows. *)

  val create : title:string -> w:int -> h:int -> t
  (** [create ~title ~w ~h] is a resizable, high-DPI-aware window.

      Raises [Failure] if window creation fails. *)

  val destroy : t -> unit
  (** [destroy t] frees the window. Safe to call more than once. *)
end

(** {1:renderer Renderer} *)

module Renderer : sig
  type t = renderer
  (** The type for SDL renderers. *)

  val create : Window.t -> t
  (** [create win] is a hardware-accelerated, vsync-enabled renderer for [win].

      Raises [Failure] if renderer creation fails. *)

  val output_size : t -> int * int
  (** [output_size t] is [(w, h)] in pixels (accounting for high-DPI scaling).

      Raises [Failure] if the query fails. *)

  val clear : t -> unit
  (** [clear t] clears the render target. *)

  val copy : t -> texture -> unit
  (** [copy t tex] copies [tex] to the entire render target. *)

  val present : t -> unit
  (** [present t] presents the composed backbuffer. *)

  val destroy : t -> unit
  (** [destroy t] frees the renderer. Safe to call more than once. *)
end

(** {1:surface Surface} *)

module Surface : sig
  type t = surface
  (** The type for SDL surfaces. *)

  val create_argb8888 : w:int -> h:int -> t
  (** [create_argb8888 ~w ~h] is a 32-bit ARGB8888 surface.

      Raises [Failure] if allocation fails. *)

  val pitch : t -> int
  (** [pitch t] is the byte length of one row. *)

  val pixels :
    t -> (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  (** [pixels t] is the raw pixel buffer. The bigarray is a view onto
      SDL-managed memory; it must not outlive the surface. *)

  val destroy : t -> unit
  (** [destroy t] frees the surface. Safe to call more than once. *)
end

(** {1:texture Texture} *)

module Texture : sig
  type t = texture
  (** The type for SDL textures. *)

  val of_surface : renderer -> surface -> t
  (** [of_surface ren surf] creates a texture from [surf].

      Raises [Failure] if texture creation fails. *)

  val destroy : t -> unit
  (** [destroy t] frees the texture. Safe to call more than once. *)
end

(** {1:event Events} *)

module Event : sig
  type t
  (** The type for event storage. *)

  type event_type = [ `Quit | `Window_event | `Key_down | `Unknown of int ]
  (** The type for event kinds. *)

  type window_event =
    [ `Resized | `Size_changed | `Exposed | `Close | `Unknown of int ]
  (** The type for window event sub-kinds. *)

  val create : unit -> t
  (** [create ()] allocates event storage. *)

  val wait : t -> bool
  (** [wait t] blocks until an event arrives. Returns [true] if an event was
      received, [false] on error. Releases the runtime lock while blocking. *)

  val typ : t -> event_type
  (** [typ t] is the kind of the last received event. *)

  val window_event_id : t -> window_event
  (** [window_event_id t] is the window event sub-kind. Only meaningful when
      [typ t] is [`Window_event]. *)

  val keycode : t -> int
  (** [keycode t] is the key code. Only meaningful when [typ t] is [`Key_down].
  *)
end

(** {1:keycode Key codes} *)

module Keycode : sig
  val escape : int
  val q : int
end
