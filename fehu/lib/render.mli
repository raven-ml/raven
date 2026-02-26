(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Visualization primitives.

    {!image} is the standard frame type for rgb-rendered environments.
    {!rollout} runs a policy and feeds rendered frames to a user-provided sink.
*)

(** {1:pixel Pixel formats} *)

module Pixel : sig
  (** The type for pixel formats. *)
  type format =
    | Rgb  (** 3 channels. *)
    | Rgba  (** 4 channels. *)
    | Gray  (** 1 channel. *)

  val channels : format -> int
  (** [channels fmt] is the number of channels for [fmt]. *)
end

(** {1:image Images} *)

type image = {
  width : int;  (** Width in pixels. *)
  height : int;  (** Height in pixels. *)
  pixel_format : Pixel.format;  (** Pixel layout. *)
  data : (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
      (** Raw pixel data of length [width * height * channels]. *)
}
(** The type for rendered frames. *)

val image :
  width:int ->
  height:int ->
  ?pixel_format:Pixel.format ->
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  image
(** [image ~width ~height data] constructs a frame. [pixel_format] defaults to
    [Rgb].

    Raises [Invalid_argument] if [Bigarray.Array1.dim data] does not equal
    [width * height * channels]. *)

(** {1:rollout Rollout} *)

val rollout :
  ('obs, 'act, image) Env.t ->
  policy:('obs -> 'act) ->
  steps:int ->
  sink:(image -> unit) ->
  unit ->
  unit
(** [rollout env ~policy ~steps ~sink ()] runs [policy] in [env] for up to
    [steps] steps. Each rendered frame is passed to [sink]. The environment is
    reset at the start and on episode boundaries. *)

(** {1:recording Recording} *)

val on_render :
  sink:(image -> unit) -> ('obs, 'act, image) Env.t -> ('obs, 'act, image) Env.t
(** [on_render ~sink env] wraps [env] so that every rendered frame after
    {!Env.reset} and {!Env.step} is passed to [sink]. The wrapper is
    transparent: observations, actions, rewards, and termination signals pass
    through unchanged. *)
