(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Overlays for augmenting rendered frames with diagnostic annotations. *)

type ctx = {
  step_idx : int;
  episode_idx : int;
  info : Fehu.Info.t;
  action : Fehu.Space.Value.t option;
  value : float option;
  log_prob : float option;
  reward : float;
  done_ : bool;
}
(** Context provided to overlays for each frame. *)

type t = Fehu.Render.image -> ctx -> Fehu.Render.image
(** Overlay transformation applied to a rendered image. *)

val compose : t list -> t
(** Compose multiple overlays sequentially. *)

val text : ?pos:int * int -> ?color:int * int * int -> (ctx -> string) -> t
(** Draw text at the given position (default: [10, 20]) in RGB color. *)

val bar :
  ?pos:int * int ->
  ?size:int * int ->
  ?color:int * int * int ->
  ?background:int * int * int ->
  value:(ctx -> float) ->
  t
(** Draw a horizontal progress bar representing a value in [0, 1]. *)

val identity : t
(** Overlay that leaves the image unchanged. *)
