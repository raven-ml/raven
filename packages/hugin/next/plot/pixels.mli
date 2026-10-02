(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images: converting pixels. *)

module Box2 := Hugin_next_gg.Box2
module Picture := Hugin_next_vg.Picture

val rgba : ('a, 'b) Nx.t -> Nx.uint8_t
(** [rgba px] is the image [px], of shape [[|h; w|]] or [[|h; w; c|]], as
    {!Picture.image} takes it, on the device of [px]: [uint8] values as they
    are, and floating-point values clamped into \[[0];[1]\] and scaled to
    \[[0];[255]\], a pixel with a NaN component transparent. *)
