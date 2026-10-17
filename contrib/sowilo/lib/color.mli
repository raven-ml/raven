(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Color space conversion and adjustment, the implementation of {!Sowilo}'s
    color functions.

    The functions take the channels from the last axis. Only [to_grayscale]
    checks the rank. *)

val to_grayscale : Nx.float32_t -> Nx.float32_t
val rgb_to_hsv : Nx.float32_t -> Nx.float32_t
val hsv_to_rgb : Nx.float32_t -> Nx.float32_t
val adjust_brightness : float -> Nx.float32_t -> Nx.float32_t
val adjust_contrast : float -> Nx.float32_t -> Nx.float32_t
val adjust_saturation : float -> Nx.float32_t -> Nx.float32_t
val adjust_hue : float -> Nx.float32_t -> Nx.float32_t
val adjust_gamma : float -> Nx.float32_t -> Nx.float32_t
val invert : Nx.float32_t -> Nx.float32_t
