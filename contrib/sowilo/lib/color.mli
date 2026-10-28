(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Color space conversion and adjustment, the implementation of {!Sowilo}'s
    color functions.

    The functions take the channels from the last axis. Only [to_grayscale]
    checks the rank. *)

val to_grayscale : 'd Nx.float32_t -> 'd Nx.float32_t
val rgb_to_hsv : 'd Nx.float32_t -> 'd Nx.float32_t
val hsv_to_rgb : 'd Nx.float32_t -> 'd Nx.float32_t
val adjust_brightness : float -> 'd Nx.float32_t -> 'd Nx.float32_t
val adjust_contrast : float -> 'd Nx.float32_t -> 'd Nx.float32_t
val adjust_saturation : float -> 'd Nx.float32_t -> 'd Nx.float32_t
val adjust_hue : float -> 'd Nx.float32_t -> 'd Nx.float32_t
val adjust_gamma : float -> 'd Nx.float32_t -> 'd Nx.float32_t
val invert : 'd Nx.float32_t -> 'd Nx.float32_t
