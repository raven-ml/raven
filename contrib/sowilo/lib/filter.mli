(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Spatial filters, the implementation of {!Sowilo}'s filtering functions.

    Every filter counts pixels outside the image as zero, and raises
    [Invalid_argument] for an image whose rank is neither 3 nor 4. *)

val gaussian_blur :
  sigma:float -> ?ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t

val box_blur : ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t
val median_blur : ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t
val filter2d : 'd Nx.float32_t -> 'd Nx.float32_t -> 'd Nx.float32_t

val unsharp_mask :
  sigma:float -> ?amount:float -> 'd Nx.float32_t -> 'd Nx.float32_t
