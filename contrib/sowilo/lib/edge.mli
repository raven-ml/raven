(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Edge detection, the implementation of {!Sowilo}'s edge functions.

    Each function correlates its kernels with the image, pixels outside the
    image counting as zero, and raises [Invalid_argument] for an image whose
    rank is neither 3 nor 4. *)

val sobel : ?ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t * 'd Nx.float32_t
val scharr : 'd Nx.float32_t -> 'd Nx.float32_t * 'd Nx.float32_t
val laplacian : ?ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t

val canny :
  low:float -> high:float -> ?sigma:float -> 'd Nx.float32_t -> 'd Nx.float32_t
