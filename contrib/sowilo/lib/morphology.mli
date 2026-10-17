(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Morphological operations, the implementation of {!Sowilo}'s morphology
    functions.

    A structuring element is a [[kH; kW]] tensor whose non-zero entries
    select the neighbourhood, centred at row [kH / 2] and column [kW / 2].
    Erosion counts pixels outside the image as the dtype's largest value,
    and dilation counts them as zero. The operations raise [Invalid_argument]
    for a structuring element with no non-zero entry, or an image whose
    rank is neither 3 nor 4. *)

type kernel_shape = Rect | Cross | Ellipse

val structuring_element : kernel_shape -> int * int -> Nx.uint8_t
val erode : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
val dilate : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
val opening : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
val closing : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
val morphological_gradient :
  kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
