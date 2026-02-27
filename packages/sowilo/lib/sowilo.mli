(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable computer vision on {!Rune}.

    Sowilo provides image processing operations expressed purely through {!Nx}
    tensor operations. All operations are compatible with {!Rune.grad}
    and {!Rune.vmap}.

    {1:conventions Image conventions}

    Images are {!Nx.t} tensors with channels-last layout:
    - Single image: [[H; W; C]] (height, width, channels).
    - Batch: [[N; H; W; C]] (batch, height, width, channels).
    - Grayscale: [C = 1], RGB: [C = 3], RGBA: [C = 4].

    Operations expect float32 tensors with values in \[0, 1\]. Use {!to_float}
    and {!to_uint8} to convert between integer and float representations. *)

(** {1:converting Type conversion and preprocessing} *)

val to_float : ('a, 'b) Nx.t -> Nx.float32_t
(** [to_float img] is [img] cast to float32 and scaled to \[0, 1\] by dividing
    by 255. *)

val to_uint8 : Nx.float32_t -> Nx.uint8_t
(** [to_uint8 img] is [img] scaled from \[0, 1\] to \[0, 255\] and cast to
    uint8. Values are clipped to \[0, 1\] before scaling. *)

val normalize :
  mean:float list -> std:float list -> Nx.float32_t -> Nx.float32_t
(** [normalize ~mean ~std img] is per-channel normalization:
    [(img - mean) / std]. [mean] and [std] must have the same length as the
    channel dimension.

    Raises [Invalid_argument] if [mean] or [std] length does not match the
    number of channels. *)

val threshold : float -> Nx.float32_t -> Nx.float32_t
(** [threshold t img] is [1.0] where [img > t], [0.0] elsewhere. *)

(** {1:color Color space conversion and adjustment} *)

val to_grayscale : Nx.float32_t -> Nx.float32_t
(** [to_grayscale img] converts RGB to single-channel grayscale using ITU-R
    BT.601 weights: [0.299 * R + 0.587 * G + 0.114 * B]. Input must have
    [C >= 3]. Output has [C = 1]. *)

val rgb_to_hsv : Nx.float32_t -> Nx.float32_t
(** [rgb_to_hsv img] converts RGB \[0, 1\] to HSV. H is in \[0, 1\] (normalized
    from \[0, 360\]), S and V are in \[0, 1\]. *)

val hsv_to_rgb : Nx.float32_t -> Nx.float32_t
(** [hsv_to_rgb img] converts HSV back to RGB \[0, 1\]. *)

val adjust_brightness : float -> Nx.float32_t -> Nx.float32_t
(** [adjust_brightness factor img] scales pixel values by [factor] and clips to
    \[0, 1\]. *)

val adjust_contrast : float -> Nx.float32_t -> Nx.float32_t
(** [adjust_contrast factor img] adjusts contrast around the per-channel mean.
    [0] produces solid gray, [1] is the original image. *)

val adjust_saturation : float -> Nx.float32_t -> Nx.float32_t
(** [adjust_saturation factor img] adjusts color saturation via HSV. [0]
    produces grayscale, [1] is the original image. *)

val adjust_hue : float -> Nx.float32_t -> Nx.float32_t
(** [adjust_hue delta img] rotates hue by [delta]. [delta] is in \[-0.5, 0.5\],
    corresponding to a full rotation of the hue circle. *)

val adjust_gamma : float -> Nx.float32_t -> Nx.float32_t
(** [adjust_gamma gamma img] applies gamma correction: [img ** gamma]. *)

val invert : Nx.float32_t -> Nx.float32_t
(** [invert img] is [1.0 - img]. *)

(** {1:transform Geometric transforms} *)

(** The type for interpolation methods. *)
type interpolation =
  | Nearest  (** Nearest-neighbor interpolation. *)
  | Bilinear  (** Bilinear interpolation (default). *)

val resize :
  ?interpolation:interpolation ->
  height:int ->
  width:int ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [resize ~height ~width img] resizes to target dimensions. [interpolation]
    defaults to {!Bilinear}. Casts to float32 internally for bilinear
    interpolation.

    Raises [Invalid_argument] if [height] or [width] is not positive. *)

val crop :
  y:int -> x:int -> height:int -> width:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [crop ~y ~x ~height ~width img] extracts a rectangular region starting at
    [(y, x)] with the given dimensions.

    Raises [Invalid_argument] if the region exceeds image bounds. *)

val center_crop : height:int -> width:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [center_crop ~height ~width img] crops a centered rectangle.

    Raises [Invalid_argument] if [height] or [width] exceeds the image
    dimensions. *)

val hflip : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [hflip img] flips horizontally (left to right). *)

val vflip : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [vflip img] flips vertically (top to bottom). *)

val rotate90 : ?k:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [rotate90 img] rotates by [k * 90] degrees counter-clockwise. [k] defaults
    to [1]. Negative values rotate clockwise. *)

val pad :
  ?value:float -> int * int * int * int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [pad (top, bottom, left, right) img] zero-pads the spatial dimensions.
    [value] defaults to [0.0]. *)

(** {1:filter Spatial filtering} *)

val gaussian_blur : sigma:float -> ?ksize:int -> Nx.float32_t -> Nx.float32_t
(** [gaussian_blur ~sigma img] applies isotropic Gaussian blur using separable
    convolution. [ksize] defaults to [2 * ceil(3 * sigma) + 1], capturing 99.7%
    of the distribution.

    Raises [Invalid_argument] if [ksize] is even or not positive. *)

val box_blur : ksize:int -> Nx.float32_t -> Nx.float32_t
(** [box_blur ~ksize img] applies a [ksize * ksize] averaging filter.

    Raises [Invalid_argument] if [ksize] is not positive. *)

val median_blur : ksize:int -> Nx.float32_t -> Nx.float32_t
(** [median_blur ~ksize img] applies a median filter.

    {b Note.} Not differentiable: uses sort internally, gradient is zero almost
    everywhere.

    Raises [Invalid_argument] if [ksize] is not a positive odd integer. *)

val filter2d : Nx.float32_t -> Nx.float32_t -> Nx.float32_t
(** [filter2d kernel img] applies a custom 2D convolution [kernel] to [img].
    [kernel] has shape [[kH; kW]]. Applied independently to each channel with
    [Same] padding. *)

val unsharp_mask : sigma:float -> ?amount:float -> Nx.float32_t -> Nx.float32_t
(** [unsharp_mask ~sigma img] sharpens by subtracting a Gaussian blur:
    [img + amount * (img - gaussian_blur ~sigma img)]. [amount] defaults to
    [1.0]. *)

(** {1:morphology Morphological operations} *)

(** The type for structuring element shapes. *)
type kernel_shape =
  | Rect  (** Full rectangle. *)
  | Cross  (** Cross-shaped element. *)
  | Ellipse  (** Elliptical element. *)

val structuring_element : kernel_shape -> int * int -> Nx.uint8_t
(** [structuring_element shape (h, w)] is a structuring element of the given
    [shape] and size. [h] and [w] must be positive odd integers.

    Raises [Invalid_argument] if [h] or [w] is not positive or not odd. *)

val erode : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [erode ~kernel img] replaces each pixel with the minimum over the
    kernel-shaped neighborhood. *)

val dilate : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [dilate ~kernel img] replaces each pixel with the maximum over the
    kernel-shaped neighborhood. *)

val opening : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [opening ~kernel img] is [dilate ~kernel (erode ~kernel img)]. Removes small
    bright regions. *)

val closing : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [closing ~kernel img] is [erode ~kernel (dilate ~kernel img)]. Fills small
    dark regions. *)

val morphological_gradient : kernel:Nx.uint8_t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [morphological_gradient ~kernel img] is
    [dilate ~kernel img - erode ~kernel img]. Highlights edges. *)

(** {1:edge Edge detection} *)

val sobel : ?ksize:int -> Nx.float32_t -> Nx.float32_t * Nx.float32_t
(** [sobel img] computes Sobel gradients. Returns [(gx, gy)] where [gx] is the
    horizontal gradient and [gy] is the vertical gradient. [ksize] defaults to
    [3]. Input must have [C = 1]. *)

val scharr : Nx.float32_t -> Nx.float32_t * Nx.float32_t
(** [scharr img] computes Scharr gradients, which are more rotationally accurate
    than Sobel. Returns [(gx, gy)]. Input must have [C = 1]. *)

val laplacian : ?ksize:int -> Nx.float32_t -> Nx.float32_t
(** [laplacian img] computes the Laplacian (sum of second spatial derivatives).
    [ksize] defaults to [3]. Input must have [C = 1]. *)

val canny :
  low:float -> high:float -> ?sigma:float -> Nx.float32_t -> Nx.float32_t
(** [canny ~low ~high img] applies the Canny edge detector. Returns [1.0] for
    edge pixels, [0.0] otherwise. [low] and [high] are the hysteresis
    thresholds. [sigma] controls the initial Gaussian blur and defaults to
    [1.4]. Input must have [C = 1].

    {b Note.} Not differentiable: uses non-maximum suppression and hysteresis
    thresholding. *)
