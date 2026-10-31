(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable computer vision on [Rune].

    Sowilo provides image processing operations expressed purely through {!Nx}
    tensor operations. All operations are compatible with [Rune.grad'] and
    [Rune.vmap'].

    {1:conventions Image conventions}

    Images are {!Nx.t} tensors with channels-last layout:
    - Single image: [[H; W; C]] (height, width, channels).
    - Batch: [[N; H; W; C]] (batch, height, width, channels).
    - Grayscale: [C = 1], RGB: [C = 3], RGBA: [C = 4].

    Operations expect float32 tensors with values in \[0, 1\]. Use {!to_float}
    and {!to_uint8} to convert between integer and float representations.

    {1:sets Device sets}

    An image lies on any device set, and an operation computes where it lies
    and gives its result on the image's set. The kernels and constants an
    operation needs are made beside the image, whole on each device of its
    placement: those it computes, as a Gaussian of a given sigma, are computed
    there, and those written as numbers, as {!sobel}'s taps or {!normalize}'s
    [mean], are made on the host and placed there. A structuring element
    ({!section-morphology}) is a host value: it shapes the computation. *)

(** {1:converting Type conversion and preprocessing} *)

val to_float : ('a, 'b, 'd) Nx.t -> 'd Nx.float32_t
(** [to_float img] is [img] cast to float32 and scaled to \[0, 1\] by dividing
    by 255. *)

val to_uint8 : 'd Nx.float32_t -> 'd Nx.uint8_t
(** [to_uint8 img] is [img] scaled from \[0, 1\] to \[0, 255\] and cast to
    uint8. Values are clipped to \[0, 1\] before scaling. *)

val normalize :
  mean:float list -> std:float list -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [normalize ~mean ~std img] is per-channel normalization:
    [(img - mean) / std]. [mean] and [std] must have the same length as the
    channel dimension.

    Raises [Invalid_argument] if [mean] or [std] length does not match the
    number of channels. *)

val threshold : float -> 'd Nx.float32_t -> 'd Nx.bool_t
(** [threshold t img] is [true] where [img]'s element is greater than [t],
    and [false] elsewhere, NaN elements included. [t] is stored in float32 as
    {!Nx.greater_s} stores it: for [t = 0.1], an element equal to the float32
    nearest [0.1] is [false]. *)

(** {1:color Color space conversion and adjustment} *)

val to_grayscale : 'd Nx.float32_t -> 'd Nx.float32_t
(** [to_grayscale img] converts RGB to single-channel grayscale using ITU-R
    BT.601 weights: [0.299 * R + 0.587 * G + 0.114 * B]. Input must have
    [C >= 3]; channels after the third, such as alpha, are ignored. Output has
    [C = 1].

    Raises [Invalid_argument] if [img]'s rank is less than 3 or [C < 3]. *)

val rgb_to_hsv : 'd Nx.float32_t -> 'd Nx.float32_t
(** [rgb_to_hsv img] converts RGB \[0, 1\] to HSV. H is in \[0, 1\] (normalized
    from \[0, 360\]), S and V are in \[0, 1\]. *)

val hsv_to_rgb : 'd Nx.float32_t -> 'd Nx.float32_t
(** [hsv_to_rgb img] converts HSV back to RGB \[0, 1\]. *)

val adjust_brightness : float -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [adjust_brightness factor img] scales pixel values by [factor] and clips to
    \[0, 1\]. *)

val adjust_contrast : float -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [adjust_contrast factor img] adjusts contrast around the per-channel mean.
    [0] produces solid gray, [1] is the original image. *)

val adjust_saturation : float -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [adjust_saturation factor img] adjusts color saturation via HSV. [0]
    produces grayscale, [1] is the original image. *)

val adjust_hue : float -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [adjust_hue delta img] rotates hue by [delta]. [delta] is in \[-0.5, 0.5\],
    corresponding to a full rotation of the hue circle. *)

val adjust_gamma : float -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [adjust_gamma gamma img] applies gamma correction: [img ** gamma]. *)

val invert : 'd Nx.float32_t -> 'd Nx.float32_t
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
  ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t
(** [resize ~height ~width img] resizes to target dimensions. [interpolation]
    defaults to {!Bilinear}. Casts to float32 internally for bilinear
    interpolation.

    Raises [Invalid_argument] if [height] or [width] is not positive. *)

val crop :
  y:int ->
  x:int ->
  height:int ->
  width:int ->
  ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t
(** [crop ~y ~x ~height ~width img] extracts a rectangular region starting at
    [(y, x)] with the given dimensions.

    Raises [Invalid_argument] if the region exceeds image bounds. *)

val center_crop :
  height:int -> width:int -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [center_crop ~height ~width img] crops a centered rectangle.

    Raises [Invalid_argument] if [height] or [width] exceeds the image
    dimensions. *)

val hflip : ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [hflip img] flips horizontally (left to right). *)

val vflip : ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [vflip img] flips vertically (top to bottom). *)

val rotate90 : ?k:int -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [rotate90 img] rotates by [k * 90] degrees counter-clockwise. [k] defaults
    to [1]. Negative values rotate clockwise. *)

val pad :
  ?value:float ->
  int * int * int * int ->
  ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t
(** [pad (top, bottom, left, right) img] zero-pads the spatial dimensions.
    [value] defaults to [0.0]. *)

(** {1:filter Spatial filtering} *)

val gaussian_blur :
  sigma:float -> ?ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [gaussian_blur ~sigma img] applies isotropic Gaussian blur using separable
    convolution. [ksize] defaults to [2 * ceil(3 * sigma) + 1], capturing 99.7%
    of the distribution.

    If [sigma] is [0] or less, the kernel follows from the given [ksize]. For
    [ksize] [1] to [9] it is fixed, each row below over its sum:
    {v
    ksize 1: 1
    ksize 3: 1 2 1
    ksize 5: 1 4 6 4 1
    ksize 7: 2 7 14 18 14 7 2
    ksize 9: 4 13 30 51 60 51 30 13 4
    v}
    For a larger [ksize] it is the Gaussian of sigma
    [0.3 * ((ksize - 1) / 2 - 1) + 0.8].

    Raises [Invalid_argument] if [ksize] is even or not positive, if [sigma]
    is NaN, or if [ksize] is not given and [sigma] is not positive or so large
    that [2 * ceil(3 * sigma) + 1] exceeds [max_int]. *)

val box_blur : ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [box_blur ~ksize img] applies a [ksize * ksize] averaging filter. Each
    pixel's window spans [ksize / 2] pixels before it and [(ksize - 1) / 2]
    after it along each axis, pixels outside the image counting as zero.

    Raises [Invalid_argument] if [ksize] is not positive. *)

val median_blur : ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [median_blur ~ksize img] applies a median filter.

    {b Note.} Not differentiable: uses sort internally, gradient is zero almost
    everywhere.

    Raises [Invalid_argument] if [ksize] is not a positive odd integer. *)

val filter2d : 'd Nx.float32_t -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [filter2d kernel img] applies a custom 2D convolution [kernel] to [img].
    [kernel] has shape [[kH; kW]]. Applied independently to each channel, it
    keeps [img]'s size: each pixel's window spans [kH / 2] rows and [kW / 2]
    columns before it, pixels outside the image counting as zero. *)

val unsharp_mask :
  sigma:float -> ?amount:float -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [unsharp_mask ~sigma img] sharpens by subtracting a Gaussian blur:
    [img + amount * (img - gaussian_blur ~sigma img)]. [amount] defaults to
    [1.0].

    Raises [Invalid_argument] if [sigma] is not positive. *)

(** {1:morphology Morphological operations}

    A {e structuring element} is a boolean host [[kH; kW]] tensor whose [true]
    entries select each pixel's {e neighbourhood}: the entry at row [i] and
    column [j] selects the pixel [i - kH / 2] rows below and [j - kW / 2]
    columns right of it, so the entry at row [kH / 2] and column [kW / 2]
    selects the pixel itself. {!erode} takes the least value over each
    pixel's neighbourhood and {!dilate} the greatest, by the dtype's order
    ({!Nx.maximum}). Erosion counts a pixel outside the image as the dtype's
    greatest value and dilation as its least. On a boolean image, where
    [false] orders below [true], they are the binary erosion and dilation: a
    pixel erodes to [true] iff its whole neighbourhood is [true], and dilates
    to [true] iff some pixel of it is.

    An operation reads the element once per call, for all its erosions and
    dilations. The image must not be complex; the result has its dtype and
    shape. The operations raise [Invalid_argument] if [kernel]'s rank is not
    2, if it has no [true] entry, if the image's rank is neither 3 nor 4, or
    if its dtype is complex. *)

(** The type for structuring element shapes. *)
type kernel_shape =
  | Rect  (** Full rectangle. *)
  | Cross  (** Cross-shaped element. *)
  | Ellipse  (** Elliptical element. *)

val structuring_element : kernel_shape -> int * int -> Nx.host Nx.bool_t
(** [structuring_element shape (h, w)] is a host structuring element of the
    given [shape] and size. It is [true] at its centre and symmetric about
    it. [h] and [w] must be positive odd integers.

    Raises [Invalid_argument] if [h] or [w] is not positive or not odd. *)

val erode :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [erode ~kernel img] is [img] with each pixel replaced by the least value
    over its neighbourhood. *)

val dilate :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [dilate ~kernel img] is [img] with each pixel replaced by the greatest
    value over its neighbourhood. *)

val opening :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [opening ~kernel img] is the erosion of [img] followed by the dilation by
    the element reflected about its centre, whose neighbourhood holds the
    pixel [dy] rows above and [dx] columns left where [kernel]'s holds the
    pixel [dy] rows below and [dx] columns right. It is at most [img] and
    idempotent, and removes bright regions [kernel] does not fit in. For an
    element symmetric about its centre it is
    [dilate ~kernel (erode ~kernel img)]. *)

val closing :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [closing ~kernel img] is the dilation of [img] followed by the erosion by
    the reflected element, as {!opening}. It is at least [img] and idempotent,
    and fills dark regions [kernel] does not fit in. *)

val morphological_gradient :
  kernel:Nx.host Nx.bool_t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [morphological_gradient ~kernel img] is
    [dilate ~kernel img - erode ~kernel img], as {!Nx.sub} computes it, and on
    a boolean image [true] where the dilation is [true] and the erosion
    [false]. Highlights edges. Where a pixel's neighbourhood lies wholly
    outside the image, which needs a [kernel] [false] at its centre, it is
    the dtype's least value minus its greatest: it wraps on integers, is
    [neg_infinity] on floats with infinities and [false] on booleans. *)

(** {1:edge Edge detection} *)

val sobel : ?ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t * 'd Nx.float32_t
(** [sobel img] computes Sobel gradients. Returns [(gx, gy)] where [gx] is the
    horizontal gradient and [gy] is the vertical gradient. Input must have
    [C = 1].

    [ksize] is the aperture, [1], [3], [5] or [7], and defaults to [3]. For
    [ksize >= 3], [gx] correlates with the [ksize * ksize] kernel whose rows
    are the binomial difference of [ksize] taps (for [3], [[-1 0 1]]), each
    weighted by the binomial smoothing of [ksize] taps (for [3],
    [[1 2 1]]); [gy] uses its transpose. For [ksize = 3]:
    {v
    gx: -1 0 1      gy: -1 -2 -1
        -2 0 2           0  0  0
        -1 0 1           1  2  1
    v}
    [ksize = 1] does not smooth: [gx] correlates with the 1x3 kernel
    [[-1 0 1]] and [gy] with its 3x1 transpose. A kernel of [ksize >= 3]
    scales a first derivative by [2 ^ (2 * ksize - 3)], [ksize = 1] by [2].

    Raises [Invalid_argument] if [ksize] is not [1], [3], [5] or [7]. *)

val scharr : 'd Nx.float32_t -> 'd Nx.float32_t * 'd Nx.float32_t
(** [scharr img] computes Scharr gradients, which are more rotationally accurate
    than Sobel. Returns [(gx, gy)]. Input must have [C = 1]. *)

val laplacian : ?ksize:int -> 'd Nx.float32_t -> 'd Nx.float32_t
(** [laplacian img] computes the Laplacian (sum of second spatial derivatives):
    the sum of the second x and y derivatives, each from {!sobel}'s kernel of
    aperture [ksize] with the binomial difference taken twice (for [ksize] [1]
    and [3], [[1 -2 1]] in place of [[-1 0 1]]). Input must have [C = 1].

    [ksize] is a positive odd integer and defaults to [1]. [ksize = 1]
    correlates with the 3x3 aperture on the left, [ksize = 3] with the one on
    the right:
    {v
    0  1  0      2  0  2
    1 -4  1      0 -8  0
    0  1  0      2  0  2
    v}
    A kernel of [ksize >= 3] scales the Laplacian by [2 ^ (2 * ksize - 4)],
    [ksize = 1] by [1].

    Raises [Invalid_argument] if [ksize] is not a positive odd integer. *)

val canny :
  low:float -> high:float -> ?sigma:float -> 'd Nx.float32_t -> 'd Nx.bool_t
(** [canny ~low ~high img] applies the Canny edge detector. Returns [true] for
    edge pixels, [false] otherwise. After non-maximum suppression a pixel is
    strong where its gradient magnitude is greater than [high], and a
    candidate where it is strong or at least [low], [low] and [high] stored
    in float32 as {!Nx.greater_s} stores them. The edge pixels are the
    candidates within two pixels, along each axis, of a strong pixel.
    [sigma] controls the initial Gaussian blur and defaults to [1.4]. Input
    must have [C = 1].

    {b Note.} Not differentiable: uses non-maximum suppression and hysteresis
    thresholding.

    Raises [Invalid_argument] if [sigma] is not positive. *)
