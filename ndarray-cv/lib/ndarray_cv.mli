(** ndarray-cv: Computer vision operations on [Ndarray].

    This module provides image manipulation, color conversion, datatype
    conversion, resizing, filtering, morphological operations, thresholding, and
    edge detection using [Ndarray] buffers. *)

type uint8_t = Ndarray.uint8_t
(** [uint8_t]

    3-D ([H; W; C]) or 4-D ([N; H; W; C]) unsigned 8-bit image tensor type. *)

type float32_t = Ndarray.float32_t
(** [float32_t]

    3-D or 4-D single-precision float image tensor type, with values typically
    normalized to [0.0, 1.0]. *)

type int16_t = Ndarray.int16_t
(** [int16_t]

    3-D or 4-D signed 16-bit integer tensor type, commonly used for derivative
    filters (e.g., Sobel). *)

(** {1 Basic Image Manipulations} *)

val flip_vertical : uint8_t -> uint8_t
(** [flip_vertical img]

    Flip the image vertically (top to bottom).

    {2 Parameters}
    - [img]: input image tensor ([H; W; C] or [N; H; W; C]).

    {2 Returns}
    - tensor with rows reversed along the vertical axis.

    {2 Notes}
    - Preserves datatype and channel count.

    {2 Examples}
    {[
      let flipped = flip_vertical img in
      (* flipped.{0;0;*} = img.{0;H-1;*} *)
    ]} *)

val flip_horizontal : uint8_t -> uint8_t
(** [flip_horizontal img]

    Flip the image horizontally (left to right).

    {2 Parameters}
    - [img]: input image tensor ([H; W; C] or [N; H; W; C]).

    {2 Returns}
    - tensor with columns reversed along the horizontal axis.

    {2 Notes}
    - Preserves datatype and channel count.

    {2 Examples}
    {[
      let flipped = flip_horizontal img in
      (* flipped.{0;*;0} = img.{0;*;W-1} *)
    ]} *)

val crop : y:int -> x:int -> height:int -> width:int -> uint8_t -> uint8_t
(** [crop ~y ~x ~height ~width img]

    Extract a rectangular region of interest from the image.

    {2 Parameters}
    - [y]: starting row index (0-based).
    - [x]: starting column index (0-based).
    - [height]: number of rows in the crop.
    - [width]: number of columns in the crop.
    - [img]: input image tensor of rank 3 ([H; W; C]).

    {2 Returns}
    - tensor of shape [|height; width; C|] containing the cropped region.

    {2 Raises}
    - [Invalid_argument] if the specified region exceeds image bounds.

    {2 Examples}
    {[
      let roi = crop ~y:10 ~x:20 ~height:50 ~width:50 img in
      (* roi has shape [50;50;C] *)
    ]} *)

(** {1 Color Space Conversion} *)

val to_grayscale : uint8_t -> uint8_t
(** [to_grayscale img]

    Convert a color image to grayscale using standard luminance weights.

    {2 Parameters}
    - [img]: input uint8 tensor ([H; W; C] or [N; H; W; C]) with C>=1.

    {2 Returns}
    - grayscale tensor ([H; W; 1] or [N; H; W; 1]).

    {2 Notes}
    - If input has multiple channels, uses weights [0.299, 0.587, 0.114].
    - If input has C=1, returns img unchanged.

    {2 Examples}
    {[
      let gray = to_grayscale img in
      (* gray.{0;i;j;0} = 0.299*R + 0.587*G + 0.114*B *)
    ]} *)

val rgb_to_bgr : uint8_t -> uint8_t
(** [rgb_to_bgr img]

    Swap red and blue channels in an RGB image to produce BGR.

    {2 Parameters}
    - [img]: input uint8 tensor with C=3.

    {2 Returns}
    - tensor with channels reordered to BGR.

    {2 Examples}
    {[
      let bgr = rgb_to_bgr img in
      (* bgr.{i;j;0} = img.{i;j;2} *)
    ]} *)

val bgr_to_rgb : uint8_t -> uint8_t
(** [bgr_to_rgb img]

    Swap blue and red channels in a BGR image to produce RGB.

    {2 Parameters}
    - [img]: input uint8 tensor with C=3.

    {2 Returns}
    - tensor with channels reordered to RGB.

    {2 Examples}
    {[
      let rgb = bgr_to_rgb img in
      (* rgb.{i;j;2} = img.{i;j;0} *)
    ]} *)

(** {1 Data Type Conversion} *)

val to_float : uint8_t -> float32_t
(** [to_float img]

    Convert uint8 image values [0,255] to float32 [0.0,1.0].

    {2 Parameters}
    - [img]: input uint8 tensor.

    {2 Returns}
    - float32 tensor of same shape with values scaled by 1.0 /. 255.0.

    {2 Examples}
    {[
      let f = to_float img in
      (* f.{i;j;k} = float img.{i;j;k} /. 255.0 *)
    ]} *)

val to_uint8 : float32_t -> uint8_t
(** [to_uint8 img]

    Convert float32 image values [0.0,1.0] to uint8 [0,255], with clipping.

    {2 Parameters}
    - [img]: input float32 tensor.

    {2 Returns}
    - uint8 tensor of same shape with values scaled by 255 and clipped to
      [0,255].

    {2 Examples}
    {[
      let u = to_uint8 f in
      (* u.{i;j;k} = clamp(int_of_float (f.{i;j;k} *. 255.), 0, 255) *)
    ]} *)

(** {1 Image Resizing} *)

(** Interpolation methods for image resizing. *)
type interpolation =
  | Nearest  (** nearest-neighbor interpolation (fast, may alias). *)
  | Linear  (** bilinear interpolation (default). *)

val resize :
  ?interpolation:interpolation -> height:int -> width:int -> uint8_t -> uint8_t
(** [resize ?interpolation ~height ~width img]

    Resize the image to the specified dimensions.

    {2 Parameters}
    - [?interpolation]: method ([Nearest] default or [Linear]).
    - [height]: target number of rows.
    - [width]: target number of columns.
    - [img]: input uint8 tensor of rank 3 ([H; W; C]) or rank 4 ([N; H; W; C]).

    {2 Returns}
    - resized uint8 tensor with shape [|...; height; width; C|].

    {2 Examples}
    {[
      let small = resize ~height:100 ~width:200 img in
      (* small has shape [100;200;C] *)
    ]} *)

(** {1 Image Filtering} *)

val gaussian_blur :
  ksize:int * int ->
  sigmaX:float ->
  ?sigmaY:float ->
  ('a, 'b) Ndarray.t ->
  ('a, 'b) Ndarray.t
(** [gaussian_blur ~ksize ~sigmaX ?sigmaY img]

    Apply a Gaussian blur to the image.

    {2 Parameters}
    - [ksize]: (height, width) of the Gaussian kernel; must be positive odd
      integers.
    - [sigmaX]: standard deviation in the X direction.
    - [?sigmaY]: standard deviation in the Y direction; defaults to [sigmaX].
    - [img]: input tensor of type uint8 or float32.

    {2 Returns}
    - tensor of same type and shape as [img], blurred by the Gaussian kernel.

    {2 Raises}
    - [Invalid_argument] if any [ksize] component is even or non-positive.

    {2 Examples}
    {[
      let blurred = gaussian_blur ~ksize:(5,5) ~sigmaX:1.0 img in
      ...
    ]} *)

val box_filter : ksize:int * int -> ('a, 'b) Ndarray.t -> ('a, 'b) Ndarray.t
(** [box_filter ~ksize img]

    Apply a normalized box (average) filter to the image.

    {2 Parameters}
    - [ksize]: (height, width) of the filter kernel; must be positive integers.
    - [img]: input tensor of type uint8 or float32.

    {2 Returns}
    - tensor of same type and shape as [img], averaged over each kernel window.

    {2 Examples}
    {[
      let avg = box_filter ~ksize:(3,3) img in
      ...
    ]} *)

val blur : ksize:int * int -> ('a, 'b) Ndarray.t -> ('a, 'b) Ndarray.t
(** [blur ~ksize img]

    Alias for [box_filter ~ksize img], applying an average filter.

    {2 Parameters}
    - [ksize]: (height, width) of the filter kernel.
    - [img]: input tensor.

    {2 Returns}
    - tensor of same type and shape as [img], blurred by box filter.

    {2 Examples}
    {[
      let b = blur ~ksize:(5,5) img in
      ...
    ]} *)

val median_blur : ksize:int -> uint8_t -> uint8_t
(** [median_blur ~ksize img]

    Apply a median filter to a grayscale uint8 image to remove noise.

    {2 Parameters}
    - [ksize]: size of the square kernel (positive odd integer).
    - [img]: input uint8 tensor of rank 3 or 4 with single channel.

    {2 Returns}
    - uint8 tensor of same shape with median-filtered values.

    {2 Raises}
    - [Invalid_argument] if [ksize] is not a positive odd integer.

    {2 Examples}
    {[
      let clean = median_blur ~ksize:3 gray_img in
      ...
    ]} *)

(** {1 Morphological Operations} *)

(** Shapes for structuring elements in morphological operations. *)
type structuring_element_shape =
  | Rect  (** full rectangle. *)
  | Cross  (** cross-shaped element. *)

val get_structuring_element :
  shape:structuring_element_shape -> ksize:int * int -> uint8_t
(** [get_structuring_element ~shape ~ksize]

    Create a structuring element for morphological operations.

    {2 Parameters}
    - [shape]: element shape ([Rect] or [Cross]).
    - [ksize]: (height, width) of the kernel; positive odd integers.

    {2 Returns}
    - uint8 tensor of shape [|height; width|] with ones at active elements.

    {2 Raises}
    - [Invalid_argument] if [ksize] components are invalid.

    {2 Examples}
    {[
      let k = get_structuring_element ~shape:Rect ~ksize:(3,3) in
      ...
    ]} *)

val erode : kernel:uint8_t -> uint8_t -> uint8_t
(** [erode ~kernel img]

    Perform morphological erosion on a grayscale uint8 image.

    {2 Parameters}
    - [kernel]: 2D uint8 structuring element.
    - [img]: input uint8 tensor with C=1.

    {2 Returns}
    - eroded tensor where each pixel is the minimum over the kernel window.

    {2 Examples}
    {[
      let e = erode ~kernel img in
      ...
    ]} *)

val dilate : kernel:uint8_t -> uint8_t -> uint8_t
(** [dilate ~kernel img]

    Perform morphological dilation on a grayscale uint8 image.

    {2 Parameters}
    - [kernel]: 2D uint8 structuring element.
    - [img]: input uint8 tensor with C=1.

    {2 Returns}
    - dilated tensor where each pixel is the maximum over the kernel window.

    {2 Examples}
    {[
      let d = dilate ~kernel img in
      ...
    ]} *)

(** {1 Image Thresholding} *)

(** Types of fixed-level thresholding operations. *)
type threshold_type =
  | Binary  (** dst = maxval if src > thresh else 0. *)
  | BinaryInv  (** dst = 0 if src > thresh else maxval. *)
  | Trunc  (** dst = min(src, thresh). *)
  | ToZero  (** dst = src if src > thresh else 0. *)
  | ToZeroInv  (** dst = 0 if src > thresh else src. *)

val threshold :
  thresh:int -> maxval:int -> type_:threshold_type -> uint8_t -> uint8_t
(** [threshold ~thresh ~maxval ~type_ img]

    Apply fixed-level thresholding to a grayscale uint8 image.

    {2 Parameters}
    - [thresh]: threshold value.
    - [maxval]: value used for [Binary]/[BinaryInv] operations.
    - [type_]: thresholding type ([threshold_type]).
    - [img]: input uint8 tensor of shape [[|H;W;1|]] or [[|N;H;W;1|]].

    {2 Returns}
    - uint8 tensor after thresholding (binary or truncated values).

    {2 Raises}
    - [Invalid_argument] if [thresh] or [maxval] are out of range.

    {2 Examples}
    {[
      let bw = threshold ~thresh:128 ~maxval:255 ~type_:Binary gray in
      (* values >128 become 255, else 0 *)
    ]} *)

(** {1 Edge Detection} *)

val sobel : dx:int -> dy:int -> ?ksize:int -> uint8_t -> int16_t
(** [sobel ~dx ~dy ?ksize img]

    Compute the Sobel derivative of a grayscale uint8 image.

    {2 Parameters}
    - [dx]: derivative order in x direction (0 or 1).
    - [dy]: derivative order in y direction (0 or 1).
    - [?ksize]: aperture size for Sobel operator (default 3; only 3 supported).
    - [img]: input uint8 tensor of shape [[|H;W;1|]] or [[|N;H;W;1|]].

    {2 Returns}
    - int16 tensor of same shape, containing derivative values.

    {2 Raises}
    - [Invalid_argument] if unsupported [ksize] is provided.

    {2 Examples}
    {[
      let gx = sobel ~dx:1 ~dy:0 img in
      (* x-gradient of image *)
    ]} *)

val canny :
  threshold1:float -> threshold2:float -> ?ksize:int -> uint8_t -> uint8_t
(** [canny ~threshold1 ~threshold2 ?ksize img]

    Apply the Canny edge detector to a grayscale uint8 image.

    {2 Parameters}
    - [threshold1]: first threshold for hysteresis procedure.
    - [threshold2]: second threshold for hysteresis procedure.
    - [?ksize]: Sobel aperture size for gradient computation (default 3).
    - [img]: input uint8 tensor of shape [[|H;W;1|]] or [[|N;H;W;1|]].

    {2 Returns}
    - uint8 tensor binary edge map (0 for non-edges, 255 for edges).

    {2 Raises}
    - [Invalid_argument] if threshold values are invalid or out of range.

    {2 Examples}
    {[
      let edges = canny ~threshold1:50. ~threshold2:150. img in
      (* binary edges *)
    ]} *)
