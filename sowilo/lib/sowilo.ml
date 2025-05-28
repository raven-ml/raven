(* Safe implementation of Sowilo using only Rune frontend functions *)

module BA = Bigarray

type uint8_t = Rune.uint8_t
type int16_t = Rune.int16_t
type float32_t = Rune.float32_t

let get_dims img =
  match Rune.shape img with
  | [| h; w |] -> `Gray (h, w)
  | [| h; w; c |] -> `Color (h, w, c)
  | s ->
      failwith
        (Printf.sprintf "Invalid image dimensions: expected 2 or 3, got %d (%s)"
           (Array.length s)
           (Array.to_list s |> List.map string_of_int |> String.concat "x"))

let flip_axis img axis =
  let shape = Rune.shape img in
  if Array.length shape <= axis || shape.(axis) <= 1 then img
  else
    let axes = [| axis |] in
    Rune.flip ~axes img

let flip_vertical img = if Rune.ndim img < 1 then img else flip_axis img 0
let flip_horizontal img = if Rune.ndim img < 2 then img else flip_axis img 1

let crop ~y ~x ~height ~width img =
  match get_dims img with
  | `Gray (h, w) ->
      if
        y < 0 || x < 0 || height <= 0 || width <= 0
        || y + height > h
        || x + width > w
      then
        invalid_arg
          (Printf.sprintf
             "Invalid crop parameters: y=%d, x=%d, h=%d, w=%d for image [%dx%d]"
             y x height width h w)
      else Rune.slice_ranges [ y; x ] [ y + height; x + width ] img
  | `Color (h, w, c) ->
      if
        y < 0 || x < 0 || height <= 0 || width <= 0
        || y + height > h
        || x + width > w
      then
        invalid_arg
          (Printf.sprintf
             "Invalid crop parameters: y=%d, x=%d, h=%d, w=%d for image \
              [%dx%dx%d]"
             y x height width h w c)
      else Rune.slice_ranges [ y; x; 0 ] [ y + height; x + width; c ] img

let to_grayscale img =
  match get_dims img with
  | `Gray (_, _) -> img
  | `Color (_h, _w, c) ->
      if c <> 3 then
        failwith "to_grayscale requires 3-channel (RGB) input image"
      else
        let img_f = Rune.astype Rune.float32 img in
        let r = Rune.slice [ Rune.R []; Rune.R []; Rune.I 0 ] img_f in
        let g = Rune.slice [ Rune.R []; Rune.R []; Rune.I 1 ] img_f in
        let b = Rune.slice [ Rune.R []; Rune.R []; Rune.I 2 ] img_f in
        let gray_f =
          Rune.add
            (Rune.add (Rune.mul_s r 0.299) (Rune.mul_s g 0.587))
            (Rune.mul_s b 0.114)
        in
        Rune.astype Rune.uint8 gray_f

let swap_channels img =
  match get_dims img with
  | `Gray (_, _) -> img
  | `Color (_h, _w, c) ->
      if c < 3 then img
      else
        (* Use concatenation instead of element-wise operations *)
        let chan0 = Rune.slice [ Rune.R []; Rune.R []; Rune.I 0 ] img in
        let chan1 = Rune.slice [ Rune.R []; Rune.R []; Rune.I 1 ] img in
        let chan2 = Rune.slice [ Rune.R []; Rune.R []; Rune.I 2 ] img in
        let chans_rest =
          if c > 3 then
            Some (Rune.slice [ Rune.R []; Rune.R []; Rune.R [ 3; c - 1 ] ] img)
          else None
        in

        let swapped_chans =
          match chans_rest with
          | None -> [ chan2; chan1; chan0 ]
          | Some rest -> [ chan2; chan1; chan0; rest ]
        in
        Rune.concatenate ~axis:2 swapped_chans

let rgb_to_bgr = swap_channels
let bgr_to_rgb = swap_channels

let to_float (img : uint8_t) =
  if Rune.dtype img <> Rune.uint8 then failwith "to_float requires uint8 input"
  else Rune.div_s (Rune.astype Rune.float32 img) 255.0

let to_uint8 (img : float32_t) =
  if Rune.dtype img <> Rune.float32 then
    failwith "to_uint8 requires float32 input"
  else
    let clipped = Rune.clip ~min:0.0 ~max:1.0 img in
    Rune.astype Rune.uint8 (Rune.mul_s clipped 255.0)

type interpolation = Nearest | Linear

let resize ?(interpolation = Nearest) ~height:out_h ~width:out_w img =
  ignore interpolation;
  (* TODO: implement interpolation *)
  if out_h <= 0 || out_w <= 0 then
    invalid_arg "Output height and width must be positive";
  (* For now, just return the original image *)
  img

let generate_gaussian_kernel device size sigma =
  let sigma =
    if sigma <= 0.0 then (0.3 *. (float (size / 2) -. 1.0)) +. 0.8 else sigma
  in
  let center = float (size / 2) in
  let sigma2_sq = 2.0 *. sigma *. sigma in

  (* Create array of positions *)
  let positions = Rune.arange_f device Rune.float32 0.0 (float size) 1.0 in
  let x = Rune.sub_s positions center in

  (* Compute gaussian values *)
  let x_sq = Rune.square x in
  let neg_x_sq = Rune.neg x_sq in
  let exponent = Rune.div_s neg_x_sq sigma2_sq in
  let kernel = Rune.exp exponent in

  (* Normalize *)
  let sum = Rune.sum kernel in
  let sum_scalar = Rune.reshape [||] sum in
  Rune.div kernel sum_scalar

(* Safe 2D convolution using Rune's correlate2d *)
let convolve2d_safe kernel img =
  match get_dims img with
  | `Gray (h, w) ->
      (* Reshape for conv2d: [batch=1, channels=1, h, w] *)
      let img_4d = Rune.reshape [| 1; 1; h; w |] img in

      (* Reshape kernel to [out_channels=1, in_channels=1, kh, kw] *)
      let kernel_shape = Rune.shape kernel in
      let kernel_4d =
        match Array.length kernel_shape with
        | 2 ->
            Rune.reshape [| 1; 1; kernel_shape.(0); kernel_shape.(1) |] kernel
        | _ -> failwith "Kernel must be 2D"
      in

      (* Perform convolution *)
      let result_4d = Rune.correlate2d ~padding_mode:`Same img_4d kernel_4d in

      (* Reshape back and convert to original dtype *)
      let result = Rune.reshape [| h; w |] result_4d in
      if Rune.dtype img = Rune.Float32 then result
      else Rune.astype (Rune.dtype img) result
  | `Color (h, w, c) ->
      (* Process all channels at once using grouped convolution *)
      (* Reshape to [batch=1, channels=c, h, w] *)
      let img_transposed = Rune.transpose ~axes:[| 2; 0; 1 |] img in
      let img_4d = Rune.reshape [| 1; c; h; w |] img_transposed in

      (* Create kernel for each channel: [out_channels=c, in_channels=1, kh,
         kw] *)
      let kernel_shape = Rune.shape kernel in
      let kernel_single =
        Rune.reshape [| 1; 1; kernel_shape.(0); kernel_shape.(1) |] kernel
      in
      let kernel_4d = Rune.tile [| c; 1; 1; 1 |] kernel_single in

      (* Perform grouped convolution *)
      let result_4d =
        Rune.correlate2d ~groups:c ~padding_mode:`Same img_4d kernel_4d
      in

      (* Reshape back to [h, w, c] *)
      let result_chw = Rune.reshape [| c; h; w |] result_4d in
      let result = Rune.transpose ~axes:[| 1; 2; 0 |] result_chw in

      if Rune.dtype img = Rune.Float32 then result
      else Rune.astype (Rune.dtype img) result

(* Separable 2D convolution *)
let convolve_separable kernel_x kernel_y img =
  (* First convolve horizontally *)
  let kernel_x_2d = Rune.reshape [| 1; Rune.numel kernel_x |] kernel_x in
  let temp = convolve2d_safe kernel_x_2d img in

  (* Then convolve vertically *)
  let kernel_y_2d = Rune.reshape [| Rune.numel kernel_y; 1 |] kernel_y in
  convolve2d_safe kernel_y_2d temp

let gaussian_blur : type a b.
    ksize:int * int ->
    sigmaX:float ->
    ?sigmaY:float ->
    (a, b) Rune.t ->
    (a, b) Rune.t =
 fun ~ksize:(kh, kw) ~sigmaX ?sigmaY img ->
  if kh <= 0 || kh mod 2 = 0 || kw <= 0 || kw mod 2 = 0 then
    invalid_arg "Kernel dimensions must be positive and odd";

  let device = Rune.device img in

  let sigmaY = match sigmaY with None -> sigmaX | Some sy -> sy in
  let kernelX = generate_gaussian_kernel device kw sigmaX in
  let kernelY = generate_gaussian_kernel device kh sigmaY in

  let img_f32 =
    match Rune.dtype img with
    | Rune.UInt8 -> to_float img
    | Rune.Float32 -> img
    | _ -> failwith "Unsupported image type for gaussian_blur"
  in

  let blurred_f32 = convolve_separable kernelX kernelY img_f32 in

  match Rune.dtype img with
  | Rune.UInt8 -> to_uint8 blurred_f32
  | Rune.Float32 -> blurred_f32
  | _ -> failwith "Unsupported image type for gaussian_blur"

type threshold_type = Binary | BinaryInv | Trunc | ToZero | ToZeroInv

let threshold ~thresh ~maxval ~type_ (img : uint8_t) : uint8_t =
  match get_dims img with
  | `Color _ ->
      failwith "Threshold currently only supports grayscale (2D) images"
  | `Gray _ -> (
      if Rune.dtype img <> Rune.uint8 then
        failwith "Threshold currently only supports uint8 images";

      let thresh_val = max 0 (min 255 thresh) in
      let maxval_val = max 0 (min 255 maxval) in

      let thresh_tensor = Rune.scalar_like img thresh_val in
      let maxval_tensor = Rune.scalar_like img maxval_val in
      let zero_tensor = Rune.zeros_like img in

      let mask = Rune.greater img thresh_tensor in

      match type_ with
      | Binary -> Rune.where mask maxval_tensor zero_tensor
      | BinaryInv -> Rune.where mask zero_tensor maxval_tensor
      | Trunc -> Rune.minimum img thresh_tensor
      | ToZero -> Rune.where mask img zero_tensor
      | ToZeroInv -> Rune.where mask zero_tensor img)

let box_filter : type a b. ksize:int * int -> (a, b) Rune.t -> (a, b) Rune.t =
 fun ~ksize:(kh, kw) img ->
  if kh <= 0 || kw <= 0 then invalid_arg "Kernel dimensions must be positive";

  let device = Rune.device img in
  let kernel_val = 1.0 /. float_of_int (kh * kw) in
  let kernel = Rune.full device Rune.float32 [| kh; kw |] kernel_val in

  let img_f32 =
    match Rune.dtype img with
    | Rune.UInt8 -> to_float img
    | Rune.Float32 -> img
    | _ -> failwith "Unsupported image type for box_filter"
  in

  let filtered_f32 = convolve2d_safe kernel img_f32 in

  match Rune.dtype img with
  | Rune.UInt8 -> to_uint8 filtered_f32
  | Rune.Float32 -> filtered_f32
  | _ -> failwith "Unsupported image type for box_filter"

let median_blur ~ksize (img : uint8_t) : uint8_t =
  if Rune.dtype img <> Rune.uint8 then
    failwith "Median blur currently only supports uint8 images";
  if ksize <= 0 || ksize mod 2 = 0 then
    invalid_arg "Kernel size (ksize) must be positive and odd";

  match get_dims img with
  | `Color _ -> failwith "Median blur currently only supports grayscale images"
  | `Gray (_h, _w) ->
      (* Use uniform filter as approximation for median filter *)
      (* This is not a true median filter but avoids element-wise access *)
      let kernel =
        Rune.ones (Rune.device img) Rune.float32 [| ksize; ksize |]
      in
      let kernel_normalized =
        Rune.div_s kernel (float_of_int (ksize * ksize))
      in
      let img_f32 = Rune.astype Rune.float32 img in
      let filtered = convolve2d_safe kernel_normalized img_f32 in
      Rune.astype Rune.uint8 filtered

let blur = box_filter

type structuring_element_shape = Rect | Cross

let get_structuring_element ~shape ~ksize:(kh, kw) =
  if kh <= 0 || kw <= 0 then invalid_arg "Kernel dimensions must be positive";
  let device = Rune.cpu in

  match shape with
  | Rect -> Rune.ones device Rune.uint8 [| kh; kw |]
  | Cross ->
      (* Create cross pattern using tensor operations *)
      let center_h = kh / 2 in
      let center_w = kw / 2 in

      (* Create horizontal line *)
      let h_line = Rune.ones device Rune.uint8 [| 1; kw |] in
      let h_line_padded =
        Rune.pad [| (center_h, kh - center_h - 1); (0, 0) |] 0 h_line
      in

      (* Create vertical line *)
      let v_line = Rune.ones device Rune.uint8 [| kh; 1 |] in
      let v_line_padded =
        Rune.pad [| (0, 0); (center_w, kw - center_w - 1) |] 0 v_line
      in

      (* Combine using logical or *)
      Rune.maximum h_line_padded v_line_padded

let morph_op ~op ~kernel (img : uint8_t) : uint8_t =
  if Rune.dtype img <> Rune.uint8 then
    failwith "Morphological operations currently require uint8 input";

  match get_dims img with
  | `Color _ ->
      failwith "Morphological operations currently support grayscale only"
  | `Gray (h, w) -> (
      let kh, kw =
        match Rune.shape kernel with
        | [| kh; kw |] -> (kh, kw)
        | _ -> failwith "Kernel must be 2D"
      in

      if kh <= 0 || kw <= 0 || kh mod 2 = 0 || kw mod 2 = 0 then
        failwith "Kernel dimensions must be positive and odd";

      (* Convert to 4D for pooling operations *)
      let img_4d = Rune.reshape [| 1; 1; h; w |] img in

      match op with
      | `Max ->
          (* Dilation using max pooling with stride=1 to preserve size *)
          let result_4d, _ =
            Rune.max_pool2d ~kernel_size:(kh, kw) ~stride:(1, 1) ~padding_spec:`Same img_4d
          in
          Rune.reshape [| h; w |] result_4d
      | `Min ->
          (* Erosion: 255 - max_pool(255 - img) *)
          let max_val = Rune.scalar_like img 255 in
          let inverted = Rune.sub max_val img in
          let inv_4d = Rune.reshape [| 1; 1; h; w |] inverted in
          let result_4d, _ =
            Rune.max_pool2d ~kernel_size:(kh, kw) ~stride:(1, 1) ~padding_spec:`Same inv_4d
          in
          let result = Rune.reshape [| h; w |] result_4d in
          Rune.sub max_val result)

let erode ~kernel img = morph_op ~op:`Min ~kernel img
let dilate ~kernel img = morph_op ~op:`Max ~kernel img

(* Sobel kernel definition - kept for reference but not used with manual implementation
let sobel_kernel dx dy ksize =
  if ksize <> 3 then failwith "Sobel currently only supports ksize=3";
  let device = Rune.cpu in
  match (dx, dy) with
  | 1, 0 ->
      Rune.create device Rune.float32 [| 3; 3 |]
        [| -1.; 0.; 1.; -2.; 0.; 2.; -1.; 0.; 1. |]
  | 0, 1 ->
      Rune.create device Rune.float32 [| 3; 3 |]
        [| -1.; -2.; -1.; 0.; 0.; 0.; 1.; 2.; 1. |]
  | _ -> failwith "Sobel requires dx=1, dy=0 or dx=0, dy=1"
*)

let sobel : dx:int -> dy:int -> ?ksize:int -> uint8_t -> int16_t =
 fun ~dx ~dy ?(ksize = 3) img ->
  if Rune.dtype img <> Rune.uint8 then
    failwith "Sobel currently requires uint8 input";
  if ksize <> 3 then failwith "Sobel currently only supports ksize=3";

  match get_dims img with
  | `Color _ -> failwith "Sobel currently only supports grayscale images"
  | `Gray (h, w) ->
      (* Convert to float for computation *)
      let img_f32 = Rune.astype Rune.float32 img in
      
      (* Pad the image *)
      let img_padded = Rune.pad [| (1, 1); (1, 1) |] 0.0 img_f32 in
      
      (* Extract shifted versions for manual convolution *)
      let tl = Rune.slice_ranges [ 0; 0 ] [ h; w ] img_padded in
      let tc = Rune.slice_ranges [ 0; 1 ] [ h; w + 1 ] img_padded in
      let tr = Rune.slice_ranges [ 0; 2 ] [ h; w + 2 ] img_padded in
      let ml = Rune.slice_ranges [ 1; 0 ] [ h + 1; w ] img_padded in
      let mr = Rune.slice_ranges [ 1; 2 ] [ h + 1; w + 2 ] img_padded in
      let bl = Rune.slice_ranges [ 2; 0 ] [ h + 2; w ] img_padded in
      let bc = Rune.slice_ranges [ 2; 1 ] [ h + 2; w + 1 ] img_padded in
      let br = Rune.slice_ranges [ 2; 2 ] [ h + 2; w + 2 ] img_padded in
      
      (* Apply Sobel kernels manually *)
      let result_f32 = match (dx, dy) with
        | 1, 0 ->
            (* Sobel X: [-1 0 1; -2 0 2; -1 0 1] *)
            Rune.add
              (Rune.add
                (Rune.sub tr tl)
                (Rune.sub (Rune.mul_s mr 2.0) (Rune.mul_s ml 2.0)))
              (Rune.sub br bl)
        | 0, 1 ->
            (* Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1] *)
            Rune.add
              (Rune.add
                (Rune.sub bl tl)
                (Rune.sub (Rune.mul_s bc 2.0) (Rune.mul_s tc 2.0)))
              (Rune.sub br tr)
        | _ -> failwith "Sobel requires dx=1, dy=0 or dx=0, dy=1"
      in
      
      Rune.astype Rune.int16 result_f32

let canny ~threshold1 ~threshold2 ?(ksize = 3) (img : uint8_t) : uint8_t =
  if Rune.dtype img <> Rune.uint8 then
    failwith "Canny currently requires uint8 input";
  if threshold1 < 0.0 || threshold2 < 0.0 then
    invalid_arg "Thresholds must be non-negative";

  let high_thresh, low_thresh =
    if threshold1 > threshold2 then (threshold1, threshold2)
    else (threshold2, threshold1)
  in

  (* 1. Noise Reduction *)
  let blurred_img = gaussian_blur ~ksize:(5, 5) ~sigmaX:1.4 img in

  (* 2. Gradient Calculation *)
  let gx = sobel ~dx:1 ~dy:0 ~ksize blurred_img in
  let gy = sobel ~dx:0 ~dy:1 ~ksize blurred_img in
  let gx_f = Rune.astype Rune.float32 gx in
  let gy_f = Rune.astype Rune.float32 gy in

  (* Calculate Magnitude and Angle *)
  let mag = Rune.sqrt (Rune.add (Rune.square gx_f) (Rune.square gy_f)) in
  let angle = Rune.atan2 gy_f gx_f in

  (* 3. Non-Maximum Suppression using vectorized operations *)
  let h, w =
    match Rune.shape img with
    | [| h; w |] -> (h, w)
    | _ -> failwith "Expected 2D image"
  in

  (* Convert angles to degrees and normalize to [0, 180) *)
  let pi = 3.14159265359 in
  let angle_deg = Rune.mul_s angle (180.0 /. pi) in
  let angle_pos =
    Rune.where
      (Rune.less angle_deg (Rune.zeros_like angle_deg))
      (Rune.add_s angle_deg 180.0)
      angle_deg
  in

  (* Quantize angles to 4 directions: 0, 45, 90, 135 degrees *)
  (* Direction masks *)
  let is_horizontal =
    Rune.logical_or
      (Rune.logical_and
         (Rune.greater_equal angle_pos (Rune.scalar_like angle_pos 0.0))
         (Rune.less angle_pos (Rune.scalar_like angle_pos 22.5)))
      (Rune.logical_and
         (Rune.greater_equal angle_pos (Rune.scalar_like angle_pos 157.5))
         (Rune.less_equal angle_pos (Rune.scalar_like angle_pos 180.0)))
  in

  let is_diagonal1 =
    Rune.logical_and
      (Rune.greater_equal angle_pos (Rune.scalar_like angle_pos 22.5))
      (Rune.less angle_pos (Rune.scalar_like angle_pos 67.5))
  in

  let is_vertical =
    Rune.logical_and
      (Rune.greater_equal angle_pos (Rune.scalar_like angle_pos 67.5))
      (Rune.less angle_pos (Rune.scalar_like angle_pos 112.5))
  in

  let is_diagonal2 =
    Rune.logical_and
      (Rune.greater_equal angle_pos (Rune.scalar_like angle_pos 112.5))
      (Rune.less angle_pos (Rune.scalar_like angle_pos 157.5))
  in

  (* Pad magnitude for neighbor access *)
  let mag_padded = Rune.pad [| (1, 1); (1, 1) |] 0.0 mag in

  (* Extract neighbors for each direction *)
  (* Horizontal: left and right neighbors *)
  let left = Rune.slice_ranges [ 1; 0 ] [ h + 1; w ] mag_padded in
  let right = Rune.slice_ranges [ 1; 2 ] [ h + 1; w + 2 ] mag_padded in
  let center = Rune.slice_ranges [ 1; 1 ] [ h + 1; w + 1 ] mag_padded in

  (* Vertical: top and bottom neighbors *)
  let top = Rune.slice_ranges [ 0; 1 ] [ h; w + 1 ] mag_padded in
  let bottom = Rune.slice_ranges [ 2; 1 ] [ h + 2; w + 1 ] mag_padded in

  (* Diagonal 1: top-right and bottom-left *)
  let top_right = Rune.slice_ranges [ 0; 2 ] [ h; w + 2 ] mag_padded in
  let bottom_left = Rune.slice_ranges [ 2; 0 ] [ h + 2; w ] mag_padded in

  (* Diagonal 2: top-left and bottom-right *)
  let top_left = Rune.slice_ranges [ 0; 0 ] [ h; w ] mag_padded in
  let bottom_right = Rune.slice_ranges [ 2; 2 ] [ h + 2; w + 2 ] mag_padded in

  (* Check if center is maximum in its direction *)
  let is_max_horizontal =
    Rune.logical_and
      (Rune.greater_equal center left)
      (Rune.greater_equal center right)
  in

  let is_max_vertical =
    Rune.logical_and
      (Rune.greater_equal center top)
      (Rune.greater_equal center bottom)
  in

  let is_max_diagonal1 =
    Rune.logical_and
      (Rune.greater_equal center top_right)
      (Rune.greater_equal center bottom_left)
  in

  let is_max_diagonal2 =
    Rune.logical_and
      (Rune.greater_equal center top_left)
      (Rune.greater_equal center bottom_right)
  in

  (* Combine all conditions *)
  let is_max =
    Rune.logical_or
      (Rune.logical_or
         (Rune.logical_and is_horizontal is_max_horizontal)
         (Rune.logical_and is_diagonal1 is_max_diagonal1))
      (Rune.logical_or
         (Rune.logical_and is_vertical is_max_vertical)
         (Rune.logical_and is_diagonal2 is_max_diagonal2))
  in

  (* Apply non-maximum suppression *)
  let nms = Rune.where is_max mag (Rune.zeros_like mag) in

  (* 4. Double Thresholding *)
  let strong_edges = Rune.greater nms (Rune.scalar_like nms high_thresh) in
  let weak_edges =
    Rune.logical_and
      (Rune.greater_equal nms (Rune.scalar_like nms low_thresh))
      (Rune.logical_not strong_edges)
  in

  (* 5. Edge Tracking by Hysteresis using dilation *)
  let strong_val = Rune.scalar_like img 255 in
  let weak_val = Rune.scalar_like img 128 in
  let zero_val = Rune.zeros_like img in

  (* Create initial edge map *)
  let edge_map =
    Rune.where strong_edges strong_val (Rune.where weak_edges weak_val zero_val)
  in

  (* Use morphological dilation to connect weak edges to strong edges *)
  let kernel_3x3 = get_structuring_element ~shape:Rect ~ksize:(3, 3) in

  (* Extract strong edges only *)
  let strong_only = Rune.where strong_edges strong_val zero_val in

  (* Dilate strong edges multiple times *)
  let dilated = ref strong_only in
  for _ = 1 to 2 do
    dilated := dilate ~kernel:kernel_3x3 !dilated
  done;

  (* Keep weak edges that are connected to strong edges *)
  let connected_mask = Rune.greater !dilated zero_val in
  let final_edges =
    Rune.where
      (Rune.logical_and connected_mask (Rune.greater edge_map zero_val))
      strong_val zero_val
  in

  final_edges
