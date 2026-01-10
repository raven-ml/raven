(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type uint8_t = Rune.uint8_t
type int16_t = Rune.int16_t
type float32_t = Rune.float32_t

let float_range size = Rune.arange_f Rune.float32 0.0 (float size) 1.0

let compute_nearest_indices ~size_in ~size_out =
  if size_out <= 0 then
    invalid_arg "compute_nearest_indices: size_out must be positive";
  if size_in <= 0 then
    invalid_arg "compute_nearest_indices: size_in must be positive";
  let shape = [| size_out |] in
  if size_out = 1 || size_in = 1 then Rune.full Rune.int32 shape Int32.zero
  else
    let scale = float size_in /. float size_out in
    let coords = float_range size_out in
    let src = Rune.sub_s (Rune.mul_s (Rune.add_s coords 0.5) scale) 0.5 in
    let src_clipped = Rune.clip ~min:0.0 ~max:(float (size_in - 1)) src in
    Rune.astype Rune.int32 (Rune.round src_clipped)

let compute_linear_axis ~size_in ~size_out =
  if size_out <= 0 then
    invalid_arg "compute_linear_axis: size_out must be positive";
  if size_in <= 0 then
    invalid_arg "compute_linear_axis: size_in must be positive";
  let shape = [| size_out |] in
  if size_out = 1 || size_in = 1 then
    let zeros_i = Rune.full Rune.int32 shape Int32.zero in
    let zeros_f = Rune.full Rune.float32 shape 0.0 in
    (zeros_i, zeros_i, zeros_f)
  else
    let scale = float (size_in - 1) /. float (size_out - 1) in
    let src = Rune.mul_s (float_range size_out) scale in
    let idx0 = src |> Rune.floor |> Rune.astype Rune.int32 in
    let one = Rune.scalar_like idx0 Int32.(of_int 1) in
    let max_idx = Rune.scalar_like idx0 Int32.(of_int (size_in - 1)) in
    let idx1 = Rune.minimum (Rune.add idx0 one) max_idx in
    let idx0_f = Rune.astype Rune.float32 idx0 in
    let delta = Rune.sub src idx0_f in
    (idx0, idx1, delta)

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
    let axes = [ axis ] in
    Rune.flip ~axes img

let flip_vertical img =
  let shape = Rune.shape img in
  match Array.length shape with
  | 0 | 1 -> img
  | 2 -> flip_axis img 0
  | 3 ->
      let channels = shape.(2) in
      if channels = 1 || channels = 3 || channels = 4 then flip_axis img 0
      else flip_axis img 1
  | 4 -> flip_axis img 1
  | _ -> img

let flip_horizontal img =
  let shape = Rune.shape img in
  match Array.length shape with
  | 0 | 1 -> img
  | 2 -> flip_axis img 1
  | 3 ->
      let channels = shape.(2) in
      if channels = 1 || channels = 3 || channels = 4 then flip_axis img 1
      else flip_axis img 2
  | 4 -> flip_axis img 2
  | _ -> img

let crop ~y ~x ~height ~width img =
  let shape = Rune.shape img in
  let invalidate dims =
    invalid_arg
      (Printf.sprintf
         "Invalid crop parameters: y=%d, x=%d, h=%d, w=%d for image %s" y x
         height width dims)
  in
  match Array.length shape with
  | 2 ->
      let h = shape.(0) in
      let w = shape.(1) in
      if
        y < 0 || x < 0 || height <= 0 || width <= 0
        || y + height > h
        || x + width > w
      then invalidate (Printf.sprintf "[%dx%d]" h w)
      else Rune.slice [ Rune.R (y, y + height); Rune.R (x, x + width) ] img
  | 3 ->
      let d0 = shape.(0) in
      let d1 = shape.(1) in
      let d2 = shape.(2) in
      if d2 = 1 || d2 = 3 || d2 = 4 then
        let h = d0 and w = d1 and c = d2 in
        if
          y < 0 || x < 0 || height <= 0 || width <= 0
          || y + height > h
          || x + width > w
        then invalidate (Printf.sprintf "[%dx%dx%d]" h w c)
        else
          Rune.slice
            [ Rune.R (y, y + height); Rune.R (x, x + width); Rune.A ]
            img
      else
        let n = d0 and h = d1 and w = d2 in
        if
          y < 0 || x < 0 || height <= 0 || width <= 0
          || y + height > h
          || x + width > w
        then invalidate (Printf.sprintf "[%dx%dx%d]" n h w)
        else
          Rune.slice
            [ Rune.A; Rune.R (y, y + height); Rune.R (x, x + width) ]
            img
  | 4 ->
      let n = shape.(0) in
      let h = shape.(1) in
      let w = shape.(2) in
      let c = shape.(3) in
      if
        y < 0 || x < 0 || height <= 0 || width <= 0
        || y + height > h
        || x + width > w
      then invalidate (Printf.sprintf "[%dx%dx%dx%d]" n h w c)
      else
        Rune.slice
          [ Rune.A; Rune.R (y, y + height); Rune.R (x, x + width); Rune.A ]
          img
  | _ ->
      failwith "Unsupported image dimensions: expected 2D/3D/4D tensor for crop"

let to_grayscale img =
  let shape = Rune.shape img in
  match Array.length shape with
  | 0 | 1 | 2 -> img
  | 3 ->
      let channels = shape.(2) in
      if channels <> 3 then img
      else
        let weights =
          Rune.reshape [| 1; 1; 3 |]
            (Rune.create Rune.float32 [| 3 |] [| 0.299; 0.587; 0.114 |])
        in
        let img_f = Rune.astype Rune.float32 img in
        let gray_f =
          Rune.mul img_f weights |> fun weighted ->
          Rune.sum ~axes:[ 2 ] weighted
        in
        Rune.astype Rune.uint8 gray_f
  | 4 ->
      let channels = shape.(3) in
      if channels <> 3 then img
      else
        let weights =
          Rune.reshape [| 1; 1; 1; 3 |]
            (Rune.create Rune.float32 [| 3 |] [| 0.299; 0.587; 0.114 |])
        in
        let img_f = Rune.astype Rune.float32 img in
        let gray_f =
          Rune.mul img_f weights |> fun weighted ->
          Rune.sum ~axes:[ 3 ] weighted
        in
        Rune.astype Rune.uint8 gray_f
  | _ -> failwith "Unsupported image dimensions for to_grayscale"

let swap_channels img =
  let shape = Rune.shape img in
  match Array.length shape with
  | 0 | 1 | 2 -> img
  | 3 ->
      let channels = shape.(2) in
      if channels < 3 then img
      else
        let chan0 = Rune.slice [ Rune.A; Rune.A; Rune.R (0, 1) ] img in
        let chan1 = Rune.slice [ Rune.A; Rune.A; Rune.R (1, 2) ] img in
        let chan2 = Rune.slice [ Rune.A; Rune.A; Rune.R (2, 3) ] img in
        let rest =
          if channels > 3 then
            Some (Rune.slice [ Rune.A; Rune.A; Rune.R (3, channels) ] img)
          else None
        in
        let channels_swapped =
          match rest with
          | None -> [ chan2; chan1; chan0 ]
          | Some r -> [ chan2; chan1; chan0; r ]
        in
        Rune.concatenate ~axis:2 channels_swapped
  | 4 ->
      let channels = shape.(3) in
      if channels < 3 then img
      else
        let chan0 = Rune.slice [ Rune.A; Rune.A; Rune.A; Rune.R (0, 1) ] img in
        let chan1 = Rune.slice [ Rune.A; Rune.A; Rune.A; Rune.R (1, 2) ] img in
        let chan2 = Rune.slice [ Rune.A; Rune.A; Rune.A; Rune.R (2, 3) ] img in
        let rest =
          if channels > 3 then
            Some
              (Rune.slice [ Rune.A; Rune.A; Rune.A; Rune.R (3, channels) ] img)
          else None
        in
        let channels_swapped =
          match rest with
          | None -> [ chan2; chan1; chan0 ]
          | Some r -> [ chan2; chan1; chan0; r ]
        in
        Rune.concatenate ~axis:3 channels_swapped
  | _ -> failwith "Unsupported image dimensions for swap_channels"

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
  if out_h <= 0 || out_w <= 0 then
    invalid_arg "Output height and width must be positive";

  if Rune.dtype img <> Rune.uint8 then
    failwith "resize currently supports uint8 images";

  let resize_nhwc input =
    let shape = Rune.shape input in
    if Array.length shape <> 4 then failwith "Expected NHWC tensor";
    let in_h = shape.(1) in
    let in_w = shape.(2) in
    match interpolation with
    | Nearest ->
        let y_idx = compute_nearest_indices ~size_in:in_h ~size_out:out_h in
        let x_idx = compute_nearest_indices ~size_in:in_w ~size_out:out_w in
        input |> Rune.take ~axis:1 y_idx |> Rune.take ~axis:2 x_idx
    | Linear ->
        let input_f = Rune.astype Rune.float32 input in
        let y0, y1, dy = compute_linear_axis ~size_in:in_h ~size_out:out_h in
        let x0, x1, dx = compute_linear_axis ~size_in:in_w ~size_out:out_w in
        let top = Rune.take ~axis:1 y0 input_f in
        let bottom = Rune.take ~axis:1 y1 input_f in
        let top_left = Rune.take ~axis:2 x0 top in
        let top_right = Rune.take ~axis:2 x1 top in
        let bottom_left = Rune.take ~axis:2 x0 bottom in
        let bottom_right = Rune.take ~axis:2 x1 bottom in
        let dx_b = Rune.reshape [| 1; 1; out_w; 1 |] dx in
        let one_minus_dx =
          let ones = Rune.ones Rune.float32 (Rune.shape dx_b) in
          Rune.sub ones dx_b
        in
        let top_interp =
          Rune.add (Rune.mul one_minus_dx top_left) (Rune.mul dx_b top_right)
        in
        let bottom_interp =
          Rune.add
            (Rune.mul one_minus_dx bottom_left)
            (Rune.mul dx_b bottom_right)
        in
        let dy_b = Rune.reshape [| 1; out_h; 1; 1 |] dy in
        let one_minus_dy =
          let ones = Rune.ones Rune.float32 (Rune.shape dy_b) in
          Rune.sub ones dy_b
        in
        let blended =
          Rune.add
            (Rune.mul one_minus_dy top_interp)
            (Rune.mul dy_b bottom_interp)
        in
        let clipped = Rune.clip ~min:0.0 ~max:255.0 blended in
        let rounded = Rune.round clipped in
        Rune.astype Rune.uint8 rounded
  in

  let shape = Rune.shape img in
  match Array.length shape with
  | 2 ->
      let h = shape.(0) in
      let w = shape.(1) in
      let nhwc = Rune.reshape [| 1; h; w; 1 |] img in
      let resized = resize_nhwc nhwc in
      Rune.reshape [| out_h; out_w |] resized
  | 3 ->
      let d0 = shape.(0) in
      let d1 = shape.(1) in
      let d2 = shape.(2) in
      if d2 = 1 || d2 = 3 || d2 = 4 then
        let nhwc = Rune.reshape [| 1; d0; d1; d2 |] img in
        let resized = resize_nhwc nhwc in
        Rune.reshape [| out_h; out_w; d2 |] resized
      else
        let nhwc = Rune.reshape [| d0; d1; d2; 1 |] img in
        let resized = resize_nhwc nhwc in
        Rune.reshape [| d0; out_h; out_w |] resized
  | 4 ->
      let n = shape.(0) in
      let c = shape.(3) in
      let resized = resize_nhwc img in
      Rune.reshape [| n; out_h; out_w; c |] resized
  | _ -> failwith "Unsupported image dimensions for resize"

let generate_gaussian_kernel size sigma =
  let sigma =
    if sigma <= 0.0 then (0.3 *. (float (size / 2) -. 1.0)) +. 0.8 else sigma
  in
  let center = float (size / 2) in
  let sigma2_sq = 2.0 *. sigma *. sigma in

  (* Create array of positions *)
  let positions = Rune.arange_f Rune.float32 0.0 (float size) 1.0 in
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
      let img_transposed = Rune.transpose ~axes:[ 2; 0; 1 ] img in
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
      let result = Rune.transpose ~axes:[ 1; 2; 0 ] result_chw in

      if Rune.dtype img = Rune.Float32 then result
      else Rune.astype (Rune.dtype img) result

let gaussian_blur : type a b.
    ksize:int * int ->
    sigmaX:float ->
    ?sigmaY:float ->
    (a, b) Rune.t ->
    (a, b) Rune.t =
 fun ~ksize:(kh, kw) ~sigmaX ?sigmaY img ->
  if kh <= 0 || kh mod 2 = 0 || kw <= 0 || kw mod 2 = 0 then
    invalid_arg "Kernel dimensions must be positive and odd";

  let sigmaY = match sigmaY with None -> sigmaX | Some sy -> sy in
  let kernelX = generate_gaussian_kernel kw sigmaX in
  let kernelY = generate_gaussian_kernel kh sigmaY in

  let img_f32 =
    match Rune.dtype img with
    | Rune.UInt8 -> to_float img
    | Rune.Float32 -> img
    | _ -> failwith "Unsupported image type for gaussian_blur"
  in

  (* Optimized separable convolution using 2D convolution with 1D kernels *)
  let blur_1d_horizontal kernel img =
    match get_dims img with
    | `Gray (_h, _w) ->
        (* Use 2D convolution with a 1xN kernel for horizontal blur *)
        (* Reshape kernel to [1, ksize] *)
        let ksize = Rune.numel kernel in
        let kernel_2d = Rune.reshape [| 1; ksize |] kernel in

        (* Apply convolution using convolve2d_safe which handles the shapes
           correctly *)
        convolve2d_safe kernel_2d img
    | `Color (_h, _w, _c) ->
        (* For color images, also use 2D convolution *)
        let ksize = Rune.numel kernel in
        let kernel_2d = Rune.reshape [| 1; ksize |] kernel in

        (* Apply convolution using convolve2d_safe which handles color
           correctly *)
        convolve2d_safe kernel_2d img
  in

  let blur_1d_vertical kernel img =
    match get_dims img with
    | `Gray (_h, _w) ->
        (* Use 2D convolution with a Nx1 kernel for vertical blur *)
        let ksize = Rune.numel kernel in
        let kernel_2d = Rune.reshape [| ksize; 1 |] kernel in

        (* Apply convolution using convolve2d_safe *)
        convolve2d_safe kernel_2d img
    | `Color (_h, _w, _c) ->
        (* For color images, also use 2D convolution *)
        let ksize = Rune.numel kernel in
        let kernel_2d = Rune.reshape [| ksize; 1 |] kernel in

        (* Apply convolution using convolve2d_safe *)
        convolve2d_safe kernel_2d img
  in

  (* Apply horizontal then vertical blur *)
  let temp = blur_1d_horizontal kernelX img_f32 in
  let blurred_f32 = blur_1d_vertical kernelY temp in

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
      | Trunc ->
          (* For Trunc, pixels above threshold are capped at threshold, pixels
             at or below threshold remain unchanged *)
          Rune.where mask thresh_tensor img
      | ToZero -> Rune.where mask img zero_tensor
      | ToZeroInv -> Rune.where mask zero_tensor img)

let box_filter : type a b. ksize:int * int -> (a, b) Rune.t -> (a, b) Rune.t =
 fun ~ksize:(kh, kw) img ->
  if kh <= 0 || kw <= 0 then invalid_arg "Kernel dimensions must be positive";

  let img_f32 =
    match Rune.dtype img with
    | Rune.UInt8 -> to_float img
    | Rune.Float32 -> img
    | _ -> failwith "Unsupported image type for box_filter"
  in

  (* Optimized box filter using cumulative approach *)
  match get_dims img_f32 with
  | `Gray (h, w) -> (
      (* Pad the image *)
      let pad_h = kh / 2 in
      let pad_w = kw / 2 in
      let padded = Rune.pad [| (pad_h, pad_h); (pad_w, pad_w) |] 0.0 img_f32 in

      (* Sum over window using shifts and additions *)
      let result = ref (Rune.zeros_like img_f32) in
      for i = 0 to kh - 1 do
        for j = 0 to kw - 1 do
          let shifted = Rune.slice [ R (i, i + h); R (j, j + w) ] padded in
          result := Rune.add !result shifted
        done
      done;

      (* Divide by kernel size to get average *)
      let filtered_f32 = Rune.div_s !result (float_of_int (kh * kw)) in

      match Rune.dtype img with
      | Rune.UInt8 -> to_uint8 filtered_f32
      | Rune.Float32 -> filtered_f32
      | _ -> failwith "Unsupported image type for box_filter")
  | `Color (h, w, _c) -> (
      (* Pad the image *)
      let pad_h = kh / 2 in
      let pad_w = kw / 2 in
      let padded =
        Rune.pad [| (pad_h, pad_h); (pad_w, pad_w); (0, 0) |] 0.0 img_f32
      in

      (* Sum over window using shifts and additions *)
      let result = ref (Rune.zeros_like img_f32) in
      for i = 0 to kh - 1 do
        for j = 0 to kw - 1 do
          let shifted = Rune.slice [ R (i, i + h); R (j, j + w); A ] padded in
          result := Rune.add !result shifted
        done
      done;

      (* Divide by kernel size to get average *)
      let filtered_f32 = Rune.div_s !result (float_of_int (kh * kw)) in

      match Rune.dtype img with
      | Rune.UInt8 -> to_uint8 filtered_f32
      | Rune.Float32 -> filtered_f32
      | _ -> failwith "Unsupported image type for box_filter")

let median_blur ~ksize (img : uint8_t) : uint8_t =
  if Rune.dtype img <> Rune.uint8 then
    failwith "Median blur currently only supports uint8 images";
  if ksize <= 0 || ksize mod 2 = 0 then
    invalid_arg "Kernel size (ksize) must be positive and odd";

  let pad = ksize / 2 in
  let shape = Rune.shape img in
  match Array.length shape with
  | 2 ->
      let h = shape.(0) in
      let w = shape.(1) in
      let padded = Rune.pad [| (pad, pad); (pad, pad) |] 0 img in
      let windows =
        let acc = ref [] in
        for dy = 0 to ksize - 1 do
          for dx = 0 to ksize - 1 do
            let slice =
              Rune.slice [ Rune.R (dy, dy + h); Rune.R (dx, dx + w) ] padded
            in
            acc := slice :: !acc
          done
        done;
        List.rev !acc
      in
      let stacked = Rune.stack ~axis:0 windows in
      let sorted, _ = Rune.sort ~axis:0 stacked in
      let median_idx = ksize * ksize / 2 in
      Rune.slice [ Rune.I median_idx; Rune.A; Rune.A ] sorted
  | 3 ->
      let h = shape.(1) in
      let w = shape.(2) in
      let padded = Rune.pad [| (0, 0); (pad, pad); (pad, pad) |] 0 img in
      let windows =
        let acc = ref [] in
        for dy = 0 to ksize - 1 do
          for dx = 0 to ksize - 1 do
            let slice =
              Rune.slice
                [ Rune.A; Rune.R (dy, dy + h); Rune.R (dx, dx + w) ]
                padded
            in
            acc := slice :: !acc
          done
        done;
        List.rev !acc
      in
      let stacked = Rune.stack ~axis:0 windows in
      let sorted, _ = Rune.sort ~axis:0 stacked in
      let median_idx = ksize * ksize / 2 in
      Rune.slice [ Rune.I median_idx; Rune.A; Rune.A; Rune.A ] sorted
  | _ ->
      failwith "Median blur currently only supports 2D or 3D grayscale images"

let blur = box_filter

type structuring_element_shape = Rect | Cross

let get_structuring_element ~shape ~ksize:(kh, kw) =
  if kh <= 0 || kw <= 0 then invalid_arg "Kernel dimensions must be positive";

  match shape with
  | Rect -> Rune.ones Rune.uint8 [| kh; kw |]
  | Cross ->
      (* Create cross pattern using tensor operations *)
      let center_h = kh / 2 in
      let center_w = kw / 2 in

      (* Create horizontal line *)
      let h_line = Rune.ones Rune.uint8 [| 1; kw |] in
      let h_line_padded =
        Rune.pad [| (center_h, kh - center_h - 1); (0, 0) |] 0 h_line
      in

      (* Create vertical line *)
      let v_line = Rune.ones Rune.uint8 [| kh; 1 |] in
      let v_line_padded =
        Rune.pad [| (0, 0); (center_w, kw - center_w - 1) |] 0 v_line
      in

      (* Combine using logical or *)
      Rune.maximum h_line_padded v_line_padded

let morph_op ~op ~kernel (img : uint8_t) : uint8_t =
  if Rune.dtype img <> Rune.uint8 then
    failwith "Morphological operations currently require uint8 input";

  let kernel_shape = Rune.shape kernel in
  let kh, kw =
    match kernel_shape with
    | [| kh; kw |] -> (kh, kw)
    | _ -> failwith "Kernel must be 2D"
  in

  if kh <= 0 || kw <= 0 || kh mod 2 = 0 || kw mod 2 = 0 then
    failwith "Kernel dimensions must be positive and odd";

  let pad_h = kh / 2 in
  let pad_w = kw / 2 in

  let active_positions =
    let positions = ref [] in
    for i = 0 to kh - 1 do
      for j = 0 to kw - 1 do
        if Rune.item [ i; j ] kernel <> 0 then positions := (i, j) :: !positions
      done
    done;
    match List.rev !positions with
    | [] -> invalid_arg "Kernel must contain at least one non-zero element"
    | xs -> xs
  in

  let reduce_max slices =
    match slices with
    | [] -> failwith "Empty slice list"
    | first :: rest ->
        List.fold_left (fun acc slice -> Rune.maximum acc slice) first rest
  in
  let reduce_min slices =
    match slices with
    | [] -> failwith "Empty slice list"
    | first :: rest ->
        List.fold_left (fun acc slice -> Rune.minimum acc slice) first rest
  in

  match Array.length (Rune.shape img) with
  | 2 -> (
      let h = (Rune.shape img).(0) in
      let w = (Rune.shape img).(1) in
      let pad_value = match op with `Max -> 0 | `Min -> 255 in
      let padded =
        Rune.pad [| (pad_h, pad_h); (pad_w, pad_w) |] pad_value img
      in
      let slices =
        List.map
          (fun (dy, dx) ->
            Rune.slice [ Rune.R (dy, dy + h); Rune.R (dx, dx + w) ] padded)
          active_positions
      in
      match op with `Max -> reduce_max slices | `Min -> reduce_min slices)
  | 3 -> (
      let shape = Rune.shape img in
      let h = shape.(1) in
      let w = shape.(2) in
      let pad_value = match op with `Max -> 0 | `Min -> 255 in
      let padded =
        Rune.pad [| (0, 0); (pad_h, pad_h); (pad_w, pad_w) |] pad_value img
      in
      let slices =
        List.map
          (fun (dy, dx) ->
            Rune.slice
              [ Rune.A; Rune.R (dy, dy + h); Rune.R (dx, dx + w) ]
              padded)
          active_positions
      in
      match op with `Max -> reduce_max slices | `Min -> reduce_min slices)
  | _ -> failwith "Morphological operations currently support 2D or 3D tensors"

let erode ~kernel img = morph_op ~op:`Min ~kernel img
let dilate ~kernel img = morph_op ~op:`Max ~kernel img

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
      let tl = Rune.slice [ R (0, h); R (0, w) ] img_padded in
      let tc = Rune.slice [ R (0, h); R (1, w + 1) ] img_padded in
      let tr = Rune.slice [ R (0, h); R (2, w + 2) ] img_padded in
      let ml = Rune.slice [ R (1, h + 1); R (0, w) ] img_padded in
      let mr = Rune.slice [ R (1, h + 1); R (2, w + 2) ] img_padded in
      let bl = Rune.slice [ R (2, h + 2); R (0, w) ] img_padded in
      let bc = Rune.slice [ R (2, h + 2); R (1, w + 1) ] img_padded in
      let br = Rune.slice [ R (2, h + 2); R (2, w + 2) ] img_padded in

      (* Apply Sobel kernels manually *)
      let result_f32 =
        match (dx, dy) with
        | 1, 0 ->
            (* Sobel X: [-1 0 1; -2 0 2; -1 0 1] *)
            Rune.add
              (Rune.add (Rune.sub tr tl)
                 (Rune.sub (Rune.mul_s mr 2.0) (Rune.mul_s ml 2.0)))
              (Rune.sub br bl)
        | 0, 1 ->
            (* Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1] *)
            Rune.add
              (Rune.add (Rune.sub bl tl)
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
  let left = Rune.slice [ R (1, h + 1); R (0, w) ] mag_padded in
  let right = Rune.slice [ R (1, h + 1); R (2, w + 2) ] mag_padded in
  let center = Rune.slice [ R (1, h + 1); R (1, w + 1) ] mag_padded in

  (* Vertical: top and bottom neighbors *)
  let top = Rune.slice [ R (0, h); R (1, w + 1) ] mag_padded in
  let bottom = Rune.slice [ R (2, h + 2); R (1, w + 1) ] mag_padded in

  (* Diagonal 1: top-right and bottom-left *)
  let top_right = Rune.slice [ R (0, h); R (2, w + 2) ] mag_padded in
  let bottom_left = Rune.slice [ R (2, h + 2); R (0, w) ] mag_padded in

  (* Diagonal 2: top-left and bottom-right *)
  let top_left = Rune.slice [ R (0, h); R (0, w) ] mag_padded in
  let bottom_right = Rune.slice [ R (2, h + 2); R (2, w + 2) ] mag_padded in

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

  (* Extract strong edges only *)
  let strong_only = Rune.where strong_edges strong_val zero_val in

  (* Dilate strong edges multiple times *)
  let dilated = ref strong_only in

  (* Use morphological dilation to connect weak edges to strong edges *)
  let kernel_3x3 = get_structuring_element ~shape:Rect ~ksize:(3, 3) in
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
