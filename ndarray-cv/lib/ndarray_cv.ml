module BA = Bigarray
module Array1 = BA.Array1

type uint8_t = Ndarray.uint8_t
type int16_t = Ndarray.int16_t
type float32_t = Ndarray.float32_t

let array_sort_inplace a = Array.sort compare a

let get_dims img =
  match Ndarray.shape img with
  | [| h; w |] -> `Gray (h, w)
  | [| h; w; c |] -> `Color (h, w, c)
  | s ->
      failwith
        (Printf.sprintf "Invalid image dimensions: expected 2 or 3, got %d (%s)"
           (Array.length s)
           (Array.to_list s |> List.map string_of_int |> String.concat "x"))

let flip_axis img axis =
  let shape = Ndarray.shape img in
  if Array.length shape <= axis || shape.(axis) <= 1 then img
  else
    let n = shape.(axis) in
    let ndims = Array.length shape in
    let img_copy = Ndarray.copy img in
    let starts1 = Array.make ndims 0 in
    let stops1 = Array.copy shape in
    let starts2 = Array.make ndims 0 in
    let stops2 = Array.copy shape in
    let slice_shape = Array.copy shape in
    slice_shape.(axis) <- 1;
    let temp_slice = Ndarray.empty (Ndarray.dtype img) slice_shape in

    for i = 0 to (n / 2) - 1 do
      let i' = n - 1 - i in
      starts1.(axis) <- i;
      stops1.(axis) <- i + 1;
      starts2.(axis) <- i';
      stops2.(axis) <- i' + 1;
      try
        let slice1_view = Ndarray.slice starts1 stops1 img_copy in
        let slice2_view = Ndarray.slice starts2 stops2 img_copy in
        Ndarray.blit slice1_view temp_slice;
        Ndarray.blit slice2_view slice1_view;
        Ndarray.blit temp_slice slice2_view
      with e ->
        Printf.eprintf
          "Warning: Blitting slices failed during flip. Error: %s\n"
          (Printexc.to_string e);
        failwith "Slice blitting required for flip_axis"
    done;
    img_copy

let flip_vertical img = if Ndarray.ndim img < 1 then img else flip_axis img 0
let flip_horizontal img = if Ndarray.ndim img < 2 then img else flip_axis img 1

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
      else
        let starts = [| y; x |] in
        let stops = [| y + height; x + width |] in
        Ndarray.copy (Ndarray.slice starts stops img)
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
      else
        let starts = [| y; x; 0 |] in
        let stops = [| y + height; x + width; c |] in
        Ndarray.copy (Ndarray.slice starts stops img)

let to_grayscale img =
  match get_dims img with
  | `Gray (_, _) -> img
  | `Color (h, w, c) ->
      if c <> 3 then
        failwith "to_grayscale requires 3-channel (RGB) input image"
      else
        let img_f = Ndarray.astype Ndarray.float32 img in
        let r = Ndarray.slice [| 0; 0; 0 |] [| h; w; 1 |] img_f in
        let g = Ndarray.slice [| 0; 0; 1 |] [| h; w; 2 |] img_f in
        let b = Ndarray.slice [| 0; 0; 2 |] [| h; w; 3 |] img_f in
        let r_scaled = Ndarray.mul r (Ndarray.scalar Ndarray.float32 0.299) in
        let g_scaled = Ndarray.mul g (Ndarray.scalar Ndarray.float32 0.587) in
        let b_scaled = Ndarray.mul b (Ndarray.scalar Ndarray.float32 0.114) in
        let gray_f = Ndarray.add r_scaled g_scaled in
        let gray_f = Ndarray.add_inplace gray_f b_scaled in
        let gray_f_2d = Ndarray.reshape [| h; w |] gray_f in
        Ndarray.astype Ndarray.uint8 gray_f_2d

let swap_channels img =
  match get_dims img with
  | `Gray (_, _) -> img
  | `Color (h, w, c) ->
      if c < 3 then img
      else
        let new_img = Ndarray.copy img in
        let starts_c0 = [| 0; 0; 0 |] in
        let stops_c0 = [| h; w; 1 |] in
        let starts_c2 = [| 0; 0; 2 |] in
        let stops_c2 = [| h; w; 3 |] in
        let chan0_orig_view = Ndarray.slice starts_c0 stops_c0 img in
        let chan2_orig_view = Ndarray.slice starts_c2 stops_c2 img in
        let chan0_new_view = Ndarray.slice starts_c0 stops_c0 new_img in
        let chan2_new_view = Ndarray.slice starts_c2 stops_c2 new_img in
        Ndarray.blit chan2_orig_view chan0_new_view;
        Ndarray.blit chan0_orig_view chan2_new_view;
        new_img

let rgb_to_bgr = swap_channels
let bgr_to_rgb = swap_channels

let to_float (img : uint8_t) =
  if Ndarray.dtype img <> Ndarray.uint8 then
    failwith "to_float requires uint8 input"
  else
    let float_img = Ndarray.astype Ndarray.float32 img in
    let s = Ndarray.scalar Ndarray.float32 (1.0 /. 255.0) in
    let _ = Ndarray.mul_inplace float_img s in
    float_img

let to_uint8 (img : float32_t) =
  if Ndarray.dtype img <> Ndarray.float32 then
    failwith "to_uint8 requires float32 input"
  else
    let s_zero = Ndarray.scalar Ndarray.float32 0.0 in
    let s_one = Ndarray.scalar Ndarray.float32 1.0 in
    let clipped_img = Ndarray.maximum s_zero (Ndarray.minimum s_one img) in
    let s_255 = Ndarray.scalar Ndarray.float32 255.0 in
    let scaled_img = Ndarray.mul clipped_img s_255 in
    Ndarray.astype Ndarray.uint8 scaled_img

type interpolation = Nearest | Linear

let resize ?(interpolation = Nearest) ~height:out_h ~width:out_w img =
  if out_h <= 0 || out_w <= 0 then
    invalid_arg "Output height and width must be positive";

  let dtype = Ndarray.dtype img in
  match get_dims img with
  | `Gray (in_h, in_w) ->
      let out_img = Ndarray.empty dtype [| out_h; out_w |] in
      let data_out = Ndarray.data out_img in
      (match interpolation with
      | Nearest ->
          let data_in = Ndarray.data img in
          let h_scale = float in_h /. float out_h in
          let w_scale = float in_w /. float out_w in
          for i = 0 to out_h - 1 do
            let src_i = int_of_float (float i *. h_scale) in
            let src_i = max 0 (min (in_h - 1) src_i) in
            let src_i_offset = src_i * in_w in
            let out_i_offset = i * out_w in
            for j = 0 to out_w - 1 do
              let src_j = int_of_float (float j *. w_scale) in
              let src_j = max 0 (min (in_w - 1) src_j) in
              let in_idx = src_i_offset + src_j in
              let pixel = Array1.unsafe_get data_in in_idx in
              let out_idx = out_i_offset + j in
              Array1.unsafe_set data_out out_idx pixel
            done
          done
      | Linear ->
          let img_f = to_float img in
          let out_img_f = Ndarray.empty Ndarray.float32 [| out_h; out_w |] in
          let data_in_f = Ndarray.data img_f in
          let data_out_f = Ndarray.data out_img_f in
          let h_scale = float in_h /. float out_h in
          let w_scale = float in_w /. float out_w in

          for i = 0 to out_h - 1 do
            let src_if = ((float i +. 0.5) *. h_scale) -. 0.5 in
            let y1 = int_of_float src_if in
            let yf = src_if -. float y1 in
            let y1c = max 0 (min (in_h - 1) y1) in
            let y2c = max 0 (min (in_h - 1) (y1 + 1)) in
            let y1c_offset = y1c * in_w in
            let y2c_offset = y2c * in_w in
            let out_i_offset = i * out_w in

            for j = 0 to out_w - 1 do
              let src_jf = ((float j +. 0.5) *. w_scale) -. 0.5 in
              let x1 = int_of_float src_jf in
              let xf = src_jf -. float x1 in
              let x1c = max 0 (min (in_w - 1) x1) in
              let x2c = max 0 (min (in_w - 1) (x1 + 1)) in

              let idx11 = y1c_offset + x1c in
              let idx12 = y1c_offset + x2c in
              let idx21 = y2c_offset + x1c in
              let idx22 = y2c_offset + x2c in

              let p11 = Array1.unsafe_get data_in_f idx11 in
              let p12 = Array1.unsafe_get data_in_f idx12 in
              let p21 = Array1.unsafe_get data_in_f idx21 in
              let p22 = Array1.unsafe_get data_in_f idx22 in

              let w11 = (1.0 -. yf) *. (1.0 -. xf) in
              let w12 = (1.0 -. yf) *. xf in
              let w21 = yf *. (1.0 -. xf) in
              let w22 = yf *. xf in

              let value =
                (p11 *. w11) +. (p12 *. w12) +. (p21 *. w21) +. (p22 *. w22)
              in
              let out_idx = out_i_offset + j in
              Array1.unsafe_set data_out_f out_idx value
            done
          done;
          ignore (to_uint8 out_img_f : uint8_t));
      out_img
  | `Color (in_h, in_w, c) ->
      let out_img = Ndarray.empty dtype [| out_h; out_w; c |] in
      let data_out = Ndarray.data out_img in
      (match interpolation with
      | Nearest ->
          let data_in = Ndarray.data img in
          let h_scale = float in_h /. float out_h in
          let w_scale = float in_w /. float out_w in
          let in_row_stride = in_w * c in
          let out_row_stride = out_w * c in

          for i = 0 to out_h - 1 do
            let src_i = int_of_float (float i *. h_scale) in
            let src_i = max 0 (min (in_h - 1) src_i) in
            let src_i_offset = src_i * in_row_stride in
            let out_i_offset = i * out_row_stride in
            for j = 0 to out_w - 1 do
              let src_j = int_of_float (float j *. w_scale) in
              let src_j = max 0 (min (in_w - 1) src_j) in
              let src_pixel_offset = src_i_offset + (src_j * c) in
              let out_pixel_offset = out_i_offset + (j * c) in
              for k = 0 to c - 1 do
                let pixel = Array1.unsafe_get data_in (src_pixel_offset + k) in
                Array1.unsafe_set data_out (out_pixel_offset + k) pixel
              done
            done
          done
      | Linear ->
          let img_f = to_float img in
          let out_img_f = Ndarray.empty Ndarray.float32 [| out_h; out_w; c |] in
          let data_in_f = Ndarray.data img_f in
          let data_out_f = Ndarray.data out_img_f in
          let h_scale = float in_h /. float out_h in
          let w_scale = float in_w /. float out_w in
          let in_row_stride = in_w * c in
          let out_row_stride = out_w * c in

          for i = 0 to out_h - 1 do
            let src_if = ((float i +. 0.5) *. h_scale) -. 0.5 in
            let y1 = int_of_float src_if in
            let yf = src_if -. float y1 in
            let y1c = max 0 (min (in_h - 1) y1) in
            let y2c = max 0 (min (in_h - 1) (y1 + 1)) in
            let y1c_offset = y1c * in_row_stride in
            let y2c_offset = y2c * in_row_stride in
            let out_i_offset = i * out_row_stride in

            for j = 0 to out_w - 1 do
              let src_jf = ((float j +. 0.5) *. w_scale) -. 0.5 in
              let x1 = int_of_float src_jf in
              let xf = src_jf -. float x1 in
              let x1c = max 0 (min (in_w - 1) x1) in
              let x2c = max 0 (min (in_w - 1) (x1 + 1)) in

              let x1c_pix_offset = x1c * c in
              let x2c_pix_offset = x2c * c in
              let out_pix_offset = out_i_offset + (j * c) in

              let w11 = (1.0 -. yf) *. (1.0 -. xf) in
              let w12 = (1.0 -. yf) *. xf in
              let w21 = yf *. (1.0 -. xf) in
              let w22 = yf *. xf in

              for k = 0 to c - 1 do
                let idx11 = y1c_offset + x1c_pix_offset + k in
                let idx12 = y1c_offset + x2c_pix_offset + k in
                let idx21 = y2c_offset + x1c_pix_offset + k in
                let idx22 = y2c_offset + x2c_pix_offset + k in

                let p11 = Array1.unsafe_get data_in_f idx11 in
                let p12 = Array1.unsafe_get data_in_f idx12 in
                let p21 = Array1.unsafe_get data_in_f idx21 in
                let p22 = Array1.unsafe_get data_in_f idx22 in

                let value =
                  (p11 *. w11) +. (p12 *. w12) +. (p21 *. w21) +. (p22 *. w22)
                in
                Array1.unsafe_set data_out_f (out_pix_offset + k) value
              done
            done
          done;
          ignore (to_uint8 out_img_f : uint8_t));
      out_img

let generate_gaussian_kernel size sigma =
  let sigma =
    if sigma <= 0.0 then (0.3 *. (float (size / 2) -. 1.0)) +. 0.8 else sigma
  in
  let kernel = Ndarray.empty Ndarray.float32 [| size |] in
  let center = size / 2 in
  let sigma2_sq = 2.0 *. sigma *. sigma in
  let sum = ref 0.0 in
  for i = 0 to size - 1 do
    let x = float (i - center) in
    let value = exp (-.(x *. x) /. sigma2_sq) in
    Ndarray.set_item [| i |] value kernel;
    sum := !sum +. value
  done;
  (if !sum > 0.0 then
     let s = Ndarray.scalar Ndarray.float32 (1.0 /. !sum) in
     let _ = Ndarray.mul_inplace kernel s in
     ());
  kernel

let clamp_u8 v = max 0 (min 255 v)

let convolve1d_f32 axis (kernel : float32_t) (img_f32 : float32_t) : float32_t =
  let h, w, c_opt =
    match get_dims img_f32 with
    | `Gray (h, w) -> (h, w, None)
    | `Color (h, w, c) -> (h, w, Some c)
  in
  let ksize = Ndarray.dim 0 kernel in
  let radius = ksize / 2 in
  let out_img_f32 = Ndarray.empty_like img_f32 in
  let data_in = Ndarray.data img_f32 in
  let data_out = Ndarray.data out_img_f32 in
  let data_kernel = Ndarray.data kernel in

  let () =
    match c_opt with
    | None ->
        let row_stride = w in
        for i = 0 to h - 1 do
          let i_offset = i * row_stride in
          for j = 0 to w - 1 do
            let sum = ref 0.0 in
            for k = 0 to ksize - 1 do
              let kernel_val = Array1.unsafe_get data_kernel k in
              let offset = k - radius in
              let src_i, src_j =
                if axis = 0 then (max 0 (min (h - 1) (i + offset)), j)
                else (i, max 0 (min (w - 1) (j + offset)))
              in
              let in_idx = (src_i * row_stride) + src_j in
              let pixel_val = Array1.unsafe_get data_in in_idx in
              sum := !sum +. (pixel_val *. kernel_val)
            done;
            let out_idx = i_offset + j in
            Array1.unsafe_set data_out out_idx !sum
          done
        done
    | Some c ->
        let in_row_stride = w * c in
        let out_row_stride = w * c in
        for i = 0 to h - 1 do
          let i_offset_out = i * out_row_stride in
          for j = 0 to w - 1 do
            let j_offset_out = j * c in
            for k_chan = 0 to c - 1 do
              let sum = ref 0.0 in
              for k = 0 to ksize - 1 do
                let kernel_val = Array1.unsafe_get data_kernel k in
                let offset = k - radius in
                let src_i, src_j =
                  if axis = 0 then (max 0 (min (h - 1) (i + offset)), j)
                  else (i, max 0 (min (w - 1) (j + offset)))
                in
                let in_idx = (src_i * in_row_stride) + (src_j * c) + k_chan in
                let pixel_val = Array1.unsafe_get data_in in_idx in
                sum := !sum +. (pixel_val *. kernel_val)
              done;
              let out_idx = i_offset_out + j_offset_out + k_chan in
              Array1.unsafe_set data_out out_idx !sum
            done
          done
        done
  in
  out_img_f32

let gaussian_blur : type a b.
    ksize:int * int ->
    sigmaX:float ->
    ?sigmaY:float ->
    (a, b) Ndarray.t ->
    (a, b) Ndarray.t =
 fun ~ksize:(kh, kw) ~sigmaX ?sigmaY img ->
  if kh <= 0 || kh mod 2 = 0 || kw <= 0 || kw mod 2 = 0 then
    invalid_arg "Kernel dimensions must be positive and odd";

  let sigmaY = match sigmaY with None -> sigmaX | Some sy -> sy in
  let kernelX = generate_gaussian_kernel kw sigmaX in
  let kernelY = generate_gaussian_kernel kh sigmaY in

  let img_f32 =
    match Ndarray.dtype img with
    | Ndarray.UInt8 -> to_float img
    | Ndarray.Float32 -> img
    | _ -> failwith "Unsupported image type for gaussian_blur"
  in

  let blurred_h = convolve1d_f32 1 kernelX img_f32 in
  let blurred_final_f32 = convolve1d_f32 0 kernelY blurred_h in

  match Ndarray.dtype img with
  | Ndarray.UInt8 -> to_uint8 blurred_final_f32
  | Ndarray.Float32 -> blurred_final_f32
  | _ -> failwith "Unsupported image type for gaussian_blur"

type threshold_type = Binary | BinaryInv | Trunc | ToZero | ToZeroInv

let threshold ~thresh ~maxval ~type_ (img : uint8_t) : uint8_t =
  match get_dims img with
  | `Color _ ->
      failwith "Threshold currently only supports grayscale (2D) images"
  | `Gray (h, w) ->
      if Ndarray.dtype img <> Ndarray.uint8 then
        failwith "Threshold currently only supports uint8 images";

      let thresh_val = clamp_u8 thresh in
      let maxval_val = clamp_u8 maxval in
      if thresh_val < 0 || maxval_val < 0 then
        invalid_arg "Threshold and maxval must be non-negative";

      let out_img = Ndarray.empty_like img in
      let data_in = Ndarray.data img in
      let data_out = Ndarray.data out_img in
      let row_stride = w in
      for i = 0 to h - 1 do
        let i_offset = i * row_stride in
        for j = 0 to w - 1 do
          let idx = i_offset + j in
          let pixel = Array1.unsafe_get data_in idx in
          let value =
            match type_ with
            | Binary -> if pixel > thresh_val then maxval_val else 0
            | BinaryInv -> if pixel > thresh_val then 0 else maxval_val
            | Trunc -> min pixel thresh_val
            | ToZero -> if pixel > thresh_val then pixel else 0
            | ToZeroInv -> if pixel > thresh_val then 0 else pixel
          in
          Array1.unsafe_set data_out idx value
        done
      done;
      out_img

let box_filter : type a b.
    ksize:int * int -> (a, b) Ndarray.t -> (a, b) Ndarray.t =
 fun ~ksize:(kh, kw) img ->
  if kh <= 0 || kw <= 0 then invalid_arg "Kernel dimensions must be positive";

  let kernel_val_x = 1.0 /. float_of_int kw in
  let kernel_val_y = 1.0 /. float_of_int kh in
  let kernelX = Ndarray.full Ndarray.float32 [| kw |] kernel_val_x in
  let kernelY = Ndarray.full Ndarray.float32 [| kh |] kernel_val_y in

  let img_f32 =
    match Ndarray.dtype img with
    | Ndarray.UInt8 -> to_float img
    | Ndarray.Float32 -> img
    | _ -> failwith "Unsupported image type for box_filter"
  in

  let filtered_h = convolve1d_f32 1 kernelX img_f32 in
  let filtered_final_f32 = convolve1d_f32 0 kernelY filtered_h in

  match Ndarray.dtype img with
  | Ndarray.UInt8 -> to_uint8 filtered_final_f32
  | Ndarray.Float32 -> filtered_final_f32
  | _ -> failwith "Unsupported image type for gaussian_blur"

let median_blur ~ksize (img : uint8_t) : uint8_t =
  if Ndarray.dtype img <> Ndarray.uint8 then
    failwith "Median blur currently only supports uint8 images";
  if ksize <= 0 || ksize mod 2 = 0 then
    invalid_arg "Kernel size (ksize) must be positive and odd";

  let radius = ksize / 2 in
  let num_neighbors = ksize * ksize in
  let neighbors = Array.make num_neighbors 0 in
  let median_index = num_neighbors / 2 in

  match get_dims img with
  | `Color _ -> failwith "Median blur currently only supports grayscale images"
  | `Gray (h, w) ->
      let out_img = Ndarray.empty_like img in
      let data_in = Ndarray.data img in
      let data_out = Ndarray.data out_img in
      let row_stride = w in

      for i = 0 to h - 1 do
        let out_i_offset = i * row_stride in
        for j = 0 to w - 1 do
          let n_idx = ref 0 in
          for ki = 0 to ksize - 1 do
            let src_i = max 0 (min (h - 1) (i + ki - radius)) in
            let src_i_offset = src_i * row_stride in
            for kj = 0 to ksize - 1 do
              let src_j = max 0 (min (w - 1) (j + kj - radius)) in
              let in_idx = src_i_offset + src_j in
              neighbors.(!n_idx) <- Array1.unsafe_get data_in in_idx;
              incr n_idx
            done
          done;
          array_sort_inplace neighbors;
          let out_idx = out_i_offset + j in
          Array1.unsafe_set data_out out_idx neighbors.(median_index)
        done
      done;
      out_img

let blur = box_filter

type structuring_element_shape = Rect | Cross

let get_structuring_element ~shape ~ksize:(kh, kw) =
  if kh <= 0 || kw <= 0 then invalid_arg "Kernel dimensions must be positive";
  let kernel = Ndarray.zeros Ndarray.uint8 [| kh; kw |] in
  let center_h = kh / 2 in
  let center_w = kw / 2 in
  match shape with
  | Rect ->
      Ndarray.fill 1 kernel;
      kernel
  | Cross ->
      let data_k = Ndarray.data kernel in
      let row_stride_k = kw in
      for i = 0 to kh - 1 do
        let idx = (i * row_stride_k) + center_w in
        Array1.unsafe_set data_k idx 1
      done;
      let center_row_offset = center_h * row_stride_k in
      for j = 0 to kw - 1 do
        let idx = center_row_offset + j in
        Array1.unsafe_set data_k idx 1
      done;
      kernel

let morph_op ~op ~kernel (img : uint8_t) : uint8_t =
  if Ndarray.dtype img <> Ndarray.uint8 then
    failwith "Morphological operations currently require uint8 input";
  let kh, kw =
    match Ndarray.shape kernel with
    | [| kh; kw |] -> (kh, kw)
    | _ -> failwith "Kernel must be 2D"
  in
  if kh <= 0 || kw <= 0 || kh mod 2 = 0 || kw mod 2 = 0 then
    failwith "Kernel dimensions must be positive and odd for centered operation";

  let radius_h = kh / 2 in
  let radius_w = kw / 2 in

  match get_dims img with
  | `Color _ ->
      failwith "Morphological operations currently support grayscale only"
  | `Gray (h, w) ->
      let out_img = Ndarray.empty_like img in
      let data_in = Ndarray.data img in
      let data_out = Ndarray.data out_img in
      let data_kernel = Ndarray.data kernel in
      let row_stride = w in
      let kernel_row_stride = kw in

      for i = 0 to h - 1 do
        let out_i_offset = i * row_stride in
        for j = 0 to w - 1 do
          let current_best = ref (match op with `Min -> 255 | `Max -> 0) in
          for ki = 0 to kh - 1 do
            let kernel_i_offset = ki * kernel_row_stride in
            let src_i_base = i + ki - radius_h in
            for kj = 0 to kw - 1 do
              let kernel_idx = kernel_i_offset + kj in
              if Array1.unsafe_get data_kernel kernel_idx <> 0 then
                let src_j_base = j + kj - radius_w in
                if
                  src_i_base >= 0 && src_i_base < h && src_j_base >= 0
                  && src_j_base < w
                then
                  let in_idx = (src_i_base * row_stride) + src_j_base in
                  let pixel_val = Array1.unsafe_get data_in in_idx in
                  current_best :=
                    match op with
                    | `Min -> min !current_best pixel_val
                    | `Max -> max !current_best pixel_val
            done
          done;
          let out_idx = out_i_offset + j in
          Array1.unsafe_set data_out out_idx !current_best
        done
      done;
      out_img

let erode ~kernel img = morph_op ~op:`Min ~kernel img
let dilate ~kernel img = morph_op ~op:`Max ~kernel img

let sobel_kernel dx dy ksize =
  if ksize <> 3 then failwith "Sobel currently only supports ksize=3";
  match (dx, dy) with
  | 1, 0 ->
      Ndarray.create Ndarray.int16 [| 3; 3 |] [| -1; 0; 1; -2; 0; 2; -1; 0; 1 |]
  | 0, 1 ->
      Ndarray.create Ndarray.int16 [| 3; 3 |] [| -1; -2; -1; 0; 0; 0; 1; 2; 1 |]
  | _ -> failwith "Sobel requires dx=1, dy=0 or dx=0, dy=1"

let convolve2d_int16_from_uint8 (kernel : int16_t) (img : uint8_t) : int16_t =
  let kh, kw =
    match Ndarray.shape kernel with
    | [| kh; kw |] -> (kh, kw)
    | _ -> failwith "K2D"
  in
  if kh mod 2 = 0 || kw mod 2 = 0 then failwith "Kernel dims must be odd";
  let radius_h = kh / 2 in
  let radius_w = kw / 2 in

  match get_dims img with
  | `Color _ -> failwith "convolve2d_int16_from_uint8 supports grayscale only"
  | `Gray (h, w) ->
      let out_img = Ndarray.empty Ndarray.int16 [| h; w |] in
      let data_in = Ndarray.data img in
      let data_out = Ndarray.data out_img in
      let data_kernel = Ndarray.data kernel in
      let row_stride = w in
      let kernel_row_stride = kw in

      for i = 0 to h - 1 do
        let out_i_offset = i * row_stride in
        for j = 0 to w - 1 do
          let sum = ref 0 in
          for ki = 0 to kh - 1 do
            let kernel_i_offset = ki * kernel_row_stride in
            let src_i_base = i + ki - radius_h in
            for kj = 0 to kw - 1 do
              let kernel_val =
                Array1.unsafe_get data_kernel (kernel_i_offset + kj)
              in
              (* Replicate border padding *)
              let src_i = max 0 (min (h - 1) src_i_base) in
              let src_j = max 0 (min (w - 1) (j + kj - radius_w)) in
              let in_idx = (src_i * row_stride) + src_j in
              let pixel_val = Array1.unsafe_get data_in in_idx in
              sum := !sum + (pixel_val * kernel_val)
            done
          done;
          let out_idx = out_i_offset + j in
          Array1.unsafe_set data_out out_idx !sum
        done
      done;
      out_img

let sobel : dx:int -> dy:int -> ?ksize:int -> uint8_t -> int16_t =
 fun ~dx ~dy ?(ksize = 3) img ->
  if Ndarray.dtype img <> Ndarray.uint8 then
    failwith "Sobel currently requires uint8 input";
  if ksize <> 3 then failwith "Sobel currently only supports ksize=3";

  let kernel = sobel_kernel dx dy ksize in
  convolve2d_int16_from_uint8 kernel img

let degrees_of_radians r = r *. 180.0 /. Float.pi

let check_strong_neighbor data w i j strong_val =
  i > 0 && j > 0
  && Array1.unsafe_get data (((i - 1) * w) + (j - 1)) = strong_val
  || (i > 0 && Array1.unsafe_get data (((i - 1) * w) + j) = strong_val)
  || i > 0
     && j < w - 1
     && Array1.unsafe_get data (((i - 1) * w) + (j + 1)) = strong_val
  || (j > 0 && Array1.unsafe_get data ((i * w) + (j - 1)) = strong_val)
  || (j < w - 1 && Array1.unsafe_get data ((i * w) + (j + 1)) = strong_val)
  || i < (Array1.dim data / w) - 1
     && j > 0
     && Array1.unsafe_get data (((i + 1) * w) + (j - 1)) = strong_val
  || i < (Array1.dim data / w) - 1
     && Array1.unsafe_get data (((i + 1) * w) + j) = strong_val
  || i < (Array1.dim data / w) - 1
     && j < w - 1
     && Array1.unsafe_get data (((i + 1) * w) + (j + 1)) = strong_val

let canny ~threshold1 ~threshold2 ?(ksize = 3) (img : uint8_t) : uint8_t =
  if Ndarray.dtype img <> Ndarray.uint8 then
    failwith "Canny currently requires uint8 input";
  if threshold1 < 0.0 || threshold2 < 0.0 then
    invalid_arg "Thresholds must be non-negative";

  let high_thresh, low_thresh =
    if threshold1 > threshold2 then (threshold1, threshold2)
    else (threshold2, threshold1)
  in

  (* 1. Noise Reduction (Use optimized Gaussian Blur) *)
  let blurred_img = gaussian_blur ~ksize:(5, 5) ~sigmaX:1.4 img in

  (* 2. Gradient Calculation (Use optimized Sobel) *)
  let gx = sobel ~dx:1 ~dy:0 ~ksize blurred_img in
  let gy = sobel ~dx:0 ~dy:1 ~ksize blurred_img in
  let gx_f = Ndarray.astype Ndarray.float32 gx in
  let gy_f = Ndarray.astype Ndarray.float32 gy in

  (* Calculate Magnitude and Angle using vectorized ops *)
  let mag =
    Ndarray.sqrt (Ndarray.add (Ndarray.square gx_f) (Ndarray.square gy_f))
  in
  let h, w =
    match Ndarray.shape img with [| h; w |] -> (h, w) | _ -> failwith "C2D"
  in
  let angle = Ndarray.empty Ndarray.float32 [| h; w |] in
  let data_angle = Ndarray.data angle in
  let data_gx_f = Ndarray.data gx_f in
  let data_gy_f = Ndarray.data gy_f in
  let row_stride = w in
  for i = 0 to h - 1 do
    let i_offset = i * row_stride in
    for j = 0 to w - 1 do
      let idx = i_offset + j in
      let gxv = Array1.unsafe_get data_gx_f idx in
      let gyv = Array1.unsafe_get data_gy_f idx in
      Array1.unsafe_set data_angle idx (Float.atan2 gyv gxv)
    done
  done;

  (* 3. Non-Maximum Suppression (Use buffer access) *)
  let nms_img = Ndarray.zeros_like mag in
  let data_mag = Ndarray.data mag in
  let data_angle = Ndarray.data angle in
  let data_nms = Ndarray.data nms_img in

  for i = 1 to h - 2 do
    let i_offset = i * w in
    let i_prev_offset = (i - 1) * w in
    let i_next_offset = (i + 1) * w in
    for j = 1 to w - 2 do
      let current_idx = i_offset + j in
      let current_mag = Array1.unsafe_get data_mag current_idx in
      let current_angle_rad = Array1.unsafe_get data_angle current_idx in
      let angle_deg = degrees_of_radians current_angle_rad in
      let angle_norm =
        if angle_deg < 0.0 then angle_deg +. 180.0 else angle_deg
      in

      let q1_mag, q2_mag =
        if
          (angle_norm >= 0. && angle_norm < 22.5)
          || (angle_norm >= 157.5 && angle_norm <= 180.)
        then
          ( Array1.unsafe_get data_mag (current_idx - 1),
            Array1.unsafe_get data_mag (current_idx + 1) )
        else if angle_norm >= 22.5 && angle_norm < 67.5 then
          ( Array1.unsafe_get data_mag (i_prev_offset + j + 1),
            Array1.unsafe_get data_mag (i_next_offset + j - 1) )
        else if angle_norm >= 67.5 && angle_norm < 112.5 then
          ( Array1.unsafe_get data_mag (i_prev_offset + j),
            Array1.unsafe_get data_mag (i_next_offset + j) )
        else
          ( Array1.unsafe_get data_mag (i_prev_offset + j - 1),
            Array1.unsafe_get data_mag (i_next_offset + j + 1) )
      in

      if current_mag >= q1_mag && current_mag >= q2_mag then
        Array1.unsafe_set data_nms current_idx current_mag
    done
  done;

  (* 4. Double Thresholding (Use buffer access) *)
  let edge_map = Ndarray.zeros Ndarray.uint8 [| h; w |] in
  let data_edge = Ndarray.data edge_map in
  let data_nms = Ndarray.data nms_img in
  let strong_val = 255 in
  let weak_val = 100 in

  for i = 0 to h - 1 do
    let i_offset = i * w in
    for j = 0 to w - 1 do
      let idx = i_offset + j in
      let m = Array1.unsafe_get data_nms idx in
      if m >= high_thresh then Array1.unsafe_set data_edge idx strong_val
      else if m >= low_thresh then Array1.unsafe_set data_edge idx weak_val
    done
  done;

  (* 5. Hysteresis (Use buffer access - simple version, could use
     stack/queue) *)
  let changed = ref true in
  while !changed do
    changed := false;
    for i = 1 to h - 2 do
      let i_offset = i * w in
      for j = 1 to w - 2 do
        let current_idx = i_offset + j in
        if Array1.unsafe_get data_edge current_idx = weak_val then
          if check_strong_neighbor data_edge w i j strong_val then (
            Array1.unsafe_set data_edge current_idx strong_val;
            changed := true)
      done
    done
  done;

  for i = 0 to h - 1 do
    let i_offset = i * w in
    for j = 0 to w - 1 do
      let idx = i_offset + j in
      if Array1.unsafe_get data_edge idx = weak_val then
        Array1.unsafe_set data_edge idx 0
    done
  done;

  edge_map
