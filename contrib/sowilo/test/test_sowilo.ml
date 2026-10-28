(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Sowilo

(* Helpers *)

let create_gray_f h w value = Nx.full Nx.float32 [| h; w; 1 |] value

let create_checkerboard h w =
  let data =
    Array.init (h * w) (fun i ->
        let row = i / w and col = i mod w in
        if (row + col) mod 2 = 0 then 1.0 else 0.0)
  in
  Nx.create Nx.float32 [| h; w; 1 |] data

let create_centered_square h w square_size =
  let data =
    Array.init (h * w) (fun idx ->
        let i = idx / w and j = idx mod w in
        let start_h = (h - square_size) / 2 in
        let start_w = (w - square_size) / 2 in
        if
          i >= start_h
          && i < start_h + square_size
          && j >= start_w
          && j < start_w + square_size
        then 1.0
        else 0.0)
  in
  Nx.create Nx.float32 [| h; w; 1 |] data

let check_shape msg expected_shape tensor =
  equal ~msg (array int) expected_shape (Nx.shape tensor)

let check_pixel_f msg expected tensor indices =
  let actual = Nx.item indices tensor in
  let diff = Float.abs (expected -. actual) in
  if diff > 0.01 then failf "%s: expected ~%.3f, got %.3f" msg expected actual

let check_pixel_i msg expected tensor indices =
  let actual = Nx.item indices tensor in
  equal ~msg int expected actual

(* ───── Geometric Transform Tests ───── *)

let test_flip_vertical () =
  let img = create_checkerboard 4 4 in
  let flipped = vflip img in
  check_pixel_f "top-left after flip"
    (Nx.item [ 3; 0; 0 ] img)
    flipped [ 0; 0; 0 ];
  check_pixel_f "top-right after flip"
    (Nx.item [ 3; 3; 0 ] img)
    flipped [ 0; 3; 0 ];
  check_shape "flip preserves shape" (Nx.shape img) flipped

let test_flip_horizontal () =
  let img = create_checkerboard 4 4 in
  let flipped = hflip img in
  check_pixel_f "top-left after flip"
    (Nx.item [ 0; 3; 0 ] img)
    flipped [ 0; 0; 0 ];
  check_pixel_f "bottom-left after flip"
    (Nx.item [ 3; 3; 0 ] img)
    flipped [ 3; 0; 0 ];
  check_shape "flip preserves shape" (Nx.shape img) flipped

let test_flip_batch () =
  let data = [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |] in
  let img = Nx.create Nx.float32 [| 2; 2; 2; 1 |] data in
  let flipped_v = vflip img in
  check_shape "vertical batch shape" [| 2; 2; 2; 1 |] flipped_v;
  check_pixel_f "batch 0 vertical flip" 3.0 flipped_v [ 0; 0; 0; 0 ];
  check_pixel_f "batch 1 vertical flip" 7.0 flipped_v [ 1; 0; 0; 0 ];
  let flipped_h = hflip img in
  check_pixel_f "batch 0 horizontal flip" 2.0 flipped_h [ 0; 0; 0; 0 ];
  check_pixel_f "batch 1 horizontal flip" 6.0 flipped_h [ 1; 0; 0; 0 ]

let test_crop () =
  let data = Array.init (10 * 10) (fun i -> Float.of_int i /. 100.0) in
  let img = Nx.create Nx.float32 [| 10; 10; 1 |] data in
  let cropped = crop ~y:2 ~x:3 ~height:5 ~width:4 img in
  check_shape "crop shape" [| 5; 4; 1 |] cropped;
  check_pixel_f "crop content" (Nx.item [ 2; 3; 0 ] img) cropped [ 0; 0; 0 ];
  raises ~msg:"crop out of bounds"
    (Invalid_argument "crop: region y=8 x=8 h=5 w=5 exceeds image 10x10")
    (fun () -> ignore (crop ~y:8 ~x:8 ~height:5 ~width:5 img))

let test_crop_batch () =
  let data = Array.init (2 * 4 * 4) (fun i -> Float.of_int i) in
  let img = Nx.create Nx.float32 [| 2; 4; 4; 1 |] data in
  let cropped = crop ~y:1 ~x:1 ~height:2 ~width:2 img in
  check_shape "batch crop shape" [| 2; 2; 2; 1 |] cropped;
  check_pixel_f "batch crop value"
    (Nx.item [ 0; 1; 1; 0 ] img)
    cropped [ 0; 0; 0; 0 ];
  check_pixel_f "batch crop second batch"
    (Nx.item [ 1; 2; 2; 0 ] img)
    cropped [ 1; 1; 1; 0 ]

let test_resize_nearest () =
  let img = Nx.create Nx.float32 [| 2; 2; 1 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let resized = resize ~interpolation:Nearest ~height:4 ~width:4 img in
  check_shape "resize nearest shape" [| 4; 4; 1 |] resized;
  check_pixel_f "nearest top-left" 0.1 resized [ 0; 0; 0 ];
  check_pixel_f "nearest top-right" 0.2 resized [ 0; 3; 0 ];
  check_pixel_f "nearest bottom-left" 0.3 resized [ 3; 0; 0 ];
  check_pixel_f "nearest bottom-right" 0.4 resized [ 3; 3; 0 ]

let test_resize_bilinear () =
  let img = Nx.create Nx.float32 [| 2; 2; 1 |] [| 0.0; 1.0; 0.0; 1.0 |] in
  let resized = resize ~height:3 ~width:3 img in
  check_shape "resize bilinear shape" [| 3; 3; 1 |] resized;
  check_pixel_f "bilinear left edge" 0.0 resized [ 0; 0; 0 ];
  check_pixel_f "bilinear right edge" 1.0 resized [ 0; 2; 0 ];
  let center = Nx.item [ 1; 1; 0 ] resized in
  if center < 0.4 || center > 0.6 then
    failf "Bilinear resize center expected ~0.5, got %.3f" center

let test_resize_batch () =
  let data = [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8 |] in
  let img = Nx.create Nx.float32 [| 2; 2; 2; 1 |] data in
  let resized = resize ~interpolation:Nearest ~height:4 ~width:4 img in
  check_shape "resize batch shape" [| 2; 4; 4; 1 |] resized;
  check_pixel_f "batch0 top-left" 0.1 resized [ 0; 0; 0; 0 ];
  check_pixel_f "batch1 bottom-right" 0.8 resized [ 1; 3; 3; 0 ]

let test_resize_color_bilinear () =
  let img =
    Nx.create Nx.float32 [| 1; 2; 2; 3 |]
      [| 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 1.0; 1.0; 0.0 |]
  in
  let resized = resize ~height:3 ~width:3 img in
  check_shape "resize color shape" [| 1; 3; 3; 3 |] resized;
  let center_r = Nx.item [ 0; 1; 1; 0 ] resized in
  let center_g = Nx.item [ 0; 1; 1; 1 ] resized in
  if center_r < 0.4 || center_r > 0.6 then
    failf "Color bilinear resize R expected ~0.5, got %.3f" center_r;
  if center_g < 0.4 || center_g > 0.6 then
    failf "Color bilinear resize G expected ~0.5, got %.3f" center_g;
  check_pixel_f "corner preserves blue" 0.0 resized [ 0; 0; 0; 2 ]

(* ───── Color Conversion Tests ───── *)

let test_to_grayscale () =
  let rgb =
    Nx.create Nx.float32 [| 2; 2; 3 |]
      [|
        1.0;
        0.0;
        0.0;
        (* Red *)
        0.0;
        1.0;
        0.0;
        (* Green *)
        0.0;
        0.0;
        1.0;
        (* Blue *)
        1.0;
        1.0;
        1.0;
        (* White *)
      |]
  in
  let gray = to_grayscale rgb in
  check_shape "grayscale shape" [| 2; 2; 1 |] gray;
  check_pixel_f "red to gray" 0.299 gray [ 0; 0; 0 ];
  check_pixel_f "green to gray" 0.587 gray [ 0; 1; 0 ];
  check_pixel_f "blue to gray" 0.114 gray [ 1; 0; 0 ];
  check_pixel_f "white to gray" 1.0 gray [ 1; 1; 0 ]

(* Channels past the third, such as alpha, do not change the gray value. *)
let grayscale_extra_channels c =
  let pixels = [| [| 1.0; 0.0; 0.0 |]; [| 0.2; 0.4; 0.6 |] |] in
  let data =
    Array.concat
      (Array.to_list
         (Array.map
            (fun rgb -> Array.init c (fun i -> if i < 3 then rgb.(i) else 0.5))
            pixels))
  in
  let gray = to_grayscale (Nx.create Nx.float32 [| 1; 2; c |] data) in
  check_shape "one channel" [| 1; 2; 1 |] gray;
  let luma rgb =
    (0.299 *. rgb.(0)) +. (0.587 *. rgb.(1)) +. (0.114 *. rgb.(2))
  in
  equal (array (float 1e-6)) (Array.map luma pixels) (Nx.to_array gray)

(* Fewer than 3 channels, an image already gray among them, has no R, G and B
   to weigh. *)
let grayscale_few_channels c =
  let img = Nx.zeros Nx.float32 [| 2; 2; c |] in
  raises_match (Exn.invalid_arg ~substring:"channels") (fun () ->
      to_grayscale img)

(* A rank-2 array has no channel axis, whatever its last dimension. *)
let test_grayscale_rank_2 () =
  let img = Nx.zeros Nx.float32 [| 2; 3 |] in
  raises_match (Exn.invalid_arg ~substring:"rank") (fun () -> to_grayscale img)

(* ───── Data Type Conversion Tests ───── *)

let test_float_conversions () =
  let uint8_img = Nx.full Nx.uint8 [| 2; 2; 1 |] 255 in
  let float_img = to_float uint8_img in
  let float_val = Nx.item [ 0; 0; 0 ] float_img in
  equal ~msg:"to_float normalization" (float 0.001) 1.0 float_val;
  let uint8_back = to_uint8 float_img in
  check_shape "round-trip shape" [| 2; 2; 1 |] uint8_back;
  check_pixel_i "round-trip value" 255 uint8_back [ 0; 0; 0 ];
  let out_of_range =
    Nx.create Nx.float32 [| 2; 2; 1 |] [| -0.5; 0.5; 1.5; 0.75 |]
  in
  let clipped = to_uint8 out_of_range in
  check_pixel_i "clipped negative" 0 clipped [ 0; 0; 0 ];
  check_pixel_i "clipped middle" 127 clipped [ 0; 1; 0 ];
  check_pixel_i "clipped overflow" 255 clipped [ 1; 0; 0 ];
  check_pixel_i "clipped normal" 191 clipped [ 1; 1; 0 ]

(* ───── Filtering Tests ───── *)

let test_gaussian_blur () =
  let img = create_centered_square 10 10 4 in
  let blurred = gaussian_blur ~sigma:1.0 ~ksize:3 img in
  check_shape "blur preserves shape" (Nx.shape img) blurred;
  let edge_val = Nx.item [ 3; 3; 0 ] blurred in
  if edge_val = 0.0 || edge_val = 1.0 then
    failf "Edge not smoothed: got %.3f" edge_val

let test_box_blur () =
  let data = [| 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0 |] in
  let img = Nx.create Nx.float32 [| 3; 3; 1 |] data in
  let filtered = box_blur ~ksize:3 img in
  let center = Nx.item [ 1; 1; 0 ] filtered in
  let expected = 1.0 /. 9.0 in
  if Float.abs (center -. expected) > 0.02 then
    failf "Box filter center: expected ~%.3f, got %.3f" expected center

(* An even window spans [ksize / 2] pixels before each pixel and
   [ksize / 2 - 1] after it: a pixel's value spreads to the pixel itself and the
   ones after it. *)
let test_box_blur_even () =
  let img =
    Nx.create Nx.float32 [| 5; 5; 1 |]
      (Array.init 25 (fun i -> if i = 12 then 1.0 else 0.0))
  in
  let expected =
    Array.init 25 (fun i ->
        let row = i / 5 and col = i mod 5 in
        if (row = 2 || row = 3) && (col = 2 || col = 3) then 0.25 else 0.0)
  in
  equal (array (float 1e-6)) expected (Nx.to_array (box_blur ~ksize:2 img))

(* A kernel larger than the image keeps the image's size: a 7x7 box over a 3x3
   image covers the whole image from every pixel. *)
let test_box_blur_larger_than_image () =
  let img = Nx.ones Nx.float32 [| 3; 3; 1 |] in
  equal (array (float 1e-6))
    (Array.make 9 (9. /. 49.))
    (Nx.to_array (box_blur ~ksize:7 img))

let test_unsharp_mask_kernel_larger_than_image () =
  let img = create_centered_square 10 12 4 in
  check_shape "gaussian_blur keeps the shape" (Nx.shape img)
    (gaussian_blur ~sigma:3.0 img);
  check_shape "unsharp_mask keeps the shape" (Nx.shape img)
    (unsharp_mask ~sigma:3.0 img)

(* A blurred impulse is nonzero exactly on the kernel's [ksize * ksize]
   footprint, so its support shows the default [2 * ceil(3 * sigma) + 1]. *)
let gaussian_default_ksize (sigma, ksize) =
  let n = 21 in
  let img =
    Nx.create Nx.float32 [| n; n; 1 |]
      (Array.init (n * n) (fun i -> if i = n * n / 2 then 1.0 else 0.0))
  in
  let blurred = Nx.to_array (gaussian_blur ~sigma img) in
  let support = Array.fold_left (fun c v -> if v <> 0.0 then c + 1 else c) 0 in
  equal ~msg:"nonzero pixels" int (ksize * ksize) (support blurred)

let gaussian_bad_ksize ksize =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~substring:"ksize") (fun () ->
      gaussian_blur ~sigma:1.0 ~ksize img)

(* A sigma of 0 or less takes a fixed kernel for a ksize of 1 to 9, the
   tables OpenCV's getGaussianKernel uses: a blurred impulse is the outer
   product of the table with itself. *)
let gaussian_tables =
  [
    (0.0, [| 1. |], 1.);
    (0.0, [| 1.; 2.; 1. |], 4.);
    (-0.0, [| 1.; 4.; 6.; 4.; 1. |], 16.);
    (-1.0, [| 2.; 7.; 14.; 18.; 14.; 7.; 2. |], 64.);
    (0.0, [| 4.; 13.; 30.; 51.; 60.; 51.; 30.; 13.; 4. |], 256.);
  ]

let gaussian_table (sigma, taps, denominator) =
  let n = 11 and c = 5 in
  let k = Array.length taps in
  let h = k / 2 in
  let img =
    Nx.create Nx.float32 [| n; n; 1 |]
      (Array.init (n * n) (fun i -> if i = n * n / 2 then 1.0 else 0.0))
  in
  let tap d = if abs d > h then 0.0 else taps.(d + h) /. denominator in
  let expected =
    Array.init (n * n) (fun i -> tap ((i / n) - c) *. tap ((i mod n) - c))
  in
  equal (array (float 1e-7)) expected
    (Nx.to_array (gaussian_blur ~sigma ~ksize:k img))

(* Beyond the tables, a sigma of 0 or less is derived from ksize as
   [0.3 * ((ksize - 1) * 0.5 - 1) + 0.8], so the blur equals the one with that
   sigma. *)
let gaussian_derived_sigma (sigma, ksize) =
  let img = create_centered_square 9 9 3 in
  let derived = (0.3 *. ((Float.of_int (ksize - 1) *. 0.5) -. 1.0)) +. 0.8 in
  equal (array (float 1e-6))
    (Nx.to_array (gaussian_blur ~sigma:derived ~ksize img))
    (Nx.to_array (gaussian_blur ~sigma ~ksize img))

let gaussian_no_sigma sigma =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~substring:"sigma") (fun () ->
      gaussian_blur ~sigma img)

(* A NaN sigma is neither positive nor 0 or less: nothing derives a kernel
   from it. *)
let gaussian_nan_sigma ksize =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~substring:"sigma") (fun () ->
      gaussian_blur ~sigma:Float.nan ~ksize img)

(* A sigma whose default ksize [2 * ceil(3 * sigma) + 1] exceeds [max_int]
   has no kernel to blur with. *)
let gaussian_huge_sigma sigma =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~substring:"sigma") (fun () ->
      gaussian_blur ~sigma img)

let test_median_blur () =
  let img = create_gray_f 5 5 0.5 in
  let filtered = median_blur ~ksize:3 img in
  check_shape "median blur shape" (Nx.shape img) filtered

let test_median_blur_preserves_median () =
  let data = [| 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0 |] in
  let img = Nx.create Nx.float32 [| 3; 3; 1 |] data in
  let filtered = median_blur ~ksize:3 img in
  check_pixel_f "median removes impulse noise" 0.0 filtered [ 1; 1; 0 ]

(* ───── Thresholding Tests ───── *)

let test_threshold () =
  let img =
    Nx.create Nx.float32 [| 2; 3; 1 |] [| 0.2; 0.4; 0.6; 0.8; 0.99; 0.1 |]
  in
  let binary = threshold 0.5 img in
  check_pixel_f "below threshold" 0.0 binary [ 0; 0; 0 ];
  check_pixel_f "above threshold" 1.0 binary [ 1; 0; 0 ]

(* ───── Morphological Operations Tests ───── *)

let test_structuring_elements () =
  let rect = structuring_element Rect (3, 5) in
  check_shape "rect shape" [| 3; 5 |] rect;
  check_pixel_i "rect filled" 1 rect [ 1; 2 ];
  let cross = structuring_element Cross (5, 5) in
  check_shape "cross shape" [| 5; 5 |] cross;
  check_pixel_i "cross center" 1 cross [ 2; 2 ];
  check_pixel_i "cross arm" 1 cross [ 2; 0 ];
  check_pixel_i "cross corner" 0 cross [ 0; 0 ]

let test_erosion () =
  let img = create_centered_square 10 10 4 in
  let kernel = structuring_element Rect (3, 3) in
  let eroded = erode ~kernel img in
  let white_count = ref 0 in
  for i = 0 to 9 do
    for j = 0 to 9 do
      if Nx.item [ i; j; 0 ] eroded > 0.5 then incr white_count
    done
  done;
  equal ~msg:"erosion reduces white area" int 4 !white_count;
  let center = Nx.item [ 4; 4; 0 ] eroded in
  if center < 0.5 then failf "erosion center not preserved: %.3f" center

let test_dilation () =
  let img = create_centered_square 10 10 4 in
  let kernel = structuring_element Rect (3, 3) in
  let dilated = dilate ~kernel img in
  let white_count = ref 0 in
  for i = 0 to 9 do
    for j = 0 to 9 do
      if Nx.item [ i; j; 0 ] dilated > 0.5 then incr white_count
    done
  done;
  equal ~msg:"dilation expands white area" int 36 !white_count

(* The extreme over a neighborhood counts the image's pixels only: a constant
   image erodes and dilates to itself at its borders too, whatever its sign. *)
let constant_morphology (type a b) (dt : (a, b) Nx.dtype) (t : a testable) v =
  let img = Nx.full dt [| 4; 5; 1 |] v in
  let kernel = structuring_element Rect (3, 3) in
  let expected = Array.make 20 v in
  equal ~msg:"dilate" (array t) expected (Nx.to_array (dilate ~kernel img));
  equal ~msg:"erode" (array t) expected (Nx.to_array (erode ~kernel img))

let test_dilation_kernel_shape () =
  let data = Array.make (5 * 5) 0.0 in
  data.((2 * 5) + 2) <- 1.0;
  let img = Nx.create Nx.float32 [| 5; 5; 1 |] data in
  let rect = structuring_element Rect (3, 3) in
  let cross = structuring_element Cross (3, 3) in
  let dilated_rect = dilate ~kernel:rect img in
  let dilated_cross = dilate ~kernel:cross img in
  let count_white tensor =
    let shape = Nx.shape tensor in
    let h = shape.(0) and w = shape.(1) in
    let total = ref 0 in
    for i = 0 to h - 1 do
      for j = 0 to w - 1 do
        if Nx.item [ i; j; 0 ] tensor > 0.5 then incr total
      done
    done;
    !total
  in
  equal ~msg:"rect kernel produces 3x3 block" int 9 (count_white dilated_rect);
  equal ~msg:"cross kernel preserves cross shape" int 5
    (count_white dilated_cross)

(* ───── Edge Detection Tests ───── *)

let test_sobel () =
  (* Vertical edge: left half black, right half white *)
  let img_data =
    Array.init 25 (fun idx ->
        let j = idx mod 5 in
        if j >= 2 then 1.0 else 0.0)
  in
  let img = Nx.create Nx.float32 [| 5; 5; 1 |] img_data in
  let gx, _gy = sobel img in
  check_shape "sobel shape" (Nx.shape img) gx;
  let edge_response = Float.abs (Nx.item [ 2; 2; 0 ] gx) in
  if edge_response < 0.1 then
    failf "Sobel X edge response too weak: %.3f" edge_response;
  (* Horizontal edge: top half black, bottom half white *)
  let img_h_data =
    Array.init 25 (fun idx ->
        let i = idx / 5 in
        if i >= 2 then 1.0 else 0.0)
  in
  let img_h = Nx.create Nx.float32 [| 5; 5; 1 |] img_h_data in
  let _gx, gy = sobel img_h in
  let edge_response_y = Float.abs (Nx.item [ 2; 2; 0 ] gy) in
  if edge_response_y < 0.1 then
    failf "Sobel Y edge response too weak: %.3f" edge_response_y

(* Correlating an impulse gives the kernel reversed along both axes: for a
   3x3 kernel, its row-major array reversed. *)
let impulse_response f =
  let n = 7 and c = 3 in
  let img =
    Nx.create Nx.float32 [| n; n; 1 |]
      (Array.init (n * n) (fun i -> if i = n * n / 2 then 1.0 else 0.0))
  in
  let around = [ Nx.R (c - 1, c + 2); Nx.R (c - 1, c + 2); Nx.I 0 ] in
  let r = Nx.to_array (Nx.slice around (f img)) in
  Array.init 9 (fun i -> r.(8 - i))

(* Kernels stated by OpenCV's Sobel and Laplacian documentation, which define
   ksize: the 3x3 Sobel kernels, ksize 1's unsmoothed [-1 0 1], and
   Laplacian ksize 1's aperture. *)
let edge_kernels =
  let gx k img = fst (sobel ~ksize:k img) in
  let gy k img = snd (sobel ~ksize:k img) in
  [
    ("sobel 3 gx", gx 3, [| -1.; 0.; 1.; -2.; 0.; 2.; -1.; 0.; 1. |]);
    ("sobel 3 gy", gy 3, [| -1.; -2.; -1.; 0.; 0.; 0.; 1.; 2.; 1. |]);
    ("sobel 1 gx", gx 1, [| 0.; 0.; 0.; -1.; 0.; 1.; 0.; 0.; 0. |]);
    ("sobel 1 gy", gy 1, [| 0.; -1.; 0.; 0.; 0.; 0.; 0.; 1.; 0. |]);
    ( "laplacian 1",
      laplacian ~ksize:1,
      [| 0.; 1.; 0.; 1.; -4.; 1.; 0.; 1.; 0. |] );
    ( "laplacian 3",
      laplacian ~ksize:3,
      [| 2.; 0.; 2.; 0.; -8.; 0.; 2.; 0.; 2. |] );
    ( "laplacian default",
      (fun img -> laplacian img),
      [| 0.; 1.; 0.; 1.; -4.; 1.; 0.; 1.; 0. |] );
  ]

let edge_kernel (_, f, kernel) =
  equal (array (float 1e-6)) kernel (impulse_response f)

(* A Sobel kernel of aperture [k >= 3] for derivative orders [dx], [dy] sums
   to the derivative times [2 ^ (2k - dx - dy - 2)], the denominator OpenCV's
   getDerivKernels states. Aperture 1 only differentiates: [-1 0 1] gives
   twice a first derivative and [1 -2 1] a second one. So on [a x + b y],
   interior gradients are [a] and [b] times [ramp_gain k], and on
   [x^2 + y^2] the interior Laplacian is [4 * square_gain k]. *)
let ramp_gain k = if k = 1 then 2.0 else Float.ldexp 1.0 ((2 * k) - 3)
let square_gain k = if k = 1 then 1.0 else Float.ldexp 1.0 ((2 * k) - 4)

(* The pixels of an [n x n] image far enough from the border that zero
   padding does not reach the window of aperture [k], which spans 3 pixels
   for [k = 1]. *)
let interior n k a =
  let r = max 1 (k / 2) in
  let a = Nx.reshape [| n; n |] a in
  Nx.to_array (Nx.slice [ Nx.R (r, n - r); Nx.R (r, n - r) ] a)

let constant a v = Array.make (Array.length a) v

let polynomial n f =
  let c = n / 2 in
  Nx.create Nx.float32 [| n; n; 1 |]
    (Array.init (n * n) (fun i ->
         f (Float.of_int ((i mod n) - c)) (Float.of_int ((i / n) - c))))

let sobel_ramp k =
  let n = 15 and a = 0.5 and b = -0.25 in
  let ramp = polynomial n (fun x y -> (a *. x) +. (b *. y)) in
  let gx, gy = sobel ~ksize:k ramp in
  let gx = interior n k gx and gy = interior n k gy in
  let g = ramp_gain k in
  equal ~msg:"gx" (array (float 1e-3)) (constant gx (a *. g)) gx;
  equal ~msg:"gy" (array (float 1e-3)) (constant gy (b *. g)) gy

let laplacian_square k =
  let n = 15 in
  let square = polynomial n (fun x y -> (x *. x) +. (y *. y)) in
  let lap = interior n k (laplacian ~ksize:k square) in
  let v = 4.0 *. square_gain k in
  equal (array (float (1e-4 *. v))) (constant lap v) lap

let sobel_bad_ksize ksize =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~substring:"ksize") (fun () -> sobel ~ksize img)

let laplacian_bad_ksize ksize =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~substring:"ksize") (fun () ->
      laplacian ~ksize img)

let test_canny () =
  let img = create_centered_square 20 20 10 in
  let edges = canny ~low:0.2 ~high:0.6 img in
  check_shape "canny shape" (Nx.shape img) edges;
  let edge_count = ref 0 in
  for i = 0 to 19 do
    for j = 0 to 19 do
      if Nx.item [ i; j; 0 ] edges > 0.5 then incr edge_count
    done
  done;
  if !edge_count = 0 then fail "Canny detected no edges";
  if !edge_count > 100 then
    failf "Canny detected too many edges: %d" !edge_count

(* ───── Integration Tests ───── *)

let test_pipeline () =
  let img = create_centered_square 20 20 8 in
  let blurred = gaussian_blur ~sigma:1.5 ~ksize:5 img in
  let binary = threshold 0.5 blurred in
  let kernel = structuring_element Rect (3, 3) in
  let cleaned = erode ~kernel binary in
  let final = dilate ~kernel cleaned in
  check_shape "pipeline preserves shape" (Nx.shape img) final;
  let white_count = ref 0 in
  for i = 0 to 19 do
    for j = 0 to 19 do
      if Nx.item [ i; j; 0 ] final > 0.5 then incr white_count
    done
  done;
  if !white_count = 0 then fail "Pipeline eliminated all features"

(* ───── Device Sets ───── *)

(* A set of one memory device, computed by nx.cpu as the host is: an operation
   on an image there computes there, its kernels and constants beside the image,
   and gives the host's elements. *)
let memory n = Result.get_ok (Rig.memory_device n)

module Memory = (val Nx.devices [ memory "m0" ])

(* Two memory devices, over which a batch is cut: a kernel made at the image's
   placement itself, and not whole on each device, would be cut with it. *)
module Two = (val Nx.devices [ memory "m1"; memory "m2" ])

let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

type op = { op : 'd. 'd Nx.float32_t -> 'd Nx.float32_t }

let gray x = to_grayscale x
let at_image x k = Nx.place (Nx.beside x) k
let disk = structuring_element Ellipse (3, 3)

let sharpen =
  Nx.create Nx.float32 [| 3; 3 |] [| 0.; -1.; 0.; -1.; 5.; -1.; 0.; -1.; 0. |]

let set_ops =
  [
    ("to_float of to_uint8", { op = (fun x -> to_float (to_uint8 x)) });
    ( "normalize",
      {
        op =
          (fun x -> normalize ~mean:[ 0.5; 0.4; 0.3 ] ~std:[ 0.2; 0.3; 0.4 ] x);
      } );
    ("threshold", { op = (fun x -> threshold 0.5 x) });
    ("to_grayscale", { op = gray });
    ("hsv_to_rgb of rgb_to_hsv", { op = (fun x -> hsv_to_rgb (rgb_to_hsv x)) });
    ("adjust_contrast", { op = (fun x -> adjust_contrast 0.5 x) });
    ("gaussian_blur", { op = (fun x -> gaussian_blur ~sigma:1.2 x) });
    ( "gaussian_blur of a fixed table",
      { op = (fun x -> gaussian_blur ~sigma:0. ~ksize:5 x) } );
    ( "gaussian_blur of a derived sigma",
      { op = (fun x -> gaussian_blur ~sigma:0. ~ksize:11 x) } );
    ("box_blur", { op = (fun x -> box_blur ~ksize:3 x) });
    ("median_blur", { op = (fun x -> median_blur ~ksize:3 x) });
    ("filter2d", { op = (fun x -> filter2d (at_image x sharpen) x) });
    ("unsharp_mask", { op = (fun x -> unsharp_mask ~sigma:1. x) });
    ( "resize nearest",
      { op = (fun x -> resize ~interpolation:Nearest ~height:3 ~width:9 x) } );
    ("resize bilinear", { op = (fun x -> resize ~height:3 ~width:9 x) });
    ("resize to one row", { op = (fun x -> resize ~height:1 ~width:4 x) });
    ("pad", { op = (fun x -> pad ~value:0.5 (1, 0, 2, 1) x) });
    ("erode", { op = (fun x -> erode ~kernel:disk x) });
    ( "dilate by a kernel of another set",
      { op = (fun x -> dilate ~kernel:(Nx.place Memory.on disk) x) } );
    ("opening", { op = (fun x -> opening ~kernel:disk x) });
    ("closing", { op = (fun x -> closing ~kernel:disk x) });
    ( "morphological_gradient",
      { op = (fun x -> morphological_gradient ~kernel:disk x) } );
    ("sobel x", { op = (fun x -> fst (sobel (gray x))) });
    ("sobel y", { op = (fun x -> snd (sobel ~ksize:5 (gray x))) });
    ("scharr x", { op = (fun x -> fst (scharr (gray x))) });
    ("laplacian", { op = (fun x -> laplacian ~ksize:3 (gray x)) });
    ("canny", { op = (fun x -> canny ~low:0.1 ~high:0.3 (gray x)) });
  ]

let batch () = Nx.Rng.uniform ~key:(Nx.Rng.key 3) Nx.float32 [| 2; 5; 6; 3 |]

let on_another_set (_, { op }) =
  let img = batch () in
  let host = op img and there = op (Nx.place Memory.on img) in
  equal (placement ()) Memory.on (Nx.placement there);
  equal (array int) (Nx.shape host) (Nx.shape there);
  equal (array float_exact) (Nx.to_array host) (Nx.to_array there)

(* A batch cut along its images over two devices stays cut so, each device
   filtering its own images. A reduction over the cut axis sums by partials,
   in another order than the host's: elements agree within rounding. *)
let cut_over_two (_, { op }) =
  let img = batch () in
  let cut = Two.split ~axis:0 in
  let host = op img and there = op (Nx.place cut img) in
  equal (placement ()) cut (Nx.placement there);
  equal (array int) (Nx.shape host) (Nx.shape there);
  equal (array (float 1e-6)) (Nx.to_array host) (Nx.to_array there)

(* ───── Test Suite ───── *)

let () =
  exit (run "Sowilo"
    [
      group "transforms"
        [
          test "vflip" test_flip_vertical;
          test "hflip" test_flip_horizontal;
          test "crop" test_crop;
          test "flip_batch" test_flip_batch;
          test "crop_batch" test_crop_batch;
          test "resize_nearest" test_resize_nearest;
          test "resize_bilinear" test_resize_bilinear;
          test "resize_batch" test_resize_batch;
          test "resize_color_bilinear" test_resize_color_bilinear;
        ];
      group "color"
        [
          test "to_grayscale" test_to_grayscale;
          cases "to_grayscale_channels"
            ~name:(Printf.sprintf "C = %d")
            [ 3; 4; 5 ] grayscale_extra_channels;
          cases "to_grayscale_few_channels"
            ~name:(Printf.sprintf "C = %d")
            [ 1; 2 ] grayscale_few_channels;
          test "to_grayscale_rank_2" test_grayscale_rank_2;
        ];
      group "type_conversion"
        [ test "float_conversions" test_float_conversions ];
      group "filtering"
        [
          test "gaussian_blur" test_gaussian_blur;
          test "box_blur" test_box_blur;
          test "box_blur_even" test_box_blur_even;
          test "box_blur_larger_than_image" test_box_blur_larger_than_image;
          test "unsharp_mask_larger_kernel"
            test_unsharp_mask_kernel_larger_than_image;
          cases "gaussian_blur_default_ksize"
            ~name:(fun (sigma, k) -> Printf.sprintf "sigma %g: %d taps" sigma k)
            [ (0.4, 5); (1.0, 7); (1.1, 9); (2.0, 13) ]
            gaussian_default_ksize;
          cases "gaussian_blur_bad_ksize" ~name:string_of_int [ 0; -3; 2; 4 ]
            gaussian_bad_ksize;
          cases "gaussian_blur_table"
            ~name:(fun (sigma, taps, _) ->
              Printf.sprintf "sigma %g, ksize %d" sigma (Array.length taps))
            gaussian_tables gaussian_table;
          cases "gaussian_blur_derived_sigma"
            ~name:(fun (sigma, k) ->
              Printf.sprintf "sigma %g, ksize %d" sigma k)
            [ (0.0, 11); (-1.0, 13); (-0.0, 15) ]
            gaussian_derived_sigma;
          cases "gaussian_blur_no_sigma" ~name:string_of_float
            [ 0.0; -0.0; -2.5; Float.nan ] gaussian_no_sigma;
          cases "gaussian_blur_nan_sigma" ~name:string_of_int [ 1; 3; 11 ]
            gaussian_nan_sigma;
          cases "gaussian_blur_huge_sigma" ~name:string_of_float
            [ infinity; 1e300; Float.of_int max_int ]
            gaussian_huge_sigma;
          test "median_blur" test_median_blur;
          test "median_blur_median" test_median_blur_preserves_median;
        ];
      group "thresholding" [ test "threshold" test_threshold ];
      group "morphology"
        [
          test "structuring_elements" test_structuring_elements;
          test "erosion" test_erosion;
          test "dilation" test_dilation;
          cases "constant_int8" ~name:string_of_int [ -128; -5; 0; 127 ]
            (constant_morphology Nx.int8 int);
          cases "constant_float32" ~name:string_of_float
            [ Float.neg_infinity; -0.5; -0.; 1.; Float.infinity ]
            (constant_morphology Nx.float32 float_exact);
          test "dilation_kernel_shape" test_dilation_kernel_shape;
        ];
      group "edge_detection"
        [
          test "sobel" test_sobel;
          cases "kernels" ~name:(fun (n, _, _) -> n) edge_kernels edge_kernel;
          cases "sobel_ramp" ~name:(Printf.sprintf "ksize %d") [ 1; 3; 5; 7 ]
            sobel_ramp;
          cases "laplacian_square" ~name:(Printf.sprintf "ksize %d")
            [ 1; 3; 5; 7; 9; 11 ] laplacian_square;
          cases "sobel_bad_ksize" ~name:string_of_int [ -1; 0; 2; 9 ]
            sobel_bad_ksize;
          cases "laplacian_bad_ksize" ~name:string_of_int [ -1; 0; 2; 10 ]
            laplacian_bad_ksize;
          slow "canny" test_canny;
        ];
      group "integration" [ test "pipeline" test_pipeline ];
      group "device_sets"
        [
          cases "an operation computes on its image's set"
            ~name:(fun (n, _) -> n)
            set_ops on_another_set;
          cases "an operation computes on a batch cut over two devices"
            ~name:(fun (n, _) -> n)
            set_ops cut_over_two;
        ];
    ])
