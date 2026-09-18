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

(* ───── Test Suite ───── *)

let () =
  run "Sowilo"
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
      group "color" [ test "to_grayscale" test_to_grayscale ];
      group "type_conversion"
        [ test "float_conversions" test_float_conversions ];
      group "filtering"
        [
          test "gaussian_blur" test_gaussian_blur;
          test "box_blur" test_box_blur;
          test "median_blur" test_median_blur;
          test "median_blur_median" test_median_blur_preserves_median;
        ];
      group "thresholding" [ test "threshold" test_threshold ];
      group "morphology"
        [
          test "structuring_elements" test_structuring_elements;
          test "erosion" test_erosion;
          test "dilation" test_dilation;
          test "dilation_kernel_shape" test_dilation_kernel_shape;
        ];
      group "edge_detection"
        [ test "sobel" test_sobel; slow "canny" test_canny ];
      group "integration" [ test "pipeline" test_pipeline ];
    ]
