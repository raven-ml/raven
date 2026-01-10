(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Comprehensive tests for Sowilo image processing library *)

open Alcotest
open Sowilo

(* Helper functions *)

let create_test_image_gray h w value = Rune.full Rune.uint8 [| h; w |] value

let create_checkerboard h w =
  let data =
    Array.init (h * w) (fun i ->
        let row = i / w in
        let col = i mod w in
        if (row + col) mod 2 = 0 then 255 else 0)
  in
  Rune.create Rune.uint8 [| h; w |] data

let create_centered_square h w square_size =
  let data =
    Array.init (h * w) (fun idx ->
        let i = idx / w in
        let j = idx mod w in
        let start_h = (h - square_size) / 2 in
        let start_w = (w - square_size) / 2 in
        if
          i >= start_h
          && i < start_h + square_size
          && j >= start_w
          && j < start_w + square_size
        then 255
        else 0)
  in
  Rune.create Rune.uint8 [| h; w |] data

let check_tensor msg expected actual =
  if Rune.shape expected <> Rune.shape actual then
    failf "%s: shapes differ - expected %s, got %s" msg
      (String.concat "x"
         (List.map string_of_int (Array.to_list (Rune.shape expected))))
      (String.concat "x"
         (List.map string_of_int (Array.to_list (Rune.shape actual))))
  else
    (* Compare all elements *)
    let total_elements = Array.fold_left ( * ) 1 (Rune.shape expected) in
    let expected_flat = Rune.reshape [| total_elements |] expected in
    let actual_flat = Rune.reshape [| total_elements |] actual in
    for i = 0 to total_elements - 1 do
      let e = Rune.item [ i ] expected_flat in
      let a = Rune.item [ i ] actual_flat in
      if e <> a then
        failf "%s: values differ at flat index %d - expected %d, got %d" msg i e
          a
    done

let check_shape msg expected_shape tensor =
  check (array int) msg expected_shape (Rune.shape tensor)

let check_pixel msg expected tensor indices =
  let actual = Rune.item indices tensor in
  check int msg expected actual

(* ───── Basic Image Manipulation Tests ───── *)

let test_flip_vertical () =
  let img = create_checkerboard 4 4 in
  let flipped = flip_vertical img in

  (* Check that first row becomes last row *)
  check_pixel "top-left after flip" (Rune.item [ 3; 0 ] img) flipped [ 0; 0 ];
  check_pixel "top-right after flip" (Rune.item [ 3; 3 ] img) flipped [ 0; 3 ];
  check_shape "flip preserves shape" (Rune.shape img) flipped

let test_flip_horizontal () =
  let img = create_checkerboard 4 4 in
  let flipped = flip_horizontal img in

  (* Check that first column becomes last column *)
  check_pixel "top-left after flip" (Rune.item [ 0; 3 ] img) flipped [ 0; 0 ];
  check_pixel "bottom-left after flip" (Rune.item [ 3; 3 ] img) flipped [ 3; 0 ];
  check_shape "flip preserves shape" (Rune.shape img) flipped

let test_flip_batch () =
  let data = [| 1; 2; 3; 4; 5; 6; 7; 8 |] in
  let img = Rune.create Rune.uint8 [| 2; 2; 2; 1 |] data in
  let flipped_v = flip_vertical img in
  check_shape "vertical batch shape" [| 2; 2; 2; 1 |] flipped_v;
  check_pixel "batch 0 vertical flip" 3 flipped_v [ 0; 0; 0; 0 ];
  check_pixel "batch 1 vertical flip" 7 flipped_v [ 1; 0; 0; 0 ];
  let flipped_h = flip_horizontal img in
  check_pixel "batch 0 horizontal flip" 2 flipped_h [ 0; 0; 0; 0 ];
  check_pixel "batch 1 horizontal flip" 6 flipped_h [ 1; 0; 0; 0 ]

let test_crop () =
  let img = create_checkerboard 10 10 in
  let cropped = crop ~y:2 ~x:3 ~height:5 ~width:4 img in

  check_shape "crop shape" [| 5; 4 |] cropped;
  check_pixel "crop content" (Rune.item [ 2; 3 ] img) cropped [ 0; 0 ];

  (* Test invalid crop *)
  check_raises "crop out of bounds"
    (Invalid_argument
       "Invalid crop parameters: y=8, x=8, h=5, w=5 for image [10x10]")
    (fun () -> ignore (crop ~y:8 ~x:8 ~height:5 ~width:5 img))

let test_crop_batch () =
  let data = Array.init (2 * 4 * 4) (fun i -> i) in
  let img = Rune.create Rune.uint8 [| 2; 4; 4; 1 |] data in
  let cropped = crop ~y:1 ~x:1 ~height:2 ~width:2 img in
  check_shape "batch crop shape" [| 2; 2; 2; 1 |] cropped;
  check_pixel "batch crop value"
    (Rune.item [ 0; 1; 1; 0 ] img)
    cropped [ 0; 0; 0; 0 ];
  check_pixel "batch crop second batch"
    (Rune.item [ 1; 2; 2; 0 ] img)
    cropped [ 1; 1; 1; 0 ]

let test_resize_nearest () =
  let img = Rune.create Rune.uint8 [| 2; 2 |] [| 10; 20; 30; 40 |] in
  let resized = resize ~height:4 ~width:4 img in
  check_shape "resize nearest shape" [| 4; 4 |] resized;
  check_pixel "nearest top-left" 10 resized [ 0; 0 ];
  check_pixel "nearest top-right" 20 resized [ 0; 3 ];
  check_pixel "nearest bottom-left" 30 resized [ 3; 0 ];
  check_pixel "nearest bottom-right" 40 resized [ 3; 3 ]

let test_resize_linear () =
  let img = Rune.create Rune.uint8 [| 2; 2 |] [| 0; 255; 0; 255 |] in
  let resized = resize ~interpolation:Linear ~height:3 ~width:3 img in
  check_shape "resize linear shape" [| 3; 3 |] resized;
  check_pixel "linear left edge" 0 resized [ 0; 0 ];
  check_pixel "linear right edge" 255 resized [ 0; 2 ];
  let center = Rune.item [ 1; 1 ] resized in
  if center <= 120 || center >= 140 then
    failf "Linear resize center expected ~128, got %d" center

let test_resize_batch () =
  let img =
    Rune.create Rune.uint8 [| 2; 2; 2 |] [| 10; 20; 30; 40; 50; 60; 70; 80 |]
  in
  let resized = resize ~height:4 ~width:4 img in
  check_shape "resize batch shape" [| 2; 4; 4 |] resized;
  check_pixel "batch0 top-left" 10 resized [ 0; 0; 0 ];
  check_pixel "batch1 bottom-right" 80 resized [ 1; 3; 3 ]

let test_resize_color_linear () =
  let img =
    Rune.create Rune.uint8 [| 1; 2; 2; 3 |]
      [| 0; 0; 0; 255; 0; 0; 0; 255; 0; 255; 255; 0 |]
  in
  let resized = resize ~interpolation:Linear ~height:3 ~width:3 img in
  check_shape "resize color shape" [| 1; 3; 3; 3 |] resized;
  let center_r = Rune.item [ 0; 1; 1; 0 ] resized in
  let center_g = Rune.item [ 0; 1; 1; 1 ] resized in
  if center_r < 120 || center_r > 140 then
    failf "Color linear resize R expected ~128, got %d" center_r;
  if center_g < 120 || center_g > 140 then
    failf "Color linear resize G expected ~128, got %d" center_g;
  check_pixel "corner preserves blue" 0 resized [ 0; 0; 0; 2 ]

(* ───── Color Conversion Tests ───── *)

let test_to_grayscale () =
  (* Test RGB to grayscale *)
  let rgb =
    Rune.create Rune.uint8 [| 2; 2; 3 |]
      [|
        255;
        0;
        0;
        (* Red *)
        0;
        255;
        0;
        (* Green *)
        0;
        0;
        255;
        (* Blue *)
        255;
        255;
        255 (* White *);
      |]
  in
  let gray = to_grayscale rgb in

  check_shape "grayscale shape" [| 2; 2 |] gray;

  (* Check conversion values (approximate due to rounding) *)
  let check_gray_value msg expected actual =
    let diff = abs (expected - actual) in
    if diff > 1 then failf "%s: expected ~%d, got %d" msg expected actual
  in

  check_gray_value "red to gray" 76 (Rune.item [ 0; 0 ] gray);
  check_gray_value "green to gray" 150 (Rune.item [ 0; 1 ] gray);
  check_gray_value "blue to gray" 29 (Rune.item [ 1; 0 ] gray);
  check_gray_value "white to gray" 255 (Rune.item [ 1; 1 ] gray);

  (* Test grayscale passthrough *)
  let gray_img = create_test_image_gray 5 5 128 in
  let gray_result = to_grayscale gray_img in
  check_shape "grayscale passthrough shape" [| 5; 5 |] gray_result;
  (* Should be identical for grayscale input *)
  check_pixel "grayscale passthrough value" 128 gray_result [ 2; 2 ]

let test_rgb_bgr_conversion () =
  let rgb = Rune.create Rune.uint8 [| 1; 1; 3 |] [| 10; 20; 30 |] in
  let bgr = rgb_to_bgr rgb in

  check_pixel "R becomes B" 30 bgr [ 0; 0; 0 ];
  check_pixel "G stays G" 20 bgr [ 0; 0; 1 ];
  check_pixel "B becomes R" 10 bgr [ 0; 0; 2 ];

  (* Test double swap returns original *)
  let rgb2 = bgr_to_rgb bgr in
  check_tensor "double swap" rgb rgb2

(* ───── Data Type Conversion Tests ───── *)

let test_float_conversions () =
  let uint8_img = create_test_image_gray 2 2 255 in
  let float_img = to_float uint8_img in

  (* Check normalized to [0, 1] *)
  let float_val = Rune.item [ 0; 0 ] float_img in
  check (float 0.001) "to_float normalization" 1.0 float_val;

  (* Convert back *)
  let uint8_back = to_uint8 float_img in
  check_shape "round-trip shape" [| 2; 2 |] uint8_back;
  check_pixel "round-trip value" 255 uint8_back [ 0; 0 ];

  (* Test clipping *)
  let out_of_range =
    Rune.create Rune.float32 [| 2; 2 |] [| -0.5; 0.5; 1.5; 0.75 |]
  in
  let clipped = to_uint8 out_of_range in
  check_pixel "clipped negative" 0 clipped [ 0; 0 ];
  check_pixel "clipped middle" 127 clipped [ 0; 1 ];
  check_pixel "clipped overflow" 255 clipped [ 1; 0 ];
  check_pixel "clipped normal" 191 clipped [ 1; 1 ]

(* ───── Filtering Tests ───── *)

let test_gaussian_blur () =
  (* Create image with sharp edge *)
  let img = create_centered_square 10 10 4 in
  let blurred = gaussian_blur ~ksize:(3, 3) ~sigmaX:1.0 img in

  check_shape "blur preserves shape" (Rune.shape img) blurred;

  (* Check that edges are smoothed (values between 0 and 255) *)
  let edge_val = Rune.item [ 3; 3 ] blurred in
  if edge_val = 0 || edge_val = 255 then
    failf "Edge not smoothed: got %d" edge_val;

  (* Test separable sigmas *)
  let blurred_xy = gaussian_blur ~ksize:(5, 3) ~sigmaX:2.0 ~sigmaY:1.0 img in
  check_shape "asymmetric blur shape" (Rune.shape img) blurred_xy

let test_box_filter () =
  (* Create simple test pattern *)
  let img =
    Rune.create Rune.uint8 [| 3; 3 |] [| 0; 0; 0; 0; 255; 0; 0; 0; 0 |]
  in
  let filtered = box_filter ~ksize:(3, 3) img in

  (* Center pixel should be average of 9 pixels = 255/9 ≈ 28 *)
  let center = Rune.item [ 1; 1 ] filtered in
  let expected = 255 / 9 in
  if abs (center - expected) > 1 then
    failf "Box filter center: expected ~%d, got %d" expected center

let test_median_blur () =
  (* Note: Current implementation approximates with box filter *)
  let img = create_test_image_gray 5 5 128 in
  let filtered = median_blur ~ksize:3 img in
  check_shape "median blur shape" (Rune.shape img) filtered

let test_median_blur_preserves_median () =
  let img =
    Rune.create Rune.uint8 [| 3; 3 |] [| 0; 0; 0; 0; 255; 0; 0; 0; 0 |]
  in
  let filtered = median_blur ~ksize:3 img in
  check_pixel "median removes impulse noise" 0 filtered [ 1; 1 ]

(* ───── Thresholding Tests ───── *)

let test_threshold () =
  let img =
    Rune.create Rune.uint8 [| 2; 3 |] [| 50; 100; 150; 200; 250; 25 |]
  in

  (* Binary threshold *)
  let binary = threshold ~thresh:128 ~maxval:255 ~type_:Binary img in
  check_pixel "binary below" 0 binary [ 0; 0 ];
  check_pixel "binary above" 255 binary [ 1; 0 ];

  (* Binary inverse *)
  let binary_inv = threshold ~thresh:128 ~maxval:255 ~type_:BinaryInv img in
  check_pixel "binary_inv below" 255 binary_inv [ 0; 0 ];
  check_pixel "binary_inv above" 0 binary_inv [ 1; 0 ];

  (* Truncate *)
  let trunc = threshold ~thresh:128 ~maxval:255 ~type_:Trunc img in
  check_pixel "trunc below" 50 trunc [ 0; 0 ];
  check_pixel "trunc above" 128 trunc [ 1; 0 ];

  (* ToZero *)
  let to_zero = threshold ~thresh:128 ~maxval:255 ~type_:ToZero img in
  check_pixel "to_zero below" 0 to_zero [ 0; 0 ];
  check_pixel "to_zero above" 200 to_zero [ 1; 0 ]

(* ───── Morphological Operations Tests ───── *)

let test_structuring_elements () =
  (* Rectangle *)
  let rect = get_structuring_element ~shape:Rect ~ksize:(3, 5) in
  check_shape "rect shape" [| 3; 5 |] rect;
  check_pixel "rect filled" 1 rect [ 1; 2 ];

  (* Cross *)
  let cross = get_structuring_element ~shape:Cross ~ksize:(5, 5) in
  check_shape "cross shape" [| 5; 5 |] cross;
  check_pixel "cross center" 1 cross [ 2; 2 ];
  check_pixel "cross arm" 1 cross [ 2; 0 ];
  check_pixel "cross corner" 0 cross [ 0; 0 ]

let test_erosion () =
  (* Create 4x4 white square in 10x10 image *)
  let img = create_centered_square 10 10 4 in
  let kernel = get_structuring_element ~shape:Rect ~ksize:(3, 3) in
  let eroded = erode ~kernel img in

  (* Count white pixels - should be 2x2 = 4 *)
  let white_count = ref 0 in
  for i = 0 to 9 do
    for j = 0 to 9 do
      if Rune.item [ i; j ] eroded = 255 then incr white_count
    done
  done;
  check int "erosion reduces white area" 4 !white_count;

  (* Check center is preserved *)
  check_pixel "erosion center preserved" 255 eroded [ 4; 4 ]

let test_dilation () =
  (* Create 4x4 white square in 10x10 image *)
  let img = create_centered_square 10 10 4 in
  let kernel = get_structuring_element ~shape:Rect ~ksize:(3, 3) in
  let dilated = dilate ~kernel img in

  (* Count white pixels - should be 6x6 = 36 *)
  let white_count = ref 0 in
  for i = 0 to 9 do
    for j = 0 to 9 do
      if Rune.item [ i; j ] dilated = 255 then incr white_count
    done
  done;
  check int "dilation expands white area" 36 !white_count

let test_dilation_kernel_shape () =
  let data = Array.make (5 * 5) 0 in
  data.((2 * 5) + 2) <- 255;
  let img = Rune.create Rune.uint8 [| 5; 5 |] data in
  let rect = get_structuring_element ~shape:Rect ~ksize:(3, 3) in
  let cross = get_structuring_element ~shape:Cross ~ksize:(3, 3) in
  let dilated_rect = dilate ~kernel:rect img in
  let dilated_cross = dilate ~kernel:cross img in
  let count_white tensor =
    let h, w =
      match Rune.shape tensor with
      | [| h; w |] -> (h, w)
      | _ -> failwith "Unexpected shape in kernel shape test"
    in
    let total = ref 0 in
    for i = 0 to h - 1 do
      for j = 0 to w - 1 do
        if Rune.item [ i; j ] tensor = 255 then incr total
      done
    done;
    !total
  in
  check int "rect kernel produces 3x3 block" 9 (count_white dilated_rect);
  check int "cross kernel preserves cross shape" 5 (count_white dilated_cross)

(* ───── Edge Detection Tests ───── *)

let test_sobel () =
  (* Create image with vertical edge - left half black, right half white *)
  let img_data =
    Array.init 25 (fun idx ->
        let j = idx mod 5 in
        if j >= 2 then 255 else 0)
  in
  let img = Rune.create Rune.uint8 [| 5; 5 |] img_data in

  (* Sobel X should detect vertical edges *)
  let sobel_x = sobel ~dx:1 ~dy:0 img in
  check_shape "sobel shape" (Rune.shape img) sobel_x;

  (* Check edge detection at boundary *)
  let edge_response = abs (Rune.item [ 2; 2 ] sobel_x) in
  if edge_response < 100 then
    failf "Sobel X edge response too weak: %d" edge_response;

  (* Sobel Y should detect horizontal edges - top half black, bottom half
     white *)
  let img_h_data =
    Array.init 25 (fun idx ->
        let i = idx / 5 in
        if i >= 2 then 255 else 0)
  in
  let img_h = Rune.create Rune.uint8 [| 5; 5 |] img_h_data in

  let sobel_y = sobel ~dx:0 ~dy:1 img_h in
  let edge_response_y = abs (Rune.item [ 2; 2 ] sobel_y) in
  if edge_response_y < 100 then
    failf "Sobel Y edge response too weak: %d" edge_response_y

let test_canny () =
  (* Create image with clear edges *)
  let img = create_centered_square 20 20 10 in
  let edges = canny ~threshold1:50.0 ~threshold2:150.0 img in

  check_shape "canny shape" (Rune.shape img) edges;

  (* Check that we have edge pixels *)
  let edge_count = ref 0 in
  for i = 0 to 19 do
    for j = 0 to 19 do
      if Rune.item [ i; j ] edges = 255 then incr edge_count
    done
  done;

  if !edge_count = 0 then fail "Canny detected no edges";
  if !edge_count > 100 then
    failf "Canny detected too many edges: %d" !edge_count

(* ───── Integration Tests ───── *)

let test_pipeline () =
  (* Test a typical image processing pipeline *)
  let img = create_centered_square 20 20 8 in

  (* Apply Gaussian blur *)
  let blurred = gaussian_blur ~ksize:(5, 5) ~sigmaX:1.5 img in

  (* Threshold *)
  let binary = threshold ~thresh:128 ~maxval:255 ~type_:Binary blurred in

  (* Morphological operations *)
  let kernel = get_structuring_element ~shape:Rect ~ksize:(3, 3) in
  let cleaned = erode ~kernel binary in
  let final = dilate ~kernel cleaned in

  check_shape "pipeline preserves shape" (Rune.shape img) final;

  (* Check we still have some white pixels *)
  let white_count = ref 0 in
  for i = 0 to 19 do
    for j = 0 to 19 do
      if Rune.item [ i; j ] final = 255 then incr white_count
    done
  done;

  if !white_count = 0 then fail "Pipeline eliminated all features"

(* ───── Test Suite ───── *)

let () =
  let open Alcotest in
  run "Sowilo"
    [
      ( "image_manipulation",
        [
          test_case "flip_vertical" `Quick test_flip_vertical;
          test_case "flip_horizontal" `Quick test_flip_horizontal;
          test_case "crop" `Quick test_crop;
          test_case "flip_batch" `Quick test_flip_batch;
          test_case "crop_batch" `Quick test_crop_batch;
          test_case "resize_nearest" `Quick test_resize_nearest;
          test_case "resize_linear" `Quick test_resize_linear;
          test_case "resize_batch" `Quick test_resize_batch;
          test_case "resize_color_linear" `Quick test_resize_color_linear;
        ] );
      ( "color_conversion",
        [
          test_case "to_grayscale" `Quick test_to_grayscale;
          test_case "rgb_bgr_conversion" `Quick test_rgb_bgr_conversion;
        ] );
      ( "type_conversion",
        [ test_case "float_conversions" `Quick test_float_conversions ] );
      ( "filtering",
        [
          test_case "gaussian_blur" `Quick test_gaussian_blur;
          test_case "box_filter" `Quick test_box_filter;
          test_case "median_blur" `Quick test_median_blur;
          test_case "median_blur_median" `Quick
            test_median_blur_preserves_median;
        ] );
      ("thresholding", [ test_case "threshold" `Quick test_threshold ]);
      ( "morphology",
        [
          test_case "structuring_elements" `Quick test_structuring_elements;
          test_case "erosion" `Quick test_erosion;
          test_case "dilation" `Quick test_dilation;
          test_case "dilation_kernel_shape" `Quick test_dilation_kernel_shape;
        ] );
      ( "edge_detection",
        [
          test_case "sobel" `Quick test_sobel;
          test_case "canny" `Slow test_canny;
        ] );
      ("integration", [ test_case "pipeline" `Quick test_pipeline ]);
    ]
