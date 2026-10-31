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

let check_pixel_i msg expected tensor indices =
  let actual = Nx.item indices tensor in
  equal ~msg int expected actual

let check_pixel_f msg expected tensor indices =
  let actual = Nx.item indices tensor in
  let diff = Float.abs (expected -. actual) in
  if diff > 0.01 then failf "%s: expected ~%.3f, got %.3f" msg expected actual

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
  raises_match (Exn.invalid_arg ~sub:"channels") (fun () ->
      to_grayscale img)

(* A rank-2 array has no channel axis, whatever its last dimension. *)
let test_grayscale_rank_2 () =
  let img = Nx.zeros Nx.float32 [| 2; 3 |] in
  raises_match (Exn.invalid_arg ~sub:"rank") (fun () -> to_grayscale img)

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
  raises_match (Exn.invalid_arg ~sub:"ksize") (fun () ->
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
  raises_match (Exn.invalid_arg ~sub:"sigma") (fun () ->
      gaussian_blur ~sigma img)

(* A NaN sigma is neither positive nor 0 or less: nothing derives a kernel
   from it. *)
let gaussian_nan_sigma ksize =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~sub:"sigma") (fun () ->
      gaussian_blur ~sigma:Float.nan ~ksize img)

(* A sigma whose default ksize [2 * ceil(3 * sigma) + 1] exceeds [max_int]
   has no kernel to blur with. *)
let gaussian_huge_sigma sigma =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~sub:"sigma") (fun () ->
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

(* [f32 x] is the float32 nearest [x]. [f32_step x n] is the float32 whose bit
   pattern is [n] above the float32 [x]'s: [n] float32s away from zero, toward
   it for a negative [n]. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let f32_step x n = Int32.float_of_bits (Int32.add (Int32.bits_of_float x) n)

(* [f32_below x] is the float32 next below the float32 [x], for [x] other
   than [0.]. *)
let f32_below x = f32_step x (if x > 0. then -1l else 1l)
let row xs = Nx.create Nx.float32 [| 1; Array.length xs; 1 |] xs

(* Each element is [true] iff it is greater than [t] stored in float32: an
   element equal to that is not, nor a NaN element, nor any element for a NaN
   [t]. *)
let threshold_cases =
  let half = 0.5 and tenth = f32 0.1 in
  [
    ( "at 0.5 and its neighbours",
      0.5,
      [| f32_step half (-1l); half; f32_step half 1l |],
      [| false; false; true |] );
    ( "at 0.1, stored as the float32 nearest it",
      0.1,
      [| f32_step tenth (-1l); tenth; f32_step tenth 1l |],
      [| false; false; true |] );
    ( "at 0., with signed zeros and the least subnormal",
      0.,
      [| -0.; 0.; f32_step 0. 1l |],
      [| false; false; true |] );
    ( "infinities and NaN",
      0.5,
      [| Float.nan; Float.infinity; Float.neg_infinity |],
      [| false; true; false |] );
    ("a NaN threshold", Float.nan, [| 0.; Float.infinity |], [| false; false |]);
    ( "an infinite threshold",
      Float.neg_infinity,
      [| Float.neg_infinity; -1e38; Float.nan |],
      [| false; true; false |] );
  ]

let threshold_case (_, t, xs, expected) =
  equal (array bool) expected (Nx.to_array (threshold t (row xs)))

(* A threshold and a row of float32 elements: some next to the float32 nearest
   the threshold, some drawn over every float32 bit pattern. The threshold is a
   float32 half the time and any float otherwise, NaN and infinities among
   them. *)
let threshold_draw =
  let open Gen in
  let* t = frequency [ (1, map f32 any_float); (1, any_float) ] in
  let near = map (f32_step (f32 t)) (int32_range (-2l) 2l) in
  let anywhere = map Int32.float_of_bits int32 in
  let special =
    of_list ~pp:Format.pp_print_float
      [ Float.nan; Float.infinity; Float.neg_infinity; 0.; -0. ]
  in
  let+ xs =
    array ~size:(int_range 1 12)
      (frequency [ (3, near); (1, anywhere); (1, special) ])
  in
  (t, xs)

let threshold_draw =
  let pp ppf (t, xs) =
    Format.fprintf ppf "threshold %h, elements [|%a|]" t
      (Format.pp_print_seq
         ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
         (fun ppf x -> Format.fprintf ppf "%h" x))
      (Array.to_seq xs)
  in
  Gen.with_pp pp threshold_draw

let law_threshold (t, xs) =
  let t32 = f32 t in
  cover "an element equal to the stored threshold"
    (Array.exists (fun x -> x = t32) xs);
  cover "an element next above the stored threshold"
    (Array.exists (fun x -> x > t32 && x <> 0. && f32_below x <= t32) xs);
  cover "an element equal to a stored threshold above t"
    (t32 > t && Array.exists (fun x -> x = t32) xs);
  cover "a NaN element" (Array.exists Float.is_nan xs);
  let xs = Nx.to_array (row xs) in
  equal (array bool)
    (Array.map (fun x -> x > t32) xs)
    (Nx.to_array (threshold t (row xs)))

(* ───── Morphological Operations Tests ───── *)

let test_structuring_elements () =
  let rect = structuring_element Rect (3, 5) in
  check_shape "rect shape" [| 3; 5 |] rect;
  equal ~msg:"rect" (array bool) (Array.make 15 true) (Nx.to_array rect);
  let cross = structuring_element Cross (3, 5) in
  let on_cross i = i / 5 = 1 || i mod 5 = 2 in
  equal ~msg:"cross: the centre row and column" (array bool)
    (Array.init 15 on_cross) (Nx.to_array cross)

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

(* An image of rank 3 or 4 of up to 6 by 6 pixels, its elements drawn from
   [elt]. *)
let image dt elt =
  let open Gen in
  let* n = int_range 0 2
  and+ h = int_range 1 6
  and+ w = int_range 1 6
  and+ c = int_range 1 2 in
  let shape = if n = 0 then [| h; w; c |] else [| n; h; w; c |] in
  let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 shape)) elt in
  Nx.create dt shape xs

(* A structuring element of up to 5 by 5 entries, even extents included, with
   a [true] entry. [centred] makes its centre [true]. *)
let element ?(centred = false) () =
  let open Gen in
  let* kh = int_range 1 5 and+ kw = int_range 1 5 in
  let size = kh * kw and centre = (kh / 2 * kw) + (kw / 2) in
  let+ taps = array ~size:(constant size) bool in
  let taps = Array.mapi (fun i b -> (centred && i = centre) || b) taps in
  let taps =
    if Array.exists Fun.id taps then taps
    else Array.init size (fun i -> i = centre)
  in
  Nx.create Nx.bool [| kh; kw |] taps

(* The extreme of each pixel's neighbourhood, read from the section preamble:
   entry [(i, j)] selects the pixel [i - kh / 2] rows below and [j - kw / 2]
   columns right, and a pixel outside the image is [outside]. *)
let neighbourhood_extreme pick outside kernel img =
  let s = Nx.shape img in
  let r = Array.length s in
  let h = s.(r - 3) and w = s.(r - 2) and c = s.(r - 1) in
  let kh = (Nx.shape kernel).(0) and kw = (Nx.shape kernel).(1) in
  let k = Nx.to_array kernel and x = Nx.to_array img in
  Array.init (Array.length x) (fun idx ->
      let ch = idx mod c and q = idx / c mod w and p = idx / c / w mod h in
      let base = idx - (((p * w) + q) * c) - ch in
      let acc = ref outside in
      for i = 0 to kh - 1 do
        for j = 0 to kw - 1 do
          let y = p + i - (kh / 2) and z = q + j - (kw / 2) in
          if k.((i * kw) + j) && y >= 0 && y < h && z >= 0 && z < w then
            acc := pick !acc x.(base + (((y * w) + z) * c) + ch)
        done
      done;
      !acc)

let law_extreme (type a b) (dt : (a, b) Nx.dtype) (t : a testable) (kernel, img)
    =
  let kh = (Nx.shape kernel).(0) and kw = (Nx.shape kernel).(1) in
  let s = Nx.shape img in
  let r = Array.length s in
  cover "an even extent" (kh mod 2 = 0 || kw mod 2 = 0);
  cover "an element wider than the image" (kh > s.(r - 3) || kw > s.(r - 2));
  cover "a batch" (r = 4);
  let lo = Nx.Dtype.min_value dt and hi = Nx.Dtype.max_value dt in
  equal ~msg:"erode" (array t)
    (neighbourhood_extreme min hi kernel img)
    (Nx.to_array (erode ~kernel img));
  equal ~msg:"dilate" (array t)
    (neighbourhood_extreme max lo kernel img)
    (Nx.to_array (dilate ~kernel img))

let int8_elt =
  Gen.frequency
    [
      (3, Gen.int_range (-128) 127);
      (1, Gen.of_list ~pp:Format.pp_print_int [ -128; 127 ]);
    ]

let bools img = Nx.to_array img

(* An element holding its centre erodes to a subset of the image and dilates
   to a superset. *)
let law_inclusion (kernel, img) =
  equal ~msg:"erode img is within img" (array bool) (bools img)
    (bools (Nx.logical_or (erode ~kernel img) img));
  equal ~msg:"img is within dilate img" (array bool)
    (bools (dilate ~kernel img))
    (bools (Nx.logical_or img (dilate ~kernel img)))

(* Erosion is dilation of the complement, complemented, borders included. *)
let law_duality (kernel, img) =
  let ( ~! ) = Nx.logical_not in
  equal ~msg:"erode" (array bool)
    (bools ~!(dilate ~kernel ~!img))
    (bools (erode ~kernel img));
  equal ~msg:"dilate" (array bool)
    (bools ~!(erode ~kernel ~!img))
    (bools (dilate ~kernel img))

(* An element whose entries are symmetric about its centre, which needs odd
   extents. *)
let symmetric kernel =
  let s = Nx.shape kernel and k = bools kernel in
  let n = Array.length k in
  s.(0) mod 2 = 1
  && s.(1) mod 2 = 1
  && Array.for_all Fun.id (Array.init n (fun i -> k.(i) = k.(n - 1 - i)))

(* Opening and closing are idempotent; the opening is at most the image and
   the closing at least, whatever the element. *)
let law_opening (kernel, img) =
  cover "an element not symmetric about its centre" (not (symmetric kernel));
  cover "an element without its centre"
    (not
       (Nx.item [ (Nx.shape kernel).(0) / 2; (Nx.shape kernel).(1) / 2 ] kernel));
  let shape = Nx.shape img in
  let on f xs = bools (f ~kernel (Nx.create Nx.bool shape xs)) in
  Law.idempotent ~msg:"opening" (array bool) (on opening) (bools img);
  Law.idempotent ~msg:"closing" (array bool) (on closing) (bools img);
  let opened = opening ~kernel img and closed = closing ~kernel img in
  equal ~msg:"opening img is within img" (array bool) (bools img)
    (bools (Nx.logical_or opened img));
  equal ~msg:"img is within closing img" (array bool) (bools closed)
    (bools (Nx.logical_or img closed))

(* With the element [. X X], whose neighbourhood is the pixel and the one to
   its right, the opening keeps a row [. X X], which the element fits, and the
   closing keeps its complement. *)
let test_opening_asymmetric () =
  let kernel = Nx.create Nx.bool [| 1; 3 |] [| false; true; true |] in
  let row xs = Nx.create Nx.bool [| 1; 3; 1 |] xs in
  equal ~msg:"opening" (array bool) [| false; true; true |]
    (bools (opening ~kernel (row [| false; true; true |])));
  equal ~msg:"closing" (array bool) [| true; false; false |]
    (bools (closing ~kernel (row [| true; false; false |])))

(* Each shape, at each odd size up to 9 by 9, is [true] at its centre and
   symmetric about it. *)
let law_shape (shape, (h, w)) =
  let k = structuring_element shape (h, w) in
  equal ~msg:"shape" (array int) [| h; w |] (Nx.shape k);
  equal ~msg:"centre" bool true (Nx.item [ h / 2; w / 2 ] k);
  let xs = bools k in
  let n = Array.length xs in
  equal ~msg:"symmetric" (array bool) xs
    (Array.init n (fun i -> xs.(n - 1 - i)))

let shape_draw =
  let open Gen in
  let odd = map (fun k -> (2 * k) + 1) (int_range 0 4) in
  pair
    (of_list
       ~pp:(fun ppf s ->
         Format.pp_print_string ppf
           (match s with
           | Rect -> "Rect"
           | Cross -> "Cross"
           | Ellipse -> "Ellipse"))
       [ Rect; Cross; Ellipse ])
    (pair odd odd)

(* With the element [. . X] on a one-pixel row, the neighbourhood lies wholly
   outside the image: the gradient is the dtype's least value minus its
   greatest. *)
let test_gradient_outside () =
  let kernel = Nx.create Nx.bool [| 1; 3 |] [| false; false; true |] in
  let grad dt v =
    Nx.to_array (morphological_gradient ~kernel (Nx.full dt [| 1; 1; 1 |] v))
  in
  equal ~msg:"uint8" (array int) [| 1 |] (grad Nx.uint8 7);
  equal ~msg:"float32" (array float_exact) [| Float.neg_infinity |]
    (grad Nx.float32 0.5);
  equal ~msg:"bool" (array bool) [| false |] (grad Nx.bool true)

let img3 = Nx.zeros Nx.float32 [| 3; 3; 1 |]

let kernel_refusals =
  [
    ("a kernel of rank 1", Nx.full Nx.bool [| 3 |] true);
    ("a kernel of rank 3", Nx.full Nx.bool [| 1; 1; 1 |] true);
  ]

let kernel_refusal (_, kernel) =
  raises_match (Exn.invalid_arg ~sub:"rank 2") (fun () -> erode ~kernel img3)

let test_complex_image () =
  let kernel = structuring_element Rect (3, 3) in
  raises_match Exn.invalid_arg (fun () ->
      dilate ~kernel (Nx.zeros Nx.complex64 [| 3; 3; 1 |]))

(* Where each neighbourhood holds a pixel of the image, the gradient of a
   boolean image is its uint8 gradient, [true] read as 1. *)
let law_gradient_cast (kernel, img) =
  let as_uint8 = Nx.cast Nx.uint8 in
  equal (array int)
    (Nx.to_array (morphological_gradient ~kernel (as_uint8 img)))
    (Nx.to_array (as_uint8 (morphological_gradient ~kernel img)))

let morphology_laws =
  let bool_image = image Nx.bool Gen.bool in
  let with_element ?centred img = Gen.pair (element ?centred ()) img in
  [
    prop "erode and dilate take each neighbourhood's extreme, of booleans"
      (with_element bool_image) (law_extreme Nx.bool bool);
    prop "erode and dilate take each neighbourhood's extreme, of int8"
      (with_element (image Nx.int8 int8_elt))
      (law_extreme Nx.int8 int);
    prop "erode img <= img <= dilate img for an element holding its centre"
      (with_element ~centred:true bool_image)
      law_inclusion;
    prop "erode is the complement's dilation complemented"
      (with_element bool_image) law_duality;
    prop "opening <= img <= closing, both idempotent" (with_element bool_image)
      law_opening;
    test "opening and closing by an asymmetric element" test_opening_asymmetric;
    prop "structuring_element is true at its centre and symmetric" shape_draw
      law_shape;
    cases "a kernel not of rank 2 raises" ~id:fst kernel_refusals kernel_refusal;
    test "a complex image raises" test_complex_image;
    test "morphological_gradient of a neighbourhood outside the image"
      test_gradient_outside;
    prop "morphological_gradient of booleans is their uint8 gradient"
      (with_element ~centred:true bool_image)
      law_gradient_cast;
  ]

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
  raises_match (Exn.invalid_arg ~sub:"ksize") (fun () -> sobel ~ksize img)

let laplacian_bad_ksize ksize =
  let img = create_gray_f 5 5 0.5 in
  raises_match (Exn.invalid_arg ~sub:"ksize") (fun () ->
      laplacian ~ksize img)

(* A bright vertical bar whose two sides each step through 0.5 at one column,
   8 and 23: the gradient magnitude across each side peaks there alone, so on
   the rows far from the image's top and bottom those two columns are the
   edges. *)
let test_canny () =
  let h = 32 and w = 32 in
  let level j =
    if j = 8 || j = 23 then 0.5 else if j > 8 && j < 23 then 1. else 0.
  in
  let img = Nx.init Nx.float32 [| h; w; 1 |] (fun idx -> level idx.(1)) in
  let edges = canny ~low:0.1 ~high:0.3 img in
  check_shape "canny shape" (Nx.shape img) edges;
  let rows = Nx.slice [ Nx.R (10, 22); Nx.A; Nx.A ] edges in
  let expected = Array.init (12 * w) (fun i -> i mod w = 8 || i mod w = 23) in
  equal (array bool) expected (Nx.to_array rows)

(* ───── Integration Tests ───── *)

let test_pipeline () =
  let img = create_centered_square 20 20 8 in
  let blurred = gaussian_blur ~sigma:1.5 ~ksize:5 img in
  let binary = threshold 0.5 blurred in
  let kernel = structuring_element Rect (3, 3) in
  let cleaned = erode ~kernel binary in
  let final = dilate ~kernel cleaned in
  check_shape "pipeline preserves shape" (Nx.shape img) final;
  if not (Array.exists Fun.id (Nx.to_array final)) then
    fail "Pipeline eliminated all features"

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

(* An operation giving float32 elements, [F], or boolean ones, [B]. *)
type op =
  | F of { op : 'd. 'd Nx.float32_t -> 'd Nx.float32_t }
  | B of { mask : 'd. 'd Nx.float32_t -> 'd Nx.bool_t }

let gray x = to_grayscale x
let at_image x k = Nx.place (Nx.beside x) k
let disk = structuring_element Ellipse (3, 3)

let sharpen =
  Nx.create Nx.float32 [| 3; 3 |] [| 0.; -1.; 0.; -1.; 5.; -1.; 0.; -1.; 0. |]

let set_ops =
  [
    ("to_float of to_uint8", F { op = (fun x -> to_float (to_uint8 x)) });
    ( "normalize",
      F
        {
          op =
            (fun x ->
              normalize ~mean:[ 0.5; 0.4; 0.3 ] ~std:[ 0.2; 0.3; 0.4 ] x);
        } );
    ("threshold", B { mask = (fun x -> threshold 0.5 x) });
    ("to_grayscale", F { op = gray });
    ("hsv_to_rgb of rgb_to_hsv", F { op = (fun x -> hsv_to_rgb (rgb_to_hsv x)) });
    ("adjust_contrast", F { op = (fun x -> adjust_contrast 0.5 x) });
    ("gaussian_blur", F { op = (fun x -> gaussian_blur ~sigma:1.2 x) });
    ( "gaussian_blur of a fixed table",
      F { op = (fun x -> gaussian_blur ~sigma:0. ~ksize:5 x) } );
    ( "gaussian_blur of a derived sigma",
      F { op = (fun x -> gaussian_blur ~sigma:0. ~ksize:11 x) } );
    ("box_blur", F { op = (fun x -> box_blur ~ksize:3 x) });
    ("median_blur", F { op = (fun x -> median_blur ~ksize:3 x) });
    ("filter2d", F { op = (fun x -> filter2d (at_image x sharpen) x) });
    ("unsharp_mask", F { op = (fun x -> unsharp_mask ~sigma:1. x) });
    ( "resize nearest",
      F { op = (fun x -> resize ~interpolation:Nearest ~height:3 ~width:9 x) }
    );
    ("resize bilinear", F { op = (fun x -> resize ~height:3 ~width:9 x) });
    ("resize to one row", F { op = (fun x -> resize ~height:1 ~width:4 x) });
    ("pad", F { op = (fun x -> pad ~value:0.5 (1, 0, 2, 1) x) });
    ("erode", F { op = (fun x -> erode ~kernel:disk x) });
    ("dilate", F { op = (fun x -> dilate ~kernel:disk x) });
    ("opening", F { op = (fun x -> opening ~kernel:disk x) });
    ("closing", F { op = (fun x -> closing ~kernel:disk x) });
    ( "morphological_gradient",
      F { op = (fun x -> morphological_gradient ~kernel:disk x) } );
    ("sobel x", F { op = (fun x -> fst (sobel (gray x))) });
    ("sobel y", F { op = (fun x -> snd (sobel ~ksize:5 (gray x))) });
    ("scharr x", F { op = (fun x -> fst (scharr (gray x))) });
    ("laplacian", F { op = (fun x -> laplacian ~ksize:3 (gray x)) });
    ("canny", B { mask = (fun x -> canny ~low:0.1 ~high:0.3 (gray x)) });
  ]

let batch () = Nx.Rng.uniform ~key:(Nx.Rng.key 3) Nx.float32 [| 2; 5; 6; 3 |]

let same_on_set t set host there =
  equal (placement ()) set (Nx.placement there);
  equal (array int) (Nx.shape host) (Nx.shape there);
  equal (array t) (Nx.to_array host) (Nx.to_array there)

let on_another_set (_, op) =
  let img = batch () in
  let img' = Nx.place Memory.on img in
  match op with
  | F { op } -> same_on_set float_exact Memory.on (op img) (op img')
  | B { mask } -> same_on_set bool Memory.on (mask img) (mask img')

(* A batch cut along its images over two devices stays cut so, each device
   filtering its own images. A reduction over the cut axis sums by partials,
   in another order than the host's: elements agree within rounding. *)
let cut_over_two (_, op) =
  let img = batch () in
  let cut = Two.split ~axis:0 in
  let img' = Nx.place cut img in
  match op with
  | F { op } -> same_on_set (float 1e-6) cut (op img) (op img')
  | B { mask } -> same_on_set bool cut (mask img) (mask img')

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
            ~id:(Printf.sprintf "C = %d")
            [ 3; 4; 5 ] grayscale_extra_channels;
          cases "to_grayscale_few_channels"
            ~id:(Printf.sprintf "C = %d")
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
            ~id:(fun (sigma, k) -> Printf.sprintf "sigma %g: %d taps" sigma k)
            [ (0.4, 5); (1.0, 7); (1.1, 9); (2.0, 13) ]
            gaussian_default_ksize;
          cases "gaussian_blur_bad_ksize" ~id:string_of_int [ 0; -3; 2; 4 ]
            gaussian_bad_ksize;
          cases "gaussian_blur_table"
            ~id:(fun (sigma, taps, _) ->
              Printf.sprintf "sigma %g, ksize %d" sigma (Array.length taps))
            gaussian_tables gaussian_table;
          cases "gaussian_blur_derived_sigma"
            ~id:(fun (sigma, k) ->
              Printf.sprintf "sigma %g, ksize %d" sigma k)
            [ (0.0, 11); (-1.0, 13); (-0.0, 15) ]
            gaussian_derived_sigma;
          cases "gaussian_blur_no_sigma" ~id:string_of_float
            [ 0.0; -0.0; -2.5; Float.nan ] gaussian_no_sigma;
          cases "gaussian_blur_nan_sigma" ~id:string_of_int [ 1; 3; 11 ]
            gaussian_nan_sigma;
          cases "gaussian_blur_huge_sigma" ~id:string_of_float
            [ infinity; 1e300; Float.of_int max_int ]
            gaussian_huge_sigma;
          test "median_blur" test_median_blur;
          test "median_blur_median" test_median_blur_preserves_median;
        ];
      group "thresholding"
        [
          cases "threshold" ~id:(fun (n, _, _, _) -> n) threshold_cases
            threshold_case;
          prop "threshold is each element's comparison with t" threshold_draw
            law_threshold;
        ];
      group "morphology"
        ([
          test "structuring_elements" test_structuring_elements;
          test "erosion" test_erosion;
          test "dilation" test_dilation;
          cases "constant_int8" ~id:string_of_int [ -128; -5; 0; 127 ]
            (constant_morphology Nx.int8 int);
          cases "constant_float32" ~id:string_of_float
            [ Float.neg_infinity; -0.5; -0.; 1.; Float.infinity ]
            (constant_morphology Nx.float32 float_exact);
          test "dilation_kernel_shape" test_dilation_kernel_shape;
        ]
        @ morphology_laws);
      group "edge_detection"
        [
          test "sobel" test_sobel;
          cases "kernels" ~id:(fun (n, _, _) -> n) edge_kernels edge_kernel;
          cases "sobel_ramp" ~id:(Printf.sprintf "ksize %d") [ 1; 3; 5; 7 ]
            sobel_ramp;
          cases "laplacian_square" ~id:(Printf.sprintf "ksize %d")
            [ 1; 3; 5; 7; 9; 11 ] laplacian_square;
          cases "sobel_bad_ksize" ~id:string_of_int [ -1; 0; 2; 9 ]
            sobel_bad_ksize;
          cases "laplacian_bad_ksize" ~id:string_of_int [ -1; 0; 2; 10 ]
            laplacian_bad_ksize;
          test "canny" test_canny;
        ];
      group "integration" [ test "pipeline" test_pipeline ];
      group "device_sets"
        [
          cases "an operation computes on its image's set"
            ~id:(fun (n, _) -> n)
            set_ops on_another_set;
          cases "an operation computes on a batch cut over two devices"
            ~id:(fun (n, _) -> n)
            set_ops cut_over_two;
        ];
    ])
