(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Test_nx_support

let pi = 4.0 *. atan 1.0
let two_pi = 2.0 *. pi

(* Naive reference: gather every window element with plain loops. *)
let reference_windows ~axis ~window ~step shape data =
  let rank = Array.length shape in
  let count = ((shape.(axis) - window) / step) + 1 in
  let out_shape =
    Array.init (rank + 1) (fun i ->
        if i = rank then window else if i = axis then count else shape.(i))
  in
  let in_strides = Array.make rank 1 in
  for i = rank - 2 downto 0 do
    in_strides.(i) <- in_strides.(i + 1) * shape.(i + 1)
  done;
  let out_numel = Array.fold_left ( * ) 1 out_shape in
  let out = Array.make out_numel 0.0 in
  let index = Array.make (rank + 1) 0 in
  for flat = 0 to out_numel - 1 do
    let rem = ref flat in
    for i = rank downto 0 do
      index.(i) <- !rem mod out_shape.(i);
      rem := !rem / out_shape.(i)
    done;
    let src = ref 0 in
    for i = 0 to rank - 1 do
      let pos =
        if i = axis then (index.(i) * step) + index.(rank) else index.(i)
      in
      src := !src + (pos * in_strides.(i))
    done;
    out.(flat) <- data.(!src)
  done;
  (out_shape, out)

let check_against_reference msg ~axis ~window ~step shape data =
  let expected_shape, expected =
    reference_windows ~axis ~window ~step shape data
  in
  let windows =
    Nx.sliding_window_view ~axis ~window ~step (Nx.create Nx.float64 shape data)
  in
  check_t msg expected_shape expected windows

let test_basic () =
  let x = Nx.create Nx.int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |] in
  let windows = Nx.sliding_window_view ~window:3 x in
  check_t "window 3 step 1" [| 3; 3 |]
    [| 1l; 2l; 3l; 2l; 3l; 4l; 3l; 4l; 5l |]
    windows

let test_step () =
  let data = Array.init 10 float_of_int in
  check_against_reference "step 2 window 4" ~axis:0 ~window:4 ~step:2 [| 10 |]
    data;
  (* step > window skips elements between windows *)
  check_against_reference "step 3 window 2" ~axis:0 ~window:2 ~step:3 [| 10 |]
    data;
  (* step = window partitions the axis *)
  let x = Nx.create Nx.int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  check_t "step 2 window 2" [| 3; 2 |]
    [| 1l; 2l; 3l; 4l; 5l; 6l |]
    (Nx.sliding_window_view ~window:2 ~step:2 x);
  (* trailing elements that do not fill a window are dropped *)
  check_against_reference "step 2 window 3 remainder" ~axis:0 ~window:3 ~step:2
    [| 8 |]
    (Array.init 8 float_of_int)

let test_window_equals_axis () =
  let x = Nx.create Nx.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
  check_t "single window" [| 1; 4 |] [| 1l; 2l; 3l; 4l |]
    (Nx.sliding_window_view ~window:4 x)

let test_batched () =
  let data = Array.init 48 float_of_int in
  (* default axis is the last one *)
  let expected_shape, expected =
    reference_windows ~axis:2 ~window:3 ~step:2 [| 2; 3; 8 |] data
  in
  let windows =
    Nx.sliding_window_view ~window:3 ~step:2
      (Nx.create Nx.float64 [| 2; 3; 8 |] data)
  in
  check_t "batched last axis" expected_shape expected windows;
  check_against_reference "middle axis" ~axis:1 ~window:2 ~step:1 [| 2; 5; 3 |]
    (Array.init 30 float_of_int);
  check_against_reference "first axis" ~axis:0 ~window:2 ~step:2 [| 4; 3 |]
    (Array.init 12 float_of_int)

let test_negative_axis () =
  let x = Nx.create Nx.float64 [| 2; 5; 3 |] (Array.init 30 float_of_int) in
  let negative = Nx.sliding_window_view ~axis:(-2) ~window:2 x in
  let positive = Nx.sliding_window_view ~axis:1 ~window:2 x in
  check_t "axis -2 matches axis 1" (Nx.shape positive) (Nx.to_array positive)
    negative

let test_strided_input () =
  let x = Nx.create Nx.float64 [| 4; 6 |] (Array.init 24 float_of_int) in
  let xt = Nx.transpose x in
  let expected_shape, expected =
    reference_windows ~axis:1 ~window:2 ~step:2 [| 6; 4 |] (Nx.to_array xt)
  in
  check_t "transposed input" expected_shape expected
    (Nx.sliding_window_view ~axis:1 ~window:2 ~step:2 xt)

let test_view_shares_storage () =
  let x = Nx.create Nx.float32 [| 5 |] [| 0.; 1.; 2.; 3.; 4. |] in
  let windows = Nx.sliding_window_view ~window:3 x in
  equal ~msg:"same buffer" bool true (Nx.data windows == Nx.data x);
  equal ~msg:"byte strides" (array int) [| 4; 4 |] (Nx.strides windows);
  equal ~msg:"offset" int 0 (Nx.offset windows);
  equal ~msg:"not contiguous" bool false (Nx.is_c_contiguous windows);
  Nx.set_item [ 1 ] 10. x;
  check_data "write to base visible in overlapping windows"
    [| 0.; 10.; 2.; 10.; 2.; 3.; 2.; 3.; 4. |]
    windows

let test_rfft_over_view () =
  let n = 128 in
  let signal =
    Array.init n (fun i ->
        sin (two_pi *. float_of_int i /. 16.0)
        +. (0.5 *. cos (two_pi *. float_of_int i /. 5.0)))
  in
  let x = Nx.create Nx.float64 [| n |] signal in
  let frames = Nx.sliding_window_view ~window:32 ~step:8 x in
  let spectrum_view = Nx.rfft frames ~axis:(-1) in
  let spectrum_copy = Nx.rfft (Nx.contiguous frames) ~axis:(-1) in
  check_nx ~epsilon:1e-12 "rfft over view vs copy" spectrum_copy spectrum_view

let test_errors () =
  let x = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  check_invalid_arg "window zero"
    "sliding_window_view: window must be >= 1, got 0" (fun () ->
      Nx.sliding_window_view ~window:0 x);
  check_invalid_arg "step zero" "sliding_window_view: step must be >= 1, got 0"
    (fun () -> Nx.sliding_window_view ~window:2 ~step:0 x);
  check_invalid_arg "window too large"
    "sliding_window_view: cannot slide window 6 along axis 0 in shape [5] (6>5)"
    (fun () -> Nx.sliding_window_view ~window:6 x);
  check_invalid_arg "axis out of bounds"
    "sliding_window_view: axis 1 out of bounds for 1D tensor" (fun () ->
      Nx.sliding_window_view ~axis:1 ~window:2 x);
  check_invalid_arg "scalar input"
    "sliding_window_view: axis -1 out of bounds for 0D tensor" (fun () ->
      Nx.sliding_window_view ~window:1 (Nx.scalar Nx.float32 1.0))

let suite =
  [
    group "windows"
      [
        test "basic" test_basic;
        test "step" test_step;
        test "window equals axis" test_window_equals_axis;
        test "batched" test_batched;
        test "negative axis" test_negative_axis;
        test "strided input" test_strided_input;
      ];
    group "views" [ test "shares storage" test_view_shares_storage ];
    group "rfft" [ test "view vs copy" test_rfft_over_view ];
    group "errors" [ test "invalid arguments" test_errors ];
  ]

let () = run "Nx Sliding window" suite
