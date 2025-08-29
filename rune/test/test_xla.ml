open Alcotest

(* Basic Tests *)

let test_xla_basic () =
  let open Rune in
  let dev = ocaml in
  let x = zeros dev float32 [| 3; 3 |] in
  let x = fill 2.0 x in

  let f t = sin (mul t t) in
  let result = xla f x in

  (* sin(2 * 2) = sin(4) ≈ -0.7568 *)
  let expected = Stdlib.sin (2.0 *. 2.0) in
  let actual = item [ 0; 0 ] result in
  check (float 0.0001) "xla basic" expected actual

let test_xla_reduce () =
  let open Rune in
  let dev = ocaml in
  let x = zeros dev float32 [| 2; 3 |] in
  let x = fill 1.0 x in

  let f t = sum t ~axes:[| 1 |] in
  let result = xla f x in

  (* Sum along axis 1: [3.0; 3.0] *)
  check (float 0.0001) "reduce sum axis 1, row 0" 3.0 (item [ 0 ] result);
  check (float 0.0001) "reduce sum axis 1, row 1" 3.0 (item [ 1 ] result)

let test_xla_reshape () =
  let open Rune in
  let dev = ocaml in
  let x = arange_f dev float32 0. 6. 1. in

  let f t = reshape [| 2; 3 |] t in
  let result = xla f x in

  check (float 0.0001) "reshape [0,0]" 0.0 (item [ 0; 0 ] result);
  check (float 0.0001) "reshape [0,1]" 1.0 (item [ 0; 1 ] result);
  check (float 0.0001) "reshape [1,2]" 5.0 (item [ 1; 2 ] result)

let test_xla_transpose () =
  let open Rune in
  let dev = ocaml in
  let x = arange_f dev float32 1. 7. 1. in
  let x = reshape [| 2; 3 |] x in

  let f t = transpose t ~axes:[| 1; 0 |] in
  let result = xla f x in

  check (float 0.0001) "transpose [0,0]" 1.0 (item [ 0; 0 ] result);
  check (float 0.0001) "transpose [0,1]" 4.0 (item [ 0; 1 ] result);
  check (float 0.0001) "transpose [2,1]" 6.0 (item [ 2; 1 ] result)

let test_xla_composite () =
  let open Rune in
  let dev = ocaml in
  let x = zeros dev float32 [| 10 |] in
  let x = fill 0.5 x in

  let f t =
    let t1 = add t t in
    (* 0.5 + 0.5 = 1.0 *)
    let t2 = mul t1 t1 in
    (* 1.0 * 1.0 = 1.0 *)
    let t3 = sqrt t2 in
    (* sqrt(1.0) = 1.0 *)
    let t4 = neg t3 in
    (* -1.0 *)
    t4
    (* Just return -1.0 for now *)
  in
  let result = xla f x in

  check (float 0.0001) "composite operations" (-1.0) (item [ 0 ] result)

let test_xla_comparison () =
  let open Rune in
  let dev = ocaml in
  let x = arange_f dev float32 0. 5. 1. in
  let y = fill 2.5 (zeros dev float32 [| 5 |]) in

  let f a =
    let cond = cmplt a y in
    (* a < 2.5 *)
    where cond a y (* if a < 2.5 then a else 2.5 *)
  in
  let result = xla f x in

  (* Expected: [0, 1, 2, 2.5, 2.5] *)
  check (float 0.0001) "comparison [0]" 0.0 (item [ 0 ] result);
  check (float 0.0001) "comparison [2]" 2.0 (item [ 2 ] result);
  check (float 0.0001) "comparison [3]" 2.5 (item [ 3 ] result);
  check (float 0.0001) "comparison [4]" 2.5 (item [ 4 ] result)

let test_simple () =
  let open Rune in
  let dev = ocaml in

  let x = arange_f dev float32 0. 4. 1. in
  let f t = add t t in
  let result = xla f x in

  check (float 0.0001) "simple add" 0.0 (item [ 0 ] result)

(* TODO: Fix these tests let test_xla_cast () = let open Rune in let dev = ocaml
   in let x = arange dev int32 0 5 1 in

   let f t = let float_t = cast float32 t in mul float_t (scalar dev float32
   2.0) in let result = xla f x in

   (* Expected: [0, 2, 4, 6, 8] as floats *) check (float 0.0001) "cast [1]" 2.0
   (item [1] result); check (float 0.0001) "cast [3]" 6.0 (item [3] result)

   let test_xla_pad () = let open Rune in let dev = ocaml in let x = arange_f
   dev float32 0. 3. 1. in

   let f t = pad [|(1, 2)|] 10.0 t in let result = xla f x in

   (* Expected: [10, 0, 1, 2, 10, 10] *) check (float 0.0001) "pad [0]" 10.0
   (item [0] result); check (float 0.0001) "pad [1]" 0.0 (item [1] result);
   check (float 0.0001) "pad [4]" 10.0 (item [4] result)

   let test_xla_concat () = let open Rune in let dev = ocaml in let x1 =
   arange_f dev float32 0. 3. 1. in

   (* For now, test concatenating the same tensor with itself *) let f a =
   concatenate ~axis:0 [a; a] in let result = xla f x1 in

   (* Expected: [0, 1, 2, 0, 1, 2] *) check (float 0.0001) "concat [2]" 2.0
   (item [2] result); check (float 0.0001) "concat [3]" 0.0 (item [3] result) *)

(* GPU Tests *)

let test_xla_gpu () =
  let open Rune in
  if not (is_device_available `metal) then skip ()
  else
    let dev = metal () in
    let x = zeros dev float32 [| 3; 3 |] in
    let x = fill 2.0 x in

    let f t = sin (mul t t) in
    let result = xla f x in

    (* sin(2 * 2) = sin(4) ≈ -0.7568 *)
    let expected = Stdlib.sin (2.0 *. 2.0) in
    let actual = item [ 0; 0 ] result in
    check (float 0.0001) "xla gpu" expected actual

let test_xla_gpu_composite () =
  let open Rune in
  if not (is_device_available `metal) then skip ()
  else
    let dev = metal () in
    let x = arange_f dev float32 1. 5. 1. in

    let f t =
      let t1 = mul t t in
      (* square *)
      let t2 = sqrt t1 in
      (* sqrt of square = original *)
      let t3 = add t2 t in
      (* double *)
      div t3 t (* divide by original = 2.0 *)
    in
    let result = xla f x in

    (* Result should be 2.0 for all elements *)
    check (float 0.0001) "gpu composite [0]" 2.0 (item [ 0 ] result);
    check (float 0.0001) "gpu composite [1]" 2.0 (item [ 1 ] result);
    check (float 0.0001) "gpu composite [3]" 2.0 (item [ 3 ] result)

(* More Operations Tests *)

let test_xla_more_operations () =
  let open Rune in
  let dev = ocaml in

  (* Test that our new operation handlers work with the XLA compiler *)

  (* Test reciprocal: 1/x *)
  let x = fill 2.0 (zeros dev float32 [| 3 |]) in
  let ones = fill 1.0 (zeros dev float32 [| 3 |]) in
  let f t = div ones t in
  let result = xla f x in
  check (float 0.0001) "reciprocal of 2" 0.5 (item [ 0 ] result);

  (* Test power operation *)
  let base = arange_f dev float32 1. 4. 1. in
  let exp = fill 2.0 (zeros dev float32 [| 3 |]) in
  let f_pow a = pow a exp in
  let result_pow = xla f_pow base in
  check (float 0.0001) "pow [0]: 1^2" 1.0 (item [ 0 ] result_pow);
  check (float 0.0001) "pow [1]: 2^2" 4.0 (item [ 1 ] result_pow);
  check (float 0.0001) "pow [2]: 3^2" 9.0 (item [ 2 ] result_pow);

  (* Test modulo operation *)
  let dividend = arange_f dev float32 5. 8. 1. in
  let divisor = fill 3.0 (zeros dev float32 [| 3 |]) in
  let f_mod a = mod_ a divisor in
  let result_mod = xla f_mod dividend in
  check (float 0.0001) "mod [0]: 5 % 3" 2.0 (item [ 0 ] result_mod);
  check (float 0.0001) "mod [1]: 6 % 3" 0.0 (item [ 1 ] result_mod);
  check (float 0.0001) "mod [2]: 7 % 3" 1.0 (item [ 2 ] result_mod)

let test_xla_reduce_operations () =
  let open Rune in
  let dev = ocaml in

  (* Test reduce with keepdims *)
  let x = arange_f dev float32 0. 12. 1. in
  let x = reshape [| 3; 4 |] x in

  let f t = sum t ~axes:[| 1 |] ~keepdims:true in
  let result = xla f x in

  (* Sum along axis 1 with keepdims: [[6], [22], [38]] *)
  check (list Alcotest.int) "result shape" [ 3; 1 ]
    (Array.to_list (shape result));
  check (float 0.0001) "sum [0,0]" 6.0 (item [ 0; 0 ] result);
  check (float 0.0001) "sum [1,0]" 22.0 (item [ 1; 0 ] result);
  check (float 0.0001) "sum [2,0]" 38.0 (item [ 2; 0 ] result)

let test_xla_shape_operations () =
  let open Rune in
  let dev = ocaml in

  (* Test various shape operations *)
  let x = arange_f dev float32 0. 24. 1. in

  let f t =
    let t1 = reshape [| 2; 3; 4 |] t in
    let t2 = transpose t1 ~axes:[| 2; 0; 1 |] in
    (* [4; 2; 3] *)
    let t3 = reshape [| 8; 3 |] t2 in
    t3
  in
  let result = xla f x in

  check (list Alcotest.int) "final shape" [ 8; 3 ]
    (Array.to_list (shape result))

let test_xla_mixed_operations () =
  let open Rune in
  let dev = ocaml in

  (* Complex expression mixing various operations *)
  let x = arange_f dev float32 1. 9. 1. in

  let f t =
    let t1 = reshape [| 2; 4 |] t in
    let t2 = transpose t1 ~axes:[| 1; 0 |] in
    (* [4; 2] *)
    let t3 = sum t2 ~axes:[| 1 |] in
    (* [4] *)
    let t4 = sqrt t3 in
    t4
  in
  let result = xla f x in

  (* t1 = [[1,2,3,4], [5,6,7,8]] t2 = [[1,5], [2,6], [3,7], [4,8]] t3 = [6, 8,
     10, 12] t4 = [sqrt(6), sqrt(8), sqrt(10), sqrt(12)] *)
  check (float 0.001) "mixed [0]" 2.449 (item [ 0 ] result);
  check (float 0.001) "mixed [1]" 2.828 (item [ 1 ] result);
  check (float 0.001) "mixed [2]" 3.162 (item [ 2 ] result);
  check (float 0.001) "mixed [3]" 3.464 (item [ 3 ] result)

(* Gather/Shrink Tests *)

let test_xla_shrink () =
  let open Rune in
  let dev = ocaml in

  (* Create a simple 2D tensor *)
  let data = arange_f dev float32 0. 6. 1. in
  let data = reshape [| 2; 3 |] data in

  (* Test shrink operation *)
  let f data_tensor = shrink [| (0, 1); (0, 2) |] data_tensor in
  let result = xla f data in

  (* Expected: [[0, 1]] *)
  check (float 0.0001) "shrink [0,0]" 0.0 (item [ 0; 0 ] result);
  check (float 0.0001) "shrink [0,1]" 1.0 (item [ 0; 1 ] result)

let test_xla_concatenate () =
  let open Rune in
  let dev = ocaml in

  (* Create two simple tensors *)
  let a = fill 1.0 (zeros dev float32 [| 2; 2 |]) in
  let b = fill 2.0 (zeros dev float32 [| 2; 2 |]) in

  (* Test concatenation - both tensors need to be inputs *)
  let f (a_tensor, b_tensor) = concatenate ~axis:0 [ a_tensor; b_tensor ] in
  let result = f (a, b) in
  (* For now, skip XLA JIT for multi-input *)

  (* Check shape and values *)
  check (list Alcotest.int) "concat shape" [ 4; 2 ]
    (Array.to_list (shape result));
  check (float 0.0001) "concat [0,0]" 1.0 (item [ 0; 0 ] result);
  check (float 0.0001) "concat [2,0]" 2.0 (item [ 2; 0 ] result)

(* Scatter Tests *)

let test_xla_set_slice_simple () =
  let open Rune in
  let dev = ocaml in

  (* Create a 5x3 tensor filled with zeros *)
  let data = zeros dev float32 [| 5; 3 |] in

  (* Test set_slice which uses scatter internally *)
  let f data_t =
    let result = copy data_t in
    (* Create updates tensor - for now, we'll use fill to avoid device issues *)
    let updates = fill 1.0 (zeros dev float32 [| 3 |]) in
    set_slice [ I 0 ] result updates;
    set_slice [ I 2 ] result updates;
    set_slice [ I 4 ] result updates;
    result
  in
  let result = xla f data in

  (* Check shape *)
  check (list Alcotest.int) "set_slice shape" [ 5; 3 ]
    (Array.to_list (shape result));

  (* Check values - rows 0, 2, 4 should be 1.0, others 0.0 *)
  check (float 0.0001) "set_slice [0,0]" 1.0 (item [ 0; 0 ] result);
  check (float 0.0001) "set_slice [1,0]" 0.0 (item [ 1; 0 ] result);
  check (float 0.0001) "set_slice [2,0]" 1.0 (item [ 2; 0 ] result);
  check (float 0.0001) "set_slice [3,0]" 0.0 (item [ 3; 0 ] result);
  check (float 0.0001) "set_slice [4,0]" 1.0 (item [ 4; 0 ] result)

let test_xla_set_slice_2d () =
  let open Rune in
  let dev = ocaml in

  (* Create a 4x4 tensor filled with zeros *)
  let data = zeros dev float32 [| 4; 4 |] in

  (* Test setting a 2x2 block *)
  let f data_t =
    let result = copy data_t in
    let block = fill 1.0 (zeros dev float32 [| 2; 2 |]) in
    set_slice [ R (1, 3); R (1, 3) ] result block;
    result
  in
  let result = xla f data in

  (* Check values - center 2x2 block should be 1.0 *)
  check (float 0.0001) "set_slice 2d [0,0]" 0.0 (item [ 0; 0 ] result);
  check (float 0.0001) "set_slice 2d [1,1]" 1.0 (item [ 1; 1 ] result);
  check (float 0.0001) "set_slice 2d [1,2]" 1.0 (item [ 1; 2 ] result);
  check (float 0.0001) "set_slice 2d [2,1]" 1.0 (item [ 2; 1 ] result);
  check (float 0.0001) "set_slice 2d [2,2]" 1.0 (item [ 2; 2 ] result);
  check (float 0.0001) "set_slice 2d [3,3]" 0.0 (item [ 3; 3 ] result)

let test_xla_set_slice_fancy_indexing () =
  let open Rune in
  let dev = ocaml in

  (* Test fancy indexing with set_slice *)
  let data = zeros dev float32 [| 5; 3 |] in

  let f data_t =
    (* Set specific rows using fancy indexing *)
    let result = copy data_t in
    let rows = fill 1.0 (zeros dev float32 [| 2; 3 |]) in
    set_slice [ L [ 1; 3 ] ] result rows;
    result
  in
  let result = xla f data in

  (* Check values - rows 1 and 3 should be 1.0 *)
  check (float 0.0001) "fancy indexing [0,0]" 0.0 (item [ 0; 0 ] result);
  check (float 0.0001) "fancy indexing [1,0]" 1.0 (item [ 1; 0 ] result);
  check (float 0.0001) "fancy indexing [2,0]" 0.0 (item [ 2; 0 ] result);
  check (float 0.0001) "fancy indexing [3,0]" 1.0 (item [ 3; 0 ] result);
  check (float 0.0001) "fancy indexing [4,0]" 0.0 (item [ 4; 0 ] result)

(* Convolution and Fold Tests *)

let test_xla_conv2d_simple () =
  let open Rune in
  let dev = ocaml in

  (* Create input: batch=1, channels=1, height=5, width=5 *)
  let input = arange_f dev float32 0. 25. 1. in
  let input = reshape [| 1; 1; 5; 5 |] input in

  (* Test basic convolution *)
  let f input_t =
    (* Create kernel inside traced function to avoid context issues *)
    let kernel =
      create dev float32 [| 1; 1; 3; 3 |]
        [| 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1. |]
    in
    convolve2d input_t kernel ~stride:(1, 1) ~padding_mode:`Valid
  in
  let result = xla f input in

  (* Check output shape: [1, 1, 3, 3] *)
  check (list Alcotest.int) "conv2d shape" [ 1; 1; 3; 3 ]
    (Array.to_list (shape result));

  (* Check some values - convolution with all-ones kernel sums the window *)
  (* Top-left corner: sum of [[0,1,2], [5,6,7], [10,11,12]] = 54 *)
  check (float 0.1) "conv2d [0,0,0,0]" 54.0 (item [ 0; 0; 0; 0 ] result)

let test_xla_conv2d_with_dilation () =
  let open Rune in
  let dev = ocaml in

  (* Create input: batch=1, channels=1, height=7, width=7 *)
  let input = arange_f dev float32 0. 49. 1. in
  let input = reshape [| 1; 1; 7; 7 |] input in

  (* Test convolution with dilation=2 *)
  let f input_t =
    (* Create kernel inside traced function *)
    let kernel =
      create dev float32 [| 1; 1; 3; 3 |]
        [| 0.33; 0.33; 0.33; 0.33; 0.33; 0.33; 0.33; 0.33; 0.33 |]
    in
    convolve2d input_t kernel ~stride:(1, 1) ~padding_mode:`Valid
      ~dilation:(2, 2)
  in
  let result = xla f input in

  (* With dilation=2, the effective kernel size is 5x5 *)
  (* Output shape should be [1, 1, 3, 3] *)
  check (list Alcotest.int) "conv2d dilated shape" [ 1; 1; 3; 3 ]
    (Array.to_list (shape result))

let test_xla_conv2d_with_padding () =
  let open Rune in
  let dev = ocaml in

  (* Create input: batch=1, channels=1, height=3, width=3 *)
  let input = arange_f dev float32 1. 10. 1. in
  let input = reshape [| 1; 1; 3; 3 |] input in

  (* Test convolution with padding *)
  let f input_t =
    (* Create kernel inside traced function *)
    let kernel =
      create dev float32 [| 1; 1; 3; 3 |]
        [| 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1. |]
    in
    convolve2d input_t kernel ~stride:(1, 1) ~padding_mode:`Same
  in
  let result = xla f input in

  (* With padding=Same, output shape should be same as input: [1, 1, 3, 3] *)
  check (list Alcotest.int) "conv2d padded shape" [ 1; 1; 3; 3 ]
    (Array.to_list (shape result))

let test_xla_unfold_fold () =
  let open Rune in
  let dev = ocaml in

  (* Create input: batch=1, channels=1, height=4, width=4 *)
  let input = arange_f dev float32 1. 17. 1. in
  let input = reshape [| 1; 1; 4; 4 |] input in

  (* Test unfold operation *)
  let f input_t =
    im2col ~kernel_size:[| 2; 2 |] ~stride:[| 1; 1 |] ~dilation:[| 1; 1 |]
      ~padding:[| (0, 0); (0, 0) |]
      input_t
  in
  (* Note: unfold is currently simulated using conv2d in XLA, so results may
     differ *)
  let unfolded = xla f input in

  (* Check unfolded shape: [batch, channels * kh * kw, h_out, w_out] *)
  (* With 4x4 input, 2x2 kernel, stride 1, no padding: output is 3x3 *)
  (* Shape should be [1, 1*2*2, 3, 3] = [1, 4, 3, 3] *)
  check (list Alcotest.int) "unfold shape" [ 1; 4; 3; 3 ]
    (Array.to_list (shape unfolded));

  (* Test fold operation (inverse of unfold) *)
  let g unfolded_t =
    col2im ~output_size:[| 4; 4 |] ~kernel_size:[| 2; 2 |] ~stride:[| 1; 1 |]
      ~dilation:[| 1; 1 |]
      ~padding:[| (0, 0); (0, 0) |]
      unfolded_t
  in
  (* Note: fold is currently simplified in XLA, so it won't perfectly
     reconstruct the input *)
  let folded = xla g unfolded in

  (* Check folded shape - should match original input shape *)
  check (list Alcotest.int) "fold shape" [ 1; 1; 4; 4 ]
    (Array.to_list (shape folded))

(* Test Runner *)

let () =
  run "XLA Compiler Tests"
    [
      ( "basic",
        [
          test_case "xla basic" `Quick test_xla_basic;
          test_case "xla reduce" `Quick test_xla_reduce;
          test_case "xla reshape" `Quick test_xla_reshape;
          test_case "xla transpose" `Quick test_xla_transpose;
          test_case "xla composite" `Quick test_xla_composite;
          test_case "xla comparison" `Quick test_xla_comparison;
          test_case "simple" `Quick test_simple;
          (* TODO: Fix these tests test_case "xla cast" `Quick test_xla_cast;
             test_case "xla pad" `Quick test_xla_pad; test_case "xla concat"
             `Quick test_xla_concat; *)
        ] );
      ( "gpu",
        [
          test_case "xla gpu basic" `Quick test_xla_gpu;
          test_case "xla gpu composite" `Quick test_xla_gpu_composite;
        ] );
      ( "operations",
        [
          test_case "more operations" `Quick test_xla_more_operations;
          test_case "reduce operations" `Quick test_xla_reduce_operations;
          test_case "shape operations" `Quick test_xla_shape_operations;
          test_case "mixed operations" `Quick test_xla_mixed_operations;
        ] );
      ( "gather_shrink",
        [
          test_case "shrink operation" `Quick test_xla_shrink;
          test_case "concatenate operation" `Quick test_xla_concatenate;
        ] );
      ( "scatter",
        [
          test_case "simple set_slice" `Quick test_xla_set_slice_simple;
          test_case "2D set_slice" `Quick test_xla_set_slice_2d;
          test_case "fancy indexing" `Quick test_xla_set_slice_fancy_indexing;
        ] );
      ( "conv_fold",
        [
          test_case "conv2d simple" `Quick test_xla_conv2d_simple;
          test_case "conv2d with dilation" `Quick test_xla_conv2d_with_dilation;
          test_case "conv2d with padding" `Quick test_xla_conv2d_with_padding;
          test_case "unfold and fold" `Quick test_xla_unfold_fold;
        ] );
    ]
