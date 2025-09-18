(* Linear algebra tests for Nx *)

open Alcotest
open Test_nx_support

(* ───── Convolution Tests ───── *)

let test_convolve1d_basic () =
  (* Basic 1D convolution - proper tensor format: (batch, channels, width) *)
  let input = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
  let result = Nx.convolve1d input kernel in
  (* NumPy convolve flips kernel, so result is [2., 2., 2.] *)
  check_t "convolve1d basic" [| 1; 1; 3 |] [| 2.; 2.; 2. |] result

let test_convolve1d_padding_modes () =
  let input = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in

  (* Valid padding (default) - output size = input - kernel + 1 *)
  let valid = Nx.convolve1d ~padding_mode:`Valid input kernel in
  check_t "convolve1d valid padding" [| 1; 1; 3 |] [| 6.; 9.; 12. |] valid;

  (* Same padding - output size = input size *)
  let same = Nx.convolve1d ~padding_mode:`Same input kernel in
  check_t "convolve1d same padding" [| 1; 1; 5 |] [| 3.; 6.; 9.; 12.; 9. |] same;

  (* Full padding - output size = input + kernel - 1 *)
  let full = Nx.convolve1d ~padding_mode:`Full input kernel in
  check_t "convolve1d full padding" [| 1; 1; 7 |]
    [| 1.; 3.; 6.; 9.; 12.; 9.; 5. |]
    full

let test_convolve1d_stride () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 8 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in

  (* Stride 2 *)
  let result = Nx.convolve1d ~stride:2 input kernel in
  check_t "convolve1d stride 2" [| 1; 1; 3 |] [| 6.; 12.; 18. |] result

let test_convolve1d_dilation () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 7 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7. |]
  in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; 1. |] in

  (* Dilation 2 - kernel elements are 2 positions apart *)
  (* With dilation=2, effective kernel size = 2*(3-1)+1 = 5 *)
  (* Output size = 7 - 5 + 1 = 3 *)
  let result = Nx.convolve1d ~dilation:2 input kernel in
  check_t "convolve1d dilation 2" [| 1; 1; 3 |] [| 6.; 8.; 10. |] result

let test_convolve1d_groups () =
  (* Groups=2: split 4 channels into 2 groups of 2 channels each *)
  let input = Nx.create Nx.float32 [| 1; 4; 4 |] (Array.init 16 float_of_int) in
  let kernel =
    Nx.create Nx.float32 [| 2; 2; 2 |] [| 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1. |]
  in

  let result = Nx.convolve1d ~groups:2 input kernel in
  check_shape "convolve1d groups shape" [| 1; 2; 3 |] result

let test_convolve1d_bias () =
  let input = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 2.; 3. |] in
  let kernel = Nx.create Nx.float32 [| 1; 1; 2 |] [| 1.; 1. |] in
  let bias = Nx.create Nx.float32 [| 1 |] [| 10. |] in

  let result = Nx.convolve1d ~bias input kernel in
  check_t "convolve1d with bias" [| 1; 1; 2 |] [| 13.; 15. |] result

let test_correlate1d_basic () =
  (* Basic 1D correlation - kernel is not flipped *)
  let input = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
  let result = Nx.correlate1d input kernel in
  check_t "correlate1d basic" [| 1; 1; 3 |] [| -2.; -2.; -2. |] result

let test_convolve2d_basic () =
  (* Basic 2D convolution *)
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
  in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in
  let result = Nx.convolve2d input kernel in
  check_t "convolve2d basic" [| 1; 1; 2; 2 |] [| 45.; 54.; 81.; 90. |] result

let test_convolve2d_padding_modes () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 3; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let kernel = Nx.create Nx.float32 [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |] in

  (* Valid padding *)
  let valid = Nx.convolve2d ~padding_mode:`Valid input kernel in
  check_t "convolve2d valid padding" [| 1; 1; 2; 2 |] [| 12.; 16.; 24.; 28. |]
    valid;

  (* Same padding *)
  let same = Nx.convolve2d ~padding_mode:`Same input kernel in
  check_t "convolve2d same padding" [| 1; 1; 3; 3 |]
    [| 1.; 3.; 5.; 5.; 12.; 16.; 11.; 24.; 28. |]
    (* Now uses proper convolution padding *)
    same;

  (* Full padding *)
  let full = Nx.convolve2d ~padding_mode:`Full input kernel in
  check_shape "convolve2d full padding shape" [| 1; 1; 4; 4 |] full

let test_convolve2d_stride () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 5; 5 |] (Array.init 25 float_of_int)
  in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in

  (* Stride (2,2) *)
  let result = Nx.convolve2d ~stride:(2, 2) input kernel in
  check_shape "convolve2d stride shape" [| 1; 1; 2; 2 |] result;
  check_t "convolve2d stride values" [| 1; 1; 2; 2 |] [| 54.; 72.; 144.; 162. |]
    result

let test_convolve2d_dilation () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 5; 5 |] (Array.init 25 float_of_int)
  in
  let kernel =
    Nx.create Nx.float32 [| 1; 1; 3; 3 |]
      [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1. |]
  in
  (* Only corners *)

  (* Dilation 2 - effective kernel size becomes 5x5 *)
  let result = Nx.convolve2d ~dilation:(2, 2) input kernel in
  check_shape "convolve2d dilation shape" [| 1; 1; 1; 1 |] result;
  check_t "convolve2d dilation value" [| 1; 1; 1; 1 |] [| 24. |] result

let test_convolve2d_multi_channel () =
  (* Multi-channel convolution: 3 input channels, 2 output channels *)
  let input =
    Nx.create Nx.float32 [| 1; 3; 4; 4 |] (Array.init 48 float_of_int)
  in
  let kernel = Nx.create Nx.float32 [| 2; 3; 3; 3 |] (Array.make 54 1.0) in

  let result = Nx.convolve2d input kernel in
  check_shape "convolve2d multi-channel shape" [| 1; 2; 2; 2 |] result

let test_convolve2d_winograd_eligible () =
  (* Test a convolution that should trigger Winograd optimization: - 3x3 kernel
     - stride 1 - groups 1 This specific test case helps catch reshape issues in
     Winograd path *)
  let input =
    Nx.create Nx.float32 [| 1; 1; 8; 8 |] (Array.init 64 float_of_int)
  in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in

  (* This should use Winograd optimization *)
  let result = Nx.convolve2d ~stride:(1, 1) input kernel in
  check_shape "convolve2d Winograd shape" [| 1; 1; 6; 6 |] result;

  (* Verify the computation is correct *)
  (* Each 3x3 window sums to 9 times the sum of its elements *)
  let expected_00 = 0. +. 1. +. 2. +. 8. +. 9. +. 10. +. 16. +. 17. +. 18. in
  check (float 1e-5) "convolve2d Winograd [0,0,0,0]" expected_00
    (Nx.item [ 0; 0; 0; 0 ] result)

let test_convolve2d_groups_winograd () =
  (* Test grouped convolution with parameters that might trigger Winograd but
     should be handled correctly *)
  let input =
    Nx.create Nx.float32 [| 1; 2; 8; 8 |] (Array.init 128 float_of_int)
  in
  let kernel = Nx.create Nx.float32 [| 2; 1; 3; 3 |] (Array.make 18 1.0) in

  (* Groups=2 should disable Winograd optimization *)
  let result = Nx.convolve2d ~groups:2 ~stride:(1, 1) input kernel in
  check_shape "convolve2d groups Winograd shape" [| 1; 2; 6; 6 |] result

let test_convolve2d_non_contiguous_input () =
  (* Test convolution with non-contiguous input (e.g., from transpose) *)
  let input =
    Nx.create Nx.float32 [| 1; 4; 4; 1 |] (Array.init 16 float_of_int)
  in
  let input_transposed = Nx.transpose ~axes:[| 0; 3; 1; 2 |] input in
  (* Now [1; 1; 4; 4] but non-contiguous *)
  let kernel = Nx.create Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in

  let result = Nx.convolve2d input_transposed kernel in
  check_shape "convolve2d non-contiguous shape" [| 1; 1; 2; 2 |] result;
  check_t "convolve2d non-contiguous values" [| 1; 1; 2; 2 |]
    [| 45.; 54.; 81.; 90. |] result

let test_correlate2d_basic () =
  (* Basic 2D correlation *)
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
  in
  let kernel =
    Nx.create Nx.float32 [| 1; 1; 3; 3 |]
      [| 1.; 0.; -1.; 0.; 0.; 0.; -1.; 0.; 1. |]
  in
  let result = Nx.correlate2d input kernel in
  check_shape "correlate2d shape" [| 1; 1; 2; 2 |] result

let test_convolve_invalid_shapes () =
  (* Test various invalid shape combinations *)
  let input_1d =
    Nx.create Nx.float32 [| 1; 2; 5 |] (Array.init 10 float_of_int)
  in
  let kernel_1d =
    Nx.create Nx.float32 [| 1; 3; 3 |] (Array.init 9 float_of_int)
  in

  check_invalid_arg "convolve1d channel mismatch"
    "correlate_nd: invalid channel configuration (2 \226\137\160 1\195\1513)\n\
     hint: expected 3 channels for 1 groups with 3 channels each" (fun () ->
      ignore (Nx.convolve1d input_1d kernel_1d))

let test_convolve_empty_input () =
  (* Empty input handling - empty on spatial dimension *)
  let input = Nx.create Nx.float32 [| 1; 1; 0 |] [||] in
  let kernel = Nx.create Nx.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in
  let result = Nx.convolve1d input kernel in
  check_shape "convolve1d empty input" [| 1; 1; 0 |] result

let test_convolve_single_element_kernel () =
  (* Single element kernel acts as scaling *)
  let input = Nx.create Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let kernel = Nx.create Nx.float32 [| 1; 1; 1 |] [| 2.0 |] in
  let result = Nx.convolve1d input kernel in
  check_t "convolve1d single kernel" [| 1; 1; 5 |] [| 2.; 4.; 6.; 8.; 10. |]
    result

let test_convolve2d_pool_reshape_edge_case () =
  (* Test case that might trigger the reshape error seen in sanity tests This
     tests the pool operation's reshape from [6; 6; 1; 1; 2; 2] to [6; 6; 4] *)
  let input =
    Nx.create Nx.float32 [| 1; 1; 6; 6 |] (Array.init 36 float_of_int)
  in
  let kernel = Nx.create Nx.float32 [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |] in

  (* Use stride 1 to get output shape [1; 1; 5; 5] *)
  let result = Nx.convolve2d ~stride:(1, 1) input kernel in
  check_shape "convolve2d pool edge case shape" [| 1; 1; 5; 5 |] result;

  (* Verify first output value: sum of top-left 2x2 window *)
  let expected_00 = 0. +. 1. +. 6. +. 7. in
  check (float 1e-5) "convolve2d pool edge case [0,0,0,0]" expected_00
    (Nx.item [ 0; 0; 0; 0 ] result)

let test_convolve2d_groups_reshape_issue () =
  (* Test grouped convolution that might cause reshape issues in pooling This
     specifically tests the optimized path for groups > 1 *)
  let input =
    Nx.create Nx.float32 [| 1; 4; 6; 6 |] (Array.init 144 float_of_int)
  in
  let kernel = Nx.create Nx.float32 [| 4; 2; 2; 2 |] (Array.make 32 1.0) in

  (* Groups=2: each group has 2 input channels and 2 output channels *)
  let result = Nx.convolve2d ~groups:2 ~stride:(1, 1) input kernel in
  check_shape "convolve2d groups reshape shape" [| 1; 4; 5; 5 |] result

let test_convolve2d_dilated_non_contiguous () =
  (* Test dilated convolution with non-contiguous tensor This can trigger
     complex reshapes in pool_dilated_path *)
  let input =
    Nx.create Nx.float32 [| 1; 5; 5; 1 |] (Array.init 25 float_of_int)
  in
  let input_perm = Nx.transpose ~axes:[| 0; 3; 1; 2 |] input in
  (* Now [1; 1; 5; 5] non-contiguous *)
  let kernel =
    Nx.create Nx.float32 [| 1; 1; 3; 3 |]
      [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1. |]
  in
  (* Only corners *)

  (* Dilation 2 tests the dilated pooling path *)
  let result = Nx.convolve2d ~dilation:(2, 2) input_perm kernel in
  check_shape "convolve2d dilated non-contig shape" [| 1; 1; 1; 1 |] result;
  (* With input 0-24 and corner kernel with dilation 2, we pick elements 0 and
     24 *)
  check_t "convolve2d dilated non-contig value" [| 1; 1; 1; 1 |] [| 24. |]
    result

let test_correlate2d_winograd_sanity_case () =
  (* Test the exact scenario from sanity tests that triggers the reshape bug *)
  (* This matches the failing sanity test exactly: correlate2d with 1x1x5x5 input, 1x1x3x3 kernel, all ones *)
  let x = Nx.ones Nx.float32 [| 1; 1; 5; 5 |] in
  let w = Nx.ones Nx.float32 [| 1; 1; 3; 3 |] in

  (* This correlation should work and produce 3x3 output with all 9s *)
  (* Note: correlate2d can also trigger Winograd when kernel is 3x3, stride 1, groups 1 *)
  let y = Nx.correlate2d x w in

  (* The expected result is a 3x3 output where each element is 9.0 *)
  check_t ~eps:1e-6 "correlate2d values" [| 1; 1; 3; 3 |]
    [| 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9. |]
    y

(* ───── Pooling Tests ───── *)

let test_max_pool1d_basic () =
  let input = Nx.create Nx.float32 [| 1; 1; 6 |] [| 1.; 3.; 2.; 6.; 4.; 5. |] in
  let output, _ = Nx.max_pool1d ~kernel_size:2 input in
  check_t "max_pool1d basic" [| 1; 1; 3 |] [| 3.; 6.; 5. |] output

let test_max_pool1d_stride () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 8 |] [| 1.; 3.; 2.; 6.; 4.; 5.; 7.; 8. |]
  in
  let output, _ = Nx.max_pool1d ~kernel_size:3 ~stride:2 input in
  check_t "max_pool1d stride" [| 1; 1; 3 |] [| 3.; 6.; 7. |] output

let test_max_pool2d_basic () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 3.; 2.; 4.; 5.; 7.; 6.; 8.; 9.; 11.; 10.; 12.; 13.; 15.; 14.; 16.;
      |]
  in
  let output, _ = Nx.max_pool2d ~kernel_size:(2, 2) input in
  check_t "max_pool2d basic" [| 1; 1; 2; 2 |] [| 7.; 8.; 15.; 16. |] output

let test_max_pool2d_stride () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
  in
  let output, _ = Nx.max_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) input in
  check_t "max_pool2d stride" [| 1; 1; 2; 2 |] [| 5.; 7.; 13.; 15. |] output

let test_max_pool2d_padding () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.;
      |]
  in
  (* Test with Same padding - output size should be ceil(input_size / stride) *)
  let output, _ =
    Nx.max_pool2d ~kernel_size:(3, 3) ~stride:(2, 2) ~padding_spec:`Same input
  in
  check_t "max_pool2d padding" [| 1; 1; 2; 2 |] [| 11.; 12.; 15.; 16. |] output

let test_min_pool1d_basic () =
  let input = Nx.create Nx.float32 [| 1; 1; 6 |] [| 4.; 2.; 3.; 1.; 6.; 5. |] in
  let output, _ = Nx.min_pool1d ~kernel_size:2 input in
  check_t "min_pool1d basic" [| 1; 1; 3 |] [| 2.; 1.; 5. |] output

let test_min_pool1d_stride () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 8 |] [| 4.; 2.; 3.; 1.; 6.; 5.; 7.; 8. |]
  in
  let output, _ = Nx.min_pool1d ~kernel_size:3 ~stride:2 input in
  check_t "min_pool1d stride" [| 1; 1; 3 |] [| 2.; 1.; 5. |] output

let test_min_pool2d_basic () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 3.; 2.; 4.; 5.; 7.; 6.; 8.; 9.; 11.; 10.; 12.; 13.; 15.; 14.; 16.;
      |]
  in
  let output, _ = Nx.min_pool2d ~kernel_size:(2, 2) input in
  check_t "min_pool2d basic" [| 1; 1; 2; 2 |] [| 1.; 2.; 9.; 10. |] output

let test_min_pool2d_stride () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 2.; 5.; 6.; 3.; 4.; 7.; 8.; 9.; 10.; 13.; 14.; 11.; 12.; 15.; 16.;
      |]
  in
  let output, _ = Nx.min_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) input in
  check_t "min_pool2d stride" [| 1; 1; 2; 2 |] [| 1.; 5.; 9.; 13. |] output

let test_min_pool2d_padding () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 3; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let output, _ =
    Nx.min_pool2d ~kernel_size:(2, 2) ~stride:(1, 1) ~padding_spec:`Same input
  in
  (* With zero padding, the minimums at edges will be 0 *)
  check_t "min_pool2d padding" [| 1; 1; 3; 3 |]
    [| 0.; 0.; 0.; 0.; 1.; 2.; 0.; 4.; 5. |]
    output

let test_min_pool2d_uint8 () =
  (* Test that min_pool works correctly with uint8 dtype *)
  let input =
    Nx.create Nx.uint8 [| 1; 1; 4; 4 |]
      [|
        255; 200; 150; 100; 180; 160; 140; 120; 90; 80; 70; 60; 50; 40; 30; 20;
      |]
  in
  let output, _ = Nx.min_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) input in
  check_t "min_pool2d uint8" [| 1; 1; 2; 2 |] [| 160; 100; 40; 20 |] output

let test_avg_pool1d_basic () =
  let input = Nx.create Nx.float32 [| 1; 1; 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let output = Nx.avg_pool1d ~kernel_size:2 input in
  check_t "avg_pool1d basic" [| 1; 1; 3 |] [| 1.5; 3.5; 5.5 |] output

let test_avg_pool2d_basic () =
  let input =
    Nx.create Nx.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
  in
  let output = Nx.avg_pool2d ~kernel_size:(2, 2) input in
  check_t "avg_pool2d basic" [| 1; 1; 2; 2 |] [| 2.5; 4.5; 10.5; 12.5 |] output

let test_pool_batch () =
  (* Test pooling with batch dimension *)
  let input =
    Nx.create Nx.float32 [| 2; 1; 4; 4 |] (Array.init 32 float_of_int)
  in
  let output, _ = Nx.max_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) input in
  check_shape "pool batch shape" [| 2; 1; 2; 2 |] output;
  (* Check first batch *)
  check (float 1e-6) "batch 0 [0,0]" 5. (Nx.item [ 0; 0; 0; 0 ] output);
  (* Check second batch *)
  check (float 1e-6) "batch 1 [0,0]" 21. (Nx.item [ 1; 0; 0; 0 ] output)

let test_pool_multichannel () =
  (* Test pooling with multiple channels *)
  let input =
    Nx.create Nx.float32 [| 1; 3; 4; 4 |] (Array.init 48 float_of_int)
  in
  let output, _ = Nx.max_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) input in
  check_shape "pool multichannel shape" [| 1; 3; 2; 2 |] output

let test_pool_edge_cases () =
  (* Test edge cases *)
  (* Single element *)
  let single = Nx.create Nx.float32 [| 1; 1; 1; 1 |] [| 42. |] in
  let out_single, _ = Nx.max_pool2d ~kernel_size:(1, 1) single in
  check_t "pool single element" [| 1; 1; 1; 1 |] [| 42. |] out_single;

  (* Empty spatial dimensions *)
  let empty = Nx.create Nx.float32 [| 1; 1; 0; 4 |] [||] in
  let out_empty, _ = Nx.max_pool2d ~kernel_size:(1, 1) empty in
  check_shape "pool empty spatial" [| 1; 1; 0; 4 |] out_empty
(* Test Suite Organization *)

let convolution_tests =
  [
    ("convolve1d basic", `Quick, test_convolve1d_basic);
    ("convolve1d padding modes", `Quick, test_convolve1d_padding_modes);
    ("convolve1d stride", `Quick, test_convolve1d_stride);
    ("convolve1d dilation", `Quick, test_convolve1d_dilation);
    ("convolve1d groups", `Quick, test_convolve1d_groups);
    ("convolve1d bias", `Quick, test_convolve1d_bias);
    ("correlate1d basic", `Quick, test_correlate1d_basic);
    ("convolve2d basic", `Quick, test_convolve2d_basic);
    ("convolve2d padding modes", `Quick, test_convolve2d_padding_modes);
    ("convolve2d stride", `Quick, test_convolve2d_stride);
    ("convolve2d dilation", `Quick, test_convolve2d_dilation);
    ("convolve2d multi-channel", `Quick, test_convolve2d_multi_channel);
    ("convolve2d winograd eligible", `Quick, test_convolve2d_winograd_eligible);
    ("convolve2d groups winograd", `Quick, test_convolve2d_groups_winograd);
    ( "convolve2d non-contiguous input",
      `Quick,
      test_convolve2d_non_contiguous_input );
    ( "convolve2d pool reshape edge case",
      `Quick,
      test_convolve2d_pool_reshape_edge_case );
    ( "convolve2d groups reshape issue",
      `Quick,
      test_convolve2d_groups_reshape_issue );
    ( "convolve2d dilated non-contiguous",
      `Quick,
      test_convolve2d_dilated_non_contiguous );
    ("convolve invalid shapes", `Quick, test_convolve_invalid_shapes);
    ("convolve empty input", `Quick, test_convolve_empty_input);
    ( "convolve single element kernel",
      `Quick,
      test_convolve_single_element_kernel );
    ("correlate2d basic", `Quick, test_correlate2d_basic);
    ( "correlate2d winograd sanity case",
      `Quick,
      test_correlate2d_winograd_sanity_case );
  ]

let pooling_tests =
  [
    ("max_pool1d basic", `Quick, test_max_pool1d_basic);
    ("max_pool1d stride", `Quick, test_max_pool1d_stride);
    ("max_pool2d basic", `Quick, test_max_pool2d_basic);
    ("max_pool2d stride", `Quick, test_max_pool2d_stride);
    ("max_pool2d padding", `Quick, test_max_pool2d_padding);
    ("min_pool1d basic", `Quick, test_min_pool1d_basic);
    ("min_pool1d stride", `Quick, test_min_pool1d_stride);
    ("min_pool2d basic", `Quick, test_min_pool2d_basic);
    ("min_pool2d stride", `Quick, test_min_pool2d_stride);
    ("min_pool2d padding", `Quick, test_min_pool2d_padding);
    ("min_pool2d uint8", `Quick, test_min_pool2d_uint8);
    ("avg_pool1d basic", `Quick, test_avg_pool1d_basic);
    ("avg_pool2d basic", `Quick, test_avg_pool2d_basic);
    ("pool batch", `Quick, test_pool_batch);
    ("pool multichannel", `Quick, test_pool_multichannel);
    ("pool edge cases", `Quick, test_pool_edge_cases);
  ]

let suite =
  [
    ("Neural Net :: Convolution", convolution_tests);
    ("Neural Net :: Pooling", pooling_tests);
  ]

let () = Alcotest.run "Nx Neural Net" suite
