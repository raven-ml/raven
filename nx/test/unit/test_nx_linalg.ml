(* Linear algebra tests for Nx *)

open Alcotest

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Support = Test_nx_support.Make (Backend)
  module Nx = Support.Nx
  open Support

  (* ───── Matrix Multiply Tests ───── *)

  let test_matmul_2d_2d ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let b = Nx.create ctx Nx.float32 [| 4; 5 |] (Array.init 20 float_of_int) in
    let result = Nx.matmul a b in
    check_shape "matmul 2d x 2d shape" [| 3; 5 |] result;
    (* Check a few values *)
    check (float 1e-6) "matmul[0,0]" 70.0 (Nx.unsafe_get [ 0; 0 ] result);
    check (float 1e-6) "matmul[2,4]" 462.0 (Nx.unsafe_get [ 2; 4 ] result)

  let test_matmul_1d_1d ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
    let result = Nx.matmul a b in
    check_t "matmul 1d x 1d" [||] [| 32.0 |] result

  let test_matmul_1d_2d ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let result = Nx.matmul a b in
    check_t "matmul 1d x 2d" [| 4 |] [| 32.; 38.; 44.; 50. |] result

  let test_matmul_2d_1d ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let b = Nx.create ctx Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
    let result = Nx.matmul a b in
    check_t "matmul 2d x 1d" [| 3 |] [| 20.; 60.; 100. |] result

  let test_matmul_batch ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int)
    in
    let b =
      Nx.create ctx Nx.float32 [| 2; 4; 2 |] (Array.init 16 float_of_int)
    in
    let result = Nx.matmul a b in
    check_shape "matmul batch shape" [| 2; 3; 2 |] result;
    (* Check first batch *)
    check (float 1e-6) "batch[0,0,0]" 28.0 (Nx.unsafe_get [ 0; 0; 0 ] result);
    check (float 1e-6) "batch[0,0,1]" 34.0 (Nx.unsafe_get [ 0; 0; 1 ] result)

  let test_matmul_broadcast_batch ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 1; 3; 4 |] (Array.init 12 float_of_int)
    in
    let b =
      Nx.create ctx Nx.float32 [| 5; 4; 2 |] (Array.init 40 float_of_int)
    in
    let result = Nx.matmul a b in
    check_shape "matmul broadcast batch shape" [| 5; 3; 2 |] result

  let test_matmul_2d_3d_broadcast ctx () =
    (*
     * Test case: A (2D) @ B (3D)
     * A shape: (2, 3) - to be broadcasted
     * B shape: (4, 3, 2) - batched tensor
     * Expected output shape: (4, 2, 2)
     *)

    (* A is a single 2x3 matrix *)
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in

    (* B is a batch of four 3x2 matrices *)
    let b =
      Nx.create ctx Nx.float32 [| 4; 3; 2 |]
        [|
          (* Batch 0 *)
          1.;
          2.;
          3.;
          4.;
          5.;
          6.;
          (* Batch 1 *)
          7.;
          8.;
          9.;
          10.;
          11.;
          12.;
          (* Batch 2 *)
          1.;
          0.;
          0.;
          1.;
          1.;
          0.;
          (* Batch 3 *)
          0.;
          1.;
          1.;
          0.;
          0.;
          1.;
        |]
    in

    (* Perform the matmul *)
    let result = Nx.matmul a b in

    (* Check shape *)
    check_shape "matmul 2d @ 3d shape" [| 4; 2; 2 |] result;

    (*
     * Manually calculate the expected result:
     *
     * A = [[1, 2, 3],
     *      [4, 5, 6]]
     *
     * B[0] = [[1, 2], [3, 4], [5, 6]]
     * A @ B[0] = [[22, 28], [49, 64]]
     *
     * B[1] = [[7, 8], [9, 10], [11, 12]]
     * A @ B[1] = [[58, 64], [139, 154]]
     *
     * B[2] = [[1, 0], [0, 1], [1, 0]]
     * A @ B[2] = [[4, 2], [10, 5]]
     *
     * B[3] = [[0, 1], [1, 0], [0, 1]]
     * A @ B[3] = [[2, 4], [5, 10]]
     *)

    (* Check batch 0 *)
    check (float 1e-6) "batch 0 [0,0]" 22. (Nx.unsafe_get [ 0; 0; 0 ] result);
    check (float 1e-6) "batch 0 [0,1]" 28. (Nx.unsafe_get [ 0; 0; 1 ] result);
    check (float 1e-6) "batch 0 [1,0]" 49. (Nx.unsafe_get [ 0; 1; 0 ] result);
    check (float 1e-6) "batch 0 [1,1]" 64. (Nx.unsafe_get [ 0; 1; 1 ] result);

    (* Check batch 1 *)
    check (float 1e-6) "batch 1 [0,0]" 58. (Nx.unsafe_get [ 1; 0; 0 ] result);
    check (float 1e-6) "batch 1 [0,1]" 64. (Nx.unsafe_get [ 1; 0; 1 ] result);
    check (float 1e-6) "batch 1 [1,0]" 139. (Nx.unsafe_get [ 1; 1; 0 ] result);
    check (float 1e-6) "batch 1 [1,1]" 154. (Nx.unsafe_get [ 1; 1; 1 ] result);

    (* Check batch 2 *)
    check (float 1e-6) "batch 2 [0,0]" 4. (Nx.unsafe_get [ 2; 0; 0 ] result);
    check (float 1e-6) "batch 2 [0,1]" 2. (Nx.unsafe_get [ 2; 0; 1 ] result);
    check (float 1e-6) "batch 2 [1,0]" 10. (Nx.unsafe_get [ 2; 1; 0 ] result);
    check (float 1e-6) "batch 2 [1,1]" 5. (Nx.unsafe_get [ 2; 1; 1 ] result);

    (* Check batch 3 *)
    check (float 1e-6) "batch 3 [0,0]" 2. (Nx.unsafe_get [ 3; 0; 0 ] result);
    check (float 1e-6) "batch 3 [0,1]" 4. (Nx.unsafe_get [ 3; 0; 1 ] result);
    check (float 1e-6) "batch 3 [1,0]" 5. (Nx.unsafe_get [ 3; 1; 0 ] result);
    check (float 1e-6) "batch 3 [1,1]" 10. (Nx.unsafe_get [ 3; 1; 1 ] result)

  let test_matmul_shape_error ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let b = Nx.create ctx Nx.float32 [| 5; 6 |] (Array.init 30 float_of_int) in
    check_raises "matmul shape error"
      (Invalid_argument
         "dot: cannot contract [3,4] (last axis: 4) to [5,6] (axis 0: 5) (size \
          4\226\137\1605)") (fun () -> ignore (Nx.matmul a b))

  let test_matmul_empty ctx () =
    let a = Nx.create ctx Nx.float32 [| 0; 5 |] [||] in
    let b = Nx.create ctx Nx.float32 [| 5; 3 |] (Array.init 15 float_of_int) in
    let result = Nx.matmul a b in
    check_shape "matmul empty shape" [| 0; 3 |] result

  let test_matmul_transpose_optimization ctx () =
    (* Test that matmul handles transposed inputs efficiently *)
    let a = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let b = Nx.create ctx Nx.float32 [| 5; 4 |] (Array.init 20 float_of_int) in
    let bt = Nx.transpose b in
    let result = Nx.matmul a bt in
    check_shape "matmul with transpose" [| 3; 5 |] result

  (* ───── Dot Product Tests ───── *)

  let test_dot_1d_1d ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
    let result = Nx.dot a b in
    check_t "dot 1d x 1d" [||] [| 32.0 |] result

  let test_dot_2d_1d ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 7.; 8.; 9. |] in
    let result = Nx.dot a b in
    check_t "dot 2d x 1d" [| 2 |] [| 50.; 122. |] result

  let test_dot_2d_2d ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b =
      Nx.create ctx Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    let result = Nx.dot a b in
    check_t "dot 2d x 2d" [| 2; 2 |] [| 58.; 64.; 139.; 154. |] result

  let test_dot_higher_d ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 2; 2; 3 |] (Array.init 12 float_of_int)
    in
    let b = Nx.create ctx Nx.float32 [| 3; 2 |] (Array.init 6 float_of_int) in
    let result = Nx.dot a b in
    check_t "dot higher-d" [| 2; 2; 2 |]
      [| 10.; 13.; 28.; 40.; 46.; 67.; 64.; 94. |]
      result

  let test_dot_scalar_result ctx () =
    (* Ensure dot product of 1D arrays returns proper scalar *)
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
    let result = Nx.dot a b in
    check_shape "dot scalar shape" [||] result;
    check (float 1e-6) "dot scalar value" 32.0 (Nx.unsafe_get [] result)

  (* ───── Convolution Tests ───── *)

  let test_convolve1d_basic ctx () =
    (* Basic 1D convolution - proper tensor format: (batch, channels, width) *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
    let result = Nx.convolve1d input kernel in
    (* NumPy convolve flips kernel, so result is [2., 2., 2.] *)
    check_t "convolve1d basic" [| 1; 1; 3 |] [| 2.; 2.; 2. |] result

  let test_convolve1d_padding_modes ctx () =
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in

    (* Valid padding (default) - output size = input - kernel + 1 *)
    let valid = Nx.convolve1d ~padding_mode:`Valid input kernel in
    check_t "convolve1d valid padding" [| 1; 1; 3 |] [| 6.; 9.; 12. |] valid;

    (* Same padding - output size = input size *)
    let same = Nx.convolve1d ~padding_mode:`Same input kernel in
    check_t "convolve1d same padding" [| 1; 1; 5 |] [| 3.; 6.; 9.; 12.; 9. |]
      same;

    (* Full padding - output size = input + kernel - 1 *)
    let full = Nx.convolve1d ~padding_mode:`Full input kernel in
    check_t "convolve1d full padding" [| 1; 1; 7 |]
      [| 1.; 3.; 6.; 9.; 12.; 9.; 5. |]
      full

  let test_convolve1d_stride ctx () =
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 8 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in

    (* Stride 2 *)
    let result = Nx.convolve1d ~stride:2 input kernel in
    check_t "convolve1d stride 2" [| 1; 1; 3 |] [| 6.; 12.; 18. |] result

  let test_convolve1d_dilation ctx () =
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 7 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7. |]
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; 1. |] in

    (* Dilation 2 - kernel elements are 2 positions apart *)
    (* With dilation=2, effective kernel size = 2*(3-1)+1 = 5 *)
    (* Output size = 7 - 5 + 1 = 3 *)
    let result = Nx.convolve1d ~dilation:2 input kernel in
    check_t "convolve1d dilation 2" [| 1; 1; 3 |] [| 6.; 8.; 10. |] result

  let test_convolve1d_groups ctx () =
    (* Groups=2: split 4 channels into 2 groups of 2 channels each *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 4; 4 |] (Array.init 16 float_of_int)
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 2; 2; 2 |]
        [| 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1. |]
    in

    let result = Nx.convolve1d ~groups:2 input kernel in
    check_shape "convolve1d groups shape" [| 1; 2; 3 |] result

  let test_convolve1d_bias ctx () =
    let input = Nx.create ctx Nx.float32 [| 1; 1; 3 |] [| 1.; 2.; 3. |] in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 2 |] [| 1.; 1. |] in
    let bias = Nx.create ctx Nx.float32 [| 1 |] [| 10. |] in

    let result = Nx.convolve1d ~bias input kernel in
    check_t "convolve1d with bias" [| 1; 1; 2 |] [| 13.; 15. |] result

  let test_correlate1d_basic ctx () =
    (* Basic 1D correlation - kernel is not flipped *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
    let result = Nx.correlate1d input kernel in
    check_t "correlate1d basic" [| 1; 1; 3 |] [| -2.; -2.; -2. |] result

  let test_convolve2d_basic ctx () =
    (* Basic 2D convolution *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in
    let result = Nx.convolve2d input kernel in
    check_t "convolve2d basic" [| 1; 1; 2; 2 |] [| 45.; 54.; 81.; 90. |] result

  let test_convolve2d_padding_modes ctx () =
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |]
    in

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

  let test_convolve2d_stride ctx () =
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 5; 5 |] (Array.init 25 float_of_int)
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in

    (* Stride (2,2) *)
    let result = Nx.convolve2d ~stride:(2, 2) input kernel in
    check_shape "convolve2d stride shape" [| 1; 1; 2; 2 |] result;
    check_t "convolve2d stride values" [| 1; 1; 2; 2 |]
      [| 54.; 72.; 144.; 162. |] result

  let test_convolve2d_dilation ctx () =
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 5; 5 |] (Array.init 25 float_of_int)
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |]
        [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1. |]
    in
    (* Only corners *)

    (* Dilation 2 - effective kernel size becomes 5x5 *)
    let result = Nx.convolve2d ~dilation:(2, 2) input kernel in
    check_shape "convolve2d dilation shape" [| 1; 1; 1; 1 |] result;
    check_t "convolve2d dilation value" [| 1; 1; 1; 1 |] [| 24. |] result

  let test_convolve2d_multi_channel ctx () =
    (* Multi-channel convolution: 3 input channels, 2 output channels *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 3; 4; 4 |] (Array.init 48 float_of_int)
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 2; 3; 3; 3 |] (Array.make 54 1.0)
    in

    let result = Nx.convolve2d input kernel in
    check_shape "convolve2d multi-channel shape" [| 1; 2; 2; 2 |] result

  let test_convolve2d_winograd_eligible ctx () =
    (* Test a convolution that should trigger Winograd optimization: - 3x3
       kernel - stride 1 - groups 1 This specific test case helps catch reshape
       issues in Winograd path *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 8; 8 |] (Array.init 64 float_of_int)
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in

    (* This should use Winograd optimization *)
    let result = Nx.convolve2d ~stride:(1, 1) input kernel in
    check_shape "convolve2d Winograd shape" [| 1; 1; 6; 6 |] result;

    (* Verify the computation is correct *)
    (* Each 3x3 window sums to 9 times the sum of its elements *)
    let expected_00 = 0. +. 1. +. 2. +. 8. +. 9. +. 10. +. 16. +. 17. +. 18. in
    check (float 1e-5) "convolve2d Winograd [0,0,0,0]" expected_00
      (Nx.unsafe_get [ 0; 0; 0; 0 ] result)

  let test_convolve2d_groups_winograd ctx () =
    (* Test grouped convolution with parameters that might trigger Winograd but
       should be handled correctly *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 2; 8; 8 |] (Array.init 128 float_of_int)
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 2; 1; 3; 3 |] (Array.make 18 1.0)
    in

    (* Groups=2 should disable Winograd optimization *)
    let result = Nx.convolve2d ~groups:2 ~stride:(1, 1) input kernel in
    check_shape "convolve2d groups Winograd shape" [| 1; 2; 6; 6 |] result

  let test_convolve2d_non_contiguous_input ctx () =
    (* Test convolution with non-contiguous input (e.g., from transpose) *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 4; 4; 1 |] (Array.init 16 float_of_int)
    in
    let input_transposed = Nx.transpose ~axes:[| 0; 3; 1; 2 |] input in
    (* Now [1; 1; 4; 4] but non-contiguous *)
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |] (Array.make 9 1.0) in

    let result = Nx.convolve2d input_transposed kernel in
    check_shape "convolve2d non-contiguous shape" [| 1; 1; 2; 2 |] result;
    check_t "convolve2d non-contiguous values" [| 1; 1; 2; 2 |]
      [| 45.; 54.; 81.; 90. |] result

  let test_correlate2d_basic ctx () =
    (* Basic 2D correlation *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 4; 4 |] (Array.init 16 float_of_int)
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |]
        [| 1.; 0.; -1.; 0.; 0.; 0.; -1.; 0.; 1. |]
    in
    let result = Nx.correlate2d input kernel in
    check_shape "correlate2d shape" [| 1; 1; 2; 2 |] result

  let test_convolve_invalid_shapes ctx () =
    (* Test various invalid shape combinations *)
    let input_1d =
      Nx.create ctx Nx.float32 [| 1; 2; 5 |] (Array.init 10 float_of_int)
    in
    let kernel_1d =
      Nx.create ctx Nx.float32 [| 1; 3; 3 |] (Array.init 9 float_of_int)
    in

    check_invalid_arg "convolve1d channel mismatch"
      "correlate_nd: invalid channel configuration (2 \226\137\160 1\195\1513)\n\
       hint: expected 3 channels for 1 groups with 3 channels each" (fun () ->
        ignore (Nx.convolve1d input_1d kernel_1d))

  let test_convolve_empty_input ctx () =
    (* Empty input handling - empty on spatial dimension *)
    let input = Nx.create ctx Nx.float32 [| 1; 1; 0 |] [||] in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in
    let result = Nx.convolve1d input kernel in
    check_shape "convolve1d empty input" [| 1; 1; 0 |] result

  let test_convolve_single_element_kernel ctx () =
    (* Single element kernel acts as scaling *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let kernel = Nx.create ctx Nx.float32 [| 1; 1; 1 |] [| 2.0 |] in
    let result = Nx.convolve1d input kernel in
    check_t "convolve1d single kernel" [| 1; 1; 5 |] [| 2.; 4.; 6.; 8.; 10. |]
      result

  let test_convolve2d_pool_reshape_edge_case ctx () =
    (* Test case that might trigger the reshape error seen in sanity tests This
       tests the pool operation's reshape from [6; 6; 1; 1; 2; 2] to [6; 6;
       4] *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 1; 6; 6 |] (Array.init 36 float_of_int)
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |]
    in

    (* Use stride 1 to get output shape [1; 1; 5; 5] *)
    let result = Nx.convolve2d ~stride:(1, 1) input kernel in
    check_shape "convolve2d pool edge case shape" [| 1; 1; 5; 5 |] result;

    (* Verify first output value: sum of top-left 2x2 window *)
    let expected_00 = 0. +. 1. +. 6. +. 7. in
    check (float 1e-5) "convolve2d pool edge case [0,0,0,0]" expected_00
      (Nx.unsafe_get [ 0; 0; 0; 0 ] result)

  let test_convolve2d_groups_reshape_issue ctx () =
    (* Test grouped convolution that might cause reshape issues in pooling This
       specifically tests the optimized path for groups > 1 *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 4; 6; 6 |] (Array.init 144 float_of_int)
    in
    let kernel =
      Nx.create ctx Nx.float32 [| 4; 2; 2; 2 |] (Array.make 32 1.0)
    in

    (* Groups=2: each group has 2 input channels and 2 output channels *)
    let result = Nx.convolve2d ~groups:2 ~stride:(1, 1) input kernel in
    check_shape "convolve2d groups reshape shape" [| 1; 4; 5; 5 |] result

  let test_convolve2d_dilated_non_contiguous ctx () =
    (* Test dilated convolution with non-contiguous tensor This can trigger
       complex reshapes in pool_dilated_path *)
    let input =
      Nx.create ctx Nx.float32 [| 1; 5; 5; 1 |] (Array.init 25 float_of_int)
    in
    let input_perm = Nx.transpose ~axes:[| 0; 3; 1; 2 |] input in
    (* Now [1; 1; 5; 5] non-contiguous *)
    let kernel =
      Nx.create ctx Nx.float32 [| 1; 1; 3; 3 |]
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

  let test_correlate2d_winograd_sanity_case ctx () =
    (* Test the exact scenario from sanity tests that triggers the reshape bug *)
    (* This matches the failing sanity test exactly: correlate2d with 1x1x5x5 input, 1x1x3x3 kernel, all ones *)
    let x = Nx.ones ctx Nx.float32 [| 1; 1; 5; 5 |] in
    let w = Nx.ones ctx Nx.float32 [| 1; 1; 3; 3 |] in

    (* This correlation should work and produce 3x3 output with all 9s *)
    (* Note: correlate2d can also trigger Winograd when kernel is 3x3, stride 1, groups 1 *)
    let y = Nx.correlate2d x w in

    (* The expected result is a 3x3 output where each element is 9.0 *)
    check_t ~eps:1e-6 "correlate2d values" [| 1; 1; 3; 3 |]
      [| 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9.; 9. |]
      y

  (*  ─────  Solve Inverse Tests  ─────  *)
  (* Note: These functions are not exposed in nx.ml, so tests are commented out *)

  (* let test_solve_identity ctx () = (* solve(I, b) = b *) let identity =
     Nx.eye Nx.float32 3 in let b = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.;
     3. |] in let x = Nx.solve identity b in check_t "solve identity" [| 3 |] [|
     1.; 2.; 3. |] x

     let test_solve_simple ctx () = (* Simple 2x2 system *) let a = Nx.create
     ctx Nx.float32 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in let b = Nx.create ctx
     Nx.float32 [| 2 |] [| 9.; 8. |] in let x = Nx.solve a b in (* Verify A @ x
     = b *) let result = Nx.dot a x in check_approx_equal "solve simple" b
     result

     let test_solve_batch ctx () = (* Multiple systems *) let a = Nx.create ctx
     Nx.float32 [| 2; 3; 3 |] [| (* First system *) 1.; 0.; 0.; 0.; 1.; 0.; 0.;
     0.; 1.; (* Second system *) 2.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 2. |] in let b
     = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6. |] in let x
     = Nx.solve a b in check_shape "solve batch shape" [| 2; 3 |] x

     let test_solve_singular ctx () = (* Singular matrix *) let a = Nx.create
     ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in let b = Nx.create ctx
     Nx.float32 [| 2 |] [| 1.; 2. |] in check_invalid_arg "solve singular"
     "solve: matrix is singular" (fun () -> ignore (Nx.solve a b))

     let test_solve_non_square ctx () = (* Non-square matrix *) let a =
     Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in let b =
     Nx.create ctx Nx.float32 [| 2 |] [| 1.; 2. |] in check_invalid_arg "solve
     non-square" "solve: coefficient matrix must be square" (fun () -> ignore
     (Nx.solve a b))

     let test_inv_identity ctx () = (* inv(I) = I *) let identity = Nx.eye
     Nx.float32 3 in let inv = Nx.inv identity in check_approx_equal "inv
     identity" identity inv

     let test_inv_inverse ctx () = (* inv(inv(A)) = A *) let a = Nx.create ctx
     Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in let inv_a = Nx.inv a in let
     inv_inv_a = Nx.inv inv_a in check_approx_equal "inv inverse" a inv_inv_a

     let test_inv_singular ctx () = (* Singular matrix *) let a = Nx.create ctx
     Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in check_invalid_arg "inv
     singular" "inv: matrix is singular" (fun () -> ignore (Nx.inv a)) *)

  (*  ─────  Decomposition Tests  ─────  *)
  (* Note: These functions are not exposed in nx.ml, so tests are commented out *)

  (* let test_qr_shape ctx () = let a = Nx.create ctx Nx.float32 [| 4; 3 |]
     (Array.init 12 float_of_int) in let q, r = Nx.qr a in check_shape "qr q
     shape" [| 4; 4 |] q; check_shape "qr r shape" [| 4; 3 |] r

     let test_qr_property ctx () = (* Q @ R = A *) let a = Nx.create ctx
     Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let q, r
     = Nx.qr a in let reconstructed = Nx.matmul q r in check_approx_equal "qr
     property" a reconstructed

     let test_qr_orthogonal ctx () = (* Q.T @ Q = I *) let a = Nx.create ctx
     Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let q, _
     = Nx.qr a in let qt_q = Nx.matmul (Nx.transpose q) q in let identity =
     Nx.eye Nx.float32 3 in check_approx_equal "qr orthogonal" identity qt_q

     let test_svd_shape ctx () = let a = Nx.create ctx Nx.float32 [| 3; 4 |]
     (Array.init 12 float_of_int) in let u, s, v = Nx.svd a in check_shape "svd
     u shape" [| 3; 3 |] u; check_shape "svd s shape" [| 3 |] s; check_shape
     "svd v shape" [| 4; 4 |] v

     let test_svd_property ctx () = (* U @ S @ V.T = A *) let a = Nx.create ctx
     Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let u,
     s, v = Nx.svd a in (* Create diagonal matrix from s *) let s_diag =
     Nx.zeros Nx.float32 [| 3; 3 |] in for i = 0 to 2 do Nx.set_item [i; i]
     s_diag (Nx.unsafe_get [i] s) done; let reconstructed = Nx.matmul u
     (Nx.matmul s_diag (Nx.transpose v)) in check_approx_equal "svd property" a
     reconstructed

     let test_cholesky_posdef ctx () = (* Create positive definite matrix: A.T @
     A *) let a = Nx.create ctx Nx.float32 [| 3; 3 |] [| 1.; 0.; 0.; 1.; 1.; 0.;
     1.; 1.; 1. |] in let posdef = Nx.matmul (Nx.transpose a) a in let l =
     Nx.cholesky posdef in check_shape "cholesky shape" [| 3; 3 |] l

     let test_cholesky_property ctx () = (* L @ L.T = A *) let a = Nx.create ctx
     Nx.float32 [| 2; 2 |] [| 1.; 0.; 1.; 1. |] in let posdef = Nx.matmul
     (Nx.transpose a) a in let l = Nx.cholesky posdef in let reconstructed =
     Nx.matmul l (Nx.transpose l) in check_approx_equal "cholesky property"
     posdef reconstructed

     let test_eig_shape ctx () = let a = Nx.create ctx Nx.float32 [| 3; 3 |] [|
     1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let eigenvalues, eigenvectors =
     Nx.eig a in check_shape "eig eigenvalues shape" [| 3 |] eigenvalues;
     check_shape "eig eigenvectors shape" [| 3; 3 |] eigenvectors

     let test_eig_property ctx () = (* A @ v = lambda * v *) let a = Nx.create
     ctx Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in let eigenvalues,
     eigenvectors = Nx.eig a in (* Check first eigenvector *) let v1 = Nx.get
     (Nx.LR [Nx.All; [0; 1]]) eigenvectors in let lambda1 = Nx.unsafe_get [0]
     eigenvalues in let av1 = Nx.dot a v1 in let lambda_v1 = Nx.mul_s v1 lambda1
     in check_approx_equal "eig property" av1 lambda_v1 *)

  (* ───── Norm Tests ───── *)

  (* let test_norm_vector_1 ctx () = let v = Nx.create ctx Nx.float32 [| 4 |] [|
     -1.; 2.; -3.; 4. |] in let result = Nx.norm ~ord:(`L 1.) v in check_t "norm
     L1" [||] [| 10.0 |] result *)

  (* let test_norm_vector_2 ctx () = let v = Nx.create ctx Nx.float32 [| 3 |] [|
     3.; 4.; 0. |] in let result = Nx.norm v in (* Default is L2 *) check_t
     "norm L2" [||] [| 5.0 |] result *)

  (* let test_norm_vector_inf ctx () = let v = Nx.create ctx Nx.float32 [| 4 |]
     [| -1.; 2.; -5.; 4. |] in let result = Nx.norm ~ord:`Inf v in check_t "norm
     Linf" [||] [| 5.0 |] result *)

  (* let test_norm_matrix_fro ctx () = let m = Nx.create ctx Nx.float32 [| 2; 2
     |] [| 1.; 2.; 3.; 4. |] in let result = Nx.norm ~ord:`Fro m in check_t
     ~eps:1e-5 "norm Frobenius" [||] [| 5.477226 |] result *)

  (* let test_norm_matrix_1 ctx () = let m = Nx.create ctx Nx.float32 [| 2; 2 |]
     [| 1.; -2.; 3.; 4. |] in let result = Nx.norm ~ord:(`L 1.) m in check_t
     "norm matrix L1" [||] [| 6.0 |] result *)

  (* let test_norm_axis ctx () = let m = Nx.create ctx Nx.float32 [| 2; 3 |] [|
     1.; 2.; 3.; 4.; 5.; 6. |] in let result = Nx.norm ~axis:[1] m in check_t
     ~eps:1e-5 "norm along axis" [| 2 |] [| 3.741657; 8.774964 |] result *)

  (* let test_norm_empty ctx () = let v = Nx.create ctx Nx.float32 [| 0 |] [||]
     in let result = Nx.norm v in check_t "norm empty" [||] [| 0.0 |] result *)

  (* ───── Linear Algebra Utilities ───── *)

  (* let test_det_2x2 ctx () = let a = Nx.create ctx Nx.float32 [| 2; 2 |] [|
     3.; 8.; 4.; 6. |] in let det = Nx.det a in check_t "det 2x2" [||] [| -14.0
     |] det

     let test_det_singular ctx () = let a = Nx.create ctx Nx.float32 [| 2; 2 |]
     [| 1.; 2.; 2.; 4. |] in let det = Nx.det a in check_t ~eps:1e-6 "det
     singular" [||] [| 0.0 |] det *)

  (* let test_trace ctx () = let a = Nx.create ctx Nx.float32 [| 3; 3 |] [| 1.;
     2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |] in let tr = Nx.trace a in check_t "trace"
     [||] [| 15.0 |] tr *)

  (* let test_diag_extract ctx () = let a = Nx.create ctx Nx.float32 [| 3; 3 |]
     [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |] in let diag = Nx.diag a in check_t
     "diag extract" [| 3 |] [| 1.; 5.; 9. |] diag *)

  (* let test_diag_create ctx () = let v = Nx.create ctx Nx.float32 [| 3 |] [|
     1.; 2.; 3. |] in let result = Nx.diag v in check_t "diag create" [| 3; 3 |]
     [| 1.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 3. |] result *)

  (* let test_tril_triu ctx () = let a = Nx.create ctx Nx.float32 [| 3; 3 |] [|
     1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |] in

     let lower = Nx.tril a in check_t "tril" [| 3; 3 |] [| 1.; 0.; 0.; 4.; 5.;
     0.; 7.; 8.; 9. |] lower;

     let upper = Nx.triu a in check_t "triu" [| 3; 3 |] [| 1.; 2.; 3.; 0.; 5.;
     6.; 0.; 0.; 9. |] upper *)

  (* Test Suite Organization *)

  let matmul_tests ctx =
    [
      ("matmul 2d x 2d", `Quick, test_matmul_2d_2d ctx);
      ("matmul 1d x 1d", `Quick, test_matmul_1d_1d ctx);
      ("matmul 1d x 2d", `Quick, test_matmul_1d_2d ctx);
      ("matmul 2d x 1d", `Quick, test_matmul_2d_1d ctx);
      ("matmul batch", `Quick, test_matmul_batch ctx);
      ("matmul broadcast batch", `Quick, test_matmul_broadcast_batch ctx);
      ("matmul 2d @ 3d broadcast", `Quick, test_matmul_2d_3d_broadcast ctx);
      ("matmul shape error", `Quick, test_matmul_shape_error ctx);
      ("matmul empty", `Quick, test_matmul_empty ctx);
      ( "matmul transpose optimization",
        `Quick,
        test_matmul_transpose_optimization ctx );
    ]

  let dot_tests ctx =
    [
      ("dot 1d x 1d", `Quick, test_dot_1d_1d ctx);
      ("dot 2d x 1d", `Quick, test_dot_2d_1d ctx);
      ("dot 2d x 2d", `Quick, test_dot_2d_2d ctx);
      ("dot higher-d", `Quick, test_dot_higher_d ctx);
      ("dot scalar result", `Quick, test_dot_scalar_result ctx);
    ]

  let convolution_tests ctx =
    [
      ("convolve1d basic", `Quick, test_convolve1d_basic ctx);
      ("convolve1d padding modes", `Quick, test_convolve1d_padding_modes ctx);
      ("convolve1d stride", `Quick, test_convolve1d_stride ctx);
      ("convolve1d dilation", `Quick, test_convolve1d_dilation ctx);
      ("convolve1d groups", `Quick, test_convolve1d_groups ctx);
      ("convolve1d bias", `Quick, test_convolve1d_bias ctx);
      ("correlate1d basic", `Quick, test_correlate1d_basic ctx);
      ("convolve2d basic", `Quick, test_convolve2d_basic ctx);
      ("convolve2d padding modes", `Quick, test_convolve2d_padding_modes ctx);
      ("convolve2d stride", `Quick, test_convolve2d_stride ctx);
      ("convolve2d dilation", `Quick, test_convolve2d_dilation ctx);
      ("convolve2d multi-channel", `Quick, test_convolve2d_multi_channel ctx);
      ( "convolve2d winograd eligible",
        `Quick,
        test_convolve2d_winograd_eligible ctx );
      ("convolve2d groups winograd", `Quick, test_convolve2d_groups_winograd ctx);
      ( "convolve2d non-contiguous input",
        `Quick,
        test_convolve2d_non_contiguous_input ctx );
      ( "convolve2d pool reshape edge case",
        `Quick,
        test_convolve2d_pool_reshape_edge_case ctx );
      ( "convolve2d groups reshape issue",
        `Quick,
        test_convolve2d_groups_reshape_issue ctx );
      ( "convolve2d dilated non-contiguous",
        `Quick,
        test_convolve2d_dilated_non_contiguous ctx );
      ("convolve invalid shapes", `Quick, test_convolve_invalid_shapes ctx);
      ("convolve empty input", `Quick, test_convolve_empty_input ctx);
      ( "convolve single element kernel",
        `Quick,
        test_convolve_single_element_kernel ctx );
      ("correlate2d basic", `Quick, test_correlate2d_basic ctx);
      ( "correlate2d winograd sanity case",
        `Quick,
        test_correlate2d_winograd_sanity_case ctx );
    ]

  let solve_inverse_tests _ctx =
    [ (* ("solve identity", `Quick, test_solve_identity ctx); *)
      (* ("solve simple", `Quick, test_solve_simple ctx); *)
      (* ("solve batch", `Quick, test_solve_batch ctx); *)
      (* ("solve singular", `Quick, test_solve_singular ctx); *)
      (* ("solve non-square", `Quick, test_solve_non_square ctx); *)
      (* ("inv identity", `Quick, test_inv_identity ctx); *)
      (* ("inv inverse", `Quick, test_inv_inverse ctx); *)
      (* ("inv singular", `Quick, test_inv_singular ctx); *) ]

  let decomposition_tests _ctx =
    [ (* ("qr shape", `Quick, test_qr_shape ctx); *)
      (* ("qr property", `Quick, test_qr_property ctx); *)
      (* ("qr orthogonal", `Quick, test_qr_orthogonal ctx); *)
      (* ("svd shape", `Quick, test_svd_shape ctx); *)
      (* ("svd property", `Quick, test_svd_property ctx); *)
      (* ("cholesky posdef", `Quick, test_cholesky_posdef ctx); *)
      (* ("cholesky property", `Quick, test_cholesky_property ctx); *)
      (* ("eig shape", `Quick, test_eig_shape ctx); *)
      (* ("eig property", `Quick, test_eig_property ctx); *) ]

  let norm_tests _ctx =
    [ (* ("norm vector L1", `Quick, test_norm_vector_1 ctx); *)
      (* ("norm vector L2", `Quick, test_norm_vector_2 ctx); *)
      (* ("norm vector Linf", `Quick, test_norm_vector_inf ctx); *)
      (* ("norm matrix Frobenius", `Quick, test_norm_matrix_fro ctx); *)
      (* ("norm matrix L1", `Quick, test_norm_matrix_1 ctx); *)
      (* ("norm axis", `Quick, test_norm_axis ctx); *)
      (* ("norm empty", `Quick, test_norm_empty ctx); *) ]

  let utility_tests _ctx =
    [ (* ("det 2x2", `Quick, test_det_2x2 ctx); *)
      (* ("det singular", `Quick, test_det_singular ctx); *)
      (* ("trace", `Quick, test_trace ctx); *)
      (* ("diag extract", `Quick, test_diag_extract ctx); *)
      (* ("diag create", `Quick, test_diag_create ctx); *)
      (* ("tril triu", `Quick, test_tril_triu ctx); *) ]

  let suite backend_name ctx =
    [
      ("Linalg :: " ^ backend_name ^ " Matrix Multiply", matmul_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Dot Product", dot_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Convolution", convolution_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Solve/Inverse", solve_inverse_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Decompositions", decomposition_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Norms", norm_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Utilities", utility_tests ctx);
    ]
end
