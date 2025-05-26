(* Linear algebra tests for Nx *)

open Test_nx_support
open Alcotest

(* ───── Matrix Multiply Tests ───── *)

let test_matmul_2d_2d () =
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 4; 5 |] (Array.init 20 float_of_int) in
  let result = Nx.matmul a b in
  check_shape "matmul 2d x 2d shape" [| 3; 5 |] result;
  (* Check a few values *)
  check (float 1e-6) "matmul[0,0]" 70.0 (Nx.get_item [ 0; 0 ] result);
  check (float 1e-6) "matmul[2,4]" 462.0 (Nx.get_item [ 2; 4 ] result)

let test_matmul_1d_1d () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let result = Nx.matmul a b in
  check_t "matmul 1d x 1d" [||] [| 32.0 |] result

let test_matmul_1d_2d () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let result = Nx.matmul a b in
  check_t "matmul 1d x 2d" [| 4 |] [| 32.; 38.; 44.; 50. |] result

let test_matmul_2d_1d () =
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let result = Nx.matmul a b in
  check_t "matmul 2d x 1d" [| 3 |] [| 20.; 60.; 100. |] result

let test_matmul_batch () =
  let a = Nx.create Nx.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let b = Nx.create Nx.float32 [| 2; 4; 2 |] (Array.init 16 float_of_int) in
  let result = Nx.matmul a b in
  check_shape "matmul batch shape" [| 2; 3; 2 |] result;
  (* Check first batch *)
  check (float 1e-6) "batch[0,0,0]" 28.0 (Nx.get_item [ 0; 0; 0 ] result);
  check (float 1e-6) "batch[0,0,1]" 34.0 (Nx.get_item [ 0; 0; 1 ] result)

let test_matmul_broadcast_batch () =
  let a = Nx.create Nx.float32 [| 1; 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 5; 4; 2 |] (Array.init 40 float_of_int) in
  let result = Nx.matmul a b in
  check_shape "matmul broadcast batch shape" [| 5; 3; 2 |] result

let test_matmul_shape_error () =
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 5; 6 |] (Array.init 30 float_of_int) in
  check_raises "matmul shape error"
    (Invalid_argument
       "dot: shape mismatch on contracting dimension. x_shape: [3; 4], \
        w_shape: [5; 6]. x_contract_dim_size: 4, w_contract_dim_size: 5")
    (fun () -> ignore (Nx.matmul a b))

let test_matmul_empty () =
  let a = Nx.create Nx.float32 [| 0; 5 |] [||] in
  let b = Nx.create Nx.float32 [| 5; 3 |] (Array.init 15 float_of_int) in
  let result = Nx.matmul a b in
  check_shape "matmul empty shape" [| 0; 3 |] result

let test_matmul_transpose_optimization () =
  (* Test that matmul handles transposed inputs efficiently *)
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 5; 4 |] (Array.init 20 float_of_int) in
  let bt = Nx.transpose b in
  let result = Nx.matmul a bt in
  check_shape "matmul with transpose" [| 3; 5 |] result

(* ───── Dot Product Tests ───── *)

let test_dot_1d_1d () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let result = Nx.dot a b in
  check_t "dot 1d x 1d" [||] [| 32.0 |] result

let test_dot_2d_1d () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 7.; 8.; 9. |] in
  let result = Nx.dot a b in
  check_t "dot 2d x 1d" [| 2 |] [| 50.; 122. |] result

let test_dot_2d_2d () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let result = Nx.dot a b in
  check_t "dot 2d x 2d" [| 2; 2 |] [| 58.; 64.; 139.; 154. |] result

let test_dot_higher_d () =
  let a = Nx.create Nx.float32 [| 2; 2; 3 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 3; 2 |] (Array.init 6 float_of_int) in
  let result = Nx.dot a b in
  check_t "dot higher-d" [| 2; 2; 2 |]
    [| 10.; 13.; 28.; 40.; 46.; 67.; 64.; 94. |]
    result

let test_dot_scalar_result () =
  (* Ensure dot product of 1D arrays returns proper scalar *)
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let result = Nx.dot a b in
  check_shape "dot scalar shape" [||] result;
  check (float 1e-6) "dot scalar value" 32.0 (Nx.get_item [] result)

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
    "Input channels 2 not compatible with groups 1 and weight cin_per_group 3"
    (fun () -> ignore (Nx.convolve1d input_1d kernel_1d))

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

(*  ─────  Solve Inverse Tests  ─────  *)
(* Note: These functions are not exposed in nx.ml, so tests are commented out *)

(* let test_solve_identity () = (* solve(I, b) = b *) let identity = Nx.eye
   Nx.float32 3 in let b = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in let
   x = Nx.solve identity b in check_t "solve identity" [| 3 |] [| 1.; 2.; 3. |]
   x

   let test_solve_simple () = (* Simple 2x2 system *) let a = Nx.create
   Nx.float32 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in let b = Nx.create Nx.float32 [|
   2 |] [| 9.; 8. |] in let x = Nx.solve a b in (* Verify A @ x = b *) let
   result = Nx.dot a x in check_approx_equal "solve simple" b result

   let test_solve_batch () = (* Multiple systems *) let a = Nx.create Nx.float32
   [| 2; 3; 3 |] [| (* First system *) 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.; (*
   Second system *) 2.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 2. |] in let b = Nx.create
   Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6. |] in let x = Nx.solve a b in
   check_shape "solve batch shape" [| 2; 3 |] x

   let test_solve_singular () = (* Singular matrix *) let a = Nx.create
   Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in let b = Nx.create Nx.float32 [|
   2 |] [| 1.; 2. |] in check_invalid_arg "solve singular" "solve: matrix is
   singular" (fun () -> ignore (Nx.solve a b))

   let test_solve_non_square () = (* Non-square matrix *) let a = Nx.create
   Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in let b = Nx.create
   Nx.float32 [| 2 |] [| 1.; 2. |] in check_invalid_arg "solve non-square"
   "solve: coefficient matrix must be square" (fun () -> ignore (Nx.solve a b))

   let test_inv_identity () = (* inv(I) = I *) let identity = Nx.eye Nx.float32
   3 in let inv = Nx.inv identity in check_approx_equal "inv identity" identity
   inv

   let test_inv_inverse () = (* inv(inv(A)) = A *) let a = Nx.create Nx.float32
   [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in let inv_a = Nx.inv a in let inv_inv_a =
   Nx.inv inv_a in check_approx_equal "inv inverse" a inv_inv_a

   let test_inv_singular () = (* Singular matrix *) let a = Nx.create Nx.float32
   [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in check_invalid_arg "inv singular" "inv:
   matrix is singular" (fun () -> ignore (Nx.inv a)) *)

(*  ─────  Decomposition Tests  ─────  *)
(* Note: These functions are not exposed in nx.ml, so tests are commented out *)

(* let test_qr_shape () = let a = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12
   float_of_int) in let q, r = Nx.qr a in check_shape "qr q shape" [| 4; 4 |] q;
   check_shape "qr r shape" [| 4; 3 |] r

   let test_qr_property () = (* Q @ R = A *) let a = Nx.create Nx.float32 [| 3;
   3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let q, r = Nx.qr a in let
   reconstructed = Nx.matmul q r in check_approx_equal "qr property" a
   reconstructed

   let test_qr_orthogonal () = (* Q.T @ Q = I *) let a = Nx.create Nx.float32 [|
   3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let q, _ = Nx.qr a in
   let qt_q = Nx.matmul (Nx.transpose q) q in let identity = Nx.eye Nx.float32 3
   in check_approx_equal "qr orthogonal" identity qt_q

   let test_svd_shape () = let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init
   12 float_of_int) in let u, s, v = Nx.svd a in check_shape "svd u shape" [| 3;
   3 |] u; check_shape "svd s shape" [| 3 |] s; check_shape "svd v shape" [| 4;
   4 |] v

   let test_svd_property () = (* U @ S @ V.T = A *) let a = Nx.create Nx.float32
   [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let u, s, v = Nx.svd
   a in (* Create diagonal matrix from s *) let s_diag = Nx.zeros Nx.float32 [|
   3; 3 |] in for i = 0 to 2 do Nx.set_item [i; i] s_diag (Nx.get_item [i] s)
   done; let reconstructed = Nx.matmul u (Nx.matmul s_diag (Nx.transpose v)) in
   check_approx_equal "svd property" a reconstructed

   let test_cholesky_posdef () = (* Create positive definite matrix: A.T @ A *)
   let a = Nx.create Nx.float32 [| 3; 3 |] [| 1.; 0.; 0.; 1.; 1.; 0.; 1.; 1.; 1.
   |] in let posdef = Nx.matmul (Nx.transpose a) a in let l = Nx.cholesky posdef
   in check_shape "cholesky shape" [| 3; 3 |] l

   let test_cholesky_property () = (* L @ L.T = A *) let a = Nx.create
   Nx.float32 [| 2; 2 |] [| 1.; 0.; 1.; 1. |] in let posdef = Nx.matmul
   (Nx.transpose a) a in let l = Nx.cholesky posdef in let reconstructed =
   Nx.matmul l (Nx.transpose l) in check_approx_equal "cholesky property" posdef
   reconstructed

   let test_eig_shape () = let a = Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.;
   3.; 4.; 5.; 6.; 7.; 8.; 10. |] in let eigenvalues, eigenvectors = Nx.eig a in
   check_shape "eig eigenvalues shape" [| 3 |] eigenvalues; check_shape "eig
   eigenvectors shape" [| 3; 3 |] eigenvectors

   let test_eig_property () = (* A @ v = lambda * v *) let a = Nx.create
   Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in let eigenvalues, eigenvectors =
   Nx.eig a in (* Check first eigenvector *) let v1 = Nx.get (Nx.LR [Nx.All; [0;
   1]]) eigenvectors in let lambda1 = Nx.get_item [0] eigenvalues in let av1 =
   Nx.dot a v1 in let lambda_v1 = Nx.mul_s v1 lambda1 in check_approx_equal "eig
   property" av1 lambda_v1 *)

(* ───── Norm Tests ───── *)

(* let test_norm_vector_1 () = let v = Nx.create Nx.float32 [| 4 |] [| -1.; 2.;
   -3.; 4. |] in let result = Nx.norm ~ord:(`L 1.) v in check_t "norm L1" [||]
   [| 10.0 |] result *)

(* let test_norm_vector_2 () = let v = Nx.create Nx.float32 [| 3 |] [| 3.; 4.;
   0. |] in let result = Nx.norm v in (* Default is L2 *) check_t "norm L2" [||]
   [| 5.0 |] result *)

(* let test_norm_vector_inf () = let v = Nx.create Nx.float32 [| 4 |] [| -1.;
   2.; -5.; 4. |] in let result = Nx.norm ~ord:`Inf v in check_t "norm Linf"
   [||] [| 5.0 |] result *)

(* let test_norm_matrix_fro () = let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.;
   2.; 3.; 4. |] in let result = Nx.norm ~ord:`Fro m in check_t ~eps:1e-5 "norm
   Frobenius" [||] [| 5.477226 |] result *)

(* let test_norm_matrix_1 () = let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.;
   -2.; 3.; 4. |] in let result = Nx.norm ~ord:(`L 1.) m in check_t "norm matrix
   L1" [||] [| 6.0 |] result *)

(* let test_norm_axis () = let m = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.;
   3.; 4.; 5.; 6. |] in let result = Nx.norm ~axis:[1] m in check_t ~eps:1e-5
   "norm along axis" [| 2 |] [| 3.741657; 8.774964 |] result *)

(* let test_norm_empty () = let v = Nx.create Nx.float32 [| 0 |] [||] in let
   result = Nx.norm v in check_t "norm empty" [||] [| 0.0 |] result *)

(* ───── Linear Algebra Utilities ───── *)

(* let test_det_2x2 () = let a = Nx.create Nx.float32 [| 2; 2 |] [| 3.; 8.; 4.;
   6. |] in let det = Nx.det a in check_t "det 2x2" [||] [| -14.0 |] det

   let test_det_singular () = let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.;
   2.; 4. |] in let det = Nx.det a in check_t ~eps:1e-10 "det singular" [||] [|
   0.0 |] det *)

(* let test_trace () = let a = Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.;
   4.; 5.; 6.; 7.; 8.; 9. |] in let tr = Nx.trace a in check_t "trace" [||] [|
   15.0 |] tr *)

(* let test_diag_extract () = let a = Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.;
   3.; 4.; 5.; 6.; 7.; 8.; 9. |] in let diag = Nx.diag a in check_t "diag
   extract" [| 3 |] [| 1.; 5.; 9. |] diag *)

(* let test_diag_create () = let v = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3.
   |] in let result = Nx.diag v in check_t "diag create" [| 3; 3 |] [| 1.; 0.;
   0.; 0.; 2.; 0.; 0.; 0.; 3. |] result *)

(* let test_tril_triu () = let a = Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.;
   3.; 4.; 5.; 6.; 7.; 8.; 9. |] in

   let lower = Nx.tril a in check_t "tril" [| 3; 3 |] [| 1.; 0.; 0.; 4.; 5.; 0.;
   7.; 8.; 9. |] lower;

   let upper = Nx.triu a in check_t "triu" [| 3; 3 |] [| 1.; 2.; 3.; 0.; 5.; 6.;
   0.; 0.; 9. |] upper *)

(* Test Suite Organization *)

let matmul_tests =
  [
    ("matmul 2d x 2d", `Quick, test_matmul_2d_2d);
    ("matmul 1d x 1d", `Quick, test_matmul_1d_1d);
    ("matmul 1d x 2d", `Quick, test_matmul_1d_2d);
    ("matmul 2d x 1d", `Quick, test_matmul_2d_1d);
    ("matmul batch", `Quick, test_matmul_batch);
    ("matmul broadcast batch", `Quick, test_matmul_broadcast_batch);
    ("matmul shape error", `Quick, test_matmul_shape_error);
    ("matmul empty", `Quick, test_matmul_empty);
    ("matmul transpose optimization", `Quick, test_matmul_transpose_optimization);
  ]

let dot_tests =
  [
    ("dot 1d x 1d", `Quick, test_dot_1d_1d);
    ("dot 2d x 1d", `Quick, test_dot_2d_1d);
    ("dot 2d x 2d", `Quick, test_dot_2d_2d);
    ("dot higher-d", `Quick, test_dot_higher_d);
    ("dot scalar result", `Quick, test_dot_scalar_result);
  ]

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
    ("correlate2d basic", `Quick, test_correlate2d_basic);
    ("convolve invalid shapes", `Quick, test_convolve_invalid_shapes);
    ("convolve empty input", `Quick, test_convolve_empty_input);
    ( "convolve single element kernel",
      `Quick,
      test_convolve_single_element_kernel );
  ]

let solve_inverse_tests =
  [ (* ("solve identity", `Quick, test_solve_identity); *)
    (* ("solve simple", `Quick, test_solve_simple); *)
    (* ("solve batch", `Quick, test_solve_batch); *)
    (* ("solve singular", `Quick, test_solve_singular); *)
    (* ("solve non-square", `Quick, test_solve_non_square); *)
    (* ("inv identity", `Quick, test_inv_identity); *)
    (* ("inv inverse", `Quick, test_inv_inverse); *)
    (* ("inv singular", `Quick, test_inv_singular); *) ]

let decomposition_tests =
  [ (* ("qr shape", `Quick, test_qr_shape); *)
    (* ("qr property", `Quick, test_qr_property); *)
    (* ("qr orthogonal", `Quick, test_qr_orthogonal); *)
    (* ("svd shape", `Quick, test_svd_shape); *)
    (* ("svd property", `Quick, test_svd_property); *)
    (* ("cholesky posdef", `Quick, test_cholesky_posdef); *)
    (* ("cholesky property", `Quick, test_cholesky_property); *)
    (* ("eig shape", `Quick, test_eig_shape); *)
    (* ("eig property", `Quick, test_eig_property); *) ]

let norm_tests =
  [ (* ("norm vector L1", `Quick, test_norm_vector_1); *)
    (* ("norm vector L2", `Quick, test_norm_vector_2); *)
    (* ("norm vector Linf", `Quick, test_norm_vector_inf); *)
    (* ("norm matrix Frobenius", `Quick, test_norm_matrix_fro); *)
    (* ("norm matrix L1", `Quick, test_norm_matrix_1); *)
    (* ("norm axis", `Quick, test_norm_axis); *)
    (* ("norm empty", `Quick, test_norm_empty); *) ]

let utility_tests =
  [ (* ("det 2x2", `Quick, test_det_2x2); *)
    (* ("det singular", `Quick, test_det_singular); *)
    (* ("trace", `Quick, test_trace); *)
    (* ("diag extract", `Quick, test_diag_extract); *)
    (* ("diag create", `Quick, test_diag_create); *)
    (* ("tril triu", `Quick, test_tril_triu); *) ]

let () =
  Printexc.record_backtrace true;
  Alcotest.run "Nx Linear Algebra"
    [
      ("Matrix Multiply", matmul_tests);
      ("Dot Product", dot_tests);
      ("Convolution", convolution_tests);
      ("Solve/Inverse", solve_inverse_tests);
      ("Decompositions", decomposition_tests);
      ("Norms", norm_tests);
      ("Utilities", utility_tests);
    ]
