(* Linear algebra tests for Nx *)

open Alcotest
open Test_nx_support

(* ───── Matrix Multiply Tests ───── *)

let test_matmul_2d_2d () =
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 4; 5 |] (Array.init 20 float_of_int) in
  let result = Nx.matmul a b in
  check_shape "matmul 2d x 2d shape" [| 3; 5 |] result;
  (* Check a few values *)
  check (float 1e-6) "matmul[0,0]" 70.0 (Nx.item [ 0; 0 ] result);
  check (float 1e-6) "matmul[2,4]" 462.0 (Nx.item [ 2; 4 ] result)

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
  check (float 1e-6) "batch[0,0,0]" 28.0 (Nx.item [ 0; 0; 0 ] result);
  check (float 1e-6) "batch[0,0,1]" 34.0 (Nx.item [ 0; 0; 1 ] result)

let test_matmul_broadcast_batch () =
  let a = Nx.create Nx.float32 [| 1; 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 5; 4; 2 |] (Array.init 40 float_of_int) in
  let result = Nx.matmul a b in
  check_shape "matmul broadcast batch shape" [| 5; 3; 2 |] result

let test_matmul_2d_3d_broadcast () =
  (*
   * Test case: A (2D) @ B (3D)
   * A shape: (2, 3) - to be broadcasted
   * B shape: (4, 3, 2) - batched tensor
   * Expected output shape: (4, 2, 2)
   *)

  (* A is a single 2x3 matrix *)
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in

  (* B is a batch of four 3x2 matrices *)
  let b =
    Nx.create Nx.float32 [| 4; 3; 2 |]
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
  check (float 1e-6) "batch 0 [0,0]" 22. (Nx.item [ 0; 0; 0 ] result);
  check (float 1e-6) "batch 0 [0,1]" 28. (Nx.item [ 0; 0; 1 ] result);
  check (float 1e-6) "batch 0 [1,0]" 49. (Nx.item [ 0; 1; 0 ] result);
  check (float 1e-6) "batch 0 [1,1]" 64. (Nx.item [ 0; 1; 1 ] result);

  (* Check batch 1 *)
  check (float 1e-6) "batch 1 [0,0]" 58. (Nx.item [ 1; 0; 0 ] result);
  check (float 1e-6) "batch 1 [0,1]" 64. (Nx.item [ 1; 0; 1 ] result);
  check (float 1e-6) "batch 1 [1,0]" 139. (Nx.item [ 1; 1; 0 ] result);
  check (float 1e-6) "batch 1 [1,1]" 154. (Nx.item [ 1; 1; 1 ] result);

  (* Check batch 2 *)
  check (float 1e-6) "batch 2 [0,0]" 4. (Nx.item [ 2; 0; 0 ] result);
  check (float 1e-6) "batch 2 [0,1]" 2. (Nx.item [ 2; 0; 1 ] result);
  check (float 1e-6) "batch 2 [1,0]" 10. (Nx.item [ 2; 1; 0 ] result);
  check (float 1e-6) "batch 2 [1,1]" 5. (Nx.item [ 2; 1; 1 ] result);

  (* Check batch 3 *)
  check (float 1e-6) "batch 3 [0,0]" 2. (Nx.item [ 3; 0; 0 ] result);
  check (float 1e-6) "batch 3 [0,1]" 4. (Nx.item [ 3; 0; 1 ] result);
  check (float 1e-6) "batch 3 [1,0]" 5. (Nx.item [ 3; 1; 0 ] result);
  check (float 1e-6) "batch 3 [1,1]" 10. (Nx.item [ 3; 1; 1 ] result)

let test_matmul_shape_error () =
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.create Nx.float32 [| 5; 6 |] (Array.init 30 float_of_int) in
  check_raises "matmul shape error"
    (Invalid_argument
       "dot: cannot contract [3,4] (last axis: 4) to [5,6] (axis 0: 5) (size \
        4\226\137\1605)") (fun () -> ignore (Nx.matmul a b))

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
  check (float 1e-6) "dot scalar value" 32.0 (Nx.item [] result)

(* ───── Solve Inverse Tests ───── *)

let test_solve_identity () =
  let identity = Nx.eye Nx.float32 3 in
  let b = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let x = Nx.solve identity b in
  check_t "solve identity" [| 3 |] [| 1.; 2.; 3. |] x

let test_solve_simple () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 9.; 8. |] in
  let x = Nx.solve a b in
  let result = Nx.dot a x in
  check_nx "solve simple" b result

let test_solve_batch () =
  let a =
    Nx.create Nx.float32 [| 2; 3; 3 |]
      [|
        1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.; 2.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 2.;
      |]
  in
  let b = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6. |] in
  let x = Nx.solve a b in
  check_shape "solve batch shape" [| 2; 3 |] x

let test_solve_singular () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  check_invalid_arg "solve singular" "solve: matrix is singular" (fun () ->
      ignore (Nx.solve a b))

let test_solve_non_square () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  check_invalid_arg "solve non-square"
    "solve: coefficient matrix must be square" (fun () -> ignore (Nx.solve a b))

let test_inv_identity () =
  let identity = Nx.eye Nx.float32 3 in
  let inv = Nx.inv identity in
  check_nx "inv identity" identity inv

let test_inv_inverse () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let inv_a = Nx.inv a in
  let inv_inv_a = Nx.inv inv_a in
  check_nx "inv inverse" a inv_inv_a

let test_inv_singular () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  check_invalid_arg "inv singular" "inv: matrix is singular" (fun () ->
      ignore (Nx.inv a))

(* ───── Decomposition Tests ───── *)

let test_qr_shape () =
  let a = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
  let q, r = Nx.qr a in
  check_shape "qr q shape" [| 4; 4 |] q;
  check_shape "qr r shape" [| 4; 3 |] r

let test_qr_property () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
  in
  let q, r = Nx.qr a in
  let reconstructed = Nx.matmul q r in
  check_nx ~epsilon:1e-5 "qr property" a reconstructed

let test_qr_orthogonal () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
  in
  let q, _ = Nx.qr a in
  let qt_q = Nx.matmul (Nx.transpose q) q in
  let identity = Nx.eye Nx.float32 3 in
  check_nx "qr orthogonal" identity qt_q

let test_svd_shape () =
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let u, s, vt = Nx.svd a in
  check_shape "svd u shape" [| 3; 3 |] u;
  check_shape "svd s shape" [| 3 |] s;
  check_shape "svd vt shape (V^H)" [| 3; 4 |] vt

let test_svd_property () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
  in
  let u, s, vh = Nx.svd a in
  let s_diag = Nx.zeros Nx.float32 [| 3; 3 |] in
  let s_float32 = Nx.cast Nx.float32 s in
  for i = 0 to 2 do
    let s_val = Nx.item [ i ] s_float32 in
    Nx.set_item [ i; i ] s_val s_diag
  done;
  let reconstructed = Nx.matmul u (Nx.matmul s_diag vh) in
  check_nx ~epsilon:1e-5 "svd property" a reconstructed

let test_cholesky_posdef () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 0.; 0.; 1.; 1.; 0.; 1.; 1.; 1. |]
  in
  let posdef = Nx.matmul (Nx.transpose a) a in
  let l = Nx.cholesky posdef in
  check_shape "cholesky shape" [| 3; 3 |] l

let test_cholesky_property () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 0.; 1.; 1. |] in
  let posdef = Nx.matmul (Nx.transpose a) a in
  let l = Nx.cholesky posdef in
  let reconstructed = Nx.matmul l (Nx.transpose l) in
  check_nx ~epsilon:1e-5 "cholesky property" posdef reconstructed

let test_eig_shape () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
  in
  let eigenvalues, eigenvectors = Nx.eig a in
  check_shape "eig eigenvalues shape" [| 3 |] eigenvalues;
  check_shape "eig eigenvectors shape" [| 3; 3 |] eigenvectors

let test_eig_property () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let eigenvalues, eigenvectors = Nx.eig a in
  (* Cast to float32 to match a's type *)
  let eigenvalues_f32 = Nx.cast Nx.float32 eigenvalues in
  let eigenvectors_f32 = Nx.cast Nx.float32 eigenvectors in
  let v1 = Nx.slice [ Nx.R (0, 2); Nx.I 0 ] eigenvectors_f32 in
  let lambda1 = Nx.item [ 0 ] eigenvalues_f32 in
  let av1 = Nx.dot a v1 in
  let lambda1_scalar = Nx.scalar Nx.float32 lambda1 in
  let lambda_v1 = Nx.mul lambda1_scalar v1 in
  check_nx "eig property" av1 lambda_v1

(* ───── Norm Tests ───── *)

let test_norm_vector_1 () =
  let v = Nx.create Nx.float32 [| 4 |] [| -1.; 2.; -3.; 4. |] in
  let result = Nx.norm ~ord:(`P 1.) v in
  check_t "norm L1" [||] [| 10.0 |] result

let test_norm_vector_2 () =
  let v = Nx.create Nx.float32 [| 3 |] [| 3.; 4.; 0. |] in
  let result = Nx.norm v in
  check_t "norm L2" [||] [| 5.0 |] result

let test_norm_vector_inf () =
  let v = Nx.create Nx.float32 [| 4 |] [| -1.; 2.; -5.; 4. |] in
  let result = Nx.norm ~ord:`Inf v in
  check_t "norm Linf" [||] [| 5.0 |] result

let test_norm_matrix_fro () =
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let result = Nx.norm ~ord:`Fro m in
  check_t ~eps:1e-5 "norm Frobenius" [||] [| 5.477226 |] result

let test_norm_matrix_1 () =
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.; -2.; 3.; 4. |] in
  let result = Nx.norm ~ord:(`P 1.) m in
  check_t "norm matrix L1" [||] [| 6.0 |] result

let test_norm_axis () =
  let m = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.norm ~axes:[ 1 ] m in
  check_t ~eps:1e-5 "norm along axis" [| 2 |] [| 3.741657; 8.774964 |] result

let test_norm_empty () =
  let v = Nx.create Nx.float32 [| 0 |] [||] in
  let result = Nx.norm v in
  check_t "norm empty" [||] [| 0.0 |] result

(* ───── Linear Algebra Utilities ───── *)

let test_det_2x2 () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 3.; 8.; 4.; 6. |] in
  let det = Nx.det a in
  check_t "det 2x2" [||] [| -14.0 |] det

let test_det_singular () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  let det = Nx.det a in
  check_t ~eps:1e-6 "det singular" [||] [| 0.0 |] det

let test_trace () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let tr = Nx.trace a in
  check_t "trace" [||] [| 15.0 |] tr

let test_diag_extract () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let diag = Nx.diagonal a in
  check_t "diag extract" [| 3 |] [| 1.; 5.; 9. |] diag

(* ───── Additional Utility Tests ───── *)

let test_diagonal () =
  let a = Nx.create Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let d = Nx.diagonal a in
  check_t "diagonal main" [| 3 |] [| 0.; 4.; 8. |] d;
  let d_offset = Nx.diagonal ~offset:1 a in
  check_t "diagonal offset 1" [| 2 |] [| 1.; 5. |] d_offset;
  let a_higher =
    Nx.create Nx.float32 [| 2; 3; 3 |] (Array.init 18 float_of_int)
  in
  let d_higher = Nx.diagonal a_higher in
  check_shape "diagonal higher dim" [| 2; 3 |] d_higher

let test_diagonal_edge () =
  let a_empty = Nx.create Nx.float32 [| 0; 0 |] [||] in
  let d_empty = Nx.diagonal a_empty in
  check_shape "diagonal empty" [| 0 |] d_empty;
  check_raises "diagonal invalid axes"
    (Invalid_argument "diagonal: invalid axes (axes must be different)") (fun () ->
      ignore (Nx.diagonal ~axis1:0 ~axis2:0 a_empty))

let test_matrix_transpose () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t = Nx.matrix_transpose a in
  check_shape "matrix transpose shape" [| 3; 2 |] t;
  check_t "matrix transpose values" [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |] t;
  let a1d = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let t1d = Nx.matrix_transpose a1d in
  check_t "matrix transpose 1d unchanged" [| 3 |] [| 1.; 2.; 3. |] t1d

let test_trace_offset () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let tr_offset = Nx.trace ~offset:1 a in
  check_t "trace offset 1" [||] [| 8. |] tr_offset

let test_det_batch () =
  let a =
    Nx.create Nx.float32 [| 2; 2; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let d = Nx.det a in
  check_shape "det batch" [| 2 |] d;
  check_t "det batch values" [| 2 |] [| -2.; -2. |] d

let test_slogdet () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 3.; 8.; 4.; 6. |] in
  let sign, logdet = Nx.slogdet a in
  check_t "slogdet sign" [||] [| -1. |] sign;
  check (float 1e-5) "slogdet logdet" (log 14.) (Nx.item [] logdet)

let test_slogdet_singular () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  let sign, logdet = Nx.slogdet a in
  check_t "slogdet singular sign" [||] [| 0. |] sign;
  check (float 1e-5) "slogdet singular logdet" neg_infinity (Nx.item [] logdet)

let test_matrix_rank () =
  let a = Nx.create Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let r = Nx.matrix_rank a in
  check int "matrix rank full" 2 r;
  let a_low =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6.; 3.; 6.; 9. |]
  in
  let r_low = Nx.matrix_rank a_low in
  check int "matrix_rank low" 1 r_low

let test_matrix_rank_tol () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; 1e-10 |] in
  let r = Nx.matrix_rank ~tol:1e-8 a in
  check int "matrix_rank with tol" 1 r

let test_matrix_rank_hermitian () =
  (* Create a symmetric matrix with known rank *)
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 2.; 1.; 0.; 1.; 2.; 0.; 0.; 0.; 0. |]
  in
  let r = Nx.matrix_rank ~hermitian:true a in
  check int "matrix_rank hermitian" 2 r;
  (* Test that hermitian flag is actually used by checking it works on a non-square matrix *)
  (* This will fail if hermitian flag is ignored because eigh requires square matrices *)
  let non_square =
    Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
  in
  check_raises "matrix_rank hermitian non-square"
    (Failure "eig: input must be square matrix") (fun () ->
      ignore (Nx.matrix_rank ~hermitian:true non_square))

let test_matrix_rank_hermitian_negative () =
  (* Test negative-definite matrix *)
  let a = Nx.create Nx.float32 [| 2; 2 |] [| -2.; 0.; 0.; -1. |] in
  let r = Nx.matrix_rank ~hermitian:true a in
  check int "matrix_rank hermitian negative" 2 r;
  (* Compare with non-hermitian version *)
  let r_svd = Nx.matrix_rank a in
  check int "matrix_rank hermitian negative vs svd" r_svd r

let test_matrix_rank_hermitian_complex () =
  (* Complex Hermitian matrix with full rank *)
  let a =
    Nx.create Nx.complex64 [| 2; 2 |]
      [|
        Complex.{ re = 2.; im = 0. };
        Complex.{ re = 0.; im = 1.5 };
        Complex.{ re = 0.; im = -1.5 };
        Complex.{ re = 3.; im = 0. };
      |]
  in
  let r = Nx.matrix_rank ~hermitian:true a in
  check int "matrix_rank hermitian complex" 2 r;
  let r_svd = Nx.matrix_rank a in
  check int "matrix_rank hermitian complex vs svd" r_svd r

let test_pinv_hermitian () =
  (* Create a symmetric matrix *)
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let pinv_a = Nx.pinv ~hermitian:true a in
  (* Check that a @ pinv_a @ a ≈ a (pseudoinverse property) *)
  let recon = Nx.matmul a (Nx.matmul pinv_a a) in
  check_nx ~epsilon:1e-5 "pinv hermitian recon" a recon;
  (* Test that hermitian flag is actually used by checking it works on a non-square matrix *)
  (* This will fail if hermitian flag is ignored because eigh requires square matrices *)
  let non_square =
    Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
  in
  check_raises "pinv hermitian non-square"
    (Failure "eig: input must be square matrix") (fun () ->
      ignore (Nx.pinv ~hermitian:true non_square))

let test_pinv_hermitian_negative () =
  (* Test negative-definite matrix *)
  let a = Nx.create Nx.float32 [| 2; 2 |] [| -2.; 0.; 0.; -1. |] in
  let pinv_a = Nx.pinv ~hermitian:true a in
  (* Check that a @ pinv_a @ a ≈ a (pseudoinverse property) *)
  let recon = Nx.matmul a (Nx.matmul pinv_a a) in
  check_nx ~epsilon:1e-5 "pinv hermitian negative recon" a recon;
  (* Compare with non-hermitian version *)
  let pinv_svd = Nx.pinv a in
  check_nx ~epsilon:1e-5 "pinv hermitian negative vs svd" pinv_svd pinv_a

let test_pinv_hermitian_complex () =
  (* Complex Hermitian matrix *)
  let a =
    Nx.create Nx.complex64 [| 2; 2 |]
      [|
        Complex.{ re = 4.; im = 0. };
        Complex.{ re = 1.; im = 2. };
        Complex.{ re = 1.; im = -2. };
        Complex.{ re = 5.; im = 0. };
      |]
  in
  let pinv_a = Nx.pinv ~hermitian:true a in
  let identity = Nx.identity Nx.complex64 2 in
  let product = Nx.matmul a pinv_a in
  check_nx ~epsilon:1e-5 "pinv hermitian complex identity" identity product;
  let recon = Nx.matmul a (Nx.matmul pinv_a a) in
  check_nx ~epsilon:1e-5 "pinv hermitian complex recon" a recon;
  let pinv_svd = Nx.pinv a in
  check_nx ~epsilon:1e-5 "pinv hermitian complex vs svd" pinv_svd pinv_a

(* ───── Product Ops Tests ───── *)

let test_vdot () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let res = Nx.vdot a b in
  check_t "vdot 1d" [||] [| 32. |] res;
  let a2 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let res2 = Nx.vdot a2 b in
  check_t "vdot flatten" [||] [| 4. +. 10. +. 18. +. 16. +. 25. +. 36. |] res2

let test_vdot_complex () =
  (* Test complex vdot with conjugation *)
  let a =
    Nx.create Nx.complex32 [| 2 |]
      [| Complex.{ re = 1.; im = 2. }; Complex.{ re = 3.; im = 4. } |]
  in
  let b =
    Nx.create Nx.complex32 [| 2 |]
      [| Complex.{ re = 5.; im = 6. }; Complex.{ re = 7.; im = 8. } |]
  in
  let result = Nx.vdot a b in
  (* Expected: conj(a) * b = [(1-2i)(5+6i), (3-4i)(7+8i)] = [17-4i, 53-4i] =
     70-8i *)
  let expected = Complex.{ re = 70.; im = -8. } in
  let actual = Nx.item [] result in
  check (float 1e-6) "vdot complex real part" expected.re actual.re;
  check (float 1e-6) "vdot complex imag part" expected.im actual.im

let test_conjugate () =
  (* Test complex conjugate *)
  let x =
    Nx.create Nx.complex32 [| 2 |]
      [| Complex.{ re = 1.; im = 2. }; Complex.{ re = 3.; im = 4. } |]
  in
  let conj_x = Nx.conjugate x in
  let expected =
    [| Complex.{ re = 1.; im = -2. }; Complex.{ re = 3.; im = -4. } |]
  in
  let actual = Nx.to_array conj_x in
  check (float 1e-6) "conjugate[0] real" expected.(0).re actual.(0).re;
  check (float 1e-6) "conjugate[0] imag" expected.(0).im actual.(0).im;
  check (float 1e-6) "conjugate[1] real" expected.(1).re actual.(1).re;
  check (float 1e-6) "conjugate[1] imag" expected.(1).im actual.(1).im;
  (* Test that real tensors are unchanged *)
  let real_x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  let conj_real = Nx.conjugate real_x in
  check_nx "conjugate real unchanged" real_x conj_real

let test_vdot_mismatch () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 4 |] [| 4.; 5.; 6.; 7. |] in
  check_raises "vdot mismatch"
    (Invalid_argument "vdot: different number of elements") (fun () ->
      ignore (Nx.vdot a b))

let test_vecdot () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let res = Nx.vecdot a b in
  check_t "vecdot default axis" [| 2 |] [| 50.; 167. |] res;
  let res_axis0 = Nx.vecdot ~axis:0 a b in
  check_t "vecdot axis 0" [| 3 |] [| 47.; 71.; 99. |] res_axis0

let test_inner () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let res = Nx.inner a b in
  check_t "inner 1d" [||] [| 32. |] res;
  let a2 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let res2 = Nx.inner a2 a in
  check_t "inner higher" [| 2 |] [| 14.; 32. |] res2

let test_inner_mismatch () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 4 |] [| 4.; 5.; 6.; 7. |] in
  check_raises "inner mismatch"
    (Invalid_argument "inner: last dimensions differ") (fun () ->
      ignore (Nx.inner a b))

let test_outer () =
  let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 3.; 4.; 5. |] in
  let res = Nx.outer a b in
  check_t "outer" [| 2; 3 |] [| 3.; 4.; 5.; 6.; 8.; 10. |] res;
  let a_scalar = Nx.create Nx.float32 [||] [| 2. |] in
  let res_scalar = Nx.outer a_scalar b in
  check_t "outer scalar" [| 3 |] [| 6.; 8.; 10. |] res_scalar

let test_tensordot () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let res = Nx.tensordot a b in
  check_t "tensordot default" [| 2; 2 |] [| 58.; 64.; 139.; 154. |] res;
  let res_axes = Nx.tensordot ~axes:([ 0 ], [ 1 ]) a b in
  check_shape "tensordot custom axes" [| 3; 3 |] res_axes

let test_tensordot_mismatch () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 4; 2 |] (Array.init 8 float_of_int) in
  check_raises "tensordot mismatch"
    (Invalid_argument "tensordot: axes have different sizes") (fun () ->
      ignore (Nx.tensordot ~axes:([ 1 ], [ 0 ]) a b))

let test_einsum_error () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  check_raises "einsum no input operands"
    (Invalid_argument "einsum: no input operands") (fun () ->
      ignore (Nx.einsum "" [||]));
  check_raises "einsum bad format"
    (Invalid_argument "einsum: invalid format, expected inputs->output") (fun () ->
      ignore (Nx.einsum "IJ,JK-IK" [| a; b |]));
  check_raises "einsum wrong inputs"
    (Invalid_argument "einsum: number of inputs must equal number of operands")
    (fun () -> ignore (Nx.einsum "ij->ij" [| a; b |]));
  check_raises "einsum rectangular diagonal"
    (Invalid_argument
       "einsum: index var 'i' must have consistent dimensions (2 vs 3)")
    (fun () -> ignore (Nx.einsum "ii->i" [| a |]));
  check_raises "einsum mismatched rank"
    (Invalid_argument "einsum: operand rank too small for subscripts")
    (fun () -> ignore (Nx.einsum "ijl,jk->ik" [| a; b |]));
  check_raises "einsum contracted vars mismatch"
    (Invalid_argument "einsum: output index 'k' not found in inputs") (fun () ->
      ignore (Nx.einsum "ij,jl->ki" [| a; b |]));
  check_raises "einsum dimension mismatch"
    (Invalid_argument
       "einsum: index var 'j' must have consistent dimensions (3 vs 2)")
    (fun () -> ignore (Nx.einsum "ij,kj->ik" [| a; b |]));

  check_raises "einsum output ell without input"
    (Invalid_argument "einsum: output ellipsis requires ellipsis in inputs")
    (fun () -> ignore (Nx.einsum "ij->..." [| a |]));
  check_raises "einsum multi ellipsis"
    (Invalid_argument "einsum: multiple ellipsis in operand") (fun () ->
      ignore (Nx.einsum "i...j...->ij" [| a |]))

(* Weighted broadcast dot retained from legacy spec *)
let einsum_weighted_broadcast () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let vec = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let got = Nx.einsum "...i,i->..." [| a; vec |] in
  let expected =
    let mul = Nx.mul a (Nx.reshape [| 1; 3 |] vec) in
    Nx.sum ~axes:[ 1 ] mul
  in
  check_nx "einsum weighted broadcast ...i,i->..." expected got

let einsum_complex_fro_inner () =
  let open Complex in
  let a =
    Nx.create Nx.Complex32 [| 2; 2 |]
      [|
        { re = 1.; im = 2. };
        { re = 3.; im = 4. };
        { re = -1.; im = 0. };
        { re = 0.5; im = -1.5 };
      |]
  in
  let b =
    Nx.create Nx.Complex32 [| 2; 2 |]
      [|
        { re = -2.; im = 1. };
        { re = 0.; im = 1. };
        { re = 2.; im = -1. };
        { re = -0.5; im = 2. };
      |]
  in
  let got = Nx.einsum "ij,ij->" [| a; b |] in
  let expected = Nx.sum (Nx.mul a b) in
  check_nx "einsum complex fro inner ij,ij->" expected got

let einsum_int_dot_scalar () =
  let a = Nx.create Nx.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
  let b = Nx.create Nx.int32 [| 4 |] [| 5l; 6l; 7l; 8l |] in
  let got = Nx.einsum "i,i->" [| a; b |] in
  let expected = Nx.sum (Nx.mul a b) in
  check_nx "einsum int dot scalar i,i->" expected got

let test_einsum_regression_axis_order () =
  (* Case 1: i,jk->jki should order as j, k, i *)
  let a1 = Nx.randn Nx.float32 [| 5 |] in
  let b1 = Nx.randn Nx.float32 [| 7; 7 |] in
  let r1 = Nx.einsum "i,jk->jki" [| a1; b1 |] in
  check_shape "einsum axis order i,jk->jki" [| 7; 7; 5 |] r1;

  (* Case 2: ij,klj->kli should order as k, l, i *)
  let a2 = Nx.randn Nx.float32 [| 5; 5 |] in
  let b2 = Nx.randn Nx.float32 [| 3; 7; 5 |] in
  let r2 = Nx.einsum "ij,klj->kli" [| a2; b2 |] in
  check_shape "einsum axis order ij,klj->kli" [| 3; 7; 5 |] r2

let einsum_dot_scalar () =
  let a0 = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let a1 = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let got = Nx.einsum "i,i->" [| a0; a1 |] in
  let expected = Nx.create Nx.float32 [||] [| 55. |] in
  check_nx "einsum_dot_scalar i,i->" expected got

let einsum_matmul () =
  let a0 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let a1 = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let got = Nx.einsum "ij,jk->ik" [| a0; a1 |] in
  let expected = Nx.create Nx.float32 [| 2; 2 |] [| 22.; 28.; 49.; 64. |] in
  check_nx "einsum_matmul ij,jk->ik" expected got

let einsum_transpose () =
  let a0 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let got = Nx.einsum "ij->ji" [| a0 |] in
  let expected = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |] in
  check_nx "einsum_transpose ij->ji" expected got

let einsum_outer () =
  let a0 = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  let a1 = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let got = Nx.einsum "i,j->ij" [| a0; a1 |] in
  let expected = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6. |] in
  check_nx "einsum_outer i,j->ij" expected got

let einsum_total_sum () =
  let a0 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let got = Nx.einsum "ij->" [| a0 |] in
  let expected = Nx.create Nx.float32 [||] [| 21. |] in
  check_nx "einsum_total_sum ij->" expected got

let einsum_diag_extract () =
  let a0 =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let got = Nx.einsum "ii->i" [| a0 |] in
  let expected = Nx.create Nx.float32 [| 3 |] [| 1.; 5.; 9. |] in
  check_nx "einsum_diag_extract ii->i" expected got

let einsum_batched_diag () =
  let a0 =
    Nx.create Nx.float32 [| 2; 3; 3 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
      |]
  in
  let got = Nx.einsum "...ii->...i" [| a0 |] in
  let expected =
    Nx.create Nx.float32 [| 2; 3 |] [| 1.; 5.; 9.; 10.; 14.; 18. |]
  in
  check_nx "einsum_batched_diag ...ii->...i" expected got

let einsum_batched_matmul () =
  let a0 =
    Nx.create Nx.float32 [| 2; 3; 4 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
        19.;
        20.;
        21.;
        22.;
        23.;
        24.;
      |]
  in
  let a1 =
    Nx.create Nx.float32 [| 1; 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let got = Nx.einsum "...ij,...jk->...ik" [| a0; a1 |] in
  let expected =
    Nx.create Nx.float32 [| 2; 3; 2 |]
      [| 50.; 60.; 114.; 140.; 178.; 220.; 242.; 300.; 306.; 380.; 370.; 460. |]
  in
  check_nx "einsum_batched_matmul ...ij,...jk->...ik" expected got

let einsum_free_order1 () =
  let a0 = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let a1 = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "i,jk->jki" [| a0; a1 |] in
  let expected =
    Nx.create Nx.float32 [| 2; 2; 3 |]
      [| 1.; 2.; 3.; 2.; 4.; 6.; 3.; 6.; 9.; 4.; 8.; 12. |]
  in
  check_nx "einsum_free_order1 i,jk->jki" expected got

let einsum_free_order2 () =
  let a0 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let a1 =
    Nx.create Nx.float32 [| 4; 5; 3 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
        19.;
        20.;
        21.;
        22.;
        23.;
        24.;
        25.;
        26.;
        27.;
        28.;
        29.;
        30.;
        31.;
        32.;
        33.;
        34.;
        35.;
        36.;
        37.;
        38.;
        39.;
        40.;
        41.;
        42.;
        43.;
        44.;
        45.;
        46.;
        47.;
        48.;
        49.;
        50.;
        51.;
        52.;
        53.;
        54.;
        55.;
        56.;
        57.;
        58.;
        59.;
        60.;
      |]
  in
  let got = Nx.einsum "ij,klj->kli" [| a0; a1 |] in
  let expected =
    Nx.create Nx.float32 [| 4; 5; 2 |]
      [|
        14.;
        32.;
        32.;
        77.;
        50.;
        122.;
        68.;
        167.;
        86.;
        212.;
        104.;
        257.;
        122.;
        302.;
        140.;
        347.;
        158.;
        392.;
        176.;
        437.;
        194.;
        482.;
        212.;
        527.;
        230.;
        572.;
        248.;
        617.;
        266.;
        662.;
        284.;
        707.;
        302.;
        752.;
        320.;
        797.;
        338.;
        842.;
        356.;
        887.;
      |]
  in
  check_nx "einsum_free_order2 ij,klj->kli" expected got

let einsum_mix_reorder () =
  let a0 =
    Nx.create Nx.float32 [| 2; 3; 2 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let a1 = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let got = Nx.einsum "abc,bd->dac" [| a0; a1 |] in
  let expected =
    Nx.create Nx.float32 [| 2; 2; 2 |]
      [| 35.; 44.; 89.; 98.; 44.; 56.; 116.; 128. |]
  in
  check_nx "einsum_mix_reorder abc,bd->dac" expected got

let einsum_chain () =
  let a0 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let a1 = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let a2 = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "ab,bc,cd->ad" [| a0; a1; a2 |] in
  let expected = Nx.create Nx.float32 [| 2; 2 |] [| 106.; 156.; 241.; 354. |] in
  check_nx "einsum_chain ab,bc,cd->ad" expected got

let einsum_diag_sum () =
  let a0 =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let got = Nx.einsum "ii" [| a0 |] in
  let expected = Nx.create Nx.float32 [||] [| 15. |] in
  check_nx "einsum_diag_sum ii" expected got

let einsum_hadamard_vec () =
  let a0 = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let a1 = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "i,i->i" [| a0; a1 |] in
  let expected = Nx.create Nx.float32 [| 4 |] [| 1.; 4.; 9.; 16. |] in
  check_nx "einsum_hadamard_vec i,i->i" expected got

let einsum_fro_inner () =
  let a0 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let a1 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let got = Nx.einsum "ij,ij->" [| a0; a1 |] in
  let expected = Nx.create Nx.float32 [||] [| 91. |] in
  check_nx "einsum_fro_inner ij,ij->" expected got

let einsum_contract_last () =
  let a0 =
    Nx.create Nx.float32 [| 2; 3; 4 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
        19.;
        20.;
        21.;
        22.;
        23.;
        24.;
      |]
  in
  let a1 = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "ijk,k->ij" [| a0; a1 |] in
  let expected =
    Nx.create Nx.float32 [| 2; 3 |] [| 30.; 70.; 110.; 150.; 190.; 230. |]
  in
  check_nx "einsum_contract_last ijk,k->ij" expected got

let einsum_matvec () =
  let a0 =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let a1 = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "ab,b->a" [| a0; a1 |] in
  let expected = Nx.create Nx.float32 [| 3 |] [| 30.; 70.; 110. |] in
  check_nx "einsum_matvec ab,b->a" expected got

let einsum_contract_3d_vec () =
  let a0 =
    Nx.create Nx.float32 [| 2; 3; 4 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
        19.;
        20.;
        21.;
        22.;
        23.;
        24.;
      |]
  in
  let a1 = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "abc,c->ab" [| a0; a1 |] in
  let expected =
    Nx.create Nx.float32 [| 2; 3 |] [| 30.; 70.; 110.; 150.; 190.; 230. |]
  in
  check_nx "einsum_contract_3d_vec abc,c->ab" expected got

let einsum_broadcast_last_dot () =
  let a0 =
    Nx.create Nx.float32 [| 2; 3; 4 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
        19.;
        20.;
        21.;
        22.;
        23.;
        24.;
      |]
  in
  let a1 = Nx.create Nx.float32 [| 1; 1; 4 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "...i,...i->..." [| a0; a1 |] in
  let expected =
    Nx.create Nx.float32 [| 2; 3 |] [| 30.; 70.; 110.; 150.; 190.; 230. |]
  in
  check_nx "einsum_broadcast_last_dot ...i,...i->..." expected got

let einsum_move_first_axis_to_last () =
  let a0 =
    Nx.create Nx.float32 [| 2; 3; 4 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
        19.;
        20.;
        21.;
        22.;
        23.;
        24.;
      |]
  in
  let got = Nx.einsum "i...->...i" [| a0 |] in
  let expected =
    Nx.create Nx.float32 [| 3; 4; 2 |]
      [|
        1.;
        13.;
        2.;
        14.;
        3.;
        15.;
        4.;
        16.;
        5.;
        17.;
        6.;
        18.;
        7.;
        19.;
        8.;
        20.;
        9.;
        21.;
        10.;
        22.;
        11.;
        23.;
        12.;
        24.;
      |]
  in
  check_nx "einsum_move_first_axis_to_last i...->...i" expected got

let einsum_rowwise_dot () =
  let a0 =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let a1 = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let got = Nx.einsum "ij,j->i" [| a0; a1 |] in
  let expected = Nx.create Nx.float32 [| 3 |] [| 30.; 70.; 110. |] in
  check_nx "einsum_rowwise_dot ij,j->i" expected got

let test_kron () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let b = Nx.create Nx.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
  let res = Nx.kron a b in
  check_t "kron" [| 4; 4 |]
    [|
      5.; 6.; 10.; 12.; 7.; 8.; 14.; 16.; 15.; 18.; 20.; 24.; 21.; 24.; 28.; 32.;
    |]
    res

let test_multi_dot () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let c =
    Nx.create Nx.float32 [| 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let res = Nx.multi_dot [| a; b; c |] in
  let manual = Nx.matmul a (Nx.matmul b c) in
  check_nx "multi_dot" manual res

let test_multi_dot_empty () =
  check_raises "multi_dot empty" (Invalid_argument "multi_dot: empty array")
    (fun () -> ignore (Nx.multi_dot [||]))

let test_matrix_power () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 1.; 1.; 0. |] in
  let pow3 = Nx.matrix_power a 3 in
  check_t "matrix_power positive" [| 2; 2 |] [| 3.; 2.; 2.; 1. |] pow3;
  let pow0 = Nx.matrix_power a 0 in
  let id = Nx.eye Nx.float32 2 in
  check_nx "matrix_power zero" id pow0;
  let pow_neg2 = Nx.matrix_power a (-2) in
  let inv = Nx.inv a in
  let inv2 = Nx.matmul inv inv in
  check_nx "matrix_power negative" inv2 pow_neg2

let test_matrix_power_singular () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  check_raises "matrix_power singular negative"
    (Invalid_argument "matrix_power: singular for negative exponent") (fun () ->
      ignore (Nx.matrix_power a (-1)))

let test_cross () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let res = Nx.cross a b in
  check_t "cross 3d" [| 3 |] [| -3.; 6.; -3. |] res;
  let a_batch = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b_batch =
    Nx.create Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let res_batch = Nx.cross ~axis:1 a_batch b_batch in
  check_shape "cross batch" [| 2; 3 |] res_batch

let test_cross_invalid () =
  let a = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let b = Nx.create Nx.float32 [| 4 |] [| 5.; 6.; 7.; 8. |] in
  check_raises "cross invalid dim" (Invalid_argument "cross: axis dim not 3")
    (fun () -> ignore (Nx.cross a b))

(* ───── Advanced Decomposition Tests ───── *)

let test_cholesky_upper () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let u = Nx.cholesky ~upper:true a in
  let recon = Nx.matmul (Nx.transpose u) u in
  check_nx "cholesky upper" a recon

let test_cholesky_non_posdef () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  check_raises "cholesky non posdef"
    (Invalid_argument "cholesky: not positive-definite") (fun () ->
      ignore (Nx.cholesky a))

let test_qr_mode () =
  let a = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
  let q_red, r_red = Nx.qr ~mode:`Reduced a in
  check_shape "qr reduced q" [| 4; 3 |] q_red;
  check_shape "qr reduced r" [| 3; 3 |] r_red;
  let q_comp, r_comp = Nx.qr ~mode:`Complete a in
  check_shape "qr complete q" [| 4; 4 |] q_comp;
  check_shape "qr complete r" [| 4; 3 |] r_comp

let test_svd_full_matrices () =
  let a = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let u, s, vh = Nx.svd ~full_matrices:true a in
  check_shape "svd full u" [| 3; 3 |] u;
  check_shape "svd full vh" [| 4; 4 |] vh;
  let u_econ, s_econ, vh_econ = Nx.svd ~full_matrices:false a in
  check_shape "svd econ u" [| 3; 3 |] u_econ;
  check_shape "svd econ vh" [| 3; 4 |] vh_econ;
  check_nx "svd s equal" s s_econ

let test_svdvals () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
  in
  let s = Nx.svdvals a in
  check_shape "svdvals shape" [| 3 |] s;
  let _, s_full, _ = Nx.svd a in
  check_nx "svdvals match svd" s s_full

(* ───── Eigen Tests ───── *)

let test_eigh () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let vals, vecs = Nx.eigh a in
  check_t ~eps:1e-5 "eigh vals" [| 2 |] [| 1.; 3. |] vals;
  let diag_vals =
    let zeros = Nx.zeros Nx.float32 [| 2; 2 |] in
    let z_with_diag = Nx.copy zeros in
    Nx.set_item [ 0; 0 ] (Nx.item [ 0 ] vals) z_with_diag;
    Nx.set_item [ 1; 1 ] (Nx.item [ 1 ] vals) z_with_diag;
    z_with_diag
  in
  let recon = Nx.matmul vecs (Nx.matmul diag_vals (Nx.transpose vecs)) in
  check_nx "eigh recon" a recon

let test_eigh_uplo () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 2.; 4.; 5.; 3.; 5.; 6. |]
  in
  let vals_l = Nx.eigh ~uplo:`L a |> fst in
  let vals_u = Nx.eigh ~uplo:`U a |> fst in
  check_nx "eigh uplo L=U" vals_l vals_u

let test_eigvals () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let vals = Nx.eigvals a in
  let vals_full, _ = Nx.eig a in
  check_nx "eigvals match eig" vals vals_full

let test_eigvalsh () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let vals = Nx.eigvalsh a in
  let vals_full, _ = Nx.eigh a in
  check_nx "eigvalsh match eigh" vals vals_full

(* ───── Advanced Norm Tests ───── *)

let test_norm_ord () =
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |] in
  let n_nuc = Nx.norm ~ord:`Nuc m in
  check (float 1e-3) "norm nuclear" 5.83095 (Nx.item [] n_nuc);
  let n_two = Nx.norm ~ord:`Two m in
  check (float 1e-3) "norm two" 5.46499 (Nx.item [] n_two);
  let n_neg_two = Nx.norm ~ord:`NegTwo m in
  check (float 1e-3) "norm neg two" 0.36597 (Nx.item [] n_neg_two)

let test_norm_keepdims () =
  let v = Nx.create Nx.float32 [| 3 |] [| 3.; 4.; 0. |] in
  let n = Nx.norm ~keepdims:true v in
  check_shape "norm keepdims" [| 1 |] n;
  check_t "norm keepdims value" [| 1 |] [| 5. |] n

let test_cond () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; 1. |] in
  let c = Nx.cond a in
  check_t "cond default" [||] [| 1. |] c;
  let c_inf = Nx.cond ~p:`Inf a in
  check_t "cond inf" [||] [| 1. |] c_inf

(* ───── Advanced Solve Tests ───── *)

let test_lstsq () =
  let a = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 1.; 1.; 2.; 1.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 3.; 6.; 9. |] in
  let x, _res, rank, _s = Nx.lstsq a b in
  check_shape "lstsq x" [| 2 |] x;
  check int "lstsq rank" 2 rank;
  let approx_b = Nx.matmul a x in
  check_nx ~epsilon:1e-5 "lstsq approx" b approx_b

let test_lstsq_rcond () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; 1e-10 |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 1.; 0. |] in
  let _, _, rank, _ = Nx.lstsq ~rcond:1e-8 a b in
  check int "lstsq rcond rank" 1 rank

let test_lstsq_underdetermined () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 0.; 2.; 3.; 2.; 4. |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 1.; 0. |] in

  let x, _res, rank, _s = Nx.lstsq ~rcond:1e-8 a b in
  check_shape "lstsq x underdetermined" [| 3 |] x;
  check int "lstsq rank underdetermined" 2 rank;
  let approx_b_underdetermined = Nx.matmul a x in
  check_nx "lstsq approx underdetermined" b approx_b_underdetermined

let test_pinv () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let pinv = Nx.pinv a in
  check_shape "pinv shape" [| 3; 2 |] pinv;
  let recon = Nx.matmul a (Nx.matmul pinv a) in
  check_nx "pinv recon" a recon

let test_pinv_singular () =
  let a = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  let pinv = Nx.pinv a in
  let recon = Nx.matmul a (Nx.matmul pinv a) in
  check_nx "pinv singular recon" a recon

let test_tensorsolve () =
  let a = Nx.create Nx.float32 [| 2; 2; 2; 2 |] (Array.init 16 float_of_int) in
  let b = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let x = Nx.tensorsolve a b in
  check_shape "tensorsolve shape" [| 2; 2 |] x;
  let recon = Nx.tensordot a x ~axes:([ 2; 3 ], [ 0; 1 ]) in
  check_nx ~epsilon:1e-5 "tensorsolve recon" b recon

let test_tensorsolve_axes () =
  let a =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let b = Nx.create Nx.float32 [| 3 |] [| 14.; 32.; 50. |] in
  let x = Nx.tensorsolve ~axes:[ 1 ] a b in
  (* Matrix is singular, so we get minimum norm solution via pinv *)
  check_t ~eps:1e-5 "tensorsolve axes" [| 3 |] [| 1.; 2.; 3. |] x

let test_tensorinv () =
  (* Use an invertible tensor *)
  let a =
    Nx.create Nx.float32 [| 2; 2; 2; 2 |]
      [|
        0.49671414;
        -0.1382643;
        0.64768857;
        1.5230298;
        -0.23415338;
        -0.23413695;
        1.5792128;
        0.7674347;
        -0.46947438;
        0.54256004;
        -0.46341768;
        -0.46572974;
        0.24196227;
        -1.9132802;
        -1.7249179;
        -0.5622875;
      |]
  in
  let inv = Nx.tensorinv ~ind:2 a in
  check_shape "tensorinv shape" [| 2; 2; 2; 2 |] inv;
  let recon = Nx.tensordot a inv ~axes:([ 2; 3 ], [ 0; 1 ]) in
  let id = Nx.eye Nx.float32 4 |> Nx.reshape [| 2; 2; 2; 2 |] in
  check_nx ~epsilon:1e-5 "tensorinv recon" id recon

let test_tensorinv_ind () =
  let a = Nx.create Nx.float32 [| 4; 4 |] (Array.init 16 float_of_int) in
  let inv = Nx.tensorinv ~ind:1 a in
  check_shape "tensorinv ind shape" [| 4; 4 |] inv

(* Test Suite Organization *)

let matmul_tests =
  [
    ("matmul 2d x 2d", `Quick, test_matmul_2d_2d);
    ("matmul 1d x 1d", `Quick, test_matmul_1d_1d);
    ("matmul 1d x 2d", `Quick, test_matmul_1d_2d);
    ("matmul 2d x 1d", `Quick, test_matmul_2d_1d);
    ("matmul batch", `Quick, test_matmul_batch);
    ("matmul broadcast batch", `Quick, test_matmul_broadcast_batch);
    ("matmul 2d @ 3d broadcast", `Quick, test_matmul_2d_3d_broadcast);
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

let solve_inverse_tests =
  [
    ("solve identity", `Quick, test_solve_identity);
    ("solve simple", `Quick, test_solve_simple);
    ("solve batch", `Quick, test_solve_batch);
    ("solve singular", `Quick, test_solve_singular);
    ("solve non-square", `Quick, test_solve_non_square);
    ("inv identity", `Quick, test_inv_identity);
    ("inv inverse", `Quick, test_inv_inverse);
    ("inv singular", `Quick, test_inv_singular);
  ]

let decomposition_tests =
  [
    ("qr shape", `Quick, test_qr_shape);
    ("qr property", `Quick, test_qr_property);
    ("qr orthogonal", `Quick, test_qr_orthogonal);
    ("svd shape", `Quick, test_svd_shape);
    ("svd property", `Quick, test_svd_property);
    ("cholesky posdef", `Quick, test_cholesky_posdef);
    ("cholesky property", `Quick, test_cholesky_property);
    ("eig shape", `Quick, test_eig_shape);
    ("eig property", `Quick, test_eig_property);
  ]

let norm_tests =
  [
    ("norm vector L1", `Quick, test_norm_vector_1);
    ("norm vector L2", `Quick, test_norm_vector_2);
    ("norm vector Linf", `Quick, test_norm_vector_inf);
    ("norm matrix Frobenius", `Quick, test_norm_matrix_fro);
    ("norm matrix L1", `Quick, test_norm_matrix_1);
    ("norm axis", `Quick, test_norm_axis);
    ("norm empty", `Quick, test_norm_empty);
  ]

let utility_tests =
  [
    ("det 2x2", `Quick, test_det_2x2);
    ("det singular", `Quick, test_det_singular);
    ("trace", `Quick, test_trace);
    ("diag extract", `Quick, test_diag_extract);
  ]

let advanced_utility_tests =
  [
    ("diagonal", `Quick, test_diagonal);
    ("diagonal edge", `Quick, test_diagonal_edge);
    ("matrix transpose", `Quick, test_matrix_transpose);
    ("trace offset", `Quick, test_trace_offset);
    ("det batch", `Quick, test_det_batch);
    ("slogdet", `Quick, test_slogdet);
    ("slogdet singular", `Quick, test_slogdet_singular);
    ("matrix rank", `Quick, test_matrix_rank);
    ("matrix rank tol", `Quick, test_matrix_rank_tol);
    ("matrix rank hermitian", `Quick, test_matrix_rank_hermitian);
    ( "matrix rank hermitian negative",
      `Quick,
      test_matrix_rank_hermitian_negative );
    ("matrix rank hermitian complex", `Quick, test_matrix_rank_hermitian_complex);
    ("pinv hermitian", `Quick, test_pinv_hermitian);
    ("pinv hermitian negative", `Quick, test_pinv_hermitian_negative);
    ("pinv hermitian complex", `Quick, test_pinv_hermitian_complex);
  ]

let product_tests =
  [
    ("vdot", `Quick, test_vdot);
    ("vdot complex", `Quick, test_vdot_complex);
    ("conjugate", `Quick, test_conjugate);
    ("vdot mismatch", `Quick, test_vdot_mismatch);
    ("vecdot", `Quick, test_vecdot);
    ("inner", `Quick, test_inner);
    ("inner mismatch", `Quick, test_inner_mismatch);
    ("outer", `Quick, test_outer);
    ("tensordot", `Quick, test_tensordot);
    ("tensordot mismatch", `Quick, test_tensordot_mismatch);
    ("kron", `Quick, test_kron);
    ("multi dot", `Quick, test_multi_dot);
    ("multi dot empty", `Quick, test_multi_dot_empty);
    ("matrix power", `Quick, test_matrix_power);
    ("matrix power singular", `Quick, test_matrix_power_singular);
    ("cross", `Quick, test_cross);
    ("cross invalid", `Quick, test_cross_invalid);
  ]

(* Dedicated suite for einsum; avoids duplication in product_tests *)
let einsum_tests =
  [
    ("einsum error cases", `Quick, test_einsum_error);
    ("einsum weighted broadcast", `Quick, einsum_weighted_broadcast);
    ("einsum complex fro inner", `Quick, einsum_complex_fro_inner);
    ("einsum int dot scalar", `Quick, einsum_int_dot_scalar);
    ("einsum axis order regression", `Quick, test_einsum_regression_axis_order);
    ("dot scalar i,i->", `Quick, einsum_dot_scalar);
    ("matmul ij,jk->ik", `Quick, einsum_matmul);
    ("transpose ij->ji", `Quick, einsum_transpose);
    ("outer i,j->ij", `Quick, einsum_outer);
    ("total sum ij->", `Quick, einsum_total_sum);
    ("diag extract ii->i", `Quick, einsum_diag_extract);
    ("batched diag ...ii->...i", `Quick, einsum_batched_diag);
    ("batched matmul ...ij,...jk->...ik", `Quick, einsum_batched_matmul);
    ("free order1 i,jk->jki", `Quick, einsum_free_order1);
    ("free order2 ij,klj->kli", `Quick, einsum_free_order2);
    ("mix reorder abc,bd->dac", `Quick, einsum_mix_reorder);
    ("chain ab,bc,cd->ad", `Quick, einsum_chain);
    ("diag sum ii", `Quick, einsum_diag_sum);
    ("hadamard vec i,i->i", `Quick, einsum_hadamard_vec);
    ("fro inner ij,ij->", `Quick, einsum_fro_inner);
    ("contract last ijk,k->ij", `Quick, einsum_contract_last);
    ("matvec ab,b->a", `Quick, einsum_matvec);
    ("contract 3d vec abc,c->ab", `Quick, einsum_contract_3d_vec);
    ("broadcast last dot ...i,...i->...", `Quick, einsum_broadcast_last_dot);
    ("move first axis i...->...i", `Quick, einsum_move_first_axis_to_last);
    ("rowwise dot ij,j->i", `Quick, einsum_rowwise_dot);
  ]

let advanced_decomposition_tests =
  [
    ("cholesky upper", `Quick, test_cholesky_upper);
    ("cholesky non posdef", `Quick, test_cholesky_non_posdef);
    ("qr mode", `Quick, test_qr_mode);
    ("svd full matrices", `Quick, test_svd_full_matrices);
    ("svdvals", `Quick, test_svdvals);
  ]

let eigen_tests =
  [
    ("eigh", `Quick, test_eigh);
    ("eigh uplo", `Quick, test_eigh_uplo);
    ("eigvals", `Quick, test_eigvals);
    ("eigvalsh", `Quick, test_eigvalsh);
  ]

let advanced_norm_tests =
  [
    ("norm ord", `Quick, test_norm_ord);
    ("norm keepdims", `Quick, test_norm_keepdims);
    ("cond", `Quick, test_cond);
  ]

let advanced_solve_tests =
  [
    ("lstsq", `Quick, test_lstsq);
    ("lstsq rcond", `Quick, test_lstsq_rcond);
    ("lstsq underdetermined", `Quick, test_lstsq_underdetermined);
    ("pinv", `Quick, test_pinv);
    ("pinv singular", `Quick, test_pinv_singular);
    ("tensorsolve", `Quick, test_tensorsolve);
    ("tensorsolve axes", `Quick, test_tensorsolve_axes);
    ("tensorinv", `Quick, test_tensorinv);
    ("tensorinv ind", `Quick, test_tensorinv_ind);
  ]

let suite =
  [
    ("Linalg :: Matrix Multiply", matmul_tests);
    ("Linalg :: Dot Product", dot_tests);
    ("Linalg :: Solve/Inverse", solve_inverse_tests);
    ("Linalg :: Decompositions", decomposition_tests);
    ("Linalg :: Norms", norm_tests);
    ("Linalg :: Utilities", utility_tests);
    ("Linalg :: Advanced Utilities", advanced_utility_tests);
    ("Linalg :: Product Ops", product_tests);
    ("Linalg :: Einsum", einsum_tests);
    ("Linalg :: Advanced Decompositions", advanced_decomposition_tests);
    ("Linalg :: Eigen", eigen_tests);
    ("Linalg :: Advanced Norms", advanced_norm_tests);
    ("Linalg :: Advanced Solve", advanced_solve_tests);
  ]

let () = Alcotest.run "Nx Linalg" suite
