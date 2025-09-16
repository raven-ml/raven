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
    check (float 1e-6) "dot scalar value" 32.0 (Nx.item [] result)

  (* ───── Solve Inverse Tests ───── *)

  let test_solve_identity ctx () =
    let identity = Nx.eye ctx Nx.float32 3 in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let x = Nx.solve identity b in
    check_t "solve identity" [| 3 |] [| 1.; 2.; 3. |] x

  let test_solve_simple ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in
    let b = Nx.create ctx Nx.float32 [| 2 |] [| 9.; 8. |] in
    let x = Nx.solve a b in
    let result = Nx.dot a x in
    check_nx "solve simple" b result

  let test_solve_batch ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 2; 3; 3 |]
        [|
          1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.; 2.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 2.;
        |]
    in
    let b = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6. |] in
    let x = Nx.solve a b in
    check_shape "solve batch shape" [| 2; 3 |] x

  let test_solve_singular ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
    let b = Nx.create ctx Nx.float32 [| 2 |] [| 1.; 2. |] in
    check_invalid_arg "solve singular" "solve: matrix is singular" (fun () ->
        ignore (Nx.solve a b))

  let test_solve_non_square ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b = Nx.create ctx Nx.float32 [| 2 |] [| 1.; 2. |] in
    check_invalid_arg "solve non-square"
      "solve: coefficient matrix must be square" (fun () ->
        ignore (Nx.solve a b))

  let test_inv_identity ctx () =
    let identity = Nx.eye ctx Nx.float32 3 in
    let inv = Nx.inv identity in
    check_nx "inv identity" identity inv

  let test_inv_inverse ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
    let inv_a = Nx.inv a in
    let inv_inv_a = Nx.inv inv_a in
    check_nx "inv inverse" a inv_inv_a

  let test_inv_singular ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
    check_invalid_arg "inv singular" "inv: matrix is singular" (fun () ->
        ignore (Nx.inv a))

  (* ───── Decomposition Tests ───── *)

  let test_qr_shape ctx () =
    let a = Nx.create ctx Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
    let q, r = Nx.qr a in
    check_shape "qr q shape" [| 4; 4 |] q;
    check_shape "qr r shape" [| 4; 3 |] r

  let test_qr_property ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
    in
    let q, r = Nx.qr a in
    let reconstructed = Nx.matmul q r in
    check_nx "qr property" a reconstructed

  let test_qr_orthogonal ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
    in
    let q, _ = Nx.qr a in
    let qt_q = Nx.matmul (Nx.transpose q) q in
    let identity = Nx.eye ctx Nx.float32 3 in
    check_nx "qr orthogonal" identity qt_q

  let test_svd_shape ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let u, s, vt = Nx.svd a in
    check_shape "svd u shape" [| 3; 3 |] u;
    check_shape "svd s shape" [| 3 |] s;
    check_shape "svd vt shape (V^H)" [| 3; 4 |] vt

  let test_svd_property ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
    in
    let u, s, vh = Nx.svd a in
    let s_diag = Nx.zeros ctx Nx.float32 [| 3; 3 |] in
    let s_float32 = Nx.cast Nx.float32 s in
    for i = 0 to 2 do
      let s_val = Nx.unsafe_get [ i ] s_float32 in
      Nx.unsafe_set [ i; i ] s_val s_diag
    done;
    let reconstructed = Nx.matmul u (Nx.matmul s_diag vh) in
    check_nx "svd property" a reconstructed

  let test_cholesky_posdef ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 0.; 0.; 1.; 1.; 0.; 1.; 1.; 1. |]
    in
    let posdef = Nx.matmul (Nx.transpose a) a in
    let l = Nx.cholesky posdef in
    check_shape "cholesky shape" [| 3; 3 |] l

  let test_cholesky_property ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 0.; 1.; 1. |] in
    let posdef = Nx.matmul (Nx.transpose a) a in
    let l = Nx.cholesky posdef in
    let reconstructed = Nx.matmul l (Nx.transpose l) in
    check_nx "cholesky property" posdef reconstructed

  let test_eig_shape ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
    in
    let eigenvalues, eigenvectors = Nx.eig a in
    check_shape "eig eigenvalues shape" [| 3 |] eigenvalues;
    check_shape "eig eigenvectors shape" [| 3; 3 |] eigenvectors

  let test_eig_property ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
    let eigenvalues, eigenvectors = Nx.eig a in
    (* Cast to float32 to match a's type *)
    let eigenvalues_f32 = Nx.cast Nx.float32 eigenvalues in
    let eigenvectors_f32 = Nx.cast Nx.float32 eigenvectors in
    let v1 = Nx.slice [ Nx.R (0, 2); Nx.I 0 ] eigenvectors_f32 in
    let lambda1 = Nx.unsafe_get [ 0 ] eigenvalues_f32 in
    let av1 = Nx.dot a v1 in
    let lambda1_scalar = Nx.scalar ctx Nx.float32 lambda1 in
    let lambda_v1 = Nx.mul lambda1_scalar v1 in
    check_nx "eig property" av1 lambda_v1

  (* ───── Norm Tests ───── *)

  let test_norm_vector_1 ctx () =
    let v = Nx.create ctx Nx.float32 [| 4 |] [| -1.; 2.; -3.; 4. |] in
    let result = Nx.norm ~ord:(`P 1.) v in
    check_t "norm L1" [||] [| 10.0 |] result

  let test_norm_vector_2 ctx () =
    let v = Nx.create ctx Nx.float32 [| 3 |] [| 3.; 4.; 0. |] in
    let result = Nx.norm v in
    check_t "norm L2" [||] [| 5.0 |] result

  let test_norm_vector_inf ctx () =
    let v = Nx.create ctx Nx.float32 [| 4 |] [| -1.; 2.; -5.; 4. |] in
    let result = Nx.norm ~ord:`Inf v in
    check_t "norm Linf" [||] [| 5.0 |] result

  let test_norm_matrix_fro ctx () =
    let m = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
    let result = Nx.norm ~ord:`Fro m in
    check_t ~eps:1e-5 "norm Frobenius" [||] [| 5.477226 |] result

  let test_norm_matrix_1 ctx () =
    let m = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; -2.; 3.; 4. |] in
    let result = Nx.norm ~ord:(`P 1.) m in
    check_t "norm matrix L1" [||] [| 6.0 |] result

  let test_norm_axis ctx () =
    let m = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let result = Nx.norm ~axes:[| 1 |] m in
    check_t ~eps:1e-5 "norm along axis" [| 2 |] [| 3.741657; 8.774964 |] result

  let test_norm_empty ctx () =
    let v = Nx.create ctx Nx.float32 [| 0 |] [||] in
    let result = Nx.norm v in
    check_t "norm empty" [||] [| 0.0 |] result

  (* ───── Linear Algebra Utilities ───── *)

  let test_det_2x2 ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 3.; 8.; 4.; 6. |] in
    let det = Nx.det a in
    check_t "det 2x2" [||] [| -14.0 |] det

  let test_det_singular ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
    let det = Nx.det a in
    check_t ~eps:1e-6 "det singular" [||] [| 0.0 |] det

  let test_trace ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
    in
    let tr = Nx.trace a in
    check_t "trace" [||] [| 15.0 |] tr

  let test_diag_extract ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
    in
    let diag = Nx.diagonal a in
    check_t "diag extract" [| 3 |] [| 1.; 5.; 9. |] diag

  (* ───── Additional Utility Tests ───── *)

  let test_diagonal ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
    let d = Nx.diagonal a in
    check_t "diagonal main" [| 3 |] [| 0.; 4.; 8. |] d;
    let d_offset = Nx.diagonal ~offset:1 a in
    check_t "diagonal offset 1" [| 2 |] [| 1.; 5. |] d_offset;
    let a_higher =
      Nx.create ctx Nx.float32 [| 2; 3; 3 |] (Array.init 18 float_of_int)
    in
    let d_higher = Nx.diagonal a_higher in
    check_shape "diagonal higher dim" [| 2; 3 |] d_higher

  let test_diagonal_edge ctx () =
    let a_empty = Nx.create ctx Nx.float32 [| 0; 0 |] [||] in
    let d_empty = Nx.diagonal a_empty in
    check_shape "diagonal empty" [| 0 |] d_empty;
    check_raises "diagonal invalid axes"
      (Invalid_argument "diagonal: axis1 = axis2") (fun () ->
        ignore (Nx.diagonal ~axis1:0 ~axis2:0 a_empty))

  let test_matrix_transpose ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let t = Nx.matrix_transpose a in
    check_shape "matrix transpose shape" [| 3; 2 |] t;
    check_t "matrix transpose values" [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |] t;
    let a1d = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let t1d = Nx.matrix_transpose a1d in
    check_t "matrix transpose 1d unchanged" [| 3 |] [| 1.; 2.; 3. |] t1d

  let test_trace_offset ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
    in
    let tr_offset = Nx.trace ~offset:1 a in
    check_t "trace offset 1" [||] [| 8. |] tr_offset

  let test_det_batch ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 2; 2; 2 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    in
    let d = Nx.det a in
    check_shape "det batch" [| 2 |] d;
    check_t "det batch values" [| 2 |] [| -2.; -2. |] d

  let test_slogdet ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 3.; 8.; 4.; 6. |] in
    let sign, logdet = Nx.slogdet a in
    check_t "slogdet sign" [||] [| -1. |] sign;
    check (float 1e-5) "slogdet logdet" (log 14.) (Nx.item [] logdet)

  let test_slogdet_singular ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
    let sign, logdet = Nx.slogdet a in
    check_t "slogdet singular sign" [||] [| 0. |] sign;
    check (float 1e-5) "slogdet singular logdet" neg_infinity
      (Nx.item [] logdet)

  let test_matrix_rank ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
    let r = Nx.matrix_rank a in
    check int "matrix rank full" 2 r;
    let a_low =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 2.; 4.; 6.; 3.; 6.; 9. |]
    in
    let r_low = Nx.matrix_rank a_low in
    check int "matrix_rank low" 1 r_low

  let test_matrix_rank_tol ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; 1e-10 |] in
    let r = Nx.matrix_rank ~tol:1e-8 a in
    check int "matrix_rank with tol" 1 r

  (* ───── Product Ops Tests ───── *)

  let test_vdot ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
    let res = Nx.vdot a b in
    check_t "vdot 1d" [||] [| 32. |] res;
    let a2 = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let res2 = Nx.vdot a2 b in
    check_t "vdot flatten" [||] [| 4. +. 10. +. 18. +. 16. +. 25. +. 36. |] res2

  let test_vdot_mismatch ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 4 |] [| 4.; 5.; 6.; 7. |] in
    check_raises "vdot mismatch"
      (Invalid_argument "vdot: different number of elements") (fun () ->
        ignore (Nx.vdot a b))

  let test_vecdot ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b =
      Nx.create ctx Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    let res = Nx.vecdot a b in
    check_t "vecdot default axis" [| 2 |] [| 50.; 167. |] res;
    let res_axis0 = Nx.vecdot ~axis:0 a b in
    check_t "vecdot axis 0" [| 3 |] [| 47.; 71.; 99. |] res_axis0

  let test_inner ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
    let res = Nx.inner a b in
    check_t "inner 1d" [||] [| 32. |] res;
    let a2 = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let res2 = Nx.inner a2 a in
    check_t "inner higher" [| 2 |] [| 14.; 32. |] res2

  let test_inner_mismatch ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 4 |] [| 4.; 5.; 6.; 7. |] in
    check_raises "inner mismatch"
      (Invalid_argument "inner: last dimensions differ") (fun () ->
        ignore (Nx.inner a b))

  let test_outer ctx () =
    let a = Nx.create ctx Nx.float32 [| 2 |] [| 1.; 2. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 3.; 4.; 5. |] in
    let res = Nx.outer a b in
    check_t "outer" [| 2; 3 |] [| 3.; 4.; 5.; 6.; 8.; 10. |] res;
    let a_scalar = Nx.create ctx Nx.float32 [||] [| 2. |] in
    let res_scalar = Nx.outer a_scalar b in
    check_t "outer scalar" [| 3 |] [| 6.; 8.; 10. |] res_scalar

  let test_tensordot ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b =
      Nx.create ctx Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    let res = Nx.tensordot a b in
    check_t "tensordot default" [| 2; 2 |] [| 58.; 64.; 139.; 154. |] res;
    let res_axes = Nx.tensordot ~axes:([| 0 |], [| 1 |]) a b in
    check_shape "tensordot custom axes" [| 3; 3 |] res_axes

  let test_tensordot_mismatch ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b = Nx.create ctx Nx.float32 [| 4; 2 |] (Array.init 8 float_of_int) in
    check_raises "tensordot mismatch"
      (Invalid_argument "tensordot: axes have different sizes") (fun () ->
        ignore (Nx.tensordot ~axes:([| 1 |], [| 0 |]) a b))

  let test_einsum_error ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b =
      Nx.create ctx Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    check_raises "einsum no input operands"
      (Invalid_argument "einsum: no input operands") (fun () ->
        ignore (Nx.einsum "" [||]));
    check_raises "einsum bad format"
      (Invalid_argument
         "einsum: subscript must be of form [a-z]+(,[a-z]+)*->[a-z]+")
      (fun () -> ignore (Nx.einsum "IJ,JK-IK" [| a; b |]));
    check_raises "einsum wrong inputs"
      (Invalid_argument "einsum: number of inputs must equal number of operands")
      (fun () -> ignore (Nx.einsum "ij->ij" [| a; b |]));
    check_raises "einsum repeated index"
      (Invalid_argument "einsum: operand 0 must have distinct a-z characters")
      (fun () -> ignore (Nx.einsum "iij,jk->ik" [| a; b |]));
    check_raises "einsum mismatched rank"
      (Invalid_argument "einsum: rank of input 'ijl' must match operand 0")
      (fun () -> ignore (Nx.einsum "ijl,jk->ik" [| a; b |]));
    check_raises "einsum contracted vars mismatch"
      (Invalid_argument
         "einsum: contracted input vars 'il' must match output vars 'ki'")
      (fun () -> ignore (Nx.einsum "ij,jl->ki" [| a; b |]));
    check_raises "einsum dimension mismatch"
      (Invalid_argument
         "einsum: index var 'j' must have consistent dimensions (3 on the \
          left, 2 on the right)") (fun () ->
        ignore (Nx.einsum "ij,kj->ik" [| a; b |]))

  let test_einsum ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b =
      Nx.create ctx Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    let c =
      Nx.create ctx Nx.float32 [| 2; 2; 2 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    in
    let res_matmul = Nx.einsum "ij,jk->ik" [| a; b |] in
    check_t "einsum matmul" [| 2; 2 |] [| 58.; 64.; 139.; 154. |] res_matmul;
    let res_diag = Nx.einsum "ii->i" [| a |] in
    check_t "einsum diag" [| 2 |] [| 1.; 5. |] res_diag;
    let res_trans = Nx.einsum "ij->ji" [| a |] in
    check_t "einsum transpose" [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |] res_trans;
    let res_three_way = Nx.einsum "xy,yz,zkw->xkw" [| a; b; c |] in
    check_t "einsum three-way" [| 2; 2; 2 |]
      [| 378.; 500.; 622.; 744.; 909.; 1202.; 1495.; 1788. |]
      res_three_way;
    (* let res_scalar = *)
    (* let p = Nx.create ctx Nx.int [| 2 |] [| 1; 2 |] in *)
    (* let q = Nx.create ctx Nx.int [| 2; 2 |] [| 3; 4; 5; 6 |] in *)
    (* let r = Nx.create ctx Nx.int [| 2; 2 |] [| 7; 8; 9; 10 |] in *)
    (* Nx.einsum "z,mz,zm->" [| p; q; r |] in *)
    (* check_t "einsum scalar" [| 2 ; 2; 2 |] [| 253 |] res_scalar; *)
    let res_outer = Nx.einsum "ij,km->ijkm" [| a; b |] in
    check_t "einsum outer" [| 2; 3; 3; 2 |]
      [|
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        14.;
        16.;
        18.;
        20.;
        22.;
        24.;
        21.;
        24.;
        27.;
        30.;
        33.;
        36.;
        28.;
        32.;
        36.;
        40.;
        44.;
        48.;
        35.;
        40.;
        45.;
        50.;
        55.;
        60.;
        42.;
        48.;
        54.;
        60.;
        66.;
        72.;
      |]
      res_outer

  let test_kron ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
    let b = Nx.create ctx Nx.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
    let res = Nx.kron a b in
    check_t "kron" [| 4; 4 |]
      [|
        5.;
        6.;
        10.;
        12.;
        7.;
        8.;
        14.;
        16.;
        15.;
        18.;
        20.;
        24.;
        21.;
        24.;
        28.;
        32.;
      |]
      res

  let test_multi_dot ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let c =
      Nx.create ctx Nx.float32 [| 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    in
    let res = Nx.multi_dot [| a; b; c |] in
    let manual = Nx.matmul a (Nx.matmul b c) in
    check_nx "multi_dot" manual res

  let test_multi_dot_empty _ctx () =
    check_raises "multi_dot empty" (Invalid_argument "multi_dot: empty array")
      (fun () -> ignore (Nx.multi_dot [||]))

  let test_matrix_power ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 1.; 1.; 0. |] in
    let pow3 = Nx.matrix_power a 3 in
    check_t "matrix_power positive" [| 2; 2 |] [| 3.; 2.; 2.; 1. |] pow3;
    let pow0 = Nx.matrix_power a 0 in
    let id = Nx.eye ctx Nx.float32 2 in
    check_nx "matrix_power zero" id pow0;
    let pow_neg2 = Nx.matrix_power a (-2) in
    let inv = Nx.inv a in
    let inv2 = Nx.matmul inv inv in
    check_nx "matrix_power negative" inv2 pow_neg2

  let test_matrix_power_singular ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
    check_raises "matrix_power singular negative"
      (Invalid_argument "matrix_power: singular for negative exponent")
      (fun () -> ignore (Nx.matrix_power a (-1)))

  let test_cross ctx () =
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
    let res = Nx.cross a b in
    check_t "cross 3d" [| 3 |] [| -3.; 6.; -3. |] res;
    let a_batch =
      Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let b_batch =
      Nx.create ctx Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    let res_batch = Nx.cross ~axis:1 a_batch b_batch in
    check_shape "cross batch" [| 2; 3 |] res_batch

  let test_cross_invalid ctx () =
    let a = Nx.create ctx Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
    let b = Nx.create ctx Nx.float32 [| 4 |] [| 5.; 6.; 7.; 8. |] in
    check_raises "cross invalid dim" (Invalid_argument "cross: axis dim not 3")
      (fun () -> ignore (Nx.cross a b))

  (* ───── Advanced Decomposition Tests ───── *)

  let test_cholesky_upper ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
    let u = Nx.cholesky ~upper:true a in
    let recon = Nx.matmul (Nx.transpose u) u in
    check_nx "cholesky upper" a recon

  let test_cholesky_non_posdef ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
    check_raises "cholesky non posdef"
      (Invalid_argument "cholesky: not positive-definite") (fun () ->
        ignore (Nx.cholesky a))

  let test_qr_mode ctx () =
    let a = Nx.create ctx Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
    let q_red, r_red = Nx.qr ~mode:`Reduced a in
    check_shape "qr reduced q" [| 4; 3 |] q_red;
    check_shape "qr reduced r" [| 3; 3 |] r_red;
    let q_comp, r_comp = Nx.qr ~mode:`Complete a in
    check_shape "qr complete q" [| 4; 4 |] q_comp;
    check_shape "qr complete r" [| 4; 3 |] r_comp

  let test_svd_full_matrices ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let u, s, vh = Nx.svd ~full_matrices:true a in
    check_shape "svd full u" [| 3; 3 |] u;
    check_shape "svd full vh" [| 4; 4 |] vh;
    let u_econ, s_econ, vh_econ = Nx.svd ~full_matrices:false a in
    check_shape "svd econ u" [| 3; 3 |] u_econ;
    check_shape "svd econ vh" [| 3; 4 |] vh_econ;
    check_nx "svd s equal" s s_econ

  let test_svdvals ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |]
    in
    let s = Nx.svdvals a in
    check_shape "svdvals shape" [| 3 |] s;
    let _, s_full, _ = Nx.svd a in
    check_nx "svdvals match svd" s s_full

  (* ───── Eigen Tests ───── *)

  let test_eigh ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
    let vals, vecs = Nx.eigh a in
    check_t ~eps:1e-5 "eigh vals" [| 2 |] [| 1.; 3. |] vals;
    let diag_vals =
      let zeros = Nx.zeros ctx Nx.float32 [| 2; 2 |] in
      let z_with_diag = Nx.copy zeros in
      Nx.unsafe_set [ 0; 0 ] (Nx.unsafe_get [ 0 ] vals) z_with_diag;
      Nx.unsafe_set [ 1; 1 ] (Nx.unsafe_get [ 1 ] vals) z_with_diag;
      z_with_diag
    in
    let recon = Nx.matmul vecs (Nx.matmul diag_vals (Nx.transpose vecs)) in
    check_nx "eigh recon" a recon

  let test_eigh_uplo ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 2.; 4.; 5.; 3.; 5.; 6. |]
    in
    let vals_l = Nx.eigh ~uplo:`L a |> fst in
    let vals_u = Nx.eigh ~uplo:`U a |> fst in
    check_nx "eigh uplo L=U" vals_l vals_u

  let test_eigvals ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
    let vals = Nx.eigvals a in
    let vals_full, _ = Nx.eig a in
    check_nx "eigvals match eig" vals vals_full

  let test_eigvalsh ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
    let vals = Nx.eigvalsh a in
    let vals_full, _ = Nx.eigh a in
    check_nx "eigvalsh match eigh" vals vals_full

  (* ───── Advanced Norm Tests ───── *)

  let test_norm_ord ctx () =
    let m = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |] in
    let n_nuc = Nx.norm ~ord:`Nuc m in
    check (float 1e-5) "norm nuclear" 5.477 (Nx.item [] n_nuc);
    let n_two = Nx.norm ~ord:`Two m in
    check (float 1e-5) "norm two" 5.477 (Nx.item [] n_two);
    let n_neg_two = Nx.norm ~ord:`NegTwo m in
    check (float 1e-5) "norm neg two" 0.366 (Nx.item [] n_neg_two)

  let test_norm_keepdims ctx () =
    let v = Nx.create ctx Nx.float32 [| 3 |] [| 3.; 4.; 0. |] in
    let n = Nx.norm ~keepdims:true v in
    check_shape "norm keepdims" [| 1 |] n;
    check_t "norm keepdims value" [| 1 |] [| 5. |] n

  let test_cond ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; 1. |] in
    let c = Nx.cond a in
    check_t "cond default" [||] [| 1. |] c;
    let c_inf = Nx.cond ~p:`Inf a in
    check_t "cond inf" [||] [| 1. |] c_inf

  (* ───── Advanced Solve Tests ───── *)

  let test_lstsq ctx () =
    let a = Nx.create ctx Nx.float32 [| 3; 2 |] [| 1.; 1.; 1.; 2.; 1.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 3.; 6.; 9. |] in
    let x, _res, rank, _s = Nx.lstsq a b in
    check_shape "lstsq x" [| 2 |] x;
    check int "lstsq rank" 2 rank;
    let approx_b = Nx.matmul a x in
    check_nx "lstsq approx" b approx_b

  let test_lstsq_rcond ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 0.; 0.; 1e-10 |] in
    let b = Nx.create ctx Nx.float32 [| 2 |] [| 1.; 0. |] in
    let _, _, rank, _ = Nx.lstsq ~rcond:1e-8 a b in
    check int "lstsq rcond rank" 1 rank

  let test_pinv ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let pinv = Nx.pinv a in
    check_shape "pinv shape" [| 3; 2 |] pinv;
    let recon = Nx.matmul a (Nx.matmul pinv a) in
    check_nx "pinv recon" a recon

  let test_pinv_singular ctx () =
    let a = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
    let pinv = Nx.pinv a in
    let recon = Nx.matmul a (Nx.matmul pinv a) in
    check_nx "pinv singular recon" a recon

  let test_tensorsolve ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 2; 2; 2; 2 |] (Array.init 16 float_of_int)
    in
    let b = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
    let x = Nx.tensorsolve a b in
    check_shape "tensorsolve shape" [| 2; 2 |] x;
    let recon = Nx.tensordot a x ~axes:([| 2; 3 |], [| 0; 1 |]) in
    check_nx "tensorsolve recon" b recon

  let test_tensorsolve_axes ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
    in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 14.; 32.; 50. |] in
    let x = Nx.tensorsolve ~axes:[ 0 ] a b in
    check_t "tensorsolve axes" [| 3 |] [| 1.; 2.; 3. |] x

  let test_tensorinv ctx () =
    let a =
      Nx.create ctx Nx.float32 [| 2; 2; 2; 2 |] (Array.init 16 float_of_int)
    in
    let inv = Nx.tensorinv ~ind:2 a in
    check_shape "tensorinv shape" [| 2; 2; 2; 2 |] inv;
    let recon = Nx.tensordot a inv ~axes:([| 2; 3 |], [| 0; 1 |]) in
    let id = Nx.eye ctx Nx.float32 4 |> Nx.reshape [| 2; 2; 2; 2 |] in
    check_nx "tensorinv recon" id recon

  let test_tensorinv_ind ctx () =
    let a = Nx.create ctx Nx.float32 [| 4; 4 |] (Array.init 16 float_of_int) in
    let inv = Nx.tensorinv ~ind:1 a in
    check_shape "tensorinv ind shape" [| 4; 4 |] inv

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

  let solve_inverse_tests ctx =
    [
      ("solve identity", `Quick, test_solve_identity ctx);
      ("solve simple", `Quick, test_solve_simple ctx);
      ("solve batch", `Quick, test_solve_batch ctx);
      ("solve singular", `Quick, test_solve_singular ctx);
      ("solve non-square", `Quick, test_solve_non_square ctx);
      ("inv identity", `Quick, test_inv_identity ctx);
      ("inv inverse", `Quick, test_inv_inverse ctx);
      ("inv singular", `Quick, test_inv_singular ctx);
    ]

  let decomposition_tests ctx =
    [
      ("qr shape", `Quick, test_qr_shape ctx);
      ("qr property", `Quick, test_qr_property ctx);
      ("qr orthogonal", `Quick, test_qr_orthogonal ctx);
      ("svd shape", `Quick, test_svd_shape ctx);
      ("svd property", `Quick, test_svd_property ctx);
      ("cholesky posdef", `Quick, test_cholesky_posdef ctx);
      ("cholesky property", `Quick, test_cholesky_property ctx);
      ("eig shape", `Quick, test_eig_shape ctx);
      ("eig property", `Quick, test_eig_property ctx);
    ]

  let norm_tests ctx =
    [
      ("norm vector L1", `Quick, test_norm_vector_1 ctx);
      ("norm vector L2", `Quick, test_norm_vector_2 ctx);
      ("norm vector Linf", `Quick, test_norm_vector_inf ctx);
      ("norm matrix Frobenius", `Quick, test_norm_matrix_fro ctx);
      ("norm matrix L1", `Quick, test_norm_matrix_1 ctx);
      ("norm axis", `Quick, test_norm_axis ctx);
      ("norm empty", `Quick, test_norm_empty ctx);
    ]

  let utility_tests ctx =
    [
      ("det 2x2", `Quick, test_det_2x2 ctx);
      ("det singular", `Quick, test_det_singular ctx);
      ("trace", `Quick, test_trace ctx);
      ("diag extract", `Quick, test_diag_extract ctx);
    ]

  let advanced_utility_tests ctx =
    [
      ("diagonal", `Quick, test_diagonal ctx);
      ("diagonal edge", `Quick, test_diagonal_edge ctx);
      ("matrix transpose", `Quick, test_matrix_transpose ctx);
      ("trace offset", `Quick, test_trace_offset ctx);
      ("det batch", `Quick, test_det_batch ctx);
      ("slogdet", `Quick, test_slogdet ctx);
      ("slogdet singular", `Quick, test_slogdet_singular ctx);
      ("matrix rank", `Quick, test_matrix_rank ctx);
      ("matrix rank tol", `Quick, test_matrix_rank_tol ctx);
    ]

  let product_tests ctx =
    [
      ("vdot", `Quick, test_vdot ctx);
      ("vdot mismatch", `Quick, test_vdot_mismatch ctx);
      ("vecdot", `Quick, test_vecdot ctx);
      ("inner", `Quick, test_inner ctx);
      ("inner mismatch", `Quick, test_inner_mismatch ctx);
      ("outer", `Quick, test_outer ctx);
      ("tensordot", `Quick, test_tensordot ctx);
      ("tensordot mismatch", `Quick, test_tensordot_mismatch ctx);
      ("einsum error", `Quick, test_einsum_error ctx);
      ("einsum", `Quick, test_einsum ctx);
      ("kron", `Quick, test_kron ctx);
      ("multi dot", `Quick, test_multi_dot ctx);
      ("multi dot empty", `Quick, test_multi_dot_empty ctx);
      ("matrix power", `Quick, test_matrix_power ctx);
      ("matrix power singular", `Quick, test_matrix_power_singular ctx);
      ("cross", `Quick, test_cross ctx);
      ("cross invalid", `Quick, test_cross_invalid ctx);
    ]

  let advanced_decomposition_tests ctx =
    [
      ("cholesky upper", `Quick, test_cholesky_upper ctx);
      ("cholesky non posdef", `Quick, test_cholesky_non_posdef ctx);
      ("qr mode", `Quick, test_qr_mode ctx);
      ("svd full matrices", `Quick, test_svd_full_matrices ctx);
      ("svdvals", `Quick, test_svdvals ctx);
    ]

  let eigen_tests ctx =
    [
      ("eigh", `Quick, test_eigh ctx);
      ("eigh uplo", `Quick, test_eigh_uplo ctx);
      ("eigvals", `Quick, test_eigvals ctx);
      ("eigvalsh", `Quick, test_eigvalsh ctx);
    ]

  let advanced_norm_tests ctx =
    [
      ("norm ord", `Quick, test_norm_ord ctx);
      ("norm keepdims", `Quick, test_norm_keepdims ctx);
      ("cond", `Quick, test_cond ctx);
    ]

  let advanced_solve_tests ctx =
    [
      ("lstsq", `Quick, test_lstsq ctx);
      ("lstsq rcond", `Quick, test_lstsq_rcond ctx);
      ("pinv", `Quick, test_pinv ctx);
      ("pinv singular", `Quick, test_pinv_singular ctx);
      ("tensorsolve", `Quick, test_tensorsolve ctx);
      ("tensorsolve axes", `Quick, test_tensorsolve_axes ctx);
      ("tensorinv", `Quick, test_tensorinv ctx);
      ("tensorinv ind", `Quick, test_tensorinv_ind ctx);
    ]

  let suite backend_name ctx =
    [
      ("Linalg :: " ^ backend_name ^ " Matrix Multiply", matmul_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Dot Product", dot_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Solve/Inverse", solve_inverse_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Decompositions", decomposition_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Norms", norm_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Utilities", utility_tests ctx);
      ( "Linalg :: " ^ backend_name ^ " Advanced Utilities",
        advanced_utility_tests ctx );
      ("Linalg :: " ^ backend_name ^ " Product Ops", product_tests ctx);
      ( "Linalg :: " ^ backend_name ^ " Advanced Decompositions",
        advanced_decomposition_tests ctx );
      ("Linalg :: " ^ backend_name ^ " Eigen", eigen_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Advanced Norms", advanced_norm_tests ctx);
      ("Linalg :: " ^ backend_name ^ " Advanced Solve", advanced_solve_tests ctx);
    ]
end
