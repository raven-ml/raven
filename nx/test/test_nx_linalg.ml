open Alcotest
module Nd = Nx

let nx_int32 : (int32, Nd.int32_elt) Nd.t testable =
  Alcotest.testable Nx.pp Nx.array_equal

let nx_float32 : (float, Nd.float32_elt) Nd.t testable =
  Alcotest.testable Nx.pp Nx.array_equal

(* Helper function for approximate equality of float tensors *)
let check_approx_equal ?(epsilon = 1e-6) msg t1 t2 =
  let diff = Nd.sub t1 t2 in
  let max_diff = Nd.get_item [||] (Nd.max (Nd.abs diff)) in
  Alcotest.(check bool) msg true (max_diff < epsilon)

(* Existing tests *)
let test_dot_1d_1d () =
  let t1 = Nd.create Nd.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let t2 = Nd.create Nd.int32 [| 3 |] [| 4l; 5l; 6l |] in
  let result = Nd.dot t1 t2 in
  let expected = Nd.scalar Nd.int32 32l in
  check nx_int32 "Dot product 1D x 1D" expected result

let test_dot_2d_2d () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let result = Nd.dot t1 t2 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 19.0; 22.0; 43.0; 50.0 |] in
  check nx_float32 "Dot product 2D x 2D" expected result

let test_dot_2d_1d () =
  let t1 = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let t2 = Nd.create Nd.int32 [| 2 |] [| 10l; 20l |] in
  let result = Nd.dot t1 t2 in
  let expected = Nd.create Nd.int32 [| 2 |] [| 50l; 110l |] in
  check nx_int32 "Dot product 2D x 1D" expected result

let test_dot_3d_2d () =
  let t1 = Nd.create Nd.float32 [| 2; 2; 3 |] (Array.init 12 float_of_int) in
  let t2 = Nd.create Nd.float32 [| 3; 2 |] (Array.init 6 float_of_int) in
  let result = Nd.dot t1 t2 in
  let expected =
    Nd.create Nd.float32 [| 2; 2; 2 |]
      [| 10.; 13.; 28.; 40.; 46.; 67.; 64.; 94. |]
  in
  check nx_float32 "Dot product 3D x 2D" expected result

let test_matmul_2x3_3x2 () =
  let t1 = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let t2 =
    Nd.create Nd.float32 [| 3; 2 |] [| 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |]
  in
  let result = Nd.matmul t1 t2 in
  let expected =
    Nd.create Nd.float32 [| 2; 2 |] [| 58.0; 64.0; 139.0; 154.0 |]
  in
  check nx_float32 "Matmul 2x3 with 3x2" expected result

let test_matmul_incompatible_shapes () =
  let t1 = Nd.create Nd.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  let t2 = Nd.create Nd.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check_raises "Matmul incompatible shapes"
    (Invalid_argument
       "matmul: incompatible shapes for matrix multiplication ([2; 3] vs [2; \
        3] -> inner dimensions 3 vs 2 mismatch)") (fun () ->
      ignore (Nd.matmul t1 t2))

let test_convolve1d () =
  let signal = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let kernel = Nd.create Nd.float32 [| 2 |] [| 1.0; 1.0 |] in
  let result = Nd.convolve1d signal kernel in
  let expected = Nd.create Nd.float32 [| 5 |] [| 1.0; 3.0; 5.0; 7.0; 4.0 |] in
  check nx_float32 "Convolve 1D" expected result

let test_inv_correct () =
  let a = Nd.create Nd.float32 [| 2; 2 |] [| 2.0; 0.0; 0.0; 2.0 |] in
  let inv_a = Nd.inv a in
  let expected_inv = Nd.create Nd.float32 [| 2; 2 |] [| 0.5; 0.0; 0.0; 0.5 |] in
  check_approx_equal "Inverse of diagonal matrix" expected_inv inv_a;
  let identity = Nd.eye Nd.float32 2 in
  let product = Nd.matmul a inv_a in
  check_approx_equal "A * inv(A) should be identity" identity product

let test_inv_non_square () =
  let a = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check_raises "Inv on non-square matrix"
    (Invalid_argument "inv: input must be a square matrix") (fun () ->
      ignore (Nd.inv a))

let test_inv_singular () =
  let a = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 2.0; 4.0 |] in
  check_raises "Inv on singular matrix"
    (Invalid_argument "inv: matrix is singular") (fun () -> ignore (Nd.inv a))

let test_solve_2x2 () =
  let a = Nd.create Nd.float32 [| 2; 2 |] [| 3.0; 1.0; 1.0; 2.0 |] in
  let b = Nd.create Nd.float32 [| 2 |] [| 9.0; 8.0 |] in
  let x = Nd.solve a b in
  let reconstructed_b = Nd.dot a x in
  check_approx_equal "Solve 2x2 system" b reconstructed_b

let test_solve_3x2 () =
  let a = Nd.create Nd.float32 [| 3; 2 |] [| 1.0; 1.0; 1.0; 2.0; 1.0; 3.0 |] in
  let b = Nd.create Nd.float32 [| 3 |] [| 6.0; 8.0; 10.0 |] in
  let x = Nd.solve a b in
  let reconstructed_b = Nd.dot a x in
  check_approx_equal "Solve 3x2 system" b reconstructed_b

let test_solve_with_matrix_b () =
  let a = Nd.create Nd.float32 [| 2; 2 |] [| 3.0; 1.0; 1.0; 2.0 |] in
  let b = Nd.create Nd.float32 [| 2; 2 |] [| 9.0; 7.0; 8.0; 6.0 |] in
  let x = Nd.solve a b in
  let reconstructed_b = Nd.matmul a x in
  check nx_float32 "Solve with matrix b" b reconstructed_b

let test_solve_not_2d () =
  let a = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Nd.create Nd.float32 [| 3 |] [| 4.0; 5.0; 6.0 |] in
  check_raises "Solve with 1-D a" (Invalid_argument "solve: A must be 2D")
    (fun () -> ignore (Nd.solve a b))

let test_svd_reconstruction () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let u, s, v = Nd.svd t in
  let s_diag =
    Nd.create Nd.float32 [| 2; 2 |]
      [| Nd.get_item [| 0 |] s; 0.0; 0.0; Nd.get_item [| 1 |] s |]
  in
  let reconstructed = Nd.matmul u (Nd.matmul s_diag (Nd.transpose v)) in
  check nx_float32 "SVD reconstruction" t reconstructed

let test_svd_not_2d () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  check_raises "SVD on 1-D tensor" (Invalid_argument "svd: input must be 2D")
    (fun () -> ignore (Nd.svd t))

let test_eig_decomposition () =
  let a = Nd.create Nd.float32 [| 2; 2 |] [| 2.0; 1.0; 1.0; 2.0 |] in
  let w, v = Nd.eig a in
  let w_diag =
    Nd.create Nd.float32 [| 2; 2 |]
      [| Nd.get_item [| 0 |] w; 0.0; 0.0; Nd.get_item [| 1 |] w |]
  in
  let left = Nd.matmul a v in
  let right = Nd.matmul v w_diag in
  check_approx_equal "Eig decomposition: a * v = v * diag(w)" left right

let test_eig_not_square () =
  let a = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check_raises "Eig on non-square matrix"
    (Invalid_argument "eig: input must be a square matrix") (fun () ->
      ignore (Nd.eig a))

let test_eigh_decomposition () =
  let a = Nd.create Nd.float32 [| 2; 2 |] [| 2.0; 1.0; 1.0; 2.0 |] in
  let w, v = Nd.eigh a in
  let w_diag =
    Nd.create Nd.float32 [| 2; 2 |]
      [| Nd.get_item [| 0 |] w; 0.0; 0.0; Nd.get_item [| 1 |] w |]
  in
  let left = Nd.matmul a v in
  let right = Nd.matmul v w_diag in
  check_approx_equal "Eigh decomposition: a * v = v * diag(w)" left right;
  let vt_v = Nd.matmul (Nd.transpose v) v in
  let identity = Nd.eye Nd.float32 2 in
  check_approx_equal "Eigh: v^T * v should be identity" identity vt_v

let test_eigh_not_square () =
  let a = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check_raises "Eigh on non-square matrix"
    (Invalid_argument "eigh: input must be a square matrix") (fun () ->
      ignore (Nd.eigh a))

(* Updated operation_tests list *)
let operation_tests =
  [
    ("Dot product 1D x 1D", `Quick, test_dot_1d_1d);
    ("Dot product 2D x 2D", `Quick, test_dot_2d_2d);
    ("Dot product 2D x 1D", `Quick, test_dot_2d_1d);
    ("Dot product 3D x 2D", `Quick, test_dot_3d_2d);
    ("Matmul 2x3 with 3x2", `Quick, test_matmul_2x3_3x2);
    ("Matmul incompatible shapes", `Quick, test_matmul_incompatible_shapes);
    ("Convolve 1D", `Quick, test_convolve1d);
    ("Inverse correct", `Quick, test_inv_correct);
    ("Inverse non-square", `Quick, test_inv_non_square);
    ("Inverse singular", `Quick, test_inv_singular);
    ("Solve 2x2 system", `Quick, test_solve_2x2);
    ("Solve 3x2 system", `Quick, test_solve_3x2);
    ("Solve with matrix b", `Quick, test_solve_with_matrix_b);
    ("Solve not 2D", `Quick, test_solve_not_2d);
    ("SVD reconstruction", `Quick, test_svd_reconstruction);
    ("SVD not 2D", `Quick, test_svd_not_2d);
    ("Eig decomposition", `Quick, test_eig_decomposition);
    ("Eig not square", `Quick, test_eig_not_square);
    ("Eigh decomposition", `Quick, test_eigh_decomposition);
    ("Eigh not square", `Quick, test_eigh_not_square);
  ]

let () =
  Printexc.record_backtrace true;
  Alcotest.run "Nx Operations" [ ("Operations", operation_tests) ]
