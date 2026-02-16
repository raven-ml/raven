(** Solve systems, decompose matrices, and fit models — linear algebra made
    practical.

    Fit a line to noisy data with least squares, verify matrix inverses,
    decompose matrices with SVD and eigendecomposition. *)

open Nx
open Nx.Infix

let () =
  (* --- Matrix multiplication --- *)
  let a = create float64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let b = create float64 [| 3; 2 |] [| 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |] in
  Printf.printf "A (2×3):\n%s\n" (data_to_string a);
  Printf.printf "B (3×2):\n%s\n" (data_to_string b);
  Printf.printf "A @@ B:\n%s\n\n" (data_to_string (a @@ b));

  (* --- Dot product --- *)
  let u = create float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let v = create float64 [| 3 |] [| 4.0; 5.0; 6.0 |] in
  Printf.printf "u · v = %s\n\n" (data_to_string (u <.> v));

  (* --- Solving linear systems: A x = b --- *)
  let coeff =
    create float64 [| 3; 3 |]
      [| 2.0; 1.0; -1.0; -3.0; -1.0; 2.0; -2.0; 1.0; 2.0 |]
  in
  let rhs = create float64 [| 3; 1 |] [| 8.0; -11.0; -3.0 |] in
  let x = coeff /@ rhs in
  Printf.printf "System Ax = b:\n";
  Printf.printf "A:\n%s\n" (data_to_string coeff);
  Printf.printf "b: %s\n" (data_to_string (flatten rhs));
  Printf.printf "x: %s\n\n" (data_to_string (flatten x));

  (* --- Inverse: verify A @@ inv(A) ≈ I --- *)
  let m = create float64 [| 2; 2 |] [| 4.0; 7.0; 2.0; 6.0 |] in
  let m_inv = inv m in
  let product = m @@ m_inv in
  Printf.printf "M:\n%s\n" (data_to_string m);
  Printf.printf "M⁻¹:\n%s\n" (data_to_string m_inv);
  Printf.printf "M × M⁻¹ ≈ I:\n%s\n\n" (data_to_string product);

  (* --- Determinant and norm --- *)
  Printf.printf "det(M) = %.1f\n" (item [] (det m));
  Printf.printf "‖M‖_F  = %.4f\n\n" (item [] (norm m));

  (* --- Least squares: fit y = mx + c to noisy data --- *)
  let x_data = create float64 [| 6 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let y_data = create float64 [| 6 |] [| 1.1; 2.9; 5.2; 6.8; 9.1; 10.8 |] in

  (* Build design matrix: [x, 1] for y = m*x + c *)
  let x_col = reshape [| 6; 1 |] x_data in
  let design = hstack [ x_col; ones float64 [| 6; 1 |] ] in
  let y_col = reshape [| 6; 1 |] y_data in

  let coeffs, _residuals, _rank, _sv = lstsq design y_col in
  Printf.printf "Least squares fit  y = m·x + c:\n";
  Printf.printf "  m = %.4f\n" (item [ 0; 0 ] coeffs);
  Printf.printf "  c = %.4f\n\n" (item [ 1; 0 ] coeffs);

  (* --- Eigendecomposition of a symmetric matrix --- *)
  let sym =
    create float64 [| 3; 3 |]
      [| 2.0; -1.0; 0.0; -1.0; 2.0; -1.0; 0.0; -1.0; 2.0 |]
  in
  let eigenvalues, eigenvectors = eigh sym in
  Printf.printf "Symmetric matrix:\n%s\n" (data_to_string sym);
  Printf.printf "Eigenvalues:  %s\n" (data_to_string eigenvalues);
  Printf.printf "Eigenvectors:\n%s\n\n" (data_to_string eigenvectors);

  (* --- SVD: decompose and reconstruct --- *)
  let data = create float64 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let u_mat, s_vec, vt = svd data in
  (* Reconstruct: U @ diag(S) @ Vt *)
  let s_diag = diag s_vec in
  let reconstructed = u_mat.${[ A; R (0, 2) ]} @@ s_diag @@ vt in
  Printf.printf "Original:\n%s\n" (data_to_string data);
  Printf.printf "Singular values: %s\n" (data_to_string s_vec);
  Printf.printf "Reconstructed (U·S·Vt):\n%s\n" (data_to_string reconstructed)
