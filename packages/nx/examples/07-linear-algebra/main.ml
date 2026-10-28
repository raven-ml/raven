(** Solve systems, decompose matrices, and fit models: linear algebra made
    practical.

    Fit a line to noisy data with least squares, verify matrix inverses,
    decompose matrices with SVD and eigendecomposition. A function whose
    result is real, such as a norm or singular values, takes that result's
    dtype first. *)

let () =
  (* --- Matrix multiplication --- *)
  let a =
    Nx.create Nx.float64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let b =
    Nx.create Nx.float64 [| 3; 2 |] [| 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |]
  in
  Printf.printf "A (2×3):\n%s\n" (Nx.to_string a);
  Printf.printf "B (3×2):\n%s\n" (Nx.to_string b);
  Printf.printf "A *@ B:\n%s\n\n" (Nx.to_string Nx.Infix.(a *@ b));

  (* --- Dot product --- *)
  let u = Nx.create Nx.float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let v = Nx.create Nx.float64 [| 3 |] [| 4.0; 5.0; 6.0 |] in
  Printf.printf "u · v = %s\n\n" (Nx.to_string (Nx.dot u v));

  (* --- Solving linear systems: A x = b --- *)
  let coeff =
    Nx.create Nx.float64 [| 3; 3 |]
      [| 2.0; 1.0; -1.0; -3.0; -1.0; 2.0; -2.0; 1.0; 2.0 |]
  in
  let rhs = Nx.create Nx.float64 [| 3; 1 |] [| 8.0; -11.0; -3.0 |] in
  let x = Nx.solve coeff rhs in
  Printf.printf "System Ax = b:\n";
  Printf.printf "A:\n%s\n" (Nx.to_string coeff);
  Printf.printf "b: %s\n" (Nx.to_string (Nx.flatten rhs));
  Printf.printf "x: %s\n\n" (Nx.to_string (Nx.flatten x));

  (* --- Inverse: verify A *@ inv(A) ≈ I --- *)
  let m = Nx.create Nx.float64 [| 2; 2 |] [| 4.0; 7.0; 2.0; 6.0 |] in
  let m_inv = Nx.inv m in
  let product = Nx.Infix.(m *@ m_inv) in
  Printf.printf "M:\n%s\n" (Nx.to_string m);
  Printf.printf "M⁻¹:\n%s\n" (Nx.to_string m_inv);
  Printf.printf "M × M⁻¹ ≈ I:\n%s\n\n" (Nx.to_string product);

  (* --- Determinant and norm --- *)
  Printf.printf "det(M) = %.1f\n" (Nx.item [] (Nx.det m));
  Printf.printf "‖M‖_F  = %.4f\n\n"
    (Nx.item [] (Nx.matrix_norm Nx.float64 m));

  (* --- Least squares: fit y = mx + c to noisy data --- *)
  let x_data =
    Nx.create Nx.float64 [| 6 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
  in
  let y_data =
    Nx.create Nx.float64 [| 6 |] [| 1.1; 2.9; 5.2; 6.8; 9.1; 10.8 |]
  in

  (* Build design matrix: [x, 1] for y = m*x + c *)
  let x_col = Nx.reshape [| 6; 1 |] x_data in
  let design =
    Nx.concatenate ~axis:1 [ x_col; Nx.ones Nx.float64 [| 6; 1 |] ]
  in
  let y_col = Nx.reshape [| 6; 1 |] y_data in

  let coeffs, _, _, _ = Nx.lstsq Nx.float64 design y_col in
  Printf.printf "Least squares fit  y = m·x + c:\n";
  Printf.printf "  m = %.4f\n" (Nx.item [ 0; 0 ] coeffs);
  Printf.printf "  c = %.4f\n\n" (Nx.item [ 1; 0 ] coeffs);

  (* --- Eigendecomposition of a symmetric matrix --- *)
  let sym =
    Nx.create Nx.float64 [| 3; 3 |]
      [| 2.0; -1.0; 0.0; -1.0; 2.0; -1.0; 0.0; -1.0; 2.0 |]
  in
  let eigenvalues, eigenvectors = Nx.eigh Nx.float64 sym in
  Printf.printf "Symmetric matrix:\n%s\n" (Nx.to_string sym);
  Printf.printf "Eigenvalues:  %s\n" (Nx.to_string eigenvalues);
  Printf.printf "Eigenvectors:\n%s\n\n" (Nx.to_string eigenvectors);

  (* --- SVD: decompose and reconstruct --- *)
  let data =
    Nx.create Nx.float64 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  (* The reduced factors: U is 3×2, S holds 2 values and Vt is 2×2. *)
  let u_mat, s_vec, vt = Nx.svd Nx.float64 data in
  (* Reconstruct: U @ diag(S) @ Vt *)
  let reconstructed = Nx.(Infix.(u_mat *@ diag s_vec *@ vt)) in
  Printf.printf "Original:\n%s\n" (Nx.to_string data);
  Printf.printf "Singular values: %s\n" (Nx.to_string s_vec);
  Printf.printf "Reconstructed (U·S·Vt):\n%s\n" (Nx.to_string reconstructed)
