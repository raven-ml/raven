open Nx

(* Linear Algebra Operations with Nx *)

let print_separator () = Printf.printf "\n%s\n\n" (String.make 50 '-')

let () =
  (* Create sample matrices for linear algebra operations *)
  Printf.printf "Nx Linear Algebra Examples\n";
  print_separator ();

  (* 1. Matrix multiplication *)
  Printf.printf "1. Matrix Multiplication\n";

  let a =
    init float64 [| 2; 3 |] (fun idx ->
        float_of_int ((idx.(0) * 3) + idx.(1) + 1))
  in

  let b =
    init float64 [| 3; 2 |] (fun idx ->
        float_of_int ((idx.(0) * 2) + idx.(1) + 1))
  in

  Printf.printf "Matrix A (2x3):\n%s\n" (to_string a);
  Printf.printf "Matrix B (3x2):\n%s\n" (to_string b);

  let matrix_product = matmul a b in
  Printf.printf "Matrix product A x B (2x2):\n%s\n" (to_string matrix_product);

  (* Dot product of vectors *)
  let v1 = create float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let v2 = create float64 [| 3 |] [| 4.0; 5.0; 6.0 |] in

  Printf.printf "Vector v1:\n%s\n" (to_string v1);
  Printf.printf "Vector v2:\n%s\n" (to_string v2);

  let dot_product = dot v1 v2 in
  Printf.printf "Dot product v1 · v2:\n%s\n" (to_string dot_product);

  print_separator ();

  (* 2. Matrix inverse *)
  Printf.printf "2. Matrix Inverse\n";

  let square_matrix = create float64 [| 2; 2 |] [| 4.0; 7.0; 2.0; 6.0 |] in
  Printf.printf "Original matrix:\n%s\n" (to_string square_matrix);

  let inverse = inv square_matrix in
  Printf.printf "Inverse matrix:\n%s\n" (to_string inverse);

  (* Verify: A x A^(-1) = I *)
  let product = matmul square_matrix inverse in
  Printf.printf "Original x Inverse (should be identity):\n%s\n"
    (to_string product);

  print_separator ();

  (* 3. Solving linear systems *)
  Printf.printf "3. Solving Linear Systems (Ax = b)\n";

  let coef_matrix =
    create float64 [| 3; 3 |] [| 2.0; 1.0; 1.0; 1.0; 3.0; 2.0; 1.0; 0.0; 0.0 |]
  in

  let constants = create float64 [| 3; 1 |] [| 4.0; 5.0; 6.0 |] in

  Printf.printf "Coefficient matrix A:\n%s\n" (to_string coef_matrix);
  Printf.printf "Constants b:\n%s\n" (to_string constants);

  let solution = solve coef_matrix constants in
  Printf.printf "Solution x:\n%s\n" (to_string solution);

  (* Verify: A x x = b *)
  let verification = matmul coef_matrix solution in
  Printf.printf "Verification (A x x, should equal b):\n%s\n"
    (to_string verification);

  print_separator ();

  (* 4. Eigenvalues and eigenvectors *)
  Printf.printf "4. Eigenvalues and Eigenvectors\n";

  let symmetric_matrix = create float64 [| 2; 2 |] [| 4.0; 1.0; 1.0; 3.0 |] in
  Printf.printf "Symmetric matrix:\n%s\n" (to_string symmetric_matrix);

  let eigenvalues, eigenvectors = eigh symmetric_matrix in
  Printf.printf "Eigenvalues:\n%s\n" (to_string eigenvalues);
  Printf.printf "Eigenvectors (columns):\n%s\n" (to_string eigenvectors);

  (* Non-symmetric case *)
  let non_symmetric = create float64 [| 2; 2 |] [| 4.0; 1.0; 3.0; 2.0 |] in
  Printf.printf "Non-symmetric matrix:\n%s\n" (to_string non_symmetric);

  let eigen_vals, eigen_vecs = eig non_symmetric in
  Printf.printf "Eigenvalues:\n%s\n" (to_string eigen_vals);
  Printf.printf "Eigenvectors (columns):\n%s\n" (to_string eigen_vecs);

  print_separator ();

  (* 5. Singular Value Decomposition (SVD) *)
  Printf.printf "5. Singular Value Decomposition (SVD)\n";

  let matrix_for_svd =
    create float64 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  Printf.printf "Matrix for SVD:\n%s\n" (to_string matrix_for_svd);

  let u, s, vt = svd matrix_for_svd in
  Printf.printf "U (left singular vectors):\n%s\n" (to_string u);
  Printf.printf "S (singular values):\n%s\n" (to_string s);
  Printf.printf "V^T (right singular vectors transposed):\n%s\n" (to_string vt);

  print_separator ();

  (* 6. Linear Regression Example *)
  Printf.printf "6. Linear Regression Example\n";

  (* Generate synthetic data: y = 2x + 1 + noise *)
  let x_data = arange_f float64 0.0 10.0 1.0 in

  (* Create design matrix [ones, x] *)
  let n = dim 0 x_data in
  let ones = ones float64 [| n; 1 |] in
  let x_col = reshape [| n; 1 |] x_data in
  let x = concatenate ~axis:1 [ ones; x_col ] in

  (* intercept and slope *)
  let y_true = add_scalar (mul_scalar x_data 2.0) 1.0 in

  (* Add some random noise - using a fixed seed for reproducibility *)
  let noise = randn float64 ~seed:42 [| n |] in
  let noise_scaled = mul_scalar noise 0.5 in
  let y = add y_true noise_scaled in

  Printf.printf "X (first few rows):\n%s\n"
    (to_string (slice [| 0; 0 |] [| 5; 2 |] x));
  Printf.printf "y (first few values):\n%s\n"
    (to_string (slice [| 0 |] [| 5 |] y));

  (* Solve for parameters using normal equation: β = (X^T X)^(-1) X^T y *)
  let xtx = matmul (transpose x) x in
  let xty = matmul (transpose x) (reshape [| n; 1 |] y) in
  let beta = solve xtx xty in

  Printf.printf "Estimated parameters (intercept, slope):\n%s\n"
    (to_string beta);
  Printf.printf "True parameters: [1.0; 2.0]\n";

  (* Make predictions *)
  let y_pred = matmul x beta in

  Printf.printf "Predicted y (first few values):\n%s\n"
    (to_string (slice [| 0 |] [| 5 |] (flatten y_pred)));

  (* Calculate R² score *)
  let y_mean = mean y in
  let ss_total =
    sum (square (sub y (full float64 [| n |] (get_item [||] y_mean))))
  in
  let y_pred_flat = flatten y_pred in
  let ss_residual = sum (square (sub y y_pred_flat)) in
  let r_squared = sub (scalar float64 1.0) (div ss_residual ss_total) in

  Printf.printf "R² score: %s\n" (to_string r_squared)
