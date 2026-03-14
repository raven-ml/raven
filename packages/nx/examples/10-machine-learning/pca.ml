(** Principal component analysis via SVD.

    Generate synthetic data with known structure, project to lower dimensions,
    and verify the explained variance captures the signal. *)

open Nx
open Nx.Infix

let () =
  (* 200 points in 5D: most variance along axes 0 and 1 *)
  let n = 200 in
  let scale = create Float64 [| 1; 5 |] [| 10.0; 5.0; 1.0; 1.0; 1.0 |] in
  let data = randn Float64 [| n; 5 |] * scale in
  Printf.printf "Data shape: [%d; %d]\n\n" (shape data).(0) (shape data).(1);

  (* Center *)
  let mu = mean ~axes:[ 0 ] ~keepdims:true data in
  let centered = data - mu in

  (* Economy SVD: centered = U diag(S) Vt *)
  let _u, s, vt = svd ~full_matrices:false centered in

  (* Explained variance ratio: s_i^2 / sum(s^2) *)
  let s2 = square s in
  let ratios = s2 /$ item [] (sum s2) in
  Printf.printf "Singular values:          %s\n" (data_to_string s);
  Printf.printf "Explained variance ratio: %s\n" (data_to_string ratios);
  Printf.printf "Cumulative:               %s\n\n"
    (data_to_string (cumsum ratios));

  (* Project to 2 components *)
  let n_components = 2 in
  let components = slice [ R (0, n_components); A ] vt in
  let projected = matmul centered (matrix_transpose components) in
  Printf.printf "Projected shape: [%d; %d]\n"
    (shape projected).(0)
    (shape projected).(1);

  (* Reconstruct and measure error *)
  let reconstructed = matmul projected components + mu in
  let rmse = sqrt (mean (square (data - reconstructed))) in
  Printf.printf "Reconstruction RMSE (2 of 5 components): %.4f\n" (item [] rmse)
