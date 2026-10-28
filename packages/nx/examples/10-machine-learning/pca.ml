(** Principal component analysis via SVD.

    Generate synthetic data with known structure, project to lower dimensions,
    and verify the explained variance captures the signal. *)

let () =
  (* 200 points in 5D: most variance along axes 0 and 1 *)
  let n = 200 in
  let scale = Nx.create Nx.float64 [| 1; 5 |] [| 10.0; 5.0; 1.0; 1.0; 1.0 |] in
  let data = Nx.(Rng.normal float64 [| n; 5 |] * scale) in
  Printf.printf "Data shape: [%d; %d]\n\n" (Nx.dim 0 data) (Nx.dim 1 data);

  (* Center *)
  let mu = Nx.mean ~axes:[ 0 ] ~keepdims:true data in
  let centered = Nx.(data - mu) in

  (* Reduced SVD: centered = U diag(S) Vt, with S in float64 *)
  let _, s, vt = Nx.svd Nx.float64 centered in

  (* Explained variance ratio: s_i^2 / sum(s^2) *)
  let s2 = Nx.square s in
  let ratios = Nx.(s2 / sum s2) in
  Printf.printf "Singular values:          %s\n" (Nx.to_string s);
  Printf.printf "Explained variance ratio: %s\n" (Nx.to_string ratios);
  Printf.printf "Cumulative:               %s\n\n"
    (Nx.to_string (Nx.cumsum ratios));

  (* Project to 2 components *)
  let n_components = 2 in
  let components = Nx.slice [ Nx.R (0, n_components); Nx.A ] vt in
  let projected = Nx.matmul centered (Nx.matrix_transpose components) in
  Printf.printf "Projected shape: [%d; %d]\n" (Nx.dim 0 projected)
    (Nx.dim 1 projected);

  (* Reconstruct and measure error *)
  let reconstructed = Nx.(matmul projected components + mu) in
  let rmse = Nx.(sqrt (mean (square (data - reconstructed)))) in
  Printf.printf "Reconstruction RMSE (2 of 5 components): %.4f\n"
    (Nx.item [] rmse)
