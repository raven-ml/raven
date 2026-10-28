(** t-SNE dimensionality reduction.

    Embed three 10-dimensional clusters into 2D using the exact t-SNE algorithm.
    Reports KL divergence and per-cluster spread. *)

(* Pairwise squared distances: [n, d] -> [n, n] *)
let pairwise_sq data =
  let diff =
    Nx.sub (Nx.unsqueeze ~axes:[ 1 ] data) (Nx.unsqueeze ~axes:[ 0 ] data)
  in
  Nx.sum ~axes:[ 2 ] (Nx.square diff)

(* Off-diagonal mask: 1 everywhere except the diagonal. *)
let off_diag n =
  Nx.sub (Nx.full Nx.float64 [| n; n |] 1.0) (Nx.eye Nx.float64 n)

(* Compute symmetric P matrix via binary search for each row's bandwidth. *)
let compute_p dist_sq ~perplexity =
  let n = Nx.dim 0 dist_sq in
  let target = Float.log perplexity in
  let rows = ref [] in
  for i = 0 to n - 1 do
    let di = Nx.get [ i ] dist_sq in
    let lo = ref 1e-10 in
    let hi = ref 1e4 in
    let row = ref (Nx.zeros Nx.float64 [| n |]) in
    for _ = 0 to 50 do
      let sigma = (!lo +. !hi) /. 2.0 in
      let beta = 1.0 /. (2.0 *. sigma *. sigma) in
      let pi = Nx.exp (Nx.mul_s di (-.beta)) in
      let pi = Nx.set [ Nx.I i ] (Nx.scalar Nx.float64 0.0) pi in
      let s = Nx.item [] (Nx.sum pi) in
      let pi = Nx.div_s pi (Float.max s 1e-30) in
      let h =
        -.Nx.item [] (Nx.sum (Nx.mul pi (Nx.log (Nx.clamp ~min:1e-30 pi))))
      in
      row := pi;
      if h > target then hi := sigma else lo := sigma
    done;
    rows := !row :: !rows
  done;
  let p = Nx.stack ~axis:0 (List.rev !rows) in
  (* Symmetrise: P = (P + P^T) / (2n) *)
  let p = Nx.div_s (Nx.add p (Nx.matrix_transpose p)) (2.0 *. Float.of_int n) in
  Nx.clamp ~min:1e-12 p

let () =
  let n_per = 50 in
  let dim = 10 in
  let perplexity = 20.0 in
  let max_iter = 500 in
  let lr = 100.0 in

  (* Three well-separated clusters in 10D *)
  let c0 = Nx.Rng.normal Nx.float64 [| n_per; dim |] in
  let c1 = Nx.add_s (Nx.Rng.normal Nx.float64 [| n_per; dim |]) 8.0 in
  let c2 = Nx.sub_s (Nx.Rng.normal Nx.float64 [| n_per; dim |]) 8.0 in
  let data = Nx.concatenate ~axis:0 [ c0; c1; c2 ] in
  let n = Nx.dim 0 data in
  Printf.printf "Data: %d points in %dD, perplexity=%.0f\n\n" n dim perplexity;

  let dist_sq = pairwise_sq data in
  let p = compute_p dist_sq ~perplexity in

  let y = ref (Nx.mul_s (Nx.Rng.normal Nx.float64 [| n; 2 |]) 1e-4) in
  let vel = ref (Nx.zeros Nx.float64 [| n; 2 |]) in
  let mask = off_diag n in

  for iter = 1 to max_iter do
    let y_diff =
      Nx.sub (Nx.unsqueeze ~axes:[ 1 ] !y) (Nx.unsqueeze ~axes:[ 0 ] !y)
    in
    let y_dsq = Nx.sum ~axes:[ 2 ] (Nx.square y_diff) in
    let inv_d = Nx.mul (Nx.recip (Nx.add_s y_dsq 1.0)) mask in
    let q_sum = Float.max (Nx.item [] (Nx.sum inv_d)) 1e-30 in
    let q = Nx.clamp ~min:1e-12 (Nx.div_s inv_d q_sum) in

    let p_eff = if iter <= 100 then Nx.mul_s p 4.0 else p in

    (* Gradient: 4 sum_j (p_ij - q_ij)(y_i - y_j)(1+||y_i-y_j||^2)^{-1} *)
    let mult = Nx.mul (Nx.sub p_eff q) inv_d in
    let grad =
      Nx.mul_s
        (Nx.sum ~axes:[ 1 ] (Nx.mul (Nx.unsqueeze ~axes:[ 2 ] mult) y_diff))
        4.0
    in

    let momentum = if iter <= 100 then 0.5 else 0.8 in
    vel := Nx.sub (Nx.mul_s !vel momentum) (Nx.mul_s grad lr);
    y := Nx.add !y !vel;

    if iter = 1 || iter mod 100 = 0 then begin
      let kl = Nx.item [] (Nx.sum (Nx.mul p (Nx.log (Nx.div p q)))) in
      Printf.printf "  iter %4d  KL = %.4f\n" iter kl
    end
  done;

  Printf.printf "\nPer-cluster spread (mean std of embedded coordinates):\n";
  for c = 0 to 2 do
    let lo = c * n_per in
    let cluster = Nx.slice [ Nx.R (lo, lo + n_per); Nx.A ] !y in
    let sx = Nx.item [] (Nx.mean (Nx.std ~axes:[ 0 ] cluster)) in
    Printf.printf "  Cluster %d: %.4f\n" c sx
  done
