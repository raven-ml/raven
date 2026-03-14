(** t-SNE dimensionality reduction.

    Embed three 10-dimensional clusters into 2D using the exact t-SNE algorithm.
    Reports KL divergence and per-cluster spread. *)

open Nx

(* Pairwise squared distances: [n, d] -> [n, n] *)
let pairwise_sq data =
  let diff = sub (expand_dims [ 1 ] data) (expand_dims [ 0 ] data) in
  sum ~axes:[ 2 ] (square diff)

(* Off-diagonal mask: 1 everywhere except the diagonal. *)
let off_diag n =
  sub (full Float64 [| n; n |] 1.0) (cast Float64 (eye Float64 n))

(* Compute symmetric P matrix via binary search for each row's bandwidth. *)
let compute_p dist_sq ~perplexity =
  let n = (shape dist_sq).(0) in
  let target = Float.log perplexity in
  let p = zeros Float64 [| n; n |] in
  for i = 0 to n - 1 do
    let di = get [ i ] dist_sq in
    let lo = ref 1e-10 in
    let hi = ref 1e4 in
    let row = ref (zeros Float64 [| n |]) in
    for _ = 0 to 50 do
      let sigma = (!lo +. !hi) /. 2.0 in
      let beta = 1.0 /. (2.0 *. sigma *. sigma) in
      let pi = exp (mul_s di (-.beta)) in
      set_item [ i ] 0.0 pi;
      let s = item [] (sum pi) in
      let pi = div_s pi (Float.max s 1e-30) in
      let h = -.item [] (sum (mul pi (log (clamp ~min:1e-30 pi)))) in
      row := pi;
      if h > target then hi := sigma else lo := sigma
    done;
    set [ i ] p !row
  done;
  (* Symmetrise: P = (P + P^T) / (2n) *)
  let p = div_s (add p (matrix_transpose p)) (2.0 *. Float.of_int n) in
  clamp ~min:1e-12 p

let () =
  let n_per = 50 in
  let dim = 10 in
  let perplexity = 20.0 in
  let max_iter = 500 in
  let lr = 100.0 in

  (* Three well-separated clusters in 10D *)
  let c0 = randn Float64 [| n_per; dim |] in
  let c1 = add_s (randn Float64 [| n_per; dim |]) 8.0 in
  let c2 = sub_s (randn Float64 [| n_per; dim |]) 8.0 in
  let data = concatenate ~axis:0 [ c0; c1; c2 ] in
  let n = (shape data).(0) in
  Printf.printf "Data: %d points in %dD, perplexity=%.0f\n\n" n dim perplexity;

  let dist_sq = pairwise_sq data in
  let p = compute_p dist_sq ~perplexity in

  let y = ref (mul_s (randn Float64 [| n; 2 |]) 1e-4) in
  let vel = ref (zeros Float64 [| n; 2 |]) in
  let mask = off_diag n in

  for iter = 1 to max_iter do
    let y_diff = sub (expand_dims [ 1 ] !y) (expand_dims [ 0 ] !y) in
    let y_dsq = sum ~axes:[ 2 ] (square y_diff) in
    let inv_d = mul (div (scalar Float64 1.0) (add_s y_dsq 1.0)) mask in
    let q_sum = Float.max (item [] (sum inv_d)) 1e-30 in
    let q = clamp ~min:1e-12 (div_s inv_d q_sum) in

    let p_eff = if iter <= 100 then mul_s p 4.0 else p in

    (* Gradient: 4 sum_j (p_ij - q_ij)(y_i - y_j)(1+||y_i-y_j||^2)^{-1} *)
    let mult = mul (sub p_eff q) inv_d in
    let grad =
      mul_s (sum ~axes:[ 1 ] (mul (expand_dims [ 2 ] mult) y_diff)) 4.0
    in

    let momentum = if iter <= 100 then 0.5 else 0.8 in
    vel := sub (mul_s !vel momentum) (mul_s grad lr);
    y := add !y !vel;

    if iter = 1 || iter mod 100 = 0 then begin
      let kl = item [] (sum (mul p (log (div p q)))) in
      Printf.printf "  iter %4d  KL = %.4f\n" iter kl
    end
  done;

  Printf.printf "\nPer-cluster spread (mean std of embedded coordinates):\n";
  for c = 0 to 2 do
    let lo = c * n_per in
    let cluster = slice [ R (lo, lo + n_per); A ] !y in
    let sx = item [] (mean (std ~axes:[ 0 ] cluster)) in
    Printf.printf "  Cluster %d: %.4f\n" c sx
  done
