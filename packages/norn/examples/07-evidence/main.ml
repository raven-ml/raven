(* Evidence: nested sampling and sequential Monte Carlo.

   The evidence Z is the integral of the likelihood over the prior: the
   probability of the data under the model, which compares models. Nested
   sampling and tempered SMC estimate ln Z with a standard error, and their
   particles, weighted, are posterior draws. *)

module D = Norn.Dist

let f64 = Nx.float64
let t = Nx.Ptree.tensor
let s x = Nx.scalar f64 x

(* mu ~ N(0, 1) and five observations y_i ~ N(mu, 1). Positions are [n] values
   of mu. *)
let y = Nx.create f64 [| 5 |] [| 0.8; 1.3; 0.2; 1.9; 1.1 |]
let prior mu = D.factors (D.normal ~loc:(s 0.) ~scale:(s 1.)) mu

let likelihood mu =
  let column = Nx.reshape [| -1; 1 |] mu in
  Nx.sum ~axes:[ 1 ] (D.factors (D.normal ~loc:column ~scale:(s 1.)) y)

(* The exact ln Z: y is normal with covariance I + 1 1^T. *)
let exact =
  let n = 5. in
  let ys = Nx.to_array y in
  let sum = Array.fold_left ( +. ) 0. ys in
  let sq = Array.fold_left (fun a v -> a +. (v *. v)) 0. ys in
  (-.n /. 2. *. log (2. *. Float.pi))
  -. (0.5 *. log (1. +. n))
  -. (0.5 *. (sq -. (sum *. sum /. (1. +. n))))

let () =
  Printf.printf "exact        ln Z = %.3f\n" exact;
  let live =
    D.sample (Nx.Rng.key 0)
      (D.iid [| 500 |] (D.normal ~loc:(s 0.) ~scale:(s 1.)))
  in

  let z =
    Norn.Nested.run t ~budget:200 ~prior ~likelihood (Nx.Rng.key 1) live
  in
  Format.printf "nested       %a@." Norn.Evidence.pp z;

  let z = Norn.Smc.run t ~budget:50 ~prior ~likelihood (Nx.Rng.key 2) live in
  Format.printf "smc, hmc     %a@." Norn.Evidence.pp z;
  let z' =
    Norn.Smc.run t ~move:Slice ~budget:50 ~prior ~likelihood (Nx.Rng.key 3) live
  in
  Format.printf "smc, slice   %a@." Norn.Evidence.pp z';

  (* The weighted sample is the posterior: N(sum y / 6, 1 / 6). *)
  let w = Norn.Evidence.sample z in
  Printf.printf "effective sample size of the particles: %.0f\n"
    (Nx.item [] (Norn.Weighted.ess w));
  let draws = Norn.Weighted.resample t (Nx.Rng.key 4) ~n:2000 w in
  Printf.printf "posterior mean %.3f (exact %.3f), sd %.3f (exact %.3f)\n"
    (Nx.item [] (Nx.mean draws))
    (Nx.item [] (Nx.sum y) /. 6.)
    (Nx.item [] (Nx.std draws))
    (1. /. sqrt 6.)
