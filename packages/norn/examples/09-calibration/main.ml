(* Simulation-based calibration.

   Draw a truth from the prior, simulate data from it, and rank the truth among
   posterior draws given that data. With exact posterior draws the ranks are
   uniform; [Diag.rank_uniformity] tests them exactly. A posterior that is too
   wide, too narrow or shifted fails. *)

module D = Norn.Dist
module M = Norn_model

let f64 = Nx.float64
let t = Nx.Ptree.tensor
let s x = Nx.scalar f64 x
let n = 5

(* mu ~ N(0, 1), y_i ~ N(mu, 1). *)
let model =
  M.v f64 t t (fun () ->
      let mu = M.sample (D.normal ~loc:(s 0.) ~scale:(s 1.)) in
      (mu, M.sample (D.iid [| n |] (D.normal ~loc:mu ~scale:(s 1.)))))

(* The posterior is N(sum y / (n + 1), 1 / (n + 1)); [widen] scales its spread
   to make a miscalibrated one. *)
let posterior ~widen key y ~draws =
  let mean = Nx.div_s (Nx.sum y) (float_of_int (n + 1)) in
  let sd = widen /. sqrt (float_of_int (n + 1)) in
  let z = Nx.Rng.normal key f64 [| 1; draws |] in
  Norn.Draws.v t (Nx.add mean (Nx.mul_s z sd))

let draws = 99

let p_value ~widen =
  let rank r =
    let k = Nx.Rng.fold_in (Nx.Rng.key 0) r in
    let truth, y = M.simulate model (Nx.Rng.fold_in k 0) in
    Norn.Diag.rank t ~truth (posterior ~widen (Nx.Rng.fold_in k 1) y ~draws)
  in
  Nx.item [] (Norn.Diag.rank_uniformity t ~draws (List.init 200 rank))

let () =
  List.iter
    (fun widen ->
      Printf.printf "posterior sd x %.1f: p-value of uniform ranks %.4f\n" widen
        (p_value ~widen))
    [ 1.0; 0.5; 2.0 ]
