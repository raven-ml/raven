(* Hidden Markov models: summing discrete states out.

   A series switches between two regimes, each a normal with its own mean,
   and stays in its regime from one step to the next with some probability.
   The regime at each step is discrete, so a gradient sampler cannot move
   it; [Dist.hmm] sums it out with the forward algorithm, leaving a density
   over the means, the noise and the stay probabilities that NUTS samples.
   [Dist.membership] then gives each step's regime's posterior. *)

module D = Norn.Dist
module M = Norn_model

let f64 = Nx.float64
let s x = Nx.scalar f64 x
let vec xs = Nx.create f64 [| Array.length xs |] xs
let steps = 120

type 'a regimes = { mu : 'a; sigma : 'a; stay : 'a }

module Regimes = struct
  type ('a, _) t = 'a regimes

  let walk c { mu; sigma; stay } =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf mu in
    let sigma = field c "sigma" leaf sigma in
    let stay = field c "stay" leaf stay in
    { mu; sigma; stay }
end

let regimes : (Nx.host Nx.float64_t regimes, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module Regimes)

(* The transition logits: row [i] stays with probability [stay_i]. *)
let transition stay =
  let log_stay = Nx.log stay and log_move = Nx.log1p (Nx.neg stay) in
  let at i x = Nx.get [ i ] x in
  Nx.stack
    [
      Nx.stack [ at 0 log_stay; at 0 log_move ];
      Nx.stack [ at 1 log_move; at 1 log_stay ];
    ]

(* The means are sorted, so the regimes keep their labels. *)
let model =
  M.v f64 regimes Nx.Ptree.tensor @@ fun () ->
  let mu = M.sample (D.sorted 2 (D.normal ~loc:(s 0.) ~scale:(s 3.))) in
  let sigma = M.sample (D.half_normal ~scale:(s 1.)) in
  let stay = M.sample (D.beta ~a:(vec [| 8.; 8. |]) ~b:(vec [| 2.; 2. |])) in
  let y =
    M.sample
      (D.hmm ~initial:(vec [| 0.; 0. |]) ~transition:(transition stay) steps
         (D.normal ~loc:mu ~scale:sigma))
  in
  ({ mu; sigma; stay }, y)

let () =
  Format.printf "%a@." M.pp model;

  (* Simulate a series from known values. *)
  let truth = M.fix (fun p -> p.mu) (vec [| -1.; 1.5 |]) model in
  let truth = M.fix (fun p -> p.sigma) (s 0.5) truth in
  let truth = M.fix (fun p -> p.stay) (vec [| 0.95; 0.9 |]) truth in
  let _, y = M.simulate truth (Nx.Rng.key 0) in

  (* Sample the posterior with the states summed out. *)
  let lp = M.log_density model y in
  let u = M.coords model in
  let start = M.init model y ~chains:4 (Nx.Rng.key 1) in
  let key = Nx.Rng.key 2 in
  let state = Norn.Nuts.init u lp start in
  let state = Norn.Nuts.warmup u lp key ~steps:100 state in
  let _, coords, stats = Norn.Nuts.sample u lp key ~draws:100 state in
  let draws = Norn.Draws.map u regimes (fun c -> M.constrain model c) coords in
  Format.printf "%a@." Norn.Summary.pp (Norn.Summary.v regimes ~stats draws);

  (* Each step's regime's posterior at the posterior means. *)
  let mean t = Nx.mean ~axes:[ 0; 1 ] t in
  let p =
    Nx.Ptree.map regimes
      (fun _ t -> mean t)
      (draws :> Nx.host Nx.float64_t regimes)
  in
  let d =
    D.hmm ~initial:(vec [| 0.; 0. |]) ~transition:(transition p.stay) steps
      (D.normal ~loc:p.mu ~scale:p.sigma)
  in
  let upper = Nx.to_array (Nx.slice [ Nx.A; Nx.I 1 ] (D.membership d y)) in
  let ys = Nx.to_array y in
  print_string "step  y       P(upper regime)\n";
  List.iter
    (fun t -> Printf.printf "%4d  %6.3f  %.3f\n" t ys.(t) upper.(t))
    [ 0; 20; 40; 60; 80; 100; 119 ]
