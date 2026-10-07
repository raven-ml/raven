(* Models as generative functions.

   [norn.model] turns one function that draws every random variable into the
   densities samplers need, with the bijectors applied, and maps draws back to
   values of your types. The same function simulates data and predicts. *)

module D = Norn.Dist
module M = Norn_model

let f64 = Nx.float64
let s x = Nx.scalar f64 x

(* A line with noise: slope, intercept and noise scale. *)
type 'a line = { slope : 'a; intercept : 'a; sigma : 'a }

module Line = struct
  type 'a t = 'a line

  let walk c { slope; intercept; sigma } =
    let open Nx.Ptree.Walk in
    let slope = field c "slope" leaf slope in
    let intercept = field c "intercept" leaf intercept in
    let sigma = field c "sigma" leaf sigma in
    { slope; intercept; sigma }
end

let line : Nx.float64_t line Nx.Ptree.t = Nx.Ptree.instantiate (module Line)
let x = Nx.linspace f64 (-1.) 1. 30

(* Every random variable is a [sample]; the function returns the latent ones and
   the observed ones. *)
let model =
  M.v f64 line Nx.Ptree.tensor @@ fun () ->
  let slope = M.sample (D.normal ~loc:(s 0.) ~scale:(s 5.)) in
  let intercept = M.sample (D.normal ~loc:(s 0.) ~scale:(s 5.)) in
  let sigma = M.sample (D.half_normal ~scale:(s 1.)) in
  let mean = Nx.add intercept (Nx.mul slope x) in
  let y = M.sample (D.normal ~loc:mean ~scale:sigma) in
  ({ slope; intercept; sigma }, y)

let () =
  Format.printf "%a@." M.pp model;

  (* Simulate data from known values with [fix], then forget them. *)
  let truth = M.fix (fun p -> p.slope) (s 2.) model in
  let truth = M.fix (fun p -> p.intercept) (s (-1.)) truth in
  let truth = M.fix (fun p -> p.sigma) (s 0.3) truth in
  let _, y = M.simulate truth (Nx.Rng.key 0) in

  (* The posterior density over unconstrained coordinates: sigma moves as its
     logarithm. *)
  let lp = M.log_density model y in
  let u = M.coords model in
  let start = M.init model y ~chains:4 (Nx.Rng.key 1) in
  let key = Nx.Rng.key 2 in
  let state = Norn.Nuts.init u lp start in
  let state = Norn.Nuts.warmup u lp key ~steps:150 state in
  let _, coords, stats = Norn.Nuts.sample u lp key ~draws:200 state in

  (* Draws are coordinates; [constrain] maps each back to values. *)
  let draws = Norn.Draws.map u line (fun c -> M.constrain model c) coords in
  Format.printf "%a@." Norn.Summary.pp (Norn.Summary.v line ~stats draws);

  (* Posterior predictions at the first draw of the first chain. *)
  let first =
    Nx.Ptree.map line
      (fun _ t -> Nx.get [ 0; 0 ] t)
      (draws :> Nx.float64_t line)
  in
  let y_new = M.predict model (Nx.Rng.key 3) first in
  Printf.printf "predicted y at x = -1, 1: %.3f %.3f\n" (Nx.item [ 0 ] y_new)
    (Nx.item [ 29 ] y_new)
