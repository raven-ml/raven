(* Hamiltonian Monte Carlo and ensemble sampling.

   [Hmc] shares one step size, trajectory length and geometry across chains and
   tunes them all in warmup. [Ensemble] needs no gradient and tunes nothing: its
   stretch move is unchanged by an affine map, so a correlated target costs what
   an isotropic one does. *)

let f64 = Nx.float64
let t = Nx.Ptree.tensor

(* A Gaussian in two dimensions with correlation 0.95 and scales 1 and 3, over
   positions of shape [chains; 2]. *)
let rho = 0.95
let scales = Nx.create f64 [| 2 |] [| 1.; 3. |]

let log_density x =
  let z = Nx.div x scales in
  let a = Nx.slice [ Nx.A; Nx.I 0 ] z and b = Nx.slice [ Nx.A; Nx.I 1 ] z in
  let q =
    Nx.sub
      (Nx.add (Nx.square a) (Nx.square b))
      (Nx.mul_s (Nx.mul a b) (2. *. rho))
  in
  Nx.mul_s q (-0.5 /. (1. -. (rho *. rho)))

let report name (draws : Nx.float64_t Norn.Draws.t) =
  let d = (draws :> Nx.float64_t) in
  let flat = Nx.reshape [| -1; 2 |] d in
  let sd = Nx.std ~axes:[ 0 ] flat in
  let centred = Nx.sub flat (Nx.mean ~axes:[ 0 ] ~keepdims:true flat) in
  let cov = Nx.mean (Nx.prod ~axes:[ 1 ] centred) in
  let ess = Norn.Diag.ess_bulk t draws in
  Printf.printf
    "%-9s sd %.2f %.2f (exact 1, 3), correlation %.3f, ESS %.0f %.0f\n" name
    (Nx.item [ 0 ] sd) (Nx.item [ 1 ] sd)
    (Nx.item [] cov /. (Nx.item [ 0 ] sd *. Nx.item [ 1 ] sd))
    (Nx.item [ 0 ] ess) (Nx.item [ 1 ] ess)

let () =
  let key = Nx.Rng.key 0 in

  (* HMC over 8 chains. *)
  let start = Nx.Rng.normal (Nx.Rng.key 1) f64 [| 8; 2 |] in
  let s = Norn.Hmc.init t log_density start in
  let s = Norn.Hmc.warmup t log_density key ~steps:200 s in
  Printf.printf "hmc: step size %.3f, trajectory length %.3f\n"
    (Nx.item [] s.step_size) (Nx.item [] s.length);
  let _, draws, stats = Norn.Hmc.sample t log_density key ~draws:200 s in
  Printf.printf "hmc: mean acceptance %.2f\n"
    (Nx.item [] (Nx.mean (stats :> Nx.float64_elt Norn.Stats.t).acceptance));
  report "hmc" draws;

  (* An ensemble of 16 walkers, each walker a chain of the draws. *)
  let start = Nx.Rng.normal (Nx.Rng.key 2) f64 [| 16; 2 |] in
  let s = Norn.Ensemble.init t log_density start in
  let s = Norn.Ensemble.warmup t log_density key ~steps:300 s in
  let _, draws, stats = Norn.Ensemble.sample t log_density key ~draws:500 s in
  Printf.printf "ensemble: mean acceptance %.2f\n"
    (Nx.item []
       (Nx.mean (stats :> Nx.float64_elt Norn.Ensemble.stats).acceptance));
  report "ensemble" draws
