(* Gaussian approximations.

   A [Gaussian] is diagonal plus low rank over your structure, with no matrix
   over all its elements. A Laplace approximation is the Gaussian whose
   precision is the curvature at the mode; samplers use such a Gaussian as their
   geometry. *)

module D = Norn.Dist

let f64 = Nx.float64
let t = Nx.Ptree.tensor

(* Three counts, each Poisson with its own log rate u_i ~ N(0, 3). *)
let counts = Nx.create f64 [| 3 |] [| 2.; 15.; 120. |]

(* The log posterior of one position [3], up to a constant. *)
let log_posterior u =
  let prior = Nx.div_s (Nx.square u) (-18.) in
  Nx.sum (Nx.add prior (Nx.sub (Nx.mul counts u) (Nx.exp u)))

let () =
  (* The mode, by BFGS on the negative log posterior. *)
  let mode =
    Jera.Minimize.solve t
      (Jera.Minimize.bfgs ~linear:Jera.Linear.dense)
      ~tol:(Jera.Tol.v ~rel:1e-10 ~abs:1e-12)
      ~budget:100
      (fun u -> Nx.neg (log_posterior u))
      (Nx.zeros f64 [| 3 |])
    |> Jera.Solution.get
  in
  (* The curvature there: the posterior is separable, so the precision is
     diagonal, exp u + 1/9. *)
  let precision = Nx.add_s (Nx.exp mode) (1. /. 9.) in
  let laplace = Norn.Gaussian.of_precision t f64 ~mean:mode precision in
  Format.printf "%a@." (Norn.Gaussian.pp t) laplace;
  Format.printf "mode       %a@." Nx.pp mode;
  Format.printf "sd         %a@." Nx.pp
    (Nx.sqrt (Norn.Gaussian.variance t laplace));

  (* Draws and densities of the approximation. *)
  let x = Norn.Gaussian.sample t (Nx.Rng.key 0) ~n:4000 laplace in
  Format.printf "draws' sd  %a@." Nx.pp (Nx.std ~axes:[ 0 ] x);
  Format.printf "log density at the mode %a@." Nx.pp
    (Norn.Gaussian.log_density t laplace (Nx.reshape [| 1; 3 |] mode));

  (* The approximation as NUTS's geometry: sampling starts already
     preconditioned, so a short warmup suffices. *)
  let chains = 4 in
  (* A density over chains maps the one-position density over axis 0. *)
  let lp = Rune.vmap' log_posterior in
  let start = Norn.Gaussian.sample t (Nx.Rng.key 1) ~n:chains laplace in
  let s = Norn.Nuts.init t ~geometry:laplace lp start in
  let s = Norn.Nuts.warmup t lp (Nx.Rng.key 2) ~steps:50 s in
  let _, draws, stats = Norn.Nuts.sample t lp (Nx.Rng.key 2) ~draws:200 s in
  (* The first rate's posterior is skewed, which the Gaussian misses. *)
  Format.printf "NUTS sd    %a@." Nx.pp
    (Nx.std ~axes:[ 0; 1 ] (draws :> Nx.float64_t));
  Format.printf "mean leapfrog steps %.1f@."
    (Nx.item []
       (Nx.mean (Nx.cast f64 (stats :> Nx.float64_elt Norn.Stats.t).n_steps)))
