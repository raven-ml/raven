(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

let invalid_argf = Rows.invalid_argf

type move = Hmc | Slice

type ('u, 'f) state = {
  particles : 'u;
  prior : (float, 'f) Nx.t;
  likelihood : (float, 'f) Nx.t;
  beta : (float, 'f) Nx.t;
  log_evidence : (float, 'f) Nx.t;
  variance : (float, 'f) Nx.t;
  steps : Nx.int32_t;
  step_size : (float, 'f) Nx.t;
  length : (float, 'f) Nx.t;
  length_moment : (float, 'f) Nx.t;
  resampled : int;
  move : move;
}

type ('u, 'f) smc = ('u, 'f) state

let ptree (type u f) (u : u P.t) : (u, f) state P.t =
  let module S = struct
    type _ t = (u, f) smc

    let walk c s =
      let open P.Walk in
      let particles = field c "particles" (structure u) s.particles in
      let prior = field c "prior" tensor s.prior in
      let likelihood = field c "likelihood" tensor s.likelihood in
      let beta = field c "beta" tensor s.beta in
      let log_evidence = field c "log_evidence" tensor s.log_evidence in
      let variance = field c "variance" tensor s.variance in
      let steps = field c "steps" tensor s.steps in
      let step_size = field c "step_size" tensor s.step_size in
      let length = field c "length" tensor s.length in
      let length_moment = field c "length_moment" tensor s.length_moment in
      let resampled = field c "resampled" int s.resampled in
      let move =
        match
          field c "move" int (match s.move with Hmc -> 0 | Slice -> 1)
        with
        | 0 -> Hmc
        | _ -> Slice
      in
      {
        particles;
        prior;
        likelihood;
        beta;
        log_evidence;
        variance;
        steps;
        step_size;
        length;
        length_moment;
        resampled;
        move;
      }
  end in
  P.nest (module S) P.unit

(* Temperatures *)

(* [increments l db] is [db l], the log incremental weights of a temperature
   step [db], [-inf] where [l] is: [0 · -inf] is not a weight. *)
let increments l db =
  Nx.where (Nx.equal l (Nx.full_like l Float.neg_infinity)) l (Nx.mul db l)

let log_ess lw =
  Nx.sub (Nx.mul_s (Nx.logsumexp lw) 2.) (Nx.logsumexp (Nx.mul_s lw 2.))

(* The tolerance on the next temperature. *)
let beta_tol = Jera.Tol.rel 1e-3

(* [next s] is the next temperature: [1] if the incremental weights to it keep
   an ESS of half the particles, else the temperature in (beta, 1] where the ESS
   is half the particles. *)
let next (type f) (s : (_, f) state) : (float, f) Nx.t =
  let n = float_of_int (Nx.shape s.likelihood).(0) in
  let half = Nx.scalar_like s.beta (Float.log (n /. 2.)) in
  let f b = Nx.sub (log_ess (increments s.likelihood (Nx.sub b s.beta))) half in
  let one = Nx.ones_like s.beta in
  let root = Jera.Root.bracket ~tol:beta_tol f ~lo:s.beta ~hi:one in
  Nx.where
    (Nx.greater_equal (f one) (Nx.zeros_like one))
    one (Jera.Solution.best root)

(* Particles

   The [N] particles are [M] chains of [P = N / M] states, particle [i] in chain
   [i mod M]. A step resamples [M] ancestors and runs [P - 1] moves from each,
   keeping every state. *)

(* [chain_variance m v] is the variance of the mean of [v], the [N] values at
   the particles, from its [M] chains of [P] states read as one stationary chain
   (Dau and Chopin 2022, sec. 4.3): the autocovariances [γ_q] pooled over the
   chains, summed by Geyer's initial monotone sequence, over [N]. *)
let chain_variance m v =
  let n = (Nx.shape v).(0) in
  let p = n / m in
  (* Particle [i] is state [i / M] of chain [i mod M]: [x] is [[P; M]]. *)
  let x = Nx.reshape [| p; m |] (Nx.sub v (Nx.mean v)) in
  (* [S = x xᵀ] holds [Σ_m x_pm x_p'm]; [γ_q] sums its [q]-th diagonal. Read row
     by row into rows of [P + 1], the upper triangle's [q]-th diagonal falls in
     column [q]. *)
  let upper = Nx.triu (Nx.matmul x (Nx.transpose x)) in
  let skewed =
    Nx.reshape
      [| p; p + 1 |]
      (Nx.concatenate ~axis:0
         [ Nx.reshape [| p * p |] upper; Nx.zeros (Nx.dtype v) [| p |] ])
  in
  let gamma =
    Nx.div_s
      (Nx.slice [ Nx.R (0, p) ] (Nx.sum ~axes:[ 0 ] skewed))
      (float_of_int n)
  in
  let k = p / 2 in
  let pairs =
    Nx.sum ~axes:[ 1 ]
      (Nx.reshape [| k; 2 |] (Nx.slice [ Nx.R (0, 2 * k) ] gamma))
  in
  let positive =
    Nx.cumprod (Nx.cast (Nx.dtype v) (Nx.greater pairs (Nx.zeros_like pairs)))
  in
  let tau =
    Nx.sub
      (Nx.mul_s (Nx.sum (Nx.mul positive (Nx.cummin pairs))) 2.)
      (Nx.slice [ Nx.I 0 ] gamma)
  in
  Nx.div_s (Nx.maximum tau (Nx.zeros_like tau)) (float_of_int n)

let resample u k ~n lw x =
  Weighted.resample u k ~n Weighted.{ values = x; log_weights = lw }

(* Moves

   [P - 1] moves from each of the [M] ancestors [x0] on the tempered density,
   every state kept: each move's positions, prior and likelihood, on a leading
   axis of [P - 1]; Hamiltonian moves also give the step size for the next
   temperature. *)

(* [slice_moves] moves by hit-and-run slice steps; chain [i]'s move [j] draws
   from [fold_in] of row [i] of [split_batch ~n:M keys] with [j]. *)
let slice_moves context u ~prior ~likelihood keys g ~m moves tempered x0 pl0 ll0
    =
  let eval _ y =
    let pl = prior y and ll = likelihood y in
    Rows.check_range context pl;
    Rows.check_range context ll;
    (tempered pl ll, (pl, ll))
  in
  let keys = Nx.Rng.split_batch ~n:m keys in
  let move (x, (lp, aux)) j =
    let ks =
      Rune.vmap
        P.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
        (fun k -> Nx.Rng.fold_in_tensor k j)
        keys
    in
    let x, lp, aux, _ =
      Slice.hit_and_run context u P.(pair tensor tensor) eval ks g x lp aux
    in
    ((x, (lp, aux)), (x, aux))
  in
  let _, (xs, (pls, lls)) =
    Rune.scan
      P.(pair u (pair tensor (pair tensor tensor)))
      P.tensor
      P.(pair u (pair tensor tensor))
      ~f:move
      ~init:(x0, (tempered pl0 ll0, (pl0, ll0)))
      moves
  in
  (xs, pls, lls)

(* The acceptance the step size aims for, and the gain of its one Robbins-Monro
   step per temperature on [log eps]. *)
let target_acceptance = 0.8
let gain = 1.

(* [hamiltonian_moves] moves by Hamiltonian transitions of one step size, one
   length and the reweighted particles' Gaussian; move [j] is a transition of
   key [fold_in keys j]. Both are tuned between temperatures, which are close,
   so the last temperature's values are a good start: the step size by one
   Robbins-Monro step on [log eps] toward a mean acceptance of
   [target_acceptance], the length by an Adam step of the ChEES criterion per
   move, as Hmc's warmup tunes it, the [M] chains estimating each gradient. The
   search runs across the run, [searched] steps so far and its second moment
   [moment]: restarted at each temperature it lags behind the length a narrowing
   posterior needs. Within a temperature both stay fixed, so every move leaves
   its density invariant. *)
let hamiltonian_moves context u ~prior ~likelihood keys g moves tempered x0
    ~step_size ~length ~moment ~searched =
  let density x = tempered (prior x) (likelihood x) in
  let lp0, grad0 = Rows.evaluate u density x0 in
  Rows.check_range context lp0;
  let log_length = Nx.log length in
  let chees =
    {
      Hamiltonian.log_length;
      second = moment;
      bar = log_length;
      count = searched;
    }
  in
  let move ((x, (lp, grad)), chees) j =
    let t =
      Hamiltonian.transition context u density
        (Nx.Rng.fold_in_tensor keys j)
        g ~step_size ~length x lp grad
    in
    let chees =
      Hamiltonian.chees_step chees step_size (Hamiltonian.chees_gradient t)
    in
    (((t.position, (t.lp, t.grad)), chees), (t.position, t.alpha))
  in
  let (_, chees), (xs, alphas) =
    Rune.scan
      P.(pair (pair u (pair tensor u)) (Hamiltonian.chees_ptree ()))
      P.tensor
      P.(pair u tensor)
      ~f:move
      ~init:((x0, (lp0, grad0)), chees)
      moves
  in
  (* The states' prior and likelihood, [M] rows per call. *)
  let at f = Rune.vmap P.(u @-> returns tensor) f xs in
  let pls = at prior and lls = at likelihood in
  Rows.check_range context (Nx.reshape [| -1 |] lls);
  let log_eps =
    Nx.add (Nx.log step_size)
      (Nx.mul_s (Nx.sub_s (Nx.mean alphas) target_acceptance) gain)
  in
  (xs, pls, lls, Nx.exp log_eps, (Nx.exp chees.log_length, chees.second))

let step (type f) u ~prior ~likelihood k (s : (_, f) state) =
  let context = "Norn.Smc.step" in
  let dt = Nx.dtype s.beta in
  let n = (Nx.shape s.likelihood).(0) and m = s.resampled in
  let p = n / m in
  let beta = next s in
  let lw = increments s.likelihood (Nx.sub beta s.beta) in
  let log_mean = Nx.logmeanexp lw in
  let relative = Nx.exp (Nx.sub lw log_mean) in
  let log_evidence = Nx.add s.log_evidence log_mean in
  let variance = Nx.add s.variance (chain_variance m relative) in
  (* Ancestors and the geometry of the reweighted particles. *)
  let values = P.(pair u (pair tensor tensor)) in
  let x0, (pl0, ll0) =
    resample values (Nx.Rng.fold_in k 0) ~n:m lw
      (s.particles, (s.prior, s.likelihood))
  in
  let g = Adapt.population u dt lw s.particles in
  let tempered pl ll = Nx.add pl (increments ll beta) in
  let move_keys = Nx.Rng.fold_in k 1 in
  let moves = Nx.arange Nx.int32 0 (p - 1) 1 in
  let xs, pls, lls, step_size, (length, length_moment) =
    match s.move with
    | Slice ->
        let xs, pls, lls =
          slice_moves context u ~prior ~likelihood move_keys g ~m moves tempered
            x0 pl0 ll0
        in
        (xs, pls, lls, s.step_size, (s.length, s.length_moment))
    | Hmc ->
        hamiltonian_moves context u ~prior ~likelihood move_keys g moves
          tempered x0 ~step_size:s.step_size ~length:s.length
          ~moment:s.length_moment
          ~searched:(Nx.mul_s (Nx.cast dt s.steps) (float_of_int (p - 1)))
  in
  (* The ancestors, then each move's states: [P] rows of [M] chains. *)
  let stack first rest =
    Nx.concatenate ~axis:0
      [
        first;
        Nx.reshape
          (Array.append
             [| (p - 1) * m |]
             (Array.sub (Nx.shape rest) 2 (Nx.ndim rest - 2)))
          rest;
      ]
  in
  {
    s with
    particles = P.map2 u (fun _ a r -> stack a r) x0 xs;
    prior = stack pl0 pls;
    likelihood = stack ll0 lls;
    beta;
    log_evidence;
    variance;
    steps = Nx.add s.steps (Nx.scalar Nx.int32 1l);
    step_size;
    length;
    length_moment;
  }

(* Starting *)

(* [chains n] is the largest divisor of [n] not above [sqrt n]: chains long
   beside their number, as the evidence's error needs. *)
let chains n =
  let rec down m = if n mod m = 0 then m else down (m - 1) in
  down (max 1 (int_of_float (Float.sqrt (float_of_int n))))

let init u ?(move = Hmc) ?resampled ~prior ~likelihood start =
  let context = "Norn.Smc.init" in
  let n = Rows.count context u start in
  let resampled = Option.value resampled ~default:(chains n) in
  if resampled < 1 || n mod resampled <> 0 || n / resampled < 2 then
    invalid_argf
      "%s: resampled = %d does not divide %d particles into chains of two or \
       more"
      context resampled n;
  let pl = prior start and ll = likelihood start in
  Rows.check_density context u prior start pl;
  Rows.check_density context u likelihood start ll;
  let dt = Nx.dtype pl in
  let zero = Nx.zeros dt [||] in
  {
    particles = start;
    prior = pl;
    likelihood = ll;
    beta = zero;
    log_evidence = zero;
    variance = zero;
    steps = Nx.scalar Nx.int32 0l;
    step_size = Nx.scalar dt 1.;
    length = Nx.scalar dt 1.;
    length_moment = zero;
    resampled;
    move;
  }

let finished (s : (_, _) state) = Nx.greater_equal s.beta (Nx.ones_like s.beta)

let evidence (type f) (s : (_, f) state) =
  let dt = Nx.dtype s.beta in
  let n = (Nx.shape s.likelihood).(0) in
  let log_weights = Nx.full dt [| n |] (-.Float.log (float_of_int n)) in
  Evidence.v
    ~sample:{ Weighted.values = s.particles; log_weights }
    ~log_evidence:s.log_evidence ~error:(Nx.sqrt s.variance)
    ~information:(Nx.sub (Nx.mean s.likelihood) s.log_evidence)
    ~stop:
      (Nx.where (finished s)
         (Nx.scalar Nx.int32 Evidence.converged)
         (Nx.scalar Nx.int32 Evidence.temperature))
    ~reached:s.beta
    ~replicates:(Nx.reshape [| 1; n |] log_weights)
    ~groups:
      (Nx.mod_ (Nx.arange Nx.int32 0 n 1)
         (Nx.scalar Nx.int32 (Int32.of_int s.resampled)))
    ~group_count:s.resampled

let run u ?move ?resampled ~budget ~prior ~likelihood k start =
  if budget < 1 then
    invalid_argf "Norn.Smc.run: budget = %d is not positive" budget;
  let s = init u ?move ?resampled ~prior ~likelihood start in
  let s =
    Rune.iterate (ptree u) ~max:budget
      ~until:(fun s ->
        Nx.logical_or (finished s)
          (Nx.greater_equal s.steps (Nx.scalar Nx.int32 (Int32.of_int budget))))
      ~f:(fun s ->
        step u ~prior ~likelihood (Nx.Rng.fold_in_tensor k s.steps) s)
      s
  in
  evidence s
