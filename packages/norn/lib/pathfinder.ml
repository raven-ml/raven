(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pathfinder (Zhang, Carpenter, Gelman and Vehtari 2022).

   Each chain follows L-BFGS's iterates [θ_0, ..., θ_L] toward a mode of the
   density [p]. At each iterate a Gaussian approximates [p]: its covariance
   [Σ_l] is L-BFGS's inverse Hessian from the last [J] pairs of steps [s] and
   changes [z] of [-log p]'s gradient with positive curvature, over a diagonal
   [α_l] updated at each pair, and its mean is [θ_l + Σ_l g_l], [g_l] the
   gradient of [log p] at [θ_l]. The chain keeps the Gaussian of the largest
   ELBO, each estimated from [K] draws.

   Everything below is on flat rows over a batch of axes [[c; n]], [c] chains
   and [n] iterates. *)

module P = Nx.Ptree

let invalid_argf = Rows.invalid_argf
let context = "Norn.Gaussian.pathfinder"

(* A pair enters the covariance when its curvature [sᵀz] is above this fraction
   of [|z|²]: its gradient changes at most [10¹²] times more than its position
   along [z] (Zhang et al. 2022, section 2.4). *)
let curvature_floor = 1e-12

(* The chosen Gaussian's bound is estimated anew from this many draws: the
   estimates that chose it are the largest of many and biased high. *)
let bound_draws = 100

(* [dot a b] is the inner product along the last axis. *)
let dot a b = Nx.sum ~axes:[ -1 ] (Nx.mul a b)

(* [last x] is [x] with a trailing axis of one, to broadcast along it. *)
let last x = Nx.unsqueeze ~axes:[ -1 ] x

(* Paths *)

(* [iterates u lp ~history ~steps c start] is the first [steps] iterates of
   L-BFGS of memory [history] on [-lp] from each chain of [start], a position of
   [c * steps] chains, chain by chain. *)
let iterates u lp ~history ~steps c start =
  let m = Jera.Minimize.lbfgs ~memory:history ~linear:Jera.Linear.dense in
  let objective y = Nx.neg (Nx.reshape [||] (lp y)) in
  let path x =
    let x = P.map u (fun _ t -> Nx.unsqueeze ~axes:[ 0 ] t) x in
    Jera.Minimize.iterates u m ~steps objective x
  in
  let xs = Rune.vmap P.(u @-> returns u) path start in
  P.map u
    (fun _ t ->
      let s = Nx.shape t in
      Nx.reshape
        (Array.append [| c * steps |] (Array.sub s 3 (Array.length s - 3)))
        t)
    xs

(* Diagonals and histories *)

(* [recover ~j xs gs] is, at each iterate, the diagonal [α], the last [j] pairs
   that entered the covariance, oldest first, as rows [[c; n; j; d]] of steps
   and of gradient changes, and which of the [j] slots hold a pair, from the
   iterates [xs] and the gradients [gs] of [log p], [[c; n; d]]. An empty slot
   holds zeros. [α] starts at one and, at each pair entering, takes Gilbert and
   Lemaréchal's (1989) diagonal update (Zhang et al. 2022, algorithm 3); an
   element the update makes other than positive and finite keeps its value. *)
let recover ~j xs gs =
  let c = (Nx.shape xs).(0) and n = (Nx.shape xs).(1) in
  let d = (Nx.shape xs).(2) in
  let at = Nx.beside xs and dt = Nx.dtype xs in
  let step (alpha, (ss, (zs, filled))) (s, z) =
    let sz = dot s z in
    let enters = Nx.greater sz (Nx.mul_s (dot z z) curvature_floor) in
    let a = last (dot (Nx.mul z alpha) z) and b = last sz in
    let cc = last (dot (Nx.div s alpha) s) in
    let inv =
      Nx.sub
        (Nx.add (Nx.div a (Nx.mul b alpha)) (Nx.div (Nx.square z) b))
        (Nx.div
           (Nx.mul a (Nx.square s))
           (Nx.mul (Nx.mul b cc) (Nx.square alpha)))
    in
    let next = Nx.recip inv in
    let kept =
      Nx.logical_and (last enters)
        (Nx.logical_and (Nx.greater_s next 0.) (Nx.isfinite next))
    in
    let alpha = Nx.where kept next alpha in
    let shift buffer x =
      let older = Nx.slice [ Nx.A; Nx.R (1, j) ] buffer in
      Nx.concatenate ~axis:1 [ older; Nx.unsqueeze ~axes:[ 1 ] x ]
    in
    let into buffer x =
      let e =
        Nx.reshape
          (Array.append [| c |] (Array.make (Nx.ndim buffer - 1) 1))
          enters
      in
      Nx.where e (shift buffer x) buffer
    in
    let carry =
      ( alpha,
        (into ss s, (into zs z, into filled (Nx.ones_at at Nx.bool [| c |]))) )
    in
    (carry, carry)
  in
  let init =
    ( Nx.ones_at at dt [| c; d |],
      ( Nx.zeros_at at dt [| c; j; d |],
        (Nx.zeros_at at dt [| c; j; d |], Nx.zeros_at at Nx.bool [| c; j |]) )
    )
  in
  let by_iterate x = Nx.swapaxes 0 1 x in
  let steps =
    Nx.sub
      (Nx.slice [ Nx.A; Nx.R (1, n) ] xs)
      (Nx.slice [ Nx.A; Nx.R (0, n - 1) ] xs)
  in
  let changes =
    Nx.sub
      (Nx.slice [ Nx.A; Nx.R (0, n - 1) ] gs)
      (Nx.slice [ Nx.A; Nx.R (1, n) ] gs)
  in
  let carry = P.(pair tensor (pair tensor (pair tensor tensor))) in
  let _, (alphas, (ss, (zs, filled))) =
    Rune.scan carry
      P.(pair tensor tensor)
      carry step ~init
      (by_iterate steps, by_iterate changes)
  in
  let with_first first rest =
    Nx.concatenate ~axis:1 [ Nx.unsqueeze ~axes:[ 1 ] first; by_iterate rest ]
  in
  let alpha0, (ss0, (zs0, filled0)) = init in
  ( with_first alpha0 alphas,
    with_first ss0 ss,
    with_first zs0 zs,
    with_first filled0 filled )

(* Approximations *)

type ('f, 'd) approximations = {
  mean : (float, 'f, 'd) Nx.t; (* [[c; n; d]] *)
  scale : (float, 'f, 'd) Nx.t; (* [sqrt α], [[c; n; d]] *)
  directions : (float, 'f, 'd) Nx.t; (* [[c; n; k; d]] *)
  variances : (float, 'f, 'd) Nx.t; (* [[c; n; k]] *)
  valid : 'd Nx.bool_t; (* [[c; n]] *)
}

(* [approximations ~alpha ~ss ~zs ~filled xs gs] is each iterate's Gaussian in
   the form of {!Gaussian}: scale [S = sqrt α], orthonormal directions and
   variances along them.

   With [β = [diag α Z, S]] and the compact form's [γ] (Byrd, Nocedal and
   Schnabel 1994, eq. 2.6), [Σ = diag α + β γ βᵀ = S (I + W γ Wᵀ) S] for [W =
   S⁻¹ β] (Zhang et al. 2022, eq. 3), whose directions and variances
   {!Gaussian.factor} finds from a QR of [W]. An empty slot of the history has
   zero columns in [β] and a unit diagonal in [E], whose inverse [γ] needs: it
   adds nothing. An iterate whose variances are not all positive and finite has
   no Gaussian. *)
let approximations ~alpha ~ss ~zs ~filled xs gs =
  let at = Nx.beside xs and dt = Nx.dtype xs in
  let j = (Nx.shape ss).(2) in
  let mT = Nx.matrix_transpose in
  let sz = Nx.matmul ss (mT zs) in
  let empty = Nx.cast dt (Nx.logical_not filled) in
  let eye = Nx.eye_at at dt j in
  let e = Nx.add (Nx.triu sz) (Nx.mul eye (Nx.unsqueeze ~axes:[ -2 ] empty)) in
  let eta = Nx.diagonal ~axis1:2 ~axis2:3 sz in
  let e_inv =
    Nx.solve_triangular ~triangle:Upper e (Nx.broadcast_to (Nx.shape e) eye)
  in
  let za = Nx.mul zs (Nx.unsqueeze ~axes:[ 2 ] alpha) in
  let inner =
    Nx.add (Nx.mul eye (Nx.unsqueeze ~axes:[ -2 ] eta)) (Nx.matmul za (mT zs))
  in
  let lower = Nx.matmul (mT e_inv) (Nx.matmul inner e_inv) in
  let zero = Nx.zeros_like e_inv in
  let gamma =
    Nx.concatenate ~axis:(-2)
      [
        Nx.concatenate ~axis:(-1) [ zero; Nx.neg e_inv ];
        Nx.concatenate ~axis:(-1) [ Nx.neg (mT e_inv); lower ];
      ]
  in
  let scale = Nx.sqrt alpha in
  let w =
    mT
      (Nx.concatenate ~axis:2
         [
           Nx.mul zs (Nx.unsqueeze ~axes:[ 2 ] scale);
           Nx.div ss (Nx.unsqueeze ~axes:[ 2 ] scale);
         ])
  in
  let directions, variances = Gaussian.factor dt w gamma in
  let valid =
    Nx.all ~axes:[ -1 ]
      (Nx.logical_and (Nx.greater_s variances 0.) (Nx.isfinite variances))
  in
  (* [Σ g = S (I + U (v - 1) Uᵀ) S g], the rows of [U] the directions. *)
  let t = Nx.mul scale gs in
  let along =
    Nx.mul (Nx.sub_s variances 1.) (dot directions (Nx.unsqueeze ~axes:[ 2 ] t))
  in
  let t = Nx.add t (Nx.sum ~axes:[ 2 ] (Nx.mul (last along) directions)) in
  let mean = Nx.add xs (Nx.mul scale t) in
  let valid = Nx.logical_and valid (Nx.all ~axes:[ -1 ] (Nx.isfinite mean)) in
  { mean; scale; directions; variances; valid }

(* ELBO *)

(* [elbo u lp start keys ~draws a] is each iterate's ELBO estimate from [draws]
   standard normal draws [z], [[c; n; draws; d]], chain [i]'s from row [i] of
   [keys]: the mean of [log p (φ) - log q (φ)] at [φ = μ + S A z], where [log q
   (φ) = -(log |Σ| + zᵀz + d log 2π) / 2] (Zhang et al. 2022, eq. 6). An iterate
   without a Gaussian is drawn from a unit one and has an ELBO of [-inf]. A
   chain's draws take its structure in [start]. *)
let elbo u lp start keys ~draws a =
  let c = (Nx.shape a.mean).(0) and n = (Nx.shape a.mean).(1) in
  let d = (Nx.shape a.mean).(2) in
  let dt = Nx.dtype a.mean in
  let valid = last a.valid in
  let mean = Nx.where valid a.mean (Nx.zeros_like a.mean) in
  let variances = Nx.where valid a.variances (Nx.ones_like a.variances) in
  let directions =
    Nx.where (last valid) a.directions (Nx.zeros_like a.directions)
  in
  let z =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns tensor)
      (fun k -> Noise.normal k dt [| n; draws; d |])
      keys
  in
  let w = Nx.unsqueeze ~axes:[ 2 ] in
  let along =
    Nx.mul
      (w (Nx.sub_s (Nx.sqrt variances) 1.))
      (Nx.matmul z (Nx.matrix_transpose directions))
  in
  let x =
    Nx.add (w mean) (Nx.mul (w a.scale) (Nx.add z (Nx.matmul along directions)))
  in
  let log_det =
    Nx.add
      (Nx.sum ~axes:[ -1 ] (Nx.log a.scale))
      (Nx.mul_s (Nx.sum ~axes:[ -1 ] (Nx.log variances)) 0.5)
  in
  let log_q =
    Nx.sub
      (Nx.mul_s (Nx.sum ~axes:[ -1 ] (Nx.square z)) (-0.5))
      (last
         (Nx.add_s log_det
            (0.5 *. float_of_int d *. Float.log (2. *. Float.pi))))
  in
  let rows = c * n * draws in
  let template = P.map u (fun _ t -> Nx.repeat ~axis:0 (n * draws) t) start in
  let lp =
    Nx.reshape [| c; n; draws |]
      (Rows.at_rows u lp template (Nx.reshape [| rows; d |] x))
  in
  let e = Nx.mean ~axes:[ -1 ] (Nx.sub lp log_q) in
  let ninf = Nx.scalar_like e Float.neg_infinity in
  Nx.where (Nx.logical_and a.valid (Nx.logical_not (Nx.isnan e))) e ninf

(* Run *)

let pathfinder u ?(history = 6) ?(elbo_draws = 5) ~steps lp k start =
  if history < 1 then
    invalid_argf "%s: history = %d is not positive" context history;
  if elbo_draws < 1 then
    invalid_argf "%s: elbo_draws = %d is not positive" context elbo_draws;
  if steps < 1 then invalid_argf "%s: steps = %d is not positive" context steps;
  let c = Rows.count context u start in
  let l0, g0 = Rows.evaluate u lp start in
  Rows.check_density context u lp start l0;
  let d = Rows.elements u c start in
  if d = 0 then invalid_argf "%s: the position has no float element" context;
  let finite = Nx.logical_and (Nx.isfinite l0) (Rows.finite u g0) in
  Nx.check P.tensor finite
    (Nx.arange_at (Nx.beside l0) Nx.int32 0 c 1)
    (fun _ i ->
      Invalid_argument
        (Printf.sprintf
           "%s: the density or its gradient is not finite at chain %ld of the \
            start"
           context (Nx.item [] i)));
  let at = Nx.beside l0 in
  let k = Nx.Rng.place at k in
  let xs = iterates u lp ~history ~steps c start in
  let lx, gx = Rows.evaluate u lp xs in
  let flat x = Nx.reshape [| c; steps; d |] (Rows.ravel u lx x) in
  let xs_flat = flat xs and gs = flat gx in
  let alpha, ss, zs, filled = recover ~j:history xs_flat gs in
  let a = approximations ~alpha ~ss ~zs ~filled xs_flat gs in
  let keys = Nx.Rng.split_batch ~n:c k in
  let e = elbo u lp start (Nx.Rng.fold_in keys 0) ~draws:elbo_draws a in
  (* The choice is among [θ_1, ..., θ_L] (Zhang et al. 2022, algorithm 1):
     [θ_0]'s Gaussian, of unit covariance, is a chain's only when no other takes
     part, [argmax] taking the first of equal estimates. An iterate equal to the
     one before is a stopped path repeating its estimate: its Gaussian is the
     one before's, and would only add draws of the ELBO's noise to the
     choice. *)
  let moved =
    Nx.concatenate ~axis:1
      [
        Nx.zeros_at at Nx.bool [| c; 1 |];
        Nx.any ~axes:[ -1 ]
          (Nx.not_equal
             (Nx.slice [ Nx.A; Nx.R (1, steps) ] xs_flat)
             (Nx.slice [ Nx.A; Nx.R (0, steps - 1) ] xs_flat));
      ]
  in
  let e = Nx.where moved e (Nx.scalar_like e Float.neg_infinity) in
  let best = Nx.argmax ~axis:1 ~keepdims:true e in
  let pick x =
    let extra = Nx.ndim x - 2 in
    let i = Nx.reshape (Array.append [| c; 1 |] (Array.make extra 1)) best in
    Nx.squeeze ~axes:[ 1 ] (Nx.take_along_axis ~axis:1 i x)
  in
  (* A chain none of whose iterates takes part keeps [θ_0]'s Gaussian of unit
     covariance, centred at [θ_0] itself, and an ELBO of [-inf]. *)
  let failed = Nx.equal_s (pick e) Float.neg_infinity in
  let x0 = Nx.reshape [| c; d |] (Nx.slice [ Nx.A; Nx.R (0, 1) ] xs_flat) in
  let chosen =
    {
      mean = Nx.where (Nx.unsqueeze ~axes:[ 1 ] failed) x0 (pick a.mean);
      scale = pick a.scale;
      directions = pick a.directions;
      variances = pick a.variances;
      valid = Nx.logical_not failed;
    }
  in
  let one x = Nx.unsqueeze ~axes:[ 1 ] x in
  let fresh =
    elbo u lp start (Nx.Rng.fold_in keys 1) ~draws:bound_draws
      {
        mean = one chosen.mean;
        scale = one chosen.scale;
        directions = one chosen.directions;
        variances = one chosen.variances;
        valid = one chosen.valid;
      }
  in
  let gaussian =
    Gaussian.of_rows u start ~mean:chosen.mean ~scale:chosen.scale
      ~directions:chosen.directions ~variances:chosen.variances
  in
  Gaussian.{ gaussian; elbo = Nx.squeeze ~axes:[ 1 ] fresh }
