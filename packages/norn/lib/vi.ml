(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Variational inference over factor covariances.

   Each chain's Gaussian is [N (μ, B Bᵀ + D²)], [B] a [d × r] matrix whose upper
   triangle is zero and [D] diagonal (Ong, Nott and Smith 2018): [x = μ + B z +
   D ε] for standard normal [z] of [r] and [ε] of [d] elements. Both are linear
   in their parameters, of either sign, as Ong et al. take them: a step scales
   no parameter's effect exponentially. [r = 0] is mean field, [r = d] any
   covariance. The bound is ascended by stochastic gradient with ADVI's step
   sizes (Kucukelbir, Tran, Ranganath, Gelman and Blei 2017, eqs. 10 and 11).

   The gradient is Ong et al.'s (eqs. 6, 9 and 10), from [lp]'s gradient [g] at
   each draw: [μ]'s is [E g], [B]'s [E ((g + a) zᵀ)] and [D]'s diagonal [E ((g +
   a) ⊙ ε)], [a = Σ⁻¹ (x - μ)]. For a Gaussian target [g + a] vanishes draw by
   draw at the optimum.

   Everything is on flat rows of [c] chains. *)

module P = Nx.Ptree

let invalid_argf = Rows.invalid_argf
let context = "Norn.Gaussian.vi"

(* The step-size sequence is [ρ_i = η i^(-1/2 + ε) / (τ + sqrt s_i)], where [s_i
   = a g_i² + (1 - a) s_(i-1)] from [s_1 = g_1²] (Kucukelbir et al. 2017, eqs.
   10 and 11). *)
let weight = 0.1
let tau = 1.
let epsilon = 1e-16

(* [η] is searched over these, each run for [search_steps] steps and judged by a
   bound of [search_draws] draws (Kucukelbir et al. 2017, section 2.6). *)
let etas = [| 0.01; 0.1; 1.; 10.; 100. |]
let search_steps = 50
let search_draws = 100

type ('f, 'd) params = {
  mu : (float, 'f, 'd) Nx.t; (* [[c; d]] *)
  diag : (float, 'f, 'd) Nx.t; (* [D]'s diagonal, [[c; d]] *)
  b : (float, 'f, 'd) Nx.t; (* [[c; d; r]], its upper triangle ignored *)
}

let params_ptree () = P.(pair tensor (pair tensor tensor))
let pack p = (p.mu, (p.diag, p.b))
let unpack (mu, (diag, b)) = { mu; diag; b }
let rank p = (Nx.shape p.b).(2)
let map2 f p q = { mu = f p.mu q.mu; diag = f p.diag q.diag; b = f p.b q.b }

(* [noise dt keys ~m ~r ~d] is each chain's draws [z], [[c; m; r]], and [ε],
   [[c; m; d]], chain [i]'s from row [i] of [keys]. *)
let noise dt keys ~m ~r ~d =
  let draw j n =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns tensor)
      (fun k -> Noise.normal (Nx.Rng.fold_in k j) dt [| m; n |])
      keys
  in
  (draw 0 r, draw 1 d)

(* [w p] is [D⁻¹ B], [[c; d; r]]. *)
let w p = Nx.div (Nx.tril p.b) (Nx.unsqueeze ~axes:[ 2 ] p.diag)

(* [inner w] is [I + Wᵀ W], [[c; r; r]]. *)
let inner w =
  let r = (Nx.shape w).(2) in
  Nx.add
    (Nx.eye_at (Nx.beside w) (Nx.dtype w) r)
    (Nx.matmul (Nx.matrix_transpose w) w)

(* [log_det p] is each chain's [log |Σ|], by the determinant lemma: [Σ log D_ii²
   + log |I + Wᵀ W|]. *)
let log_det p =
  let diag = Nx.sum ~axes:[ 1 ] (Nx.log (Nx.square p.diag)) in
  if rank p = 0 then diag
  else
    let l = Nx.cholesky (inner (w p)) in
    Nx.add diag
      (Nx.mul_s
         (Nx.sum ~axes:[ 1 ] (Nx.log (Nx.diagonal ~axis1:1 ~axis2:2 l)))
         2.)

(* [bounds u lp like p (z, eps)] is each chain's estimate, from the draws, of
   the bound [E log p (x) + log |Σ| / 2 + d (1 + log 2π) / 2] and of its
   gradient. A chain's positions take its structure in [like].

   [a = Σ⁻¹ v] for [v = B z + D ε] is [D⁻¹ (ε + W K⁻¹ (z - Wᵀ ε))], by
   Woodbury's identity on [Σ = D (I + W Wᵀ) D], [K = I + Wᵀ W]. It subtracts no
   terms of size [|W|²], but [K]'s condition grows as [(|B| / |D|)²], so a solve
   loses accuracy as [D] shrinks. *)
let bounds u lp like p (z, eps) =
  let c = (Nx.shape eps).(0) and m = (Nx.shape eps).(1) in
  let d = (Nx.shape eps).(2) in
  let scale = Nx.unsqueeze ~axes:[ 1 ] p.diag in
  let b = Nx.tril p.b in
  let v = Nx.mul scale eps in
  let v =
    if rank p = 0 then v else Nx.add v (Nx.matmul z (Nx.matrix_transpose b))
  in
  let x =
    Nx.reshape [| c * m; d |] (Nx.add (Nx.unsqueeze ~axes:[ 1 ] p.mu) v)
  in
  let rows = P.map u (fun _ t -> Nx.repeat ~axis:0 m t) like in
  let _, g, l =
    Rune.value_and_grad_aux P.tensor P.tensor
      (fun x ->
        let l = Rows.at_rows u lp rows x in
        (Nx.sum l, l))
      x
  in
  let l = Nx.reshape [| c; m |] l and g = Nx.reshape [| c; m; d |] g in
  let y =
    if rank p = 0 then eps
    else
      let w = w p in
      let rhs = Nx.sub z (Nx.matmul eps w) in
      let k =
        Nx.matrix_transpose (Nx.solve (inner w) (Nx.matrix_transpose rhs))
      in
      Nx.add eps (Nx.matmul k (Nx.matrix_transpose w))
  in
  let h = Nx.add g (Nx.div y scale) in
  let mean x = Nx.mean ~axes:[ 1 ] x in
  let gb =
    if rank p = 0 then p.b
    else
      Nx.tril (Nx.div_s (Nx.matmul (Nx.matrix_transpose h) z) (float_of_int m))
  in
  let grad = { mu = mean g; diag = mean (Nx.mul h eps); b = gb } in
  let constant = 0.5 *. float_of_int d *. (1. +. Float.log (2. *. Float.pi)) in
  (Nx.add_s (Nx.add (mean l) (Nx.mul_s (log_det p) 0.5)) constant, grad)

(* [ascend u lp like ~draws ~steps eta keys p] is [p] after [steps] steps of
   stochastic gradient ascent, chain [i] at [eta.{i}], its step [j] drawing from
   [Nx.Rng.fold_in] of row [i] of [keys] with [j]. A chain whose bound estimate,
   gradient or next parameters are not finite stays where it is: on a bound of
   [-inf] the gradient is the entropy's alone, which would widen the Gaussian
   without end. *)
let ascend u lp like ~draws ~steps eta keys p =
  let c = (Nx.shape p.mu).(0) and d = (Nx.shape p.mu).(1) in
  let r = rank p in
  let dt = Nx.dtype p.mu and at = Nx.beside p.mu in
  let per x e =
    Nx.reshape (Array.append [| c |] (Array.make (Nx.ndim x - 1) 1)) e
  in
  let finite x = Nx.all ~axes:[ 1 ] (Nx.reshape [| c; -1 |] (Nx.isfinite x)) in
  let step (p, s) i =
    let p = unpack p and s = unpack s in
    let z = noise dt (Nx.Rng.fold_in_tensor keys i) ~m:draws ~r ~d in
    let elbo, g = bounds u lp like p z in
    let first = Nx.equal_s i 0l in
    let n = Nx.add_s (Nx.cast dt i) 1. in
    let decay = Nx.pow n (Nx.scalar_like n (-0.5 +. epsilon)) in
    let s' =
      map2
        (fun s g ->
          Nx.where first (Nx.square g)
            (Nx.add (Nx.mul_s (Nx.square g) weight) (Nx.mul_s s (1. -. weight))))
        s g
    in
    let move x s g =
      let rho = Nx.div (Nx.mul (per s eta) decay) (Nx.add_s (Nx.sqrt s) tau) in
      Nx.add x (Nx.mul rho g)
    in
    let next =
      {
        mu = move p.mu s'.mu g.mu;
        diag = move p.diag s'.diag g.diag;
        b = move p.b s'.b g.b;
      }
    in
    let ok =
      List.fold_left Nx.logical_and (Nx.isfinite elbo)
        (List.map finite [ g.mu; g.diag; g.b; next.mu; next.diag; next.b ])
    in
    let hold a b = Nx.where (per a ok) a b in
    ((pack (map2 hold next p), pack (map2 hold s' s)), ())
  in
  let zero = map2 (fun x _ -> Nx.zeros_like x) p p in
  let carry = P.pair (params_ptree ()) (params_ptree ()) in
  let (p, _), () =
    Rune.scan carry P.tensor P.unit step
      ~init:(pack p, pack zero)
      (Nx.arange_at at Nx.int32 0 steps 1)
  in
  unpack p

(* [estimate u lp like keys p] is each chain's bound from [search_draws] draws,
   [-inf] where it is not finite. *)
let estimate u lp like keys p =
  let d = (Nx.shape p.mu).(1) in
  let z = noise (Nx.dtype p.mu) keys ~m:search_draws ~r:(rank p) ~d in
  let elbo, _ = bounds u lp like p z in
  Nx.where (Nx.isfinite elbo) elbo (Nx.scalar_like elbo Float.neg_infinity)

let vi u ?(rank = 0) ?(draws = 1) ~steps lp k start =
  if steps < 1 then invalid_argf "%s: steps = %d is not positive" context steps;
  if draws < 1 then invalid_argf "%s: draws = %d is not positive" context draws;
  if rank < 0 then invalid_argf "%s: rank = %d is negative" context rank;
  let c = Rows.count context u start in
  let l0, g0 = Rows.evaluate u lp start in
  Rows.check_density context u lp start l0;
  let d = Rows.elements u c start in
  if d = 0 then invalid_argf "%s: the position has no float element" context;
  Nx.check P.tensor
    (Nx.logical_and (Nx.isfinite l0) (Rows.finite u g0))
    (Nx.arange_at (Nx.beside l0) Nx.int32 0 c 1)
    (fun _ i ->
      Invalid_argument
        (Printf.sprintf
           "%s: the density or its gradient is not finite at chain %ld of the \
            start"
           context (Nx.item [] i)));
  let at = Nx.beside l0 and dt = Nx.dtype l0 in
  let r = min rank d in
  let keys = Nx.Rng.split_batch ~n:c (Nx.Rng.place at k) in
  (* The start has unit covariance: [B]'s columns are the first [r] of [I / sqrt
     2], and [D² = I - B Bᵀ]. *)
  let first = Nx.less_s (Nx.arange_at at Nx.int32 0 d 1) (Int32.of_int r) in
  let diag =
    Nx.where first
      (Nx.full_at at dt [| d |] (Float.sqrt 0.5))
      (Nx.ones_at at dt [| d |])
  in
  let p =
    {
      mu = Rows.ravel u l0 start;
      diag = Nx.copy (Nx.broadcast_to [| c; d |] diag);
      b =
        Nx.copy
          (Nx.broadcast_to [| c; d; r |]
             (Nx.mul_s (Nx.eye_at ~m:r at dt d) (Float.sqrt 0.5)));
    }
  in
  (* Candidate [e] of the search runs on chains [e c] to [e c + c - 1], chain
     [i] of it from [Nx.Rng.fold_in] of its key with [e]. *)
  let e = Array.length etas in
  let tile x =
    Nx.tile (Array.append [| e |] (Array.make (Nx.ndim x - 1) 1)) x
  in
  let candidates j =
    let k = Nx.Rng.fold_in keys j in
    Nx.Rng.of_tensor
      (Nx.concatenate ~axis:0
         (List.init e (fun i -> Nx.Rng.to_tensor (Nx.Rng.fold_in k i))))
  in
  let all = Nx.place at (Nx.create dt [| e |] etas) in
  let like = P.map u (fun _ t -> tile t) start in
  let tried =
    ascend u lp like ~draws ~steps:(min steps search_steps)
      (Nx.repeat ~axis:0 c all) (candidates 0)
      { mu = tile p.mu; diag = tile p.diag; b = tile p.b }
  in
  let score = estimate u lp like (candidates 1) tried in
  let eta = Nx.take (Nx.argmax ~axis:0 (Nx.reshape [| e; c |] score)) all in
  let p = ascend u lp start ~draws ~steps eta (Nx.Rng.fold_in keys 2) p in
  let elbo = estimate u lp start (Nx.Rng.fold_in keys 3) p in
  (* [Σ = |D| (I + W Wᵀ) |D|] for [W = B / |D|], row by row. *)
  let scale = Nx.abs p.diag in
  let directions, variances =
    if r = 0 then (Nx.zeros_at at dt [| c; 0; d |], Nx.zeros_at at dt [| c; 0 |])
    else
      let w = Nx.div (Nx.tril p.b) (Nx.unsqueeze ~axes:[ 2 ] scale) in
      Gaussian.factor dt w (Nx.eye_at at dt r)
  in
  let gaussian =
    Gaussian.of_rows u start ~mean:p.mu ~scale ~directions ~variances
  in
  Gaussian.{ gaussian; elbo }
