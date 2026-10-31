(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

type ('u, 'f, 'd) point = { z : 'u; p : 'u; g : 'u; lp : (float, 'f, 'd) Nx.t }

let point_ptree (type u f) (u : (u, 'd) P.t) : ((u, f, 'd) point, 'd) P.t =
  let walk c (s : (u, f, 'd) point) : (u, f, 'd) point =
    let open P.Walk in
    let z = field c "z" (structure u) s.z in
    let p = field c "p" (structure u) s.p in
    let g = field c "g" (structure u) s.g in
    let lp = field c "lp" tensor s.lp in
    { z; p; g; lp }
  in
  P.v walk

let choose_point u mask a b =
  {
    z = Rows.choose u mask a.z b.z;
    p = Rows.choose u mask a.p b.p;
    g = Rows.choose u mask a.g b.g;
    lp = Nx.where mask a.lp b.lp;
  }

let momentum u keys like =
  Rune.vmap
    P.(Nx.Rng.ptree @-> u @-> returns u)
    (fun k x ->
      let k = Nx.Rng.fold_in k 0 in
      let j = ref (-1) in
      P.map u
        (fun _ t ->
          incr j;
          Noise.normal_like t (Nx.Rng.fold_in k !j) (Nx.shape t))
        x)
    keys like

let kinetic u like p = Nx.mul_s (Rows.dot u like p p) 0.5
let max_energy_error = 1000.

let leapfrog context u lp_z running h s =
  let finite = ref running in
  let kick h s = { s with p = Rows.axpy u h s.g s.p } in
  let drift h p =
    let z = Rows.axpy u h p.p p.z in
    let ok = Rows.finite u z in
    finite := ok;
    let z = Rows.choose u (Nx.logical_and running ok) z s.z in
    let lp, g = Rows.evaluate u lp_z z in
    Rows.check_range context lp;
    { z; p = p.p; g; lp }
  in
  let s' = Jera.Split.step Jera.Split.leapfrog ~kick ~drift h s in
  (s', !finite)

(* The acceptance a step-size search aims for, Stan's. *)
let search_acceptance = 0.8

let search (type f) context u lp_z ~reduce keys start (eps : (float, f, 'd) Nx.t) =
  let on = Nx.beside eps in
  let c = (Nx.shape start.lp).(0) in
  let all = Nx.ones_at on Nx.bool [| c |] in
  let log_ratio eps i =
    let keys =
      Rune.vmap
        P.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
        (fun k -> Nx.Rng.fold_in_tensor (Nx.Rng.fold_in k 2) i)
        keys
    in
    let p = momentum u keys start.z in
    let h = Nx.broadcast_to [| c |] eps in
    let leaf, _ = leapfrog context u lp_z all h { start with p } in
    let d =
      Nx.sub
        (Nx.sub (kinetic u h p) start.lp)
        (Nx.sub (kinetic u h leaf.p) leaf.lp)
    in
    reduce (Nx.where (Nx.isnan d) (Nx.scalar_like d Float.neg_infinity) d)
  in
  let threshold = Nx.scalar_like eps (Float.log search_acceptance) in
  let zero = Nx.scalar_at on Nx.int32 0l in
  let up = Nx.greater (log_ratio eps zero) threshold in
  let running = Nx.ones_like up in
  let carry = P.(pair tensor (pair tensor tensor)) in
  let eps, _ =
    Rune.iterate carry ~max:200
      ~until:(fun (_, (_, running)) -> Nx.logical_not (Nx.any running))
      (fun (eps, (i, running)) ->
        let i = Nx.add_s i 1l in
        let d = log_ratio eps i in
        let crossed =
          Nx.where up
            (Nx.logical_not (Nx.greater d threshold))
            (Nx.logical_not (Nx.less d threshold))
        in
        let running = Nx.logical_and running (Nx.logical_not crossed) in
        let next = Nx.where up (Nx.mul_s eps 2.) (Nx.mul_s eps 0.5) in
        (Nx.where running next eps, (i, running)))
      (eps, (zero, running))
  in
  eps

(* Whitened rows

   Chains move in the whitened coordinates of a Gaussian, one flat row each
   ([Gaussian.flat]); the density and the positions keep their structure, [x]
   the template whose structure a row takes back. *)

let rows_density u lp f x z = lp (Rows.unravel u x (Gaussian.color_flat f z))

let enter u f x lp grad =
  let z = Gaussian.whiten_flat f (Rows.ravel u lp x) in
  { z; p = z; g = Gaussian.pull f (Rows.ravel u lp grad); lp }

let leave u f x at =
  ( Rows.unravel u x (Gaussian.color_flat f at.z),
    Rows.unravel u x (Gaussian.push f at.g) )

(* Fixed-length transitions

   Every chain shares one step size, one length and one Gaussian, and moves with
   unit metric in the Gaussian's whitened coordinates. *)

let max_steps = 1024
let chain_keys k c = Nx.Rng.split_batch ~n:c (Nx.Rng.fold_in k 0)

type ('u, 'f, 'd) trip = {
  at : ('u, 'f, 'd) point;
  running : 'd Nx.bool_t;
  diverging : 'd Nx.bool_t;
  steps : 'd Nx.int32_t; (* each chain's *)
  trip : 'd Nx.int32_t; (* the shared count *)
}


let trip_ptree (type u f) (u : (u, 'd) P.t) : ((u, f, 'd) trip, 'd) P.t =
  let point = point_ptree u in
  let walk c (t : (u, f, 'd) trip) : (u, f, 'd) trip =
    let open P.Walk in
    let at = field c "at" (structure point) t.at in
    let running = field c "running" tensor t.running in
    let diverging = field c "diverging" tensor t.diverging in
    let steps = field c "steps" tensor t.steps in
    let trip = field c "trip" tensor t.trip in
    { at; running; diverging; steps; trip }
  in
  P.v walk

type ('u, 'f, 'd) transition = {
  position : 'u;
  lp : (float, 'f, 'd) Nx.t;
  grad : 'u;
  alpha : (float, 'f, 'd) Nx.t;
  steps : 'd Nx.int32_t;
  diverging : 'd Nx.bool_t;
  energy : (float, 'f, 'd) Nx.t;
  z0 : (float, 'f, 'd) Nx.t;
  z1 : (float, 'f, 'd) Nx.t;
  p1 : (float, 'f, 'd) Nx.t;
  time : (float, 'f, 'd) Nx.t;
}

let transition (type f) context u lp k g ~step_size:eps ~length x
    (lp0 : (float, f, 'd) Nx.t) grad =
  let dt = Nx.dtype eps in
  let c = (Nx.shape lp0).(0) in
  let on = Nx.beside lp0 in
  let keys = chain_keys k c in
  let f = Gaussian.flat u x g in
  let lp_z = rows_density u lp f x in
  let { z = z0; g = g0; _ } = enter u f x lp0 grad in
  let rows = P.tensor in
  let p0 = momentum rows keys z0 in
  let h = Nx.broadcast_to [| c |] eps in
  let h0 = Nx.sub (kinetic rows h p0) lp0 in
  let jitter = Nx.mul_s (Noise.uniform (Nx.Rng.fold_in k 1) dt [||]) 2. in
  let n =
    Nx.clamp ~min:(Nx.Dtype.of_float dt 1.)
      ~max:(Nx.Dtype.of_float dt (float_of_int max_steps))
      (Nx.ceil (Nx.div (Nx.mul jitter length) eps))
  in
  let n = Nx.cast Nx.int32 n in
  let energy_error (at : (_, f, 'd) point) =
    let d = Nx.sub (Nx.sub (kinetic rows h at.p) at.lp) h0 in
    Nx.where (Nx.isnan d) (Nx.scalar_like d Float.infinity) d
  in
  let step t =
    let leaf, finite = leapfrog context rows lp_z t.running h t.at in
    let diverged =
      Nx.logical_and t.running
        (Nx.logical_or (Nx.logical_not finite)
           (Nx.greater_s (energy_error leaf) max_energy_error))
    in
    let moved = Nx.logical_and t.running (Nx.logical_not diverged) in
    {
      at = choose_point rows moved leaf t.at;
      running = moved;
      diverging = Nx.logical_or t.diverging diverged;
      steps = Nx.add t.steps (Nx.cast Nx.int32 t.running);
      trip = Nx.add_s t.trip 1l;
    }
  in
  let start = { z = z0; p = p0; g = g0; lp = lp0 } in
  let t =
    Rune.iterate (trip_ptree rows) ~max:max_steps
      ~until:(fun t ->
        Nx.logical_or
          (Nx.greater_equal t.trip n)
          (Nx.logical_not (Nx.any t.running)))
      step
      {
        at = start;
        running = Nx.ones_at on Nx.bool [| c |];
        diverging = Nx.zeros_at on Nx.bool [| c |];
        steps = Nx.zeros_at on Nx.int32 [| c |];
        trip = Nx.scalar_at on Nx.int32 0l;
      }
  in
  let alpha =
    Nx.where t.diverging (Nx.zeros_at on dt [| c |])
      (Nx.minimum_s (Nx.exp (Nx.neg (energy_error t.at))) 1.)
  in
  let uniform =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns tensor)
      (fun k -> Noise.uniform (Nx.Rng.fold_in k 1) dt [||])
      keys
  in
  let accepted = Nx.less uniform alpha in
  let x1, g1 = leave u f x t.at in
  {
    position = Rows.choose u accepted x1 x;
    lp = Nx.where accepted t.at.lp lp0;
    grad = Rows.choose u accepted g1 grad;
    alpha;
    steps = t.steps;
    diverging = t.diverging;
    energy = h0;
    z0;
    z1 = t.at.z;
    p1 = t.at.p;
    time = Nx.mul (Nx.cast dt n) eps;
  }

(* ChEES

   The length maximises the change in expected squared jumped distance (Hoffman,
   Radul and Sountsov 2021): a transition estimates the criterion's gradient in
   log length from the chains, in whitened coordinates, weighted by acceptance,
   and Adam steps it. *)

(* Adam's step and second-moment decay, without momentum. *)
let adam_rate = 0.025
let adam_decay = 0.95
let adam_eps = 1e-8

(* The weight [t^(-κ)] of iterate [t] in the length's average. *)
let average_kappa = 0.75

type ('f, 'd) chees = {
  log_length : (float, 'f, 'd) Nx.t;
  second : (float, 'f, 'd) Nx.t; (* Adam's second moment *)
  bar : (float, 'f, 'd) Nx.t; (* the averaged log length *)
  count : (float, 'f, 'd) Nx.t;
}


let chees_ptree (type f) () : ((f, 'd) chees, 'd) P.t =
  let walk c (a : (f, 'd) chees) : (f, 'd) chees =
    let open P.Walk in
    let log_length = field c "log_length" tensor a.log_length in
    let second = field c "second" tensor a.second in
    let bar = field c "bar" tensor a.bar in
    let count = field c "count" tensor a.count in
    { log_length; second; bar; count }
  in
  P.v walk

(* [centre w x] is the rows [x] less their mean over chains weighted by [w]. *)
let centre w x =
  let w = Rows.column w x in
  Nx.sub x
    (Nx.div
       (Nx.sum ~axes:[ 0 ] ~keepdims:true (Nx.mul w x))
       (Nx.sum ~axes:[ 0 ] ~keepdims:true w))

(* [chees_gradient m] is the acceptance-weighted mean over chains of the ChEES
   criterion's derivative in log length, [t d ((z1 - mean z1) · p1)] with [d =
   |z1 - mean z1|² - |z0 - mean z0|²], the means over chains, [z1]'s weighted by
   acceptance; zero when no chain accepts. *)
let chees_gradient (m : (_, _, 'd) transition) =
  let w = m.alpha in
  let total = Nx.sum w in
  let none = Nx.equal total (Nx.zeros_like total) in
  let w = Nx.where none (Nx.zeros_like w) w in
  let weights = Nx.where none (Nx.ones_like w) w in
  let d0 = centre (Nx.ones_like w) m.z0 and d1 = centre weights m.z1 in
  let d = Nx.sub (Rows.dot P.tensor w d1 d1) (Rows.dot P.tensor w d0 d0) in
  let per_chain = Nx.mul (Nx.mul m.time d) (Rows.dot P.tensor w d1 m.p1) in
  Nx.div
    (Nx.sum (Nx.mul w per_chain))
    (Nx.where none (Nx.ones_like total) total)

(* One Adam step up the criterion, the log length kept where the trajectory fits
   [max_steps] steps of [eps]. *)
let chees_step (a : _ chees) eps grad =
  let count = Nx.add_s a.count 1. in
  let second =
    Nx.add
      (Nx.mul_s a.second adam_decay)
      (Nx.mul_s (Nx.square grad) (1. -. adam_decay))
  in
  let corrected =
    Nx.div second
      (Nx.rsub_s 1. (Nx.pow (Nx.scalar_like count adam_decay) count))
  in
  let step = Nx.div grad (Nx.add_s (Nx.sqrt corrected) adam_eps) in
  let log_length =
    Nx.minimum
      (Nx.add a.log_length (Nx.mul_s step adam_rate))
      (Nx.log (Nx.mul_s eps (float_of_int max_steps)))
  in
  let weight = Nx.pow count (Nx.scalar_like count (-.average_kappa)) in
  let bar =
    Nx.add (Nx.mul (Nx.rsub_s 1. weight) a.bar) (Nx.mul weight log_length)
  in
  { log_length; second; bar; count }
