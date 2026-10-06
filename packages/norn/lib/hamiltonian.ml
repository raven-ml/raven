(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

type ('u, 'f) point = { z : 'u; p : 'u; g : 'u; lp : (float, 'f) Nx.t }
type ('u, 'f) point' = ('u, 'f) point

let point_ptree (type u f) (u : u P.t) : (u, f) point P.t =
  let module S = struct
    type _ t = (u, f) point'

    let walk c (s : (u, f) point) : (u, f) point =
      let open P.Walk in
      let z = field c "z" (structure u) s.z in
      let p = field c "p" (structure u) s.p in
      let g = field c "g" (structure u) s.g in
      let lp = field c "lp" tensor s.lp in
      { z; p; g; lp }
  end in
  P.nest (module S) P.unit

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

let search (type f) context u lp_z ~reduce keys start (eps : (float, f) Nx.t) =
  let dt = Nx.dtype eps in
  let c = (Nx.shape start.lp).(0) in
  let all = Nx.ones Nx.bool [| c |] in
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
    reduce (Nx.where (Nx.isnan d) (Nx.scalar dt Float.neg_infinity) d)
  in
  let threshold = Nx.scalar dt (Float.log search_acceptance) in
  let up = Nx.greater (log_ratio eps (Nx.scalar Nx.int32 0l)) threshold in
  let running = Nx.ones_like up in
  let carry = P.(pair tensor (pair tensor tensor)) in
  let eps, _ =
    Rune.iterate carry ~max:200
      ~until:(fun (_, (_, running)) -> Nx.logical_not (Nx.any running))
      ~f:(fun (eps, (i, running)) ->
        let i = Nx.add i (Nx.scalar Nx.int32 1l) in
        let d = log_ratio eps i in
        let crossed =
          Nx.where up
            (Nx.logical_not (Nx.greater d threshold))
            (Nx.logical_not (Nx.less d threshold))
        in
        let running = Nx.logical_and running (Nx.logical_not crossed) in
        let next = Nx.where up (Nx.mul_s eps 2.) (Nx.mul_s eps 0.5) in
        (Nx.where running next eps, (i, running)))
      (eps, (Nx.scalar Nx.int32 0l, running))
  in
  eps

(* Fixed-length transitions

   Every chain shares one step size, one length and one Gaussian, and moves with
   unit metric in the Gaussian's whitened coordinates. *)

let max_steps = 1024
let color u g z = Rune.vmap P.(u @-> returns u) (Geometry.color u g) z
let whiten u g x = Rune.vmap P.(u @-> returns u) (Geometry.whiten u g) x

(* [to_whitened u g z gx] is the gradient in whitened coordinates of a gradient
   [gx] at [color z]: [color]'s transpose applied to it; [to_original] is
   [whiten]'s. *)
let to_whitened u g z gx =
  Rune.vmap
    P.(u @-> u @-> returns u)
    (fun z gx -> snd (Rune.vjp u u (Geometry.color u g) z) gx)
    z gx

let to_original u g x gz =
  Rune.vmap
    P.(u @-> u @-> returns u)
    (fun x gz -> snd (Rune.vjp u u (Geometry.whiten u g) x) gz)
    x gz

let chain_keys k c = Nx.Rng.split_batch ~n:c (Nx.Rng.fold_in k 0)

type ('u, 'f) trip = {
  at : ('u, 'f) point;
  running : Nx.bool_t;
  diverging : Nx.bool_t;
  steps : Nx.int32_t; (* each chain's *)
  trip : Nx.int32_t; (* the shared count *)
}

type ('u, 'f) trip' = ('u, 'f) trip

let trip_ptree (type u f) (u : u P.t) : (u, f) trip P.t =
  let point = point_ptree u in
  let module S = struct
    type _ t = (u, f) trip'

    let walk c (t : (u, f) trip) : (u, f) trip =
      let open P.Walk in
      let at = field c "at" (structure point) t.at in
      let running = field c "running" tensor t.running in
      let diverging = field c "diverging" tensor t.diverging in
      let steps = field c "steps" tensor t.steps in
      let trip = field c "trip" tensor t.trip in
      { at; running; diverging; steps; trip }
  end in
  P.nest (module S) P.unit

type ('u, 'f) transition = {
  position : 'u;
  lp : (float, 'f) Nx.t;
  grad : 'u;
  alpha : (float, 'f) Nx.t;
  steps : Nx.int32_t;
  diverging : Nx.bool_t;
  energy : (float, 'f) Nx.t;
  z0 : 'u;
  z1 : 'u;
  p1 : 'u;
  time : (float, 'f) Nx.t;
}

let transition (type f) context u lp k g ~step_size:eps ~length x
    (lp0 : (float, f) Nx.t) grad =
  let dt = Nx.dtype eps in
  let c = (Nx.shape lp0).(0) in
  let i32 v = Nx.scalar Nx.int32 (Int32.of_int v) in
  let keys = chain_keys k c in
  let lp_z z = lp (color u g z) in
  let z0 = whiten u g x in
  let g0 = to_whitened u g z0 grad in
  let p0 = momentum u keys z0 in
  let h = Nx.broadcast_to [| c |] eps in
  let h0 = Nx.sub (kinetic u h p0) lp0 in
  let jitter = Nx.mul_s (Nx.Rng.uniform (Nx.Rng.fold_in k 1) dt [||]) 2. in
  let n =
    Nx.clamp ~min:(Nx_dtype.of_float dt 1.)
      ~max:(Nx_dtype.of_float dt (float_of_int max_steps))
      (Nx.ceil (Nx.div (Nx.mul jitter length) eps))
  in
  let n = Nx.cast Nx.int32 n in
  let energy_error (at : (_, f) point) =
    let d = Nx.sub (Nx.sub (kinetic u h at.p) at.lp) h0 in
    Nx.where (Nx.isnan d) (Nx.scalar dt Float.infinity) d
  in
  let step t =
    let leaf, finite = leapfrog context u lp_z t.running h t.at in
    let diverged =
      Nx.logical_and t.running
        (Nx.logical_or (Nx.logical_not finite)
           (Nx.greater (energy_error leaf) (Nx.scalar dt max_energy_error)))
    in
    let moved = Nx.logical_and t.running (Nx.logical_not diverged) in
    {
      at = choose_point u moved leaf t.at;
      running = moved;
      diverging = Nx.logical_or t.diverging diverged;
      steps = Nx.add t.steps (Nx.cast Nx.int32 t.running);
      trip = Nx.add t.trip (i32 1);
    }
  in
  let start = { z = z0; p = p0; g = g0; lp = lp0 } in
  let t =
    Rune.iterate (trip_ptree u) ~max:max_steps
      ~until:(fun t ->
        Nx.logical_or
          (Nx.greater_equal t.trip n)
          (Nx.logical_not (Nx.any t.running)))
      ~f:step
      {
        at = start;
        running = Nx.ones Nx.bool [| c |];
        diverging = Nx.zeros Nx.bool [| c |];
        steps = Nx.zeros Nx.int32 [| c |];
        trip = i32 0;
      }
  in
  let alpha =
    Nx.where t.diverging (Nx.zeros dt [| c |])
      (Nx.minimum (Nx.exp (Nx.neg (energy_error t.at))) (Nx.ones dt [| c |]))
  in
  let uniform =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns tensor)
      (fun k -> Nx.Rng.uniform (Nx.Rng.fold_in k 1) dt [||])
      keys
  in
  let accepted = Nx.less uniform alpha in
  let x1 = color u g t.at.z in
  let g1 = to_original u g x1 t.at.g in
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
