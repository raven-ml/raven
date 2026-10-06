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
