(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

let invalid_argf = Rows.invalid_argf

type 'f stats = { lp : (float, 'f) Nx.t; evaluations : Nx.int32_t }
type 'f stats' = 'f stats

let stats (type f) (_ : (float, f) Nx.dtype) : f stats P.t =
  let module S = struct
    type _ t = f stats'

    let walk c (s : f stats) : f stats =
      let open P.Walk in
      let lp = field c "lp" tensor s.lp in
      let evaluations = field c "evaluations" tensor s.evaluations in
      { lp; evaluations }
  end in
  P.nest (module S) P.unit

type ('u, 'f) state = {
  position : 'u;
  lp : (float, 'f) Nx.t;
  stats : 'f stats;
  draw : Nx.int32_t;
  ensembles : int;
}

type ('u, 'f) ensemble = ('u, 'f) state

let ptree (type u f) (u : u P.t) : (u, f) state P.t =
  let module S = struct
    type _ t = (u, f) ensemble

    let walk c s =
      let open P.Walk in
      let position = field c "position" (structure u) s.position in
      let lp = field c "lp" tensor s.lp in
      let stats = field c "stats" (structure (stats (Nx.dtype s.lp))) s.stats in
      let draw = field c "draw" tensor s.draw in
      let ensembles = field c "ensembles" int s.ensembles in
      { position; lp; stats; draw; ensembles }
  end in
  P.nest (module S) P.unit

(* Groups

   Walkers lie on the chain axis in [e] consecutive groups of [n]. Group [g]'s
   first half is its walkers [0, n / 2), its second the rest. *)

let grouped u e x =
  P.map u
    (fun _ t ->
      let s = Nx.shape t in
      Nx.reshape
        (Array.concat
           [ [| e; s.(0) / e |]; Array.sub s 1 (Array.length s - 1) ])
        t)
    x

let ungrouped u x =
  P.map u
    (fun _ t ->
      let s = Nx.shape t in
      Nx.reshape
        (Array.append [| s.(0) * s.(1) |] (Array.sub s 2 (Array.length s - 2)))
        t)
    x

let rows u lo hi x = P.map u (fun _ t -> Nx.slice [ Nx.A; Nx.R (lo, hi) ] t) x

(* Keys

   Transition [n] of a run has the key [fold_in run n], and walker [i] row [i]
   of its [split_batch]. Half [j]'s move draws the walker's slice from [fold_in
   key j] and its direction from [fold_in key (2 + j)]. *)

(* [half u lp e keys j x lp0] moves half [j] of every group along directions
   drawn from the Gaussian of the group's other half, the density evaluated on
   the moving half only. *)
let half (type f) u lp e keys j position (lp0 : (float, f) Nx.t) =
  let dt = Nx.dtype lp0 in
  let n = (Nx.shape lp0).(0) / e in
  let h = n / 2 in
  let moving, held = if j = 0 then ((0, h), (h, n)) else ((h, n), (0, h)) in
  let split p v =
    ( rows p (fst moving) (snd moving) (grouped p e v),
      rows p (fst held) (snd held) (grouped p e v) )
  in
  let x, others = split u position in
  let l, held_lp = split P.tensor lp0 in
  let keys, _ = split Nx.Rng.ptree keys in
  let gp = Gaussian.ptree u in
  let gaussians =
    Rune.vmap
      P.(u @-> returns gp)
      (fun x -> Adapt.population u dt (Nx.zeros dt [| snd held - fst held |]) x)
      others
  in
  let flat p v = ungrouped p v in
  let keys = flat Nx.Rng.ptree keys in
  (* A direction of unit length in whitened coordinates. *)
  let z =
    Slice.unit_directions u
      (Rune.vmap
         P.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
         (fun k -> Nx.Rng.fold_in k (2 + j))
         keys)
      (flat P.tensor l) (flat u x)
  in
  let direction =
    Rune.vmap
      P.(gp @-> u @-> returns u)
      (fun g zs -> Rune.vmap P.(u @-> returns u) (Geometry.direction u g) zs)
      gaussians (grouped u e z)
  in
  let slice_keys =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
      (fun k -> Nx.Rng.fold_in k j)
      keys
  in
  let x, l, (), evaluations =
    Slice.move "Norn.Ensemble.step" u P.unit
      (fun _ x -> (lp x, ()))
      slice_keys ~direction:(flat u direction) (flat u x) (flat P.tensor l) ()
  in
  (* The halves back in their places, and the held half's evaluations. *)
  let join p a b =
    let a = grouped p e a in
    P.map2 p
      (fun _ a b ->
        Nx.concatenate ~axis:1 (if j = 0 then [ a; b ] else [ b; a ]))
      a b
    |> ungrouped p
  in
  let none = Nx.zeros Nx.int32 [| e; snd held - fst held |] in
  (join u x others, join P.tensor l held_lp, join P.tensor evaluations none)

(* Starting *)

let init u ?(ensembles = 1) lp position =
  let context = "Norn.Ensemble.init" in
  if ensembles < 1 then
    invalid_argf "%s: ensembles = %d is not positive" context ensembles;
  let c = Rows.count context u position in
  if c mod ensembles <> 0 then
    invalid_argf "%s: %d walkers do not split into %d ensembles" context c
      ensembles;
  let n = c / ensembles and d = Rows.elements u c position in
  if n < 2 * d then
    invalid_argf
      "%s: an ensemble of %d walkers over %d coordinates; an ensemble needs at \
       least twice as many walkers as coordinates"
      context n d;
  let l = lp position in
  Rows.check_density context u lp position l;
  {
    position;
    lp = l;
    stats = { lp = l; evaluations = Nx.zeros Nx.int32 [| c |] };
    draw = Nx.scalar Nx.int32 0l;
    ensembles;
  }

(* Transitions *)

let step u lp k (s : (_, _) state) =
  let c = (Nx.shape s.lp).(0) in
  let keys = Nx.Rng.split_batch ~n:c k in
  let x, l, first = half u lp s.ensembles keys 0 s.position s.lp in
  let x, l, second = half u lp s.ensembles keys 1 x l in
  {
    s with
    position = x;
    lp = l;
    stats = { lp = l; evaluations = Nx.add first second };
    draw = Nx.add s.draw (Nx.scalar Nx.int32 1l);
  }

let warmup u lp k ~steps (s : (_, _) state) =
  if steps < 0 then
    invalid_argf "Norn.Ensemble.warmup: steps = %d is negative" steps;
  if steps = 0 then s
  else
    fst
      (Rune.scan (ptree u) P.tensor P.unit
         ~f:(fun (s : (_, _) state) _ ->
           (step u lp (Nx.Rng.fold_in_tensor k s.draw) s, ()))
         ~init:s
         (Nx.zeros Nx.int32 [| steps |]))

let sample u lp k ~draws (s : (_, _) state) =
  if draws < 1 then
    invalid_argf "Norn.Ensemble.sample: draws = %d is not positive" draws;
  let st = stats (Nx.dtype s.lp) in
  let s, (xs, ss) =
    Rune.scan (ptree u) P.tensor
      P.(pair u st)
      ~f:(fun (s : (_, _) state) _ ->
        let s = step u lp (Nx.Rng.fold_in_tensor k s.draw) s in
        (s, (s.position, s.stats)))
      ~init:s
      (Nx.zeros Nx.int32 [| draws |])
  in
  let swap x = Nx.moveaxis 0 1 x in
  let xs = P.map u (fun _ t -> swap t) xs in
  let ss = P.map st (fun _ t -> swap t) ss in
  (s, Draws.v u xs, Draws.v st ss)
