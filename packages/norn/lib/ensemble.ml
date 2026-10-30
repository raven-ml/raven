(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

let invalid_argf = Rows.invalid_argf

type ('f, 'd) stats = { lp : (float, 'f, 'd) Nx.t; acceptance : (float, 'f, 'd) Nx.t }

let stats (type f) (_ : (float, f) Nx.dtype) : ((f, 'd) stats, 'd) P.t =
  let walk c (s : (f, 'd) stats) : (f, 'd) stats =
    let open P.Walk in
    let lp = field c "lp" tensor s.lp in
    let acceptance = field c "acceptance" tensor s.acceptance in
    { lp; acceptance }
  in
  P.v walk

type ('u, 'f, 'd) state = {
  position : 'u;
  lp : (float, 'f, 'd) Nx.t;
  stats : ('f, 'd) stats;
  draw : 'd Nx.int32_t;
  ensembles : int;
}

type ('u, 'f, 'd) ensemble = ('u, 'f, 'd) state

let ptree (type u f) (u : (u, 'd) P.t) : ((u, f, 'd) state, 'd) P.t =
  let walk c (s : (u, f, 'd) ensemble) : (u, f, 'd) ensemble =
    let open P.Walk in
    let position = field c "position" (structure u) s.position in
    let lp = field c "lp" tensor s.lp in
    let stats = field c "stats" (structure (stats (Nx.dtype s.lp))) s.stats in
    let draw = field c "draw" tensor s.draw in
    let ensembles = field c "ensembles" int s.ensembles in
    { position; lp; stats; draw; ensembles }
  in
  P.v walk

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

(* The stretch move's scale [a]: a walker moves to [x_j + z (x_k - x_j)] with
   [z] of density proportional to [1 / sqrt z] on [[1 / a, a]], Goodman and
   Weare's value. *)
let scale = 2.

(* Keys

   Transition [n] of a run has the key [fold_in run n], and walker [i] row [i]
   of its [split_batch]. In half [j]'s move a walker draws its partner from
   [fold_in (fold_in key j) 0], its stretch from [... 1] and its acceptance from
   [... 2]. *)

(* [half u lp e keys j x lp0] moves half [j] of every group by the stretch move
   toward or away from a walker of the group's other half, the density evaluated
   on the moving half only, and is the walkers' positions, log densities and
   acceptance probabilities, [0] in the held half. *)
let half (type f) u lp e keys j position (lp0 : (float, f, 'd) Nx.t) =
  let dt = Nx.dtype lp0 in
  let n = (Nx.shape lp0).(0) / e in
  let h = n / 2 in
  let moving, held = if j = 0 then ((0, h), (h, n)) else ((h, n), (0, h)) in
  let m = snd moving - fst moving and o = snd held - fst held in
  let split p v =
    let g = grouped p e v in
    (rows p (fst moving) (snd moving) g, rows p (fst held) (snd held) g)
  in
  let x, others = split u position in
  let l, held_lp = split P.tensor lp0 in
  let keys, _ = split (Nx.Ptree.v Nx.Rng.walk) keys in
  let x = ungrouped u x and l = ungrouped P.tensor l in
  let draw i =
    Rune.vmap
      P.((Nx.Ptree.v Nx.Rng.walk) @-> returns tensor)
      (fun k -> Noise.uniform (Nx.Rng.fold_in (Nx.Rng.fold_in k j) i) dt [||])
      (ungrouped (Nx.Ptree.v Nx.Rng.walk) keys)
  in
  (* A partner drawn from the held half of the walker's own group. *)
  let pick =
    Nx.minimum_s
      (Nx.cast Nx.int64 (Nx.floor (Nx.mul_s (draw 0) (float_of_int o))))
      (Int64.of_int (o - 1))
  in
  let first =
    Nx.mul_s (Nx.arange_at (Nx.beside pick) Nx.int64 0 e 1) (Int64.of_int o)
  in
  let index =
    Nx.reshape
      [| e * m |]
      (Nx.add (Nx.reshape [| e; 1 |] first) (Nx.reshape [| e; m |] pick))
  in
  let partner =
    P.map u (fun _ t -> Nx.take ~axis:0 index t) (ungrouped u others)
  in
  let z =
    Nx.div_s (Nx.square (Nx.add_s (Nx.mul_s (draw 1) (scale -. 1.)) 1.)) scale
  in
  let y =
    P.map2 u
      (fun _ p t ->
        if Rows.float_leaf t then
          Nx.add p (Nx.mul (Rows.column z t) (Nx.sub t p))
        else t)
      partner x
  in
  let ly = lp y in
  Rows.check_range "Norn.Ensemble.step" ly;
  let d = Rows.elements u (e * m) x in
  let log_ratio =
    Nx.add (Nx.mul_s (Nx.log z) (float_of_int (d - 1))) (Nx.sub ly l)
  in
  let accepted = Nx.less (Nx.log (draw 2)) log_ratio in
  let alpha = Nx.minimum (Nx.exp log_ratio) (Nx.ones_like log_ratio) in
  let x = Rows.choose u accepted y x and l = Nx.where accepted ly l in
  (* The halves back in their places. *)
  let join p a b =
    let a = grouped p e a in
    P.map2 p
      (fun _ a b ->
        Nx.concatenate ~axis:1 (if j = 0 then [ a; b ] else [ b; a ]))
      a b
    |> ungrouped p
  in
  let none = Nx.zeros_at (Nx.beside alpha) dt [| e; o |] in
  (join u x others, join P.tensor l held_lp, join P.tensor alpha none)

(* Starting *)

let init u ?(ensembles = 1) lp position =
  let context = "Norn.Ensemble.init" in
  if ensembles < 1 then
    invalid_argf "%s: ensembles = %d is not positive" context ensembles;
  let c = Rows.count context u position in
  Rows.whole_chains context ~reads:"a walker moves along others" u position;
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
    stats = { lp = l; acceptance = Nx.zeros_like l };
    draw = Nx.scalar_at (Nx.beside l) Nx.int32 0l;
    ensembles;
  }

(* Transitions *)

(* [split e keys] is a permutation of the walkers that shuffles each group:
   group [g]'s order comes from its first walker's key, so it does not depend on
   the number of groups. Splitting the shuffled groups in halves splits each
   group at random, which mixes faster than fixed halves. *)
let split e keys =
  let c = (Nx.shape (keys : 'd Nx.Rng.key :> 'd Nx.int32_t)).(0) in
  let n = c / e in
  let firsts =
    Nx.Rng.of_tensor
      (Nx.slice [ Nx.A; Nx.I 0 ]
         (Nx.reshape [| e; n; 2 |] (keys : 'd Nx.Rng.key :> 'd Nx.int32_t)))
  in
  let order =
    Rune.vmap
      P.((Nx.Ptree.v Nx.Rng.walk) @-> returns tensor)
      (fun k ->
        Nx.argsort (Noise.uniform (Nx.Rng.fold_in k 2) Nx.float32 [| n |]))
      firsts
  in
  let first =
    Nx.mul_s (Nx.arange_at (Nx.beside order) Nx.int64 0 e 1) (Int64.of_int n)
  in
  Nx.reshape [| c |]
    (Nx.add (Nx.reshape [| e; 1 |] first) (Nx.cast Nx.int64 order))

let step u lp k (s : (_, _, 'd) state) =
  let k = Nx.Rng.place (Nx.beside s.lp) k in
  let c = (Nx.shape s.lp).(0) in
  let keys = Nx.Rng.split_batch ~n:c k in
  let perm = split s.ensembles keys in
  let back = Nx.cast Nx.int64 (Nx.argsort perm) in
  let take p v = P.map p (fun _ t -> Nx.take ~axis:0 perm t) v in
  let untake p v = P.map p (fun _ t -> Nx.take ~axis:0 back t) v in
  let keys' = take (Nx.Ptree.v Nx.Rng.walk) keys in
  let x, l, first =
    half u lp s.ensembles keys' 0 (take u s.position) (take P.tensor s.lp)
  in
  let x, l, second = half u lp s.ensembles keys' 1 x l in
  let l = untake P.tensor l in
  {
    s with
    position = untake u x;
    lp = l;
    stats = { lp = l; acceptance = untake P.tensor (Nx.add first second) };
    draw = Nx.add_s s.draw 1l;
  }

let warmup u lp k ~steps (s : (_, _, 'd) state) =
  let k = Nx.Rng.place (Nx.beside s.lp) k in
  if steps < 0 then
    invalid_argf "Norn.Ensemble.warmup: steps = %d is negative" steps;
  if steps = 0 then s
  else
    fst
      (Rune.scan (ptree u) P.tensor P.unit
         (fun (s : (_, _, 'd) state) _ ->
           (step u lp (Nx.Rng.fold_in_tensor k s.draw) s, ()))
         ~init:s
         (Nx.zeros_at (Nx.beside s.lp) Nx.int32 [| steps |]))

let sample u lp k ~draws (s : (_, _, 'd) state) =
  let k = Nx.Rng.place (Nx.beside s.lp) k in
  if draws < 1 then
    invalid_argf "Norn.Ensemble.sample: draws = %d is not positive" draws;
  let st = stats (Nx.dtype s.lp) in
  let s, (xs, ss) =
    Rune.scan (ptree u) P.tensor
      P.(pair u st)
      (fun (s : (_, _, 'd) state) _ ->
        let s = step u lp (Nx.Rng.fold_in_tensor k s.draw) s in
        (s, (s.position, s.stats)))
      ~init:s
      (Nx.zeros_at (Nx.beside s.lp) Nx.int32 [| draws |])
  in
  let swap x = Nx.moveaxis 0 1 x in
  let xs = P.map u (fun _ t -> swap t) xs in
  let ss = P.map st (fun _ t -> swap t) ss in
  (s, Draws.v u xs, Draws.v st ss)
