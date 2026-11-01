(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree
module H = Hamiltonian

let invalid_argf = Rows.invalid_argf

type ('u, 'f, 'd) state = {
  position : 'u;
  lp : (float, 'f, 'd) Nx.t;
  grad : 'u;
  step_size : (float, 'f, 'd) Nx.t;
  geometry : ('u, 'f, 'd) Gaussian.t;
  stats : ('f, 'd) Stats.t;
  draw : 'd Nx.int32_t;
  accept : (float, 'f, 'd) Nx.t;
  max_depth : int;
}

type ('u, 'f, 'd) nuts = ('u, 'f, 'd) state

let ptree (type u f) (u : (u, 'd) P.t) : ((u, f, 'd) state, 'd) P.t =
  let g = Gaussian.ptree u in
  let walk c (s : (u, f, 'd) nuts) : (u, f, 'd) nuts =
    let open P.Walk in
    let position = field c "position" (structure u) s.position in
    let lp = field c "lp" tensor s.lp in
    let grad = field c "grad" (structure u) s.grad in
    let step_size = field c "step_size" tensor s.step_size in
    let geometry = field c "geometry" (structure g) s.geometry in
    let stats =
      field c "stats" (structure (Stats.ptree (Nx.dtype s.lp))) s.stats
    in
    let draw = field c "draw" tensor s.draw in
    let accept = field c "accept" tensor s.accept in
    let max_depth = field c "max_depth" int s.max_depth in
    {
      position;
      lp;
      grad;
      step_size;
      geometry;
      stats;
      draw;
      accept;
      max_depth;
    }
  in
  P.v walk

let stats = Stats.ptree

(* [logaddexp a b] is [log (exp a + exp b)], [-inf] where both are. *)
let logaddexp a b =
  let m = Nx.maximum a b in
  let ninf = Nx.scalar_like m Float.neg_infinity in
  let s =
    Nx.add m (Nx.log (Nx.add (Nx.exp (Nx.sub a m)) (Nx.exp (Nx.sub b m))))
  in
  Nx.where (Nx.equal m ninf) ninf s

(* Keys

   Transition [n] of a run has the key [fold_in k n], warmup's transitions
   counted with the rest, so each transition key has one consumer. Chain [i] has
   row [i] of [split_batch] of the transition's key. Its momentum draws from
   [fold_in key 0]; doubling [j] draws its direction, the uniforms of its merges
   and of its top-level merge from [fold_in (fold_in (fold_in key 1) j) id],
   with [id] [0] for the direction, [l 2^D + n] for the merge into node [n] of
   level [l], and [(D + 1) 2^D] for the top-level merge, [D] the maximum depth.
   A step-size search before a warmup transition draws its trial [i]'s momentum
   from [fold_in (fold_in key 2) i]. *)

(* [uniforms dt keys j ids] is each chain's uniform of every id in [ids], shape
   [[c; n]] for [n] ids: chain [i]'s for [id] draws from [fold_in (fold_in
   (fold_in keys.(i) 1) j) id]. *)
let uniforms (type f) dt keys j ids : (float, f, 'd) Nx.t =
  Rune.vmap
    P.(Nx.Rng.ptree @-> returns tensor)
    (fun k ->
      let k = Nx.Rng.fold_in_tensor (Nx.Rng.fold_in k 1) j in
      Rune.vmap
        P.(tensor @-> returns tensor)
        (fun id -> Noise.uniform (Nx.Rng.fold_in_tensor k id) dt [||])
        ids)
    keys

(* Trees

   A trajectory is built by doublings: doubling [j] extends it by a subtree of
   [2^j] leapfrog steps in a random direction. The subtree is built leaf by
   leaf; the leaf of index [i] completes the node of level [l] that ends at it
   whenever [2^l] divides [i + 1]. A completed node that is the first half of
   its parent waits in its level's slot; one that is the second half merges with
   the slot into its parent. Every chain takes the same leaf schedule, so the
   leaf counters are shared and only each chain's stop differs. A leaf computes
   only the merges its index completes, one on average, in a loop over them,
   and the subtree joins the trajectory only at its doubling's last leaf, in a
   branch: both read the shared counters.

   The tree moves in whitened coordinates held as rows ([H.enter]): a position
   or a momentum is a [[c; d]] matrix, and the slots of every level are one node
   whose tensors lead with an axis of levels. *)

type ('f, 'd) rows = (float, 'f, 'd) Nx.t

type ('u, 'f, 'd) point = ('u, 'f, 'd) H.point = {
  z : 'u;
  p : 'u;
  g : 'u;
  lp : ('f, 'd) rows;
}

(* A completed subtree: the momenta of its first and last leaf, in the order it
   was built, their sum, its log weight and its proposal. *)
type ('f, 'd) node = {
  first : ('f, 'd) rows;
  last : ('f, 'd) rows;
  rho : ('f, 'd) rows;
  weight : ('f, 'd) rows;
  prop : (('f, 'd) rows, 'f, 'd) point;
}

type ('f, 'd) tree = {
  back : (('f, 'd) rows, 'f, 'd) point; (* the trajectory's backward end *)
  front : (('f, 'd) rows, 'f, 'd) point; (* its forward end *)
  cursor : (('f, 'd) rows, 'f, 'd) point; (* the last leaf *)
  sample : (('f, 'd) rows, 'f, 'd) point;
  sum : ('f, 'd) rows; (* the trajectory's momenta, summed *)
  total : ('f, 'd) rows; (* its log weight *)
  slots : ('f, 'd) node; (* every level's, stacked *)
  forward : 'd Nx.bool_t;
  running : 'd Nx.bool_t;
  diverging : 'd Nx.bool_t;
  saturated : 'd Nx.bool_t;
  depth : 'd Nx.int32_t;
  steps : 'd Nx.int32_t;
  accepted : ('f, 'd) rows; (* the sum of the leaves' acceptance *)
  doubling : 'd Nx.int32_t; (* [j], shared by the chains *)
  leaf : 'd Nx.int32_t; (* [i] *)
  size : 'd Nx.int32_t; (* [2^j] *)
}


let node_ptree (type f) () : ((f, 'd) node, 'd) P.t =
  let point = H.point_ptree P.tensor in
  let walk c (s : (f, 'd) node) : (f, 'd) node =
    let open P.Walk in
    let first = field c "first" tensor s.first in
    let last = field c "last" tensor s.last in
    let rho = field c "rho" tensor s.rho in
    let weight = field c "weight" tensor s.weight in
    let prop = field c "prop" (structure point) s.prop in
    { first; last; rho; weight; prop }
  in
  P.v walk

let tree_ptree (type f) () : ((f, 'd) tree, 'd) P.t =
  let point = H.point_ptree P.tensor and node = node_ptree () in
  let walk c (s : (f, 'd) tree) : (f, 'd) tree =
    let open P.Walk in
    let back = field c "back" (structure point) s.back in
    let front = field c "front" (structure point) s.front in
    let cursor = field c "cursor" (structure point) s.cursor in
    let sample = field c "sample" (structure point) s.sample in
    let sum = field c "sum" tensor s.sum in
    let total = field c "total" tensor s.total in
    let slots = field c "slots" (structure node) s.slots in
    let forward = field c "forward" tensor s.forward in
    let running = field c "running" tensor s.running in
    let diverging = field c "diverging" tensor s.diverging in
    let saturated = field c "saturated" tensor s.saturated in
    let depth = field c "depth" tensor s.depth in
    let steps = field c "steps" tensor s.steps in
    let accepted = field c "accepted" tensor s.accepted in
    let doubling = field c "doubling" tensor s.doubling in
    let leaf = field c "leaf" tensor s.leaf in
    let size = field c "size" tensor s.size in
    {
      back;
      front;
      cursor;
      sample;
      sum;
      total;
      slots;
      forward;
      running;
      diverging;
      saturated;
      depth;
      steps;
      accepted;
      doubling;
      leaf;
      size;
    }
  in
  P.v walk

(* What a doubling's last leaf changes: the trajectory with the subtree merged
   in, and whether each chain stops or saturates. *)
type ('f, 'd) ends = {
  back_end : (('f, 'd) rows, 'f, 'd) point;
  front_end : (('f, 'd) rows, 'f, 'd) point;
  chosen : (('f, 'd) rows, 'f, 'd) point;
  summed : ('f, 'd) rows;
  weighed : ('f, 'd) rows;
  deeper : 'd Nx.int32_t;
  stop : 'd Nx.bool_t;
  saturates : 'd Nx.bool_t;
}

let ends_ptree (type f) () : ((f, 'd) ends, 'd) P.t =
  let point = H.point_ptree P.tensor in
  let walk c (e : (f, 'd) ends) : (f, 'd) ends =
    let open P.Walk in
    let back_end = field c "back_end" (structure point) e.back_end in
    let front_end = field c "front_end" (structure point) e.front_end in
    let chosen = field c "chosen" (structure point) e.chosen in
    let summed = field c "summed" tensor e.summed in
    let weighed = field c "weighed" tensor e.weighed in
    let deeper = field c "deeper" tensor e.deeper in
    let stop = field c "stop" tensor e.stop in
    let saturates = field c "saturates" tensor e.saturates in
    { back_end; front_end; chosen; summed; weighed; deeper; stop; saturates }
  in
  P.v walk

(* [mask m x] is the mask [m] over [x]'s shape, [m]'s axes leading [x]'s: a
   [where] of operands of one shape moves none of them. *)
let mask m x =
  let ones = Array.make (Nx.ndim x - Nx.ndim m) 1 in
  Nx.broadcast_to (Nx.shape x)
    (Nx.reshape (Array.append (Nx.shape m) ones) m)

(* [pick m a b] is [a] where the mask [m] holds and [b] elsewhere, [m]'s axes
   leading [b]'s. *)
let pick m a b = Nx.where (mask m b) a b

(* [choose_point m w a b] is [pick_point m a b] with [w] the mask [m] over the
   rows. *)
let choose_point m w a b =
  {
    z = Nx.where w a.z b.z;
    p = Nx.where w a.p b.p;
    g = Nx.where w a.g b.g;
    lp = pick m a.lp b.lp;
  }

let pick_point m a b = choose_point m (mask m b.z) a b

let pick_node m a b =
  let w = mask m b.first in
  {
    first = Nx.where w a.first b.first;
    last = Nx.where w a.last b.last;
    rho = Nx.where w a.rho b.rho;
    weight = pick m a.weight b.weight;
    prop = choose_point m w a.prop b.prop;
  }

let map_node f n =
  {
    first = f n.first;
    last = f n.last;
    rho = f n.rho;
    weight = f n.weight;
    prop = { z = f n.prop.z; p = f n.prop.p; g = f n.prop.g; lp = f n.prop.lp };
  }

(* [forward_all pairs] is, per chain, whether every pair [(a, b)] of momenta has
   [a · b > 0]: the generalised no-U-turn criterion of a stretch whose end
   momenta are [minus] and [plus] and whose momenta sum to [rho] is the pairs
   [(plus, rho)] and [(minus, rho)], both ends still moving along [rho]. With
   unit metric the sharp momentum is the momentum. The pairs' inner products are
   one reduction. *)
let forward_all pairs =
  let a = Nx.stack ~axis:0 (List.map fst pairs)
  and b = Nx.stack ~axis:0 (List.map snd pairs) in
  let dots = Nx.sum ~axes:[ 2 ] (Nx.mul a b) in
  Nx.all ~axes:[ 0 ] (Nx.greater dots (Nx.zeros_like dots))

(* [merge u init final] is the parent of the sibling nodes [init] and [final],
   built in that order, its proposal [final]'s if the uniform [u] falls below
   [final]'s share of the weight, and whether it keeps the criterion around
   itself and between its halves. *)
let merge u init final =
  let weight = logaddexp init.weight final.weight in
  let take = Nx.less u (Nx.exp (Nx.sub final.weight weight)) in
  let rho = Nx.add init.rho final.rho in
  let inner = Nx.add init.rho final.first
  and outer = Nx.add final.rho init.last in
  let ok =
    forward_all
      [
        (final.last, rho);
        (init.first, rho);
        (final.first, inner);
        (init.first, inner);
        (final.last, outer);
        (init.last, outer);
      ]
  in
  ( {
      first = init.first;
      last = final.last;
      rho;
      weight;
      prop = pick_point take final.prop init.prop;
    },
    ok )

let leapfrog lp_z running h s =
  H.leapfrog "Norn.Nuts.step" P.tensor lp_z running h s

(* [transition u lp max_depth eps keys geometry s] is one transition from the
   position [s] of each chain. *)
let transition (type f) u lp max_depth (eps : (float, f, 'd) Nx.t) keys geometry
    position lp0 grad =
  let dt = Nx.dtype eps in
  let c = (Nx.shape eps).(0) in
  let on = Nx.beside eps in
  let i32 v = Nx.scalar_at on Nx.int32 (Int32.of_int v) in
  let ints f =
    Nx.place on
      (Nx.create Nx.int32 [| max_depth |]
         (Array.init max_depth (fun l -> Int32.of_int (f l))))
  in
  let f = Gaussian.flat u position geometry in
  let lp_z = H.rows_density u lp f position in
  let { H.z = z0; g = g0; _ } = H.enter u f position lp0 grad in
  (* The tree's per-chain values lie where the chains do, so that what a step
     computes from them lies where they did. *)
  let chains = Nx.placement lp0 in
  let p0 = Nx.place (Nx.placement z0) (H.momentum P.tensor keys z0) in
  let kinetic p = H.kinetic P.tensor eps p in
  let h0 = Nx.sub (kinetic p0) lp0 in
  let start = { z = z0; p = p0; g = g0; lp = lp0 } in
  let empty = { first = p0; last = p0; rho = p0; weight = lp0; prop = start } in
  let no = Nx.zeros_at chains Nx.bool [| c |] in
  let init =
    {
      back = start;
      front = start;
      cursor = start;
      sample = start;
      sum = p0;
      total = Nx.zeros_at chains dt [| c |];
      slots =
        map_node
          (fun x ->
            Nx.broadcast_to (Array.append [| max_depth |] (Nx.shape x)) x)
          empty;
      forward = no;
      running = Nx.ones_at chains Nx.bool [| c |];
      diverging = no;
      saturated = no;
      depth = Nx.zeros_at chains Nx.int32 [| c |];
      steps = Nx.zeros_at chains Nx.int32 [| c |];
      accepted = Nx.zeros_at chains dt [| c |];
      doubling = i32 0;
      leaf = i32 0;
      size = i32 1;
    }
  in
  (* Level [l]'s nodes span [2^(l + 1)] leaves; the merge into node [n] of level
     [l + 1] draws id [(l + 1) 2^D + n]. *)
  let spans = ints (fun l -> 1 lsl (l + 1))
  and halves = ints (fun l -> 1 lsl l) in
  let parents = ints (fun l -> (l + 1) lsl max_depth) in
  let above = ints (fun l -> l + 1) in
  let top_code = (max_depth + 1) lsl max_depth in
  let cascading = P.(pair (pair (node_ptree ()) tensor) tensor) in
  let ended = ends_ptree () in
  let trip t =
    let running = t.running in
    let next = Nx.add t.leaf (i32 1) in
    let ids =
      Nx.concatenate ~axis:0
        [
          Nx.reshape [| 1 |] (i32 0);
          Nx.add parents (Nx.div next spans);
          Nx.reshape [| 1 |] (i32 top_code);
        ]
    in
    let us = uniforms dt keys t.doubling ids in
    let uniform k = Nx.slice [ Nx.A; Nx.I k ] us in
    let fresh = Nx.equal t.leaf (i32 0) in
    let forward =
      Nx.where fresh (Nx.greater_s (uniform 0) 0.5) t.forward
    in
    let ahead = mask forward t.front.z in
    let from = choose_point forward ahead t.front t.back in
    let cursor = pick_point fresh from t.cursor in
    let h = Nx.where forward eps (Nx.neg eps) in
    let leaf, finite = leapfrog lp_z running h cursor in
    let delta = Nx.sub (Nx.sub (kinetic leaf.p) leaf.lp) h0 in
    let delta = Nx.where (Nx.isnan delta) (Nx.scalar_like delta Float.infinity) delta in
    let diverged =
      Nx.logical_or (Nx.logical_not finite)
        (Nx.greater_s delta H.max_energy_error)
    in
    let accept = Nx.minimum_s (Nx.exp (Nx.neg delta)) 1. in
    let node =
      {
        first = leaf.p;
        last = leaf.p;
        rho = leaf.p;
        weight = Nx.neg delta;
        prop = leaf;
      }
    in
    (* The cascade: the merges this leaf completes. Level [l] joins the node
       arriving at it with its slot when that node is a second half within the
       doubling: where [2^(l + 1)] divides [i + 1] and [l < j]. The levels that
       join are a prefix [0, k), so the loop stops at level [k], where the node
       is a first half, which that level's slot keeps. *)
    let at = Nx.mod_ next spans in
    let firsts = Nx.equal at halves in
    let joins =
      Nx.logical_and
        (Nx.equal at (Nx.zeros_like at))
        (Nx.less_equal above t.doubling)
    in
    let (sub, turned), _ =
      Rune.iterate cascading ~max:max_depth
        ~until:(fun (_, l) -> Nx.logical_not (Nx.take ~axis:0 l joins))
        (fun ((node, turned), l) ->
          let slot = map_node (Nx.take ~axis:0 l) t.slots in
          let up = Nx.add_s l 1L in
          let parent, ok = merge (Nx.take ~axis:1 up us) slot node in
          let turned = Nx.logical_or turned (Nx.logical_not ok) in
          ((parent, turned), up))
        ((node, no), Nx.scalar_at on Nx.int64 0L)
    in
    let kept =
      Nx.logical_and
        (Nx.reshape [| max_depth; 1 |] firsts)
        (Nx.reshape [| 1; c |] running)
    in
    let valid =
      Nx.logical_and (Nx.logical_not diverged) (Nx.logical_not turned)
    in
    let held = mask running leaf.z in
    let keep a b = Nx.where running a b
    and keep_point a b = choose_point running held a b in
    (* The leaf that completes the doubling's subtree merges it into the
       trajectory, where the subtree is valid; any other leaf leaves the
       trajectory as it is and stops a chain whose subtree is not. *)
    let merged () =
      (* Biased progressive sampling: the subtree's proposal replaces the
         trajectory's with probability [min (1, w_sub / w_traj)]. *)
      let take =
        Nx.logical_or
          (Nx.greater sub.weight t.total)
          (Nx.less
             (uniform (max_depth + 1))
             (Nx.exp (Nx.sub sub.weight t.total)))
      in
      let sample = pick_point (Nx.logical_and valid take) sub.prop t.sample in
      let total = Nx.where valid (logaddexp t.total sub.weight) t.total in
      let sum = pick valid (Nx.add t.sum sub.rho) t.sum in
      (* The criterion around the merged trajectory and between its halves,
         the backward half first. *)
      let bb = Nx.where ahead t.back.p sub.last
      and bf = Nx.where ahead t.front.p sub.first in
      let fb = Nx.where ahead sub.first t.back.p
      and ff = Nx.where ahead sub.last t.front.p in
      let rb = Nx.where ahead t.sum sub.rho
      and rf = Nx.where ahead sub.rho t.sum in
      let back_half = Nx.add rb fb and front_half = Nx.add rf bf in
      let persists =
        forward_all
          [
            (ff, sum);
            (bb, sum);
            (fb, back_half);
            (bb, back_half);
            (ff, front_half);
            (bf, front_half);
          ]
      in
      let depth = Nx.where valid (Nx.add t.depth (i32 1)) t.depth in
      let full =
        Nx.logical_and valid (Nx.greater_equal depth (i32 max_depth))
      in
      let stop =
        Nx.logical_or (Nx.logical_not valid)
          (Nx.logical_or
             (Nx.logical_and valid (Nx.logical_not persists))
             full)
      in
      let front = pick_point (Nx.logical_and valid forward) leaf t.front in
      let back =
        pick_point (Nx.logical_and valid (Nx.logical_not forward)) leaf t.back
      in
      {
        back_end = keep_point back t.back;
        front_end = keep_point front t.front;
        chosen = keep_point sample t.sample;
        summed = Nx.where held sum t.sum;
        weighed = keep total t.total;
        deeper = keep depth t.depth;
        stop;
        saturates = Nx.logical_and full persists;
      }
    in
    let unmerged =
      {
        back_end = t.back;
        front_end = t.front;
        chosen = t.sample;
        summed = t.sum;
        weighed = t.total;
        deeper = t.depth;
        stop = Nx.logical_not valid;
        saturates = no;
      }
    in
    let e =
      Rune.cond ended (Nx.equal next t.size) merged (fun () -> unmerged)
    in
    let size = Nx.where (Nx.equal next t.size) (Nx.mul t.size (i32 2)) t.size in
    {
      back = e.back_end;
      front = e.front_end;
      cursor = keep_point leaf t.cursor;
      sample = e.chosen;
      sum = e.summed;
      total = e.weighed;
      slots = pick_node kept sub t.slots;
      forward = keep forward t.forward;
      running = Nx.logical_and running (Nx.logical_not e.stop);
      diverging = Nx.logical_or t.diverging (Nx.logical_and running diverged);
      saturated =
        Nx.logical_or t.saturated (Nx.logical_and running e.saturates);
      depth = e.deeper;
      steps = Nx.add t.steps (Nx.cast Nx.int32 running);
      accepted =
        Nx.add t.accepted (Nx.where running accept (Nx.zeros_like accept));
      doubling =
        Nx.where (Nx.equal next t.size) (Nx.add t.doubling (i32 1)) t.doubling;
      leaf = Nx.where (Nx.equal next t.size) (i32 0) next;
      size;
    }
  in
  let t =
    Rune.iterate (tree_ptree ())
      ~max:((1 lsl max_depth) - 1)
      ~until:(fun t -> Nx.logical_not (Nx.any t.running))
      trip init
  in
  let s = t.sample in
  let x, gx = H.leave u f position s in
  let stats =
    Stats.
      {
        lp = s.lp;
        acceptance = Nx.div t.accepted (Nx.cast dt t.steps);
        step_size = eps;
        n_steps = t.steps;
        diverging = t.diverging;
        saturated = t.saturated;
        energy = h0;
      }
  in
  (x, s.lp, gx, stats)

(* Starting *)

let init u ?(max_depth = 10) ?(accept = 0.8) ?(rank = 2) ?geometry lp position =
  let context = "Norn.Nuts.init" in
  if max_depth < 1 then
    invalid_argf "%s: max_depth = %d is not positive" context max_depth;
  if not (accept > 0. && accept < 1.) then
    invalid_argf "%s: accept = %s is not in (0, 1)" context
      (Rows.number accept);
  if rank < 0 then invalid_argf "%s: rank = %d is negative" context rank;
  let c = Rows.count context u position in
  let l, g = Rows.evaluate u lp position in
  Rows.check_density context u lp position l;
  let dt = Nx.dtype l and on = Nx.beside l in
  let gp = Gaussian.ptree u in
  (* Orthonormal directions number at most a chain's float elements. *)
  let rank = min rank (Rows.elements u c position) in
  let geometry =
    match geometry with
    | Some g ->
        P.map gp
          (fun _ t ->
            Nx.copy (Nx.broadcast_to (Array.append [| c |] (Nx.shape t)) t))
          g
    | None ->
        let scale =
          P.map u (fun _ g -> Nx.sqrt (Adapt.clip (Nx.recip (Nx.abs g)))) g
        in
        let fit =
          if rank = 0 then fun m s -> Gaussian.diagonal u dt ~mean:m ~scale:s
          else fun m s ->
            let directions =
              P.map u
                (fun _ t ->
                  Nx.zeros_at (Nx.beside t) (Nx.dtype t)
                    (Array.append [| rank |] (Nx.shape t)))
                m
            in
            Gaussian.low_rank u ~mean:m ~scale:s ~directions
              ~variances:(Nx.ones_at on dt [| rank |])
        in
        Rune.vmap P.(u @-> u @-> returns gp) fit position scale
  in
  let zero = Nx.zeros_at on dt [| c |] and no = Nx.zeros_at on Nx.bool [| c |] in
  {
    position;
    lp = l;
    grad = g;
    step_size = Nx.ones_at on dt [| c |];
    geometry;
    stats =
      Stats.
        {
          lp = l;
          acceptance = zero;
          step_size = Nx.ones_at on dt [| c |];
          n_steps = Nx.zeros_at on Nx.int32 [| c |];
          diverging = no;
          saturated = no;
          energy = zero;
        };
    draw = Nx.scalar_at on Nx.int32 0l;
    accept = Nx.scalar_at on dt accept;
    max_depth;
  }

(* Transitions *)

let step u lp k (s : (_, _, 'd) state) =
  let k = Nx.Rng.place (Nx.beside s.lp) k in
  let c = (Nx.shape s.lp).(0) in
  let keys = Nx.Rng.split_batch ~n:c k in
  let position, lp, grad, stats =
    transition u lp s.max_depth s.step_size keys s.geometry s.position s.lp
      s.grad
  in
  {
    s with
    position;
    lp;
    grad;
    stats;
    draw = Nx.add_s s.draw 1l;
  }

(* [leading_swap x] moves the chain axis before the draw axis. *)
let leading_swap x = Nx.moveaxis 0 1 x

let sample u lp k ~draws (s : (_, _, 'd) state) =
  let k = Nx.Rng.place (Nx.beside s.lp) k in
  if draws < 1 then
    invalid_argf "Norn.Nuts.sample: draws = %d is not positive" draws;
  let dt = Nx.dtype s.lp in
  let sp = ptree u and st = Stats.ptree dt in
  let s, (xs, ss) =
    Rune.scan sp P.tensor
      P.(pair u st)
      (fun (s : (_, _, 'd) state) _ ->
        let s = step u lp (Nx.Rng.fold_in_tensor k s.draw) s in
        (s, (s.position, s.stats)))
      ~init:s
      (Nx.zeros_at (Nx.beside s.lp) Nx.int32 [| draws |])
  in
  let xs = P.map u (fun _ t -> leading_swap t) xs in
  let ss = P.map st (fun _ t -> leading_swap t) ss in
  (s, Draws.v u xs, Draws.v st ss)

(* Warmup *)

type ('u, 'f, 'd) adapt = {
  st : ('u, 'f, 'd) state;
  averaging : ('f, 'd) Adapt.averaging;
  window : ('u, 'd) Adapt.window;
}


let adapt_ptree (type u f) (u : (u, 'd) P.t) : ((u, f, 'd) adapt, 'd) P.t =
  let sp = ptree u in
  let walk c (a : (u, f, 'd) adapt) : (u, f, 'd) adapt =
    let open P.Walk in
    let st = field c "st" (structure sp) a.st in
    let averaging =
      field c "averaging" (structure (Adapt.averaging_ptree ())) a.averaging
    in
    let window =
      field c "window" (structure (Adapt.window_ptree u)) a.window
    in
    { st; averaging; window }
  in
  P.v walk

(* [init_step_size u lp k s] is each chain's step size doubled or halved, from
   [s.step_size], until one leapfrog step from its position with a fresh
   momentum crosses an acceptance of 0.8. [k] is the key of the transition the
   search precedes. *)
let init_step_size u lp k (s : (_, _, 'd) state) =
  let c = (Nx.shape s.step_size).(0) in
  let f = Gaussian.flat u s.position s.geometry in
  H.search "Norn.Nuts.warmup" P.tensor
    (H.rows_density u lp f s.position)
    ~reduce:Fun.id
    (Nx.Rng.split_batch ~n:c k)
    (H.enter u f s.position s.lp s.grad)
    s.step_size

let warmup u lp k ~steps (s : (_, _, 'd) state) =
  let k = Nx.Rng.place (Nx.beside s.lp) k in
  if steps < 0 then
    invalid_argf "Norn.Nuts.warmup: steps = %d is negative" steps;
  let schedule = Adapt.schedule steps in
  if schedule = [] then s
  else
    let dt = Nx.dtype s.lp in
    let rank = Gaussian.rank s.geometry in
    let longest = List.fold_left (fun m (n, _) -> max m n) 0 schedule in
    (* A low-rank fit reads the window's draws; a diagonal one their sums. *)
    let buffer = if rank = 0 then 0 else longest in
    let key (st : (_, _, 'd) state) = Nx.Rng.fold_in_tensor k st.draw in
    let s0 = { s with step_size = init_step_size u lp (key s) s } in
    let a0 =
      {
        st = s0;
        averaging = Adapt.restart s0.step_size;
        window = Adapt.empty u ~buffer s0.position;
      }
    in
    let one (a : (_, _, 'd) adapt) =
      let st = step u lp (key a.st) a.st in
      let stat =
        Nx.minimum st.stats.acceptance (Nx.ones_like st.stats.acceptance)
      in
      let averaging, step_size =
        Adapt.average a.averaging ~target:st.accept stat
      in
      let st = { st with step_size } in
      { st; averaging; window = Adapt.record u a.window st.position st.grad }
    in
    let window (a : (_, _, 'd) adapt) (length, refit) =
      let a = { a with window = Adapt.reopen u a.window a.st.position } in
      let a =
        Rune.iterate (adapt_ptree u) ~max:longest
          ~until:(fun a -> Nx.greater_equal a.window.n length)
          one a
      in
      (* At a slow window's end: refit the geometry, find a step size for it and
         restart the averaging there. *)
      let refit = Nx.not_equal_s refit 0l in
      let geometry = Adapt.per_chain u dt ~rank a.window in
      let gp = Gaussian.ptree u in
      let refitted =
        {
          a.st with
          geometry =
            P.map2 gp (fun _ g o -> Nx.where refit g o) geometry a.st.geometry;
        }
      in
      let eps = init_step_size u lp (key refitted) refitted in
      let eps = Nx.where refit eps a.st.step_size in
      let st = { refitted with step_size = eps } in
      let restarted = Adapt.restart st.step_size in
      let averaging =
        P.map2 (Adapt.averaging_ptree ())
          (fun _ r o -> Nx.where refit r o)
          restarted a.averaging
      in
      ({ a with st; averaging }, ())
    in
    let a, () =
      Rune.scan (adapt_ptree u)
        P.(pair tensor tensor)
        P.unit window ~init:a0
        (Adapt.windows (Nx.beside s0.step_size) schedule)
    in
    { a.st with step_size = Adapt.final a.averaging }
