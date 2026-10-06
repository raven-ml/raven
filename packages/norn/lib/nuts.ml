(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree
module H = Hamiltonian

let invalid_argf = Rows.invalid_argf

type ('u, 'f) state = {
  position : 'u;
  lp : (float, 'f) Nx.t;
  grad : 'u;
  step_size : (float, 'f) Nx.t;
  geometry : ('u, 'f) Gaussian.t;
  stats : 'f Stats.t;
  draw : Nx.int32_t;
  accept : (float, 'f) Nx.t;
  max_depth : int;
}

type ('u, 'f) nuts = ('u, 'f) state

let ptree (type u f) (u : u P.t) : (u, f) state P.t =
  let g = Gaussian.ptree u in
  let module S = struct
    type _ t = (u, f) nuts

    let walk c s =
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
  end in
  P.nest (module S) P.unit

let stats = Stats.ptree

(* [logaddexp a b] is [log (exp a + exp b)], [-inf] where both are. *)
let logaddexp a b =
  let m = Nx.maximum a b in
  let ninf = Nx.scalar_like m Float.neg_infinity in
  let s =
    Nx.add m (Nx.log (Nx.add (Nx.exp (Nx.sub a m)) (Nx.exp (Nx.sub b m))))
  in
  Nx.where (Nx.equal m ninf) ninf s

(* Geometry per chain *)

let per_chain_color u g z =
  Rune.vmap P.(Gaussian.ptree u @-> u @-> returns u) (Geometry.color u) g z

let per_chain_whiten u g x =
  Rune.vmap P.(Gaussian.ptree u @-> u @-> returns u) (Geometry.whiten u) g x

(* [to_whitened u g x gx] is the gradient in whitened coordinates of a gradient
   [gx] at [x]: [color]'s transpose applied to it. *)
let to_whitened u g z gx =
  Rune.vmap
    P.(Gaussian.ptree u @-> u @-> u @-> returns u)
    (fun g z gx -> snd (Rune.vjp u u (Geometry.color u g) z) gx)
    g z gx

let to_original u g x gz =
  Rune.vmap
    P.(Gaussian.ptree u @-> u @-> u @-> returns u)
    (fun g x gz -> snd (Rune.vjp u u (Geometry.whiten u g) x) gz)
    g x gz

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

let uniforms (type f) dt keys j id : (float, f) Nx.t =
  Rune.vmap
    P.(Nx.Rng.ptree @-> returns tensor)
    (fun k ->
      let k =
        Nx.Rng.fold_in_tensor (Nx.Rng.fold_in_tensor (Nx.Rng.fold_in k 1) j) id
      in
      Nx.Rng.uniform k dt [||])
    keys

(* Trees

   A trajectory is built by doublings: doubling [j] extends it by a subtree of
   [2^j] leapfrog steps in a random direction. The subtree is built leaf by
   leaf; the leaf of index [i] completes the node of level [l] that ends at it
   whenever [2^l] divides [i + 1]. A completed node that is the first half of
   its parent waits in its level's slot; one that is the second half merges with
   the slot into its parent. Every chain takes the same leaf schedule, so the
   leaf counters are shared and only each chain's stop differs. *)

type ('u, 'f) point = ('u, 'f) H.point = {
  z : 'u;
  p : 'u;
  g : 'u;
  lp : (float, 'f) Nx.t;
}

(* A completed subtree: the momenta of its first and last leaf, in the order it
   was built, their sum, its log weight and its proposal. *)
type ('u, 'f) node = {
  first : 'u;
  last : 'u;
  rho : 'u;
  weight : (float, 'f) Nx.t;
  prop : ('u, 'f) point;
}

type ('u, 'f) tree = {
  back : ('u, 'f) point; (* the trajectory's backward end *)
  front : ('u, 'f) point; (* its forward end *)
  cursor : ('u, 'f) point; (* the last leaf *)
  sample : ('u, 'f) point;
  sum : 'u; (* the trajectory's momenta, summed *)
  total : (float, 'f) Nx.t; (* its log weight *)
  slots : ('u, 'f) node list; (* one per level *)
  forward : Nx.bool_t;
  running : Nx.bool_t;
  diverging : Nx.bool_t;
  saturated : Nx.bool_t;
  depth : Nx.int32_t;
  steps : Nx.int32_t;
  accepted : (float, 'f) Nx.t; (* the sum of the leaves' acceptance *)
  doubling : Nx.int32_t; (* [j], shared by the chains *)
  leaf : Nx.int32_t; (* [i] *)
  size : Nx.int32_t; (* [2^j] *)
}

type ('u, 'f) node' = ('u, 'f) node
type ('u, 'f) tree' = ('u, 'f) tree

let node_ptree (type u f) (u : u P.t) : (u, f) node P.t =
  let point = H.point_ptree u in
  let module S = struct
    type _ t = (u, f) node'

    let walk c (s : (u, f) node) : (u, f) node =
      let open P.Walk in
      let first = field c "first" (structure u) s.first in
      let last = field c "last" (structure u) s.last in
      let rho = field c "rho" (structure u) s.rho in
      let weight = field c "weight" tensor s.weight in
      let prop = field c "prop" (structure point) s.prop in
      { first; last; rho; weight; prop }
  end in
  P.nest (module S) P.unit

let tree_ptree (type u f) (u : u P.t) : (u, f) tree P.t =
  let point = H.point_ptree u and nodes = P.list (node_ptree u) in
  let module S = struct
    type _ t = (u, f) tree'

    let walk c (s : (u, f) tree) : (u, f) tree =
      let open P.Walk in
      let back = field c "back" (structure point) s.back in
      let front = field c "front" (structure point) s.front in
      let cursor = field c "cursor" (structure point) s.cursor in
      let sample = field c "sample" (structure point) s.sample in
      let sum = field c "sum" (structure u) s.sum in
      let total = field c "total" tensor s.total in
      let slots = field c "slots" (structure nodes) s.slots in
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
  end in
  P.nest (module S) P.unit

let choose_node u mask a b =
  {
    first = Rows.choose u mask a.first b.first;
    last = Rows.choose u mask a.last b.last;
    rho = Rows.choose u mask a.rho b.rho;
    weight = Nx.where mask a.weight b.weight;
    prop = H.choose_point u mask a.prop b.prop;
  }

(* [criterion u like minus plus rho] is the generalised no-U-turn criterion of a
   stretch of trajectory whose end momenta are [minus] and [plus] and whose
   momenta sum to [rho]: both ends still move along [rho]. With unit metric the
   sharp momentum is the momentum. *)
let criterion u like minus plus rho =
  let zero = Nx.zeros_like like in
  Nx.logical_and
    (Nx.greater (Rows.dot u like plus rho) zero)
    (Nx.greater (Rows.dot u like minus rho) zero)

(* [merge u like keys code init final] is the parent of the sibling nodes [init]
   and [final], built in that order, and whether it keeps the criterion around
   itself and between its halves. *)
let merge u like keys doubling code init final =
  let weight = logaddexp init.weight final.weight in
  let take =
    Nx.less
      (uniforms (Nx.dtype like) keys doubling code)
      (Nx.exp (Nx.sub final.weight weight))
  in
  let rho = Rows.add u init.rho final.rho in
  let ok =
    Nx.logical_and
      (criterion u like init.first final.last rho)
      (Nx.logical_and
         (criterion u like init.first final.first
            (Rows.add u init.rho final.first))
         (criterion u like init.last final.last
            (Rows.add u final.rho init.last)))
  in
  ( {
      first = init.first;
      last = final.last;
      rho;
      weight;
      prop = H.choose_point u take final.prop init.prop;
    },
    ok )

let leapfrog u lp_z running h s = H.leapfrog "Norn.Nuts.step" u lp_z running h s

(* [transition u lp max_depth eps keys geometry s] is one transition from the
   position [s] of each chain. *)
let transition (type f) u lp max_depth (eps : (float, f) Nx.t) keys geometry
    position lp0 grad =
  let dt = Nx.dtype eps in
  let c = (Nx.shape eps).(0) in
  let i32 v = Nx.scalar Nx.int32 (Int32.of_int v) in
  let color z = per_chain_color u geometry z in
  let lp_z z = lp (color z) in
  let z0 = per_chain_whiten u geometry position in
  let g0 = to_whitened u geometry z0 grad in
  let p0 = H.momentum u keys z0 in
  let kinetic p = H.kinetic u eps p in
  let h0 = Nx.sub (kinetic p0) lp0 in
  let start = { z = z0; p = p0; g = g0; lp = lp0 } in
  let empty = { first = p0; last = p0; rho = p0; weight = lp0; prop = start } in
  let no = Nx.zeros Nx.bool [| c |] in
  let init =
    {
      back = start;
      front = start;
      cursor = start;
      sample = start;
      sum = p0;
      total = Nx.zeros dt [| c |];
      slots = List.init max_depth (fun _ -> empty);
      forward = no;
      running = Nx.ones Nx.bool [| c |];
      diverging = no;
      saturated = no;
      depth = Nx.zeros Nx.int32 [| c |];
      steps = Nx.zeros Nx.int32 [| c |];
      accepted = Nx.zeros dt [| c |];
      doubling = i32 0;
      leaf = i32 0;
      size = i32 1;
    }
  in
  let top_code = (max_depth + 1) lsl max_depth in
  let trip t =
    let running = t.running in
    let fresh = Nx.equal t.leaf (i32 0) in
    let forward =
      Nx.where fresh
        (Nx.greater (uniforms dt keys t.doubling (i32 0)) (Nx.scalar dt 0.5))
        t.forward
    in
    let from = H.choose_point u forward t.front t.back in
    let cursor =
      H.choose_point u (Nx.broadcast_to [| c |] fresh) from t.cursor
    in
    let h = Nx.where forward eps (Nx.neg eps) in
    let leaf, finite = leapfrog u lp_z running h cursor in
    let delta = Nx.sub (Nx.sub (kinetic leaf.p) leaf.lp) h0 in
    let delta = Nx.where (Nx.isnan delta) (Nx.scalar dt Float.infinity) delta in
    let diverged =
      Nx.logical_or (Nx.logical_not finite)
        (Nx.greater delta (Nx.scalar dt H.max_energy_error))
    in
    let accept = Nx.minimum (Nx.exp (Nx.neg delta)) (Nx.scalar dt 1.) in
    let node =
      {
        first = leaf.p;
        last = leaf.p;
        rho = leaf.p;
        weight = Nx.neg delta;
        prop = leaf;
      }
    in
    (* The cascade: the node of each level that this leaf completes. *)
    let next = Nx.add t.leaf (i32 1) in
    let turned = ref no in
    let rec cascade l node slots acc =
      match slots with
      | [] -> (node, List.rev acc)
      | slot :: rest ->
          let span = 1 lsl (l + 1) and half = 1 lsl l in
          let at = Nx.mod_ next (i32 span) in
          let is_first = Nx.equal at (i32 half) in
          let joins =
            Nx.logical_and
              (Nx.equal at (i32 0))
              (Nx.less_equal (i32 (l + 1)) t.doubling)
          in
          let slot' =
            choose_node u (Nx.broadcast_to [| c |] is_first) node slot
          in
          let code =
            Nx.add (i32 ((l + 1) lsl max_depth)) (Nx.div next (i32 span))
          in
          let parent, ok = merge u eps keys t.doubling code slot node in
          let joins = Nx.broadcast_to [| c |] joins in
          turned :=
            Nx.logical_or !turned (Nx.logical_and joins (Nx.logical_not ok));
          cascade (l + 1) (choose_node u joins parent node) rest (slot' :: acc)
    in
    let sub, slots = cascade 0 node t.slots [] in
    let complete = Nx.broadcast_to [| c |] (Nx.equal next t.size) in
    let valid =
      Nx.logical_and (Nx.logical_not diverged) (Nx.logical_not !turned)
    in
    let merges = Nx.logical_and complete valid in
    (* Biased progressive sampling: the subtree's proposal replaces the
       trajectory's with probability [min (1, w_sub / w_traj)]. *)
    let u_top = uniforms dt keys t.doubling (i32 top_code) in
    let take =
      Nx.logical_or
        (Nx.greater sub.weight t.total)
        (Nx.less u_top (Nx.exp (Nx.sub sub.weight t.total)))
    in
    let sample =
      H.choose_point u (Nx.logical_and merges take) sub.prop t.sample
    in
    let total = Nx.where merges (logaddexp t.total sub.weight) t.total in
    let sum = Rows.choose u merges (Rows.add u t.sum sub.rho) t.sum in
    (* The criterion around the merged trajectory and between its halves, the
       backward half first. *)
    let bb = Rows.choose u forward t.back.p sub.last
    and bf = Rows.choose u forward t.front.p sub.first in
    let fb = Rows.choose u forward sub.first t.back.p
    and ff = Rows.choose u forward sub.last t.front.p in
    let rb = Rows.choose u forward t.sum sub.rho
    and rf = Rows.choose u forward sub.rho t.sum in
    let persists =
      Nx.logical_and
        (criterion u eps bb ff sum)
        (Nx.logical_and
           (criterion u eps bb fb (Rows.add u rb fb))
           (criterion u eps bf ff (Rows.add u rf bf)))
    in
    let depth = Nx.where merges (Nx.add t.depth (i32 1)) t.depth in
    let full = Nx.logical_and merges (Nx.greater_equal depth (i32 max_depth)) in
    let stop =
      Nx.logical_or (Nx.logical_not valid)
        (Nx.logical_or (Nx.logical_and merges (Nx.logical_not persists)) full)
    in
    let front = H.choose_point u (Nx.logical_and merges forward) leaf t.front in
    let back =
      H.choose_point u
        (Nx.logical_and merges (Nx.logical_not forward))
        leaf t.back
    in
    let keep a b = Rows.choose u running a b
    and keep_point a b = H.choose_point u running a b in
    let keep_t a b = Nx.where running a b in
    let size = Nx.where (Nx.equal next t.size) (Nx.mul t.size (i32 2)) t.size in
    {
      back = keep_point back t.back;
      front = keep_point front t.front;
      cursor = keep_point leaf t.cursor;
      sample = keep_point sample t.sample;
      sum = keep sum t.sum;
      total = keep_t total t.total;
      slots = List.map2 (fun s o -> choose_node u running s o) slots t.slots;
      forward = keep_t forward t.forward;
      running = Nx.logical_and running (Nx.logical_not stop);
      diverging = Nx.logical_or t.diverging (Nx.logical_and running diverged);
      saturated =
        Nx.logical_or t.saturated
          (Nx.logical_and running (Nx.logical_and full persists));
      depth = keep_t depth t.depth;
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
    Rune.iterate (tree_ptree u)
      ~max:((1 lsl max_depth) - 1)
      ~until:(fun t -> Nx.logical_not (Nx.any t.running))
      ~f:trip init
  in
  let s = t.sample in
  let x = color s.z in
  let gx = to_original u geometry x s.g in
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
    invalid_argf "%s: accept = %g is not in (0, 1)" context accept;
  if rank < 0 then invalid_argf "%s: rank = %d is negative" context rank;
  let c = Rows.count context u position in
  let l, g = Rows.evaluate u lp position in
  Rows.check_density context u lp position l;
  let dt = Nx.dtype l in
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
                  Nx.zeros (Nx.dtype t) (Array.append [| rank |] (Nx.shape t)))
                m
            in
            Gaussian.low_rank u ~mean:m ~scale:s ~directions
              ~variances:(Nx.ones dt [| rank |])
        in
        Rune.vmap P.(u @-> u @-> returns gp) fit position scale
  in
  let zero = Nx.zeros dt [| c |] and no = Nx.zeros Nx.bool [| c |] in
  {
    position;
    lp = l;
    grad = g;
    step_size = Nx.ones dt [| c |];
    geometry;
    stats =
      Stats.
        {
          lp = l;
          acceptance = zero;
          step_size = Nx.ones dt [| c |];
          n_steps = Nx.zeros Nx.int32 [| c |];
          diverging = no;
          saturated = no;
          energy = zero;
        };
    draw = Nx.scalar Nx.int32 0l;
    accept = Nx.scalar dt accept;
    max_depth;
  }

(* Transitions *)

let step u lp k (s : (_, _) state) =
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
    draw = Nx.add s.draw (Nx.scalar Nx.int32 1l);
  }

(* [leading_swap x] moves the chain axis before the draw axis. *)
let leading_swap x = Nx.moveaxis 0 1 x

let sample u lp k ~draws (s : (_, _) state) =
  if draws < 1 then
    invalid_argf "Norn.Nuts.sample: draws = %d is not positive" draws;
  let dt = Nx.dtype s.lp in
  let sp = ptree u and st = Stats.ptree dt in
  let s, (xs, ss) =
    Rune.scan sp P.tensor
      P.(pair u st)
      ~f:(fun (s : (_, _) state) _ ->
        let s = step u lp (Nx.Rng.fold_in_tensor k s.draw) s in
        (s, (s.position, s.stats)))
      ~init:s
      (Nx.zeros Nx.int32 [| draws |])
  in
  let xs = P.map u (fun _ t -> leading_swap t) xs in
  let ss = P.map st (fun _ t -> leading_swap t) ss in
  (s, Draws.v u xs, Draws.v st ss)

(* Warmup *)

type ('u, 'f) adapt = {
  st : ('u, 'f) state;
  averaging : 'f Adapt.averaging;
  window : 'u Adapt.window;
}

type ('u, 'f) adapt' = ('u, 'f) adapt

let adapt_ptree (type u f) (u : u P.t) : (u, f) adapt P.t =
  let sp = ptree u in
  let module S = struct
    type _ t = (u, f) adapt'

    let walk c (a : (u, f) adapt) : (u, f) adapt =
      let open P.Walk in
      let st = field c "st" (structure sp) a.st in
      let averaging =
        field c "averaging" (structure (Adapt.averaging_ptree ())) a.averaging
      in
      let window =
        field c "window" (structure (Adapt.window_ptree u)) a.window
      in
      { st; averaging; window }
  end in
  P.nest (module S) P.unit

(* [init_step_size u lp k s] is each chain's step size doubled or halved, from
   [s.step_size], until one leapfrog step from its position with a fresh
   momentum crosses an acceptance of 0.8. [k] is the key of the transition the
   search precedes. *)
let init_step_size u lp k (s : (_, _) state) =
  let c = (Nx.shape s.step_size).(0) in
  let lp_z z = lp (per_chain_color u s.geometry z) in
  let z0 = per_chain_whiten u s.geometry s.position in
  let g0 = to_whitened u s.geometry z0 s.grad in
  let start = { z = z0; p = z0; g = g0; lp = s.lp } in
  H.search "Norn.Nuts.warmup" u lp_z ~reduce:Fun.id
    (Nx.Rng.split_batch ~n:c k)
    start s.step_size

let warmup u lp k ~steps (s : (_, _) state) =
  if steps < 0 then
    invalid_argf "Norn.Nuts.warmup: steps = %d is negative" steps;
  let schedule = Adapt.schedule steps in
  if schedule = [] then s
  else
    let dt = Nx.dtype s.lp in
    let rank = Geometry.rank s.geometry in
    let longest = List.fold_left (fun m (n, _) -> max m n) 0 schedule in
    (* A low-rank fit reads the window's draws; a diagonal one their sums. *)
    let buffer = if rank = 0 then 0 else longest in
    let key (st : (_, _) state) = Nx.Rng.fold_in_tensor k st.draw in
    let s0 = { s with step_size = init_step_size u lp (key s) s } in
    let a0 =
      {
        st = s0;
        averaging = Adapt.restart s0.step_size;
        window = Adapt.empty u ~buffer s0.position;
      }
    in
    let one (a : (_, _) adapt) =
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
    let window (a : (_, _) adapt) (length, refit) =
      let a = { a with window = Adapt.reopen u a.window a.st.position } in
      let a =
        Rune.iterate (adapt_ptree u) ~max:longest
          ~until:(fun a -> Nx.greater_equal a.window.n length)
          ~f:one a
      in
      (* At a slow window's end: refit the geometry, find a step size for it and
         restart the averaging there. *)
      let refit = Nx.not_equal refit (Nx.scalar Nx.int32 0l) in
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
        P.unit ~f:window ~init:a0 (Adapt.windows schedule)
    in
    { a.st with step_size = Adapt.final a.averaging }
