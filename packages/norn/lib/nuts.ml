(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

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

(* Rows

   Positions lead with the chain axis. A row operation acts on each chain:
   [rows_dot] is each chain's inner product over every float tensor, and a
   per-chain scalar broadcasts against a tensor's trailing axes. *)

let float_leaf x = Nx_dtype.is_float (Nx.dtype x)

let column (type f) (h : (float, f) Nx.t) x =
  let c = (Nx.shape h).(0) in
  Nx.reshape
    (Array.append [| c |] (Array.make (Nx.ndim x - 1) 1))
    (Nx.cast (Nx.dtype x) h)

let rows_dot (type f) u (like : (float, f) Nx.t) a b : (float, f) Nx.t =
  let c = (Nx.shape like).(0) in
  P.fold u
    (fun _ t acc ->
      if not (float_leaf t) then acc
      else
        Nx.add acc
          (Nx.cast (Nx.dtype like)
             (Nx.sum ~axes:[ 1 ] (Nx.reshape [| c; -1 |] t))))
    (P.map2 u (fun _ x y -> Nx.mul x y) a b)
    (Nx.zeros_like like)

(* [axpy u h x y] is [y + h x], [h] one scalar per chain. *)
let axpy u h x y =
  P.map2 u
    (fun _ x y -> if float_leaf x then Nx.add y (Nx.mul (column h x) x) else y)
    x y

(* [choose u mask a b] is [a] in the chains where [mask] holds, [b]
   elsewhere. *)
let choose u mask a b =
  let c = (Nx.shape mask).(0) in
  P.map2 u
    (fun _ x y ->
      let m =
        Nx.reshape (Array.append [| c |] (Array.make (Nx.ndim x - 1) 1)) mask
      in
      Nx.where m x y)
    a b

let finite_rows u x =
  let c = P.fold u (fun _ t _ -> (Nx.shape t).(0)) x 0 in
  P.fold u
    (fun _ t acc ->
      if not (float_leaf t) then acc
      else
        Nx.logical_and acc
          (Nx.all ~axes:[ 1 ] (Nx.reshape [| c; -1 |] (Nx.isfinite t))))
    x (Nx.ones Nx.bool [| c |])

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

(* [evaluate u lp x] is the density at [x] and its gradient, chain by chain: the
   gradient of the summed density is each chain's, rows being independent. *)
let evaluate u lp x =
  let _, g, l =
    Rune.value_and_grad_aux u P.tensor
      (fun x ->
        let l = lp x in
        (Nx.sum l, l))
      x
  in
  (l, g)

(* The density's value is finite or -inf at a finite position. *)
let check_range context l =
  let c = (Nx.shape l).(0) in
  let chain = Nx.arange Nx.int32 0 c 1 in
  let ok =
    Nx.logical_not
      (Nx.logical_or (Nx.isnan l)
         (Nx.equal l (Nx.scalar_like l Float.infinity)))
  in
  Nx.check
    P.(pair tensor tensor)
    ok (l, chain)
    (fun _ (v, chain) ->
      Invalid_argument
        (Printf.sprintf "%s: the density is %s at chain %ld, a finite position"
           context (Nx.to_string v) (Nx.item [] chain)))

(* Keys

   Chain [i] has row [i] of [split_batch] of the transition's key. Its momentum
   draws from [fold_in key 0]; doubling [j] draws its direction, the uniforms of
   its merges and of its top-level merge from [fold_in (fold_in (fold_in key 1)
   j) id], with [id] [0] for the direction, [l 2^D + n] for the merge into node
   [n] of level [l], and [(D + 1) 2^D] for the top-level merge, [D] the maximum
   depth. *)

let uniforms (type f) dt keys j id : (float, f) Nx.t =
  Rune.vmap
    P.(Nx.Rng.ptree @-> returns tensor)
    (fun k ->
      let k =
        Nx.Rng.fold_in_tensor (Nx.Rng.fold_in_tensor (Nx.Rng.fold_in k 1) j) id
      in
      Nx.Rng.uniform k dt [||])
    keys

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

(* Trees

   A trajectory is built by doublings: doubling [j] extends it by a subtree of
   [2^j] leapfrog steps in a random direction. The subtree is built leaf by
   leaf; the leaf of index [i] completes the node of level [l] that ends at it
   whenever [2^l] divides [i + 1]. A completed node that is the first half of
   its parent waits in its level's slot; one that is the second half merges with
   the slot into its parent. Every chain takes the same leaf schedule, so the
   leaf counters are shared and only each chain's stop differs. *)

type ('u, 'f) point = { z : 'u; p : 'u; g : 'u; lp : (float, 'f) Nx.t }

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

type ('u, 'f) point' = ('u, 'f) point
type ('u, 'f) node' = ('u, 'f) node
type ('u, 'f) tree' = ('u, 'f) tree

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

let node_ptree (type u f) (u : u P.t) : (u, f) node P.t =
  let point = point_ptree u in
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
  let point = point_ptree u and nodes = P.list (node_ptree u) in
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

let choose_point u mask a b =
  {
    z = choose u mask a.z b.z;
    p = choose u mask a.p b.p;
    g = choose u mask a.g b.g;
    lp = Nx.where mask a.lp b.lp;
  }

let choose_node u mask a b =
  {
    first = choose u mask a.first b.first;
    last = choose u mask a.last b.last;
    rho = choose u mask a.rho b.rho;
    weight = Nx.where mask a.weight b.weight;
    prop = choose_point u mask a.prop b.prop;
  }

(* [criterion u like minus plus rho] is the generalised no-U-turn criterion of a
   stretch of trajectory whose end momenta are [minus] and [plus] and whose
   momenta sum to [rho]: both ends still move along [rho]. With unit metric the
   sharp momentum is the momentum. *)
let criterion u like minus plus rho =
  let zero = Nx.zeros_like like in
  Nx.logical_and
    (Nx.greater (rows_dot u like plus rho) zero)
    (Nx.greater (rows_dot u like minus rho) zero)

let add u a b = P.map2 u (fun _ x y -> Nx.add x y) a b

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
  let rho = add u init.rho final.rho in
  let ok =
    Nx.logical_and
      (criterion u like init.first final.last rho)
      (Nx.logical_and
         (criterion u like init.first final.first (add u init.rho final.first))
         (criterion u like init.last final.last (add u final.rho init.last)))
  in
  ( {
      first = init.first;
      last = final.last;
      rho;
      weight;
      prop = choose_point u take final.prop init.prop;
    },
    ok )

let max_energy_error = 1000.

(* [leapfrog u lp_z running h s] is one leapfrog step of [h], one per chain,
   from [s] with its cached gradient, by one density evaluation, and whether
   each chain's step stayed finite. A chain held by [running], or whose step
   left the reals, evaluates the density again at [s]. *)
let leapfrog u lp_z running h s =
  let finite = ref running in
  let kick h s = { s with p = axpy u h s.g s.p } in
  let drift h p =
    let z = axpy u h p.p p.z in
    let ok = finite_rows u z in
    finite := ok;
    let z = choose u (Nx.logical_and running ok) z s.z in
    let lp, g = evaluate u lp_z z in
    check_range "Norn.Nuts.step" lp;
    { z; p = p.p; g; lp }
  in
  let s' = Jera.Split.step Jera.Split.leapfrog ~kick ~drift h s in
  (s', !finite)

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
  let p0 = momentum u keys z0 in
  let kinetic p = Nx.mul_s (rows_dot u eps p p) 0.5 in
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
    let from = choose_point u forward t.front t.back in
    let cursor = choose_point u (Nx.broadcast_to [| c |] fresh) from t.cursor in
    let h = Nx.where forward eps (Nx.neg eps) in
    let leaf, finite = leapfrog u lp_z running h cursor in
    let delta = Nx.sub (Nx.sub (kinetic leaf.p) leaf.lp) h0 in
    let delta = Nx.where (Nx.isnan delta) (Nx.scalar dt Float.infinity) delta in
    let diverged =
      Nx.logical_or (Nx.logical_not finite)
        (Nx.greater delta (Nx.scalar dt max_energy_error))
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
      choose_point u (Nx.logical_and merges take) sub.prop t.sample
    in
    let total = Nx.where merges (logaddexp t.total sub.weight) t.total in
    let sum = choose u merges (add u t.sum sub.rho) t.sum in
    (* The criterion around the merged trajectory and between its halves, the
       backward half first. *)
    let bb = choose u forward t.back.p sub.last
    and bf = choose u forward t.front.p sub.first in
    let fb = choose u forward sub.first t.back.p
    and ff = choose u forward sub.last t.front.p in
    let rb = choose u forward t.sum sub.rho
    and rf = choose u forward sub.rho t.sum in
    let persists =
      Nx.logical_and
        (criterion u eps bb ff sum)
        (Nx.logical_and
           (criterion u eps bb fb (add u rb fb))
           (criterion u eps bf ff (add u rf bf)))
    in
    let depth = Nx.where merges (Nx.add t.depth (i32 1)) t.depth in
    let full = Nx.logical_and merges (Nx.greater_equal depth (i32 max_depth)) in
    let stop =
      Nx.logical_or (Nx.logical_not valid)
        (Nx.logical_or (Nx.logical_and merges (Nx.logical_not persists)) full)
    in
    let front = choose_point u (Nx.logical_and merges forward) leaf t.front in
    let back =
      choose_point u
        (Nx.logical_and merges (Nx.logical_not forward))
        leaf t.back
    in
    let keep a b = choose u running a b
    and keep_point a b = choose_point u running a b in
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

let chains u x =
  let c =
    P.fold u
      (fun _ t c -> match c with None -> Some (Nx.shape t).(0) | c -> c)
      x None
  in
  match c with
  | Some c -> c
  | None -> invalid_arg "Norn.Nuts: the position has no tensor"

(* [check_rows context u lp x l] refuses a density whose rows read each other:
   evaluated on the chains reversed, its result is not reversed. *)
let check_rows context u lp x l =
  let c = (Nx.shape l).(0) in
  if c > 1 then begin
    let flip = P.map u (fun _ t -> Nx.flip ~axes:[ 0 ] t) x in
    let r = Nx.flip ~axes:[ 0 ] (lp flip) in
    let same =
      Nx.logical_or (Nx.equal l r)
        (Nx.less_equal
           (Nx.abs (Nx.sub l r))
           (Nx.mul_s (Nx.add_s (Nx.abs l) 1.) 1e-5))
    in
    Nx.check P.tensor same (Nx.arange Nx.int32 0 c 1) (fun _ row ->
        Invalid_argument
          (Printf.sprintf
             "%s: the density's row %ld changes when the chains are reversed; \
              a chain's log density reads only its own row"
             context (Nx.item [] row)))
  end

(* The variance limits of a Fisher fit. *)
let variance_low = 1e-20
let variance_high = 1e20

let init u ?(max_depth = 10) ?(accept = 0.8) ?(rank = 0) ?geometry lp position =
  let context = "Norn.Nuts.init" in
  if max_depth < 1 then
    invalid_argf "%s: max_depth = %d is not positive" context max_depth;
  if not (accept > 0. && accept < 1.) then
    invalid_argf "%s: accept = %g is not in (0, 1)" context accept;
  if rank < 0 then invalid_argf "%s: rank = %d is negative" context rank;
  let c = chains u position in
  let l, g = evaluate u lp position in
  if Nx.shape l <> [| c |] then
    invalid_argf
      "%s: the density returned shape [%s] for a position of %d chains; a \
       density returns one log density per chain, shape [%d]"
      context
      (shape_string (Nx.shape l))
      c c;
  check_range context l;
  check_rows context u lp position l;
  let dt = Nx.dtype l in
  let gp = Gaussian.ptree u in
  let geometry =
    match geometry with
    | Some g ->
        P.map gp
          (fun _ t ->
            Nx.copy (Nx.broadcast_to (Array.append [| c |] (Nx.shape t)) t))
          g
    | None ->
        let scale =
          P.map u
            (fun _ g ->
              let v = Nx.recip (Nx.abs g) in
              Nx.sqrt
                (Nx.clamp
                   ~min:(Nx_dtype.of_float (Nx.dtype v) variance_low)
                   ~max:(Nx_dtype.of_float (Nx.dtype v) variance_high)
                   v))
            g
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

(* Stan's windows: an initial buffer, slow windows that double in length, the
   last stretched to the final buffer, and the final buffer; each window says
   whether the geometry is refitted at its end. Too few steps for a slow window
   adapt the step size alone. *)
let schedule n =
  if n < 20 then if n = 0 then [] else [ (n, false) ]
  else
    let first, last, base =
      if 75 + 50 + 25 > n then
        let first = int_of_float (0.15 *. float_of_int n)
        and last = int_of_float (0.1 *. float_of_int n) in
        (first, last, n - first - last)
      else (75, 50, 25)
    in
    let slow_end = n - last in
    let rec slow start size acc =
      if start >= slow_end then List.rev acc
      else
        let size =
          if start + (3 * size) > slow_end then slow_end - start else size
        in
        slow (start + size) (2 * size) ((size, true) :: acc)
    in
    List.filter
      (fun (n, _) -> n > 0)
      ([ (first, false) ] @ slow first base [] @ [ (last, false) ])

(* Dual averaging (Hoffman and Gelman 2014), Stan's constants. *)
let da_gamma = 0.05
let da_kappa = 0.75
let da_t0 = 10.

type ('u, 'f) adapt = {
  st : ('u, 'f) state;
  mu : (float, 'f) Nx.t;
  s_bar : (float, 'f) Nx.t;
  x_bar : (float, 'f) Nx.t;
  count : (float, 'f) Nx.t; (* steps since the last restart *)
  index : Nx.int32_t; (* warmup steps taken *)
  n : Nx.int32_t; (* the window's draws so far *)
  shift : 'u; (* the window's first position *)
  sx : 'u;
  sxx : 'u;
  sg : 'u;
  sgg : 'u;
  xs : 'u; (* the window's draws and gradients, for a low-rank fit *)
  gs : 'u;
}

type ('u, 'f) adapt' = ('u, 'f) adapt

let adapt_ptree (type u f) (u : u P.t) : (u, f) adapt P.t =
  let sp = ptree u in
  let module S = struct
    type _ t = (u, f) adapt'

    let walk c (a : (u, f) adapt) : (u, f) adapt =
      let open P.Walk in
      let st = field c "st" (structure sp) a.st in
      let mu = field c "mu" tensor a.mu in
      let s_bar = field c "s_bar" tensor a.s_bar in
      let x_bar = field c "x_bar" tensor a.x_bar in
      let count = field c "count" tensor a.count in
      let index = field c "index" tensor a.index in
      let n = field c "n" tensor a.n in
      let shift = field c "shift" (structure u) a.shift in
      let sx = field c "sx" (structure u) a.sx in
      let sxx = field c "sxx" (structure u) a.sxx in
      let sg = field c "sg" (structure u) a.sg in
      let sgg = field c "sgg" (structure u) a.sgg in
      let xs = field c "xs" (structure u) a.xs in
      let gs = field c "gs" (structure u) a.gs in
      { st; mu; s_bar; x_bar; count; index; n; shift; sx; sxx; sg; sgg; xs; gs }
  end in
  P.nest (module S) P.unit

let zeros u x = P.map u (fun _ t -> Nx.zeros_like t) x
let sq u x = P.map u (fun _ t -> Nx.mul t t) x

(* [init_step_size u lp k s] is each chain's step size doubled or halved, from
   [s.step_size], until one leapfrog step from its position with a fresh
   momentum crosses an acceptance of 0.8 (Stan's heuristic). *)
let init_step_size (type f) u lp k (s : (_, f) state) : (float, f) Nx.t =
  let eps = s.step_size in
  let dt = Nx.dtype eps in
  let c = (Nx.shape eps).(0) in
  let lp_z z = lp (per_chain_color u s.geometry z) in
  let z0 = per_chain_whiten u s.geometry s.position in
  let g0 = to_whitened u s.geometry z0 s.grad in
  let all = Nx.ones Nx.bool [| c |] in
  let log_ratio eps i =
    let keys = Nx.Rng.split_batch ~n:c (Nx.Rng.fold_in_tensor k i) in
    let p = momentum u keys z0 in
    let kinetic p = Nx.mul_s (rows_dot u eps p p) 0.5 in
    let start = { z = z0; p; g = g0; lp = s.lp } in
    let leaf, _ = leapfrog u lp_z all eps start in
    let d =
      Nx.sub (Nx.sub (kinetic p) s.lp) (Nx.sub (kinetic leaf.p) leaf.lp)
    in
    Nx.where (Nx.isnan d) (Nx.scalar dt Float.neg_infinity) d
  in
  let threshold = Nx.scalar dt (Float.log 0.8) in
  let up = Nx.greater (log_ratio eps (Nx.scalar Nx.int32 0l)) threshold in
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
      (eps, (Nx.scalar Nx.int32 0l, all))
  in
  eps

(* Elements as one vector *)

(* [rows_matrix u dt x] is [x], each tensor with a leading axis of [L], as an
   [[L; d]] matrix of the float elements; [vector_of u dt m] reads the [k] rows
   of an [[k; d]] matrix back into [x]'s structure. *)
let rows_matrix (type f) u (dt : (float, f) Nx.dtype) x : (float, f) Nx.t =
  let rows =
    P.fold u
      (fun _ t acc ->
        if not (float_leaf t) then acc
        else
          let s = Nx.shape t in
          let n =
            Array.fold_left ( * ) 1 (Array.sub s 1 (Array.length s - 1))
          in
          Nx.cast dt (Nx.reshape [| s.(0); n |] t) :: acc)
      x []
  in
  Nx.concatenate ~axis:1 (List.rev rows)

let matrix_rows u like m =
  let k = (Nx.shape m).(0) in
  let offset = ref 0 in
  P.map u
    (fun _ t ->
      let shape = Array.append [| k |] (Nx.shape t) in
      if not (float_leaf t) then Nx.zeros (Nx.dtype t) shape
      else
        let n = Nx.numel t in
        let cols = Nx.shrink [| (0, k); (!offset, !offset + n) |] m in
        offset := !offset + n;
        Nx.reshape shape (Nx.cast (Nx.dtype t) cols))
    like

(* The regularisation of the covariances a low-rank fit compares. *)
let low_rank_ridge = 1e-5

(* [low_rank_fit u dt ~rank n m s xs gs] is one chain's Gaussian of mean [m],
   diagonal scale [s], and the [rank] directions where the window's first [n]
   draws [xs] and scores [gs], whitened by [s], disagree most. In the span [Q]
   of both, the covariance minimising the Fisher divergence is the geometric
   mean of the draws' covariance [C_x] and the inverse of the scores' [C_g]; its
   eigenvalues farthest from 1 in ratio give the directions. *)
let low_rank_fit (type f) u (dt : (float, f) Nx.dtype) ~rank n m s xs gs =
  let x = rows_matrix u dt xs and g = rows_matrix u dt gs in
  let l = (Nx.shape x).(0) in
  let valid =
    Nx.reshape [| l; 1 |] (Nx.cast dt (Nx.less (Nx.arange Nx.int32 0 l 1) n))
  in
  let count = Nx.cast dt n in
  let mean =
    rows_matrix u dt (P.map u (fun _ t -> Nx.unsqueeze ~axes:[ 0 ] t) m)
  in
  let scale =
    rows_matrix u dt (P.map u (fun _ t -> Nx.unsqueeze ~axes:[ 0 ] t) s)
  in
  let x = Nx.mul valid (Nx.div (Nx.sub x mean) scale) in
  let g_mean =
    Nx.div (Nx.sum ~axes:[ 0 ] ~keepdims:true (Nx.mul valid g)) count
  in
  let g = Nx.mul valid (Nx.mul (Nx.sub g g_mean) scale) in
  let q, _ = Nx.qr (Nx.transpose (Nx.concatenate ~axis:0 [ x; g ])) in
  let r = (Nx.shape q).(1) in
  let ridge = Nx.mul_s (Nx.eye dt r) low_rank_ridge in
  let cov a =
    let p = Nx.matmul a q in
    Nx.add (Nx.div (Nx.matmul (Nx.transpose p) p) count) ridge
  in
  let cx = cov x and cg = cov g in
  let power m e =
    let w, v = Nx.eigh m in
    let w = Nx.pow (Nx.cast dt w) (Nx.scalar dt e) in
    Nx.matmul (Nx.mul v (Nx.unsqueeze ~axes:[ 0 ] w)) (Nx.transpose v)
  in
  let half = power cg 0.5 and inv_half = power cg (-0.5) in
  let inner = power (Nx.matmul half (Nx.matmul cx half)) 0.5 in
  let sigma = Nx.matmul inv_half (Nx.matmul inner inv_half) in
  let w, v = Nx.eigh sigma in
  let w = Nx.cast dt w in
  let order = Nx.argsort ~descending:true (Nx.abs (Nx.log w)) in
  let keep = Nx.shrink [| (0, min rank r) |] order in
  let variances = Nx.take ~indices:keep w in
  let directions =
    Nx.transpose (Nx.matmul q (Nx.take ~axis:1 ~indices:keep v))
  in
  let directions, variances =
    if rank <= r then (directions, variances)
    else
      (* Fewer independent draws than directions: the rest change nothing. *)
      ( Nx.concatenate ~axis:0
          [ directions; Nx.zeros dt [| rank - r; (Nx.shape q).(0) |] ],
        Nx.concatenate ~axis:0 [ variances; Nx.ones dt [| rank - r |] ] )
  in
  let variances =
    Nx.clamp
      ~min:(Nx_dtype.of_float dt variance_low)
      ~max:(Nx_dtype.of_float dt variance_high)
      variances
  in
  Gaussian.low_rank u ~mean:m ~scale:s
    ~directions:(matrix_rows u m directions)
    ~variances

(* [fisher u ~rank n a] is each chain's Gaussian fitted to its window's [n]
   draws and gradients by the Fisher divergence: a diagonal scale [sqrt (sd x /
   sd score)] per element, clipped where a score does not vary, then the [rank]
   directions along which the draws' and the scores' covariances, whitened by
   it, disagree most, from their span. *)
let fisher (type u f) (u : u P.t) ~rank (a : (u, f) adapt) : (u, f) Gaussian.t =
  let dt = Nx.dtype a.mu in
  let n = Nx.cast dt a.n in
  let mean_of s = P.map u (fun _ t -> Nx.div t (Nx.cast (Nx.dtype t) n)) s in
  let var_of s ss =
    P.map2 u
      (fun _ s ss ->
        let n = Nx.cast (Nx.dtype s) n in
        let m = Nx.div s n in
        Nx.maximum (Nx.sub (Nx.div ss n) (Nx.mul m m)) (Nx.zeros_like m))
      s ss
  in
  let mean = P.map2 u (fun _ m sh -> Nx.add m sh) (mean_of a.sx) a.shift in
  let vx = var_of a.sx a.sxx and vg = var_of a.sg a.sgg in
  let scale =
    P.map2 u
      (fun _ vx vg ->
        let dtx = Nx.dtype vx in
        let r = Nx.sqrt (Nx.div vx vg) in
        let r = Nx.where (Nx.isnan r) (Nx.ones_like r) r in
        Nx.sqrt
          (Nx.clamp
             ~min:(Nx_dtype.of_float dtx variance_low)
             ~max:(Nx_dtype.of_float dtx variance_high)
             r))
      vx vg
  in
  let gp = Gaussian.ptree u in
  if rank = 0 then
    Rune.vmap
      P.(u @-> u @-> returns gp)
      (fun m s -> Gaussian.diagonal u dt ~mean:m ~scale:s)
      mean scale
  else
    Rune.vmap
      P.(u @-> u @-> u @-> u @-> returns gp)
      (fun m s xs gs -> low_rank_fit u dt ~rank a.n m s xs gs)
      mean scale a.xs a.gs

let warmup u lp k ~steps (s : (_, _) state) =
  if steps < 0 then
    invalid_argf "Norn.Nuts.warmup: steps = %d is negative" steps;
  let windows = schedule steps in
  if windows = [] then s
  else
    let dt = Nx.dtype s.lp in
    let c = (Nx.shape s.lp).(0) in
    let rank = Geometry.rank s.geometry in
    let longest = List.fold_left (fun m (n, _) -> max m n) 0 windows in
    (* A low-rank fit reads the window's draws; a diagonal one their sums. *)
    let buffer = if rank = 0 then 0 else longest in
    let buffers x =
      P.map u
        (fun _ t ->
          let sh = Nx.shape t in
          Nx.zeros (Nx.dtype t)
            (Array.concat
               [ [| sh.(0); buffer |]; Array.sub sh 1 (Array.length sh - 1) ]))
        x
    in
    let restart st =
      ( Nx.log (Nx.mul_s st.step_size 10.),
        Nx.zeros dt [| c |],
        Nx.zeros dt [| c |],
        Nx.zeros dt [||] )
    in
    let step_keys = Nx.Rng.fold_in k 0 and size_keys = Nx.Rng.fold_in k 1 in
    let s0 =
      { s with step_size = init_step_size u lp (Nx.Rng.fold_in size_keys 0) s }
    in
    let mu, s_bar, x_bar, count = restart s0 in
    let a0 =
      {
        st = s0;
        mu;
        s_bar;
        x_bar;
        count;
        index = Nx.scalar Nx.int32 0l;
        n = Nx.scalar Nx.int32 0l;
        shift = s0.position;
        sx = zeros u s0.position;
        sxx = zeros u s0.position;
        sg = zeros u s0.position;
        sgg = zeros u s0.position;
        xs = buffers s0.position;
        gs = buffers s0.position;
      }
    in
    let one (a : (_, _) adapt) =
      let st = step u lp (Nx.Rng.fold_in_tensor step_keys a.index) a.st in
      (* Dual averaging toward the target acceptance. *)
      let count = Nx.add_s a.count 1. in
      let eta = Nx.recip (Nx.add_s count da_t0) in
      let stat =
        Nx.minimum st.stats.acceptance (Nx.ones_like st.stats.acceptance)
      in
      let s_bar =
        Nx.add
          (Nx.mul (Nx.sub (Nx.ones_like eta) eta) a.s_bar)
          (Nx.mul eta (Nx.sub st.accept stat))
      in
      let x =
        Nx.sub a.mu
          (Nx.div (Nx.mul s_bar (Nx.sqrt count)) (Nx.scalar dt da_gamma))
      in
      let x_eta = Nx.pow count (Nx.scalar dt (-.da_kappa)) in
      let x_bar =
        Nx.add
          (Nx.mul (Nx.sub (Nx.ones_like x_eta) x_eta) a.x_bar)
          (Nx.mul x_eta x)
      in
      let st = { st with step_size = Nx.exp x } in
      (* The window's sums, shifted by its first position. *)
      let dx = P.map2 u (fun _ x sh -> Nx.sub x sh) st.position a.shift in
      let at buf v =
        if buffer = 0 then buf
        else
          P.map2 u
            (fun _ b v ->
              Nx.set
                [ Nx.A; Nx.D (Nx.cast Nx.int64 a.n, 1) ]
                (Nx.unsqueeze ~axes:[ 1 ] v)
                b)
            buf v
      in
      {
        a with
        st;
        s_bar;
        x_bar;
        count;
        index = Nx.add a.index (Nx.scalar Nx.int32 1l);
        n = Nx.add a.n (Nx.scalar Nx.int32 1l);
        sx = add u a.sx dx;
        sxx = add u a.sxx (sq u dx);
        sg = add u a.sg st.grad;
        sgg = add u a.sgg (sq u st.grad);
        xs = at a.xs st.position;
        gs = at a.gs st.grad;
      }
    in
    let window (a : (_, _) adapt) (length, refit) =
      let a =
        {
          a with
          n = Nx.scalar Nx.int32 0l;
          shift = a.st.position;
          sx = zeros u a.sx;
          sxx = zeros u a.sxx;
          sg = zeros u a.sg;
          sgg = zeros u a.sgg;
        }
      in
      let a =
        Rune.iterate (adapt_ptree u) ~max:longest
          ~until:(fun a -> Nx.greater_equal a.n length)
          ~f:one a
      in
      (* At a slow window's end: refit the geometry, find a step size for it and
         restart the averaging there. *)
      let refit = Nx.not_equal refit (Nx.scalar Nx.int32 0l) in
      let geometry = fisher u ~rank a in
      let gp = Gaussian.ptree u in
      let refitted =
        {
          a.st with
          geometry =
            P.map2 gp (fun _ g o -> Nx.where refit g o) geometry a.st.geometry;
        }
      in
      let eps =
        init_step_size u lp
          (Nx.Rng.fold_in_tensor size_keys
             (Nx.add a.index (Nx.scalar Nx.int32 1l)))
          refitted
      in
      let eps = Nx.where refit eps a.st.step_size in
      let st = { refitted with step_size = eps } in
      let mu', s_bar', x_bar', count' = restart st in
      ( {
          a with
          st;
          mu = Nx.where refit mu' a.mu;
          s_bar = Nx.where refit s_bar' a.s_bar;
          x_bar = Nx.where refit x_bar' a.x_bar;
          count = Nx.where refit count' a.count;
        },
        () )
    in
    let lengths =
      Nx.create Nx.int32
        [| List.length windows |]
        (Array.of_list (List.map (fun (n, _) -> Int32.of_int n) windows))
    in
    let refits =
      Nx.create Nx.int32
        [| List.length windows |]
        (Array.of_list (List.map (fun (_, r) -> if r then 1l else 0l) windows))
    in
    let a, () =
      Rune.scan (adapt_ptree u)
        P.(pair tensor tensor)
        P.unit ~f:window ~init:a0 (lengths, refits)
    in
    { a.st with step_size = Nx.exp a.x_bar; draw = s.draw }
