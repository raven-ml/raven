(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

(* The bracket's initial width along a whitened unit direction: the expected
   width of a standard normal's slice at a point of it. With [x] standard normal
   and the level [Exp (1)] below its log density, the slice is [|t| < sqrt (x² +
   2 E)], whose half-width is a chi variable of 3 degrees of freedom, of mean [2
   sqrt (2 / π)]. *)
let width = 4. *. Float.sqrt (2. /. Float.pi)

(* The dtype's significand bits. The bracket doubles at most this many times,
   Neal's [p], so the index of a width-long cell of the bracket is an exact
   float. *)
let bits dt = 1 + int_of_float (Float.round (-.Float.log2 (Prec.eps dt)))

(* A trip evaluates one point: the first bracket's two ends, then one per
   doubling, then per point drawn while shrinking that point and, if it lies in
   the slice, one per level of its check, fewer than the doublings. A rejection
   shrinks the bracket by a factor whose log has mean below [-0.3], so eight
   draws per bit of the ratio of the widest bracket to the narrowest, [2^(2
   bits)], bound a run on any density. *)
let bound dt =
  let b = bits dt in
  2 + b + (b * 8 * 2 * b)

(* Phases *)

let lower = 0l (* evaluating the first bracket's lower end *)
let upper = 1l (* its upper end *)
let double = 2l (* the end a doubling adds *)
let shrink = 3l (* a point drawn in the bracket *)
let check = 4l (* a midpoint of the drawn point's acceptance check *)
let finished = 5l

type ('u, 'f, 'a) walk = {
  last : 'u; (* the last position each walker evaluated *)
  x : 'u; (* the walker, or the point under check *)
  lp : (float, 'f) Nx.t;
  aux : 'a;
  lo : (float, 'f) Nx.t; (* the bracket, in steps along the direction *)
  hi : (float, 'f) Nx.t;
  left : (float, 'f) Nx.t; (* the doubled bracket's lower end *)
  home : (float, 'f) Nx.t; (* the walker's cell in it, from 0 *)
  doublings : Nx.int32_t;
  in_lo : Nx.bool_t; (* whether the doubled or checked interval's ends lie *)
  in_hi : Nx.bool_t; (* in the slice *)
  inner : Nx.bool_t; (* [[walker; bits]]: per doubling, whether the inner and *)
  outer : Nx.bool_t; (* outer ends of the half it added lie in the slice *)
  candidate : (float, 'f) Nx.t; (* the point under check *)
  check_lo : (float, 'f) Nx.t; (* its checked interval's lower end *)
  levels : Nx.int32_t; (* the halvings left of its check *)
  phase : Nx.int32_t;
  evaluations : Nx.int32_t;
  trip : Nx.int32_t;
}

type ('u, 'f, 'a) walk' = ('u, 'f, 'a) walk

let walk_ptree (type u f a) (u : u P.t) (a : a P.t) : (u, f, a) walk P.t =
  let module S = struct
    type _ t = (u, f, a) walk'

    let walk c (w : (u, f, a) walk) : (u, f, a) walk =
      let open P.Walk in
      let last = field c "last" (structure u) w.last in
      let x = field c "x" (structure u) w.x in
      let lp = field c "lp" tensor w.lp in
      let aux = field c "aux" (structure a) w.aux in
      let lo = field c "lo" tensor w.lo in
      let hi = field c "hi" tensor w.hi in
      let left = field c "left" tensor w.left in
      let home = field c "home" tensor w.home in
      let doublings = field c "doublings" tensor w.doublings in
      let in_lo = field c "in_lo" tensor w.in_lo in
      let in_hi = field c "in_hi" tensor w.in_hi in
      let inner = field c "inner" tensor w.inner in
      let outer = field c "outer" tensor w.outer in
      let candidate = field c "candidate" tensor w.candidate in
      let check_lo = field c "check_lo" tensor w.check_lo in
      let levels = field c "levels" tensor w.levels in
      let phase = field c "phase" tensor w.phase in
      let evaluations = field c "evaluations" tensor w.evaluations in
      let trip = field c "trip" tensor w.trip in
      {
        last;
        x;
        lp;
        aux;
        lo;
        hi;
        left;
        home;
        doublings;
        in_lo;
        in_hi;
        inner;
        outer;
        candidate;
        check_lo;
        levels;
        phase;
        evaluations;
        trip;
      }
  end in
  P.nest (module S) P.unit

(* [per_row keys f] is [f k] for each row's key [k]. *)
let per_row keys f = Rune.vmap P.(Nx.Rng.ptree @-> returns tensor) f keys

(* Per-walker flags, [[walker; bits]]: [at flags column i] is row [r]'s flag
   [i.(r)], and [put flags column i mask v] sets it to [v.(r)] where [mask];
   [column] is [[0; bits)] as a row. *)
let at flags column i =
  Nx.any ~axes:[ 1 ]
    (Nx.logical_and flags (Nx.equal column (Nx.unsqueeze ~axes:[ 1 ] i)))

let put flags column i mask v =
  let col t = Nx.unsqueeze ~axes:[ 1 ] t in
  Nx.where (Nx.logical_and (col mask) (Nx.equal column (col i))) (col v) flags

(* Neal's doubling (2003, figs. 4 to 6). The bracket, [width] placed at random
   around the walker, doubles toward a random side while an end lies in the
   slice. It is the union of width-long cells; the halves a doubling added nest,
   so the cells of a point and of the walker first part at the half added by the
   doubling of index the highest bit where their cell indices differ. A point
   drawn in the slice is accepted unless, in that half or a half of it that
   holds the point, both ends lie outside the slice: from that point the
   doubling could not have produced this bracket, and accepting it would break
   reversibility. The ends of the half first parted come from the doubling; each
   further level evaluates its midpoint. *)
let move (type f) context u a eval keys ~direction x0 (lp0 : (float, f) Nx.t)
    aux0 =
  let dt = Nx.dtype lp0 in
  let c = (Nx.shape lp0).(0) in
  let b = bits dt in
  let i32 v = Nx.scalar Nx.int32 v in
  let f v = Nx.scalar dt v in
  let pow2 e = Nx.pow (f 2.) (Nx.cast dt e) in
  let uniform i =
    per_row keys (fun k -> Nx.Rng.uniform (Nx.Rng.fold_in k i) dt [||])
  in
  let lo = Nx.mul_s (uniform 0) (-.width) in
  let level = Nx.add lp0 (Nx.log (uniform 2)) in
  let resolution = f (width *. Prec.eps dt) in
  let powers =
    Nx.unsqueeze ~axes:[ 0 ]
      (Nx.pow (f 2.) (Nx.arange_f dt 0. (float_of_int b) 1.))
  in
  let column = Nx.unsqueeze ~axes:[ 0 ] (Nx.arange Nx.int32 0 b 1) in
  let trip_keys =
    Rune.vmap
      P.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
      (fun k -> Nx.Rng.fold_in k 3)
      keys
  in
  let zero = f 0. and half = f 0.5 and one = i32 1l in
  let at_most_bits = i32 (Int32.of_int b) in
  let lower', upper', double', shrink', check', finished' =
    (i32 lower, i32 upper, i32 double, i32 shrink, i32 check, i32 finished)
  in
  let trip w =
    let active = Nx.less w.phase finished' in
    let is p = Nx.logical_and active (Nx.equal w.phase p) in
    let is_lower = is lower' and is_upper = is upper' in
    let doubling = is double' and drawn = is shrink' and checked = is check' in
    (* A draw, and a key for the density, per walker and trip. *)
    let draw, eval_keys =
      Rune.vmap
        P.(Nx.Rng.ptree @-> returns (pair tensor Nx.Rng.ptree))
        (fun k ->
          let k = Nx.Rng.fold_in_tensor k w.trip in
          (Nx.Rng.uniform (Nx.Rng.fold_in k 0) dt [||], Nx.Rng.fold_in k 1))
        trip_keys
    in
    let span = Nx.sub w.hi w.lo in
    let leftward = Nx.less draw half in
    let midpoint =
      Nx.add w.check_lo (Nx.mul_s (pow2 (Nx.sub w.levels one)) width)
    in
    let t =
      Nx.where is_lower w.lo
        (Nx.where is_upper w.hi
           (Nx.where doubling
              (Nx.where leftward (Nx.sub w.lo span) (Nx.add w.hi span))
              (Nx.where checked midpoint (Nx.add w.lo (Nx.mul draw span)))))
    in
    let proposal = Rows.axpy u t direction x0 in
    let point = Rows.choose u active proposal w.last in
    let lp, aux = eval eval_keys point in
    Rows.check_range context lp;
    let inside = Nx.greater lp level in
    (* The first bracket's ends, then doublings. *)
    let to_lo = Nx.logical_and doubling leftward in
    let to_hi = Nx.logical_and doubling (Nx.logical_not leftward) in
    let inner =
      put w.inner column w.doublings doubling
        (Nx.where leftward w.in_lo w.in_hi)
    in
    let outer = put w.outer column w.doublings doubling inside in
    let home = Nx.where to_lo (Nx.add w.home (pow2 w.doublings)) w.home in
    let doublings = Nx.add w.doublings (Nx.cast Nx.int32 doubling) in
    let in_lo = Nx.where (Nx.logical_or is_lower to_lo) inside w.in_lo in
    let in_hi = Nx.where (Nx.logical_or is_upper to_hi) inside w.in_hi in
    let lo = Nx.where to_lo t w.lo and hi = Nx.where to_hi t w.hi in
    let left = Nx.where to_lo t w.left in
    let grows =
      Nx.logical_and
        (Nx.logical_or in_lo in_hi)
        (Nx.less doublings at_most_bits)
    in
    let bracketed = Nx.logical_or is_upper doubling in
    (* A point drawn in the bracket: its cell, and the doubling whose half first
       parts it from the walker's. *)
    let cell =
      Nx.maximum zero
        (Nx.minimum
           (Nx.floor (Nx.div_s (Nx.sub t w.left) width))
           (Nx.sub_s (pow2 w.doublings) 1.))
    in
    let parted =
      Nx.cast Nx.int32
        (Nx.sum ~axes:[ 1 ]
           (Nx.cast dt
              (Nx.not_equal
                 (Nx.floor (Nx.div (Nx.unsqueeze ~axes:[ 1 ] w.home) powers))
                 (Nx.floor (Nx.div (Nx.unsqueeze ~axes:[ 1 ] cell) powers)))))
    in
    let h = Nx.maximum (Nx.sub parted one) (i32 0l) in
    let right = Nx.greater cell w.home in
    let near = at w.inner column h and far = at w.outer column h in
    let half_lo = Nx.where right near far
    and half_hi = Nx.where right far near in
    let separated = Nx.greater parted (i32 0l) in
    let excluded =
      Nx.logical_and separated (Nx.logical_not (Nx.logical_or half_lo half_hi))
    in
    let taken =
      Nx.logical_and drawn (Nx.logical_and inside (Nx.logical_not excluded))
    in
    let checking = Nx.logical_and taken (Nx.greater h (i32 0l)) in
    let accepted_drawn = Nx.logical_and taken (Nx.logical_not checking) in
    let refused_drawn = Nx.logical_and drawn (Nx.logical_not taken) in
    (* A level of a check: the half of the checked interval that holds the
       point. *)
    let low_half = Nx.less w.candidate midpoint in
    let c_lo = Nx.where low_half w.in_lo inside
    and c_hi = Nx.where low_half inside w.in_hi in
    let c_out = Nx.logical_not (Nx.logical_or c_lo c_hi) in
    let levels_left = Nx.sub w.levels one in
    let refused_checked = Nx.logical_and checked c_out in
    let accepted_checked =
      Nx.logical_and checked
        (Nx.logical_and (Nx.logical_not c_out) (Nx.equal levels_left (i32 0l)))
    in
    let in_lo = Nx.where checking half_lo (Nx.where checked c_lo in_lo) in
    let in_hi = Nx.where checking half_hi (Nx.where checked c_hi in_hi) in
    let check_lo =
      Nx.where checking
        (Nx.add w.left
           (Nx.mul_s (Nx.mul (Nx.floor (Nx.div cell (pow2 h))) (pow2 h)) width))
        (Nx.where
           (Nx.logical_and checked (Nx.logical_not low_half))
           midpoint w.check_lo)
    in
    let levels = Nx.where checking h (Nx.where checked levels_left w.levels) in
    let candidate = Nx.where checking t w.candidate in
    (* A refused point shrinks the bracket toward the walker. *)
    let refused = Nx.logical_or refused_drawn refused_checked in
    let r = Nx.where refused_checked w.candidate t in
    let below = Nx.less r zero in
    let lo = Nx.where (Nx.logical_and refused below) r lo in
    let hi = Nx.where (Nx.logical_and refused (Nx.logical_not below)) r hi in
    let collapsed =
      Nx.logical_and refused (Nx.less (Nx.sub hi lo) resolution)
    in
    let accepted = Nx.logical_or accepted_drawn accepted_checked in
    let phase =
      Nx.where is_lower upper'
        (Nx.where bracketed
           (Nx.where grows double' shrink')
           (Nx.where
              (Nx.logical_or accepted collapsed)
              finished'
              (Nx.where checking check'
                 (Nx.where refused_checked shrink' w.phase))))
    in
    {
      last = point;
      x = Rows.choose u taken proposal (Rows.choose u refused_checked x0 w.x);
      lp = Nx.where taken lp (Nx.where refused_checked lp0 w.lp);
      aux = Rows.choose a taken aux (Rows.choose a refused_checked aux0 w.aux);
      lo;
      hi;
      left;
      home;
      doublings;
      in_lo;
      in_hi;
      inner;
      outer;
      candidate;
      check_lo;
      levels;
      phase;
      evaluations = Nx.add w.evaluations (Nx.cast Nx.int32 active);
      trip = Nx.add w.trip one;
    }
  in
  let no = Nx.zeros Nx.bool [| c |] in
  let zero = Nx.zeros dt [| c |] in
  let w =
    Rune.iterate (walk_ptree u a) ~max:(bound dt)
      ~until:(fun w -> Nx.logical_not (Nx.any (Nx.less w.phase finished')))
      ~f:trip
      {
        last = x0;
        x = x0;
        lp = lp0;
        aux = aux0;
        lo;
        hi = Nx.add_s lo width;
        left = lo;
        home = zero;
        doublings = Nx.zeros Nx.int32 [| c |];
        in_lo = no;
        in_hi = no;
        inner = Nx.zeros Nx.bool [| c; b |];
        outer = Nx.zeros Nx.bool [| c; b |];
        candidate = zero;
        check_lo = zero;
        levels = Nx.zeros Nx.int32 [| c |];
        phase = Nx.full Nx.int32 [| c |] lower;
        evaluations = Nx.zeros Nx.int32 [| c |];
        trip = i32 0l;
      }
  in
  (w.x, w.lp, w.aux, w.evaluations)

(* Directions *)

let unit_directions u keys lp like =
  let z =
    Rune.vmap
      P.(Nx.Rng.ptree @-> u @-> returns u)
      (fun k x ->
        let i = ref (-1) in
        P.map u
          (fun _ t ->
            incr i;
            Noise.normal_like t (Nx.Rng.fold_in k !i) (Nx.shape t))
          x)
      keys like
  in
  let norm = Nx.sqrt (Rows.dot u lp z z) in
  P.map u
    (fun _ t -> if Rows.float_leaf t then Nx.div t (Rows.column norm t) else t)
    z

(* The walkers move as flat rows ([Rows.ravel]): a step is then a few operations
   on one matrix rather than a few per tensor. *)
let hit_and_run context u a eval keys g x lp aux =
  let rows = Rows.ravel u lp x in
  let z =
    unit_directions P.tensor
      (Rune.vmap
         P.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
         (fun k -> Nx.Rng.fold_in k 5)
         keys)
      lp rows
  in
  let direction = Gaussian.direction_flat (Gaussian.flat u x g) z in
  let eval ks y = eval ks (Rows.unravel u x y) in
  let y, lp, aux, evaluations =
    move context P.tensor a eval keys ~direction rows lp aux
  in
  (Rows.unravel u x y, lp, aux, evaluations)
