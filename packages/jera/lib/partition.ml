(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('d, 'b) t = {
  level : (int32, Nx.int32_elt) Nx.t;
  index : (int64, Nx.int64_elt) Nx.t;
  axis : (int64, Nx.int64_elt) Nx.t;
  data : 'd;
  error : (float, 'b) Nx.t;
  used : (int32, Nx.int32_elt) Nx.t;
  status : (int32, Nx.int32_elt) Nx.t;
  evaluations : (int32, Nx.int32_elt) Nx.t;
}

let max_level = 62
let chunk = 32

let fractions dtype level index =
  let scale = Nx.exp2 (Nx.neg (Nx.cast dtype level)) in
  let i = Nx.cast dtype index in
  (Nx.mul i scale, Nx.mul (Nx.add_s i 1.) scale)

let tail v = Array.sub (Nx.shape v) 1 (Nx.ndim v - 1)

(* [m] with trailing axes of size 1 up to [v]'s rank, broadcast to [v]. *)
let widen m v =
  if Nx.shape m = Nx.shape v then m
  else
    let extra = Nx.ndim v - Nx.ndim m in
    Nx.broadcast_to (Nx.shape v)
      (Nx.reshape (Array.append (Nx.shape m) (Array.make extra 1)) m)

(* The slot numbers, [[budget] @ lanes]. *)
let slots ~budget ~lanes =
  Nx.broadcast_to
    (Array.append [| budget |] lanes)
    (Nx.reshape
       (Array.append [| budget |] (Array.make (Array.length lanes) 1))
       (Nx.arange Nx.int32 0 budget 1))

let live slot p =
  Nx.less slot
    (Nx.broadcast_to (Nx.shape slot) (Nx.unsqueeze ~axes:[ 0 ] p.used))

let in_use p =
  live (slots ~budget:(Nx.dim 0 p.error) ~lanes:(Nx.shape p.used)) p

let sum live v =
  Nx.sum ~axes:[ 0 ] (Nx.where (widen live v) v (Nx.zeros_like v))

let worst_slot live p =
  Nx.argmax ~axis:0
    (Nx.where live p.error (Nx.full_like p.error Float.neg_infinity))

(* Slot [j] of [v], [[budget] @ lanes @ rest], per lane. *)
let pick j v =
  if Nx.ndim j = 0 then Nx.take ~axis:0 ~indices:j v
  else
    let j = Nx.unsqueeze ~axes:[ 0 ] j in
    let j =
      Nx.reshape
        (Array.append (Nx.shape j) (Array.make (Nx.ndim v - Nx.ndim j) 1))
        j
    in
    let j = Nx.broadcast_to (Array.append [| 1 |] (tail v)) j in
    Nx.squeeze ~axes:[ 0 ] (Nx.take_along_axis ~axis:0 ~indices:j v)

(* Each lane's worst box: its split axis, and its level and ends along it. A box
   of one axis splits along it. *)
let worst_box live p =
  let j = worst_slot live p in
  let r = Nx.ndim j in
  let one_axis = Nx.dim (r + 1) p.level = 1 in
  let a = if one_axis then Nx.zeros_like j else pick j p.axis in
  let along v =
    if one_axis then Nx.squeeze ~axes:[ r ] v
    else
      Nx.squeeze ~axes:[ r ]
        (Nx.take_along_axis ~axis:r ~indices:(Nx.unsqueeze ~axes:[ r ] a) v)
  in
  let level = along (pick j p.level) in
  let t0, t1 = fractions (Nx.dtype p.error) level (along (pick j p.index)) in
  (j, a, level, t0, t1)

let worst p ~point =
  let _, a, _, t0, t1 = worst_box (in_use p) p in
  (point a t0, point a t1)

let refine s ~budget ~lanes ~dims ~cost ~evaluate ~point ~verdict =
  let tree () =
    Nx.Ptree.iso
      (fun ( level,
             (index, (axis, (data, (error, (used, (status, evaluations)))))) )
         -> { level; index; axis; data; error; used; status; evaluations })
      (fun p ->
        ( p.level,
          ( p.index,
            (p.axis, (p.data, (p.error, (p.used, (p.status, p.evaluations)))))
          ) ))
      Nx.Ptree.(
        pair tensor
          (pair tensor
             (pair tensor
                (pair s (pair tensor (pair tensor (pair tensor tensor)))))))
  in
  let slot = slots ~budget ~lanes in
  let settle p =
    let live = live slot p in
    let not_finite, met = verdict p live in
    let st = Elementwise.settle p.status not_finite Not_finite in
    let st = Elementwise.settle st met Converged in
    let _, a, level, t0, t1 = worst_box live p in
    let x0 = point a t0 and x1 = point a t1 in
    let flat =
      Nx.logical_or
        (Nx.greater_equal_s level (Int32.of_int max_level))
        (Num.adjacent (Nx.minimum x0 x1) (Nx.maximum x0 x1))
    in
    let st = Elementwise.settle st flat Stalled in
    let st =
      Elementwise.settle st
        (Nx.greater_equal_s p.used (Int32.of_int budget))
        Budget_spent
    in
    { p with status = st }
  in
  (* Bisects each searching lane's worst box across its split axis: the left
     half replaces it and the right one takes the next free slot. *)
  let step p =
    let run = Elementwise.searching p.status in
    let j = worst_slot (live slot p) p in
    let level = pick j p.level and index = pick j p.index in
    let child, left_index, right_index =
      let twice = Nx.mul_s index 2L in
      if dims = 1 then (Nx.add_s level 1l, twice, Nx.add_s twice 1L)
      else
        let a = pick j p.axis in
        let on_a =
          Nx.equal
            (Nx.broadcast_to (Nx.shape index) (Nx.arange Nx.int64 0 dims 1))
            (Nx.unsqueeze ~axes:[ Nx.ndim a ] a)
        in
        ( Nx.where on_a (Nx.add_s level 1l) level,
          Nx.where on_a twice index,
          Nx.where on_a (Nx.add_s twice 1L) index )
    in
    let level2 = Nx.stack [ child; child ]
    and index2 = Nx.stack [ left_index; right_index ] in
    let data2, error2, axis2 = evaluate level2 index2 in
    let at s =
      Nx.logical_and
        (Nx.unsqueeze ~axes:[ 0 ] run)
        (Nx.equal slot (Nx.unsqueeze ~axes:[ 0 ] s))
    in
    let left = at (Nx.cast Nx.int32 j) and right = at p.used in
    let put v two =
      let child i =
        Nx.broadcast_to (Nx.shape v)
          (Nx.unsqueeze ~axes:[ 0 ] (Nx.slice [ Nx.I i ] two))
      in
      Nx.where (widen left v) (child 0) (Nx.where (widen right v) (child 1) v)
    in
    settle
      {
        level = put p.level level2;
        index = put p.index index2;
        axis = (if dims = 1 then p.axis else put p.axis axis2);
        data = Nx.Ptree.map2 s (fun _ v two -> put v two) p.data data2;
        error = put p.error error2;
        used = Nx.add p.used (Nx.cast Nx.int32 run);
        status = p.status;
        evaluations =
          Nx.add p.evaluations
            (Nx.mul_s (Nx.cast Nx.int32 run) (Int32.of_int (2 * cost)));
      }
  in
  let initial =
    let whole dt = Nx.zeros dt (Array.concat [ [| 1 |]; lanes; [| dims |] ]) in
    let data, error, axis = evaluate (whole Nx.int32) (whole Nx.int64) in
    let shape = Array.append [| budget |] lanes in
    let first v =
      let v = Nx.broadcast_to (Array.append [| budget |] (tail v)) v in
      let at0 =
        Nx.reshape
          (Array.append [| budget |] (Array.make (Nx.ndim v - 1) 1))
          (Nx.equal_s (Nx.arange Nx.int32 0 budget 1) 0l)
      in
      Nx.where (Nx.broadcast_to (Nx.shape v) at0) v (Nx.zeros_like v)
    in
    settle
      {
        level = Nx.zeros Nx.int32 (Array.append shape [| dims |]);
        index = Nx.zeros Nx.int64 (Array.append shape [| dims |]);
        axis = first axis;
        data = Nx.Ptree.map s (fun _ v -> first v) data;
        error = first error;
        used = Nx.ones Nx.int32 lanes;
        status = Nx.full Nx.int32 lanes Elementwise.running;
        evaluations = Nx.full Nx.int32 lanes (Int32.of_int cost);
      }
  in
  Rune.iterate (tree ()) ~max:budget
    ~until:(fun p -> Nx.logical_not (Nx.any (Elementwise.searching p.status)))
    ~f:step initial

let integrate p f =
  let budget = Nx.dim 0 p.error in
  let padded = chunk * ((budget + chunk - 1) / chunk) in
  let pad v =
    if padded = budget then v
    else
      Nx.concatenate ~axis:0
        [
          v; Nx.zeros (Nx.dtype v) (Array.append [| padded - budget |] (tail v));
        ]
  in
  let chunks v =
    Nx.reshape (Array.append [| padded / chunk; chunk |] (tail v)) (pad v)
  in
  let used = in_use p in
  let unused v = Nx.where (widen used v) v (Nx.zeros_like v) in
  let sum_chunk total (level, (index, weight)) =
    (Nx.add total (Nx.sum ~axes:[ 0 ] (Nx.mul weight (f level index))), ())
  in
  let dtype = Nx.dtype p.error in
  let total, () =
    Rune.scan Nx.Ptree.tensor
      Nx.Ptree.(pair tensor (pair tensor tensor))
      Nx.Ptree.unit ~f:sum_chunk
      ~init:(Nx.zeros dtype (Nx.shape p.used))
      ( chunks (unused p.level),
        (chunks (unused p.index), chunks (Nx.cast dtype used)) )
  in
  total
