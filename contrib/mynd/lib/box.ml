(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

(* Conversions and measures *)

let corners fn b =
  if Nx.ndim b < 1 || Nx.dim (-1) b <> 4 then
    invalid_argf "%s: boxes have axes [...; 4]" fn;
  (coordinate b 0, coordinate b 1, coordinate b 2, coordinate b 3)

let pack a b c d = Nx.stack ~axis:(-1) [ a; b; c; d ]

let of_xywh b =
  let x, y, w, h = corners "Mynd.Box.of_xywh" b in
  pack x y (Nx.add x w) (Nx.add y h)

let to_xywh b =
  let x0, y0, x1, y1 = corners "Mynd.Box.to_xywh" b in
  pack x0 y0 (Nx.sub x1 x0) (Nx.sub y1 y0)

let of_cxcywh b =
  let cx, cy, w, h = corners "Mynd.Box.of_cxcywh" b in
  let hw = Nx.mul_s w 0.5 and hh = Nx.mul_s h 0.5 in
  pack (Nx.sub cx hw) (Nx.sub cy hh) (Nx.add cx hw) (Nx.add cy hh)

let to_cxcywh b =
  let x0, y0, x1, y1 = corners "Mynd.Box.to_cxcywh" b in
  pack
    (Nx.mul_s (Nx.add x0 x1) 0.5)
    (Nx.mul_s (Nx.add y0 y1) 0.5)
    (Nx.sub x1 x0) (Nx.sub y1 y0)

(* An empty box, [x₁ <= x₀] or [y₁ <= y₀], has area [0]: a difference's sign is
   exact. *)
let area_of (x0, y0, x1, y1) =
  let w = Nx.sub x1 x0 and h = Nx.sub y1 y0 in
  let empty = Nx.logical_or (Nx.less_equal_s w 0.) (Nx.less_equal_s h 0.) in
  Nx.where empty (Nx.zeros_like w) (Nx.mul w h)

let area b = area_of (corners "Mynd.Box.area" b)

(* torchvision's arithmetic: the overlap's sides clamped at [0], their product,
   and [(area a + area b) - i]. *)
let overlap a b =
  let ((ax0, ay0, ax1, ay1) as ca) = a and ((bx0, by0, bx1, by1) as cb) = b in
  let row x = Nx.unsqueeze ~axes:[ -1 ] x
  and col x = Nx.unsqueeze ~axes:[ -2 ] x in
  let side lo lo' hi hi' =
    Nx.maximum_s
      (Nx.sub (Nx.minimum (row hi) (col hi')) (Nx.maximum (row lo) (col lo')))
      0.
  in
  let i = Nx.mul (side ax0 bx0 ax1 bx1) (side ay0 by0 ay1 by1) in
  let union = Nx.sub (Nx.add (row (area_of ca)) (col (area_of cb))) i in
  let empty = Nx.equal_s union 0. in
  let safe = Nx.where empty (Nx.ones_like union) union in
  Nx.where empty (Nx.zeros_like i) (Nx.div i safe)

let iou a b =
  let fn = "Mynd.Box.iou" in
  if Nx.ndim a < 2 || Nx.ndim b < 2 then
    invalid_argf "%s: boxes have axes [...; n; 4]" fn;
  overlap (corners fn a) (corners fn b)

let of_mask dt m =
  let fn = "Mynd.Box.of_mask" in
  mask_shape fn m;
  let h = Nx.dim (-3) m and w = Nx.dim (-2) m in
  if h = 0 || w = 0 then
    (* No pixel: [any] of nothing is [false], [0] once cast. *)
    let none = Nx.cast dt (Nx.any ~axes:[ -3; -2; -1 ] ~keepdims:true m) in
    let s = Nx.shape m in
    let lead = Array.sub s 0 (Array.length s - 3) in
    Nx.copy
      (Nx.broadcast_to
         (Array.append lead [| 4 |])
         (Nx.reshape (Array.append lead [| 1 |]) none))
  else
    (* [first a n] is the first index where [a] holds along its last axis;
       [last] one past the last. *)
    let first a = Nx.argmax ~axis:(-1) (Nx.cast Nx.uint8 a) in
    let past a n =
      Nx.rsub_s (Int64.of_int n) (first (Nx.flip ~axes:[ -1 ] a))
    in
    let rows = Nx.any ~axes:[ -2; -1 ] m and cols = Nx.any ~axes:[ -3; -1 ] m in
    let b =
      Nx.stack ~axis:(-1) [ first cols; first rows; past cols w; past rows h ]
    in
    let some = Nx.unsqueeze ~axes:[ -1 ] (Nx.any ~axes:[ -1 ] cols) in
    Nx.cast dt (Nx.where some b (Nx.zeros_like b))

(* Non-maximum suppression *)

(* [above x t] is [x > t] for the real [t]: with [c] the value of [x]'s dtype
   nearest [t], [x > t] is [x >= c] where [c > t], since no value lies in [(t,
   c)], and [x > c] otherwise, since none lies in [(c, t]]. *)
let above x t =
  let c = Nx.Dtype.of_float (Nx.dtype x) t in
  if c > t then Nx.greater_equal_s x c else Nx.greater_s x c

(* The keep mask is the fixed point of [keep i = ¬∃ j earlier than i: keep j ∧
   conflict i j] from all true. A box ranked [r] is final after round [r + 1],
   so [n + 1] rounds suffice, the last changing nothing; a box Blelloch, Fineman
   and Shun's parallel greedy step decides in its step [s] is final by round
   [2s]. *)
let suppress ?classes ~iou b s =
  let fn = "Mynd.Box.nms" in
  if Nx.ndim b < 2 || Nx.dim (-1) b <> 4 then
    invalid_argf "%s: boxes have axes [...; n; 4]" fn;
  let lead = Array.sub (Nx.shape b) 0 (Nx.ndim b - 1) in
  let same name x =
    if Nx.shape x <> lead then
      invalid_argf "%s: %s's axes differ from the boxes' [...; n]" fn name
  in
  same "the scores" s;
  Option.iter (same "the classes") classes;
  let n = lead.(Array.length lead - 1) in
  (* The sort orders [0.] before [-0.]; adding [0.] makes every zero [0.], so
     equal scores tie by position. *)
  let order = Nx.argsort ~descending:true ~axis:(-1) (Nx.add_s s 0.) in
  let rank = Nx.argsort ~axis:(-1) order in
  let earlier =
    Nx.less (Nx.unsqueeze ~axes:[ -2 ] rank) (Nx.unsqueeze ~axes:[ -1 ] rank)
  in
  let c = corners fn b in
  let conflict = Nx.logical_and earlier (above (overlap c c) iou) in
  let conflict =
    match classes with
    | None -> conflict
    | Some k ->
        Nx.logical_and conflict
          (Nx.equal (Nx.unsqueeze ~axes:[ -1 ] k) (Nx.unsqueeze ~axes:[ -2 ] k))
  in
  let step (keep, (_, k)) =
    let hit = Nx.logical_and conflict (Nx.unsqueeze ~axes:[ -2 ] keep) in
    let keep' = Nx.logical_not (Nx.any ~axes:[ -1 ] hit) in
    (keep', (Nx.any (Nx.not_equal keep' keep), Nx.add_s k 1l))
  in
  let at = Nx.placement b in
  let carry = Nx.Ptree.(pair tensor (pair tensor tensor)) in
  let keep, (_, rounds) =
    Rune.iterate carry ~max:(n + 1)
      ~until:(fun (_, (changed, _)) -> Nx.logical_not changed)
      step
      ( Nx.copy (Nx.broadcast_to lead (Nx.scalar_at at Nx.bool true)),
        (Nx.scalar_at at Nx.bool true, Nx.scalar_at at Nx.int32 0l) )
  in
  (keep, rounds)

let nms ?classes ~iou b s = fst (suppress ?classes ~iou b s)
let nms_rounds = suppress
