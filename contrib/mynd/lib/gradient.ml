(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

type stencil = [ `Central | `Sobel | `Scharr ]

(* [beside ~edge ~axis x] is the pixels before and after each pixel along
   [axis], read under [edge]. *)
let beside ~edge ~axis x =
  let n = Nx.dim axis x in
  let e = Nx.extend ~axis edge (1, 1) x in
  (cut ~axis 0 n e, cut ~axis 2 n e)

(* The centre's weight and each outer one's. The outer weight multiplies the sum
   of its two pixels, which a flip of the image leaves as it is, so a flipped
   image smooths to the flipped result bit for bit. *)
let across = function
  | `Central -> None
  | `Sobel -> Some (0.5, 0.25)
  | `Scharr -> Some (0.625, 0.1875)

let smooth ~edge ~axis s x =
  match across s with
  | None -> x
  | Some (centre, outer) ->
      let before, after = beside ~edge ~axis x in
      Nx.add (Nx.mul_s (Nx.add before after) outer) (Nx.mul_s x centre)

(* Half the difference of the pixels after and before. A flip negates it
   exactly, since [a - b] is [-(b - a)]. *)
let difference ~edge ~axis x =
  let before, after = beside ~edge ~axis x in
  Nx.mul_s (Nx.sub after before) 0.5

let gradient ~edge s x =
  let fn = "Mynd.Filter.gradient" in
  image fn x;
  let edge = (edge :> Nx.extension) in
  check_extension fn ~edge x;
  let go x =
    ( difference ~edge ~axis:(-2) (smooth ~edge ~axis:(-3) s x),
      difference ~edge ~axis:(-3) (smooth ~edge ~axis:(-2) s x) )
  in
  if not (narrow x) then go x
  else
    let dt = Nx.dtype x in
    let gx, gy = go (Nx.cast Nx.float32 x) in
    (Nx.cast dt gx, Nx.cast dt gy)

(* Canny's interpolation over the ring of eight neighbours. The step [u] is [g]
   over its larger absolute component. Where [|gx| >= |gy|] ([x] major), the
   point [p + u] lies in the column [j + sign gx], between the pixel of [p]'s
   row, the axis pixel, and the diagonal one in row [i + sign gy], at [w = |gy|
   / |gx|] from the first; otherwise in the row [i + sign gy], at [w = |gx| /
   |gy|]. At [w = 0] or [1] it is one pixel and reads no other, so a gradient
   along an axis or a diagonal compares exactly the neighbour it points to. The
   point is outside where a pixel it weighs is: the axis pixel, or the diagonal
   one where [w > 0]. The strengths, padded by a ring of zeros, and an [inside]
   mask, padded by one of [false], are read at the nine offsets and picked per
   pixel by the signs. *)
let maxima gx gy =
  let m = Nx.hypot gx gy in
  let r = Nx.ndim m in
  let h = Nx.dim (-3) m and w = Nx.dim (-2) m in
  let pads =
    Array.init r (fun k -> if k = r - 3 || k = r - 2 then (1, 1) else (0, 0))
  in
  let ax = Nx.abs gx and ay = Nx.abs gy in
  let major = Nx.greater_equal ax ay in
  let none = Nx.zeros_like major in
  let mp = Nx.pad pads 0. m in
  let ip = Nx.pad pads false (Nx.logical_not none) in
  let at p di dj = cut ~axis:(-2) (1 + dj) w (cut ~axis:(-3) (1 + di) h p) in
  let wt = Nx.where major (Nx.div ay ax) (Nx.div ax ay) in
  let zero = Nx.zeros_like gx in
  let neg_x = Nx.less gx zero and pos_x = Nx.greater gx zero in
  let neg_y = Nx.less gy zero and pos_y = Nx.greater gy zero in
  (* [row p di (nj, pj)] reads [p] in row [di] at column [-1] where [nj], [1]
     where [pj] and [0] elsewhere; [pick] picks the row so too. *)
  let row p di (nj, pj) =
    Nx.where nj (at p di (-1)) (Nx.where pj (at p di 1) (at p di 0))
  in
  let pick p (ni, pi) cols =
    Nx.where ni (row p (-1) cols) (Nx.where pi (row p 1 cols) (row p 0 cols))
  in
  let point rows cols =
    let axis p = Nx.where major (row p 0 cols) (pick p rows (none, none)) in
    let diagonal p = pick p rows cols in
    let a = axis mp and b = diagonal mp in
    (* Two ring pixels that tie, infinite ones included, give their strength; an
       infinite one beside a finite one gives [∞], their sum, since strengths
       are not negative; others [a + w (b - a)]. A NaN stays. *)
    let lerp =
      Nx.where (Nx.equal a b) a
        (Nx.where
           (Nx.logical_or (Nx.isinf a) (Nx.isinf b))
           (Nx.add a b)
           (Nx.add a (Nx.mul wt (Nx.sub b a))))
    in
    let v =
      Nx.where (Nx.equal_s wt 0.) a (Nx.where (Nx.equal_s wt 1.) b lerp)
    in
    let outside =
      Nx.logical_or
        (Nx.logical_not (axis ip))
        (Nx.logical_and (Nx.greater_s wt 0.) (Nx.logical_not (diagonal ip)))
    in
    (v, outside)
  in
  let ahead, out_ahead = point (neg_y, pos_y) (neg_x, pos_x) in
  let behind, out_behind = point (pos_y, neg_y) (pos_x, neg_x) in
  Nx.logical_and (Nx.greater_s m 0.)
    (Nx.logical_and
       (Nx.logical_or out_behind (Nx.greater m behind))
       (Nx.logical_or out_ahead (Nx.greater_equal m ahead)))

let gradient_maxima (gx, gy) =
  let fn = "Mynd.Filter.gradient_maxima" in
  image fn gx;
  if Nx.shape gx <> Nx.shape gy then
    invalid_argf "%s: gradients of shapes [%s] and [%s]" fn
      (dims (Nx.shape gx))
      (dims (Nx.shape gy));
  if narrow gx then maxima (Nx.cast Nx.float32 gx) (Nx.cast Nx.float32 gy)
  else maxima gx gy
