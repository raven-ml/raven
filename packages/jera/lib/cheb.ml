(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type extension = Bounded | Hold | Polynomial

let locate fn e breaks x =
  let n = Nx.dim 0 breaks - 1 in
  let first = Nx.get [ 0 ] breaks and last = Nx.get [ n ] breaks in
  (match e with
  | Bounded ->
      let inside =
        Nx.logical_or (Nx.isnan x)
          (Nx.logical_and (Nx.greater_equal x first) (Nx.less_equal x last))
      in
      Nx.check
        Nx.Ptree.(pair tensor (pair tensor tensor))
        inside
        (x, (first, last))
        (fun i (x, (first, last)) ->
          Invalid_argument
            (Printf.sprintf
               "%s: the point at [%d] is %g, outside the domain [%g, %g]" fn
               i.(0) (Nx.item [] x) (Nx.item [] first) (Nx.item [] last)))
  | Hold | Polynomial -> ());
  let x =
    match e with
    | Hold -> Nx.minimum (Nx.maximum x first) last
    | Bounded | Polynomial -> x
  in
  let i = Nx.sub_s (Nx.searchsorted ~side:`Left breaks x) 1L in
  let i = Nx.clamp ~min:0L ~max:(Int64.of_int (n - 1)) i in
  let lo = Nx.take ~indices:i breaks
  and hi = Nx.take ~indices:(Nx.add_s i 1L) breaks in
  (i, Nx.sub_s (Nx.div (Nx.mul_s (Nx.sub x lo) 2.) (Nx.sub hi lo)) 1.)

let nodes n =
  if n = 0 then [| 0. |]
  else
    (* sin is odd and exact at 0, so the points are symmetric. *)
    Array.init (n + 1) (fun j ->
        Float.sin (Float.pi *. float ((2 * j) - n) /. float (2 * n)))

(* [m] of [r] rows applied along axis 1 of [c] of shape [[p; k] @ rest]. *)
let along m c =
  let shape = Nx.shape c in
  let rest = Array.sub shape 2 (Array.length shape - 2) in
  let r = Array.length m and k = shape.(1) in
  let m = Nx.create (Nx.dtype c) [| r; k |] (Array.concat (Array.to_list m)) in
  let flat = Nx.reshape [| shape.(0); k; Array.fold_left ( * ) 1 rest |] c in
  Nx.reshape (Array.concat [ [| shape.(0); r |]; rest ]) (Nx.matmul m flat)

(* [w] of shape [[p]] reshaped to broadcast against [[p; k] @ rest]. *)
let per_piece c w =
  Nx.reshape (Array.append [| Nx.dim 0 w |] (Array.make (Nx.ndim c - 1) 1)) w

(* c_k = (2/n) Σ''_j f_j T_k(x_j), with x_j = −cos(πj/n) and the terms of the
   first and last node halved, then c_0 and c_n halved. *)
let fit v =
  let n = Nx.dim 1 v - 1 in
  if n = 0 then v
  else
    let half i = if i = 0 || i = n then 0.5 else 1. in
    let m =
      Array.init (n + 1) (fun k ->
          Array.init (n + 1) (fun j ->
              let cos =
                Float.cos (Float.pi *. float (k * (n - j)) /. float n)
              in
              2. /. float n *. half j *. half k *. cos))
    in
    along m v

(* On [−1, 1]: c3 = (d0 + d1 − (y1 − y0)) / 16, c2 = (d1 − d0) / 8, c1 = (y1 −
   y0) / 2 − c3 and c0 = (y0 + y1) / 2 − c2, from the cubic's monomial form and
   u² = (T0 + T2) / 2, u³ = (3 T1 + T3) / 4. *)
let hermite y0 y1 d0 d1 =
  let open Nx in
  let dy = sub y1 y0 in
  let c3 = div_s (sub (add d0 d1) dy) 16. in
  let c2 = div_s (sub d1 d0) 8. in
  let c1 = sub (div_s dy 2.) c3 in
  let c0 = sub (div_s (add y0 y1) 2.) c2 in
  stack ~axis:1 [ c0; c1; c2; c3 ]

let clenshaw u g =
  let n = Nx.dim 1 g - 1 in
  let g = Nx.moveaxis 1 0 g in
  let u =
    Nx.reshape (Array.append [| Nx.dim 0 u |] (Array.make (Nx.ndim g - 2) 1)) u
  in
  let c k = Nx.get [ k ] g in
  if n = 0 then c 0
  else
    let two_u = Nx.mul_s u 2. in
    (* b_k = c_k + 2u b_(k+1) − b_(k+2), and the value c_0 + u b_1 − b_2. *)
    let rec go k b1 b2 =
      if k = 0 then
        let v = Nx.add (c 0) (Nx.mul u b1) in
        Option.fold ~none:v ~some:(Nx.sub v) b2
      else
        let b = Nx.add (c k) (Nx.mul two_u b1) in
        let b = Option.fold ~none:b ~some:(Nx.sub b) b2 in
        go (k - 1) b (Some b1)
    in
    go (n - 1) (c n) None

(* d/du of Σ c_j T_j is Σ_k c'_k T_k with c'_k = (2 / (1 + δ_k0)) Σ_(j > k, j −
   k odd) j c_j, and d/dx = (2 / w) d/du. *)
let derivative widths c =
  let n = Nx.dim 1 c - 1 in
  if n = 0 then Nx.zeros_like c
  else
    let m =
      Array.init n (fun k ->
          Array.init (n + 1) (fun j ->
              if j > k && (j - k) mod 2 = 1 then
                float (2 * j) /. if k = 0 then 2. else 1.
              else 0.))
    in
    (* An empty piece, which no point reaches, has a zero derivative. *)
    let empty = Nx.equal widths (Nx.zeros_like widths) in
    let scale =
      Nx.div (Nx.full_like widths 2.)
        (Nx.where empty (Nx.ones_like widths) widths)
    in
    Nx.mul (along m c)
      (per_piece c (Nx.where empty (Nx.zeros_like scale) scale))

(* ∫T_0 = T_1, ∫T_1 = T_2 / 4 and ∫T_k = T_(k+1) / 2(k+1) − T_(k−1) / 2(k−1), up
   to constants, and dx = (w / 2) du. Each piece's constant then makes the
   antiderivative continuous and zero at the first break. *)
let integral widths c =
  let n = Nx.dim 1 c - 1 in
  let m = Array.make_matrix (n + 2) (n + 1) 0. in
  m.(1).(0) <- 1.;
  for k = 1 to n do
    m.(k + 1).(k) <- m.(k + 1).(k) +. (1. /. float (2 * (k + 1)));
    if k >= 2 then m.(k - 1).(k) <- m.(k - 1).(k) -. (1. /. float (2 * (k - 1)))
  done;
  let c = Nx.mul (along m c) (per_piece c (Nx.div_s widths 2.)) in
  let sign k = if k mod 2 = 0 then 1. else -1. in
  let left = along [| Array.init (n + 2) sign |] c in
  let right = along [| Array.make (n + 2) 1. |] c in
  let pieces = Nx.dim 0 c in
  let gain = Nx.sub right left in
  let start =
    Nx.concatenate ~axis:0
      [
        Nx.zeros_like (Nx.slice [ Nx.R (0, 1) ] gain);
        Nx.cumsum ~axis:0 (Nx.slice [ Nx.R (0, pieces - 1) ] gain);
      ]
  in
  let c0 =
    Nx.add (Nx.slice [ Nx.R (0, pieces); Nx.R (0, 1) ] c) (Nx.sub start left)
  in
  Nx.concatenate ~axis:1
    [ c0; Nx.slice [ Nx.R (0, pieces); Nx.R (1, n + 2) ] c ]
