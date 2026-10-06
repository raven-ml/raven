(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'b ends =
  [ `Natural | `Not_a_knot | `Clamped of (float, 'b) Nx.t * (float, 'b) Nx.t ]

let rows i j v = Nx.slice [ Nx.R (i, j) ] v

(* [v] of shape [[k]] reshaped to broadcast against [y] of shape [[k] @
   rest]. *)
let column y v =
  Nx.reshape (Array.append [| Nx.dim 0 v |] (Array.make (Nx.ndim y - 1) 1)) v

(* Interval widths, of shape [[n − 1]], and secant slopes, of shape [[n − 1] @
   rest]. *)
let secants x y =
  let n = Nx.dim 0 x in
  let h = Nx.sub (rows 1 n x) (rows 0 (n - 1) x) in
  let dy = Nx.sub (rows 1 n y) (rows 0 (n - 1) y) in
  (h, Nx.div dy (column dy h))

let hermite x y m =
  let n = Nx.dim 0 x in
  let h, _ = secants x y in
  let half = column (rows 0 (n - 1) y) (Nx.div_s h 2.) in
  Cheb.hermite
    (rows 0 (n - 1) y)
    (rows 1 n y)
    (Nx.mul half (rows 0 (n - 1) m))
    (Nx.mul half (rows 1 n m))

let linear x y =
  let n = Nx.dim 0 x in
  let y0 = rows 0 (n - 1) y and y1 = rows 1 n y in
  Nx.stack ~axis:1 [ Nx.div_s (Nx.add y0 y1) 2.; Nx.div_s (Nx.sub y1 y0) 2. ]

let cat l = Nx.concatenate ~axis:0 l
let one v = Nx.unsqueeze ~axes:[ 0 ] v

(* The slopes m of a cubic spline solve, at each interior knot i, h_i m_(i−1) +
   2 (h_(i−1) + h_i) m_i + h_(i−1) m_(i+1) = 3 (h_i δ_(i−1) + h_(i−1) δ_i), with
   h the widths and δ the secants, and one row per end. *)
let cubic ends x y =
  let n = Nx.dim 0 x in
  let h, d = secants x y in
  let slopes =
    if n = 2 then cat [ d; d ]
    else
      let col = column (rows 0 (n - 2) d) in
      let h0 = rows 0 (n - 2) h and h1 = rows 1 (n - 1) h in
      let d0 = rows 0 (n - 2) d and d1 = rows 1 (n - 1) d in
      let sub = h1 and diag = Nx.mul_s (Nx.add h0 h1) 2. and sup = h0 in
      let rhs =
        Nx.mul_s (Nx.add (Nx.mul (col h1) d0) (Nx.mul (col h0) d1)) 3.
      in
      let const v = Nx.full_like (rows 0 1 h) v in
      match ends with
      | `Natural ->
          let first = Nx.mul_s (rows 0 1 d) 3.
          and last = Nx.mul_s (rows (n - 2) (n - 1) d) 3. in
          Tridiag.solve
            ~sub:(cat [ const 0.; sub; const 1. ])
            ~diag:(cat [ const 2.; diag; const 2. ])
            ~sup:(cat [ const 1.; sup; const 0. ])
            (cat [ first; rhs; last ])
      | `Clamped (s0, s1) ->
          Tridiag.solve
            ~sub:(cat [ const 0.; sub; const 0. ])
            ~diag:(cat [ const 1.; diag; const 1. ])
            ~sup:(cat [ const 0.; sup; const 0. ])
            (cat [ one s0; rhs; one s1 ])
      | `Not_a_knot when n = 3 ->
          (* The parabola through the three knots. *)
          let k =
            Nx.div
              (Nx.sub (rows 1 2 d) (rows 0 1 d))
              (col (Nx.add (rows 0 1 h) (rows 1 2 h)))
          in
          let hk w = Nx.mul (col w) k in
          let d0 = rows 0 1 d in
          cat
            [
              Nx.sub d0 (hk (rows 0 1 h));
              Nx.add d0 (hk (rows 0 1 h));
              Nx.add d0 (hk (Nx.add (rows 0 1 h) (Nx.mul_s (rows 1 2 h) 2.)));
            ]
      | `Not_a_knot ->
          (* The third derivative is continuous at knots 1 and n − 2. The end
             rows, h_1 m_0 + (h_0 + h_1) m_1 = r_0 and its mirror, are
             subtracted from their neighbours' first, which leaves a diagonally
             dominant system in m_1 … m_(n−2). *)
          let w k = rows k (k + 1) h and s k = rows k (k + 1) d in
          let e0 = Nx.add (w 0) (w 1) and e1 = Nx.add (w (n - 3)) (w (n - 2)) in
          let r0 =
            Nx.div
              (Nx.add
                 (Nx.mul
                    (col (Nx.mul (Nx.add (w 0) (Nx.mul_s e0 2.)) (w 1)))
                    (s 0))
                 (Nx.mul (col (Nx.square (w 0))) (s 1)))
              (col e0)
          in
          let r1 =
            Nx.div
              (Nx.add
                 (Nx.mul (col (Nx.square (w (n - 2)))) (s (n - 3)))
                 (Nx.mul
                    (col
                       (Nx.mul
                          (Nx.add (Nx.mul_s e1 2.) (w (n - 2)))
                          (w (n - 3))))
                    (s (n - 2))))
              (col e1)
          in
          let m = n - 2 in
          let first = Nx.sub (rows 0 1 rhs) r0
          and last = Nx.sub (rows (m - 1) m rhs) r1 in
          let inner =
            Tridiag.solve
              ~sub:(cat [ const 0.; rows 1 m sub ])
              ~diag:(cat [ e0; rows 1 (m - 1) diag; e1 ])
              ~sup:(cat [ rows 0 (m - 1) sup; const 0. ])
              (cat [ first; rows 1 (m - 1) rhs; last ])
          in
          let m0 =
            Nx.div (Nx.sub r0 (Nx.mul (col e0) (rows 0 1 inner))) (col (w 1))
          in
          let mn =
            Nx.div
              (Nx.sub r1 (Nx.mul (col e1) (rows (m - 1) m inner)))
              (col (w (n - 3)))
          in
          cat [ m0; inner; mn ]
  in
  hermite x y slopes

(* Steffen (1990): at an interior knot the slope is (sign δ_(i−1) + sign δ_i)
   min(|δ_(i−1)|, |δ_i|, |p_i| / 2), p_i the parabola's slope, zero where the
   secants change sign; at the ends, the end secants. *)
let steffen x y =
  let n = Nx.dim 0 x in
  let h, d = secants x y in
  let slopes =
    if n = 2 then cat [ d; d ]
    else
      let col = column (rows 0 (n - 2) d) in
      let h0 = rows 0 (n - 2) h and h1 = rows 1 (n - 1) h in
      let d0 = rows 0 (n - 2) d and d1 = rows 1 (n - 1) d in
      let p =
        Nx.div
          (Nx.add (Nx.mul d0 (col h1)) (Nx.mul d1 (col h0)))
          (col (Nx.add h0 h1))
      in
      let bound =
        Nx.minimum (Nx.minimum (Nx.abs d0) (Nx.abs d1)) (Nx.div_s (Nx.abs p) 2.)
      in
      let inner = Nx.mul (Nx.add (Nx.sign d0) (Nx.sign d1)) bound in
      cat [ rows 0 1 d; inner; rows (n - 2) (n - 1) d ]
  in
  hermite x y slopes
