(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The distortion polynomials, SIP's and TPV's, with their Jacobians and
   inverses.

   SIP (Shupe et al. 2005) adds [f (u, v) = Σ A_pq uᵖ v^q] and [g (u, v) = Σ
   B_pq uᵖ v^q] to pixel offsets [(u, v)] from CRPIX. TPV (the FITS registry's
   TPV convention) replaces intermediate coordinates [(ξ, η)], in degrees, by
   [ξ' = Σ PV1_k t_k (ξ, η)] and [η' = Σ PV2_k t_k (η, ξ)] over forty terms
   [t_k] of degree up to 7, four of them odd powers of [r = √(ξ² + η²)].

   Both maps are bijections where their Jacobian keeps the sign it has at the
   origin. Their inverses solve, each point its own system of two unknowns,
   with Newton's method on the Jacobian; the answer's derivative is the
   implicit one. *)

let last v = Nx.ndim v - 1
let component v k = Nx.slice (List.init (last v) (fun _ -> Nx.A) @ [ Nx.I k ]) v

let entry m i j =
  Nx.slice (List.init (Nx.ndim m - 2) (fun _ -> Nx.A) @ [ Nx.I i; Nx.I j ]) m

let pair v = (component v 0, component v 1)
let stack2 a b =
  let a, b = Nx.broadcasted a b in
  Nx.stack ~axis:(-1) [ a; b ]

(* [matrix a b c d] is [[a; b]; [c; d]] on the last two axes. *)
let matrix a b c d =
  let row x y = stack2 x y in
  let r0 = row a b and r1 = row c d in
  let r0, r1 = Nx.broadcasted r0 r1 in
  Nx.stack ~axis:(-2) [ r0; r1 ]

let det2 m =
  Nx.sub
    (Nx.mul (entry m 0 0) (entry m 1 1))
    (Nx.mul (entry m 0 1) (entry m 1 0))

(* SIP *)

(* [polynomial c u v] is [Σ c_pq uᵖ v^q] for [c] of [[...; n; n]], by Horner's
   rule in [v] inside Horner's rule in [u], and its partial derivatives. *)
let polynomial c u v =
  let n = Nx.dim (-1) c in
  let in_v p =
    let acc = ref (entry c p (n - 1)) and d = ref (Nx.zeros_like (entry c p 0)) in
    for q = n - 2 downto 0 do
      d := Nx.fma !d v !acc;
      acc := Nx.fma !acc v (entry c p q)
    done;
    (!acc, !d)
  in
  let value = ref (fst (in_v (n - 1)))
  and du = ref (Nx.zeros_like (fst (in_v 0)))
  and dv = ref (snd (in_v (n - 1))) in
  for p = n - 2 downto 0 do
    let w, wv = in_v p in
    du := Nx.fma !du u !value;
    value := Nx.fma !value u w;
    dv := Nx.fma !dv u wv
  done;
  (!value, !du, !dv)

let sip_forward a b x =
  let u, v = pair x in
  let f, _, _ = polynomial a u v and g, _, _ = polynomial b u v in
  stack2 (Nx.add u f) (Nx.add v g)

let sip_jacobian a b x =
  let u, v = pair x in
  let _, fu, fv = polynomial a u v and _, gu, gv = polynomial b u v in
  matrix (Nx.add_s fu 1.) fv gu (Nx.add_s gv 1.)

(* Pixel offsets from CRPIX in the largest mosaics. *)
let sip_scale = 1e4

let sip_inverse ?seed a b x =
  let guess =
    match seed with
    | None -> x
    | Some (ap, bp) ->
        let u, v = pair x in
        let f, _, _ = polynomial ap u v and g, _, _ = polynomial bp u v in
        stack2 (Nx.add u f) (Nx.add v g)
  in
  let guess = Nx.broadcast_to (Nx.shape (Nx.add guess x)) guess in
  Solve.lanes ~scale:sip_scale ~jacobian:(sip_jacobian a b)
    (fun w -> Nx.sub (sip_forward a b w) x)
    guess

(* TPV *)

(* TPV's terms [t_k = xⁱ yʲ rᵖ], as [(i, j, p)] for [k = 0 .. 39]. *)
let tpv_terms =
  let degree d = List.init (d + 1) (fun j -> (d - j, j, 0)) in
  Array.of_list
    (List.concat
       [
         [ (0, 0, 0) ];
         degree 1;
         [ (0, 0, 1) ];
         degree 2;
         degree 3;
         [ (0, 0, 3) ];
         degree 4;
         degree 5;
         [ (0, 0, 5) ];
         degree 6;
         degree 7;
         [ (0, 0, 7) ];
       ])

let tpv_count = Array.length tpv_terms

(* [powers x n] is [x⁰ .. xⁿ]. *)
let powers x n =
  let a = Array.make (n + 1) (Nx.ones_like x) in
  for k = 1 to n do
    a.(k) <- Nx.mul a.(k - 1) x
  done;
  a

(* [series c x y] is [Σ c_k t_k (x, y)] and its partial derivatives, [c] of
   [[...; 40]]. [r] has no derivative at the origin; its terms' derivatives
   there are taken as 0, the limit of the odd powers above 1. *)
let series c x y =
  let r2 = Nx.add (Nx.square x) (Nx.square y) in
  let origin = Nx.equal_s r2 0. in
  let r = Guard.sqrt origin (Nx.zeros_like r2) r2 in
  let px = powers x 7 and py = powers y 7 and pr = powers r 7 in
  let inv_r = Nx.where origin (Nx.zeros_like r) (Nx.recip (Nx.where origin (Nx.ones_like r) r)) in
  let value = ref (Nx.zeros_like r) and dx = ref (Nx.zeros_like r)
  and dy = ref (Nx.zeros_like r) in
  Array.iteri
    (fun k (i, j, p) ->
      let ck = component c k in
      let t = Nx.mul (Nx.mul px.(i) py.(j)) pr.(p) in
      value := Nx.fma ck t !value;
      if p > 0 then begin
        (* d rᵖ / dx = p x rᵖ⁻². *)
        let rp2 = if p >= 2 then pr.(p - 2) else inv_r in
        let d = Nx.mul_s rp2 (float p) in
        dx := Nx.fma ck (Nx.mul d x) !dx;
        dy := Nx.fma ck (Nx.mul d y) !dy
      end
      else begin
        if i > 0 then
          dx := Nx.fma ck (Nx.mul_s (Nx.mul px.(i - 1) py.(j)) (float i)) !dx;
        if j > 0 then
          dy := Nx.fma ck (Nx.mul_s (Nx.mul px.(i) py.(j - 1)) (float j)) !dy
      end)
    tpv_terms;
  (!value, !dx, !dy)

let row pv k =
  Nx.slice (List.init (Nx.ndim pv - 2) (fun _ -> Nx.A) @ [ Nx.I k; Nx.A ]) pv

let tpv_forward pv x =
  let xi, eta = pair x in
  let a, _, _ = series (row pv 0) xi eta and b, _, _ = series (row pv 1) eta xi in
  stack2 a b

let tpv_jacobian pv x =
  let xi, eta = pair x in
  let _, ax, ay = series (row pv 0) xi eta in
  let _, bx, by = series (row pv 1) eta xi in
  (* The second polynomial takes [(η, ξ)]: its [x] is [η]. *)
  matrix ax ay by bx

(* Intermediate coordinates over a wide field, in degrees. *)
let tpv_scale = 10.

let tpv_inverse pv x =
  let guess = Nx.broadcast_to (Nx.shape (Nx.add (tpv_forward pv x) x)) x in
  Solve.lanes ~scale:tpv_scale ~jacobian:(tpv_jacobian pv)
    (fun w -> Nx.sub (tpv_forward pv w) x)
    guess
