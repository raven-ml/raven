(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rules *)

module Rule = struct
  (* Nodes in increasing order on [-1, 1] and their weights, in float64. *)
  type 'k t = { x : float array; w : float array }

  (* Double-word arithmetic: [hi + lo] with [|lo| <= ulp hi / 2], so the nodes
     and weights below are computed to about 32 digits and rounded once to
     float64. *)
  type dd = { hi : float; lo : float }

  let two_sum a b =
    let s = a +. b in
    let v = s -. a in
    (s, a -. (s -. v) +. (b -. v))

  let norm hi lo =
    let s, e = two_sum hi lo in
    { hi = s; lo = e }

  let dd x = { hi = x; lo = 0. }

  let add a b =
    let s, e = two_sum a.hi b.hi in
    norm s (e +. a.lo +. b.lo)

  let neg a = { hi = -.a.hi; lo = -.a.lo }
  let sub a b = add a (neg b)

  let mul a b =
    let p = a.hi *. b.hi in
    norm p (Float.fma a.hi b.hi (-.p) +. (a.hi *. b.lo) +. (a.lo *. b.hi))

  let scale a k = mul a (dd k)

  let div a b =
    let q = a.hi /. b.hi in
    let r = sub a (scale b q) in
    add (dd q) (dd ((r.hi +. r.lo) /. b.hi))

  (* P_(k−1)(x) and P_k(x) from P_(k−2) and P_(k−1). *)
  let next k x (p0, p1) =
    let k = float k in
    ( p1,
      div (sub (scale (mul x p1) ((2. *. k) -. 1.)) (scale p0 (k -. 1.))) (dd k)
    )

  (* P_(n−1)(x) and P_n(x). *)
  let legendre n x =
    let rec go k ps = if k > n then ps else go (k + 1) (next k x ps) in
    go 2 (dd 1., x)

  (* The Christoffel number at [x], [1 / Σ_(k<n) (k + 1/2) P_k(x)²], a sum of
     positive terms. *)
  let christoffel n x =
    let rec go k (p0, p1) sum =
      if k = n then sum
      else
        let sum = add sum (scale (mul p1 p1) (float k +. 0.5)) in
        go (k + 1) (next (k + 1) x (p0, p1)) sum
    in
    div (dd 1.) (go 1 (dd 1., x) (dd 0.5))

  (* Newton's method from Tricomi's estimate of the i-th largest root: [P_n /
     P_n'] with [P_n' = n (x P_n − P_(n−1)) / (x² − 1)]. Thirty steps reach the
     root to the double word's precision from any estimate this good. *)
  let root n i =
    let x =
      ref (dd (Float.cos (Float.pi *. (float i +. 0.75) /. (float n +. 0.5))))
    in
    for _ = 1 to 30 do
      let pm, p = legendre n !x in
      let slope =
        div (scale (sub (mul !x p) pm) (float n)) (sub (mul !x !x) (dd 1.))
      in
      if slope.hi <> 0. then x := sub !x (div p slope)
    done;
    (!x.hi, (christoffel n !x).hi)

  (* A symmetric rule from its nodes in [0, 1), largest first, and the center's
     weight when it has one. *)
  let symmetric positive center =
    let half = Array.length positive in
    let n = (2 * half) + Option.fold ~none:0 ~some:(fun _ -> 1) center in
    let x = Array.make n 0. and w = Array.make n 0. in
    Array.iteri
      (fun i (xi, wi) ->
        x.(i) <- -.xi;
        w.(i) <- wi;
        x.(n - 1 - i) <- xi;
        w.(n - 1 - i) <- wi)
      positive;
    Option.iter (fun wc -> w.(half) <- wc) center;
    { x; w }

  let gauss n =
    if n < 1 then
      invalid_arg (Printf.sprintf "Jera.Quad.Rule.gauss: n = %d is below 1" n);
    let positive = Array.init (n / 2) (root n) in
    let center = if n mod 2 = 1 then Some (snd (root n (n / 2))) else None in
    symmetric positive center

  (* QUADPACK's qk15 and qk21 (Piessens, de Doncker-Kapenga, Überhuber and
     Kahaner, 1983): the Kronrod nodes in (0, 1), largest first, and their
     weights, then the center's weight. *)
  let kronrod15_x =
    [|
      0.991455371120812639206854697526329;
      0.949107912342758524526189684047851;
      0.864864423359769072789712788640926;
      0.741531185599394439863864773280788;
      0.586087235467691130294144845693013;
      0.405845151377397166906606412076961;
      0.207784955007898467600689403773245;
    |]

  let kronrod15_w =
    [|
      0.022935322010529224963732008058970;
      0.063092092629978553290700663189204;
      0.104790010322250183839876322541518;
      0.140653259715525918745189590510238;
      0.169004726639267902826583426598550;
      0.190350578064785409913256402421014;
      0.204432940075298892414161999234649;
      0.209482141084727828012999174891714;
    |]

  let kronrod21_x =
    [|
      0.995657163025808080735527280689003;
      0.973906528517171720077964012084452;
      0.930157491355708226001207180059508;
      0.865063366688984510732096688423493;
      0.780817726586416897063717578345042;
      0.679409568299024406234327365114874;
      0.562757134668604683339000099272694;
      0.433395394129247190799265943165784;
      0.294392862701460198131126603103866;
      0.148874338981631210884826001129720;
    |]

  let kronrod21_w =
    [|
      0.011694638867371874278064396062192;
      0.032558162307964727478818972459390;
      0.054755896574351996031381300244580;
      0.075039674810919952767043140916190;
      0.093125454583697605535065465083366;
      0.109387158802297641899210590325805;
      0.123491976262065851077958109831074;
      0.134709217311473325928054001771707;
      0.142775938577060080797094273138717;
      0.147739104901338491374841515972068;
      0.149445554002916905664936468389821;
    |]

  let kronrod n =
    let x, w =
      match n with
      | 7 -> (kronrod15_x, kronrod15_w)
      | 10 -> (kronrod21_x, kronrod21_w)
      | n ->
          invalid_arg
            (Printf.sprintf "Jera.Quad.Rule.kronrod: n = %d is neither 7 nor 10"
               n)
    in
    let half = Array.length x in
    symmetric (Array.init half (fun i -> (x.(i), w.(i)))) (Some w.(half))

  let nodes r dtype = (Num.constant dtype r.x, Num.constant dtype r.w)
end

(* Ranges *)

module Range = struct
  type 'b t =
    | Finite of (float, 'b) Nx.t * (float, 'b) Nx.t
    | From of (float, 'b) Nx.t
    | Line of (float, 'b) Nx.t

  let v a b =
    match Nx.broadcast_arrays [ a; b ] with
    | [ a; b ] -> Finite (a, b)
    | _ -> assert false

  let from a = From a
  let line c = Line c
  let shape = function Finite (a, _) | From a | Line a -> Nx.shape a
  let dtype = function Finite (a, _) | From a | Line a -> Nx.dtype a
end

type 'b integrand = (float, 'b) Nx.t -> (float, 'b) Nx.t

(* Formulas *)

(* [c] reshaped to [[m] @ ones], to broadcast against lanes of [rank] axes. *)
let along rank c =
  Nx.reshape (Array.append [| Nx.dim 0 c |] (Array.make rank 1)) c

let call fn f points =
  let y = f points in
  if Nx.shape y <> Nx.shape points then
    invalid_arg
      (Printf.sprintf
         "%s: the integrand returned shape %s for points of shape %s" fn
         (Num.shape (Nx.shape y))
         (Num.shape (Nx.shape points)));
  y

(* The infinite ranges' changes of variable: each node's offset from the range's
   anchor and its weight times the change's derivative, in float64. *)
let half_line (r : _ Rule.t) =
  ( Array.map (fun u -> (1. +. u) /. (1. -. u)) r.x,
    Array.mapi (fun i u -> r.w.(i) *. 2. /. ((1. -. u) *. (1. -. u))) r.x )

let whole_line (r : _ Rule.t) =
  ( Array.map (fun u -> u /. (1. -. (u *. u))) r.x,
    Array.mapi
      (fun i u ->
        let d = 1. -. (u *. u) in
        r.w.(i) *. (1. +. (u *. u)) /. (d *. d))
      r.x )

let sum fn r f range =
  let dtype = Range.dtype range in
  let rank = Array.length (Range.shape range) in
  let weighted anchor (offsets, weights) =
    let points = Nx.add anchor (along rank (Num.constant dtype offsets)) in
    let y = call fn f points in
    Nx.sum ~axes:[ 0 ] (Nx.mul (along rank (Num.constant dtype weights)) y)
  in
  match range with
  | Range.Finite (a, b) ->
      let mid = Nx.div_s (Nx.add a b) 2. and half = Nx.div_s (Nx.sub b a) 2. in
      let x = along rank (Num.constant dtype r.Rule.x) in
      let y = call fn f (Nx.add mid (Nx.mul half x)) in
      let w = along rank (Num.constant dtype r.w) in
      Nx.mul half (Nx.sum ~axes:[ 0 ] (Nx.mul w y))
  | From a -> weighted a (half_line r)
  | Line c -> weighted c (whole_line r)

let fixed r f range = sum "Jera.Quad.fixed" r f range

let cumulative r f knots =
  let fn = "Jera.Quad.cumulative" in
  if Nx.ndim knots = 0 || Nx.dim 0 knots = 0 then
    invalid_arg
      (Printf.sprintf "%s: knots must hold at least one knot, got shape %s" fn
         (Num.shape (Nx.shape knots)));
  let n = Nx.dim 0 knots in
  let zero = Nx.zeros_like (Nx.slice [ Nx.R (0, 1) ] knots) in
  if n = 1 then zero
  else
    let a = Nx.slice [ Nx.R (0, n - 1) ] knots
    and b = Nx.slice [ Nx.R (1, n) ] knots in
    let parts = sum fn r f (Range.v a b) in
    Nx.concatenate ~axis:0 [ zero; Nx.cumsum ~axis:0 parts ]
