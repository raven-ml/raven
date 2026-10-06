(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rules *)

module Rule = struct
  (* Nodes in increasing order on [-1, 1] and their weights, in float64. *)
  type -'k t = { name : string; x : float array; w : float array }

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
     positive terms. The usual [2 / ((1 − x²) P_n'(x)²)] loses digits to [1 −
     x²] near the ends: about 20 ulps at ten points in float64, where the
     weights must be correctly rounded. *)
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
  let symmetric name positive center =
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
    { name; x; w }

  let gauss n =
    if n < 1 then
      invalid_arg (Printf.sprintf "Jera.Quad.Rule.gauss: n = %d is below 1" n);
    let positive = Array.init (n / 2) (root n) in
    let center = if n mod 2 = 1 then Some (snd (root n (n / 2))) else None in
    symmetric (Printf.sprintf "gauss %d" n) positive center

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
    symmetric
      (Printf.sprintf "kronrod %d" n)
      (Array.init half (fun i -> (x.(i), w.(i))))
      (Some w.(half))

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

module Box = struct
  type 'b t = { lo : (float, 'b) Nx.t; hi : (float, 'b) Nx.t }

  let v lo hi =
    let fail () =
      invalid_arg
        (Printf.sprintf
           "Jera.Quad.Box.v: the corners have shapes %s and %s; they need to \
            broadcast to one shape lanes @ [d]"
           (Num.shape (Nx.shape lo))
           (Num.shape (Nx.shape hi)))
    in
    if Nx.ndim lo = 0 || Nx.ndim hi = 0 then fail ();
    match Nx.broadcast_arrays [ lo; hi ] with
    | [ lo; hi ] -> { lo; hi }
    | _ -> assert false
    | exception Invalid_argument _ -> fail ()
end

type 'b integrand = (float, 'b) Nx.t -> (float, 'b) Nx.t

(* The facts a report prints of a range. *)
let range_facts : _ Range.t -> Solution.fact list = function
  | Range.Finite (a, b) -> [ Fact ("a", a); Fact ("b", b) ]
  | From a -> [ Fact ("a", a) ]
  | Line c -> [ Fact ("c", c) ]

(* What to change for an integral that stopped short. *)
let quad_fix tol ~budget ~stalled (st : Solution.status) _ =
  match st with
  | Budget_spent -> budget
  | Stalled -> stalled ^ Tol.zero_hint tol
  | Not_finite ->
      "The integrand is not finite at a point it was given: Quad.tanh_sinh \
       keeps away from the ends of a range."
  | Converged | Not_bracketed -> ""

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

(* Adaptive *)

(* A Kronrod rule's embedded Gauss weights at its nodes: the Gauss nodes are the
   Kronrod nodes of odd position. *)
let embedded (r : _ Rule.t) =
  let g = Rule.gauss ((Array.length r.x - 1) / 2) in
  let w = Array.make (Array.length r.x) 0. in
  Array.iteri (fun i gw -> w.((2 * i) + 1) <- gw) g.w;
  w

(* A range as the image of t in [0, 1]: the point and dx/dt. *)
let image range t =
  match range with
  | Range.Finite (a, b) ->
      let w = Nx.sub b a in
      (Nx.add a (Nx.mul w t), Nx.broadcast_to (Nx.shape t) w)
  | From a ->
      (* x = a + t / (1 − t), the half-line map of u = 2t − 1. *)
      let s = Nx.rsub_s 1. t in
      (Nx.add a (Nx.div t s), Nx.recip (Nx.square s))
  | Line c ->
      (* x = c + u / (1 − u²) with u = 2t − 1. *)
      let u = Nx.sub_s (Nx.mul_s t 2.) 1. in
      let d = Nx.rsub_s 1. (Nx.square u) in
      ( Nx.add c (Nx.div u d),
        Nx.div (Nx.mul_s (Nx.add_s (Nx.square u) 1.) 2.) (Nx.square d) )

let rules fn (r : _ Rule.t) gauss f range (t0, t1) =
  let dtype = Nx.dtype t0 in
  let rank = Nx.ndim t0 in
  let along c =
    Nx.reshape
      (Array.append [| Array.length c |] (Array.make rank 1))
      (Num.constant dtype c)
  in
  let half = Nx.div_s (Nx.sub t1 t0) 2. in
  let t =
    Nx.add
      (Nx.unsqueeze ~axes:[ 0 ] t0)
      (Nx.mul
         (Nx.unsqueeze ~axes:[ 0 ] half)
         (along (Array.map (fun x -> 1. +. x) r.x)))
  in
  let x, jac = image range t in
  let y = Nx.mul (call fn f x) (Nx.mul jac (Nx.unsqueeze ~axes:[ 0 ] half)) in
  ( Nx.sum ~axes:[ 0 ] (Nx.mul (along r.w) y),
    Nx.sum ~axes:[ 0 ] (Nx.mul (along gauss) y) )

let adaptive r ~tol ~budget f range =
  let fn = "Jera.Quad.adaptive" in
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  let gauss = embedded r in
  let m = Array.length r.Rule.x in
  let dtype = Range.dtype range in
  let lanes = Range.shape range in
  let detached =
    match range with
    | Range.Finite (a, b) -> Range.Finite (Rune.detach a, Rune.detach b)
    | From a -> From (Rune.detach a)
    | Line c -> Line (Rune.detach c)
  in
  let search x = Rune.detach (f x) in
  (* Pieces of one axis, [[k] @ lanes @ [1]]. *)
  let pieces f range level index =
    let flat v = Nx.squeeze ~axes:[ Nx.ndim v - 1 ] v in
    rules fn r gauss f range
      (Partition.fractions dtype (flat level) (flat index))
  in
  let p =
    Partition.refine Nx.Ptree.tensor ~budget ~lanes ~dims:1 ~cost:m
      ~evaluate:(fun level index ->
        let k, g = pieces search detached level index in
        (k, Nx.abs (Nx.sub k g), Nx.zeros Nx.int64 (Nx.shape k)))
      ~point:(fun _ t -> fst (image detached t))
      ~verdict:(fun p live ->
        let total = Partition.sum live p.data in
        ( Nx.logical_not (Nx.isfinite total),
          Elementwise.accepted tol ~e:(Partition.sum live p.error) ~y:total ))
  in
  let ok = Nx.equal_s p.status (Solution.code Converged) in
  (* The answer: the rule over the final partition, tracked. *)
  let total =
    Partition.integrate p (fun level index -> fst (pieces f range level index))
  in
  let live = Partition.in_use p in
  let estimate = Partition.sum live p.data
  and error = Partition.sum live p.error in
  Solution.v ~fn
    ~settings:
      (Format.asprintf "rule %s, tol %a, budget %d" r.name Tol.pp tol budget)
    ~spent:{ used = p.used; unit = "pieces"; budget }
    ~fix:
      (quad_fix tol
         ~budget:
           "Raise the budget or loosen tol; an endpoint singularity converges \
            in fewer evaluations with Quad.tanh_sinh."
         ~stalled:
           "A piece cannot be bisected further: the integrand has a \
            singularity inside the range; split the range there, or use \
            Quad.tanh_sinh for one at an end.")
    ~value:(Nx.where ok total estimate)
    ~error ~status:p.status ~evaluations:p.evaluations
    ~facts:
      (range_facts range
      @ [ Fact ("estimate", estimate); Fact ("error", error) ])
    ()

(* Double-exponential rules *)

(* The schedule of a double-exponential rule: its nodes by level, each level's
   nodes padded with nodes of weight zero to whole chunks. Level 0 holds the
   integers t of the truncated range, level k ≥ 1 the odd multiples of 2^-k;
   together, levels 0 to k are the nodes of step 2^-k. A node is its side of a
   finite range, its offset from the anchor or the end, which is a fraction of
   the half-width for a finite range, its weight, and whether it lies in the
   last unit of the truncation. *)
type schedule = {
  side : float array array;
  offset : float array array;
  weight : float array array;
  outer : float array array;
  level : int array;
  last : bool array;
  count : int array;
  finest : int;
}

let de_chunk = 64

let schedule kind dtype =
  let finest =
    1 + int_of_float (Float.ceil (Float.log2 (float (Num.precision dtype))))
  in
  let tiny = Num.tiny dtype and huge = Num.huge dtype in
  let half_pi = Float.pi /. 2. in
  (* A node at t: (side, offset, weight), if its numbers are normal floats of
     the dtype. *)
  let node t =
    let s = half_pi *. Float.sinh t and c = half_pi *. Float.cosh t in
    let ok x =
      Float.is_finite x && Float.abs x <= huge && (x = 0. || Float.abs x >= tiny)
    in
    let n =
      match kind with
      | `Finite ->
          (* The distance to the nearer end, over the half-width: 1 − tanh |s| =
             2 / (e^(2|s|) + 1). *)
          let u = 2. /. (Float.exp (2. *. Float.abs s) +. 1.) in
          let w = c /. (Float.cosh s *. Float.cosh s) in
          ((if t < 0. then -1. else if t > 0. then 1. else 0.), u, w)
      | `From -> (0., Float.exp s, c *. Float.exp s)
      | `Line -> (0., Float.sinh s, c *. Float.cosh s)
    in
    let _, u, w = n in
    if ok u && ok w && w > 0. && (kind <> `Finite || u > 0.) then Some n
    else None
  in
  (* The truncation: the nodes on either side of 0 up to the first that is not a
     normal float, at step 2^-finest. *)
  let step = Float.ldexp 1. (-finest) in
  let rec reach t =
    match node (t +. step) with
    | Some _ when t < 64. -> reach (t +. step)
    | _ -> t
  in
  let tmax = reach 0. in
  let level_nodes k =
    let ts =
      if k = 0 then
        let m = int_of_float (Float.floor tmax) in
        List.init ((2 * m) + 1) (fun i -> float (i - m))
      else
        let h = Float.ldexp 1. (-k) in
        let m = int_of_float (Float.floor (((tmax /. h) -. 1.) /. 2.)) in
        List.concat_map
          (fun j ->
            let t = float ((2 * j) + 1) *. h in
            [ -.t; t ])
          (List.init (m + 1) Fun.id)
    in
    List.filter_map (fun t -> Option.map (fun n -> (t, n)) (node t)) ts
  in
  let chunks = ref [] in
  for k = 0 to finest do
    let nodes = Array.of_list (level_nodes k) in
    let count = max 1 ((Array.length nodes + de_chunk - 1) / de_chunk) in
    for c = 0 to count - 1 do
      let get i f d =
        let j = (c * de_chunk) + i in
        if j < Array.length nodes then f nodes.(j) else d
      in
      chunks :=
        ( Array.init de_chunk (fun i -> get i (fun (_, (s, _, _)) -> s) 0.),
          Array.init de_chunk (fun i ->
              get i
                (fun (_, (_, u, _)) -> u)
                (if kind = `Finite then 1. else 0.)),
          Array.init de_chunk (fun i -> get i (fun (_, (_, _, w)) -> w) 0.),
          Array.init de_chunk (fun i ->
              get i
                (fun (t, _) -> if Float.abs t > tmax -. 0.5 then 1. else 0.)
                0.),
          k,
          c = count - 1,
          max 0 (min de_chunk (Array.length nodes - (c * de_chunk))) )
        :: !chunks
    done
  done;
  let chunks = Array.of_list (List.rev !chunks) in
  {
    side = Array.map (fun (s, _, _, _, _, _, _) -> s) chunks;
    offset = Array.map (fun (_, u, _, _, _, _, _) -> u) chunks;
    weight = Array.map (fun (_, _, w, _, _, _, _) -> w) chunks;
    outer = Array.map (fun (_, _, _, o, _, _, _) -> o) chunks;
    level = Array.map (fun (_, _, _, _, k, _, _) -> k) chunks;
    last = Array.map (fun (_, _, _, _, _, l, _) -> l) chunks;
    count = Array.map (fun (_, _, _, _, _, _, n) -> n) chunks;
    finest;
  }

(* The terms [w f(x)] of one chunk's nodes, [side], [offset] and [weight] of
   shape [[c] @ ones]. A node whose point rounds to an end of a finite range is
   unused, its input replaced by the midpoint. *)
let terms fn f range (side, offset, weight) =
  match range with
  | Range.Finite (a, b) ->
      let half = Nx.div_s (Nx.sub b a) 2. in
      let mid = Nx.add a half in
      let left = Nx.add a (Nx.mul half offset)
      and right = Nx.sub b (Nx.mul half offset) in
      let x =
        Nx.where (Nx.less_s side 0.) left
          (Nx.where (Nx.greater_s side 0.) right mid)
      in
      let valid =
        Nx.logical_or (Nx.equal_s side 0.)
          (Nx.logical_and (Nx.not_equal x a) (Nx.not_equal x b))
      in
      let x = Nx.where valid x (Nx.broadcast_to (Nx.shape x) mid) in
      let y = call fn f x in
      Nx.where valid (Nx.mul (Nx.mul half weight) y) (Nx.zeros_like y)
  | From a -> Nx.mul weight (call fn f (Nx.add a offset))
  | Line c -> Nx.mul weight (call fn f (Nx.add c offset))

let tanh_sinh ~tol f range =
  let fn = "Jera.Quad.tanh_sinh" in
  let dtype = Range.dtype range in
  let lanes = Range.shape range in
  let kind =
    match range with
    | Range.Finite _ -> `Finite
    | From _ -> `From
    | Line _ -> `Line
  in
  let sch = schedule kind dtype in
  let n_chunks = Array.length sch.level in
  let table a =
    Nx.create dtype [| n_chunks; de_chunk |] (Array.concat (Array.to_list a))
  in
  let side = table sch.side and offset = table sch.offset in
  let weight = table sch.weight and outer = table sch.outer in
  let levels =
    Nx.create Nx.int32 [| n_chunks |] (Array.map Int32.of_int sch.level)
  in
  let lasts = Nx.create Nx.bool [| n_chunks |] sch.last in
  let counts =
    Nx.create Nx.int32 [| n_chunks |] (Array.map Int32.of_int sch.count)
  in
  let column v =
    Nx.reshape
      (Array.append [| de_chunk |] (Array.make (Array.length lanes) 1))
      v
  in
  let row t i =
    column
      (Nx.reshape [| de_chunk |]
         (Nx.take ~axis:0 ~indices:(Nx.reshape [| 1 |] i) t))
  in
  let detached =
    match range with
    | Range.Finite (a, b) -> Range.Finite (Rune.detach a, Rune.detach b)
    | From a -> From (Rune.detach a)
    | Line c -> Line (Rune.detach c)
  in
  let search x = Rune.detach (f x) in
  let zeros = Nx.zeros dtype lanes in
  (* The carry: the next chunk, then per lane the running sum of the level, the
     last two estimates, the largest term in the truncation's last unit, the
     final level, the status and the evaluations. *)
  let step (i, (sum, (prev, (current, (tail, (final, (st, n))))))) =
    let run = Elementwise.searching st in
    let i64 = Nx.cast Nx.int64 i in
    let t =
      terms fn search detached (row side i64, row offset i64, row weight i64)
    in
    let sum = Nx.add sum (Nx.sum ~axes:[ 0 ] t) in
    let tail =
      Nx.maximum tail (Nx.max ~axes:[ 0 ] (Nx.mul (Nx.abs t) (row outer i64)))
    in
    let at table =
      Nx.reshape [||] (Nx.take ~indices:(Nx.reshape [| 1 |] i64) table)
    in
    let n = Nx.add n (Nx.mul (Nx.cast Nx.int32 run) (at counts)) in
    let k = at levels and ends = at lasts in
    (* At a level's end: I_k = I_(k−1) / 2 + 2^-k R_k, I_0 = R_0. *)
    let h = Nx.exp2 (Nx.neg (Nx.cast dtype k)) in
    let estimate =
      Nx.where (Nx.equal_s k 0l) sum
        (Nx.add (Nx.div_s current 2.) (Nx.mul h sum))
    in
    let closing = Nx.logical_and run (Nx.broadcast_to lanes ends) in
    let prev' = Nx.where closing current prev
    and current' = Nx.where closing estimate current in
    let final = Nx.where closing (Nx.broadcast_to lanes k) final in
    let st' =
      Elementwise.settle st
        (Nx.logical_and closing (Nx.logical_not (Nx.isfinite estimate)))
        Not_finite
    in
    let e = Nx.abs (Nx.sub current' prev') in
    let met =
      Nx.logical_and
        (Nx.broadcast_to lanes (Nx.greater_equal_s k 1l))
        (Elementwise.accepted tol ~e ~y:current')
    in
    let truncated = Elementwise.accepted tol ~e:(Nx.mul h tail) ~y:current' in
    let st' =
      Elementwise.settle st'
        (Nx.logical_and closing (Nx.logical_and met truncated))
        Converged
    in
    let st' = Elementwise.settle st' (Nx.logical_and closing met) Stalled in
    let st' =
      Elementwise.settle st'
        (Nx.logical_and closing
           (Nx.broadcast_to lanes
              (Nx.greater_equal_s k (Int32.of_int sch.finest))))
        Stalled
    in
    let sum = Nx.where (Nx.broadcast_to lanes ends) zeros sum in
    (Nx.add_s i 1l, (sum, (prev', (current', (tail, (final, (st', n)))))))
  in
  let carry =
    Nx.Ptree.(
      pair tensor
        (pair tensor
           (pair tensor
              (pair tensor (pair tensor (pair tensor (pair tensor tensor)))))))
  in
  let _, (_, (prev, (current, (_, (final, (st, n)))))) =
    Rune.iterate carry ~max:n_chunks
      ~until:(fun (_, (_, (_, (_, (_, (_, (st, _))))))) ->
        Nx.logical_not (Nx.any (Elementwise.searching st)))
      ~f:step
      ( Nx.scalar Nx.int32 0l,
        ( zeros,
          ( zeros,
            ( zeros,
              ( zeros,
                ( Nx.zeros Nx.int32 lanes,
                  ( Nx.full Nx.int32 lanes Elementwise.running,
                    Nx.zeros Nx.int32 lanes ) ) ) ) ) ) )
  in
  let ok = Nx.equal_s st (Solution.code Converged) in
  (* The answer: 2^-K times the terms of every level up to each lane's final
     level K, tracked, chunk by chunk. *)
  let chunk_sum total (k, (s, (u, w))) =
    let t = terms fn f range (column s, column u, column w) in
    let keep = Nx.less_equal (Nx.broadcast_to lanes k) final in
    (Nx.add total (Nx.where keep (Nx.sum ~axes:[ 0 ] t) zeros), ())
  in
  let total, () =
    Rune.scan Nx.Ptree.tensor
      Nx.Ptree.(pair tensor (pair tensor (pair tensor tensor)))
      Nx.Ptree.unit ~f:chunk_sum ~init:zeros
      (levels, (side, (offset, weight)))
  in
  let answer = Nx.mul total (Nx.exp2 (Nx.neg (Nx.cast dtype final))) in
  let error = Nx.abs (Nx.sub current prev) in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a" Tol.pp tol)
    ~fix:
      (quad_fix tol ~budget:""
         ~stalled:
           "The terms at the truncation, or the finest level's change, stay \
            above the tolerance: the integrand decays too slowly at an end, or \
            the tolerance is below the dtype's reach. Scale the variable so \
            the integrand's width is near 1.")
    ~value:(Nx.where ok answer current)
    ~error ~status:st ~evaluations:n
    ~facts:
      (range_facts range @ [ Fact ("estimate", current); Fact ("error", error) ])
    ()

(* Cubature *)

(* Genz and Malik's (1980) rule on [−1, 1]^d: the center, the points ±λ₂e_i and
   ±λ₃e_i, ±λ₄e_i ± λ₄e_j for i < j, and the 2^d points (±λ₅, …), with weights
   of degree 7 and of the embedded degree 5, normalised to sum to 1. [axis]
   gives, per axis, the rows of the center and of ±λ₂e_i and ±λ₃e_i, from which
   the fourth difference along that axis is read. *)
type genz_malik = {
  points : float array array;
  w7 : float array;
  w5 : float array;
  axis : (int * int * int * int * int) array;
}

let genz_malik d =
  let fd = float d in
  let l2 = Float.sqrt (9. /. 70.) and l3 = Float.sqrt (9. /. 10.) in
  let l4 = Float.sqrt (9. /. 10.) and l5 = Float.sqrt (9. /. 19.) in
  let pts = ref [] in
  let add p w7 w5 = pts := (p, w7, w5) :: !pts in
  let unit i v = Array.init d (fun k -> if k = i then v else 0.) in
  add (Array.make d 0.)
    ((12824. -. (9120. *. fd) +. (400. *. fd *. fd)) /. 19683.)
    ((729. -. (950. *. fd) +. (50. *. fd *. fd)) /. 729.);
  for i = 0 to d - 1 do
    add (unit i l2) (980. /. 6561.) (245. /. 486.);
    add (unit i (-.l2)) (980. /. 6561.) (245. /. 486.);
    add (unit i l3)
      ((1820. -. (400. *. fd)) /. 19683.)
      ((265. -. (100. *. fd)) /. 1458.);
    add (unit i (-.l3))
      ((1820. -. (400. *. fd)) /. 19683.)
      ((265. -. (100. *. fd)) /. 1458.)
  done;
  for i = 0 to d - 1 do
    for j = i + 1 to d - 1 do
      List.iter
        (fun (si, sj) ->
          add
            (Array.init d (fun k ->
                 if k = i then si *. l4 else if k = j then sj *. l4 else 0.))
            (200. /. 19683.) (25. /. 729.))
        [ (1., 1.); (1., -1.); (-1., 1.); (-1., -1.) ]
    done
  done;
  let corner = 6859. /. 19683. /. Float.ldexp 1. d in
  for c = 0 to (1 lsl d) - 1 do
    add
      (Array.init d (fun k -> if c land (1 lsl k) = 0 then l5 else -.l5))
      corner 0.
  done;
  let all = Array.of_list (List.rev !pts) in
  {
    points = Array.map (fun (p, _, _) -> p) all;
    w7 = Array.map (fun (_, w, _) -> w) all;
    w5 = Array.map (fun (_, _, w) -> w) all;
    axis =
      Array.init d (fun i ->
          (0, 1 + (4 * i), 2 + (4 * i), 3 + (4 * i), 4 + (4 * i)));
  }

(* The ratio of the fourth differences' scales, λ₂² / λ₃². *)
let ratio = 9. /. 70. /. (9. /. 10.)

let cubature ~tol ~budget f (box : _ Box.t) =
  let fn = "Jera.Quad.cubature" in
  let shape = Nx.shape box.lo in
  let rank = Array.length shape in
  let d = shape.(rank - 1) in
  if d < 2 || d > 10 then
    invalid_arg (Printf.sprintf "%s: d = %d is not in [2, 10]" fn d);
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  let lanes = Array.sub shape 0 (rank - 1) in
  let dtype = Nx.dtype box.lo in
  let rule = genz_malik d in
  let n_points = Array.length rule.points in
  let u =
    Nx.create dtype [| n_points; d |] (Array.concat (Array.to_list rule.points))
  in
  let ones k = Array.make k 1 in
  (* Each box's degree-7 sum, its difference from the degree-5 sum and its split
     axis, for [k] boxes per lane given as their bounds of shape [[k] @ lanes @
     [d]]. *)
  let sums f lo hi =
    let c = Nx.div_s (Nx.add lo hi) 2. and h = Nx.div_s (Nx.sub hi lo) 2. in
    let u =
      Nx.reshape (Array.concat [ [| n_points |]; ones rank; [| d |] ]) u
    in
    let points =
      Nx.add
        (Nx.unsqueeze ~axes:[ 0 ] c)
        (Nx.mul (Nx.unsqueeze ~axes:[ 0 ] h) u)
    in
    let y = f points in
    let expected = Array.sub (Nx.shape points) 0 (Nx.ndim points - 1) in
    if Nx.shape y <> expected then
      invalid_arg
        (Printf.sprintf
           "%s: the integrand returned shape %s for points of shape %s" fn
           (Num.shape (Nx.shape y))
           (Num.shape (Nx.shape points)));
    let vol = Nx.prod ~axes:[ Nx.ndim h - 1 ] (Nx.mul_s h 2.) in
    let along w =
      Nx.reshape
        (Array.append [| n_points |] (ones rank))
        (Num.constant dtype w)
    in
    let s7 = Nx.mul vol (Nx.sum ~axes:[ 0 ] (Nx.mul (along rule.w7) y)) in
    let s5 = Nx.mul vol (Nx.sum ~axes:[ 0 ] (Nx.mul (along rule.w5) y)) in
    let row i = Nx.get [ i ] y in
    let fourth =
      Array.map
        (fun (c, p2, m2, p3, m3) ->
          let centre = Nx.mul_s (row c) 2. in
          Nx.abs
            (Nx.sub
               (Nx.sub (Nx.add (row p2) (row m2)) centre)
               (Nx.mul_s (Nx.sub (Nx.add (row p3) (row m3)) centre) ratio)))
        rule.axis
    in
    ( s7,
      Nx.abs (Nx.sub s7 s5),
      Nx.argmax ~axis:0 (Nx.stack (Array.to_list fourth)) )
  in
  let lo0 = Rune.detach box.lo and hi0 = Rune.detach box.hi in
  let search x = Rune.detach (f x) in
  (* Box bounds from integers: lo + (hi − lo) index / 2^level along each
     axis. *)
  let bounds lo hi levels indices =
    let scale = Nx.exp2 (Nx.neg (Nx.cast dtype levels)) in
    let i = Nx.cast dtype indices in
    let w = Nx.unsqueeze ~axes:[ 0 ] (Nx.sub hi lo)
    and lo = Nx.unsqueeze ~axes:[ 0 ] lo in
    ( Nx.add lo (Nx.mul w (Nx.mul i scale)),
      Nx.add lo (Nx.mul w (Nx.mul (Nx.add_s i 1.) scale)) )
  in
  (* A coordinate along axis [a] of each lane's box. *)
  let point a t =
    let along v =
      Nx.squeeze
        ~axes:[ rank - 1 ]
        (Nx.take_along_axis ~axis:(rank - 1)
           ~indices:(Nx.unsqueeze ~axes:[ rank - 1 ] a)
           v)
    in
    let lo = along lo0 in
    Nx.add lo (Nx.mul (Nx.sub (along hi0) lo) t)
  in
  let b =
    Partition.refine Nx.Ptree.tensor ~budget ~lanes ~dims:d ~cost:n_points
      ~evaluate:(fun levels indices ->
        let lo, hi = bounds lo0 hi0 levels indices in
        sums search lo hi)
      ~point
      ~verdict:(fun b live ->
        let total = Partition.sum live b.data in
        ( Nx.logical_not (Nx.isfinite total),
          Elementwise.accepted tol ~e:(Partition.sum live b.error) ~y:total ))
  in
  let ok = Nx.equal_s b.status (Solution.code Converged) in
  (* The answer: the degree-7 rule over the final partition, tracked. *)
  let total =
    Partition.integrate b (fun levels indices ->
        let lo, hi = bounds box.lo box.hi levels indices in
        let s7, _, _ = sums f lo hi in
        s7)
  in
  let live = Partition.in_use b in
  let estimate = Partition.sum live b.data
  and error = Partition.sum live b.error in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a, budget %d" Tol.pp tol budget)
    ~spent:{ used = b.used; unit = "boxes"; budget }
    ~fix:
      (quad_fix tol
         ~budget:
           "Raise the budget or loosen tol; above about six dimensions \
            Quad.qmc converges in fewer evaluations."
         ~stalled:
           "A box cannot be bisected further: the integrand has a singularity \
            inside the box.")
    ~value:(Nx.where ok total estimate)
    ~error ~status:b.status ~evaluations:b.evaluations
    ~facts:[ Fact ("estimate", estimate); Fact ("error", error) ]
    ()

(* Quasi-Monte Carlo *)

let shifts = 16
let qmc_chunk = 64
let sobol_dims = Array.length Sobol.poly

(* Dimension [j]'s 32 direction numbers, as Bratley and Fox's recurrence builds
   them from its polynomial and initial numbers, scaled to 32 bits. *)
let directions j =
  let v = Array.make 32 1 in
  if j > 0 then begin
    let p = Sobol.poly.(j) in
    let m = Array.length Sobol.vinit.(j) in
    Array.blit Sobol.vinit.(j) 0 v 0 m;
    for i = m to 31 do
      let x = ref v.(i - m) and pow2 = ref 1 in
      for k = 0 to m - 1 do
        pow2 := !pow2 lsl 1;
        if (p lsr (m - 1 - k)) land 1 = 1 then
          x := !x lxor (!pow2 * v.(i - k - 1))
      done;
      v.(i) <- !x
    done
  end;
  Array.mapi (fun b x -> x lsl (31 - b)) v

(* The Sobol points of indices [i] (uint32, of shape [[c]]) in [d] dimensions,
   as 32-bit words: the XOR of the direction numbers of the bits set in the Gray
   code of [i]. *)
let sobol_words d i =
  let v = Array.init d directions in
  let gray = Nx.bitwise_xor i (Nx.rshift i 1) in
  let gray = Nx.unsqueeze ~axes:[ 1 ] gray in
  let x = ref (Nx.zeros Nx.uint32 [| Nx.dim 0 i; d |]) in
  for b = 0 to 31 do
    let column =
      Nx.create Nx.uint32 [| 1; d |] (Array.map (fun r -> Int32.of_int r.(b)) v)
    in
    let set =
      Nx.not_equal
        (Nx.bitwise_and (Nx.rshift gray b) (Nx.ones_like gray))
        (Nx.zeros_like gray)
    in
    x :=
      Nx.where (Nx.broadcast_to (Nx.shape !x) set) (Nx.bitwise_xor !x column) !x
  done;
  !x

(* Words as points of (0, 1): (w + 1/2) / 2^k from the top [k] bits, [k] the
   bits the dtype holds below 1. *)
let unit_points dtype words =
  let k = min 32 (Num.precision dtype - 1) in
  let top = if k = 32 then words else Nx.rshift words (32 - k) in
  Nx.mul_s (Nx.add_s (Nx.cast dtype top) 0.5) (Float.ldexp 1. (-k))

let qmc key ~tol ~budget f (box : _ Box.t) =
  let fn = "Jera.Quad.qmc" in
  let shape = Nx.shape box.lo in
  let rank = Array.length shape in
  let d = shape.(rank - 1) in
  if d > sobol_dims then
    invalid_arg (Printf.sprintf "%s: d = %d is above %d" fn d sobol_dims);
  if budget < 1 then
    invalid_arg (Printf.sprintf "%s: budget = %d is below 1" fn budget);
  let lanes = Array.sub shape 0 (rank - 1) in
  let dtype = Nx.dtype box.lo in
  let shift = Nx.bitcast Nx.uint32 (Nx.Rng.bits key [| shifts; d |]) in
  (* The integrand's sum over chunk [j]'s points under each shift, of shape
     [[shifts] @ lanes]. *)
  let chunk_sums f lo hi j =
    let first = Nx.mul_s (Nx.cast Nx.uint32 j) (Int32.of_int qmc_chunk) in
    let i =
      Nx.add
        (Nx.arange Nx.uint32 0 qmc_chunk 1)
        (Nx.broadcast_to [| qmc_chunk |] first)
    in
    let words =
      Nx.bitwise_xor
        (Nx.unsqueeze ~axes:[ 1 ] (sobol_words d i))
        (Nx.unsqueeze ~axes:[ 0 ] shift)
    in
    let u = unit_points dtype words in
    let u =
      Nx.reshape
        (Array.concat
           [ [| qmc_chunk; shifts |]; Array.make (rank - 1) 1; [| d |] ])
        u
    in
    let points = Nx.add lo (Nx.mul (Nx.sub hi lo) u) in
    let y = f points in
    let expected = Array.sub (Nx.shape points) 0 (Nx.ndim points - 1) in
    if Nx.shape y <> expected then
      invalid_arg
        (Printf.sprintf
           "%s: the integrand returned shape %s for points of shape %s" fn
           (Num.shape (Nx.shape y))
           (Num.shape (Nx.shape points)));
    Nx.sum ~axes:[ 0 ] y
  in
  let lo0 = Rune.detach box.lo and hi0 = Rune.detach box.hi in
  let search x = Rune.detach (f x) in
  let per_lane v = Nx.broadcast_to lanes v in
  (* The carry: the next chunk, then per lane the sums under each shift, the
     chunks used, the estimate, its standard error and the status. *)
  let step (j, (sums, (used, (estimate, (error, st))))) =
    let run = Elementwise.searching st in
    let sums =
      Nx.add sums
        (Nx.where
           (Nx.unsqueeze ~axes:[ 0 ] run)
           (chunk_sums search lo0 hi0 j)
           (Nx.zeros_like sums))
    in
    let used = Nx.add used (Nx.cast Nx.int32 run) in
    let n = Nx.add_s j 1l in
    (* At a power of two the points are balanced: test the standard error over
       the shifts. *)
    let power = Nx.equal (Nx.bitwise_and n (Nx.sub_s n 1l)) (Nx.zeros_like n) in
    let count = Nx.mul_s (Nx.cast dtype n) (float qmc_chunk) in
    let means = Nx.div sums count in
    let mean = Nx.mean ~axes:[ 0 ] means in
    let se =
      Nx.sqrt
        (Nx.div_s
           (Nx.sum ~axes:[ 0 ] (Nx.square (Nx.sub means mean)))
           (float (shifts * (shifts - 1))))
    in
    let testing = Nx.logical_and run (per_lane power) in
    let estimate = Nx.where testing mean estimate
    and error = Nx.where testing se error in
    let st =
      Elementwise.settle st
        (Nx.logical_and testing (Nx.logical_not (Nx.isfinite mean)))
        Not_finite
    in
    let st =
      Elementwise.settle st
        (Nx.logical_and testing (Elementwise.accepted tol ~e:se ~y:mean))
        Converged
    in
    let st =
      Elementwise.settle st
        (per_lane (Nx.greater_equal_s n (Int32.of_int budget)))
        Budget_spent
    in
    (n, (sums, (used, (estimate, (error, st)))))
  in
  let zeros = Nx.zeros dtype lanes in
  let carry =
    Nx.Ptree.(
      pair tensor (pair tensor (pair tensor (pair tensor (pair tensor tensor)))))
  in
  let _, (_, (used, (estimate, (error, st)))) =
    Rune.iterate carry ~max:budget
      ~until:(fun (_, (_, (_, (_, (_, st))))) ->
        Nx.logical_not (Nx.any (Elementwise.searching st)))
      ~f:step
      ( Nx.scalar Nx.int32 0l,
        ( Nx.zeros dtype (Array.append [| shifts |] lanes),
          ( Nx.zeros Nx.int32 lanes,
            (zeros, (zeros, Nx.full Nx.int32 lanes Elementwise.running)) ) ) )
  in
  let ok = Nx.equal_s st (Solution.code Converged) in
  (* The answer: the mean over each lane's final points, tracked, chunk by
     chunk. *)
  let sum_chunk total j =
    let keep = Nx.unsqueeze ~axes:[ 0 ] (Nx.less (per_lane j) used) in
    let s = chunk_sums f box.lo box.hi j in
    (Nx.add total (Nx.sum ~axes:[ 0 ] (Nx.where keep s (Nx.zeros_like s))), ())
  in
  let total, () =
    Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit ~f:sum_chunk
      ~init:zeros
      (Nx.arange Nx.int32 0 budget 1)
  in
  let count = Nx.mul_s (Nx.cast dtype used) (float (qmc_chunk * shifts)) in
  Solution.v ~fn
    ~settings:(Format.asprintf "tol %a, budget %d" Tol.pp tol budget)
    ~spent:{ used; unit = "chunks"; budget }
    ~fix:
      (quad_fix tol
         ~budget:
           "Raise the budget or loosen tol: the standard error of a smooth \
            integrand falls about as the inverse of the point count."
         ~stalled:"")
    ~value:(Nx.where ok (Nx.div total count) estimate)
    ~error ~status:st
    ~evaluations:(Nx.mul_s used (Int32.of_int (qmc_chunk * shifts)))
    ~facts:[ Fact ("estimate", estimate); Fact ("error", error) ]
    ()
