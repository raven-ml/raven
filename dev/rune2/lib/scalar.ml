(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog
module D = Nx_array.Dtype

let ln2 = 0.693147180559945309417
let two_over_sqrt_pi = 1.12837916709551257390

type t = {
  ins : D.any array;
  mutable nodes : P.node array;
  mutable dts : D.any array;
  mutable n : int;
  values : int array; (* Each output's node. *)
  mutable tangents : int option array; (* Each output's tangent. *)
}

let tangible (D.Any dt) =
  match D.kind dt with
  | Float | Complex -> true
  | Signed | Unsigned | Boolean -> false

let is_complex (D.Any dt) = D.is Complex dt
let ins b = b.ins
let outputs b = Array.length b.values
let value b k = b.values.(k)
let tangent b k = b.tangents.(k)
let dtype b n = b.dts.(n)

let node_dtype b : P.node -> D.any = function
  | In i -> b.ins.(i)
  | Coord _ -> D.Any D.Int64
  | Const (dt, _) | Op1 (_, dt, _) -> dt
  | Op2 (Binary _, i, _) -> b.dts.(i)
  | Op2 (Compare _, _, _) -> D.Any D.Bool
  | Op3 (_, _, j, _) -> b.dts.(j)

let node b n =
  if b.n = Array.length b.nodes then begin
    let size = (2 * b.n) + 8 in
    let grow a fill = Array.append a (Array.make (size - b.n) fill) in
    b.nodes <- grow b.nodes (P.Coord 0);
    b.dts <- grow b.dts (D.Any D.Bool)
  end;
  b.nodes.(b.n) <- n;
  b.dts.(b.n) <- node_dtype b n;
  b.n <- b.n + 1;
  b.n - 1

let operand b j = node b (In j)

let const b (D.Any dt) x =
  node b (Const (D.Any dt, P.bits dt (D.of_float dt x)))

(* The imaginary unit times [x] in the complex dtype [dt]. *)
let imaginary b (D.Any dt) x =
  match D.kind dt with
  | Complex -> node b (Const (D.Any dt, P.bits dt { Complex.re = 0.; im = x }))
  | Float | Signed | Unsigned | Boolean -> invalid_arg "Scalar.imaginary"

let cast b dt i = node b (Op1 (Cast, dt, i))
let unary b k i = node b (Op1 (Unary k, b.dts.(i), i))
let binary b k i j = node b (Op2 (Binary k, i, j))
let mul b i j = binary b Mul i j
let sub b i j = binary b Sub i j
let exp b i = unary b Exp i
let equal b i j = node b (Op2 (Compare Equal, i, j))
let where b c i j = node b (Op3 (Where, c, i, j))

(* Tangents *)

let plus b x y =
  match (x, y) with
  | Some a, Some c -> Some (binary b Add a c)
  | (Some _ as s), None | None, (Some _ as s) -> s
  | None, None -> None

let minus b x y =
  match (x, y) with
  | Some a, Some c -> Some (sub b a c)
  | (Some _ as s), None -> s
  | None, Some c -> Some (unary b Neg c)
  | None, None -> None

let scaled b t c = Option.map (fun t -> mul b t c) t

(* The real dtype of a complex one's parts. *)
let part (D.Any dt) =
  match dt with
  | D.Complex64 -> D.Any D.Float32
  | D.Complex128 -> D.Any D.Float64
  | _ -> invalid_arg "Scalar.part"

(* [1 - x²] as [(1 - x) (1 + x)], exact where [x²] rounds near [|x| = 1]. *)
let one_minus_square b x =
  let one = const b b.dts.(x) 1. in
  mul b (sub b one x) (binary b Add one x)

(* The tangent of a complex [sign x = y], [y] turning by [Im (t / x)] radians:
   [i y Im (t / x)], zero at the origin, where [y] is. *)
let complex_sign b x y t =
  let dt = b.dts.(x) in
  let q = binary b Fdiv t x in
  let im = cast b (part dt) (mul b q (imaginary b dt (-1.))) in
  let turn = mul b y (mul b (imaginary b dt 1.) (cast b dt im)) in
  let zero = const b dt 0. in
  where b (equal b x zero) zero turn

let unary_tangent b (k : P.unary) ~x ~y t =
  let lit v = const b b.dts.(x) v in
  let div = binary b Fdiv and add = binary b Add and neg = unary b Neg in
  let sqrt = unary b Sqrt and mul = mul b in
  match k with
  | Neg -> Some (neg t)
  | Recip -> Some (neg (mul t (mul y y)))
  | Abs ->
      Some (where b (node b (Op2 (Compare Less_equal, lit 0., x))) t (neg t))
  | Sign -> if is_complex b.dts.(x) then Some (complex_sign b x y t) else None
  | Sqrt -> Some (mul t (div (lit 0.5) y))
  | Exp -> Some (mul t y)
  | Exp2 -> Some (mul (lit ln2) (mul t y))
  | Log -> Some (div t x)
  | Log2 -> Some (div t (mul x (lit ln2)))
  | Log1p -> Some (div t (add x (lit 1.)))
  | Expm1 -> Some (mul t (unary b Exp x))
  | Sin -> Some (mul t (unary b Cos x))
  | Cos -> Some (neg (mul t (unary b Sin x)))
  | Tan -> Some (mul t (add (lit 1.) (mul y y)))
  | Asin -> Some (div t (sqrt (one_minus_square b x)))
  | Acos -> Some (neg (div t (sqrt (one_minus_square b x))))
  | Atan -> Some (div t (add (lit 1.) (mul x x)))
  | Sinh -> Some (mul t (unary b Cosh x))
  | Cosh -> Some (mul t (unary b Sinh x))
  | Tanh -> Some (mul (add t (mul t y)) (sub b (lit 1.) y))
  | Asinh -> Some (div t (sqrt (add (mul x x) (lit 1.))))
  | Acosh ->
      Some (div t (mul (sqrt (sub b x (lit 1.))) (sqrt (add x (lit 1.)))))
  | Atanh -> Some (div t (one_minus_square b x))
  | Erf ->
      Some (mul t (mul (lit two_over_sqrt_pi) (unary b Exp (neg (mul x x)))))
  | Floor | Ceil | Round | Trunc -> None

(* The tangent of an extreme of [a] and [b] whose operand [first] wins where [a]
   does: where they tie each takes half of each tangent, where neither wins (a
   NaN) nothing. *)
let extreme b ~first a c ta tc =
  let dt = b.dts.(a) in
  let zero = const b dt 0. in
  let ta = Option.value ta ~default:zero
  and tc = Option.value tc ~default:zero in
  let half = const b dt 0.5 in
  let tie = binary b Add (mul b half ta) (mul b half tc) in
  let less x y = node b (Op2 (Compare Less, x, y)) in
  let wins_a, wins_c =
    if first then (less c a, less a c) else (less a c, less c a)
  in
  where b (equal b a c) tie (where b wins_a ta (where b wins_c tc zero))

let binary_tangent b (k : P.binary) ~x ~z ~y tx tz =
  let lit v = const b b.dts.(x) v in
  let div = binary b Fdiv and mul = mul b in
  let zero_at v = equal b v (lit 0.) in
  match k with
  | Add -> plus b tx tz
  | Sub -> minus b tx tz
  | Mul -> plus b (scaled b tx z) (Option.map (mul x) tz)
  | Fdiv -> minus b (Option.map (fun t -> div t z) tx) (scaled b tz (div y z))
  | Pow ->
      let base =
        where b (zero_at z) (lit 0.) (mul z (binary b Pow x (sub b z (lit 1.))))
      in
      let exponent = where b (zero_at x) (lit 0.) (mul (unary b Log x) y) in
      plus b (scaled b tx base) (scaled b tz exponent)
  | Mod -> minus b tx (scaled b tz (unary b Trunc (div x z)))
  | Atan2 ->
      let den = binary b Add (mul x x) (mul z z) in
      minus b (scaled b tx (div z den)) (scaled b tz (div x den))
  | Maximum -> Some (extreme b ~first:true x z tx tz)
  | Minimum -> Some (extreme b ~first:false x z tx tz)
  | Idiv | And | Or | Xor | Threefry -> None

let node_tangent b ts k : P.node -> int option = function
  | In _ | Coord _ | Const _ -> None
  | Op1 (Copy, _, i) -> ts.(i)
  | Op1 (Cast, dt, i) ->
      if tangible dt then Option.map (cast b dt) ts.(i) else None
  | Op1 (Bitcast, _, _) -> None
  | Op1 (Unary u, _, i) ->
      Option.bind ts.(i) (fun t -> unary_tangent b u ~x:i ~y:k t)
  | Op2 (Compare _, _, _) -> None
  | Op2 (Binary bk, i, j) ->
      if ts.(i) = None && ts.(j) = None then None
      else binary_tangent b bk ~x:i ~z:j ~y:k ts.(i) ts.(j)
  | Op3 (Where, c, i, j) ->
      if ts.(i) = None && ts.(j) = None then None
      else
        let zero = const b b.dts.(i) 0. in
        let t s = Option.value s ~default:zero in
        Some (where b c (t ts.(i)) (t ts.(j)))
  | Op3 (Fma, i, j, l) ->
      plus b (plus b (scaled b ts.(i) j) (Option.map (mul b i) ts.(j))) ts.(l)

let start (p : P.t) ~tangents ~extra =
  let pins = Iarray.to_array p.ins in
  let has j = tangents.(j) && tangible pins.(j) in
  let tins =
    List.filter_map
      (fun j -> if has j then Some pins.(j) else None)
      (List.init (Array.length pins) Fun.id)
  in
  let ins = Array.concat [ pins; Array.of_list tins; extra ] in
  let count = Iarray.length p.nodes in
  let b =
    {
      ins;
      nodes = [||];
      dts = [||];
      n = 0;
      values = Iarray.to_array p.outs;
      tangents = [||];
    }
  in
  Iarray.iter (fun n -> ignore (node b n)) p.nodes;
  (* The tangent operand of each operand that has one. *)
  let tin = Array.make (Array.length pins) None
  and next = ref (Array.length pins) in
  Array.iteri
    (fun j _ ->
      if has j then begin
        tin.(j) <- Some (operand b !next);
        incr next
      end)
    pins;
  let ts = Array.make count None in
  for k = 0 to count - 1 do
    let n = b.nodes.(k) in
    let t = match n with In j -> tin.(j) | n -> node_tangent b ts k n in
    ts.(k) <- (if tangible b.dts.(k) then t else None)
  done;
  b.tangents <- Array.map (fun v -> ts.(v)) b.values;
  b

let finish b ~outs =
  let ins = Iarray.of_array b.ins in
  let p =
    P.v ~ins
      (Iarray.of_array (Array.sub b.nodes 0 b.n))
      ~outs:(Iarray.of_array outs)
  in
  P.subst ~ins (fun _ -> None) p
