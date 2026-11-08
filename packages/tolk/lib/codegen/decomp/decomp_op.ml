(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let z_const z = const (`Int z)

(* Integer division *)

(* The m and s with x // d = (x * m) >> s for 0 <= x <= vmax and d > 0, from
   Hacker's Delight, chapter 10. *)
let magicgu vmax d =
  let nc = Bigint.(pred (succ vmax / d * d)) in
  let rec search s =
    if s > 2 * Bigint.numbits vmax then assert false
    else
      let p = Bigint.shift_left Bigint.one s in
      let r = Bigint.rem (Bigint.pred p) d in
      if Bigint.gt p Bigint.(nc * (d - one - r)) then (Bigint.((p + d - one - r) / d), s)
      else search (s + 1)
  in
  search 0

(* The next integer type that holds x * m. *)
let widen : Dtype.t -> Dtype.t option = function
  | Int8 -> Some Int16
  | Int16 -> Some Int32
  | Int32 -> Some Int64
  | Int64 -> Some Uint64
  | Uint8 -> Some Uint16
  | Uint16 -> Some Uint32
  | Uint32 -> Some Uint64
  | _ -> None

let rec fast_idiv' ~dont_cast r x d =
  let dmax dt = Dtype.Value.to_z (Dtype.max dt) in
  if Bigint.leq d Bigint.zero || Dtype.Value.(vmin x < of_int 0) then None
  else
    let vmax = Bigint.min (Dtype.Value.to_z (vmax x)) (dmax (dtype x)) in
    if Bigint.lt vmax d then Some (const_like x (`Int Bigint.zero))
    else
      let m, s = magicgu vmax d in
      if Bigint.leq Bigint.(m * vmax) (dmax (dtype x)) then
        Some O.((x * z_const m) lsr int s)
      else
        let k = Bigint.trailing_zeros d in
        match
          if k > 0 then
            fast_idiv' ~dont_cast:true r O.(x lsr int k) (Bigint.shift_right d k)
          else None
        with
        | Some q -> Some q
        | None when dont_cast -> None
        | None -> (
            match widen (dtype x) with
            | Some next
              when Dtype.mem next (Renderer.supported_dtypes r)
                   && Bigint.leq Bigint.(m * vmax) (dmax next) ->
                Some (cast O.((cast x next * z_const m) lsr int s) (dtype x))
            | _ -> None)

let fast_idiv r x d = fast_idiv' ~dont_cast:false r x d

(* Threefry *)

let threefry2x32 x key =
  let u32 u = cast u Dtype.Uint32 in
  let x0 = u32 x and x1 = u32 O.(x lsr int 32) in
  let key0 = u32 key and key1 = u32 O.(key lsr int 32) in
  let rotations = [| [ 13; 15; 26; 6 ]; [ 17; 29; 16; 24 ] |] in
  let ks = [| key1; O.(key0 lxor key1 lxor int 0x1BD11BDA); key0 |] in
  let xr0 = ref O.(x0 + ks.(2)) and xr1 = ref O.(x1 + ks.(0)) in
  for i = 0 to 4 do
    List.iter
      (fun r ->
        let sum = O.(!xr0 + !xr1) and rr = 32 - r in
        (xr1 := O.(sum lxor ((!xr1 lsl int r) + (!xr1 lsr int rr))));
        xr0 := sum)
      rotations.(i mod 2);
    (xr0 := O.(!xr0 + ks.(i mod 3)));
    let k = ks.((i + 1) mod 3) in
    xr1 := O.(!xr1 + k + int i + int 1)
  done;
  O.((cast !xr1 Dtype.Uint64 lsl int 32) lor cast !xr0 Dtype.Uint64)

(* Patterns *)

(* Both operands have one sign, so truncating and flooring agree. *)
let same_sign a b =
  Dtype.Value.(
    (vmin a >= of_int 0 && vmin b >= of_int 0)
    || (vmax a <= of_int 0 && vmax b <= of_int 0))

let sign_differs a b = O.(a < int 0 <> (b < int 0))

(* The value of the integer constant [c] as an operation on [x] reads it: at
   [x]'s width, as a product does, [`Product], or as a quotient or remainder
   does, [`Quotient]. A product reads it modulo 2^bits, so on int32 2^31 is
   2^31 and 2^32 is 0, and a quotient in [x]'s type, where 2^31 is the least
   int32. A weak [x] reads it whole. *)
let at_width reading x c =
  let dt = dtype x in
  match value c with
  | `Int z when Dtype.mem dt Dtype.weaks -> Some z
  | `Int z -> (
      match (Dtype.truncate dt (`Int z), reading) with
      | `Int w, `Product when Bigint.sign w < 0 ->
          Some Bigint.(w + shift_left one (Dtype.bitsize dt))
      | `Int w, _ -> Some w
      | _ -> None)
  | _ -> None

(* [d] as a division of [a] reads it ({!at_width}): a weak constant past
   [a]'s width wrapped to [a]'s type, as on int8 200 is -56 and on uint8 259
   is 3, so that the rules read the sign and value of the divisor compiled
   code divides by. Any other [d] is itself. *)
let divisor a d =
  if op d <> Op.Const || not (Dtype.mem (dtype d) Dtype.weaks) then d
  else
    match (at_width `Quotient a d, value d) with
    | Some w, `Int z when not (Bigint.equal w z) -> const_like d (`Int w)
    | _ -> d

(* A quotient by 0 is 0 and the remainder the dividend, whose sign differs
   from 0's when it is negative: the correction skips a divisor of 0 where
   the divisor's bounds reach it. *)
let floordiv_to_idiv a b =
  let b = divisor a b in
  if same_sign a b then alu a Truncdiv [ b ]
  else
    let differs = sign_differs a b in
    let differs =
      if Dtype.Value.(vmin b <= of_int 0 && of_int 0 <= vmax b) then
        O.(differs land (b <> int 0))
      else differs
    in
    O.(alu a Truncdiv [ b ] - ((alu a Truncmod [ b ] <> int 0) land differs))

let floormod_to_mod a b =
  let b = divisor a b in
  if same_sign a b then alu a Truncmod [ b ]
  else
    let r = alu a Truncmod [ b ] in
    (* A selection, not a multiplication, so that no multiply-add fuses it:
       64-bit integer emulation does not lower one. *)
    O.(
      r
      + where
          ((r <> int 0) land sign_differs a b)
          b
          (const_like b (`Int Bigint.zero)))

(* The exponent of a constant that is 2^i, 0 <= i < 64, as an operation on [x]
   reads it ({!at_width}): below [x]'s width, so that no shift by the width or
   more, which C leaves undefined, computes it. *)
let power_of_two reading x c =
  match at_width reading x c with
  | Some z
    when Bigint.sign z > 0 && Bigint.popcount z = 1 && Bigint.numbits z <= 64 ->
      Some (Bigint.numbits z - 1)
  | _ -> None

(* As {!power_of_two}, 1 <= i: a shift by 0 is left as it is. *)
let shift_of reading x c =
  match power_of_two reading x c with Some v when v > 0 -> Some v | _ -> None

let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let when_ ops op rules = if Op.Set.mem op ops then rules else []

(* The matcher of [key] in [made], made on the key's first call, so that a
   target's rules compile once in the process rather than for each kernel.
   Domains share the matchers, whose lazy state is atomic
   ({!Ops.Pattern_matcher.v}). The list is replaced whole: a domain that loses
   the race to add a key looks again in the list that won. Making a matcher
   builds none of its rules, so a lost race costs nothing. A rule that domains
   first try at once compiles on each of them. *)
let rec cached made equal key make =
  let l = Atomic.get made in
  match List.find_opt (fun (k, _) -> equal k key) l with
  | Some (_, m) -> m
  | None ->
      let m = make () in
      if Atomic.compare_and_set made l ((key, m) :: l) then m
      else cached made equal key make

let simplifying_made = Atomic.make []

let simplifying_patterns ops =
  cached simplifying_made Op.Set.equal ops @@ fun () ->
  let x_int = Upat.var ~dtype:Dtype.ints "x" and c = Upat.cvar "c" in
  let a = Upat.var "a" and b = Upat.var "b" in
  Pattern_matcher.v
    (fun () -> List.concat
       [
         when_ ops Shr
           [
             rule
               Upat.O.(x_int // c)
               (fun m ->
                 Option.map
                   (fun v -> O.(m "x" lsr int v))
                   (shift_of `Quotient (m "x") (m "c")));
           ];
         [
           rule
             Upat.O.(a // b)
             (fun m -> Some (floordiv_to_idiv (m "a") (m "b")));
         ];
         when_ ops And
           [
             rule
               Upat.O.(x_int % c)
               (fun m ->
                 Option.map
                   (fun v -> O.(m "x" land z_const Bigint.(pred (shift_left one v))))
                   (power_of_two `Quotient (m "x") (m "c")));
           ];
         [
           rule Upat.O.(a % b) (fun m -> Some (floormod_to_mod (m "a") (m "b")));
         ];
         (if Op.Set.mem Threefry ops then []
          else
            [
              rule
                (Upat.op Threefry ~src:[ Upat.var "x"; Upat.var "key" ])
                (fun m -> Some (threefry2x32 (m "x") (m "key")));
            ]);
       ])

(* Nested extremes *)

(* Answers of {!holds}, a table of nodes per float type. *)
type memo = (Dtype.t * bool Ops.Tbl.t) list ref

let answers (memo : memo) dt =
  match List.find_opt (fun (d, _) -> Dtype.equal d dt) !memo with
  | Some (_, t) -> t
  | None ->
      let t = Ops.Tbl.create 16 in
      memo := (dt, t) :: !memo;
      t

(* Whether every value of the float node [u] is a value of the float type
   [dt]: a value of a type [dt] holds, a constant [dt] holds, a selection of
   such values, or such a value converted without change. Each node's answer
   is kept, so a graph costs its size however its nodes are shared. *)
let rec holds memo dt u =
  let t = answers memo dt in
  match Ops.Tbl.find_opt t u with
  | Some b -> b
  | None ->
      let b =
        Dtype.equal (dtype u) dt
        || Dtype.can_lossless_cast (dtype u) dt
        ||
        match op u with
        | Const -> (
            match value u with
            | `Float x when Float.is_nan x -> true
            | `Float _ as v -> Dtype.equal_const (Dtype.truncate dt v) v
            | _ -> false)
        | Where -> holds memo dt (nth u 1) && holds memo dt (nth u 2)
        | Cast -> keeps memo u && holds memo dt (nth u 0)
        | _ -> false
      in
      Ops.Tbl.replace t u b;
      b

(* Whether the cast [u] keeps its float operand's value: it widens to a type
   that holds all of its operand's type, or narrows a value its type holds. *)
and keeps memo u =
  let s = nth u 0 in
  Dtype.is_float (dtype s)
  && Dtype.is_float (dtype u)
  && (Dtype.can_lossless_cast (dtype s) (dtype u) || holds memo (dtype u) s)

(* [u] without the casts that keep its value. LLVM's InstCombine narrows a
   float32 selection of widened float16 values to a float16 selection, which
   keeps every value, and folds through a widening that keeps it, so a float16
   selection widened to float32 and narrowed back, as rune computes one,
   nests as the selection does. *)
let unwrapped u =
  let memo = ref [] in
  let rec go u =
    if Op.equal (op u) Cast && keeps memo u then go (nth u 0) else u
  in
  go u

(* Whether [a] and [b] are one value: one node once unwrapped, or float
   constants of the same bits. *)
let same a b =
  let a = unwrapped a and b = unwrapped b in
  a == b
  || Op.equal (op a) Const
     && Op.equal (op b) Const
     && Dtype.is_float (dtype a)
     && Dtype.is_float (dtype b)
     && Dtype.equal_const (value a) (value b)

(* [u] as a selection of [v] and [c] by their comparison, [(v < c) ? v : c] or
   [(c < v) ? c : v] a minimum, [(v < c) ? c : v] or [(c < v) ? v : c] a
   maximum, each up to casts that keep values: [v], whether it is a minimum,
   whether it picks [c] where the comparison is false, at a tie or a NaN, the
   comparison's [c] and the selection. *)
let extreme_of c u =
  let u = unwrapped u in
  if not (Op.equal (op u) Where) then None
  else
    match src u with
    | [ cond; t; f ] when Op.equal (op cond) Cmplt -> (
        match src cond with
        | [ p; q ]
          when ((same t p && same f q) || (same t q && same f p))
               && not (same p q) ->
            if same p c || same q c then
              let v, c' = if same p c then (q, p) else (p, q) in
              Some (v, same t p, same f c, c', u)
            else None
        | _ -> None)
    | _ -> None

(* A minimum of a minimum against one [c], or a maximum of a maximum, is one
   selection: the inner one where it picks [c] at a tie or the two pick alike,
   and otherwise the one of [v] that picks [c] there, [(v < c) ? v : c] or
   [(c < v) ? v : c], cast to the outer one's type, which holds its values.
   LLVM's InstCombine folds the outer into the inner one whatever they pick:
   with [m = (0.0 < v) ? 0.0 : v], [(m < 0.0) ? m : 0.0] becomes [m], -0.0 at
   v = -0.0 where the selection gives 0.0, from -O1 on every target in clang
   19, and in NVRTC 13.4. Made one here, LLVM finds no such nest to fold, up to
   the casts {!unwrapped} sees through. *)
let nested_extreme o =
  match src o with
  | [ cond; _; _ ] when Op.equal (op cond) Cmplt ->
      List.find_map
        (fun m ->
          let c = if m == nth cond 0 then nth cond 1 else nth cond 0 in
          match (extreme_of c o, extreme_of c m) with
          | Some (v_o, min_o, keeps_o, _, _), Some (v, min_i, keeps_i, c, inner)
            when same v_o m && min_o = min_i ->
              let r =
                if keeps_i || keeps_o = keeps_i then inner
                else if min_i then where O.(v < c) v c
                else where O.(c < v) v c
              in
              Some (if Dtype.equal (dtype r) (dtype o) then r else cast r (dtype o))
          | _ -> None)
        (src cond)
  | _ -> None

let late_made = Atomic.make []

let late_patterns ~disable_fast_idiv ops =
  let equal (d, o) (d', o') = Bool.equal d d' && Op.Set.equal o o' in
  cached late_made equal (disable_fast_idiv, ops) @@ fun () ->
  let var = Upat.var and cvar = Upat.cvar in
  let x_int = var ~dtype:Dtype.ints "x" in
  let c = cvar "c" in
  let cdiv x d = Upat.op Truncdiv ~src:[ x; d ]
  and cmod x d = Upat.op Truncmod ~src:[ x; d ] in
  let int_value u = match value u with `Int z -> Some z | _ -> None in
  let fast ctx m =
    Option.bind (int_value (divisor (m "x") (m "d"))) (fast_idiv ctx (m "x"))
  in
  Pattern_matcher.v
    (fun () -> List.concat
       [
         [ rule (Upat.op Where ~name:"o") (fun m -> nested_extreme (m "o")) ];
         (if Op.Set.mem Max ops || not (Op.Set.mem Cmplt ops) then []
          else
            [
              rule (Upat.op Max ~name:"m") (fun m ->
                  let a = nth (m "m") 0 and b = nth (m "m") 1 in
                  Some (where O.(a < b) b a));
            ]);
         when_ ops Or
           [
             rule
               Upat.O.(
                 Upat.logical_not (var ~dtype:[ Bool ] "x")
                 land Upat.logical_not (var ~dtype:[ Bool ] "y"))
               (fun m -> Some (logical_not O.(m "x" lor m "y")));
           ];
         when_ ops Shl
           [
             rule
               Upat.O.(x_int * c)
               (fun m ->
                 Option.map
                   (fun v -> O.(m "x" lsl int v))
                   (shift_of `Product (m "x") (m "c")));
           ];
         when_ ops Shr
           (List.concat
              [
                [
                  rule
                    (cdiv (var ~dtype:Dtype.uints "x") c)
                    (fun m ->
                      Option.map
                        (fun v -> O.(m "x" lsr int v))
                        (shift_of `Quotient (m "x") (m "c")));
                  rule (cdiv x_int c) (fun m ->
                      Option.map
                        (fun v ->
                          let x = m "x" in
                          let l = O.(x < int 0) in
                          let l =
                            if Dtype.Value.(vmin l = vmax l) then
                              const_like l (vmin l :> Dtype.const)
                            else l
                          in
                          O.((x + where l (m "c" - int 1) (int 0)) lsr int v))
                        (shift_of `Quotient (m "x") (m "c")));
                ];
                (if disable_fast_idiv then []
                 else
                   [
                     rule_ctx (cdiv x_int (cvar "d")) fast;
                     rule_ctx
                       (cmod x_int (cvar "d"))
                       (fun ctx m ->
                         Option.map
                           (fun q ->
                             O.(m "x" - (divisor (m "x") (m "d") * q)))
                           (fast ctx m));
                   ]);
              ]);
         when_ ops Neg
           (rule
              Upat.O.(var "x" * Upat.cvar "c")
              (fun m ->
                match
                  (value (m "c"), Dtype.const (dtype (m "x")) (`Int Bigint.minus_one))
                with
                | (#Dtype.value as c), (#Dtype.value as minus_one)
                  when Dtype.Value.(c = minus_one) ->
                    Some (alu (m "x") Neg [])
                | _ -> None)
           :: when_ ops Sub
                [
                  rule
                    Upat.O.(var "x" + Upat.alu (var "y") Neg [])
                    (fun m -> Some (alu (m "x") Sub [ m "y" ]));
                ]);
         when_ ops Cmplt
           [
             (* not x < c is c-1 < x, and not c < x is x < c+1, unless c-1 or
                c+1 wraps *)
             rule
               (Upat.logical_not Upat.O.(x_int < c))
               (fun m ->
                 if Dtype.Value.(vmin (m "c") = Dtype.min (dtype (m "x"))) then
                   None
                 else Some O.(m "c" - int 1 < m "x"));
             rule
               (Upat.logical_not Upat.O.(c < x_int))
               (fun m ->
                 if Dtype.Value.(vmin (m "c") = Dtype.max (dtype (m "x"))) then
                   None
                 else Some O.(m "x" < m "c" + int 1));
             rule
               Upat.O.((cvar "c1" < x_int) land (x_int < cvar "c2"))
               (fun m ->
                 match (value (m "c1"), value (m "c2")) with
                 | (#Dtype.value as c1), (#Dtype.value as c2)
                   when Dtype.Value.(c1 + of_int 1 = c2 - of_int 1) ->
                     Some (eq (m "x") O.(m "c1" + int 1))
                 | _ -> None);
           ];
         when_ ops Cmpne
           [
             rule
               (Upat.logical_not (Upat.eq (var "x") (var "y")))
               (fun m -> Some (alu (m "x") Cmpne [ m "y" ]));
           ];
         when_ ops Mulacc
           (rule
              Upat.O.((var "a" * var "b") + var "c")
              (fun m -> Some (alu (m "a") Mulacc [ m "b"; m "c" ]))
           :: when_ ops Shl
                [
                  rule
                    Upat.O.(Upat.alu (var "x") Shl [ cvar "n" ] + var "c")
                    (fun m ->
                      Option.map
                        (fun n ->
                          let x = m "x" in
                          alu x Mulacc
                            [
                              const_like x (`Int Bigint.(shift_left one (to_int n)));
                              m "c";
                            ])
                        (int_value (m "n")));
                ]);
         (* A reciprocal is [1 / x], the same value rounded once. A product by
            one stays a product: [a / b] would round once where [a * (1 / b)]
            rounds twice, and change its value. *)
         when_ ops Fdiv
           [
             rule
               (Upat.reciprocal (var "x"))
               (fun m -> Some (alu (float 1.0) Fdiv [ m "x" ]));
           ];
       ])
