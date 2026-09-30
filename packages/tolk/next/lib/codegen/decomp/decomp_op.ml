(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let z_const z = const (`Int z)

(* Integer division *)

(* The m and s with x // d = (x * m) >> s for 0 <= x <= vmax and d > 0, from
   Hacker's Delight, chapter 10. *)
let magicgu vmax d =
  let nc = Z.(pred (succ vmax / d * d)) in
  let rec search s =
    if s > 2 * Z.numbits vmax then assert false
    else
      let p = Z.shift_left Z.one s in
      let r = Z.rem (Z.pred p) d in
      if Z.gt p Z.(nc * (d - one - r)) then (Z.((p + d - one - r) / d), s)
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
  if Z.leq d Z.zero || Dtype.Value.(vmin x < of_int 0) then None
  else
    let vmax = Z.min (Dtype.Value.to_z (vmax x)) (dmax (dtype x)) in
    if Z.lt vmax d then Some (const_like x (`Int Z.zero))
    else
      let m, s = magicgu vmax d in
      if Z.leq Z.(m * vmax) (dmax (dtype x)) then
        Some O.((x * z_const m) lsr int s)
      else
        let k = Z.trailing_zeros d in
        match
          if k > 0 then
            fast_idiv' ~dont_cast:true r O.(x lsr int k) (Z.shift_right d k)
          else None
        with
        | Some q -> Some q
        | None when dont_cast -> None
        | None -> (
            match widen (dtype x) with
            | Some next
              when List.mem next (Renderer.supported_dtypes r)
                   && Z.leq Z.(m * vmax) (dmax next) ->
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

let floordiv_to_idiv a b =
  if same_sign a b then alu a Cdiv [ b ]
  else
    O.(alu a Cdiv [ b ] - ((alu a Cmod [ b ] <> int 0) land sign_differs a b))

let floormod_to_mod a b =
  if same_sign a b then alu a Cmod [ b ]
  else
    let r = alu a Cmod [ b ] in
    (* A selection, not a multiplication, so that no multiply-add fuses it:
       64-bit integer emulation does not lower one. *)
    O.(
      r
      + where
          ((r <> int 0) land sign_differs a b)
          b
          (const_like b (`Int Z.zero)))

(* The exponent of a constant that is 2^i, 0 <= i < 64. *)
let power_of_two c =
  match value c with
  | `Int z when Z.sign z > 0 && Z.popcount z = 1 && Z.numbits z <= 64 ->
      Some (Z.numbits z - 1)
  | _ -> None

(* The exponent of a constant that is 2^i, 1 <= i < 64: a shift by 0 is left as
   it is. *)
let shift_of c =
  match power_of_two c with Some v when v > 0 -> Some v | _ -> None

let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let when_ ops op rules = if Op.Set.mem op ops then rules else []

let simplifying_patterns ops =
  let x_int = Upat.var ~dtype:Dtype.ints "x" and c = Upat.cvar "c" in
  let a = Upat.var "a" and b = Upat.var "b" in
  Pattern_matcher.v
    (List.concat
       [
         when_ ops Shr
           [
             rule
               Upat.O.(x_int // c)
               (fun m ->
                 Option.map (fun v -> O.(m "x" lsr int v)) (shift_of (m "c")));
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
                   (fun v -> O.(m "x" land z_const Z.(pred (shift_left one v))))
                   (power_of_two (m "c")));
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

let late_patterns ~disable_fast_idiv ops =
  let var = Upat.var and cvar = Upat.cvar in
  let x_int = var ~dtype:Dtype.ints "x"
  and x_sint = var ~dtype:Dtype.sints "x" in
  let c = cvar "c" in
  let cdiv x d = Upat.op Cdiv ~src:[ x; d ]
  and cmod x d = Upat.op Cmod ~src:[ x; d ] in
  let int_value u = match value u with `Int z -> Some z | _ -> None in
  let fast ctx m = Option.bind (int_value (m "d")) (fast_idiv ctx (m "x")) in
  Pattern_matcher.v
    (List.concat
       [
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
                 Option.map (fun v -> O.(m "x" lsl int v)) (shift_of (m "c")));
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
                        (shift_of (m "c")));
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
                        (shift_of (m "c")));
                ];
                (if disable_fast_idiv then []
                 else
                   [
                     rule_ctx (cdiv x_int (cvar "d")) fast;
                     rule_ctx
                       (cmod x_int (cvar "d"))
                       (fun ctx m ->
                         Option.map
                           (fun q -> O.(m "x" - (m "d" * q)))
                           (fast ctx m));
                   ]);
              ]);
         when_ ops Neg
           (rule
              Upat.O.(var "x" * int (-1))
              (fun m -> Some (alu (m "x") Neg []))
           :: when_ ops Sub
                [
                  rule
                    Upat.O.(var "x" + Upat.alu (var "y") Neg [])
                    (fun m -> Some (alu (m "x") Sub [ m "y" ]));
                ]);
         when_ ops Cmplt
           [
             rule
               (Upat.logical_not Upat.O.(x_sint < c))
               (fun m -> Some O.(m "c" - int 1 < m "x"));
             rule
               (Upat.logical_not Upat.O.(c < x_sint))
               (fun m -> Some O.(m "x" < m "c" + int 1));
             rule
               Upat.O.(x_sint * int (-1) < var ~dtype:Dtype.sints "y" * c)
               (fun m -> Some O.(m "y" * neg (m "c") < m "x"));
             rule
               Upat.O.(x_sint * int (-1) < c)
               (fun m -> Some O.(neg (m "c") < m "x"));
             rule
               Upat.O.((cvar "c1" < x_sint) land (x_sint < cvar "c2"))
               (fun m ->
                 match (value (m "c1"), value (m "c2")) with
                 | (#Dtype.value as c1), (#Dtype.value as c2)
                   when Dtype.Value.(c1 + of_int 1 = c2 - of_int 1) ->
                     Some (eq (m "x") O.(m "c1" + int 1))
                 | _ -> None);
           ];
         when_ ops Cmpeq
           [
             rule
               (Upat.logical_not Upat.O.(var "x" <> var "y"))
               (fun m -> Some (alu (m "x") Cmpeq [ m "y" ]));
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
                              const_like x (`Int Z.(shift_left one (to_int n)));
                              m "c";
                            ])
                        (int_value (m "n")));
                ]);
         when_ ops Fdiv
           [
             rule
               (Upat.reciprocal (var "x"))
               (fun m -> Some (alu (float 1.0) Fdiv [ m "x" ]));
             rule
               Upat.O.(
                 var "a"
                 * Upat.op Fdiv ~dtype:Dtype.floats
                     ~src:[ Upat.const (`Int Z.one); var "b" ])
               (fun m -> Some (alu (m "a") Fdiv [ m "b" ]));
           ];
       ])
