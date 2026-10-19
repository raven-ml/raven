(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
open Lower_bits
module D = Nx_array.Dtype
module P = Nx_kernel.Prog

(* Elements *)

let dtype (D.Any dt) : Dtype.t =
  match dt with
  | D.Float64 | D.Complex128 -> Float64
  | D.Float32 | D.Complex64 -> Float32
  | D.Float16 -> Float16
  | D.Bfloat16 -> Bfloat16
  | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn -> Uint8
  | D.Int64 -> Int64
  | D.Uint64 -> Uint64
  | D.Int32 -> Int32
  | D.Uint32 -> Uint32
  | D.Int16 -> Int16
  | D.Uint16 -> Uint16
  | D.Int8 | D.Int4 -> Int8
  | D.Uint8 | D.Uint4 -> Uint8
  | D.Bool | D.Bit -> Uint8

let is_complex (D.Any dt) = D.is D.Complex dt

let dims dt s =
  let s = List.map (fun n -> Ops.Int n) (Array.to_list s) in
  if is_complex dt then s @ [ Ops.Int 2 ] else s

let parts dt u =
  if not (is_complex dt) then [ u ]
  else
    let s = Shape.shape u in
    let r = List.length s in
    let part i =
      Shape.reshape
        (Shape.shrink u
           (List.init r (fun j ->
                if j = r - 1 then Some (Ops.Int i, Ops.Int (i + 1)) else None)))
        (List.filteri (fun j _ -> j < r - 1) s)
    in
    [ part 0; part 1 ]

let of_parts dt us =
  match us with
  | [ u ] when not (is_complex dt) -> u
  | [ re; im ] when is_complex dt ->
      let last u = Shape.reshape u (Shape.shape u @ [ Ops.Int 1 ]) in
      Shape.cat ~axis:(Shape.ndim re) (last re) [ last im ]
  | _ -> invalid_arg "Lower_arith.of_parts: parts of another dtype"

let int n = `Int (Bigint.of_int n)

(* Nodes

   Constants take the type and shape of the node they meet. Integer arithmetic
   runs on unsigned types of 32 or 64 bits only: C wraps those, and promotes
   narrower ones to [int], whose sums carry past their width. *)

let num ?dtype u n = Shape.const_like ?dtype u (int n)
let truth u b = Shape.const_like ~dtype:Bool u (`Bool b)
let where = Ops.where
let not_ = Ops.logical_not
let ( && ) = Ops.bitwise_and
let ( || ) = Ops.bitwise_or
let ( + ) = Ops.add
let ( - ) = Ops.sub
let ( * ) = Ops.mul
let ( land ) x n = Ops.bitwise_and x (num x n)
let ( lor ) = Ops.bitwise_or
let ( lxor ) = Ops.bitwise_xor
let ( lsl ) x n = Ops.shl x (num x n)
let ( lsr ) x n = Ops.shr x (num x n)
let ( < ) = Ops.lt
let ( = ) = Ops.eq
let ( <> ) = Ops.ne

(* IEEE's [x <= y], false on NaN. *)
let at_most x y = x < y || x = y
let of_bits t u = Ops.bitcast u t
let one_if c u = where c (num u 1) (num u 0)

(* Narrow floats

   float16, bfloat16, the float8 dtypes and float4_e2m1fn compute in float32: a
   code decodes exactly, and a result encodes once as [Dtype.of_float] stores,
   by integer formulas on float32's bits. No target's own conversion decides a
   tie, a subnormal or a NaN's bits. *)

let narrow (D.Any dt) =
  match dt with
  | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn
    ->
      true
  | _ -> false

let code (D.Any dt) x =
  match dt with
  | D.Float16 | D.Bfloat16 -> Ops.cast (Ops.bitcast x Uint16) Uint32
  | _ -> Ops.cast x Uint32

let of_code (D.Any dt) c =
  match dt with
  | D.Float16 -> Ops.bitcast (Ops.cast c Uint16) Float16
  | D.Bfloat16 -> Ops.bitcast (Ops.cast c Uint16) Bfloat16
  | _ -> Ops.cast c Uint8

let f32 u = of_bits Float32 u
let pow2 u e = Shape.const_like ~dtype:Float32 u (`Float (Float.ldexp 1. e))

(* The value of the magnitude code [q] of [m] fraction bits and bias [bias]: a
   subnormal's is its fraction times the least subnormal, a normal's its fields
   moved into float32's. Both are exact. *)
let mini_value q m bias =
  let e = q lsr m and frac = q land Stdlib.((1 lsl m) - 1) in
  let sub = Ops.cast frac Float32 * pow2 q Stdlib.(1 - bias - m) in
  let normal =
    f32 (((e + num q Stdlib.(127 - bias)) lsl 23) lor (frac lsl Stdlib.(23 - m)))
  in
  where (e = num q 0) sub normal

(* [v], positive, with the sign bit [s], 0 or 1. *)
let with_sign v s = f32 (bits v lor (s lsl 31))

let decode (D.Any dt as d) x =
  let c = code d x in
  match dt with
  | D.Float16 ->
      let q = c land 0x7FFF in
      let frac = q land 0x3FF in
      let quiet = where (frac <> num frac 0) (num q 0x400000) (num q 0) in
      let special = f32 (num q 0x7F800000 lor (frac lsl 13) lor quiet) in
      with_sign
        (where (q < num q 0x7C00) (mini_value q 10 15) special)
        (c lsr 15)
  | D.Bfloat16 ->
      (* A NaN gains float32's quiet bit, bit 6 of the code: a magnitude past
         the infinity's makes the difference wrap, setting bit 15. *)
      let past = (num c 0x7F80 - (c land 0x7FFF)) land 0xFFFF in
      f32 ((c lor ((past lsr 9) land 0x40)) lsl 16)
  | D.Float8_e4m3fn ->
      let q = c land 0x7F in
      let v = where (q = num q 0x7F) (nan (f32 q)) (mini_value q 3 7) in
      with_sign v (c lsr 7)
  | D.Float8_e5m2 ->
      let q = c land 0x7F in
      let nan = f32 (num q 0x7FC00000 lor ((q land 1) lsl 21)) in
      let inf = f32 (num q 0x7F800000) in
      let v =
        where
          (num q 0x7C < q)
          nan
          (where (q = num q 0x7C) inf (mini_value q 2 15))
      in
      with_sign v (c lsr 7)
  | _ -> with_sign (mini_value (c land 7) 1 1) ((c lsr 3) land 1)

(* The magnitude code of [f], not NaN, rounded to nearest, ties to even; one
   past the format's largest finite code means [f] overflows. Below the least
   normal, adding 2^(24-bias-m) makes the float sum round to a multiple of the
   least subnormal; above it, the integer rounds the fraction and a carry runs
   into the exponent. *)
let mini_round f m bias =
  let u = bits f land 0x7FFFFFFF in
  let magic = pow2 u Stdlib.(24 - bias - m) in
  let sub =
    bits (f32 u + magic) - num u Stdlib.((127 + 24 - bias - m) lsl 23)
  in
  let odd = (u lsr Stdlib.(23 - m)) land 1 in
  let normal =
    (u
    - num u Stdlib.((127 - bias) lsl 23)
    + num u Stdlib.((1 lsl (22 - m)) - 1)
    + odd)
    lsr Stdlib.(23 - m)
  in
  where (u < num u Stdlib.((128 - bias) lsl 23)) sub normal

let saturate q top = where (num q top < q) (num q top) q

(* [f], a float32, stored into the narrow float [d]. *)
let encode (D.Any dt as d) f =
  let i = bits f in
  let u = i land 0x7FFFFFFF in
  let nan = num u 0x7F800000 < u in
  let sign = i lsr 31 in
  of_code d
    (match dt with
    | D.Float16 ->
        let q = mini_round f 10 15 in
        let quiet = num u 0x7E00 lor ((u lsr 13) land 0x3FF) in
        (sign lsl 15) lor where nan quiet (saturate q 0x7C00)
    | D.Bfloat16 ->
        (* NaN first: the rounding bias could carry a small NaN's fraction into
           the exponent. *)
        let rounded = (i + num i 0x7FFF + ((i lsr 16) land 1)) lsr 16 in
        where nan ((i lsr 16) lor num i 0x40) rounded
    | D.Float8_e4m3fn ->
        let q = saturate (mini_round f 3 7) 0x7E in
        (sign lsl 7) lor where nan (num q 0x7F) q
    | D.Float8_e5m2 ->
        let q = saturate (mini_round f 2 15) 0x7C in
        let quiet = num u 0x7E lor ((u lsr 21) land 1) in
        (sign lsl 7) lor where nan quiet q
    | _ ->
        let q = saturate (mini_round f 1 1) 7 in
        where nan (num q 0) ((sign lsl 3) lor q))

(* [x], a float64, as the float32 rounded to odd: truncated, with its last bit
   set where a discarded bit was. float32's grid is at least four times finer
   than a narrow format's, so a narrow float rounded from it rounds once, as
   from [x]. *)
let float_odd x =
  let f = Ops.cast x Float32 in
  let back = Ops.cast f Float64 in
  let i = bits f in
  let away = one_if (magnitude x < magnitude back) i in
  let inexact = one_if (back <> x && x = x) i in
  f32 ((i - away) lor inexact)

(* [hi + lo], [|hi| >= |lo|] or [hi = 0], rounded to odd: the sum's error, exact
   by Fast2Sum, says whether it rounded and which way. *)
let sum_odd hi lo =
  let s = hi + lo in
  let e = lo - (s - hi) in
  let zero = Shape.const_like e (`Float 0.) in
  let inexact = e <> zero in
  let away = inexact && e < zero <> (s < zero) in
  of_bits Float64 ((bits s - one_if away (bits s)) lor one_if inexact (bits s))

(* A 64-bit integer as a float64 rounded to odd from its exact value, from its
   two 32-bit halves, each exact. *)
let odd64 ~signed x =
  let w = bits x in
  let hi = Ops.cast (w lsr 32) Uint32 in
  let hi = if signed then Ops.bitcast hi Int32 else hi in
  let scale = Shape.const_like ~dtype:Float64 w (`Float 0x1p32) in
  sum_odd (Ops.cast hi Float64 * scale) (Ops.cast (Ops.cast w Uint32) Float64)

(* Integers

   An integer dtype narrower than 32 bits computes in the 32-bit integer of its
   signedness; int4 and uint4 wrap to four bits on the store. *)

let is_signed (D.Any dt) = D.is D.Signed dt
let is_wide (D.Any dt) = Stdlib.( = ) (D.bits dt) 64

(* [x] as the unsigned bits of its compute type, sign-extended where [d] is
   signed. *)
let lift d x =
  if is_wide d then bits x
  else if is_signed d then Ops.bitcast (Ops.cast x Int32) Uint32
  else Ops.cast x Uint32

(* [x] as the signed or unsigned value of its compute type. *)
let value d x =
  if is_wide d then x else Ops.cast x (if is_signed d then Int32 else Uint32)

(* [r], unsigned bits of 32 or 64, stored into the integer dtype [d] modulo its
   width. *)
let store (D.Any dt as d) r =
  match dt with
  | D.Int4 ->
      let r = Ops.cast r Uint32 in
      Ops.bitcast (Ops.cast ((r land 15 lxor num r 8) - num r 8) Uint8) Int8
  | D.Uint4 -> Ops.cast (r land 15) Uint8
  | _ ->
      let t = dtype d in
      Ops.bitcast (Ops.cast r (unsigned t)) t

(* [x], an integer of [s], stored into the integer dtype [d] modulo its width:
   sign- or zero-extended to 64 bits first. *)
let wrap s d x =
  let w =
    if is_signed s then Ops.bitcast (Ops.cast x Int64) Uint64
    else Ops.cast x Uint64
  in
  store d w

let int_unary d (k : P.unary) x =
  let zero = num x 0 and one = num x 1 in
  let neg = store d (num (lift d x) 0 - lift d x) in
  match k with
  | Neg -> neg
  | Recip ->
      let unit = if is_signed d then x = one || x = num x (-1) else x = one in
      where unit x zero
  | Abs -> if is_signed d then where (x < zero) neg x else x
  | Sign ->
      if is_signed d then where (x < zero) (num x (-1)) (one_if (zero < x) x)
      else one_if (x <> zero) x
  | Floor | Ceil | Round | Trunc -> x
  | _ -> invalid_arg "Lower_arith: a float kind of an integer"

(* [x] to the power [e], wrapping, by squaring over [e]'s bits. A negative
   exponent gives 0 but for a base of 1 or -1. *)
let int_pow d x e =
  let b = lift d x and n = lift d e in
  let width = Stdlib.( * ) 8 (Dtype.itemsize (Ops.dtype b)) in
  let rec go k r b =
    if Stdlib.( = ) k width then r
    else
      let set = Ops.shr n (num n k) land 1 <> num n 0 in
      go (Stdlib.( + ) k 1) (where set (r * b) r) (b * b)
  in
  let power = store d (go 0 (num b 1) b) in
  if not (is_signed d) then power
  else
    let one = num x 1 and minus = num x (-1) in
    let odd = e land 1 <> num e 0 in
    let inverse =
      where (x = one) one (where (x = minus) (where odd minus one) (num x 0))
    in
    where (e < num e 0) inverse power

(* Integer division. A zero divisor gives 0, and the remainder the dividend; -1
   gives the negation and a remainder of 0. Both sides of a selection compute,
   so the divisor is made safe before it divides. *)
let int_div d ~rem x y =
  let vx = value d x and vy = value d y in
  let zero = vy = num vy 0 in
  let minus = if is_signed d then vy = num vy (-1) else truth vy false in
  let safe = where (zero || minus) (num vy 1) vy in
  let back v = store d (if is_wide d then bits v else Ops.bitcast v Uint32) in
  if rem then
    where zero x (where minus (num x 0) (back (Ops.alu vx Op.Cmod [ safe ])))
  else
    let neg = store d (num (lift d x) 0 - lift d x) in
    where zero (num x 0) (where minus neg (back (Ops.alu vx Op.Cdiv [ safe ])))

(* Threefry-2x32 with 20 rounds: the counter and the key are word pairs, the low
   word first, and so is the result. Each group of four rounds gives its
   rotations, then the key words it injects and its count. *)
let schedule =
  [
    ([ 13; 15; 26; 6 ], 1, 2, 1);
    ([ 17; 29; 16; 24 ], 2, 0, 2);
    ([ 13; 15; 26; 6 ], 0, 1, 3);
    ([ 17; 29; 16; 24 ], 1, 2, 4);
    ([ 13; 15; 26; 6 ], 2, 0, 5);
  ]

let threefry counter key =
  let low w = Ops.cast w Uint32 and high w = Ops.cast (w lsr 32) Uint32 in
  let k0 = low key and k1 = high key in
  let ks = [| k0; k1; num k0 0x1BD11BDA lxor k0 lxor k1 |] in
  let round (x0, x1) r =
    let x0 = x0 + x1 in
    (x0, (x1 lsl r) lor (x1 lsr Stdlib.(32 - r)) lxor x0)
  in
  let four xs (rs, a, b, s) =
    let x0, x1 = List.fold_left round xs rs in
    (x0 + ks.(a), x1 + ks.(b) + num x1 s)
  in
  let x0, x1 =
    List.fold_left four (low counter + k0, high counter + k1) schedule
  in
  (Ops.cast x1 Uint64 lsl 32) lor Ops.cast x0 Uint64

let int_binary d (k : P.binary) x y =
  let arith f = store d (f (lift d x) (lift d y)) in
  match k with
  | Add -> arith ( + )
  | Sub -> arith ( - )
  | Mul -> arith ( * )
  | Idiv -> int_div d ~rem:false x y
  | Mod -> int_div d ~rem:true x y
  | Pow -> int_pow d x y
  | Maximum -> where (x < y) y x
  | Minimum -> where (y < x) y x
  | And -> Ops.bitwise_and x y
  | Or -> x lor y
  | Xor -> x lxor y
  | Threefry -> threefry x y
  | Fdiv | Atan2 -> invalid_arg "Lower_arith: a float kind of integers"

(* Booleans

   A boolean is its byte, true where it is not zero: a bitcast keeps the byte,
   and the kinds read its truth and give 0 or 1. *)

let truth x = x <> num x 0

let one = function
  | [ x ] -> x
  | _ -> invalid_arg "Lower_arith: a value of parts"

let byte b = Ops.cast b Uint8

let bool_binary (k : P.binary) x y =
  byte
  @@
  let x = truth x and y = truth y in
  match k with
  | And | Minimum -> x && y
  | Or | Maximum -> x || y
  | Xor -> x lxor y
  | _ -> invalid_arg "Lower_arith: a numeric kind of booleans"

let compare_exact (k : P.compare) x y =
  match k with
  | Equal -> x = y
  | Not_equal -> x <> y
  | Less -> x < y
  | Less_equal -> not_ (y < x)

let bool_compare (k : P.compare) x y =
  let x = truth x and y = truth y in
  match k with
  | Equal -> x = y
  | Not_equal -> x <> y
  | Less -> not_ x && y
  | Less_equal -> not_ x || y

(* Complex

   Parts are float32 or float64 nodes. Add, Sub and Neg are the parts' kinds.
   Mul is the textbook product, each part one fused multiply-add over the other
   product; Fdiv is Smith's algorithm, the divisor's larger part dividing its
   smaller. Their NaN parts are the quiet NaN of a clear sign bit. *)

let fdiv x y = Ops.alu x Op.Fdiv [ y ]
let quiet x = where (is_nan x) (nan x) x

let c_mul (ar, ai) (br, bi) =
  let ii = ai * bi and ir = ai * br in
  (quiet (mulacc ar br (negated ii)), quiet (mulacc ar bi ir))

let c_div (ar, ai) (br, bi) =
  let pb = magnitude br and pi = magnitude bi in
  let big = pi < pb || pb = pi in
  let p = where big br bi and q = where big bi br in
  let x = where big ar ai and y = where big ai ar in
  let r = fdiv q p in
  let d = mulacc r q p in
  let im = where big (mulacc (negated r) x y) (mulacc r x (negated y)) in
  (quiet (fdiv (mulacc r y x) d), quiet (fdiv im d))

let c_nan (re, im) = is_nan re || is_nan im

(* Whether [x] precedes [y] in the order with -0 below +0, neither NaN. *)
let before x y = x < y || (x = y && negative x && not_ (negative y))
let same x y = x = y && negative x = negative y
let c_before (ar, ai) (br, bi) = before ar br || (same ar br && before ai bi)

(* The first NaN operand, else [pick a b], whether to give [b]. *)
let c_extreme pick a b =
  let take c (x, y) (u, v) = (where c x u, where c y v) in
  let gives_b = not_ (c_nan a) && (c_nan b || pick a b) in
  take gives_b b a

let c_compare (k : P.compare) ((ar, ai) as a) ((br, bi) as b) =
  let ordered = not_ (c_nan a || c_nan b) in
  match k with
  | Equal -> ar = br && ai = bi
  | Not_equal -> not_ (ar = br && ai = bi)
  | Less -> ordered && (ar < br || (ar = br && ai < bi))
  | Less_equal -> ordered && (ar < br || (ar = br && at_most ai bi))

let pair = function
  | [ re; im ] -> (re, im)
  | _ -> invalid_arg "Lower_arith: a complex value"

let unpair (re, im) = [ re; im ]

let c_unary (k : P.unary) xs =
  match k with
  | Neg -> List.map (Lower_real.unary Neg) xs
  | Recip ->
      let re, _ = pair xs in
      let one = Shape.const_like re (`Float 1.)
      and zero = Shape.const_like re (`Float 0.) in
      unpair (c_div (one, zero) (pair xs))
  | _ -> invalid_arg "Lower_arith: a real kind of complex numbers"

let c_binary (k : P.binary) xs ys =
  let a = pair xs and b = pair ys in
  match k with
  | Add | Sub -> List.map2 (Lower_real.binary k) xs ys
  | Mul -> unpair (c_mul a b)
  | Fdiv -> unpair (c_div a b)
  | Maximum -> unpair (c_extreme (fun a b -> c_before a b) a b)
  | Minimum -> unpair (c_extreme (fun a b -> c_before b a) a b)
  | _ -> invalid_arg "Lower_arith: a real kind of complex numbers"

(* Casts

   A source reads as a float32, a float64, or an integer: a narrow float decodes
   to float32, a complex number reads by its real part, a boolean as 0 or 1. *)

type source = Single of Ops.t | Double of Ops.t | Integer of D.any * Ops.t

let source (D.Any dt as d) xs =
  match (D.kind dt, xs) with
  | Float, [ x ] when D.equal dt D.Float64 -> Double x
  | Float, [ x ] when D.equal dt D.Float32 -> Single x
  | Float, [ x ] -> Single (decode d x)
  | Complex, re :: _ when D.equal dt D.Complex128 -> Double re
  | Complex, re :: _ -> Single re
  | (Signed | Unsigned), [ x ] -> Integer (d, x)
  | Boolean, [ x ] -> Integer (D.Any D.Uint8, byte (truth x))
  | _ -> invalid_arg "Lower_arith: parts of another dtype"

let node_of = function Single x | Double x | Integer (_, x) -> x

(* An integer's float64, rounded to odd past 2^53, exact below. *)
let int_double (D.Any dt) x =
  match dt with
  | D.Int64 -> odd64 ~signed:true x
  | D.Uint64 -> odd64 ~signed:false x
  | _ -> Ops.cast x Float64

(* The bounds of an integer dtype: its least value and the power of two past its
   greatest. *)
let bounds (D.Any dt) =
  let w = D.bits dt in
  if D.is D.Signed dt then
    (-.Float.ldexp 1. Stdlib.(w - 1), Float.ldexp 1. Stdlib.(w - 1))
  else (0., Float.ldexp 1. w)

(* A float truncated toward zero into the integer dtype [d], saturated to its
   range, NaN giving 0. Only values in range convert, since C leaves the others
   undefined. *)
let truncate d x =
  let t = dtype d in
  let least, past = bounds d in
  let lo = Shape.const_like x (`Float least)
  and hi = Shape.const_like x (`Float past) in
  let nan = is_nan x and low = x < lo and high = not_ (x < hi) in
  let safe = where (nan || low || high) (Shape.const_like x (`Float 0.)) x in
  let edge v = Shape.const_like ~dtype:t x (`Int v) in
  let greatest = Bigint.pred (Bigint.of_string (Printf.sprintf "%.0f" past)) in
  let in_range = Ops.cast safe t in
  where nan (edge Bigint.zero)
    (where low
       (edge (Bigint.of_string (Printf.sprintf "%.0f" least)))
       (where high (edge greatest) in_range))

let to_float32 = function
  | Single x -> x
  | Double x -> Ops.cast x Float32
  | Integer ((D.Any dt as d), x) when Stdlib.( = ) (D.bits dt) 64 ->
      Ops.cast (int_double d x) Float32
  | Integer (_, x) -> Ops.cast x Float32

let to_float64 = function
  | Single x | Integer (_, x) -> Ops.cast x Float64
  | Double x -> x

(* [s] stored into [d], neither complex nor boolean. *)
let store_real (D.Any dt as d) s =
  match dt with
  | D.Float64 -> to_float64 s
  | D.Float32 -> to_float32 s
  | _ when narrow d -> (
      match s with
      | Single x -> encode d x
      | Double x -> encode d (float_odd x)
      | Integer (si, x) -> encode d (float_odd (int_double si x)))
  | _ -> (
      match s with
      | Single x | Double x -> truncate d x
      | Integer (si, x) -> wrap si d x)

let is_zero x =
  if Dtype.is_float (Ops.dtype x) then x = Shape.const_like x (`Float 0.)
  else x = num x 0

let cast_parts ~src:(D.Any s as src) (D.Any dt as d) xs =
  match D.kind dt with
  | _ when D.equal s dt -> xs
  | Boolean when D.is D.Complex s ->
      let re, im = pair xs in
      [ byte (not_ (is_zero re && is_zero im)) ]
  | Boolean when D.is D.Boolean s -> [ byte (truth (one xs)) ]
  | Boolean -> [ byte (not_ (is_zero (node_of (source src xs)))) ]
  | Complex when D.is D.Complex s -> List.map (fun x -> Ops.cast x (dtype d)) xs
  | Complex ->
      let part =
        if D.equal dt D.Complex128 then D.Any D.Float64 else D.Any D.Float32
      in
      let re = store_real part (source src xs) in
      [ re; Shape.const_like re (`Float 0.) ]
  | _ -> [ store_real d (source src xs) ]

(* Bitcasts

   A value's bits as the unsigned integer of its width, a sub-byte code in the
   low bits of a byte, and back. A complex64 number is its real part's bits in
   the low word, as the host stores it. *)

let raw (D.Any dt) xs =
  match (dt, xs) with
  | D.Complex64, [ re; im ] ->
      (Ops.cast (bits im) Uint64 lsl 32) lor Ops.cast (bits re) Uint64
  | D.Int4, [ x ] -> bits x land 15
  | _, [ x ] -> bits x
  | _ -> invalid_arg "Lower_arith: parts of another dtype"

let of_raw (D.Any dt as d) r =
  match dt with
  | D.Complex64 -> [ f32 (Ops.cast r Uint32); f32 (Ops.cast (r lsr 32) Uint32) ]
  | D.Int4 -> [ store d r ]
  | _ -> [ of_bits (dtype d) r ]

let bitcast_parts ~src:(D.Any s as src) (D.Any dt as d) xs =
  if D.equal s dt then xs else of_raw d (raw src xs)

(* Kinds

   A kind takes and gives each value as its parts ({!parts}). A narrow float's
   kind computes on its decoded float32s and encodes its result once. *)

let widen d x = if narrow d then decode d x else x

let narrowed d f xs =
  let y = f (List.map (widen d) xs) in
  if narrow d then encode d y else y

let op1 (k : P.op1) ~src:(D.Any s as src) dt xs =
  match k with
  | Copy -> xs
  | Cast -> cast_parts ~src dt xs
  | Bitcast -> bitcast_parts ~src dt xs
  | Unary u -> (
      match D.kind s with
      | Float -> [ narrowed src (fun xs -> Lower_real.unary u (one xs)) xs ]
      | Complex -> c_unary u xs
      | Signed | Unsigned -> [ int_unary src u (one xs) ]
      | Boolean -> invalid_arg "Lower_arith: a numeric kind of booleans")

let op2 (k : P.op2) (D.Any s as d) xs ys =
  match (k, D.kind s) with
  | Binary b, Float ->
      [
        narrowed d
          (function [ x; y ] -> Lower_real.binary b x y | _ -> assert false)
          (xs @ ys);
      ]
  | Binary b, Complex -> c_binary b xs ys
  | Binary b, (Signed | Unsigned) -> [ int_binary d b (one xs) (one ys) ]
  | Binary b, Boolean -> [ bool_binary b (one xs) (one ys) ]
  | Compare c, Float ->
      [ byte (Lower_real.compare c (widen d (one xs)) (widen d (one ys))) ]
  | Compare c, Complex -> [ byte (c_compare c (pair xs) (pair ys)) ]
  | Compare c, (Signed | Unsigned) ->
      [ byte (compare_exact c (one xs) (one ys)) ]
  | Compare c, Boolean -> [ byte (bool_compare c (one xs) (one ys)) ]

let op3 (k : P.op3) (D.Any s as d) cs xs ys =
  match (k, cs) with
  | Where, [ c ] -> List.map2 (where (truth c)) xs ys
  | Fma, _ when D.is D.Float s ->
      [
        narrowed d
          (function [ a; b; c ] -> Lower_real.fma a b c | _ -> assert false)
          (cs @ xs @ ys);
      ]
  | Fma, _ ->
      let a = one cs and b = one xs and c = one ys in
      [ store d ((lift d a * lift d b) + lift d c) ]
  | Where, _ -> invalid_arg "Lower_arith: a condition of parts"

(* Leaves *)

let broadcast c s =
  let dims = List.map (fun n -> Ops.Int n) (Array.to_list s) in
  Shape.expand (Shape.reshape c (List.map (fun _ -> Ops.Int 1) dims)) dims

(* [word b] is the bytes [b], in the host's order, as an unsigned integer. *)
let word b =
  let n = ref Bigint.zero in
  for i = Stdlib.(String.length b - 1) downto 0 do
    n := Bigint.add (Bigint.shift_left !n 8) (Bigint.of_int (Char.code b.[i]))
  done;
  !n

(* [value tdt b] is the constant of [tdt] whose bits are [b]. A float is its
   bits' integer read as [tdt], which keeps a NaN's sign and payload. *)
let value tdt b : Dtype.const =
  let w = `Int (word b) in
  if Dtype.is_float tdt then (Dtype.bitcast (unsigned tdt) tdt w :> Dtype.const)
  else (Dtype.truncate tdt w :> Dtype.const)

(* [values d b] is the constant of each part of the element of [d] whose bits
   are [b]. *)
let values (D.Any dt as d) b =
  let parts =
    match dt with
    | D.Int4 ->
        (* The code's low four bits, sign-extended. *)
        let c = Stdlib.(Char.code b.[0] land 0xf) in
        [ String.make 1 (Char.chr Stdlib.(if c >= 8 then c + 0xf0 else c)) ]
    | _ when is_complex d ->
        let h = Stdlib.(String.length b / 2) in
        [ String.sub b 0 h; String.sub b h h ]
    | _ -> [ b ]
  in
  List.map (value (dtype d)) parts

let const d b s =
  List.map (fun c -> broadcast (Ops.const ~dtype:(dtype d) c) s) (values d b)

let fill dt x = values (D.Any dt) (P.bits dt x)

let coord s a =
  let r = Array.length s in
  let v = Shape.arange ~dtype:Int64 s.(a) in
  Shape.expand
    (Shape.reshape v
       (List.init r (fun j -> Ops.Int (if Stdlib.(j = a) then s.(a) else 1))))
    (List.map (fun n -> Ops.Int n) (Array.to_list s))

(* Programs *)

let prog (p : P.t) s ins =
  if Stdlib.(Array.length ins <> Iarray.length p.ins) then
    invalid_arg "Lower_arith.prog: one node per operand";
  let dtype_of i = P.dtype p i in
  let values = Array.make (Iarray.length p.nodes) [] in
  Iarray.iteri
    (fun i (n : P.node) ->
      values.(i) <-
        (match n with
        | In k -> parts (Iarray.get p.ins k) ins.(k)
        | Coord a -> [ coord s a ]
        | Const (dt, b) -> const dt b s
        | Op1 (k, dt, j) -> op1 k ~src:(dtype_of j) dt values.(j)
        | Op2 (k, j, l) -> op2 k (dtype_of j) values.(j) values.(l)
        | Op3 (k, j, l, m) ->
            op3 k (dtype_of l) values.(j) values.(l) values.(m)))
    p.nodes;
  Iarray.to_array p.outs
  |> Array.map (fun o -> of_parts (dtype_of o) values.(o))

let cast ~src dt x = of_parts dt (cast_parts ~src dt (parts src x))
